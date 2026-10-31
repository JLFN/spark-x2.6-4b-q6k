//! vq-tune — measure the VQ decode and prefill kernels on one artifact and
//! write the per-geometry profile the runtime reads.
//!
//! The tool is model-agnostic: it reads whatever manifest.txt the artifact
//! directory carries, groups the matrices into their distinct geometries, asks
//! the kernel library which variants and tile shapes each geometry can run,
//! times them on the real payloads, rejects any candidate whose output moves
//! past the kernel-precision gate, and writes the winners to
//! <artifact>/vq-tune.profile.
//!
//! Usage:
//!   vq-tune DIR [--profile PATH] [--tokens N] [--iters N] [--force] [--dry-run]

mod artifact;
mod kernels;
mod profile;

use std::process::ExitCode;

use artifact::Geom;
use profile::Entry;

struct Opt {
    dir: String,
    profile: Option<String>,
    tokens: i32,
    iters: i32,
    force: bool,
    dry: bool,
}

fn usage() -> String {
    "usage: vq-tune DIR [--profile PATH] [--tokens N] [--iters N] [--force] [--dry-run]\n\
     \x20  DIR            artifact directory holding manifest.txt and the containers\n\
     \x20  --profile PATH profile to write (default DIR/vq-tune.profile)\n\
     \x20  --tokens N     prompt tokens per prefill timing (default 512)\n\
     \x20  --iters N      timed launches per candidate (default 7)\n\
     \x20  --force        overwrite a profile that belongs to another artifact\n\
     \x20  --dry-run      sweep and report, write nothing"
        .into()
}

fn parse_args() -> Result<Opt, String> {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 2 {
        return Err(usage());
    }
    let mut o = Opt {
        dir: a[1].clone(),
        profile: None,
        tokens: 512,
        iters: 7,
        force: false,
        dry: false,
    };
    let mut i = 2;
    while i < a.len() {
        match a[i].as_str() {
            "--profile" => {
                i += 1;
                o.profile = Some(a.get(i).ok_or("--profile needs a path")?.clone());
            }
            "--tokens" => {
                i += 1;
                o.tokens = a.get(i).ok_or("--tokens needs a number")?.parse().map_err(|e| format!("--tokens: {e}"))?;
            }
            "--iters" => {
                i += 1;
                o.iters = a.get(i).ok_or("--iters needs a number")?.parse().map_err(|e| format!("--iters: {e}"))?;
            }
            "--force" => o.force = true,
            "--dry-run" => o.dry = true,
            "-h" | "--help" => return Err(usage()),
            other => return Err(format!("unknown argument {other}\n{}", usage())),
        }
        i += 1;
    }
    if o.tokens < 1 || o.tokens > 4096 {
        return Err("--tokens must be between 1 and 4096".into());
    }
    if o.iters < 1 || o.iters > 1000 {
        return Err("--iters must be between 1 and 1000".into());
    }
    Ok(o)
}

/// Deterministic activations: the sweep must be repeatable, and a value that
/// decomposes cleanly would hide a swapped index, so this is a sine.
fn activations(n: usize) -> Vec<f32> {
    (0..n).map(|i| ((i as f64) * 0.001).sin() as f32).collect()
}

struct Pick {
    variant: i32,
    threads: i32,
    bn: i32,
    bk: i32,
    rate: f64,
    runner: Option<(i32, i32, i32, i32, f64)>,
}

struct Sweep {
    gemv: Pick,
    prefill: Pick,
}

const GEMV_GATE: f64 = 1e-4;
const PREFILL_GATE: f64 = 1e-4;

fn tune_geometry(dir: &str, g: Geom, rep: &artifact::Matrix, o: &Opt) -> Result<Sweep, String> {
    let stages = artifact::read_stages(dir, rep)?;
    let p = kernels::upload(&stages)?;
    let d32 = g.dim as i32;
    let nc = g.nc as i32;
    let rows = g.rows as i32;
    let cols = g.cols as i32;
    let v3 = g.v3 as i32;

    /* ---- decode: one token, every variant at every launch width ---- */
    let xh = activations(g.cols as usize);
    let x = kernels::Dev::alloc(xh.len() * 4)?;
    x.upload(bytemuck_f32(&xh), 0)?;
    let y = kernels::Dev::alloc(g.rows as usize * 4)?;

    let ref_variant = if kernels::gemv_ready(g, 3) { 3 } else if kernels::gemv_ready(g, 2) { 2 } else { 1 };
    y.zero()?;
    let (_, ref_out) = kernels::gemv(&p, g, &x, &y, ref_variant, 256, 1)?;

    let mut best: Option<(i32, i32, f64)> = None;
    let mut runner: Option<(i32, i32, f64)> = None;
    for variant in [3, 2, 1, 0] {
        if !kernels::gemv_ready(g, variant) {
            continue;
        }
        let widths: &[i32] = if variant == 0 { &[256] } else { &[128, 256, 512] };
        for &threads in widths {
            y.zero()?;
            let (_, out) = kernels::gemv(&p, g, &x, &y, variant, threads, 1)?;
            let rel = kernels::rel_l2(&out, &ref_out);
            if rel > GEMV_GATE {
                println!("  gemv    reject variant {variant} threads {threads}: relL2 {rel:.3e}");
                continue;
            }
            y.zero()?;
            let (ms, _) = kernels::gemv(&p, g, &x, &y, variant, threads, o.iters)?;
            let gbs = p.bytes as f64 / (ms * 1e-3) / 1e9;
            let cand = (variant, threads, gbs);
            if best.map(|b| gbs > b.2).unwrap_or(true) {
                runner = best;
                best = Some(cand);
            } else if runner.map(|r| gbs > r.2).unwrap_or(true) {
                runner = Some(cand);
            }
        }
    }
    let best = best.ok_or_else(|| format!("no runnable gemv variant for {}", g.key()))?;
    let gemv = Pick {
        variant: best.0,
        threads: best.1,
        bn: 0,
        bk: 0,
        rate: best.2,
        runner: runner.map(|r| (r.0, r.1, 0, 0, r.2)),
    };

    /* ---- prefill: the runtime's prompt chunk, every legal tile ---- */
    let ntok = o.tokens;
    let xt_h = activations(g.cols as usize * ntok as usize);
    let xt = kernels::Dev::alloc(xt_h.len() * 4)?;
    xt.upload(bytemuck_f32(&xt_h), 0)?;
    let yb = kernels::Dev::alloc(ntok as usize * g.rows as usize * 4)?;

    // The reference is the default tiled tile, because that is what the
    // runtime runs without a profile; the legacy kernel is a different
    // (f16-free) path and is compared only when no tiled tile is legal.
    let (ref_v, ref_bn, ref_bk) = if kernels::prefill_ready(g, 128, 64, 1) {
        (1, 128, 64)
    } else {
        (3, 0, 0)
    };
    yb.zero()?;
    let (_, ref_out) = kernels::prefill(&p, g, &xt, &yb, ntok, ref_v, ref_bn, ref_bk, 1)?;

    let mut best: Option<(i32, i32, i32, f64)> = None;
    let mut runner: Option<(i32, i32, i32, f64)> = None;
    /* Legacy is a different arithmetic path (f32 codebook products, no f16
     * activation tile), so it cannot be compared to the tiled reference at the
     * kernel-precision gate, and the runtime's chunked prompt path requires
     * every matrix to be tiled. It is therefore offered only when no tiled
     * tile fits the geometry; otherwise the sweep is tiled-only, which is what
     * keeps the whole model on the fast prompt path. */
    let mut cands: Vec<(i32, i32, i32)> = Vec::new();
    if ref_v == 3 {
        cands.push((3, 0, 0));
    } else {
        for (bn, bk) in kernels::prefill_shapes() {
            for variant in [1, 2] {
                if kernels::prefill_ready(g, bn, bk, variant) {
                    cands.push((variant, bn, bk));
                }
            }
        }
    }
    for (variant, bn, bk) in cands {
        yb.zero()?;
        let (_, out) = kernels::prefill(&p, g, &xt, &yb, ntok, variant, bn, bk, 1)?;
        let rel = kernels::rel_l2(&out, &ref_out);
        if rel > PREFILL_GATE {
            println!("  prefill reject variant {variant} tile {bn}x{bk}: relL2 {rel:.3e}");
            continue;
        }
        yb.zero()?;
        let (ms, _) = kernels::prefill(&p, g, &xt, &yb, ntok, variant, bn, bk, o.iters)?;
        let tps = ntok as f64 / (ms * 1e-3);
        let cand = (variant, bn, bk, tps);
        if best.map(|b| tps > b.3).unwrap_or(true) {
            runner = best;
            best = Some(cand);
        } else if runner.map(|r| tps > r.3).unwrap_or(true) {
            runner = Some(cand);
        }
    }
    let best = best.ok_or_else(|| format!("no runnable prefill path for {}", g.key()))?;
    let prefill = Pick {
        variant: best.0,
        threads: 0,
        bn: best.1,
        bk: best.2,
        rate: best.3,
        runner: runner.map(|r| (r.0, 0, r.1, r.2, r.3)),
    };

    let _ = (d32, nc, rows, cols, v3);
    Ok(Sweep { gemv, prefill })
}

/// The FFI takes a byte slice; this is the only unsafe-free way to view an
/// f32 slice as bytes without another dependency.
fn bytemuck_f32(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}

fn main() -> ExitCode {
    let o = match parse_args() {
        Ok(o) => o,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::from(2);
        }
    };
    if let Err(e) = run(&o) {
        eprintln!("vq-tune: {e}");
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}

fn run(o: &Opt) -> Result<(), String> {
    if !artifact::exists(&o.dir) {
        return Err(format!("{}: no manifest.txt", o.dir));
    }
    let man = artifact::load(&o.dir)?;
    if man.dim != 4 && man.dim != 8 {
        return Err(format!("{}: dim {} has no kernel", o.dir, man.dim));
    }

    let mut groups: Vec<(Geom, Vec<usize>)> = Vec::new();
    for (i, m) in man.matrices.iter().enumerate() {
        if m.stages == 0 {
            continue;
        }
        let (dim, nc, v3) = artifact::probe(&o.dir, m)?;
        if dim != man.dim || nc != man.nc {
            return Err(format!(
                "{}: header says dim {dim} nc {nc}, the manifest encoding says dim {} nc {}",
                m.name, man.dim, man.nc
            ));
        }
        let g = Geom { dim, nc, rows: m.rows, cols: m.cols, v3 };
        match groups.iter_mut().find(|(gg, _)| *gg == g) {
            Some((_, v)) => v.push(i),
            None => groups.push((g, vec![i])),
        }
    }
    if groups.is_empty() {
        return Err(format!("{}: no matrix containers in the manifest", o.dir));
    }

    let fp = artifact::fingerprint(&o.dir, &man.matrices)?;
    let gpu = kernels::device_name()?;
    let driver = kernels::driver_version();
    println!("vq-tune     {}", o.dir);
    println!("artifact    {fp:016x} on {gpu} (driver {driver})");
    println!("sweep       {} geometries, {} tokens per prefill timing, {} timed launches",
             groups.len(), o.tokens, o.iters);

    let mut entries = Vec::new();
    let mut report = String::new();
    for (g, idx) in &groups {
        let rep = &man.matrices[idx[0]];
        println!();
        println!("{} ({} matrices, {} stages)", g.key(), idx.len(), rep.stages);
        let s = tune_geometry(&o.dir, *g, rep, o)?;
        let gemv_line = format!(
            "  gemv    variant {} threads {}   {:.1} GB/s{}",
            s.gemv.variant,
            s.gemv.threads,
            s.gemv.rate,
            match s.gemv.runner {
                Some((v, t, _, _, r)) => format!("   (runner-up variant {v} threads {t}: {r:.1})"),
                None => String::new(),
            }
        );
        let prefill_line = if s.prefill.variant == 3 {
            format!("  prefill legacy (no tiled tile fits)   {:.1} tok/s", s.prefill.rate)
        } else {
            format!(
                "  prefill variant {} tile {}x{}   {:.1} tok/s{}",
                s.prefill.variant,
                s.prefill.bn,
                s.prefill.bk,
                s.prefill.rate,
                match s.prefill.runner {
                    Some((v, _, bn, bk, r)) => format!("   (runner-up variant {v} {bn}x{bk}: {r:.1})"),
                    None => String::new(),
                }
            )
        };
        println!("{gemv_line}");
        println!("{prefill_line}");
        report.push_str(&format!("{}\n{gemv_line}\n{prefill_line}\n\n", g.key()));

        entries.push(Entry {
            kind: 0,
            dim: g.dim,
            nc: g.nc,
            rows: g.rows,
            cols: g.cols,
            v3: g.v3,
            variant: s.gemv.variant,
            threads: s.gemv.threads,
            bn: 0,
            bk: 0,
            rate: s.gemv.rate,
        });
        entries.push(Entry {
            kind: 1,
            dim: g.dim,
            nc: g.nc,
            rows: g.rows,
            cols: g.cols,
            v3: g.v3,
            variant: s.prefill.variant,
            threads: 0,
            bn: s.prefill.bn,
            bk: s.prefill.bk,
            rate: s.prefill.rate,
        });
    }

    if o.dry {
        println!("\ndry run: no profile written");
        return Ok(());
    }

    let path = o.profile.clone().unwrap_or_else(|| format!("{}/vq-tune.profile", o.dir));
    if let Some(old) = profile::existing_artifact(&path) {
        if old != fp && !o.force {
            return Err(format!(
                "{path} was tuned on artifact {old:016x}, this one is {fp:016x}; pass --force to replace it"
            ));
        }
    }
    let text = profile::render(fp, &gpu, &driver, &entries);
    profile::write_atomic(&path, &text).map_err(|e| format!("{path}: {e}"))?;
    println!("\nprofile     {path} ({} entries)", entries.len());
    Ok(())
}
