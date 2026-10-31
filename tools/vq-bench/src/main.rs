//! vq-bench — Rust benchmark harness for RQVQ VQ artifacts.
//!
//! Emits the ds4-perf column set so runs are comparable with that tooling:
//!   ctx_tokens,prefill_tokens,prefill_tps,gen_tokens,gen_tps,first_token_sec,
//!   kvcache_bytes
//!
//! prefill is a real N-token GEMM over the fused decoder, decode is batch 1,
//! and kvcache_bytes is the checkpoint geometry (full-attention layers hold the
//! context, sliding layers hold their window). Everything measured is the VQ
//! weight path: no attention, no KV read, no sampling.
//!
//! Usage:
//!   vq-bench <artifact-dir> [--iters N] [--prefill N] [--csv]
//!              [--ctx N] [--kv f16|q8_0] [--kvcache-bytes N]
//!              [--layers N --full-layers N --kv-heads N --head-dim N]

mod bench;
mod compare;
mod csv;
mod cuda;

use bench::{read_manifest, Harness, Report, CSV_HEADER};
use std::process::ExitCode;
use std::time::Instant;

struct Opt {
    dir: String,
    iters: u32,
    prefill: u32,
    csv: bool,
    ctx: u64,
    kv_q8: bool,
    kv_override: Option<u64>,
    layers: u64,
    full_layers: u64,
    kv_heads: u64,
    head_dim: u64,
}

fn usage() -> String {
    "usage: vq-bench <artifact-dir> [--iters N] [--prefill N] [--csv]\n\
     \x20             [--ctx N] [--kv f16|q8_0] [--kvcache-bytes N]\n\
     \x20             [--layers N --full-layers N --kv-heads N --head-dim N]\n\
     \x20      vq-bench compare <baseline.csv> <candidate.csv>"
        .into()
}

/// Baseline vs candidate on the ds4-bench columns. Positive means the candidate
/// is slower per token; the band is the no-overlap envelope, not a confidence
/// interval.
fn compare_files(a: &str, b: &str) -> Result<(), String> {
    let read = |p: &str| -> Result<Vec<csv::Row>, String> {
        let text = std::fs::read_to_string(p).map_err(|e| format!("{p}: {e}"))?;
        let rows = csv::parse(&text)?;
        if rows.is_empty() {
            return Err(format!("{p}: no ds4-bench rows"));
        }
        Ok(rows)
    };
    let rows = compare::compare(&read(a)?, &read(b)?)?;
    println!("metric        ctx   baseline s    candidate s   median     band");
    for d in &rows {
        println!(
            "{:<12} {:>6}  {:>10.6}  {:>12.6}  {:+7.2}%  [{:+.2}%, {:+.2}%]",
            d.metric,
            d.ctx,
            d.baseline_median,
            d.candidate_median,
            d.median_percent,
            d.lower_percent,
            d.upper_percent
        );
    }
    Ok(())
}

fn parse_args() -> Result<Opt, String> {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 2 {
        return Err(usage());
    }
    let mut o = Opt {
        dir: a[1].clone(),
        iters: 5,
        prefill: 0,
        csv: false,
        ctx: 131072,
        kv_q8: true,
        kv_override: None,
        /* Spark geometry, from models/hf/config.json. */
        layers: 36,
        full_layers: 9,
        kv_heads: 4,
        head_dim: 256,
    };
    let mut i = 2;
    while i < a.len() {
        let need = |i: usize| -> Result<&str, String> {
            a.get(i + 1)
                .map(|s| s.as_str())
                .ok_or_else(|| format!("{} needs a value", a[i]))
        };
        match a[i].as_str() {
            "--iters" => { o.iters = need(i)?.parse().map_err(|_| "bad --iters")?; i += 2; }
            "--prefill" => { o.prefill = need(i)?.parse().map_err(|_| "bad --prefill")?; i += 2; }
            "--csv" => { o.csv = true; i += 1; }
            "--ctx" => { o.ctx = need(i)?.parse().map_err(|_| "bad --ctx")?; i += 2; }
            "--kv" => { o.kv_q8 = need(i)? == "q8_0"; i += 2; }
            "--kvcache-bytes" => { o.kv_override = Some(need(i)?.parse().map_err(|_| "bad --kvcache-bytes")?); i += 2; }
            "--layers" => { o.layers = need(i)?.parse().map_err(|_| "bad --layers")?; i += 2; }
            "--full-layers" => { o.full_layers = need(i)?.parse().map_err(|_| "bad --full-layers")?; i += 2; }
            "--kv-heads" => { o.kv_heads = need(i)?.parse().map_err(|_| "bad --kv-heads")?; i += 2; }
            "--head-dim" => { o.head_dim = need(i)?.parse().map_err(|_| "bad --head-dim")?; i += 2; }
            other => return Err(format!("unknown argument {other}\n{}", usage())),
        }
    }
    if o.iters == 0 {
        o.iters = 1;
    }
    Ok(o)
}

/// One sequence: full-attention layers hold `ctx` tokens, sliding layers hold
/// the 512-token window (Spark: 9 full, 27 sliding). q8_0 stores 34 bytes per
/// 32 values.
fn kvcache_bytes(o: &Opt) -> u64 {
    if let Some(v) = o.kv_override {
        return v;
    }
    let elem = if o.kv_q8 { 34.0 / 32.0 } else { 2.0 };
    let full = o.ctx as f64 * o.full_layers as f64 * 2.0 * o.kv_heads as f64 * o.head_dim as f64 * elem;
    let sliding = 512.0 * (o.layers - o.full_layers) as f64 * 2.0 * o.kv_heads as f64 * o.head_dim as f64 * elem;
    (full + sliding) as u64
}

fn run(o: &Opt) -> Result<(), String> {
    let dir = std::path::Path::new(&o.dir);
    let m = read_manifest(dir)?;
    let h = Harness::new(&m, o.iters, o.prefill)?;

    let kv = kvcache_bytes(o);
    let mut r = Report {
        tensors: 0,
        container_bytes: 0,
        weights: 0,
        decode_ms: 0.0,
        prefill_ms: 0.0,
        dequant_ms: 0.0,
        peak_dev_bytes: h.peak_dev_bytes(),
        prefill_tokens: o.prefill,
        kvcache_bytes: kv,
        slowest: Vec::new(),
    };

    let started = Instant::now();
    let mut v3blk = 0usize;
    let mut fast = 0usize;
    let mut generic = 0usize;
    for t in &m.tensors {
        let s = h.measure(t)?;
        match s.variant {
            3 => v3blk += 1,
            2 => fast += 1,
            _ => generic += 1,
        }
        r.tensors += 1;
        r.container_bytes += t.file_bytes;
        r.weights += t.rows as u64 * t.cols as u64;
        r.decode_ms += s.decode_ms;
        r.prefill_ms += s.prefill_ms;
        r.dequant_ms += s.dequant_ms;

        r.slowest.push((t.name.clone(), s.decode_ms));
        r.slowest.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        r.slowest.truncate(5);
    }
    let wall = started.elapsed().as_secs_f64();

    if o.csv {
        if o.prefill == 0 {
            return Err("--csv needs --prefill N".into());
        }
        print!("{CSV_HEADER}\n{}", r.csv_row());
        return Ok(());
    }

    let gb = r.container_bytes as f64 / 1e9;
    let bpw = r.container_bytes as f64 * 8.0 / r.weights as f64;
    println!("artifact    {}", m.dir.display());
    println!("tensors     {} matrices, {} weights", r.tensors, r.weights);
    println!("container   {gb:.4} GB ({bpw:.4} bits/weight)");
    let mut paths = Vec::new();
    if v3blk > 0 {
        paths.push(format!("{v3blk} DQV3 block"));
    }
    if fast > 0 {
        paths.push(format!("{fast} dim-4 fast"));
    }
    if generic > 0 {
        paths.push(format!("{generic} generic warp (untuned)"));
    }
    println!("kernels     {}", paths.join(", "));
    println!(
        "decode      {:.3} ms/token, {:.2} GB/s of container",
        r.decode_ms,
        gb / (r.decode_ms * 1e-3)
    );
    if o.prefill > 0 {
        println!(
            "prefill     {} tokens in {:.3} ms, {:.1} tok/s, {:.2} GB/s of container",
            o.prefill,
            r.prefill_ms,
            r.prefill_tps(),
            gb / (r.prefill_ms * 1e-3)
        );
        println!("first token {:.1} ms (weight path only)", r.prefill_ms);
    }
    println!(
        "dequant     {:.3} ms, {:.2} GB/s of container",
        r.dequant_ms,
        gb / (r.dequant_ms * 1e-3)
    );
    println!(
        "decode rate {:.1} tok/s (matrices only, batch 1)",
        r.decode_tps()
    );
    println!(
        "kv geometry {:.3} GB at ctx {} ({})",
        kv as f64 / 1e9,
        o.ctx,
        if o.kv_q8 { "q8_0" } else { "f16" }
    );
    println!("peak dev    {:.1} MiB (shared buffers)", r.peak_dev_bytes as f64 / (1024.0 * 1024.0));
    println!("wall        {wall:.1} s for {:.0} timed passes", r.tensors as f64 * o.iters as f64);
    print!("slowest     ");
    for (i, (name, ms)) in r.slowest.iter().enumerate() {
        print!("{name} {ms:.3} ms{}", if i + 1 < r.slowest.len() { ", " } else { "\n" });
    }
    println!("csv         {}", r.csv_row().trim_end());
    Ok(())
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() >= 4 && args[1] == "compare" {
        return match compare_files(&args[2], &args[3]) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("vq-bench: {e}");
                ExitCode::FAILURE
            }
        };
    }
    match parse_args().and_then(|o| run(&o)) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("vq-bench: {e}");
            ExitCode::FAILURE
        }
    }
}
