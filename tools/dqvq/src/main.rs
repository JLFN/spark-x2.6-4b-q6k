//! dqvq — reference decoder and decode-speed bench for RQVQ containers.
//!
//! The container layout and the decode live in the `vq-format` crate, which the
//! runtime links too, so there is exactly one decoder in the repository. This
//! binary is the command-line face of it:
//!
//!   dqvq info   <file.bin>
//!   dqvq check  <file.bin> <reference.f32>
//!   dqvq gemv   <file.bin> <vec.f32> [--iters N] [--fused]
//!
//! `check` compares the decode with a raw little-endian f32 dump of the same
//! matrix (produced by `decode dump` in C) and fails on any bit difference.
//! `gemv` times the two shapes a runtime can choose between: decode the whole
//! matrix once and multiply (the naive loader), or fuse the index gather and the
//! dot product row by row (what a streaming decoder does). It reports both in
//! milliseconds and as GB/s of container bytes, plus a checksum so the two paths
//! can be compared with each other.

use std::env;
use std::fs;
use std::process::ExitCode;
use std::time::Instant;

use vq_format::Container;

fn read_f32(path: &str) -> Result<Vec<f32>, String> {
    let b = fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    if b.len() % 4 != 0 {
        return Err(format!("{path}: {} bytes is not a whole f32 dump", b.len()));
    }
    let mut v = Vec::with_capacity(b.len() / 4);
    for w in b.chunks_exact(4) {
        v.push(f32::from_le_bytes([w[0], w[1], w[2], w[3]]));
    }
    Ok(v)
}

fn cmd_info(path: &str) -> Result<(), String> {
    let buf = fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    let c = Container::open(&buf)?;
    let bytes = buf.len();
    let weights = c.rows * c.cols;
    let per = (bytes - 32) / c.stages.len();
    let hdr = if c.stages[0].v3 { 24 } else { 16 };
    let indices = per - hdr - c.nc * c.dim * 2 - c.rows * 2;
    println!("file        {path}");
    println!("bytes       {bytes} ({:.4} GB)", bytes as f64 / 1e9);
    println!("shape       {} x {}, {} weights", c.rows, c.cols, weights);
    println!(
        "per stage   dim {}, nc {} ({}-bit index), {} bytes",
        c.dim, c.nc, c.stages[0].nbit, per
    );
    println!(
        "stages      {}, payload {}, total {}",
        c.stages.len(),
        bytes - 32,
        bytes
    );
    println!("bits/weight {:.4}", bytes as f64 * 8.0 / weights as f64);
    println!(
        "layout      codebook {}, gains {}, indices {}",
        c.nc * c.dim * 2,
        c.rows * 2,
        indices
    );
    println!("consistent  yes");
    Ok(())
}

fn cmd_check(path: &str, reference: &str) -> Result<(), String> {
    let buf = fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    let c = Container::open(&buf)?;
    let want = read_f32(reference)?;
    if want.len() != c.rows * c.cols {
        return Err(format!(
            "{reference} holds {} floats, matrix is {}",
            want.len(),
            c.rows * c.cols
        ));
    }
    let got = c.decode();
    let mut exact = 0usize;
    let mut mx = 0f32;
    let mut se = 0f64;
    let mut sn = 0f64;
    for (a, b) in got.iter().zip(want.iter()) {
        if a.to_bits() == b.to_bits() {
            exact += 1;
        }
        let d = f64::from(*a) - f64::from(*b);
        se += d * d;
        sn += f64::from(*b) * f64::from(*b);
        mx = mx.max(d.abs() as f32);
    }
    println!(
        "{} x {}  stages {}  exact {}/{}  relL2 {:.6}%  maxabs {:.6}",
        c.rows,
        c.cols,
        c.stages.len(),
        exact,
        got.len(),
        100.0 * (se / (sn + 1e-30)).sqrt(),
        mx
    );
    if exact == got.len() {
        Ok(())
    } else {
        Err("decode differs from the reference dump".into())
    }
}

fn cmd_gemv(path: &str, vecpath: &str, iters: usize, fused: bool) -> Result<(), String> {
    let buf = fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    let t0 = Instant::now();
    let c = Container::open(&buf)?;
    let prepare = t0.elapsed().as_secs_f64();

    let x = read_f32(vecpath)?;
    if x.len() != c.cols {
        return Err(format!(
            "{vecpath} holds {} floats, matrix has {} columns",
            x.len(),
            c.cols
        ));
    }

    let mut y = vec![0f32; c.rows];
    let mut decode_s = 0.0;
    let mut best = f64::INFINITY;

    if fused {
        for _ in 0..iters {
            for v in y.iter_mut() {
                *v = 0.0;
            }
            let t = Instant::now();
            for s in &c.stages {
                for r in 0..c.rows {
                    s.dot_row(r, &x, &mut y);
                }
            }
            best = best.min(t.elapsed().as_secs_f64());
        }
    } else {
        let t = Instant::now();
        let w = c.decode();
        decode_s = t.elapsed().as_secs_f64();
        for _ in 0..iters {
            let t = Instant::now();
            for r in 0..c.rows {
                let row = &w[r * c.cols..(r + 1) * c.cols];
                let mut s = 0.0f32;
                for (a, b) in row.iter().zip(x.iter()) {
                    s += a * b;
                }
                y[r] = s;
            }
            best = best.min(t.elapsed().as_secs_f64());
        }
    }

    let sum: f64 = y.iter().map(|v| f64::from(*v)).sum();
    let nrm: f64 = y.iter().map(|v| f64::from(*v) * f64::from(*v)).sum();
    let container_gb = buf.len() as f64 / 1e9;
    let weights_gb = (c.rows * c.cols * 4) as f64 / 1e9;

    println!(
        "file        {path} ({} x {}, {} stages)",
        c.rows,
        c.cols,
        c.stages.len()
    );
    if fused {
        println!("mode        fused gather+dot, {} passes", iters);
        println!(
            "gemv        {:.3} ms best, {:.2} GB/s of container",
            best * 1e3,
            container_gb / best
        );
    } else {
        println!(
            "prepare     {:.3} ms codebooks to f32",
            prepare * 1e3
        );
        println!(
            "decode      {:.3} ms, {:.2} GB/s of container",
            decode_s * 1e3,
            container_gb / decode_s
        );
        println!("mode        decoded f32 matvec, {} passes", iters);
        println!(
            "matvec      {:.3} ms best, {:.2} GB/s of f32 weights",
            best * 1e3,
            weights_gb / best
        );
    }
    println!("checksum    sum {sum:.6}, norm {nrm:.6}");
    Ok(())
}

fn main() -> ExitCode {
    let a: Vec<String> = env::args().collect();
    if a.len() < 3 {
        eprintln!(
            "usage: {} info <file.bin>\n       {} check <file.bin> <reference.f32>\n       {} gemv <file.bin> <vec.f32> [--iters N] [--fused]",
            a[0], a[0], a[0]
        );
        return ExitCode::from(2);
    }
    let r = match a[1].as_str() {
        "info" => cmd_info(&a[2]),
        "check" if a.len() >= 4 => cmd_check(&a[2], &a[3]),
        "gemv" if a.len() >= 4 => {
            let mut iters = 20usize;
            let mut fused = false;
            let mut i = 4;
            while i < a.len() {
                match a[i].as_str() {
                    "--iters" if i + 1 < a.len() => {
                        iters = a[i + 1].parse().unwrap_or(20);
                        i += 2;
                    }
                    "--fused" => {
                        fused = true;
                        i += 1;
                    }
                    other => {
                        eprintln!("unknown argument {other}");
                        return ExitCode::from(2);
                    }
                }
            }
            cmd_gemv(&a[2], &a[3], iters.max(1), fused)
        }
        _ => Err("unknown command or missing arguments".into()),
    };
    match r {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("dqvq: {e}");
            ExitCode::FAILURE
        }
    }
}
