//! dspark — inspect, convert and validate the DSpark draft checkpoint, and run
//! the CPU draft forward and speculative loop against the target.
//!
//! Usage:
//!   dspark dump <file.gguf>
//!   dspark convert <file.gguf> <outdir> [--force]
//!   dspark check <outdir>
//!   dspark forward --draft DIR --hidden FILE --target DIR --anchor ID ...
//!   dspark spec --draft DIR --target DIR --ids LIST --gen N ...
//!
//! `dump` prints what the GGUF says. `convert` repacks it into a role manifest
//! and a weights.bin that a generic executor can read; `check` validates that
//! directory on its own, with no access to the source file. `forward` runs one
//! block decode from a captured feature file; `spec` runs the full
//! propose/verify loop against the vq-runtime target.

mod check;
mod convert;
mod draft;
mod gguf;
mod manifest;
mod roles;
mod spec;
mod target;

use std::path::Path;
use std::process::ExitCode;

fn usage() -> &'static str {
    "usage: dspark dump <file.gguf>\n       \
     dspark convert <file.gguf> <outdir> [--force] [--verify]\n       \
     dspark check <outdir>\n       \
     dspark forward --draft DIR --hidden FILE --target DIR --anchor ID \
[--pos0 N] [--n-draft K]\n       \
     dspark spec --draft DIR --target DIR --ids LIST --gen N \
[--weights hf|vq] [--config PATH] [--n-max K] [--ctx N] [--threads N] \
[--rounds] [--trace-logits] [--out FILE]"
}

fn forward(args: &[String]) -> Result<(), String> {
    let mut draft_dir = String::new();
    let mut hidden_path = String::new();
    let mut target_dir = String::new();
    let mut anchor: Option<u32> = None;
    let mut pos0: Option<usize> = None;
    let mut n_draft: Option<usize> = None;
    let mut i = 0;
    while i < args.len() {
        let take = || -> Result<String, String> {
            args.get(i + 1)
                .cloned()
                .ok_or_else(|| format!("{} needs a value", args[i]))
        };
        match args[i].as_str() {
            "--draft" => {
                draft_dir = take()?;
                i += 2;
            }
            "--hidden" => {
                hidden_path = take()?;
                i += 2;
            }
            "--target" => {
                target_dir = take()?;
                i += 2;
            }
            "--anchor" => {
                anchor = Some(take()?.parse().map_err(|_| "bad --anchor".to_string())?);
                i += 2;
            }
            "--pos0" => {
                pos0 = Some(take()?.parse().map_err(|_| "bad --pos0".to_string())?);
                i += 2;
            }
            "--n-draft" => {
                n_draft = Some(take()?.parse().map_err(|_| "bad --n-draft".to_string())?);
                i += 2;
            }
            other => return Err(format!("unknown forward argument {other}")),
        }
    }
    if draft_dir.is_empty() || hidden_path.is_empty() || target_dir.is_empty() {
        return Err("forward needs --draft, --hidden and --target".into());
    }
    let anchor = anchor.ok_or("forward needs --anchor")?;

    let d = draft::Draft::open(Path::new(&draft_dir))?;
    let h = draft::Hidden::open(Path::new(&hidden_path))?;
    let e = target::open(Path::new(&target_dir), "model.embedding.weight")?;
    let pos0 = pos0.unwrap_or(h.tokens);
    let n_draft = n_draft.unwrap_or(d.m.header.block_size as usize);

    let t0 = std::time::Instant::now();
    let inj = draft::encode_features(&d, &h)?;
    let t_enc = t0.elapsed();
    let r = draft::decode_block(
        &d,
        &inj,
        &e,
        anchor,
        d.m.header.mask_token as u32,
        pos0,
        n_draft,
    )?;
    let t_dec = t0.elapsed() - t_enc;

    println!(
        "draft       {} layers, block {} rows, n_draft {}, pos0 {}, anchor {}, mask {}",
        d.m.header.layers,
        n_block_rows(&d, n_draft),
        n_draft,
        pos0,
        anchor,
        d.m.header.mask_token
    );
    println!(
        "encode      {} positions in {:.3} s, decode {:.3} s",
        h.tokens,
        t_enc.as_secs_f64(),
        t_dec.as_secs_f64()
    );
    for (i, l) in r.logits.iter().enumerate() {
        let mut order: Vec<usize> = (0..l.len()).collect();
        order.sort_unstable_by(|a, b| l[*b].partial_cmp(&l[*a]).unwrap());
        let top: Vec<String> = order
            .iter()
            .take(5)
            .map(|&k| format!("{k} {:.4}", l[k]))
            .collect();
        println!(
            "row {:<2} pos {:<3} argmax {:<6} conf {:<8} top [{}]",
            i,
            pos0 + i,
            r.argmax[i],
            if r.conf[i].is_nan() {
                "nan".to_string()
            } else {
                format!("{:.4}", r.conf[i])
            },
            top.join(", ")
        );
    }
    println!("proposals   {:?}", r.argmax);
    Ok(())
}

fn n_block_rows(d: &draft::Draft, n_draft: usize) -> usize {
    let is_dspark = d.markov_w1.is_some();
    if is_dspark {
        if d.m.header.sample_from_anchor {
            n_draft
        } else {
            n_draft + 1
        }
    } else {
        n_draft + 1
    }
}

fn dump(path: &str) -> Result<(), String> {
    let g = gguf::Gguf::open(path)?;
    println!(
        "file        {path} ({} bytes, GGUF v{})",
        g.file_len, g.version
    );
    println!("alignment   {}", g.alignment);

    println!("\nmetadata");
    for (k, v) in &g.kv {
        if k.starts_with("tokenizer.") {
            continue; // tens of thousands of tokens; summarized below
        }
        let text = match v {
            gguf::Value::Array(a) => {
                let small: Vec<String> = a
                    .iter()
                    .take(12)
                    .map(|x| match x {
                        gguf::Value::String(s) => s.clone(),
                        other => format!("{:?}", other),
                    })
                    .collect();
                if a.len() <= 12 {
                    format!("[{}]", small.join(", "))
                } else {
                    format!("[{} ... {} values]", small.join(", "), a.len())
                }
            }
            gguf::Value::String(s) => s.clone(),
            gguf::Value::F32(x) => format!("{x}"),
            gguf::Value::F64(x) => format!("{x}"),
            gguf::Value::Bool(b) => format!("{b}"),
            other => format!("{:?}", other),
        };
        println!("  {k:<48} {text}");
    }
    if let Some(arr) = g.get_arr("tokenizer.ggml.tokens") {
        println!(
            "\ntokenizer   {} tokens, model {}",
            arr.len(),
            g.get_str("tokenizer.ggml.model").unwrap_or("?")
        );
    }

    println!("\ntensors     {} entries", g.tensors.len());
    println!(
        "  {:<28} {:>14} {:>10} {:>8}",
        "name", "shape", "type", "bytes"
    );
    let mut total = 0u64;
    for t in &g.tensors {
        let bytes = gguf::type_size(t.dtype, t.ne()).unwrap_or(0);
        total += bytes;
        let shape: Vec<String> = t.dims.iter().map(|d| d.to_string()).collect();
        println!(
            "  {:<28} {:>14} {:>10} {:>8}",
            t.name,
            shape.join("x"),
            gguf::type_name(t.dtype),
            bytes
        );
    }
    println!("  total {:.3} GB", total as f64 / 1e9);
    Ok(())
}

fn main() -> ExitCode {
    let a: Vec<String> = std::env::args().collect();
    let run = match a.get(1).map(|s| s.as_str()) {
        Some("dump") if a.len() == 3 => dump(&a[2]),
        Some("check") if a.len() == 3 => check::check(&a[2]),
        Some("forward") => forward(&a[2..]),
        Some("spec") => spec::run(&a[2..]),
        Some("convert") if a.len() >= 4 => {
            let mut force = false;
            let mut verify = false;
            for f in &a[4..] {
                match f.as_str() {
                    "--force" => force = true,
                    "--verify" => verify = true,
                    other => {
                        eprintln!("dspark: unknown option {other}\n{}", usage());
                        return ExitCode::from(2);
                    }
                }
            }
            convert::convert(&a[2], &a[3], force, verify)
        }
        _ => {
            eprintln!("{}", usage());
            return ExitCode::from(2);
        }
    };
    match run {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("dspark: {e}");
            ExitCode::FAILURE
        }
    }
}
