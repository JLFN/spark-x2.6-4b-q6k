//! vq-runtime — the reference inference runtime for Spark2_5.
//!
//! It reads the bf16 checkpoint (safetensors) or an RQVQ VQ artifact, runs the
//! forward pass in f32 on the CPU, and emits the next-token distribution or a
//! greedy continuation. It takes token ids, not text: tokenization is a later
//! step, and the gate compares ids, which is what llama.cpp can reproduce
//! exactly.
//!
//!   vq-runtime --weights vq --dir models/vq-build-8192-2-v3 \
//!              --config models/hf/config.json --ids 1,2,3 --gen 8
//!
//! --logits FILE writes the f32 logits of the last prompt position, for the
//! gate against llama-server's /completion probabilities.

mod config;
mod model;
mod safetensors;
mod vq_source;

use model::{Model, Weights};
use std::fs;
use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

const USAGE: &str = "\
usage: vq-runtime --weights hf|vq --dir DIR [options]
  --config PATH     model config.json (default models/hf/config.json)
  --ids LIST        comma or space separated prompt token ids
  --gen N           greedy continuation length (default 0)
  --ctx N           cache size in tokens (default 4096)
  --threads N       rayon threads (default all cores)
  --top N           print the N best next tokens (default 5)
  --logits FILE     write the last prompt position's f32 logits
  --weights hf|vq   bf16 safetensors checkpoint or RQVQ artifact
";

fn run() -> Result<(), String> {
    let a: Vec<String> = std::env::args().collect();
    let mut weights = String::new();
    let mut dir = String::new();
    let mut cfg_path = "models/hf/config.json".to_string();
    let mut ids_arg = String::new();
    let mut gen = 0usize;
    let mut ctx = 4096usize;
    let mut threads = 0usize;
    let mut topn = 5usize;
    let mut logits_out: Option<String> = None;

    let mut i = 1;
    while i < a.len() {
        let take = |i: usize| -> Result<String, String> {
            a.get(i + 1)
                .cloned()
                .ok_or_else(|| format!("{} needs a value", a[i]))
        };
        match a[i].as_str() {
            "--weights" => {
                weights = take(i)?;
                i += 2;
            }
            "--dir" => {
                dir = take(i)?;
                i += 2;
            }
            "--config" => {
                cfg_path = take(i)?;
                i += 2;
            }
            "--ids" => {
                ids_arg = take(i)?;
                i += 2;
            }
            "--gen" => {
                gen = take(i)?.parse().map_err(|_| "bad --gen".to_string())?;
                i += 2;
            }
            "--ctx" => {
                ctx = take(i)?.parse().map_err(|_| "bad --ctx".to_string())?;
                i += 2;
            }
            "--threads" => {
                threads = take(i)?.parse().map_err(|_| "bad --threads".to_string())?;
                i += 2;
            }
            "--top" => {
                topn = take(i)?.parse().map_err(|_| "bad --top".to_string())?;
                i += 2;
            }
            "--logits" => {
                logits_out = Some(take(i)?);
                i += 2;
            }
            "-h" | "--help" => {
                print!("{USAGE}");
                return Ok(());
            }
            other => return Err(format!("unknown argument {other}")),
        }
    }

    if weights != "hf" && weights != "vq" {
        return Err("--weights must be hf or vq".into());
    }
    if dir.is_empty() {
        return Err("--dir is required".into());
    }
    let ids: Vec<u32> = ids_arg
        .split(|c: char| c == ',' || c == ' ' || c == ';')
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<u32>().map_err(|e| format!("bad token id {s}: {e}")))
        .collect::<Result<_, _>>()?;
    if ids.is_empty() {
        return Err("--ids needs at least one token id".into());
    }

    if threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .map_err(|e| format!("thread pool: {e}"))?;
    }

    let cfg = config::Config::load(&cfg_path)?;
    println!(
        "model       {} layers, {} heads / {} kv, {} hidden, {} intermediate, vocab {}",
        cfg.layers, cfg.heads, cfg.kv_heads, cfg.hidden, cfg.intermediate, cfg.vocab
    );
    println!(
        "rope        full theta {} partial {}, sliding theta {} partial {}",
        cfg.rope_full.theta, cfg.rope_full.partial, cfg.rope_sliding.theta, cfg.rope_sliding.partial
    );

    let t0 = Instant::now();
    let w = match weights.as_str() {
        "hf" => Weights::from_hf(cfg, Path::new(&dir))?,
        _ => Weights::from_vq(cfg, Path::new(&dir))?,
    };
    println!(
        "weights     {} ({:.2} GB f32) in {:.1} s",
        dir,
        w.bytes() as f64 / 1e9,
        t0.elapsed().as_secs_f64()
    );

    let total = ids.len() + gen + 1;
    let mut m = Model::new(w, ctx.max(total));

    let mut logits_last = Vec::new();
    let t0 = Instant::now();
    for (pos, id) in ids.iter().enumerate() {
        let l = m.forward(*id, pos)?;
        if pos + 1 == ids.len() {
            logits_last = l.to_vec();
        }
    }
    let prompt_s = t0.elapsed().as_secs_f64();
    println!(
        "prompt      {} tokens in {:.2} s ({:.2} tok/s)",
        ids.len(),
        prompt_s,
        ids.len() as f64 / prompt_s
    );

    if let Some(path) = &logits_out {
        let mut b = Vec::with_capacity(logits_last.len() * 4);
        for v in &logits_last {
            b.extend_from_slice(&v.to_le_bytes());
        }
        fs::write(path, &b).map_err(|e| format!("{path}: {e}"))?;
        println!("logits      {} floats written to {path}", logits_last.len());
    }

    let best = Model::top(&logits_last, topn);
    print!("next        ");
    for (id, v) in &best {
        print!("[{id} {v:.4}] ");
    }
    println!();

    let mut next = Model::argmax(&logits_last);
    let mut out = Vec::with_capacity(gen);
    let t0 = Instant::now();
    for k in 0..gen {
        out.push(next);
        let pos = ids.len() + k;
        if pos + 1 >= total {
            break;
        }
        let l = m.forward(next, pos)?;
        next = Model::argmax(l);
    }
    let gen_s = t0.elapsed().as_secs_f64();
    if gen > 0 {
        println!(
            "generated   {:?} in {:.2} s ({:.2} tok/s)",
            out,
            gen_s,
            out.len() as f64 / gen_s
        );
    }
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("vq-runtime: {e}");
            ExitCode::from(2)
        }
    }
}
