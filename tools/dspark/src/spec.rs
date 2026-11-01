//! The CPU DSpark speculative loop: propose a block with the draft, verify it
//! with the target, commit the accepted prefix, roll the target back and
//! repeat.
//!
//! This is the same loop the reference example runs
//! (examples/speculative-simple/speculative-simple.cpp:220-330):
//!
//!   - the target holds `pos0` committed positions and `anchor`, the token at
//!     position `pos0` that has not been processed yet;
//!   - the draft block runs at positions `pos0..pos0+n_draft-1`, anchor first;
//!   - the target verifies `[anchor, p0, ..., p_{n-1}]` at `pos0..pos0+n_draft`,
//!     row `pos0+i` predicts proposal `i`;
//!   - the first mismatch `k` commits `p0..p_{k-1}` and the target's own token
//!     at row `pos0+k`; the target keeps rows `0..k` and the new anchor is the
//!     target token at position `pos0+k+1`;
//!   - the draft's injected features are the target layer inputs of the
//!     committed rows, so a rollback truncates both caches to the same length.

use crate::draft::{self, Draft, Hidden};
use std::path::Path;
use std::time::Instant;
use vq_runtime::config::Config;
use vq_runtime::model::{Model, Weights};

pub fn run(args: &[String]) -> Result<(), String> {
    let mut draft_dir = String::new();
    let mut target_dir = String::new();
    let mut weights = "hf".to_string();
    let mut cfg_path = "models/hf/config.json".to_string();
    let mut ids_arg = String::new();
    let mut gen = 0usize;
    let mut n_max: Option<usize> = None;
    let mut ctx: Option<usize> = None;
    let mut threads = 0usize;
    let mut rounds = false;
    let mut trace_logits = false;
    let mut out: Option<String> = None;

    let mut i = 0;
    while i < args.len() {
        let take = |i: usize| -> Result<String, String> {
            args.get(i + 1)
                .cloned()
                .ok_or_else(|| format!("{} needs a value", args[i]))
        };
        match args[i].as_str() {
            "--draft" => {
                draft_dir = take(i)?;
                i += 2;
            }
            "--target" => {
                target_dir = take(i)?;
                i += 2;
            }
            "--weights" => {
                weights = take(i)?;
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
            "--n-max" => {
                n_max = Some(take(i)?.parse().map_err(|_| "bad --n-max".to_string())?);
                i += 2;
            }
            "--ctx" => {
                ctx = Some(take(i)?.parse().map_err(|_| "bad --ctx".to_string())?);
                i += 2;
            }
            "--threads" => {
                threads = take(i)?.parse().map_err(|_| "bad --threads".to_string())?;
                i += 2;
            }
            "--rounds" => {
                rounds = true;
                i += 1;
            }
            "--trace-logits" => {
                trace_logits = true;
                i += 1;
            }
            "--out" => {
                out = Some(take(i)?);
                i += 2;
            }
            other => return Err(format!("unknown spec argument {other}")),
        }
    }
    if draft_dir.is_empty() || target_dir.is_empty() || ids_arg.is_empty() {
        return Err("spec needs --draft, --target and --ids".into());
    }
    if gen == 0 {
        return Err("spec needs a positive --gen".into());
    }
    let ids: Vec<u32> = ids_arg
        .split(|c: char| c == ',' || c.is_whitespace())
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<u32>().map_err(|_| format!("bad token id {s}")))
        .collect::<Result<_, _>>()?;
    if ids.len() < 2 {
        return Err("spec needs at least two prompt ids".into());
    }

    if threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .map_err(|e| format!("thread pool: {e}"))?;
    }

    let cfg = Config::load(&cfg_path)?;
    let d = Draft::open(Path::new(&draft_dir))?;
    let n_block_max = draft_max(&d);
    let n_max = n_max.unwrap_or(n_block_max).min(n_block_max);
    if n_max == 0 {
        return Err("n_max is zero for this draft".into());
    }
    let mask = d.m.header.mask_token as u32;
    let is_dspark = d.markov_w1.is_some();
    let anchor_mode = d.m.header.sample_from_anchor;
    let i_beg = if is_dspark && anchor_mode { 0 } else { 1 };

    let total = ids.len() + gen + n_max + 2;
    let ctx = ctx.unwrap_or(total).max(total);
    let t_load = Instant::now();
    let w = match weights.as_str() {
        "hf" => Weights::from_hf(cfg.clone(), Path::new(&target_dir))?,
        "vq" => Weights::from_vq(cfg.clone(), Path::new(&target_dir))?,
        other => return Err(format!("unknown --weights {other}")),
    };
    let mut m = Model::new(w, ctx);
    m.enable_capture(&d.m.header.target_layers, cfg.hidden)?;
    println!(
        "target      {} layers, hidden {}, vocab {}, {:.2} GB f32 in {:.1} s",
        cfg.layers,
        cfg.hidden,
        cfg.vocab,
        m.w.bytes() as f64 / 1e9,
        t_load.elapsed().as_secs_f64()
    );
    println!(
        "draft       {} layers, block {}, target layers {:?}, anchor mode {}, mask {}",
        d.layers.len(),
        d.m.header.block_size,
        d.m.header.target_layers,
        anchor_mode,
        mask
    );

    // The reference holds the last prompt token back: the target processes
    // positions 0..P-2 and the first verify batch starts with that token at
    // position P-1 (speculative-simple.cpp:130-150).
    let prompt_len = ids.len();
    let mut pos0 = prompt_len - 1;
    let t_prompt = Instant::now();
    for (p, id) in ids[..pos0].iter().enumerate() {
        m.forward(*id, p)?;
    }
    let prompt_s = t_prompt.elapsed().as_secs_f64();
    let mut anchor = ids[pos0];

    let mut generated: Vec<u32> = Vec::with_capacity(gen + n_max + 1);
    let mut n_drafted = 0usize;
    let mut n_accept = 0usize;
    let mut n_rounds = 0usize;
    let t_dec = Instant::now();

    while generated.len() < gen {
        // The reference drafts a full block every round and only clamps near
        // the end of the generation budget (speculative-simple.cpp:180-186),
        // so a short remaining budget must not shrink this block: the block's
        // rows condition each other and a shorter block is a different draft.
        let n_draft = n_max;
        if n_draft == 0 {
            break;
        }
        let h = hidden_from_captures(&m, &d, pos0)?;
        let inj = draft::encode_features(&d, &h)?;
        let r = draft::decode_block(&d, &inj, &m, anchor, mask, pos0, n_draft)?;
        let proposals: Vec<u32> = r.argmax[i_beg..].to_vec();
        if proposals.len() != n_draft {
            return Err(format!(
                "draft produced {} proposals for n_draft {n_draft}",
                proposals.len()
            ));
        }

        // verify: row pos0+i predicts proposal i, the extra row after the last
        // proposal predicts the bonus token when every proposal is accepted.
        let mut tgt = Vec::with_capacity(n_draft + 1);
        tgt.push(Model::argmax(m.forward(anchor, pos0)?));
        for (i, p) in proposals.iter().enumerate() {
            tgt.push(Model::argmax(m.forward(*p, pos0 + 1 + i)?));
        }
        let mut k = 0usize;
        while k < n_draft && tgt[k] == proposals[k] {
            k += 1;
        }

        generated.extend_from_slice(&proposals[..k]);
        generated.push(tgt[k]);
        n_drafted += n_draft;
        n_accept += k;
        n_rounds += 1;

        if rounds {
            println!(
                "round {:>3}  pos0 {:>3}  draft [{}]  target [{}]  accepted {}",
                n_rounds,
                pos0,
                join(&proposals),
                join(&tgt),
                k
            );
        }
        if trace_logits {
            for (i, l) in r.logits.iter().enumerate() {
                let mut order: Vec<usize> = (0..l.len()).collect();
                order.sort_unstable_by(|a, b| l[*b].partial_cmp(&l[*a]).unwrap());
                let top: Vec<String> = order
                    .iter()
                    .take(3)
                    .map(|&c| format!("{c}:{:.6}", l[c]))
                    .collect();
                println!(
                    "trace       round {:>3} pos0 {:>3} row {}: {}",
                    n_rounds,
                    pos0,
                    i,
                    top.join(" ")
                );
            }
        }

        // Keep anchor + the k accepted proposals; drop the rejected tail and
        // leave the target's own token as the next anchor.
        m.rollback(pos0 + k + 1)?;
        pos0 += k + 1;
        anchor = tgt[k];
    }

    let dec_s = t_dec.elapsed().as_secs_f64();
    generated.truncate(gen);
    println!(
        "prompt      {} tokens in {:.2} s ({:.2} tok/s)",
        prompt_len,
        prompt_s,
        prompt_len as f64 / prompt_s
    );
    println!("generated   {}", join(&generated));
    println!(
        "spec        {} rounds, {} drafted, {} accepted, acceptance {:.3}%",
        n_rounds,
        n_drafted,
        n_accept,
        100.0 * n_accept as f64 / n_drafted.max(1) as f64
    );
    println!(
        "decode      {:.2} s ({:.2} tok/s over {} committed tokens)",
        dec_s,
        (n_accept + n_rounds) as f64 / dec_s,
        n_accept + n_rounds
    );
    if let Some(path) = out {
        let text = generated
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(" ");
        std::fs::write(&path, text + "\n").map_err(|e| format!("{path}: {e}"))?;
        println!("out         {} ids written to {path}", generated.len());
    }
    Ok(())
}

fn draft_max(d: &Draft) -> usize {
    let is_dspark = d.markov_w1.is_some();
    if is_dspark && d.m.header.sample_from_anchor {
        d.m.header.block_size as usize
    } else {
        (d.m.header.block_size as usize).saturating_sub(1)
    }
}

fn join(ids: &[u32]) -> String {
    ids.iter()
        .map(|v| v.to_string())
        .collect::<Vec<_>>()
        .join(" ")
}

/// Rebuild the encoder input from the target's live layer-input captures: one
/// slot per manifest target layer, `rows` positions each.
fn hidden_from_captures(m: &Model, d: &Draft, rows: usize) -> Result<Hidden, String> {
    let h = d.hidden();
    let nl = d.m.header.target_layers.len();
    let mut slots = Vec::with_capacity(nl);
    for k in 0..nl {
        let c = m.captured(k);
        if c.len() != rows * h {
            return Err(format!(
                "capture slot {k} holds {} floats, expected {} positions x {}",
                c.len(),
                rows,
                h
            ));
        }
        slots.push(c.to_vec());
    }
    Ok(Hidden {
        hidden: h,
        tokens: rows,
        layers: d.m.header.target_layers.clone(),
        positions: (0..rows).collect(),
        rows: slots,
    })
}
