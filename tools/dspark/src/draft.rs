//! The CPU draft executor: feature encoder and block-diffusion decoder for a
//! converted DSpark draft, line for line against the reference fork.
//!
//! Reference: the encoder graph at src/models/dflash.cpp:364-392, the decoder
//! graph at 856-1178, the DSpark Markov head at 396-512. The batches and
//! proposals are driven at common/speculative.cpp:1240-1580. Every dimension
//! comes from the manifest; this file carries no Spark constant.

use crate::manifest::{dtype_bytes, dtype_name, Manifest, Tensor};
use rayon::prelude::*;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

pub const DTYPE_F32: u8 = 0;
pub const DTYPE_F16: u8 = 1;
pub const DTYPE_BF16: u8 = 2;

/// The target's tied input/output embedding, as the draft block reads it.
///
/// The draft needs two rows per decode: the anchor token's embedding and the
/// mask token's. In the reference those rows come from the target model's own
/// `token_embd` tensor (src/models/dflash.cpp:900, `llm_build_lora_mm` on the
/// target embedding); the CPU executor takes them through this trait so it can
/// read the vq-runtime target in place instead of copying 131072 x 2560 floats.
pub trait TargetEmbed: Sync {
    fn hidden(&self) -> usize;
    fn vocab(&self) -> usize;
    fn row(&self, id: u32) -> &[f32];
}

impl TargetEmbed for crate::target::Embedding {
    fn hidden(&self) -> usize {
        self.cols
    }

    fn vocab(&self) -> usize {
        self.rows
    }

    fn row(&self, id: u32) -> &[f32] {
        crate::target::Embedding::row(self, id as usize)
    }
}

impl TargetEmbed for vq_runtime::model::Model {
    fn hidden(&self) -> usize {
        self.w.cfg.hidden
    }

    fn vocab(&self) -> usize {
        self.w.cfg.vocab
    }

    fn row(&self, id: u32) -> &[f32] {
        self.embed_row(id)
    }
}

#[derive(Clone, Debug)]
pub struct Mat {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

impl Mat {
    fn row(&self, r: usize) -> &[f32] {
        &self.data[r * self.cols..(r + 1) * self.cols]
    }
}

#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let n = a.len();
    let mut i = 0;
    while i + 8 <= n {
        for k in 0..8 {
            acc[k] += a[i + k] * b[i + k];
        }
        i += 8;
    }
    let mut s = ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
    while i < n {
        s += a[i] * b[i];
        i += 1;
    }
    s
}

/// y = W x, W stored as ggml does: ne0 fastest, so row r occupies
/// data[r*cols .. r*cols + cols] (ggml/src/ggml.c:3391, result a.ne1 rows).
fn mm(t: &Mat, x: &[f32]) -> Vec<f32> {
    debug_assert_eq!(x.len(), t.cols);
    let mut y = vec![0f32; t.rows];
    y.par_iter_mut().enumerate().for_each(|(r, o)| {
        *o = dot(t.row(r), x);
    });
    y
}

fn rms_mul(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let mut ss = 0f64;
    for v in x {
        ss += f64::from(*v) * f64::from(*v);
    }
    let inv = (1.0 / ((ss / x.len() as f64) + f64::from(eps)).sqrt()) as f32;
    x.iter().zip(w.iter()).map(|(a, b)| a * inv * b).collect()
}

fn rms_only(x: &[f32], eps: f32) -> Vec<f32> {
    let mut ss = 0f64;
    for v in x {
        ss += f64::from(*v) * f64::from(*v);
    }
    let inv = (1.0 / ((ss / x.len() as f64) + f64::from(eps)).sqrt()) as f32;
    x.iter().map(|a| a * inv).collect()
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    (1.0 + (-x).exp()).recip()
}

/// Rotary on the first `dim` of one head vector: theta frequencies and the two
/// rotated halves. Reference calls ggml_rope_ext with the layer's n_rot, which
/// is the SWA rope dimension (256) for these all-SWA draft layers.
fn rope_head(x: &mut [f32], pos: usize, theta: f32, dim: usize) {
    let half = dim / 2;
    for i in 0..half {
        let freq = 1.0 / theta.powf((2 * i) as f32 / dim as f32);
        let a = pos as f32 * freq;
        let (c, s) = (a.cos(), a.sin());
        let lo = x[i];
        let hi = x[i + half];
        x[i] = lo * c - hi * s;
        x[i + half] = hi * c + lo * s;
    }
}

/// Read one manifest tensor's region and convert it to f32. The bytes are
/// copied under a conversion, not a decoding: F32, F16 and BF16 pass here, and
/// a quantized draft needs a kernel this build does not carry.
fn load_mat(path: &Path, t: &Tensor) -> Result<Mat, String> {
    let per = dtype_bytes(t.dtype)
        .ok_or_else(|| format!("{}: dtype code {} is not implemented", t.label(), t.dtype))?;
    let elems = (t.ne0 * t.ne1) as usize;
    if t.bytes != (elems as u64) * per {
        return Err(format!(
            "{}: {} bytes recorded, {}x{} of {} needs {}",
            t.label(),
            t.bytes,
            t.ne0,
            t.ne1,
            dtype_name(t.dtype),
            (elems as u64) * per
        ));
    }
    let mut f = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    f.seek(SeekFrom::Start(t.offset))
        .map_err(|e| format!("{}: {e}", path.display()))?;
    let mut raw = vec![0u8; t.bytes as usize];
    f.read_exact(&mut raw)
        .map_err(|e| format!("{}: {e}", path.display()))?;

    let mut data = Vec::with_capacity(elems);
    match t.dtype {
        DTYPE_F32 => {
            for w in raw.chunks_exact(4) {
                data.push(f32::from_le_bytes([w[0], w[1], w[2], w[3]]));
            }
        }
        DTYPE_BF16 => {
            for w in raw.chunks_exact(2) {
                // ggml_compute_bf16_to_fp32: shift the 16 BF16 bits into the
                // high half of the f32 word (ggml-impl.h:606).
                let bits = u32::from(u16::from_le_bytes([w[0], w[1]])) << 16;
                data.push(f32::from_bits(bits));
            }
        }
        DTYPE_F16 => {
            // IEEE half: 1 sign, 5 exponent bits (bias 15), 10 mantissa bits.
            for w in raw.chunks_exact(2) {
                let h = u16::from_le_bytes([w[0], w[1]]);
                let sign = (h & 0x8000) != 0;
                let exp = (h & 0x7c00) >> 10;
                let man = (h & 0x03ff) as f32 / 1024.0;
                let v = if exp == 0 {
                    man * 2f32.powf(-14.0)
                } else if exp == 0x1f {
                    if man == 0.0 {
                        f32::INFINITY
                    } else {
                        f32::NAN
                    }
                } else {
                    (1.0 + man) * 2f32.powf(exp as f32 - 15.0)
                };
                data.push(if sign { -v } else { v });
            }
        }
        other => return Err(format!("{}: dtype code {other} is not readable", t.label())),
    }
    Ok(Mat {
        rows: t.ne1 as usize,
        cols: t.ne0 as usize,
        data,
    })
}

/// A vector tensor is [ne0, 1]: one contiguous row of ne0 floats.
fn vector(dir: &Path, m: &Manifest, role: &str, layer: i32) -> Result<Vec<f32>, String> {
    let t = m
        .find(role, layer)
        .ok_or_else(|| format!("manifest has no {role} {layer} tensor"))?;
    let mat = load_mat(&dir.join("weights.bin"), t)?;
    if mat.rows != 1 {
        return Err(format!(
            "{}: is {}x{}, not a vector",
            t.label(),
            t.ne0,
            t.ne1
        ));
    }
    Ok(mat.data)
}

fn matrix(dir: &Path, m: &Manifest, role: &str, layer: i32) -> Result<Mat, String> {
    let t = m
        .find(role, layer)
        .ok_or_else(|| format!("manifest has no {role} {layer} tensor"))?;
    load_mat(&dir.join("weights.bin"), t)
}

fn optional(dir: &Path, m: &Manifest, role: &str, layer: i32) -> Result<Option<Mat>, String> {
    if m.find(role, layer).is_none() {
        return Ok(None);
    }
    matrix(dir, m, role, layer).map(Some)
}

#[derive(Clone)]
pub struct DraftLayer {
    pub norm: Vec<f32>,
    pub q: Mat,
    pub k: Mat,
    pub v: Mat,
    pub gate: Option<Mat>,
    pub out: Mat,
    pub q_norm: Option<Vec<f32>>,
    pub k_norm: Option<Vec<f32>>,
    pub ffn_norm: Vec<f32>,
    pub ffn_gate: Mat,
    pub ffn_up: Mat,
    pub ffn_down: Mat,
    pub attn_post: Option<Vec<f32>>,
    pub ffn_post: Option<Vec<f32>>,
    pub out_scale: f32,
}

pub struct Draft {
    pub m: Manifest,
    pub layers: Vec<DraftLayer>,
    pub fc: Mat,
    pub fc_norm: Vec<Vec<f32>>,
    pub enc_norm: Vec<f32>,
    pub out_norm: Vec<f32>,
    pub markov_w1: Option<Mat>,
    pub markov_w2: Option<Mat>,
    pub conf_proj: Option<Mat>,
    pub conf_bias: f32,
}

impl Draft {
    pub fn open(dir: &Path) -> Result<Draft, String> {
        let text = std::fs::read_to_string(dir.join("manifest.txt"))
            .map_err(|e| format!("{}: {e}", dir.display()))?;
        let m = Manifest::parse(&text)?;
        let len = std::fs::metadata(dir.join("weights.bin"))
            .map_err(|e| format!("{}: {e}", dir.display()))?
            .len();
        m.validate(len)?;

        let hidden = m.header.hidden as usize;
        let layers_n = m.header.layers as usize;
        let mut layers = Vec::with_capacity(layers_n);
        for l in 0..layers_n {
            let li = l as i32;
            let norm = vector(dir, &m, "attn_norm", li)?;
            let q = matrix(dir, &m, "attn_q", li)?;
            let k = matrix(dir, &m, "attn_k", li)?;
            let v = matrix(dir, &m, "attn_v", li)?;
            let out = matrix(dir, &m, "attn_out", li)?;
            if norm.len() != hidden
                || q.cols != hidden
                || k.cols != hidden
                || v.cols != hidden
                || out.cols != self_q_dim(&m)
            {
                return Err(format!(
                    "layer {li}: a norm or Q/K/V/out box does not match the header"
                ));
            }
            let gate = optional(dir, &m, "attn_gate", li)?;
            let q_norm = if m.find("attn_q_norm", li).is_some() {
                Some(vector(dir, &m, "attn_q_norm", li)?)
            } else {
                None
            };
            let k_norm = if m.find("attn_k_norm", li).is_some() {
                Some(vector(dir, &m, "attn_k_norm", li)?)
            } else {
                None
            };
            let ffn_norm = vector(dir, &m, "ffn_norm", li)?;
            let ffn_gate = matrix(dir, &m, "ffn_gate", li)?;
            let ffn_up = matrix(dir, &m, "ffn_up", li)?;
            let ffn_down = matrix(dir, &m, "ffn_down", li)?;
            if ffn_norm.len() != hidden
                || ffn_gate.cols != hidden
                || ffn_up.cols != hidden
                || ffn_down.rows != hidden
            {
                return Err(format!("layer {li}: ffn box does not match the header"));
            }
            let attn_post = if m.find("attn_post_norm", li).is_some() {
                Some(vector(dir, &m, "attn_post_norm", li)?)
            } else {
                None
            };
            let ffn_post = if m.find("ffn_post_norm", li).is_some() {
                Some(vector(dir, &m, "ffn_post_norm", li)?)
            } else {
                None
            };
            let out_scale = if m.find("layer_out_scale", li).is_some() {
                let t = m.find("layer_out_scale", li).unwrap();
                load_mat(&dir.join("weights.bin"), t)?.data[0]
            } else {
                0.0
            };
            layers.push(DraftLayer {
                norm,
                q,
                k,
                v,
                gate,
                out,
                q_norm,
                k_norm,
                ffn_norm,
                ffn_gate,
                ffn_up,
                ffn_down,
                attn_post,
                ffn_post,
                out_scale,
            });
        }

        let fc = matrix(dir, &m, "fc", -1)?;
        let fc_t = m.find("fc_norm", -1).ok_or("manifest has no fc_norm")?;
        let fc_mat = load_mat(&dir.join("weights.bin"), fc_t)?;
        if fc_mat.rows != m.header.target_layers.len() as usize || fc_mat.cols != hidden {
            return Err(format!(
                "fc_norm: {} rows and {} floats, expected {} rows of {hidden}",
                fc_mat.rows,
                fc_mat.cols,
                m.header.target_layers.len()
            ));
        }
        let fc_norm: Vec<Vec<f32>> = (0..fc_mat.rows).map(|r| fc_mat.row(r).to_vec()).collect();
        let enc_norm = vector(dir, &m, "enc_norm", -1)?;
        let out_norm = vector(dir, &m, "output_norm", -1)?;
        if enc_norm.len() != hidden || out_norm.len() != hidden {
            return Err("enc_norm or output_norm are not hidden-wide".into());
        }

        let markov_w1 = optional(dir, &m, "markov_w1", -1)?;
        let markov_w2 = optional(dir, &m, "markov_w2", -1)?;
        let conf_proj = optional(dir, &m, "conf_proj", -1)?;
        let conf_bias = if m.find("conf_bias", -1).is_some() {
            let t = m.find("conf_bias", -1).unwrap();
            let mat = load_mat(&dir.join("weights.bin"), t)?;
            if mat.data.len() != 1 {
                return Err("conf_bias is not one float".into());
            }
            mat.data[0]
        } else {
            0.0
        };

        Ok(Draft {
            m,
            layers,
            fc,
            fc_norm,
            enc_norm,
            out_norm,
            markov_w1,
            markov_w2,
            conf_proj,
            conf_bias,
        })
    }

    pub fn hidden(&self) -> usize {
        self.m.header.hidden as usize
    }

    pub fn q_dim(&self) -> usize {
        self.m.header.heads as usize * self.m.header.head_dim as usize
    }
}

fn self_q_dim(m: &Manifest) -> usize {
    m.header.heads as usize * m.header.head_dim as usize
}

/// The layer-input hidden rows a target forward exports for the draft's
/// declared target_layers. SPHID 1, written by vq-runtime --capture-layers:
/// layer-major, position-major f32 blocks.
pub struct Hidden {
    pub hidden: usize,
    pub tokens: usize,
    pub layers: Vec<u32>,
    /// The target positions the rows were computed at, 0-based and contiguous.
    pub positions: Vec<usize>,
    pub rows: Vec<Vec<f32>>,
}

/// A tagged capture file: `fields` text lines at the head, then f32 values.
fn read_headed(path: &Path, tag: &str, fields: usize) -> Result<(Vec<String>, Vec<u8>), String> {
    let raw = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut marks = Vec::new();
    for (i, b) in raw.iter().enumerate() {
        if *b == b'\n' {
            marks.push(i);
            if marks.len() == fields {
                break;
            }
        }
    }
    if marks.len() != fields {
        return Err(format!(
            "{}: header has {} lines, expected {fields}",
            path.display(),
            marks.len()
        ));
    }
    let cut = marks[fields - 1] + 1;
    let text = std::str::from_utf8(&raw[..cut])
        .map_err(|_| format!("{}: capturing header is not UTF-8", path.display()))?;
    let lines: Vec<String> = text.lines().map(|l| l.to_string()).collect();
    let head_words: Vec<String> = lines[0].split_whitespace().map(|s| s.to_string()).collect();
    if head_words != [tag.to_string(), "1".to_string()] {
        return Err(format!(
            "{}: first line is {:?}, expected {tag} 1",
            path.display(),
            head_words
        ));
    }
    Ok((lines, raw[cut..].to_vec()))
}

impl Hidden {
    pub fn open(path: &Path) -> Result<Hidden, String> {
        let (lines, data) = read_headed(path, "SPHID", 4)?;
        // The format's four fields are each one line with one argument.
        let hidden = lines
            .get(1)
            .filter(|l| l.starts_with("hidden "))
            .and_then(|l| l.split_whitespace().nth(1))
            .ok_or("SPHID: hidden line is malformed")?
            .parse::<usize>();
        let hidden = hidden.map_err(|_| "SPHID: hidden is not an integer")?;
        if hidden == 0 {
            return Err("hidden must be non-zero".into());
        }
        let tokens = lines
            .get(2)
            .filter(|l| l.starts_with("tokens "))
            .and_then(|l| l.split_whitespace().nth(1))
            .ok_or("SPHID: tokens line is malformed")?
            .parse::<usize>();
        let tokens = tokens.map_err(|_| "SPHID: tokens is not an integer")?;
        let layers_text = lines
            .get(3)
            .filter(|l| l.starts_with("layers "))
            .and_then(|l| l.split_whitespace().nth(1))
            .ok_or("SPHID: layers line is malformed")?;
        let layers = layers_text
            .split(',')
            .map(|s| {
                s.parse::<u32>()
                    .map_err(|_| format!("SPHID: layer '{s}' is not a number"))
            })
            .collect::<Result<Vec<u32>, String>>()?;
        if layers.is_empty() {
            return Err("SPHID: layers list is empty".into());
        }
        let want = layers.len() * tokens * hidden * 4;
        if data.len() != want {
            return Err(format!(
                "SPHID: carries {} bytes, {} layers x {} positions x {} floats need {want}",
                data.len(),
                layers.len(),
                tokens,
                hidden
            ));
        }
        let mut rows = Vec::with_capacity(layers.len());
        for k in 0..layers.len() {
            let start = k * tokens * hidden * 4;
            let block = &data[start..start + tokens * hidden * 4];
            let mut v = Vec::with_capacity(tokens * hidden);
            for w in block.chunks_exact(4) {
                v.push(f32::from_le_bytes([w[0], w[1], w[2], w[3]]));
            }
            rows.push(v);
        }
        Ok(Hidden {
            hidden,
            tokens,
            layers,
            positions: (0..tokens).collect(),
            rows,
        })
    }

    /// One position's captured row in one capture slot.
    pub fn position(&self, slot: usize, pos: usize) -> &[f32] {
        &self.rows[slot][pos * self.hidden..(pos + 1) * self.hidden]
    }
}

/// The fused encoder output and the K/V it injected into the draft cache.
/// Reference: dflash.cpp graph<true> at 364-392 builds fc and enc_norm, then
/// the per-layer K/V at 920-966; common/speculative.cpp:1286-1335 drives the
/// embd batch at the prompt positions.
pub struct Injection {
    pub positions: Vec<usize>,
    pub k: Vec<Vec<Vec<f32>>>,
    pub v: Vec<Vec<Vec<f32>>>,
}

fn rope_for(m: &Manifest, layer: usize) -> (usize, f32) {
    let swa = m.header.swa_pattern.get(layer).copied().unwrap_or(false);
    if swa {
        (m.header.rope_swa_dims as usize, m.header.rope_swa_theta)
    } else {
        (m.header.rope_full_dims as usize, m.header.rope_full_theta)
    }
}

fn rope_vec(v: &mut [f32], heads: usize, head_dim: usize, pos: usize, nrot: usize, theta: f32) {
    for h in 0..heads {
        let lo = h * head_dim;
        let dim = nrot.min(head_dim);
        rope_head(&mut v[lo..lo + head_dim], pos, theta, dim);
    }
}

/// Feature encoder: per-target-layer RMS norm with fc_norm, concatenate in
/// target_layers order, fc, enc_norm, then per-layer K/V for the draft cache.
pub fn encode_features(d: &Draft, hidden: &Hidden) -> Result<Injection, String> {
    if hidden.layers != d.m.header.target_layers {
        return Err(format!(
            "capture layers {:?} are not the manifest target_layers {:?}",
            hidden.layers, d.m.header.target_layers
        ));
    }
    let nl = hidden.layers.len();
    let h = d.hidden();
    let eps = d.m.header.eps;
    let mut g = Vec::with_capacity(hidden.tokens);
    for p in 0..hidden.tokens {
        let mut concat = Vec::with_capacity(nl * h);
        for k in 0..nl {
            let row = hidden.position(k, p);
            if row.len() != h {
                return Err(format!("capture slot {k} row {p} is {} wide", row.len()));
            }
            let normed = rms_mul(row, &d.fc_norm[k], eps);
            concat.extend_from_slice(&normed);
        }
        let fused = mm(&d.fc, &concat);
        g.push(rms_mul(&fused, &d.enc_norm, eps));
    }

    let mut k = Vec::with_capacity(d.layers.len());
    let mut v = Vec::with_capacity(d.layers.len());
    for (li, layer) in d.layers.iter().enumerate() {
        let (nrot, theta) = rope_for(&d.m, li);
        let mut kl = Vec::with_capacity(hidden.tokens);
        let mut vl = Vec::with_capacity(hidden.tokens);
        for p in 0..hidden.tokens {
            let mut kv = mm(&layer.k, &g[p]);
            let vv = mm(&layer.v, &g[p]);
            if let Some(w) = &layer.k_norm {
                for c in 0..d.m.header.kv_heads as usize {
                    let lo = c * d.m.header.head_dim as usize;
                    let hi = lo + d.m.header.head_dim as usize;
                    let n = rms_mul(&kv[lo..hi], w, eps);
                    kv[lo..hi].copy_from_slice(&n);
                }
            }
            rope_vec(
                &mut kv,
                d.m.header.kv_heads as usize,
                d.m.header.head_dim as usize,
                hidden.positions[p],
                nrot,
                theta,
            );
            kl.push(kv);
            vl.push(vv);
        }
        k.push(kl);
        v.push(vl);
    }
    Ok(Injection {
        positions: hidden.positions.clone(),
        k,
        v,
    })
}

pub struct BlockResult {
    /// Biased logits per block row, vocab-wide (the Markov head output).
    pub logits: Vec<Vec<f32>>,
    /// sigmoid(conf_proj . [hidden; markov_w1[prev]] + bias), one per row.
    pub conf: Vec<f32>,
    /// Greedy argmax per row, the deterministic proposal a gate can compare.
    pub argmax: Vec<u32>,
}

/// One block-diffusion decode. `n_draft` is the requested draft length; the
/// block row count follows the reference's anchor rule (common/speculative.cpp:1146).
pub fn decode_block(
    d: &Draft,
    inj: &Injection,
    embed: &dyn TargetEmbed,
    anchor: u32,
    mask: u32,
    pos0: usize,
    n_draft: usize,
) -> Result<BlockResult, String> {
    let m = &d.m;
    let is_dspark = d.markov_w1.is_some();
    let anchor_mode = m.header.sample_from_anchor;
    let n_block = if is_dspark {
        if anchor_mode {
            n_draft
        } else {
            n_draft + 1
        }
    } else {
        n_draft + 1
    };
    if n_draft == 0 || n_draft > m.header.block_size as usize {
        return Err(format!(
            "n_draft {n_draft} is outside the trained block_size {}",
            m.header.block_size
        ));
    }
    if n_block > m.header.block_size as usize + 1 {
        return Err(format!("block row count {n_block} exceeds block_size+1"));
    }
    if embed.hidden() != d.hidden() || embed.vocab() < m.header.vocab as usize {
        return Err("target embedding box does not match the draft header".into());
    }

    let h = d.hidden();
    let hd = m.header.head_dim as usize;
    let heads = m.header.heads as usize;
    let kv_heads = m.header.kv_heads as usize;
    let qdim = d.q_dim();
    let eps = m.header.eps;
    let scale = 1.0 / (hd as f32).sqrt();
    // DSpark adds one to the trained window for the anchor row
    // (src/models/dflash.cpp:119-122), and the block mask pins every query to
    // the anchor row's history (dflash.cpp:513-535).
    let nswa = if is_dspark {
        m.header.swa_window as i64 + 1
    } else {
        m.header.swa_window as i64
    };

    let mut tokens = vec![mask; n_block];
    tokens[0] = anchor;
    let positions: Vec<usize> = (0..n_block).map(|i| pos0 + i).collect();

    let mut x: Vec<Vec<f32>> = Vec::with_capacity(n_block);
    for i in 0..n_block {
        let ti = tokens[i];
        if ti as usize >= embed.vocab() {
            return Err(format!(
                "token {} is outside the target embedding",
                tokens[i]
            ));
        }
        x.push(embed.row(ti).to_vec());
    }

    // Per-layer own K/V, computed for every block row before any query runs,
    // because the reference copies the block's K/V into the cache and then
    // attends over cache plus block in one non-causal graph.
    let mut own_k: Vec<Vec<Vec<f32>>> = Vec::with_capacity(d.layers.len());
    let mut own_v: Vec<Vec<Vec<f32>>> = Vec::with_capacity(d.layers.len());
    let mut layer_x = x.clone();
    for (li, layer) in d.layers.iter().enumerate() {
        let (nrot, theta) = rope_for(m, li);
        let mut qk = Vec::with_capacity(n_block);
        let mut vv = Vec::with_capacity(n_block);
        let mut qrows = Vec::with_capacity(n_block);
        for i in 0..n_block {
            let xn = rms_mul(&layer_x[i], &layer.norm, eps);
            let mut q = mm(&layer.q, &xn);
            let mut kk = mm(&layer.k, &xn);
            let mut vvi = mm(&layer.v, &xn);
            if let (Some(qw), Some(kw)) = (&layer.q_norm, &layer.k_norm) {
                for c in 0..heads {
                    let lo = c * hd;
                    let n = rms_mul(&q[lo..lo + hd], qw, eps);
                    q[lo..lo + hd].copy_from_slice(&n);
                }
                for c in 0..kv_heads {
                    let lo = c * hd;
                    let n = rms_mul(&kk[lo..lo + hd], kw, eps);
                    kk[lo..lo + hd].copy_from_slice(&n);
                }
            }
            if layer.v.cols == 0 {
                // shared KV: the reference RMS-norms V when wv is absent
                vvi = rms_only(&kk, eps);
            }
            rope_vec(&mut q, heads, hd, positions[i], nrot, theta);
            rope_vec(&mut kk, kv_heads, hd, positions[i], nrot, theta);
            qrows.push(q);
            qk.push(kk);
            vv.push(vvi);
        }
        own_k.push(qk);
        own_v.push(vv);

        // Attention: every query uses the anchor row's mask, so the key set is
        // the same for all rows and is fixed by pos0.
        let p1 = positions[0] as i64;
        let mut keys: Vec<(i64, &[f32], &[f32])> = Vec::new();
        for (j, p) in inj.positions.iter().enumerate() {
            let p0 = *p as i64;
            let keep = (!m.header.causal || p0 <= p1) && (p1 - p0 < nswa);
            if keep {
                keys.push((p0, &inj.k[li][j], &inj.v[li][j]));
            }
        }
        for (i, p) in positions.iter().enumerate() {
            let p0 = *p as i64;
            let keep = (!m.header.causal || p0 <= p1) && (p1 - p0 < nswa);
            if keep {
                keys.push((p0, &own_k[li][i], &own_v[li][i]));
            }
        }
        if keys.is_empty() {
            return Err(format!("layer {li}: the anchor row's mask keeps no keys"));
        }

        let mut next_x = Vec::with_capacity(n_block);
        for i in 0..n_block {
            let xn = rms_mul(&layer_x[i], &layer.norm, eps);
            // the per-head gate reads this row's normed input, not row 0's
            let gate_row = layer.gate.as_ref().map(|g| mm(g, &xn));
            let q = &qrows[i];
            let mut attn = vec![0f32; qdim];
            for hh in 0..heads {
                let group = hh * kv_heads / heads;
                let qh = &q[hh * hd..(hh + 1) * hd];
                let mut sc = vec![0f32; keys.len()];
                let mut mx = f32::NEG_INFINITY;
                for (j, (_, kk, _)) in keys.iter().enumerate() {
                    let kh = &kk[group * hd..(group + 1) * hd];
                    sc[j] = dot(qh, kh) * scale;
                    mx = mx.max(sc[j]);
                }
                let mut sum = 0f32;
                for s in sc.iter_mut() {
                    *s = (*s - mx).exp();
                    sum += *s;
                }
                let inv = 1.0 / sum;
                let mut o = vec![0f32; hd];
                for (j, (_, _, vrow)) in keys.iter().enumerate() {
                    let p = sc[j] * inv;
                    let vh = &vrow[group * hd..(group + 1) * hd];
                    for d2 in 0..hd {
                        o[d2] += p * vh[d2];
                    }
                }
                if let Some(gv) = &gate_row {
                    let g = sigmoid(gv[hh]);
                    for d2 in 0..hd {
                        o[d2] *= g;
                    }
                }
                attn[hh * hd..(hh + 1) * hd].copy_from_slice(&o);
            }
            let mut out = mm(&layer.out, &attn);
            if let Some(w) = &layer.attn_post {
                let n = rms_mul(&out, w, eps);
                out = n;
            }
            let mut res = layer_x[i].clone();
            for k2 in 0..h {
                res[k2] += out[k2];
            }
            let xn2 = rms_mul(&res, &layer.ffn_norm, eps);
            let up = mm(&layer.ffn_up, &xn2);
            let gt = mm(&layer.ffn_gate, &xn2);
            let mut act = vec![0f32; up.len()];
            match m.header.act.as_str() {
                "gelu" | "gelu_pytorch_tanh" => {
                    for k2 in 0..act.len() {
                        act[k2] = gelu_tanh(gt[k2]) * up[k2];
                    }
                }
                "silu" => {
                    for k2 in 0..act.len() {
                        act[k2] = silu(gt[k2]) * up[k2];
                    }
                }
                other => return Err(format!("activation {other} is not implemented")),
            }
            let down = mm(&layer.ffn_down, &act);
            for k2 in 0..h {
                res[k2] += down[k2];
            }
            if let Some(w) = &layer.ffn_post {
                res = rms_mul(&res, w, eps);
            }
            if layer.out_scale != 0.0 {
                for k2 in 0..h {
                    res[k2] *= layer.out_scale;
                }
            }
            next_x.push(res);
        }
        layer_x = next_x;
    }

    let mut hid = Vec::with_capacity(n_block);
    for i in 0..n_block {
        hid.push(rms_mul(&layer_x[i], &d.out_norm, eps));
    }

    let mut logits = Vec::with_capacity(n_block);
    for i in 0..n_block {
        logits.push(mm_embed(embed, &hid[i]));
    }

    let mut conf = vec![f32::NAN; n_block];
    if let (Some(w1), Some(w2)) = (&d.markov_w1, &d.markov_w2) {
        let rank = w1.cols;
        let i_beg = if anchor_mode { 0 } else { 1 };
        let mut prev = anchor;
        if !anchor_mode {
            // slot 0 is the bonus anchor: its logits stay unbiased
            conf[0] = if d.conf_proj.is_some() {
                sigmoid(logits[0][0])
            } else {
                f32::NAN
            };
        }
        for i in i_beg..n_block {
            let w1_prev = w1.row(prev as usize);
            let bias = mm(w2, w1_prev);
            for t in 0..embed.vocab().min(m.header.vocab as usize) {
                logits[i][t] += bias[t];
            }
            if let Some(cp) = &d.conf_proj {
                let mut feat = Vec::with_capacity(h + rank);
                feat.extend_from_slice(&hid[i]);
                feat.extend_from_slice(w1_prev);
                let mut c = mm(cp, &feat);
                c[0] += d.conf_bias;
                conf[i] = sigmoid(c[0]);
            }
            if i + 1 < n_block {
                prev = argmax(&logits[i]) as u32;
            }
        }
    }

    let argmax = logits.iter().map(|l| argmax(l) as u32).collect();
    Ok(BlockResult {
        logits,
        conf,
        argmax,
    })
}

/// The draft's gelu is ggml's LLM_FFN_GELU, the tanh form
/// (ggml-cpu/vec.h:962), not torch's erf form.
fn gelu_tanh(x: f32) -> f32 {
    const A: f32 = 0.044715;
    // sqrt(2/pi) at f32 precision; ggml-cpu/vec.h:962 uses the same constant
    const S: f32 = 0.797_884_6;
    0.5 * x * (1.0 + (S * x * (1.0 + A * x * x)).tanh())
}

#[inline]
fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

fn argmax(x: &[f32]) -> usize {
    let mut best = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, v) in x.iter().enumerate() {
        if *v > bv {
            bv = *v;
            best = i;
        }
    }
    best
}

/// logits = E x, E stored [vocab, hidden] row-major like the reference's
/// token_embd/output tensor.
fn mm_embed(e: &dyn TargetEmbed, x: &[f32]) -> Vec<f32> {
    let cols = e.hidden();
    let rows = e.vocab();
    debug_assert_eq!(x.len(), cols);
    let mut y = vec![0f32; rows];
    y.par_iter_mut().enumerate().for_each(|(r, o)| {
        *o = dot(e.row(r as u32), x);
    });
    y
}
