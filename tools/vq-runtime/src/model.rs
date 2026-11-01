//! The Spark2_5 forward pass, line for line against models/hf/modeling_spark.py.
//!
//! Per layer (Spark2_5DecoderLayer.forward):
//!
//!   x = input_layernorm(h); a = attention(x); h = h + a
//!   x = post_attention_layernorm(h); h = h + mlp(x)
//!
//! attention (Spark2_5Attention.forward): one fused q_k_v_proj of width
//! q_dim + 2*kv_dim, q/k rope, GQA over the cache, then a per-head sigmoid gate
//! from g_proj applied after the attention, then out_proj.
//!
//! mlp (Spark2_5MLP.forward): down_proj(gelu(gate_proj(x)) * up_proj(x)), with
//! the exact erf gelu, not the tanh approximation.
//!
//! There is one deliberate, named deviation: torch runs the projections in the
//! checkpoint dtype (bf16) and keeps the residual stream in f32; this runtime
//! runs everything in f32. That is strictly more precise, and it means logits
//! can differ from llama.cpp's bf16 path in the last bits even with identical
//! weights; the gate is top-k agreement, not bit equality.

use crate::config::Config;
use crate::safetensors::Store;
use crate::vq_source::VqDir;
use rayon::prelude::*;
use std::path::Path;

extern "C" {
    fn erf(x: f64) -> f64;
}

pub struct Tensor {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

impl Tensor {
    #[inline]
    pub fn row(&self, r: usize) -> &[f32] {
        &self.data[r * self.cols..(r + 1) * self.cols]
    }
}

pub struct LayerW {
    pub in_norm: Vec<f32>,
    pub post_norm: Vec<f32>,
    pub qkv: Tensor,
    pub attn_gate: Tensor,
    pub out: Tensor,
    pub gate: Tensor,
    pub up: Tensor,
    pub down: Tensor,
    pub full: bool,
    pub rope: crate::config::Rope,
}

pub struct Weights {
    pub cfg: Config,
    pub embed: Tensor,
    pub norm: Vec<f32>,
    pub layers: Vec<LayerW>,
}

enum Loader {
    Hf(Store),
    Vq(VqDir),
}

impl Loader {
    fn tensor(&mut self, name: &str) -> Result<Tensor, String> {
        match self {
            Loader::Hf(s) => {
                let t = s.read(name)?;
                if t.shape.len() != 2 {
                    return Err(format!("{name}: expected a matrix, got {:?}", t.shape));
                }
                Ok(Tensor {
                    rows: t.shape[0],
                    cols: t.shape[1],
                    data: t.data,
                })
            }
            Loader::Vq(v) => v.read(name),
        }
    }

    fn vector(&mut self, name: &str) -> Result<Vec<f32>, String> {
        match self {
            Loader::Hf(s) => {
                let t = s.read(name)?;
                if t.shape.len() != 1 {
                    return Err(format!("{name}: expected a vector, got {:?}", t.shape));
                }
                Ok(t.data)
            }
            Loader::Vq(v) => Ok(v.read(name)?.data),
        }
    }
}

fn expect(t: &Tensor, name: &str, rows: usize, cols: usize) -> Result<(), String> {
    if t.rows != rows || t.cols != cols {
        return Err(format!(
            "{name}: is {}x{}, the architecture needs {rows}x{cols}",
            t.rows, t.cols
        ));
    }
    Ok(())
}

impl Weights {
    pub fn from_hf(cfg: Config, dir: &Path) -> Result<Weights, String> {
        let store = Store::open(dir)?;
        Self::load(cfg, Loader::Hf(store))
    }

    pub fn from_vq(cfg: Config, dir: &Path) -> Result<Weights, String> {
        let vq = VqDir::open(dir)?;
        Self::load(cfg, Loader::Vq(vq))
    }

    fn load(cfg: Config, mut l: Loader) -> Result<Weights, String> {
        let embed = l.tensor("model.embedding.weight")?;
        expect(&embed, "model.embedding.weight", cfg.vocab, cfg.hidden)?;
        let norm = l.vector("model.norm.weight")?;
        if norm.len() != cfg.hidden {
            return Err(format!("model.norm.weight: {} values", norm.len()));
        }

        let mut layers = Vec::with_capacity(cfg.layers);
        for i in 0..cfg.layers {
            let p = |s: &str| format!("model.layers.{i}.{s}");
            let in_norm = l.vector(&p("input_layernorm.weight"))?;
            let post_norm = l.vector(&p("post_attention_layernorm.weight"))?;
            if in_norm.len() != cfg.hidden || post_norm.len() != cfg.hidden {
                return Err(format!("layer {i}: a norm is not {} wide", cfg.hidden));
            }
            let qkv = l.tensor(&p("self_attn.q_k_v_proj.weight"))?;
            expect(
                &qkv,
                &p("self_attn.q_k_v_proj.weight"),
                cfg.q_dim() + 2 * cfg.kv_dim(),
                cfg.hidden,
            )?;
            let attn_gate = l.tensor(&p("self_attn.g_proj.weight"))?;
            expect(
                &attn_gate,
                &p("self_attn.g_proj.weight"),
                cfg.heads,
                cfg.hidden,
            )?;
            let out = l.tensor(&p("self_attn.out_proj.weight"))?;
            expect(
                &out,
                &p("self_attn.out_proj.weight"),
                cfg.hidden,
                cfg.q_dim(),
            )?;
            let gate = l.tensor(&p("mlp.gate_proj.weight"))?;
            expect(
                &gate,
                &p("mlp.gate_proj.weight"),
                cfg.intermediate,
                cfg.hidden,
            )?;
            let up = l.tensor(&p("mlp.up_proj.weight"))?;
            expect(&up, &p("mlp.up_proj.weight"), cfg.intermediate, cfg.hidden)?;
            let down = l.tensor(&p("mlp.down_proj.weight"))?;
            expect(
                &down,
                &p("mlp.down_proj.weight"),
                cfg.hidden,
                cfg.intermediate,
            )?;
            layers.push(LayerW {
                in_norm,
                post_norm,
                qkv,
                attn_gate,
                out,
                gate,
                up,
                down,
                full: cfg.is_full(i),
                rope: cfg.rope(i),
            });
        }
        Ok(Weights {
            cfg,
            embed,
            norm,
            layers,
        })
    }

    /// Bytes held by the weights, for the load report.
    pub fn bytes(&self) -> usize {
        let mut n = self.embed.data.len() * 4 + self.norm.len() * 4;
        for l in &self.layers {
            n += (l.in_norm.len() + l.post_norm.len()) * 4;
            for t in [&l.qkv, &l.attn_gate, &l.out, &l.gate, &l.up, &l.down] {
                n += t.data.len() * 4;
            }
        }
        n
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

/// y = W x, one rayon task per row.
fn matvec(t: &Tensor, x: &[f32], y: &mut [f32]) {
    debug_assert_eq!(x.len(), t.cols);
    y.par_iter_mut().enumerate().for_each(|(r, o)| {
        *o = dot(t.row(r), x);
    });
}

fn rms_norm(x: &[f32], w: &[f32], eps: f32, out: &mut [f32]) {
    let mut ss = 0f64;
    for v in x {
        ss += f64::from(*v) * f64::from(*v);
    }
    let inv = (1.0 / ((ss / x.len() as f64) + f64::from(eps)).sqrt()) as f32;
    for i in 0..x.len() {
        out[i] = x[i] * inv * w[i];
    }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
fn gelu(x: f32) -> f32 {
    /* torch's exact gelu: x * 0.5 * (1 + erf(x / sqrt(2))). libm's erf is the
     * same function torch's CPU kernel calls. */
    let e = unsafe { erf(f64::from(x) * std::f64::consts::FRAC_1_SQRT_2) };
    0.5 * x * (1.0 + e as f32)
}

pub struct Model {
    pub w: Weights,
    kv_k: Vec<Vec<f32>>,
    kv_v: Vec<Vec<f32>>,
    /// Positions cached so far, and the capacity of each cache.
    pub len: usize,
    cap: usize,
    max_ctx: usize,
    scratch: Scratch,
    logits: Vec<f32>,
    cos: Vec<f32>,
    sin: Vec<f32>,
    /// Hidden rows at the residual input of each requested layer, captured
    /// while the forward pass runs. One slot per requested layer; a row is
    /// appended at every position it visits. Layer-input capture is what the
    /// draft feature encoder reads (llama-graph.cpp:1451, res->t_layer_inp[il]
    /// = inpL at the start of a layer in the reference trunk graphs).
    capture_slots: Vec<Option<usize>>,
    captures: Vec<Vec<f32>>,
}

struct Scratch {
    h: Vec<f32>,
    xn: Vec<f32>,
    qkv: Vec<f32>,
    gate: Vec<f32>,
    attn: Vec<f32>,
    o: Vec<f32>,
    g: Vec<f32>,
    u: Vec<f32>,
    act: Vec<f32>,
}

impl Model {
    pub fn new(w: Weights, max_ctx: usize) -> Model {
        let c = &w.cfg;
        Model {
            kv_k: (0..c.layers).map(|_| Vec::new()).collect(),
            kv_v: (0..c.layers).map(|_| Vec::new()).collect(),
            len: 0,
            cap: 0,
            max_ctx,
            scratch: Scratch {
                h: vec![0f32; c.hidden],
                xn: vec![0f32; c.hidden],
                qkv: vec![0f32; c.q_dim() + 2 * c.kv_dim()],
                gate: vec![0f32; c.heads],
                attn: vec![0f32; c.q_dim()],
                o: vec![0f32; c.hidden],
                g: vec![0f32; c.intermediate],
                u: vec![0f32; c.intermediate],
                act: vec![0f32; c.intermediate],
            },
            logits: vec![0f32; c.vocab],
            cos: vec![0f32; c.head_dim],
            sin: vec![0f32; c.head_dim],
            capture_slots: vec![None; c.layers],
            captures: Vec::new(),
            w,
        }
    }

    /// Enable layer-input capture for the given 0-based layer ids. The capture
    /// destination is ordered by the request, and every forward position then
    /// appends its hidden row to that slot.
    pub fn enable_capture(&mut self, ids: &[u32], hidden: usize) -> Result<(), String> {
        for (k, &lid) in ids.iter().enumerate() {
            let li = lid as usize;
            if li >= self.w.cfg.layers {
                return Err(format!(
                    "capture layer {lid} is outside the {} layers",
                    self.w.cfg.layers
                ));
            }
            if self.capture_slots[li].is_some() {
                return Err(format!("capture layer {lid} is listed twice"));
            }
            self.capture_slots[li] = Some(k);
        }
        self.captures = (0..ids.len())
            .map(|_| Vec::with_capacity(self.max_ctx * hidden))
            .collect();
        Ok(())
    }

    /// The captured hidden rows for slot k, position-major.
    pub fn captured(&self, k: usize) -> &[f32] {
        &self.captures[k]
    }

    /// The target embedding row for `id`, the tied input/output matrix row the
    /// draft block reads for its anchor and mask tokens.
    pub fn embed_row(&self, id: u32) -> &[f32] {
        self.w.embed.row(id as usize)
    }

    /// Truncate the KV cache and the layer-input captures to the first `pos`
    /// positions. Speculative verification processes a whole draft block and
    /// then keeps only the rows the target accepted, so the rejected rows have
    /// to leave the cache before the next round
    /// (examples/speculative-simple/speculative-simple.cpp:330: seq_rm(n_past)).
    pub fn rollback(&mut self, pos: usize) -> Result<(), String> {
        if pos > self.len {
            return Err(format!(
                "rollback to {pos} exceeds the {} cached positions",
                self.len
            ));
        }
        let kvd = self.w.cfg.kv_dim();
        let hidden = self.w.cfg.hidden;
        for k in self.kv_k.iter_mut() {
            k.truncate(pos * kvd);
        }
        for v in self.kv_v.iter_mut() {
            v.truncate(pos * kvd);
        }
        for c in self.captures.iter_mut() {
            c.truncate(pos * hidden);
        }
        self.len = pos;
        self.cap = pos;
        Ok(())
    }

    pub fn reset(&mut self) {
        self.len = 0;
        for k in self.kv_k.iter_mut() {
            k.clear();
        }
        for v in self.kv_v.iter_mut() {
            v.clear();
        }
        for c in self.captures.iter_mut() {
            c.clear();
        }
        self.cap = 0;
    }

    fn grow(&mut self) {
        let want = (self.len + 1).min(self.max_ctx);
        if want <= self.cap {
            return;
        }
        let new_cap = (self.cap * 2).max(64).min(self.max_ctx);
        let kv = self.w.cfg.kv_dim();
        for k in self.kv_k.iter_mut() {
            k.resize(new_cap * kv, 0.0);
        }
        for v in self.kv_v.iter_mut() {
            v.resize(new_cap * kv, 0.0);
        }
        self.cap = new_cap;
    }

    /// The rope table for one position and layer, over the first
    /// head_dim*partial_rotary_factor elements of each head.
    fn rope_table(&mut self, pos: usize, layer: usize) {
        let cfg = &self.w.cfg;
        let r = cfg.rope(layer);
        let dim = ((cfg.head_dim as f32 * r.partial) as usize).min(cfg.head_dim);
        let half = dim / 2;
        for i in 0..half {
            let freq = 1.0 / r.theta.powf((2 * i) as f32 / dim as f32);
            let a = pos as f32 * freq;
            self.cos[i] = a.cos();
            self.cos[i + half] = a.cos();
            self.sin[i] = a.sin();
            self.sin[i + half] = a.sin();
        }
    }

    fn apply_rope(x: &mut [f32], cos: &[f32], sin: &[f32], dim: usize) {
        let half = dim / 2;
        for i in 0..half {
            let a = x[i];
            let b = x[i + half];
            x[i] = a * cos[i] - b * sin[i];
            x[i + half] = b * cos[i + half] + a * sin[i + half];
        }
    }

    /// Run one token at `pos` and return the logits of its next-token row.
    pub fn forward(&mut self, token: u32, pos: usize) -> Result<&[f32], String> {
        let cfg = self.w.cfg.clone();
        if token as usize >= cfg.vocab {
            return Err(format!("token {token} is outside the vocabulary"));
        }
        if pos >= self.max_ctx {
            return Err(format!(
                "position {pos} exceeds the {}-token cache",
                self.max_ctx
            ));
        }
        self.len = pos + 1;
        self.grow();

        let hidden = cfg.hidden;
        let qd = cfg.q_dim();
        let kvd = cfg.kv_dim();
        let hd = cfg.head_dim;
        let scale = 1.0 / (hd as f32).sqrt();
        let win = cfg.sliding_window;

        for i in 0..hidden {
            self.scratch.h[i] = self.w.embed.data[token as usize * hidden + i];
        }

        for li in 0..cfg.layers {
            // the layer's input is the residual entering it, exactly the
            // t_layer_inp[il] = inpL row the reference trunk graph exports
            if let Some(slot) = self.capture_slots[li] {
                self.captures[slot].extend_from_slice(&self.scratch.h);
            }
            self.rope_table(pos, li);
            let layer = &self.w.layers[li];
            rms_norm(
                &self.scratch.h,
                &layer.in_norm,
                cfg.eps,
                &mut self.scratch.xn,
            );
            matvec(&layer.qkv, &self.scratch.xn, &mut self.scratch.qkv);
            matvec(&layer.attn_gate, &self.scratch.xn, &mut self.scratch.gate);

            {
                let cos = std::mem::take(&mut self.cos);
                let sin = std::mem::take(&mut self.sin);
                let dim = ((hd as f32 * layer.rope.partial) as usize).min(hd);
                for h in 0..cfg.heads {
                    Self::apply_rope(&mut self.scratch.qkv[h * hd..(h + 1) * hd], &cos, &sin, dim);
                }
                for h in 0..cfg.kv_heads {
                    let o = qd + h * hd;
                    Self::apply_rope(&mut self.scratch.qkv[o..o + hd], &cos, &sin, dim);
                }
                self.cos = cos;
                self.sin = sin;
            }

            {
                let (k, v) = (&mut self.kv_k[li], &mut self.kv_v[li]);
                let base = pos * kvd;
                for i in 0..kvd {
                    k[base + i] = self.scratch.qkv[qd + i];
                    v[base + i] = self.scratch.qkv[qd + kvd + i];
                }
            }

            let start = if layer.full {
                0
            } else {
                (pos + 1).saturating_sub(win)
            };
            let n = pos + 1 - start;
            let kc = &self.kv_k[li];
            let vc = &self.kv_v[li];
            let qkv_s = &self.scratch.qkv;
            let gate_s = &self.scratch.gate;

            let heads: Vec<Vec<f32>> = (0..cfg.heads)
                .into_par_iter()
                .map(|h| {
                    let kvh = h * cfg.kv_heads / cfg.heads;
                    let q = &qkv_s[h * hd..(h + 1) * hd];
                    let mut sc = vec![0f32; n];
                    for j in 0..n {
                        let o = (start + j) * kvd + kvh * hd;
                        sc[j] = dot(q, &kc[o..o + hd]) * scale;
                    }
                    let mut mx = f32::NEG_INFINITY;
                    for s in &sc {
                        mx = mx.max(*s);
                    }
                    let mut sum = 0f32;
                    for s in sc.iter_mut() {
                        *s = (*s - mx).exp();
                        sum += *s;
                    }
                    let inv = 1.0 / sum;
                    let mut o = vec![0f32; hd];
                    for j in 0..n {
                        let p = sc[j] * inv;
                        let base = (start + j) * kvd + kvh * hd;
                        for d in 0..hd {
                            o[d] += p * vc[base + d];
                        }
                    }
                    let g = sigmoid(gate_s[h]);
                    for d in o.iter_mut() {
                        *d *= g;
                    }
                    o
                })
                .collect();

            for (h, o) in heads.iter().enumerate() {
                self.scratch.attn[h * hd..(h + 1) * hd].copy_from_slice(o);
            }

            matvec(&layer.out, &self.scratch.attn, &mut self.scratch.o);
            for i in 0..hidden {
                self.scratch.h[i] += self.scratch.o[i];
            }

            rms_norm(
                &self.scratch.h,
                &layer.post_norm,
                cfg.eps,
                &mut self.scratch.xn,
            );
            matvec(&layer.gate, &self.scratch.xn, &mut self.scratch.g);
            matvec(&layer.up, &self.scratch.xn, &mut self.scratch.u);
            for i in 0..cfg.intermediate {
                self.scratch.act[i] = gelu(self.scratch.g[i]) * self.scratch.u[i];
            }
            matvec(&layer.down, &self.scratch.act, &mut self.scratch.o);
            for i in 0..hidden {
                self.scratch.h[i] += self.scratch.o[i];
            }
        }

        {
            let n = &self.w.norm;
            rms_norm(&self.scratch.h, n, cfg.eps, &mut self.scratch.xn);
        }
        /* Tied embeddings: logits = embedding @ hidden. */
        let embed = &self.w.embed;
        self.logits.par_iter_mut().enumerate().for_each(|(r, o)| {
            *o = dot(embed.row(r), &self.scratch.xn);
        });
        Ok(&self.logits)
    }

    /// The single highest-scoring token.
    pub fn argmax(logits: &[f32]) -> u32 {
        let mut best = 0usize;
        let mut bv = f32::NEG_INFINITY;
        for (i, v) in logits.iter().enumerate() {
            if *v > bv {
                bv = *v;
                best = i;
            }
        }
        best as u32
    }

    /// The n highest scores, sorted, for the gate and for a look at the
    /// distribution.
    pub fn top(logits: &[f32], n: usize) -> Vec<(u32, f32)> {
        let mut idx: Vec<usize> = (0..logits.len()).collect();
        idx.sort_unstable_by(|a, b| logits[*b].partial_cmp(&logits[*a]).unwrap());
        idx.into_iter()
            .take(n)
            .map(|i| (i as u32, logits[i]))
            .collect()
    }
}
