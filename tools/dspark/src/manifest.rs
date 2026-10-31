//! The draft manifest: the role-indexed description a converted draft carries
//! into the executor.
//!
//! The format is line oriented, one record per line, no quoting:
//!
//!     DRAFT 1
//!     arch dflash
//!     hidden 2560
//!     ...
//!     tensor <role> <layer> <dtype> <ne0> <ne1> <offset> <bytes>
//!     end
//!
//! A tensor line names a role from roles.rs, the block index (0-based, -1 for a
//! global tensor), the storage type (0 F32, 1 F16, 2 BF16), the two GGUF
//! dimensions, the byte offset inside weights.bin and the byte count. Anything
//! quantized is refused at conversion, because no kernel reads it and a manifest
//! that promises one would fail at run time instead of at the door.
//!
//! Two fields go beyond the handoff's first sketch, both because a check needs
//! them: `vocab`, so the Markov head can be validated without a target, and a
//! `conf_bias` role, because conf_proj ships a bias in the reference
//! (src/models/dflash.cpp:149) and an unmapped tensor is an error.

use std::collections::HashSet;

use crate::roles::{self, ROLES};

/// Every tensor region starts on this boundary in weights.bin.
pub const ALIGN: u64 = 64;

pub const DTYPE_F32: u8 = 0;
pub const DTYPE_F16: u8 = 1;
pub const DTYPE_BF16: u8 = 2;

pub fn dtype_bytes(dtype: u8) -> Option<u64> {
    match dtype {
        DTYPE_F32 => Some(4),
        DTYPE_F16 | DTYPE_BF16 => Some(2),
        _ => None,
    }
}

pub fn dtype_name(dtype: u8) -> &'static str {
    match dtype {
        DTYPE_F32 => "F32",
        DTYPE_F16 => "F16",
        DTYPE_BF16 => "BF16",
        _ => "unknown",
    }
}

/// How a role's shape follows from the header.
#[derive(Clone, Copy, Debug)]
pub enum Shape {
    /// [hidden] (a norm or a 1-D scale)
    Norm,
    /// [1]
    Scalar,
    /// [hidden, heads*head_dim]
    Q,
    /// [hidden, kv_heads*head_dim]
    Kv,
    /// [heads*head_dim, hidden]
    Out,
    /// [hidden, heads]
    Heads,
    /// [head_dim]
    HeadDim,
    /// [hidden, ffn]
    FfnGate,
    /// [ffn, hidden]
    FfnDown,
    /// [hidden, len(target_layers)*hidden]
    Fc,
    /// [hidden, len(target_layers)]
    FcNorm,
    /// [rank, vocab]
    Markov,
    /// [hidden+rank, 1]
    ConfProj,
}

#[derive(Clone, Debug)]
pub struct Header {
    pub arch: String,
    pub hidden: u64,
    pub layers: u32,
    pub ffn: u64,
    pub heads: u32,
    pub kv_heads: u32,
    pub head_dim: u64,
    pub vocab: u64,
    pub rope_full_dims: u64,
    pub rope_swa_dims: u64,
    pub rope_full_theta: f32,
    pub rope_swa_theta: f32,
    pub eps: f32,
    pub act: String,
    pub swa_window: u64,
    pub swa_pattern: Vec<bool>,
    pub causal: bool,
    pub block_size: u32,
    pub sample_from_anchor: bool,
    pub target_layers: Vec<u32>,
}

#[derive(Clone, Debug)]
pub struct Tensor {
    pub role: String,
    pub layer: i32,
    pub dtype: u8,
    pub ne0: u64,
    pub ne1: u64,
    pub offset: u64,
    pub bytes: u64,
}

impl Tensor {
    pub fn label(&self) -> String {
        if self.layer >= 0 {
            format!("{} {}", self.role, self.layer)
        } else {
            self.role.clone()
        }
    }

    pub fn elems(&self) -> u64 {
        self.ne0 * self.ne1
    }

    pub fn end(&self) -> u64 {
        self.offset + self.bytes
    }
}

#[derive(Clone, Debug)]
pub struct Manifest {
    pub header: Header,
    pub tensors: Vec<Tensor>,
}

impl Header {
    /// Rules that only involve the header itself.
    pub fn check(&self) -> Result<(), String> {
        if self.hidden == 0 || self.ffn == 0 || self.head_dim == 0 {
            return Err("hidden, ffn and head_dim must be non-zero".into());
        }
        if self.layers == 0 || self.heads == 0 || self.kv_heads == 0 {
            return Err("layers, heads and kv_heads must be non-zero".into());
        }
        if !self.heads.is_multiple_of(self.kv_heads) {
            return Err(format!(
                "heads {} is not a multiple of kv_heads {}",
                self.heads, self.kv_heads
            ));
        }
        if self.vocab == 0 {
            return Err("vocab must be non-zero".into());
        }
        for (name, d) in [
            ("rope_full_dims", self.rope_full_dims),
            ("rope_swa_dims", self.rope_swa_dims),
        ] {
            if d == 0 || !d.is_multiple_of(2) {
                return Err(format!("{name} {d} must be a non-zero even number"));
            }
            if d > self.head_dim {
                return Err(format!("{name} {d} exceeds head_dim {}", self.head_dim));
            }
        }
        if self.rope_full_theta <= 0.0 || self.rope_swa_theta <= 0.0 {
            return Err("rope theta must be positive".into());
        }
        if self.eps <= 0.0 {
            return Err("eps must be positive".into());
        }
        if !matches!(self.act.as_str(), "gelu" | "gelu_pytorch_tanh" | "silu") {
            return Err(format!(
                "activation {} is not one the reference implements (gelu, gelu_pytorch_tanh, silu)",
                self.act
            ));
        }
        if self.swa_window == 0 {
            return Err("swa_window must be non-zero".into());
        }
        if self.swa_pattern.len() != self.layers as usize {
            return Err(format!(
                "swa_pattern has {} entries for {} layers",
                self.swa_pattern.len(),
                self.layers
            ));
        }
        if self.block_size < 1 {
            return Err("block_size must be at least 1".into());
        }
        if self.target_layers.is_empty() {
            return Err("target_layers must name at least one target layer".into());
        }
        let mut seen = HashSet::new();
        for &l in &self.target_layers {
            if !seen.insert(l) {
                return Err(format!("target layer {l} is listed twice"));
            }
        }
        Ok(())
    }
}

impl Manifest {
    pub fn find(&self, role: &str, layer: i32) -> Option<&Tensor> {
        self.tensors
            .iter()
            .find(|t| t.role == role && t.layer == layer)
    }

    pub fn markov_rank(&self) -> Option<u64> {
        self.find("markov_w1", -1).map(|t| t.ne0)
    }

    /// Validate the description and the layout it promises for weights.bin.
    /// `weights_len` is the file's length; the regions must end exactly there.
    pub fn validate(&self, weights_len: u64) -> Result<(), String> {
        let h = &self.header;
        h.check()?;
        if self.tensors.is_empty() {
            return Err("the manifest lists no tensors".into());
        }

        let mut seen = HashSet::new();
        for t in &self.tensors {
            let Some(spec) = roles::spec(&t.role) else {
                return Err(format!(
                    "tensor {}: role {} is not in this build's vocabulary",
                    t.label(),
                    t.role
                ));
            };
            if spec.per_layer {
                if t.layer < 0 || t.layer as u32 >= h.layers {
                    return Err(format!(
                        "{}: layer index {} is outside 0..{}",
                        t.label(),
                        t.layer,
                        h.layers
                    ));
                }
            } else if t.layer != -1 {
                return Err(format!(
                    "{}: a global role must carry layer -1, not {}",
                    t.label(),
                    t.layer
                ));
            }
            if !seen.insert((t.role.clone(), t.layer)) {
                return Err(format!("tensor {} appears more than once", t.label()));
            }
            if t.ne0 == 0 || t.ne1 == 0 {
                return Err(format!("tensor {}: a zero dimension", t.label()));
            }
            let Some(elem) = dtype_bytes(t.dtype) else {
                return Err(format!(
                    "tensor {}: dtype code {} is not F32, F16 or BF16",
                    t.label(),
                    t.dtype
                ));
            };
            let want = t
                .elems()
                .checked_mul(elem)
                .ok_or_else(|| format!("tensor {}: byte count overflows", t.label()))?;
            if t.bytes != want {
                return Err(format!(
                    "tensor {}: {} bytes recorded, but {} {}x{} is {} bytes",
                    t.label(),
                    t.bytes,
                    dtype_name(t.dtype),
                    t.ne0,
                    t.ne1,
                    want
                ));
            }
        }

        // Required roles, per layer and global.
        for spec in ROLES {
            if !spec.required {
                continue;
            }
            if spec.per_layer {
                for l in 0..h.layers as i32 {
                    if self.find(spec.name, l).is_none() {
                        return Err(format!("layer {l} has no {} tensor", spec.name));
                    }
                }
            } else if self.find(spec.name, -1).is_none() {
                return Err(format!("the manifest has no {} tensor", spec.name));
            }
        }

        // The reference picks one gate family for the whole draft
        // (src/models/dflash.cpp:325-333), and borrows V only uniformly.
        self.uniform("attn_v")?;
        self.uniform("attn_gate")?;
        self.uniform("attn_q_norm")?;
        self.uniform("attn_k_norm")?;
        let gate = self.count("attn_gate");
        let qkn = self.count("attn_q_norm") + self.count("attn_k_norm");
        if gate > 0 && qkn > 0 {
            return Err(
                "attn_gate and the q/k norms are alternatives in the reference; a draft must have one or the other"
                    .into(),
            );
        }
        if self.count("attn_q_norm") != self.count("attn_k_norm") {
            return Err("attn_q_norm and attn_k_norm must be present together".into());
        }

        // Heads that exist only as a group.
        let markov = [self.find("markov_w1", -1), self.find("markov_w2", -1)];
        if markov.iter().any(|t| t.is_some()) && markov.iter().any(|t| t.is_none()) {
            return Err("markov_w1 and markov_w2 must be present together".into());
        }
        let conf = [self.find("conf_proj", -1), self.find("conf_bias", -1)];
        if conf.iter().any(|t| t.is_some()) && conf.iter().any(|t| t.is_none()) {
            return Err("conf_proj and conf_bias must be present together".into());
        }
        if conf[0].is_some() && markov[0].is_none() {
            return Err(
                "the confidence head reads markov_w1, so conf_proj needs the Markov head".into(),
            );
        }

        let rank = self.markov_rank();
        for t in &self.tensors {
            let spec = roles::spec(&t.role).expect("role was checked above");
            let (ne0, ne1) = roles::expected(spec.shape, h, rank)?;
            if t.ne0 != ne0 || t.ne1 != ne1 {
                return Err(format!(
                    "tensor {}: shape {}x{}, the header requires {}x{}",
                    t.label(),
                    t.ne0,
                    t.ne1,
                    ne0,
                    ne1
                ));
            }
        }

        self.validate_layout(weights_len)
    }

    /// Regions must start at 0, be 64-byte aligned, not overlap, leave only
    /// alignment padding between them, and end exactly at the file's end.
    fn validate_layout(&self, weights_len: u64) -> Result<(), String> {
        let mut order: Vec<&Tensor> = self.tensors.iter().collect();
        order.sort_by_key(|t| t.offset);
        let mut end = 0u64;
        for (k, t) in order.iter().enumerate() {
            if k == 0 && t.offset != 0 {
                return Err(format!(
                    "tensor {} starts at byte {}, not 0",
                    t.label(),
                    t.offset
                ));
            }
            if t.offset % ALIGN != 0 {
                return Err(format!(
                    "tensor {} starts at byte {}, not a multiple of {ALIGN}",
                    t.label(),
                    t.offset
                ));
            }
            if t.offset < end {
                return Err(format!(
                    "tensor {} starts at byte {}, inside the previous region which ends at {end}",
                    t.label(),
                    t.offset
                ));
            }
            if t.offset - end >= ALIGN {
                return Err(format!(
                    "tensor {} starts at byte {}, but the previous region ends at {end}: a {}-byte gap cannot be alignment padding",
                    t.label(),
                    t.offset,
                    t.offset - end
                ));
            }
            end = t.end();
        }
        if end != weights_len {
            return Err(format!(
                "the last tensor {} ends at byte {end}, but weights.bin is {weights_len} bytes",
                order.last().map(|t| t.label()).unwrap_or_default()
            ));
        }
        Ok(())
    }

    fn count(&self, role: &str) -> u32 {
        self.tensors.iter().filter(|t| t.role == role).count() as u32
    }

    fn uniform(&self, role: &str) -> Result<(), String> {
        let n = self.count(role);
        if n != 0 && n != self.header.layers {
            return Err(format!(
                "{role} exists on {n} of {} layers; the reference creates it on all or none",
                self.header.layers
            ));
        }
        Ok(())
    }

    pub fn render(&self) -> String {
        let h = &self.header;
        let mut s = String::new();
        s.push_str("DRAFT 1\n");
        s.push_str(&format!("arch {}\n", h.arch));
        s.push_str(&format!("hidden {}\n", h.hidden));
        s.push_str(&format!("layers {}\n", h.layers));
        s.push_str(&format!("ffn {}\n", h.ffn));
        s.push_str(&format!("heads {}\n", h.heads));
        s.push_str(&format!("kv_heads {}\n", h.kv_heads));
        s.push_str(&format!("head_dim {}\n", h.head_dim));
        s.push_str(&format!("vocab {}\n", h.vocab));
        s.push_str(&format!("rope_full_dims {}\n", h.rope_full_dims));
        s.push_str(&format!("rope_swa_dims {}\n", h.rope_swa_dims));
        s.push_str(&format!("rope_full_theta {}\n", h.rope_full_theta));
        s.push_str(&format!("rope_swa_theta {}\n", h.rope_swa_theta));
        s.push_str(&format!("eps {}\n", h.eps));
        s.push_str(&format!("act {}\n", h.act));
        s.push_str(&format!("swa_window {}\n", h.swa_window));
        let pat: Vec<&str> = h
            .swa_pattern
            .iter()
            .map(|&b| if b { "1" } else { "0" })
            .collect();
        s.push_str(&format!("swa_pattern {}\n", pat.join(",")));
        s.push_str(&format!("causal {}\n", if h.causal { 1 } else { 0 }));
        s.push_str(&format!("block_size {}\n", h.block_size));
        s.push_str(&format!(
            "sample_from_anchor {}\n",
            if h.sample_from_anchor { 1 } else { 0 }
        ));
        let tl: Vec<String> = h.target_layers.iter().map(|l| l.to_string()).collect();
        s.push_str(&format!("target_layers {}\n", tl.join(",")));
        for t in &self.tensors {
            s.push_str(&format!(
                "tensor {} {} {} {} {} {} {}\n",
                t.role, t.layer, t.dtype, t.ne0, t.ne1, t.offset, t.bytes
            ));
        }
        s.push_str("end\n");
        s
    }

    pub fn parse(text: &str) -> Result<Manifest, String> {
        let mut lines = text
            .lines()
            .enumerate()
            .filter(|(_, l)| !l.trim().is_empty());
        let (_, first) = lines.next().ok_or("manifest is empty")?;
        if first.trim() != "DRAFT 1" {
            return Err(format!(
                "line 1: expected 'DRAFT 1', found '{}'",
                first.trim()
            ));
        }

        // Values as written, converted after the loop when every key is known.
        let mut kv: Vec<(String, usize, Vec<String>)> = Vec::new();
        let mut tensors = Vec::new();
        let mut ended = false;
        for (n, raw) in lines {
            let line = raw.trim();
            let f: Vec<&str> = line.split_whitespace().collect();
            let at = n + 1;
            if f[0] == "end" {
                if f.len() != 1 {
                    return Err(format!("line {at}: 'end' takes no argument"));
                }
                ended = true;
                break;
            }
            if f[0] == "tensor" {
                if f.len() != 8 {
                    return Err(format!(
                        "line {at}: a tensor line needs 7 fields after 'tensor' (role layer dtype ne0 ne1 offset bytes), found {}",
                        f.len() - 1
                    ));
                }
                let layer: i32 = f[2]
                    .parse()
                    .map_err(|_| format!("line {at}: layer '{}' is not an integer", f[2]))?;
                let dtype: u8 = f[3]
                    .parse()
                    .map_err(|_| format!("line {at}: dtype '{}' is not an integer", f[3]))?;
                let mut nums = [0u64; 4];
                for (i, tok) in [f[4], f[5], f[6], f[7]].iter().enumerate() {
                    nums[i] = tok
                        .parse()
                        .map_err(|_| format!("line {at}: '{}' is not a byte count", tok))?;
                }
                tensors.push(Tensor {
                    role: f[1].to_string(),
                    layer,
                    dtype,
                    ne0: nums[0],
                    ne1: nums[1],
                    offset: nums[2],
                    bytes: nums[3],
                });
                continue;
            }
            let key = f[0].to_string();
            if f.len() < 2 {
                return Err(format!("line {at}: key '{key}' has no value"));
            }
            if kv.iter().any(|(k, _, _)| *k == key) {
                return Err(format!("line {at}: key '{key}' appears twice"));
            }
            kv.push((key, at, f[1..].iter().map(|s| s.to_string()).collect()));
        }
        if !ended {
            return Err("the manifest has no 'end' line".into());
        }

        let get = |name: &str| -> Result<(usize, Vec<String>), String> {
            kv.iter()
                .find(|(k, _, _)| k == name)
                .map(|(_, at, v)| (*at, v.clone()))
                .ok_or_else(|| format!("the manifest has no '{name}' line"))
        };
        let text_val = |name: &str| -> Result<String, String> {
            let (at, v) = get(name)?;
            if v.len() != 1 {
                return Err(format!("line {at}: '{name}' takes one value"));
            }
            Ok(v[0].clone())
        };
        let num_val = |name: &str| -> Result<u64, String> {
            let (at, v) = get(name)?;
            if v.len() != 1 {
                return Err(format!("line {at}: '{name}' takes one value"));
            }
            v[0].parse()
                .map_err(|_| format!("line {at}: '{name}' value '{}' is not a number", v[0]))
        };
        let float_val = |name: &str| -> Result<f32, String> {
            let (at, v) = get(name)?;
            if v.len() != 1 {
                return Err(format!("line {at}: '{name}' takes one value"));
            }
            v[0].parse()
                .map_err(|_| format!("line {at}: '{name}' value '{}' is not a number", v[0]))
        };
        let bit = |name: &str| -> Result<bool, String> {
            match num_val(name)? {
                0 => Ok(false),
                1 => Ok(true),
                other => Err(format!("'{name}' must be 0 or 1, found {other}")),
            }
        };
        let list = |name: &str| -> Result<Vec<String>, String> {
            let (at, v) = get(name)?;
            if v.len() != 1 {
                return Err(format!(
                    "line {at}: '{name}' takes one comma-separated list"
                ));
            }
            Ok(v[0].split(',').map(|s| s.trim().to_string()).collect())
        };

        let u32_val = |name: &str| -> Result<u32, String> {
            let n = num_val(name)?;
            u32::try_from(n).map_err(|_| format!("'{name}' value {n} does not fit 32 bits"))
        };

        let arch = text_val("arch")?;
        let hidden = num_val("hidden")?;
        let layers = u32_val("layers")?;
        let ffn = num_val("ffn")?;
        let heads = u32_val("heads")?;
        let kv_heads = u32_val("kv_heads")?;
        let head_dim = num_val("head_dim")?;
        let vocab = num_val("vocab")?;
        let rope_full_dims = num_val("rope_full_dims")?;
        let rope_swa_dims = num_val("rope_swa_dims")?;
        let rope_full_theta = float_val("rope_full_theta")?;
        let rope_swa_theta = float_val("rope_swa_theta")?;
        let eps = float_val("eps")?;
        let act = text_val("act")?;
        let swa_window = num_val("swa_window")?;
        let swa_pattern = list("swa_pattern")?
            .iter()
            .map(|s| match s.as_str() {
                "1" => Ok(true),
                "0" => Ok(false),
                other => Err(format!("swa_pattern entry '{other}' is not 0 or 1")),
            })
            .collect::<Result<Vec<bool>, String>>()?;
        let causal = bit("causal")?;
        let block_size = u32_val("block_size")?;
        let sample_from_anchor = bit("sample_from_anchor")?;
        let target_layers = list("target_layers")?
            .iter()
            .map(|s| {
                s.parse::<u32>()
                    .map_err(|_| format!("target_layers entry '{s}' is not a layer index"))
            })
            .collect::<Result<Vec<u32>, String>>()?;

        let known: &[&str] = &[
            "arch",
            "hidden",
            "layers",
            "ffn",
            "heads",
            "kv_heads",
            "head_dim",
            "vocab",
            "rope_full_dims",
            "rope_swa_dims",
            "rope_full_theta",
            "rope_swa_theta",
            "eps",
            "act",
            "swa_window",
            "swa_pattern",
            "causal",
            "block_size",
            "sample_from_anchor",
            "target_layers",
        ];
        for (k, at, _) in &kv {
            if !known.contains(&k.as_str()) {
                return Err(format!("line {at}: unknown key '{k}'"));
            }
        }

        Ok(Manifest {
            header: Header {
                arch,
                hidden,
                layers,
                ffn,
                heads,
                kv_heads,
                head_dim,
                vocab,
                rope_full_dims,
                rope_swa_dims,
                rope_full_theta,
                rope_swa_theta,
                eps,
                act,
                swa_window,
                swa_pattern,
                causal,
                block_size,
                sample_from_anchor,
                target_layers,
            },
            tensors,
        })
    }
}
