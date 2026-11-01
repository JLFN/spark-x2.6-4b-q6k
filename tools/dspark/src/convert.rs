//! dspark convert: turn a dflash draft GGUF into a role manifest and a repacked
//! weights.bin.
//!
//! The output is deliberately dumb: the source bytes are copied, not decoded, so
//! conversion cannot change a weight. What the manifest adds is the only thing a
//! generic executor needs and the GGUF does not spell out in one place: which
//! tensor is which role, where it lives, and the architecture the role set
//! implies. Every dimension is read from the file; nothing about the Spark
//! checkpoint is compiled in.

use std::fs::{self, File};
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

use crate::gguf::{self, Gguf, TensorInfo};
use crate::manifest::{
    dtype_bytes, dtype_name, Header, Manifest, Tensor, ALIGN, DTYPE_BF16, DTYPE_F16, DTYPE_F32,
};
use crate::roles::{self, Lookup};

/// Role order inside one block, mirroring the reference's create_tensor order
/// (src/models/dflash.cpp:326-350).
const LAYER_ORDER: &[&str] = &[
    "attn_norm",
    "attn_q",
    "attn_k",
    "attn_v",
    "attn_gate",
    "attn_q_norm",
    "attn_k_norm",
    "attn_out",
    "attn_post_norm",
    "ffn_norm",
    "ffn_gate",
    "ffn_up",
    "ffn_down",
    "ffn_post_norm",
    "layer_out_scale",
];

/// Global order, mirroring src/models/dflash.cpp:137-149 and 239-255.
const GLOBAL_ORDER: &[&str] = &[
    "fc",
    "fc_norm",
    "enc_norm",
    "output_norm",
    "markov_w1",
    "markov_w2",
    "conf_proj",
    "conf_bias",
];

struct Item<'a> {
    info: &'a TensorInfo,
    role: &'static str,
    layer: i32,
    dtype: u8,
    ne0: u64,
    ne1: u64,
    bytes: u64,
    offset: u64,
}

impl Item<'_> {
    fn end(&self) -> u64 {
        self.offset + self.bytes
    }
}

pub fn convert(src: &str, out_dir: &str, force: bool, verify: bool) -> Result<(), String> {
    let g = Gguf::open(src)?;
    let header = header_from_gguf(&g)?;

    let mut items: Vec<Item> = Vec::with_capacity(g.tensors.len());
    for t in &g.tensors {
        let (role, layer) = match roles::classify(&t.name) {
            Lookup::Mapped { role, layer } => (role, layer),
            Lookup::Unimplemented(why) => {
                return Err(format!("tensor {}: {why}", t.name));
            }
            Lookup::Unknown => {
                return Err(format!(
                    "tensor {}: no role in this build's vocabulary; refusing to ignore it",
                    t.name
                ));
            }
        };
        let dtype = match t.dtype {
            0 => DTYPE_F32,
            1 => DTYPE_F16,
            30 => DTYPE_BF16,
            other => {
                return Err(format!(
                    "tensor {}: {} weights are not implemented; only F32, F16 and BF16 convert",
                    t.name,
                    gguf::type_name(other)
                ));
            }
        };
        if t.dims.len() > 2 {
            return Err(format!(
                "tensor {}: {} dimensions; this build carries one or two",
                t.name,
                t.dims.len()
            ));
        }
        let ne0 = t.dims[0];
        let ne1 = t.dims.get(1).copied().unwrap_or(1);
        let bytes = dtype_bytes(dtype).expect("dtype came from the table above") * t.ne();
        let source =
            gguf::type_size(t.dtype, t.ne()).map_err(|e| format!("tensor {}: {e}", t.name))?;
        if bytes != source {
            return Err(format!(
                "tensor {}: {} bytes of {} but the source says {source}",
                t.name,
                bytes,
                dtype_name(dtype)
            ));
        }
        items.push(Item {
            info: t,
            role,
            layer,
            dtype,
            ne0,
            ne1,
            bytes,
            offset: 0,
        });
    }

    // Layers first, then the globals, each in the reference's order: a stable
    // layout, so two conversions of the same file produce the same manifest.
    let rank_key = |it: &Item| -> (u8, i32, usize) {
        let (group, idx) = if it.layer >= 0 {
            (0u8, LAYER_ORDER.iter().position(|r| *r == it.role))
        } else {
            (1u8, GLOBAL_ORDER.iter().position(|r| *r == it.role))
        };
        (
            group,
            it.layer,
            idx.unwrap_or_else(|| panic!("role {} has no order", it.role)),
        )
    };
    items.sort_by_key(rank_key);

    let mut end = 0u64;
    for it in &mut items {
        it.offset = end.div_ceil(ALIGN) * ALIGN;
        end = it.offset + it.bytes;
    }
    let weights_len = end;

    let manifest = Manifest {
        header,
        tensors: items
            .iter()
            .map(|it| Tensor {
                role: it.role.to_string(),
                layer: it.layer,
                dtype: it.dtype,
                ne0: it.ne0,
                ne1: it.ne1,
                offset: it.offset,
                bytes: it.bytes,
            })
            .collect(),
    };
    manifest.validate(weights_len)?;

    let dir = Path::new(out_dir);
    fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let weights_path = dir.join("weights.bin");
    let manifest_path = dir.join("manifest.txt");
    if !force {
        for p in [&weights_path, &manifest_path] {
            if p.exists() {
                return Err(format!(
                    "{} already exists; pass --force to overwrite",
                    p.display()
                ));
            }
        }
    }

    let weights_tmp = dir.join("weights.bin.tmp");
    let manifest_tmp = dir.join("manifest.txt.tmp");
    let written = (|| -> Result<(), String> {
        let f =
            File::create(&weights_tmp).map_err(|e| format!("{}: {e}", weights_tmp.display()))?;
        let mut w = BufWriter::with_capacity(8 << 20, f);
        let mut pos = 0u64;
        let pad = [0u8; ALIGN as usize];
        for it in &items {
            if it.offset > pos {
                let n = (it.offset - pos) as usize;
                if n >= pad.len() {
                    return Err(format!(
                        "{}: {} bytes of padding is not alignment",
                        it.role, n
                    ));
                }
                w.write_all(&pad[..n]).map_err(|e| e.to_string())?;
                pos = it.offset;
            }
            let bytes = g.tensor_bytes(it.info)?;
            w.write_all(bytes).map_err(|e| e.to_string())?;
            pos += bytes.len() as u64;
        }
        w.flush().map_err(|e| e.to_string())?;
        drop(w);
        fs::rename(&weights_tmp, &weights_path)
            .map_err(|e| format!("{}: {e}", weights_path.display()))?;
        fs::write(&manifest_tmp, manifest.render())
            .map_err(|e| format!("{}: {e}", manifest_tmp.display()))?;
        fs::rename(&manifest_tmp, &manifest_path)
            .map_err(|e| format!("{}: {e}", manifest_path.display()))?;
        Ok(())
    })();
    if written.is_err() {
        let _ = fs::remove_file(&weights_tmp);
        let _ = fs::remove_file(&manifest_tmp);
        return written;
    }
    if verify {
        verify_copy(&weights_path, &items, &g)?;
    }

    let h = &manifest.header;
    let source_bytes: u64 = items.iter().map(|i| i.bytes).sum();
    println!(
        "source      {src} ({} bytes, GGUF v{})",
        g.file_len, g.version
    );
    println!(
        "arch        dflash, {} blocks, {} hidden, {} ffn, {} heads / {} kv of {}",
        h.layers, h.hidden, h.ffn, h.heads, h.kv_heads, h.head_dim
    );
    println!(
        "targets     layers {:?}, {} features per position",
        h.target_layers,
        h.target_layers.len()
    );
    println!("capabilities {}", capabilities(&manifest));
    println!(
        "weights     {} tensors, {} bytes of source weights, {} bytes in weights.bin ({} of alignment)",
        items.len(),
        source_bytes,
        weights_len,
        weights_len - source_bytes
    );
    println!("wrote       {}", weights_path.display());
    println!("            {}", manifest_path.display());
    if verify {
        println!("verify      every tensor byte matches the source and all padding is zero");
    }
    println!(
        "\n  {:<16} {:>5} {:>5} {:>16} {:>12} {:>12}",
        "role", "layer", "dtype", "shape", "offset", "bytes"
    );
    for it in &items {
        println!(
            "  {:<16} {:>5} {:>5} {:>16} {:>12} {:>12}",
            it.role,
            it.layer,
            dtype_name(it.dtype),
            format!("{}x{}", it.ne0, it.ne1),
            it.offset,
            it.bytes
        );
    }
    Ok(())
}

/// Re-read weights.bin and compare every region with the source bytes, and
/// every alignment gap with zero. This is what backs the claim that conversion
/// copies weights instead of decoding them.
fn verify_copy(path: &Path, items: &[Item], g: &Gguf) -> Result<(), String> {
    let mut f = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut buf = vec![0u8; 8 << 20];
    let mut pos = 0u64;
    for it in items {
        let label = if it.layer >= 0 {
            format!("{} {}", it.role, it.layer)
        } else {
            it.role.to_string()
        };
        if it.offset > pos {
            let gap = it.offset - pos;
            f.seek(SeekFrom::Start(pos))
                .map_err(|e| format!("{}: {e}", path.display()))?;
            f.read_exact(&mut buf[..gap as usize])
                .map_err(|e| format!("{}: {e}", path.display()))?;
            if buf[..gap as usize].iter().any(|&x| x != 0) {
                return Err(format!(
                    "the {gap}-byte padding before tensor {label} is not zero"
                ));
            }
        }
        let want = g.tensor_bytes(it.info)?;
        f.seek(SeekFrom::Start(it.offset))
            .map_err(|e| format!("{}: {e}", path.display()))?;
        let mut off = 0usize;
        while off < want.len() {
            let n = (want.len() - off).min(buf.len());
            f.read_exact(&mut buf[..n])
                .map_err(|e| format!("{}: {e}", path.display()))?;
            if buf[..n] != want[off..off + n] {
                return Err(format!(
                    "tensor {label}: weights.bin differs from the source at byte {}",
                    it.offset + off as u64
                ));
            }
            off += n;
        }
        pos = it.end();
    }
    Ok(())
}

fn capabilities(m: &Manifest) -> String {
    let mut c: Vec<String> = Vec::new();
    if m.find("attn_gate", 0).is_some() {
        c.push("per-head attn gate".into());
    }
    if m.find("attn_q_norm", 0).is_some() {
        c.push("q/k norms".into());
    }
    if m.find("markov_w1", -1).is_some() {
        c.push("markov head".into());
    }
    if m.find("conf_proj", -1).is_some() {
        c.push("confidence head".into());
    }
    let swa = m.header.swa_pattern.iter().filter(|&&b| b).count();
    if swa > 0 {
        c.push(format!("{swa}/{} sliding-window layers", m.header.layers));
    }
    if c.is_empty() {
        "none".to_string()
    } else {
        c.join(", ")
    }
}

pub fn capabilities_of(m: &Manifest) -> String {
    capabilities(m)
}

fn num(g: &Gguf, key: &str) -> Result<f64, String> {
    if let Some(v) = g.get_u64(key) {
        return Ok(v as f64);
    }
    if let Some(v) = g.get_f32(key) {
        return Ok(v as f64);
    }
    Err(format!("metadata key {key} is missing or not a number"))
}

fn req_u64(g: &Gguf, key: &str) -> Result<u64, String> {
    let v = num(g, key)?;
    if v < 0.0 || v.fract() != 0.0 {
        return Err(format!("metadata key {key} is not a whole number: {v}"));
    }
    Ok(v as u64)
}

fn req_u32(g: &Gguf, key: &str) -> Result<u32, String> {
    let v = req_u64(g, key)?;
    u32::try_from(v).map_err(|_| format!("metadata key {key} value {v} does not fit 32 bits"))
}

fn header_from_gguf(g: &Gguf) -> Result<Header, String> {
    let arch = g
        .get_str("general.architecture")
        .ok_or("general.architecture is missing")?;
    if arch != "dflash" {
        return Err(format!(
            "architecture {arch} is not implemented; this build converts dflash drafts"
        ));
    }
    let k = |s: &str| format!("dflash.{s}");

    let hidden = req_u64(g, &k("embedding_length"))?;
    let layers = req_u32(g, &k("block_count"))?;
    let ffn = req_u64(g, &k("feed_forward_length"))?;
    let heads = req_u32(g, &k("attention.head_count"))?;
    let kv_heads = req_u32(g, &k("attention.head_count_kv"))?;
    let head_dim = req_u64(g, &k("attention.key_length"))?;
    let value_dim = req_u64(g, &k("attention.value_length"))?;
    if value_dim != head_dim {
        return Err(format!(
            "attention value_length {value_dim} differs from key_length {head_dim}; this build assumes one head dimension"
        ));
    }
    let vocab = req_u64(g, &k("vocab_size"))?;
    // The decoder fills the noise slots with the draft's BLANK token, so the
    // tokenizer's mask id is a runtime requirement, not a detail.
    let mask_token = g.get_u64("tokenizer.ggml.mask_token_id").ok_or_else(|| {
        "tokenizer.ggml.mask_token_id is missing; DFlash block diffusion needs the draft's BLANK token".to_string()
    })?;
    if mask_token >= vocab {
        return Err(format!(
            "mask token {mask_token} is outside the {vocab}-id vocabulary"
        ));
    }
    let rope_full_dims = req_u64(g, &k("rope.dimension_count"))?;
    let rope_swa_dims = req_u64(g, &k("rope.dimension_count_swa"))?;
    let rope_full_theta = num(g, &k("rope.freq_base"))?;
    let rope_swa_theta = num(g, &k("rope.freq_base_swa"))?;
    let eps = num(g, &k("attention.layer_norm_rms_epsilon"))?;
    let act = g
        .get_str(&k("hidden_activation"))
        .ok_or_else(|| format!("metadata key {} is missing", k("hidden_activation")))?
        .to_string();
    let swa_window = req_u64(g, &k("attention.sliding_window"))?;
    let swa_pattern = g
        .get_arr(&k("attention.sliding_window_pattern"))
        .ok_or_else(|| {
            format!(
                "metadata key {} is missing",
                k("attention.sliding_window_pattern")
            )
        })?
        .iter()
        .map(|v| match v {
            gguf::Value::Bool(b) => Ok(*b),
            other => Err(format!(
                "sliding_window_pattern holds {other:?}, not a boolean"
            )),
        })
        .collect::<Result<Vec<bool>, String>>()?;
    // Absent means the reference's default (src/llama-hparams.h:246).
    let causal = match g.kv.get(&k("attention.causal")) {
        Some(gguf::Value::Bool(b)) => *b,
        Some(other) => return Err(format!("attention.causal holds {other:?}, not a boolean")),
        None => true,
    };
    let block_size = req_u32(g, &k("block_size"))?;
    // Absent means true in the reference (src/models/dflash.cpp:424-425).
    let sample_from_anchor = match g.kv.get(&k("sample_from_anchor")) {
        Some(gguf::Value::Bool(b)) => *b,
        Some(other) => return Err(format!("sample_from_anchor holds {other:?}, not a boolean")),
        None => true,
    };
    let target_layers = g
        .get_arr(&k("target_layers"))
        .ok_or_else(|| format!("metadata key {} is missing", k("target_layers")))?
        .iter()
        .map(|v| {
            let n = v
                .as_u64()
                .ok_or_else(|| format!("target_layers holds {v:?}, not a layer index"))?;
            u32::try_from(n).map_err(|_| format!("target layer {n} does not fit 32 bits"))
        })
        .collect::<Result<Vec<u32>, String>>()?;

    Ok(Header {
        arch: arch.to_string(),
        hidden,
        layers,
        ffn,
        heads,
        kv_heads,
        head_dim,
        vocab,
        mask_token,
        rope_full_dims,
        rope_swa_dims,
        rope_full_theta: rope_full_theta as f32,
        rope_swa_theta: rope_swa_theta as f32,
        eps: eps as f32,
        act,
        swa_window,
        swa_pattern,
        causal,
        block_size,
        sample_from_anchor,
        target_layers,
    })
}
