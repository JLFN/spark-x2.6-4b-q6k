//! The target's shared token embedding, read from the bf16 checkpoint's
//! safetensors shards. The draft decoder and its LM head both borrow this
//! tensor (src/models/dflash.cpp:981 and 1130: tok_embd/output from ctx_other),
//! so a draft executor needs the target's embedding rows and the tied output
//! matrix, not a second copy of them.

use serde_json::Value;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

pub struct Embedding {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

impl Embedding {
    pub fn row(&self, r: usize) -> &[f32] {
        &self.data[r * self.cols..(r + 1) * self.cols]
    }
}

struct Meta {
    dtype: String,
    shape: Vec<usize>,
    begin: u64,
    end: u64,
}

fn parse_header(path: &Path) -> Result<(File, u64, HashMap<String, Meta>), String> {
    let mut file = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut len = [0u8; 8];
    file.read_exact(&mut len)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    let n = u64::from_le_bytes(len) as usize;
    if n == 0 || n > 100_000_000 {
        return Err(format!("{}: implausible header length {n}", path.display()));
    }
    let mut buf = vec![0u8; n];
    file.read_exact(&mut buf)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    let v: Value = serde_json::from_slice(&buf)
        .map_err(|e| format!("{}: header is not JSON: {e}", path.display()))?;
    let obj = v.as_object().ok_or("safetensors header is not an object")?;
    let mut meta = HashMap::new();
    for (name, e) in obj {
        if name == "__metadata__" {
            continue;
        }
        let dtype = e
            .get("dtype")
            .and_then(Value::as_str)
            .ok_or("tensor without dtype")?
            .to_string();
        let shape = e
            .get("shape")
            .and_then(Value::as_array)
            .ok_or("tensor without shape")?
            .iter()
            .map(|x| x.as_u64().unwrap_or(0) as usize)
            .collect::<Vec<_>>();
        let off = e
            .get("data_offsets")
            .and_then(Value::as_array)
            .ok_or("tensor without data_offsets")?;
        if off.len() != 2 {
            return Err("data_offsets is not a pair".into());
        }
        meta.insert(
            name.clone(),
            Meta {
                dtype,
                shape,
                begin: off[0].as_u64().unwrap_or(0),
                end: off[1].as_u64().unwrap_or(0),
            },
        );
    }
    Ok((file, 8 + n as u64, meta))
}

fn find_file(dir: &Path, name: &str) -> Result<String, String> {
    let index = dir.join("model.safetensors.index.json");
    if index.exists() {
        let text =
            std::fs::read_to_string(&index).map_err(|e| format!("{}: {e}", index.display()))?;
        let v: Value =
            serde_json::from_str(&text).map_err(|e| format!("{}: {e}", index.display()))?;
        let wm = v
            .get("weight_map")
            .and_then(Value::as_object)
            .ok_or("index has no weight_map")?;
        if let Some(f) = wm.get(name).and_then(Value::as_str) {
            return Ok(f.to_string());
        }
        return Err(format!("{name} is not in the checkpoint index"));
    }
    let single = dir.join("model.safetensors");
    if !single.exists() {
        return Err(format!(
            "{}: no model.safetensors(.index.json)",
            dir.display()
        ));
    }
    Ok("model.safetensors".to_string())
}

/// Read the tied embedding/output matrix. Its shape is [vocab, hidden] in the
/// checkpoint, so rows = vocab and cols = hidden, exactly the ggml weight
/// orientation the reference's build_lora_mm reads.
pub fn open(dir: &Path, name: &str) -> Result<Embedding, String> {
    let file = find_file(dir, name)?;
    let (mut f, start, meta) = parse_header(&dir.join(&file))?;
    let m = meta
        .get(name)
        .ok_or_else(|| format!("{name} is not in {file}'s header"))?;
    if m.shape.len() != 2 {
        return Err(format!("{name}: shape {:?} is not a matrix", m.shape));
    }
    let rows = m.shape[0];
    let cols = m.shape[1];
    let elems = rows * cols;
    let per = match m.dtype.as_str() {
        "BF16" | "F16" => 2usize,
        "F32" => 4usize,
        other => return Err(format!("{name}: dtype {other} is not supported")),
    };
    let bytes = m.end.saturating_sub(m.begin) as usize;
    if bytes != elems * per {
        return Err(format!(
            "{name}: header says {bytes} bytes for {elems} elements"
        ));
    }
    f.seek(SeekFrom::Start(start + m.begin))
        .map_err(|e| format!("{name}: {e}"))?;
    let mut raw = vec![0u8; bytes];
    f.read_exact(&mut raw).map_err(|e| format!("{name}: {e}"))?;
    let mut data = Vec::with_capacity(elems);
    match m.dtype.as_str() {
        "BF16" => {
            // ggml_compute_bf16_to_fp32: shift the 16 BF16 bits into the high
            // half of the f32 word (ggml-impl.h:606).
            for w in raw.chunks_exact(2) {
                let bits = ((u32::from(w[1]) << 8) + u32::from(w[0])) << 16;
                data.push(f32::from_bits(bits));
            }
        }
        "F16" => {
            for w in raw.chunks_exact(2) {
                let h = (u16::from(w[1]) << 8) + u16::from(w[0]);
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
        _ => {
            for w in raw.chunks_exact(4) {
                let u = u32::from(w[0])
                    + (u32::from(w[1]) << 8)
                    + (u32::from(w[2]) << 16)
                    + (u32::from(w[3]) << 24);
                data.push(f32::from_bits(u));
            }
        }
    }
    Ok(Embedding { rows, cols, data })
}
