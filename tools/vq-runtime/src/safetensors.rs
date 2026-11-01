//! Minimal safetensors reader for the bf16 checkpoint.
//!
//! The file layout (safetensors spec): [u64 header length][header JSON][data].
//! Every tensor's header entry gives dtype, shape and a byte range relative to
//! the data buffer. This reader only needs BF16 (what this checkpoint ships),
//! plus F16 and F32 so a converted copy still loads. Tensors are read one at a
//! time with seek+read: a 1.6 GB shard is never held in memory beside the f32
//! copy of everything already loaded.

use serde_json::Value;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

pub struct TensorData {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

struct Meta {
    dtype: String,
    shape: Vec<usize>,
    begin: u64,
    end: u64,
}

struct Shard {
    file: File,
    start: u64,
    meta: HashMap<String, Meta>,
}

pub struct Store {
    dir: PathBuf,
    map: HashMap<String, String>,
    shards: HashMap<String, Shard>,
}

fn parse_header(path: &Path) -> Result<Shard, String> {
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
    let start = 8 + n as u64;
    Ok(Shard { file, start, meta })
}

impl Store {
    pub fn open(dir: &Path) -> Result<Store, String> {
        let index = dir.join("model.safetensors.index.json");
        let mut map = HashMap::new();
        if index.exists() {
            let text =
                std::fs::read_to_string(&index).map_err(|e| format!("{}: {e}", index.display()))?;
            let v: Value =
                serde_json::from_str(&text).map_err(|e| format!("{}: {e}", index.display()))?;
            let wm = v
                .get("weight_map")
                .and_then(Value::as_object)
                .ok_or("index has no weight_map")?;
            for (name, f) in wm {
                map.insert(
                    name.clone(),
                    f.as_str()
                        .ok_or("weight_map value is not a path")?
                        .to_string(),
                );
            }
        } else {
            /* A single-file checkpoint: every name resolves to that file, and
             * the header decides whether the name exists. */
            let single = dir.join("model.safetensors");
            if !single.exists() {
                return Err(format!(
                    "{}: no model.safetensors(.index.json)",
                    dir.display()
                ));
            }
            map.insert(String::new(), "model.safetensors".into());
        }
        Ok(Store {
            dir: dir.to_path_buf(),
            map,
            shards: HashMap::new(),
        })
    }

    fn shard(&mut self, file: &str) -> Result<&mut Shard, String> {
        if !self.shards.contains_key(file) {
            let p = self.dir.join(file);
            let s = parse_header(&p)?;
            self.shards.insert(file.to_string(), s);
        }
        Ok(self.shards.get_mut(file).expect("just inserted"))
    }

    fn find(&self, name: &str) -> Result<&str, String> {
        if let Some(f) = self.map.get(name) {
            return Ok(f);
        }
        if self.map.contains_key("") {
            return Ok("model.safetensors");
        }
        Err(format!("{name} is not in the checkpoint index"))
    }

    pub fn read(&mut self, name: &str) -> Result<TensorData, String> {
        let file = self.find(name)?.to_string();
        let shard = self.shard(&file)?;
        let meta = shard
            .meta
            .get(name)
            .ok_or_else(|| format!("{name} is not in {file}'s header"))?;
        let elems: usize = meta.shape.iter().product();
        let per = match meta.dtype.as_str() {
            "BF16" | "F16" => 2usize,
            "F32" => 4usize,
            other => return Err(format!("{name}: dtype {other} is not supported")),
        };
        let bytes = meta.end.saturating_sub(meta.begin) as usize;
        if bytes != elems * per {
            return Err(format!(
                "{name}: header says {bytes} bytes, shape needs {}",
                elems * per
            ));
        }
        shard
            .file
            .seek(SeekFrom::Start(shard.start + meta.begin))
            .map_err(|e| format!("{name}: {e}"))?;
        let mut raw = vec![0u8; bytes];
        shard
            .file
            .read_exact(&mut raw)
            .map_err(|e| format!("{name}: {e}"))?;

        let mut data = Vec::with_capacity(elems);
        match meta.dtype.as_str() {
            "BF16" => {
                for w in raw.chunks_exact(2) {
                    let h = u16::from_le_bytes([w[0], w[1]]);
                    data.push(f32::from_bits(u32::from(h) << 16));
                }
            }
            "F16" => {
                for w in raw.chunks_exact(2) {
                    data.push(vq_format::f16(u16::from_le_bytes([w[0], w[1]])));
                }
            }
            _ => {
                for w in raw.chunks_exact(4) {
                    data.push(f32::from_le_bytes([w[0], w[1], w[2], w[3]]));
                }
            }
        }
        Ok(TensorData {
            shape: meta.shape.clone(),
            data,
        })
    }
}
