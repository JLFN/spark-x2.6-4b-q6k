//! Reader for an artifact written by build-vq / build-vq-gpu.
//!
//! One `<tensor>.bin` per tensor next to a manifest.txt:
//!
//! ```text
//! # name rows cols stages file_bytes relL2
//! model.layers.0.mlp.down_proj.weight 2560 10240 2 21440592 3.2789
//! ```
//!
//! Two-dimensional tensors decode through the shared vq-format reader; the
//! 1-D norms are RQVF raw f32 sidecars (stages 0 in the manifest). The manifest
//! is the index: a file that is missing, or whose header disagrees with the
//! manifest, is an error and not a silent zero tensor.

use crate::model::Tensor;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

struct Entry {
    rows: usize,
    cols: usize,
    stages: usize,
}

pub struct VqDir {
    dir: PathBuf,
    entries: HashMap<String, Entry>,
}

impl VqDir {
    pub fn open(dir: &Path) -> Result<VqDir, String> {
        let path = dir.join("manifest.txt");
        let text = fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        let mut entries = HashMap::new();
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let f: Vec<&str> = line.split_whitespace().collect();
            if f.len() < 5 {
                return Err(format!("{}: short manifest line: {line}", path.display()));
            }
            let rows: usize = f[1].parse().map_err(|_| format!("bad rows in: {line}"))?;
            let cols: usize = f[2].parse().map_err(|_| format!("bad cols in: {line}"))?;
            let stages: usize = f[3].parse().map_err(|_| format!("bad stages in: {line}"))?;
            entries.insert(f[0].to_string(), Entry { rows, cols, stages });
        }
        Ok(VqDir {
            dir: dir.to_path_buf(),
            entries,
        })
    }

    pub fn names(&self) -> impl Iterator<Item = &String> {
        self.entries.keys()
    }

    pub fn has(&self, name: &str) -> bool {
        self.entries.contains_key(name)
    }

    pub fn read(&self, name: &str) -> Result<Tensor, String> {
        let e = self
            .entries
            .get(name)
            .ok_or_else(|| format!("{name} is not in the manifest"))?;
        let path = self.dir.join(format!("{name}.bin"));
        let bytes = fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;

        if e.stages == 0 {
            let raw = vq_format::RawF32::open(&bytes)?;
            if raw.rows != e.rows || raw.cols != e.cols {
                return Err(format!(
                    "{name}: sidecar is {}x{}, manifest says {}x{}",
                    raw.rows, raw.cols, e.rows, e.cols
                ));
            }
            return Ok(Tensor {
                rows: raw.rows,
                cols: raw.cols,
                data: raw.data,
            });
        }

        let c = vq_format::Container::open(&bytes)?;
        if c.rows != e.rows || c.cols != e.cols || c.stages.len() != e.stages {
            return Err(format!(
                "{name}: container is {}x{} with {} stages, manifest says {}x{} with {}",
                c.rows,
                c.cols,
                c.stages.len(),
                e.rows,
                e.cols,
                e.stages
            ));
        }
        Ok(Tensor {
            rows: c.rows,
            cols: c.cols,
            data: c.decode(),
        })
    }
}
