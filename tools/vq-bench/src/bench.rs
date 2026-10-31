//! Artifact manifest, the per-matrix harness loop, and the ds4-bench CSV row.
//!
//! This is the Rust port of the measurement side of ds4-perf: the same column
//! set (`ctx_tokens,prefill_tokens,prefill_tps,gen_tokens,gen_tps,
//! first_token_sec,kvcache_bytes`), so a run here drops into the same compare
//! tooling. What it measures is the VQ weight path: prefill is a real N-token
//! GEMM over the fused decoder, decode is batch 1, and kvcache_bytes is the
//! checkpoint geometry, not a runtime allocation.
//!
//! Memory discipline is the reason this is Rust: every device buffer is an RAII
//! `DevBuf`, the container file is dropped as soon as it is uploaded, and the
//! activation and output buffers are allocated once at the largest shape in the
//! manifest and reused. Peak device use is reported, so a run that would not fit
//! a card is visible before it starts.

use crate::cuda::{vqg_dequant, vqg_gemv, vqg_prefill, vqg_prefill_ready, DevBuf};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct Tensor {
    pub name: String,
    pub rows: u32,
    pub cols: u32,
    pub stages: u32,
    pub file_bytes: u64,
    pub path: PathBuf,
}

pub struct Manifest {
    pub tensors: Vec<Tensor>,
    pub dir: PathBuf,
}

pub fn read_manifest(dir: &Path) -> Result<Manifest, String> {
    let path = dir.join("manifest.txt");
    let text = fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut tensors = Vec::new();
    for line in text.lines() {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let f: Vec<&str> = line.split_whitespace().collect();
        if f.len() < 5 {
            continue;
        }
        let num = |i: usize, what: &str| -> Result<u64, String> {
            f[i].parse::<u64>()
                .map_err(|_| format!("manifest {what} on line: {line}"))
        };
        let rows = num(1, "rows")? as u32;
        let cols = num(2, "cols")? as u32;
        let stages = num(3, "stages")? as u32;
        if stages == 0 {
            continue; /* 1-D tensors are copied f32 and never decoded */
        }
        tensors.push(Tensor {
            name: f[0].to_string(),
            rows,
            cols,
            stages,
            file_bytes: num(4, "bytes")?,
            path: dir.join(format!("{}.bin", f[0])),
        });
    }
    if tensors.is_empty() {
        return Err(format!("{}: no matrices", path.display()));
    }
    Ok(Manifest {
        tensors,
        dir: dir.to_path_buf(),
    })
}

pub struct Harness {
    iters: u32,
    prefill: u32,
    x: DevBuf,
    xt: DevBuf,
    yt: DevBuf,
}

impl Harness {
    /// Allocate the activation and output buffers once, at the largest shape in
    /// the manifest. sin() is the test activation; only its shape matters for
    /// timing, and generating it once keeps the harness out of the measurement.
    pub fn new(m: &Manifest, iters: u32, prefill: u32) -> Result<Harness, String> {
        let max_cols = m.tensors.iter().map(|t| t.cols).max().unwrap_or(1) as usize;
        let max_rows = m.tensors.iter().map(|t| t.rows).max().unwrap_or(1) as usize;
        let ntok = prefill.max(1) as usize;

        let x = DevBuf::alloc(max_cols * 4)?;
        let host: Vec<f32> = (0..max_cols).map(|i| (i as f32 * 0.001).sin()).collect();
        x.upload(as_bytes(&host), 0)?;

        let xt = DevBuf::alloc(max_cols * ntok * 4)?;
        if prefill > 0 {
            let n = max_cols * ntok;
            let host: Vec<f32> = (0..n).map(|i| (i as f32 * 0.0001).sin()).collect();
            xt.upload(as_bytes(&host), 0)?;
        }
        let yt = DevBuf::alloc(max_rows * ntok * 4)?;
        yt.zero()?;

        Ok(Harness {
            iters,
            prefill,
            x,
            xt,
            yt,
        })
    }

    pub fn peak_dev_bytes(&self) -> usize {
        self.x.bytes + self.xt.bytes + self.yt.bytes
    }

    /// Run one matrix: upload its container, time decode, prefill and dequant.
    /// The container bytes are freed when this returns; nothing is retained
    /// between tensors except the shared buffers above.
    pub fn measure(&self, t: &Tensor) -> Result<Sample, String> {
        let raw = fs::read(&t.path).map_err(|e| format!("{}: {e}", t.path.display()))?;
        if raw.len() < 32 || &raw[0..4] != b"RQVQ" {
            return Err(format!("{}: not an RQVQ container", t.path.display()));
        }
        let payload = u64::from_le_bytes(raw[24..32].try_into().unwrap()) as usize;
        if raw.len() != 32 + payload {
            return Err(format!("{}: header and file size disagree", t.path.display()));
        }
        let v3 = &raw[32..36] == b"DQV3";
        let dim = u16::from_le_bytes([raw[32 + 4], raw[32 + 5]]) as i32;
        let nc = u16::from_le_bytes([raw[32 + 6], raw[32 + 7]]) as i32;
        if t.cols as i32 % dim != 0 {
            return Err(format!(
                "{}: cols {} is not a multiple of dim {dim}",
                t.path.display(),
                t.cols
            ));
        }
        /* Fastest kernel this geometry can run: the DQV3 block kernel when the
         * payload is v3 and its rows are aligned, else the dim-4 fast path,
         * else the generic warp-per-row kernel (correct but untuned). The
         * report names the path so a slow number is not mistaken for the
         * format's. */
        let ready = |v: i32| unsafe {
            crate::cuda::vqg_gemv_ready(dim, nc, t.rows as i32, t.cols as i32, v3 as i32, v) != 0
        };
        let variant: i32 = if ready(3) { 3 } else if ready(2) { 2 } else { 1 };
        let per = payload / t.stages as usize;
        let aper = (per + 7) & !7usize;

        let dev = DevBuf::alloc(32 + aper * t.stages as usize + 8)?;
        dev.zero()?;
        for s in 0..t.stages as usize {
            dev.upload(&raw[32 + per * s..32 + per * (s + 1)], 32 + aper * s)?;
        }
        drop(raw);

        let stage = |s: usize| -> *const u8 {
            unsafe { (dev.ptr() as *const u8).add(32 + aper * s) }
        };

        let rows = t.rows as i32;
        let cols = t.cols as i32;

        let mut decode_ms = 0.0f64;
        for s in 0..t.stages as usize {
            let mut ms = 0.0f64;
            let rc = unsafe {
                vqg_gemv(
                    stage(s) as *const _,
                    dim,
                    nc,
                    rows,
                    cols,
                    v3 as i32,
                    self.x.ptr() as *const f32,
                    self.yt.ptr() as *mut f32,
                    variant,
                    self.iters as i32,
                    &mut ms,
                )
            };
            if rc != 0 {
                return Err(format!("{}: {}", t.path.display(), crate::cuda::last_error()));
            }
            decode_ms += ms;
        }

        /* Prefill path this geometry supports: the tiled f16-mma kernel with the
         * codebook staged in shared when it fits, else with it in L1, else the
         * legacy per-(row, token) kernel. Named in the report so a slow number
         * is not read as the format's. */
        let pv = if self.prefill > 0 {
            let ok = |v: i32| unsafe {
                vqg_prefill_ready(dim, nc, rows, cols, v3 as i32, v) != 0
            };
            if ok(2) { 2 } else if ok(1) { 1 } else { 3 }
        } else {
            0
        };

        let mut prefill_ms = 0.0f64;
        if self.prefill > 0 {
            for s in 0..t.stages as usize {
                let mut ms = 0.0f64;
                let rc = unsafe {
                    vqg_prefill(
                        stage(s) as *const _,
                        dim,
                        nc,
                        rows,
                        cols,
                        v3 as i32,
                        self.xt.ptr() as *const f32,
                        self.prefill as i32,
                        self.yt.ptr() as *mut f32,
                        self.iters as i32,
                        &mut ms,
                        pv,
                    )
                };
                if rc != 0 {
                    return Err(format!("{}: {}", t.path.display(), crate::cuda::last_error()));
                }
                prefill_ms += ms;
            }
        }

        let mut dequant_ms = 0.0f64;
        {
            let scratch = DevBuf::alloc(t.rows as usize * t.cols as usize * 4)?;
            for s in 0..t.stages as usize {
                scratch.zero()?;
                let mut ms = 0.0f64;
                let rc = unsafe {
                    vqg_dequant(
                        stage(s) as *const _,
                        dim,
                        nc,
                        rows,
                        cols,
                        v3 as i32,
                        scratch.ptr() as *mut f32,
                        self.iters as i32,
                        &mut ms,
                    )
                };
                if rc != 0 {
                    return Err(format!("{}: {}", t.path.display(), crate::cuda::last_error()));
                }
                dequant_ms += ms;
            }
        }

        Ok(Sample {
            decode_ms,
            prefill_ms,
            dequant_ms,
            variant,
            prefill_variant: pv,
        })
    }
}

pub struct Sample {
    pub decode_ms: f64,
    pub prefill_ms: f64,
    pub dequant_ms: f64,
    /// Kernel used for the fused GEMV: 2 = dim-4 fast path, 1 = generic warp.
    pub variant: i32,
    /// Prefill path: 2 = tiled f16 mma with the codebook in shared, 1 = tiled
    /// with it in L1, 3 = legacy one-thread-per-(row, token) kernel.
    pub prefill_variant: i32,
}

/// Aggregate over the whole artifact, in the ds4-bench column order.
pub struct Report {
    pub tensors: usize,
    pub container_bytes: u64,
    pub weights: u64,
    pub decode_ms: f64,
    pub prefill_ms: f64,
    pub dequant_ms: f64,
    pub peak_dev_bytes: usize,
    pub prefill_tokens: u32,
    pub kvcache_bytes: u64,
    pub slowest: Vec<(String, f64)>,
}

impl Report {
    pub fn decode_tps(&self) -> f64 {
        1000.0 / self.decode_ms
    }

    pub fn prefill_tps(&self) -> f64 {
        if self.prefill_ms <= 0.0 {
            0.0
        } else {
            self.prefill_tokens as f64 * 1000.0 / self.prefill_ms
        }
    }

    pub fn csv_row(&self) -> String {
        format!(
            "{},{},{:.1},{},{:.1},{:.4},{}\n",
            self.prefill_tokens,
            self.prefill_tokens,
            self.prefill_tps(),
            1,
            self.decode_tps(),
            self.prefill_ms / 1000.0,
            self.kvcache_bytes
        )
    }
}

pub const CSV_HEADER: &str =
    "ctx_tokens,prefill_tokens,prefill_tps,gen_tokens,gen_tps,first_token_sec,kvcache_bytes";

pub fn as_bytes(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}
