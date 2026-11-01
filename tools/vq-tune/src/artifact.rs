//! The artifact side of the tuner: what the manifest says, what a matrix's
//! geometry is, and the fingerprint a profile is bound to.
//!
//! The fingerprint must match tools/vq/gpu/vq-profile.h byte for byte. It is
//! FNV-1a 64 over the manifest bytes, then for every matrix container in
//! manifest order: the name, the size as 8 little-endian bytes, the first
//! min(4096, size) bytes, and the last min(4096, size) bytes (the same edge is
//! hashed twice when the file is 4 KB or smaller, because the C reader does).

use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

pub const FNV_OFFSET: u64 = 1469598103934665603;
const FNV_PRIME: u64 = 1099511628211;

pub fn fnv(mut h: u64, bytes: &[u8]) -> u64 {
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

#[derive(Clone, Debug)]
pub struct Matrix {
    pub name: String,
    pub rows: u32,
    pub cols: u32,
    pub stages: u32,
    pub file_bytes: u64,
}

#[derive(Clone, Debug)]
pub struct Manifest {
    pub dim: u32,
    pub nc: u32,
    pub matrices: Vec<Matrix>,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Geom {
    pub dim: u32,
    pub nc: u32,
    pub rows: u32,
    pub cols: u32,
    pub v3: bool,
}

impl Geom {
    pub fn key(&self) -> String {
        format!("{}x{} dim {} nc {} {}", self.rows, self.cols, self.dim, self.nc,
                if self.v3 { "DQV3" } else { "DQVQ" })
    }
}

/// Parse manifest.txt and read each container's payload tag. The encoding line
/// is `# encoding: dim=4 nc=8192 stages=2 (gpu)`; matrices are
/// `name rows cols stages file_bytes relL2`.
pub fn load(dir: &str) -> Result<Manifest, String> {
    let text = fs::read(format!("{dir}/manifest.txt")).map_err(|e| format!("{dir}/manifest.txt: {e}"))?;
    let mut dim = 0u32;
    let mut nc = 0u32;
    let mut stages = 0u32;
    let mut matrices = Vec::new();
    let text = String::from_utf8_lossy(&text);
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Some(rest) = line.strip_prefix("# encoding:") {
            for tok in rest.split_whitespace() {
                let val = |p: &str| -> Result<u32, String> {
                    tok.strip_prefix(p)
                        .ok_or_else(|| format!("unexpected encoding token {tok}"))?
                        .parse::<u32>()
                        .map_err(|e| format!("{tok}: {e}"))
                };
                if tok.starts_with("dim=") {
                    dim = val("dim=")?;
                } else if tok.starts_with("nc=") {
                    nc = val("nc=")?;
                } else if tok.starts_with("stages=") {
                    stages = val("stages=")?;
                }
            }
            continue;
        }
        if line.starts_with('#') {
            continue;
        }
        let f: Vec<&str> = line.split_whitespace().collect();
        if f.len() < 5 {
            continue;
        }
        matrices.push(Matrix {
            name: f[0].to_string(),
            rows: f[1].parse().map_err(|e| format!("{}: rows {e}", f[0]))?,
            cols: f[2].parse().map_err(|e| format!("{}: cols {e}", f[0]))?,
            stages: f[3].parse().map_err(|e| format!("{}: stages {e}", f[0]))?,
            file_bytes: f[4].parse().map_err(|e| format!("{}: bytes {e}", f[0]))?,
        });
    }
    if matrices.is_empty() {
        return Err(format!("{dir}/manifest.txt lists no matrices"));
    }
    let _ = stages;
    Ok(Manifest { dim, nc, matrices })
}

/// Read a container's header: RQVQ, then dim/nc/stages/rows/cols, then the
/// first stage's payload tag at offset 32. Refuses a manifest line that
/// disagrees with the file it names.
pub fn probe(dir: &str, m: &Matrix) -> Result<(u32, u32, bool), String> {
    let path = format!("{dir}/{}.bin", m.name);
    let mut f = File::open(&path).map_err(|e| format!("{path}: {e}"))?;
    let mut h = [0u8; 64];
    let n = f.read(&mut h).map_err(|e| format!("{path}: {e}"))?;
    if n < 36 || &h[0..4] != b"RQVQ" {
        return Err(format!("{path}: not an RQVQ container"));
    }
    let dim = u16::from_le_bytes([h[4], h[5]]) as u32;
    let nc = u16::from_le_bytes([h[6], h[7]]) as u32;
    let nst = u16::from_le_bytes([h[8], h[9]]) as u32;
    let rows = u32::from_le_bytes([h[12], h[13], h[14], h[15]]);
    let cols = u32::from_le_bytes([h[16], h[17], h[18], h[19]]);
    if rows != m.rows || cols != m.cols {
        return Err(format!("{path}: header is {rows}x{cols}, manifest says {}x{}", m.rows, m.cols));
    }
    if nst != m.stages {
        return Err(format!("{path}: header says {nst} stages, manifest says {}", m.stages));
    }
    let size = fs::metadata(&path).map_err(|e| format!("{path}: {e}"))?.len();
    if size != m.file_bytes {
        return Err(format!("{path}: file is {size} bytes, manifest says {}", m.file_bytes));
    }
    if m.stages == 0 {
        return Ok((dim, nc, false));
    }
    // The first stage starts at byte 32; every stage must carry the same tag,
    // which the runtime enforces on load and this mirrors for the sweep.
    let v3 = &h[32..36] == b"DQV3";
    let _ = f.seek(SeekFrom::Start(32));
    Ok((dim, nc, v3))
}

pub fn fingerprint(dir: &str, matrices: &[Matrix]) -> Result<u64, String> {
    let man = fs::read(format!("{dir}/manifest.txt")).map_err(|e| format!("{dir}/manifest.txt: {e}"))?;
    let mut h = fnv(FNV_OFFSET, &man);
    for m in matrices {
        if m.stages == 0 {
            continue;
        }
        let path = format!("{dir}/{}.bin", m.name);
        let size = fs::metadata(&path).map_err(|e| format!("{path}: {e}"))?.len();
        h = fnv(h, m.name.as_bytes());
        h = fnv(h, &size.to_le_bytes());
        let edge = size.min(4096) as usize;
        let mut f = File::open(&path).map_err(|e| format!("{path}: {e}"))?;
        let mut buf = vec![0u8; edge];
        f.read_exact(&mut buf).map_err(|e| format!("{path}: {e}"))?;
        h = fnv(h, &buf);
        if size > 4096 {
            f.seek(SeekFrom::Start(size - 4096)).map_err(|e| format!("{path}: {e}"))?;
            f.read_exact(&mut buf).map_err(|e| format!("{path}: {e}"))?;
        }
        h = fnv(h, &buf);
    }
    Ok(h)
}

/// The payload bytes of one stage.
pub fn read_stages(dir: &str, m: &Matrix) -> Result<Vec<Vec<u8>>, String> {
    let path = format!("{dir}/{}.bin", m.name);
    let mut buf = fs::read(&path).map_err(|e| format!("{path}: {e}"))?;
    if buf.len() < 32 || &buf[0..4] != b"RQVQ" {
        return Err(format!("{path}: not an RQVQ container"));
    }
    let payload = u64::from_le_bytes(buf[24..32].try_into().unwrap()) as usize;
    if buf.len() != 32 + payload {
        return Err(format!("{path}: header says {payload} payload bytes, file has {}", buf.len() - 32));
    }
    let nst = m.stages as usize;
    if nst == 0 || payload % nst != 0 {
        return Err(format!("{path}: {payload} payload bytes do not split into {nst} stages"));
    }
    let per = payload / nst;
    let body = buf.split_off(32);
    Ok((0..nst).map(|s| body[s * per..(s + 1) * per].to_vec()).collect())
}

pub fn exists(dir: &str) -> bool {
    Path::new(dir).join("manifest.txt").exists()
}
