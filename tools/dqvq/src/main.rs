//! dqvq — reference decoder and decode-speed bench for RQVQ containers.
//!
//! One binary, no crates. It reads the container `build-vq` and `build-vq-gpu`
//! write:
//!
//!   [u8 'RQVQ'][u16 dim][u16 nc][u16 stages][u16 pad][u32 rows][u32 cols]
//!   [u64 payload_bytes]
//!   `stages` payloads, each:
//!   [u8 'DQVQ'][u16 dim][u16 nc][u32 rows][u32 cols]
//!   [codebook nc*dim f16][row gains rows f16][bit-packed indices]
//!
//! The decode follows the C source of truth line for line: `vq_pack` and
//! `vq_unpack_dequant` in tools/vq/vendor/vq_qc.h, and the byte pricing in
//! `vq_qc_bytes.h`. The Rust port in ds4-core (crates/ds4-core/src/vq.rs) uses
//! the same bit window and the same f16 rules; it is fixed to dim 8 and a slotted
//! blob, so this tool generalises it to dim 4 and to the multi-stage RQVQ
//! container instead of importing it.
//!
//! Usage:
//!   dqvq info   <file.bin>
//!   dqvq check  <file.bin> <reference.f32>
//!   dqvq gemv   <file.bin> <vec.f32> [--iters N] [--fused]
//!
//! `check` compares the decode with a raw little-endian f32 dump of the same
//! matrix (produced by `decode dump` in C) and fails on any bit difference.
//! `gemv` times the two shapes a runtime can choose between: decode the whole
//! matrix once and multiply (the naive loader), or fuse the index gather and the
//! dot product row by row (what a streaming decoder does). It reports both in
//! milliseconds and as GB/s of container bytes, plus a checksum so the two paths
//! can be compared with each other.

use std::env;
use std::fs;
use std::process::ExitCode;
use std::time::Instant;

const RQVQ: &[u8; 4] = b"RQVQ";
const DQVQ: &[u8; 4] = b"DQVQ";
const DQV3: &[u8; 4] = b"DQV3";

#[inline]
fn u16le(b: &[u8], o: usize) -> u16 {
    u16::from_le_bytes([b[o], b[o + 1]])
}

#[inline]
fn u32le(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}

#[inline]
fn u64le(b: &[u8], o: usize) -> u64 {
    let mut w = [0u8; 8];
    w.copy_from_slice(&b[o..o + 8]);
    u64::from_le_bytes(w)
}

/// Half to float, subnormals exact. Same arithmetic as `ds4vq_f16`
/// (vq_fmt.h:22-28). Note the exponent starts at 113 = 127 - 15 + 1: the ds4
/// Rust port (ds4-core vq.rs) starts at 112 and decodes subnormals one binade
/// low, which this port deliberately does not copy.
#[inline]
fn f16(h: u16) -> f32 {
    let s = (u32::from(h) & 0x8000) << 16;
    let e = (u32::from(h) >> 10) & 0x1f;
    let m = u32::from(h) & 0x3ff;
    let bits = if e == 0 {
        if m == 0 {
            s
        } else {
            let mut e = 113u32;
            let mut m = m;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            s | (e << 23) | ((m & 0x3ff) << 13)
        }
    } else if e == 31 {
        s | 0x7f80_0000 | (m << 13)
    } else {
        s | ((e + 112) << 23) | (m << 13)
    };
    f32::from_bits(bits)
}

struct Stage<'a> {
    dim: usize,
    nc: usize,
    rows: usize,
    cols: usize,
    nbit: u32,
    /// DQV3 row-local stream: bytes of the min(bit width, 12)-bit main stream and of the plane,
    /// per row. Zero for DQVQ, whose bits run across the whole matrix.
    v3: bool,
    mrow: usize,
    prow: usize,
    gains: &'a [u8],
    stream: &'a [u8],
    /// Codebook as f32, the form the hot loops use.
    cb32: Vec<f32>,
}

impl<'a> Stage<'a> {
    fn open(p: &'a [u8]) -> Result<Stage<'a>, String> {
        let v3 = p.len() >= 4 && &p[0..4] == DQV3;
        if p.len() < 16 || (!v3 && &p[0..4] != DQVQ) {
            return Err("stage payload magic is neither DQVQ nor DQV3".into());
        }
        let hdr = if v3 { 24 } else { 16 };
        let dim = u16le(p, 4) as usize;
        let nc = u16le(p, 6) as usize;
        let rows = u32le(p, 8) as usize;
        let cols = u32le(p, 12) as usize;
        if dim == 0 || nc == 0 || rows == 0 || cols == 0 || cols % dim != 0 {
            return Err("stage header is inconsistent".into());
        }
        let cb_len = nc * dim * 2;
        let gain_len = rows * 2;
        if p.len() < hdr + cb_len + gain_len {
            return Err("stage payload is truncated".into());
        }
        let mut nbit = 1u32;
        while (1usize << nbit) < nc {
            nbit += 1;
        }
        if v3 && nbit > 13 {
            return Err(format!(
                "DQV3 carries 13 index bits, nc {nc} needs {nbit}; rewrite with --format v2"
            ));
        }
        let nidx_row = cols / dim;
        let (mrow, prow, idx_bytes) = if v3 {
            let mb = (nbit as usize).min(12);
            let mrow = (nidx_row * mb + 7) / 8;
            let prow = if nbit > 12 { (nidx_row + 7) / 8 } else { 0 };
            (mrow, prow, rows * (mrow + prow))
        } else {
            let nidx = rows * nidx_row;
            let idx = if nbit == 8 { nidx } else { (nidx * nbit as usize + 7) / 8 + 1 };
            (0, 0, idx)
        };
        if p.len() < hdr + cb_len + gain_len + idx_bytes {
            return Err("stage index stream is truncated".into());
        }
        let cb = &p[hdr..hdr + cb_len];
        let gains = &p[hdr + cb_len..hdr + cb_len + gain_len];
        let stream = &p[hdr + cb_len + gain_len..];

        let mut cb32 = Vec::with_capacity(nc * dim);
        for i in 0..nc * dim {
            cb32.push(f16(u16le(cb, i * 2)));
        }
        Ok(Stage {
            dim,
            nc,
            rows,
            cols,
            nbit,
            v3,
            mrow,
            prow,
            gains,
            stream,
            cb32,
        })
    }

    /// Index of the `i`th group in `row`. DQVQ numbers bits across the matrix;
    /// DQV3 restarts at every row and takes the 13th bit from the plane
    /// (ds41_vq_row.cuh:7-20, :122-137).
    #[inline]
    fn index(&self, row: usize, i: usize) -> u32 {
        if self.v3 {
            /* The main stream is min(bit width, 12) bits per index, not always
             * 12: build-vq-gpu.c packs mb = bits > 12 ? 12 : bits, so a 1024
             * codebook has 10-bit slots and shorter rows. A fixed 12 here read
             * the wrong weights for every nc below 4096. */
            let mb = (self.nbit as usize).min(12);
            let bit = i * mb;
            let q = row * self.mrow + (bit >> 3);
            let byte = |k: usize| self.stream.get(q + k).copied().unwrap_or(0) as u32;
            let w = byte(0) | byte(1) << 8 | byte(2) << 16 | byte(3) << 24;
            let mut v = (w >> (bit & 7)) & ((1u32 << mb) - 1);
            if self.prow > 0 {
                let p = self.mrow * self.rows + row * self.prow + (i >> 3);
                v |= (u32::from(self.stream.get(p).copied().unwrap_or(0)) >> (i & 7) & 1) << 12;
            }
            return v;
        }
        let gi = row * (self.cols / self.dim) + i;
        if self.nbit == 8 {
            return u32::from(self.stream[gi]);
        }
        let bit = gi * self.nbit as usize;
        let w = u32::from(self.stream[bit >> 3])
            | u32::from(self.stream[(bit >> 3) + 1]) << 8
            | u32::from(self.stream[(bit >> 3) + 2]) << 16;
        (w >> (bit & 7)) & ((1u32 << self.nbit) - 1)
    }

    #[inline]
    fn gain(&self, row: usize) -> f32 {
        f16(u16le(self.gains, row * 2))
    }

    /// Add this stage's reconstruction into `out` (row-major, zeroed).
    fn add_into(&self, out: &mut [f32]) {
        let nidx_row = self.cols / self.dim;
        for r in 0..self.rows {
            let g = self.gain(r);
            let base = r * self.cols;
            for i in 0..nidx_row {
                let v = self.index(r, i) as usize;
                let cb = &self.cb32[v * self.dim..v * self.dim + self.dim];
                let o = &mut out[base + i * self.dim..base + i * self.dim + self.dim];
                for d in 0..self.dim {
                    o[d] += cb[d] * g;
                }
            }
        }
    }

    /// Fused gather and dot: no decoded matrix is ever materialised. The same
    /// activation vector multiplies every row, because decode is batch 1.
    fn dot_row(&self, r: usize, x: &[f32], acc: &mut [f32]) {
        let nidx_row = self.cols / self.dim;
        let g = self.gain(r);
        let mut s = 0.0f32;
        for i in 0..nidx_row {
            let v = self.index(r, i) as usize;
            let cb = &self.cb32[v * self.dim..v * self.dim + self.dim];
            for d in 0..self.dim {
                s += cb[d] * x[i * self.dim + d];
            }
        }
        acc[r] += s * g;
    }
}

struct Container<'a> {
    dim: usize,
    nc: usize,
    stages: Vec<Stage<'a>>,
    rows: usize,
    cols: usize,
}

fn open(buf: &[u8]) -> Result<Container<'_>, String> {
    if buf.len() < 32 || &buf[0..4] != RQVQ {
        return Err("not an RQVQ container".into());
    }
    let dim = u16le(buf, 4) as usize;
    let nc = u16le(buf, 6) as usize;
    let nstages = u16le(buf, 8) as usize;
    let rows = u32le(buf, 12) as usize;
    let cols = u32le(buf, 16) as usize;
    let payload = u64le(buf, 24) as usize;
    if dim == 0 || nc == 0 || nstages == 0 || rows == 0 || cols == 0 {
        return Err("container header is inconsistent".into());
    }
    let mut nbit = 1u32;
    while (1usize << nbit) < nc {
        nbit += 1;
    }
    let v3 = buf.len() >= 36 && &buf[32..36] == b"DQV3";
    /* DQV3 carries 12 main-stream bits plus one plane bit; a wider index
     * width would decode truncated indices with no error (v3_pack and every
     * decoder cap the width at 13), so refuse it here. */
    if v3 && nbit > 13 {
        return Err(format!(
            "DQV3 carries 13 index bits, nc {nc} needs {nbit}; rewrite with --format v2"
        ));
    }
    let nidx_row = cols / dim;
    let per = if v3 {
        let mb = (nbit as usize).min(12);
        let mrow = (nidx_row * mb + 7) / 8;
        let prow = if nbit > 12 { (nidx_row + 7) / 8 } else { 0 };
        24 + nc * dim * 2 + rows * 2 + rows * (mrow + prow)
    } else {
        let nidx = rows * nidx_row;
        let idx_bytes = if nbit == 8 { nidx } else { (nidx * nbit as usize + 7) / 8 + 1 };
        16 + nc * dim * 2 + rows * 2 + idx_bytes
    };
    if payload != nstages * per {
        return Err(format!(
            "payload byte count says {payload}, layout implies {}",
            nstages * per
        ));
    }
    if buf.len() != 32 + payload {
        return Err(format!(
            "file is {} bytes, header and payload are {}",
            buf.len(),
            32 + payload
        ));
    }
    let mut stages = Vec::with_capacity(nstages);
    for s in 0..nstages {
        let p = &buf[32 + s * per..32 + (s + 1) * per];
        let st = Stage::open(p)?;
        if st.rows != rows || st.cols != cols || st.dim != dim || st.nc != nc {
            return Err(format!("stage {s} disagrees with the container header"));
        }
        stages.push(st);
    }
    Ok(Container {
        dim,
        nc,
        stages,
        rows,
        cols,
    })
}

fn decode(c: &Container<'_>) -> Vec<f32> {
    let mut out = vec![0f32; c.rows * c.cols];
    for s in &c.stages {
        s.add_into(&mut out);
    }
    out
}

fn read_f32(path: &str) -> Result<Vec<f32>, String> {
    let b = fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    if b.len() % 4 != 0 {
        return Err(format!("{path}: {} bytes is not a whole f32 dump", b.len()));
    }
    let mut v = Vec::with_capacity(b.len() / 4);
    for w in b.chunks_exact(4) {
        v.push(f32::from_le_bytes([w[0], w[1], w[2], w[3]]));
    }
    Ok(v)
}

fn cmd_info(path: &str) -> Result<(), String> {
    let buf = fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    let c = open(&buf)?;
    let bytes = buf.len();
    let weights = c.rows * c.cols;
    let per = (bytes - 32) / c.stages.len();
    println!("file        {path}");
    println!("bytes       {bytes} ({:.4} GB)", bytes as f64 / 1e9);
    println!("shape       {} x {}, {} weights", c.rows, c.cols, weights);
    println!(
        "per stage   dim {}, nc {} ({}-bit index), {} bytes",
        c.dim,
        c.nc,
        c.stages[0].nbit,
        per
    );
    println!(
        "stages      {}, payload {}, total {}",
        c.stages.len(),
        bytes - 32,
        bytes
    );
    println!("bits/weight {:.4}", bytes as f64 * 8.0 / weights as f64);
    println!(
        "layout      codebook {}, gains {}, indices {}",
        c.nc * c.dim * 2,
        c.rows * 2,
        c.stages[0].stream.len()
    );
    println!("consistent  yes");
    Ok(())
}

fn cmd_check(path: &str, reference: &str) -> Result<(), String> {
    let buf = fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    let c = open(&buf)?;
    let want = read_f32(reference)?;
    if want.len() != c.rows * c.cols {
        return Err(format!(
            "{reference} holds {} floats, matrix is {}",
            want.len(),
            c.rows * c.cols
        ));
    }
    let got = decode(&c);
    let mut exact = 0usize;
    let mut mx = 0f32;
    let mut se = 0f64;
    let mut sn = 0f64;
    for (a, b) in got.iter().zip(want.iter()) {
        if a.to_bits() == b.to_bits() {
            exact += 1;
        }
        let d = f64::from(*a) - f64::from(*b);
        se += d * d;
        sn += f64::from(*b) * f64::from(*b);
        mx = mx.max(d.abs() as f32);
    }
    println!(
        "{} x {}  stages {}  exact {}/{}  relL2 {:.6}%  maxabs {:.6}",
        c.rows,
        c.cols,
        c.stages.len(),
        exact,
        got.len(),
        100.0 * (se / (sn + 1e-30)).sqrt(),
        mx
    );
    if exact == got.len() {
        Ok(())
    } else {
        Err("decode differs from the reference dump".into())
    }
}

fn cmd_gemv(path: &str, vecpath: &str, iters: usize, fused: bool) -> Result<(), String> {
    let buf = fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    let t0 = Instant::now();
    let c = open(&buf)?;
    let prepare = t0.elapsed().as_secs_f64();

    let x = read_f32(vecpath)?;
    if x.len() != c.cols {
        return Err(format!(
            "{vecpath} holds {} floats, matrix has {} columns",
            x.len(),
            c.cols
        ));
    }

    let mut y = vec![0f32; c.rows];
    let mut decode_s = 0.0;
    let mut best = f64::INFINITY;

    if fused {
        for _ in 0..iters {
            for v in y.iter_mut() {
                *v = 0.0;
            }
            let t = Instant::now();
            for s in &c.stages {
                for r in 0..c.rows {
                    s.dot_row(r, &x, &mut y);
                }
            }
            best = best.min(t.elapsed().as_secs_f64());
        }
    } else {
        let t = Instant::now();
        let w = decode(&c);
        decode_s = t.elapsed().as_secs_f64();
        for _ in 0..iters {
            let t = Instant::now();
            for r in 0..c.rows {
                let row = &w[r * c.cols..(r + 1) * c.cols];
                let mut s = 0.0f32;
                for (a, b) in row.iter().zip(x.iter()) {
                    s += a * b;
                }
                y[r] = s;
            }
            best = best.min(t.elapsed().as_secs_f64());
        }
    }

    let sum: f64 = y.iter().map(|v| f64::from(*v)).sum();
    let nrm: f64 = y.iter().map(|v| f64::from(*v) * f64::from(*v)).sum();
    let container_gb = buf.len() as f64 / 1e9;
    let weights_gb = (c.rows * c.cols * 4) as f64 / 1e9;

    println!(
        "file        {path} ({} x {}, {} stages)",
        c.rows,
        c.cols,
        c.stages.len()
    );
    if fused {
        println!("mode        fused gather+dot, {} passes", iters);
        println!(
            "gemv        {:.3} ms best, {:.2} GB/s of container",
            best * 1e3,
            container_gb / best
        );
    } else {
        println!(
            "prepare     {:.3} ms codebooks to f32",
            prepare * 1e3
        );
        println!(
            "decode      {:.3} ms, {:.2} GB/s of container",
            decode_s * 1e3,
            container_gb / decode_s
        );
        println!("mode        decoded f32 matvec, {} passes", iters);
        println!(
            "matvec      {:.3} ms best, {:.2} GB/s of f32 weights",
            best * 1e3,
            weights_gb / best
        );
    }
    println!("checksum    sum {sum:.6}, norm {nrm:.6}");
    Ok(())
}

fn main() -> ExitCode {
    let a: Vec<String> = env::args().collect();
    if a.len() < 3 {
        eprintln!(
            "usage: {} info <file.bin>\n       {} check <file.bin> <reference.f32>\n       {} gemv <file.bin> <vec.f32> [--iters N] [--fused]",
            a[0], a[0], a[0]
        );
        return ExitCode::from(2);
    }
    let r = match a[1].as_str() {
        "info" => cmd_info(&a[2]),
        "check" if a.len() >= 4 => cmd_check(&a[2], &a[3]),
        "gemv" if a.len() >= 4 => {
            let mut iters = 20usize;
            let mut fused = false;
            let mut i = 4;
            while i < a.len() {
                match a[i].as_str() {
                    "--iters" if i + 1 < a.len() => {
                        iters = a[i + 1].parse().unwrap_or(20);
                        i += 2;
                    }
                    "--fused" => {
                        fused = true;
                        i += 1;
                    }
                    other => {
                        eprintln!("unknown argument {other}");
                        return ExitCode::from(2);
                    }
                }
            }
            cmd_gemv(&a[2], &a[3], iters.max(1), fused)
        }
        _ => Err("unknown command or missing arguments".into()),
    };
    match r {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("dqvq: {e}");
            ExitCode::FAILURE
        }
    }
}
