//! vq-format — one reader for the two artifacts the VQ builders write.
//!
//! The 2-D tensors live in RQVQ containers:
//!
//!   [u8 'RQVQ'][u16 dim][u16 nc][u16 stages][u16 pad][u32 rows][u32 cols]
//!   [u64 payload_bytes]
//!   `stages` payloads, each either
//!     DQVQ: [u8 'DQVQ'][u16 dim][u16 nc][u32 rows][u32 cols]
//!           [codebook nc*dim f16][row gains rows f16][bit-packed indices]
//!     DQV3: the same fields plus a 24-byte header (4 pad) and a row-local
//!           min(bit width, 12)-bit main stream with a 1-bit plane above 12.
//!
//! The 1-D tensors (norms) live in RQVF sidecars:
//!
//!   [u8 'RQVF'][u32 rows][u32 cols][rows*cols f32]
//!
//! The decode follows the C source of truth line for line: `vq_pack` and
//! `vq_unpack_dequant` in tools/vq/vendor/vq_qc.h, the byte pricing in
//! `vq_qc_bytes.h`, and for DQV3 `v3_pack` in tools/vq/gpu/build-vq-gpu.c and
//! cuda/ds41_vq_row.cuh:7-20, :122-137 in ds4-dfm-rs. dqvq and the runtime both
//! link this crate, so there is exactly one decoder in the repository.

const RQVQ: &[u8; 4] = b"RQVQ";
const DQVQ: &[u8; 4] = b"DQVQ";
const DQV3: &[u8; 4] = b"DQV3";
const RQVF: &[u8; 4] = b"RQVF";

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
pub fn f16(h: u16) -> f32 {
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

/// One residual stage of an RQVQ container, borrowing its bytes.
pub struct Stage<'a> {
    pub dim: usize,
    pub nc: usize,
    pub rows: usize,
    pub cols: usize,
    pub nbit: u32,
    /// DQV3 row-local stream: bytes of the min(bit width, 12)-bit main stream
    /// and of the plane, per row. False for DQVQ, whose bits run across the
    /// whole matrix.
    pub v3: bool,
    pub mrow: usize,
    pub prow: usize,
    gains: &'a [u8],
    stream: &'a [u8],
    /// Codebook as f32, the form the hot loops use.
    cb32: Vec<f32>,
}

impl<'a> Stage<'a> {
    pub fn open(p: &'a [u8]) -> Result<Stage<'a>, String> {
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
    pub fn index(&self, row: usize, i: usize) -> u32 {
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
    pub fn gain(&self, row: usize) -> f32 {
        f16(u16le(self.gains, row * 2))
    }

    /// Add this stage's reconstruction into `out` (row-major, zeroed).
    pub fn add_into(&self, out: &mut [f32]) {
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
    pub fn dot_row(&self, r: usize, x: &[f32], acc: &mut [f32]) {
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

/// An RQVQ container over the bytes of one `<tensor>.bin` file.
pub struct Container<'a> {
    pub dim: usize,
    pub nc: usize,
    pub stages: Vec<Stage<'a>>,
    pub rows: usize,
    pub cols: usize,
}

impl<'a> Container<'a> {
    pub fn open(buf: &'a [u8]) -> Result<Container<'a>, String> {
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
        let v3 = buf.len() >= 36 && &buf[32..36] == DQV3;
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

    /// Sum every stage into one row-major f32 matrix.
    pub fn decode(&self) -> Vec<f32> {
        let mut out = vec![0f32; self.rows * self.cols];
        for s in &self.stages {
            s.add_into(&mut out);
        }
        out
    }
}

/// A raw f32 sidecar (the norms), `rows x cols`, written by build-vq-gpu as RQVF.
pub struct RawF32 {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

impl RawF32 {
    pub fn open(b: &[u8]) -> Result<RawF32, String> {
        if b.len() < 12 || &b[0..4] != RQVF {
            return Err("not an RQVF sidecar".into());
        }
        let rows = u32le(b, 4) as usize;
        let cols = u32le(b, 8) as usize;
        if rows == 0 || cols == 0 {
            return Err("RQVF header is inconsistent".into());
        }
        let n = rows
            .checked_mul(cols)
            .ok_or_else(|| "RQVF shape overflows".to_string())?;
        if b.len() != 12 + n * 4 {
            return Err(format!(
                "RQVF file is {} bytes, shape needs {}",
                b.len(),
                12 + n * 4
            ));
        }
        let mut data = Vec::with_capacity(n);
        for w in b[12..].chunks_exact(4) {
            data.push(f32::from_le_bytes([w[0], w[1], w[2], w[3]]));
        }
        Ok(RawF32 { rows, cols, data })
    }
}
