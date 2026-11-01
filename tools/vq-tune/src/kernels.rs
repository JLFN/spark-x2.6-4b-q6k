//! CUDA bindings and the measured sweep. Every timing is a CUDA event around
//! the real launch, the same entry points the model runtime calls, so a win
//! here is a win there and not a benchmark-only shape.
//!
//! The tuner never invents a candidate: it asks the library which variants and
//! tile shapes are legal for a geometry (vqg_gemv_ready, vqg_prefill_cfg_ready)
//! and times only those. A candidate whose output moves past the gate is
//! recorded as rejected, never selected.

use std::ffi::{c_char, c_void, CStr};

use crate::artifact::Geom;

pub const CUDA_MEMCPY_HOST_TO_DEVICE: i32 = 1;
pub const CUDA_MEMCPY_DEVICE_TO_HOST: i32 = 2;

extern "C" {
    fn cudaMalloc(dev: *mut *mut c_void, n: usize) -> i32;
    fn cudaFree(dev: *mut c_void) -> i32;
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, n: usize, kind: i32) -> i32;
    fn cudaMemset(dev: *mut c_void, v: i32, n: usize) -> i32;
}

extern "C" {
    fn vqg_gemv_ready(dim: i32, nc: i32, rows: i32, cols: i32, v3: i32, variant: i32) -> i32;
    fn vqg_gemv_t(
        payload: *const c_void, dim: i32, nc: i32, rows: i32, cols: i32, v3: i32,
        x: *const f32, y: *mut f32, variant: i32, threads: i32, iters: i32, ms: *mut f64,
    ) -> i32;
    fn vqg_prefill_cfg_count() -> i32;
    fn vqg_prefill_cfg(idx: i32, bn: *mut i32, bk: *mut i32) -> i32;
    fn vqg_prefill_cfg_ready(
        dim: i32, nc: i32, rows: i32, cols: i32, v3: i32, bn: i32, bk: i32, variant: i32,
    ) -> i32;
    fn vqg_prefill_tile(
        payload: *const c_void, dim: i32, nc: i32, rows: i32, cols: i32, v3: i32,
        xt: *const f32, ntok: i32, y: *mut f32, iters: i32, ms: *mut f64,
        bn: i32, bk: i32, variant: i32,
    ) -> i32;
    fn vqg_device_name(out: *mut c_char, n: i32) -> i32;
    fn vqg_driver_version() -> i32;
    fn vqg_decode_error() -> *const c_char;
}

pub fn last_error() -> String {
    unsafe {
        let p = vqg_decode_error();
        if p.is_null() {
            "unknown CUDA error".into()
        } else {
            CStr::from_ptr(p).to_string_lossy().into_owned()
        }
    }
}

fn check(rc: i32, what: &str) -> Result<(), String> {
    if rc == 0 {
        Ok(())
    } else {
        Err(format!("{what}: {}", last_error()))
    }
}

pub struct Dev {
    p: *mut c_void,
    bytes: usize,
}

impl Dev {
    pub fn alloc(bytes: usize) -> Result<Dev, String> {
        let mut p: *mut c_void = std::ptr::null_mut();
        check(unsafe { cudaMalloc(&mut p, bytes.max(4)) }, "cudaMalloc")?;
        Ok(Dev { p, bytes })
    }

    pub fn ptr(&self) -> *mut c_void {
        self.p
    }

    pub fn zero(&self) -> Result<(), String> {
        check(unsafe { cudaMemset(self.p, 0, self.bytes) }, "cudaMemset")
    }

    pub fn fill(&self, offset: usize, len: usize, value: i32) -> Result<(), String> {
        check(
            unsafe { cudaMemset((self.p as *mut u8).add(offset) as *mut c_void, value, len) },
            "cudaMemset range",
        )
    }

    pub fn upload(&self, src: &[u8], offset: usize) -> Result<(), String> {
        if offset + src.len() > self.bytes {
            return Err(format!("upload of {} bytes at {offset} does not fit {}", src.len(), self.bytes));
        }
        check(
            unsafe {
                cudaMemcpy(
                    (self.p as *mut u8).add(offset) as *mut c_void,
                    src.as_ptr() as *const c_void,
                    src.len(),
                    CUDA_MEMCPY_HOST_TO_DEVICE,
                )
            },
            "cudaMemcpy host to device",
        )
    }

    pub fn download_f32(&self, n: usize) -> Result<Vec<f32>, String> {
        let mut v = vec![0f32; n];
        check(
            unsafe {
                cudaMemcpy(
                    v.as_mut_ptr() as *mut c_void,
                    self.p,
                    n * 4,
                    CUDA_MEMCPY_DEVICE_TO_HOST,
                )
            },
            "cudaMemcpy device to host",
        )?;
        Ok(v)
    }
}

impl Drop for Dev {
    fn drop(&mut self) {
        unsafe {
            cudaFree(self.p);
        }
    }
}

pub fn device_name() -> Result<String, String> {
    let mut buf = [0 as c_char; 128];
    check(unsafe { vqg_device_name(buf.as_mut_ptr(), buf.len() as i32) }, "device name")?;
    Ok(unsafe { CStr::from_ptr(buf.as_ptr()) }.to_string_lossy().into_owned())
}

pub fn driver_version() -> String {
    let v = unsafe { vqg_driver_version() };
    if v <= 0 {
        "unknown".into()
    } else {
        format!("{}.{}", v / 1000, (v % 1000) / 10)
    }
}

pub struct Payload {
    pub stages: Vec<Dev>,
    pub bytes: usize,
}

pub fn upload(stages: &[Vec<u8>]) -> Result<Payload, String> {
    let mut devs = Vec::new();
    let mut bytes = 0;
    for s in stages {
        // 8 bytes of slack: the kernels read aligned 4- and 8-byte index
        // windows, so the last index of the last row can reach past the
        // payload (see rd_idx_fmt in tools/vq/gpu/decode-gpu.cu).
        let d = Dev::alloc(s.len() + 8)?;
        d.upload(s, 0)?;
        d.fill(s.len(), 8, 0)?;
        bytes += s.len();
        devs.push(d);
    }
    Ok(Payload { stages: devs, bytes })
}

/// One GEMV candidate: every stage accumulated into y, exactly as the runtime
/// does it. `iters` > 0 times the launches; the returned rate is the summed
/// best time of the stages, and `out` is only meaningful for `iters == 1`.
pub fn gemv(
    p: &Payload, g: Geom, x: &Dev, y: &Dev, variant: i32, threads: i32, iters: i32,
) -> Result<(f64, Vec<f32>), String> {
    let mut ms = 0.0;
    for s in &p.stages {
        let mut one = 0.0;
        check(
            unsafe {
                vqg_gemv_t(
                    s.ptr(), g.dim as i32, g.nc as i32, g.rows as i32, g.cols as i32, g.v3 as i32,
                    x.ptr() as *const f32, y.ptr() as *mut f32, variant, threads, iters, &mut one,
                )
            },
            "vqg_gemv_t",
        )?;
        ms += one;
    }
    let out = if iters == 1 { y.download_f32(g.rows as usize)? } else { Vec::new() };
    Ok((ms, out))
}

pub fn gemv_ready(g: Geom, variant: i32) -> bool {
    unsafe { vqg_gemv_ready(g.dim as i32, g.nc as i32, g.rows as i32, g.cols as i32, g.v3 as i32, variant) != 0 }
}

pub fn prefill_shapes() -> Vec<(i32, i32)> {
    let n = unsafe { vqg_prefill_cfg_count() };
    let mut out = Vec::new();
    for i in 0..n {
        let (mut bn, mut bk) = (0, 0);
        if unsafe { vqg_prefill_cfg(i, &mut bn, &mut bk) } == 0 {
            out.push((bn, bk));
        }
    }
    out
}

pub fn prefill_ready(g: Geom, bn: i32, bk: i32, variant: i32) -> bool {
    unsafe {
        vqg_prefill_cfg_ready(
            g.dim as i32, g.nc as i32, g.rows as i32, g.cols as i32, g.v3 as i32, bn, bk, variant,
        ) != 0
    }
}

/// One prefill candidate, all stages accumulated into y[ntok][rows].
pub fn prefill(
    p: &Payload, g: Geom, xt: &Dev, y: &Dev, ntok: i32, variant: i32, bn: i32, bk: i32, iters: i32,
) -> Result<(f64, Vec<f32>), String> {
    let mut ms = 0.0;
    for s in &p.stages {
        let mut one = 0.0;
        check(
            unsafe {
                vqg_prefill_tile(
                    s.ptr(), g.dim as i32, g.nc as i32, g.rows as i32, g.cols as i32, g.v3 as i32,
                    xt.ptr() as *const f32, ntok, y.ptr() as *mut f32, iters, &mut one, bn, bk, variant,
                )
            },
            "vqg_prefill_tile",
        )?;
        ms += one;
    }
    let out = if iters == 1 { y.download_f32((ntok as usize) * (g.rows as usize))? } else { Vec::new() };
    Ok((ms, out))
}

/// Normwise relative difference, the same shape of gate the project already
/// uses for logits: ||a - b|| / ||b||.
pub fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    let mut d = 0f64;
    let mut n = 0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let dx = *x as f64 - *y as f64;
        d += dx * dx;
        n += (*y as f64) * (*y as f64);
    }
    if n == 0.0 {
        if d == 0.0 { 0.0 } else { f64::INFINITY }
    } else {
        (d / n).sqrt()
    }
}
