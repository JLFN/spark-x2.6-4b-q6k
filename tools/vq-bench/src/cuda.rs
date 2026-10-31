//! CUDA bindings for the benchmark harness: the runtime entry points the harness
//! needs, plus the kernels compiled from tools/vq/gpu/decode-gpu.cu.
//!
//! The kernels stay in CUDA; the harness, the container mapping and the report
//! are Rust. That split is deliberate: the part that decides memory use is the
//! mapping and the buffers, and those are easier to keep tight here than in C.

use std::ffi::c_char;
use std::ffi::c_void;
use std::ffi::CStr;

pub const CUDA_MEMCPY_HOST_TO_DEVICE: i32 = 1;

#[link(name = "cudart")]
extern "C" {
    fn cudaMalloc(dev: *mut *mut c_void, size: usize) -> i32;
    fn cudaFree(dev: *mut c_void) -> i32;
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, n: usize, kind: i32) -> i32;
    fn cudaMemset(dev: *mut c_void, value: i32, n: usize) -> i32;
    pub fn cudaDeviceSynchronize() -> i32;
    fn cudaGetErrorString(e: i32) -> *const c_char;
}

extern "C" {
    /// Fused gather+dot per stage; variant 2 is the dim-4 fast kernel.
    pub fn vqg_gemv(
        payload: *const c_void,
        dim: i32,
        nc: i32,
        rows: i32,
        cols: i32,
        v3: i32,
        x: *const f32,
        y: *mut f32,
        variant: i32,
        iters: i32,
        ms: *mut f64,
    ) -> i32;
    /// Prefill of `ntok` tokens; xT is [cols][ntok], y is [ntok][rows].
    pub fn vqg_prefill(
        payload: *const c_void,
        dim: i32,
        nc: i32,
        rows: i32,
        cols: i32,
        v3: i32,
        xt: *const f32,
        ntok: i32,
        y: *mut f32,
        iters: i32,
        ms: *mut f64,
    ) -> i32;
    /// Dequantise one stage into an f32 matrix (accumulates).
    pub fn vqg_dequant(
        payload: *const c_void,
        dim: i32,
        nc: i32,
        rows: i32,
        cols: i32,
        v3: i32,
        out: *mut f32,
        iters: i32,
        ms: *mut f64,
    ) -> i32;
    pub fn vqg_decode_error() -> *const c_char;
}

pub fn last_error() -> String {
    unsafe {
        let p = vqg_decode_error();
        if p.is_null() {
            return "unknown CUDA error".into();
        }
        CStr::from_ptr(p).to_string_lossy().into_owned()
    }
}

fn check(rc: i32, what: &str) -> Result<(), String> {
    if rc == 0 {
        Ok(())
    } else {
        Err(format!("{what}: {}", last_error()))
    }
}

/// A device allocation that frees itself. Drop is the only place `cudaFree` is
/// called, so an early return cannot leak the 4 GB a full matrix would take.
pub struct DevBuf {
    ptr: *mut c_void,
    pub bytes: usize,
}

impl DevBuf {
    pub fn alloc(bytes: usize) -> Result<DevBuf, String> {
        let mut ptr: *mut c_void = std::ptr::null_mut();
        check(unsafe { cudaMalloc(&mut ptr, bytes) }, "cudaMalloc")?;
        Ok(DevBuf { ptr, bytes })
    }

    pub fn ptr(&self) -> *mut c_void {
        self.ptr
    }

    pub fn zero(&self) -> Result<(), String> {
        check(unsafe { cudaMemset(self.ptr, 0, self.bytes) }, "cudaMemset")
    }

    pub fn upload(&self, src: &[u8], offset: usize) -> Result<(), String> {
        if offset + src.len() > self.bytes {
            return Err(format!(
                "upload of {} bytes at {offset} does not fit {}",
                src.len(),
                self.bytes
            ));
        }
        let dst = unsafe { (self.ptr as *mut u8).add(offset) } as *mut c_void;
        check(
            unsafe { cudaMemcpy(dst, src.as_ptr() as *const c_void, src.len(), CUDA_MEMCPY_HOST_TO_DEVICE) },
            "cudaMemcpy",
        )
    }
}

impl Drop for DevBuf {
    fn drop(&mut self) {
        unsafe {
            cudaFree(self.ptr);
        }
    }
}
