//! Compile the CUDA kernels this tuner measures and link cuBLAS-free cudart.
//!
//! The kernels live with the rest of the C tools (tools/vq/gpu/decode-gpu.cu)
//! so there is one copy of the decoder for the C driver, the benchmark harness,
//! the model runtime and this tuner. nvcc is a prerequisite, not an option: a
//! build without it must fail here rather than produce a binary that cannot
//! measure anything.

use std::env;
use std::path::PathBuf;
use std::process::Command;

/// Find nvcc without pinning a toolkit version: an explicit NVCC wins, then
/// whatever the PATH resolves, then the unversioned /usr/local/cuda symlink.
/// A hardcoded /usr/local/cuda-13.3 would break the day the toolkit is
/// upgraded, which is exactly what this avoids.
fn find_nvcc() -> String {
    if let Ok(v) = env::var("NVCC") {
        return v;
    }
    if let Ok(out) = Command::new("sh").args(["-c", "command -v nvcc"]).output() {
        if out.status.success() {
            let p = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if !p.is_empty() {
                return p;
            }
        }
    }
    "/usr/local/cuda/bin/nvcc".into()
}

fn main() {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let cu = manifest.join("../vq/gpu/decode-gpu.cu");
    if !cu.exists() {
        panic!("missing CUDA source: {}", cu.display());
    }

    let nvcc = find_nvcc();
    let arch = env::var("CUDA_ARCH").unwrap_or_else(|_| "89".into());
    let out = PathBuf::from(env::var("OUT_DIR").unwrap()).join("decode_gpu.o");

    let status = Command::new(&nvcc)
        .args(["-O3", "-std=c++17", &format!("-arch=sm_{arch}"), "-c", "-o"])
        .arg(&out)
        .arg(&cu)
        .status()
        .unwrap_or_else(|e| panic!("cannot run {nvcc}: {e} (install the CUDA toolkit or set NVCC)"));

    if !status.success() {
        panic!("{nvcc} failed on {}", cu.display());
    }

    println!("cargo:rustc-link-arg={}", out.display());
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rustc-link-arg=-Wl,-rpath,/usr/local/cuda/lib64");
    println!("cargo:rerun-if-changed={}", cu.display());
    println!("cargo:rerun-if-env-changed=NVCC");
    println!("cargo:rerun-if-env-changed=CUDA_ARCH");
}
