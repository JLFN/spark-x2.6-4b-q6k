//! Compile the CUDA kernels this harness calls and link cuBLAS-free cudart.
//!
//! The kernels live with the rest of the C tools (tools/vq/gpu/decode-gpu.cu) so
//! there is one copy of the decoder that both the C driver and this crate use.
//! nvcc and the CUDA toolkit are prerequisites of this crate, not optional: a
//! build without them should fail here, loudly, rather than produce a binary
//! that cannot time anything.

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let cu = manifest.join("../vq/gpu/decode-gpu.cu");
    if !cu.exists() {
        panic!("missing CUDA source: {}", cu.display());
    }

    let nvcc = env::var("NVCC").unwrap_or_else(|_| "/usr/local/cuda/bin/nvcc".into());
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
