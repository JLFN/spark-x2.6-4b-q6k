//! vq-runtime — the reference inference runtime for Spark2_5.
//!
//! The `vq-runtime` binary in `src/main.rs` is a thin driver over these
//! modules. They are public so another tool can drive the same target model
//! in-process: `dspark spec` runs the DSpark propose/verify loop on top of
//! this target instead of loading a second copy of the weights.

pub mod config;
pub mod model;
pub mod safetensors;
pub mod vq_source;
