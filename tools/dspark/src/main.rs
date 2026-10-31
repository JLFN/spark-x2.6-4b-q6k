//! dspark — inspect the DSpark draft checkpoint, and later run it.
//!
//! Usage:
//!   dspark dump <file.gguf>

mod gguf;

use std::process::ExitCode;

fn usage() -> &'static str {
    "usage: dspark dump <file.gguf>"
}

fn dump(path: &str) -> Result<(), String> {
    let g = gguf::Gguf::open(path)?;
    println!("file        {path} ({} bytes, GGUF v{})", g.file_len, g.version);
    println!("alignment   {}", g.alignment);

    println!("\nmetadata");
    for (k, v) in &g.kv {
        if k.starts_with("tokenizer.") {
            continue; // tens of thousands of tokens; summarized below
        }
        let text = match v {
            gguf::Value::Array(a) => {
                let small: Vec<String> = a.iter().take(12).map(|x| match x {
                    gguf::Value::String(s) => s.clone(),
                    other => format!("{:?}", other),
                }).collect();
                if a.len() <= 12 { format!("[{}]", small.join(", ")) }
                else { format!("[{} ... {} values]", small.join(", "), a.len()) }
            }
            gguf::Value::String(s) => s.clone(),
            gguf::Value::F32(x) => format!("{x}"),
            gguf::Value::F64(x) => format!("{x}"),
            gguf::Value::Bool(b) => format!("{b}"),
            other => format!("{:?}", other),
        };
        println!("  {k:<48} {text}");
    }
    if let Some(arr) = g.get_arr("tokenizer.ggml.tokens") {
        println!("\ntokenizer   {} tokens, model {}", arr.len(),
                 g.get_str("tokenizer.ggml.model").unwrap_or("?"));
    }

    println!("\ntensors     {} entries", g.tensors.len());
    println!("  {:<28} {:>14} {:>10} {:>8}", "name", "shape", "type", "bytes");
    let mut total = 0u64;
    for t in &g.tensors {
        let bytes = gguf::type_size(t.dtype, t.ne()).unwrap_or(0);
        total += bytes;
        let shape: Vec<String> = t.dims.iter().map(|d| d.to_string()).collect();
        println!("  {:<28} {:>14} {:>10} {:>8}", t.name, shape.join("x"),
                 gguf::type_name(t.dtype), bytes);
    }
    println!("  total {:.3} GB", total as f64 / 1e9);
    Ok(())
}

fn main() -> ExitCode {
    let a: Vec<String> = std::env::args().collect();
    if a.len() != 3 || a[1] != "dump" {
        eprintln!("{}", usage());
        return ExitCode::from(2);
    }
    match dump(&a[2]) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("dspark: {e}");
            ExitCode::FAILURE
        }
    }
}
