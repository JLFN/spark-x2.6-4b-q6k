//! Reader for the ds4-bench CSV schema, ported from ds4-perf's bench.rs so the
//! two tools agree line for line:
//!
//!   ctx_tokens,prefill_tokens,prefill_tps,gen_tokens,gen_tps,first_token_sec,kvcache_bytes
//!
//! Columns are matched by name, not position, and a line whose header is not
//! this one is ignored. That is what lets `vq-bench compare` read a file written
//! by the ds4 bench as well as one written here.

#[derive(Clone, Debug, PartialEq)]
pub struct Row {
    pub ctx: u64,
    pub prefill_tokens: u64,
    pub prefill_tps: f64,
    pub gen_tokens: u64,
    pub gen_tps: f64,
    pub first_token_sec: f64,
    pub kvcache_bytes: u64,
}

pub const HEADER: &str =
    "ctx_tokens,prefill_tokens,prefill_tps,gen_tokens,gen_tps,first_token_sec,kvcache_bytes";

fn split(line: &str) -> Vec<&str> {
    line.split(',').map(|s| s.trim()).collect()
}

fn column(header: &[&str], names: &[&str]) -> Option<usize> {
    header.iter().position(|h| names.iter().any(|n| h.eq_ignore_ascii_case(n)))
}

fn value<'a>(row: &[&'a str], header: &[&str], names: &[&str]) -> Option<&'a str> {
    column(header, names).and_then(|i| row.get(i).copied())
}

/// Parse every data row. A file with no recognisable header yields no rows,
/// which mirrors ds4-perf: the caller decides whether that is an error.
pub fn parse(text: &str) -> Result<Vec<Row>, String> {
    let mut header: Vec<&str> = Vec::new();
    let mut rows = Vec::new();
    for line in text.lines() {
        let row = split(line);
        if column(&row, &["prefill_tps"]).is_some() {
            header = row;
            continue;
        }
        if header.is_empty() || row.iter().all(|v| v.is_empty()) {
            continue;
        }
        let field = |name: &str| -> Result<&str, String> {
            value(&row, &header, &[name]).ok_or_else(|| format!("missing ds4-bench column: {name}"))
        };
        let integer = |name: &str| -> Result<u64, String> {
            field(name)?
                .parse::<u64>()
                .map_err(|_| format!("invalid {name}"))
        };
        let number = |name: &str| -> Result<f64, String> {
            let v: f64 = field(name)?
                .parse()
                .map_err(|_| format!("invalid {name}"))?;
            if v.is_finite() {
                Ok(v)
            } else {
                Err(format!("non-finite {name}"))
            }
        };
        rows.push(Row {
            ctx: integer("ctx_tokens")?,
            prefill_tokens: integer("prefill_tokens")?,
            prefill_tps: number("prefill_tps")?,
            gen_tokens: integer("gen_tokens")?,
            gen_tps: number("gen_tps")?,
            first_token_sec: number("first_token_sec")?,
            kvcache_bytes: integer("kvcache_bytes")?,
        });
    }
    Ok(rows)
}
