//! The profile file: the format documented in tools/vq/gpu/vq-profile.h.
//! Every field the C reader looks at is written in the order it scans; the
//! measured rate is an extra trailing field it ignores, kept so a human can
//! see why an entry won without re-running the sweep.

use std::fs;
use std::io::Write;

pub struct Entry {
    pub kind: u8, // 0 = gemv, 1 = prefill
    pub dim: u32,
    pub nc: u32,
    pub rows: u32,
    pub cols: u32,
    pub v3: bool,
    pub variant: i32,
    pub threads: i32, // gemv only
    pub bn: i32,      // prefill only
    pub bk: i32,      // prefill only
    pub rate: f64,    // GB/s for gemv, tok/s for prefill (informational)
}

pub fn render(artifact: u64, gpu: &str, driver: &str, entries: &[Entry]) -> String {
    let mut s = String::new();
    s.push_str("VQPROF 1\n");
    s.push_str(&format!("artifact {artifact:016x}\n"));
    s.push_str(&format!("gpu {gpu}\n"));
    s.push_str(&format!("driver {driver}\n"));
    for e in entries {
        let v3 = e.v3 as i32;
        if e.kind == 0 {
            s.push_str(&format!(
                "gemv {} {} {} {} {} {} {} {:.1}\n",
                e.dim, e.nc, e.rows, e.cols, v3, e.variant, e.threads, e.rate
            ));
        } else {
            s.push_str(&format!(
                "prefill {} {} {} {} {} {} {} {} {:.1}\n",
                e.dim, e.nc, e.rows, e.cols, v3, e.variant, e.bn, e.bk, e.rate
            ));
        }
    }
    s
}

/// Read the artifact line of an existing profile, so the tuner can refuse to
/// overwrite a profile that belongs to a different artifact without --force.
pub fn existing_artifact(path: &str) -> Option<u64> {
    let text = fs::read_to_string(path).ok()?;
    for line in text.lines() {
        if let Some(hex) = line.strip_prefix("artifact ") {
            return u64::from_str_radix(hex.trim(), 16).ok();
        }
    }
    None
}

/// Write through a temporary file and rename, so a reader never sees a partial
/// profile and an interrupted sweep leaves the previous one intact.
pub fn write_atomic(path: &str, text: &str) -> std::io::Result<()> {
    let tmp = format!("{path}.tmp");
    {
        let mut f = fs::File::create(&tmp)?;
        f.write_all(text.as_bytes())?;
        f.sync_all()?;
    }
    fs::rename(&tmp, path)
}
