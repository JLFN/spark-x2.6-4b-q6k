//! dspark check: validate a converted draft directory.
//!
//! The check is offline and needs nothing but OUTDIR: it re-derives every shape
//! from the manifest's own header, verifies that the tensor regions tile
//! weights.bin with nothing but alignment padding between them, and reads those
//! padding bytes to prove they are zero. A truncated tensor, an edited offset, a
//! missing required role or a shape the header contradicts is refused with the
//! tensor's name.

use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use crate::convert::capabilities_of;
use crate::manifest::{dtype_name, Manifest, Tensor, ALIGN};

pub fn check(dir: &str) -> Result<(), String> {
    let dp = Path::new(dir);
    let manifest_path = dp.join("manifest.txt");
    let weights_path = dp.join("weights.bin");

    let text = fs::read_to_string(&manifest_path)
        .map_err(|e| format!("{}: {e}", manifest_path.display()))?;
    let m = Manifest::parse(&text).map_err(|e| format!("{}: {e}", manifest_path.display()))?;
    let len = fs::metadata(&weights_path)
        .map_err(|e| format!("{}: {e}", weights_path.display()))?
        .len();
    m.validate(len)
        .map_err(|e| format!("{}: {e}", manifest_path.display()))?;
    if m.header.arch != "dflash" {
        return Err(format!(
            "{}: architecture {} is not implemented",
            manifest_path.display(),
            m.header.arch
        ));
    }
    verify_padding(&weights_path, &m)?;

    let h = &m.header;
    let source: u64 = m.tensors.iter().map(|t| t.bytes).sum();
    println!("directory   {}", dp.display());
    println!(
        "arch        {}, {} blocks, {} hidden, {} ffn, {} heads / {} kv of {}",
        h.arch, h.layers, h.hidden, h.ffn, h.heads, h.kv_heads, h.head_dim
    );
    println!(
        "rope        full {} dims theta {}, sliding {} dims theta {}",
        h.rope_full_dims, h.rope_full_theta, h.rope_swa_dims, h.rope_swa_theta
    );
    println!("norm        eps {}, activation {}", h.eps, h.act);
    let swa = h.swa_pattern.iter().filter(|&&b| b).count();
    println!(
        "attention   {}-token window on {}/{} layers, causal {}, block {}, sample_from_anchor {}",
        h.swa_window,
        swa,
        h.layers,
        if h.causal { "on" } else { "off" },
        h.block_size,
        if h.sample_from_anchor { "on" } else { "off" }
    );
    println!("targets     layers {:?}", h.target_layers);
    println!("capabilities {}", capabilities_of(&m));
    println!(
        "weights     {} bytes, {} tensors, {} bytes of tensors, {} of alignment",
        len,
        m.tensors.len(),
        source,
        len - source
    );

    let mut order: Vec<&Tensor> = m.tensors.iter().collect();
    order.sort_by_key(|t| t.offset);
    println!(
        "\n  {:<16} {:>5} {:>5} {:>16} {:>12} {:>12}",
        "role", "layer", "dtype", "shape", "offset", "bytes"
    );
    for t in &order {
        println!(
            "  {:<16} {:>5} {:>5} {:>16} {:>12} {:>12}",
            t.role,
            t.layer,
            dtype_name(t.dtype),
            format!("{}x{}", t.ne0, t.ne1),
            t.offset,
            t.bytes
        );
    }
    println!("\nok");
    Ok(())
}

/// The layout check proves the gaps are smaller than one alignment step; this
/// proves the bytes inside them are padding and not the tail of a weight.
fn verify_padding(path: &Path, m: &Manifest) -> Result<(), String> {
    let mut order: Vec<&Tensor> = m.tensors.iter().collect();
    order.sort_by_key(|t| t.offset);
    let mut f = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    for pair in order.windows(2) {
        let (a, b) = (pair[0], pair[1]);
        let gap = b.offset - a.end();
        if gap == 0 {
            continue;
        }
        f.seek(SeekFrom::Start(a.end()))
            .map_err(|e| format!("{}: {e}", path.display()))?;
        let mut buf = [0u8; ALIGN as usize];
        f.read_exact(&mut buf[..gap as usize])
            .map_err(|e| format!("{}: {e}", path.display()))?;
        if buf[..gap as usize].iter().any(|&x| x != 0) {
            return Err(format!(
                "{}: the {gap}-byte alignment padding after tensor {} is not zero",
                path.display(),
                a.label()
            ));
        }
    }
    Ok(())
}
