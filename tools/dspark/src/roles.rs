//! The role vocabulary: the one place a GGUF tensor name becomes something the
//! executor can ask for.
//!
//! The names are taken from the reference implementation, not from this
//! checkpoint: llama.cpp fork XHToken/llama.cpp at 22c0ae1, the name table in
//! src/llama-arch.cpp and the shape table in
//! src/models/dflash.cpp::llama_model_dflash::load_arch_tensors. A second
//! dflash draft therefore converts with the same table, and a family this build
//! does not implement fails loudly by tensor name instead of being skipped
//! (which would change the proposals without changing the shape).
//!
//! Shapes are the *GGUF* shapes: ne0 is the fastest-varying dimension, so a
//! linear weight that multiplies an n_embd vector into n_out values is
//! {n_embd, n_out}. The reference's create_tensor calls state them that way.

use crate::manifest::{Header, Shape};

pub struct Role {
    pub name: &'static str,
    /// True when the role carries a block index in the manifest.
    pub per_layer: bool,
    /// True when a draft without it cannot run at all.
    pub required: bool,
    pub shape: Shape,
}

pub const ROLES: &[Role] = &[
    // Per block (src/models/dflash.cpp:326-350). attn_gate and the q/k norms
    // are the two mutually exclusive gate families the reference accepts.
    Role {
        name: "attn_norm",
        per_layer: true,
        required: true,
        shape: Shape::Norm,
    },
    Role {
        name: "attn_q",
        per_layer: true,
        required: true,
        shape: Shape::Q,
    },
    Role {
        name: "attn_k",
        per_layer: true,
        required: true,
        shape: Shape::Kv,
    },
    Role {
        name: "attn_v",
        per_layer: true,
        required: false,
        shape: Shape::Kv,
    },
    Role {
        name: "attn_out",
        per_layer: true,
        required: true,
        shape: Shape::Out,
    },
    Role {
        name: "attn_gate",
        per_layer: true,
        required: false,
        shape: Shape::Heads,
    },
    Role {
        name: "attn_q_norm",
        per_layer: true,
        required: false,
        shape: Shape::HeadDim,
    },
    Role {
        name: "attn_k_norm",
        per_layer: true,
        required: false,
        shape: Shape::HeadDim,
    },
    Role {
        name: "attn_post_norm",
        per_layer: true,
        required: false,
        shape: Shape::Norm,
    },
    Role {
        name: "ffn_norm",
        per_layer: true,
        required: true,
        shape: Shape::Norm,
    },
    Role {
        name: "ffn_gate",
        per_layer: true,
        required: true,
        shape: Shape::FfnGate,
    },
    Role {
        name: "ffn_up",
        per_layer: true,
        required: true,
        shape: Shape::FfnGate,
    },
    Role {
        name: "ffn_down",
        per_layer: true,
        required: true,
        shape: Shape::FfnDown,
    },
    Role {
        name: "ffn_post_norm",
        per_layer: true,
        required: false,
        shape: Shape::Norm,
    },
    Role {
        name: "layer_out_scale",
        per_layer: true,
        required: false,
        shape: Shape::Scalar,
    },
    // Global (src/models/dflash.cpp:239-255 and 137-149).
    Role {
        name: "fc",
        per_layer: false,
        required: true,
        shape: Shape::Fc,
    },
    Role {
        name: "fc_norm",
        per_layer: false,
        required: true,
        shape: Shape::FcNorm,
    },
    Role {
        name: "enc_norm",
        per_layer: false,
        required: true,
        shape: Shape::Norm,
    },
    Role {
        name: "output_norm",
        per_layer: false,
        required: true,
        shape: Shape::Norm,
    },
    Role {
        name: "markov_w1",
        per_layer: false,
        required: false,
        shape: Shape::Markov,
    },
    Role {
        name: "markov_w2",
        per_layer: false,
        required: false,
        shape: Shape::Markov,
    },
    Role {
        name: "conf_proj",
        per_layer: false,
        required: false,
        shape: Shape::ConfProj,
    },
    Role {
        name: "conf_bias",
        per_layer: false,
        required: false,
        shape: Shape::Scalar,
    },
];

pub fn spec(name: &str) -> Option<&'static Role> {
    ROLES.iter().find(|r| r.name == name)
}

/// Result of reading one GGUF tensor name.
pub enum Lookup {
    Mapped {
        role: &'static str,
        layer: i32,
    },
    /// A name the reference vocabulary knows, which this build deliberately
    /// does not implement. The reason is printed with the tensor name.
    Unimplemented(&'static str),
    /// Nothing in this build knows the name.
    Unknown,
}

/// Map a GGUF tensor name to (role, layer).
pub fn classify(name: &str) -> Lookup {
    if let Some(rest) = name.strip_prefix("blk.") {
        let Some((idx, suffix)) = rest.split_once('.') else {
            return Lookup::Unknown;
        };
        let Ok(layer) = idx.parse::<i32>() else {
            return Lookup::Unknown;
        };
        if let Some(role) = layer_role(suffix) {
            return Lookup::Mapped { role, layer };
        }
        if let Some(why) = layer_unimplemented(suffix) {
            return Lookup::Unimplemented(why);
        }
        return Lookup::Unknown;
    }
    if let Some(role) = global_role(name) {
        return Lookup::Mapped { role, layer: -1 };
    }
    if let Some(why) = global_unimplemented(name) {
        return Lookup::Unimplemented(why);
    }
    Lookup::Unknown
}

/// The suffix after "blk.<n>.". Names follow llama-arch.cpp exactly, including
/// `post_attention_norm` (not `attn_post_norm`) and `post_ffw_norm`.
fn layer_role(suffix: &str) -> Option<&'static str> {
    Some(match suffix {
        "attn_norm.weight" => "attn_norm",
        "attn_q.weight" => "attn_q",
        "attn_k.weight" => "attn_k",
        "attn_v.weight" => "attn_v",
        "attn_output.weight" => "attn_out",
        "attn_gate.weight" => "attn_gate",
        "attn_q_norm.weight" => "attn_q_norm",
        "attn_k_norm.weight" => "attn_k_norm",
        "post_attention_norm.weight" => "attn_post_norm",
        "ffn_norm.weight" => "ffn_norm",
        "ffn_gate.weight" => "ffn_gate",
        "ffn_up.weight" => "ffn_up",
        "ffn_down.weight" => "ffn_down",
        "post_ffw_norm.weight" => "ffn_post_norm",
        "layer_output_scale.weight" => "layer_out_scale",
        _ => return None,
    })
}

fn layer_unimplemented(suffix: &str) -> Option<&'static str> {
    Some(match suffix {
        "attn_conv_base" | "attn_conv_proj.weight" | "ffn_conv_base"
        | "ffn_conv_proj.weight" => {
            "dflash2 convolution tensors (the reference's conv path) are not implemented"
        }
        "attn_sinks.weight" => "attention sinks are not implemented",
        "rope_freqs.weight" => {
            "per-layer rope frequencies are not implemented; the draft uses the manifest's rope parameters"
        }
        "post_ffw_norm_1.weight" | "post_ffw_norm_2.weight" => {
            "the second and third feed-forward norms of this architecture variant are not implemented"
        }
        "attn_v_gate.weight" | "attn_v_exps.weight" => {
            "value-head gating and value experts are not implemented"
        }
        _ => return None,
    })
}

fn global_role(name: &str) -> Option<&'static str> {
    Some(match name {
        "fc.weight" => "fc",
        "fc_norm.weight" => "fc_norm",
        "enc.output_norm.weight" => "enc_norm",
        "output_norm.weight" => "output_norm",
        "markov_w1.weight" => "markov_w1",
        "markov_w2.weight" => "markov_w2",
        "conf_proj.weight" => "conf_proj",
        "conf_proj.bias" => "conf_bias",
        _ => return None,
    })
}

fn global_unimplemented(name: &str) -> Option<&'static str> {
    Some(match name {
        "markov_w2.scale" => "the Markov output scale is not implemented",
        "fc.scale" | "fc.input_scale" => "a quantized fc scale is not implemented",
        "d2t" => "a reduced draft vocabulary (d2t) is not implemented",
        "token_embd.weight" => {
            "this build borrows the target's token embedding; a draft with its own is not implemented"
        }
        "output.weight" => {
            "this build borrows the target's LM head; a draft with its own is not implemented"
        }
        "selector_predecessor.weight" | "selector_successor.weight"
        | "selector_hidden.weight" => "the dflash2 selector is not implemented",
        "token_embd_norm.weight" | "per_layer_token_embd.weight" => {
            "embedding-side extras are not implemented"
        }
        _ if name.starts_with("lilicorr.") => "the lilicorr head is not implemented",
        _ => return None,
    })
}

/// The expected (ne0, ne1) for a role, from the manifest's header.
/// `rank` is the Markov rank, read from markov_w1 when the head exists.
pub fn expected(shape: Shape, h: &Header, rank: Option<u64>) -> Result<(u64, u64), String> {
    let q = h.heads as u64 * h.head_dim;
    let kv = h.kv_heads as u64 * h.head_dim;
    let ntar = h.target_layers.len() as u64;
    Ok(match shape {
        Shape::Norm => (h.hidden, 1),
        Shape::Scalar => (1, 1),
        Shape::Q => (h.hidden, q),
        Shape::Kv => (h.hidden, kv),
        Shape::Out => (q, h.hidden),
        Shape::Heads => (h.hidden, h.heads as u64),
        Shape::HeadDim => (h.head_dim, 1),
        Shape::FfnGate => (h.hidden, h.ffn),
        Shape::FfnDown => (h.ffn, h.hidden),
        Shape::Fc => (ntar * h.hidden, h.hidden),
        Shape::FcNorm => (h.hidden, ntar),
        Shape::Markov => {
            let rank = rank.ok_or("a Markov tensor needs markov_w1 to define the rank")?;
            (rank, h.vocab)
        }
        Shape::ConfProj => {
            let rank = rank.ok_or("conf_proj needs markov_w1 to define the Markov rank")?;
            (h.hidden + rank, 1)
        }
    })
}
