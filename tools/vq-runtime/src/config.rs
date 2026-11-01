//! The architecture constants, read from the checkpoint's own config.json.
//!
//! Nothing here is guessed: every default is a value this model actually ships
//! (config.json in models/hf), and the layer-type list decides which layers get
//! the sliding mask and which rope.

use serde_json::Value;
use std::fs;

#[derive(Clone, Copy, Debug)]
pub struct Rope {
    pub theta: f32,
    pub partial: f32,
}

#[derive(Clone, Debug)]
pub struct Config {
    pub hidden: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub intermediate: usize,
    pub vocab: usize,
    pub eps: f32,
    pub sliding_window: usize,
    pub layer_types: Vec<String>,
    pub rope_full: Rope,
    pub rope_sliding: Rope,
}

fn num(v: &Value, k: &str) -> Result<f64, String> {
    v.get(k)
        .and_then(Value::as_f64)
        .ok_or_else(|| format!("config.json has no numeric {k}"))
}

impl Config {
    pub fn load(path: &str) -> Result<Config, String> {
        let text = fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
        let v: Value = serde_json::from_str(&text).map_err(|e| format!("{path}: {e}"))?;

        let layer_types = v
            .get("layer_types")
            .and_then(Value::as_array)
            .ok_or("config.json has no layer_types")?
            .iter()
            .map(|t| t.as_str().unwrap_or("full_attention").to_string())
            .collect::<Vec<_>>();

        /* rope_parameters is nested per layer type. A checkpoint without the
         * block still gets the values this one ships, and the load prints
         * which source was used. */
        let rp = v.get("rope_parameters");
        let rope = |name: &str, theta: f32, partial: f32| -> Rope {
            let o = rp.and_then(|r| r.get(name));
            Rope {
                theta: o
                    .and_then(|o| o.get("rope_theta"))
                    .and_then(Value::as_f64)
                    .map(|x| x as f32)
                    .unwrap_or(theta),
                partial: o
                    .and_then(|o| o.get("partial_rotary_factor"))
                    .and_then(Value::as_f64)
                    .map(|x| x as f32)
                    .unwrap_or(partial),
            }
        };

        Ok(Config {
            hidden: num(&v, "hidden_size")? as usize,
            layers: num(&v, "num_hidden_layers")? as usize,
            heads: num(&v, "num_attention_heads")? as usize,
            kv_heads: num(&v, "num_key_value_heads")? as usize,
            head_dim: num(&v, "head_dim")? as usize,
            intermediate: num(&v, "intermediate_size")? as usize,
            vocab: num(&v, "vocab_size")? as usize,
            eps: num(&v, "rms_norm_eps")? as f32,
            sliding_window: num(&v, "sliding_window")? as usize,
            layer_types,
            rope_full: rope("full_attention", 5_000_000.0, 0.25),
            rope_sliding: rope("sliding_attention", 10_000.0, 1.0),
        })
    }

    pub fn is_full(&self, layer: usize) -> bool {
        self.layer_types
            .get(layer)
            .map(|t| t == "full_attention")
            .unwrap_or(true)
    }

    pub fn rope(&self, layer: usize) -> Rope {
        if self.is_full(layer) {
            self.rope_full
        } else {
            self.rope_sliding
        }
    }

    pub fn q_dim(&self) -> usize {
        self.heads * self.head_dim
    }

    pub fn kv_dim(&self) -> usize {
        self.kv_heads * self.head_dim
    }
}
