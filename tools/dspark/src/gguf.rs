//! Minimal GGUF v2/v3 reader: enough to load a draft checkpoint and print it.
//!
//! The format, per ggml's docs/gguf.md and the reference loader in
//! llama.cpp's gguf.cpp: a header (magic, version, tensor count, kv count), a
//! key/value table whose values are typed (including arrays), a tensor-info
//! table (name, dims, ggml type, offset), padding to the file's alignment
//! (general.alignment, default 32), then the tensor data at `data_start +
//! alignment_pad + offset`.
//!
//! Only the types a draft checkpoint uses are decoded into values; any other
//! type is skipped by width, which is how a reader stays forward-compatible
//! with metadata it does not care about.

use std::collections::BTreeMap;
use std::fs;

pub const GGUF_MAGIC: u32 = 0x4655_4747; // "GGUF"

#[derive(Clone, Debug)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    Array(Vec<Value>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl Value {
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Value::U8(v) => Some(*v as u64),
            Value::U16(v) => Some(*v as u64),
            Value::U32(v) => Some(*v as u64),
            Value::U64(v) => Some(*v),
            Value::I8(v) if *v >= 0 => Some(*v as u64),
            Value::I16(v) if *v >= 0 => Some(*v as u64),
            Value::I32(v) if *v >= 0 => Some(*v as u64),
            Value::I64(v) if *v >= 0 => Some(*v as u64),
            _ => None,
        }
    }
    pub fn as_f32(&self) -> Option<f32> {
        match self {
            Value::F32(v) => Some(*v),
            Value::F64(v) => Some(*v as f32),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }
    pub fn as_array(&self) -> Option<&[Value]> {
        match self {
            Value::Array(v) => Some(v),
            _ => None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub name: String,
    pub dims: Vec<u64>,
    pub dtype: u32,
    pub offset: u64,
}

impl TensorInfo {
    pub fn ne(&self) -> u64 {
        self.dims.iter().product()
    }
}

pub struct Gguf {
    pub version: u32,
    pub kv: BTreeMap<String, Value>,
    pub tensors: Vec<TensorInfo>,
    pub alignment: u64,
    pub data_start: u64,
    pub path: String,
    pub file_len: u64,
    bytes: Vec<u8>,
}

impl Gguf {
    pub fn open(path: &str) -> Result<Gguf, String> {
        let bytes = fs::read(path).map_err(|e| format!("{path}: {e}"))?;
        let mut r = Reader { b: &bytes, p: 0 };
        if r.u32()? != GGUF_MAGIC {
            return Err(format!("{path}: not a GGUF file"));
        }
        let version = r.u32()?;
        if !(2..=3).contains(&version) {
            return Err(format!("{path}: GGUF version {version} is not supported"));
        }
        let n_tensors = r.u64()?;
        let n_kv = r.u64()?;
        let mut kv = BTreeMap::new();
        for _ in 0..n_kv {
            let key = r.string()?;
            let ty = r.u32()?;
            let val = read_value(&mut r, ty)?;
            kv.insert(key, val);
        }
        let mut tensors = Vec::with_capacity(n_tensors as usize);
        for _ in 0..n_tensors {
            let name = r.string()?;
            let n_dims = r.u32()?;
            let mut dims = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                dims.push(r.u64()?);
            }
            let dtype = r.u32()?;
            let offset = r.u64()?;
            tensors.push(TensorInfo {
                name,
                dims,
                dtype,
                offset,
            });
        }
        let alignment = kv
            .get("general.alignment")
            .and_then(|v| v.as_u64())
            .unwrap_or(32);
        let data_start = align_up(r.p as u64, alignment);
        if data_start > bytes.len() as u64 {
            return Err(format!("{path}: tensor table runs past the file"));
        }
        Ok(Gguf {
            version,
            kv,
            tensors,
            alignment,
            data_start,
            path: path.to_string(),
            file_len: bytes.len() as u64,
            bytes,
        })
    }

    pub fn get_u64(&self, key: &str) -> Option<u64> {
        self.kv.get(key).and_then(|v| v.as_u64())
    }
    pub fn get_f32(&self, key: &str) -> Option<f32> {
        self.kv.get(key).and_then(|v| v.as_f32())
    }
    pub fn get_str(&self, key: &str) -> Option<&str> {
        self.kv.get(key).and_then(|v| v.as_str())
    }
    pub fn get_arr(&self, key: &str) -> Option<&[Value]> {
        self.kv.get(key).and_then(|v| v.as_array())
    }

    /// Tensor data as a byte slice into the file, checked against the file
    /// length so a truncated download is an error, not a wild read.
    pub fn tensor_bytes(&self, t: &TensorInfo) -> Result<&[u8], String> {
        let n = type_size(t.dtype, t.ne())?;
        let start = self.data_start + t.offset;
        let end = start.checked_add(n).ok_or("tensor offset overflow")?;
        if end > self.bytes.len() as u64 {
            return Err(format!(
                "{}: tensor {} needs bytes {start}..{end}, file is {}",
                self.path,
                t.name,
                self.bytes.len()
            ));
        }
        Ok(&self.bytes[start as usize..end as usize])
    }
}

fn align_up(v: u64, a: u64) -> u64 {
    if a <= 1 {
        v
    } else {
        v.div_ceil(a) * a
    }
}

/// Bytes one tensor occupies, for the ggml types a draft checkpoint uses.
/// Unsupported types are refused by name rather than guessed.
pub fn type_size(dtype: u32, ne: u64) -> Result<u64, String> {
    let (block, bytes) = match dtype {
        0 => (1, 4),   // F32
        1 => (1, 2),   // F16
        30 => (1, 2),  // BF16
        2 => (32, 18), // Q4_0
        3 => (32, 20), // Q4_1
        8 => (32, 34), // Q8_0
        7 => (32, 34), // Q5_1
        6 => (32, 22), // Q5_0
        other => return Err(format!("ggml type {other} is not supported by this reader")),
    };
    if !ne.is_multiple_of(block) {
        return Err(format!(
            "{ne} elements is not a whole number of {block}-element blocks"
        ));
    }
    Ok(ne / block * bytes)
}

pub fn type_name(dtype: u32) -> &'static str {
    match dtype {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        6 => "Q5_0",
        7 => "Q5_1",
        8 => "Q8_0",
        30 => "BF16",
        _ => "other",
    }
}

struct Reader<'a> {
    b: &'a [u8],
    p: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        let end = self.p.checked_add(n).ok_or("read overflow")?;
        if end > self.b.len() {
            return Err("unexpected end of file".into());
        }
        let s = &self.b[self.p..end];
        self.p = end;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8, String> {
        Ok(self.take(1)?[0])
    }
    fn i8(&mut self) -> Result<i8, String> {
        Ok(self.u8()? as i8)
    }
    fn u16(&mut self) -> Result<u16, String> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn i16(&mut self) -> Result<i16, String> {
        Ok(i16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn i32(&mut self) -> Result<i32, String> {
        Ok(i32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn i64(&mut self) -> Result<i64, String> {
        Ok(i64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn f32(&mut self) -> Result<f32, String> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn f64(&mut self) -> Result<f64, String> {
        Ok(f64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn string(&mut self) -> Result<String, String> {
        let n = self.u64()? as usize;
        let raw = self.take(n)?;
        Ok(String::from_utf8_lossy(raw).into_owned())
    }
}

const TYPE_U8: u32 = 0;
const TYPE_I8: u32 = 1;
const TYPE_U16: u32 = 2;
const TYPE_I16: u32 = 3;
const TYPE_U32: u32 = 4;
const TYPE_I32: u32 = 5;
const TYPE_F32: u32 = 6;
const TYPE_BOOL: u32 = 7;
const TYPE_STRING: u32 = 8;
const TYPE_ARRAY: u32 = 9;
const TYPE_U64: u32 = 10;
const TYPE_I64: u32 = 11;
const TYPE_F64: u32 = 12;

fn read_value(r: &mut Reader, ty: u32) -> Result<Value, String> {
    Ok(match ty {
        TYPE_U8 => Value::U8(r.u8()?),
        TYPE_I8 => Value::I8(r.i8()?),
        TYPE_U16 => Value::U16(r.u16()?),
        TYPE_I16 => Value::I16(r.i16()?),
        TYPE_U32 => Value::U32(r.u32()?),
        TYPE_I32 => Value::I32(r.i32()?),
        TYPE_F32 => Value::F32(r.f32()?),
        TYPE_BOOL => Value::Bool(r.u8()? != 0),
        TYPE_STRING => Value::String(r.string()?),
        TYPE_ARRAY => {
            let elem = r.u32()?;
            let n = r.u64()?;
            // A count that cannot fit the remaining bytes is corruption, not
            // an invitation to allocate.
            if n > (r.b.len() - r.p) as u64 + 1 && elem != TYPE_ARRAY {
                return Err(format!("array of {n} elements does not fit the file"));
            }
            let mut v = Vec::with_capacity(n.min(1 << 20) as usize);
            for _ in 0..n {
                v.push(read_value(r, elem)?);
            }
            Value::Array(v)
        }
        TYPE_U64 => Value::U64(r.u64()?),
        TYPE_I64 => Value::I64(r.i64()?),
        TYPE_F64 => Value::F64(r.f64()?),
        other => return Err(format!("metadata value type {other} is not supported")),
    })
}
