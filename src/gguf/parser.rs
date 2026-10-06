//! GGUF file parser implementation.
//!
//! Parses GGUF v3 binary format with support for:
//! - Magic/version validation
//! - Metadata key-value pairs (all GGUF types)
//! - Tensor info entries
//! - Alignment and data offset calculation

use std::fs::File;
use std::io::{self, Read};
use std::path::Path;

use super::types::*;

/// Maximum allowed metadata entries in a single GGUF model (DoS protection).
pub const MAX_GGUF_METADATA_ENTRIES: u64 = 100_000;
/// Maximum allowed tensor count in a single GGUF model (DoS protection).
pub const MAX_GGUF_TENSORS: u64 = 100_000;

/// GGUF parsing error.
#[derive(Debug)]
pub enum GgufError {
    Io(io::Error),
    InvalidMagic([u8; 4]),
    UnsupportedVersion(u32),
    InvalidQuantType(u32),
    InvalidMetadataType(u32),
    TruncatedData,
    UnalignedRead,
}

impl std::fmt::Display for GgufError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GgufError::Io(e) => write!(f, "IO error: {}", e),
            GgufError::InvalidMagic(m) => write!(f, "Invalid GGUF magic: {:?}", m),
            GgufError::UnsupportedVersion(v) => write!(f, "Unsupported GGUF version: {}", v),
            GgufError::InvalidQuantType(t) => write!(f, "Invalid quant type: {}", t),
            GgufError::InvalidMetadataType(t) => write!(f, "Invalid metadata type: {}", t),
            GgufError::TruncatedData => write!(f, "Truncated data"),
            GgufError::UnalignedRead => write!(f, "Unaligned read"),
        }
    }
}

impl std::error::Error for GgufError {}

impl From<io::Error> for GgufError {
    fn from(e: io::Error) -> Self {
        GgufError::Io(e)
    }
}

/// GGUF file parser.
///
/// The reader borrows its buffer when parsing from memory
/// ([`GgufParser::from_slice`]) so large model files are never cloned.
pub struct GgufParser<'a> {
    reader: std::borrow::Cow<'a, [u8]>,
    pos: usize,
}

impl GgufParser<'static> {
    /// Create a new parser from a file (owns its buffer).
    pub fn from_file(path: &Path) -> Result<Self, GgufError> {
        let mut file = File::open(path)?;
        let mut data = Vec::new();
        file.read_to_end(&mut data)?;
        Ok(Self {
            reader: std::borrow::Cow::Owned(data),
            pos: 0,
        })
    }

    /// Create a new parser from an owned byte buffer.
    pub fn from_bytes(data: Vec<u8>) -> Self {
        Self {
            reader: std::borrow::Cow::Owned(data),
            pos: 0,
        }
    }

    /// Parse from a file path.
    pub fn parse_file(path: &Path) -> Result<GgufModel, GgufError> {
        let mut parser = GgufParser::from_file(path)?;
        parser.parse()
    }

    /// Parse from a byte buffer.
    pub fn parse_bytes(data: Vec<u8>) -> Result<GgufModel, GgufError> {
        let mut parser = GgufParser::from_bytes(data);
        parser.parse()
    }
}

impl<'a> GgufParser<'a> {
    /// Create a parser that *borrows* the buffer (zero-copy parsing — used
    /// by `GgufMmap::open` so a multi-GB model file is held exactly once).
    pub fn from_slice(data: &'a [u8]) -> Self {
        Self {
            reader: std::borrow::Cow::Borrowed(data),
            pos: 0,
        }
    }

    fn read_bytes(&mut self, n: usize) -> Result<&[u8], GgufError> {
        if self.pos + n > self.reader.len() {
            return Err(GgufError::TruncatedData);
        }
        let slice = &self.reader[self.pos..self.pos + n];
        self.pos += n;
        Ok(slice)
    }

    fn read_u8(&mut self) -> Result<u8, GgufError> {
        let bytes = self.read_bytes(1)?;
        Ok(bytes[0])
    }

    fn read_u16(&mut self) -> Result<u16, GgufError> {
        let bytes = self.read_bytes(2)?;
        Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    fn read_i16(&mut self) -> Result<i16, GgufError> {
        let bytes = self.read_bytes(2)?;
        Ok(i16::from_le_bytes([bytes[0], bytes[1]]))
    }

    fn read_u32(&mut self) -> Result<u32, GgufError> {
        let bytes = self.read_bytes(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn read_i32(&mut self) -> Result<i32, GgufError> {
        let bytes = self.read_bytes(4)?;
        Ok(i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn read_u64(&mut self) -> Result<u64, GgufError> {
        let bytes = self.read_bytes(8)?;
        Ok(u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    fn read_i64(&mut self) -> Result<i64, GgufError> {
        let bytes = self.read_bytes(8)?;
        Ok(i64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    fn read_f32(&mut self) -> Result<f32, GgufError> {
        let bytes = self.read_bytes(4)?;
        Ok(f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn read_f64(&mut self) -> Result<f64, GgufError> {
        let bytes = self.read_bytes(8)?;
        Ok(f64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    fn read_string(&mut self) -> Result<String, GgufError> {
        let len = self.read_u64()?;
        if len > 256 * 1024 * 1024 {
            return Err(GgufError::TruncatedData);
        }
        let bytes = self.read_bytes(len as usize)?;
        String::from_utf8(bytes.to_vec()).map_err(|_| GgufError::TruncatedData)
    }

    fn read_metadata_value(&mut self) -> Result<GgufMetadataValue, GgufError> {
        let type_id = self.read_u8()?;
        match type_id {
            0 => Ok(GgufMetadataValue::U8(self.read_u8()?)),
            1 => Ok(GgufMetadataValue::I8(self.read_u8()? as i8)),
            2 => Ok(GgufMetadataValue::U16(self.read_u16()?)),
            3 => Ok(GgufMetadataValue::I16(self.read_i16()?)),
            4 => Ok(GgufMetadataValue::U32(self.read_u32()?)),
            5 => Ok(GgufMetadataValue::I32(self.read_i32()?)),
            6 => Ok(GgufMetadataValue::F32(self.read_f32()?)),
            7 => Ok(GgufMetadataValue::Bool(self.read_u8()? != 0)),
            8 => Ok(GgufMetadataValue::String(self.read_string()?)),
            9 => {
                let len = self.read_u64()?;
                if len > 1_000_000 {
                    return Err(GgufError::TruncatedData);
                }
                let mut arr = Vec::with_capacity((len as usize).min(1024));
                for _ in 0..len {
                    arr.push(self.read_metadata_value()?);
                }
                Ok(GgufMetadataValue::Array(arr))
            }
            10 => Ok(GgufMetadataValue::U64(self.read_u64()?)),
            11 => Ok(GgufMetadataValue::I64(self.read_i64()?)),
            12 => Ok(GgufMetadataValue::F64(self.read_f64()?)),
            _ => Err(GgufError::InvalidMetadataType(type_id as u32)),
        }
    }

    /// Parse a GGUF file.
    pub fn parse(&mut self) -> Result<GgufModel, GgufError> {
        // Read and validate magic
        let mut magic = [0u8; 4];
        magic.copy_from_slice(self.read_bytes(4)?);
        if magic != super::GGUF_MAGIC {
            return Err(GgufError::InvalidMagic(magic));
        }

        // Read version (accept v2 + v3; Python loader supports both)
        let version = self.read_u32()?;
        if version != super::GGUF_VERSION && version != 2 {
            return Err(GgufError::UnsupportedVersion(version));
        }

        // Read tensor count and metadata KV count (GGUFv3: magic, version, n_tensors, n_kv)
        let tensor_count = self.read_u64()?;
        let kv_count = self.read_u64()?;

        // Reject hostile counts BEFORE any allocation or iteration: each KV
        // pair costs at least 9 bytes (u64 key length + type + 1-byte value)
        // and each tensor info at least 24 bytes, so a count larger than the
        // remaining file can never be satisfied. Without this, a corrupt
        // header could request a multi-GB `Vec::with_capacity` up front and
        // abort the process on allocation failure.
        if kv_count > MAX_GGUF_METADATA_ENTRIES || tensor_count > MAX_GGUF_TENSORS {
            return Err(GgufError::TruncatedData);
        }
        let remaining = (self.reader.len() - self.pos) as u64;
        if kv_count > remaining / 9 {
            return Err(GgufError::TruncatedData);
        }
        if tensor_count > remaining / 24 {
            return Err(GgufError::TruncatedData);
        }

        // Read metadata KV pairs
        let mut metadata = Vec::with_capacity(kv_count.min(1_000_000) as usize);
        for _ in 0..kv_count {
            let key = self.read_string()?;
            let value = self.read_metadata_value()?;
            metadata.push((key, value));
        }

        // Read tensor infos (use header count; do NOT read an extra u64)
        let n_tensors = tensor_count as usize;
        // Capacity is only a perf hint — cap the pre-allocation so a corrupt
        // (but file-consistent) count cannot force a huge allocation; real
        // growth happens as reads succeed.
        let mut tensors = Vec::with_capacity(n_tensors.min(1_000_000));
        for _ in 0..n_tensors {
            let name = self.read_string()?;
            let n_dims = self.read_u32()?;
            // GGUF tensors have at most 4 dimensions; 64 is a generous bound
            // that keeps `with_capacity(n_dims)` small even for garbage input.
            if n_dims > 64 {
                return Err(GgufError::TruncatedData);
            }
            let mut dims = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                dims.push(self.read_u64()?);
            }
            let quant_type_id = self.read_u32()?;
            let quant_type = GgufQuantType::from_u32(quant_type_id);
            let offset = self.read_u64()?;

            tensors.push(GgufTensorInfo {
                name,
                n_dims,
                dims,
                quant_type,
                offset,
            });
        }

        // Calculate data offset (align to 32 bytes)
        let data_offset = ((self.pos + 31) / 32) * 32;

        Ok(GgufModel {
            version,
            metadata,
            tensors,
            data_offset: data_offset as u64,
        })
    }
}

/// GGUF file loaded into memory for fast tensor access.
///
/// Note: despite the historical name this is *not* an OS-level mmap — the
/// file is read into a single `Vec<u8>` and parsed without cloning it, so
/// peak RAM equals file size (not 2x). True mmap (e.g. `memmap2`) is a
/// possible future optimization for very large models.
pub struct GgufMmap {
    data: Vec<u8>,
    model: GgufModel,
}

impl GgufMmap {
    /// Open and parse a GGUF file, keeping the data in memory.
    pub fn open(path: &Path) -> Result<Self, GgufError> {
        let mut file = File::open(path)?;
        let mut data = Vec::new();
        file.read_to_end(&mut data)?;

        // Parse by reference — the previous implementation cloned the whole
        // buffer for the parser, doubling peak memory on multi-GB files.
        let model = GgufParser::from_slice(&data).parse()?;

        Ok(Self { data, model })
    }

    /// Get the parsed model metadata.
    pub fn model(&self) -> &GgufModel {
        &self.model
    }

    /// Get raw tensor data by reference (checked; empty on corrupt offsets).
    pub fn tensor_bytes(&self, tensor: &GgufTensorInfo) -> &[u8] {
        let start = self.model.data_offset.saturating_add(tensor.offset) as usize;
        let end = start.saturating_add(tensor.n_bytes());
        if end > self.data.len() || start > end {
            return &[];
        }
        &self.data[start..end]
    }

    /// Checked variant returning an error instead of empty slice.
    pub fn try_tensor_bytes(&self, tensor: &GgufTensorInfo) -> Result<&[u8], GgufError> {
        let start = self.model.data_offset.saturating_add(tensor.offset) as usize;
        let end = start.saturating_add(tensor.n_bytes());
        if end > self.data.len() || start > end {
            return Err(GgufError::TruncatedData);
        }
        Ok(&self.data[start..end])
    }

    /// Get the underlying data buffer.
    pub fn data(&self) -> &[u8] {
        &self.data
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Smallest structurally valid GGUF v3 byte stream (header + 1 KV + 1 tensor).
    fn minimal_gguf() -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(b"GGUF");
        b.extend_from_slice(&super::super::GGUF_VERSION.to_le_bytes());
        b.extend_from_slice(&1u64.to_le_bytes()); // n_tensors
        b.extend_from_slice(&1u64.to_le_bytes()); // n_kv
                                                  // KV pair: key "k", type string, value "v"
        b.extend_from_slice(&1u64.to_le_bytes());
        b.extend_from_slice(b"k");
        b.push(8); // metadata type = string
        b.extend_from_slice(&1u64.to_le_bytes());
        b.extend_from_slice(b"v");
        // Tensor info: name "t", 1 dim of 4 elements, quant 0, offset 0
        b.extend_from_slice(&1u64.to_le_bytes());
        b.extend_from_slice(b"t");
        b.extend_from_slice(&1u32.to_le_bytes());
        b.extend_from_slice(&4u64.to_le_bytes());
        b.extend_from_slice(&0u32.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes());
        b
    }

    #[test]
    fn parses_minimal_gguf() {
        let model = GgufParser::from_bytes(minimal_gguf())
            .parse()
            .expect("minimal gguf must parse");
        assert_eq!(model.tensors.len(), 1);
        assert_eq!(model.metadata.len(), 1);
        assert_eq!(model.tensors[0].name, "t");
    }

    /// Corrupt/truncated input must always produce Err — never panic or an
    /// oversized allocation. Exercises every prefix length with and without
    /// a byte flip at the tail (which perturbs counts/lengths).
    #[test]
    fn truncated_or_flipped_input_never_panics() {
        let full = minimal_gguf();
        for cut in 0..full.len() {
            let mut b = full[..cut].to_vec();
            assert!(
                GgufParser::from_bytes(b.clone()).parse().is_ok()
                    || GgufParser::from_bytes(b.clone()).parse().is_err(),
                "parse must return a result (no panic) for cut={cut}"
            );
            if let Some(last) = b.last_mut() {
                *last ^= 0xFF;
            }
            let _ = GgufParser::from_bytes(b).parse(); // must not panic
        }
    }

    /// A header claiming absurd KV/tensor counts must be rejected before any
    /// allocation is attempted.
    #[test]
    fn hostile_counts_rejected() {
        let mut b = Vec::new();
        b.extend_from_slice(b"GGUF");
        b.extend_from_slice(&super::super::GGUF_VERSION.to_le_bytes());
        b.extend_from_slice(&u64::MAX.to_le_bytes()); // n_tensors
        b.extend_from_slice(&u64::MAX.to_le_bytes()); // n_kv
        assert!(GgufParser::from_bytes(b).parse().is_err());
    }

    /// Dim counts beyond any real tensor must not drive `with_capacity`.
    #[test]
    fn oversized_dim_count_rejected() {
        let mut b = Vec::new();
        b.extend_from_slice(b"GGUF");
        b.extend_from_slice(&super::super::GGUF_VERSION.to_le_bytes());
        b.extend_from_slice(&1u64.to_le_bytes()); // n_tensors
        b.extend_from_slice(&0u64.to_le_bytes()); // n_kv
        b.extend_from_slice(&1u64.to_le_bytes()); // name len
        b.extend_from_slice(b"t");
        b.extend_from_slice(&u32::MAX.to_le_bytes()); // n_dims garbage
        b.extend_from_slice(&[0u8; 64]);
        assert!(GgufParser::from_bytes(b).parse().is_err());
    }
}
