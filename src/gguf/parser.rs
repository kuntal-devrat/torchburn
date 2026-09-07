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
pub struct GgufParser {
    reader: Vec<u8>,
    pos: usize,
}

impl GgufParser {
    /// Create a new parser from a file.
    pub fn from_file(path: &Path) -> Result<Self, GgufError> {
        let mut file = File::open(path)?;
        let mut data = Vec::new();
        file.read_to_end(&mut data)?;
        Ok(Self { reader: data, pos: 0 })
    }

    /// Create a new parser from a byte buffer.
    pub fn from_bytes(data: Vec<u8>) -> Self {
        Self { reader: data, pos: 0 }
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
            bytes[0], bytes[1], bytes[2], bytes[3],
            bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    fn read_i64(&mut self) -> Result<i64, GgufError> {
        let bytes = self.read_bytes(8)?;
        Ok(i64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3],
            bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    fn read_f32(&mut self) -> Result<f32, GgufError> {
        let bytes = self.read_bytes(4)?;
        Ok(f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn read_f64(&mut self) -> Result<f64, GgufError> {
        let bytes = self.read_bytes(8)?;
        Ok(f64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3],
            bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    fn read_string(&mut self) -> Result<String, GgufError> {
        let len = self.read_u64()? as usize;
        let bytes = self.read_bytes(len)?;
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
                let len = self.read_u64()? as usize;
                let mut arr = Vec::with_capacity(len);
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

        // Read version
        let version = self.read_u32()?;
        if version != super::GGUF_VERSION {
            return Err(GgufError::UnsupportedVersion(version));
        }

        // Read tensor count and metadata KV count
        let _tensor_count = self.read_u64()?;
        let kv_count = self.read_u64()?;

        // Read metadata KV pairs
        let mut metadata = Vec::with_capacity(kv_count as usize);
        for _ in 0..kv_count {
            let key = self.read_string()?;
            let value = self.read_metadata_value()?;
            metadata.push((key, value));
        }

        // Read tensor infos
        let n_tensors = self.read_u64()? as usize;
        let mut tensors = Vec::with_capacity(n_tensors);
        for _ in 0..n_tensors {
            let name = self.read_string()?;
            let n_dims = self.read_u32()?;
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

    /// Parse from a file path.
    pub fn parse_file(path: &Path) -> Result<GgufModel, GgufError> {
        let mut parser = Self::from_file(path)?;
        parser.parse()
    }

    /// Parse from a byte buffer.
    pub fn parse_bytes(data: Vec<u8>) -> Result<GgufModel, GgufError> {
        let mut parser = Self::from_bytes(data);
        parser.parse()
    }
}

/// Memory-mapped GGUF file for zero-copy tensor access.
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

        let model = {
            let mut parser = GgufParser::from_bytes(data.clone());
            parser.parse()?
        };

        Ok(Self { data, model })
    }

    /// Get the parsed model metadata.
    pub fn model(&self) -> &GgufModel {
        &self.model
    }

    /// Get raw tensor data by reference.
    pub fn tensor_bytes(&self, tensor: &GgufTensorInfo) -> &[u8] {
        let start = (self.model.data_offset + tensor.offset) as usize;
        let end = start + tensor.n_bytes();
        &self.data[start..end]
    }

    /// Get the underlying data buffer.
    pub fn data(&self) -> &[u8] {
        &self.data
    }
}
