//! GGUF type definitions.

/// GGUF metadata value types.
#[derive(Debug, Clone)]
pub enum GgufMetadataValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    Array(Vec<GgufMetadataValue>),
    U64(u64),
    I64(i64),
    F64(f64),
}

/// GGUF quantization types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
#[allow(non_camel_case_types)]
pub enum GgufQuantType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    IQ2_XXS = 16,
    IQ2_XS = 17,
    IQ3_XXS = 18,
    IQ1_S = 19,
    IQ4_NL = 20,
    IQ3_S = 21,
    IQ2_S = 22,
    IQ4_XS = 23,
    Unknown = 255,
}

impl GgufQuantType {
    pub fn from_u32(v: u32) -> Self {
        match v {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2_K,
            11 => Self::Q3_K,
            12 => Self::Q4_K,
            13 => Self::Q5_K,
            14 => Self::Q6_K,
            15 => Self::Q8_K,
            16 => Self::IQ2_XXS,
            17 => Self::IQ2_XS,
            18 => Self::IQ3_XXS,
            19 => Self::IQ1_S,
            20 => Self::IQ4_NL,
            21 => Self::IQ3_S,
            22 => Self::IQ2_S,
            23 => Self::IQ4_XS,
            _ => Self::Unknown,
        }
    }

    /// Block size in elements for this quant type.
    pub fn block_size(&self) -> usize {
        match self {
            Self::F32 => 1,
            Self::F16 => 1,
            Self::Q4_0 | Self::Q4_1 => 32,
            Self::Q5_0 | Self::Q5_1 => 32,
            Self::Q8_0 | Self::Q8_1 => 32,
            Self::Q2_K => 256,
            Self::Q3_K => 256,
            Self::Q4_K => 256,
            Self::Q5_K => 256,
            Self::Q6_K => 256,
            Self::Q8_K => 256,
            _ => 32,
        }
    }

    /// Bytes per block.
    pub fn bytes_per_block(&self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
            Self::Q4_0 => 16 + 2,     // 32 nibbles + f16 scale
            Self::Q4_1 => 16 + 2 + 2, // 32 nibbles + f16 min + f16 scale
            Self::Q8_0 => 32 + 2,     // 32 bytes + f16 scale
            Self::Q8_1 => 32 + 2 + 2,
            Self::Q4_K => 144, // 256 weights, complex layout
            Self::Q6_K => 210,
            Self::Q8_K => 256 + 16, // 256 bytes + scales
            _ => 0,
        }
    }

    /// Whether this quant type can be directly mapped to a TorchBurn kernel.
    pub fn has_native_support(&self) -> bool {
        matches!(
            self,
            Self::Q4_0 | Self::Q4_1 | Self::Q8_0 | Self::F32 | Self::F16
        )
    }

    /// Map to TorchBurn's native quant type if supported.
    pub fn to_native_quant(&self) -> Option<NativeQuantType> {
        match self {
            Self::Q4_0 | Self::Q4_1 | Self::Q4_K => Some(NativeQuantType::W4A32),
            Self::Q8_0 | Self::Q8_1 | Self::Q8_K => Some(NativeQuantType::W8A32),
            Self::F32 => Some(NativeQuantType::F32),
            Self::F16 => Some(NativeQuantType::F16),
            _ => None,
        }
    }
}

/// TorchBurn native quantization types for mapping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeQuantType {
    F32,
    F16,
    W4A32,
    W8A32,
}

/// GGUF tensor metadata.
#[derive(Debug, Clone)]
pub struct GgufTensorInfo {
    pub name: String,
    pub n_dims: u32,
    pub dims: Vec<u64>,
    pub quant_type: GgufQuantType,
    pub offset: u64,
}

impl GgufTensorInfo {
    /// Total number of elements.
    pub fn n_elements(&self) -> usize {
        self.dims.iter().product::<u64>() as usize
    }

    /// Total number of bytes in the quantized data.
    pub fn n_bytes(&self) -> usize {
        let n_blocks = self.n_elements() / self.quant_type.block_size();
        n_blocks * self.quant_type.bytes_per_block()
    }
}

/// Parsed GGUF model.
#[derive(Debug)]
pub struct GgufModel {
    pub version: u32,
    pub metadata: Vec<(String, GgufMetadataValue)>,
    pub tensors: Vec<GgufTensorInfo>,
    pub data_offset: u64,
}

impl GgufModel {
    /// Look up a metadata value by key.
    pub fn get_metadata(&self, key: &str) -> Option<&GgufMetadataValue> {
        self.metadata.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    /// Get the architecture string (e.g., "llama", "qwen2").
    pub fn architecture(&self) -> Option<&str> {
        self.get_metadata("general.architecture").and_then(|v| {
            if let GgufMetadataValue::String(s) = v {
                Some(s.as_str())
            } else {
                None
            }
        })
    }

    /// Get the model name.
    pub fn name(&self) -> Option<&str> {
        self.get_metadata("general.name").and_then(|v| {
            if let GgufMetadataValue::String(s) = v {
                Some(s.as_str())
            } else {
                None
            }
        })
    }

    /// Find tensor by name.
    pub fn find_tensor(&self, name: &str) -> Option<&GgufTensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// List all tensors with a given prefix (e.g., "blk.0.attn_q").
    pub fn tensors_with_prefix(&self, prefix: &str) -> Vec<&GgufTensorInfo> {
        self.tensors
            .iter()
            .filter(|t| t.name.starts_with(prefix))
            .collect()
    }

    /// Get tensor data from a memory-mapped file.
    pub fn tensor_data<'a>(&self, tensor: &GgufTensorInfo, mmap: &'a [u8]) -> &'a [u8] {
        let start = (self.data_offset + tensor.offset) as usize;
        let end = start + tensor.n_bytes();
        &mmap[start..end]
    }
}
