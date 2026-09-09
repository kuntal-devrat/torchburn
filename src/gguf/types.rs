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
            4 => Self::Q4_0,
            5 => Self::Q4_1,
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
            Self::IQ4_NL | Self::IQ4_XS => 32,
            _ => 256,
        }
    }

    /// Bytes per block (llama.cpp canonical sizes; 0 = unsupported).
    pub fn bytes_per_block(&self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
            Self::Q4_0 => 16 + 2,
            Self::Q4_1 => 16 + 2 + 2,
            Self::Q5_0 => 16 + 4 + 2,
            Self::Q5_1 => 16 + 4 + 2 + 2,
            Self::Q8_0 => 32 + 2,
            Self::Q8_1 => 32 + 4 + 4,
            Self::Q2_K => 84,
            Self::Q3_K => 110,
            Self::Q4_K => 144,
            Self::Q5_K => 176,
            Self::Q6_K => 210,
            Self::Q8_K => 256 + 16,
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
    /// NOTE: K-quants have super-block scales and are NOT directly executable
    /// by W4A32/W8A32 kernels — they require explicit dequant.
    pub fn to_native_quant(&self) -> Option<NativeQuantType> {
        match self {
            Self::Q4_0 | Self::Q4_1 => Some(NativeQuantType::W4A32),
            Self::Q8_0 | Self::Q8_1 => Some(NativeQuantType::W8A32),
            Self::F32 => Some(NativeQuantType::F32),
            Self::F16 => Some(NativeQuantType::F16),
            _ => None,
        }
    }

    /// Expected raw bytes for `n_elem` elements (None if unsupported/overflow).
    /// Used by loaders to validate offsets before slicing (no panic).
    pub fn expected_bytes(&self, n_elem: usize) -> Option<usize> {
        let bpb = self.bytes_per_block();
        if bpb == 0 {
            return None;
        }
        let bs = self.block_size().max(1);
        n_elem.div_ceil(bs).checked_mul(bpb)
    }

    /// True if the type must be dequantized to F32/F16 on load (K-quants, IQ).
    /// Python `gguf_loader` should route these through dequant, not zeros.
    pub fn needs_dequant(&self) -> bool {
        matches!(
            self,
            Self::Q2_K
                | Self::Q3_K
                | Self::Q4_K
                | Self::Q5_K
                | Self::Q6_K
                | Self::Q8_K
                | Self::Q5_0
                | Self::Q5_1
                | Self::Q8_1
                | Self::IQ2_XXS
                | Self::IQ2_XS
                | Self::IQ3_XXS
                | Self::IQ1_S
                | Self::IQ4_NL
                | Self::IQ3_S
                | Self::IQ2_S
                | Self::IQ4_XS
        )
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
    /// Total number of elements (checked, saturating).
    pub fn n_elements(&self) -> usize {
        let mut acc: u64 = 1;
        for &d in &self.dims {
            acc = acc.saturating_mul(d);
            if acc > (isize::MAX as u64) {
                return isize::MAX as usize;
            }
        }
        acc as usize
    }

    /// Total number of bytes in the quantized data (checked, round up partial).
    pub fn n_bytes(&self) -> usize {
        let bpb = self.quant_type.bytes_per_block();
        if bpb == 0 {
            return 0;
        }
        let bs = self.quant_type.block_size().max(1);
        let n_blocks = self.n_elements().div_ceil(bs);
        n_blocks.saturating_mul(bpb)
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

    /// Get tensor data from a memory-mapped file (checked).
    pub fn tensor_data<'a>(&self, tensor: &GgufTensorInfo, mmap: &'a [u8]) -> &'a [u8] {
        let start = self.data_offset.saturating_add(tensor.offset) as usize;
        let end = start.saturating_add(tensor.n_bytes());
        if end > mmap.len() || start > end {
            return &[];
        }
        &mmap[start..end]
    }

    /// Checked variant with explicit error.
    pub fn try_tensor_data<'a>(
        &self,
        tensor: &GgufTensorInfo,
        mmap: &'a [u8],
    ) -> Result<&'a [u8], String> {
        let start = self.data_offset.saturating_add(tensor.offset) as usize;
        let end = start.saturating_add(tensor.n_bytes());
        if end > mmap.len() || start > end {
            return Err(format!(
                "tensor {} out of bounds",
                tensor.name
            ));
        }
        Ok(&mmap[start..end])
    }
}
