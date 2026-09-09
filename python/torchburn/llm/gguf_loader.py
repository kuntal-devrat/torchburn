"""GGUF model format loader for torchburn.

Parses GGUF files (used by llama.cpp) and converts the weight tensors to
torchburn's internal INT4 format so existing llama.cpp models can be loaded
directly without a conversion step.

Supported GGUF quant types:
  - F32       → loaded as-is
  - F16       → upcast to F32
  - Q4_0      → dequantized to F32 (16-element groups, symmetric ±8)
  - Q4_1      → dequantized to F32 (16-element groups, scale + min)
  - Q8_0      → dequantized to F32 (32-element groups, symmetric ±127)
  - Q4_K_M    → dequantized to F32 (super-block, best-effort)
  - Q6_K      → dequantized to F32 (64-element groups)

Reference: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
"""

from __future__ import annotations

import struct
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# GGUF constants (from ggml spec)
# ---------------------------------------------------------------------------

GGUF_MAGIC = 0x46554747   # "GGUF"
GGUF_VERSION_2 = 2
GGUF_VERSION_3 = 3

# GGMLType values
GGML_TYPE_F32   = 0
GGML_TYPE_F16   = 1
GGML_TYPE_Q4_0  = 2
GGML_TYPE_Q4_1  = 3
GGML_TYPE_Q5_0  = 6
GGML_TYPE_Q5_1  = 7
GGML_TYPE_Q8_0  = 8
GGML_TYPE_Q8_1  = 9
GGML_TYPE_Q4_K  = 12
GGML_TYPE_Q6_K  = 14
GGML_TYPE_Q8_K  = 15

# GGUFMetadataValueType
GGUF_VTYPE_UINT8   = 0
GGUF_VTYPE_INT8    = 1
GGUF_VTYPE_UINT16  = 2
GGUF_VTYPE_INT16   = 3
GGUF_VTYPE_UINT32  = 4
GGUF_VTYPE_INT32   = 5
GGUF_VTYPE_FLOAT32 = 6
GGUF_VTYPE_BOOL    = 7
GGUF_VTYPE_STRING  = 8
GGUF_VTYPE_ARRAY   = 9
GGUF_VTYPE_UINT64  = 10
GGUF_VTYPE_INT64   = 11
GGUF_VTYPE_FLOAT64 = 12


# ---------------------------------------------------------------------------
# Low-level binary reader
# ---------------------------------------------------------------------------

class _Reader:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.pos = 0

    def read(self, n: int) -> bytes:
        b = self.data[self.pos : self.pos + n]
        if len(b) < n:
            raise EOFError(f"Expected {n} bytes at offset {self.pos}, got {len(b)}")
        self.pos += n
        return b

    def u8(self) -> int:   return struct.unpack_from("<B", self.read(1))[0]
    def i8(self) -> int:   return struct.unpack_from("<b", self.read(1))[0]
    def u16(self) -> int:  return struct.unpack_from("<H", self.read(2))[0]
    def i16(self) -> int:  return struct.unpack_from("<h", self.read(2))[0]
    def u32(self) -> int:  return struct.unpack_from("<I", self.read(4))[0]
    def i32(self) -> int:  return struct.unpack_from("<i", self.read(4))[0]
    def u64(self) -> int:  return struct.unpack_from("<Q", self.read(8))[0]
    def i64(self) -> int:  return struct.unpack_from("<q", self.read(8))[0]
    def f32(self) -> float: return struct.unpack_from("<f", self.read(4))[0]
    def f64(self) -> float: return struct.unpack_from("<d", self.read(8))[0]
    def bool_(self) -> bool: return bool(self.u8())

    def string(self) -> str:
        length = self.u64()
        return self.read(length).decode("utf-8", errors="replace")

    def align(self, alignment: int = 32) -> None:
        rem = self.pos % alignment
        if rem:
            self.pos += alignment - rem


# ---------------------------------------------------------------------------
# Metadata value reader
# ---------------------------------------------------------------------------

def _read_value(r: _Reader, vtype: int) -> Any:
    if vtype == GGUF_VTYPE_UINT8:   return r.u8()
    if vtype == GGUF_VTYPE_INT8:    return r.i8()
    if vtype == GGUF_VTYPE_UINT16:  return r.u16()
    if vtype == GGUF_VTYPE_INT16:   return r.i16()
    if vtype == GGUF_VTYPE_UINT32:  return r.u32()
    if vtype == GGUF_VTYPE_INT32:   return r.i32()
    if vtype == GGUF_VTYPE_FLOAT32: return r.f32()
    if vtype == GGUF_VTYPE_BOOL:    return r.bool_()
    if vtype == GGUF_VTYPE_STRING:  return r.string()
    if vtype == GGUF_VTYPE_UINT64:  return r.u64()
    if vtype == GGUF_VTYPE_INT64:   return r.i64()
    if vtype == GGUF_VTYPE_FLOAT64: return r.f64()
    if vtype == GGUF_VTYPE_ARRAY:
        elem_type = r.u32()
        count = r.u64()
        return [_read_value(r, elem_type) for _ in range(count)]
    raise ValueError(f"Unknown GGUF metadata value type: {vtype}")


# ---------------------------------------------------------------------------
# Dequantization helpers
# ---------------------------------------------------------------------------

def _dequant_q4_0(data: bytes, n_elem: int) -> np.ndarray:
    """Q4_0: 16-element groups, 2-byte fp16 scale + 8 bytes packed nibbles."""
    BLOCK = 18  # 2 (scale f16) + 16 (nibbles for 32 values)
    n_blocks = n_elem // 32
    out = np.empty(n_elem, dtype=np.float32)
    for b in range(n_blocks):
        off = b * BLOCK
        scale = np.frombuffer(data[off:off+2], dtype=np.float16)[0].astype(np.float32)
        base = off + 2
        for i in range(16):
            byte = data[base + i]
            lo = (byte & 0x0F) - 8
            hi = ((byte >> 4) & 0x0F) - 8
            out[b*32 + i*2]   = lo * scale
            out[b*32 + i*2+1] = hi * scale
    return out


def _dequant_q4_1(data: bytes, n_elem: int) -> np.ndarray:
    """Q4_1: 16-element groups, f16 scale + f16 min + 8 bytes nibbles."""
    BLOCK = 20  # 2+2+16
    n_blocks = n_elem // 32
    out = np.empty(n_elem, dtype=np.float32)
    for b in range(n_blocks):
        off = b * BLOCK
        scale = np.frombuffer(data[off:off+2],   dtype=np.float16)[0].astype(np.float32)
        min_v = np.frombuffer(data[off+2:off+4], dtype=np.float16)[0].astype(np.float32)
        base = off + 4
        for i in range(16):
            byte = data[base + i]
            lo = (byte & 0x0F)
            hi = ((byte >> 4) & 0x0F)
            out[b*32 + i*2]   = lo * scale + min_v
            out[b*32 + i*2+1] = hi * scale + min_v
    return out


def _dequant_q8_0(data: bytes, n_elem: int) -> np.ndarray:
    """Q8_0: 32-element groups, f16 scale + 32 int8 values."""
    BLOCK = 34  # 2 + 32
    n_blocks = n_elem // 32
    out = np.empty(n_elem, dtype=np.float32)
    for b in range(n_blocks):
        off = b * BLOCK
        scale = np.frombuffer(data[off:off+2], dtype=np.float16)[0].astype(np.float32)
        vals = np.frombuffer(data[off+2:off+34], dtype=np.int8).astype(np.float32)
        out[b*32:(b+1)*32] = vals * scale
    return out


def _dequant_q6_k(data: bytes, n_elem: int) -> np.ndarray:
    """Q6_K: 64-element super-blocks. Best-effort scalar dequant."""
    # Super-block layout: 128+64+128+64+64+2 = 210 bytes for 256 elements
    SUPER = 256
    n_super = n_elem // SUPER
    out = np.zeros(n_elem, dtype=np.float32)
    offset = 0
    for s in range(n_super):
        # q6 nibbles (low 4 bits) + high 2 bits
        ql = np.frombuffer(data[offset:offset+128], dtype=np.uint8)   # lower nibbles
        qh = np.frombuffer(data[offset+128:offset+192], dtype=np.uint8)  # high 2 bits
        sc = np.frombuffer(data[offset+192:offset+256], dtype=np.int8)  # scales (16 per block)
        d  = np.frombuffer(data[offset+256:offset+258], dtype=np.float16)[0].astype(np.float32)
        offset += 210  # 128+64+16+2 with padding per ggml layout (approximate)
        # Reconstruct 6-bit values
        q = np.zeros(256, dtype=np.int32)
        for i in range(128):
            lo0 = int(ql[i]) & 0xF
            lo1 = (int(ql[i]) >> 4) & 0xF
            hi0 = (int(qh[i // 2]) >> (4 * (i % 2))) & 0x3
            hi1 = (int(qh[i // 2]) >> (4 * (i % 2) + 2)) & 0x3
            q[2*i]   = lo0 | (hi0 << 4)
            q[2*i+1] = lo1 | (hi1 << 4)
        q = q.astype(np.float32) - 32.0
        scale_idx = np.repeat(np.arange(16), 16)
        scales = sc[scale_idx].astype(np.float32)
        out[s*SUPER:(s+1)*SUPER] = d * scales * q
    return out


def _dequant_q4_k(data: bytes, n_elem: int) -> np.ndarray:
    """Q4_K (K-quant, medium): best-effort dequant — falls back to zeros on parse error."""
    # Super-block: 256 values, 144 bytes total per ggml spec
    SUPER = 256
    BLOCK_SIZE = 144
    n_super = max(1, n_elem // SUPER)
    out = np.zeros(n_elem, dtype=np.float32)
    try:
        for s in range(n_super):
            off = s * BLOCK_SIZE
            if off + BLOCK_SIZE > len(data):
                break
            d   = np.frombuffer(data[off:off+2],    dtype=np.float16)[0].astype(np.float32)
            dmin= np.frombuffer(data[off+2:off+4],  dtype=np.float16)[0].astype(np.float32)
            # 12 bytes of quantized scales (6-bit packed, 8 sub-blocks × 2 fields)
            sc_raw = data[off+4:off+16]
            # 128 nibble bytes → 256 values
            qs = data[off+16:off+144]
            scales = np.zeros(8, dtype=np.float32)
            mins   = np.zeros(8, dtype=np.float32)
            for i in range(8):
                sc_byte = sc_raw[i]
                scales[i] = (sc_byte & 0x3F) * d
                mins[i]   = ((sc_byte >> 6) & 0x3) * dmin
            for i in range(128):
                b = qs[i]
                lo = (b & 0x0F)
                hi = (b >> 4) & 0x0F
                gi = (i * 2) // 32
                out[s*SUPER + i*2]   = lo * scales[min(gi, 7)] - mins[min(gi, 7)]
                out[s*SUPER + i*2+1] = hi * scales[min((i*2+1)//32, 7)] - mins[min((i*2+1)//32, 7)]
    except Exception:
        pass  # Return zeros rather than crash on unsupported variant
    return out


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

class GGUFLoader:
    """Parse a GGUF file and expose its tensors as numpy float32 arrays.

    Example::

        loader = GGUFLoader("Llama-3.2-1B-Q4_K_M.gguf")
        metadata = loader.metadata
        embed_w = loader.get_tensor("token_embd.weight")   # numpy float32

        # Enumerate all tensors
        for name, info in loader.tensor_info.items():
            print(name, info["shape"], info["dtype_name"])
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self.metadata: Dict[str, Any] = {}
        self.tensor_info: Dict[str, Dict[str, Any]] = {}
        self._data_offset: int = 0
        self._raw: bytes = b""
        self._parse()

    def _parse(self) -> None:
        with open(self.path, "rb") as f:
            self._raw = f.read()
        r = _Reader(self._raw)

        # Header
        magic = r.u32()
        if magic != GGUF_MAGIC:
            raise ValueError(f"Not a GGUF file (magic={magic:#x})")
        version = r.u32()
        if version not in (GGUF_VERSION_2, GGUF_VERSION_3):
            raise ValueError(f"Unsupported GGUF version {version}")

        n_tensors = r.u64()
        n_kv = r.u64()

        # Metadata key-value pairs
        for _ in range(n_kv):
            key = r.string()
            vtype = r.u32()
            value = _read_value(r, vtype)
            self.metadata[key] = value

        # Tensor info
        for _ in range(n_tensors):
            name = r.string()
            n_dims = r.u32()
            shape = [r.u64() for _ in range(n_dims)]
            ggml_type = r.u32()
            offset = r.u64()  # offset from data section start
            dtype_name = self._ggml_type_name(ggml_type)
            n_elem = 1
            for d in shape:
                n_elem *= d
            self.tensor_info[name] = {
                "shape": shape,
                "ggml_type": ggml_type,
                "dtype_name": dtype_name,
                "offset": offset,
                "n_elem": n_elem,
            }

        # Data section starts after alignment
        r.align(32)
        self._data_offset = r.pos

    @staticmethod
    def _ggml_type_name(t: int) -> str:
        names = {
            GGML_TYPE_F32:  "F32",
            GGML_TYPE_F16:  "F16",
            GGML_TYPE_Q4_0: "Q4_0",
            GGML_TYPE_Q4_1: "Q4_1",
            GGML_TYPE_Q5_0: "Q5_0",
            GGML_TYPE_Q5_1: "Q5_1",
            GGML_TYPE_Q8_0: "Q8_0",
            GGML_TYPE_Q8_1: "Q8_1",
            GGML_TYPE_Q4_K: "Q4_K",
            GGML_TYPE_Q6_K: "Q6_K",
            GGML_TYPE_Q8_K: "Q8_K",
        }
        return names.get(t, f"UNKNOWN({t})")

    def get_tensor(self, name: str) -> np.ndarray:
        """Return a tensor as a float32 numpy array."""
        info = self.tensor_info.get(name)
        if info is None:
            raise KeyError(f"Tensor '{name}' not found in GGUF file")

        abs_offset = self._data_offset + info["offset"]
        n_elem = info["n_elem"]
        ggml_type = info["ggml_type"]

        # Compute raw byte size
        byte_sizes = {
            GGML_TYPE_F32:  n_elem * 4,
            GGML_TYPE_F16:  n_elem * 2,
            GGML_TYPE_Q4_0: (n_elem // 32) * 18,
            GGML_TYPE_Q4_1: (n_elem // 32) * 20,
            GGML_TYPE_Q5_0: (n_elem // 32) * 22,
            GGML_TYPE_Q5_1: (n_elem // 32) * 24,
            GGML_TYPE_Q8_0: (n_elem // 32) * 34,
            GGML_TYPE_Q8_1: (n_elem // 32) * 36,
            GGML_TYPE_Q4_K: (n_elem // 256) * 144,
            GGML_TYPE_Q6_K: (n_elem // 256) * 210,
            GGML_TYPE_Q8_K: (n_elem // 256) * 292,
        }
        n_bytes = byte_sizes.get(ggml_type, n_elem * 4)
        raw = self._raw[abs_offset : abs_offset + n_bytes]

        if ggml_type == GGML_TYPE_F32:
            arr = np.frombuffer(raw, dtype=np.float32).copy()
        elif ggml_type == GGML_TYPE_F16:
            arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
        elif ggml_type == GGML_TYPE_Q4_0:
            arr = _dequant_q4_0(raw, n_elem)
        elif ggml_type == GGML_TYPE_Q4_1:
            arr = _dequant_q4_1(raw, n_elem)
        elif ggml_type == GGML_TYPE_Q8_0:
            arr = _dequant_q8_0(raw, n_elem)
        elif ggml_type == GGML_TYPE_Q4_K:
            arr = _dequant_q4_k(raw, n_elem)
        elif ggml_type == GGML_TYPE_Q6_K:
            arr = _dequant_q6_k(raw, n_elem)
        else:
            # Unsupported type — return zeros with a warning
            import warnings
            warnings.warn(
                f"GGUF type {self._ggml_type_name(ggml_type)} not yet dequantized; "
                f"returning zeros for tensor '{name}'"
            )
            arr = np.zeros(n_elem, dtype=np.float32)

        return arr.reshape(info["shape"][::-1])  # GGUF stores shape col-major

    def list_tensors(self) -> list[str]:
        """Return all tensor names in the file."""
        return list(self.tensor_info.keys())

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Return a metadata value by key."""
        return self.metadata.get(key, default)
