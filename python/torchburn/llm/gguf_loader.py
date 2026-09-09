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


def _dequant_q8_1(data: bytes, n_elem: int) -> np.ndarray:
    """Q8_1: 32-element groups, f16 scale + f16 sum + 32 int8 values."""
    BLOCK = 36  # 2 + 2 + 32
    n_blocks = n_elem // 32
    out = np.empty(n_elem, dtype=np.float32)
    for b in range(n_blocks):
        off = b * BLOCK
        scale = np.frombuffer(data[off:off+2], dtype=np.float16)[0].astype(np.float32)
        vals = np.frombuffer(data[off+4:off+36], dtype=np.int8).astype(np.float32)
        out[b*32:(b+1)*32] = vals * scale
    return out


def _dequant_q8_k(data: bytes, n_elem: int) -> np.ndarray:
    """Q8_K: 256-element super-blocks, f32 scale (d) + 256 int8 values + 16 int16 sums."""
    SUPER = 256
    BLOCK = 292  # 4 + 256 + 32
    n_blocks = n_elem // SUPER
    out = np.empty(n_elem, dtype=np.float32)
    for b in range(n_blocks):
        off = b * BLOCK
        d = np.frombuffer(data[off:off+4], dtype=np.float32)[0]
        qs = np.frombuffer(data[off+4:off+260], dtype=np.int8).astype(np.float32)
        out[b*SUPER:(b+1)*SUPER] = qs * d
    return out



def _dequant_q6_k(data: bytes, n_elem: int) -> np.ndarray:
    """Dequantize Q6_K super-blocks using the GGML 210-byte layout."""
    SUPER = 256
    BLOCK = 210
    if n_elem % SUPER:
        raise ValueError(f"Q6_K tensor length must be a multiple of {SUPER}, got {n_elem}")
    expected = (n_elem // SUPER) * BLOCK
    if len(data) != expected:
        raise ValueError(f"Q6_K data has {len(data)} bytes; expected {expected}")

    out = np.empty(n_elem, dtype=np.float32)
    for block in range(n_elem // SUPER):
        base = block * BLOCK
        ql = np.frombuffer(data[base:base + 128], dtype=np.uint8)
        qh = np.frombuffer(data[base + 128:base + 192], dtype=np.uint8)
        scales = np.frombuffer(data[base + 192:base + 208], dtype=np.int8)
        d = float(np.frombuffer(data[base + 208:base + 210], dtype=np.float16)[0])
        block_out = out[block * SUPER:(block + 1) * SUPER]

        for i in range(32):
            high = int(qh[i])
            values = (
                (int(ql[i]) & 0x0F) | ((high & 0x03) << 4),
                (int(ql[i + 32]) & 0x0F) | (((high >> 2) & 0x03) << 4),
                (int(ql[i]) >> 4) | (((high >> 4) & 0x03) << 4),
                (int(ql[i + 32]) >> 4) | (((high >> 6) & 0x03) << 4),
            )
            for group, value in enumerate(values):
                position = group * 32 + i
                block_out[position] = d * float(scales[position // 16]) * (value - 32)

            high = int(qh[i + 32])
            values = (
                (int(ql[i + 64]) & 0x0F) | ((high & 0x03) << 4),
                (int(ql[i + 96]) & 0x0F) | (((high >> 2) & 0x03) << 4),
                (int(ql[i + 64]) >> 4) | (((high >> 4) & 0x03) << 4),
                (int(ql[i + 96]) >> 4) | (((high >> 6) & 0x03) << 4),
            )
            for group, value in enumerate(values):
                position = 128 + group * 32 + i
                block_out[position] = d * float(scales[position // 16]) * (value - 32)
    return out


def _dequant_q4_k(data: bytes, n_elem: int) -> np.ndarray:
    """Dequantize Q4_K super-blocks using the GGML 144-byte layout."""
    SUPER = 256
    BLOCK_SIZE = 144
    if n_elem % SUPER:
        raise ValueError(f"Q4_K tensor length must be a multiple of {SUPER}, got {n_elem}")
    expected = (n_elem // SUPER) * BLOCK_SIZE
    if len(data) != expected:
        raise ValueError(f"Q4_K data has {len(data)} bytes; expected {expected}")

    out = np.empty(n_elem, dtype=np.float32)
    for block in range(n_elem // SUPER):
        base = block * BLOCK_SIZE
        d = float(np.frombuffer(data[base:base + 2], dtype=np.float16)[0])
        dmin = float(np.frombuffer(data[base + 2:base + 4], dtype=np.float16)[0])
        packed = np.frombuffer(data[base + 4:base + 16], dtype=np.uint8)
        qs = np.frombuffer(data[base + 16:base + 144], dtype=np.uint8)

        scales = np.empty(8, dtype=np.float32)
        mins = np.empty(8, dtype=np.float32)
        for i in range(4):
            scales[i] = d * float(packed[i] & 0x3F)
            mins[i] = dmin * float(packed[i + 4] & 0x3F)
        for i in range(4, 8):
            scales[i] = d * float((packed[i + 4] & 0x0F) | ((packed[i - 4] >> 6) << 4))
            mins[i] = dmin * float((packed[i + 4] >> 4) | ((packed[i] >> 6) << 4))

        block_out = out[block * SUPER:(block + 1) * SUPER]
        for group in range(8):
            q_group = qs[group * 16:(group + 1) * 16]
            start = group * 32
            block_out[start:start + 16] = (q_group & 0x0F) * scales[group] - mins[group]
            block_out[start + 16:start + 32] = (q_group >> 4) * scales[group] - mins[group]
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
        block_sizes = {
            GGML_TYPE_Q4_0: 32,
            GGML_TYPE_Q4_1: 32,
            GGML_TYPE_Q5_0: 32,
            GGML_TYPE_Q5_1: 32,
            GGML_TYPE_Q8_0: 32,
            GGML_TYPE_Q8_1: 32,
            GGML_TYPE_Q4_K: 256,
            GGML_TYPE_Q6_K: 256,
            GGML_TYPE_Q8_K: 256,
        }
        block_size = block_sizes.get(ggml_type)
        if block_size is not None and n_elem % block_size:
            raise ValueError(
                f"GGUF tensor '{name}' has {n_elem} elements; "
                f"{self._ggml_type_name(ggml_type)} requires multiples of {block_size}"
            )
        n_bytes = byte_sizes.get(ggml_type)
        if n_bytes is None:
            raise NotImplementedError(
                f"GGUF tensor type {self._ggml_type_name(ggml_type)} is not supported; "
                "refusing to fabricate tensor values"
            )
        if n_elem <= 0:
            raise ValueError(f"GGUF tensor '{name}' has invalid element count {n_elem}")
        raw = self._raw[abs_offset : abs_offset + n_bytes]
        if len(raw) != n_bytes:
            raise ValueError(
                f"GGUF tensor '{name}' is truncated: got {len(raw)} bytes, expected {n_bytes}"
            )

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
        elif ggml_type == GGML_TYPE_Q8_1:
            arr = _dequant_q8_1(raw, n_elem)
        elif ggml_type == GGML_TYPE_Q4_K:
            arr = _dequant_q4_k(raw, n_elem)
        elif ggml_type == GGML_TYPE_Q6_K:
            arr = _dequant_q6_k(raw, n_elem)
        elif ggml_type == GGML_TYPE_Q8_K:
            arr = _dequant_q8_k(raw, n_elem)
        else:
            raise NotImplementedError(
                f"GGUF tensor type {self._ggml_type_name(ggml_type)} is not supported"
            )

        if arr.size != n_elem:
            raise ValueError(
                f"Dequantizer returned {arr.size} values for tensor '{name}', expected {n_elem}"
            )
        return arr.reshape(info["shape"][::-1])  # GGUF stores shape col-major

    def list_tensors(self) -> list[str]:
        """Return all tensor names in the file."""
        return list(self.tensor_info.keys())

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Return a metadata value by key."""
        return self.metadata.get(key, default)
