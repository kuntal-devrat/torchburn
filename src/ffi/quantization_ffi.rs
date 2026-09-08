//! Quantization FFI: INT8/INT4 linear, fused MLP, fused attention, fused transformer layer.

use pyo3::prelude::*;
use pyo3::types::PyCapsule;

use crate::dlpack;

#[pyfunction]
#[pyo3(signature = (x, w, scales, bias=None))]
pub fn w8a32_linear(
    py: Python<'_>,
    x: &Bound<'_, PyCapsule>,
    w: &Bound<'_, PyCapsule>,
    scales: &Bound<'_, PyCapsule>,
    bias: Option<&Bound<'_, PyCapsule>>,
) -> PyResult<Py<PyCapsule>> {
    let x_view = unsafe { dlpack::BorrowedTensor::from_capsule(x)? };
    let w_view = unsafe { dlpack::BorrowedTensor::from_capsule(w)? };
    let s_view = unsafe { dlpack::BorrowedTensor::from_capsule(scales)? };
    let b_view = match bias {
        Some(b) => Some(unsafe { dlpack::BorrowedTensor::from_capsule(b)? }),
        None => None,
    };

    let out = py.allow_threads(|| {
        crate::quantization::w8a32_linear(&x_view, &w_view, &s_view, b_view.as_ref())
    })?;

    dlpack::owned_to_capsule_owned(py, out)
}

#[pyfunction]
#[pyo3(signature = (x, w_packed, scales, bias=None))]
pub fn w4a32_linear(
    py: Python<'_>,
    x: &Bound<'_, PyCapsule>,
    w_packed: &Bound<'_, PyCapsule>,
    scales: &Bound<'_, PyCapsule>,
    bias: Option<&Bound<'_, PyCapsule>>,
) -> PyResult<Py<PyCapsule>> {
    let x_view = unsafe { dlpack::BorrowedTensor::from_capsule(x)? };
    let w_view = unsafe { dlpack::BorrowedTensor::from_capsule(w_packed)? };
    let s_view = unsafe { dlpack::BorrowedTensor::from_capsule(scales)? };
    let b_view = match bias {
        Some(b) => Some(unsafe { dlpack::BorrowedTensor::from_capsule(b)? }),
        None => None,
    };

    let out = py.allow_threads(|| {
        crate::quantization::w4a32_linear(&x_view, &w_view, &s_view, b_view.as_ref())
    })?;

    dlpack::owned_to_capsule_owned(py, out)
}

#[pyfunction]
#[pyo3(signature = (x, w_packed, scales, bias=None, group_size=64))]
pub fn w4a32_grouped_linear(
    py: Python<'_>,
    x: &Bound<'_, PyCapsule>,
    w_packed: &Bound<'_, PyCapsule>,
    scales: &Bound<'_, PyCapsule>,
    bias: Option<&Bound<'_, PyCapsule>>,
    group_size: usize,
) -> PyResult<Py<PyCapsule>> {
    let x_view = unsafe { dlpack::BorrowedTensor::from_capsule(x)? };
    let w_view = unsafe { dlpack::BorrowedTensor::from_capsule(w_packed)? };
    let s_view = unsafe { dlpack::BorrowedTensor::from_capsule(scales)? };
    let b_view = match bias {
        Some(b) => Some(unsafe { dlpack::BorrowedTensor::from_capsule(b)? }),
        None => None,
    };

    let out = py.allow_threads(|| {
        crate::quantization::w4a32_grouped_linear(
            &x_view,
            &w_view,
            &s_view,
            b_view.as_ref(),
            group_size,
        )
    })?;

    dlpack::owned_to_capsule_owned(py, out)
}

#[pyfunction]
#[pyo3(signature = (x, w_blocked, bias=None, group_size=64))]
pub fn w4a32_grouped_linear_v2(
    py: Python<'_>,
    x: &Bound<'_, PyCapsule>,
    w_blocked: &Bound<'_, PyCapsule>,
    bias: Option<&Bound<'_, PyCapsule>>,
    group_size: usize,
) -> PyResult<Py<PyCapsule>> {
    let x_view = unsafe { dlpack::BorrowedTensor::from_capsule(x)? };
    let w_view = unsafe { dlpack::BorrowedTensor::from_capsule(w_blocked)? };
    let b_view = match bias {
        Some(b) => Some(unsafe { dlpack::BorrowedTensor::from_capsule(b)? }),
        None => None,
    };

    let out = py.allow_threads(|| {
        crate::quantization::w4a32_grouped_linear_v2(&x_view, &w_view, b_view.as_ref(), group_size)
    })?;

    dlpack::owned_to_capsule_owned(py, out)
}

#[pyfunction]
#[pyo3(signature = (x, w_packed, scales, bias=None, group_size=64))]
pub fn wgpu_w4a32_grouped_linear(
    py: Python<'_>,
    x: &Bound<'_, PyCapsule>,
    w_packed: &Bound<'_, PyCapsule>,
    scales: &Bound<'_, PyCapsule>,
    bias: Option<&Bound<'_, PyCapsule>>,
    group_size: usize,
) -> PyResult<Py<PyCapsule>> {
    let x_view = unsafe { dlpack::BorrowedTensor::from_capsule(x)? };
    let w_view = unsafe { dlpack::BorrowedTensor::from_capsule(w_packed)? };
    let s_view = unsafe { dlpack::BorrowedTensor::from_capsule(scales)? };
    let b_view = match bias {
        Some(b) => Some(unsafe { dlpack::BorrowedTensor::from_capsule(b)? }),
        None => None,
    };

    #[cfg(feature = "burn-wgpu")]
    {
        let out = py.allow_threads(|| {
            crate::quantization::wgpu_w4a32_grouped_linear(
                &x_view,
                &w_view,
                &s_view,
                b_view.as_ref(),
                group_size,
            )
        })?;
        dlpack::owned_to_capsule_owned(py, out)
    }
    #[cfg(not(feature = "burn-wgpu"))]
    {
        let _ = (py, x_view, w_view, s_view, b_view, group_size);
        Err(pyo3::exceptions::PyRuntimeError::new_err(
            "burn-wgpu feature not enabled",
        ))
    }
}

#[pyfunction]
#[pyo3(signature = (x, gate_w, gate_s, gate_b, up_w, up_s, up_b, down_w, down_s, down_b))]
pub fn fused_swiglu_mlp_w8a32(
    py: Python<'_>,
    x: &Bound<'_, PyCapsule>,
    gate_w: &Bound<'_, PyCapsule>,
    gate_s: &Bound<'_, PyCapsule>,
    gate_b: Option<&Bound<'_, PyCapsule>>,
    up_w: &Bound<'_, PyCapsule>,
    up_s: &Bound<'_, PyCapsule>,
    up_b: Option<&Bound<'_, PyCapsule>>,
    down_w: &Bound<'_, PyCapsule>,
    down_s: &Bound<'_, PyCapsule>,
    down_b: Option<&Bound<'_, PyCapsule>>,
) -> PyResult<Py<PyCapsule>> {
    let x_view = unsafe { dlpack::BorrowedTensor::from_capsule(x)? };
    let gw_view = unsafe { dlpack::BorrowedTensor::from_capsule(gate_w)? };
    let gs_view = unsafe { dlpack::BorrowedTensor::from_capsule(gate_s)? };
    let gb_view = match gate_b {
        Some(b) => Some(unsafe { dlpack::BorrowedTensor::from_capsule(b)? }),
        None => None,
    };
    let uw_view = unsafe { dlpack::BorrowedTensor::from_capsule(up_w)? };
    let us_view = unsafe { dlpack::BorrowedTensor::from_capsule(up_s)? };
    let ub_view = match up_b {
        Some(b) => Some(unsafe { dlpack::BorrowedTensor::from_capsule(b)? }),
        None => None,
    };
    let dw_view = unsafe { dlpack::BorrowedTensor::from_capsule(down_w)? };
    let ds_view = unsafe { dlpack::BorrowedTensor::from_capsule(down_s)? };
    let db_view = match down_b {
        Some(b) => Some(unsafe { dlpack::BorrowedTensor::from_capsule(b)? }),
        None => None,
    };

    let out = py.allow_threads(|| {
        crate::quantization::fused_swiglu_mlp_w8a32(
            &x_view,
            &gw_view,
            &gs_view,
            gb_view.as_ref(),
            &uw_view,
            &us_view,
            ub_view.as_ref(),
            &dw_view,
            &ds_view,
            db_view.as_ref(),
        )
    })?;

    dlpack::owned_to_capsule_owned(py, out)
}

#[pyfunction]
#[pyo3(signature = (x, gate_w, gate_s, gate_b, up_w, up_s, up_b, down_w, down_s, down_b, group_size=64))]
pub fn fused_swiglu_mlp_w4a32(
    py: Python<'_>,
    x: &Bound<'_, PyCapsule>,
    gate_w: &Bound<'_, PyCapsule>,
    gate_s: &Bound<'_, PyCapsule>,
    gate_b: Option<&Bound<'_, PyCapsule>>,
    up_w: &Bound<'_, PyCapsule>,
    up_s: &Bound<'_, PyCapsule>,
    up_b: Option<&Bound<'_, PyCapsule>>,
    down_w: &Bound<'_, PyCapsule>,
    down_s: &Bound<'_, PyCapsule>,
    down_b: Option<&Bound<'_, PyCapsule>>,
    group_size: usize,
) -> PyResult<Py<PyCapsule>> {
    let x_view = unsafe { dlpack::BorrowedTensor::from_capsule(x)? };
    let gw_view = unsafe { dlpack::BorrowedTensor::from_capsule(gate_w)? };
    let gs_view = unsafe { dlpack::BorrowedTensor::from_capsule(gate_s)? };
    let gb_view = match gate_b {
        Some(b) => Some(unsafe { dlpack::BorrowedTensor::from_capsule(b)? }),
        None => None,
    };
    let uw_view = unsafe { dlpack::BorrowedTensor::from_capsule(up_w)? };
    let us_view = unsafe { dlpack::BorrowedTensor::from_capsule(up_s)? };
    let ub_view = match up_b {
        Some(b) => Some(unsafe { dlpack::BorrowedTensor::from_capsule(b)? }),
        None => None,
    };
    let dw_view = unsafe { dlpack::BorrowedTensor::from_capsule(down_w)? };
    let ds_view = unsafe { dlpack::BorrowedTensor::from_capsule(down_s)? };
    let db_view = match down_b {
        Some(b) => Some(unsafe { dlpack::BorrowedTensor::from_capsule(b)? }),
        None => None,
    };

    let out = py.allow_threads(|| {
        crate::quantization::fused_swiglu_mlp_w4a32(
            &x_view,
            &gw_view,
            &gs_view,
            gb_view.as_ref(),
            &uw_view,
            &us_view,
            ub_view.as_ref(),
            &dw_view,
            &ds_view,
            db_view.as_ref(),
            group_size,
        )
    })?;

    dlpack::owned_to_capsule_owned(py, out)
}

#[pyfunction]
pub fn quantize_linear_int8(
    py: Python<'_>,
    w: &Bound<'_, PyCapsule>,
) -> PyResult<(Py<PyCapsule>, Py<PyCapsule>)> {
    let w_view = unsafe { dlpack::BorrowedTensor::from_capsule(w)? };
    let (out_w, out_s) =
        py.allow_threads(|| crate::quantization::quantize_linear_weights_int8(&w_view))?;
    let cap_w = dlpack::owned_to_capsule_typed(py, out_w, dlpack::DL_DTYPE_INT, 8)?;
    let cap_s = dlpack::owned_to_capsule_owned(py, out_s)?;
    Ok((cap_w, cap_s))
}

#[pyfunction]
pub fn quantize_linear_int4(
    py: Python<'_>,
    w: &Bound<'_, PyCapsule>,
) -> PyResult<(Py<PyCapsule>, Py<PyCapsule>)> {
    let w_view = unsafe { dlpack::BorrowedTensor::from_capsule(w)? };
    let (out_w, out_s) =
        py.allow_threads(|| crate::quantization::quantize_linear_weights_int4(&w_view))?;
    let cap_w = dlpack::owned_to_capsule_typed(py, out_w, dlpack::DL_DTYPE_UINT, 8)?;
    let cap_s = dlpack::owned_to_capsule_owned(py, out_s)?;
    Ok((cap_w, cap_s))
}

#[pyfunction]
pub fn fused_attention_step_w8a32(
    py: Python<'_>,
    x: &Bound<'_, PyCapsule>,
    qkv_w: &Bound<'_, PyCapsule>,
    qkv_s: &Bound<'_, PyCapsule>,
    qkv_b: Option<&Bound<'_, PyCapsule>>,
    o_w: &Bound<'_, PyCapsule>,
    o_s: &Bound<'_, PyCapsule>,
    o_b: Option<&Bound<'_, PyCapsule>>,
    k_cache: &Bound<'_, PyCapsule>,
    v_cache: &Bound<'_, PyCapsule>,
    cos: &Bound<'_, PyCapsule>,
    sin: &Bound<'_, PyCapsule>,
    offset: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> PyResult<Py<PyCapsule>> {
    let x_view = unsafe { dlpack::BorrowedTensor::from_capsule(x)? };
    let qkv_w_view = unsafe { dlpack::BorrowedTensor::from_capsule(qkv_w)? };
    let qkv_s_view = unsafe { dlpack::BorrowedTensor::from_capsule(qkv_s)? };
    let qkv_b_view = match qkv_b {
        Some(b) => Some(unsafe { dlpack::BorrowedTensor::from_capsule(b)? }),
        None => None,
    };
    let o_w_view = unsafe { dlpack::BorrowedTensor::from_capsule(o_w)? };
    let o_s_view = unsafe { dlpack::BorrowedTensor::from_capsule(o_s)? };
    let o_b_view = match o_b {
        Some(b) => Some(unsafe { dlpack::BorrowedTensor::from_capsule(b)? }),
        None => None,
    };
    let k_cache_view = unsafe { dlpack::BorrowedTensor::from_capsule(k_cache)? };
    let v_cache_view = unsafe { dlpack::BorrowedTensor::from_capsule(v_cache)? };
    let cos_view = unsafe { dlpack::BorrowedTensor::from_capsule(cos)? };
    let sin_view = unsafe { dlpack::BorrowedTensor::from_capsule(sin)? };

    let out = py.allow_threads(|| {
        crate::quantization::fused_attention_step_w8a32(
            &x_view,
            &qkv_w_view,
            &qkv_s_view,
            qkv_b_view.as_ref(),
            &o_w_view,
            &o_s_view,
            o_b_view.as_ref(),
            &k_cache_view,
            &v_cache_view,
            &cos_view,
            &sin_view,
            offset,
            num_heads,
            num_kv_heads,
            head_dim,
        )
    })?;

    dlpack::owned_to_capsule_owned(py, out)
}

#[pyfunction]
pub fn fused_attention_step_w4a32(
    py: Python<'_>,
    x: &Bound<'_, PyCapsule>,
    qkv_w: &Bound<'_, PyCapsule>,
    qkv_s: &Bound<'_, PyCapsule>,
    qkv_b: Option<&Bound<'_, PyCapsule>>,
    o_w: &Bound<'_, PyCapsule>,
    o_s: &Bound<'_, PyCapsule>,
    o_b: Option<&Bound<'_, PyCapsule>>,
    k_cache: &Bound<'_, PyCapsule>,
    v_cache: &Bound<'_, PyCapsule>,
    cos: &Bound<'_, PyCapsule>,
    sin: &Bound<'_, PyCapsule>,
    offset: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    group_size: usize,
) -> PyResult<Py<PyCapsule>> {
    let x_view = unsafe { dlpack::BorrowedTensor::from_capsule(x)? };
    let qkv_w_view = unsafe { dlpack::BorrowedTensor::from_capsule(qkv_w)? };
    let qkv_s_view = unsafe { dlpack::BorrowedTensor::from_capsule(qkv_s)? };
    let qkv_b_view = match qkv_b {
        Some(b) => Some(unsafe { dlpack::BorrowedTensor::from_capsule(b)? }),
        None => None,
    };
    let o_w_view = unsafe { dlpack::BorrowedTensor::from_capsule(o_w)? };
    let o_s_view = unsafe { dlpack::BorrowedTensor::from_capsule(o_s)? };
    let o_b_view = match o_b {
        Some(b) => Some(unsafe { dlpack::BorrowedTensor::from_capsule(b)? }),
        None => None,
    };
    let k_cache_view = unsafe { dlpack::BorrowedTensor::from_capsule(k_cache)? };
    let v_cache_view = unsafe { dlpack::BorrowedTensor::from_capsule(v_cache)? };
    let cos_view = unsafe { dlpack::BorrowedTensor::from_capsule(cos)? };
    let sin_view = unsafe { dlpack::BorrowedTensor::from_capsule(sin)? };

    let out = py.allow_threads(|| {
        crate::quantization::fused_attention_step_w4a32(
            &x_view,
            &qkv_w_view,
            &qkv_s_view,
            qkv_b_view.as_ref(),
            &o_w_view,
            &o_s_view,
            o_b_view.as_ref(),
            &k_cache_view,
            &v_cache_view,
            &cos_view,
            &sin_view,
            offset,
            num_heads,
            num_kv_heads,
            head_dim,
            group_size,
        )
    })?;

    dlpack::owned_to_capsule_owned(py, out)
}

#[pyfunction]
#[pyo3(signature = (x, input_norm_w, qkv_w, qkv_s, qkv_b, o_w, o_s, o_b, post_norm_w, gate_w, gate_s, gate_b, up_w, up_s, up_b, down_w, down_s, down_b, k_cache, v_cache, cos, sin, offset, num_heads, num_kv_heads, head_dim, group_size, eps=1e-6))]
pub fn fused_transformer_layer_step_w4a32(
    py: Python<'_>,
    x: &Bound<'_, PyCapsule>,
    input_norm_w: &Bound<'_, PyCapsule>,
    qkv_w: &Bound<'_, PyCapsule>,
    qkv_s: &Bound<'_, PyCapsule>,
    qkv_b: Option<&Bound<'_, PyCapsule>>,
    o_w: &Bound<'_, PyCapsule>,
    o_s: &Bound<'_, PyCapsule>,
    o_b: Option<&Bound<'_, PyCapsule>>,
    post_norm_w: &Bound<'_, PyCapsule>,
    gate_w: &Bound<'_, PyCapsule>,
    gate_s: &Bound<'_, PyCapsule>,
    gate_b: Option<&Bound<'_, PyCapsule>>,
    up_w: &Bound<'_, PyCapsule>,
    up_s: &Bound<'_, PyCapsule>,
    up_b: Option<&Bound<'_, PyCapsule>>,
    down_w: &Bound<'_, PyCapsule>,
    down_s: &Bound<'_, PyCapsule>,
    down_b: Option<&Bound<'_, PyCapsule>>,
    k_cache: &Bound<'_, PyCapsule>,
    v_cache: &Bound<'_, PyCapsule>,
    cos: &Bound<'_, PyCapsule>,
    sin: &Bound<'_, PyCapsule>,
    offset: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    group_size: usize,
    eps: f64,
) -> PyResult<()> {
    let mut x_view = unsafe { dlpack::BorrowedTensor::from_capsule(x)? };
    let input_norm_w_view = unsafe { dlpack::BorrowedTensor::from_capsule(input_norm_w)? };
    let qkv_w_view = unsafe { dlpack::BorrowedTensor::from_capsule(qkv_w)? };
    let qkv_s_view = unsafe { dlpack::BorrowedTensor::from_capsule(qkv_s)? };
    let qkv_b_view = match qkv_b {
        Some(b) => Some(unsafe { dlpack::BorrowedTensor::from_capsule(b)? }),
        None => None,
    };
    let o_w_view = unsafe { dlpack::BorrowedTensor::from_capsule(o_w)? };
    let o_s_view = unsafe { dlpack::BorrowedTensor::from_capsule(o_s)? };
    let o_b_view = match o_b {
        Some(b) => Some(unsafe { dlpack::BorrowedTensor::from_capsule(b)? }),
        None => None,
    };
    let post_norm_w_view = unsafe { dlpack::BorrowedTensor::from_capsule(post_norm_w)? };
    let gate_w_view = unsafe { dlpack::BorrowedTensor::from_capsule(gate_w)? };
    let gate_s_view = unsafe { dlpack::BorrowedTensor::from_capsule(gate_s)? };
    let gate_b_view = match gate_b {
        Some(b) => Some(unsafe { dlpack::BorrowedTensor::from_capsule(b)? }),
        None => None,
    };
    let up_w_view = unsafe { dlpack::BorrowedTensor::from_capsule(up_w)? };
    let up_s_view = unsafe { dlpack::BorrowedTensor::from_capsule(up_s)? };
    let up_b_view = match up_b {
        Some(b) => Some(unsafe { dlpack::BorrowedTensor::from_capsule(b)? }),
        None => None,
    };
    let down_w_view = unsafe { dlpack::BorrowedTensor::from_capsule(down_w)? };
    let down_s_view = unsafe { dlpack::BorrowedTensor::from_capsule(down_s)? };
    let down_b_view = match down_b {
        Some(b) => Some(unsafe { dlpack::BorrowedTensor::from_capsule(b)? }),
        None => None,
    };
    let k_cache_view = unsafe { dlpack::BorrowedTensor::from_capsule(k_cache)? };
    let v_cache_view = unsafe { dlpack::BorrowedTensor::from_capsule(v_cache)? };
    let cos_view = unsafe { dlpack::BorrowedTensor::from_capsule(cos)? };
    let sin_view = unsafe { dlpack::BorrowedTensor::from_capsule(sin)? };

    py.allow_threads(|| {
        crate::quantization::fused_transformer_layer_step_w4a32(
            &mut x_view,
            &input_norm_w_view,
            &qkv_w_view,
            &qkv_s_view,
            qkv_b_view.as_ref(),
            &o_w_view,
            &o_s_view,
            o_b_view.as_ref(),
            &post_norm_w_view,
            &gate_w_view,
            &gate_s_view,
            gate_b_view.as_ref(),
            &up_w_view,
            &up_s_view,
            up_b_view.as_ref(),
            &down_w_view,
            &down_s_view,
            down_b_view.as_ref(),
            &k_cache_view,
            &v_cache_view,
            &cos_view,
            &sin_view,
            offset,
            num_heads,
            num_kv_heads,
            head_dim,
            group_size,
            eps,
        )
    })
}

/// Batched prefill SwiGLU MLP for INT4 (W4A32) — dispatches to the Rayon-parallel
/// `gemm_w4a32_grouped` path when seq_len > 1, giving 2-5x prefill speedup over
/// the per-token GEMV loop. The Python layer calls this for `x.shape[1] > 1`.
///
/// Layout: x is (batch, T, K); gate_w/up_w/down_w are (N_out, K/2) packed INT4;
/// output is (batch, T, N_down) — same shape as F.silu(gate(x)) * up(x) → down(x).
#[pyfunction]
#[pyo3(signature = (x, gate_w, gate_s, gate_b, up_w, up_s, up_b, down_w, down_s, down_b, group_size=64))]
pub fn fused_swiglu_mlp_batched_w4a32(
    py: Python<'_>,
    x: &Bound<'_, PyCapsule>,
    gate_w: &Bound<'_, PyCapsule>,
    gate_s: &Bound<'_, PyCapsule>,
    gate_b: Option<&Bound<'_, PyCapsule>>,
    up_w: &Bound<'_, PyCapsule>,
    up_s: &Bound<'_, PyCapsule>,
    up_b: Option<&Bound<'_, PyCapsule>>,
    down_w: &Bound<'_, PyCapsule>,
    down_s: &Bound<'_, PyCapsule>,
    down_b: Option<&Bound<'_, PyCapsule>>,
    group_size: usize,
) -> PyResult<Py<PyCapsule>> {
    let x_view = unsafe { dlpack::BorrowedTensor::from_capsule(x)? };
    let gw_view = unsafe { dlpack::BorrowedTensor::from_capsule(gate_w)? };
    let gs_view = unsafe { dlpack::BorrowedTensor::from_capsule(gate_s)? };
    let gb_view = match gate_b {
        Some(b) => Some(unsafe { dlpack::BorrowedTensor::from_capsule(b)? }),
        None => None,
    };
    let uw_view = unsafe { dlpack::BorrowedTensor::from_capsule(up_w)? };
    let us_view = unsafe { dlpack::BorrowedTensor::from_capsule(up_s)? };
    let ub_view = match up_b {
        Some(b) => Some(unsafe { dlpack::BorrowedTensor::from_capsule(b)? }),
        None => None,
    };
    let dw_view = unsafe { dlpack::BorrowedTensor::from_capsule(down_w)? };
    let ds_view = unsafe { dlpack::BorrowedTensor::from_capsule(down_s)? };
    let db_view = match down_b {
        Some(b) => Some(unsafe { dlpack::BorrowedTensor::from_capsule(b)? }),
        None => None,
    };

    let out = py.allow_threads(|| {
        crate::quantization::fused_swiglu_mlp_batched_w4a32(
            &x_view,
            &gw_view,
            &gs_view,
            gb_view.as_ref(),
            &uw_view,
            &us_view,
            ub_view.as_ref(),
            &dw_view,
            &ds_view,
            db_view.as_ref(),
            group_size,
        )
    })?;

    dlpack::owned_to_capsule_owned(py, out)
}
