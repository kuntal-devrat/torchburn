//! Single-op backward: the per-target backward match.
//! Inherits autograd root via super-super; pure move.

use super::super::*;

/// Convenience: execute backward for a single op given saved inputs.
/// This is used by the compiled callable's backward method.
pub fn backward_single(
    target: &str,
    upstream: &OwnedTensor,
    saved_inputs: &[&OwnedTensor],
    kwargs: &std::collections::HashMap<String, serde_json::Value>,
) -> Vec<OwnedTensor> {
    match target {
        "add" => {
            vec![upstream.clone(), upstream.clone()]
        }
        "sub" => {
            vec![upstream.clone(), {
                let n = elem_count(&upstream.shape);
                let mut neg = OwnedTensor::new(upstream.dtype, upstream.shape.clone());
                match upstream.dtype {
                    DType::F32 => {
                        let g = unsafe {
                            std::slice::from_raw_parts(upstream.data.as_ptr() as *const f32, n)
                        };
                        let o = unsafe {
                            std::slice::from_raw_parts_mut(neg.data.as_mut_ptr() as *mut f32, n)
                        };
                        for i in 0..n {
                            o[i] = -g[i];
                        }
                    }
                    DType::F64 => {
                        let g = unsafe {
                            std::slice::from_raw_parts(upstream.data.as_ptr() as *const f64, n)
                        };
                        let o = unsafe {
                            std::slice::from_raw_parts_mut(neg.data.as_mut_ptr() as *mut f64, n)
                        };
                        for i in 0..n {
                            o[i] = -g[i];
                        }
                    }
                    _ => {}
                }
                neg
            }]
        }
        "mul" => {
            assert!(saved_inputs.len() >= 2);
            let a = saved_inputs[0];
            let b = saved_inputs[1];
            let n = elem_count(&upstream.shape);
            let mut grad_a = OwnedTensor::new(upstream.dtype, upstream.shape.clone());
            let mut grad_b = OwnedTensor::new(upstream.dtype, upstream.shape.clone());
            match upstream.dtype {
                DType::F32 => {
                    let g = unsafe {
                        std::slice::from_raw_parts(upstream.data.as_ptr() as *const f32, n)
                    };
                    let ad = unsafe {
                        std::slice::from_raw_parts(
                            a.data.as_ptr() as *const f32,
                            elem_count(&a.shape),
                        )
                    };
                    let bd = unsafe {
                        std::slice::from_raw_parts(
                            b.data.as_ptr() as *const f32,
                            elem_count(&b.shape),
                        )
                    };
                    let ga = unsafe {
                        std::slice::from_raw_parts_mut(grad_a.data.as_mut_ptr() as *mut f32, n)
                    };
                    let gb = unsafe {
                        std::slice::from_raw_parts_mut(grad_b.data.as_mut_ptr() as *mut f32, n)
                    };
                    let an = elem_count(&a.shape);
                    let bn = elem_count(&b.shape);
                    for i in 0..n {
                        ga[i] = g[i] * bd[if bn == 1 { 0 } else { i % bn }];
                        gb[i] = g[i] * ad[if an == 1 { 0 } else { i % an }];
                    }
                }
                DType::F64 => {
                    let g = unsafe {
                        std::slice::from_raw_parts(upstream.data.as_ptr() as *const f64, n)
                    };
                    let ad = unsafe {
                        std::slice::from_raw_parts(
                            a.data.as_ptr() as *const f64,
                            elem_count(&a.shape),
                        )
                    };
                    let bd = unsafe {
                        std::slice::from_raw_parts(
                            b.data.as_ptr() as *const f64,
                            elem_count(&b.shape),
                        )
                    };
                    let ga = unsafe {
                        std::slice::from_raw_parts_mut(grad_a.data.as_mut_ptr() as *mut f64, n)
                    };
                    let gb = unsafe {
                        std::slice::from_raw_parts_mut(grad_b.data.as_mut_ptr() as *mut f64, n)
                    };
                    let an = elem_count(&a.shape);
                    let bn = elem_count(&b.shape);
                    for i in 0..n {
                        ga[i] = g[i] * bd[if bn == 1 { 0 } else { i % bn }];
                        gb[i] = g[i] * ad[if an == 1 { 0 } else { i % an }];
                    }
                }
                _ => {}
            }
            vec![grad_a, grad_b]
        }
        "relu" => {
            assert!(!saved_inputs.is_empty());
            let x = saved_inputs[0];
            let n = elem_count(&upstream.shape);
            let mut grad = OwnedTensor::new(upstream.dtype, upstream.shape.clone());
            match upstream.dtype {
                DType::F32 => {
                    let g = unsafe {
                        std::slice::from_raw_parts(upstream.data.as_ptr() as *const f32, n)
                    };
                    let xd = unsafe {
                        std::slice::from_raw_parts(
                            x.data.as_ptr() as *const f32,
                            elem_count(&x.shape),
                        )
                    };
                    let o = unsafe {
                        std::slice::from_raw_parts_mut(grad.data.as_mut_ptr() as *mut f32, n)
                    };
                    let xn = elem_count(&x.shape);
                    for i in 0..n {
                        let xi = if xn == 1 { 0 } else { i % xn };
                        o[i] = if xd[xi] > 0.0 { g[i] } else { 0.0 };
                    }
                }
                DType::F64 => {
                    let g = unsafe {
                        std::slice::from_raw_parts(upstream.data.as_ptr() as *const f64, n)
                    };
                    let xd = unsafe {
                        std::slice::from_raw_parts(
                            x.data.as_ptr() as *const f64,
                            elem_count(&x.shape),
                        )
                    };
                    let o = unsafe {
                        std::slice::from_raw_parts_mut(grad.data.as_mut_ptr() as *mut f64, n)
                    };
                    let xn = elem_count(&x.shape);
                    for i in 0..n {
                        let xi = if xn == 1 { 0 } else { i % xn };
                        o[i] = if xd[xi] > 0.0 { g[i] } else { 0.0 };
                    }
                }
                _ => {}
            }
            vec![grad]
        }
        "matmul" => {
            assert!(saved_inputs.len() >= 2);
            let a = saved_inputs[0];
            let b = saved_inputs[1];
            // Handle batched matmul (..., M, K) @ (..., K, N) = (..., M, N)
            if a.shape.len() >= 2 && b.shape.len() >= 2 {
                let k = *a.shape.last().unwrap_or(&1) as usize;
                let n = *b.shape.last().unwrap_or(&1) as usize;
                let batch_a: usize = a.shape[..a.shape.len() - 1]
                    .iter()
                    .map(|&d| d.max(0) as usize)
                    .product();
                let _batch_b: usize = b.shape[..b.shape.len() - 1]
                    .iter()
                    .map(|&d| d.max(0) as usize)
                    .product();
                let _m =
                    batch_a / k.max(1) * *a.shape.get(a.shape.len() - 2).unwrap_or(&1) as usize;
                // For 2D case, use the fast path
                if a.shape.len() == 2 && b.shape.len() == 2 {
                    let _m = a.shape[0] as usize;
                    let mut grad_a = OwnedTensor::new(upstream.dtype, a.shape.clone());
                    let mut grad_b = OwnedTensor::new(upstream.dtype, b.shape.clone());
                    match upstream.dtype {
                        DType::F32 => {
                            let g = unsafe {
                                std::slice::from_raw_parts(
                                    upstream.data.as_ptr() as *const f32,
                                    _m * n,
                                )
                            };
                            let bd = unsafe {
                                std::slice::from_raw_parts(b.data.as_ptr() as *const f32, k * n)
                            };
                            let ad = unsafe {
                                std::slice::from_raw_parts(a.data.as_ptr() as *const f32, _m * k)
                            };
                            let ga = unsafe {
                                std::slice::from_raw_parts_mut(
                                    grad_a.data.as_mut_ptr() as *mut f32,
                                    _m * k,
                                )
                            };
                            let gb = unsafe {
                                std::slice::from_raw_parts_mut(
                                    grad_b.data.as_mut_ptr() as *mut f32,
                                    k * n,
                                )
                            };
                            for i in 0.._m {
                                for j in 0..k {
                                    let mut s = 0.0f32;
                                    for kk in 0..n {
                                        s += g[i * n + kk] * bd[j * n + kk];
                                    }
                                    ga[i * k + j] = s;
                                }
                            }
                            for i in 0..k {
                                for j in 0..n {
                                    let mut s = 0.0f32;
                                    for kk in 0.._m {
                                        s += ad[kk * k + i] * g[kk * n + j];
                                    }
                                    gb[i * n + j] = s;
                                }
                            }
                        }
                        DType::F64 => {
                            let g = unsafe {
                                std::slice::from_raw_parts(
                                    upstream.data.as_ptr() as *const f64,
                                    _m * n,
                                )
                            };
                            let bd = unsafe {
                                std::slice::from_raw_parts(b.data.as_ptr() as *const f64, k * n)
                            };
                            let ad = unsafe {
                                std::slice::from_raw_parts(a.data.as_ptr() as *const f64, _m * k)
                            };
                            let ga = unsafe {
                                std::slice::from_raw_parts_mut(
                                    grad_a.data.as_mut_ptr() as *mut f64,
                                    _m * k,
                                )
                            };
                            let gb = unsafe {
                                std::slice::from_raw_parts_mut(
                                    grad_b.data.as_mut_ptr() as *mut f64,
                                    k * n,
                                )
                            };
                            for i in 0.._m {
                                for j in 0..k {
                                    let mut s = 0.0f64;
                                    for kk in 0..n {
                                        s += g[i * n + kk] * bd[j * n + kk];
                                    }
                                    ga[i * k + j] = s;
                                }
                            }
                            for i in 0..k {
                                for j in 0..n {
                                    let mut s = 0.0f64;
                                    for kk in 0.._m {
                                        s += ad[kk * k + i] * g[kk * n + j];
                                    }
                                    gb[i * n + j] = s;
                                }
                            }
                        }
                        _ => {}
                    }
                    vec![grad_a, grad_b]
                } else {
                    // Batched matmul: (..., M, K) @ (K, N) or (..., M, K) @ (..., K, N)
                    // Determine batch shape and per-batch sizes
                    let b_k = *a.shape.last().unwrap_or(&1) as usize;
                    let b_n = *b.shape.last().unwrap_or(&1) as usize;
                    let b_m = if a.shape.len() >= 2 {
                        *a.shape.get(a.shape.len() - 2).unwrap_or(&1) as usize
                    } else {
                        1
                    };
                    let batch: usize = a.shape[..a.shape.len().saturating_sub(2)]
                        .iter()
                        .map(|&d| d.max(0) as usize)
                        .product::<usize>()
                        .max(1);
                    let b_batch: usize = b.shape[..b.shape.len().saturating_sub(2)]
                        .iter()
                        .map(|&d| d.max(0) as usize)
                        .product::<usize>()
                        .max(1);
                    let g_batch: usize = upstream.shape[..upstream.shape.len().saturating_sub(2)]
                        .iter()
                        .map(|&d| d.max(0) as usize)
                        .product::<usize>()
                        .max(1);
                    let mut grad_a = OwnedTensor::new(upstream.dtype, a.shape.clone());
                    let mut grad_b = OwnedTensor::new(upstream.dtype, b.shape.clone());
                    match upstream.dtype {
                        DType::F32 => {
                            let gd = unsafe {
                                std::slice::from_raw_parts(
                                    upstream.data.as_ptr() as *const f32,
                                    elem_count(&upstream.shape),
                                )
                            };
                            let ad = unsafe {
                                std::slice::from_raw_parts(
                                    a.data.as_ptr() as *const f32,
                                    elem_count(&a.shape),
                                )
                            };
                            let bd = unsafe {
                                std::slice::from_raw_parts(
                                    b.data.as_ptr() as *const f32,
                                    elem_count(&b.shape),
                                )
                            };
                            let ga = unsafe {
                                std::slice::from_raw_parts_mut(
                                    grad_a.data.as_mut_ptr() as *mut f32,
                                    elem_count(&a.shape),
                                )
                            };
                            let gb = unsafe {
                                std::slice::from_raw_parts_mut(
                                    grad_b.data.as_mut_ptr() as *mut f32,
                                    elem_count(&b.shape),
                                )
                            };
                            // grad_a = grad_output @ b^T, batched over leading dims
                            for bi in 0..batch.min(g_batch) {
                                for i in 0..b_m {
                                    for j in 0..b_k {
                                        let mut s = 0.0f32;
                                        for kk in 0..b_n {
                                            s += gd[bi * b_m * b_n + i * b_n + kk]
                                                * bd[bi.min(b_batch - 1) * b_k * b_n
                                                    + j * b_n
                                                    + kk];
                                        }
                                        ga[bi * b_m * b_k + i * b_k + j] = s;
                                    }
                                }
                            }
                            // grad_b = a^T @ grad_output
                            for bi in 0..b_batch.min(g_batch) {
                                for i in 0..b_k {
                                    for j in 0..b_n {
                                        let mut s = 0.0f32;
                                        for kk in 0..b_m {
                                            s += ad[bi.min(batch - 1) * b_m * b_k + kk * b_k + i]
                                                * gd[bi * b_m * b_n + kk * b_n + j];
                                        }
                                        gb[bi * b_k * b_n + i * b_n + j] = s;
                                    }
                                }
                            }
                        }
                        DType::F64 => {
                            let gd = unsafe {
                                std::slice::from_raw_parts(
                                    upstream.data.as_ptr() as *const f64,
                                    elem_count(&upstream.shape),
                                )
                            };
                            let ad = unsafe {
                                std::slice::from_raw_parts(
                                    a.data.as_ptr() as *const f64,
                                    elem_count(&a.shape),
                                )
                            };
                            let bd = unsafe {
                                std::slice::from_raw_parts(
                                    b.data.as_ptr() as *const f64,
                                    elem_count(&b.shape),
                                )
                            };
                            let ga = unsafe {
                                std::slice::from_raw_parts_mut(
                                    grad_a.data.as_mut_ptr() as *mut f64,
                                    elem_count(&a.shape),
                                )
                            };
                            let gb = unsafe {
                                std::slice::from_raw_parts_mut(
                                    grad_b.data.as_mut_ptr() as *mut f64,
                                    elem_count(&b.shape),
                                )
                            };
                            for bi in 0..batch.min(g_batch) {
                                for i in 0..b_m {
                                    for j in 0..b_k {
                                        let mut s = 0.0f64;
                                        for kk in 0..b_n {
                                            s += gd[bi * b_m * b_n + i * b_n + kk]
                                                * bd[bi.min(b_batch - 1) * b_k * b_n
                                                    + j * b_n
                                                    + kk];
                                        }
                                        ga[bi * b_m * b_k + i * b_k + j] = s;
                                    }
                                }
                            }
                            for bi in 0..b_batch.min(g_batch) {
                                for i in 0..b_k {
                                    for j in 0..b_n {
                                        let mut s = 0.0f64;
                                        for kk in 0..b_m {
                                            s += ad[bi.min(batch - 1) * b_m * b_k + kk * b_k + i]
                                                * gd[bi * b_m * b_n + kk * b_n + j];
                                        }
                                        gb[bi * b_k * b_n + i * b_n + j] = s;
                                    }
                                }
                            }
                        }
                        _ => {}
                    }
                    vec![grad_a, grad_b]
                }
            } else {
                vec![
                    OwnedTensor::new(upstream.dtype, a.shape.clone()),
                    OwnedTensor::new(upstream.dtype, b.shape.clone()),
                ]
            }
        }
        "mse_loss" => {
            assert!(saved_inputs.len() >= 2);
            let input = saved_inputs[0];
            let target = saved_inputs[1];
            let n = elem_count(&input.shape);
            let reduction: i64 = kwargs
                .get("reduction")
                .and_then(|v| {
                    if let Some(i) = v.as_i64() {
                        Some(i)
                    } else if let Some(s) = v.as_str() {
                        match s {
                            "none" => Some(0),
                            "mean" => Some(1),
                            "sum" => Some(2),
                            _ => None,
                        }
                    } else {
                        None
                    }
                })
                .unwrap_or(1);
            let scale = match reduction {
                1 => 2.0 / n.max(1) as f64, // mean
                2 => 2.0,                   // sum
                _ => 2.0,                   // none
            };
            let mut grad = OwnedTensor::new(input.dtype, input.shape.clone());
            let un = elem_count(&upstream.shape);
            match input.dtype {
                DType::F32 => {
                    let up = unsafe {
                        std::slice::from_raw_parts(upstream.data.as_ptr() as *const f32, un)
                    };
                    let inp =
                        unsafe { std::slice::from_raw_parts(input.data.as_ptr() as *const f32, n) };
                    let tgt = unsafe {
                        std::slice::from_raw_parts(
                            target.data.as_ptr() as *const f32,
                            elem_count(&target.shape),
                        )
                    };
                    let o = unsafe {
                        std::slice::from_raw_parts_mut(grad.data.as_mut_ptr() as *mut f32, n)
                    };
                    let tn = elem_count(&target.shape);
                    let s = scale as f32;
                    if reduction == 0 {
                        for i in 0..n {
                            let ti = if tn == 1 { 0 } else { i % tn };
                            let ui = if un == 1 { 0 } else { i % un.max(1) };
                            let u_val = if un > 0 { up[ui] } else { 1.0 };
                            o[i] = u_val * s * (inp[i] - tgt[ti]);
                        }
                    } else {
                        let u_val = if un > 0 { up[0] } else { 1.0 };
                        for i in 0..n {
                            let ti = if tn == 1 { 0 } else { i % tn };
                            o[i] = u_val * s * (inp[i] - tgt[ti]);
                        }
                    }
                }
                DType::F64 => {
                    let up = unsafe {
                        std::slice::from_raw_parts(upstream.data.as_ptr() as *const f64, un)
                    };
                    let inp =
                        unsafe { std::slice::from_raw_parts(input.data.as_ptr() as *const f64, n) };
                    let tgt = unsafe {
                        std::slice::from_raw_parts(
                            target.data.as_ptr() as *const f64,
                            elem_count(&target.shape),
                        )
                    };
                    let o = unsafe {
                        std::slice::from_raw_parts_mut(grad.data.as_mut_ptr() as *mut f64, n)
                    };
                    let tn = elem_count(&target.shape);
                    if reduction == 0 {
                        for i in 0..n {
                            let ti = if tn == 1 { 0 } else { i % tn };
                            let ui = if un == 1 { 0 } else { i % un.max(1) };
                            let u_val = if un > 0 { up[ui] } else { 1.0 };
                            o[i] = u_val * scale * (inp[i] - tgt[ti]);
                        }
                    } else {
                        let u_val = if un > 0 { up[0] } else { 1.0 };
                        for i in 0..n {
                            let ti = if tn == 1 { 0 } else { i % tn };
                            o[i] = u_val * scale * (inp[i] - tgt[ti]);
                        }
                    }
                }
                _ => {}
            }
            vec![grad]
        }
        "sum" | "mean" => {
            assert!(!saved_inputs.is_empty());
            let input = saved_inputs[0];
            // Broadcast upstream back to input shape
            let n = elem_count(&input.shape);
            let un = elem_count(&upstream.shape);
            let scale = if target == "mean" && n > 0 {
                1.0 / n as f64
            } else {
                1.0
            };
            let mut grad = OwnedTensor::new(input.dtype, input.shape.clone());
            match input.dtype {
                DType::F32 => {
                    let g = unsafe {
                        std::slice::from_raw_parts(upstream.data.as_ptr() as *const f32, un)
                    };
                    let o = unsafe {
                        std::slice::from_raw_parts_mut(grad.data.as_mut_ptr() as *mut f32, n)
                    };
                    let s = scale as f32;
                    if un == 1 {
                        for i in 0..n {
                            o[i] = g[0] * s;
                        }
                    } else {
                        for i in 0..n.min(un) {
                            o[i] = g[i] * s;
                        }
                    }
                }
                DType::F64 => {
                    let g = unsafe {
                        std::slice::from_raw_parts(upstream.data.as_ptr() as *const f64, un)
                    };
                    let o = unsafe {
                        std::slice::from_raw_parts_mut(grad.data.as_mut_ptr() as *mut f64, n)
                    };
                    if un == 1 {
                        for i in 0..n {
                            o[i] = g[0] * scale;
                        }
                    } else {
                        for i in 0..n.min(un) {
                            o[i] = g[i] * scale;
                        }
                    }
                }
                _ => {}
            }
            vec![grad]
        }
        "div" => {
            assert!(saved_inputs.len() >= 2);
            let a = saved_inputs[0];
            let b = saved_inputs[1];
            let n = elem_count(&upstream.shape);
            let mut grad_a = OwnedTensor::new(upstream.dtype, upstream.shape.clone());
            let mut grad_b = OwnedTensor::new(upstream.dtype, upstream.shape.clone());
            match upstream.dtype {
                DType::F32 => {
                    let g = unsafe {
                        std::slice::from_raw_parts(upstream.data.as_ptr() as *const f32, n)
                    };
                    let ad = unsafe {
                        std::slice::from_raw_parts(
                            a.data.as_ptr() as *const f32,
                            elem_count(&a.shape),
                        )
                    };
                    let bd = unsafe {
                        std::slice::from_raw_parts(
                            b.data.as_ptr() as *const f32,
                            elem_count(&b.shape),
                        )
                    };
                    let ga = unsafe {
                        std::slice::from_raw_parts_mut(grad_a.data.as_mut_ptr() as *mut f32, n)
                    };
                    let gb = unsafe {
                        std::slice::from_raw_parts_mut(grad_b.data.as_mut_ptr() as *mut f32, n)
                    };
                    let an = elem_count(&a.shape);
                    let bn = elem_count(&b.shape);
                    for i in 0..n {
                        let ai = if an == 1 { 0 } else { i % an };
                        let bi = if bn == 1 { 0 } else { i % bn };
                        ga[i] = g[i] / bd[bi];
                        gb[i] = -g[i] * ad[ai] / (bd[bi] * bd[bi]);
                    }
                }
                DType::F64 => {
                    let g = unsafe {
                        std::slice::from_raw_parts(upstream.data.as_ptr() as *const f64, n)
                    };
                    let ad = unsafe {
                        std::slice::from_raw_parts(
                            a.data.as_ptr() as *const f64,
                            elem_count(&a.shape),
                        )
                    };
                    let bd = unsafe {
                        std::slice::from_raw_parts(
                            b.data.as_ptr() as *const f64,
                            elem_count(&b.shape),
                        )
                    };
                    let ga = unsafe {
                        std::slice::from_raw_parts_mut(grad_a.data.as_mut_ptr() as *mut f64, n)
                    };
                    let gb = unsafe {
                        std::slice::from_raw_parts_mut(grad_b.data.as_mut_ptr() as *mut f64, n)
                    };
                    let an = elem_count(&a.shape);
                    let bn = elem_count(&b.shape);
                    for i in 0..n {
                        let ai = if an == 1 { 0 } else { i % an };
                        let bi = if bn == 1 { 0 } else { i % bn };
                        ga[i] = g[i] / bd[bi];
                        gb[i] = -g[i] * ad[ai] / (bd[bi] * bd[bi]);
                    }
                }
                _ => {}
            }
            vec![grad_a, grad_b]
        }
        "pow" => {
            assert!(saved_inputs.len() >= 2);
            let a = saved_inputs[0];
            let b = saved_inputs[1];
            let n = elem_count(&upstream.shape);
            let mut grad_a = OwnedTensor::new(upstream.dtype, upstream.shape.clone());
            let mut grad_b = OwnedTensor::new(upstream.dtype, upstream.shape.clone());
            match upstream.dtype {
                DType::F32 => {
                    let g = unsafe {
                        std::slice::from_raw_parts(upstream.data.as_ptr() as *const f32, n)
                    };
                    let ad = unsafe {
                        std::slice::from_raw_parts(
                            a.data.as_ptr() as *const f32,
                            elem_count(&a.shape),
                        )
                    };
                    let bd = unsafe {
                        std::slice::from_raw_parts(
                            b.data.as_ptr() as *const f32,
                            elem_count(&b.shape),
                        )
                    };
                    let ga = unsafe {
                        std::slice::from_raw_parts_mut(grad_a.data.as_mut_ptr() as *mut f32, n)
                    };
                    let gb = unsafe {
                        std::slice::from_raw_parts_mut(grad_b.data.as_mut_ptr() as *mut f32, n)
                    };
                    let an = elem_count(&a.shape);
                    let bn = elem_count(&b.shape);
                    for i in 0..n {
                        let ai = if an == 1 { 0 } else { i % an };
                        let bi = if bn == 1 { 0 } else { i % bn };
                        let aval = ad[ai];
                        let bval = bd[bi];
                        ga[i] = g[i] * bval * aval.powf(bval - 1.0);
                        gb[i] = if aval > 0.0 {
                            g[i] * aval.powf(bval) * aval.ln()
                        } else {
                            0.0
                        };
                    }
                }
                DType::F64 => {
                    let g = unsafe {
                        std::slice::from_raw_parts(upstream.data.as_ptr() as *const f64, n)
                    };
                    let ad = unsafe {
                        std::slice::from_raw_parts(
                            a.data.as_ptr() as *const f64,
                            elem_count(&a.shape),
                        )
                    };
                    let bd = unsafe {
                        std::slice::from_raw_parts(
                            b.data.as_ptr() as *const f64,
                            elem_count(&b.shape),
                        )
                    };
                    let ga = unsafe {
                        std::slice::from_raw_parts_mut(grad_a.data.as_mut_ptr() as *mut f64, n)
                    };
                    let gb = unsafe {
                        std::slice::from_raw_parts_mut(grad_b.data.as_mut_ptr() as *mut f64, n)
                    };
                    let an = elem_count(&a.shape);
                    let bn = elem_count(&b.shape);
                    for i in 0..n {
                        let ai = if an == 1 { 0 } else { i % an };
                        let bi = if bn == 1 { 0 } else { i % bn };
                        let aval = ad[ai];
                        let bval = bd[bi];
                        ga[i] = g[i] * bval * aval.powf(bval - 1.0);
                        gb[i] = if aval > 0.0 {
                            g[i] * aval.powf(bval) * aval.ln()
                        } else {
                            0.0
                        };
                    }
                }
                _ => {}
            }
            vec![grad_a, grad_b]
        }
        "sigmoid" => {
            assert!(!saved_inputs.is_empty());
            // saved_inputs[0] is the OUTPUT of sigmoid (not input)
            let s = saved_inputs[0];
            let n = elem_count(&upstream.shape);
            let mut grad = OwnedTensor::new(upstream.dtype, upstream.shape.clone());
            match upstream.dtype {
                DType::F32 => {
                    let g = unsafe {
                        std::slice::from_raw_parts(upstream.data.as_ptr() as *const f32, n)
                    };
                    let sd = unsafe {
                        std::slice::from_raw_parts(
                            s.data.as_ptr() as *const f32,
                            elem_count(&s.shape),
                        )
                    };
                    let o = unsafe {
                        std::slice::from_raw_parts_mut(grad.data.as_mut_ptr() as *mut f32, n)
                    };
                    let sn = elem_count(&s.shape);
                    for i in 0..n {
                        let si = if sn == 1 { 0 } else { i % sn };
                        o[i] = g[i] * sd[si] * (1.0 - sd[si]);
                    }
                }
                DType::F64 => {
                    let g = unsafe {
                        std::slice::from_raw_parts(upstream.data.as_ptr() as *const f64, n)
                    };
                    let sd = unsafe {
                        std::slice::from_raw_parts(
                            s.data.as_ptr() as *const f64,
                            elem_count(&s.shape),
                        )
                    };
                    let o = unsafe {
                        std::slice::from_raw_parts_mut(grad.data.as_mut_ptr() as *mut f64, n)
                    };
                    let sn = elem_count(&s.shape);
                    for i in 0..n {
                        let si = if sn == 1 { 0 } else { i % sn };
                        o[i] = g[i] * sd[si] * (1.0 - sd[si]);
                    }
                }
                _ => {}
            }
            vec![grad]
        }
        "tanh" => {
            assert!(!saved_inputs.is_empty());
            // saved_inputs[0] is the OUTPUT of tanh
            let s = saved_inputs[0];
            let n = elem_count(&upstream.shape);
            let mut grad = OwnedTensor::new(upstream.dtype, upstream.shape.clone());
            match upstream.dtype {
                DType::F32 => {
                    let g = unsafe {
                        std::slice::from_raw_parts(upstream.data.as_ptr() as *const f32, n)
                    };
                    let sd = unsafe {
                        std::slice::from_raw_parts(
                            s.data.as_ptr() as *const f32,
                            elem_count(&s.shape),
                        )
                    };
                    let o = unsafe {
                        std::slice::from_raw_parts_mut(grad.data.as_mut_ptr() as *mut f32, n)
                    };
                    let sn = elem_count(&s.shape);
                    for i in 0..n {
                        let si = if sn == 1 { 0 } else { i % sn };
                        o[i] = g[i] * (1.0 - sd[si] * sd[si]);
                    }
                }
                DType::F64 => {
                    let g = unsafe {
                        std::slice::from_raw_parts(upstream.data.as_ptr() as *const f64, n)
                    };
                    let sd = unsafe {
                        std::slice::from_raw_parts(
                            s.data.as_ptr() as *const f64,
                            elem_count(&s.shape),
                        )
                    };
                    let o = unsafe {
                        std::slice::from_raw_parts_mut(grad.data.as_mut_ptr() as *mut f64, n)
                    };
                    let sn = elem_count(&s.shape);
                    for i in 0..n {
                        let si = if sn == 1 { 0 } else { i % sn };
                        o[i] = g[i] * (1.0 - sd[si] * sd[si]);
                    }
                }
                _ => {}
            }
            vec![grad]
        }
        "gelu" => {
            assert!(!saved_inputs.is_empty());
            // saved_inputs[0] is the INPUT to gelu
            let x = saved_inputs[0];
            let n = elem_count(&upstream.shape);
            let mut grad = OwnedTensor::new(upstream.dtype, upstream.shape.clone());
            match upstream.dtype {
                DType::F32 => {
                    let g = unsafe {
                        std::slice::from_raw_parts(upstream.data.as_ptr() as *const f32, n)
                    };
                    let xd = unsafe {
                        std::slice::from_raw_parts(
                            x.data.as_ptr() as *const f32,
                            elem_count(&x.shape),
                        )
                    };
                    let o = unsafe {
                        std::slice::from_raw_parts_mut(grad.data.as_mut_ptr() as *mut f32, n)
                    };
                    let xn = elem_count(&x.shape);
                    let c = 0.7978845608028654f32;
                    let b = 0.044715f32;
                    for i in 0..n {
                        let xi = if xn == 1 { 0 } else { i % xn };
                        let v = xd[xi];
                        let x3 = v * v * v;
                        let inner = c * (v + b * x3);
                        let tanh_inner = inner.tanh();
                        let sech2 = 1.0 - tanh_inner * tanh_inner;
                        let d_inner = c * (1.0 + 3.0 * b * v * v);
                        o[i] = g[i] * (0.5 * (1.0 + tanh_inner) + 0.5 * v * sech2 * d_inner);
                    }
                }
                DType::F64 => {
                    let g = unsafe {
                        std::slice::from_raw_parts(upstream.data.as_ptr() as *const f64, n)
                    };
                    let xd = unsafe {
                        std::slice::from_raw_parts(
                            x.data.as_ptr() as *const f64,
                            elem_count(&x.shape),
                        )
                    };
                    let o = unsafe {
                        std::slice::from_raw_parts_mut(grad.data.as_mut_ptr() as *mut f64, n)
                    };
                    let xn = elem_count(&x.shape);
                    let c = 0.7978845608028654f64;
                    let b = 0.044715f64;
                    for i in 0..n {
                        let xi = if xn == 1 { 0 } else { i % xn };
                        let v = xd[xi];
                        let x3 = v * v * v;
                        let inner = c * (v + b * x3);
                        let tanh_inner = inner.tanh();
                        let sech2 = 1.0 - tanh_inner * tanh_inner;
                        let d_inner = c * (1.0 + 3.0 * b * v * v);
                        o[i] = g[i] * (0.5 * (1.0 + tanh_inner) + 0.5 * v * sech2 * d_inner);
                    }
                }
                _ => {}
            }
            vec![grad]
        }
        "linear" => {
            // saved_inputs: [input, weight, optional_bias]
            // grad_output is (..., out_features)
            assert!(saved_inputs.len() >= 2);
            let input = saved_inputs[0];
            let weight = saved_inputs[1];
            // weight is (out_features, in_features)
            // grad_input = grad_output @ weight
            let m = upstream.shape[upstream.shape.len() - 1] as usize; // out_features
            let k = weight.shape[1] as usize; // in_features
            let batch: usize = upstream.shape[..upstream.shape.len() - 1]
                .iter()
                .map(|&d| d.max(0) as usize)
                .product();
            let mut grad_input = OwnedTensor::new(input.dtype, input.shape.clone());
            let mut grad_weight = OwnedTensor::new(weight.dtype, weight.shape.clone());
            match upstream.dtype {
                DType::F32 => {
                    let g = unsafe {
                        std::slice::from_raw_parts(upstream.data.as_ptr() as *const f32, batch * m)
                    };
                    let wd = unsafe {
                        std::slice::from_raw_parts(weight.data.as_ptr() as *const f32, m * k)
                    };
                    let gi = unsafe {
                        std::slice::from_raw_parts_mut(
                            grad_input.data.as_mut_ptr() as *mut f32,
                            batch * k,
                        )
                    };
                    let gw = unsafe {
                        std::slice::from_raw_parts_mut(
                            grad_weight.data.as_mut_ptr() as *mut f32,
                            m * k,
                        )
                    };
                    // grad_input = g @ W
                    for b in 0..batch {
                        for j in 0..k {
                            let mut s = 0.0f32;
                            for i in 0..m {
                                s += g[b * m + i] * wd[i * k + j];
                            }
                            gi[b * k + j] = s;
                        }
                    }
                    // grad_weight = grad_output^T @ input (use input data, not grad_input)
                    let id = unsafe {
                        std::slice::from_raw_parts(input.data.as_ptr() as *const f32, batch * k)
                    };
                    for i in 0..m {
                        for j in 0..k {
                            let mut s = 0.0f32;
                            for bi in 0..batch {
                                s += g[bi * m + i] * id[bi * k + j];
                            }
                            gw[i * k + j] = s;
                        }
                    }
                }
                DType::F64 => {
                    let g = unsafe {
                        std::slice::from_raw_parts(upstream.data.as_ptr() as *const f64, batch * m)
                    };
                    let wd = unsafe {
                        std::slice::from_raw_parts(weight.data.as_ptr() as *const f64, m * k)
                    };
                    let gi = unsafe {
                        std::slice::from_raw_parts_mut(
                            grad_input.data.as_mut_ptr() as *mut f64,
                            batch * k,
                        )
                    };
                    let gw = unsafe {
                        std::slice::from_raw_parts_mut(
                            grad_weight.data.as_mut_ptr() as *mut f64,
                            m * k,
                        )
                    };
                    for b in 0..batch {
                        for j in 0..k {
                            let mut s = 0.0f64;
                            for i in 0..m {
                                s += g[b * m + i] * wd[i * k + j];
                            }
                            gi[b * k + j] = s;
                        }
                    }
                    // grad_weight = grad_output^T @ input (use input data, not grad_input)
                    let id = unsafe {
                        std::slice::from_raw_parts(input.data.as_ptr() as *const f64, batch * k)
                    };
                    for i in 0..m {
                        for j in 0..k {
                            let mut s = 0.0f64;
                            for bi in 0..batch {
                                s += g[bi * m + i] * id[bi * k + j];
                            }
                            gw[i * k + j] = s;
                        }
                    }
                }
                _ => {}
            }
            let mut result = vec![grad_input, grad_weight];
            if saved_inputs.len() > 2 {
                // grad_bias = sum(g, dim=batch_dims)
                let bias = saved_inputs[2];
                let mut grad_bias = OwnedTensor::new(bias.dtype, bias.shape.clone());
                match upstream.dtype {
                    DType::F32 => {
                        let g = unsafe {
                            std::slice::from_raw_parts(
                                upstream.data.as_ptr() as *const f32,
                                batch * m,
                            )
                        };
                        let gb = unsafe {
                            std::slice::from_raw_parts_mut(
                                grad_bias.data.as_mut_ptr() as *mut f32,
                                m,
                            )
                        };
                        gb.fill(0.0);
                        for b in 0..batch {
                            for i in 0..m {
                                gb[i] += g[b * m + i];
                            }
                        }
                    }
                    DType::F64 => {
                        let g = unsafe {
                            std::slice::from_raw_parts(
                                upstream.data.as_ptr() as *const f64,
                                batch * m,
                            )
                        };
                        let gb = unsafe {
                            std::slice::from_raw_parts_mut(
                                grad_bias.data.as_mut_ptr() as *mut f64,
                                m,
                            )
                        };
                        gb.fill(0.0);
                        for b in 0..batch {
                            for i in 0..m {
                                gb[i] += g[b * m + i];
                            }
                        }
                    }
                    _ => {}
                }
                result.push(grad_bias);
            }
            result
        }
        "layer_norm" => {
            // saved_inputs: [input, weight, optional_bias]
            // Proper layer_norm backward:
            //   x_hat = (x - mean) / sqrt(var + eps)
            //   grad_x_hat = grad_output * weight
            //   grad_x = inv_std * (grad_x_hat - mean(grad_x_hat) - x_hat * mean(grad_x_hat * x_hat))
            //   grad_weight = sum_over_batch(grad_output * x_hat)
            //   grad_bias = sum_over_batch(grad_output)
            assert!(saved_inputs.len() >= 2);
            let input = saved_inputs[0];
            let weight = saved_inputs[1];
            let last_dim = *input.shape.last().unwrap_or(&1) as usize;
            let n = elem_count(&input.shape);
            let batch = if last_dim > 0 { n / last_dim } else { 1 };
            let mut grad_input = OwnedTensor::new(input.dtype, input.shape.clone());
            let mut grad_weight = OwnedTensor::new(weight.dtype, weight.shape.clone());
            match upstream.dtype {
                DType::F32 => {
                    let g = unsafe {
                        std::slice::from_raw_parts(upstream.data.as_ptr() as *const f32, n)
                    };
                    let xd =
                        unsafe { std::slice::from_raw_parts(input.data.as_ptr() as *const f32, n) };
                    let wd = unsafe {
                        std::slice::from_raw_parts(weight.data.as_ptr() as *const f32, last_dim)
                    };
                    let gi = unsafe {
                        std::slice::from_raw_parts_mut(grad_input.data.as_mut_ptr() as *mut f32, n)
                    };
                    let gw = unsafe {
                        std::slice::from_raw_parts_mut(
                            grad_weight.data.as_mut_ptr() as *mut f32,
                            last_dim,
                        )
                    };
                    gw.fill(0.0);
                    for b in 0..batch {
                        let base = b * last_dim;
                        let mut mu = 0.0f32;
                        for j in 0..last_dim {
                            mu += xd[base + j];
                        }
                        mu /= last_dim as f32;
                        let mut var = 0.0f32;
                        for j in 0..last_dim {
                            let d = xd[base + j] - mu;
                            var += d * d;
                        }
                        var /= last_dim as f32;
                        let inv_std = 1.0f32 / (var + 1e-5f32).sqrt();
                        // grad_x_hat = g * weight
                        // mean(grad_x_hat)
                        let mut ghat_mean = 0.0f32;
                        for j in 0..last_dim {
                            ghat_mean += g[base + j] * wd[j];
                        }
                        ghat_mean /= last_dim as f32;
                        // mean(grad_x_hat * x_hat)
                        let mut ghat_xhat_mean = 0.0f32;
                        for j in 0..last_dim {
                            let xh = (xd[base + j] - mu) * inv_std;
                            ghat_xhat_mean += g[base + j] * wd[j] * xh;
                        }
                        ghat_xhat_mean /= last_dim as f32;
                        for j in 0..last_dim {
                            let xh = (xd[base + j] - mu) * inv_std;
                            gi[base + j] =
                                inv_std * (g[base + j] * wd[j] - ghat_mean - xh * ghat_xhat_mean);
                            gw[j] += g[base + j] * xh;
                        }
                    }
                }
                DType::F64 => {
                    let g = unsafe {
                        std::slice::from_raw_parts(upstream.data.as_ptr() as *const f64, n)
                    };
                    let xd =
                        unsafe { std::slice::from_raw_parts(input.data.as_ptr() as *const f64, n) };
                    let wd = unsafe {
                        std::slice::from_raw_parts(weight.data.as_ptr() as *const f64, last_dim)
                    };
                    let gi = unsafe {
                        std::slice::from_raw_parts_mut(grad_input.data.as_mut_ptr() as *mut f64, n)
                    };
                    let gw = unsafe {
                        std::slice::from_raw_parts_mut(
                            grad_weight.data.as_mut_ptr() as *mut f64,
                            last_dim,
                        )
                    };
                    gw.fill(0.0);
                    for b in 0..batch {
                        let base = b * last_dim;
                        let mut mu = 0.0f64;
                        for j in 0..last_dim {
                            mu += xd[base + j];
                        }
                        mu /= last_dim as f64;
                        let mut var = 0.0f64;
                        for j in 0..last_dim {
                            let d = xd[base + j] - mu;
                            var += d * d;
                        }
                        var /= last_dim as f64;
                        let inv_std = 1.0f64 / (var + 1e-5f64).sqrt();
                        let mut ghat_mean = 0.0f64;
                        for j in 0..last_dim {
                            ghat_mean += g[base + j] * wd[j];
                        }
                        ghat_mean /= last_dim as f64;
                        let mut ghat_xhat_mean = 0.0f64;
                        for j in 0..last_dim {
                            let xh = (xd[base + j] - mu) * inv_std;
                            ghat_xhat_mean += g[base + j] * wd[j] * xh;
                        }
                        ghat_xhat_mean /= last_dim as f64;
                        for j in 0..last_dim {
                            let xh = (xd[base + j] - mu) * inv_std;
                            gi[base + j] =
                                inv_std * (g[base + j] * wd[j] - ghat_mean - xh * ghat_xhat_mean);
                            gw[j] += g[base + j] * xh;
                        }
                    }
                }
                _ => {}
            }
            let mut result = vec![grad_input, grad_weight];
            if saved_inputs.len() > 2 {
                let bias = saved_inputs[2];
                let mut grad_bias = OwnedTensor::new(bias.dtype, bias.shape.clone());
                match upstream.dtype {
                    DType::F32 => {
                        let g = unsafe {
                            std::slice::from_raw_parts(upstream.data.as_ptr() as *const f32, n)
                        };
                        let gb = unsafe {
                            std::slice::from_raw_parts_mut(
                                grad_bias.data.as_mut_ptr() as *mut f32,
                                last_dim,
                            )
                        };
                        gb.fill(0.0);
                        for b in 0..batch {
                            for j in 0..last_dim {
                                gb[j] += g[b * last_dim + j];
                            }
                        }
                    }
                    DType::F64 => {
                        let g = unsafe {
                            std::slice::from_raw_parts(upstream.data.as_ptr() as *const f64, n)
                        };
                        let gb = unsafe {
                            std::slice::from_raw_parts_mut(
                                grad_bias.data.as_mut_ptr() as *mut f64,
                                last_dim,
                            )
                        };
                        gb.fill(0.0);
                        for b in 0..batch {
                            for j in 0..last_dim {
                                gb[j] += g[b * last_dim + j];
                            }
                        }
                    }
                    _ => {}
                }
                result.push(grad_bias);
            }
            result
        }
        "softmax" => {
            // saved_inputs[0] = output of softmax
            assert!(!saved_inputs.is_empty());
            let s = saved_inputs[0];
            let n = elem_count(&upstream.shape);
            let mut grad = OwnedTensor::new(upstream.dtype, upstream.shape.clone());
            match upstream.dtype {
                DType::F32 => {
                    let g = unsafe {
                        std::slice::from_raw_parts(upstream.data.as_ptr() as *const f32, n)
                    };
                    let sd = unsafe {
                        std::slice::from_raw_parts(
                            s.data.as_ptr() as *const f32,
                            elem_count(&s.shape),
                        )
                    };
                    let o = unsafe {
                        std::slice::from_raw_parts_mut(grad.data.as_mut_ptr() as *mut f32, n)
                    };
                    let dim_size = *upstream.shape.last().unwrap_or(&1) as usize;
                    if dim_size > 0 {
                        for base in (0..n).step_by(dim_size) {
                            let mut dot = 0.0f32;
                            for j in 0..dim_size {
                                dot += sd[base + j] * g[base + j];
                            }
                            for j in 0..dim_size {
                                o[base + j] = sd[base + j] * (g[base + j] - dot);
                            }
                        }
                    }
                }
                DType::F64 => {
                    let g = unsafe {
                        std::slice::from_raw_parts(upstream.data.as_ptr() as *const f64, n)
                    };
                    let sd = unsafe {
                        std::slice::from_raw_parts(
                            s.data.as_ptr() as *const f64,
                            elem_count(&s.shape),
                        )
                    };
                    let o = unsafe {
                        std::slice::from_raw_parts_mut(grad.data.as_mut_ptr() as *mut f64, n)
                    };
                    let dim_size = *upstream.shape.last().unwrap_or(&1) as usize;
                    if dim_size > 0 {
                        for base in (0..n).step_by(dim_size) {
                            let mut dot = 0.0f64;
                            for j in 0..dim_size {
                                dot += sd[base + j] * g[base + j];
                            }
                            for j in 0..dim_size {
                                o[base + j] = sd[base + j] * (g[base + j] - dot);
                            }
                        }
                    }
                }
                _ => {}
            }
            vec![grad]
        }
        "nll_loss" => {
            assert!(saved_inputs.len() >= 2);
            let input = saved_inputs[0];
            let target = saved_inputs[1];
            let n_batch = input.shape[0] as usize;
            let n_classes = if input.shape.len() > 1 {
                *input.shape.last().unwrap_or(&1) as usize
            } else {
                1
            };
            let reduction_str = kwargs
                .get("reduction")
                .and_then(|v| v.as_str())
                .unwrap_or("mean");
            let scale = match reduction_str {
                "mean" => 1.0 / n_batch as f64,
                "sum" => 1.0,
                _ => 1.0,
            };
            let mut grad = OwnedTensor::new(input.dtype, input.shape.clone());
            match input.dtype {
                DType::F32 => {
                    let g_raw = unsafe {
                        std::slice::from_raw_parts(
                            upstream.data.as_ptr() as *const f32,
                            elem_count(&upstream.shape),
                        )
                    };
                    let tgt = unsafe {
                        std::slice::from_raw_parts(target.data.as_ptr() as *const i64, n_batch)
                    };
                    let o = unsafe {
                        std::slice::from_raw_parts_mut(
                            grad.data.as_mut_ptr() as *mut f32,
                            n_batch * n_classes,
                        )
                    };
                    o.fill(0.0);
                    let g_val = if g_raw.len() == 1 { g_raw[0] } else { 0.0 };
                    for b in 0..n_batch {
                        let t = tgt[b] as usize;
                        if t < n_classes {
                            o[b * n_classes + t] = -(scale as f32) * g_val;
                        }
                    }
                }
                DType::F64 => {
                    let g_raw = unsafe {
                        std::slice::from_raw_parts(
                            upstream.data.as_ptr() as *const f64,
                            elem_count(&upstream.shape),
                        )
                    };
                    let tgt = unsafe {
                        std::slice::from_raw_parts(target.data.as_ptr() as *const i64, n_batch)
                    };
                    let o = unsafe {
                        std::slice::from_raw_parts_mut(
                            grad.data.as_mut_ptr() as *mut f64,
                            n_batch * n_classes,
                        )
                    };
                    o.fill(0.0);
                    let g_val = if g_raw.len() == 1 { g_raw[0] } else { 0.0 };
                    for b in 0..n_batch {
                        let t = tgt[b] as usize;
                        if t < n_classes {
                            o[b * n_classes + t] = -scale * g_val;
                        }
                    }
                }
                _ => {}
            }
            vec![grad]
        }
        "cross_entropy" => {
            // cross_entropy(input, target) = nll_loss(log_softmax(input), target)
            // grad = scale * upstream * (softmax(input) - one_hot(target))
            assert!(saved_inputs.len() >= 2);
            let input = saved_inputs[0];
            let target = saved_inputs[1];
            let n_batch = input.shape[0] as usize;
            let n_classes = if input.shape.len() > 1 {
                *input.shape.last().unwrap_or(&1) as usize
            } else {
                1
            };
            let reduction_str = kwargs
                .get("reduction")
                .and_then(|v| v.as_str())
                .unwrap_or("mean");
            let scale = match reduction_str {
                "mean" => 1.0 / n_batch as f64,
                "sum" => 1.0,
                _ => 1.0,
            };
            let n_total = n_batch * n_classes;
            let mut grad = OwnedTensor::new(input.dtype, input.shape.clone());
            match input.dtype {
                DType::F32 => {
                    let g_raw = unsafe {
                        std::slice::from_raw_parts(
                            upstream.data.as_ptr() as *const f32,
                            elem_count(&upstream.shape),
                        )
                    };
                    let inp = unsafe {
                        std::slice::from_raw_parts(input.data.as_ptr() as *const f32, n_total)
                    };
                    let tgt = unsafe {
                        std::slice::from_raw_parts(target.data.as_ptr() as *const i64, n_batch)
                    };
                    let o = unsafe {
                        std::slice::from_raw_parts_mut(grad.data.as_mut_ptr() as *mut f32, n_total)
                    };
                    let g_val = if g_raw.len() == 1 { g_raw[0] } else { 1.0f32 };
                    for b in 0..n_batch {
                        // compute softmax for this row
                        let mut max_val = f32::NEG_INFINITY;
                        for c in 0..n_classes {
                            let v = inp[b * n_classes + c];
                            if v > max_val {
                                max_val = v;
                            }
                        }
                        let mut sum_exp = 0.0f32;
                        for c in 0..n_classes {
                            sum_exp += (inp[b * n_classes + c] - max_val).exp();
                        }
                        for c in 0..n_classes {
                            let prob = (inp[b * n_classes + c] - max_val).exp() / sum_exp;
                            let target_one_hot = if c == tgt[b] as usize { 1.0f32 } else { 0.0f32 };
                            o[b * n_classes + c] = (scale as f32) * g_val * (prob - target_one_hot);
                        }
                    }
                }
                DType::F64 => {
                    let g_raw = unsafe {
                        std::slice::from_raw_parts(
                            upstream.data.as_ptr() as *const f64,
                            elem_count(&upstream.shape),
                        )
                    };
                    let inp = unsafe {
                        std::slice::from_raw_parts(input.data.as_ptr() as *const f64, n_total)
                    };
                    let tgt = unsafe {
                        std::slice::from_raw_parts(target.data.as_ptr() as *const i64, n_batch)
                    };
                    let o = unsafe {
                        std::slice::from_raw_parts_mut(grad.data.as_mut_ptr() as *mut f64, n_total)
                    };
                    let g_val = if g_raw.len() == 1 { g_raw[0] } else { 1.0f64 };
                    for b in 0..n_batch {
                        let mut max_val = f64::NEG_INFINITY;
                        for c in 0..n_classes {
                            let v = inp[b * n_classes + c];
                            if v > max_val {
                                max_val = v;
                            }
                        }
                        let mut sum_exp = 0.0f64;
                        for c in 0..n_classes {
                            sum_exp += (inp[b * n_classes + c] - max_val).exp();
                        }
                        for c in 0..n_classes {
                            let prob = (inp[b * n_classes + c] - max_val).exp() / sum_exp;
                            let target_one_hot = if c == tgt[b] as usize { 1.0f64 } else { 0.0f64 };
                            o[b * n_classes + c] = scale * g_val * (prob - target_one_hot);
                        }
                    }
                }
                _ => {}
            }
            vec![grad]
        }
        _ => {
            // Unsupported backward: return zero gradients
            let mut grads = Vec::new();
            for input in saved_inputs {
                grads.push(OwnedTensor::new(input.dtype, input.shape.clone()));
            }
            grads
        }
    }
}
