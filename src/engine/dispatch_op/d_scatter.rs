//! Dispatch arms: Scatter, topk/sort, repeat, einsum, prelu, nonzero. Inherits engine root via super; pure move.

use super::*;

pub(crate) fn try_dispatch(
    node: &Node,
    slots: &mut Vec<Slot>,
    capsules: &[CapsuleRef],
) -> PyResult<bool> {
    let target = node.target.as_str();

    match target {
        // Phase 7: extended ops
        "scatter" => {
            // call_method: scatter(self, dim, index, src) -> 4 args
            // call_function: scatter(src, dim, index) -> 3 args
            let is_method = node.args.len() >= 4;
            if is_method {
                // Read dim from slot (arg 1 is a scalar tensor)
                let dim = node
                    .args
                    .get(1)
                    .and_then(|a| a.index)
                    .and_then(|idx| match slots.get(idx) {
                        Some(Slot::Owned(t)) => unsafe {
                            match t.dtype {
                                DType::I64 => Some(*(t.data.as_ptr() as *const i64) as isize),
                                DType::F32 => Some(*(t.data.as_ptr() as *const f32) as isize),
                                _ => None,
                            }
                        },
                        Some(Slot::Input(i)) => unsafe {
                            let bt = BorrowedTensor::from_managed(capsules[*i].0);
                            bt.ok().and_then(|b| match b.dtype {
                                DType::I64 => Some(*(b.data as *const i64) as isize),
                                DType::F32 => Some(*(b.data as *const f32) as isize),
                                _ => None,
                            })
                        },
                        _ => None,
                    })
                    .unwrap_or(0);
                let self_tensor = slot_view(slots, capsules, arg_index(node, 0)?)?;
                let index = slot_view(slots, capsules, arg_index(node, 2)?)?;
                let src = slot_view(slots, capsules, arg_index(node, 3)?)?;
                slots.push(Slot::Owned(kernels::special::scatter_method(
                    &self_tensor,
                    dim,
                    &index,
                    &src,
                )?));
            } else {
                let dim = kw_isize(node, "dim", 0);
                let src = slot_view(slots, capsules, arg_index(node, 0)?)?;
                let index = slot_view(slots, capsules, arg_index(node, 1)?)?;
                slots.push(Slot::Owned(kernels::special::scatter(&src, dim, &index)?));
            }
        }
        "scatter_add" => {
            let (dim, idx_pos, src_pos): (isize, usize, usize) = if node.args.len() >= 4 {
                (kw_isize(node, "dim", -1), 2, 3)
            } else {
                (kw_isize(node, "dim", 0), 1, 0)
            };
            let dim = if dim >= 0 {
                dim
            } else {
                node.args
                    .get(1)
                    .and_then(|a| a.index)
                    .and_then(|idx| match slots.get(idx) {
                        Some(Slot::Owned(t)) => unsafe {
                            match t.dtype {
                                DType::I64 => Some(*(t.data.as_ptr() as *const i64) as isize),
                                DType::F32 => Some(*(t.data.as_ptr() as *const f32) as isize),
                                _ => None,
                            }
                        },
                        Some(Slot::Input(i)) => unsafe {
                            let bt = BorrowedTensor::from_managed(capsules[*i].0);
                            bt.ok().and_then(|b| match b.dtype {
                                DType::I64 => Some(*(b.data as *const i64) as isize),
                                DType::F32 => Some(*(b.data as *const f32) as isize),
                                _ => None,
                            })
                        },
                        _ => None,
                    })
                    .unwrap_or(0)
            };
            let index = slot_view(slots, capsules, arg_index(node, idx_pos)?)?;
            let src = slot_view(slots, capsules, arg_index(node, src_pos)?)?;
            slots.push(Slot::Owned(kernels::special::scatter_add(
                &src, dim, &index,
            )?));
        }
        "topk" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let k = kw_usize(node, "k", 1);
            let dim = kw_isize(node, "dim", -1);
            let largest = kw_bool(node, "largest", true);
            let (values, indices) = kernels::special::topk(&a, k, dim, largest)?;
            slots.push(Slot::Tuple(vec![values, indices]));
        }
        // sort returns only values (indices dropped by parser aliasing)
        "sort" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let dim = kw_isize(node, "dim", -1);
            let descending = kw_bool(node, "descending", false);
            let (values, indices) = kernels::special::sort(&a, dim, descending)?;
            slots.push(Slot::Tuple(vec![values, indices]));
        }
        "argsort" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let dim = kw_isize(node, "dim", -1);
            let descending = kw_bool(node, "descending", false);
            slots.push(Slot::Owned(kernels::special::argsort(&a, dim, descending)?));
        }
        "repeat_interleave" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let reps = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let dim = kw_isize(node, "dim", 0);
            slots.push(Slot::Owned(kernels::special::repeat_interleave(
                &a, &reps, dim,
            )?));
        }
        "repeat" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            // repeat() is emitted as call_method with positional args: [self, *repeats]
            // The repeats come from slots (constant tensors promoted to capsules).
            let reps: Vec<i64> = node.args[1..]
                .iter()
                .filter_map(|arg| {
                    // Try slot reference first (constant tensor in capsule)
                    if let Some(idx) = arg.index {
                        match slots.get(idx) {
                            Some(Slot::Owned(t)) => {
                                return unsafe {
                                    match t.dtype {
                                        DType::F32 => Some(*(t.data.as_ptr() as *const f32) as i64),
                                        DType::F64 => Some(*(t.data.as_ptr() as *const f64) as i64),
                                        DType::I64 => Some(*(t.data.as_ptr() as *const i64)),
                                        _ => None,
                                    }
                                };
                            }
                            Some(Slot::Input(i)) => {
                                let bt = unsafe { BorrowedTensor::from_managed(capsules[*i].0) };
                                if let Ok(bt) = bt {
                                    return unsafe {
                                        match bt.dtype {
                                            DType::F32 => Some(*(bt.data as *const f32) as i64),
                                            DType::F64 => Some(*(bt.data as *const f64) as i64),
                                            DType::I64 => Some(*(bt.data as *const i64)),
                                            _ => None,
                                        }
                                    };
                                }
                            }
                            _ => {}
                        }
                    }
                    // Fall back to value field
                    arg.value
                        .as_ref()
                        .and_then(|v| v.as_i64().or_else(|| v.as_f64().map(|f| f as i64)))
                })
                .collect();
            slots.push(Slot::Owned(kernels::special::repeat(&a, &reps)?));
        }
        "einsum" => {
            let eq = node
                .kwargs
                .get("equation")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let tensor_indices: Vec<usize> = node.args.iter().filter_map(|a| a.index).collect();
            let tensors: Vec<BorrowedTensor> = tensor_indices
                .iter()
                .map(|&i| slot_view(slots, capsules, i))
                .collect::<PyResult<_>>()?;
            let refs: Vec<&BorrowedTensor> = tensors.iter().collect();
            slots.push(Slot::Owned(kernels::special::einsum(eq, &refs)?));
        }
        "prelu" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let w = slot_view(slots, capsules, arg_index(node, 1)?)?;
            slots.push(Slot::Owned(kernels::special::prelu(&a, &w)?));
        }
        "nonzero" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(kernels::special::nonzero(&a)?));
        }
        "clamp_tensor" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let lo = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let hi = slot_view(slots, capsules, arg_index(node, 2)?)?;
            slots.push(Slot::Owned(kernels::special::clamp_tensor(&a, &lo, &hi)?));
        }
        _ => return Ok(false),
    }
    Ok(true)
}
