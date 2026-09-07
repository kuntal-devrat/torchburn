//! Dispatch arms: Shape, select, gather, chunk, squeeze/unsqueeze. Inherits engine root via super; pure move.

use super::*;
use std::sync::Arc;

pub(crate) fn try_dispatch(
    node: &Node,
    slots: &mut Vec<Slot>,
    capsules: &[CapsuleRef],
) -> PyResult<bool> {
    let target = node.target.as_str();

    match target {
        // Phase 2: shape ops
        "cat" => {
            // Two calling conventions:
            // 1. args = [{kind:"slot", value:[i,j,...]}, {dim kwarg}] — list-in-first-arg
            // 2. args = [{kind:"slot", index:i}, {kind:"slot", index:j}, ...] — flat args
            let indices_arg = node
                .args
                .get(0)
                .ok_or_else(|| unsupported("cat: missing tensors argument"))?;
            let tensor_indices: Vec<usize> =
                if let Some(arr) = indices_arg.value.as_ref().and_then(|v| v.as_array()) {
                    // Convention 1: first arg holds a JSON array of slot indices
                    arr.iter()
                        .filter_map(|v| v.as_u64().map(|x| x as usize))
                        .collect()
                } else {
                    // Convention 2: all positional args are individual slot refs
                    node.args.iter().filter_map(|a| a.index).collect()
                };
            if tensor_indices.is_empty() {
                return Err(unsupported("cat: no tensor slot indices found"));
            }
            let tensors: Vec<BorrowedTensor> = tensor_indices
                .iter()
                .map(|&i| slot_view(slots, capsules, i))
                .collect::<PyResult<_>>()?;
            let dim = kw_isize(node, "dim", 0);
            slots.push(Slot::Owned(shape_ops::cat(&tensors, dim)?));
        }
        "stack" => {
            let indices_arg = node
                .args
                .get(0)
                .ok_or_else(|| unsupported("stack: missing tensors argument"))?;
            let tensor_indices: Vec<usize> =
                if let Some(arr) = indices_arg.value.as_ref().and_then(|v| v.as_array()) {
                    arr.iter()
                        .filter_map(|v| v.as_u64().map(|x| x as usize))
                        .collect()
                } else {
                    node.args.iter().filter_map(|a| a.index).collect()
                };
            if tensor_indices.is_empty() {
                return Err(unsupported("stack: no tensor slot indices found"));
            }
            let tensors: Vec<BorrowedTensor> = tensor_indices
                .iter()
                .map(|&i| slot_view(slots, capsules, i))
                .collect::<PyResult<_>>()?;
            let dim = kw_isize(node, "dim", 0);
            slots.push(Slot::Owned(shape_ops::stack(&tensors, dim)?));
        }
        "reshape" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            // Shape comes from kwargs["shape"] (set by the parser's seq/const
            // promotion).  A missing shape means the parser failed to convey
            // it — reject loudly rather than silently returning a copy.
            let shape = kw_i64_vec(node, "shape");
            if shape.is_empty() {
                return Err(unsupported("reshape: shape not conveyed via kwargs"));
            }
            let resolved = shape_ops::resolve_shape(&a.shape, &shape)?;
            if a.strides == contiguous_strides(&a.shape) {
                let new_strides = contiguous_strides(&resolved);
                slots.push(Slot::View {
                    data: a.data,
                    shape: Arc::from(resolved),
                    strides: Arc::from(new_strides),
                    dtype: a.dtype,
                });
            } else {
                slots.push(Slot::Owned(shape_ops::reshape(&a, &shape)?));
            }
        }
        "permute" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let dims = kw_isize_vec(node, "dims");
            if dims.is_empty() {
                return Err(unsupported("permute: no dims provided"));
            }
            let (new_shape, new_strides) = shape_ops::permute_view(&a, &dims)?;
            slots.push(Slot::View {
                data: a.data,
                shape: Arc::from(new_shape),
                strides: Arc::from(new_strides),
                dtype: a.dtype,
            });
        }
        "transpose" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let d0 = kw_isize(node, "d0", 0);
            let d1 = kw_isize(node, "d1", 1);
            let (new_shape, new_strides) = shape_ops::transpose_view(&a, d0, d1)?;
            slots.push(Slot::View {
                data: a.data,
                shape: Arc::from(new_shape),
                strides: Arc::from(new_strides),
                dtype: a.dtype,
            });
        }
        "index_select" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let index = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let dim = kw_isize(node, "dim", 0);
            slots.push(Slot::Owned(shape_ops::index_select(&a, dim, &index)?));
        }
        "gather" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let index = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let dim = kw_isize(node, "dim", 0);
            slots.push(Slot::Owned(shape_ops::gather(&a, dim, &index)?));
        }
        "chunk" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let num_chunks = kw_usize(node, "chunks", 1);
            let dim = kw_isize(node, "dim", 0);
            let ndim = a.shape.len() as isize;
            let normalized_dim = if dim < 0 { dim + ndim } else { dim };
            let dim_size = a.shape[normalized_dim as usize] as usize;
            let chunk_size = dim_size / num_chunks;
            let mut parts = Vec::with_capacity(num_chunks);
            for i in 0..num_chunks {
                let start = i * chunk_size;
                let length = if i == num_chunks - 1 {
                    dim_size - start
                } else {
                    chunk_size
                };
                parts.push(shape_ops::narrow(&a, dim, start, length)?);
            }
            slots.push(Slot::Tuple(parts));
        }
        "unbind" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let dim = kw_isize(node, "dim", 0);
            let ndim = a.shape.len() as isize;
            let normalized_dim = if dim < 0 { dim + ndim } else { dim };
            let dim_size = a.shape[normalized_dim as usize] as usize;
            let mut parts = Vec::with_capacity(dim_size);
            for i in 0..dim_size {
                parts.push(shape_ops::select(&a, dim, i)?);
            }
            slots.push(Slot::Tuple(parts));
        }
        "t" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            if a.shape.len() > 2 {
                return Err(unsupported("t: input must be <= 2D"));
            }
            if a.shape.len() <= 1 {
                slots.push(Slot::View {
                    data: a.data,
                    shape: Arc::from(a.shape.as_slice()),
                    strides: Arc::from(a.strides.as_slice()),
                    dtype: a.dtype,
                });
            } else {
                let (new_shape, new_strides) = shape_ops::transpose_view(&a, 0, 1)?;
                slots.push(Slot::View {
                    data: a.data,
                    shape: Arc::from(new_shape),
                    strides: Arc::from(new_strides),
                    dtype: a.dtype,
                });
            }
        }
        "expand" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let shape = kw_i64_vec(node, "shape");
            if shape.is_empty() {
                return Err(unsupported("expand: no shape provided"));
            }
            slots.push(Slot::Owned(shape_ops::expand(&a, &shape)?));
        }
        "where" => {
            let cond = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let x = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let y = slot_view(slots, capsules, arg_index(node, 2)?)?;
            slots.push(Slot::Owned(shape_ops::where_op(&cond, &x, &y)?));
        }
        "masked_fill" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let mask = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let value = kw_f64(node, "value", 0.0);
            slots.push(Slot::Owned(shape_ops::masked_fill(&a, &mask, value)?));
        }
        "flip" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let dims: Vec<isize> = kw_isize_vec(node, "dims");
            slots.push(Slot::Owned(shape_ops::flip(&a, &dims)?));
        }
        "narrow" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let dim = kw_isize(node, "dim", 0);
            let start = kw_usize(node, "start", 0);
            let length = kw_usize(node, "length", 0);
            slots.push(Slot::Owned(shape_ops::narrow(&a, dim, start, length)?));
        }
        "select" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let dim = kw_isize(node, "dim", 0);
            let index = kw_usize(node, "index", 0);
            slots.push(Slot::Owned(shape_ops::select(&a, dim, index)?));
        }
        "getitem" => {
            // getitem(tuple_slot, index) -> tuple_slot[index]
            let slot_idx = arg_index(node, 0)?;
            let elem = kw_usize(node, "index", 0);
            let len = slots[slot_idx].tuple_len();
            if len == 0 {
                return Err(unsupported(&format!(
                    "getitem: slot {slot_idx} is not a tuple"
                )));
            }
            if elem >= len {
                return Err(unsupported(&format!(
                    "getitem: index {elem} out of range for tuple of len {len}"
                )));
            }
            if let Slot::Tuple(elems) = &slots[slot_idx] {
                slots.push(Slot::Owned(elems[elem].clone()));
            } else {
                return Err(unsupported(&format!(
                    "getitem: slot {slot_idx} is not a tuple"
                )));
            }
        }
        "chunk_narrow" => {
            // chunk decomposition: getitem(chunk(x, N), i) -> narrow(x, dim, start, length)
            // Compute start and length at runtime from the tensor's actual shape.
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let dim = kw_isize(node, "dim", 0);
            let chunk_index = kw_usize(node, "chunk_index", 0);
            let num_chunks = kw_usize(node, "num_chunks", 1);
            let ndim = a.shape.len() as isize;
            let normalized_dim = if dim < 0 { dim + ndim } else { dim };
            let dim_size = a.shape[normalized_dim as usize] as usize;
            let chunk_size = dim_size / num_chunks;
            let start = chunk_index * chunk_size;
            let length = if chunk_index == num_chunks - 1 {
                dim_size - start // last chunk gets remainder
            } else {
                chunk_size
            };
            slots.push(Slot::Owned(shape_ops::narrow(&a, dim, start, length)?));
        }
        // contiguous() is a no-op for already-contiguous inputs (the interpreter
        // ensures inputs are contiguous before passing to the Rust engine).
        "contiguous" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            if a.strides == contiguous_strides(&a.shape) {
                slots.push(Slot::View {
                    data: a.data,
                    shape: Arc::from(a.shape.as_slice()),
                    strides: Arc::from(a.strides.as_slice()),
                    dtype: a.dtype,
                });
            } else {
                slots.push(Slot::Owned(shape_ops::to_contiguous(&a)?));
            }
        }
        "squeeze" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let dim = kw_isize(node, "dim", 0);
            let (new_shape, new_strides) = shape_ops::squeeze_view(&a, dim)?;
            slots.push(Slot::View {
                data: a.data,
                shape: Arc::from(new_shape),
                strides: Arc::from(new_strides),
                dtype: a.dtype,
            });
        }
        "unsqueeze" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let dim = kw_isize(node, "dim", 0);
            let (new_shape, new_strides) = shape_ops::unsqueeze_view(&a, dim)?;
            slots.push(Slot::View {
                data: a.data,
                shape: Arc::from(new_shape),
                strides: Arc::from(new_strides),
                dtype: a.dtype,
            });
        }
        "unflatten" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let dim = kw_isize(node, "dim", 0);
            let sizes = kw_i64_vec(node, "sizes");
            slots.push(Slot::Owned(shape_ops::unflatten(&a, dim, &sizes)?));
        }
        // dropout is a no-op during inference (training=false)
        "dropout" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::View {
                data: a.data,
                shape: Arc::from(a.shape.as_slice()),
                strides: Arc::from(a.strides.as_slice()),
                dtype: a.dtype,
            });
        }

        // Phase 3: flatten
        "flatten" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let start_dim = kw_isize(node, "start_dim", 0);
            let end_dim = kw_isize(node, "end_dim", -1);
            slots.push(Slot::Owned(shape_ops::flatten(&a, start_dim, end_dim)?));
        }
        _ => return Ok(false),
    }
    Ok(true)
}
