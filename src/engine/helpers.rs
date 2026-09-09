//! Node-argument helpers for the execution engine.
//!
//! Small accessors over Node kwargs and zero-copy slot views.
//! Extracted from engine.rs; re-exported at the engine root.

use crate::dlpack::{unsupported, BorrowedTensor, CapsuleRef};
use crate::engine::payload::{Node, Slot};
use pyo3::prelude::*;
use smallvec::SmallVec;
/// View a slot as a borrowed tensor (zero-copy for inputs).
pub(crate) fn slot_view<'p>(
    slots: &[Slot],
    capsules: &'p [CapsuleRef],
    index: usize,
) -> PyResult<BorrowedTensor> {
    match slots.get(index) {
        Some(Slot::Input(i)) => unsafe { BorrowedTensor::from_managed(capsules[*i].0) },
        Some(Slot::Owned(t)) => Ok(BorrowedTensor::from_owned(t)),
        Some(Slot::View {
            data,
            shape,
            strides,
            dtype,
        }) => Ok(BorrowedTensor {
            data: *data,
            shape: shape.to_vec(),
            strides: strides.to_vec(),
            dtype: *dtype,
        }),
        Some(Slot::Tuple(_)) => Err(unsupported(&format!(
            "slot {index} is a tuple; index it with getitem"
        ))),
        None => Err(unsupported(&format!(
            "argument references missing slot {index}"
        ))),
    }
}

/// Resolve a node argument to a slot index.
pub(crate) fn arg_index(node: &Node, position: usize) -> PyResult<usize> {
    let arg = node.args.get(position).ok_or_else(|| {
        unsupported(&format!(
            "node '{}' missing argument #{position}",
            node.target
        ))
    })?;
    arg.index
        .ok_or_else(|| unsupported(&format!("node '{}' has an unindexed argument", node.target)))
}

/// Get a scalar f64 from kwargs or a default.
pub(crate) fn kw_f64(node: &Node, key: &str, default: f64) -> f64 {
    node.kwargs
        .get(key)
        .and_then(|v| v.as_f64())
        .unwrap_or(default)
}

/// Read a scalar from kwargs, decoding the "inf"/"-inf"/"nan" string
/// tokens that the parser emits for non-finite constants (serde_json rejects
/// raw Infinity/NaN literals).
pub(crate) fn kw_f64_allow_inf(node: &Node, key: &str, default: f64) -> f64 {
    match node.kwargs.get(key) {
        Some(v) => {
            if let Some(x) = v.as_f64() {
                x
            } else if let Some(s) = v.as_str() {
                match s {
                    "inf" => f64::INFINITY,
                    "-inf" => f64::NEG_INFINITY,
                    "nan" => f64::NAN,
                    _ => default,
                }
            } else {
                default
            }
        }
        None => default,
    }
}

pub(crate) fn kw_bool(node: &Node, key: &str, default: bool) -> bool {
    node.kwargs
        .get(key)
        .and_then(|v| v.as_bool())
        .unwrap_or(default)
}

pub(crate) fn kw_isize(node: &Node, key: &str, default: isize) -> isize {
    node.kwargs
        .get(key)
        .and_then(|v| {
            if let Some(n) = v.as_i64() {
                Some(n)
            } else if let Some(arr) = v.as_array() {
                arr.first().and_then(|x| x.as_i64())
            } else {
                None
            }
        })
        .map(|v| v as isize)
        .unwrap_or(default)
}

/// Read an optional reduction dim from kwargs.  Returns None when absent.
/// Supports single dim (scalar) or multi-dim (list of ints).
/// For multi-dim, returns the list via kw_isize_vec("dim").
pub(crate) fn kw_opt_dim(node: &Node) -> PyResult<Option<isize>> {
    match node.kwargs.get("dim") {
        None => Ok(None),
        Some(v) => match v.as_i64() {
            Some(x) => Ok(Some(x as isize)),
            None => match v.as_array() {
                Some(arr) if arr.len() == 1 => Ok(arr[0].as_i64().map(|x| x as isize)),
                Some(arr) => {
                    // Multi-dim: for now, if it's [dim], treat as scalar;
                    // otherwise signal the caller to do iterative reduction.
                    let dims: Vec<isize> = arr
                        .iter()
                        .filter_map(|v| v.as_i64().map(|x| x as isize))
                        .collect();
                    if dims.is_empty() {
                        return Ok(None);
                    }
                    if dims.len() > 1 {
                        return Err(unsupported(
                            "multi-dim reductions (dim as a list) are not supported by the native engine",
                        ));
                    }
                    Ok(Some(dims[0]))
                }
                None => Ok(None),
            },
        },
    }
}

/// Read an optional reduction dim LIST from kwargs: scalar int, [int], or
/// [int, ...].  Returns an empty vec when absent.  Multi-dim lists are handled
/// natively by sum_dims/mean_dims (iterative single-dim reduction).
pub(crate) fn kw_opt_dims(node: &Node) -> SmallVec<[isize; 4]> {
    match node.kwargs.get("dim") {
        None => SmallVec::new(),
        Some(v) => {
            if let Some(x) = v.as_i64() {
                let mut sv = SmallVec::new();
                sv.push(x as isize);
                sv
            } else if let Some(arr) = v.as_array() {
                arr.iter()
                    .filter_map(|v| v.as_i64().map(|x| x as isize))
                    .collect()
            } else {
                SmallVec::new()
            }
        }
    }
}

pub(crate) fn kw_i64_vec(node: &Node, key: &str) -> Vec<i64> {
    node.kwargs
        .get(key)
        .and_then(|v| {
            v.as_array()
                .map(|arr| arr.iter().filter_map(|v| v.as_i64()).collect())
        })
        .unwrap_or_default()
}

pub(crate) fn kw_isize_vec(node: &Node, key: &str) -> Vec<isize> {
    node.kwargs
        .get(key)
        .and_then(|v| {
            v.as_array().map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_i64().map(|x| x as isize))
                    .collect()
            })
        })
        .unwrap_or_default()
}

pub(crate) fn kw_usize(node: &Node, key: &str, default: usize) -> usize {
    node.kwargs
        .get(key)
        .and_then(|v| {
            if let Some(n) = v.as_i64() {
                if n < 0 {
                    None
                } else {
                    Some(n as usize)
                }
            } else if let Some(arr) = v.as_array() {
                arr.first().and_then(|x| x.as_i64()).and_then(|n| {
                    if n < 0 {
                        None
                    } else {
                        Some(n as usize)
                    }
                })
            } else if let Some(n) = v.as_u64() {
                if n <= (isize::MAX as u64) {
                    Some(n as usize)
                } else {
                    None
                }
            } else {
                None
            }
        })
        .unwrap_or(default)
}

pub(crate) fn kw_i64(node: &Node, key: &str, default: i64) -> i64 {
    node.kwargs
        .get(key)
        .and_then(|v| {
            if let Some(n) = v.as_i64() {
                Some(n)
            } else if let Some(arr) = v.as_array() {
                arr.first().and_then(|x| x.as_i64())
            } else {
                None
            }
        })
        .unwrap_or(default)
}

pub(crate) fn kw_str<'a>(node: &'a Node, key: &str, default: &'a str) -> &'a str {
    node.kwargs
        .get(key)
        .and_then(|v| v.as_str())
        .unwrap_or(default)
}
