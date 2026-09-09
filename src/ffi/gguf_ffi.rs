//! GGUF FFI — Python-accessible functions for GGUF model inspection.

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::gguf::GgufMmap;

/// Get information about a GGUF file (architecture, name, version, tensor count).
#[pyfunction]
pub fn gguf_info(py: Python<'_>, path: String) -> PyResult<PyObject> {
    // Release GIL during file IO + parse
    let mmap = py
        .allow_threads(|| GgufMmap::open(std::path::Path::new(&path)))
        .map_err(|e| {
            let s = e.to_string();
            if s.contains("magic") || s.contains("version") || s.contains("quant") {
                pyo3::exceptions::PyValueError::new_err(s)
            } else {
                pyo3::exceptions::PyIOError::new_err(s)
            }
        })?;
    let model = mmap.model();

    {

        let dict = PyDict::new(py);
        dict.set_item("version", model.version)?;
        dict.set_item("tensor_count", model.tensors.len())?;
        dict.set_item("data_offset", model.data_offset)?;

        if let Some(arch) = model.architecture() {
            dict.set_item("architecture", arch)?;
        }
        if let Some(name) = model.name() {
            dict.set_item("name", name)?;
        }

        let mut quant_summary = std::collections::HashMap::new();
        for t in &model.tensors {
            *quant_summary
                .entry(format!("{:?}", t.quant_type))
                .or_insert(0usize) += 1;
        }
        dict.set_item("quant_types", quant_summary)?;

        Ok(dict.into())
    }
}

/// List all tensors in a GGUF file with their shapes and quant types.
#[pyfunction]
pub fn gguf_tensors(py: Python<'_>, path: String) -> PyResult<Vec<PyObject>> {
    let mmap = py
        .allow_threads(|| GgufMmap::open(std::path::Path::new(&path)))
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    let model = mmap.model();

    {

        let mut result = Vec::with_capacity(model.tensors.len());
        for t in &model.tensors {
            let dict = PyDict::new(py);
            dict.set_item("name", &t.name)?;
            dict.set_item("dims", &t.dims)?;
            dict.set_item("n_elements", t.n_elements())?;
            dict.set_item("n_bytes", t.n_bytes())?;
            dict.set_item("quant_type", format!("{:?}", t.quant_type))?;
            dict.set_item("has_native_support", t.quant_type.has_native_support())?;
            if let Some(native) = t.quant_type.to_native_quant() {
                dict.set_item("native_quant", format!("{:?}", native))?;
            }
            result.push(dict.into());
        }
        Ok(result)
    }
}

/// Get metadata key-value pairs from a GGUF file.
#[pyfunction]
pub fn gguf_metadata(py: Python<'_>, path: String) -> PyResult<Vec<(String, PyObject)>> {
    let mmap = py
        .allow_threads(|| GgufMmap::open(std::path::Path::new(&path)))
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    let model = mmap.model();

    {

        let mut result = Vec::new();
        for (key, value) in &model.metadata {
            let py_val = gguf_value_to_pyobject(py, value)?;
            result.push((key.clone(), py_val));
        }
        Ok(result)
    }
}

fn gguf_value_to_pyobject(
    py: Python<'_>,
    value: &crate::gguf::GgufMetadataValue,
) -> PyResult<PyObject> {
    match value {
        crate::gguf::GgufMetadataValue::U8(v) => Ok(v.into_pyobject(py)?.into()),
        crate::gguf::GgufMetadataValue::I8(v) => Ok(v.into_pyobject(py)?.into()),
        crate::gguf::GgufMetadataValue::U16(v) => Ok(v.into_pyobject(py)?.into()),
        crate::gguf::GgufMetadataValue::I16(v) => Ok(v.into_pyobject(py)?.into()),
        crate::gguf::GgufMetadataValue::U32(v) => Ok(v.into_pyobject(py)?.into()),
        crate::gguf::GgufMetadataValue::I32(v) => Ok(v.into_pyobject(py)?.into()),
        crate::gguf::GgufMetadataValue::F32(v) => Ok(v.into_pyobject(py)?.into()),
        crate::gguf::GgufMetadataValue::Bool(v) => {
            Ok(pyo3::types::PyBool::new(py, *v).to_owned().into())
        }
        crate::gguf::GgufMetadataValue::String(v) => Ok(v.into_pyobject(py)?.into()),
        crate::gguf::GgufMetadataValue::U64(v) => Ok(v.into_pyobject(py)?.into()),
        crate::gguf::GgufMetadataValue::I64(v) => Ok(v.into_pyobject(py)?.into()),
        crate::gguf::GgufMetadataValue::F64(v) => Ok(v.into_pyobject(py)?.into()),
        crate::gguf::GgufMetadataValue::Array(arr) => {
            let mut py_arr: Vec<PyObject> = Vec::with_capacity(arr.len());
            for item in arr {
                py_arr.push(gguf_value_to_pyobject(py, item)?);
            }
            Ok(py_arr.into_pyobject(py)?.into())
        }
    }
}
