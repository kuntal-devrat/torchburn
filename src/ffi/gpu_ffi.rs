//! GPU information FFI wrappers.

use pyo3::prelude::*;

/// Returns GPU adapter information.
///
/// Returns a dict with keys: available, adapter_name, backend, vram_bytes.
#[pyfunction]
pub fn gpu_info(py: Python<'_>) -> PyResult<pyo3::PyObject> {
    #[cfg(feature = "burn-wgpu")]
    {
        let (available, name, backend, vram) = crate::wgpu::backend::gpu_info();
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("available", available)?;
        dict.set_item("adapter_name", &name)?;
        dict.set_item("backend", &backend)?;
        dict.set_item("vram_bytes", vram)?;
        dict.set_item(
            "device_override",
            crate::wgpu::backend::device_override().unwrap_or_default(),
        )?;
        Ok(dict.into())
    }
    #[cfg(not(feature = "burn-wgpu"))]
    {
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("available", false)?;
        dict.set_item("adapter_name", "burn-wgpu feature not compiled")?;
        dict.set_item("backend", "none")?;
        dict.set_item("vram_bytes", 0u64)?;
        dict.set_item(
            "device_override",
            std::env::var("TORCHBURN_DEVICE").unwrap_or_default(),
        )?;
        Ok(dict.into())
    }
}

/// Returns the name of the active GPU backend (e.g. "Metal", "Vulkan", "none").
#[pyfunction]
pub fn gpu_backend() -> String {
    #[cfg(feature = "burn-wgpu")]
    {
        if crate::wgpu::backend::gpu_available() {
            crate::wgpu::backend::gpu_info().2
        } else {
            "none".to_string()
        }
    }
    #[cfg(not(feature = "burn-wgpu"))]
    {
        "none".to_string()
    }
}

/// Check if a GPU adapter is available.
#[pyfunction]
pub fn gpu_available() -> bool {
    #[cfg(feature = "burn-wgpu")]
    {
        crate::wgpu::backend::gpu_available()
    }
    #[cfg(not(feature = "burn-wgpu"))]
    {
        false
    }
}
