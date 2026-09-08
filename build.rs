//! Build script: links MSVC-built static OpenBLAS when `openblas` feature is enabled.
//!
//! The static library is built from source via CMake with MSVC (NOFORTRAN=1,
//! NO_LAPACK=1, BUILD_SHARED_LIBS=OFF).  This avoids DLL call overhead and
//! links the optimized Skylake kernels directly into the .pyd.

fn main() {
    // Target probes for peak-per-platform codegen (Windows/Linux/macOS/iOS).
    // Emits cfg(has_avx512) / cfg(has_metal) and links the right system
    // frameworks without disturbing the portable baseline in .cargo/config.toml.
    println!("cargo:rustc-check-cfg=cfg(has_avx512)");
    println!("cargo:rustc-check-cfg=cfg(has_metal)");
    println!("cargo:rerun-if-changed=build.rs");
    let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if arch == "x86_64" {
        println!("cargo:rustc-cfg=has_avx512");
    }
    if os == "macos" || os == "ios" {
        println!("cargo:rustc-cfg=has_metal");
        if os == "macos" {
            println!("cargo:rustc-link-arg=-framework=Metal");
            println!("cargo:rustc-link-arg=-framework=Foundation");
        } else {
            println!("cargo:rustc-link-arg=-framework=Metal");
        }
    }
    // Vulkan / DX12 / WGPU are runtime-selected via TORCHBURN_WGPU_BACKEND;
    // no link args needed (wgpu handles loader discovery per-OS).

    #[cfg(feature = "openblas")]
    {
        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set");

        // Try MSVC-built static lib first (preferred — no DLL overhead).
        let msvc_lib = std::path::PathBuf::from(&manifest_dir)
            .join("vendor")
            .join("OpenBLAS_msvc")
            .join("build_msvc")
            .join("lib")
            .join("RELEASE")
            .join("openblas.lib");

        // Fallback: prebuilt DLL import lib (has DLL overhead).
        let prebuilt_lib = std::path::PathBuf::from(&manifest_dir)
            .join("vendor")
            .join("OpenBLAS_prebuilt")
            .join("lib")
            .join("openblas.lib");

        if msvc_lib.exists() {
            println!("cargo:warning=Using MSVC-built static OpenBLAS (no DLL overhead)");
            let lib_dir = msvc_lib.parent().unwrap();
            println!("cargo:rustc-link-search=native={}", lib_dir.display());
            println!("cargo:rustc-link-lib=openblas");
            // MSVC static OpenBLAS depends on these Windows system libraries.
            println!("cargo:rustc-link-lib=advapi32");
            println!("cargo:rustc-link-lib=bcrypt");
            println!("cargo:rustc-link-lib=userenv");
            println!("cargo:rustc-link-lib=ws2_32");
        } else if prebuilt_lib.exists() {
            println!("cargo:warning=Using prebuilt OpenBLAS DLL (has call overhead)");
            let lib_dir = prebuilt_lib.parent().unwrap();
            println!("cargo:rustc-link-search=native={}", lib_dir.display());
            println!("cargo:rustc-link-lib=openblas");
        } else {
            println!("cargo:warning=No vendor OpenBLAS found; attempting system openblas linkage");
            println!("cargo:rustc-link-lib=openblas");
        }

        println!("cargo:rerun-if-changed={}", msvc_lib.display());
        println!("cargo:rerun-if-changed={}", prebuilt_lib.display());
    }
}
