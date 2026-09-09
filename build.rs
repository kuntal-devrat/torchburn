//! Build script: links MSVC-built static OpenBLAS when `openblas` feature is enabled.
//!
//! The static library is built from source via CMake with MSVC (NOFORTRAN=1,
//! NO_LAPACK=1, BUILD_SHARED_LIBS=OFF).  This avoids DLL call overhead and
//! links the optimized Skylake kernels directly into the .pyd.

fn main() {
    // Target probes for peak-per-platform codegen (Windows/Linux/macOS/iOS).
    println!("cargo:rustc-check-cfg=cfg(has_avx512)");
    println!("cargo:rustc-check-cfg=cfg(has_metal)");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=RUSTFLAGS");
    println!("cargo:rerun-if-env-changed=CARGO_ENCODED_RUSTFLAGS");
    println!("cargo:rerun-if-env-changed=OPENBLAS_LIB_DIR");
    let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    // Emit has_avx512 ONLY for native/v4 builds or when host has AVX-512F.
    // Emitting unconditionally would bake AVX-512 into portable wheels (#UD).
    if arch == "x86_64" {
        let rustflags = std::env::var("RUSTFLAGS").unwrap_or_default()
            + &std::env::var("CARGO_ENCODED_RUSTFLAGS").unwrap_or_default();
        let explicit = rustflags.contains("avx512")
            || rustflags.contains("target-cpu=native")
            || rustflags.contains("target-cpu=x86-64-v4");
        #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
        let host_has = {
            let is_native = std::env::var("HOST")
                .map(|h| h.contains("x86_64"))
                .unwrap_or(false);
            is_native && std::arch::is_x86_feature_detected!("avx512f")
        };
        #[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
        let host_has = false;
        if explicit || host_has {
            println!("cargo:rustc-cfg=has_avx512");
        }
    }
    if os == "macos" || os == "ios" {
        println!("cargo:rustc-cfg=has_metal");
        // NOTE: no `rustc-link-arg` frameworks here. The `metal` crate links
        // its own frameworks, and `-framework=X` (single-arg `=` form) is
        // rejected by clang ("unknown argument"), breaking every macOS link.
    }
    // Vulkan / DX12 / WGPU are runtime-selected via TORCHBURN_WGPU_BACKEND;
    // no link args needed (wgpu handles loader discovery per-OS).

    // NOTE: #[cfg(feature)] never works in build scripts (build-script features
    // != crate features). Detect via CARGO_FEATURE_<NAME> env instead.
    if std::env::var("CARGO_FEATURE_OPENBLAS").is_ok() {
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

        // OPENBLAS_LIB_DIR override for custom installs
        if let Ok(dir) = std::env::var("OPENBLAS_LIB_DIR") {
            println!("cargo:rustc-link-search=native={dir}");
            println!("cargo:rustc-link-lib=openblas");
        } else if msvc_lib.exists() {
            println!("cargo:warning=Using MSVC-built static OpenBLAS (no DLL overhead)");
            let lib_dir = msvc_lib.parent().unwrap();
            println!("cargo:rustc-link-search=native={}", lib_dir.display());
            println!("cargo:rustc-link-lib=openblas");
            // MSVC static OpenBLAS sys deps — Windows only
            if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
                println!("cargo:rustc-link-lib=advapi32");
                println!("cargo:rustc-link-lib=bcrypt");
                println!("cargo:rustc-link-lib=userenv");
                println!("cargo:rustc-link-lib=ws2_32");
            }
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
