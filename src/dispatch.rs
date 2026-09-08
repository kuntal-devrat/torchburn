//! Runtime CPU feature dispatch (Phase 0.3).
//!
//! CPU capabilities are probed **once** and cached in a `static` (OnceLock),
//! then read by every kernel entry point. Before this module, hot kernels
//! called `is_x86_feature_detected!` on *every* invocation (e.g. inside
//! `gemv_w4a32_grouped`), which re-runs CPUID each time. Resolving once turns
//! the per-token dispatch cost into two cache-line loads.
//!
//! The portable wheel builds for a baseline ISA; this module is what lets it
//! still run AVX2 / AVX-512 VNNI kernels where the CPU supports them.
//!
//! Test hook: with the `dispatch-test` feature, `force_tier` overrides the
//! detected features so parity tests can assert identical outputs across every
//! feature tier on one machine.

#[cfg(feature = "dispatch-test")]
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::OnceLock;

/// Coarse execution tier, ordered from slowest to fastest.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum CpuTier {
    /// Baseline ISA (SSE2 on x86-64, baseline on ARM) or unknown.
    Scalar = 0,
    /// ARM NEON with FP16 support.
    Neon = 1,
    /// AVX2 + FMA.
    Avx2 = 2,
    /// AVX-512F + AVX-512BW (no VNNI).
    Avx512 = 3,
    /// AVX-512F + AVX-512BW + AVX-512 VNNI (vpdpbusd).
    Avx512Vnni = 4,
    /// AVX-512 BF16 (Sapphire Rapids+, Zen 4+).
    Avx512Bf16 = 5,
    /// AVX-512 FP16 (Sapphire Rapids+ with native half-precision).
    Avx512Fp16 = 6,
    /// Intel AMX (Advanced Matrix Extensions — tile matmul).
    Amx = 7,
}

impl CpuTier {
    pub const fn name(self) -> &'static str {
        match self {
            CpuTier::Scalar => "scalar",
            CpuTier::Neon => "neon",
            CpuTier::Avx2 => "avx2",
            CpuTier::Avx512 => "avx512",
            CpuTier::Avx512Vnni => "avx512_vnni",
            CpuTier::Avx512Bf16 => "avx512_bf16",
            CpuTier::Avx512Fp16 => "avx512_fp16",
            CpuTier::Amx => "amx",
        }
    }
}

/// Feature flags resolved once. Extra fields for ARM so builds have real
/// detection paths instead of silently reporting scalar.
#[derive(Debug, Clone, Copy, Default)]
pub struct CpuFeatures {
    pub avx2: bool,
    pub fma: bool,
    pub avx512f: bool,
    pub avx512bw: bool,
    pub avx512vnni: bool,
    pub avx512bf16: bool,
    pub avx512fp16: bool,
    pub amx_tile: bool,
    pub amx_int8: bool,
    pub amx_bf16: bool,
    pub neon: bool,
    pub neon_fp16: bool,
    /// ARM I8MM: int8 matrix multiply extensions (M1 Pro+, Cortex-A510+).
    pub neon_i8mm: bool,
    /// ARM dotprod: int8 dot product (Cortex-A75+, Apple M1+).
    pub neon_dotprod: bool,
    /// ARM SVE / SVE2: scalable vector extensions.
    pub neon_sve: bool,
    /// ARM SVE2: integer dot product and widening operations.
    pub neon_sve2: bool,
}

impl CpuFeatures {
    pub const fn tier(&self) -> CpuTier {
        if self.amx_tile && self.amx_int8 {
            CpuTier::Amx
        } else if self.avx512fp16 && self.avx512f {
            CpuTier::Avx512Fp16
        } else if self.avx512bf16 && self.avx512f {
            CpuTier::Avx512Bf16
        } else if self.avx512vnni && self.avx512f && self.avx512bw {
            CpuTier::Avx512Vnni
        } else if self.avx512f && self.avx512bw {
            CpuTier::Avx512
        } else if self.avx2 && self.fma {
            CpuTier::Avx2
        } else if self.neon {
            CpuTier::Neon
        } else {
            CpuTier::Scalar
        }
    }

    pub const fn tier_name(&self) -> &'static str {
        self.tier().name()
    }

    /// Check if this CPU supports any VNNI-class integer dot product
    /// acceleration (AVX-512 VNNI on x86, dotprod/I8MM on ARM).
    pub const fn has_int_dot(&self) -> bool {
        self.avx512vnni || self.neon_dotprod || self.neon_i8mm
    }
}

/// Per-tier instances used by the `dispatch-test` override.
/// Covers all 8 [`CpuTier`] variants so parity tests can force every tier.
pub(crate) const TIER_FEATURES: [CpuFeatures; 8] = [
    CpuFeatures {
        avx2: false,
        fma: false,
        avx512f: false,
        avx512bw: false,
        avx512vnni: false,
        avx512bf16: false,
        avx512fp16: false,
        amx_tile: false,
        amx_int8: false,
        amx_bf16: false,
        neon: false,
        neon_fp16: false,
        neon_i8mm: false,
        neon_dotprod: false,
        neon_sve: false,
        neon_sve2: false,
    },
    CpuFeatures {
        avx2: false,
        fma: false,
        avx512f: false,
        avx512bw: false,
        avx512vnni: false,
        avx512bf16: false,
        avx512fp16: false,
        amx_tile: false,
        amx_int8: false,
        amx_bf16: false,
        neon: true,
        neon_fp16: true,
        neon_i8mm: false,
        neon_dotprod: true,
        neon_sve: false,
        neon_sve2: false,
    },
    CpuFeatures {
        avx2: true,
        fma: true,
        avx512f: false,
        avx512bw: false,
        avx512vnni: false,
        avx512bf16: false,
        avx512fp16: false,
        amx_tile: false,
        amx_int8: false,
        amx_bf16: false,
        neon: false,
        neon_fp16: false,
        neon_i8mm: false,
        neon_dotprod: false,
        neon_sve: false,
        neon_sve2: false,
    },
    CpuFeatures {
        avx2: true,
        fma: true,
        avx512f: true,
        avx512bw: true,
        avx512vnni: false,
        avx512bf16: false,
        avx512fp16: false,
        amx_tile: false,
        amx_int8: false,
        amx_bf16: false,
        neon: false,
        neon_fp16: false,
        neon_i8mm: false,
        neon_dotprod: false,
        neon_sve: false,
        neon_sve2: false,
    },
    CpuFeatures {
        avx2: true,
        fma: true,
        avx512f: true,
        avx512bw: true,
        avx512vnni: true,
        avx512bf16: false,
        avx512fp16: false,
        amx_tile: false,
        amx_int8: false,
        amx_bf16: false,
        neon: false,
        neon_fp16: false,
        neon_i8mm: false,
        neon_dotprod: false,
        neon_sve: false,
        neon_sve2: false,
    },
    CpuFeatures {
        avx2: true,
        fma: true,
        avx512f: true,
        avx512bw: true,
        avx512vnni: true,
        avx512bf16: true,
        avx512fp16: false,
        amx_tile: false,
        amx_int8: false,
        amx_bf16: true,
        neon: false,
        neon_fp16: false,
        neon_i8mm: false,
        neon_dotprod: false,
        neon_sve: false,
        neon_sve2: false,
    },
    CpuFeatures {
        avx2: true,
        fma: true,
        avx512f: true,
        avx512bw: true,
        avx512vnni: true,
        avx512bf16: true,
        avx512fp16: true,
        amx_tile: false,
        amx_int8: false,
        amx_bf16: true,
        neon: false,
        neon_fp16: false,
        neon_i8mm: false,
        neon_dotprod: false,
        neon_sve: false,
        neon_sve2: false,
    },
    CpuFeatures {
        avx2: true,
        fma: true,
        avx512f: true,
        avx512bw: true,
        avx512vnni: true,
        avx512bf16: true,
        avx512fp16: true,
        amx_tile: true,
        amx_int8: true,
        amx_bf16: true,
        neon: false,
        neon_fp16: false,
        neon_i8mm: false,
        neon_dotprod: false,
        neon_sve: false,
        neon_sve2: false,
    },
];

fn detect() -> CpuFeatures {
    #[cfg(target_arch = "x86_64")]
    {
        CpuFeatures {
            avx2: std::arch::is_x86_feature_detected!("avx2"),
            fma: std::arch::is_x86_feature_detected!("fma"),
            avx512f: std::arch::is_x86_feature_detected!("avx512f"),
            avx512bw: std::arch::is_x86_feature_detected!("avx512bw"),
            avx512vnni: std::arch::is_x86_feature_detected!("avx512vnni"),
            avx512bf16: std::arch::is_x86_feature_detected!("avx512bf16"),
            // avx512fp16 detection — not all Rust toolchains support this yet
            avx512fp16: cfg!(target_feature = "avx512fp16"),
            // AMX detection requires nightly (unstable feature x86_amx_intrinsics).
            // Default to false on stable; users on nightly can override via cfg.
            amx_tile: false,
            amx_int8: false,
            amx_bf16: false,
            neon: false,
            neon_fp16: false,
            neon_i8mm: false,
            neon_dotprod: false,
            neon_sve: false,
            neon_sve2: false,
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        CpuFeatures {
            avx2: false,
            fma: false,
            avx512f: false,
            avx512bw: false,
            avx512vnni: false,
            avx512bf16: false,
            avx512fp16: false,
            amx_tile: false,
            amx_int8: false,
            amx_bf16: false,
            neon: std::arch::is_aarch64_feature_detected!("neon"),
            neon_fp16: std::arch::is_aarch64_feature_detected!("fp16"),
            neon_i8mm: std::arch::is_aarch64_feature_detected!("i8mm"),
            neon_dotprod: std::arch::is_aarch64_feature_detected!("dotprod"),
            neon_sve: std::arch::is_aarch64_feature_detected!("sve"),
            neon_sve2: std::arch::is_aarch64_feature_detected!("sve2"),
        }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        CpuFeatures::default()
    }
}

static FEATURES: OnceLock<CpuFeatures> = OnceLock::new();

/// Test-only override: 0 = auto-detect, 1..=5 = CpuTier + 1.
#[cfg(feature = "dispatch-test")]
static OVERRIDE: AtomicU8 = AtomicU8::new(0);

/// Cached CPU feature set, resolved once per process.
///
/// Hot kernels MUST call this instead of `is_x86_feature_detected!` — CPUID
/// inside a per-row/per-token loop costs ~100-300ns per layer. This is two
/// cache-line loads after the one-time [`OnceLock`] init.
#[inline(always)]
pub fn cpu_features() -> &'static CpuFeatures {
    #[cfg(feature = "dispatch-test")]
    {
        let o = OVERRIDE.load(Ordering::Relaxed);
        if o >= 1 && (o as usize) <= TIER_FEATURES.len() {
            return &TIER_FEATURES[(o - 1) as usize];
        }
    }
    FEATURES.get_or_init(detect)
}

/// Fast tier check for hot loops: hoist `cpu_features()` out of the loop,
/// then call `tier_at_least` / field reads on the cached reference.
#[inline(always)]
pub fn tier_at_least(tier: CpuTier) -> bool {
    cpu_features().tier() >= tier
}

/// Force a specific feature tier (tests only, `dispatch-test` feature).
#[cfg(feature = "dispatch-test")]
pub fn force_tier(tier: CpuTier) {
    OVERRIDE.store((tier as u8) + 1, Ordering::Relaxed);
}

/// Clear a forced tier and return to auto-detection (tests only).
#[cfg(feature = "dispatch-test")]
pub fn clear_override() {
    OVERRIDE.store(0, Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiers_are_ordered() {
        assert!(CpuTier::Scalar < CpuTier::Neon);
        assert!(CpuTier::Neon < CpuTier::Avx2);
        assert!(CpuTier::Avx2 < CpuTier::Avx512);
        assert!(CpuTier::Avx512 < CpuTier::Avx512Vnni);
        assert_eq!(CpuTier::Avx512Vnni.name(), "avx512_vnni");
        assert_eq!(CpuTier::Neon.name(), "neon");
    }

    #[test]
    fn tier_table_has_correct_flags() {
        assert_eq!(TIER_FEATURES[0].tier(), CpuTier::Scalar);
        assert_eq!(TIER_FEATURES[1].tier(), CpuTier::Neon);
        assert_eq!(TIER_FEATURES[2].tier(), CpuTier::Avx2);
        assert_eq!(TIER_FEATURES[3].tier(), CpuTier::Avx512);
        assert_eq!(TIER_FEATURES[4].tier(), CpuTier::Avx512Vnni);
        assert_eq!(TIER_FEATURES[5].tier(), CpuTier::Avx512Bf16);
        assert_eq!(TIER_FEATURES[6].tier(), CpuTier::Avx512Fp16);
        assert_eq!(TIER_FEATURES[7].tier(), CpuTier::Amx);
    }

    #[test]
    fn detection_is_cached_and_consistent() {
        let a = cpu_features();
        let b = cpu_features();
        assert!(std::ptr::eq(a, b), "cpu_features must be resolved once");
        #[cfg(target_arch = "x86_64")]
        {
            let f = cpu_features();
            let t = f.tier();
            match t {
                CpuTier::Scalar => assert!(!f.avx2 || !f.fma),
                CpuTier::Avx2 => assert!(f.avx2 && f.fma),
                CpuTier::Avx512 => assert!(f.avx512f && f.avx512bw),
                CpuTier::Avx512Vnni => assert!(f.avx512vnni),
                _ => {}
            }
        }
    }

    #[cfg(feature = "dispatch-test")]
    #[test]
    fn forced_tiers_override_detection() {
        for tier in [
            CpuTier::Scalar,
            CpuTier::Neon,
            CpuTier::Avx2,
            CpuTier::Avx512,
            CpuTier::Avx512Vnni,
            CpuTier::Avx512Bf16,
            CpuTier::Avx512Fp16,
            CpuTier::Amx,
        ] {
            force_tier(tier);
            assert_eq!(cpu_features().tier(), tier);
            assert_eq!(cpu_features().tier_name(), tier.name());
        }
        clear_override();
        let _ = cpu_features().tier();
    }
}
