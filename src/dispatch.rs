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
    /// Baseline ISA (SSE2 on x86-64, NEON on ARM) or unknown.
    Scalar = 0,
    /// AVX2 + FMA.
    Avx2 = 1,
    /// AVX-512F + AVX-512BW (no VNNI).
    Avx512 = 2,
    /// AVX-512F + AVX-512BW + AVX-512 VNNI (vpdpbusd).
    Avx512Vnni = 3,
}

impl CpuTier {
    pub const fn name(self) -> &'static str {
        match self {
            CpuTier::Scalar => "scalar",
            CpuTier::Avx2 => "avx2",
            CpuTier::Avx512 => "avx512",
            CpuTier::Avx512Vnni => "avx512_vnni",
        }
    }
}

/// Feature flags resolved once. Extra fields (neon, neon_fp16) exist so ARM
/// builds have a real detection path instead of silently reporting scalar.
#[derive(Debug, Clone, Copy, Default)]
pub struct CpuFeatures {
    pub avx2: bool,
    pub fma: bool,
    pub avx512f: bool,
    pub avx512bw: bool,
    pub avx512vnni: bool,
    pub neon: bool,
    pub neon_fp16: bool,
}

impl CpuFeatures {
    pub const fn tier(&self) -> CpuTier {
        if self.avx512vnni && self.avx512f && self.avx512bw {
            CpuTier::Avx512Vnni
        } else if self.avx512f && self.avx512bw {
            CpuTier::Avx512
        } else if self.avx2 && self.fma {
            CpuTier::Avx2
        } else {
            CpuTier::Scalar
        }
    }

    pub const fn tier_name(&self) -> &'static str {
        self.tier().name()
    }
}

/// Per-tier instances used by the `dispatch-test` override (and by any code
/// that needs a concrete feature set for a specific tier).
pub(crate) const TIER_FEATURES: [CpuFeatures; 4] = [
    CpuFeatures {
        avx2: false,
        fma: false,
        avx512f: false,
        avx512bw: false,
        avx512vnni: false,
        neon: false,
        neon_fp16: false,
    },
    CpuFeatures {
        avx2: true,
        fma: true,
        avx512f: false,
        avx512bw: false,
        avx512vnni: false,
        neon: false,
        neon_fp16: false,
    },
    CpuFeatures {
        avx2: true,
        fma: true,
        avx512f: true,
        avx512bw: true,
        avx512vnni: false,
        neon: false,
        neon_fp16: false,
    },
    CpuFeatures {
        avx2: true,
        fma: true,
        avx512f: true,
        avx512bw: true,
        avx512vnni: true,
        neon: false,
        neon_fp16: false,
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
            neon: false,
            neon_fp16: false,
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
            neon: std::arch::is_aarch64_feature_detected!("neon"),
            neon_fp16: std::arch::is_aarch64_feature_detected!("fp16"),
        }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        CpuFeatures::default()
    }
}

static FEATURES: OnceLock<CpuFeatures> = OnceLock::new();

/// Test-only override: 0 = auto-detect, 1..=4 = CpuTier + 1.
#[cfg(feature = "dispatch-test")]
static OVERRIDE: AtomicU8 = AtomicU8::new(0);

/// Cached CPU feature set, resolved once per process.
///
/// With `dispatch-test` a forced tier takes precedence over detection so
/// tests can exercise every code path on a single machine.
pub fn cpu_features() -> &'static CpuFeatures {
    #[cfg(feature = "dispatch-test")]
    {
        let o = OVERRIDE.load(Ordering::Relaxed);
        if o >= 1 && o <= 4 {
            return &TIER_FEATURES[(o - 1) as usize];
        }
    }
    FEATURES.get_or_init(detect)
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
        assert!(CpuTier::Scalar < CpuTier::Avx2);
        assert!(CpuTier::Avx2 < CpuTier::Avx512);
        assert!(CpuTier::Avx512 < CpuTier::Avx512Vnni);
        assert_eq!(CpuTier::Avx512Vnni.name(), "avx512_vnni");
    }

    #[test]
    fn tier_table_has_correct_flags() {
        assert_eq!(TIER_FEATURES[0].tier(), CpuTier::Scalar);
        assert_eq!(TIER_FEATURES[1].tier(), CpuTier::Avx2);
        assert_eq!(TIER_FEATURES[2].tier(), CpuTier::Avx512);
        assert_eq!(TIER_FEATURES[3].tier(), CpuTier::Avx512Vnni);
    }

    #[test]
    fn detection_is_cached_and_consistent() {
        let a = cpu_features();
        let b = cpu_features();
        assert!(std::ptr::eq(a, b), "cpu_features must be resolved once");
        #[cfg(target_arch = "x86_64")]
        {
            // sanity: on x86-64 the reported tier must match the flags
            let f = cpu_features();
            let t = f.tier();
            match t {
                CpuTier::Scalar => assert!(!f.avx2 || !f.fma),
                CpuTier::Avx2 => assert!(f.avx2 && f.fma),
                CpuTier::Avx512 => assert!(f.avx512f && f.avx512bw),
                CpuTier::Avx512Vnni => assert!(f.avx512vnni),
            }
        }
    }

    #[cfg(feature = "dispatch-test")]
    #[test]
    fn forced_tiers_override_detection() {
        for tier in [
            CpuTier::Scalar,
            CpuTier::Avx2,
            CpuTier::Avx512,
            CpuTier::Avx512Vnni,
        ] {
            force_tier(tier);
            assert_eq!(cpu_features().tier(), tier);
            assert_eq!(cpu_features().tier_name(), tier.name());
        }
        clear_override();
        // After clearing, detection returns a *real* feature set again.
        let _ = cpu_features().tier();
    }
}
