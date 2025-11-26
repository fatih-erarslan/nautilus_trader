//! CPU Feature Detection
//!
//! Runtime detection of SIMD capabilities across different architectures.

use std::sync::OnceLock;

/// CPU SIMD feature flags
#[derive(Debug, Clone, Copy)]
pub struct CpuFeatures {
    /// SSE2 support (x86_64 baseline)
    pub has_sse2: bool,
    /// AVX support
    pub has_avx: bool,
    /// AVX2 support
    pub has_avx2: bool,
    /// AVX-512 support
    pub has_avx512f: bool,
    /// FMA (Fused Multiply-Add) support
    pub has_fma: bool,
    /// ARM NEON support
    pub has_neon: bool,
}

impl CpuFeatures {
    /// Get the recommended SIMD lane size based on available features
    pub fn recommended_f32_lanes(&self) -> usize {
        if self.has_avx512f {
            16  // 512-bit SIMD
        } else if self.has_avx2 || self.has_avx {
            8   // 256-bit SIMD
        } else if self.has_neon {
            4   // 128-bit SIMD (ARM NEON)
        } else {
            1   // Scalar fallback
        }
    }

    /// Get the recommended SIMD lane size for f64
    pub fn recommended_f64_lanes(&self) -> usize {
        if self.has_avx512f {
            8   // 512-bit SIMD
        } else if self.has_avx2 || self.has_avx {
            4   // 256-bit SIMD
        } else if self.has_neon {
            2   // 128-bit SIMD (ARM NEON)
        } else {
            1   // Scalar fallback
        }
    }

    /// Check if we should use SIMD
    pub fn should_use_simd(&self) -> bool {
        self.has_avx2 || self.has_neon
    }
}

static CPU_FEATURES: OnceLock<CpuFeatures> = OnceLock::new();

/// Detect CPU features at runtime
pub fn detect_cpu_features() -> CpuFeatures {
    *CPU_FEATURES.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        {
            CpuFeatures {
                has_sse2: is_x86_feature_detected!("sse2"),
                has_avx: is_x86_feature_detected!("avx"),
                has_avx2: is_x86_feature_detected!("avx2"),
                has_avx512f: is_x86_feature_detected!("avx512f"),
                has_fma: is_x86_feature_detected!("fma"),
                has_neon: false,
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            CpuFeatures {
                has_sse2: false,
                has_avx: false,
                has_avx2: false,
                has_avx512f: false,
                has_fma: false,
                has_neon: std::arch::is_aarch64_feature_detected!("neon"),
            }
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            // Fallback for unsupported architectures
            CpuFeatures {
                has_sse2: false,
                has_avx: false,
                has_avx2: false,
                has_avx512f: false,
                has_fma: false,
                has_neon: false,
            }
        }
    })
}

/// Get a human-readable description of CPU features
pub fn feature_description() -> String {
    let features = detect_cpu_features();
    let mut desc = Vec::new();

    if features.has_avx512f {
        desc.push("AVX-512");
    }
    if features.has_avx2 {
        desc.push("AVX2");
    }
    if features.has_avx {
        desc.push("AVX");
    }
    if features.has_fma {
        desc.push("FMA");
    }
    if features.has_neon {
        desc.push("NEON");
    }
    if features.has_sse2 {
        desc.push("SSE2");
    }

    if desc.is_empty() {
        "Scalar (no SIMD)".to_string()
    } else {
        desc.join(", ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_detection() {
        let features = detect_cpu_features();
        println!("Detected features: {:?}", features);
        println!("Description: {}", feature_description());

        // Verify basic sanity
        #[cfg(target_arch = "x86_64")]
        {
            // x86_64 always has SSE2
            assert!(features.has_sse2);
        }

        #[cfg(target_arch = "aarch64")]
        {
            // aarch64 typically has NEON
            // Note: May not be true in all environments
        }
    }

    #[test]
    fn test_recommended_lanes() {
        let features = detect_cpu_features();
        let f32_lanes = features.recommended_f32_lanes();
        let f64_lanes = features.recommended_f64_lanes();

        println!("Recommended f32 lanes: {}", f32_lanes);
        println!("Recommended f64 lanes: {}", f64_lanes);

        // Lanes should be power of 2
        assert!(f32_lanes.is_power_of_two());
        assert!(f64_lanes.is_power_of_two());

        // f32 lanes should be >= f64 lanes (same bit width)
        assert!(f32_lanes >= f64_lanes);
    }
}
