//! CPU feature detection utilities

/// Check if the CPU supports AVX-512 instructions
pub fn has_avx512() -> bool {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        is_x86_feature_detected!("avx512f")
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        false
    }
}

/// Check if the CPU supports AVX2 instructions
pub fn has_avx2() -> bool {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        is_x86_feature_detected!("avx2")
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        false
    }
}

/// Check if the CPU supports SSE4.1 instructions
pub fn has_sse41() -> bool {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        is_x86_feature_detected!("sse4.1")
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        false
    }
}

/// Check if the CPU supports FMA instructions
pub fn has_fma() -> bool {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        is_x86_feature_detected!("fma")
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        false
    }
}

/// Get a summary of all supported SIMD features
pub fn get_cpu_features() -> CpuFeatures {
    CpuFeatures {
        avx512: has_avx512(),
        avx2: has_avx2(),
        sse41: has_sse41(),
        fma: has_fma(),
    }
}

/// Struct containing CPU feature support information
#[derive(Debug, Clone, Copy)]
pub struct CpuFeatures {
    pub avx512: bool,
    pub avx2: bool,
    pub sse41: bool,
    pub fma: bool,
}

impl CpuFeatures {
    /// Get the best available SIMD instruction set
    pub fn best_available(&self) -> &'static str {
        if self.avx512 {
            "AVX-512"
        } else if self.avx2 {
            "AVX2"
        } else if self.sse41 {
            "SSE4.1"
        } else {
            "None"
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_feature_detection() {
        // These tests will pass/fail based on the actual CPU
        let features = get_cpu_features();
        println!("CPU Features: {:?}", features);
        println!("Best available: {}", features.best_available());
        
        // At minimum, we should have consistent results
        assert_eq!(has_avx512(), features.avx512);
        assert_eq!(has_avx2(), features.avx2);
        assert_eq!(has_sse41(), features.sse41);
        assert_eq!(has_fma(), features.fma);
    }
}