//! Runtime CPU feature detection and implementation selection

use std::sync::OnceLock;

/// SIMD implementation selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SimdImplementation {
    /// Intel AVX-512: 512-bit vectors (requires nightly)
    #[cfg(all(feature = "avx512", feature = "nightly"))]
    Avx512,
    
    /// Intel AVX2: 256-bit vectors (requires stable Rust intrinsics)
    #[cfg(feature = "avx2")]
    Avx2,
    
    /// Portable SIMD using the `wide` crate
    Portable,
    
    /// Optimized scalar fallback
    Scalar,
}

/// CPU features detection result
#[derive(Debug, Clone, Copy)]
pub struct CpuFeatures {
    pub has_avx2: bool,
    pub has_avx512f: bool,
    pub has_fma: bool,
    pub num_cores: usize,
}

/// Global CPU features cache
static CPU_FEATURES: OnceLock<CpuFeatures> = OnceLock::new();
static BEST_IMPL: OnceLock<SimdImplementation> = OnceLock::new();

/// Detect CPU features at runtime
pub fn detect_cpu_features() -> CpuFeatures {
    *CPU_FEATURES.get_or_init(|| {
        let mut features = CpuFeatures {
            has_avx2: false,
            has_avx512f: false,
            has_fma: false,
            num_cores: std::thread::available_parallelism()
                .map(|p| p.get())
                .unwrap_or(1),
        };
        
        // CPU feature detection for x86_64
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                features.has_avx2 = true;
            }
            if is_x86_feature_detected!("avx512f") {
                features.has_avx512f = true;
            }
            if is_x86_feature_detected!("fma") {
                features.has_fma = true;
            }
        }
        
        features
    })
}

/// Get the best available SIMD implementation for current CPU
pub fn best_implementation() -> SimdImplementation {
    *BEST_IMPL.get_or_init(|| {
        let features = detect_cpu_features();
        
        // Choose best implementation based on available features
        #[cfg(all(feature = "avx512", feature = "nightly"))]
        {
            if features.has_avx512f {
                return SimdImplementation::Avx512;
            }
        }
        
        #[cfg(feature = "avx2")]
        {
            if features.has_avx2 {
                return SimdImplementation::Avx2;
            }
        }
        
        // Default to portable SIMD which works everywhere
        SimdImplementation::Portable
    })
}

/// Get implementation name as string
pub fn implementation_name(impl_type: SimdImplementation) -> &'static str {
    match impl_type {
        #[cfg(all(feature = "avx512", feature = "nightly"))]
        SimdImplementation::Avx512 => "AVX-512",
        
        #[cfg(feature = "avx2")]
        SimdImplementation::Avx2 => "AVX2",
        
        SimdImplementation::Portable => "Portable SIMD",
        SimdImplementation::Scalar => "Scalar",
    }
}

/// Print CPU capabilities
pub fn print_cpu_info() {
    let features = detect_cpu_features();
    let best = best_implementation();
    
    println!("CPU Information:");
    println!("  Cores: {}", features.num_cores);
    println!("  AVX2: {}", features.has_avx2);
    println!("  AVX-512F: {}", features.has_avx512f);
    println!("  FMA: {}", features.has_fma);
    println!("  Best implementation: {}", implementation_name(best));
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cpu_detection() {
        let features = detect_cpu_features();
        assert!(features.num_cores > 0);
        
        let best = best_implementation();
        println!("Best implementation: {:?}", best);
    }
}