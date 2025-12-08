//! Basic SIMD operations

use crate::types::Float;

/// Check if SIMD acceleration is available
pub fn simd_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx2")
    }
    #[cfg(target_arch = "aarch64")]
    {
        std::is_aarch64_feature_detected!("neon")
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        false
    }
}

/// SIMD-accelerated sum
pub fn simd_sum(data: &[Float]) -> Float {
    // Fallback to scalar implementation
    data.iter().sum()
}

/// SIMD-accelerated mean
pub fn simd_mean(data: &[Float]) -> Float {
    if data.is_empty() {
        return 0.0;
    }
    simd_sum(data) / data.len() as Float
}