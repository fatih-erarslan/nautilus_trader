//! AVX optimizations for x86_64

#[cfg(target_arch = "x86_64")]
pub fn avx_enabled() -> bool {
    is_x86_feature_detected!("avx")
}

#[cfg(not(target_arch = "x86_64"))]
pub fn avx_enabled() -> bool {
    false
}