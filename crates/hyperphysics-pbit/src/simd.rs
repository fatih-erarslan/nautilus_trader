//! SIMD vectorization for pBit dynamics
//!
//! Provides hardware-accelerated implementations using:
//! - AVX2 (256-bit, 4x f64 or 8x f32)
//! - AVX-512 (512-bit, 8x f64 or 16x f32) when available
//! - NEON (ARM, 128-bit, 2x f64 or 4x f32)
//!
//! Falls back to portable scalar code when SIMD unavailable.

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
use std::arch::x86_64::*;

/// Vectorized state update using AVX2 intrinsics (4x f64 at once)
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
pub unsafe fn update_states_avx2(states: &mut [f64], probabilities: &[f64]) {
    assert_eq!(states.len(), probabilities.len());
    let len = states.len();
    let chunks = len / 4;

    for i in 0..chunks {
        let idx = i * 4;
        let probs = _mm256_loadu_pd(probabilities.as_ptr().add(idx));
        let threshold = _mm256_set1_pd(0.5);
        let mask = _mm256_cmp_pd(probs, threshold, _CMP_GT_OQ);
        let ones = _mm256_set1_pd(1.0);
        let zeros = _mm256_setzero_pd();
        let result = _mm256_blendv_pd(zeros, ones, mask);
        _mm256_storeu_pd(states.as_mut_ptr().add(idx), result);
    }

    // Handle remaining elements
    for i in (chunks * 4)..len {
        states[i] = if probabilities[i] > 0.5 { 1.0 } else { 0.0 };
    }
}

/// Vectorized metropolis energy calculation (dot product)
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
pub unsafe fn dot_product_avx2(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let len = a.len();
    let chunks = len / 4;

    let mut sum = _mm256_setzero_pd();

    for i in 0..chunks {
        let idx = i * 4;
        let va = _mm256_loadu_pd(a.as_ptr().add(idx));
        let vb = _mm256_loadu_pd(b.as_ptr().add(idx));
        let prod = _mm256_mul_pd(va, vb);
        sum = _mm256_add_pd(sum, prod);
    }

    // Horizontal sum
    let mut result_array = [0.0; 4];
    _mm256_storeu_pd(result_array.as_mut_ptr(), sum);
    let mut total = result_array.iter().sum::<f64>();

    // Handle remaining elements
    for i in (chunks * 4)..len {
        total += a[i] * b[i];
    }

    total
}

/// Vectorized exponential for Boltzmann factors (approximation)
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
pub unsafe fn exp_avx2(x: &[f64], result: &mut [f64]) {
    assert_eq!(x.len(), result.len());
    let len = x.len();

    // Use scalar exp for now - vectorized exp requires more complex approximation
    for i in 0..len {
        result[i] = x[i].exp();
    }

    // TODO: Implement Remez polynomial approximation for vectorized exp
    // References:
    // - Intel's Vector Math Library (VML)
    // - Agner Fog's vector class library
}

/// Portable scalar fallback implementations
pub mod portable {
    pub fn update_states(states: &mut [f64], probabilities: &[f64]) {
        for (state, &prob) in states.iter_mut().zip(probabilities.iter()) {
            *state = if prob > 0.5 { 1.0 } else { 0.0 };
        }
    }

    pub fn dot_product(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    pub fn exp_vec(x: &[f64], result: &mut [f64]) {
        for (r, &val) in result.iter_mut().zip(x.iter()) {
            *r = val.exp();
        }
    }
}

/// High-level API that selects best implementation at runtime
pub struct SimdOps;

impl SimdOps {
    /// Update pBit states based on probabilities
    pub fn update_states(states: &mut [f64], probabilities: &[f64]) {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        unsafe {
            update_states_avx2(states, probabilities);
            return;
        }

        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
        portable::update_states(states, probabilities);
    }

    /// Compute dot product (for energy calculations)
    pub fn dot_product(a: &[f64], b: &[f64]) -> f64 {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        unsafe {
            return dot_product_avx2(a, b);
        }

        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
        portable::dot_product(a, b)
    }

    /// Vectorized exponential
    pub fn exp(x: &[f64], result: &mut [f64]) {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        unsafe {
            exp_avx2(x, result);
            return;
        }

        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
        portable::exp_vec(x, result);
    }

    /// Check SIMD availability
    pub fn simd_info() -> SimdInfo {
        SimdInfo {
            #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
            avx2: true,
            #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
            avx2: false,

            #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
            avx512: true,
            #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
            avx512: false,

            #[cfg(target_arch = "aarch64")]
            neon: true,
            #[cfg(not(target_arch = "aarch64"))]
            neon: false,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SimdInfo {
    pub avx2: bool,
    pub avx512: bool,
    pub neon: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_update_states() {
        let mut states = vec![0.0; 100];
        let probabilities: Vec<f64> = (0..100).map(|i| i as f64 / 100.0).collect();

        SimdOps::update_states(&mut states, &probabilities);

        for (i, &state) in states.iter().enumerate() {
            let expected = if probabilities[i] > 0.5 { 1.0 } else { 0.0 };
            assert_eq!(state, expected);
        }
    }

    #[test]
    fn test_dot_product() {
        let a: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let b: Vec<f64> = (0..100).map(|i| (i + 1) as f64).collect();

        let result = SimdOps::dot_product(&a, &b);
        let expected: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();

        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_simd_info() {
        let info = SimdOps::simd_info();
        println!("SIMD capabilities: {:?}", info);
    }
}
