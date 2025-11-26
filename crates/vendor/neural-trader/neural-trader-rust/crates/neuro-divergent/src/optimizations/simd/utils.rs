//! SIMD Utility Functions
//!
//! Helper functions for SIMD operations, memory alignment, and vectorization.

use super::cpu_features::detect_cpu_features;

/// Vector reduce (sum all elements)
pub fn reduce_sum(x: &[f32]) -> f32 {
    let features = detect_cpu_features();

    if features.has_avx2 {
        reduce_sum_avx2(x)
    } else if features.has_neon {
        reduce_sum_neon(x)
    } else {
        x.iter().sum()
    }
}

/// Vector reduce (maximum element)
pub fn reduce_max(x: &[f32]) -> f32 {
    let features = detect_cpu_features();

    if features.has_avx2 {
        reduce_max_avx2(x)
    } else if features.has_neon {
        reduce_max_neon(x)
    } else {
        x.iter().copied().fold(f32::NEG_INFINITY, f32::max)
    }
}

/// Vector reduce (minimum element)
pub fn reduce_min(x: &[f32]) -> f32 {
    let features = detect_cpu_features();

    if features.has_avx2 {
        reduce_min_avx2(x)
    } else if features.has_neon {
        reduce_min_neon(x)
    } else {
        x.iter().copied().fold(f32::INFINITY, f32::min)
    }
}

/// Scalar multiply: x * scalar
pub fn scalar_mul(x: &[f32], scalar: f32) -> Vec<f32> {
    let features = detect_cpu_features();

    if features.has_avx2 {
        scalar_mul_avx2(x, scalar)
    } else if features.has_neon {
        scalar_mul_neon(x, scalar)
    } else {
        x.iter().map(|&v| v * scalar).collect()
    }
}

/// Scalar add: x + scalar
pub fn scalar_add(x: &[f32], scalar: f32) -> Vec<f32> {
    let features = detect_cpu_features();

    if features.has_avx2 {
        scalar_add_avx2(x, scalar)
    } else if features.has_neon {
        scalar_add_neon(x, scalar)
    } else {
        x.iter().map(|&v| v + scalar).collect()
    }
}

/// Fill vector with scalar value
pub fn fill(len: usize, value: f32) -> Vec<f32> {
    vec![value; len]
}

/// Clamp vector values to range [min, max]
pub fn clamp(x: &[f32], min: f32, max: f32) -> Vec<f32> {
    let features = detect_cpu_features();

    if features.has_avx2 {
        clamp_avx2(x, min, max)
    } else if features.has_neon {
        clamp_neon(x, min, max)
    } else {
        x.iter().map(|&v| v.clamp(min, max)).collect()
    }
}

/// Compute vector norm (L2 norm)
pub fn norm_l2(x: &[f32]) -> f32 {
    let features = detect_cpu_features();

    if features.has_avx2 {
        norm_l2_avx2(x)
    } else if features.has_neon {
        norm_l2_neon(x)
    } else {
        x.iter().map(|&v| v * v).sum::<f32>().sqrt()
    }
}

// ===== AVX2 Implementations =====

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn reduce_sum_avx2(x: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let len = x.len();
    let mut sum = _mm256_setzero_ps();
    let mut i = 0;

    while i + 8 <= len {
        let vec = _mm256_loadu_ps(x.as_ptr().add(i));
        sum = _mm256_add_ps(sum, vec);
        i += 8;
    }

    let mut temp = [0.0f32; 8];
    _mm256_storeu_ps(temp.as_mut_ptr(), sum);
    let mut result: f32 = temp.iter().sum();

    while i < len {
        result += x[i];
        i += 1;
    }

    result
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn reduce_max_avx2(x: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let len = x.len();
    let mut max_vec = _mm256_set1_ps(f32::NEG_INFINITY);
    let mut i = 0;

    while i + 8 <= len {
        let vec = _mm256_loadu_ps(x.as_ptr().add(i));
        max_vec = _mm256_max_ps(max_vec, vec);
        i += 8;
    }

    let mut temp = [f32::NEG_INFINITY; 8];
    _mm256_storeu_ps(temp.as_mut_ptr(), max_vec);
    let mut result = temp.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    while i < len {
        result = result.max(x[i]);
        i += 1;
    }

    result
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn reduce_min_avx2(x: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let len = x.len();
    let mut min_vec = _mm256_set1_ps(f32::INFINITY);
    let mut i = 0;

    while i + 8 <= len {
        let vec = _mm256_loadu_ps(x.as_ptr().add(i));
        min_vec = _mm256_min_ps(min_vec, vec);
        i += 8;
    }

    let mut temp = [f32::INFINITY; 8];
    _mm256_storeu_ps(temp.as_mut_ptr(), min_vec);
    let mut result = temp.iter().copied().fold(f32::INFINITY, f32::min);

    while i < len {
        result = result.min(x[i]);
        i += 1;
    }

    result
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn scalar_mul_avx2(x: &[f32], scalar: f32) -> Vec<f32> {
    use std::arch::x86_64::*;

    let len = x.len();
    let mut result = vec![0.0f32; len];
    let scalar_vec = _mm256_set1_ps(scalar);
    let mut i = 0;

    while i + 8 <= len {
        let vec = _mm256_loadu_ps(x.as_ptr().add(i));
        let prod = _mm256_mul_ps(vec, scalar_vec);
        _mm256_storeu_ps(result.as_mut_ptr().add(i), prod);
        i += 8;
    }

    while i < len {
        result[i] = x[i] * scalar;
        i += 1;
    }

    result
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn scalar_add_avx2(x: &[f32], scalar: f32) -> Vec<f32> {
    use std::arch::x86_64::*;

    let len = x.len();
    let mut result = vec![0.0f32; len];
    let scalar_vec = _mm256_set1_ps(scalar);
    let mut i = 0;

    while i + 8 <= len {
        let vec = _mm256_loadu_ps(x.as_ptr().add(i));
        let sum = _mm256_add_ps(vec, scalar_vec);
        _mm256_storeu_ps(result.as_mut_ptr().add(i), sum);
        i += 8;
    }

    while i < len {
        result[i] = x[i] + scalar;
        i += 1;
    }

    result
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn clamp_avx2(x: &[f32], min: f32, max: f32) -> Vec<f32> {
    use std::arch::x86_64::*;

    let len = x.len();
    let mut result = vec![0.0f32; len];
    let min_vec = _mm256_set1_ps(min);
    let max_vec = _mm256_set1_ps(max);
    let mut i = 0;

    while i + 8 <= len {
        let vec = _mm256_loadu_ps(x.as_ptr().add(i));
        let clamped = _mm256_min_ps(_mm256_max_ps(vec, min_vec), max_vec);
        _mm256_storeu_ps(result.as_mut_ptr().add(i), clamped);
        i += 8;
    }

    while i < len {
        result[i] = x[i].clamp(min, max);
        i += 1;
    }

    result
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn norm_l2_avx2(x: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let len = x.len();
    let mut sum = _mm256_setzero_ps();
    let mut i = 0;

    while i + 8 <= len {
        let vec = _mm256_loadu_ps(x.as_ptr().add(i));
        sum = _mm256_fmadd_ps(vec, vec, sum);
        i += 8;
    }

    let mut temp = [0.0f32; 8];
    _mm256_storeu_ps(temp.as_mut_ptr(), sum);
    let mut result: f32 = temp.iter().sum();

    while i < len {
        result += x[i] * x[i];
        i += 1;
    }

    result.sqrt()
}

// ===== ARM NEON Implementations =====

#[cfg(target_arch = "aarch64")]
fn reduce_sum_neon(x: &[f32]) -> f32 {
    x.iter().sum()
}

#[cfg(target_arch = "aarch64")]
fn reduce_max_neon(x: &[f32]) -> f32 {
    x.iter().copied().fold(f32::NEG_INFINITY, f32::max)
}

#[cfg(target_arch = "aarch64")]
fn reduce_min_neon(x: &[f32]) -> f32 {
    x.iter().copied().fold(f32::INFINITY, f32::min)
}

#[cfg(target_arch = "aarch64")]
fn scalar_mul_neon(x: &[f32], scalar: f32) -> Vec<f32> {
    x.iter().map(|&v| v * scalar).collect()
}

#[cfg(target_arch = "aarch64")]
fn scalar_add_neon(x: &[f32], scalar: f32) -> Vec<f32> {
    x.iter().map(|&v| v + scalar).collect()
}

#[cfg(target_arch = "aarch64")]
fn clamp_neon(x: &[f32], min: f32, max: f32) -> Vec<f32> {
    x.iter().map(|&v| v.clamp(min, max)).collect()
}

#[cfg(target_arch = "aarch64")]
fn norm_l2_neon(x: &[f32]) -> f32 {
    x.iter().map(|&v| v * v).sum::<f32>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reduce_sum() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let sum = reduce_sum(&x);
        assert_eq!(sum, 15.0);
    }

    #[test]
    fn test_reduce_max() {
        let x = vec![1.0, 5.0, 3.0, 2.0, 4.0];
        let max = reduce_max(&x);
        assert_eq!(max, 5.0);
    }

    #[test]
    fn test_reduce_min() {
        let x = vec![3.0, 1.0, 5.0, 2.0, 4.0];
        let min = reduce_min(&x);
        assert_eq!(min, 1.0);
    }

    #[test]
    fn test_scalar_mul() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let result = scalar_mul(&x, 2.0);
        assert_eq!(result, vec![2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_clamp() {
        let x = vec![-1.0, 0.5, 2.0, 5.0];
        let result = clamp(&x, 0.0, 3.0);
        assert_eq!(result, vec![0.0, 0.5, 2.0, 3.0]);
    }

    #[test]
    fn test_norm_l2() {
        let x = vec![3.0, 4.0];
        let norm = norm_l2(&x);
        assert_eq!(norm, 5.0);
    }
}
