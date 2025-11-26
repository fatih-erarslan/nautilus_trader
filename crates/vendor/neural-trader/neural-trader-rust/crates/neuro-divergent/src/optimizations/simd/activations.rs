//! SIMD-Accelerated Activation Functions
//!
//! Vectorized implementations of common neural network activation functions.

use super::cpu_features::detect_cpu_features;

/// ReLU activation: max(0, x)
pub fn relu(x: &[f32]) -> Vec<f32> {
    let features = detect_cpu_features();

    if features.has_avx2 {
        relu_avx2(x)
    } else if features.has_neon {
        relu_neon(x)
    } else {
        relu_scalar(x)
    }
}

/// GELU activation (Gaussian Error Linear Unit)
pub fn gelu(x: &[f32]) -> Vec<f32> {
    let features = detect_cpu_features();

    if features.has_avx2 {
        gelu_avx2(x)
    } else if features.has_neon {
        gelu_neon(x)
    } else {
        gelu_scalar(x)
    }
}

/// Tanh activation
pub fn tanh(x: &[f32]) -> Vec<f32> {
    let features = detect_cpu_features();

    if features.has_avx2 {
        tanh_avx2(x)
    } else if features.has_neon {
        tanh_neon(x)
    } else {
        tanh_scalar(x)
    }
}

/// Sigmoid activation: 1 / (1 + exp(-x))
pub fn sigmoid(x: &[f32]) -> Vec<f32> {
    let features = detect_cpu_features();

    if features.has_avx2 {
        sigmoid_avx2(x)
    } else if features.has_neon {
        sigmoid_neon(x)
    } else {
        sigmoid_scalar(x)
    }
}

/// Softmax activation
pub fn softmax(x: &[f32]) -> Vec<f32> {
    let features = detect_cpu_features();

    if features.has_avx2 {
        softmax_avx2(x)
    } else if features.has_neon {
        softmax_neon(x)
    } else {
        softmax_scalar(x)
    }
}

/// Leaky ReLU activation
pub fn leaky_relu(x: &[f32], alpha: f32) -> Vec<f32> {
    let features = detect_cpu_features();

    if features.has_avx2 {
        leaky_relu_avx2(x, alpha)
    } else if features.has_neon {
        leaky_relu_neon(x, alpha)
    } else {
        leaky_relu_scalar(x, alpha)
    }
}

// ===== AVX2 Implementations =====

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn relu_avx2(x: &[f32]) -> Vec<f32> {
    use std::arch::x86_64::*;

    let len = x.len();
    let mut result = vec![0.0f32; len];
    let zeros = _mm256_setzero_ps();
    let mut i = 0;

    while i + 8 <= len {
        let x_vec = _mm256_loadu_ps(x.as_ptr().add(i));
        let relu_vec = _mm256_max_ps(x_vec, zeros);
        _mm256_storeu_ps(result.as_mut_ptr().add(i), relu_vec);
        i += 8;
    }

    while i < len {
        result[i] = x[i].max(0.0);
        i += 1;
    }

    result
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn gelu_avx2(x: &[f32]) -> Vec<f32> {
    // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    use std::arch::x86_64::*;

    let len = x.len();
    let mut result = vec![0.0f32; len];

    let half = _mm256_set1_ps(0.5);
    let one = _mm256_set1_ps(1.0);
    let sqrt_2_over_pi = _mm256_set1_ps(0.7978845608);
    let coeff = _mm256_set1_ps(0.044715);

    let mut i = 0;
    while i + 8 <= len {
        let x_vec = _mm256_loadu_ps(x.as_ptr().add(i));
        let x_cubed = _mm256_mul_ps(_mm256_mul_ps(x_vec, x_vec), x_vec);
        let inner = _mm256_fmadd_ps(coeff, x_cubed, x_vec);
        let scaled = _mm256_mul_ps(sqrt_2_over_pi, inner);

        // Fast tanh approximation
        let tanh_vec = fast_tanh_avx2(scaled);
        let one_plus_tanh = _mm256_add_ps(one, tanh_vec);
        let gelu_vec = _mm256_mul_ps(_mm256_mul_ps(half, x_vec), one_plus_tanh);

        _mm256_storeu_ps(result.as_mut_ptr().add(i), gelu_vec);
        i += 8;
    }

    while i < len {
        result[i] = gelu_scalar_single(x[i]);
        i += 1;
    }

    result
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn fast_tanh_avx2(x: __m256) -> __m256 {
    use std::arch::x86_64::*;

    // Clamp to [-3, 3] for numerical stability
    let three = _mm256_set1_ps(3.0);
    let neg_three = _mm256_set1_ps(-3.0);
    let x_clamped = _mm256_min_ps(_mm256_max_ps(x, neg_three), three);

    // Polynomial approximation: x * (1 - x^2/3)
    let x_sq = _mm256_mul_ps(x_clamped, x_clamped);
    let one = _mm256_set1_ps(1.0);
    let one_third = _mm256_set1_ps(0.333333);
    let term = _mm256_fnmadd_ps(x_sq, one_third, one);
    _mm256_mul_ps(x_clamped, term)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn tanh_avx2(x: &[f32]) -> Vec<f32> {
    use std::arch::x86_64::*;

    let len = x.len();
    let mut result = vec![0.0f32; len];
    let mut i = 0;

    while i + 8 <= len {
        let x_vec = _mm256_loadu_ps(x.as_ptr().add(i));
        let tanh_vec = fast_tanh_avx2(x_vec);
        _mm256_storeu_ps(result.as_mut_ptr().add(i), tanh_vec);
        i += 8;
    }

    while i < len {
        result[i] = x[i].tanh();
        i += 1;
    }

    result
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sigmoid_avx2(x: &[f32]) -> Vec<f32> {
    use std::arch::x86_64::*;

    let len = x.len();
    let mut result = vec![0.0f32; len];
    let one = _mm256_set1_ps(1.0);
    let half = _mm256_set1_ps(0.5);

    let mut i = 0;
    while i + 8 <= len {
        let x_vec = _mm256_loadu_ps(x.as_ptr().add(i));

        // Sigmoid approximation: 0.5 * (tanh(x/2) + 1)
        let x_half = _mm256_mul_ps(x_vec, half);
        let tanh_vec = fast_tanh_avx2(x_half);
        let sigmoid_vec = _mm256_mul_ps(half, _mm256_add_ps(tanh_vec, one));

        _mm256_storeu_ps(result.as_mut_ptr().add(i), sigmoid_vec);
        i += 8;
    }

    while i < len {
        result[i] = 1.0 / (1.0 + (-x[i]).exp());
        i += 1;
    }

    result
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn softmax_avx2(x: &[f32]) -> Vec<f32> {
    use std::arch::x86_64::*;

    // Find max for numerical stability
    let max_val = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let max_vec = _mm256_set1_ps(max_val);

    let len = x.len();
    let mut exp_vals = vec![0.0f32; len];
    let mut i = 0;

    // Compute exp(x - max)
    while i + 8 <= len {
        let x_vec = _mm256_loadu_ps(x.as_ptr().add(i));
        let shifted = _mm256_sub_ps(x_vec, max_vec);

        // Fast exp approximation (would need proper implementation)
        let mut temp = [0.0f32; 8];
        _mm256_storeu_ps(temp.as_mut_ptr(), shifted);
        for j in 0..8 {
            temp[j] = temp[j].exp();
        }
        let exp_vec = _mm256_loadu_ps(temp.as_ptr());

        _mm256_storeu_ps(exp_vals.as_mut_ptr().add(i), exp_vec);
        i += 8;
    }

    while i < len {
        exp_vals[i] = (x[i] - max_val).exp();
        i += 1;
    }

    // Compute sum and normalize
    let sum: f32 = exp_vals.iter().sum();
    exp_vals.iter().map(|&v| v / sum).collect()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn leaky_relu_avx2(x: &[f32], alpha: f32) -> Vec<f32> {
    use std::arch::x86_64::*;

    let len = x.len();
    let mut result = vec![0.0f32; len];
    let zeros = _mm256_setzero_ps();
    let alpha_vec = _mm256_set1_ps(alpha);
    let mut i = 0;

    while i + 8 <= len {
        let x_vec = _mm256_loadu_ps(x.as_ptr().add(i));
        let mask = _mm256_cmp_ps(x_vec, zeros, _CMP_GT_OQ);
        let negative_part = _mm256_mul_ps(x_vec, alpha_vec);
        let leaky_relu_vec = _mm256_blendv_ps(negative_part, x_vec, mask);
        _mm256_storeu_ps(result.as_mut_ptr().add(i), leaky_relu_vec);
        i += 8;
    }

    while i < len {
        result[i] = if x[i] > 0.0 { x[i] } else { alpha * x[i] };
        i += 1;
    }

    result
}

// ===== ARM NEON Implementations =====

#[cfg(target_arch = "aarch64")]
fn relu_neon(x: &[f32]) -> Vec<f32> {
    relu_scalar(x)
}

#[cfg(target_arch = "aarch64")]
fn gelu_neon(x: &[f32]) -> Vec<f32> {
    gelu_scalar(x)
}

#[cfg(target_arch = "aarch64")]
fn tanh_neon(x: &[f32]) -> Vec<f32> {
    tanh_scalar(x)
}

#[cfg(target_arch = "aarch64")]
fn sigmoid_neon(x: &[f32]) -> Vec<f32> {
    sigmoid_scalar(x)
}

#[cfg(target_arch = "aarch64")]
fn softmax_neon(x: &[f32]) -> Vec<f32> {
    softmax_scalar(x)
}

#[cfg(target_arch = "aarch64")]
fn leaky_relu_neon(x: &[f32], alpha: f32) -> Vec<f32> {
    leaky_relu_scalar(x, alpha)
}

// ===== Scalar Fallback Implementations =====

fn relu_scalar(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| v.max(0.0)).collect()
}

fn gelu_scalar_single(x: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.7978845608;
    0.5 * x * (1.0 + (SQRT_2_OVER_PI * (x + 0.044715 * x.powi(3))).tanh())
}

fn gelu_scalar(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| gelu_scalar_single(v)).collect()
}

fn tanh_scalar(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| v.tanh()).collect()
}

fn sigmoid_scalar(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| 1.0 / (1.0 + (-v).exp())).collect()
}

fn softmax_scalar(x: &[f32]) -> Vec<f32> {
    let max_val = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp_vals: Vec<f32> = x.iter().map(|&v| (v - max_val).exp()).collect();
    let sum: f32 = exp_vals.iter().sum();
    exp_vals.iter().map(|&v| v / sum).collect()
}

fn leaky_relu_scalar(x: &[f32], alpha: f32) -> Vec<f32> {
    x.iter().map(|&v| if v > 0.0 { v } else { alpha * v }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relu() {
        let x = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let result = relu(&x);
        assert_eq!(result, vec![0.0, 0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_sigmoid() {
        let x = vec![0.0, 1.0, -1.0];
        let result = sigmoid(&x);

        assert!((result[0] - 0.5).abs() < 0.01);
        assert!((result[1] - 0.731).abs() < 0.01);
        assert!((result[2] - 0.268).abs() < 0.01);
    }

    #[test]
    fn test_softmax() {
        let x = vec![1.0, 2.0, 3.0];
        let result = softmax(&x);

        let sum: f32 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(result[2] > result[1]);
        assert!(result[1] > result[0]);
    }

    #[test]
    fn test_leaky_relu() {
        let x = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let alpha = 0.01;
        let result = leaky_relu(&x, alpha);

        assert!((result[0] + 0.02).abs() < 1e-5);
        assert!((result[1] + 0.01).abs() < 1e-5);
        assert_eq!(result[2], 0.0);
        assert_eq!(result[3], 1.0);
        assert_eq!(result[4], 2.0);
    }
}
