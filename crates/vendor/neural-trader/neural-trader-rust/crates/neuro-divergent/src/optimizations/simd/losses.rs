//! SIMD-Accelerated Loss Functions
//!
//! Vectorized loss calculations and gradient computations.

use super::cpu_features::detect_cpu_features;

/// Mean Squared Error loss
pub fn mse(predictions: &[f32], targets: &[f32]) -> f32 {
    let features = detect_cpu_features();

    if features.has_avx2 {
        mse_avx2(predictions, targets)
    } else if features.has_neon {
        mse_neon(predictions, targets)
    } else {
        mse_scalar(predictions, targets)
    }
}

/// Mean Absolute Error loss
pub fn mae(predictions: &[f32], targets: &[f32]) -> f32 {
    let features = detect_cpu_features();

    if features.has_avx2 {
        mae_avx2(predictions, targets)
    } else if features.has_neon {
        mae_neon(predictions, targets)
    } else {
        mae_scalar(predictions, targets)
    }
}

/// MSE gradient: 2 * (predictions - targets) / n
pub fn mse_gradient(predictions: &[f32], targets: &[f32]) -> Vec<f32> {
    let features = detect_cpu_features();

    if features.has_avx2 {
        mse_gradient_avx2(predictions, targets)
    } else if features.has_neon {
        mse_gradient_neon(predictions, targets)
    } else {
        mse_gradient_scalar(predictions, targets)
    }
}

/// MAE gradient: sign(predictions - targets) / n
pub fn mae_gradient(predictions: &[f32], targets: &[f32]) -> Vec<f32> {
    let features = detect_cpu_features();

    if features.has_avx2 {
        mae_gradient_avx2(predictions, targets)
    } else if features.has_neon {
        mae_gradient_neon(predictions, targets)
    } else {
        mae_gradient_scalar(predictions, targets)
    }
}

/// Huber loss (combination of MSE and MAE)
pub fn huber_loss(predictions: &[f32], targets: &[f32], delta: f32) -> f32 {
    let features = detect_cpu_features();

    if features.has_avx2 {
        huber_loss_avx2(predictions, targets, delta)
    } else if features.has_neon {
        huber_loss_neon(predictions, targets, delta)
    } else {
        huber_loss_scalar(predictions, targets, delta)
    }
}

/// Cross-entropy loss (for classification)
pub fn cross_entropy(predictions: &[f32], targets: &[f32]) -> f32 {
    let features = detect_cpu_features();

    if features.has_avx2 {
        cross_entropy_avx2(predictions, targets)
    } else if features.has_neon {
        cross_entropy_neon(predictions, targets)
    } else {
        cross_entropy_scalar(predictions, targets)
    }
}

// ===== AVX2 Implementations =====

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn mse_avx2(predictions: &[f32], targets: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let len = predictions.len().min(targets.len());
    let mut sum = _mm256_setzero_ps();
    let mut i = 0;

    while i + 8 <= len {
        let pred = _mm256_loadu_ps(predictions.as_ptr().add(i));
        let targ = _mm256_loadu_ps(targets.as_ptr().add(i));
        let diff = _mm256_sub_ps(pred, targ);
        sum = _mm256_fmadd_ps(diff, diff, sum);
        i += 8;
    }

    let mut temp = [0.0f32; 8];
    _mm256_storeu_ps(temp.as_mut_ptr(), sum);
    let mut result: f32 = temp.iter().sum();

    while i < len {
        let diff = predictions[i] - targets[i];
        result += diff * diff;
        i += 1;
    }

    result / len as f32
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn mae_avx2(predictions: &[f32], targets: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let len = predictions.len().min(targets.len());
    let mut sum = _mm256_setzero_ps();
    let sign_mask = _mm256_set1_ps(-0.0);
    let mut i = 0;

    while i + 8 <= len {
        let pred = _mm256_loadu_ps(predictions.as_ptr().add(i));
        let targ = _mm256_loadu_ps(targets.as_ptr().add(i));
        let diff = _mm256_sub_ps(pred, targ);
        // Absolute value: clear sign bit
        let abs_diff = _mm256_andnot_ps(sign_mask, diff);
        sum = _mm256_add_ps(sum, abs_diff);
        i += 8;
    }

    let mut temp = [0.0f32; 8];
    _mm256_storeu_ps(temp.as_mut_ptr(), sum);
    let mut result: f32 = temp.iter().sum();

    while i < len {
        result += (predictions[i] - targets[i]).abs();
        i += 1;
    }

    result / len as f32
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn mse_gradient_avx2(predictions: &[f32], targets: &[f32]) -> Vec<f32> {
    use std::arch::x86_64::*;

    let len = predictions.len().min(targets.len());
    let mut gradient = vec![0.0f32; len];
    let scale = _mm256_set1_ps(2.0 / len as f32);
    let mut i = 0;

    while i + 8 <= len {
        let pred = _mm256_loadu_ps(predictions.as_ptr().add(i));
        let targ = _mm256_loadu_ps(targets.as_ptr().add(i));
        let diff = _mm256_sub_ps(pred, targ);
        let grad = _mm256_mul_ps(diff, scale);
        _mm256_storeu_ps(gradient.as_mut_ptr().add(i), grad);
        i += 8;
    }

    let scale_scalar = 2.0 / len as f32;
    while i < len {
        gradient[i] = (predictions[i] - targets[i]) * scale_scalar;
        i += 1;
    }

    gradient
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn mae_gradient_avx2(predictions: &[f32], targets: &[f32]) -> Vec<f32> {
    use std::arch::x86_64::*;

    let len = predictions.len().min(targets.len());
    let mut gradient = vec![0.0f32; len];
    let scale = _mm256_set1_ps(1.0 / len as f32);
    let zero = _mm256_setzero_ps();
    let one = _mm256_set1_ps(1.0);
    let neg_one = _mm256_set1_ps(-1.0);
    let mut i = 0;

    while i + 8 <= len {
        let pred = _mm256_loadu_ps(predictions.as_ptr().add(i));
        let targ = _mm256_loadu_ps(targets.as_ptr().add(i));
        let diff = _mm256_sub_ps(pred, targ);

        // Sign function
        let pos_mask = _mm256_cmp_ps(diff, zero, _CMP_GT_OQ);
        let neg_mask = _mm256_cmp_ps(diff, zero, _CMP_LT_OQ);
        let sign = _mm256_blendv_ps(
            _mm256_blendv_ps(zero, neg_one, neg_mask),
            one,
            pos_mask
        );

        let grad = _mm256_mul_ps(sign, scale);
        _mm256_storeu_ps(gradient.as_mut_ptr().add(i), grad);
        i += 8;
    }

    let scale_scalar = 1.0 / len as f32;
    while i < len {
        let diff = predictions[i] - targets[i];
        gradient[i] = diff.signum() * scale_scalar;
        i += 1;
    }

    gradient
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn huber_loss_avx2(predictions: &[f32], targets: &[f32], delta: f32) -> f32 {
    use std::arch::x86_64::*;

    let len = predictions.len().min(targets.len());
    let delta_vec = _mm256_set1_ps(delta);
    let half = _mm256_set1_ps(0.5);
    let mut sum = _mm256_setzero_ps();
    let sign_mask = _mm256_set1_ps(-0.0);
    let mut i = 0;

    while i + 8 <= len {
        let pred = _mm256_loadu_ps(predictions.as_ptr().add(i));
        let targ = _mm256_loadu_ps(targets.as_ptr().add(i));
        let diff = _mm256_sub_ps(pred, targ);
        let abs_diff = _mm256_andnot_ps(sign_mask, diff);

        // If |diff| <= delta: 0.5 * diff^2
        // Else: delta * (|diff| - 0.5 * delta)
        let mask = _mm256_cmp_ps(abs_diff, delta_vec, _CMP_LE_OQ);
        let squared = _mm256_mul_ps(_mm256_mul_ps(half, diff), diff);
        let linear = _mm256_mul_ps(delta_vec,
            _mm256_sub_ps(abs_diff, _mm256_mul_ps(half, delta_vec)));
        let loss = _mm256_blendv_ps(linear, squared, mask);

        sum = _mm256_add_ps(sum, loss);
        i += 8;
    }

    let mut temp = [0.0f32; 8];
    _mm256_storeu_ps(temp.as_mut_ptr(), sum);
    let mut result: f32 = temp.iter().sum();

    while i < len {
        let diff = (predictions[i] - targets[i]).abs();
        result += if diff <= delta {
            0.5 * diff * diff
        } else {
            delta * (diff - 0.5 * delta)
        };
        i += 1;
    }

    result / len as f32
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn cross_entropy_avx2(predictions: &[f32], targets: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let len = predictions.len().min(targets.len());
    let epsilon = _mm256_set1_ps(1e-7);
    let mut sum = _mm256_setzero_ps();
    let mut i = 0;

    while i + 8 <= len {
        let pred = _mm256_loadu_ps(predictions.as_ptr().add(i));
        let targ = _mm256_loadu_ps(targets.as_ptr().add(i));

        // Clip predictions to avoid log(0)
        let pred_clipped = _mm256_max_ps(pred, epsilon);

        // Compute -target * log(prediction) element-wise
        let mut temp = [0.0f32; 8];
        _mm256_storeu_ps(temp.as_mut_ptr(), pred_clipped);
        for j in 0..8 {
            temp[j] = temp[j].ln();
        }
        let log_pred = _mm256_loadu_ps(temp.as_ptr());

        let prod = _mm256_mul_ps(targ, log_pred);
        sum = _mm256_sub_ps(sum, prod);
        i += 8;
    }

    let mut temp = [0.0f32; 8];
    _mm256_storeu_ps(temp.as_mut_ptr(), sum);
    let mut result: f32 = temp.iter().sum();

    while i < len {
        let pred = predictions[i].max(1e-7);
        result -= targets[i] * pred.ln();
        i += 1;
    }

    result / len as f32
}

// ===== ARM NEON Implementations =====

#[cfg(target_arch = "aarch64")]
fn mse_neon(predictions: &[f32], targets: &[f32]) -> f32 {
    mse_scalar(predictions, targets)
}

#[cfg(target_arch = "aarch64")]
fn mae_neon(predictions: &[f32], targets: &[f32]) -> f32 {
    mae_scalar(predictions, targets)
}

#[cfg(target_arch = "aarch64")]
fn mse_gradient_neon(predictions: &[f32], targets: &[f32]) -> Vec<f32> {
    mse_gradient_scalar(predictions, targets)
}

#[cfg(target_arch = "aarch64")]
fn mae_gradient_neon(predictions: &[f32], targets: &[f32]) -> Vec<f32> {
    mae_gradient_scalar(predictions, targets)
}

#[cfg(target_arch = "aarch64")]
fn huber_loss_neon(predictions: &[f32], targets: &[f32], delta: f32) -> f32 {
    huber_loss_scalar(predictions, targets, delta)
}

#[cfg(target_arch = "aarch64")]
fn cross_entropy_neon(predictions: &[f32], targets: &[f32]) -> f32 {
    cross_entropy_scalar(predictions, targets)
}

// ===== Scalar Fallback Implementations =====

fn mse_scalar(predictions: &[f32], targets: &[f32]) -> f32 {
    let len = predictions.len().min(targets.len());
    let sum: f32 = predictions.iter()
        .zip(targets.iter())
        .map(|(p, t)| (p - t).powi(2))
        .sum();
    sum / len as f32
}

fn mae_scalar(predictions: &[f32], targets: &[f32]) -> f32 {
    let len = predictions.len().min(targets.len());
    let sum: f32 = predictions.iter()
        .zip(targets.iter())
        .map(|(p, t)| (p - t).abs())
        .sum();
    sum / len as f32
}

fn mse_gradient_scalar(predictions: &[f32], targets: &[f32]) -> Vec<f32> {
    let len = predictions.len().min(targets.len());
    let scale = 2.0 / len as f32;
    predictions.iter()
        .zip(targets.iter())
        .map(|(p, t)| (p - t) * scale)
        .collect()
}

fn mae_gradient_scalar(predictions: &[f32], targets: &[f32]) -> Vec<f32> {
    let len = predictions.len().min(targets.len());
    let scale = 1.0 / len as f32;
    predictions.iter()
        .zip(targets.iter())
        .map(|(p, t)| (p - t).signum() * scale)
        .collect()
}

fn huber_loss_scalar(predictions: &[f32], targets: &[f32], delta: f32) -> f32 {
    let len = predictions.len().min(targets.len());
    let sum: f32 = predictions.iter()
        .zip(targets.iter())
        .map(|(p, t)| {
            let diff = (p - t).abs();
            if diff <= delta {
                0.5 * diff * diff
            } else {
                delta * (diff - 0.5 * delta)
            }
        })
        .sum();
    sum / len as f32
}

fn cross_entropy_scalar(predictions: &[f32], targets: &[f32]) -> f32 {
    let len = predictions.len().min(targets.len());
    let sum: f32 = predictions.iter()
        .zip(targets.iter())
        .map(|(p, t)| -t * p.max(1e-7).ln())
        .sum();
    sum / len as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mse() {
        let predictions = vec![1.0, 2.0, 3.0, 4.0];
        let targets = vec![1.5, 2.5, 2.5, 4.5];

        let loss = mse(&predictions, &targets);
        let expected = ((0.5_f32.powi(2) + 0.5_f32.powi(2) + 0.5_f32.powi(2) + 0.5_f32.powi(2)) / 4.0);

        assert!((loss - expected).abs() < 1e-5);
    }

    #[test]
    fn test_mae() {
        let predictions = vec![1.0, 2.0, 3.0, 4.0];
        let targets = vec![1.5, 2.5, 2.5, 4.5];

        let loss = mae(&predictions, &targets);
        let expected = (0.5 + 0.5 + 0.5 + 0.5) / 4.0;

        assert!((loss - expected).abs() < 1e-5);
    }

    #[test]
    fn test_mse_gradient() {
        let predictions = vec![1.0, 2.0, 3.0, 4.0];
        let targets = vec![1.5, 2.5, 2.5, 4.5];

        let grad = mse_gradient(&predictions, &targets);

        assert_eq!(grad.len(), 4);
        // Each gradient should be 2 * (pred - target) / n
        assert!((grad[0] - (-0.25)).abs() < 1e-5);
    }

    #[test]
    fn test_huber_loss() {
        let predictions = vec![1.0, 2.0, 3.0, 4.0];
        let targets = vec![1.1, 2.1, 5.0, 4.1];
        let delta = 1.0;

        let loss = huber_loss(&predictions, &targets, delta);

        // First three should use quadratic, last one linear
        assert!(loss > 0.0);
    }
}
