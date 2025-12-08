//! Numerically stable variants of common mathematical operations
//!
//! This module provides implementations of common mathematical operations
//! that are numerically stable and avoid common pitfalls in floating-point
//! arithmetic.

use crate::error::{CdfaError, CdfaResult};
use super::kahan_simple::KahanAccumulator;

/// Numerically stable computation of log(1 + x) for small x
///
/// Uses the built-in ln_1p function which is more accurate than log(1 + x)
/// for small values of x.
pub fn log1p_stable(x: f64) -> f64 {
    x.ln_1p()
}

/// Numerically stable computation of exp(x) - 1 for small x
///
/// Uses the built-in expm1 function which is more accurate than exp(x) - 1
/// for small values of x.
pub fn expm1_stable(x: f64) -> f64 {
    x.exp_m1()
}

/// Numerically stable computation of sqrt(x^2 + y^2)
///
/// Avoids overflow/underflow issues that can occur with naive implementation.
pub fn hypot_stable(x: f64, y: f64) -> f64 {
    x.hypot(y)
}

/// Numerically stable computation of log(sum(exp(x_i)))
///
/// This function computes log(∑ exp(x_i)) in a numerically stable way
/// by factoring out the maximum value to prevent overflow.
///
/// # Mathematical Background
///
/// log(∑ exp(x_i)) = max_i + log(∑ exp(x_i - max_i))
///
/// This prevents overflow since all terms in the sum are ≤ 1.
pub fn logsumexp_stable(values: &[f64]) -> CdfaResult<f64> {
    if values.is_empty() {
        return Err(CdfaError::InvalidInput("Cannot compute logsumexp of empty array".to_string()));
    }

    let max_val = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    
    if max_val.is_infinite() && max_val.is_sign_negative() {
        return Ok(f64::NEG_INFINITY);
    }

    let mut sum_acc = KahanAccumulator::new();
    for &x in values {
        sum_acc.add((x - max_val).exp());
    }

    Ok(max_val + sum_acc.sum().ln())
}

/// Numerically stable computation of softmax function
///
/// Computes softmax(x_i) = exp(x_i) / ∑ exp(x_j) in a stable way.
pub fn softmax_stable(values: &[f64]) -> CdfaResult<Vec<f64>> {
    if values.is_empty() {
        return Err(CdfaError::InvalidInput("Cannot compute softmax of empty array".to_string()));
    }

    let max_val = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    
    // Compute shifted exponentials
    let shifted_exp: Vec<f64> = values.iter()
        .map(|&x| (x - max_val).exp())
        .collect();
    
    // Sum using Kahan for precision
    let sum = KahanAccumulator::sum_slice(&shifted_exp);
    
    // Normalize
    Ok(shifted_exp.iter().map(|&x| x / sum).collect())
}

/// Numerically stable computation of log-softmax function
///
/// Computes log_softmax(x_i) = x_i - log(∑ exp(x_j)) in a stable way.
pub fn log_softmax_stable(values: &[f64]) -> CdfaResult<Vec<f64>> {
    if values.is_empty() {
        return Err(CdfaError::InvalidInput("Cannot compute log_softmax of empty array".to_string()));
    }

    let logsumexp = logsumexp_stable(values)?;
    Ok(values.iter().map(|&x| x - logsumexp).collect())
}

/// Numerically stable computation of sample variance using Welford's algorithm
///
/// This algorithm computes the sample variance in a single pass while
/// maintaining numerical stability.
///
/// # Mathematical Background
///
/// Welford's algorithm updates the mean and sum of squared differences:
/// ```text
/// delta = x - mean
/// mean += delta / n
/// M2 += delta * (x - mean)
/// variance = M2 / (n - 1)
/// ```
pub fn welford_variance(values: &[f64]) -> CdfaResult<(f64, f64)> {
    if values.len() < 2 {
        return Err(CdfaError::InvalidInput("Need at least 2 values for variance".to_string()));
    }

    let mut mean = 0.0;
    let mut m2 = 0.0;

    for (i, &value) in values.iter().enumerate() {
        let n = (i + 1) as f64;
        let delta = value - mean;
        mean += delta / n;
        let delta2 = value - mean;
        m2 += delta * delta2;
    }

    let variance = m2 / (values.len() - 1) as f64;
    Ok((mean, variance))
}

/// Numerically stable computation of standard deviation using Welford's algorithm
pub fn welford_std_dev(values: &[f64]) -> CdfaResult<(f64, f64)> {
    let (mean, variance) = welford_variance(values)?;
    Ok((mean, variance.sqrt()))
}

/// Numerically stable computation of covariance between two series
pub fn stable_covariance(x: &[f64], y: &[f64]) -> CdfaResult<f64> {
    if x.len() != y.len() {
        return Err(CdfaError::DimensionMismatch {
            expected: x.len(),
            actual: y.len(),
        });
    }

    if x.len() < 2 {
        return Err(CdfaError::InvalidInput("Need at least 2 values for covariance".to_string()));
    }

    let mut mean_x = 0.0;
    let mut mean_y = 0.0;
    let mut c = 0.0;

    for (i, (&xi, &yi)) in x.iter().zip(y.iter()).enumerate() {
        let n = (i + 1) as f64;
        let delta_x = xi - mean_x;
        let delta_y = yi - mean_y;
        
        mean_x += delta_x / n;
        mean_y += delta_y / n;
        
        c += delta_x * delta_y * (n - 1.0) / n;
    }

    Ok(c / (x.len() - 1) as f64)
}

/// Numerically stable computation of Pearson correlation coefficient
pub fn stable_correlation(x: &[f64], y: &[f64]) -> CdfaResult<f64> {
    if x.len() != y.len() {
        return Err(CdfaError::DimensionMismatch {
            expected: x.len(),
            actual: y.len(),
        });
    }

    if x.len() < 2 {
        return Err(CdfaError::InvalidInput("Need at least 2 values for correlation".to_string()));
    }

    let mut mean_x = 0.0;
    let mut mean_y = 0.0;
    let mut sum_sq_x = 0.0;
    let mut sum_sq_y = 0.0;
    let mut sum_coproduct = 0.0;

    for (i, (&xi, &yi)) in x.iter().zip(y.iter()).enumerate() {
        let n = (i + 1) as f64;
        let delta_x = xi - mean_x;
        let delta_y = yi - mean_y;
        
        mean_x += delta_x / n;
        mean_y += delta_y / n;
        
        sum_sq_x += delta_x * (xi - mean_x);
        sum_sq_y += delta_y * (yi - mean_y);
        sum_coproduct += delta_x * (yi - mean_y);
    }

    let denominator = (sum_sq_x * sum_sq_y).sqrt();
    if denominator == 0.0 {
        Ok(0.0)
    } else {
        Ok(sum_coproduct / denominator)
    }
}

/// Numerically stable computation of compound returns
///
/// Computes (1 + r1) * (1 + r2) * ... * (1 + rn) - 1 in a stable way.
pub fn stable_compound_return(returns: &[f64]) -> CdfaResult<f64> {
    if returns.is_empty() {
        return Ok(0.0);
    }

    // Use logarithms to avoid overflow/underflow
    let mut log_acc = KahanAccumulator::new();
    
    for &r in returns {
        if r <= -1.0 {
            return Err(CdfaError::InvalidInput("Return cannot be <= -100%".to_string()));
        }
        log_acc.add((1.0 + r).ln());
    }

    Ok(log_acc.sum().exp() - 1.0)
}

/// Numerically stable computation of geometric mean
pub fn stable_geometric_mean(values: &[f64]) -> CdfaResult<f64> {
    if values.is_empty() {
        return Err(CdfaError::InvalidInput("Cannot compute geometric mean of empty array".to_string()));
    }

    for &value in values {
        if value <= 0.0 {
            return Err(CdfaError::InvalidInput("All values must be positive for geometric mean".to_string()));
        }
    }

    // Use logarithms for stability
    let mut log_acc = KahanAccumulator::new();
    for &value in values {
        log_acc.add(value.ln());
    }

    let log_mean = log_acc.sum() / values.len() as f64;
    Ok(log_mean.exp())
}

/// Numerically stable computation of harmonic mean
pub fn stable_harmonic_mean(values: &[f64]) -> CdfaResult<f64> {
    if values.is_empty() {
        return Err(CdfaError::InvalidInput("Cannot compute harmonic mean of empty array".to_string()));
    }

    let mut reciprocal_acc = KahanAccumulator::new();
    
    for &value in values {
        if value == 0.0 {
            return Err(CdfaError::InvalidInput("Values cannot be zero for harmonic mean".to_string()));
        }
        reciprocal_acc.add(1.0 / value);
    }

    let reciprocal_mean = reciprocal_acc.sum() / values.len() as f64;
    Ok(1.0 / reciprocal_mean)
}

/// Numerically stable computation of relative error
pub fn stable_relative_error(actual: f64, expected: f64) -> f64 {
    if expected == 0.0 {
        if actual == 0.0 {
            0.0
        } else {
            f64::INFINITY
        }
    } else {
        ((actual - expected) / expected).abs()
    }
}

/// Check if two floating-point numbers are approximately equal within tolerance
pub fn approximately_equal(a: f64, b: f64, abs_tol: f64, rel_tol: f64) -> bool {
    if a == b {
        return true;
    }

    let diff = (a - b).abs();
    
    // Check absolute tolerance
    if diff <= abs_tol {
        return true;
    }

    // Check relative tolerance
    let max_val = a.abs().max(b.abs());
    diff <= rel_tol * max_val
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_logsumexp_stable() {
        let values = vec![1000.0, 1001.0, 1002.0];
        let result = logsumexp_stable(&values).unwrap();
        
        // Should not overflow and should be approximately 1002 + log(1 + e^(-1) + e^(-2))
        assert!(result.is_finite());
        assert!(result > 1002.0);
    }

    #[test]
    fn test_softmax_stable() {
        let values = vec![1000.0, 1001.0, 1002.0];
        let result = softmax_stable(&values).unwrap();
        
        // Should sum to 1.0 and not overflow
        let sum: f64 = result.iter().sum();
        assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-15);
        
        // Largest input should have largest probability
        assert!(result[2] > result[1]);
        assert!(result[1] > result[0]);
    }

    #[test]
    fn test_welford_variance() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (mean, variance) = welford_variance(&values).unwrap();
        
        assert_abs_diff_eq!(mean, 3.0, epsilon = 1e-15);
        assert_abs_diff_eq!(variance, 2.5, epsilon = 1e-15);
    }

    #[test]
    fn test_stable_correlation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        
        let correlation = stable_correlation(&x, &y).unwrap();
        assert_abs_diff_eq!(correlation, 1.0, epsilon = 1e-15);
    }

    #[test]
    fn test_stable_compound_return() {
        let returns = vec![0.1, 0.05, -0.02, 0.08];
        let compound = stable_compound_return(&returns).unwrap();
        
        // Manual calculation: (1.1 * 1.05 * 0.98 * 1.08) - 1
        let expected = 1.1 * 1.05 * 0.98 * 1.08 - 1.0;
        assert_abs_diff_eq!(compound, expected, epsilon = 1e-15);
    }

    #[test]
    fn test_stable_geometric_mean() {
        let values = vec![1.0, 4.0, 9.0, 16.0];
        let geo_mean = stable_geometric_mean(&values).unwrap();
        
        // Geometric mean of 1, 4, 9, 16 is 4th root of 576 = 4.898...
        let expected = (1.0 * 4.0 * 9.0 * 16.0).powf(1.0 / 4.0);
        assert_abs_diff_eq!(geo_mean, expected, epsilon = 1e-15);
    }

    #[test]
    fn test_stable_harmonic_mean() {
        let values = vec![1.0, 2.0, 4.0];
        let harm_mean = stable_harmonic_mean(&values).unwrap();
        
        // Harmonic mean of 1, 2, 4 is 3 / (1/1 + 1/2 + 1/4) = 3 / 1.75 = 12/7
        let expected = 3.0 / (1.0 + 0.5 + 0.25);
        assert_abs_diff_eq!(harm_mean, expected, epsilon = 1e-15);
    }

    #[test]
    fn test_approximately_equal() {
        assert!(approximately_equal(1.0, 1.0, 1e-15, 1e-15));
        assert!(approximately_equal(1.0, 1.0 + 1e-16, 1e-15, 1e-15));
        assert!(!approximately_equal(1.0, 1.1, 1e-15, 1e-15));
        
        // Test relative tolerance
        assert!(approximately_equal(1e6, 1e6 + 1.0, 1e-15, 1e-6));
    }

    #[test]
    fn test_log1p_expm1_stability() {
        let small_x = 1e-15;
        
        // These should be more accurate than naive implementations
        let log1p_result = log1p_stable(small_x);
        let expm1_result = expm1_stable(small_x);
        
        // For small x, log(1+x) ≈ x and exp(x)-1 ≈ x
        assert_abs_diff_eq!(log1p_result, small_x, epsilon = 1e-30);
        assert_abs_diff_eq!(expm1_result, small_x, epsilon = 1e-30);
    }
}