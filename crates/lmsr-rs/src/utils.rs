//! Utility functions for numerical stability and performance

use crate::errors::{LMSRError, Result};
use std::f64;

/// Numerical constants for stability
pub const EPSILON: f64 = 1e-15;
pub const MAX_EXP_ARG: f64 = 700.0; // Prevents overflow in exp()
pub const MIN_EXP_ARG: f64 = -700.0; // Prevents underflow in exp()
pub const LOG_EPSILON: f64 = -34.5; // log(EPSILON)

/// Numerically stable computation of log(sum(exp(x_i)))
/// Uses the log-sum-exp trick to prevent overflow/underflow
pub fn log_sum_exp(values: &[f64]) -> Result<f64> {
    if values.is_empty() {
        return Err(LMSRError::invalid_quantity("Empty values array"));
    }
    
    // Find maximum value to subtract for numerical stability
    let max_val = values.iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    
    if !max_val.is_finite() {
        return Err(LMSRError::numerical_error("Non-finite maximum value"));
    }
    
    // If max_val is too large, cap it to prevent overflow
    let safe_max = max_val.min(MAX_EXP_ARG);
    
    let sum: f64 = values.iter()
        .map(|&x| {
            let diff = x - safe_max;
            if diff < MIN_EXP_ARG {
                0.0 // Underflow to zero
            } else {
                diff.exp()
            }
        })
        .sum();
    
    if sum <= 0.0 {
        return Err(LMSRError::numerical_error("Sum of exponentials is non-positive"));
    }
    
    let result = safe_max + sum.ln();
    
    if !result.is_finite() {
        return Err(LMSRError::numerical_error("log_sum_exp result is not finite"));
    }
    
    Ok(result)
}

/// Numerically stable computation of exp(x) / sum(exp(x_i))
/// Returns softmax probabilities
pub fn softmax(values: &[f64]) -> Result<Vec<f64>> {
    if values.is_empty() {
        return Err(LMSRError::invalid_quantity("Empty values array"));
    }
    
    let log_sum = log_sum_exp(values)?;
    
    let mut result = Vec::with_capacity(values.len());
    for &x in values {
        let diff = x - log_sum;
        let prob = if diff < MIN_EXP_ARG {
            0.0
        } else {
            diff.exp()
        };
        
        if !prob.is_finite() {
            return Err(LMSRError::numerical_error("Softmax probability is not finite"));
        }
        
        result.push(prob);
    }
    
    // Normalize to ensure probabilities sum to 1.0
    let sum: f64 = result.iter().sum();
    if sum < EPSILON {
        return Err(LMSRError::numerical_error("Softmax probabilities sum to near zero"));
    }
    
    for prob in &mut result {
        *prob /= sum;
    }
    
    Ok(result)
}

/// Safe natural logarithm that handles edge cases
pub fn safe_ln(x: f64) -> Result<f64> {
    if x <= 0.0 {
        return Err(LMSRError::numerical_error(format!("Cannot take log of non-positive value: {}", x)));
    }
    
    if x < EPSILON {
        return Ok(LOG_EPSILON);
    }
    
    let result = x.ln();
    if !result.is_finite() {
        return Err(LMSRError::numerical_error(format!("Logarithm of {} is not finite", x)));
    }
    
    Ok(result)
}

/// Safe exponential that prevents overflow/underflow
pub fn safe_exp(x: f64) -> Result<f64> {
    if !x.is_finite() {
        return Err(LMSRError::numerical_error(format!("Cannot exponentiate non-finite value: {}", x)));
    }
    
    if x > MAX_EXP_ARG {
        return Ok(f64::INFINITY);
    }
    
    if x < MIN_EXP_ARG {
        return Ok(0.0);
    }
    
    let result = x.exp();
    if !result.is_finite() && x.is_finite() {
        return Err(LMSRError::numerical_error(format!("Exponential of {} is not finite", x)));
    }
    
    Ok(result)
}

/// Validate that a vector contains only finite values
pub fn validate_finite_vector(values: &[f64], name: &str) -> Result<()> {
    for (i, &val) in values.iter().enumerate() {
        if !val.is_finite() {
            return Err(LMSRError::numerical_error(
                format!("{} contains non-finite value at index {}: {}", name, i, val)
            ));
        }
    }
    Ok(())
}

/// Validate that a value is positive and finite
pub fn validate_positive_finite(value: f64, name: &str) -> Result<()> {
    if !value.is_finite() {
        return Err(LMSRError::numerical_error(
            format!("{} is not finite: {}", name, value)
        ));
    }
    
    if value <= 0.0 {
        return Err(LMSRError::invalid_liquidity(value));
    }
    
    Ok(())
}

/// Clamp a value to prevent numerical issues
pub fn clamp_for_stability(value: f64, min_val: f64, max_val: f64) -> f64 {
    value.max(min_val).min(max_val)
}

/// Validate liquidity parameter
pub fn validate_liquidity(liquidity: f64) -> Result<()> {
    validate_positive_finite(liquidity, "liquidity")
}

/// Validate that array is not empty
pub fn validate_not_empty<T>(arr: &[T], name: &str) -> Result<()> {
    if arr.is_empty() {
        return Err(LMSRError::invalid_quantity(format!("{} array cannot be empty", name)));
    }
    Ok(())
}

/// Validate probabilities array (must sum to 1.0)
pub fn validate_probabilities(probs: &[f64]) -> Result<()> {
    validate_not_empty(probs, "probabilities")?;
    validate_finite_vector(probs, "probabilities")?;
    
    // Check all probabilities are non-negative
    for (i, &p) in probs.iter().enumerate() {
        if p < 0.0 {
            return Err(LMSRError::invalid_quantity(
                format!("Probability at index {} is negative: {}", i, p)
            ));
        }
    }
    
    // Check probabilities sum to approximately 1.0
    let sum: f64 = probs.iter().sum();
    if (sum - 1.0).abs() > EPSILON * 100.0 {
        return Err(LMSRError::invalid_quantity(
            format!("Probabilities sum to {} instead of 1.0", sum)
        ));
    }
    
    Ok(())
}

/// Safe division that handles edge cases
pub fn safe_divide(numerator: f64, denominator: f64) -> Result<f64> {
    if !numerator.is_finite() {
        return Err(LMSRError::numerical_error(format!("Numerator is not finite: {}", numerator)));
    }
    if !denominator.is_finite() {
        return Err(LMSRError::numerical_error(format!("Denominator is not finite: {}", denominator)));
    }
    if denominator.abs() < EPSILON {
        return Err(LMSRError::numerical_error("Division by zero or near-zero"));
    }
    
    let result = numerator / denominator;
    if !result.is_finite() {
        return Err(LMSRError::numerical_error(format!("Division result is not finite: {} / {}", numerator, denominator)));
    }
    
    Ok(result)
}

/// Safe log function
pub fn safe_log(x: f64) -> Result<f64> {
    safe_ln(x)
}

/// Normalize probabilities to ensure they sum to 1.0
pub fn normalize_probabilities(probs: &[f64]) -> Result<Vec<f64>> {
    validate_not_empty(probs, "probabilities")?;
    validate_finite_vector(probs, "probabilities")?;
    
    let sum: f64 = probs.iter().sum();
    if sum <= 0.0 {
        return Err(LMSRError::numerical_error("Cannot normalize probabilities with non-positive sum"));
    }
    
    Ok(probs.iter().map(|&p| p / sum).collect())
}

/// Convert log odds to probabilities
pub fn log_odds_to_probabilities(log_odds: &[f64]) -> Result<Vec<f64>> {
    validate_not_empty(log_odds, "log_odds")?;
    softmax(log_odds)
}

/// Convert probabilities to log odds
pub fn probabilities_to_log_odds(probs: &[f64]) -> Result<Vec<f64>> {
    validate_probabilities(probs)?;
    
    let mut log_odds = Vec::with_capacity(probs.len());
    for &p in probs {
        if p <= 0.0 {
            log_odds.push(MIN_EXP_ARG); // Very small probability
        } else {
            log_odds.push(safe_ln(p)?);
        }
    }
    
    Ok(log_odds)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_log_sum_exp_stability() {
        // Test with large values that would overflow naive implementation
        let large_values = vec![1000.0, 1001.0, 1002.0];
        let result = log_sum_exp(&large_values).unwrap();
        assert!(result.is_finite());
        assert!(result > 1000.0);
        
        // Test with small values
        let small_values = vec![-1000.0, -1001.0, -1002.0];
        let result = log_sum_exp(&small_values).unwrap();
        assert!(result.is_finite());
        assert!(result < -999.0);
    }

    #[test]
    fn test_softmax_probabilities() {
        let values = vec![1.0, 2.0, 3.0];
        let probs = softmax(&values).unwrap();
        
        // Check that probabilities sum to 1.0
        let sum: f64 = probs.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-10);
        
        // Check that all probabilities are positive
        for &p in &probs {
            assert!(p > 0.0);
        }
        
        // Check that probabilities are ordered correctly
        assert!(probs[0] < probs[1]);
        assert!(probs[1] < probs[2]);
    }

    #[test]
    fn test_safe_ln() {
        assert!(safe_ln(1.0).unwrap() == 0.0);
        assert_relative_eq!(safe_ln(std::f64::consts::E).unwrap(), 1.0, epsilon = 1e-10);
        assert!(safe_ln(0.0).is_err());
        assert!(safe_ln(-1.0).is_err());
    }

    #[test]
    fn test_safe_exp() {
        assert_relative_eq!(safe_exp(0.0).unwrap(), 1.0, epsilon = 1e-10);
        assert_relative_eq!(safe_exp(1.0).unwrap(), std::f64::consts::E, epsilon = 1e-10);
        assert_eq!(safe_exp(1000.0).unwrap(), f64::INFINITY);
        assert_eq!(safe_exp(-1000.0).unwrap(), 0.0);
    }
}