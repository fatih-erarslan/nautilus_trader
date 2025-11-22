//! Utility functions for prospect theory calculations

use crate::errors::{ProspectTheoryError, Result};
use crate::FINANCIAL_PRECISION;

/// Check if two floating point numbers are equal within financial precision
pub fn approx_equal(a: f64, b: f64) -> bool {
    (a - b).abs() < FINANCIAL_PRECISION
}

/// Safe power function that handles edge cases
pub fn safe_pow(base: f64, exponent: f64) -> Result<f64> {
    if !base.is_finite() || !exponent.is_finite() {
        return Err(ProspectTheoryError::computation_failed(
            "Non-finite input to power function",
        ));
    }

    if base == 0.0 && exponent < 0.0 {
        return Err(ProspectTheoryError::division_by_zero(
            "zero base with negative exponent in power function",
        ));
    }

    let result = base.powf(exponent);
    
    if !result.is_finite() {
        if result.is_infinite() {
            return Err(ProspectTheoryError::numerical_overflow(
                &format!("{}^{}", base, exponent),
            ));
        } else {
            return Err(ProspectTheoryError::numerical_underflow(
                &format!("{}^{}", base, exponent),
            ));
        }
    }

    Ok(result)
}

/// Safe absolute value with overflow protection
pub fn safe_abs(value: f64) -> Result<f64> {
    if !value.is_finite() {
        return Err(ProspectTheoryError::computation_failed(
            "Non-finite input to absolute value",
        ));
    }
    
    Ok(value.abs())
}

/// Safe division with zero check
pub fn safe_div(numerator: f64, denominator: f64) -> Result<f64> {
    if !numerator.is_finite() || !denominator.is_finite() {
        return Err(ProspectTheoryError::computation_failed(
            "Non-finite input to division",
        ));
    }

    if denominator.abs() < FINANCIAL_PRECISION {
        return Err(ProspectTheoryError::division_by_zero(
            "denominator too close to zero",
        ));
    }

    let result = numerator / denominator;
    
    if !result.is_finite() {
        if result.is_infinite() {
            return Err(ProspectTheoryError::numerical_overflow("division"));
        } else {
            return Err(ProspectTheoryError::numerical_underflow("division"));
        }
    }

    Ok(result)
}

/// Clamp value to safe range
pub fn clamp_safe(value: f64, min: f64, max: f64) -> f64 {
    if !value.is_finite() {
        return 0.0;
    }
    value.max(min).min(max)
}

/// Thread-safe calculation wrapper
pub fn thread_safe_calc<T, F>(calculation: F) -> Result<T>
where
    F: FnOnce() -> Result<T> + Send,
    T: Send,
{
    // In a real implementation, you might use thread-local storage
    // or other synchronization primitives here
    calculation()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_approx_equal() {
        assert!(approx_equal(1.0, 1.0 + FINANCIAL_PRECISION / 2.0));
        assert!(!approx_equal(1.0, 1.0 + FINANCIAL_PRECISION * 2.0));
    }

    #[test]
    fn test_safe_pow() {
        assert!(safe_pow(2.0, 3.0).unwrap() - 8.0 < FINANCIAL_PRECISION);
        assert!(safe_pow(0.0, -1.0).is_err());
        assert!(safe_pow(f64::NAN, 1.0).is_err());
    }

    #[test]
    fn test_safe_div() {
        assert!(safe_div(6.0, 2.0).unwrap() - 3.0 < FINANCIAL_PRECISION);
        assert!(safe_div(1.0, 0.0).is_err());
        assert!(safe_div(1.0, FINANCIAL_PRECISION / 2.0).is_err());
    }
}