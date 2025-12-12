//! Safe mathematical operations for financial calculations
//! 
//! This module provides bulletproof mathematical operations that prevent
//! division by zero, overflow, underflow, and NaN/Infinity issues that
//! could cause financial losses in trading systems.

use crate::TalebianRiskError;
use std::f64;

/// Epsilon for floating point comparisons
pub const EPSILON: f64 = 1e-10;

/// Safe division that handles all edge cases
pub fn safe_divide(numerator: f64, denominator: f64) -> Result<f64, TalebianRiskError> {
    // Check for invalid inputs
    if numerator.is_nan() || numerator.is_infinite() {
        return Err(TalebianRiskError::InvalidInput("Numerator is NaN or infinite".to_string()));
    }
    
    if denominator.is_nan() || denominator.is_infinite() {
        return Err(TalebianRiskError::InvalidInput("Denominator is NaN or infinite".to_string()));
    }
    
    // Check for division by zero (using epsilon for floating point comparison)
    if denominator.abs() < EPSILON {
        return Err(TalebianRiskError::InvalidInput("Division by zero or near-zero value".to_string()));
    }
    
    let result = numerator / denominator;
    
    // Verify result is valid
    if result.is_nan() || result.is_infinite() {
        return Err(TalebianRiskError::InvalidInput("Division result is NaN or infinite".to_string()));
    }
    
    Ok(result)
}

/// Safe multiplication with overflow protection
pub fn safe_multiply(a: f64, b: f64) -> Result<f64, TalebianRiskError> {
    if a.is_nan() || a.is_infinite() || b.is_nan() || b.is_infinite() {
        return Err(TalebianRiskError::InvalidInput("Input is NaN or infinite".to_string()));
    }
    
    let result = a * b;
    
    if result.is_nan() || result.is_infinite() {
        return Err(TalebianRiskError::InvalidInput("Multiplication overflow".to_string().to_string()));
    }
    
    Ok(result)
}

/// Safe addition with overflow protection
pub fn safe_add(a: f64, b: f64) -> Result<f64, TalebianRiskError> {
    if a.is_nan() || a.is_infinite() || b.is_nan() || b.is_infinite() {
        return Err(TalebianRiskError::InvalidInput("Input is NaN or infinite".to_string()));
    }
    
    let result = a + b;
    
    if result.is_nan() || result.is_infinite() {
        return Err(TalebianRiskError::InvalidInput("Addition overflow".to_string().to_string()));
    }
    
    Ok(result)
}

/// Safe subtraction with underflow protection
pub fn safe_subtract(a: f64, b: f64) -> Result<f64, TalebianRiskError> {
    if a.is_nan() || a.is_infinite() || b.is_nan() || b.is_infinite() {
        return Err(TalebianRiskError::InvalidInput("Input is NaN or infinite".to_string()));
    }
    
    let result = a - b;
    
    if result.is_nan() || result.is_infinite() {
        return Err(TalebianRiskError::InvalidInput("Subtraction underflow".to_string().to_string()));
    }
    
    Ok(result)
}

/// Safe percentage calculation
pub fn safe_percentage(value: f64, total: f64) -> Result<f64, TalebianRiskError> {
    let percentage = safe_divide(value, total)?;
    safe_multiply(percentage, 100.0)
}

/// Safe Kelly fraction calculation
pub fn safe_kelly_fraction(
    win_probability: f64,
    win_amount: f64,
    loss_amount: f64,
) -> Result<f64, TalebianRiskError> {
    // Validate inputs
    if win_probability < 0.0 || win_probability > 1.0 {
        return Err(TalebianRiskError::InvalidInput("Win probability must be between 0 and 1".to_string()));
    }
    
    if win_amount <= 0.0 {
        return Err(TalebianRiskError::InvalidInput("Win amount must be positive".to_string()));
    }
    
    if loss_amount <= 0.0 {
        return Err(TalebianRiskError::InvalidInput("Loss amount must be positive".to_string()));
    }
    
    let lose_probability = safe_subtract(1.0, win_probability)?;
    let odds = safe_divide(win_amount, loss_amount)?;
    
    // Kelly formula: f = (bp - q) / b
    // where b = odds, p = win_probability, q = lose_probability
    let numerator = safe_subtract(safe_multiply(win_probability, odds)?, lose_probability)?;
    let kelly = safe_divide(numerator, odds)?;
    
    // Ensure Kelly fraction is reasonable (between 0 and 1)
    if kelly < 0.0 {
        Ok(0.0) // No bet if Kelly is negative
    } else if kelly > 1.0 {
        Ok(1.0) // Cap at 100%
    } else {
        Ok(kelly)
    }
}

/// SafeMath trait for secure mathematical operations
pub trait SafeMath {
    fn safe_div(&self, other: Self) -> Result<Self, TalebianRiskError>
    where
        Self: Sized;
    fn safe_mul(&self, other: Self) -> Result<Self, TalebianRiskError>
    where
        Self: Sized;
    fn safe_add(&self, other: Self) -> Result<Self, TalebianRiskError>
    where
        Self: Sized;
    fn safe_sub(&self, other: Self) -> Result<Self, TalebianRiskError>
    where
        Self: Sized;
}

impl SafeMath for f64 {
    fn safe_div(&self, other: Self) -> Result<Self, TalebianRiskError> {
        safe_divide(*self, other)
    }
    
    fn safe_mul(&self, other: Self) -> Result<Self, TalebianRiskError> {
        safe_multiply(*self, other)
    }
    
    fn safe_add(&self, other: Self) -> Result<Self, TalebianRiskError> {
        safe_add(*self, other)
    }
    
    fn safe_sub(&self, other: Self) -> Result<Self, TalebianRiskError> {
        safe_subtract(*self, other)
    }
}

/// Verify mathematical operation integrity
pub fn verify_math_integrity() -> bool {
    // Test basic operations with known values
    if let Ok(result) = safe_divide(10.0, 2.0) {
        if (result - 5.0).abs() > EPSILON {
            return false;
        }
    } else {
        return false;
    }
    
    // Test division by zero protection
    if safe_divide(10.0, 0.0).is_ok() {
        return false;
    }
    
    // Test NaN protection
    if safe_divide(f64::NAN, 1.0).is_ok() {
        return false;
    }
    
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_safe_divide_normal() {
        let result = safe_divide(10.0, 2.0).unwrap();
        assert!((result - 5.0).abs() < EPSILON);
    }

    #[test]
    fn test_safe_divide_by_zero() {
        assert!(safe_divide(10.0, 0.0).is_err());
    }

    #[test]
    fn test_safe_divide_nan() {
        assert!(safe_divide(f64::NAN, 1.0).is_err());
        assert!(safe_divide(1.0, f64::NAN).is_err());
    }

    #[test]
    fn test_safe_kelly_fraction() {
        let kelly = safe_kelly_fraction(0.6, 2.0, 1.0).unwrap();
        assert!(kelly > 0.0 && kelly <= 1.0);
    }

    #[test]
    fn test_safe_kelly_invalid_probability() {
        assert!(safe_kelly_fraction(1.5, 2.0, 1.0).is_err());
        assert!(safe_kelly_fraction(-0.1, 2.0, 1.0).is_err());
    }

    #[test]
    fn test_math_integrity() {
        assert!(verify_math_integrity());
    }
}