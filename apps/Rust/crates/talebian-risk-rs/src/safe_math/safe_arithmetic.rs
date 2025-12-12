//! Safe arithmetic operations with overflow protection and NaN/Infinity detection

use super::{constants::*, SafeMathResult, EPSILON, MAX_SAFE_VALUE, MIN_SAFE_VALUE};
use crate::error::TalebianError;

/// Safe division with comprehensive error handling
pub fn safe_divide(numerator: f64, denominator: f64) -> SafeMathResult<f64> {
    // Check for NaN or infinite inputs
    if !numerator.is_finite() {
        return Err(TalebianError::math(format!(
            "Invalid numerator: {}", numerator
        )));
    }
    
    if !denominator.is_finite() {
        return Err(TalebianError::math(format!(
            "Invalid denominator: {}", denominator
        )));
    }
    
    // Check for division by zero with epsilon tolerance
    if denominator.abs() < SAFE_DIVISION_MIN {
        return Err(TalebianError::math(format!(
            "Division by zero or near-zero: denominator = {}", denominator
        )));
    }
    
    let result = numerator / denominator;
    
    // Validate result
    if !result.is_finite() {
        return Err(TalebianError::math(format!(
            "Division resulted in invalid value: {} / {} = {}", 
            numerator, denominator, result
        )));
    }
    
    // Check for overflow
    if result.abs() > MAX_SAFE_VALUE {
        return Err(TalebianError::math(format!(
            "Division result overflow: {}", result
        )));
    }
    
    Ok(result)
}

/// Safe division with fallback value
pub fn safe_divide_with_fallback(numerator: f64, denominator: f64, fallback: f64) -> f64 {
    match safe_divide(numerator, denominator) {
        Ok(result) => result,
        Err(_) => fallback,
    }
}

/// Safe multiplication with overflow checking
pub fn safe_multiply(a: f64, b: f64) -> SafeMathResult<f64> {
    if !a.is_finite() || !b.is_finite() {
        return Err(TalebianError::math(format!(
            "Invalid inputs for multiplication: {} * {}", a, b
        )));
    }
    
    // Check for potential overflow before multiplication
    if a.abs() > 1.0 && b.abs() > MAX_SAFE_VALUE / a.abs() {
        return Err(TalebianError::math(format!(
            "Multiplication would overflow: {} * {}", a, b
        )));
    }
    
    let result = a * b;
    
    if !result.is_finite() {
        return Err(TalebianError::math(format!(
            "Multiplication resulted in invalid value: {}", result
        )));
    }
    
    Ok(result)
}

/// Safe addition with overflow checking
pub fn safe_add(a: f64, b: f64) -> SafeMathResult<f64> {
    if !a.is_finite() || !b.is_finite() {
        return Err(TalebianError::math(format!(
            "Invalid inputs for addition: {} + {}", a, b
        )));
    }
    
    let result = a + b;
    
    if !result.is_finite() {
        return Err(TalebianError::math(format!(
            "Addition resulted in invalid value: {}", result
        )));
    }
    
    if result.abs() > MAX_SAFE_VALUE {
        return Err(TalebianError::math(format!(
            "Addition result overflow: {}", result
        )));
    }
    
    Ok(result)
}

/// Safe subtraction with overflow checking
pub fn safe_subtract(a: f64, b: f64) -> SafeMathResult<f64> {
    if !a.is_finite() || !b.is_finite() {
        return Err(TalebianError::math(format!(
            "Invalid inputs for subtraction: {} - {}", a, b
        )));
    }
    
    let result = a - b;
    
    if !result.is_finite() {
        return Err(TalebianError::math(format!(
            "Subtraction resulted in invalid value: {}", result
        )));
    }
    
    if result.abs() > MAX_SAFE_VALUE {
        return Err(TalebianError::math(format!(
            "Subtraction result overflow: {}", result
        )));
    }
    
    Ok(result)
}

/// Safe power operation with overflow protection
pub fn safe_pow(base: f64, exponent: f64) -> SafeMathResult<f64> {
    if !base.is_finite() || !exponent.is_finite() {
        return Err(TalebianError::math(format!(
            "Invalid inputs for power: {}^{}", base, exponent
        )));
    }
    
    // Special cases
    if exponent == 0.0 {
        return Ok(1.0);
    }
    
    if base == 0.0 {
        if exponent > 0.0 {
            return Ok(0.0);
        } else {
            return Err(TalebianError::math(
                "Cannot raise zero to negative power".to_string()
            ));
        }
    }
    
    // Check for potential overflow
    if base.abs() > 1.0 && exponent > 0.0 {
        let log_max = MAX_SAFE_VALUE.ln();
        let log_base = base.abs().ln();
        if exponent * log_base > log_max {
            return Err(TalebianError::math(format!(
                "Power operation would overflow: {}^{}", base, exponent
            )));
        }
    }
    
    let result = base.powf(exponent);
    
    if !result.is_finite() {
        return Err(TalebianError::math(format!(
            "Power operation resulted in invalid value: {}^{} = {}", 
            base, exponent, result
        )));
    }
    
    Ok(result)
}

/// Safe logarithm with domain checking
pub fn safe_ln(value: f64) -> SafeMathResult<f64> {
    if !value.is_finite() {
        return Err(TalebianError::math(format!(
            "Invalid input for logarithm: {}", value
        )));
    }
    
    if value <= 0.0 {
        return Err(TalebianError::math(format!(
            "Logarithm of non-positive value: {}", value
        )));
    }
    
    let result = value.ln();
    
    if !result.is_finite() {
        return Err(TalebianError::math(format!(
            "Logarithm resulted in invalid value: ln({}) = {}", value, result
        )));
    }
    
    Ok(result)
}

/// Safe square root with domain checking
pub fn safe_sqrt(value: f64) -> SafeMathResult<f64> {
    if !value.is_finite() {
        return Err(TalebianError::math(format!(
            "Invalid input for square root: {}", value
        )));
    }
    
    if value < 0.0 {
        return Err(TalebianError::math(format!(
            "Square root of negative value: {}", value
        )));
    }
    
    let result = value.sqrt();
    
    if !result.is_finite() {
        return Err(TalebianError::math(format!(
            "Square root resulted in invalid value: sqrt({}) = {}", value, result
        )));
    }
    
    Ok(result)
}

/// Safe percentage calculation
pub fn safe_percentage(part: f64, whole: f64) -> SafeMathResult<f64> {
    if whole.abs() < SAFE_DIVISION_MIN {
        return Err(TalebianError::math(
            "Cannot calculate percentage with zero or near-zero denominator".to_string()
        ));
    }
    
    let ratio = safe_divide(part, whole)?;
    safe_multiply(ratio, 100.0)
}

/// Safe percentage change calculation
pub fn safe_percentage_change(old_value: f64, new_value: f64) -> SafeMathResult<f64> {
    if old_value.abs() < SAFE_DIVISION_MIN {
        return Err(TalebianError::math(
            "Cannot calculate percentage change from zero or near-zero base".to_string()
        ));
    }
    
    let difference = safe_subtract(new_value, old_value)?;
    safe_percentage(difference, old_value)
}

/// Check if two floats are approximately equal
pub fn approximately_equal(a: f64, b: f64) -> bool {
    (a - b).abs() < EPSILON
}

/// Check if two floats are approximately equal with custom tolerance
pub fn approximately_equal_with_tolerance(a: f64, b: f64, tolerance: f64) -> bool {
    (a - b).abs() < tolerance
}

/// Clamp value to safe range
pub fn clamp_to_safe_range(value: f64) -> f64 {
    if !value.is_finite() {
        return 0.0;
    }
    
    value.max(MIN_SAFE_VALUE).min(MAX_SAFE_VALUE)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_safe_divide() {
        assert!(safe_divide(10.0, 2.0).is_ok());
        assert_eq!(safe_divide(10.0, 2.0).unwrap(), 5.0);
        
        assert!(safe_divide(10.0, 0.0).is_err());
        assert!(safe_divide(10.0, 1e-20).is_err());
        assert!(safe_divide(f64::NAN, 2.0).is_err());
        assert!(safe_divide(10.0, f64::NAN).is_err());
        assert!(safe_divide(f64::INFINITY, 2.0).is_err());
    }

    #[test]
    fn test_safe_divide_with_fallback() {
        assert_eq!(safe_divide_with_fallback(10.0, 2.0, 0.0), 5.0);
        assert_eq!(safe_divide_with_fallback(10.0, 0.0, -1.0), -1.0);
        assert_eq!(safe_divide_with_fallback(f64::NAN, 2.0, 0.0), 0.0);
    }

    #[test]
    fn test_safe_multiply() {
        assert!(safe_multiply(10.0, 2.0).is_ok());
        assert_eq!(safe_multiply(10.0, 2.0).unwrap(), 20.0);
        
        assert!(safe_multiply(f64::NAN, 2.0).is_err());
        assert!(safe_multiply(1e10, 1e10).is_err()); // Overflow
    }

    #[test]
    fn test_safe_add() {
        assert!(safe_add(10.0, 2.0).is_ok());
        assert_eq!(safe_add(10.0, 2.0).unwrap(), 12.0);
        
        assert!(safe_add(f64::NAN, 2.0).is_err());
        assert!(safe_add(f64::INFINITY, 2.0).is_err());
    }

    #[test]
    fn test_safe_subtract() {
        assert!(safe_subtract(10.0, 2.0).is_ok());
        assert_eq!(safe_subtract(10.0, 2.0).unwrap(), 8.0);
        
        assert!(safe_subtract(f64::NAN, 2.0).is_err());
        assert!(safe_subtract(f64::INFINITY, 2.0).is_err());
    }

    #[test]
    fn test_safe_pow() {
        assert!(safe_pow(2.0, 3.0).is_ok());
        assert_eq!(safe_pow(2.0, 3.0).unwrap(), 8.0);
        
        assert!(safe_pow(0.0, -1.0).is_err());
        assert!(safe_pow(f64::NAN, 2.0).is_err());
        assert!(safe_pow(1e6, 1e6).is_err()); // Overflow
    }

    #[test]
    fn test_safe_ln() {
        assert!(safe_ln(2.718281828).is_ok());
        assert!((safe_ln(2.718281828).unwrap() - 1.0).abs() < 1e-8);
        
        assert!(safe_ln(0.0).is_err());
        assert!(safe_ln(-1.0).is_err());
        assert!(safe_ln(f64::NAN).is_err());
    }

    #[test]
    fn test_safe_sqrt() {
        assert!(safe_sqrt(4.0).is_ok());
        assert_eq!(safe_sqrt(4.0).unwrap(), 2.0);
        
        assert!(safe_sqrt(-1.0).is_err());
        assert!(safe_sqrt(f64::NAN).is_err());
    }

    #[test]
    fn test_safe_percentage() {
        assert!(safe_percentage(50.0, 200.0).is_ok());
        assert_eq!(safe_percentage(50.0, 200.0).unwrap(), 25.0);
        
        assert!(safe_percentage(50.0, 0.0).is_err());
    }

    #[test]
    fn test_safe_percentage_change() {
        assert!(safe_percentage_change(100.0, 110.0).is_ok());
        assert_eq!(safe_percentage_change(100.0, 110.0).unwrap(), 10.0);
        
        assert!(safe_percentage_change(0.0, 10.0).is_err());
    }

    #[test]
    fn test_approximately_equal() {
        assert!(approximately_equal(1.0, 1.0 + 1e-16));
        assert!(!approximately_equal(1.0, 1.1));
    }

    #[test]
    fn test_clamp_to_safe_range() {
        assert_eq!(clamp_to_safe_range(500.0), 500.0);
        assert_eq!(clamp_to_safe_range(f64::NAN), 0.0);
        assert_eq!(clamp_to_safe_range(f64::INFINITY), MAX_SAFE_VALUE);
        assert_eq!(clamp_to_safe_range(1e-20), MIN_SAFE_VALUE);
    }
}