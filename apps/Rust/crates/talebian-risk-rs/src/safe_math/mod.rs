//! Safe mathematical operations for financial calculations
//!
//! This module provides bulletproof mathematical operations for financial trading systems
//! with zero tolerance for panics, comprehensive edge case handling, and deterministic results.

pub mod safe_arithmetic;
pub mod validation;
pub mod position_sizing;
pub mod financial_metrics;
pub mod error_handling;

pub use safe_arithmetic::*;
pub use validation::*;
pub use position_sizing::*;
pub use financial_metrics::*;
pub use error_handling::*;

use crate::error::TalebianError;

/// Safe mathematical operations result type
pub type SafeMathResult<T> = Result<T, TalebianError>;

/// Epsilon value for float comparisons
pub const EPSILON: f64 = 1e-15;

/// Maximum safe value for financial calculations
pub const MAX_SAFE_VALUE: f64 = 1e12;

/// Minimum safe value for financial calculations
pub const MIN_SAFE_VALUE: f64 = 1e-12;

/// Constants for numerical stability
pub mod constants {
    /// Machine epsilon for f64
    pub const F64_EPSILON: f64 = f64::EPSILON;
    
    /// Square root of machine epsilon
    pub const SQRT_EPSILON: f64 = 1.49011611938477e-8;
    
    /// Cube root of machine epsilon
    pub const CBRT_EPSILON: f64 = 6.05545445239334e-6;
    
    /// Safe minimum value for division
    pub const SAFE_DIVISION_MIN: f64 = 1e-10;
    
    /// Maximum iterations for numerical algorithms
    pub const MAX_ITERATIONS: usize = 1000;
    
    /// Default tolerance for convergence
    pub const DEFAULT_TOLERANCE: f64 = 1e-8;
}

/// Check if a value is safe for financial calculations
#[inline]
pub fn is_safe_value(value: f64) -> bool {
    value.is_finite() && 
    value.abs() >= MIN_SAFE_VALUE && 
    value.abs() <= MAX_SAFE_VALUE
}

/// Check if a value represents a valid price
#[inline]
pub fn is_valid_price(price: f64) -> bool {
    price.is_finite() && price > 0.0 && price <= MAX_SAFE_VALUE
}

/// Check if a value represents a valid volume
#[inline]
pub fn is_valid_volume(volume: f64) -> bool {
    volume.is_finite() && volume >= 0.0 && volume <= MAX_SAFE_VALUE
}

/// Check if a value represents a valid percentage (0-100)
#[inline]
pub fn is_valid_percentage(value: f64) -> bool {
    value.is_finite() && value >= 0.0 && value <= 100.0
}

/// Check if a value represents a valid ratio (0-1)
#[inline]
pub fn is_valid_ratio(value: f64) -> bool {
    value.is_finite() && value >= 0.0 && value <= 1.0
}

/// Check if a value represents a valid return (-1 to positive infinity)
#[inline]
pub fn is_valid_return(return_value: f64) -> bool {
    return_value.is_finite() && return_value >= -1.0 && return_value <= MAX_SAFE_VALUE
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_safe_value_checks() {
        assert!(is_safe_value(1.0));
        assert!(is_safe_value(-1.0));
        assert!(is_safe_value(1e6));
        assert!(!is_safe_value(f64::NAN));
        assert!(!is_safe_value(f64::INFINITY));
        assert!(!is_safe_value(0.0));
        assert!(!is_safe_value(1e-20));
        assert!(!is_safe_value(1e15));
    }

    #[test]
    fn test_valid_price_checks() {
        assert!(is_valid_price(100.0));
        assert!(is_valid_price(0.01));
        assert!(!is_valid_price(0.0));
        assert!(!is_valid_price(-10.0));
        assert!(!is_valid_price(f64::NAN));
        assert!(!is_valid_price(f64::INFINITY));
    }

    #[test]
    fn test_valid_percentage_checks() {
        assert!(is_valid_percentage(50.0));
        assert!(is_valid_percentage(0.0));
        assert!(is_valid_percentage(100.0));
        assert!(!is_valid_percentage(-1.0));
        assert!(!is_valid_percentage(101.0));
        assert!(!is_valid_percentage(f64::NAN));
    }

    #[test]
    fn test_valid_ratio_checks() {
        assert!(is_valid_ratio(0.5));
        assert!(is_valid_ratio(0.0));
        assert!(is_valid_ratio(1.0));
        assert!(!is_valid_ratio(-0.1));
        assert!(!is_valid_ratio(1.1));
        assert!(!is_valid_ratio(f64::NAN));
    }

    #[test]
    fn test_valid_return_checks() {
        assert!(is_valid_return(0.1));
        assert!(is_valid_return(-0.5));
        assert!(is_valid_return(-1.0));
        assert!(is_valid_return(2.0));
        assert!(!is_valid_return(-1.1));
        assert!(!is_valid_return(f64::NAN));
        assert!(!is_valid_return(f64::INFINITY));
    }
}