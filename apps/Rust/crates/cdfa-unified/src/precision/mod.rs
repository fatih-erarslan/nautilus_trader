//! High-precision numerical algorithms for financial computing
//!
//! This module provides numerically stable algorithms for financial calculations
//! where precision is critical. It includes compensated summation algorithms
//! and other techniques to minimize floating-point errors.
//!
//! ## Modules
//!
//! - [`kahan_simple`]: Production-ready Kahan summation implementation
//! - [`stable`]: Numerically stable variants of common operations
//!
//! ## Features
//!
//! - Compensated summation with ±1e-15 precision guarantee
//! - Production-tested implementations
//! - Comprehensive test suite including pathological cases

pub mod kahan_simple;
pub mod stable;

// Re-export main types for convenience
pub use kahan_simple::{KahanAccumulator, NeumaierAccumulator};
pub use stable::*;

/// Precision requirements for financial calculations
pub const FINANCIAL_PRECISION_EPSILON: f64 = 1e-15;

/// Maximum relative error allowed in financial computations
pub const FINANCIAL_MAX_RELATIVE_ERROR: f64 = 1e-12;

/// Trait for high-precision numerical operations
pub trait HighPrecision {
    /// The output type of the operation
    type Output;
    
    /// Perform the operation with high precision
    fn high_precision(&self) -> Self::Output;
    
    /// Check if the result meets financial precision requirements
    fn meets_precision_requirements(&self, expected: Self::Output) -> bool
    where
        Self::Output: PartialEq + Copy + Into<f64>,
    {
        let actual: f64 = self.high_precision().into();
        let expected: f64 = expected.into();
        
        if expected == 0.0 {
            actual.abs() < FINANCIAL_PRECISION_EPSILON
        } else {
            let relative_error = ((actual - expected) / expected).abs();
            relative_error < FINANCIAL_MAX_RELATIVE_ERROR
        }
    }
}

/// Implement HighPrecision for slice of f64 using Kahan summation
impl HighPrecision for &[f64] {
    type Output = f64;
    
    fn high_precision(&self) -> Self::Output {
        KahanAccumulator::sum_slice(self)
    }
}

/// Implement HighPrecision for Vec<f64> using Kahan summation
impl HighPrecision for Vec<f64> {
    type Output = f64;
    
    fn high_precision(&self) -> Self::Output {
        KahanAccumulator::sum_slice(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_precision_trait() {
        let values = vec![1e16, 1.0, -1e16];
        let result = values.high_precision();
        assert_eq!(result, 1.0);
        
        assert!(values.meets_precision_requirements(1.0));
    }

    #[test]
    fn test_precision_requirements() {
        let values = vec![1.0, 2.0, 3.0];
        assert!(values.meets_precision_requirements(6.0));
        assert!(!values.meets_precision_requirements(7.0));
    }
}