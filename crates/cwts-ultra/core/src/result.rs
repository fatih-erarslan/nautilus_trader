//! Result type definitions for CWTS Ultra
//!
//! Standardized Result<T, E> types replacing all panic-prone operations

use crate::error::CwtsError;

/// Standard result type for CWTS Ultra operations
pub type CwtsResult<T> = Result<T, CwtsError>;

/// Helper trait for result chaining and error context
pub trait CwtsResultExt<T> {
    /// Add context to an error
    fn with_context<F>(self, f: F) -> CwtsResult<T>
    where
        F: FnOnce() -> String;

    /// Map error with additional context
    fn map_context(self, context: &str) -> CwtsResult<T>;

    /// Log error and continue with default value
    fn or_log_default(self, operation: &str) -> T;

    /// Convert to Option, logging error
    fn ok_or_log(self, operation: &str) -> Option<T>;
}

impl<T: Default> CwtsResultExt<T> for CwtsResult<T> {
    fn with_context<F>(self, f: F) -> CwtsResult<T>
    where
        F: FnOnce() -> String,
    {
        self.map_err(|e| CwtsError::operation_failed(f(), e.to_string()))
    }

    fn map_context(self, context: &str) -> CwtsResult<T> {
        self.map_err(|e| CwtsError::operation_failed(context, e.to_string()))
    }

    fn or_log_default(self, operation: &str) -> T {
        match self {
            Ok(value) => value,
            Err(e) => {
                tracing::error!("Operation '{}' failed: {}", operation, e);
                T::default()
            }
        }
    }

    fn ok_or_log(self, operation: &str) -> Option<T> {
        match self {
            Ok(value) => Some(value),
            Err(e) => {
                tracing::warn!("Operation '{}' failed: {}", operation, e);
                None
            }
        }
    }
}

/// Helper for safe arithmetic operations
pub trait SafeArithmetic<T> {
    type Output;
    fn safe_add(self, other: T) -> CwtsResult<Self::Output>;
    fn safe_sub(self, other: T) -> CwtsResult<Self::Output>;
    fn safe_mul(self, other: T) -> CwtsResult<Self::Output>;
    fn safe_div(self, other: T) -> CwtsResult<Self::Output>;
}

impl SafeArithmetic<u64> for u64 {
    type Output = u64;

    fn safe_add(self, other: u64) -> CwtsResult<u64> {
        self.checked_add(other).ok_or_else(|| {
            CwtsError::validation(format!(
                "Integer overflow in addition: {} + {}",
                self, other
            ))
        })
    }

    fn safe_sub(self, other: u64) -> CwtsResult<u64> {
        self.checked_sub(other).ok_or_else(|| {
            CwtsError::validation(format!(
                "Integer underflow in subtraction: {} - {}",
                self, other
            ))
        })
    }

    fn safe_mul(self, other: u64) -> CwtsResult<u64> {
        self.checked_mul(other).ok_or_else(|| {
            CwtsError::validation(format!(
                "Integer overflow in multiplication: {} * {}",
                self, other
            ))
        })
    }

    fn safe_div(self, other: u64) -> CwtsResult<u64> {
        if other == 0 {
            return Err(CwtsError::validation("Division by zero"));
        }
        Ok(self / other)
    }
}

impl SafeArithmetic<f64> for f64 {
    type Output = f64;

    fn safe_add(self, other: f64) -> CwtsResult<f64> {
        let result = self + other;
        if result.is_finite() {
            Ok(result)
        } else {
            Err(CwtsError::validation(format!(
                "Non-finite result in addition: {} + {} = {}",
                self, other, result
            )))
        }
    }

    fn safe_sub(self, other: f64) -> CwtsResult<f64> {
        let result = self - other;
        if result.is_finite() {
            Ok(result)
        } else {
            Err(CwtsError::validation(format!(
                "Non-finite result in subtraction: {} - {} = {}",
                self, other, result
            )))
        }
    }

    fn safe_mul(self, other: f64) -> CwtsResult<f64> {
        let result = self * other;
        if result.is_finite() {
            Ok(result)
        } else {
            Err(CwtsError::validation(format!(
                "Non-finite result in multiplication: {} * {} = {}",
                self, other, result
            )))
        }
    }

    fn safe_div(self, other: f64) -> CwtsResult<f64> {
        if other == 0.0 {
            return Err(CwtsError::validation("Division by zero"));
        }
        let result = self / other;
        if result.is_finite() {
            Ok(result)
        } else {
            Err(CwtsError::validation(format!(
                "Non-finite result in division: {} / {} = {}",
                self, other, result
            )))
        }
    }
}

/// Safe collection access
pub trait SafeAccess<T> {
    fn safe_get(&self, index: usize) -> CwtsResult<&T>;
    fn safe_get_mut(&mut self, index: usize) -> CwtsResult<&mut T>;
}

impl<T> SafeAccess<T> for Vec<T> {
    fn safe_get(&self, index: usize) -> CwtsResult<&T> {
        self.get(index).ok_or_else(|| {
            CwtsError::validation(format!(
                "Index {} out of bounds for vector of length {}",
                index,
                self.len()
            ))
        })
    }

    fn safe_get_mut(&mut self, index: usize) -> CwtsResult<&mut T> {
        let len = self.len(); // Capture length before mutable borrow
        self.get_mut(index).ok_or_else(|| {
            CwtsError::validation(format!(
                "Index {} out of bounds for vector of length {}",
                index, len
            ))
        })
    }
}

impl<T> SafeAccess<T> for [T] {
    fn safe_get(&self, index: usize) -> CwtsResult<&T> {
        self.get(index).ok_or_else(|| {
            CwtsError::validation(format!(
                "Index {} out of bounds for slice of length {}",
                index,
                self.len()
            ))
        })
    }

    fn safe_get_mut(&mut self, index: usize) -> CwtsResult<&mut T> {
        let len = self.len(); // Capture length before mutable borrow
        self.get_mut(index).ok_or_else(|| {
            CwtsError::validation(format!(
                "Index {} out of bounds for slice of length {}",
                index, len
            ))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::CwtsError;

    #[test]
    fn test_result_extensions() {
        let result: CwtsResult<i32> = Err(CwtsError::validation("test error"));
        let contextual = result.map_context("test operation");
        assert!(contextual.is_err());
    }

    #[test]
    fn test_safe_arithmetic() {
        // Test safe addition
        let result = 10u64.safe_add(20u64);
        assert_eq!(result.unwrap(), 30u64);

        // Test overflow
        let overflow_result = u64::MAX.safe_add(1u64);
        assert!(overflow_result.is_err());

        // Test division by zero
        let div_zero = 10u64.safe_div(0u64);
        assert!(div_zero.is_err());
    }

    #[test]
    fn test_safe_access() {
        let vec = vec![1, 2, 3];
        assert_eq!(vec.safe_get(1).unwrap(), &2);
        assert!(vec.safe_get(10).is_err());
    }

    #[test]
    fn test_float_arithmetic() {
        let result = 1.0f64.safe_div(0.0f64);
        assert!(result.is_err());

        let valid = 10.0f64.safe_div(2.0f64);
        assert_eq!(valid.unwrap(), 5.0f64);
    }
}
