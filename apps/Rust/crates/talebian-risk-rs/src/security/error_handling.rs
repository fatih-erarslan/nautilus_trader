//! Enhanced error handling for financial operations
//!
//! This module provides production-grade error handling patterns that ensure
//! financial operations never panic and always provide meaningful error context.

use crate::TalebianRiskError;
use std::fmt;

/// Financial error types with specific context
#[derive(Debug, Clone)]
pub enum FinancialError {
    /// Mathematical calculation error
    Calculation { operation: String, context: String },
    /// Input validation error
    Validation { field: String, value: String, constraint: String },
    /// Division by zero or near-zero
    DivisionByZero { numerator: f64, denominator: f64 },
    /// Overflow in financial calculation
    Overflow { operation: String, values: Vec<f64> },
    /// Position sizing error
    PositionSizing { requested: f64, max_allowed: f64, reason: String },
    /// Market data integrity error
    DataIntegrity { field: String, issue: String },
}

impl fmt::Display for FinancialError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FinancialError::Calculation { operation, context } => {
                write!(f, "Calculation error in {}: {}", operation, context)
            }
            FinancialError::Validation { field, value, constraint } => {
                write!(f, "Validation error for {}: value '{}' violates constraint '{}'", 
                       field, value, constraint)
            }
            FinancialError::DivisionByZero { numerator, denominator } => {
                write!(f, "Division by zero: {} / {} (denominator too close to zero)", 
                       numerator, denominator)
            }
            FinancialError::Overflow { operation, values } => {
                write!(f, "Overflow in {}: values {:?} exceed safe range", operation, values)
            }
            FinancialError::PositionSizing { requested, max_allowed, reason } => {
                write!(f, "Position sizing error: requested {} exceeds maximum {} ({})", 
                       requested, max_allowed, reason)
            }
            FinancialError::DataIntegrity { field, issue } => {
                write!(f, "Data integrity error in {}: {}", field, issue)
            }
        }
    }
}

impl std::error::Error for FinancialError {}

/// Security result type for financial operations
pub type SecurityResult<T> = Result<T, FinancialError>;

/// Convert FinancialError to TalebianRiskError
impl From<FinancialError> for TalebianRiskError {
    fn from(err: FinancialError) -> Self {
        match err {
            FinancialError::Calculation { operation, context } => {
                TalebianRiskError::CalculationError(format!("{}: {}", operation, context))
            }
            FinancialError::Validation { field, value, constraint } => {
                TalebianRiskError::InvalidInput(format!(
                    "Validation failed for {}: '{}' violates '{}'", field, value, constraint
                ))
            }
            FinancialError::DivisionByZero { numerator, denominator } => {
                TalebianRiskError::CalculationError(format!(
                    "Division by zero: {} / {}", numerator, denominator
                ))
            }
            FinancialError::Overflow { operation, values } => {
                TalebianRiskError::CalculationError(format!(
                    "Overflow in {}: values {:?}", operation, values
                ))
            }
            FinancialError::PositionSizing { requested, max_allowed, reason } => {
                TalebianRiskError::InvalidInput(format!(
                    "Position size {} exceeds maximum {} ({})", requested, max_allowed, reason
                ))
            }
            FinancialError::DataIntegrity { field, issue } => {
                TalebianRiskError::MarketDataError(format!("Data integrity error in {}: {}", field, issue))
            }
        }
    }
}

/// Safe unwrap alternative that provides context
pub trait SafeUnwrap<T> {
    fn safe_unwrap(self, context: &str) -> Result<T, TalebianRiskError>;
    fn safe_unwrap_or_default(self, default: T) -> T;
}

impl<T> SafeUnwrap<T> for Option<T> {
    fn safe_unwrap(self, context: &str) -> Result<T, TalebianRiskError> {
        self.ok_or_else(|| TalebianRiskError::CalculationError(format!("None value in {}", context)))
    }
    
    fn safe_unwrap_or_default(self, default: T) -> T {
        self.unwrap_or(default)
    }
}

impl<T, E> SafeUnwrap<T> for Result<T, E> 
where 
    E: std::error::Error + Send + Sync + 'static
{
    fn safe_unwrap(self, context: &str) -> Result<T, TalebianRiskError> {
        self.map_err(|e| TalebianRiskError::CalculationError(format!("{}: {}", context, e)))
    }
    
    fn safe_unwrap_or_default(self, default: T) -> T {
        self.unwrap_or(default)
    }
}

/// Create a financial calculation error
pub fn calculation_error(operation: &str, context: &str) -> FinancialError {
    FinancialError::Calculation {
        operation: operation.to_string(),
        context: context.to_string(),
    }
}

/// Create a validation error
pub fn validation_error(field: &str, value: &str, constraint: &str) -> FinancialError {
    FinancialError::Validation {
        field: field.to_string(),
        value: value.to_string(),
        constraint: constraint.to_string(),
    }
}

/// Create a division by zero error
pub fn division_by_zero_error(numerator: f64, denominator: f64) -> FinancialError {
    FinancialError::DivisionByZero { numerator, denominator }
}

/// Create an overflow error
pub fn overflow_error(operation: &str, values: Vec<f64>) -> FinancialError {
    FinancialError::Overflow {
        operation: operation.to_string(),
        values,
    }
}

/// Create a position sizing error
pub fn position_sizing_error(requested: f64, max_allowed: f64, reason: &str) -> FinancialError {
    FinancialError::PositionSizing {
        requested,
        max_allowed,
        reason: reason.to_string(),
    }
}

/// Create a data integrity error
pub fn data_integrity_error(field: &str, issue: &str) -> FinancialError {
    FinancialError::DataIntegrity {
        field: field.to_string(),
        issue: issue.to_string(),
    }
}

/// Macro for safe operations with automatic error context
#[macro_export]
macro_rules! safe_op {
    ($op:expr, $context:expr) => {
        $op.map_err(|e| crate::TalebianRiskError::CalculationError(format!("{}: {}", $context, e)))
    };
}

/// Macro for safe array access
#[macro_export]
macro_rules! safe_get {
    ($array:expr, $index:expr, $context:expr) => {
        $array.get($index).ok_or_else(|| {
            crate::TalebianRiskError::InvalidInput(format!(
                "Index {} out of bounds in {} (length {})", 
                $index, $context, $array.len()
            ))
        })
    };
}

/// Macro for safe first/last access
#[macro_export]
macro_rules! safe_first {
    ($array:expr, $context:expr) => {
        $array.first().ok_or_else(|| {
            crate::TalebianRiskError::InvalidInput(format!("{} array is empty", $context))
        })
    };
}

#[macro_export]
macro_rules! safe_last {
    ($array:expr, $context:expr) => {
        $array.last().ok_or_else(|| {
            crate::TalebianRiskError::InvalidInput(format!("{} array is empty", $context))
        })
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_financial_error_display() {
        let err = FinancialError::Calculation {
            operation: "division".to_string(),
            context: "Kelly calculation".to_string(),
        };
        
        let display = format!("{}", err);
        assert!(display.contains("division"));
        assert!(display.contains("Kelly calculation"));
    }

    #[test]
    fn test_safe_unwrap_option_some() {
        let value = Some(42);
        let result = value.safe_unwrap("test context").unwrap();
        assert_eq!(result, 42);
    }

    #[test]
    fn test_safe_unwrap_option_none() {
        let value: Option<i32> = None;
        let result = value.safe_unwrap("test context");
        assert!(result.is_err());
    }

    #[test]
    fn test_safe_unwrap_or_default() {
        let value: Option<i32> = None;
        let result = value.safe_unwrap_or_default(42);
        assert_eq!(result, 42);
    }

    #[test]
    fn test_error_conversion() {
        let fin_err = calculation_error("test_op", "test context");
        let tal_err: TalebianRiskError = fin_err.into();
        
        // Just verify it converts without panicking
        assert!(!tal_err.to_string().is_empty());
    }
}