//! Error handling for prospect theory calculations

use thiserror::Error;

/// Result type for prospect theory operations
pub type Result<T> = std::result::Result<T, ProspectTheoryError>;

/// Comprehensive error types for prospect theory calculations
#[derive(Error, Debug, Clone, PartialEq)]
pub enum ProspectTheoryError {
    #[error("Invalid parameter: {parameter} = {value}, expected {constraint}")]
    InvalidParameter {
        parameter: String,
        value: f64,
        constraint: String,
    },

    #[error("Input value {value} is out of range [{min}, {max}]")]
    OutOfRange { value: f64, min: f64, max: f64 },

    #[error("Numerical overflow in calculation: {operation}")]
    NumericalOverflow { operation: String },

    #[error("Numerical underflow in calculation: {operation}")]
    NumericalUnderflow { operation: String },

    #[error("Division by zero in calculation: {context}")]
    DivisionByZero { context: String },

    #[error("Invalid probability: {probability}, must be in [0, 1]")]
    InvalidProbability { probability: f64 },

    #[error("Computation failed: {reason}")]
    ComputationFailed { reason: String },

    #[error("Thread safety violation: {details}")]
    ThreadSafetyViolation { details: String },

    #[error("Memory allocation failed: {size} bytes")]
    MemoryAllocationFailed { size: usize },
}

impl ProspectTheoryError {
    /// Create an invalid parameter error
    pub fn invalid_parameter(parameter: &str, value: f64, constraint: &str) -> Self {
        Self::InvalidParameter {
            parameter: parameter.to_string(),
            value,
            constraint: constraint.to_string(),
        }
    }

    /// Create an out of range error
    pub fn out_of_range(value: f64, min: f64, max: f64) -> Self {
        Self::OutOfRange { value, min, max }
    }

    /// Create a numerical overflow error
    pub fn numerical_overflow(operation: &str) -> Self {
        Self::NumericalOverflow {
            operation: operation.to_string(),
        }
    }

    /// Create a numerical underflow error
    pub fn numerical_underflow(operation: &str) -> Self {
        Self::NumericalUnderflow {
            operation: operation.to_string(),
        }
    }

    /// Create a division by zero error
    pub fn division_by_zero(context: &str) -> Self {
        Self::DivisionByZero {
            context: context.to_string(),
        }
    }

    /// Create an invalid probability error
    pub fn invalid_probability(probability: f64) -> Self {
        Self::InvalidProbability { probability }
    }

    /// Create a computation failed error
    pub fn computation_failed(reason: &str) -> Self {
        Self::ComputationFailed {
            reason: reason.to_string(),
        }
    }
}

/// Validate that a value is within financial precision bounds
pub fn validate_financial_bounds(value: f64, name: &str) -> Result<()> {
    if !value.is_finite() {
        return Err(ProspectTheoryError::computation_failed(&format!(
            "{} is not finite: {}",
            name, value
        )));
    }

    if value < crate::MIN_INPUT_VALUE || value > crate::MAX_INPUT_VALUE {
        return Err(ProspectTheoryError::out_of_range(
            value,
            crate::MIN_INPUT_VALUE,
            crate::MAX_INPUT_VALUE,
        ));
    }

    Ok(())
}

/// Validate probability is in [0, 1]
pub fn validate_probability(prob: f64) -> Result<()> {
    if !prob.is_finite() || prob < 0.0 || prob > 1.0 {
        return Err(ProspectTheoryError::invalid_probability(prob));
    }
    Ok(())
}