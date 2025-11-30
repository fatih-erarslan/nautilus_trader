//! Error types for optimization algorithms.

use thiserror::Error;

/// Errors that can occur during optimization.
#[derive(Debug, Error)]
pub enum OptimizationError {
    /// Invalid configuration parameter.
    #[error("Configuration error: {0}")]
    Configuration(String),

    /// No solution found.
    #[error("No solution found: {0}")]
    NoSolution(String),

    /// Dimension mismatch.
    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch {
        /// Expected dimension
        expected: usize,
        /// Actual dimension received
        got: usize
    },

    /// Invalid bounds.
    #[error("Invalid bounds: {0}")]
    InvalidBounds(String),

    /// Constraint violation.
    #[error("Constraint violation: {0}")]
    ConstraintViolation(String),

    /// Numerical error (NaN, Inf).
    #[error("Numerical error: {0}")]
    NumericalError(String),

    /// Timeout exceeded.
    #[error("Timeout exceeded after {0} ms")]
    Timeout(u64),

    /// Population error.
    #[error("Population error: {0}")]
    PopulationError(String),

    /// I/O error for serialization.
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),
}

/// Result type alias for optimization operations.
pub type OptimizationResult<T> = Result<T, OptimizationError>;
