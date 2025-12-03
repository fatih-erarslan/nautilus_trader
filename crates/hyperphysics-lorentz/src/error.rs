//! Error types for Lorentz model operations

use thiserror::Error;

/// Result type for Lorentz operations
pub type Result<T> = std::result::Result<T, LorentzError>;

/// Errors that can occur in Lorentz model operations
#[derive(Debug, Error, Clone)]
pub enum LorentzError {
    /// Point is not on the hyperboloid
    #[error("Point is not on hyperboloid: Minkowski norm = {norm}, expected = {expected}")]
    NotOnHyperboloid {
        /// Actual Minkowski norm
        norm: f64,
        /// Expected Minkowski norm (-1/K)
        expected: f64,
    },

    /// Curvature must be negative
    #[error("Curvature must be negative, got {0}")]
    InvalidCurvature(f64),

    /// Dimension mismatch
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch {
        /// Expected dimension
        expected: usize,
        /// Actual dimension
        actual: usize,
    },

    /// Time coordinate must be positive
    #[error("Time coordinate must be positive for upper hyperboloid sheet, got {0}")]
    NegativeTimeCoordinate(f64),

    /// Point at infinity (boundary)
    #[error("Point at infinity (on boundary of Poincaré ball)")]
    PointAtInfinity,

    /// Numerical instability detected
    #[error("Numerical instability: {0}")]
    NumericalInstability(String),

    /// Empty input
    #[error("Empty input: {0}")]
    EmptyInput(String),
}
