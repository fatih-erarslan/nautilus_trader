//! Error types for cognition system

use thiserror::Error;

/// Cognition system errors
#[derive(Error, Debug)]
pub enum CognitionError {
    /// Configuration error
    #[error("Configuration error: {0}")]
    Config(String),

    /// Attention mechanism error
    #[error("Attention error: {0}")]
    Attention(String),

    /// Self-referential loop error
    #[error("Loop error: {0}")]
    Loop(String),

    /// Dream state error
    #[error("Dream error: {0}")]
    Dream(String),

    /// Learning error
    #[error("Learning error: {0}")]
    Learning(String),

    /// Integration error
    #[error("Integration error: {0}")]
    Integration(String),

    /// Cortical bus error
    #[error("Cortical bus error: {0}")]
    CorticalBus(String),

    /// Invalid curvature
    #[error("Invalid curvature: {0} (must be in range [{1}, {2}])")]
    InvalidCurvature(f64, f64, f64),

    /// Invalid dimension
    #[error("Invalid dimension: expected {expected}, got {actual}")]
    InvalidDimension { expected: usize, actual: usize },

    /// Synchronization error
    #[error("Synchronization error: {0}")]
    Sync(String),

    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(String),

    /// Action not implemented
    #[error("Action not implemented: {0}")]
    ActionNotImplemented(String),
}

/// Result type for cognition operations
pub type Result<T> = std::result::Result<T, CognitionError>;
