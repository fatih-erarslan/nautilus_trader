//! Error types for hyperphysics-vector

use thiserror::Error;

/// Vector store error types
#[derive(Error, Debug)]
pub enum VectorError {
    /// Dimension mismatch error
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch {
        /// Expected dimension
        expected: usize,
        /// Actual dimension
        actual: usize,
    },

    /// Vector not found
    #[error("Vector not found: {0}")]
    NotFound(String),

    /// Storage error
    #[error("Storage error: {0}")]
    Storage(String),

    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(String),

    /// Configuration error
    #[error("Configuration error: {0}")]
    Configuration(String),

    /// Search error
    #[error("Search error: {0}")]
    Search(String),

    /// Embedding generation error
    #[error("Embedding generation error: {0}")]
    Embedding(String),

    /// Underlying ruvector error
    #[error("Ruvector error: {0}")]
    Ruvector(#[from] ruvector_core::RuvectorError),
}

/// Result type alias for hyperphysics-vector
pub type Result<T> = std::result::Result<T, VectorError>;
