//! Error types for neural operations

use thiserror::Error;

/// Neural network errors
#[derive(Error, Debug)]
pub enum NeuralError {
    /// Shape mismatch in tensor operations
    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        /// Expected shape
        expected: Vec<usize>,
        /// Actual shape
        actual: Vec<usize>,
    },

    /// Invalid layer configuration
    #[error("Invalid layer configuration: {0}")]
    InvalidLayerConfig(String),

    /// Invalid network architecture
    #[error("Invalid network architecture: {0}")]
    InvalidArchitecture(String),

    /// Training error
    #[error("Training error: {0}")]
    TrainingError(String),

    /// Inference error
    #[error("Inference error: {0}")]
    InferenceError(String),

    /// Weight initialization error
    #[error("Weight initialization failed: {0}")]
    WeightInitError(String),

    /// Serialization error
    #[error("Serialization error: {0}")]
    SerializationError(String),

    /// GPU/Compute error
    #[error("Compute error: {0}")]
    ComputeError(String),

    /// I/O error
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    /// Dimension mismatch
    #[error("Dimension mismatch: input dim {input_dim}, layer expects {expected_dim}")]
    DimensionMismatch {
        /// Input dimension
        input_dim: usize,
        /// Expected dimension
        expected_dim: usize,
    },

    /// Activation error
    #[error("Activation error: {0}")]
    ActivationError(String),

    /// Backend not available
    #[error("Backend not available: {0}")]
    BackendNotAvailable(String),

    /// Timeout exceeded
    #[error("Inference timeout exceeded: {0:?}")]
    Timeout(std::time::Duration),

    /// Invalid input
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// FANN error
    #[error("FANN error: {0}")]
    FannError(String),
}

/// Result type for neural operations
pub type NeuralResult<T> = Result<T, NeuralError>;
