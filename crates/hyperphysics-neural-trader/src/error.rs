//! Error types for Neural Trader integration.

use thiserror::Error;

/// Result type alias using NeuralBridgeError
pub type Result<T> = std::result::Result<T, NeuralBridgeError>;

/// Errors that can occur during Neural Trader integration
#[derive(Error, Debug)]
pub enum NeuralBridgeError {
    /// Insufficient data for neural forecasting
    #[error("Insufficient data: need at least {required} samples, got {actual}")]
    InsufficientData {
        /// Required number of samples
        required: usize,
        /// Actual number of samples
        actual: usize,
    },

    /// Neural model inference error
    #[error("Neural inference error: {0}")]
    Inference(String),

    /// Model loading error
    #[error("Model loading error: {0}")]
    ModelLoad(String),

    /// Configuration error
    #[error("Configuration error: {0}")]
    Configuration(String),

    /// Feature extraction error
    #[error("Feature extraction error: {0}")]
    FeatureExtraction(String),

    /// Ensemble aggregation error
    #[error("Ensemble aggregation error: {0}")]
    EnsembleAggregation(String),

    /// GPU/Device error
    #[error("Device error: {0}")]
    Device(String),

    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// I/O error
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Generic error wrapper
    #[error("{0}")]
    Other(String),
}

impl From<anyhow::Error> for NeuralBridgeError {
    fn from(err: anyhow::Error) -> Self {
        NeuralBridgeError::Other(err.to_string())
    }
}
