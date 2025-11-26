//! Error types for neuro-divergent

use thiserror::Error;

/// Result type alias for neuro-divergent operations
pub type Result<T> = std::result::Result<T, NeuroDivergentError>;

/// Error types for neural forecasting operations
#[derive(Error, Debug)]
pub enum NeuroDivergentError {
    #[error("Model error: {0}")]
    ModelError(String),

    #[error("Training error: {0}")]
    TrainingError(String),

    #[error("Data error: {0}")]
    DataError(String),

    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),

    #[error("Bincode error: {0}")]
    BincodeError(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Invalid shape: expected {expected}, got {actual}")]
    InvalidShape { expected: String, actual: String },

    #[error("Model not trained")]
    ModelNotTrained,

    #[error("Convergence failed: {0}")]
    ConvergenceError(String),

    #[error("Numerical error: {0}")]
    NumericalError(String),

    #[error("Not implemented: {0}")]
    NotImplemented(String),

    #[error("Unknown error: {0}")]
    Unknown(String),
}

impl From<bincode::Error> for NeuroDivergentError {
    fn from(err: bincode::Error) -> Self {
        NeuroDivergentError::BincodeError(err.to_string())
    }
}

impl From<ndarray::ShapeError> for NeuroDivergentError {
    fn from(err: ndarray::ShapeError) -> Self {
        NeuroDivergentError::InvalidShape {
            expected: "compatible shape".to_string(),
            actual: err.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = NeuroDivergentError::ModelError("test error".to_string());
        assert_eq!(err.to_string(), "Model error: test error");
    }

    #[test]
    fn test_error_from_io() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let err: NeuroDivergentError = io_err.into();
        assert!(matches!(err, NeuroDivergentError::IoError(_)));
    }
}
