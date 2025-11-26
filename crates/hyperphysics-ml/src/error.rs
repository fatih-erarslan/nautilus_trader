//! Error types for hyperphysics-ml

use thiserror::Error;

/// Result type alias for ML operations
pub type MlResult<T> = Result<T, MlError>;

/// ML framework error types
#[derive(Error, Debug)]
pub enum MlError {
    /// Backend not available on this platform
    #[error("Backend '{backend}' not available: {reason}")]
    BackendUnavailable {
        /// Backend name
        backend: String,
        /// Reason for unavailability
        reason: String,
    },

    /// Tensor shape mismatch
    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        /// Expected shape
        expected: Vec<usize>,
        /// Actual shape
        actual: Vec<usize>,
    },

    /// Invalid model configuration
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// Model not initialized
    #[error("Model not initialized: {0}")]
    ModelNotInitialized(String),

    /// Insufficient data for operation
    #[error("Insufficient data: required {required}, got {actual}")]
    InsufficientData {
        /// Required amount
        required: usize,
        /// Actual amount
        actual: usize,
    },

    /// Numerical computation error
    #[error("Numerical error: {0}")]
    NumericalError(String),

    /// Serialization error
    #[error("Serialization error: {0}")]
    SerializationError(String),

    /// Device error (GPU initialization, memory, etc.)
    #[error("Device error: {0}")]
    DeviceError(String),

    /// Out of memory
    #[error("Out of memory: requested {requested} bytes, available {available} bytes")]
    OutOfMemory {
        /// Requested bytes
        requested: usize,
        /// Available bytes
        available: usize,
    },

    /// Feature not enabled
    #[error("Feature '{feature}' not enabled. Enable with: --features {feature}")]
    FeatureNotEnabled {
        /// Feature name
        feature: String,
    },

    /// IO error
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    /// Generic error
    #[error("{0}")]
    Other(String),

    /// Computation error
    #[error("Compute error: {0}")]
    ComputeError(String),

    /// Configuration error
    #[error("Config error: {0}")]
    ConfigError(String),

    /// Dimension mismatch error
    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch {
        /// Expected dimension
        expected: String,
        /// Actual dimension
        got: String,
    },
}

impl MlError {
    /// Create a backend unavailable error
    pub fn backend_unavailable(backend: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::BackendUnavailable {
            backend: backend.into(),
            reason: reason.into(),
        }
    }

    /// Create a shape mismatch error
    pub fn shape_mismatch(expected: Vec<usize>, actual: Vec<usize>) -> Self {
        Self::ShapeMismatch { expected, actual }
    }

    /// Create an insufficient data error
    pub fn insufficient_data(required: usize, actual: usize) -> Self {
        Self::InsufficientData { required, actual }
    }

    /// Create a feature not enabled error
    pub fn feature_not_enabled(feature: impl Into<String>) -> Self {
        Self::FeatureNotEnabled {
            feature: feature.into(),
        }
    }
}
