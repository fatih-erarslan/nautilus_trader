//! Error types for Quantum LSTM

use thiserror::Error;

/// Main error type for Quantum LSTM operations
#[derive(Error, Debug)]
pub enum QuantumLSTMError {
    /// Configuration error
    #[error("Configuration error: {0}")]
    Config(String),
    
    /// Quantum device error
    #[error("Quantum device error: {0}")]
    Device(String),
    
    /// Encoding error
    #[error("Encoding error: {0}")]
    Encoding(String),
    
    /// Circuit execution error
    #[error("Circuit execution error: {0}")]
    Circuit(String),
    
    /// Memory allocation error
    #[error("Memory allocation error: {0}")]
    Memory(String),
    
    /// Dimension mismatch error
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: String, actual: String },
    
    /// Invalid parameter error
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
    
    /// Hardware acceleration error
    #[error("Hardware acceleration error: {0}")]
    Hardware(String),
    
    /// Numerical error
    #[error("Numerical error: {0}")]
    Numerical(String),
    
    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    
    /// Quantum core error
    #[error("Quantum core error: {0}")]
    QuantumCore(#[from] quantum_core::QuantumError),
    
    /// Other errors
    #[error("Other error: {0}")]
    Other(String),
}

/// Result type for Quantum LSTM operations
pub type Result<T> = std::result::Result<T, QuantumLSTMError>;

impl From<anyhow::Error> for QuantumLSTMError {
    fn from(err: anyhow::Error) -> Self {
        QuantumLSTMError::Other(err.to_string())
    }
}

impl From<String> for QuantumLSTMError {
    fn from(err: String) -> Self {
        QuantumLSTMError::Other(err)
    }
}

impl From<&str> for QuantumLSTMError {
    fn from(err: &str) -> Self {
        QuantumLSTMError::Other(err.to_string())
    }
}