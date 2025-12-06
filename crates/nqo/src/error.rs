//! Error handling for NQO

use thiserror::Error;

/// NQO error types
#[derive(Error, Debug)]
pub enum NqoError {
    /// Quantum circuit execution error
    #[error("Quantum circuit error: {0}")]
    QuantumError(String),
    
    /// Neural network error
    #[error("Neural network error: {0}")]
    NeuralError(String),
    
    /// Hardware initialization error
    #[error("Hardware initialization error: {0}")]
    HardwareError(String),
    
    /// Invalid parameter error
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
    
    /// Optimization error
    #[error("Optimization error: {0}")]
    OptimizationError(String),
    
    /// Cache error
    #[error("Cache error: {0}")]
    CacheError(String),
    
    /// Configuration error
    #[error("Configuration error: {0}")]
    ConfigError(String),
    
    /// Numerical computation error
    #[error("Numerical error: {0}")]
    NumericalError(String),
    
    /// I/O error
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),
    
    /// Candle tensor error
    #[error("Tensor error: {0}")]
    TensorError(#[from] candle_core::Error),
    
    /// Generic error
    #[error("NQO error: {0}")]
    Generic(String),
}

/// Result type for NQO operations
pub type NqoResult<T> = Result<T, NqoError>;