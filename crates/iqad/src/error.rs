//! Error handling for IQAD

use thiserror::Error;

/// IQAD error types
#[derive(Error, Debug)]
pub enum IqadError {
    /// Quantum circuit execution error
    #[error("Quantum circuit error: {0}")]
    QuantumError(String),
    
    /// Hardware initialization error
    #[error("Hardware initialization error: {0}")]
    HardwareError(String),
    
    /// Invalid parameter error
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
    
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
    
    /// Generic error
    #[error("IQAD error: {0}")]
    Generic(String),
}

/// Result type for IQAD operations
pub type IqadResult<T> = Result<T, IqadError>;