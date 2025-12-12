//! Error types for Quantum Annealing Regression

use thiserror::Error;

/// Error type for quantum annealing operations
#[derive(Error, Debug)]
pub enum QuantumAnnealingError {
    /// Configuration error
    #[error("Configuration error: {0}")]
    Config(String),
    
    /// Optimization error
    #[error("Optimization error: {0}")]
    Optimization(String),
    
    /// Numerical error
    #[error("Numerical error: {0}")]
    Numerical(String),
    
    /// Regression error
    #[error("Regression error: {0}")]
    RegressionError(String),
    
    /// Invalid input error
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    
    /// Integration error
    #[error("Integration error: {0}")]
    IntegrationError(String),
    
    /// Other error
    #[error("Error: {0}")]
    Other(String),
}

/// Alias for QarError for compatibility
pub type QarError = QuantumAnnealingError;

/// Result type for quantum annealing operations
pub type Result<T> = std::result::Result<T, QuantumAnnealingError>;

/// Alias for QarResult for compatibility
pub type QarResult<T> = std::result::Result<T, QuantumAnnealingError>;

impl From<String> for QuantumAnnealingError {
    fn from(s: String) -> Self {
        QuantumAnnealingError::Other(s)
    }
}

impl From<&str> for QuantumAnnealingError {
    fn from(s: &str) -> Self {
        QuantumAnnealingError::Other(s.to_string())
    }
}