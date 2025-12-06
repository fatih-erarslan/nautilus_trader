//! Error types for QBMIA Core

use thiserror::Error;

/// Result type alias for QBMIA operations
pub type Result<T> = std::result::Result<T, QBMIAError>;

/// Main error type for QBMIA operations
#[derive(Error, Debug)]
pub enum QBMIAError {
    #[error("Quantum simulation error: {0}")]
    QuantumSimulation(String),
    
    #[error("Nash equilibrium convergence failed: {0}")]
    NashConvergence(String),
    
    #[error("Memory allocation error: {0}")]
    Memory(String),
    
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    
    #[error("Binary serialization error: {0}")]
    BinarySerialization(#[from] bincode::Error),
    
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Configuration error: {0}")]
    Config(String),
    
    #[error("Hardware optimization error: {0}")]
    HardwareOptimization(String),
    
    #[error("Market data error: {0}")]
    MarketData(String),
    
    #[error("Strategy execution error: {0}")]
    Strategy(String),
    
    #[error("Async operation error: {0}")]
    Async(String),
    
    #[error("Numerical computation error: {0}")]
    Numerical(String),
    
    #[error("Resource allocation error: {0}")]
    Resource(String),
    
    #[error("Validation error: {0}")]
    Validation(String),
    
    #[error("Unknown error: {0}")]
    Unknown(String),
}

impl QBMIAError {
    /// Create a new quantum simulation error
    pub fn quantum_simulation(msg: impl Into<String>) -> Self {
        Self::QuantumSimulation(msg.into())
    }
    
    /// Create a new Nash convergence error
    pub fn nash_convergence(msg: impl Into<String>) -> Self {
        Self::NashConvergence(msg.into())
    }
    
    /// Create a new memory error
    pub fn memory(msg: impl Into<String>) -> Self {
        Self::Memory(msg.into())
    }
    
    /// Create a new configuration error
    pub fn config(msg: impl Into<String>) -> Self {
        Self::Config(msg.into())
    }
    
    /// Create a new hardware optimization error
    pub fn hardware_optimization(msg: impl Into<String>) -> Self {
        Self::HardwareOptimization(msg.into())
    }
    
    /// Create a new market data error
    pub fn market_data(msg: impl Into<String>) -> Self {
        Self::MarketData(msg.into())
    }
    
    /// Create a new strategy error
    pub fn strategy(msg: impl Into<String>) -> Self {
        Self::Strategy(msg.into())
    }
    
    /// Create a new async error
    pub fn async_error(msg: impl Into<String>) -> Self {
        Self::Async(msg.into())
    }
    
    /// Create a new numerical error
    pub fn numerical(msg: impl Into<String>) -> Self {
        Self::Numerical(msg.into())
    }
    
    /// Create a new resource error
    pub fn resource(msg: impl Into<String>) -> Self {
        Self::Resource(msg.into())
    }
    
    /// Create a new validation error
    pub fn validation(msg: impl Into<String>) -> Self {
        Self::Validation(msg.into())
    }
    
    /// Create a new unknown error
    pub fn unknown(msg: impl Into<String>) -> Self {
        Self::Unknown(msg.into())
    }
}

/// Convenience trait for converting errors to QBMIA errors
pub trait IntoQBMIAError<T> {
    fn into_qbmia_error(self) -> Result<T>;
}

impl<T, E> IntoQBMIAError<T> for std::result::Result<T, E>
where
    E: std::fmt::Display,
{
    fn into_qbmia_error(self) -> Result<T> {
        self.map_err(|e| QBMIAError::unknown(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_error_creation() {
        let err = QBMIAError::quantum_simulation("test error");
        assert!(err.to_string().contains("test error"));
    }
    
    #[test]
    fn test_error_chain() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let qbmia_err = QBMIAError::from(io_err);
        assert!(qbmia_err.to_string().contains("file not found"));
    }
}