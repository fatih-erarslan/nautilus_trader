//! Error handling for LMSR operations

use thiserror::Error;

/// Result type for LMSR operations
pub type Result<T> = std::result::Result<T, LMSRError>;

/// Comprehensive error types for LMSR operations
#[derive(Error, Debug, Clone, PartialEq)]
pub enum LMSRError {
    /// Invalid market parameters
    #[error("Invalid market configuration: {message}")]
    InvalidMarket { message: String },
    
    /// Numerical computation errors
    #[error("Numerical error: {message}")]
    NumericalError { message: String },
    
    /// Invalid outcome index
    #[error("Invalid outcome index: {index}, market has {total} outcomes")]
    InvalidOutcome { index: usize, total: usize },
    
    /// Invalid quantity vector
    #[error("Invalid quantity vector: {message}")]
    InvalidQuantity { message: String },
    
    /// Market state errors
    #[error("Market state error: {message}")]
    MarketState { message: String },
    
    /// Liquidity parameter errors
    #[error("Invalid liquidity parameter: {value}, must be positive")]
    InvalidLiquidity { value: f64 },
    
    /// Precision/overflow errors
    #[error("Precision error: {message}")]
    PrecisionError { message: String },
    
    /// Thread safety violations
    #[error("Concurrency error: {message}")]
    ConcurrencyError { message: String },
}

impl LMSRError {
    /// Create an invalid market error
    pub fn invalid_market(msg: impl Into<String>) -> Self {
        Self::InvalidMarket { message: msg.into() }
    }
    
    /// Create a numerical error
    pub fn numerical_error(msg: impl Into<String>) -> Self {
        Self::NumericalError { message: msg.into() }
    }
    
    /// Create an invalid outcome error
    pub fn invalid_outcome(index: usize, total: usize) -> Self {
        Self::InvalidOutcome { index, total }
    }
    
    /// Create an invalid quantity error
    pub fn invalid_quantity(msg: impl Into<String>) -> Self {
        Self::InvalidQuantity { message: msg.into() }
    }
    
    /// Create a market state error
    pub fn market_state(msg: impl Into<String>) -> Self {
        Self::MarketState { message: msg.into() }
    }
    
    /// Create an invalid liquidity error
    pub fn invalid_liquidity(value: f64) -> Self {
        Self::InvalidLiquidity { value }
    }
    
    /// Create a precision error
    pub fn precision_error(msg: impl Into<String>) -> Self {
        Self::PrecisionError { message: msg.into() }
    }
    
    /// Create a concurrency error
    pub fn concurrency_error(msg: impl Into<String>) -> Self {
        Self::ConcurrencyError { message: msg.into() }
    }
}

#[cfg(feature = "pyo3")]
impl From<LMSRError> for pyo3::PyErr {
    fn from(err: LMSRError) -> Self {
        match err {
            LMSRError::InvalidMarket { message } => {
                pyo3::exceptions::PyValueError::new_err(format!("Invalid market: {}", message))
            }
            LMSRError::NumericalError { message } => {
                pyo3::exceptions::PyArithmeticError::new_err(format!("Numerical error: {}", message))
            }
            LMSRError::InvalidOutcome { index, total } => {
                pyo3::exceptions::PyIndexError::new_err(format!("Invalid outcome {}/{}", index, total))
            }
            LMSRError::InvalidQuantity { message } => {
                pyo3::exceptions::PyValueError::new_err(format!("Invalid quantity: {}", message))
            }
            LMSRError::MarketState { message } => {
                pyo3::exceptions::PyRuntimeError::new_err(format!("Market state: {}", message))
            }
            LMSRError::InvalidLiquidity { value } => {
                pyo3::exceptions::PyValueError::new_err(format!("Invalid liquidity: {}", value))
            }
            LMSRError::PrecisionError { message } => {
                pyo3::exceptions::PyOverflowError::new_err(format!("Precision error: {}", message))
            }
            LMSRError::ConcurrencyError { message } => {
                pyo3::exceptions::PyRuntimeError::new_err(format!("Concurrency error: {}", message))
            }
        }
    }
}