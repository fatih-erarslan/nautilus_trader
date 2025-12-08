//! Error types for the Talebian Risk Management library

use thiserror::Error;

/// Main error type for Talebian risk management operations
#[derive(Error, Debug)]
pub enum TalebianError {
    #[error("Antifragility calculation error: {0}")]
    AntifragilityError(String),

    #[error("Black swan detection error: {0}")]
    BlackSwanError(String),

    #[error("Barbell strategy error: {0}")]
    BarbellError(String),

    #[error("Distribution fitting error: {0}")]
    DistributionError(String),

    #[error("Optimization error: {0}")]
    OptimizationError(String),

    #[error("Quantum computation error: {0}")]
    QuantumError(String),

    #[error("Data validation error: {0}")]
    DataError(String),

    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),

    #[error("Mathematical error: {0}")]
    MathError(String),
}

/// Result type for Talebian risk management operations
pub type TalebianResult<T> = Result<T, TalebianError>;

impl TalebianError {
    /// Create a new antifragility error
    pub fn antifragility<S: Into<String>>(msg: S) -> Self {
        Self::AntifragilityError(msg.into())
    }

    /// Create a new black swan error
    pub fn black_swan<S: Into<String>>(msg: S) -> Self {
        Self::BlackSwanError(msg.into())
    }

    /// Create a new barbell strategy error
    pub fn barbell<S: Into<String>>(msg: S) -> Self {
        Self::BarbellError(msg.into())
    }

    /// Create a new distribution error
    pub fn distribution<S: Into<String>>(msg: S) -> Self {
        Self::DistributionError(msg.into())
    }

    /// Create a new optimization error
    pub fn optimization<S: Into<String>>(msg: S) -> Self {
        Self::OptimizationError(msg.into())
    }

    /// Create a new quantum computation error
    pub fn quantum<S: Into<String>>(msg: S) -> Self {
        Self::QuantumError(msg.into())
    }

    /// Create a new data validation error
    pub fn data<S: Into<String>>(msg: S) -> Self {
        Self::DataError(msg.into())
    }

    /// Create a new configuration error
    pub fn config<S: Into<String>>(msg: S) -> Self {
        Self::ConfigError(msg.into())
    }

    /// Create a new mathematical error
    pub fn math<S: Into<String>>(msg: S) -> Self {
        Self::MathError(msg.into())
    }

    /// Create an insufficient data error
    pub fn insufficient_data(required: usize, available: usize) -> Self {
        Self::DataError(format!(
            "Insufficient data: required {}, available {}",
            required, available
        ))
    }

    /// Create an invalid parameter error
    pub fn invalid_parameter<S: Into<String>>(param: S, msg: S) -> Self {
        Self::ConfigError(format!(
            "Invalid parameter '{}': {}",
            param.into(),
            msg.into()
        ))
    }

    /// Create a portfolio construction error
    pub fn portfolio_construction<S: Into<String>>(msg: S) -> Self {
        Self::ConfigError(format!("Portfolio construction error: {}", msg.into()))
    }

    /// Create a mathematical error
    pub fn mathematical<S: Into<String>>(msg: S) -> Self {
        Self::MathError(msg.into())
    }
}

// Re-export for backward compatibility
// pub type Result<T> = std::result::Result<T, TalebianError>;
