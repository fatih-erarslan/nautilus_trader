//! Error types for the risk management system

use thiserror::Error;

/// Result type alias for risk management operations
pub type Result<T> = std::result::Result<T, RiskError>;

/// Comprehensive error types for risk management
#[derive(Debug, Error)]
pub enum RiskError {
    /// Configuration errors
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// VaR calculation errors
    #[error("VaR calculation failed: {0}")]
    VaRCalculationFailed(String),

    /// Insufficient data for risk calculation
    #[error("Insufficient data: {message}, required: {required}, available: {available}")]
    InsufficientData {
        message: String,
        required: usize,
        available: usize,
    },

    /// Position not found
    #[error("Position not found for symbol: {0}")]
    PositionNotFound(String),

    /// Risk limit exceeded
    #[error("Risk limit exceeded: {limit_type}, current: {current:.4}, limit: {limit:.4}")]
    RiskLimitExceeded {
        limit_type: String,
        current: f64,
        limit: f64,
    },

    /// Portfolio error
    #[error("Portfolio error: {0}")]
    PortfolioError(String),

    /// Correlation calculation error
    #[error("Correlation calculation failed: {0}")]
    CorrelationError(String),

    /// Numerical computation error
    #[error("Numerical computation error: {0}")]
    NumericalError(String),

    /// Matrix operation error
    #[error("Matrix operation failed: {0}")]
    MatrixError(String),

    /// GPU acceleration error
    #[error("GPU acceleration error: {0}")]
    GpuError(String),

    /// Emergency protocol error
    #[error("Emergency protocol error: {0}")]
    EmergencyProtocolError(String),

    /// Stress test error
    #[error("Stress test failed: {0}")]
    StressTestError(String),

    /// Kelly criterion calculation error
    #[error("Kelly criterion calculation failed: {0}")]
    KellyCriterionError(String),

    /// Invalid input
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Timeout error
    #[error("Operation timed out after {0} seconds")]
    Timeout(u64),

    /// Serialization error
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),

    /// I/O error
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    /// Generic error with context
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

impl RiskError {
    /// Create an insufficient data error
    pub fn insufficient_data(message: impl Into<String>, required: usize, available: usize) -> Self {
        Self::InsufficientData {
            message: message.into(),
            required,
            available,
        }
    }

    /// Create a risk limit exceeded error
    pub fn risk_limit_exceeded(
        limit_type: impl Into<String>,
        current: f64,
        limit: f64,
    ) -> Self {
        Self::RiskLimitExceeded {
            limit_type: limit_type.into(),
            current,
            limit,
        }
    }

    /// Check if error is retryable
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            Self::Timeout(_) | Self::GpuError(_) | Self::NumericalError(_)
        )
    }

    /// Get error severity level
    pub fn severity(&self) -> ErrorSeverity {
        match self {
            Self::RiskLimitExceeded { .. } => ErrorSeverity::Critical,
            Self::EmergencyProtocolError(_) => ErrorSeverity::Critical,
            Self::PositionNotFound(_) => ErrorSeverity::Warning,
            Self::InsufficientData { .. } => ErrorSeverity::Warning,
            Self::Timeout(_) => ErrorSeverity::Warning,
            _ => ErrorSeverity::Error,
        }
    }
}

/// Error severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ErrorSeverity {
    /// Informational message
    Info,
    /// Warning that doesn't prevent operation
    Warning,
    /// Error that prevents operation
    Error,
    /// Critical error requiring immediate attention
    Critical,
}

impl std::fmt::Display for ErrorSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Info => write!(f, "INFO"),
            Self::Warning => write!(f, "WARNING"),
            Self::Error => write!(f, "ERROR"),
            Self::Critical => write!(f, "CRITICAL"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let err = RiskError::insufficient_data("Not enough historical data", 100, 50);
        assert!(matches!(err::InsufficientData { .. }));
    }

    #[test]
    fn test_error_severity() {
        let err = RiskError::risk_limit_exceeded("VaR Limit", 0.15, 0.10);
        assert_eq!(err.severity(), ErrorSeverity::Critical);
    }

    #[test]
    fn test_error_retryable() {
        let timeout_err = RiskError::Timeout(30);
        assert!(timeout_err.is_retryable());

        let config_err = RiskError::InvalidConfig("bad config".into());
        assert!(!config_err.is_retryable());
    }
}
