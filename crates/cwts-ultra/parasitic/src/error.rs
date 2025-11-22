//! Error handling for the parasitic pairlist system
//! CQGS Compliant error handling with comprehensive coverage

use std::fmt;
use thiserror::Error;

/// Result type alias for the parasitic system
pub type Result<T> = std::result::Result<T, Error>;

/// Comprehensive error enumeration for the parasitic system
#[derive(Error, Debug, Clone)]
pub enum Error {
    /// Invalid market data provided
    #[error("Invalid data: {0}")]
    InvalidData(String),

    /// Configuration error
    #[error("Configuration error: {0}")]
    Configuration(String),

    /// Performance requirement violation
    #[error("Performance violation: operation took {actual_ns}ns, expected <{max_ns}ns")]
    PerformanceViolation { actual_ns: u64, max_ns: u64 },

    /// Memory limit exceeded
    #[error("Memory limit exceeded: {used_bytes} bytes used, limit is {limit_bytes} bytes")]
    MemoryLimitExceeded {
        used_bytes: usize,
        limit_bytes: usize,
    },

    /// SIMD processing error
    #[error("SIMD processing error: {0}")]
    SimdError(String),

    /// Mathematical computation error
    #[error("Mathematical error: {0}")]
    Mathematics(String),

    /// Organism initialization error
    #[error("Failed to initialize organism {organism}: {reason}")]
    OrganismInitialization { organism: String, reason: String },

    /// Detection threshold error
    #[error("Detection threshold error: {0}")]
    ThresholdError(String),

    /// Tracking error
    #[error("Tracking error: {0}")]
    TrackingError(String),

    /// Strategy execution error
    #[error("Strategy execution failed: {0}")]
    StrategyError(String),

    /// Serialization/deserialization error
    #[error("Serialization error: {0}")]
    Serialization(String),

    /// Thread safety violation
    #[error("Thread safety violation: {0}")]
    ThreadSafety(String),

    /// Resource exhaustion
    #[error("Resource exhausted: {resource}")]
    ResourceExhausted { resource: String },

    /// Timeout error
    #[error("Operation timed out after {timeout_ms}ms")]
    Timeout { timeout_ms: u64 },

    /// Calibration error
    #[error("Calibration failed: {0}")]
    CalibrationError(String),

    /// Data validation error
    #[error("Data validation failed: {field} = {value}, {constraint}")]
    ValidationError {
        field: String,
        value: String,
        constraint: String,
    },

    /// External system error
    #[error("External system error: {0}")]
    External(String),

    /// Unknown/unexpected error
    #[error("Unknown error: {0}")]
    Unknown(String),
}

impl Error {
    /// Create an invalid data error with context
    pub fn invalid_data<S: Into<String>>(msg: S) -> Self {
        Self::InvalidData(msg.into())
    }

    /// Create a performance violation error
    pub fn performance_violation(actual_ns: u64, max_ns: u64) -> Self {
        Self::PerformanceViolation { actual_ns, max_ns }
    }

    /// Create a memory limit exceeded error
    pub fn memory_limit_exceeded(used_bytes: usize, limit_bytes: usize) -> Self {
        Self::MemoryLimitExceeded {
            used_bytes,
            limit_bytes,
        }
    }

    /// Create an organism initialization error
    pub fn organism_init<S1: Into<String>, S2: Into<String>>(organism: S1, reason: S2) -> Self {
        Self::OrganismInitialization {
            organism: organism.into(),
            reason: reason.into(),
        }
    }

    /// Create a validation error
    pub fn validation<S1, S2, S3>(field: S1, value: S2, constraint: S3) -> Self
    where
        S1: Into<String>,
        S2: fmt::Display,
        S3: Into<String>,
    {
        Self::ValidationError {
            field: field.into(),
            value: value.to_string(),
            constraint: constraint.into(),
        }
    }

    /// Check if error is recoverable
    pub fn is_recoverable(&self) -> bool {
        match self {
            Self::InvalidData(_) => false,
            Self::Configuration(_) => false,
            Self::PerformanceViolation { .. } => true,
            Self::MemoryLimitExceeded { .. } => true,
            Self::SimdError(_) => true,
            Self::Mathematics(_) => false,
            Self::OrganismInitialization { .. } => false,
            Self::ThresholdError(_) => true,
            Self::TrackingError(_) => true,
            Self::StrategyError(_) => true,
            Self::Serialization(_) => false,
            Self::ThreadSafety(_) => false,
            Self::ResourceExhausted { .. } => true,
            Self::Timeout { .. } => true,
            Self::CalibrationError(_) => true,
            Self::ValidationError { .. } => false,
            Self::External(_) => true,
            Self::Unknown(_) => false,
        }
    }

    /// Get error severity level
    pub fn severity(&self) -> ErrorSeverity {
        match self {
            Self::InvalidData(_) => ErrorSeverity::High,
            Self::Configuration(_) => ErrorSeverity::High,
            Self::PerformanceViolation { .. } => ErrorSeverity::Medium,
            Self::MemoryLimitExceeded { .. } => ErrorSeverity::High,
            Self::SimdError(_) => ErrorSeverity::Medium,
            Self::Mathematics(_) => ErrorSeverity::High,
            Self::OrganismInitialization { .. } => ErrorSeverity::High,
            Self::ThresholdError(_) => ErrorSeverity::Low,
            Self::TrackingError(_) => ErrorSeverity::Medium,
            Self::StrategyError(_) => ErrorSeverity::Medium,
            Self::Serialization(_) => ErrorSeverity::Medium,
            Self::ThreadSafety(_) => ErrorSeverity::Critical,
            Self::ResourceExhausted { .. } => ErrorSeverity::High,
            Self::Timeout { .. } => ErrorSeverity::Medium,
            Self::CalibrationError(_) => ErrorSeverity::Low,
            Self::ValidationError { .. } => ErrorSeverity::High,
            Self::External(_) => ErrorSeverity::Medium,
            Self::Unknown(_) => ErrorSeverity::High,
        }
    }

    /// Get suggested recovery action
    pub fn recovery_suggestion(&self) -> Option<&'static str> {
        match self {
            Self::PerformanceViolation { .. } => {
                Some("Reduce batch size or enable SIMD optimizations")
            }
            Self::MemoryLimitExceeded { .. } => {
                Some("Clear tracking history or reduce buffer sizes")
            }
            Self::SimdError(_) => Some("Fall back to scalar processing"),
            Self::ThresholdError(_) => Some("Recalibrate detection thresholds"),
            Self::TrackingError(_) => Some("Reset tracking state and restart"),
            Self::StrategyError(_) => Some("Switch to conservative strategy parameters"),
            Self::ResourceExhausted { .. } => Some("Wait for resources to become available"),
            Self::Timeout { .. } => Some("Increase timeout or optimize processing"),
            Self::CalibrationError(_) => Some("Use more representative calibration data"),
            Self::External(_) => Some("Check external system connectivity and retry"),
            _ => None,
        }
    }
}

/// Error severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ErrorSeverity {
    Low,
    Medium,
    High,
    Critical,
}

impl fmt::Display for ErrorSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Low => write!(f, "LOW"),
            Self::Medium => write!(f, "MEDIUM"),
            Self::High => write!(f, "HIGH"),
            Self::Critical => write!(f, "CRITICAL"),
        }
    }
}

/// Error context for debugging and logging
#[derive(Debug, Clone)]
pub struct ErrorContext {
    pub error: Error,
    pub component: String,
    pub function: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub additional_info: std::collections::HashMap<String, String>,
}

impl ErrorContext {
    pub fn new<S1: Into<String>, S2: Into<String>>(
        error: Error,
        component: S1,
        function: S2,
    ) -> Self {
        Self {
            error,
            component: component.into(),
            function: function.into(),
            timestamp: chrono::Utc::now(),
            additional_info: std::collections::HashMap::new(),
        }
    }

    pub fn with_info<K: Into<String>, V: Into<String>>(mut self, key: K, value: V) -> Self {
        self.additional_info.insert(key.into(), value.into());
        self
    }
}

// Standard library error conversions
impl From<std::io::Error> for Error {
    fn from(err: std::io::Error) -> Self {
        Self::External(format!("IO error: {}", err))
    }
}

impl From<serde_json::Error> for Error {
    fn from(err: serde_json::Error) -> Self {
        Self::Serialization(format!("JSON error: {}", err))
    }
}

// bincode Error conversion removed - dependency not available

impl From<std::num::ParseFloatError> for Error {
    fn from(err: std::num::ParseFloatError) -> Self {
        Self::InvalidData(format!("Float parsing error: {}", err))
    }
}

impl From<std::num::ParseIntError> for Error {
    fn from(err: std::num::ParseIntError) -> Self {
        Self::InvalidData(format!("Integer parsing error: {}", err))
    }
}

// Validation helper functions
pub fn validate_positive(value: f64, field: &str) -> Result<()> {
    if value <= 0.0 {
        Err(Error::validation(field, value, "must be positive"))
    } else {
        Ok(())
    }
}

pub fn validate_range(value: f64, min: f64, max: f64, field: &str) -> Result<()> {
    if value < min || value > max {
        Err(Error::validation(
            field,
            value,
            format!("must be between {} and {}", min, max),
        ))
    } else {
        Ok(())
    }
}

pub fn validate_normalized(value: f64, field: &str) -> Result<()> {
    validate_range(value, 0.0, 1.0, field)
}

pub fn validate_not_nan(value: f64, field: &str) -> Result<()> {
    if value.is_nan() {
        Err(Error::validation(field, "NaN", "must not be NaN"))
    } else {
        Ok(())
    }
}

pub fn validate_not_infinite(value: f64, field: &str) -> Result<()> {
    if value.is_infinite() {
        Err(Error::validation(field, "Infinite", "must not be infinite"))
    } else {
        Ok(())
    }
}

pub fn validate_finite(value: f64, field: &str) -> Result<()> {
    validate_not_nan(value, field)?;
    validate_not_infinite(value, field)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let err = Error::invalid_data("negative price");
        assert!(matches!(err, Error::InvalidData(_)));
        assert!(!err.is_recoverable());
        assert_eq!(err.severity(), ErrorSeverity::High);
    }

    #[test]
    fn test_performance_error() {
        let err = Error::performance_violation(150_000, 100_000);
        assert!(matches!(err, Error::PerformanceViolation { .. }));
        assert!(err.is_recoverable());
        assert!(err.recovery_suggestion().is_some());
    }

    #[test]
    fn test_validation_helpers() {
        assert!(validate_positive(1.0, "test").is_ok());
        assert!(validate_positive(-1.0, "test").is_err());

        assert!(validate_range(0.5, 0.0, 1.0, "test").is_ok());
        assert!(validate_range(1.5, 0.0, 1.0, "test").is_err());

        assert!(validate_normalized(0.8, "test").is_ok());
        assert!(validate_normalized(1.5, "test").is_err());
    }

    #[test]
    fn test_error_context() {
        let error = Error::invalid_data("test error");
        let context = ErrorContext::new(error, "komodo", "detect_wound")
            .with_info("symbol", "BTC_USD")
            .with_info("timestamp", "2024-01-01");

        assert_eq!(context.component, "komodo");
        assert_eq!(context.function, "detect_wound");
        assert!(context.additional_info.contains_key("symbol"));
    }
}
