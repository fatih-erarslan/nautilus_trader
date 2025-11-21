//! Error handling for ATS-Core
//!
//! This module provides comprehensive error handling for all ATS-CP mathematical operations.
//! It ensures that errors are properly categorized and include sufficient context for debugging.

use thiserror::Error;

/// Result type alias for ATS-Core operations
pub type Result<T> = std::result::Result<T, AtsCoreError>;

/// Comprehensive error types for ATS-Core operations
#[derive(Error, Debug, Clone)]
pub enum AtsCoreError {
    /// Configuration errors
    #[error("Configuration error: {message}")]
    Configuration { 
        /// Error message describing the configuration issue
        message: String 
    },

    /// Mathematical computation errors
    #[error("Mathematical error: {operation} failed with {details}")]
    Mathematical {
        /// The mathematical operation that failed
        operation: String,
        /// Detailed error information
        details: String,
    },

    /// Memory allocation or alignment errors
    #[error("Memory error: {message}")]
    Memory { 
        /// Error message describing the memory issue
        message: String 
    },

    /// SIMD operation errors
    #[error("SIMD error: {operation} failed - {details}")]
    Simd {
        /// The SIMD operation that failed
        operation: String,
        /// Detailed error information
        details: String,
    },

    /// Parallel processing errors
    #[error("Parallel processing error: {message}")]
    Parallel { 
        /// Error message describing the parallel processing issue
        message: String 
    },

    /// Performance monitoring errors
    #[error("Performance monitoring error: {message}")]
    Performance { 
        /// Error message describing the performance monitoring issue
        message: String 
    },

    /// Integration errors with ruv-FANN
    #[error("ruv-FANN integration error: {message}")]
    Integration { 
        /// Error message describing the integration issue
        message: String 
    },

    /// Input validation errors
    #[error("Input validation error: {field} - {message}")]
    Validation { 
        /// The field that failed validation
        field: String, 
        /// Error message describing the validation failure
        message: String 
    },

    /// Temperature scaling specific errors
    #[error("Temperature scaling error: {message}")]
    TemperatureScaling { 
        /// Error message describing the temperature scaling issue
        message: String 
    },

    /// Conformal prediction specific errors
    #[error("Conformal prediction error: {message}")]
    ConformalPrediction { 
        /// Error message describing the conformal prediction issue
        message: String 
    },

    /// Array dimension mismatch errors
    #[error("Array dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { 
        /// Expected array dimension
        expected: usize, 
        /// Actual array dimension that was received
        actual: usize 
    },

    /// Timeout errors for real-time operations
    #[error("Operation timeout: {operation} exceeded {timeout_us}μs")]
    Timeout {
        /// The operation that timed out
        operation: String,
        /// Timeout duration in microseconds
        timeout_us: u64,
    },

    /// Numerical precision errors
    #[error("Numerical precision error: {message}")]
    Precision { 
        /// Error message describing the precision issue
        message: String 
    },

    /// Resource exhaustion errors
    #[error("Resource exhaustion: {resource} - {message}")]
    ResourceExhaustion { 
        /// The resource that was exhausted
        resource: String, 
        /// Error message describing the exhaustion issue
        message: String 
    },

    /// Unknown or unexpected errors
    #[error("Unexpected error: {message}")]
    Unknown {
        /// Error message describing the unexpected issue
        message: String
    },

    /// Validation failed errors
    #[error("Validation failed: {0}")]
    ValidationFailed(String),

    /// Computation failed errors
    #[error("Computation failed: {0}")]
    ComputationFailed(String),

    /// Integration error - same as Integration variant but matches existing API usage
    #[error("Integration error: {0}")]
    IntegrationError(String),
}

/// Integration error type alias for backwards compatibility
pub type IntegrationError = AtsCoreError;

impl AtsCoreError {
    /// Creates a configuration error
    pub fn configuration<S: Into<String>>(message: S) -> Self {
        Self::Configuration {
            message: message.into(),
        }
    }

    /// Creates a mathematical error
    pub fn mathematical<S: Into<String>>(operation: S, details: S) -> Self {
        Self::Mathematical {
            operation: operation.into(),
            details: details.into(),
        }
    }

    /// Creates a memory error
    pub fn memory<S: Into<String>>(message: S) -> Self {
        Self::Memory {
            message: message.into(),
        }
    }

    /// Creates a SIMD error
    pub fn simd<S: Into<String>>(operation: S, details: S) -> Self {
        Self::Simd {
            operation: operation.into(),
            details: details.into(),
        }
    }

    /// Creates a parallel processing error
    pub fn parallel<S: Into<String>>(message: S) -> Self {
        Self::Parallel {
            message: message.into(),
        }
    }

    /// Creates a performance monitoring error
    pub fn performance<S: Into<String>>(message: S) -> Self {
        Self::Performance {
            message: message.into(),
        }
    }

    /// Creates an integration error
    pub fn integration<S: Into<String>>(message: S) -> Self {
        Self::Integration {
            message: message.into(),
        }
    }

    /// Creates a validation error
    pub fn validation<S: Into<String>>(field: S, message: S) -> Self {
        Self::Validation {
            field: field.into(),
            message: message.into(),
        }
    }

    /// Creates a temperature scaling error
    pub fn temperature_scaling<S: Into<String>>(message: S) -> Self {
        Self::TemperatureScaling {
            message: message.into(),
        }
    }

    /// Creates a conformal prediction error
    pub fn conformal_prediction<S: Into<String>>(message: S) -> Self {
        Self::ConformalPrediction {
            message: message.into(),
        }
    }

    /// Creates a dimension mismatch error
    pub fn dimension_mismatch(expected: usize, actual: usize) -> Self {
        Self::DimensionMismatch { expected, actual }
    }

    /// Creates a timeout error
    pub fn timeout<S: Into<String>>(operation: S, timeout_us: u64) -> Self {
        Self::Timeout {
            operation: operation.into(),
            timeout_us,
        }
    }

    /// Creates a precision error
    pub fn precision<S: Into<String>>(message: S) -> Self {
        Self::Precision {
            message: message.into(),
        }
    }

    /// Creates a resource exhaustion error
    pub fn resource_exhaustion<S: Into<String>>(resource: S, message: S) -> Self {
        Self::ResourceExhaustion {
            resource: resource.into(),
            message: message.into(),
        }
    }

    /// Creates an unknown error
    pub fn unknown<S: Into<String>>(message: S) -> Self {
        Self::Unknown {
            message: message.into(),
        }
    }

    /// Returns true if the error is recoverable
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            AtsCoreError::Timeout { .. }
                | AtsCoreError::ResourceExhaustion { .. }
                | AtsCoreError::Precision { .. }
        )
    }

    /// Returns true if the error is critical (system should stop)
    pub fn is_critical(&self) -> bool {
        matches!(
            self,
            AtsCoreError::Memory { .. }
                | AtsCoreError::Configuration { .. }
                | AtsCoreError::Integration { .. }
        )
    }

    /// Returns the error category for monitoring and logging
    pub fn category(&self) -> &'static str {
        match self {
            AtsCoreError::Configuration { .. } => "configuration",
            AtsCoreError::Mathematical { .. } => "mathematical",
            AtsCoreError::Memory { .. } => "memory",
            AtsCoreError::Simd { .. } => "simd",
            AtsCoreError::Parallel { .. } => "parallel",
            AtsCoreError::Performance { .. } => "performance",
            AtsCoreError::Integration { .. } => "integration",
            AtsCoreError::Validation { .. } => "validation",
            AtsCoreError::TemperatureScaling { .. } => "temperature_scaling",
            AtsCoreError::ConformalPrediction { .. } => "conformal_prediction",
            AtsCoreError::DimensionMismatch { .. } => "dimension_mismatch",
            AtsCoreError::Timeout { .. } => "timeout",
            AtsCoreError::Precision { .. } => "precision",
            AtsCoreError::ResourceExhaustion { .. } => "resource_exhaustion",
            AtsCoreError::Unknown { .. } => "unknown",
            AtsCoreError::ValidationFailed(..) => "validation_failed",
            AtsCoreError::ComputationFailed(..) => "computation_failed",
            AtsCoreError::IntegrationError(..) => "integration_error",
        }
    }
}

impl From<serde_json::Error> for AtsCoreError {
    fn from(err: serde_json::Error) -> Self {
        AtsCoreError::Configuration {
            message: format!("JSON serialization error: {}", err),
        }
    }
}

impl From<crate::ruv_fann_integration::IntegrationError> for AtsCoreError {
    fn from(err: crate::ruv_fann_integration::IntegrationError) -> Self {
        AtsCoreError::Integration {
            message: err.to_string(),
        }
    }
}

impl From<std::io::Error> for AtsCoreError {
    fn from(err: std::io::Error) -> Self {
        AtsCoreError::Configuration {
            message: format!("IO error: {}", err),
        }
    }
}

impl From<Box<dyn std::error::Error>> for AtsCoreError {
    fn from(err: Box<dyn std::error::Error>) -> Self {
        AtsCoreError::Unknown {
            message: err.to_string(),
        }
    }
}

impl From<Box<dyn std::error::Error + Send + Sync>> for AtsCoreError {
    fn from(err: Box<dyn std::error::Error + Send + Sync>) -> Self {
        AtsCoreError::Unknown {
            message: err.to_string(),
        }
    }
}

impl From<String> for AtsCoreError {
    fn from(err: String) -> Self {
        AtsCoreError::Unknown {
            message: err,
        }
    }
}

impl From<&str> for AtsCoreError {
    fn from(err: &str) -> Self {
        AtsCoreError::Unknown {
            message: err.to_string(),
        }
    }
}

/// Convenience macro for creating mathematical errors
#[macro_export]
macro_rules! math_error {
    ($op:expr, $details:expr) => {
        AtsCoreError::mathematical($op, $details)
    };
}

/// Convenience macro for creating validation errors
#[macro_export]
macro_rules! validation_error {
    ($field:expr, $message:expr) => {
        AtsCoreError::validation($field, $message)
    };
}

/// Convenience macro for creating timeout errors
#[macro_export]
macro_rules! timeout_error {
    ($op:expr, $timeout:expr) => {
        AtsCoreError::timeout($op, $timeout)
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let config_error = AtsCoreError::configuration("Invalid parameter");
        assert!(matches!(config_error, AtsCoreError::Configuration { .. }));

        let math_error = AtsCoreError::mathematical("matrix_multiply", "singular matrix");
        assert!(matches!(math_error, AtsCoreError::Mathematical { .. }));

        let dim_error = AtsCoreError::dimension_mismatch(100, 50);
        assert!(matches!(dim_error, AtsCoreError::DimensionMismatch { .. }));
    }

    #[test]
    fn test_error_properties() {
        let timeout_error = AtsCoreError::timeout("temperature_scale", 5000);
        assert!(timeout_error.is_recoverable());
        assert!(!timeout_error.is_critical());
        assert_eq!(timeout_error.category(), "timeout");

        let memory_error = AtsCoreError::memory("allocation failed");
        assert!(!memory_error.is_recoverable());
        assert!(memory_error.is_critical());
        assert_eq!(memory_error.category(), "memory");
    }

    #[test]
    fn test_error_macros() {
        let math_err = math_error!("division", "divide by zero");
        assert!(matches!(math_err, AtsCoreError::Mathematical { .. }));

        let validation_err = validation_error!("temperature", "must be positive");
        assert!(matches!(validation_err, AtsCoreError::Validation { .. }));

        let timeout_err = timeout_error!("conformal_predict", 20000);
        assert!(matches!(timeout_err, AtsCoreError::Timeout { .. }));
    }

    #[test]
    fn test_error_display() {
        let error = AtsCoreError::temperature_scaling("Invalid temperature value");
        let error_string = format!("{}", error);
        assert!(error_string.contains("Temperature scaling error"));
        assert!(error_string.contains("Invalid temperature value"));
    }
}