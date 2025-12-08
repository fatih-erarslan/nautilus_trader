//! Error handling for the CDFA system
//!
//! This module provides a comprehensive error type using `thiserror` for
//! ergonomic error handling throughout the CDFA ecosystem.

use core::fmt;

#[cfg(feature = "std")]
use thiserror::Error;

/// Result type alias for CDFA operations
pub type Result<T> = core::result::Result<T, Error>;

/// Main error type for CDFA operations
#[derive(Debug)]
#[cfg_attr(feature = "std", derive(Error))]
pub enum Error {
    /// Configuration error
    #[cfg_attr(feature = "std", error("Configuration error: {message}"))]
    Config {
        /// Error message
        message: String,
    },

    /// Invalid input data
    #[cfg_attr(feature = "std", error("Invalid input: {message}"))]
    InvalidInput {
        /// Error message
        message: String,
    },

    /// Insufficient data for analysis
    #[cfg_attr(feature = "std", error("Insufficient data: expected at least {expected}, got {actual}"))]
    InsufficientData {
        /// Expected minimum data points
        expected: usize,
        /// Actual data points provided
        actual: usize,
    },

    /// Numerical computation error
    #[cfg_attr(feature = "std", error("Numerical error: {message}"))]
    Numerical {
        /// Error message
        message: String,
    },

    /// Dimension mismatch error
    #[cfg_attr(feature = "std", error("Dimension mismatch: expected {expected}, got {actual}"))]
    DimensionMismatch {
        /// Expected dimensions
        expected: usize,
        /// Actual dimensions
        actual: usize,
    },

    /// Hardware feature not available
    #[cfg_attr(feature = "std", error("Hardware feature not available: {feature}"))]
    HardwareNotAvailable {
        /// Missing hardware feature
        feature: String,
    },

    /// Analysis timeout
    #[cfg_attr(feature = "std", error("Analysis timeout: exceeded {timeout_ms}ms"))]
    Timeout {
        /// Timeout in milliseconds
        timeout_ms: u64,
    },

    /// Memory allocation error
    #[cfg_attr(feature = "std", error("Memory allocation failed: {message}"))]
    AllocationError {
        /// Error message
        message: String,
    },

    /// Not implemented
    #[cfg_attr(feature = "std", error("Not implemented: {feature}"))]
    NotImplemented {
        /// Feature not implemented
        feature: String,
    },

    /// Fusion error
    #[cfg_attr(feature = "std", error("Fusion error: {message}"))]
    FusionError {
        /// Error message
        message: String,
    },

    /// Invalid state
    #[cfg_attr(feature = "std", error("Invalid state: {message}"))]
    InvalidState {
        /// Error message
        message: String,
    },

    /// I/O error (when std feature is enabled)
    #[cfg(feature = "std")]
    #[cfg_attr(feature = "std", error("I/O error: {0}"))]
    Io(#[from] std::io::Error),

    /// Serialization error (when serde feature is enabled)
    #[cfg(all(feature = "std", feature = "serde"))]
    #[cfg_attr(all(feature = "std", feature = "serde"), error("Serialization error: {0}"))]
    Serialization(String),

    /// Custom error
    #[cfg_attr(feature = "std", error("{0}"))]
    Custom(String),
}

impl Clone for Error {
    fn clone(&self) -> Self {
        match self {
            Self::Config { message } => Self::Config {
                message: message.clone(),
            },
            Self::InvalidInput { message } => Self::InvalidInput {
                message: message.clone(),
            },
            Self::InsufficientData { expected, actual } => Self::InsufficientData {
                expected: *expected,
                actual: *actual,
            },
            Self::Numerical { message } => Self::Numerical {
                message: message.clone(),
            },
            Self::DimensionMismatch { expected, actual } => Self::DimensionMismatch {
                expected: *expected,
                actual: *actual,
            },
            Self::HardwareNotAvailable { feature } => Self::HardwareNotAvailable {
                feature: feature.clone(),
            },
            Self::Timeout { timeout_ms } => Self::Timeout {
                timeout_ms: *timeout_ms,
            },
            Self::AllocationError { message } => Self::AllocationError {
                message: message.clone(),
            },
            Self::NotImplemented { feature } => Self::NotImplemented {
                feature: feature.clone(),
            },
            Self::FusionError { message } => Self::FusionError {
                message: message.clone(),
            },
            Self::InvalidState { message } => Self::InvalidState {
                message: message.clone(),
            },
            #[cfg(feature = "std")]
            Self::Io(e) => Self::Custom(format!("I/O error: {}", e)),
            #[cfg(all(feature = "std", feature = "serde"))]
            Self::Serialization(msg) => Self::Serialization(msg.clone()),
            Self::Custom(msg) => Self::Custom(msg.clone()),
        }
    }
}

impl Error {
    /// Creates a configuration error
    pub fn config(message: impl Into<String>) -> Self {
        Self::Config {
            message: message.into(),
        }
    }

    /// Creates an invalid input error
    pub fn invalid_input(message: impl Into<String>) -> Self {
        Self::InvalidInput {
            message: message.into(),
        }
    }

    /// Creates an insufficient data error
    pub fn insufficient_data(expected: usize, actual: usize) -> Self {
        Self::InsufficientData { expected, actual }
    }

    /// Creates a numerical error
    pub fn numerical(message: impl Into<String>) -> Self {
        Self::Numerical {
            message: message.into(),
        }
    }

    /// Creates a dimension mismatch error
    pub fn dimension_mismatch(expected: usize, actual: usize) -> Self {
        Self::DimensionMismatch { expected, actual }
    }

    /// Creates a hardware not available error
    pub fn hardware_not_available(feature: impl Into<String>) -> Self {
        Self::HardwareNotAvailable {
            feature: feature.into(),
        }
    }

    /// Creates a timeout error
    pub fn timeout(timeout_ms: u64) -> Self {
        Self::Timeout { timeout_ms }
    }

    /// Creates an allocation error
    pub fn allocation(message: impl Into<String>) -> Self {
        Self::AllocationError {
            message: message.into(),
        }
    }

    /// Creates a not implemented error
    pub fn not_implemented(feature: impl Into<String>) -> Self {
        Self::NotImplemented {
            feature: feature.into(),
        }
    }

    /// Creates a fusion error
    pub fn fusion(message: impl Into<String>) -> Self {
        Self::FusionError {
            message: message.into(),
        }
    }

    /// Creates an invalid state error
    pub fn invalid_state(message: impl Into<String>) -> Self {
        Self::InvalidState {
            message: message.into(),
        }
    }

    /// Creates a custom error
    pub fn custom(message: impl Into<String>) -> Self {
        Self::Custom(message.into())
    }
}

// Manual Display implementation for no_std compatibility
#[cfg(not(feature = "std"))]
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Config { message } => write!(f, "Configuration error: {}", message),
            Error::InvalidInput { message } => write!(f, "Invalid input: {}", message),
            Error::InsufficientData { expected, actual } => {
                write!(f, "Insufficient data: expected at least {}, got {}", expected, actual)
            }
            Error::Numerical { message } => write!(f, "Numerical error: {}", message),
            Error::DimensionMismatch { expected, actual } => {
                write!(f, "Dimension mismatch: expected {}, got {}", expected, actual)
            }
            Error::HardwareNotAvailable { feature } => {
                write!(f, "Hardware feature not available: {}", feature)
            }
            Error::Timeout { timeout_ms } => write!(f, "Analysis timeout: exceeded {}ms", timeout_ms),
            Error::AllocationError { message } => write!(f, "Memory allocation failed: {}", message),
            Error::NotImplemented { feature } => write!(f, "Not implemented: {}", feature),
            Error::FusionError { message } => write!(f, "Fusion error: {}", message),
            Error::InvalidState { message } => write!(f, "Invalid state: {}", message),
            Error::Custom(message) => write!(f, "{}", message),
        }
    }
}

/// Extension trait for Result types
pub trait ResultExt<T> {
    /// Maps an error to a CDFA error with context
    fn context(self, context: &str) -> Result<T>;
}

impl<T, E> ResultExt<T> for core::result::Result<T, E>
where
    E: fmt::Display,
{
    fn context(self, context: &str) -> Result<T> {
        self.map_err(|e| Error::custom(format!("{}: {}", context, e)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let err = Error::config("invalid parameter");
        match err {
            Error::Config { message } => assert_eq!(message, "invalid parameter"),
            _ => panic!("Wrong error type"),
        }
    }

    #[test]
    fn test_insufficient_data_error() {
        let err = Error::insufficient_data(10, 5);
        match err {
            Error::InsufficientData { expected, actual } => {
                assert_eq!(expected, 10);
                assert_eq!(actual, 5);
            }
            _ => panic!("Wrong error type"),
        }
    }

    #[test]
    fn test_result_context() {
        let result: core::result::Result<i32, &str> = Err("base error");
        let cdfa_result = result.context("during processing");
        
        assert!(cdfa_result.is_err());
        match cdfa_result.unwrap_err() {
            Error::Custom(msg) => assert!(msg.contains("during processing")),
            _ => panic!("Wrong error type"),
        }
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_error_display() {
        let err = Error::dimension_mismatch(3, 5);
        let display = format!("{}", err);
        assert!(display.contains("expected 3"));
        assert!(display.contains("got 5"));
    }
}