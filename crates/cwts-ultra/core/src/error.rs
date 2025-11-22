//! Comprehensive error handling for CWTS Ultra
//!
//! This module provides structured, type-safe error handling replacing
//! all panic! and unwrap() patterns with proper Result<T, E> propagation.

use std::fmt;
use thiserror::Error;

/// Main error type for CWTS Ultra operations
#[derive(Debug, Error)]
pub enum CwtsError {
    /// Memory safety violations
    #[error("Memory safety violation: {message}")]
    MemorySafety { message: String },

    /// Concurrency and synchronization errors
    #[error("Concurrency error: {message}")]
    Concurrency { message: String },

    /// Order matching and trading logic errors
    #[error("Trading error: {message}")]
    Trading { message: String },

    /// System requirement failures
    #[error("System requirement failed: {message}")]
    SystemRequirement { message: String },

    /// Configuration errors
    #[error("Configuration error: {message}")]
    Configuration { message: String },

    /// Network and I/O errors
    #[error("Network error: {message}")]
    Network { message: String },

    /// Serialization/deserialization errors
    #[error("Serialization error: {message}")]
    Serialization { message: String },

    /// Validation failures
    #[error("Validation error: {message}")]
    Validation { message: String },

    /// Resource exhaustion
    #[error("Resource exhausted: {message}")]
    ResourceExhausted { message: String },

    /// External service errors
    #[error("External service error: {message}")]
    ExternalService { message: String },

    /// Recovery and rollback errors
    #[error("Recovery failed: {message}")]
    Recovery { message: String },

    /// Generic operation errors with context
    #[error("Operation failed: {operation} - {message}")]
    Operation { operation: String, message: String },
}

/// Error kind enumeration for programmatic error handling
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorKind {
    MemorySafety,
    Concurrency,
    Trading,
    SystemRequirement,
    Configuration,
    Network,
    Serialization,
    Validation,
    ResourceExhausted,
    ExternalService,
    Recovery,
    Operation,
}

impl CwtsError {
    /// Get the error kind for programmatic handling
    pub fn kind(&self) -> ErrorKind {
        match self {
            CwtsError::MemorySafety { .. } => ErrorKind::MemorySafety,
            CwtsError::Concurrency { .. } => ErrorKind::Concurrency,
            CwtsError::Trading { .. } => ErrorKind::Trading,
            CwtsError::SystemRequirement { .. } => ErrorKind::SystemRequirement,
            CwtsError::Configuration { .. } => ErrorKind::Configuration,
            CwtsError::Network { .. } => ErrorKind::Network,
            CwtsError::Serialization { .. } => ErrorKind::Serialization,
            CwtsError::Validation { .. } => ErrorKind::Validation,
            CwtsError::ResourceExhausted { .. } => ErrorKind::ResourceExhausted,
            CwtsError::ExternalService { .. } => ErrorKind::ExternalService,
            CwtsError::Recovery { .. } => ErrorKind::Recovery,
            CwtsError::Operation { .. } => ErrorKind::Operation,
        }
    }

    /// Check if error is recoverable
    pub fn is_recoverable(&self) -> bool {
        match self.kind() {
            ErrorKind::MemorySafety => false,      // Never recoverable
            ErrorKind::Concurrency => false,       // Usually indicates design flaw
            ErrorKind::Trading => true,            // Business logic errors can be retried
            ErrorKind::SystemRequirement => false, // Environment issue
            ErrorKind::Configuration => true,      // Can be fixed by reconfiguration
            ErrorKind::Network => true,            // Often transient
            ErrorKind::Serialization => true,      // Data format issues
            ErrorKind::Validation => true,         // Input validation
            ErrorKind::ResourceExhausted => true,  // May resolve over time
            ErrorKind::ExternalService => true,    // External system issues
            ErrorKind::Recovery => false,          // Recovery itself failed
            ErrorKind::Operation => true,          // Depends on context
        }
    }

    /// Get severity level for logging and monitoring
    pub fn severity(&self) -> Severity {
        match self.kind() {
            ErrorKind::MemorySafety => Severity::Critical,
            ErrorKind::Concurrency => Severity::Critical,
            ErrorKind::Trading => Severity::High,
            ErrorKind::SystemRequirement => Severity::Critical,
            ErrorKind::Configuration => Severity::Medium,
            ErrorKind::Network => Severity::Medium,
            ErrorKind::Serialization => Severity::Medium,
            ErrorKind::Validation => Severity::Low,
            ErrorKind::ResourceExhausted => Severity::High,
            ErrorKind::ExternalService => Severity::Medium,
            ErrorKind::Recovery => Severity::Critical,
            ErrorKind::Operation => Severity::Medium,
        }
    }

    // Constructor methods for common error types

    /// Memory safety violation error
    pub fn memory_safety(message: impl Into<String>) -> Self {
        Self::MemorySafety {
            message: message.into(),
        }
    }

    /// Concurrency error
    pub fn concurrency(message: impl Into<String>) -> Self {
        Self::Concurrency {
            message: message.into(),
        }
    }

    /// Trading logic error
    pub fn trading(message: impl Into<String>) -> Self {
        Self::Trading {
            message: message.into(),
        }
    }

    /// System requirement failure
    pub fn system_requirement_failed(message: impl Into<String>) -> Self {
        Self::SystemRequirement {
            message: message.into(),
        }
    }

    /// Configuration error
    pub fn configuration(message: impl Into<String>) -> Self {
        Self::Configuration {
            message: message.into(),
        }
    }

    /// Network error
    pub fn network(message: impl Into<String>) -> Self {
        Self::Network {
            message: message.into(),
        }
    }

    /// Serialization error
    pub fn serialization(message: impl Into<String>) -> Self {
        Self::Serialization {
            message: message.into(),
        }
    }

    /// Validation error
    pub fn validation(message: impl Into<String>) -> Self {
        Self::Validation {
            message: message.into(),
        }
    }

    /// Resource exhausted error
    pub fn resource_exhausted(message: impl Into<String>) -> Self {
        Self::ResourceExhausted {
            message: message.into(),
        }
    }

    /// External service error
    pub fn external_service(message: impl Into<String>) -> Self {
        Self::ExternalService {
            message: message.into(),
        }
    }

    /// Recovery failed error
    pub fn recovery_failed(message: impl Into<String>) -> Self {
        Self::Recovery {
            message: message.into(),
        }
    }

    /// Operation failed with context
    pub fn operation_failed(operation: impl Into<String>, message: impl Into<String>) -> Self {
        Self::Operation {
            operation: operation.into(),
            message: message.into(),
        }
    }
}

/// Error severity levels for monitoring and alerting
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    Low = 1,
    Medium = 2,
    High = 3,
    Critical = 4,
}

impl fmt::Display for Severity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Severity::Low => write!(f, "LOW"),
            Severity::Medium => write!(f, "MEDIUM"),
            Severity::High => write!(f, "HIGH"),
            Severity::Critical => write!(f, "CRITICAL"),
        }
    }
}

/// Helper trait for converting standard errors to CwtsError
pub trait IntoCwtsError<T> {
    fn into_cwts_error(self, context: &str) -> Result<T, CwtsError>;
}

impl<T, E: std::error::Error> IntoCwtsError<T> for Result<T, E> {
    fn into_cwts_error(self, context: &str) -> Result<T, CwtsError> {
        self.map_err(|e| CwtsError::operation_failed(context, e.to_string()))
    }
}

/// Macro for safe error propagation replacing unwrap()
#[macro_export]
macro_rules! safe_unwrap {
    ($result:expr, $context:expr) => {
        match $result {
            Ok(value) => value,
            Err(e) => {
                return Err(CwtsError::operation_failed($context, format!("{:?}", e)));
            }
        }
    };
}

/// Macro for safe option extraction replacing unwrap()
#[macro_export]
macro_rules! safe_extract {
    ($option:expr, $context:expr) => {
        match $option {
            Some(value) => value,
            None => {
                return Err(CwtsError::validation(format!(
                    "{}: value not found",
                    $context
                )));
            }
        }
    };
}

#[derive(Debug, Clone)]
pub enum OptimizationError {
    NoViableGenome,
    E2BValidationFailed(String),
    ResultParsingFailed(String),
}

impl std::fmt::Display for OptimizationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OptimizationError::NoViableGenome => write!(f, "No viable genome found"),
            OptimizationError::E2BValidationFailed(msg) => {
                write!(f, "E2B validation failed: {}", msg)
            }
            OptimizationError::ResultParsingFailed(msg) => {
                write!(f, "Result parsing failed: {}", msg)
            }
        }
    }
}

impl std::error::Error for OptimizationError {}

#[derive(Debug, Clone)]
pub enum LearningError {
    PipelineInitializationFailed(String),
    MetricsCollectionFailed(String),
    EventProcessingFailed(String),
    ValidationFailed(String),
    DeploymentFailed(String),
    ConfigurationError(String),
}

impl std::fmt::Display for LearningError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LearningError::PipelineInitializationFailed(msg) => {
                write!(f, "Pipeline initialization failed: {}", msg)
            }
            LearningError::MetricsCollectionFailed(msg) => {
                write!(f, "Metrics collection failed: {}", msg)
            }
            LearningError::EventProcessingFailed(msg) => {
                write!(f, "Event processing failed: {}", msg)
            }
            LearningError::ValidationFailed(msg) => write!(f, "Validation failed: {}", msg),
            LearningError::DeploymentFailed(msg) => write!(f, "Deployment failed: {}", msg),
            LearningError::ConfigurationError(msg) => write!(f, "Configuration error: {}", msg),
        }
    }
}

impl std::error::Error for LearningError {}

#[derive(Debug, Clone)]
pub enum AdaptationError {
    IntegrationFailed(String),
    ValidationFailed(String),
    DeploymentHealthCheckFailed(String),
    ProductionIntegrationFailed(String),
    TriggerSendFailed(String),
    ConfigurationError(String),
    OptimizationFailed(String),
}

impl std::fmt::Display for AdaptationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AdaptationError::IntegrationFailed(msg) => write!(f, "Integration failed: {}", msg),
            AdaptationError::ValidationFailed(msg) => write!(f, "Validation failed: {}", msg),
            AdaptationError::DeploymentHealthCheckFailed(msg) => {
                write!(f, "Deployment health check failed: {}", msg)
            }
            AdaptationError::ProductionIntegrationFailed(msg) => {
                write!(f, "Production integration failed: {}", msg)
            }
            AdaptationError::TriggerSendFailed(msg) => write!(f, "Trigger send failed: {}", msg),
            AdaptationError::ConfigurationError(msg) => write!(f, "Configuration error: {}", msg),
            AdaptationError::OptimizationFailed(msg) => write!(f, "Optimization failed: {}", msg),
        }
    }
}

impl std::error::Error for AdaptationError {}

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SystemError {
    ConfigurationError(String),
    DatabaseError(String),
    NetworkError(String),
    ValidationError(String),
    AuthenticationError(String),
    InternalError(String),
}

impl std::fmt::Display for SystemError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SystemError::ConfigurationError(msg) => write!(f, "Configuration error: {}", msg),
            SystemError::DatabaseError(msg) => write!(f, "Database error: {}", msg),
            SystemError::NetworkError(msg) => write!(f, "Network error: {}", msg),
            SystemError::ValidationError(msg) => write!(f, "Validation error: {}", msg),
            SystemError::AuthenticationError(msg) => write!(f, "Authentication error: {}", msg),
            SystemError::InternalError(msg) => write!(f, "Internal error: {}", msg),
        }
    }
}

impl std::error::Error for SystemError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_kinds() {
        let mem_error = CwtsError::memory_safety("test");
        assert_eq!(mem_error.kind(), ErrorKind::MemorySafety);
        assert!(!mem_error.is_recoverable());
        assert_eq!(mem_error.severity(), Severity::Critical);
    }

    #[test]
    fn test_error_constructors() {
        let trading_error = CwtsError::trading("invalid order");
        assert!(trading_error.is_recoverable());

        let config_error = CwtsError::configuration("missing parameter");
        assert_eq!(config_error.severity(), Severity::Medium);
    }

    #[test]
    fn test_severity_ordering() {
        assert!(Severity::Critical > Severity::High);
        assert!(Severity::High > Severity::Medium);
        assert!(Severity::Medium > Severity::Low);
    }
}
