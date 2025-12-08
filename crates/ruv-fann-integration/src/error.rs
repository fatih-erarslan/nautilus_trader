//! Error types for ruv_FANN Neural Divergent Integration
//!
//! This module provides comprehensive error handling for all aspects of the ruv_FANN
//! integration including neural network operations, GPU acceleration, parallel processing,
//! quantum ML bridges, and performance optimization.

use std::fmt;
use thiserror::Error;
use serde::{Serialize, Deserialize};

/// Result type for ruv_FANN operations
pub type RuvFannResult<T> = Result<T, RuvFannError>;

/// Comprehensive error types for ruv_FANN integration
#[derive(Error, Debug, Clone, Serialize, Deserialize)]
pub enum RuvFannError {
    /// Configuration errors
    #[error("Configuration error: {0}")]
    ConfigurationError(String),
    
    /// Neural network errors
    #[error("Neural network error: {0}")]
    NeuralNetworkError(String),
    
    /// Neural divergent processing errors
    #[error("Neural divergent error: {0}")]
    NeuralDivergentError(String),
    
    /// GPU acceleration errors
    #[error("GPU acceleration error: {0}")]
    GPUAccelerationError(String),
    
    /// Parallel processing errors
    #[error("Parallel processing error: {0}")]
    ParallelProcessingError(String),
    
    /// Quantum ML errors
    #[error("Quantum ML error: {0}")]
    QuantumMLError(String),
    
    /// Performance optimization errors
    #[error("Performance optimization error: {0}")]
    PerformanceError(String),
    
    /// Trading network errors
    #[error("Trading network error: {0}")]
    TradingNetworkError(String),
    
    /// Data flow errors
    #[error("Data flow error: {0}")]
    DataFlowError(String),
    
    /// Real-time inference errors
    #[error("Real-time inference error: {0}")]
    RealTimeInferenceError(String),
    
    /// Metrics and monitoring errors
    #[error("Metrics error: {0}")]
    MetricsError(String),
    
    /// Memory management errors
    #[error("Memory management error: {0}")]
    MemoryError(String),
    
    /// Resource allocation errors
    #[error("Resource allocation error: {0}")]
    ResourceError(String),
    
    /// Initialization errors
    #[error("Initialization error: {0}")]
    InitializationError(String),
    
    /// Connection errors
    #[error("Connection error: {0}")]
    ConnectionError(String),
    
    /// Validation errors
    #[error("Validation error: {0}")]
    ValidationError(String),
    
    /// Serialization errors
    #[error("Serialization error: {0}")]
    SerializationError(String),
    
    /// I/O errors
    #[error("I/O error: {0}")]
    IoError(String),
    
    /// Timeout errors
    #[error("Timeout error: operation timed out after {timeout_ms}ms: {operation}")]
    TimeoutError {
        operation: String,
        timeout_ms: u64,
    },
    
    /// Capacity errors
    #[error("Capacity error: {resource} at {current}/{maximum} capacity")]
    CapacityError {
        resource: String,
        current: usize,
        maximum: usize,
    },
    
    /// State errors
    #[error("State error: invalid state '{current}' for operation '{operation}', expected '{expected}'")]
    StateError {
        operation: String,
        current: String,
        expected: String,
    },
    
    /// Compatibility errors
    #[error("Compatibility error: {component} version {version} is not compatible with {requirement}")]
    CompatibilityError {
        component: String,
        version: String,
        requirement: String,
    },
    
    /// Security errors
    #[error("Security error: {0}")]
    SecurityError(String),
    
    /// Hardware errors
    #[error("Hardware error: {0}")]
    HardwareError(String),
    
    /// Network errors
    #[error("Network error: {0}")]
    NetworkError(String),
    
    /// Database errors
    #[error("Database error: {0}")]
    DatabaseError(String),
    
    /// Threading errors
    #[error("Threading error: {0}")]
    ThreadingError(String),
    
    /// Lock contention errors
    #[error("Lock contention error: failed to acquire {lock_type} lock within {timeout_ms}ms")]
    LockContentionError {
        lock_type: String,
        timeout_ms: u64,
    },
    
    /// Not initialized error
    #[error("Component not initialized: {component}")]
    NotInitialized { component: String },
    
    /// Already initialized error
    #[error("Component already initialized: {component}")]
    AlreadyInitialized { component: String },
    
    /// Not connected error
    #[error("Not connected to: {target}")]
    NotConnected { target: String },
    
    /// Already connected error
    #[error("Already connected to: {target}")]
    AlreadyConnected { target: String },
    
    /// Shutdown in progress error
    #[error("Shutdown in progress")]
    ShutdownInProgress,
    
    /// Service unavailable error
    #[error("Service unavailable: {service}")]
    ServiceUnavailable { service: String },
    
    /// Rate limit exceeded error
    #[error("Rate limit exceeded: {limit} operations per {window_ms}ms")]
    RateLimitExceeded {
        limit: usize,
        window_ms: u64,
    },
    
    /// Quota exceeded error
    #[error("Quota exceeded: {resource} quota of {quota} exceeded")]
    QuotaExceeded {
        resource: String,
        quota: usize,
    },
    
    /// Invalid input error
    #[error("Invalid input: {parameter} = {value}, {reason}")]
    InvalidInput {
        parameter: String,
        value: String,
        reason: String,
    },
    
    /// Missing dependency error
    #[error("Missing dependency: {dependency} required for {feature}")]
    MissingDependency {
        dependency: String,
        feature: String,
    },
    
    /// Version mismatch error
    #[error("Version mismatch: {component} version {found} does not match required {required}")]
    VersionMismatch {
        component: String,
        found: String,
        required: String,
    },
    
    /// Feature not supported error
    #[error("Feature not supported: {feature} is not supported on {platform}")]
    FeatureNotSupported {
        feature: String,
        platform: String,
    },
    
    /// License error
    #[error("License error: {0}")]
    LicenseError(String),
    
    /// Recovery error
    #[error("Recovery error: failed to recover from {original_error}: {recovery_error}")]
    RecoveryError {
        original_error: String,
        recovery_error: String,
    },
    
    /// Multiple errors
    #[error("Multiple errors occurred: {count} errors")]
    MultipleErrors {
        count: usize,
        errors: Vec<RuvFannError>,
    },
    
    /// Critical system error
    #[error("Critical system error: {0}")]
    CriticalError(String),
    
    /// Unexpected error
    #[error("Unexpected error: {0}")]
    UnexpectedError(String),
}

impl RuvFannError {
    /// Create a configuration error
    pub fn config_error<S: Into<String>>(message: S) -> Self {
        Self::ConfigurationError(message.into())
    }
    
    /// Create a neural network error
    pub fn neural_network_error<S: Into<String>>(message: S) -> Self {
        Self::NeuralNetworkError(message.into())
    }
    
    /// Create a neural divergent error
    pub fn neural_divergent_error<S: Into<String>>(message: S) -> Self {
        Self::NeuralDivergentError(message.into())
    }
    
    /// Create a GPU acceleration error
    pub fn gpu_error<S: Into<String>>(message: S) -> Self {
        Self::GPUAccelerationError(message.into())
    }
    
    /// Create a parallel processing error
    pub fn parallel_error<S: Into<String>>(message: S) -> Self {
        Self::ParallelProcessingError(message.into())
    }
    
    /// Create a quantum ML error
    pub fn quantum_error<S: Into<String>>(message: S) -> Self {
        Self::QuantumMLError(message.into())
    }
    
    /// Create a performance error
    pub fn performance_error<S: Into<String>>(message: S) -> Self {
        Self::PerformanceError(message.into())
    }
    
    /// Create a trading network error
    pub fn trading_error<S: Into<String>>(message: S) -> Self {
        Self::TradingNetworkError(message.into())
    }
    
    /// Create a data flow error
    pub fn data_flow_error<S: Into<String>>(message: S) -> Self {
        Self::DataFlowError(message.into())
    }
    
    /// Create a real-time inference error
    pub fn inference_error<S: Into<String>>(message: S) -> Self {
        Self::RealTimeInferenceError(message.into())
    }
    
    /// Create a metrics error
    pub fn metrics_error<S: Into<String>>(message: S) -> Self {
        Self::MetricsError(message.into())
    }
    
    /// Create a memory error
    pub fn memory_error<S: Into<String>>(message: S) -> Self {
        Self::MemoryError(message.into())
    }
    
    /// Create a resource error
    pub fn resource_error<S: Into<String>>(message: S) -> Self {
        Self::ResourceError(message.into())
    }
    
    /// Create an initialization error
    pub fn init_error<S: Into<String>>(message: S) -> Self {
        Self::InitializationError(message.into())
    }
    
    /// Create a connection error
    pub fn connection_error<S: Into<String>>(message: S) -> Self {
        Self::ConnectionError(message.into())
    }
    
    /// Create a validation error
    pub fn validation_error<S: Into<String>>(message: S) -> Self {
        Self::ValidationError(message.into())
    }
    
    /// Create a timeout error
    pub fn timeout_error<S: Into<String>>(operation: S, timeout_ms: u64) -> Self {
        Self::TimeoutError {
            operation: operation.into(),
            timeout_ms,
        }
    }
    
    /// Create a capacity error
    pub fn capacity_error<S: Into<String>>(resource: S, current: usize, maximum: usize) -> Self {
        Self::CapacityError {
            resource: resource.into(),
            current,
            maximum,
        }
    }
    
    /// Create a state error
    pub fn state_error<S: Into<String>>(operation: S, current: S, expected: S) -> Self {
        Self::StateError {
            operation: operation.into(),
            current: current.into(),
            expected: expected.into(),
        }
    }
    
    /// Create a compatibility error
    pub fn compatibility_error<S: Into<String>>(component: S, version: S, requirement: S) -> Self {
        Self::CompatibilityError {
            component: component.into(),
            version: version.into(),
            requirement: requirement.into(),
        }
    }
    
    /// Create a security error
    pub fn security_error<S: Into<String>>(message: S) -> Self {
        Self::SecurityError(message.into())
    }
    
    /// Create a hardware error
    pub fn hardware_error<S: Into<String>>(message: S) -> Self {
        Self::HardwareError(message.into())
    }
    
    /// Create a lock contention error
    pub fn lock_contention_error<S: Into<String>>(lock_type: S, timeout_ms: u64) -> Self {
        Self::LockContentionError {
            lock_type: lock_type.into(),
            timeout_ms,
        }
    }
    
    /// Create a not initialized error
    pub fn not_initialized<S: Into<String>>(component: S) -> Self {
        Self::NotInitialized {
            component: component.into(),
        }
    }
    
    /// Create an already initialized error
    pub fn already_initialized<S: Into<String>>(component: S) -> Self {
        Self::AlreadyInitialized {
            component: component.into(),
        }
    }
    
    /// Create a not connected error
    pub fn not_connected<S: Into<String>>(target: S) -> Self {
        Self::NotConnected {
            target: target.into(),
        }
    }
    
    /// Create an already connected error
    pub fn already_connected<S: Into<String>>(target: S) -> Self {
        Self::AlreadyConnected {
            target: target.into(),
        }
    }
    
    /// Create a service unavailable error
    pub fn service_unavailable<S: Into<String>>(service: S) -> Self {
        Self::ServiceUnavailable {
            service: service.into(),
        }
    }
    
    /// Create a rate limit exceeded error
    pub fn rate_limit_exceeded(limit: usize, window_ms: u64) -> Self {
        Self::RateLimitExceeded { limit, window_ms }
    }
    
    /// Create a quota exceeded error
    pub fn quota_exceeded<S: Into<String>>(resource: S, quota: usize) -> Self {
        Self::QuotaExceeded {
            resource: resource.into(),
            quota,
        }
    }
    
    /// Create an invalid input error
    pub fn invalid_input<S: Into<String>>(parameter: S, value: S, reason: S) -> Self {
        Self::InvalidInput {
            parameter: parameter.into(),
            value: value.into(),
            reason: reason.into(),
        }
    }
    
    /// Create a missing dependency error
    pub fn missing_dependency<S: Into<String>>(dependency: S, feature: S) -> Self {
        Self::MissingDependency {
            dependency: dependency.into(),
            feature: feature.into(),
        }
    }
    
    /// Create a version mismatch error
    pub fn version_mismatch<S: Into<String>>(component: S, found: S, required: S) -> Self {
        Self::VersionMismatch {
            component: component.into(),
            found: found.into(),
            required: required.into(),
        }
    }
    
    /// Create a feature not supported error
    pub fn feature_not_supported<S: Into<String>>(feature: S, platform: S) -> Self {
        Self::FeatureNotSupported {
            feature: feature.into(),
            platform: platform.into(),
        }
    }
    
    /// Create a recovery error
    pub fn recovery_error<S: Into<String>>(original_error: S, recovery_error: S) -> Self {
        Self::RecoveryError {
            original_error: original_error.into(),
            recovery_error: recovery_error.into(),
        }
    }
    
    /// Create a multiple errors error
    pub fn multiple_errors(errors: Vec<RuvFannError>) -> Self {
        let count = errors.len();
        Self::MultipleErrors { count, errors }
    }
    
    /// Create a critical error
    pub fn critical_error<S: Into<String>>(message: S) -> Self {
        Self::CriticalError(message.into())
    }
    
    /// Create an unexpected error
    pub fn unexpected_error<S: Into<String>>(message: S) -> Self {
        Self::UnexpectedError(message.into())
    }
    
    /// Check if this is a critical error that requires immediate attention
    pub fn is_critical(&self) -> bool {
        matches!(
            self,
            Self::CriticalError(_) |
            Self::SecurityError(_) |
            Self::HardwareError(_) |
            Self::MemoryError(_) |
            Self::ShutdownInProgress
        )
    }
    
    /// Check if this is a recoverable error
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            Self::TimeoutError { .. } |
            Self::NetworkError(_) |
            Self::LockContentionError { .. } |
            Self::RateLimitExceeded { .. } |
            Self::ServiceUnavailable { .. } |
            Self::CapacityError { .. }
        )
    }
    
    /// Check if this is a configuration error
    pub fn is_configuration_error(&self) -> bool {
        matches!(
            self,
            Self::ConfigurationError(_) |
            Self::ValidationError(_) |
            Self::InvalidInput { .. } |
            Self::MissingDependency { .. } |
            Self::VersionMismatch { .. } |
            Self::CompatibilityError { .. }
        )
    }
    
    /// Check if this is a neural network related error
    pub fn is_neural_network_error(&self) -> bool {
        matches!(
            self,
            Self::NeuralNetworkError(_) |
            Self::NeuralDivergentError(_) |
            Self::TradingNetworkError(_) |
            Self::RealTimeInferenceError(_)
        )
    }
    
    /// Check if this is a performance related error
    pub fn is_performance_error(&self) -> bool {
        matches!(
            self,
            Self::PerformanceError(_) |
            Self::GPUAccelerationError(_) |
            Self::ParallelProcessingError(_) |
            Self::MemoryError(_) |
            Self::TimeoutError { .. } |
            Self::CapacityError { .. }
        )
    }
    
    /// Get error category
    pub fn category(&self) -> ErrorCategory {
        match self {
            Self::ConfigurationError(_) |
            Self::ValidationError(_) |
            Self::InvalidInput { .. } => ErrorCategory::Configuration,
            
            Self::NeuralNetworkError(_) |
            Self::NeuralDivergentError(_) |
            Self::TradingNetworkError(_) => ErrorCategory::NeuralNetwork,
            
            Self::GPUAccelerationError(_) |
            Self::ParallelProcessingError(_) |
            Self::PerformanceError(_) => ErrorCategory::Performance,
            
            Self::QuantumMLError(_) => ErrorCategory::QuantumML,
            
            Self::DataFlowError(_) |
            Self::RealTimeInferenceError(_) => ErrorCategory::DataProcessing,
            
            Self::MemoryError(_) |
            Self::ResourceError(_) => ErrorCategory::Resource,
            
            Self::ConnectionError(_) |
            Self::NetworkError(_) => ErrorCategory::Network,
            
            Self::SecurityError(_) => ErrorCategory::Security,
            
            Self::HardwareError(_) => ErrorCategory::Hardware,
            
            Self::ThreadingError(_) |
            Self::LockContentionError { .. } => ErrorCategory::Threading,
            
            Self::TimeoutError { .. } => ErrorCategory::Timeout,
            
            Self::CriticalError(_) => ErrorCategory::Critical,
            
            _ => ErrorCategory::Other,
        }
    }
    
    /// Get error severity
    pub fn severity(&self) -> ErrorSeverity {
        match self {
            Self::CriticalError(_) |
            Self::SecurityError(_) |
            Self::ShutdownInProgress => ErrorSeverity::Critical,
            
            Self::HardwareError(_) |
            Self::MemoryError(_) |
            Self::InitializationError(_) => ErrorSeverity::High,
            
            Self::NeuralNetworkError(_) |
            Self::GPUAccelerationError(_) |
            Self::ParallelProcessingError(_) |
            Self::QuantumMLError(_) |
            Self::PerformanceError(_) => ErrorSeverity::Medium,
            
            Self::ConfigurationError(_) |
            Self::ValidationError(_) |
            Self::TimeoutError { .. } |
            Self::ConnectionError(_) => ErrorSeverity::Low,
            
            Self::MetricsError(_) |
            Self::SerializationError(_) => ErrorSeverity::Info,
            
            _ => ErrorSeverity::Low,
        }
    }
    
    /// Get suggested recovery action
    pub fn recovery_action(&self) -> RecoveryAction {
        match self {
            Self::TimeoutError { .. } => RecoveryAction::Retry,
            Self::NetworkError(_) => RecoveryAction::Retry,
            Self::LockContentionError { .. } => RecoveryAction::RetryWithBackoff,
            Self::RateLimitExceeded { .. } => RecoveryAction::RetryWithDelay,
            Self::ServiceUnavailable { .. } => RecoveryAction::RetryWithBackoff,
            Self::CapacityError { .. } => RecoveryAction::ReduceLoad,
            Self::MemoryError(_) => RecoveryAction::FreeResources,
            Self::ConfigurationError(_) => RecoveryAction::FixConfiguration,
            Self::ValidationError(_) => RecoveryAction::FixConfiguration,
            Self::CriticalError(_) => RecoveryAction::RestartSystem,
            Self::SecurityError(_) => RecoveryAction::RestartSystem,
            Self::HardwareError(_) => RecoveryAction::RestartSystem,
            _ => RecoveryAction::LogAndContinue,
        }
    }
    
    /// Convert to anyhow error
    pub fn to_anyhow(self) -> anyhow::Error {
        anyhow::Error::new(self)
    }
    
    /// Create error context
    pub fn with_context<S: Into<String>>(self, context: S) -> Self {
        let context_str = context.into();
        match self {
            Self::ConfigurationError(msg) => Self::ConfigurationError(format!("{}: {}", context_str, msg)),
            Self::NeuralNetworkError(msg) => Self::NeuralNetworkError(format!("{}: {}", context_str, msg)),
            Self::NeuralDivergentError(msg) => Self::NeuralDivergentError(format!("{}: {}", context_str, msg)),
            Self::GPUAccelerationError(msg) => Self::GPUAccelerationError(format!("{}: {}", context_str, msg)),
            Self::ParallelProcessingError(msg) => Self::ParallelProcessingError(format!("{}: {}", context_str, msg)),
            Self::QuantumMLError(msg) => Self::QuantumMLError(format!("{}: {}", context_str, msg)),
            Self::PerformanceError(msg) => Self::PerformanceError(format!("{}: {}", context_str, msg)),
            Self::TradingNetworkError(msg) => Self::TradingNetworkError(format!("{}: {}", context_str, msg)),
            Self::DataFlowError(msg) => Self::DataFlowError(format!("{}: {}", context_str, msg)),
            Self::RealTimeInferenceError(msg) => Self::RealTimeInferenceError(format!("{}: {}", context_str, msg)),
            Self::MetricsError(msg) => Self::MetricsError(format!("{}: {}", context_str, msg)),
            Self::MemoryError(msg) => Self::MemoryError(format!("{}: {}", context_str, msg)),
            Self::ResourceError(msg) => Self::ResourceError(format!("{}: {}", context_str, msg)),
            Self::InitializationError(msg) => Self::InitializationError(format!("{}: {}", context_str, msg)),
            Self::ConnectionError(msg) => Self::ConnectionError(format!("{}: {}", context_str, msg)),
            Self::ValidationError(msg) => Self::ValidationError(format!("{}: {}", context_str, msg)),
            Self::SerializationError(msg) => Self::SerializationError(format!("{}: {}", context_str, msg)),
            Self::IoError(msg) => Self::IoError(format!("{}: {}", context_str, msg)),
            Self::SecurityError(msg) => Self::SecurityError(format!("{}: {}", context_str, msg)),
            Self::HardwareError(msg) => Self::HardwareError(format!("{}: {}", context_str, msg)),
            Self::NetworkError(msg) => Self::NetworkError(format!("{}: {}", context_str, msg)),
            Self::DatabaseError(msg) => Self::DatabaseError(format!("{}: {}", context_str, msg)),
            Self::ThreadingError(msg) => Self::ThreadingError(format!("{}: {}", context_str, msg)),
            Self::LicenseError(msg) => Self::LicenseError(format!("{}: {}", context_str, msg)),
            Self::CriticalError(msg) => Self::CriticalError(format!("{}: {}", context_str, msg)),
            Self::UnexpectedError(msg) => Self::UnexpectedError(format!("{}: {}", context_str, msg)),
            other => other, // For structured errors, don't modify
        }
    }
}

/// Error categories for classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ErrorCategory {
    /// Configuration related errors
    Configuration,
    /// Neural network related errors
    NeuralNetwork,
    /// Performance related errors
    Performance,
    /// Quantum ML related errors
    QuantumML,
    /// Data processing related errors
    DataProcessing,
    /// Resource related errors
    Resource,
    /// Network related errors
    Network,
    /// Security related errors
    Security,
    /// Hardware related errors
    Hardware,
    /// Threading related errors
    Threading,
    /// Timeout related errors
    Timeout,
    /// Critical system errors
    Critical,
    /// Other errors
    Other,
}

impl fmt::Display for ErrorCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Configuration => write!(f, "Configuration"),
            Self::NeuralNetwork => write!(f, "Neural Network"),
            Self::Performance => write!(f, "Performance"),
            Self::QuantumML => write!(f, "Quantum ML"),
            Self::DataProcessing => write!(f, "Data Processing"),
            Self::Resource => write!(f, "Resource"),
            Self::Network => write!(f, "Network"),
            Self::Security => write!(f, "Security"),
            Self::Hardware => write!(f, "Hardware"),
            Self::Threading => write!(f, "Threading"),
            Self::Timeout => write!(f, "Timeout"),
            Self::Critical => write!(f, "Critical"),
            Self::Other => write!(f, "Other"),
        }
    }
}

/// Error severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ErrorSeverity {
    /// Informational - for logging purposes
    Info,
    /// Low severity - minor issues
    Low,
    /// Medium severity - significant issues
    Medium,
    /// High severity - serious issues
    High,
    /// Critical severity - system threatening issues
    Critical,
}

impl fmt::Display for ErrorSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Info => write!(f, "INFO"),
            Self::Low => write!(f, "LOW"),
            Self::Medium => write!(f, "MEDIUM"),
            Self::High => write!(f, "HIGH"),
            Self::Critical => write!(f, "CRITICAL"),
        }
    }
}

/// Suggested recovery actions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecoveryAction {
    /// Retry the operation immediately
    Retry,
    /// Retry with exponential backoff
    RetryWithBackoff,
    /// Retry after a fixed delay
    RetryWithDelay,
    /// Reduce system load
    ReduceLoad,
    /// Free system resources
    FreeResources,
    /// Fix configuration and retry
    FixConfiguration,
    /// Restart the system
    RestartSystem,
    /// Log error and continue
    LogAndContinue,
    /// Fail fast - propagate error immediately
    FailFast,
    /// Enter safe mode
    SafeMode,
}

impl fmt::Display for RecoveryAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Retry => write!(f, "Retry"),
            Self::RetryWithBackoff => write!(f, "Retry with backoff"),
            Self::RetryWithDelay => write!(f, "Retry with delay"),
            Self::ReduceLoad => write!(f, "Reduce load"),
            Self::FreeResources => write!(f, "Free resources"),
            Self::FixConfiguration => write!(f, "Fix configuration"),
            Self::RestartSystem => write!(f, "Restart system"),
            Self::LogAndContinue => write!(f, "Log and continue"),
            Self::FailFast => write!(f, "Fail fast"),
            Self::SafeMode => write!(f, "Enter safe mode"),
        }
    }
}

/// Error statistics for monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorStatistics {
    /// Total number of errors
    pub total_errors: u64,
    /// Errors by category
    pub errors_by_category: std::collections::HashMap<ErrorCategory, u64>,
    /// Errors by severity
    pub errors_by_severity: std::collections::HashMap<ErrorSeverity, u64>,
    /// Recent error rate (errors per second)
    pub recent_error_rate: f64,
    /// Average time between errors
    pub avg_time_between_errors: std::time::Duration,
    /// Most common error
    pub most_common_error: Option<String>,
    /// Last update timestamp
    pub last_updated: chrono::DateTime<chrono::Utc>,
}

impl Default for ErrorStatistics {
    fn default() -> Self {
        Self {
            total_errors: 0,
            errors_by_category: std::collections::HashMap::new(),
            errors_by_severity: std::collections::HashMap::new(),
            recent_error_rate: 0.0,
            avg_time_between_errors: std::time::Duration::from_secs(0),
            most_common_error: None,
            last_updated: chrono::Utc::now(),
        }
    }
}

/// Error context for enhanced debugging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorContext {
    /// Component that generated the error
    pub component: String,
    /// Operation that was being performed
    pub operation: String,
    /// Thread ID
    pub thread_id: String,
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Stack trace (if available)
    pub stack_trace: Option<String>,
    /// Additional metadata
    pub metadata: std::collections::HashMap<String, serde_json::Value>,
}

impl Default for ErrorContext {
    fn default() -> Self {
        Self {
            component: "unknown".to_string(),
            operation: "unknown".to_string(),
            thread_id: format!("{:?}", std::thread::current().id()),
            timestamp: chrono::Utc::now(),
            stack_trace: None,
            metadata: std::collections::HashMap::new(),
        }
    }
}

/// Enhanced error with context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextualError {
    /// The underlying error
    pub error: RuvFannError,
    /// Error context
    pub context: ErrorContext,
    /// Error ID for tracking
    pub error_id: uuid::Uuid,
}

impl ContextualError {
    /// Create a new contextual error
    pub fn new(error: RuvFannError, context: ErrorContext) -> Self {
        Self {
            error,
            context,
            error_id: uuid::Uuid::new_v4(),
        }
    }
    
    /// Create with minimal context
    pub fn with_component<S: Into<String>>(error: RuvFannError, component: S) -> Self {
        let mut context = ErrorContext::default();
        context.component = component.into();
        Self::new(error, context)
    }
    
    /// Add metadata to the error context
    pub fn with_metadata<K, V>(mut self, key: K, value: V) -> Self
    where
        K: Into<String>,
        V: serde::Serialize,
    {
        let value_json = serde_json::to_value(value).unwrap_or(serde_json::Value::Null);
        self.context.metadata.insert(key.into(), value_json);
        self
    }
}

impl fmt::Display for ContextualError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{}] {}: {} (component: {}, operation: {}, thread: {})",
            self.error_id,
            self.error.category(),
            self.error,
            self.context.component,
            self.context.operation,
            self.context.thread_id
        )
    }
}

impl std::error::Error for ContextualError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}

// Conversion implementations for standard library errors
impl From<std::io::Error> for RuvFannError {
    fn from(err: std::io::Error) -> Self {
        Self::IoError(err.to_string())
    }
}

impl From<serde_json::Error> for RuvFannError {
    fn from(err: serde_json::Error) -> Self {
        Self::SerializationError(err.to_string())
    }
}

impl From<std::num::ParseIntError> for RuvFannError {
    fn from(err: std::num::ParseIntError) -> Self {
        Self::ValidationError(format!("Integer parsing error: {}", err))
    }
}

impl From<std::num::ParseFloatError> for RuvFannError {
    fn from(err: std::num::ParseFloatError) -> Self {
        Self::ValidationError(format!("Float parsing error: {}", err))
    }
}

impl From<uuid::Error> for RuvFannError {
    fn from(err: uuid::Error) -> Self {
        Self::ValidationError(format!("UUID error: {}", err))
    }
}

impl From<chrono::ParseError> for RuvFannError {
    fn from(err: chrono::ParseError) -> Self {
        Self::ValidationError(format!("DateTime parsing error: {}", err))
    }
}

// Integration with anyhow
impl From<anyhow::Error> for RuvFannError {
    fn from(err: anyhow::Error) -> Self {
        Self::UnexpectedError(err.to_string())
    }
}

impl From<RuvFannError> for anyhow::Error {
    fn from(err: RuvFannError) -> Self {
        anyhow::Error::new(err)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_error_creation() {
        let error = RuvFannError::config_error("Test configuration error");
        assert!(matches!(error, RuvFannError::ConfigurationError(_)));
        assert!(error.is_configuration_error());
        assert_eq!(error.category(), ErrorCategory::Configuration);
    }
    
    #[test]
    fn test_error_severity() {
        let critical_error = RuvFannError::critical_error("Critical failure");
        assert_eq!(critical_error.severity(), ErrorSeverity::Critical);
        assert!(critical_error.is_critical());
        
        let config_error = RuvFannError::config_error("Config issue");
        assert_eq!(config_error.severity(), ErrorSeverity::Low);
        assert!(!config_error.is_critical());
    }
    
    #[test]
    fn test_error_recovery_action() {
        let timeout_error = RuvFannError::timeout_error("test_op", 1000);
        assert_eq!(timeout_error.recovery_action(), RecoveryAction::Retry);
        assert!(timeout_error.is_recoverable());
        
        let critical_error = RuvFannError::critical_error("Critical failure");
        assert_eq!(critical_error.recovery_action(), RecoveryAction::RestartSystem);
        assert!(!critical_error.is_recoverable());
    }
    
    #[test]
    fn test_error_context() {
        let base_error = RuvFannError::neural_network_error("Neural error");
        let contextual_error = base_error.with_context("During training phase");
        
        if let RuvFannError::NeuralNetworkError(msg) = contextual_error {
            assert!(msg.contains("During training phase"));
            assert!(msg.contains("Neural error"));
        } else {
            panic!("Expected NeuralNetworkError");
        }
    }
    
    #[test]
    fn test_contextual_error() {
        let error = RuvFannError::gpu_error("GPU initialization failed");
        let contextual = ContextualError::with_component(error, "gpu_accelerator")
            .with_metadata("gpu_model", "RTX 4090")
            .with_metadata("driver_version", "535.104.05");
        
        assert_eq!(contextual.context.component, "gpu_accelerator");
        assert!(contextual.context.metadata.contains_key("gpu_model"));
        assert!(contextual.context.metadata.contains_key("driver_version"));
    }
    
    #[test]
    fn test_multiple_errors() {
        let errors = vec![
            RuvFannError::config_error("Config error 1"),
            RuvFannError::config_error("Config error 2"),
            RuvFannError::neural_network_error("Neural error"),
        ];
        
        let multiple_error = RuvFannError::multiple_errors(errors.clone());
        
        if let RuvFannError::MultipleErrors { count, errors: inner_errors } = multiple_error {
            assert_eq!(count, 3);
            assert_eq!(inner_errors.len(), 3);
        } else {
            panic!("Expected MultipleErrors");
        }
    }
    
    #[test]
    fn test_error_serialization() {
        let error = RuvFannError::timeout_error("test_operation", 5000);
        let serialized = serde_json::to_string(&error).unwrap();
        let deserialized: RuvFannError = serde_json::from_str(&serialized).unwrap();
        
        if let (
            RuvFannError::TimeoutError { operation: op1, timeout_ms: t1 },
            RuvFannError::TimeoutError { operation: op2, timeout_ms: t2 }
        ) = (&error, &deserialized) {
            assert_eq!(op1, op2);
            assert_eq!(t1, t2);
        } else {
            panic!("Serialization/deserialization failed");
        }
    }
    
    #[test]
    fn test_error_statistics() {
        let mut stats = ErrorStatistics::default();
        assert_eq!(stats.total_errors, 0);
        assert!(stats.errors_by_category.is_empty());
        assert!(stats.errors_by_severity.is_empty());
    }
}