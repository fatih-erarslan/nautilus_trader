//! Error handling for MCP orchestration system.

use std::fmt;
use thiserror::Error;

/// Result type for MCP orchestration operations
pub type Result<T> = std::result::Result<T, OrchestrationError>;

/// Comprehensive error types for the MCP orchestration system
#[derive(Error, Debug)]
pub enum OrchestrationError {
    /// Configuration errors
    #[error("Configuration error: {message}")]
    Config { message: String },
    
    /// Agent-related errors
    #[error("Agent error: {message}")]
    Agent { message: String },
    
    /// Communication errors
    #[error("Communication error: {message}")]
    Communication { message: String },
    
    /// Task processing errors
    #[error("Task error: {message}")]
    Task { message: String },
    
    /// Memory management errors
    #[error("Memory error: {message}")]
    Memory { message: String },
    
    /// Load balancing errors
    #[error("Load balancing error: {message}")]
    LoadBalancing { message: String },
    
    /// Health monitoring errors
    #[error("Health monitoring error: {message}")]
    HealthMonitoring { message: String },
    
    /// Recovery errors
    #[error("Recovery error: {message}")]
    Recovery { message: String },
    
    /// Metrics collection errors
    #[error("Metrics error: {message}")]
    Metrics { message: String },
    
    /// Serialization/deserialization errors
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    
    /// I/O errors
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    
    /// Network errors
    #[error("Network error: {0}")]
    Network(#[from] Box<dyn std::error::Error + Send + Sync>),
    
    /// Timeout errors
    #[error("Timeout error: operation timed out after {timeout_ms}ms")]
    Timeout { timeout_ms: u64 },
    
    /// Resource exhaustion errors
    #[error("Resource exhaustion: {resource} limit exceeded")]
    ResourceExhausted { resource: String },
    
    /// Invalid state errors
    #[error("Invalid state: {message}")]
    InvalidState { message: String },
    
    /// Not found errors
    #[error("Not found: {resource}")]
    NotFound { resource: String },
    
    /// Already exists errors
    #[error("Already exists: {resource}")]
    AlreadyExists { resource: String },
    
    /// Permission denied errors
    #[error("Permission denied: {message}")]
    PermissionDenied { message: String },
    
    /// Internal system errors
    #[error("Internal error: {message}")]
    Internal { message: String },
}

impl OrchestrationError {
    /// Create a configuration error
    pub fn config<S: Into<String>>(message: S) -> Self {
        Self::Config {
            message: message.into(),
        }
    }
    
    /// Create an agent error
    pub fn agent<S: Into<String>>(message: S) -> Self {
        Self::Agent {
            message: message.into(),
        }
    }
    
    /// Create a communication error
    pub fn communication<S: Into<String>>(message: S) -> Self {
        Self::Communication {
            message: message.into(),
        }
    }
    
    /// Create a task error
    pub fn task<S: Into<String>>(message: S) -> Self {
        Self::Task {
            message: message.into(),
        }
    }
    
    /// Create a memory error
    pub fn memory<S: Into<String>>(message: S) -> Self {
        Self::Memory {
            message: message.into(),
        }
    }
    
    /// Create a load balancing error
    pub fn load_balancing<S: Into<String>>(message: S) -> Self {
        Self::LoadBalancing {
            message: message.into(),
        }
    }
    
    /// Create a health monitoring error
    pub fn health_monitoring<S: Into<String>>(message: S) -> Self {
        Self::HealthMonitoring {
            message: message.into(),
        }
    }
    
    /// Create a recovery error
    pub fn recovery<S: Into<String>>(message: S) -> Self {
        Self::Recovery {
            message: message.into(),
        }
    }
    
    /// Create a metrics error
    pub fn metrics<S: Into<String>>(message: S) -> Self {
        Self::Metrics {
            message: message.into(),
        }
    }
    
    /// Create a timeout error
    pub fn timeout(timeout_ms: u64) -> Self {
        Self::Timeout { timeout_ms }
    }
    
    /// Create a resource exhausted error
    pub fn resource_exhausted<S: Into<String>>(resource: S) -> Self {
        Self::ResourceExhausted {
            resource: resource.into(),
        }
    }
    
    /// Create an invalid state error
    pub fn invalid_state<S: Into<String>>(message: S) -> Self {
        Self::InvalidState {
            message: message.into(),
        }
    }
    
    /// Create a not found error
    pub fn not_found<S: Into<String>>(resource: S) -> Self {
        Self::NotFound {
            resource: resource.into(),
        }
    }
    
    /// Create an already exists error
    pub fn already_exists<S: Into<String>>(resource: S) -> Self {
        Self::AlreadyExists {
            resource: resource.into(),
        }
    }
    
    /// Create a permission denied error
    pub fn permission_denied<S: Into<String>>(message: S) -> Self {
        Self::PermissionDenied {
            message: message.into(),
        }
    }
    
    /// Create an internal error
    pub fn internal<S: Into<String>>(message: S) -> Self {
        Self::Internal {
            message: message.into(),
        }
    }
    
    /// Check if this error is retryable
    pub fn is_retryable(&self) -> bool {
        match self {
            Self::Config { .. } => false,
            Self::Agent { .. } => true,
            Self::Communication { .. } => true,
            Self::Task { .. } => true,
            Self::Memory { .. } => true,
            Self::LoadBalancing { .. } => true,
            Self::HealthMonitoring { .. } => true,
            Self::Recovery { .. } => true,
            Self::Metrics { .. } => true,
            Self::Serialization(..) => false,
            Self::Io(..) => true,
            Self::Network(..) => true,
            Self::Timeout { .. } => true,
            Self::ResourceExhausted { .. } => true,
            Self::InvalidState { .. } => false,
            Self::NotFound { .. } => false,
            Self::AlreadyExists { .. } => false,
            Self::PermissionDenied { .. } => false,
            Self::Internal { .. } => false,
        }
    }
    
    /// Check if this error is critical
    pub fn is_critical(&self) -> bool {
        match self {
            Self::Config { .. } => true,
            Self::Agent { .. } => false,
            Self::Communication { .. } => false,
            Self::Task { .. } => false,
            Self::Memory { .. } => true,
            Self::LoadBalancing { .. } => false,
            Self::HealthMonitoring { .. } => false,
            Self::Recovery { .. } => true,
            Self::Metrics { .. } => false,
            Self::Serialization(..) => false,
            Self::Io(..) => false,
            Self::Network(..) => false,
            Self::Timeout { .. } => false,
            Self::ResourceExhausted { .. } => true,
            Self::InvalidState { .. } => true,
            Self::NotFound { .. } => false,
            Self::AlreadyExists { .. } => false,
            Self::PermissionDenied { .. } => true,
            Self::Internal { .. } => true,
        }
    }
    
    /// Get the error category
    pub fn category(&self) -> ErrorCategory {
        match self {
            Self::Config { .. } => ErrorCategory::Configuration,
            Self::Agent { .. } => ErrorCategory::Agent,
            Self::Communication { .. } => ErrorCategory::Communication,
            Self::Task { .. } => ErrorCategory::Task,
            Self::Memory { .. } => ErrorCategory::Memory,
            Self::LoadBalancing { .. } => ErrorCategory::LoadBalancing,
            Self::HealthMonitoring { .. } => ErrorCategory::HealthMonitoring,
            Self::Recovery { .. } => ErrorCategory::Recovery,
            Self::Metrics { .. } => ErrorCategory::Metrics,
            Self::Serialization(..) => ErrorCategory::Serialization,
            Self::Io(..) => ErrorCategory::Io,
            Self::Network(..) => ErrorCategory::Network,
            Self::Timeout { .. } => ErrorCategory::Timeout,
            Self::ResourceExhausted { .. } => ErrorCategory::Resource,
            Self::InvalidState { .. } => ErrorCategory::State,
            Self::NotFound { .. } => ErrorCategory::NotFound,
            Self::AlreadyExists { .. } => ErrorCategory::AlreadyExists,
            Self::PermissionDenied { .. } => ErrorCategory::Permission,
            Self::Internal { .. } => ErrorCategory::Internal,
        }
    }
}

/// Error categories for classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorCategory {
    Configuration,
    Agent,
    Communication,
    Task,
    Memory,
    LoadBalancing,
    HealthMonitoring,
    Recovery,
    Metrics,
    Serialization,
    Io,
    Network,
    Timeout,
    Resource,
    State,
    NotFound,
    AlreadyExists,
    Permission,
    Internal,
}

impl fmt::Display for ErrorCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Configuration => write!(f, "Configuration"),
            Self::Agent => write!(f, "Agent"),
            Self::Communication => write!(f, "Communication"),
            Self::Task => write!(f, "Task"),
            Self::Memory => write!(f, "Memory"),
            Self::LoadBalancing => write!(f, "LoadBalancing"),
            Self::HealthMonitoring => write!(f, "HealthMonitoring"),
            Self::Recovery => write!(f, "Recovery"),
            Self::Metrics => write!(f, "Metrics"),
            Self::Serialization => write!(f, "Serialization"),
            Self::Io => write!(f, "Io"),
            Self::Network => write!(f, "Network"),
            Self::Timeout => write!(f, "Timeout"),
            Self::Resource => write!(f, "Resource"),
            Self::State => write!(f, "State"),
            Self::NotFound => write!(f, "NotFound"),
            Self::AlreadyExists => write!(f, "AlreadyExists"),
            Self::Permission => write!(f, "Permission"),
            Self::Internal => write!(f, "Internal"),
        }
    }
}

/// Error context for debugging
#[derive(Debug, Clone)]
pub struct ErrorContext {
    pub operation: String,
    pub component: String,
    pub timestamp: std::time::SystemTime,
    pub thread_id: String,
    pub additional_info: std::collections::HashMap<String, String>,
}

impl ErrorContext {
    /// Create a new error context
    pub fn new(operation: &str, component: &str) -> Self {
        Self {
            operation: operation.to_string(),
            component: component.to_string(),
            timestamp: std::time::SystemTime::now(),
            thread_id: format!("{:?}", std::thread::current().id()),
            additional_info: std::collections::HashMap::new(),
        }
    }
    
    /// Add additional information to the context
    pub fn with_info<K, V>(mut self, key: K, value: V) -> Self
    where
        K: Into<String>,
        V: Into<String>,
    {
        self.additional_info.insert(key.into(), value.into());
        self
    }
}

/// Macro for creating context-aware errors
#[macro_export]
macro_rules! error_with_context {
    ($error:expr, $operation:expr, $component:expr) => {
        {
            let context = $crate::error::ErrorContext::new($operation, $component);
            tracing::error!(
                operation = %context.operation,
                component = %context.component,
                thread_id = %context.thread_id,
                error = %$error,
                "Error occurred"
            );
            $error
        }
    };
    ($error:expr, $operation:expr, $component:expr, $($key:expr => $value:expr),*) => {
        {
            let mut context = $crate::error::ErrorContext::new($operation, $component);
            $(
                context = context.with_info($key, $value);
            )*
            tracing::error!(
                operation = %context.operation,
                component = %context.component,
                thread_id = %context.thread_id,
                error = %$error,
                additional_info = ?context.additional_info,
                "Error occurred"
            );
            $error
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_error_creation() {
        let err = OrchestrationError::config("test error");
        assert_eq!(err.category(), ErrorCategory::Configuration);
        assert!(!err.is_retryable());
        assert!(err.is_critical());
    }
    
    #[test]
    fn test_error_retryability() {
        assert!(OrchestrationError::communication("test").is_retryable());
        assert!(!OrchestrationError::config("test").is_retryable());
        assert!(!OrchestrationError::invalid_state("test").is_retryable());
    }
    
    #[test]
    fn test_error_criticality() {
        assert!(OrchestrationError::config("test").is_critical());
        assert!(OrchestrationError::memory("test").is_critical());
        assert!(!OrchestrationError::communication("test").is_critical());
        assert!(!OrchestrationError::task("test").is_critical());
    }
    
    #[test]
    fn test_error_context() {
        let ctx = ErrorContext::new("test_op", "test_component")
            .with_info("key1", "value1")
            .with_info("key2", "value2");
        
        assert_eq!(ctx.operation, "test_op");
        assert_eq!(ctx.component, "test_component");
        assert_eq!(ctx.additional_info.get("key1").unwrap(), "value1");
        assert_eq!(ctx.additional_info.get("key2").unwrap(), "value2");
    }
}