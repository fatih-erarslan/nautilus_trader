//! Error handling for the PADS system
//!
//! This module provides comprehensive error handling for all PADS components,
//! ensuring robust operation and clear error reporting.

use thiserror::Error;
use std::fmt;

/// Result type for PADS operations
pub type PadsResult<T> = Result<T, PadsError>;

/// Main error type for the PADS system
#[derive(Error, Debug)]
pub enum PadsError {
    /// Configuration errors
    #[error("Configuration error: {message}")]
    Configuration { message: String },

    /// Agent-related errors
    #[error("Agent error ({agent}): {message}")]
    Agent { agent: String, message: String },

    /// Quantum computation errors
    #[error("Quantum error: {message}")]
    Quantum { message: String },

    /// Board consensus errors
    #[error("Board consensus error: {message}")]
    BoardConsensus { message: String },

    /// Risk management errors
    #[error("Risk management error: {message}")]
    RiskManagement { message: String },

    /// Decision strategy errors
    #[error("Decision strategy error ({strategy}): {message}")]
    DecisionStrategy { strategy: String, message: String },

    /// Analyzer/detector errors
    #[error("Analyzer error ({analyzer}): {message}")]
    Analyzer { analyzer: String, message: String },

    /// Panarchy system errors
    #[error("Panarchy system error: {message}")]
    Panarchy { message: String },

    /// Hardware acceleration errors
    #[error("Hardware acceleration error: {message}")]
    Hardware { message: String },

    /// Memory management errors
    #[error("Memory error: {message}")]
    Memory { message: String },

    /// Network/communication errors
    #[error("Network error: {message}")]
    Network { message: String },

    /// Serialization/deserialization errors
    #[error("Serialization error: {message}")]
    Serialization { message: String },

    /// Database/storage errors
    #[error("Storage error: {message}")]
    Storage { message: String },

    /// Performance/latency errors
    #[error("Performance error: {message}")]
    Performance { message: String },

    /// Validation errors
    #[error("Validation error: {message}")]
    Validation { message: String },

    /// Timeout errors
    #[error("Timeout error: {message}")]
    Timeout { message: String },

    /// Resource exhaustion errors
    #[error("Resource exhaustion: {message}")]
    ResourceExhaustion { message: String },

    /// Python integration errors
    #[cfg(feature = "python-integration")]
    #[error("Python integration error: {message}")]
    Python { message: String },

    /// External service errors
    #[error("External service error ({service}): {message}")]
    ExternalService { service: String, message: String },

    /// Generic internal error
    #[error("Internal error: {message}")]
    Internal { message: String },

    /// IO errors
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON serialization errors
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// TOML parsing errors
    #[error("TOML error: {0}")]
    Toml(#[from] toml::de::Error),

    /// HTTP request errors
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    /// Database errors
    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),

    /// Tokio join errors
    #[error("Tokio join error: {0}")]
    TokioJoin(#[from] tokio::task::JoinError),

    /// Chrono parsing errors
    #[error("Chrono parse error: {0}")]
    ChronoParse(#[from] chrono::ParseError),

    /// UUID errors
    #[error("UUID error: {0}")]
    Uuid(#[from] uuid::Error),

    /// Generic error for any other error type
    #[error("Generic error: {0}")]
    Generic(#[from] Box<dyn std::error::Error + Send + Sync>),
}

impl PadsError {
    /// Create a configuration error
    pub fn configuration(message: impl Into<String>) -> Self {
        Self::Configuration {
            message: message.into(),
        }
    }

    /// Create an agent error
    pub fn agent(agent: impl Into<String>, message: impl Into<String>) -> Self {
        Self::Agent {
            agent: agent.into(),
            message: message.into(),
        }
    }

    /// Create a quantum error
    pub fn quantum(message: impl Into<String>) -> Self {
        Self::Quantum {
            message: message.into(),
        }
    }

    /// Create a board consensus error
    pub fn board_consensus(message: impl Into<String>) -> Self {
        Self::BoardConsensus {
            message: message.into(),
        }
    }

    /// Create a risk management error
    pub fn risk_management(message: impl Into<String>) -> Self {
        Self::RiskManagement {
            message: message.into(),
        }
    }

    /// Create a decision strategy error
    pub fn decision_strategy(strategy: impl Into<String>, message: impl Into<String>) -> Self {
        Self::DecisionStrategy {
            strategy: strategy.into(),
            message: message.into(),
        }
    }

    /// Create an analyzer error
    pub fn analyzer(analyzer: impl Into<String>, message: impl Into<String>) -> Self {
        Self::Analyzer {
            analyzer: analyzer.into(),
            message: message.into(),
        }
    }

    /// Create a panarchy system error
    pub fn panarchy(message: impl Into<String>) -> Self {
        Self::Panarchy {
            message: message.into(),
        }
    }

    /// Create a hardware acceleration error
    pub fn hardware(message: impl Into<String>) -> Self {
        Self::Hardware {
            message: message.into(),
        }
    }

    /// Create a memory error
    pub fn memory(message: impl Into<String>) -> Self {
        Self::Memory {
            message: message.into(),
        }
    }

    /// Create a network error
    pub fn network(message: impl Into<String>) -> Self {
        Self::Network {
            message: message.into(),
        }
    }

    /// Create a serialization error
    pub fn serialization(message: impl Into<String>) -> Self {
        Self::Serialization {
            message: message.into(),
        }
    }

    /// Create a storage error
    pub fn storage(message: impl Into<String>) -> Self {
        Self::Storage {
            message: message.into(),
        }
    }

    /// Create a performance error
    pub fn performance(message: impl Into<String>) -> Self {
        Self::Performance {
            message: message.into(),
        }
    }

    /// Create a validation error
    pub fn validation(message: impl Into<String>) -> Self {
        Self::Validation {
            message: message.into(),
        }
    }

    /// Create a timeout error
    pub fn timeout(message: impl Into<String>) -> Self {
        Self::Timeout {
            message: message.into(),
        }
    }

    /// Create a resource exhaustion error
    pub fn resource_exhaustion(message: impl Into<String>) -> Self {
        Self::ResourceExhaustion {
            message: message.into(),
        }
    }

    /// Create a Python integration error
    #[cfg(feature = "python-integration")]
    pub fn python(message: impl Into<String>) -> Self {
        Self::Python {
            message: message.into(),
        }
    }

    /// Create an external service error
    pub fn external_service(service: impl Into<String>, message: impl Into<String>) -> Self {
        Self::ExternalService {
            service: service.into(),
            message: message.into(),
        }
    }

    /// Create an internal error
    pub fn internal(message: impl Into<String>) -> Self {
        Self::Internal {
            message: message.into(),
        }
    }

    /// Check if error is recoverable
    pub fn is_recoverable(&self) -> bool {
        match self {
            // Recoverable errors
            Self::Network { .. } => true,
            Self::Timeout { .. } => true,
            Self::ExternalService { .. } => true,
            Self::Http(_) => true,
            Self::Performance { .. } => true,
            
            // Non-recoverable errors
            Self::Configuration { .. } => false,
            Self::Memory { .. } => false,
            Self::Hardware { .. } => false,
            Self::Validation { .. } => false,
            Self::Internal { .. } => false,
            
            // Potentially recoverable errors
            Self::Agent { .. } => true,
            Self::Quantum { .. } => true,
            Self::BoardConsensus { .. } => true,
            Self::RiskManagement { .. } => true,
            Self::DecisionStrategy { .. } => true,
            Self::Analyzer { .. } => true,
            Self::Panarchy { .. } => true,
            Self::Serialization { .. } => true,
            Self::Storage { .. } => true,
            Self::ResourceExhaustion { .. } => true,
            
            #[cfg(feature = "python-integration")]
            Self::Python { .. } => true,
            
            // External errors - depends on the specific error
            Self::Io(_) => true,
            Self::Json(_) => false,
            Self::Toml(_) => false,
            Self::Database(_) => true,
            Self::TokioJoin(_) => false,
            Self::ChronoParse(_) => false,
            Self::Uuid(_) => false,
            Self::Generic(_) => false,
        }
    }

    /// Get error category
    pub fn category(&self) -> ErrorCategory {
        match self {
            Self::Configuration { .. } => ErrorCategory::Configuration,
            Self::Agent { .. } => ErrorCategory::Agent,
            Self::Quantum { .. } => ErrorCategory::Quantum,
            Self::BoardConsensus { .. } => ErrorCategory::Board,
            Self::RiskManagement { .. } => ErrorCategory::Risk,
            Self::DecisionStrategy { .. } => ErrorCategory::Strategy,
            Self::Analyzer { .. } => ErrorCategory::Analyzer,
            Self::Panarchy { .. } => ErrorCategory::Panarchy,
            Self::Hardware { .. } => ErrorCategory::Hardware,
            Self::Memory { .. } => ErrorCategory::Memory,
            Self::Network { .. } => ErrorCategory::Network,
            Self::Serialization { .. } => ErrorCategory::Serialization,
            Self::Storage { .. } => ErrorCategory::Storage,
            Self::Performance { .. } => ErrorCategory::Performance,
            Self::Validation { .. } => ErrorCategory::Validation,
            Self::Timeout { .. } => ErrorCategory::Timeout,
            Self::ResourceExhaustion { .. } => ErrorCategory::Resource,
            #[cfg(feature = "python-integration")]
            Self::Python { .. } => ErrorCategory::Python,
            Self::ExternalService { .. } => ErrorCategory::External,
            Self::Internal { .. } => ErrorCategory::Internal,
            Self::Io(_) => ErrorCategory::Io,
            Self::Json(_) => ErrorCategory::Serialization,
            Self::Toml(_) => ErrorCategory::Serialization,
            Self::Http(_) => ErrorCategory::Network,
            Self::Database(_) => ErrorCategory::Storage,
            Self::TokioJoin(_) => ErrorCategory::Runtime,
            Self::ChronoParse(_) => ErrorCategory::Validation,
            Self::Uuid(_) => ErrorCategory::Validation,
            Self::Generic(_) => ErrorCategory::Generic,
        }
    }

    /// Get error severity
    pub fn severity(&self) -> ErrorSeverity {
        match self {
            // Critical errors that require immediate attention
            Self::Memory { .. } => ErrorSeverity::Critical,
            Self::Hardware { .. } => ErrorSeverity::Critical,
            Self::ResourceExhaustion { .. } => ErrorSeverity::Critical,
            Self::Internal { .. } => ErrorSeverity::Critical,
            
            // High severity errors
            Self::Configuration { .. } => ErrorSeverity::High,
            Self::Panarchy { .. } => ErrorSeverity::High,
            Self::BoardConsensus { .. } => ErrorSeverity::High,
            
            // Medium severity errors
            Self::Agent { .. } => ErrorSeverity::Medium,
            Self::Quantum { .. } => ErrorSeverity::Medium,
            Self::RiskManagement { .. } => ErrorSeverity::Medium,
            Self::DecisionStrategy { .. } => ErrorSeverity::Medium,
            Self::Analyzer { .. } => ErrorSeverity::Medium,
            Self::Storage { .. } => ErrorSeverity::Medium,
            Self::Database(_) => ErrorSeverity::Medium,
            
            // Low severity errors
            Self::Network { .. } => ErrorSeverity::Low,
            Self::Serialization { .. } => ErrorSeverity::Low,
            Self::Performance { .. } => ErrorSeverity::Low,
            Self::Validation { .. } => ErrorSeverity::Low,
            Self::Timeout { .. } => ErrorSeverity::Low,
            Self::ExternalService { .. } => ErrorSeverity::Low,
            Self::Http(_) => ErrorSeverity::Low,
            Self::Io(_) => ErrorSeverity::Low,
            Self::Json(_) => ErrorSeverity::Low,
            Self::Toml(_) => ErrorSeverity::Low,
            Self::TokioJoin(_) => ErrorSeverity::Low,
            Self::ChronoParse(_) => ErrorSeverity::Low,
            Self::Uuid(_) => ErrorSeverity::Low,
            
            #[cfg(feature = "python-integration")]
            Self::Python { .. } => ErrorSeverity::Medium,
            
            Self::Generic(_) => ErrorSeverity::Medium,
        }
    }
}

/// Error category enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorCategory {
    /// Configuration errors
    Configuration,
    /// Agent-related errors
    Agent,
    /// Quantum computation errors
    Quantum,
    /// Board consensus errors
    Board,
    /// Risk management errors
    Risk,
    /// Decision strategy errors
    Strategy,
    /// Analyzer errors
    Analyzer,
    /// Panarchy system errors
    Panarchy,
    /// Hardware errors
    Hardware,
    /// Memory errors
    Memory,
    /// Network errors
    Network,
    /// Serialization errors
    Serialization,
    /// Storage errors
    Storage,
    /// Performance errors
    Performance,
    /// Validation errors
    Validation,
    /// Timeout errors
    Timeout,
    /// Resource errors
    Resource,
    /// Python integration errors
    #[cfg(feature = "python-integration")]
    Python,
    /// External service errors
    External,
    /// Internal errors
    Internal,
    /// IO errors
    Io,
    /// Runtime errors
    Runtime,
    /// Generic errors
    Generic,
}

/// Error severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ErrorSeverity {
    /// Low severity - warning level
    Low,
    /// Medium severity - error level
    Medium,
    /// High severity - critical error
    High,
    /// Critical severity - system failure
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

/// Error context for enhanced error reporting
#[derive(Debug, Clone)]
pub struct ErrorContext {
    /// Error category
    pub category: ErrorCategory,
    /// Error severity
    pub severity: ErrorSeverity,
    /// Whether error is recoverable
    pub recoverable: bool,
    /// Additional context information
    pub context: std::collections::HashMap<String, String>,
}

impl ErrorContext {
    /// Create new error context
    pub fn new(category: ErrorCategory, severity: ErrorSeverity, recoverable: bool) -> Self {
        Self {
            category,
            severity,
            recoverable,
            context: std::collections::HashMap::new(),
        }
    }

    /// Add context information
    pub fn with_context(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.context.insert(key.into(), value.into());
        self
    }
}

/// Extension trait for adding context to errors
pub trait ErrorContextExt<T> {
    /// Add context to error
    fn with_context(self, key: impl Into<String>, value: impl Into<String>) -> PadsResult<T>;
    
    /// Add context with closure
    fn with_context_lazy<F>(self, f: F) -> PadsResult<T>
    where
        F: FnOnce() -> (String, String);
}

impl<T> ErrorContextExt<T> for PadsResult<T> {
    fn with_context(self, key: impl Into<String>, value: impl Into<String>) -> PadsResult<T> {
        self.map_err(|mut err| {
            // Add context to error message if possible
            match &mut err {
                PadsError::Agent { message, .. } => {
                    *message = format!("{} [{}={}]", message, key.into(), value.into());
                }
                PadsError::Quantum { message } => {
                    *message = format!("{} [{}={}]", message, key.into(), value.into());
                }
                PadsError::BoardConsensus { message } => {
                    *message = format!("{} [{}={}]", message, key.into(), value.into());
                }
                PadsError::RiskManagement { message } => {
                    *message = format!("{} [{}={}]", message, key.into(), value.into());
                }
                PadsError::DecisionStrategy { message, .. } => {
                    *message = format!("{} [{}={}]", message, key.into(), value.into());
                }
                PadsError::Analyzer { message, .. } => {
                    *message = format!("{} [{}={}]", message, key.into(), value.into());
                }
                PadsError::Panarchy { message } => {
                    *message = format!("{} [{}={}]", message, key.into(), value.into());
                }
                PadsError::Hardware { message } => {
                    *message = format!("{} [{}={}]", message, key.into(), value.into());
                }
                PadsError::Memory { message } => {
                    *message = format!("{} [{}={}]", message, key.into(), value.into());
                }
                PadsError::Network { message } => {
                    *message = format!("{} [{}={}]", message, key.into(), value.into());
                }
                PadsError::Performance { message } => {
                    *message = format!("{} [{}={}]", message, key.into(), value.into());
                }
                PadsError::Internal { message } => {
                    *message = format!("{} [{}={}]", message, key.into(), value.into());
                }
                _ => {} // For other error types, context is not added to message
            }
            err
        })
    }
    
    fn with_context_lazy<F>(self, f: F) -> PadsResult<T>
    where
        F: FnOnce() -> (String, String),
    {
        match self {
            Ok(value) => Ok(value),
            Err(err) => {
                let (key, value) = f();
                Err(err).with_context(key, value)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let err = PadsError::agent("test_agent", "test message");
        assert!(matches!(err, PadsError::Agent { .. }));
        assert_eq!(err.category(), ErrorCategory::Agent);
        assert_eq!(err.severity(), ErrorSeverity::Medium);
        assert!(err.is_recoverable());
    }

    #[test]
    fn test_error_context() {
        let result: PadsResult<()> = Err(PadsError::quantum("test error"));
        let result = result.with_context("component", "quantum_processor");
        
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("component=quantum_processor"));
    }

    #[test]
    fn test_error_severity_ordering() {
        assert!(ErrorSeverity::Low < ErrorSeverity::Medium);
        assert!(ErrorSeverity::Medium < ErrorSeverity::High);
        assert!(ErrorSeverity::High < ErrorSeverity::Critical);
    }

    #[test]
    fn test_error_recoverability() {
        assert!(PadsError::network("test").is_recoverable());
        assert!(!PadsError::configuration("test").is_recoverable());
        assert!(!PadsError::memory("test").is_recoverable());
    }

    #[test]
    fn test_error_categories() {
        assert_eq!(PadsError::agent("test", "test").category(), ErrorCategory::Agent);
        assert_eq!(PadsError::quantum("test").category(), ErrorCategory::Quantum);
        assert_eq!(PadsError::hardware("test").category(), ErrorCategory::Hardware);
    }

    #[test]
    fn test_error_context_creation() {
        let context = ErrorContext::new(ErrorCategory::Agent, ErrorSeverity::High, true)
            .with_context("component", "test_agent")
            .with_context("operation", "make_decision");
        
        assert_eq!(context.category, ErrorCategory::Agent);
        assert_eq!(context.severity, ErrorSeverity::High);
        assert!(context.recoverable);
        assert_eq!(context.context.get("component"), Some(&"test_agent".to_string()));
        assert_eq!(context.context.get("operation"), Some(&"make_decision".to_string()));
    }
}