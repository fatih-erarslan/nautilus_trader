//! Unified error types for the integration layer.

use thiserror::Error;

/// Result type alias using the integration Error type.
pub type Result<T> = std::result::Result<T, Error>;

/// Unified error type for the Neural Trader integration layer.
#[derive(Error, Debug)]
pub enum Error {
    /// Configuration errors
    #[error("Configuration error: {0}")]
    Config(String),

    /// Runtime initialization errors
    #[error("Runtime error: {0}")]
    Runtime(String),

    /// Broker-related errors
    #[error("Broker error: {0}")]
    Broker(String),

    /// Strategy execution errors
    #[error("Strategy error: {0}")]
    Strategy(String),

    /// Neural model errors
    #[error("Neural error: {0}")]
    Neural(String),

    /// Risk management errors
    #[error("Risk error: {0}")]
    Risk(String),

    /// Memory/storage errors
    #[error("Memory error: {0}")]
    Memory(String),

    /// Service errors
    #[error("Service error: {0}")]
    Service(String),

    /// API errors
    #[error("API error: {0}")]
    Api(String),

    /// IO errors
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Serialization errors
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// Configuration file errors
    #[error("Config file error: {0}")]
    ConfigFile(#[from] config::ConfigError),

    /// Timeout errors
    #[error("Operation timed out: {0}")]
    Timeout(String),

    /// Not found errors
    #[error("Not found: {0}")]
    NotFound(String),

    /// Already exists errors
    #[error("Already exists: {0}")]
    AlreadyExists(String),

    /// Invalid state errors
    #[error("Invalid state: {0}")]
    InvalidState(String),

    /// Invalid input errors
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Authentication errors
    #[error("Authentication error: {0}")]
    Auth(String),

    /// Permission errors
    #[error("Permission denied: {0}")]
    Permission(String),

    /// Rate limit errors
    #[error("Rate limit exceeded: {0}")]
    RateLimit(String),

    /// Internal errors
    #[error("Internal error: {0}")]
    Internal(String),

    /// Generic error with context
    #[error("{context}: {source}")]
    WithContext {
        context: String,
        source: Box<Error>,
    },
}

impl Error {
    /// Adds context to an error.
    pub fn context(self, context: impl Into<String>) -> Self {
        Self::WithContext {
            context: context.into(),
            source: Box::new(self),
        }
    }

    /// Creates a configuration error.
    pub fn config(msg: impl Into<String>) -> Self {
        Self::Config(msg.into())
    }

    /// Creates a broker error.
    pub fn broker(msg: impl Into<String>) -> Self {
        Self::Broker(msg.into())
    }

    /// Creates a strategy error.
    pub fn strategy(msg: impl Into<String>) -> Self {
        Self::Strategy(msg.into())
    }

    /// Creates a neural error.
    pub fn neural(msg: impl Into<String>) -> Self {
        Self::Neural(msg.into())
    }

    /// Creates a risk error.
    pub fn risk(msg: impl Into<String>) -> Self {
        Self::Risk(msg.into())
    }

    /// Creates a memory error.
    pub fn memory(msg: impl Into<String>) -> Self {
        Self::Memory(msg.into())
    }

    /// Creates a service error.
    pub fn service(msg: impl Into<String>) -> Self {
        Self::Service(msg.into())
    }

    /// Creates an API error.
    pub fn api(msg: impl Into<String>) -> Self {
        Self::Api(msg.into())
    }

    /// Creates a timeout error.
    pub fn timeout(msg: impl Into<String>) -> Self {
        Self::Timeout(msg.into())
    }

    /// Creates a not found error.
    pub fn not_found(msg: impl Into<String>) -> Self {
        Self::NotFound(msg.into())
    }

    /// Creates an already exists error.
    pub fn already_exists(msg: impl Into<String>) -> Self {
        Self::AlreadyExists(msg.into())
    }

    /// Creates an invalid state error.
    pub fn invalid_state(msg: impl Into<String>) -> Self {
        Self::InvalidState(msg.into())
    }

    /// Creates an invalid input error.
    pub fn invalid_input(msg: impl Into<String>) -> Self {
        Self::InvalidInput(msg.into())
    }
}

// Implement From for common error types
impl From<anyhow::Error> for Error {
    fn from(err: anyhow::Error) -> Self {
        Self::Internal(err.to_string())
    }
}

impl From<tokio::time::error::Elapsed> for Error {
    fn from(err: tokio::time::error::Elapsed) -> Self {
        Self::Timeout(err.to_string())
    }
}
