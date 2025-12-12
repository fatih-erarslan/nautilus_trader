//! Error types for E2B client

use thiserror::Error;

/// Result type alias for E2B operations
pub type Result<T> = std::result::Result<T, Error>;

/// E2B client errors
#[derive(Error, Debug)]
pub enum Error {
    /// HTTP request failed
    #[error("HTTP request failed: {0}")]
    Http(#[from] reqwest::Error),

    /// JSON serialization/deserialization failed
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// API returned an error
    #[error("API error: {status} - {message}")]
    Api { status: u16, message: String },

    /// Invalid API key
    #[error("Invalid API key")]
    InvalidApiKey,

    /// Sandbox not found
    #[error("Sandbox not found: {0}")]
    SandboxNotFound(String),

    /// Sandbox execution timeout
    #[error("Execution timeout after {0}ms")]
    Timeout(u64),

    /// File system operation failed
    #[error("Filesystem error: {0}")]
    Filesystem(String),

    /// Process execution failed
    #[error("Process execution failed: {0}")]
    ProcessExecution(String),

    /// Configuration error
    #[error("Configuration error: {0}")]
    Config(String),

    /// Sandbox already closed
    #[error("Sandbox already closed")]
    SandboxClosed,

    /// Rate limit exceeded
    #[error("Rate limit exceeded, retry after {retry_after}s")]
    RateLimited { retry_after: u64 },
}
