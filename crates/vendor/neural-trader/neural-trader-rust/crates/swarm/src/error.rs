//! Error types for QUIC swarm coordination

use thiserror::Error;

/// Result type for swarm operations
pub type Result<T> = std::result::Result<T, SwarmError>;

/// Errors that can occur during swarm coordination
#[derive(Error, Debug)]
pub enum SwarmError {
    /// QUIC connection error
    #[error("QUIC connection error: {0}")]
    Connection(#[from] quinn::ConnectionError),

    /// QUIC write error
    #[error("QUIC write error: {0}")]
    Write(#[from] quinn::WriteError),

    /// QUIC read error
    #[error("QUIC read error: {0}")]
    Read(#[from] quinn::ReadError),

    /// TLS configuration error
    #[error("TLS configuration error: {0}")]
    Tls(#[from] rustls::Error),

    /// Certificate generation error
    #[error("Certificate error: {0}")]
    Certificate(#[from] rcgen::Error),

    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Agent not found
    #[error("Agent not found: {0}")]
    AgentNotFound(String),

    /// Stream closed
    #[error("Stream closed unexpectedly")]
    StreamClosed,

    /// Invalid handshake
    #[error("Invalid handshake: {0}")]
    InvalidHandshake(String),

    /// Task processing error
    #[error("Task processing error: {0}")]
    TaskProcessing(String),

    /// Timeout error
    #[error("Operation timed out")]
    Timeout,

    /// Configuration error
    #[error("Configuration error: {0}")]
    Configuration(String),

    /// ReasoningBank error
    #[error("ReasoningBank error: {0}")]
    ReasoningBank(String),

    /// Generic error
    #[error("{0}")]
    Other(String),
}

impl From<anyhow::Error> for SwarmError {
    fn from(e: anyhow::Error) -> Self {
        SwarmError::Other(e.to_string())
    }
}
