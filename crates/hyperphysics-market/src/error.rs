//! Error types for market data operations

use thiserror::Error;

/// Errors that can occur during market data operations
#[derive(Error, Debug)]
pub enum MarketError {
    /// API request failed with provider-specific error
    #[error("API request failed: {0}")]
    ApiError(String),

    /// Invalid or unsupported symbol
    #[error("Invalid symbol: {0}")]
    InvalidSymbol(String),

    /// Network-related errors
    #[error("Network error: {0}")]
    NetworkError(String),

    /// JSON parsing errors
    #[error("Parse error: {0}")]
    ParseError(String),

    /// DateTime parsing errors
    #[error("DateTime parse error: {0}")]
    DateTimeParseError(String),

    /// Authentication errors
    #[error("Authentication failed: {0}")]
    AuthenticationError(String),

    /// Rate limiting errors (deprecated, use RateLimitExceeded)
    #[error("Rate limit exceeded: {0}")]
    RateLimitError(String),

    /// Rate limit exceeded - includes retry-after information
    #[error("Rate limit exceeded: {0}")]
    RateLimitExceeded(String),

    /// Invalid timeframe requested
    #[error("Invalid timeframe: {0}")]
    InvalidTimeframe(String),

    /// Data not available for requested period
    #[error("Data unavailable: {0}")]
    DataUnavailable(String),

    /// Data integrity violation (e.g., OHLC validation failure)
    #[error("Data integrity error: {0}")]
    DataIntegrityError(String),

    /// Configuration error
    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// Connection error
    #[error("Connection error: {0}")]
    ConnectionError(String),

    /// Timeout error
    #[error("Request timeout: {0}")]
    TimeoutError(String),
}

// Implement conversions for common error types
impl From<reqwest::Error> for MarketError {
    fn from(err: reqwest::Error) -> Self {
        if err.is_timeout() {
            MarketError::TimeoutError(err.to_string())
        } else if err.is_connect() {
            MarketError::ConnectionError(err.to_string())
        } else {
            MarketError::NetworkError(err.to_string())
        }
    }
}

impl From<serde_json::Error> for MarketError {
    fn from(err: serde_json::Error) -> Self {
        MarketError::ParseError(err.to_string())
    }
}

/// Result type alias for market operations
pub type MarketResult<T> = Result<T, MarketError>;
