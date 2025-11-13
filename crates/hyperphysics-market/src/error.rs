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
    NetworkError(#[from] reqwest::Error),

    /// JSON parsing errors
    #[error("Parse error: {0}")]
    ParseError(#[from] serde_json::Error),

    /// DateTime parsing errors
    #[error("DateTime parse error: {0}")]
    DateTimeParseError(String),

    /// Authentication errors
    #[error("Authentication failed: {0}")]
    AuthenticationError(String),

    /// Rate limiting errors
    #[error("Rate limit exceeded: {0}")]
    RateLimitError(String),

    /// Invalid timeframe requested
    #[error("Invalid timeframe: {0}")]
    InvalidTimeframe(String),

    /// Data not available for requested period
    #[error("Data unavailable: {0}")]
    DataUnavailable(String),
}

/// Result type alias for market operations
pub type MarketResult<T> = Result<T, MarketError>;
