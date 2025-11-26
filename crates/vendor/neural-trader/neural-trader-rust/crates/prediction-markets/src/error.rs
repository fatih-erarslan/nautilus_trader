//! Error types for prediction markets

use thiserror::Error;

/// Result type for prediction market operations
pub type Result<T> = std::result::Result<T, PredictionMarketError>;

/// Errors that can occur in prediction market operations
#[derive(Error, Debug)]
pub enum PredictionMarketError {
    /// HTTP request failed
    #[error("HTTP request failed: {0}")]
    HttpError(String),

    /// JSON serialization/deserialization error
    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),

    /// WebSocket error
    #[error("WebSocket error: {0}")]
    WebSocketError(String),

    /// Authentication error
    #[error("Authentication failed: {0}")]
    AuthError(String),

    /// API error with status code
    #[error("API error {status}: {message}")]
    ApiError { status: u16, message: String },

    /// Order validation error
    #[error("Invalid order: {0}")]
    InvalidOrder(String),

    /// Market not found
    #[error("Market not found: {0}")]
    MarketNotFound(String),

    /// Insufficient liquidity
    #[error("Insufficient liquidity for order size {0}")]
    InsufficientLiquidity(String),

    /// Rate limit exceeded
    #[error("Rate limit exceeded, retry after {0}s")]
    RateLimitExceeded(u64),

    /// Invalid configuration
    #[error("Invalid configuration: {0}")]
    ConfigError(String),

    /// Internal error
    #[error("Internal error: {0}")]
    InternalError(String),

    /// Decimal conversion error
    #[error("Decimal conversion error: {0}")]
    DecimalError(String),

    /// URL parse error
    #[error("URL parse error: {0}")]
    UrlError(String),
}

impl From<reqwest::Error> for PredictionMarketError {
    fn from(err: reqwest::Error) -> Self {
        PredictionMarketError::HttpError(err.to_string())
    }
}

impl From<rust_decimal::Error> for PredictionMarketError {
    fn from(err: rust_decimal::Error) -> Self {
        PredictionMarketError::DecimalError(err.to_string())
    }
}

impl From<url::ParseError> for PredictionMarketError {
    fn from(err: url::ParseError) -> Self {
        PredictionMarketError::UrlError(err.to_string())
    }
}

impl PredictionMarketError {
    /// Create an API error from response
    pub fn from_status(status: u16, message: String) -> Self {
        Self::ApiError { status, message }
    }

    /// Check if error is retryable
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            Self::RateLimitExceeded(_) | Self::HttpError(_) | Self::WebSocketError(_)
        )
    }

    /// Get retry delay in seconds
    pub fn retry_delay(&self) -> Option<u64> {
        match self {
            Self::RateLimitExceeded(delay) => Some(*delay),
            Self::HttpError(_) => Some(1),
            Self::WebSocketError(_) => Some(5),
            _ => None,
        }
    }
}
