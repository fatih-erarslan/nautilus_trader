//! Error types for multi-market trading

use thiserror::Error;

/// Result type for multi-market operations
pub type Result<T> = std::result::Result<T, MultiMarketError>;

/// Comprehensive error types for multi-market trading
#[derive(Error, Debug)]
pub enum MultiMarketError {
    /// API communication errors
    #[error("API error: {0}")]
    ApiError(String),

    /// Authentication/authorization errors
    #[error("Authentication failed: {0}")]
    AuthError(String),

    /// Rate limiting errors
    #[error("Rate limit exceeded: {0}")]
    RateLimitError(String),

    /// Validation errors
    #[error("Validation failed: {0}")]
    ValidationError(String),

    /// Market data errors
    #[error("Market data error: {0}")]
    MarketDataError(String),

    /// Order placement errors
    #[error("Order error: {0}")]
    OrderError(String),

    /// Arbitrage detection errors
    #[error("Arbitrage error: {0}")]
    ArbitrageError(String),

    /// Risk management errors
    #[error("Risk limit exceeded: {0}")]
    RiskError(String),

    /// Network/connection errors
    #[error("Network error: {0}")]
    NetworkError(#[from] reqwest::Error),

    /// WebSocket errors
    #[error("WebSocket error: {0}")]
    WebSocketError(String),

    /// Serialization errors
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),

    /// Decimal parsing errors
    #[error("Decimal error: {0}")]
    DecimalError(#[from] rust_decimal::Error),

    /// Configuration errors
    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// Timeout errors
    #[error("Operation timed out: {0}")]
    TimeoutError(String),

    /// Generic errors
    #[error("Internal error: {0}")]
    InternalError(String),
}

impl MultiMarketError {
    /// Check if error is retryable
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            Self::NetworkError(_)
                | Self::TimeoutError(_)
                | Self::RateLimitError(_)
                | Self::ApiError(_)
        )
    }

    /// Get error category for metrics
    pub fn category(&self) -> &'static str {
        match self {
            Self::ApiError(_) => "api",
            Self::AuthError(_) => "auth",
            Self::RateLimitError(_) => "rate_limit",
            Self::ValidationError(_) => "validation",
            Self::MarketDataError(_) => "market_data",
            Self::OrderError(_) => "order",
            Self::ArbitrageError(_) => "arbitrage",
            Self::RiskError(_) => "risk",
            Self::NetworkError(_) => "network",
            Self::WebSocketError(_) => "websocket",
            Self::SerializationError(_) => "serialization",
            Self::DecimalError(_) => "decimal",
            Self::ConfigError(_) => "config",
            Self::TimeoutError(_) => "timeout",
            Self::InternalError(_) => "internal",
        }
    }
}
