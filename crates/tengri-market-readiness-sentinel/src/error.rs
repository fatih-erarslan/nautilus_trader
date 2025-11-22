//! Error handling for TENGRI Market Readiness Sentinel

use thiserror::Error;

/// Main error type for the market readiness sentinel
#[derive(Error, Debug)]
pub enum MarketReadinessError {
    #[error("Configuration error: {0}")]
    Configuration(String),

    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),

    #[error("Redis error: {0}")]
    Redis(#[from] redis::RedisError),

    #[error("HTTP client error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("WebSocket error: {0}")]
    WebSocket(#[from] tungstenite::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Validation error: {0}")]
    Validation(String),

    #[error("Authentication error: {0}")]
    Authentication(String),

    #[error("Market connectivity error: {0}")]
    MarketConnectivity(String),

    #[error("Trading system error: {0}")]
    TradingSystem(String),

    #[error("Risk management error: {0}")]
    RiskManagement(String),

    #[error("Deployment error: {0}")]
    Deployment(String),

    #[error("Monitoring error: {0}")]
    Monitoring(String),

    #[error("System timeout: {0}")]
    Timeout(String),

    #[error("Resource unavailable: {0}")]
    ResourceUnavailable(String),

    #[error("Invalid state: {0}")]
    InvalidState(String),

    #[error("External service error: {0}")]
    ExternalService(String),
}

/// Result type alias for convenience
pub type Result<T> = std::result::Result<T, MarketReadinessError>;

impl MarketReadinessError {
    pub fn configuration(msg: impl Into<String>) -> Self {
        Self::Configuration(msg.into())
    }

    pub fn validation(msg: impl Into<String>) -> Self {
        Self::Validation(msg.into())
    }

    pub fn authentication(msg: impl Into<String>) -> Self {
        Self::Authentication(msg.into())
    }

    pub fn market_connectivity(msg: impl Into<String>) -> Self {
        Self::MarketConnectivity(msg.into())
    }

    pub fn trading_system(msg: impl Into<String>) -> Self {
        Self::TradingSystem(msg.into())
    }

    pub fn risk_management(msg: impl Into<String>) -> Self {
        Self::RiskManagement(msg.into())
    }

    pub fn deployment(msg: impl Into<String>) -> Self {
        Self::Deployment(msg.into())
    }

    pub fn monitoring(msg: impl Into<String>) -> Self {
        Self::Monitoring(msg.into())
    }

    pub fn timeout(msg: impl Into<String>) -> Self {
        Self::Timeout(msg.into())
    }

    pub fn resource_unavailable(msg: impl Into<String>) -> Self {
        Self::ResourceUnavailable(msg.into())
    }

    pub fn invalid_state(msg: impl Into<String>) -> Self {
        Self::InvalidState(msg.into())
    }

    pub fn external_service(msg: impl Into<String>) -> Self {
        Self::ExternalService(msg.into())
    }
}