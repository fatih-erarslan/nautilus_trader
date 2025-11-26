//! Error types for sports betting operations

use thiserror::Error;

/// Result type alias for sports betting operations
pub type Result<T> = std::result::Result<T, Error>;

/// Sports betting error types
#[derive(Error, Debug)]
pub enum Error {
    /// Risk limit exceeded
    #[error("Risk limit exceeded: {0}")]
    RiskLimitExceeded(String),

    /// Insufficient capital
    #[error("Insufficient capital: required {required}, available {available}")]
    InsufficientCapital {
        required: f64,
        available: f64,
    },

    /// Member not found
    #[error("Member not found: {0}")]
    MemberNotFound(String),

    /// Insufficient voting power
    #[error("Insufficient voting power: required {required}%, have {actual}%")]
    InsufficientVotingPower {
        required: f64,
        actual: f64,
    },

    /// Bet placement failed
    #[error("Bet placement failed: {0}")]
    BetPlacementFailed(String),

    /// Odds API error
    #[error("Odds API error: {0}")]
    OddsApiError(String),

    /// Configuration error
    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// Network error
    #[error("Network error: {0}")]
    NetworkError(#[from] reqwest::Error),

    /// Serialization error
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),

    /// Internal error
    #[error("Internal error: {0}")]
    Internal(String),
}
