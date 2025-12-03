//! Error types for HyperPhysics AgentDB

use thiserror::Error;

/// AgentDB error type
#[derive(Error, Debug)]
pub enum AgentDBError {
    /// Database operation failed
    #[error("Database error: {0}")]
    DatabaseError(String),

    /// Vector not found
    #[error("Vector not found: {0}")]
    VectorNotFound(String),

    /// Episode not found
    #[error("Episode not found: {0}")]
    EpisodeNotFound(String),

    /// Skill not found
    #[error("Skill not found: {0}")]
    SkillNotFound(String),

    /// Session not found
    #[error("Learning session not found: {0}")]
    SessionNotFound(String),

    /// Insufficient confidence for causal learning
    #[error("Insufficient confidence: required {required}, got {actual}")]
    InsufficientConfidence { required: f64, actual: f64 },

    /// Embedding generation failed
    #[error("Embedding error: {0}")]
    EmbeddingError(String),

    /// Serialization/deserialization failed
    #[error("Serialization error: {0}")]
    SerializationError(String),

    /// Invalid input
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// IO error
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    /// Configuration error
    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// Model not loaded
    #[error("Embedding model not loaded")]
    ModelNotLoaded,

    /// Dimension mismatch
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    /// Factor already exists
    #[error("Alpha factor already exists: {0}")]
    FactorExists(String),

    /// Factor correlation too high
    #[error("Factor correlation too high with {existing}: {correlation}")]
    HighCorrelation { existing: String, correlation: f64 },

    /// Decay detected - factor no longer valid
    #[error("Alpha factor has decayed below threshold: {factor_id}")]
    FactorDecayed { factor_id: String },

    /// GNN layer error
    #[cfg(feature = "gnn")]
    #[error("GNN error: {0}")]
    GnnError(String),

    /// Attention mechanism error
    #[cfg(feature = "attention")]
    #[error("Attention error: {0}")]
    AttentionError(String),

    /// pBit dynamics error
    #[cfg(feature = "pbit")]
    #[error("pBit error: {0}")]
    PBitError(String),

    /// Neural routing error
    #[cfg(feature = "routing")]
    #[error("Routing error: {0}")]
    RoutingError(String),
}

impl From<serde_json::Error> for AgentDBError {
    fn from(e: serde_json::Error) -> Self {
        AgentDBError::SerializationError(e.to_string())
    }
}

/// Result type alias
pub type Result<T> = std::result::Result<T, AgentDBError>;
