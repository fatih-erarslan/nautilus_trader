//! Error types for the HiveMind system

use thiserror::Error;

pub type Result<T> = std::result::Result<T, HiveMindError>;

#[derive(Debug, Error)]
pub enum HiveMindError {
    #[error("Maximum number of workers reached")]
    MaxWorkersReached,

    #[error("Worker not found: {0}")]
    WorkerNotFound(String),

    #[error("Queen error: {0}")]
    QueenError(String),

    #[error("Memory error: {0}")]
    MemoryError(String),

    #[error("Consensus error: {0}")]
    ConsensusError(String),

    #[error("Task execution failed: {0}")]
    TaskExecutionFailed(String),

    #[error("Invalid configuration: {0}")]
    InvalidConfiguration(String),

    #[error("Agent spawn failed: {0}")]
    AgentSpawnFailed(String),

    #[error("Coordination error: {0}")]
    CoordinationError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),

    #[error("Unknown error: {0}")]
    Unknown(String),
}
