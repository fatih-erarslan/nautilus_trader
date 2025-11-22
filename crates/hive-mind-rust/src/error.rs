//! Error handling for the hive mind system

use thiserror::Error;

/// Main error type for the hive mind system
#[derive(Error, Debug)]
pub enum HiveMindError {
    #[error("Consensus error: {0}")]
    Consensus(#[from] ConsensusError),

    #[error("Memory error: {0}")]
    Memory(#[from] MemoryError),

    #[error("Neural error: {0}")]
    Neural(#[from] NeuralError),

    #[error("Network error: {0}")]
    Network(#[from] NetworkError),

    #[error("Agent error: {0}")]
    Agent(#[from] AgentError),

    #[error("Configuration error: {0}")]
    Config(#[from] ConfigError),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),

    #[error("Timeout error: operation timed out after {timeout_ms}ms")]
    Timeout { timeout_ms: u64 },

    #[error("Invalid state: {message}")]
    InvalidState { message: String },

    #[error("Resource exhausted: {resource}")]
    ResourceExhausted { resource: String },

    #[error("Internal error: {0}")]
    Internal(String),
}

/// Consensus-specific errors
#[derive(Error, Debug)]
pub enum ConsensusError {
    #[error("Failed to reach consensus within timeout")]
    ConsensusTimeout,

    #[error("Invalid consensus proposal: {reason}")]
    InvalidProposal { reason: String },

    #[error("Leader election failed")]
    LeaderElectionFailed,

    #[error("Node not part of consensus group")]
    NotInConsensusGroup,

    #[error("Byzantine fault detected from node {node_id}")]
    ByzantineFault { node_id: String },

    #[error("Quorum not reached: required {required}, got {actual}")]
    QuorumNotReached { required: usize, actual: usize },
}

/// Memory-specific errors
#[derive(Error, Debug)]
pub enum MemoryError {
    #[error("Memory allocation failed")]
    AllocationFailed,

    #[error("Memory corruption detected")]
    CorruptionDetected,

    #[error("Knowledge not found: {key}")]
    KnowledgeNotFound { key: String },

    #[error("Memory synchronization failed")]
    SynchronizationFailed,

    #[error("Memory consistency violation")]
    ConsistencyViolation,

    #[error("Memory capacity exceeded")]
    CapacityExceeded,
}

/// Neural processing errors
#[derive(Error, Debug)]
pub enum NeuralError {
    #[error("Pattern recognition failed: {reason}")]
    PatternRecognitionFailed { reason: String },

    #[error("Neural network initialization failed")]
    NetworkInitializationFailed,

    #[error("Training convergence failed")]
    TrainingFailed,

    #[error("Model inference error: {details}")]
    InferenceError { details: String },

    #[error("Unsupported neural operation: {operation}")]
    UnsupportedOperation { operation: String },
}

/// Network communication errors
#[derive(Error, Debug)]
pub enum NetworkError {
    #[error("Connection failed to {peer}")]
    ConnectionFailed { peer: String },

    #[error("Message delivery failed")]
    DeliveryFailed,

    #[error("Protocol version mismatch")]
    ProtocolMismatch,

    #[error("Network partition detected")]
    NetworkPartition,

    #[error("Peer discovery failed")]
    PeerDiscoveryFailed,

    #[error("Message validation failed: {reason}")]
    MessageValidationFailed { reason: String },
}

/// Agent-specific errors
#[derive(Error, Debug)]
pub enum AgentError {
    #[error("Agent spawn failed: {reason}")]
    SpawnFailed { reason: String },

    #[error("Agent {agent_id} not found")]
    AgentNotFound { agent_id: String },

    #[error("Agent coordination failed")]
    CoordinationFailed,

    #[error("Agent capability mismatch")]
    CapabilityMismatch,

    #[error("Agent overloaded")]
    AgentOverloaded,

    #[error("Agent initialization failed")]
    InitializationFailed,
}

/// Configuration errors
#[derive(Error, Debug)]
pub enum ConfigError {
    #[error("Invalid configuration parameter: {parameter}")]
    InvalidParameter { parameter: String },

    #[error("Missing required configuration: {field}")]
    MissingRequired { field: String },

    #[error("Configuration file not found: {path}")]
    FileNotFound { path: String },

    #[error("Configuration parsing failed: {reason}")]
    ParseFailed { reason: String },
}

/// Result type for hive mind operations
pub type Result<T> = std::result::Result<T, HiveMindError>;

impl HiveMindError {
    /// Check if the error is recoverable
    pub fn is_recoverable(&self) -> bool {
        match self {
            HiveMindError::Consensus(ConsensusError::ConsensusTimeout) => true,
            HiveMindError::Network(NetworkError::ConnectionFailed { .. }) => true,
            HiveMindError::Network(NetworkError::DeliveryFailed) => true,
            HiveMindError::Memory(MemoryError::SynchronizationFailed) => true,
            HiveMindError::Agent(AgentError::AgentOverloaded) => true,
            HiveMindError::Timeout { .. } => true,
            _ => false,
        }
    }

    /// Get error severity level
    pub fn severity(&self) -> ErrorSeverity {
        match self {
            HiveMindError::Memory(MemoryError::CorruptionDetected) => ErrorSeverity::Critical,
            HiveMindError::Consensus(ConsensusError::ByzantineFault { .. }) => ErrorSeverity::Critical,
            HiveMindError::InvalidState { .. } => ErrorSeverity::High,
            HiveMindError::ResourceExhausted { .. } => ErrorSeverity::High,
            HiveMindError::Network(NetworkError::NetworkPartition) => ErrorSeverity::Medium,
            HiveMindError::Agent(AgentError::CoordinationFailed) => ErrorSeverity::Medium,
            _ => ErrorSeverity::Low,
        }
    }
}

/// Error severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorSeverity {
    Low,
    Medium,
    High,
    Critical,
}

impl ErrorSeverity {
    pub fn as_str(&self) -> &'static str {
        match self {
            ErrorSeverity::Low => "low",
            ErrorSeverity::Medium => "medium", 
            ErrorSeverity::High => "high",
            ErrorSeverity::Critical => "critical",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_recoverability() {
        let recoverable_error = HiveMindError::Timeout { timeout_ms: 5000 };
        assert!(recoverable_error.is_recoverable());

        let non_recoverable_error = HiveMindError::Memory(MemoryError::CorruptionDetected);
        assert!(!non_recoverable_error.is_recoverable());
    }

    #[test]
    fn test_error_severity() {
        let critical_error = HiveMindError::Memory(MemoryError::CorruptionDetected);
        assert_eq!(critical_error.severity(), ErrorSeverity::Critical);

        let low_error = HiveMindError::Config(ConfigError::FileNotFound { 
            path: "test.toml".to_string() 
        });
        assert_eq!(low_error.severity(), ErrorSeverity::Low);
    }
}