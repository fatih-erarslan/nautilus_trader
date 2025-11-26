//! Message and type definitions for QUIC swarm coordination

use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use std::collections::HashMap;

/// Agent type classification
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum AgentType {
    /// Pattern matching agent
    PatternMatcher,
    /// Strategy correlation agent
    StrategyCorrelator,
    /// Feature engineering agent
    FeatureEngineer,
    /// Neural training agent
    NeuralTrainer,
    /// ReasoningBank agent
    ReasoningBanker,
    /// Generic worker
    Worker,
}

/// Stream purpose classification
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum StreamPurpose {
    /// Pattern matching results
    PatternMatching,
    /// Strategy correlation data
    StrategyCorrelation,
    /// Feature engineering output
    FeatureEngineering,
    /// Neural training gradients
    NeuralTraining,
    /// ReasoningBank experiences
    ReasoningExchange,
    /// Task assignment
    TaskAssignment,
    /// Control messages
    Control,
}

/// Agent handshake message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentHandshake {
    /// Unique agent identifier
    pub agent_id: String,
    /// Agent type
    pub agent_type: AgentType,
    /// Agent capabilities
    pub capabilities: Vec<String>,
    /// Agent version
    pub version: String,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Coordinator acknowledgment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentAck {
    /// Coordinator identifier
    pub coordinator_id: String,
    /// Assigned stream configurations
    pub assigned_streams: Vec<StreamAssignment>,
    /// Session token for reconnection
    pub session_token: String,
    /// Configuration for the agent
    pub config: AgentConfig,
}

/// Stream assignment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamAssignment {
    /// Stream identifier
    pub stream_id: u64,
    /// Stream purpose
    pub purpose: StreamPurpose,
    /// Priority level (0-10)
    pub priority: u8,
}

/// Agent configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    /// Maximum buffer size
    pub max_buffer_size: usize,
    /// Heartbeat interval in seconds
    pub heartbeat_interval: u64,
    /// Task timeout in seconds
    pub task_timeout: u64,
    /// Enable compression
    pub compression: bool,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            max_buffer_size: 65536,
            heartbeat_interval: 10,
            task_timeout: 300,
            compression: false,
        }
    }
}

/// Agent message types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "payload")]
pub enum AgentMessage {
    /// Pattern matching result
    PatternMatchResult(PatternMatchResult),
    /// Strategy correlation data
    StrategyCorrelation(StrategyCorrelation),
    /// ReasoningBank experience
    ReasoningExperience(ReasoningExperience),
    /// Neural gradients
    NeuralGradients(NeuralGradients),
    /// Heartbeat
    Heartbeat(HeartbeatMessage),
    /// Task completion
    TaskComplete(TaskCompletion),
    /// Error report
    Error(ErrorReport),
}

/// Pattern matching result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternMatchResult {
    /// Message ID
    pub message_id: String,
    /// Pattern type
    pub pattern_type: String,
    /// Pattern vector representation
    pub pattern_vector: Vec<f32>,
    /// Similarity score (0-1)
    pub similarity: f64,
    /// Confidence score (0-1)
    pub confidence: f64,
    /// Expected outcome
    pub expected_outcome: f64,
    /// Actual outcome (if available)
    pub actual_outcome: Option<f64>,
    /// Computation time in milliseconds
    pub compute_time_ms: f64,
    /// Market context
    pub market_context: serde_json::Value,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

/// Strategy correlation matrix
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyCorrelation {
    /// Message ID
    pub message_id: String,
    /// Correlation matrix
    pub matrix: Vec<Vec<f64>>,
    /// Strategy identifiers
    pub strategy_ids: Vec<String>,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

/// ReasoningBank experience record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningExperience {
    /// Message ID
    pub message_id: String,
    /// Agent ID that generated the experience
    pub agent_id: String,
    /// Action taken
    pub action: String,
    /// Outcome metrics
    pub outcome: OutcomeMetrics,
    /// Context information
    pub context: serde_json::Value,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

/// Outcome metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutcomeMetrics {
    /// Primary metric value
    pub value: f64,
    /// Confidence in the outcome
    pub confidence: f64,
    /// Latency in milliseconds
    pub latency_ms: f64,
    /// Additional metrics
    pub additional: HashMap<String, f64>,
}

/// Neural gradients
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralGradients {
    /// Message ID
    pub message_id: String,
    /// Model identifier
    pub model_id: String,
    /// Gradients
    pub gradients: Vec<f32>,
    /// Batch size
    pub batch_size: usize,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

/// Heartbeat message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeartbeatMessage {
    /// Agent ID
    pub agent_id: String,
    /// Current load (0-1)
    pub load: f64,
    /// Active tasks
    pub active_tasks: usize,
    /// Memory usage in bytes
    pub memory_usage: u64,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

/// Task completion message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskCompletion {
    /// Task ID
    pub task_id: String,
    /// Agent ID
    pub agent_id: String,
    /// Success flag
    pub success: bool,
    /// Result data
    pub result: Option<serde_json::Value>,
    /// Error message if failed
    pub error: Option<String>,
    /// Duration in milliseconds
    pub duration_ms: f64,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

/// Error report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorReport {
    /// Agent ID
    pub agent_id: String,
    /// Error code
    pub code: String,
    /// Error message
    pub message: String,
    /// Additional context
    pub context: Option<serde_json::Value>,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

/// Message acknowledgment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageAck {
    /// Message ID being acknowledged
    pub message_id: String,
    /// Processing status
    pub status: AckStatus,
    /// Optional response data
    pub response: Option<serde_json::Value>,
}

/// Acknowledgment status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum AckStatus {
    /// Message received and queued
    Received,
    /// Message processed successfully
    Processed,
    /// Message processing failed
    Failed,
    /// Message rejected
    Rejected,
}

/// Agent statistics
#[derive(Debug, Clone, Default)]
pub struct AgentStats {
    /// Total messages sent
    pub messages_sent: u64,
    /// Total messages received
    pub messages_received: u64,
    /// Total bytes sent
    pub bytes_sent: u64,
    /// Total bytes received
    pub bytes_received: u64,
    /// Average latency in milliseconds
    pub avg_latency_ms: f64,
    /// Connection start time
    pub connected_at: Option<DateTime<Utc>>,
    /// Last activity time
    pub last_activity: Option<DateTime<Utc>>,
    /// Error count
    pub errors: u64,
}

/// Task assignment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentTask {
    /// Task ID
    pub task_id: String,
    /// Task type
    pub task_type: TaskType,
    /// Task priority (0-10)
    pub priority: u8,
    /// Timeout in seconds
    pub timeout: u64,
    /// Task context
    pub context: serde_json::Value,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

/// Task types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", content = "data")]
pub enum TaskType {
    /// Pattern matching task
    PatternMatch {
        /// Current pattern
        current: Vec<f32>,
        /// Historical patterns
        historical: Vec<Vec<f32>>,
    },
    /// Strategy correlation task
    StrategyCorrelation {
        /// Strategies to correlate
        strategies: Vec<Vec<f32>>,
    },
    /// Feature engineering task
    FeatureEngineering {
        /// Raw data
        data: Vec<f32>,
    },
    /// Neural training task
    NeuralTraining {
        /// Training batch
        batch: Vec<Vec<f32>>,
        /// Labels
        labels: Vec<f32>,
    },
    /// Generic computation
    Compute {
        /// Computation description
        description: String,
        /// Input data
        input: serde_json::Value,
    },
}

/// ReasoningBank verdict
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningVerdict {
    /// Experience ID
    pub experience_id: String,
    /// Verdict score (-1 to 1)
    pub score: f64,
    /// Should the agent adapt?
    pub should_adapt: bool,
    /// Suggested changes
    pub suggested_changes: Vec<AdaptationSuggestion>,
    /// Confidence in verdict
    pub confidence: f64,
}

/// Adaptation suggestion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptationSuggestion {
    /// Parameter to change
    pub parameter: String,
    /// Current value
    pub current_value: f64,
    /// Suggested value
    pub suggested_value: f64,
    /// Reason for change
    pub reason: String,
}
