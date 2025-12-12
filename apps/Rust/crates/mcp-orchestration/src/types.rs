//! Core types and data structures for MCP orchestration.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use uuid::Uuid;

/// Unique identifier for agents in the swarm
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AgentId(pub Uuid);

impl AgentId {
    /// Generate a new random agent ID
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
    
    /// Create an agent ID from a UUID
    pub fn from_uuid(uuid: Uuid) -> Self {
        Self(uuid)
    }
    
    /// Get the underlying UUID
    pub fn uuid(&self) -> Uuid {
        self.0
    }
}

impl fmt::Display for AgentId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Default for AgentId {
    fn default() -> Self {
        Self::new()
    }
}

/// Types of agents in the TENGRI trading swarm
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AgentType {
    /// Risk management agent
    Risk,
    /// Neural network forecasting agent
    Neural,
    /// Quantum uncertainty agent
    Quantum,
    /// Market data agent
    MarketData,
    /// Order execution agent
    OrderExecution,
    /// Portfolio management agent
    Portfolio,
    /// Technical analysis agent
    TechnicalAnalysis,
    /// Sentiment analysis agent
    SentimentAnalysis,
    /// News processing agent
    NewsProcessing,
    /// Backtesting agent
    Backtesting,
    /// Performance monitoring agent
    PerformanceMonitoring,
    /// Memory management agent
    MemoryManagement,
    /// Configuration management agent
    ConfigManagement,
    /// Logging agent
    Logging,
    /// Metrics collection agent
    MetricsCollection,
    /// Alert management agent
    AlertManagement,
    /// System health agent
    SystemHealth,
    /// Load balancing agent
    LoadBalancing,
    /// Fault tolerance agent
    FaultTolerance,
    /// MCP orchestration agent
    McpOrchestration,
}

impl AgentType {
    /// Get all agent types in the swarm
    pub fn all() -> Vec<Self> {
        vec![
            Self::Risk,
            Self::Neural,
            Self::Quantum,
            Self::MarketData,
            Self::OrderExecution,
            Self::Portfolio,
            Self::TechnicalAnalysis,
            Self::SentimentAnalysis,
            Self::NewsProcessing,
            Self::Backtesting,
            Self::PerformanceMonitoring,
            Self::MemoryManagement,
            Self::ConfigManagement,
            Self::Logging,
            Self::MetricsCollection,
            Self::AlertManagement,
            Self::SystemHealth,
            Self::LoadBalancing,
            Self::FaultTolerance,
            Self::McpOrchestration,
        ]
    }
    
    /// Get the priority level for this agent type
    pub fn priority(&self) -> TaskPriority {
        match self {
            Self::Risk => TaskPriority::Critical,
            Self::Neural => TaskPriority::High,
            Self::Quantum => TaskPriority::High,
            Self::MarketData => TaskPriority::High,
            Self::OrderExecution => TaskPriority::Critical,
            Self::Portfolio => TaskPriority::High,
            Self::TechnicalAnalysis => TaskPriority::Medium,
            Self::SentimentAnalysis => TaskPriority::Medium,
            Self::NewsProcessing => TaskPriority::Medium,
            Self::Backtesting => TaskPriority::Low,
            Self::PerformanceMonitoring => TaskPriority::Medium,
            Self::MemoryManagement => TaskPriority::High,
            Self::ConfigManagement => TaskPriority::Medium,
            Self::Logging => TaskPriority::Low,
            Self::MetricsCollection => TaskPriority::Low,
            Self::AlertManagement => TaskPriority::High,
            Self::SystemHealth => TaskPriority::High,
            Self::LoadBalancing => TaskPriority::Medium,
            Self::FaultTolerance => TaskPriority::High,
            Self::McpOrchestration => TaskPriority::Critical,
        }
    }
}

impl fmt::Display for AgentType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Risk => write!(f, "Risk"),
            Self::Neural => write!(f, "Neural"),
            Self::Quantum => write!(f, "Quantum"),
            Self::MarketData => write!(f, "MarketData"),
            Self::OrderExecution => write!(f, "OrderExecution"),
            Self::Portfolio => write!(f, "Portfolio"),
            Self::TechnicalAnalysis => write!(f, "TechnicalAnalysis"),
            Self::SentimentAnalysis => write!(f, "SentimentAnalysis"),
            Self::NewsProcessing => write!(f, "NewsProcessing"),
            Self::Backtesting => write!(f, "Backtesting"),
            Self::PerformanceMonitoring => write!(f, "PerformanceMonitoring"),
            Self::MemoryManagement => write!(f, "MemoryManagement"),
            Self::ConfigManagement => write!(f, "ConfigManagement"),
            Self::Logging => write!(f, "Logging"),
            Self::MetricsCollection => write!(f, "MetricsCollection"),
            Self::AlertManagement => write!(f, "AlertManagement"),
            Self::SystemHealth => write!(f, "SystemHealth"),
            Self::LoadBalancing => write!(f, "LoadBalancing"),
            Self::FaultTolerance => write!(f, "FaultTolerance"),
            Self::McpOrchestration => write!(f, "McpOrchestration"),
        }
    }
}

/// Agent state in the swarm
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AgentState {
    /// Agent is starting up
    Starting,
    /// Agent is running and available
    Running,
    /// Agent is busy processing tasks
    Busy,
    /// Agent is temporarily unavailable
    Unavailable,
    /// Agent is shutting down
    Stopping,
    /// Agent has stopped
    Stopped,
    /// Agent has failed
    Failed,
}

impl fmt::Display for AgentState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Starting => write!(f, "Starting"),
            Self::Running => write!(f, "Running"),
            Self::Busy => write!(f, "Busy"),
            Self::Unavailable => write!(f, "Unavailable"),
            Self::Stopping => write!(f, "Stopping"),
            Self::Stopped => write!(f, "Stopped"),
            Self::Failed => write!(f, "Failed"),
        }
    }
}

/// Task priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum TaskPriority {
    /// Critical tasks that must be processed immediately
    Critical = 0,
    /// High priority tasks
    High = 1,
    /// Medium priority tasks
    Medium = 2,
    /// Low priority tasks
    Low = 3,
}

impl fmt::Display for TaskPriority {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Critical => write!(f, "Critical"),
            Self::High => write!(f, "High"),
            Self::Medium => write!(f, "Medium"),
            Self::Low => write!(f, "Low"),
        }
    }
}

/// Unique identifier for tasks
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TaskId(pub Uuid);

impl TaskId {
    /// Generate a new random task ID
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
    
    /// Create a task ID from a UUID
    pub fn from_uuid(uuid: Uuid) -> Self {
        Self(uuid)
    }
    
    /// Get the underlying UUID
    pub fn uuid(&self) -> Uuid {
        self.0
    }
}

impl fmt::Display for TaskId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Default for TaskId {
    fn default() -> Self {
        Self::new()
    }
}

/// Task status in the system
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskStatus {
    /// Task is queued and waiting for processing
    Queued,
    /// Task is being processed
    Processing,
    /// Task completed successfully
    Completed,
    /// Task failed with an error
    Failed,
    /// Task was cancelled
    Cancelled,
    /// Task timed out
    TimedOut,
}

impl fmt::Display for TaskStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Queued => write!(f, "Queued"),
            Self::Processing => write!(f, "Processing"),
            Self::Completed => write!(f, "Completed"),
            Self::Failed => write!(f, "Failed"),
            Self::Cancelled => write!(f, "Cancelled"),
            Self::TimedOut => write!(f, "TimedOut"),
        }
    }
}

/// Message identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MessageId(pub Uuid);

impl MessageId {
    /// Generate a new random message ID
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
    
    /// Create a message ID from a UUID
    pub fn from_uuid(uuid: Uuid) -> Self {
        Self(uuid)
    }
    
    /// Get the underlying UUID
    pub fn uuid(&self) -> Uuid {
        self.0
    }
}

impl fmt::Display for MessageId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Default for MessageId {
    fn default() -> Self {
        Self::new()
    }
}

/// Message types for inter-agent communication
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MessageType {
    /// Request message
    Request,
    /// Response message
    Response,
    /// Event notification
    Event,
    /// Heartbeat message
    Heartbeat,
    /// System control message
    Control,
    /// Error message
    Error,
}

impl fmt::Display for MessageType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Request => write!(f, "Request"),
            Self::Response => write!(f, "Response"),
            Self::Event => write!(f, "Event"),
            Self::Heartbeat => write!(f, "Heartbeat"),
            Self::Control => write!(f, "Control"),
            Self::Error => write!(f, "Error"),
        }
    }
}

/// Health status of system components
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum HealthStatus {
    /// Component is healthy
    Healthy,
    /// Component is degraded but functional
    Degraded,
    /// Component is unhealthy
    Unhealthy,
    /// Component health is unknown
    Unknown,
}

impl fmt::Display for HealthStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Healthy => write!(f, "Healthy"),
            Self::Degraded => write!(f, "Degraded"),
            Self::Unhealthy => write!(f, "Unhealthy"),
            Self::Unknown => write!(f, "Unknown"),
        }
    }
}

/// Load balancing strategies
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LoadBalancingStrategy {
    /// Round-robin distribution
    RoundRobin,
    /// Least connections
    LeastConnections,
    /// Weighted round-robin
    WeightedRoundRobin,
    /// Random selection
    Random,
    /// Consistent hashing
    ConsistentHashing,
    /// Performance-based
    PerformanceBased,
}

impl fmt::Display for LoadBalancingStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RoundRobin => write!(f, "RoundRobin"),
            Self::LeastConnections => write!(f, "LeastConnections"),
            Self::WeightedRoundRobin => write!(f, "WeightedRoundRobin"),
            Self::Random => write!(f, "Random"),
            Self::ConsistentHashing => write!(f, "ConsistentHashing"),
            Self::PerformanceBased => write!(f, "PerformanceBased"),
        }
    }
}

/// Recovery strategies for fault tolerance
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecoveryStrategy {
    /// Restart the failed component
    Restart,
    /// Failover to backup
    Failover,
    /// Circuit breaker pattern
    CircuitBreaker,
    /// Retry with exponential backoff
    Retry,
    /// Graceful degradation
    GracefulDegradation,
}

impl fmt::Display for RecoveryStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Restart => write!(f, "Restart"),
            Self::Failover => write!(f, "Failover"),
            Self::CircuitBreaker => write!(f, "CircuitBreaker"),
            Self::Retry => write!(f, "Retry"),
            Self::GracefulDegradation => write!(f, "GracefulDegradation"),
        }
    }
}

/// System timestamps
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Timestamp(pub u64);

impl Timestamp {
    /// Create a new timestamp from the current time
    pub fn now() -> Self {
        Self(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
        )
    }
    
    /// Create a timestamp from milliseconds since Unix epoch
    pub fn from_millis(millis: u64) -> Self {
        Self(millis)
    }
    
    /// Get the timestamp as milliseconds since Unix epoch
    pub fn as_millis(&self) -> u64 {
        self.0
    }
    
    /// Get the duration since this timestamp
    pub fn elapsed(&self) -> Duration {
        Duration::from_millis(Self::now().0 - self.0)
    }
}

impl fmt::Display for Timestamp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// System metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetrics {
    /// CPU usage percentage
    pub cpu_usage: f64,
    /// Memory usage in bytes
    pub memory_usage: u64,
    /// Network I/O statistics
    pub network_io: NetworkMetrics,
    /// Task processing metrics
    pub task_metrics: TaskMetrics,
    /// Agent metrics
    pub agent_metrics: HashMap<AgentId, AgentMetrics>,
    /// Timestamp of metrics collection
    pub timestamp: Timestamp,
}

/// Network I/O metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkMetrics {
    /// Bytes sent
    pub bytes_sent: u64,
    /// Bytes received
    pub bytes_received: u64,
    /// Packets sent
    pub packets_sent: u64,
    /// Packets received
    pub packets_received: u64,
    /// Error count
    pub errors: u64,
}

/// Task processing metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskMetrics {
    /// Total tasks processed
    pub total_processed: u64,
    /// Tasks completed successfully
    pub completed: u64,
    /// Tasks failed
    pub failed: u64,
    /// Tasks cancelled
    pub cancelled: u64,
    /// Average processing time in milliseconds
    pub avg_processing_time: f64,
    /// Queue size
    pub queue_size: usize,
}

/// Agent-specific metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentMetrics {
    /// Agent ID
    pub agent_id: AgentId,
    /// Agent type
    pub agent_type: AgentType,
    /// Current state
    pub state: AgentState,
    /// Tasks processed
    pub tasks_processed: u64,
    /// CPU usage
    pub cpu_usage: f64,
    /// Memory usage
    pub memory_usage: u64,
    /// Last heartbeat timestamp
    pub last_heartbeat: Timestamp,
    /// Health status
    pub health_status: HealthStatus,
}

/// Configuration for system components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentConfig {
    /// Component name
    pub name: String,
    /// Configuration parameters
    pub parameters: HashMap<String, String>,
    /// Resource limits
    pub limits: ResourceLimits,
}

/// Resource limits for components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    /// Maximum memory usage in bytes
    pub max_memory: u64,
    /// Maximum CPU usage percentage
    pub max_cpu: f64,
    /// Maximum network bandwidth in bytes/second
    pub max_network_bandwidth: u64,
    /// Maximum task queue size
    pub max_task_queue_size: usize,
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            max_memory: 1024 * 1024 * 1024, // 1GB
            max_cpu: 80.0,                  // 80%
            max_network_bandwidth: 1024 * 1024 * 10, // 10MB/s
            max_task_queue_size: 1000,      // 1000 tasks
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_agent_id_creation() {
        let id1 = AgentId::new();
        let id2 = AgentId::new();
        assert_ne!(id1, id2);
    }
    
    #[test]
    fn test_agent_type_priority() {
        assert_eq!(AgentType::Risk.priority(), TaskPriority::Critical);
        assert_eq!(AgentType::Neural.priority(), TaskPriority::High);
        assert_eq!(AgentType::Backtesting.priority(), TaskPriority::Low);
    }
    
    #[test]
    fn test_timestamp_creation() {
        let ts1 = Timestamp::now();
        std::thread::sleep(std::time::Duration::from_millis(1));
        let ts2 = Timestamp::now();
        assert!(ts2.0 > ts1.0);
    }
    
    #[test]
    fn test_task_priority_ordering() {
        assert!(TaskPriority::Critical < TaskPriority::High);
        assert!(TaskPriority::High < TaskPriority::Medium);
        assert!(TaskPriority::Medium < TaskPriority::Low);
    }
}