//! # Base Data Agent Framework
//!
//! Foundation for all data processing agents in the ruv-swarm system.
//! Provides common interfaces, state management, and coordination capabilities.

use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, mpsc, oneshot};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use tracing::{info, warn, error, debug};
use async_trait::async_trait;

/// Unique identifier for data agents
pub type DataAgentId = uuid::Uuid;

/// Base trait for all data processing agents
#[async_trait]
pub trait DataAgent: Send + Sync {
    /// Get the agent's unique identifier
    fn get_id(&self) -> DataAgentId;
    
    /// Get the agent's type
    fn get_type(&self) -> DataAgentType;
    
    /// Get the agent's current state
    async fn get_state(&self) -> DataAgentState;
    
    /// Get agent information
    async fn get_info(&self) -> DataAgentInfo;
    
    /// Start the agent
    async fn start(&self) -> Result<()>;
    
    /// Stop the agent
    async fn stop(&self) -> Result<()>;
    
    /// Process data
    async fn process(&self, data: DataMessage) -> Result<DataMessage>;
    
    /// Health check
    async fn health_check(&self) -> Result<HealthStatus>;
    
    /// Get performance metrics
    async fn get_metrics(&self) -> Result<AgentMetrics>;
    
    /// Reset agent state
    async fn reset(&self) -> Result<()>;
    
    /// Handle coordination message
    async fn handle_coordination(&self, message: CoordinationMessage) -> Result<()>;
}

/// Data agent types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DataAgentType {
    DataIngestion,
    FeatureEngineering,
    DataValidation,
    StreamProcessing,
    DataTransformation,
    CacheManagement,
}

impl std::fmt::Display for DataAgentType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DataAgentType::DataIngestion => write!(f, "DataIngestion"),
            DataAgentType::FeatureEngineering => write!(f, "FeatureEngineering"),
            DataAgentType::DataValidation => write!(f, "DataValidation"),
            DataAgentType::StreamProcessing => write!(f, "StreamProcessing"),
            DataAgentType::DataTransformation => write!(f, "DataTransformation"),
            DataAgentType::CacheManagement => write!(f, "CacheManagement"),
        }
    }
}

/// Data agent state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataAgentState {
    Initializing,
    Ready,
    Running,
    Paused,
    Stopping,
    Stopped,
    Error(String),
}

/// Data agent information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataAgentInfo {
    pub id: DataAgentId,
    pub agent_type: DataAgentType,
    pub version: String,
    pub state: DataAgentState,
    pub capabilities: Vec<String>,
    pub resource_requirements: ResourceRequirements,
    pub performance_profile: PerformanceProfile,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub last_updated: chrono::DateTime<chrono::Utc>,
}

/// Resource requirements for agents
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequirements {
    pub min_memory_mb: usize,
    pub max_memory_mb: usize,
    pub cpu_cores: usize,
    pub network_bandwidth_mbps: f64,
    pub disk_space_mb: usize,
}

/// Performance profile for agents
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceProfile {
    pub max_throughput_ops_per_sec: f64,
    pub target_latency_us: u64,
    pub memory_efficiency: f64,
    pub cpu_efficiency: f64,
}

/// Data message for inter-agent communication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataMessage {
    pub id: uuid::Uuid,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub source: DataAgentId,
    pub destination: Option<DataAgentId>,
    pub message_type: DataMessageType,
    pub payload: serde_json::Value,
    pub metadata: MessageMetadata,
}

/// Data message types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataMessageType {
    MarketData,
    ProcessedData,
    Features,
    ValidationResult,
    StreamData,
    TransformedData,
    CacheRequest,
    CacheResponse,
    CoordinationMessage,
    HealthCheck,
    MetricsRequest,
    MetricsResponse,
}

/// Message metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageMetadata {
    pub priority: MessagePriority,
    pub expires_at: Option<chrono::DateTime<chrono::Utc>>,
    pub retry_count: u32,
    pub trace_id: String,
    pub span_id: String,
}

/// Message priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum MessagePriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

/// Coordination message for swarm coordination
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinationMessage {
    pub id: uuid::Uuid,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub source: DataAgentId,
    pub coordination_type: CoordinationType,
    pub payload: serde_json::Value,
}

/// Coordination message types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CoordinationType {
    TaskAssignment,
    LoadBalancing,
    StateSync,
    HealthUpdate,
    ResourceRequest,
    ResourceRelease,
    Emergency,
}

/// Health status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthStatus {
    pub status: HealthLevel,
    pub last_check: chrono::DateTime<chrono::Utc>,
    pub uptime: Duration,
    pub issues: Vec<String>,
    pub metrics: HealthMetrics,
}

/// Health levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HealthLevel {
    Healthy,
    Warning,
    Critical,
    Unknown,
}

/// Health metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthMetrics {
    pub cpu_usage_percent: f64,
    pub memory_usage_mb: f64,
    pub network_usage_mbps: f64,
    pub disk_usage_mb: f64,
    pub error_rate: f64,
    pub response_time_ms: f64,
}

/// Agent performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentMetrics {
    pub messages_processed: u64,
    pub messages_failed: u64,
    pub average_latency_us: f64,
    pub throughput_ops_per_sec: f64,
    pub memory_usage_mb: f64,
    pub cpu_usage_percent: f64,
    pub uptime_seconds: u64,
    pub last_reset: chrono::DateTime<chrono::Utc>,
}

/// Base data agent implementation
pub struct BaseDataAgent {
    pub id: DataAgentId,
    pub agent_type: DataAgentType,
    pub state: Arc<RwLock<DataAgentState>>,
    pub info: Arc<RwLock<DataAgentInfo>>,
    pub metrics: Arc<RwLock<AgentMetrics>>,
    pub message_tx: mpsc::UnboundedSender<DataMessage>,
    pub message_rx: Arc<RwLock<mpsc::UnboundedReceiver<DataMessage>>>,
    pub coordination_tx: mpsc::UnboundedSender<CoordinationMessage>,
    pub coordination_rx: Arc<RwLock<mpsc::UnboundedReceiver<CoordinationMessage>>>,
    pub shutdown_tx: Option<oneshot::Sender<()>>,
    pub shutdown_rx: Arc<RwLock<Option<oneshot::Receiver<()>>>>,
    pub start_time: Instant,
}

impl BaseDataAgent {
    /// Create a new base data agent
    pub fn new(agent_type: DataAgentType) -> Self {
        let id = DataAgentId::new_v4();
        let (message_tx, message_rx) = mpsc::unbounded_channel();
        let (coordination_tx, coordination_rx) = mpsc::unbounded_channel();
        let (shutdown_tx, shutdown_rx) = oneshot::channel();
        
        let info = DataAgentInfo {
            id,
            agent_type,
            version: env!("CARGO_PKG_VERSION").to_string(),
            state: DataAgentState::Initializing,
            capabilities: Vec::new(),
            resource_requirements: ResourceRequirements {
                min_memory_mb: 64,
                max_memory_mb: 512,
                cpu_cores: 1,
                network_bandwidth_mbps: 100.0,
                disk_space_mb: 100,
            },
            performance_profile: PerformanceProfile {
                max_throughput_ops_per_sec: 10000.0,
                target_latency_us: 100,
                memory_efficiency: 0.8,
                cpu_efficiency: 0.8,
            },
            created_at: chrono::Utc::now(),
            last_updated: chrono::Utc::now(),
        };
        
        let metrics = AgentMetrics {
            messages_processed: 0,
            messages_failed: 0,
            average_latency_us: 0.0,
            throughput_ops_per_sec: 0.0,
            memory_usage_mb: 0.0,
            cpu_usage_percent: 0.0,
            uptime_seconds: 0,
            last_reset: chrono::Utc::now(),
        };
        
        Self {
            id,
            agent_type,
            state: Arc::new(RwLock::new(DataAgentState::Initializing)),
            info: Arc::new(RwLock::new(info)),
            metrics: Arc::new(RwLock::new(metrics)),
            message_tx,
            message_rx: Arc::new(RwLock::new(message_rx)),
            coordination_tx,
            coordination_rx: Arc::new(RwLock::new(coordination_rx)),
            shutdown_tx: Some(shutdown_tx),
            shutdown_rx: Arc::new(RwLock::new(Some(shutdown_rx))),
            start_time: Instant::now(),
        }
    }
    
    /// Update agent state
    pub async fn update_state(&self, new_state: DataAgentState) -> Result<()> {
        {
            let mut state = self.state.write().await;
            *state = new_state.clone();
        }
        
        {
            let mut info = self.info.write().await;
            info.state = new_state;
            info.last_updated = chrono::Utc::now();
        }
        
        debug!("Agent {} state updated to {:?}", self.id, new_state);
        Ok(())
    }
    
    /// Update metrics
    pub async fn update_metrics(&self, update: MetricsUpdate) -> Result<()> {
        let mut metrics = self.metrics.write().await;
        
        match update {
            MetricsUpdate::MessageProcessed(latency_us) => {
                metrics.messages_processed += 1;
                metrics.average_latency_us = (metrics.average_latency_us + latency_us) / 2.0;
            }
            MetricsUpdate::MessageFailed => {
                metrics.messages_failed += 1;
            }
            MetricsUpdate::ResourceUsage { memory_mb, cpu_percent } => {
                metrics.memory_usage_mb = memory_mb;
                metrics.cpu_usage_percent = cpu_percent;
            }
            MetricsUpdate::Throughput(ops_per_sec) => {
                metrics.throughput_ops_per_sec = ops_per_sec;
            }
        }
        
        metrics.uptime_seconds = self.start_time.elapsed().as_secs();
        Ok(())
    }
    
    /// Send a message to another agent
    pub async fn send_message(&self, message: DataMessage) -> Result<()> {
        self.message_tx.send(message)?;
        Ok(())
    }
    
    /// Send a coordination message
    pub async fn send_coordination(&self, message: CoordinationMessage) -> Result<()> {
        self.coordination_tx.send(message)?;
        Ok(())
    }
    
    /// Check if agent should shutdown
    pub async fn should_shutdown(&self) -> bool {
        if let Some(mut rx) = self.shutdown_rx.write().await.take() {
            rx.try_recv().is_ok()
        } else {
            false
        }
    }
}

/// Metrics update types
#[derive(Debug, Clone)]
pub enum MetricsUpdate {
    MessageProcessed(f64),
    MessageFailed,
    ResourceUsage { memory_mb: f64, cpu_percent: f64 },
    Throughput(f64),
}

/// Data agent builder
pub struct DataAgentBuilder {
    agent_type: DataAgentType,
    capabilities: Vec<String>,
    resource_requirements: Option<ResourceRequirements>,
    performance_profile: Option<PerformanceProfile>,
}

impl DataAgentBuilder {
    /// Create a new data agent builder
    pub fn new(agent_type: DataAgentType) -> Self {
        Self {
            agent_type,
            capabilities: Vec::new(),
            resource_requirements: None,
            performance_profile: None,
        }
    }
    
    /// Add capability
    pub fn with_capability(mut self, capability: String) -> Self {
        self.capabilities.push(capability);
        self
    }
    
    /// Set resource requirements
    pub fn with_resource_requirements(mut self, requirements: ResourceRequirements) -> Self {
        self.resource_requirements = Some(requirements);
        self
    }
    
    /// Set performance profile
    pub fn with_performance_profile(mut self, profile: PerformanceProfile) -> Self {
        self.performance_profile = Some(profile);
        self
    }
    
    /// Build the agent
    pub fn build(self) -> BaseDataAgent {
        let mut agent = BaseDataAgent::new(self.agent_type);
        
        // Update info with builder settings
        tokio::spawn(async move {
            let mut info = agent.info.write().await;
            info.capabilities = self.capabilities;
            
            if let Some(requirements) = self.resource_requirements {
                info.resource_requirements = requirements;
            }
            
            if let Some(profile) = self.performance_profile {
                info.performance_profile = profile;
            }
        });
        
        agent
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::test;
    
    #[test]
    async fn test_base_agent_creation() {
        let agent = BaseDataAgent::new(DataAgentType::DataIngestion);
        assert_eq!(agent.agent_type, DataAgentType::DataIngestion);
        
        let state = agent.state.read().await;
        assert!(matches!(*state, DataAgentState::Initializing));
    }
    
    #[test]
    async fn test_agent_builder() {
        let builder = DataAgentBuilder::new(DataAgentType::FeatureEngineering)
            .with_capability("quantum_enhancement".to_string())
            .with_capability("real_time_processing".to_string());
        
        let agent = builder.build();
        assert_eq!(agent.agent_type, DataAgentType::FeatureEngineering);
    }
    
    #[test]
    async fn test_state_updates() {
        let agent = BaseDataAgent::new(DataAgentType::DataValidation);
        
        let result = agent.update_state(DataAgentState::Ready).await;
        assert!(result.is_ok());
        
        let state = agent.state.read().await;
        assert!(matches!(*state, DataAgentState::Ready));
    }
    
    #[test]
    async fn test_metrics_updates() {
        let agent = BaseDataAgent::new(DataAgentType::StreamProcessing);
        
        let result = agent.update_metrics(MetricsUpdate::MessageProcessed(50.0)).await;
        assert!(result.is_ok());
        
        let metrics = agent.metrics.read().await;
        assert_eq!(metrics.messages_processed, 1);
        assert_eq!(metrics.average_latency_us, 50.0);
    }
}