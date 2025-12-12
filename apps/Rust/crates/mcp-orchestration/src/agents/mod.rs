//! MCP Orchestration Agents
//!
//! This module contains the specialized agents for Claude-Flow MCP coordination
//! and swarm management using ruv-swarm topology with ultra-low latency routing.

pub mod mcp_orchestrator;
pub mod swarm_topology_manager;
pub mod agent_lifecycle_manager;
pub mod message_router;
pub mod load_balancer_agent;
pub mod health_monitor;

// Re-export key types
pub use mcp_orchestrator::McpOrchestratorAgent;
pub use swarm_topology_manager::SwarmTopologyManager;
pub use agent_lifecycle_manager::AgentLifecycleManager;
pub use message_router::MessageRouterAgent;
pub use load_balancer_agent::LoadBalancerAgent;
pub use health_monitor::HealthMonitorAgent;

use crate::{AgentConfig, SwarmType, HierarchyLevel, MCPOrchestrationError};
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};

/// Base agent trait for MCP orchestration
#[async_trait::async_trait]
pub trait MCPAgent: Send + Sync {
    /// Agent unique identifier
    fn id(&self) -> &str;
    
    /// Agent name
    fn name(&self) -> &str;
    
    /// Agent type
    fn agent_type(&self) -> SwarmType;
    
    /// Hierarchy level
    fn hierarchy_level(&self) -> HierarchyLevel;
    
    /// Start the agent
    async fn start(&self) -> Result<(), MCPOrchestrationError>;
    
    /// Stop the agent
    async fn stop(&self) -> Result<(), MCPOrchestrationError>;
    
    /// Get agent health status
    async fn health_check(&self) -> Result<AgentHealth, MCPOrchestrationError>;
    
    /// Process incoming message
    async fn process_message(&self, message: MCPMessage) -> Result<Option<MCPMessage>, MCPOrchestrationError>;
    
    /// Get agent metrics
    async fn get_metrics(&self) -> Result<AgentMetrics, MCPOrchestrationError>;
}

/// Agent health status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentHealth {
    pub status: crate::AgentStatus,
    pub cpu_usage: f64,
    pub memory_usage: f64,
    pub uptime_seconds: u64,
    pub last_error: Option<String>,
    pub response_time_us: u64,
}

/// Agent metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentMetrics {
    pub messages_processed: u64,
    pub messages_failed: u64,
    pub average_processing_time_us: u64,
    pub peak_memory_usage: u64,
    pub total_cpu_time_ms: u64,
    pub throughput_per_second: f64,
}

/// MCP message structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCPMessage {
    pub id: String,
    pub source: String,
    pub target: String,
    pub message_type: MCPMessageType,
    pub payload: serde_json::Value,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub priority: MessagePriority,
    pub routing_info: RoutingInfo,
}

/// MCP message types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MCPMessageType {
    Command,
    Response,
    Event,
    Heartbeat,
    Coordination,
    LoadBalancing,
    HealthCheck,
    Topology,
}

/// Message priority levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MessagePriority {
    Critical,
    High,
    Normal,
    Low,
}

/// Routing information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingInfo {
    pub route_id: String,
    pub hop_count: u8,
    pub max_hops: u8,
    pub latency_target_us: u64,
    pub compression_enabled: bool,
}
