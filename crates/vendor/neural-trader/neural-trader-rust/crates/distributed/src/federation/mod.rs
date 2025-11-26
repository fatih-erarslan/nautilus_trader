// Agentic-flow federation module for distributed agent coordination
//
// Provides:
// - Multiple topology patterns (hierarchical, mesh, adaptive)
// - Agent coordination and messaging
// - Consensus protocols
// - Work distribution and load balancing

mod topology;
mod coordination;
mod messaging;
mod consensus;

pub use topology::{FederationTopology, TopologyType, TopologyConfig};
pub use coordination::{AgentCoordinator, CoordinationStrategy};
pub use messaging::{MessageBus, Message, MessageType};
pub use consensus::{ConsensusProtocol, ConsensusResult};

use crate::{DistributedError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Agent identifier in the federation
pub type AgentId = String;

/// Agent metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentMetadata {
    /// Unique agent ID
    pub id: AgentId,

    /// Agent type/role
    pub agent_type: String,

    /// Capabilities
    pub capabilities: Vec<String>,

    /// Current status
    pub status: AgentStatus,

    /// Node hostname where agent is running
    pub node: String,

    /// Resource limits
    pub resources: ResourceLimits,

    /// Custom metadata
    pub metadata: HashMap<String, String>,
}

/// Agent status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AgentStatus {
    /// Agent is idle and available
    Idle,

    /// Agent is currently executing a task
    Busy,

    /// Agent is unavailable
    Unavailable,

    /// Agent encountered an error
    Error,
}

/// Resource limits for an agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    /// Maximum CPU cores
    pub max_cpu: f32,

    /// Maximum memory (MB)
    pub max_memory_mb: u32,

    /// Maximum concurrent tasks
    pub max_concurrent_tasks: u32,
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            max_cpu: 2.0,
            max_memory_mb: 2048,
            max_concurrent_tasks: 5,
        }
    }
}

/// Task to be executed by an agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    /// Unique task ID
    pub id: Uuid,

    /// Task type
    pub task_type: String,

    /// Task payload
    pub payload: serde_json::Value,

    /// Priority (higher = more important)
    pub priority: u32,

    /// Required capabilities
    pub required_capabilities: Vec<String>,

    /// Task deadline
    pub deadline: Option<chrono::DateTime<chrono::Utc>>,

    /// Assigned agent (if any)
    pub assigned_to: Option<AgentId>,
}

/// Task result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskResult {
    /// Task ID
    pub task_id: Uuid,

    /// Agent that executed the task
    pub agent_id: AgentId,

    /// Success flag
    pub success: bool,

    /// Result data
    pub result: serde_json::Value,

    /// Error message (if failed)
    pub error: Option<String>,

    /// Execution duration (ms)
    pub duration_ms: u64,
}

/// Federation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederationConfig {
    /// Federation ID
    pub id: String,

    /// Topology type
    pub topology: TopologyType,

    /// Maximum number of agents
    pub max_agents: usize,

    /// Enable consensus protocols
    pub enable_consensus: bool,

    /// Consensus threshold (fraction of agents that must agree)
    pub consensus_threshold: f64,

    /// Enable work stealing for load balancing
    pub enable_work_stealing: bool,

    /// Heartbeat interval (seconds)
    pub heartbeat_interval: u64,

    /// Agent timeout (seconds)
    pub agent_timeout: u64,
}

impl Default for FederationConfig {
    fn default() -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            topology: TopologyType::Mesh,
            max_agents: 50,
            enable_consensus: true,
            consensus_threshold: 0.66,
            enable_work_stealing: true,
            heartbeat_interval: 10,
            agent_timeout: 60,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_metadata_default() {
        let metadata = AgentMetadata {
            id: "agent-1".to_string(),
            agent_type: "worker".to_string(),
            capabilities: vec!["compute".to_string()],
            status: AgentStatus::Idle,
            node: "node-1".to_string(),
            resources: ResourceLimits::default(),
            metadata: HashMap::new(),
        };

        assert_eq!(metadata.id, "agent-1");
        assert_eq!(metadata.status, AgentStatus::Idle);
    }

    #[test]
    fn test_task_creation() {
        let task = Task {
            id: Uuid::new_v4(),
            task_type: "compute".to_string(),
            payload: serde_json::json!({"data": "test"}),
            priority: 5,
            required_capabilities: vec!["compute".to_string()],
            deadline: None,
            assigned_to: None,
        };

        assert_eq!(task.priority, 5);
        assert!(task.assigned_to.is_none());
    }

    #[test]
    fn test_federation_config_default() {
        let config = FederationConfig::default();
        assert_eq!(config.topology, TopologyType::Mesh);
        assert_eq!(config.max_agents, 50);
        assert!(config.enable_consensus);
    }
}
