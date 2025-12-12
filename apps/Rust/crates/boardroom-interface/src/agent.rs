//! Agent representation and management

use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fmt;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

/// Unique identifier for an agent
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AgentId(Uuid);

impl AgentId {
    /// Create a new random agent ID
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    /// Create from existing UUID
    pub fn from_uuid(uuid: Uuid) -> Self {
        Self(uuid)
    }

    /// Get the inner UUID
    pub fn as_uuid(&self) -> &Uuid {
        &self.0
    }
}

impl Default for AgentId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for AgentId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Agent capabilities that define what an agent can do
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AgentCapability {
    /// Can perform market analysis
    MarketAnalysis,
    /// Can execute trades
    Trading,
    /// Can detect whales
    WhaleDetection,
    /// Can perform risk assessment
    RiskAssessment,
    /// Can generate predictions
    Prediction,
    /// Can coordinate other agents
    Coordination,
    /// Custom capability
    Custom(String),
}

/// Agent state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AgentState {
    /// Agent is starting up
    Initializing,
    /// Agent is ready to receive tasks
    Ready,
    /// Agent is currently busy
    Busy,
    /// Agent is shutting down
    Stopping,
    /// Agent has stopped
    Stopped,
    /// Agent has encountered an error
    Error,
}

/// Information about an agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentInfo {
    pub id: AgentId,
    pub name: String,
    pub capabilities: HashSet<AgentCapability>,
    pub state: AgentState,
    pub endpoint: String,
    pub metadata: serde_json::Value,
    pub last_heartbeat: chrono::DateTime<chrono::Utc>,
}

impl AgentInfo {
    /// Create new agent info
    pub fn new(
        id: AgentId,
        name: impl Into<String>,
        endpoint: impl Into<String>,
    ) -> Self {
        Self {
            id,
            name: name.into(),
            capabilities: HashSet::new(),
            state: AgentState::Initializing,
            endpoint: endpoint.into(),
            metadata: serde_json::Value::Object(serde_json::Map::new()),
            last_heartbeat: chrono::Utc::now(),
        }
    }

    /// Add a capability to the agent
    pub fn with_capability(mut self, capability: AgentCapability) -> Self {
        self.capabilities.insert(capability);
        self
    }

    /// Set agent metadata
    pub fn with_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = metadata;
        self
    }

    /// Update heartbeat timestamp
    pub fn update_heartbeat(&mut self) {
        self.last_heartbeat = chrono::Utc::now();
    }

    /// Check if agent has a specific capability
    pub fn has_capability(&self, capability: &AgentCapability) -> bool {
        self.capabilities.contains(capability)
    }

    /// Check if agent is healthy (recent heartbeat)
    pub fn is_healthy(&self, timeout_secs: i64) -> bool {
        let now = chrono::Utc::now();
        let elapsed = now - self.last_heartbeat;
        elapsed.num_seconds() < timeout_secs && self.state != AgentState::Error
    }
}

/// Agent trait that all agents must implement
#[async_trait::async_trait]
pub trait Agent: Send + Sync {
    /// Get agent ID
    fn id(&self) -> AgentId;

    /// Get agent info
    async fn info(&self) -> AgentInfo;

    /// Initialize the agent
    async fn initialize(&mut self) -> anyhow::Result<()>;

    /// Start the agent
    async fn start(&mut self) -> anyhow::Result<()>;

    /// Stop the agent
    async fn stop(&mut self) -> anyhow::Result<()>;

    /// Handle incoming message
    async fn handle_message(&mut self, message: crate::message::Message) -> anyhow::Result<()>;

    /// Get current state
    async fn state(&self) -> AgentState;

    /// Health check
    async fn health_check(&self) -> anyhow::Result<()> {
        Ok(())
    }
}

/// Base agent implementation
pub struct BaseAgent {
    info: Arc<RwLock<AgentInfo>>,
}

impl BaseAgent {
    /// Create new base agent
    pub fn new(name: impl Into<String>, endpoint: impl Into<String>) -> Self {
        let id = AgentId::new();
        let info = AgentInfo::new(id, name, endpoint);
        
        Self {
            info: Arc::new(RwLock::new(info)),
        }
    }

    /// Add capability
    pub async fn add_capability(&self, capability: AgentCapability) {
        let mut info = self.info.write().await;
        info.capabilities.insert(capability);
    }

    /// Set state
    pub async fn set_state(&self, state: AgentState) {
        let mut info = self.info.write().await;
        info.state = state;
    }

    /// Update heartbeat
    pub async fn heartbeat(&self) {
        let mut info = self.info.write().await;
        info.update_heartbeat();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_id() {
        let id1 = AgentId::new();
        let id2 = AgentId::new();
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_agent_info() {
        let id = AgentId::new();
        let mut info = AgentInfo::new(id, "test-agent", "tcp://localhost:5555");
        
        info = info
            .with_capability(AgentCapability::Trading)
            .with_capability(AgentCapability::MarketAnalysis);
        
        assert_eq!(info.capabilities.len(), 2);
        assert!(info.has_capability(&AgentCapability::Trading));
        assert!(!info.has_capability(&AgentCapability::WhaleDetection));
    }

    #[test]
    fn test_agent_health() {
        let id = AgentId::new();
        let mut info = AgentInfo::new(id, "test-agent", "tcp://localhost:5555");
        
        // Should be healthy initially
        assert!(info.is_healthy(60));
        
        // Set to error state
        info.state = AgentState::Error;
        assert!(!info.is_healthy(60));
    }
}