//! Agent management and registry for the MCP orchestration system.

use crate::communication::{CommunicationLayer, Message, MessageRouter};
use crate::error::{OrchestrationError, Result};
use crate::types::{AgentId, AgentState, AgentType, HealthStatus, Timestamp};
use async_trait::async_trait;
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tokio::time::{interval, sleep};
use tracing::{debug, error, info, warn};

/// Agent information and metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentInfo {
    /// Unique agent identifier
    pub id: AgentId,
    /// Agent type
    pub agent_type: AgentType,
    /// Current agent state
    pub state: AgentState,
    /// Agent name/description
    pub name: String,
    /// Agent version
    pub version: String,
    /// Agent capabilities
    pub capabilities: Vec<String>,
    /// Agent configuration
    pub config: HashMap<String, String>,
    /// Health status
    pub health_status: HealthStatus,
    /// Last heartbeat timestamp
    pub last_heartbeat: Timestamp,
    /// Registration timestamp
    pub registered_at: Timestamp,
    /// Agent endpoint information
    pub endpoint: Option<String>,
    /// Resource usage statistics
    pub resource_usage: ResourceUsage,
    /// Agent-specific metadata
    pub metadata: HashMap<String, String>,
}

impl AgentInfo {
    /// Create new agent information
    pub fn new(
        id: AgentId,
        agent_type: AgentType,
        name: String,
        version: String,
    ) -> Self {
        Self {
            id,
            agent_type,
            state: AgentState::Starting,
            name,
            version,
            capabilities: Vec::new(),
            config: HashMap::new(),
            health_status: HealthStatus::Unknown,
            last_heartbeat: Timestamp::now(),
            registered_at: Timestamp::now(),
            endpoint: None,
            resource_usage: ResourceUsage::default(),
            metadata: HashMap::new(),
        }
    }
    
    /// Update agent state
    pub fn update_state(&mut self, state: AgentState) {
        self.state = state;
    }
    
    /// Update health status
    pub fn update_health(&mut self, status: HealthStatus) {
        self.health_status = status;
    }
    
    /// Update heartbeat timestamp
    pub fn update_heartbeat(&mut self) {
        self.last_heartbeat = Timestamp::now();
    }
    
    /// Add capability to the agent
    pub fn add_capability(&mut self, capability: String) {
        if !self.capabilities.contains(&capability) {
            self.capabilities.push(capability);
        }
    }
    
    /// Check if agent has a specific capability
    pub fn has_capability(&self, capability: &str) -> bool {
        self.capabilities.contains(&capability.to_string())
    }
    
    /// Set configuration parameter
    pub fn set_config<K, V>(&mut self, key: K, value: V)
    where
        K: Into<String>,
        V: Into<String>,
    {
        self.config.insert(key.into(), value.into());
    }
    
    /// Get configuration parameter
    pub fn get_config(&self, key: &str) -> Option<&String> {
        self.config.get(key)
    }
    
    /// Check if agent is healthy
    pub fn is_healthy(&self) -> bool {
        matches!(self.health_status, HealthStatus::Healthy)
    }
    
    /// Check if agent is available for tasks
    pub fn is_available(&self) -> bool {
        matches!(self.state, AgentState::Running) && self.is_healthy()
    }
    
    /// Get time since last heartbeat
    pub fn heartbeat_age(&self) -> Duration {
        Duration::from_millis(self.last_heartbeat.elapsed().as_millis() as u64)
    }
    
    /// Check if agent heartbeat is stale
    pub fn is_heartbeat_stale(&self, threshold: Duration) -> bool {
        self.heartbeat_age() > threshold
    }
}

/// Resource usage statistics for an agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsage {
    /// CPU usage percentage (0-100)
    pub cpu_usage: f64,
    /// Memory usage in bytes
    pub memory_usage: u64,
    /// Network bytes sent
    pub network_bytes_sent: u64,
    /// Network bytes received
    pub network_bytes_received: u64,
    /// Number of tasks processed
    pub tasks_processed: u64,
    /// Number of active tasks
    pub active_tasks: u64,
    /// Average task processing time in milliseconds
    pub avg_task_processing_time: f64,
    /// Last update timestamp
    pub last_updated: Timestamp,
}

impl Default for ResourceUsage {
    fn default() -> Self {
        Self {
            cpu_usage: 0.0,
            memory_usage: 0,
            network_bytes_sent: 0,
            network_bytes_received: 0,
            tasks_processed: 0,
            active_tasks: 0,
            avg_task_processing_time: 0.0,
            last_updated: Timestamp::now(),
        }
    }
}

/// Agent trait for implementing agent behavior
#[async_trait]
pub trait Agent: Send + Sync {
    /// Get agent information
    fn get_info(&self) -> &AgentInfo;
    
    /// Get mutable agent information
    fn get_info_mut(&mut self) -> &mut AgentInfo;
    
    /// Initialize the agent
    async fn initialize(&mut self) -> Result<()>;
    
    /// Start the agent
    async fn start(&mut self) -> Result<()>;
    
    /// Stop the agent
    async fn stop(&mut self) -> Result<()>;
    
    /// Process a message
    async fn process_message(&mut self, message: Message) -> Result<Option<Message>>;
    
    /// Perform health check
    async fn health_check(&mut self) -> Result<HealthStatus>;
    
    /// Handle agent-specific tasks
    async fn handle_task(&mut self, task: &str, payload: &[u8]) -> Result<Vec<u8>>;
    
    /// Get agent capabilities
    fn get_capabilities(&self) -> &[String];
    
    /// Update resource usage
    fn update_resource_usage(&mut self, usage: ResourceUsage);
}

/// Agent registry for managing all agents in the swarm
#[derive(Debug)]
pub struct AgentRegistry {
    /// Registered agents
    agents: Arc<DashMap<AgentId, Arc<RwLock<AgentInfo>>>>,
    /// Agent type index
    type_index: Arc<DashMap<AgentType, Vec<AgentId>>>,
    /// Communication layer
    communication: Arc<dyn CommunicationLayer>,
    /// Heartbeat interval
    heartbeat_interval: Duration,
    /// Heartbeat timeout threshold
    heartbeat_timeout: Duration,
}

impl AgentRegistry {
    /// Create a new agent registry
    pub fn new(communication: Arc<dyn CommunicationLayer>) -> Self {
        Self {
            agents: Arc::new(DashMap::new()),
            type_index: Arc::new(DashMap::new()),
            communication,
            heartbeat_interval: Duration::from_secs(30),
            heartbeat_timeout: Duration::from_secs(90),
        }
    }
    
    /// Start the agent registry
    pub async fn start(&self) -> Result<()> {
        // Start heartbeat monitoring
        self.start_heartbeat_monitor().await?;
        
        // Start health checking
        self.start_health_checker().await?;
        
        info!("Agent registry started successfully");
        Ok(())
    }
    
    /// Register a new agent
    pub async fn register_agent(&self, mut agent_info: AgentInfo) -> Result<()> {
        let agent_id = agent_info.id;
        let agent_type = agent_info.agent_type;
        
        // Check if agent already exists
        if self.agents.contains_key(&agent_id) {
            return Err(OrchestrationError::already_exists(format!("Agent {}", agent_id)));
        }
        
        // Update registration timestamp
        agent_info.registered_at = Timestamp::now();
        agent_info.last_heartbeat = Timestamp::now();
        
        // Register with communication layer
        self.communication.register_agent(agent_id).await?;
        
        // Add to registry
        self.agents.insert(agent_id, Arc::new(RwLock::new(agent_info)));
        
        // Update type index
        self.type_index.entry(agent_type).or_insert_with(Vec::new).push(agent_id);
        
        info!("Agent {} ({}) registered successfully", agent_id, agent_type);
        Ok(())
    }
    
    /// Unregister an agent
    pub async fn unregister_agent(&self, agent_id: AgentId) -> Result<()> {
        // Remove from registry
        let agent_info = self.agents.remove(&agent_id)
            .ok_or_else(|| OrchestrationError::not_found(format!("Agent {}", agent_id)))?;
        
        let agent_type = agent_info.1.read().await.agent_type;
        
        // Update type index
        if let Some(mut agents) = self.type_index.get_mut(&agent_type) {
            agents.retain(|&id| id != agent_id);
        }
        
        // Unregister from communication layer
        self.communication.unregister_agent(agent_id).await?;
        
        info!("Agent {} unregistered successfully", agent_id);
        Ok(())
    }
    
    /// Get agent information
    pub async fn get_agent(&self, agent_id: AgentId) -> Result<AgentInfo> {
        let agent = self.agents.get(&agent_id)
            .ok_or_else(|| OrchestrationError::not_found(format!("Agent {}", agent_id)))?;
        
        Ok(agent.read().await.clone())
    }
    
    /// Get all agents
    pub async fn get_all_agents(&self) -> Result<Vec<AgentInfo>> {
        let mut agents = Vec::new();
        
        for agent in self.agents.iter() {
            agents.push(agent.read().await.clone());
        }
        
        Ok(agents)
    }
    
    /// Get agents by type
    pub async fn get_agents_by_type(&self, agent_type: AgentType) -> Result<Vec<AgentInfo>> {
        let mut agents = Vec::new();
        
        if let Some(agent_ids) = self.type_index.get(&agent_type) {
            for &agent_id in agent_ids.iter() {
                if let Some(agent) = self.agents.get(&agent_id) {
                    agents.push(agent.read().await.clone());
                }
            }
        }
        
        Ok(agents)
    }
    
    /// Get healthy agents
    pub async fn get_healthy_agents(&self) -> Result<Vec<AgentInfo>> {
        let mut healthy_agents = Vec::new();
        
        for agent in self.agents.iter() {
            let agent_info = agent.read().await;
            if agent_info.is_healthy() {
                healthy_agents.push(agent_info.clone());
            }
        }
        
        Ok(healthy_agents)
    }
    
    /// Get available agents for task assignment
    pub async fn get_available_agents(&self) -> Result<Vec<AgentInfo>> {
        let mut available_agents = Vec::new();
        
        for agent in self.agents.iter() {
            let agent_info = agent.read().await;
            if agent_info.is_available() {
                available_agents.push(agent_info.clone());
            }
        }
        
        Ok(available_agents)
    }
    
    /// Update agent state
    pub async fn update_agent_state(&self, agent_id: AgentId, state: AgentState) -> Result<()> {
        let agent = self.agents.get(&agent_id)
            .ok_or_else(|| OrchestrationError::not_found(format!("Agent {}", agent_id)))?;
        
        agent.write().await.update_state(state);
        debug!("Agent {} state updated to {}", agent_id, state);
        Ok(())
    }
    
    /// Update agent health status
    pub async fn update_agent_health(&self, agent_id: AgentId, status: HealthStatus) -> Result<()> {
        let agent = self.agents.get(&agent_id)
            .ok_or_else(|| OrchestrationError::not_found(format!("Agent {}", agent_id)))?;
        
        agent.write().await.update_health(status);
        debug!("Agent {} health updated to {}", agent_id, status);
        Ok(())
    }
    
    /// Record agent heartbeat
    pub async fn record_heartbeat(&self, agent_id: AgentId) -> Result<()> {
        let agent = self.agents.get(&agent_id)
            .ok_or_else(|| OrchestrationError::not_found(format!("Agent {}", agent_id)))?;
        
        agent.write().await.update_heartbeat();
        debug!("Heartbeat recorded for agent {}", agent_id);
        Ok(())
    }
    
    /// Update agent resource usage
    pub async fn update_resource_usage(&self, agent_id: AgentId, usage: ResourceUsage) -> Result<()> {
        let agent = self.agents.get(&agent_id)
            .ok_or_else(|| OrchestrationError::not_found(format!("Agent {}", agent_id)))?;
        
        agent.write().await.resource_usage = usage;
        Ok(())
    }
    
    /// Start heartbeat monitoring
    async fn start_heartbeat_monitor(&self) -> Result<()> {
        let agents = Arc::clone(&self.agents);
        let heartbeat_timeout = self.heartbeat_timeout;
        
        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(30));
            
            loop {
                interval.tick().await;
                
                for agent_entry in agents.iter() {
                    let agent_id = *agent_entry.key();
                    let agent = agent_entry.value();
                    
                    let mut agent_info = agent.write().await;
                    if agent_info.is_heartbeat_stale(heartbeat_timeout) {
                        warn!("Agent {} heartbeat is stale, marking as unhealthy", agent_id);
                        agent_info.update_health(HealthStatus::Unhealthy);
                        agent_info.update_state(AgentState::Unavailable);
                    }
                }
            }
        });
        
        Ok(())
    }
    
    /// Start health checking
    async fn start_health_checker(&self) -> Result<()> {
        let agents = Arc::clone(&self.agents);
        
        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(60));
            
            loop {
                interval.tick().await;
                
                for agent_entry in agents.iter() {
                    let agent_id = *agent_entry.key();
                    let agent = agent_entry.value();
                    
                    let agent_info = agent.read().await;
                    if agent_info.health_status == HealthStatus::Unknown {
                        // Try to determine health status based on other factors
                        let mut agent_info = agent.write().await;
                        if agent_info.is_heartbeat_stale(Duration::from_secs(120)) {
                            agent_info.update_health(HealthStatus::Unhealthy);
                        } else if agent_info.state == AgentState::Running {
                            agent_info.update_health(HealthStatus::Healthy);
                        }
                    }
                }
            }
        });
        
        Ok(())
    }
    
    /// Get registry statistics
    pub async fn get_statistics(&self) -> Result<RegistryStatistics> {
        let mut stats = RegistryStatistics::default();
        
        for agent_entry in self.agents.iter() {
            let agent_info = agent_entry.read().await;
            
            stats.total_agents += 1;
            
            match agent_info.state {
                AgentState::Running => stats.running_agents += 1,
                AgentState::Busy => stats.busy_agents += 1,
                AgentState::Unavailable => stats.unavailable_agents += 1,
                AgentState::Failed => stats.failed_agents += 1,
                _ => {}
            }
            
            match agent_info.health_status {
                HealthStatus::Healthy => stats.healthy_agents += 1,
                HealthStatus::Degraded => stats.degraded_agents += 1,
                HealthStatus::Unhealthy => stats.unhealthy_agents += 1,
                HealthStatus::Unknown => stats.unknown_health_agents += 1,
            }
            
            *stats.agents_by_type.entry(agent_info.agent_type).or_insert(0) += 1;
        }
        
        Ok(stats)
    }
}

/// Registry statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RegistryStatistics {
    /// Total number of agents
    pub total_agents: usize,
    /// Number of running agents
    pub running_agents: usize,
    /// Number of busy agents
    pub busy_agents: usize,
    /// Number of unavailable agents
    pub unavailable_agents: usize,
    /// Number of failed agents
    pub failed_agents: usize,
    /// Number of healthy agents
    pub healthy_agents: usize,
    /// Number of degraded agents
    pub degraded_agents: usize,
    /// Number of unhealthy agents
    pub unhealthy_agents: usize,
    /// Number of agents with unknown health
    pub unknown_health_agents: usize,
    /// Agents by type
    pub agents_by_type: HashMap<AgentType, usize>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::communication::MessageRouter;
    use std::sync::Arc;
    
    #[tokio::test]
    async fn test_agent_info_creation() {
        let agent_id = AgentId::new();
        let agent_info = AgentInfo::new(
            agent_id,
            AgentType::Risk,
            "Risk Agent".to_string(),
            "1.0.0".to_string(),
        );
        
        assert_eq!(agent_info.id, agent_id);
        assert_eq!(agent_info.agent_type, AgentType::Risk);
        assert_eq!(agent_info.name, "Risk Agent");
        assert_eq!(agent_info.version, "1.0.0");
        assert_eq!(agent_info.state, AgentState::Starting);
    }
    
    #[tokio::test]
    async fn test_agent_capabilities() {
        let agent_id = AgentId::new();
        let mut agent_info = AgentInfo::new(
            agent_id,
            AgentType::Neural,
            "Neural Agent".to_string(),
            "1.0.0".to_string(),
        );
        
        agent_info.add_capability("forecasting".to_string());
        agent_info.add_capability("optimization".to_string());
        
        assert!(agent_info.has_capability("forecasting"));
        assert!(agent_info.has_capability("optimization"));
        assert!(!agent_info.has_capability("risk_analysis"));
    }
    
    #[tokio::test]
    async fn test_agent_registry() {
        let communication = Arc::new(MessageRouter::new());
        let registry = AgentRegistry::new(communication);
        
        let agent_id = AgentId::new();
        let agent_info = AgentInfo::new(
            agent_id,
            AgentType::Risk,
            "Risk Agent".to_string(),
            "1.0.0".to_string(),
        );
        
        // Register agent
        registry.register_agent(agent_info).await.unwrap();
        
        // Get agent
        let retrieved_agent = registry.get_agent(agent_id).await.unwrap();
        assert_eq!(retrieved_agent.id, agent_id);
        assert_eq!(retrieved_agent.agent_type, AgentType::Risk);
        
        // Update agent state
        registry.update_agent_state(agent_id, AgentState::Running).await.unwrap();
        let updated_agent = registry.get_agent(agent_id).await.unwrap();
        assert_eq!(updated_agent.state, AgentState::Running);
        
        // Unregister agent
        registry.unregister_agent(agent_id).await.unwrap();
        
        // Should not find agent after unregistering
        let result = registry.get_agent(agent_id).await;
        assert!(result.is_err());
    }
    
    #[tokio::test]
    async fn test_agent_heartbeat() {
        let agent_id = AgentId::new();
        let mut agent_info = AgentInfo::new(
            agent_id,
            AgentType::Risk,
            "Risk Agent".to_string(),
            "1.0.0".to_string(),
        );
        
        let initial_heartbeat = agent_info.last_heartbeat;
        
        // Wait a bit and update heartbeat
        sleep(Duration::from_millis(10)).await;
        agent_info.update_heartbeat();
        
        assert!(agent_info.last_heartbeat.0 > initial_heartbeat.0);
        assert!(!agent_info.is_heartbeat_stale(Duration::from_secs(1)));
    }
    
    #[tokio::test]
    async fn test_registry_statistics() {
        let communication = Arc::new(MessageRouter::new());
        let registry = AgentRegistry::new(communication);
        
        // Register multiple agents
        for i in 0..5 {
            let agent_id = AgentId::new();
            let agent_info = AgentInfo::new(
                agent_id,
                AgentType::Risk,
                format!("Risk Agent {}", i),
                "1.0.0".to_string(),
            );
            registry.register_agent(agent_info).await.unwrap();
        }
        
        let stats = registry.get_statistics().await.unwrap();
        assert_eq!(stats.total_agents, 5);
        assert_eq!(stats.agents_by_type.get(&AgentType::Risk), Some(&5));
    }
}