//! Service discovery and registry

use crate::agent::{AgentId, AgentInfo, AgentCapability};
use dashmap::DashMap;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::RwLock;
use anyhow::Result;

/// Service registry for agent discovery
pub struct ServiceRegistry {
    agents: DashMap<AgentId, AgentInfo>,
    capabilities_index: Arc<RwLock<HashMap<AgentCapability, HashSet<AgentId>>>>,
    topics_index: Arc<RwLock<HashMap<String, HashSet<AgentId>>>>,
}

impl ServiceRegistry {
    /// Create new service registry
    pub fn new() -> Self {
        Self {
            agents: DashMap::new(),
            capabilities_index: Arc::new(RwLock::new(HashMap::new())),
            topics_index: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Register an agent
    pub async fn register(&self, info: AgentInfo) -> Result<()> {
        let agent_id = info.id;
        
        // Update capabilities index
        {
            let mut cap_index = self.capabilities_index.write().await;
            for capability in &info.capabilities {
                cap_index
                    .entry(capability.clone())
                    .or_insert_with(HashSet::new)
                    .insert(agent_id);
            }
        }
        
        // Store agent info
        self.agents.insert(agent_id, info);
        
        tracing::info!("Agent {} registered", agent_id);
        Ok(())
    }

    /// Unregister an agent
    pub async fn unregister(&self, agent_id: AgentId) -> Result<()> {
        if let Some((_, info)) = self.agents.remove(&agent_id) {
            // Remove from capabilities index
            let mut cap_index = self.capabilities_index.write().await;
            for capability in &info.capabilities {
                if let Some(agents) = cap_index.get_mut(capability) {
                    agents.remove(&agent_id);
                    if agents.is_empty() {
                        cap_index.remove(capability);
                    }
                }
            }
            
            // Remove from topics index
            let mut topics_index = self.topics_index.write().await;
            topics_index.retain(|_, agents| {
                agents.remove(&agent_id);
                !agents.is_empty()
            });
            
            tracing::info!("Agent {} unregistered", agent_id);
        }
        
        Ok(())
    }

    /// Update agent info
    pub async fn update(&self, info: AgentInfo) -> Result<()> {
        let agent_id = info.id;
        
        // Remove old capabilities
        if let Some(old_info) = self.agents.get(&agent_id) {
            let mut cap_index = self.capabilities_index.write().await;
            for capability in &old_info.capabilities {
                if !info.capabilities.contains(capability) {
                    if let Some(agents) = cap_index.get_mut(capability) {
                        agents.remove(&agent_id);
                    }
                }
            }
        }
        
        // Add new capabilities
        {
            let mut cap_index = self.capabilities_index.write().await;
            for capability in &info.capabilities {
                cap_index
                    .entry(capability.clone())
                    .or_insert_with(HashSet::new)
                    .insert(agent_id);
            }
        }
        
        // Update agent info
        self.agents.insert(agent_id, info);
        
        Ok(())
    }

    /// Find agents by capability
    pub async fn find_by_capability(&self, capability: &AgentCapability) -> Vec<AgentInfo> {
        let cap_index = self.capabilities_index.read().await;
        
        if let Some(agent_ids) = cap_index.get(capability) {
            agent_ids
                .iter()
                .filter_map(|id| self.agents.get(id).map(|kv| kv.value().clone()))
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Find agents by multiple capabilities (AND operation)
    pub async fn find_by_capabilities(&self, capabilities: &[AgentCapability]) -> Vec<AgentInfo> {
        if capabilities.is_empty() {
            return self.list_all().await;
        }
        
        let cap_index = self.capabilities_index.read().await;
        
        // Get agents that have all capabilities
        let mut agent_sets: Vec<HashSet<AgentId>> = Vec::new();
        
        for capability in capabilities {
            if let Some(agents) = cap_index.get(capability) {
                agent_sets.push(agents.clone());
            } else {
                // No agents have this capability
                return Vec::new();
            }
        }
        
        // Find intersection of all sets
        let mut result = agent_sets[0].clone();
        for set in agent_sets.iter().skip(1) {
            result.retain(|id| set.contains(id));
        }
        
        // Get agent infos
        result
            .iter()
            .filter_map(|id| self.agents.get(id).map(|kv| kv.value().clone()))
            .collect()
    }

    /// Get agent by ID
    pub async fn get(&self, agent_id: AgentId) -> Option<AgentInfo> {
        self.agents.get(&agent_id).map(|kv| kv.value().clone())
    }

    /// List all agents
    pub async fn list_all(&self) -> Vec<AgentInfo> {
        self.agents
            .iter()
            .map(|kv| kv.value().clone())
            .collect()
    }

    /// List healthy agents
    pub async fn list_healthy(&self, timeout_secs: i64) -> Vec<AgentInfo> {
        self.agents
            .iter()
            .map(|kv| kv.value().clone())
            .filter(|info| info.is_healthy(timeout_secs))
            .collect()
    }

    /// Subscribe agent to topic
    pub async fn subscribe_to_topic(&self, agent_id: AgentId, topic: String) -> Result<()> {
        let mut topics_index = self.topics_index.write().await;
        topics_index
            .entry(topic)
            .or_insert_with(HashSet::new)
            .insert(agent_id);
        Ok(())
    }

    /// Unsubscribe agent from topic
    pub async fn unsubscribe_from_topic(&self, agent_id: AgentId, topic: &str) -> Result<()> {
        let mut topics_index = self.topics_index.write().await;
        if let Some(agents) = topics_index.get_mut(topic) {
            agents.remove(&agent_id);
            if agents.is_empty() {
                topics_index.remove(topic);
            }
        }
        Ok(())
    }

    /// Get subscribers for a topic
    pub async fn get_topic_subscribers(&self, topic: &str) -> Vec<AgentId> {
        let topics_index = self.topics_index.read().await;
        topics_index
            .get(topic)
            .map(|agents| agents.iter().copied().collect())
            .unwrap_or_default()
    }

    /// Clean up stale agents
    pub async fn cleanup_stale(&self, timeout_secs: i64) -> Result<Vec<AgentId>> {
        let now = chrono::Utc::now();
        let mut stale_agents = Vec::new();
        
        // Find stale agents
        for kv in self.agents.iter() {
            let info = kv.value();
            let elapsed = now - info.last_heartbeat;
            
            if elapsed.num_seconds() > timeout_secs {
                stale_agents.push(info.id);
            }
        }
        
        // Remove stale agents
        for agent_id in &stale_agents {
            self.unregister(*agent_id).await?;
        }
        
        if !stale_agents.is_empty() {
            tracing::info!("Cleaned up {} stale agents", stale_agents.len());
        }
        
        Ok(stale_agents)
    }
}

impl Default for ServiceRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Discovery service for finding agents
#[async_trait::async_trait]
pub trait DiscoveryService: Send + Sync {
    /// Find agents with specific capability
    async fn find_agents(&self, capability: &AgentCapability) -> Result<Vec<AgentInfo>>;
    
    /// Get specific agent
    async fn get_agent(&self, agent_id: AgentId) -> Result<Option<AgentInfo>>;
    
    /// Register this agent
    async fn register(&self, info: AgentInfo) -> Result<()>;
    
    /// Unregister this agent
    async fn unregister(&self, agent_id: AgentId) -> Result<()>;
    
    /// Update heartbeat
    async fn heartbeat(&self, agent_id: AgentId) -> Result<()>;
}

/// Local discovery service implementation
pub struct LocalDiscoveryService {
    registry: Arc<ServiceRegistry>,
}

impl LocalDiscoveryService {
    pub fn new(registry: Arc<ServiceRegistry>) -> Self {
        Self { registry }
    }
}

#[async_trait::async_trait]
impl DiscoveryService for LocalDiscoveryService {
    async fn find_agents(&self, capability: &AgentCapability) -> Result<Vec<AgentInfo>> {
        Ok(self.registry.find_by_capability(capability).await)
    }
    
    async fn get_agent(&self, agent_id: AgentId) -> Result<Option<AgentInfo>> {
        Ok(self.registry.get(agent_id).await)
    }
    
    async fn register(&self, info: AgentInfo) -> Result<()> {
        self.registry.register(info).await
    }
    
    async fn unregister(&self, agent_id: AgentId) -> Result<()> {
        self.registry.unregister(agent_id).await
    }
    
    async fn heartbeat(&self, agent_id: AgentId) -> Result<()> {
        if let Some(mut entry) = self.registry.agents.get_mut(&agent_id) {
            entry.update_heartbeat();
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_service_registry() {
        let registry = ServiceRegistry::new();
        
        // Create test agent
        let agent_id = AgentId::new();
        let info = AgentInfo::new(agent_id, "test-agent", "tcp://localhost:5555")
            .with_capability(AgentCapability::Trading)
            .with_capability(AgentCapability::MarketAnalysis);
        
        // Register agent
        registry.register(info.clone()).await.unwrap();
        
        // Find by capability
        let traders = registry.find_by_capability(&AgentCapability::Trading).await;
        assert_eq!(traders.len(), 1);
        assert_eq!(traders[0].id, agent_id);
        
        // Find by multiple capabilities
        let agents = registry.find_by_capabilities(&[
            AgentCapability::Trading,
            AgentCapability::MarketAnalysis,
        ]).await;
        assert_eq!(agents.len(), 1);
        
        // Unregister
        registry.unregister(agent_id).await.unwrap();
        let traders = registry.find_by_capability(&AgentCapability::Trading).await;
        assert_eq!(traders.len(), 0);
    }

    #[tokio::test]
    async fn test_topic_subscription() {
        let registry = ServiceRegistry::new();
        
        let agent1 = AgentId::new();
        let agent2 = AgentId::new();
        
        // Subscribe to topic
        registry.subscribe_to_topic(agent1, "market-data".to_string()).await.unwrap();
        registry.subscribe_to_topic(agent2, "market-data".to_string()).await.unwrap();
        
        // Get subscribers
        let subscribers = registry.get_topic_subscribers("market-data").await;
        assert_eq!(subscribers.len(), 2);
        assert!(subscribers.contains(&agent1));
        assert!(subscribers.contains(&agent2));
        
        // Unsubscribe
        registry.unsubscribe_from_topic(agent1, "market-data").await.unwrap();
        let subscribers = registry.get_topic_subscribers("market-data").await;
        assert_eq!(subscribers.len(), 1);
        assert!(!subscribers.contains(&agent1));
    }
}