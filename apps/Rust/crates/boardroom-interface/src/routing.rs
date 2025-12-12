//! Message routing and load balancing

use crate::agent::{AgentId, AgentInfo, AgentCapability, AgentState};
use crate::message::{Message, MessageType};
use dashmap::DashMap;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use anyhow::Result;

/// Routing strategy for message distribution
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RoutingStrategy {
    /// Send to specific agent
    Direct,
    /// Round-robin distribution
    RoundRobin,
    /// Send to least loaded agent
    LeastLoaded,
    /// Random selection
    Random,
    /// Based on agent capabilities
    CapabilityBased,
    /// Custom routing logic
    Custom,
}

/// Load metrics for an agent
#[derive(Debug, Clone)]
pub struct LoadMetrics {
    pub agent_id: AgentId,
    pub active_tasks: usize,
    pub message_queue_size: usize,
    pub cpu_usage: f32,
    pub memory_usage: f32,
    pub response_time_ms: f64,
    pub last_updated: chrono::DateTime<chrono::Utc>,
}

impl LoadMetrics {
    /// Calculate overall load score (0.0 = no load, 1.0 = max load)
    pub fn load_score(&self) -> f64 {
        let task_score = (self.active_tasks as f64 / 100.0).min(1.0);
        let queue_score = (self.message_queue_size as f64 / 1000.0).min(1.0);
        let cpu_score = self.cpu_usage as f64;
        let memory_score = self.memory_usage as f64;
        let response_score = (self.response_time_ms / 1000.0).min(1.0);
        
        // Weighted average
        (task_score * 0.3 + 
         queue_score * 0.2 + 
         cpu_score * 0.2 + 
         memory_score * 0.2 + 
         response_score * 0.1)
    }
}

/// Load balancer for distributing messages
pub struct LoadBalancer {
    agents: Arc<DashMap<AgentId, AgentInfo>>,
    load_metrics: Arc<DashMap<AgentId, LoadMetrics>>,
    round_robin_indices: Arc<RwLock<HashMap<AgentCapability, usize>>>,
}

impl LoadBalancer {
    /// Create new load balancer
    pub fn new() -> Self {
        Self {
            agents: Arc::new(DashMap::new()),
            load_metrics: Arc::new(DashMap::new()),
            round_robin_indices: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Register an agent
    pub fn register_agent(&self, info: AgentInfo) {
        let agent_id = info.id;
        self.agents.insert(agent_id, info);
        
        // Initialize load metrics
        let metrics = LoadMetrics {
            agent_id,
            active_tasks: 0,
            message_queue_size: 0,
            cpu_usage: 0.0,
            memory_usage: 0.0,
            response_time_ms: 0.0,
            last_updated: chrono::Utc::now(),
        };
        self.load_metrics.insert(agent_id, metrics);
    }

    /// Unregister an agent
    pub fn unregister_agent(&self, agent_id: AgentId) {
        self.agents.remove(&agent_id);
        self.load_metrics.remove(&agent_id);
    }

    /// Update load metrics for an agent
    pub fn update_metrics(&self, metrics: LoadMetrics) {
        self.load_metrics.insert(metrics.agent_id, metrics);
    }

    /// Select agent based on routing strategy
    pub async fn select_agent(
        &self,
        strategy: RoutingStrategy,
        capability: Option<&AgentCapability>,
        exclude: Option<&[AgentId]>,
    ) -> Result<Option<AgentId>> {
        let agents = self.get_available_agents(capability, exclude);
        
        if agents.is_empty() {
            return Ok(None);
        }
        
        match strategy {
            RoutingStrategy::Direct => Ok(None), // Direct routing doesn't use selection
            RoutingStrategy::RoundRobin => self.round_robin_select(&agents, capability).await,
            RoutingStrategy::LeastLoaded => Ok(self.least_loaded_select(&agents)),
            RoutingStrategy::Random => Ok(self.random_select(&agents)),
            RoutingStrategy::CapabilityBased => Ok(agents.first().map(|a| a.id)),
            RoutingStrategy::Custom => Ok(None), // Custom logic would be implemented separately
        }
    }

    /// Get available agents matching criteria
    fn get_available_agents(
        &self,
        capability: Option<&AgentCapability>,
        exclude: Option<&[AgentId]>,
    ) -> Vec<AgentInfo> {
        self.agents
            .iter()
            .filter_map(|entry| {
                let info = entry.value();
                
                // Check if agent should be excluded
                if let Some(exclude_list) = exclude {
                    if exclude_list.contains(&info.id) {
                        return None;
                    }
                }
                
                // Check if agent is ready
                if info.state != AgentState::Ready {
                    return None;
                }
                
                // Check capability if specified
                if let Some(cap) = capability {
                    if !info.has_capability(cap) {
                        return None;
                    }
                }
                
                Some(info.clone())
            })
            .collect()
    }

    /// Round-robin selection
    async fn round_robin_select(
        &self,
        agents: &[AgentInfo],
        capability: Option<&AgentCapability>,
    ) -> Result<Option<AgentId>> {
        if agents.is_empty() {
            return Ok(None);
        }
        
        let mut indices = self.round_robin_indices.write().await;
        
        let key = capability.cloned().unwrap_or(AgentCapability::Custom("default".to_string()));
        let index = indices.entry(key).or_insert(0);
        
        let selected = agents[*index % agents.len()].id;
        *index = (*index + 1) % agents.len();
        
        Ok(Some(selected))
    }

    /// Least loaded selection
    fn least_loaded_select(&self, agents: &[AgentInfo]) -> Option<AgentId> {
        agents
            .iter()
            .filter_map(|agent| {
                self.load_metrics
                    .get(&agent.id)
                    .map(|metrics| (agent.id, metrics.load_score()))
            })
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(id, _)| id)
    }

    /// Random selection
    fn random_select(&self, agents: &[AgentInfo]) -> Option<AgentId> {
        use rand::Rng;
        
        if agents.is_empty() {
            return None;
        }
        
        let mut rng = rand::thread_rng();
        let index = rng.gen_range(0..agents.len());
        Some(agents[index].id)
    }

    /// Get current load for an agent
    pub fn get_load(&self, agent_id: AgentId) -> Option<f64> {
        self.load_metrics
            .get(&agent_id)
            .map(|metrics| metrics.load_score())
    }

    /// Get all agents with their load scores
    pub fn get_load_distribution(&self) -> Vec<(AgentId, f64)> {
        self.load_metrics
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().load_score()))
            .collect()
    }
}

/// Message router implementation
pub struct MessageRouterImpl {
    load_balancer: Arc<LoadBalancer>,
    routing_rules: Arc<RwLock<HashMap<String, RoutingStrategy>>>,
    default_strategy: RoutingStrategy,
}

impl MessageRouterImpl {
    /// Create new message router
    pub fn new(load_balancer: Arc<LoadBalancer>, default_strategy: RoutingStrategy) -> Self {
        Self {
            load_balancer,
            routing_rules: Arc::new(RwLock::new(HashMap::new())),
            default_strategy,
        }
    }

    /// Add routing rule for specific message types
    pub async fn add_rule(&self, pattern: String, strategy: RoutingStrategy) {
        let mut rules = self.routing_rules.write().await;
        rules.insert(pattern, strategy);
    }

    /// Route a message based on its type and configured rules
    pub async fn route_message(&self, message: &Message) -> Result<Option<AgentId>> {
        // If message has specific recipient, use direct routing
        if let Some(to) = message.to {
            return Ok(Some(to));
        }
        
        // Determine routing strategy based on message type
        let strategy = self.get_routing_strategy(message).await;
        
        // Extract required capability from message if applicable
        let capability = self.extract_capability(message);
        
        // Select agent based on strategy
        self.load_balancer
            .select_agent(strategy, capability.as_ref(), None)
            .await
    }

    /// Get routing strategy for a message
    async fn get_routing_strategy(&self, message: &Message) -> RoutingStrategy {
        let rules = self.routing_rules.read().await;
        
        // Check message type patterns
        match &message.message_type {
            MessageType::Request { method, .. } => {
                if let Some(strategy) = rules.get(method) {
                    return *strategy;
                }
            }
            MessageType::Broadcast { topic, .. } => {
                if let Some(strategy) = rules.get(topic) {
                    return *strategy;
                }
            }
            _ => {}
        }
        
        self.default_strategy
    }

    /// Extract required capability from message
    fn extract_capability(&self, message: &Message) -> Option<AgentCapability> {
        // This could be extended to parse message content
        // and determine required capabilities
        match &message.message_type {
            MessageType::Request { method, .. } => {
                match method.as_str() {
                    "execute_trade" => Some(AgentCapability::Trading),
                    "analyze_market" => Some(AgentCapability::MarketAnalysis),
                    "detect_whale" => Some(AgentCapability::WhaleDetection),
                    "assess_risk" => Some(AgentCapability::RiskAssessment),
                    _ => None,
                }
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_metrics() {
        let metrics = LoadMetrics {
            agent_id: AgentId::new(),
            active_tasks: 50,
            message_queue_size: 500,
            cpu_usage: 0.6,
            memory_usage: 0.4,
            response_time_ms: 200.0,
            last_updated: chrono::Utc::now(),
        };
        
        let score = metrics.load_score();
        assert!(score > 0.0 && score < 1.0);
    }

    #[tokio::test]
    async fn test_round_robin_selection() {
        let lb = LoadBalancer::new();
        
        // Register agents
        let agents: Vec<_> = (0..3)
            .map(|i| {
                let info = AgentInfo::new(
                    AgentId::new(),
                    format!("agent-{}", i),
                    "tcp://localhost:5555",
                );
                lb.register_agent(info.clone());
                info
            })
            .collect();
        
        // Update agent states to ready
        for agent in &agents {
            if let Some(mut entry) = lb.agents.get_mut(&agent.id) {
                entry.state = AgentState::Ready;
            }
        }
        
        // Test round-robin selection
        let mut selected = vec![];
        for _ in 0..6 {
            let id = lb.select_agent(RoutingStrategy::RoundRobin, None, None)
                .await
                .unwrap()
                .unwrap();
            selected.push(id);
        }
        
        // Should cycle through all agents twice
        assert_eq!(selected.len(), 6);
    }

    #[test]
    fn test_least_loaded_selection() {
        let lb = LoadBalancer::new();
        
        // Register agents with different loads
        let agent1 = AgentId::new();
        let agent2 = AgentId::new();
        
        let info1 = AgentInfo::new(agent1, "agent-1", "tcp://localhost:5555");
        let info2 = AgentInfo::new(agent2, "agent-2", "tcp://localhost:5556");
        
        lb.register_agent(info1);
        lb.register_agent(info2);
        
        // Set different load metrics
        lb.update_metrics(LoadMetrics {
            agent_id: agent1,
            active_tasks: 10,
            message_queue_size: 100,
            cpu_usage: 0.3,
            memory_usage: 0.2,
            response_time_ms: 50.0,
            last_updated: chrono::Utc::now(),
        });
        
        lb.update_metrics(LoadMetrics {
            agent_id: agent2,
            active_tasks: 50,
            message_queue_size: 500,
            cpu_usage: 0.8,
            memory_usage: 0.7,
            response_time_ms: 200.0,
            last_updated: chrono::Utc::now(),
        });
        
        // Update states to ready
        if let Some(mut entry) = lb.agents.get_mut(&agent1) {
            entry.state = AgentState::Ready;
        }
        if let Some(mut entry) = lb.agents.get_mut(&agent2) {
            entry.state = AgentState::Ready;
        }
        
        // Should select the least loaded agent
        let agents = lb.get_available_agents(None, None);
        let selected = lb.least_loaded_select(&agents).unwrap();
        assert_eq!(selected, agent1);
    }
}