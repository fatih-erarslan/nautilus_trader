//! Load balancing system for distributing tasks across agents in the swarm.

use crate::agent::{AgentInfo, AgentRegistry};
use crate::error::{OrchestrationError, Result};
use crate::types::{AgentId, AgentType, LoadBalancingStrategy, Timestamp};
use async_trait::async_trait;
use dashmap::DashMap;
use parking_lot::RwLock;
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::time::interval;
use tracing::{debug, info, warn};

/// Agent load information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentLoad {
    /// Agent ID
    pub agent_id: AgentId,
    /// Current number of active tasks
    pub active_tasks: u64,
    /// Total tasks processed
    pub total_tasks: u64,
    /// Average task processing time in milliseconds
    pub avg_processing_time: f64,
    /// Current CPU usage
    pub cpu_usage: f64,
    /// Current memory usage
    pub memory_usage: u64,
    /// Agent response time in milliseconds
    pub response_time: f64,
    /// Agent availability score (0.0 to 1.0)
    pub availability_score: f64,
    /// Last update timestamp
    pub last_updated: Timestamp,
}

impl AgentLoad {
    /// Create new agent load information
    pub fn new(agent_id: AgentId) -> Self {
        Self {
            agent_id,
            active_tasks: 0,
            total_tasks: 0,
            avg_processing_time: 0.0,
            cpu_usage: 0.0,
            memory_usage: 0,
            response_time: 0.0,
            availability_score: 1.0,
            last_updated: Timestamp::now(),
        }
    }
    
    /// Update agent load metrics
    pub fn update_metrics(
        &mut self,
        active_tasks: u64,
        cpu_usage: f64,
        memory_usage: u64,
        response_time: f64,
    ) {
        self.active_tasks = active_tasks;
        self.cpu_usage = cpu_usage;
        self.memory_usage = memory_usage;
        self.response_time = response_time;
        self.last_updated = Timestamp::now();
        
        // Calculate availability score based on multiple factors
        let cpu_factor = 1.0 - (cpu_usage / 100.0).min(1.0);
        let memory_factor = 1.0 - (memory_usage as f64 / (1024.0 * 1024.0 * 1024.0)).min(1.0); // Assume 1GB max
        let response_factor = 1.0 - (response_time / 1000.0).min(1.0); // Assume 1s max
        let task_factor = 1.0 - (active_tasks as f64 / 10.0).min(1.0); // Assume 10 tasks max
        
        self.availability_score = (cpu_factor + memory_factor + response_factor + task_factor) / 4.0;
    }
    
    /// Get load score for comparison (higher is more loaded)
    pub fn load_score(&self) -> f64 {
        1.0 - self.availability_score
    }
}

/// Load balancer trait
#[async_trait]
pub trait LoadBalancer: Send + Sync {
    /// Select the best agent for a task
    async fn select_agent(&self, agent_type: Option<AgentType>) -> Result<Option<AgentId>>;
    
    /// Update agent load information
    async fn update_agent_load(&self, agent_id: AgentId, load: AgentLoad) -> Result<()>;
    
    /// Get current load statistics
    async fn get_load_stats(&self) -> Result<LoadBalancingStats>;
    
    /// Get agent load information
    async fn get_agent_load(&self, agent_id: AgentId) -> Result<AgentLoad>;
    
    /// Get all agent loads
    async fn get_all_agent_loads(&self) -> Result<Vec<AgentLoad>>;
    
    /// Set load balancing strategy
    async fn set_strategy(&self, strategy: LoadBalancingStrategy) -> Result<()>;
}

/// Load balancing statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadBalancingStats {
    /// Total number of agents
    pub total_agents: u64,
    /// Number of available agents
    pub available_agents: u64,
    /// Number of overloaded agents
    pub overloaded_agents: u64,
    /// Average load across all agents
    pub average_load: f64,
    /// Maximum load
    pub max_load: f64,
    /// Minimum load
    pub min_load: f64,
    /// Load distribution by agent type
    pub load_by_type: HashMap<AgentType, f64>,
    /// Total tasks distributed
    pub total_tasks_distributed: u64,
    /// Tasks distributed by strategy
    pub tasks_by_strategy: HashMap<LoadBalancingStrategy, u64>,
}

/// Adaptive load balancer implementation
#[derive(Debug)]
pub struct AdaptiveLoadBalancer {
    /// Agent registry
    agent_registry: Arc<AgentRegistry>,
    /// Agent load tracking
    agent_loads: Arc<DashMap<AgentId, AgentLoad>>,
    /// Current load balancing strategy
    strategy: Arc<RwLock<LoadBalancingStrategy>>,
    /// Round-robin counters
    round_robin_counters: Arc<DashMap<AgentType, AtomicU64>>,
    /// Consistent hashing ring
    hash_ring: Arc<RwLock<ConsistentHashRing>>,
    /// Load balancing statistics
    stats: Arc<RwLock<LoadBalancingStats>>,
    /// Performance history for adaptive selection
    performance_history: Arc<DashMap<AgentId, VecDeque<f64>>>,
}

impl AdaptiveLoadBalancer {
    /// Create a new adaptive load balancer
    pub fn new(agent_registry: Arc<AgentRegistry>) -> Self {
        Self {
            agent_registry,
            agent_loads: Arc::new(DashMap::new()),
            strategy: Arc::new(RwLock::new(LoadBalancingStrategy::PerformanceBased)),
            round_robin_counters: Arc::new(DashMap::new()),
            hash_ring: Arc::new(RwLock::new(ConsistentHashRing::new())),
            stats: Arc::new(RwLock::new(LoadBalancingStats {
                total_agents: 0,
                available_agents: 0,
                overloaded_agents: 0,
                average_load: 0.0,
                max_load: 0.0,
                min_load: 0.0,
                load_by_type: HashMap::new(),
                total_tasks_distributed: 0,
                tasks_by_strategy: HashMap::new(),
            })),
            performance_history: Arc::new(DashMap::new()),
        }
    }
    
    /// Start the load balancer
    pub async fn start(&self) -> Result<()> {
        // Start load monitoring
        self.start_load_monitoring().await?;
        
        // Start strategy adaptation
        self.start_strategy_adaptation().await?;
        
        info!("Adaptive load balancer started successfully");
        Ok(())
    }
    
    /// Start load monitoring
    async fn start_load_monitoring(&self) -> Result<()> {
        let agent_registry = Arc::clone(&self.agent_registry);
        let agent_loads = Arc::clone(&self.agent_loads);
        let stats = Arc::clone(&self.stats);
        
        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(10));
            
            loop {
                interval.tick().await;
                
                if let Ok(agents) = agent_registry.get_all_agents().await {
                    let mut total_load = 0.0;
                    let mut max_load = 0.0;
                    let mut min_load = f64::MAX;
                    let mut available_count = 0;
                    let mut overloaded_count = 0;
                    let mut load_by_type: HashMap<AgentType, f64> = HashMap::new();
                    
                    for agent_info in agents {
                        let agent_load = agent_loads.entry(agent_info.id)
                            .or_insert_with(|| AgentLoad::new(agent_info.id));
                        
                        // Update load from agent resource usage
                        agent_load.update_metrics(
                            agent_info.resource_usage.active_tasks,
                            agent_info.resource_usage.cpu_usage,
                            agent_info.resource_usage.memory_usage,
                            agent_info.resource_usage.avg_task_processing_time,
                        );
                        
                        let load_score = agent_load.load_score();
                        total_load += load_score;
                        max_load = max_load.max(load_score);
                        min_load = min_load.min(load_score);
                        
                        if agent_info.is_available() {
                            available_count += 1;
                        }
                        
                        if load_score > 0.8 {
                            overloaded_count += 1;
                        }
                        
                        *load_by_type.entry(agent_info.agent_type).or_insert(0.0) += load_score;
                    }
                    
                    // Update statistics
                    let mut stats = stats.write();
                    stats.total_agents = agents.len() as u64;
                    stats.available_agents = available_count;
                    stats.overloaded_agents = overloaded_count;
                    stats.average_load = if agents.len() > 0 { total_load / agents.len() as f64 } else { 0.0 };
                    stats.max_load = max_load;
                    stats.min_load = if min_load == f64::MAX { 0.0 } else { min_load };
                    stats.load_by_type = load_by_type;
                }
            }
        });
        
        Ok(())
    }
    
    /// Start strategy adaptation
    async fn start_strategy_adaptation(&self) -> Result<()> {
        let strategy = Arc::clone(&self.strategy);
        let stats = Arc::clone(&self.stats);
        let performance_history = Arc::clone(&self.performance_history);
        
        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(60));
            
            loop {
                interval.tick().await;
                
                // Analyze performance and adapt strategy
                let stats = stats.read();
                let current_strategy = *strategy.read();
                
                // Simple adaptation logic
                let new_strategy = if stats.average_load > 0.7 {
                    // High load: use performance-based balancing
                    LoadBalancingStrategy::PerformanceBased
                } else if stats.available_agents < 3 {
                    // Few agents: use least connections
                    LoadBalancingStrategy::LeastConnections
                } else {
                    // Normal operation: use round-robin
                    LoadBalancingStrategy::RoundRobin
                };
                
                if new_strategy != current_strategy {
                    *strategy.write() = new_strategy;
                    info!("Load balancing strategy adapted to {:?}", new_strategy);
                }
            }
        });
        
        Ok(())
    }
    
    /// Select agent using round-robin strategy
    async fn select_round_robin(&self, agent_type: Option<AgentType>) -> Result<Option<AgentId>> {
        let agents = if let Some(agent_type) = agent_type {
            self.agent_registry.get_agents_by_type(agent_type).await?
        } else {
            self.agent_registry.get_available_agents().await?
        };
        
        if agents.is_empty() {
            return Ok(None);
        }
        
        let counter = self.round_robin_counters.entry(agent_type.unwrap_or(AgentType::Risk))
            .or_insert_with(|| AtomicU64::new(0));
        
        let index = counter.fetch_add(1, Ordering::Relaxed) as usize % agents.len();
        Ok(Some(agents[index].id))
    }
    
    /// Select agent using least connections strategy
    async fn select_least_connections(&self, agent_type: Option<AgentType>) -> Result<Option<AgentId>> {
        let agents = if let Some(agent_type) = agent_type {
            self.agent_registry.get_agents_by_type(agent_type).await?
        } else {
            self.agent_registry.get_available_agents().await?
        };
        
        if agents.is_empty() {
            return Ok(None);
        }
        
        let mut best_agent = None;
        let mut min_connections = u64::MAX;
        
        for agent in agents {
            let load = self.agent_loads.get(&agent.id)
                .map(|load| load.active_tasks)
                .unwrap_or(0);
            
            if load < min_connections {
                min_connections = load;
                best_agent = Some(agent.id);
            }
        }
        
        Ok(best_agent)
    }
    
    /// Select agent using performance-based strategy
    async fn select_performance_based(&self, agent_type: Option<AgentType>) -> Result<Option<AgentId>> {
        let agents = if let Some(agent_type) = agent_type {
            self.agent_registry.get_agents_by_type(agent_type).await?
        } else {
            self.agent_registry.get_available_agents().await?
        };
        
        if agents.is_empty() {
            return Ok(None);
        }
        
        let mut best_agent = None;
        let mut best_score = f64::MIN;
        
        for agent in agents {
            let load = self.agent_loads.get(&agent.id)
                .map(|load| load.availability_score)
                .unwrap_or(1.0);
            
            if load > best_score {
                best_score = load;
                best_agent = Some(agent.id);
            }
        }
        
        Ok(best_agent)
    }
    
    /// Select agent using consistent hashing strategy
    async fn select_consistent_hash(&self, agent_type: Option<AgentType>) -> Result<Option<AgentId>> {
        let agents = if let Some(agent_type) = agent_type {
            self.agent_registry.get_agents_by_type(agent_type).await?
        } else {
            self.agent_registry.get_available_agents().await?
        };
        
        if agents.is_empty() {
            return Ok(None);
        }
        
        let hash_ring = self.hash_ring.read();
        let key = format!("{:?}_{}", agent_type, Timestamp::now().as_millis());
        Ok(hash_ring.get_node(&key))
    }
    
    /// Select agent using weighted round-robin strategy
    async fn select_weighted_round_robin(&self, agent_type: Option<AgentType>) -> Result<Option<AgentId>> {
        let agents = if let Some(agent_type) = agent_type {
            self.agent_registry.get_agents_by_type(agent_type).await?
        } else {
            self.agent_registry.get_available_agents().await?
        };
        
        if agents.is_empty() {
            return Ok(None);
        }
        
        // Calculate weights based on agent capabilities
        let mut weighted_agents = Vec::new();
        let mut total_weight = 0.0;
        
        for agent in agents {
            let load = self.agent_loads.get(&agent.id)
                .map(|load| load.availability_score)
                .unwrap_or(1.0);
            
            let weight = load * 100.0; // Scale to make it more significant
            total_weight += weight;
            weighted_agents.push((agent.id, weight));
        }
        
        // Select based on weight
        let mut rng = rand::thread_rng();
        let mut random_value = rng.gen::<f64>() * total_weight;
        
        for (agent_id, weight) in weighted_agents {
            random_value -= weight;
            if random_value <= 0.0 {
                return Ok(Some(agent_id));
            }
        }
        
        // Fallback to first agent
        Ok(weighted_agents.first().map(|(id, _)| *id))
    }
    
    /// Select agent using random strategy
    async fn select_random(&self, agent_type: Option<AgentType>) -> Result<Option<AgentId>> {
        let agents = if let Some(agent_type) = agent_type {
            self.agent_registry.get_agents_by_type(agent_type).await?
        } else {
            self.agent_registry.get_available_agents().await?
        };
        
        if agents.is_empty() {
            return Ok(None);
        }
        
        let mut rng = rand::thread_rng();
        let index = rng.gen_range(0..agents.len());
        Ok(Some(agents[index].id))
    }
}

#[async_trait]
impl LoadBalancer for AdaptiveLoadBalancer {
    async fn select_agent(&self, agent_type: Option<AgentType>) -> Result<Option<AgentId>> {
        let strategy = *self.strategy.read();
        
        let selected_agent = match strategy {
            LoadBalancingStrategy::RoundRobin => {
                self.select_round_robin(agent_type).await?
            }
            LoadBalancingStrategy::LeastConnections => {
                self.select_least_connections(agent_type).await?
            }
            LoadBalancingStrategy::WeightedRoundRobin => {
                self.select_weighted_round_robin(agent_type).await?
            }
            LoadBalancingStrategy::Random => {
                self.select_random(agent_type).await?
            }
            LoadBalancingStrategy::ConsistentHashing => {
                self.select_consistent_hash(agent_type).await?
            }
            LoadBalancingStrategy::PerformanceBased => {
                self.select_performance_based(agent_type).await?
            }
        };
        
        // Update statistics
        if selected_agent.is_some() {
            let mut stats = self.stats.write();
            stats.total_tasks_distributed += 1;
            *stats.tasks_by_strategy.entry(strategy).or_insert(0) += 1;
        }
        
        debug!("Selected agent {:?} using strategy {:?}", selected_agent, strategy);
        Ok(selected_agent)
    }
    
    async fn update_agent_load(&self, agent_id: AgentId, load: AgentLoad) -> Result<()> {
        self.agent_loads.insert(agent_id, load);
        
        // Update performance history
        let mut history = self.performance_history.entry(agent_id).or_insert_with(VecDeque::new);
        history.push_back(load.availability_score);
        
        // Keep only last 100 entries
        while history.len() > 100 {
            history.pop_front();
        }
        
        Ok(())
    }
    
    async fn get_load_stats(&self) -> Result<LoadBalancingStats> {
        Ok(self.stats.read().clone())
    }
    
    async fn get_agent_load(&self, agent_id: AgentId) -> Result<AgentLoad> {
        self.agent_loads.get(&agent_id)
            .map(|load| load.clone())
            .ok_or_else(|| OrchestrationError::not_found(format!("Agent load for {}", agent_id)))
    }
    
    async fn get_all_agent_loads(&self) -> Result<Vec<AgentLoad>> {
        Ok(self.agent_loads.iter().map(|entry| entry.value().clone()).collect())
    }
    
    async fn set_strategy(&self, strategy: LoadBalancingStrategy) -> Result<()> {
        *self.strategy.write() = strategy;
        info!("Load balancing strategy set to {:?}", strategy);
        Ok(())
    }
}

/// Consistent hashing ring for load balancing
#[derive(Debug)]
struct ConsistentHashRing {
    nodes: HashMap<u64, AgentId>,
    ring: Vec<u64>,
}

impl ConsistentHashRing {
    fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            ring: Vec::new(),
        }
    }
    
    fn add_node(&mut self, agent_id: AgentId) {
        // Add multiple virtual nodes for better distribution
        for i in 0..100 {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            format!("{}_{}", agent_id, i).hash(&mut hasher);
            let hash = hasher.finish();
            
            self.nodes.insert(hash, agent_id);
            self.ring.push(hash);
        }
        
        self.ring.sort();
    }
    
    fn remove_node(&mut self, agent_id: AgentId) {
        self.nodes.retain(|_, &mut v| v != agent_id);
        self.ring.retain(|&hash| self.nodes.contains_key(&hash));
    }
    
    fn get_node(&self, key: &str) -> Option<AgentId> {
        if self.ring.is_empty() {
            return None;
        }
        
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        key.hash(&mut hasher);
        let hash = hasher.finish();
        
        // Find the first node with hash >= key hash
        match self.ring.binary_search(&hash) {
            Ok(index) => self.nodes.get(&self.ring[index]).copied(),
            Err(index) => {
                if index >= self.ring.len() {
                    // Wrap around to the first node
                    self.nodes.get(&self.ring[0]).copied()
                } else {
                    self.nodes.get(&self.ring[index]).copied()
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::communication::MessageRouter;
    use std::sync::Arc;
    
    #[tokio::test]
    async fn test_agent_load_creation() {
        let agent_id = AgentId::new();
        let mut load = AgentLoad::new(agent_id);
        
        assert_eq!(load.agent_id, agent_id);
        assert_eq!(load.active_tasks, 0);
        assert_eq!(load.availability_score, 1.0);
        
        load.update_metrics(5, 50.0, 512 * 1024 * 1024, 100.0);
        assert_eq!(load.active_tasks, 5);
        assert!(load.availability_score < 1.0);
    }
    
    #[tokio::test]
    async fn test_consistent_hash_ring() {
        let mut ring = ConsistentHashRing::new();
        
        let agent1 = AgentId::new();
        let agent2 = AgentId::new();
        
        ring.add_node(agent1);
        ring.add_node(agent2);
        
        let selected1 = ring.get_node("test_key_1");
        let selected2 = ring.get_node("test_key_2");
        
        assert!(selected1.is_some());
        assert!(selected2.is_some());
        
        // Same key should always return same node
        assert_eq!(ring.get_node("test_key_1"), selected1);
    }
    
    #[tokio::test]
    async fn test_load_balancer_strategies() {
        let communication = Arc::new(MessageRouter::new());
        let agent_registry = Arc::new(AgentRegistry::new(communication));
        let load_balancer = AdaptiveLoadBalancer::new(agent_registry.clone());
        
        // Register some agents
        for i in 0..3 {
            let agent_info = crate::agent::AgentInfo::new(
                AgentId::new(),
                AgentType::Risk,
                format!("Agent {}", i),
                "1.0.0".to_string(),
            );
            agent_registry.register_agent(agent_info).await.unwrap();
        }
        
        // Test different strategies
        load_balancer.set_strategy(LoadBalancingStrategy::RoundRobin).await.unwrap();
        let agent1 = load_balancer.select_agent(Some(AgentType::Risk)).await.unwrap();
        assert!(agent1.is_some());
        
        load_balancer.set_strategy(LoadBalancingStrategy::Random).await.unwrap();
        let agent2 = load_balancer.select_agent(Some(AgentType::Risk)).await.unwrap();
        assert!(agent2.is_some());
        
        load_balancer.set_strategy(LoadBalancingStrategy::LeastConnections).await.unwrap();
        let agent3 = load_balancer.select_agent(Some(AgentType::Risk)).await.unwrap();
        assert!(agent3.is_some());
    }
    
    #[tokio::test]
    async fn test_load_balancer_statistics() {
        let communication = Arc::new(MessageRouter::new());
        let agent_registry = Arc::new(AgentRegistry::new(communication));
        let load_balancer = AdaptiveLoadBalancer::new(agent_registry.clone());
        
        // Register an agent
        let agent_info = crate::agent::AgentInfo::new(
            AgentId::new(),
            AgentType::Risk,
            "Test Agent".to_string(),
            "1.0.0".to_string(),
        );
        let agent_id = agent_info.id;
        agent_registry.register_agent(agent_info).await.unwrap();
        
        // Update load
        let load = AgentLoad::new(agent_id);
        load_balancer.update_agent_load(agent_id, load).await.unwrap();
        
        // Get statistics
        let stats = load_balancer.get_load_stats().await.unwrap();
        assert_eq!(stats.total_tasks_distributed, 0);
        
        // Select agent to increment counter
        let _selected = load_balancer.select_agent(Some(AgentType::Risk)).await.unwrap();
        
        let stats = load_balancer.get_load_stats().await.unwrap();
        assert_eq!(stats.total_tasks_distributed, 1);
    }
}