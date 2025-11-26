// Load balancing strategies

use crate::federation::AgentId;
use crate::{DistributedError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Load balancing strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LoadBalancingStrategy {
    /// Round-robin distribution
    RoundRobin,

    /// Least connections
    LeastConnections,

    /// Least response time
    LeastResponseTime,

    /// Random selection
    Random,

    /// Weighted round-robin
    WeightedRoundRobin,

    /// IP hash (sticky sessions)
    IpHash,
}

/// Agent load information
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AgentLoad {
    /// Number of active connections
    pub active_connections: usize,

    /// Average response time (ms)
    pub avg_response_time_ms: u64,

    /// Total requests handled
    pub total_requests: u64,

    /// Error count
    pub error_count: u64,

    /// Agent weight (for weighted strategies)
    pub weight: f64,
}

impl AgentLoad {
    /// Calculate load score (lower is better)
    pub fn load_score(&self) -> f64 {
        let connection_weight = 0.5;
        let response_time_weight = 0.3;
        let error_weight = 0.2;

        let connection_score = self.active_connections as f64;
        let response_time_score = self.avg_response_time_ms as f64 / 1000.0;
        let error_score = if self.total_requests > 0 {
            (self.error_count as f64 / self.total_requests as f64) * 100.0
        } else {
            0.0
        };

        connection_weight * connection_score
            + response_time_weight * response_time_score
            + error_weight * error_score
    }
}

/// Load balancer
pub struct LoadBalancer {
    /// Load balancing strategy
    strategy: LoadBalancingStrategy,

    /// Agent loads
    agent_loads: Arc<RwLock<HashMap<AgentId, AgentLoad>>>,

    /// Round-robin counter
    round_robin_counter: Arc<RwLock<usize>>,
}

impl LoadBalancer {
    /// Create new load balancer
    pub fn new(strategy: LoadBalancingStrategy) -> Self {
        Self {
            strategy,
            agent_loads: Arc::new(RwLock::new(HashMap::new())),
            round_robin_counter: Arc::new(RwLock::new(0)),
        }
    }

    /// Register agent
    pub async fn register_agent(&self, agent_id: AgentId, weight: f64) {
        let mut loads = self.agent_loads.write().await;
        loads.insert(
            agent_id,
            AgentLoad {
                weight,
                ..Default::default()
            },
        );
    }

    /// Unregister agent
    pub async fn unregister_agent(&self, agent_id: &AgentId) {
        self.agent_loads.write().await.remove(agent_id);
    }

    /// Select agent for new request
    pub async fn select_agent(&self) -> Result<AgentId> {
        let loads = self.agent_loads.read().await;

        if loads.is_empty() {
            return Err(DistributedError::FederationError(
                "No agents available".to_string(),
            ));
        }

        match self.strategy {
            LoadBalancingStrategy::RoundRobin => self.select_round_robin(&loads).await,
            LoadBalancingStrategy::LeastConnections => self.select_least_connections(&loads),
            LoadBalancingStrategy::LeastResponseTime => self.select_least_response_time(&loads),
            LoadBalancingStrategy::Random => self.select_random(&loads),
            LoadBalancingStrategy::WeightedRoundRobin => {
                self.select_weighted_round_robin(&loads).await
            }
            LoadBalancingStrategy::IpHash => self.select_round_robin(&loads).await, // Fallback to round-robin
        }
    }

    /// Round-robin selection
    async fn select_round_robin(&self, loads: &HashMap<AgentId, AgentLoad>) -> Result<AgentId> {
        let agents: Vec<_> = loads.keys().cloned().collect();
        let mut counter = self.round_robin_counter.write().await;

        let index = *counter % agents.len();
        *counter += 1;

        Ok(agents[index].clone())
    }

    /// Least connections selection
    fn select_least_connections(&self, loads: &HashMap<AgentId, AgentLoad>) -> Result<AgentId> {
        loads
            .iter()
            .min_by_key(|(_, load)| load.active_connections)
            .map(|(id, _)| id.clone())
            .ok_or_else(|| DistributedError::FederationError("No agents available".to_string()))
    }

    /// Least response time selection
    fn select_least_response_time(&self, loads: &HashMap<AgentId, AgentLoad>) -> Result<AgentId> {
        loads
            .iter()
            .min_by_key(|(_, load)| load.avg_response_time_ms)
            .map(|(id, _)| id.clone())
            .ok_or_else(|| DistributedError::FederationError("No agents available".to_string()))
    }

    /// Random selection
    fn select_random(&self, loads: &HashMap<AgentId, AgentLoad>) -> Result<AgentId> {
        use rand::seq::IteratorRandom;
        let mut rng = rand::thread_rng();

        loads
            .keys()
            .choose(&mut rng)
            .cloned()
            .ok_or_else(|| DistributedError::FederationError("No agents available".to_string()))
    }

    /// Weighted round-robin selection
    async fn select_weighted_round_robin(
        &self,
        loads: &HashMap<AgentId, AgentLoad>,
    ) -> Result<AgentId> {
        // Simplified: select based on weight scores
        loads
            .iter()
            .min_by(|(_, a), (_, b)| {
                let score_a = a.load_score() / a.weight.max(0.1);
                let score_b = b.load_score() / b.weight.max(0.1);
                score_a.partial_cmp(&score_b).unwrap()
            })
            .map(|(id, _)| id.clone())
            .ok_or_else(|| DistributedError::FederationError("No agents available".to_string()))
    }

    /// Update agent load
    pub async fn update_load(
        &self,
        agent_id: &AgentId,
        connections_delta: i32,
        response_time_ms: Option<u64>,
        error: bool,
    ) {
        let mut loads = self.agent_loads.write().await;

        if let Some(load) = loads.get_mut(agent_id) {
            // Update connections
            if connections_delta > 0 {
                load.active_connections += connections_delta as usize;
            } else {
                load.active_connections = load.active_connections.saturating_sub((-connections_delta) as usize);
            }

            // Update response time (moving average)
            if let Some(rt) = response_time_ms {
                load.avg_response_time_ms = if load.total_requests == 0 {
                    rt
                } else {
                    (load.avg_response_time_ms * load.total_requests + rt) / (load.total_requests + 1)
                };
            }

            // Update counters
            load.total_requests += 1;
            if error {
                load.error_count += 1;
            }
        }
    }

    /// Get load statistics
    pub async fn get_loads(&self) -> HashMap<AgentId, AgentLoad> {
        self.agent_loads.read().await.clone()
    }

    /// Get statistics
    pub async fn stats(&self) -> LoadBalancerStats {
        let loads = self.agent_loads.read().await;

        let total_agents = loads.len();
        let total_connections: usize = loads.values().map(|l| l.active_connections).sum();
        let total_requests: u64 = loads.values().map(|l| l.total_requests).sum();
        let total_errors: u64 = loads.values().map(|l| l.error_count).sum();

        let avg_response_time = if !loads.is_empty() {
            loads.values().map(|l| l.avg_response_time_ms).sum::<u64>() / loads.len() as u64
        } else {
            0
        };

        LoadBalancerStats {
            strategy: self.strategy,
            total_agents,
            total_connections,
            total_requests,
            total_errors,
            avg_response_time_ms: avg_response_time,
        }
    }
}

/// Load balancer statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadBalancerStats {
    /// Strategy used
    pub strategy: LoadBalancingStrategy,

    /// Total agents
    pub total_agents: usize,

    /// Total active connections
    pub total_connections: usize,

    /// Total requests processed
    pub total_requests: u64,

    /// Total errors
    pub total_errors: u64,

    /// Average response time
    pub avg_response_time_ms: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_load_balancer_registration() {
        let lb = LoadBalancer::new(LoadBalancingStrategy::RoundRobin);

        lb.register_agent("agent-1".to_string(), 1.0).await;
        lb.register_agent("agent-2".to_string(), 1.0).await;

        let stats = lb.stats().await;
        assert_eq!(stats.total_agents, 2);
    }

    #[tokio::test]
    async fn test_round_robin_selection() {
        let lb = LoadBalancer::new(LoadBalancingStrategy::RoundRobin);

        lb.register_agent("agent-1".to_string(), 1.0).await;
        lb.register_agent("agent-2".to_string(), 1.0).await;

        let agent1 = lb.select_agent().await.unwrap();
        let agent2 = lb.select_agent().await.unwrap();

        assert_ne!(agent1, agent2);
    }

    #[tokio::test]
    async fn test_load_update() {
        let lb = LoadBalancer::new(LoadBalancingStrategy::LeastConnections);

        lb.register_agent("agent-1".to_string(), 1.0).await;

        lb.update_load(&"agent-1".to_string(), 1, Some(100), false)
            .await;

        let loads = lb.get_loads().await;
        let load = loads.get(&"agent-1".to_string()).unwrap();

        assert_eq!(load.active_connections, 1);
        assert_eq!(load.total_requests, 1);
    }
}
