//! Swarm Intelligence - Emergent behavior and collective learning

use crate::lattice::{LatticeNode, NodeHealth};
use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use dashmap::DashMap;
use parking_lot::RwLock;
use std::sync::Arc;

/// Swarm intelligence coordinator
#[derive(Debug)]
pub struct SwarmIntelligence {
    /// Pheromone trails between nodes (edge weights)
    pub pheromone_trails: Arc<DashMap<(u32, u32), f64>>,
    
    /// Collective memory of successful strategies
    pub collective_memory: Arc<RwLock<Vec<SuccessfulStrategy>>>,
    
    /// Detected emergence patterns
    pub emergence_patterns: Arc<RwLock<Vec<EmergencePattern>>>,
    
    /// Pheromone evaporation rate
    evaporation_rate: f64,
    
    /// Pheromone deposit strength
    deposit_strength: f64,
}

impl SwarmIntelligence {
    pub fn new() -> Self {
        Self {
            pheromone_trails: Arc::new(DashMap::new()),
            collective_memory: Arc::new(RwLock::new(Vec::new())),
            emergence_patterns: Arc::new(RwLock::new(Vec::new())),
            evaporation_rate: 0.1,
            deposit_strength: 1.0,
        }
    }
    
    /// Update swarm intelligence based on node behavior
    pub fn update(&mut self, nodes: &[LatticeNode]) {
        // Evaporate existing pheromones
        self.evaporate_pheromones();
        
        // Detect successful node clusters
        let successful_clusters = self.detect_successful_clusters(nodes);
        
        // Strengthen connections between successful nodes
        for cluster in &successful_clusters {
            self.strengthen_cluster_connections(cluster);
        }
        
        // Detect emergence patterns
        self.detect_emergence_patterns(nodes, &successful_clusters);
        
        // Update collective memory
        self.update_collective_memory(nodes);
    }
    
    /// Evaporate pheromones over time
    fn evaporate_pheromones(&self) {
        for mut entry in self.pheromone_trails.iter_mut() {
            *entry.value_mut() *= (1.0 - self.evaporation_rate);
        }
        
        // Remove trails that are too weak
        self.pheromone_trails.retain(|_, &mut strength| strength > 0.001);
    }
    
    /// Detect clusters of successful nodes
    pub fn detect_successful_clusters(&self, nodes: &[LatticeNode]) -> Vec<Vec<u32>> {
        let mut clusters = Vec::new();
        let mut visited = HashMap::new();
        
        for node in nodes {
            if visited.contains_key(&node.id) {
                continue;
            }
            
            let health = node.get_health();
            if health.success_rate > 0.6 {
                // Start a new cluster
                let cluster = self.explore_cluster(node, nodes, &mut visited);
                if cluster.len() > 2 {
                    clusters.push(cluster);
                }
            }
        }
        
        clusters
    }
    
    /// Explore a cluster starting from a node
    fn explore_cluster(
        &self,
        start_node: &LatticeNode,
        all_nodes: &[LatticeNode],
        visited: &mut HashMap<u32, bool>,
    ) -> Vec<u32> {
        let mut cluster = vec![start_node.id];
        visited.insert(start_node.id, true);
        
        let mut to_explore = start_node.neighbors.clone();
        
        while let Some(neighbor_id) = to_explore.pop() {
            if visited.contains_key(&neighbor_id) {
                continue;
            }
            
            if let Some(neighbor) = all_nodes.iter().find(|n| n.id == neighbor_id) {
                let health = neighbor.get_health();
                if health.success_rate > 0.5 {
                    cluster.push(neighbor_id);
                    visited.insert(neighbor_id, true);
                    to_explore.extend(&neighbor.neighbors);
                }
            }
        }
        
        cluster
    }
    
    /// Strengthen pheromone trails between cluster nodes
    fn strengthen_cluster_connections(&self, cluster: &[u32]) {
        for i in 0..cluster.len() {
            for j in i+1..cluster.len() {
                let edge = if cluster[i] < cluster[j] {
                    (cluster[i], cluster[j])
                } else {
                    (cluster[j], cluster[i])
                };
                
                self.pheromone_trails
                    .entry(edge)
                    .and_modify(|strength| *strength += self.deposit_strength)
                    .or_insert(self.deposit_strength);
            }
        }
    }
    
    /// Detect emergence patterns in the swarm
    fn detect_emergence_patterns(
        &self,
        nodes: &[LatticeNode],
        successful_clusters: &[Vec<u32>],
    ) {
        let mut patterns = self.emergence_patterns.write();
        
        for (idx, cluster) in successful_clusters.iter().enumerate() {
            let mut total_pnl = 0.0;
            let mut sync_strength = 0.0;
            
            for &node_id in cluster {
                if let Some(node) = nodes.iter().find(|n| n.id == node_id) {
                    let stats = node.execution_stats.lock();
                    total_pnl += stats.total_pnl;
                }
            }
            
            // Calculate synchronization strength based on pheromone trails
            for i in 0..cluster.len() {
                for j in i+1..cluster.len() {
                    let edge = if cluster[i] < cluster[j] {
                        (cluster[i], cluster[j])
                    } else {
                        (cluster[j], cluster[i])
                    };
                    
                    if let Some(strength) = self.pheromone_trails.get(&edge) {
                        sync_strength += *strength;
                    }
                }
            }
            
            if cluster.len() > 1 {
                sync_strength /= (cluster.len() * (cluster.len() - 1) / 2) as f64;
            }
            
            let pattern_id = patterns.len() as u64;
            patterns.push(EmergencePattern {
                pattern_id,
                node_cluster: cluster.clone(),
                synchronization_strength: sync_strength,
                profit_contribution: total_pnl,
            });
        }
        
        // Keep only recent patterns
        if patterns.len() > 100 {
            let pattern_len = patterns.len();
            patterns.drain(0..pattern_len - 100);
        }
    }
    
    /// Update collective memory with successful strategies
    fn update_collective_memory(&self, nodes: &[LatticeNode]) {
        let mut memory = self.collective_memory.write();
        
        for node in nodes {
            let health = node.get_health();
            if health.success_rate > 0.7 && health.trades_executed > 100 {
                // Create a pattern signature based on node behavior
                let pattern_signature = self.create_pattern_signature(node);
                
                memory.push(SuccessfulStrategy {
                    pattern_signature,
                    success_rate: health.success_rate,
                    market_conditions: crate::core::MarketRegime::LowVolatility, // Placeholder
                    timestamp: chrono::Utc::now().timestamp() as u64,
                    node_id: node.id,
                });
            }
        }
        
        // Keep only recent memories
        if memory.len() > 1000 {
            let mem_len = memory.len();
            memory.drain(0..mem_len - 1000);
        }
    }
    
    /// Create a pattern signature for a node's behavior
    fn create_pattern_signature(&self, node: &LatticeNode) -> [u8; 32] {
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;
        
        let mut hasher = DefaultHasher::new();
        node.id.hash(&mut hasher);
        
        let hash = hasher.finish();
        let mut signature = [0u8; 32];
        
        for (i, byte) in hash.to_le_bytes().iter().enumerate() {
            signature[i % 32] ^= byte;
        }
        
        signature
    }
    
    /// Get pheromone strength between two nodes
    pub fn get_pheromone_strength(&self, node1: u32, node2: u32) -> f64 {
        let edge = if node1 < node2 {
            (node1, node2)
        } else {
            (node2, node1)
        };
        
        self.pheromone_trails
            .get(&edge)
            .map(|entry| *entry.value())
            .unwrap_or(0.0)
    }
}

/// Successful strategy record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuccessfulStrategy {
    pub pattern_signature: [u8; 32],
    pub success_rate: f64,
    pub market_conditions: crate::core::MarketRegime,
    pub timestamp: u64,
    pub node_id: u32,
}

/// Detected emergence pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmergencePattern {
    pub pattern_id: u64,
    pub node_cluster: Vec<u32>,
    pub synchronization_strength: f64,
    pub profit_contribution: f64,
}

impl Default for SwarmIntelligence {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_swarm_intelligence_creation() {
        let swarm = SwarmIntelligence::new();
        assert_eq!(swarm.evaporation_rate, 0.1);
        assert_eq!(swarm.deposit_strength, 1.0);
    }

    #[test]
    fn test_pheromone_operations() {
        let swarm = SwarmIntelligence::new();
        
        // Add pheromone trail
        swarm.pheromone_trails.insert((0, 1), 1.0);
        
        // Check retrieval
        assert_eq!(swarm.get_pheromone_strength(0, 1), 1.0);
        assert_eq!(swarm.get_pheromone_strength(1, 0), 1.0); // Should work both ways
        
        // Evaporate
        swarm.evaporate_pheromones();
        assert_eq!(swarm.get_pheromone_strength(0, 1), 0.9);
    }
}