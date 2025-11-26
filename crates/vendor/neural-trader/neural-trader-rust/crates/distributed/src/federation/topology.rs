// Federation topology patterns

use super::{AgentId, AgentMetadata};
use crate::{DistributedError, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Topology type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TopologyType {
    /// Hierarchical: Queen-led coordination with worker layers
    Hierarchical,

    /// Mesh: Peer-to-peer, all agents can communicate directly
    Mesh,

    /// Ring: Agents connected in a circular pattern
    Ring,

    /// Star: Central coordinator with spoke agents
    Star,

    /// Adaptive: Dynamic topology that changes based on workload
    Adaptive,
}

/// Topology configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologyConfig {
    /// Topology type
    pub topology_type: TopologyType,

    /// Leader election enabled (for hierarchical/star)
    pub leader_election: bool,

    /// Maximum hops for message routing
    pub max_hops: u32,

    /// Redundancy factor for fault tolerance
    pub redundancy_factor: u32,
}

impl Default for TopologyConfig {
    fn default() -> Self {
        Self {
            topology_type: TopologyType::Mesh,
            leader_election: true,
            max_hops: 5,
            redundancy_factor: 2,
        }
    }
}

/// Federation topology manager
pub struct FederationTopology {
    /// Topology configuration
    config: TopologyConfig,

    /// Agent registry
    agents: HashMap<AgentId, AgentMetadata>,

    /// Adjacency list for connections
    connections: HashMap<AgentId, HashSet<AgentId>>,

    /// Leader agent (for hierarchical/star topologies)
    leader: Option<AgentId>,
}

impl FederationTopology {
    /// Create new topology
    pub fn new(config: TopologyConfig) -> Self {
        Self {
            config,
            agents: HashMap::new(),
            connections: HashMap::new(),
            leader: None,
        }
    }

    /// Register an agent in the topology
    pub fn register_agent(&mut self, metadata: AgentMetadata) -> Result<()> {
        let agent_id = metadata.id.clone();

        // Add agent to registry
        self.agents.insert(agent_id.clone(), metadata);

        // Initialize connections based on topology
        self.initialize_connections(&agent_id)?;

        // Update leader if needed
        if self.config.leader_election && self.leader.is_none() {
            self.leader = Some(agent_id);
        }

        Ok(())
    }

    /// Remove agent from topology
    pub fn unregister_agent(&mut self, agent_id: &AgentId) -> Result<()> {
        // Remove agent
        self.agents
            .remove(agent_id)
            .ok_or_else(|| DistributedError::AgentNotFound(agent_id.clone()))?;

        // Remove all connections
        self.connections.remove(agent_id);
        for connections in self.connections.values_mut() {
            connections.remove(agent_id);
        }

        // Update leader if needed
        if self.leader.as_ref() == Some(agent_id) {
            self.elect_leader();
        }

        Ok(())
    }

    /// Get agent neighbors (connected agents)
    pub fn get_neighbors(&self, agent_id: &AgentId) -> Result<Vec<AgentId>> {
        let connections = self
            .connections
            .get(agent_id)
            .ok_or_else(|| DistributedError::AgentNotFound(agent_id.clone()))?;

        Ok(connections.iter().cloned().collect())
    }

    /// Get all agents
    pub fn get_agents(&self) -> Vec<&AgentMetadata> {
        self.agents.values().collect()
    }

    /// Get leader agent
    pub fn get_leader(&self) -> Option<&AgentId> {
        self.leader.as_ref()
    }

    /// Initialize connections based on topology type
    fn initialize_connections(&mut self, agent_id: &AgentId) -> Result<()> {
        match self.config.topology_type {
            TopologyType::Mesh => self.connect_mesh(agent_id),
            TopologyType::Hierarchical => self.connect_hierarchical(agent_id),
            TopologyType::Ring => self.connect_ring(agent_id),
            TopologyType::Star => self.connect_star(agent_id),
            TopologyType::Adaptive => self.connect_adaptive(agent_id),
        }
    }

    /// Connect agent in mesh topology (all-to-all)
    fn connect_mesh(&mut self, agent_id: &AgentId) -> Result<()> {
        let all_agents: Vec<AgentId> = self.agents.keys().cloned().collect();

        // Connect new agent to all existing agents
        for other_id in &all_agents {
            if other_id != agent_id {
                self.connections
                    .entry(agent_id.clone())
                    .or_insert_with(HashSet::new)
                    .insert(other_id.clone());

                self.connections
                    .entry(other_id.clone())
                    .or_insert_with(HashSet::new)
                    .insert(agent_id.clone());
            }
        }

        Ok(())
    }

    /// Connect agent in hierarchical topology
    fn connect_hierarchical(&mut self, agent_id: &AgentId) -> Result<()> {
        // If there's a leader, connect to leader
        if let Some(leader_id) = &self.leader {
            if leader_id != agent_id {
                self.connections
                    .entry(agent_id.clone())
                    .or_insert_with(HashSet::new)
                    .insert(leader_id.clone());

                self.connections
                    .entry(leader_id.clone())
                    .or_insert_with(HashSet::new)
                    .insert(agent_id.clone());
            }
        }

        Ok(())
    }

    /// Connect agent in ring topology
    fn connect_ring(&mut self, agent_id: &AgentId) -> Result<()> {
        let mut agent_ids: Vec<AgentId> = self.agents.keys().cloned().collect();
        agent_ids.sort();

        let idx = agent_ids.iter().position(|id| id == agent_id).unwrap();
        let n = agent_ids.len();

        if n > 1 {
            // Connect to previous agent
            let prev_idx = if idx == 0 { n - 1 } else { idx - 1 };
            let prev_id = &agent_ids[prev_idx];

            self.connections
                .entry(agent_id.clone())
                .or_insert_with(HashSet::new)
                .insert(prev_id.clone());

            // Connect to next agent
            let next_idx = (idx + 1) % n;
            let next_id = &agent_ids[next_idx];

            self.connections
                .entry(agent_id.clone())
                .or_insert_with(HashSet::new)
                .insert(next_id.clone());

            // Update neighbors' connections
            self.connections
                .entry(prev_id.clone())
                .or_insert_with(HashSet::new)
                .insert(agent_id.clone());

            self.connections
                .entry(next_id.clone())
                .or_insert_with(HashSet::new)
                .insert(agent_id.clone());
        }

        Ok(())
    }

    /// Connect agent in star topology
    fn connect_star(&mut self, agent_id: &AgentId) -> Result<()> {
        // Similar to hierarchical for now
        self.connect_hierarchical(agent_id)
    }

    /// Connect agent in adaptive topology
    fn connect_adaptive(&mut self, agent_id: &AgentId) -> Result<()> {
        // Start with mesh, can adapt later based on load
        self.connect_mesh(agent_id)
    }

    /// Elect a new leader
    fn elect_leader(&mut self) {
        // Simple election: choose agent with most capabilities
        self.leader = self
            .agents
            .iter()
            .max_by_key(|(_, metadata)| metadata.capabilities.len())
            .map(|(id, _)| id.clone());
    }

    /// Get topology statistics
    pub fn stats(&self) -> TopologyStats {
        TopologyStats {
            total_agents: self.agents.len(),
            total_connections: self.connections.values().map(|c| c.len()).sum(),
            topology_type: self.config.topology_type,
            has_leader: self.leader.is_some(),
        }
    }
}

/// Topology statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologyStats {
    /// Total number of agents
    pub total_agents: usize,

    /// Total number of connections
    pub total_connections: usize,

    /// Topology type
    pub topology_type: TopologyType,

    /// Has leader
    pub has_leader: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::federation::{AgentStatus, ResourceLimits};

    fn create_test_agent(id: &str) -> AgentMetadata {
        AgentMetadata {
            id: id.to_string(),
            agent_type: "worker".to_string(),
            capabilities: vec!["compute".to_string()],
            status: AgentStatus::Idle,
            node: "node-1".to_string(),
            resources: ResourceLimits::default(),
            metadata: HashMap::new(),
        }
    }

    #[test]
    fn test_mesh_topology() {
        let _config = TopologyConfig {
            topology_type: TopologyType::Mesh,
            ..Default::default()
        };

        let mut topology = FederationTopology::new(config);

        // Register agents
        topology.register_agent(create_test_agent("agent-1")).unwrap();
        topology.register_agent(create_test_agent("agent-2")).unwrap();
        topology.register_agent(create_test_agent("agent-3")).unwrap();

        // In mesh, each agent should connect to all others
        let neighbors = topology.get_neighbors(&"agent-1".to_string()).unwrap();
        assert_eq!(neighbors.len(), 2);

        let stats = topology.stats();
        assert_eq!(stats.total_agents, 3);
    }

    #[test]
    fn test_agent_registration() {
        let _config = TopologyConfig::default();
        let mut topology = FederationTopology::new(config);

        let agent = create_test_agent("agent-1");
        topology.register_agent(agent).unwrap();

        assert_eq!(topology.get_agents().len(), 1);
        assert!(topology.get_leader().is_some());
    }

    #[test]
    fn test_agent_unregistration() {
        let _config = TopologyConfig::default();
        let mut topology = FederationTopology::new(config);

        topology.register_agent(create_test_agent("agent-1")).unwrap();
        topology.register_agent(create_test_agent("agent-2")).unwrap();

        topology.unregister_agent(&"agent-1".to_string()).unwrap();

        assert_eq!(topology.get_agents().len(), 1);
    }
}
