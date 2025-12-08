//! Swarm Topologies
//!
//! Multiple network topologies for swarm agent organization:
//! - **Star**: Central coordinator with all agents connected to center
//! - **Ring**: Circular connection, each agent knows neighbors
//! - **Mesh**: Full connectivity, all agents see all others
//! - **Hierarchical**: Tree structure with leaders and followers
//! - **Hyperbolic**: Poincaré disk embedding for hierarchical + local
//! - **Small World**: Random rewiring for short paths (Watts-Strogatz)
//! - **Scale Free**: Power law degree distribution (Barabási-Albert)
//! - **Dynamic**: Topology that evolves based on performance

use std::collections::{HashMap, HashSet, VecDeque};
use serde::{Deserialize, Serialize};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use uuid::Uuid;

use crate::SwarmResult;

/// Types of swarm topologies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TopologyType {
    /// Star topology - all agents connect to central hub
    Star,
    /// Ring topology - circular connection
    Ring,
    /// Mesh topology - fully connected
    Mesh,
    /// Hierarchical - tree structure
    Hierarchical,
    /// Hyperbolic - Poincaré disk embedding
    Hyperbolic,
    /// Small World - Watts-Strogatz model
    SmallWorld,
    /// Scale Free - Barabási-Albert model
    ScaleFree,
    /// Dynamic - evolves over time
    Dynamic,
    /// Random - Erdős-Rényi random graph
    Random,
    /// Lattice - grid-based topology
    Lattice,
}

impl TopologyType {
    /// Get communication pattern characteristics
    pub fn characteristics(&self) -> TopologyCharacteristics {
        match self {
            TopologyType::Star => TopologyCharacteristics {
                avg_path_length: 2.0,
                clustering_coefficient: 0.0,
                fault_tolerance: "Low (hub failure critical)",
                scalability: "High",
                broadcast_efficiency: "High",
            },
            TopologyType::Ring => TopologyCharacteristics {
                avg_path_length: 0.5, // N/2 average
                clustering_coefficient: 0.0,
                fault_tolerance: "Medium",
                scalability: "Medium",
                broadcast_efficiency: "Low",
            },
            TopologyType::Mesh => TopologyCharacteristics {
                avg_path_length: 1.0,
                clustering_coefficient: 1.0,
                fault_tolerance: "Very High",
                scalability: "Low (O(n²) connections)",
                broadcast_efficiency: "Very High",
            },
            TopologyType::Hierarchical => TopologyCharacteristics {
                avg_path_length: 3.0, // log(N) average
                clustering_coefficient: 0.0,
                fault_tolerance: "Medium (depends on leader)",
                scalability: "Very High",
                broadcast_efficiency: "High (top-down)",
            },
            TopologyType::Hyperbolic => TopologyCharacteristics {
                avg_path_length: 2.0, // log(N)
                clustering_coefficient: 0.8,
                fault_tolerance: "High",
                scalability: "Very High",
                broadcast_efficiency: "High",
            },
            TopologyType::SmallWorld => TopologyCharacteristics {
                avg_path_length: 2.0, // log(N)
                clustering_coefficient: 0.6,
                fault_tolerance: "High",
                scalability: "High",
                broadcast_efficiency: "Medium",
            },
            TopologyType::ScaleFree => TopologyCharacteristics {
                avg_path_length: 2.0,
                clustering_coefficient: 0.3,
                fault_tolerance: "Low (hub attack vulnerability)",
                scalability: "Very High",
                broadcast_efficiency: "High",
            },
            _ => TopologyCharacteristics {
                avg_path_length: 3.0,
                clustering_coefficient: 0.5,
                fault_tolerance: "Medium",
                scalability: "Medium",
                broadcast_efficiency: "Medium",
            },
        }
    }
}

/// Characteristics of a topology
#[derive(Debug, Clone)]
pub struct TopologyCharacteristics {
    pub avg_path_length: f64,
    pub clustering_coefficient: f64,
    pub fault_tolerance: &'static str,
    pub scalability: &'static str,
    pub broadcast_efficiency: &'static str,
}

/// A connection between two agents
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Connection {
    /// Source agent ID
    pub from: Uuid,
    /// Target agent ID
    pub to: Uuid,
    /// Connection weight/strength
    pub weight: f64,
    /// Latency in milliseconds
    pub latency: f64,
    /// Bandwidth (messages per second)
    pub bandwidth: f64,
    /// Is connection bidirectional?
    pub bidirectional: bool,
    /// Connection health (0.0 to 1.0)
    pub health: f64,
}

/// Metrics for a topology
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologyMetrics {
    /// Number of nodes
    pub node_count: usize,
    /// Number of edges
    pub edge_count: usize,
    /// Average degree
    pub avg_degree: f64,
    /// Clustering coefficient
    pub clustering_coefficient: f64,
    /// Average path length
    pub avg_path_length: f64,
    /// Diameter (longest shortest path)
    pub diameter: usize,
    /// Is connected?
    pub is_connected: bool,
    /// Density (actual edges / possible edges)
    pub density: f64,
}

/// Swarm topology manager
#[derive(Debug)]
pub struct SwarmTopology {
    /// Topology type
    topology_type: TopologyType,
    /// Agent IDs
    agents: Vec<Uuid>,
    /// Connections (adjacency list)
    connections: HashMap<Uuid, Vec<Connection>>,
    /// Agent positions (for spatial topologies)
    positions: HashMap<Uuid, (f64, f64)>, // Poincaré disk coordinates
    /// Random number generator
    rng: ChaCha8Rng,
}

impl SwarmTopology {
    /// Create a new topology
    pub fn new(topology_type: TopologyType, agent_count: usize) -> SwarmResult<Self> {
        let agents: Vec<Uuid> = (0..agent_count).map(|_| Uuid::new_v4()).collect();
        let mut rng = ChaCha8Rng::from_entropy();
        
        let mut topology = Self {
            topology_type,
            agents: agents.clone(),
            connections: HashMap::new(),
            positions: HashMap::new(),
            rng,
        };
        
        // Initialize connections based on topology type
        topology.build_topology()?;
        
        Ok(topology)
    }
    
    /// Build the topology connections
    fn build_topology(&mut self) -> SwarmResult<()> {
        match self.topology_type {
            TopologyType::Star => self.build_star(),
            TopologyType::Ring => self.build_ring(),
            TopologyType::Mesh => self.build_mesh(),
            TopologyType::Hierarchical => self.build_hierarchical(),
            TopologyType::Hyperbolic => self.build_hyperbolic(),
            TopologyType::SmallWorld => self.build_small_world(4, 0.1),
            TopologyType::ScaleFree => self.build_scale_free(2),
            TopologyType::Random => self.build_random(0.3),
            TopologyType::Lattice => self.build_lattice(),
            TopologyType::Dynamic => self.build_mesh(), // Start fully connected
        }
    }
    
    /// Build star topology
    fn build_star(&mut self) -> SwarmResult<()> {
        if self.agents.is_empty() {
            return Ok(());
        }
        
        let hub = self.agents[0];
        let other_agents: Vec<Uuid> = self.agents[1..].to_vec();
        for agent in other_agents {
            self.add_connection(hub, agent, 1.0, true);
        }
        
        Ok(())
    }
    
    /// Build ring topology
    fn build_ring(&mut self) -> SwarmResult<()> {
        let agents = self.agents.clone();
        let n = agents.len();
        for i in 0..n {
            let next = (i + 1) % n;
            self.add_connection(agents[i], agents[next], 1.0, true);
        }
        
        Ok(())
    }
    
    /// Build fully connected mesh
    fn build_mesh(&mut self) -> SwarmResult<()> {
        let agents = self.agents.clone();
        let n = agents.len();
        for i in 0..n {
            for j in (i + 1)..n {
                self.add_connection(agents[i], agents[j], 1.0, true);
            }
        }
        
        Ok(())
    }
    
    /// Build hierarchical tree topology
    fn build_hierarchical(&mut self) -> SwarmResult<()> {
        let n = self.agents.len();
        if n == 0 { return Ok(()); }
        
        // Binary tree structure
        for i in 0..n {
            let left = 2 * i + 1;
            let right = 2 * i + 2;
            
            if left < n {
                self.add_connection(self.agents[i], self.agents[left], 1.0, true);
            }
            if right < n {
                self.add_connection(self.agents[i], self.agents[right], 1.0, true);
            }
        }
        
        Ok(())
    }
    
    /// Build hyperbolic (Poincaré disk) topology
    fn build_hyperbolic(&mut self) -> SwarmResult<()> {
        let n = self.agents.len();
        
        // Place agents in Poincaré disk
        for (i, agent) in self.agents.iter().enumerate() {
            // Spiral layout in Poincaré disk
            let layer = ((i as f64).sqrt() as usize).max(1);
            let angle = 2.0 * std::f64::consts::PI * (i as f64) / (layer as f64 * 6.0);
            let r = 1.0 - 1.0 / (layer as f64 + 1.0);
            
            let x = r * angle.cos();
            let y = r * angle.sin();
            self.positions.insert(*agent, (x, y));
        }
        
        // Connect based on hyperbolic distance
        for i in 0..n {
            for j in (i + 1)..n {
                let dist = self.hyperbolic_distance(self.agents[i], self.agents[j]);
                
                // Connect if within threshold (closer = stronger connection)
                if dist < 2.0 {
                    let weight = 1.0 / (1.0 + dist);
                    self.add_connection(self.agents[i], self.agents[j], weight, true);
                }
            }
        }
        
        Ok(())
    }
    
    /// Compute hyperbolic distance in Poincaré disk
    fn hyperbolic_distance(&self, a: Uuid, b: Uuid) -> f64 {
        let (x1, y1) = self.positions.get(&a).unwrap_or(&(0.0, 0.0));
        let (x2, y2) = self.positions.get(&b).unwrap_or(&(0.0, 0.0));
        
        let dx = x1 - x2;
        let dy = y1 - y2;
        let diff_sq = dx * dx + dy * dy;
        
        let norm1_sq = x1 * x1 + y1 * y1;
        let norm2_sq = x2 * x2 + y2 * y2;
        
        if norm1_sq >= 1.0 || norm2_sq >= 1.0 {
            return f64::INFINITY;
        }
        
        let denom_sq = (1.0 - norm1_sq) * (1.0 - norm2_sq);
        let cosh_dist = 1.0 + 2.0 * diff_sq / denom_sq;
        
        cosh_dist.acosh()
    }
    
    /// Build small-world topology (Watts-Strogatz)
    fn build_small_world(&mut self, k: usize, p: f64) -> SwarmResult<()> {
        let n = self.agents.len();
        
        // Start with ring lattice with k neighbors on each side
        for i in 0..n {
            for offset in 1..=k {
                let j = (i + offset) % n;
                self.add_connection(self.agents[i], self.agents[j], 1.0, true);
            }
        }
        
        // Rewire with probability p
        let agents_copy = self.agents.clone();
        for i in 0..n {
            for offset in 1..=k {
                if self.rng.gen::<f64>() < p {
                    let j = (i + offset) % n;
                    // Remove old connection
                    self.remove_connection(agents_copy[i], agents_copy[j]);
                    
                    // Add new random connection
                    let new_target_idx = self.rng.gen_range(0..n);
                    if new_target_idx != i {
                        self.add_connection(agents_copy[i], agents_copy[new_target_idx], 1.0, true);
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Build scale-free topology (Barabási-Albert)
    fn build_scale_free(&mut self, m: usize) -> SwarmResult<()> {
        let n = self.agents.len();
        if n < m + 1 { return self.build_mesh(); }
        
        // Start with complete graph on m+1 nodes
        for i in 0..=m {
            for j in (i + 1)..=m {
                self.add_connection(self.agents[i], self.agents[j], 1.0, true);
            }
        }
        
        // Preferential attachment for remaining nodes
        let mut degrees: Vec<usize> = vec![m; m + 1];
        
        for i in (m + 1)..n {
            // Connect to m existing nodes with probability proportional to degree
            let total_degree: usize = degrees.iter().sum();
            let mut connected = HashSet::new();
            
            while connected.len() < m {
                let r = self.rng.gen_range(0..total_degree);
                let mut cumsum = 0;
                for (j, &d) in degrees.iter().enumerate() {
                    cumsum += d;
                    if cumsum > r && !connected.contains(&j) {
                        connected.insert(j);
                        self.add_connection(self.agents[i], self.agents[j], 1.0, true);
                        break;
                    }
                }
            }
            
            // Update degrees
            degrees.push(m);
            for j in &connected {
                degrees[*j] += 1;
            }
        }
        
        Ok(())
    }
    
    /// Build random topology (Erdős-Rényi)
    fn build_random(&mut self, p: f64) -> SwarmResult<()> {
        let n = self.agents.len();
        
        for i in 0..n {
            for j in (i + 1)..n {
                if self.rng.gen::<f64>() < p {
                    self.add_connection(self.agents[i], self.agents[j], 1.0, true);
                }
            }
        }
        
        Ok(())
    }
    
    /// Build lattice topology
    fn build_lattice(&mut self) -> SwarmResult<()> {
        let n = self.agents.len();
        let side = (n as f64).sqrt() as usize;
        
        for i in 0..n {
            let row = i / side;
            let col = i % side;
            
            // Right neighbor
            if col + 1 < side && i + 1 < n {
                self.add_connection(self.agents[i], self.agents[i + 1], 1.0, true);
            }
            
            // Down neighbor
            if row + 1 < side && i + side < n {
                self.add_connection(self.agents[i], self.agents[i + side], 1.0, true);
            }
        }
        
        Ok(())
    }
    
    /// Add a connection
    fn add_connection(&mut self, from: Uuid, to: Uuid, weight: f64, bidirectional: bool) {
        let conn = Connection {
            from,
            to,
            weight,
            latency: 1.0,
            bandwidth: 100.0,
            bidirectional,
            health: 1.0,
        };
        
        self.connections.entry(from).or_default().push(conn.clone());
        
        if bidirectional {
            let reverse = Connection {
                from: to,
                to: from,
                weight,
                latency: 1.0,
                bandwidth: 100.0,
                bidirectional: true,
                health: 1.0,
            };
            self.connections.entry(to).or_default().push(reverse);
        }
    }
    
    /// Remove a connection
    fn remove_connection(&mut self, from: Uuid, to: Uuid) {
        if let Some(conns) = self.connections.get_mut(&from) {
            conns.retain(|c| c.to != to);
        }
        if let Some(conns) = self.connections.get_mut(&to) {
            conns.retain(|c| c.to != from);
        }
    }
    
    /// Get neighbors of an agent
    pub fn get_neighbors(&self, agent: Uuid) -> Vec<Uuid> {
        self.connections
            .get(&agent)
            .map(|conns| conns.iter().map(|c| c.to).collect())
            .unwrap_or_default()
    }
    
    /// Get all agents
    pub fn agents(&self) -> &[Uuid] {
        &self.agents
    }
    
    /// Add a new agent
    pub fn add_agent(&mut self) -> Uuid {
        let agent = Uuid::new_v4();
        self.agents.push(agent);
        
        // Connect based on topology type
        match self.topology_type {
            TopologyType::Star => {
                if self.agents.len() > 1 {
                    let hub = self.agents[0];
                    self.add_connection(hub, agent, 1.0, true);
                }
            }
            TopologyType::Mesh => {
                let existing_agents: Vec<Uuid> = self.agents[..self.agents.len() - 1].to_vec();
                for existing in existing_agents {
                    self.add_connection(existing, agent, 1.0, true);
                }
            }
            TopologyType::ScaleFree => {
                // Preferential attachment
                let degrees: Vec<usize> = self.agents[..self.agents.len() - 1]
                    .iter()
                    .map(|a| self.connections.get(a).map(|c| c.len()).unwrap_or(0))
                    .collect();
                let total: usize = degrees.iter().sum();
                if total > 0 {
                    let r = self.rng.gen_range(0..total);
                    let mut cumsum = 0;
                    for (i, &d) in degrees.iter().enumerate() {
                        cumsum += d;
                        if cumsum > r {
                            self.add_connection(self.agents[i], agent, 1.0, true);
                            break;
                        }
                    }
                }
            }
            _ => {}
        }
        
        agent
    }
    
    /// Compute topology metrics
    pub fn metrics(&self) -> TopologyMetrics {
        let node_count = self.agents.len();
        let edge_count: usize = self.connections.values().map(|v| v.len()).sum::<usize>() / 2;
        
        let avg_degree = if node_count > 0 {
            2.0 * edge_count as f64 / node_count as f64
        } else {
            0.0
        };
        
        let max_edges = node_count * (node_count - 1) / 2;
        let density = if max_edges > 0 {
            edge_count as f64 / max_edges as f64
        } else {
            0.0
        };
        
        TopologyMetrics {
            node_count,
            edge_count,
            avg_degree,
            clustering_coefficient: self.compute_clustering_coefficient(),
            avg_path_length: self.compute_avg_path_length(),
            diameter: self.compute_diameter(),
            is_connected: self.is_connected(),
            density,
        }
    }
    
    /// Compute clustering coefficient
    fn compute_clustering_coefficient(&self) -> f64 {
        let mut total = 0.0;
        let mut count = 0;
        
        for agent in &self.agents {
            let neighbors = self.get_neighbors(*agent);
            let k = neighbors.len();
            if k < 2 { continue; }
            
            let mut triangles = 0;
            for i in 0..neighbors.len() {
                for j in (i + 1)..neighbors.len() {
                    if self.get_neighbors(neighbors[i]).contains(&neighbors[j]) {
                        triangles += 1;
                    }
                }
            }
            
            let possible = k * (k - 1) / 2;
            total += triangles as f64 / possible as f64;
            count += 1;
        }
        
        if count > 0 { total / count as f64 } else { 0.0 }
    }
    
    /// Compute average path length using BFS
    fn compute_avg_path_length(&self) -> f64 {
        let mut total = 0.0;
        let mut count = 0;
        
        for start in &self.agents {
            let mut distances = HashMap::new();
            let mut queue = VecDeque::new();
            
            distances.insert(*start, 0);
            queue.push_back(*start);
            
            while let Some(current) = queue.pop_front() {
                let current_dist = distances[&current];
                for neighbor in self.get_neighbors(current) {
                    if !distances.contains_key(&neighbor) {
                        distances.insert(neighbor, current_dist + 1);
                        queue.push_back(neighbor);
                    }
                }
            }
            
            for (_, &dist) in &distances {
                if dist > 0 {
                    total += dist as f64;
                    count += 1;
                }
            }
        }
        
        if count > 0 { total / count as f64 } else { 0.0 }
    }
    
    /// Compute diameter
    fn compute_diameter(&self) -> usize {
        let mut max_dist = 0;
        
        for start in &self.agents {
            let mut distances = HashMap::new();
            let mut queue = VecDeque::new();
            
            distances.insert(*start, 0);
            queue.push_back(*start);
            
            while let Some(current) = queue.pop_front() {
                let current_dist = distances[&current];
                for neighbor in self.get_neighbors(current) {
                    if !distances.contains_key(&neighbor) {
                        let new_dist = current_dist + 1;
                        distances.insert(neighbor, new_dist);
                        max_dist = max_dist.max(new_dist);
                        queue.push_back(neighbor);
                    }
                }
            }
        }
        
        max_dist
    }
    
    /// Check if topology is connected
    fn is_connected(&self) -> bool {
        if self.agents.is_empty() { return true; }
        
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        
        queue.push_back(self.agents[0]);
        visited.insert(self.agents[0]);
        
        while let Some(current) = queue.pop_front() {
            for neighbor in self.get_neighbors(current) {
                if !visited.contains(&neighbor) {
                    visited.insert(neighbor);
                    queue.push_back(neighbor);
                }
            }
        }
        
        visited.len() == self.agents.len()
    }
    
    /// Get topology type
    pub fn topology_type(&self) -> TopologyType {
        self.topology_type
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_star_topology() {
        let topology = SwarmTopology::new(TopologyType::Star, 10).unwrap();
        let metrics = topology.metrics();
        assert!(metrics.is_connected);
        assert_eq!(metrics.diameter, 2);
    }
    
    #[test]
    fn test_mesh_topology() {
        let topology = SwarmTopology::new(TopologyType::Mesh, 5).unwrap();
        let metrics = topology.metrics();
        assert!(metrics.is_connected);
        assert_eq!(metrics.diameter, 1);
        assert!((metrics.density - 1.0).abs() < 0.01);
    }
    
    #[test]
    fn test_hyperbolic_distance() {
        let topology = SwarmTopology::new(TopologyType::Hyperbolic, 20).unwrap();
        let metrics = topology.metrics();
        assert!(metrics.is_connected);
    }
}
