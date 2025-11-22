//! Mycelial Network organism - distributed coordination specialist
//!
//! The Mycelial Network organism represents a fungal-inspired parasitic system that
//! creates distributed networks across multiple trading pairs, sharing information
//! and coordinating attacks through underground communication channels.

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use tracing::{debug, info, warn};
use uuid::Uuid;
// use nalgebra::DVector; // Removed: unused
use dashmap::DashMap;
use std::sync::Arc;

use super::{
    AdaptationFeedback, BaseOrganism, InfectionResult, OrganismError, OrganismGenetics,
    ParasiticOrganism, ResourceMetrics,
};

/// Configuration for Mycelial Network organism
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MycelialConfig {
    /// Maximum network nodes (hyphae endpoints)
    pub max_network_nodes: usize,
    /// Information propagation speed across network
    pub propagation_speed: f64,
    /// Resource sharing efficiency between nodes
    pub resource_sharing_efficiency: f64,
    /// Network resilience to node failures
    pub network_resilience: f64,
    /// Spore production rate for network expansion
    pub spore_production_rate: f64,
    /// Inter-network communication strength
    pub communication_strength: f64,
    /// Nutrient extraction efficiency
    pub nutrient_extraction_rate: f64,
}

impl Default for MycelialConfig {
    fn default() -> Self {
        Self {
            max_network_nodes: 50,
            propagation_speed: 0.85,
            resource_sharing_efficiency: 0.90,
            network_resilience: 0.80,
            spore_production_rate: 0.20,
            communication_strength: 0.75,
            nutrient_extraction_rate: 0.65,
        }
    }
}

/// Individual node in the mycelial network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MycelialNode {
    pub node_id: Uuid,
    pub pair_id: String,
    pub position: (f64, f64),   // Virtual position in network space
    pub connections: Vec<Uuid>, // Connected node IDs
    pub nutrient_level: f64,
    pub information_cache: HashMap<String, serde_json::Value>,
    pub creation_time: DateTime<Utc>,
    pub last_activity: DateTime<Utc>,
    pub health_status: NodeHealthStatus,
    pub specialization: NodeSpecialization,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NodeHealthStatus {
    Healthy,
    Stressed,
    Damaged,
    Dormant,
    Reproducing,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NodeSpecialization {
    Scout,        // Information gathering
    Extractor,    // Resource extraction
    Communicator, // Network coordination
    Reproducer,   // Network expansion
    Defender,     // Network protection
}

/// Spore for network expansion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MycelialSpore {
    pub spore_id: Uuid,
    pub origin_node: Uuid,
    pub target_pair: String,
    pub germination_time: DateTime<Utc>,
    pub genetic_material: OrganismGenetics,
    pub specialization_preference: NodeSpecialization,
    pub viability: f64,
}

/// Information packet traveling through the network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkPacket {
    pub packet_id: Uuid,
    pub source_node: Uuid,
    pub target_nodes: Vec<Uuid>,
    pub packet_type: PacketType,
    pub data: serde_json::Value,
    pub hop_count: u32,
    pub timestamp: DateTime<Utc>,
    pub priority: PacketPriority,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PacketType {
    MarketData,
    ResourceRequest,
    ResourceShare,
    ThreatWarning,
    CoordinationCommand,
    HealthStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum PacketPriority {
    Low,
    Normal,
    High,
    Critical,
}

/// Mycelial Network organism - distributed coordination specialist
#[derive(Debug)]
pub struct MycelialNetworkOrganism {
    base: BaseOrganism,
    config: MycelialConfig,
    /// Network nodes (hyphae)
    network_nodes: Arc<DashMap<Uuid, MycelialNode>>,
    /// Active spores for expansion
    active_spores: Vec<MycelialSpore>,
    /// Network packet queue
    packet_queue: Vec<NetworkPacket>,
    /// Network topology map
    topology_map: HashMap<Uuid, Vec<Uuid>>,
    /// Shared resource pool
    shared_resources: f64,
    /// Information database
    information_database: HashMap<String, (DateTime<Utc>, serde_json::Value)>,
    /// Network performance metrics
    network_metrics: NetworkMetrics,
    /// Inter-network connections (connections to other mycelial networks)
    inter_network_connections: Vec<Uuid>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkMetrics {
    pub total_nodes: usize,
    pub active_connections: usize,
    pub information_flow_rate: f64,
    pub resource_distribution_efficiency: f64,
    pub network_coverage: f64,
    pub resilience_score: f64,
    pub average_response_time: f64,
}

impl MycelialNetworkOrganism {
    pub fn new() -> Self {
        Self::with_config(MycelialConfig::default())
    }

    pub fn with_config(config: MycelialConfig) -> Self {
        let mut base = BaseOrganism::new();

        // Mycelial networks excel at cooperation and information sharing
        base.genetics.cooperation = 0.95; // Extremely cooperative
        base.genetics.adaptability = 0.88; // Highly adaptable network
        base.genetics.efficiency = 0.92; // Very efficient resource usage
        base.genetics.resilience = 0.90; // Excellent network resilience
        base.genetics.stealth = 0.70; // Moderate stealth - networks can be detected
        base.genetics.aggression = 0.40; // Low aggression - focuses on cooperation

        Self {
            base,
            config,
            network_nodes: Arc::new(DashMap::new()),
            active_spores: Vec::new(),
            packet_queue: Vec::new(),
            topology_map: HashMap::new(),
            shared_resources: 0.0,
            information_database: HashMap::new(),
            network_metrics: NetworkMetrics {
                total_nodes: 0,
                active_connections: 0,
                information_flow_rate: 0.0,
                resource_distribution_efficiency: 0.0,
                network_coverage: 0.0,
                resilience_score: 0.0,
                average_response_time: 0.0,
            },
            inter_network_connections: Vec::new(),
        }
    }

    /// Create a new mycelial node at specified position
    pub async fn create_node(
        &mut self,
        pair_id: String,
        position: (f64, f64),
        specialization: NodeSpecialization,
    ) -> Result<Uuid, OrganismError> {
        if self.network_nodes.len() >= self.config.max_network_nodes {
            return Err(OrganismError::ResourceExhausted(
                "Maximum network nodes reached".to_string(),
            ));
        }

        let node_id = Uuid::new_v4();
        let node = MycelialNode {
            node_id,
            pair_id: pair_id.clone(),
            position,
            connections: Vec::new(),
            nutrient_level: 10.0, // Starting nutrient level
            information_cache: HashMap::new(),
            creation_time: Utc::now(),
            last_activity: Utc::now(),
            health_status: NodeHealthStatus::Healthy,
            specialization,
        };

        self.network_nodes.insert(node_id, node);

        // Establish connections to nearby nodes
        self.establish_connections(node_id).await?;

        // Update topology map
        self.update_topology_map().await;

        info!(
            "🍄 Created mycelial node {} at position ({:.2}, {:.2}) for pair {}",
            node_id, position.0, position.1, pair_id
        );

        Ok(node_id)
    }

    /// Establish connections between nodes based on proximity and compatibility
    async fn establish_connections(&mut self, new_node_id: Uuid) -> Result<(), OrganismError> {
        let new_node_pos = {
            let node_ref = self
                .network_nodes
                .get(&new_node_id)
                .ok_or_else(|| OrganismError::InfectionFailed("Node not found".to_string()))?;
            node_ref.position
        };

        let connection_radius = 5.0; // Maximum connection distance
        let max_connections = 6; // Maximum connections per node

        let mut potential_connections = Vec::new();

        // Find nearby nodes
        for node_ref in self.network_nodes.iter() {
            if *node_ref.key() == new_node_id {
                continue;
            }

            let distance = ((new_node_pos.0 - node_ref.position.0).powi(2)
                + (new_node_pos.1 - node_ref.position.1).powi(2))
            .sqrt();

            if distance <= connection_radius && node_ref.connections.len() < max_connections {
                potential_connections.push((*node_ref.key(), distance));
            }
        }

        // Sort by distance and connect to closest nodes
        potential_connections.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        for (target_node_id, _) in potential_connections.iter().take(3) {
            // Create bidirectional connection
            if let Some(mut new_node) = self.network_nodes.get_mut(&new_node_id) {
                if !new_node.connections.contains(target_node_id) {
                    new_node.connections.push(*target_node_id);
                }
            }

            if let Some(mut target_node) = self.network_nodes.get_mut(target_node_id) {
                if !target_node.connections.contains(&new_node_id) {
                    target_node.connections.push(new_node_id);
                }
            }

            debug!("🔗 Connected nodes {} ↔ {}", new_node_id, target_node_id);
        }

        Ok(())
    }

    /// Update network topology map for efficient routing
    async fn update_topology_map(&mut self) {
        self.topology_map.clear();

        for node_ref in self.network_nodes.iter() {
            self.topology_map
                .insert(*node_ref.key(), node_ref.connections.clone());
        }

        // Update network metrics
        self.network_metrics.total_nodes = self.network_nodes.len();
        self.network_metrics.active_connections = self
            .topology_map
            .values()
            .map(|connections| connections.len())
            .sum::<usize>()
            / 2; // Divide by 2 since connections are bidirectional

        // Calculate network coverage (how well-connected the network is)
        if self.network_nodes.len() > 1 {
            let max_possible_connections =
                self.network_nodes.len() * (self.network_nodes.len() - 1) / 2;
            self.network_metrics.network_coverage =
                self.network_metrics.active_connections as f64 / max_possible_connections as f64;
        }
    }

    /// Produce spores for network expansion
    pub async fn produce_spores(
        &mut self,
        source_node_id: Uuid,
        target_pairs: Vec<String>,
    ) -> Result<Vec<Uuid>, OrganismError> {
        let source_node = self
            .network_nodes
            .get(&source_node_id)
            .ok_or_else(|| OrganismError::InfectionFailed("Source node not found".to_string()))?;

        // Check if source node can produce spores
        if source_node.nutrient_level < 20.0 {
            return Err(OrganismError::ResourceExhausted(
                "Insufficient nutrients for spore production".to_string(),
            ));
        }

        let mut produced_spores = Vec::new();

        for target_pair in target_pairs {
            if fastrand::f64() < self.config.spore_production_rate {
                let spore = MycelialSpore {
                    spore_id: Uuid::new_v4(),
                    origin_node: source_node_id,
                    target_pair,
                    germination_time: Utc::now() + chrono::Duration::minutes(30), // 30 min germination
                    genetic_material: self.base.genetics.clone(),
                    specialization_preference: match fastrand::u32(0..5) {
                        0 => NodeSpecialization::Scout,
                        1 => NodeSpecialization::Extractor,
                        2 => NodeSpecialization::Communicator,
                        3 => NodeSpecialization::Reproducer,
                        4 => NodeSpecialization::Defender,
                        _ => NodeSpecialization::Scout,
                    },
                    viability: 0.8 + fastrand::f64() * 0.2, // 80-100% viability
                };

                produced_spores.push(spore.spore_id);
                self.active_spores.push(spore);
            }
        }

        // Consume nutrients for spore production
        if let Some(mut source_node_mut) = self.network_nodes.get_mut(&source_node_id) {
            source_node_mut.nutrient_level -= produced_spores.len() as f64 * 5.0;
            source_node_mut.health_status = if produced_spores.len() > 0 {
                NodeHealthStatus::Reproducing
            } else {
                NodeHealthStatus::Healthy
            };
        }

        info!(
            "🌱 Produced {} spores from node {}",
            produced_spores.len(),
            source_node_id
        );

        Ok(produced_spores)
    }

    /// Process spore germination and create new nodes
    pub async fn process_spore_germination(&mut self) -> Result<Vec<Uuid>, OrganismError> {
        let now = Utc::now();
        let mut germinated_spores = Vec::new();
        let mut new_nodes = Vec::new();

        // Collect spores that are ready for germination
        let ready_spores: Vec<_> = self
            .active_spores
            .iter()
            .filter(|spore| now >= spore.germination_time && spore.viability > 0.6)
            .cloned()
            .collect();

        // Process each ready spore
        for spore in ready_spores {
            // Germinate spore into new node
            let position = (
                fastrand::f64() * 20.0 - 10.0, // Random position in -10 to +10 range
                fastrand::f64() * 20.0 - 10.0,
            );

            match self
                .create_node(
                    spore.target_pair.clone(),
                    position,
                    spore.specialization_preference.clone(),
                )
                .await
            {
                Ok(node_id) => {
                    new_nodes.push(node_id);
                    germinated_spores.push(spore.spore_id);

                    // Set genetics from spore
                    if let Some(mut new_node) = self.network_nodes.get_mut(&node_id) {
                        // Inherit some characteristics from parent
                        new_node.nutrient_level = 5.0; // Start with lower nutrients
                    }
                }
                Err(_) => {
                    // Germination failed, spore dies
                    germinated_spores.push(spore.spore_id);
                }
            }
        }

        // Remove processed spores
        self.active_spores
            .retain(|spore| !germinated_spores.contains(&spore.spore_id));

        if !new_nodes.is_empty() {
            info!("🌿 Germinated {} spores into new nodes", new_nodes.len());
        }

        Ok(new_nodes)
    }

    /// Send information packet through the network
    pub async fn send_packet(
        &mut self,
        source_node: Uuid,
        target_nodes: Vec<Uuid>,
        packet_type: PacketType,
        data: serde_json::Value,
        priority: PacketPriority,
    ) -> Result<Uuid, OrganismError> {
        let priority_clone = priority.clone();
        let packet = NetworkPacket {
            packet_id: Uuid::new_v4(),
            source_node,
            target_nodes,
            packet_type,
            data,
            hop_count: 0,
            timestamp: Utc::now(),
            priority,
        };

        self.packet_queue.push(packet.clone());

        // Sort packet queue by priority
        self.packet_queue
            .sort_by(|a, b| b.priority.cmp(&a.priority));

        debug!(
            "📦 Queued packet {} from node {} with priority {:?}",
            packet.packet_id, source_node, priority_clone
        );

        Ok(packet.packet_id)
    }

    /// Process network packets and route them through the network
    pub async fn process_network_packets(&mut self) -> Result<usize, OrganismError> {
        let mut processed_count = 0;
        let max_packets_per_cycle = 20; // Limit processing to prevent overwhelming

        // Process up to max_packets_per_cycle packets
        let packets_to_process = self.packet_queue.len().min(max_packets_per_cycle);
        let mut packets = self
            .packet_queue
            .drain(0..packets_to_process)
            .collect::<Vec<_>>();

        for mut packet in packets {
            // Route packet through network
            if let Err(e) = self.route_packet(&mut packet).await {
                warn!("Failed to route packet {}: {}", packet.packet_id, e);
            } else {
                processed_count += 1;
            }
        }

        // Update network metrics based on packets processed
        self.network_metrics.information_flow_rate = self.network_metrics.information_flow_rate
            * 0.9
            + (processed_count as f64 / self.config.propagation_speed) * 0.1;

        Ok(processed_count)
    }

    /// Route a packet through the network to its targets
    async fn route_packet(&mut self, packet: &mut NetworkPacket) -> Result<(), OrganismError> {
        packet.hop_count += 1;

        // Prevent infinite loops
        if packet.hop_count > 20 {
            return Err(OrganismError::InfectionFailed(
                "Packet hop limit exceeded".to_string(),
            ));
        }

        // Find shortest paths to target nodes using simplified routing
        for target_node in &packet.target_nodes {
            if let Some(path) = self.find_shortest_path(packet.source_node, *target_node) {
                // Deliver packet data to target node
                if let Some(mut target_node_ref) = self.network_nodes.get_mut(target_node) {
                    let key = format!("packet_{}", packet.packet_id);
                    target_node_ref
                        .information_cache
                        .insert(key, packet.data.clone());
                    target_node_ref.last_activity = Utc::now();

                    debug!(
                        "📬 Delivered packet {} to node {} via {} hops",
                        packet.packet_id,
                        target_node,
                        path.len()
                    );
                }
            }
        }

        Ok(())
    }

    /// Find shortest path between two nodes using BFS
    fn find_shortest_path(&self, start: Uuid, end: Uuid) -> Option<Vec<Uuid>> {
        use std::collections::VecDeque;

        if start == end {
            return Some(vec![start]);
        }

        let mut queue = VecDeque::new();
        let mut visited = HashSet::new();
        let mut parent = HashMap::new();

        queue.push_back(start);
        visited.insert(start);

        while let Some(current) = queue.pop_front() {
            if let Some(connections) = self.topology_map.get(&current) {
                for &neighbor in connections {
                    if !visited.contains(&neighbor) {
                        visited.insert(neighbor);
                        parent.insert(neighbor, current);
                        queue.push_back(neighbor);

                        if neighbor == end {
                            // Reconstruct path
                            let mut path = Vec::new();
                            let mut node = end;

                            while let Some(&p) = parent.get(&node) {
                                path.push(node);
                                node = p;
                            }
                            path.push(start);
                            path.reverse();
                            return Some(path);
                        }
                    }
                }
            }
        }

        None // No path found
    }

    /// Share resources across the network
    pub async fn share_resources(&mut self) -> Result<f64, OrganismError> {
        let total_nutrients: f64 = self
            .network_nodes
            .iter()
            .map(|node_ref| node_ref.nutrient_level)
            .sum();

        let node_count = self.network_nodes.len() as f64;
        if node_count == 0.0 {
            return Ok(0.0);
        }

        let average_nutrients = total_nutrients / node_count;
        let sharing_efficiency = self.config.resource_sharing_efficiency;

        // Redistribute nutrients towards average
        for mut node_ref in self.network_nodes.iter_mut() {
            let current_level = node_ref.nutrient_level;
            let adjustment = (average_nutrients - current_level) * sharing_efficiency * 0.1;
            node_ref.nutrient_level += adjustment;
            node_ref.last_activity = Utc::now();
        }

        // Update shared resource pool
        self.shared_resources = total_nutrients * 0.1; // 10% goes to shared pool

        // Update resource distribution efficiency
        let variance = self
            .network_nodes
            .iter()
            .map(|node_ref| (node_ref.nutrient_level - average_nutrients).powi(2))
            .sum::<f64>()
            / node_count;

        self.network_metrics.resource_distribution_efficiency =
            1.0 / (1.0 + variance / average_nutrients.max(1.0));

        debug!(
            "🔄 Shared resources across network - avg nutrients: {:.2}, efficiency: {:.2}",
            average_nutrients, self.network_metrics.resource_distribution_efficiency
        );

        Ok(total_nutrients)
    }

    /// Get current network status
    pub fn get_network_status(&self) -> MycelialNetworkStatus {
        let active_nodes = self
            .network_nodes
            .iter()
            .filter(|node_ref| matches!(node_ref.health_status, NodeHealthStatus::Healthy))
            .count();

        let total_nutrients = self
            .network_nodes
            .iter()
            .map(|node_ref| node_ref.nutrient_level)
            .sum();

        MycelialNetworkStatus {
            total_nodes: self.network_nodes.len(),
            active_nodes,
            active_spores: self.active_spores.len(),
            total_nutrients,
            shared_resources: self.shared_resources,
            network_coverage: self.network_metrics.network_coverage,
            information_flow_rate: self.network_metrics.information_flow_rate,
            resource_sharing_efficiency: self.network_metrics.resource_distribution_efficiency,
            inter_network_connections: self.inter_network_connections.len(),
            packet_queue_size: self.packet_queue.len(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MycelialNetworkStatus {
    pub total_nodes: usize,
    pub active_nodes: usize,
    pub active_spores: usize,
    pub total_nutrients: f64,
    pub shared_resources: f64,
    pub network_coverage: f64,
    pub information_flow_rate: f64,
    pub resource_sharing_efficiency: f64,
    pub inter_network_connections: usize,
    pub packet_queue_size: usize,
}

#[async_trait]
impl ParasiticOrganism for MycelialNetworkOrganism {
    fn id(&self) -> Uuid {
        self.base.id
    }

    fn organism_type(&self) -> &'static str {
        "mycelial_network"
    }

    fn fitness(&self) -> f64 {
        let base_fitness = self.base.fitness;
        let network_bonus = (self.network_nodes.len() as f64 / 20.0).min(0.4);
        let coverage_bonus = self.network_metrics.network_coverage * 0.3;
        let efficiency_bonus = self.network_metrics.resource_distribution_efficiency * 0.2;
        let flow_bonus = (self.network_metrics.information_flow_rate / 10.0).min(0.1);

        (base_fitness + network_bonus + coverage_bonus + efficiency_bonus + flow_bonus).min(1.0)
    }

    fn calculate_infection_strength(&self, vulnerability: f64) -> f64 {
        let base_strength = self.base.calculate_base_infection_strength(vulnerability);

        // Network size multiplier
        let network_multiplier = 1.0 + (self.network_nodes.len() as f64 / 50.0).min(0.8);

        // Resource sharing enhances strength
        let resource_multiplier = 1.0 + self.network_metrics.resource_distribution_efficiency * 0.4;

        // Information flow provides coordination bonus
        let coordination_multiplier =
            1.0 + (self.network_metrics.information_flow_rate / 20.0).min(0.3);

        base_strength * network_multiplier * resource_multiplier * coordination_multiplier
    }

    async fn infect_pair(
        &self,
        pair_id: &str,
        vulnerability: f64,
    ) -> Result<InfectionResult, OrganismError> {
        // Mycelial networks can establish with lower vulnerability due to distributed approach
        if vulnerability < 0.25 {
            return Err(OrganismError::UnsuitableConditions(format!(
                "Vulnerability {:.3} insufficient for network establishment (need >0.25)",
                vulnerability
            )));
        }

        let infection_strength = self.calculate_infection_strength(vulnerability);

        // Very long infections due to persistent network nature
        let estimated_duration = (28800.0 * (3.0 - vulnerability)) as u64; // 8-24 hours

        let network_overhead = self.network_nodes.len() as f64 * 0.5;

        Ok(InfectionResult {
            success: true,
            infection_id: Uuid::new_v4(),
            initial_profit: infection_strength * 100.0, // Moderate initial profit, scales with network
            estimated_duration,
            resource_usage: ResourceMetrics {
                cpu_usage: 25.0 + network_overhead,
                memory_mb: 80.0 + self.network_nodes.len() as f64 * 2.0,
                network_bandwidth_kbps: 512.0 + self.network_metrics.information_flow_rate * 10.0,
                api_calls_per_second: 12.0 + self.network_nodes.len() as f64 * 0.5,
                latency_overhead_ns: 80_000 + (network_overhead * 1000.0) as u64,
            },
        })
    }

    async fn adapt(&mut self, feedback: AdaptationFeedback) -> Result<(), OrganismError> {
        self.base.update_fitness(feedback.performance_score);

        // Adapt network parameters based on performance
        if feedback.success_rate > 0.8 {
            // Successful network - increase expansion
            self.config.spore_production_rate = (self.config.spore_production_rate * 1.05).min(0.4);
            self.config.propagation_speed = (self.config.propagation_speed * 1.02).min(1.0);
        } else if feedback.success_rate < 0.5 {
            // Poor performance - increase resilience and resource sharing
            self.config.network_resilience = (self.config.network_resilience * 1.03).min(1.0);
            self.config.resource_sharing_efficiency =
                (self.config.resource_sharing_efficiency * 1.02).min(1.0);
        }

        // Adapt based on resource availability
        if feedback.profit_generated > 50.0 {
            // Abundant resources - can afford more nodes
            if self.config.max_network_nodes < 100 {
                self.config.max_network_nodes += 5;
            }
        } else if feedback.profit_generated < 10.0 {
            // Resource scarcity - improve efficiency
            self.config.nutrient_extraction_rate =
                (self.config.nutrient_extraction_rate * 1.05).min(1.0);
        }

        // Update shared resource pool based on profit
        self.shared_resources += feedback.profit_generated * 0.1;

        // Share resources after adaptation
        let _shared = self.share_resources().await?;

        Ok(())
    }

    fn mutate(&mut self, rate: f64) {
        self.base.genetics.mutate(rate);

        use rand::Rng;
        let mut rng = rand::thread_rng();

        if rng.gen::<f64>() < rate {
            self.config.propagation_speed =
                (self.config.propagation_speed + rng.gen_range(-0.05..0.05)).clamp(0.5, 1.0);
        }
        if rng.gen::<f64>() < rate {
            self.config.resource_sharing_efficiency = (self.config.resource_sharing_efficiency
                + rng.gen_range(-0.05..0.05))
            .clamp(0.5, 1.0);
        }
        if rng.gen::<f64>() < rate {
            self.config.network_resilience =
                (self.config.network_resilience + rng.gen_range(-0.05..0.05)).clamp(0.5, 1.0);
        }
        if rng.gen::<f64>() < rate {
            self.config.spore_production_rate =
                (self.config.spore_production_rate + rng.gen_range(-0.03..0.03)).clamp(0.05, 0.5);
        }
        if rng.gen::<f64>() < rate {
            let max_nodes_change = rng.gen_range(-5..=5);
            self.config.max_network_nodes =
                ((self.config.max_network_nodes as i32) + max_nodes_change).max(10) as usize;
        }
    }

    fn crossover(
        &self,
        other: &dyn ParasiticOrganism,
    ) -> Result<Box<dyn ParasiticOrganism + Send + Sync>, OrganismError> {
        if other.organism_type() != "mycelial_network" {
            return Err(OrganismError::CrossoverFailed(
                "Can only crossover with same organism type".to_string(),
            ));
        }

        let mut offspring = MycelialNetworkOrganism::new();
        offspring.base.genetics = self.base.genetics.crossover(&other.get_genetics());

        // Average the network-specific configurations
        offspring.config.propagation_speed =
            (self.config.propagation_speed + self.config.propagation_speed) / 2.0;
        offspring.config.resource_sharing_efficiency = (self.config.resource_sharing_efficiency
            + self.config.resource_sharing_efficiency)
            / 2.0;
        offspring.config.network_resilience =
            (self.config.network_resilience + self.config.network_resilience) / 2.0;
        offspring.config.spore_production_rate =
            (self.config.spore_production_rate + self.config.spore_production_rate) / 2.0;

        Ok(Box::new(offspring))
    }

    fn get_genetics(&self) -> OrganismGenetics {
        self.base.genetics.clone()
    }

    fn set_genetics(&mut self, genetics: OrganismGenetics) {
        self.base.genetics = genetics;
    }

    fn should_terminate(&self) -> bool {
        // Terminate if network has collapsed or resources are exhausted
        let network_collapsed = self.network_nodes.len() < 2 && self.active_spores.is_empty();
        let resources_exhausted = self.shared_resources < 1.0
            && self
                .network_nodes
                .iter()
                .all(|node_ref| node_ref.nutrient_level < 5.0);

        network_collapsed || resources_exhausted || self.base.should_terminate_base()
    }

    fn resource_consumption(&self) -> ResourceMetrics {
        let node_cost = self.network_nodes.len() as f64 * 1.2;
        let packet_cost = self.packet_queue.len() as f64 * 0.3;
        let spore_cost = self.active_spores.len() as f64 * 0.8;
        let network_cost = self.network_metrics.information_flow_rate * 2.0;

        ResourceMetrics {
            cpu_usage: 20.0 + node_cost + packet_cost + network_cost,
            memory_mb: 60.0 + node_cost * 1.5 + spore_cost,
            network_bandwidth_kbps: 400.0 + network_cost * 20.0,
            api_calls_per_second: 10.0 + self.network_nodes.len() as f64 * 0.8,
            latency_overhead_ns: 90_000 + (node_cost * 1500.0) as u64,
        }
    }

    fn get_strategy_params(&self) -> HashMap<String, f64> {
        let mut params = HashMap::new();
        params.insert(
            "max_network_nodes".to_string(),
            self.config.max_network_nodes as f64,
        );
        params.insert(
            "propagation_speed".to_string(),
            self.config.propagation_speed,
        );
        params.insert(
            "resource_sharing_efficiency".to_string(),
            self.config.resource_sharing_efficiency,
        );
        params.insert(
            "network_resilience".to_string(),
            self.config.network_resilience,
        );
        params.insert(
            "spore_production_rate".to_string(),
            self.config.spore_production_rate,
        );
        params.insert("current_nodes".to_string(), self.network_nodes.len() as f64);
        params.insert("active_spores".to_string(), self.active_spores.len() as f64);
        params.insert("shared_resources".to_string(), self.shared_resources);
        params.insert(
            "network_coverage".to_string(),
            self.network_metrics.network_coverage,
        );
        params.insert(
            "information_flow_rate".to_string(),
            self.network_metrics.information_flow_rate,
        );
        params
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mycelial_network_creation() {
        let network = MycelialNetworkOrganism::new();
        assert_eq!(network.organism_type(), "mycelial_network");
        assert!(network.network_nodes.is_empty());
        assert!(network.active_spores.is_empty());
    }

    #[tokio::test]
    async fn test_node_creation() {
        let mut network = MycelialNetworkOrganism::new();

        let node_id = network
            .create_node("BTCUSD".to_string(), (0.0, 0.0), NodeSpecialization::Scout)
            .await
            .unwrap();

        assert!(!network.network_nodes.is_empty());
        assert!(network.network_nodes.contains_key(&node_id));
    }

    #[tokio::test]
    async fn test_spore_production() {
        let mut network = MycelialNetworkOrganism::new();

        // Create a source node first
        let node_id = network
            .create_node(
                "BTCUSD".to_string(),
                (0.0, 0.0),
                NodeSpecialization::Reproducer,
            )
            .await
            .unwrap();

        // Set sufficient nutrients
        if let Some(mut node) = network.network_nodes.get_mut(&node_id) {
            node.nutrient_level = 50.0;
        }

        let spores = network
            .produce_spores(node_id, vec!["ETHUSD".to_string()])
            .await
            .unwrap();
        // May produce 0-1 spores based on probability
        assert!(spores.len() <= 1);
    }

    #[tokio::test]
    async fn test_resource_sharing() {
        let mut network = MycelialNetworkOrganism::new();

        // Create multiple nodes with different nutrient levels
        let node1 = network
            .create_node(
                "BTCUSD".to_string(),
                (0.0, 0.0),
                NodeSpecialization::Extractor,
            )
            .await
            .unwrap();
        let node2 = network
            .create_node(
                "ETHUSD".to_string(),
                (1.0, 0.0),
                NodeSpecialization::Extractor,
            )
            .await
            .unwrap();

        // Set different nutrient levels
        if let Some(mut node) = network.network_nodes.get_mut(&node1) {
            node.nutrient_level = 100.0;
        }
        if let Some(mut node) = network.network_nodes.get_mut(&node2) {
            node.nutrient_level = 10.0;
        }

        let total_nutrients = network.share_resources().await.unwrap();
        assert!(total_nutrients > 0.0);

        // Nutrients should be more evenly distributed after sharing
        let node1_nutrients = network.network_nodes.get(&node1).unwrap().nutrient_level;
        let node2_nutrients = network.network_nodes.get(&node2).unwrap().nutrient_level;
        assert!((node1_nutrients - node2_nutrients).abs() < 90.0); // Should be closer than original 90 difference
    }
}
