//! # Topology Evolution Module - Phase 3: Language Evolution
//!
//! Implements dynamic network topology adaptation for the "Creating Language"
//! framework (Christiansen & Chater).
//!
//! ## Features
//! - Activity-dependent connection creation/deletion
//! - Synaptic pruning based on usage patterns
//! - Hyperbolic tessellation refinement
//! - Critical period modeling
//!
//! ## References
//! - Chechik et al. (1998) "Synaptic pruning in development"
//! - Turrigiano (2008) "The self-tuning neuron"
//! - Markram et al. (2015) "Reconstruction of neocortical microcircuitry"

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};

use crate::hyperbolic_snn::LorentzVec;

// ============================================================================
// Topology Evolution Configuration
// ============================================================================

/// Configuration for topology evolution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologyConfig {
    /// Minimum connection strength before pruning
    pub prune_threshold: f64,
    /// Maximum connection strength for potentiation
    pub max_weight: f64,
    /// Activity threshold for connection creation (correlation)
    pub creation_threshold: f64,
    /// Maximum hyperbolic distance for new connections
    pub max_connection_distance: f64,
    /// Pruning interval (in simulation time)
    pub prune_interval: f64,
    /// Connection creation interval
    pub creation_interval: f64,
    /// Critical period duration (after which plasticity reduces)
    pub critical_period: f64,
    /// Plasticity decay rate after critical period
    pub plasticity_decay: f64,
    /// Maximum connections per neuron
    pub max_connections_per_neuron: usize,
    /// Minimum connections per neuron
    pub min_connections_per_neuron: usize,
    /// Enable hyperbolic distance-based probability
    pub hyperbolic_preference: bool,
}

impl Default for TopologyConfig {
    fn default() -> Self {
        Self {
            prune_threshold: 0.01,
            max_weight: 1.0,
            creation_threshold: 0.5,
            max_connection_distance: 5.0,
            prune_interval: 1000.0,    // Every second
            creation_interval: 5000.0,  // Every 5 seconds
            critical_period: 100000.0,  // 100 seconds
            plasticity_decay: 0.1,
            max_connections_per_neuron: 100,
            min_connections_per_neuron: 5,
            hyperbolic_preference: true,
        }
    }
}

// ============================================================================
// Connection State
// ============================================================================

/// State of a synaptic connection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionState {
    /// Pre-synaptic neuron ID
    pub pre_id: u32,
    /// Post-synaptic neuron ID
    pub post_id: u32,
    /// Current weight
    pub weight: f64,
    /// Hyperbolic distance
    pub distance: f64,
    /// Creation time
    pub created_at: f64,
    /// Last activation time
    pub last_active: f64,
    /// Activation count
    pub activation_count: u64,
    /// Cumulative correlation
    pub correlation_sum: f64,
    /// Is marked for pruning
    pub marked_for_pruning: bool,
}

impl ConnectionState {
    /// Create new connection state
    pub fn new(pre_id: u32, post_id: u32, weight: f64, distance: f64, time: f64) -> Self {
        Self {
            pre_id,
            post_id,
            weight,
            distance,
            created_at: time,
            last_active: time,
            activation_count: 0,
            correlation_sum: 0.0,
            marked_for_pruning: false,
        }
    }

    /// Record activation
    pub fn activate(&mut self, time: f64, correlation: f64) {
        self.last_active = time;
        self.activation_count += 1;
        self.correlation_sum += correlation;
    }

    /// Get average correlation
    pub fn average_correlation(&self) -> f64 {
        if self.activation_count > 0 {
            self.correlation_sum / self.activation_count as f64
        } else {
            0.0
        }
    }

    /// Get time since last activation
    pub fn time_since_active(&self, current_time: f64) -> f64 {
        current_time - self.last_active
    }

    /// Get connection age
    pub fn age(&self, current_time: f64) -> f64 {
        current_time - self.created_at
    }
}

// ============================================================================
// Neuron State for Topology
// ============================================================================

/// Neuron state tracking for topology evolution
#[derive(Debug, Clone)]
pub struct NeuronTopologyState {
    /// Neuron ID
    pub id: u32,
    /// Position in hyperbolic space
    pub position: LorentzVec,
    /// Outgoing connection IDs
    pub outgoing: HashSet<u32>,
    /// Incoming connection IDs
    pub incoming: HashSet<u32>,
    /// Recent spike times for correlation
    pub spike_history: VecDeque<f64>,
    /// Maximum spike history
    max_history: usize,
    /// Activity level (exponential average)
    pub activity_level: f64,
    /// Last spike time
    pub last_spike: f64,
}

impl NeuronTopologyState {
    /// Create new neuron topology state
    pub fn new(id: u32, position: LorentzVec) -> Self {
        Self {
            id,
            position,
            outgoing: HashSet::new(),
            incoming: HashSet::new(),
            spike_history: VecDeque::with_capacity(100),
            max_history: 100,
            activity_level: 0.0,
            last_spike: f64::NEG_INFINITY,
        }
    }

    /// Record spike
    pub fn record_spike(&mut self, time: f64) {
        self.spike_history.push_back(time);
        while self.spike_history.len() > self.max_history {
            self.spike_history.pop_front();
        }
        self.last_spike = time;
        // Update activity with exponential average
        self.activity_level = self.activity_level * 0.99 + 0.01;
    }

    /// Decay activity
    pub fn decay_activity(&mut self, dt: f64, tau: f64) {
        self.activity_level *= (-dt / tau).exp();
    }

    /// Get connection count
    pub fn connection_count(&self) -> usize {
        self.outgoing.len() + self.incoming.len()
    }

    /// Compute spike correlation with another neuron
    pub fn spike_correlation(&self, other: &NeuronTopologyState, window: f64) -> f64 {
        let mut correlation = 0.0;
        let mut count = 0;

        for &t1 in &self.spike_history {
            for &t2 in &other.spike_history {
                let dt = (t1 - t2).abs();
                if dt < window {
                    // Gaussian-weighted correlation
                    correlation += (-dt * dt / (2.0 * window * window)).exp();
                    count += 1;
                }
            }
        }

        if count > 0 {
            correlation / count as f64
        } else {
            0.0
        }
    }
}

// ============================================================================
// Pruning Strategies
// ============================================================================

/// Synaptic pruning strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PruningStrategy {
    /// Prune connections below weight threshold
    WeightThreshold,
    /// Prune connections inactive for too long
    ActivityBased { inactivity_threshold: f64 },
    /// Prune weakest connections to maintain max count
    CompetitivePruning,
    /// Prune based on correlation (uncorrelated connections)
    CorrelationBased { correlation_threshold: f64 },
    /// Combined strategy
    Combined {
        weight_factor: f64,
        activity_factor: f64,
        correlation_factor: f64,
    },
}

impl Default for PruningStrategy {
    fn default() -> Self {
        Self::Combined {
            weight_factor: 0.4,
            activity_factor: 0.3,
            correlation_factor: 0.3,
        }
    }
}

// ============================================================================
// Connection Creation Strategies
// ============================================================================

/// Connection creation strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CreationStrategy {
    /// Create connections between correlated neurons
    CorrelationBased,
    /// Create connections based on hyperbolic distance
    DistanceBased,
    /// Random creation with distance-based probability
    StochasticHyperbolic { temperature: f64 },
    /// Create to maintain network connectivity
    ConnectivityPreserving,
}

impl Default for CreationStrategy {
    fn default() -> Self {
        Self::StochasticHyperbolic { temperature: 1.0 }
    }
}

// ============================================================================
// Topology Evolver
// ============================================================================

/// Main topology evolution engine
#[derive(Debug)]
pub struct TopologyEvolver {
    /// Configuration
    config: TopologyConfig,
    /// Neuron states
    neurons: HashMap<u32, NeuronTopologyState>,
    /// Connection states
    connections: HashMap<(u32, u32), ConnectionState>,
    /// Pruning strategy
    pruning_strategy: PruningStrategy,
    /// Creation strategy
    creation_strategy: CreationStrategy,
    /// Current simulation time
    current_time: f64,
    /// Last pruning time
    last_prune_time: f64,
    /// Last creation time
    last_creation_time: f64,
    /// Statistics
    stats: TopologyStats,
}

/// Topology evolution statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TopologyStats {
    /// Total connections created
    pub connections_created: u64,
    /// Total connections pruned
    pub connections_pruned: u64,
    /// Current connection count
    pub current_connections: usize,
    /// Average connection weight
    pub avg_weight: f64,
    /// Average connections per neuron
    pub avg_connections_per_neuron: f64,
    /// Network clustering coefficient
    pub clustering_coefficient: f64,
}

impl TopologyEvolver {
    /// Create new topology evolver
    pub fn new(config: TopologyConfig) -> Self {
        Self {
            config,
            neurons: HashMap::new(),
            connections: HashMap::new(),
            pruning_strategy: PruningStrategy::default(),
            creation_strategy: CreationStrategy::default(),
            current_time: 0.0,
            last_prune_time: 0.0,
            last_creation_time: 0.0,
            stats: TopologyStats::default(),
        }
    }

    /// Set pruning strategy
    pub fn set_pruning_strategy(&mut self, strategy: PruningStrategy) {
        self.pruning_strategy = strategy;
    }

    /// Set creation strategy
    pub fn set_creation_strategy(&mut self, strategy: CreationStrategy) {
        self.creation_strategy = strategy;
    }

    /// Add neuron to topology
    pub fn add_neuron(&mut self, id: u32, position: LorentzVec) {
        self.neurons.insert(id, NeuronTopologyState::new(id, position));
    }

    /// Add connection
    pub fn add_connection(&mut self, pre_id: u32, post_id: u32, weight: f64) -> bool {
        if !self.neurons.contains_key(&pre_id) || !self.neurons.contains_key(&post_id) {
            return false;
        }

        if pre_id == post_id {
            return false; // No self-connections
        }

        let distance = {
            let pre = &self.neurons[&pre_id];
            let post = &self.neurons[&post_id];
            pre.position.hyperbolic_distance(&post.position)
        };

        let key = (pre_id, post_id);
        if self.connections.contains_key(&key) {
            return false; // Connection already exists
        }

        let conn = ConnectionState::new(pre_id, post_id, weight, distance, self.current_time);
        self.connections.insert(key, conn);

        // Update neuron connection sets
        if let Some(pre) = self.neurons.get_mut(&pre_id) {
            pre.outgoing.insert(post_id);
        }
        if let Some(post) = self.neurons.get_mut(&post_id) {
            post.incoming.insert(pre_id);
        }

        self.stats.connections_created += 1;
        self.stats.current_connections = self.connections.len();
        true
    }

    /// Remove connection
    pub fn remove_connection(&mut self, pre_id: u32, post_id: u32) -> bool {
        let key = (pre_id, post_id);
        if self.connections.remove(&key).is_some() {
            // Update neuron connection sets
            if let Some(pre) = self.neurons.get_mut(&pre_id) {
                pre.outgoing.remove(&post_id);
            }
            if let Some(post) = self.neurons.get_mut(&post_id) {
                post.incoming.remove(&pre_id);
            }
            self.stats.connections_pruned += 1;
            self.stats.current_connections = self.connections.len();
            true
        } else {
            false
        }
    }

    /// Record spike for neuron
    pub fn record_spike(&mut self, neuron_id: u32, time: f64) {
        if let Some(neuron) = self.neurons.get_mut(&neuron_id) {
            neuron.record_spike(time);
        }
    }

    /// Record connection activation
    pub fn record_activation(&mut self, pre_id: u32, post_id: u32, time: f64) {
        let key = (pre_id, post_id);

        // Compute correlation
        let correlation = if let (Some(pre), Some(post)) =
            (self.neurons.get(&pre_id), self.neurons.get(&post_id)) {
            pre.spike_correlation(post, 50.0) // 50ms window
        } else {
            0.0
        };

        if let Some(conn) = self.connections.get_mut(&key) {
            conn.activate(time, correlation);
        }
    }

    /// Update weight for a connection
    pub fn update_weight(&mut self, pre_id: u32, post_id: u32, new_weight: f64) {
        let key = (pre_id, post_id);
        if let Some(conn) = self.connections.get_mut(&key) {
            conn.weight = new_weight.clamp(0.0, self.config.max_weight);
        }
    }

    /// Get plasticity factor based on critical period
    fn plasticity_factor(&self) -> f64 {
        if self.current_time < self.config.critical_period {
            1.0
        } else {
            let excess = self.current_time - self.config.critical_period;
            (-excess * self.config.plasticity_decay / self.config.critical_period).exp()
        }
    }

    /// Compute pruning score for a connection
    fn pruning_score(&self, conn: &ConnectionState) -> f64 {
        match &self.pruning_strategy {
            PruningStrategy::WeightThreshold => {
                if conn.weight < self.config.prune_threshold {
                    1.0 // Prune
                } else {
                    0.0
                }
            }

            PruningStrategy::ActivityBased { inactivity_threshold } => {
                let inactive_time = conn.time_since_active(self.current_time);
                if inactive_time > *inactivity_threshold {
                    1.0
                } else {
                    inactive_time / inactivity_threshold
                }
            }

            PruningStrategy::CompetitivePruning => {
                // Lower weight = higher pruning score
                1.0 - conn.weight / self.config.max_weight
            }

            PruningStrategy::CorrelationBased { correlation_threshold } => {
                let corr = conn.average_correlation();
                if corr < *correlation_threshold {
                    1.0 - corr / correlation_threshold
                } else {
                    0.0
                }
            }

            PruningStrategy::Combined { weight_factor, activity_factor, correlation_factor } => {
                let weight_score = 1.0 - conn.weight / self.config.max_weight;
                let activity_score = conn.time_since_active(self.current_time) /
                    self.config.prune_interval;
                let correlation_score = 1.0 - conn.average_correlation();

                weight_factor * weight_score +
                activity_factor * activity_score.min(1.0) +
                correlation_factor * correlation_score
            }
        }
    }

    /// Execute pruning pass
    pub fn prune(&mut self) -> Vec<(u32, u32)> {
        let plasticity = self.plasticity_factor();
        let threshold = self.config.prune_threshold / plasticity.max(0.1);

        // Collect connections to prune
        let to_prune: Vec<(u32, u32)> = self.connections.iter()
            .filter(|(_, conn)| {
                let score = self.pruning_score(conn);
                score > 0.5 || conn.weight < threshold
            })
            .filter(|((pre_id, _post_id), _)| {
                // Don't prune if neuron would have too few connections
                if let Some(neuron) = self.neurons.get(pre_id) {
                    neuron.outgoing.len() > self.config.min_connections_per_neuron
                } else {
                    true
                }
            })
            .map(|(&key, _)| key)
            .collect();

        // Remove pruned connections
        for (pre, post) in &to_prune {
            self.remove_connection(*pre, *post);
        }

        self.last_prune_time = self.current_time;
        to_prune
    }

    /// Compute connection creation probability
    fn creation_probability(&self, pre: &NeuronTopologyState, post: &NeuronTopologyState) -> f64 {
        let distance = pre.position.hyperbolic_distance(&post.position);

        if distance > self.config.max_connection_distance {
            return 0.0;
        }

        match &self.creation_strategy {
            CreationStrategy::CorrelationBased => {
                let corr = pre.spike_correlation(post, 50.0);
                if corr > self.config.creation_threshold {
                    corr
                } else {
                    0.0
                }
            }

            CreationStrategy::DistanceBased => {
                // Probability decays with hyperbolic distance
                (-distance / 2.0).exp()
            }

            CreationStrategy::StochasticHyperbolic { temperature } => {
                let distance_factor = (-distance / (2.0 * temperature)).exp();
                let activity_factor = (pre.activity_level + post.activity_level) / 2.0;
                distance_factor * activity_factor.sqrt()
            }

            CreationStrategy::ConnectivityPreserving => {
                // Higher probability for neurons with few connections
                let pre_deficit = (self.config.min_connections_per_neuron as f64 -
                    pre.outgoing.len() as f64).max(0.0);
                let post_deficit = (self.config.min_connections_per_neuron as f64 -
                    post.incoming.len() as f64).max(0.0);

                let deficit_factor = (pre_deficit + post_deficit) /
                    (2.0 * self.config.min_connections_per_neuron as f64);
                let distance_factor = (-distance / 3.0).exp();

                deficit_factor.max(0.1) * distance_factor
            }
        }
    }

    /// Execute connection creation pass
    pub fn create_connections(&mut self, max_new: usize) -> Vec<(u32, u32)> {
        let plasticity = self.plasticity_factor();
        if plasticity < 0.01 {
            return Vec::new(); // Outside critical period
        }

        let mut created = Vec::new();
        let neuron_ids: Vec<u32> = self.neurons.keys().copied().collect();

        // Simple deterministic iteration for reproducibility
        'outer: for &pre_id in &neuron_ids {
            if created.len() >= max_new {
                break;
            }

            let pre_state = match self.neurons.get(&pre_id) {
                Some(n) => n.clone(),
                None => continue,
            };

            if pre_state.outgoing.len() >= self.config.max_connections_per_neuron {
                continue;
            }

            for &post_id in &neuron_ids {
                if created.len() >= max_new {
                    break 'outer;
                }

                if pre_id == post_id {
                    continue;
                }

                // Skip if connection exists
                if self.connections.contains_key(&(pre_id, post_id)) {
                    continue;
                }

                let post_state = match self.neurons.get(&post_id) {
                    Some(n) => n,
                    None => continue,
                };

                if post_state.incoming.len() >= self.config.max_connections_per_neuron {
                    continue;
                }

                let prob = self.creation_probability(&pre_state, post_state);

                // Use deterministic threshold based on neuron IDs for reproducibility
                let threshold = ((pre_id as f64 * 0.618033 + post_id as f64 * 0.381966) % 1.0)
                    * (1.0 / plasticity);

                if prob > threshold {
                    let initial_weight = self.config.max_weight * 0.1 * plasticity;
                    if self.add_connection(pre_id, post_id, initial_weight) {
                        created.push((pre_id, post_id));
                    }
                }
            }
        }

        self.last_creation_time = self.current_time;
        created
    }

    /// Main evolution step
    pub fn step(&mut self, time: f64) -> TopologyUpdate {
        self.current_time = time;

        // Decay neuron activities
        let dt = 1.0; // Assuming 1ms steps
        for neuron in self.neurons.values_mut() {
            neuron.decay_activity(dt, 1000.0);
        }

        let mut update = TopologyUpdate::default();

        // Check if it's time to prune
        if time - self.last_prune_time >= self.config.prune_interval {
            update.pruned = self.prune();
        }

        // Check if it's time to create
        if time - self.last_creation_time >= self.config.creation_interval {
            update.created = self.create_connections(10);
        }

        // Update statistics
        self.update_stats();

        update
    }

    /// Update statistics
    fn update_stats(&mut self) {
        self.stats.current_connections = self.connections.len();

        if !self.connections.is_empty() {
            self.stats.avg_weight = self.connections.values()
                .map(|c| c.weight)
                .sum::<f64>() / self.connections.len() as f64;
        }

        if !self.neurons.is_empty() {
            self.stats.avg_connections_per_neuron =
                self.connections.len() as f64 * 2.0 / self.neurons.len() as f64;

            // Compute clustering coefficient (simplified)
            let mut total_clustering = 0.0;
            for neuron in self.neurons.values() {
                let neighbors: HashSet<u32> = neuron.outgoing.union(&neuron.incoming)
                    .copied().collect();
                let k = neighbors.len();
                if k >= 2 {
                    let mut triangles = 0;
                    let neighbor_vec: Vec<u32> = neighbors.iter().copied().collect();
                    for i in 0..neighbor_vec.len() {
                        for j in (i + 1)..neighbor_vec.len() {
                            let n1 = neighbor_vec[i];
                            let n2 = neighbor_vec[j];
                            if self.connections.contains_key(&(n1, n2)) ||
                               self.connections.contains_key(&(n2, n1)) {
                                triangles += 1;
                            }
                        }
                    }
                    let possible = k * (k - 1) / 2;
                    if possible > 0 {
                        total_clustering += triangles as f64 / possible as f64;
                    }
                }
            }
            self.stats.clustering_coefficient = total_clustering / self.neurons.len() as f64;
        }
    }

    /// Get current statistics
    pub fn stats(&self) -> &TopologyStats {
        &self.stats
    }

    /// Get connection state
    pub fn get_connection(&self, pre_id: u32, post_id: u32) -> Option<&ConnectionState> {
        self.connections.get(&(pre_id, post_id))
    }

    /// Get all connections
    pub fn connections(&self) -> impl Iterator<Item = &ConnectionState> {
        self.connections.values()
    }

    /// Get neuron state
    pub fn get_neuron(&self, id: u32) -> Option<&NeuronTopologyState> {
        self.neurons.get(&id)
    }

    /// Get plasticity factor
    pub fn get_plasticity(&self) -> f64 {
        self.plasticity_factor()
    }
}

/// Result of topology evolution step
#[derive(Debug, Clone, Default)]
pub struct TopologyUpdate {
    /// Connections that were pruned
    pub pruned: Vec<(u32, u32)>,
    /// Connections that were created
    pub created: Vec<(u32, u32)>,
}

impl TopologyUpdate {
    /// Check if any changes occurred
    pub fn has_changes(&self) -> bool {
        !self.pruned.is_empty() || !self.created.is_empty()
    }
}

// ============================================================================
// Hyperbolic Tessellation Refinement
// ============================================================================

/// Adaptive tessellation refinement based on activity
#[derive(Debug)]
pub struct TessellationRefinement {
    /// Activity density per region
    activity_density: HashMap<(i32, i32, i32), f64>,
    /// Region resolution
    resolution: f64,
    /// Refinement threshold
    refinement_threshold: f64,
}

impl TessellationRefinement {
    /// Create new tessellation refinement manager
    pub fn new(resolution: f64, refinement_threshold: f64) -> Self {
        Self {
            activity_density: HashMap::new(),
            resolution,
            refinement_threshold,
        }
    }

    /// Get region key for position
    fn region_key(&self, pos: &LorentzVec) -> (i32, i32, i32) {
        (
            (pos.x / self.resolution).floor() as i32,
            (pos.y / self.resolution).floor() as i32,
            (pos.z / self.resolution).floor() as i32,
        )
    }

    /// Record activity at position
    pub fn record_activity(&mut self, position: &LorentzVec) {
        let key = self.region_key(position);
        *self.activity_density.entry(key).or_insert(0.0) += 1.0;
    }

    /// Decay all activity
    pub fn decay(&mut self, factor: f64) {
        for density in self.activity_density.values_mut() {
            *density *= factor;
        }
        // Remove near-zero entries
        self.activity_density.retain(|_, d| *d > 0.01);
    }

    /// Get regions that need refinement
    pub fn regions_to_refine(&self) -> Vec<(i32, i32, i32)> {
        self.activity_density.iter()
            .filter(|(_, &density)| density > self.refinement_threshold)
            .map(|(&key, _)| key)
            .collect()
    }

    /// Get activity density for a position
    pub fn get_density(&self, position: &LorentzVec) -> f64 {
        let key = self.region_key(position);
        self.activity_density.get(&key).copied().unwrap_or(0.0)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_connection_state() {
        let mut conn = ConnectionState::new(0, 1, 0.5, 1.0, 0.0);

        conn.activate(10.0, 0.8);
        assert_eq!(conn.activation_count, 1);
        assert!((conn.average_correlation() - 0.8).abs() < 0.01);

        assert!((conn.time_since_active(20.0) - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_topology_evolver_basic() {
        let config = TopologyConfig::default();
        let mut evolver = TopologyEvolver::new(config);

        // Add neurons
        evolver.add_neuron(0, LorentzVec::origin());
        evolver.add_neuron(1, LorentzVec::from_spatial(0.5, 0.0, 0.0));

        // Add connection
        assert!(evolver.add_connection(0, 1, 0.5));
        assert_eq!(evolver.stats().current_connections, 1);

        // Can't add duplicate
        assert!(!evolver.add_connection(0, 1, 0.5));

        // Remove connection
        assert!(evolver.remove_connection(0, 1));
        assert_eq!(evolver.stats().current_connections, 0);
    }

    #[test]
    fn test_pruning() {
        let mut config = TopologyConfig::default();
        config.prune_threshold = 0.1;
        config.prune_interval = 100.0;
        config.min_connections_per_neuron = 0;

        let mut evolver = TopologyEvolver::new(config);

        evolver.add_neuron(0, LorentzVec::origin());
        evolver.add_neuron(1, LorentzVec::from_spatial(0.5, 0.0, 0.0));

        // Add weak connection
        evolver.add_connection(0, 1, 0.05);

        // Should be pruned
        evolver.current_time = 100.0;
        let pruned = evolver.prune();
        assert_eq!(pruned.len(), 1);
    }

    #[test]
    fn test_spike_correlation() {
        let mut n1 = NeuronTopologyState::new(0, LorentzVec::origin());
        let mut n2 = NeuronTopologyState::new(1, LorentzVec::from_spatial(0.5, 0.0, 0.0));

        // Record correlated spikes
        n1.record_spike(10.0);
        n2.record_spike(11.0);
        n1.record_spike(20.0);
        n2.record_spike(21.0);

        let corr = n1.spike_correlation(&n2, 5.0);
        assert!(corr > 0.0);
    }

    #[test]
    fn test_plasticity_factor() {
        let mut config = TopologyConfig::default();
        config.critical_period = 100.0;
        config.plasticity_decay = 0.1;

        let mut evolver = TopologyEvolver::new(config);

        evolver.current_time = 0.0;
        assert!((evolver.plasticity_factor() - 1.0).abs() < 0.01);

        evolver.current_time = 200.0;
        assert!(evolver.plasticity_factor() < 1.0);
    }

    #[test]
    fn test_tessellation_refinement() {
        let mut refiner = TessellationRefinement::new(0.5, 10.0);

        let pos = LorentzVec::from_spatial(0.25, 0.25, 0.0);

        for _ in 0..15 {
            refiner.record_activity(&pos);
        }

        let regions = refiner.regions_to_refine();
        assert_eq!(regions.len(), 1);
    }
}
