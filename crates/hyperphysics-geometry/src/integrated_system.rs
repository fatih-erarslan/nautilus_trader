//! # Integrated Hyperbolic Cognitive System
//!
//! Central coordination layer that connects all SNN modules into a coherent
//! cognitive architecture with proper feedback loops and data flow.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    SOCCoordinator (global)                       │
//! │         Monitors criticality, modulates all subsystems           │
//! └─────────────────────────────────────────────────────────────────┘
//!                               │
//!       ┌───────────────────────┼───────────────────────┐
//!       │                       │                       │
//!       ▼                       ▼                       ▼
//! ┌───────────┐       ┌─────────────────┐       ┌───────────────┐
//! │   SNN     │──────▶│ ChunkProcessor  │──────▶│ EnactiveLayer │
//! │  spikes   │       │   temporal      │       │   beliefs     │
//! └───────────┘       │   chunking      │       │   policies    │
//!       │             └─────────────────┘       └───────────────┘
//!       │                       │                       │
//!       ▼                       ▼                       ▼
//! ┌───────────┐       ┌─────────────────┐       ┌───────────────┐
//! │   STDP    │◀─────▶│ TopologyEvolver │       │ Environment   │
//! │  weights  │       │   structure     │       │   feedback    │
//! └───────────┘       └─────────────────┘       └───────────────┘
//!       │                       │                       │
//!       └───────────────────────┴───────────────────────┘
//!                               │
//!                               ▼
//!                    ┌─────────────────┐
//!                    │  MarkovKernels  │
//!                    │ belief diffusion│
//!                    └─────────────────┘
//! ```
//!
//! ## References
//!
//! - Friston, K. (2010) "The free-energy principle: a unified brain theory?"
//! - Bak, P. et al. (1987) "Self-organized criticality"
//! - Christiansen & Chater (2016) "Creating Language"

use crate::hyperbolic_snn::{HyperbolicSNN, LorentzVec, SNNConfig, SOCStats};
use crate::chunk_processor::{ChunkProcessor, ChunkProcessorConfig, SpikeEvent};
use crate::enactive_layer::{EnactiveLayer, EnactiveConfig, Observation, Action};
use crate::stdp_learning::{ChunkAwareSTDP, STDPConfig};
use crate::topology_evolution::{TopologyEvolver, TopologyConfig, TopologyUpdate};
use crate::markov_kernels::{HyperbolicHeatKernel, TransitionOperator};
use crate::adversarial_lattice::AdversarialLattice;

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

// ============================================================================
// Global SOC Coordinator
// ============================================================================

/// Global SOC coordinator that monitors criticality across all subsystems
/// and modulates their behavior to maintain the system at the critical point.
#[derive(Debug, Clone)]
pub struct SOCCoordinator {
    /// Central SOC statistics
    pub stats: SOCStats,
    /// Target branching ratio (σ = 1 for criticality)
    pub sigma_target: f64,
    /// Current measured branching ratio
    pub sigma_measured: f64,
    /// Is system at criticality?
    pub is_critical: bool,
    /// Tolerance for criticality detection
    pub critical_tolerance: f64,
    /// Power-law exponent τ (should be ~1.5 at criticality)
    pub power_law_tau: f64,
    /// Avalanche size history for power-law fitting
    avalanche_sizes: VecDeque<usize>,
    /// Max history length
    max_history: usize,
    /// Modulation factors for subsystems
    pub modulation: SOCModulation,
}

/// SOC-based modulation factors for all subsystems
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SOCModulation {
    /// Learning rate multiplier (higher at criticality)
    pub learning_rate_factor: f64,
    /// Exploration factor (higher at criticality)
    pub exploration_factor: f64,
    /// Chunk window scale (adapts to avalanche statistics)
    pub chunk_window_scale: f64,
    /// Topology plasticity (higher when subcritical)
    pub topology_plasticity: f64,
    /// Attention sharpness (higher when supercritical)
    pub attention_sharpness: f64,
}

impl Default for SOCModulation {
    fn default() -> Self {
        Self {
            learning_rate_factor: 1.0,
            exploration_factor: 0.1,
            chunk_window_scale: 1.0,
            topology_plasticity: 1.0,
            attention_sharpness: 1.0,
        }
    }
}

impl SOCCoordinator {
    /// Create new SOC coordinator
    pub fn new(sigma_target: f64) -> Self {
        Self {
            stats: SOCStats::default(),
            sigma_target,
            sigma_measured: 1.0,
            is_critical: false,
            critical_tolerance: 0.1,
            power_law_tau: 1.5,
            avalanche_sizes: VecDeque::with_capacity(1000),
            max_history: 1000,
            modulation: SOCModulation::default(),
        }
    }

    /// Record avalanche from any subsystem
    pub fn record_avalanche(&mut self, size: usize, _duration: f64) {
        self.avalanche_sizes.push_back(size);
        if self.avalanche_sizes.len() > self.max_history {
            self.avalanche_sizes.pop_front();
        }

        // Update power-law exponent estimate using Hill estimator
        self.update_power_law();

        // Update stats
        self.stats.total_avalanches += 1;
        self.stats.largest_avalanche = self.stats.largest_avalanche.max(size);
    }

    /// Update branching ratio from spike statistics
    pub fn update_sigma(&mut self, initiating_spikes: u64, triggered_spikes: u64) {
        if initiating_spikes > 0 {
            self.sigma_measured = triggered_spikes as f64 / initiating_spikes as f64;
        }

        // Check criticality
        self.is_critical = (self.sigma_measured - self.sigma_target).abs() < self.critical_tolerance;

        // Update modulation factors based on criticality
        self.update_modulation();
    }

    /// Update power-law exponent using Hill estimator
    fn update_power_law(&mut self) {
        if self.avalanche_sizes.len() < 10 {
            return;
        }

        let mut sorted: Vec<usize> = self.avalanche_sizes.iter().copied().collect();
        sorted.sort_unstable();

        // Use top 10% for Hill estimator
        let k = (sorted.len() / 10).max(5);
        let x_k = sorted[sorted.len() - k] as f64;

        if x_k > 0.0 {
            let sum_log: f64 = sorted[sorted.len() - k..]
                .iter()
                .map(|&x| (x as f64 / x_k).ln())
                .sum();

            if sum_log > 0.0 {
                self.power_law_tau = 1.0 + k as f64 / sum_log;
            }
        }
    }

    /// Update modulation factors based on current SOC state
    fn update_modulation(&mut self) {
        let sigma_error = self.sigma_measured - self.sigma_target;

        // At criticality: maximize learning and exploration
        if self.is_critical {
            self.modulation.learning_rate_factor = 1.5;
            self.modulation.exploration_factor = 0.3;
            self.modulation.topology_plasticity = 1.0;
            self.modulation.attention_sharpness = 1.0;
        } else if sigma_error < 0.0 {
            // Subcritical: increase activity, more plasticity
            self.modulation.learning_rate_factor = 0.8;
            self.modulation.exploration_factor = 0.5;  // Explore more to find activity
            self.modulation.topology_plasticity = 1.5;  // More structural changes
            self.modulation.attention_sharpness = 0.8;  // Broader attention
        } else {
            // Supercritical: dampen activity
            self.modulation.learning_rate_factor = 0.5;
            self.modulation.exploration_factor = 0.1;
            self.modulation.topology_plasticity = 0.5;  // Less structural change
            self.modulation.attention_sharpness = 1.5;  // Sharper attention (more selective)
        }

        // Adjust chunk window based on avalanche statistics
        // Subcritical (small avalanches) → shorter chunks
        // Supercritical (large avalanches) → longer chunks
        if self.power_law_tau < 1.4 {
            self.modulation.chunk_window_scale = 0.8;
        } else if self.power_law_tau > 1.6 {
            self.modulation.chunk_window_scale = 1.2;
        } else {
            self.modulation.chunk_window_scale = 1.0;
        }
    }

    /// Get SOC factor for STDP modulation
    pub fn soc_factor(&self) -> f64 {
        1.0 + 0.5 * (self.sigma_target - self.sigma_measured)
    }

    /// Get current stats
    pub fn stats(&self) -> &SOCStats {
        &self.stats
    }
}

// ============================================================================
// Hyperbolic Environment
// ============================================================================

/// Simple hyperbolic environment for sensorimotor loop closure
#[derive(Debug, Clone)]
pub struct HyperbolicEnvironment {
    /// Current agent position
    pub agent_position: LorentzVec,
    /// Goal positions with rewards
    pub goals: Vec<(LorentzVec, f64)>,
    /// Obstacles with positions and radii
    pub obstacles: Vec<(LorentzVec, f64)>,
    /// Environment state
    pub state: Vec<f64>,
    /// Time step
    pub time: f64,
    /// Step size for movement
    pub step_size: f64,
}

impl HyperbolicEnvironment {
    /// Create new environment
    pub fn new(agent_start: LorentzVec) -> Self {
        Self {
            agent_position: agent_start,
            goals: Vec::new(),
            obstacles: Vec::new(),
            state: vec![0.0; 8],
            time: 0.0,
            step_size: 0.1,
        }
    }

    /// Add goal to environment
    pub fn add_goal(&mut self, position: LorentzVec, reward: f64) {
        self.goals.push((position, reward));
    }

    /// Add obstacle to environment
    pub fn add_obstacle(&mut self, position: LorentzVec, radius: f64) {
        self.obstacles.push((position, radius));
    }

    /// Generate observation from current state
    pub fn observe(&self) -> Observation {
        let mut features = vec![0.0; 8];

        // Distance and direction to nearest goal
        if let Some((nearest_goal, _)) = self.goals.iter()
            .min_by(|(a, _), (b, _)| {
                let da = self.agent_position.hyperbolic_distance(a);
                let db = self.agent_position.hyperbolic_distance(b);
                da.partial_cmp(&db).unwrap()
            })
        {
            let dist = self.agent_position.hyperbolic_distance(nearest_goal);
            features[0] = dist;
            features[1] = (nearest_goal.x - self.agent_position.x) / (dist + 0.1);
            features[2] = (nearest_goal.y - self.agent_position.y) / (dist + 0.1);
        }

        // Distance to nearest obstacle
        if let Some((nearest_obs, radius)) = self.obstacles.iter()
            .min_by(|(a, _), (b, _)| {
                let da = self.agent_position.hyperbolic_distance(a);
                let db = self.agent_position.hyperbolic_distance(b);
                da.partial_cmp(&db).unwrap()
            })
        {
            let dist = self.agent_position.hyperbolic_distance(nearest_obs);
            features[3] = (dist - radius).max(0.0);
            features[4] = (nearest_obs.x - self.agent_position.x) / (dist + 0.1);
            features[5] = (nearest_obs.y - self.agent_position.y) / (dist + 0.1);
        }

        // Current velocity estimate (from state)
        features[6] = self.state.get(0).copied().unwrap_or(0.0);
        features[7] = self.state.get(1).copied().unwrap_or(0.0);

        Observation {
            time: self.time,
            modality: crate::enactive_layer::Modality::Proprioceptive,
            position: self.agent_position,
            value: features.clone(),
            features,
            precision: 1.0,
        }
    }

    /// Execute action and return new observation + reward
    pub fn step(&mut self, action: &Action) -> (Observation, f64) {
        self.time += 1.0;

        // Compute movement direction
        let direction = self.agent_position.log_map(&action.target);
        let step_dist = self.step_size * action.intensity;

        // Move agent along geodesic
        let new_pos = self.agent_position.exp_map(&direction, step_dist);

        // Check obstacle collision
        let mut collision = false;
        for (obs_pos, radius) in &self.obstacles {
            if new_pos.hyperbolic_distance(obs_pos) < *radius {
                collision = true;
                break;
            }
        }

        // Compute reward
        let mut reward = -0.01; // Small cost for time

        if collision {
            reward -= 1.0; // Collision penalty
        } else {
            self.agent_position = new_pos;
        }

        // Check goal reaching
        for (goal_pos, goal_reward) in &self.goals {
            if self.agent_position.hyperbolic_distance(goal_pos) < 0.2 {
                reward += goal_reward;
            }
        }

        // Update state with velocity
        self.state[0] = direction.x;
        self.state[1] = direction.y;

        (self.observe(), reward)
    }

    /// Reset environment
    pub fn reset(&mut self, start: LorentzVec) {
        self.agent_position = start;
        self.time = 0.0;
        self.state = vec![0.0; 8];
    }
}

// ============================================================================
// Active Inference Layer
// ============================================================================

/// Active inference layer connecting free energy to SNN dynamics
#[derive(Debug, Clone)]
pub struct ActiveInferenceLayer {
    /// Beliefs as probability distribution over neurons
    pub beliefs: Vec<f64>,
    /// Prior probabilities (from hyperbolic structure)
    pub priors: Vec<f64>,
    /// Precision (inverse variance) per neuron
    pub precision: Vec<f64>,
    /// Current free energy
    pub free_energy: f64,
    /// Learning rate for belief updates
    pub learning_rate: f64,
    /// Heat kernel for belief diffusion
    kernel: HyperbolicHeatKernel,
}

impl ActiveInferenceLayer {
    /// Create new active inference layer
    pub fn new(num_neurons: usize, positions: &[LorentzVec]) -> Self {
        // Initialize priors based on hyperbolic distance from origin
        let priors: Vec<f64> = positions.iter()
            .map(|p| {
                let dist = p.hyperbolic_distance(&LorentzVec::origin());
                (-dist / 2.0).exp()  // Prior decreases with distance
            })
            .collect();

        // Normalize priors
        let prior_sum: f64 = priors.iter().sum();
        let priors: Vec<f64> = priors.iter().map(|p| p / prior_sum).collect();

        Self {
            beliefs: priors.clone(),
            priors,
            precision: vec![1.0; num_neurons],
            free_energy: 0.0,
            learning_rate: 0.1,
            kernel: HyperbolicHeatKernel::new(1.0),
        }
    }

    /// Compute free energy: F = KL[q||p] - E_q[log p(x|z)]
    pub fn compute_free_energy(&mut self, observation: &[f64]) -> f64 {
        // Complexity: KL divergence between beliefs and priors
        let mut complexity = 0.0;
        for i in 0..self.beliefs.len() {
            if self.beliefs[i] > 1e-10 && self.priors[i] > 1e-10 {
                complexity += self.beliefs[i] * (self.beliefs[i] / self.priors[i]).ln();
            }
        }

        // Accuracy: Expected log-likelihood under beliefs
        let mut accuracy = 0.0;
        let obs_len = observation.len().min(self.beliefs.len());
        for i in 0..obs_len {
            let pred = self.beliefs[i];
            accuracy -= self.precision[i] * (observation[i] - pred).powi(2);
        }

        self.free_energy = complexity - accuracy;
        self.free_energy
    }

    /// Update beliefs via free energy minimization
    pub fn minimize_free_energy(&mut self, observation: &[f64]) {
        let obs_len = observation.len().min(self.beliefs.len());

        // Gradient descent on free energy
        for i in 0..self.beliefs.len() {
            // Gradient of complexity term
            let grad_complexity = if self.priors[i] > 1e-10 {
                (self.beliefs[i] / self.priors[i]).ln() + 1.0
            } else {
                0.0
            };

            // Gradient of accuracy term
            let grad_accuracy = if i < obs_len {
                2.0 * self.precision[i] * (self.beliefs[i] - observation[i])
            } else {
                0.0
            };

            let grad = grad_complexity + grad_accuracy;
            self.beliefs[i] = (self.beliefs[i] - self.learning_rate * grad).clamp(1e-10, 1.0);
        }

        // Normalize beliefs
        let sum: f64 = self.beliefs.iter().sum();
        if sum > 1e-10 {
            for b in &mut self.beliefs {
                *b /= sum;
            }
        }
    }

    /// Diffuse beliefs using heat kernel on hyperbolic lattice
    pub fn diffuse_beliefs(&mut self, positions: &[LorentzVec], diffusion_time: f64) {
        let transition = TransitionOperator::new(positions.to_vec(), diffusion_time);
        self.beliefs = transition.apply(&self.beliefs);
    }

    /// Get threshold modulation for neurons based on beliefs
    /// High belief → lower threshold (easier to spike)
    pub fn get_threshold_modulation(&self) -> Vec<f64> {
        self.beliefs.iter()
            .map(|b| -10.0 * b)  // Reduce threshold proportional to belief
            .collect()
    }

    /// Update precision based on prediction errors
    pub fn update_precision(&mut self, prediction_errors: &[f64], learning_rate: f64) {
        for i in 0..self.precision.len().min(prediction_errors.len()) {
            // Precision increases when predictions are accurate
            let error_sq = prediction_errors[i].powi(2);
            self.precision[i] = (self.precision[i] * (1.0 - learning_rate)
                + learning_rate / (error_sq + 0.1)).clamp(0.1, 10.0);
        }
    }
}

// ============================================================================
// Integrated Cognitive System
// ============================================================================

/// Configuration for the integrated system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegratedSystemConfig {
    /// SNN configuration
    pub snn_config: SNNConfig,
    /// Chunk processor configuration
    pub chunk_config: ChunkProcessorConfig,
    /// Enactive layer configuration
    pub enactive_config: EnactiveConfig,
    /// STDP configuration
    pub stdp_config: STDPConfig,
    /// Topology configuration
    pub topology_config: TopologyConfig,
    /// Target branching ratio for SOC
    pub sigma_target: f64,
    /// Enable active inference
    pub enable_active_inference: bool,
}

impl Default for IntegratedSystemConfig {
    fn default() -> Self {
        Self {
            snn_config: SNNConfig::default(),
            chunk_config: ChunkProcessorConfig::default(),
            enactive_config: EnactiveConfig::default(),
            stdp_config: STDPConfig::default(),
            topology_config: TopologyConfig::default(),
            sigma_target: 1.0,
            enable_active_inference: true,
        }
    }
}

/// Integrated hyperbolic cognitive system
/// Coordinates all subsystems with proper feedback loops
pub struct IntegratedCognitiveSystem {
    /// Hyperbolic spiking neural network
    pub snn: HyperbolicSNN,
    /// Temporal chunk processor
    pub chunks: ChunkProcessor,
    /// Enactive cognition layer
    pub enactive: EnactiveLayer,
    /// STDP learning
    pub stdp: ChunkAwareSTDP,
    /// Topology evolution
    pub topology: TopologyEvolver,
    /// Global SOC coordinator
    pub soc: SOCCoordinator,
    /// Active inference layer (optional)
    pub active_inference: Option<ActiveInferenceLayer>,
    /// Environment (optional)
    pub environment: Option<HyperbolicEnvironment>,
    /// Current time
    pub time: f64,
    /// Statistics
    pub stats: IntegratedStats,
}

/// Statistics for integrated system
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IntegratedStats {
    /// Total simulation steps
    pub total_steps: u64,
    /// Total spikes generated
    pub total_spikes: u64,
    /// Total chunks formed
    pub total_chunks: u64,
    /// Total actions taken
    pub total_actions: u64,
    /// Total weight updates
    pub total_weight_updates: u64,
    /// Total topology changes
    pub total_topology_changes: u64,
    /// Average free energy
    pub avg_free_energy: f64,
    /// Average reward (if environment present)
    pub avg_reward: f64,
}

impl IntegratedCognitiveSystem {
    /// Create from adversarial lattice
    pub fn from_lattice(
        lattice: &AdversarialLattice,
        config: IntegratedSystemConfig,
    ) -> Result<Self, crate::GeometryError> {
        let snn = HyperbolicSNN::from_lattice(lattice, config.snn_config.clone())?;

        let positions: Vec<LorentzVec> = snn.neurons.iter()
            .map(|n| n.position)
            .collect();

        let active_inference = if config.enable_active_inference {
            Some(ActiveInferenceLayer::new(snn.neurons.len(), &positions))
        } else {
            None
        };

        // Initialize topology evolver with positions
        let mut topology = TopologyEvolver::new(config.topology_config.clone());
        for (i, pos) in positions.iter().enumerate() {
            topology.add_neuron(i as u32, *pos);
        }
        // Initialize connections from SNN synapses
        for synapse in &snn.synapses {
            topology.add_connection(synapse.pre_id as u32, synapse.post_id as u32, synapse.weight);
        }

        Ok(Self {
            snn,
            chunks: ChunkProcessor::new(config.chunk_config),
            enactive: EnactiveLayer::new(config.enactive_config),
            stdp: ChunkAwareSTDP::new(config.stdp_config),
            topology,
            soc: SOCCoordinator::new(config.sigma_target),
            active_inference,
            environment: None,
            time: 0.0,
            stats: IntegratedStats::default(),
        })
    }

    /// Set environment for sensorimotor loop
    pub fn set_environment(&mut self, env: HyperbolicEnvironment) {
        self.environment = Some(env);
    }

    /// Main simulation step - coordinates all subsystems
    pub fn step(&mut self, external_input: &[f64]) -> Vec<usize> {
        self.time += 1.0;
        self.stats.total_steps += 1;

        // 1. Apply SOC modulation to learning rates
        self.apply_soc_modulation();

        // 2. Apply active inference modulation to SNN
        if let Some(ref mut ai) = self.active_inference {
            let threshold_mod = ai.get_threshold_modulation();
            for (i, neuron) in self.snn.neurons.iter_mut().enumerate() {
                if i < threshold_mod.len() {
                    neuron.threshold = -55.0 + threshold_mod[i];
                }
            }
        }

        // 3. Run SNN step
        let spiked = self.snn.step_with_input(external_input);
        self.stats.total_spikes += spiked.len() as u64;

        // 4. Feed spikes to chunk processor
        for &neuron_id in &spiked {
            let neuron = &self.snn.neurons[neuron_id];
            let spike_event = SpikeEvent {
                time: self.time,
                neuron_id,
                position: neuron.position,
                amplitude: 1.0,
            };
            self.chunks.process_spike(spike_event);
        }

        // 5. Update SOC from SNN
        let soc_stats = self.snn.soc_monitor.stats();
        self.soc.update_sigma(
            soc_stats.total_initiating_spikes,
            soc_stats.total_triggered_spikes,
        );
        if let Some(size) = self.snn.soc_monitor.current_avalanche_size() {
            self.soc.record_avalanche(size, 1.0);
        }

        // 6. Process STDP for spiking neurons
        for &neuron_id in &spiked {
            let connections: Vec<(u32, f64, f64)> = self.snn.synapses.iter()
                .filter(|s| s.pre_id == neuron_id || s.post_id == neuron_id)
                .map(|s| (
                    if s.pre_id == neuron_id { s.post_id } else { s.pre_id } as u32,
                    s.weight,
                    s.distance as f64,
                ))
                .collect();

            self.stdp.on_spike(
                neuron_id as u32,
                self.time,
                true, // is_pre
                &connections,
            );
        }

        // 7. Flush STDP updates and apply to SNN + topology
        let weight_updates = self.stdp.flush_updates(self.time);
        for ((pre_id, post_id), delta_w) in weight_updates {
            // Apply to SNN
            if let Some(synapse) = self.snn.synapses.iter_mut()
                .find(|s| s.pre_id == pre_id as usize && s.post_id == post_id as usize)
            {
                synapse.weight = (synapse.weight + delta_w).clamp(0.0, 1.0);

                // Update topology evolver
                self.topology.update_weight(pre_id, post_id, synapse.weight);
            }
            self.stats.total_weight_updates += 1;
        }

        // 8. Run topology evolution step
        let TopologyUpdate { pruned, created } = self.topology.step(self.time);

        // Track changes for stats
        let num_pruned = pruned.len();
        let num_created = created.len();

        // Remove pruned synapses
        self.snn.synapses.retain(|s| {
            !pruned.contains(&(s.pre_id as u32, s.post_id as u32))
        });

        // Add new synapses
        for (pre_id, post_id) in created {
            if let Some(conn) = self.topology.get_connection(pre_id, post_id) {
                self.snn.add_synapse(
                    pre_id as usize,
                    post_id as usize,
                    0.1, // Initial weight
                    conn.distance as f32,
                );
            }
        }
        self.stats.total_topology_changes += (num_pruned + num_created) as u64;

        // 9. Update active inference
        if let Some(ref mut ai) = self.active_inference {
            // Create observation from spike rates
            let obs: Vec<f64> = self.snn.neurons.iter()
                .map(|n| if n.spiked_within(self.time, 10.0) { 1.0 } else { 0.0 })
                .collect();

            ai.minimize_free_energy(&obs);

            // Diffuse beliefs on hyperbolic lattice
            let positions: Vec<LorentzVec> = self.snn.neurons.iter()
                .map(|n| n.position)
                .collect();
            ai.diffuse_beliefs(&positions, 0.1);

            self.stats.avg_free_energy = 0.99 * self.stats.avg_free_energy
                + 0.01 * ai.free_energy;
        }

        // 10. Generate observation for enactive layer (from chunks)
        let chunk_stats = self.chunks.stats();
        self.stats.total_chunks = chunk_stats.chunks_formed as u64;

        // Create observation from latest chunks
        if let Some(chunk) = self.chunks.get_chunks(0).last() {
            let features = chunk.representation.temporal_signature.clone();
            let obs = Observation {
                time: self.time,
                modality: crate::enactive_layer::Modality::Proprioceptive,
                position: chunk.representation.centroid,
                value: features.clone(),
                features,
                precision: chunk.representation.confidence,
            };
            self.enactive.process_observation(obs);
        }

        // 11. Generate action if environment present
        if self.environment.is_some() {
            if let Some(action) = self.enactive.try_action() {
                let env = self.environment.as_mut().unwrap();
                let (outcome, reward) = env.step(&action);

                self.enactive.execute_action(&action, &outcome);

                self.stats.total_actions += 1;
                self.stats.avg_reward = 0.99 * self.stats.avg_reward + 0.01 * reward;
            }
        }

        spiked
    }

    /// Apply SOC modulation to all subsystems
    fn apply_soc_modulation(&mut self) {
        let mod_factors = &self.soc.modulation;

        // Modulate STDP learning rate
        self.stdp.set_learning_rate_factor(mod_factors.learning_rate_factor);

        // Modulate enactive exploration
        self.enactive.set_exploration(mod_factors.exploration_factor);

        // Modulate topology plasticity
        self.topology.modulate_plasticity(mod_factors.topology_plasticity);
    }

    /// Get comprehensive statistics
    pub fn stats(&self) -> IntegratedStats {
        self.stats.clone()
    }

    /// Get SOC coordinator state
    pub fn soc_state(&self) -> &SOCCoordinator {
        &self.soc
    }

    /// Check if system is at criticality
    pub fn is_critical(&self) -> bool {
        self.soc.is_critical
    }

    /// Get current free energy
    pub fn free_energy(&self) -> f64 {
        self.active_inference.as_ref()
            .map(|ai| ai.free_energy)
            .unwrap_or(0.0)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adversarial_lattice::DefenseTopology;
    use crate::enactive_layer::ActionType;

    #[test]
    fn test_soc_coordinator() {
        let mut soc = SOCCoordinator::new(1.0);

        // Record some avalanches
        for size in [1, 2, 3, 4, 5, 10, 15, 20] {
            soc.record_avalanche(size, 1.0);
        }

        // Update sigma - 105 triggered by 100 initiating = sigma 1.05
        soc.update_sigma(100, 105);
        assert!(soc.sigma_measured > 1.0);
        // 1.05 is within tolerance (0.1) of target (1.0), so is_critical = true
        assert!(soc.is_critical);

        // Test clearly supercritical case (outside tolerance)
        soc.update_sigma(100, 150);  // sigma = 1.5, far from target
        assert!(!soc.is_critical);
    }

    #[test]
    fn test_hyperbolic_environment() {
        let mut env = HyperbolicEnvironment::new(LorentzVec::origin());
        env.add_goal(LorentzVec::from_spatial(0.5, 0.0, 0.0), 1.0);
        env.add_obstacle(LorentzVec::from_spatial(-0.5, 0.0, 0.0), 0.1);

        let obs = env.observe();
        assert_eq!(obs.features.len(), 8);

        let action = Action {
            time: 0.0,
            target: LorentzVec::from_spatial(0.3, 0.0, 0.0),
            action_type: ActionType::Approach,
            intensity: 1.0,
            expected_outcome: vec![],
        };

        let (new_obs, reward) = env.step(&action);
        assert!(new_obs.time > obs.time);
        assert!(reward < 0.0); // Only time penalty, no goal reached yet
    }

    #[test]
    fn test_active_inference_layer() {
        let positions = vec![
            LorentzVec::origin(),
            LorentzVec::from_spatial(0.3, 0.0, 0.0),
            LorentzVec::from_spatial(0.0, 0.3, 0.0),
        ];

        let mut ai = ActiveInferenceLayer::new(3, &positions);

        // Compute free energy
        let obs = vec![0.5, 0.3, 0.2];
        let fe = ai.compute_free_energy(&obs);
        assert!(fe.is_finite());

        // Minimize free energy
        ai.minimize_free_energy(&obs);
        let fe2 = ai.compute_free_energy(&obs);
        assert!(fe2 <= fe); // Should decrease or stay same
    }

    #[test]
    fn test_integrated_system_creation() {
        let topology = DefenseTopology::maximum_connectivity(2);
        let lattice = AdversarialLattice::new(topology).unwrap();
        let config = IntegratedSystemConfig::default();

        let system = IntegratedCognitiveSystem::from_lattice(&lattice, config);
        assert!(system.is_ok());

        let system = system.unwrap();
        assert!(system.snn.neurons.len() > 0);
        assert!(system.active_inference.is_some());
    }

    #[test]
    fn test_integrated_step() {
        let topology = DefenseTopology::maximum_connectivity(1);
        let lattice = AdversarialLattice::new(topology).unwrap();
        let config = IntegratedSystemConfig::default();

        let mut system = IntegratedCognitiveSystem::from_lattice(&lattice, config).unwrap();

        // Add environment
        let mut env = HyperbolicEnvironment::new(LorentzVec::origin());
        env.add_goal(LorentzVec::from_spatial(0.5, 0.0, 0.0), 1.0);
        system.set_environment(env);

        // Run a few steps
        let input = vec![10.0; system.snn.neurons.len()];
        for _ in 0..10 {
            let _spikes = system.step(&input);
        }

        assert!(system.stats.total_steps == 10);
        assert!(system.time == 10.0);
    }
}
