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
// Phase 4 Integration: GlobalWorkspace ↔ PredictiveEnactive Bridge
// ============================================================================

use crate::global_workspace::{GlobalWorkspace, GlobalWorkspaceConfig, BroadcastEvent, WorkspaceContent, SpecialistType};
use crate::predictive_coding::{PredictiveEnactiveSystem, PredictiveCodingConfig, LevelError};
use crate::iit_phi::{PhiCalculator, PhiConfig, PhiResult};
use crate::bateson_ecology::{EcologicalMind, EcologyConfig, LearningLevel};

/// Bridge connecting Global Workspace broadcasts to Predictive Coding hierarchy
///
/// Implements the interface between:
/// - GWT conscious broadcast → top-down predictions
/// - Prediction errors → bottom-up specialist activation
#[derive(Debug)]
pub struct WorkspacePredictiveBridge {
    /// Coupling strength from workspace to predictive system
    pub broadcast_to_prediction_coupling: f64,
    /// Coupling strength from prediction errors to workspace
    pub error_to_activation_coupling: f64,
    /// Prediction error history for specialist modulation
    error_history: VecDeque<Vec<LevelError>>,
    /// Maximum history length
    max_history: usize,
    /// Specialist activation modulations from prediction errors
    pub specialist_modulations: Vec<f64>,
}

impl WorkspacePredictiveBridge {
    /// Create new bridge with coupling strengths
    pub fn new(broadcast_coupling: f64, error_coupling: f64, num_specialists: usize) -> Self {
        Self {
            broadcast_to_prediction_coupling: broadcast_coupling,
            error_to_activation_coupling: error_coupling,
            error_history: VecDeque::with_capacity(100),
            max_history: 100,
            specialist_modulations: vec![0.0; num_specialists],
        }
    }

    /// Process broadcast event: inject into predictive system's top level
    pub fn process_broadcast(
        &mut self,
        broadcast: &BroadcastEvent,
        predictive: &mut PredictiveEnactiveSystem,
    ) {
        // Broadcast content becomes top-level prediction target
        let features = broadcast.content.features.clone();
        let position = broadcast.content.position;

        // Modulate top level belief based on broadcast
        if let Some(top_belief) = predictive.belief.levels.last_mut() {
            // Inject broadcast position into belief (weighted by coupling)
            let direction = top_belief.position_mean.log_map(&position);
            top_belief.position_mean = top_belief.position_mean.exp_map(
                &direction,
                self.broadcast_to_prediction_coupling * broadcast.strength,
            );

            // Inject broadcast features into hidden state
            for i in 0..top_belief.hidden_state.len().min(features.len()) {
                let current = top_belief.hidden_state[i];
                let broadcast_val = features[i];
                top_belief.hidden_state[i] = current * (1.0 - self.broadcast_to_prediction_coupling)
                    + broadcast_val * self.broadcast_to_prediction_coupling * broadcast.strength;
            }

            // Increase precision at top level (broadcast = confident content)
            top_belief.hidden_precision *= 1.0 + 0.1 * broadcast.strength;
        }
    }

    /// Process prediction errors: modulate specialist activations
    pub fn process_errors(
        &mut self,
        errors: &[LevelError],
        workspace: &mut GlobalWorkspace,
    ) {
        // Store error history
        self.error_history.push_back(errors.to_vec());
        if self.error_history.len() > self.max_history {
            self.error_history.pop_front();
        }

        // Compute specialist modulations from error distribution
        // High errors at low levels → boost sensory specialists
        // High errors at high levels → boost cognitive specialists
        let num_specialists = workspace.specialists.len();
        self.specialist_modulations = vec![0.0; num_specialists];

        for (i, specialist) in workspace.specialists.iter_mut().enumerate() {
            let modulation = match specialist.module_type {
                SpecialistType::Sensory | SpecialistType::Motor => {
                    // Sensory/motor specialists activated by low-level errors
                    errors.first()
                        .map(|e| e.weighted_error * self.error_to_activation_coupling)
                        .unwrap_or(0.0)
                }
                SpecialistType::Memory | SpecialistType::Attention => {
                    // Memory/attention by mid-level errors
                    errors.get(1)
                        .map(|e| e.weighted_error * self.error_to_activation_coupling)
                        .unwrap_or(0.0)
                }
                SpecialistType::Language | SpecialistType::Spatial | SpecialistType::Temporal => {
                    // Higher cognitive by high-level errors
                    errors.last()
                        .map(|e| e.weighted_error * self.error_to_activation_coupling)
                        .unwrap_or(0.0)
                }
                _ => {
                    // Default: average across all levels
                    let avg_error: f64 = errors.iter().map(|e| e.weighted_error).sum::<f64>()
                        / errors.len().max(1) as f64;
                    avg_error * self.error_to_activation_coupling
                }
            };

            self.specialist_modulations[i] = modulation;

            // Apply modulation to specialist activation
            specialist.activation = (specialist.activation + modulation).clamp(0.0, 1.0);
        }
    }

    /// Get average prediction error for monitoring
    pub fn average_error(&self) -> f64 {
        if self.error_history.is_empty() {
            return 0.0;
        }

        let total: f64 = self.error_history.iter()
            .flat_map(|errors| errors.iter().map(|e| e.weighted_error))
            .sum();
        let count = self.error_history.iter()
            .map(|errors| errors.len())
            .sum::<usize>();

        if count > 0 { total / count as f64 } else { 0.0 }
    }
}

// ============================================================================
// Phase 4 Integration: PhiCalculator ↔ GlobalWorkspace Bridge
// ============================================================================

/// Bridge connecting IIT Φ computation to Global Workspace ignition
///
/// Based on the hypothesis that conscious content corresponds to
/// information that is both integrated (high Φ) and globally available (broadcast)
pub struct PhiWorkspaceBridge {
    /// Minimum Φ required for workspace ignition boost
    pub phi_ignition_threshold: f64,
    /// Coupling strength from Φ to ignition threshold
    pub phi_to_ignition_coupling: f64,
    /// Recent Φ values for averaging
    phi_history: VecDeque<f64>,
    /// Maximum history
    max_history: usize,
}

impl PhiWorkspaceBridge {
    pub fn new(phi_threshold: f64, coupling: f64) -> Self {
        Self {
            phi_ignition_threshold: phi_threshold,
            phi_to_ignition_coupling: coupling,
            phi_history: VecDeque::with_capacity(100),
            max_history: 100,
        }
    }

    /// Process Φ computation: modulate workspace ignition threshold
    pub fn process_phi(
        &mut self,
        phi_result: &PhiResult,
        workspace: &mut GlobalWorkspace,
    ) {
        let phi = phi_result.phi;
        self.phi_history.push_back(phi);
        if self.phi_history.len() > self.max_history {
            self.phi_history.pop_front();
        }

        // High Φ indicates integrated information → lower ignition threshold
        // This implements the hypothesis that integrated content more easily
        // achieves conscious access
        if phi > self.phi_ignition_threshold {
            let phi_excess = phi - self.phi_ignition_threshold;
            let threshold_reduction = phi_excess * self.phi_to_ignition_coupling;

            // Apply to workspace (would need accessor method in GlobalWorkspace)
            // For now, boost specialist activations uniformly
            for specialist in &mut workspace.specialists {
                specialist.activation *= 1.0 + threshold_reduction;
            }
        }
    }

    /// Identify which specialists contribute most to Φ
    pub fn phi_specialist_contribution(
        &self,
        phi_result: &PhiResult,
        workspace: &GlobalWorkspace,
    ) -> Vec<(usize, f64)> {
        // Map mechanism state to specialists based on position
        let mut contributions = Vec::new();

        // PhiResult has a single mechanism, not a vector
        // Find closest specialist based on mechanism elements
        for &element_idx in &phi_result.mechanism.elements {
            if let Some((specialist_id, _)) = workspace.specialists.iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    let dist_a = element_idx.abs_diff(a.id);
                    let dist_b = element_idx.abs_diff(b.id);
                    dist_a.cmp(&dist_b)
                })
            {
                // Contribution proportional to intrinsic causal power if available
                let icp = phi_result.intrinsic_causal_power
                    .as_ref()
                    .map(|icp| icp.icp_total)
                    .unwrap_or(1.0);
                contributions.push((specialist_id, icp));
            }
        }

        contributions
    }

    /// Average Φ over history
    pub fn average_phi(&self) -> f64 {
        if self.phi_history.is_empty() {
            return 0.0;
        }
        self.phi_history.iter().sum::<f64>() / self.phi_history.len() as f64
    }
}

// ============================================================================
// Phase 4 Integration: EcologicalMind ↔ Learning Bridge
// ============================================================================

/// Bridge connecting Bateson's Ecological Mind to learning systems
///
/// Implements multi-level learning where:
/// - Level 0: Direct stimulus-response (STDP)
/// - Level 1: Learning to learn (topology evolution)
/// - Level 2: Deutero-learning (meta-learning from double binds)
pub struct EcologyLearningBridge {
    /// Coupling from ecological learning to STDP
    pub ecology_to_stdp_coupling: f64,
    /// Coupling from ecological learning to topology
    pub ecology_to_topology_coupling: f64,
    /// Recent learning results
    learning_history: VecDeque<LearningLevel>,
    /// Maximum history
    max_history: usize,
}

impl EcologyLearningBridge {
    pub fn new(stdp_coupling: f64, topology_coupling: f64) -> Self {
        Self {
            ecology_to_stdp_coupling: stdp_coupling,
            ecology_to_topology_coupling: topology_coupling,
            learning_history: VecDeque::with_capacity(100),
            max_history: 100,
        }
    }

    /// Process ecological learning: modulate STDP and topology
    pub fn process_learning(
        &mut self,
        ecology: &EcologicalMind,
        stdp: &mut ChunkAwareSTDP,
        topology: &mut TopologyEvolver,
    ) {
        // Learning level strength based on current level
        let level_strength = match ecology.learning_level {
            LearningLevel::Zero => 1.0,  // Basic stimulus-response
            LearningLevel::One => 0.8,   // Context-dependent learning
            LearningLevel::Two => 0.6,   // Learning to learn
            LearningLevel::Three => 0.4, // Meta-learning
        };

        // Level 0/1 learning modulates STDP rates
        if matches!(ecology.learning_level, LearningLevel::Zero | LearningLevel::One) {
            stdp.set_learning_rate_factor(1.0 + self.ecology_to_stdp_coupling * level_strength);
        }

        // Level 1/2 learning modulates topology plasticity
        if matches!(ecology.learning_level, LearningLevel::One | LearningLevel::Two) {
            topology.modulate_plasticity(1.0 + self.ecology_to_topology_coupling * level_strength);
        }

        // Level 2/3 (deutero) affects both in complex ways
        if matches!(ecology.learning_level, LearningLevel::Two | LearningLevel::Three) {
            // High deutero-learning: system is learning how to learn
            // Increase both STDP and topology plasticity
            stdp.set_learning_rate_factor(1.5);
            topology.modulate_plasticity(1.5);
        }

        // Record learning level history
        self.learning_history.push_back(ecology.learning_level);
        if self.learning_history.len() > self.max_history {
            self.learning_history.pop_front();
        }
    }

    /// Detect if system is in a double bind (contradictory constraints)
    pub fn detect_double_bind(&self, ecology: &EcologicalMind) -> bool {
        ecology.deutero_stats.double_binds_encountered > 0
    }

    /// Get dominant learning level
    pub fn dominant_level(&self) -> Option<LearningLevel> {
        if self.learning_history.is_empty() {
            return None;
        }

        // Count occurrences
        let mut counts = std::collections::HashMap::new();
        for level in &self.learning_history {
            *counts.entry(*level).or_insert(0) += 1;
        }

        counts.into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(level, _)| level)
    }
}

// ============================================================================
// Unified Conscious Integration Hub
// ============================================================================

/// Configuration for the conscious integration hub
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousHubConfig {
    /// Global workspace configuration
    pub workspace_config: GlobalWorkspaceConfig,
    /// Predictive coding configuration
    pub predictive_config: PredictiveCodingConfig,
    /// IIT Φ configuration
    pub phi_config: PhiConfig,
    /// Ecological mind configuration
    pub ecology_config: EcologyConfig,
    /// Bridge coupling strengths
    pub broadcast_to_prediction: f64,
    pub error_to_activation: f64,
    pub phi_threshold: f64,
    pub phi_coupling: f64,
    pub ecology_to_stdp: f64,
    pub ecology_to_topology: f64,
}

impl Default for ConsciousHubConfig {
    fn default() -> Self {
        Self {
            workspace_config: GlobalWorkspaceConfig::default(),
            predictive_config: PredictiveCodingConfig::default(),
            phi_config: PhiConfig::default(),
            ecology_config: EcologyConfig::default(),
            broadcast_to_prediction: 0.3,
            error_to_activation: 0.2,
            phi_threshold: 0.1,
            phi_coupling: 0.1,
            ecology_to_stdp: 0.2,
            ecology_to_topology: 0.2,
        }
    }
}

/// Unified hub for conscious integration
///
/// Coordinates all Phase 4 cognitive systems:
/// - GlobalWorkspace (Baars): conscious broadcast
/// - PredictiveEnactive (Friston/Clark): hierarchical inference
/// - PhiCalculator (Tononi): integrated information
/// - EcologicalMind (Bateson): multi-level learning
pub struct ConsciousIntegrationHub {
    /// Global workspace
    pub workspace: GlobalWorkspace,
    /// Predictive-enactive system
    pub predictive: PredictiveEnactiveSystem,
    /// IIT Φ calculator
    pub phi: PhiCalculator,
    /// Ecological mind
    pub ecology: EcologicalMind,
    /// Workspace ↔ Predictive bridge
    workspace_predictive_bridge: WorkspacePredictiveBridge,
    /// Φ ↔ Workspace bridge
    phi_workspace_bridge: PhiWorkspaceBridge,
    /// Ecology ↔ Learning bridge (stored for use with external STDP/topology)
    pub ecology_learning_config: (f64, f64),
    /// Current time
    time: f64,
    /// Integration statistics
    pub stats: ConsciousHubStats,
}

/// Statistics for conscious integration hub
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConsciousHubStats {
    /// Total steps
    pub total_steps: u64,
    /// Total broadcasts
    pub total_broadcasts: u64,
    /// Total ignitions
    pub total_ignitions: u64,
    /// Average Φ
    pub avg_phi: f64,
    /// Average prediction error
    pub avg_prediction_error: f64,
    /// Dominant learning level
    pub dominant_learning_level: Option<String>,
    /// System coherence (correlation between subsystems)
    pub coherence: f64,
}

impl ConsciousIntegrationHub {
    /// Create new conscious integration hub
    pub fn new(config: ConsciousHubConfig) -> Self {
        let num_specialists = config.workspace_config.num_specialists;
        let num_levels = config.predictive_config.num_levels;

        Self {
            workspace: GlobalWorkspace::new(config.workspace_config),
            predictive: PredictiveEnactiveSystem::new(config.predictive_config),
            // PhiCalculator needs system_size (use num_levels as proxy for mechanism elements)
            phi: PhiCalculator::new(config.phi_config, num_levels),
            ecology: EcologicalMind::new(config.ecology_config),
            workspace_predictive_bridge: WorkspacePredictiveBridge::new(
                config.broadcast_to_prediction,
                config.error_to_activation,
                num_specialists,
            ),
            phi_workspace_bridge: PhiWorkspaceBridge::new(
                config.phi_threshold,
                config.phi_coupling,
            ),
            ecology_learning_config: (config.ecology_to_stdp, config.ecology_to_topology),
            time: 0.0,
            stats: ConsciousHubStats::default(),
        }
    }

    /// Step all systems with proper coordination
    pub fn step(&mut self, dt: f64, observation: Observation) -> Option<BroadcastEvent> {
        self.time += dt;
        self.stats.total_steps += 1;

        // 1. Process observation through predictive hierarchy
        let process_result = self.predictive.process_observation(observation.clone());

        // 2. Feed prediction errors to workspace (bottom-up attention)
        self.workspace_predictive_bridge.process_errors(
            &process_result.errors,
            &mut self.workspace,
        );

        // 3. Run workspace competition
        let broadcast_event = self.workspace.step(dt);

        // 4. If broadcast, inject into predictive top level (top-down)
        if let Some(ref broadcast) = broadcast_event {
            self.workspace_predictive_bridge.process_broadcast(
                broadcast,
                &mut self.predictive,
            );
            self.stats.total_broadcasts += 1;
            if broadcast.ignited {
                self.stats.total_ignitions += 1;
            }
        }

        // 5. Compute Φ (periodically, as it's expensive)
        if self.stats.total_steps % 10 == 0 {
            // Combine hidden states from all levels into a single binary state vector
            // Take one element from each level to form the state
            let combined_state: Vec<bool> = self.predictive.belief.levels.iter()
                .map(|belief| {
                    // Average of hidden state > 0.5 indicates "active"
                    let avg: f64 = belief.hidden_state.iter().sum::<f64>()
                        / belief.hidden_state.len().max(1) as f64;
                    avg > 0.5
                })
                .collect();

            // Set the combined state
            self.phi.set_state(combined_state);

            // Compute Φ (returns PhiResult directly, not Result)
            let phi_result = self.phi.compute_phi();
            self.phi_workspace_bridge.process_phi(&phi_result, &mut self.workspace);
            self.stats.avg_phi = 0.95 * self.stats.avg_phi + 0.05 * phi_result.phi;
        }

        // 6. Update ecological learning from context changes
        // (Ecological mind updates are context-driven, not time-stepped)

        // 7. Update statistics
        self.stats.avg_prediction_error = self.workspace_predictive_bridge.average_error();
        self.stats.coherence = self.compute_coherence();

        broadcast_event
    }

    /// Compute system coherence (correlation between subsystem states)
    fn compute_coherence(&self) -> f64 {
        // Coherence = correlation between workspace activation and prediction confidence
        let workspace_activation: f64 = self.workspace.specialists.iter()
            .map(|s| s.activation)
            .sum::<f64>() / self.workspace.specialists.len() as f64;

        let prediction_confidence: f64 = self.predictive.belief.levels.iter()
            .map(|b| b.hidden_precision)
            .sum::<f64>() / self.predictive.belief.levels.len() as f64;

        // Simple coherence: product of normalized values
        let max_precision = 10.0; // From predictive config
        (workspace_activation * prediction_confidence / max_precision).min(1.0)
    }

    /// Get the current conscious content (if any)
    pub fn conscious_content(&self) -> Option<&WorkspaceContent> {
        self.workspace.workspace.as_ref()
    }

    /// Check if workspace is ignited (conscious access)
    pub fn is_conscious(&self) -> bool {
        self.workspace.is_ignited
    }

    /// Get integration statistics
    pub fn stats(&self) -> &ConsciousHubStats {
        &self.stats
    }

    /// Create ecology-learning bridge for external use
    pub fn create_ecology_learning_bridge(&self) -> EcologyLearningBridge {
        EcologyLearningBridge::new(
            self.ecology_learning_config.0,
            self.ecology_learning_config.1,
        )
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
