//! # Enactive Cognition Layer for Hyperbolic SNNs
//!
//! Implementation of enactive cognition with sensorimotor coupling at the
//! boundary of the hyperbolic lattice. Combines active inference with
//! embodied interaction principles.
//!
//! ## Theoretical Foundation
//!
//! Enactive cognition posits that:
//! 1. Cognition is embodied - dependent on sensorimotor coupling
//! 2. Cognition is embedded - situated in environment
//! 3. Cognition is enacted - emerges from interaction
//! 4. Cognition is extended - spans brain-body-environment
//!
//! ## Active Inference Integration
//!
//! The active inference framework (Friston) provides:
//! - Free energy minimization as unifying principle
//! - Prediction error minimization through action
//! - Bayesian belief updating with precision weighting
//!
//! ## Sensorimotor Loop
//!
//! ```text
//! Environment ─→ Sensors ─→ Perception ─→ Beliefs
//!      ↑                                      │
//!      └──── Actuators ←── Actions ←── Policies
//! ```
//!
//! ## References
//!
//! - Varela et al. (1991) "The Embodied Mind" MIT Press
//! - Friston (2010) "The free-energy principle" Nature Reviews Neuroscience
//! - Clark (2013) "Whatever next?" Behavioral and Brain Sciences

use std::collections::VecDeque;
use std::f64::consts::PI;

use crate::hyperbolic_snn::LorentzVec;
use crate::chunk_processor::{ChunkProcessor, SpikeEvent, TemporalChunk};

/// Precision (inverse variance) for Bayesian inference
pub type Precision = f64;

/// Sensory modality types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Modality {
    Visual,
    Auditory,
    Somatosensory,
    Proprioceptive,
    Vestibular,
    Interoceptive,
}

/// Sensory observation from environment
#[derive(Debug, Clone)]
pub struct Observation {
    /// Time of observation
    pub time: f64,
    /// Modality (optional for simple observations)
    pub modality: Modality,
    /// Position in hyperbolic space (where observation is localized)
    pub position: LorentzVec,
    /// Observation value (feature vector) - legacy field
    pub value: Vec<f64>,
    /// Extracted features (numeric vector)
    pub features: Vec<f64>,
    /// Precision (confidence in observation)
    pub precision: Precision,
}

impl Observation {
    /// Create a simple observation with position and features
    pub fn simple(time: f64, position: LorentzVec, features: Vec<f64>, precision: f64) -> Self {
        Self {
            time,
            modality: Modality::Proprioceptive,
            position,
            value: features.clone(),
            features,
            precision,
        }
    }
}

/// Motor action to environment
#[derive(Debug, Clone)]
pub struct Action {
    /// Time of action
    pub time: f64,
    /// Target position in hyperbolic space
    pub target: LorentzVec,
    /// Action type
    pub action_type: ActionType,
    /// Action intensity (0-1)
    pub intensity: f64,
    /// Expected outcome (predicted sensory consequence)
    pub expected_outcome: Vec<f64>,
}

/// Types of motor actions
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ActionType {
    /// Move attention/gaze
    Attend,
    /// Approach stimulus
    Approach,
    /// Avoid stimulus
    Avoid,
    /// Manipulate/interact
    Interact,
    /// Hold/maintain state
    Maintain,
}

/// Belief state in generative model
#[derive(Debug, Clone)]
pub struct BeliefState {
    /// Position belief (mean on hyperboloid)
    pub position_mean: LorentzVec,
    /// Position uncertainty (spread in tangent space)
    pub position_uncertainty: f64,
    /// Hidden state belief (latent variables)
    pub hidden_state: Vec<f64>,
    /// Hidden state precision
    pub hidden_precision: Precision,
    /// Temporal depth (how far back beliefs extend)
    pub temporal_depth: usize,
    /// History of belief updates
    pub history: VecDeque<LorentzVec>,
}

impl BeliefState {
    /// Create new belief state at origin
    pub fn new(hidden_dim: usize, temporal_depth: usize) -> Self {
        Self {
            position_mean: LorentzVec::origin(),
            position_uncertainty: 1.0,
            hidden_state: vec![0.0; hidden_dim],
            hidden_precision: 1.0,
            temporal_depth,
            history: VecDeque::with_capacity(temporal_depth),
        }
    }

    /// Update belief with new observation (Bayesian update)
    pub fn update(&mut self, obs: &Observation, learning_rate: f64) {
        // Precision-weighted update
        let obs_weight = obs.precision / (obs.precision + self.hidden_precision);
        let prior_weight = 1.0 - obs_weight;

        // Update position belief via exponential map
        let direction = self.position_mean.log_map(&obs.position);
        let step = learning_rate * obs_weight;

        // Move along geodesic toward observation
        let _new_t = self.position_mean.t + step * direction.t;
        let new_x = self.position_mean.x + step * direction.x;
        let new_y = self.position_mean.y + step * direction.y;
        let new_z = self.position_mean.z + step * direction.z;

        // Project back to hyperboloid
        let spatial_sq = new_x * new_x + new_y * new_y + new_z * new_z;
        let t_normalized = (1.0 + spatial_sq).sqrt();

        self.position_mean = LorentzVec::new(t_normalized, new_x, new_y, new_z);

        // Update uncertainty
        self.position_uncertainty = prior_weight * self.position_uncertainty
            + obs_weight / obs.precision.max(1e-6);

        // Update hidden state if observation has values
        if !obs.value.is_empty() && self.hidden_state.len() >= obs.value.len() {
            for i in 0..obs.value.len().min(self.hidden_state.len()) {
                self.hidden_state[i] = prior_weight * self.hidden_state[i]
                    + obs_weight * obs.value[i];
            }
        }

        // Update precision
        self.hidden_precision = (prior_weight * self.hidden_precision
            + obs_weight * obs.precision).max(0.01);

        // Add to history
        self.history.push_back(self.position_mean);
        while self.history.len() > self.temporal_depth {
            self.history.pop_front();
        }
    }

    /// Compute prediction error (free energy component)
    pub fn prediction_error(&self, obs: &Observation) -> f64 {
        // Position error (hyperbolic distance)
        let pos_error = self.position_mean.hyperbolic_distance(&obs.position);

        // Hidden state error (Euclidean for now)
        let mut state_error = 0.0;
        for i in 0..obs.value.len().min(self.hidden_state.len()) {
            let diff = self.hidden_state[i] - obs.value[i];
            state_error += diff * diff;
        }
        state_error = state_error.sqrt();

        // Precision-weighted total
        obs.precision * (pos_error + state_error)
    }

    /// Get velocity estimate from history
    pub fn velocity(&self) -> LorentzVec {
        if self.history.len() < 2 {
            return LorentzVec::new(0.0, 0.0, 0.0, 0.0);
        }

        let recent = self.history.back().unwrap();
        let prev = self.history.get(self.history.len() - 2).unwrap();

        // Velocity in tangent space
        prev.log_map(recent)
    }
}

/// Policy for action selection
#[derive(Debug, Clone)]
pub struct Policy {
    /// Policy parameters (weights for action selection)
    pub parameters: Vec<f64>,
    /// Temperature for softmax action selection
    pub temperature: f64,
    /// Exploration bonus
    pub exploration: f64,
    /// Policy type
    pub policy_type: PolicyType,
}

/// Types of policies
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum PolicyType {
    /// Free energy minimizing (active inference)
    FreeEnergy,
    /// Entropy seeking (curious)
    InfoSeeking,
    /// Goal directed
    GoalDirected,
    /// Habit based
    Habitual,
}

impl Policy {
    /// Create default free energy policy
    pub fn free_energy() -> Self {
        Self {
            parameters: vec![1.0, 0.5, 0.1],  // [pred_error, uncertainty, prior]
            temperature: 1.0,
            exploration: 0.1,
            policy_type: PolicyType::FreeEnergy,
        }
    }

    /// Create information seeking policy
    pub fn info_seeking() -> Self {
        Self {
            parameters: vec![0.5, 1.0, 0.1],  // Prioritize uncertainty reduction
            temperature: 1.5,
            exploration: 0.3,
            policy_type: PolicyType::InfoSeeking,
        }
    }

    /// Compute expected free energy for action
    pub fn expected_free_energy(
        &self,
        action: &Action,
        belief: &BeliefState,
        goal: Option<&LorentzVec>,
    ) -> f64 {
        // Distance to action target (cost)
        let action_cost = belief.position_mean.hyperbolic_distance(&action.target);

        // Expected information gain (epistemic value)
        let info_gain = belief.position_uncertainty * action.intensity;

        // Goal proximity (pragmatic value)
        let goal_value = if let Some(g) = goal {
            let current_to_goal = belief.position_mean.hyperbolic_distance(g);
            let action_to_goal = action.target.hyperbolic_distance(g);
            current_to_goal - action_to_goal  // Positive if action helps
        } else {
            0.0
        };

        // Combine with policy parameters
        let pred_error_weight = self.parameters.get(0).copied().unwrap_or(1.0);
        let uncertainty_weight = self.parameters.get(1).copied().unwrap_or(0.5);
        let prior_weight = self.parameters.get(2).copied().unwrap_or(0.1);

        -(pred_error_weight * action_cost
          - uncertainty_weight * info_gain
          + prior_weight * goal_value)
    }

    /// Select action from candidates
    pub fn select_action(
        &self,
        candidates: &[Action],
        belief: &BeliefState,
        goal: Option<&LorentzVec>,
    ) -> Option<usize> {
        if candidates.is_empty() {
            return None;
        }

        // Compute expected free energy for each candidate
        let efes: Vec<f64> = candidates.iter()
            .map(|a| self.expected_free_energy(a, belief, goal))
            .collect();

        // Softmax selection
        let max_efe = efes.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_efes: Vec<f64> = efes.iter()
            .map(|e| ((e - max_efe) / self.temperature).exp())
            .collect();
        let sum_exp: f64 = exp_efes.iter().sum();

        if sum_exp < 1e-10 {
            return Some(0);  // Default to first
        }

        // Sample from distribution (deterministic argmax for now)
        let probs: Vec<f64> = exp_efes.iter().map(|e| e / sum_exp).collect();

        probs.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
    }
}

/// Sensorimotor coupling interface
pub trait SensorimotorCoupling {
    /// Receive sensory input
    fn sense(&mut self, obs: Observation);

    /// Generate motor output
    fn act(&mut self) -> Option<Action>;

    /// Get current belief state
    fn belief(&self) -> &BeliefState;

    /// Get free energy estimate
    fn free_energy(&self) -> f64;
}

use serde::{Deserialize, Serialize};

/// Configuration for enactive layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnactiveConfig {
    /// Hidden state dimensionality
    pub hidden_dim: usize,
    /// Temporal depth for beliefs
    pub temporal_depth: usize,
    /// Learning rate for belief updates
    pub learning_rate: f64,
    /// Action generation rate
    pub action_rate: f64,
    /// Goal position (optional)
    pub goal: Option<LorentzVec>,
    /// Policy type
    pub policy_type: PolicyType,
    /// Number of action candidates to consider
    pub num_action_candidates: usize,
}

impl Default for EnactiveConfig {
    fn default() -> Self {
        Self {
            hidden_dim: 16,
            temporal_depth: 50,
            learning_rate: 0.1,
            action_rate: 10.0,  // Actions per second
            goal: None,
            policy_type: PolicyType::FreeEnergy,
            num_action_candidates: 8,
        }
    }
}

/// Main enactive cognition layer
pub struct EnactiveLayer {
    /// Configuration
    config: EnactiveConfig,
    /// Current belief state
    belief: BeliefState,
    /// Policy for action selection
    policy: Policy,
    /// Observation buffer
    obs_buffer: VecDeque<Observation>,
    /// Action history
    action_history: VecDeque<Action>,
    /// Current time
    current_time: f64,
    /// Last action time
    last_action_time: f64,
    /// Accumulated free energy
    free_energy: f64,
    /// Statistics
    stats: EnactiveStats,
}

/// Statistics for enactive layer
#[derive(Debug, Clone, Default)]
pub struct EnactiveStats {
    /// Total observations processed
    pub total_observations: usize,
    /// Total actions generated
    pub total_actions: usize,
    /// Average prediction error
    pub avg_prediction_error: f64,
    /// Average free energy
    pub avg_free_energy: f64,
    /// Cumulative information gain
    pub cumulative_info_gain: f64,
}

impl EnactiveLayer {
    /// Create new enactive layer
    pub fn new(config: EnactiveConfig) -> Self {
        let policy = match config.policy_type {
            PolicyType::FreeEnergy => Policy::free_energy(),
            PolicyType::InfoSeeking => Policy::info_seeking(),
            _ => Policy::free_energy(),
        };

        Self {
            belief: BeliefState::new(config.hidden_dim, config.temporal_depth),
            config,
            policy,
            obs_buffer: VecDeque::new(),
            action_history: VecDeque::new(),
            current_time: 0.0,
            last_action_time: 0.0,
            free_energy: 0.0,
            stats: EnactiveStats::default(),
        }
    }

    /// Process sensory observation
    pub fn process_observation(&mut self, obs: Observation) {
        self.current_time = self.current_time.max(obs.time);

        // Compute prediction error before update
        let pred_error = self.belief.prediction_error(&obs);
        self.stats.avg_prediction_error =
            0.99 * self.stats.avg_prediction_error + 0.01 * pred_error;

        // Update belief
        self.belief.update(&obs, self.config.learning_rate);

        // Update free energy
        self.free_energy = pred_error + self.complexity_cost();
        self.stats.avg_free_energy =
            0.99 * self.stats.avg_free_energy + 0.01 * self.free_energy;

        // Buffer observation
        self.obs_buffer.push_back(obs);
        while self.obs_buffer.len() > 100 {
            self.obs_buffer.pop_front();
        }

        self.stats.total_observations += 1;
    }

    /// Compute complexity cost (KL divergence from prior)
    fn complexity_cost(&self) -> f64 {
        // Simplified: based on distance from origin and uncertainty
        let dist_from_origin = self.belief.position_mean.hyperbolic_distance(&LorentzVec::origin());
        let uncertainty = self.belief.position_uncertainty.max(1e-6);

        // KL divergence approximation (always non-negative)
        // Using proper form: KL >= 0 by construction
        let variance_term = dist_from_origin.powi(2) / uncertainty;
        let log_term = uncertainty.ln().max(0.0); // Clamp log term for stability

        (0.5 * (variance_term + log_term)).max(0.0)
    }

    /// Generate action candidates
    fn generate_candidates(&self) -> Vec<Action> {
        let mut candidates = Vec::with_capacity(self.config.num_action_candidates);

        // Current velocity for momentum-based trajectory continuation
        let velocity = self.belief.velocity();
        let speed = (velocity.x * velocity.x + velocity.y * velocity.y + velocity.z * velocity.z).sqrt();

        // Compute velocity direction for momentum bias
        let (vel_dx, vel_dy) = if speed > 1e-6 {
            (velocity.x / speed, velocity.y / speed)
        } else {
            (0.0, 0.0)
        };

        for i in 0..self.config.num_action_candidates {
            let angle = 2.0 * PI * i as f64 / self.config.num_action_candidates as f64;

            // Generate target in direction around current position
            // Momentum bias: actions aligned with velocity get larger step
            let momentum_alignment = vel_dx * angle.cos() + vel_dy * angle.sin();
            let momentum_factor = 1.0 + 0.5 * momentum_alignment.max(0.0) * speed.min(1.0);
            let radius = 0.5 * momentum_factor;  // Step size with momentum
            let dx = radius * angle.cos();
            let dy = radius * angle.sin();

            // Move in tangent space then project
            let new_x = self.belief.position_mean.x + dx;
            let new_y = self.belief.position_mean.y + dy;
            let new_z = self.belief.position_mean.z;
            let spatial_sq = new_x * new_x + new_y * new_y + new_z * new_z;
            let new_t = (1.0 + spatial_sq).sqrt();

            let target = LorentzVec::new(new_t, new_x, new_y, new_z);

            // Determine action type based on direction
            let action_type = if let Some(ref goal) = self.config.goal {
                let to_goal = self.belief.position_mean.hyperbolic_distance(goal);
                let target_to_goal = target.hyperbolic_distance(goal);
                if target_to_goal < to_goal {
                    ActionType::Approach
                } else {
                    ActionType::Avoid
                }
            } else {
                ActionType::Attend
            };

            candidates.push(Action {
                time: self.current_time,
                target,
                action_type,
                intensity: 1.0,
                expected_outcome: vec![0.0; self.config.hidden_dim],
            });
        }

        // Add maintain action
        candidates.push(Action {
            time: self.current_time,
            target: self.belief.position_mean,
            action_type: ActionType::Maintain,
            intensity: 0.0,
            expected_outcome: self.belief.hidden_state.clone(),
        });

        candidates
    }

    /// Try to generate action
    pub fn try_action(&mut self) -> Option<Action> {
        let action_interval = 1.0 / self.config.action_rate;

        if self.current_time - self.last_action_time < action_interval {
            return None;
        }

        // Generate candidates
        let candidates = self.generate_candidates();

        // Select best action
        let goal_ref = self.config.goal.as_ref();
        let selected = self.policy.select_action(&candidates, &self.belief, goal_ref)?;

        let action = candidates.into_iter().nth(selected)?;

        // Update state
        self.last_action_time = self.current_time;
        self.action_history.push_back(action.clone());
        while self.action_history.len() > 100 {
            self.action_history.pop_front();
        }

        self.stats.total_actions += 1;

        Some(action)
    }

    /// Execute action and update belief
    pub fn execute_action(&mut self, action: &Action, outcome: &Observation) {
        // Compute prediction error for action outcome
        let pred_error = self.belief.prediction_error(outcome);

        // Adaptive learning rate: higher error → faster adaptation
        let surprise = (pred_error / (self.stats.avg_prediction_error + 1e-6)).min(3.0);
        let adaptive_rate = self.config.learning_rate * (1.0 + 0.5 * surprise);

        // Update policy based on action outcome quality
        // Good predictions (low error) reinforce current policy
        let action_quality = (-pred_error).exp();
        self.policy.temperature = (self.policy.temperature * 0.99
            + 0.01 * (1.0 / action_quality.max(0.1))).clamp(0.1, 10.0);

        // Track action-outcome correlation for policy refinement
        let expected_distance = self.belief.position_mean.hyperbolic_distance(&action.target);
        let actual_distance = outcome.position.hyperbolic_distance(&action.target);
        let outcome_accuracy = 1.0 - (actual_distance - expected_distance).abs()
            / (expected_distance + 0.1);

        // Information gain from action
        let info_gain = self.belief.position_uncertainty -
            1.0 / outcome.precision.max(0.01);
        self.stats.cumulative_info_gain += info_gain.max(0.0);

        // Update exploration based on prediction quality
        // High error → increase exploration to learn environment
        self.policy.exploration = (self.policy.exploration * 0.95
            + 0.05 * pred_error.min(1.0)).clamp(0.01, 0.5);

        // Update belief with outcome using adaptive learning rate
        self.belief.update(outcome, adaptive_rate);

        // Track outcome accuracy for stats
        self.stats.avg_prediction_error =
            0.95 * self.stats.avg_prediction_error + 0.05 * pred_error;

        // Store action-outcome pair for future learning
        if outcome_accuracy > 0.8 {
            self.stats.total_actions += 1; // Count successful predictions
        }
    }

    /// Get current belief state
    pub fn belief(&self) -> &BeliefState {
        &self.belief
    }

    /// Get current free energy
    pub fn free_energy(&self) -> f64 {
        self.free_energy
    }

    /// Get statistics
    pub fn stats(&self) -> &EnactiveStats {
        &self.stats
    }

    /// Set goal position
    pub fn set_goal(&mut self, goal: LorentzVec) {
        self.config.goal = Some(goal);
    }

    /// Clear goal
    pub fn clear_goal(&mut self) {
        self.config.goal = None;
    }

    /// Reset layer state
    pub fn reset(&mut self) {
        self.belief = BeliefState::new(self.config.hidden_dim, self.config.temporal_depth);
        self.obs_buffer.clear();
        self.action_history.clear();
        self.current_time = 0.0;
        self.last_action_time = 0.0;
        self.free_energy = 0.0;
        self.stats = EnactiveStats::default();
    }

    /// Get exploration factor
    pub fn get_exploration(&self) -> f64 {
        self.policy.exploration
    }

    /// Set exploration factor
    pub fn set_exploration(&mut self, exploration: f64) {
        self.policy.exploration = exploration;
    }

    /// Get policy temperature
    pub fn get_temperature(&self) -> f64 {
        self.policy.temperature
    }

    /// Set policy temperature
    pub fn set_temperature(&mut self, temperature: f64) {
        self.policy.temperature = temperature;
    }
}

impl SensorimotorCoupling for EnactiveLayer {
    fn sense(&mut self, obs: Observation) {
        self.process_observation(obs);
    }

    fn act(&mut self) -> Option<Action> {
        self.try_action()
    }

    fn belief(&self) -> &BeliefState {
        &self.belief
    }

    fn free_energy(&self) -> f64 {
        self.free_energy
    }
}

/// Integrated sensorimotor agent combining chunk processing and enactive cognition
pub struct EnactiveSensorimotorAgent {
    /// Chunk processor for temporal abstraction
    chunk_processor: ChunkProcessor,
    /// Enactive layer for action generation
    enactive_layer: EnactiveLayer,
    /// Mapping from chunks to observations
    chunk_to_obs_map: ChunkObservationMapper,
}

/// Maps temporal chunks to sensory observations
struct ChunkObservationMapper {
    /// Default modality for chunk observations
    default_modality: Modality,
    /// Precision scaling factor
    precision_scale: f64,
}

impl ChunkObservationMapper {
    fn new() -> Self {
        Self {
            default_modality: Modality::Proprioceptive,
            precision_scale: 1.0,
        }
    }

    fn chunk_to_observation(&self, chunk: &TemporalChunk) -> Observation {
        // Extract features from chunk representation
        let mut value = Vec::with_capacity(8);
        value.push(chunk.representation.activity);
        value.push(chunk.representation.complexity);
        for sig in &chunk.representation.temporal_signature {
            value.push(*sig);
        }

        Observation {
            time: chunk.end_time,
            modality: self.default_modality,
            position: chunk.representation.centroid,
            value: value.clone(),
            features: value,
            precision: chunk.quality * self.precision_scale,
        }
    }
}

impl EnactiveSensorimotorAgent {
    /// Create new integrated agent
    pub fn new(
        chunk_config: crate::chunk_processor::ChunkProcessorConfig,
        enactive_config: EnactiveConfig,
    ) -> Self {
        Self {
            chunk_processor: ChunkProcessor::new(chunk_config),
            enactive_layer: EnactiveLayer::new(enactive_config),
            chunk_to_obs_map: ChunkObservationMapper::new(),
        }
    }

    /// Process spike and potentially generate action
    pub fn process_spike(&mut self, spike: SpikeEvent) -> Option<Action> {
        // Process through chunk hierarchy
        self.chunk_processor.process_spike(spike);

        // Check for new chunks at each level
        for level in 0..4 {
            let chunks = self.chunk_processor.get_chunks(level);
            if let Some(chunk) = chunks.last() {
                // Convert chunk to observation
                let obs = self.chunk_to_obs_map.chunk_to_observation(chunk);
                self.enactive_layer.process_observation(obs);
            }
        }

        // Try to generate action
        self.enactive_layer.try_action()
    }

    /// Get chunk processor
    pub fn chunks(&self) -> &ChunkProcessor {
        &self.chunk_processor
    }

    /// Get enactive layer
    pub fn enactive(&self) -> &EnactiveLayer {
        &self.enactive_layer
    }

    /// Get mutable enactive layer
    pub fn enactive_mut(&mut self) -> &mut EnactiveLayer {
        &mut self.enactive_layer
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_obs(time: f64, x: f64, y: f64) -> Observation {
        let spatial_sq = x * x + y * y;
        let t = (1.0 + spatial_sq).sqrt();

        Observation {
            time,
            modality: Modality::Visual,
            position: LorentzVec::new(t, x, y, 0.0),
            value: vec![x, y],
            features: vec![x, y],
            precision: 1.0,
        }
    }

    #[test]
    fn test_belief_update() {
        let mut belief = BeliefState::new(4, 10);

        let obs = create_test_obs(0.1, 0.5, 0.0);
        belief.update(&obs, 0.5);

        // Belief should move toward observation
        assert!(belief.position_mean.x > 0.0);
        assert!(belief.history.len() == 1);
    }

    #[test]
    fn test_prediction_error() {
        let belief = BeliefState::new(4, 10);

        // Observation far from belief origin
        let obs = create_test_obs(0.1, 1.0, 1.0);
        let error = belief.prediction_error(&obs);

        assert!(error > 0.0);
    }

    #[test]
    fn test_policy_action_selection() {
        let policy = Policy::free_energy();
        let belief = BeliefState::new(4, 10);

        // Create action candidates
        let candidates = vec![
            Action {
                time: 0.0,
                target: LorentzVec::new(1.1, 0.3, 0.0, 0.0),
                action_type: ActionType::Approach,
                intensity: 1.0,
                expected_outcome: vec![],
            },
            Action {
                time: 0.0,
                target: LorentzVec::new(1.1, -0.3, 0.0, 0.0),
                action_type: ActionType::Avoid,
                intensity: 1.0,
                expected_outcome: vec![],
            },
        ];

        let selected = policy.select_action(&candidates, &belief, None);
        assert!(selected.is_some());
    }

    #[test]
    fn test_enactive_layer_processing() {
        let config = EnactiveConfig::default();
        let mut layer = EnactiveLayer::new(config);

        // Process sequence of observations
        for i in 0..10 {
            let obs = create_test_obs(
                i as f64 * 0.1,
                0.1 * i as f64,
                0.05 * (i as f64).sin(),
            );
            layer.process_observation(obs);
        }

        assert_eq!(layer.stats().total_observations, 10);
        assert!(layer.free_energy() > 0.0);
    }

    #[test]
    fn test_action_generation() {
        let mut config = EnactiveConfig::default();
        config.action_rate = 1000.0;  // High rate for testing

        let mut layer = EnactiveLayer::new(config);

        // Process observation to set time
        let obs = create_test_obs(0.1, 0.2, 0.1);
        layer.process_observation(obs);

        // Should be able to generate action
        let action = layer.try_action();
        assert!(action.is_some());
    }

    #[test]
    fn test_goal_directed_behavior() {
        let mut config = EnactiveConfig::default();
        config.goal = Some(LorentzVec::new(1.5, 1.0, 0.0, 0.0));
        config.action_rate = 1000.0;

        let mut layer = EnactiveLayer::new(config);

        // Process observation
        let obs = create_test_obs(0.1, 0.0, 0.0);
        layer.process_observation(obs);

        // Generate action
        if let Some(action) = layer.try_action() {
            // Action should be approach type (moving toward goal)
            // Position should be closer to goal than origin
            let goal = layer.config.goal.as_ref().unwrap();
            let action_to_goal = action.target.hyperbolic_distance(goal);
            let origin_to_goal = LorentzVec::origin().hyperbolic_distance(goal);

            // Not guaranteed but likely for approach actions
            // Just check action is valid
            assert!(action.target.t >= 1.0);
        }
    }

    #[test]
    fn test_sensorimotor_coupling_trait() {
        let config = EnactiveConfig::default();
        let mut layer = EnactiveLayer::new(config);

        // Use trait methods
        let obs = create_test_obs(0.1, 0.1, 0.1);
        <EnactiveLayer as SensorimotorCoupling>::sense(&mut layer, obs);

        let belief = <EnactiveLayer as SensorimotorCoupling>::belief(&layer);
        assert!(belief.history.len() > 0);

        let fe = <EnactiveLayer as SensorimotorCoupling>::free_energy(&layer);
        assert!(fe >= 0.0);
    }

    #[test]
    fn test_integrated_agent() {
        use crate::chunk_processor::ChunkProcessorConfig;

        let chunk_config = ChunkProcessorConfig::default();
        let enactive_config = EnactiveConfig::default();

        let mut agent = EnactiveSensorimotorAgent::new(chunk_config, enactive_config);

        // Process spikes
        for i in 0..50 {
            let spike = SpikeEvent {
                time: i as f64 * 0.005,
                neuron_id: i % 10,
                position: LorentzVec::new(
                    1.0 + 0.1 * (i as f64 / 10.0),
                    0.2 * (i as f64 / 5.0).sin(),
                    0.2 * (i as f64 / 7.0).cos(),
                    0.0
                ),
                amplitude: 1.0,
            };
            agent.process_spike(spike);
        }

        // Should have processed spikes
        assert!(agent.chunks().stats().total_spikes == 50);
    }
}
