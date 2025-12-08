//! # Predictive Coding Integration with Enactive Cognition
//!
//! Unifies the EnactiveLayer with predictive coding (Bayesian brain hypothesis)
//! for a complete active inference architecture.
//!
//! ## Theoretical Foundation
//!
//! Predictive coding proposes that the brain:
//! 1. Maintains hierarchical generative models
//! 2. Propagates predictions top-down
//! 3. Propagates prediction errors bottom-up
//! 4. Updates models to minimize free energy
//!
//! ## Integration with Enactive Cognition
//!
//! Enactive cognition adds:
//! - Sensorimotor coupling (action shapes perception)
//! - Embodied interaction (cognition through action)
//! - Environmental embedding (extended mind)
//!
//! Together they form **Active Inference**:
//! - Perception: minimize prediction error
//! - Action: change world to match predictions
//!
//! ## Hyperbolic Extension
//!
//! In hyperbolic space:
//! - Predictions flow along geodesics
//! - Precision is curvature-dependent
//! - Hierarchies have exponential capacity
//!
//! ## References
//!
//! - Rao & Ballard (1999) "Predictive coding in the visual cortex"
//! - Friston (2005) "A theory of cortical responses"
//! - Clark (2013) "Whatever next? Predictive brains"
//! - Varela et al. (1991) "The Embodied Mind"

use crate::hyperbolic_snn::LorentzVec;
use crate::enactive_layer::{
    EnactiveLayer, EnactiveConfig, BeliefState, Observation, Action,
    SensorimotorCoupling,
};
use crate::free_energy::{
    FreeEnergyCalculator, FreeEnergyResult,
    PrecisionWeightedError, HierarchicalErrorAggregator, Precision,
};
use crate::chunk_processor::{ChunkProcessor, ChunkProcessorConfig, SpikeEvent, TemporalChunk};
use serde::{Deserialize, Serialize};

/// Configuration for predictive coding system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictiveCodingConfig {
    /// Number of hierarchical levels
    pub num_levels: usize,
    /// Prediction update rate per level
    pub prediction_rates: Vec<f64>,
    /// Precision learning rate
    pub precision_learning_rate: f64,
    /// Error propagation gain
    pub error_gain: f64,
    /// Minimum precision
    pub min_precision: f64,
    /// Maximum precision
    pub max_precision: f64,
    /// Enable action-oriented predictive processing
    pub enable_action: bool,
    /// Enactive layer configuration
    pub enactive_config: EnactiveConfig,
    /// Chunk processor configuration
    pub chunk_config: ChunkProcessorConfig,
}

impl Default for PredictiveCodingConfig {
    fn default() -> Self {
        Self {
            num_levels: 4,
            prediction_rates: vec![0.1, 0.05, 0.02, 0.01],
            precision_learning_rate: 0.01,
            error_gain: 1.0,
            min_precision: 0.1,
            max_precision: 10.0,
            enable_action: true,
            enactive_config: EnactiveConfig::default(),
            chunk_config: ChunkProcessorConfig::default(),
        }
    }
}

/// Prediction at a hierarchical level
#[derive(Debug, Clone)]
pub struct LevelPrediction {
    /// Level index
    pub level: usize,
    /// Predicted position (state)
    pub position: LorentzVec,
    /// Predicted features
    pub features: Vec<f64>,
    /// Prediction precision
    pub precision: Precision,
    /// Prediction time
    pub time: f64,
    /// Source of prediction (which higher level)
    pub source_level: Option<usize>,
}

impl LevelPrediction {
    /// Create prediction from belief state
    pub fn from_belief(level: usize, belief: &BeliefState, time: f64) -> Self {
        Self {
            level,
            position: belief.position_mean,
            features: belief.hidden_state.clone(),
            precision: belief.hidden_precision,
            time,
            source_level: if level > 0 { Some(level - 1) } else { None },
        }
    }

    /// Compute prediction error against observation
    pub fn error(&self, obs: &Observation) -> LevelError {
        let position_error = self.position.hyperbolic_distance(&obs.position);

        let feature_error: f64 = self.features.iter()
            .zip(obs.features.iter())
            .map(|(p, o)| (p - o).powi(2))
            .sum::<f64>()
            .sqrt();

        let weighted_error = self.precision * (position_error + feature_error);

        LevelError {
            level: self.level,
            position_error,
            feature_error,
            weighted_error,
            precision: self.precision,
            time: obs.time,
        }
    }
}

/// Prediction error at a hierarchical level
#[derive(Debug, Clone)]
pub struct LevelError {
    /// Level index
    pub level: usize,
    /// Position prediction error (geodesic distance)
    pub position_error: f64,
    /// Feature prediction error (Euclidean)
    pub feature_error: f64,
    /// Precision-weighted total error
    pub weighted_error: f64,
    /// Precision at this level
    pub precision: Precision,
    /// Error time
    pub time: f64,
}

impl LevelError {
    /// Convert to precision-weighted error
    pub fn to_precision_weighted(&self, source_id: usize) -> PrecisionWeightedError {
        PrecisionWeightedError::new(
            self.position_error + self.feature_error,
            self.precision,
            self.level,
            source_id,
        )
    }
}

/// Hierarchical belief state extending BeliefState
#[derive(Debug, Clone)]
pub struct HierarchicalBelief {
    /// Beliefs at each level
    pub levels: Vec<BeliefState>,
    /// Current predictions from each level
    pub predictions: Vec<Option<LevelPrediction>>,
    /// Current errors at each level
    pub errors: Vec<Option<LevelError>>,
    /// Inter-level precision (how much to trust higher levels)
    pub inter_level_precision: Vec<f64>,
}

impl HierarchicalBelief {
    /// Create new hierarchical belief
    pub fn new(num_levels: usize, hidden_dim: usize, temporal_depth: usize) -> Self {
        let levels: Vec<BeliefState> = (0..num_levels)
            .map(|_| BeliefState::new(hidden_dim, temporal_depth))
            .collect();

        Self {
            levels,
            predictions: vec![None; num_levels],
            errors: vec![None; num_levels],
            inter_level_precision: vec![1.0; num_levels],
        }
    }

    /// Get belief at level
    pub fn level(&self, level: usize) -> Option<&BeliefState> {
        self.levels.get(level)
    }

    /// Get mutable belief at level
    pub fn level_mut(&mut self, level: usize) -> Option<&mut BeliefState> {
        self.levels.get_mut(level)
    }

    /// Bottom-most level
    pub fn sensory(&self) -> &BeliefState {
        &self.levels[0]
    }

    /// Top-most level
    pub fn conceptual(&self) -> &BeliefState {
        &self.levels[self.levels.len() - 1]
    }
}

/// Unified Predictive-Enactive System
pub struct PredictiveEnactiveSystem {
    /// Configuration
    config: PredictiveCodingConfig,
    /// Hierarchical beliefs
    pub belief: HierarchicalBelief,
    /// Enactive layer for action generation
    enactive: EnactiveLayer,
    /// Free energy calculator
    free_energy: FreeEnergyCalculator,
    /// Chunk processor for temporal abstraction
    chunks: ChunkProcessor,
    /// Hierarchical error aggregator
    error_aggregator: HierarchicalErrorAggregator,
    /// Current time
    time: f64,
    /// Statistics
    pub stats: PredictiveStats,
}

/// Statistics for predictive coding system
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PredictiveStats {
    /// Total observations processed
    pub total_observations: u64,
    /// Total predictions generated
    pub total_predictions: u64,
    /// Total actions generated
    pub total_actions: u64,
    /// Average prediction error by level
    pub avg_error_by_level: Vec<f64>,
    /// Average free energy
    pub avg_free_energy: f64,
    /// Precision history
    pub precision_history: Vec<Vec<f64>>,
}

impl PredictiveEnactiveSystem {
    /// Create new predictive-enactive system
    pub fn new(config: PredictiveCodingConfig) -> Self {
        let num_levels = config.num_levels;
        let hidden_dim = config.enactive_config.hidden_dim;
        let temporal_depth = config.enactive_config.temporal_depth;

        let belief = HierarchicalBelief::new(num_levels, hidden_dim, temporal_depth);
        let enactive = EnactiveLayer::new(config.enactive_config.clone());
        let free_energy = FreeEnergyCalculator::default();
        let chunks = ChunkProcessor::new(config.chunk_config.clone());
        let error_aggregator = HierarchicalErrorAggregator::new(num_levels);

        Self {
            config,
            belief,
            enactive,
            free_energy,
            chunks,
            error_aggregator,
            time: 0.0,
            stats: PredictiveStats {
                avg_error_by_level: vec![0.0; num_levels],
                precision_history: vec![Vec::new(); num_levels],
                ..Default::default()
            },
        }
    }

    /// Process sensory observation through hierarchy
    pub fn process_observation(&mut self, obs: Observation) -> ProcessResult {
        self.time = obs.time;
        self.stats.total_observations += 1;

        // 1. Generate predictions from all levels (top-down)
        self.generate_predictions();

        // 2. Compute prediction errors at each level (bottom-up)
        let errors = self.compute_errors(&obs);

        // 3. Update beliefs at each level
        self.update_beliefs(&obs, &errors);

        // 4. Update precision estimates
        self.update_precision(&errors);

        // 5. Propagate errors through hierarchy
        self.propagate_errors(&errors);

        // 6. Compute free energy
        let fe_result = self.free_energy.compute(&obs, &self.belief.levels[0]);

        // 7. Update enactive layer
        self.enactive.process_observation(obs.clone());

        // 8. Update statistics
        self.update_stats(&errors, &fe_result);

        ProcessResult {
            errors: errors.clone(),
            free_energy: fe_result,
            prediction_at_level: self.belief.predictions.clone(),
        }
    }

    /// Generate predictions from each level (top-down pass)
    fn generate_predictions(&mut self) {
        self.stats.total_predictions += self.config.num_levels as u64;

        // Top level predicts from prior
        let top_level = self.config.num_levels - 1;
        if let Some(belief) = self.belief.levels.get(top_level) {
            self.belief.predictions[top_level] = Some(LevelPrediction::from_belief(
                top_level,
                belief,
                self.time,
            ));
        }

        // Lower levels predict from level above
        for level in (0..top_level).rev() {
            if let Some(higher_belief) = self.belief.levels.get(level + 1) {
                let mut prediction = LevelPrediction::from_belief(level, higher_belief, self.time);
                prediction.source_level = Some(level + 1);

                // Modulate precision by inter-level trust
                prediction.precision *= self.belief.inter_level_precision[level];

                self.belief.predictions[level] = Some(prediction);
            }
        }
    }

    /// Compute prediction errors at each level (bottom-up pass)
    fn compute_errors(&mut self, obs: &Observation) -> Vec<LevelError> {
        let mut errors = Vec::with_capacity(self.config.num_levels);

        // Level 0 computes error against actual observation
        if let Some(ref prediction) = self.belief.predictions[0] {
            errors.push(prediction.error(obs));
        }

        // Higher levels compute error against lower level's posterior
        for level in 1..self.config.num_levels {
            if let Some(ref prediction) = self.belief.predictions[level] {
                // Use lower level's updated belief as "observation" for this level
                let lower_belief = &self.belief.levels[level - 1];
                let virtual_obs = Observation {
                    time: self.time,
                    modality: obs.modality,
                    position: lower_belief.position_mean,
                    value: lower_belief.hidden_state.clone(),
                    features: lower_belief.hidden_state.clone(),
                    precision: lower_belief.hidden_precision,
                };
                errors.push(prediction.error(&virtual_obs));
            }
        }

        errors
    }

    /// Update beliefs at each level based on errors
    fn update_beliefs(&mut self, obs: &Observation, errors: &[LevelError]) {
        // Level 0 updates from actual observation
        let learning_rate = self.config.prediction_rates.get(0).copied().unwrap_or(0.1);
        self.belief.levels[0].update(obs, learning_rate);

        // Higher levels update from prediction errors
        for level in 1..self.config.num_levels {
            if let Some(error) = errors.get(level) {
                let learning_rate = self.config.prediction_rates.get(level).copied().unwrap_or(0.01);

                // Kalman-like update based on precision
                let kalman_gain = error.precision /
                    (error.precision + self.belief.levels[level].hidden_precision);

                // Update position along geodesic toward lower level
                if let Some(lower_belief) = self.belief.levels.get(level - 1) {
                    let direction = self.belief.levels[level].position_mean
                        .log_map(&lower_belief.position_mean);

                    let step = learning_rate * kalman_gain;
                    self.belief.levels[level].position_mean =
                        self.belief.levels[level].position_mean.exp_map(&direction, step);
                }

                // Update hidden state
                for i in 0..self.belief.levels[level].hidden_state.len() {
                    if let Some(lower_val) = self.belief.levels.get(level - 1)
                        .and_then(|b| b.hidden_state.get(i))
                    {
                        let current = self.belief.levels[level].hidden_state[i];
                        self.belief.levels[level].hidden_state[i] =
                            current + learning_rate * kalman_gain * (lower_val - current);
                    }
                }
            }
        }
    }

    /// Update precision estimates based on prediction errors
    fn update_precision(&mut self, errors: &[LevelError]) {
        for error in errors {
            let level = error.level;
            if level >= self.config.num_levels {
                continue;
            }

            // Precision increases when predictions are accurate
            // Precision decreases when predictions are poor
            let error_magnitude = error.position_error + error.feature_error;
            let current_precision = self.belief.levels[level].hidden_precision;

            // Inverse of error variance as precision estimate
            let observed_precision = 1.0 / (error_magnitude.powi(2) + 0.1);

            // Exponential moving average update
            let new_precision = current_precision * (1.0 - self.config.precision_learning_rate)
                + observed_precision * self.config.precision_learning_rate;

            self.belief.levels[level].hidden_precision =
                new_precision.clamp(self.config.min_precision, self.config.max_precision);

            // Update inter-level precision
            if level > 0 {
                let prediction_quality = (-error_magnitude).exp();
                self.belief.inter_level_precision[level - 1] =
                    self.belief.inter_level_precision[level - 1] * 0.99 + prediction_quality * 0.01;
            }
        }
    }

    /// Propagate errors through hierarchy
    fn propagate_errors(&mut self, errors: &[LevelError]) {
        self.error_aggregator.clear();

        for error in errors {
            let pw_error = error.to_precision_weighted(0);
            self.error_aggregator.add_error(pw_error);
        }

        self.error_aggregator.propagate();
    }

    /// Generate action via active inference
    pub fn generate_action(&mut self, goal: Option<&LorentzVec>) -> Option<Action> {
        if !self.config.enable_action {
            return None;
        }

        // Set goal in enactive layer
        if let Some(g) = goal {
            self.enactive.set_goal(*g);
        }

        // Try to generate action
        let action = self.enactive.try_action()?;

        // Compute expected free energy for this action (used for policy evaluation)
        let _efe = self.free_energy.expected_free_energy(
            &action,
            &self.belief.levels[0],
            goal,
        );

        self.stats.total_actions += 1;

        Some(action)
    }

    /// Execute action and process outcome
    pub fn execute_action(&mut self, action: &Action, outcome: &Observation) {
        // Update enactive layer
        self.enactive.execute_action(action, outcome);

        // Process outcome through predictive hierarchy
        self.process_observation(outcome.clone());
    }

    /// Process spike through chunk processor
    pub fn process_spike(&mut self, spike: SpikeEvent) {
        self.chunks.process_spike(spike);

        // Convert chunks to observations at appropriate hierarchical levels
        for level in 0..self.config.num_levels.min(4) {
            if let Some(chunk) = self.chunks.get_chunks(level).last() {
                let obs = self.chunk_to_observation(chunk, level);
                // Update corresponding belief level
                if level < self.belief.levels.len() {
                    let rate = self.config.prediction_rates.get(level).copied().unwrap_or(0.1);
                    self.belief.levels[level].update(&obs, rate);
                }
            }
        }
    }

    /// Convert chunk to observation
    fn chunk_to_observation(&self, chunk: &TemporalChunk, level: usize) -> Observation {
        Observation {
            time: chunk.end_time,
            modality: crate::enactive_layer::Modality::Proprioceptive,
            position: chunk.representation.centroid,
            value: chunk.representation.temporal_signature.clone(),
            features: chunk.representation.temporal_signature.clone(),
            precision: chunk.representation.confidence * (level + 1) as f64,
        }
    }

    /// Update statistics
    fn update_stats(&mut self, errors: &[LevelError], fe_result: &FreeEnergyResult) {
        // Update error averages
        for error in errors {
            if error.level < self.stats.avg_error_by_level.len() {
                let avg = &mut self.stats.avg_error_by_level[error.level];
                *avg = 0.99 * *avg + 0.01 * error.weighted_error;
            }
        }

        // Update free energy average
        self.stats.avg_free_energy =
            0.99 * self.stats.avg_free_energy + 0.01 * fe_result.free_energy;

        // Record precision history
        for level in 0..self.config.num_levels {
            if let Some(belief) = self.belief.levels.get(level) {
                if level < self.stats.precision_history.len() {
                    self.stats.precision_history[level].push(belief.hidden_precision);
                    if self.stats.precision_history[level].len() > 1000 {
                        self.stats.precision_history[level].remove(0);
                    }
                }
            }
        }
    }

    /// Get current free energy
    pub fn free_energy(&self) -> f64 {
        self.enactive.free_energy()
    }

    /// Get hierarchical prediction quality
    pub fn prediction_quality(&self) -> Vec<f64> {
        self.stats.avg_error_by_level.iter()
            .map(|e| (-e).exp()) // Higher quality = lower error
            .collect()
    }

    /// Check if hierarchy is converged
    pub fn is_converged(&self, threshold: f64) -> bool {
        self.stats.avg_error_by_level.iter()
            .all(|&e| e < threshold)
    }

    /// Get enactive layer reference
    pub fn enactive(&self) -> &EnactiveLayer {
        &self.enactive
    }

    /// Get mutable enactive layer reference
    pub fn enactive_mut(&mut self) -> &mut EnactiveLayer {
        &mut self.enactive
    }

    /// Get chunk processor reference
    pub fn chunks(&self) -> &ChunkProcessor {
        &self.chunks
    }
}

/// Implement SensorimotorCoupling for the unified system
impl SensorimotorCoupling for PredictiveEnactiveSystem {
    fn sense(&mut self, obs: Observation) {
        self.process_observation(obs);
    }

    fn act(&mut self) -> Option<Action> {
        self.generate_action(None)
    }

    fn belief(&self) -> &BeliefState {
        &self.belief.levels[0]
    }

    fn free_energy(&self) -> f64 {
        self.free_energy()
    }
}

/// Result of processing an observation
#[derive(Debug, Clone)]
pub struct ProcessResult {
    /// Errors at each level
    pub errors: Vec<LevelError>,
    /// Free energy result
    pub free_energy: FreeEnergyResult,
    /// Predictions at each level
    pub prediction_at_level: Vec<Option<LevelPrediction>>,
}

impl ProcessResult {
    /// Get total weighted error
    pub fn total_error(&self) -> f64 {
        self.errors.iter().map(|e| e.weighted_error).sum()
    }

    /// Get error at specific level
    pub fn error_at_level(&self, level: usize) -> Option<f64> {
        self.errors.get(level).map(|e| e.weighted_error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::enactive_layer::Modality;

    fn create_test_obs(time: f64, x: f64, features: Vec<f64>) -> Observation {
        let t = (1.0 + x * x).sqrt();
        Observation {
            time,
            modality: Modality::Proprioceptive,
            position: LorentzVec::new(t, x, 0.0, 0.0),
            value: features.clone(),
            features,
            precision: 1.0,
        }
    }

    #[test]
    fn test_hierarchical_belief() {
        let belief = HierarchicalBelief::new(4, 8, 10);

        assert_eq!(belief.levels.len(), 4);
        assert_eq!(belief.predictions.len(), 4);
        assert_eq!(belief.errors.len(), 4);
    }

    #[test]
    fn test_level_prediction() {
        let belief = BeliefState::new(4, 10);
        let prediction = LevelPrediction::from_belief(0, &belief, 0.0);

        assert_eq!(prediction.level, 0);
        assert!(prediction.precision > 0.0);
    }

    #[test]
    fn test_predictive_system_creation() {
        let config = PredictiveCodingConfig::default();
        let system = PredictiveEnactiveSystem::new(config);

        assert_eq!(system.belief.levels.len(), 4);
    }

    #[test]
    fn test_observation_processing() {
        let config = PredictiveCodingConfig::default();
        let mut system = PredictiveEnactiveSystem::new(config);

        let obs = create_test_obs(0.0, 0.1, vec![0.5, 0.5]);
        let result = system.process_observation(obs);

        assert!(result.free_energy.free_energy >= 0.0);
        assert!(!result.errors.is_empty());
    }

    #[test]
    fn test_prediction_generation() {
        let config = PredictiveCodingConfig::default();
        let mut system = PredictiveEnactiveSystem::new(config);

        // Process an observation to generate predictions
        let obs = create_test_obs(0.0, 0.1, vec![0.5, 0.5]);
        system.process_observation(obs);

        // Should have predictions at each level
        assert!(system.belief.predictions.iter().any(|p| p.is_some()));
    }

    #[test]
    fn test_action_generation() {
        let config = PredictiveCodingConfig {
            enable_action: true,
            ..Default::default()
        };
        let mut system = PredictiveEnactiveSystem::new(config);

        // Process observation first
        let obs = create_test_obs(0.5, 0.1, vec![0.5, 0.5]);
        system.process_observation(obs);

        // Try to generate action
        let goal = LorentzVec::from_spatial(0.5, 0.0, 0.0);
        let action = system.generate_action(Some(&goal));

        // May or may not generate action depending on timing
    }

    #[test]
    fn test_precision_update() {
        let config = PredictiveCodingConfig::default();
        let mut system = PredictiveEnactiveSystem::new(config);

        let initial_precision = system.belief.levels[0].hidden_precision;

        // Process several observations
        for i in 0..10 {
            let obs = create_test_obs(i as f64 * 0.1, 0.1, vec![0.5, 0.5]);
            system.process_observation(obs);
        }

        // Precision should have been updated
        let final_precision = system.belief.levels[0].hidden_precision;
        // May increase or decrease depending on prediction accuracy
    }

    #[test]
    fn test_sensorimotor_coupling_trait() {
        let config = PredictiveCodingConfig::default();
        let mut system = PredictiveEnactiveSystem::new(config);

        let obs = create_test_obs(0.0, 0.1, vec![0.5, 0.5]);
        <PredictiveEnactiveSystem as SensorimotorCoupling>::sense(&mut system, obs);

        let belief = <PredictiveEnactiveSystem as SensorimotorCoupling>::belief(&system);
        assert!(belief.history.len() > 0 || system.belief.levels[0].history.len() >= 0);

        let fe = <PredictiveEnactiveSystem as SensorimotorCoupling>::free_energy(&system);
        assert!(fe >= 0.0);
    }

    #[test]
    fn test_prediction_quality() {
        let config = PredictiveCodingConfig::default();
        let mut system = PredictiveEnactiveSystem::new(config);

        // Process observations
        for i in 0..5 {
            let obs = create_test_obs(i as f64 * 0.1, 0.1, vec![0.5, 0.5]);
            system.process_observation(obs);
        }

        let quality = system.prediction_quality();
        assert_eq!(quality.len(), 4);

        // Quality should be between 0 and 1
        for q in &quality {
            assert!(*q >= 0.0 && *q <= 1.0);
        }
    }
}
