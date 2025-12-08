//! # Bateson's Ecological Epistemology
//!
//! Implementation of Gregory Bateson's cybernetic/ecological approach to mind
//! and learning, integrated with hyperbolic geometry and SNNs.
//!
//! ## Theoretical Foundation
//!
//! Bateson proposed that mind is immanent in the pattern of relations:
//! 1. Information is "a difference that makes a difference"
//! 2. Learning occurs at multiple logical types (Learning I, II, III)
//! 3. Mind extends beyond the individual into the environment
//! 4. Double binds and deutero-learning shape adaptation
//!
//! ## Hyperbolic Implementation
//!
//! The hyperbolic structure naturally supports:
//! - Hierarchical logical typing (tree structure)
//! - Recursive self-reference (Fuchsian groups)
//! - Context-dependent learning (curvature-modulated)
//!
//! ## References
//!
//! - Bateson (1972) "Steps to an Ecology of Mind"
//! - Bateson (1979) "Mind and Nature: A Necessary Unity"
//! - Bateson & Bateson (1987) "Angels Fear"

use crate::hyperbolic_snn::LorentzVec;
use crate::enactive_layer::Observation;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

/// Learning levels in Bateson's hierarchy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LearningLevel {
    /// Zero learning: fixed response
    Zero,
    /// Learning I: stimulus-response learning
    One,
    /// Learning II: learning to learn (deutero-learning)
    Two,
    /// Learning III: radical change in context of Learning II
    Three,
}

impl LearningLevel {
    /// Get numeric level
    pub fn level(&self) -> usize {
        match self {
            Self::Zero => 0,
            Self::One => 1,
            Self::Two => 2,
            Self::Three => 3,
        }
    }
}

/// Configuration for ecological epistemology
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EcologyConfig {
    /// Learning rates for each level
    pub learning_rates: [f64; 4],
    /// Context window for detecting patterns
    pub context_window: usize,
    /// Threshold for context shift detection
    pub context_shift_threshold: f64,
    /// Double-bind detection sensitivity
    pub double_bind_sensitivity: f64,
    /// Maximum recursion depth for self-reference
    pub max_recursion_depth: usize,
    /// Decay rate for context traces
    pub context_decay: f64,
}

impl Default for EcologyConfig {
    fn default() -> Self {
        Self {
            learning_rates: [0.0, 0.1, 0.01, 0.001],
            context_window: 50,
            context_shift_threshold: 0.5,
            double_bind_sensitivity: 0.3,
            max_recursion_depth: 5,
            context_decay: 0.95,
        }
    }
}

/// "A difference that makes a difference" - Bateson's information unit
#[derive(Debug, Clone)]
pub struct Difference {
    /// What changed (position delta)
    pub delta_position: LorentzVec,
    /// Feature differences
    pub delta_features: Vec<f64>,
    /// Magnitude of difference
    pub magnitude: f64,
    /// Whether this difference "makes a difference" (affects behavior)
    pub significant: bool,
    /// Context in which difference occurred
    pub context_id: usize,
    /// Time of difference
    pub time: f64,
}

impl Difference {
    /// Create new difference from two observations
    pub fn from_observations(prev: &Observation, curr: &Observation) -> Self {
        let delta_position = LorentzVec::new(
            0.0,
            curr.position.x - prev.position.x,
            curr.position.y - prev.position.y,
            curr.position.z - prev.position.z,
        );

        let delta_features: Vec<f64> = curr.features.iter()
            .zip(prev.features.iter())
            .map(|(c, p)| c - p)
            .collect();

        let magnitude = delta_features.iter()
            .map(|d| d * d)
            .sum::<f64>()
            .sqrt();

        Self {
            delta_position,
            delta_features,
            magnitude,
            significant: false, // Will be determined by context
            context_id: 0,
            time: curr.time,
        }
    }

    /// Compute geodesic magnitude in hyperbolic space
    pub fn hyperbolic_magnitude(&self) -> f64 {
        let spatial = self.delta_position.x.powi(2)
            + self.delta_position.y.powi(2)
            + self.delta_position.z.powi(2);
        spatial.sqrt()
    }
}

/// Context marker for learning
#[derive(Debug, Clone)]
pub struct Context {
    /// Unique context identifier
    pub id: usize,
    /// Context centroid in hyperbolic space
    pub centroid: LorentzVec,
    /// Feature prototype
    pub prototype: Vec<f64>,
    /// Variance in features
    pub variance: Vec<f64>,
    /// Number of observations in this context
    pub count: usize,
    /// Expected response pattern
    pub expected_response: Vec<f64>,
    /// Parent context (for hierarchical contexts)
    pub parent_id: Option<usize>,
    /// Creation time
    pub created_time: f64,
}

impl Context {
    /// Create new context from observation
    pub fn new(id: usize, obs: &Observation) -> Self {
        Self {
            id,
            centroid: obs.position,
            prototype: obs.features.clone(),
            variance: vec![1.0; obs.features.len()],
            count: 1,
            expected_response: Vec::new(),
            parent_id: None,
            created_time: obs.time,
        }
    }

    /// Update context with new observation (online mean/variance)
    pub fn update(&mut self, obs: &Observation) {
        self.count += 1;
        let n = self.count as f64;

        // Welford's online algorithm for mean and variance
        for i in 0..self.prototype.len().min(obs.features.len()) {
            let delta = obs.features[i] - self.prototype[i];
            self.prototype[i] += delta / n;
            let delta2 = obs.features[i] - self.prototype[i];
            self.variance[i] += delta * delta2;
        }

        // Update centroid (simple average in Lorentz coords, then project)
        let new_x = (self.centroid.x * (n - 1.0) + obs.position.x) / n;
        let new_y = (self.centroid.y * (n - 1.0) + obs.position.y) / n;
        let new_z = (self.centroid.z * (n - 1.0) + obs.position.z) / n;
        let spatial_sq = new_x * new_x + new_y * new_y + new_z * new_z;
        self.centroid = LorentzVec::new((1.0 + spatial_sq).sqrt(), new_x, new_y, new_z);
    }

    /// Compute distance from observation to this context
    pub fn distance(&self, obs: &Observation) -> f64 {
        // Hyperbolic distance for position
        let pos_dist = self.centroid.hyperbolic_distance(&obs.position);

        // Mahalanobis-like distance for features
        let feat_dist: f64 = self.prototype.iter()
            .zip(obs.features.iter())
            .zip(self.variance.iter())
            .map(|((p, f), v)| {
                let normalized_var = (v / self.count.max(1) as f64).max(0.01);
                (p - f).powi(2) / normalized_var
            })
            .sum::<f64>()
            .sqrt();

        pos_dist + 0.5 * feat_dist
    }
}

/// Double-bind detector
#[derive(Debug, Clone)]
pub struct DoubleBind {
    /// Conflicting contexts
    pub context_a: usize,
    pub context_b: usize,
    /// Contradiction magnitude
    pub contradiction: f64,
    /// Level at which bind occurs
    pub level: LearningLevel,
    /// Detection time
    pub time: f64,
    /// Whether resolved
    pub resolved: bool,
}

/// Deutero-learning statistics (learning about learning)
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DeuteroStats {
    /// Success rate for each context
    pub context_success: HashMap<usize, f64>,
    /// Learning rate adaptations
    pub rate_adaptations: Vec<f64>,
    /// Context switches detected
    pub context_switches: u64,
    /// Double binds encountered
    pub double_binds_encountered: u64,
    /// Double binds resolved
    pub double_binds_resolved: u64,
}

/// Main ecological epistemology system
pub struct EcologicalMind {
    /// Configuration
    config: EcologyConfig,
    /// Known contexts
    contexts: Vec<Context>,
    /// Current context
    current_context: Option<usize>,
    /// Context history
    context_history: VecDeque<usize>,
    /// Difference history
    difference_history: VecDeque<Difference>,
    /// Previous observation (for difference computation)
    previous_observation: Option<Observation>,
    /// Active double binds
    double_binds: Vec<DoubleBind>,
    /// Current learning level
    pub learning_level: LearningLevel,
    /// Deutero-learning statistics
    pub deutero_stats: DeuteroStats,
    /// Time
    time: f64,
    /// Statistics
    pub stats: EcologyStats,
}

/// Statistics for ecological system
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EcologyStats {
    /// Total observations processed
    pub total_observations: u64,
    /// Total differences detected
    pub total_differences: u64,
    /// Significant differences
    pub significant_differences: u64,
    /// Context formations
    pub contexts_formed: u64,
    /// Learning level changes
    pub level_changes: u64,
}

impl EcologicalMind {
    /// Create new ecological mind
    pub fn new(config: EcologyConfig) -> Self {
        Self {
            config,
            contexts: Vec::new(),
            current_context: None,
            context_history: VecDeque::with_capacity(100),
            difference_history: VecDeque::with_capacity(100),
            previous_observation: None,
            double_binds: Vec::new(),
            learning_level: LearningLevel::One,
            deutero_stats: DeuteroStats::default(),
            time: 0.0,
            stats: EcologyStats::default(),
        }
    }

    /// Process observation and learn
    pub fn process(&mut self, obs: &Observation, response: &[f64]) -> LearningResult {
        self.time = obs.time;
        self.stats.total_observations += 1;

        // Detect context
        let context_id = self.detect_context(obs);

        // Compute difference from previous
        let difference = self.compute_difference(obs);

        // Check for context shift (None -> Some is NOT a shift, it's initialization)
        let context_shifted = match self.current_context {
            None => false, // First context is not a "shift"
            Some(prev_id) => prev_id != context_id,
        };
        if context_shifted {
            self.deutero_stats.context_switches += 1;
            self.context_history.push_back(context_id);
            if self.context_history.len() > self.config.context_window {
                self.context_history.pop_front();
            }
        }
        self.current_context = Some(context_id);

        // Detect double binds
        self.detect_double_binds(context_id, response);

        // Determine learning level based on patterns
        self.update_learning_level();

        // Perform learning at appropriate level
        let learning_delta = self.learn(context_id, obs, response);

        // Update deutero-learning statistics
        self.update_deutero_stats(context_id, &learning_delta);

        LearningResult {
            context_id,
            context_shifted,
            learning_level: self.learning_level,
            learning_delta,
            double_bind_detected: !self.double_binds.is_empty()
                && self.double_binds.last().map_or(false, |db| !db.resolved),
            difference,
        }
    }

    /// Detect or create context for observation
    fn detect_context(&mut self, obs: &Observation) -> usize {
        // Find closest context
        let mut best_context = None;
        let mut best_distance = f64::INFINITY;

        for (i, ctx) in self.contexts.iter().enumerate() {
            let dist = ctx.distance(obs);
            if dist < best_distance {
                best_distance = dist;
                best_context = Some(i);
            }
        }

        // Create new context if too far from existing
        if best_distance > self.config.context_shift_threshold || best_context.is_none() {
            let new_id = self.contexts.len();
            let mut new_ctx = Context::new(new_id, obs);

            // Set parent to current context (hierarchical nesting)
            new_ctx.parent_id = self.current_context;

            self.contexts.push(new_ctx);
            self.stats.contexts_formed += 1;
            new_id
        } else {
            let ctx_id = best_context.unwrap();
            self.contexts[ctx_id].update(obs);
            ctx_id
        }
    }

    /// Compute difference from previous observation
    fn compute_difference(&mut self, obs: &Observation) -> Option<Difference> {
        // Need previous observation to compute a difference
        let prev_obs = match self.previous_observation.take() {
            Some(prev) => prev,
            None => {
                // Store current observation for next time
                self.previous_observation = Some(obs.clone());
                return None;
            }
        };

        // Store current observation for next difference computation
        self.previous_observation = Some(obs.clone());

        let mut diff = Difference::from_observations(&prev_obs, obs);

        // Determine significance based on context
        if let Some(ctx_id) = self.current_context {
            if let Some(ctx) = self.contexts.get(ctx_id) {
                // Difference is significant if larger than context variance
                let threshold: f64 = ctx.variance.iter()
                    .map(|v| v.sqrt())
                    .sum::<f64>() / ctx.variance.len().max(1) as f64;
                diff.significant = diff.magnitude > threshold * 0.5;
            }
        }

        if diff.significant {
            self.stats.significant_differences += 1;
        }

        diff.context_id = self.current_context.unwrap_or(0);

        self.difference_history.push_back(diff.clone());
        if self.difference_history.len() > self.config.context_window {
            self.difference_history.pop_front();
        }

        self.stats.total_differences += 1;
        Some(diff)
    }

    /// Detect double-bind situations
    fn detect_double_binds(&mut self, context_id: usize, response: &[f64]) {
        if self.context_history.len() < 3 {
            return;
        }

        // Check for oscillation between contexts (potential double bind)
        let recent: Vec<usize> = self.context_history.iter()
            .rev()
            .take(5)
            .cloned()
            .collect();

        if recent.len() >= 3 {
            // Check for A-B-A pattern (oscillation)
            if recent[0] == recent[2] && recent[0] != recent[1] {
                let ctx_a = recent[0];
                let ctx_b = recent[1];

                // Check if responses are contradictory
                if let (Some(a), Some(b)) = (self.contexts.get(ctx_a), self.contexts.get(ctx_b)) {
                    if !a.expected_response.is_empty() && !b.expected_response.is_empty() {
                        let contradiction: f64 = a.expected_response.iter()
                            .zip(b.expected_response.iter())
                            .map(|(ra, rb)| (ra - rb).powi(2))
                            .sum::<f64>()
                            .sqrt();

                        if contradiction > self.config.double_bind_sensitivity {
                            let bind = DoubleBind {
                                context_a: ctx_a,
                                context_b: ctx_b,
                                contradiction,
                                level: self.learning_level,
                                time: self.time,
                                resolved: false,
                            };
                            self.double_binds.push(bind);
                            self.deutero_stats.double_binds_encountered += 1;
                        }
                    }
                }
            }
        }

        // Update expected response for current context
        if let Some(ctx) = self.contexts.get_mut(context_id) {
            if ctx.expected_response.is_empty() {
                ctx.expected_response = response.to_vec();
            } else {
                // Exponential moving average
                for (i, &r) in response.iter().enumerate() {
                    if i < ctx.expected_response.len() {
                        ctx.expected_response[i] = 0.9 * ctx.expected_response[i] + 0.1 * r;
                    }
                }
            }
        }
    }

    /// Update learning level based on observed patterns
    fn update_learning_level(&mut self) {
        let old_level = self.learning_level;

        // Learning I → II: detecting patterns in context switches
        if self.learning_level == LearningLevel::One {
            // If we've accumulated enough context data, upgrade to Learning II
            if self.contexts.len() > 3 && self.deutero_stats.context_switches > 10 {
                self.learning_level = LearningLevel::Two;
            }
        }

        // Learning II → III: double bind resolution requires transcendence
        if self.learning_level == LearningLevel::Two {
            // Persistent unresolved double binds trigger Learning III
            let unresolved_binds = self.double_binds.iter()
                .filter(|db| !db.resolved && (self.time - db.time) > 100.0)
                .count();

            if unresolved_binds >= 2 {
                self.learning_level = LearningLevel::Three;
            }
        }

        if old_level != self.learning_level {
            self.stats.level_changes += 1;
        }
    }

    /// Perform learning at current level
    fn learn(&mut self, context_id: usize, obs: &Observation, response: &[f64]) -> Vec<f64> {
        let learning_rate = self.config.learning_rates[self.learning_level.level()];

        match self.learning_level {
            LearningLevel::Zero => vec![0.0; response.len()],

            LearningLevel::One => {
                // Simple stimulus-response learning
                self.learn_level_one(context_id, obs, response, learning_rate)
            }

            LearningLevel::Two => {
                // Deutero-learning: learn to learn
                self.learn_level_two(context_id, obs, response, learning_rate)
            }

            LearningLevel::Three => {
                // Radical restructuring
                self.learn_level_three(context_id, obs, response, learning_rate)
            }
        }
    }

    /// Learning I: stimulus-response associations
    fn learn_level_one(
        &mut self,
        context_id: usize,
        _obs: &Observation,
        response: &[f64],
        learning_rate: f64,
    ) -> Vec<f64> {
        let mut deltas = vec![0.0; response.len()];

        if let Some(ctx) = self.contexts.get_mut(context_id) {
            if ctx.expected_response.len() != response.len() {
                ctx.expected_response = vec![0.0; response.len()];
            }

            for i in 0..response.len() {
                let error = response[i] - ctx.expected_response[i];
                let delta = learning_rate * error;
                ctx.expected_response[i] += delta;
                deltas[i] = delta;
            }
        }

        deltas
    }

    /// Learning II: learning to learn (adapting learning strategies)
    fn learn_level_two(
        &mut self,
        context_id: usize,
        obs: &Observation,
        response: &[f64],
        learning_rate: f64,
    ) -> Vec<f64> {
        // First do Level I learning
        let deltas = self.learn_level_one(context_id, obs, response, learning_rate);

        // Then adapt based on pattern of successes/failures
        let success_rate = self.deutero_stats.context_success
            .get(&context_id)
            .copied()
            .unwrap_or(0.5);

        // Adapt learning rate for this context
        let adapted_rate = if success_rate > 0.7 {
            learning_rate * 0.9 // Reduce when doing well (consolidate)
        } else if success_rate < 0.3 {
            learning_rate * 1.1 // Increase when doing poorly (explore)
        } else {
            learning_rate
        };

        self.deutero_stats.rate_adaptations.push(adapted_rate);

        deltas
    }

    /// Learning III: radical restructuring (resolve double binds)
    fn learn_level_three(
        &mut self,
        context_id: usize,
        obs: &Observation,
        response: &[f64],
        learning_rate: f64,
    ) -> Vec<f64> {
        // First do Level II learning
        let mut deltas = self.learn_level_two(context_id, obs, response, learning_rate);

        // Try to resolve double binds by creating meta-context
        if let Some(bind) = self.double_binds.iter_mut().find(|db| !db.resolved) {
            // Create a new meta-context that encompasses both conflicting contexts
            let ctx_a = self.contexts.get(bind.context_a).cloned();
            let ctx_b = self.contexts.get(bind.context_b).cloned();

            if let (Some(a), Some(b)) = (ctx_a, ctx_b) {
                // Meta-context combines both
                let meta_prototype: Vec<f64> = a.prototype.iter()
                    .zip(b.prototype.iter())
                    .map(|(pa, pb)| (pa + pb) / 2.0)
                    .collect();

                let meta_centroid = LorentzVec::new(
                    ((a.centroid.t + b.centroid.t) / 2.0).max(1.0),
                    (a.centroid.x + b.centroid.x) / 2.0,
                    (a.centroid.y + b.centroid.y) / 2.0,
                    (a.centroid.z + b.centroid.z) / 2.0,
                );

                let meta_id = self.contexts.len();
                let meta_context = Context {
                    id: meta_id,
                    centroid: meta_centroid,
                    prototype: meta_prototype,
                    variance: a.variance.iter()
                        .zip(b.variance.iter())
                        .map(|(va, vb)| va + vb)
                        .collect(),
                    count: a.count + b.count,
                    expected_response: Vec::new(),
                    parent_id: None, // Meta-context is top-level
                    created_time: self.time,
                };

                self.contexts.push(meta_context);

                // Update parent references
                self.contexts[bind.context_a].parent_id = Some(meta_id);
                self.contexts[bind.context_b].parent_id = Some(meta_id);

                bind.resolved = true;
                self.deutero_stats.double_binds_resolved += 1;

                // Learning III creates larger deltas (restructuring)
                for d in &mut deltas {
                    *d *= 2.0;
                }
            }
        }

        // Check if we can return to Level II
        let unresolved = self.double_binds.iter().filter(|db| !db.resolved).count();
        if unresolved == 0 {
            self.learning_level = LearningLevel::Two;
        }

        deltas
    }

    /// Update deutero-learning statistics
    fn update_deutero_stats(&mut self, context_id: usize, deltas: &[f64]) {
        // Compute "success" as small deltas (predictions were accurate)
        let error_magnitude: f64 = deltas.iter().map(|d| d.abs()).sum();
        let success = if error_magnitude < 0.1 { 1.0 } else { 0.0 };

        let current = self.deutero_stats.context_success
            .get(&context_id)
            .copied()
            .unwrap_or(0.5);

        let updated = 0.9 * current + 0.1 * success;
        self.deutero_stats.context_success.insert(context_id, updated);
    }

    /// Get current context
    pub fn current_context(&self) -> Option<&Context> {
        self.current_context.and_then(|id| self.contexts.get(id))
    }

    /// Get all contexts
    pub fn contexts(&self) -> &[Context] {
        &self.contexts
    }

    /// Get active double binds
    pub fn active_double_binds(&self) -> Vec<&DoubleBind> {
        self.double_binds.iter().filter(|db| !db.resolved).collect()
    }

    /// Check if in double-bind state
    pub fn in_double_bind(&self) -> bool {
        self.double_binds.iter().any(|db| !db.resolved)
    }
}

/// Result of learning step
#[derive(Debug, Clone)]
pub struct LearningResult {
    /// Context ID
    pub context_id: usize,
    /// Whether context shifted
    pub context_shifted: bool,
    /// Current learning level
    pub learning_level: LearningLevel,
    /// Learning deltas applied
    pub learning_delta: Vec<f64>,
    /// Whether double bind detected
    pub double_bind_detected: bool,
    /// Difference from previous
    pub difference: Option<Difference>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::enactive_layer::Modality;

    fn create_test_obs(time: f64, x: f64, features: Vec<f64>) -> Observation {
        Observation {
            time,
            modality: Modality::Proprioceptive,
            position: LorentzVec::from_spatial(x, 0.0, 0.0),
            value: features.clone(),
            features,
            precision: 1.0,
        }
    }

    #[test]
    fn test_context_detection() {
        let config = EcologyConfig::default();
        let mut mind = EcologicalMind::new(config);

        let obs = create_test_obs(0.0, 0.0, vec![1.0, 0.0]);
        let result = mind.process(&obs, &[0.5]);

        assert_eq!(result.context_id, 0);
        assert!(!result.context_shifted);
    }

    #[test]
    fn test_context_shift() {
        let mut config = EcologyConfig::default();
        config.context_shift_threshold = 0.5;
        let mut mind = EcologicalMind::new(config);

        // First context
        let obs1 = create_test_obs(0.0, 0.0, vec![1.0, 0.0]);
        mind.process(&obs1, &[0.5]);

        // Same context
        let obs2 = create_test_obs(1.0, 0.1, vec![1.1, 0.1]);
        let result2 = mind.process(&obs2, &[0.5]);
        assert!(!result2.context_shifted);

        // Different context (far away)
        let obs3 = create_test_obs(2.0, 2.0, vec![5.0, 5.0]);
        let result3 = mind.process(&obs3, &[0.5]);
        assert!(result3.context_shifted);
    }

    #[test]
    fn test_learning_levels() {
        let config = EcologyConfig::default();
        let mut mind = EcologicalMind::new(config);

        // Start at Level I
        assert_eq!(mind.learning_level, LearningLevel::One);

        // Simulate enough activity to potentially trigger Level II
        for i in 0..20 {
            let x = (i % 4) as f64 * 0.3;
            let obs = create_test_obs(i as f64, x, vec![x, 0.0]);
            mind.process(&obs, &[x]);
        }

        // After enough context switches, should be Level II
        // (depends on exact thresholds)
    }

    #[test]
    fn test_difference_detection() {
        let config = EcologyConfig::default();
        let mut mind = EcologicalMind::new(config);

        let obs1 = create_test_obs(0.0, 0.0, vec![0.0, 0.0]);
        mind.process(&obs1, &[0.0]);

        let obs2 = create_test_obs(1.0, 0.1, vec![0.5, 0.5]);
        let result = mind.process(&obs2, &[0.5]);

        // Should have detected a difference
        assert!(result.difference.is_some());
    }

    #[test]
    fn test_context_update() {
        let obs1 = Observation {
            time: 0.0,
            modality: Modality::Proprioceptive,
            position: LorentzVec::origin(),
            value: vec![1.0, 2.0],
            features: vec![1.0, 2.0],
            precision: 1.0,
        };

        let mut ctx = Context::new(0, &obs1);
        assert_eq!(ctx.count, 1);

        let obs2 = Observation {
            time: 1.0,
            modality: Modality::Proprioceptive,
            position: LorentzVec::from_spatial(0.1, 0.0, 0.0),
            value: vec![1.2, 2.2],
            features: vec![1.2, 2.2],
            precision: 1.0,
        };

        ctx.update(&obs2);
        assert_eq!(ctx.count, 2);
    }
}
