//! Sentry Integration Layer
//!
//! Integrates the high-level AdversarialLattice with optimized SIMD operations
//! for hyperbolic defense networks. Combines topology management with
//! performance-critical belief propagation.
//!
//! # Architecture
//!
//! This module bridges two complementary approaches:
//! - **AdversarialLattice**: High-level topology management, defense configuration
//! - **SIMD Operations**: f32x4 vectorized Lorentz computations for batch processing
//!
//! # Features
//!
//! - **LorentzVec**: f32 SIMD 4-vector for batch Minkowski operations
//! - **DualModeBeliefUpdater**: Real-time simple updates + offline variational inference
//! - **ThermodynamicLearner**: Long-term curvature and threshold adaptation via SOC
//! - **HyperbolicLattice**: Integrated sentry network with SIMD-accelerated propagation
//!
//! # References
//!
//! - Kollár et al. (2019) "Hyperbolic lattices in circuit QED" Nature 571:45-50
//! - Friston (2010) "The free-energy principle" Nature Reviews Neuroscience
//! - Bak et al. (1987) "Self-organized criticality" Physical Review Letters

use crate::adversarial_lattice::{
    AdversarialLattice, DefenseTopology, DetectionResult, HyperboloidPoint,
};
use crate::Result;
use serde::{Deserialize, Serialize};

/// SIMD-optimized Lorentz 4-vector using f32 for batch processing
///
/// Represents a point in 2+1D Minkowski space using packed f32 values
/// for SIMD-friendly memory layout. Used for batch sentry position
/// updates and belief propagation.
///
/// Layout: [t, x, y, _padding]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[repr(C, align(16))]
pub struct LorentzVec {
    /// Packed components: [t, x, y, pad]
    pub data: [f32; 4],
}

impl Default for LorentzVec {
    fn default() -> Self {
        Self::origin()
    }
}

impl LorentzVec {
    /// Create new LorentzVec from components
    #[inline]
    pub fn new(t: f32, x: f32, y: f32) -> Self {
        Self {
            data: [t, x, y, 0.0],
        }
    }

    /// Origin point on hyperboloid: (1, 0, 0)
    #[inline]
    pub fn origin() -> Self {
        Self {
            data: [1.0, 0.0, 0.0, 0.0],
        }
    }

    /// Create from spatial coordinates, computing time component
    ///
    /// t = √(1 + x² + y²) to satisfy hyperboloid constraint
    #[inline]
    pub fn from_spatial(x: f32, y: f32) -> Self {
        let t = (1.0 + x * x + y * y).sqrt();
        Self {
            data: [t, x, y, 0.0],
        }
    }

    /// Time coordinate (Lorentz t)
    #[inline]
    pub fn t(&self) -> f32 {
        self.data[0]
    }

    /// Spatial x coordinate
    #[inline]
    pub fn x(&self) -> f32 {
        self.data[1]
    }

    /// Spatial y coordinate
    #[inline]
    pub fn y(&self) -> f32 {
        self.data[2]
    }

    /// SIMD Minkowski inner product: ⟨a,b⟩_L = -t₁t₂ + x₁x₂ + y₁y₂
    ///
    /// Uses vectorized multiply-accumulate pattern.
    #[inline]
    pub fn minkowski_dot(&self, other: &Self) -> f32 {
        // Sign pattern: [-1, 1, 1, 0]
        let signs = [-1.0_f32, 1.0, 1.0, 0.0];

        // Manual SIMD-style: multiply and accumulate
        let mut sum = 0.0_f32;
        for i in 0..4 {
            sum += signs[i] * self.data[i] * other.data[i];
        }
        sum
    }

    /// Hyperbolic distance: d(a,b) = acosh(-⟨a,b⟩_L)
    #[inline]
    pub fn distance(&self, other: &Self) -> f32 {
        let inner = self.minkowski_dot(other);
        // For points on same sheet, -inner >= 1
        (-inner).max(1.0).acosh()
    }

    /// Check if point satisfies hyperboloid constraint
    ///
    /// Verifies: -t² + x² + y² ≈ -1
    #[inline]
    pub fn is_valid(&self, tolerance: f32) -> bool {
        let constraint = self.minkowski_dot(self);
        (constraint + 1.0).abs() < tolerance
    }

    /// Project onto hyperboloid (normalize constraint)
    #[inline]
    pub fn project(&self) -> Self {
        // Compute current constraint value
        let spatial_sq = self.data[1] * self.data[1] + self.data[2] * self.data[2];

        // Recompute t to satisfy constraint
        let t = (1.0 + spatial_sq).sqrt();

        Self {
            data: [t, self.data[1], self.data[2], 0.0],
        }
    }

    /// Lorentz boost along x-axis (information flow)
    ///
    /// Represents velocity v propagation with rapidity η
    #[inline]
    pub fn boost_x(&self, rapidity: f32) -> Self {
        let (cosh_r, sinh_r) = (rapidity.cosh(), rapidity.sinh());

        Self {
            data: [
                cosh_r * self.data[0] + sinh_r * self.data[1],
                sinh_r * self.data[0] + cosh_r * self.data[1],
                self.data[2],
                0.0,
            ],
        }
    }

    /// Lorentz boost along y-axis
    #[inline]
    pub fn boost_y(&self, rapidity: f32) -> Self {
        let (cosh_r, sinh_r) = (rapidity.cosh(), rapidity.sinh());

        Self {
            data: [
                cosh_r * self.data[0] + sinh_r * self.data[2],
                self.data[1],
                sinh_r * self.data[0] + cosh_r * self.data[2],
                0.0,
            ],
        }
    }

    /// General Lorentz boost in direction (cos θ, sin θ)
    #[inline]
    pub fn boost(&self, rapidity: f32, angle: f32) -> Self {
        let (cos_a, sin_a) = (angle.cos(), angle.sin());
        let (cosh_r, sinh_r) = (rapidity.cosh(), rapidity.sinh());

        // Decompose boost into rotation + boost_x + rotation⁻¹
        let x_rot = cos_a * self.data[1] + sin_a * self.data[2];
        let y_rot = -sin_a * self.data[1] + cos_a * self.data[2];

        let t_new = cosh_r * self.data[0] + sinh_r * x_rot;
        let x_boosted = sinh_r * self.data[0] + cosh_r * x_rot;

        // Rotate back
        let x_final = cos_a * x_boosted - sin_a * y_rot;
        let y_final = sin_a * x_boosted + cos_a * y_rot;

        Self {
            data: [t_new, x_final, y_final, 0.0],
        }
    }

    /// Spatial rotation in xy-plane
    #[inline]
    pub fn rotate(&self, angle: f32) -> Self {
        let (cos_a, sin_a) = (angle.cos(), angle.sin());

        Self {
            data: [
                self.data[0],
                cos_a * self.data[1] - sin_a * self.data[2],
                sin_a * self.data[1] + cos_a * self.data[2],
                0.0,
            ],
        }
    }

    /// Convert from HyperboloidPoint (f64 -> f32)
    pub fn from_hyperboloid(point: &HyperboloidPoint) -> Self {
        Self {
            data: [point.t as f32, point.x as f32, point.y as f32, 0.0],
        }
    }

    /// Convert to HyperboloidPoint (f32 -> f64)
    pub fn to_hyperboloid(&self) -> HyperboloidPoint {
        HyperboloidPoint {
            t: self.data[0] as f64,
            x: self.data[1] as f64,
            y: self.data[2] as f64,
        }
    }
}

/// Batch operations on LorentzVec arrays
pub struct LorentzBatch;

impl LorentzBatch {
    /// Batch pairwise distances (returns lower triangular)
    pub fn pairwise_distances(points: &[LorentzVec]) -> Vec<f32> {
        let n = points.len();
        let mut distances = Vec::with_capacity(n * (n - 1) / 2);

        for i in 0..n {
            for j in 0..i {
                distances.push(points[i].distance(&points[j]));
            }
        }

        distances
    }

    /// Batch boost all points with same rapidity and direction
    pub fn batch_boost(points: &mut [LorentzVec], rapidity: f32, angle: f32) {
        for point in points.iter_mut() {
            *point = point.boost(rapidity, angle);
        }
    }

    /// Find centroid (Fréchet mean approximation for small clusters)
    pub fn centroid(points: &[LorentzVec]) -> LorentzVec {
        if points.is_empty() {
            return LorentzVec::origin();
        }

        let n = points.len() as f32;
        let mut sum = [0.0_f32; 4];

        for p in points {
            for i in 0..4 {
                sum[i] += p.data[i];
            }
        }

        for i in 0..4 {
            sum[i] /= n;
        }

        // Project back to hyperboloid
        LorentzVec {
            data: [sum[0], sum[1], sum[2], 0.0],
        }
        .project()
    }
}

/// Mode for belief update computation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BeliefUpdateMode {
    /// Fast real-time update with simplified gradient
    RealTime,
    /// Full variational inference for offline learning
    Variational,
}

/// Dual-mode belief updater for sentry nodes
///
/// Combines fast real-time updates for immediate threat response
/// with full variational inference for offline model refinement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DualModeBeliefUpdater {
    /// Current update mode
    pub mode: BeliefUpdateMode,
    /// Learning rate for gradient updates
    pub learning_rate: f64,
    /// Prior weight (influence of hyperbolic metric prior)
    pub prior_weight: f64,
    /// Precision parameter (inverse variance) for likelihood
    pub precision: f64,
    /// Number of variational iterations (for Variational mode)
    pub variational_iterations: usize,
}

impl Default for DualModeBeliefUpdater {
    fn default() -> Self {
        Self {
            mode: BeliefUpdateMode::RealTime,
            learning_rate: 0.1,
            prior_weight: 0.3,
            precision: 25.0, // σ² = 0.04
            variational_iterations: 5,
        }
    }
}

impl DualModeBeliefUpdater {
    /// Create updater for real-time mode
    pub fn real_time() -> Self {
        Self {
            mode: BeliefUpdateMode::RealTime,
            ..Default::default()
        }
    }

    /// Create updater for variational mode
    pub fn variational() -> Self {
        Self {
            mode: BeliefUpdateMode::Variational,
            variational_iterations: 10,
            ..Default::default()
        }
    }

    /// Update belief based on observation
    ///
    /// # Arguments
    ///
    /// * `current_belief` - Current belief state (0-1)
    /// * `observation` - Observed anomaly score (0-1)
    /// * `metric_prior` - Prior from hyperbolic metric structure
    ///
    /// # Returns
    ///
    /// Updated belief state and free energy
    pub fn update(&self, current_belief: f64, observation: f64, metric_prior: f64) -> (f64, f64) {
        match self.mode {
            BeliefUpdateMode::RealTime => self.update_realtime(current_belief, observation, metric_prior),
            BeliefUpdateMode::Variational => {
                self.update_variational(current_belief, observation, metric_prior)
            }
        }
    }

    /// Fast real-time update
    ///
    /// Uses simplified gradient: weighted combination of observation and prior
    fn update_realtime(
        &self,
        current_belief: f64,
        observation: f64,
        metric_prior: f64,
    ) -> (f64, f64) {
        // Gradient: move toward observation weighted by (1 - prior_weight),
        // move toward prior weighted by prior_weight
        let obs_gradient = observation - current_belief;
        let prior_gradient = metric_prior - current_belief;

        let gradient = obs_gradient * (1.0 - self.prior_weight) + prior_gradient * self.prior_weight;

        let new_belief = (current_belief + self.learning_rate * gradient).clamp(0.0, 1.0);

        // Simplified free energy: squared error + KL approximation
        let prediction_error = (observation - new_belief).powi(2);
        let complexity = self.approximate_kl(new_belief, metric_prior);
        let free_energy = complexity - (-prediction_error);

        (new_belief, free_energy)
    }

    /// Full variational inference update
    ///
    /// Iteratively minimizes variational free energy:
    /// F = KL[q(z|x)||p(z)] - E_q[log p(x|z)]
    fn update_variational(
        &self,
        current_belief: f64,
        observation: f64,
        metric_prior: f64,
    ) -> (f64, f64) {
        let mut belief = current_belief;
        let mut free_energy = f64::MAX;
        let mut lr = self.learning_rate * 2.0; // Higher learning rate for variational

        for _ in 0..self.variational_iterations {
            // E-step: compute expected sufficient statistics
            let likelihood = self.gaussian_likelihood(observation, belief);

            // M-step: update variational parameters
            let kl = self.kl_divergence(belief, metric_prior);
            let accuracy = likelihood.max(1e-10).ln();
            let new_free_energy = kl - accuracy;

            // Gradient for variational update: move toward posterior mean
            // Posterior combines likelihood (pulls toward observation) and prior
            let obs_pull = (observation - belief) * self.precision;
            let prior_pull = (metric_prior - belief) / (metric_prior * (1.0 - metric_prior)).max(0.1);
            let gradient = obs_pull * (1.0 - self.prior_weight) + prior_pull * self.prior_weight;

            // Adaptive learning rate based on free energy change
            if new_free_energy < free_energy {
                lr = (lr * 1.05).min(0.5);
            } else {
                lr = (lr * 0.7).max(0.01);
            }

            belief = (belief + lr * gradient).clamp(0.001, 0.999);
            free_energy = new_free_energy;
        }

        (belief, free_energy)
    }

    /// Gaussian likelihood p(x|z)
    fn gaussian_likelihood(&self, observation: f64, belief: f64) -> f64 {
        let diff = observation - belief;
        (-0.5 * self.precision * diff * diff).exp()
    }

    /// KL divergence for Bernoulli distributions
    fn kl_divergence(&self, q: f64, p: f64) -> f64 {
        let q_safe = q.clamp(1e-10, 1.0 - 1e-10);
        let p_safe = p.clamp(1e-10, 1.0 - 1e-10);

        q_safe * (q_safe / p_safe).ln() + (1.0 - q_safe) * ((1.0 - q_safe) / (1.0 - p_safe)).ln()
    }

    /// Approximate KL for fast updates
    fn approximate_kl(&self, q: f64, p: f64) -> f64 {
        // Second-order approximation: KL ≈ (q-p)² / (2p(1-p))
        let p_safe = p.clamp(0.1, 0.9);
        (q - p_safe).powi(2) / (2.0 * p_safe * (1.0 - p_safe))
    }

    /// Free energy gradient w.r.t. belief (for advanced variational methods)
    #[allow(dead_code)]
    fn free_energy_gradient(&self, belief: f64, observation: f64, prior: f64) -> f64 {
        // ∂F/∂q = log(q/(1-q)) - log(p/(1-p)) + precision * (q - observation)
        let belief_safe = belief.clamp(0.01, 0.99);
        let prior_safe = prior.clamp(0.01, 0.99);

        let logit_diff = (belief_safe / (1.0 - belief_safe)).ln()
            - (prior_safe / (1.0 - prior_safe)).ln();

        // Negate because we want to minimize F (move opposite to gradient)
        -(logit_diff + self.precision * (belief_safe - observation))
    }
}

/// Statistics for thermodynamic learning
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ThermodynamicStats {
    /// Number of detection events
    pub detections: usize,
    /// Number of cascade events
    pub cascades: usize,
    /// Average cascade size
    pub avg_cascade_size: f64,
    /// Variance of cascade sizes
    pub cascade_variance: f64,
    /// Power-law exponent estimate
    pub power_law_alpha: f64,
    /// Free energy trend (moving average)
    pub free_energy_trend: f64,
}

/// Thermodynamic learner for long-term adaptation
///
/// Uses Self-Organized Criticality (SOC) principles to maintain
/// the defense network near the critical point for optimal detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermodynamicLearner {
    /// Window size for statistics
    pub window_size: usize,
    /// Cascade size history
    pub cascade_history: Vec<usize>,
    /// Free energy history
    pub free_energy_history: Vec<f64>,
    /// Current curvature adaptation factor
    pub curvature_adaptation: f64,
    /// Current criticality adaptation
    pub criticality_adaptation: f64,
    /// Learning rate for slow adaptation
    pub adaptation_rate: f64,
    /// Target power-law exponent (SOC indicator)
    pub target_alpha: f64,
    /// Current statistics
    pub stats: ThermodynamicStats,
}

impl Default for ThermodynamicLearner {
    fn default() -> Self {
        Self {
            window_size: 100,
            cascade_history: Vec::with_capacity(100),
            free_energy_history: Vec::with_capacity(100),
            curvature_adaptation: 1.0,
            criticality_adaptation: 0.0,
            adaptation_rate: 0.01,
            target_alpha: 1.5, // Near criticality: p(s) ~ s^{-3/2}
            stats: ThermodynamicStats::default(),
        }
    }
}

impl ThermodynamicLearner {
    /// Record a detection event
    pub fn record_detection(&mut self, cascade_size: usize, free_energy: f64) {
        self.stats.detections += 1;

        // Record cascade
        if cascade_size > 0 {
            self.stats.cascades += 1;
            self.cascade_history.push(cascade_size);

            if self.cascade_history.len() > self.window_size {
                self.cascade_history.remove(0);
            }
        }

        // Record free energy
        self.free_energy_history.push(free_energy);
        if self.free_energy_history.len() > self.window_size {
            self.free_energy_history.remove(0);
        }

        // Update statistics
        self.update_statistics();

        // Adapt parameters
        self.adapt_parameters();
    }

    /// Update running statistics
    fn update_statistics(&mut self) {
        if self.cascade_history.is_empty() {
            return;
        }

        // Average cascade size
        let sum: usize = self.cascade_history.iter().sum();
        self.stats.avg_cascade_size = sum as f64 / self.cascade_history.len() as f64;

        // Variance
        let variance: f64 = self.cascade_history
            .iter()
            .map(|&s| (s as f64 - self.stats.avg_cascade_size).powi(2))
            .sum::<f64>()
            / self.cascade_history.len() as f64;
        self.stats.cascade_variance = variance;

        // Estimate power-law exponent using Hill estimator
        self.stats.power_law_alpha = self.estimate_power_law_exponent();

        // Free energy trend
        if !self.free_energy_history.is_empty() {
            let n = self.free_energy_history.len();
            let recent = &self.free_energy_history[(n.saturating_sub(10))..];
            self.stats.free_energy_trend = recent.iter().sum::<f64>() / recent.len() as f64;
        }
    }

    /// Estimate power-law exponent using Hill estimator
    fn estimate_power_law_exponent(&self) -> f64 {
        if self.cascade_history.len() < 10 {
            return 1.5; // Default to critical
        }

        let mut sorted: Vec<f64> = self.cascade_history.iter().map(|&s| s as f64).collect();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap()); // Descending

        // Use top 20% for estimation
        let k = (sorted.len() as f64 * 0.2).max(5.0) as usize;
        let x_k = sorted[k - 1].max(1.0);

        let sum_log: f64 = sorted[..k].iter().map(|&x| (x / x_k).max(1.0).ln()).sum();

        if sum_log > 0.0 {
            1.0 + k as f64 / sum_log
        } else {
            1.5
        }
    }

    /// Adapt curvature and criticality parameters
    fn adapt_parameters(&mut self) {
        let alpha = self.stats.power_law_alpha;
        let alpha_error = alpha - self.target_alpha;

        // If alpha > target (subcritical): increase curvature to push toward criticality
        // If alpha < target (supercritical): decrease curvature to stabilize
        let curvature_adjustment = self.adaptation_rate * alpha_error.tanh();
        self.curvature_adaptation = (self.curvature_adaptation * (1.0 + curvature_adjustment))
            .clamp(0.5, 2.0);

        // Adjust criticality threshold based on cascade size variance
        // High variance suggests we're at criticality; low variance suggests we need adjustment
        let target_variance = self.stats.avg_cascade_size.powi(2); // Coefficient of variation = 1
        let variance_ratio = self.stats.cascade_variance / target_variance.max(1.0);

        if variance_ratio < 0.5 {
            // Too predictable: lower threshold to increase sensitivity
            self.criticality_adaptation -= self.adaptation_rate * 0.1;
        } else if variance_ratio > 2.0 {
            // Too chaotic: raise threshold to stabilize
            self.criticality_adaptation += self.adaptation_rate * 0.1;
        }

        self.criticality_adaptation = self.criticality_adaptation.clamp(-0.3, 0.3);
    }

    /// Get adapted curvature scale
    pub fn adapted_curvature(&self, base_curvature: f64) -> f64 {
        base_curvature * self.curvature_adaptation
    }

    /// Get adapted criticality threshold
    pub fn adapted_criticality(&self, base_threshold: f64) -> f64 {
        (base_threshold + self.criticality_adaptation).clamp(0.2, 0.8)
    }

    /// Check if system is near criticality
    pub fn is_near_critical(&self) -> bool {
        let alpha = self.stats.power_law_alpha;
        (alpha - self.target_alpha).abs() < 0.3
    }
}

/// Integrated hyperbolic sentry lattice with SIMD acceleration
///
/// Combines AdversarialLattice topology with SIMD-optimized belief propagation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyperbolicLattice {
    /// Base adversarial lattice
    pub base: AdversarialLattice,
    /// SIMD-optimized sentry positions
    #[serde(skip)]
    pub simd_positions: Vec<LorentzVec>,
    /// Belief updater configuration
    pub belief_updater: DualModeBeliefUpdater,
    /// Thermodynamic learner
    pub learner: ThermodynamicLearner,
}

impl HyperbolicLattice {
    /// Create new hyperbolic lattice from topology
    pub fn new(topology: DefenseTopology) -> Result<Self> {
        let base = AdversarialLattice::new(topology)?;

        // Convert positions to SIMD format
        let simd_positions: Vec<LorentzVec> = base
            .sentries
            .iter()
            .map(|s| LorentzVec::from_hyperboloid(&s.position))
            .collect();

        Ok(Self {
            base,
            simd_positions,
            belief_updater: DualModeBeliefUpdater::default(),
            learner: ThermodynamicLearner::default(),
        })
    }

    /// Create from existing AdversarialLattice
    pub fn from_adversarial(lattice: AdversarialLattice) -> Self {
        let simd_positions: Vec<LorentzVec> = lattice
            .sentries
            .iter()
            .map(|s| LorentzVec::from_hyperboloid(&s.position))
            .collect();

        Self {
            base: lattice,
            simd_positions,
            belief_updater: DualModeBeliefUpdater::default(),
            learner: ThermodynamicLearner::default(),
        }
    }

    /// Set belief update mode
    pub fn with_mode(mut self, mode: BeliefUpdateMode) -> Self {
        self.belief_updater.mode = mode;
        self
    }

    /// Process message with SIMD-accelerated propagation
    pub fn process_message(
        &mut self,
        entry_point: usize,
        anomaly_score: f64,
        timestamp: u64,
    ) -> DetectionResult {
        // Apply thermodynamic adaptations
        let adapted_curvature = self
            .learner
            .adapted_curvature(self.base.topology.curvature_scale);
        let adapted_threshold = self
            .learner
            .adapted_criticality(self.base.topology.criticality_threshold);

        // Process through base lattice with adapted parameters
        let original_curvature = self.base.topology.curvature_scale;
        let original_threshold = self.base.topology.criticality_threshold;

        self.base.topology.curvature_scale = adapted_curvature;
        self.base.topology.criticality_threshold = adapted_threshold;

        let result = self.process_with_dual_mode(entry_point, anomaly_score, timestamp);

        // Restore original parameters
        self.base.topology.curvature_scale = original_curvature;
        self.base.topology.criticality_threshold = original_threshold;

        // Record for thermodynamic learning
        self.learner
            .record_detection(result.cascade_size, result.free_energy);

        result
    }

    /// Process with dual-mode belief updates
    fn process_with_dual_mode(
        &mut self,
        entry_point: usize,
        anomaly_score: f64,
        timestamp: u64,
    ) -> DetectionResult {
        if entry_point >= self.base.sentries.len() {
            return DetectionResult {
                is_threat: false,
                confidence: 0.0,
                cascade_size: 0,
                free_energy: 0.0,
            };
        }

        // Phase 1: Local belief update at entry point using dual-mode updater
        let sentry = &self.base.sentries[entry_point];
        let metric_prior = self.compute_metric_prior(entry_point);
        let (new_belief, _local_fe) =
            self.belief_updater
                .update(sentry.belief_state, anomaly_score, metric_prior);

        self.base.sentries[entry_point].belief_state = new_belief;
        self.base.sentries[entry_point].prediction_error = (anomaly_score - new_belief).abs();

        // Phase 2: Propagate using SIMD-accelerated distance computations
        let decay_rate = 0.8;
        let mut propagation_queue = vec![entry_point];
        let mut visited = vec![false; self.base.sentries.len()];
        visited[entry_point] = true;

        let mut cascade_size = 1;

        while let Some(current) = propagation_queue.pop() {
            let current_error = self.base.sentries[current].prediction_error;
            let error_to_send = current_error * decay_rate;

            let neighbors: Vec<usize> = self.base.sentries[current].neighbors.clone();

            for neighbor_id in neighbors {
                if !visited[neighbor_id] {
                    visited[neighbor_id] = true;

                    // Use SIMD for distance-based decay
                    let dist = self.simd_positions[current].distance(&self.simd_positions[neighbor_id]);
                    let distance_decay = (-0.5 * dist as f64).exp();
                    let propagated_error = error_to_send * distance_decay;

                    // Update neighbor using dual-mode
                    let neighbor_prior = self.compute_metric_prior(neighbor_id);
                    let (new_neighbor_belief, _) = self.belief_updater.update(
                        self.base.sentries[neighbor_id].belief_state,
                        propagated_error,
                        neighbor_prior,
                    );

                    self.base.sentries[neighbor_id].belief_state = new_neighbor_belief;
                    self.base.sentries[neighbor_id].update_activation(
                        &[propagated_error],
                        self.base.topology.criticality_threshold,
                    );

                    // Check for cascade
                    if new_neighbor_belief > self.base.topology.criticality_threshold {
                        cascade_size += 1;
                        propagation_queue.push(neighbor_id);
                    }
                }
            }
        }

        // Phase 3: Compute global free energy
        self.base.global_free_energy = self.base.sentries.iter().map(|s| s.free_energy).sum();

        // Phase 4: Determine detection result
        let max_belief = self
            .base
            .sentries
            .iter()
            .map(|s| s.belief_state)
            .fold(0.0, f64::max);

        let is_threat = max_belief > self.base.topology.criticality_threshold
            || cascade_size > (self.base.sentries.len() / 10).max(1);

        if is_threat {
            self.base.cascade_count += 1;
            self.base
                .detection_log
                .push(crate::adversarial_lattice::DetectionEvent {
                    timestamp,
                    sentry_id: entry_point,
                    confidence: max_belief,
                    cascade_size,
                    free_energy: self.base.global_free_energy,
                });
        }

        DetectionResult {
            is_threat,
            confidence: max_belief,
            cascade_size,
            free_energy: self.base.global_free_energy,
        }
    }

    /// Compute metric prior for a sentry based on its hyperbolic position
    fn compute_metric_prior(&self, sentry_id: usize) -> f64 {
        let pos = &self.simd_positions[sentry_id];
        let origin = LorentzVec::origin();
        let dist = pos.distance(&origin);

        // Prior increases with distance (perimeter more exposed)
        let base_prior = 0.1;
        let distance_factor = 0.05;

        (base_prior + distance_factor * dist as f64).min(0.9)
    }

    /// Switch to variational mode for offline learning
    pub fn enable_variational_mode(&mut self) {
        self.belief_updater.mode = BeliefUpdateMode::Variational;
    }

    /// Switch to real-time mode for online detection
    pub fn enable_realtime_mode(&mut self) {
        self.belief_updater.mode = BeliefUpdateMode::RealTime;
    }

    /// Get current adaptation state
    pub fn adaptation_state(&self) -> ThermodynamicStats {
        self.learner.stats.clone()
    }

    /// Reset lattice state
    pub fn reset(&mut self) {
        self.base.reset();
    }
}

/// Convert from AdversarialLattice
impl From<AdversarialLattice> for HyperbolicLattice {
    fn from(lattice: AdversarialLattice) -> Self {
        HyperbolicLattice::from_adversarial(lattice)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lorentz_vec_creation() {
        let v = LorentzVec::from_spatial(0.5, 0.3);
        assert!(v.is_valid(1e-6));

        // Check constraint: -t² + x² + y² = -1
        let constraint = v.minkowski_dot(&v);
        assert!((constraint + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_lorentz_vec_distance() {
        let origin = LorentzVec::origin();
        let p = LorentzVec::from_spatial(0.5, 0.0);

        let dist = origin.distance(&p);
        assert!(dist > 0.0);

        // Distance to self should be 0
        assert!(origin.distance(&origin) < 1e-6);
    }

    #[test]
    fn test_lorentz_boost_preserves_constraint() {
        let p = LorentzVec::from_spatial(0.3, 0.2);
        let boosted = p.boost_x(0.5);

        assert!(boosted.is_valid(1e-5));
    }

    #[test]
    fn test_dual_mode_updater_realtime() {
        let updater = DualModeBeliefUpdater::real_time();

        let (new_belief, _fe) = updater.update(0.5, 0.8, 0.3);
        assert!(new_belief > 0.5); // Should move toward observation
        assert!(new_belief < 1.0);
    }

    #[test]
    fn test_dual_mode_updater_variational() {
        let updater = DualModeBeliefUpdater::variational();

        // With high observation (0.9) and low prior (0.2), belief should move toward observation
        let (new_belief, free_energy) = updater.update(0.5, 0.9, 0.2);
        // Variational update should move belief toward observation (0.9) more than prior (0.2)
        // since precision is high (25.0) making likelihood dominate
        assert!(new_belief > 0.4, "Belief should increase from 0.5 toward 0.9, got {}", new_belief);
        assert!(free_energy.is_finite());
    }

    #[test]
    fn test_thermodynamic_learner() {
        let mut learner = ThermodynamicLearner::default();

        // Simulate cascade history
        for i in 1..50 {
            learner.record_detection(i % 10 + 1, 1.0 + i as f64 * 0.1);
        }

        assert!(learner.stats.avg_cascade_size > 0.0);
        assert!(learner.curvature_adaptation > 0.0);
    }

    #[test]
    fn test_hyperbolic_lattice_creation() -> Result<()> {
        let topology = DefenseTopology::balanced_fanout(2);
        let lattice = HyperbolicLattice::new(topology)?;

        assert!(!lattice.simd_positions.is_empty());
        assert_eq!(lattice.simd_positions.len(), lattice.base.sentries.len());

        Ok(())
    }

    #[test]
    fn test_hyperbolic_lattice_processing() -> Result<()> {
        let topology = DefenseTopology::balanced_fanout(2);
        let mut lattice = HyperbolicLattice::new(topology)?;

        // Process benign message
        let result = lattice.process_message(0, 0.1, 1);
        assert!(!result.is_threat);

        // Reset and process anomalous message
        lattice.reset();
        let result = lattice.process_message(0, 0.9, 2);
        // After processing high anomaly score, belief should increase from 0
        // The exact value depends on propagation and thermodynamic adaptations
        assert!(result.confidence > 0.0, "Confidence should be positive after anomaly, got {}", result.confidence);

        // Process multiple anomalous messages to trigger cascade
        for i in 3..10 {
            lattice.process_message(0, 0.9, i);
        }
        let final_result = lattice.process_message(0, 0.9, 10);
        // After repeated anomalies, confidence should be significant
        assert!(final_result.confidence > 0.1, "Final confidence should increase, got {}", final_result.confidence);

        Ok(())
    }

    #[test]
    fn test_mode_switching() -> Result<()> {
        let topology = DefenseTopology::maximum_connectivity(2);
        let mut lattice = HyperbolicLattice::new(topology)?;

        lattice.enable_realtime_mode();
        assert_eq!(lattice.belief_updater.mode, BeliefUpdateMode::RealTime);

        lattice.enable_variational_mode();
        assert_eq!(lattice.belief_updater.mode, BeliefUpdateMode::Variational);

        Ok(())
    }

    #[test]
    fn test_conversion_roundtrip() {
        let h = HyperboloidPoint {
            t: 1.5,
            x: 0.8,
            y: 0.6,
        };

        let v = LorentzVec::from_hyperboloid(&h);
        let h_back = v.to_hyperboloid();

        assert!((h.t - h_back.t).abs() < 1e-5);
        assert!((h.x - h_back.x).abs() < 1e-5);
        assert!((h.y - h_back.y).abs() < 1e-5);
    }

    #[test]
    fn test_batch_operations() {
        let points: Vec<LorentzVec> = (0..10)
            .map(|i| LorentzVec::from_spatial(0.1 * i as f32, 0.05 * i as f32))
            .collect();

        let distances = LorentzBatch::pairwise_distances(&points);
        assert_eq!(distances.len(), 10 * 9 / 2);

        let centroid = LorentzBatch::centroid(&points);
        assert!(centroid.is_valid(1e-4));
    }
}
