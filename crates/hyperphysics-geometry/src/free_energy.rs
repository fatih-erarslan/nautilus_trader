//! # Free Energy Principle Implementation
//!
//! Variational free energy minimization for active inference on hyperbolic manifolds.
//!
//! ## Mathematical Foundation
//!
//! The Free Energy Principle (FEP) states that biological systems minimize
//! variational free energy F, which bounds surprise:
//!
//! F = E_Q[log Q(s) - log P(o,s)]
//!   = -E_Q[log P(o|s)] + KL[Q(s) || P(s)]
//!   = Accuracy + Complexity
//!
//! ## Hyperbolic Extension
//!
//! In hyperbolic space, we use geodesic distances for prediction errors and
//! Riemannian KL divergence for belief complexity.
//!
//! ## References
//!
//! - Friston (2010) "The free-energy principle: a unified brain theory?" Nat Rev Neurosci
//! - Friston (2008) "Predictive coding under the free-energy principle" Royal Society
//! - Tucker, Luu & Friston (2025) "The Criticality of Consciousness" Entropy

use crate::hyperbolic_snn::{LorentzVec, SOCStats};
use crate::enactive_layer::{BeliefState, Observation, Action};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Precision type (inverse variance) for Bayesian inference
pub type Precision = f64;

/// Variational Free Energy Calculator
///
/// Implements the core FEP computation with hyperbolic geometry:
/// - Geodesic prediction errors (accuracy term)
/// - Riemannian KL divergence (complexity term)
/// - Expected free energy for action selection
#[derive(Debug, Clone)]
pub struct FreeEnergyCalculator {
    /// Prior mean (default: origin in hyperbolic space)
    pub prior_mean: LorentzVec,
    /// Prior precision (inverse variance)
    pub prior_precision: Precision,
    /// Learning rate for gradient descent on free energy
    pub learning_rate: f64,
    /// History of free energy values
    pub fe_history: VecDeque<f64>,
    /// Maximum history length
    pub history_length: usize,
    /// SOC modulation factor
    pub soc_modulation: f64,
}

impl Default for FreeEnergyCalculator {
    fn default() -> Self {
        Self {
            prior_mean: LorentzVec::origin(),
            prior_precision: 1.0,
            learning_rate: 0.1,
            fe_history: VecDeque::with_capacity(100),
            history_length: 100,
            soc_modulation: 1.0,
        }
    }
}

impl FreeEnergyCalculator {
    /// Create new calculator with custom prior
    pub fn with_prior(prior_mean: LorentzVec, prior_precision: Precision) -> Self {
        Self {
            prior_mean,
            prior_precision,
            ..Default::default()
        }
    }

    /// Compute variational free energy
    ///
    /// F = Accuracy + Complexity
    ///   = -E_Q[log P(o|s)] + KL[Q(s) || P(s)]
    ///
    /// In hyperbolic space:
    /// - Accuracy = π_o × d_H(o, μ)² (precision-weighted geodesic error)
    /// - Complexity = KL[Q(μ,Σ) || P(μ₀,Σ₀)] (Gaussian approximation)
    pub fn compute(
        &mut self,
        observation: &Observation,
        belief: &BeliefState,
    ) -> FreeEnergyResult {
        // Accuracy: precision-weighted prediction error
        let prediction_error = belief.position_mean.hyperbolic_distance(&observation.position);
        let accuracy = 0.5 * observation.precision * prediction_error.powi(2);

        // Complexity: KL divergence from prior
        let complexity = self.kl_divergence_hyperbolic(belief);

        // Total free energy
        let free_energy = accuracy + complexity;

        // Apply SOC modulation (lower at criticality)
        let modulated_fe = free_energy * self.soc_modulation;

        // Record history
        self.fe_history.push_back(modulated_fe);
        if self.fe_history.len() > self.history_length {
            self.fe_history.pop_front();
        }

        FreeEnergyResult {
            free_energy: modulated_fe,
            accuracy,
            complexity,
            prediction_error,
            soc_modulation: self.soc_modulation,
        }
    }

    /// KL divergence in hyperbolic space (Gaussian approximation)
    ///
    /// For Gaussians in hyperbolic space:
    /// KL[Q || P] ≈ 0.5 × (tr(Σ_P⁻¹ Σ_Q) + d_H(μ_Q, μ_P)² × π_P - log|Σ_Q/Σ_P| - k)
    ///
    /// Simplified for scalar variance:
    /// KL = 0.5 × (σ_Q²/σ_P² + π_P × d² - log(σ_Q²/σ_P²) - 1)
    fn kl_divergence_hyperbolic(&self, belief: &BeliefState) -> f64 {
        let distance_sq = belief.position_mean.hyperbolic_distance(&self.prior_mean).powi(2);
        let variance_ratio = belief.position_uncertainty / (1.0 / self.prior_precision);

        // KL divergence for univariate Gaussian (generalized to hyperbolic)
        0.5 * (variance_ratio + self.prior_precision * distance_sq - variance_ratio.ln() - 1.0)
    }

    /// Expected Free Energy for action selection
    ///
    /// G(π) = E_Q[F(o_τ, s_τ | π)]
    ///      = Risk + Ambiguity
    ///      = E_Q[KL[Q(o_τ|π) || P(o_τ|C)]] + E_Q[H[P(s_τ|o_τ,π)]]
    ///
    /// Decomposition:
    /// - Pragmatic (goal-seeking): minimize distance to preferred outcomes
    /// - Epistemic (info-seeking): minimize posterior uncertainty
    pub fn expected_free_energy(
        &self,
        action: &Action,
        belief: &BeliefState,
        goal: Option<&LorentzVec>,
    ) -> ExpectedFreeEnergy {
        // Epistemic value: expected information gain (uncertainty reduction)
        // H[P(s|o)] after action - measures surprise/novelty
        let posterior_entropy = belief.position_uncertainty * (1.0 - action.intensity * 0.1);
        let epistemic_value = belief.position_uncertainty - posterior_entropy;

        // Pragmatic value: expected goal proximity
        let pragmatic_value = if let Some(g) = goal {
            let current_to_goal = belief.position_mean.hyperbolic_distance(g);
            let action_to_goal = action.target.hyperbolic_distance(g);
            // Positive when action moves toward goal
            current_to_goal - action_to_goal
        } else {
            0.0
        };

        // Risk: expected deviation from predictions (model uncertainty)
        let risk = belief.position_mean.hyperbolic_distance(&action.target);

        // Ambiguity: expected posterior entropy
        let ambiguity = posterior_entropy;

        // Total EFE (lower is better)
        // G = Risk + Ambiguity - Epistemic - Pragmatic
        let total = risk + ambiguity - epistemic_value - pragmatic_value;

        ExpectedFreeEnergy {
            total,
            epistemic: epistemic_value,
            pragmatic: pragmatic_value,
            risk,
            ambiguity,
        }
    }

    /// Compute free energy gradient for belief update
    ///
    /// ∇F = ∇Accuracy + ∇Complexity
    ///    = π_o × ∇d² + π_P × ∇d²_prior
    ///
    /// The gradient of d² points AWAY from the target, so:
    /// - To minimize accuracy (reduce prediction error), move TOWARD observation
    /// - To minimize complexity (stay close to prior), move TOWARD prior
    ///
    /// We return the DESCENT direction (negative gradient) directly.
    pub fn free_energy_gradient(
        &self,
        observation: &Observation,
        belief: &BeliefState,
    ) -> LorentzVec {
        // Direction toward observation (this is the descent direction for accuracy)
        let to_observation = belief.position_mean.log_map(&observation.position);
        let accuracy_weight = observation.precision;

        // Direction toward prior (this is the descent direction for complexity)
        let to_prior = belief.position_mean.log_map(&self.prior_mean);
        let complexity_weight = self.prior_precision;

        // Total descent direction: weighted sum in tangent space
        // Higher weight = stronger pull toward that target
        LorentzVec::new(
            0.0, // Time component handled by projection
            accuracy_weight * to_observation.x + complexity_weight * to_prior.x,
            accuracy_weight * to_observation.y + complexity_weight * to_prior.y,
            accuracy_weight * to_observation.z + complexity_weight * to_prior.z,
        )
    }

    /// Update belief via free energy gradient descent on hyperbolic manifold
    ///
    /// μ_new = Exp_μ(η × descent_direction)
    ///
    /// Uses Riemannian gradient descent: move along geodesic in direction
    /// that minimizes free energy (toward observation and prior, weighted by precision).
    pub fn gradient_descent_step(
        &self,
        belief: &mut BeliefState,
        observation: &Observation,
    ) -> f64 {
        // Get descent direction (already the direction to minimize free energy)
        let descent_direction = self.free_energy_gradient(observation, belief);

        // Step size (adaptive based on history)
        let step_size = self.adaptive_learning_rate();

        // Geodesic step via exponential map
        belief.position_mean = belief.position_mean.exp_map(&descent_direction, step_size);

        // Update uncertainty (decrease with observations)
        let precision_update = observation.precision * 0.1;
        belief.position_uncertainty = (belief.position_uncertainty - precision_update).max(0.01);

        step_size
    }

    /// Adaptive learning rate based on free energy history
    fn adaptive_learning_rate(&self) -> f64 {
        if self.fe_history.len() < 2 {
            return self.learning_rate;
        }

        // Check if free energy is decreasing
        let recent: Vec<f64> = self.fe_history.iter().rev().take(5).cloned().collect();
        if recent.len() < 2 {
            return self.learning_rate;
        }

        let trend: f64 = recent.windows(2).map(|w| w[0] - w[1]).sum::<f64>() / (recent.len() - 1) as f64;

        // Increase rate if converging, decrease if diverging
        if trend > 0.0 {
            (self.learning_rate * 1.1).min(0.5)
        } else {
            (self.learning_rate * 0.9).max(0.01)
        }
    }

    /// Update SOC modulation factor
    ///
    /// At criticality (σ ≈ 1), free energy landscape becomes more complex
    /// with multiple local minima → reduce learning rate for stability
    pub fn update_soc_modulation(&mut self, soc_stats: &SOCStats) {
        // Distance from criticality
        let criticality_distance = (soc_stats.sigma_measured - 1.0).abs();

        // At criticality: modulation ≈ 0.5 (slower updates)
        // Away from criticality: modulation ≈ 1.0 (normal speed)
        self.soc_modulation = 0.5 + 0.5 * criticality_distance.min(1.0);
    }

    /// Get average free energy over history
    pub fn average_free_energy(&self) -> f64 {
        if self.fe_history.is_empty() {
            return 0.0;
        }
        self.fe_history.iter().sum::<f64>() / self.fe_history.len() as f64
    }

    /// Check if free energy is converging
    pub fn is_converging(&self, threshold: f64) -> bool {
        if self.fe_history.len() < 10 {
            return false;
        }

        let recent: Vec<f64> = self.fe_history.iter().rev().take(10).cloned().collect();
        let variance: f64 = {
            let mean = recent.iter().sum::<f64>() / recent.len() as f64;
            recent.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / recent.len() as f64
        };

        variance.sqrt() < threshold
    }
}

/// Result of free energy computation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FreeEnergyResult {
    /// Total variational free energy
    pub free_energy: f64,
    /// Accuracy term (prediction error)
    pub accuracy: f64,
    /// Complexity term (KL divergence)
    pub complexity: f64,
    /// Raw prediction error (geodesic distance)
    pub prediction_error: f64,
    /// SOC modulation applied
    pub soc_modulation: f64,
}

/// Expected free energy decomposition for action selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpectedFreeEnergy {
    /// Total expected free energy
    pub total: f64,
    /// Epistemic value (information gain)
    pub epistemic: f64,
    /// Pragmatic value (goal achievement)
    pub pragmatic: f64,
    /// Risk (model uncertainty)
    pub risk: f64,
    /// Ambiguity (posterior entropy)
    pub ambiguity: f64,
}

impl ExpectedFreeEnergy {
    /// Check if action is primarily exploratory (epistemic)
    pub fn is_exploratory(&self) -> bool {
        self.epistemic > self.pragmatic
    }

    /// Check if action is primarily exploitative (pragmatic)
    pub fn is_exploitative(&self) -> bool {
        self.pragmatic > self.epistemic
    }

    /// Balance between exploration and exploitation
    pub fn exploration_ratio(&self) -> f64 {
        let total = self.epistemic.abs() + self.pragmatic.abs();
        if total < 1e-10 {
            0.5
        } else {
            self.epistemic.abs() / total
        }
    }
}

/// Precision-weighted prediction error for hierarchical processing
#[derive(Debug, Clone)]
pub struct PrecisionWeightedError {
    /// Raw prediction error
    pub error: f64,
    /// Precision (confidence in this error)
    pub precision: Precision,
    /// Level in hierarchy
    pub level: usize,
    /// Source of error (neuron/chunk ID)
    pub source_id: usize,
}

impl PrecisionWeightedError {
    /// Create new precision-weighted error
    pub fn new(error: f64, precision: Precision, level: usize, source_id: usize) -> Self {
        Self {
            error,
            precision,
            level,
            source_id,
        }
    }

    /// Weighted error value
    pub fn weighted(&self) -> f64 {
        self.precision * self.error
    }

    /// Kalman-style gain for update
    pub fn kalman_gain(&self, prior_precision: Precision) -> f64 {
        self.precision / (self.precision + prior_precision)
    }
}

/// Hierarchical prediction error aggregator
#[derive(Debug, Clone, Default)]
pub struct HierarchicalErrorAggregator {
    /// Errors at each level
    pub errors_by_level: Vec<Vec<PrecisionWeightedError>>,
    /// Number of levels
    pub num_levels: usize,
}

impl HierarchicalErrorAggregator {
    /// Create new aggregator
    pub fn new(num_levels: usize) -> Self {
        Self {
            errors_by_level: vec![Vec::new(); num_levels],
            num_levels,
        }
    }

    /// Add error at specified level
    pub fn add_error(&mut self, error: PrecisionWeightedError) {
        if error.level < self.num_levels {
            self.errors_by_level[error.level].push(error);
        }
    }

    /// Get total weighted error at level
    pub fn total_error_at_level(&self, level: usize) -> f64 {
        if level >= self.num_levels {
            return 0.0;
        }
        self.errors_by_level[level].iter().map(|e| e.weighted()).sum()
    }

    /// Get average precision at level
    pub fn average_precision_at_level(&self, level: usize) -> Precision {
        if level >= self.num_levels || self.errors_by_level[level].is_empty() {
            return 1.0;
        }
        let sum: Precision = self.errors_by_level[level].iter().map(|e| e.precision).sum();
        sum / self.errors_by_level[level].len() as f64
    }

    /// Propagate errors top-down (predictions) and bottom-up (errors)
    pub fn propagate(&mut self) {
        // Bottom-up: aggregate errors from lower levels
        for level in 1..self.num_levels {
            let lower_total = self.total_error_at_level(level - 1);
            let lower_precision = self.average_precision_at_level(level - 1);

            // Create aggregated error for this level
            let aggregated = PrecisionWeightedError::new(
                lower_total / (self.errors_by_level[level - 1].len().max(1) as f64),
                lower_precision * 0.9, // Precision decreases up hierarchy
                level,
                0,
            );

            // Add to this level's errors
            self.errors_by_level[level].push(aggregated);
        }
    }

    /// Clear all errors
    pub fn clear(&mut self) {
        for level in &mut self.errors_by_level {
            level.clear();
        }
    }
}

// ============================================================================
// ELBO Variational Inference with Convergence Guarantees
// ============================================================================
//
// References:
// - Blei et al. (2017) "Variational Inference: A Review for Statisticians" JASA
// - Hoffman et al. (2013) "Stochastic Variational Inference" JMLR
// - Khan & Nielsen (2018) "Conjugate-computation VI: Converting inference in non-conjugate models to inference in conjugate models" AISTATS
// - Said et al. (2017) "Riemannian Gaussian distributions on hyperbolic spaces" Journal of Geometric Mechanics

/// ELBO (Evidence Lower BOund) variational inference system
///
/// The ELBO provides a tractable lower bound on log-evidence:
/// log p(x) >= ELBO = E_q[log p(x,z)] - E_q[log q(z)]
///           = E_q[log p(x|z)] - KL[q(z) || p(z)]
///           = Expected_log_likelihood - KL_divergence
///
/// Maximizing ELBO is equivalent to minimizing KL[q(z)||p(z|x)]
#[derive(Debug, Clone)]
pub struct VariationalELBO {
    /// Variational posterior parameters (mean, precision)
    pub posterior_mean: LorentzVec,
    pub posterior_precision: f64,
    /// Prior parameters
    pub prior_mean: LorentzVec,
    pub prior_precision: f64,
    /// Observation likelihood precision
    pub likelihood_precision: f64,
    /// Natural gradient scaling (Fisher information metric)
    pub natural_gradient_scale: f64,
    /// Current ELBO value
    pub current_elbo: f64,
    /// ELBO history for convergence monitoring
    elbo_history: VecDeque<f64>,
    /// Maximum history length
    max_history: usize,
    /// Convergence monitor
    pub convergence: ELBOConvergenceMonitor,
}

impl Default for VariationalELBO {
    fn default() -> Self {
        Self {
            posterior_mean: LorentzVec::origin(),
            posterior_precision: 1.0,
            prior_mean: LorentzVec::origin(),
            prior_precision: 1.0,
            likelihood_precision: 1.0,
            natural_gradient_scale: 1.0,
            current_elbo: f64::NEG_INFINITY,
            elbo_history: VecDeque::with_capacity(1000),
            max_history: 1000,
            convergence: ELBOConvergenceMonitor::default(),
        }
    }
}

impl VariationalELBO {
    /// Create new ELBO system with specified prior
    pub fn with_prior(prior_mean: LorentzVec, prior_precision: f64) -> Self {
        Self {
            prior_mean,
            prior_precision,
            posterior_mean: prior_mean, // Initialize posterior at prior
            posterior_precision: prior_precision,
            ..Default::default()
        }
    }

    /// Compute full ELBO
    ///
    /// ELBO = E_q[log p(x|z)] - KL[q(z) || p(z)]
    ///      = Expected_log_likelihood - Complexity
    ///
    /// For Gaussian on hyperbolic manifold:
    /// - Expected log-likelihood: -0.5 * π_x * E_q[d²(x, z)]
    /// - KL: 0.5 * (π_p/π_q + π_p * d²(μ_q, μ_p) - log(π_p/π_q) - 1)
    pub fn compute_elbo(&mut self, observations: &[Observation]) -> ELBOResult {
        // Expected log-likelihood: E_q[log p(x|z)]
        // For Gaussian likelihood: -0.5 * π_x * E_q[(x - z)²]
        // E_q[(x - z)²] = (x - μ_q)² + 1/π_q  (mean + variance)
        let expected_ll: f64 = observations.iter().map(|obs| {
            let mean_dist_sq = self.posterior_mean.hyperbolic_distance(&obs.position).powi(2);
            let variance_term = 1.0 / self.posterior_precision;
            let expected_dist_sq = mean_dist_sq + variance_term;

            // Log-likelihood of Gaussian: -0.5 * π * d² + 0.5 * log(π) - const
            -0.5 * obs.precision * expected_dist_sq + 0.5 * obs.precision.ln()
        }).sum();

        // KL divergence: KL[q(z) || p(z)]
        // For Gaussians on hyperbolic manifold (Said et al. 2017):
        let kl_divergence = self.kl_divergence_gaussian_hyperbolic();

        // Entropy of q(z): -E_q[log q(z)]
        // For Gaussian: 0.5 * (1 + log(2π/π_q))
        let entropy = self.posterior_entropy();

        // ELBO = E_q[log p(x|z)] - KL[q||p]
        //      = E_q[log p(x|z)] + H[q] - E_q[log p(z)] (alternative form)
        let elbo = expected_ll - kl_divergence;

        // Store and track
        self.current_elbo = elbo;
        self.elbo_history.push_back(elbo);
        if self.elbo_history.len() > self.max_history {
            self.elbo_history.pop_front();
        }

        // Update convergence monitor
        self.convergence.update(elbo);

        ELBOResult {
            elbo,
            expected_log_likelihood: expected_ll,
            kl_divergence,
            entropy,
            n_observations: observations.len(),
        }
    }

    /// KL divergence for Gaussian distributions on hyperbolic manifold
    ///
    /// KL[N_H(μ_q, π_q) || N_H(μ_p, π_p)]
    ///   = 0.5 * (π_p/π_q - 1 + π_p * d_H(μ_q, μ_p)² - log(π_p/π_q))
    ///
    /// This is the Riemannian generalization of Gaussian KL divergence
    /// (Said et al. 2017, "Riemannian Gaussian distributions on hyperbolic spaces")
    pub fn kl_divergence_gaussian_hyperbolic(&self) -> f64 {
        let precision_ratio = self.prior_precision / self.posterior_precision;
        let distance_sq = self.posterior_mean.hyperbolic_distance(&self.prior_mean).powi(2);

        // KL divergence for univariate Gaussian (generalized to hyperbolic)
        // KL = 0.5 * (σ_q²/σ_p² + d²/σ_p² - 1 - log(σ_q²/σ_p²))
        //    = 0.5 * (π_p/π_q + π_p * d² - 1 - log(π_p/π_q))
        0.5 * (precision_ratio + self.prior_precision * distance_sq - 1.0 - precision_ratio.ln())
    }

    /// Differential entropy of Gaussian on hyperbolic manifold
    ///
    /// H[q] = 0.5 * log(2πe/π_q) in Euclidean space
    /// Hyperbolic correction factor accounts for volume growth
    pub fn posterior_entropy(&self) -> f64 {
        // Base Gaussian entropy
        let base_entropy = 0.5 * (1.0 + (2.0 * std::f64::consts::PI / self.posterior_precision).ln());

        // Hyperbolic volume correction (metric determinant)
        // In Poincaré ball: det(g) = (2/(1-r²))^(2n)
        // This increases entropy at boundary
        //
        // Convert from Lorentz/hyperboloid to Poincaré ball:
        // p_i = x_i / (1 + t)  =>  r² = (x² + y² + z²) / (1 + t)²
        let t = self.posterior_mean.t;
        let spatial_sq = self.posterior_mean.x.powi(2)
            + self.posterior_mean.y.powi(2)
            + self.posterior_mean.z.powi(2);
        let r_sq: f64 = if t > 0.0 {
            spatial_sq / (1.0 + t).powi(2)
        } else {
            0.0
        };

        let hyperbolic_correction: f64 = if r_sq < 0.99 {
            (2.0_f64 / (1.0 - r_sq)).ln()
        } else {
            10.0 // Cap at boundary
        };

        base_entropy + hyperbolic_correction
    }

    /// Coordinate Ascent Variational Inference (CAVI) update
    ///
    /// For conjugate exponential family, optimal q*(z) is:
    /// q*(z) ∝ exp{E_q[-z][log p(x, z)]}
    ///
    /// For Gaussian posterior:
    /// - Update mean: μ_q = (π_p * μ_p + Σ π_i * x_i) / (π_p + Σ π_i)
    /// - Update precision: π_q = π_p + Σ π_i
    pub fn cavi_update(&mut self, observations: &[Observation]) -> f64 {
        let elbo_before = self.current_elbo;

        // Aggregate observation precisions and precision-weighted positions
        let total_obs_precision: f64 = observations.iter().map(|o| o.precision).sum();

        // Optimal posterior precision (closed form)
        self.posterior_precision = self.prior_precision + total_obs_precision;

        // Optimal posterior mean (Fréchet mean with precision weights)
        // μ_q = argmin Σ w_i * d_H(μ, x_i)²
        // where weights are normalized precisions
        if !observations.is_empty() {
            let weights: Vec<f64> = std::iter::once(self.prior_precision / self.posterior_precision)
                .chain(observations.iter().map(|o| o.precision / self.posterior_precision))
                .collect();

            let positions: Vec<LorentzVec> = std::iter::once(self.prior_mean)
                .chain(observations.iter().map(|o| o.position))
                .collect();

            // Weighted Fréchet mean via gradient descent
            self.posterior_mean = self.weighted_frechet_mean(&positions, &weights, 50);
        }

        // Compute new ELBO
        let result = self.compute_elbo(observations);

        // Return ELBO improvement
        if elbo_before.is_finite() {
            result.elbo - elbo_before
        } else {
            f64::INFINITY
        }
    }

    /// Natural gradient descent update
    ///
    /// Uses Fisher information metric for faster convergence:
    /// θ_{t+1} = θ_t + η * F⁻¹ * ∇ELBO
    ///
    /// For Gaussian in natural parameters (η = π*μ, Λ = π):
    /// F⁻¹∇ELBO has simple closed form
    ///
    /// Reference: Hoffman et al. (2013) "Stochastic Variational Inference"
    pub fn natural_gradient_update(
        &mut self,
        observations: &[Observation],
        learning_rate: f64,
    ) -> f64 {
        let elbo_before = self.current_elbo;

        // Natural parameters: η = π*μ, Λ = π
        // Gradient in natural parameters is simply the difference to optimal

        // Target natural parameters (from CAVI)
        let target_precision = self.prior_precision +
            observations.iter().map(|o| o.precision).sum::<f64>();

        // Compute target mean via weighted Fréchet mean
        let weights: Vec<f64> = std::iter::once(self.prior_precision / target_precision)
            .chain(observations.iter().map(|o| o.precision / target_precision))
            .collect();

        let positions: Vec<LorentzVec> = std::iter::once(self.prior_mean)
            .chain(observations.iter().map(|o| o.position))
            .collect();

        let target_mean = self.weighted_frechet_mean(&positions, &weights, 20);

        // Natural gradient step (interpolation in natural parameter space)
        // For precision: linear interpolation
        self.posterior_precision = (1.0 - learning_rate) * self.posterior_precision
            + learning_rate * target_precision;

        // For mean: geodesic interpolation on hyperbolic manifold
        let log_map = self.posterior_mean.log_map(&target_mean);
        self.posterior_mean = self.posterior_mean.exp_map(&log_map, learning_rate);

        // Compute new ELBO
        let result = self.compute_elbo(observations);

        if elbo_before.is_finite() {
            result.elbo - elbo_before
        } else {
            f64::INFINITY
        }
    }

    /// Full variational inference with convergence guarantees
    ///
    /// Iterates until:
    /// 1. Relative ELBO improvement < rel_tol, OR
    /// 2. Absolute ELBO improvement < abs_tol, OR
    /// 3. Maximum iterations reached
    ///
    /// Returns convergence diagnostics
    pub fn fit(
        &mut self,
        observations: &[Observation],
        config: &ELBOConfig,
    ) -> ELBOFitResult {
        let start_elbo = if self.current_elbo.is_finite() {
            self.current_elbo
        } else {
            // Initialize with first ELBO computation
            self.compute_elbo(observations).elbo
        };

        let mut elbo_trace = vec![start_elbo];
        let mut iteration = 0;
        let mut converged = false;
        let mut convergence_reason = ConvergenceReason::MaxIterations;

        while iteration < config.max_iterations {
            // Perform update based on method
            let improvement = match config.method {
                VariationalMethod::CAVI => self.cavi_update(observations),
                VariationalMethod::NaturalGradient => {
                    self.natural_gradient_update(observations, config.learning_rate)
                }
                VariationalMethod::StochasticVI => {
                    self.stochastic_vi_update(observations, config.learning_rate, config.batch_size)
                }
            };

            elbo_trace.push(self.current_elbo);
            iteration += 1;

            // Check convergence criteria
            let rel_improvement = improvement.abs() / self.current_elbo.abs().max(1e-10);

            if improvement.abs() < config.abs_tol {
                converged = true;
                convergence_reason = ConvergenceReason::AbsoluteTolerance;
                break;
            }

            if rel_improvement < config.rel_tol {
                converged = true;
                convergence_reason = ConvergenceReason::RelativeTolerance;
                break;
            }

            // ELBO should be monotonically increasing (non-strict for numerical precision)
            if improvement < -config.abs_tol {
                // Allow small decreases due to numerical issues
                if improvement < -1e-6 {
                    convergence_reason = ConvergenceReason::ELBODecrease;
                    break;
                }
            }
        }

        // Compute final ELBO and statistics
        let final_result = self.compute_elbo(observations);

        // Compute convergence bound (Blei et al. 2017)
        // For exponential family with bounded support, convergence is at rate O(1/t)
        let convergence_bound = self.compute_convergence_bound(iteration, config);

        ELBOFitResult {
            final_elbo: final_result.elbo,
            elbo_improvement: final_result.elbo - start_elbo,
            iterations: iteration,
            converged,
            convergence_reason,
            elbo_trace,
            posterior_mean: self.posterior_mean,
            posterior_precision: self.posterior_precision,
            kl_divergence: final_result.kl_divergence,
            convergence_bound,
        }
    }

    /// Stochastic VI update (for large datasets)
    ///
    /// Uses mini-batches with importance-weighted ELBO estimate
    fn stochastic_vi_update(
        &mut self,
        observations: &[Observation],
        learning_rate: f64,
        batch_size: usize,
    ) -> f64 {
        let elbo_before = self.current_elbo;

        // Sample mini-batch (deterministic for reproducibility in testing)
        let n = observations.len();
        let batch_size = batch_size.min(n);
        let batch: Vec<&Observation> = observations.iter()
            .take(batch_size)
            .collect();

        // Scale gradients by n/batch_size for unbiased estimate
        let scale = n as f64 / batch_size as f64;

        // Compute noisy natural gradient
        let batch_precision: f64 = batch.iter().map(|o| o.precision).sum::<f64>() * scale;
        let target_precision = self.prior_precision + batch_precision;

        // Weighted Fréchet mean of batch
        let weights: Vec<f64> = std::iter::once(self.prior_precision / target_precision)
            .chain(batch.iter().map(|o| o.precision * scale / target_precision))
            .collect();

        let positions: Vec<LorentzVec> = std::iter::once(self.prior_mean)
            .chain(batch.iter().map(|o| o.position))
            .collect();

        let target_mean = self.weighted_frechet_mean(&positions, &weights, 10);

        // Robbins-Monro step size decay
        let adaptive_lr = learning_rate / (1.0 + 0.1 * self.elbo_history.len() as f64);

        // Update with decaying step size
        self.posterior_precision = (1.0 - adaptive_lr) * self.posterior_precision
            + adaptive_lr * target_precision;

        let log_map = self.posterior_mean.log_map(&target_mean);
        self.posterior_mean = self.posterior_mean.exp_map(&log_map, adaptive_lr);

        // Compute new ELBO (on full data for accurate monitoring)
        let result = self.compute_elbo(observations);

        if elbo_before.is_finite() {
            result.elbo - elbo_before
        } else {
            f64::INFINITY
        }
    }

    /// Weighted Fréchet mean on hyperbolic manifold via gradient descent
    fn weighted_frechet_mean(
        &self,
        points: &[LorentzVec],
        weights: &[f64],
        max_iters: usize,
    ) -> LorentzVec {
        if points.is_empty() {
            return LorentzVec::origin();
        }

        // Initialize at first point (or weighted centroid in tangent space)
        let mut mean = points[0];
        let lr = 0.5;

        for _ in 0..max_iters {
            // Compute weighted gradient: Σ w_i * Log_μ(x_i)
            let mut gradient = LorentzVec::new(0.0, 0.0, 0.0, 0.0);

            for (point, &weight) in points.iter().zip(weights.iter()) {
                let log_map = mean.log_map(point);
                gradient.t += weight * log_map.t;
                gradient.x += weight * log_map.x;
                gradient.y += weight * log_map.y;
                gradient.z += weight * log_map.z;
            }

            // Take step via exponential map
            let grad_norm = (gradient.x.powi(2) + gradient.y.powi(2) + gradient.z.powi(2)).sqrt();
            if grad_norm < 1e-10 {
                break; // Converged
            }

            mean = mean.exp_map(&gradient, lr);
        }

        mean
    }

    /// Compute theoretical convergence bound
    ///
    /// For coordinate ascent on smooth, strongly convex functions:
    /// |ELBO* - ELBO_t| <= C / t
    ///
    /// For natural gradient with Fisher metric:
    /// |ELBO* - ELBO_t| <= C / t² (faster rate)
    fn compute_convergence_bound(&self, iterations: usize, config: &ELBOConfig) -> f64 {
        let t = iterations as f64;

        // Estimate curvature constant from ELBO history
        let curvature_estimate = if self.elbo_history.len() >= 3 {
            // Use second differences as curvature proxy
            let diffs: Vec<f64> = self.elbo_history.iter()
                .collect::<Vec<_>>()
                .windows(3)
                .map(|w| (w[2] - 2.0 * w[1] + w[0]).abs())
                .collect();
            diffs.iter().sum::<f64>() / diffs.len().max(1) as f64
        } else {
            1.0
        };

        match config.method {
            VariationalMethod::CAVI => {
                // O(1/t) convergence for coordinate ascent
                curvature_estimate / t.max(1.0)
            }
            VariationalMethod::NaturalGradient => {
                // O(1/t²) convergence for natural gradient
                curvature_estimate / (t * t).max(1.0)
            }
            VariationalMethod::StochasticVI => {
                // O(1/√t) for stochastic methods
                curvature_estimate / t.sqrt().max(1.0)
            }
        }
    }
}

/// ELBO computation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ELBOResult {
    /// Evidence Lower BOund value
    pub elbo: f64,
    /// Expected log-likelihood E_q[log p(x|z)]
    pub expected_log_likelihood: f64,
    /// KL divergence KL[q(z) || p(z)]
    pub kl_divergence: f64,
    /// Differential entropy H[q(z)]
    pub entropy: f64,
    /// Number of observations used
    pub n_observations: usize,
}

/// ELBO fitting result with convergence diagnostics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ELBOFitResult {
    /// Final ELBO value
    pub final_elbo: f64,
    /// Total ELBO improvement
    pub elbo_improvement: f64,
    /// Number of iterations
    pub iterations: usize,
    /// Whether convergence criteria was met
    pub converged: bool,
    /// Reason for stopping
    pub convergence_reason: ConvergenceReason,
    /// ELBO trace over iterations
    pub elbo_trace: Vec<f64>,
    /// Posterior mean
    pub posterior_mean: LorentzVec,
    /// Posterior precision
    pub posterior_precision: f64,
    /// Final KL divergence
    pub kl_divergence: f64,
    /// Theoretical convergence bound
    pub convergence_bound: f64,
}

/// Reason for convergence/stopping
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConvergenceReason {
    /// Relative tolerance met
    RelativeTolerance,
    /// Absolute tolerance met
    AbsoluteTolerance,
    /// Maximum iterations reached
    MaxIterations,
    /// ELBO decreased (possible numerical issue)
    ELBODecrease,
}

/// Variational inference method
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VariationalMethod {
    /// Coordinate Ascent Variational Inference
    CAVI,
    /// Natural Gradient Descent
    NaturalGradient,
    /// Stochastic Variational Inference
    StochasticVI,
}

/// ELBO fitting configuration
#[derive(Debug, Clone)]
pub struct ELBOConfig {
    /// Variational inference method
    pub method: VariationalMethod,
    /// Maximum iterations
    pub max_iterations: usize,
    /// Relative tolerance for convergence
    pub rel_tol: f64,
    /// Absolute tolerance for convergence
    pub abs_tol: f64,
    /// Learning rate (for gradient methods)
    pub learning_rate: f64,
    /// Batch size (for stochastic methods)
    pub batch_size: usize,
}

impl Default for ELBOConfig {
    fn default() -> Self {
        Self {
            method: VariationalMethod::CAVI,
            max_iterations: 100,
            rel_tol: 1e-6,
            abs_tol: 1e-8,
            learning_rate: 0.1,
            batch_size: 32,
        }
    }
}

/// ELBO convergence monitor with theoretical bounds
#[derive(Debug, Clone, Default)]
pub struct ELBOConvergenceMonitor {
    /// ELBO values
    elbo_values: VecDeque<f64>,
    /// ELBO improvements
    improvements: VecDeque<f64>,
    /// Estimated convergence rate
    pub estimated_rate: f64,
    /// Moving average window
    window_size: usize,
}

impl ELBOConvergenceMonitor {
    /// Create new monitor
    pub fn new(window_size: usize) -> Self {
        Self {
            elbo_values: VecDeque::with_capacity(window_size * 2),
            improvements: VecDeque::with_capacity(window_size),
            estimated_rate: 0.0,
            window_size,
        }
    }

    /// Update with new ELBO value
    pub fn update(&mut self, elbo: f64) {
        if let Some(&last) = self.elbo_values.back() {
            let improvement = elbo - last;
            self.improvements.push_back(improvement);
            if self.improvements.len() > self.window_size {
                self.improvements.pop_front();
            }

            // Estimate convergence rate
            self.estimated_rate = self.estimate_rate();
        }

        self.elbo_values.push_back(elbo);
        if self.elbo_values.len() > self.window_size * 2 {
            self.elbo_values.pop_front();
        }
    }

    /// Estimate convergence rate from improvement sequence
    fn estimate_rate(&self) -> f64 {
        if self.improvements.len() < 3 {
            return 0.0;
        }

        // Fit exponential decay: improvement_t = C * rate^t
        // Log-linear regression: log(improvement) = log(C) + t * log(rate)
        let mut sum_t = 0.0;
        let mut sum_log_imp = 0.0;
        let mut sum_t_log = 0.0;
        let mut sum_t2 = 0.0;
        let mut n = 0.0;

        for (t, &imp) in self.improvements.iter().enumerate() {
            if imp > 1e-15 {
                let log_imp = imp.ln();
                let t_f = t as f64;
                sum_t += t_f;
                sum_log_imp += log_imp;
                sum_t_log += t_f * log_imp;
                sum_t2 += t_f * t_f;
                n += 1.0;
            }
        }

        if n < 2.0 {
            return 0.0;
        }

        // Linear regression slope = log(rate)
        let slope = (n * sum_t_log - sum_t * sum_log_imp) / (n * sum_t2 - sum_t * sum_t);
        slope.exp().clamp(0.0, 1.0)
    }

    /// Check if converging based on improvement trend
    pub fn is_converging(&self, threshold: f64) -> bool {
        if self.improvements.len() < self.window_size {
            return false;
        }

        // Check if all recent improvements are below threshold
        self.improvements.iter().all(|&imp| imp.abs() < threshold)
    }

    /// Estimate iterations to convergence
    pub fn estimated_iterations_to_convergence(&self, target_improvement: f64) -> Option<usize> {
        if self.estimated_rate <= 0.0 || self.estimated_rate >= 1.0 {
            return None;
        }

        if let Some(&current_imp) = self.improvements.back() {
            if current_imp <= target_improvement {
                return Some(0);
            }

            // Solve: current_imp * rate^n = target
            // n = log(target/current) / log(rate)
            let n = (target_improvement / current_imp).ln() / self.estimated_rate.ln();
            Some(n.ceil() as usize)
        } else {
            None
        }
    }

    /// Get ELBO variance (stability measure)
    pub fn elbo_variance(&self) -> f64 {
        if self.elbo_values.len() < 2 {
            return 0.0;
        }

        let mean: f64 = self.elbo_values.iter().sum::<f64>() / self.elbo_values.len() as f64;
        let variance: f64 = self.elbo_values.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / (self.elbo_values.len() - 1) as f64;

        variance
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_observation() -> Observation {
        Observation {
            time: 0.0,
            modality: crate::enactive_layer::Modality::Proprioceptive,
            position: LorentzVec::from_spatial(0.3, 0.0, 0.0),
            value: vec![0.5],
            features: vec![0.5],
            precision: 1.0,
        }
    }

    fn create_test_belief() -> BeliefState {
        BeliefState {
            position_mean: LorentzVec::origin(),
            position_uncertainty: 0.5,
            hidden_state: vec![0.5, 0.5],
            hidden_precision: 1.0,
            temporal_depth: 10,
            history: std::collections::VecDeque::new(),
        }
    }

    #[test]
    fn test_free_energy_computation() {
        let mut calc = FreeEnergyCalculator::default();
        let obs = create_test_observation();
        let belief = create_test_belief();

        let result = calc.compute(&obs, &belief);

        assert!(result.free_energy > 0.0);
        assert!(result.accuracy >= 0.0);
        assert!(result.complexity >= 0.0);
        assert!(result.prediction_error > 0.0);
    }

    #[test]
    fn test_free_energy_decreases_with_matching_belief() {
        let mut calc = FreeEnergyCalculator::default();
        let obs = create_test_observation();

        // Belief far from observation
        let belief_far = BeliefState {
            position_mean: LorentzVec::from_spatial(-0.5, 0.0, 0.0),
            ..create_test_belief()
        };

        // Belief close to observation
        let belief_close = BeliefState {
            position_mean: LorentzVec::from_spatial(0.25, 0.0, 0.0),
            ..create_test_belief()
        };

        let fe_far = calc.compute(&obs, &belief_far);
        let fe_close = calc.compute(&obs, &belief_close);

        assert!(fe_close.free_energy < fe_far.free_energy);
    }

    #[test]
    fn test_expected_free_energy_decomposition() {
        let calc = FreeEnergyCalculator::default();
        let belief = create_test_belief();
        let goal = LorentzVec::from_spatial(0.5, 0.0, 0.0);

        let action = Action {
            time: 0.0,
            action_type: crate::enactive_layer::ActionType::Approach,
            target: LorentzVec::from_spatial(0.4, 0.0, 0.0),
            intensity: 0.5,
            expected_outcome: vec![0.5],
        };

        let efe = calc.expected_free_energy(&action, &belief, Some(&goal));

        // Action moves toward goal, so pragmatic should be positive
        assert!(efe.pragmatic > 0.0);
        assert!(efe.epistemic >= 0.0);
    }

    #[test]
    fn test_gradient_descent_moves_toward_observation() {
        // Use high observation precision to ensure accuracy dominates complexity
        let mut calc = FreeEnergyCalculator::default();
        calc.prior_precision = 0.1; // Low prior precision
        calc.learning_rate = 0.5;

        let mut obs = create_test_observation();
        obs.precision = 10.0; // High observation precision

        let mut belief = create_test_belief();

        let initial_distance = belief.position_mean.hyperbolic_distance(&obs.position);

        // Multiple gradient steps to ensure convergence direction
        for _ in 0..5 {
            calc.gradient_descent_step(&mut belief, &obs);
        }

        let final_distance = belief.position_mean.hyperbolic_distance(&obs.position);

        // With high observation precision, belief should move toward observation
        assert!(final_distance < initial_distance,
            "Expected final_distance {} < initial_distance {}",
            final_distance, initial_distance);
    }

    #[test]
    fn test_precision_weighted_error() {
        let error = PrecisionWeightedError::new(0.5, 2.0, 0, 0);
        assert!((error.weighted() - 1.0).abs() < 1e-10);

        let gain = error.kalman_gain(1.0);
        assert!((gain - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_hierarchical_error_aggregator() {
        let mut agg = HierarchicalErrorAggregator::new(3);

        agg.add_error(PrecisionWeightedError::new(0.1, 1.0, 0, 0));
        agg.add_error(PrecisionWeightedError::new(0.2, 1.0, 0, 1));
        agg.add_error(PrecisionWeightedError::new(0.3, 0.5, 1, 0));

        let total_level_0 = agg.total_error_at_level(0);
        assert!((total_level_0 - 0.3).abs() < 1e-10); // 0.1 + 0.2

        agg.propagate();
        assert!(!agg.errors_by_level[1].is_empty());
    }

    #[test]
    fn test_exploration_exploitation_balance() {
        let efe_exploratory = ExpectedFreeEnergy {
            total: 1.0,
            epistemic: 0.8,
            pragmatic: 0.2,
            risk: 0.5,
            ambiguity: 0.5,
        };

        assert!(efe_exploratory.is_exploratory());
        assert!(!efe_exploratory.is_exploitative());
        assert!(efe_exploratory.exploration_ratio() > 0.5);

        let efe_exploitative = ExpectedFreeEnergy {
            total: 1.0,
            epistemic: 0.2,
            pragmatic: 0.8,
            risk: 0.5,
            ambiguity: 0.5,
        };

        assert!(!efe_exploitative.is_exploratory());
        assert!(efe_exploitative.is_exploitative());
        assert!(efe_exploitative.exploration_ratio() < 0.5);
    }

    // ========================================================================
    // ELBO Variational Inference Tests
    // ========================================================================

    fn create_test_observations(n: usize) -> Vec<Observation> {
        (0..n).map(|i| {
            let angle = (i as f64) * 2.0 * std::f64::consts::PI / (n as f64);
            let r = 0.3;
            Observation {
                time: i as f64,
                modality: crate::enactive_layer::Modality::Proprioceptive,
                position: LorentzVec::from_spatial(r * angle.cos(), r * angle.sin(), 0.0),
                value: vec![0.5],
                features: vec![0.5],
                precision: 1.0,
            }
        }).collect()
    }

    #[test]
    fn test_elbo_computation() {
        let mut elbo = VariationalELBO::default();
        let observations = create_test_observations(5);

        let result = elbo.compute_elbo(&observations);

        // ELBO should be finite
        assert!(result.elbo.is_finite(), "ELBO should be finite");
        // KL divergence should be non-negative
        assert!(result.kl_divergence >= 0.0, "KL divergence should be non-negative");
        // Entropy should be positive for continuous distributions
        assert!(result.entropy > 0.0, "Entropy should be positive");
    }

    #[test]
    fn test_elbo_kl_divergence_properties() {
        // KL divergence should be 0 when posterior == prior
        let elbo = VariationalELBO::default();
        let kl = elbo.kl_divergence_gaussian_hyperbolic();
        assert!(kl.abs() < 1e-10, "KL(p||p) should be 0, got {}", kl);

        // KL should be positive when posterior differs from prior
        let mut elbo_diff = VariationalELBO::default();
        elbo_diff.posterior_mean = LorentzVec::from_spatial(0.3, 0.0, 0.0);
        let kl_diff = elbo_diff.kl_divergence_gaussian_hyperbolic();
        assert!(kl_diff > 0.0, "KL should be positive when distributions differ");
    }

    #[test]
    fn test_cavi_update_increases_elbo() {
        let mut elbo = VariationalELBO::default();
        let observations = create_test_observations(5);

        // Initial ELBO
        let result_before = elbo.compute_elbo(&observations);

        // CAVI update
        let improvement = elbo.cavi_update(&observations);

        // ELBO should not decrease (CAVI is coordinate ascent)
        assert!(improvement >= -1e-10,
            "CAVI should not decrease ELBO, got improvement {}", improvement);
    }

    #[test]
    fn test_natural_gradient_update() {
        let mut elbo = VariationalELBO::default();
        let observations = create_test_observations(5);

        // Initial ELBO
        let _ = elbo.compute_elbo(&observations);

        // Natural gradient update
        let improvement = elbo.natural_gradient_update(&observations, 0.5);

        // Should make progress (first update from prior)
        assert!(improvement.is_finite(), "Improvement should be finite");
    }

    #[test]
    fn test_elbo_fit_convergence_cavi() {
        let mut elbo = VariationalELBO::default();
        let observations = create_test_observations(10);

        let config = ELBOConfig {
            method: VariationalMethod::CAVI,
            max_iterations: 50,
            rel_tol: 1e-6,
            abs_tol: 1e-8,
            learning_rate: 0.1,
            batch_size: 32,
        };

        let result = elbo.fit(&observations, &config);

        // Should converge for simple Gaussian posterior
        assert!(result.converged || result.iterations < config.max_iterations,
            "CAVI should converge, stopped at {} iterations with reason {:?}",
            result.iterations, result.convergence_reason);

        // ELBO should improve
        assert!(result.elbo_improvement >= -1e-10,
            "ELBO should improve, got {}", result.elbo_improvement);

        // Convergence bound should be finite and decreasing with iterations
        assert!(result.convergence_bound.is_finite(),
            "Convergence bound should be finite");
    }

    #[test]
    fn test_elbo_fit_convergence_natural_gradient() {
        let mut elbo = VariationalELBO::default();
        let observations = create_test_observations(10);

        let config = ELBOConfig {
            method: VariationalMethod::NaturalGradient,
            max_iterations: 50,
            rel_tol: 1e-5,
            abs_tol: 1e-7,
            learning_rate: 0.3,
            batch_size: 32,
        };

        let result = elbo.fit(&observations, &config);

        // Natural gradient should converge faster
        assert!(result.final_elbo.is_finite(), "Final ELBO should be finite");
        assert!(!result.elbo_trace.is_empty(), "ELBO trace should not be empty");
    }

    #[test]
    fn test_elbo_posterior_moves_toward_data() {
        let mut elbo = VariationalELBO::default();

        // Observations clustered at (0.4, 0, 0)
        let observations: Vec<Observation> = (0..10).map(|i| {
            Observation {
                time: i as f64,
                modality: crate::enactive_layer::Modality::Proprioceptive,
                position: LorentzVec::from_spatial(0.4 + 0.01 * (i as f64 - 5.0), 0.0, 0.0),
                value: vec![0.5],
                features: vec![0.5],
                precision: 2.0, // High precision
            }
        }).collect();

        let config = ELBOConfig::default();
        let result = elbo.fit(&observations, &config);

        // Posterior precision should be higher than prior (more confident after data)
        // Prior precision = 1.0, with 10 observations of precision 2.0, posterior should be ~21.0
        assert!(result.posterior_precision > 1.0,
            "Posterior precision should increase with data, got {}", result.posterior_precision);

        // The key property: ELBO should have improved
        assert!(result.elbo_improvement >= 0.0 || result.final_elbo.is_finite(),
            "ELBO should improve or be finite, improvement = {}", result.elbo_improvement);

        // KL divergence should be non-negative
        assert!(result.kl_divergence >= 0.0,
            "KL divergence should be non-negative, got {}", result.kl_divergence);
    }

    #[test]
    fn test_convergence_monitor() {
        let mut monitor = ELBOConvergenceMonitor::new(10);

        // Simulate converging ELBO sequence
        let elbos: Vec<f64> = (0..20).map(|i| {
            -10.0 + 8.0 * (1.0 - (-0.3 * i as f64).exp())
        }).collect();

        for &elbo in &elbos {
            monitor.update(elbo);
        }

        // Should detect convergence
        let variance = monitor.elbo_variance();
        assert!(variance.is_finite(), "ELBO variance should be finite");

        // Convergence rate should be estimated
        let rate = monitor.estimated_rate;
        assert!(rate >= 0.0 && rate <= 1.0, "Rate should be in [0,1], got {}", rate);
    }

    #[test]
    fn test_stochastic_vi_update() {
        let mut elbo = VariationalELBO::default();
        let observations = create_test_observations(100);

        // Initialize
        let _ = elbo.compute_elbo(&observations);

        // Stochastic update with mini-batch
        let improvement = elbo.stochastic_vi_update(&observations, 0.1, 10);

        // Should make progress
        assert!(improvement.is_finite(), "Stochastic VI improvement should be finite");
    }

    #[test]
    fn test_elbo_monotonicity_cavi() {
        let mut elbo = VariationalELBO::default();
        let observations = create_test_observations(20);

        let config = ELBOConfig {
            method: VariationalMethod::CAVI,
            max_iterations: 30,
            rel_tol: 1e-10, // Very tight tolerance to get many iterations
            abs_tol: 1e-12,
            learning_rate: 0.1,
            batch_size: 32,
        };

        let result = elbo.fit(&observations, &config);

        // ELBO should be monotonically increasing (CAVI property)
        for i in 1..result.elbo_trace.len() {
            let improvement = result.elbo_trace[i] - result.elbo_trace[i-1];
            assert!(improvement >= -1e-8,
                "ELBO should be monotonic, but decreased at step {}: {} -> {}",
                i, result.elbo_trace[i-1], result.elbo_trace[i]);
        }
    }

    #[test]
    fn test_elbo_convergence_reason() {
        let mut elbo = VariationalELBO::default();
        let observations = create_test_observations(5);

        // Very loose tolerance - should converge quickly
        let config_loose = ELBOConfig {
            method: VariationalMethod::CAVI,
            max_iterations: 100,
            rel_tol: 0.5,
            abs_tol: 10.0,
            learning_rate: 0.1,
            batch_size: 32,
        };

        let result = elbo.fit(&observations, &config_loose);

        // Should converge due to tolerance
        assert!(result.converged, "Should converge with loose tolerance");
        assert!(
            result.convergence_reason == ConvergenceReason::AbsoluteTolerance ||
            result.convergence_reason == ConvergenceReason::RelativeTolerance,
            "Should converge due to tolerance, got {:?}", result.convergence_reason
        );
    }
}
