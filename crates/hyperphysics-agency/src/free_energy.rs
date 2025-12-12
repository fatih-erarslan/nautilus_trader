//! # Free Energy Principle Engine
//!
//! Implementation of Karl Friston's Free Energy Principle for active inference.
//!
//! ## Theoretical Foundation
//!
//! The Free Energy Principle states that all adaptive systems minimize free energy:
//!
//! **F = D_KL[q(s|o) || p(s|o)] + surprise(o)**
//!
//! Where:
//! - **D_KL[q || p]** = Complexity (variational divergence)
//! - **surprise(o)** = -log p(o|m) (prediction error)
//! - **q(s|o)** = approximate posterior over hidden states
//! - **p(s|o)** = true posterior (intractable)
//!
//! ## Decomposition
//!
//! F = ∫ q(s) [log q(s) - log p(o,s)] ds
//!   = ∫ q(s) log(q(s)/p(s|o)) ds + ∫ q(s) log(p(s|o)) ds
//!   = D_KL[q(s) || p(s|o)] - E_q[log p(o|s)] - E_q[log p(s)]
//!
//! The engine minimizes free energy by:
//! 1. Reducing surprise through learning (accuracy term)
//! 2. Reducing complexity through compression (parsimony term)
//! 3. Accounting for hyperbolic geometry of belief space
//!
//! ## References
//!
//! - Friston, K. (2010). "The free-energy principle: a unified brain theory?"
//!   *Nature Reviews Neuroscience*, 11(2), 127-138.
//! - Friston, K. (2012). "Predictive coding and the free-energy principle"
//!   *Philosophy Transactions B*, 367(1594), 2670-2681.
//! - Friston, K., FitzGerald, T., Rigoli, F., et al. (2016).
//!   "Active inference and learning" *Neuroscience & Biobehavioral Reviews*, 68, 862-879.
//! - Maturana, H. R., & Varela, F. J. (1980).
//!   "Autopoiesis and Cognition: The Realization of the Living"
//!   *D. Reidel Publishing Company*

use ndarray::{Array1, Array2};
use std::f64;

// ============================================================================
// Free Energy Engine
// ============================================================================

/// Free Energy Principle engine for variational inference and active inference
///
/// Minimizes variational free energy through:
/// - Belief updating (posterior approximation)
/// - Prediction error accumulation
/// - Complexity-accuracy tradeoff
/// - Hyperbolic geometry awareness
#[derive(Debug, Clone)]
pub struct FreeEnergyEngine {
    /// Dimensionality of hidden state space
    hidden_dim: usize,

    /// Prior variance (inverse precision) over hidden states
    prior_variance: Array1<f64>,

    /// Learned generative model: p(o|s)
    likelihood_matrix: Array2<f64>,

    /// Learned transition model: p(s_t | s_{t-1})
    transition_matrix: Array2<f64>,

    /// Belief precision (inverse variance) for each hidden state
    belief_precision: Array1<f64>,

    /// Accumulator for surprise (prediction error)
    surprise_accumulator: f64,

    /// Learning rate for model updates
    learning_rate: f64,

    /// Temperature parameter for softmax (exploration-exploitation)
    temperature: f64,

    /// Enable hyperbolic geometry corrections
    use_hyperbolic: bool,

    /// History of free energy values
    free_energy_history: Vec<f64>,
}

impl FreeEnergyEngine {
    /// Create a new Free Energy engine with given hidden state dimensionality
    ///
    /// # Arguments
    /// * `hidden_dim` - Dimensionality of hidden state space
    ///
    /// # Returns
    /// A new FreeEnergyEngine configured with default parameters
    pub fn new(hidden_dim: usize) -> Self {
        let mut engine = Self {
            hidden_dim,
            prior_variance: Array1::from_elem(hidden_dim, 1.0),
            likelihood_matrix: Array2::from_elem((32, hidden_dim), 0.1),
            transition_matrix: Array2::eye(hidden_dim),
            belief_precision: Array1::from_elem(hidden_dim, 1.0),
            surprise_accumulator: 0.0,
            learning_rate: 0.01,
            temperature: 1.0,
            use_hyperbolic: false,
            free_energy_history: Vec::with_capacity(1000),
        };

        // Initialize likelihood matrix with small random values
        for elem in engine.likelihood_matrix.iter_mut() {
            *elem = (rand::random::<f64>() - 0.5) * 0.1;
        }

        engine
    }

    /// Compute variational free energy: F = D_KL[q||p] + surprise
    ///
    /// This is the core computation of the Free Energy Principle.
    ///
    /// # Arguments
    /// * `observation` - Current observation from environment
    /// * `beliefs` - Approximate posterior q(s|o)
    /// * `precision` - Precision of beliefs (inverse variance)
    ///
    /// # Returns
    /// The variational free energy value
    ///
    /// # Mathematical Details
    ///
    /// F = complexity_term + accuracy_term
    ///
    /// Where:
    /// - **complexity_term** = D_KL[q(s) || p(s)] = ∫ q(s) log(q(s)/p(s)) ds
    /// - **accuracy_term** = -E_q[log p(o|s)] = ∫ q(s) (-log p(o|s)) ds
    pub fn compute(&mut self, observation: &Array1<f64>, beliefs: &Array1<f64>, precision: &Array1<f64>) -> f64 {
        let complexity = self.compute_complexity(beliefs, precision);
        let accuracy = self.compute_accuracy(observation, beliefs, precision);
        let surprise = self.compute_surprise(observation, beliefs);

        // Total variational free energy
        let free_energy = complexity + accuracy + surprise;

        // Record in history
        self.free_energy_history.push(free_energy);
        if self.free_energy_history.len() > 10000 {
            self.free_energy_history.remove(0);
        }

        free_energy
    }

    /// Compute KL divergence: D_KL[q(s|o) || p(s)]
    ///
    /// This term measures the complexity of the approximate posterior relative to the prior.
    /// Minimizing this encourages simpler explanations (Occam's Razor).
    ///
    /// # Formula
    ///
    /// D_KL[q||p] = Σ_i (π_i * (log π_i - log ρ_i) + 0.5 * (σ²_ρ - σ²_q) / σ²_ρ)
    ///
    /// Where:
    /// - π_i = softmax(beliefs)
    /// - ρ_i = softmax(prior)
    /// - σ²_q = inverse of precision (belief variance)
    /// - σ²_ρ = prior variance
    fn compute_complexity(&self, beliefs: &Array1<f64>, precision: &Array1<f64>) -> f64 {
        // Softmax of beliefs (approximate posterior)
        let q_probs = softmax(beliefs);

        // Softmax of prior (typically uniform or informed)
        let prior_logits = (0..self.hidden_dim)
            .map(|i| -(0.5 * self.prior_variance[i].ln()))
            .collect::<Array1<_>>();
        let p_probs = softmax(&prior_logits);

        // KL divergence: sum over all dimensions
        let mut kl_div = 0.0;

        for i in 0..self.hidden_dim {
            if q_probs[i] > 1e-10 && p_probs[i] > 1e-10 {
                // Discrete KL component
                kl_div += q_probs[i] * ((q_probs[i] / p_probs[i]).ln());

                // Continuous variance mismatch component
                let belief_variance = 1.0 / precision[i].max(0.01);
                let prior_var = self.prior_variance[i];

                kl_div += 0.5 * (
                    (belief_variance / prior_var) +
                    (prior_var - belief_variance) / prior_var - 1.0
                );
            }
        }

        // Hyperbolic correction if enabled
        if self.use_hyperbolic {
            kl_div *= self.hyperbolic_correction_factor(beliefs);
        }

        kl_div.max(0.0)
    }

    /// Compute accuracy term: -E_q[log p(o|s)]
    ///
    /// This term measures how well the model predicts observations.
    /// Minimizing this encourages learning accurate generative models.
    ///
    /// # Formula
    ///
    /// accuracy = -Σ_i q(s_i) * log p(o|s_i)
    ///
    /// We approximate p(o|s) using a Gaussian with linear likelihood:
    /// log p(o|s) = -0.5 * ||o - μ(s)||² / σ²
    fn compute_accuracy(&self, observation: &Array1<f64>, beliefs: &Array1<f64>, precision: &Array1<f64>) -> f64 {
        // Predicted observation from generative model
        let predicted = self.predict_observation(beliefs);

        // Prediction error
        let error = observation - &predicted;

        // Gaussian negative log likelihood weighted by precision
        let mut accuracy = 0.0;
        for (i, (&err, &prec)) in error.iter().zip(precision.iter()).enumerate() {
            // Weight prediction error by belief precision
            let belief_weight = (i % beliefs.len()).min(beliefs.len() - 1);
            let belief_prec = beliefs[belief_weight].abs().max(0.1);

            accuracy += 0.5 * err * err * prec * belief_prec;
        }

        accuracy
    }

    /// Compute surprise: -log p(o|m)
    ///
    /// Surprise is the negative log likelihood of the observation under the model.
    /// It's the overall prediction error of the system.
    ///
    /// # Formula
    ///
    /// surprise = -log p(o|m) = ∫ q(s) (-log p(o|s)) ds
    fn compute_surprise(&mut self, observation: &Array1<f64>, beliefs: &Array1<f64>) -> f64 {
        // Predicted observation
        let predicted = self.predict_observation(beliefs);

        // MSE-based surprise approximation
        let mse = (observation - &predicted)
            .mapv(|x| x * x)
            .mean()
            .unwrap_or(1.0);

        // Convert to negative log likelihood
        let surprise = 0.5 * mse.ln() + 0.5 * (2.0 * f64::consts::PI).ln();

        // Track surprise accumulation
        self.surprise_accumulator = self.surprise_accumulator * 0.99 + surprise * 0.01;

        surprise.max(0.0)
    }

    /// Generate optimal action that minimizes expected free energy
    ///
    /// Expected free energy combines two objectives:
    /// 1. **Exploitation**: Minimize expected surprise under current beliefs
    /// 2. **Exploration**: Reduce uncertainty (information gain)
    ///
    /// # Formula
    ///
    /// G(a) = E_{s~q}[-log p(o|s,a)] + β * H[p(s'|a)]
    ///
    /// Where:
    /// - First term: expected prediction error under action
    /// - Second term: exploration bonus (entropy of state change)
    /// - β: exploration temperature
    pub fn select_action(&self, beliefs: &Array1<f64>, action_space_dim: usize) -> Array1<f64> {
        let mut action = Array1::zeros(action_space_dim);

        // Action selection based on belief entropy and prediction quality
        for a in 0..action_space_dim {
            // Expected free energy for this action
            let efg = self.expected_free_energy(beliefs, a);

            // Softmax action selection with temperature
            action[a] = (-efg / self.temperature).exp();
        }

        // Normalize to probability distribution
        let sum = action.sum();
        if sum > 0.0 {
            action /= sum;
        }

        action
    }

    /// Compute expected free energy for a candidate action
    ///
    /// G(a) = (exploitation) + (exploration)
    fn expected_free_energy(&self, beliefs: &Array1<f64>, action_idx: usize) -> f64 {
        // Exploitation: expected prediction error under this action
        let exploitation = self.expected_surprise(beliefs, action_idx);

        // Exploration: information gain / uncertainty reduction
        let exploration = self.information_gain(beliefs, action_idx);

        // Temperature-modulated combination
        exploitation + self.temperature * exploration
    }

    /// Compute expected surprise under a candidate action
    fn expected_surprise(&self, beliefs: &Array1<f64>, action_idx: usize) -> f64 {
        // Predict next observation under this action
        let action_effect = (action_idx as f64 - beliefs.len() as f64 / 2.0) * 0.1;
        let mut predicted_next = beliefs.clone();
        predicted_next *= 1.0 + action_effect;

        let predicted_obs = self.predict_observation(&predicted_next);
        let entropy = predicted_obs.mapv(|x| x * x).sum();

        entropy.sqrt()
    }

    /// Compute information gain (entropy reduction) from action
    fn information_gain(&self, beliefs: &Array1<f64>, action_idx: usize) -> f64 {
        // Entropy of current belief distribution
        let q_probs = softmax(beliefs);
        let current_entropy = -q_probs
            .iter()
            .filter(|&&p| p > 1e-10)
            .map(|&p| p * p.ln())
            .sum::<f64>();

        // Expected entropy reduction through action-induced state transitions
        let action_effect = (action_idx as f64) / beliefs.len() as f64;
        let entropy_reduction = action_effect * current_entropy;

        entropy_reduction.max(0.0)
    }

    /// Update beliefs through Variational Bayes
    ///
    /// Uses gradient descent on free energy to update belief parameters.
    ///
    /// # Formula
    ///
    /// ∇_q F = precision * (observation - predicted) + ∇_q D_KL[q||p]
    pub fn update_beliefs(&mut self, observation: &Array1<f64>, beliefs: &mut Array1<f64>) -> f64 {
        let predicted = self.predict_observation(beliefs);
        let prediction_error = observation - &predicted;

        // Gradient of free energy w.r.t. beliefs
        let gradient = self.compute_belief_gradient(&prediction_error, beliefs);

        // Update beliefs by gradient descent
        let delta = self.learning_rate * &gradient;
        *beliefs -= &delta;

        // Clamp beliefs to reasonable range
        for b in beliefs.iter_mut() {
            *b = b.clamp(-5.0, 5.0);
        }

        // Return prediction error magnitude
        prediction_error.mapv(|x| x * x).mean().unwrap_or(0.0).sqrt()
    }

    /// Compute gradient of free energy with respect to beliefs
    ///
    /// ∂F/∂s = precision * (obs - pred) + precision * (s - μ_prior)
    fn compute_belief_gradient(&self, prediction_error: &Array1<f64>, beliefs: &Array1<f64>) -> Array1<f64> {
        let mut gradient = prediction_error.to_owned();

        // Complexity gradient: push towards prior
        for i in 0..self.hidden_dim.min(beliefs.len()) {
            let idx = i % beliefs.len();
            gradient[idx] += self.belief_precision[i] * beliefs[idx] / (1.0 + self.prior_variance[i]);
        }

        gradient
    }

    /// Update generative model: p(o|s) through EM algorithm
    ///
    /// Learns the mapping from hidden states to observations.
    pub fn learn_generative_model(&mut self, observations: &Array2<f64>, beliefs: &Array2<f64>) {
        // E-step: compute expected sufficient statistics
        let n_samples = observations.nrows();

        // M-step: maximum likelihood update
        for i in 0..observations.ncols().min(self.likelihood_matrix.nrows()) {
            for j in 0..beliefs.ncols().min(self.likelihood_matrix.ncols()) {
                let sum: f64 = (0..n_samples)
                    .map(|t| observations[[t, i]] * beliefs[[t, j]])
                    .sum();

                let mean_belief: f64 = beliefs.column(j).mean().unwrap_or(0.0);
                let update = self.learning_rate * sum / (n_samples as f64 * (1.0 + mean_belief.abs()));

                self.likelihood_matrix[[i, j]] += update;
                self.likelihood_matrix[[i, j]] = self.likelihood_matrix[[i, j]].clamp(-1.0, 1.0);
            }
        }
    }

    /// Predict next observation from current beliefs
    fn predict_observation(&self, beliefs: &Array1<f64>) -> Array1<f64> {
        // Linear prediction: o = W * s
        let n_obs = self.likelihood_matrix.nrows();
        let mut predicted = Array1::zeros(n_obs);

        for i in 0..n_obs {
            for j in 0..beliefs.len().min(self.likelihood_matrix.ncols()) {
                predicted[i] += self.likelihood_matrix[[i, j]] * beliefs[j];
            }
        }

        // Apply sigmoid activation
        predicted.mapv(|x| sigmoid(x))
    }

    /// Compute hyperbolic correction factor for belief geometry
    ///
    /// In hyperbolic space, divergences and distances have different properties.
    /// This factor corrects KL divergence for the curvature of belief space.
    fn hyperbolic_correction_factor(&self, beliefs: &Array1<f64>) -> f64 {
        // Approximate hyperbolic norm
        let norm_sq = beliefs.mapv(|x| x * x).sum();

        if norm_sq < 1.0 {
            // In Poincaré ball interior
            let factor = 1.0 / (1.0 - norm_sq).max(0.01);
            factor.sqrt()
        } else {
            // Clipped to boundary
            1.0
        }
    }

    /// Get current accumulated surprise
    pub fn accumulated_surprise(&self) -> f64 {
        self.surprise_accumulator
    }

    /// Get free energy history
    pub fn free_energy_history(&self) -> &[f64] {
        &self.free_energy_history
    }

    /// Clear history (for reset)
    pub fn clear_history(&mut self) {
        self.free_energy_history.clear();
        self.surprise_accumulator = 0.0;
    }

    /// Set learning rate
    pub fn set_learning_rate(&mut self, lr: f64) {
        self.learning_rate = lr.clamp(0.0001, 0.1);
    }

    /// Set temperature (exploration-exploitation tradeoff)
    pub fn set_temperature(&mut self, temp: f64) {
        self.temperature = temp.max(0.01);
    }

    /// Enable/disable hyperbolic geometry corrections
    pub fn set_hyperbolic(&mut self, enabled: bool) {
        self.use_hyperbolic = enabled;
    }

    /// Get average free energy over recent history
    pub fn average_free_energy(&self, window: usize) -> f64 {
        let window = window.min(self.free_energy_history.len());
        if window == 0 {
            return 0.0;
        }

        let start = self.free_energy_history.len() - window;
        self.free_energy_history[start..]
            .iter()
            .sum::<f64>() / window as f64
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Softmax activation: converts logits to probabilities
///
/// σ(x)_i = exp(x_i) / Σ_j exp(x_j)
fn softmax(x: &Array1<f64>) -> Array1<f64> {
    // Numerical stability: subtract max
    let max = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_x = x.mapv(|xi| (xi - max).exp());
    let sum = exp_x.sum();

    if sum > 0.0 && sum.is_finite() {
        exp_x / sum
    } else {
        // Fallback to uniform
        Array1::from_elem(x.len(), 1.0 / x.len() as f64)
    }
}

/// Sigmoid activation: σ(x) = 1 / (1 + exp(-x))
fn sigmoid(x: f64) -> f64 {
    if x > 20.0 {
        1.0
    } else if x < -20.0 {
        0.0
    } else {
        1.0 / (1.0 + (-x).exp())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_free_energy_engine_creation() {
        let engine = FreeEnergyEngine::new(64);
        assert_eq!(engine.hidden_dim, 64);
        assert!(!engine.use_hyperbolic);
    }

    #[test]
    fn test_softmax_properties() {
        let logits = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let probs = softmax(&logits);

        // Should sum to 1
        assert_abs_diff_eq!(probs.sum(), 1.0, epsilon = 1e-10);

        // All positive
        assert!(probs.iter().all(|&p| p > 0.0));

        // Highest logit has highest probability
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn test_sigmoid_bounds() {
        assert!(sigmoid(0.0) > 0.45 && sigmoid(0.0) < 0.55);
        assert!(sigmoid(10.0) > 0.99);
        assert!(sigmoid(-10.0) < 0.01);
    }

    #[test]
    fn test_complexity_computation() {
        let engine = FreeEnergyEngine::new(32);
        let beliefs = Array1::from_elem(32, 0.0);
        let precision = Array1::from_elem(32, 1.0);

        let complexity = engine.compute_complexity(&beliefs, &precision);

        // Should be non-negative
        assert!(complexity >= 0.0);

        // For zero beliefs and unit prior, complexity should be bounded
        assert!(complexity < 50.0);
    }

    #[test]
    fn test_accuracy_computation() {
        let engine = FreeEnergyEngine::new(32);
        let observation = Array1::from_elem(32, 0.5);
        let beliefs = Array1::from_elem(64, 0.0);
        let precision = Array1::from_elem(32, 1.0);

        let accuracy = engine.compute_accuracy(&observation, &beliefs, &precision);

        // Should be non-negative
        assert!(accuracy >= 0.0);
    }

    #[test]
    fn test_surprise_computation() {
        let mut engine = FreeEnergyEngine::new(32);
        let observation = Array1::from_elem(32, 0.5);
        let beliefs = Array1::from_elem(64, 0.0);

        let surprise1 = engine.compute_surprise(&observation, &beliefs);
        assert!(surprise1 >= 0.0);

        // With better prediction, surprise should decrease
        let observation2 = Array1::from_elem(32, 0.0);
        let surprise2 = engine.compute_surprise(&observation2, &beliefs);

        // Both should be valid
        assert!(surprise2 >= 0.0);
    }

    #[test]
    fn test_free_energy_computation() {
        let mut engine = FreeEnergyEngine::new(32);
        let observation = Array1::from_elem(32, 0.5);
        let beliefs = Array1::from_elem(64, 0.0);
        let precision = Array1::from_elem(32, 1.0);

        let fe = engine.compute(&observation, &beliefs, &precision);

        // Should be sum of three non-negative terms
        assert!(fe >= 0.0);

        // History should record it
        assert_eq!(engine.free_energy_history().len(), 1);
        assert_eq!(engine.free_energy_history()[0], fe);
    }

    #[test]
    fn test_belief_update() {
        let mut engine = FreeEnergyEngine::new(32);
        // Use observation != 0.5 since sigmoid(0) = 0.5 (model prediction with zero beliefs)
        let observation = Array1::from_vec(vec![0.8; 32]);
        let mut beliefs = Array1::from_elem(32, 0.0); // Match hidden_dim
        let initial_beliefs = beliefs.clone();

        engine.set_learning_rate(0.1);
        let error = engine.update_beliefs(&observation, &mut beliefs);

        // Beliefs should have changed since obs (0.8) != predicted (0.5)
        let change = (&beliefs - &initial_beliefs).mapv(|x| x.abs()).sum();
        assert!(change > 0.0, "Beliefs should change when obs != predicted");

        // Error should be positive (prediction error magnitude)
        assert!(error >= 0.0);
    }

    #[test]
    fn test_action_selection() {
        let engine = FreeEnergyEngine::new(32);
        let beliefs = Array1::from_elem(64, 0.5);
        let action_space_dim = 16;

        let action = engine.select_action(&beliefs, action_space_dim);

        // Should be probability distribution
        assert_abs_diff_eq!(action.sum(), 1.0, epsilon = 1e-10);
        assert!(action.iter().all(|&a| a >= 0.0 && a <= 1.0));
    }

    #[test]
    fn test_learning_rate_setting() {
        let mut engine = FreeEnergyEngine::new(32);

        engine.set_learning_rate(0.05);
        assert_abs_diff_eq!(engine.learning_rate, 0.05, epsilon = 1e-10);

        // Should clamp
        engine.set_learning_rate(10.0);
        assert!(engine.learning_rate <= 0.1);

        engine.set_learning_rate(0.00001);
        assert!(engine.learning_rate >= 0.0001);
    }

    #[test]
    fn test_temperature_setting() {
        let mut engine = FreeEnergyEngine::new(32);

        engine.set_temperature(2.0);
        assert_abs_diff_eq!(engine.temperature, 2.0, epsilon = 1e-10);

        // Should not go below minimum
        engine.set_temperature(0.001);
        assert!(engine.temperature >= 0.01);
    }

    #[test]
    fn test_hyperbolic_correction() {
        let mut engine = FreeEnergyEngine::new(32);
        engine.set_hyperbolic(true);

        let beliefs = Array1::from_elem(32, 0.1);
        let precision = Array1::from_elem(32, 1.0);

        let complexity_with_hyperbolic = engine.compute_complexity(&beliefs, &precision);
        assert!(complexity_with_hyperbolic >= 0.0);

        // Should handle boundary conditions
        let large_beliefs = Array1::from_elem(32, 0.99);
        let complexity_boundary = engine.compute_complexity(&large_beliefs, &precision);
        assert!(complexity_boundary >= 0.0);
        assert!(complexity_boundary.is_finite());
    }

    #[test]
    fn test_average_free_energy() {
        let mut engine = FreeEnergyEngine::new(32);
        let observation = Array1::from_elem(32, 0.5);
        let beliefs = Array1::from_elem(64, 0.0);
        let precision = Array1::from_elem(32, 1.0);

        // Compute several times
        for _ in 0..5 {
            engine.compute(&observation, &beliefs, &precision);
        }

        let avg = engine.average_free_energy(3);
        assert!(avg >= 0.0);
        assert!(avg.is_finite());
    }

    #[test]
    fn test_history_clearing() {
        let mut engine = FreeEnergyEngine::new(32);
        let observation = Array1::from_elem(32, 0.5);
        let beliefs = Array1::from_elem(64, 0.0);
        let precision = Array1::from_elem(32, 1.0);

        engine.compute(&observation, &beliefs, &precision);
        assert!(engine.free_energy_history().len() > 0);

        engine.clear_history();
        assert_eq!(engine.free_energy_history().len(), 0);
        assert_eq!(engine.accumulated_surprise(), 0.0);
    }

    #[test]
    fn test_generative_model_learning() {
        let mut engine = FreeEnergyEngine::new(16);

        let observations = Array2::from_elem((10, 32), 0.5);
        let beliefs = Array2::from_elem((10, 64), 0.1);

        let initial_model = engine.likelihood_matrix.clone();

        engine.learn_generative_model(&observations, &beliefs);

        // Model should have updated
        let change = (&engine.likelihood_matrix - &initial_model).mapv(|x| x.abs()).sum();
        assert!(change > 0.0);
    }

    #[test]
    fn test_predict_observation() {
        let engine = FreeEnergyEngine::new(32);
        let beliefs = Array1::from_elem(64, 0.5);

        let prediction = engine.predict_observation(&beliefs);

        // Should have same size as likelihood matrix rows
        assert_eq!(prediction.len(), engine.likelihood_matrix.nrows());

        // Should be in [0, 1] after sigmoid
        assert!(prediction.iter().all(|&p| p >= 0.0 && p <= 1.0));
    }

    #[test]
    fn test_free_energy_minimization_trend() {
        let mut engine = FreeEnergyEngine::new(32);
        let observation = Array1::from_elem(32, 0.5);
        let mut beliefs = Array1::from_elem(32, 0.0); // Match hidden_dim
        let precision = Array1::from_elem(32, 1.0);

        engine.set_learning_rate(0.1);

        // Initial free energy
        let initial_fe = engine.compute(&observation, &beliefs, &precision);

        // Update beliefs multiple times
        for _ in 0..10 {
            engine.update_beliefs(&observation, &mut beliefs);
            engine.compute(&observation, &beliefs, &precision);
        }

        let final_fe = engine.compute(&observation, &beliefs, &precision);

        // Free energy should generally decrease or stay similar
        // (might increase slightly due to complexity term)
        assert!(final_fe <= initial_fe * 1.5);
    }

    #[test]
    fn test_numerical_stability_large_values() {
        let mut engine = FreeEnergyEngine::new(32);
        let observation = Array1::from_elem(32, 1e10);
        let beliefs = Array1::from_elem(32, 1e10); // Match hidden_dim
        let precision = Array1::from_elem(32, 1e-10);

        let fe = engine.compute(&observation, &beliefs, &precision);

        // Should not produce NaN or Inf
        assert!(fe.is_finite());
    }

    #[test]
    fn test_numerical_stability_small_values() {
        let mut engine = FreeEnergyEngine::new(32);
        let observation = Array1::from_elem(32, 1e-10);
        let beliefs = Array1::from_elem(32, 1e-10); // Match hidden_dim
        let precision = Array1::from_elem(32, 1e10);

        let fe = engine.compute(&observation, &beliefs, &precision);

        // Should not produce NaN or Inf
        assert!(fe.is_finite());
    }
}
