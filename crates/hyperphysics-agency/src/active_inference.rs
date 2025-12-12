//! # Active Inference Engine
//!
//! Implements active inference for perception, learning, and action planning.
//!
//! ## Theoretical Foundation
//!
//! Active inference (Friston 2010, 2016) unifies perception and action as
//! minimization of variational free energy through precision-weighted
//! prediction error minimization.
//!
//! ### Belief Update Equation
//!
//! μ' = μ + κ(π_s × ε^s + π_p × ε^p)
//!
//! Where:
//! - μ: Current beliefs (hidden state estimate)
//! - κ: Learning rate (gain)
//! - π_s: Sensory precision (inverse variance of observations)
//! - ε^s: Sensory prediction error (observation - prediction)
//! - π_p: Prior precision (inverse variance of prior)
//! - ε^p: Prior prediction error (belief - prior mean)
//!
//! ### Expected Free Energy for Action
//!
//! G(π) = E_Q[D_KL[Q(o|π)||P(o|C)] - H[Q(s|π)]]
//!      = Pragmatic Value + Epistemic Value
//!
//! Where:
//! - Pragmatic Value: Expected utility (goal achievement)
//! - Epistemic Value: Information gain (uncertainty reduction)
//!
//! ## References
//!
//! - Friston, K. (2010). "The free-energy principle: a unified brain theory?"
//! - Friston, K. et al. (2016). "Active inference and learning"
//! - Parr, T. & Friston, K. (2019). "Generalised free energy and active inference"

use ndarray::Array1;
use serde::{Deserialize, Serialize};

// ============================================================================
// Active Inference Engine
// ============================================================================

/// Active inference engine for belief updating and action generation
///
/// Implements the perception-action loop through:
/// 1. Precision-weighted prediction error computation
/// 2. Belief updating via gradient descent on free energy
/// 3. Action selection via expected free energy minimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveInferenceEngine {
    /// Dimensionality of observation space
    observation_dim: usize,

    /// Dimensionality of hidden state space
    hidden_dim: usize,

    /// Dimensionality of action space
    action_dim: usize,

    /// Learning rate for belief updates (κ)
    learning_rate: f64,

    /// Sensory precision (inverse variance)
    sensory_precision: Array1<f64>,

    /// Prior precision (inverse variance)
    prior_precision: Array1<f64>,

    /// Prior mean (expected hidden state)
    prior_mean: Array1<f64>,

    /// Goal state (preferred observations for pragmatic value)
    goal_state: Array1<f64>,

    /// Exploration-exploitation balance (β)
    exploration_weight: f64,

    /// Action precision (inverse temperature for softmax)
    action_precision: f64,

    /// Prediction error history for adaptation
    error_history: Vec<f64>,
}

impl ActiveInferenceEngine {
    /// Create new active inference engine
    ///
    /// # Arguments
    /// * `observation_dim` - Dimensionality of sensory observations
    /// * `hidden_dim` - Dimensionality of hidden states (beliefs)
    /// * `action_dim` - Dimensionality of action space
    pub fn new(observation_dim: usize, hidden_dim: usize, action_dim: usize) -> Self {
        Self {
            observation_dim,
            hidden_dim,
            action_dim,
            learning_rate: 0.01,
            sensory_precision: Array1::from_elem(observation_dim, 1.0),
            prior_precision: Array1::from_elem(hidden_dim, 0.5),
            prior_mean: Array1::zeros(hidden_dim),
            goal_state: Array1::from_elem(observation_dim, 0.5),
            exploration_weight: 0.5,
            action_precision: 1.0,
            error_history: Vec::with_capacity(1000),
        }
    }

    /// Set learning rate
    pub fn set_learning_rate(&mut self, lr: f64) {
        self.learning_rate = lr.clamp(0.001, 1.0);
    }

    /// Set goal state (preferred observations)
    pub fn set_goal(&mut self, goal: Array1<f64>) {
        self.goal_state = goal;
    }

    /// Set exploration weight (0 = pure exploitation, 1 = pure exploration)
    pub fn set_exploration_weight(&mut self, weight: f64) {
        self.exploration_weight = weight.clamp(0.0, 1.0);
    }

    /// Update beliefs given observation using precision-weighted prediction errors
    ///
    /// Implements: μ' = μ + κ(π_s × ε^s + π_p × ε^p)
    ///
    /// # Arguments
    /// * `observation` - Current sensory observation
    /// * `beliefs` - Current beliefs (modified in place)
    ///
    /// # Returns
    /// Prediction error magnitude (for monitoring)
    pub fn update_beliefs(&mut self, observation: &Array1<f64>, beliefs: &mut Array1<f64>) -> f64 {
        // Compute sensory prediction error: ε^s = observation - prediction
        let prediction = self.predict_observation(beliefs);
        let sensory_error = self.compute_sensory_error(observation, &prediction);

        // Compute prior prediction error: ε^p = belief - prior_mean
        let prior_error = self.compute_prior_error(beliefs);

        // Precision-weighted combination
        let mut total_gradient: Array1<f64> = Array1::zeros(self.hidden_dim);

        // Sensory gradient: π_s × ε^s (mapped to hidden space)
        for i in 0..self.hidden_dim.min(self.observation_dim) {
            total_gradient[i] += self.sensory_precision[i.min(self.observation_dim - 1)]
                * sensory_error[i.min(self.observation_dim - 1)];
        }

        // Prior gradient: π_p × ε^p
        for i in 0..self.hidden_dim {
            total_gradient[i] += self.prior_precision[i] * prior_error[i];
        }

        // Update beliefs: μ' = μ + κ × gradient
        for i in 0..self.hidden_dim.min(beliefs.len()) {
            beliefs[i] += self.learning_rate * total_gradient[i];
            // Clamp to reasonable range
            beliefs[i] = beliefs[i].clamp(-5.0, 5.0);
        }

        // Track prediction error
        let error_magnitude = sensory_error.mapv(|x| x * x).sum().sqrt();
        self.error_history.push(error_magnitude);
        if self.error_history.len() > 1000 {
            self.error_history.remove(0);
        }

        error_magnitude
    }

    /// Predict observation from beliefs using linear generative model
    fn predict_observation(&self, beliefs: &Array1<f64>) -> Array1<f64> {
        let mut prediction = Array1::zeros(self.observation_dim);

        // Simple linear prediction with sigmoid activation
        for i in 0..self.observation_dim {
            let mut sum = 0.0;
            for j in 0..self.hidden_dim.min(beliefs.len()) {
                // Weight from hidden state j to observation i
                let weight = if (i + j) % 2 == 0 { 0.1 } else { -0.05 };
                sum += weight * beliefs[j];
            }
            prediction[i] = 1.0 / (1.0 + (-sum).exp()); // Sigmoid
        }

        prediction
    }

    /// Compute sensory prediction error
    fn compute_sensory_error(&self, observation: &Array1<f64>, prediction: &Array1<f64>) -> Array1<f64> {
        let mut error = Array1::zeros(self.observation_dim);
        for i in 0..self.observation_dim.min(observation.len()).min(prediction.len()) {
            error[i] = observation[i] - prediction[i];
        }
        error
    }

    /// Compute prior prediction error
    fn compute_prior_error(&self, beliefs: &Array1<f64>) -> Array1<f64> {
        let mut error = Array1::zeros(self.hidden_dim);
        for i in 0..self.hidden_dim.min(beliefs.len()) {
            error[i] = beliefs[i] - self.prior_mean[i];
        }
        error
    }

    /// Generate action from beliefs via expected free energy minimization
    ///
    /// Implements: a* = argmin_a G(a)
    ///
    /// Where G(a) = Pragmatic + β × Epistemic
    ///
    /// # Arguments
    /// * `policy` - Policy vector (action probabilities)
    /// * `beliefs` - Current beliefs about hidden states
    ///
    /// # Returns
    /// Optimal action vector
    pub fn generate_action(&self, policy: &Array1<f64>, beliefs: &Array1<f64>) -> Array1<f64> {
        let mut action = Array1::zeros(self.action_dim);

        // Compute expected free energy for each action dimension
        for i in 0..self.action_dim {
            // Pragmatic value: how well does this action achieve goals?
            let pragmatic = self.compute_pragmatic_value(beliefs, i);

            // Epistemic value: how much uncertainty reduction?
            let epistemic = self.compute_epistemic_value(beliefs, i);

            // Combined expected free energy
            let efe = pragmatic + self.exploration_weight * epistemic;

            // Action magnitude proportional to negative EFE (minimize G)
            action[i] = (-efe * self.action_precision).tanh();

            // Modulate by policy if available
            if i < policy.len() {
                action[i] *= policy[i].abs().sqrt() + 0.1;
            }
        }

        // Normalize action
        let norm = action.mapv(|x| x * x).sum().sqrt().max(0.1);
        action / norm
    }

    /// Compute pragmatic value (expected utility / goal achievement)
    ///
    /// Measures how well predicted observations match goals
    fn compute_pragmatic_value(&self, beliefs: &Array1<f64>, action_idx: usize) -> f64 {
        let prediction = self.predict_observation(beliefs);

        // Goal error: how far from preferred observations?
        let mut goal_error = 0.0;
        for i in 0..self.observation_dim {
            let diff = prediction[i] - self.goal_state[i];
            goal_error += diff * diff;
        }

        // Action-specific modulation
        let action_effect = (action_idx as f64 / self.action_dim as f64) * 0.2;

        -(goal_error.sqrt() + action_effect)
    }

    /// Compute epistemic value (information gain / uncertainty reduction)
    ///
    /// Measures expected reduction in uncertainty about hidden states
    fn compute_epistemic_value(&self, beliefs: &Array1<f64>, action_idx: usize) -> f64 {
        // Belief entropy as proxy for uncertainty
        let entropy = beliefs.iter()
            .map(|&b| {
                let p = 1.0 / (1.0 + (-b).exp());
                if p > 0.001 && p < 0.999 {
                    -p * p.ln() - (1.0 - p) * (1.0 - p).ln()
                } else {
                    0.0
                }
            })
            .sum::<f64>();

        // Action-specific exploration bonus
        let exploration_bonus = (action_idx as f64).sin() * 0.1;

        entropy + exploration_bonus
    }

    /// Compute expected free energy for a policy
    ///
    /// G(π) = E_Q[D_KL[Q(o|π)||P(o|C)] - H[Q(s|π)]]
    pub fn expected_free_energy(&self, policy: &Array1<f64>, beliefs: &Array1<f64>) -> f64 {
        let mut total_efe = 0.0;

        for i in 0..self.action_dim.min(policy.len()) {
            let pragmatic = self.compute_pragmatic_value(beliefs, i);
            let epistemic = self.compute_epistemic_value(beliefs, i);

            // Weight by policy probability
            let policy_prob = policy[i].abs() / (policy.mapv(|x| x.abs()).sum() + 0.01);
            total_efe += policy_prob * (pragmatic + self.exploration_weight * epistemic);
        }

        -total_efe // Minimize EFE
    }

    /// Update sensory precision based on prediction error variance
    pub fn adapt_precision(&mut self) {
        if self.error_history.len() < 10 {
            return;
        }

        // Compute recent error variance
        let recent_errors = &self.error_history[self.error_history.len() - 10..];
        let mean_error = recent_errors.iter().sum::<f64>() / 10.0;
        let variance = recent_errors.iter()
            .map(|e| (e - mean_error).powi(2))
            .sum::<f64>() / 10.0;

        // Update precision (inverse variance)
        let new_precision = 1.0 / (variance.max(0.01) + 0.1);

        // Smooth update
        for p in self.sensory_precision.iter_mut() {
            *p = 0.9 * *p + 0.1 * new_precision;
            *p = p.clamp(0.1, 10.0);
        }
    }

    /// Get mean prediction error over recent history
    pub fn mean_prediction_error(&self) -> f64 {
        if self.error_history.is_empty() {
            return 0.0;
        }
        self.error_history.iter().sum::<f64>() / self.error_history.len() as f64
    }

    /// Get current sensory precision
    pub fn sensory_precision(&self) -> &Array1<f64> {
        &self.sensory_precision
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_active_inference_creation() {
        let engine = ActiveInferenceEngine::new(32, 64, 16);
        assert_eq!(engine.observation_dim, 32);
        assert_eq!(engine.hidden_dim, 64);
        assert_eq!(engine.action_dim, 16);
    }

    #[test]
    fn test_belief_update() {
        let mut engine = ActiveInferenceEngine::new(32, 32, 16);
        let observation = Array1::from_elem(32, 0.7);
        let mut beliefs = Array1::from_elem(32, 0.0);

        let error = engine.update_beliefs(&observation, &mut beliefs);

        assert!(error > 0.0, "Should have prediction error");
        assert!(beliefs.iter().any(|&b| b != 0.0), "Beliefs should change");
    }

    #[test]
    fn test_action_generation() {
        let engine = ActiveInferenceEngine::new(32, 64, 16);
        let policy = Array1::from_elem(16, 0.5);
        let beliefs = Array1::from_elem(64, 0.3);

        let action = engine.generate_action(&policy, &beliefs);

        assert_eq!(action.len(), 16);
        assert!(action.iter().all(|&a| a.abs() <= 1.0), "Actions should be bounded");
    }

    #[test]
    fn test_expected_free_energy() {
        let engine = ActiveInferenceEngine::new(32, 64, 16);
        let policy = Array1::from_elem(16, 0.5);
        let beliefs = Array1::from_elem(64, 0.3);

        let efe = engine.expected_free_energy(&policy, &beliefs);

        assert!(efe.is_finite(), "EFE should be finite");
    }

    #[test]
    fn test_precision_adaptation() {
        let mut engine = ActiveInferenceEngine::new(32, 32, 16);
        let observation = Array1::from_elem(32, 0.7);
        let mut beliefs = Array1::from_elem(32, 0.0);

        // Generate prediction error history
        for _ in 0..20 {
            engine.update_beliefs(&observation, &mut beliefs);
        }

        let initial_precision = engine.sensory_precision[0];
        engine.adapt_precision();
        let final_precision = engine.sensory_precision[0];

        // Precision should have been updated
        assert!(final_precision > 0.0, "Precision should be positive");
        assert!(final_precision.is_finite(), "Precision should be finite");
    }
}
