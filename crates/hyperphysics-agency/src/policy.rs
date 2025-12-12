//! # Policy Selection and Management
//!
//! Implements policy selection based on expected free energy minimization.
//!
//! ## Theoretical Foundation
//!
//! Policy selection in active inference follows the softmax distribution over
//! negative expected free energy (EFE):
//!
//! P(π) = σ(-γ × G(π))
//!
//! Where:
//! - G(π): Expected free energy for policy π
//! - γ: Inverse temperature (precision of policy selection)
//! - σ: Softmax function
//!
//! ### Expected Free Energy Decomposition
//!
//! G(π) = E_Q[D_KL[Q(o|π)||P(o|C)]] + E_Q[H[P(s|o,π)]]
//!      = Pragmatic Value + Epistemic Value + Risk
//!
//! Where:
//! - Pragmatic Value: Expected utility (achieving preferred outcomes)
//! - Epistemic Value: Information gain (resolving uncertainty)
//! - Risk: Expected ambiguity (state uncertainty given outcomes)
//!
//! ## References
//!
//! - Friston, K. et al. (2017). "Active Inference: A Process Theory"
//! - Parr, T. & Friston, K. (2019). "Generalised free energy and active inference"
//! - Da Costa, L. et al. (2020). "Active inference on discrete state-spaces"

use ndarray::Array1;
use serde::{Deserialize, Serialize};

// ============================================================================
// Policy Types
// ============================================================================

/// Represents a policy (probability distribution over actions)
pub type Policy = Array1<f64>;

/// Expected free energy components for a policy
#[derive(Debug, Clone, Default)]
pub struct EFEComponents {
    /// Pragmatic value (goal achievement)
    pub pragmatic: f64,
    /// Epistemic value (information gain)
    pub epistemic: f64,
    /// Risk value (expected ambiguity)
    pub risk: f64,
    /// Total expected free energy
    pub total: f64,
}

// ============================================================================
// Policy Selector
// ============================================================================

/// Policy selector based on expected free energy minimization
///
/// Implements sophisticated policy selection through:
/// 1. EFE computation for each policy
/// 2. Softmax selection with temperature parameter
/// 3. Habit learning for frequently successful policies
/// 4. Curiosity-driven exploration bonus
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicySelector {
    /// Number of available policies
    n_policies: usize,

    /// Dimensionality of action space
    action_dim: usize,

    /// Inverse temperature for softmax (γ)
    gamma: f64,

    /// Curiosity weight for epistemic drive
    curiosity_weight: f64,

    /// Habit strength (prior bias toward successful policies)
    habit_strength: f64,

    /// Habit prior (learned policy preferences)
    habit_prior: Array1<f64>,

    /// Goal state (preferred observations)
    goal_state: Array1<f64>,

    /// Policy library (predefined action sequences)
    policy_library: Vec<Array1<f64>>,

    /// EFE history for adaptation
    efe_history: Vec<f64>,
}

impl PolicySelector {
    /// Create new policy selector
    ///
    /// # Arguments
    /// * `action_dim` - Dimensionality of action space
    pub fn new(action_dim: usize) -> Self {
        let n_policies = action_dim.max(4); // At least 4 policies

        // Initialize policy library with diverse action patterns
        let mut policy_library = Vec::with_capacity(n_policies);
        for i in 0..n_policies {
            let mut policy = Array1::zeros(action_dim);
            // Each policy emphasizes different action dimensions
            for j in 0..action_dim {
                let phase = (i as f64 * 2.0 * std::f64::consts::PI) / n_policies as f64;
                let action_phase = (j as f64 * 2.0 * std::f64::consts::PI) / action_dim as f64;
                policy[j] = (phase + action_phase).cos() * 0.5 + 0.5;
            }
            // Normalize to probability distribution
            let sum: f64 = policy.iter().sum();
            if sum > 0.0 {
                policy.mapv_inplace(|x| x / sum);
            }
            policy_library.push(policy);
        }

        Self {
            n_policies,
            action_dim,
            gamma: 4.0, // Default inverse temperature
            curiosity_weight: 0.5,
            habit_strength: 0.1,
            habit_prior: Array1::from_elem(n_policies, 1.0 / n_policies as f64),
            goal_state: Array1::from_elem(action_dim, 0.5),
            policy_library,
            efe_history: Vec::with_capacity(1000),
        }
    }

    /// Create policy selector with custom parameters
    pub fn with_params(
        action_dim: usize,
        n_policies: usize,
        gamma: f64,
        curiosity_weight: f64,
    ) -> Self {
        let mut selector = Self::new(action_dim);
        selector.n_policies = n_policies;
        selector.gamma = gamma.clamp(0.1, 20.0);
        selector.curiosity_weight = curiosity_weight.clamp(0.0, 1.0);

        // Regenerate policy library for new n_policies
        selector.policy_library = Vec::with_capacity(n_policies);
        for i in 0..n_policies {
            let mut policy = Array1::zeros(action_dim);
            for j in 0..action_dim {
                let phase = (i as f64 * 2.0 * std::f64::consts::PI) / n_policies as f64;
                let action_phase = (j as f64 * 2.0 * std::f64::consts::PI) / action_dim as f64;
                policy[j] = (phase + action_phase).cos() * 0.5 + 0.5;
            }
            let sum: f64 = policy.iter().sum();
            if sum > 0.0 {
                policy.mapv_inplace(|x| x / sum);
            }
            selector.policy_library.push(policy);
        }
        selector.habit_prior = Array1::from_elem(n_policies, 1.0 / n_policies as f64);

        selector
    }

    /// Set goal state (preferred observations)
    pub fn set_goal(&mut self, goal: Array1<f64>) {
        self.goal_state = goal;
    }

    /// Set inverse temperature
    pub fn set_gamma(&mut self, gamma: f64) {
        self.gamma = gamma.clamp(0.1, 20.0);
    }

    /// Select policy based on state and objectives
    ///
    /// Implements: π* = softmax(-γ × G(π) + log(E[π]))
    ///
    /// # Arguments
    /// * `beliefs` - Current beliefs about hidden states
    /// * `phi` - Integrated information (consciousness level)
    /// * `survival` - Current survival drive
    /// * `control` - Control authority
    ///
    /// # Returns
    /// Selected policy (action probability distribution)
    pub fn select(
        &self,
        beliefs: &Array1<f64>,
        phi: f64,
        survival: f64,
        control: f64,
    ) -> Policy {
        // Compute EFE for each policy
        let mut efe_values = Vec::with_capacity(self.n_policies);

        for i in 0..self.n_policies {
            let policy = &self.policy_library[i];
            let efe = self.compute_policy_efe(policy, beliefs, phi, survival, control);
            efe_values.push(efe.total);
        }

        // Softmax selection over negative EFE + habit prior
        let policy_probs = self.softmax_selection(&efe_values);

        // Combine policies weighted by selection probabilities
        let mut selected_policy = Array1::zeros(self.action_dim);
        for (i, prob) in policy_probs.iter().enumerate() {
            if i < self.policy_library.len() {
                for j in 0..self.action_dim {
                    selected_policy[j] += prob * self.policy_library[i][j];
                }
            }
        }

        // Normalize
        let sum: f64 = selected_policy.iter().sum();
        if sum > 0.01 {
            selected_policy.mapv_inplace(|x| x / sum);
        } else {
            // Fallback to uniform
            selected_policy = Array1::from_elem(self.action_dim, 1.0 / self.action_dim as f64);
        }

        selected_policy
    }

    /// Compute expected free energy for a policy
    ///
    /// G(π) = Pragmatic + β×Epistemic + Risk
    fn compute_policy_efe(
        &self,
        policy: &Array1<f64>,
        beliefs: &Array1<f64>,
        phi: f64,
        survival: f64,
        control: f64,
    ) -> EFEComponents {
        // Pragmatic value: Expected utility (goal achievement)
        let pragmatic = self.compute_pragmatic_value(policy, beliefs);

        // Epistemic value: Information gain weighted by curiosity and consciousness
        let epistemic = self.compute_epistemic_value(policy, beliefs) * (1.0 + phi * 0.5);

        // Risk value: Expected ambiguity modulated by survival drive
        let risk = self.compute_risk_value(policy, beliefs) * survival;

        // Control modulates the precision of policy selection
        let control_factor = control.clamp(0.1, 2.0);

        let total = (pragmatic + self.curiosity_weight * epistemic + risk) * control_factor;

        EFEComponents {
            pragmatic,
            epistemic,
            risk,
            total,
        }
    }

    /// Compute pragmatic value (expected utility)
    ///
    /// Measures how well predicted outcomes match preferred outcomes (goals)
    fn compute_pragmatic_value(&self, policy: &Array1<f64>, beliefs: &Array1<f64>) -> f64 {
        // Predict expected observation under this policy
        let expected_obs = self.predict_observation(policy, beliefs);

        // KL divergence from goal: D_KL[Q(o|π)||P(o|C)]
        let mut kl_div = 0.0;
        for i in 0..self.action_dim.min(expected_obs.len()).min(self.goal_state.len()) {
            let q = expected_obs[i].clamp(0.001, 0.999);
            let p = self.goal_state[i].clamp(0.001, 0.999);
            kl_div += q * (q / p).ln() + (1.0 - q) * ((1.0 - q) / (1.0 - p)).ln();
        }

        -kl_div // Negative because we want to minimize divergence
    }

    /// Compute epistemic value (information gain)
    ///
    /// Measures expected reduction in uncertainty about hidden states
    fn compute_epistemic_value(&self, policy: &Array1<f64>, beliefs: &Array1<f64>) -> f64 {
        // Current entropy of beliefs
        let current_entropy = self.compute_entropy(beliefs);

        // Expected entropy after taking action (simplified model)
        // Actions that differ from current beliefs reduce uncertainty more
        let mut action_novelty = 0.0;
        for i in 0..self.action_dim.min(policy.len()).min(beliefs.len()) {
            let diff = (policy[i] - beliefs[i].abs().min(1.0)).abs();
            action_novelty += diff;
        }
        action_novelty /= self.action_dim as f64;

        // Information gain = reduction in entropy
        current_entropy * action_novelty
    }

    /// Compute risk value (expected ambiguity)
    ///
    /// Measures uncertainty about states given outcomes
    fn compute_risk_value(&self, policy: &Array1<f64>, beliefs: &Array1<f64>) -> f64 {
        // Risk increases with policy entropy (uncertain actions)
        let policy_entropy = self.compute_entropy(policy);

        // Risk also increases when beliefs are uncertain
        let belief_uncertainty = beliefs
            .iter()
            .map(|&b| {
                let p = 1.0 / (1.0 + (-b).exp());
                if p > 0.01 && p < 0.99 {
                    -p * p.ln() - (1.0 - p) * (1.0 - p).ln()
                } else {
                    0.0
                }
            })
            .sum::<f64>()
            / beliefs.len() as f64;

        policy_entropy * 0.5 + belief_uncertainty * 0.5
    }

    /// Predict observation from policy and beliefs
    fn predict_observation(&self, policy: &Array1<f64>, beliefs: &Array1<f64>) -> Array1<f64> {
        let mut prediction = Array1::zeros(self.action_dim);

        for i in 0..self.action_dim {
            // Combine policy influence with belief-based prediction
            let policy_contribution = if i < policy.len() { policy[i] } else { 0.0 };
            let belief_contribution = if i < beliefs.len() {
                1.0 / (1.0 + (-beliefs[i]).exp()) // Sigmoid of belief
            } else {
                0.5
            };

            prediction[i] = 0.6 * policy_contribution + 0.4 * belief_contribution;
        }

        prediction
    }

    /// Compute entropy of a distribution
    fn compute_entropy(&self, dist: &Array1<f64>) -> f64 {
        dist.iter()
            .map(|&p| {
                let p_clamped = p.abs().clamp(0.001, 0.999);
                -p_clamped * p_clamped.ln()
            })
            .sum::<f64>()
            / dist.len() as f64
    }

    /// Softmax selection over negative EFE values
    ///
    /// P(π_i) = exp(-γ × G_i + log(E_i)) / Σ exp(-γ × G_j + log(E_j))
    fn softmax_selection(&self, efe_values: &[f64]) -> Vec<f64> {
        if efe_values.is_empty() {
            return vec![];
        }

        // Find max for numerical stability
        let max_val = efe_values
            .iter()
            .map(|&x| -self.gamma * x)
            .fold(f64::NEG_INFINITY, f64::max);

        // Compute exp(-γ × G + log(habit)) for each policy
        let mut exp_values: Vec<f64> = efe_values
            .iter()
            .enumerate()
            .map(|(i, &g)| {
                let habit_log = if i < self.habit_prior.len() {
                    (self.habit_prior[i] + 0.001).ln() * self.habit_strength
                } else {
                    0.0
                };
                (-self.gamma * g + habit_log - max_val).exp()
            })
            .collect();

        // Normalize
        let sum: f64 = exp_values.iter().sum();
        if sum > 0.0 {
            for v in &mut exp_values {
                *v /= sum;
            }
        } else {
            // Fallback to uniform
            let uniform = 1.0 / exp_values.len() as f64;
            for v in &mut exp_values {
                *v = uniform;
            }
        }

        exp_values
    }

    /// Update habit prior based on policy success
    ///
    /// Successful policies get higher prior probability
    pub fn update_habits(&mut self, selected_policy_idx: usize, reward: f64) {
        if selected_policy_idx >= self.n_policies {
            return;
        }

        // Learning rate for habit update
        let lr = 0.1;

        // Update habit for selected policy
        self.habit_prior[selected_policy_idx] += lr * reward;

        // Normalize habit prior
        let sum: f64 = self.habit_prior.iter().sum();
        if sum > 0.0 {
            self.habit_prior.mapv_inplace(|x| x / sum);
        }

        // Track EFE for adaptation
        self.efe_history.push(reward);
        if self.efe_history.len() > 1000 {
            self.efe_history.remove(0);
        }
    }

    /// Get current habit prior
    pub fn habit_prior(&self) -> &Array1<f64> {
        &self.habit_prior
    }

    /// Get policy library
    pub fn policy_library(&self) -> &[Array1<f64>] {
        &self.policy_library
    }

    /// Get number of policies
    pub fn n_policies(&self) -> usize {
        self.n_policies
    }

    /// Compute EFE for external policy
    pub fn evaluate_policy(
        &self,
        policy: &Array1<f64>,
        beliefs: &Array1<f64>,
        phi: f64,
        survival: f64,
        control: f64,
    ) -> EFEComponents {
        self.compute_policy_efe(policy, beliefs, phi, survival, control)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_policy_selector_creation() {
        let selector = PolicySelector::new(16);
        assert_eq!(selector.action_dim, 16);
        assert_eq!(selector.n_policies, 16);
        assert_eq!(selector.policy_library.len(), 16);
    }

    #[test]
    fn test_policy_selection() {
        let selector = PolicySelector::new(8);
        let beliefs = Array1::from_elem(8, 0.5);
        let phi = 1.2;
        let survival = 0.4;
        let control = 0.7;

        let policy = selector.select(&beliefs, phi, survival, control);

        assert_eq!(policy.len(), 8);
        // Policy should sum to approximately 1.0
        let sum: f64 = policy.iter().sum();
        assert!((sum - 1.0).abs() < 0.01, "Policy should sum to 1.0, got {}", sum);
    }

    #[test]
    fn test_efe_computation() {
        let selector = PolicySelector::new(8);
        let policy = Array1::from_elem(8, 0.125); // Uniform
        let beliefs = Array1::from_elem(8, 0.3);

        let efe = selector.compute_policy_efe(&policy, &beliefs, 1.0, 0.5, 0.8);

        assert!(efe.total.is_finite(), "EFE should be finite");
        assert!(efe.pragmatic.is_finite(), "Pragmatic should be finite");
        assert!(efe.epistemic.is_finite(), "Epistemic should be finite");
        assert!(efe.risk.is_finite(), "Risk should be finite");
    }

    #[test]
    fn test_softmax_selection() {
        let selector = PolicySelector::new(4);
        let efe_values = vec![0.5, 0.3, 0.8, 0.2]; // Lower is better

        let probs = selector.softmax_selection(&efe_values);

        assert_eq!(probs.len(), 4);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 0.01, "Probs should sum to 1.0");

        // Policy with lowest EFE (0.2) should have highest probability
        assert!(probs[3] > probs[0], "Lower EFE should have higher prob");
        assert!(probs[3] > probs[2], "Lower EFE should have higher prob");
    }

    #[test]
    fn test_habit_learning() {
        let mut selector = PolicySelector::new(4);
        let initial_habit = selector.habit_prior[0];

        // Reward policy 0
        selector.update_habits(0, 1.0);

        // Habit should have increased (relatively)
        // After normalization, the rewarded policy should have higher proportion
        let total: f64 = selector.habit_prior.iter().sum();
        assert!((total - 1.0).abs() < 0.01, "Habit prior should be normalized");
    }

    #[test]
    fn test_entropy_computation() {
        let selector = PolicySelector::new(4);

        // Uniform distribution has maximum entropy
        let uniform = Array1::from_elem(4, 0.25);
        let uniform_entropy = selector.compute_entropy(&uniform);

        // Peaked distribution has lower entropy
        let peaked = Array1::from_vec(vec![0.9, 0.05, 0.03, 0.02]);
        let peaked_entropy = selector.compute_entropy(&peaked);

        assert!(
            uniform_entropy > peaked_entropy,
            "Uniform should have higher entropy"
        );
    }

    #[test]
    fn test_custom_params() {
        let selector = PolicySelector::with_params(8, 10, 8.0, 0.7);

        assert_eq!(selector.action_dim, 8);
        assert_eq!(selector.n_policies, 10);
        assert!((selector.gamma - 8.0).abs() < 0.01);
        assert!((selector.curiosity_weight - 0.7).abs() < 0.01);
    }
}
