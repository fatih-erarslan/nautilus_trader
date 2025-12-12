//! # Layer 3: Decision Making API (Active Inference)
//!
//! Active inference framework for decision-making under uncertainty.
//!
//! ## Scientific Foundation
//!
//! **Free Energy Principle** (Karl Friston):
//! - Minimize surprise = Minimize variational free energy
//! - Expected Free Energy (EFE) = Expected surprise - Expected information gain
//!
//! ## Key Equations
//!
//! ```text
//! Free Energy:
//!   F = E_q[ln q(s) - ln p(o,s)] = Complexity - Accuracy
//!
//! Expected Free Energy:
//!   G = E_q[ln q(s|π) - ln p(o,s|π)]
//!     = Pragmatic value (goal-seeking)
//!     - Epistemic value (information-seeking)
//!
//! Action Selection:
//!   π* = argmin_π G(π)
//! ```
//!
//! ## References
//! - Friston et al. (2015). Active inference and epistemic value. Cognitive Neuroscience.
//! - Da Costa et al. (2020). Active inference on discrete state-spaces. Entropy.

use crate::{Result, QksError};
use std::collections::HashMap;

/// Minimum precision value to avoid numerical issues
pub const MIN_PRECISION: f64 = 1e-8;

/// Default temperature for policy selection (lower = more deterministic)
pub const POLICY_TEMPERATURE: f64 = 1.0;

/// Belief state over hidden states
#[derive(Debug, Clone)]
pub struct BeliefState {
    /// Probability distribution over states
    pub probabilities: Vec<f64>,
    /// State labels
    pub states: Vec<String>,
    /// Precision (inverse variance)
    pub precision: f64,
}

/// Policy (sequence of actions)
#[derive(Debug, Clone)]
pub struct Policy {
    /// Unique policy identifier
    pub id: String,
    /// Sequence of actions
    pub actions: Vec<String>,
    /// Expected free energy
    pub expected_free_energy: f64,
    /// Probability of selection
    pub probability: f64,
}

/// Preferences (C-vector in active inference)
#[derive(Debug, Clone)]
pub struct Preferences {
    /// Preferred observations
    pub observations: HashMap<String, f64>,
    /// Precision of preferences
    pub precision: f64,
}

/// Action with expected outcomes
#[derive(Debug, Clone)]
pub struct Action {
    /// Action identifier
    pub id: String,
    /// Expected state transitions
    pub transitions: HashMap<String, f64>,
    /// Expected observations
    pub observations: HashMap<String, f64>,
}

impl BeliefState {
    /// Create uniform belief state
    pub fn uniform(states: Vec<String>) -> Self {
        let n = states.len();
        Self {
            probabilities: vec![1.0 / n as f64; n],
            states,
            precision: 1.0,
        }
    }

    /// Entropy of belief state
    pub fn entropy(&self) -> f64 {
        -self
            .probabilities
            .iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| p * p.ln())
            .sum::<f64>()
    }

    /// Update belief using Bayes rule
    pub fn update(&mut self, observation: &str, likelihood: &HashMap<String, f64>) -> Result<()> {
        // TODO: Bayesian update
        Ok(())
    }
}

/// Compute Expected Free Energy (EFE) for a policy
///
/// # Arguments
/// * `policy` - Policy to evaluate
/// * `beliefs` - Current belief state
/// * `preferences` - Goal preferences
///
/// # Returns
/// Expected free energy G(π)
///
/// # Formula
/// ```text
/// G(π) = E_q[ln q(o|π) - ln p(o|C)]  (pragmatic value)
///      - E_q[H[p(s|o,π)]]            (epistemic value)
/// ```
///
/// # Example
/// ```rust,ignore
/// let efe = compute_efe(&policy, &beliefs, &preferences)?;
/// println!("EFE = {}", efe); // Lower is better
/// ```
pub fn compute_efe(
    policy: &Policy,
    beliefs: &BeliefState,
    preferences: &Preferences,
) -> Result<f64> {
    // Pragmatic value: How well does this policy achieve our goals?
    let pragmatic_value = compute_pragmatic_value(policy, preferences)?;

    // Epistemic value: How much information do we gain?
    let epistemic_value = compute_epistemic_value(policy, beliefs)?;

    // EFE = Pragmatic - Epistemic (both positive, want to minimize)
    let efe = pragmatic_value - epistemic_value;

    Ok(efe)
}

/// Compute pragmatic value (goal-seeking)
///
/// # Arguments
/// * `policy` - Policy to evaluate
/// * `preferences` - Goal preferences
///
/// # Returns
/// Pragmatic value (KL divergence from preferred outcomes)
fn compute_pragmatic_value(policy: &Policy, preferences: &Preferences) -> Result<f64> {
    // TODO: Implement full KL divergence calculation
    // For now, return simple distance from preferences
    Ok(0.0)
}

/// Compute epistemic value (information-seeking)
///
/// # Arguments
/// * `policy` - Policy to evaluate
/// * `beliefs` - Current belief state
///
/// # Returns
/// Expected information gain
fn compute_epistemic_value(policy: &Policy, beliefs: &BeliefState) -> Result<f64> {
    // Expected reduction in entropy
    // H[q(s)] - E[H[q(s|o,π)]]
    let current_entropy = beliefs.entropy();

    // TODO: Compute expected posterior entropy after observations
    let expected_posterior_entropy = current_entropy * 0.9; // Placeholder

    let information_gain = current_entropy - expected_posterior_entropy;

    Ok(information_gain.max(0.0))
}

/// Select action using active inference
///
/// # Arguments
/// * `beliefs` - Current belief state over hidden states
/// * `preferences` - Goal preferences (C-vector)
/// * `available_policies` - Set of candidate policies
///
/// # Returns
/// Selected action (first action of best policy)
///
/// # Example
/// ```rust,ignore
/// let action = select_action(&beliefs, &preferences)?;
/// println!("Selected action: {}", action);
/// ```
pub fn select_action(
    beliefs: &BeliefState,
    preferences: &Preferences,
) -> Result<String> {
    // TODO: Generate or retrieve available policies
    let policies = generate_policies(beliefs)?;

    // Compute EFE for each policy
    let mut policy_efes: Vec<(String, f64)> = Vec::new();
    for policy in &policies {
        let efe = compute_efe(policy, beliefs, preferences)?;
        policy_efes.push((policy.id.clone(), efe));
    }

    // Select policy with minimum EFE (softmax for stochasticity)
    let selected_policy = softmax_selection(&policy_efes, POLICY_TEMPERATURE)?;

    // Return first action of selected policy
    let policy = policies
        .iter()
        .find(|p| p.id == selected_policy)
        .ok_or_else(|| QksError::Internal("Policy not found".to_string()))?;

    policy
        .actions
        .first()
        .cloned()
        .ok_or_else(|| QksError::Internal("Empty policy".to_string()))
}

/// Generate candidate policies
fn generate_policies(_beliefs: &BeliefState) -> Result<Vec<Policy>> {
    // TODO: Implement policy generation (e.g., tree search, habits)
    Ok(vec![
        Policy {
            id: "explore".to_string(),
            actions: vec!["look_around".to_string()],
            expected_free_energy: 0.0,
            probability: 0.0,
        },
        Policy {
            id: "exploit".to_string(),
            actions: vec!["take_action".to_string()],
            expected_free_energy: 0.0,
            probability: 0.0,
        },
    ])
}

/// Softmax selection from (id, value) pairs
fn softmax_selection(values: &[(String, f64)], temperature: f64) -> Result<String> {
    if values.is_empty() {
        return Err(QksError::InvalidConfig("Empty value array".to_string()));
    }

    // Convert to negative for minimization (exp(-EFE/T))
    let max_val = values
        .iter()
        .map(|(_, v)| -v)
        .fold(f64::NEG_INFINITY, f64::max);

    let exp_sum: f64 = values
        .iter()
        .map(|(_, v)| ((-v - max_val) / temperature).exp())
        .sum();

    // Sample using inverse CDF
    let mut rng = rand::thread_rng();
    use rand::Rng;
    let u: f64 = rng.gen();

    let mut cumulative = 0.0;
    for (id, v) in values {
        let prob = ((-v - max_val) / temperature).exp() / exp_sum;
        cumulative += prob;
        if u <= cumulative {
            return Ok(id.clone());
        }
    }

    Ok(values.last().unwrap().0.clone())
}

/// Perform Bayesian inference on hidden states
///
/// # Arguments
/// * `prior` - Prior belief state
/// * `observation` - Observed outcome
/// * `likelihood` - Likelihood model p(o|s)
///
/// # Returns
/// Updated posterior belief state
///
/// # Example
/// ```rust,ignore
/// let posterior = infer_state(&prior, &observation, &likelihood)?;
/// ```
pub fn infer_state(
    prior: &BeliefState,
    observation: &str,
    likelihood: &HashMap<String, f64>,
) -> Result<BeliefState> {
    let mut posterior = prior.clone();

    // Bayes rule: p(s|o) ∝ p(o|s) p(s)
    for (i, state) in prior.states.iter().enumerate() {
        let p_o_given_s = likelihood.get(state).copied().unwrap_or(0.0);
        let p_s = prior.probabilities[i];
        posterior.probabilities[i] = p_o_given_s * p_s;
    }

    // Normalize
    let total: f64 = posterior.probabilities.iter().sum();
    if total > 0.0 {
        for p in &mut posterior.probabilities {
            *p /= total;
        }
    }

    Ok(posterior)
}

/// Compute variational free energy F
///
/// # Arguments
/// * `beliefs` - Approximate posterior q(s)
/// * `observations` - Observed data
/// * `preferences` - Prior preferences p(o)
///
/// # Returns
/// Variational free energy F
///
/// # Formula
/// ```text
/// F = E_q[ln q(s) - ln p(o,s)]
///   = Complexity - Accuracy
///   = KL[q(s)||p(s)] - E_q[ln p(o|s)]
/// ```
pub fn variational_free_energy(
    beliefs: &BeliefState,
    observations: &[String],
    preferences: &Preferences,
) -> f64 {
    // Complexity term: KL divergence from prior
    let complexity = beliefs.entropy(); // Simplified: H[q]

    // Accuracy term: Expected log-likelihood
    let accuracy = 0.0; // TODO: Compute E_q[ln p(o|s)]

    complexity - accuracy
}

/// Minimize free energy through belief updates
///
/// # Arguments
/// * `observations` - Sequence of observations
/// * `max_iterations` - Maximum iterations for optimization
///
/// # Returns
/// Optimized belief state
pub fn minimize_free_energy(
    observations: &[String],
    max_iterations: usize,
) -> Result<BeliefState> {
    // TODO: Implement variational inference (e.g., mean-field, gradient descent)
    let states = vec!["state_0".to_string(), "state_1".to_string()];
    Ok(BeliefState::uniform(states))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_belief_state_uniform() {
        let states = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let belief = BeliefState::uniform(states);

        assert_eq!(belief.probabilities.len(), 3);
        for p in belief.probabilities {
            assert!((p - 1.0 / 3.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_belief_entropy() {
        let states = vec!["A".to_string(), "B".to_string()];
        let belief = BeliefState::uniform(states);

        let entropy = belief.entropy();
        let expected = -(0.5_f64 * 0.5_f64.ln() + 0.5_f64 * 0.5_f64.ln());
        assert!((entropy - expected).abs() < 1e-10);
    }

    #[test]
    fn test_infer_state() {
        let prior = BeliefState::uniform(vec!["A".to_string(), "B".to_string()]);

        let mut likelihood = HashMap::new();
        likelihood.insert("A".to_string(), 0.8);
        likelihood.insert("B".to_string(), 0.2);

        let posterior = infer_state(&prior, "obs", &likelihood).unwrap();

        // A should have higher probability
        assert!(posterior.probabilities[0] > posterior.probabilities[1]);
    }
}
