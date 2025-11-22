//! Active Inference Agent implementing the Free Energy Principle
//!
//! This module implements agents that minimize Variational Free Energy (VFE)
//! by maintaining a generative model of their environment and acting to
//! minimize prediction error (surprise).
//!
//! # Mathematical Foundation
//!
//! The Free Energy Principle states that agents minimize:
//! F = E_q[ln q(φ) - ln p(φ, y)]
//!
//! where:
//! - q(φ) is the agent's belief distribution over hidden states
//! - p(φ, y) is the joint distribution of hidden states and observations
//! - F is the Variational Free Energy (upper bound on surprise)

use nalgebra as na;
use rand::Rng;
use serde::{Deserialize, Serialize};
use tracing::debug;

/// Generative model for the agent (POMDP)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerativeModel {
    /// State transition matrix A (dynamics)
    pub transition: na::DMatrix<f64>,
    /// Observation likelihood matrix B (sensor model)
    pub likelihood: na::DMatrix<f64>,
    /// Prior preferences C (goal states)
    pub preferences: na::DVector<f64>,
    /// State dimensionality
    pub state_dim: usize,
    /// Observation dimensionality
    pub obs_dim: usize,
}

impl GenerativeModel {
    /// Create a new generative model
    pub fn new(state_dim: usize, obs_dim: usize) -> Self {
        Self {
            transition: na::DMatrix::identity(state_dim, state_dim),
            likelihood: na::DMatrix::identity(obs_dim, state_dim),
            preferences: na::DVector::zeros(state_dim),
            state_dim,
            obs_dim,
        }
    }

    /// Predict next observation given current belief
    pub fn predict_observation(&self, belief: &na::DVector<f64>) -> na::DVector<f64> {
        &self.likelihood * belief
    }

    /// Update belief given observation (Bayesian inference)
    pub fn update_belief(
        &self,
        prior: &na::DVector<f64>,
        observation: &na::DVector<f64>,
    ) -> na::DVector<f64> {
        // Bayes rule: P(s|o) ∝ P(o|s) * P(s)
        // We interpret 'observation' as evidence for each observation channel.
        // We compute the likelihood of each state given this evidence:
        // L(s) = B^T * observation
        let state_likelihood = self.likelihood.transpose() * observation;

        // Posterior ∝ state_likelihood * prior
        let posterior = state_likelihood.component_mul(prior);

        // Normalize
        let sum = posterior.sum();
        if sum > 1e-10 {
            posterior / sum
        } else {
            // If numerical issues or zero likelihood, fall back to prior
            prior.clone()
        }
    }
}

/// Active Inference Agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveInferenceAgent {
    /// Agent's generative model
    pub model: GenerativeModel,
    /// Current belief state (posterior distribution)
    pub belief: na::DVector<f64>,
    /// Action repertoire
    pub actions: Vec<na::DVector<f64>>,
    /// Precision (inverse temperature) for action selection
    pub precision: f64,
}

impl ActiveInferenceAgent {
    /// Create a new Active Inference agent
    pub fn new(model: GenerativeModel, initial_belief: na::DVector<f64>) -> Self {
        Self {
            model,
            belief: initial_belief,
            actions: Vec::new(),
            precision: 1.0,
        }
    }

    /// Compute Variational Free Energy
    pub fn compute_free_energy(&self, observation: &na::DVector<f64>) -> f64 {
        // F = E_q[ln q(s)] - E_q[ln p(s, o)]
        // Simplified: F ≈ -ln p(o|s) + KL[q(s)||p(s)]

        let predicted_obs = self.model.predict_observation(&self.belief);

        // Prediction error (negative log-likelihood)
        let prediction_error = (observation - predicted_obs).norm_squared();

        // KL divergence from prior (regularization)
        let kl_divergence = self
            .belief
            .iter()
            .zip(self.model.preferences.iter())
            .map(|(q, p)| {
                if *q > 1e-10 && *p > 1e-10 {
                    q * (q.ln() - p.ln())
                } else {
                    0.0
                }
            })
            .sum::<f64>();

        prediction_error + kl_divergence
    }

    /// Compute Expected Free Energy for an action (planning)
    pub fn expected_free_energy(&self, action: &na::DVector<f64>) -> f64 {
        // G = E[F(s', o')] where s' = f(s, a)
        // Simplified: Expected prediction error + Expected information gain

        // Simulate action effect on belief
        let next_belief = &self.belief + action * 0.1; // Simplified dynamics

        // Expected observation
        let expected_obs = self.model.predict_observation(&next_belief);

        // Epistemic value (information gain)
        let epistemic = -expected_obs
            .iter()
            .map(|p| if *p > 1e-10 { p * p.ln() } else { 0.0 })
            .sum::<f64>();

        // Pragmatic value (preference satisfaction)
        let pragmatic = -next_belief.dot(&self.model.preferences);

        epistemic + pragmatic
    }

    /// Select action by minimizing Expected Free Energy
    pub fn select_action(&self) -> Option<na::DVector<f64>> {
        if self.actions.is_empty() {
            return None;
        }

        // Compute EFE for all actions
        let efe_values: Vec<f64> = self
            .actions
            .iter()
            .map(|a| self.expected_free_energy(a))
            .collect();

        // Softmax action selection (precision-weighted)
        let max_efe = efe_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_values: Vec<f64> = efe_values
            .iter()
            .map(|efe| (-(efe - max_efe) * self.precision).exp())
            .collect();

        let sum_exp: f64 = exp_values.iter().sum();
        let probabilities: Vec<f64> = exp_values.iter().map(|e| e / sum_exp).collect();

        // Sample action according to probabilities
        let mut rng = rand::thread_rng();
        let sample: f64 = rng.gen();
        let mut cumulative = 0.0;

        for (i, &prob) in probabilities.iter().enumerate() {
            cumulative += prob;
            if sample <= cumulative {
                return Some(self.actions[i].clone());
            }
        }

        Some(self.actions[0].clone())
    }

    /// Update agent state given observation
    pub fn step(&mut self, observation: &na::DVector<f64>) {
        debug!("Active Inference step with observation: {:?}", observation);

        // 1. Update belief (perception)
        self.belief = self.model.update_belief(&self.belief, observation);

        // 2. Compute current free energy
        let free_energy = self.compute_free_energy(observation);
        debug!("Current Free Energy: {:.4}", free_energy);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generative_model() {
        let model = GenerativeModel::new(3, 2);
        assert_eq!(model.state_dim, 3);
        assert_eq!(model.obs_dim, 2);
    }

    #[test]
    fn test_active_inference_agent() {
        let model = GenerativeModel::new(3, 2);
        let initial_belief = na::DVector::from_vec(vec![0.33, 0.33, 0.34]);
        let mut agent = ActiveInferenceAgent::new(model, initial_belief);

        // Add some actions
        agent
            .actions
            .push(na::DVector::from_vec(vec![1.0, 0.0, 0.0]));
        agent
            .actions
            .push(na::DVector::from_vec(vec![0.0, 1.0, 0.0]));

        // Test observation
        let obs = na::DVector::from_vec(vec![0.8, 0.2]);
        agent.step(&obs);

        // Belief should have updated
        assert!(agent.belief.sum() > 0.99 && agent.belief.sum() < 1.01);
    }

    #[test]
    fn test_free_energy_computation() {
        let model = GenerativeModel::new(2, 2);
        let belief = na::DVector::from_vec(vec![0.5, 0.5]);
        let agent = ActiveInferenceAgent::new(model, belief);

        let obs = na::DVector::from_vec(vec![1.0, 0.0]);
        let fe = agent.compute_free_energy(&obs);

        assert!(fe.is_finite());
        assert!(fe >= 0.0);
    }
}
