//! Active Inference Agent implementing the Free Energy Principle
//!
//! This module implements agents that minimize Variational Free Energy (VFE)
//! by maintaining a generative model of their environment and acting to
//! minimize prediction error (surprise).
//!
//! # pbRTCA v4.1 Integration
//!
//! Enhanced with Hoffman's Conscious Agent Theory (CAT):
//! - Qualia Kernel: Q = P ∘ D ∘ A (self-referential experiencing)
//! - Thermodynamic constraints (Landauer bound)
//! - Markovian kernel dynamics
//! - Temporal consciousness (retention/primal/protention)
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
use thiserror::Error;
use tracing::debug;

pub mod gpu_accelerated;
pub mod hyperphysics_integration;
pub mod markov_kernel;
pub mod pbit_substrate;
pub mod qualia;
pub mod temporal;
pub mod thermodynamics;

pub use gpu_accelerated::*;
pub use hyperphysics_integration::*;
pub use markov_kernel::*;
pub use pbit_substrate::*;
pub use qualia::*;
pub use temporal::*;
pub use thermodynamics::*;

/// Errors in conscious processing
#[derive(Debug, Error)]
pub enum ConsciousnessError {
    #[error("Thermodynamic violation: {0}")]
    ThermodynamicViolation(String),

    #[error("Landauer bound violated: provided {provided} J, required {required} J")]
    LandauerBoundViolated { provided: f64, required: f64 },

    #[error("Markovian property violated: row sum = {sum}, expected 1.0")]
    MarkovianViolation { sum: f64 },

    #[error("Numerical instability: {0}")]
    NumericalInstability(String),

    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
}

/// Result type for consciousness operations
pub type ConsciousnessResult<T> = Result<T, ConsciousnessError>;

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
            preferences: na::DVector::from_element(state_dim, 1.0 / state_dim as f64),
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
        let state_likelihood = self.likelihood.transpose() * observation;
        let posterior = state_likelihood.component_mul(prior);

        // Normalize
        let sum = posterior.sum();
        if sum > 1e-10 {
            posterior / sum
        } else {
            prior.clone()
        }
    }
}

/// Active Inference Agent with pbRTCA consciousness extensions
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
    /// Thermodynamic state
    #[serde(skip)]
    pub thermodynamics: Option<ThermodynamicState>,
    /// Temporal consciousness state
    #[serde(skip)]
    pub temporal: Option<TemporalConsciousness>,
}

impl ActiveInferenceAgent {
    /// Create a new Active Inference agent
    pub fn new(model: GenerativeModel, initial_belief: na::DVector<f64>) -> Self {
        Self {
            model,
            belief: initial_belief,
            actions: Vec::new(),
            precision: 1.0,
            thermodynamics: Some(ThermodynamicState::default()),
            temporal: Some(TemporalConsciousness::new(16)),
        }
    }

    /// Create agent with full pbRTCA consciousness features
    pub fn with_consciousness(
        model: GenerativeModel,
        initial_belief: na::DVector<f64>,
        temperature: f64,
        energy_budget: f64,
    ) -> Self {
        let mut agent = Self::new(model, initial_belief);
        agent.thermodynamics = Some(ThermodynamicState::new(temperature, energy_budget));
        agent.temporal = Some(TemporalConsciousness::new(16));
        agent
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
        // Expected prediction error + Expected information gain

        // Simulate action effect on belief
        let next_belief = &self.belief + action * 0.1;

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

    /// Update agent state given observation with thermodynamic tracking
    pub fn step(&mut self, observation: &na::DVector<f64>) -> ConsciousnessResult<f64> {
        debug!("Active Inference step with observation dim: {}", observation.len());

        // Check thermodynamic feasibility
        if let Some(ref mut thermo) = self.thermodynamics {
            thermo.verify_landauer_bound()?;
        }

        // Update temporal consciousness (retention → primal → protention)
        if let Some(ref mut temporal) = self.temporal {
            temporal.update(&self.belief);
        }

        // 1. Update belief (perception)
        self.belief = self.model.update_belief(&self.belief, observation);

        // 2. Compute current free energy
        let free_energy = self.compute_free_energy(observation);
        debug!("Current Free Energy: {:.4}", free_energy);

        // 3. Record thermodynamic cost
        if let Some(ref mut thermo) = self.thermodynamics {
            thermo.record_processing_cost(free_energy)?;
        }

        Ok(free_energy)
    }

    /// Conscious processing cycle with full pbRTCA features
    ///
    /// Implements the Qualia Kernel cycle: X → X
    /// Q = P ∘ D ∘ A (Perception → Decision → Action)
    pub fn conscious_cycle(&mut self, observation: &na::DVector<f64>) -> ConsciousnessResult<ConsciousExperience> {
        // 1. Verify thermodynamic feasibility (Landauer bound)
        if let Some(ref mut thermo) = self.thermodynamics {
            thermo.verify_landauer_bound()?;
        }

        // 2. Update temporal consciousness structure
        let temporal_state = if let Some(ref mut temporal) = self.temporal {
            temporal.update(&self.belief);
            Some(temporal.get_temporal_thickness())
        } else {
            None
        };

        // 3. Perception: Update belief given observation
        self.belief = self.model.update_belief(&self.belief, observation);

        // 4. Compute free energy
        let free_energy = self.compute_free_energy(observation);

        // 5. Decision: Select action via EFE minimization
        let selected_action = self.select_action();

        // 6. Record thermodynamic cost
        if let Some(ref mut thermo) = self.thermodynamics {
            thermo.record_processing_cost(free_energy)?;
        }

        Ok(ConsciousExperience {
            belief: self.belief.clone(),
            free_energy,
            selected_action,
            temporal_thickness: temporal_state.unwrap_or(0.0),
            entropy_rate: self.compute_entropy_rate(),
        })
    }

    /// Compute entropy rate of the belief dynamics
    ///
    /// H(Q) = -Σᵢ μᵢ Σⱼ Qᵢⱼ log(Qᵢⱼ)
    ///
    /// Hoffman's interpretation: Entropy rate → particle mass
    /// H(Q) = 0 (periodic) → massless
    /// H(Q) > 0 → massive
    pub fn compute_entropy_rate(&self) -> f64 {
        // Use belief as stationary distribution approximation
        let mu = &self.belief;
        let q_matrix = &self.model.transition;

        let mut h = 0.0;
        for i in 0..q_matrix.nrows() {
            for j in 0..q_matrix.ncols() {
                let q_ij = q_matrix[(i, j)];
                if q_ij > 1e-12 {
                    h -= mu[i] * q_ij * q_ij.ln();
                }
            }
        }

        h
    }
}

/// Result of a conscious processing cycle
#[derive(Debug, Clone)]
pub struct ConsciousExperience {
    /// Current belief distribution
    pub belief: na::DVector<f64>,
    /// Variational free energy
    pub free_energy: f64,
    /// Selected action (if any)
    pub selected_action: Option<na::DVector<f64>>,
    /// Temporal thickness (phenomenological depth)
    pub temporal_thickness: f64,
    /// Entropy rate (Hoffman's mass proposal)
    pub entropy_rate: f64,
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
        agent.actions.push(na::DVector::from_vec(vec![1.0, 0.0, 0.0]));
        agent.actions.push(na::DVector::from_vec(vec![0.0, 1.0, 0.0]));

        // Test observation
        let obs = na::DVector::from_vec(vec![0.8, 0.2]);
        let result = agent.step(&obs);
        assert!(result.is_ok(), "Step failed: {:?}", result.err());

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

    #[test]
    fn test_conscious_cycle() {
        let model = GenerativeModel::new(3, 2);
        let initial_belief = na::DVector::from_vec(vec![0.33, 0.33, 0.34]);
        let mut agent = ActiveInferenceAgent::with_consciousness(
            model,
            initial_belief,
            300.0, // Room temperature (K)
            1e-12, // 1 pJ energy budget (sufficient for conscious cycles)
        );

        agent.actions.push(na::DVector::from_vec(vec![1.0, 0.0, 0.0]));

        let obs = na::DVector::from_vec(vec![0.8, 0.2]);
        let experience = agent.conscious_cycle(&obs);

        assert!(experience.is_ok());
        let exp = experience.unwrap();
        assert!(exp.free_energy.is_finite());
        assert!(exp.entropy_rate.is_finite());
    }

    #[test]
    fn test_entropy_rate() {
        let model = GenerativeModel::new(2, 2);
        let belief = na::DVector::from_vec(vec![0.5, 0.5]);
        let agent = ActiveInferenceAgent::new(model, belief);

        let h = agent.compute_entropy_rate();
        assert!(h >= 0.0);
        assert!(h.is_finite());
    }
}
