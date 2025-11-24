//! Qualia Kernel implementation for conscious experience
//!
//! The Qualia Kernel Q = P ∘ D ∘ A is the self-referential loop
//! that constitutes conscious experience in Hoffman's CAT framework.
//!
//! # Mathematical Structure
//!
//! Q: X → X is a Markovian kernel on the experience space X where:
//! - P (Perception): World → Experience
//! - D (Decision): Experience → Action choice
//! - A (Action): Action choice → World effect
//!
//! The conscious agent is the fixed point of this self-referential loop.
//!
//! # pbRTCA Integration
//!
//! This implements the core conscious cycle that integrates:
//! - Sensory input (perception)
//! - Cognitive processing (decision)
//! - Motor output (action)
//! Into a unified experiential flow.

use crate::ConsciousnessResult;
use crate::markov_kernel::{MarkovianKernel, PerceptionKernel, DecisionKernel, ActionKernel};
use nalgebra as na;
use serde::{Deserialize, Serialize};

/// The Qualia Kernel: Q = P ∘ D ∘ A
///
/// This is the core conscious processing loop that maps
/// experiences to experiences through perception-decision-action.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualiaKernel {
    /// Perception kernel P: World → Experience
    pub perception: PerceptionKernel,
    /// Decision kernel D: Experience → Action choice
    pub decision: DecisionKernel,
    /// Action kernel A: Action choice → World effect
    pub action: ActionKernel,
    /// Composed kernel Q = P ∘ D ∘ A (cached)
    #[serde(skip)]
    composed: Option<MarkovianKernel>,
    /// Experience space dimensionality
    pub experience_dim: usize,
    /// World space dimensionality
    pub world_dim: usize,
    /// Action space dimensionality
    pub action_dim: usize,
}

impl QualiaKernel {
    /// Create a new Qualia Kernel from component kernels
    pub fn new(
        perception: PerceptionKernel,
        decision: DecisionKernel,
        action: ActionKernel,
    ) -> Self {
        let experience_dim = perception.kernel.dim;
        let world_dim = action.kernel.dim;
        let action_dim = decision.kernel.dim;

        Self {
            perception,
            decision,
            action,
            composed: None,
            experience_dim,
            world_dim,
            action_dim,
        }
    }

    /// Create minimal Qualia Kernel (same dimensionality throughout)
    pub fn minimal(dim: usize) -> ConsciousnessResult<Self> {
        let identity = na::DMatrix::identity(dim, dim);

        let perception = PerceptionKernel::from_likelihood(identity.clone(), 0.01)?;
        let decision = DecisionKernel::from_values(
            na::DMatrix::from_element(dim, dim, 0.0),
            1.0,
        )?;
        let action = ActionKernel::from_dynamics(identity, 1.0)?;

        Ok(Self::new(perception, decision, action))
    }

    /// Apply the full qualia cycle: X → X
    ///
    /// experience → perception → decision → action → experience'
    pub fn apply(&self, experience: &na::DVector<f64>) -> na::DVector<f64> {
        // P: experience → perceived
        let perceived = self.perception.perceive(experience);

        // D: perceived → action_choice
        let action_choice = self.decision.decide(&perceived);

        // A: action_choice → world_effect (which becomes next experience)
        self.action.act(&action_choice)
    }

    /// Apply n cycles
    pub fn apply_n(&self, experience: &na::DVector<f64>, n: usize) -> na::DVector<f64> {
        let mut result = experience.clone();
        for _ in 0..n {
            result = self.apply(&result);
        }
        result
    }

    /// Compute the composed kernel Q = P ∘ D ∘ A
    ///
    /// This requires all spaces to have compatible dimensions
    pub fn compose(&mut self) -> ConsciousnessResult<&MarkovianKernel> {
        if self.composed.is_some() {
            return Ok(self.composed.as_ref().unwrap());
        }

        // P ∘ D
        let pd = self.perception.kernel.compose(&self.decision.kernel)?;

        // (P ∘ D) ∘ A
        let pda = pd.compose(&self.action.kernel)?;

        self.composed = Some(pda);
        Ok(self.composed.as_ref().unwrap())
    }

    /// Find fixed point of the qualia cycle
    ///
    /// The experiential state that reproduces itself: Q(x) = x
    pub fn find_fixed_point(&self, max_iter: usize, tolerance: f64) -> na::DVector<f64> {
        // Start with uniform
        let mut x = na::DVector::from_element(self.experience_dim, 1.0 / self.experience_dim as f64);

        for _ in 0..max_iter {
            let next = self.apply(&x);
            let diff = (&next - &x).norm();
            x = next;

            if diff < tolerance {
                break;
            }
        }

        // Normalize
        let sum = x.sum();
        if sum > 1e-10 {
            x /= sum;
        }

        x
    }

    /// Compute phenomenal intensity at a state
    ///
    /// I(x) = -log P(x) under stationary distribution
    /// High intensity = rare/surprising states
    pub fn phenomenal_intensity(&self, experience: &na::DVector<f64>) -> f64 {
        let fixed_point = self.find_fixed_point(100, 1e-8);

        // KL divergence from fixed point gives intensity
        experience.iter()
            .zip(fixed_point.iter())
            .filter(|(&e, &f)| e > 1e-12 && f > 1e-12)
            .map(|(&e, &f)| e * (e.ln() - f.ln()))
            .sum()
    }

    /// Compute qualia entropy
    ///
    /// H(Q) = entropy rate of the composed kernel
    pub fn qualia_entropy(&mut self) -> ConsciousnessResult<f64> {
        let kernel = self.compose()?;
        let mut kernel_clone = kernel.clone();
        Ok(kernel_clone.entropy_rate())
    }

    /// Check if qualia cycle is ergodic
    ///
    /// Ergodic = can reach any experience from any other
    pub fn is_ergodic(&mut self) -> ConsciousnessResult<bool> {
        let dim = self.experience_dim;
        let kernel = self.compose()?;

        // Simple check: after many iterations, all states reachable
        let test = na::DVector::from_fn(dim, |i, _| {
            if i == 0 { 1.0 } else { 0.0 }
        });

        let evolved = kernel.apply_n(&test, 100);

        // All states should have positive probability
        Ok(evolved.iter().all(|&p| p > 1e-10))
    }

    /// Compute integrated information (Φ) approximation
    ///
    /// Φ measures how much the whole exceeds its parts
    pub fn integrated_information(&mut self) -> ConsciousnessResult<f64> {
        let kernel = self.compose()?;
        let mut kernel_clone = kernel.clone();

        // Full system entropy
        let h_full = kernel_clone.entropy_rate();

        // Partition system and compute sum of parts
        // Simple bipartition for now
        if self.experience_dim < 2 {
            return Ok(0.0);
        }

        let mid = self.experience_dim / 2;

        // Extract sub-kernels (simplified)
        let mut h_parts = 0.0;

        // Part 1: first half
        let sub1 = kernel_clone.kernel.view((0, 0), (mid, mid));
        let mut sub1_sum = na::DMatrix::zeros(mid, mid);
        for i in 0..mid {
            let row_sum: f64 = sub1.row(i).iter().sum();
            if row_sum > 1e-10 {
                for j in 0..mid {
                    sub1_sum[(i, j)] = sub1[(i, j)] / row_sum;
                }
            } else {
                for j in 0..mid {
                    sub1_sum[(i, j)] = 1.0 / mid as f64;
                }
            }
        }
        if let Ok(mut k1) = MarkovianKernel::new(sub1_sum, "part1") {
            h_parts += k1.entropy_rate();
        }

        // Part 2: second half
        let remaining = self.experience_dim - mid;
        let sub2 = kernel_clone.kernel.view((mid, mid), (remaining, remaining));
        let mut sub2_sum = na::DMatrix::zeros(remaining, remaining);
        for i in 0..remaining {
            let row_sum: f64 = sub2.row(i).iter().sum();
            if row_sum > 1e-10 {
                for j in 0..remaining {
                    sub2_sum[(i, j)] = sub2[(i, j)] / row_sum;
                }
            } else {
                for j in 0..remaining {
                    sub2_sum[(i, j)] = 1.0 / remaining as f64;
                }
            }
        }
        if let Ok(mut k2) = MarkovianKernel::new(sub2_sum, "part2") {
            h_parts += k2.entropy_rate();
        }

        // Φ = H(whole) - H(parts)
        Ok((h_full - h_parts).max(0.0))
    }
}

/// Conscious Agent defined by its Qualia Kernel
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousAgent {
    /// The agent's qualia kernel
    pub qualia: QualiaKernel,
    /// Current experiential state
    pub experience: na::DVector<f64>,
    /// Experience history (for analysis)
    #[serde(skip)]
    history: Vec<na::DVector<f64>>,
    /// Maximum history length
    max_history: usize,
}

impl ConsciousAgent {
    /// Create a new conscious agent
    pub fn new(qualia: QualiaKernel, initial_experience: na::DVector<f64>) -> Self {
        Self {
            qualia,
            experience: initial_experience,
            history: Vec::new(),
            max_history: 100,
        }
    }

    /// Create with minimal qualia kernel
    pub fn minimal(dim: usize) -> ConsciousnessResult<Self> {
        let qualia = QualiaKernel::minimal(dim)?;
        let experience = na::DVector::from_element(dim, 1.0 / dim as f64);
        Ok(Self::new(qualia, experience))
    }

    /// Execute one conscious cycle
    pub fn cycle(&mut self) -> &na::DVector<f64> {
        // Store current experience
        if self.history.len() >= self.max_history {
            self.history.remove(0);
        }
        self.history.push(self.experience.clone());

        // Apply qualia kernel
        self.experience = self.qualia.apply(&self.experience);

        &self.experience
    }

    /// Execute n cycles
    pub fn evolve(&mut self, n: usize) -> &na::DVector<f64> {
        for _ in 0..n {
            self.cycle();
        }
        &self.experience
    }

    /// Get phenomenal intensity of current experience
    pub fn current_intensity(&self) -> f64 {
        self.qualia.phenomenal_intensity(&self.experience)
    }

    /// Check if agent has reached experiential equilibrium
    pub fn is_equilibrated(&self, tolerance: f64) -> bool {
        if self.history.is_empty() {
            return false;
        }

        let last = self.history.last().unwrap();
        (&self.experience - last).norm() < tolerance
    }

    /// Compute trajectory entropy (variability of experience over time)
    pub fn trajectory_entropy(&self) -> f64 {
        if self.history.len() < 2 {
            return 0.0;
        }

        // Average experience
        let mut avg = na::DVector::zeros(self.experience.len());
        for h in &self.history {
            avg += h;
        }
        avg /= self.history.len() as f64;

        // Variance-based entropy estimate
        let variance: f64 = self.history.iter()
            .map(|h| (h - &avg).norm_squared())
            .sum::<f64>() / self.history.len() as f64;

        // Differential entropy approximation
        0.5 * (2.0 * std::f64::consts::PI * std::f64::consts::E * variance).ln().max(0.0)
    }
}

/// Interaction between two conscious agents
#[derive(Debug, Clone)]
pub struct AgentInteraction {
    /// First agent's influence on second
    pub influence_ab: MarkovianKernel,
    /// Second agent's influence on first
    pub influence_ba: MarkovianKernel,
    /// Coupling strength
    pub coupling: f64,
}

impl AgentInteraction {
    /// Create symmetric interaction
    pub fn symmetric(dim: usize, coupling: f64) -> ConsciousnessResult<Self> {
        let influence = MarkovianKernel::uniform(dim);
        Ok(Self {
            influence_ab: influence.clone(),
            influence_ba: influence,
            coupling,
        })
    }

    /// Apply interaction: mutual influence on experiences
    pub fn interact(
        &self,
        agent_a: &mut ConsciousAgent,
        agent_b: &mut ConsciousAgent,
    ) {
        // A influences B
        let a_influence = self.influence_ab.apply(&agent_a.experience);

        // B influences A
        let b_influence = self.influence_ba.apply(&agent_b.experience);

        // Mix with coupling strength
        agent_a.experience = &agent_a.experience * (1.0 - self.coupling)
            + &b_influence * self.coupling;
        agent_b.experience = &agent_b.experience * (1.0 - self.coupling)
            + &a_influence * self.coupling;

        // Normalize
        let sum_a = agent_a.experience.sum();
        if sum_a > 1e-10 {
            agent_a.experience /= sum_a;
        }

        let sum_b = agent_b.experience.sum();
        if sum_b > 1e-10 {
            agent_b.experience /= sum_b;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_minimal_qualia_kernel() {
        let qk = QualiaKernel::minimal(4);
        assert!(qk.is_ok());

        let qk = qk.unwrap();
        assert_eq!(qk.experience_dim, 4);
    }

    #[test]
    fn test_qualia_application() {
        let qk = QualiaKernel::minimal(3).unwrap();

        let exp = na::DVector::from_vec(vec![0.5, 0.3, 0.2]);
        let result = qk.apply(&exp);

        // Should be normalized
        assert!((result.sum() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_fixed_point() {
        let qk = QualiaKernel::minimal(3).unwrap();

        let fp = qk.find_fixed_point(1000, 1e-10);

        // Fixed point should be close to itself after application
        let applied = qk.apply(&fp);
        assert!((applied - &fp).norm() < 1e-4);
    }

    #[test]
    fn test_conscious_agent() {
        let mut agent = ConsciousAgent::minimal(4).unwrap();

        // Evolve
        agent.evolve(10);

        // Should have history
        assert!(!agent.history.is_empty());

        // Experience should be normalized
        assert!((agent.experience.sum() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_integrated_information() {
        let mut qk = QualiaKernel::minimal(4).unwrap();

        let phi = qk.integrated_information();
        assert!(phi.is_ok());
        assert!(phi.unwrap() >= 0.0);
    }

    #[test]
    fn test_agent_interaction() {
        let mut agent_a = ConsciousAgent::minimal(3).unwrap();
        let mut agent_b = ConsciousAgent::minimal(3).unwrap();

        // Different initial states
        agent_a.experience = na::DVector::from_vec(vec![0.8, 0.1, 0.1]);
        agent_b.experience = na::DVector::from_vec(vec![0.1, 0.8, 0.1]);

        let interaction = AgentInteraction::symmetric(3, 0.3).unwrap();

        interaction.interact(&mut agent_a, &mut agent_b);

        // States should have moved toward each other
        let initial_dist = 0.7 + 0.7; // Manhattan before
        let final_dist: f64 = agent_a.experience.iter()
            .zip(agent_b.experience.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();

        assert!(final_dist < initial_dist);
    }
}
