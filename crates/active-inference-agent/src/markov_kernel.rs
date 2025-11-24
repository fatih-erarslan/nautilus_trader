//! Markovian Kernel implementation for conscious agent dynamics
//!
//! In Hoffman's Conscious Agent Theory (CAT), conscious agents are defined
//! by Markovian kernels - probability measure-preserving transformations
//! that map experiences to actions and vice versa.
//!
//! # Mathematical Foundation
//!
//! A Markovian kernel K: X → X is a measurable function where:
//! - K(x, A) gives probability of transitioning from x to set A
//! - K preserves the probability measure: ∫ K(x, ·) dμ(x) = μ(·)
//! - Stochastic matrix rows sum to 1
//!
//! # pbRTCA Integration
//!
//! The Markovian kernel layer provides the mathematical substrate for:
//! - Perception kernel P: World → Experience
//! - Decision kernel D: Experience → Action choice
//! - Action kernel A: Action choice → World effect

use crate::ConsciousnessError;
use nalgebra as na;
use serde::{Deserialize, Serialize};

/// A Markovian kernel (stochastic matrix) for state transitions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarkovianKernel {
    /// The stochastic transition matrix K[i,j] = P(j|i)
    pub kernel: na::DMatrix<f64>,
    /// Stationary distribution (if computed)
    pub stationary: Option<na::DVector<f64>>,
    /// Kernel name/label for debugging
    pub name: String,
    /// Dimensionality
    pub dim: usize,
}

impl MarkovianKernel {
    /// Create a new Markovian kernel from a stochastic matrix
    ///
    /// Validates that all rows sum to 1 (within tolerance)
    pub fn new(matrix: na::DMatrix<f64>, name: impl Into<String>) -> Result<Self, ConsciousnessError> {
        let dim = matrix.nrows();

        if matrix.nrows() != matrix.ncols() {
            return Err(ConsciousnessError::DimensionMismatch {
                expected: matrix.nrows(),
                actual: matrix.ncols(),
            });
        }

        // Verify stochasticity (rows sum to 1)
        for i in 0..dim {
            let row_sum: f64 = matrix.row(i).iter().sum();
            if (row_sum - 1.0).abs() > 1e-6 {
                return Err(ConsciousnessError::MarkovianViolation { sum: row_sum });
            }
        }

        Ok(Self {
            kernel: matrix,
            stationary: None,
            name: name.into(),
            dim,
        })
    }

    /// Create identity kernel (no state change)
    pub fn identity(dim: usize) -> Self {
        Self {
            kernel: na::DMatrix::identity(dim, dim),
            stationary: Some(na::DVector::from_element(dim, 1.0 / dim as f64)),
            name: "identity".to_string(),
            dim,
        }
    }

    /// Create uniform kernel (all transitions equally likely)
    pub fn uniform(dim: usize) -> Self {
        let val = 1.0 / dim as f64;
        Self {
            kernel: na::DMatrix::from_element(dim, dim, val),
            stationary: Some(na::DVector::from_element(dim, val)),
            name: "uniform".to_string(),
            dim,
        }
    }

    /// Create from transition probabilities with noise
    pub fn with_noise(base: na::DMatrix<f64>, noise_level: f64, name: impl Into<String>) -> Result<Self, ConsciousnessError> {
        let dim = base.nrows();
        let uniform_component = 1.0 / dim as f64;

        // Mix base with uniform: (1-ε)K + ε*U
        let mut mixed = base * (1.0 - noise_level);
        for i in 0..dim {
            for j in 0..dim {
                mixed[(i, j)] += noise_level * uniform_component;
            }
        }

        Self::new(mixed, name)
    }

    /// Apply kernel to a probability distribution
    ///
    /// μ' = μ K (left multiplication, distribution as row vector)
    pub fn apply(&self, distribution: &na::DVector<f64>) -> na::DVector<f64> {
        // K^T * μ for column vector representation
        &self.kernel.transpose() * distribution
    }

    /// Apply kernel n times (power iteration)
    pub fn apply_n(&self, distribution: &na::DVector<f64>, n: usize) -> na::DVector<f64> {
        let mut result = distribution.clone();
        for _ in 0..n {
            result = self.apply(&result);
        }
        result
    }

    /// Compute stationary distribution using power iteration
    ///
    /// π such that π K = π
    pub fn compute_stationary(&mut self, max_iter: usize, tolerance: f64) -> na::DVector<f64> {
        let mut pi = na::DVector::from_element(self.dim, 1.0 / self.dim as f64);

        for _ in 0..max_iter {
            let next = self.apply(&pi);
            let diff = (&next - &pi).norm();
            pi = next;

            if diff < tolerance {
                break;
            }
        }

        // Normalize
        let sum = pi.sum();
        if sum > 1e-10 {
            pi /= sum;
        }

        self.stationary = Some(pi.clone());
        pi
    }

    /// Get or compute stationary distribution
    pub fn stationary_distribution(&mut self) -> na::DVector<f64> {
        if let Some(ref stat) = self.stationary {
            stat.clone()
        } else {
            self.compute_stationary(1000, 1e-10)
        }
    }

    /// Compute entropy rate of the kernel
    ///
    /// H(K) = -Σᵢ πᵢ Σⱼ Kᵢⱼ log(Kᵢⱼ)
    ///
    /// This is Hoffman's connection to particle mass
    pub fn entropy_rate(&mut self) -> f64 {
        let pi = self.stationary_distribution();

        let mut h = 0.0;
        for i in 0..self.dim {
            for j in 0..self.dim {
                let k_ij = self.kernel[(i, j)];
                if k_ij > 1e-12 {
                    h -= pi[i] * k_ij * k_ij.ln();
                }
            }
        }

        h
    }

    /// Check if kernel is doubly stochastic
    ///
    /// Doubly stochastic: both rows AND columns sum to 1
    /// These have uniform stationary distribution
    pub fn is_doubly_stochastic(&self) -> bool {
        // Check columns
        for j in 0..self.dim {
            let col_sum: f64 = self.kernel.column(j).iter().sum();
            if (col_sum - 1.0).abs() > 1e-6 {
                return false;
            }
        }
        true
    }

    /// Compose two kernels: K1 ∘ K2
    ///
    /// The composition represents sequential application
    pub fn compose(&self, other: &MarkovianKernel) -> Result<MarkovianKernel, ConsciousnessError> {
        if self.dim != other.dim {
            return Err(ConsciousnessError::DimensionMismatch {
                expected: self.dim,
                actual: other.dim,
            });
        }

        let composed = &self.kernel * &other.kernel;
        MarkovianKernel::new(composed, format!("{}∘{}", self.name, other.name))
    }

    /// Compute mixing time estimate (spectral gap based)
    ///
    /// τ_mix ≈ 1 / (1 - λ₂) where λ₂ is second largest eigenvalue
    pub fn mixing_time_estimate(&self) -> f64 {
        // Simple power method to estimate spectral gap
        // For production, use proper eigenvalue computation

        let mut v = na::DVector::from_fn(self.dim, |i, _| {
            if i == 0 { 1.0 } else { -1.0 / (self.dim - 1) as f64 }
        });

        // Orthogonalize to stationary
        let pi = self.stationary.as_ref()
            .cloned()
            .unwrap_or_else(|| na::DVector::from_element(self.dim, 1.0 / self.dim as f64));

        let proj = v.dot(&pi);
        v -= &pi * proj;

        // Power iteration for second eigenvalue
        for _ in 0..100 {
            v = self.apply(&v);
            let norm = v.norm();
            if norm > 1e-10 {
                v /= norm;
            }
            // Re-orthogonalize
            let proj = v.dot(&pi);
            v -= &pi * proj;
        }

        let lambda2 = self.apply(&v).norm() / v.norm().max(1e-10);

        if lambda2 < 0.999 {
            1.0 / (1.0 - lambda2)
        } else {
            1000.0 // Very slow mixing
        }
    }
}

/// Perception kernel: maps world states to experiences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerceptionKernel {
    /// The underlying Markovian kernel
    pub kernel: MarkovianKernel,
    /// Sensory noise level
    pub noise_level: f64,
}

impl PerceptionKernel {
    /// Create perception kernel from likelihood matrix
    pub fn from_likelihood(likelihood: na::DMatrix<f64>, noise: f64) -> Result<Self, ConsciousnessError> {
        // Normalize rows to make it stochastic
        let mut normalized = likelihood.clone();
        for i in 0..normalized.nrows() {
            let row_sum: f64 = normalized.row(i).iter().sum();
            if row_sum > 1e-10 {
                for j in 0..normalized.ncols() {
                    normalized[(i, j)] /= row_sum;
                }
            } else {
                // Uniform if degenerate
                let val = 1.0 / normalized.ncols() as f64;
                for j in 0..normalized.ncols() {
                    normalized[(i, j)] = val;
                }
            }
        }

        let kernel = MarkovianKernel::with_noise(normalized, noise, "perception")?;

        Ok(Self {
            kernel,
            noise_level: noise,
        })
    }

    /// Apply perception to world state
    pub fn perceive(&self, world_state: &na::DVector<f64>) -> na::DVector<f64> {
        self.kernel.apply(world_state)
    }
}

/// Decision kernel: maps experiences to action probabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionKernel {
    /// The underlying Markovian kernel
    pub kernel: MarkovianKernel,
    /// Precision (inverse temperature) for softmax
    pub precision: f64,
}

impl DecisionKernel {
    /// Create decision kernel from value function
    ///
    /// Uses softmax: P(a|s) ∝ exp(-β * V(s,a))
    pub fn from_values(values: na::DMatrix<f64>, precision: f64) -> Result<Self, ConsciousnessError> {
        let n_states = values.nrows();
        let n_actions = values.ncols();

        let mut probs = na::DMatrix::zeros(n_states, n_actions);

        for i in 0..n_states {
            // Softmax over actions
            let max_val = values.row(i).iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let mut sum = 0.0;

            for j in 0..n_actions {
                let exp_val = (-(values[(i, j)] - max_val) * precision).exp();
                probs[(i, j)] = exp_val;
                sum += exp_val;
            }

            // Normalize
            for j in 0..n_actions {
                probs[(i, j)] /= sum;
            }
        }

        let kernel = MarkovianKernel::new(probs, "decision")?;

        Ok(Self { kernel, precision })
    }

    /// Decide action given experience
    pub fn decide(&self, experience: &na::DVector<f64>) -> na::DVector<f64> {
        self.kernel.apply(experience)
    }
}

/// Action kernel: maps action choices to world effects
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionKernel {
    /// The underlying Markovian kernel
    pub kernel: MarkovianKernel,
    /// Effectiveness (how much action changes world)
    pub effectiveness: f64,
}

impl ActionKernel {
    /// Create action kernel from dynamics model
    pub fn from_dynamics(dynamics: na::DMatrix<f64>, effectiveness: f64) -> Result<Self, ConsciousnessError> {
        let kernel = MarkovianKernel::new(dynamics, "action")?;

        Ok(Self {
            kernel,
            effectiveness,
        })
    }

    /// Apply action to world
    pub fn act(&self, action_choice: &na::DVector<f64>) -> na::DVector<f64> {
        self.kernel.apply(action_choice)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_markov_kernel_validation() {
        // Valid stochastic matrix
        let valid = na::DMatrix::from_row_slice(2, 2, &[
            0.7, 0.3,
            0.4, 0.6,
        ]);
        assert!(MarkovianKernel::new(valid, "test").is_ok());

        // Invalid (rows don't sum to 1)
        let invalid = na::DMatrix::from_row_slice(2, 2, &[
            0.5, 0.3,
            0.4, 0.6,
        ]);
        assert!(MarkovianKernel::new(invalid, "test").is_err());
    }

    #[test]
    fn test_stationary_distribution() {
        let matrix = na::DMatrix::from_row_slice(2, 2, &[
            0.9, 0.1,
            0.2, 0.8,
        ]);
        let mut kernel = MarkovianKernel::new(matrix, "test").unwrap();

        let stat = kernel.compute_stationary(1000, 1e-10);

        // Verify it's stationary
        let applied = kernel.apply(&stat);
        assert!((applied - &stat).norm() < 1e-6);
    }

    #[test]
    fn test_entropy_rate() {
        let matrix = na::DMatrix::from_row_slice(2, 2, &[
            0.5, 0.5,
            0.5, 0.5,
        ]);
        let mut kernel = MarkovianKernel::new(matrix, "uniform").unwrap();

        let h = kernel.entropy_rate();

        // Maximum entropy for 2 states = ln(2) ≈ 0.693
        assert!(h > 0.0);
        assert!(h < 1.0);
    }

    #[test]
    fn test_kernel_composition() {
        let k1 = MarkovianKernel::identity(3);
        let k2 = MarkovianKernel::uniform(3);

        let composed = k1.compose(&k2).unwrap();

        // Identity composed with anything = that thing
        assert!((composed.kernel - k2.kernel).norm() < 1e-10);
    }

    #[test]
    fn test_doubly_stochastic() {
        let uniform = MarkovianKernel::uniform(3);
        assert!(uniform.is_doubly_stochastic());

        let non_doubly = na::DMatrix::from_row_slice(2, 2, &[
            0.9, 0.1,
            0.2, 0.8,
        ]);
        let kernel = MarkovianKernel::new(non_doubly, "test").unwrap();
        assert!(!kernel.is_doubly_stochastic());
    }
}
