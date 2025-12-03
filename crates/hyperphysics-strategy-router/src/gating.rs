//! Gating network for expert selection
//!
//! Implements the routing logic that decides which experts to activate
//! for a given input.

use crate::{RouterError, Result};
use serde::{Deserialize, Serialize};

/// Configuration for the gating network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GatingConfig {
    /// Input dimension
    pub input_dim: usize,
    /// Number of experts
    pub num_experts: usize,
    /// Number of top experts to select (top-K)
    pub top_k: usize,
    /// Temperature for softmax
    pub temperature: f64,
    /// Jitter noise for exploration
    pub jitter_noise: f64,
    /// Whether to use noisy top-k gating
    pub noisy_gating: bool,
}

impl Default for GatingConfig {
    fn default() -> Self {
        Self {
            input_dim: 64,
            num_experts: 8,
            top_k: 2,
            temperature: 1.0,
            jitter_noise: 0.01,
            noisy_gating: true,
        }
    }
}

/// Gating network for expert routing
#[derive(Debug, Clone)]
pub struct GatingNetwork {
    config: GatingConfig,
    /// Gate weights: [input_dim x num_experts]
    gate_weights: Vec<f64>,
    /// Noise weights for noisy gating
    noise_weights: Vec<f64>,
}

impl GatingNetwork {
    /// Create new gating network
    pub fn new(config: GatingConfig) -> Result<Self> {
        if config.top_k > config.num_experts {
            return Err(RouterError::TopKExceedsExperts {
                k: config.top_k,
                n: config.num_experts,
            });
        }

        if config.temperature <= 0.0 {
            return Err(RouterError::InvalidTemperature(config.temperature));
        }

        let weight_size = config.input_dim * config.num_experts;

        // Initialize weights (in production, load from trained model)
        let gate_weights: Vec<f64> = (0..weight_size)
            .map(|i| ((i as f64) * 0.01).sin() * 0.1)
            .collect();

        let noise_weights: Vec<f64> = (0..weight_size)
            .map(|i| ((i as f64) * 0.02).cos() * 0.05)
            .collect();

        Ok(Self {
            config,
            gate_weights,
            noise_weights,
        })
    }

    /// Compute gating scores for input
    ///
    /// Returns (expert_indices, weights) for top-K experts
    pub fn route(&self, input: &[f64], noise: Option<&[f64]>) -> Result<Vec<(usize, f64)>> {
        if input.len() != self.config.input_dim {
            return Err(RouterError::DimensionMismatch {
                expected: self.config.input_dim,
                actual: input.len(),
            });
        }

        // Compute gate logits: W @ x
        let mut logits = vec![0.0; self.config.num_experts];
        for i in 0..self.config.num_experts {
            for j in 0..self.config.input_dim {
                logits[i] += self.gate_weights[i * self.config.input_dim + j] * input[j];
            }
        }

        // Add noise for exploration if enabled
        if self.config.noisy_gating {
            if let Some(noise_vals) = noise {
                for (i, logit) in logits.iter_mut().enumerate() {
                    if i < noise_vals.len() {
                        // Compute noise scale
                        let mut noise_scale = 0.0;
                        for j in 0..self.config.input_dim {
                            noise_scale += self.noise_weights[i * self.config.input_dim + j] * input[j];
                        }
                        noise_scale = noise_scale.abs().max(self.config.jitter_noise);
                        *logit += noise_vals[i] * noise_scale;
                    }
                }
            }
        }

        // Apply temperature
        for logit in &mut logits {
            *logit /= self.config.temperature;
        }

        // Softmax
        let max_logit = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_sum: f64 = logits.iter().map(|&l| (l - max_logit).exp()).sum();
        let probs: Vec<f64> = logits.iter().map(|&l| (l - max_logit).exp() / exp_sum).collect();

        // Select top-K experts
        let mut indexed_probs: Vec<(usize, f64)> = probs.into_iter().enumerate().collect();
        indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        indexed_probs.truncate(self.config.top_k);

        // Renormalize weights for selected experts
        let selected_sum: f64 = indexed_probs.iter().map(|(_, w)| w).sum();
        for (_, w) in &mut indexed_probs {
            *w /= selected_sum;
        }

        Ok(indexed_probs)
    }

    /// Get all expert probabilities (for auxiliary loss computation)
    pub fn all_probabilities(&self, input: &[f64]) -> Result<Vec<f64>> {
        if input.len() != self.config.input_dim {
            return Err(RouterError::DimensionMismatch {
                expected: self.config.input_dim,
                actual: input.len(),
            });
        }

        let mut logits = vec![0.0; self.config.num_experts];
        for i in 0..self.config.num_experts {
            for j in 0..self.config.input_dim {
                logits[i] += self.gate_weights[i * self.config.input_dim + j] * input[j];
            }
            logits[i] /= self.config.temperature;
        }

        let max_logit = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_sum: f64 = logits.iter().map(|&l| (l - max_logit).exp()).sum();

        Ok(logits.iter().map(|&l| (l - max_logit).exp() / exp_sum).collect())
    }

    /// Set gate weights
    pub fn set_weights(&mut self, weights: Vec<f64>) -> Result<()> {
        let expected = self.config.input_dim * self.config.num_experts;
        if weights.len() != expected {
            return Err(RouterError::DimensionMismatch {
                expected,
                actual: weights.len(),
            });
        }
        self.gate_weights = weights;
        Ok(())
    }

    /// Get number of experts
    pub fn num_experts(&self) -> usize {
        self.config.num_experts
    }

    /// Get top-K value
    pub fn top_k(&self) -> usize {
        self.config.top_k
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gating_creation() {
        let config = GatingConfig {
            input_dim: 4,
            num_experts: 4,
            top_k: 2,
            ..Default::default()
        };
        let gate = GatingNetwork::new(config).unwrap();

        assert_eq!(gate.num_experts(), 4);
        assert_eq!(gate.top_k(), 2);
    }

    #[test]
    fn test_routing() {
        let config = GatingConfig {
            input_dim: 4,
            num_experts: 4,
            top_k: 2,
            noisy_gating: false,
            ..Default::default()
        };
        let gate = GatingNetwork::new(config).unwrap();

        let input = vec![1.0, 2.0, 3.0, 4.0];
        let routing = gate.route(&input, None).unwrap();

        assert_eq!(routing.len(), 2);

        // Weights should sum to 1
        let sum: f64 = routing.iter().map(|(_, w)| w).sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_all_probabilities() {
        let config = GatingConfig {
            input_dim: 4,
            num_experts: 4,
            ..Default::default()
        };
        let gate = GatingNetwork::new(config).unwrap();

        let input = vec![1.0, 2.0, 3.0, 4.0];
        let probs = gate.all_probabilities(&input).unwrap();

        assert_eq!(probs.len(), 4);

        // Probabilities should sum to 1
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_top_k_exceeds_experts() {
        let config = GatingConfig {
            input_dim: 4,
            num_experts: 4,
            top_k: 10, // Invalid
            ..Default::default()
        };
        let result = GatingNetwork::new(config);

        assert!(result.is_err());
    }
}
