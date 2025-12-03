//! Expert implementations for strategy routing
//!
//! Provides different expert types that can be used in the MoE system.

use crate::{RouterError, Result};
use serde::{Deserialize, Serialize};
use hyperphysics_lorentz::{SimdMinkowski, poincare_to_lorentz, lorentz_to_poincare};

/// Type of expert computation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExpertType {
    /// Standard dot-product attention
    Standard,
    /// Hyperbolic (Poincaré/Lorentz) attention
    Hyperbolic,
    /// Linear random features
    Linear,
    /// Momentum-based expert
    Momentum,
    /// Mean-reversion expert
    MeanReversion,
    /// Volatility-adjusted expert
    Volatility,
}

/// Configuration for an expert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertConfig {
    /// Input/output dimension
    pub dim: usize,
    /// Expert type
    pub expert_type: ExpertType,
    /// Temperature for attention (if applicable)
    pub temperature: f64,
    /// Curvature for hyperbolic experts
    pub curvature: f64,
    /// Expert-specific parameters
    pub params: Vec<f64>,
}

impl Default for ExpertConfig {
    fn default() -> Self {
        Self {
            dim: 64,
            expert_type: ExpertType::Standard,
            temperature: 1.0,
            curvature: -1.0,
            params: Vec::new(),
        }
    }
}

/// Trait for expert computation
pub trait Expert: Send + Sync {
    /// Compute expert output for given input
    fn compute(&self, input: &[f64]) -> Result<Vec<f64>>;

    /// Get expert type
    fn expert_type(&self) -> ExpertType;

    /// Get input dimension
    fn dim(&self) -> usize;

    /// Get expert name/identifier
    fn name(&self) -> &str;
}

/// Standard expert using dot-product attention
#[derive(Debug, Clone)]
pub struct StandardExpert {
    config: ExpertConfig,
    name: String,
    /// Weight matrix (flattened)
    weights: Vec<f64>,
}

impl StandardExpert {
    /// Create new standard expert
    pub fn new(config: ExpertConfig, name: String) -> Self {
        let dim = config.dim;
        // Initialize weights (in practice, load from trained model)
        let weights = vec![0.01; dim * dim];
        Self { config, name, weights }
    }

    /// Set weights
    pub fn set_weights(&mut self, weights: Vec<f64>) -> Result<()> {
        let expected = self.config.dim * self.config.dim;
        if weights.len() != expected {
            return Err(RouterError::DimensionMismatch {
                expected,
                actual: weights.len(),
            });
        }
        self.weights = weights;
        Ok(())
    }
}

impl Expert for StandardExpert {
    fn compute(&self, input: &[f64]) -> Result<Vec<f64>> {
        if input.len() != self.config.dim {
            return Err(RouterError::DimensionMismatch {
                expected: self.config.dim,
                actual: input.len(),
            });
        }

        let dim = self.config.dim;
        let mut output = vec![0.0; dim];

        // Matrix-vector multiplication: output = W * input
        for i in 0..dim {
            for j in 0..dim {
                output[i] += self.weights[i * dim + j] * input[j];
            }
        }

        Ok(output)
    }

    fn expert_type(&self) -> ExpertType {
        ExpertType::Standard
    }

    fn dim(&self) -> usize {
        self.config.dim
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Hyperbolic expert using Lorentz model
#[derive(Debug, Clone)]
pub struct HyperbolicExpert {
    config: ExpertConfig,
    name: String,
    /// Reference points in Lorentz coordinates
    reference_points: Vec<Vec<f64>>,
}

impl HyperbolicExpert {
    /// Create new hyperbolic expert
    pub fn new(config: ExpertConfig, name: String) -> Self {
        Self {
            config,
            name,
            reference_points: Vec::new(),
        }
    }

    /// Add reference point (in Poincaré coordinates)
    pub fn add_reference_point(&mut self, poincare_point: &[f64]) -> Result<()> {
        if poincare_point.len() != self.config.dim {
            return Err(RouterError::DimensionMismatch {
                expected: self.config.dim,
                actual: poincare_point.len(),
            });
        }

        let lorentz = poincare_to_lorentz(poincare_point, self.config.curvature)?;
        self.reference_points.push(lorentz);
        Ok(())
    }

    /// Compute hyperbolic attention weights
    fn compute_attention(&self, query: &[f64]) -> Result<Vec<f64>> {
        if self.reference_points.is_empty() {
            return Err(RouterError::ComputationError(
                "No reference points set".to_string(),
            ));
        }

        let query_lorentz = poincare_to_lorentz(query, self.config.curvature)?;
        let mut weights = Vec::with_capacity(self.reference_points.len());

        for ref_point in &self.reference_points {
            // Use negative Lorentz distance as similarity
            let dist = SimdMinkowski::distance(&query_lorentz, ref_point, self.config.curvature)?;
            weights.push(-dist / self.config.temperature);
        }

        // Softmax
        let max_w = weights.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_sum: f64 = weights.iter().map(|&w| (w - max_w).exp()).sum();

        for w in &mut weights {
            *w = (*w - max_w).exp() / exp_sum;
        }

        Ok(weights)
    }
}

impl Expert for HyperbolicExpert {
    fn compute(&self, input: &[f64]) -> Result<Vec<f64>> {
        if input.len() != self.config.dim {
            return Err(RouterError::DimensionMismatch {
                expected: self.config.dim,
                actual: input.len(),
            });
        }

        if self.reference_points.is_empty() {
            // If no reference points, return identity
            return Ok(input.to_vec());
        }

        // Compute attention weights
        let weights = self.compute_attention(input)?;

        // Weighted aggregation in Lorentz space (simplified)
        let input_lorentz = poincare_to_lorentz(input, self.config.curvature)?;
        let mut aggregated = vec![0.0; input_lorentz.len()];

        for (w, ref_point) in weights.iter().zip(&self.reference_points) {
            for (i, &r) in ref_point.iter().enumerate() {
                aggregated[i] += w * r;
            }
        }

        // Project back to Poincaré
        let result = lorentz_to_poincare(&aggregated, self.config.curvature)?;
        Ok(result)
    }

    fn expert_type(&self) -> ExpertType {
        ExpertType::Hyperbolic
    }

    fn dim(&self) -> usize {
        self.config.dim
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Linear expert using random features
#[derive(Debug, Clone)]
pub struct LinearExpert {
    config: ExpertConfig,
    name: String,
    /// Random projection matrix
    projection: Vec<f64>,
    /// Bias vector
    bias: Vec<f64>,
}

impl LinearExpert {
    /// Create new linear expert with random features
    pub fn new(config: ExpertConfig, name: String, seed: u64) -> Self {
        let dim = config.dim;

        // Simple deterministic initialization based on seed
        // In production, use proper random initialization
        let projection: Vec<f64> = (0..dim * dim)
            .map(|i| {
                let x = (i as f64 + seed as f64) * 0.1;
                (x.sin() * 0.1).clamp(-0.5, 0.5)
            })
            .collect();

        let bias: Vec<f64> = (0..dim)
            .map(|i| ((i as f64 + seed as f64) * 0.05).sin() * 0.01)
            .collect();

        Self {
            config,
            name,
            projection,
            bias,
        }
    }
}

impl Expert for LinearExpert {
    fn compute(&self, input: &[f64]) -> Result<Vec<f64>> {
        if input.len() != self.config.dim {
            return Err(RouterError::DimensionMismatch {
                expected: self.config.dim,
                actual: input.len(),
            });
        }

        let dim = self.config.dim;
        let mut output = self.bias.clone();

        for i in 0..dim {
            for j in 0..dim {
                output[i] += self.projection[i * dim + j] * input[j];
            }
        }

        // ReLU activation
        for o in &mut output {
            *o = o.max(0.0);
        }

        Ok(output)
    }

    fn expert_type(&self) -> ExpertType {
        ExpertType::Linear
    }

    fn dim(&self) -> usize {
        self.config.dim
    }

    fn name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_standard_expert() {
        let config = ExpertConfig {
            dim: 4,
            ..Default::default()
        };
        let expert = StandardExpert::new(config, "test".to_string());

        let input = vec![1.0, 2.0, 3.0, 4.0];
        let output = expert.compute(&input).unwrap();

        assert_eq!(output.len(), 4);
    }

    #[test]
    fn test_linear_expert() {
        let config = ExpertConfig {
            dim: 4,
            expert_type: ExpertType::Linear,
            ..Default::default()
        };
        let expert = LinearExpert::new(config, "linear_test".to_string(), 42);

        let input = vec![1.0, 2.0, 3.0, 4.0];
        let output = expert.compute(&input).unwrap();

        assert_eq!(output.len(), 4);
        // ReLU ensures non-negative
        assert!(output.iter().all(|&x| x >= 0.0));
    }

    #[test]
    fn test_hyperbolic_expert() {
        let config = ExpertConfig {
            dim: 2,
            expert_type: ExpertType::Hyperbolic,
            curvature: -1.0,
            ..Default::default()
        };
        let mut expert = HyperbolicExpert::new(config, "hyp_test".to_string());

        // Add some reference points
        expert.add_reference_point(&[0.1, 0.2]).unwrap();
        expert.add_reference_point(&[0.3, 0.1]).unwrap();

        let input = vec![0.2, 0.15];
        let output = expert.compute(&input).unwrap();

        assert_eq!(output.len(), 2);
    }

    #[test]
    fn test_dimension_mismatch() {
        let config = ExpertConfig {
            dim: 4,
            ..Default::default()
        };
        let expert = StandardExpert::new(config, "test".to_string());

        let input = vec![1.0, 2.0]; // Wrong dimension
        let result = expert.compute(&input);

        assert!(result.is_err());
    }
}
