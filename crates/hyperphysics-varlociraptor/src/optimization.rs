//! Bayesian parameter optimization for variant calling
//!
//! Uses HyperPhysics optimization framework to tune Varlociraptor parameters

use anyhow::Result;
use hyperphysics_optimization::bayesian::BayesianOptimizer;
use serde::{Deserialize, Serialize};

/// Varlociraptor parameter space
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariantCallingParameters {
    /// Prior probability of somatic variant
    pub prior_somatic: f64,

    /// Prior probability of germline variant
    pub prior_germline: f64,

    /// Heterozygous allele frequency prior
    pub het_af_prior: f64,

    /// Homozygous allele frequency prior
    pub hom_af_prior: f64,

    /// Minimum mapping quality
    pub min_mapq: u32,

    /// Minimum base quality
    pub min_baseq: u32,
}

impl Default for VariantCallingParameters {
    fn default() -> Self {
        Self {
            prior_somatic: 0.001,
            prior_germline: 0.001,
            het_af_prior: 0.5,
            hom_af_prior: 1.0,
            min_mapq: 20,
            min_baseq: 20,
        }
    }
}

/// Bayesian parameter optimizer for variant calling
pub struct BayesianParameterOptimizer {
    optimizer: BayesianOptimizer,
}

impl BayesianParameterOptimizer {
    /// Create new optimizer
    pub fn new() -> Self {
        Self {
            optimizer: BayesianOptimizer::new(6), // 6 parameters
        }
    }

    /// Optimize parameters for given validation set
    ///
    /// Uses Bayesian optimization to find optimal parameters that maximize
    /// variant calling accuracy on validation set
    pub async fn optimize<F>(
        &mut self,
        objective_fn: F,
        n_iterations: usize,
    ) -> Result<VariantCallingParameters>
    where
        F: Fn(&VariantCallingParameters) -> f64,
    {
        let mut best_params = VariantCallingParameters::default();
        let mut best_score = f64::NEG_INFINITY;

        for _ in 0..n_iterations {
            // Sample parameters from Bayesian optimizer
            let params = self.sample_parameters()?;

            // Evaluate objective function
            let score = objective_fn(&params);

            // Update optimizer with result
            self.optimizer.observe(self.params_to_vec(&params), score);

            if score > best_score {
                best_score = score;
                best_params = params;
            }
        }

        Ok(best_params)
    }

    /// Sample parameters from optimizer
    fn sample_parameters(&self) -> Result<VariantCallingParameters> {
        let vec = self.optimizer.suggest();
        Ok(self.vec_to_params(&vec))
    }

    /// Convert parameters to vector
    fn params_to_vec(&self, params: &VariantCallingParameters) -> Vec<f64> {
        vec![
            params.prior_somatic,
            params.prior_germline,
            params.het_af_prior,
            params.hom_af_prior,
            params.min_mapq as f64 / 60.0, // Normalize to [0, 1]
            params.min_baseq as f64 / 60.0,
        ]
    }

    /// Convert vector to parameters
    fn vec_to_params(&self, vec: &[f64]) -> VariantCallingParameters {
        VariantCallingParameters {
            prior_somatic: vec[0].max(0.0).min(1.0),
            prior_germline: vec[1].max(0.0).min(1.0),
            het_af_prior: vec[2].max(0.0).min(1.0),
            hom_af_prior: vec[3].max(0.0).min(1.0),
            min_mapq: (vec[4] * 60.0).max(0.0).min(60.0) as u32,
            min_baseq: (vec[5] * 60.0).max(0.0).min(60.0) as u32,
        }
    }
}

impl Default for BayesianParameterOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parameter_conversion() {
        let optimizer = BayesianParameterOptimizer::new();
        let params = VariantCallingParameters::default();

        let vec = optimizer.params_to_vec(&params);
        let reconstructed = optimizer.vec_to_params(&vec);

        assert!((reconstructed.prior_somatic - params.prior_somatic).abs() < 1e-6);
        assert_eq!(reconstructed.min_mapq, params.min_mapq);
    }

    #[tokio::test]
    async fn test_optimization() {
        let mut optimizer = BayesianParameterOptimizer::new();

        // Simple objective: maximize prior_somatic
        let objective = |params: &VariantCallingParameters| params.prior_somatic;

        let result = optimizer.optimize(objective, 10).await;
        assert!(result.is_ok());
    }
}
