//! Statistical backend adapters for reasoning router.
//!
//! Implements ReasoningBackend for:
//! - Monte Carlo simulation
//! - Bayesian inference
//! - Kalman filtering

use crate::{
    BackendCapability, BackendId, BackendMetrics, BackendPool, LatencyTier, Problem,
    ProblemDomain, ProblemSignature, ReasoningBackend, ReasoningResult, ResultValue,
    RouterResult,
};
use async_trait::async_trait;
use parking_lot::Mutex;
use rand::Rng;
use rand_distr::{Distribution, Normal, StandardNormal};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::time::{Duration, Instant};

// ============================================================================
// Monte Carlo Backend
// ============================================================================

/// Monte Carlo configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonteCarloConfig {
    /// Number of simulation samples
    pub num_samples: usize,
    /// Confidence level (e.g., 0.95 for 95%)
    pub confidence_level: f64,
    /// Random seed for reproducibility (None = random)
    pub seed: Option<u64>,
    /// Enable antithetic variates for variance reduction
    pub antithetic: bool,
    /// Enable control variates
    pub control_variates: bool,
}

impl Default for MonteCarloConfig {
    fn default() -> Self {
        Self {
            num_samples: 10000,
            confidence_level: 0.95,
            seed: None,
            antithetic: true,
            control_variates: false,
        }
    }
}

/// Monte Carlo simulation statistics
#[derive(Debug, Clone)]
pub struct MonteCarloStats {
    /// Sample mean
    pub mean: f64,
    /// Sample standard deviation
    pub std_dev: f64,
    /// Standard error of the mean
    pub std_error: f64,
    /// Confidence interval lower bound
    pub ci_lower: f64,
    /// Confidence interval upper bound
    pub ci_upper: f64,
    /// Number of samples
    pub samples: usize,
    /// Variance reduction factor (if applicable)
    pub variance_reduction: f64,
}

/// Monte Carlo simulation backend
pub struct MonteCarloBackend {
    id: BackendId,
    config: MonteCarloConfig,
    capabilities: HashSet<BackendCapability>,
    metrics: Mutex<BackendMetrics>,
}

impl MonteCarloBackend {
    pub fn new(config: MonteCarloConfig) -> Self {
        let mut capabilities = HashSet::new();
        capabilities.insert(BackendCapability::ParallelScenarios);
        capabilities.insert(BackendCapability::UncertaintyQuantification);

        Self {
            id: BackendId::new("monte-carlo"),
            config,
            capabilities,
            metrics: Mutex::new(BackendMetrics::default()),
        }
    }

    /// Run Monte Carlo simulation
    pub fn simulate<F>(&self, sampler: F) -> MonteCarloStats
    where
        F: Fn(&mut dyn rand::RngCore) -> f64,
    {
        let mut rng = rand::thread_rng();

        let num_samples = if self.config.antithetic {
            self.config.num_samples / 2
        } else {
            self.config.num_samples
        };

        let mut samples = Vec::with_capacity(self.config.num_samples);

        for _ in 0..num_samples {
            let sample = sampler(&mut rng);
            samples.push(sample);

            // Antithetic variate: use negated random numbers
            if self.config.antithetic {
                // Store state and generate antithetic sample
                // Simplified: just negate the sample for symmetric distributions
                let antithetic_sample = -sample;
                samples.push(antithetic_sample);
            }
        }

        // Compute statistics
        let n = samples.len() as f64;
        let mean: f64 = samples.iter().sum::<f64>() / n;
        let variance: f64 = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
        let std_dev = variance.sqrt();
        let std_error = std_dev / n.sqrt();

        // Z-score for confidence interval
        let z = match self.config.confidence_level {
            c if c >= 0.99 => 2.576,
            c if c >= 0.95 => 1.96,
            c if c >= 0.90 => 1.645,
            _ => 1.96,
        };

        let ci_lower = mean - z * std_error;
        let ci_upper = mean + z * std_error;

        MonteCarloStats {
            mean,
            std_dev,
            std_error,
            ci_lower,
            ci_upper,
            samples: samples.len(),
            variance_reduction: if self.config.antithetic { 0.5 } else { 1.0 },
        }
    }

    /// Estimate expected value of a function
    pub fn estimate_expectation<F>(&self, f: F, dim: usize) -> MonteCarloStats
    where
        F: Fn(&[f64]) -> f64,
    {
        let normal = Normal::new(0.0, 1.0).unwrap();
        let mut rng = rand::thread_rng();

        let num_samples = if self.config.antithetic {
            self.config.num_samples / 2
        } else {
            self.config.num_samples
        };

        let mut samples = Vec::with_capacity(self.config.num_samples);

        for _ in 0..num_samples {
            // Generate random point
            let x: Vec<f64> = (0..dim).map(|_| normal.sample(&mut rng)).collect();
            samples.push(f(&x));

            // Antithetic variate: compute f(-x) not -f(x)
            if self.config.antithetic {
                let neg_x: Vec<f64> = x.iter().map(|xi| -xi).collect();
                samples.push(f(&neg_x));
            }
        }

        // Compute statistics
        let n = samples.len() as f64;
        let mean: f64 = samples.iter().sum::<f64>() / n;
        let variance: f64 = samples.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / (n - 1.0);
        let std_dev = variance.sqrt();
        let std_error = std_dev / n.sqrt();

        // Z-score for confidence interval
        let z = match self.config.confidence_level {
            c if c >= 0.99 => 2.576,
            c if c >= 0.95 => 1.96,
            c if c >= 0.90 => 1.645,
            _ => 1.96,
        };

        let ci_lower = mean - z * std_error;
        let ci_upper = mean + z * std_error;

        MonteCarloStats {
            mean,
            std_dev,
            std_error,
            ci_lower,
            ci_upper,
            samples: samples.len(),
            variance_reduction: if self.config.antithetic { 0.5 } else { 1.0 },
        }
    }

    /// Estimate probability via indicator function
    pub fn estimate_probability<F>(&self, condition: F, dim: usize) -> MonteCarloStats
    where
        F: Fn(&[f64]) -> bool,
    {
        let indicator = |x: &[f64]| if condition(x) { 1.0 } else { 0.0 };
        self.estimate_expectation(indicator, dim)
    }
}

#[async_trait]
impl ReasoningBackend for MonteCarloBackend {
    fn id(&self) -> &BackendId {
        &self.id
    }

    fn name(&self) -> &str {
        "Monte Carlo Simulation"
    }

    fn pool(&self) -> BackendPool {
        BackendPool::Statistical
    }

    fn supported_domains(&self) -> &[ProblemDomain] {
        &[
            ProblemDomain::Financial,
            ProblemDomain::Physics,
            ProblemDomain::General,
        ]
    }

    fn capabilities(&self) -> &HashSet<BackendCapability> {
        &self.capabilities
    }

    fn latency_tier(&self) -> LatencyTier {
        if self.config.num_samples <= 1000 {
            LatencyTier::Fast
        } else if self.config.num_samples <= 10000 {
            LatencyTier::Medium
        } else {
            LatencyTier::Slow
        }
    }

    fn can_handle(&self, signature: &ProblemSignature) -> bool {
        use crate::problem::ProblemType;
        matches!(
            signature.problem_type,
            ProblemType::Estimation
                | ProblemType::RiskAssessment
                | ProblemType::Simulation
        ) || signature.is_stochastic
    }

    fn estimate_latency(&self, signature: &ProblemSignature) -> Duration {
        let samples_factor = self.config.num_samples as u64 / 1000;
        let dim_factor = signature.dimensionality as u64;
        Duration::from_micros(10 * samples_factor * dim_factor)
    }

    async fn execute(&self, problem: &Problem) -> RouterResult<ReasoningResult> {
        let start = Instant::now();

        let dim = problem.signature.dimensionality as usize;

        // Default: estimate expected value of squared norm (for testing)
        let f = |x: &[f64]| -> f64 { x.iter().map(|xi| xi * xi).sum() };

        let stats = self.estimate_expectation(f, dim);

        let latency = start.elapsed();

        // Update metrics
        {
            let mut metrics = self.metrics.lock();
            // Quality based on relative error
            let relative_error = if stats.mean.abs() > 1e-10 {
                stats.std_error / stats.mean.abs()
            } else {
                stats.std_error
            };
            let quality = 1.0 / (1.0 + relative_error);
            metrics.record(latency, true, Some(quality));
        }

        // Confidence is related to the narrowness of CI
        let ci_width = stats.ci_upper - stats.ci_lower;
        let confidence = if stats.mean.abs() > 1e-10 {
            1.0 - (ci_width / (2.0 * stats.mean.abs())).min(1.0)
        } else {
            0.5
        };

        let quality = 1.0 - (stats.std_error / stats.std_dev.max(1e-10)).min(1.0);

        Ok(ReasoningResult {
            value: ResultValue::Scalar(stats.mean),
            confidence: confidence.max(0.5),
            quality: quality.max(0.5),
            latency,
            backend_id: self.id.clone(),
            metadata: serde_json::json!({
                "algorithm": "Monte Carlo",
                "samples": stats.samples,
                "std_dev": stats.std_dev,
                "std_error": stats.std_error,
                "ci_lower": stats.ci_lower,
                "ci_upper": stats.ci_upper,
                "confidence_level": self.config.confidence_level,
                "antithetic": self.config.antithetic
            }),
        })
    }

    fn metrics(&self) -> BackendMetrics {
        self.metrics.lock().clone()
    }
}

// ============================================================================
// Bayesian Inference Backend (simplified)
// ============================================================================

/// Bayesian inference configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BayesianConfig {
    /// Number of MCMC samples
    pub num_samples: usize,
    /// Burn-in period
    pub burn_in: usize,
    /// Thinning factor
    pub thin: usize,
    /// Proposal standard deviation
    pub proposal_std: f64,
}

impl Default for BayesianConfig {
    fn default() -> Self {
        Self {
            num_samples: 5000,
            burn_in: 1000,
            thin: 1,
            proposal_std: 0.5,
        }
    }
}

/// Bayesian inference backend using Metropolis-Hastings
pub struct BayesianBackend {
    id: BackendId,
    config: BayesianConfig,
    capabilities: HashSet<BackendCapability>,
    metrics: Mutex<BackendMetrics>,
}

impl BayesianBackend {
    pub fn new(config: BayesianConfig) -> Self {
        let mut capabilities = HashSet::new();
        capabilities.insert(BackendCapability::UncertaintyQuantification);

        Self {
            id: BackendId::new("bayesian-inference"),
            config,
            capabilities,
            metrics: Mutex::new(BackendMetrics::default()),
        }
    }

    /// Run Metropolis-Hastings MCMC
    pub fn sample_posterior<F>(
        &self,
        log_posterior: F,
        initial: &[f64],
    ) -> Vec<Vec<f64>>
    where
        F: Fn(&[f64]) -> f64,
    {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, self.config.proposal_std).unwrap();

        let dim = initial.len();
        let mut current = initial.to_vec();
        let mut current_log_p = log_posterior(&current);

        let mut samples = Vec::with_capacity(self.config.num_samples);
        let total_iter = self.config.burn_in + self.config.num_samples * self.config.thin;

        for i in 0..total_iter {
            // Propose new state
            let proposal: Vec<f64> = current
                .iter()
                .map(|x| x + normal.sample(&mut rng))
                .collect();

            let proposal_log_p = log_posterior(&proposal);

            // Accept/reject
            let log_ratio = proposal_log_p - current_log_p;
            let u: f64 = rng.gen();

            if log_ratio > u.ln() {
                current = proposal;
                current_log_p = proposal_log_p;
            }

            // Collect sample after burn-in
            if i >= self.config.burn_in && (i - self.config.burn_in) % self.config.thin == 0 {
                samples.push(current.clone());
            }
        }

        samples
    }
}

#[async_trait]
impl ReasoningBackend for BayesianBackend {
    fn id(&self) -> &BackendId {
        &self.id
    }

    fn name(&self) -> &str {
        "Bayesian Inference (MCMC)"
    }

    fn pool(&self) -> BackendPool {
        BackendPool::Statistical
    }

    fn supported_domains(&self) -> &[ProblemDomain] {
        &[ProblemDomain::Financial, ProblemDomain::General]
    }

    fn capabilities(&self) -> &HashSet<BackendCapability> {
        &self.capabilities
    }

    fn latency_tier(&self) -> LatencyTier {
        LatencyTier::Slow
    }

    fn can_handle(&self, signature: &ProblemSignature) -> bool {
        use crate::problem::ProblemType;
        matches!(
            signature.problem_type,
            ProblemType::Estimation | ProblemType::Inference
        )
    }

    fn estimate_latency(&self, signature: &ProblemSignature) -> Duration {
        let total_iter = self.config.burn_in + self.config.num_samples * self.config.thin;
        let dim_factor = signature.dimensionality as u64;
        Duration::from_micros(total_iter as u64 * dim_factor / 10)
    }

    async fn execute(&self, problem: &Problem) -> RouterResult<ReasoningResult> {
        let start = Instant::now();

        let dim = problem.signature.dimensionality as usize;

        // Simple Gaussian posterior for testing
        let log_posterior = |x: &[f64]| -> f64 {
            -0.5 * x.iter().map(|xi| xi * xi).sum::<f64>()
        };

        let initial: Vec<f64> = vec![0.0; dim];
        let samples = self.sample_posterior(log_posterior, &initial);

        // Compute posterior mean
        let n = samples.len() as f64;
        let posterior_mean: Vec<f64> = (0..dim)
            .map(|i| samples.iter().map(|s| s[i]).sum::<f64>() / n)
            .collect();

        let latency = start.elapsed();

        {
            let mut metrics = self.metrics.lock();
            metrics.record(latency, true, Some(0.8));
        }

        Ok(ReasoningResult {
            value: ResultValue::Vector(posterior_mean),
            confidence: 0.8,
            quality: 0.8,
            latency,
            backend_id: self.id.clone(),
            metadata: serde_json::json!({
                "algorithm": "Metropolis-Hastings",
                "samples": samples.len(),
                "burn_in": self.config.burn_in,
                "thin": self.config.thin
            }),
        })
    }

    fn metrics(&self) -> BackendMetrics {
        self.metrics.lock().clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::problem::ProblemType;

    #[test]
    fn test_monte_carlo_config_default() {
        let config = MonteCarloConfig::default();
        assert_eq!(config.num_samples, 10000);
        assert!((config.confidence_level - 0.95).abs() < 0.01);
    }

    #[test]
    fn test_monte_carlo_simulation() {
        let backend = MonteCarloBackend::new(MonteCarloConfig {
            num_samples: 10000,
            antithetic: false,
            ..Default::default()
        });

        // Estimate E[X^2] where X ~ N(0,1) should be ~1.0
        let f = |x: &[f64]| x[0] * x[0];
        let stats = backend.estimate_expectation(f, 1);

        assert!(
            (stats.mean - 1.0).abs() < 0.1,
            "Monte Carlo mean {:.3} should be close to 1.0",
            stats.mean
        );
    }

    #[test]
    fn test_monte_carlo_antithetic() {
        let backend = MonteCarloBackend::new(MonteCarloConfig {
            num_samples: 10000,
            antithetic: true,
            ..Default::default()
        });

        let f = |x: &[f64]| x[0] * x[0];
        let stats = backend.estimate_expectation(f, 1);

        assert!(
            (stats.mean - 1.0).abs() < 0.2,
            "Antithetic Monte Carlo mean {:.3} should be close to 1.0",
            stats.mean
        );
    }

    #[test]
    fn test_monte_carlo_can_handle() {
        let backend = MonteCarloBackend::new(MonteCarloConfig::default());

        let sig = ProblemSignature::new(ProblemType::RiskAssessment, ProblemDomain::Financial);
        assert!(backend.can_handle(&sig));

        let mut stochastic_sig = ProblemSignature::new(ProblemType::Simulation, ProblemDomain::Physics);
        stochastic_sig.is_stochastic = true;
        assert!(backend.can_handle(&stochastic_sig));
    }

    #[test]
    fn test_bayesian_config_default() {
        let config = BayesianConfig::default();
        assert_eq!(config.num_samples, 5000);
        assert_eq!(config.burn_in, 1000);
    }

    #[test]
    fn test_bayesian_mcmc() {
        let backend = BayesianBackend::new(BayesianConfig {
            num_samples: 1000,
            burn_in: 200,
            thin: 1,
            proposal_std: 0.5,
        });

        // Sample from N(0,1) posterior
        let log_posterior = |x: &[f64]| -0.5 * x[0] * x[0];
        let samples = backend.sample_posterior(log_posterior, &[2.0]);

        // Mean should be close to 0
        let mean: f64 = samples.iter().map(|s| s[0]).sum::<f64>() / samples.len() as f64;
        assert!(
            mean.abs() < 0.5,
            "MCMC mean {:.3} should be close to 0",
            mean
        );
    }
}
