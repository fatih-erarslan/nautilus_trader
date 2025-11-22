use rand::prelude::*;
use rand_distr::{Beta, Gamma, Normal, StudentT};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::f64::consts::{E, PI};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ProbabilisticRiskError {
    #[error("Invalid probability distribution parameters: {0}")]
    InvalidDistributionParams(String),
    #[error("Insufficient data for statistical inference: required {required}, got {available}")]
    InsufficientData { required: usize, available: usize },
    #[error("Bayesian inference failed: {0}")]
    BayesianInferenceFailed(String),
    #[error("Monte Carlo simulation error: {0}")]
    MonteCarloError(String),
    #[error("Numerical instability detected: {0}")]
    NumericalInstability(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbabilisticRiskMetrics {
    pub confidence_interval_95: (f64, f64),
    pub confidence_interval_99: (f64, f64),
    pub expected_shortfall: f64,
    pub tail_risk_probability: f64,
    pub uncertainty_score: f64,
    pub bayesian_var: f64,
    pub heavy_tail_index: f64,
    pub regime_probability: HashMap<String, f64>,
    pub monte_carlo_iterations: usize,
    pub timestamp: u64,
}

#[derive(Debug, Clone)]
pub struct BayesianParameters {
    pub prior_alpha: f64,
    pub prior_beta: f64,
    pub volatility_prior_shape: f64,
    pub volatility_prior_rate: f64,
    pub learning_rate: f64,
    pub evidence_weight: f64,
}

impl Default for BayesianParameters {
    fn default() -> Self {
        Self {
            prior_alpha: 2.0,
            prior_beta: 5.0,
            volatility_prior_shape: 2.0,
            volatility_prior_rate: 1.0,
            learning_rate: 0.1,
            evidence_weight: 0.95,
        }
    }
}

#[derive(Debug, Clone)]
pub struct HeavyTailDistribution {
    pub degrees_of_freedom: f64,
    pub location: f64,
    pub scale: f64,
    pub tail_index: f64,
    pub kurtosis: f64,
}

pub struct ProbabilisticRiskEngine {
    bayesian_params: BayesianParameters,
    historical_returns: Vec<f64>,
    volatility_estimates: Vec<f64>,
    regime_probabilities: HashMap<String, f64>,
    tail_distribution: Option<HeavyTailDistribution>,
    rng: StdRng,
}

impl ProbabilisticRiskEngine {
    pub fn new(bayesian_params: BayesianParameters) -> Self {
        let mut rng = StdRng::from_entropy();

        // Initialize regime probabilities
        let mut regime_probabilities = HashMap::new();
        regime_probabilities.insert("low_volatility".to_string(), 0.4);
        regime_probabilities.insert("medium_volatility".to_string(), 0.4);
        regime_probabilities.insert("high_volatility".to_string(), 0.15);
        regime_probabilities.insert("crisis".to_string(), 0.05);

        Self {
            bayesian_params,
            historical_returns: Vec::new(),
            volatility_estimates: Vec::new(),
            regime_probabilities,
            tail_distribution: None,
            rng,
        }
    }

    /// Bayesian parameter estimation with adaptive thresholds
    pub fn bayesian_parameter_estimation(
        &mut self,
        new_returns: &[f64],
    ) -> Result<(f64, f64), ProbabilisticRiskError> {
        if new_returns.len() < 10 {
            return Err(ProbabilisticRiskError::InsufficientData {
                required: 10,
                available: new_returns.len(),
            });
        }

        // Update historical data
        self.historical_returns.extend_from_slice(new_returns);

        // Keep only recent data for better adaptation
        if self.historical_returns.len() > 252 * 2 {
            // 2 years of daily data
            let excess = self.historical_returns.len() - 252 * 2;
            self.historical_returns.drain(0..excess);
        }

        // Bayesian estimation of mean and volatility
        let sample_mean = new_returns.iter().sum::<f64>() / new_returns.len() as f64;
        let sample_var = new_returns
            .iter()
            .map(|&x| (x - sample_mean).powi(2))
            .sum::<f64>()
            / (new_returns.len() - 1) as f64;

        // Prior parameters
        let n = new_returns.len() as f64;
        let alpha_posterior = self.bayesian_params.prior_alpha + n / 2.0;
        let beta_posterior = self.bayesian_params.prior_beta
            + (n * sample_var + self.bayesian_params.evidence_weight * (sample_mean.powi(2))) / 2.0;

        // Posterior mean and variance for volatility
        let volatility_posterior_mean = alpha_posterior / beta_posterior;
        let volatility_posterior_var = alpha_posterior / (beta_posterior.powi(2));

        // Adaptive threshold based on uncertainty
        let uncertainty_adjustment =
            (volatility_posterior_var.sqrt() * self.bayesian_params.learning_rate).min(0.5);

        Ok((volatility_posterior_mean, uncertainty_adjustment))
    }

    /// Monte Carlo simulation with variance reduction techniques
    pub fn monte_carlo_var_with_variance_reduction(
        &mut self,
        portfolio_value: f64,
        confidence_levels: &[f64],
        iterations: usize,
    ) -> Result<HashMap<String, f64>, ProbabilisticRiskError> {
        if self.historical_returns.len() < 30 {
            return Err(ProbabilisticRiskError::InsufficientData {
                required: 30,
                available: self.historical_returns.len(),
            });
        }

        let mut results = HashMap::new();

        // Estimate parameters from historical data
        let mean_return =
            self.historical_returns.iter().sum::<f64>() / self.historical_returns.len() as f64;
        let volatility = self.estimate_volatility_with_garch()?;

        // Antithetic variates for variance reduction
        let half_iterations = iterations / 2;
        let mut portfolio_returns = Vec::with_capacity(iterations);

        for _ in 0..half_iterations {
            // Generate random normal variable
            let z = Normal::new(0.0, 1.0)
                .map_err(|e| ProbabilisticRiskError::MonteCarloError(e.to_string()))?
                .sample(&mut self.rng);

            // Antithetic pair
            let return1 = mean_return + volatility * z;
            let return2 = mean_return - volatility * z; // Antithetic variate

            portfolio_returns.push(portfolio_value * return1);
            portfolio_returns.push(portfolio_value * return2);
        }

        // Control variates using historical mean
        let control_mean = portfolio_value * mean_return;
        let mut adjusted_returns = Vec::new();

        for &sim_return in &portfolio_returns {
            let adjustment = self.bayesian_params.evidence_weight * (sim_return - control_mean);
            adjusted_returns.push(sim_return - adjustment);
        }

        // Calculate VaR for different confidence levels
        adjusted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());

        for &confidence in confidence_levels {
            let index = ((1.0 - confidence) * adjusted_returns.len() as f64) as usize;
            let var_value = adjusted_returns
                .get(index.min(adjusted_returns.len() - 1))
                .unwrap_or(&0.0);

            results.insert(
                format!("var_{}", (confidence * 100.0) as u8),
                var_value.abs(),
            );
        }

        // Expected Shortfall (CVaR)
        for &confidence in confidence_levels {
            let index = ((1.0 - confidence) * adjusted_returns.len() as f64) as usize;
            let tail_returns: Vec<f64> = adjusted_returns
                .iter()
                .take(index.max(1))
                .cloned()
                .collect();

            let expected_shortfall = tail_returns.iter().sum::<f64>() / tail_returns.len() as f64;
            results.insert(
                format!("es_{}", (confidence * 100.0) as u8),
                expected_shortfall.abs(),
            );
        }

        Ok(results)
    }

    /// Heavy-tail probability distribution modeling
    pub fn model_heavy_tail_distribution(
        &mut self,
    ) -> Result<HeavyTailDistribution, ProbabilisticRiskError> {
        if self.historical_returns.len() < 50 {
            return Err(ProbabilisticRiskError::InsufficientData {
                required: 50,
                available: self.historical_returns.len(),
            });
        }

        // Calculate moments
        let n = self.historical_returns.len() as f64;
        let mean = self.historical_returns.iter().sum::<f64>() / n;

        let variance = self
            .historical_returns
            .iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>()
            / (n - 1.0);

        let skewness = self
            .historical_returns
            .iter()
            .map(|&x| (x - mean).powi(3))
            .sum::<f64>()
            / (n * variance.powf(1.5));

        let kurtosis = self
            .historical_returns
            .iter()
            .map(|&x| (x - mean).powi(4))
            .sum::<f64>()
            / (n * variance.powi(2))
            - 3.0;

        // Estimate degrees of freedom for Student's t-distribution
        // Using method of moments: kurtosis = 6/(nu-4) for nu > 4
        let degrees_of_freedom = if kurtosis > 0.0 {
            (6.0 / kurtosis + 4.0).max(5.0).min(30.0)
        } else {
            30.0 // Normal-like behavior
        };

        // Scale parameter for t-distribution
        let scale = (variance * (degrees_of_freedom - 2.0) / degrees_of_freedom).sqrt();

        // Hill estimator for tail index
        let tail_index = self.estimate_tail_index_hill()?;

        let distribution = HeavyTailDistribution {
            degrees_of_freedom,
            location: mean,
            scale,
            tail_index,
            kurtosis,
        };

        self.tail_distribution = Some(distribution.clone());
        Ok(distribution)
    }

    /// Real-time uncertainty propagation
    pub fn propagate_uncertainty_real_time(
        &mut self,
        new_price: f64,
        previous_uncertainty: f64,
    ) -> Result<f64, ProbabilisticRiskError> {
        if self.historical_returns.is_empty() {
            return Ok(0.5); // Default uncertainty
        }

        // Calculate new return
        if let Some(&last_price) = self.historical_returns.last() {
            let new_return = (new_price - last_price) / last_price;

            // Update Kalman filter-like uncertainty propagation
            let process_noise = 0.01; // Model uncertainty
            let observation_noise = 0.05; // Measurement uncertainty

            // Prediction step
            let predicted_uncertainty = previous_uncertainty + process_noise;

            // Update step
            let kalman_gain = predicted_uncertainty / (predicted_uncertainty + observation_noise);
            let innovation = new_return.abs() - predicted_uncertainty;

            let updated_uncertainty = predicted_uncertainty + kalman_gain * innovation;

            // Clamp between reasonable bounds
            Ok(updated_uncertainty.max(0.01).min(1.0))
        } else {
            Ok(previous_uncertainty)
        }
    }

    /// Regime detection with Markov switching
    pub fn update_regime_probabilities(
        &mut self,
        market_conditions: &HashMap<String, f64>,
    ) -> Result<(), ProbabilisticRiskError> {
        // Transition matrix (simplified)
        let transition_matrix = [
            [0.95, 0.04, 0.005, 0.005], // from low_vol
            [0.05, 0.90, 0.04, 0.01],   // from medium_vol
            [0.01, 0.10, 0.85, 0.04],   // from high_vol
            [0.02, 0.08, 0.30, 0.60],   // from crisis
        ];

        let regimes = [
            "low_volatility",
            "medium_volatility",
            "high_volatility",
            "crisis",
        ];
        let mut new_probabilities = HashMap::new();

        // Get current volatility indicator
        let current_vol = market_conditions.get("volatility").unwrap_or(&0.1);
        let vol_regime_index = match *current_vol {
            x if x < 0.1 => 0,
            x if x < 0.25 => 1,
            x if x < 0.5 => 2,
            _ => 3,
        };

        // Update probabilities using Bayesian updating
        for (i, regime) in regimes.iter().enumerate() {
            let prior = self.regime_probabilities.get(*regime).unwrap_or(&0.25);
            let likelihood = transition_matrix[vol_regime_index][i];
            let posterior = prior * likelihood;
            new_probabilities.insert(regime.to_string(), posterior);
        }

        // Normalize probabilities
        let total: f64 = new_probabilities.values().sum();
        if total > 0.0 {
            for prob in new_probabilities.values_mut() {
                *prob /= total;
            }
        }

        self.regime_probabilities = new_probabilities;
        Ok(())
    }

    /// Generate comprehensive probabilistic risk metrics
    pub fn generate_comprehensive_metrics(
        &mut self,
        portfolio_value: f64,
        market_conditions: &HashMap<String, f64>,
    ) -> Result<ProbabilisticRiskMetrics, ProbabilisticRiskError> {
        // Update regime probabilities
        self.update_regime_probabilities(market_conditions)?;

        // Run Monte Carlo simulation
        let confidence_levels = vec![0.95, 0.99];
        let var_results = self.monte_carlo_var_with_variance_reduction(
            portfolio_value,
            &confidence_levels,
            10000,
        )?;

        // Model heavy tail distribution
        let tail_dist = self.model_heavy_tail_distribution()?;

        // Calculate uncertainty score
        let uncertainty_score = self.calculate_uncertainty_score()?;

        // Bayesian VaR
        let (bayesian_mean, bayesian_std) = self.bayesian_parameter_estimation(
            &self.historical_returns[self.historical_returns.len().saturating_sub(30)..],
        )?;
        let bayesian_var = portfolio_value * (bayesian_mean - 2.33 * bayesian_std); // 99% VaR

        Ok(ProbabilisticRiskMetrics {
            confidence_interval_95: (
                var_results.get("var_95").unwrap_or(&0.0) * 0.8,
                var_results.get("var_95").unwrap_or(&0.0) * 1.2,
            ),
            confidence_interval_99: (
                var_results.get("var_99").unwrap_or(&0.0) * 0.8,
                var_results.get("var_99").unwrap_or(&0.0) * 1.2,
            ),
            expected_shortfall: var_results.get("es_99").unwrap_or(&0.0).clone(),
            tail_risk_probability: self.calculate_tail_risk_probability()?,
            uncertainty_score,
            bayesian_var,
            heavy_tail_index: tail_dist.tail_index,
            regime_probability: self.regime_probabilities.clone(),
            monte_carlo_iterations: 10000,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
        })
    }

    // Private helper methods

    fn estimate_volatility_with_garch(&self) -> Result<f64, ProbabilisticRiskError> {
        if self.historical_returns.len() < 20 {
            return Err(ProbabilisticRiskError::InsufficientData {
                required: 20,
                available: self.historical_returns.len(),
            });
        }

        // Simplified GARCH(1,1) estimation
        let alpha0 = 0.000001; // Long-term variance
        let alpha1 = 0.05; // ARCH term
        let beta1 = 0.90; // GARCH term

        let mut variance = self
            .historical_returns
            .iter()
            .map(|&x| x.powi(2))
            .sum::<f64>()
            / self.historical_returns.len() as f64;

        for &return_val in
            &self.historical_returns[self.historical_returns.len().saturating_sub(10)..]
        {
            variance = alpha0 + alpha1 * return_val.powi(2) + beta1 * variance;
        }

        Ok(variance.sqrt())
    }

    fn estimate_tail_index_hill(&self) -> Result<f64, ProbabilisticRiskError> {
        if self.historical_returns.len() < 50 {
            return Ok(2.5); // Default tail index
        }

        // Sort returns by absolute value in descending order
        let mut abs_returns: Vec<f64> = self.historical_returns.iter().map(|&x| x.abs()).collect();
        abs_returns.sort_by(|a, b| b.partial_cmp(a).unwrap());

        // Hill estimator with optimal number of upper order statistics
        let k = (self.historical_returns.len() as f64).sqrt() as usize;
        let k = k.min(abs_returns.len() / 4).max(10);

        let mut log_sum = 0.0;
        for i in 0..k {
            if abs_returns[i] > 0.0 && abs_returns[k] > 0.0 {
                log_sum += (abs_returns[i] / abs_returns[k]).ln();
            }
        }

        let tail_index = k as f64 / log_sum;

        // Clamp to reasonable bounds
        Ok(tail_index.max(1.5).min(10.0))
    }

    fn calculate_uncertainty_score(&self) -> Result<f64, ProbabilisticRiskError> {
        if self.historical_returns.len() < 20 {
            return Ok(0.5);
        }

        // Combine multiple uncertainty measures
        let vol_uncertainty = self.calculate_volatility_uncertainty();
        let regime_uncertainty = self.calculate_regime_uncertainty();
        let tail_uncertainty = self.calculate_tail_uncertainty();

        let combined_uncertainty = (vol_uncertainty + regime_uncertainty + tail_uncertainty) / 3.0;

        Ok(combined_uncertainty.max(0.0).min(1.0))
    }

    fn calculate_volatility_uncertainty(&self) -> f64 {
        if self.volatility_estimates.len() < 10 {
            return 0.5;
        }

        let mean_vol =
            self.volatility_estimates.iter().sum::<f64>() / self.volatility_estimates.len() as f64;
        let vol_variance = self
            .volatility_estimates
            .iter()
            .map(|&x| (x - mean_vol).powi(2))
            .sum::<f64>()
            / self.volatility_estimates.len() as f64;

        // Coefficient of variation as uncertainty measure
        (vol_variance.sqrt() / mean_vol).min(1.0)
    }

    fn calculate_regime_uncertainty(&self) -> f64 {
        // Entropy-based measure
        let entropy = -self
            .regime_probabilities
            .values()
            .filter(|&&p| p > 0.0)
            .map(|&p| p * p.ln())
            .sum::<f64>();

        // Normalize by maximum entropy
        let max_entropy = (self.regime_probabilities.len() as f64).ln();
        if max_entropy > 0.0 {
            entropy / max_entropy
        } else {
            0.0
        }
    }

    fn calculate_tail_uncertainty(&self) -> f64 {
        // Based on tail index stability
        if let Some(ref dist) = self.tail_distribution {
            // Lower tail index = higher tail uncertainty
            (4.0 - dist.tail_index).max(0.0).min(1.0) / 4.0
        } else {
            0.5
        }
    }

    fn calculate_tail_risk_probability(&self) -> Result<f64, ProbabilisticRiskError> {
        if let Some(ref dist) = self.tail_distribution {
            // Probability of extreme events using Pareto tail approximation
            let threshold = 2.0 * dist.scale; // 2 standard deviations
            let tail_prob = (threshold / dist.scale).powf(-dist.tail_index);
            Ok(tail_prob.min(0.1)) // Cap at 10%
        } else {
            Ok(0.05) // Default 5%
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bayesian_parameter_estimation() {
        let mut engine = ProbabilisticRiskEngine::new(BayesianParameters::default());
        let returns = vec![
            0.01, -0.02, 0.015, -0.005, 0.02, -0.01, 0.008, -0.012, 0.018, -0.007,
        ];

        let result = engine.bayesian_parameter_estimation(&returns);
        assert!(result.is_ok());

        let (volatility, uncertainty) = result.unwrap();
        assert!(volatility > 0.0);
        assert!(uncertainty >= 0.0 && uncertainty <= 1.0);
    }

    #[test]
    fn test_monte_carlo_var() {
        let mut engine = ProbabilisticRiskEngine::new(BayesianParameters::default());

        // Generate some synthetic data
        let mut rng = StdRng::seed_from_u64(42);
        let normal = Normal::new(0.0, 0.02).unwrap();
        let returns: Vec<f64> = (0..100).map(|_| normal.sample(&mut rng)).collect();

        engine.historical_returns = returns;

        let result = engine.monte_carlo_var_with_variance_reduction(100000.0, &[0.95, 0.99], 1000);

        assert!(result.is_ok());
        let vars = result.unwrap();
        assert!(vars.contains_key("var_95"));
        assert!(vars.contains_key("var_99"));
        assert!(vars["var_99"] >= vars["var_95"]); // 99% VaR should be higher
    }

    #[test]
    fn test_heavy_tail_modeling() {
        let mut engine = ProbabilisticRiskEngine::new(BayesianParameters::default());

        // Generate synthetic heavy-tailed data
        let mut rng = StdRng::seed_from_u64(42);
        let t_dist = StudentT::new(5.0).unwrap(); // Heavy tails
        let returns: Vec<f64> = (0..100).map(|_| t_dist.sample(&mut rng) * 0.02).collect();

        engine.historical_returns = returns;

        let result = engine.model_heavy_tail_distribution();
        assert!(result.is_ok());

        let dist = result.unwrap();
        assert!(dist.degrees_of_freedom > 0.0);
        assert!(dist.tail_index > 0.0);
    }
}
