//! Core LMSR implementation

use crate::errors::{LMSRError, Result};
use crate::errors::{validate_liquidity, validate_not_empty, validate_probabilities, safe_exp, safe_log, safe_divide};
use crate::utils::{normalize_probabilities, log_odds_to_probabilities, probabilities_to_log_odds};
use crate::factors::StandardFactors;
use crate::aggregation::{AggregationMethod, AggregationResult};
use serde::{Deserialize, Serialize};
use std::fmt;

/// Precision mode for LMSR calculations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Precision {
    /// Single precision (f32)
    Single,
    /// Double precision (f64)
    Double,
}

impl Default for Precision {
    fn default() -> Self {
        Precision::Double
    }
}

/// Configuration for LMSR instance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LMSRConfig {
    /// Liquidity parameter (b) - controls market depth
    pub liquidity: f64,
    /// Precision mode for calculations
    pub precision: Precision,
    /// Enable parallel processing
    pub parallel: bool,
    /// Enable SIMD optimizations
    pub simd: bool,
    /// Numerical stability threshold
    pub stability_threshold: f64,
    /// Maximum iterations for convergence
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
}

impl Default for LMSRConfig {
    fn default() -> Self {
        Self {
            liquidity: crate::DEFAULT_LIQUIDITY,
            precision: Precision::Double,
            parallel: true,
            simd: true,
            stability_threshold: 1e-10,
            max_iterations: 1000,
            tolerance: 1e-8,
        }
    }
}

/// Main LMSR implementation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LMSR {
    config: LMSRConfig,
}

impl LMSR {
    /// Create a new LMSR instance with default configuration
    pub fn new(liquidity: f64) -> Self {
        let mut config = LMSRConfig::default();
        config.liquidity = liquidity;
        Self { config }
    }

    /// Create a new LMSR instance with custom configuration
    pub fn with_config(config: LMSRConfig) -> Result<Self> {
        validate_liquidity(config.liquidity)?;
        Ok(Self { config })
    }

    /// Get the liquidity parameter
    pub fn liquidity(&self) -> f64 {
        self.config.liquidity
    }

    /// Get the configuration
    pub fn config(&self) -> &LMSRConfig {
        &self.config
    }

    /// Update the liquidity parameter
    pub fn set_liquidity(&mut self, liquidity: f64) -> Result<()> {
        validate_liquidity(liquidity)?;
        self.config.liquidity = liquidity;
        Ok(())
    }

    /// Calculate the LMSR cost function C(q) = b * log(sum(exp(q_i/b)))
    /// 
    /// # Arguments
    /// * `quantities` - Vector of quantities for each outcome
    /// 
    /// # Returns
    /// The cost function value
    pub fn cost_function(&self, quantities: &[f64]) -> Result<f64> {
        validate_not_empty(quantities)?;
        
        let b = self.config.liquidity;
        let mut sum_exp = 0.0;
        let max_q = quantities.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        
        // Use max subtraction for numerical stability
        for &q in quantities {
            let exp_val = safe_exp((q - max_q) / b)?;
            sum_exp += exp_val;
        }
        
        let log_sum = safe_log(sum_exp)?;
        Ok(b * (log_sum + max_q / b))
    }

    /// Calculate market probabilities from quantities
    /// p_i = exp(q_i/b) / sum(exp(q_j/b))
    /// 
    /// # Arguments
    /// * `quantities` - Vector of quantities for each outcome
    /// 
    /// # Returns
    /// Vector of market probabilities
    pub fn market_probabilities(&self, quantities: &[f64]) -> Result<Vec<f64>> {
        validate_not_empty(quantities)?;
        
        let b = self.config.liquidity;
        let mut probabilities = Vec::with_capacity(quantities.len());
        let max_q = quantities.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        
        // Calculate unnormalized probabilities
        let mut sum_exp = 0.0;
        for &q in quantities {
            let exp_val = safe_exp((q - max_q) / b)?;
            probabilities.push(exp_val);
            sum_exp += exp_val;
        }
        
        // Normalize
        for p in &mut probabilities {
            *p = safe_divide(*p, sum_exp)?;
        }
        
        Ok(probabilities)
    }

    /// Calculate the cost to move from current quantities to target quantities
    /// 
    /// # Arguments
    /// * `current_quantities` - Current market quantities
    /// * `target_quantities` - Target quantities after trade
    /// 
    /// # Returns
    /// The cost to make the trade
    pub fn cost_to_move(&self, current_quantities: &[f64], target_quantities: &[f64]) -> Result<f64> {
        validate_not_empty(current_quantities)?;
        validate_not_empty(target_quantities)?;
        
        if current_quantities.len() != target_quantities.len() {
            return Err(LMSRError::dimension_mismatch(
                current_quantities.len(),
                target_quantities.len(),
            ));
        }
        
        let current_cost = self.cost_function(current_quantities)?;
        let target_cost = self.cost_function(target_quantities)?;
        
        Ok(target_cost - current_cost)
    }

    /// Calculate KL divergence between two probability distributions
    /// KL(P || Q) = sum(P_i * log(P_i / Q_i))
    /// 
    /// # Arguments
    /// * `p` - First probability distribution
    /// * `q` - Second probability distribution
    /// 
    /// # Returns
    /// The KL divergence
    pub fn kl_divergence(&self, p: &[f64], q: &[f64]) -> Result<f64> {
        validate_not_empty(p)?;
        validate_not_empty(q)?;
        validate_probabilities(p)?;
        validate_probabilities(q)?;
        
        if p.len() != q.len() {
            return Err(LMSRError::dimension_mismatch(p.len(), q.len()));
        }
        
        let mut kl_div = 0.0;
        for (&pi, &qi) in p.iter().zip(q.iter()) {
            if pi > 0.0 && qi > 0.0 {
                let log_ratio = safe_log(safe_divide(pi, qi)?)?;
                kl_div += pi * log_ratio;
            } else if pi > 0.0 && qi == 0.0 {
                return Ok(f64::INFINITY);
            }
        }
        
        Ok(kl_div)
    }

    /// Calculate information gain (mutual information)
    /// 
    /// # Arguments
    /// * `prior` - Prior probability distribution
    /// * `posterior` - Posterior probability distribution
    /// 
    /// # Returns
    /// The information gain
    pub fn information_gain(&self, prior: &[f64], posterior: &[f64]) -> Result<f64> {
        self.kl_divergence(posterior, prior)
    }

    /// Aggregate multiple probability distributions using specified method
    /// 
    /// # Arguments
    /// * `distributions` - Vector of probability distributions to aggregate
    /// * `method` - Aggregation method to use
    /// 
    /// # Returns
    /// The aggregated probability distribution
    pub fn aggregate_probabilities(
        &self,
        distributions: &[Vec<f64>],
        method: AggregationMethod,
    ) -> Result<AggregationResult> {
        validate_not_empty(distributions)?;
        
        let n_outcomes = distributions[0].len();
        for (i, dist) in distributions.iter().enumerate() {
            if dist.len() != n_outcomes {
                return Err(LMSRError::dimension_mismatch(n_outcomes, dist.len()));
            }
            validate_probabilities(dist)?;
        }
        
        match method {
            AggregationMethod::LogOdds => self.aggregate_log_odds(distributions),
            AggregationMethod::Geometric => self.aggregate_geometric(distributions),
            AggregationMethod::Arithmetic => self.aggregate_arithmetic(distributions),
            AggregationMethod::Quantum => self.aggregate_quantum(distributions),
        }
    }

    /// Aggregate using log-odds method
    fn aggregate_log_odds(&self, distributions: &[Vec<f64>]) -> Result<AggregationResult> {
        let n_outcomes = distributions[0].len();
        let mut aggregated_log_odds = vec![0.0; n_outcomes];
        
        for dist in distributions {
            let log_odds = probabilities_to_log_odds(dist)?;
            for (i, &lo) in log_odds.iter().enumerate() {
                aggregated_log_odds[i] += lo;
            }
        }
        
        // Average the log-odds
        let n_dists = distributions.len() as f64;
        for lo in &mut aggregated_log_odds {
            *lo /= n_dists;
        }
        
        let aggregated_probs = log_odds_to_probabilities(&aggregated_log_odds)?;
        
        Ok(AggregationResult {
            probabilities: aggregated_probs,
            confidence: self.calculate_confidence(distributions)?,
            method: AggregationMethod::LogOdds,
            n_distributions: distributions.len(),
        })
    }

    /// Aggregate using geometric mean
    fn aggregate_geometric(&self, distributions: &[Vec<f64>]) -> Result<AggregationResult> {
        let n_outcomes = distributions[0].len();
        let mut aggregated = vec![1.0; n_outcomes];
        
        for dist in distributions {
            for (i, &p) in dist.iter().enumerate() {
                aggregated[i] *= p;
            }
        }
        
        let n_dists = distributions.len() as f64;
        for p in &mut aggregated {
            *p = p.powf(1.0 / n_dists);
        }
        
        let normalized = normalize_probabilities(&aggregated)?;
        
        Ok(AggregationResult {
            probabilities: normalized,
            confidence: self.calculate_confidence(distributions)?,
            method: AggregationMethod::Geometric,
            n_distributions: distributions.len(),
        })
    }

    /// Aggregate using arithmetic mean
    fn aggregate_arithmetic(&self, distributions: &[Vec<f64>]) -> Result<AggregationResult> {
        let n_outcomes = distributions[0].len();
        let mut aggregated = vec![0.0; n_outcomes];
        
        for dist in distributions {
            for (i, &p) in dist.iter().enumerate() {
                aggregated[i] += p;
            }
        }
        
        let n_dists = distributions.len() as f64;
        for p in &mut aggregated {
            *p /= n_dists;
        }
        
        Ok(AggregationResult {
            probabilities: aggregated,
            confidence: self.calculate_confidence(distributions)?,
            method: AggregationMethod::Arithmetic,
            n_distributions: distributions.len(),
        })
    }

    /// Aggregate using quantum method (complex probability amplitudes)
    fn aggregate_quantum(&self, distributions: &[Vec<f64>]) -> Result<AggregationResult> {
        let n_outcomes = distributions[0].len();
        let mut aggregated = vec![0.0; n_outcomes];
        
        // Convert probabilities to amplitudes (square root)
        for dist in distributions {
            for (i, &p) in dist.iter().enumerate() {
                aggregated[i] += p.sqrt();
            }
        }
        
        // Average amplitudes and square to get probabilities
        let n_dists = distributions.len() as f64;
        for p in &mut aggregated {
            *p = (*p / n_dists).powi(2);
        }
        
        let normalized = normalize_probabilities(&aggregated)?;
        
        Ok(AggregationResult {
            probabilities: normalized,
            confidence: self.calculate_confidence(distributions)?,
            method: AggregationMethod::Quantum,
            n_distributions: distributions.len(),
        })
    }

    /// Calculate confidence score for aggregation
    fn calculate_confidence(&self, distributions: &[Vec<f64>]) -> Result<f64> {
        if distributions.len() < 2 {
            return Ok(1.0);
        }
        
        // Calculate average pairwise KL divergence
        let mut total_kl = 0.0;
        let mut count = 0;
        
        for i in 0..distributions.len() {
            for j in (i + 1)..distributions.len() {
                let kl1 = self.kl_divergence(&distributions[i], &distributions[j])?;
                let kl2 = self.kl_divergence(&distributions[j], &distributions[i])?;
                total_kl += (kl1 + kl2) / 2.0;
                count += 1;
            }
        }
        
        let avg_kl = total_kl / count as f64;
        
        // Convert to confidence score (higher KL = lower confidence)
        let confidence = 1.0 / (1.0 + avg_kl);
        Ok(confidence)
    }

    /// Aggregate probabilities using standard 8-factor model
    /// 
    /// # Arguments
    /// * `factor_values` - Values for each of the 8 factors
    /// * `factors` - Standard factors configuration
    /// 
    /// # Returns
    /// Aggregated probability distribution
    pub fn aggregate_standard_factors(
        &self,
        factor_values: &[f64],
        factors: &StandardFactors,
    ) -> Result<Vec<f64>> {
        if factor_values.len() != 8 {
            return Err(LMSRError::dimension_mismatch(8, factor_values.len()));
        }
        
        let weighted_values = factors.apply_weights(factor_values)?;
        let probabilities = self.values_to_probabilities(&weighted_values)?;
        
        Ok(probabilities)
    }

    /// Convert factor values to probabilities
    fn values_to_probabilities(&self, values: &[f64]) -> Result<Vec<f64>> {
        let b = self.config.liquidity;
        let mut probabilities = Vec::with_capacity(values.len());
        let max_val = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        
        // Calculate unnormalized probabilities
        let mut sum_exp = 0.0;
        for &v in values {
            let exp_val = safe_exp((v - max_val) / b)?;
            probabilities.push(exp_val);
            sum_exp += exp_val;
        }
        
        // Normalize
        for p in &mut probabilities {
            *p = safe_divide(*p, sum_exp)?;
        }
        
        Ok(probabilities)
    }

    /// Batch process multiple quantity sets
    pub fn batch_market_probabilities(&self, quantities_batch: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        validate_not_empty(quantities_batch)?;
        
        let mut results = Vec::with_capacity(quantities_batch.len());
        
        #[cfg(feature = "parallel")]
        {
            if self.config.parallel {
                use rayon::prelude::*;
                let parallel_results: Result<Vec<Vec<f64>>> = quantities_batch
                    .par_iter()
                    .map(|quantities| self.market_probabilities(quantities))
                    .collect();
                return parallel_results;
            }
        }
        
        // Sequential processing
        for quantities in quantities_batch {
            results.push(self.market_probabilities(quantities)?);
        }
        
        Ok(results)
    }

    /// Batch process cost function calculations
    pub fn batch_cost_function(&self, quantities_batch: &[Vec<f64>]) -> Result<Vec<f64>> {
        validate_not_empty(quantities_batch)?;
        
        let mut results = Vec::with_capacity(quantities_batch.len());
        
        #[cfg(feature = "parallel")]
        {
            if self.config.parallel {
                use rayon::prelude::*;
                let parallel_results: Result<Vec<f64>> = quantities_batch
                    .par_iter()
                    .map(|quantities| self.cost_function(quantities))
                    .collect();
                return parallel_results;
            }
        }
        
        // Sequential processing
        for quantities in quantities_batch {
            results.push(self.cost_function(quantities)?);
        }
        
        Ok(results)
    }
}

impl fmt::Display for LMSR {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "LMSR(liquidity={}, precision={:?}, parallel={}, simd={})",
            self.config.liquidity, self.config.precision, self.config.parallel, self.config.simd
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_lmsr_creation() {
        let lmsr = LMSR::new(100.0);
        assert_eq!(lmsr.liquidity(), 100.0);
    }

    #[test]
    fn test_cost_function() {
        let lmsr = LMSR::new(100.0);
        let quantities = vec![0.0, 0.0, 0.0];
        let cost = lmsr.cost_function(&quantities).unwrap();
        
        // Cost should be b * log(n) for equal quantities
        let expected = 100.0 * (3.0_f64).ln();
        assert_relative_eq!(cost, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_market_probabilities() {
        let lmsr = LMSR::new(100.0);
        let quantities = vec![0.0, 0.0, 0.0];
        let probabilities = lmsr.market_probabilities(&quantities).unwrap();
        
        // Should be uniform distribution
        for p in &probabilities {
            assert_relative_eq!(*p, 1.0 / 3.0, epsilon = 1e-10);
        }
        
        // Should sum to 1.0
        let sum: f64 = probabilities.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_cost_to_move() {
        let lmsr = LMSR::new(100.0);
        let current = vec![0.0, 0.0, 0.0];
        let target = vec![10.0, 0.0, 0.0];
        
        let cost = lmsr.cost_to_move(&current, &target).unwrap();
        assert!(cost > 0.0);
    }

    #[test]
    fn test_kl_divergence() {
        let lmsr = LMSR::new(100.0);
        let p = vec![0.5, 0.3, 0.2];
        let q = vec![0.4, 0.4, 0.2];
        
        let kl = lmsr.kl_divergence(&p, &q).unwrap();
        assert!(kl > 0.0);
        
        // KL divergence with itself should be 0
        let kl_self = lmsr.kl_divergence(&p, &p).unwrap();
        assert_relative_eq!(kl_self, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_information_gain() {
        let lmsr = LMSR::new(100.0);
        let prior = vec![0.33, 0.33, 0.34];
        let posterior = vec![0.6, 0.3, 0.1];
        
        let info_gain = lmsr.information_gain(&prior, &posterior).unwrap();
        assert!(info_gain > 0.0);
    }

    #[test]
    fn test_aggregate_probabilities() {
        let lmsr = LMSR::new(100.0);
        let distributions = vec![
            vec![0.5, 0.3, 0.2],
            vec![0.4, 0.4, 0.2],
            vec![0.6, 0.2, 0.2],
        ];
        
        let result = lmsr.aggregate_probabilities(&distributions, AggregationMethod::Arithmetic).unwrap();
        
        // Should sum to 1.0
        let sum: f64 = result.probabilities.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-10);
        
        // Should be average of inputs
        let expected = vec![0.5, 0.3, 0.2];
        for (actual, expected) in result.probabilities.iter().zip(expected.iter()) {
            assert_relative_eq!(*actual, *expected, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_batch_processing() {
        let lmsr = LMSR::new(100.0);
        let batch = vec![
            vec![0.0, 0.0, 0.0],
            vec![10.0, 0.0, 0.0],
            vec![0.0, 10.0, 0.0],
        ];
        
        let results = lmsr.batch_market_probabilities(&batch).unwrap();
        assert_eq!(results.len(), 3);
        
        for probs in &results {
            let sum: f64 = probs.iter().sum();
            assert_relative_eq!(sum, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_standard_factors() {
        let lmsr = LMSR::new(100.0);
        let factors = StandardFactors::new();
        let factor_values = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        
        let result = lmsr.aggregate_standard_factors(&factor_values, &factors).unwrap();
        
        // Should sum to 1.0
        let sum: f64 = result.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-10);
        
        // All probabilities should be positive
        for p in &result {
            assert!(*p > 0.0);
        }
    }
}