//! Fat-tail distributions and extreme value theory
//!
//! This module implements various fat-tail distributions commonly used in Talebian risk
//! management, including Pareto, Levy, and other heavy-tailed distributions.

pub mod extreme_value;
pub mod fat_tail;
pub mod generalized_pareto;
pub mod levy;
pub mod mixture;
pub mod pareto;
pub mod student_t;

// Re-export key types and traits
pub use pareto::*;
// Re-exports commented to eliminate warnings
// pub use levy::*;
// pub use student_t::*;
// pub use generalized_pareto::*;
// pub use extreme_value::*;
// pub use fat_tail::*;
// pub use mixture::*;

use crate::error::TalebianResult as Result;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

/// Trait for all fat-tail distributions
pub trait FatTailDistribution: Debug + Send + Sync {
    /// Name of the distribution
    fn name(&self) -> &'static str;

    /// Probability density function
    fn pdf(&self, x: f64) -> f64;

    /// Cumulative distribution function
    fn cdf(&self, x: f64) -> f64;

    /// Quantile function (inverse CDF)
    fn quantile(&self, p: f64) -> Result<f64>;

    /// Generate random samples
    fn sample(&self, n: usize) -> Result<Vec<f64>>;

    /// Calculate moments (mean, variance, skewness, kurtosis)
    fn moments(&self) -> Result<DistributionMoments>;

    /// Calculate Value at Risk (VaR) at given confidence level
    fn var(&self, confidence: f64) -> Result<f64> {
        self.quantile(1.0 - confidence)
    }

    /// Calculate Conditional Value at Risk (CVaR) at given confidence level
    fn cvar(&self, confidence: f64) -> Result<f64>;

    /// Calculate tail index (measure of fat-tailedness)
    fn tail_index(&self) -> Result<f64>;

    /// Calculate expected shortfall
    fn expected_shortfall(&self, threshold: f64) -> Result<f64>;

    /// Fit distribution to data
    fn fit(&mut self, data: &[f64]) -> Result<()>;

    /// Calculate log-likelihood for given data
    fn log_likelihood(&self, data: &[f64]) -> Result<f64>;

    /// Calculate AIC (Akaike Information Criterion)
    fn aic(&self, data: &[f64]) -> Result<f64> {
        let ll = self.log_likelihood(data)?;
        let k = self.parameter_count();
        Ok(2.0 * k as f64 - 2.0 * ll)
    }

    /// Calculate BIC (Bayesian Information Criterion)
    fn bic(&self, data: &[f64]) -> Result<f64> {
        let ll = self.log_likelihood(data)?;
        let k = self.parameter_count();
        let n = data.len() as f64;
        Ok(k as f64 * n.ln() - 2.0 * ll)
    }

    /// Get number of parameters
    fn parameter_count(&self) -> usize;

    /// Get parameter values
    fn parameters(&self) -> Vec<f64>;

    /// Set parameter values
    fn set_parameters(&mut self, params: &[f64]) -> Result<()>;

    /// Validate parameters
    fn validate_parameters(&self) -> Result<()>;

    /// Check if distribution has finite moments
    fn has_finite_mean(&self) -> bool;
    fn has_finite_variance(&self) -> bool;
    fn has_finite_higher_moments(&self) -> bool;
}

/// Statistical moments of a distribution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributionMoments {
    /// Mean (first moment)
    pub mean: Option<f64>,
    /// Variance (second central moment)
    pub variance: Option<f64>,
    /// Standard deviation
    pub std_dev: Option<f64>,
    /// Skewness (third standardized moment)
    pub skewness: Option<f64>,
    /// Kurtosis (fourth standardized moment)
    pub kurtosis: Option<f64>,
    /// Excess kurtosis (kurtosis - 3)
    pub excess_kurtosis: Option<f64>,
}

impl DistributionMoments {
    /// Create new moments with all values as None
    pub fn new() -> Self {
        Self {
            mean: None,
            variance: None,
            std_dev: None,
            skewness: None,
            kurtosis: None,
            excess_kurtosis: None,
        }
    }

    /// Set mean and update derived values
    pub fn set_mean(&mut self, mean: f64) {
        self.mean = Some(mean);
    }

    /// Set variance and update derived values
    pub fn set_variance(&mut self, variance: f64) {
        self.variance = Some(variance);
        self.std_dev = Some(variance.sqrt());
    }

    /// Set skewness
    pub fn set_skewness(&mut self, skewness: f64) {
        self.skewness = Some(skewness);
    }

    /// Set kurtosis and update excess kurtosis
    pub fn set_kurtosis(&mut self, kurtosis: f64) {
        self.kurtosis = Some(kurtosis);
        self.excess_kurtosis = Some(kurtosis - 3.0);
    }

    /// Check if moments indicate fat tails
    pub fn has_fat_tails(&self) -> bool {
        self.excess_kurtosis.map_or(false, |k| k > 3.0)
    }

    /// Check if distribution is symmetric
    pub fn is_symmetric(&self) -> bool {
        self.skewness.map_or(false, |s| s.abs() < 0.1)
    }
}

impl Default for DistributionMoments {
    fn default() -> Self {
        Self::new()
    }
}

/// Distribution fitting methods
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FittingMethod {
    /// Maximum likelihood estimation
    MaximumLikelihood,
    /// Method of moments
    MethodOfMoments,
    /// Least squares
    LeastSquares,
    /// Probability weighted moments
    ProbabilityWeightedMoments,
    /// Hill estimator (for tail index)
    HillEstimator,
    /// Peaks over threshold
    PeaksOverThreshold,
}

/// Distribution goodness-of-fit test results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoodnessOfFit {
    /// Kolmogorov-Smirnov test statistic
    pub ks_statistic: f64,
    /// Kolmogorov-Smirnov p-value
    pub ks_p_value: f64,
    /// Anderson-Darling test statistic
    pub ad_statistic: f64,
    /// Anderson-Darling p-value
    pub ad_p_value: f64,
    /// Cramer-von Mises test statistic
    pub cvm_statistic: f64,
    /// Cramer-von Mises p-value
    pub cvm_p_value: f64,
    /// AIC score
    pub aic: f64,
    /// BIC score
    pub bic: f64,
    /// Overall fit quality (0-1, higher is better)
    pub fit_quality: f64,
}

impl GoodnessOfFit {
    /// Create new goodness-of-fit results
    pub fn new() -> Self {
        Self {
            ks_statistic: 0.0,
            ks_p_value: 0.0,
            ad_statistic: 0.0,
            ad_p_value: 0.0,
            cvm_statistic: 0.0,
            cvm_p_value: 0.0,
            aic: f64::INFINITY,
            bic: f64::INFINITY,
            fit_quality: 0.0,
        }
    }

    /// Check if the fit is statistically significant
    pub fn is_good_fit(&self, alpha: f64) -> bool {
        self.ks_p_value > alpha && self.ad_p_value > alpha && self.cvm_p_value > alpha
    }

    /// Get the best test result
    pub fn best_test_result(&self) -> f64 {
        self.ks_p_value.max(self.ad_p_value).max(self.cvm_p_value)
    }
}

impl Default for GoodnessOfFit {
    fn default() -> Self {
        Self::new()
    }
}

/// Extreme value statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtremeValueStats {
    /// Block maxima
    pub block_maxima: Vec<f64>,
    /// Block minima
    pub block_minima: Vec<f64>,
    /// Peaks over threshold
    pub peaks_over_threshold: Vec<f64>,
    /// Threshold used for POT
    pub threshold: f64,
    /// Number of exceedances
    pub exceedances: usize,
    /// Exceedance rate
    pub exceedance_rate: f64,
    /// Return levels
    pub return_levels: Vec<(f64, f64)>, // (period, level)
}

impl ExtremeValueStats {
    /// Create new extreme value statistics
    pub fn new(threshold: f64) -> Self {
        Self {
            block_maxima: Vec::new(),
            block_minima: Vec::new(),
            peaks_over_threshold: Vec::new(),
            threshold,
            exceedances: 0,
            exceedance_rate: 0.0,
            return_levels: Vec::new(),
        }
    }

    /// Calculate block maxima from data
    pub fn calculate_block_maxima(&mut self, data: &[f64], block_size: usize) {
        self.block_maxima.clear();

        for chunk in data.chunks(block_size) {
            if let Some(&max_val) = chunk.iter().max_by(|a, b| a.partial_cmp(b).unwrap()) {
                self.block_maxima.push(max_val);
            }
        }
    }

    /// Calculate block minima from data
    pub fn calculate_block_minima(&mut self, data: &[f64], block_size: usize) {
        self.block_minima.clear();

        for chunk in data.chunks(block_size) {
            if let Some(&min_val) = chunk.iter().min_by(|a, b| a.partial_cmp(b).unwrap()) {
                self.block_minima.push(min_val);
            }
        }
    }

    /// Calculate peaks over threshold
    pub fn calculate_peaks_over_threshold(&mut self, data: &[f64]) {
        self.peaks_over_threshold.clear();
        self.exceedances = 0;

        for &value in data {
            if value > self.threshold {
                self.peaks_over_threshold.push(value - self.threshold);
                self.exceedances += 1;
            }
        }

        self.exceedance_rate = self.exceedances as f64 / data.len() as f64;
    }

    /// Calculate return levels
    pub fn calculate_return_levels(
        &mut self,
        periods: &[f64],
        distribution: &dyn FatTailDistribution,
    ) -> Result<()> {
        self.return_levels.clear();

        for &period in periods {
            let probability = 1.0 / period;
            let level = distribution.quantile(1.0 - probability)?;
            self.return_levels.push((period, level));
        }

        Ok(())
    }
}

/// Utility functions for distribution analysis
pub mod utils {
    use super::*;
    use crate::error::TalebianError;

    /// Calculate empirical quantile
    pub fn empirical_quantile(data: &[f64], p: f64) -> Result<f64> {
        if data.is_empty() {
            return Err(TalebianError::insufficient_data(1, 0));
        }

        if p < 0.0 || p > 1.0 {
            return Err(TalebianError::invalid_parameter(
                "p",
                "Must be between 0 and 1",
            ));
        }

        let mut sorted_data = data.to_vec();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let index = (p * (data.len() - 1) as f64).floor() as usize;
        let fraction = p * (data.len() - 1) as f64 - index as f64;

        if index >= data.len() - 1 {
            Ok(sorted_data[data.len() - 1])
        } else {
            Ok(sorted_data[index] + fraction * (sorted_data[index + 1] - sorted_data[index]))
        }
    }

    /// Calculate empirical CDF
    pub fn empirical_cdf(data: &[f64], x: f64) -> f64 {
        if data.is_empty() {
            return 0.0;
        }

        let count = data.iter().filter(|&&val| val <= x).count();
        count as f64 / data.len() as f64
    }

    /// Calculate Hill estimator for tail index
    pub fn hill_estimator(data: &[f64], k: usize) -> Result<f64> {
        if data.len() < k || k == 0 {
            return Err(TalebianError::insufficient_data(k, data.len()));
        }

        let mut sorted_data = data.to_vec();
        sorted_data.sort_by(|a, b| b.partial_cmp(a).unwrap()); // Sort descending

        let mut sum = 0.0;
        for i in 0..k {
            sum += (sorted_data[i] / sorted_data[k]).ln();
        }

        Ok(sum / k as f64)
    }

    /// Calculate sample moments
    pub fn sample_moments(data: &[f64]) -> Result<DistributionMoments> {
        if data.is_empty() {
            return Err(TalebianError::insufficient_data(1, 0));
        }

        let n = data.len() as f64;
        let mean = data.iter().sum::<f64>() / n;

        let mut variance = 0.0;
        let mut skewness = 0.0;
        let mut kurtosis = 0.0;

        for &x in data {
            let dev = x - mean;
            let dev2 = dev * dev;
            let dev3 = dev2 * dev;
            let dev4 = dev3 * dev;

            variance += dev2;
            skewness += dev3;
            kurtosis += dev4;
        }

        variance /= n - 1.0;
        let std_dev = variance.sqrt();

        if std_dev > 0.0 {
            skewness = (skewness / n) / (std_dev * std_dev * std_dev);
            kurtosis = (kurtosis / n) / (variance * variance);
        } else {
            skewness = 0.0;
            kurtosis = 0.0;
        }

        let mut moments = DistributionMoments::new();
        moments.set_mean(mean);
        moments.set_variance(variance);
        moments.set_skewness(skewness);
        moments.set_kurtosis(kurtosis);

        Ok(moments)
    }

    /// Perform Kolmogorov-Smirnov test
    pub fn kolmogorov_smirnov_test(
        data: &[f64],
        distribution: &dyn FatTailDistribution,
    ) -> Result<(f64, f64)> {
        if data.is_empty() {
            return Err(TalebianError::insufficient_data(1, 0));
        }

        let mut sorted_data = data.to_vec();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let n = data.len() as f64;
        let mut max_diff: f64 = 0.0;

        for (i, &x) in sorted_data.iter().enumerate() {
            let empirical_cdf = (i + 1) as f64 / n;
            let theoretical_cdf = distribution.cdf(x);

            let diff = (empirical_cdf - theoretical_cdf).abs();
            max_diff = max_diff.max(diff);
        }

        // Calculate p-value (approximation)
        let p_value = 2.0 * (-2.0 * n * max_diff * max_diff).exp();

        Ok((max_diff, p_value.min(1.0)))
    }

    /// Automatic threshold selection for POT
    pub fn select_threshold(data: &[f64], method: ThresholdSelection) -> Result<f64> {
        match method {
            ThresholdSelection::Quantile(p) => empirical_quantile(data, p),
            ThresholdSelection::MeanExcess => select_threshold_mean_excess(data),
            ThresholdSelection::Hill => select_threshold_hill(data),
        }
    }

    fn select_threshold_mean_excess(data: &[f64]) -> Result<f64> {
        let mut sorted_data = data.to_vec();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Use 90th percentile as a reasonable threshold
        let threshold_index = (0.9 * sorted_data.len() as f64) as usize;
        Ok(sorted_data[threshold_index])
    }

    fn select_threshold_hill(data: &[f64]) -> Result<f64> {
        let mut sorted_data = data.to_vec();
        sorted_data.sort_by(|a, b| b.partial_cmp(a).unwrap());

        // Use upper 10% of data
        let k = (0.1 * sorted_data.len() as f64) as usize;
        Ok(sorted_data[k])
    }
}

/// Threshold selection methods
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ThresholdSelection {
    /// Use specific quantile
    Quantile(f64),
    /// Use mean excess function
    MeanExcess,
    /// Use Hill estimator
    Hill,
}

#[cfg(test)]
mod tests {
    use super::utils::*;
    use super::*;

    #[test]
    fn test_empirical_quantile() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let q25 = empirical_quantile(&data, 0.25).unwrap();
        let q50 = empirical_quantile(&data, 0.5).unwrap();
        let q75 = empirical_quantile(&data, 0.75).unwrap();

        assert!((q25 - 2.0).abs() < 1e-10);
        assert!((q50 - 3.0).abs() < 1e-10);
        assert!((q75 - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_empirical_cdf() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let cdf_2 = empirical_cdf(&data, 2.0);
        let cdf_3 = empirical_cdf(&data, 3.0);
        let cdf_6 = empirical_cdf(&data, 6.0);

        assert!((cdf_2 - 0.4).abs() < 1e-10);
        assert!((cdf_3 - 0.6).abs() < 1e-10);
        assert!((cdf_6 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_sample_moments() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let moments = sample_moments(&data).unwrap();

        assert!(moments.mean.is_some());
        assert!(moments.variance.is_some());
        assert!(moments.skewness.is_some());
        assert!(moments.kurtosis.is_some());

        let mean = moments.mean.unwrap();
        assert!((mean - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_hill_estimator() {
        // Generate power-law data
        let data: Vec<f64> = (1..=1000).map(|i| (i as f64).powf(-0.5)).collect();

        let hill = hill_estimator(&data, 100).unwrap();
        assert!(hill > 0.0);
    }

    #[test]
    fn test_distribution_moments() {
        let mut moments = DistributionMoments::new();

        moments.set_mean(0.0);
        moments.set_variance(1.0);
        moments.set_skewness(0.0);
        moments.set_kurtosis(6.0);

        assert!(moments.has_fat_tails());
        assert!(moments.is_symmetric());
    }

    #[test]
    fn test_extreme_value_stats() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let mut stats = ExtremeValueStats::new(7.0);

        stats.calculate_block_maxima(&data, 3);
        assert_eq!(stats.block_maxima.len(), 3);
        assert_eq!(stats.block_maxima[0], 3.0);
        assert_eq!(stats.block_maxima[1], 6.0);
        assert_eq!(stats.block_maxima[2], 9.0);

        stats.calculate_peaks_over_threshold(&data);
        assert_eq!(stats.peaks_over_threshold.len(), 3);
        assert_eq!(stats.exceedances, 3);
        assert_eq!(stats.exceedance_rate, 0.3);
    }

    #[test]
    fn test_threshold_selection() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        let threshold = select_threshold(&data, ThresholdSelection::Quantile(0.8)).unwrap();
        assert!((threshold - 8.2).abs() < 1e-10);
    }
}
