//! Power law fitting and analysis for SOC systems

use crate::{Result, SocError};

/// Power law fitter for analyzing avalanche distributions
pub struct PowerLawFitter {
    min_data_points: usize,
    xmin_search_fraction: f32,
}

impl PowerLawFitter {
    /// Create a new power law fitter
    pub fn new() -> Self {
        Self {
            min_data_points: 30,
            xmin_search_fraction: 0.25,
        }
    }
    
    /// Fit power law to avalanche size distribution
    pub fn fit(&self, avalanche_sizes: &[f32], window: usize) -> Result<PowerLawMetrics> {
        let n = avalanche_sizes.len();
        if n < window {
            return Err(SocError::InsufficientData(
                format!("Need at least {} data points, got {}", window, n)
            ));
        }
        
        let mut power_law_fit = vec![0.5f32; n];
        let mut criticality_distance = vec![0.5f32; n];
        let mut tail_weight = vec![0.5f32; n];
        let mut distribution_entropy = vec![0.5f32; n];
        
        // Process each window
        for i in window..n {
            let window_data = &avalanche_sizes[(i - window)..i];
            
            // Skip if not enough unique values
            let unique_count = count_unique(window_data);
            if unique_count < 10 {
                continue;
            }
            
            // Fit power law to this window
            match self.fit_window(window_data) {
                Ok(fit_result) => {
                    power_law_fit[i] = fit_result.goodness_of_fit;
                    criticality_distance[i] = fit_result.criticality_distance;
                    tail_weight[i] = fit_result.tail_weight;
                    distribution_entropy[i] = fit_result.distribution_entropy;
                }
                Err(_) => {
                    // Keep default values on error
                }
            }
        }
        
        Ok(PowerLawMetrics {
            power_law_fit,
            criticality_distance,
            tail_weight,
            distribution_entropy,
        })
    }
    
    /// Fit power law to a single window of data
    fn fit_window(&self, data: &[f32]) -> Result<PowerLawFitResult> {
        // Remove zeros and sort data
        let mut sorted_data: Vec<f32> = data.iter()
            .filter(|&&x| x > 0.0)
            .copied()
            .collect();
        
        if sorted_data.len() < self.min_data_points {
            return Err(SocError::InsufficientData(
                "Not enough non-zero data points".to_string()
            ));
        }
        
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        // Find optimal xmin using Kolmogorov-Smirnov statistic
        let xmin = self.find_optimal_xmin(&sorted_data)?;
        
        // Fit power law above xmin
        let tail_data: Vec<f32> = sorted_data.iter()
            .filter(|&&x| x >= xmin)
            .copied()
            .collect();
        
        if tail_data.len() < 10 {
            return Err(SocError::InsufficientData(
                "Not enough data in tail".to_string()
            ));
        }
        
        // Estimate alpha using maximum likelihood
        let alpha = self.estimate_alpha(&tail_data, xmin);
        
        // Calculate goodness of fit using Kolmogorov-Smirnov test
        let goodness_of_fit = self.calculate_ks_statistic(&tail_data, xmin, alpha);
        
        // Calculate criticality distance (deviation from α = 1.5)
        let criticality_distance = ((alpha - 1.5).abs() / 1.5).min(1.0).max(0.0);
        let criticality_distance = 1.0 - criticality_distance; // Invert so 1 means critical
        
        // Calculate tail weight
        let tail_weight = (tail_data.len() as f32 / sorted_data.len() as f32).min(1.0);
        
        // Calculate distribution entropy
        let distribution_entropy = self.calculate_distribution_entropy(&sorted_data);
        
        Ok(PowerLawFitResult {
            alpha,
            xmin,
            goodness_of_fit: 1.0 - goodness_of_fit, // Convert KS to goodness
            criticality_distance,
            tail_weight,
            distribution_entropy,
        })
    }
    
    /// Find optimal xmin using Kolmogorov-Smirnov statistic
    fn find_optimal_xmin(&self, sorted_data: &[f32]) -> Result<f32> {
        let n = sorted_data.len();
        let search_range = (n as f32 * self.xmin_search_fraction) as usize;
        
        let mut best_xmin = sorted_data[0];
        let mut best_ks = f32::INFINITY;
        
        // Search over possible xmin values
        for i in 0..search_range.min(n - self.min_data_points) {
            let xmin_candidate = sorted_data[i];
            
            // Get tail data
            let tail_data: Vec<f32> = sorted_data[i..].to_vec();
            
            if tail_data.len() < self.min_data_points {
                continue;
            }
            
            // Estimate alpha for this xmin
            let alpha = self.estimate_alpha(&tail_data, xmin_candidate);
            
            // Calculate KS statistic
            let ks = self.calculate_ks_statistic(&tail_data, xmin_candidate, alpha);
            
            if ks < best_ks {
                best_ks = ks;
                best_xmin = xmin_candidate;
            }
        }
        
        Ok(best_xmin)
    }
    
    /// Estimate power law exponent using maximum likelihood
    fn estimate_alpha(&self, tail_data: &[f32], xmin: f32) -> f32 {
        let n = tail_data.len() as f32;
        let sum_log: f32 = tail_data.iter()
            .map(|&x| (x / xmin).ln())
            .sum();
        
        1.0 + n / sum_log
    }
    
    /// Calculate Kolmogorov-Smirnov statistic
    fn calculate_ks_statistic(&self, data: &[f32], xmin: f32, alpha: f32) -> f32 {
        let n = data.len();
        let mut max_distance = 0.0f32;
        
        for (i, &x) in data.iter().enumerate() {
            // Empirical CDF
            let p_empirical = (i + 1) as f32 / n as f32;
            
            // Theoretical CDF for power law
            let p_theoretical = 1.0 - (xmin / x).powf(alpha - 1.0);
            
            let distance = (p_empirical - p_theoretical).abs();
            max_distance = max_distance.max(distance);
        }
        
        max_distance
    }
    
    /// Calculate entropy of the distribution
    fn calculate_distribution_entropy(&self, data: &[f32]) -> f32 {
        // Create histogram
        let n_bins = 20.min(data.len() / 5);
        let (hist, _) = histogram(data, n_bins);
        
        // Normalize to get probabilities
        let total: f32 = hist.iter().sum();
        if total == 0.0 {
            return 0.5;
        }
        
        // Calculate entropy
        let entropy: f32 = hist.iter()
            .filter(|&&count| count > 0.0)
            .map(|&count| {
                let p = count / total;
                -p * p.ln()
            })
            .sum();
        
        // Normalize by log(n_bins)
        let max_entropy = (n_bins as f32).ln();
        if max_entropy > 0.0 {
            (entropy / max_entropy).min(1.0).max(0.0)
        } else {
            0.5
        }
    }
    
    /// Test power law vs alternative distributions
    pub fn compare_distributions(&self, data: &[f32]) -> DistributionComparison {
        let sorted_data: Vec<f32> = data.iter()
            .filter(|&&x| x > 0.0)
            .copied()
            .collect();
        
        if sorted_data.is_empty() {
            return DistributionComparison::default();
        }
        
        // Fit power law
        let power_law_ll = match self.fit_window(&sorted_data) {
            Ok(fit) => self.calculate_log_likelihood_power_law(&sorted_data, fit.xmin, fit.alpha),
            Err(_) => f32::NEG_INFINITY,
        };
        
        // Fit exponential
        let exp_rate = 1.0 / (sorted_data.iter().sum::<f32>() / sorted_data.len() as f32);
        let exp_ll = self.calculate_log_likelihood_exponential(&sorted_data, exp_rate);
        
        // Fit log-normal
        let (ln_mu, ln_sigma) = self.fit_lognormal(&sorted_data);
        let lognormal_ll = self.calculate_log_likelihood_lognormal(&sorted_data, ln_mu, ln_sigma);
        
        // Calculate likelihood ratios
        let power_law_vs_exp = power_law_ll - exp_ll;
        let power_law_vs_lognormal = power_law_ll - lognormal_ll;
        
        DistributionComparison {
            power_law_likelihood: power_law_ll,
            exponential_likelihood: exp_ll,
            lognormal_likelihood: lognormal_ll,
            power_law_vs_exponential_lr: power_law_vs_exp,
            power_law_vs_lognormal_lr: power_law_vs_lognormal,
        }
    }
    
    /// Calculate log-likelihood for power law
    fn calculate_log_likelihood_power_law(&self, data: &[f32], xmin: f32, alpha: f32) -> f32 {
        data.iter()
            .filter(|&&x| x >= xmin)
            .map(|&x| ((alpha - 1.0) / xmin) * (xmin / x).powf(alpha).ln())
            .sum()
    }
    
    /// Calculate log-likelihood for exponential distribution
    fn calculate_log_likelihood_exponential(&self, data: &[f32], rate: f32) -> f32 {
        data.iter()
            .map(|&x| rate.ln() - rate * x)
            .sum()
    }
    
    /// Calculate log-likelihood for log-normal distribution
    fn calculate_log_likelihood_lognormal(&self, data: &[f32], mu: f32, sigma: f32) -> f32 {
        let two_pi = 2.0 * std::f32::consts::PI;
        data.iter()
            .map(|&x| {
                -0.5 * (two_pi * sigma.powi(2)).ln() - x.ln() 
                    - 0.5 * ((x.ln() - mu) / sigma).powi(2)
            })
            .sum()
    }
    
    /// Fit log-normal distribution
    fn fit_lognormal(&self, data: &[f32]) -> (f32, f32) {
        let log_data: Vec<f32> = data.iter().map(|&x| x.ln()).collect();
        let n = log_data.len() as f32;
        
        let mu = log_data.iter().sum::<f32>() / n;
        let variance = log_data.iter()
            .map(|&x| (x - mu).powi(2))
            .sum::<f32>() / n;
        let sigma = variance.sqrt();
        
        (mu, sigma)
    }
}

impl Default for PowerLawFitter {
    fn default() -> Self {
        Self::new()
    }
}

/// Power law fit metrics
#[derive(Debug, Clone)]
pub struct PowerLawMetrics {
    pub power_law_fit: Vec<f32>,
    pub criticality_distance: Vec<f32>,
    pub tail_weight: Vec<f32>,
    pub distribution_entropy: Vec<f32>,
}

/// Result of fitting power law to data
#[derive(Debug, Clone)]
struct PowerLawFitResult {
    alpha: f32,
    xmin: f32,
    goodness_of_fit: f32,
    criticality_distance: f32,
    tail_weight: f32,
    distribution_entropy: f32,
}

/// Comparison of different distribution fits
#[derive(Debug, Clone, Default)]
pub struct DistributionComparison {
    pub power_law_likelihood: f32,
    pub exponential_likelihood: f32,
    pub lognormal_likelihood: f32,
    pub power_law_vs_exponential_lr: f32,
    pub power_law_vs_lognormal_lr: f32,
}

/// Count unique values in a slice
fn count_unique(data: &[f32]) -> usize {
    let mut unique = std::collections::HashSet::new();
    for &x in data {
        unique.insert(x.to_bits()); // Use bit representation for exact comparison
    }
    unique.len()
}

/// Create histogram from data
fn histogram(data: &[f32], n_bins: usize) -> (Vec<f32>, Vec<f32>) {
    if data.is_empty() || n_bins == 0 {
        return (vec![], vec![]);
    }
    
    let min_val = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max_val = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    
    if min_val >= max_val {
        return (vec![1.0], vec![min_val]);
    }
    
    let bin_width = (max_val - min_val) / n_bins as f32;
    let mut counts = vec![0.0f32; n_bins];
    let mut edges = Vec::with_capacity(n_bins + 1);
    
    // Generate bin edges
    for i in 0..=n_bins {
        edges.push(min_val + i as f32 * bin_width);
    }
    
    // Count values in each bin
    for &value in data {
        let bin_idx = ((value - min_val) / bin_width).floor() as usize;
        let bin_idx = bin_idx.min(n_bins - 1);
        counts[bin_idx] += 1.0;
    }
    
    (counts, edges)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::distributions::{Distribution, Uniform};
    
    #[test]
    fn test_power_law_fitting() {
        let fitter = PowerLawFitter::new();
        
        // Generate synthetic power law data
        let mut rng = rand::thread_rng();
        let uniform = Uniform::new(0.0, 1.0);
        let alpha = 2.5;
        let xmin = 1.0;
        
        let data: Vec<f32> = (0..1000)
            .map(|_| {
                let u: f32 = uniform.sample(&mut rng);
                xmin * (1.0 - u).powf(-1.0 / (alpha - 1.0))
            })
            .collect();
        
        let result = fitter.fit_window(&data).unwrap();
        
        // Check that estimated alpha is close to true value
        assert!((result.alpha - alpha).abs() < 0.5);
        
        // Check that goodness of fit is high
        assert!(result.goodness_of_fit > 0.7);
    }
    
    #[test]
    fn test_distribution_comparison() {
        let fitter = PowerLawFitter::new();
        
        // Generate exponential data
        let data: Vec<f32> = (0..500)
            .map(|i| (-0.1 * i as f32).exp())
            .collect();
        
        let comparison = fitter.compare_distributions(&data);
        
        // Exponential should have higher likelihood than power law
        assert!(comparison.exponential_likelihood > comparison.power_law_likelihood);
        assert!(comparison.power_law_vs_exponential_lr < 0.0);
    }
    
    #[test]
    fn test_histogram() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let (counts, edges) = histogram(&data, 5);
        
        assert_eq!(counts.len(), 5);
        assert_eq!(edges.len(), 6);
        assert_eq!(counts.iter().sum::<f32>(), data.len() as f32);
    }
}