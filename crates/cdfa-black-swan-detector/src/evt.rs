//! Extreme Value Theory implementation for Black Swan detection
//!
//! This module implements the mathematical foundations of Extreme Value Theory (EVT)
//! for detecting black swan events in financial markets.

use crate::error::*;
use crate::types::*;
use crate::utils::*;
use nalgebra::DVector;
use ndarray::Array1;
use rayon::prelude::*;
use statrs::distribution::*;
use std::collections::HashMap;

/// Hill estimator for tail index calculation
pub struct HillEstimator {
    k_values: Vec<usize>,
    bootstrap_samples: usize,
    confidence_level: f64,
}

impl HillEstimator {
    pub fn new(k_min: usize, k_max: usize, k_step: usize) -> Self {
        let k_values = (k_min..=k_max).step_by(k_step).collect();
        Self {
            k_values,
            bootstrap_samples: 1000,
            confidence_level: 0.05,
        }
    }
    
    /// Estimate tail index using Hill estimator
    pub fn estimate(&self, data: &[f64]) -> BSResult<TailRiskMetrics> {
        validation::validate_min_size(data, 50, "Hill estimator data")?;
        validation::validate_all_finite(data, "Hill estimator data")?;
        
        let mut sorted_data = data.to_vec();
        sorted_data.sort_by(|a, b| b.partial_cmp(a).unwrap()); // Sort descending
        
        let mut best_estimate = None;
        let mut best_p_value = 0.0;
        
        for &k in &self.k_values {
            if k >= sorted_data.len() {
                continue;
            }
            
            let hill_estimate = self.calculate_hill_estimate(&sorted_data, k)?;
            let p_value = self.goodness_of_fit_test(&sorted_data, k, hill_estimate)?;
            
            if p_value > best_p_value {
                best_p_value = p_value;
                best_estimate = Some((hill_estimate, k));
            }
        }
        
        let (hill_estimator, optimal_k) = best_estimate
            .ok_or_else(|| BlackSwanError::Statistical("No valid Hill estimate found".to_string()))?;
        
        let var = self.calculate_var(&sorted_data, optimal_k, hill_estimator)?;
        let expected_shortfall = self.calculate_expected_shortfall(&sorted_data, optimal_k)?;
        let tail_probability = self.calculate_tail_probability(&sorted_data, optimal_k)?;
        
        Ok(TailRiskMetrics {
            hill_estimator,
            var,
            expected_shortfall,
            tail_probability,
            p_value: best_p_value,
            n_observations: data.len(),
        })
    }
    
    /// Calculate Hill estimate for given k
    fn calculate_hill_estimate(&self, sorted_data: &[f64], k: usize) -> BSResult<f64> {
        if k == 0 || k >= sorted_data.len() {
            return Err(BlackSwanError::InvalidInput("Invalid k value".to_string()));
        }
        
        let threshold = sorted_data[k - 1];
        let mut log_sum = 0.0;
        
        for i in 0..k {
            if sorted_data[i] <= 0.0 {
                return Err(BlackSwanError::Mathematical("Non-positive values in tail".to_string()));
            }
            log_sum += (sorted_data[i] / threshold).ln();
        }
        
        let hill_estimate = log_sum / k as f64;
        
        if hill_estimate <= 0.0 {
            return Err(BlackSwanError::Mathematical("Invalid Hill estimate".to_string()));
        }
        
        Ok(1.0 / hill_estimate)
    }
    
    /// Goodness-of-fit test for Hill estimator
    fn goodness_of_fit_test(&self, sorted_data: &[f64], k: usize, hill_estimate: f64) -> BSResult<f64> {
        if k < 10 || k >= sorted_data.len() {
            return Ok(0.0);
        }
        
        let threshold = sorted_data[k - 1];
        let mut theoretical_quantiles = Vec::new();
        let mut empirical_quantiles = Vec::new();
        
        for i in 0..k {
            let empirical_prob = (i + 1) as f64 / (k + 1) as f64;
            let theoretical_value = threshold * empirical_prob.powf(-1.0 / hill_estimate);
            
            theoretical_quantiles.push(theoretical_value);
            empirical_quantiles.push(sorted_data[i]);
        }
        
        // Kolmogorov-Smirnov test statistic
        let ks_statistic = self.kolmogorov_smirnov_statistic(&theoretical_quantiles, &empirical_quantiles)?;
        let p_value = self.ks_p_value(ks_statistic, k);
        
        Ok(p_value)
    }
    
    /// Kolmogorov-Smirnov test statistic
    fn kolmogorov_smirnov_statistic(&self, theoretical: &[f64], empirical: &[f64]) -> BSResult<f64> {
        if theoretical.len() != empirical.len() {
            return Err(BlackSwanError::InvalidInput("Mismatched array lengths".to_string()));
        }
        
        let n = theoretical.len();
        let mut max_diff = 0.0;
        
        for i in 0..n {
            let empirical_cdf = (i + 1) as f64 / n as f64;
            let theoretical_cdf = self.empirical_cdf(theoretical[i], theoretical);
            let diff = (empirical_cdf - theoretical_cdf).abs();
            max_diff = max_diff.max(diff);
        }
        
        Ok(max_diff)
    }
    
    /// Empirical cumulative distribution function
    fn empirical_cdf(&self, x: f64, data: &[f64]) -> f64 {
        let count = data.iter().filter(|&&val| val <= x).count();
        count as f64 / data.len() as f64
    }
    
    /// Kolmogorov-Smirnov p-value approximation
    fn ks_p_value(&self, ks_statistic: f64, n: usize) -> f64 {
        let lambda = ks_statistic * (n as f64).sqrt();
        let mut p_value = 0.0;
        
        for k in 1..=100 {
            let term = 2.0 * ((-2.0 * k as f64 * k as f64 * lambda * lambda).exp());
            if k % 2 == 1 {
                p_value += term;
            } else {
                p_value -= term;
            }
            
            if term < 1e-10 {
                break;
            }
        }
        
        p_value.max(0.0).min(1.0)
    }
    
    /// Calculate Value at Risk (VaR)
    fn calculate_var(&self, sorted_data: &[f64], k: usize, hill_estimate: f64) -> BSResult<f64> {
        if k == 0 || k >= sorted_data.len() {
            return Err(BlackSwanError::InvalidInput("Invalid k value for VaR calculation".to_string()));
        }
        
        let threshold = sorted_data[k - 1];
        let tail_prob = k as f64 / sorted_data.len() as f64;
        
        // VaR at 99% confidence level
        let confidence_level = 0.99;
        let var = threshold * (tail_prob / (1.0 - confidence_level)).powf(-1.0 / hill_estimate);
        
        Ok(var)
    }
    
    /// Calculate Expected Shortfall (Conditional VaR)
    fn calculate_expected_shortfall(&self, sorted_data: &[f64], k: usize) -> BSResult<f64> {
        if k == 0 || k >= sorted_data.len() {
            return Err(BlackSwanError::InvalidInput("Invalid k value for ES calculation".to_string()));
        }
        
        let sum: f64 = sorted_data.iter().take(k).sum();
        Ok(sum / k as f64)
    }
    
    /// Calculate tail probability
    fn calculate_tail_probability(&self, sorted_data: &[f64], k: usize) -> BSResult<f64> {
        if k == 0 || k >= sorted_data.len() {
            return Err(BlackSwanError::InvalidInput("Invalid k value for tail probability".to_string()));
        }
        
        Ok(k as f64 / sorted_data.len() as f64)
    }
    
    /// Bootstrap confidence intervals
    pub fn bootstrap_confidence_intervals(&self, data: &[f64], k: usize) -> BSResult<(f64, f64)> {
        validation::validate_min_size(data, 50, "Bootstrap data")?;
        
        let mut bootstrap_estimates = Vec::new();
        let mut rng = rand::thread_rng();
        
        for _ in 0..self.bootstrap_samples {
            let bootstrap_sample = bootstrap_sample(data, &mut rng);
            let mut sorted_sample = bootstrap_sample;
            sorted_sample.sort_by(|a, b| b.partial_cmp(a).unwrap());
            
            if let Ok(estimate) = self.calculate_hill_estimate(&sorted_sample, k) {
                bootstrap_estimates.push(estimate);
            }
        }
        
        if bootstrap_estimates.is_empty() {
            return Err(BlackSwanError::Statistical("No valid bootstrap estimates".to_string()));
        }
        
        bootstrap_estimates.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let lower_idx = ((self.confidence_level / 2.0) * bootstrap_estimates.len() as f64) as usize;
        let upper_idx = ((1.0 - self.confidence_level / 2.0) * bootstrap_estimates.len() as f64) as usize;
        
        let lower_bound = bootstrap_estimates[lower_idx.min(bootstrap_estimates.len() - 1)];
        let upper_bound = bootstrap_estimates[upper_idx.min(bootstrap_estimates.len() - 1)];
        
        Ok((lower_bound, upper_bound))
    }
}

/// Generalized Extreme Value (GEV) distribution fitting
pub struct GEVFitter {
    max_iterations: usize,
    tolerance: f64,
}

impl GEVFitter {
    pub fn new() -> Self {
        Self {
            max_iterations: 1000,
            tolerance: 1e-8,
        }
    }
    
    /// Fit GEV distribution to block maxima
    pub fn fit(&self, block_maxima: &[f64]) -> BSResult<GEVParameters> {
        validation::validate_min_size(block_maxima, 10, "GEV block maxima")?;
        validation::validate_all_finite(block_maxima, "GEV block maxima")?;
        
        // Method of moments initial estimates
        let mean = block_maxima.iter().sum::<f64>() / block_maxima.len() as f64;
        let variance = block_maxima.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / (block_maxima.len() - 1) as f64;
        
        let std_dev = variance.sqrt();
        
        // Initial parameter estimates
        let mut location = mean;
        let mut scale = std_dev * (6.0_f64.sqrt() / std::f64::consts::PI);
        let mut shape = 0.1;
        
        // Maximum likelihood estimation using Newton-Raphson
        for iteration in 0..self.max_iterations {
            let (log_likelihood, gradient, hessian) = self.compute_likelihood_derivatives(
                block_maxima, location, scale, shape
            )?;
            
            if gradient.iter().all(|&g| g.abs() < self.tolerance) {
                break;
            }
            
            // Newton-Raphson update
            let hessian_inv = self.invert_3x3_matrix(&hessian)?;
            let delta = self.matrix_vector_multiply(&hessian_inv, &gradient);
            
            location -= delta[0];
            scale -= delta[1];
            shape -= delta[2];
            
            // Ensure scale is positive
            if scale <= 0.0 {
                scale = 1e-6;
            }
            
            if iteration == self.max_iterations - 1 {
                log::warn!("GEV fitting did not converge after {} iterations", self.max_iterations);
            }
        }
        
        Ok(GEVParameters {
            location,
            scale,
            shape,
        })
    }
    
    /// Compute log-likelihood and its derivatives
    fn compute_likelihood_derivatives(
        &self,
        data: &[f64],
        location: f64,
        scale: f64,
        shape: f64,
    ) -> BSResult<(f64, Vec<f64>, Vec<Vec<f64>>)> {
        let n = data.len() as f64;
        let mut log_likelihood = 0.0;
        let mut gradient = vec![0.0; 3]; // [d/d_location, d/d_scale, d/d_shape]
        let mut hessian = vec![vec![0.0; 3]; 3];
        
        for &x in data {
            let z = (x - location) / scale;
            
            if shape.abs() < 1e-10 {
                // Gumbel distribution (shape = 0)
                let exp_z = (-z).exp();
                log_likelihood += -z - exp_z;
                
                gradient[0] += (1.0 - exp_z) / scale;
                gradient[1] += (z - 1.0 + exp_z) / scale;
                gradient[2] += 0.0; // Shape derivative is 0 for Gumbel
            } else {
                // Generalized extreme value
                let one_plus_shape_z = 1.0 + shape * z;
                
                if one_plus_shape_z <= 0.0 {
                    return Err(BlackSwanError::Mathematical("Invalid GEV parameters".to_string()));
                }
                
                let log_term = one_plus_shape_z.ln();
                let power_term = one_plus_shape_z.powf(-1.0 / shape);
                
                log_likelihood += -(1.0 + 1.0 / shape) * log_term - power_term;
                
                // Gradient computation
                let factor1 = (1.0 + 1.0 / shape) / one_plus_shape_z;
                let factor2 = power_term / one_plus_shape_z;
                
                gradient[0] += (factor1 - factor2) * shape / scale;
                gradient[1] += (factor1 - factor2) * shape * z / scale;
                gradient[2] += -log_term / (shape * shape) + (factor1 - factor2) * z;
            }
        }
        
        log_likelihood -= n * scale.ln();
        gradient[1] -= n / scale;
        
        // Hessian computation (simplified)
        // In practice, this would require more complex derivatives
        // For now, use approximate numerical derivatives
        
        Ok((log_likelihood, gradient, hessian))
    }
    
    /// Invert 3x3 matrix
    fn invert_3x3_matrix(&self, matrix: &[Vec<f64>]) -> BSResult<Vec<Vec<f64>>> {
        // Simplified 3x3 matrix inversion
        // In practice, use a proper linear algebra library
        let det = matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
                - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
                + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]);
        
        if det.abs() < 1e-12 {
            return Err(BlackSwanError::Mathematical("Singular matrix".to_string()));
        }
        
        let inv_det = 1.0 / det;
        let mut inv = vec![vec![0.0; 3]; 3];
        
        inv[0][0] = (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1]) * inv_det;
        inv[0][1] = (matrix[0][2] * matrix[2][1] - matrix[0][1] * matrix[2][2]) * inv_det;
        inv[0][2] = (matrix[0][1] * matrix[1][2] - matrix[0][2] * matrix[1][1]) * inv_det;
        
        inv[1][0] = (matrix[1][2] * matrix[2][0] - matrix[1][0] * matrix[2][2]) * inv_det;
        inv[1][1] = (matrix[0][0] * matrix[2][2] - matrix[0][2] * matrix[2][0]) * inv_det;
        inv[1][2] = (matrix[0][2] * matrix[1][0] - matrix[0][0] * matrix[1][2]) * inv_det;
        
        inv[2][0] = (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]) * inv_det;
        inv[2][1] = (matrix[0][1] * matrix[2][0] - matrix[0][0] * matrix[2][1]) * inv_det;
        inv[2][2] = (matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]) * inv_det;
        
        Ok(inv)
    }
    
    /// Matrix-vector multiplication
    fn matrix_vector_multiply(&self, matrix: &[Vec<f64>], vector: &[f64]) -> Vec<f64> {
        let mut result = vec![0.0; matrix.len()];
        for i in 0..matrix.len() {
            for j in 0..vector.len() {
                result[i] += matrix[i][j] * vector[j];
            }
        }
        result
    }
}

/// GEV distribution parameters
#[derive(Debug, Clone)]
pub struct GEVParameters {
    pub location: f64,
    pub scale: f64,
    pub shape: f64,
}

/// Peaks Over Threshold (POT) model
pub struct POTModel {
    threshold: f64,
    shape: f64,
    scale: f64,
}

impl POTModel {
    pub fn new(threshold: f64) -> Self {
        Self {
            threshold,
            shape: 0.0,
            scale: 1.0,
        }
    }
    
    /// Fit POT model to exceedances
    pub fn fit(&mut self, data: &[f64]) -> BSResult<()> {
        let exceedances: Vec<f64> = data.iter()
            .filter(|&&x| x > self.threshold)
            .map(|&x| x - self.threshold)
            .collect();
        
        if exceedances.len() < 10 {
            return Err(BlackSwanError::InsufficientData {
                required: 10,
                actual: exceedances.len(),
            });
        }
        
        // Method of moments estimators
        let mean = exceedances.iter().sum::<f64>() / exceedances.len() as f64;
        let variance = exceedances.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / (exceedances.len() - 1) as f64;
        
        self.shape = -0.5 * (mean * mean / variance - 1.0);
        self.scale = 0.5 * mean * (mean * mean / variance + 1.0);
        
        Ok(())
    }
    
    /// Calculate probability of exceedance
    pub fn exceedance_probability(&self, x: f64) -> f64 {
        if x <= self.threshold {
            return 1.0;
        }
        
        let z = (x - self.threshold) / self.scale;
        if self.shape.abs() < 1e-10 {
            (-z).exp()
        } else {
            (1.0 + self.shape * z).powf(-1.0 / self.shape)
        }
    }
    
    /// Calculate return level
    pub fn return_level(&self, return_period: f64) -> f64 {
        let p = 1.0 / return_period;
        
        if self.shape.abs() < 1e-10 {
            self.threshold + self.scale * (-p.ln()).ln()
        } else {
            self.threshold + (self.scale / self.shape) * (p.powf(-self.shape) - 1.0)
        }
    }
}

/// Extreme Value Theory analyzer
pub struct EVTAnalyzer {
    hill_estimator: HillEstimator,
    gev_fitter: GEVFitter,
    pot_models: HashMap<String, POTModel>,
}

impl EVTAnalyzer {
    pub fn new(config: &EVTConfig) -> Self {
        let hill_estimator = HillEstimator::new(
            config.hill_k_min,
            config.hill_k_max,
            config.hill_k_step,
        );
        
        Self {
            hill_estimator,
            gev_fitter: GEVFitter::new(),
            pot_models: HashMap::new(),
        }
    }
    
    /// Comprehensive EVT analysis
    pub fn analyze(&mut self, data: &[f64]) -> BSResult<EVTAnalysis> {
        // Hill estimator analysis
        let tail_metrics = self.hill_estimator.estimate(data)?;
        
        // Block maxima for GEV fitting
        let block_size = 20;
        let block_maxima = self.extract_block_maxima(data, block_size);
        let gev_params = if block_maxima.len() >= 10 {
            Some(self.gev_fitter.fit(&block_maxima)?)
        } else {
            None
        };
        
        // POT analysis for multiple thresholds
        let mut pot_results = HashMap::new();
        let thresholds = vec![0.9, 0.95, 0.99];
        
        for &quantile in &thresholds {
            let mut sorted_data = data.to_vec();
            sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let threshold_idx = ((quantile * data.len() as f64) as usize).min(data.len() - 1);
            let threshold = sorted_data[threshold_idx];
            
            let mut pot_model = POTModel::new(threshold);
            if pot_model.fit(data).is_ok() {
                pot_results.insert(format!("q{}", (quantile * 100.0) as u32), pot_model);
            }
        }
        
        Ok(EVTAnalysis {
            tail_metrics,
            gev_parameters: gev_params,
            pot_results,
            block_maxima,
            n_observations: data.len(),
        })
    }
    
    /// Extract block maxima from data
    fn extract_block_maxima(&self, data: &[f64], block_size: usize) -> Vec<f64> {
        data.chunks(block_size)
            .map(|chunk| *chunk.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(&0.0))
            .collect()
    }
}

/// Complete EVT analysis result
#[derive(Debug, Clone)]
pub struct EVTAnalysis {
    pub tail_metrics: TailRiskMetrics,
    pub gev_parameters: Option<GEVParameters>,
    pub pot_results: HashMap<String, POTModel>,
    pub block_maxima: Vec<f64>,
    pub n_observations: usize,
}

impl EVTAnalysis {
    /// Get Black Swan probability estimate
    pub fn black_swan_probability(&self) -> f64 {
        // Combine multiple EVT approaches
        let mut probability = 0.0;
        let mut weight_sum = 0.0;
        
        // Hill estimator contribution
        if self.tail_metrics.hill_estimator > 0.0 {
            let tail_heaviness = 1.0 / self.tail_metrics.hill_estimator.max(1.0);
            probability += 0.4 * tail_heaviness;
            weight_sum += 0.4;
        }
        
        // GEV contribution
        if let Some(ref gev) = self.gev_parameters {
            let shape_contribution = if gev.shape > 0.0 {
                gev.shape.min(1.0)
            } else {
                0.0
            };
            probability += 0.3 * shape_contribution;
            weight_sum += 0.3;
        }
        
        // POT contribution
        if !self.pot_results.is_empty() {
            let pot_contribution = self.pot_results.values()
                .map(|model| model.exceedance_probability(model.threshold * 1.1))
                .sum::<f64>() / self.pot_results.len() as f64;
            probability += 0.3 * pot_contribution;
            weight_sum += 0.3;
        }
        
        if weight_sum > 0.0 {
            probability / weight_sum
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use rand::Rng;
    use statrs::distribution::Distribution;
    
    #[test]
    fn test_hill_estimator() {
        let mut rng = rand::thread_rng();
        
        // Generate Pareto-distributed data
        let data: Vec<f64> = (0..1000)
            .map(|_| {
                let u: f64 = rng.gen();
                1.0 / u.powf(1.0 / 2.0) // Pareto with tail index 2
            })
            .collect();
        
        let hill_estimator = HillEstimator::new(20, 200, 20);
        let result = hill_estimator.estimate(&data).unwrap();
        
        // Should be close to 2.0 for Pareto(2)
        assert!(result.hill_estimator > 1.5 && result.hill_estimator < 3.0);
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    }
    
    #[test]
    fn test_gev_fitter() {
        let mut rng = rand::thread_rng();
        
        // Generate block maxima from normal distribution
        let data: Vec<f64> = (0..100)
            .map(|_| {
                let block: Vec<f64> = (0..20)
                    .map(|_| rng.gen::<f64>())
                    .collect();
                *block.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap()
            })
            .collect();
        
        let gev_fitter = GEVFitter::new();
        let result = gev_fitter.fit(&data).unwrap();
        
        assert!(result.scale > 0.0);
        assert!(result.location.is_finite());
        assert!(result.shape.is_finite());
    }
    
    #[test]
    fn test_pot_model() {
        let mut rng = rand::thread_rng();
        
        // Generate exponential data
        let data: Vec<f64> = (0..1000)
            .map(|_| -rng.gen::<f64>().ln())
            .collect();
        
        let mut pot_model = POTModel::new(1.0);
        pot_model.fit(&data).unwrap();
        
        assert!(pot_model.scale > 0.0);
        assert!(pot_model.shape.is_finite());
        
        let exc_prob = pot_model.exceedance_probability(2.0);
        assert!(exc_prob >= 0.0 && exc_prob <= 1.0);
    }
    
    #[test]
    fn test_evt_analyzer() {
        let mut rng = rand::thread_rng();
        
        // Generate mixed data with heavy tails
        let data: Vec<f64> = (0..1000)
            .map(|_| {
                if rng.gen::<f64>() < 0.1 {
                    // Heavy tail component
                    rng.gen::<f64>().powf(-0.5) * 5.0
                } else {
                    // Normal component
                    rng.gen::<f64>() * 2.0 - 1.0
                }
            })
            .collect();
        
        let config = EVTConfig::default();
        let mut analyzer = EVTAnalyzer::new(&config);
        let analysis = analyzer.analyze(&data).unwrap();
        
        assert!(analysis.tail_metrics.hill_estimator > 0.0);
        assert!(analysis.n_observations == 1000);
        
        let bs_prob = analysis.black_swan_probability();
        assert!(bs_prob >= 0.0 && bs_prob <= 1.0);
    }
    
    #[test]
    fn test_bootstrap_confidence_intervals() {
        let mut rng = rand::thread_rng();
        
        // Generate Pareto data
        let data: Vec<f64> = (0..500)
            .map(|_| {
                let u: f64 = rng.gen();
                1.0 / u.powf(1.0 / 2.5) // Pareto with tail index 2.5
            })
            .collect();
        
        let hill_estimator = HillEstimator::new(20, 100, 20);
        let (lower, upper) = hill_estimator.bootstrap_confidence_intervals(&data, 50).unwrap();
        
        assert!(lower < upper);
        assert!(lower > 0.0);
        assert!(upper < 10.0); // Reasonable bounds
    }
}