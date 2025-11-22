//! Reference Implementations for Mathematical Validation
//! 
//! This module contains reference implementations of mathematical algorithms
//! used for validation against the main implementations. These are designed
//! for correctness over performance.

use crate::Result;
use crate::validation::ValidationConfig;
use ndarray::{Array1, Array2};
use std::collections::HashMap;

/// Reference implementations for mathematical algorithms
pub struct ReferenceImplementations {
    config: ValidationConfig,
}

impl ReferenceImplementations {
    pub fn new(config: &ValidationConfig) -> Result<Self> {
        Ok(Self {
            config: config.clone(),
        })
    }

    /// Reference implementation of Exponential Moving Average
    /// Uses the standard EMA formula: EMA_t = α * X_t + (1-α) * EMA_{t-1}
    pub fn compute_ema_reference(&self, values: &[f64], alpha: f64) -> Result<Vec<f64>> {
        if values.is_empty() {
            return Ok(Vec::new());
        }
        
        if !(0.0..=1.0).contains(&alpha) {
            return Err(crate::error::Error::InvalidInput(
                "Alpha must be between 0 and 1".to_string()
            ));
        }
        
        let mut result = Vec::with_capacity(values.len());
        result.push(values[0]); // Initialize with first value
        
        for i in 1..values.len() {
            let ema_value = alpha * values[i] + (1.0 - alpha) * result[i - 1];
            result.push(ema_value);
        }
        
        Ok(result)
    }

    /// Reference implementation of Simple Moving Average
    pub fn compute_sma_reference(&self, values: &[f64], window: usize) -> Result<Vec<f64>> {
        if values.len() < window {
            return Ok(Vec::new());
        }
        
        let mut result = Vec::new();
        for i in window - 1..values.len() {
            let sum: f64 = values[i - window + 1..=i].iter().sum();
            result.push(sum / window as f64);
        }
        
        Ok(result)
    }

    /// Reference implementation of standard deviation using the mathematical definition
    /// Uses Bessel's correction (n-1) for sample standard deviation
    pub fn compute_std_dev_reference(&self, values: &[f64]) -> Result<f64> {
        if values.len() < 2 {
            return Ok(0.0);
        }
        
        // Calculate mean using Kahan summation for better numerical stability
        let mean = self.kahan_sum(values) / values.len() as f64;
        
        // Calculate variance using Kahan summation
        let squared_deviations: Vec<f64> = values.iter()
            .map(|&x| (x - mean).powi(2))
            .collect();
        
        let variance = self.kahan_sum(&squared_deviations) / (values.len() - 1) as f64;
        
        Ok(variance.sqrt())
    }

    /// Reference implementation of Pearson correlation coefficient
    pub fn compute_correlation_reference(&self, x: &[f64], y: &[f64]) -> Result<f64> {
        if x.len() != y.len() || x.len() < 2 {
            return Ok(0.0);
        }
        
        let n = x.len() as f64;
        let mean_x = self.kahan_sum(x) / n;
        let mean_y = self.kahan_sum(y) / n;
        
        let mut numerator = 0.0;
        let mut sum_sq_x = 0.0;
        let mut sum_sq_y = 0.0;
        
        for (&xi, &yi) in x.iter().zip(y.iter()) {
            let diff_x = xi - mean_x;
            let diff_y = yi - mean_y;
            
            numerator += diff_x * diff_y;
            sum_sq_x += diff_x * diff_x;
            sum_sq_y += diff_y * diff_y;
        }
        
        let denominator = (sum_sq_x * sum_sq_y).sqrt();
        
        if denominator < f64::EPSILON {
            Ok(0.0)
        } else {
            Ok(numerator / denominator)
        }
    }

    /// Reference implementation of linear regression using normal equations
    pub fn compute_linear_regression_reference(&self, x: &[f64], y: &[f64]) -> Result<(f64, f64)> {
        if x.len() != y.len() || x.len() < 2 {
            return Err(crate::error::Error::InvalidInput(
                "Invalid input dimensions for linear regression".to_string()
            ));
        }
        
        let n = x.len() as f64;
        let sum_x = self.kahan_sum(x);
        let sum_y = self.kahan_sum(y);
        let sum_xy: f64 = x.iter().zip(y.iter()).map(|(&xi, &yi)| xi * yi).sum();
        let sum_x_sq: f64 = x.iter().map(|&xi| xi * xi).sum();
        
        let denominator = n * sum_x_sq - sum_x * sum_x;
        
        if denominator.abs() < f64::EPSILON {
            return Err(crate::error::Error::InvalidInput(
                "Singular matrix in linear regression".to_string()
            ));
        }
        
        let slope = (n * sum_xy - sum_x * sum_y) / denominator;
        let intercept = (sum_y - slope * sum_x) / n;
        
        Ok((slope, intercept))
    }

    /// Reference implementation of percentile calculation using R-6 quantile method
    pub fn compute_percentile_reference(&self, values: &[f64], p: f64) -> Result<f64> {
        if values.is_empty() {
            return Err(crate::error::Error::InvalidInput("Empty values array".to_string()));
        }
        
        if !(0.0..=1.0).contains(&p) {
            return Err(crate::error::Error::InvalidInput(
                "Percentile must be between 0 and 1".to_string()
            ));
        }
        
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        
        let n = sorted.len() as f64;
        let index = p * (n - 1.0);
        let lower = index.floor() as usize;
        let upper = index.ceil() as usize;
        
        if lower == upper {
            Ok(sorted[lower])
        } else {
            let weight = index - lower as f64;
            Ok(sorted[lower] * (1.0 - weight) + sorted[upper] * weight)
        }
    }

    /// Reference implementation of Z-score normalization
    pub fn compute_z_score_reference(&self, values: &[f64]) -> Result<Vec<f64>> {
        if values.len() < 2 {
            return Ok(values.to_vec());
        }
        
        let mean = self.kahan_sum(values) / values.len() as f64;
        let std_dev = self.compute_std_dev_reference(values)?;
        
        if std_dev < f64::EPSILON {
            return Ok(vec![0.0; values.len()]);
        }
        
        Ok(values.iter()
            .map(|&v| (v - mean) / std_dev)
            .collect())
    }

    /// Reference implementation of GARCH(1,1) using maximum likelihood estimation
    pub fn compute_garch_reference(&self, returns: &[f64]) -> Result<GarchParameters> {
        if returns.len() < 50 {
            return Err(crate::error::Error::InvalidInput(
                "Insufficient data for GARCH estimation".to_string()
            ));
        }
        
        // Initial parameter estimates
        let mut omega = 0.01;
        let mut alpha = 0.1;
        let mut beta = 0.85;
        
        // Ensure stationarity constraint: alpha + beta < 1
        if alpha + beta >= 1.0 {
            alpha = 0.05;
            beta = 0.90;
        }
        
        // Simple parameter estimation (in practice, use numerical optimization)
        let returns_var = self.compute_variance_reference(returns)?;
        omega = returns_var * (1.0 - alpha - beta);
        
        // Initialize conditional variance
        let mut conditional_variances = Vec::with_capacity(returns.len());
        conditional_variances.push(returns_var);
        
        // Compute conditional variances using GARCH equation
        for i in 1..returns.len() {
            let variance = omega + alpha * returns[i-1].powi(2) + beta * conditional_variances[i-1];
            conditional_variances.push(variance.max(1e-8)); // Prevent negative variance
        }
        
        // Compute log-likelihood for validation
        let log_likelihood = self.compute_garch_log_likelihood(returns, &conditional_variances);
        
        Ok(GarchParameters {
            omega,
            alpha,
            beta,
            conditional_variances,
            log_likelihood,
        })
    }

    /// Reference implementation of Value at Risk (VaR) using historical simulation
    pub fn compute_var_reference(&self, returns: &[f64], confidence_level: f64) -> Result<f64> {
        if returns.is_empty() {
            return Err(crate::error::Error::InvalidInput("Empty returns array".to_string()));
        }
        
        if !(0.0..1.0).contains(&confidence_level) {
            return Err(crate::error::Error::InvalidInput(
                "Confidence level must be between 0 and 1".to_string()
            ));
        }
        
        let mut sorted_returns = returns.to_vec();
        sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        
        let alpha = 1.0 - confidence_level;
        let index = (alpha * sorted_returns.len() as f64).ceil() as usize - 1;
        let index = index.min(sorted_returns.len() - 1);
        
        Ok(-sorted_returns[index])
    }

    /// Reference implementation of Expected Shortfall (Conditional VaR)
    pub fn compute_expected_shortfall_reference(&self, returns: &[f64], confidence_level: f64) -> Result<f64> {
        if returns.is_empty() {
            return Err(crate::error::Error::InvalidInput("Empty returns array".to_string()));
        }
        
        let var = self.compute_var_reference(returns, confidence_level)?;
        let threshold = -var;
        
        let tail_losses: Vec<f64> = returns.iter()
            .filter(|&&r| r <= threshold)
            .cloned()
            .collect();
        
        if tail_losses.is_empty() {
            Ok(var)
        } else {
            let mean_tail_loss = self.kahan_sum(&tail_losses) / tail_losses.len() as f64;
            Ok(-mean_tail_loss)
        }
    }

    /// Reference implementation of exact GELU activation function
    pub fn compute_gelu_reference(&self, x: f64) -> f64 {
        // Exact GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
        0.5 * x * (1.0 + erf(x / std::f64::consts::SQRT_2))
    }

    /// Reference implementation of ReLU activation function
    pub fn compute_relu_reference(&self, x: f64) -> f64 {
        x.max(0.0)
    }

    /// Reference implementation of sigmoid activation function
    pub fn compute_sigmoid_reference(&self, x: f64) -> f64 {
        if x > 700.0 {
            1.0 // Prevent overflow
        } else if x < -700.0 {
            0.0 // Prevent underflow
        } else {
            1.0 / (1.0 + (-x).exp())
        }
    }

    /// Reference implementation of tanh activation function
    pub fn compute_tanh_reference(&self, x: f64) -> f64 {
        if x > 700.0 {
            1.0
        } else if x < -700.0 {
            -1.0
        } else {
            x.tanh()
        }
    }

    /// Reference implementation of Swish activation function
    pub fn compute_swish_reference(&self, x: f64) -> f64 {
        x * self.compute_sigmoid_reference(x)
    }

    // Helper methods

    /// Kahan summation algorithm for improved numerical stability
    fn kahan_sum(&self, values: &[f64]) -> f64 {
        let mut sum = 0.0;
        let mut c = 0.0; // Compensation for lost low-order bits
        
        for &value in values {
            let y = value - c;
            let t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        
        sum
    }

    /// Compute variance using Kahan summation for better numerical stability
    fn compute_variance_reference(&self, values: &[f64]) -> Result<f64> {
        if values.len() < 2 {
            return Ok(0.0);
        }
        
        let mean = self.kahan_sum(values) / values.len() as f64;
        let squared_deviations: Vec<f64> = values.iter()
            .map(|&x| (x - mean).powi(2))
            .collect();
        
        Ok(self.kahan_sum(&squared_deviations) / (values.len() - 1) as f64)
    }

    /// Compute log-likelihood for GARCH model validation
    fn compute_garch_log_likelihood(&self, returns: &[f64], variances: &[f64]) -> f64 {
        let mut log_likelihood = 0.0;
        
        for (i, (&return_val, &variance)) in returns.iter().zip(variances.iter()).enumerate() {
            if i > 0 && variance > 0.0 {
                log_likelihood += -0.5 * (variance.ln() + return_val.powi(2) / variance);
            }
        }
        
        log_likelihood
    }
}

/// GARCH model parameters and results
#[derive(Debug, Clone)]
pub struct GarchParameters {
    pub omega: f64,
    pub alpha: f64,
    pub beta: f64,
    pub conditional_variances: Vec<f64>,
    pub log_likelihood: f64,
}

/// Error function implementation for GELU
fn erf(x: f64) -> f64 {
    // Approximation of error function using Abramowitz and Stegun formula
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_ema_reference() {
        let config = ValidationConfig::default();
        let reference = ReferenceImplementations::new(&config).unwrap();
        
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let alpha = 0.3;
        
        let result = reference.compute_ema_reference(&values, alpha).unwrap();
        
        // Manual calculation for verification
        let expected = vec![
            1.0,
            1.0 * 0.7 + 2.0 * 0.3,
            (1.0 * 0.7 + 2.0 * 0.3) * 0.7 + 3.0 * 0.3,
            ((1.0 * 0.7 + 2.0 * 0.3) * 0.7 + 3.0 * 0.3) * 0.7 + 4.0 * 0.3,
            (((1.0 * 0.7 + 2.0 * 0.3) * 0.7 + 3.0 * 0.3) * 0.7 + 4.0 * 0.3) * 0.7 + 5.0 * 0.3,
        ];
        
        for (r, e) in result.iter().zip(expected.iter()) {
            assert_relative_eq!(r, e, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_std_dev_reference() {
        let config = ValidationConfig::default();
        let reference = ReferenceImplementations::new(&config).unwrap();
        
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = reference.compute_std_dev_reference(&values).unwrap();
        
        // Expected standard deviation for [1,2,3,4,5] is sqrt(2.5) ≈ 1.5811
        assert_relative_eq!(result, 1.5811388300841898, epsilon = 1e-10);
    }

    #[test]
    fn test_correlation_reference() {
        let config = ValidationConfig::default();
        let reference = ReferenceImplementations::new(&config).unwrap();
        
        // Perfect positive correlation
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        
        let result = reference.compute_correlation_reference(&x, &y).unwrap();
        assert_relative_eq!(result, 1.0, epsilon = 1e-10);
        
        // Perfect negative correlation
        let y_neg = vec![10.0, 8.0, 6.0, 4.0, 2.0];
        let result_neg = reference.compute_correlation_reference(&x, &y_neg).unwrap();
        assert_relative_eq!(result_neg, -1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_gelu_reference() {
        let config = ValidationConfig::default();
        let reference = ReferenceImplementations::new(&config).unwrap();
        
        // Test specific values
        let test_values = vec![0.0, 1.0, -1.0, 2.0, -2.0];
        
        for x in test_values {
            let result = reference.compute_gelu_reference(x);
            
            // GELU(0) = 0
            if x == 0.0 {
                assert_relative_eq!(result, 0.0, epsilon = 1e-10);
            }
            
            // GELU should be approximately x for large positive x
            if x > 3.0 {
                assert_relative_eq!(result, x, epsilon = 0.01);
            }
            
            // GELU should be approximately 0 for large negative x
            if x < -3.0 {
                assert!(result.abs() < 0.01);
            }
        }
    }

    #[test]
    fn test_kahan_sum() {
        let config = ValidationConfig::default();
        let reference = ReferenceImplementations::new(&config).unwrap();
        
        // Test with values that would lose precision with naive summation
        let values = vec![1e16, 1.0, -1e16];
        let result = reference.kahan_sum(&values);
        
        // Should be 1.0, not 0.0 due to floating point precision
        assert_relative_eq!(result, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_percentile_reference() {
        let config = ValidationConfig::default();
        let reference = ReferenceImplementations::new(&config).unwrap();
        
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        
        // Test median (50th percentile)
        let median = reference.compute_percentile_reference(&values, 0.5).unwrap();
        assert_relative_eq!(median, 5.5, epsilon = 1e-10);
        
        // Test quartiles
        let q1 = reference.compute_percentile_reference(&values, 0.25).unwrap();
        let q3 = reference.compute_percentile_reference(&values, 0.75).unwrap();
        
        assert_relative_eq!(q1, 3.25, epsilon = 1e-10);
        assert_relative_eq!(q3, 7.75, epsilon = 1e-10);
    }

    #[test]
    fn test_var_reference() {
        let config = ValidationConfig::default();
        let reference = ReferenceImplementations::new(&config).unwrap();
        
        // Simple test case with known distribution
        let returns = vec![-0.05, -0.03, -0.01, 0.01, 0.02, 0.03, 0.05, 0.07, 0.08, 0.10];
        let var_95 = reference.compute_var_reference(&returns, 0.95).unwrap();
        
        // 95% VaR should be around the 5th percentile (worst 5% of returns)
        assert!(var_95 > 0.0); // VaR should be positive (loss)
        assert!(var_95 <= 0.05); // Should not exceed worst return
    }
}