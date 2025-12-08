//! Main SOC analyzer implementation

use crate::{Result, SOCError, SOCParameters, SOCResult, SOCRegime};
use crate::entropy::{sample_entropy, entropy_rate};
use crate::regimes::classify_regime;
use ndarray::{Array1, ArrayView1};
use std::time::Instant;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// High-performance Self-Organized Criticality analyzer
pub struct SOCAnalyzer {
    params: SOCParameters,
    
    #[cfg(feature = "cache_aligned")]
    cache_buffer: aligned_vec::AVec<f64>,
}

impl SOCAnalyzer {
    /// Create new SOC analyzer with parameters
    pub fn new(params: SOCParameters) -> Self {
        Self {
            params,
            #[cfg(feature = "cache_aligned")]
            cache_buffer: aligned_vec::AVec::with_capacity(64, 1024),
        }
    }
    
    /// Create SOC analyzer with default parameters
    pub fn default() -> Self {
        Self::new(SOCParameters::default())
    }
    
    /// Perform complete SOC analysis on time series data
    pub fn analyze(&self, data: &Array1<f64>) -> Result<SOCResult> {
        let start_time = Instant::now();
        
        if data.len() < self.params.sample_entropy_min_points {
            return Err(SOCError::InsufficientData {
                required: self.params.sample_entropy_min_points,
                actual: data.len(),
            });
        }
        
        let mut result = SOCResult::new();
        
        // Calculate sample entropy
        let entropy_start = Instant::now();
        result.sample_entropy = sample_entropy(
            data.view(),
            self.params.sample_entropy_m,
            self.params.sample_entropy_r,
        )?;
        let entropy_time = entropy_start.elapsed();
        
        // Calculate entropy rate
        let rate_start = Instant::now();
        result.entropy_rate = entropy_rate(
            data.view(),
            self.params.entropy_rate_lag,
            self.params.n_bins,
        )?;
        let rate_time = rate_start.elapsed();
        
        // Calculate complexity measure
        result.complexity_measure = self.calculate_complexity(data.view())?;
        
        // Detect avalanche events
        let avalanche_start = Instant::now();
        result.avalanche_events = self.detect_avalanches(data.view())?;
        result.total_avalanche_magnitude = result.avalanche_events
            .iter()
            .map(|a| a.magnitude)
            .sum();
        result.avalanche_frequency = result.avalanche_events.len() as f64 / data.len() as f64;
        let avalanche_time = avalanche_start.elapsed();
        
        // Calculate equilibrium and fragility scores
        result.equilibrium_score = self.calculate_equilibrium_score(data.view())?;
        result.fragility_score = self.calculate_fragility_score(data.view())?;
        
        // Classify SOC regime
        let regime_start = Instant::now();
        let (regime, confidence) = classify_regime(
            result.complexity_measure,
            result.equilibrium_score,
            result.fragility_score,
            result.sample_entropy,
            &self.params,
        )?;
        result.regime = regime;
        result.regime_confidence = confidence;
        let regime_time = regime_start.elapsed();
        
        result.computation_time_ns = start_time.elapsed().as_nanos() as u64;
        
        Ok(result)
    }
    
    /// Calculate complexity measure from multiple entropy-based metrics
    fn calculate_complexity(&self, data: ArrayView1<f64>) -> Result<f64> {
        // Use multiple scales for complexity estimation
        let mut complexities = Vec::with_capacity(3);
        
        // Short-term complexity (m=1)
        if data.len() >= 10 {
            let short_entropy = sample_entropy(data, 1, self.params.sample_entropy_r * 0.5)?;
            complexities.push(short_entropy);
        }
        
        // Medium-term complexity (m=2, default)
        complexities.push(sample_entropy(data, self.params.sample_entropy_m, self.params.sample_entropy_r)?);
        
        // Long-term complexity (m=3)
        if data.len() >= self.params.sample_entropy_min_points + 10 {
            let long_entropy = sample_entropy(data, 3, self.params.sample_entropy_r * 1.5)?;
            complexities.push(long_entropy);
        }
        
        // Weighted average with emphasis on medium-term
        let weights: Vec<f64> = if complexities.len() == 3 {
            vec![0.2, 0.6, 0.2]
        } else if complexities.len() == 2 {
            vec![0.4, 0.6]
        } else {
            vec![1.0]
        };
        
        let complexity = complexities
            .iter()
            .zip(weights.iter())
            .map(|(c, w)| c * w)
            .sum::<f64>();
            
        Ok(complexity)
    }
    
    /// Calculate equilibrium score based on autocorrelation and stability
    fn calculate_equilibrium_score(&self, data: ArrayView1<f64>) -> Result<f64> {
        let n = data.len();
        if n < 4 {
            return Ok(0.0);
        }
        
        // Calculate mean and variance
        let mean = data.mean().unwrap_or(0.0);
        let variance = data.var(0.0);
        
        if variance < 1e-12 {
            return Ok(1.0); // Perfect stability for constant series
        }
        
        // Calculate autocorrelation at lag 1
        let mut autocorr = 0.0;
        for i in 1..n {
            autocorr += (data[i] - mean) * (data[i-1] - mean);
        }
        autocorr /= (n - 1) as f64 * variance;
        
        // Calculate return to mean tendency
        let mut mean_reversion = 0.0;
        for i in 1..n {
            let deviation = (data[i-1] - mean).abs();
            let direction = if data[i-1] > mean { -1.0 } else { 1.0 };
            let change = data[i] - data[i-1];
            mean_reversion += direction * change / (deviation + 1e-12);
        }
        mean_reversion /= (n - 1) as f64;
        
        // Combine autocorrelation and mean reversion
        let equilibrium = (autocorr.abs() + mean_reversion.max(0.0)) / 2.0;
        Ok(equilibrium.clamp(0.0, 1.0))
    }
    
    /// Calculate fragility score based on volatility clustering and extremes
    fn calculate_fragility_score(&self, data: ArrayView1<f64>) -> Result<f64> {
        let n = data.len();
        if n < 4 {
            return Ok(0.0);
        }
        
        // Calculate returns
        let mut returns = Vec::with_capacity(n - 1);
        for i in 1..n {
            if data[i-1].abs() > 1e-12 {
                returns.push((data[i] - data[i-1]) / data[i-1]);
            }
        }
        
        if returns.is_empty() {
            return Ok(0.0);
        }
        
        // Calculate volatility clustering (GARCH-like effect)
        let mut vol_clustering = 0.0;
        if returns.len() >= 3 {
            for i in 2..returns.len() {
                let vol_prev = returns[i-1].abs();
                let vol_curr = returns[i].abs();
                vol_clustering += (vol_prev * vol_curr).sqrt();
            }
            vol_clustering /= (returns.len() - 2) as f64;
        }
        
        // Calculate extreme value frequency
        let return_std = {
            let mean = returns.iter().sum::<f64>() / returns.len() as f64;
            let variance = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64;
            variance.sqrt()
        };
        
        let extreme_threshold = 2.0 * return_std;
        let extreme_count = returns.iter().filter(|&&r| r.abs() > extreme_threshold).count();
        let extreme_frequency = extreme_count as f64 / returns.len() as f64;
        
        // Combine volatility clustering and extreme frequency
        let fragility = (vol_clustering + extreme_frequency * 2.0) / 3.0;
        Ok(fragility.clamp(0.0, 1.0))
    }
    
    /// Detect avalanche events in the time series
    fn detect_avalanches(&self, data: ArrayView1<f64>) -> Result<Vec<crate::AvalancheEvent>> {
        let n = data.len();
        if n < 10 {
            return Ok(Vec::new());
        }
        
        // Calculate rolling variance to detect volatility changes
        let window = (n / 10).max(5).min(20);
        let mut variances = Vec::with_capacity(n - window + 1);
        
        for i in window..=n {
            let segment = &data.as_slice().unwrap()[i-window..i];
            let mean = segment.iter().sum::<f64>() / segment.len() as f64;
            let variance = segment.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / segment.len() as f64;
            variances.push(variance);
        }
        
        if variances.is_empty() {
            return Ok(Vec::new());
        }
        
        // Find avalanche threshold (2 standard deviations above mean)
        let var_mean = variances.iter().sum::<f64>() / variances.len() as f64;
        let var_std = {
            let var_variance = variances.iter().map(|v| (v - var_mean).powi(2)).sum::<f64>() / variances.len() as f64;
            var_variance.sqrt()
        };
        
        let threshold = var_mean + 2.0 * var_std;
        
        // Detect avalanche events
        let mut events = Vec::new();
        let mut in_avalanche = false;
        let mut start_idx = 0;
        
        for (i, &variance) in variances.iter().enumerate() {
            if variance > threshold && !in_avalanche {
                // Start of avalanche
                in_avalanche = true;
                start_idx = i + window - 1;
            } else if variance <= threshold && in_avalanche {
                // End of avalanche
                in_avalanche = false;
                let end_idx = i + window - 1;
                let duration = end_idx - start_idx;
                
                if duration >= 2 {
                    // Calculate avalanche magnitude and intensity
                    let segment = &data.as_slice().unwrap()[start_idx..=end_idx];
                    let magnitude = segment.iter().map(|x| x.abs()).sum::<f64>();
                    let intensity = magnitude / duration as f64;
                    
                    events.push(crate::AvalancheEvent {
                        start_index: start_idx,
                        end_index: end_idx,
                        magnitude,
                        duration,
                        intensity,
                    });
                }
            }
        }
        
        // Handle case where avalanche continues to end
        if in_avalanche {
            let end_idx = n - 1;
            let duration = end_idx - start_idx;
            
            if duration >= 2 {
                let segment = &data.as_slice().unwrap()[start_idx..=end_idx];
                let magnitude = segment.iter().map(|x| x.abs()).sum::<f64>();
                let intensity = magnitude / duration as f64;
                
                events.push(crate::AvalancheEvent {
                    start_index: start_idx,
                    end_index: end_idx,
                    magnitude,
                    duration,
                    intensity,
                });
            }
        }
        
        Ok(events)
    }
    
    /// Analyze time series with parallel processing
    #[cfg(feature = "parallel")]
    pub fn analyze_parallel(&self, data: &Array1<f64>) -> Result<SOCResult> {
        // For parallel analysis, we can split certain computations
        // This is a simplified example - full parallel implementation would be more complex
        self.analyze(data)
    }
    
    /// Get current parameters
    pub fn parameters(&self) -> &SOCParameters {
        &self.params
    }
    
    /// Update parameters
    pub fn set_parameters(&mut self, params: SOCParameters) {
        self.params = params;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;
    use crate::SOCParameters;
    
    #[test]
    fn test_soc_analyzer_creation() {
        let params = SOCParameters::default();
        let analyzer = SOCAnalyzer::new(params);
        assert_eq!(analyzer.parameters().sample_entropy_m, 2);
    }
    
    #[test]
    fn test_soc_analysis() {
        let analyzer = SOCAnalyzer::default();
        let data = Array1::from_vec((0..100).map(|x| (x as f64 * 0.1).sin()).collect());
        
        let result = analyzer.analyze(&data);
        assert!(result.is_ok());
        
        let result = result.unwrap();
        assert!(result.sample_entropy >= 0.0);
        assert!(result.entropy_rate >= 0.0);
        assert!(result.complexity_measure >= 0.0);
        assert!(result.equilibrium_score >= 0.0 && result.equilibrium_score <= 1.0);
        assert!(result.fragility_score >= 0.0 && result.fragility_score <= 1.0);
    }
    
    #[test]
    fn test_insufficient_data() {
        let analyzer = SOCAnalyzer::default();
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0]); // Too few points
        
        let result = analyzer.analyze(&data);
        assert!(result.is_err());
        
        match result.unwrap_err() {
            SOCError::InsufficientData { required, actual } => {
                assert_eq!(required, 20);
                assert_eq!(actual, 3);
            }
            _ => panic!("Expected InsufficientData error"),
        }
    }
}