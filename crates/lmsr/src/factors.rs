//! Standard factors for LMSR trading systems

use crate::errors::{LMSRError, Result};
use crate::errors::validate_not_empty;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Individual factor in the trading system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Factor {
    /// Factor name
    pub name: String,
    /// Factor weight (default 1.0)
    pub weight: f64,
    /// Factor description
    pub description: String,
    /// Minimum expected value
    pub min_value: f64,
    /// Maximum expected value
    pub max_value: f64,
    /// Whether this factor is enabled
    pub enabled: bool,
}

impl Factor {
    /// Create a new factor
    pub fn new(name: impl Into<String>, weight: f64, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            weight,
            description: description.into(),
            min_value: -1.0,
            max_value: 1.0,
            enabled: true,
        }
    }

    /// Apply weight to a factor value
    pub fn apply_weight(&self, value: f64) -> Result<f64> {
        if !self.enabled {
            return Ok(0.0);
        }
        
        // Clamp value to expected range
        let clamped = value.max(self.min_value).min(self.max_value);
        Ok(clamped * self.weight)
    }

    /// Normalize a value to the factor's range
    pub fn normalize(&self, value: f64) -> f64 {
        if self.max_value == self.min_value {
            return 0.0;
        }
        
        (value - self.min_value) / (self.max_value - self.min_value)
    }

    /// Denormalize a value from [0,1] to the factor's range
    pub fn denormalize(&self, normalized: f64) -> f64 {
        self.min_value + normalized * (self.max_value - self.min_value)
    }
}

/// Standard 8-factor model for trading systems
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StandardFactors {
    factors: Vec<Factor>,
    factor_map: HashMap<String, usize>,
}

impl StandardFactors {
    /// Create a new standard factors instance with default 8 factors
    pub fn new() -> Self {
        let factors = vec![
            Factor::new(
                "trend", 
                1.0, 
                "Price trend strength and direction"
            ),
            Factor::new(
                "volatility", 
                0.8, 
                "Price volatility and uncertainty"
            ),
            Factor::new(
                "momentum", 
                0.9, 
                "Price momentum and acceleration"
            ),
            Factor::new(
                "sentiment", 
                0.7, 
                "Market sentiment and positioning"
            ),
            Factor::new(
                "liquidity", 
                0.6, 
                "Market liquidity and depth"
            ),
            Factor::new(
                "correlation", 
                0.5, 
                "Cross-asset correlation patterns"
            ),
            Factor::new(
                "cycle", 
                0.4, 
                "Market cycle and seasonality"
            ),
            Factor::new(
                "anomaly", 
                0.3, 
                "Market anomalies and inefficiencies"
            ),
        ];
        
        let factor_map: HashMap<String, usize> = factors
            .iter()
            .enumerate()
            .map(|(i, f)| (f.name.clone(), i))
            .collect();
        
        Self { factors, factor_map }
    }

    /// Create a custom factors instance
    pub fn custom(factors: Vec<Factor>) -> Self {
        let factor_map: HashMap<String, usize> = factors
            .iter()
            .enumerate()
            .map(|(i, f)| (f.name.clone(), i))
            .collect();
        
        Self { factors, factor_map }
    }

    /// Get the number of factors
    pub fn len(&self) -> usize {
        self.factors.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.factors.is_empty()
    }

    /// Get factor by index
    pub fn get(&self, index: usize) -> Option<&Factor> {
        self.factors.get(index)
    }

    /// Get factor by name
    pub fn get_by_name(&self, name: &str) -> Option<&Factor> {
        self.factor_map.get(name).and_then(|&index| self.factors.get(index))
    }

    /// Get factor names
    pub fn names(&self) -> Vec<&str> {
        self.factors.iter().map(|f| f.name.as_str()).collect()
    }

    /// Get factor weights
    pub fn weights(&self) -> Vec<f64> {
        self.factors.iter().map(|f| f.weight).collect()
    }

    /// Apply weights to factor values
    pub fn apply_weights(&self, values: &[f64]) -> Result<Vec<f64>> {
        validate_not_empty(values)?;
        
        if values.len() != self.factors.len() {
            return Err(LMSRError::dimension_mismatch(self.factors.len(), values.len()));
        }
        
        let mut weighted_values = Vec::with_capacity(values.len());
        for (factor, &value) in self.factors.iter().zip(values.iter()) {
            weighted_values.push(factor.apply_weight(value)?);
        }
        
        Ok(weighted_values)
    }

    /// Normalize factor values to [0,1] range
    pub fn normalize_values(&self, values: &[f64]) -> Result<Vec<f64>> {
        validate_not_empty(values)?;
        
        if values.len() != self.factors.len() {
            return Err(LMSRError::dimension_mismatch(self.factors.len(), values.len()));
        }
        
        let mut normalized = Vec::with_capacity(values.len());
        for (factor, &value) in self.factors.iter().zip(values.iter()) {
            normalized.push(factor.normalize(value));
        }
        
        Ok(normalized)
    }

    /// Denormalize factor values from [0,1] to original range
    pub fn denormalize_values(&self, normalized: &[f64]) -> Result<Vec<f64>> {
        validate_not_empty(normalized)?;
        
        if normalized.len() != self.factors.len() {
            return Err(LMSRError::dimension_mismatch(self.factors.len(), normalized.len()));
        }
        
        let mut denormalized = Vec::with_capacity(normalized.len());
        for (factor, &value) in self.factors.iter().zip(normalized.iter()) {
            denormalized.push(factor.denormalize(value));
        }
        
        Ok(denormalized)
    }

    /// Update factor weight
    pub fn set_weight(&mut self, index: usize, weight: f64) -> Result<()> {
        if index >= self.factors.len() {
            return Err(LMSRError::invalid_input(format!(
                "Factor index {} out of bounds (max {})",
                index,
                self.factors.len() - 1
            )));
        }
        
        self.factors[index].weight = weight;
        Ok(())
    }

    /// Update factor weight by name
    pub fn set_weight_by_name(&mut self, name: &str, weight: f64) -> Result<()> {
        let index = self.factor_map.get(name).ok_or_else(|| {
            LMSRError::invalid_input(format!("Factor '{}' not found", name))
        })?;
        
        self.factors[*index].weight = weight;
        Ok(())
    }

    /// Enable/disable factor
    pub fn set_enabled(&mut self, index: usize, enabled: bool) -> Result<()> {
        if index >= self.factors.len() {
            return Err(LMSRError::invalid_input(format!(
                "Factor index {} out of bounds (max {})",
                index,
                self.factors.len() - 1
            )));
        }
        
        self.factors[index].enabled = enabled;
        Ok(())
    }

    /// Enable/disable factor by name
    pub fn set_enabled_by_name(&mut self, name: &str, enabled: bool) -> Result<()> {
        let index = self.factor_map.get(name).ok_or_else(|| {
            LMSRError::invalid_input(format!("Factor '{}' not found", name))
        })?;
        
        self.factors[*index].enabled = enabled;
        Ok(())
    }

    /// Get enabled factors
    pub fn enabled_factors(&self) -> Vec<&Factor> {
        self.factors.iter().filter(|f| f.enabled).collect()
    }

    /// Calculate factor correlation matrix
    pub fn correlation_matrix(&self, data: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        validate_not_empty(data)?;
        
        let n_factors = self.factors.len();
        let n_samples = data.len();
        
        // Validate input dimensions
        for (i, sample) in data.iter().enumerate() {
            if sample.len() != n_factors {
                return Err(LMSRError::dimension_mismatch(n_factors, sample.len()));
            }
        }
        
        // Calculate means
        let mut means = vec![0.0; n_factors];
        for sample in data {
            for (i, &value) in sample.iter().enumerate() {
                means[i] += value;
            }
        }
        for mean in &mut means {
            *mean /= n_samples as f64;
        }
        
        // Calculate correlation matrix
        let mut correlation = vec![vec![0.0; n_factors]; n_factors];
        let mut variances = vec![0.0; n_factors];
        
        // Calculate variances and covariances
        for sample in data {
            for i in 0..n_factors {
                let diff_i = sample[i] - means[i];
                variances[i] += diff_i * diff_i;
                
                for j in 0..n_factors {
                    let diff_j = sample[j] - means[j];
                    correlation[i][j] += diff_i * diff_j;
                }
            }
        }
        
        // Normalize to get correlation coefficients
        for i in 0..n_factors {
            variances[i] = (variances[i] / (n_samples - 1) as f64).sqrt();
        }
        
        for i in 0..n_factors {
            for j in 0..n_factors {
                if i == j {
                    correlation[i][j] = 1.0;
                } else {
                    correlation[i][j] /= (n_samples - 1) as f64;
                    correlation[i][j] /= variances[i] * variances[j];
                }
            }
        }
        
        Ok(correlation)
    }

    /// Calculate factor loadings using PCA
    pub fn factor_loadings(&self, data: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        // Simplified PCA implementation
        let correlation = self.correlation_matrix(data)?;
        let n_factors = self.factors.len();
        
        // Return identity matrix as placeholder for full PCA implementation
        let mut loadings = vec![vec![0.0; n_factors]; n_factors];
        for i in 0..n_factors {
            loadings[i][i] = 1.0;
        }
        
        Ok(loadings)
    }

    /// Calculate factor importance scores
    pub fn importance_scores(&self, data: &[Vec<f64>]) -> Result<Vec<f64>> {
        validate_not_empty(data)?;
        
        let n_factors = self.factors.len();
        let mut scores = vec![0.0; n_factors];
        
        // Calculate variance for each factor
        for factor_idx in 0..n_factors {
            let values: Vec<f64> = data.iter().map(|sample| sample[factor_idx]).collect();
            let mean = values.iter().sum::<f64>() / values.len() as f64;
            let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
            
            scores[factor_idx] = variance * self.factors[factor_idx].weight;
        }
        
        // Normalize scores
        let max_score = scores.iter().fold(0.0, |a, &b| a.max(b));
        if max_score > 0.0 {
            for score in &mut scores {
                *score /= max_score;
            }
        }
        
        Ok(scores)
    }

    /// Create factor subset
    pub fn subset(&self, indices: &[usize]) -> Result<Self> {
        validate_not_empty(indices)?;
        
        let mut factors = Vec::new();
        for &index in indices {
            if index >= self.factors.len() {
                return Err(LMSRError::invalid_input(format!(
                    "Factor index {} out of bounds (max {})",
                    index,
                    self.factors.len() - 1
                )));
            }
            factors.push(self.factors[index].clone());
        }
        
        Ok(Self::custom(factors))
    }

    /// Create factor subset by names
    pub fn subset_by_names(&self, names: &[&str]) -> Result<Self> {
        validate_not_empty(names)?;
        
        let mut factors = Vec::new();
        for &name in names {
            let factor = self.get_by_name(name).ok_or_else(|| {
                LMSRError::invalid_input(format!("Factor '{}' not found", name))
            })?;
            factors.push(factor.clone());
        }
        
        Ok(Self::custom(factors))
    }
}

impl Default for StandardFactors {
    fn default() -> Self {
        Self::new()
    }
}

/// Predefined factor configurations
pub mod presets {
    use super::*;

    /// Trend-focused factor configuration
    pub fn trend_focused() -> StandardFactors {
        let mut factors = StandardFactors::new();
        factors.set_weight_by_name("trend", 2.0).unwrap();
        factors.set_weight_by_name("momentum", 1.5).unwrap();
        factors.set_weight_by_name("volatility", 0.5).unwrap();
        factors
    }

    /// Volatility-focused factor configuration
    pub fn volatility_focused() -> StandardFactors {
        let mut factors = StandardFactors::new();
        factors.set_weight_by_name("volatility", 2.0).unwrap();
        factors.set_weight_by_name("sentiment", 1.5).unwrap();
        factors.set_weight_by_name("trend", 0.5).unwrap();
        factors
    }

    /// Momentum-focused factor configuration
    pub fn momentum_focused() -> StandardFactors {
        let mut factors = StandardFactors::new();
        factors.set_weight_by_name("momentum", 2.0).unwrap();
        factors.set_weight_by_name("trend", 1.5).unwrap();
        factors.set_weight_by_name("correlation", 1.2).unwrap();
        factors
    }

    /// Sentiment-focused factor configuration
    pub fn sentiment_focused() -> StandardFactors {
        let mut factors = StandardFactors::new();
        factors.set_weight_by_name("sentiment", 2.0).unwrap();
        factors.set_weight_by_name("anomaly", 1.5).unwrap();
        factors.set_weight_by_name("volatility", 1.2).unwrap();
        factors
    }

    /// Balanced factor configuration
    pub fn balanced() -> StandardFactors {
        StandardFactors::new()
    }

    /// Conservative factor configuration
    pub fn conservative() -> StandardFactors {
        let mut factors = StandardFactors::new();
        factors.set_weight_by_name("liquidity", 2.0).unwrap();
        factors.set_weight_by_name("correlation", 1.5).unwrap();
        factors.set_weight_by_name("volatility", 0.3).unwrap();
        factors.set_weight_by_name("anomaly", 0.1).unwrap();
        factors
    }

    /// Aggressive factor configuration
    pub fn aggressive() -> StandardFactors {
        let mut factors = StandardFactors::new();
        factors.set_weight_by_name("anomaly", 2.0).unwrap();
        factors.set_weight_by_name("momentum", 1.8).unwrap();
        factors.set_weight_by_name("trend", 1.5).unwrap();
        factors.set_weight_by_name("liquidity", 0.2).unwrap();
        factors
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_factor_creation() {
        let factor = Factor::new("test", 1.0, "Test factor");
        assert_eq!(factor.name, "test");
        assert_eq!(factor.weight, 1.0);
        assert_eq!(factor.description, "Test factor");
        assert!(factor.enabled);
    }

    #[test]
    fn test_factor_weight_application() {
        let factor = Factor::new("test", 2.0, "Test factor");
        let result = factor.apply_weight(0.5).unwrap();
        assert_eq!(result, 1.0);
    }

    #[test]
    fn test_factor_normalization() {
        let mut factor = Factor::new("test", 1.0, "Test factor");
        factor.min_value = 0.0;
        factor.max_value = 10.0;
        
        let normalized = factor.normalize(5.0);
        assert_eq!(normalized, 0.5);
        
        let denormalized = factor.denormalize(0.5);
        assert_eq!(denormalized, 5.0);
    }

    #[test]
    fn test_standard_factors_creation() {
        let factors = StandardFactors::new();
        assert_eq!(factors.len(), 8);
        
        let names = factors.names();
        assert_eq!(names[0], "trend");
        assert_eq!(names[1], "volatility");
        assert_eq!(names[2], "momentum");
        assert_eq!(names[3], "sentiment");
        assert_eq!(names[4], "liquidity");
        assert_eq!(names[5], "correlation");
        assert_eq!(names[6], "cycle");
        assert_eq!(names[7], "anomaly");
    }

    #[test]
    fn test_apply_weights() {
        let factors = StandardFactors::new();
        let values = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        
        let weighted = factors.apply_weights(&values).unwrap();
        assert_eq!(weighted.len(), 8);
        
        // First factor has weight 1.0
        assert_eq!(weighted[0], 0.1);
        // Second factor has weight 0.8
        assert_eq!(weighted[1], 0.2 * 0.8);
    }

    #[test]
    fn test_normalize_values() {
        let factors = StandardFactors::new();
        let values = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        
        let normalized = factors.normalize_values(&values).unwrap();
        assert_eq!(normalized.len(), 8);
        
        // All values should be 0.5 (middle of [-1, 1] range)
        for &n in &normalized {
            assert_relative_eq!(n, 0.5, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_set_weight() {
        let mut factors = StandardFactors::new();
        factors.set_weight(0, 2.0).unwrap();
        
        assert_eq!(factors.get(0).unwrap().weight, 2.0);
        
        factors.set_weight_by_name("volatility", 1.5).unwrap();
        assert_eq!(factors.get_by_name("volatility").unwrap().weight, 1.5);
    }

    #[test]
    fn test_enable_disable() {
        let mut factors = StandardFactors::new();
        factors.set_enabled(0, false).unwrap();
        
        assert!(!factors.get(0).unwrap().enabled);
        
        factors.set_enabled_by_name("trend", true).unwrap();
        assert!(factors.get_by_name("trend").unwrap().enabled);
    }

    #[test]
    fn test_correlation_matrix() {
        let factors = StandardFactors::new();
        let data = vec![
            vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            vec![0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        ];
        
        let correlation = factors.correlation_matrix(&data).unwrap();
        assert_eq!(correlation.len(), 8);
        assert_eq!(correlation[0].len(), 8);
        
        // Diagonal should be 1.0
        for i in 0..8 {
            assert_relative_eq!(correlation[i][i], 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_subset() {
        let factors = StandardFactors::new();
        let subset = factors.subset(&[0, 2, 4]).unwrap();
        
        assert_eq!(subset.len(), 3);
        assert_eq!(subset.names(), vec!["trend", "momentum", "liquidity"]);
    }

    #[test]
    fn test_subset_by_names() {
        let factors = StandardFactors::new();
        let subset = factors.subset_by_names(&["trend", "volatility", "momentum"]).unwrap();
        
        assert_eq!(subset.len(), 3);
        assert_eq!(subset.names(), vec!["trend", "volatility", "momentum"]);
    }

    #[test]
    fn test_presets() {
        let trend_focused = presets::trend_focused();
        assert_eq!(trend_focused.get_by_name("trend").unwrap().weight, 2.0);
        
        let volatility_focused = presets::volatility_focused();
        assert_eq!(volatility_focused.get_by_name("volatility").unwrap().weight, 2.0);
        
        let momentum_focused = presets::momentum_focused();
        assert_eq!(momentum_focused.get_by_name("momentum").unwrap().weight, 2.0);
    }
}