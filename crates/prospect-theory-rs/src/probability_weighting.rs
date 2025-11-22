//! Probability weighting functions for prospect theory

use crate::errors::{ProspectTheoryError, Result, validate_probability};
use crate::utils::{safe_pow, safe_div, approx_equal};
use crate::FINANCIAL_PRECISION;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Parameters for probability weighting functions
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WeightingParams {
    /// Curvature parameter for gains (typically 0.61)
    pub gamma_gains: f64,
    /// Curvature parameter for losses (typically 0.69)
    pub gamma_losses: f64,
    /// Optimism parameter for gains (typically 1.0)
    pub delta_gains: f64,
    /// Optimism parameter for losses (typically 1.0)
    pub delta_losses: f64,
}

impl Default for WeightingParams {
    fn default() -> Self {
        Self {
            gamma_gains: 0.61,
            gamma_losses: 0.69,
            delta_gains: 1.0,
            delta_losses: 1.0,
        }
    }
}

impl WeightingParams {
    /// Create new parameters with validation
    pub fn new(
        gamma_gains: f64,
        gamma_losses: f64,
        delta_gains: f64,
        delta_losses: f64,
    ) -> Result<Self> {
        // Validate gamma parameters (curvature)
        if gamma_gains <= 0.0 || gamma_gains > 1.0 {
            return Err(ProspectTheoryError::invalid_parameter(
                "gamma_gains",
                gamma_gains,
                "0 < gamma_gains <= 1",
            ));
        }

        if gamma_losses <= 0.0 || gamma_losses > 1.0 {
            return Err(ProspectTheoryError::invalid_parameter(
                "gamma_losses",
                gamma_losses,
                "0 < gamma_losses <= 1",
            ));
        }

        // Validate delta parameters (optimism)
        if delta_gains <= 0.0 {
            return Err(ProspectTheoryError::invalid_parameter(
                "delta_gains",
                delta_gains,
                "delta_gains > 0",
            ));
        }

        if delta_losses <= 0.0 {
            return Err(ProspectTheoryError::invalid_parameter(
                "delta_losses",
                delta_losses,
                "delta_losses > 0",
            ));
        }

        Ok(Self {
            gamma_gains,
            gamma_losses,
            delta_gains,
            delta_losses,
        })
    }

    /// Validate all parameters
    pub fn validate(&self) -> Result<()> {
        Self::new(
            self.gamma_gains,
            self.gamma_losses,
            self.delta_gains,
            self.delta_losses,
        )?;
        Ok(())
    }
}

/// Probability weighting function types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightingFunction {
    /// Tversky-Kahneman (1992) weighting function
    TverskyKahneman,
    /// Prelec (1998) weighting function
    Prelec,
    /// Linear weighting (no distortion)
    Linear,
}

/// Thread-safe probability weighting calculator
#[derive(Debug, Clone)]
pub struct ProbabilityWeighting {
    params: Arc<WeightingParams>,
    function_type: WeightingFunction,
}

impl ProbabilityWeighting {
    /// Create new probability weighting with parameters
    pub fn new(params: WeightingParams, function_type: WeightingFunction) -> Result<Self> {
        params.validate()?;
        Ok(Self {
            params: Arc::new(params),
            function_type,
        })
    }

    /// Create with default Tversky-Kahneman parameters
    pub fn default_tk() -> Self {
        Self {
            params: Arc::new(WeightingParams::default()),
            function_type: WeightingFunction::TverskyKahneman,
        }
    }

    /// Create Prelec weighting function
    pub fn prelec(params: WeightingParams) -> Result<Self> {
        Self::new(params, WeightingFunction::Prelec)
    }

    /// Create linear weighting function (no distortion)
    pub fn linear() -> Self {
        Self {
            params: Arc::new(WeightingParams::default()),
            function_type: WeightingFunction::Linear,
        }
    }

    /// Calculate probability weight for gains domain
    /// 
    /// Tversky-Kahneman: w+(p) = δ * p^γ / (δ * p^γ + (1-p)^γ)^(1/γ)
    /// Prelec: w+(p) = exp(-δ * (-ln(p))^γ)
    pub fn weight_gains(&self, probability: f64) -> Result<f64> {
        validate_probability(probability)?;

        match self.function_type {
            WeightingFunction::Linear => Ok(probability),
            WeightingFunction::TverskyKahneman => {
                self.tversky_kahneman_weight(probability, self.params.gamma_gains, self.params.delta_gains)
            }
            WeightingFunction::Prelec => {
                self.prelec_weight(probability, self.params.gamma_gains, self.params.delta_gains)
            }
        }
    }

    /// Calculate probability weight for losses domain
    pub fn weight_losses(&self, probability: f64) -> Result<f64> {
        validate_probability(probability)?;

        match self.function_type {
            WeightingFunction::Linear => Ok(probability),
            WeightingFunction::TverskyKahneman => {
                self.tversky_kahneman_weight(probability, self.params.gamma_losses, self.params.delta_losses)
            }
            WeightingFunction::Prelec => {
                self.prelec_weight(probability, self.params.gamma_losses, self.params.delta_losses)
            }
        }
    }

    /// Tversky-Kahneman probability weighting function
    fn tversky_kahneman_weight(&self, p: f64, gamma: f64, delta: f64) -> Result<f64> {
        if approx_equal(p, 0.0) {
            return Ok(0.0);
        }
        if approx_equal(p, 1.0) {
            return Ok(1.0);
        }

        // w(p) = δ * p^γ / (δ * p^γ + (1-p)^γ)
        let p_gamma = safe_pow(p, gamma)?;
        let one_minus_p = 1.0 - p;
        let one_minus_p_gamma = safe_pow(one_minus_p, gamma)?;
        
        let numerator = delta * p_gamma;
        let denominator = delta * p_gamma + one_minus_p_gamma;
        
        safe_div(numerator, denominator)
    }

    /// Prelec probability weighting function
    fn prelec_weight(&self, p: f64, gamma: f64, delta: f64) -> Result<f64> {
        if approx_equal(p, 0.0) {
            return Ok(0.0);
        }
        if approx_equal(p, 1.0) {
            return Ok(1.0);
        }

        // w(p) = exp(-δ * (-ln(p))^γ)
        let ln_p = p.ln();
        if !ln_p.is_finite() {
            return Err(ProspectTheoryError::computation_failed(
                "Logarithm of probability in Prelec function",
            ));
        }

        let neg_ln_p = -ln_p;
        let powered = safe_pow(neg_ln_p, gamma)?;
        let exponent = -delta * powered;
        
        let result = exponent.exp();
        if !result.is_finite() {
            return Err(ProspectTheoryError::numerical_overflow("Prelec weighting"));
        }

        Ok(result)
    }

    /// Calculate weights for multiple probabilities (vectorized)
    pub fn weights_gains(&self, probabilities: &[f64]) -> Result<Vec<f64>> {
        if probabilities.is_empty() {
            return Ok(Vec::new());
        }

        probabilities
            .iter()
            .map(|&prob| self.weight_gains(prob))
            .collect()
    }

    /// Calculate weights for losses (vectorized)
    pub fn weights_losses(&self, probabilities: &[f64]) -> Result<Vec<f64>> {
        if probabilities.is_empty() {
            return Ok(Vec::new());
        }

        probabilities
            .iter()
            .map(|&prob| self.weight_losses(prob))
            .collect()
    }

    /// Calculate weights with parallel processing for large datasets
    #[cfg(feature = "parallel")]
    pub fn weights_gains_parallel(&self, probabilities: &[f64]) -> Result<Vec<f64>> {
        use rayon::prelude::*;

        if probabilities.is_empty() {
            return Ok(Vec::new());
        }

        if probabilities.len() > 1000 {
            probabilities
                .par_iter()
                .map(|&prob| self.weight_gains(prob))
                .collect()
        } else {
            self.weights_gains(probabilities)
        }
    }

    /// Calculate weights with parallel processing for large datasets (fallback)
    #[cfg(not(feature = "parallel"))]
    pub fn weights_gains_parallel(&self, probabilities: &[f64]) -> Result<Vec<f64>> {
        self.weights_gains(probabilities)
    }

    /// Calculate weights for losses with parallel processing
    #[cfg(feature = "parallel")]
    pub fn weights_losses_parallel(&self, probabilities: &[f64]) -> Result<Vec<f64>> {
        use rayon::prelude::*;

        if probabilities.is_empty() {
            return Ok(Vec::new());
        }

        if probabilities.len() > 1000 {
            probabilities
                .par_iter()
                .map(|&prob| self.weight_losses(prob))
                .collect()
        } else {
            self.weights_losses(probabilities)
        }
    }

    /// Calculate weights for losses with parallel processing (fallback)
    #[cfg(not(feature = "parallel"))]
    pub fn weights_losses_parallel(&self, probabilities: &[f64]) -> Result<Vec<f64>> {
        self.weights_losses(probabilities)
    }

    /// Calculate decision weights for a probability distribution
    /// Handles rank-dependent weighting
    pub fn decision_weights(&self, probabilities: &[f64], outcomes: &[f64]) -> Result<Vec<f64>> {
        if probabilities.len() != outcomes.len() {
            return Err(ProspectTheoryError::computation_failed(
                "Probabilities and outcomes must have same length",
            ));
        }

        if probabilities.is_empty() {
            return Ok(Vec::new());
        }

        // Create pairs and sort by outcomes
        let mut pairs: Vec<(f64, f64)> = probabilities
            .iter()
            .zip(outcomes.iter())
            .map(|(&p, &o)| (p, o))
            .collect();
        
        pairs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let mut decision_weights = Vec::with_capacity(pairs.len());
        let mut cumulative_prob = 0.0;
        
        // Calculate decision weights using rank-dependent approach
        for (i, (prob, outcome)) in pairs.iter().enumerate() {
            let new_cumulative = cumulative_prob + prob;
            
            let weight = if *outcome >= 0.0 {
                // Gains: w(p₁ + ... + pᵢ) - w(p₁ + ... + pᵢ₋₁)
                let weight_upper = self.weight_gains(new_cumulative)?;
                let weight_lower = if i == 0 { 0.0 } else { self.weight_gains(cumulative_prob)? };
                weight_upper - weight_lower
            } else {
                // Losses: w(pᵢ + ... + pₙ) - w(pᵢ₊₁ + ... + pₙ)
                let remaining_prob = 1.0 - cumulative_prob;
                let next_remaining_prob = 1.0 - new_cumulative;
                let weight_upper = self.weight_losses(remaining_prob)?;
                let weight_lower = self.weight_losses(next_remaining_prob)?;
                weight_upper - weight_lower
            };
            
            decision_weights.push(weight);
            cumulative_prob = new_cumulative;
        }

        Ok(decision_weights)
    }

    /// Get parameters (immutable reference)
    pub fn params(&self) -> &WeightingParams {
        &self.params
    }

    /// Get function type
    pub fn function_type(&self) -> WeightingFunction {
        self.function_type
    }

    /// Calculate attractiveness (inverse S-curve measure)
    pub fn attractiveness(&self, probability: f64) -> Result<f64> {
        validate_probability(probability)?;
        
        let weight_gains = self.weight_gains(probability)?;
        let weight_losses = self.weight_losses(probability)?;
        
        // Measure of probability distortion
        let gains_distortion = (weight_gains - probability).abs();
        let losses_distortion = (weight_losses - probability).abs();
        
        Ok((gains_distortion + losses_distortion) / 2.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_default_parameters() {
        let params = WeightingParams::default();
        assert_eq!(params.gamma_gains, 0.61);
        assert_eq!(params.gamma_losses, 0.69);
        assert_eq!(params.delta_gains, 1.0);
        assert_eq!(params.delta_losses, 1.0);
    }

    #[test]
    fn test_parameter_validation() {
        // Valid parameters
        assert!(WeightingParams::new(0.61, 0.69, 1.0, 1.0).is_ok());
        
        // Invalid gamma
        assert!(WeightingParams::new(0.0, 0.69, 1.0, 1.0).is_err());
        assert!(WeightingParams::new(1.5, 0.69, 1.0, 1.0).is_err());
        
        // Invalid delta
        assert!(WeightingParams::new(0.61, 0.69, 0.0, 1.0).is_err());
        assert!(WeightingParams::new(0.61, 0.69, 1.0, -1.0).is_err());
    }

    #[test]
    fn test_linear_weighting() {
        let pw = ProbabilityWeighting::linear();
        
        assert_relative_eq!(pw.weight_gains(0.0).unwrap(), 0.0, epsilon = FINANCIAL_PRECISION);
        assert_relative_eq!(pw.weight_gains(0.5).unwrap(), 0.5, epsilon = FINANCIAL_PRECISION);
        assert_relative_eq!(pw.weight_gains(1.0).unwrap(), 1.0, epsilon = FINANCIAL_PRECISION);
        
        assert_relative_eq!(pw.weight_losses(0.0).unwrap(), 0.0, epsilon = FINANCIAL_PRECISION);
        assert_relative_eq!(pw.weight_losses(0.5).unwrap(), 0.5, epsilon = FINANCIAL_PRECISION);
        assert_relative_eq!(pw.weight_losses(1.0).unwrap(), 1.0, epsilon = FINANCIAL_PRECISION);
    }

    #[test]
    fn test_tversky_kahneman_weighting() {
        let pw = ProbabilityWeighting::default_tk();
        
        // Test boundary conditions
        assert_relative_eq!(pw.weight_gains(0.0).unwrap(), 0.0, epsilon = FINANCIAL_PRECISION);
        assert_relative_eq!(pw.weight_gains(1.0).unwrap(), 1.0, epsilon = FINANCIAL_PRECISION);
        
        // Test inverse S-curve property
        let w_01 = pw.weight_gains(0.1).unwrap();
        let w_05 = pw.weight_gains(0.5).unwrap();
        let w_09 = pw.weight_gains(0.9).unwrap();
        
        // Should overweight small probabilities
        assert!(w_01 > 0.1);
        // Should underweight medium probabilities
        assert!(w_05 < 0.5);
        // Should underweight large probabilities
        assert!(w_09 < 0.9);
    }

    #[test]
    fn test_prelec_weighting() {
        let params = WeightingParams::default();
        let pw = ProbabilityWeighting::prelec(params).unwrap();
        
        // Test boundary conditions
        assert_relative_eq!(pw.weight_gains(0.0).unwrap(), 0.0, epsilon = FINANCIAL_PRECISION);
        assert_relative_eq!(pw.weight_gains(1.0).unwrap(), 1.0, epsilon = FINANCIAL_PRECISION);
        
        // Test that it produces valid weights
        let w_05 = pw.weight_gains(0.5).unwrap();
        assert!(w_05 > 0.0 && w_05 < 1.0);
    }

    #[test]
    fn test_vectorized_calculation() {
        let pw = ProbabilityWeighting::default_tk();
        let probabilities = vec![0.0, 0.1, 0.5, 0.9, 1.0];
        
        let weights = pw.weights_gains(&probabilities).unwrap();
        assert_eq!(weights.len(), probabilities.len());
        
        // Compare with individual calculations
        for (i, &prob) in probabilities.iter().enumerate() {
            let individual_weight = pw.weight_gains(prob).unwrap();
            assert_relative_eq!(weights[i], individual_weight, epsilon = FINANCIAL_PRECISION);
        }
    }

    #[test]
    fn test_decision_weights() {
        let pw = ProbabilityWeighting::default_tk();
        let probabilities = vec![0.3, 0.4, 0.3];
        let outcomes = vec![100.0, 0.0, -100.0];
        
        let decision_weights = pw.decision_weights(&probabilities, &outcomes).unwrap();
        assert_eq!(decision_weights.len(), probabilities.len());
        
        // Decision weights should sum to approximately 1
        let sum: f64 = decision_weights.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = FINANCIAL_PRECISION * 10.0);
    }

    #[test]
    fn test_probability_validation() {
        let pw = ProbabilityWeighting::default_tk();
        
        // Invalid probabilities
        assert!(pw.weight_gains(-0.1).is_err());
        assert!(pw.weight_gains(1.1).is_err());
        assert!(pw.weight_gains(f64::NAN).is_err());
        assert!(pw.weight_gains(f64::INFINITY).is_err());
    }

    #[test]
    fn test_thread_safety() {
        use std::thread;
        
        let pw = ProbabilityWeighting::default_tk();
        let pw_clone = pw.clone();
        
        let handle = thread::spawn(move || {
            pw_clone.weight_gains(0.5).unwrap()
        });
        
        let weight1 = pw.weight_gains(0.5).unwrap();
        let weight2 = handle.join().unwrap();
        
        assert_relative_eq!(weight1, weight2, epsilon = FINANCIAL_PRECISION);
    }

    #[test]
    fn test_attractiveness() {
        let pw = ProbabilityWeighting::default_tk();
        
        // Should be higher for extreme probabilities due to inverse S-curve
        let attract_01 = pw.attractiveness(0.1).unwrap();
        let attract_05 = pw.attractiveness(0.5).unwrap();
        let attract_09 = pw.attractiveness(0.9).unwrap();
        
        assert!(attract_01 > attract_05 || attract_09 > attract_05);
    }
}