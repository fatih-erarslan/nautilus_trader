//! Kahneman-Tversky value function implementation with financial precision

use crate::errors::{ProspectTheoryError, Result, validate_financial_bounds};
use crate::utils::{safe_pow, safe_abs, approx_equal};
use crate::FINANCIAL_PRECISION;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Parameters for the Kahneman-Tversky value function
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ValueFunctionParams {
    /// Risk aversion parameter for gains (typically 0.88)
    pub alpha: f64,
    /// Risk seeking parameter for losses (typically 0.88)
    pub beta: f64,
    /// Loss aversion coefficient (typically 2.25)
    pub lambda: f64,
    /// Reference point (typically 0.0)
    pub reference_point: f64,
}

impl Default for ValueFunctionParams {
    fn default() -> Self {
        Self {
            alpha: 0.88,
            beta: 0.88,
            lambda: 2.25,
            reference_point: 0.0,
        }
    }
}

impl ValueFunctionParams {
    /// Create new parameters with validation
    pub fn new(alpha: f64, beta: f64, lambda: f64, reference_point: f64) -> Result<Self> {
        // Validate alpha
        if alpha <= 0.0 || alpha > 1.0 {
            return Err(ProspectTheoryError::invalid_parameter(
                "alpha",
                alpha,
                "0 < alpha <= 1",
            ));
        }

        // Validate beta
        if beta <= 0.0 || beta > 1.0 {
            return Err(ProspectTheoryError::invalid_parameter(
                "beta",
                beta,
                "0 < beta <= 1",
            ));
        }

        // Validate lambda
        if lambda <= 1.0 {
            return Err(ProspectTheoryError::invalid_parameter(
                "lambda",
                lambda,
                "lambda > 1",
            ));
        }

        // Validate reference point
        validate_financial_bounds(reference_point, "reference_point")?;

        Ok(Self {
            alpha,
            beta,
            lambda,
            reference_point,
        })
    }

    /// Validate all parameters
    pub fn validate(&self) -> Result<()> {
        Self::new(self.alpha, self.beta, self.lambda, self.reference_point)?;
        Ok(())
    }
}

/// Thread-safe value function calculator
#[derive(Debug, Clone)]
pub struct ValueFunction {
    params: Arc<ValueFunctionParams>,
}

impl ValueFunction {
    /// Create new value function with parameters
    pub fn new(params: ValueFunctionParams) -> Result<Self> {
        params.validate()?;
        Ok(Self {
            params: Arc::new(params),
        })
    }

    /// Create with default Kahneman-Tversky parameters
    pub fn default_kt() -> Self {
        Self {
            params: Arc::new(ValueFunctionParams::default()),
        }
    }

    /// Calculate value function for a single outcome
    /// 
    /// V(x) = {
    ///   x^α                    if x ≥ 0 (gains)
    ///   -λ * |x|^β            if x < 0 (losses)
    /// }
    /// 
    /// where x is relative to the reference point
    pub fn value(&self, outcome: f64) -> Result<f64> {
        validate_financial_bounds(outcome, "outcome")?;
        
        let relative_outcome = outcome - self.params.reference_point;
        
        if relative_outcome >= 0.0 {
            // Gains: V(x) = x^α
            if approx_equal(relative_outcome, 0.0) {
                Ok(0.0)
            } else {
                safe_pow(relative_outcome, self.params.alpha)
            }
        } else {
            // Losses: V(x) = -λ * |x|^β
            let abs_loss = safe_abs(relative_outcome)?;
            let powered_loss = safe_pow(abs_loss, self.params.beta)?;
            Ok(-self.params.lambda * powered_loss)
        }
    }

    /// Calculate value function for multiple outcomes (vectorized)
    pub fn values(&self, outcomes: &[f64]) -> Result<Vec<f64>> {
        if outcomes.is_empty() {
            return Ok(Vec::new());
        }

        outcomes
            .iter()
            .map(|&outcome| self.value(outcome))
            .collect()
    }

    /// Calculate value function with parallel processing for large datasets
    #[cfg(feature = "parallel")]
    pub fn values_parallel(&self, outcomes: &[f64]) -> Result<Vec<f64>> {
        use rayon::prelude::*;

        if outcomes.is_empty() {
            return Ok(Vec::new());
        }

        // Use parallel processing for large datasets
        if outcomes.len() > 1000 {
            outcomes
                .par_iter()
                .map(|&outcome| self.value(outcome))
                .collect()
        } else {
            self.values(outcomes)
        }
    }

    /// Calculate value function with parallel processing for large datasets (fallback without rayon)
    #[cfg(not(feature = "parallel"))]
    pub fn values_parallel(&self, outcomes: &[f64]) -> Result<Vec<f64>> {
        // Fallback to sequential processing
        self.values(outcomes)
    }

    /// Calculate marginal value (derivative)
    /// 
    /// V'(x) = {
    ///   α * x^(α-1)           if x > 0
    ///   λ * β * |x|^(β-1)     if x < 0
    ///   undefined             if x = 0
    /// }
    pub fn marginal_value(&self, outcome: f64) -> Result<f64> {
        validate_financial_bounds(outcome, "outcome")?;
        
        let relative_outcome = outcome - self.params.reference_point;
        
        if approx_equal(relative_outcome, 0.0) {
            return Err(ProspectTheoryError::computation_failed(
                "Marginal value undefined at reference point",
            ));
        }

        if relative_outcome > 0.0 {
            // Gains: V'(x) = α * x^(α-1)
            let power = self.params.alpha - 1.0;
            let base_power = safe_pow(relative_outcome, power)?;
            Ok(self.params.alpha * base_power)
        } else {
            // Losses: V'(x) = λ * β * |x|^(β-1)
            let abs_loss = safe_abs(relative_outcome)?;
            let power = self.params.beta - 1.0;
            let base_power = safe_pow(abs_loss, power)?;
            Ok(self.params.lambda * self.params.beta * base_power)
        }
    }

    /// Calculate risk premium for a given lottery
    pub fn risk_premium(&self, outcomes: &[f64], probabilities: &[f64]) -> Result<f64> {
        if outcomes.len() != probabilities.len() {
            return Err(ProspectTheoryError::computation_failed(
                "Outcomes and probabilities must have same length",
            ));
        }

        // Calculate expected value
        let expected_value: f64 = outcomes
            .iter()
            .zip(probabilities.iter())
            .map(|(&outcome, &prob)| outcome * prob)
            .sum();

        // Calculate prospect value
        let prospect_values = self.values(outcomes)?;
        let prospect_value: f64 = prospect_values
            .iter()
            .zip(probabilities.iter())
            .map(|(&value, &prob)| value * prob)
            .sum();

        // Find certainty equivalent (inverse of value function)
        let certainty_equivalent = self.inverse_value(prospect_value)?;
        
        Ok(expected_value - certainty_equivalent)
    }

    /// Calculate inverse value function (certainty equivalent)
    pub fn inverse_value(&self, value: f64) -> Result<f64> {
        validate_financial_bounds(value, "value")?;

        if approx_equal(value, 0.0) {
            return Ok(self.params.reference_point);
        }

        if value > 0.0 {
            // Gains domain: x = value^(1/α)
            let power = 1.0 / self.params.alpha;
            let outcome = safe_pow(value, power)?;
            Ok(outcome + self.params.reference_point)
        } else {
            // Losses domain: x = -(|value|/λ)^(1/β)
            let abs_value = safe_abs(value)?;
            let normalized = abs_value / self.params.lambda;
            let power = 1.0 / self.params.beta;
            let outcome = safe_pow(normalized, power)?;
            Ok(-outcome + self.params.reference_point)
        }
    }

    /// Get parameters (immutable reference)
    pub fn params(&self) -> &ValueFunctionParams {
        &self.params
    }

    /// Calculate loss aversion ratio at specific point
    pub fn loss_aversion_ratio(&self, gain: f64, loss: f64) -> Result<f64> {
        if gain <= 0.0 || loss >= 0.0 {
            return Err(ProspectTheoryError::invalid_parameter(
                "gain/loss",
                0.0,
                "gain > 0, loss < 0",
            ));
        }

        let gain_value = self.value(gain + self.params.reference_point)?;
        let loss_value = self.value(loss + self.params.reference_point)?;
        
        if approx_equal(gain_value, 0.0) {
            return Err(ProspectTheoryError::division_by_zero(
                "gain value in loss aversion ratio",
            ));
        }

        Ok(-loss_value / gain_value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_default_parameters() {
        let params = ValueFunctionParams::default();
        assert_eq!(params.alpha, 0.88);
        assert_eq!(params.beta, 0.88);
        assert_eq!(params.lambda, 2.25);
        assert_eq!(params.reference_point, 0.0);
    }

    #[test]
    fn test_parameter_validation() {
        // Valid parameters
        assert!(ValueFunctionParams::new(0.88, 0.88, 2.25, 0.0).is_ok());
        
        // Invalid alpha
        assert!(ValueFunctionParams::new(0.0, 0.88, 2.25, 0.0).is_err());
        assert!(ValueFunctionParams::new(1.5, 0.88, 2.25, 0.0).is_err());
        
        // Invalid beta
        assert!(ValueFunctionParams::new(0.88, 0.0, 2.25, 0.0).is_err());
        assert!(ValueFunctionParams::new(0.88, 1.5, 2.25, 0.0).is_err());
        
        // Invalid lambda
        assert!(ValueFunctionParams::new(0.88, 0.88, 1.0, 0.0).is_err());
        assert!(ValueFunctionParams::new(0.88, 0.88, 0.5, 0.0).is_err());
    }

    #[test]
    fn test_value_function_gains() {
        let vf = ValueFunction::default_kt();
        
        // Test gains
        let value_100 = vf.value(100.0).unwrap();
        assert!(value_100 > 0.0);
        assert!(value_100 < 100.0); // Diminishing sensitivity
        
        // Test reference point
        let value_ref = vf.value(0.0).unwrap();
        assert_relative_eq!(value_ref, 0.0, epsilon = FINANCIAL_PRECISION);
    }

    #[test]
    fn test_value_function_losses() {
        let vf = ValueFunction::default_kt();
        
        // Test losses
        let value_neg100 = vf.value(-100.0).unwrap();
        assert!(value_neg100 < 0.0);
        
        // Test loss aversion
        let value_pos100 = vf.value(100.0).unwrap();
        assert!(value_neg100.abs() > value_pos100); // Loss aversion
    }

    #[test]
    fn test_vectorized_calculation() {
        let vf = ValueFunction::default_kt();
        let outcomes = vec![100.0, 0.0, -100.0, 50.0, -50.0];
        
        let values = vf.values(&outcomes).unwrap();
        assert_eq!(values.len(), outcomes.len());
        
        // Compare with individual calculations
        for (i, &outcome) in outcomes.iter().enumerate() {
            let individual_value = vf.value(outcome).unwrap();
            assert_relative_eq!(values[i], individual_value, epsilon = FINANCIAL_PRECISION);
        }
    }

    #[test]
    fn test_marginal_value() {
        let vf = ValueFunction::default_kt();
        
        // Marginal value for gains should be positive and decreasing
        let mv_10 = vf.marginal_value(10.0).unwrap();
        let mv_100 = vf.marginal_value(100.0).unwrap();
        assert!(mv_10 > 0.0);
        assert!(mv_100 > 0.0);
        assert!(mv_10 > mv_100); // Diminishing marginal utility
        
        // Marginal value undefined at reference point
        assert!(vf.marginal_value(0.0).is_err());
    }

    #[test]
    fn test_inverse_value() {
        let vf = ValueFunction::default_kt();
        
        let original = 100.0;
        let value = vf.value(original).unwrap();
        let recovered = vf.inverse_value(value).unwrap();
        
        assert_relative_eq!(original, recovered, epsilon = FINANCIAL_PRECISION * 10.0);
    }

    #[test]
    fn test_loss_aversion_ratio() {
        let vf = ValueFunction::default_kt();
        
        let ratio = vf.loss_aversion_ratio(100.0, -100.0).unwrap();
        assert!(ratio > 1.0); // Should show loss aversion
        assert!(ratio > 2.0); // Should be close to lambda parameter
    }

    #[test]
    fn test_thread_safety() {
        use std::thread;
        
        let vf = ValueFunction::default_kt();
        let vf_clone = vf.clone();
        
        let handle = thread::spawn(move || {
            vf_clone.value(100.0).unwrap()
        });
        
        let value1 = vf.value(100.0).unwrap();
        let value2 = handle.join().unwrap();
        
        assert_relative_eq!(value1, value2, epsilon = FINANCIAL_PRECISION);
    }
}