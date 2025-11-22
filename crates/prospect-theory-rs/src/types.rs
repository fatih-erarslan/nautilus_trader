//! Core types for Prospect Theory
//!
//! This module defines the main types used throughout the prospect theory crate.

use num_traits::Float;
use serde::{Deserialize, Serialize};
use crate::errors::{ProspectTheoryError, Result};

/// Helper function to cast f64 to T
fn cast_f64<T: num_traits::Float>(value: f64) -> T {
    num_traits::cast::cast(value).unwrap()
}

/// Frame types for prospect theory
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Frame {
    /// Neutral frame - baseline reference
    Neutral,
    /// Gain frame - focus on potential gains
    Gain,
    /// Loss frame - focus on potential losses
    Loss,
    /// Survival frame - focus on avoiding losses
    Survival,
    /// Opportunity frame - focus on capturing gains
    Opportunity,
}

impl Frame {
    /// Get the bias multiplier for this frame type
    pub fn bias_multiplier<F: Float>(&self) -> F 
    where
        F: num_traits::Float,
    {
        match self {
            Frame::Neutral => cast_f64(1.0),
            Frame::Gain => cast_f64(1.1),      // 10% positive bias
            Frame::Loss => cast_f64(0.9),      // 10% negative bias
            Frame::Survival => cast_f64(0.8),  // 20% negative bias (more risk averse)
            Frame::Opportunity => cast_f64(1.2), // 20% positive bias (more risk seeking)
        }
    }
    
    /// Get the string representation of the frame
    pub fn as_str(&self) -> &'static str {
        match self {
            Frame::Neutral => "neutral",
            Frame::Gain => "gain",
            Frame::Loss => "loss",
            Frame::Survival => "survival",
            Frame::Opportunity => "opportunity",
        }
    }
    
    /// Create a frame from a string
    pub fn from_str(s: &str) -> Frame {
        match s.to_lowercase().as_str() {
            "neutral" => Frame::Neutral,
            "gain" => Frame::Gain,
            "loss" => Frame::Loss,
            "survival" => Frame::Survival,
            "opportunity" => Frame::Opportunity,
            _ => Frame::Neutral,
        }
    }
}

/// A prospect representing a set of outcomes with associated probabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Prospect<T: Float> {
    /// Possible outcomes
    pub outcomes: Vec<T>,
    /// Probabilities for each outcome
    pub probabilities: Vec<T>,
    /// Context or description
    pub context: String,
    /// Mental accounting category
    pub account: String,
    /// Ambiguity level (0.0 = no ambiguity, 1.0 = complete ambiguity)
    pub ambiguity_level: T,
    /// Reference frame for evaluation
    pub frame: Frame,
}

impl<T: Float> Prospect<T> 
where
    T: num_traits::Float + Into<f64> + std::convert::From<f64>,
{
    /// Create a new prospect
    pub fn new(outcomes: Vec<T>, probabilities: Vec<T>, context: String) -> Result<Self> {
        if outcomes.is_empty() || probabilities.is_empty() {
            return Err(ProspectTheoryError::computation_failed("Outcomes and probabilities cannot be empty"));
        }
        
        if outcomes.len() != probabilities.len() {
            return Err(ProspectTheoryError::computation_failed("Outcomes and probabilities must have the same length"));
        }
        
        // Check that probabilities sum to approximately 1.0
        let prob_sum: T = probabilities.iter().fold(T::zero(), |acc, &p| acc + p);
        let diff = (prob_sum - T::one()).abs();
        if diff > <T as From<f64>>::from(0.001) {
            return Err(ProspectTheoryError::computation_failed("Probabilities must sum to 1.0"));
        }
        
        // Check that all probabilities are non-negative
        for &prob in &probabilities {
            if prob < T::zero() {
                return Err(ProspectTheoryError::computation_failed("Probabilities must be non-negative"));
            }
        }
        
        Ok(Self {
            outcomes,
            probabilities,
            context,
            account: "default".to_string(),
            ambiguity_level: T::zero(),
            frame: Frame::Neutral,
        })
    }
    
    /// Set the frame for this prospect
    pub fn with_frame(mut self, frame: Frame) -> Self {
        self.frame = frame;
        self
    }
    
    /// Set the mental accounting category
    pub fn with_account(mut self, account: String) -> Self {
        self.account = account;
        self
    }
    
    /// Set the ambiguity level
    pub fn with_ambiguity(mut self, level: T) -> Self {
        self.ambiguity_level = level;
        self
    }
    
    /// Calculate the expected value of this prospect
    pub fn expected_value(&self) -> T {
        self.outcomes.iter()
            .zip(self.probabilities.iter())
            .fold(T::zero(), |acc, (&outcome, &prob)| acc + outcome * prob)
    }
    
    /// Calculate the variance of this prospect
    pub fn variance(&self) -> T {
        let expected = self.expected_value();
        let variance = self.outcomes.iter()
            .zip(self.probabilities.iter())
            .fold(T::zero(), |acc, (&outcome, &prob)| {
                let diff = outcome - expected;
                acc + diff * diff * prob
            });
        variance
    }
    
    /// Calculate the standard deviation
    pub fn standard_deviation(&self) -> T {
        self.variance().sqrt()
    }
    
    /// Get the minimum outcome
    pub fn min_outcome(&self) -> Option<T> {
        self.outcomes.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).copied()
    }
    
    /// Get the maximum outcome
    pub fn max_outcome(&self) -> Option<T> {
        self.outcomes.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).copied()
    }
    
    /// Get the probability of a positive outcome
    pub fn probability_of_gain(&self) -> T {
        self.outcomes.iter()
            .zip(self.probabilities.iter())
            .fold(T::zero(), |acc, (&outcome, &prob)| {
                if outcome > T::zero() {
                    acc + prob
                } else {
                    acc
                }
            })
    }
    
    /// Get the probability of a negative outcome
    pub fn probability_of_loss(&self) -> T {
        self.outcomes.iter()
            .zip(self.probabilities.iter())
            .fold(T::zero(), |acc, (&outcome, &prob)| {
                if outcome < T::zero() {
                    acc + prob
                } else {
                    acc
                }
            })
    }
    
    /// Calculate the skewness of the distribution
    pub fn skewness(&self) -> T {
        let expected = self.expected_value();
        let variance = self.variance();
        let std_dev = variance.sqrt();
        
        if std_dev == T::zero() {
            return T::zero();
        }
        
        let skewness = self.outcomes.iter()
            .zip(self.probabilities.iter())
            .fold(T::zero(), |acc, (&outcome, &prob)| {
                let standardized = (outcome - expected) / std_dev;
                acc + standardized * standardized * standardized * prob
            });
        
        skewness
    }
    
    /// Calculate the kurtosis of the distribution
    pub fn kurtosis(&self) -> T {
        let expected = self.expected_value();
        let variance = self.variance();
        let std_dev = variance.sqrt();
        
        if std_dev == T::zero() {
            return T::zero();
        }
        
        let kurtosis = self.outcomes.iter()
            .zip(self.probabilities.iter())
            .fold(T::zero(), |acc, (&outcome, &prob)| {
                let standardized = (outcome - expected) / std_dev;
                let fourth_power = standardized * standardized * standardized * standardized;
                acc + fourth_power * prob
            });
        
        kurtosis - <T as From<f64>>::from(3.0) // Excess kurtosis
    }
    
    /// Check if this prospect is valid
    pub fn is_valid(&self) -> bool {
        // Check basic constraints
        if self.outcomes.is_empty() || self.probabilities.is_empty() {
            return false;
        }
        
        if self.outcomes.len() != self.probabilities.len() {
            return false;
        }
        
        // Check probabilities sum to 1.0 (within tolerance)
        let prob_sum: T = self.probabilities.iter().fold(T::zero(), |acc, &p| acc + p);
        let diff = (prob_sum - T::one()).abs();
        if diff > <T as From<f64>>::from(0.001) {
            return false;
        }
        
        // Check all probabilities are non-negative
        for &prob in &self.probabilities {
            if prob < T::zero() {
                return false;
            }
        }
        
        // Check ambiguity level is in valid range
        if self.ambiguity_level < T::zero() || self.ambiguity_level > T::one() {
            return false;
        }
        
        true
    }
    
    /// Apply frame bias to the prospect
    pub fn apply_frame_bias(&self) -> Self {
        let bias_multiplier = self.frame.bias_multiplier::<T>();
        let adjusted_outcomes: Vec<T> = self.outcomes.iter()
            .map(|&outcome| outcome * bias_multiplier)
            .collect();
        
        let mut result = self.clone();
        result.outcomes = adjusted_outcomes;
        result
    }
    
    /// Convert to a different numeric type
    pub fn convert<U: Float>(&self) -> Prospect<U> 
    where
        U: From<f64> + Into<f64> + Copy + PartialOrd + std::fmt::Debug + std::ops::Add<Output = U> + std::ops::Mul<Output = U> + std::ops::Sub<Output = U> + std::ops::Div<Output = U>,
    {
        Prospect {
            outcomes: self.outcomes.iter().map(|&x| <U as From<f64>>::from(x.into())).collect(),
            probabilities: self.probabilities.iter().map(|&x| <U as From<f64>>::from(x.into())).collect(),
            context: self.context.clone(),
            account: self.account.clone(),
            ambiguity_level: <U as From<f64>>::from(self.ambiguity_level.into()),
            frame: self.frame.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    
    #[test]
    fn test_prospect_creation() {
        let outcomes = vec![100.0, -50.0, 0.0];
        let probabilities = vec![0.3, 0.5, 0.2];
        let context = "Test prospect".to_string();
        
        let prospect = Prospect::new(outcomes.clone(), probabilities.clone(), context.clone()).unwrap();
        
        assert_eq!(prospect.outcomes, outcomes);
        assert_eq!(prospect.probabilities, probabilities);
        assert_eq!(prospect.context, context);
        assert_eq!(prospect.account, "default");
        assert_eq!(prospect.ambiguity_level, 0.0);
        assert_eq!(prospect.frame, Frame::Neutral);
    }
    
    #[test]
    fn test_prospect_invalid_inputs() {
        // Empty outcomes
        let result = Prospect::new(vec![], vec![0.5, 0.5], "test".to_string());
        assert!(result.is_err());
        
        // Mismatched lengths
        let result = Prospect::new(vec![1.0], vec![0.5, 0.5], "test".to_string());
        assert!(result.is_err());
        
        // Probabilities don't sum to 1.0
        let result = Prospect::new(vec![1.0, 2.0], vec![0.3, 0.5], "test".to_string());
        assert!(result.is_err());
        
        // Negative probability
        let result = Prospect::new(vec![1.0, 2.0], vec![-0.1, 1.1], "test".to_string());
        assert!(result.is_err());
    }
    
    #[test]
    fn test_expected_value() {
        let outcomes = vec![100.0, -50.0, 0.0];
        let probabilities = vec![0.3, 0.5, 0.2];
        let prospect = Prospect::new(outcomes, probabilities, "test".to_string()).unwrap();
        
        let expected = 100.0 * 0.3 + (-50.0) * 0.5 + 0.0 * 0.2;
        assert_abs_diff_eq!(prospect.expected_value(), expected, epsilon = 1e-10);
    }
    
    #[test]
    fn test_variance() {
        let outcomes = vec![100.0, -50.0, 0.0];
        let probabilities = vec![0.3, 0.5, 0.2];
        let prospect = Prospect::new(outcomes, probabilities, "test".to_string()).unwrap();
        
        let expected = prospect.expected_value();
        let variance = (100.0 - expected).powi(2) * 0.3 + 
                      (-50.0 - expected).powi(2) * 0.5 + 
                      (0.0 - expected).powi(2) * 0.2;
        
        assert_abs_diff_eq!(prospect.variance(), variance, epsilon = 1e-10);
    }
    
    #[test]
    fn test_with_frame() {
        let outcomes = vec![100.0, -50.0];
        let probabilities = vec![0.5, 0.5];
        let prospect = Prospect::new(outcomes, probabilities, "test".to_string()).unwrap()
            .with_frame(Frame::Gain);
        
        assert_eq!(prospect.frame, Frame::Gain);
    }
    
    #[test]
    fn test_with_account() {
        let outcomes = vec![100.0, -50.0];
        let probabilities = vec![0.5, 0.5];
        let prospect = Prospect::new(outcomes, probabilities, "test".to_string()).unwrap()
            .with_account("trading".to_string());
        
        assert_eq!(prospect.account, "trading");
    }
    
    #[test]
    fn test_with_ambiguity() {
        let outcomes = vec![100.0, -50.0];
        let probabilities = vec![0.5, 0.5];
        let prospect = Prospect::new(outcomes, probabilities, "test".to_string()).unwrap()
            .with_ambiguity(0.3);
        
        assert_eq!(prospect.ambiguity_level, 0.3);
    }
    
    #[test]
    fn test_frame_bias_multiplier() {
        assert_eq!(Frame::Neutral.bias_multiplier::<f64>(), 1.0);
        assert_eq!(Frame::Gain.bias_multiplier::<f64>(), 1.1);
        assert_eq!(Frame::Loss.bias_multiplier::<f64>(), 0.9);
        assert_eq!(Frame::Survival.bias_multiplier::<f64>(), 0.8);
        assert_eq!(Frame::Opportunity.bias_multiplier::<f64>(), 1.2);
    }
    
    #[test]
    fn test_frame_from_str() {
        assert_eq!(Frame::from_str("neutral"), Frame::Neutral);
        assert_eq!(Frame::from_str("gain"), Frame::Gain);
        assert_eq!(Frame::from_str("loss"), Frame::Loss);
        assert_eq!(Frame::from_str("survival"), Frame::Survival);
        assert_eq!(Frame::from_str("opportunity"), Frame::Opportunity);
        assert_eq!(Frame::from_str("unknown"), Frame::Neutral);
    }
    
    #[test]
    fn test_probability_of_gain_loss() {
        let outcomes = vec![100.0, -50.0, 0.0];
        let probabilities = vec![0.3, 0.5, 0.2];
        let prospect = Prospect::new(outcomes, probabilities, "test".to_string()).unwrap();
        
        assert_abs_diff_eq!(prospect.probability_of_gain(), 0.3, epsilon = 1e-10);
        assert_abs_diff_eq!(prospect.probability_of_loss(), 0.5, epsilon = 1e-10);
    }
    
    #[test]
    fn test_min_max_outcomes() {
        let outcomes = vec![100.0, -50.0, 0.0];
        let probabilities = vec![0.3, 0.5, 0.2];
        let prospect = Prospect::new(outcomes, probabilities, "test".to_string()).unwrap();
        
        assert_eq!(prospect.min_outcome(), Some(-50.0));
        assert_eq!(prospect.max_outcome(), Some(100.0));
    }
    
    #[test]
    fn test_is_valid() {
        let outcomes = vec![100.0, -50.0];
        let probabilities = vec![0.5, 0.5];
        let prospect = Prospect::new(outcomes, probabilities, "test".to_string()).unwrap();
        
        assert!(prospect.is_valid());
        
        // Invalid ambiguity level
        let mut invalid_prospect = prospect.clone();
        invalid_prospect.ambiguity_level = 1.5;
        assert!(!invalid_prospect.is_valid());
    }
    
    #[test]
    fn test_apply_frame_bias() {
        let outcomes = vec![100.0, -50.0];
        let probabilities = vec![0.5, 0.5];
        let prospect = Prospect::new(outcomes, probabilities, "test".to_string()).unwrap()
            .with_frame(Frame::Gain);
        
        let biased_prospect = prospect.apply_frame_bias();
        assert_eq!(biased_prospect.outcomes[0], 100.0 * 1.1);
        assert_eq!(biased_prospect.outcomes[1], -50.0 * 1.1);
    }
    
    #[test]
    fn test_statistical_measures() {
        let outcomes = vec![1.0, 2.0, 3.0];
        let probabilities = vec![0.3, 0.4, 0.3];
        let prospect = Prospect::new(outcomes, probabilities, "test".to_string()).unwrap();
        
        // Test skewness and kurtosis don't panic
        let _skewness = prospect.skewness();
        let _kurtosis = prospect.kurtosis();
        let _std_dev = prospect.standard_deviation();
    }
}