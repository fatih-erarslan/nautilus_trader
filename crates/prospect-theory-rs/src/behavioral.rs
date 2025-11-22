//! Behavioral economics factors implementation
//! 
//! Core behavioral economics components for Prospect Theory including:
//! - Loss aversion mechanisms
//! - Probability weighting biases
//! - Mental accounting effects

use serde::{Deserialize, Serialize};

/// Behavioral factors that influence decision making
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehavioralFactors {
    /// Loss aversion impact (-2.0 to 2.0, where negative indicates loss domain)
    pub loss_aversion_impact: f64,
    /// Probability weighting bias (-1.0 to 1.0)
    pub probability_weighting_bias: f64,
    /// Mental accounting bias (-1.0 to 1.0)
    pub mental_accounting_bias: f64,
}

impl Default for BehavioralFactors {
    fn default() -> Self {
        Self {
            loss_aversion_impact: 0.0,
            probability_weighting_bias: 0.0,
            mental_accounting_bias: 0.0,
        }
    }
}

impl BehavioralFactors {
    /// Create new behavioral factors
    pub fn new(loss_aversion: f64, probability_weighting: f64, mental_accounting: f64) -> Self {
        Self {
            loss_aversion_impact: loss_aversion.clamp(-2.0, 2.0),
            probability_weighting_bias: probability_weighting.clamp(-1.0, 1.0),
            mental_accounting_bias: mental_accounting.clamp(-1.0, 1.0),
        }
    }
    
    /// Calculate combined behavioral impact
    pub fn combined_impact(&self) -> f64 {
        (self.loss_aversion_impact * 0.5 + 
         self.probability_weighting_bias * 0.3 + 
         self.mental_accounting_bias * 0.2).clamp(-1.0, 1.0)
    }
}