//! Tail risk analysis module
//!
//! Provides comprehensive tail risk measurement and analysis

use serde::{Deserialize, Serialize};

/// Comprehensive tail risk analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TailRiskAnalysis {
    /// Probability of extreme events
    pub extreme_event_probability: f64,
    /// Expected loss in tail scenarios
    pub expected_tail_loss: f64,
    /// Confidence level for analysis
    pub confidence_level: f64,
    /// Overall tail risk score
    pub tail_risk_score: f64,
    /// Value at Risk (95%)
    pub var_95: f64,
    /// Conditional Value at Risk (95%)
    pub cvar_95: f64,
    /// Maximum expected drawdown
    pub maximum_drawdown: f64,
}

impl Default for TailRiskAnalysis {
    fn default() -> Self {
        Self {
            extreme_event_probability: 0.05,
            expected_tail_loss: -0.15,
            confidence_level: 0.95,
            tail_risk_score: 0.1,
            var_95: -0.08,
            cvar_95: -0.12,
            maximum_drawdown: -0.20,
        }
    }
}
