//! Alpha Factor Management for AgentDB
//!
//! Track, validate, and decay alpha factors over time.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Alpha factor definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlphaFactor {
    /// Unique identifier
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// Description of the signal
    pub description: String,
    /// Factor category (momentum, mean-reversion, value, etc.)
    pub category: AlphaCategory,
    /// Current estimated Information Coefficient (IC)
    pub ic: f64,
    /// IC standard deviation
    pub ic_std: f64,
    /// T-statistic for IC significance
    pub ic_t_stat: f64,
    /// Sharpe ratio from factor returns
    pub sharpe: f64,
    /// Average turnover
    pub turnover: f64,
    /// Correlation with other factors
    pub factor_correlations: HashMap<String, f64>,
    /// Decay half-life in days
    pub decay_half_life: f64,
    /// Number of observations
    pub observations: usize,
    /// Last updated timestamp
    pub last_updated: i64,
    /// Factor formula/code
    pub formula: Option<String>,
}

/// Alpha factor category
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlphaCategory {
    /// Price momentum signals
    Momentum,
    /// Mean reversion signals
    MeanReversion,
    /// Value-based signals
    Value,
    /// Volatility signals
    Volatility,
    /// Volume/liquidity signals
    Volume,
    /// Order flow signals
    OrderFlow,
    /// Cross-asset signals
    CrossAsset,
    /// Sentiment signals
    Sentiment,
    /// Machine learning signals
    MachineLearning,
    /// Other/composite
    Other,
}

/// Alpha factor performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlphaPerformance {
    /// Factor ID
    pub factor_id: String,
    /// Period start
    pub period_start: i64,
    /// Period end
    pub period_end: i64,
    /// Realized IC for period
    pub realized_ic: f64,
    /// Realized returns
    pub returns: f64,
    /// Hit rate (% correct direction)
    pub hit_rate: f64,
    /// Average position holding period
    pub avg_holding_period: f64,
}

/// Alpha decay tracker
#[derive(Debug, Clone)]
pub struct AlphaDecayTracker {
    /// Rolling IC values
    ic_history: Vec<(i64, f64)>,
    /// Decay estimation window (in samples)
    window_size: usize,
}

impl AlphaDecayTracker {
    /// Create new decay tracker
    pub fn new(window_size: usize) -> Self {
        Self {
            ic_history: Vec::with_capacity(window_size),
            window_size,
        }
    }

    /// Add IC observation
    pub fn add_observation(&mut self, timestamp: i64, ic: f64) {
        self.ic_history.push((timestamp, ic));
        if self.ic_history.len() > self.window_size {
            self.ic_history.remove(0);
        }
    }

    /// Estimate decay half-life using exponential fit
    pub fn estimate_half_life(&self) -> Option<f64> {
        if self.ic_history.len() < 10 {
            return None;
        }

        // Simple exponential decay estimation: IC(t) = IC_0 * exp(-λt)
        // Half-life = ln(2) / λ
        
        let n = self.ic_history.len();
        let t0 = self.ic_history[0].0;
        
        // Log-linear regression to estimate λ
        let mut sum_t = 0.0;
        let mut sum_log_ic = 0.0;
        let mut sum_t_log_ic = 0.0;
        let mut sum_t2 = 0.0;
        let mut valid_count = 0;

        for (t, ic) in &self.ic_history {
            if *ic > 0.0 {
                let t_days = (*t - t0) as f64 / 86400.0;
                let log_ic = ic.ln();
                sum_t += t_days;
                sum_log_ic += log_ic;
                sum_t_log_ic += t_days * log_ic;
                sum_t2 += t_days * t_days;
                valid_count += 1;
            }
        }

        if valid_count < 5 {
            return None;
        }

        let n = valid_count as f64;
        let lambda = (n * sum_t_log_ic - sum_t * sum_log_ic) / (n * sum_t2 - sum_t * sum_t);
        
        if lambda < 0.0 {
            // Positive lambda means decay
            Some(0.693 / (-lambda)) // ln(2) ≈ 0.693
        } else {
            None // No decay detected
        }
    }

    /// Check if alpha is still valid (above threshold)
    pub fn is_valid(&self, ic_threshold: f64) -> bool {
        if let Some(&(_, latest_ic)) = self.ic_history.last() {
            latest_ic.abs() > ic_threshold
        } else {
            false
        }
    }
}

/// Alpha portfolio optimizer
#[derive(Debug, Clone)]
pub struct AlphaPortfolio {
    /// Active factors
    pub factors: Vec<AlphaFactor>,
    /// Factor weights
    pub weights: HashMap<String, f64>,
    /// Target IC
    pub target_ic: f64,
    /// Maximum factor correlation
    pub max_correlation: f64,
}

impl AlphaPortfolio {
    /// Create new alpha portfolio
    pub fn new(target_ic: f64, max_correlation: f64) -> Self {
        Self {
            factors: Vec::new(),
            weights: HashMap::new(),
            target_ic,
            max_correlation,
        }
    }

    /// Add factor to portfolio
    pub fn add_factor(&mut self, factor: AlphaFactor) -> bool {
        // Check correlation with existing factors
        for existing in &self.factors {
            if let Some(&corr) = factor.factor_correlations.get(&existing.id) {
                if corr.abs() > self.max_correlation {
                    return false; // Too correlated
                }
            }
        }

        // Check IC threshold
        if factor.ic.abs() < self.target_ic {
            return false; // Below IC threshold
        }

        self.factors.push(factor);
        self.rebalance_weights();
        true
    }

    /// Remove factor from portfolio
    pub fn remove_factor(&mut self, factor_id: &str) {
        self.factors.retain(|f| f.id != factor_id);
        self.weights.remove(factor_id);
        self.rebalance_weights();
    }

    /// Rebalance factor weights (IC-weighted)
    fn rebalance_weights(&mut self) {
        let total_ic: f64 = self.factors.iter().map(|f| f.ic.abs()).sum();
        
        if total_ic > 0.0 {
            for factor in &self.factors {
                let weight = factor.ic.abs() / total_ic;
                self.weights.insert(factor.id.clone(), weight);
            }
        }
    }

    /// Get combined signal for a position
    pub fn get_combined_signal(&self, factor_signals: &HashMap<String, f64>) -> f64 {
        let mut combined = 0.0;
        
        for (factor_id, weight) in &self.weights {
            if let Some(&signal) = factor_signals.get(factor_id) {
                combined += weight * signal;
            }
        }
        
        combined
    }

    /// Decay all factors by elapsed time
    pub fn apply_decay(&mut self, elapsed_days: f64) {
        for factor in &mut self.factors {
            if factor.decay_half_life > 0.0 {
                let decay_factor = 0.5_f64.powf(elapsed_days / factor.decay_half_life);
                factor.ic *= decay_factor;
            }
        }
        
        // Remove decayed factors
        self.factors.retain(|f| f.ic.abs() > self.target_ic * 0.5);
        self.rebalance_weights();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alpha_decay_estimation() {
        let mut tracker = AlphaDecayTracker::new(100);
        
        // Simulate exponential decay with half-life of 30 days
        let half_life = 30.0;
        let lambda = 0.693 / half_life;
        let ic_0 = 0.05;
        
        for day in 0..60 {
            let timestamp = day * 86400;
            let ic = ic_0 * (-lambda * day as f64).exp();
            tracker.add_observation(timestamp, ic);
        }
        
        let estimated = tracker.estimate_half_life();
        assert!(estimated.is_some());
        // Should be close to 30 days
        let estimated_hl = estimated.unwrap();
        assert!((estimated_hl - 30.0).abs() < 5.0);
    }

    #[test]
    fn test_alpha_portfolio_correlation_filter() {
        let mut portfolio = AlphaPortfolio::new(0.02, 0.5);
        
        let factor1 = AlphaFactor {
            id: "momentum".into(),
            name: "Price Momentum".into(),
            description: "12-1 month momentum".into(),
            category: AlphaCategory::Momentum,
            ic: 0.05,
            ic_std: 0.02,
            ic_t_stat: 2.5,
            sharpe: 1.2,
            turnover: 0.3,
            factor_correlations: HashMap::new(),
            decay_half_life: 60.0,
            observations: 1000,
            last_updated: 0,
            formula: None,
        };
        
        assert!(portfolio.add_factor(factor1.clone()));
        
        // Try to add highly correlated factor
        let mut factor2 = factor1.clone();
        factor2.id = "momentum_short".into();
        factor2.factor_correlations.insert("momentum".into(), 0.9); // High correlation
        
        assert!(!portfolio.add_factor(factor2)); // Should be rejected
    }
}
