//! Portfolio risk management

use crate::{Error, Result};
use crate::models::{BetPosition, RiskMetrics as BetRiskMetrics};
use rust_decimal::Decimal;

/// Portfolio risk manager for sports betting
pub struct PortfolioRiskManager {
    /// Target variance for portfolio
    target_variance: f64,
}

impl PortfolioRiskManager {
    /// Create new portfolio risk manager
    pub fn new() -> Self {
        Self {
            target_variance: 0.1,  // 10% target variance
        }
    }

    /// Calculate portfolio variance
    pub fn calculate_variance(&self, positions: &[BetPosition]) -> f64 {
        if positions.is_empty() {
            return 0.0;
        }

        let mean_stake: f64 = positions.iter()
            .filter_map(|p| p.stake.to_string().parse::<f64>().ok())
            .sum::<f64>() / positions.len() as f64;

        let variance = positions.iter()
            .filter_map(|p| p.stake.to_string().parse::<f64>().ok())
            .map(|stake| (stake - mean_stake).powi(2))
            .sum::<f64>() / positions.len() as f64;

        variance
    }

    /// Calculate expected value of portfolio
    pub fn calculate_expected_value(&self, positions: &[BetPosition], win_probabilities: &[f64]) -> Result<Decimal> {
        if positions.len() != win_probabilities.len() {
            return Err(Error::Internal("Positions and probabilities length mismatch".to_string()));
        }

        let mut total_ev = Decimal::ZERO;

        for (position, &win_prob) in positions.iter().zip(win_probabilities.iter()) {
            let win_amount = position.potential_payout - position.stake;
            let loss_amount = position.stake;

            let ev = win_amount * Decimal::from_f64_retain(win_prob).unwrap()
                - loss_amount * Decimal::from_f64_retain(1.0 - win_prob).unwrap();

            total_ev += ev;
        }

        Ok(total_ev)
    }
}

impl Default for PortfolioRiskManager {
    fn default() -> Self {
        Self::new()
    }
}
