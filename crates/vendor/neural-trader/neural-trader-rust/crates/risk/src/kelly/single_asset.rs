//! Single asset Kelly Criterion implementation
//!
//! Optimal position sizing for a single asset based on:
//! - Win probability
//! - Average win/loss ratio
//! - Fractional Kelly for safety

use crate::{Result, RiskError};
use serde::{Deserialize, Serialize};
use tracing::{debug, info};

/// Kelly Criterion calculator for single asset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KellySingleAsset {
    /// Probability of winning trade (0.0 to 1.0)
    pub win_rate: f64,
    /// Average win amount (positive)
    pub avg_win: f64,
    /// Average loss amount (positive)
    pub avg_loss: f64,
    /// Fractional Kelly multiplier (0.0 to 1.0, typically 0.25-0.5)
    pub fractional: f64,
}

impl KellySingleAsset {
    /// Create new Kelly calculator
    ///
    /// # Arguments
    /// * `win_rate` - Probability of winning (0.0 to 1.0)
    /// * `avg_win` - Average win amount (positive)
    /// * `avg_loss` - Average loss amount (positive)
    /// * `fractional` - Fractional Kelly (0.0 to 1.0, default 0.25)
    pub fn new(win_rate: f64, avg_win: f64, avg_loss: f64, fractional: f64) -> Result<Self> {
        if !(0.0..=1.0).contains(&win_rate) {
            return Err(RiskError::KellyCriterionError(format!(
                "Win rate must be between 0 and 1, got {}",
                win_rate
            )));
        }
        if avg_win <= 0.0 {
            return Err(RiskError::KellyCriterionError(format!(
                "Average win must be positive, got {}",
                avg_win
            )));
        }
        if avg_loss <= 0.0 {
            return Err(RiskError::KellyCriterionError(format!(
                "Average loss must be positive, got {}",
                avg_loss
            )));
        }
        if !(0.0..=1.0).contains(&fractional) {
            return Err(RiskError::KellyCriterionError(format!(
                "Fractional must be between 0 and 1, got {}",
                fractional
            )));
        }

        Ok(Self {
            win_rate,
            avg_win,
            avg_loss,
            fractional,
        })
    }

    /// Create with default fractional Kelly (0.25)
    pub fn with_default_fractional(win_rate: f64, avg_win: f64, avg_loss: f64) -> Result<Self> {
        Self::new(win_rate, avg_win, avg_loss, 0.25)
    }

    /// Calculate optimal position size as fraction of bankroll
    ///
    /// Kelly formula: f* = (p * b - q) / b
    /// where:
    /// - p = win probability
    /// - q = loss probability (1 - p)
    /// - b = win/loss ratio
    ///
    /// Returns position size as fraction (0.0 to 1.0)
    pub fn calculate_fraction(&self) -> f64 {
        let win_loss_ratio = self.avg_win / self.avg_loss;
        let loss_rate = 1.0 - self.win_rate;

        // Kelly formula
        let kelly = (self.win_rate * win_loss_ratio - loss_rate) / win_loss_ratio;

        // Apply fractional Kelly and clamp to [0, 1]
        let fractional_kelly = kelly * self.fractional;
        fractional_kelly.max(0.0).min(1.0)
    }

    /// Calculate position size given bankroll
    ///
    /// # Arguments
    /// * `bankroll` - Total available capital
    ///
    /// Returns absolute position size
    pub fn calculate_position_size(&self, bankroll: f64) -> Result<f64> {
        if bankroll <= 0.0 {
            return Err(RiskError::KellyCriterionError(format!(
                "Bankroll must be positive, got {}",
                bankroll
            )));
        }

        let fraction = self.calculate_fraction();
        let position_size = bankroll * fraction;

        debug!(
            "Kelly calculation: win_rate={:.3}, avg_win={:.2}, avg_loss={:.2}, fraction={:.3}, position_size={:.2}",
            self.win_rate, self.avg_win, self.avg_loss, fraction, position_size
        );

        Ok(position_size)
    }

    /// Calculate position size with constraints
    ///
    /// # Arguments
    /// * `bankroll` - Total available capital
    /// * `max_position` - Maximum position size
    /// * `min_position` - Minimum position size
    pub fn calculate_with_constraints(
        &self,
        bankroll: f64,
        max_position: f64,
        min_position: f64,
    ) -> Result<f64> {
        let position = self.calculate_position_size(bankroll)?;
        let constrained = position.max(min_position).min(max_position);

        if constrained != position {
            debug!(
                "Position constrained: calculated={:.2}, constrained={:.2}",
                position, constrained
            );
        }

        Ok(constrained)
    }

    /// Calculate expected growth rate (geometric expectation)
    pub fn expected_growth_rate(&self) -> f64 {
        let fraction = self.calculate_fraction();
        let win_outcome = 1.0 + fraction * (self.avg_win / self.avg_loss);
        let loss_outcome = 1.0 - fraction;

        self.win_rate * win_outcome.ln() + (1.0 - self.win_rate) * loss_outcome.ln()
    }

    /// Estimate from historical trades
    ///
    /// # Arguments
    /// * `returns` - Historical trade returns (positive for wins, negative for losses)
    pub fn from_historical_returns(returns: &[f64], fractional: f64) -> Result<Self> {
        if returns.is_empty() {
            return Err(RiskError::InsufficientData {
                message: "No historical returns provided".to_string(),
                required: 1,
                available: 0,
            });
        }

        let wins: Vec<f64> = returns.iter().filter(|&&r| r > 0.0).copied().collect();
        let losses: Vec<f64> = returns.iter().filter(|&&r| r < 0.0).map(|&r| -r).collect();

        if wins.is_empty() || losses.is_empty() {
            return Err(RiskError::KellyCriterionError(
                "Need both winning and losing trades for Kelly calculation".to_string(),
            ));
        }

        let win_rate = wins.len() as f64 / returns.len() as f64;
        let avg_win = wins.iter().sum::<f64>() / wins.len() as f64;
        let avg_loss = losses.iter().sum::<f64>() / losses.len() as f64;

        info!(
            "Kelly from historical: {} trades, win_rate={:.3}, avg_win={:.2}, avg_loss={:.2}",
            returns.len(),
            win_rate,
            avg_win,
            avg_loss
        );

        Self::new(win_rate, avg_win, avg_loss, fractional)
    }

    /// Get recommended fractional Kelly based on risk tolerance
    pub fn recommended_fractional(risk_tolerance: RiskTolerance) -> f64 {
        match risk_tolerance {
            RiskTolerance::Conservative => 0.10,
            RiskTolerance::Moderate => 0.25,
            RiskTolerance::Aggressive => 0.50,
            RiskTolerance::FullKelly => 1.00,
        }
    }
}

/// Risk tolerance levels for fractional Kelly
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum RiskTolerance {
    /// 10% Kelly - very conservative
    Conservative,
    /// 25% Kelly - moderate (recommended)
    Moderate,
    /// 50% Kelly - aggressive
    Aggressive,
    /// 100% Kelly - full Kelly (not recommended)
    FullKelly,
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_kelly_basic() {
        // Win rate 60%, avg win $2, avg loss $1
        let kelly = KellySingleAsset::new(0.6, 2.0, 1.0, 0.25).unwrap();
        let fraction = kelly.calculate_fraction();

        // Full Kelly: (0.6 * 2 - 0.4) / 2 = 0.4
        // Quarter Kelly: 0.4 * 0.25 = 0.1
        assert_relative_eq!(fraction, 0.1, epsilon = 0.001);
    }

    #[test]
    fn test_kelly_position_size() {
        let kelly = KellySingleAsset::new(0.55, 1.5, 1.0, 0.5).unwrap();
        let bankroll = 10000.0;
        let position = kelly.calculate_position_size(bankroll).unwrap();

        // Should be positive and less than bankroll
        assert!(position > 0.0);
        assert!(position <= bankroll);
    }

    #[test]
    fn test_kelly_with_constraints() {
        let kelly = KellySingleAsset::new(0.6, 2.0, 1.0, 1.0).unwrap();
        let bankroll = 10000.0;
        let max_position = 2000.0;

        let position = kelly
            .calculate_with_constraints(bankroll, max_position, 0.0)
            .unwrap();

        assert!(position <= max_position);
    }

    #[test]
    fn test_kelly_from_historical() {
        let returns = vec![1.0, -0.5, 2.0, -0.5, 1.5, -1.0, 3.0, -0.5];
        let kelly = KellySingleAsset::from_historical_returns(&returns, 0.25).unwrap();

        assert!(kelly.win_rate > 0.0 && kelly.win_rate < 1.0);
        assert!(kelly.avg_win > 0.0);
        assert!(kelly.avg_loss > 0.0);
    }

    #[test]
    fn test_kelly_invalid_inputs() {
        assert!(KellySingleAsset::new(1.5, 1.0, 1.0, 0.25).is_err());
        assert!(KellySingleAsset::new(0.5, -1.0, 1.0, 0.25).is_err());
        assert!(KellySingleAsset::new(0.5, 1.0, -1.0, 0.25).is_err());
        assert!(KellySingleAsset::new(0.5, 1.0, 1.0, 1.5).is_err());
    }

    #[test]
    fn test_expected_growth_rate() {
        let kelly = KellySingleAsset::new(0.6, 2.0, 1.0, 0.25).unwrap();
        let growth = kelly.expected_growth_rate();

        // Should be positive for favorable odds
        assert!(growth > 0.0);
    }

    #[test]
    fn test_risk_tolerance_fractional() {
        assert_eq!(
            KellySingleAsset::recommended_fractional(RiskTolerance::Conservative),
            0.10
        );
        assert_eq!(
            KellySingleAsset::recommended_fractional(RiskTolerance::Moderate),
            0.25
        );
        assert_eq!(
            KellySingleAsset::recommended_fractional(RiskTolerance::Aggressive),
            0.50
        );
        assert_eq!(
            KellySingleAsset::recommended_fractional(RiskTolerance::FullKelly),
            1.00
        );
    }
}
