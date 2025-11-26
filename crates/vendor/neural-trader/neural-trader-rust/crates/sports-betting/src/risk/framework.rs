//! Comprehensive risk management framework

use crate::{Error, Result};
use crate::models::{BetPosition, RiskMetrics};
use rust_decimal::Decimal;
use std::sync::Arc;
use parking_lot::RwLock;

/// Risk management framework coordinating all risk components
pub struct RiskFramework {
    config: Arc<RwLock<RiskConfig>>,
    metrics: Arc<RwLock<RiskMetrics>>,
}

/// Risk configuration
#[derive(Debug, Clone)]
pub struct RiskConfig {
    /// Maximum bet size as percentage of bankroll
    pub max_bet_percentage: f64,
    /// Maximum number of concurrent bets
    pub max_concurrent_bets: usize,
    /// Maximum total exposure as percentage of bankroll
    pub max_exposure_percentage: f64,
    /// Minimum odds for acceptance
    pub min_odds: f64,
    /// Maximum odds for acceptance
    pub max_odds: f64,
    /// Use Kelly criterion for sizing
    pub use_kelly_criterion: bool,
    /// Kelly multiplier (e.g., 0.5 for half-Kelly)
    pub kelly_multiplier: f64,
}

impl Default for RiskConfig {
    fn default() -> Self {
        Self {
            max_bet_percentage: 0.05,  // 5% of bankroll
            max_concurrent_bets: 10,
            max_exposure_percentage: 0.25,  // 25% of bankroll
            min_odds: 1.5,
            max_odds: 20.0,
            use_kelly_criterion: true,
            kelly_multiplier: 0.5,  // Half-Kelly for safety
        }
    }
}

impl RiskFramework {
    /// Create a new risk framework
    pub fn new() -> Self {
        Self {
            config: Arc::new(RwLock::new(RiskConfig::default())),
            metrics: Arc::new(RwLock::new(RiskMetrics {
                total_exposure: Decimal::ZERO,
                max_bet_size: Decimal::ZERO,
                active_bets: 0,
                win_rate: 0.0,
                avg_odds: 0.0,
                variance: 0.0,
                expected_value: Decimal::ZERO,
                kelly_fraction: 0.0,
            })),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: RiskConfig) -> Self {
        Self {
            config: Arc::new(RwLock::new(config)),
            metrics: Arc::new(RwLock::new(RiskMetrics {
                total_exposure: Decimal::ZERO,
                max_bet_size: Decimal::ZERO,
                active_bets: 0,
                win_rate: 0.0,
                avg_odds: 0.0,
                variance: 0.0,
                expected_value: Decimal::ZERO,
                kelly_fraction: 0.0,
            })),
        }
    }

    /// Validate a proposed bet against risk limits
    pub fn validate_bet(
        &self,
        bet: &BetPosition,
        bankroll: Decimal,
        active_positions: &[BetPosition],
    ) -> Result<()> {
        let config = self.config.read();

        // Check bet size limit
        let max_bet = bankroll * Decimal::from_f64_retain(config.max_bet_percentage)
            .ok_or_else(|| Error::Internal("Invalid max bet percentage".to_string()))?;

        if bet.stake > max_bet {
            return Err(Error::RiskLimitExceeded(
                format!("Bet size {} exceeds maximum {}", bet.stake, max_bet)
            ));
        }

        // Check concurrent bet limit
        if active_positions.len() >= config.max_concurrent_bets {
            return Err(Error::RiskLimitExceeded(
                format!("Maximum concurrent bets ({}) reached", config.max_concurrent_bets)
            ));
        }

        // Check total exposure
        let current_exposure: Decimal = active_positions.iter()
            .map(|p| p.stake)
            .sum();
        let total_exposure = current_exposure + bet.stake;
        let max_exposure = bankroll * Decimal::from_f64_retain(config.max_exposure_percentage)
            .ok_or_else(|| Error::Internal("Invalid exposure percentage".to_string()))?;

        if total_exposure > max_exposure {
            return Err(Error::RiskLimitExceeded(
                format!("Total exposure {} exceeds maximum {}", total_exposure, max_exposure)
            ));
        }

        // Check odds range
        let odds_f64 = bet.odds.to_string().parse::<f64>()
            .map_err(|_| Error::Internal("Invalid odds format".to_string()))?;

        if odds_f64 < config.min_odds || odds_f64 > config.max_odds {
            return Err(Error::RiskLimitExceeded(
                format!("Odds {} outside acceptable range [{}, {}]",
                    odds_f64, config.min_odds, config.max_odds)
            ));
        }

        Ok(())
    }

    /// Calculate recommended bet size using Kelly criterion
    pub fn calculate_kelly_size(
        &self,
        win_probability: f64,
        odds: f64,
        bankroll: Decimal,
    ) -> Result<Decimal> {
        if win_probability <= 0.0 || win_probability >= 1.0 {
            return Err(Error::ConfigError(
                "Win probability must be between 0 and 1".to_string()
            ));
        }

        if odds < 1.0 {
            return Err(Error::ConfigError(
                "Odds must be greater than 1.0".to_string()
            ));
        }

        let config = self.config.read();

        // Kelly formula: f = (bp - q) / b
        // where f = fraction of bankroll to bet
        //       b = decimal odds - 1
        //       p = probability of winning
        //       q = probability of losing (1 - p)
        let b = odds - 1.0;
        let p = win_probability;
        let q = 1.0 - p;
        let kelly_fraction = (b * p - q) / b;

        // Apply Kelly multiplier for safety
        let adjusted_fraction = kelly_fraction * config.kelly_multiplier;

        // Ensure non-negative and within limits
        let final_fraction = adjusted_fraction.max(0.0).min(config.max_bet_percentage);

        let bet_size = bankroll * Decimal::from_f64_retain(final_fraction)
            .ok_or_else(|| Error::Internal("Invalid Kelly fraction".to_string()))?;

        Ok(bet_size)
    }

    /// Update risk metrics based on current portfolio
    pub fn update_metrics(&self, positions: &[BetPosition], _bankroll: Decimal) {
        let mut metrics = self.metrics.write();

        let active_positions: Vec<_> = positions.iter()
            .filter(|p| p.status == crate::models::BetStatus::Active)
            .collect();

        metrics.active_bets = active_positions.len();
        metrics.total_exposure = active_positions.iter()
            .map(|p| p.stake)
            .sum();

        if let Some(max) = active_positions.iter().map(|p| p.stake).max() {
            metrics.max_bet_size = max;
        }

        // Calculate win rate from settled positions
        let settled: Vec<_> = positions.iter()
            .filter(|p| matches!(p.status,
                crate::models::BetStatus::Won |
                crate::models::BetStatus::Lost |
                crate::models::BetStatus::Push))
            .collect();

        if !settled.is_empty() {
            let wins = settled.iter().filter(|p| p.status == crate::models::BetStatus::Won).count();
            metrics.win_rate = wins as f64 / settled.len() as f64;
        }

        // Calculate average odds
        if !active_positions.is_empty() {
            let total_odds: f64 = active_positions.iter()
                .filter_map(|p| p.odds.to_string().parse::<f64>().ok())
                .sum();
            metrics.avg_odds = total_odds / active_positions.len() as f64;
        }
    }

    /// Get current risk metrics
    pub fn get_metrics(&self) -> RiskMetrics {
        self.metrics.read().clone()
    }

    /// Set maximum bet size percentage
    pub fn set_max_bet_percentage(&self, percentage: f64) -> Result<()> {
        if percentage <= 0.0 || percentage > 1.0 {
            return Err(Error::ConfigError(
                "Bet percentage must be between 0 and 1".to_string()
            ));
        }

        let mut config = self.config.write();
        config.max_bet_percentage = percentage;
        Ok(())
    }

    /// Set Kelly multiplier
    pub fn set_kelly_multiplier(&self, multiplier: f64) -> Result<()> {
        if multiplier <= 0.0 || multiplier > 1.0 {
            return Err(Error::ConfigError(
                "Kelly multiplier must be between 0 and 1".to_string()
            ));
        }

        let mut config = self.config.write();
        config.kelly_multiplier = multiplier;
        Ok(())
    }
}

impl Default for RiskFramework {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{BetStatus, BetPosition};
    use uuid::Uuid;
    use chrono::Utc;

    fn create_test_bet(stake: f64, odds: f64) -> BetPosition {
        BetPosition {
            id: Uuid::new_v4(),
            sport: "Football".to_string(),
            event: "Team A vs Team B".to_string(),
            bet_type: "Moneyline".to_string(),
            selection: "Team A".to_string(),
            odds: Decimal::from_f64_retain(odds).unwrap(),
            stake: Decimal::from_f64_retain(stake).unwrap(),
            potential_payout: Decimal::from_f64_retain(stake * odds).unwrap(),
            status: BetStatus::Active,
            placed_at: Utc::now(),
            settled_at: None,
            actual_payout: None,
        }
    }

    #[test]
    fn test_validate_bet_within_limits() {
        let framework = RiskFramework::new();
        let bankroll = Decimal::from(10000);
        let bet = create_test_bet(400.0, 2.0);  // 4% of bankroll
        let active_positions = vec![];

        assert!(framework.validate_bet(&bet, bankroll, &active_positions).is_ok());
    }

    #[test]
    fn test_validate_bet_exceeds_size_limit() {
        let framework = RiskFramework::new();
        let bankroll = Decimal::from(10000);
        let bet = create_test_bet(600.0, 2.0);  // 6% of bankroll (exceeds 5% limit)
        let active_positions = vec![];

        assert!(framework.validate_bet(&bet, bankroll, &active_positions).is_err());
    }

    #[test]
    fn test_kelly_criterion_calculation() {
        let framework = RiskFramework::new();
        let bankroll = Decimal::from(10000);

        // Example: 60% win probability, 2.0 odds
        let kelly_size = framework.calculate_kelly_size(0.6, 2.0, bankroll).unwrap();

        // Kelly formula: f = (bp - q) / b = (1.0 * 0.6 - 0.4) / 1.0 = 0.2
        // With 0.5 multiplier: 0.1 (10% of bankroll)
        assert!(kelly_size > Decimal::ZERO);
        assert!(kelly_size <= bankroll * Decimal::from_f64_retain(0.1).unwrap());
    }

    #[test]
    fn test_metrics_update() {
        let framework = RiskFramework::new();
        let bankroll = Decimal::from(10000);

        let positions = vec![
            create_test_bet(400.0, 2.0),
            create_test_bet(300.0, 1.8),
        ];

        framework.update_metrics(&positions, bankroll);
        let metrics = framework.get_metrics();

        assert_eq!(metrics.active_bets, 2);
        assert_eq!(metrics.total_exposure, Decimal::from(700));
    }
}
