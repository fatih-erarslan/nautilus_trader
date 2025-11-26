//! Betting limits controller

use crate::{Error, Result};
use rust_decimal::Decimal;
use std::collections::HashMap;

/// Betting limits controller
pub struct BettingLimitsController {
    /// Per-sport limits
    sport_limits: HashMap<String, Decimal>,
    /// Global daily limit
    daily_limit: Decimal,
}

impl BettingLimitsController {
    /// Create new limits controller
    pub fn new(daily_limit: Decimal) -> Self {
        Self {
            sport_limits: HashMap::new(),
            daily_limit,
        }
    }

    /// Set limit for specific sport
    pub fn set_sport_limit(&mut self, sport: String, limit: Decimal) {
        self.sport_limits.insert(sport, limit);
    }

    /// Check if bet is within limits
    pub fn check_limit(&self, sport: &str, amount: Decimal, daily_total: Decimal) -> Result<()> {
        // Check sport-specific limit
        if let Some(&sport_limit) = self.sport_limits.get(sport) {
            if amount > sport_limit {
                return Err(Error::RiskLimitExceeded(
                    format!("Bet amount {} exceeds sport limit {} for {}", amount, sport_limit, sport)
                ));
            }
        }

        // Check daily limit
        if daily_total + amount > self.daily_limit {
            return Err(Error::RiskLimitExceeded(
                format!("Daily limit {} would be exceeded", self.daily_limit)
            ));
        }

        Ok(())
    }
}

impl Default for BettingLimitsController {
    fn default() -> Self {
        Self::new(Decimal::from(10000))
    }
}
