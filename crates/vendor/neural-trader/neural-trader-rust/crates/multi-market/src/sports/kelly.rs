//! Kelly Criterion Calculator
//!
//! Optimal bet sizing using the Kelly Criterion formula

use crate::error::{MultiMarketError, Result};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use serde::{Deserialize, Serialize};

/// Betting opportunity for Kelly Criterion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BettingOpportunity {
    /// Event identifier
    pub event_id: String,
    /// Outcome description
    pub outcome: String,
    /// Decimal odds (e.g., 2.5)
    pub odds: Decimal,
    /// Estimated win probability (0-1)
    pub win_probability: Decimal,
    /// Maximum stake allowed
    pub max_stake: Option<Decimal>,
}

/// Kelly Criterion calculation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KellyResult {
    /// Optimal stake as fraction of bankroll
    pub kelly_fraction: Decimal,
    /// Optimal stake in dollars
    pub optimal_stake: Decimal,
    /// Expected value
    pub expected_value: Decimal,
    /// Expected growth rate
    pub expected_growth: Decimal,
    /// Risk of ruin probability
    pub risk_of_ruin: Decimal,
}

/// Kelly Criterion optimizer
#[derive(Debug, Clone)]
pub struct KellyOptimizer {
    /// Total bankroll
    bankroll: Decimal,
    /// Kelly fraction multiplier (conservative: 0.25-0.5, full: 1.0)
    kelly_multiplier: Decimal,
    /// Maximum bet as percentage of bankroll
    max_bet_pct: Decimal,
    /// Minimum edge required (default: 2%)
    min_edge: Decimal,
}

impl KellyOptimizer {
    /// Create new Kelly optimizer
    ///
    /// # Arguments
    /// * `bankroll` - Total bankroll in dollars
    /// * `kelly_multiplier` - Fraction of Kelly to use (0.25 = quarter Kelly, conservative)
    pub fn new(bankroll: Decimal, kelly_multiplier: Decimal) -> Self {
        Self {
            bankroll,
            kelly_multiplier,
            max_bet_pct: dec!(0.05), // Max 5% of bankroll per bet
            min_edge: dec!(0.02),     // Minimum 2% edge
        }
    }

    /// Set maximum bet percentage
    pub fn with_max_bet_pct(mut self, pct: Decimal) -> Self {
        self.max_bet_pct = pct;
        self
    }

    /// Set minimum edge required
    pub fn with_min_edge(mut self, edge: Decimal) -> Self {
        self.min_edge = edge;
        self
    }

    /// Update bankroll
    pub fn update_bankroll(&mut self, new_bankroll: Decimal) {
        self.bankroll = new_bankroll;
    }

    /// Calculate optimal stake using Kelly Criterion
    ///
    /// Formula: f* = (bp - q) / b
    /// Where:
    /// - f* = fraction of bankroll to wager
    /// - b = net odds received (decimal odds - 1)
    /// - p = probability of winning
    /// - q = probability of losing (1 - p)
    pub fn calculate(&self, opportunity: &BettingOpportunity) -> Result<Option<KellyResult>> {
        // Validate inputs
        if opportunity.odds <= Decimal::ONE {
            return Err(MultiMarketError::ValidationError(
                "Odds must be greater than 1.0".to_string(),
            ));
        }

        if opportunity.win_probability <= Decimal::ZERO
            || opportunity.win_probability >= Decimal::ONE
        {
            return Err(MultiMarketError::ValidationError(
                "Win probability must be between 0 and 1".to_string(),
            ));
        }

        // Calculate net odds (b)
        let net_odds = opportunity.odds - Decimal::ONE;

        // Calculate probabilities
        let p = opportunity.win_probability;
        let q = Decimal::ONE - p;

        // Calculate expected value
        let expected_value = (p * net_odds) - q;

        // Check if bet has positive edge
        let edge = expected_value;
        if edge < self.min_edge {
            return Ok(None); // Not profitable enough
        }

        // Calculate Kelly fraction: f* = (bp - q) / b
        let kelly_fraction = ((net_odds * p) - q) / net_odds;

        // Apply Kelly multiplier for conservative betting
        let adjusted_kelly = kelly_fraction * self.kelly_multiplier;

        // Cap at maximum bet percentage
        let final_fraction = adjusted_kelly.min(self.max_bet_pct);

        // Don't bet if fraction is negative or too small
        if final_fraction <= dec!(0.001) {
            return Ok(None);
        }

        // Calculate optimal stake
        let mut optimal_stake = self.bankroll * final_fraction;

        // Apply max stake if specified
        if let Some(max_stake) = opportunity.max_stake {
            optimal_stake = optimal_stake.min(max_stake);
        }

        // Calculate expected growth rate (simplified)
        let expected_growth = p * ((Decimal::ONE + final_fraction * net_odds).ln())
            + q * ((Decimal::ONE - final_fraction).ln());

        // Estimate risk of ruin (simplified)
        let risk_of_ruin = if final_fraction > dec!(0.5) {
            dec!(0.5) // High risk if betting more than half Kelly
        } else {
            (final_fraction / dec!(0.5)).powi(2) * dec!(0.1)
        };

        Ok(Some(KellyResult {
            kelly_fraction: final_fraction,
            optimal_stake,
            expected_value,
            expected_growth,
            risk_of_ruin,
        }))
    }

    /// Calculate multiple betting opportunities simultaneously
    pub fn calculate_multiple(
        &self,
        opportunities: &[BettingOpportunity],
    ) -> Result<Vec<(BettingOpportunity, Option<KellyResult>)>> {
        opportunities
            .iter()
            .map(|opp| {
                let result = self.calculate(opp)?;
                Ok((opp.clone(), result))
            })
            .collect()
    }

    /// Get current bankroll
    pub fn bankroll(&self) -> Decimal {
        self.bankroll
    }

    /// Get Kelly multiplier
    pub fn kelly_multiplier(&self) -> Decimal {
        self.kelly_multiplier
    }
}

// Decimal ln approximation (Taylor series)
trait DecimalExt {
    fn ln(&self) -> Decimal;
    fn powi(&self, n: i32) -> Decimal;
}

impl DecimalExt for Decimal {
    fn ln(&self) -> Decimal {
        // Simple approximation: ln(x) ≈ 2 * (x-1)/(x+1) for x close to 1
        // For better accuracy, use Taylor series
        if *self <= Decimal::ZERO {
            return Decimal::MIN;
        }

        let x = *self;
        if x == Decimal::ONE {
            return Decimal::ZERO;
        }

        // Use natural log approximation for x close to 1
        let two = Decimal::from(2);
        let numerator = x - Decimal::ONE;
        let denominator = x + Decimal::ONE;

        two * numerator / denominator
    }

    fn powi(&self, n: i32) -> Decimal {
        let mut result = Decimal::ONE;
        let mut base = *self;
        let mut exp = n.abs();

        while exp > 0 {
            if exp % 2 == 1 {
                result *= base;
            }
            base *= base;
            exp /= 2;
        }

        if n < 0 {
            Decimal::ONE / result
        } else {
            result
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kelly_calculation() {
        let optimizer = KellyOptimizer::new(dec!(10000), dec!(0.25));

        let opportunity = BettingOpportunity {
            event_id: "test_event".to_string(),
            outcome: "Home Win".to_string(),
            odds: dec!(2.5),
            win_probability: dec!(0.45),
            max_stake: None,
        };

        let result = optimizer.calculate(&opportunity).unwrap();
        assert!(result.is_some());

        let kelly = result.unwrap();
        assert!(kelly.kelly_fraction > Decimal::ZERO);
        assert!(kelly.optimal_stake > Decimal::ZERO);
        assert!(kelly.optimal_stake <= dec!(500)); // Max 5% of 10000
    }

    #[test]
    fn test_no_edge() {
        let optimizer = KellyOptimizer::new(dec!(10000), dec!(0.25));

        let opportunity = BettingOpportunity {
            event_id: "test_event".to_string(),
            outcome: "Home Win".to_string(),
            odds: dec!(2.0),
            win_probability: dec!(0.4), // Expected value = (0.4 * 1.0) - 0.6 = -0.2 (negative)
            max_stake: None,
        };

        let result = optimizer.calculate(&opportunity).unwrap();
        assert!(result.is_none()); // Should not bet with negative edge
    }

    #[test]
    fn test_kelly_multiplier() {
        let full_kelly = KellyOptimizer::new(dec!(10000), dec!(1.0));
        let quarter_kelly = KellyOptimizer::new(dec!(10000), dec!(0.25));

        let opportunity = BettingOpportunity {
            event_id: "test_event".to_string(),
            outcome: "Home Win".to_string(),
            odds: dec!(2.5),
            win_probability: dec!(0.5),
            max_stake: None,
        };

        let full_result = full_kelly.calculate(&opportunity).unwrap().unwrap();
        let quarter_result = quarter_kelly.calculate(&opportunity).unwrap().unwrap();

        // Quarter Kelly should bet approximately 1/4 of full Kelly
        assert!(quarter_result.optimal_stake < full_result.optimal_stake);
    }

    #[test]
    fn test_max_stake_limit() {
        let optimizer = KellyOptimizer::new(dec!(10000), dec!(1.0));

        let opportunity = BettingOpportunity {
            event_id: "test_event".to_string(),
            outcome: "Home Win".to_string(),
            odds: dec!(3.0),
            win_probability: dec!(0.6),
            max_stake: Some(dec!(100)),
        };

        let result = optimizer.calculate(&opportunity).unwrap().unwrap();
        assert!(result.optimal_stake <= dec!(100));
    }

    #[test]
    fn test_invalid_odds() {
        let optimizer = KellyOptimizer::new(dec!(10000), dec!(0.25));

        let opportunity = BettingOpportunity {
            event_id: "test_event".to_string(),
            outcome: "Home Win".to_string(),
            odds: dec!(0.5), // Invalid odds < 1
            win_probability: dec!(0.5),
            max_stake: None,
        };

        assert!(optimizer.calculate(&opportunity).is_err());
    }

    #[test]
    fn test_multiple_opportunities() {
        let optimizer = KellyOptimizer::new(dec!(10000), dec!(0.25));

        let opportunities = vec![
            BettingOpportunity {
                event_id: "event1".to_string(),
                outcome: "Home Win".to_string(),
                odds: dec!(2.5),
                win_probability: dec!(0.45),
                max_stake: None,
            },
            BettingOpportunity {
                event_id: "event2".to_string(),
                outcome: "Away Win".to_string(),
                odds: dec!(3.0),
                win_probability: dec!(0.4),
                max_stake: None,
            },
        ];

        let results = optimizer.calculate_multiple(&opportunities).unwrap();
        assert_eq!(results.len(), 2);
    }
}
