//! Expected Value Calculator for Prediction Markets

use crate::error::Result;
use crate::prediction::polymarket::{Market, OrderSide};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use serde::{Deserialize, Serialize};

/// Expected value opportunity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EVOpportunity {
    /// Market ID
    pub market_id: String,
    /// Outcome index
    pub outcome_index: usize,
    /// Outcome name
    pub outcome: String,
    /// Current market price
    pub market_price: Decimal,
    /// Estimated true probability
    pub true_probability: Decimal,
    /// Expected value
    pub expected_value: Decimal,
    /// Expected profit per $1 bet
    pub profit_per_dollar: Decimal,
    /// Recommended side
    pub side: OrderSide,
    /// Recommended stake
    pub recommended_stake: Decimal,
    /// Confidence (0-1)
    pub confidence: Decimal,
}

/// Expected value calculator
pub struct ExpectedValueCalculator {
    /// Default bankroll for stake calculations
    bankroll: Decimal,
    /// Kelly fraction (0.25 = quarter Kelly)
    kelly_fraction: Decimal,
    /// Minimum EV threshold
    min_ev: Decimal,
}

impl ExpectedValueCalculator {
    pub fn new(bankroll: Decimal) -> Self {
        Self {
            bankroll,
            kelly_fraction: dec!(0.25),
            min_ev: dec!(0.05), // 5% minimum EV
        }
    }

    pub fn with_kelly_fraction(mut self, fraction: Decimal) -> Self {
        self.kelly_fraction = fraction;
        self
    }

    /// Calculate EV opportunities for a market
    pub fn find_opportunities(&self, market: &Market) -> Result<Vec<EVOpportunity>> {
        let mut opportunities = Vec::new();

        for (idx, (outcome, &market_price)) in market.outcome_prices.iter().enumerate() {
            // Estimate true probability (simplified - would use ML models)
            let true_prob = self.estimate_true_probability(market, outcome)?;

            // Calculate expected value
            let ev = self.calculate_ev(market_price, true_prob);

            if ev >= self.min_ev {
                let side = if true_prob > market_price {
                    OrderSide::Buy
                } else {
                    OrderSide::Sell
                };

                let stake = self.calculate_kelly_stake(market_price, true_prob);
                let _profit = ev * stake;

                opportunities.push(EVOpportunity {
                    market_id: market.condition_id.clone(),
                    outcome_index: idx,
                    outcome: outcome.clone(),
                    market_price,
                    true_probability: true_prob,
                    expected_value: ev,
                    profit_per_dollar: ev,
                    side,
                    recommended_stake: stake,
                    confidence: dec!(0.75),
                });
            }
        }

        opportunities.sort_by(|a, b| b.expected_value.cmp(&a.expected_value));
        Ok(opportunities)
    }

    fn estimate_true_probability(&self, market: &Market, outcome: &str) -> Result<Decimal> {
        // Simplified: Adjust market price based on liquidity and volume
        let default_price = dec!(0.5);
        let market_price = market.outcome_prices.get(outcome).unwrap_or(&default_price);

        let adjustment = if market.liquidity > dec!(100000) {
            Decimal::ZERO // High liquidity = trust market
        } else {
            dec!(0.05) // Low liquidity = add uncertainty
        };

        Ok(*market_price + adjustment)
    }

    fn calculate_ev(&self, market_price: Decimal, true_prob: Decimal) -> Decimal {
        // EV = (true_prob * payout) - (1 - true_prob) * stake
        // For binary outcome: payout = 1/market_price, stake = 1
        if market_price == Decimal::ZERO {
            return Decimal::ZERO;
        }

        let payout = Decimal::ONE / market_price;
        (true_prob * payout) - (Decimal::ONE - true_prob)
    }

    fn calculate_kelly_stake(&self, market_price: Decimal, true_prob: Decimal) -> Decimal {
        // Kelly Criterion: f = (bp - q) / b
        // where b = odds, p = true prob, q = 1-p
        if market_price == Decimal::ZERO {
            return Decimal::ZERO;
        }

        let odds = (Decimal::ONE / market_price) - Decimal::ONE;
        let q = Decimal::ONE - true_prob;

        if odds == Decimal::ZERO {
            return Decimal::ZERO;
        }

        let kelly = ((odds * true_prob) - q) / odds;
        let adjusted = kelly * self.kelly_fraction;

        // Cap at 5% of bankroll
        (adjusted * self.bankroll).min(self.bankroll * dec!(0.05))
    }
}

impl Default for ExpectedValueCalculator {
    fn default() -> Self {
        Self::new(dec!(10000))
    }
}
