//! Trading Strategies for Prediction Markets

use crate::error::Result;
use crate::prediction::polymarket::{Market, OrderSide};
use crate::types::ArbitrageOpportunity;
use chrono::Utc;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use serde::{Deserialize, Serialize};

/// Market making strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketMakingStrategy {
    /// Target spread percentage
    pub target_spread: Decimal,
    /// Inventory limits
    pub max_inventory: Decimal,
    /// Quote size
    pub quote_size: Decimal,
}

impl MarketMakingStrategy {
    pub fn new(target_spread: Decimal, max_inventory: Decimal, quote_size: Decimal) -> Self {
        Self {
            target_spread,
            max_inventory,
            quote_size,
        }
    }

    /// Generate quotes for market making
    pub fn generate_quotes(&self, mid_price: Decimal, current_inventory: Decimal) -> Result<(Decimal, Decimal)> {
        // Adjust spread based on inventory
        let inventory_adjustment = (current_inventory / self.max_inventory) * dec!(0.01);

        let bid_price = mid_price * (Decimal::ONE - self.target_spread / Decimal::from(2) - inventory_adjustment);
        let ask_price = mid_price * (Decimal::ONE + self.target_spread / Decimal::from(2) + inventory_adjustment);

        Ok((bid_price, ask_price))
    }
}

/// Arbitrage detector for prediction markets
pub struct ArbitrageDetector {
    min_profit_pct: Decimal,
}

impl ArbitrageDetector {
    pub fn new(min_profit_pct: Decimal) -> Self {
        Self { min_profit_pct }
    }

    /// Detect arbitrage between binary outcomes
    pub fn detect_binary_arbitrage(&self, market: &Market) -> Result<Option<ArbitrageOpportunity>> {
        if market.outcomes.len() != 2 {
            return Ok(None);
        }

        let prices: Vec<Decimal> = market.outcome_prices.values().copied().collect();
        let total_implied_prob: Decimal = prices.iter().sum();

        // Arbitrage exists if sum of probabilities < 1
        if total_implied_prob >= Decimal::ONE {
            return Ok(None);
        }

        let profit_pct = (Decimal::ONE - total_implied_prob) * Decimal::from(100);

        if profit_pct < self.min_profit_pct {
            return Ok(None);
        }

        let stake = dec!(1000);
        let profit = stake * (Decimal::ONE - total_implied_prob);

        Ok(Some(ArbitrageOpportunity {
            id: uuid::Uuid::new_v4().to_string(),
            market_a: market.condition_id.clone(),
            market_b: market.condition_id.clone(),
            price_a: prices[0],
            price_b: prices[1],
            profit_pct,
            profit_amount: profit,
            confidence: dec!(0.9),
            required_capital: stake,
            detected_at: Utc::now(),
            expires_at: market.end_date,
        }))
    }

    /// Detect cross-market arbitrage
    pub fn detect_cross_market_arbitrage(
        &self,
        market_a: &Market,
        market_b: &Market,
        outcome_idx: usize,
    ) -> Result<Option<ArbitrageOpportunity>> {
        let price_a = market_a
            .outcome_prices
            .values()
            .nth(outcome_idx)
            .copied()
            .unwrap_or(Decimal::ZERO);

        let price_b = market_b
            .outcome_prices
            .values()
            .nth(outcome_idx)
            .copied()
            .unwrap_or(Decimal::ZERO);

        if price_a == Decimal::ZERO || price_b == Decimal::ZERO {
            return Ok(None);
        }

        let price_diff = (price_a - price_b).abs();
        let profit_pct = (price_diff / price_a.min(price_b)) * Decimal::from(100);

        if profit_pct < self.min_profit_pct {
            return Ok(None);
        }

        let stake = dec!(1000);
        let profit = stake * price_diff;

        Ok(Some(ArbitrageOpportunity {
            id: uuid::Uuid::new_v4().to_string(),
            market_a: market_a.condition_id.clone(),
            market_b: market_b.condition_id.clone(),
            price_a,
            price_b,
            profit_pct,
            profit_amount: profit,
            confidence: dec!(0.85),
            required_capital: stake,
            detected_at: Utc::now(),
            expires_at: market_a.end_date.min(market_b.end_date),
        }))
    }
}

/// Mean reversion strategy
#[derive(Debug, Clone)]
pub struct MeanReversionStrategy {
    lookback_period: usize,
    entry_threshold: Decimal,
    exit_threshold: Decimal,
}

impl MeanReversionStrategy {
    pub fn new() -> Self {
        Self {
            lookback_period: 20,
            entry_threshold: dec!(2.0),  // 2 standard deviations
            exit_threshold: dec!(0.5),   // 0.5 standard deviations
        }
    }

    /// Check if price has reverted to mean
    pub fn should_enter(&self, current_price: Decimal, historical_prices: &[Decimal]) -> Option<OrderSide> {
        if historical_prices.len() < self.lookback_period {
            return None;
        }

        let recent_prices = &historical_prices[historical_prices.len() - self.lookback_period..];
        let mean = recent_prices.iter().sum::<Decimal>() / Decimal::from(recent_prices.len());

        let variance: Decimal = recent_prices
            .iter()
            .map(|p| (*p - mean) * (*p - mean))
            .sum::<Decimal>()
            / Decimal::from(recent_prices.len());

        let std_dev = variance.sqrt().unwrap_or(Decimal::ZERO);

        if std_dev == Decimal::ZERO {
            return None;
        }

        let z_score = (current_price - mean) / std_dev;

        if z_score > self.entry_threshold {
            Some(OrderSide::Sell)  // Price too high, sell
        } else if z_score < -self.entry_threshold {
            Some(OrderSide::Buy)   // Price too low, buy
        } else {
            None
        }
    }
}

impl Default for MeanReversionStrategy {
    fn default() -> Self {
        Self::new()
    }
}

// Helper trait for Decimal sqrt
trait DecimalSqrt {
    fn sqrt(&self) -> Option<Decimal>;
}

impl DecimalSqrt for Decimal {
    fn sqrt(&self) -> Option<Decimal> {
        if *self < Decimal::ZERO {
            return None;
        }

        if *self == Decimal::ZERO {
            return Some(Decimal::ZERO);
        }

        // Newton's method for square root
        let mut x = *self / Decimal::from(2);
        for _ in 0..10 {
            let x_next = (x + *self / x) / Decimal::from(2);
            if (x_next - x).abs() < dec!(0.0001) {
                return Some(x_next);
            }
            x = x_next;
        }

        Some(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_market_making_quotes() {
        let strategy = MarketMakingStrategy::new(dec!(0.02), dec!(1000), dec!(100));

        let (bid, ask) = strategy.generate_quotes(dec!(0.5), dec!(0)).unwrap();

        assert!(bid < dec!(0.5));
        assert!(ask > dec!(0.5));
        assert!(ask - bid > Decimal::ZERO);
    }

    #[test]
    fn test_mean_reversion() {
        let strategy = MeanReversionStrategy::new();

        let historical = vec![dec!(0.5); 20];
        let current = dec!(0.7); // High price

        let signal = strategy.should_enter(current, &historical);
        assert_eq!(signal, Some(OrderSide::Sell));
    }
}
