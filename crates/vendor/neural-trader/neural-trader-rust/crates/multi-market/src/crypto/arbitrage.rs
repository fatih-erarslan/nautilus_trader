//! Cross-Exchange Arbitrage for Cryptocurrency

use crate::error::Result;
use crate::types::ArbitrageOpportunity;
use chrono::Utc;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use serde::{Deserialize, Serialize};

/// Cross-exchange arbitrage opportunity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossExchangeOpportunity {
    /// Base opportunity
    pub base: ArbitrageOpportunity,
    /// Exchange A
    pub exchange_a: String,
    /// Exchange B
    pub exchange_b: String,
    /// Trading pair
    pub pair: String,
    /// Transfer time estimate (minutes)
    pub transfer_time_mins: u32,
    /// Network fees
    pub network_fees: Decimal,
}

/// Arbitrage engine
pub struct ArbitrageEngine {
    min_profit_pct: Decimal,
    max_transfer_time: u32,
}

impl ArbitrageEngine {
    pub fn new(min_profit_pct: Decimal) -> Self {
        Self {
            min_profit_pct,
            max_transfer_time: 30, // 30 minutes
        }
    }

    /// Detect arbitrage between exchanges
    pub fn detect_arbitrage(
        &self,
        pair: &str,
        price_a: Decimal,
        price_b: Decimal,
        exchange_a: &str,
        exchange_b: &str,
    ) -> Option<CrossExchangeOpportunity> {
        let price_diff = (price_a - price_b).abs();
        let lower_price = price_a.min(price_b);

        if lower_price == Decimal::ZERO {
            return None;
        }

        let profit_pct = (price_diff / lower_price) * Decimal::from(100);

        if profit_pct < self.min_profit_pct {
            return None;
        }

        let stake = dec!(10000);
        let network_fees = self.estimate_network_fees(pair);
        let profit = (stake * price_diff / lower_price) - network_fees;

        Some(CrossExchangeOpportunity {
            base: ArbitrageOpportunity {
                id: uuid::Uuid::new_v4().to_string(),
                market_a: exchange_a.to_string(),
                market_b: exchange_b.to_string(),
                price_a,
                price_b,
                profit_pct,
                profit_amount: profit,
                confidence: dec!(0.85),
                required_capital: stake,
                detected_at: Utc::now(),
                expires_at: None,
            },
            exchange_a: exchange_a.to_string(),
            exchange_b: exchange_b.to_string(),
            pair: pair.to_string(),
            transfer_time_mins: self.estimate_transfer_time(pair),
            network_fees,
        })
    }

    fn estimate_network_fees(&self, pair: &str) -> Decimal {
        if pair.contains("ETH") {
            dec!(20.0)
        } else if pair.contains("BTC") {
            dec!(5.0)
        } else {
            dec!(1.0)
        }
    }

    fn estimate_transfer_time(&self, pair: &str) -> u32 {
        if pair.contains("BTC") {
            60 // Bitcoin: ~60 mins
        } else if pair.contains("ETH") {
            15 // Ethereum: ~15 mins
        } else {
            5 // Fast chains: ~5 mins
        }
    }
}

impl Default for ArbitrageEngine {
    fn default() -> Self {
        Self::new(dec!(1.0))
    }
}
