//! Cryptocurrency Trading Strategies

use crate::error::Result;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use serde::Serialize;

/// DEX arbitrage strategy
#[derive(Debug, Clone)]
pub struct DexArbitrageStrategy {
    min_profit_usd: Decimal,
    max_slippage: Decimal,
}

impl DexArbitrageStrategy {
    pub fn new() -> Self {
        Self {
            min_profit_usd: dec!(50),
            max_slippage: dec!(0.01), // 1%
        }
    }

    pub fn should_execute(&self, profit: Decimal, slippage: Decimal) -> bool {
        profit >= self.min_profit_usd && slippage <= self.max_slippage
    }
}

/// Liquidity pool strategy
#[derive(Debug, Clone)]
pub struct LiquidityPoolStrategy {
    min_apr: Decimal,
    max_impermanent_loss: Decimal,
}

impl LiquidityPoolStrategy {
    pub fn new() -> Self {
        Self {
            min_apr: dec!(15.0),
            max_impermanent_loss: dec!(5.0), // 5%
        }
    }

    /// Calculate impermanent loss
    pub fn calculate_impermanent_loss(&self, price_ratio: Decimal) -> Decimal {
        if price_ratio <= Decimal::ZERO {
            return Decimal::ZERO;
        }

        // IL = 2 * sqrt(price_ratio) / (1 + price_ratio) - 1
        let sqrt_ratio = self.sqrt(price_ratio);
        let il = (Decimal::from(2) * sqrt_ratio) / (Decimal::ONE + price_ratio) - Decimal::ONE;
        il * Decimal::from(100)
    }

    fn sqrt(&self, x: Decimal) -> Decimal {
        if x <= Decimal::ZERO {
            return Decimal::ZERO;
        }

        let mut result = x / Decimal::from(2);
        for _ in 0..10 {
            result = (result + x / result) / Decimal::from(2);
        }
        result
    }
}

impl Default for DexArbitrageStrategy {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for LiquidityPoolStrategy {
    fn default() -> Self {
        Self::new()
    }
}
