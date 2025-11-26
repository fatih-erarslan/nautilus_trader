//! Gas Optimization and MEV Protection

use crate::error::Result;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use serde::{Deserialize, Serialize};

/// Gas estimate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GasEstimate {
    pub gas_limit: u64,
    pub gas_price_gwei: Decimal,
    pub total_cost_eth: Decimal,
    pub total_cost_usd: Decimal,
}

/// MEV protection level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MevProtection {
    None,
    Basic,
    Full,
}

/// Gas optimizer
pub struct GasOptimizer {
    eth_price_usd: Decimal,
}

impl GasOptimizer {
    pub fn new(eth_price_usd: Decimal) -> Self {
        Self { eth_price_usd }
    }

    /// Estimate gas cost
    pub fn estimate_gas(&self, operation: &str) -> GasEstimate {
        let (gas_limit, gas_price_gwei) = match operation {
            "swap" => (150000, dec!(30)),
            "add_liquidity" => (200000, dec!(30)),
            "remove_liquidity" => (180000, dec!(30)),
            "claim_rewards" => (100000, dec!(30)),
            _ => (100000, dec!(30)),
        };

        let total_cost_eth = Decimal::from(gas_limit) * gas_price_gwei / dec!(1_000_000_000);
        let total_cost_usd = total_cost_eth * self.eth_price_usd;

        GasEstimate {
            gas_limit,
            gas_price_gwei,
            total_cost_eth,
            total_cost_usd,
        }
    }

    /// Optimize gas price for transaction
    pub fn optimize_gas_price(&self, urgency: &str) -> Decimal {
        match urgency {
            "low" => dec!(15),      // 15 gwei
            "medium" => dec!(30),   // 30 gwei
            "high" => dec!(50),     // 50 gwei
            "instant" => dec!(100), // 100 gwei
            _ => dec!(30),
        }
    }

    /// Enable MEV protection
    pub fn protect_from_mev(&self, level: MevProtection) -> bool {
        match level {
            MevProtection::None => false,
            MevProtection::Basic => true,  // Use Flashbots
            MevProtection::Full => true,   // Use private RPC + Flashbots
        }
    }
}

impl Default for GasOptimizer {
    fn default() -> Self {
        Self::new(dec!(3000)) // $3000 ETH price
    }
}
