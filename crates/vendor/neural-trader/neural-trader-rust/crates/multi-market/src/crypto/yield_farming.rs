//! Yield Farming Strategies

use crate::error::Result;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use serde::{Deserialize, Serialize};

/// Farming pool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FarmingPool {
    pub pool_id: String,
    pub name: String,
    pub protocol: String,
    pub token_a: String,
    pub token_b: String,
    pub apr: Decimal,
    pub tvl: Decimal,
    pub rewards_token: String,
}

/// Yield farming strategy
pub struct YieldFarmingStrategy {
    min_apr: Decimal,
    max_risk: u8,
}

impl YieldFarmingStrategy {
    pub fn new() -> Self {
        Self {
            min_apr: dec!(20.0),
            max_risk: 5,
        }
    }

    /// Find best farming pools
    pub fn find_best_pools(&self, pools: &[FarmingPool]) -> Vec<FarmingPool> {
        let mut filtered: Vec<_> = pools
            .iter()
            .filter(|p| p.apr >= self.min_apr)
            .cloned()
            .collect();

        filtered.sort_by(|a, b| b.apr.cmp(&a.apr));
        filtered
    }
}

/// Reward calculator
pub struct RewardCalculator;

impl RewardCalculator {
    pub fn calculate_daily_rewards(&self, pool: &FarmingPool, invested: Decimal) -> Decimal {
        let daily_apr = pool.apr / Decimal::from(365);
        invested * daily_apr / Decimal::from(100)
    }
}

impl Default for YieldFarmingStrategy {
    fn default() -> Self {
        Self::new()
    }
}
