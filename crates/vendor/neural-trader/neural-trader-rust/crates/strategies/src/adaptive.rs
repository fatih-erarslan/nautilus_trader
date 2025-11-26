//! Adaptive Multi-Strategy
//!
//! Dynamically allocates between multiple strategies based on market conditions.

use crate::config::{StrategyConfig, ParameterType};

pub struct AdaptiveStrategy;

pub fn config() -> StrategyConfig {
    StrategyConfig::new(
        "adaptive",
        "Machine learning-based strategy allocation across market regimes"
    )
    .with_sharpe_ratio(3.42)
    .with_risk_level("medium")
    .with_gpu_capable(true)
    .with_status("active")
    .with_asset_types(vec![
        "stocks".to_string(),
        "etfs".to_string(),
        "forex".to_string(),
        "crypto".to_string(),
    ])
    .with_min_capital(10000.0)
    .with_holding_period(30)
    .with_parameter("rebalance_period", ParameterType::Days, 1.0, 30.0, 7.0)
    .with_parameter("volatility_threshold", ParameterType::Percentage, 10.0, 50.0, 20.0)
    .with_parameter("regime_lookback", ParameterType::Days, 20.0, 100.0, 60.0)
    .with_parameter("strategy_count", ParameterType::Integer, 2.0, 6.0, 4.0)
}

impl AdaptiveStrategy {
    pub fn new() -> Self {
        Self
    }
}

impl Default for AdaptiveStrategy {
    fn default() -> Self {
        Self::new()
    }
}
