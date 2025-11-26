//! Statistical Arbitrage Strategy
//!
//! Quantitative strategy exploiting statistical relationships.

use crate::config::{StrategyConfig, ParameterType};

pub struct StatisticalArbitrageStrategy;

pub fn config() -> StrategyConfig {
    StrategyConfig::new(
        "statistical_arbitrage",
        "High-frequency statistical arbitrage using cross-asset correlations"
    )
    .with_sharpe_ratio(3.89)
    .with_risk_level("medium")
    .with_gpu_capable(true)
    .with_status("active")
    .with_asset_types(vec![
        "stocks".to_string(),
        "etfs".to_string(),
        "futures".to_string(),
    ])
    .with_min_capital(25000.0)
    .with_holding_period(1)
    .with_parameter("num_securities", ParameterType::Integer, 5.0, 50.0, 20.0)
    .with_parameter("correlation_window", ParameterType::Days, 20.0, 120.0, 60.0)
    .with_parameter("entry_threshold", ParameterType::Float, 1.0, 3.0, 2.0)
    .with_parameter("holding_time_minutes", ParameterType::Integer, 5.0, 120.0, 30.0)
}

impl StatisticalArbitrageStrategy {
    pub fn new() -> Self {
        Self
    }
}

impl Default for StatisticalArbitrageStrategy {
    fn default() -> Self {
        Self::new()
    }
}
