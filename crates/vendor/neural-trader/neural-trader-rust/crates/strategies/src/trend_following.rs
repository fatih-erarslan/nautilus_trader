//! Trend Following Strategy
//!
//! Classic trend-following using moving averages and breakouts.

use crate::config::{StrategyConfig, ParameterType};

pub struct TrendFollowingStrategy;

pub fn config() -> StrategyConfig {
    StrategyConfig::new(
        "trend_following",
        "Follow established trends using moving average crossovers"
    )
    .with_sharpe_ratio(2.15)
    .with_risk_level("medium")
    .with_gpu_capable(false)
    .with_status("active")
    .with_asset_types(vec![
        "stocks".to_string(),
        "commodities".to_string(),
        "forex".to_string(),
    ])
    .with_min_capital(3000.0)
    .with_holding_period(14)
    .with_parameter("fast_ma_period", ParameterType::Days, 5.0, 50.0, 20.0)
    .with_parameter("slow_ma_period", ParameterType::Days, 20.0, 200.0, 50.0)
    .with_parameter("atr_multiplier", ParameterType::Float, 1.0, 3.0, 2.0)
    .with_parameter("min_trend_strength", ParameterType::Percentage, 1.0, 5.0, 2.0)
}

impl TrendFollowingStrategy {
    pub fn new() -> Self {
        Self
    }
}

impl Default for TrendFollowingStrategy {
    fn default() -> Self {
        Self::new()
    }
}
