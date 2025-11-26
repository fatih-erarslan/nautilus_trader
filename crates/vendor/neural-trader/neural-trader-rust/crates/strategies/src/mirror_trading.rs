//! Mirror Trading Strategy
//!
//! Copies trades from successful traders with risk management overlays.
//! Highest Sharpe ratio strategy.

use crate::config::{StrategyConfig, ParameterType};

pub struct MirrorTradingStrategy;

pub fn config() -> StrategyConfig {
    StrategyConfig::new(
        "mirror_trading",
        "Copy trades from top-performing traders with intelligent risk scaling"
    )
    .with_sharpe_ratio(6.01)
    .with_risk_level("medium")
    .with_gpu_capable(true)
    .with_status("active")
    .with_asset_types(vec![
        "stocks".to_string(),
        "forex".to_string(),
        "crypto".to_string(),
    ])
    .with_min_capital(5000.0)
    .with_holding_period(1)
    .with_parameter("mirror_count", ParameterType::Integer, 1.0, 10.0, 3.0)
    .with_parameter("position_size_pct", ParameterType::Percentage, 1.0, 20.0, 5.0)
    .with_parameter("max_drawdown", ParameterType::Percentage, 5.0, 30.0, 15.0)
    .with_parameter("correlation_threshold", ParameterType::Float, 0.3, 0.9, 0.7)
}

impl MirrorTradingStrategy {
    pub fn new() -> Self {
        Self
    }
}

impl Default for MirrorTradingStrategy {
    fn default() -> Self {
        Self::new()
    }
}
