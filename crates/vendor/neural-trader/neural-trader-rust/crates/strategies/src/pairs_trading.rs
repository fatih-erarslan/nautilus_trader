//! Pairs Trading Strategy
//!
//! Market-neutral strategy trading correlated asset pairs.

use crate::config::{StrategyConfig, ParameterType};

pub struct PairsTradingStrategy;

pub fn config() -> StrategyConfig {
    StrategyConfig::new(
        "pairs_trading",
        "Trade cointegrated pairs for market-neutral returns"
    )
    .with_sharpe_ratio(2.31)
    .with_risk_level("low")
    .with_gpu_capable(true)
    .with_status("active")
    .with_asset_types(vec![
        "stocks".to_string(),
        "etfs".to_string(),
    ])
    .with_min_capital(5000.0)
    .with_holding_period(5)
    .with_parameter("cointegration_threshold", ParameterType::Float, 0.01, 0.1, 0.05)
    .with_parameter("entry_z_score", ParameterType::Float, 1.5, 3.0, 2.0)
    .with_parameter("exit_z_score", ParameterType::Float, 0.0, 1.0, 0.5)
    .with_parameter("lookback_period", ParameterType::Days, 30.0, 252.0, 60.0)
}

impl PairsTradingStrategy {
    pub fn new() -> Self {
        Self
    }
}

impl Default for PairsTradingStrategy {
    fn default() -> Self {
        Self::new()
    }
}
