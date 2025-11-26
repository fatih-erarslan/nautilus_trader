//! Breakout Trading Strategy
//!
//! Trades price breakouts from consolidation patterns.

use crate::config::{StrategyConfig, ParameterType};

pub struct BreakoutStrategy;

pub fn config() -> StrategyConfig {
    StrategyConfig::new(
        "breakout",
        "Trade breakouts from support/resistance with volume confirmation"
    )
    .with_sharpe_ratio(2.68)
    .with_risk_level("high")
    .with_gpu_capable(true)
    .with_status("active")
    .with_asset_types(vec![
        "stocks".to_string(),
        "crypto".to_string(),
    ])
    .with_min_capital(2000.0)
    .with_holding_period(2)
    .with_parameter("consolidation_period", ParameterType::Days, 5.0, 30.0, 14.0)
    .with_parameter("breakout_threshold", ParameterType::Percentage, 1.0, 5.0, 2.5)
    .with_parameter("volume_multiplier", ParameterType::Float, 1.2, 3.0, 1.5)
    .with_parameter("stop_loss_pct", ParameterType::Percentage, 1.0, 5.0, 2.0)
}

impl BreakoutStrategy {
    pub fn new() -> Self {
        Self
    }
}

impl Default for BreakoutStrategy {
    fn default() -> Self {
        Self::new()
    }
}
