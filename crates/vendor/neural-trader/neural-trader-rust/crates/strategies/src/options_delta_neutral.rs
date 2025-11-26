//! Options Delta-Neutral Strategy
//!
//! Market-neutral options trading with dynamic hedging.

use crate::config::{StrategyConfig, ParameterType};

pub struct OptionsDeltaNeutralStrategy;

pub fn config() -> StrategyConfig {
    StrategyConfig::new(
        "options_delta_neutral",
        "Volatility trading with delta-hedged options positions"
    )
    .with_sharpe_ratio(2.57)
    .with_risk_level("medium")
    .with_gpu_capable(true)
    .with_status("active")
    .with_asset_types(vec![
        "options".to_string(),
        "stocks".to_string(),
    ])
    .with_min_capital(50000.0)
    .with_holding_period(7)
    .with_parameter("iv_percentile_threshold", ParameterType::Percentage, 30.0, 90.0, 50.0)
    .with_parameter("delta_tolerance", ParameterType::Float, 0.01, 0.1, 0.05)
    .with_parameter("gamma_threshold", ParameterType::Float, 0.01, 0.2, 0.05)
    .with_parameter("rehedge_frequency_hours", ParameterType::Integer, 1.0, 24.0, 4.0)
}

impl OptionsDeltaNeutralStrategy {
    pub fn new() -> Self {
        Self
    }
}

impl Default for OptionsDeltaNeutralStrategy {
    fn default() -> Self {
        Self::new()
    }
}
