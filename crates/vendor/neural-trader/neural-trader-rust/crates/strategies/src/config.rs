//! Strategy configuration types and validation

use rust_decimal::Decimal;
use serde::{Serialize, Deserialize};
use crate::{Result, StrategyError};

/// Risk level configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Aggressive,
}

impl RiskLevel {
    /// Get position size multiplier for this risk level
    pub fn position_multiplier(&self) -> f64 {
        match self {
            RiskLevel::Low => 0.25,
            RiskLevel::Medium => 0.50,
            RiskLevel::High => 0.75,
            RiskLevel::Aggressive => 1.0,
        }
    }

    /// Get stop loss percentage for this risk level
    pub fn stop_loss_percentage(&self) -> f64 {
        match self {
            RiskLevel::Low => 0.01,      // 1%
            RiskLevel::Medium => 0.02,   // 2%
            RiskLevel::High => 0.03,     // 3%
            RiskLevel::Aggressive => 0.05, // 5%
        }
    }
}

/// Time frame for trading
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TimeFrame {
    #[serde(rename = "1m")]
    OneMinute,
    #[serde(rename = "5m")]
    FiveMinutes,
    #[serde(rename = "15m")]
    FifteenMinutes,
    #[serde(rename = "1h")]
    OneHour,
    #[serde(rename = "4h")]
    FourHours,
    #[serde(rename = "1d")]
    OneDay,
}

impl TimeFrame {
    /// Get lookback period in bars for this timeframe
    pub fn default_lookback(&self) -> usize {
        match self {
            TimeFrame::OneMinute => 60,
            TimeFrame::FiveMinutes => 100,
            TimeFrame::FifteenMinutes => 100,
            TimeFrame::OneHour => 168,
            TimeFrame::FourHours => 180,
            TimeFrame::OneDay => 252,
        }
    }
}

/// Global strategy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyConfig {
    /// Active strategy IDs
    pub strategies: Vec<String>,
    /// Symbols to trade
    pub symbols: Vec<String>,
    /// Risk level
    pub risk_level: RiskLevel,
    /// Maximum position size in USD
    pub max_position_size: Decimal,
    /// Stop loss percentage
    pub stop_loss_percentage: f64,
    /// Take profit percentage
    pub take_profit_percentage: f64,
    /// Time frame for trading
    pub time_frame: TimeFrame,
    /// Use GPU acceleration
    pub use_gpu: bool,
    /// Enable news trading
    pub enable_news_trading: bool,
    /// Enable sentiment analysis
    pub enable_sentiment_analysis: bool,
}

impl Default for StrategyConfig {
    fn default() -> Self {
        Self {
            strategies: vec!["momentum_trader".to_string()],
            symbols: vec!["SPY".to_string()],
            risk_level: RiskLevel::Medium,
            max_position_size: Decimal::from(10000),
            stop_loss_percentage: 0.02,
            take_profit_percentage: 0.05,
            time_frame: TimeFrame::OneHour,
            use_gpu: false,
            enable_news_trading: false,
            enable_sentiment_analysis: false,
        }
    }
}

impl StrategyConfig {
    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        if self.strategies.is_empty() {
            return Err(StrategyError::ConfigError(
                "At least one strategy must be specified".to_string(),
            ));
        }

        if self.symbols.is_empty() {
            return Err(StrategyError::ConfigError(
                "At least one symbol must be specified".to_string(),
            ));
        }

        if self.max_position_size <= Decimal::ZERO {
            return Err(StrategyError::ConfigError(
                "Max position size must be positive".to_string(),
            ));
        }

        if self.stop_loss_percentage <= 0.0 || self.stop_loss_percentage > 1.0 {
            return Err(StrategyError::ConfigError(
                "Stop loss percentage must be between 0 and 1".to_string(),
            ));
        }

        if self.take_profit_percentage <= 0.0 || self.take_profit_percentage > 1.0 {
            return Err(StrategyError::ConfigError(
                "Take profit percentage must be between 0 and 1".to_string(),
            ));
        }

        Ok(())
    }
}

/// Momentum strategy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MomentumConfig {
    pub period: usize,
    pub entry_threshold: f64,
    pub exit_threshold: f64,
}

impl Default for MomentumConfig {
    fn default() -> Self {
        Self {
            period: 20,
            entry_threshold: 2.0,
            exit_threshold: 0.5,
        }
    }
}

/// Mean reversion strategy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeanReversionConfig {
    pub period: usize,
    pub num_std: f64,
    pub rsi_period: usize,
}

impl Default for MeanReversionConfig {
    fn default() -> Self {
        Self {
            period: 20,
            num_std: 2.0,
            rsi_period: 14,
        }
    }
}

/// Pairs trading configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairsConfig {
    pub pairs: Vec<(String, String)>,
    pub cointegration_period: usize,
    pub entry_threshold: f64,
    pub exit_threshold: f64,
}

impl Default for PairsConfig {
    fn default() -> Self {
        Self {
            pairs: vec![],
            cointegration_period: 60,
            entry_threshold: 2.0,
            exit_threshold: 0.5,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_risk_level_multipliers() {
        assert_eq!(RiskLevel::Low.position_multiplier(), 0.25);
        assert_eq!(RiskLevel::Medium.position_multiplier(), 0.50);
        assert_eq!(RiskLevel::High.position_multiplier(), 0.75);
        assert_eq!(RiskLevel::Aggressive.position_multiplier(), 1.0);
    }

    #[test]
    fn test_strategy_config_validation() {
        let mut config = StrategyConfig::default();
        assert!(config.validate().is_ok());

        config.strategies.clear();
        assert!(config.validate().is_err());

        config.strategies.push("momentum".to_string());
        config.symbols.clear();
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_timeframe_lookback() {
        assert_eq!(TimeFrame::OneMinute.default_lookback(), 60);
        assert_eq!(TimeFrame::OneHour.default_lookback(), 168);
        assert_eq!(TimeFrame::OneDay.default_lookback(), 252);
    }
}
