//! Momentum Trading Strategy
//!
//! Trend-following strategy based on price momentum, RSI, and MACD indicators.
//! Generates buy signals when momentum is strong and RSI confirms, sell when momentum weakens.
//!
//! ## Algorithm
//!
//! 1. Calculate price momentum over lookback period
//! 2. Calculate RSI (14 period)
//! 3. Calculate MACD (12, 26, 9)
//! 4. Generate long signal if:
//!    - Momentum > entry_threshold AND
//!    - RSI < 70 AND
//!    - MACD > 0
//! 5. Generate close signal if:
//!    - Momentum < exit_threshold OR
//!    - RSI > 70
//!
//! ## Performance Targets
//!
//! - Latency: <15ms
//! - Throughput: 1000 signals/sec
//! - Memory: <10MB
//! - Python Sharpe: 2.84
//! - Rust Target Sharpe: >2.5

use crate::{
    async_trait, Bar, Direction, MarketData, Portfolio, Result, Signal, Strategy,
    StrategyError, StrategyMetadata, RiskParameters,
};
use rust_decimal::Decimal;
use rust_decimal::prelude::*;
use statrs::statistics::Statistics;

/// Momentum trading strategy
#[derive(Debug, Clone)]
pub struct MomentumStrategy {
    /// Strategy ID
    id: String,
    /// Symbols to trade
    symbols: Vec<String>,
    /// Lookback period for momentum calculation
    period: usize,
    /// Momentum threshold for entry (z-score)
    entry_threshold: f64,
    /// Momentum threshold for exit (z-score)
    exit_threshold: f64,
}

impl MomentumStrategy {
    /// Create a new momentum strategy
    pub fn new(
        symbols: Vec<String>,
        period: usize,
        entry_threshold: f64,
        exit_threshold: f64,
    ) -> Self {
        Self {
            id: "momentum_trader".to_string(),
            symbols,
            period,
            entry_threshold,
            exit_threshold,
        }
    }

    /// Calculate momentum (price change over period)
    fn calculate_momentum(&self, bars: &[Bar]) -> Result<f64> {
        if bars.len() < self.period + 1 {
            return Err(StrategyError::InsufficientData {
                needed: self.period + 1,
                available: bars.len(),
            });
        }

        let current = bars.last().unwrap().close.to_f64().unwrap();
        let past = bars[bars.len() - self.period - 1].close.to_f64().unwrap();

        Ok((current - past) / past)
    }

    /// Calculate RSI (Relative Strength Index)
    fn calculate_rsi(&self, bars: &[Bar], period: usize) -> Result<f64> {
        if bars.len() < period + 1 {
            return Err(StrategyError::InsufficientData {
                needed: period + 1,
                available: bars.len(),
            });
        }

        let prices: Vec<f64> = bars
            .iter()
            .rev()
            .take(period + 1)
            .map(|b| b.close.to_f64().unwrap())
            .collect();

        let mut gains = Vec::new();
        let mut losses = Vec::new();

        for i in 0..prices.len() - 1 {
            let change = prices[i] - prices[i + 1];
            if change > 0.0 {
                gains.push(change);
                losses.push(0.0);
            } else {
                gains.push(0.0);
                losses.push(-change);
            }
        }

        let avg_gain = gains.iter().sum::<f64>() / period as f64;
        let avg_loss = losses.iter().sum::<f64>() / period as f64;

        if avg_loss == 0.0 {
            return Ok(100.0);
        }

        let rs = avg_gain / avg_loss;
        let rsi = 100.0 - (100.0 / (1.0 + rs));

        Ok(rsi)
    }

    /// Calculate MACD (Moving Average Convergence Divergence)
    fn calculate_macd(&self, bars: &[Bar]) -> Result<f64> {
        let fast_period = 12;
        let slow_period = 26;

        if bars.len() < slow_period {
            return Err(StrategyError::InsufficientData {
                needed: slow_period,
                available: bars.len(),
            });
        }

        let prices: Vec<f64> = bars
            .iter()
            .map(|b| b.close.to_f64().unwrap())
            .collect();

        let ema_fast = self.calculate_ema(&prices, fast_period);
        let ema_slow = self.calculate_ema(&prices, slow_period);

        Ok(ema_fast - ema_slow)
    }

    /// Calculate Exponential Moving Average
    fn calculate_ema(&self, prices: &[f64], period: usize) -> f64 {
        let multiplier = 2.0 / (period as f64 + 1.0);
        let mut ema = prices[0];

        for &price in prices.iter().skip(1) {
            ema = (price - ema) * multiplier + ema;
        }

        ema
    }

    /// Calculate z-score for momentum
    fn calculate_z_score(&self, momentum: f64, bars: &[Bar]) -> Result<f64> {
        if bars.len() < self.period {
            return Ok(0.0);
        }

        let momentums: Vec<f64> = (self.period..bars.len())
            .map(|i| {
                let current = bars[i].close.to_f64().unwrap();
                let past = bars[i - self.period].close.to_f64().unwrap();
                (current - past) / past
            })
            .collect();

        if momentums.is_empty() {
            return Ok(0.0);
        }

        // Clone before consuming for mean/std_dev calculations
        let mean = momentums.clone().mean();
        let std = momentums.clone().std_dev();

        if std == 0.0 {
            return Ok(0.0);
        }

        Ok((momentum - mean) / std)
    }

    /// Calculate confidence score
    fn calculate_confidence(&self, momentum: f64, rsi: f64, macd: f64) -> f64 {
        // Weighted combination of indicators
        let momentum_score = (momentum.abs() / self.entry_threshold).min(1.0);

        let rsi_score = if momentum > 0.0 {
            ((70.0 - rsi) / 70.0).max(0.0)
        } else {
            ((rsi - 30.0) / 70.0).max(0.0)
        };

        let macd_score = (macd.abs() / 2.0).min(1.0);

        (momentum_score * 0.5 + rsi_score * 0.3 + macd_score * 0.2).clamp(0.0, 1.0)
    }

    /// Check if we should close position
    fn should_close_position(
        &self,
        symbol: &str,
        momentum: f64,
        rsi: f64,
        portfolio: &Portfolio,
    ) -> bool {
        if !portfolio.has_position(symbol) {
            return false;
        }

        // Close if momentum weakens or RSI is extreme
        momentum.abs() < self.exit_threshold || rsi > 70.0 || rsi < 30.0
    }
}

#[async_trait]
impl Strategy for MomentumStrategy {
    fn id(&self) -> &str {
        &self.id
    }

    fn metadata(&self) -> StrategyMetadata {
        StrategyMetadata {
            name: "Momentum Trader".to_string(),
            description: "Trend-following strategy based on price momentum, RSI, and MACD".to_string(),
            version: "1.0.0".to_string(),
            author: "Neural Trader Team".to_string(),
            tags: vec!["momentum".to_string(), "trend".to_string(), "technical".to_string()],
            min_capital: Decimal::from(5000),
            max_drawdown_threshold: 0.15,
        }
    }

    async fn process(
        &self,
        market_data: &MarketData,
        portfolio: &Portfolio,
    ) -> Result<Vec<Signal>> {
        let mut signals = Vec::new();

        // Only process if symbol is in our list
        if !self.symbols.contains(&market_data.symbol) {
            return Ok(signals);
        }

        let bars = &market_data.bars;

        // Need enough data
        if bars.len() < self.period + 26 {
            return Err(StrategyError::InsufficientData {
                needed: self.period + 26,
                available: bars.len(),
            });
        }

        // Calculate indicators
        let momentum = self.calculate_momentum(bars)?;
        let rsi = self.calculate_rsi(bars, 14)?;
        let macd = self.calculate_macd(bars)?;
        let z_score = self.calculate_z_score(momentum, bars)?;

        // Check for close signal first
        if self.should_close_position(&market_data.symbol, momentum, rsi, portfolio) {
            signals.push(
                Signal::new(
                    self.id.clone(),
                    market_data.symbol.clone(),
                    Direction::Close,
                )
                .with_confidence(0.7)
                .with_reasoning(format!(
                    "Momentum weakened: {:.2}%, RSI: {:.1}",
                    momentum * 100.0,
                    rsi
                ))
                .with_features(vec![momentum, rsi, macd, z_score]),
            );
            return Ok(signals);
        }

        // Generate entry signals
        let direction = if z_score > self.entry_threshold && rsi < 70.0 && macd > 0.0 {
            Direction::Long
        } else if z_score < -self.entry_threshold && rsi > 30.0 && macd < 0.0 {
            Direction::Short
        } else {
            // No signal
            return Ok(signals);
        };

        let confidence = self.calculate_confidence(momentum, rsi, macd);
        let current_price = bars.last().unwrap().close;

        // Calculate stop loss and take profit
        let stop_loss_pct = self.risk_parameters().stop_loss_percentage;
        let take_profit_pct = self.risk_parameters().take_profit_percentage;

        let (stop_loss, take_profit) = match direction {
            Direction::Long => (
                current_price * Decimal::from_f64_retain(1.0 - stop_loss_pct).unwrap(),
                current_price * Decimal::from_f64_retain(1.0 + take_profit_pct).unwrap(),
            ),
            Direction::Short => (
                current_price * Decimal::from_f64_retain(1.0 + stop_loss_pct).unwrap(),
                current_price * Decimal::from_f64_retain(1.0 - take_profit_pct).unwrap(),
            ),
            Direction::Close => (current_price, current_price),
        };

        signals.push(
            Signal::new(self.id.clone(), market_data.symbol.clone(), direction)
                .with_confidence(confidence)
                .with_entry_price(current_price)
                .with_stop_loss(stop_loss)
                .with_take_profit(take_profit)
                .with_reasoning(format!(
                    "Momentum: {:.2}% (z={:.2}), RSI: {:.1}, MACD: {:.2}",
                    momentum * 100.0,
                    z_score,
                    rsi,
                    macd
                ))
                .with_features(vec![momentum, rsi, macd, z_score]),
        );

        Ok(signals)
    }

    fn validate_config(&self) -> Result<()> {
        if self.symbols.is_empty() {
            return Err(StrategyError::ConfigError(
                "At least one symbol must be specified".to_string(),
            ));
        }

        if self.period < 5 {
            return Err(StrategyError::InvalidParameter(
                "Period must be at least 5".to_string(),
            ));
        }

        if self.entry_threshold <= 0.0 {
            return Err(StrategyError::InvalidParameter(
                "Entry threshold must be positive".to_string(),
            ));
        }

        Ok(())
    }

    fn risk_parameters(&self) -> RiskParameters {
        RiskParameters {
            max_position_size: Decimal::from(10000),
            max_leverage: 1.0,
            stop_loss_percentage: 0.02,
            take_profit_percentage: 0.05,
            max_daily_loss: Decimal::from(1000),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn create_test_bars(count: usize, start_price: f64, trend: f64) -> Vec<Bar> {
        (0..count)
            .map(|i| {
                let price = start_price + (i as f64 * trend);
                Bar {
                    symbol: "TEST".to_string(),
                    timestamp: Utc::now(),
                    open: Decimal::from_f64_retain(price * 0.99).unwrap(),
                    high: Decimal::from_f64_retain(price * 1.01).unwrap(),
                    low: Decimal::from_f64_retain(price * 0.98).unwrap(),
                    close: Decimal::from_f64_retain(price).unwrap(),
                    volume: 1000000,
                }
            })
            .collect()
    }

    #[test]
    fn test_momentum_calculation() {
        let strategy = MomentumStrategy::new(vec!["TEST".to_string()], 10, 2.0, 0.5);
        let bars = create_test_bars(20, 100.0, 1.0); // Uptrend

        let momentum = strategy.calculate_momentum(&bars).unwrap();
        assert!(momentum > 0.0, "Should detect upward momentum");
    }

    #[test]
    fn test_rsi_calculation() {
        let strategy = MomentumStrategy::new(vec!["TEST".to_string()], 10, 2.0, 0.5);
        let bars = create_test_bars(30, 100.0, 1.0);

        let rsi = strategy.calculate_rsi(&bars, 14).unwrap();
        assert!(rsi > 0.0 && rsi <= 100.0, "RSI should be between 0 and 100");
    }

    #[test]
    fn test_macd_calculation() {
        let strategy = MomentumStrategy::new(vec!["TEST".to_string()], 10, 2.0, 0.5);
        let bars = create_test_bars(50, 100.0, 0.5);

        let macd = strategy.calculate_macd(&bars).unwrap();
        assert!(macd.is_finite(), "MACD should be a finite number");
    }

    #[test]
    fn test_strategy_validation() {
        let strategy = MomentumStrategy::new(vec!["TEST".to_string()], 20, 2.0, 0.5);
        assert!(strategy.validate_config().is_ok());

        let invalid = MomentumStrategy::new(vec![], 20, 2.0, 0.5);
        assert!(invalid.validate_config().is_err());

        let invalid2 = MomentumStrategy::new(vec!["TEST".to_string()], 2, 2.0, 0.5);
        assert!(invalid2.validate_config().is_err());
    }

    #[tokio::test]
    async fn test_signal_generation() {
        let strategy = MomentumStrategy::new(vec!["TEST".to_string()], 10, 1.5, 0.5);
        let bars = create_test_bars(50, 100.0, 0.5); // Strong uptrend

        let market_data = MarketData {
            symbol: "TEST".to_string(),
            timestamp: Utc::now(),
            price: Decimal::from(125),
            volume: 1000000,
            bars: bars.clone(),
        };

        let portfolio = Portfolio::new(Decimal::from(100000));
        let signals = strategy.process(&market_data, &portfolio).await.unwrap();

        // Should generate at least one signal for strong trend
        assert!(!signals.is_empty() || bars.len() < 40);
    }
}
