//! Mean Reversion Trading Strategy
//!
//! Statistical arbitrage strategy based on Bollinger Bands and RSI.
//! Buys when price is oversold (below lower band + low RSI),
//! sells when overbought (above upper band + high RSI).
//!
//! ## Algorithm
//!
//! 1. Calculate Bollinger Bands (SMA ± N*std)
//! 2. Calculate RSI (14 period)
//! 3. Generate long signal if:
//!    - Price < Lower Band AND RSI < 30
//! 4. Generate short signal if:
//!    - Price > Upper Band AND RSI > 70
//! 5. Close when price returns to mean
//!
//! ## Performance Targets
//!
//! - Latency: <10ms
//! - Throughput: 1500 signals/sec
//! - Memory: <8MB
//! - Python Sharpe: 2.15
//! - Rust Target Sharpe: >2.0

use crate::{
    async_trait, Bar, Direction, MarketData, Portfolio, Result, Signal, Strategy,
    StrategyError, StrategyMetadata, RiskParameters,
};
use rust_decimal::Decimal;
use rust_decimal::prelude::*;
// Statistics trait is used implicitly by .mean() and .std_dev() methods on Vec<f64>

/// Mean reversion trading strategy
#[derive(Debug, Clone)]
pub struct MeanReversionStrategy {
    /// Strategy ID
    id: String,
    /// Symbols to trade
    symbols: Vec<String>,
    /// Lookback period for mean calculation
    period: usize,
    /// Standard deviation multiplier for bands
    num_std: f64,
    /// RSI period
    rsi_period: usize,
}

impl MeanReversionStrategy {
    /// Create a new mean reversion strategy
    pub fn new(symbols: Vec<String>, period: usize, num_std: f64, rsi_period: usize) -> Self {
        Self {
            id: "mean_reversion".to_string(),
            symbols,
            period,
            num_std,
            rsi_period,
        }
    }

    /// Calculate Simple Moving Average
    fn calculate_sma(&self, prices: &[f64]) -> f64 {
        if prices.is_empty() {
            return 0.0;
        }
        prices.iter().sum::<f64>() / prices.len() as f64
    }

    /// Calculate standard deviation
    fn calculate_std(&self, prices: &[f64], mean: f64) -> f64 {
        if prices.len() < 2 {
            return 0.0;
        }

        let variance = prices
            .iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / (prices.len() - 1) as f64;

        variance.sqrt()
    }

    /// Calculate Bollinger Bands
    fn calculate_bollinger_bands(&self, bars: &[Bar]) -> Result<(f64, f64, f64)> {
        if bars.len() < self.period {
            return Err(StrategyError::InsufficientData {
                needed: self.period,
                available: bars.len(),
            });
        }

        let prices: Vec<f64> = bars
            .iter()
            .rev()
            .take(self.period)
            .map(|b| b.close.to_f64().unwrap())
            .collect();

        let mean = self.calculate_sma(&prices);
        let std = self.calculate_std(&prices, mean);

        let upper_band = mean + self.num_std * std;
        let lower_band = mean - self.num_std * std;

        Ok((upper_band, mean, lower_band))
    }

    /// Calculate RSI (Relative Strength Index)
    fn calculate_rsi(&self, bars: &[Bar]) -> Result<f64> {
        if bars.len() < self.rsi_period + 1 {
            return Err(StrategyError::InsufficientData {
                needed: self.rsi_period + 1,
                available: bars.len(),
            });
        }

        let prices: Vec<f64> = bars
            .iter()
            .rev()
            .take(self.rsi_period + 1)
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

        let avg_gain = gains.iter().sum::<f64>() / self.rsi_period as f64;
        let avg_loss = losses.iter().sum::<f64>() / self.rsi_period as f64;

        if avg_loss == 0.0 {
            return Ok(100.0);
        }

        let rs = avg_gain / avg_loss;
        let rsi = 100.0 - (100.0 / (1.0 + rs));

        Ok(rsi)
    }

    /// Calculate confidence based on distance from bands and RSI
    fn calculate_reversion_confidence(&self, price: f64, band: f64, rsi: f64) -> f64 {
        let band_distance = ((price - band).abs() / band).min(0.1) * 10.0; // 0-1 scale

        let rsi_extreme = if rsi < 30.0 {
            (30.0 - rsi) / 30.0
        } else if rsi > 70.0 {
            (rsi - 70.0) / 30.0
        } else {
            0.0
        }
        .max(0.0)
        .min(1.0);

        (band_distance * 0.6 + rsi_extreme * 0.4).clamp(0.5, 1.0)
    }

    /// Check if position should be closed (price returned to mean)
    fn should_close_position(
        &self,
        symbol: &str,
        current_price: f64,
        mean: f64,
        lower_band: f64,
        upper_band: f64,
        portfolio: &Portfolio,
    ) -> bool {
        if !portfolio.has_position(symbol) {
            return false;
        }

        // Close if price is back in normal range
        current_price > lower_band && current_price < upper_band
    }
}

#[async_trait]
impl Strategy for MeanReversionStrategy {
    fn id(&self) -> &str {
        &self.id
    }

    fn metadata(&self) -> StrategyMetadata {
        StrategyMetadata {
            name: "Mean Reversion".to_string(),
            description: "Statistical arbitrage based on Bollinger Bands and RSI".to_string(),
            version: "1.0.0".to_string(),
            author: "Neural Trader Team".to_string(),
            tags: vec![
                "mean_reversion".to_string(),
                "statistical".to_string(),
                "technical".to_string(),
            ],
            min_capital: Decimal::from(3000),
            max_drawdown_threshold: 0.10,
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
        if bars.len() < self.period.max(self.rsi_period + 1) {
            return Err(StrategyError::InsufficientData {
                needed: self.period.max(self.rsi_period + 1),
                available: bars.len(),
            });
        }

        // Calculate indicators
        let (upper_band, mean, lower_band) = self.calculate_bollinger_bands(bars)?;
        let rsi = self.calculate_rsi(bars)?;
        let current_price = bars.last().unwrap().close.to_f64().unwrap();

        // Check for close signal first
        if self.should_close_position(
            &market_data.symbol,
            current_price,
            mean,
            lower_band,
            upper_band,
            portfolio,
        ) {
            signals.push(
                Signal::new(
                    self.id.clone(),
                    market_data.symbol.clone(),
                    Direction::Close,
                )
                .with_confidence(0.7)
                .with_reasoning(format!(
                    "Price returned to mean: ${:.2} (mean: ${:.2})",
                    current_price, mean
                ))
                .with_features(vec![current_price, mean, upper_band, lower_band, rsi]),
            );
            return Ok(signals);
        }

        // Generate entry signals
        let (direction, confidence, reasoning) = if current_price < lower_band && rsi < 30.0 {
            (
                Direction::Long,
                self.calculate_reversion_confidence(current_price, lower_band, rsi),
                format!(
                    "Oversold: Price ${:.2} < Lower Band ${:.2}, RSI: {:.1}",
                    current_price, lower_band, rsi
                ),
            )
        } else if current_price > upper_band && rsi > 70.0 {
            (
                Direction::Short,
                self.calculate_reversion_confidence(current_price, upper_band, rsi),
                format!(
                    "Overbought: Price ${:.2} > Upper Band ${:.2}, RSI: {:.1}",
                    current_price, upper_band, rsi
                ),
            )
        } else {
            // No signal
            return Ok(signals);
        };

        let current_price_decimal = bars.last().unwrap().close;

        // Calculate stop loss and take profit
        let stop_loss_pct = self.risk_parameters().stop_loss_percentage;
        let take_profit_pct = self.risk_parameters().take_profit_percentage;

        let (stop_loss, take_profit) = match direction {
            Direction::Long => (
                current_price_decimal * Decimal::from_f64_retain(1.0 - stop_loss_pct).unwrap(),
                current_price_decimal * Decimal::from_f64_retain(1.0 + take_profit_pct).unwrap(),
            ),
            Direction::Short => (
                current_price_decimal * Decimal::from_f64_retain(1.0 + stop_loss_pct).unwrap(),
                current_price_decimal * Decimal::from_f64_retain(1.0 - take_profit_pct).unwrap(),
            ),
            Direction::Close => (current_price_decimal, current_price_decimal),
        };

        signals.push(
            Signal::new(self.id.clone(), market_data.symbol.clone(), direction)
                .with_confidence(confidence)
                .with_entry_price(current_price_decimal)
                .with_stop_loss(stop_loss)
                .with_take_profit(take_profit)
                .with_reasoning(reasoning)
                .with_features(vec![current_price, mean, upper_band, lower_band, rsi]),
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

        if self.num_std <= 0.0 {
            return Err(StrategyError::InvalidParameter(
                "Standard deviation multiplier must be positive".to_string(),
            ));
        }

        if self.rsi_period < 5 {
            return Err(StrategyError::InvalidParameter(
                "RSI period must be at least 5".to_string(),
            ));
        }

        Ok(())
    }

    fn risk_parameters(&self) -> RiskParameters {
        RiskParameters {
            max_position_size: Decimal::from(8000),
            max_leverage: 1.0,
            stop_loss_percentage: 0.015,
            take_profit_percentage: 0.03,
            max_daily_loss: Decimal::from(800),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn create_oscillating_bars(count: usize, center: f64, amplitude: f64) -> Vec<Bar> {
        use std::f64::consts::PI;

        (0..count)
            .map(|i| {
                let price = center + amplitude * (2.0 * PI * i as f64 / 20.0).sin();
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
    fn test_bollinger_bands() {
        let strategy = MeanReversionStrategy::new(vec!["TEST".to_string()], 20, 2.0, 14);
        let bars = create_oscillating_bars(50, 100.0, 10.0);

        let (upper, mean, lower) = strategy.calculate_bollinger_bands(&bars).unwrap();

        assert!(upper > mean);
        assert!(mean > lower);
        assert!((mean - 100.0).abs() < 5.0); // Should be near center
    }

    #[test]
    fn test_rsi_calculation() {
        let strategy = MeanReversionStrategy::new(vec!["TEST".to_string()], 20, 2.0, 14);
        let bars = create_oscillating_bars(50, 100.0, 10.0);

        let rsi = strategy.calculate_rsi(&bars).unwrap();
        assert!(rsi >= 0.0 && rsi <= 100.0, "RSI should be between 0 and 100");
    }

    #[test]
    fn test_strategy_validation() {
        let strategy = MeanReversionStrategy::new(vec!["TEST".to_string()], 20, 2.0, 14);
        assert!(strategy.validate_config().is_ok());

        let invalid = MeanReversionStrategy::new(vec![], 20, 2.0, 14);
        assert!(invalid.validate_config().is_err());
    }

    #[tokio::test]
    async fn test_signal_generation() {
        let strategy = MeanReversionStrategy::new(vec!["TEST".to_string()], 20, 2.0, 14);
        let bars = create_oscillating_bars(50, 100.0, 15.0); // Large oscillations

        let market_data = MarketData {
            symbol: "TEST".to_string(),
            timestamp: Utc::now(),
            price: Decimal::from(85), // Below mean
            volume: 1000000,
            bars: bars.clone(),
        };

        let portfolio = Portfolio::new(Decimal::from(100000));
        let result = strategy.process(&market_data, &portfolio).await;

        assert!(result.is_ok());
    }
}
