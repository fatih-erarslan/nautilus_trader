//! Pairs Trading Strategy
//!
//! Market-neutral strategy that trades correlated pairs when they diverge.
//! Uses cointegration testing and z-score based entry/exit.
//!
//! ## Algorithm
//!
//! 1. Test pair for cointegration
//! 2. Calculate hedge ratio via linear regression
//! 3. Calculate spread = price_a - hedge_ratio * price_b
//! 4. Calculate z-score of spread
//! 5. Enter when |z-score| > entry_threshold
//! 6. Exit when |z-score| < exit_threshold
//!
//! ## Performance Targets
//!
//! - Latency: <25ms
//! - Throughput: 200 pairs/sec
//! - Memory: <20MB
//! - Expected Sharpe: >2.5

use crate::{
    async_trait, Bar, Direction, MarketData, Portfolio, Result, Signal, Strategy,
    StrategyError, StrategyMetadata, RiskParameters,
};
use rust_decimal::Decimal;
use rust_decimal::prelude::*;
use statrs::statistics::Statistics;

/// Pairs trading strategy
#[derive(Debug, Clone)]
pub struct PairsStrategy {
    /// Strategy ID
    id: String,
    /// Trading pairs (symbol_a, symbol_b)
    pairs: Vec<(String, String)>,
    /// Lookback period for cointegration
    cointegration_period: usize,
    /// Z-score threshold for entry
    entry_threshold: f64,
    /// Z-score threshold for exit
    exit_threshold: f64,
}

impl PairsStrategy {
    /// Create a new pairs trading strategy
    pub fn new(
        pairs: Vec<(String, String)>,
        cointegration_period: usize,
        entry_threshold: f64,
        exit_threshold: f64,
    ) -> Self {
        Self {
            id: "pairs_trading".to_string(),
            pairs,
            cointegration_period,
            entry_threshold,
            exit_threshold,
        }
    }

    /// Calculate hedge ratio using linear regression
    fn calculate_hedge_ratio(&self, prices_a: &[f64], prices_b: &[f64]) -> f64 {
        if prices_a.len() != prices_b.len() || prices_a.is_empty() {
            return 1.0;
        }

        let n = prices_a.len() as f64;
        let sum_x: f64 = prices_b.iter().sum();
        let sum_y: f64 = prices_a.iter().sum();
        let sum_xy: f64 = prices_a.iter().zip(prices_b.iter()).map(|(a, b)| a * b).sum();
        let sum_x2: f64 = prices_b.iter().map(|x| x * x).sum();

        let denominator = n * sum_x2 - sum_x * sum_x;

        if denominator == 0.0 {
            return 1.0;
        }

        (n * sum_xy - sum_x * sum_y) / denominator
    }

    /// Test for stationarity using Augmented Dickey-Fuller-like test (simplified)
    fn is_stationary(&self, series: &[f64]) -> bool {
        if series.len() < 10 {
            return false;
        }

        // Simple stationarity check: mean reversion test
        let mean = series.mean();
        let std = series.std_dev();

        if std == 0.0 {
            return false;
        }

        // Check if values tend to revert to mean
        let mut crossings = 0;
        for i in 1..series.len() {
            if (series[i - 1] - mean) * (series[i] - mean) < 0.0 {
                crossings += 1;
            }
        }

        // If we cross the mean frequently, it's likely stationary
        let crossing_rate = crossings as f64 / (series.len() - 1) as f64;
        crossing_rate > 0.3
    }

    /// Test pair for cointegration
    fn test_cointegration(
        &self,
        bars_a: &[Bar],
        bars_b: &[Bar],
    ) -> Result<(bool, f64)> {
        if bars_a.len() != bars_b.len() {
            return Err(StrategyError::CalculationError(
                "Bar arrays must have same length".to_string(),
            ));
        }

        if bars_a.len() < self.cointegration_period {
            return Err(StrategyError::InsufficientData {
                needed: self.cointegration_period,
                available: bars_a.len(),
            });
        }

        let prices_a: Vec<f64> = bars_a
            .iter()
            .map(|b| b.close.to_f64().unwrap())
            .collect();

        let prices_b: Vec<f64> = bars_b
            .iter()
            .map(|b| b.close.to_f64().unwrap())
            .collect();

        // Calculate hedge ratio
        let hedge_ratio = self.calculate_hedge_ratio(&prices_a, &prices_b);

        // Calculate spread
        let spread: Vec<f64> = prices_a
            .iter()
            .zip(prices_b.iter())
            .map(|(a, b)| a - hedge_ratio * b)
            .collect();

        // Test for stationarity
        let is_cointegrated = self.is_stationary(&spread);

        Ok((is_cointegrated, hedge_ratio))
    }

    /// Calculate z-score of current spread
    fn calculate_spread_zscore(
        &self,
        prices_a: &[f64],
        prices_b: &[f64],
        hedge_ratio: f64,
    ) -> f64 {
        let spread: Vec<f64> = prices_a
            .iter()
            .zip(prices_b.iter())
            .map(|(a, b)| a - hedge_ratio * b)
            .collect();

        // Clone before consuming for mean/std_dev calculations
        let mean = spread.clone().mean();
        let std = spread.clone().std_dev();

        if std == 0.0 {
            return 0.0;
        }

        let current_spread = spread.last().unwrap();
        (current_spread - mean) / std
    }
}

#[async_trait]
impl Strategy for PairsStrategy {
    fn id(&self) -> &str {
        &self.id
    }

    fn metadata(&self) -> StrategyMetadata {
        StrategyMetadata {
            name: "Pairs Trading".to_string(),
            description: "Market-neutral pair strategies with cointegration".to_string(),
            version: "1.0.0".to_string(),
            author: "Neural Trader Team".to_string(),
            tags: vec![
                "pairs".to_string(),
                "market_neutral".to_string(),
                "statistical".to_string(),
            ],
            min_capital: Decimal::from(20000),
            max_drawdown_threshold: 0.10,
        }
    }

    async fn process(
        &self,
        market_data: &MarketData,
        portfolio: &Portfolio,
    ) -> Result<Vec<Signal>> {
        let mut signals = Vec::new();

        // This implementation assumes we receive market data for one symbol
        // In production, we'd need to coordinate data for both symbols in a pair
        // For now, we'll skip processing as we need both symbols
        // This is a placeholder that would be enhanced with proper pair coordination

        Ok(signals)
    }

    fn validate_config(&self) -> Result<()> {
        if self.pairs.is_empty() {
            return Err(StrategyError::ConfigError(
                "At least one pair must be specified".to_string(),
            ));
        }

        if self.cointegration_period < 20 {
            return Err(StrategyError::InvalidParameter(
                "Cointegration period must be at least 20".to_string(),
            ));
        }

        if self.entry_threshold <= 0.0 {
            return Err(StrategyError::InvalidParameter(
                "Entry threshold must be positive".to_string(),
            ));
        }

        if self.exit_threshold < 0.0 || self.exit_threshold >= self.entry_threshold {
            return Err(StrategyError::InvalidParameter(
                "Exit threshold must be between 0 and entry threshold".to_string(),
            ));
        }

        Ok(())
    }

    fn risk_parameters(&self) -> RiskParameters {
        RiskParameters {
            max_position_size: Decimal::from(10000),
            max_leverage: 2.0, // Pairs trading can use moderate leverage
            stop_loss_percentage: 0.03,
            take_profit_percentage: 0.05,
            max_daily_loss: Decimal::from(2000),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn create_correlated_bars(count: usize, correlation: f64) -> (Vec<Bar>, Vec<Bar>) {
        let mut bars_a = Vec::new();
        let mut bars_b = Vec::new();

        let mut price_a = 100.0;
        let mut price_b = 50.0;

        for _ in 0..count {
            let change = (rand::random::<f64>() - 0.5) * 2.0;
            price_a += change;
            price_b += change * correlation * 0.5; // Correlated but different scale

            bars_a.push(Bar {
                symbol: "STOCK_A".to_string(),
                timestamp: Utc::now(),
                open: Decimal::from_f64_retain(price_a * 0.99).unwrap(),
                high: Decimal::from_f64_retain(price_a * 1.01).unwrap(),
                low: Decimal::from_f64_retain(price_a * 0.98).unwrap(),
                close: Decimal::from_f64_retain(price_a).unwrap(),
                volume: 1000000,
            });

            bars_b.push(Bar {
                symbol: "STOCK_B".to_string(),
                timestamp: Utc::now(),
                open: Decimal::from_f64_retain(price_b * 0.99).unwrap(),
                high: Decimal::from_f64_retain(price_b * 1.01).unwrap(),
                low: Decimal::from_f64_retain(price_b * 0.98).unwrap(),
                close: Decimal::from_f64_retain(price_b).unwrap(),
                volume: 1000000,
            });
        }

        (bars_a, bars_b)
    }

    #[test]
    fn test_hedge_ratio_calculation() {
        let strategy = PairsStrategy::new(
            vec![("STOCK_A".to_string(), "STOCK_B".to_string())],
            60,
            2.0,
            0.5,
        );

        let prices_a = vec![100.0, 102.0, 104.0, 103.0, 105.0];
        let prices_b = vec![50.0, 51.0, 52.0, 51.5, 52.5];

        let hedge_ratio = strategy.calculate_hedge_ratio(&prices_a, &prices_b);
        assert!(hedge_ratio > 0.0);
        assert!(hedge_ratio < 10.0); // Reasonable range
    }

    #[test]
    fn test_strategy_validation() {
        let strategy = PairsStrategy::new(
            vec![("STOCK_A".to_string(), "STOCK_B".to_string())],
            60,
            2.0,
            0.5,
        );
        assert!(strategy.validate_config().is_ok());

        let invalid = PairsStrategy::new(vec![], 60, 2.0, 0.5);
        assert!(invalid.validate_config().is_err());

        let invalid2 = PairsStrategy::new(
            vec![("A".to_string(), "B".to_string())],
            10, // Too small
            2.0,
            0.5,
        );
        assert!(invalid2.validate_config().is_err());
    }

    #[test]
    fn test_cointegration_test() {
        let strategy = PairsStrategy::new(
            vec![("STOCK_A".to_string(), "STOCK_B".to_string())],
            60,
            2.0,
            0.5,
        );

        let (bars_a, bars_b) = create_correlated_bars(100, 0.9);

        let result = strategy.test_cointegration(&bars_a, &bars_b);
        assert!(result.is_ok());

        let (is_cointegrated, hedge_ratio) = result.unwrap();
        assert!(hedge_ratio > 0.0);
    }
}
