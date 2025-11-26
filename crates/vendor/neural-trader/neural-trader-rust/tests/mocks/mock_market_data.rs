//! Mock market data provider for testing

use chrono::{DateTime, Duration, Utc};
use nt_core::types::Symbol;
use nt_market_data::{Bar, MarketDataProvider, MarketDataError, Timeframe};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::collections::HashMap;

/// Mock market data provider
pub struct MockMarketDataProvider {
    bars: HashMap<Symbol, Vec<Bar>>,
    pattern: MarketPattern,
}

/// Market pattern types for test data generation
#[derive(Debug, Clone, Copy)]
pub enum MarketPattern {
    Uptrend,
    Downtrend,
    Sideways,
    Volatile,
}

impl MockMarketDataProvider {
    /// Create new mock provider with default data
    pub fn new() -> Self {
        Self {
            bars: HashMap::new(),
            pattern: MarketPattern::Sideways,
        }
    }

    /// Create provider with specific pattern
    pub fn with_pattern(pattern: MarketPattern) -> Self {
        Self {
            bars: HashMap::new(),
            pattern,
        }
    }

    /// Add custom bars for a symbol
    pub fn add_bars(&mut self, symbol: Symbol, bars: Vec<Bar>) {
        self.bars.insert(symbol, bars);
    }

    /// Generate synthetic bars with the configured pattern
    pub fn generate_bars(
        &mut self,
        symbol: Symbol,
        count: usize,
        start_price: Decimal,
        start_time: DateTime<Utc>,
    ) {
        let bars = match self.pattern {
            MarketPattern::Uptrend => generate_uptrend(count, start_price, start_time),
            MarketPattern::Downtrend => generate_downtrend(count, start_price, start_time),
            MarketPattern::Sideways => generate_sideways(count, start_price, start_time),
            MarketPattern::Volatile => generate_volatile(count, start_price, start_time),
        };

        self.bars.insert(symbol, bars);
    }
}

impl Default for MockMarketDataProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl MarketDataProvider for MockMarketDataProvider {
    fn get_bars(
        &self,
        symbol: &Symbol,
        _timeframe: Timeframe,
        _start: DateTime<Utc>,
        _end: DateTime<Utc>,
    ) -> Result<Vec<Bar>, MarketDataError> {
        self.bars
            .get(symbol)
            .cloned()
            .ok_or_else(|| MarketDataError::SymbolNotFound(symbol.to_string()))
    }

    fn get_latest_bar(&self, symbol: &Symbol) -> Result<Bar, MarketDataError> {
        self.bars
            .get(symbol)
            .and_then(|bars| bars.last().cloned())
            .ok_or_else(|| MarketDataError::SymbolNotFound(symbol.to_string()))
    }
}

// Helper functions to generate different market patterns

fn generate_uptrend(count: usize, start_price: Decimal, start_time: DateTime<Utc>) -> Vec<Bar> {
    let increment = start_price * dec!(0.001); // 0.1% per bar

    (0..count)
        .map(|i| {
            let price = start_price + increment * Decimal::from(i);
            create_bar(
                price,
                start_time + Duration::minutes(i as i64),
                dec!(1000),
            )
        })
        .collect()
}

fn generate_downtrend(count: usize, start_price: Decimal, start_time: DateTime<Utc>) -> Vec<Bar> {
    let decrement = start_price * dec!(0.001);

    (0..count)
        .map(|i| {
            let price = start_price - decrement * Decimal::from(i);
            create_bar(
                price,
                start_time + Duration::minutes(i as i64),
                dec!(1000),
            )
        })
        .collect()
}

fn generate_sideways(count: usize, start_price: Decimal, start_time: DateTime<Utc>) -> Vec<Bar> {
    (0..count)
        .map(|i| {
            // Small random noise around start price
            let noise = (i % 7) as i64 - 3; // -3 to +3
            let price = start_price + Decimal::new(noise, 2); // ±0.03
            create_bar(
                price,
                start_time + Duration::minutes(i as i64),
                dec!(1000),
            )
        })
        .collect()
}

fn generate_volatile(count: usize, start_price: Decimal, start_time: DateTime<Utc>) -> Vec<Bar> {
    (0..count)
        .map(|i| {
            // Large swings
            let swing = if i % 2 == 0 { 5 } else { -5 };
            let price = start_price + Decimal::new(swing, 1); // ±0.5
            create_bar(
                price,
                start_time + Duration::minutes(i as i64),
                dec!(2000),
            )
        })
        .collect()
}

fn create_bar(close: Decimal, timestamp: DateTime<Utc>, volume: Decimal) -> Bar {
    let spread = close * dec!(0.0001); // 0.01% spread

    Bar {
        timestamp,
        open: close - spread,
        high: close + spread,
        low: close - spread * dec!(2),
        close,
        volume,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_uptrend() {
        let bars = generate_uptrend(10, dec!(100), Utc::now());

        assert_eq!(bars.len(), 10);
        assert!(bars.last().unwrap().close > bars.first().unwrap().close);
    }

    #[test]
    fn test_generate_downtrend() {
        let bars = generate_downtrend(10, dec!(100), Utc::now());

        assert_eq!(bars.len(), 10);
        assert!(bars.last().unwrap().close < bars.first().unwrap().close);
    }

    #[test]
    fn test_mock_provider_get_bars() {
        let mut provider = MockMarketDataProvider::new();
        let symbol = "AAPL".to_string();

        provider.generate_bars(symbol.clone(), 50, dec!(150), Utc::now());

        let bars = provider
            .get_bars(&symbol, Timeframe::OneMinute, Utc::now(), Utc::now())
            .unwrap();

        assert_eq!(bars.len(), 50);
    }
}
