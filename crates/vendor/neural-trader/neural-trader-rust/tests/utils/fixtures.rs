//! Test fixtures and data builders

use chrono::{DateTime, Duration, Utc};
use nt_core::types::Symbol;
use nt_market_data::Bar;
use nt_portfolio::Portfolio;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;

/// Builder for creating test portfolios
pub struct PortfolioFixture {
    cash: Decimal,
}

impl PortfolioFixture {
    pub fn new() -> Self {
        Self {
            cash: dec!(100000),
        }
    }

    pub fn with_cash(mut self, cash: Decimal) -> Self {
        self.cash = cash;
        self
    }

    pub fn build(self) -> Portfolio {
        Portfolio::new(self.cash)
    }
}

impl Default for PortfolioFixture {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for creating test market data
pub struct MarketDataFixture {
    symbol: Symbol,
    bars: Vec<Bar>,
}

impl MarketDataFixture {
    pub fn new(symbol: impl Into<String>) -> Self {
        Self {
            symbol: symbol.into(),
            bars: Vec::new(),
        }
    }

    /// Add an uptrend pattern
    pub fn with_uptrend(mut self, bars: usize, start: Decimal, end: Decimal) -> Self {
        let increment = (end - start) / Decimal::from(bars);
        let start_time = Utc::now();

        for i in 0..bars {
            let close = start + increment * Decimal::from(i);
            self.bars.push(Bar {
                timestamp: start_time + Duration::minutes(i as i64),
                open: close - dec!(0.01),
                high: close + dec!(0.02),
                low: close - dec!(0.02),
                close,
                volume: dec!(10000),
            });
        }

        self
    }

    /// Add a downtrend pattern
    pub fn with_downtrend(mut self, bars: usize, start: Decimal, end: Decimal) -> Self {
        let decrement = (start - end) / Decimal::from(bars);
        let start_time = Utc::now();

        for i in 0..bars {
            let close = start - decrement * Decimal::from(i);
            self.bars.push(Bar {
                timestamp: start_time + Duration::minutes(i as i64),
                open: close + dec!(0.01),
                high: close + dec!(0.02),
                low: close - dec!(0.02),
                close,
                volume: dec!(10000),
            });
        }

        self
    }

    /// Add sideways movement
    pub fn with_sideways(mut self, bars: usize, price: Decimal) -> Self {
        let start_time = Utc::now();

        for i in 0..bars {
            self.bars.push(Bar {
                timestamp: start_time + Duration::minutes(i as i64),
                open: price,
                high: price + dec!(0.01),
                low: price - dec!(0.01),
                close: price,
                volume: dec!(10000),
            });
        }

        self
    }

    pub fn build(self) -> Vec<Bar> {
        self.bars
    }
}

/// Create a simple test bar
pub fn create_test_bar(price: Decimal, timestamp: DateTime<Utc>) -> Bar {
    Bar {
        timestamp,
        open: price,
        high: price + dec!(0.01),
        low: price - dec!(0.01),
        close: price,
        volume: dec!(1000),
    }
}

/// Create a sequence of test bars
pub fn create_test_bars(count: usize, start_price: Decimal) -> Vec<Bar> {
    let start_time = Utc::now();
    (0..count)
        .map(|i| create_test_bar(start_price, start_time + Duration::minutes(i as i64)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_portfolio_fixture() {
        let portfolio = PortfolioFixture::new()
            .with_cash(dec!(50000))
            .build();

        assert_eq!(portfolio.cash(), dec!(50000));
    }

    #[test]
    fn test_market_data_fixture_uptrend() {
        let bars = MarketDataFixture::new("AAPL")
            .with_uptrend(10, dec!(100), dec!(110))
            .build();

        assert_eq!(bars.len(), 10);
        assert!(bars.last().unwrap().close > bars.first().unwrap().close);
    }
}
