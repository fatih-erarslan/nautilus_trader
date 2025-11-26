//! Integration tests for market data pipeline

use chrono::Utc;
use nt_market_data::{Bar, MarketDataProvider, Timeframe};
use rust_decimal_macros::dec;

mod mocks {
    include!("../mocks/mock_market_data.rs");
}

use mocks::{MockMarketDataProvider, MarketPattern};

#[test]
fn test_market_data_uptrend_generation() {
    let mut provider = MockMarketDataProvider::with_pattern(MarketPattern::Uptrend);
    let symbol = "AAPL".to_string();

    provider.generate_bars(symbol.clone(), 100, dec!(150), Utc::now());

    let bars = provider
        .get_bars(&symbol, Timeframe::OneMinute, Utc::now(), Utc::now())
        .expect("Should get bars");

    assert_eq!(bars.len(), 100);

    // Verify uptrend
    let first_price = bars.first().unwrap().close;
    let last_price = bars.last().unwrap().close;
    assert!(last_price > first_price, "Should be uptrend");
}

#[test]
fn test_market_data_downtrend_generation() {
    let mut provider = MockMarketDataProvider::with_pattern(MarketPattern::Downtrend);
    let symbol = "TSLA".to_string();

    provider.generate_bars(symbol.clone(), 100, dec!(200), Utc::now());

    let bars = provider
        .get_bars(&symbol, Timeframe::OneMinute, Utc::now(), Utc::now())
        .expect("Should get bars");

    assert_eq!(bars.len(), 100);

    // Verify downtrend
    let first_price = bars.first().unwrap().close;
    let last_price = bars.last().unwrap().close;
    assert!(last_price < first_price, "Should be downtrend");
}

#[test]
fn test_market_data_sideways_pattern() {
    let mut provider = MockMarketDataProvider::with_pattern(MarketPattern::Sideways);
    let symbol = "SPY".to_string();

    provider.generate_bars(symbol.clone(), 50, dec!(400), Utc::now());

    let bars = provider
        .get_bars(&symbol, Timeframe::OneMinute, Utc::now(), Utc::now())
        .expect("Should get bars");

    assert_eq!(bars.len(), 50);

    // Verify sideways (prices should stay relatively flat)
    let first_price = bars.first().unwrap().close;
    let last_price = bars.last().unwrap().close;
    let change = ((last_price - first_price) / first_price).abs();
    assert!(change < dec!(0.05), "Should be sideways with < 5% change");
}

#[test]
fn test_market_data_get_latest_bar() {
    let mut provider = MockMarketDataProvider::new();
    let symbol = "AAPL".to_string();

    provider.generate_bars(symbol.clone(), 10, dec!(150), Utc::now());

    let latest = provider.get_latest_bar(&symbol).expect("Should get latest bar");
    let all_bars = provider
        .get_bars(&symbol, Timeframe::OneMinute, Utc::now(), Utc::now())
        .expect("Should get all bars");

    assert_eq!(latest.close, all_bars.last().unwrap().close);
}

#[test]
fn test_market_data_symbol_not_found() {
    let provider = MockMarketDataProvider::new();
    let result = provider.get_latest_bar(&"INVALID".to_string());

    assert!(result.is_err(), "Should error for non-existent symbol");
}
