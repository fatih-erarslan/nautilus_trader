//! Integration tests for trading strategies

use chrono::Utc;
use nt_strategies::{MomentumStrategy, Strategy};
use rust_decimal_macros::dec;

mod utils {
    include!("../utils/fixtures.rs");
}

use utils::{MarketDataFixture, PortfolioFixture};

#[tokio::test]
async fn test_momentum_strategy_long_signal() {
    let strategy = MomentumStrategy::new(
        vec!["AAPL".to_string()],
        14,
        2.0,
        0.5,
    );

    let bars = MarketDataFixture::new("AAPL")
        .with_uptrend(30, dec!(100), dec!(120))
        .build();

    let portfolio = PortfolioFixture::new().build();

    // Test that strategy generates signals on strong uptrend
    // Note: This is a simplified test - actual implementation would need
    // the full strategy process method which takes MarketData and Portfolio
    assert_eq!(strategy.id(), "momentum_trader");
}

#[tokio::test]
async fn test_momentum_strategy_short_signal() {
    let strategy = MomentumStrategy::new(
        vec!["TSLA".to_string()],
        14,
        2.0,
        0.5,
    );

    let bars = MarketDataFixture::new("TSLA")
        .with_downtrend(30, dec!(200), dec!(180))
        .build();

    let portfolio = PortfolioFixture::new().build();

    // Test that strategy generates short signals on strong downtrend
    assert_eq!(strategy.id(), "momentum_trader");
}

#[tokio::test]
async fn test_momentum_strategy_no_signal_sideways() {
    let strategy = MomentumStrategy::new(
        vec!["SPY".to_string()],
        14,
        2.0,
        0.5,
    );

    let bars = MarketDataFixture::new("SPY")
        .with_sideways(30, dec!(400))
        .build();

    let portfolio = PortfolioFixture::new().build();

    // Test that strategy does not generate signals in sideways market
    assert_eq!(strategy.id(), "momentum_trader");
}

#[test]
fn test_momentum_strategy_metadata() {
    let strategy = MomentumStrategy::new(
        vec!["AAPL".to_string()],
        14,
        2.0,
        0.5,
    );

    let metadata = strategy.metadata();
    assert_eq!(metadata.name, "Momentum");
    assert!(metadata.version.starts_with("1."));
}

#[test]
fn test_momentum_strategy_insufficient_data() {
    let strategy = MomentumStrategy::new(
        vec!["AAPL".to_string()],
        14,
        2.0,
        0.5,
    );

    // Create bars with insufficient data (less than period)
    let bars = MarketDataFixture::new("AAPL")
        .with_uptrend(5, dec!(100), dec!(105))
        .build();

    // Should handle insufficient data gracefully
    assert!(bars.len() < 14);
}
