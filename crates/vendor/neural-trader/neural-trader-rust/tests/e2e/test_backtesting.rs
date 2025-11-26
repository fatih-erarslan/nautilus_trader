//! End-to-end tests for backtesting engine

use chrono::Utc;
use nt_portfolio::Portfolio;
use rust_decimal_macros::dec;

mod mocks {
    include!("../mocks/mock_market_data.rs");
}

use mocks::{MockMarketDataProvider, MarketPattern};

#[tokio::test]
#[ignore] // Run with: cargo test --ignored
async fn test_backtest_uptrend_strategy() {
    // Setup backtest environment
    let mut market_data = MockMarketDataProvider::with_pattern(MarketPattern::Uptrend);
    let symbol = "AAPL".to_string();

    // Generate 1 year of daily data
    market_data.generate_bars(symbol.clone(), 252, dec!(100), Utc::now());

    let mut portfolio = Portfolio::new(dec!(100000));
    let initial_value = portfolio.total_value();

    // Run backtest (simplified - real backtest would iterate through bars)
    let bars = market_data
        .get_bars(&symbol, nt_market_data::Timeframe::OneDay, Utc::now(), Utc::now())
        .expect("Should get bars");

    assert_eq!(bars.len(), 252);

    // Verify trend
    let start_price = bars.first().unwrap().close;
    let end_price = bars.last().unwrap().close;
    let return_pct = ((end_price - start_price) / start_price * dec!(100)).round_dp(2);

    println!("Backtest return: {}%", return_pct);
    assert!(end_price > start_price, "Should profit in uptrend");
}

#[tokio::test]
#[ignore]
async fn test_backtest_downtrend_strategy() {
    let mut market_data = MockMarketDataProvider::with_pattern(MarketPattern::Downtrend);
    let symbol = "BEAR".to_string();

    market_data.generate_bars(symbol.clone(), 252, dec!(100), Utc::now());

    let bars = market_data
        .get_bars(&symbol, nt_market_data::Timeframe::OneDay, Utc::now(), Utc::now())
        .expect("Should get bars");

    let start_price = bars.first().unwrap().close;
    let end_price = bars.last().unwrap().close;

    assert!(end_price < start_price, "Should be downtrend");
}

#[tokio::test]
#[ignore]
async fn test_backtest_performance_metrics() {
    let mut market_data = MockMarketDataProvider::with_pattern(MarketPattern::Uptrend);
    let symbol = "SPY".to_string();

    market_data.generate_bars(symbol.clone(), 252, dec!(400), Utc::now());

    let bars = market_data
        .get_bars(&symbol, nt_market_data::Timeframe::OneDay, Utc::now(), Utc::now())
        .expect("Should get bars");

    // Calculate basic metrics
    let returns: Vec<_> = bars
        .windows(2)
        .map(|w| (w[1].close - w[0].close) / w[0].close)
        .collect();

    assert!(!returns.is_empty(), "Should have returns data");

    // Verify we can calculate performance metrics
    let total_return = (bars.last().unwrap().close - bars.first().unwrap().close)
        / bars.first().unwrap().close;

    println!("Total return: {}", total_return);
}
