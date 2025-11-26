// Integration Test Scenario 3: Backtesting with Risk Management
// Tests complete backtesting workflow with strategy execution

use neural_trader_backtesting::engine::BacktestEngine;
use neural_trader_strategies::pairs::PairsStrategy;
use neural_trader_risk::manager::RiskManager;
use neural_trader_core::types::*;
use rust_decimal_macros::dec;
use polars::prelude::*;
use chrono::{Utc, Duration};

#[tokio::test]
async fn test_backtest_with_pairs_strategy() -> anyhow::Result<()> {
    // Generate synthetic historical data
    let dates: Vec<_> = (0..1000)
        .map(|i| Utc::now() - Duration::days(1000 - i))
        .collect();

    // Create cointegrated pair
    let aapl_prices: Vec<f64> = (0..1000)
        .map(|i| 150.0 + (i as f64 * 0.05) + ((i as f64 / 20.0).sin() * 3.0))
        .collect();

    let msft_prices: Vec<f64> = (0..1000)
        .map(|i| 300.0 + (i as f64 * 0.1) + ((i as f64 / 20.0).sin() * 6.0))
        .collect();

    let aapl_df = DataFrame::new(vec![
        Series::new("timestamp", dates.clone()),
        Series::new("symbol", vec!["AAPL"; 1000]),
        Series::new("close", aapl_prices),
    ])?;

    let msft_df = DataFrame::new(vec![
        Series::new("timestamp", dates.clone()),
        Series::new("symbol", vec!["MSFT"; 1000]),
        Series::new("close", msft_prices),
    ])?;

    // Combine data
    let historical_data = aapl_df.vstack(&msft_df)?;

    // Create pairs strategy
    let strategy = PairsStrategy::new(
        "AAPL".to_string(),
        "MSFT".to_string(),
        20,           // lookback
        dec!(2.0),    // entry_threshold
        dec!(0.5),    // exit_threshold
    );

    // Create risk manager
    let risk_manager = RiskManager::new(
        dec!(100000), // initial capital
        dec!(0.02),   // max risk per trade
        dec!(0.06),   // max portfolio risk
    );

    // Create backtest engine
    let mut engine = BacktestEngine::new(
        Box::new(strategy),
        risk_manager,
        dec!(100000), // initial capital
        dec!(0.001),  // commission rate (0.1%)
    );

    // Run backtest
    let results = engine.run(&historical_data).await?;

    // Validate results
    assert!(results.total_trades > 0, "Should execute trades");
    assert!(results.final_capital > dec!(0), "Should have positive final capital");
    assert!(
        results.sharpe_ratio.is_some(),
        "Should calculate Sharpe ratio"
    );
    assert!(
        results.max_drawdown < dec!(1.0),
        "Max drawdown should be less than 100%"
    );

    // Check risk management was applied
    assert!(
        results.max_position_size <= dec!(100000) * dec!(0.06),
        "Position sizes should respect risk limits"
    );

    println!("✅ Backtest Results:");
    println!("   Total Trades: {}", results.total_trades);
    println!("   Final Capital: ${}", results.final_capital);
    println!("   Return: {:.2}%", results.total_return * dec!(100));
    println!("   Sharpe Ratio: {:.2}", results.sharpe_ratio.unwrap_or(dec!(0)));
    println!("   Max Drawdown: {:.2}%", results.max_drawdown * dec!(100));

    Ok(())
}

#[tokio::test]
async fn test_backtest_performance_metrics() -> anyhow::Result<()> {
    use neural_trader_backtesting::metrics::PerformanceMetrics;

    // Test metrics calculation
    let returns = vec![
        dec!(0.01),
        dec!(0.02),
        dec!(-0.01),
        dec!(0.015),
        dec!(-0.005),
    ];

    let metrics = PerformanceMetrics::calculate(&returns);

    assert!(metrics.total_return != dec!(0), "Should calculate total return");
    assert!(metrics.sharpe_ratio > dec!(0), "Should calculate Sharpe ratio");
    assert!(metrics.max_drawdown >= dec!(0), "Max drawdown should be non-negative");
    assert!(metrics.win_rate >= dec!(0) && metrics.win_rate <= dec!(1), "Win rate should be 0-1");

    Ok(())
}

#[tokio::test]
async fn test_backtest_with_slippage() -> anyhow::Result<()> {
    // Test that slippage is properly applied
    use neural_trader_backtesting::config::BacktestConfig;

    let config = BacktestConfig {
        initial_capital: dec!(100000),
        commission_rate: dec!(0.001),
        slippage_bps: 5, // 5 basis points slippage
        use_bid_ask: false,
    };

    // Verify slippage calculation
    let price = dec!(100.0);
    let slippage = config.calculate_slippage(price);

    assert_eq!(slippage, dec!(0.05), "Slippage should be 5 bps of price");

    Ok(())
}

#[tokio::test]
async fn test_walk_forward_optimization() -> anyhow::Result<()> {
    // Test walk-forward analysis capability
    use neural_trader_backtesting::optimization::WalkForwardOptimizer;

    let optimizer = WalkForwardOptimizer::new(
        252,  // training_window (1 year)
        63,   // testing_window (3 months)
        21,   // step_size (1 month)
    );

    assert_eq!(optimizer.training_window(), 252);
    assert_eq!(optimizer.testing_window(), 63);

    Ok(())
}
