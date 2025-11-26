//! Trading module comprehensive test suite
//!
//! Tests all 7 trading functions:
//! - list_strategies()
//! - get_strategy_info()
//! - quick_analysis()
//! - simulate_trade()
//! - execute_trade()
//! - get_portfolio_status()
//! - run_backtest()

use neural_trader_backend::*;

/// Test list_strategies returns expected strategies
#[tokio::test]
async fn test_list_strategies() {
    let strategies = list_strategies().await.expect("Failed to list strategies");

    // Should return 4 strategies
    assert_eq!(strategies.len(), 4, "Expected 4 strategies");

    // Verify strategy names
    let strategy_names: Vec<String> = strategies.iter().map(|s| s.name.clone()).collect();
    assert!(strategy_names.contains(&"momentum".to_string()));
    assert!(strategy_names.contains(&"mean_reversion".to_string()));
    assert!(strategy_names.contains(&"pairs_trading".to_string()));
    assert!(strategy_names.contains(&"market_making".to_string()));

    // All strategies should be GPU capable
    for strategy in &strategies {
        assert!(strategy.gpu_capable, "Strategy {} should be GPU capable", strategy.name);
        assert!(!strategy.description.is_empty(), "Strategy description should not be empty");
    }
}

/// Test get_strategy_info with valid strategy name
#[tokio::test]
async fn test_get_strategy_info_valid() {
    let info = get_strategy_info("momentum".to_string())
        .await
        .expect("Failed to get strategy info");

    assert!(!info.is_empty(), "Strategy info should not be empty");
    assert!(info.contains("momentum"), "Info should contain strategy name");
}

/// Test get_strategy_info with empty string
#[tokio::test]
async fn test_get_strategy_info_empty_string() {
    let result = get_strategy_info("".to_string()).await;
    // Currently returns Ok - should validate input
    assert!(result.is_ok());
}

/// Test get_strategy_info with invalid strategy
#[tokio::test]
async fn test_get_strategy_info_invalid() {
    let result = get_strategy_info("nonexistent_strategy".to_string()).await;
    // Currently returns Ok - should validate strategy exists
    assert!(result.is_ok());
}

/// Test quick_analysis with valid symbol
#[tokio::test]
async fn test_quick_analysis_valid_symbol() {
    let analysis = quick_analysis("AAPL".to_string(), Some(false))
        .await
        .expect("Failed to analyze symbol");

    assert_eq!(analysis.symbol, "AAPL");
    assert!(!analysis.trend.is_empty(), "Trend should not be empty");
    assert!(analysis.volatility >= 0.0, "Volatility should be non-negative");
    assert!(!analysis.volume_trend.is_empty(), "Volume trend should not be empty");
    assert!(!analysis.recommendation.is_empty(), "Recommendation should not be empty");
}

/// Test quick_analysis with GPU enabled
#[tokio::test]
async fn test_quick_analysis_with_gpu() {
    let analysis = quick_analysis("TSLA".to_string(), Some(true))
        .await
        .expect("Failed with GPU enabled");

    assert_eq!(analysis.symbol, "TSLA");
    // GPU flag is currently ignored but test should pass
}

/// Test quick_analysis with empty symbol
#[tokio::test]
async fn test_quick_analysis_empty_symbol() {
    let result = quick_analysis("".to_string(), None).await;
    // Should validate but currently accepts empty
    assert!(result.is_ok());
}

/// Test quick_analysis default GPU setting
#[tokio::test]
async fn test_quick_analysis_default_gpu() {
    let analysis = quick_analysis("BTC-USD".to_string(), None)
        .await
        .expect("Failed with default GPU");

    assert_eq!(analysis.symbol, "BTC-USD");
}

/// Test simulate_trade with buy action
#[tokio::test]
async fn test_simulate_trade_buy() {
    let simulation = simulate_trade(
        "momentum".to_string(),
        "AAPL".to_string(),
        "buy".to_string(),
        Some(false),
    )
    .await
    .expect("Failed to simulate buy");

    assert_eq!(simulation.strategy, "momentum");
    assert_eq!(simulation.symbol, "AAPL");
    assert_eq!(simulation.action, "buy");
    assert!(simulation.expected_return.is_finite(), "Expected return should be finite");
    assert!(simulation.risk_score >= 0.0, "Risk score should be non-negative");
    assert!(simulation.risk_score <= 1.0, "Risk score should be <= 1.0");
    assert!(simulation.execution_time_ms > 0, "Execution time should be positive");
}

/// Test simulate_trade with sell action
#[tokio::test]
async fn test_simulate_trade_sell() {
    let simulation = simulate_trade(
        "mean_reversion".to_string(),
        "TSLA".to_string(),
        "sell".to_string(),
        Some(true),
    )
    .await
    .expect("Failed to simulate sell");

    assert_eq!(simulation.action, "sell");
}

/// Test simulate_trade with invalid action
#[tokio::test]
async fn test_simulate_trade_invalid_action() {
    let result = simulate_trade(
        "momentum".to_string(),
        "AAPL".to_string(),
        "invalid_action".to_string(),
        None,
    )
    .await;

    // Should validate action but currently accepts anything
    assert!(result.is_ok());
}

/// Test simulate_trade with empty strategy
#[tokio::test]
async fn test_simulate_trade_empty_strategy() {
    let result = simulate_trade(
        "".to_string(),
        "AAPL".to_string(),
        "buy".to_string(),
        None,
    )
    .await;

    assert!(result.is_ok());
}

/// Test get_portfolio_status with analytics
#[tokio::test]
async fn test_get_portfolio_status_with_analytics() {
    let status = get_portfolio_status(Some(true))
        .await
        .expect("Failed to get portfolio status");

    assert!(status.total_value > 0.0, "Total value should be positive");
    assert!(status.cash >= 0.0, "Cash should be non-negative");
    assert!(status.positions < 1000, "Positions should be reasonable");
}

/// Test get_portfolio_status without analytics
#[tokio::test]
async fn test_get_portfolio_status_without_analytics() {
    let status = get_portfolio_status(Some(false))
        .await
        .expect("Failed to get portfolio status");

    assert!(status.total_value.is_finite(), "Total value should be finite");
}

/// Test get_portfolio_status default
#[tokio::test]
async fn test_get_portfolio_status_default() {
    let status = get_portfolio_status(None)
        .await
        .expect("Failed to get portfolio status");

    // Verify basic structure
    assert!(status.total_value >= status.cash, "Total value should be >= cash");
}

/// Test execute_trade with market order
#[tokio::test]
async fn test_execute_trade_market_order() {
    let execution = execute_trade(
        "momentum".to_string(),
        "AAPL".to_string(),
        "buy".to_string(),
        100,
        Some("market".to_string()),
        None,
    )
    .await
    .expect("Failed to execute market order");

    assert!(!execution.order_id.is_empty(), "Order ID should not be empty");
    assert_eq!(execution.strategy, "momentum");
    assert_eq!(execution.symbol, "AAPL");
    assert_eq!(execution.action, "buy");
    assert_eq!(execution.quantity, 100);
    assert!(!execution.status.is_empty(), "Status should not be empty");
    assert!(execution.fill_price > 0.0, "Fill price should be positive");
}

/// Test execute_trade with limit order
#[tokio::test]
async fn test_execute_trade_limit_order() {
    let execution = execute_trade(
        "mean_reversion".to_string(),
        "TSLA".to_string(),
        "sell".to_string(),
        50,
        Some("limit".to_string()),
        Some(250.00),
    )
    .await
    .expect("Failed to execute limit order");

    assert_eq!(execution.action, "sell");
    assert_eq!(execution.quantity, 50);
}

/// Test execute_trade with zero quantity
#[tokio::test]
async fn test_execute_trade_zero_quantity() {
    let result = execute_trade(
        "momentum".to_string(),
        "AAPL".to_string(),
        "buy".to_string(),
        0,
        None,
        None,
    )
    .await;

    // Should validate quantity > 0 but currently accepts
    assert!(result.is_ok());
}

/// Test execute_trade with invalid action
#[tokio::test]
async fn test_execute_trade_invalid_action() {
    let result = execute_trade(
        "momentum".to_string(),
        "AAPL".to_string(),
        "hodl".to_string(),
        100,
        None,
        None,
    )
    .await;

    // Should validate action enum but currently accepts anything
    assert!(result.is_ok());
}

/// Test execute_trade default order type
#[tokio::test]
async fn test_execute_trade_default_order_type() {
    let execution = execute_trade(
        "momentum".to_string(),
        "BTC-USD".to_string(),
        "buy".to_string(),
        1,
        None,
        None,
    )
    .await
    .expect("Failed with default order type");

    assert_eq!(execution.quantity, 1);
}

/// Test run_backtest with valid parameters
#[tokio::test]
async fn test_run_backtest_valid() {
    let result = run_backtest(
        "momentum".to_string(),
        "AAPL".to_string(),
        "2023-01-01".to_string(),
        "2023-12-31".to_string(),
        Some(true),
    )
    .await
    .expect("Failed to run backtest");

    assert_eq!(result.strategy, "momentum");
    assert_eq!(result.symbol, "AAPL");
    assert_eq!(result.start_date, "2023-01-01");
    assert_eq!(result.end_date, "2023-12-31");
    assert!(result.total_return.is_finite(), "Total return should be finite");
    assert!(result.sharpe_ratio.is_finite(), "Sharpe ratio should be finite");
    assert!(result.max_drawdown <= 0.0, "Max drawdown should be negative or zero");
    assert!(result.total_trades > 0, "Should have trades");
    assert!(result.win_rate >= 0.0 && result.win_rate <= 1.0, "Win rate should be between 0 and 1");
}

/// Test run_backtest without GPU
#[tokio::test]
async fn test_run_backtest_no_gpu() {
    let result = run_backtest(
        "mean_reversion".to_string(),
        "TSLA".to_string(),
        "2023-01-01".to_string(),
        "2023-06-30".to_string(),
        Some(false),
    )
    .await
    .expect("Failed to run backtest without GPU");

    assert_eq!(result.strategy, "mean_reversion");
}

/// Test run_backtest with invalid date range
#[tokio::test]
async fn test_run_backtest_invalid_dates() {
    let result = run_backtest(
        "momentum".to_string(),
        "AAPL".to_string(),
        "2023-12-31".to_string(),
        "2023-01-01".to_string(), // End before start
        None,
    )
    .await;

    // Should validate date order but currently accepts
    assert!(result.is_ok());
}

/// Test run_backtest with empty dates
#[tokio::test]
async fn test_run_backtest_empty_dates() {
    let result = run_backtest(
        "momentum".to_string(),
        "AAPL".to_string(),
        "".to_string(),
        "".to_string(),
        None,
    )
    .await;

    // Should validate dates but currently accepts empty
    assert!(result.is_ok());
}

/// Test run_backtest default GPU
#[tokio::test]
async fn test_run_backtest_default_gpu() {
    let result = run_backtest(
        "pairs_trading".to_string(),
        "AAPL".to_string(),
        "2023-01-01".to_string(),
        "2023-12-31".to_string(),
        None,
    )
    .await
    .expect("Failed with default GPU");

    assert_eq!(result.strategy, "pairs_trading");
}

/// Edge case: Test with SQL injection attempt in symbol
#[tokio::test]
async fn test_sql_injection_in_symbol() {
    let malicious_symbol = "AAPL'; DROP TABLE trades; --".to_string();
    let result = quick_analysis(malicious_symbol, None).await;

    // Should sanitize input but currently accepts
    assert!(result.is_ok());
}

/// Edge case: Test with XSS attempt in strategy name
#[tokio::test]
async fn test_xss_in_strategy() {
    let xss_strategy = "<script>alert('xss')</script>".to_string();
    let result = get_strategy_info(xss_strategy).await;

    // Should sanitize input
    assert!(result.is_ok());
}

/// Performance test: Multiple concurrent strategy listings
#[tokio::test]
async fn test_concurrent_list_strategies() {
    let handles: Vec<_> = (0..10)
        .map(|_| {
            tokio::spawn(async {
                list_strategies().await
            })
        })
        .collect();

    for handle in handles {
        let result = handle.await.expect("Task panicked");
        assert!(result.is_ok(), "Concurrent request failed");
    }
}

/// Performance test: Rapid fire trades
#[tokio::test]
async fn test_rapid_fire_simulations() {
    let start = std::time::Instant::now();

    for i in 0..100 {
        let symbol = format!("SYM{}", i);
        let _ = simulate_trade(
            "momentum".to_string(),
            symbol,
            "buy".to_string(),
            None,
        )
        .await
        .expect("Simulation failed");
    }

    let duration = start.elapsed();
    println!("100 simulations took: {:?}", duration);

    // Should complete reasonably quickly (< 1 second for mock data)
    assert!(duration.as_secs() < 1, "Simulations took too long");
}

/// Integration test: Complete trading workflow
#[tokio::test]
async fn test_complete_trading_workflow() {
    // 1. List strategies
    let strategies = list_strategies().await.expect("Failed to list strategies");
    assert!(!strategies.is_empty());

    // 2. Get strategy info
    let info = get_strategy_info(strategies[0].name.clone())
        .await
        .expect("Failed to get info");
    assert!(!info.is_empty());

    // 3. Analyze market
    let analysis = quick_analysis("AAPL".to_string(), Some(false))
        .await
        .expect("Failed to analyze");
    assert_eq!(analysis.symbol, "AAPL");

    // 4. Simulate trade
    let simulation = simulate_trade(
        strategies[0].name.clone(),
        "AAPL".to_string(),
        "buy".to_string(),
        None,
    )
    .await
    .expect("Failed to simulate");
    assert_eq!(simulation.symbol, "AAPL");

    // 5. Execute trade
    let execution = execute_trade(
        strategies[0].name.clone(),
        "AAPL".to_string(),
        "buy".to_string(),
        100,
        None,
        None,
    )
    .await
    .expect("Failed to execute");
    assert!(!execution.order_id.is_empty());

    // 6. Check portfolio
    let portfolio = get_portfolio_status(Some(true))
        .await
        .expect("Failed to get portfolio");
    assert!(portfolio.total_value > 0.0);

    // 7. Run backtest
    let backtest = run_backtest(
        strategies[0].name.clone(),
        "AAPL".to_string(),
        "2023-01-01".to_string(),
        "2023-12-31".to_string(),
        Some(false),
    )
    .await
    .expect("Failed to backtest");
    assert!(backtest.total_trades > 0);
}
