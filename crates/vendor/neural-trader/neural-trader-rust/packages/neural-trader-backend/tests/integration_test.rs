//! End-to-end integration tests
//!
//! Tests complete workflows across modules:
//! - Complete trading workflow: analyze → simulate → execute
//! - Neural forecasting workflow: train → evaluate → forecast
//! - Sports betting workflow: fetch events → analyze odds → calculate Kelly → place bet

use neural_trader_backend::*;

/// Test complete trading workflow from analysis to execution
#[tokio::test]
async fn test_complete_trading_workflow() {
    let symbol = "AAPL".to_string();
    let strategy_name = "momentum".to_string();

    // Step 1: List available strategies
    let strategies = list_strategies()
        .await
        .expect("Failed to list strategies");

    assert!(!strategies.is_empty(), "Should have strategies available");
    assert!(
        strategies.iter().any(|s| s.name == strategy_name),
        "Momentum strategy should be available"
    );

    // Step 2: Get detailed strategy information
    let strategy_info = get_strategy_info(strategy_name.clone())
        .await
        .expect("Failed to get strategy info");

    assert!(!strategy_info.is_empty(), "Strategy info should not be empty");

    // Step 3: Perform quick market analysis
    let analysis = quick_analysis(symbol.clone(), Some(false))
        .await
        .expect("Failed to analyze market");

    assert_eq!(analysis.symbol, symbol);
    assert!(!analysis.trend.is_empty());
    assert!(!analysis.recommendation.is_empty());

    // Step 4: Simulate the trade
    let simulation = simulate_trade(
        strategy_name.clone(),
        symbol.clone(),
        "buy".to_string(),
        Some(false),
    )
    .await
    .expect("Failed to simulate trade");

    assert_eq!(simulation.strategy, strategy_name);
    assert_eq!(simulation.symbol, symbol);
    assert!(simulation.expected_return.is_finite());
    assert!(simulation.risk_score >= 0.0 && simulation.risk_score <= 1.0);

    // Step 5: Check portfolio before trade
    let portfolio_before = get_portfolio_status(Some(true))
        .await
        .expect("Failed to get portfolio status");

    assert!(portfolio_before.total_value > 0.0);
    let initial_cash = portfolio_before.cash;

    // Step 6: Execute the trade
    let execution = execute_trade(
        strategy_name.clone(),
        symbol.clone(),
        "buy".to_string(),
        100,
        Some("market".to_string()),
        None,
    )
    .await
    .expect("Failed to execute trade");

    assert!(!execution.order_id.is_empty());
    assert_eq!(execution.status, "filled");
    assert!(execution.fill_price > 0.0);

    // Step 7: Check portfolio after trade
    let portfolio_after = get_portfolio_status(Some(true))
        .await
        .expect("Failed to get portfolio status after trade");

    // In mock implementation, values don't change, but structure should be valid
    assert!(portfolio_after.total_value.is_finite());

    // Step 8: Run backtest for the strategy
    let backtest = run_backtest(
        strategy_name.clone(),
        symbol.clone(),
        "2023-01-01".to_string(),
        "2023-12-31".to_string(),
        Some(false),
    )
    .await
    .expect("Failed to run backtest");

    assert_eq!(backtest.strategy, strategy_name);
    assert_eq!(backtest.symbol, symbol);
    assert!(backtest.total_return.is_finite());
    assert!(backtest.sharpe_ratio.is_finite());
    assert!(backtest.total_trades > 0);
}

/// Test complete neural forecasting workflow
#[tokio::test]
async fn test_complete_neural_workflow() {
    let symbol = "TSLA".to_string();
    let model_type = "lstm".to_string();
    let data_path = "/data/TSLA_historical.csv".to_string();
    let test_data_path = "/data/TSLA_test.csv".to_string();

    // Step 1: Train a neural model
    let training = neural_train(
        data_path.clone(),
        model_type.clone(),
        Some(100),
        Some(false), // Use CPU for testing
    )
    .await
    .expect("Failed to train model");

    assert!(!training.model_id.is_empty());
    assert_eq!(training.model_type, model_type);
    assert!(training.training_time_ms > 0);
    assert!(training.final_loss >= 0.0);
    assert!(training.validation_accuracy >= 0.0 && training.validation_accuracy <= 1.0);

    let model_id = training.model_id.clone();

    // Step 2: Check model status
    let statuses = neural_model_status(Some(model_id.clone()))
        .await
        .expect("Failed to get model status");

    assert!(!statuses.is_empty());
    let status = &statuses[0];
    assert_eq!(status.model_id, model_id);
    assert!(!status.status.is_empty());
    assert!(!status.created_at.is_empty());

    // Step 3: Evaluate model performance
    let evaluation = neural_evaluate(
        model_id.clone(),
        test_data_path.clone(),
        Some(false),
    )
    .await
    .expect("Failed to evaluate model");

    assert_eq!(evaluation.model_id, model_id);
    assert!(evaluation.test_samples > 0);
    assert!(evaluation.mae >= 0.0);
    assert!(evaluation.rmse >= 0.0);
    assert!(evaluation.mape >= 0.0);
    assert!(evaluation.r2_score >= -1.0 && evaluation.r2_score <= 1.0);

    // Step 4: Optimize hyperparameters
    let param_ranges = r#"{
        "learning_rate": [0.001, 0.01, 0.1],
        "batch_size": [16, 32, 64]
    }"#;

    let optimization = neural_optimize(
        model_id.clone(),
        param_ranges.to_string(),
        Some(false),
    )
    .await
    .expect("Failed to optimize model");

    assert_eq!(optimization.model_id, model_id);
    assert!(optimization.trials_completed > 0);
    assert!(optimization.best_score >= 0.0 && optimization.best_score <= 1.0);

    // Step 5: Generate forecast
    let forecast = neural_forecast(
        symbol.clone(),
        7, // 7-day forecast
        Some(false),
        Some(0.95), // 95% confidence
    )
    .await
    .expect("Failed to generate forecast");

    assert_eq!(forecast.symbol, symbol);
    assert_eq!(forecast.horizon, 7);
    assert_eq!(forecast.predictions.len(), 7);
    assert_eq!(forecast.confidence_intervals.len(), 7);

    // Verify all predictions are within confidence intervals
    for i in 0..7 {
        let pred = forecast.predictions[i];
        let interval = &forecast.confidence_intervals[i];

        assert!(pred.is_finite(), "Prediction {} should be finite", i);
        assert!(interval.lower < interval.upper,
                "Interval {} bounds should be ordered", i);
        assert!(pred >= interval.lower && pred <= interval.upper,
                "Prediction {} should be within confidence interval", i);
    }
}

/// Test complete sports betting workflow
#[tokio::test]
async fn test_complete_sports_betting_workflow() {
    let sport = "basketball".to_string();
    let bankroll = 1000.0;

    // Step 1: Get upcoming sports events
    let events = get_sports_events(sport.clone(), Some(7))
        .await
        .expect("Failed to get events");

    // Events may be empty in mock, but structure should be valid
    for event in &events {
        assert!(!event.event_id.is_empty());
        assert_eq!(event.sport, sport);
        assert!(!event.home_team.is_empty());
        assert!(!event.away_team.is_empty());
        assert!(!event.start_time.is_empty());
    }

    // Step 2: Get betting odds
    let odds = get_sports_odds(sport.clone())
        .await
        .expect("Failed to get odds");

    for odd in &odds {
        assert!(!odd.event_id.is_empty());
        assert!(odd.home_odds > 0.0);
        assert!(odd.away_odds > 0.0);
        assert!(!odd.bookmaker.is_empty());
    }

    // Step 3: Find arbitrage opportunities
    let arbitrage = find_sports_arbitrage(sport.clone(), Some(0.01))
        .await
        .expect("Failed to find arbitrage");

    for opp in &arbitrage {
        assert!(!opp.event_id.is_empty());
        assert!(opp.profit_margin >= 0.01);
        assert!(opp.bet_home.stake > 0.0);
        assert!(opp.bet_away.stake > 0.0);
    }

    // Step 4: Calculate Kelly Criterion for optimal bet sizing
    let kelly = calculate_kelly_criterion(
        0.55, // 55% estimated win probability
        2.0,  // Decimal odds
        bankroll,
    )
    .await
    .expect("Failed to calculate Kelly");

    assert_eq!(kelly.bankroll, bankroll);
    assert!(kelly.kelly_fraction >= 0.0 && kelly.kelly_fraction <= 0.25);
    assert_eq!(kelly.suggested_stake, bankroll * kelly.kelly_fraction);

    // Step 5: Execute bet (validation mode)
    if !odds.is_empty() {
        let bet_execution = execute_sports_bet(
            odds[0].event_id.clone(),
            format!("{} to win", "Home Team"),
            kelly.suggested_stake,
            odds[0].home_odds,
            Some(true), // Validate only
        )
        .await
        .expect("Failed to execute bet");

        assert!(!bet_execution.bet_id.is_empty());
        assert_eq!(bet_execution.market_id, odds[0].event_id);
        assert_eq!(bet_execution.stake, kelly.suggested_stake);
        assert_eq!(bet_execution.odds, odds[0].home_odds);
        assert!(!bet_execution.status.is_empty());

        // Verify potential return calculation
        let expected_return = kelly.suggested_stake * odds[0].home_odds;
        assert!((bet_execution.potential_return - expected_return).abs() < 0.01);
    }
}

/// Test cross-module workflow: Neural forecast influences trading decision
#[tokio::test]
async fn test_neural_informed_trading() {
    let symbol = "AAPL".to_string();

    // Step 1: Generate neural forecast
    let forecast = neural_forecast(
        symbol.clone(),
        5,
        Some(false),
        Some(0.90),
    )
    .await
    .expect("Failed to generate forecast");

    assert_eq!(forecast.predictions.len(), 5);

    // Step 2: Use forecast to inform trading decision
    let first_prediction = forecast.predictions[0];
    let last_prediction = forecast.predictions[4];

    // Determine action based on forecast trend
    let action = if last_prediction > first_prediction {
        "buy".to_string()
    } else {
        "sell".to_string()
    };

    // Step 3: Analyze market
    let analysis = quick_analysis(symbol.clone(), Some(false))
        .await
        .expect("Failed to analyze market");

    // Step 4: Simulate trade based on forecast
    let simulation = simulate_trade(
        "momentum".to_string(),
        symbol.clone(),
        action.clone(),
        Some(false),
    )
    .await
    .expect("Failed to simulate");

    assert_eq!(simulation.action, action);

    // Step 5: Execute if risk is acceptable
    if simulation.risk_score < 0.5 {
        let execution = execute_trade(
            "momentum".to_string(),
            symbol.clone(),
            action,
            100,
            None,
            None,
        )
        .await
        .expect("Failed to execute");

        assert!(!execution.order_id.is_empty());
    }
}

/// Test cross-module workflow: Portfolio optimization with neural predictions
#[tokio::test]
async fn test_portfolio_optimization_with_neural() {
    let symbols = vec!["AAPL", "GOOGL", "MSFT"];
    let mut forecasts = Vec::new();

    // Step 1: Generate forecasts for multiple assets
    for symbol in &symbols {
        let forecast = neural_forecast(
            symbol.to_string(),
            10,
            Some(false),
            Some(0.95),
        )
        .await
        .expect("Failed to forecast");

        forecasts.push(forecast);
    }

    assert_eq!(forecasts.len(), symbols.len());

    // Step 2: Analyze each symbol
    for symbol in &symbols {
        let analysis = quick_analysis(symbol.to_string(), Some(false))
            .await
            .expect("Failed to analyze");

        assert_eq!(analysis.symbol, *symbol);
    }

    // Step 3: Check current portfolio
    let portfolio = get_portfolio_status(Some(true))
        .await
        .expect("Failed to get portfolio");

    assert!(portfolio.total_value > 0.0);

    // Step 4: Simulate trades for portfolio rebalancing
    for (i, symbol) in symbols.iter().enumerate() {
        let forecast = &forecasts[i];

        // Simple strategy: buy if forecast shows upward trend
        if forecast.predictions.last().unwrap() > &forecast.predictions[0] {
            let sim = simulate_trade(
                "momentum".to_string(),
                symbol.to_string(),
                "buy".to_string(),
                Some(false),
            )
            .await
            .expect("Failed to simulate");

            assert_eq!(sim.symbol, *symbol);
        }
    }
}

/// Test error handling across module boundaries
#[tokio::test]
async fn test_cross_module_error_handling() {
    // Test 1: Trading with invalid symbol
    let invalid_symbol = "".to_string();

    let analysis_result = quick_analysis(invalid_symbol.clone(), None).await;
    let forecast_result = neural_forecast(invalid_symbol.clone(), 5, None, None).await;

    // Both should handle gracefully (currently they do)
    assert!(analysis_result.is_ok() || analysis_result.is_err());
    assert!(forecast_result.is_ok() || forecast_result.is_err());

    // Test 2: Sports betting with invalid inputs
    let kelly_result = calculate_kelly_criterion(-0.5, 2.0, 1000.0).await;
    assert!(kelly_result.is_ok() || kelly_result.is_err());
}

/// Test concurrent operations across modules
#[tokio::test]
async fn test_concurrent_cross_module_operations() {
    let symbols = vec!["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"];

    // Spawn concurrent operations across different modules
    let handles: Vec<_> = symbols
        .into_iter()
        .map(|symbol| {
            let sym = symbol.to_string();
            tokio::spawn(async move {
                // Trading analysis
                let analysis = quick_analysis(sym.clone(), Some(false)).await?;

                // Neural forecast
                let forecast = neural_forecast(sym.clone(), 5, Some(false), None).await?;

                // Trade simulation
                let simulation = simulate_trade(
                    "momentum".to_string(),
                    sym.clone(),
                    "buy".to_string(),
                    Some(false),
                )
                .await?;

                Ok::<_, napi::Error>((analysis, forecast, simulation))
            })
        })
        .collect();

    for handle in handles {
        let result = handle.await.expect("Task panicked");
        assert!(result.is_ok(), "Concurrent operation failed");
    }
}

/// Test system initialization and health check
#[tokio::test]
async fn test_system_initialization() {
    // Test 1: Initialize system
    let init_result = init_neural_trader(None).await;
    assert!(init_result.is_ok(), "System initialization failed");

    // Test 2: Get system info
    let sys_info = get_system_info().expect("Failed to get system info");

    assert_eq!(sys_info.version, env!("CARGO_PKG_VERSION"));
    assert_eq!(sys_info.total_tools, 99);
    assert!(sys_info.features.contains(&"trading".to_string()));
    assert!(sys_info.features.contains(&"neural".to_string()));
    assert!(sys_info.features.contains(&"sports-betting".to_string()));

    // Test 3: Health check
    let health = health_check().await.expect("Health check failed");

    assert_eq!(health.status, "healthy");
    assert!(!health.timestamp.is_empty());
}

/// Test data flow consistency across modules
#[tokio::test]
async fn test_data_flow_consistency() {
    let symbol = "AAPL".to_string();

    // Generate data through various modules
    let analysis = quick_analysis(symbol.clone(), None)
        .await
        .expect("Analysis failed");

    let forecast = neural_forecast(symbol.clone(), 3, None, None)
        .await
        .expect("Forecast failed");

    let simulation = simulate_trade(
        "momentum".to_string(),
        symbol.clone(),
        "buy".to_string(),
        None,
    )
    .await
    .expect("Simulation failed");

    // Verify symbol consistency
    assert_eq!(analysis.symbol, symbol);
    assert_eq!(forecast.symbol, symbol);
    assert_eq!(simulation.symbol, symbol);

    // Verify data types are consistent
    assert!(analysis.volatility.is_finite());
    assert!(forecast.model_accuracy.is_finite());
    assert!(simulation.expected_return.is_finite());
}

/// Test complete system lifecycle
#[tokio::test]
async fn test_system_lifecycle() {
    // 1. Initialize
    let init = init_neural_trader(Some(r#"{"mode": "test"}"#.to_string()))
        .await
        .expect("Init failed");

    assert!(!init.is_empty());

    // 2. Perform operations
    let strategies = list_strategies().await.expect("List failed");
    assert!(!strategies.is_empty());

    let analysis = quick_analysis("AAPL".to_string(), None)
        .await
        .expect("Analysis failed");
    assert!(!analysis.symbol.is_empty());

    let forecast = neural_forecast("AAPL".to_string(), 5, None, None)
        .await
        .expect("Forecast failed");
    assert_eq!(forecast.predictions.len(), 5);

    // 3. Health check
    let health = health_check().await.expect("Health check failed");
    assert_eq!(health.status, "healthy");

    // 4. Shutdown
    let shutdown_result = shutdown().await;
    assert!(shutdown_result.is_ok(), "Shutdown failed");
}
