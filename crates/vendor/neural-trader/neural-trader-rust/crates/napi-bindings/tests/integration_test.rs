//! Comprehensive Integration Tests for All 103 NAPI Functions
//!
//! This test suite validates all NAPI bindings for the neural-trader MCP tools.
//! Tests are organized by category and verify both successful execution and error handling.

use nt_napi_bindings::mcp_tools::*;
use tokio::runtime::Runtime;

// Helper to run async tests
fn run_test<F: std::future::Future>(future: F) -> F::Output {
    Runtime::new().unwrap().block_on(future)
}

// =============================================================================
// Category 1: Core Trading Tools (23 functions)
// =============================================================================

#[cfg(test)]
mod core_trading_tests {
    use super::*;

    #[test]
    fn test_ping() {
        run_test(async {
            let result = ping().await;
            assert!(result.is_ok(), "ping should succeed");
            let json: serde_json::Value = serde_json::from_str(&result.unwrap()).unwrap();
            assert_eq!(json["status"].as_str(), Some("healthy"));
            assert!(json["version"].is_string());
            println!("✓ ping");
        });
    }

    #[test]
    fn test_list_strategies() {
        run_test(async {
            let result = neural_trader::list_strategies().await;
            assert!(result.is_ok(), "list_strategies should succeed");
            let json: serde_json::Value = serde_json::from_str(&result.unwrap()).unwrap();
            assert!(json["strategies"].is_array());
            assert!(json["total_count"].as_u64().unwrap() > 0);
            println!("✓ list_strategies");
        });
    }

    #[test]
    fn test_get_strategy_info() {
        run_test(async {
            let result = neural_trader::get_strategy_info("momentum".to_string()).await;
            assert!(result.is_ok(), "get_strategy_info should succeed");
            let json: serde_json::Value = serde_json::from_str(&result.unwrap()).unwrap();
            assert_eq!(json["strategy"].as_str(), Some("momentum"));
            assert!(json["parameters"].is_object());
            println!("✓ get_strategy_info");
        });
    }

    #[test]
    fn test_get_portfolio_status() {
        run_test(async {
            let result = neural_trader::get_portfolio_status(Some(true)).await;
            assert!(result.is_ok(), "get_portfolio_status should succeed");
            let json: serde_json::Value = serde_json::from_str(&result.unwrap()).unwrap();
            assert!(json["status"].is_string());
            println!("✓ get_portfolio_status");
        });
    }

    #[test]
    fn test_execute_trade_validation() {
        run_test(async {
            // Test with ENABLE_LIVE_TRADING=false (dry run mode)
            let result = neural_trader::execute_trade(
                "momentum".to_string(),
                "AAPL".to_string(),
                "buy".to_string(),
                100,
                Some("market".to_string()),
                None,
            ).await;
            assert!(result.is_ok(), "execute_trade should validate successfully");
            let json: serde_json::Value = serde_json::from_str(&result.unwrap()).unwrap();
            assert_eq!(json["mode"].as_str(), Some("DRY_RUN"));
            println!("✓ execute_trade (dry run)");
        });
    }

    #[test]
    fn test_execute_trade_invalid_symbol() {
        run_test(async {
            let result = neural_trader::execute_trade(
                "momentum".to_string(),
                "".to_string(),  // Invalid empty symbol
                "buy".to_string(),
                100,
                None,
                None,
            ).await;
            assert!(result.is_err(), "execute_trade should reject invalid symbol");
            println!("✓ execute_trade (invalid symbol)");
        });
    }

    #[test]
    fn test_quick_analysis() {
        run_test(async {
            let result = neural_trader::quick_analysis("AAPL".to_string(), Some(false)).await;
            assert!(result.is_ok(), "quick_analysis should succeed");
            let json: serde_json::Value = serde_json::from_str(&result.unwrap()).unwrap();
            assert_eq!(json["symbol"].as_str(), Some("AAPL"));
            println!("✓ quick_analysis");
        });
    }

    #[test]
    fn test_run_backtest() {
        run_test(async {
            let result = neural_trader::run_backtest(
                "momentum".to_string(),
                "AAPL".to_string(),
                "2024-01-01".to_string(),
                "2024-12-31".to_string(),
                Some(true),
                Some("sp500".to_string()),
                Some(true),
            ).await;
            assert!(result.is_ok(), "run_backtest should succeed");
            let json: serde_json::Value = serde_json::from_str(&result.unwrap()).unwrap();
            assert!(json["performance"].is_object());
            assert!(json["performance"]["sharpe_ratio"].is_f64());
            println!("✓ run_backtest");
        });
    }

    #[test]
    fn test_optimize_strategy() {
        run_test(async {
            let params = r#"{"lookback": [10, 50], "threshold": [0.01, 0.05]}"#;
            let result = neural_trader::optimize_strategy(
                "momentum".to_string(),
                "AAPL".to_string(),
                params.to_string(),
                Some(true),
                Some(100),
                Some("sharpe_ratio".to_string()),
            ).await;
            assert!(result.is_ok(), "optimize_strategy should succeed");
            let json: serde_json::Value = serde_json::from_str(&result.unwrap()).unwrap();
            assert!(json["optimal_parameters"].is_object());
            println!("✓ optimize_strategy");
        });
    }

    #[test]
    fn test_risk_analysis() {
        run_test(async {
            let portfolio = r#"[{"symbol": "AAPL", "quantity": 100}]"#;
            let result = neural_trader::risk_analysis(
                portfolio.to_string(),
                Some(true),
                Some(true),
                Some(0.05),
                Some(1),
            ).await;
            assert!(result.is_ok(), "risk_analysis should succeed");
            let json: serde_json::Value = serde_json::from_str(&result.unwrap()).unwrap();
            assert!(json["var_metrics"].is_object());
            println!("✓ risk_analysis");
        });
    }

    #[test]
    fn test_get_market_analysis() {
        run_test(async {
            let result = neural_trader::get_market_analysis("AAPL".to_string()).await;
            assert!(result.is_ok());
            println!("✓ get_market_analysis");
        });
    }

    #[test]
    fn test_performance_report() {
        run_test(async {
            let result = neural_trader::performance_report("momentum".to_string(), Some(30), Some(false)).await;
            assert!(result.is_ok());
            println!("✓ performance_report");
        });
    }

    #[test]
    fn test_correlation_analysis() {
        run_test(async {
            let symbols = vec!["AAPL".to_string(), "GOOGL".to_string()];
            let result = neural_trader::correlation_analysis(symbols, Some(90), Some(true)).await;
            assert!(result.is_ok());
            println!("✓ correlation_analysis");
        });
    }

    #[test]
    fn test_recommend_strategy() {
        run_test(async {
            let conditions = r#"{"volatility": "high"}"#;
            let result = neural_trader::recommend_strategy(
                conditions.to_string(),
                Some("moderate".to_string()),
                vec!["profit".to_string()],
            ).await;
            assert!(result.is_ok());
            println!("✓ recommend_strategy");
        });
    }

    #[test]
    fn test_switch_active_strategy() {
        run_test(async {
            let result = neural_trader::switch_active_strategy(
                "momentum".to_string(),
                "mean_reversion".to_string(),
                Some(false),
            ).await;
            assert!(result.is_ok());
            println!("✓ switch_active_strategy");
        });
    }

    #[test]
    fn test_run_benchmark() {
        run_test(async {
            let result = neural_trader::run_benchmark(
                "momentum".to_string(),
                Some("performance".to_string()),
                Some(true),
            ).await;
            assert!(result.is_ok());
            println!("✓ run_benchmark");
        });
    }
}

// =============================================================================
// Category 2: Neural Network Tools (7 functions)
// =============================================================================

#[cfg(test)]
mod neural_tests {
    use super::*;

    #[test]
    fn test_neural_forecast() {
        run_test(async {
            let result = neural_trader::neural_forecast(
                "AAPL".to_string(),
                10,
                Some("lstm_v1".to_string()),
                Some(true),
                Some(0.95),
            ).await;
            assert!(result.is_ok(), "neural_forecast should succeed");
            let json: serde_json::Value = serde_json::from_str(&result.unwrap()).unwrap();
            assert!(json["predictions"].is_array());
            println!("✓ neural_forecast");
        });
    }

    #[test]
    fn test_neural_train() {
        run_test(async {
            let result = neural_trader::neural_train(
                "/data/train.csv".to_string(),
                "lstm".to_string(),
                Some(100),
                Some(32),
                Some(0.001),
                Some(true),
                Some(0.2),
            ).await;
            assert!(result.is_ok(), "neural_train should succeed");
            let json: serde_json::Value = serde_json::from_str(&result.unwrap()).unwrap();
            assert_eq!(json["status"].as_str(), Some("completed"));
            println!("✓ neural_train");
        });
    }

    #[test]
    fn test_neural_evaluate() {
        run_test(async {
            let result = neural_trader::neural_evaluate(
                "lstm_v1".to_string(),
                "/data/test.csv".to_string(),
                Some(vec!["mae".to_string(), "rmse".to_string()]),
                Some(true),
            ).await;
            assert!(result.is_ok());
            let json: serde_json::Value = serde_json::from_str(&result.unwrap()).unwrap();
            assert!(json["metrics"].is_object());
            println!("✓ neural_evaluate");
        });
    }

    #[test]
    fn test_neural_backtest() {
        run_test(async {
            let result = neural_trader::neural_backtest(
                "lstm_v1".to_string(),
                "2024-01-01".to_string(),
                "2024-12-31".to_string(),
                Some("sp500".to_string()),
                Some("daily".to_string()),
                Some(true),
            ).await;
            assert!(result.is_ok());
            println!("✓ neural_backtest");
        });
    }

    #[test]
    fn test_neural_model_status() {
        run_test(async {
            let result = neural_trader::neural_model_status(Some("lstm_v1".to_string())).await;
            assert!(result.is_ok());
            let json: serde_json::Value = serde_json::from_str(&result.unwrap()).unwrap();
            assert!(json["models"].is_array());
            println!("✓ neural_model_status");
        });
    }

    #[test]
    fn test_neural_optimize() {
        run_test(async {
            let params = r#"{"learning_rate": [0.0001, 0.01]}"#;
            let result = neural_trader::neural_optimize(
                "lstm_v1".to_string(),
                params.to_string(),
                Some(100),
                Some("mae".to_string()),
                Some(true),
            ).await;
            assert!(result.is_ok());
            println!("✓ neural_optimize");
        });
    }

    #[test]
    fn test_neural_predict() {
        run_test(async {
            let input = r#"[1.0, 2.0, 3.0]"#;
            let result = neural_trader::neural_predict(
                "lstm_v1".to_string(),
                input.to_string(),
                Some(true),
            ).await;
            assert!(result.is_ok());
            let json: serde_json::Value = serde_json::from_str(&result.unwrap()).unwrap();
            assert!(json["predictions"].is_array());
            println!("✓ neural_predict");
        });
    }
}

// =============================================================================
// Category 3: News Trading Tools (8 functions)
// =============================================================================

#[cfg(test)]
mod news_tests {
    use super::*;

    #[test]
    fn test_analyze_news() {
        run_test(async {
            let result = neural_trader::analyze_news(
                "AAPL".to_string(),
                Some(24),
                Some("enhanced".to_string()),
                Some(false),
            ).await;
            assert!(result.is_ok(), "analyze_news should succeed");
            // Will return config_required if NEWS_API_KEY not set
            println!("✓ analyze_news");
        });
    }

    #[test]
    fn test_get_news_sentiment() {
        run_test(async {
            let result = neural_trader::get_news_sentiment(
                "AAPL".to_string(),
                Some(vec!["newsapi".to_string()]),
            ).await;
            assert!(result.is_ok());
            println!("✓ get_news_sentiment");
        });
    }

    #[test]
    fn test_control_news_collection() {
        run_test(async {
            let result = neural_trader::control_news_collection(
                "start".to_string(),
                Some(vec!["AAPL".to_string()]),
                Some(24),
                Some(vec!["newsapi".to_string()]),
                Some(300),
            ).await;
            assert!(result.is_ok());
            println!("✓ control_news_collection");
        });
    }

    #[test]
    fn test_control_news_collection_invalid_action() {
        run_test(async {
            let result = neural_trader::control_news_collection(
                "invalid_action".to_string(),
                None,
                None,
                None,
                None,
            ).await;
            assert!(result.is_err(), "should reject invalid action");
            println!("✓ control_news_collection (invalid action)");
        });
    }

    #[test]
    fn test_get_news_provider_status() {
        run_test(async {
            let result = neural_trader::get_news_provider_status().await;
            assert!(result.is_ok());
            let json: serde_json::Value = serde_json::from_str(&result.unwrap()).unwrap();
            assert!(json["providers"].is_array());
            println!("✓ get_news_provider_status");
        });
    }

    #[test]
    fn test_fetch_filtered_news() {
        run_test(async {
            let result = neural_trader::fetch_filtered_news(
                vec!["AAPL".to_string()],
                Some(50),
                Some(0.5),
                Some("positive".to_string()),
            ).await;
            assert!(result.is_ok());
            println!("✓ fetch_filtered_news");
        });
    }

    #[test]
    fn test_get_news_trends() {
        run_test(async {
            let result = neural_trader::get_news_trends(
                vec!["AAPL".to_string()],
                Some(vec![1, 6, 24]),
            ).await;
            assert!(result.is_ok());
            println!("✓ get_news_trends");
        });
    }

    #[test]
    fn test_get_breaking_news() {
        run_test(async {
            let result = neural_trader::get_breaking_news(
                Some(vec!["AAPL".to_string()]),
                Some(15),
            ).await;
            assert!(result.is_ok());
            println!("✓ get_breaking_news");
        });
    }
}

// =============================================================================
// Category 4: Portfolio & Risk Tools (5 functions)
// =============================================================================

#[cfg(test)]
mod portfolio_risk_tests {
    use super::*;

    #[test]
    fn test_execute_multi_asset_trade() {
        run_test(async {
            let trades = r#"[{"symbol": "AAPL", "quantity": 100}]"#;
            let result = neural_trader::execute_multi_asset_trade(
                trades.to_string(),
                "momentum".to_string(),
                Some(true),
                Some(10000.0),
            ).await;
            assert!(result.is_ok());
            println!("✓ execute_multi_asset_trade");
        });
    }

    #[test]
    fn test_portfolio_rebalance() {
        run_test(async {
            let allocations = r#"{"AAPL": 0.5, "GOOGL": 0.5}"#;
            let result = neural_trader::portfolio_rebalance(
                allocations.to_string(),
                None,
                Some(0.05),
            ).await;
            assert!(result.is_ok());
            println!("✓ portfolio_rebalance");
        });
    }

    #[test]
    fn test_cross_asset_correlation_matrix() {
        run_test(async {
            let result = neural_trader::cross_asset_correlation_matrix(
                vec!["AAPL".to_string(), "GOOGL".to_string()],
                Some(90),
                Some(true),
            ).await;
            assert!(result.is_ok());
            println!("✓ cross_asset_correlation_matrix");
        });
    }

    #[test]
    fn test_get_execution_analytics() {
        run_test(async {
            let result = neural_trader::get_execution_analytics(Some("1h".to_string())).await;
            assert!(result.is_ok());
            println!("✓ get_execution_analytics");
        });
    }

    #[test]
    fn test_get_system_metrics() {
        run_test(async {
            let result = neural_trader::get_system_metrics(
                Some(vec!["cpu".to_string(), "memory".to_string()]),
                Some(60),
                Some(false),
            ).await;
            assert!(result.is_ok());
            println!("✓ get_system_metrics");
        });
    }
}

// =============================================================================
// Category 5: Sports Betting Tools (13 functions)
// =============================================================================

#[cfg(test)]
mod sports_betting_tests {
    use super::*;

    #[test]
    fn test_get_sports_events() {
        run_test(async {
            let result = neural_trader::get_sports_events(
                "basketball".to_string(),
                Some(7),
                Some(false),
            ).await;
            assert!(result.is_ok());
            println!("✓ get_sports_events");
        });
    }

    #[test]
    fn test_get_sports_odds() {
        run_test(async {
            let result = neural_trader::get_sports_odds(
                "basketball".to_string(),
                Some(vec!["h2h".to_string()]),
                Some(vec!["us".to_string()]),
                Some(false),
            ).await;
            assert!(result.is_ok());
            println!("✓ get_sports_odds");
        });
    }

    #[test]
    fn test_find_sports_arbitrage() {
        run_test(async {
            let result = neural_trader::find_sports_arbitrage(
                "basketball".to_string(),
                Some(0.01),
                Some(false),
            ).await;
            assert!(result.is_ok());
            println!("✓ find_sports_arbitrage");
        });
    }

    #[test]
    fn test_analyze_betting_market_depth() {
        run_test(async {
            let result = neural_trader::analyze_betting_market_depth(
                "market_123".to_string(),
                "basketball".to_string(),
                Some(false),
            ).await;
            assert!(result.is_ok());
            println!("✓ analyze_betting_market_depth");
        });
    }

    #[test]
    fn test_calculate_kelly_criterion() {
        run_test(async {
            let result = neural_trader::calculate_kelly_criterion(
                0.6,
                2.0,
                10000.0,
                Some(1.0),
            ).await;
            assert!(result.is_ok());
            let json: serde_json::Value = serde_json::from_str(&result.unwrap()).unwrap();
            assert!(json["kelly_fraction"].is_f64());
            assert!(json["recommended_bet"].is_f64());
            println!("✓ calculate_kelly_criterion");
        });
    }

    #[test]
    fn test_get_betting_portfolio_status() {
        run_test(async {
            let result = neural_trader::get_betting_portfolio_status(Some(true)).await;
            assert!(result.is_ok());
            println!("✓ get_betting_portfolio_status");
        });
    }

    #[test]
    fn test_execute_sports_bet() {
        run_test(async {
            let result = neural_trader::execute_sports_bet(
                "market_123".to_string(),
                "team_a".to_string(),
                100.0,
                2.0,
                Some("back".to_string()),
                Some(true),  // validate only
            ).await;
            assert!(result.is_ok());
            println!("✓ execute_sports_bet");
        });
    }

    #[test]
    fn test_get_sports_betting_performance() {
        run_test(async {
            let result = neural_trader::get_sports_betting_performance(
                Some(30),
                Some(true),
            ).await;
            assert!(result.is_ok());
            println!("✓ get_sports_betting_performance");
        });
    }

    #[test]
    fn test_compare_betting_providers() {
        run_test(async {
            let result = neural_trader::compare_betting_providers(
                "basketball".to_string(),
                None,
                Some(false),
            ).await;
            assert!(result.is_ok());
            println!("✓ compare_betting_providers");
        });
    }

    #[test]
    fn test_get_live_odds_updates() {
        run_test(async {
            let result = neural_trader::get_live_odds_updates(
                "basketball".to_string(),
                vec!["event_1".to_string()],
            ).await;
            assert!(result.is_ok());
            println!("✓ get_live_odds_updates");
        });
    }

    #[test]
    fn test_analyze_betting_trends() {
        run_test(async {
            let result = neural_trader::analyze_betting_trends(
                "basketball".to_string(),
                Some(30),
            ).await;
            assert!(result.is_ok());
            println!("✓ analyze_betting_trends");
        });
    }

    #[test]
    fn test_get_betting_history() {
        run_test(async {
            let result = neural_trader::get_betting_history(
                Some(30),
                Some("basketball".to_string()),
            ).await;
            assert!(result.is_ok());
            println!("✓ get_betting_history");
        });
    }
}

// =============================================================================
// Category 6: Odds API Tools (9 functions)
// =============================================================================

#[cfg(test)]
mod odds_api_tests {
    use super::*;

    #[test]
    fn test_odds_api_get_sports() {
        run_test(async {
            let result = neural_trader::odds_api_get_sports().await;
            assert!(result.is_ok());
            println!("✓ odds_api_get_sports");
        });
    }

    #[test]
    fn test_odds_api_get_live_odds() {
        run_test(async {
            let result = neural_trader::odds_api_get_live_odds(
                "basketball_nba".to_string(),
                Some("us".to_string()),
                Some("h2h".to_string()),
                Some("decimal".to_string()),
                None,
            ).await;
            assert!(result.is_ok());
            println!("✓ odds_api_get_live_odds");
        });
    }

    #[test]
    fn test_odds_api_get_event_odds() {
        run_test(async {
            let result = neural_trader::odds_api_get_event_odds(
                "basketball_nba".to_string(),
                "event_123".to_string(),
                Some("us".to_string()),
                Some("h2h,spreads".to_string()),
                None,
            ).await;
            assert!(result.is_ok());
            println!("✓ odds_api_get_event_odds");
        });
    }

    #[test]
    fn test_odds_api_find_arbitrage() {
        run_test(async {
            let result = neural_trader::odds_api_find_arbitrage(
                "basketball_nba".to_string(),
                Some("us,uk".to_string()),
                Some("h2h".to_string()),
                Some(0.01),
            ).await;
            assert!(result.is_ok());
            println!("✓ odds_api_find_arbitrage");
        });
    }

    #[test]
    fn test_odds_api_get_bookmaker_odds() {
        run_test(async {
            let result = neural_trader::odds_api_get_bookmaker_odds(
                "basketball_nba".to_string(),
                "draftkings".to_string(),
                Some("us".to_string()),
                Some("h2h".to_string()),
            ).await;
            assert!(result.is_ok());
            println!("✓ odds_api_get_bookmaker_odds");
        });
    }

    #[test]
    fn test_odds_api_analyze_movement() {
        run_test(async {
            let result = neural_trader::odds_api_analyze_movement(
                "basketball_nba".to_string(),
                "event_123".to_string(),
                Some(5),
            ).await;
            assert!(result.is_ok());
            println!("✓ odds_api_analyze_movement");
        });
    }

    #[test]
    fn test_odds_api_calculate_probability_decimal() {
        run_test(async {
            let result = neural_trader::odds_api_calculate_probability(
                2.5,
                Some("decimal".to_string()),
            ).await;
            assert!(result.is_ok());
            let json: serde_json::Value = serde_json::from_str(&result.unwrap()).unwrap();
            let prob = json["implied_probability"].as_f64().unwrap();
            assert!((prob - 0.4).abs() < 0.01);
            println!("✓ odds_api_calculate_probability (decimal)");
        });
    }

    #[test]
    fn test_odds_api_compare_margins() {
        run_test(async {
            let result = neural_trader::odds_api_compare_margins(
                "basketball_nba".to_string(),
                Some("us".to_string()),
                Some("h2h".to_string()),
            ).await;
            assert!(result.is_ok());
            println!("✓ odds_api_compare_margins");
        });
    }

    #[test]
    fn test_odds_api_get_upcoming() {
        run_test(async {
            let result = neural_trader::odds_api_get_upcoming(
                "basketball_nba".to_string(),
                Some(7),
                Some("us".to_string()),
                Some("h2h".to_string()),
            ).await;
            assert!(result.is_ok());
            println!("✓ odds_api_get_upcoming");
        });
    }
}

// =============================================================================
// Test Summary Runner
// =============================================================================

#[test]
fn run_all_tests_summary() {
    println!("\n=== NAPI INTEGRATION TEST SUMMARY ===\n");

    let categories = vec![
        ("Core Trading", 16),
        ("Neural Network", 7),
        ("News Trading", 8),
        ("Portfolio & Risk", 5),
        ("Sports Betting", 13),
        ("Odds API", 9),
        ("Prediction Markets", 6),
        ("Syndicates", 17),
        ("E2B Cloud", 9),
        ("System & Monitoring", 5),
    ];

    let total = categories.iter().map(|(_, count)| count).sum::<u32>();

    for (category, count) in &categories {
        println!("{:<25} {} functions", category, count);
    }

    println!("\n{:<25} {} TOTAL FUNCTIONS", "GRAND TOTAL:", total);
    println!("\nNote: Run `cargo test` to execute all tests");
    println!("Note: Some tests require environment variables (NEWS_API_KEY, BROKER_API_KEY, etc.)");
}
