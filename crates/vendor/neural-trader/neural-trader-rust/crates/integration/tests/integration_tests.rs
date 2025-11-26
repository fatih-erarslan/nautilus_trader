//! Integration tests for the neural-trader system.

use neural_trader_integration::{Config, NeuralTrader, NeuralTraderBuilder};

#[tokio::test]
async fn test_config_loading() {
    let _config = Config::default();
    assert!(config.validate().is_err()); // Should fail without valid broker credentials
}

#[tokio::test]
async fn test_builder_pattern() {
    let _config = Config::default();
    let builder = NeuralTraderBuilder::new().with_config(config);

    // This will fail without proper configuration but tests the API
    assert!(builder.build().await.is_err());
}

#[tokio::test]
#[ignore = "Requires broker credentials"]
async fn test_full_initialization() {
    let _config = Config::default();
    let result = NeuralTrader::new(config).await;

    // This should fail without proper configuration
    assert!(result.is_err());
}

#[tokio::test]
async fn test_health_check_structure() {
    // Test that health check types work correctly
    use neural_trader_integration::types::{HealthStatus, ComponentHealth, HealthStatusEnum};
    use chrono::Utc;

    let health = ComponentHealth {
        status: HealthStatusEnum::Healthy,
        message: Some("Test".to_string()),
        last_check: Utc::now(),
        uptime: std::time::Duration::from_secs(100),
    };

    assert!(health.is_healthy());
}

#[tokio::test]
async fn test_portfolio_structure() {
    use neural_trader_integration::types::Portfolio;
    use chrono::Utc;

    let portfolio = Portfolio {
        total_value: rust_decimal::Decimal::from(100000),
        cash: rust_decimal::Decimal::from(50000),
        positions: vec![],
        updated_at: Utc::now(),
    };

    assert_eq!(portfolio.total_value, rust_decimal::Decimal::from(100000));
}

#[tokio::test]
async fn test_execution_result_structure() {
    use neural_trader_integration::types::ExecutionResult;
    use chrono::Utc;

    let result = ExecutionResult {
        strategy_name: "test".to_string(),
        timestamp: Utc::now(),
        orders: vec![],
        total_value: rust_decimal::Decimal::ZERO,
        profit_loss: rust_decimal::Decimal::ZERO,
        metadata: serde_json::json!({}),
    };

    assert_eq!(result.strategy_name, "test");
}
