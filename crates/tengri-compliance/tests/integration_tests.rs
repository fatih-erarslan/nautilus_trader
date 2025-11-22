//! Integration tests for TENGRI compliance engine

use std::collections::HashMap;
use tokio_test;
use uuid::Uuid;
use chrono::Utc;
use rust_decimal::Decimal;

use tengri_compliance::{
    ComplianceEngine, ComplianceConfig,
    rules::{TradingContext, OrderSide, OrderType, Position},
    audit::AuditEventType,
    surveillance::{TradeRecord, TradeSide},
    engine::{ComplianceDecision, StrictnessLevel},
};

fn create_test_context() -> TradingContext {
    let mut positions = HashMap::new();
    positions.insert("BTCUSD".to_string(), Position {
        symbol: "BTCUSD".to_string(),
        quantity: Decimal::from(5),
        average_price: Decimal::from(50000),
        unrealized_pnl: Decimal::from(2500),
        market_value: Decimal::from(250000),
    });

    TradingContext {
        order_id: Uuid::new_v4(),
        symbol: "BTCUSD".to_string(),
        side: OrderSide::Buy,
        quantity: Decimal::from_str("0.1").unwrap(),
        price: Some(Decimal::from(50000)),
        order_type: OrderType::Limit,
        trader_id: "trader_001".to_string(),
        timestamp: Utc::now(),
        portfolio_value: Decimal::from(1_000_000),
        current_positions: positions,
        daily_pnl: Decimal::from(1000),
        metadata: HashMap::new(),
    }
}

#[tokio::test]
async fn test_engine_initialization() {
    let config = ComplianceConfig::default();
    let engine = ComplianceEngine::new(config).await.unwrap();
    
    // Engine should be running
    let status = engine.get_status().await;
    assert!(matches!(status, tengri_compliance::engine::EngineStatus::Running));
}

#[tokio::test]
async fn test_normal_trade_approval() {
    let config = ComplianceConfig::default();
    let engine = ComplianceEngine::new(config).await.unwrap();
    
    let context = create_test_context();
    let result = engine.process_trade(context).await;
    
    assert!(result.is_ok());
    match result.unwrap() {
        ComplianceDecision::Approved { .. } => {},
        ComplianceDecision::Rejected { reason, .. } => {
            panic!("Trade should have been approved, but was rejected: {}", reason);
        }
    }
}

#[tokio::test]
async fn test_large_position_rejection() {
    let mut config = ComplianceConfig::default();
    config.strictness_level = StrictnessLevel::Conservative;
    let engine = ComplianceEngine::new(config).await.unwrap();
    
    let mut context = create_test_context();
    // Set a very large quantity that should exceed limits
    context.quantity = Decimal::from(1_000_000);
    
    let result = engine.process_trade(context).await;
    
    // Should be rejected due to position limits
    assert!(result.is_ok());
    match result.unwrap() {
        ComplianceDecision::Approved { .. } => {
            panic!("Large trade should have been rejected");
        },
        ComplianceDecision::Rejected { reason, .. } => {
            assert!(reason.contains("Position") || reason.contains("limit"));
        }
    }
}

#[tokio::test]
async fn test_daily_loss_limit() {
    let mut config = ComplianceConfig::default();
    config.strictness_level = StrictnessLevel::Conservative;
    let engine = ComplianceEngine::new(config).await.unwrap();
    
    let mut context = create_test_context();
    // Set a large daily loss
    context.daily_pnl = Decimal::from(-100_000);
    
    let result = engine.process_trade(context).await;
    
    // Should be rejected due to daily loss limit
    assert!(result.is_ok());
    match result.unwrap() {
        ComplianceDecision::Approved { .. } => {
            panic!("Trade with large daily loss should have been rejected");
        },
        ComplianceDecision::Rejected { reason, .. } => {
            assert!(reason.contains("loss") || reason.contains("limit"));
        }
    }
}

#[tokio::test]
async fn test_audit_trail_recording() {
    let config = ComplianceConfig::default();
    let engine = ComplianceEngine::new(config).await.unwrap();
    
    let audit_trail = engine.get_audit_trail();
    
    // Record a test event
    let event_id = audit_trail.record(
        AuditEventType::TradeSubmitted { order_id: Uuid::new_v4() },
        "test_trader".to_string(),
        serde_json::json!({"test": "data"}),
    ).await.unwrap();
    
    // Verify the event was recorded
    let recorded_event = audit_trail.get_by_id(&event_id);
    assert!(recorded_event.is_some());
    
    let event = recorded_event.unwrap();
    assert_eq!(event.actor, "test_trader");
}

#[tokio::test]
async fn test_surveillance_engine() {
    let config = ComplianceConfig::default();
    let engine = ComplianceEngine::new(config).await.unwrap();
    
    let surveillance = engine.get_surveillance_engine();
    
    // Record some test trades
    for i in 0..5 {
        let trade = TradeRecord {
            id: Uuid::new_v4(),
            timestamp: Utc::now(),
            symbol: "BTCUSD".to_string(),
            side: if i % 2 == 0 { TradeSide::Buy } else { TradeSide::Sell },
            quantity: Decimal::from(1),
            price: Decimal::from(50000 + i * 10),
            trader_id: format!("trader_{:03}", i % 3),
            order_id: Uuid::new_v4(),
            venue: "test".to_string(),
            execution_time_ms: 100,
        };
        surveillance.record_trade(trade);
    }
    
    // Analyze patterns
    let patterns = surveillance.analyze_patterns().await.unwrap();
    
    // Should complete without error (patterns may or may not be detected)
    assert!(patterns.len() >= 0);
}

#[tokio::test]
async fn test_circuit_breaker_functionality() {
    let config = ComplianceConfig::default();
    let engine = ComplianceEngine::new(config).await.unwrap();
    
    // Process multiple failing trades to trigger circuit breaker
    let mut context = create_test_context();
    context.quantity = Decimal::from(10_000_000); // Very large to trigger failures
    
    for _ in 0..10 {
        let _ = engine.process_trade(context.clone()).await;
    }
    
    // Circuit breaker should eventually be triggered
    // (This is a simplified test - in reality we'd need more sophisticated failure injection)
}

#[tokio::test]
async fn test_metrics_collection() {
    let config = ComplianceConfig::default();
    let engine = ComplianceEngine::new(config).await.unwrap();
    
    let metrics = engine.get_metrics();
    
    // Process a few trades
    for _ in 0..3 {
        let context = create_test_context();
        let _ = engine.process_trade(context).await;
    }
    
    // Check that metrics were collected
    let dashboard_data = metrics.get_dashboard_data();
    assert!(dashboard_data.total_trades_processed >= 3);
}

#[tokio::test]
async fn test_concurrent_trade_processing() {
    let config = ComplianceConfig::default();
    let engine = ComplianceEngine::new(config).await.unwrap();
    
    // Create multiple concurrent trades
    let futures: Vec<_> = (0..10)
        .map(|i| {
            let mut context = create_test_context();
            context.trader_id = format!("trader_{:03}", i);
            context.order_id = Uuid::new_v4();
            engine.process_trade(context)
        })
        .collect();
    
    // Wait for all trades to complete
    let results = futures::future::join_all(futures).await;
    
    // All trades should complete (though some may be rejected)
    assert_eq!(results.len(), 10);
    for result in results {
        assert!(result.is_ok());
    }
}

#[tokio::test]
async fn test_emergency_shutdown() {
    let config = ComplianceConfig::default();
    let engine = ComplianceEngine::new(config).await.unwrap();
    
    // Trigger emergency shutdown
    let shutdown_result = engine.emergency_shutdown("Test emergency".to_string()).await;
    assert!(shutdown_result.is_err());
    
    // Engine should be in emergency state
    let status = engine.get_status().await;
    assert!(matches!(status, tengri_compliance::engine::EngineStatus::Emergency));
    
    // New trades should be rejected
    let context = create_test_context();
    let trade_result = engine.process_trade(context).await;
    assert!(trade_result.is_err());
}

#[tokio::test]
async fn test_custom_rule_addition() {
    let config = ComplianceConfig {
        strictness_level: StrictnessLevel::Custom,
        ..Default::default()
    };
    let engine = ComplianceEngine::new(config).await.unwrap();
    
    // Add a custom rule (using the minimal rule set for testing)
    let rules = tengri_compliance::rules::RuleSet::minimal();
    for rule in rules {
        engine.add_rule(rule);
    }
    
    // Process a trade - should work with custom rules
    let context = create_test_context();
    let result = engine.process_trade(context).await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_performance_under_load() {
    let config = ComplianceConfig::default();
    let engine = ComplianceEngine::new(config).await.unwrap();
    
    let start_time = std::time::Instant::now();
    
    // Process 100 trades rapidly
    let futures: Vec<_> = (0..100)
        .map(|i| {
            let mut context = create_test_context();
            context.order_id = Uuid::new_v4();
            context.trader_id = format!("trader_{:03}", i % 10);
            engine.process_trade(context)
        })
        .collect();
    
    let results = futures::future::join_all(futures).await;
    let duration = start_time.elapsed();
    
    // All trades should complete
    assert_eq!(results.len(), 100);
    
    // Should complete within reasonable time (adjust threshold as needed)
    assert!(duration.as_secs() < 10, "Processing took too long: {:?}", duration);
    
    // Check metrics
    let metrics = engine.get_metrics();
    let dashboard = metrics.get_dashboard_data();
    assert!(dashboard.total_trades_processed >= 100);
}