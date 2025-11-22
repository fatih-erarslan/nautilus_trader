//! Tests for MCP Parasitic Pairlist Tools
//! Following TDD methodology - tests written FIRST
//! Zero mocks allowed - all implementations must be real

use parasitic::mcp::tools::ParasiticPairlistTools;
use parasitic::mcp::handlers::*;
use parasitic::traits::{MarketData, PairData};
use parasitic::{Result, Error};
use serde_json::{json, Value};
use std::sync::Arc;
use chrono::Utc;

#[tokio::test]
async fn test_scan_parasitic_opportunities() {
    let tools = create_test_tools().await;
    
    let input = json!({
        "min_volume": 1000.0,
        "organisms": ["komodo", "electric_eel", "octopus"],
        "risk_limit": 0.1
    });
    
    let handler = ParasiticScanHandler::new(tools.get_manager());
    let result = handler.handle(input).await;
    
    assert!(result.is_ok(), "Scan should succeed: {:?}", result.err());
    
    let response = result.unwrap();
    assert!(response.is_object());
    assert!(response.get("opportunities").is_some());
    assert!(response.get("scan_timestamp").is_some());
    assert!(response.get("total_pairs_scanned").is_some());
    
    // Performance requirement: sub-millisecond
    let execution_time = response.get("execution_time_ns").unwrap().as_u64().unwrap();
    assert!(execution_time < 1_000_000, "Must be sub-millisecond: {}ns", execution_time);
}

#[tokio::test]
async fn test_detect_whale_nests() {
    let tools = create_test_tools().await;
    
    let input = json!({
        "min_whale_size": 100000.0,
        "vulnerability_threshold": 0.7
    });
    
    let handler = WhaleNestDetectorHandler::new(tools.get_manager());
    let result = handler.handle(input).await;
    
    assert!(result.is_ok(), "Whale nest detection should succeed: {:?}", result.err());
    
    let response = result.unwrap();
    assert!(response.get("whale_nests").is_some());
    assert!(response.get("cuckoo_opportunities").is_some());
    assert!(response.get("vulnerability_scores").is_some());
}

#[tokio::test]
async fn test_identify_zombie_pairs() {
    let tools = create_test_tools().await;
    
    let input = json!({
        "min_predictability": 0.8,
        "pattern_depth": 5
    });
    
    let handler = ZombiePairHandler::new(tools.get_manager());
    let result = handler.handle(input).await;
    
    assert!(result.is_ok(), "Zombie pair identification should succeed: {:?}", result.err());
    
    let response = result.unwrap();
    assert!(response.get("zombie_pairs").is_some());
    assert!(response.get("algorithmic_patterns").is_some());
    assert!(response.get("cordyceps_exploitation_score").is_some());
}

#[tokio::test]
async fn test_analyze_mycelial_network() {
    let tools = create_test_tools().await;
    
    let input = json!({
        "correlation_threshold": 0.75,
        "network_depth": 3
    });
    
    let handler = MycelialNetworkHandler::new(tools.get_manager());
    let result = handler.handle(input).await;
    
    assert!(result.is_ok(), "Mycelial network analysis should succeed: {:?}", result.err());
    
    let response = result.unwrap();
    assert!(response.get("correlation_matrix").is_some());
    assert!(response.get("network_clusters").is_some());
    assert!(response.get("spore_propagation_paths").is_some());
}

#[tokio::test]
async fn test_activate_octopus_camouflage() {
    let tools = create_test_tools().await;
    
    let input = json!({
        "threat_level": "high",
        "camouflage_pattern": "mimetic"
    });
    
    let handler = CamouflageHandler::new(tools.get_manager());
    let result = handler.handle(input).await;
    
    assert!(result.is_ok(), "Octopus camouflage should succeed: {:?}", result.err());
    
    let response = result.unwrap();
    assert!(response.get("camouflage_active").is_some());
    assert!(response.get("threat_assessment").is_some());
    assert!(response.get("adaptation_strategy").is_some());
}

#[tokio::test]
async fn test_deploy_anglerfish_lure() {
    let tools = create_test_tools().await;
    
    let input = json!({
        "lure_pairs": ["BTC_USD", "ETH_USD"],
        "intensity": 0.8
    });
    
    let handler = AnglerfishLureHandler::new(tools.get_manager());
    let result = handler.handle(input).await;
    
    assert!(result.is_ok(), "Anglerfish lure deployment should succeed: {:?}", result.err());
    
    let response = result.unwrap();
    assert!(response.get("lure_deployed").is_some());
    assert!(response.get("artificial_activity_generated").is_some());
    assert!(response.get("attraction_metrics").is_some());
}

#[tokio::test]
async fn test_track_wounded_pairs() {
    let tools = create_test_tools().await;
    
    let input = json!({
        "volatility_threshold": 0.1,
        "tracking_duration": 300000 // 5 minutes
    });
    
    let handler = KomodoTrackerHandler::new(tools.get_manager());
    let result = handler.handle(input).await;
    
    assert!(result.is_ok(), "Wounded pair tracking should succeed: {:?}", result.err());
    
    let response = result.unwrap();
    assert!(response.get("tracked_pairs").is_some());
    assert!(response.get("persistence_scores").is_some());
    assert!(response.get("exploitation_readiness").is_some());
}

#[tokio::test]
async fn test_enter_cryptobiosis() {
    let tools = create_test_tools().await;
    
    let input = json!({
        "trigger_conditions": {
            "market_volatility_threshold": 0.5,
            "liquidity_drop_threshold": 0.3
        },
        "revival_conditions": {
            "stability_period_ms": 60000,
            "liquidity_recovery_threshold": 0.7
        }
    });
    
    let handler = TardigradeHandler::new(tools.get_manager());
    let result = handler.handle(input).await;
    
    assert!(result.is_ok(), "Cryptobiosis entry should succeed: {:?}", result.err());
    
    let response = result.unwrap();
    assert!(response.get("cryptobiosis_state").is_some());
    assert!(response.get("suspended_processes").is_some());
    assert!(response.get("revival_monitoring").is_some());
}

#[tokio::test]
async fn test_electric_shock() {
    let tools = create_test_tools().await;
    
    let input = json!({
        "shock_pairs": ["BTC_USD", "ETH_USD"],
        "voltage": 0.9
    });
    
    let handler = ElectricEelHandler::new(tools.get_manager());
    let result = handler.handle(input).await;
    
    assert!(result.is_ok(), "Electric shock should succeed: {:?}", result.err());
    
    let response = result.unwrap();
    assert!(response.get("shock_result").is_some());
    assert!(response.get("disruption_magnitude").is_some());
    assert!(response.get("hidden_liquidity_revealed").is_some());
    
    // Electric Eel specific validation
    let bioelectric_charge = response.get("bioelectric_charge_remaining").unwrap().as_f64().unwrap();
    assert!(bioelectric_charge < 1.0, "Charge should be depleted after shock");
}

#[tokio::test]
async fn test_electroreception_scan() {
    let tools = create_test_tools().await;
    
    let input = json!({
        "sensitivity": 0.95,
        "frequency_range": [0.1, 100.0]
    });
    
    let handler = PlatypusHandler::new(tools.get_manager());
    let result = handler.handle(input).await;
    
    assert!(result.is_ok(), "Electroreception scan should succeed: {:?}", result.err());
    
    let response = result.unwrap();
    assert!(response.get("electrical_signals").is_some());
    assert!(response.get("order_flow_patterns").is_some());
    assert!(response.get("bioelectric_anomalies").is_some());
}

#[tokio::test]
async fn test_websocket_subscription_support() {
    let tools = create_test_tools().await;
    
    // Test that all handlers support WebSocket subscriptions
    let subscription_data = json!({
        "subscription_id": "test-sub-001",
        "real_time": true,
        "update_interval_ms": 100
    });
    
    let scan_handler = ParasiticScanHandler::new(tools.get_manager());
    let supports_ws = scan_handler.supports_websocket();
    assert!(supports_ws, "All handlers must support WebSocket subscriptions");
    
    let subscription_result = scan_handler.subscribe(subscription_data).await;
    assert!(subscription_result.is_ok(), "WebSocket subscription should succeed");
}

#[tokio::test]
async fn test_json_schema_validation() {
    let tools = create_test_tools().await;
    
    // Test invalid input schema
    let invalid_input = json!({
        "invalid_field": "should_fail"
    });
    
    let handler = ParasiticScanHandler::new(tools.get_manager());
    let result = handler.validate_input(&invalid_input).await;
    
    assert!(result.is_err(), "Invalid input should be rejected");
    
    // Test valid input schema
    let valid_input = json!({
        "min_volume": 1000.0,
        "organisms": ["komodo"],
        "risk_limit": 0.1
    });
    
    let result = handler.validate_input(&valid_input).await;
    assert!(result.is_ok(), "Valid input should be accepted");
}

#[tokio::test]
async fn test_performance_requirements() {
    let tools = create_test_tools().await;
    
    let input = json!({
        "min_volume": 1000.0,
        "organisms": ["komodo", "electric_eel"],
        "risk_limit": 0.1
    });
    
    let handler = ParasiticScanHandler::new(tools.get_manager());
    
    // Measure performance
    let start = std::time::Instant::now();
    let result = handler.handle(input).await;
    let duration = start.elapsed();
    
    assert!(result.is_ok(), "Handler should succeed");
    assert!(duration.as_nanos() < 1_000_000, "Must be sub-millisecond: {:?}", duration);
}

#[tokio::test]
async fn test_integration_with_existing_organisms() {
    let tools = create_test_tools().await;
    
    // Test that MCP handlers integrate with existing organisms
    let input = json!({
        "sensitivity": 0.8,
        "frequency_range": [1.0, 50.0]
    });
    
    let handler = PlatypusHandler::new(tools.get_manager());
    let result = handler.handle(input).await;
    
    assert!(result.is_ok(), "Should integrate with PlatypusElectroreceptor");
    
    // Verify real organism connection
    let response = result.unwrap();
    let bioelectric_data = response.get("bioelectric_anomalies").unwrap();
    assert!(bioelectric_data.is_array(), "Should return real electroreception data");
}

// Helper function to create test tools
async fn create_test_tools() -> ParasiticPairlistTools {
    let manager = create_test_manager().await;
    ParasiticPairlistTools::new(Arc::new(manager))
}

// Helper function to create test manager
async fn create_test_manager() -> parasitic::mcp::ParasiticPairlistManager {
    let config = parasitic::mcp::ManagerConfig {
        max_pairs: 1000,
        update_interval_ms: 100,
        performance_threshold_ns: 1_000_000,
        websocket_enabled: true,
    };
    
    parasitic::mcp::ParasiticPairlistManager::new(config).await.unwrap()
}

// Helper function to create test market data
fn create_test_market_data() -> MarketData {
    MarketData {
        symbol: "BTC_USD".to_string(),
        timestamp: Utc::now(),
        price: 50000.0,
        volume: 1000.0,
        volatility: 0.15,
        bid: 49950.0,
        ask: 50050.0,
        spread_percent: 0.2,
        market_cap: Some(1000000000.0),
        liquidity_score: 0.8,
    }
}

// Helper function to create test pair data
fn create_test_pair_data() -> PairData {
    use std::collections::HashMap;
    use parasitic::traits::OrderBookDepth;
    
    PairData {
        base_asset: "BTC".to_string(),
        quote_asset: "USD".to_string(),
        exchange: "test_exchange".to_string(),
        market_data: create_test_market_data(),
        technical_indicators: HashMap::new(),
        order_book_depth: OrderBookDepth {
            bids: vec![],
            asks: vec![],
            total_bid_volume: 0.0,
            total_ask_volume: 0.0,
            imbalance_ratio: 0.0,
        },
    }
}