//! Integration tests for QBMIA Core
//! 
//! Comprehensive end-to-end testing of the complete QBMIA system

use qbmia_core::{
    QBMIAAgent, Config,
    agent::{MarketData, AnalysisResult},
    strategy::OrderEvent,
};
use std::collections::HashMap;
use tokio_test;

#[tokio::test]
async fn test_full_qbmia_analysis_pipeline() {
    // Initialize agent with test configuration
    let config = Config {
        agent_id: "QBMIA_TEST_001".to_string(),
        quantum: qbmia_core::config::QuantumConfig {
            num_qubits: 8,
            max_iterations: 20, // Faster for testing
            ..Default::default()
        },
        memory: qbmia_core::config::MemoryConfig {
            capacity: 100,
            short_term_size: 10,
            ..Default::default()
        },
        ..Default::default()
    };
    
    let mut agent = QBMIAAgent::new(config).await.expect("Failed to create agent");
    
    // Create realistic market data
    let market_data = create_test_market_data();
    
    // Perform analysis
    let result = agent.analyze_market(market_data).await.expect("Analysis failed");
    
    // Validate results
    validate_analysis_result(&result);
    
    // Test agent status
    let status = agent.get_status();
    assert_eq!(status.agent_id, "QBMIA_TEST_001");
    assert!(status.performance.total_analyses > 0);
}

#[tokio::test]
async fn test_quantum_nash_integration() {
    let config = Config {
        quantum: qbmia_core::config::QuantumConfig {
            num_qubits: 6,
            max_iterations: 10,
            ..Default::default()
        },
        ..Default::default()
    };
    
    let mut agent = QBMIAAgent::new(config).await.unwrap();
    let market_data = create_simple_market_data();
    
    let result = agent.analyze_market(market_data).await.unwrap();
    
    // Should have quantum Nash results
    assert!(result.component_results.quantum_nash.is_some());
    let quantum_result = result.component_results.quantum_nash.as_ref().unwrap();
    
    // Validate quantum Nash result structure
    assert!(quantum_result.convergence_score >= 0.0);
    assert!(quantum_result.convergence_score <= 1.0);
    assert!(quantum_result.nash_loss >= 0.0);
    assert!(quantum_result.iterations > 0);
    assert!(!quantum_result.strategies.is_empty());
    
    // Check strategy validity
    for strategy in quantum_result.strategies.values() {
        let sum: f64 = strategy.sum();
        assert!((sum - 1.0).abs() < 0.1, "Strategy sum: {}", sum); // Allow some tolerance
        assert!(strategy.iter().all(|&p| p >= -0.01 && p <= 1.01)); // Small tolerance for numerical errors
    }
}

#[tokio::test]
async fn test_machiavellian_integration() {
    let config = Config::default();
    let mut agent = QBMIAAgent::new(config).await.unwrap();
    
    // Create market data with suspicious order patterns
    let market_data = create_manipulated_market_data();
    
    let result = agent.analyze_market(market_data).await.unwrap();
    
    // Should have Machiavellian results
    assert!(result.component_results.machiavellian.is_some());
    let machiavellian_result = result.component_results.machiavellian.as_ref().unwrap();
    
    // Validate manipulation detection
    assert!(machiavellian_result.confidence >= 0.0);
    assert!(machiavellian_result.confidence <= 1.0);
    assert!(!machiavellian_result.manipulation_scores.is_empty());
    assert!(!machiavellian_result.primary_pattern.is_empty());
    assert!(!machiavellian_result.recommended_action.is_empty());
}

#[tokio::test]
async fn test_memory_integration() {
    let config = Config {
        memory: qbmia_core::config::MemoryConfig {
            capacity: 50,
            short_term_size: 5,
            ..Default::default()
        },
        ..Default::default()
    };
    
    let mut agent = QBMIAAgent::new(config).await.unwrap();
    
    // Perform multiple analyses to test memory storage
    for i in 0..10 {
        let mut market_data = create_test_market_data();
        // Vary the market data slightly
        if let Some(price) = market_data.snapshot.get_mut("price") {
            *price = serde_json::json!(50000.0 + i as f64 * 100.0);
        }
        
        let result = agent.analyze_market(market_data).await.unwrap();
        assert!(!result.agent_id.is_empty());
    }
    
    // Check memory usage
    let status = agent.get_status();
    assert!(status.memory_usage.total_memory_mb > 0.0);
    assert!(status.performance.total_analyses == 10);
}

#[tokio::test]
async fn test_performance_requirements() {
    let config = Config {
        quantum: qbmia_core::config::QuantumConfig {
            num_qubits: 8,
            max_iterations: 20,
            ..Default::default()
        },
        performance: qbmia_core::config::PerformanceConfig {
            target_latency_ms: 10.0, // Sub-10ms target
            ..Default::default()
        },
        ..Default::default()
    };
    
    let mut agent = QBMIAAgent::new(config).await.unwrap();
    let market_data = create_test_market_data();
    
    let start = std::time::Instant::now();
    let result = agent.analyze_market(market_data).await.unwrap();
    let elapsed = start.elapsed();
    
    // Performance validation
    assert!(elapsed.as_millis() < 50, "Execution took {} ms, expected < 50ms", elapsed.as_millis());
    assert!(result.execution_time < 50.0, "Reported execution time: {} ms", result.execution_time);
    
    // Quality validation
    assert!(result.confidence >= 0.0);
    assert!(result.confidence <= 1.0);
}

#[tokio::test]
async fn test_concurrent_analyses() {
    let config = Config {
        quantum: qbmia_core::config::QuantumConfig {
            num_qubits: 6,
            max_iterations: 10,
            ..Default::default()
        },
        ..Default::default()
    };
    
    // Create multiple agents
    let mut agents = Vec::new();
    for i in 0..3 {
        let mut agent_config = config.clone();
        agent_config.agent_id = format!("QBMIA_CONCURRENT_{}", i);
        agents.push(QBMIAAgent::new(agent_config).await.unwrap());
    }
    
    // Run concurrent analyses
    let tasks: Vec<_> = agents.into_iter().enumerate().map(|(i, mut agent)| {
        let market_data = create_test_market_data();
        tokio::spawn(async move {
            let result = agent.analyze_market(market_data).await.unwrap();
            (i, result)
        })
    }).collect();
    
    let results = futures::future::try_join_all(tasks).await.unwrap();
    
    // All analyses should succeed
    assert_eq!(results.len(), 3);
    for (i, result) in results {
        assert_eq!(result.agent_id, format!("QBMIA_CONCURRENT_{}", i));
        assert!(result.execution_time > 0.0);
    }
}

// Helper functions for creating test data

fn create_test_market_data() -> MarketData {
    let mut snapshot = HashMap::new();
    snapshot.insert("price".to_string(), serde_json::json!(50000.0));
    snapshot.insert("volume".to_string(), serde_json::json!(1000000.0));
    snapshot.insert("volatility".to_string(), serde_json::json!(0.02));
    snapshot.insert("trend".to_string(), serde_json::json!(0.1));
    
    let order_flow = vec![
        OrderEvent {
            timestamp: 1.0,
            side: "buy".to_string(),
            size: 100.0,
            price: 49990.0,
            cancelled: false,
        },
        OrderEvent {
            timestamp: 2.0,
            side: "sell".to_string(),
            size: 150.0,
            price: 50010.0,
            cancelled: false,
        },
        OrderEvent {
            timestamp: 3.0,
            side: "buy".to_string(),
            size: 200.0,
            price: 50000.0,
            cancelled: true,
        },
    ];
    
    let price_history = vec![49000.0, 49500.0, 50000.0, 50200.0, 50100.0];
    
    let mut conditions = HashMap::new();
    conditions.insert("volatility".to_string(), 0.02);
    conditions.insert("trend".to_string(), 0.1);
    conditions.insert("volume".to_string(), 1000000.0);
    
    let participants = vec!["trader1".to_string(), "trader2".to_string(), "market_maker".to_string()];
    
    let mut competitors = HashMap::new();
    competitors.insert("hft_firm_1".to_string(), 0.3);
    competitors.insert("hedge_fund_1".to_string(), 0.5);
    
    MarketData {
        snapshot,
        order_flow,
        price_history,
        time_series: HashMap::new(),
        conditions,
        participants,
        competitors,
    }
}

fn create_simple_market_data() -> MarketData {
    let mut data = create_test_market_data();
    data.order_flow.clear(); // Simplify for quantum tests
    data.price_history = vec![50000.0, 50000.0, 50000.0]; // Stable prices
    data
}

fn create_manipulated_market_data() -> MarketData {
    let mut data = create_test_market_data();
    
    // Add suspicious order patterns
    data.order_flow = vec![
        // Large order that gets cancelled (spoofing)
        OrderEvent {
            timestamp: 1.0,
            side: "buy".to_string(),
            size: 10000.0, // Very large
            price: 50000.0,
            cancelled: true, // Cancelled
        },
        // Multiple similar orders at different prices (layering)
        OrderEvent {
            timestamp: 2.0,
            side: "buy".to_string(),
            size: 100.0,
            price: 49990.0,
            cancelled: false,
        },
        OrderEvent {
            timestamp: 2.1,
            side: "buy".to_string(),
            size: 100.0,
            price: 49985.0,
            cancelled: false,
        },
        OrderEvent {
            timestamp: 2.2,
            side: "buy".to_string(),
            size: 100.0,
            price: 49980.0,
            cancelled: false,
        },
    ];
    
    data
}

fn validate_analysis_result(result: &AnalysisResult) {
    // Basic structure validation
    assert!(!result.timestamp.is_empty());
    assert!(!result.agent_id.is_empty());
    assert!(result.execution_time > 0.0);
    assert!(result.confidence >= 0.0);
    assert!(result.confidence <= 1.0);
    
    // Component results validation
    let components = &result.component_results;
    
    // At least one component should have results
    let has_results = components.quantum_nash.is_some() || 
                     components.machiavellian.is_some() || 
                     components.strategy.is_some();
    assert!(has_results, "No component results found");
    
    // If we have an integrated decision, validate it
    if let Some(ref decision) = result.integrated_decision {
        assert!(!decision.action.is_empty());
        assert!(decision.confidence >= 0.0);
        assert!(decision.confidence <= 1.0);
        assert!(!decision.reasoning.is_empty());
        assert_eq!(decision.decision_vector.len(), 4); // [buy, sell, hold, wait]
        
        // Decision vector should be normalized
        let sum: f64 = decision.decision_vector.iter().sum();
        if sum > 0.0 {
            assert!((sum - 1.0).abs() < 0.01, "Decision vector not normalized: sum = {}", sum);
        }
    }
}

#[test]
fn test_config_validation() {
    // Test valid config
    let valid_config = Config::default();
    assert!(valid_config.validate().is_ok());
    
    // Test invalid configs
    let mut invalid_config = Config::default();
    invalid_config.quantum.num_qubits = 0;
    assert!(invalid_config.validate().is_err());
    
    invalid_config = Config::default();
    invalid_config.quantum.learning_rate = 2.0;
    assert!(invalid_config.validate().is_err());
    
    invalid_config = Config::default();
    invalid_config.memory.capacity = 0;
    assert!(invalid_config.validate().is_err());
}

#[test]
fn test_error_handling() {
    use qbmia_core::error::QBMIAError;
    
    // Test error creation
    let error = QBMIAError::quantum_simulation("test error");
    assert!(error.to_string().contains("test error"));
    
    let error = QBMIAError::memory("memory error");
    assert!(error.to_string().contains("memory error"));
    
    let error = QBMIAError::strategy("strategy error");
    assert!(error.to_string().contains("strategy error"));
}