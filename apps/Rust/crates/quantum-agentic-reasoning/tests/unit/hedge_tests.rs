//! Comprehensive unit tests for Quantum Hedge algorithm integration
//! 
//! Tests all aspects of multiplicative weights hedge algorithms:
//! - Expert strategy management and performance tracking
//! - Quantum-enhanced weight updates and portfolio optimization
//! - Real-time performance optimization
//! - Multi-strategy coordination and consensus mechanisms
//! - Performance constraints and sub-microsecond execution

use quantum_agentic_reasoning::{hedge_integration::*, MarketData, Result};
use std::collections::HashMap;

#[tokio::test]
async fn test_hedge_engine_creation() {
    let config = HedgeConfig::default();
    let engine = QuantumHedgeEngine::new(config);
    
    assert!(engine.is_ok());
    let engine = engine.unwrap();
    
    // Should start with configured number of experts
    assert_eq!(engine.get_expert_count(), 8); // Default config
    assert_eq!(engine.get_active_strategies().len(), 8);
}

#[tokio::test]
async fn test_basic_portfolio_optimization() {
    let config = HedgeConfig::default();
    let mut engine = QuantumHedgeEngine::new(config).unwrap();
    
    let market_data = MarketData {
        symbol: "BTC/USDT".to_string(),
        current_price: 50000.0,
        possible_outcomes: vec![52000.0, 51000.0, 49000.0, 48000.0],
        buy_probabilities: vec![0.3, 0.3, 0.2, 0.2],
        sell_probabilities: vec![0.2, 0.2, 0.3, 0.3],
        hold_probabilities: vec![0.25, 0.25, 0.25, 0.25],
        frame: prospect_theory::FramingContext {
            frame_type: prospect_theory::FrameType::Neutral,
            emphasis: 0.5,
        },
        timestamp: 1640995200000,
    };
    
    let optimization = engine.optimize_portfolio(&market_data, None).await;
    assert!(optimization.is_ok());
    
    let result = optimization.unwrap();
    assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
    assert!(result.portfolio_weights.len() > 0);
    
    // Portfolio weights should sum to approximately 1.0
    let weight_sum: f64 = result.portfolio_weights.iter().map(|w| w.weight).sum();
    assert!((weight_sum - 1.0).abs() < 0.01);
    
    // All weights should be non-negative
    for weight in &result.portfolio_weights {
        assert!(weight.weight >= 0.0);
    }
}

#[tokio::test]
async fn test_expert_weight_updates() {
    let mut config = HedgeConfig::default();
    config.learning_rate = 0.1;
    
    let mut engine = QuantumHedgeEngine::new(config).unwrap();
    
    let market_data = MarketData {
        symbol: "BTC/USDT".to_string(),
        current_price: 50000.0,
        possible_outcomes: vec![52000.0, 48000.0],
        buy_probabilities: vec![0.6, 0.4],
        sell_probabilities: vec![0.4, 0.6],
        hold_probabilities: vec![0.5, 0.5],
        frame: prospect_theory::FramingContext {
            frame_type: prospect_theory::FrameType::Gain,
            emphasis: 0.7,
        },
        timestamp: 1640995200000,
    };
    
    // Get initial optimization
    let initial_result = engine.optimize_portfolio(&market_data, None).await.unwrap();
    let initial_weights = initial_result.portfolio_weights.clone();
    
    // Simulate performance feedback for experts
    let expert_performances = vec![
        ExpertPerformance {
            expert_id: "momentum_expert".to_string(),
            strategy_type: StrategyType::Momentum,
            actual_return: 0.05, // 5% gain
            predicted_return: 0.04,
            accuracy: 0.9,
            timestamp: market_data.timestamp + 3600000,
        },
        ExpertPerformance {
            expert_id: "mean_reversion_expert".to_string(),
            strategy_type: StrategyType::MeanReversion,
            actual_return: -0.02, // 2% loss
            predicted_return: 0.03,
            accuracy: 0.3,
            timestamp: market_data.timestamp + 3600000,
        },
    ];
    
    // Update weights based on performance
    for performance in expert_performances {
        engine.update_expert_weights(&performance).await.unwrap();
    }
    
    // Get updated optimization
    let mut updated_data = market_data.clone();
    updated_data.timestamp += 7200000; // 2 hours later
    let updated_result = engine.optimize_portfolio(&updated_data, None).await.unwrap();
    
    // Weights should have changed based on performance
    assert_ne!(initial_weights, updated_result.portfolio_weights);
    
    // Momentum expert should have higher weight after good performance
    let momentum_weight_initial = initial_weights.iter()
        .find(|w| w.expert_strategy == StrategyType::Momentum)
        .map(|w| w.weight)
        .unwrap_or(0.0);
    
    let momentum_weight_updated = updated_result.portfolio_weights.iter()
        .find(|w| w.expert_strategy == StrategyType::Momentum)
        .map(|w| w.weight)
        .unwrap_or(0.0);
    
    assert!(momentum_weight_updated >= momentum_weight_initial);
}

#[tokio::test]
async fn test_quantum_enhancement() {
    // Test with quantum enhancement disabled
    let mut config_classical = HedgeConfig::default();
    config_classical.quantum_enhancement = false;
    let mut engine_classical = QuantumHedgeEngine::new(config_classical).unwrap();
    
    // Test with quantum enhancement enabled
    let mut config_quantum = HedgeConfig::default();
    config_quantum.quantum_enhancement = true;
    config_quantum.amplitude_amplification_iterations = 3;
    let mut engine_quantum = QuantumHedgeEngine::new(config_quantum).unwrap();
    
    let market_data = MarketData {
        symbol: "BTC/USDT".to_string(),
        current_price: 50000.0,
        possible_outcomes: vec![55000.0, 45000.0],
        buy_probabilities: vec![0.8, 0.2], // Strong signal
        sell_probabilities: vec![0.2, 0.8],
        hold_probabilities: vec![0.3, 0.7],
        frame: prospect_theory::FramingContext {
            frame_type: prospect_theory::FrameType::Gain,
            emphasis: 0.9,
        },
        timestamp: 1640995200000,
    };
    
    let classical_result = engine_classical.optimize_portfolio(&market_data, None).await.unwrap();
    let quantum_result = engine_quantum.optimize_portfolio(&market_data, None).await.unwrap();
    
    // Quantum enhancement should improve confidence for strong signals
    if quantum_result.confidence > 0.7 {
        assert!(quantum_result.confidence >= classical_result.confidence);
    }
    
    // Both should produce valid results
    assert!(classical_result.confidence >= 0.0 && classical_result.confidence <= 1.0);
    assert!(quantum_result.confidence >= 0.0 && quantum_result.confidence <= 1.0);
    
    // Portfolio weights should sum to ~1.0 for both
    let classical_sum: f64 = classical_result.portfolio_weights.iter().map(|w| w.weight).sum();
    let quantum_sum: f64 = quantum_result.portfolio_weights.iter().map(|w| w.weight).sum();
    assert!((classical_sum - 1.0).abs() < 0.01);
    assert!((quantum_sum - 1.0).abs() < 0.01);
}

#[tokio::test]
async fn test_multi_strategy_coordination() {
    let mut config = HedgeConfig::default();
    config.num_experts = 6; // Test with different strategies
    
    let mut engine = QuantumHedgeEngine::new(config).unwrap();
    
    // Market data with mixed signals
    let market_data = MarketData {
        symbol: "BTC/USDT".to_string(),
        current_price: 50000.0,
        possible_outcomes: vec![53000.0, 52000.0, 49000.0, 47000.0],
        buy_probabilities: vec![0.4, 0.3, 0.2, 0.1], // Moderate bullish
        sell_probabilities: vec![0.1, 0.2, 0.3, 0.4], // Moderate bearish
        hold_probabilities: vec![0.25, 0.25, 0.25, 0.25],
        frame: prospect_theory::FramingContext {
            frame_type: prospect_theory::FrameType::Neutral,
            emphasis: 0.5,
        },
        timestamp: 1640995200000,
    };
    
    let result = engine.optimize_portfolio(&market_data, None).await.unwrap();
    
    // Should have weights for multiple strategies
    assert!(result.portfolio_weights.len() >= 4);
    
    // Check that different strategy types are represented
    let strategy_types: std::collections::HashSet<_> = result.portfolio_weights
        .iter()
        .map(|w| &w.expert_strategy)
        .collect();
    
    assert!(strategy_types.len() >= 3); // At least 3 different strategy types
    
    // Verify strategy diversity
    assert!(strategy_types.contains(&StrategyType::Momentum) || 
            strategy_types.contains(&StrategyType::MeanReversion) ||
            strategy_types.contains(&StrategyType::Arbitrage));
}

#[tokio::test]
async fn test_real_time_optimization() {
    let mut config = HedgeConfig::default();
    config.real_time_optimization = true;
    config.optimization_frequency_ms = 100; // 100ms updates
    
    let mut engine = QuantumHedgeEngine::new(config).unwrap();
    
    let base_data = MarketData {
        symbol: "BTC/USDT".to_string(),
        current_price: 50000.0,
        possible_outcomes: vec![52000.0, 48000.0],
        buy_probabilities: vec![0.6, 0.4],
        sell_probabilities: vec![0.4, 0.6],
        hold_probabilities: vec![0.5, 0.5],
        frame: prospect_theory::FramingContext {
            frame_type: prospect_theory::FrameType::Neutral,
            emphasis: 0.5,
        },
        timestamp: 1640995200000,
    };
    
    // Start real-time optimization
    engine.start_real_time_optimization().await.unwrap();
    
    // Simulate rapid market updates
    for i in 0..5 {
        let mut data = base_data.clone();
        data.current_price = 50000.0 + (i as f64 * 100.0);
        data.timestamp += (i as u64 * 100);
        
        let result = engine.optimize_portfolio(&data, None).await.unwrap();
        
        // Each result should be valid
        assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
        assert!(!result.portfolio_weights.is_empty());
        
        // Should adapt to changing conditions
        let weight_sum: f64 = result.portfolio_weights.iter().map(|w| w.weight).sum();
        assert!((weight_sum - 1.0).abs() < 0.01);
    }
    
    engine.stop_real_time_optimization().await.unwrap();
}

#[tokio::test]
async fn test_performance_constraints() {
    let mut config = HedgeConfig::default();
    config.target_latency_ns = 500; // 500ns target
    
    let mut engine = QuantumHedgeEngine::new(config).unwrap();
    
    let market_data = MarketData {
        symbol: "BTC/USDT".to_string(),
        current_price: 50000.0,
        possible_outcomes: vec![52000.0, 48000.0],
        buy_probabilities: vec![0.6, 0.4],
        sell_probabilities: vec![0.4, 0.6],
        hold_probabilities: vec![0.5, 0.5],
        frame: prospect_theory::FramingContext {
            frame_type: prospect_theory::FrameType::Neutral,
            emphasis: 0.5,
        },
        timestamp: 1640995200000,
    };
    
    let start = std::time::Instant::now();
    let result = engine.optimize_portfolio(&market_data, None).await.unwrap();
    let elapsed = start.elapsed();
    
    // Should meet performance target (allowing overhead for test environment)
    assert!(elapsed.as_nanos() < 5000); // 5μs max in test
    
    // Should still produce valid optimization
    assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
    assert!(!result.portfolio_weights.is_empty());
    
    let weight_sum: f64 = result.portfolio_weights.iter().map(|w| w.weight).sum();
    assert!((weight_sum - 1.0).abs() < 0.01);
}

#[tokio::test]
async fn test_expert_performance_tracking() {
    let config = HedgeConfig::default();
    let mut engine = QuantumHedgeEngine::new(config).unwrap();
    
    // Submit performance data for different experts
    let performances = vec![
        ExpertPerformance {
            expert_id: "momentum_expert".to_string(),
            strategy_type: StrategyType::Momentum,
            actual_return: 0.08,
            predicted_return: 0.07,
            accuracy: 0.95,
            timestamp: 1640995200000,
        },
        ExpertPerformance {
            expert_id: "arbitrage_expert".to_string(),
            strategy_type: StrategyType::Arbitrage,
            actual_return: 0.02,
            predicted_return: 0.015,
            accuracy: 0.85,
            timestamp: 1640995200000,
        },
        ExpertPerformance {
            expert_id: "mean_reversion_expert".to_string(),
            strategy_type: StrategyType::MeanReversion,
            actual_return: -0.03,
            predicted_return: 0.02,
            accuracy: 0.6,
            timestamp: 1640995200000,
        },
    ];
    
    for performance in performances {
        engine.update_expert_weights(&performance).await.unwrap();
    }
    
    // Get performance metrics
    let metrics = engine.get_performance_metrics().await.unwrap();
    
    // Should track performance for all experts
    assert!(metrics.total_experts >= 3);
    assert!(metrics.average_accuracy >= 0.0 && metrics.average_accuracy <= 1.0);
    assert!(metrics.total_optimizations > 0);
    
    // Best performing expert should have high weight
    let expert_stats = engine.get_expert_statistics().await.unwrap();
    
    // Momentum expert (best performance) should have higher weight
    let momentum_stats = expert_stats.iter()
        .find(|s| s.strategy_type == StrategyType::Momentum)
        .unwrap();
    
    let mean_reversion_stats = expert_stats.iter()
        .find(|s| s.strategy_type == StrategyType::MeanReversion)
        .unwrap();
    
    assert!(momentum_stats.current_weight >= mean_reversion_stats.current_weight);
}

#[tokio::test]
async fn test_portfolio_rebalancing() {
    let mut config = HedgeConfig::default();
    config.rebalancing_threshold = 0.1; // 10% threshold
    
    let mut engine = QuantumHedgeEngine::new(config).unwrap();
    
    let initial_data = MarketData {
        symbol: "BTC/USDT".to_string(),
        current_price: 50000.0,
        possible_outcomes: vec![52000.0, 48000.0],
        buy_probabilities: vec![0.5, 0.5],
        sell_probabilities: vec![0.5, 0.5],
        hold_probabilities: vec![0.5, 0.5],
        frame: prospect_theory::FramingContext {
            frame_type: prospect_theory::FrameType::Neutral,
            emphasis: 0.5,
        },
        timestamp: 1640995200000,
    };
    
    // Get initial portfolio
    let initial_result = engine.optimize_portfolio(&initial_data, None).await.unwrap();
    
    // Simulate significant market change
    let changed_data = MarketData {
        symbol: "BTC/USDT".to_string(),
        current_price: 55000.0, // 10% price increase
        possible_outcomes: vec![60000.0, 50000.0],
        buy_probabilities: vec![0.8, 0.2], // Strong bullish signal
        sell_probabilities: vec![0.2, 0.8],
        hold_probabilities: vec![0.3, 0.7],
        frame: prospect_theory::FramingContext {
            frame_type: prospect_theory::FrameType::Gain,
            emphasis: 0.9,
        },
        timestamp: 1640998800000, // 1 hour later
    };
    
    // Get rebalanced portfolio
    let rebalanced_result = engine.optimize_portfolio(&changed_data, Some(&initial_result)).await.unwrap();
    
    // Portfolio should have changed significantly due to market shift
    assert_ne!(initial_result.portfolio_weights, rebalanced_result.portfolio_weights);
    
    // Should trigger rebalancing due to threshold
    assert!(rebalanced_result.rebalancing_required);
    
    // Weights should still sum to ~1.0
    let weight_sum: f64 = rebalanced_result.portfolio_weights.iter().map(|w| w.weight).sum();
    assert!((weight_sum - 1.0).abs() < 0.01);
}

#[tokio::test]
async fn test_risk_management() {
    let mut config = HedgeConfig::default();
    config.max_position_size = 0.3; // 30% max position
    config.risk_tolerance = 0.2; // 20% risk tolerance
    
    let mut engine = QuantumHedgeEngine::new(config).unwrap();
    
    // High-risk market scenario
    let high_risk_data = MarketData {
        symbol: "BTC/USDT".to_string(),
        current_price: 50000.0,
        possible_outcomes: vec![70000.0, 30000.0], // High volatility
        buy_probabilities: vec![0.5, 0.5],
        sell_probabilities: vec![0.5, 0.5],
        hold_probabilities: vec![0.5, 0.5],
        frame: prospect_theory::FramingContext {
            frame_type: prospect_theory::FrameType::Loss,
            emphasis: 0.8,
        },
        timestamp: 1640995200000,
    };
    
    let result = engine.optimize_portfolio(&high_risk_data, None).await.unwrap();
    
    // Risk management should limit position sizes
    for weight in &result.portfolio_weights {
        assert!(weight.weight <= 0.35); // Allow small buffer for rounding
    }
    
    // Should have distributed risk across multiple strategies
    assert!(result.portfolio_weights.len() >= 3);
    
    // Total risk should be within tolerance
    assert!(result.total_risk <= 0.25); // Allow small buffer
}

#[tokio::test]
async fn test_edge_cases() {
    let config = HedgeConfig::default();
    let mut engine = QuantumHedgeEngine::new(config).unwrap();
    
    // Test with extreme market data
    let extreme_data = MarketData {
        symbol: "BTC/USDT".to_string(),
        current_price: 1.0, // Very low price
        possible_outcomes: vec![1000000.0, 0.001], // Extreme range
        buy_probabilities: vec![0.001, 0.999], // Extreme probabilities
        sell_probabilities: vec![0.999, 0.001],
        hold_probabilities: vec![0.5, 0.5],
        frame: prospect_theory::FramingContext {
            frame_type: prospect_theory::FrameType::Loss,
            emphasis: 1.0,
        },
        timestamp: 1640995200000,
    };
    
    let extreme_result = engine.optimize_portfolio(&extreme_data, None).await;
    assert!(extreme_result.is_ok());
    
    let result = extreme_result.unwrap();
    assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
    
    // Weights should still sum to ~1.0 even in extreme cases
    let weight_sum: f64 = result.portfolio_weights.iter().map(|w| w.weight).sum();
    assert!((weight_sum - 1.0).abs() < 0.1); // Allow larger tolerance for extreme cases
    
    // Test with empty market data
    let empty_data = MarketData {
        symbol: "BTC/USDT".to_string(),
        current_price: 50000.0,
        possible_outcomes: vec![],
        buy_probabilities: vec![],
        sell_probabilities: vec![],
        hold_probabilities: vec![],
        frame: prospect_theory::FramingContext {
            frame_type: prospect_theory::FrameType::Neutral,
            emphasis: 0.5,
        },
        timestamp: 1640995200000,
    };
    
    let empty_result = engine.optimize_portfolio(&empty_data, None).await;
    // Should either handle gracefully or return appropriate error
    match empty_result {
        Ok(result) => {
            assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
        },
        Err(_) => {
            // Acceptable to return error for invalid input
        }
    }
}

#[tokio::test]
async fn test_consensus_mechanisms() {
    let mut config = HedgeConfig::default();
    config.consensus_threshold = 0.75; // 75% consensus required
    
    let mut engine = QuantumHedgeEngine::new(config).unwrap();
    
    let market_data = MarketData {
        symbol: "BTC/USDT".to_string(),
        current_price: 50000.0,
        possible_outcomes: vec![53000.0, 47000.0],
        buy_probabilities: vec![0.7, 0.3], // Moderate bullish
        sell_probabilities: vec![0.3, 0.7],
        hold_probabilities: vec![0.4, 0.6],
        frame: prospect_theory::FramingContext {
            frame_type: prospect_theory::FrameType::Gain,
            emphasis: 0.6,
        },
        timestamp: 1640995200000,
    };
    
    let result = engine.optimize_portfolio(&market_data, None).await.unwrap();
    
    // Check consensus metrics
    assert!(result.consensus_strength >= 0.0 && result.consensus_strength <= 1.0);
    
    // If consensus is strong, confidence should be high
    if result.consensus_strength > 0.8 {
        assert!(result.confidence > 0.6);
    }
    
    // Portfolio should reflect consensus
    let dominant_weight = result.portfolio_weights.iter()
        .map(|w| w.weight)
        .fold(0.0, f64::max);
    
    if result.consensus_strength > 0.75 {
        assert!(dominant_weight > 0.3); // Strong consensus should lead to concentrated allocation
    }
}