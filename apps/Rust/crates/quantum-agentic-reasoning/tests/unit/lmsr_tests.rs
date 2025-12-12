//! Comprehensive unit tests for Quantum LMSR integration
//! 
//! Tests all aspects of Logarithmic Market Scoring Rule integration:
//! - Market belief updating and adaptation
//! - Quantum amplitude amplification of high-confidence beliefs
//! - Memory decay and temporal dynamics
//! - Performance optimization and caching
//! - Multi-symbol prediction markets

use quantum_agentic_reasoning::{lmsr_integration::*, MarketData, Result};
use std::collections::HashMap;

#[tokio::test]
async fn test_lmsr_predictor_creation() {
    let config = LMSRConfig::default();
    let predictor = QuantumLMSRPredictor::new(config);
    
    assert!(predictor.is_ok());
    let predictor = predictor.unwrap();
    
    // Should start with empty beliefs
    assert_eq!(predictor.get_active_symbols().len(), 0);
}

#[tokio::test]
async fn test_basic_prediction() {
    let config = LMSRConfig::default();
    let mut predictor = QuantumLMSRPredictor::new(config).unwrap();
    
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
    
    let prediction = predictor.predict(&market_data).await;
    assert!(prediction.is_ok());
    
    let prediction = prediction.unwrap();
    assert!(prediction.confidence >= 0.0 && prediction.confidence <= 1.0);
    assert!(prediction.expected_return.is_finite());
    assert!(prediction.risk_metric >= 0.0);
    
    // Should have created beliefs for BTC/USDT
    assert_eq!(predictor.get_active_symbols().len(), 1);
    assert!(predictor.get_active_symbols().contains(&"BTC/USDT".to_string()));
}

#[tokio::test]
async fn test_belief_updating() {
    let config = LMSRConfig::default();
    let mut predictor = QuantumLMSRPredictor::new(config).unwrap();
    
    let initial_data = MarketData {
        symbol: "BTC/USDT".to_string(),
        current_price: 50000.0,
        possible_outcomes: vec![52000.0, 48000.0],
        buy_probabilities: vec![0.7, 0.3], // Strong buy signal
        sell_probabilities: vec![0.3, 0.7],
        hold_probabilities: vec![0.5, 0.5],
        frame: prospect_theory::FramingContext {
            frame_type: prospect_theory::FrameType::Gain,
            emphasis: 0.8,
        },
        timestamp: 1640995200000,
    };
    
    let initial_prediction = predictor.predict(&initial_data).await.unwrap();
    
    // Update with new contradictory information
    let mut updated_data = initial_data.clone();
    updated_data.buy_probabilities = vec![0.2, 0.8]; // Now bearish
    updated_data.sell_probabilities = vec![0.8, 0.2];
    updated_data.timestamp += 1000;
    
    let updated_prediction = predictor.predict(&updated_data).await.unwrap();
    
    // Predictions should differ as beliefs update
    assert_ne!(initial_prediction.confidence, updated_prediction.confidence);
    assert_ne!(initial_prediction.expected_return, updated_prediction.expected_return);
    
    // Should still be valid predictions
    assert!(updated_prediction.confidence >= 0.0 && updated_prediction.confidence <= 1.0);
}

#[tokio::test]
async fn test_adaptive_learning() {
    let mut config = LMSRConfig::default();
    config.adaptive_learning = true;
    config.learning_rate = 0.1;
    
    let mut predictor = QuantumLMSRPredictor::new(config).unwrap();
    
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
    
    // Make initial prediction
    let initial_prediction = predictor.predict(&market_data).await.unwrap();
    
    // Update with actual outcome (positive)
    let outcome = TrainingOutcome {
        symbol: "BTC/USDT".to_string(),
        predicted_action: initial_prediction.predicted_action.clone(),
        actual_return: 0.08, // 8% gain
        prediction_accuracy: 0.85,
        timestamp: market_data.timestamp + 3600000, // 1 hour later
    };
    
    predictor.update_with_outcome(&outcome).await.unwrap();
    
    // Make new prediction - should be influenced by learning
    let mut new_data = market_data.clone();
    new_data.timestamp += 7200000; // 2 hours later
    let learned_prediction = predictor.predict(&new_data).await.unwrap();
    
    // Learning should influence future predictions
    // (exact behavior depends on the learning algorithm)
    assert!(learned_prediction.confidence >= 0.0 && learned_prediction.confidence <= 1.0);
}

#[tokio::test]
async fn test_quantum_enhancement() {
    // Test with quantum enhancement disabled
    let mut config_classical = LMSRConfig::default();
    config_classical.quantum_enhancement = false;
    let mut predictor_classical = QuantumLMSRPredictor::new(config_classical).unwrap();
    
    // Test with quantum enhancement enabled
    let mut config_quantum = LMSRConfig::default();
    config_quantum.quantum_enhancement = true;
    config_quantum.amplitude_amplification_iterations = 3;
    let mut predictor_quantum = QuantumLMSRPredictor::new(config_quantum).unwrap();
    
    let market_data = MarketData {
        symbol: "BTC/USDT".to_string(),
        current_price: 50000.0,
        possible_outcomes: vec![55000.0, 45000.0],
        buy_probabilities: vec![0.8, 0.2], // High confidence signal
        sell_probabilities: vec![0.2, 0.8],
        hold_probabilities: vec![0.3, 0.7],
        frame: prospect_theory::FramingContext {
            frame_type: prospect_theory::FrameType::Gain,
            emphasis: 0.9,
        },
        timestamp: 1640995200000,
    };
    
    let classical_prediction = predictor_classical.predict(&market_data).await.unwrap();
    let quantum_prediction = predictor_quantum.predict(&market_data).await.unwrap();
    
    // Quantum enhancement should amplify high-confidence predictions
    if quantum_prediction.confidence > 0.7 {
        assert!(quantum_prediction.confidence >= classical_prediction.confidence);
    }
    
    // Both should produce valid results
    assert!(classical_prediction.confidence >= 0.0 && classical_prediction.confidence <= 1.0);
    assert!(quantum_prediction.confidence >= 0.0 && quantum_prediction.confidence <= 1.0);
}

#[tokio::test]
async fn test_memory_decay() {
    let mut config = LMSRConfig::default();
    config.memory_decay_rate = 0.1; // 10% decay
    config.memory_window_hours = 24;
    
    let mut predictor = QuantumLMSRPredictor::new(config).unwrap();
    
    let old_data = MarketData {
        symbol: "BTC/USDT".to_string(),
        current_price: 50000.0,
        possible_outcomes: vec![52000.0, 48000.0],
        buy_probabilities: vec![0.8, 0.2], // Strong signal
        sell_probabilities: vec![0.2, 0.8],
        hold_probabilities: vec![0.4, 0.6],
        frame: prospect_theory::FramingContext {
            frame_type: prospect_theory::FrameType::Gain,
            emphasis: 0.8,
        },
        timestamp: 1640995200000, // 24+ hours ago
    };
    
    // Make prediction with old data
    predictor.predict(&old_data).await.unwrap();
    
    // Wait and make prediction with recent data
    let recent_data = MarketData {
        symbol: "BTC/USDT".to_string(),
        current_price: 50000.0,
        possible_outcomes: vec![52000.0, 48000.0],
        buy_probabilities: vec![0.3, 0.7], // Opposite signal
        sell_probabilities: vec![0.7, 0.3],
        hold_probabilities: vec![0.5, 0.5],
        frame: prospect_theory::FramingContext {
            frame_type: prospect_theory::FrameType::Loss,
            emphasis: 0.6,
        },
        timestamp: 1641081600000, // Current time
    };
    
    let recent_prediction = predictor.predict(&recent_data).await.unwrap();
    
    // Recent data should dominate due to memory decay
    assert!(recent_prediction.confidence >= 0.0 && recent_prediction.confidence <= 1.0);
    
    // Check that memory decay is working
    let beliefs = predictor.get_market_beliefs("BTC/USDT").unwrap();
    assert!(beliefs.belief_strength < 1.0); // Should have decayed from initial strong signal
}

#[tokio::test]
async fn test_multi_symbol_predictions() {
    let config = LMSRConfig::default();
    let mut predictor = QuantumLMSRPredictor::new(config).unwrap();
    
    let btc_data = MarketData {
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
    
    let eth_data = MarketData {
        symbol: "ETH/USDT".to_string(),
        current_price: 4000.0,
        possible_outcomes: vec![4200.0, 3800.0],
        buy_probabilities: vec![0.7, 0.3],
        sell_probabilities: vec![0.3, 0.7],
        hold_probabilities: vec![0.5, 0.5],
        frame: prospect_theory::FramingContext {
            frame_type: prospect_theory::FrameType::Gain,
            emphasis: 0.7,
        },
        timestamp: 1640995200000,
    };
    
    // Make predictions for both symbols
    let btc_prediction = predictor.predict(&btc_data).await.unwrap();
    let eth_prediction = predictor.predict(&eth_data).await.unwrap();
    
    // Should have beliefs for both symbols
    assert_eq!(predictor.get_active_symbols().len(), 2);
    assert!(predictor.get_active_symbols().contains(&"BTC/USDT".to_string()));
    assert!(predictor.get_active_symbols().contains(&"ETH/USDT".to_string()));
    
    // Predictions should be independent and valid
    assert!(btc_prediction.confidence >= 0.0 && btc_prediction.confidence <= 1.0);
    assert!(eth_prediction.confidence >= 0.0 && eth_prediction.confidence <= 1.0);
    
    // ETH has stronger signal, should have higher confidence
    assert!(eth_prediction.confidence >= btc_prediction.confidence);
}

#[tokio::test]
async fn test_performance_constraints() {
    let mut config = LMSRConfig::default();
    config.target_latency_ns = 500; // 500ns target
    
    let mut predictor = QuantumLMSRPredictor::new(config).unwrap();
    
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
    let prediction = predictor.predict(&market_data).await.unwrap();
    let elapsed = start.elapsed();
    
    // Should meet performance target (allowing overhead for test environment)
    assert!(elapsed.as_nanos() < 5000); // 5μs max in test
    
    // Should still produce valid prediction
    assert!(prediction.confidence >= 0.0 && prediction.confidence <= 1.0);
    assert!(prediction.expected_return.is_finite());
}

#[tokio::test]
async fn test_performance_cache() {
    let mut config = LMSRConfig::default();
    config.cache_size = 1000;
    
    let mut predictor = QuantumLMSRPredictor::new(config).unwrap();
    
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
    
    // First prediction should populate cache
    let start1 = std::time::Instant::now();
    let prediction1 = predictor.predict(&market_data).await.unwrap();
    let elapsed1 = start1.elapsed();
    
    // Second prediction with same data should be faster (cache hit)
    let start2 = std::time::Instant::now();
    let prediction2 = predictor.predict(&market_data).await.unwrap();
    let elapsed2 = start2.elapsed();
    
    // Cache hit should be faster
    assert!(elapsed2 <= elapsed1);
    
    // Results should be identical for same input
    assert_eq!(prediction1.confidence, prediction2.confidence);
    assert_eq!(prediction1.expected_return, prediction2.expected_return);
}

#[tokio::test]
async fn test_belief_strength_calculation() {
    let config = LMSRConfig::default();
    let mut predictor = QuantumLMSRPredictor::new(config).unwrap();
    
    // Test with weak signal
    let weak_data = MarketData {
        symbol: "BTC/USDT".to_string(),
        current_price: 50000.0,
        possible_outcomes: vec![51000.0, 49000.0],
        buy_probabilities: vec![0.52, 0.48], // Weak signal
        sell_probabilities: vec![0.48, 0.52],
        hold_probabilities: vec![0.5, 0.5],
        frame: prospect_theory::FramingContext {
            frame_type: prospect_theory::FrameType::Neutral,
            emphasis: 0.5,
        },
        timestamp: 1640995200000,
    };
    
    let weak_prediction = predictor.predict(&weak_data).await.unwrap();
    
    // Test with strong signal
    let strong_data = MarketData {
        symbol: "ETH/USDT".to_string(),
        current_price: 4000.0,
        possible_outcomes: vec![4500.0, 3500.0],
        buy_probabilities: vec![0.9, 0.1], // Strong signal
        sell_probabilities: vec![0.1, 0.9],
        hold_probabilities: vec![0.2, 0.8],
        frame: prospect_theory::FramingContext {
            frame_type: prospect_theory::FrameType::Gain,
            emphasis: 0.9,
        },
        timestamp: 1640995200000,
    };
    
    let strong_prediction = predictor.predict(&strong_data).await.unwrap();
    
    // Strong signal should have higher confidence
    assert!(strong_prediction.confidence > weak_prediction.confidence);
    
    // Check belief strengths
    let btc_beliefs = predictor.get_market_beliefs("BTC/USDT").unwrap();
    let eth_beliefs = predictor.get_market_beliefs("ETH/USDT").unwrap();
    
    assert!(eth_beliefs.belief_strength > btc_beliefs.belief_strength);
}

#[tokio::test]
async fn test_edge_cases() {
    let config = LMSRConfig::default();
    let mut predictor = QuantumLMSRPredictor::new(config).unwrap();
    
    // Test with extreme probabilities
    let extreme_data = MarketData {
        symbol: "BTC/USDT".to_string(),
        current_price: 50000.0,
        possible_outcomes: vec![100000.0, 1.0],
        buy_probabilities: vec![0.001, 0.999], // Extreme probabilities
        sell_probabilities: vec![0.999, 0.001],
        hold_probabilities: vec![0.5, 0.5],
        frame: prospect_theory::FramingContext {
            frame_type: prospect_theory::FrameType::Loss,
            emphasis: 1.0,
        },
        timestamp: 1640995200000,
    };
    
    let extreme_prediction = predictor.predict(&extreme_data).await;
    assert!(extreme_prediction.is_ok());
    
    let prediction = extreme_prediction.unwrap();
    assert!(prediction.confidence >= 0.0 && prediction.confidence <= 1.0);
    assert!(prediction.expected_return.is_finite());
    
    // Test with empty outcomes
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
    
    let empty_prediction = predictor.predict(&empty_data).await;
    // Should handle gracefully (either OK with default values or appropriate error)
    match empty_prediction {
        Ok(pred) => {
            assert!(pred.confidence >= 0.0 && pred.confidence <= 1.0);
        },
        Err(_) => {
            // Acceptable to return error for invalid input
        }
    }
}