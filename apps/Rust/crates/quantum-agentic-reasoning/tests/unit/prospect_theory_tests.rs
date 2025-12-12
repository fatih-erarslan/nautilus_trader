//! Comprehensive unit tests for Quantum Prospect Theory integration
//! 
//! Tests all aspects of behavioral economics integration including:
//! - Kahneman-Tversky value functions
//! - Loss aversion mechanisms  
//! - Probability weighting functions
//! - Framing effects and mental accounting
//! - Quantum amplitude amplification
//! - Reference point adaptation

use quantum_agentic_reasoning::*;
use prospect_theory::*;
use std::collections::HashMap;

#[tokio::test]
async fn test_prospect_theory_value_function() {
    let config = QuantumProspectTheoryConfig::default();
    let pt = QuantumProspectTheory::new(config).unwrap();
    
    // Test S-shaped value function (concave for gains, convex for losses)
    let gain_value = pt.evaluate_value_function(100.0, 0.0).unwrap();
    let loss_value = pt.evaluate_value_function(-100.0, 0.0).unwrap();
    
    // Gains should be concave (diminishing returns)
    let small_gain = pt.evaluate_value_function(50.0, 0.0).unwrap();
    let large_gain = pt.evaluate_value_function(200.0, 0.0).unwrap();
    assert!((large_gain - gain_value) < (gain_value - small_gain));
    
    // Losses should be convex (accelerating pain)
    let small_loss = pt.evaluate_value_function(-50.0, 0.0).unwrap();
    let large_loss = pt.evaluate_value_function(-200.0, 0.0).unwrap();
    assert!((loss_value - large_loss).abs() < (small_loss - loss_value).abs());
    
    // Loss aversion: |v(-x)| > v(x) for same magnitude
    assert!(loss_value.abs() > gain_value);
    assert!((loss_value.abs() / gain_value) >= 2.0); // λ = 2.25 default
}

#[tokio::test]
async fn test_probability_weighting_functions() {
    let config = QuantumProspectTheoryConfig::default();
    let pt = QuantumProspectTheory::new(config).unwrap();
    
    // Test both Tversky-Kahneman and Prelec weighting functions
    let probabilities = vec![0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99];
    
    for &p in &probabilities {
        let tk_weight = pt.evaluate_tversky_kahneman_weighting(p).unwrap();
        let prelec_weight = pt.evaluate_prelec_weighting(p).unwrap();
        
        // Weights should be between 0 and 1
        assert!(tk_weight >= 0.0 && tk_weight <= 1.0);
        assert!(prelec_weight >= 0.0 && prelec_weight <= 1.0);
        
        // Test overweighting of small probabilities
        if p < 0.1 {
            assert!(tk_weight > p);
            assert!(prelec_weight > p);
        }
        
        // Test underweighting of moderate probabilities
        if p > 0.3 && p < 0.7 {
            assert!(tk_weight < p);
            assert!(prelec_weight < p);
        }
    }
    
    // Test boundary conditions
    assert_eq!(pt.evaluate_tversky_kahneman_weighting(0.0).unwrap(), 0.0);
    assert_eq!(pt.evaluate_tversky_kahneman_weighting(1.0).unwrap(), 1.0);
    assert_eq!(pt.evaluate_prelec_weighting(0.0).unwrap(), 0.0);
    assert_eq!(pt.evaluate_prelec_weighting(1.0).unwrap(), 1.0);
}

#[tokio::test]
async fn test_framing_effects() {
    let config = QuantumProspectTheoryConfig::default();
    let pt = QuantumProspectTheory::new(config).unwrap();
    
    let base_market_data = MarketData {
        symbol: "BTC/USDT".to_string(),
        current_price: 50000.0,
        possible_outcomes: vec![52000.0, 48000.0],
        buy_probabilities: vec![0.5, 0.5],
        sell_probabilities: vec![0.5, 0.5],
        hold_probabilities: vec![0.5, 0.5],
        frame: FramingContext {
            frame_type: FrameType::Neutral,
            emphasis: 0.5,
        },
        timestamp: 1640995200000,
    };
    
    // Test gain frame
    let mut gain_frame_data = base_market_data.clone();
    gain_frame_data.frame = FramingContext {
        frame_type: FrameType::Gain,
        emphasis: 0.8,
    };
    
    // Test loss frame
    let mut loss_frame_data = base_market_data.clone();
    loss_frame_data.frame = FramingContext {
        frame_type: FrameType::Loss,
        emphasis: 0.8,
    };
    
    let neutral_decision = pt.make_trading_decision(&base_market_data, None).unwrap();
    let gain_decision = pt.make_trading_decision(&gain_frame_data, None).unwrap();
    let loss_decision = pt.make_trading_decision(&loss_frame_data, None).unwrap();
    
    // Framing should influence decision making
    assert_ne!(neutral_decision.confidence, gain_decision.confidence);
    assert_ne!(neutral_decision.confidence, loss_decision.confidence);
    assert_ne!(gain_decision.confidence, loss_decision.confidence);
    
    // Gain frame should generally increase confidence for positive actions
    // Loss frame should generally decrease confidence
    if matches!(gain_decision.action, TradingAction::Buy) {
        assert!(gain_decision.confidence >= neutral_decision.confidence);
    }
}

#[tokio::test]
async fn test_mental_accounting() {
    let config = QuantumProspectTheoryConfig::default();
    let pt = QuantumProspectTheory::new(config).unwrap();
    
    // Create positions in different "mental accounts"
    let conservative_position = Position {
        symbol: "BTC/USDT".to_string(),
        quantity: 1.0,
        entry_price: 48000.0,
        current_value: 50000.0,
        unrealized_pnl: 2000.0, // Gain
    };
    
    let speculative_position = Position {
        symbol: "BTC/USDT".to_string(), 
        quantity: 2.0,
        entry_price: 52000.0,
        current_value: 50000.0,
        unrealized_pnl: -4000.0, // Loss
    };
    
    let market_data = MarketData {
        symbol: "BTC/USDT".to_string(),
        current_price: 50000.0,
        possible_outcomes: vec![52000.0, 48000.0],
        buy_probabilities: vec![0.6, 0.4],
        sell_probabilities: vec![0.4, 0.6],
        hold_probabilities: vec![0.5, 0.5],
        frame: FramingContext {
            frame_type: FrameType::Neutral,
            emphasis: 0.5,
        },
        timestamp: 1640995200000,
    };
    
    let conservative_decision = pt.make_trading_decision(&market_data, Some(&conservative_position)).unwrap();
    let speculative_decision = pt.make_trading_decision(&market_data, Some(&speculative_position)).unwrap();
    let no_position_decision = pt.make_trading_decision(&market_data, None).unwrap();
    
    // Mental accounting should lead to different decisions based on position context
    assert_ne!(conservative_decision.behavioral_factors.mental_accounting_bias,
               speculative_decision.behavioral_factors.mental_accounting_bias);
    
    // Position context should influence behavioral factors
    assert_ne!(conservative_decision.behavioral_factors.loss_aversion_impact,
               speculative_decision.behavioral_factors.loss_aversion_impact);
}

#[tokio::test]
async fn test_quantum_amplitude_amplification() {
    let mut config = QuantumProspectTheoryConfig::default();
    config.quantum_enhancement = true;
    config.amplitude_amplification_iterations = 3;
    
    let pt_quantum = QuantumProspectTheory::new(config.clone()).unwrap();
    
    config.quantum_enhancement = false;
    let pt_classical = QuantumProspectTheory::new(config).unwrap();
    
    let market_data = MarketData {
        symbol: "BTC/USDT".to_string(),
        current_price: 50000.0,
        possible_outcomes: vec![55000.0, 45000.0],
        buy_probabilities: vec![0.7, 0.3], // Strong buy signal
        sell_probabilities: vec![0.2, 0.8],
        hold_probabilities: vec![0.4, 0.6],
        frame: FramingContext {
            frame_type: FrameType::Gain,
            emphasis: 0.7,
        },
        timestamp: 1640995200000,
    };
    
    let quantum_decision = pt_quantum.make_trading_decision(&market_data, None).unwrap();
    let classical_decision = pt_classical.make_trading_decision(&market_data, None).unwrap();
    
    // Quantum enhancement should amplify high-confidence decisions
    if quantum_decision.confidence > 0.7 {
        assert!(quantum_decision.confidence >= classical_decision.confidence);
    }
    
    // Both should make reasonable decisions
    assert!(quantum_decision.confidence >= 0.0 && quantum_decision.confidence <= 1.0);
    assert!(classical_decision.confidence >= 0.0 && classical_decision.confidence <= 1.0);
}

#[tokio::test]
async fn test_reference_point_adaptation() {
    let config = QuantumProspectTheoryConfig::default();
    let mut pt = QuantumProspectTheory::new(config).unwrap();
    
    // Start with reference point of 50000
    let initial_data = MarketData {
        symbol: "BTC/USDT".to_string(),
        current_price: 50000.0,
        possible_outcomes: vec![52000.0, 48000.0],
        buy_probabilities: vec![0.5, 0.5],
        sell_probabilities: vec![0.5, 0.5],
        hold_probabilities: vec![0.5, 0.5],
        frame: FramingContext {
            frame_type: FrameType::Neutral,
            emphasis: 0.5,
        },
        timestamp: 1640995200000,
    };
    
    let initial_decision = pt.make_trading_decision(&initial_data, None).unwrap();
    let initial_ref = pt.get_reference_point("BTC/USDT").unwrap();
    
    // Process multiple price movements
    let price_sequence = vec![51000.0, 52000.0, 53000.0, 52500.0];
    for price in price_sequence {
        let mut data = initial_data.clone();
        data.current_price = price;
        data.timestamp += 1000;
        
        pt.make_trading_decision(&data, None).unwrap();
    }
    
    let final_ref = pt.get_reference_point("BTC/USDT").unwrap();
    
    // Reference point should have adapted upward
    assert!(final_ref > initial_ref);
    assert!(final_ref < 53000.0); // But not fully to the highest price
    assert!(final_ref > 50000.0); // And definitely above initial
}

#[tokio::test]
async fn test_behavioral_factor_calculation() {
    let config = QuantumProspectTheoryConfig::default();
    let pt = QuantumProspectTheory::new(config).unwrap();
    
    // Test gain scenario
    let gain_position = Position {
        symbol: "BTC/USDT".to_string(),
        quantity: 1.0,
        entry_price: 45000.0,
        current_value: 50000.0,
        unrealized_pnl: 5000.0,
    };
    
    let market_data = MarketData {
        symbol: "BTC/USDT".to_string(),
        current_price: 50000.0,
        possible_outcomes: vec![52000.0, 48000.0],
        buy_probabilities: vec![0.6, 0.4],
        sell_probabilities: vec![0.3, 0.7],
        hold_probabilities: vec![0.5, 0.5],
        frame: FramingContext {
            frame_type: FrameType::Gain,
            emphasis: 0.7,
        },
        timestamp: 1640995200000,
    };
    
    let gain_decision = pt.make_trading_decision(&market_data, Some(&gain_position)).unwrap();
    
    // Test loss scenario
    let loss_position = Position {
        symbol: "BTC/USDT".to_string(),
        quantity: 1.0,
        entry_price: 55000.0,
        current_value: 50000.0,
        unrealized_pnl: -5000.0,
    };
    
    let loss_decision = pt.make_trading_decision(&market_data, Some(&loss_position)).unwrap();
    
    // Loss aversion should be stronger for loss position
    assert!(loss_decision.behavioral_factors.loss_aversion_impact.abs() > 
            gain_decision.behavioral_factors.loss_aversion_impact.abs());
    
    // Mental accounting should differ between positions
    assert_ne!(gain_decision.behavioral_factors.mental_accounting_bias,
               loss_decision.behavioral_factors.mental_accounting_bias);
    
    // Both should have reasonable probability weighting bias
    assert!(gain_decision.behavioral_factors.probability_weighting_bias.abs() <= 1.0);
    assert!(loss_decision.behavioral_factors.probability_weighting_bias.abs() <= 1.0);
}

#[tokio::test]
async fn test_performance_constraints() {
    let mut config = QuantumProspectTheoryConfig::default();
    config.target_latency_ns = 1000; // 1μs target
    
    let pt = QuantumProspectTheory::new(config).unwrap();
    
    let market_data = MarketData {
        symbol: "BTC/USDT".to_string(),
        current_price: 50000.0,
        possible_outcomes: vec![52000.0, 48000.0],
        buy_probabilities: vec![0.6, 0.4],
        sell_probabilities: vec![0.4, 0.6],
        hold_probabilities: vec![0.5, 0.5],
        frame: FramingContext {
            frame_type: FrameType::Neutral,
            emphasis: 0.5,
        },
        timestamp: 1640995200000,
    };
    
    let start = std::time::Instant::now();
    let decision = pt.make_trading_decision(&market_data, None).unwrap();
    let elapsed = start.elapsed();
    
    // Should meet performance target (allowing some overhead for test environment)
    assert!(elapsed.as_nanos() < 10_000); // 10μs max in test environment
    
    // Should still produce valid decision
    assert!(decision.confidence >= 0.0 && decision.confidence <= 1.0);
    assert!(decision.prospect_value.is_finite());
}

#[tokio::test]
async fn test_edge_cases_and_boundary_conditions() {
    let config = QuantumProspectTheoryConfig::default();
    let pt = QuantumProspectTheory::new(config).unwrap();
    
    // Test with extreme market data
    let extreme_data = MarketData {
        symbol: "BTC/USDT".to_string(),
        current_price: 1.0, // Very low price
        possible_outcomes: vec![1000000.0, 0.001], // Extreme outcomes
        buy_probabilities: vec![0.001, 0.999], // Extreme probabilities
        sell_probabilities: vec![0.999, 0.001],
        hold_probabilities: vec![0.5, 0.5],
        frame: FramingContext {
            frame_type: FrameType::Loss,
            emphasis: 1.0, // Maximum emphasis
        },
        timestamp: 1640995200000,
    };
    
    let decision = pt.make_trading_decision(&extreme_data, None);
    assert!(decision.is_ok());
    
    let decision = decision.unwrap();
    assert!(decision.confidence >= 0.0 && decision.confidence <= 1.0);
    assert!(decision.prospect_value.is_finite());
    
    // Test with empty outcomes (should handle gracefully)
    let empty_outcomes_data = MarketData {
        symbol: "BTC/USDT".to_string(),
        current_price: 50000.0,
        possible_outcomes: vec![], // Empty
        buy_probabilities: vec![],
        sell_probabilities: vec![],
        hold_probabilities: vec![],
        frame: FramingContext {
            frame_type: FrameType::Neutral,
            emphasis: 0.5,
        },
        timestamp: 1640995200000,
    };
    
    let empty_decision = pt.make_trading_decision(&empty_outcomes_data, None);
    // Should either handle gracefully or return appropriate error
    match empty_decision {
        Ok(decision) => {
            assert!(decision.confidence >= 0.0 && decision.confidence <= 1.0);
        },
        Err(_) => {
            // Acceptable to return error for invalid input
        }
    }
}

#[tokio::test]
async fn test_cache_efficiency() {
    let mut config = QuantumProspectTheoryConfig::default();
    config.cache_size = 100;
    
    let pt = QuantumProspectTheory::new(config).unwrap();
    
    let base_data = MarketData {
        symbol: "BTC/USDT".to_string(),
        current_price: 50000.0,
        possible_outcomes: vec![52000.0, 48000.0],
        buy_probabilities: vec![0.6, 0.4],
        sell_probabilities: vec![0.4, 0.6], 
        hold_probabilities: vec![0.5, 0.5],
        frame: FramingContext {
            frame_type: FrameType::Neutral,
            emphasis: 0.5,
        },
        timestamp: 1640995200000,
    };
    
    // First call should populate cache
    let start1 = std::time::Instant::now();
    let decision1 = pt.make_trading_decision(&base_data, None).unwrap();
    let elapsed1 = start1.elapsed();
    
    // Second call with same data should be faster (cache hit)
    let start2 = std::time::Instant::now();
    let decision2 = pt.make_trading_decision(&base_data, None).unwrap();
    let elapsed2 = start2.elapsed();
    
    // Cache hit should be significantly faster
    assert!(elapsed2 < elapsed1);
    
    // Results should be identical
    assert_eq!(decision1.confidence, decision2.confidence);
    assert_eq!(decision1.prospect_value, decision2.prospect_value);
    assert_eq!(decision1.action, decision2.action);
}