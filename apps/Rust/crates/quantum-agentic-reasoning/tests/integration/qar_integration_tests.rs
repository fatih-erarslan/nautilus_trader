//! Integration tests for complete QAR system
//! 
//! Tests the full integration of all QAR components:
//! - Quantum Prospect Theory + LMSR + Hedge algorithms
//! - Multi-agent coordination and decision synthesis
//! - Performance under realistic trading scenarios
//! - Cross-component behavioral consistency
//! - End-to-end decision making workflows

use quantum_agentic_reasoning::*;
use tokio::time::{sleep, Duration};
use std::collections::HashMap;

#[tokio::test]
async fn test_full_qar_integration() {
    let mut config = QARConfig::default();
    config.enable_lmsr = true;
    config.enable_hedge = true;
    config.quantum_enabled = true;
    config.target_latency_ns = 1000; // 1μs target
    
    let mut qar = QuantumAgenticReasoning::new(config).unwrap();
    
    let market_data = MarketData {
        symbol: "BTC/USDT".to_string(),
        current_price: 50000.0,
        possible_outcomes: vec![54000.0, 52000.0, 48000.0, 46000.0],
        buy_probabilities: vec![0.35, 0.25, 0.25, 0.15],
        sell_probabilities: vec![0.15, 0.25, 0.25, 0.35],
        hold_probabilities: vec![0.25, 0.25, 0.25, 0.25],
        frame: prospect_theory::FramingContext {
            frame_type: prospect_theory::FrameType::Neutral,
            emphasis: 0.6,
        },
        timestamp: 1640995200000,
    };
    
    let position = Position {
        symbol: "BTC/USDT".to_string(),
        quantity: 1.0,
        entry_price: 49000.0,
        current_value: 50000.0,
        unrealized_pnl: 1000.0,
    };
    
    let decision = qar.make_decision(&market_data, Some(&position)).unwrap();
    
    // Verify decision quality
    assert!(decision.confidence >= 0.0 && decision.confidence <= 1.0);
    assert!(decision.execution_time_ns < 10000); // Should be fast
    assert!(!decision.reasoning_chain.is_empty());
    
    // Should include inputs from all components
    assert!(decision.prospect_value.is_finite());
    assert!(decision.quantum_advantage.is_some());
    
    // Behavioral factors should be realistic
    assert!(decision.behavioral_factors.loss_aversion_impact.abs() <= 2.0);
    assert!(decision.behavioral_factors.probability_weighting_bias.abs() <= 1.0);
    assert!(decision.behavioral_factors.mental_accounting_bias.abs() <= 1.0);
}

#[tokio::test]
async fn test_multi_symbol_coordination() {
    let config = QARConfig::default();
    let mut qar = QuantumAgenticReasoning::new(config).unwrap();
    
    let symbols = vec!["BTC/USDT", "ETH/USDT", "SOL/USDT"];
    let mut decisions = Vec::new();
    
    for symbol in symbols {
        let market_data = MarketData {
            symbol: symbol.to_string(),
            current_price: if symbol == "BTC/USDT" { 50000.0 } 
                          else if symbol == "ETH/USDT" { 4000.0 } 
                          else { 100.0 },
            possible_outcomes: vec![110.0, 105.0, 95.0, 90.0],
            buy_probabilities: vec![0.3, 0.3, 0.2, 0.2],
            sell_probabilities: vec![0.2, 0.2, 0.3, 0.3],
            hold_probabilities: vec![0.25, 0.25, 0.25, 0.25],
            frame: prospect_theory::FramingContext {
                frame_type: prospect_theory::FrameType::Neutral,
                emphasis: 0.5,
            },
            timestamp: 1640995200000,
        };
        
        let decision = qar.make_decision(&market_data, None).unwrap();
        decisions.push((symbol, decision));
    }
    
    // All decisions should be valid
    assert_eq!(decisions.len(), 3);
    
    for (symbol, decision) in &decisions {
        assert!(decision.confidence >= 0.0 && decision.confidence <= 1.0);
        assert!(!decision.reasoning_chain.is_empty());
        assert!(decision.execution_time_ns < 10000);
    }
    
    // Decisions should show some variation across symbols
    let confidences: Vec<f64> = decisions.iter().map(|(_, d)| d.confidence).collect();
    let confidence_variance = variance(&confidences);
    assert!(confidence_variance > 0.001); // Some variation expected
}

#[tokio::test]
async fn test_behavioral_consistency() {
    let config = QARConfig::default();
    let mut qar = QuantumAgenticReasoning::new(config).unwrap();
    
    // Test loss aversion consistency
    let gain_position = Position {
        symbol: "BTC/USDT".to_string(),
        quantity: 1.0,
        entry_price: 45000.0,
        current_value: 50000.0,
        unrealized_pnl: 5000.0, // 11% gain
    };
    
    let loss_position = Position {
        symbol: "BTC/USDT".to_string(),
        quantity: 1.0,
        entry_price: 55000.0,
        current_value: 50000.0,
        unrealized_pnl: -5000.0, // 9% loss
    };
    
    let neutral_market_data = MarketData {
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
    
    let gain_decision = qar.make_decision(&neutral_market_data, Some(&gain_position)).unwrap();
    let loss_decision = qar.make_decision(&neutral_market_data, Some(&loss_position)).unwrap();
    
    // Loss aversion should make loss position more impactful
    assert!(loss_decision.behavioral_factors.loss_aversion_impact.abs() > 
            gain_decision.behavioral_factors.loss_aversion_impact.abs());
    
    // Decisions should differ based on position context
    assert_ne!(gain_decision.confidence, loss_decision.confidence);
    
    // Both should still be reasonable decisions
    assert!(gain_decision.confidence >= 0.0 && gain_decision.confidence <= 1.0);
    assert!(loss_decision.confidence >= 0.0 && loss_decision.confidence <= 1.0);
}

#[tokio::test]
async fn test_adaptive_learning_integration() {
    let config = QARConfig::default();
    let mut qar = QuantumAgenticReasoning::new(config).unwrap();
    
    let market_data = MarketData {
        symbol: "BTC/USDT".to_string(),
        current_price: 50000.0,
        possible_outcomes: vec![52000.0, 48000.0],
        buy_probabilities: vec![0.7, 0.3],
        sell_probabilities: vec![0.3, 0.7],
        hold_probabilities: vec![0.4, 0.6],
        frame: prospect_theory::FramingContext {
            frame_type: prospect_theory::FrameType::Gain,
            emphasis: 0.7,
        },
        timestamp: 1640995200000,
    };
    
    // Make initial decision
    let initial_decision = qar.make_decision(&market_data, None).unwrap();
    
    // Create training data with positive outcome
    let training_examples = vec![
        TrainingExample {
            market_data: market_data.clone(),
            decision_made: initial_decision.clone(),
            actual_outcome: 0.06, // 6% positive return
            decision_quality: 0.9, // High quality decision
        }
    ];
    
    // Train the system
    let training_results = qar.train(&training_examples).unwrap();
    assert!(training_results.training_completed);
    assert_eq!(training_results.examples_processed, 1);
    
    // Make new decision - should be influenced by learning
    let mut learned_data = market_data.clone();
    learned_data.timestamp += 3600000; // 1 hour later
    let learned_decision = qar.make_decision(&learned_data, None).unwrap();
    
    // Should still produce valid decisions
    assert!(learned_decision.confidence >= 0.0 && learned_decision.confidence <= 1.0);
    assert!(learned_decision.execution_time_ns < 10000);
}

#[tokio::test]
async fn test_performance_under_load() {
    let mut config = QARConfig::default();
    config.target_latency_ns = 500; // Aggressive 500ns target
    
    let mut qar = QuantumAgenticReasoning::new(config).unwrap();
    
    let base_market_data = MarketData {
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
    
    let mut total_time = 0u64;
    let num_decisions = 100;
    
    // Make many rapid decisions
    for i in 0..num_decisions {
        let mut market_data = base_market_data.clone();
        market_data.current_price += (i as f64 * 10.0); // Slight price variations
        market_data.timestamp += (i as u64);
        
        let start = std::time::Instant::now();
        let decision = qar.make_decision(&market_data, None).unwrap();
        let elapsed = start.elapsed();
        
        total_time += elapsed.as_nanos() as u64;
        
        // Each decision should be valid
        assert!(decision.confidence >= 0.0 && decision.confidence <= 1.0);
        assert!(decision.execution_time_ns < 5000); // Allow some overhead in test
    }
    
    let average_time = total_time / num_decisions;
    println!("Average decision time: {}ns", average_time);
    
    // Average should meet performance target
    assert!(average_time < 2000); // 2μs average allowing for test overhead
    
    // Check performance metrics
    let metrics = qar.get_performance_metrics();
    assert_eq!(metrics.total_decisions, num_decisions);
    assert!(metrics.average_decision_time_ns < 3000);
}

#[tokio::test]
async fn test_real_world_trading_scenario() {
    let config = QARConfig::default();
    let mut qar = QuantumAgenticReasoning::new(config).unwrap();
    
    // Simulate a realistic trading scenario: uptrend followed by reversal
    let scenarios = vec![
        // Initial uptrend
        MarketData {
            symbol: "BTC/USDT".to_string(),
            current_price: 48000.0,
            possible_outcomes: vec![52000.0, 50000.0, 46000.0, 44000.0],
            buy_probabilities: vec![0.4, 0.3, 0.2, 0.1],
            sell_probabilities: vec![0.1, 0.2, 0.3, 0.4],
            hold_probabilities: vec![0.25, 0.25, 0.25, 0.25],
            frame: prospect_theory::FramingContext {
                frame_type: prospect_theory::FrameType::Gain,
                emphasis: 0.6,
            },
            timestamp: 1640995200000,
        },
        // Momentum building
        MarketData {
            symbol: "BTC/USDT".to_string(),
            current_price: 51000.0,
            possible_outcomes: vec![55000.0, 53000.0, 49000.0, 47000.0],
            buy_probabilities: vec![0.5, 0.3, 0.15, 0.05],
            sell_probabilities: vec![0.05, 0.15, 0.3, 0.5],
            hold_probabilities: vec![0.2, 0.3, 0.3, 0.2],
            frame: prospect_theory::FramingContext {
                frame_type: prospect_theory::FrameType::Gain,
                emphasis: 0.8,
            },
            timestamp: 1640998800000,
        },
        // Peak and potential reversal
        MarketData {
            symbol: "BTC/USDT".to_string(),
            current_price: 54000.0,
            possible_outcomes: vec![57000.0, 55000.0, 51000.0, 48000.0],
            buy_probabilities: vec![0.3, 0.2, 0.3, 0.2],
            sell_probabilities: vec![0.2, 0.3, 0.2, 0.3],
            hold_probabilities: vec![0.3, 0.3, 0.2, 0.2],
            frame: prospect_theory::FramingContext {
                frame_type: prospect_theory::FrameType::Neutral,
                emphasis: 0.5,
            },
            timestamp: 1641002400000,
        },
        // Reversal confirmation
        MarketData {
            symbol: "BTC/USDT".to_string(),
            current_price: 50000.0,
            possible_outcomes: vec![52000.0, 50000.0, 46000.0, 42000.0],
            buy_probabilities: vec![0.2, 0.2, 0.3, 0.3],
            sell_probabilities: vec![0.3, 0.3, 0.2, 0.2],
            hold_probabilities: vec![0.25, 0.25, 0.25, 0.25],
            frame: prospect_theory::FramingContext {
                frame_type: prospect_theory::FrameType::Loss,
                emphasis: 0.7,
            },
            timestamp: 1641006000000,
        },
    ];
    
    let mut decisions = Vec::new();
    let mut position: Option<Position> = None;
    
    for (i, market_data) in scenarios.iter().enumerate() {
        let decision = qar.make_decision(market_data, position.as_ref()).unwrap();
        
        // Update position based on decision
        match decision.action {
            TradingAction::Buy => {
                position = Some(Position {
                    symbol: market_data.symbol.clone(),
                    quantity: 1.0,
                    entry_price: market_data.current_price,
                    current_value: market_data.current_price,
                    unrealized_pnl: 0.0,
                });
            },
            TradingAction::Sell => {
                if let Some(pos) = &position {
                    let pnl = market_data.current_price - pos.entry_price;
                    println!("Closed position with P&L: {}", pnl);
                }
                position = None;
            },
            TradingAction::Hold => {
                if let Some(pos) = &mut position {
                    pos.current_value = market_data.current_price;
                    pos.unrealized_pnl = market_data.current_price - pos.entry_price;
                }
            }
        }
        
        decisions.push(decision);
        
        println!("Scenario {}: Price={}, Action={:?}, Confidence={:.3}, P&L={:.0}", 
                i+1, 
                market_data.current_price,
                decisions[i].action,
                decisions[i].confidence,
                position.as_ref().map(|p| p.unrealized_pnl).unwrap_or(0.0));
    }
    
    // Verify decision quality throughout scenario
    for decision in &decisions {
        assert!(decision.confidence >= 0.0 && decision.confidence <= 1.0);
        assert!(decision.execution_time_ns < 10000);
        assert!(!decision.reasoning_chain.is_empty());
    }
    
    // Should show adaptive behavior as market conditions change
    let confidences: Vec<f64> = decisions.iter().map(|d| d.confidence).collect();
    assert!(variance(&confidences) > 0.01); // Should vary with market conditions
}

#[tokio::test]
async fn test_error_handling_and_recovery() {
    let config = QARConfig::default();
    let mut qar = QuantumAgenticReasoning::new(config).unwrap();
    
    // Test with invalid market data
    let invalid_data = MarketData {
        symbol: "".to_string(), // Empty symbol
        current_price: -1000.0, // Negative price
        possible_outcomes: vec![], // Empty outcomes
        buy_probabilities: vec![],
        sell_probabilities: vec![],
        hold_probabilities: vec![],
        frame: prospect_theory::FramingContext {
            frame_type: prospect_theory::FrameType::Neutral,
            emphasis: 0.5,
        },
        timestamp: 0,
    };
    
    let invalid_result = qar.make_decision(&invalid_data, None);
    
    // Should either handle gracefully or return appropriate error
    match invalid_result {
        Ok(decision) => {
            // If it handles gracefully, should still be valid
            assert!(decision.confidence >= 0.0 && decision.confidence <= 1.0);
        },
        Err(_) => {
            // Acceptable to return error for invalid input
        }
    }
    
    // Test recovery with valid data after error
    let valid_data = MarketData {
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
    
    let recovery_decision = qar.make_decision(&valid_data, None);
    assert!(recovery_decision.is_ok());
    
    let decision = recovery_decision.unwrap();
    assert!(decision.confidence >= 0.0 && decision.confidence <= 1.0);
}

#[tokio::test]
async fn test_memory_and_state_consistency() {
    let config = QARConfig::default();
    let mut qar = QuantumAgenticReasoning::new(config).unwrap();
    
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
    
    // Make multiple decisions with same data
    let decision1 = qar.make_decision(&market_data, None).unwrap();
    let decision2 = qar.make_decision(&market_data, None).unwrap();
    
    // Due to caching and state consistency, repeated calls should give similar results
    assert!((decision1.confidence - decision2.confidence).abs() < 0.1);
    assert_eq!(decision1.action, decision2.action);
    
    // Performance metrics should track multiple calls
    let metrics = qar.get_performance_metrics();
    assert!(metrics.total_decisions >= 2);
}

// Helper function to calculate variance
fn variance(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let sum_squared_diff = values.iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f64>();
    
    sum_squared_diff / values.len() as f64
}