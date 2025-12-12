//! Integration tests for Quantum Agentic Reasoning crate
//!
//! This module contains comprehensive integration tests following TDD methodology
//! to ensure all quantum components work together correctly.

use quantum_agentic_reasoning::*;
use tokio::test;
use std::collections::HashMap;
use approx::assert_relative_eq;

#[test]
async fn test_end_to_end_quantum_decision_workflow() {
    // Test the complete quantum decision-making workflow
    let mut engine = QuantumDecisionEngine::new(3, 0.3);
    
    // Create comprehensive market factors
    let mut factors = FactorMap::new();
    factors.set(StandardFactors::Momentum, 0.8).unwrap();
    factors.set(StandardFactors::Volume, 0.7).unwrap();
    factors.set(StandardFactors::Volatility, 0.3).unwrap();
    factors.set(StandardFactors::Trend, 0.6).unwrap();
    factors.set(StandardFactors::Support, 0.5).unwrap();
    factors.set(StandardFactors::Resistance, 0.4).unwrap();
    factors.set(StandardFactors::Sentiment, 0.7).unwrap();
    factors.set(StandardFactors::Liquidity, 0.8).unwrap();

    let context = MarketContext {
        symbol: "BTC/USDT".to_string(),
        timeframe: "1h".to_string(),
        current_price: 45000.0,
        current_time: chrono::Utc::now(),
        metadata: HashMap::new(),
    };

    // Make decision
    let decision = engine.make_decision(&factors, &context).await;
    assert!(decision.is_ok(), "Decision making should succeed");

    let decision = decision.unwrap();
    
    // Validate decision structure
    assert!(!decision.id.is_empty(), "Decision should have unique ID");
    assert!(decision.confidence >= 0.0 && decision.confidence <= 1.0, "Confidence should be normalized");
    assert!(decision.execution_time_ms > 0.0, "Should record execution time");
    assert!(decision.quantum_advantage, "Should use quantum advantage");
    assert!(!decision.reasoning.is_empty(), "Should provide reasoning");
    
    // Test feedback mechanism
    let outcome = DecisionOutcome::Success {
        profit: 150.0,
        duration_ms: 3600000, // 1 hour
    };
    
    let feedback_result = engine.update_with_feedback(&decision.id, outcome).await;
    assert!(feedback_result.is_ok(), "Feedback should be processed successfully");
    
    // Verify learning occurred
    let patterns = engine.get_learned_patterns().await;
    assert!(!patterns.is_empty(), "Should learn from successful decisions");
    
    let metrics = engine.get_detailed_metrics().await;
    assert_eq!(metrics.total_decisions, 1, "Should track decision count");
    assert_eq!(metrics.successful_decisions, 1, "Should track successful decisions");
}

#[test]
async fn test_quantum_vs_classical_fallback() {
    let engine = QuantumDecisionEngine::new(3, 0.3);
    
    let mut factors = FactorMap::new();
    factors.set(StandardFactors::Momentum, 0.6).unwrap();
    factors.set(StandardFactors::Volume, 0.4).unwrap();
    
    let context = MarketContext {
        symbol: "ETH/USDT".to_string(),
        timeframe: "5m".to_string(),
        current_price: 3200.0,
        current_time: chrono::Utc::now(),
        metadata: HashMap::new(),
    };

    // Test quantum execution
    let quantum_decision = engine.make_decision(&factors, &context).await.unwrap();
    assert!(quantum_decision.quantum_advantage, "Should use quantum algorithms");
    
    // Verify that the decision contains quantum-specific metadata
    assert!(quantum_decision.metadata.contains_key("market_phase"), "Should include market phase analysis");
    assert!(quantum_decision.metadata.contains_key("pattern_matches"), "Should include pattern analysis");
    assert!(quantum_decision.metadata.contains_key("information_gain"), "Should include information gain");
}

#[test]
async fn test_market_analyzer_integration() {
    let analyzer = QuantumMarketAnalyzer::new(3);
    
    let mut factors = FactorMap::new();
    factors.set(StandardFactors::Momentum, 0.9).unwrap();
    factors.set(StandardFactors::Volume, 0.8).unwrap();
    factors.set(StandardFactors::Volatility, 0.2).unwrap();
    factors.set(StandardFactors::Trend, 0.85).unwrap();

    // Test regime analysis
    let regime = analyzer.analyze_regime(&factors).await;
    assert!(regime.is_ok(), "Regime analysis should succeed");
    
    let regime = regime.unwrap();
    assert!(matches!(regime.phase, MarketPhase::Growth | MarketPhase::Decline | MarketPhase::Sideways | MarketPhase::Uncertain));
    assert!(regime.confidence >= 0.0 && regime.confidence <= 1.0);
    assert!(regime.strength >= 0.0 && regime.strength <= 1.0);
    assert!(!regime.spectral_power.is_empty(), "Should provide spectral analysis");

    // Test pattern detection
    let patterns = analyzer.detect_patterns(&factors).await;
    assert!(patterns.is_ok(), "Pattern detection should succeed");
    
    // Test direction prediction
    let prediction = analyzer.predict_direction(&factors).await;
    assert!(prediction.is_ok(), "Direction prediction should succeed");
    
    let prediction = prediction.unwrap();
    assert!(prediction.direction >= -1.0 && prediction.direction <= 1.0, "Direction should be normalized");
    assert!(prediction.confidence >= 0.0 && prediction.confidence <= 1.0);
    assert!(prediction.time_horizon_ms > 0, "Should provide time horizon");

    // Test volatility calculation
    let volatility = analyzer.calculate_volatility(&factors).await;
    assert!(volatility.is_ok(), "Volatility calculation should succeed");
    
    let volatility = volatility.unwrap();
    assert!(volatility >= 0.0 && volatility <= 1.0, "Volatility should be normalized");
}

#[test]
async fn test_quantum_circuits_integration() {
    // Test QFT Circuit
    let qft_circuit = QftCircuit::new(3);
    let params = CircuitParams::new(vec![0.5, 0.3, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1], 3);
    let context = ExecutionContext::default();
    
    // Validate parameters first
    assert!(qft_circuit.validate_parameters(&params).is_ok(), "QFT parameters should be valid");
    
    let qft_result = qft_circuit.execute(&params, &context).await;
    assert!(qft_result.is_ok(), "QFT execution should succeed");
    
    let result = qft_result.unwrap();
    assert!(!result.expectation_values.is_empty(), "Should return spectral data");
    assert!(result.execution_time_ms > 0.0, "Should measure execution time");

    // Test Decision Optimization Circuit
    let opt_circuit = DecisionOptimizationCircuit::new(3, 3, 0.4);
    let decision_result = opt_circuit.execute(&params, &context).await;
    assert!(decision_result.is_ok(), "Decision optimization should succeed");
    
    let result = decision_result.unwrap();
    assert!(!result.expectation_values.is_empty(), "Should return optimization weights");

    // Test Pattern Recognition Circuit
    let mut pattern_circuit = PatternRecognitionCircuit::new(3, 16);
    
    // Add reference patterns
    let bull_pattern = vec![0.8, 0.6, 0.4, 0.2, 0.1, 0.1, 0.1, 0.1];
    let bear_pattern = vec![0.2, 0.4, 0.6, 0.8, 0.1, 0.1, 0.1, 0.1];
    
    assert!(pattern_circuit.add_reference_pattern(bull_pattern).is_ok());
    assert!(pattern_circuit.add_reference_pattern(bear_pattern).is_ok());
    
    let pattern_result = pattern_circuit.execute(&params, &context).await;
    assert!(pattern_result.is_ok(), "Pattern recognition should succeed");
    
    let result = pattern_result.unwrap();
    assert_eq!(result.expectation_values.len(), 2, "Should return similarity scores for both patterns");
}

#[test]
async fn test_backend_fallback_mechanism() {
    let config = BackendConfig::new(BackendType::Simulator, 4);
    let backend = SimulatorBackend::new(config);
    
    // Test quantum availability
    assert!(backend.is_quantum_available().await, "Simulator should be available");
    
    // Test capabilities
    let capabilities = backend.get_capabilities().await;
    assert_eq!(capabilities.max_qubits, 4);
    assert!(!capabilities.supported_gates.is_empty());
    
    // Test circuit execution
    let qft_circuit = QftCircuit::new(3);
    let params = CircuitParams::new(vec![0.5; 8], 3);
    
    let result = backend.execute_quantum(&qft_circuit, &params).await;
    assert!(result.is_ok(), "Backend execution should succeed");
    
    // Test metrics collection
    let metrics = backend.get_metrics().await;
    assert!(metrics.quantum_time_ms >= 0.0, "Should track quantum execution time");
}

#[test]
async fn test_parallel_circuit_execution() {
    let qft_circuit = QftCircuit::new(3);
    let opt_circuit = DecisionOptimizationCircuit::new(3, 2, 0.3);
    
    let params = CircuitParams::new(vec![0.6, 0.4, 0.3, 0.2, 0.1, 0.1, 0.1, 0.1], 3);
    let context = ExecutionContext::default();
    
    // Execute circuits in parallel
    let start_time = std::time::Instant::now();
    
    let (qft_result, opt_result) = tokio::join!(
        qft_circuit.execute(&params, &context),
        opt_circuit.execute(&params, &context)
    );
    
    let parallel_time = start_time.elapsed();
    
    assert!(qft_result.is_ok(), "QFT should succeed in parallel");
    assert!(opt_result.is_ok(), "Optimization should succeed in parallel");
    
    // Verify parallel execution is faster than sequential
    let start_time = std::time::Instant::now();
    
    let _qft_result = qft_circuit.execute(&params, &context).await;
    let _opt_result = opt_circuit.execute(&params, &context).await;
    
    let sequential_time = start_time.elapsed();
    
    assert!(parallel_time < sequential_time, "Parallel execution should be faster");
}

#[test]
async fn test_error_handling_and_recovery() {
    let engine = QuantumDecisionEngine::new(3, 0.8); // High threshold
    
    // Test with invalid factors
    let mut invalid_factors = FactorMap::new();
    // Don't set any factors - should still handle gracefully
    
    let context = MarketContext {
        symbol: "INVALID/PAIR".to_string(),
        timeframe: "invalid".to_string(),
        current_price: -100.0, // Invalid price
        current_time: chrono::Utc::now(),
        metadata: HashMap::new(),
    };
    
    let decision = engine.make_decision(&invalid_factors, &context).await;
    assert!(decision.is_ok(), "Should handle invalid inputs gracefully");
    
    let decision = decision.unwrap();
    // With empty factors and high threshold, should default to Hold
    assert_eq!(decision.decision_type, DecisionType::Hold, "Should default to Hold for low confidence");
    assert!(decision.confidence < 0.8, "Should have low confidence");
    
    // Test circuit parameter validation
    let qft_circuit = QftCircuit::new(3);
    let invalid_params = CircuitParams::new(vec![0.5], 2); // Wrong number of qubits
    
    let validation_result = qft_circuit.validate_parameters(&invalid_params);
    assert!(validation_result.is_err(), "Should reject invalid parameters");
}

#[test]
async fn test_memory_and_learning() {
    let mut engine = QuantumDecisionEngine::new(3, 0.3);
    
    let mut factors = FactorMap::new();
    factors.set(StandardFactors::Momentum, 0.7).unwrap();
    factors.set(StandardFactors::Volume, 0.6).unwrap();
    
    let context = MarketContext {
        symbol: "ADA/USDT".to_string(),
        timeframe: "15m".to_string(),
        current_price: 1.5,
        current_time: chrono::Utc::now(),
        metadata: HashMap::new(),
    };
    
    // Make multiple decisions
    let decision1 = engine.make_decision(&factors, &context).await.unwrap();
    let decision2 = engine.make_decision(&factors, &context).await.unwrap();
    let decision3 = engine.make_decision(&factors, &context).await.unwrap();
    
    // Check history tracking
    let history = engine.get_decision_history().await;
    assert_eq!(history.len(), 3, "Should track decision history");
    
    // Provide feedback for learning
    let outcomes = vec![
        DecisionOutcome::Success { profit: 50.0, duration_ms: 1800000 },
        DecisionOutcome::Failure { loss: 20.0, duration_ms: 900000 },
        DecisionOutcome::Success { profit: 75.0, duration_ms: 2700000 },
    ];
    
    for (decision, outcome) in [decision1, decision2, decision3].iter().zip(outcomes.iter()) {
        let result = engine.update_with_feedback(&decision.id, outcome.clone()).await;
        assert!(result.is_ok(), "Should process feedback successfully");
    }
    
    // Verify learning metrics
    let metrics = engine.get_detailed_metrics().await;
    assert_eq!(metrics.total_decisions, 3, "Should track total decisions");
    assert_eq!(metrics.successful_decisions, 2, "Should track successful decisions");
    
    let patterns = engine.get_learned_patterns().await;
    assert_eq!(patterns.len(), 2, "Should learn from successful patterns");
}

#[test]
async fn test_performance_monitoring() {
    let engine = QuantumDecisionEngine::new(3, 0.4);
    let analyzer = QuantumMarketAnalyzer::new(3);
    
    let mut factors = FactorMap::new();
    factors.set(StandardFactors::Momentum, 0.8).unwrap();
    factors.set(StandardFactors::Volume, 0.9).unwrap();
    factors.set(StandardFactors::Volatility, 0.3).unwrap();
    
    let context = MarketContext {
        symbol: "DOT/USDT".to_string(),
        timeframe: "30m".to_string(),
        current_price: 25.0,
        current_time: chrono::Utc::now(),
        metadata: HashMap::new(),
    };
    
    // Measure performance metrics
    let start_time = std::time::Instant::now();
    
    let (decision_result, regime_result, patterns_result) = tokio::join!(
        engine.make_decision(&factors, &context),
        analyzer.analyze_regime(&factors),
        analyzer.detect_patterns(&factors)
    );
    
    let total_time = start_time.elapsed();
    
    assert!(decision_result.is_ok(), "Decision should succeed");
    assert!(regime_result.is_ok(), "Regime analysis should succeed");
    assert!(patterns_result.is_ok(), "Pattern detection should succeed");
    
    let decision = decision_result.unwrap();
    assert!(decision.execution_time_ms > 0.0, "Should measure execution time");
    assert!(total_time.as_millis() as f64 >= decision.execution_time_ms, "Measured time should be consistent");
    
    // Performance should be reasonable (under 10 seconds for this test)
    assert!(total_time.as_secs() < 10, "Should complete within reasonable time");
}

#[test]
async fn test_configuration_and_thresholds() {
    // Test different confidence thresholds
    let mut low_threshold_engine = QuantumDecisionEngine::new(3, 0.1);
    let mut high_threshold_engine = QuantumDecisionEngine::new(3, 0.9);
    
    let mut factors = FactorMap::new();
    factors.set(StandardFactors::Momentum, 0.5).unwrap(); // Moderate signal
    
    let context = MarketContext {
        symbol: "LINK/USDT".to_string(),
        timeframe: "1h".to_string(),
        current_price: 15.0,
        current_time: chrono::Utc::now(),
        metadata: HashMap::new(),
    };
    
    let low_decision = low_threshold_engine.make_decision(&factors, &context).await.unwrap();
    let high_decision = high_threshold_engine.make_decision(&factors, &context).await.unwrap();
    
    // Low threshold engine should be more likely to take action
    // High threshold engine should be more conservative (likely Hold)
    assert_eq!(high_decision.decision_type, DecisionType::Hold, "High threshold should default to Hold");
    
    // Test threshold modification
    assert_eq!(low_threshold_engine.confidence_threshold(), 0.1);
    low_threshold_engine.set_confidence_threshold(0.5);
    assert_eq!(low_threshold_engine.confidence_threshold(), 0.5);
}

#[test]
async fn test_market_phase_transitions() {
    let analyzer = QuantumMarketAnalyzer::new(3);
    
    // Test different market conditions
    let scenarios = vec![
        ("Bull Market", vec![
            (StandardFactors::Momentum, 0.9),
            (StandardFactors::Volume, 0.8),
            (StandardFactors::Trend, 0.85),
            (StandardFactors::Sentiment, 0.8),
        ]),
        ("Bear Market", vec![
            (StandardFactors::Momentum, 0.1),
            (StandardFactors::Volume, 0.7),
            (StandardFactors::Trend, 0.15),
            (StandardFactors::Sentiment, 0.2),
        ]),
        ("Sideways Market", vec![
            (StandardFactors::Momentum, 0.5),
            (StandardFactors::Volume, 0.4),
            (StandardFactors::Trend, 0.5),
            (StandardFactors::Volatility, 0.2),
        ]),
        ("High Volatility", vec![
            (StandardFactors::Volatility, 0.9),
            (StandardFactors::Volume, 0.8),
            (StandardFactors::Momentum, 0.6),
        ]),
    ];
    
    for (scenario_name, factor_values) in scenarios {
        let mut factors = FactorMap::new();
        for (factor, value) in factor_values {
            factors.set(factor, value).unwrap();
        }
        
        let regime = analyzer.analyze_regime(&factors).await.unwrap();
        let prediction = analyzer.predict_direction(&factors).await.unwrap();
        let volatility = analyzer.calculate_volatility(&factors).await.unwrap();
        
        println!("Scenario: {}", scenario_name);
        println!("  Phase: {:?}, Confidence: {:.2}", regime.phase, regime.confidence);
        println!("  Direction: {:.2}, Prediction Confidence: {:.2}", prediction.direction, prediction.confidence);
        println!("  Volatility: {:.2}", volatility);
        
        // All analyses should complete successfully
        assert!(regime.confidence >= 0.0 && regime.confidence <= 1.0);
        assert!(prediction.direction >= -1.0 && prediction.direction <= 1.0);
        assert!(volatility >= 0.0 && volatility <= 1.0);
    }
}