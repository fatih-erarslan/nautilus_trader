//! Property-based tests for QAR system
//! 
//! Tests mathematical properties and invariants that should hold across
//! all possible inputs using property-based testing (QuickCheck/Proptest style):
//! - Decision confidence bounds [0,1]
//! - Probability conservation in Prospect Theory
//! - Portfolio weight normalization in Hedge algorithms  
//! - Behavioral factor mathematical consistency
//! - Performance constraint adherence

use quantum_agentic_reasoning::*;
use prospect_theory::*;
use proptest::prelude::*;

// Property: All QAR decisions must have confidence in [0,1]
proptest! {
    #[test]
    fn prop_decision_confidence_bounds(
        price in 0.01f64..1000000.0,
        outcomes in prop::collection::vec(0.01f64..1000000.0, 2..10),
        buy_probs in prop::collection::vec(0.0f64..1.0, 2..10),
        sell_probs in prop::collection::vec(0.0f64..1.0, 2..10),
        hold_probs in prop::collection::vec(0.0f64..1.0, 2..10),
        emphasis in 0.0f64..1.0,
    ) {
        // Ensure probability vectors have same length as outcomes
        let len = outcomes.len().min(buy_probs.len()).min(sell_probs.len()).min(hold_probs.len());
        let outcomes = outcomes[..len].to_vec();
        let buy_probs = buy_probs[..len].to_vec();
        let sell_probs = sell_probs[..len].to_vec();
        let hold_probs = hold_probs[..len].to_vec();
        
        let market_data = MarketData {
            symbol: "TEST/USDT".to_string(),
            current_price: price,
            possible_outcomes: outcomes,
            buy_probabilities: buy_probs,
            sell_probabilities: sell_probs,
            hold_probabilities: hold_probs,
            frame: FramingContext {
                frame_type: FrameType::Neutral,
                emphasis,
            },
            timestamp: 1640995200000,
        };
        
        if let Ok(mut qar) = QuantumAgenticReasoning::new(QARConfig::default()) {
            if let Ok(decision) = qar.make_decision(&market_data, None) {
                prop_assert!(decision.confidence >= 0.0);
                prop_assert!(decision.confidence <= 1.0);
                prop_assert!(decision.prospect_value.is_finite());
            }
        }
    }
}

// Property: Prospect Theory value function should be monotonic in gains/losses
proptest! {
    #[test]
    fn prop_prospect_theory_value_function_monotonic(
        gain1 in 0.0f64..1000.0,
        gain2 in 0.0f64..1000.0,
        loss1 in -1000.0f64..0.0,
        loss2 in -1000.0f64..0.0,
    ) {
        let config = QuantumProspectTheoryConfig::default();
        if let Ok(pt) = QuantumProspectTheory::new(config) {
            let reference = 0.0;
            
            if let (Ok(v1), Ok(v2)) = (
                pt.evaluate_value_function(gain1, reference),
                pt.evaluate_value_function(gain2, reference)
            ) {
                // For gains: larger gains should have larger (or equal) values
                if gain1 >= gain2 {
                    prop_assert!(v1 >= v2);
                }
            }
            
            if let (Ok(v1), Ok(v2)) = (
                pt.evaluate_value_function(loss1, reference),
                pt.evaluate_value_function(loss2, reference)
            ) {
                // For losses: larger losses should have smaller (more negative) values
                if loss1 <= loss2 { // loss1 is more negative
                    prop_assert!(v1 <= v2);
                }
            }
        }
    }
}

// Property: Probability weighting functions should preserve bounds [0,1]
proptest! {
    #[test]
    fn prop_probability_weighting_bounds(
        prob in 0.0f64..1.0,
    ) {
        let config = QuantumProspectTheoryConfig::default();
        if let Ok(pt) = QuantumProspectTheory::new(config) {
            if let Ok(tk_weight) = pt.evaluate_tversky_kahneman_weighting(prob) {
                prop_assert!(tk_weight >= 0.0);
                prop_assert!(tk_weight <= 1.0);
            }
            
            if let Ok(prelec_weight) = pt.evaluate_prelec_weighting(prob) {
                prop_assert!(prelec_weight >= 0.0);
                prop_assert!(prelec_weight <= 1.0);
            }
        }
    }
}

// Property: Portfolio weights in Hedge algorithms should sum to ~1.0
proptest! {
    #[test]
    fn prop_hedge_portfolio_weights_sum(
        price in 0.01f64..1000000.0,
        outcomes in prop::collection::vec(0.01f64..1000000.0, 2..8),
        buy_probs in prop::collection::vec(0.0f64..1.0, 2..8),
        sell_probs in prop::collection::vec(0.0f64..1.0, 2..8),
        hold_probs in prop::collection::vec(0.0f64..1.0, 2..8),
    ) {
        let len = outcomes.len().min(buy_probs.len()).min(sell_probs.len()).min(hold_probs.len());
        let outcomes = outcomes[..len].to_vec();
        let buy_probs = buy_probs[..len].to_vec();
        let sell_probs = sell_probs[..len].to_vec();
        let hold_probs = hold_probs[..len].to_vec();
        
        let market_data = MarketData {
            symbol: "TEST/USDT".to_string(),
            current_price: price,
            possible_outcomes: outcomes,
            buy_probabilities: buy_probs,
            sell_probabilities: sell_probs,
            hold_probabilities: hold_probs,
            frame: FramingContext {
                frame_type: FrameType::Neutral,
                emphasis: 0.5,
            },
            timestamp: 1640995200000,
        };
        
        let config = hedge_integration::HedgeConfig::default();
        if let Ok(mut engine) = hedge_integration::QuantumHedgeEngine::new(config) {
            if let Ok(result) = engine.optimize_portfolio(&market_data, None).await {
                let weight_sum: f64 = result.portfolio_weights.iter().map(|w| w.weight).sum();
                prop_assert!((weight_sum - 1.0).abs() < 0.1); // Allow some tolerance
                
                // All individual weights should be non-negative
                for weight in &result.portfolio_weights {
                    prop_assert!(weight.weight >= 0.0);
                    prop_assert!(weight.weight <= 1.0);
                }
            }
        }
    }
}

// Property: LMSR predictions should maintain confidence bounds
proptest! {
    #[test]
    fn prop_lmsr_confidence_bounds(
        price in 0.01f64..1000000.0,
        outcomes in prop::collection::vec(0.01f64..1000000.0, 2..8),
        buy_probs in prop::collection::vec(0.01f64..0.99, 2..8), // Avoid extreme probabilities
        sell_probs in prop::collection::vec(0.01f64..0.99, 2..8),
        hold_probs in prop::collection::vec(0.01f64..0.99, 2..8),
    ) {
        let len = outcomes.len().min(buy_probs.len()).min(sell_probs.len()).min(hold_probs.len());
        let outcomes = outcomes[..len].to_vec();
        let buy_probs = buy_probs[..len].to_vec();
        let sell_probs = sell_probs[..len].to_vec();
        let hold_probs = hold_probs[..len].to_vec();
        
        let market_data = MarketData {
            symbol: "TEST/USDT".to_string(),
            current_price: price,
            possible_outcomes: outcomes,
            buy_probabilities: buy_probs,
            sell_probabilities: sell_probs,
            hold_probabilities: hold_probs,
            frame: FramingContext {
                frame_type: FrameType::Neutral,
                emphasis: 0.5,
            },
            timestamp: 1640995200000,
        };
        
        let config = lmsr_integration::LMSRConfig::default();
        if let Ok(mut predictor) = lmsr_integration::QuantumLMSRPredictor::new(config) {
            if let Ok(prediction) = predictor.predict(&market_data).await {
                prop_assert!(prediction.confidence >= 0.0);
                prop_assert!(prediction.confidence <= 1.0);
                prop_assert!(prediction.expected_return.is_finite());
                prop_assert!(prediction.risk_metric >= 0.0);
                prop_assert!(prediction.risk_metric <= 1.0);
            }
        }
    }
}

// Property: Loss aversion parameter should amplify losses vs gains
proptest! {
    #[test]
    fn prop_loss_aversion_amplification(
        magnitude in 0.1f64..1000.0,
    ) {
        let config = QuantumProspectTheoryConfig::default();
        if let Ok(pt) = QuantumProspectTheory::new(config) {
            let reference = 0.0;
            let gain = magnitude;
            let loss = -magnitude;
            
            if let (Ok(gain_value), Ok(loss_value)) = (
                pt.evaluate_value_function(gain, reference),
                pt.evaluate_value_function(loss, reference)
            ) {
                // Loss aversion: |v(-x)| > v(x) for same magnitude
                prop_assert!(loss_value.abs() > gain_value);
                
                // The ratio should be approximately the loss aversion parameter (λ ≈ 2.25)
                let ratio = loss_value.abs() / gain_value;
                prop_assert!(ratio >= 1.5); // Conservative lower bound
                prop_assert!(ratio <= 4.0);  // Conservative upper bound
            }
        }
    }
}

// Property: Behavioral factors should be bounded and finite
proptest! {
    #[test]
    fn prop_behavioral_factors_bounds(
        price in 0.01f64..1000000.0,
        entry_price in 0.01f64..1000000.0,
        quantity in -10.0f64..10.0,
    ) {
        let position = Position {
            symbol: "TEST/USDT".to_string(),
            quantity,
            entry_price,
            current_value: price,
            unrealized_pnl: (price - entry_price) * quantity,
        };
        
        let market_data = MarketData {
            symbol: "TEST/USDT".to_string(),
            current_price: price,
            possible_outcomes: vec![price * 1.1, price * 0.9],
            buy_probabilities: vec![0.6, 0.4],
            sell_probabilities: vec![0.4, 0.6],
            hold_probabilities: vec![0.5, 0.5],
            frame: FramingContext {
                frame_type: FrameType::Neutral,
                emphasis: 0.5,
            },
            timestamp: 1640995200000,
        };
        
        let config = QuantumProspectTheoryConfig::default();
        if let Ok(pt) = QuantumProspectTheory::new(config) {
            if let Ok(decision) = pt.make_trading_decision(&market_data, Some(&position)) {
                // Behavioral factors should be finite and bounded
                prop_assert!(decision.behavioral_factors.loss_aversion_impact.is_finite());
                prop_assert!(decision.behavioral_factors.probability_weighting_bias.is_finite());
                prop_assert!(decision.behavioral_factors.mental_accounting_bias.is_finite());
                
                // Reasonable bounds on behavioral factors
                prop_assert!(decision.behavioral_factors.loss_aversion_impact.abs() <= 5.0);
                prop_assert!(decision.behavioral_factors.probability_weighting_bias.abs() <= 2.0);
                prop_assert!(decision.behavioral_factors.mental_accounting_bias.abs() <= 2.0);
                
                // Decision should be valid
                prop_assert!(decision.confidence >= 0.0 && decision.confidence <= 1.0);
                prop_assert!(decision.prospect_value.is_finite());
            }
        }
    }
}

// Property: Framing effects should have expected directional bias
proptest! {
    #[test]
    fn prop_framing_effects_directional(
        price in 0.01f64..1000000.0,
        emphasis in 0.0f64..1.0,
    ) {
        let base_data = |frame_type| MarketData {
            symbol: "TEST/USDT".to_string(),
            current_price: price,
            possible_outcomes: vec![price * 1.05, price * 0.95],
            buy_probabilities: vec![0.6, 0.4],
            sell_probabilities: vec![0.4, 0.6],
            hold_probabilities: vec![0.5, 0.5],
            frame: FramingContext { frame_type, emphasis },
            timestamp: 1640995200000,
        };
        
        let config = QuantumProspectTheoryConfig::default();
        if let Ok(pt) = QuantumProspectTheory::new(config) {
            let gain_data = base_data(FrameType::Gain);
            let loss_data = base_data(FrameType::Loss);
            
            if let (Ok(gain_decision), Ok(loss_decision)) = (
                pt.make_trading_decision(&gain_data, None),
                pt.make_trading_decision(&loss_data, None)
            ) {
                // Both decisions should be valid
                prop_assert!(gain_decision.confidence >= 0.0 && gain_decision.confidence <= 1.0);
                prop_assert!(loss_decision.confidence >= 0.0 && loss_decision.confidence <= 1.0);
                
                // Gain frame should generally lead to different behavior than loss frame
                // (exact direction depends on context, but they should differ)
                if emphasis > 0.1 { // Only check when framing is meaningful
                    prop_assert!(gain_decision.confidence != loss_decision.confidence ||
                                gain_decision.action != loss_decision.action);
                }
            }
        }
    }
}

// Property: Performance constraints should be respected
proptest! {
    #[test]
    fn prop_performance_constraints(
        target_latency_ns in 100u64..10000,
        price in 0.01f64..100000.0,
    ) {
        let mut config = QARConfig::default();
        config.target_latency_ns = target_latency_ns;
        
        let market_data = MarketData {
            symbol: "TEST/USDT".to_string(),
            current_price: price,
            possible_outcomes: vec![price * 1.02, price * 0.98],
            buy_probabilities: vec![0.6, 0.4],
            sell_probabilities: vec![0.4, 0.6],
            hold_probabilities: vec![0.5, 0.5],
            frame: FramingContext {
                frame_type: FrameType::Neutral,
                emphasis: 0.5,
            },
            timestamp: 1640995200000,
        };
        
        if let Ok(mut qar) = QuantumAgenticReasoning::new(config) {
            let start = std::time::Instant::now();
            if let Ok(decision) = qar.make_decision(&market_data, None) {
                let elapsed = start.elapsed();
                
                // Decision should be valid regardless of performance
                prop_assert!(decision.confidence >= 0.0 && decision.confidence <= 1.0);
                prop_assert!(decision.execution_time_ns > 0);
                
                // In test environment, allow generous overhead but check order of magnitude
                prop_assert!(elapsed.as_nanos() < (target_latency_ns * 100) as u128);
            }
        }
    }
}

// Property: Reference point adaptation should be stable and bounded
proptest! {
    #[test]
    fn prop_reference_point_adaptation(
        initial_ref in 0.01f64..100000.0,
        price_sequence in prop::collection::vec(0.01f64..100000.0, 1..10),
        adaptation_rate in 0.01f64..0.5,
    ) {
        let mut config = QuantumProspectTheoryConfig::default();
        config.reference_adaptation_rate = adaptation_rate;
        
        if let Ok(mut pt) = QuantumProspectTheory::new(config) {
            // Set initial reference point
            pt.set_reference_point("TEST/USDT", initial_ref).ok();
            
            for price in price_sequence {
                let market_data = MarketData {
                    symbol: "TEST/USDT".to_string(),
                    current_price: price,
                    possible_outcomes: vec![price * 1.01, price * 0.99],
                    buy_probabilities: vec![0.5, 0.5],
                    sell_probabilities: vec![0.5, 0.5],
                    hold_probabilities: vec![0.5, 0.5],
                    frame: FramingContext {
                        frame_type: FrameType::Neutral,
                        emphasis: 0.5,
                    },
                    timestamp: 1640995200000,
                };
                
                pt.make_trading_decision(&market_data, None).ok();
            }
            
            if let Ok(final_ref) = pt.get_reference_point("TEST/USDT") {
                // Reference point should be finite and positive
                prop_assert!(final_ref > 0.0);
                prop_assert!(final_ref.is_finite());
                
                // Should be influenced by but not equal to latest price
                if let Some(&last_price) = price_sequence.last() {
                    prop_assert!(final_ref != last_price || initial_ref == last_price);
                }
                
                // Should not deviate too far from reasonable bounds
                let min_price = price_sequence.iter().fold(f64::INFINITY, |a, &b| a.min(b)).min(initial_ref);
                let max_price = price_sequence.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)).max(initial_ref);
                
                prop_assert!(final_ref >= min_price * 0.5); // Allow some extrapolation
                prop_assert!(final_ref <= max_price * 2.0);
            }
        }
    }
}

// Property: Quantum enhancement should not violate basic constraints
proptest! {
    #[test]
    fn prop_quantum_enhancement_bounds(
        price in 0.01f64..100000.0,
        confidence_level in 0.1f64..0.9,
    ) {
        let mut quantum_config = QARConfig::default();
        quantum_config.quantum_enabled = true;
        
        let mut classical_config = QARConfig::default();
        classical_config.quantum_enabled = false;
        
        let market_data = MarketData {
            symbol: "TEST/USDT".to_string(),
            current_price: price,
            possible_outcomes: vec![price * 1.1, price * 0.9],
            buy_probabilities: vec![confidence_level, 1.0 - confidence_level],
            sell_probabilities: vec![1.0 - confidence_level, confidence_level],
            hold_probabilities: vec![0.5, 0.5],
            frame: FramingContext {
                frame_type: FrameType::Neutral,
                emphasis: 0.5,
            },
            timestamp: 1640995200000,
        };
        
        if let (Ok(mut quantum_qar), Ok(mut classical_qar)) = (
            QuantumAgenticReasoning::new(quantum_config),
            QuantumAgenticReasoning::new(classical_config)
        ) {
            if let (Ok(quantum_decision), Ok(classical_decision)) = (
                quantum_qar.make_decision(&market_data, None),
                classical_qar.make_decision(&market_data, None)
            ) {
                // Both should produce valid decisions
                prop_assert!(quantum_decision.confidence >= 0.0 && quantum_decision.confidence <= 1.0);
                prop_assert!(classical_decision.confidence >= 0.0 && classical_decision.confidence <= 1.0);
                
                // Quantum advantage should be bounded if present
                if let Some(advantage) = quantum_decision.quantum_advantage {
                    prop_assert!(advantage >= 0.0);
                    prop_assert!(advantage <= 1.0); // 100% max advantage
                }
                
                // Quantum enhancement should not break fundamental properties
                prop_assert!(quantum_decision.prospect_value.is_finite());
                prop_assert!(quantum_decision.execution_time_ns > 0);
            }
        }
    }
}