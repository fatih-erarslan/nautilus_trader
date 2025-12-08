//! Comprehensive unit tests for Kelly Criterion implementation
//! Tests position sizing, risk adjustment, and financial invariants

use talebian_risk_rs::{
    kelly::*, MacchiavelianConfig, MarketData, TalebianRiskError, 
    WhaleDetection, WhaleDirection
};
use approx::assert_relative_eq;

/// Test helper to create sample market data
fn create_sample_market_data() -> MarketData {
    MarketData {
        timestamp: chrono::Utc::now(),
        timestamp_unix: 1640995200,
        price: 50000.0,
        volume: 1000.0,
        bid: 49990.0,
        ask: 50010.0,
        bid_volume: 500.0,
        ask_volume: 500.0,
        volatility: 0.03,
        returns: vec![0.01, 0.015, -0.005, 0.02, 0.008],
        volume_history: vec![800.0, 900.0, 1200.0, 950.0, 1000.0],
    }
}

/// Test helper to create whale detection result
fn create_whale_detection(detected: bool, confidence: f64) -> WhaleDetection {
    WhaleDetection {
        timestamp_unix: 1640995200,
        detected,
        volume_spike: if detected { 3.0 } else { 1.0 },
        direction: if detected { WhaleDirection::Buying } else { WhaleDirection::Neutral },
        confidence,
        whale_size: if detected { 5000.0 } else { 1000.0 },
        impact: if detected { 0.002 } else { 0.0001 },
        is_whale_detected: detected,
        order_book_imbalance: if detected { 0.5 } else { 0.0 },
        price_impact: if detected { 0.002 } else { 0.0001 },
    }
}

#[cfg(test)]
mod kelly_tests {
    use super::*;

    #[test]
    fn test_kelly_engine_creation() {
        let config = MacchiavelianConfig::aggressive_defaults();
        let engine = KellyEngine::new(config.clone());
        
        assert_eq!(engine.config.kelly_fraction, config.kelly_fraction);
        assert_eq!(engine.trade_history.len(), 0);
    }

    #[test]
    fn test_basic_kelly_calculation() {
        let config = MacchiavelianConfig::aggressive_defaults();
        let engine = KellyEngine::new(config.clone());
        
        let market_data = create_sample_market_data();
        let whale_detection = create_whale_detection(false, 0.3);
        let expected_return = 0.02;
        let confidence = 0.8;
        
        let result = engine.calculate_kelly_fraction(&market_data, &whale_detection, expected_return, confidence);
        assert!(result.is_ok());
        
        let calculation = result.unwrap();
        
        // Validate Kelly calculation structure
        assert_eq!(calculation.fraction, config.kelly_fraction);
        assert_eq!(calculation.expected_return, expected_return);
        assert_eq!(calculation.confidence, confidence);
        assert!(calculation.adjusted_fraction <= config.kelly_max_fraction);
        assert!(calculation.risk_adjusted_size <= config.kelly_max_fraction);
        
        // Adjusted fraction should be base fraction scaled by confidence
        let expected_adjusted = (config.kelly_fraction * confidence).min(config.kelly_max_fraction);
        assert_relative_eq!(calculation.adjusted_fraction, expected_adjusted, epsilon = 0.001);
    }

    #[test]
    fn test_kelly_with_high_confidence() {
        let config = MacchiavelianConfig::aggressive_defaults();
        let engine = KellyEngine::new(config.clone());
        
        let market_data = create_sample_market_data();
        let whale_detection = create_whale_detection(true, 0.9);
        let expected_return = 0.03;
        let high_confidence = 0.95;
        
        let calculation = engine.calculate_kelly_fraction(&market_data, &whale_detection, expected_return, high_confidence).unwrap();
        
        // High confidence should result in larger position
        assert!(calculation.adjusted_fraction > config.kelly_fraction * 0.8);
        assert_eq!(calculation.confidence, high_confidence);
        assert!(calculation.adjusted_fraction <= config.kelly_max_fraction);
    }

    #[test]
    fn test_kelly_with_low_confidence() {
        let config = MacchiavelianConfig::aggressive_defaults();
        let engine = KellyEngine::new(config.clone());
        
        let market_data = create_sample_market_data();
        let whale_detection = create_whale_detection(false, 0.2);
        let expected_return = 0.01;
        let low_confidence = 0.3;
        
        let calculation = engine.calculate_kelly_fraction(&market_data, &whale_detection, expected_return, low_confidence).unwrap();
        
        // Low confidence should result in smaller position
        assert!(calculation.adjusted_fraction < config.kelly_fraction * 0.5);
        assert_eq!(calculation.confidence, low_confidence);
        assert!(calculation.adjusted_fraction > 0.0);
    }

    #[test]
    fn test_kelly_fraction_bounds() {
        let config = MacchiavelianConfig::aggressive_defaults();
        let engine = KellyEngine::new(config.clone());
        
        let market_data = create_sample_market_data();
        let whale_detection = create_whale_detection(false, 0.5);
        
        // Test with various confidence levels
        let confidence_levels = vec![0.0, 0.25, 0.5, 0.75, 1.0, 1.5]; // Include invalid > 1.0
        
        for confidence in confidence_levels {
            let calculation = engine.calculate_kelly_fraction(&market_data, &whale_detection, 0.02, confidence).unwrap();
            
            // Kelly fraction should always be within bounds
            assert!(calculation.fraction >= 0.0, \"Kelly fraction must be non-negative\");
            assert!(calculation.fraction <= 1.0, \"Kelly fraction must not exceed 100%\");
            assert!(calculation.adjusted_fraction >= 0.0, \"Adjusted Kelly fraction must be non-negative\");
            assert!(calculation.adjusted_fraction <= config.kelly_max_fraction, \"Adjusted Kelly fraction must respect maximum\");
            assert!(calculation.risk_adjusted_size >= 0.0, \"Risk adjusted size must be non-negative\");
            assert!(calculation.risk_adjusted_size <= config.kelly_max_fraction, \"Risk adjusted size must respect maximum\");
        }
    }

    #[test]
    fn test_trade_outcome_recording() {
        let config = MacchiavelianConfig::aggressive_defaults();
        let mut engine = KellyEngine::new(config);
        
        // Record some trade outcomes
        assert!(engine.record_trade_outcome(0.02, true, 0.8).is_ok());
        assert!(engine.record_trade_outcome(-0.01, false, 0.5).is_ok());
        assert!(engine.record_trade_outcome(0.015, true, 0.9).is_ok());
        assert!(engine.record_trade_outcome(-0.005, false, 0.3).is_ok());
        
        assert_eq!(engine.trade_history.len(), 4);
        
        // Verify trade outcome structure
        let last_trade = &engine.trade_history[3];
        assert_eq!(last_trade.return_pct, -0.005);
        assert_eq!(last_trade.was_whale_trade, false);
        assert_eq!(last_trade.momentum_score, 0.3);
    }

    #[test]
    fn test_trade_history_bounds() {
        let config = MacchiavelianConfig::aggressive_defaults();
        let mut engine = KellyEngine::new(config);
        
        // Add many trades to test memory management
        for i in 0..1500 {
            let return_pct = (i as f64 / 1000.0) * 0.01; // Varying returns
            let was_whale = i % 3 == 0;
            let momentum = (i as f64 / 1500.0);
            
            engine.record_trade_outcome(return_pct, was_whale, momentum).unwrap();
        }
        
        // Should be bounded to prevent memory issues
        assert!(engine.trade_history.len() <= 1000, \"Trade history should be bounded\");
        
        // Should still function correctly
        let new_result = engine.record_trade_outcome(0.02, true, 0.8);
        assert!(new_result.is_ok());
    }

    #[test]
    fn test_kelly_status() {
        let config = MacchiavelianConfig::aggressive_defaults();
        let mut engine = KellyEngine::new(config.clone());
        
        // Add some trades
        let trades = vec![
            (0.02, true, 0.8),
            (-0.01, false, 0.4),
            (0.015, true, 0.9),
            (0.008, false, 0.6),
            (-0.005, true, 0.3),
        ];
        
        for (return_pct, was_whale, momentum) in trades.iter() {
            engine.record_trade_outcome(*return_pct, *was_whale, *momentum).unwrap();
        }
        
        let status = engine.get_kelly_status();
        
        // Validate status
        assert_eq!(status.total_trades, 5);
        assert_eq!(status.current_fraction, config.kelly_fraction);
        
        // Calculate expected average return
        let expected_avg = trades.iter().map(|(ret, _, _)| ret).sum::<f64>() / trades.len() as f64;
        assert_relative_eq!(status.avg_return, expected_avg, epsilon = 0.001);
    }

    #[test]
    fn test_kelly_status_empty_history() {
        let config = MacchiavelianConfig::aggressive_defaults();
        let engine = KellyEngine::new(config.clone());
        
        let status = engine.get_kelly_status();
        
        // Should handle empty history gracefully
        assert_eq!(status.total_trades, 0);
        assert_eq!(status.avg_return, 0.0);
        assert_eq!(status.current_fraction, config.kelly_fraction);
    }

    #[test]
    fn test_aggressive_vs_conservative_kelly() {
        let aggressive_config = MacchiavelianConfig::aggressive_defaults();
        let conservative_config = MacchiavelianConfig::conservative_baseline();
        
        let aggressive_engine = KellyEngine::new(aggressive_config.clone());
        let conservative_engine = KellyEngine::new(conservative_config.clone());
        
        let market_data = create_sample_market_data();
        let whale_detection = create_whale_detection(true, 0.8);
        let expected_return = 0.02;
        let confidence = 0.8;
        
        let aggressive_calc = aggressive_engine.calculate_kelly_fraction(&market_data, &whale_detection, expected_return, confidence).unwrap();
        let conservative_calc = conservative_engine.calculate_kelly_fraction(&market_data, &whale_detection, expected_return, confidence).unwrap();
        
        // Aggressive should recommend larger positions
        assert!(aggressive_calc.fraction > conservative_calc.fraction);
        assert!(aggressive_calc.adjusted_fraction >= conservative_calc.adjusted_fraction);
        assert!(aggressive_calc.risk_adjusted_size >= conservative_calc.risk_adjusted_size);
    }

    #[test]
    fn test_edge_case_zero_confidence() {
        let config = MacchiavelianConfig::aggressive_defaults();
        let engine = KellyEngine::new(config);
        
        let market_data = create_sample_market_data();
        let whale_detection = create_whale_detection(false, 0.0);
        let expected_return = 0.02;
        let zero_confidence = 0.0;
        
        let calculation = engine.calculate_kelly_fraction(&market_data, &whale_detection, expected_return, zero_confidence).unwrap();
        
        // Zero confidence should result in zero position
        assert_eq!(calculation.adjusted_fraction, 0.0);
        assert_eq!(calculation.risk_adjusted_size, 0.0);
        assert_eq!(calculation.confidence, 0.0);
    }

    #[test]
    fn test_edge_case_negative_expected_return() {
        let config = MacchiavelianConfig::aggressive_defaults();
        let engine = KellyEngine::new(config);
        
        let market_data = create_sample_market_data();
        let whale_detection = create_whale_detection(false, 0.5);
        let negative_return = -0.02;
        let confidence = 0.8;
        
        let calculation = engine.calculate_kelly_fraction(&market_data, &whale_detection, negative_return, confidence).unwrap();
        
        // Should still calculate but with negative expected return
        assert_eq!(calculation.expected_return, negative_return);
        assert!(calculation.adjusted_fraction >= 0.0); // Should not go negative
        assert!(calculation.risk_adjusted_size >= 0.0);
    }

    #[test]
    fn test_edge_case_extreme_values() {
        let config = MacchiavelianConfig::aggressive_defaults();
        let engine = KellyEngine::new(config);
        
        let market_data = create_sample_market_data();
        let whale_detection = create_whale_detection(false, 0.5);
        
        // Test with extreme expected returns
        let extreme_cases = vec![
            (f64::INFINITY, 0.5),
            (f64::NEG_INFINITY, 0.5),
            (1000.0, 0.5), // 100,000% return
            (-1000.0, 0.5), // 100,000% loss
            (0.02, f64::INFINITY),
            (0.02, f64::NAN),
        ];
        
        for (expected_return, confidence) in extreme_cases {
            let result = engine.calculate_kelly_fraction(&market_data, &whale_detection, expected_return, confidence);
            
            // Should either handle gracefully or return error
            if let Ok(calculation) = result {
                // If successful, results should be bounded
                assert!(calculation.adjusted_fraction >= 0.0);
                assert!(calculation.adjusted_fraction <= 1.0);
                assert!(calculation.risk_adjusted_size >= 0.0);
                assert!(calculation.risk_adjusted_size <= 1.0);
            }
        }
    }

    #[test]
    fn test_variance_handling() {
        let config = MacchiavelianConfig::aggressive_defaults();
        let engine = KellyEngine::new(config);
        
        let market_data = create_sample_market_data();
        let whale_detection = create_whale_detection(false, 0.5);
        let expected_return = 0.02;
        let confidence = 0.8;
        
        let calculation = engine.calculate_kelly_fraction(&market_data, &whale_detection, expected_return, confidence).unwrap();
        
        // Variance should be set to a reasonable default
        assert!(calculation.variance > 0.0);
        assert!(calculation.variance < 1.0); // Should be reasonable for daily returns
        assert_relative_eq!(calculation.variance, 0.04, epsilon = 0.001); // As per implementation
    }

    #[test]
    fn test_concurrent_safety_simulation() {
        use std::sync::{Arc, Mutex};
        use std::thread;
        
        let config = MacchiavelianConfig::aggressive_defaults();
        let engine = Arc::new(Mutex::new(KellyEngine::new(config)));
        let mut handles = vec![];
        
        // Simulate concurrent trade recording
        for i in 0..10 {
            let engine_clone = Arc::clone(&engine);
            let handle = thread::spawn(move || {
                let return_pct = (i as f64) * 0.001;
                let was_whale = i % 2 == 0;
                let momentum = (i as f64) / 10.0;
                
                let mut engine_guard = engine_clone.lock().unwrap();
                engine_guard.record_trade_outcome(return_pct, was_whale, momentum).unwrap();
            });
            handles.push(handle);
        }
        
        // Wait for all threads
        for handle in handles {
            handle.join().unwrap();
        }
        
        let engine_guard = engine.lock().unwrap();
        assert_eq!(engine_guard.trade_history.len(), 10);
    }

    #[test]
    fn test_performance_tracking_accuracy() {
        let config = MacchiavelianConfig::aggressive_defaults();
        let mut engine = KellyEngine::new(config);
        
        // Add trades with known statistics
        let known_trades = vec![
            (0.10, true, 0.9),   // 10% return, whale trade
            (-0.05, false, 0.2), // -5% return, no whale
            (0.08, true, 0.8),   // 8% return, whale trade
            (0.02, false, 0.5),  // 2% return, no whale
            (-0.03, true, 0.3),  // -3% return, whale trade
        ];
        
        for (return_pct, was_whale, momentum) in known_trades.iter() {
            engine.record_trade_outcome(*return_pct, *was_whale, *momentum).unwrap();
        }
        
        let status = engine.get_kelly_status();
        
        // Verify accurate calculation
        let expected_avg = known_trades.iter().map(|(ret, _, _)| ret).sum::<f64>() / known_trades.len() as f64;
        assert_relative_eq!(status.avg_return, expected_avg, epsilon = 0.0001);
        assert_eq!(status.total_trades, known_trades.len());
        
        // Count whale trades
        let whale_trades = known_trades.iter().filter(|(_, whale, _)| *whale).count();
        assert_eq!(whale_trades, 3); // Verify our test data
    }

    #[test]
    fn test_kelly_calculation_consistency() {
        let config = MacchiavelianConfig::aggressive_defaults();
        let engine = KellyEngine::new(config);
        
        let market_data = create_sample_market_data();
        let whale_detection = create_whale_detection(true, 0.7);
        let expected_return = 0.025;
        let confidence = 0.85;
        
        // Multiple calculations with same input should be identical
        let calc1 = engine.calculate_kelly_fraction(&market_data, &whale_detection, expected_return, confidence).unwrap();
        let calc2 = engine.calculate_kelly_fraction(&market_data, &whale_detection, expected_return, confidence).unwrap();
        
        assert_relative_eq!(calc1.fraction, calc2.fraction, epsilon = f64::EPSILON);
        assert_relative_eq!(calc1.adjusted_fraction, calc2.adjusted_fraction, epsilon = f64::EPSILON);
        assert_relative_eq!(calc1.risk_adjusted_size, calc2.risk_adjusted_size, epsilon = f64::EPSILON);
        assert_relative_eq!(calc1.confidence, calc2.confidence, epsilon = f64::EPSILON);
        assert_relative_eq!(calc1.expected_return, calc2.expected_return, epsilon = f64::EPSILON);
        assert_relative_eq!(calc1.variance, calc2.variance, epsilon = f64::EPSILON);
    }

    #[test]
    fn test_financial_invariants() {
        let config = MacchiavelianConfig::aggressive_defaults();
        let engine = KellyEngine::new(config.clone());
        
        let market_data = create_sample_market_data();
        
        // Test across various scenarios
        let scenarios = vec![
            (true, 0.9, 0.03, 0.95),   // High confidence whale trade
            (false, 0.1, 0.01, 0.3),   // Low confidence normal trade
            (true, 0.8, 0.02, 0.8),    // Medium confidence whale trade
            (false, 0.5, 0.015, 0.6),  // Medium confidence normal trade
        ];
        
        for (whale_detected, whale_conf, exp_return, confidence) in scenarios {
            let whale_detection = create_whale_detection(whale_detected, whale_conf);
            let calculation = engine.calculate_kelly_fraction(&market_data, &whale_detection, exp_return, confidence).unwrap();
            
            // Financial invariants
            assert!(calculation.fraction >= 0.0, \"Kelly fraction must be non-negative\");
            assert!(calculation.fraction <= 1.0, \"Kelly fraction must not exceed 100%\");
            assert!(calculation.adjusted_fraction >= 0.0, \"Adjusted Kelly fraction must be non-negative\");
            assert!(calculation.adjusted_fraction <= config.kelly_max_fraction, \"Adjusted Kelly fraction must respect maximum\");
            assert!(calculation.risk_adjusted_size >= 0.0, \"Risk adjusted size must be non-negative\");
            assert!(calculation.risk_adjusted_size <= config.kelly_max_fraction, \"Risk adjusted size must respect maximum\");
            assert!(calculation.confidence >= 0.0, \"Confidence must be non-negative\");
            assert!(calculation.confidence <= 1.0 || calculation.confidence.is_infinite(), \"Confidence should be <= 1.0 or handled specially\");
            assert!(calculation.variance >= 0.0, \"Variance must be non-negative\");
            
            // Adjusted fraction should be base fraction scaled by confidence, capped at max
            let expected_adjusted = (calculation.fraction * confidence).min(config.kelly_max_fraction);
            if confidence.is_finite() && confidence <= 1.0 {
                assert_relative_eq!(calculation.adjusted_fraction, expected_adjusted, epsilon = 0.001);
            }
            
            // Risk adjusted size should equal adjusted fraction
            assert_relative_eq!(calculation.risk_adjusted_size, calculation.adjusted_fraction, epsilon = f64::EPSILON);
        }
    }

    #[test]
    fn test_whale_impact_on_kelly() {
        let config = MacchiavelianConfig::aggressive_defaults();
        let engine = KellyEngine::new(config);
        
        let market_data = create_sample_market_data();
        let expected_return = 0.02;
        let confidence = 0.8;
        
        // Compare whale vs no whale scenarios
        let no_whale = create_whale_detection(false, 0.2);
        let whale = create_whale_detection(true, 0.9);
        
        let no_whale_calc = engine.calculate_kelly_fraction(&market_data, &no_whale, expected_return, confidence).unwrap();
        let whale_calc = engine.calculate_kelly_fraction(&market_data, &whale, expected_return, confidence).unwrap();
        
        // Whale detection itself doesn't directly affect Kelly in this implementation
        // (that's handled at the risk engine level), but high confidence should increase position
        assert_eq!(no_whale_calc.fraction, whale_calc.fraction); // Base fraction same
        
        // But confidence affects the adjustment
        if whale.confidence > no_whale.confidence {
            // If whale has higher confidence, it doesn't directly affect Kelly calculation
            // The whale multiplier is applied at the risk engine level
            assert_eq!(whale_calc.adjusted_fraction, whale_calc.fraction * confidence);
        }
    }

    #[test]
    fn test_trade_outcome_extremes() {
        let config = MacchiavelianConfig::aggressive_defaults();
        let mut engine = KellyEngine::new(config);
        
        // Test extreme trade outcomes
        let extreme_outcomes = vec![
            (f64::INFINITY, true, 0.5),
            (f64::NEG_INFINITY, false, 0.5),
            (1000.0, true, 1.0),    // 100,000% return
            (-1.0, false, 0.0),     // 100% loss
            (0.0, true, f64::INFINITY),
            (0.01, false, f64::NAN),
        ];
        
        for (return_pct, was_whale, momentum) in extreme_outcomes {
            let result = engine.record_trade_outcome(return_pct, was_whale, momentum);
            
            // Should handle gracefully
            assert!(result.is_ok(), \"Should handle extreme trade outcome: {}, {}, {}\", return_pct, was_whale, momentum);
        }
        
        // Engine should still function
        let status = engine.get_kelly_status();
        assert!(status.total_trades > 0);
        
        // Average return should be finite or handle appropriately
        if status.avg_return.is_finite() {
            assert!(status.avg_return.abs() < f64::INFINITY);
        }
    }
}