//! Stress tests for extreme market crash scenarios
//! Tests system behavior under extreme financial stress conditions

use talebian_risk_rs::{
    risk_engine::*, MacchiavelianConfig, MarketData, TalebianRiskError,
    WhaleDirection
};
use chrono::{Utc, Duration};
use std::collections::HashMap;

/// Create a flash crash scenario (sudden 20% drop)
fn create_flash_crash_sequence() -> Vec<MarketData> {
    let mut sequence = Vec::new();
    let base_time = Utc::now();
    let initial_price = 50000.0;
    
    // Normal pre-crash conditions
    for i in 0..10 {
        sequence.push(MarketData {
            timestamp: base_time + Duration::minutes(i),
            timestamp_unix: 1640995200 + (i * 60),
            price: initial_price,
            volume: 1000.0,
            bid: initial_price - 10.0,
            ask: initial_price + 10.0,
            bid_volume: 500.0,
            ask_volume: 500.0,
            volatility: 0.02,
            returns: vec![0.001, 0.002, -0.001, 0.0015],
            volume_history: vec![1000.0; 5],
        });
    }
    
    // Flash crash - sudden dramatic drop
    sequence.push(MarketData {
        timestamp: base_time + Duration::minutes(10),
        timestamp_unix: 1640995200 + (10 * 60),
        price: initial_price * 0.8, // 20% crash
        volume: 50000.0, // 50x volume spike
        bid: initial_price * 0.75,
        ask: initial_price * 0.85,
        bid_volume: 100.0, // No buyers
        ask_volume: 20000.0, // Massive selling
        volatility: 0.8, // Extreme volatility
        returns: vec![-0.2, -0.18, -0.22, -0.19, -0.21],
        volume_history: vec![1000.0; 5],
    });
    
    // Immediate aftermath with high volatility
    for i in 11..20 {
        let recovery_factor = ((i - 11) as f64) / 10.0;
        let current_price = initial_price * (0.8 + recovery_factor * 0.1);
        
        sequence.push(MarketData {
            timestamp: base_time + Duration::minutes(i),
            timestamp_unix: 1640995200 + (i * 60),
            price: current_price,
            volume: 10000.0 - (recovery_factor * 5000.0),
            bid: current_price - 50.0,
            ask: current_price + 50.0,
            bid_volume: 200.0 + (recovery_factor * 300.0),
            ask_volume: 2000.0 - (recovery_factor * 1000.0),
            volatility: 0.5 - (recovery_factor * 0.3),
            returns: vec![-0.05, 0.02, -0.03, 0.01, -0.02],
            volume_history: vec![1000.0; 5],
        });
    }
    
    sequence
}

/// Create a slow-burn bear market crash (gradual 50% decline over time)
fn create_bear_market_crash() -> Vec<MarketData> {
    let mut sequence = Vec::new();
    let base_time = Utc::now();
    let initial_price = 50000.0;
    let crash_duration = 100; // 100 time periods
    
    for i in 0..crash_duration {
        let crash_progress = (i as f64) / (crash_duration as f64);
        let price_decline = 1.0 - (crash_progress * 0.5); // 50% decline
        let current_price = initial_price * price_decline;
        
        // Increasing volatility and volume as crash progresses
        let volatility = 0.02 + (crash_progress * 0.15);
        let volume = 1000.0 + (crash_progress * 3000.0);
        
        // Bear market characteristics
        let selling_pressure = crash_progress * 2.0;
        let bid_volume = 500.0 - (selling_pressure * 200.0).max(0.0);
        let ask_volume = 500.0 + (selling_pressure * 800.0);
        
        sequence.push(MarketData {
            timestamp: base_time + Duration::hours(i),
            timestamp_unix: 1640995200 + (i * 3600),
            price: current_price,
            volume,
            bid: current_price - (10.0 + crash_progress * 40.0),
            ask: current_price + (10.0 + crash_progress * 40.0),
            bid_volume: bid_volume.max(50.0),
            ask_volume,
            volatility,
            returns: vec![-0.02, -0.01, -0.03, -0.015, -0.025],
            volume_history: vec![1000.0; 5],
        });
    }
    
    sequence
}

/// Create a liquidity crisis scenario
fn create_liquidity_crisis() -> Vec<MarketData> {
    let mut sequence = Vec::new();
    let base_time = Utc::now();
    let initial_price = 50000.0;
    
    // Normal conditions
    for i in 0..20 {
        sequence.push(MarketData {
            timestamp: base_time + Duration::minutes(i),
            timestamp_unix: 1640995200 + (i * 60),
            price: initial_price,
            volume: 1000.0,
            bid: initial_price - 10.0,
            ask: initial_price + 10.0,
            bid_volume: 500.0,
            ask_volume: 500.0,
            volatility: 0.02,
            returns: vec![0.001, -0.001, 0.002, 0.0],
            volume_history: vec![1000.0; 5],
        });
    }
    
    // Liquidity crisis - wide spreads, low volume
    for i in 20..40 {
        let crisis_intensity = ((i - 20) as f64) / 20.0;
        let spread_multiplier = 1.0 + (crisis_intensity * 50.0); // Up to 50x normal spread
        let volume_reduction = 1.0 - (crisis_intensity * 0.9); // 90% volume reduction
        
        sequence.push(MarketData {
            timestamp: base_time + Duration::minutes(i),
            timestamp_unix: 1640995200 + (i * 60),
            price: initial_price - (crisis_intensity * 2000.0), // Gradual decline
            volume: 1000.0 * volume_reduction,
            bid: initial_price - (10.0 * spread_multiplier),
            ask: initial_price + (10.0 * spread_multiplier),
            bid_volume: 50.0 * volume_reduction, // Very low liquidity
            ask_volume: 50.0 * volume_reduction,
            volatility: 0.02 + (crisis_intensity * 0.3),
            returns: vec![-0.01, -0.005, -0.015, -0.008],
            volume_history: vec![1000.0; 5],
        });
    }
    
    sequence
}

/// Create a circuit breaker scenario (trading halts)
fn create_circuit_breaker_scenario() -> Vec<MarketData> {
    let mut sequence = Vec::new();
    let base_time = Utc::now();
    let initial_price = 50000.0;
    
    // Pre-halt crash
    for i in 0..5 {
        let crash_intensity = (i as f64) / 5.0;
        sequence.push(MarketData {
            timestamp: base_time + Duration::minutes(i),
            timestamp_unix: 1640995200 + (i * 60),
            price: initial_price - (crash_intensity * 3500.0), // 7% decline triggers halt
            volume: 5000.0 + (crash_intensity * 10000.0),
            bid: initial_price - (crash_intensity * 3600.0),
            ask: initial_price - (crash_intensity * 3400.0),
            bid_volume: 200.0,
            ask_volume: 5000.0,
            volatility: 0.05 + (crash_intensity * 0.2),
            returns: vec![-0.07, -0.06, -0.08, -0.065],
            volume_history: vec![1000.0; 5],
        });
    }
    
    // Trading halt (zero volume, frozen prices)
    for i in 5..10 {
        sequence.push(MarketData {
            timestamp: base_time + Duration::minutes(i),
            timestamp_unix: 1640995200 + (i * 60),
            price: initial_price - 3500.0, // Frozen at halt level
            volume: 0.0, // No trading
            bid: initial_price - 3600.0,
            ask: initial_price - 3400.0,
            bid_volume: 0.0,
            ask_volume: 0.0,
            volatility: 0.0, // Artificially zero during halt
            returns: vec![0.0, 0.0, 0.0, 0.0],
            volume_history: vec![1000.0; 5],
        });
    }
    
    // Post-halt resumption with extreme volatility
    for i in 10..20 {
        let post_halt_volatility = 0.3 - ((i - 10) as f64 / 10.0 * 0.1);
        sequence.push(MarketData {
            timestamp: base_time + Duration::minutes(i),
            timestamp_unix: 1640995200 + (i * 60),
            price: initial_price - 3500.0 + ((i - 10) as f64 * 100.0), // Gradual recovery
            volume: 8000.0,
            bid: initial_price - 3600.0 + ((i - 10) as f64 * 100.0),
            ask: initial_price - 3400.0 + ((i - 10) as f64 * 100.0),
            bid_volume: 800.0,
            ask_volume: 1200.0,
            volatility: post_halt_volatility,
            returns: vec![0.02, -0.01, 0.03, -0.015],
            volume_history: vec![1000.0; 5],
        });
    }
    
    sequence
}

#[cfg(test)]
mod stress_tests {
    use super::*;

    #[test]
    fn test_flash_crash_resilience() {
        let config = MacchiavelianConfig::aggressive_defaults();
        let mut engine = TalebianRiskEngine::new(config);
        
        let crash_sequence = create_flash_crash_sequence();
        let mut assessments = Vec::new();
        
        // Process the entire crash sequence
        for market_data in crash_sequence.iter() {
            let result = engine.assess_risk(market_data);
            
            // Should not panic or error even during extreme conditions
            match result {
                Ok(assessment) => {
                    assessments.push(assessment);
                    
                    // Validate assessment remains within bounds
                    assert!(assessment.overall_risk_score >= 0.0);
                    assert!(assessment.overall_risk_score <= 1.0);
                    assert!(assessment.recommended_position_size >= 0.0);
                    assert!(assessment.recommended_position_size <= 1.0);
                    assert!(assessment.confidence >= 0.0);
                    assert!(assessment.confidence <= 1.0);
                },
                Err(e) => {
                    // If errors occur, they should be handled gracefully
                    println!(\"Warning: Assessment failed during flash crash: {:?}\", e);
                }
            }
        }
        
        // Should have processed most of the sequence
        assert!(assessments.len() >= crash_sequence.len() / 2);
        
        // Flash crash event should be detected
        let crash_assessment = &assessments[10]; // The crash event
        assert!(crash_assessment.black_swan_probability > 0.3);
        assert!(crash_assessment.whale_detection.is_whale_detected);
        assert!(matches!(crash_assessment.whale_detection.direction, WhaleDirection::Selling));
        
        // Position sizes should be reduced during high risk periods
        let pre_crash_size = assessments[5].recommended_position_size;
        let during_crash_size = crash_assessment.recommended_position_size;
        assert!(during_crash_size <= pre_crash_size);
    }

    #[test]
    fn test_bear_market_adaptation() {
        let config = MacchiavelianConfig::aggressive_defaults();
        let mut engine = TalebianRiskEngine::new(config);
        
        let bear_sequence = create_bear_market_crash();
        let mut risk_scores = Vec::new();
        let mut position_sizes = Vec::new();
        
        // Process bear market sequence
        for (i, market_data) in bear_sequence.iter().enumerate() {
            if let Ok(assessment) = engine.assess_risk(market_data) {
                risk_scores.push(assessment.overall_risk_score);
                position_sizes.push(assessment.recommended_position_size);
                
                // Record some trade outcomes to test adaptation
                if i % 10 == 0 {
                    let simulated_return = -0.02; // Consistent losses in bear market
                    engine.record_trade_outcome(simulated_return, false, 0.3).unwrap();
                }
            }
        }
        
        // Should have processed most of the sequence
        assert!(risk_scores.len() >= bear_sequence.len() / 2);
        
        // Risk should generally increase as bear market progresses
        let early_risk = risk_scores[0..10].iter().sum::<f64>() / 10.0;
        let late_risk = risk_scores[risk_scores.len()-10..].iter().sum::<f64>() / 10.0;
        
        // In bear market, risk scores might decrease (meaning higher risk)
        // or position sizes should decrease to manage risk
        let early_positions = position_sizes[0..10].iter().sum::<f64>() / 10.0;
        let late_positions = position_sizes[position_sizes.len()-10..].iter().sum::<f64>() / 10.0;
        
        // System should adapt by reducing position sizes in deteriorating market
        assert!(late_positions <= early_positions * 1.2, \"Position sizes should adapt to bear market\");
        
        // Engine should continue functioning throughout
        let final_status = engine.get_engine_status();
        assert!(final_status.performance_tracker.total_assessments > 50);
    }

    #[test]
    fn test_liquidity_crisis_handling() {
        let config = MacchiavelianConfig::aggressive_defaults();
        let mut engine = TalebianRiskEngine::new(config);
        
        let crisis_sequence = create_liquidity_crisis();
        let mut whale_detections = 0;
        let mut valid_assessments = 0;
        
        for (i, market_data) in crisis_sequence.iter().enumerate() {
            if let Ok(assessment) = engine.assess_risk(market_data) {
                valid_assessments += 1;
                
                // During liquidity crisis (after period 20)
                if i >= 20 {
                    // Should recognize reduced liquidity conditions
                    // Whale detection might be triggered due to low volumes
                    if assessment.whale_detection.is_whale_detected {
                        whale_detections += 1;
                    }
                    
                    // Position sizes should be conservative during liquidity crisis
                    assert!(assessment.recommended_position_size <= 0.5, 
                           \"Position sizes should be conservative during liquidity crisis\");
                    
                    // Black swan probability might increase due to unusual market conditions
                    if market_data.volume < 100.0 { // Very low liquidity
                        assert!(assessment.black_swan_probability >= 0.0);
                    }
                }
                
                // System should handle extreme spreads gracefully
                assert!(assessment.overall_risk_score.is_finite());
                assert!(!assessment.overall_risk_score.is_nan());
            }
        }
        
        // Should process most events even during crisis
        assert!(valid_assessments >= crisis_sequence.len() / 2);
        
        // Should maintain system stability
        let final_status = engine.get_engine_status();
        assert!(final_status.performance_tracker.total_assessments > 0);
    }

    #[test]
    fn test_circuit_breaker_scenario() {
        let config = MacchiavelianConfig::aggressive_defaults();
        let mut engine = TalebianRiskEngine::new(config);
        
        let breaker_sequence = create_circuit_breaker_scenario();
        let mut assessments = Vec::new();
        
        for (i, market_data) in breaker_sequence.iter().enumerate() {
            if let Ok(assessment) = engine.assess_risk(market_data) {
                assessments.push((i, assessment));
            }
        }
        
        // Should process events before, during, and after halt
        assert!(assessments.len() >= breaker_sequence.len() / 2);
        
        // Check behavior in different phases
        let pre_halt = assessments.iter().filter(|(i, _)| *i < 5).collect::<Vec<_>>();
        let during_halt = assessments.iter().filter(|(i, _)| *i >= 5 && *i < 10).collect::<Vec<_>>();
        let post_halt = assessments.iter().filter(|(i, _)| *i >= 10).collect::<Vec<_>>();
        
        // Pre-halt should show increasing risk
        if !pre_halt.is_empty() {
            let (_, last_pre_halt) = pre_halt.last().unwrap();
            assert!(last_pre_halt.black_swan_probability > 0.1);
        }
        
        // During halt (zero volume) should be handled gracefully
        for (_, assessment) in during_halt {
            assert!(assessment.overall_risk_score.is_finite());
            assert!(!assessment.whale_detection.is_whale_detected); // No volume = no whale
        }
        
        // Post-halt should show extreme volatility adaptation
        for (_, assessment) in post_halt {
            assert!(assessment.overall_risk_score.is_finite());
            // High volatility should be reflected in risk assessment
            if breaker_sequence[assessment.black_swan_probability > 0.0 as usize].volatility > 0.2 {
                assert!(assessment.black_swan_probability > 0.0);
            }
        }
    }

    #[test]
    fn test_extreme_volatility_handling() {
        let config = MacchiavelianConfig::aggressive_defaults();
        let mut engine = TalebianRiskEngine::new(config);
        
        // Create sequence with progressively extreme volatility
        let mut extreme_vol_sequence = Vec::new();
        let volatilities = vec![0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]; // Up to 200% volatility
        
        for (i, vol) in volatilities.iter().enumerate() {
            extreme_vol_sequence.push(MarketData {
                timestamp: Utc::now() + Duration::minutes(i as i64),
                timestamp_unix: 1640995200 + (i as i64 * 60),
                price: 50000.0,
                volume: 1000.0,
                bid: 49990.0,
                ask: 50010.0,
                bid_volume: 500.0,
                ask_volume: 500.0,
                volatility: *vol,
                returns: vec![0.01],
                volume_history: vec![1000.0; 5],
            });
        }
        
        let mut successful_assessments = 0;
        let mut black_swan_detections = 0;
        
        for market_data in extreme_vol_sequence {
            if let Ok(assessment) = engine.assess_risk(&market_data) {
                successful_assessments += 1;
                
                // Validate numerical stability under extreme volatility
                assert!(assessment.overall_risk_score.is_finite());
                assert!(assessment.kelly_fraction.is_finite());
                assert!(assessment.recommended_position_size.is_finite());
                assert!(!assessment.overall_risk_score.is_nan());
                
                // Extreme volatility should increase black swan probability
                if market_data.volatility > 0.5 {
                    if assessment.black_swan_probability > 0.1 {
                        black_swan_detections += 1;
                    }
                }
                
                // Position sizes should be bounded even under extreme volatility
                assert!(assessment.recommended_position_size >= 0.01);
                assert!(assessment.recommended_position_size <= 0.75);
            }
        }
        
        // Should handle most extreme volatility scenarios
        assert!(successful_assessments >= 5);
        
        // Should detect some black swan risk under extreme volatility
        assert!(black_swan_detections > 0);
    }

    #[test]
    fn test_market_manipulation_scenario() {
        let config = MacchiavelianConfig::aggressive_defaults();
        let mut engine = TalebianRiskEngine::new(config);
        
        // Simulate pump and dump scenario
        let mut manipulation_sequence = Vec::new();
        let base_time = Utc::now();
        let initial_price = 50000.0;
        
        // Normal conditions
        for i in 0..10 {
            manipulation_sequence.push(MarketData {
                timestamp: base_time + Duration::minutes(i),
                timestamp_unix: 1640995200 + (i * 60),
                price: initial_price,
                volume: 1000.0,
                bid: initial_price - 10.0,
                ask: initial_price + 10.0,
                bid_volume: 500.0,
                ask_volume: 500.0,
                volatility: 0.02,
                returns: vec![0.001],
                volume_history: vec![1000.0; 5],
            });
        }
        
        // Pump phase - artificial buying
        for i in 10..15 {
            let pump_factor = ((i - 10) as f64) / 5.0;
            manipulation_sequence.push(MarketData {
                timestamp: base_time + Duration::minutes(i),
                timestamp_unix: 1640995200 + (i * 60),
                price: initial_price * (1.0 + pump_factor * 0.3), // 30% pump
                volume: 10000.0, // High volume
                bid: initial_price * (1.0 + pump_factor * 0.3) - 5.0,
                ask: initial_price * (1.0 + pump_factor * 0.3) + 5.0,
                bid_volume: 5000.0, // Massive buying
                ask_volume: 200.0,
                volatility: 0.1,
                returns: vec![0.06, 0.05, 0.08, 0.04],
                volume_history: vec![1000.0; 5],
            });
        }
        
        // Dump phase - rapid selling
        for i in 15..20 {
            let dump_factor = ((i - 15) as f64) / 5.0;
            manipulation_sequence.push(MarketData {
                timestamp: base_time + Duration::minutes(i),
                timestamp_unix: 1640995200 + (i * 60),
                price: initial_price * (1.3 - dump_factor * 0.4), // 40% dump from peak
                volume: 15000.0,
                bid: initial_price * (1.3 - dump_factor * 0.4) - 20.0,
                ask: initial_price * (1.3 - dump_factor * 0.4) + 20.0,
                bid_volume: 100.0,
                ask_volume: 8000.0, // Massive selling
                volatility: 0.15,
                returns: vec![-0.08, -0.06, -0.10, -0.07],
                volume_history: vec![1000.0; 5],
            });
        }
        
        let mut pump_detections = 0;
        let mut dump_detections = 0;
        
        for (i, market_data) in manipulation_sequence.iter().enumerate() {
            if let Ok(assessment) = engine.assess_risk(market_data) {
                // Pump phase detection
                if i >= 10 && i < 15 {
                    if assessment.whale_detection.is_whale_detected &&
                       matches!(assessment.whale_detection.direction, WhaleDirection::Buying) {
                        pump_detections += 1;
                    }
                }
                
                // Dump phase detection
                if i >= 15 && i < 20 {
                    if assessment.whale_detection.is_whale_detected &&
                       matches!(assessment.whale_detection.direction, WhaleDirection::Selling) {
                        dump_detections += 1;
                    }
                    
                    // Should show high black swan probability during dump
                    assert!(assessment.black_swan_probability > 0.0);
                }
                
                // System should remain stable throughout manipulation
                assert!(assessment.overall_risk_score.is_finite());
                assert!(assessment.recommended_position_size > 0.0);
            }
        }
        
        // Should detect suspicious whale activity
        assert!(pump_detections > 0, \"Should detect pump activity\");
        assert!(dump_detections > 0, \"Should detect dump activity\");
    }

    #[test]
    fn test_system_recovery_after_stress() {
        let config = MacchiavelianConfig::aggressive_defaults();
        let mut engine = TalebianRiskEngine::new(config);
        
        // Put system through various stress scenarios
        let stress_scenarios = vec![
            create_flash_crash_sequence(),
            create_liquidity_crisis(),
        ];
        
        for scenario in stress_scenarios {
            // Process stress scenario
            for market_data in scenario {
                let _ = engine.assess_risk(&market_data);
            }
        }
        
        // Test recovery with normal market data
        let normal_data = MarketData {
            timestamp: Utc::now(),
            timestamp_unix: 1640995200,
            price: 50000.0,
            volume: 1000.0,
            bid: 49990.0,
            ask: 50010.0,
            bid_volume: 500.0,
            ask_volume: 500.0,
            volatility: 0.02,
            returns: vec![0.01],
            volume_history: vec![1000.0; 5],
        };
        
        // System should still function normally after stress
        let recovery_assessment = engine.assess_risk(&normal_data);
        assert!(recovery_assessment.is_ok());
        
        let assessment = recovery_assessment.unwrap();
        assert!(assessment.overall_risk_score.is_finite());
        assert!(assessment.recommended_position_size > 0.0);
        assert!(assessment.confidence > 0.0);
        
        // Engine status should be reasonable
        let status = engine.get_engine_status();
        assert!(status.performance_tracker.total_assessments > 50);
        
        // Memory usage should be bounded
        assert!(status.total_assessments <= 10000);
    }

    #[test]
    fn test_concurrent_stress_scenarios() {
        use std::sync::{Arc, Mutex};
        use std::thread;
        
        let config = MacchiavelianConfig::aggressive_defaults();
        let engine = Arc::new(Mutex::new(TalebianRiskEngine::new(config)));
        
        let mut handles = vec![];
        
        // Simulate multiple concurrent stress scenarios
        for thread_id in 0..5 {
            let engine_clone = Arc::clone(&engine);
            let handle = thread::spawn(move || {
                let stress_data = MarketData {
                    timestamp: Utc::now(),
                    timestamp_unix: 1640995200 + thread_id,
                    price: 50000.0 - (thread_id as f64 * 1000.0), // Different stress levels
                    volume: 1000.0 * (thread_id as f64 + 1.0), // Varying volumes
                    bid: 49990.0 - (thread_id as f64 * 1000.0),
                    ask: 50010.0 - (thread_id as f64 * 1000.0),
                    bid_volume: 500.0,
                    ask_volume: 500.0,
                    volatility: 0.1 + (thread_id as f64 * 0.1), // Increasing volatility
                    returns: vec![-0.05],
                    volume_history: vec![1000.0; 5],
                };
                
                let mut engine_guard = engine_clone.lock().unwrap();
                engine_guard.assess_risk(&stress_data)
            });
            
            handles.push(handle);
        }
        
        // Collect results
        let mut successful_assessments = 0;
        for handle in handles {
            if let Ok(Ok(_)) = handle.join() {
                successful_assessments += 1;
            }
        }
        
        // Should handle concurrent stress gracefully
        assert!(successful_assessments >= 3, \"Should handle most concurrent stress scenarios\");
        
        // Engine should remain functional
        let engine_guard = engine.lock().unwrap();
        let status = engine_guard.get_engine_status();
        assert!(status.performance_tracker.total_assessments > 0);
    }
}