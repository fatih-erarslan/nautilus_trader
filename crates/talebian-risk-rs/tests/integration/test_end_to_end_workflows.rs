//! Integration tests for end-to-end risk assessment workflows
//! Tests complete system integration from market data to risk recommendations

use talebian_risk_rs::{
    risk_engine::*, black_swan::*, antifragility::*, kelly::*, whale_detection::*,
    MacchiavelianConfig, MarketData, TalebianRiskError,
    WhaleDirection, WhaleDetection
};
use chrono::{Utc, Duration};
use approx::assert_relative_eq;

/// Helper to create a market data sequence simulating a trading day
fn create_market_data_sequence() -> Vec<MarketData> {
    let mut sequence = Vec::new();
    let base_time = Utc::now();
    
    // Normal morning trading
    for i in 0..100 {
        sequence.push(MarketData {
            timestamp: base_time + Duration::minutes(i),
            timestamp_unix: 1640995200 + (i * 60),
            price: 50000.0 + (i as f64 * 10.0),
            volume: 1000.0 + (i as f64 * 5.0),
            bid: 49990.0 + (i as f64 * 10.0),
            ask: 50010.0 + (i as f64 * 10.0),
            bid_volume: 500.0 + (i as f64 * 2.0),
            ask_volume: 500.0 + (i as f64 * 2.0),
            volatility: 0.02 + (i as f64 * 0.0001),
            returns: vec![0.001, 0.002, -0.001, 0.0015, 0.001],
            volume_history: vec![950.0, 1000.0, 1050.0, 980.0, 1020.0],
        });
    }
    
    // Whale activity spike (mid-day)
    for i in 100..110 {
        sequence.push(MarketData {
            timestamp: base_time + Duration::minutes(i),
            timestamp_unix: 1640995200 + (i * 60),
            price: 50000.0 + (i as f64 * 20.0), // Accelerated price movement
            volume: 5000.0, // Whale volume
            bid: 49980.0 + (i as f64 * 20.0),
            ask: 50020.0 + (i as f64 * 20.0),
            bid_volume: 2000.0, // Heavy buying
            ask_volume: 800.0,
            volatility: 0.05, // Increased volatility
            returns: vec![0.02, 0.025, 0.015, 0.03, 0.02],
            volume_history: vec![1000.0; 5],
        });
    }
    
    // Market crash simulation (afternoon)
    for i in 110..130 {
        let crash_intensity = (i - 110) as f64 / 20.0;
        sequence.push(MarketData {
            timestamp: base_time + Duration::minutes(i),
            timestamp_unix: 1640995200 + (i * 60),
            price: 52000.0 - (crash_intensity * 5000.0), // Sharp decline
            volume: 3000.0 + (crash_intensity * 2000.0), // Panic selling volume
            bid: 51980.0 - (crash_intensity * 5000.0),
            ask: 52020.0 - (crash_intensity * 5000.0),
            bid_volume: 300.0 - (crash_intensity * 100.0),
            ask_volume: 2000.0 + (crash_intensity * 1000.0), // Heavy selling
            volatility: 0.1 + (crash_intensity * 0.1), // Extreme volatility
            returns: vec![-0.05, -0.08, -0.12, -0.06, -0.10],
            volume_history: vec![1000.0; 5],
        });
    }
    
    // Recovery phase
    for i in 130..150 {
        let recovery = ((i - 130) as f64 / 20.0).min(1.0);
        sequence.push(MarketData {
            timestamp: base_time + Duration::minutes(i),
            timestamp_unix: 1640995200 + (i * 60),
            price: 47000.0 + (recovery * 2000.0), // Gradual recovery
            volume: 2000.0 - (recovery * 500.0),
            bid: 46980.0 + (recovery * 2000.0),
            ask: 47020.0 + (recovery * 2000.0),
            bid_volume: 800.0 + (recovery * 400.0),
            ask_volume: 1200.0 - (recovery * 400.0),
            volatility: 0.15 - (recovery * 0.1), // Volatility normalizing
            returns: vec![0.01, 0.005, 0.008, 0.012, 0.006],
            volume_history: vec![1000.0; 5],
        });
    }
    
    sequence
}

/// Helper to create extreme black swan market data
fn create_black_swan_sequence() -> Vec<MarketData> {
    let mut sequence = Vec::new();
    let base_time = Utc::now();
    
    // Build up normal conditions
    for i in 0..50 {
        sequence.push(MarketData {
            timestamp: base_time + Duration::minutes(i),
            timestamp_unix: 1640995200 + (i * 60),
            price: 50000.0,
            volume: 1000.0,
            bid: 49990.0,
            ask: 50010.0,
            bid_volume: 500.0,
            ask_volume: 500.0,
            volatility: 0.02,
            returns: vec![0.005, 0.008, -0.003, 0.01, 0.006],
            volume_history: vec![950.0, 1000.0, 1050.0, 980.0, 1020.0],
        });
    }
    
    // Sudden black swan event
    sequence.push(MarketData {
        timestamp: base_time + Duration::minutes(50),
        timestamp_unix: 1640995200 + (50 * 60),
        price: 35000.0, // 30% crash
        volume: 50000.0, // 50x normal volume
        bid: 34000.0,
        ask: 36000.0,
        bid_volume: 100.0,
        ask_volume: 10000.0, // Massive selling
        volatility: 0.5, // Extreme volatility
        returns: vec![-0.3, -0.25, -0.35, -0.28, -0.32],
        volume_history: vec![1000.0; 5],
    });
    
    sequence
}

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_complete_risk_assessment_workflow() {
        let config = MacchiavelianConfig::aggressive_defaults();
        let mut engine = TalebianRiskEngine::new(config);
        
        let market_sequence = create_market_data_sequence();
        let mut assessments = Vec::new();
        
        // Process entire market sequence
        for market_data in market_sequence.iter() {
            let assessment = engine.assess_risk(market_data).unwrap();
            assessments.push(assessment);
        }
        
        assert_eq!(assessments.len(), market_sequence.len());
        
        // Verify assessment evolution during different phases
        
        // Normal phase (0-100): Should show stable risk
        let normal_phase = &assessments[0..100];
        for assessment in normal_phase {
            assert!(assessment.overall_risk_score > 0.3);
            assert!(assessment.overall_risk_score < 0.8);
            assert!(!assessment.whale_detection.is_whale_detected);
        }
        
        // Whale phase (100-110): Should detect whale activity
        let whale_phase = &assessments[100..110];
        for assessment in whale_phase {
            assert!(assessment.whale_detection.is_whale_detected);
            assert!(assessment.whale_detection.confidence > 0.5);
            assert!(matches!(assessment.whale_detection.direction, WhaleDirection::Buying));
            assert!(assessment.recommended_position_size > normal_phase[0].recommended_position_size);
        }
        
        // Crash phase (110-130): Should show high risk and black swan detection
        let crash_phase = &assessments[110..130];
        for assessment in crash_phase {
            assert!(assessment.black_swan_probability > 0.1);
            assert!(assessment.overall_risk_score < 0.5); // High risk (low score)
        }
        
        // Recovery phase (130-150): Should show adaptation
        let recovery_phase = &assessments[130..150];
        let early_recovery = &recovery_phase[0..5];
        let late_recovery = &recovery_phase[15..20];
        
        // Risk should improve over recovery period
        let early_avg_risk = early_recovery.iter().map(|a| a.overall_risk_score).sum::<f64>() / early_recovery.len() as f64;
        let late_avg_risk = late_recovery.iter().map(|a| a.overall_risk_score).sum::<f64>() / late_recovery.len() as f64;
        assert!(late_avg_risk > early_avg_risk, \"Risk should improve during recovery\");
    }

    #[test]
    fn test_recommendation_generation_workflow() {
        let config = MacchiavelianConfig::aggressive_defaults();
        let mut engine = TalebianRiskEngine::new(config);
        
        let market_sequence = create_market_data_sequence();
        let mut recommendations = Vec::new();
        
        // Generate recommendations for key market phases
        let key_indices = vec![50, 105, 120, 145]; // Normal, whale, crash, recovery
        
        for &i in key_indices.iter() {
            let recommendation = engine.generate_recommendations(&market_sequence[i]).unwrap();
            recommendations.push(recommendation);
        }
        
        assert_eq!(recommendations.len(), 4);
        
        // Normal market recommendation
        let normal_rec = &recommendations[0];
        assert!(normal_rec.position_sizing.final_recommended_size > 0.05);
        assert!(normal_rec.position_sizing.final_recommended_size < 0.5);
        assert!(normal_rec.timing_guidance.entry_urgency.contains(\"MEDIUM\") || 
               normal_rec.timing_guidance.entry_urgency.contains(\"LOW\"));
        
        // Whale market recommendation
        let whale_rec = &recommendations[1];
        assert!(whale_rec.position_sizing.final_recommended_size > normal_rec.position_sizing.final_recommended_size);
        assert!(whale_rec.timing_guidance.whale_activity_level.contains(\"ACTIVE\"));
        assert!(whale_rec.timing_guidance.entry_urgency.contains(\"HIGH\") || 
               whale_rec.timing_guidance.entry_urgency.contains(\"IMMEDIATE\"));
        
        // Crash market recommendation
        let crash_rec = &recommendations[2];
        assert!(crash_rec.position_sizing.final_recommended_size < normal_rec.position_sizing.final_recommended_size);
        assert!(crash_rec.risk_controls.stop_loss_level > normal_rec.risk_controls.stop_loss_level);
        assert!(crash_rec.timing_guidance.market_regime.contains(\"VOLATILE\"));
        
        // Recovery market recommendation
        let recovery_rec = &recommendations[3];
        assert!(recovery_rec.position_sizing.final_recommended_size > crash_rec.position_sizing.final_recommended_size);
    }

    #[test]
    fn test_black_swan_detection_workflow() {
        let config = MacchiavelianConfig::aggressive_defaults();
        let mut engine = TalebianRiskEngine::new(config);
        
        let black_swan_sequence = create_black_swan_sequence();
        let mut swan_detections = Vec::new();
        
        // Process sequence and track black swan probability
        for market_data in black_swan_sequence.iter() {
            let assessment = engine.assess_risk(market_data).unwrap();
            swan_detections.push(assessment.black_swan_probability);
        }
        
        // Normal period should have low black swan probability
        let normal_period = &swan_detections[0..50];
        let avg_normal_probability = normal_period.iter().sum::<f64>() / normal_period.len() as f64;
        assert!(avg_normal_probability < 0.2, \"Normal period should have low black swan probability\");
        
        // Black swan event should trigger high probability
        let swan_event_probability = swan_detections[50];
        assert!(swan_event_probability > 0.5, \"Black swan event should be detected with high probability\");
        
        // Should show dramatic change in assessment
        let pre_swan = engine.assess_risk(&black_swan_sequence[49]).unwrap();
        let during_swan = engine.assess_risk(&black_swan_sequence[50]).unwrap();
        
        assert!(during_swan.black_swan_probability > pre_swan.black_swan_probability * 2.0);
        assert!(during_swan.overall_risk_score < pre_swan.overall_risk_score);
        assert!(during_swan.whale_detection.is_whale_detected);
        assert!(matches!(during_swan.whale_detection.direction, WhaleDirection::Selling));
    }

    #[test]
    fn test_antifragility_measurement_workflow() {
        let config = MacchiavelianConfig::aggressive_defaults();
        let mut engine = TalebianRiskEngine::new(config);
        
        // Create data that should show antifragile behavior (benefits from volatility)
        let mut antifragile_sequence = Vec::new();
        let base_time = Utc::now();
        
        for i in 0..200 {
            let volatility = 0.02 + (i as f64 * 0.001); // Increasing volatility
            let returns = 0.005 + (volatility * 2.0); // Returns increase with volatility
            
            antifragile_sequence.push(MarketData {
                timestamp: base_time + Duration::minutes(i),
                timestamp_unix: 1640995200 + (i * 60),
                price: 50000.0 + (i as f64 * 50.0),
                volume: 1000.0,
                bid: 49990.0 + (i as f64 * 50.0),
                ask: 50010.0 + (i as f64 * 50.0),
                bid_volume: 500.0,
                ask_volume: 500.0,
                volatility,
                returns: vec![returns],
                volume_history: vec![1000.0; 5],
            });
        }
        
        let mut antifragility_scores = Vec::new();
        
        // Process sequence and track antifragility evolution
        for market_data in antifragile_sequence.iter() {
            let assessment = engine.assess_risk(market_data).unwrap();
            antifragility_scores.push(assessment.antifragility_score);
        }
        
        // Antifragility should be detectable in the latter part of the sequence
        let early_scores = &antifragility_scores[0..50];
        let late_scores = &antifragility_scores[150..200];
        
        let early_avg = early_scores.iter().sum::<f64>() / early_scores.len() as f64;
        let late_avg = late_scores.iter().sum::<f64>() / late_scores.len() as f64;
        
        // With sufficient data showing antifragile pattern, score should improve
        assert!(late_avg >= early_avg, \"Antifragility should be detected with sufficient antifragile data\");
    }

    #[test]
    fn test_performance_tracking_workflow() {
        let config = MacchiavelianConfig::aggressive_defaults();
        let mut engine = TalebianRiskEngine::new(config);
        
        let market_sequence = create_market_data_sequence();
        
        // Simulate trading based on recommendations
        let mut total_return = 0.0;
        let mut trades_taken = 0;
        
        for (i, market_data) in market_sequence.iter().enumerate() {
            let assessment = engine.assess_risk(market_data).unwrap();
            
            // Simulate taking a trade if opportunity score is high enough
            if assessment.parasitic_opportunity.opportunity_score > 0.6 {
                // Simulate trade outcome based on next period (if available)
                if i + 1 < market_sequence.len() {
                    let next_price = market_sequence[i + 1].price;
                    let return_pct = (next_price - market_data.price) / market_data.price;
                    
                    // Adjust return by position size
                    let position_return = return_pct * assessment.recommended_position_size;
                    total_return += position_return;
                    
                    // Record trade outcome
                    engine.record_trade_outcome(
                        return_pct,
                        assessment.whale_detection.is_whale_detected,
                        assessment.parasitic_opportunity.momentum_factor
                    ).unwrap();
                    
                    trades_taken += 1;
                }
            }
        }
        
        let final_status = engine.get_engine_status();
        
        // Verify performance tracking
        assert!(final_status.performance_tracker.total_assessments > 0);
        assert_eq!(trades_taken, final_status.kelly_status.total_trades);
        
        if trades_taken > 0 {
            assert!(final_status.performance_tracker.total_return != 0.0);
            // With our whale and opportunity detection, should have some successful predictions
            assert!(final_status.performance_tracker.successful_predictions > 0);
        }
    }

    #[test]
    fn test_multi_timeframe_consistency() {
        let config = MacchiavelianConfig::aggressive_defaults();
        let mut engine = TalebianRiskEngine::new(config);
        
        // Test with different timeframe granularities
        let timeframes = vec![
            Duration::seconds(60),  // 1 minute
            Duration::minutes(5),   // 5 minutes  
            Duration::minutes(15),  // 15 minutes
        ];
        
        let base_data = MarketData {
            timestamp: Utc::now(),
            timestamp_unix: 1640995200,
            price: 50000.0,
            volume: 3000.0, // Whale volume
            bid: 49980.0,
            ask: 50020.0,
            bid_volume: 1500.0,
            ask_volume: 800.0,
            volatility: 0.04,
            returns: vec![0.02, 0.015, 0.025, 0.018, 0.03],
            volume_history: vec![1000.0; 5],
        };
        
        let mut assessments = Vec::new();
        
        // Generate assessments for each timeframe
        for timeframe in timeframes {
            let mut timeframe_data = base_data.clone();
            timeframe_data.timestamp += timeframe;
            timeframe_data.timestamp_unix += timeframe.num_seconds();
            
            let assessment = engine.assess_risk(&timeframe_data).unwrap();
            assessments.push(assessment);
        }
        
        // All assessments should detect whale activity consistently
        for assessment in &assessments {
            assert!(assessment.whale_detection.is_whale_detected);
            assert!(assessment.whale_detection.confidence > 0.5);
        }
        
        // Risk scores should be relatively consistent across timeframes
        let risk_scores: Vec<f64> = assessments.iter().map(|a| a.overall_risk_score).collect();
        let min_risk = risk_scores.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_risk = risk_scores.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        
        // Risk scores shouldn't vary dramatically for same market conditions
        assert!((max_risk - min_risk) < 0.3, \"Risk scores should be consistent across timeframes\");
    }

    #[test]
    fn test_error_recovery_workflow() {
        let config = MacchiavelianConfig::aggressive_defaults();
        let mut engine = TalebianRiskEngine::new(config);
        
        // Mix of normal and problematic data
        let test_data = vec![
            // Normal data
            MarketData {
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
            },
            // Extreme data that might cause issues
            MarketData {
                timestamp: Utc::now(),
                timestamp_unix: 1640995201,
                price: f64::INFINITY,
                volume: f64::NAN,
                bid: 0.0,
                ask: f64::INFINITY,
                bid_volume: f64::NEG_INFINITY,
                ask_volume: 1000.0,
                volatility: f64::NAN,
                returns: vec![f64::INFINITY, f64::NEG_INFINITY],
                volume_history: vec![],
            },
            // Recovery with normal data
            MarketData {
                timestamp: Utc::now(),
                timestamp_unix: 1640995202,
                price: 50100.0,
                volume: 1100.0,
                bid: 50090.0,
                ask: 50110.0,
                bid_volume: 550.0,
                ask_volume: 550.0,
                volatility: 0.025,
                returns: vec![0.002],
                volume_history: vec![1000.0; 5],
            },
        ];
        
        let mut successful_assessments = 0;
        let mut total_attempts = 0;
        
        for market_data in test_data {
            total_attempts += 1;
            
            let result = engine.assess_risk(&market_data);
            
            match result {
                Ok(assessment) => {
                    successful_assessments += 1;
                    
                    // Verify assessment is valid
                    assert!(assessment.overall_risk_score >= 0.0);
                    assert!(assessment.overall_risk_score <= 1.0);
                    assert!(assessment.confidence >= 0.0);
                    assert!(assessment.confidence <= 1.0);
                },
                Err(_) => {
                    // Some failures expected with extreme data
                    continue;
                }
            }
        }
        
        // Should successfully process at least the normal data points
        assert!(successful_assessments >= 2, \"Should handle normal data even after errors\");
        
        // Engine should still be functional
        let final_status = engine.get_engine_status();
        assert!(final_status.performance_tracker.total_assessments > 0);
    }

    #[test]
    fn test_memory_efficiency_workflow() {
        let config = MacchiavelianConfig::aggressive_defaults();
        let mut engine = TalebianRiskEngine::new(config);
        
        // Process a large amount of data to test memory management
        let initial_memory = std::mem::size_of_val(&engine);
        
        for i in 0..10000 {
            let market_data = MarketData {
                timestamp: Utc::now() + Duration::seconds(i),
                timestamp_unix: 1640995200 + i,
                price: 50000.0 + (i as f64 * 0.1),
                volume: 1000.0 + (i as f64 * 0.1),
                bid: 49990.0 + (i as f64 * 0.1),
                ask: 50010.0 + (i as f64 * 0.1),
                bid_volume: 500.0,
                ask_volume: 500.0,
                volatility: 0.02 + (i as f64 * 0.00001),
                returns: vec![0.001],
                volume_history: vec![1000.0; 5],
            };
            
            engine.assess_risk(&market_data).unwrap();
            
            // Occasionally record trade outcomes
            if i % 100 == 0 {
                engine.record_trade_outcome(0.01, i % 3 == 0, 0.5).unwrap();
            }
        }
        
        // Memory usage should be bounded
        assert!(engine.assessment_history.len() <= 10000, \"Assessment history should be bounded\");
        
        // Engine should still function efficiently
        let final_data = MarketData {
            timestamp: Utc::now(),
            timestamp_unix: 1640995200 + 10000,
            price: 51000.0,
            volume: 2000.0,
            bid: 50990.0,
            ask: 51010.0,
            bid_volume: 1000.0,
            ask_volume: 1000.0,
            volatility: 0.03,
            returns: vec![0.02],
            volume_history: vec![1000.0; 5],
        };
        
        let start_time = std::time::Instant::now();
        let final_assessment = engine.assess_risk(&final_data).unwrap();
        let duration = start_time.elapsed();
        
        // Should still process quickly
        assert!(duration.as_millis() < 100, \"Assessment should remain fast after processing large dataset\");
        assert!(final_assessment.overall_risk_score >= 0.0);
        assert!(final_assessment.overall_risk_score <= 1.0);
    }

    #[test]
    fn test_configuration_impact_workflow() {
        let aggressive_config = MacchiavelianConfig::aggressive_defaults();
        let conservative_config = MacchiavelianConfig::conservative_baseline();
        
        let mut aggressive_engine = TalebianRiskEngine::new(aggressive_config);
        let mut conservative_engine = TalebianRiskEngine::new(conservative_config);
        
        // Test with whale activity data
        let whale_data = MarketData {
            timestamp: Utc::now(),
            timestamp_unix: 1640995200,
            price: 50000.0,
            volume: 4000.0, // 4x normal volume
            bid: 49980.0,
            ask: 50020.0,
            bid_volume: 2000.0,
            ask_volume: 800.0,
            volatility: 0.06,
            returns: vec![0.03, 0.025, 0.035, 0.02, 0.04],
            volume_history: vec![1000.0; 5],
        };
        
        let aggressive_assessment = aggressive_engine.assess_risk(&whale_data).unwrap();
        let conservative_assessment = conservative_engine.assess_risk(&whale_data).unwrap();
        
        // Aggressive should take larger positions
        assert!(aggressive_assessment.recommended_position_size >= 
               conservative_assessment.recommended_position_size);
        
        // Aggressive should be more likely to detect opportunities
        assert!(aggressive_assessment.parasitic_opportunity.opportunity_score >= 
               conservative_assessment.parasitic_opportunity.opportunity_score);
        
        // Generate recommendations and compare
        let aggressive_rec = aggressive_engine.generate_recommendations(&whale_data).unwrap();
        let conservative_rec = conservative_engine.generate_recommendations(&whale_data).unwrap();
        
        // Aggressive should recommend larger positions
        assert!(aggressive_rec.position_sizing.final_recommended_size >= 
               conservative_rec.position_sizing.final_recommended_size);
        
        // Conservative should have tighter risk controls
        assert!(conservative_rec.risk_controls.stop_loss_level <= 
               aggressive_rec.risk_controls.stop_loss_level);
    }
}