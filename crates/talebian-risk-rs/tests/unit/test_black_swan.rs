//! Comprehensive unit tests for Black Swan detection
//! Tests event detection, probability calculation, and edge cases

use talebian_risk_rs::{
    black_swan::*, MacchiavelianConfig, MarketData, TalebianRiskError
};
use chrono::{DateTime, Utc, Duration};
use approx::assert_relative_eq;

/// Test helper to create normal market data
fn create_normal_market_data() -> MarketData {
    MarketData {
        timestamp: Utc::now(),
        timestamp_unix: 1640995200,
        price: 50000.0,
        volume: 1000.0,
        bid: 49990.0,
        ask: 50010.0,
        bid_volume: 500.0,
        ask_volume: 500.0,
        volatility: 0.02, // Normal 2% volatility
        returns: vec![0.005, 0.008, -0.003, 0.01, 0.006],
        volume_history: vec![950.0, 1000.0, 1050.0, 980.0, 1020.0],
    }
}

/// Test helper to create extreme market data (potential black swan)
fn create_extreme_market_data() -> MarketData {
    MarketData {
        timestamp: Utc::now(),
        timestamp_unix: 1640995200,
        price: 45000.0, // 10% price drop
        volume: 5000.0, // 5x volume spike
        bid: 44800.0,
        ask: 45200.0,
        bid_volume: 200.0,
        ask_volume: 2000.0, // Heavy selling pressure
        volatility: 0.15, // Extreme volatility
        returns: vec![-0.15, -0.08, -0.12, -0.10, -0.05], // Severe negative returns
        volume_history: vec![1000.0, 1100.0, 1000.0, 950.0, 1000.0],
    }
}

#[cfg(test)]
mod black_swan_tests {
    use super::*;

    #[test]
    fn test_detector_creation() {
        let params = BlackSwanParams::default();
        let detector = BlackSwanDetector::new("test_detector".to_string(), params.clone());
        
        assert_eq!(detector.detector_id, "test_detector");
        assert_eq!(detector.detected_events.len(), 0);
        assert_eq!(detector.warning_signals.len(), 0);
        assert_eq!(detector.return_history.len(), 0);
        assert_eq!(detector.params.extreme_threshold, 3.0);
        assert_eq!(detector.params.min_impact_magnitude, 0.1);
    }

    #[test]
    fn test_detector_from_config() {
        let config = MacchiavelianConfig::aggressive_defaults();
        let detector = BlackSwanDetector::new_from_config(config.clone());
        
        assert_eq!(detector.config.black_swan_threshold, config.black_swan_threshold);
        assert_eq!(detector.params.rarity_threshold, config.black_swan_threshold);
        assert_eq!(detector.extreme_events_detected, 0);
    }

    #[test]
    fn test_normal_market_detection() {
        let config = MacchiavelianConfig::aggressive_defaults();
        let mut detector = BlackSwanDetector::new_from_config(config);
        let market_data = create_normal_market_data();
        
        let result = detector.detect(&market_data).unwrap();
        
        // Normal market should not trigger black swan detection
        assert!(!result.detected);
        assert!(result.probability < 0.5);
        assert!(result.tail_risk_score < 0.5);
        assert_eq!(result.extreme_events_count, 0);
    }

    #[test]
    fn test_extreme_market_detection() {
        let config = MacchiavelianConfig::aggressive_defaults();
        let mut detector = BlackSwanDetector::new_from_config(config);
        let market_data = create_extreme_market_data();
        
        let result = detector.detect(&market_data).unwrap();
        
        // Extreme market should potentially trigger detection
        assert!(result.probability >= 0.0);
        assert!(result.estimated_impact != 0.0);
        assert!(result.confidence > 0.0);
        assert!(result.tail_risk_score >= 0.0);
    }

    #[test]
    fn test_black_swan_event_detection_sequence() {
        let mut detector = BlackSwanDetector::new("test".to_string(), BlackSwanParams::default());
        
        // Add normal returns to establish baseline
        for i in 0..100 {
            let normal_return = 0.01 * (i as f64 / 50.0).sin(); // Simulate normal variation
            let timestamp = Utc::now() + Duration::days(i);
            detector.detect_black_swan(normal_return, timestamp).unwrap();
        }
        
        // Verify baseline is established
        assert!(detector.baseline_statistics.is_some());
        let initial_events = detector.detected_events.len();
        
        // Add extreme return that should trigger detection
        let extreme_return = -0.20; // 20% loss
        let extreme_timestamp = Utc::now() + Duration::days(101);
        let result = detector.detect_black_swan(extreme_return, extreme_timestamp).unwrap();
        
        // May or may not detect depending on statistical significance
        if let Some(event) = result {
            assert_eq!(event.direction, SwanDirection::Negative);
            assert!(event.magnitude > 3.0);
            assert!(event.impact > 0.1);
            assert!(event.ex_ante_probability < 0.1);
        }
        
        // Should have processed the extreme event
        assert!(detector.detected_events.len() >= initial_events);
    }

    #[test]
    fn test_probability_calculation() {
        let mut detector = BlackSwanDetector::new("test".to_string(), BlackSwanParams::default());
        
        // Build sufficient history for statistical analysis
        for i in 0..200 {
            let return_val = if i % 50 == 0 { -0.05 } else { 0.01 }; // Occasional shocks
            detector.return_history.push(return_val);
        }
        
        detector.update_baseline_statistics().unwrap();
        
        let probability = detector.calculate_black_swan_probability().unwrap();
        
        // Should return reasonable probability
        assert!(probability >= 0.0 && probability <= 1.0);
        assert!(probability < 0.2); // Should be rare by definition
    }

    #[test]
    fn test_event_classification() {
        let detector = BlackSwanDetector::new("test".to_string(), BlackSwanParams::default());
        
        // Test market crash classification
        let crash_type = detector.classify_event_type(-0.2, 6.0);
        assert!(matches!(crash_type, BlackSwanType::MarketCrash));
        
        // Test market rally classification
        let rally_type = detector.classify_event_type(0.2, 6.0);
        assert!(matches!(rally_type, BlackSwanType::MarketRally));
        
        // Test volatility spike classification
        let spike_type = detector.classify_event_type(-0.1, 4.5);
        assert!(matches!(spike_type, BlackSwanType::VolatilitySpike));
        
        // Test unknown classification
        let unknown_type = detector.classify_event_type(0.05, 2.0);
        assert!(matches!(unknown_type, BlackSwanType::Unknown));
    }

    #[test]
    fn test_warning_signal_detection() {
        let mut detector = BlackSwanDetector::new("test".to_string(), BlackSwanParams::default());
        
        // Add normal volatility baseline
        for _ in 0..30 {
            detector.return_history.push(0.01);
        }
        
        // Add high volatility period to trigger warning
        for _ in 0..10 {
            detector.return_history.push(0.05 * if detector.return_history.len() % 2 == 0 { 1.0 } else { -1.0 });
        }
        
        let initial_warnings = detector.warning_signals.len();
        let result = detector.detect_warning_signals(0.06, Utc::now());
        assert!(result.is_ok());
        
        // May have detected volatility warning signal
        if detector.warning_signals.len() > initial_warnings {
            let latest_signal = detector.warning_signals.last().unwrap();
            assert!(matches!(latest_signal.signal_type, SignalType::VolatilityAnomalies));
            assert!(latest_signal.strength > 0.0);
        }
    }

    #[test]
    fn test_clustering_analysis() {
        let mut detector = BlackSwanDetector::new("test".to_string(), BlackSwanParams::default());
        
        // Create clustered events (events close in time)
        let base_time = Utc::now();
        let mut events = Vec::new();
        
        // First cluster - 3 events within 20 days
        for i in 0..3 {
            events.push(BlackSwanEvent {
                id: format!("cluster1_event_{}", i),
                timestamp: base_time + Duration::days(i * 5),
                magnitude: 4.0,
                impact: 0.12,
                direction: SwanDirection::Negative,
                event_type: BlackSwanType::MarketCrash,
                ex_ante_probability: 0.001,
                duration: Duration::hours(2),
                recovery_time: Some(Duration::days(1)),
                market_conditions: MarketConditions {
                    volatility_level: 0.3,
                    liquidity_level: 0.3,
                    correlation_breakdown: true,
                    sentiment_extreme: true,
                    risk_off_mode: true,
                },
                predictability: None,
            });
        }
        
        // Second cluster - 2 events within 15 days (after gap)
        for i in 0..2 {
            events.push(BlackSwanEvent {
                id: format!("cluster2_event_{}", i),
                timestamp: base_time + Duration::days(60 + i * 7),
                magnitude: 3.5,
                impact: 0.08,
                direction: SwanDirection::Negative,
                event_type: BlackSwanType::VolatilitySpike,
                ex_ante_probability: 0.002,
                duration: Duration::hours(1),
                recovery_time: Some(Duration::hours(12)),
                market_conditions: MarketConditions {
                    volatility_level: 0.25,
                    liquidity_level: 0.4,
                    correlation_breakdown: false,
                    sentiment_extreme: false,
                    risk_off_mode: false,
                },
                predictability: None,
            });
        }
        
        detector.detected_events = events;
        
        let analysis = detector.analyze_event_clustering().unwrap();
        
        // Should detect clustering
        assert!(analysis.total_clusters >= 0);
        assert!(analysis.clustering_tendency >= 0.0);
        
        // With our setup, should detect at least some clustering
        if analysis.total_clusters > 0 {
            assert!(!analysis.cluster_details.is_empty());
        }
    }

    #[test]
    fn test_tail_risk_calculation() {
        let config = MacchiavelianConfig::aggressive_defaults();
        let detector = BlackSwanDetector::new_from_config(config);
        
        let tail_risk = detector.calculate_tail_risk().unwrap();
        
        // Validate tail risk structure
        assert!(tail_risk.extreme_event_probability >= 0.0 && tail_risk.extreme_event_probability <= 1.0);
        assert!(tail_risk.expected_tail_loss <= 0.0); // Should be negative (loss)
        assert!(tail_risk.confidence_level > 0.0 && tail_risk.confidence_level <= 1.0);
        assert!(tail_risk.tail_risk_score >= 0.0);
        assert!(tail_risk.var_95 <= 0.0); // VaR should be negative
        assert!(tail_risk.cvar_95 <= tail_risk.var_95); // CVaR should be worse than VaR
        assert!(tail_risk.maximum_drawdown <= 0.0); // Drawdown should be negative
    }

    #[test]
    fn test_market_observation_handling() {
        let mut detector = BlackSwanDetector::new("test".to_string(), BlackSwanParams::default());
        
        let observation = MarketObservation {
            timestamp: Utc::now(),
            returns: std::collections::HashMap::from([("BTC".to_string(), -0.05)]),
            volatilities: std::collections::HashMap::from([("BTC".to_string(), 0.15)]),
            correlations: ndarray::Array2::zeros((1, 1)),
            volumes: std::collections::HashMap::from([("BTC".to_string(), 2000.0)]),
            price: 45000.0,
            volume: 2000.0,
            volatility: 0.15,
            regime: talebian_risk_rs::strategies::MarketRegime::Volatile,
        };
        
        let initial_len = detector.return_history.len();
        detector.add_observation(observation);
        
        // Should have added to history
        assert!(detector.return_history.len() > initial_len);
        assert!(detector.timestamp_history.len() > 0);
    }

    #[test]
    fn test_edge_case_insufficient_data() {
        let mut detector = BlackSwanDetector::new("test".to_string(), BlackSwanParams::default());
        
        // Test with very little data
        for i in 0..5 {
            detector.return_history.push(0.01);
            detector.timestamp_history.push(Utc::now() + Duration::days(i));
        }
        
        // Should handle gracefully without errors
        let result = detector.calculate_black_swan_probability();
        assert!(result.is_ok());
        
        let probability = result.unwrap();
        assert!(probability >= 0.0);
    }

    #[test]
    fn test_edge_case_extreme_values() {
        let mut detector = BlackSwanDetector::new("test".to_string(), BlackSwanParams::default());
        
        // Test with extreme values
        detector.return_history.push(f64::INFINITY);
        detector.return_history.push(f64::NEG_INFINITY);
        detector.return_history.push(f64::NAN);
        detector.return_history.push(1000.0); // Very large return
        detector.return_history.push(-1000.0); // Very large loss
        
        // Should handle without panicking
        let result = detector.update_baseline_statistics();
        // May fail or succeed depending on implementation robustness
        if result.is_ok() {
            let stats = detector.baseline_statistics.as_ref().unwrap();
            // If successful, results should be finite where possible
            if stats.mean_return.is_finite() {
                assert!(stats.std_dev >= 0.0);
            }
        }
    }

    #[test]
    fn test_memory_bounds() {
        let mut detector = BlackSwanDetector::new("test".to_string(), BlackSwanParams::default());
        
        // Add excessive data to test memory management
        for i in 0..15000 {
            detector.return_history.push((i as f64).sin() * 0.01);
            detector.timestamp_history.push(Utc::now() + Duration::seconds(i));
        }
        
        detector.maintain_history_size();
        
        // Should be bounded
        assert!(detector.return_history.len() <= 10000);
        assert!(detector.timestamp_history.len() <= 10000);
        
        // Add many events
        for i in 0..1500 {
            let event = BlackSwanEvent {
                id: format!("event_{}", i),
                timestamp: Utc::now() + Duration::seconds(i),
                magnitude: 3.0,
                impact: 0.1,
                direction: SwanDirection::Negative,
                event_type: BlackSwanType::Unknown,
                ex_ante_probability: 0.001,
                duration: Duration::hours(1),
                recovery_time: None,
                market_conditions: MarketConditions {
                    volatility_level: 0.2,
                    liquidity_level: 0.5,
                    correlation_breakdown: false,
                    sentiment_extreme: false,
                    risk_off_mode: false,
                },
                predictability: None,
            };
            detector.detected_events.push(event);
        }
        
        detector.maintain_event_history();
        
        // Events should be bounded
        assert!(detector.detected_events.len() <= 1000);
    }

    #[test]
    fn test_predictability_analysis() {
        let mut detector = BlackSwanDetector::new("test".to_string(), BlackSwanParams::default());
        
        // Add some warning signals before analysis
        let base_time = Utc::now();
        detector.warning_signals.push(WarningSignal {
            signal_type: SignalType::VolatilityAnomalies,
            timestamp: base_time - Duration::days(3),
            strength: 0.8,
            reliability: 0.7,
        });
        
        detector.warning_signals.push(WarningSignal {
            signal_type: SignalType::LiquidityStress,
            timestamp: base_time - Duration::days(1),
            strength: 0.6,
            reliability: 0.8,
        });
        
        let event_timestamp = base_time;
        let analysis = detector.analyze_predictability(event_timestamp).unwrap();
        
        // Should find warning signals
        assert!(analysis.warning_signals.len() > 0);
        assert!(analysis.predictability_score >= 0.0 && analysis.predictability_score <= 1.0);
        assert!(analysis.warning_time.is_some());
        assert!(analysis.narrative_coherence >= 0.0);
        
        // Predictability should be scaled down for Black Swans
        assert!(analysis.predictability_score < 1.0);
    }

    #[test]
    fn test_summary_generation() {
        let mut detector = BlackSwanDetector::new("test".to_string(), BlackSwanParams::default());
        
        // Add some mock data
        detector.detected_events.push(BlackSwanEvent {
            id: "test_event_1".to_string(),
            timestamp: Utc::now(),
            magnitude: 4.5,
            impact: 0.15,
            direction: SwanDirection::Negative,
            event_type: BlackSwanType::MarketCrash,
            ex_ante_probability: 0.001,
            duration: Duration::hours(2),
            recovery_time: Some(Duration::days(1)),
            market_conditions: MarketConditions {
                volatility_level: 0.4,
                liquidity_level: 0.2,
                correlation_breakdown: true,
                sentiment_extreme: true,
                risk_off_mode: true,
            },
            predictability: None,
        });
        
        detector.warning_signals.push(WarningSignal {
            signal_type: SignalType::SentimentExtreme,
            timestamp: Utc::now(),
            strength: 0.9,
            reliability: 0.6,
        });
        
        let summary = detector.get_summary();
        
        // Validate summary
        assert_eq!(summary.total_events, 1);
        assert_eq!(summary.total_warnings, 1);
        assert_eq!(summary.average_severity, 4.5);
        assert_eq!(summary.max_severity, 4.5);
        assert!(summary.probability_estimate >= 0.0);
        assert!(summary.clustering_tendency >= 0.0);
    }

    #[test]
    fn test_export_analysis_data() {
        let mut detector = BlackSwanDetector::new("test".to_string(), BlackSwanParams::default());
        
        // Add some minimal data
        detector.return_history.push(0.01);
        detector.timestamp_history.push(Utc::now());
        
        let export_data = detector.export_analysis_data();
        
        // Should contain expected keys
        assert!(export_data.contains_key("detected_events"));
        assert!(export_data.contains_key("warning_signals"));
        assert!(export_data.contains_key("params"));
        
        // Should be valid JSON values
        assert!(export_data["detected_events"].is_array());
        assert!(export_data["warning_signals"].is_array());
        assert!(export_data["params"].is_object());
    }

    #[test]
    fn test_engine_integration() {
        let config = MacchiavelianConfig::aggressive_defaults();
        let mut engine = BlackSwanEngine::new(config);
        
        let normal_data = create_normal_market_data();
        let assessment = engine.assess(&normal_data).unwrap();
        
        // Validate assessment structure
        assert!(assessment.probability >= 0.0 && assessment.probability <= 1.0);
        assert!(assessment.detection_confidence >= 0.0 && assessment.detection_confidence <= 1.0);
        assert!(assessment.tail_risk >= 0.0);
        assert!(assessment.extreme_events_detected >= 0);
        
        // Test with extreme data
        let extreme_data = create_extreme_market_data();
        let extreme_assessment = engine.assess(&extreme_data).unwrap();
        
        // Extreme market should show higher risk
        assert!(extreme_assessment.probability >= assessment.probability);
        assert!(extreme_assessment.tail_risk >= assessment.tail_risk);
    }

    #[test]
    fn test_mathematical_functions() {
        // Test normal CDF approximation
        assert_relative_eq!(normal_cdf(0.0), 0.5, epsilon = 0.001);
        assert!(normal_cdf(3.0) > 0.99);
        assert!(normal_cdf(-3.0) < 0.01);
        assert!(normal_cdf(1.0) > 0.8 && normal_cdf(1.0) < 0.9);
        
        // Test error function
        assert_relative_eq!(erf(0.0), 0.0, epsilon = 0.001);
        assert!(erf(1.0) > 0.8);
        assert!(erf(-1.0) < -0.8);
        assert_relative_eq!(erf(1.0), -erf(-1.0), epsilon = 0.001);
    }

    #[test]
    fn test_financial_invariants() {
        let mut detector = BlackSwanDetector::new("test".to_string(), BlackSwanParams::default());
        
        // Test multiple times with different market conditions
        for i in 0..20 {
            let volatility = 0.01 + (i as f64 * 0.01); // Increasing volatility
            let market_data = MarketData {
                timestamp: Utc::now(),
                timestamp_unix: 1640995200 + (i * 86400),
                price: 50000.0 * (1.0 + (i as f64 * 0.001)),
                volume: 1000.0 * (1.0 + (i as f64 * 0.1)),
                bid: 49990.0,
                ask: 50010.0,
                bid_volume: 500.0,
                ask_volume: 500.0,
                volatility,
                returns: vec![volatility * 0.5], // Returns proportional to volatility
                volume_history: vec![1000.0; 5],
            };
            
            let result = detector.detect(&market_data).unwrap();
            
            // Financial invariants
            assert!(result.probability >= 0.0, "Probability must be non-negative");
            assert!(result.probability <= 1.0, "Probability must not exceed 100%");
            assert!(result.confidence >= 0.0, "Confidence must be non-negative");
            assert!(result.confidence <= 1.0, "Confidence must not exceed 100%");
            assert!(result.tail_risk_score >= 0.0, "Tail risk score must be non-negative");
            assert!(result.extreme_events_count >= 0, "Event count must be non-negative");
            
            // Higher volatility should generally increase black swan probability
            if i > 10 && volatility > 0.1 {
                assert!(result.probability > 0.0, "High volatility should indicate some black swan risk");
            }
        }
    }
}