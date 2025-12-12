//! Comprehensive unit tests for Whale Detection
//! Tests volume spike detection, order book imbalance, and market impact analysis

use talebian_risk_rs::{
    whale_detection::*, MacchiavelianConfig, MarketData, TalebianRiskError,
    WhaleDetection, WhaleDirection
};
use approx::assert_relative_eq;

/// Test helper to create normal market data
fn create_normal_market_data() -> MarketData {
    MarketData {
        timestamp: chrono::Utc::now(),
        timestamp_unix: 1640995200,
        price: 50000.0,
        volume: 1000.0,
        bid: 49990.0,
        ask: 50010.0,
        bid_volume: 500.0,
        ask_volume: 500.0,
        volatility: 0.02,
        returns: vec![0.005, 0.008, -0.003, 0.01, 0.006],
        volume_history: vec![950.0, 1000.0, 1050.0, 980.0, 1020.0],
    }
}

/// Test helper to create whale activity market data
fn create_whale_market_data() -> MarketData {
    MarketData {
        timestamp: chrono::Utc::now(),
        timestamp_unix: 1640995200,
        price: 50100.0, // Price moved up
        volume: 5000.0, // 5x normal volume
        bid: 50080.0,
        ask: 50120.0,
        bid_volume: 2000.0, // Heavy buying
        ask_volume: 800.0,
        volatility: 0.04,
        returns: vec![0.02, 0.015, 0.025, 0.018, 0.03],
        volume_history: vec![1000.0, 1100.0, 1000.0, 950.0, 1000.0],
    }
}

/// Test helper to create selling whale market data
fn create_selling_whale_data() -> MarketData {
    MarketData {
        timestamp: chrono::Utc::now(),
        timestamp_unix: 1640995200,
        price: 49900.0, // Price moved down
        volume: 4500.0, // High volume
        bid: 49880.0,
        ask: 49920.0,
        bid_volume: 600.0,
        ask_volume: 2200.0, // Heavy selling
        volatility: 0.05,
        returns: vec![-0.02, -0.01, -0.025, -0.015, -0.02],
        volume_history: vec![1000.0, 1000.0, 1000.0, 1000.0, 1000.0],
    }
}

#[cfg(test)]
mod whale_detection_tests {
    use super::*;

    #[test]
    fn test_engine_creation() {
        let config = MacchiavelianConfig::aggressive_defaults();
        let engine = WhaleDetectionEngine::new(config.clone());
        
        assert_eq!(engine.config.whale_volume_threshold, config.whale_volume_threshold);
        assert_eq!(engine.detection_history.len(), 0);
    }

    #[test]
    fn test_normal_market_no_whale_detection() {
        let config = MacchiavelianConfig::aggressive_defaults();
        let mut engine = WhaleDetectionEngine::new(config);
        
        let market_data = create_normal_market_data();
        let result = engine.detect_whale_activity(&market_data);
        assert!(result.is_ok());
        
        let detection = result.unwrap();
        
        // Normal market should not trigger whale detection
        assert!(!detection.detected);
        assert!(!detection.is_whale_detected);
        assert!(detection.volume_spike < engine.config.whale_volume_threshold);
        assert!(detection.confidence < 0.5);
        assert_eq!(detection.timestamp_unix, market_data.timestamp_unix);
        assert!(matches!(detection.direction, WhaleDirection::Neutral));
    }

    #[test]
    fn test_whale_buying_detection() {
        let config = MacchiavelianConfig::aggressive_defaults();
        let mut engine = WhaleDetectionEngine::new(config);
        
        let whale_data = create_whale_market_data();
        let result = engine.detect_whale_activity(&whale_data);
        assert!(result.is_ok());
        
        let detection = result.unwrap();
        
        // Should detect whale activity
        assert!(detection.detected);
        assert!(detection.is_whale_detected);
        assert!(detection.volume_spike >= engine.config.whale_volume_threshold);
        assert!(detection.confidence > 0.5);
        assert!(matches!(detection.direction, WhaleDirection::Buying));
        
        // Volume spike calculation
        let avg_volume = whale_data.volume_history.iter().sum::<f64>() / whale_data.volume_history.len() as f64;
        let expected_spike = whale_data.volume / avg_volume;
        assert_relative_eq!(detection.volume_spike, expected_spike, epsilon = 0.001);
        
        // Order book imbalance (more bids than asks)
        assert!(detection.order_book_imbalance > 0.0);
        
        // Price impact should be positive
        assert!(detection.impact > 0.0);
        assert!(detection.price_impact > 0.0);
    }

    #[test]
    fn test_whale_selling_detection() {
        let config = MacchiavelianConfig::aggressive_defaults();
        let mut engine = WhaleDetectionEngine::new(config);
        
        let selling_data = create_selling_whale_data();
        let result = engine.detect_whale_activity(&selling_data);
        assert!(result.is_ok());
        
        let detection = result.unwrap();
        
        // Should detect whale selling
        assert!(detection.detected);
        assert!(detection.is_whale_detected);
        assert!(detection.volume_spike >= engine.config.whale_volume_threshold);
        assert!(matches!(detection.direction, WhaleDirection::Selling));
        
        // Order book imbalance (more asks than bids)
        assert!(detection.order_book_imbalance < 0.0);
        
        // Whale size should reflect large volume
        assert!(detection.whale_size > 3000.0);
    }

    #[test]
    fn test_confidence_calculation() {
        let config = MacchiavelianConfig::aggressive_defaults();
        let mut engine = WhaleDetectionEngine::new(config.clone());
        
        // Test various volume levels
        let volume_multipliers = vec![1.0, 2.0, 3.0, 5.0, 10.0];
        
        for multiplier in volume_multipliers {
            let mut market_data = create_normal_market_data();
            market_data.volume *= multiplier;
            
            let detection = engine.detect_whale_activity(&market_data).unwrap();
            
            if detection.detected {
                // Confidence should increase with volume spike size
                let volume_spike_excess = detection.volume_spike - config.whale_volume_threshold;
                let expected_confidence = (volume_spike_excess / config.whale_volume_threshold).min(0.95);
                assert_relative_eq!(detection.confidence, expected_confidence, epsilon = 0.01);
            } else {
                // Non-detected whales should have low confidence
                assert!(detection.confidence <= 0.2);
            }
            
            // Confidence should be bounded
            assert!(detection.confidence >= 0.0);
            assert!(detection.confidence <= 0.95);
        }
    }

    #[test]
    fn test_order_book_imbalance_calculation() {
        let config = MacchiavelianConfig::aggressive_defaults();
        let mut engine = WhaleDetectionEngine::new(config);
        
        // Test strong buying pressure
        let mut buying_data = create_normal_market_data();
        buying_data.bid_volume = 2000.0;
        buying_data.ask_volume = 500.0;
        buying_data.volume = 3000.0; // Trigger whale detection
        
        let buying_detection = engine.detect_whale_activity(&buying_data).unwrap();
        
        // Should show buying direction and positive imbalance
        assert!(matches!(buying_detection.direction, WhaleDirection::Buying));
        let expected_imbalance = (buying_data.bid_volume - buying_data.ask_volume) / 
                                (buying_data.bid_volume + buying_data.ask_volume);
        assert_relative_eq!(buying_detection.order_book_imbalance, expected_imbalance, epsilon = 0.001);
        assert!(buying_detection.order_book_imbalance > 0.5);
        
        // Test strong selling pressure
        let mut selling_data = create_normal_market_data();
        selling_data.bid_volume = 400.0;
        selling_data.ask_volume = 2200.0;
        selling_data.volume = 3200.0; // Trigger whale detection
        
        let selling_detection = engine.detect_whale_activity(&selling_data).unwrap();
        
        // Should show selling direction and negative imbalance
        assert!(matches!(selling_detection.direction, WhaleDirection::Selling));
        let expected_selling_imbalance = (selling_data.bid_volume - selling_data.ask_volume) / 
                                        (selling_data.bid_volume + selling_data.ask_volume);
        assert_relative_eq!(selling_detection.order_book_imbalance, expected_selling_imbalance, epsilon = 0.001);
        assert!(selling_detection.order_book_imbalance < -0.5);
    }

    #[test]
    fn test_price_impact_calculation() {
        let config = MacchiavelianConfig::aggressive_defaults();
        let mut engine = WhaleDetectionEngine::new(config);
        
        let market_data = create_whale_market_data();
        let detection = engine.detect_whale_activity(&market_data).unwrap();
        
        // Price impact should be spread relative to mid price
        let expected_spread = market_data.ask - market_data.bid;
        let expected_mid = (market_data.bid + market_data.ask) / 2.0;
        let expected_impact = expected_spread / expected_mid;
        
        assert_relative_eq!(detection.impact, expected_impact, epsilon = 0.0001);
        assert_relative_eq!(detection.price_impact, expected_impact, epsilon = 0.0001);
        assert!(detection.impact > 0.0);
    }

    #[test]
    fn test_detection_history_management() {
        let config = MacchiavelianConfig::aggressive_defaults();
        let mut engine = WhaleDetectionEngine::new(config);
        
        // Add many detections to test history management
        for i in 0..1200 {
            let mut market_data = create_whale_market_data();
            market_data.volume *= 1.0 + (i as f64 * 0.001); // Slight variations
            market_data.timestamp_unix += i as i64;
            
            engine.detect_whale_activity(&market_data).unwrap();
        }
        
        // History should be bounded
        assert!(engine.detection_history.len() <= 1000, \"Detection history should be bounded\");
        
        // Should still function correctly
        let new_detection = engine.detect_whale_activity(&create_whale_market_data());
        assert!(new_detection.is_ok());
    }

    #[test]
    fn test_whale_activity_summary() {
        let config = MacchiavelianConfig::aggressive_defaults();
        let mut engine = WhaleDetectionEngine::new(config);
        
        // Add a mix of whale and normal activity
        let test_data = vec![
            (create_whale_market_data(), true),      // Whale buying
            (create_selling_whale_data(), true),     // Whale selling
            (create_normal_market_data(), false),    // Normal activity
            (create_whale_market_data(), true),      // Another whale buying
            (create_normal_market_data(), false),    // Normal activity
        ];
        
        for (market_data, _expected_whale) in test_data {
            engine.detect_whale_activity(&market_data).unwrap();
        }
        
        let summary = engine.get_whale_activity_summary();
        
        // Should have detected some whales
        assert!(summary.total_detections > 0);
        assert!(summary.recent_activity_level >= 0.0 && summary.recent_activity_level <= 1.0);
        assert!(summary.confidence_avg >= 0.0 && summary.confidence_avg <= 1.0);
        
        // Should determine dominant direction based on detected whales
        assert!(matches!(summary.dominant_direction, 
            WhaleDirection::Buying | WhaleDirection::Selling | WhaleDirection::Neutral));
    }

    #[test]
    fn test_empty_history_summary() {
        let config = MacchiavelianConfig::aggressive_defaults();
        let engine = WhaleDetectionEngine::new(config);
        
        let summary = engine.get_whale_activity_summary();
        
        // Empty history should return zero values
        assert_eq!(summary.total_detections, 0);
        assert_eq!(summary.recent_activity_level, 0.0);
        assert_eq!(summary.confidence_avg, 0.0);
        assert!(matches!(summary.dominant_direction, WhaleDirection::Neutral));
    }

    #[test]
    fn test_recent_activity_level() {
        let config = MacchiavelianConfig::aggressive_defaults();
        let mut engine = WhaleDetectionEngine::new(config);
        
        // Add 15 detections, with 8 being whale activity
        for i in 0..15 {
            let market_data = if i % 2 == 0 {
                create_whale_market_data() // Whale activity
            } else {
                create_normal_market_data() // Normal activity
            };
            
            engine.detect_whale_activity(&market_data).unwrap();
        }
        
        let summary = engine.get_whale_activity_summary();
        
        // Recent activity level should consider last 10 detections
        // With our pattern, we should have whale activity in about half of recent detections
        assert!(summary.recent_activity_level >= 0.0);
        assert!(summary.recent_activity_level <= 1.0);
        
        // Given our alternating pattern, recent activity should be significant
        if summary.total_detections > 5 {
            assert!(summary.recent_activity_level > 0.0);
        }
    }

    #[test]
    fn test_dominant_direction_calculation() {
        let config = MacchiavelianConfig::aggressive_defaults();
        let mut engine = WhaleDetectionEngine::new(config);
        
        // Add more buying whale activity than selling
        for _ in 0..5 {
            engine.detect_whale_activity(&create_whale_market_data()).unwrap(); // Buying
        }
        
        for _ in 0..2 {
            engine.detect_whale_activity(&create_selling_whale_data()).unwrap(); // Selling
        }
        
        let summary = engine.get_whale_activity_summary();
        
        // Should show buying as dominant direction
        assert!(matches!(summary.dominant_direction, WhaleDirection::Buying));
        assert!(summary.total_detections >= 5); // Should have detected multiple whales
    }

    #[test]
    fn test_edge_case_zero_volume() {
        let config = MacchiavelianConfig::aggressive_defaults();
        let mut engine = WhaleDetectionEngine::new(config);
        
        let mut zero_volume_data = create_normal_market_data();
        zero_volume_data.volume = 0.0;
        zero_volume_data.bid_volume = 0.0;
        zero_volume_data.ask_volume = 0.0;
        
        let result = engine.detect_whale_activity(&zero_volume_data);
        assert!(result.is_ok());
        
        let detection = result.unwrap();
        
        // Should not detect whale with zero volume
        assert!(!detection.detected);
        assert!(!detection.is_whale_detected);
        assert_eq!(detection.whale_size, 0.0);
        assert!(matches!(detection.direction, WhaleDirection::Neutral));
    }

    #[test]
    fn test_edge_case_empty_volume_history() {
        let config = MacchiavelianConfig::aggressive_defaults();
        let mut engine = WhaleDetectionEngine::new(config);
        
        let mut no_history_data = create_whale_market_data();
        no_history_data.volume_history = vec![]; // Empty history
        
        let result = engine.detect_whale_activity(&no_history_data);
        assert!(result.is_ok());
        
        let detection = result.unwrap();
        
        // Should handle empty history gracefully
        // With no history, average volume defaults to 1 (max of len(), 1)
        assert!(detection.volume_spike > 0.0);
        // May or may not detect whale depending on implementation details
    }

    #[test]
    fn test_edge_case_extreme_values() {
        let config = MacchiavelianConfig::aggressive_defaults();
        let mut engine = WhaleDetectionEngine::new(config);
        
        let mut extreme_data = create_normal_market_data();
        extreme_data.volume = f64::INFINITY;
        extreme_data.bid_volume = f64::INFINITY;
        extreme_data.ask_volume = 1000.0;
        extreme_data.bid = f64::NAN;
        extreme_data.ask = f64::NAN;
        
        let result = engine.detect_whale_activity(&extreme_data);
        
        // Should either handle gracefully or return error
        if let Ok(detection) = result {
            // If successful, should produce finite results where possible
            if detection.volume_spike.is_finite() {
                assert!(detection.volume_spike >= 0.0);
            }
            
            if detection.confidence.is_finite() {
                assert!(detection.confidence >= 0.0);
                assert!(detection.confidence <= 1.0);
            }
        }
    }

    #[test]
    fn test_confidence_bounds() {
        let config = MacchiavelianConfig::aggressive_defaults();
        let mut engine = WhaleDetectionEngine::new(config.clone());
        
        // Test with various volume spikes
        let volume_spikes = vec![1.0, 2.0, 5.0, 10.0, 100.0];
        
        for spike_multiplier in volume_spikes {
            let mut market_data = create_normal_market_data();
            market_data.volume *= spike_multiplier;
            
            let detection = engine.detect_whale_activity(&market_data).unwrap();
            
            // Confidence should always be bounded
            assert!(detection.confidence >= 0.0, \"Confidence must be non-negative\");
            assert!(detection.confidence <= 0.95, \"Confidence should be capped at 0.95\");
            
            if detection.detected {
                // For detected whales, confidence should be positive
                assert!(detection.confidence > 0.0, \"Detected whales should have positive confidence\");
                
                // High volume spikes should result in high confidence
                if spike_multiplier >= config.whale_volume_threshold * 2.0 {
                    assert!(detection.confidence > 0.5, \"Large volume spikes should have high confidence\");
                }
            }
        }
    }

    #[test]
    fn test_direction_classification_accuracy() {
        let config = MacchiavelianConfig::aggressive_defaults();
        let mut engine = WhaleDetectionEngine::new(config);
        
        // Test strong buying (bid volume >> ask volume)
        let mut strong_buying = create_normal_market_data();
        strong_buying.bid_volume = 3000.0;
        strong_buying.ask_volume = 1000.0;
        strong_buying.volume = 5000.0;
        
        let buying_detection = engine.detect_whale_activity(&strong_buying).unwrap();
        assert!(matches!(buying_detection.direction, WhaleDirection::Buying));
        
        // Test strong selling (ask volume >> bid volume)
        let mut strong_selling = create_normal_market_data();
        strong_selling.bid_volume = 800.0;
        strong_selling.ask_volume = 2400.0; // 3x bid volume
        strong_selling.volume = 4000.0;
        
        let selling_detection = engine.detect_whale_activity(&strong_selling).unwrap();
        assert!(matches!(selling_detection.direction, WhaleDirection::Selling));
        
        // Test balanced (similar bid and ask volumes)
        let mut balanced = create_normal_market_data();
        balanced.bid_volume = 1500.0;
        balanced.ask_volume = 1400.0; // Close to bid volume
        balanced.volume = 3000.0;
        
        let balanced_detection = engine.detect_whale_activity(&balanced).unwrap();
        assert!(matches!(balanced_detection.direction, WhaleDirection::Neutral));
    }

    #[test]
    fn test_whale_size_accuracy() {
        let config = MacchiavelianConfig::aggressive_defaults();
        let mut engine = WhaleDetectionEngine::new(config);
        
        let volumes = vec![1000.0, 2500.0, 5000.0, 10000.0];
        
        for volume in volumes {
            let mut market_data = create_normal_market_data();
            market_data.volume = volume;
            
            let detection = engine.detect_whale_activity(&market_data).unwrap();
            
            // Whale size should equal the volume
            assert_relative_eq!(detection.whale_size, volume, epsilon = 0.001);
        }
    }

    #[test]
    fn test_aggressive_vs_conservative_thresholds() {
        let aggressive_config = MacchiavelianConfig::aggressive_defaults();
        let conservative_config = MacchiavelianConfig::conservative_baseline();
        
        let mut aggressive_engine = WhaleDetectionEngine::new(aggressive_config.clone());
        let mut conservative_engine = WhaleDetectionEngine::new(conservative_config.clone());
        
        // Test with moderate volume spike
        let mut moderate_spike_data = create_normal_market_data();
        moderate_spike_data.volume = 2500.0; // 2.5x normal volume
        
        let aggressive_detection = aggressive_engine.detect_whale_activity(&moderate_spike_data).unwrap();
        let conservative_detection = conservative_engine.detect_whale_activity(&moderate_spike_data).unwrap();
        
        // Aggressive should be more likely to detect whales
        assert!(aggressive_config.whale_volume_threshold <= conservative_config.whale_volume_threshold);
        
        // With same data, aggressive might detect where conservative doesn't
        if aggressive_detection.detected && !conservative_detection.detected {
            assert!(aggressive_detection.volume_spike >= aggressive_config.whale_volume_threshold);
            assert!(conservative_detection.volume_spike < conservative_config.whale_volume_threshold);
        }
    }

    #[test]
    fn test_concurrent_safety_simulation() {
        use std::sync::{Arc, Mutex};
        use std::thread;
        
        let config = MacchiavelianConfig::aggressive_defaults();
        let engine = Arc::new(Mutex::new(WhaleDetectionEngine::new(config)));
        let mut handles = vec![];
        
        // Simulate concurrent whale detection
        for i in 0..10 {
            let engine_clone = Arc::clone(&engine);
            let handle = thread::spawn(move || {
                let mut market_data = create_whale_market_data();
                market_data.volume *= 1.0 + (i as f64 * 0.1);
                market_data.timestamp_unix += i as i64;
                
                let mut engine_guard = engine_clone.lock().unwrap();
                engine_guard.detect_whale_activity(&market_data).unwrap()
            });
            handles.push(handle);
        }
        
        // Collect results
        let mut results = vec![];
        for handle in handles {
            results.push(handle.join().unwrap());
        }
        
        assert_eq!(results.len(), 10);
        
        // Verify all results are valid
        for detection in results {
            assert!(detection.confidence >= 0.0);
            assert!(detection.confidence <= 1.0);
            assert!(detection.volume_spike >= 0.0);
        }
        
        // Check engine state
        let engine_guard = engine.lock().unwrap();
        assert_eq!(engine_guard.detection_history.len(), 10);
    }

    #[test]
    fn test_financial_invariants() {
        let config = MacchiavelianConfig::aggressive_defaults();
        let mut engine = WhaleDetectionEngine::new(config);
        
        // Test with various market scenarios
        let scenarios = vec![
            create_normal_market_data(),
            create_whale_market_data(),
            create_selling_whale_data(),
        ];
        
        for market_data in scenarios {
            let detection = engine.detect_whale_activity(&market_data).unwrap();
            
            // Financial invariants
            assert!(detection.volume_spike >= 0.0, \"Volume spike must be non-negative\");
            assert!(detection.confidence >= 0.0, \"Confidence must be non-negative\");
            assert!(detection.confidence <= 1.0, \"Confidence must not exceed 100%\");
            assert!(detection.whale_size >= 0.0, \"Whale size must be non-negative\");
            assert!(detection.impact >= 0.0, \"Impact must be non-negative\");
            assert!(detection.price_impact >= 0.0, \"Price impact must be non-negative\");
            assert!(detection.order_book_imbalance >= -1.0 && detection.order_book_imbalance <= 1.0, 
                    \"Order book imbalance must be between -1 and 1\");
            
            // Consistency checks
            assert_eq!(detection.detected, detection.is_whale_detected, \"Detection flags should be consistent\");
            assert_eq!(detection.whale_size, market_data.volume, \"Whale size should equal volume\");
            
            // If whale is detected, volume spike should exceed threshold
            if detection.detected {
                assert!(detection.volume_spike >= engine.config.whale_volume_threshold, 
                        \"Detected whales must exceed volume threshold\");
                assert!(detection.confidence > 0.0, \"Detected whales must have positive confidence\");
            }
            
            // Direction consistency with order book imbalance
            match detection.direction {
                WhaleDirection::Buying => assert!(detection.order_book_imbalance > 0.2, \"Buying should show bid imbalance\"),
                WhaleDirection::Selling => assert!(detection.order_book_imbalance < -0.2, \"Selling should show ask imbalance\"),
                WhaleDirection::Neutral => assert!(detection.order_book_imbalance.abs() <= 0.2, \"Neutral should show balanced book\"),
            }
        }
    }
}