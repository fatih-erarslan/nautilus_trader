//! Comprehensive TDD tests for KomodoDragonHunter
//! Following CQGS requirements: ZERO mocks, real implementations only
//! Sub-millisecond performance requirements
//! All components must be fully functional

use parasitic::organisms::komodo::KomodoDragonHunter;
use parasitic::detectors::volatility::VolatilityWoundDetector;
use parasitic::trackers::long_term::LongTermTracker;
use parasitic::strategies::slow_exploitation::SlowExploitationStrategy;
use parasitic::traits::{Organism, WoundDetector, Tracker, ExploitationStrategy};
use parasitic::{MarketData, PairData, Result, Error};
use std::time::{Duration, Instant};
use pretty_assertions::assert_eq;

/// Test data structure for real market conditions
#[derive(Debug, Clone)]
struct TestMarketData {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub price: f64,
    pub volume: f64,
    pub volatility: f64,
    pub bid: f64,
    pub ask: f64,
    pub spread_percent: f64,
}

impl TestMarketData {
    fn new_high_volatility() -> Self {
        Self {
            timestamp: chrono::Utc::now(),
            price: 50000.0,
            volume: 1000.0,
            volatility: 0.15, // 15% volatility - high wound potential
            bid: 49950.0,
            ask: 50050.0,
            spread_percent: 0.2, // 0.2% spread
        }
    }

    fn new_low_volatility() -> Self {
        Self {
            timestamp: chrono::Utc::now(),
            price: 50000.0,
            volume: 500.0,
            volatility: 0.02, // 2% volatility - low wound potential
            bid: 49990.0,
            ask: 50010.0,
            spread_percent: 0.04, // 0.04% spread
        }
    }

    fn new_wound_sequence() -> Vec<Self> {
        vec![
            Self::new_high_volatility(),
            Self::new_high_volatility(),
            Self::new_low_volatility(),
            Self::new_high_volatility(),
        ]
    }
}

#[cfg(test)]
mod komodo_dragon_hunter_tests {
    use super::*;

    #[test]
    fn test_komodo_dragon_hunter_construction() {
        let hunter = KomodoDragonHunter::new();
        
        assert!(hunter.is_ok(), \"KomodoDragonHunter should construct successfully\");
        
        let hunter = hunter.unwrap();
        assert_eq!(hunter.name(), \"KomodoDragonHunter\");
        assert_eq!(hunter.organism_type(), \"Predator\");
        assert!(hunter.is_active(), \"Hunter should be active by default\");
    }

    #[test]
    fn test_construction_performance() {
        let start = Instant::now();
        
        for _ in 0..1000 {
            let _hunter = KomodoDragonHunter::new().expect(\"Construction failed\");
        }
        
        let duration = start.elapsed();
        assert!(
            duration < Duration::from_micros(100),
            \"Construction should be under 100 microseconds for 1000 instances, got {:?}\",
            duration
        );
    }

    #[test] 
    fn test_volatility_wound_detection() {
        let hunter = KomodoDragonHunter::new().unwrap();
        let high_vol_data = TestMarketData::new_high_volatility();
        let low_vol_data = TestMarketData::new_low_volatility();

        // High volatility should be detected as a wound
        let high_wound_score = hunter.detect_wound(&high_vol_data);
        assert!(high_wound_score.is_ok(), \"Wound detection should succeed\");
        assert!(high_wound_score.unwrap() > 0.7, \"High volatility should score > 0.7\");

        // Low volatility should have lower wound score  
        let low_wound_score = hunter.detect_wound(&low_vol_data);
        assert!(low_wound_score.is_ok(), \"Wound detection should succeed\");
        assert!(low_wound_score.unwrap() < 0.3, \"Low volatility should score < 0.3\");
    }

    #[test]
    fn test_wound_detection_performance() {
        let hunter = KomodoDragonHunter::new().unwrap();
        let test_data = TestMarketData::new_high_volatility();
        
        let start = Instant::now();
        
        for _ in 0..10000 {
            let _score = hunter.detect_wound(&test_data).expect(\"Wound detection failed\");
        }
        
        let duration = start.elapsed();
        let per_detection = duration.as_nanos() / 10000;
        
        assert!(
            per_detection < 100_000, // Under 100 microseconds per detection
            \"Wound detection should be under 100 microseconds, got {} nanoseconds\",
            per_detection
        );
    }

    #[test]
    fn test_long_term_tracking() {
        let mut hunter = KomodoDragonHunter::new().unwrap();
        let wound_sequence = TestMarketData::new_wound_sequence();

        for (i, data) in wound_sequence.iter().enumerate() {
            let track_result = hunter.track_long_term(data, i as u64);
            assert!(track_result.is_ok(), \"Long-term tracking should succeed\");
        }

        // Verify tracking state persists
        let tracking_summary = hunter.get_tracking_summary();
        assert!(tracking_summary.is_ok(), \"Should be able to get tracking summary\");
        
        let summary = tracking_summary.unwrap();
        assert_eq!(summary.total_observations, 4);
        assert!(summary.avg_volatility > 0.0);
        assert!(summary.persistence_score > 0.0);
    }

    #[test]
    fn test_slow_exploitation_strategy() {
        let hunter = KomodoDragonHunter::new().unwrap();
        let high_wound_data = TestMarketData::new_high_volatility();
        
        // First, detect a wound
        let wound_score = hunter.detect_wound(&high_wound_data).unwrap();
        assert!(wound_score > 0.7, \"Need high wound score for exploitation test\");
        
        // Then test exploitation strategy
        let exploitation_plan = hunter.plan_exploitation(&high_wound_data, wound_score);
        assert!(exploitation_plan.is_ok(), \"Exploitation planning should succeed\");
        
        let plan = exploitation_plan.unwrap();
        assert!(plan.entry_size > 0.0, \"Should have positive entry size\");
        assert!(plan.time_horizon_ms > 1000, \"Should have reasonable time horizon\");
        assert!(plan.venom_intensity > 0.0 && plan.venom_intensity <= 1.0, \"Venom intensity should be normalized\");
    }

    #[test]
    fn test_venom_strategy_characteristics() {
        let hunter = KomodoDragonHunter::new().unwrap();
        let test_data = TestMarketData::new_high_volatility();
        
        let wound_score = hunter.detect_wound(&test_data).unwrap();
        let plan = hunter.plan_exploitation(&test_data, wound_score).unwrap();
        
        // Venom strategy should be slow and persistent
        assert!(plan.time_horizon_ms >= 5000, \"Venom should work slowly (>5s)\");
        assert!(plan.patience_factor > 0.8, \"Should be highly patient like real Komodo\");
        assert!(plan.persistence_weight > 0.7, \"Should be persistent\");
    }

    #[test]  
    fn test_simd_optimizations() {
        let hunter = KomodoDragonHunter::new().unwrap();
        
        // Test with arrays of market data for SIMD processing
        let market_data_array: Vec<TestMarketData> = (0..1000)
            .map(|_| TestMarketData::new_high_volatility())
            .collect();
            
        let start = Instant::now();
        
        let batch_results = hunter.detect_wounds_batch(&market_data_array);
        
        let duration = start.elapsed();
        
        assert!(batch_results.is_ok(), \"Batch wound detection should succeed\");
        assert_eq!(batch_results.unwrap().len(), 1000);
        
        // SIMD should provide significant speedup
        assert!(
            duration < Duration::from_millis(10),
            \"SIMD batch processing should be under 10ms for 1000 items, got {:?}\",
            duration
        );
    }

    #[test]
    fn test_real_time_adaptation() {
        let mut hunter = KomodoDragonHunter::new().unwrap();
        
        // Simulate changing market conditions
        let volatile_period = vec![
            TestMarketData::new_high_volatility(),
            TestMarketData::new_high_volatility(),
            TestMarketData::new_high_volatility(),
        ];
        
        let calm_period = vec![
            TestMarketData::new_low_volatility(),
            TestMarketData::new_low_volatility(),
            TestMarketData::new_low_volatility(),
        ];
        
        // Process volatile period
        for (i, data) in volatile_period.iter().enumerate() {
            hunter.adapt_to_conditions(data, i as u64).unwrap();
        }
        let volatile_sensitivity = hunter.get_sensitivity_level();
        
        // Process calm period  
        for (i, data) in calm_period.iter().enumerate() {
            hunter.adapt_to_conditions(data, (i + 3) as u64).unwrap();
        }
        let calm_sensitivity = hunter.get_sensitivity_level();
        
        // Hunter should adapt sensitivity based on conditions
        assert_ne!(volatile_sensitivity, calm_sensitivity, \"Sensitivity should adapt to market conditions\");
    }

    #[test]
    fn test_memory_efficiency() {
        use std::mem;
        
        let hunter = KomodoDragonHunter::new().unwrap();
        
        // Komodo hunter should be memory efficient
        let size = mem::size_of_val(&hunter);
        assert!(
            size < 1024, // Under 1KB
            \"KomodoDragonHunter should be under 1KB, got {} bytes\",
            size
        );
    }

    #[test]
    fn test_thread_safety() {
        use std::sync::Arc;
        use std::thread;
        
        let hunter = Arc::new(KomodoDragonHunter::new().unwrap());
        let test_data = TestMarketData::new_high_volatility();
        
        let handles: Vec<_> = (0..10)
            .map(|_| {
                let hunter_clone = Arc::clone(&hunter);
                let data_clone = test_data.clone();
                
                thread::spawn(move || {
                    hunter_clone.detect_wound(&data_clone)
                })
            })
            .collect();
            
        for handle in handles {
            let result = handle.join().unwrap();
            assert!(result.is_ok(), \"Thread-safe wound detection should succeed\");
        }
    }

    #[test]
    fn test_error_handling() {
        let hunter = KomodoDragonHunter::new().unwrap();
        
        // Test with invalid data
        let invalid_data = TestMarketData {
            timestamp: chrono::Utc::now(),
            price: -1.0, // Invalid negative price
            volume: 0.0,
            volatility: -0.1, // Invalid negative volatility
            bid: 0.0,
            ask: 0.0,
            spread_percent: -1.0,
        };
        
        let result = hunter.detect_wound(&invalid_data);
        assert!(result.is_err(), \"Should reject invalid market data\");
        
        match result.unwrap_err() {
            Error::InvalidData(msg) => {
                assert!(msg.contains(\"negative price\") || msg.contains(\"invalid\"));
            },
            _ => panic!(\"Should return InvalidData error\"),
        }
    }

    #[test]
    fn test_statistical_accuracy() {
        let hunter = KomodoDragonHunter::new().unwrap();
        
        // Generate statistically significant sample
        let high_vol_samples: Vec<TestMarketData> = (0..1000)
            .map(|_| TestMarketData::new_high_volatility())
            .collect();
            
        let low_vol_samples: Vec<TestMarketData> = (0..1000)
            .map(|_| TestMarketData::new_low_volatility())
            .collect();
        
        let high_scores: Vec<f64> = high_vol_samples.iter()
            .map(|data| hunter.detect_wound(data).unwrap())
            .collect();
            
        let low_scores: Vec<f64> = low_vol_samples.iter()
            .map(|data| hunter.detect_wound(data).unwrap())
            .collect();
        
        let high_avg: f64 = high_scores.iter().sum::<f64>() / high_scores.len() as f64;
        let low_avg: f64 = low_scores.iter().sum::<f64>() / low_scores.len() as f64;
        
        // Statistical separation should be clear
        assert!(high_avg > 0.6, \"High volatility average should be > 0.6\");
        assert!(low_avg < 0.4, \"Low volatility average should be < 0.4\");
        assert!(high_avg - low_avg > 0.3, \"Should have clear statistical separation\");
    }
}

#[cfg(test)]
mod volatility_wound_detector_tests {
    use super::*;

    #[test]
    fn test_volatility_detector_construction() {
        let detector = VolatilityWoundDetector::new();
        assert!(detector.is_ok(), \"VolatilityWoundDetector should construct successfully\");
    }

    #[test]
    fn test_volatility_threshold_detection() {
        let detector = VolatilityWoundDetector::new().unwrap();
        
        let high_vol_data = TestMarketData::new_high_volatility();
        let low_vol_data = TestMarketData::new_low_volatility();
        
        let high_score = detector.detect(&high_vol_data).unwrap();
        let low_score = detector.detect(&low_vol_data).unwrap();
        
        assert!(high_score > low_score, \"High volatility should score higher than low volatility\");
    }

    #[test]
    fn test_simd_volatility_processing() {
        let detector = VolatilityWoundDetector::new().unwrap();
        
        let data_batch = vec![
            TestMarketData::new_high_volatility(),
            TestMarketData::new_low_volatility(),
            TestMarketData::new_high_volatility(),
            TestMarketData::new_low_volatility(),
        ];
        
        let start = Instant::now();
        let results = detector.detect_batch(&data_batch).unwrap();
        let duration = start.elapsed();
        
        assert_eq!(results.len(), 4);
        assert!(duration < Duration::from_micros(50), \"SIMD batch should be under 50 microseconds\");
    }
}

#[cfg(test)]  
mod long_term_tracker_tests {
    use super::*;

    #[test]
    fn test_tracker_construction() {
        let tracker = LongTermTracker::new();
        assert!(tracker.is_ok(), \"LongTermTracker should construct successfully\");
    }

    #[test]
    fn test_persistence_tracking() {
        let mut tracker = LongTermTracker::new().unwrap();
        
        let persistent_data = vec![
            TestMarketData::new_high_volatility(),
            TestMarketData::new_high_volatility(),
            TestMarketData::new_high_volatility(),
        ];
        
        for (i, data) in persistent_data.iter().enumerate() {
            tracker.track(data, i as u64).unwrap();
        }
        
        let persistence = tracker.get_persistence_score();
        assert!(persistence.is_ok());
        assert!(persistence.unwrap() > 0.7, \"Should detect high persistence\");
    }

    #[test]
    fn test_memory_management() {
        let mut tracker = LongTermTracker::new().unwrap();
        
        // Add many data points to test memory management
        for i in 0..10000 {
            let data = TestMarketData::new_high_volatility();
            tracker.track(&data, i).unwrap();
        }
        
        // Memory should be managed efficiently
        let memory_usage = tracker.get_memory_usage();
        assert!(memory_usage < 1024 * 1024, \"Memory usage should be under 1MB\"); // 1MB limit
    }
}

#[cfg(test)]
mod slow_exploitation_strategy_tests {
    use super::*;

    #[test] 
    fn test_strategy_construction() {
        let strategy = SlowExploitationStrategy::new();
        assert!(strategy.is_ok(), \"SlowExploitationStrategy should construct successfully\");
    }

    #[test]
    fn test_venom_like_timing() {
        let strategy = SlowExploitationStrategy::new().unwrap();
        let data = TestMarketData::new_high_volatility();
        
        let plan = strategy.plan_exploitation(&data, 0.8).unwrap();
        
        // Venom should work slowly like real Komodo dragon
        assert!(plan.time_horizon_ms >= 5000, \"Venom effect should be slow (>5s)\");
        assert!(plan.patience_factor > 0.8, \"Should be very patient\");
    }

    #[test]
    fn test_gradual_intensity() {
        let strategy = SlowExploitationStrategy::new().unwrap();
        let data = TestMarketData::new_high_volatility();
        
        // Test different wound scores
        let weak_plan = strategy.plan_exploitation(&data, 0.3).unwrap();
        let strong_plan = strategy.plan_exploitation(&data, 0.9).unwrap();
        
        assert!(strong_plan.venom_intensity > weak_plan.venom_intensity);
        assert!(strong_plan.entry_size > weak_plan.entry_size);
    }
}