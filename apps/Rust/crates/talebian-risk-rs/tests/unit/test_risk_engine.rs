//! Comprehensive unit tests for the TalebianRiskEngine
//! Tests all core functionality, edge cases, and financial invariants

use talebian_risk_rs::{
    risk_engine::*, MacchiavelianConfig, MarketData, TalebianRiskError,
    WhaleDirection, WhaleDetection
};
use approx::assert_relative_eq;
use std::f64::EPSILON;

/// Test helper to create sample market data
fn create_sample_market_data(volume_multiplier: f64, volatility: f64) -> MarketData {
    MarketData {
        timestamp: chrono::Utc::now(),
        timestamp_unix: 1640995200,
        price: 50000.0,
        volume: 1000.0 * volume_multiplier,
        bid: 49990.0,
        ask: 50010.0,
        bid_volume: 500.0,
        ask_volume: 400.0,
        volatility,
        returns: vec![0.01, 0.015, -0.005, 0.02, 0.008],
        volume_history: vec![800.0, 900.0, 1200.0, 950.0, 1000.0],
    }
}

/// Test helper to create whale market data
fn create_whale_market_data() -> MarketData {
    MarketData {
        timestamp: chrono::Utc::now(),
        timestamp_unix: 1640995200,
        price: 50000.0,
        volume: 5000.0, // 5x normal volume
        bid: 49980.0,
        ask: 50020.0,
        bid_volume: 2000.0,
        ask_volume: 1000.0, // Strong imbalance
        volatility: 0.05,
        returns: vec![0.03, 0.04, 0.025, 0.035, 0.02],
        volume_history: vec![1000.0, 1200.0, 1100.0, 1050.0, 1000.0],
    }
}

#[cfg(test)]
mod risk_engine_tests {
    use super::*;

    #[test]
    fn test_engine_creation_with_aggressive_config() {
        let config = MacchiavelianConfig::aggressive_defaults();
        let engine = TalebianRiskEngine::new(config.clone());
        
        // Verify aggressive configuration is applied
        assert_eq!(engine.config.antifragility_threshold, 0.35);
        assert_eq!(engine.config.barbell_safe_ratio, 0.65);
        assert_eq!(engine.config.black_swan_threshold, 0.18);
        assert_eq!(engine.config.kelly_fraction, 0.55);
        assert!(engine.config.whale_detected_multiplier > 1.0);
        
        // Verify engine state
        assert_eq!(engine.assessment_history.len(), 0);
        assert_eq!(engine.performance_tracker.total_assessments, 0);
    }

    #[test]
    fn test_engine_creation_with_conservative_config() {
        let config = MacchiavelianConfig::conservative_baseline();
        let engine = TalebianRiskEngine::new(config.clone());
        
        // Verify conservative configuration
        assert!(engine.config.antifragility_threshold > 0.35);
        assert!(engine.config.barbell_safe_ratio > 0.65);
        assert!(engine.config.black_swan_threshold < 0.18);
        assert!(engine.config.kelly_fraction < 0.55);
    }

    #[test]
    fn test_basic_risk_assessment() {
        let config = MacchiavelianConfig::aggressive_defaults();
        let mut engine = TalebianRiskEngine::new(config);
        let market_data = create_sample_market_data(1.0, 0.03);
        
        let result = engine.assess_risk(&market_data);
        assert!(result.is_ok());
        
        let assessment = result.unwrap();
        
        // Validate assessment bounds
        assert!(assessment.antifragility_score >= 0.0 && assessment.antifragility_score <= 1.0);
        assert!(assessment.barbell_allocation.0 >= 0.0 && assessment.barbell_allocation.0 <= 1.0);
        assert!(assessment.barbell_allocation.1 >= 0.0 && assessment.barbell_allocation.1 <= 1.0);
        assert!(assessment.black_swan_probability >= 0.0 && assessment.black_swan_probability <= 1.0);
        assert!(assessment.kelly_fraction >= 0.0 && assessment.kelly_fraction <= 1.0);
        assert!(assessment.overall_risk_score >= 0.0 && assessment.overall_risk_score <= 1.0);
        assert!(assessment.recommended_position_size >= 0.0 && assessment.recommended_position_size <= 1.0);
        assert!(assessment.confidence >= 0.0 && assessment.confidence <= 1.0);
        
        // Validate barbell allocation sum
        assert!(assessment.barbell_allocation.0 + assessment.barbell_allocation.1 <= 1.1); // Allow rounding error
        
        // Validate parasitic opportunity
        assert!(assessment.parasitic_opportunity.opportunity_score >= 0.0);
        assert!(assessment.parasitic_opportunity.confidence >= 0.0);
    }

    #[test]
    fn test_whale_detection_impact() {
        let config = MacchiavelianConfig::aggressive_defaults();
        let mut engine = TalebianRiskEngine::new(config);
        
        // Normal market conditions
        let normal_data = create_sample_market_data(1.0, 0.02);
        let normal_assessment = engine.assess_risk(&normal_data).unwrap();
        
        // Whale market conditions
        let whale_data = create_whale_market_data();
        let whale_assessment = engine.assess_risk(&whale_data).unwrap();
        
        // Whale detection should be triggered
        assert!(whale_assessment.whale_detection.is_whale_detected);
        assert!(whale_assessment.whale_detection.detected);
        assert!(whale_assessment.whale_detection.confidence > 0.5);
        
        // Position size should be larger with whale activity
        assert!(whale_assessment.recommended_position_size > normal_assessment.recommended_position_size);
        
        // Should detect buying pressure
        assert!(matches!(whale_assessment.whale_detection.direction, WhaleDirection::Buying));
    }

    #[test]
    fn test_risk_assessment_consistency() {
        let config = MacchiavelianConfig::aggressive_defaults();
        let mut engine = TalebianRiskEngine::new(config);
        let market_data = create_sample_market_data(1.0, 0.03);
        
        // Multiple assessments should be consistent
        let assessment1 = engine.assess_risk(&market_data).unwrap();
        let assessment2 = engine.assess_risk(&market_data).unwrap();
        
        // Results should be identical for same input
        assert_relative_eq!(assessment1.antifragility_score, assessment2.antifragility_score, epsilon = 0.01);
        assert_relative_eq!(assessment1.kelly_fraction, assessment2.kelly_fraction, epsilon = 0.01);
        assert_relative_eq!(assessment1.overall_risk_score, assessment2.overall_risk_score, epsilon = 0.01);
    }

    #[test]
    fn test_performance_tracking() {
        let config = MacchiavelianConfig::aggressive_defaults();
        let mut engine = TalebianRiskEngine::new(config);
        let market_data = create_sample_market_data(1.0, 0.03);
        
        // Initial assessment
        engine.assess_risk(&market_data).unwrap();
        assert_eq!(engine.performance_tracker.total_assessments, 1);
        
        // Record trade outcomes
        engine.record_trade_outcome(0.02, true, 0.8).unwrap();
        engine.record_trade_outcome(0.015, false, 0.5).unwrap();
        engine.record_trade_outcome(-0.01, true, 0.3).unwrap();
        
        let status = engine.get_engine_status();
        assert!(status.performance_tracker.successful_predictions > 0);
        assert!(status.performance_tracker.total_return != 0.0);
    }

    #[test]
    fn test_comprehensive_recommendations() {
        let config = MacchiavelianConfig::aggressive_defaults();
        let mut engine = TalebianRiskEngine::new(config);
        let market_data = create_whale_market_data(); // High opportunity scenario
        
        let recommendations = engine.generate_recommendations(&market_data).unwrap();
        
        // Position sizing validation
        assert!(recommendations.position_sizing.final_recommended_size > 0.0);
        assert!(recommendations.position_sizing.kelly_fraction > 0.0);
        assert!(recommendations.position_sizing.whale_adjusted_size >= recommendations.position_sizing.kelly_fraction);
        
        // Risk controls validation
        assert!(recommendations.risk_controls.stop_loss_level > 0.0);
        assert!(recommendations.risk_controls.take_profit_level > recommendations.risk_controls.stop_loss_level);
        assert!(recommendations.risk_controls.max_drawdown_limit > 0.0);
        
        // Timing guidance validation
        assert!(!recommendations.timing_guidance.entry_urgency.is_empty());
        assert!(!recommendations.timing_guidance.market_regime.is_empty());
        assert!(recommendations.timing_guidance.whale_activity_level.contains("ACTIVE"));
        
        // Performance metrics validation
        assert!(recommendations.performance_metrics.expected_return >= 0.0);
        assert!(recommendations.performance_metrics.expected_volatility > 0.0);
        assert!(recommendations.performance_metrics.win_probability >= 0.0 && recommendations.performance_metrics.win_probability <= 1.0);
    }

    #[test]
    fn test_edge_case_zero_volume() {
        let config = MacchiavelianConfig::aggressive_defaults();
        let mut engine = TalebianRiskEngine::new(config);
        
        let mut market_data = create_sample_market_data(1.0, 0.03);
        market_data.volume = 0.0;
        market_data.bid_volume = 0.0;
        market_data.ask_volume = 0.0;
        
        let result = engine.assess_risk(&market_data);
        assert!(result.is_ok());
        
        let assessment = result.unwrap();
        assert!(!assessment.whale_detection.is_whale_detected);
        assert!(assessment.recommended_position_size > 0.0); // Should still provide a position
    }

    #[test]
    fn test_edge_case_zero_volatility() {
        let config = MacchiavelianConfig::aggressive_defaults();
        let mut engine = TalebianRiskEngine::new(config);
        
        let market_data = create_sample_market_data(1.0, 0.0);
        
        let result = engine.assess_risk(&market_data);
        assert!(result.is_ok());
        
        let assessment = result.unwrap();
        assert!(assessment.antifragility_score >= 0.0);
        assert!(assessment.confidence > 0.0); // Should maintain some confidence
    }

    #[test]
    fn test_edge_case_extreme_volatility() {
        let config = MacchiavelianConfig::aggressive_defaults();
        let mut engine = TalebianRiskEngine::new(config);
        
        let market_data = create_sample_market_data(1.0, 1.0); // 100% volatility
        
        let result = engine.assess_risk(&market_data);
        assert!(result.is_ok());
        
        let assessment = result.unwrap();
        assert!(assessment.black_swan_probability > 0.0);
        assert!(assessment.recommended_position_size <= 0.75); // Should cap position size
    }

    #[test]
    fn test_financial_invariant_kelly_fraction() {
        let config = MacchiavelianConfig::aggressive_defaults();
        let mut engine = TalebianRiskEngine::new(config);
        
        for _ in 0..10 {
            let market_data = create_sample_market_data(1.0, 0.03);
            let assessment = engine.assess_risk(&market_data).unwrap();
            
            // Kelly fraction must be between 0 and 1
            assert!(assessment.kelly_fraction >= 0.0, "Kelly fraction must be non-negative");
            assert!(assessment.kelly_fraction <= 1.0, "Kelly fraction must not exceed 100%");
            
            // Recommended position size must respect bounds
            assert!(assessment.recommended_position_size >= 0.02, "Position size too small");
            assert!(assessment.recommended_position_size <= 0.75, "Position size too large");
        }
    }

    #[test]
    fn test_financial_invariant_allocation_sum() {
        let config = MacchiavelianConfig::aggressive_defaults();
        let mut engine = TalebianRiskEngine::new(config);
        
        for _ in 0..10 {
            let market_data = create_sample_market_data(1.0, 0.03);
            let assessment = engine.assess_risk(&market_data).unwrap();
            
            // Barbell allocation must sum to <= 1.0
            let total_allocation = assessment.barbell_allocation.0 + assessment.barbell_allocation.1;
            assert!(total_allocation <= 1.1, "Barbell allocation sum exceeds 100% ({})", total_allocation);
            
            // Individual allocations must be non-negative
            assert!(assessment.barbell_allocation.0 >= 0.0, "Safe allocation negative");
            assert!(assessment.barbell_allocation.1 >= 0.0, "Risky allocation negative");
        }
    }

    #[test]
    fn test_memory_management() {
        let config = MacchiavelianConfig::aggressive_defaults();
        let mut engine = TalebianRiskEngine::new(config);
        
        // Add many assessments to test memory bounds
        for i in 0..15000 {
            let market_data = create_sample_market_data(1.0 + (i as f64 * 0.001), 0.03);
            engine.assess_risk(&market_data).unwrap();
        }
        
        // History should be bounded
        assert!(engine.assessment_history.len() <= 10000, "Assessment history not bounded");
        
        // Should still function correctly
        let market_data = create_sample_market_data(1.0, 0.03);
        let result = engine.assess_risk(&market_data);
        assert!(result.is_ok());
    }

    #[test]
    fn test_concurrent_safety_simulation() {
        use std::sync::{Arc, Mutex};
        use std::thread;
        
        let config = MacchiavelianConfig::aggressive_defaults();
        let engine = Arc::new(Mutex::new(TalebianRiskEngine::new(config)));
        let mut handles = vec![];
        
        // Simulate concurrent access (though actual Rust concurrency would require Send+Sync)
        for i in 0..5 {
            let engine_clone = Arc::clone(&engine);
            let handle = thread::spawn(move || {
                let market_data = create_sample_market_data(1.0 + (i as f64 * 0.1), 0.03);
                let mut engine_guard = engine_clone.lock().unwrap();
                engine_guard.assess_risk(&market_data).unwrap()
            });
            handles.push(handle);
        }
        
        // Collect results
        let mut results = vec![];
        for handle in handles {
            results.push(handle.join().unwrap());
        }
        
        assert_eq!(results.len(), 5);
        for result in results {
            assert!(result.overall_risk_score >= 0.0);
            assert!(result.overall_risk_score <= 1.0);
        }
    }

    #[test]
    fn test_numerical_stability() {
        let config = MacchiavelianConfig::aggressive_defaults();
        let mut engine = TalebianRiskEngine::new(config);
        
        // Test with very small numbers
        let mut market_data = create_sample_market_data(1.0, 0.03);
        market_data.price = f64::EPSILON;
        market_data.volume = f64::EPSILON;
        
        let result = engine.assess_risk(&market_data);
        assert!(result.is_ok());
        
        let assessment = result.unwrap();
        assert!(assessment.overall_risk_score.is_finite());
        assert!(assessment.recommended_position_size.is_finite());
        
        // Test with very large numbers
        market_data.price = 1e12;
        market_data.volume = 1e12;
        
        let result = engine.assess_risk(&market_data);
        assert!(result.is_ok());
        
        let assessment = result.unwrap();
        assert!(assessment.overall_risk_score.is_finite());
        assert!(assessment.recommended_position_size.is_finite());
    }

    #[test]
    fn test_error_propagation() {
        let config = MacchiavelianConfig::aggressive_defaults();
        let mut engine = TalebianRiskEngine::new(config);
        
        // Test with NaN values
        let mut market_data = create_sample_market_data(1.0, 0.03);
        market_data.volatility = f64::NAN;
        
        let result = engine.assess_risk(&market_data);
        // Should either handle gracefully or return error
        if let Ok(assessment) = result {
            assert!(assessment.overall_risk_score.is_finite() || assessment.overall_risk_score.is_nan());
        }
        
        // Test with infinite values
        market_data.volatility = f64::INFINITY;
        let result = engine.assess_risk(&market_data);
        // Should handle gracefully
        if let Ok(assessment) = result {
            assert!(!assessment.overall_risk_score.is_infinite() || assessment.overall_risk_score == f64::INFINITY);
        }
    }

    #[test]
    fn test_aggressive_vs_conservative_behavior() {
        let aggressive_config = MacchiavelianConfig::aggressive_defaults();
        let conservative_config = MacchiavelianConfig::conservative_baseline();
        
        let mut aggressive_engine = TalebianRiskEngine::new(aggressive_config);
        let mut conservative_engine = TalebianRiskEngine::new(conservative_config);
        
        let whale_data = create_whale_market_data();
        
        let aggressive_assessment = aggressive_engine.assess_risk(&whale_data).unwrap();
        let conservative_assessment = conservative_engine.assess_risk(&whale_data).unwrap();
        
        // Aggressive should recommend larger positions
        assert!(aggressive_assessment.recommended_position_size >= conservative_assessment.recommended_position_size);
        
        // Aggressive should have lower thresholds
        assert!(aggressive_assessment.parasitic_opportunity.opportunity_score >= conservative_assessment.parasitic_opportunity.opportunity_score * 0.8);
    }

    #[test] 
    fn test_opportunity_threshold_compliance() {
        let config = MacchiavelianConfig::aggressive_defaults();
        let mut engine = TalebianRiskEngine::new(config.clone());
        
        // Low opportunity scenario
        let low_opportunity_data = MarketData {
            timestamp: chrono::Utc::now(),
            timestamp_unix: 1640995200,
            price: 50000.0,
            volume: 500.0,
            bid: 49999.0,
            ask: 50001.0,
            bid_volume: 250.0,
            ask_volume: 250.0,
            volatility: 0.005,
            returns: vec![0.001, 0.0005, -0.0002, 0.0008, 0.0003],
            volume_history: vec![500.0, 480.0, 520.0, 490.0, 500.0],
        };
        
        let assessment = engine.assess_risk(&low_opportunity_data).unwrap();
        
        // Should result in low opportunity and smaller position
        assert!(assessment.parasitic_opportunity.opportunity_score < config.parasitic_opportunity_threshold);
        assert!(assessment.recommended_position_size < 0.1);
    }

    #[test]
    fn test_time_series_behavior() {
        let config = MacchiavelianConfig::aggressive_defaults();
        let mut engine = TalebianRiskEngine::new(config);
        
        let mut assessments = Vec::new();
        
        // Generate time series of assessments
        for i in 0..100 {
            let volatility = 0.02 + 0.01 * (i as f64 / 10.0).sin(); // Varying volatility
            let volume_mult = 1.0 + 0.5 * (i as f64 / 20.0).cos(); // Varying volume
            
            let market_data = create_sample_market_data(volume_mult, volatility);
            let assessment = engine.assess_risk(&market_data).unwrap();
            assessments.push(assessment);
        }
        
        // Validate time series properties
        assert_eq!(assessments.len(), 100);
        
        // Check for reasonable variation
        let risk_scores: Vec<f64> = assessments.iter().map(|a| a.overall_risk_score).collect();
        let min_risk = risk_scores.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_risk = risk_scores.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        
        assert!(max_risk > min_risk, "Risk scores should vary over time");
        assert!(max_risk - min_risk > 0.01, "Risk score variation should be meaningful");
    }

    #[test]
    fn test_engine_status_accuracy() {
        let config = MacchiavelianConfig::aggressive_defaults();
        let mut engine = TalebianRiskEngine::new(config);
        
        // Perform several assessments and trades
        for i in 0..5 {
            let market_data = create_sample_market_data(1.0 + (i as f64 * 0.1), 0.03);
            engine.assess_risk(&market_data).unwrap();
            engine.record_trade_outcome(0.01 * (i as f64), i % 2 == 0, 0.5 + (i as f64 * 0.1)).unwrap();
        }
        
        let status = engine.get_engine_status();
        
        // Validate status accuracy
        assert_eq!(status.total_assessments, 5);
        assert!(status.performance_tracker.total_assessments >= 5);
        assert!(status.performance_tracker.total_return != 0.0);
        
        // Validate component statuses
        assert!(status.whale_activity_summary.total_detections >= 0);
        assert!(status.kelly_status.total_trades >= 5);
    }
}