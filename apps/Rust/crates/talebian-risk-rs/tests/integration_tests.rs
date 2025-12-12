//! # Integration Tests for Talebian Risk Management
//!
//! Comprehensive integration tests validating the complete aggressive
//! Machiavellian risk management system end-to-end.

use talebian_risk_rs::{
    MacchiavelianConfig, TalebianRiskEngine, MarketData, 
    WhaleDirection, TalebianRiskError,
};

mod test_data;
mod whale_scenarios;
mod black_swan_scenarios;
mod performance_tests;

use test_data::*;

/// Test complete risk assessment pipeline
#[test]
fn test_complete_risk_assessment_pipeline() {
    let config = MacchiavelianConfig::aggressive_defaults();
    let mut engine = TalebianRiskEngine::new(config);
    
    let market_data = create_bull_market_data();
    
    let assessment = engine.assess_risk(&market_data).unwrap();
    
    // Validate assessment structure
    assert!(assessment.antifragility_score >= 0.0);
    assert!(assessment.antifragility_score <= 1.0);
    assert!(assessment.barbell_allocation.0 + assessment.barbell_allocation.1 <= 1.1);
    assert!(assessment.black_swan_probability >= 0.0);
    assert!(assessment.kelly_fraction >= 0.0);
    assert!(assessment.overall_risk_score >= 0.0);
    assert!(assessment.overall_risk_score <= 1.0);
    assert!(assessment.recommended_position_size >= 0.0);
    assert!(assessment.confidence >= 0.0);
    assert!(assessment.confidence <= 1.0);
    
    // Aggressive configuration should result in higher position sizes
    assert!(assessment.recommended_position_size > 0.1);
    
    // Whale detection should be properly integrated
    if assessment.whale_detection.is_whale_detected {
        assert!(assessment.whale_detection.confidence > 0.0);
    }
}

/// Test aggressive vs conservative parameter comparison
#[test]
fn test_aggressive_vs_conservative_comparison() {
    let aggressive_config = MacchiavelianConfig::aggressive_defaults();
    let conservative_config = MacchiavelianConfig::conservative_baseline();
    
    let mut aggressive_engine = TalebianRiskEngine::new(aggressive_config);
    let mut conservative_engine = TalebianRiskEngine::new(conservative_config);
    
    let market_data = create_high_opportunity_data();
    
    let aggressive_assessment = aggressive_engine.assess_risk(&market_data).unwrap();
    let conservative_assessment = conservative_engine.assess_risk(&market_data).unwrap();
    
    // Aggressive should recommend larger positions
    assert!(aggressive_assessment.recommended_position_size > 
            conservative_assessment.recommended_position_size);
    
    // Aggressive should have lower antifragility threshold
    assert!(aggressive_assessment.antifragility_score > 
            conservative_assessment.antifragility_score);
    
    // Aggressive should have higher Kelly fractions
    assert!(aggressive_assessment.kelly_fraction > 
            conservative_assessment.kelly_fraction);
    
    // Aggressive should be more tolerant of black swans
    assert!(aggressive_assessment.black_swan_probability < 
            conservative_assessment.black_swan_probability);
}

/// Test whale detection and parasitic trading
#[test]
fn test_whale_detection_and_parasitic_trading() {
    let config = MacchiavelianConfig::aggressive_defaults();
    let mut engine = TalebianRiskEngine::new(config);
    
    // Test strong whale activity scenario
    let whale_data = create_whale_activity_data();
    let assessment = engine.assess_risk(&whale_data).unwrap();
    
    // Should detect whale activity
    assert!(assessment.whale_detection.is_whale_detected);
    assert!(assessment.whale_detection.confidence > 0.7);
    
    // Should increase position size due to whale following
    assert!(assessment.recommended_position_size > 0.2);
    
    // Parasitic opportunity score should be high
    assert!(assessment.parasitic_opportunity.opportunity_score > 0.6);
    
    // Whale alignment should be significant
    assert!(assessment.parasitic_opportunity.whale_alignment > 0.5);
}

/// Test antifragility detection with volatile markets
#[test]
fn test_antifragility_in_volatile_markets() {
    let config = MacchiavelianConfig::aggressive_defaults();
    let mut engine = TalebianRiskEngine::new(config);
    
    let high_vol_data = create_high_volatility_data();
    let assessment = engine.assess_risk(&high_vol_data).unwrap();
    
    // Aggressive system should be more antifragile in high volatility
    assert!(assessment.antifragility_score > 0.4);
    
    // Should not be overly conservative with black swan probability
    assert!(assessment.black_swan_probability < 0.3);
    
    // Should recommend reasonable position despite volatility
    assert!(assessment.recommended_position_size > 0.05);
    
    // Opportunity score should be high for volatile markets
    assert!(assessment.parasitic_opportunity.opportunity_score > 0.5);
}

/// Test Kelly criterion with extreme scenarios
#[test]
fn test_kelly_criterion_extreme_scenarios() {
    let config = MacchiavelianConfig::aggressive_defaults();
    let mut engine = TalebianRiskEngine::new(config);
    
    // Test with very high expected returns
    let high_return_data = create_high_return_scenario();
    let assessment = engine.assess_risk(&high_return_data).unwrap();
    
    // Kelly fraction should be significant but bounded
    assert!(assessment.kelly_fraction > 0.3);
    assert!(assessment.kelly_fraction <= config.kelly_max_fraction);
    
    // Test with very low expected returns
    let low_return_data = create_low_return_scenario();
    let assessment = engine.assess_risk(&low_return_data).unwrap();
    
    // Kelly fraction should be conservative for low returns
    assert!(assessment.kelly_fraction < 0.3);
    
    // Test with whale activity boosting Kelly
    let whale_return_data = create_whale_momentum_scenario();
    let assessment = engine.assess_risk(&whale_return_data).unwrap();
    
    if assessment.whale_detection.is_whale_detected {
        // Kelly should get whale multiplier boost
        assert!(assessment.kelly_fraction > 0.4);
    }
}

/// Test black swan handling
#[test]
fn test_black_swan_tolerance() {
    let config = MacchiavelianConfig::aggressive_defaults();
    let mut engine = TalebianRiskEngine::new(config);
    
    // Test beneficial black swan
    let beneficial_swan_data = create_beneficial_black_swan();
    let assessment = engine.assess_risk(&beneficial_swan_data).unwrap();
    
    // Should not be overly defensive about beneficial volatility
    assert!(assessment.black_swan_probability < config.black_swan_threshold);
    assert!(assessment.recommended_position_size > 0.1);
    
    // Test destructive black swan
    let destructive_swan_data = create_destructive_black_swan();
    let assessment = engine.assess_risk(&destructive_swan_data).unwrap();
    
    // Should provide appropriate protection
    if assessment.black_swan_probability > config.black_swan_threshold {
        assert!(assessment.recommended_position_size < 0.3);
    }
}

/// Test barbell strategy allocation
#[test]
fn test_barbell_strategy_allocation() {
    let config = MacchiavelianConfig::aggressive_defaults();
    let mut engine = TalebianRiskEngine::new(config);
    
    let market_data = create_standard_market_data();
    let assessment = engine.assess_risk(&market_data).unwrap();
    
    // Aggressive barbell should allocate less to safe assets
    assert!(assessment.barbell_allocation.0 < 0.8); // Safe allocation < 80%
    assert!(assessment.barbell_allocation.1 > 0.2); // Risky allocation > 20%
    
    // Total allocation should not exceed 100%
    assert!(assessment.barbell_allocation.0 + assessment.barbell_allocation.1 <= 1.01);
    
    // With whale activity, should shift more to risky
    let whale_data = create_whale_activity_data();
    let whale_assessment = engine.assess_risk(&whale_data).unwrap();
    
    if whale_assessment.whale_detection.is_whale_detected {
        assert!(whale_assessment.barbell_allocation.1 > assessment.barbell_allocation.1);
    }
}

/// Test recommendation generation
#[test]
fn test_recommendation_generation() {
    let config = MacchiavelianConfig::aggressive_defaults();
    let mut engine = TalebianRiskEngine::new(config);
    
    let market_data = create_high_opportunity_data();
    let recommendations = engine.generate_recommendations(&market_data).unwrap();
    
    // Validate position sizing recommendations
    assert!(recommendations.position_sizing.final_recommended_size > 0.0);
    assert!(recommendations.position_sizing.final_recommended_size <= 
            recommendations.position_sizing.max_position_size);
    
    // Validate risk controls
    assert!(recommendations.risk_controls.stop_loss_level > 0.0);
    assert!(recommendations.risk_controls.take_profit_level > 
            recommendations.risk_controls.stop_loss_level);
    
    // Validate timing guidance
    assert!(!recommendations.timing_guidance.entry_urgency.is_empty());
    assert!(!recommendations.timing_guidance.market_regime.is_empty());
    
    // Validate performance metrics
    assert!(recommendations.performance_metrics.expected_return >= 0.0);
    assert!(recommendations.performance_metrics.win_probability >= 0.0);
    assert!(recommendations.performance_metrics.win_probability <= 1.0);
}

/// Test learning and adaptation
#[test]
fn test_learning_and_adaptation() {
    let config = MacchiavelianConfig::aggressive_defaults();
    let mut engine = TalebianRiskEngine::new(config);
    
    // Record successful whale trades
    for _ in 0..10 {
        engine.record_trade_outcome(0.02, true, 0.8).unwrap();
    }
    
    // Record unsuccessful non-whale trades
    for _ in 0..5 {
        engine.record_trade_outcome(-0.01, false, 0.3).unwrap();
    }
    
    let status = engine.get_engine_status();
    
    // Should show learning progress
    assert!(status.performance_tracker.successful_predictions > 0);
    assert!(status.performance_tracker.total_return > 0.0);
    
    // Kelly engine should have learned from outcomes
    assert!(status.kelly_status.win_rate_tracker.wins > 0);
    assert!(status.kelly_status.win_rate_tracker.average_win > 0.0);
}

/// Test error handling and recovery
#[test]
fn test_error_handling_and_recovery() {
    let config = MacchiavelianConfig::aggressive_defaults();
    let mut engine = TalebianRiskEngine::new(config);
    
    // Test with invalid market data
    let invalid_data = MarketData {
        timestamp: 0,
        price: -100.0, // Invalid negative price
        volume: 1000.0,
        bid: 99.0,
        ask: 101.0,
        bid_volume: 500.0,
        ask_volume: 500.0,
        volatility: 0.02,
        returns: vec![],
        volume_history: vec![],
    };
    
    // Should handle gracefully (depends on implementation)
    let result = engine.assess_risk(&invalid_data);
    
    // Either succeeds with sanitized data or fails gracefully
    match result {
        Ok(assessment) => {
            // If it succeeds, should have reasonable values
            assert!(assessment.confidence < 0.5); // Low confidence for bad data
        },
        Err(e) => {
            // Should be a proper error, not a panic
            assert!(matches!(e, TalebianRiskError::MarketDataError(_) | 
                                TalebianRiskError::InvalidInput(_)));
        }
    }
}

/// Test extreme Machiavellian configuration
#[test]
fn test_extreme_machiavellian_configuration() {
    let extreme_config = MacchiavelianConfig::extreme_machiavellian();
    let mut engine = TalebianRiskEngine::new(extreme_config.clone());
    
    let market_data = create_extreme_opportunity_data();
    let assessment = engine.assess_risk(&market_data).unwrap();
    
    // Extreme configuration should be very aggressive
    assert!(assessment.recommended_position_size > 0.3);
    
    // Should have very low antifragility threshold
    assert_eq!(extreme_config.antifragility_threshold, 0.25);
    
    // Should have very high Kelly fraction
    assert_eq!(extreme_config.kelly_fraction, 0.7);
    
    // Should be very tolerant of black swans
    assert_eq!(extreme_config.black_swan_threshold, 0.25);
}

/// Test performance requirements
#[test]
fn test_performance_requirements() {
    let config = MacchiavelianConfig::aggressive_defaults();
    let mut engine = TalebianRiskEngine::new(config);
    
    let market_data = create_standard_market_data();
    
    // Test latency requirement (should complete in under 10ms)
    let start = std::time::Instant::now();
    let _assessment = engine.assess_risk(&market_data).unwrap();
    let duration = start.elapsed();
    
    assert!(duration.as_millis() < 10, 
           "Risk assessment took too long: {}ms", duration.as_millis());
}

/// Test bulk processing
#[test]
fn test_bulk_risk_assessment() {
    let config = MacchiavelianConfig::aggressive_defaults();
    let mut engine = TalebianRiskEngine::new(config);
    
    // Create batch of market data
    let market_data_batch = vec![
        create_bull_market_data(),
        create_bear_market_data(),
        create_whale_activity_data(),
        create_high_volatility_data(),
    ];
    
    let assessments = engine.assess_bulk_risks(&market_data_batch).unwrap();
    
    assert_eq!(assessments.len(), 4);
    
    // Each assessment should be valid
    for assessment in assessments {
        assert!(assessment.confidence > 0.0);
        assert!(assessment.overall_risk_score >= 0.0);
        assert!(assessment.overall_risk_score <= 1.0);
        assert!(assessment.recommended_position_size >= 0.0);
    }
}

/// Test configuration validation
#[test]
fn test_configuration_validation() {
    use talebian_risk_rs::errors::validation;
    
    // Valid aggressive configuration
    let valid_config = MacchiavelianConfig::aggressive_defaults();
    assert!(validation::validate_config(&valid_config).is_ok());
    
    // Invalid configuration
    let invalid_config = MacchiavelianConfig {
        antifragility_threshold: 1.5, // Invalid > 1.0
        ..valid_config
    };
    assert!(validation::validate_config(&invalid_config).is_err());
    
    // Barbell allocations exceeding 100%
    let invalid_barbell = MacchiavelianConfig {
        barbell_safe_ratio: 0.7,
        barbell_risky_ratio: 0.5, // Total > 1.0
        ..valid_config
    };
    assert!(validation::validate_config(&invalid_barbell).is_err());
}

/// Test memory efficiency
#[test]
fn test_memory_efficiency() {
    let config = MacchiavelianConfig::aggressive_defaults();
    let mut engine = TalebianRiskEngine::new(config);
    
    // Perform many assessments to test memory growth
    for i in 0..1000 {
        let mut market_data = create_standard_market_data();
        market_data.timestamp = i;
        
        let _assessment = engine.assess_risk(&market_data).unwrap();
    }
    
    // Engine should not consume excessive memory
    // (This is a basic check - in practice you'd use more sophisticated memory profiling)
    let status = engine.get_engine_status();
    assert!(status.total_assessments == 1000);
}

/// Test thread safety (if applicable)
#[cfg(feature = "parallel")]
#[test]
fn test_thread_safety() {
    use std::sync::Arc;
    use std::thread;
    
    let config = MacchiavelianConfig::aggressive_defaults();
    
    // Create multiple engines for parallel testing
    let handles: Vec<_> = (0..4).map(|i| {
        let config = config.clone();
        thread::spawn(move || {
            let mut engine = TalebianRiskEngine::new(config);
            let mut market_data = create_standard_market_data();
            market_data.timestamp = i;
            
            engine.assess_risk(&market_data)
        })
    }).collect();
    
    // All threads should complete successfully
    for handle in handles {
        let result = handle.join().unwrap();
        assert!(result.is_ok());
    }
}

/// Benchmark basic operations
#[cfg(feature = "simd")]
#[test]
fn test_simd_performance() {
    use talebian_risk_rs::performance::SimdMath;
    
    // Test SIMD Kelly calculation performance
    let expected_returns = vec![0.01; 1000];
    let variances = vec![0.001; 1000];
    let whale_multipliers = vec![1.0; 1000];
    
    let start = std::time::Instant::now();
    let _results = SimdMath::kelly_fraction_simd_x4(
        &expected_returns,
        &variances,
        &whale_multipliers
    ).unwrap();
    let duration = start.elapsed();
    
    // Should be very fast for SIMD operations
    assert!(duration.as_micros() < 1000, 
           "SIMD calculation took too long: {}μs", duration.as_micros());
}

/// Integration with real market data patterns
#[test]
fn test_real_market_patterns() {
    let config = MacchiavelianConfig::aggressive_defaults();
    let mut engine = TalebianRiskEngine::new(config);
    
    // Test crypto market crash pattern
    let crash_data = create_crypto_crash_pattern();
    let assessment = engine.assess_risk(&crash_data).unwrap();
    
    // Should detect high risk but still provide opportunities
    assert!(assessment.overall_risk_score > 0.5);
    assert!(assessment.black_swan_probability > 0.1);
    
    // Test crypto bull run pattern
    let bull_run_data = create_crypto_bull_run_pattern();
    let assessment = engine.assess_risk(&bull_run_data).unwrap();
    
    // Should be bullish and recommend higher positions
    assert!(assessment.antifragility_score > 0.4);
    assert!(assessment.recommended_position_size > 0.2);
    
    // Test whale manipulation pattern
    let manipulation_data = create_whale_manipulation_pattern();
    let assessment = engine.assess_risk(&manipulation_data).unwrap();
    
    // Should detect whale activity but be cautious
    if assessment.whale_detection.is_whale_detected {
        assert!(assessment.whale_detection.confidence > 0.5);
    }
}

/// Test parameter recalibration effectiveness
#[test]
fn test_parameter_recalibration_effectiveness() {
    let conservative = MacchiavelianConfig::conservative_baseline();
    let aggressive = MacchiavelianConfig::aggressive_defaults();
    
    let mut conservative_engine = TalebianRiskEngine::new(conservative);
    let mut aggressive_engine = TalebianRiskEngine::new(aggressive);
    
    // Test with opportunity-rich data
    let opportunity_data = create_high_opportunity_data();
    
    let conservative_assessment = conservative_engine.assess_risk(&opportunity_data).unwrap();
    let aggressive_assessment = aggressive_engine.assess_risk(&opportunity_data).unwrap();
    
    // Aggressive should capture more opportunities
    assert!(aggressive_assessment.parasitic_opportunity.opportunity_score > 
            conservative_assessment.parasitic_opportunity.opportunity_score);
    
    // Aggressive should recommend larger positions
    assert!(aggressive_assessment.recommended_position_size > 
            conservative_assessment.recommended_position_size * 1.5);
    
    // Conservative might miss opportunities due to overly strict thresholds
    if conservative_assessment.parasitic_opportunity.opportunity_score < 
       conservative.parasitic_opportunity_threshold {
        // Conservative blocked the trade
        assert!(aggressive_assessment.parasitic_opportunity.opportunity_score > 
               aggressive.parasitic_opportunity_threshold);
        // But aggressive captured it
    }
}