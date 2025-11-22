//! Comprehensive tests for Octopus Camouflage organism
//! Following TDD methodology - tests written FIRST
//! CQGS Compliance: Zero mocks, all real implementations

use parasitic::{Result, Error};
use parasitic::traits::{
    Organism, OrganismMetrics, MarketData, Adaptive, AdaptationState, 
    PerformanceMonitor, PerformanceStats
};
use parasitic::organisms::octopus::{
    OctopusCamouflage, MarketPredatorDetector, DynamicSelectionStrategy, 
    ChromatophoreState, CamouflageConfig, PredatorThreat, CamouflagePattern,
    SelectionResult, ThreatAssessment, ColorChangingResult
};
use chrono::{DateTime, Utc};
use std::collections::HashMap;

fn create_test_market_data() -> MarketData {
    MarketData {
        symbol: "BTC/USDT".to_string(),
        timestamp: Utc::now(),
        price: 50000.0,
        volume: 1000.0,
        volatility: 0.05,
        bid: 49995.0,
        ask: 50005.0,
        spread_percent: 0.02,
        market_cap: Some(1_000_000_000.0),
        liquidity_score: 0.8,
    }
}

fn create_high_volatility_market_data() -> MarketData {
    MarketData {
        symbol: "DOGE/USDT".to_string(),
        timestamp: Utc::now(),
        price: 0.08,
        volume: 100000.0,
        volatility: 0.25, // High volatility
        bid: 0.079,
        ask: 0.081,
        spread_percent: 2.5, // Wide spread
        market_cap: Some(10_000_000.0),
        liquidity_score: 0.3, // Low liquidity
    }
}

fn create_low_liquidity_market_data() -> MarketData {
    MarketData {
        symbol: "ALT/USDT".to_string(),
        timestamp: Utc::now(),
        price: 1.0,
        volume: 10.0, // Very low volume
        volatility: 0.02,
        bid: 0.98,
        ask: 1.02,
        spread_percent: 4.0, // Very wide spread
        market_cap: Some(100_000.0),
        liquidity_score: 0.05, // Very low liquidity
    }
}

#[test]
fn test_octopus_creation() {
    let octopus = OctopusCamouflage::new();
    assert!(octopus.is_ok(), "Should create Octopus Camouflage successfully");
    
    let octopus = octopus.unwrap();
    assert_eq!(octopus.name(), "OctopusCamouflage");
    assert_eq!(octopus.organism_type(), "CamouflageAdaptive");
    assert!(octopus.is_active(), "Should be active by default");
}

#[test]
fn test_octopus_custom_config() {
    let config = CamouflageConfig {
        name: "CustomOctopus".to_string(),
        max_processing_time_ns: 300_000, // 0.3ms
        threat_detection_sensitivity: 0.9,
        camouflage_adaptation_speed: 0.15,
        chromatophore_response_time_ns: 100_000,
        min_threat_threshold: 0.1,
        max_camouflage_patterns: 20,
        enable_aggressive_camouflage: true,
        enable_predator_learning: true,
        enable_pattern_caching: true,
    };

    let octopus = OctopusCamouflage::with_config(config);
    assert!(octopus.is_ok(), "Should create with custom config");
    
    let octopus = octopus.unwrap();
    assert_eq!(octopus.name(), "CustomOctopus");
}

#[test]
fn test_threat_detector_creation() {
    let detector = MarketPredatorDetector::new();
    assert!(detector.is_ok(), "Should create threat detector successfully");
}

#[test]
fn test_camouflage_strategy_creation() {
    let strategy = DynamicSelectionStrategy::new();
    assert!(strategy.is_ok(), "Should create camouflage strategy successfully");
}

#[test]
fn test_chromatophore_state_creation() {
    let state = ChromatophoreState::new();
    assert!(state.is_ok(), "Should create chromatophore state successfully");
}

#[test]
fn test_market_predator_detection_normal_conditions() {
    let octopus = OctopusCamouflage::new().unwrap();
    let market_data = create_test_market_data();
    
    let threat_assessment = octopus.detect_market_predators(&market_data);
    assert!(threat_assessment.is_ok(), "Should detect threats successfully");
    
    let assessment = threat_assessment.unwrap();
    assert!(assessment.processing_time_ns < 500_000, "Should be sub-millisecond");
    assert!((0.0..=1.0).contains(&assessment.overall_threat_level), "Threat level should be normalized");
    assert!(!assessment.identified_threats.is_empty() || assessment.overall_threat_level < 0.1, 
           "Should identify threats or have low threat level");
}

#[test]
fn test_market_predator_detection_high_volatility() {
    let octopus = OctopusCamouflage::new().unwrap();
    let market_data = create_high_volatility_market_data();
    
    let threat_assessment = octopus.detect_market_predators(&market_data);
    assert!(threat_assessment.is_ok(), "Should handle high volatility");
    
    let assessment = threat_assessment.unwrap();
    // High volatility should trigger predator detection
    assert!(assessment.overall_threat_level > 0.3, "High volatility should indicate predator presence");
    assert!(assessment.identified_threats.iter().any(|t| t.threat_type == "VolatilityPredator"), 
           "Should detect volatility predator");
}

#[test]
fn test_market_predator_detection_low_liquidity() {
    let octopus = OctopusCamouflage::new().unwrap();
    let market_data = create_low_liquidity_market_data();
    
    let threat_assessment = octopus.detect_market_predators(&market_data);
    assert!(threat_assessment.is_ok(), "Should handle low liquidity");
    
    let assessment = threat_assessment.unwrap();
    // Low liquidity should trigger predator detection
    assert!(assessment.overall_threat_level > 0.4, "Low liquidity should indicate predator presence");
    assert!(assessment.identified_threats.iter().any(|t| t.threat_type == "LiquidityPredator"), 
           "Should detect liquidity predator");
}

#[test]
fn test_dynamic_camouflage_selection_normal() {
    let octopus = OctopusCamouflage::new().unwrap();
    let market_data = create_test_market_data();
    let threat_level = 0.3;
    
    let selection_result = octopus.select_camouflage_strategy(&market_data, threat_level);
    assert!(selection_result.is_ok(), "Should select camouflage strategy");
    
    let result = selection_result.unwrap();
    assert!(result.processing_time_ns < 300_000, "Should be very fast");
    assert!(!result.selected_pattern.pattern_name.is_empty(), "Should have pattern name");
    assert!((0.0..=1.0).contains(&result.effectiveness_score), "Effectiveness should be normalized");
    assert!((0.0..=1.0).contains(&result.selected_pattern.camouflage_strength), "Strength should be normalized");
}

#[test]
fn test_dynamic_camouflage_selection_high_threat() {
    let octopus = OctopusCamouflage::new().unwrap();
    let market_data = create_high_volatility_market_data();
    let threat_level = 0.8; // High threat
    
    let selection_result = octopus.select_camouflage_strategy(&market_data, threat_level);
    assert!(selection_result.is_ok(), "Should select strategy for high threat");
    
    let result = selection_result.unwrap();
    // High threat should trigger aggressive camouflage
    assert!(result.effectiveness_score > 0.7, "Should be highly effective for high threat");
    assert!(result.selected_pattern.camouflage_strength > 0.6, "Should use strong camouflage");
    assert!(result.selected_pattern.pattern_name.contains("Aggressive") || 
           result.selected_pattern.pattern_name.contains("Stealth"), 
           "Should use aggressive or stealth pattern");
}

#[test]
fn test_chromatophore_color_changing_normal() {
    let octopus = OctopusCamouflage::new().unwrap();
    let market_data = create_test_market_data();
    
    let color_result = octopus.change_chromatophore_colors(&market_data, 0.5);
    assert!(color_result.is_ok(), "Should change colors successfully");
    
    let result = color_result.unwrap();
    assert!(result.processing_time_ns < 200_000, "Should be very fast color change");
    assert!((0.0..=1.0).contains(&result.color_intensity), "Intensity should be normalized");
    assert!((0.0..=1.0).contains(&result.pattern_complexity), "Complexity should be normalized");
    assert!(!result.active_patterns.is_empty(), "Should have active patterns");
}

#[test]
fn test_chromatophore_color_changing_rapid() {
    let octopus = OctopusCamouflage::new().unwrap();
    let market_data = create_high_volatility_market_data();
    
    let color_result = octopus.change_chromatophore_colors(&market_data, 0.9);
    assert!(color_result.is_ok(), "Should handle rapid color change");
    
    let result = color_result.unwrap();
    // High adaptation rate should produce rapid changes
    assert!(result.color_intensity > 0.7, "Should have high color intensity");
    assert!(result.change_speed > 0.6, "Should have high change speed");
    assert!(result.processing_time_ns < 150_000, "Should be very fast for rapid changes");
}

#[test]
fn test_full_camouflage_adaptation_cycle() {
    let mut octopus = OctopusCamouflage::new().unwrap();
    let market_data = create_test_market_data();
    
    // Full adaptation cycle
    let adaptation_result = octopus.adapt_camouflage_to_market(&market_data);
    assert!(adaptation_result.is_ok(), "Should complete full adaptation");
    
    let result = adaptation_result.unwrap();
    assert!(result.processing_time_ns < 500_000, "Full cycle should be sub-millisecond");
    assert!(result.adaptation_success, "Adaptation should succeed");
    assert!((0.0..=1.0).contains(&result.effectiveness_improvement), "Improvement should be normalized");
    
    // Verify all components were updated
    assert!(!result.threat_assessment.identified_threats.is_empty() || 
           result.threat_assessment.overall_threat_level < 0.1, 
           "Should assess threats");
    assert!(!result.camouflage_selection.selected_pattern.pattern_name.is_empty(), 
           "Should select camouflage");
    assert!(!result.color_change.active_patterns.is_empty(), "Should change colors");
}

#[test]
fn test_performance_requirements() {
    let octopus = OctopusCamouflage::new().unwrap();
    let market_data = create_test_market_data();
    
    // Test multiple operations for performance consistency
    for _ in 0..100 {
        let start = std::time::Instant::now();
        let _result = octopus.detect_market_predators(&market_data);
        let duration = start.elapsed().as_nanos() as u64;
        assert!(duration < 500_000, "Each operation should be sub-millisecond");
    }
}

#[test]
fn test_organism_trait_implementation() {
    let mut octopus = OctopusCamouflage::new().unwrap();
    
    // Test basic organism interface
    assert_eq!(octopus.name(), "OctopusCamouflage");
    assert_eq!(octopus.organism_type(), "CamouflageAdaptive");
    assert!(octopus.is_active());
    
    // Test deactivation
    octopus.set_active(false);
    assert!(!octopus.is_active());
    
    // Test reactivation
    octopus.set_active(true);
    assert!(octopus.is_active());
    
    // Test metrics
    let metrics = octopus.get_metrics();
    assert!(metrics.is_ok(), "Should provide metrics");
    
    let metrics = metrics.unwrap();
    assert!(metrics.total_operations >= 0, "Should track operations");
    assert!((0.0..=1.0).contains(&metrics.accuracy_rate), "Accuracy should be normalized");
    
    // Test reset
    assert!(octopus.reset().is_ok(), "Should reset successfully");
    
    let metrics_after_reset = octopus.get_metrics().unwrap();
    assert_eq!(metrics_after_reset.total_operations, 0, "Should reset operation count");
}

#[test]
fn test_adaptive_trait_implementation() {
    let mut octopus = OctopusCamouflage::new().unwrap();
    let market_data = create_test_market_data();
    let timestamp = chrono::Utc::now().timestamp_millis() as u64;
    
    // Test adaptation
    let adaptation = octopus.adapt_to_conditions(&market_data, timestamp);
    assert!(adaptation.is_ok(), "Should adapt to conditions");
    
    // Test adaptation state
    let state = octopus.get_adaptation_state();
    assert!((0.0..=1.0).contains(&state.current_sensitivity), "Sensitivity should be normalized");
    assert!(state.adaptation_speed > 0.0, "Should have adaptation speed");
    assert!(state.learning_rate > 0.0, "Should have learning rate");
    assert!((0.0..=1.0).contains(&state.confidence_level), "Confidence should be normalized");
    
    // Test sensitivity level
    let sensitivity = octopus.get_sensitivity_level();
    assert!((0.0..=1.0).contains(&sensitivity), "Sensitivity should be normalized");
    
    // Test setting sensitivity
    assert!(octopus.set_sensitivity_level(0.7).is_ok(), "Should set sensitivity");
    assert_eq!(octopus.get_sensitivity_level(), 0.7, "Should update sensitivity");
}

#[test]
fn test_performance_monitor_implementation() {
    let mut octopus = OctopusCamouflage::new().unwrap();
    
    // Test recording metrics
    octopus.record_metric("test_metric", 0.8, chrono::Utc::now().timestamp_millis() as u64);
    
    // Test getting stats
    let stats = octopus.get_stats();
    assert!(stats.is_ok(), "Should provide performance stats");
    
    let stats = stats.unwrap();
    assert!(stats.avg_processing_time_ns >= 0, "Should track processing time");
    assert!(stats.throughput_ops_per_sec >= 0.0, "Should track throughput");
    assert!((0.0..=1.0).contains(&stats.accuracy_rate), "Accuracy should be normalized");
    
    // Test requirements check
    // Initially should meet requirements
    let meets_req = octopus.meets_requirements();
    assert!(meets_req, "Fresh octopus should meet requirements");
}

#[test]
fn test_camouflage_pattern_effectiveness() {
    let octopus = OctopusCamouflage::new().unwrap();
    
    // Test different market conditions and verify appropriate patterns
    let normal_market = create_test_market_data();
    let volatile_market = create_high_volatility_market_data();
    let illiquid_market = create_low_liquidity_market_data();
    
    let normal_result = octopus.select_camouflage_strategy(&normal_market, 0.3).unwrap();
    let volatile_result = octopus.select_camouflage_strategy(&volatile_market, 0.8).unwrap();
    let illiquid_result = octopus.select_camouflage_strategy(&illiquid_market, 0.7).unwrap();
    
    // Different conditions should produce different strategies
    assert!(normal_result.selected_pattern.pattern_name != volatile_result.selected_pattern.pattern_name ||
            normal_result.selected_pattern.camouflage_strength != volatile_result.selected_pattern.camouflage_strength,
           "Different market conditions should produce different patterns");
    
    // High threat should produce higher effectiveness
    assert!(volatile_result.effectiveness_score >= normal_result.effectiveness_score,
           "High threat should produce high effectiveness");
}

#[test]
fn test_error_handling() {
    // Test invalid configurations
    let invalid_config = CamouflageConfig {
        name: "".to_string(), // Invalid empty name
        max_processing_time_ns: 0, // Invalid zero time
        threat_detection_sensitivity: 2.0, // Invalid > 1.0
        camouflage_adaptation_speed: -0.1, // Invalid negative
        chromatophore_response_time_ns: 0,
        min_threat_threshold: 1.5, // Invalid > 1.0
        max_camouflage_patterns: 0, // Invalid zero
        enable_aggressive_camouflage: true,
        enable_predator_learning: true,
        enable_pattern_caching: true,
    };
    
    let octopus_result = OctopusCamouflage::with_config(invalid_config);
    assert!(octopus_result.is_err(), "Should reject invalid config");
}

#[test]
fn test_concurrent_operations() {
    use std::sync::Arc;
    use std::thread;
    
    let octopus = Arc::new(OctopusCamouflage::new().unwrap());
    let market_data = create_test_market_data();
    
    let mut handles = vec![];
    
    // Test concurrent threat detection
    for _ in 0..10 {
        let octopus_clone = Arc::clone(&octopus);
        let market_data_clone = market_data.clone();
        
        let handle = thread::spawn(move || {
            let result = octopus_clone.detect_market_predators(&market_data_clone);
            assert!(result.is_ok(), "Concurrent threat detection should work");
        });
        
        handles.push(handle);
    }
    
    // Wait for all threads
    for handle in handles {
        handle.join().expect("Thread should complete successfully");
    }
}

#[test]
fn test_memory_efficiency() {
    let octopus = OctopusCamouflage::new().unwrap();
    let market_data = create_test_market_data();
    
    // Perform many operations to test memory usage
    for _ in 0..1000 {
        let _threat = octopus.detect_market_predators(&market_data).unwrap();
        let _camouflage = octopus.select_camouflage_strategy(&market_data, 0.5).unwrap();
        let _color = octopus.change_chromatophore_colors(&market_data, 0.5).unwrap();
    }
    
    let metrics = octopus.get_metrics().unwrap();
    // Memory should be reasonable (less than 10MB for this test)
    assert!(metrics.memory_usage_bytes < 10 * 1024 * 1024, 
           "Memory usage should be reasonable after many operations");
}

#[test]
fn test_adaptation_learning() {
    let mut octopus = OctopusCamouflage::new().unwrap();
    let mut market_data = create_test_market_data();
    
    // Initial adaptation state
    let initial_state = octopus.get_adaptation_state();
    
    // Gradually increase market volatility and adapt
    for i in 1..=10 {
        market_data.volatility = 0.05 + (i as f64) * 0.02; // Increase volatility
        let timestamp = chrono::Utc::now().timestamp_millis() as u64 + (i as u64) * 1000;
        
        octopus.adapt_to_conditions(&market_data, timestamp).unwrap();
    }
    
    let final_state = octopus.get_adaptation_state();
    
    // Adaptation should have occurred
    assert!(final_state.current_sensitivity != initial_state.current_sensitivity ||
            final_state.confidence_level != initial_state.confidence_level,
           "Adaptation state should change with market conditions");
}