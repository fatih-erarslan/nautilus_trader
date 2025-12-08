//! Comprehensive unit tests for Antifragility measurement
//! Tests measurement accuracy, stress response, and mathematical properties

use talebian_risk_rs::{
    antifragility::*, MacchiavelianConfig, MarketData, TalebianRiskError
};
use chrono::{DateTime, Utc, Duration};
use approx::assert_relative_eq;

/// Test helper to create sample antifragile data pattern
fn create_antifragile_pattern() -> Vec<(f64, f64)> {
    // Returns that increase with volatility (antifragile pattern)
    let mut data = Vec::new();
    for i in 0..100 {
        let volatility = 0.1 + (i as f64 * 0.002); // Increasing volatility
        let returns = 0.005 + (volatility * 2.0); // Returns increase with volatility
        data.push((returns, volatility));
    }
    data
}

/// Test helper to create fragile data pattern
fn create_fragile_pattern() -> Vec<(f64, f64)> {
    // Returns that decrease with volatility (fragile pattern)
    let mut data = Vec::new();
    for i in 0..100 {
        let volatility = 0.1 + (i as f64 * 0.002);
        let returns = 0.01 - (volatility * 1.5); // Returns decrease with volatility
        data.push((returns, volatility));
    }
    data
}

/// Test helper to create market data from pattern
fn create_market_data(returns: f64, volatility: f64) -> MarketData {
    MarketData {
        timestamp: Utc::now(),
        timestamp_unix: 1640995200,
        price: 50000.0,
        volume: 1000.0,
        bid: 49990.0,
        ask: 50010.0,
        bid_volume: 500.0,
        ask_volume: 500.0,
        volatility,
        returns: vec![returns],
        volume_history: vec![1000.0; 5],
    }
}

#[cfg(test)]
mod antifragility_tests {
    use super::*;

    #[test]
    fn test_measurer_creation() {
        let params = AntifragilityParams::default();
        let measurer = AntifragilityMeasurer::new("test_portfolio".to_string(), params.clone());
        
        assert_eq!(measurer.portfolio_id, "test_portfolio");
        assert_eq!(measurer.return_history.len(), 0);
        assert_eq!(measurer.volatility_history.len(), 0);
        assert_eq!(measurer.stress_events.len(), 0);
        assert_eq!(measurer.adaptation_history.len(), 0);
        assert!(measurer.last_measurement.is_none());
        
        // Verify default parameters
        assert_eq!(params.volatility_threshold, 0.2);
        assert_eq!(params.min_observation_period, 100);
        assert_eq!(params.convexity_window, 50);
        assert!(params.enable_hormesis);
    }

    #[test]
    fn test_update_with_data() {
        let mut measurer = AntifragilityMeasurer::new("test".to_string(), AntifragilityParams::default());
        
        let result = measurer.update(0.015, 0.18, Utc::now());
        assert!(result.is_ok());
        
        assert_eq!(measurer.return_history.len(), 1);
        assert_eq!(measurer.volatility_history.len(), 1);
        assert_eq!(measurer.return_history[0], 0.015);
        assert_eq!(measurer.volatility_history[0], 0.18);
    }

    #[test]
    fn test_insufficient_data_error() {
        let mut measurer = AntifragilityMeasurer::new("test".to_string(), AntifragilityParams::default());
        
        // Add only a few data points (below minimum required)
        for i in 0..50 {
            measurer.update(i as f64 * 0.001, 0.15, Utc::now()).unwrap();
        }
        
        let result = measurer.measure_antifragility();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Insufficient data"));
    }

    #[test]
    fn test_antifragile_measurement_sufficient_data() {
        let mut measurer = AntifragilityMeasurer::new("test".to_string(), AntifragilityParams::default());
        let antifragile_data = create_antifragile_pattern();
        
        // Add antifragile pattern data
        for (returns, volatility) in antifragile_data {
            measurer.update(returns, volatility, Utc::now()).unwrap();
        }
        
        let result = measurer.measure_antifragility();
        assert!(result.is_ok());
        
        let measurement = result.unwrap();
        
        // Validate measurement bounds
        assert!(measurement.overall_score >= 0.0 && measurement.overall_score <= 1.0);
        assert!(measurement.convexity >= 0.0 && measurement.convexity <= 1.0);
        assert!(measurement.volatility_benefit >= 0.0 && measurement.volatility_benefit <= 1.0);
        assert!(measurement.stress_response >= 0.0 && measurement.stress_response <= 1.0);
        assert!(measurement.hormesis_effect >= 0.0 && measurement.hormesis_effect <= 1.0);
        assert!(measurement.tail_benefit >= 0.0 && measurement.tail_benefit <= 1.0);
        assert!(measurement.regime_adaptation >= 0.0 && measurement.regime_adaptation <= 1.0);
        
        // Should detect antifragile behavior
        assert!(measurement.overall_score > 0.5, "Should detect antifragile pattern");
        assert!(measurement.convexity > 0.5, "Should show positive convexity");
        assert!(!measurement.level_description.is_empty());
        assert_eq!(measurement.component_scores.len(), 6);
        
        // Verify component scores are present
        assert!(measurement.component_scores.contains_key("convexity"));
        assert!(measurement.component_scores.contains_key("volatility_benefit"));
        assert!(measurement.component_scores.contains_key("stress_response"));
    }

    #[test]
    fn test_fragile_measurement() {
        let mut measurer = AntifragilityMeasurer::new("test".to_string(), AntifragilityParams::default());
        let fragile_data = create_fragile_pattern();
        
        // Add fragile pattern data
        for (returns, volatility) in fragile_data {
            measurer.update(returns, volatility, Utc::now()).unwrap();
        }
        
        let result = measurer.measure_antifragility();
        assert!(result.is_ok());
        
        let measurement = result.unwrap();
        
        // Should detect fragile behavior
        assert!(measurement.overall_score < 0.5, "Should detect fragile pattern");
        assert!(measurement.convexity < 0.5, "Should show negative convexity");
        assert!(measurement.volatility_benefit < 0.5, "Should show volatility harm");
        assert!(measurement.level_description.contains("Fragile"));
    }

    #[test]
    fn test_stress_event_detection() {
        let mut measurer = AntifragilityMeasurer::new("test".to_string(), AntifragilityParams::default());
        
        // Add normal volatility baseline
        for _ in 0..30 {
            measurer.update(0.01, 0.15, Utc::now()).unwrap();
        }
        
        let initial_stress_events = measurer.stress_events.len();
        
        // Add high volatility event (should trigger stress detection)
        measurer.update(0.01, 0.35, Utc::now()).unwrap(); // 35% volatility vs 15% baseline
        
        // Should have detected stress event
        assert!(measurer.stress_events.len() > initial_stress_events);
        
        if let Some(stress_event) = measurer.stress_events.last() {
            assert!(stress_event.stress_intensity > 0.0);
            assert!(stress_event.duration > 0);
            assert!(stress_event.pre_stress_performance != 0.0 || stress_event.during_stress_performance != 0.0);
        }
    }

    #[test]
    fn test_regime_change_detection() {
        let mut measurer = AntifragilityMeasurer::new("test".to_string(), AntifragilityParams::default());
        
        // Add low volatility period
        for _ in 0..25 {
            measurer.update(0.01, 0.10, Utc::now()).unwrap();
        }
        
        let initial_adaptations = measurer.adaptation_history.len();
        
        // Add high volatility period (should trigger regime change)
        for _ in 0..25 {
            measurer.update(0.01, 0.25, Utc::now()).unwrap(); // 250% increase in volatility
        }
        
        // Should have detected regime change
        assert!(measurer.adaptation_history.len() > initial_adaptations);
        
        if let Some(adaptation) = measurer.adaptation_history.last() {
            assert!(matches!(adaptation.regime_change, RegimeChange::VolatilityIncrease { .. }));
            assert!(adaptation.adaptation_speed > 0.0);
            assert!(adaptation.adaptation_quality >= 0.0);
        }
    }

    #[test]
    fn test_convexity_calculation() {
        let mut measurer = AntifragilityMeasurer::new("test".to_string(), AntifragilityParams::default());
        
        // Create strong positive correlation between volatility and returns
        for i in 0..60 {
            let vol = 0.1 + (i as f64 * 0.005); // Increasing volatility
            let ret = 0.005 + (vol * 0.5); // Returns increase with volatility
            measurer.update(ret, vol, Utc::now()).unwrap();
        }
        
        let convexity = measurer.calculate_convexity().unwrap();
        
        // Should show positive convexity (above neutral 0.5)
        assert!(convexity > 0.5, "Strong positive correlation should show antifragile convexity");
        assert!(convexity <= 1.0);
        
        // Test with negative correlation (fragile)
        let mut fragile_measurer = AntifragilityMeasurer::new("fragile".to_string(), AntifragilityParams::default());
        for i in 0..60 {
            let vol = 0.1 + (i as f64 * 0.005);
            let ret = 0.015 - (vol * 0.5); // Returns decrease with volatility
            fragile_measurer.update(ret, vol, Utc::now()).unwrap();
        }
        
        let fragile_convexity = fragile_measurer.calculate_convexity().unwrap();
        assert!(fragile_convexity < 0.5, "Negative correlation should show fragile convexity");
    }

    #[test]
    fn test_volatility_benefit_calculation() {
        let mut measurer = AntifragilityMeasurer::new("test".to_string(), AntifragilityParams::default());
        
        // Create pattern where high volatility periods have better returns
        for i in 0..100 {
            let vol = if i % 10 < 5 { 0.1 } else { 0.3 }; // Alternating low/high volatility
            let ret = if i % 10 < 5 { 0.005 } else { 0.015 }; // Better returns in high vol
            measurer.update(ret, vol, Utc::now()).unwrap();
        }
        
        let benefit = measurer.calculate_volatility_benefit().unwrap();
        
        // Should show benefit from volatility
        assert!(benefit > 0.5, "Should benefit from high volatility periods");
        
        // Test opposite pattern (fragile)
        let mut fragile_measurer = AntifragilityMeasurer::new("fragile".to_string(), AntifragilityParams::default());
        for i in 0..100 {
            let vol = if i % 10 < 5 { 0.1 } else { 0.3 };
            let ret = if i % 10 < 5 { 0.015 } else { 0.005 }; // Worse returns in high vol
            fragile_measurer.update(ret, vol, Utc::now()).unwrap();
        }
        
        let fragile_benefit = fragile_measurer.calculate_volatility_benefit().unwrap();
        assert!(fragile_benefit < 0.5, "Should be harmed by high volatility");
    }

    #[test]
    fn test_stress_response_calculation() {
        let mut measurer = AntifragilityMeasurer::new("test".to_string(), AntifragilityParams::default());
        
        // Create stress events with good performance during stress
        let stress_event = StressEvent {
            timestamp: Utc::now(),
            stress_intensity: 0.8,
            duration: 86400, // 1 day
            pre_stress_performance: 0.01,
            during_stress_performance: 0.015, // Better performance during stress
            post_stress_performance: 0.01,
            recovery_time: Some(3600), // 1 hour recovery
        };
        
        measurer.stress_events.push(stress_event);
        
        let response = measurer.calculate_stress_response().unwrap();
        
        // Should show good stress response
        assert!(response > 0.5, "Good performance during stress should yield high score");
        
        // Test poor stress response
        let mut poor_measurer = AntifragilityMeasurer::new("poor".to_string(), AntifragilityParams::default());
        let poor_stress_event = StressEvent {
            timestamp: Utc::now(),
            stress_intensity: 0.6,
            duration: 86400,
            pre_stress_performance: 0.01,
            during_stress_performance: 0.005, // Poor performance during stress
            post_stress_performance: 0.01,
            recovery_time: Some(86400), // Slow recovery
        };
        
        poor_measurer.stress_events.push(poor_stress_event);
        
        let poor_response = poor_measurer.calculate_stress_response().unwrap();
        assert!(poor_response < 0.5, "Poor stress performance should yield low score");
    }

    #[test]
    fn test_hormesis_effect_calculation() {
        let mut measurer = AntifragilityMeasurer::new("test".to_string(), AntifragilityParams::default());
        
        // Create small stress events that lead to improvement
        for i in 0..5 {
            let small_stress = StressEvent {
                timestamp: Utc::now() + Duration::days(i),
                stress_intensity: 0.3, // Small stress
                duration: 3600,
                pre_stress_performance: 0.01,
                during_stress_performance: 0.009,
                post_stress_performance: 0.012, // Improvement after small stress
                recovery_time: Some(1800),
            };
            measurer.stress_events.push(small_stress);
        }
        
        let hormesis = measurer.calculate_hormesis_effect().unwrap();
        
        // Should detect hormesis effect
        assert!(hormesis > 0.0, "Should detect improvement from small stresses");
        assert!(hormesis <= 1.0);
        
        // Test without hormesis
        let mut no_hormesis_measurer = AntifragilityMeasurer::new("no_hormesis".to_string(), AntifragilityParams::default());
        let large_stress = StressEvent {
            timestamp: Utc::now(),
            stress_intensity: 0.8, // Large stress (not hormetic)
            duration: 86400,
            pre_stress_performance: 0.01,
            during_stress_performance: 0.005,
            post_stress_performance: 0.008, // No improvement
            recovery_time: Some(86400),
        };
        
        no_hormesis_measurer.stress_events.push(large_stress);
        
        let no_hormesis = no_hormesis_measurer.calculate_hormesis_effect().unwrap();
        assert_eq!(no_hormesis, 0.5, "Large stress shouldn't contribute to hormesis");
    }

    #[test]
    fn test_tail_benefit_calculation() {
        let mut measurer = AntifragilityMeasurer::new("test".to_string(), AntifragilityParams::default());
        
        // Create positive skew distribution (antifragile)
        let mut returns = vec![0.01; 80]; // Many small positive returns
        returns.extend(vec![-0.02; 15]); // Some moderate losses
        returns.extend(vec![0.15; 5]); // Few large gains
        
        for (i, ret) in returns.iter().enumerate() {
            measurer.update(*ret, 0.15, Utc::now() + Duration::days(i as i64)).unwrap();
        }
        
        let tail_benefit = measurer.calculate_tail_benefit().unwrap();
        
        // Should show benefit from positive tail events
        assert!(tail_benefit >= 0.0);
        assert!(tail_benefit <= 1.0);
        
        // With large positive tail events, should show some benefit
        assert!(tail_benefit > 0.0, "Positive skew should show tail benefit");
    }

    #[test]
    fn test_regime_adaptation_calculation() {
        let mut measurer = AntifragilityMeasurer::new("test".to_string(), AntifragilityParams::default());
        
        // Add adaptation measurements
        for i in 0..5 {
            let adaptation = AdaptationMeasurement {
                timestamp: Utc::now() + Duration::days(i),
                regime_change: RegimeChange::VolatilityIncrease { from: 0.1, to: 0.2 },
                adaptation_speed: 0.8,
                adaptation_quality: 0.85, // Good adaptation
                learning_effect: 0.1,
            };
            measurer.adaptation_history.push(adaptation);
        }
        
        let adaptation_score = measurer.calculate_regime_adaptation().unwrap();
        
        // Should reflect good adaptation quality
        assert_relative_eq!(adaptation_score, 0.85, epsilon = 0.01);
        assert!(adaptation_score >= 0.0 && adaptation_score <= 1.0);
    }

    #[test]
    fn test_classification_levels() {
        let measurer = AntifragilityMeasurer::new("test".to_string(), AntifragilityParams::default());
        
        // Test all classification levels
        assert_eq!(measurer.classify_antifragility_level(0.1), "Highly Fragile");
        assert_eq!(measurer.classify_antifragility_level(0.3), "Fragile");
        assert_eq!(measurer.classify_antifragility_level(0.5), "Robust (Neutral)");
        assert_eq!(measurer.classify_antifragility_level(0.7), "Antifragile");
        assert_eq!(measurer.classify_antifragility_level(0.9), "Highly Antifragile");
        
        // Test boundary conditions
        assert_eq!(measurer.classify_antifragility_level(0.2), "Highly Fragile");
        assert_eq!(measurer.classify_antifragility_level(0.4), "Fragile");
        assert_eq!(measurer.classify_antifragility_level(0.6), "Robust (Neutral)");
        assert_eq!(measurer.classify_antifragility_level(0.8), "Antifragile");
    }

    #[test]
    fn test_memory_management() {
        let mut measurer = AntifragilityMeasurer::new("test".to_string(), AntifragilityParams::default());
        
        // Add excessive data to test memory bounds
        for i in 0..15000 {
            measurer.update(0.01, 0.15, Utc::now()).unwrap();
        }
        
        // Should maintain reasonable memory usage
        assert!(measurer.return_history.len() <= 10000);
        assert!(measurer.volatility_history.len() <= 10000);
        
        // Add many stress events
        for i in 0..150 {
            let stress_event = StressEvent {
                timestamp: Utc::now() + Duration::seconds(i),
                stress_intensity: 0.5,
                duration: 3600,
                pre_stress_performance: 0.01,
                during_stress_performance: 0.01,
                post_stress_performance: 0.01,
                recovery_time: Some(1800),
            };
            measurer.stress_events.push(stress_event);
        }
        
        measurer.maintain_history_size();
        
        // Should be bounded
        assert!(measurer.stress_events.len() <= 100);
        
        // Add many adaptation events
        for i in 0..150 {
            let adaptation = AdaptationMeasurement {
                timestamp: Utc::now() + Duration::seconds(i),
                regime_change: RegimeChange::VolatilityIncrease { from: 0.1, to: 0.2 },
                adaptation_speed: 0.5,
                adaptation_quality: 0.5,
                learning_effect: 0.1,
            };
            measurer.adaptation_history.push(adaptation);
        }
        
        measurer.maintain_history_size();
        
        // Should be bounded
        assert!(measurer.adaptation_history.len() <= 100);
    }

    #[test]
    fn test_edge_case_zero_volatility() {
        let mut measurer = AntifragilityMeasurer::new("test".to_string(), AntifragilityParams::default());
        
        // Add data with zero volatility
        for _ in 0..120 {
            measurer.update(0.01, 0.0, Utc::now()).unwrap();
        }
        
        let result = measurer.measure_antifragility();
        assert!(result.is_ok());
        
        let measurement = result.unwrap();
        
        // Should handle gracefully
        assert!(measurement.overall_score >= 0.0 && measurement.overall_score <= 1.0);
        assert!(measurement.convexity.is_finite());
        assert!(measurement.volatility_benefit.is_finite());
    }

    #[test]
    fn test_edge_case_extreme_values() {
        let mut measurer = AntifragilityMeasurer::new("test".to_string(), AntifragilityParams::default());
        
        // Add extreme values
        measurer.update(f64::INFINITY, 0.15, Utc::now()).unwrap();
        measurer.update(-f64::INFINITY, 0.15, Utc::now()).unwrap();
        measurer.update(1000.0, 0.15, Utc::now()).unwrap(); // Very large return
        measurer.update(-1000.0, 0.15, Utc::now()).unwrap(); // Very large loss
        
        // Add enough normal data to meet minimum requirements
        for _ in 0..120 {
            measurer.update(0.01, 0.15, Utc::now()).unwrap();
        }
        
        let result = measurer.measure_antifragility();
        
        // Should either handle gracefully or fail cleanly
        if let Ok(measurement) = result {
            // If successful, results should be finite where possible
            if measurement.overall_score.is_finite() {
                assert!(measurement.overall_score >= 0.0);
                assert!(measurement.overall_score <= 1.0);
            }
        }
    }

    #[test]
    fn test_engine_integration() {
        let config = MacchiavelianConfig::aggressive_defaults();
        let mut engine = AntifragilityEngine::new(config.clone());
        
        // Test with normal market data
        let market_data = create_market_data(0.01, 0.15);
        let assessment = engine.assess(&market_data).unwrap();
        
        // Validate assessment structure
        assert!(assessment.score >= 0.0 && assessment.score <= 1.0);
        assert!(assessment.fragility_index >= 0.0 && assessment.fragility_index <= 1.0);
        assert!(assessment.robustness >= 0.0 && assessment.robustness <= 1.0);
        assert!(assessment.volatility_benefit >= 0.0 && assessment.volatility_benefit <= 1.0);
        assert!(assessment.stress_response >= 0.0 && assessment.stress_response <= 1.0);
        assert!(assessment.confidence >= 0.0 && assessment.confidence <= 1.0);
        
        // Fragility index should be inverse of score
        assert_relative_eq!(assessment.fragility_index, 1.0 - assessment.score, epsilon = 0.001);
        
        // Test calculate_antifragility method
        let calc_assessment = engine.calculate_antifragility(&market_data).unwrap();
        assert_relative_eq!(calc_assessment.antifragility_score, assessment.score, epsilon = 0.001);
    }

    #[test]
    fn test_params_configuration() {
        let mut custom_params = AntifragilityParams::default();
        custom_params.volatility_threshold = 0.5;
        custom_params.min_observation_period = 200;
        custom_params.enable_hormesis = false;
        custom_params.convexity_window = 100;
        
        let measurer = AntifragilityMeasurer::new("custom".to_string(), custom_params.clone());
        
        assert_eq!(measurer.params.volatility_threshold, 0.5);
        assert_eq!(measurer.params.min_observation_period, 200);
        assert!(!measurer.params.enable_hormesis);
        assert_eq!(measurer.params.convexity_window, 100);
    }

    #[test]
    fn test_regime_change_types() {
        let mut measurer = AntifragilityMeasurer::new("test".to_string(), AntifragilityParams::default());
        
        // Test different regime change types
        let vol_increase = RegimeChange::VolatilityIncrease { from: 0.1, to: 0.3 };
        let vol_decrease = RegimeChange::VolatilityDecrease { from: 0.3, to: 0.1 };
        let trend_change = RegimeChange::TrendChange { 
            from: TrendDirection::Bullish, 
            to: TrendDirection::Bearish 
        };
        let market_crash = RegimeChange::MarketCrash { severity: 0.8 };
        let recovery = RegimeChange::Recovery { strength: 0.6 };
        
        // Create adaptations for each type
        let adaptations = vec![
            AdaptationMeasurement {
                timestamp: Utc::now(),
                regime_change: vol_increase,
                adaptation_speed: 0.5,
                adaptation_quality: 0.7,
                learning_effect: 0.1,
            },
            AdaptationMeasurement {
                timestamp: Utc::now(),
                regime_change: vol_decrease,
                adaptation_speed: 0.6,
                adaptation_quality: 0.8,
                learning_effect: 0.1,
            },
            AdaptationMeasurement {
                timestamp: Utc::now(),
                regime_change: trend_change,
                adaptation_speed: 0.4,
                adaptation_quality: 0.6,
                learning_effect: 0.2,
            },
        ];
        
        measurer.adaptation_history = adaptations;
        
        // Should handle all regime change types
        let adaptation_score = measurer.calculate_regime_adaptation().unwrap();
        assert!(adaptation_score > 0.0);
        
        // Verify we can access adaptation history
        assert_eq!(measurer.get_adaptation_history().len(), 3);
    }

    #[test]
    fn test_data_accessors() {
        let mut measurer = AntifragilityMeasurer::new("test".to_string(), AntifragilityParams::default());
        
        // Add some data
        for i in 0..150 {
            let ret = 0.01 + (i as f64 * 0.0001);
            let vol = 0.15 + (i as f64 * 0.001);
            measurer.update(ret, vol, Utc::now()).unwrap();
        }
        
        let measurement = measurer.measure_antifragility().unwrap();
        
        // Test accessors
        assert_eq!(measurer.get_stress_events().len(), measurer.stress_events.len());
        assert_eq!(measurer.get_adaptation_history().len(), measurer.adaptation_history.len());
        
        let last_measurement = measurer.get_last_measurement();
        assert!(last_measurement.is_some());
        
        let last_measurement = last_measurement.unwrap();
        assert_relative_eq!(last_measurement.overall_score, measurement.overall_score, epsilon = 0.001);
    }

    #[test]\n    fn test_financial_invariants() {\n        let mut measurer = AntifragilityMeasurer::new(\"test\".to_string(), AntifragilityParams::default());\n        \n        // Test with multiple different scenarios\n        let scenarios = vec![\n            create_antifragile_pattern(),\n            create_fragile_pattern(),\n        ];\n        \n        for scenario in scenarios {\n            let mut scenario_measurer = AntifragilityMeasurer::new(\"scenario\".to_string(), AntifragilityParams::default());\n            \n            for (returns, volatility) in scenario {\n                scenario_measurer.update(returns, volatility, Utc::now()).unwrap();\n            }\n            \n            let measurement = scenario_measurer.measure_antifragility().unwrap();\n            \n            // Financial invariants\n            assert!(measurement.overall_score >= 0.0, \"Overall score must be non-negative\");\n            assert!(measurement.overall_score <= 1.0, \"Overall score must not exceed 100%\");\n            assert!(measurement.convexity >= 0.0, \"Convexity must be non-negative\");\n            assert!(measurement.convexity <= 1.0, \"Convexity must not exceed 100%\");\n            assert!(measurement.volatility_benefit >= 0.0, \"Volatility benefit must be non-negative\");\n            assert!(measurement.volatility_benefit <= 1.0, \"Volatility benefit must not exceed 100%\");\n            assert!(measurement.stress_response >= 0.0, \"Stress response must be non-negative\");\n            assert!(measurement.stress_response <= 1.0, \"Stress response must not exceed 100%\");\n            assert!(measurement.hormesis_effect >= 0.0, \"Hormesis effect must be non-negative\");\n            assert!(measurement.hormesis_effect <= 1.0, \"Hormesis effect must not exceed 100%\");\n            assert!(measurement.tail_benefit >= 0.0, \"Tail benefit must be non-negative\");\n            assert!(measurement.tail_benefit <= 1.0, \"Tail benefit must not exceed 100%\");\n            assert!(measurement.regime_adaptation >= 0.0, \"Regime adaptation must be non-negative\");\n            assert!(measurement.regime_adaptation <= 1.0, \"Regime adaptation must not exceed 100%\");\n            \n            // Component scores should match individual components\n            assert_relative_eq!(measurement.component_scores[\"convexity\"], measurement.convexity, epsilon = 0.001);\n            assert_relative_eq!(measurement.component_scores[\"volatility_benefit\"], measurement.volatility_benefit, epsilon = 0.001);\n            \n            // Overall score should be weighted average of components\n            let weights = [0.2, 0.2, 0.2, 0.1, 0.15, 0.15];\n            let components = [\n                measurement.convexity,\n                measurement.volatility_benefit,\n                measurement.stress_response,\n                measurement.hormesis_effect,\n                measurement.tail_benefit,\n                measurement.regime_adaptation,\n            ];\n            \n            let expected_overall: f64 = weights.iter().zip(components.iter()).map(|(w, c)| w * c).sum();\n            assert_relative_eq!(measurement.overall_score, expected_overall, epsilon = 0.01);\n        }\n    }\n}"}, {"old_string": "    #[test]\n    fn test_financial_invariants() {\n        let mut measurer = AntifragilityMeasurer::new(\"test\".to_string(), AntifragilityParams::default());\n        \n        // Test with multiple different scenarios\n        let scenarios = vec![\n            create_antifragile_pattern(),\n            create_fragile_pattern(),\n        ];\n        \n        for scenario in scenarios {\n            let mut scenario_measurer = AntifragilityMeasurer::new(\"scenario\".to_string(), AntifragilityParams::default());\n            \n            for (returns, volatility) in scenario {\n                scenario_measurer.update(returns, volatility, Utc::now()).unwrap();\n            }\n            \n            let measurement = scenario_measurer.measure_antifragility().unwrap();\n            \n            // Financial invariants\n            assert!(measurement.overall_score >= 0.0, \"Overall score must be non-negative\");\n            assert!(measurement.overall_score <= 1.0, \"Overall score must not exceed 100%\");\n            assert!(measurement.convexity >= 0.0, \"Convexity must be non-negative\");\n            assert!(measurement.convexity <= 1.0, \"Convexity must not exceed 100%\");\n            assert!(measurement.volatility_benefit >= 0.0, \"Volatility benefit must be non-negative\");\n            assert!(measurement.volatility_benefit <= 1.0, \"Volatility benefit must not exceed 100%\");\n            assert!(measurement.stress_response >= 0.0, \"Stress response must be non-negative\");\n            assert!(measurement.stress_response <= 1.0, \"Stress response must not exceed 100%\");\n            assert!(measurement.hormesis_effect >= 0.0, \"Hormesis effect must be non-negative\");\n            assert!(measurement.hormesis_effect <= 1.0, \"Hormesis effect must not exceed 100%\");\n            assert!(measurement.tail_benefit >= 0.0, \"Tail benefit must be non-negative\");\n            assert!(measurement.tail_benefit <= 1.0, \"Tail benefit must not exceed 100%\");\n            assert!(measurement.regime_adaptation >= 0.0, \"Regime adaptation must be non-negative\");\n            assert!(measurement.regime_adaptation <= 1.0, \"Regime adaptation must not exceed 100%\");\n            \n            // Component scores should match individual components\n            assert_relative_eq!(measurement.component_scores[\"convexity\"], measurement.convexity, epsilon = 0.001);\n            assert_relative_eq!(measurement.component_scores[\"volatility_benefit\"], measurement.volatility_benefit, epsilon = 0.001);\n            \n            // Overall score should be weighted average of components\n            let weights = [0.2, 0.2, 0.2, 0.1, 0.15, 0.15];\n            let components = [\n                measurement.convexity,\n                measurement.volatility_benefit,\n                measurement.stress_response,\n                measurement.hormesis_effect,\n                measurement.tail_benefit,\n                measurement.regime_adaptation,\n            ];\n            \n            let expected_overall: f64 = weights.iter().zip(components.iter()).map(|(w, c)| w * c).sum();\n            assert_relative_eq!(measurement.overall_score, expected_overall, epsilon = 0.01);\n        }\n    }", "new_string": "    #[test]\n    fn test_financial_invariants() {\n        let mut measurer = AntifragilityMeasurer::new(\"test\".to_string(), AntifragilityParams::default());\n        \n        // Test with multiple different scenarios\n        let scenarios = vec![\n            create_antifragile_pattern(),\n            create_fragile_pattern(),\n        ];\n        \n        for scenario in scenarios {\n            let mut scenario_measurer = AntifragilityMeasurer::new(\"scenario\".to_string(), AntifragilityParams::default());\n            \n            for (returns, volatility) in scenario {\n                scenario_measurer.update(returns, volatility, Utc::now()).unwrap();\n            }\n            \n            let measurement = scenario_measurer.measure_antifragility().unwrap();\n            \n            // Financial invariants\n            assert!(measurement.overall_score >= 0.0, \"Overall score must be non-negative\");\n            assert!(measurement.overall_score <= 1.0, \"Overall score must not exceed 100%\");\n            assert!(measurement.convexity >= 0.0, \"Convexity must be non-negative\");\n            assert!(measurement.convexity <= 1.0, \"Convexity must not exceed 100%\");\n            assert!(measurement.volatility_benefit >= 0.0, \"Volatility benefit must be non-negative\");\n            assert!(measurement.volatility_benefit <= 1.0, \"Volatility benefit must not exceed 100%\");\n            assert!(measurement.stress_response >= 0.0, \"Stress response must be non-negative\");\n            assert!(measurement.stress_response <= 1.0, \"Stress response must not exceed 100%\");\n            assert!(measurement.hormesis_effect >= 0.0, \"Hormesis effect must be non-negative\");\n            assert!(measurement.hormesis_effect <= 1.0, \"Hormesis effect must not exceed 100%\");\n            assert!(measurement.tail_benefit >= 0.0, \"Tail benefit must be non-negative\");\n            assert!(measurement.tail_benefit <= 1.0, \"Tail benefit must not exceed 100%\");\n            assert!(measurement.regime_adaptation >= 0.0, \"Regime adaptation must be non-negative\");\n            assert!(measurement.regime_adaptation <= 1.0, \"Regime adaptation must not exceed 100%\");\n            \n            // Component scores should match individual components\n            assert_relative_eq!(measurement.component_scores[\"convexity\"], measurement.convexity, epsilon = 0.001);\n            assert_relative_eq!(measurement.component_scores[\"volatility_benefit\"], measurement.volatility_benefit, epsilon = 0.001);\n            \n            // Overall score should be weighted average of components\n            let weights = [0.2, 0.2, 0.2, 0.1, 0.15, 0.15];\n            let components = [\n                measurement.convexity,\n                measurement.volatility_benefit,\n                measurement.stress_response,\n                measurement.hormesis_effect,\n                measurement.tail_benefit,\n                measurement.regime_adaptation,\n            ];\n            \n            let expected_overall: f64 = weights.iter().zip(components.iter()).map(|(w, c)| w * c).sum();\n            assert_relative_eq!(measurement.overall_score, expected_overall, epsilon = 0.01);\n        }\n    }\n}"}]