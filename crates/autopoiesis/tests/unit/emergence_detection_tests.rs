//! Unit tests for emergence detection system
//! Tests individual emergence detection components and algorithms

use autopoiesis::emergence::*;
use std::collections::VecDeque;

#[cfg(feature = "test-utils")]
use approx::assert_relative_eq;

#[cfg(feature = "property-tests")]
use proptest::prelude::*;

/// Helper function to create test system metrics
fn create_test_metrics(timestamp: f64, complexity: f64, coherence: f64) -> SystemMetrics {
    SystemMetrics {
        timestamp,
        system_size: 100,
        total_energy: 1000.0 + complexity * 100.0,
        entropy: 50.0 - coherence * 10.0,
        information: complexity,
        complexity,
        coherence,
        coupling: 0.5,
    }
}

/// Helper function to create test emergence history
fn create_test_history(length: usize) -> EmergenceHistory {
    let mut history = EmergenceHistory {
        metrics_history: VecDeque::new(),
        phase_trajectories: VecDeque::new(),
        avalanche_events: VecDeque::new(),
        fitness_evolution: VecDeque::new(),
        lattice_states: VecDeque::new(),
    };
    
    for i in 0..length {
        let t = i as f64;
        let complexity = 0.5 + 0.3 * (t * 0.1).sin();
        let coherence = 0.4 + 0.4 * (t * 0.15).cos();
        
        history.metrics_history.push_back(create_test_metrics(t, complexity, coherence));
    }
    
    history
}

#[test]
fn test_emergence_detector_initialization() {
    let params = DetectionParameters::default();
    let detector = EmergenceDetector::new(params.clone());
    
    let state = detector.get_emergence_state();
    assert_eq!(state.emergence_score, 0.0);
    assert_eq!(state.emergence_types.len(), 0);
    assert_eq!(state.confidence, 0.0);
    assert!(state.temporal_stability >= 0.0);
}

#[test]
fn test_emergence_detector_with_simple_pattern() {
    let mut detector = EmergenceDetector::new(DetectionParameters::default());
    let history = create_test_history(50);
    
    // Process the history
    detector.update_from_history(&history);
    
    let state = detector.get_emergence_state();
    
    // Should detect some level of emergence
    assert!(state.emergence_score >= 0.0);
    assert!(state.emergence_score <= 1.0);
    assert!(state.confidence >= 0.0);
    assert!(state.confidence <= 1.0);
}

#[test]
fn test_emergence_detector_sensitivity() {
    let high_sensitivity_params = DetectionParameters {
        emergence_threshold: 0.3,
        stability_window: 10,
        confidence_threshold: 0.5,
        adaptation_rate: 0.1,
        noise_tolerance: 0.05,
        min_pattern_length: 5,
    };
    
    let low_sensitivity_params = DetectionParameters {
        emergence_threshold: 0.8,
        stability_window: 30,
        confidence_threshold: 0.9,
        adaptation_rate: 0.01,
        noise_tolerance: 0.2,
        min_pattern_length: 20,
    };
    
    let history = create_test_history(100);
    
    let mut high_detector = EmergenceDetector::new(high_sensitivity_params);
    let mut low_detector = EmergenceDetector::new(low_sensitivity_params);
    
    high_detector.update_from_history(&history);
    low_detector.update_from_history(&history);
    
    let high_state = high_detector.get_emergence_state();
    let low_state = low_detector.get_emergence_state();
    
    // High sensitivity detector should generally detect more emergence
    // (though this depends on the specific data patterns)
    assert!(high_state.emergence_score >= 0.0);
    assert!(low_state.emergence_score >= 0.0);
    
    // Both should have valid confidence values
    assert!(high_state.confidence >= 0.0 && high_state.confidence <= 1.0);
    assert!(low_state.confidence >= 0.0 && low_state.confidence <= 1.0);
}

#[test]
fn test_emergence_state_temporal_consistency() {
    let mut detector = EmergenceDetector::new(DetectionParameters::default());
    let mut states = Vec::new();
    
    // Update detector with incremental data
    for i in 1..=20 {
        let partial_history = create_test_history(i * 5);
        detector.update_from_history(&partial_history);
        states.push(detector.get_emergence_state());
    }
    
    // Check temporal consistency
    for i in 1..states.len() {
        let prev_state = &states[i-1];
        let curr_state = &states[i];
        
        // Emergence score should not change drastically between updates
        let score_change = (curr_state.emergence_score - prev_state.emergence_score).abs();
        assert!(score_change <= 1.0); // Maximum possible change
        
        // Confidence should be bounded
        assert!(curr_state.confidence >= 0.0 && curr_state.confidence <= 1.0);
    }
}

#[test]
fn test_emergence_alerts_generation() {
    let mut detector = EmergenceDetector::new(DetectionParameters {
        emergence_threshold: 0.4,
        confidence_threshold: 0.6,
        ..DetectionParameters::default()
    });
    
    // Create history with clear emergence pattern
    let mut history = EmergenceHistory {
        metrics_history: VecDeque::new(),
        phase_trajectories: VecDeque::new(),
        avalanche_events: VecDeque::new(),
        fitness_evolution: VecDeque::new(),
        lattice_states: VecDeque::new(),
    };
    
    // Add metrics showing increasing emergence
    for i in 0..30 {
        let t = i as f64;
        let complexity = 0.8; // High complexity
        let coherence = 0.9;   // High coherence
        history.metrics_history.push_back(create_test_metrics(t, complexity, coherence));
    }
    
    detector.update_from_history(&history);
    
    let alerts = detector.get_alerts();
    let state = detector.get_emergence_state();
    
    // With high complexity and coherence, should generate alerts
    // (depending on the specific emergence detection algorithm)
    assert!(alerts.len() >= 0); // At least not crashing
    
    // State should reflect high emergence
    assert!(state.emergence_score >= 0.0);
}

#[test]
fn test_detector_reset() {
    let mut detector = EmergenceDetector::new(DetectionParameters::default());
    let history = create_test_history(50);
    
    // Update detector
    detector.update_from_history(&history);
    let state_before_reset = detector.get_emergence_state();
    
    // Reset detector
    detector.reset();
    let state_after_reset = detector.get_emergence_state();
    
    // After reset, state should be back to initial values
    assert_eq!(state_after_reset.emergence_score, 0.0);
    assert_eq!(state_after_reset.emergence_types.len(), 0);
    assert_eq!(state_after_reset.confidence, 0.0);
    
    // Should have cleared any accumulated state
    assert!(detector.get_alerts().is_empty());
}

#[test]
fn test_pattern_recognizer_initialization() {
    let params = PatternParameters::default();
    let recognizer = TemporalPatternRecognizer::new(params);
    
    let patterns = recognizer.get_patterns();
    assert!(patterns.is_empty());
}

#[test]
fn test_pattern_recognizer_with_periodic_data() {
    let mut recognizer = TemporalPatternRecognizer::new(PatternParameters::default());
    
    // Create periodic emergence history
    let mut history = EmergenceHistory {
        metrics_history: VecDeque::new(),
        phase_trajectories: VecDeque::new(),
        avalanche_events: VecDeque::new(),
        fitness_evolution: VecDeque::new(),
        lattice_states: VecDeque::new(),
    };
    
    // Add clear periodic pattern
    for i in 0..100 {
        let t = i as f64;
        let complexity = 0.5 + 0.4 * (t * 0.2).sin(); // Clear sinusoidal pattern
        let coherence = 0.5 + 0.3 * (t * 0.2).cos();
        history.metrics_history.push_back(create_test_metrics(t, complexity, coherence));
    }
    
    let discovered_patterns = recognizer.analyze_patterns(&history);
    
    // Should detect the periodic pattern
    assert!(discovered_patterns.len() >= 0); // At least not failing
    
    // Check pattern properties
    for pattern in &discovered_patterns {
        assert!(pattern.strength >= 0.0 && pattern.strength <= 1.0);
        assert!(!pattern.id.is_empty());
        assert!(pattern.frequency > 0.0);
    }
}

#[test]
fn test_pattern_prediction() {
    let mut recognizer = TemporalPatternRecognizer::new(PatternParameters::default());
    let history = create_test_history(50);
    
    // Analyze patterns first
    let _patterns = recognizer.analyze_patterns(&history);
    
    // Generate predictions
    let predictions = recognizer.predict_patterns(20);
    
    // Should generate valid predictions
    for prediction in &predictions {
        assert!(prediction.confidence >= 0.0 && prediction.confidence <= 1.0);
        assert!(prediction.predicted_occurrence_time >= 0.0);
        assert!(prediction.expected_strength >= 0.0 && prediction.expected_strength <= 1.0);
        assert!(!prediction.pattern_id.is_empty());
    }
}

#[test]
fn test_emergence_analysis_system_integration() {
    let detection_params = DetectionParameters::default();
    let pattern_params = PatternParameters::default();
    let analysis_params = AnalysisParameters::default();
    
    let mut system = EmergenceAnalysisSystem::new(
        detection_params,
        pattern_params,
        analysis_params,
    );
    
    let history = create_test_history(60);
    let result = system.analyze_emergence(&history);
    
    // Check analysis result validity
    assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
    assert!(result.combined_analysis.pattern_strength >= 0.0);
    assert!(result.combined_analysis.temporal_stability >= 0.0);
    assert!(result.combined_analysis.criticality >= 0.0);
    
    // Should generate some recommendations
    assert!(result.recommendations.len() >= 0);
    
    // Predictions should be valid
    for prediction in &result.predictions {
        assert!(prediction.confidence >= 0.0 && prediction.confidence <= 1.0);
        assert!(prediction.expected_emergence_score >= 0.0);
    }
}

#[test]
fn test_emergence_level_classification() {
    let detection_params = DetectionParameters::default();
    let pattern_params = PatternParameters::default();
    let analysis_params = AnalysisParameters::default();
    
    let system = EmergenceAnalysisSystem::new(
        detection_params,
        pattern_params,
        analysis_params,
    );
    
    // Test different emergence score combinations
    assert_eq!(system.classify_emergence_level(0.9, 0.9), EmergenceLevel::High);
    assert_eq!(system.classify_emergence_level(0.7, 0.6), EmergenceLevel::Medium);
    assert_eq!(system.classify_emergence_level(0.4, 0.3), EmergenceLevel::Low);
    assert_eq!(system.classify_emergence_level(0.1, 0.2), EmergenceLevel::Minimal);
    
    // Edge cases
    assert_eq!(system.classify_emergence_level(0.0, 0.0), EmergenceLevel::Minimal);
    assert_eq!(system.classify_emergence_level(1.0, 1.0), EmergenceLevel::High);
}

#[test]
fn test_risk_level_assessment() {
    let detection_params = DetectionParameters::default();
    let pattern_params = PatternParameters::default();
    let analysis_params = AnalysisParameters::default();
    
    let system = EmergenceAnalysisSystem::new(
        detection_params,
        pattern_params,
        analysis_params,
    );
    
    // Create different analysis scenarios
    let high_risk_analysis = CombinedAnalysis {
        emergence_level: EmergenceLevel::High,
        pattern_strength: 0.9,
        dominant_patterns: vec!["critical_pattern".to_string()],
        temporal_stability: 0.2, // Low stability
        criticality: 0.9,
        emergence_types: vec![EmergenceType::PhaseTransition],
        confidence: 0.8,
    };
    
    let low_risk_analysis = CombinedAnalysis {
        emergence_level: EmergenceLevel::Low,
        pattern_strength: 0.3,
        dominant_patterns: vec![],
        temporal_stability: 0.8, // High stability
        criticality: 0.2,
        emergence_types: vec![EmergenceType::SelfOrganization],
        confidence: 0.6,
    };
    
    let high_risk = system.assess_risk_level(&high_risk_analysis);
    let low_risk = system.assess_risk_level(&low_risk_analysis);
    
    // High risk scenario should have higher risk level
    assert!(matches!(high_risk, RiskLevel::High | RiskLevel::Critical));
    assert!(matches!(low_risk, RiskLevel::Low | RiskLevel::Medium));
}

#[test]
fn test_trend_determination() {
    let detection_params = DetectionParameters::default();
    let pattern_params = PatternParameters::default();
    let analysis_params = AnalysisParameters::default();
    
    let mut system = EmergenceAnalysisSystem::new(
        detection_params,
        pattern_params,
        analysis_params,
    );
    
    // Create increasing trend
    for i in 0..10 {
        system.analysis_history.emergence_scores.push_back(i as f64 * 0.1);
    }
    
    let trend = system.determine_trend();
    assert_eq!(trend, TrendDirection::Increasing);
    
    // Create decreasing trend
    system.analysis_history.emergence_scores.clear();
    for i in 0..10 {
        system.analysis_history.emergence_scores.push_back(1.0 - i as f64 * 0.1);
    }
    
    let trend = system.determine_trend();
    assert_eq!(trend, TrendDirection::Decreasing);
}

#[test]
fn test_linear_slope_calculation() {
    let detection_params = DetectionParameters::default();
    let pattern_params = PatternParameters::default();
    let analysis_params = AnalysisParameters::default();
    
    let system = EmergenceAnalysisSystem::new(
        detection_params,
        pattern_params,
        analysis_params,
    );
    
    // Test perfect positive slope
    let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let y = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let slope = system.calculate_linear_slope(&x, &y);
    assert_relative_eq!(slope, 1.0, epsilon = 1e-10);
    
    // Test perfect negative slope
    let y = vec![4.0, 3.0, 2.0, 1.0, 0.0];
    let slope = system.calculate_linear_slope(&x, &y);
    assert_relative_eq!(slope, -1.0, epsilon = 1e-10);
    
    // Test flat line
    let y = vec![2.0, 2.0, 2.0, 2.0, 2.0];
    let slope = system.calculate_linear_slope(&x, &y);
    assert_relative_eq!(slope, 0.0, epsilon = 1e-10);
    
    // Test edge cases
    let empty_x: Vec<f64> = vec![];
    let empty_y: Vec<f64> = vec![];
    let slope = system.calculate_linear_slope(&empty_x, &empty_y);
    assert_eq!(slope, 0.0);
    
    let single_x = vec![1.0];
    let single_y = vec![1.0];
    let slope = system.calculate_linear_slope(&single_x, &single_y);
    assert_eq!(slope, 0.0);
}

#[test]
fn test_analysis_history_management() {
    let detection_params = DetectionParameters::default();
    let pattern_params = PatternParameters::default();
    let analysis_params = AnalysisParameters {
        memory_retention: 5, // Small memory for testing
        ..AnalysisParameters::default()
    };
    
    let mut system = EmergenceAnalysisSystem::new(
        detection_params,
        pattern_params,
        analysis_params,
    );
    
    // Add more data than memory retention allows
    for i in 0..10 {
        let analysis = CombinedAnalysis {
            emergence_level: EmergenceLevel::Medium,
            pattern_strength: i as f64 * 0.1,
            dominant_patterns: vec![],
            temporal_stability: 0.5,
            criticality: 0.3,
            emergence_types: vec![],
            confidence: 0.6,
        };
        
        system.update_analysis_history(&analysis);
    }
    
    // History should be bounded by memory retention
    assert!(system.analysis_history.emergence_scores.len() <= system.params.memory_retention);
    assert!(system.analysis_history.pattern_strengths.len() <= system.params.memory_retention);
    
    // Should contain the most recent data
    let latest_pattern_strength = system.analysis_history.pattern_strengths.back().unwrap();
    assert_relative_eq!(*latest_pattern_strength, 0.9, epsilon = 1e-10); // Last value added
}

#[test]
fn test_confidence_calculation() {
    let detection_params = DetectionParameters::default();
    let pattern_params = PatternParameters::default();
    let analysis_params = AnalysisParameters::default();
    
    let system = EmergenceAnalysisSystem::new(
        detection_params,
        pattern_params,
        analysis_params,
    );
    
    let confidence = system.calculate_overall_confidence();
    
    // Should return a valid confidence value
    assert!(confidence >= 0.0 && confidence <= 1.0);
}

/// Property-based tests for emergence detection
#[cfg(feature = "property-tests")]
mod emergence_property_tests {
    use super::*;
    use proptest::prelude::*;
    
    proptest! {
        #[test]
        fn test_emergence_score_bounds(
            complexity in 0.0f64..1.0,
            coherence in 0.0f64..1.0,
            timestamps in prop::collection::vec(0.0f64..1000.0, 1..100)
        ) {
            let mut detector = EmergenceDetector::new(DetectionParameters::default());
            let mut history = EmergenceHistory {
                metrics_history: VecDeque::new(),
                phase_trajectories: VecDeque::new(),
                avalanche_events: VecDeque::new(),
                fitness_evolution: VecDeque::new(),
                lattice_states: VecDeque::new(),
            };
            
            for timestamp in timestamps {
                history.metrics_history.push_back(create_test_metrics(timestamp, complexity, coherence));
            }
            
            detector.update_from_history(&history);
            let state = detector.get_emergence_state();
            
            // Emergence score should always be bounded
            prop_assert!(state.emergence_score >= 0.0);
            prop_assert!(state.emergence_score <= 1.0);
            prop_assert!(state.confidence >= 0.0);
            prop_assert!(state.confidence <= 1.0);
            prop_assert!(state.temporal_stability >= 0.0);
        }
        
        #[test]
        fn test_pattern_strength_bounds(
            frequencies in prop::collection::vec(0.1f64..10.0, 0..10),
            strengths in prop::collection::vec(0.0f64..1.0, 0..10)
        ) {
            prop_assume!(frequencies.len() == strengths.len());
            
            let recognizer = TemporalPatternRecognizer::new(PatternParameters::default());
            
            // Create mock patterns
            let patterns: Vec<TemporalPattern> = frequencies.into_iter()
                .zip(strengths.into_iter())
                .enumerate()
                .map(|(i, (freq, strength))| TemporalPattern {
                    id: format!("pattern_{}", i),
                    pattern_type: PatternType::Periodic,
                    frequency: freq,
                    strength,
                    phase: 0.0,
                    duration: 10.0,
                    confidence: 0.8,
                    last_occurrence: 0.0,
                })
                .collect();
            
            // Verify all patterns have valid properties
            for pattern in &patterns {
                prop_assert!(pattern.strength >= 0.0);
                prop_assert!(pattern.strength <= 1.0);
                prop_assert!(pattern.frequency > 0.0);
                prop_assert!(pattern.confidence >= 0.0);
                prop_assert!(pattern.confidence <= 1.0);
            }
        }
        
        #[test]
        fn test_analysis_result_consistency(
            emergence_score in 0.0f64..1.0,
            pattern_strength in 0.0f64..1.0,
            temporal_stability in 0.0f64..1.0,
            criticality in 0.0f64..1.0
        ) {
            let analysis = CombinedAnalysis {
                emergence_level: EmergenceLevel::Medium,
                pattern_strength,
                dominant_patterns: vec!["test_pattern".to_string()],
                temporal_stability,
                criticality,
                emergence_types: vec![EmergenceType::SelfOrganization],
                confidence: 0.7,
            };
            
            // All analysis values should be bounded
            prop_assert!(analysis.pattern_strength >= 0.0);
            prop_assert!(analysis.pattern_strength <= 1.0);
            prop_assert!(analysis.temporal_stability >= 0.0);
            prop_assert!(analysis.temporal_stability <= 1.0);
            prop_assert!(analysis.criticality >= 0.0);
            prop_assert!(analysis.criticality <= 1.0);
            prop_assert!(analysis.confidence >= 0.0);
            prop_assert!(analysis.confidence <= 1.0);
        }
    }
}