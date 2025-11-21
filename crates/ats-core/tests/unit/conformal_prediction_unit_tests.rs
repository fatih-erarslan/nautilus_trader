//! Unit Tests for Conformal Prediction - London School TDD Approach
//!
//! These tests follow the London School (mockist) approach, focusing on:
//! - Interaction testing between objects
//! - Behavior verification through mocks
//! - Contract definition and enforcement
//! - Outside-in development flow

use ats_core::{
    conformal::ConformalPredictor,
    config::{AtsCpConfig, QuantileMethod, ConformalConfig},
    types::{AtsCpVariant, Confidence},
    error::AtsCoreError,
    test_utils::create_test_config,
};
use std::time::{Duration, Instant};
use proptest::prelude::*;

/// Test fixture for conformal prediction unit tests
struct ConformalPredictionTestFixture {
    config: AtsCpConfig,
    predictor: ConformalPredictor,
    mock_calibration_data: Vec<f64>,
    mock_predictions: Vec<f64>,
}

impl ConformalPredictionTestFixture {
    fn new() -> Self {
        let config = create_test_config();
        let predictor = ConformalPredictor::new(&config).unwrap();
        let mock_calibration_data = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
        let mock_predictions = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        
        Self {
            config,
            predictor,
            mock_calibration_data,
            mock_predictions,
        }
    }
}

/// London School TDD: Test object collaboration patterns
mod collaboration_tests {
    use super::*;
    
    #[test]
    fn test_conformal_predictor_quantile_computer_collaboration() {
        // Arrange: Set up mock collaboration
        let mut fixture = ConformalPredictionTestFixture::new();
        
        // Act: Execute the collaboration
        let start_time = Instant::now();
        let result = fixture.predictor.predict(
            &fixture.mock_predictions,
            &fixture.mock_calibration_data,
        );
        let execution_time = start_time.elapsed();
        
        // Assert: Verify the collaboration succeeded
        assert!(result.is_ok(), "Conformal prediction collaboration should succeed");
        
        let intervals = result.unwrap();
        assert_eq!(intervals.len(), fixture.mock_predictions.len(),
                  "Should produce intervals for each prediction");
        
        // Verify collaboration timing constraints
        assert!(execution_time < Duration::from_micros(50),
               "Collaboration should meet latency requirements: {:?}", execution_time);
        
        // Verify interval structure (collaboration contract)
        for (i, (lower, upper)) in intervals.iter().enumerate() {
            assert!(lower <= upper, 
                   "Collaboration contract: lower <= upper for interval {}", i);
            assert!(lower.is_finite() && upper.is_finite(),
                   "Collaboration contract: intervals must be finite for interval {}", i);
        }
    }
    
    #[test]
    fn test_ats_cp_algorithm_component_collaboration() {
        // Arrange: Mock multi-component collaboration
        let mut fixture = ConformalPredictionTestFixture::new();
        
        let logits = vec![2.0, 1.0, 0.5];
        let calibration_logits = vec![
            vec![1.8, 1.2, 0.6],
            vec![2.1, 0.9, 0.4],
            vec![1.9, 1.1, 0.5],
            vec![2.0, 1.0, 0.3],
            vec![1.7, 1.3, 0.7],
        ];
        let calibration_labels = vec![0, 0, 0, 1, 1];
        let confidence = 0.95;
        let variant = AtsCpVariant::GQ;
        
        // Act: Execute complex collaboration workflow
        let start_time = Instant::now();
        let result = fixture.predictor.ats_cp_predict(
            &logits,
            &calibration_logits,
            &calibration_labels,
            confidence,
            variant.clone(),
        );
        let execution_time = start_time.elapsed();
        
        // Assert: Verify multi-component collaboration
        assert!(result.is_ok(), "ATS-CP collaboration should succeed");
        
        let ats_cp_result = result.unwrap();
        
        // Verify collaboration contracts
        assert!(!ats_cp_result.conformal_set.is_empty(),
               "Collaboration contract: conformal set cannot be empty");
        assert_eq!(ats_cp_result.calibrated_probabilities.len(), logits.len(),
                  "Collaboration contract: probabilities match logits length");
        assert_eq!(ats_cp_result.variant, variant,
                  "Collaboration contract: variant preserved");
        assert_eq!(ats_cp_result.coverage_guarantee, confidence,
                  "Collaboration contract: coverage guarantee preserved");
        assert!(ats_cp_result.optimal_temperature > 0.0,
               "Collaboration contract: temperature must be positive");
        
        // Verify timing constraints
        assert!(execution_time < Duration::from_micros(50),
               "Multi-component collaboration should meet latency: {:?}", execution_time);
        
        // Verify probability normalization (component contract)
        let prob_sum: f64 = ats_cp_result.calibrated_probabilities.iter().sum();
        assert!((prob_sum - 1.0).abs() < 1e-6,
               "Collaboration contract: probabilities should sum to 1.0, got {}", prob_sum);
    }
}

/// London School TDD: Test interaction sequences
mod interaction_sequence_tests {
    use super::*;
    
    #[test]
    fn test_predict_method_interaction_sequence() {
        // Arrange: Set up interaction tracking
        let mut fixture = ConformalPredictionTestFixture::new();
        
        // Act: Execute method with interaction tracking
        let result = fixture.predictor.predict(
            &fixture.mock_predictions,
            &fixture.mock_calibration_data,
        );
        
        // Assert: Verify interaction sequence
        assert!(result.is_ok(), "Predict interaction sequence should complete");
        
        // Expected sequence: validation -> quantile computation -> interval computation
        // This would be verified through mock interactions in real implementation
        
        let performance_stats = fixture.predictor.get_performance_stats();
        assert_eq!(performance_stats.0, 1, "Should record one interaction");
        assert!(performance_stats.1 > 0, "Should record execution time");
    }
    
    #[test]
    fn test_adaptive_prediction_interaction_sequence() {
        // Arrange: Mock adaptive prediction workflow
        let mut fixture = ConformalPredictionTestFixture::new();
        
        let predictions = vec![1.0, 2.0, 3.0];
        let true_values = vec![1.1, 1.9, 3.1];
        let confidence = 0.95;
        
        // Act: Execute adaptive interaction sequence
        let result = fixture.predictor.predict_adaptive(&predictions, &true_values, confidence);
        
        // Assert: Verify adaptive interaction workflow
        assert!(result.is_ok(), "Adaptive prediction interactions should succeed");
        
        // Expected interactions: residual computation -> calibration update -> quantile computation -> interval computation
        let intervals = result.unwrap();
        assert_eq!(intervals.len(), predictions.len(),
                  "Adaptive interactions should produce correct output");
        
        // Verify calibration was updated through interactions
        // (In real implementation, this would be verified through mock expectations)
    }
}

/// London School TDD: Test behavior verification
mod behavior_verification_tests {
    use super::*;
    
    #[test]
    fn test_quantile_method_behavior_contracts() {
        // Test each quantile method's behavior contract
        let mut fixture = ConformalPredictionTestFixture::new();
        
        let test_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let confidence = 0.9;
        
        // Test Linear method behavior
        let linear_result = fixture.predictor.compute_quantile_linear(&test_data, confidence);
        assert!(linear_result.is_ok(), "Linear quantile behavior should be valid");
        let linear_value = linear_result.unwrap();
        assert!(linear_value >= test_data[0] && linear_value <= test_data[test_data.len()-1],
               "Linear quantile behavior: result should be within data range");
        
        // Test Nearest method behavior  
        let nearest_result = fixture.predictor.compute_quantile_nearest(&test_data, confidence);
        assert!(nearest_result.is_ok(), "Nearest quantile behavior should be valid");
        let nearest_value = nearest_result.unwrap();
        assert!(test_data.contains(&nearest_value) || 
               test_data.iter().any(|&x| (x - nearest_value).abs() < 1e-10),
               "Nearest quantile behavior: result should be close to data point");
        
        // Test Higher method behavior
        let higher_result = fixture.predictor.compute_quantile_higher(&test_data, confidence);
        assert!(higher_result.is_ok(), "Higher quantile behavior should be valid");
        let higher_value = higher_result.unwrap();
        assert!(higher_value >= linear_value,
               "Higher quantile behavior: should be >= linear quantile");
        
        // Test Lower method behavior
        let lower_result = fixture.predictor.compute_quantile_lower(&test_data, confidence);
        assert!(lower_result.is_ok(), "Lower quantile behavior should be valid");
        let lower_value = lower_result.unwrap();
        assert!(lower_value <= linear_value,
               "Lower quantile behavior: should be <= linear quantile");
        
        // Test Midpoint method behavior
        let midpoint_result = fixture.predictor.compute_quantile_midpoint(&test_data, confidence);
        assert!(midpoint_result.is_ok(), "Midpoint quantile behavior should be valid");
        let midpoint_value = midpoint_result.unwrap();
        assert!(midpoint_value >= lower_value && midpoint_value <= higher_value,
               "Midpoint quantile behavior: should be between lower and higher");
    }
    
    #[test]
    fn test_conformal_score_computation_behavior() {
        // Test behavior contracts for different ATS-CP variants
        let mut fixture = ConformalPredictionTestFixture::new();
        
        let logits = vec![2.0, 1.0, 0.5];
        let calibration_logits = vec![
            vec![1.8, 1.2, 0.6],
            vec![2.1, 0.9, 0.4],
            vec![1.9, 1.1, 0.5],
        ];
        let calibration_labels = vec![0, 0, 1];
        
        // Test GQ variant behavior
        let gq_result = fixture.predictor.ats_cp_predict(
            &logits, &calibration_logits, &calibration_labels, 0.95, AtsCpVariant::GQ
        );
        assert!(gq_result.is_ok(), "GQ variant behavior should be valid");
        
        // Test AQ variant behavior  
        let aq_result = fixture.predictor.ats_cp_predict(
            &logits, &calibration_logits, &calibration_labels, 0.95, AtsCpVariant::AQ
        );
        assert!(aq_result.is_ok(), "AQ variant behavior should be valid");
        
        // Test MGQ variant behavior
        let mgq_result = fixture.predictor.ats_cp_predict(
            &logits, &calibration_logits, &calibration_labels, 0.95, AtsCpVariant::MGQ
        );
        assert!(mgq_result.is_ok(), "MGQ variant behavior should be valid");
        
        // Test MAQ variant behavior
        let maq_result = fixture.predictor.ats_cp_predict(
            &logits, &calibration_logits, &calibration_labels, 0.95, AtsCpVariant::MAQ
        );
        assert!(maq_result.is_ok(), "MAQ variant behavior should be valid");
        
        // Verify behavioral contracts across variants
        let results = vec![gq_result.unwrap(), aq_result.unwrap(), mgq_result.unwrap(), maq_result.unwrap()];
        
        for (i, result) in results.iter().enumerate() {
            assert!(!result.conformal_set.is_empty(),
                   "Variant {} behavior: conformal set should not be empty", i);
            assert!(result.optimal_temperature > 0.0,
                   "Variant {} behavior: temperature should be positive", i);
            assert!(result.quantile_threshold >= 0.0,
                   "Variant {} behavior: quantile threshold should be non-negative", i);
            
            // Verify probability constraints
            for &prob in &result.calibrated_probabilities {
                assert!(prob >= 0.0 && prob <= 1.0,
                       "Variant {} behavior: probabilities should be in [0,1]", i);
            }
        }
    }
}

/// London School TDD: Contract enforcement tests  
mod contract_enforcement_tests {
    use super::*;
    
    #[test]
    fn test_input_validation_contracts() {
        let mut fixture = ConformalPredictionTestFixture::new();
        
        // Contract: empty predictions should fail
        let empty_predictions = vec![];
        let result = fixture.predictor.predict(&empty_predictions, &fixture.mock_calibration_data);
        assert!(result.is_err(), "Contract violation: empty predictions should fail");
        assert!(matches!(result.unwrap_err(), AtsCoreError::ValidationError { .. }),
               "Contract enforcement: should return validation error");
        
        // Contract: insufficient calibration data should fail
        let small_calibration = vec![0.1, 0.2]; // Below minimum
        let result = fixture.predictor.predict(&fixture.mock_predictions, &small_calibration);
        assert!(result.is_err(), "Contract violation: insufficient calibration should fail");
        
        // Contract: invalid confidence should fail
        let result = fixture.predictor.predict_detailed(
            &fixture.mock_predictions, 
            &fixture.mock_calibration_data, 
            1.5 // Invalid confidence > 1
        );
        assert!(result.is_err(), "Contract violation: invalid confidence should fail");
        
        // Contract: negative confidence should fail
        let result = fixture.predictor.predict_detailed(
            &fixture.mock_predictions, 
            &fixture.mock_calibration_data, 
            -0.1 // Invalid confidence < 0
        );
        assert!(result.is_err(), "Contract violation: negative confidence should fail");
    }
    
    #[test]
    fn test_latency_contract_enforcement() {
        // Create config with strict latency requirements
        let strict_config = AtsCpConfig {
            conformal: ConformalConfig {
                target_latency_us: 1, // Very strict requirement
                ..Default::default()
            },
            ..Default::default()
        };
        
        let mut predictor = ConformalPredictor::new(&strict_config).unwrap();
        
        // Create large dataset that might exceed latency
        let large_predictions: Vec<f64> = (0..1000).map(|i| i as f64).collect();
        let large_calibration: Vec<f64> = (0..1000).map(|i| i as f64 * 0.001).collect();
        
        let result = predictor.predict(&large_predictions, &large_calibration);
        
        // Contract enforcement: should either succeed within latency or fail with timeout
        match result {
            Ok(_) => {
                // If it succeeds, it must have met latency requirements
                let (_, avg_latency, _) = predictor.get_performance_stats();
                assert!(avg_latency <= 1000, // 1μs in nanoseconds
                       "Contract enforcement: must meet latency if successful");
            },
            Err(AtsCoreError::TimeoutError { .. }) => {
                // Acceptable - contract enforced by rejecting slow operations
            },
            Err(e) => panic!("Unexpected error type: {:?}", e),
        }
    }
    
    #[test] 
    fn test_ats_cp_dimension_contracts() {
        let mut fixture = ConformalPredictionTestFixture::new();
        
        let logits = vec![2.0, 1.0, 0.5];
        let calibration_logits = vec![
            vec![1.8, 1.2, 0.6],
            vec![2.1, 0.9, 0.4],
        ];
        // Mismatched dimensions: 2 calibration samples, 3 labels
        let calibration_labels = vec![0, 0, 1]; 
        
        let result = fixture.predictor.ats_cp_predict(
            &logits, &calibration_logits, &calibration_labels, 0.95, AtsCpVariant::GQ
        );
        
        // Contract: dimension mismatch should be enforced
        assert!(result.is_err(), "Contract violation: dimension mismatch should fail");
        assert!(matches!(result.unwrap_err(), AtsCoreError::DimensionMismatchError { .. }),
               "Contract enforcement: should return dimension mismatch error");
    }
}

/// London School TDD: Mock interaction tests
mod mock_interaction_tests {
    use super::*;
    
    #[test]
    fn test_temperature_scaler_interaction() {
        // This test would use actual mocks in full implementation
        // Here we test the interaction patterns that would be mocked
        
        let mut fixture = ConformalPredictionTestFixture::new();
        
        let logits = vec![2.0, 1.0, 0.5];
        let temperature = 1.5;
        
        // Mock expectation: temperature_scaled_softmax should be called
        // In full mock implementation, we would verify:
        // mock_temperature_scaler.expect_scale_logits(logits.clone(), temperature, expected_result);
        
        let result = fixture.predictor.temperature_scaled_softmax(&logits, temperature);
        
        assert!(result.is_ok(), "Temperature scaler interaction should succeed");
        let probabilities = result.unwrap();
        
        // Verify interaction contract
        assert_eq!(probabilities.len(), logits.len(),
                  "Interaction contract: output length should match input");
        
        let sum: f64 = probabilities.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6,
               "Interaction contract: probabilities should sum to 1.0");
        
        for &prob in &probabilities {
            assert!(prob >= 0.0 && prob <= 1.0,
                   "Interaction contract: probabilities should be in [0,1]");
        }
    }
    
    #[test]
    fn test_quantile_computer_interaction() {
        // Test quantile computer interaction patterns
        let mut fixture = ConformalPredictionTestFixture::new();
        
        let test_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let confidence = 0.9;
        
        // Mock expectation: compute_quantile_fast should delegate to specific method
        let result = fixture.predictor.compute_quantile_fast(&test_data, confidence);
        
        assert!(result.is_ok(), "Quantile computer interaction should succeed");
        let quantile = result.unwrap();
        
        // Verify interaction contract
        assert!(quantile >= test_data[0] && quantile <= test_data[test_data.len()-1],
               "Interaction contract: quantile should be within data range");
        assert!(quantile.is_finite(),
               "Interaction contract: quantile should be finite");
    }
}

/// Property-based testing for contracts
mod property_based_contracts {
    use super::*;
    
    proptest! {
        #[test]
        fn test_prediction_intervals_contract(
            predictions in prop::collection::vec(any::<f64>(), 1..100),
            calibration in prop::collection::vec(any::<f64>(), 10..100)
        ) {
            let mut fixture = ConformalPredictionTestFixture::new();
            
            // Filter out invalid inputs
            let valid_predictions: Vec<f64> = predictions.into_iter()
                .filter(|x| x.is_finite())
                .take(50)
                .collect();
            
            let valid_calibration: Vec<f64> = calibration.into_iter()
                .filter(|x| x.is_finite() && *x >= 0.0)
                .take(50)
                .collect();
            
            if valid_predictions.is_empty() || valid_calibration.len() < 10 {
                return Ok(());
            }
            
            let result = fixture.predictor.predict(&valid_predictions, &valid_calibration);
            
            if let Ok(intervals) = result {
                // Property: intervals should have same length as predictions
                prop_assert_eq!(intervals.len(), valid_predictions.len());
                
                // Property: all intervals should be valid
                for (i, (lower, upper)) in intervals.iter().enumerate() {
                    prop_assert!(lower.is_finite(), "Lower bound should be finite for interval {}", i);
                    prop_assert!(upper.is_finite(), "Upper bound should be finite for interval {}", i);
                    prop_assert!(lower <= upper, "Lower <= upper should hold for interval {}", i);
                }
            }
        }
        
        #[test]
        fn test_temperature_scaling_contract(
            logits in prop::collection::vec(any::<f64>(), 1..10),
            temperature in 0.1f64..10.0f64
        ) {
            let mut fixture = ConformalPredictionTestFixture::new();
            
            // Filter out invalid logits
            let valid_logits: Vec<f64> = logits.into_iter()
                .filter(|x| x.is_finite())
                .collect();
            
            if valid_logits.is_empty() {
                return Ok(());
            }
            
            let result = fixture.predictor.temperature_scaled_softmax(&valid_logits, temperature);
            
            if let Ok(probabilities) = result {
                // Property: probabilities should sum to 1
                let sum: f64 = probabilities.iter().sum();
                prop_assert!((sum - 1.0).abs() < 1e-6, "Probabilities should sum to 1.0, got {}", sum);
                
                // Property: all probabilities should be in [0, 1]
                for (i, &prob) in probabilities.iter().enumerate() {
                    prop_assert!(prob >= 0.0, "Probability {} should be non-negative", i);
                    prop_assert!(prob <= 1.0, "Probability {} should be <= 1.0", i);
                    prop_assert!(prob.is_finite(), "Probability {} should be finite", i);
                }
                
                // Property: length should match input
                prop_assert_eq!(probabilities.len(), valid_logits.len());
            }
        }
    }
}

/// Performance contract tests
mod performance_contract_tests {
    use super::*;
    
    #[test]
    fn test_latency_performance_contract() {
        let mut fixture = ConformalPredictionTestFixture::new();
        
        // Small dataset should meet strict latency requirements
        let small_predictions = vec![1.0, 2.0, 3.0];
        let small_calibration = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
        
        let start_time = Instant::now();
        let result = fixture.predictor.predict(&small_predictions, &small_calibration);
        let execution_time = start_time.elapsed();
        
        assert!(result.is_ok(), "Performance contract: small dataset should succeed");
        assert!(execution_time < Duration::from_micros(50),
               "Performance contract: small dataset should be fast: {:?}", execution_time);
    }
    
    #[test] 
    fn test_parallel_performance_contract() {
        let mut fixture = ConformalPredictionTestFixture::new();
        
        // Large dataset for parallel processing
        let large_predictions: Vec<f64> = (0..1000).map(|i| i as f64 * 0.1).collect();
        let calibration_data: Vec<f64> = (0..100).map(|i| i as f64 * 0.01).collect();
        let confidence = 0.95;
        
        let start_time = Instant::now();
        let result = fixture.predictor.predict_parallel(&large_predictions, &calibration_data, confidence);
        let parallel_time = start_time.elapsed();
        
        // Compare with sequential version
        let start_time = Instant::now();
        let sequential_result = fixture.predictor.predict(&large_predictions, &calibration_data);
        let sequential_time = start_time.elapsed();
        
        if result.is_ok() && sequential_result.is_ok() {
            // Performance contract: parallel should be at least as good as sequential
            // (In practice, parallel might be slower for small datasets due to overhead)
            println!("Parallel time: {:?}, Sequential time: {:?}", parallel_time, sequential_time);
            
            // Verify results are consistent
            let parallel_intervals = result.unwrap();
            let sequential_intervals = sequential_result.unwrap();
            
            assert_eq!(parallel_intervals.len(), sequential_intervals.len(),
                      "Performance contract: parallel should produce same number of intervals");
        }
    }
}

#[cfg(test)]
mod integration_with_test_framework {
    use super::*;
    use ats_core::test_framework::{TestFramework, swarm_utils};
    
    #[tokio::test]
    async fn test_unit_tests_swarm_coordination() {
        // Initialize test framework for swarm coordination
        let mut framework = TestFramework::new(
            "conformal_prediction_swarm".to_string(),
            "unit_test_agent".to_string(),
        ).unwrap();
        
        // Signal test start to swarm
        swarm_utils::coordinate_test_execution(
            &framework.context,
            "conformal_prediction_unit_tests",
        ).await.unwrap();
        
        // Execute unit tests
        let mut fixture = ConformalPredictionTestFixture::new();
        let result = fixture.predictor.predict(
            &fixture.mock_predictions,
            &fixture.mock_calibration_data,
        );
        
        assert!(result.is_ok(), "Swarm coordinated unit test should succeed");
        
        // Share results with swarm
        let test_metrics = framework.context.execution_metrics.clone();
        swarm_utils::share_test_results(&framework.context, &test_metrics).await.unwrap();
    }
}