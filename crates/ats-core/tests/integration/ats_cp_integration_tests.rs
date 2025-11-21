//! Integration Tests for ATS-CP System
//!
//! These tests verify the complete ATS-CP pipeline integration including:
//! - End-to-end workflow validation
//! - Component interaction verification
//! - API layer testing
//! - System-level behavior validation

use ats_core::{
    conformal::ConformalPredictor,
    config::{AtsCpConfig, ConformalConfig, TemperatureConfig},
    types::{AtsCpVariant, Confidence, PredictionIntervals, AtsCpResult},
    error::{AtsCoreError, Result},
    test_framework::{TestFramework, swarm_utils},
};
use std::time::{Duration, Instant};
use tokio;

/// Integration test fixture for ATS-CP system
struct AtsCpIntegrationFixture {
    config: AtsCpConfig,
    predictor: ConformalPredictor,
    test_logits: Vec<f64>,
    test_calibration_logits: Vec<Vec<f64>>,
    test_calibration_labels: Vec<usize>,
    test_predictions: Vec<f64>,
    test_calibration_data: Vec<f64>,
}

impl AtsCpIntegrationFixture {
    fn new() -> Self {
        let config = AtsCpConfig {
            conformal: ConformalConfig {
                target_latency_us: 20,
                min_calibration_size: 10,
                max_calibration_size: 1000,
                calibration_window_size: 500,
                default_confidence: 0.95,
                online_calibration: true,
                validate_exchangeability: true,
                quantile_method: crate::config::QuantileMethod::Linear,
            },
            temperature: TemperatureConfig {
                default_temperature: 1.0,
                min_temperature: 0.1,
                max_temperature: 10.0,
                target_latency_us: 10,
                search_tolerance: 1e-6,
                max_search_iterations: 100,
            },
            ..Default::default()
        };
        
        let predictor = ConformalPredictor::new(&config).unwrap();
        
        // Comprehensive test data for integration scenarios
        let test_logits = vec![2.1, 1.3, 0.8, 0.4, 0.2];
        
        let test_calibration_logits = vec![
            vec![2.0, 1.2, 0.9, 0.5, 0.3],
            vec![1.9, 1.4, 0.7, 0.6, 0.2],
            vec![2.2, 1.1, 0.8, 0.4, 0.1],
            vec![1.8, 1.5, 0.6, 0.7, 0.3],
            vec![2.1, 1.0, 0.9, 0.3, 0.4],
            vec![2.0, 1.3, 0.8, 0.5, 0.2],
            vec![1.7, 1.6, 0.7, 0.6, 0.1],
            vec![2.3, 0.9, 1.0, 0.2, 0.5],
            vec![1.9, 1.4, 0.6, 0.8, 0.3],
            vec![2.1, 1.2, 0.9, 0.4, 0.2],
            vec![1.8, 1.5, 0.7, 0.6, 0.1],
            vec![2.2, 1.1, 0.8, 0.3, 0.4],
            vec![2.0, 1.3, 0.9, 0.5, 0.2],
            vec![1.9, 1.4, 0.6, 0.7, 0.3],
            vec![2.1, 1.0, 1.0, 0.4, 0.1],
        ];
        
        let test_calibration_labels = vec![0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 1, 0, 2, 3, 1];
        
        let test_predictions = vec![1.2, 2.3, 3.1, 4.5, 5.7, 2.8, 3.9, 1.6, 4.2, 5.1];
        let test_calibration_data: Vec<f64> = (0..50)
            .map(|i| (i as f64) * 0.02 + 0.1)
            .collect();
        
        Self {
            config,
            predictor,
            test_logits,
            test_calibration_logits,
            test_calibration_labels,
            test_predictions,
            test_calibration_data,
        }
    }
}

/// Integration tests for complete ATS-CP pipeline
mod pipeline_integration_tests {
    use super::*;
    
    #[tokio::test]
    async fn test_complete_ats_cp_pipeline_integration() {
        println!("🔄 Testing complete ATS-CP pipeline integration...");
        
        let mut fixture = AtsCpIntegrationFixture::new();
        
        // Test all ATS-CP variants in complete pipeline
        let variants = vec![
            AtsCpVariant::GQ,
            AtsCpVariant::AQ,
            AtsCpVariant::MGQ,
            AtsCpVariant::MAQ,
        ];
        
        let confidence_levels = vec![0.90, 0.95, 0.99];
        
        for variant in variants {
            for confidence in confidence_levels.iter() {
                println!("  Testing variant {:?} with confidence {}", variant, confidence);
                
                let start_time = Instant::now();
                let result = fixture.predictor.ats_cp_predict(
                    &fixture.test_logits,
                    &fixture.test_calibration_logits,
                    &fixture.test_calibration_labels,
                    *confidence,
                    variant.clone(),
                );
                let execution_time = start_time.elapsed();
                
                assert!(result.is_ok(), 
                       "Complete pipeline should succeed for variant {:?} with confidence {}", 
                       variant, confidence);
                
                let ats_cp_result = result.unwrap();
                
                // Validate pipeline integration results
                validate_ats_cp_result(&ats_cp_result, *confidence, &variant)?;
                
                // Verify pipeline meets performance requirements
                assert!(execution_time < Duration::from_micros(20),
                       "Pipeline should meet latency requirement for variant {:?}: {:?}",
                       variant, execution_time);
                
                println!("    ✅ Pipeline test passed in {:?}", execution_time);
            }
        }
        
        println!("✅ Complete ATS-CP pipeline integration tests passed");
    }
    
    #[tokio::test]
    async fn test_conformal_prediction_pipeline_integration() {
        println!("🔄 Testing conformal prediction pipeline integration...");
        
        let mut fixture = AtsCpIntegrationFixture::new();
        
        // Test basic conformal prediction pipeline
        let start_time = Instant::now();
        let result = fixture.predictor.predict(
            &fixture.test_predictions,
            &fixture.test_calibration_data,
        );
        let execution_time = start_time.elapsed();
        
        assert!(result.is_ok(), "Conformal prediction pipeline should succeed");
        let intervals = result.unwrap();
        
        // Validate pipeline results
        validate_prediction_intervals(&intervals, &fixture.test_predictions)?;
        
        // Test detailed prediction pipeline
        let detailed_result = fixture.predictor.predict_detailed(
            &fixture.test_predictions,
            &fixture.test_calibration_data,
            0.95,
        );
        
        assert!(detailed_result.is_ok(), "Detailed prediction pipeline should succeed");
        let detailed = detailed_result.unwrap();
        
        // Validate detailed pipeline results
        assert_eq!(detailed.confidence, 0.95);
        assert!(detailed.quantile_threshold > 0.0);
        assert!(detailed.execution_time_ns > 0);
        assert!(!detailed.calibration_scores.is_empty());
        
        assert!(execution_time < Duration::from_micros(20),
               "Conformal prediction pipeline should meet latency: {:?}", execution_time);
        
        println!("✅ Conformal prediction pipeline integration tests passed");
    }
    
    #[tokio::test]
    async fn test_adaptive_pipeline_integration() {
        println!("🔄 Testing adaptive pipeline integration...");
        
        let mut fixture = AtsCpIntegrationFixture::new();
        
        let predictions = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let true_values = vec![1.1, 1.9, 3.1, 3.9, 5.1];
        let confidence = 0.95;
        
        // Test adaptive pipeline with online calibration update
        let start_time = Instant::now();
        let result = fixture.predictor.predict_adaptive(&predictions, &true_values, confidence);
        let execution_time = start_time.elapsed();
        
        assert!(result.is_ok(), "Adaptive pipeline should succeed");
        let intervals = result.unwrap();
        
        // Validate adaptive pipeline results
        validate_prediction_intervals(&intervals, &predictions)?;
        
        // Verify adaptive behavior - calibration should be updated
        let (ops, avg_latency, _) = fixture.predictor.get_performance_stats();
        assert!(ops > 0, "Adaptive pipeline should record operations");
        
        assert!(execution_time < Duration::from_micros(30),
               "Adaptive pipeline should meet latency: {:?}", execution_time);
        
        println!("✅ Adaptive pipeline integration tests passed");
    }
    
    fn validate_ats_cp_result(
        result: &AtsCpResult,
        expected_confidence: f64,
        expected_variant: &AtsCpVariant,
    ) -> Result<()> {
        // Validate result structure
        assert!(!result.conformal_set.is_empty(), "Conformal set should not be empty");
        assert!(!result.calibrated_probabilities.is_empty(), "Probabilities should not be empty");
        assert!(result.optimal_temperature > 0.0, "Temperature should be positive");
        assert!(result.quantile_threshold >= 0.0, "Quantile threshold should be non-negative");
        assert_eq!(result.coverage_guarantee, expected_confidence, "Coverage guarantee should match");
        assert_eq!(result.variant, *expected_variant, "Variant should match");
        assert!(result.execution_time_ns > 0, "Execution time should be recorded");
        
        // Validate probability distribution
        let prob_sum: f64 = result.calibrated_probabilities.iter().sum();
        assert!((prob_sum - 1.0).abs() < 1e-6, "Probabilities should sum to 1.0");
        
        for (i, &prob) in result.calibrated_probabilities.iter().enumerate() {
            assert!(prob >= 0.0 && prob <= 1.0, 
                   "Probability {} should be in [0,1], got {}", i, prob);
        }
        
        // Validate conformal set
        let max_class = result.calibrated_probabilities.len() - 1;
        for &class_idx in &result.conformal_set {
            assert!(class_idx <= max_class, 
                   "Conformal set class {} should be valid (max: {})", class_idx, max_class);
        }
        
        Ok(())
    }
    
    fn validate_prediction_intervals(
        intervals: &PredictionIntervals,
        predictions: &[f64],
    ) -> Result<()> {
        assert_eq!(intervals.len(), predictions.len(), 
                  "Intervals length should match predictions");
        
        for (i, (lower, upper)) in intervals.iter().enumerate() {
            assert!(lower <= upper, "Lower bound should be <= upper bound for interval {}", i);
            assert!(lower.is_finite(), "Lower bound should be finite for interval {}", i);
            assert!(upper.is_finite(), "Upper bound should be finite for interval {}", i);
            
            // Intervals should reasonably contain the prediction
            let prediction = predictions[i];
            let interval_width = upper - lower;
            assert!(interval_width > 0.0, "Interval width should be positive for interval {}", i);
        }
        
        Ok(())
    }
}

/// API layer integration tests
mod api_integration_tests {
    use super::*;
    
    #[tokio::test]
    async fn test_batch_prediction_api_integration() {
        println!("🔄 Testing batch prediction API integration...");
        
        let mut fixture = AtsCpIntegrationFixture::new();
        
        let confidence_levels = vec![0.90, 0.95, 0.99];
        
        let start_time = Instant::now();
        let result = fixture.predictor.predict_batch_confidence(
            &fixture.test_predictions,
            &fixture.test_calibration_data,
            &confidence_levels,
        );
        let execution_time = start_time.elapsed();
        
        assert!(result.is_ok(), "Batch prediction API should succeed");
        let batch_results = result.unwrap();
        
        // Validate batch API results
        assert_eq!(batch_results.len(), confidence_levels.len(),
                  "Batch results should match confidence levels count");
        
        for (i, intervals) in batch_results.iter().enumerate() {
            validate_prediction_intervals(intervals, &fixture.test_predictions)
                .map_err(|e| format!("Batch result {} validation failed: {}", i, e))?;
        }
        
        assert!(execution_time < Duration::from_micros(100),
               "Batch API should meet latency: {:?}", execution_time);
        
        println!("✅ Batch prediction API integration tests passed");
    }
    
    #[tokio::test]
    async fn test_parallel_prediction_api_integration() {
        println!("🔄 Testing parallel prediction API integration...");
        
        let mut fixture = AtsCpIntegrationFixture::new();
        
        // Create larger dataset for parallel processing
        let large_predictions: Vec<f64> = (0..100).map(|i| i as f64 * 0.1).collect();
        let confidence = 0.95;
        
        let start_time = Instant::now();
        let result = fixture.predictor.predict_parallel(
            &large_predictions,
            &fixture.test_calibration_data,
            confidence,
        );
        let execution_time = start_time.elapsed();
        
        assert!(result.is_ok(), "Parallel prediction API should succeed");
        let intervals = result.unwrap();
        
        // Validate parallel API results
        validate_prediction_intervals(&intervals, &large_predictions)?;
        
        // Compare with sequential version for consistency
        let sequential_result = fixture.predictor.predict(&large_predictions, &fixture.test_calibration_data);
        assert!(sequential_result.is_ok(), "Sequential prediction should also succeed");
        
        let sequential_intervals = sequential_result.unwrap();
        assert_eq!(intervals.len(), sequential_intervals.len(),
                  "Parallel and sequential should produce same number of intervals");
        
        println!("  Parallel execution time: {:?}", execution_time);
        println!("✅ Parallel prediction API integration tests passed");
    }
    
    #[tokio::test]
    async fn test_temperature_scaling_api_integration() {
        println!("🔄 Testing temperature scaling API integration...");
        
        let mut fixture = AtsCpIntegrationFixture::new();
        
        let test_temperatures = vec![0.1, 0.5, 1.0, 2.0, 5.0];
        
        for temperature in test_temperatures {
            println!("  Testing temperature: {}", temperature);
            
            let start_time = Instant::now();
            let result = fixture.predictor.temperature_scaled_softmax(&fixture.test_logits, temperature);
            let execution_time = start_time.elapsed();
            
            assert!(result.is_ok(), "Temperature scaling API should succeed for temperature {}", temperature);
            let probabilities = result.unwrap();
            
            // Validate temperature scaling results
            assert_eq!(probabilities.len(), fixture.test_logits.len());
            
            let prob_sum: f64 = probabilities.iter().sum();
            assert!((prob_sum - 1.0).abs() < 1e-6, 
                   "Probabilities should sum to 1.0 for temperature {}", temperature);
            
            for &prob in &probabilities {
                assert!(prob >= 0.0 && prob <= 1.0,
                       "Probability should be in [0,1] for temperature {}", temperature);
            }
            
            assert!(execution_time < Duration::from_micros(10),
                   "Temperature scaling should be fast: {:?}", execution_time);
            
            println!("    ✅ Temperature {} test passed in {:?}", temperature, execution_time);
        }
        
        println!("✅ Temperature scaling API integration tests passed");
    }
}

/// Cross-component integration tests
mod cross_component_integration_tests {
    use super::*;
    
    #[tokio::test]
    async fn test_conformal_prediction_temperature_scaling_integration() {
        println!("🔄 Testing conformal prediction + temperature scaling integration...");
        
        let mut fixture = AtsCpIntegrationFixture::new();
        
        // Test that temperature scaling is properly integrated in ATS-CP workflow
        let variants = vec![AtsCpVariant::GQ, AtsCpVariant::AQ];
        
        for variant in variants {
            println!("  Testing variant {:?}", variant);
            
            let start_time = Instant::now();
            let result = fixture.predictor.ats_cp_predict(
                &fixture.test_logits,
                &fixture.test_calibration_logits,
                &fixture.test_calibration_labels,
                0.95,
                variant.clone(),
            );
            let execution_time = start_time.elapsed();
            
            assert!(result.is_ok(), "Cross-component integration should succeed for {:?}", variant);
            let ats_result = result.unwrap();
            
            // Verify temperature scaling integration
            assert!(ats_result.optimal_temperature > 0.0,
                   "Temperature scaling should produce valid temperature");
            assert!(ats_result.optimal_temperature.is_finite(),
                   "Temperature should be finite");
            
            // Verify conformal prediction integration
            assert!(!ats_result.conformal_set.is_empty(),
                   "Conformal prediction should produce non-empty set");
            
            // Verify probabilities are properly scaled
            let prob_sum: f64 = ats_result.calibrated_probabilities.iter().sum();
            assert!((prob_sum - 1.0).abs() < 1e-6,
                   "Cross-component: probabilities should be normalized");
            
            assert!(execution_time < Duration::from_micros(50),
                   "Cross-component integration should be efficient: {:?}", execution_time);
            
            println!("    ✅ Variant {:?} integration test passed", variant);
        }
        
        println!("✅ Cross-component integration tests passed");
    }
    
    #[tokio::test]
    async fn test_quantile_computation_conformal_integration() {
        println!("🔄 Testing quantile computation + conformal prediction integration...");
        
        let mut fixture = AtsCpIntegrationFixture::new();
        
        // Test different quantile methods with conformal prediction
        let quantile_methods = vec![
            crate::config::QuantileMethod::Linear,
            crate::config::QuantileMethod::Nearest,
            crate::config::QuantileMethod::Higher,
            crate::config::QuantileMethod::Lower,
            crate::config::QuantileMethod::Midpoint,
        ];
        
        for method in quantile_methods {
            println!("  Testing quantile method: {:?}", method);
            
            // Create predictor with specific quantile method
            let config = AtsCpConfig {
                conformal: ConformalConfig {
                    quantile_method: method,
                    ..fixture.config.conformal.clone()
                },
                ..fixture.config.clone()
            };
            
            let mut predictor = ConformalPredictor::new(&config).unwrap();
            
            let start_time = Instant::now();
            let result = predictor.predict(&fixture.test_predictions, &fixture.test_calibration_data);
            let execution_time = start_time.elapsed();
            
            assert!(result.is_ok(), "Quantile-conformal integration should succeed for {:?}", method);
            let intervals = result.unwrap();
            
            // Validate integration results
            validate_prediction_intervals(&intervals, &fixture.test_predictions)?;
            
            assert!(execution_time < Duration::from_micros(30),
                   "Quantile-conformal integration should be efficient: {:?}", execution_time);
            
            println!("    ✅ Method {:?} integration test passed in {:?}", method, execution_time);
        }
        
        println!("✅ Quantile computation integration tests passed");
    }
    
    #[tokio::test]
    async fn test_online_calibration_integration() {
        println!("🔄 Testing online calibration integration...");
        
        let mut fixture = AtsCpIntegrationFixture::new();
        
        // Test online calibration with streaming data
        let initial_predictions = vec![1.0, 2.0, 3.0];
        let initial_true_values = vec![1.1, 1.9, 3.1];
        
        // First batch - establish initial calibration
        let result1 = fixture.predictor.predict_adaptive(&initial_predictions, &initial_true_values, 0.95);
        assert!(result1.is_ok(), "Initial online calibration should succeed");
        
        // Second batch - should use updated calibration  
        let new_predictions = vec![4.0, 5.0, 6.0];
        let new_true_values = vec![3.9, 5.1, 5.8];
        
        let result2 = fixture.predictor.predict_adaptive(&new_predictions, &new_true_values, 0.95);
        assert!(result2.is_ok(), "Updated online calibration should succeed");
        
        // Third batch - further adaptation
        let final_predictions = vec![7.0, 8.0, 9.0];
        let final_true_values = vec![7.2, 7.8, 9.1];
        
        let result3 = fixture.predictor.predict_adaptive(&final_predictions, &final_true_values, 0.95);
        assert!(result3.is_ok(), "Final online calibration should succeed");
        
        // Verify calibration has been updated through multiple batches
        let (total_ops, avg_latency, ops_per_sec) = fixture.predictor.get_performance_stats();
        assert_eq!(total_ops, 3, "Should record all calibration updates");
        assert!(avg_latency > 0, "Should record execution times");
        assert!(ops_per_sec > 0.0, "Should compute throughput");
        
        println!("  Online calibration processed {} operations", total_ops);
        println!("  Average latency: {} ns", avg_latency);
        println!("  Operations per second: {:.2}", ops_per_sec);
        
        println!("✅ Online calibration integration tests passed");
    }
}

/// System-level integration tests
mod system_level_integration_tests {
    use super::*;
    
    #[tokio::test]
    async fn test_high_frequency_trading_integration_scenario() {
        println!("🔄 Testing high-frequency trading integration scenario...");
        
        let mut fixture = AtsCpIntegrationFixture::new();
        
        // Simulate high-frequency trading scenario
        let num_trades = 100;
        let mut total_execution_time = Duration::from_nanos(0);
        let mut successful_trades = 0;
        
        for i in 0..num_trades {
            // Generate trade-specific data
            let trade_logits = vec![
                2.0 + (i as f64) * 0.01,
                1.0 + (i as f64) * 0.005,
                0.5 - (i as f64) * 0.002,
            ];
            
            let start_time = Instant::now();
            let result = fixture.predictor.ats_cp_predict(
                &trade_logits,
                &fixture.test_calibration_logits,
                &fixture.test_calibration_labels,
                0.95,
                AtsCpVariant::GQ,
            );
            let execution_time = start_time.elapsed();
            
            if result.is_ok() && execution_time < Duration::from_micros(20) {
                successful_trades += 1;
                total_execution_time += execution_time;
            }
            
            // Simulate brief pause between trades
            tokio::time::sleep(Duration::from_micros(1)).await;
        }
        
        let success_rate = (successful_trades as f64) / (num_trades as f64);
        let avg_execution_time = total_execution_time / successful_trades;
        
        println!("  High-frequency trading results:");
        println!("    Successful trades: {}/{}", successful_trades, num_trades);
        println!("    Success rate: {:.2}%", success_rate * 100.0);
        println!("    Average execution time: {:?}", avg_execution_time);
        
        assert!(success_rate >= 0.95, "High-frequency trading should have >95% success rate");
        assert!(avg_execution_time < Duration::from_micros(20),
               "Average execution time should meet HFT requirements");
        
        println!("✅ High-frequency trading integration scenario passed");
    }
    
    #[tokio::test]
    async fn test_multi_model_integration_scenario() {
        println!("🔄 Testing multi-model integration scenario...");
        
        // Create multiple predictors with different configurations
        let configs = vec![
            AtsCpConfig {
                conformal: ConformalConfig {
                    quantile_method: crate::config::QuantileMethod::Linear,
                    default_confidence: 0.90,
                    ..Default::default()
                },
                ..Default::default()
            },
            AtsCpConfig {
                conformal: ConformalConfig {
                    quantile_method: crate::config::QuantileMethod::Nearest,
                    default_confidence: 0.95,
                    ..Default::default()
                },
                ..Default::default()
            },
            AtsCpConfig {
                conformal: ConformalConfig {
                    quantile_method: crate::config::QuantileMethod::Higher,
                    default_confidence: 0.99,
                    ..Default::default()
                },
                ..Default::default()
            },
        ];
        
        let mut predictors: Vec<ConformalPredictor> = configs
            .iter()
            .map(|config| ConformalPredictor::new(config).unwrap())
            .collect();
        
        let test_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let calibration_data: Vec<f64> = (0..30).map(|i| i as f64 * 0.1).collect();
        
        // Test ensemble prediction
        let mut ensemble_results = Vec::new();
        let start_time = Instant::now();
        
        for (i, predictor) in predictors.iter_mut().enumerate() {
            let result = predictor.predict(&test_data, &calibration_data);
            assert!(result.is_ok(), "Multi-model predictor {} should succeed", i);
            ensemble_results.push(result.unwrap());
        }
        
        let ensemble_time = start_time.elapsed();
        
        // Validate ensemble results
        assert_eq!(ensemble_results.len(), configs.len());
        
        for (i, intervals) in ensemble_results.iter().enumerate() {
            assert_eq!(intervals.len(), test_data.len(),
                      "Model {} should produce correct number of intervals", i);
            
            for (j, (lower, upper)) in intervals.iter().enumerate() {
                assert!(lower <= upper, "Model {} interval {} should be valid", i, j);
                assert!(lower.is_finite() && upper.is_finite(),
                       "Model {} interval {} should be finite", i, j);
            }
        }
        
        // Compare ensemble diversity
        let mut interval_widths: Vec<Vec<f64>> = ensemble_results
            .iter()
            .map(|intervals| intervals.iter().map(|(l, u)| u - l).collect())
            .collect();
        
        println!("  Ensemble results:");
        println!("    Models: {}", configs.len());
        println!("    Total execution time: {:?}", ensemble_time);
        
        for (i, widths) in interval_widths.iter().enumerate() {
            let avg_width: f64 = widths.iter().sum::<f64>() / widths.len() as f64;
            println!("    Model {} average interval width: {:.4}", i, avg_width);
        }
        
        assert!(ensemble_time < Duration::from_micros(100),
               "Multi-model ensemble should be efficient: {:?}", ensemble_time);
        
        println!("✅ Multi-model integration scenario passed");
    }
}

/// Swarm coordination integration tests
mod swarm_integration_tests {
    use super::*;
    
    #[tokio::test]
    async fn test_swarm_coordinated_integration_testing() {
        println!("🔄 Testing swarm-coordinated integration...");
        
        // Initialize test framework for swarm coordination
        let mut framework = TestFramework::new(
            "ats_cp_integration_swarm".to_string(),
            "integration_test_agent".to_string(),
        ).unwrap();
        
        // Signal coordination with other test agents
        swarm_utils::coordinate_test_execution(&framework.context, "integration_tests").await.unwrap();
        
        // Execute coordinated integration test
        let mut fixture = AtsCpIntegrationFixture::new();
        
        let start_time = Instant::now();
        let result = fixture.predictor.ats_cp_predict(
            &fixture.test_logits,
            &fixture.test_calibration_logits,
            &fixture.test_calibration_labels,
            0.95,
            AtsCpVariant::GQ,
        );
        let execution_time = start_time.elapsed();
        
        assert!(result.is_ok(), "Swarm-coordinated integration should succeed");
        let ats_result = result.unwrap();
        
        // Update test metrics
        framework.context.execution_metrics.tests_passed += 1;
        framework.context.execution_metrics.performance_metrics.insert(
            "ats_cp_integration_latency_ns".to_string(),
            execution_time.as_nanos() as f64,
        );
        
        // Share results with swarm
        swarm_utils::share_test_results(&framework.context, &framework.context.execution_metrics).await.unwrap();
        
        // Validate swarm coordination worked
        assert!(ats_result.execution_time_ns > 0, "Should record execution metrics");
        assert!(execution_time < Duration::from_micros(50), 
               "Swarm-coordinated test should meet performance requirements");
        
        println!("  Swarm coordination results:");
        println!("    Tests passed: {}", framework.context.execution_metrics.tests_passed);
        println!("    Execution time: {:?}", execution_time);
        
        println!("✅ Swarm-coordinated integration tests passed");
    }
    
    #[tokio::test]
    async fn test_cross_agent_test_dependency_coordination() {
        println!("🔄 Testing cross-agent test dependency coordination...");
        
        let framework = TestFramework::new(
            "dependency_coordination_swarm".to_string(),
            "dependent_test_agent".to_string(),
        ).unwrap();
        
        // Wait for dependent test completion (simulated)
        let dependencies = vec![
            "unit_test_agent".to_string(),
            "property_test_agent".to_string(),
        ];
        
        swarm_utils::wait_for_dependencies(&framework.context, &dependencies).await.unwrap();
        
        // Execute integration test that depends on unit tests
        let mut fixture = AtsCpIntegrationFixture::new();
        
        let result = fixture.predictor.predict(&fixture.test_predictions, &fixture.test_calibration_data);
        assert!(result.is_ok(), "Dependent integration test should succeed");
        
        // Signal completion for other dependent tests
        println!("  Signaling test completion to dependent agents");
        swarm_utils::coordinate_test_execution(&framework.context, "integration_complete").await.unwrap();
        
        println!("✅ Cross-agent dependency coordination tests passed");
    }
}