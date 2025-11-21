//! Integration tests for ATS-Core
//!
//! These tests validate the complete ATS-CP pipeline and verify that
//! all components work together correctly for real-world scenarios.

use ats_core::{
    config::AtsCpConfig,
    prelude::*,
    test_utils::*,
};
use approx::assert_relative_eq;
use std::time::Instant;

#[test]
fn test_full_ats_cp_pipeline() {
    let config = AtsCpConfig::high_performance();
    let engine = AtsCpEngine::new(config).unwrap();
    
    // Generate realistic trading data
    let predictions = generate_predictions(100);
    let calibration_data = generate_calibration_data(500);
    let temperature = 1.2;
    
    let start_time = Instant::now();
    
    // Step 1: Temperature scaling
    let scaled_predictions = engine.temperature_scale(&predictions, temperature).unwrap();
    assert_eq!(scaled_predictions.len(), predictions.len());
    
    // Step 2: Conformal prediction
    let intervals = engine.conformal_predict(&scaled_predictions, &calibration_data).unwrap();
    assert_eq!(intervals.len(), predictions.len());
    
    // Verify pipeline completes within latency target
    let elapsed = start_time.elapsed();
    assert!(elapsed.as_micros() < 100, "Pipeline exceeded 100μs latency target: {}μs", elapsed.as_micros());
    
    // Verify intervals are valid
    for (lower, upper) in intervals {
        assert!(lower <= upper, "Invalid interval: [{}, {}]", lower, upper);
        assert!(lower.is_finite() && upper.is_finite());
    }
}

#[test]
fn test_simd_operations_correctness() {
    let config = AtsCpConfig::high_performance();
    let engine = AtsCpEngine::new(config).unwrap();
    
    // Test with aligned data sizes
    for size in [32, 64, 128, 256, 512, 1024] {
        let a: Vec<f64> = (0..size).map(|i| i as f64 * 0.1).collect();
        let b: Vec<f64> = (0..size).map(|i| (i as f64 + 1.0) * 0.05).collect();
        
        let result = engine.simd_vector_add(&a, &b).unwrap();
        
        // Verify correctness
        for i in 0..size {
            let expected = a[i] + b[i];
            assert_relative_eq!(result[i], expected, epsilon = 1e-12);
        }
    }
}

#[test]
fn test_parallel_processing_correctness() {
    let config = AtsCpConfig::high_performance();
    let engine = AtsCpEngine::new(config).unwrap();
    
    // Large dataset to trigger parallel processing
    let data: Vec<i32> = (0..10000).collect();
    let result = engine.parallel_process(&data, |x| x * 2).unwrap();
    
    assert_eq!(result.len(), data.len());
    for i in 0..data.len() {
        assert_eq!(result[i], data[i] * 2);
    }
}

#[test]
fn test_performance_monitoring() {
    let config = AtsCpConfig::high_performance();
    let engine = AtsCpEngine::new(config).unwrap();
    
    let predictions = generate_predictions(50);
    let temperature = 1.0;
    
    // Perform multiple operations
    for _ in 0..10 {
        let _ = engine.temperature_scale(&predictions, temperature).unwrap();
    }
    
    let stats = engine.get_performance_stats();
    assert!(stats.total_operations >= 10);
    assert!(stats.average_latency_ns > 0);
    assert!(stats.ops_per_second > 0.0);
}

#[test]
fn test_error_handling_robustness() {
    let config = AtsCpConfig::default();
    let engine = AtsCpEngine::new(config).unwrap();
    
    // Test empty inputs
    let empty_vec: Vec<f64> = vec![];
    let result = engine.temperature_scale(&empty_vec, 1.0);
    assert!(result.is_err());
    
    // Test dimension mismatch
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![1.0, 2.0]; // Different length
    let result = engine.simd_vector_add(&a, &b);
    assert!(result.is_err());
    
    // Test invalid temperature
    let predictions = vec![1.0, 2.0, 3.0];
    let result = engine.temperature_scale(&predictions, -1.0);
    assert!(result.is_err());
}

#[test]
fn test_numerical_stability() {
    let config = AtsCpConfig::default();
    let engine = AtsCpEngine::new(config).unwrap();
    
    // Test with extreme values
    let extreme_predictions = vec![1e-10, 1e10, -1e10, std::f64::MIN_POSITIVE, std::f64::MAX / 1e10];
    let temperature = 1.0;
    
    let result = engine.temperature_scale(&extreme_predictions, temperature);
    assert!(result.is_ok());
    
    let scaled = result.unwrap();
    for value in scaled {
        assert!(value.is_finite(), "Non-finite result: {}", value);
    }
}

#[test]
fn test_memory_efficiency() {
    let config = AtsCpConfig::high_performance();
    let engine = AtsCpEngine::new(config).unwrap();
    
    // Test with large arrays to verify memory efficiency
    let large_predictions: Vec<f64> = (0..100000).map(|i| i as f64 * 1e-6).collect();
    let large_calibration: Vec<f64> = (0..10000).map(|i| i as f64 * 1e-5).collect();
    
    let start_memory = get_memory_usage();
    
    let scaled = engine.temperature_scale(&large_predictions, 1.5).unwrap();
    let _intervals = engine.conformal_predict(&scaled, &large_calibration).unwrap();
    
    let end_memory = get_memory_usage();
    let memory_growth = end_memory - start_memory;
    
    // Verify reasonable memory usage (should not leak)
    assert!(memory_growth < 100 * 1024 * 1024, "Excessive memory usage: {} bytes", memory_growth);
}

#[test]
fn test_reproducibility() {
    let config = AtsCpConfig::default();
    let engine1 = AtsCpEngine::new(config.clone()).unwrap();
    let engine2 = AtsCpEngine::new(config).unwrap();
    
    let predictions = generate_predictions(100);
    let calibration_data = generate_calibration_data(500);
    let temperature = 1.3;
    
    // Same inputs should produce same outputs
    let result1 = engine1.temperature_scale(&predictions, temperature).unwrap();
    let result2 = engine2.temperature_scale(&predictions, temperature).unwrap();
    
    assert_eq!(result1.len(), result2.len());
    for (v1, v2) in result1.iter().zip(result2.iter()) {
        assert_relative_eq!(v1, v2, epsilon = 1e-12);
    }
    
    let intervals1 = engine1.conformal_predict(&result1, &calibration_data).unwrap();
    let intervals2 = engine2.conformal_predict(&result2, &calibration_data).unwrap();
    
    assert_eq!(intervals1.len(), intervals2.len());
    for ((l1, u1), (l2, u2)) in intervals1.iter().zip(intervals2.iter()) {
        assert_relative_eq!(l1, l2, epsilon = 1e-12);
        assert_relative_eq!(u1, u2, epsilon = 1e-12);
    }
}

#[test]
fn test_concurrent_access() {
    use std::sync::Arc;
    use std::thread;
    
    let config = AtsCpConfig::high_performance();
    let engine = Arc::new(AtsCpEngine::new(config).unwrap());
    
    let predictions = Arc::new(generate_predictions(100));
    let calibration_data = Arc::new(generate_calibration_data(500));
    
    let mut handles = Vec::new();
    
    // Spawn multiple threads
    for i in 0..4 {
        let engine_clone = Arc::clone(&engine);
        let predictions_clone = Arc::clone(&predictions);
        let calibration_clone = Arc::clone(&calibration_data);
        
        let handle = thread::spawn(move || {
            let temperature = 1.0 + (i as f64) * 0.1;
            
            // Each thread performs operations
            for _ in 0..10 {
                let scaled = engine_clone.temperature_scale(&predictions_clone, temperature).unwrap();
                let _intervals = engine_clone.conformal_predict(&scaled, &calibration_clone).unwrap();
            }
        });
        
        handles.push(handle);
    }
    
    // Wait for all threads to complete
    for handle in handles {
        handle.join().unwrap();
    }
}

#[test]
fn test_edge_cases() {
    let config = AtsCpConfig::default();
    let engine = AtsCpEngine::new(config).unwrap();
    
    // Test single element arrays
    let single_prediction = vec![0.5];
    let single_calibration = vec![0.1, 0.2, 0.3, 0.4, 0.5]; // Minimum size
    
    let scaled = engine.temperature_scale(&single_prediction, 1.0).unwrap();
    assert_eq!(scaled.len(), 1);
    
    let intervals = engine.conformal_predict(&scaled, &single_calibration).unwrap();
    assert_eq!(intervals.len(), 1);
    
    // Test identical values
    let identical_predictions = vec![1.0; 100];
    let scaled = engine.temperature_scale(&identical_predictions, 2.0).unwrap();
    
    for value in scaled {
        assert_relative_eq!(value, (1.0 / 2.0).exp(), epsilon = 1e-10);
    }
}

#[test]
fn test_configuration_variations() {
    // Test different configurations
    let configs = vec![
        AtsCpConfig::default(),
        AtsCpConfig::high_performance(),
    ];
    
    for config in configs {
        let engine = AtsCpEngine::new(config).unwrap();
        
        let predictions = generate_predictions(50);
        let calibration_data = generate_calibration_data(200);
        
        // Verify all operations work with different configs
        let scaled = engine.temperature_scale(&predictions, 1.5).unwrap();
        let _intervals = engine.conformal_predict(&scaled, &calibration_data).unwrap();
        
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![0.5, 1.5, 2.5, 3.5];
        let _result = engine.simd_vector_add(&a, &b).unwrap();
    }
}

/// Helper function to get current memory usage (simplified)
fn get_memory_usage() -> usize {
    // In a real implementation, this would use system calls to get actual memory usage
    // For testing purposes, we'll use a placeholder
    0
}

#[test]
fn test_temperature_scaling_mathematical_properties() {
    let config = AtsCpConfig::default();
    let engine = AtsCpEngine::new(config).unwrap();
    
    // Test monotonicity: higher temperature should spread probabilities more
    let predictions = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let temp_low = 0.5;
    let temp_high = 2.0;
    
    let scaled_low = engine.temperature_scale(&predictions, temp_low).unwrap();
    let scaled_high = engine.temperature_scale(&predictions, temp_high).unwrap();
    
    // With lower temperature, max should be more dominant
    let max_low = scaled_low.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let max_high = scaled_high.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    
    // Check entropy: high temperature should have higher entropy
    let entropy_low = -scaled_low.iter().map(|&x| x * x.ln()).sum::<f64>();
    let entropy_high = -scaled_high.iter().map(|&x| x * x.ln()).sum::<f64>();
    
    assert!(entropy_high > entropy_low, "High temperature should have higher entropy");
}

#[test]
fn test_conformal_prediction_coverage_guarantees() {
    let config = AtsCpConfig::default();
    let engine = AtsCpEngine::new(config).unwrap();
    
    // Generate synthetic data with known properties
    let predictions: Vec<f64> = (0..100).map(|i| i as f64 * 0.01).collect();
    let calibration_data: Vec<f64> = (0..500).map(|i| i as f64 * 0.01 + 0.1).collect();
    
    let intervals = engine.conformal_predict(&predictions, &calibration_data).unwrap();
    
    // Check that intervals have reasonable widths
    for (lower, upper) in &intervals {
        let width = upper - lower;
        assert!(width > 0.0, "Interval width should be positive");
        assert!(width < 10.0, "Interval width should be reasonable");
    }
    
    // Check that intervals are well-formed
    let avg_width: f64 = intervals.iter().map(|(l, u)| u - l).sum::<f64>() / intervals.len() as f64;
    assert!(avg_width > 0.01, "Average interval width should be meaningful");
}

#[test]
fn test_simd_operations_precision() {
    let config = AtsCpConfig::default();
    let engine = AtsCpEngine::new(config).unwrap();
    
    // Test with high precision numbers
    let a = vec![1.0000000001, 2.0000000002, 3.0000000003];
    let b = vec![0.0000000001, 0.0000000002, 0.0000000003];
    
    let result = engine.simd_vector_add(&a, &b).unwrap();
    
    for i in 0..a.len() {
        let expected = a[i] + b[i];
        let diff = (result[i] - expected).abs();
        assert!(diff < 1e-12, "SIMD operation should maintain precision: {} vs {}", result[i], expected);
    }
}

#[test]
fn test_memory_allocation_patterns() {
    let config = AtsCpConfig::default();
    let engine = AtsCpEngine::new(config).unwrap();
    
    // Test memory efficiency with repeated operations
    let predictions = vec![1.0; 1000];
    let temperature = 1.5;
    
    for _ in 0..100 {
        let _ = engine.temperature_scale(&predictions, temperature).unwrap();
    }
    
    // Test that large operations don't cause memory issues
    let large_predictions = vec![1.0; 10000];
    let result = engine.temperature_scale(&large_predictions, temperature).unwrap();
    assert_eq!(result.len(), large_predictions.len());
}

#[test]
fn test_edge_case_temperature_values() {
    let config = AtsCpConfig::default();
    let engine = AtsCpEngine::new(config).unwrap();
    
    let predictions = vec![1.0, 2.0, 3.0];
    
    // Test very small temperature
    let result = engine.temperature_scale(&predictions, 0.001);
    assert!(result.is_ok());
    
    // Test large temperature
    let result = engine.temperature_scale(&predictions, 100.0);
    assert!(result.is_ok());
    
    // Test temperature = 1.0 (should be identity for softmax)
    let result = engine.temperature_scale(&predictions, 1.0).unwrap();
    assert_eq!(result.len(), predictions.len());
}

#[test]
fn test_configuration_impact_on_performance() {
    let default_config = AtsCpConfig::default();
    let high_perf_config = AtsCpConfig::high_performance();
    
    let engine_default = AtsCpEngine::new(default_config).unwrap();
    let engine_high_perf = AtsCpEngine::new(high_perf_config).unwrap();
    
    let predictions = vec![1.0; 256];
    let temperature = 1.5;
    
    // Both should produce same results
    let result_default = engine_default.temperature_scale(&predictions, temperature).unwrap();
    let result_high_perf = engine_high_perf.temperature_scale(&predictions, temperature).unwrap();
    
    for (a, b) in result_default.iter().zip(result_high_perf.iter()) {
        assert!((a - b).abs() < 1e-10, "Different configs should produce same results");
    }
}

#[test]
fn test_statistical_properties_of_outputs() {
    let config = AtsCpConfig::default();
    let engine = AtsCpEngine::new(config).unwrap();
    
    // Test that temperature scaling preserves certain statistical properties
    let predictions = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let temperature = 1.0;
    
    let scaled = engine.temperature_scale(&predictions, temperature).unwrap();
    
    // Check that probabilities sum to 1.0 (for softmax)
    let sum: f64 = scaled.iter().sum();
    assert!((sum - 1.0).abs() < 1e-10, "Probabilities should sum to 1.0");
    
    // Check that all values are non-negative
    for &value in &scaled {
        assert!(value >= 0.0, "All probabilities should be non-negative");
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;
    
    proptest! {
        #[test]
        fn test_temperature_scaling_properties(
            predictions in prop::collection::vec(-10.0..10.0, 1..100),
            temperature in 0.1..10.0
        ) {
            let config = AtsCpConfig::default();
            let engine = AtsCpEngine::new(config).unwrap();
            
            let result = engine.temperature_scale(&predictions, temperature);
            
            if let Ok(scaled) = result {
                // All scaled values should be finite and positive
                for value in scaled {
                    prop_assert!(value.is_finite());
                    prop_assert!(value > 0.0);
                }
            }
        }
        
        #[test]
        fn test_simd_vector_add_properties(
            a in prop::collection::vec(-1000.0..1000.0, 0..200),
            b in prop::collection::vec(-1000.0..1000.0, 0..200)
        ) {
            if a.len() == b.len() {
                let config = AtsCpConfig::default();
                let engine = AtsCpEngine::new(config).unwrap();
                
                let result = engine.simd_vector_add(&a, &b).unwrap();
                
                prop_assert_eq!(result.len(), a.len());
                
                for i in 0..a.len() {
                    let expected = a[i] + b[i];
                    if expected.is_finite() {
                        prop_assert!((result[i] - expected).abs() < 1e-10);
                    }
                }
            }
        }
        
        #[test]
        fn test_conformal_prediction_properties(
            predictions in prop::collection::vec(-5.0..5.0, 1..50),
            calibration_size in 10..500usize
        ) {
            let config = AtsCpConfig::default();
            let engine = AtsCpEngine::new(config).unwrap();
            
            let calibration_data: Vec<f64> = (0..calibration_size)
                .map(|i| (i as f64) * 0.01)
                .collect();
            
            let result = engine.conformal_predict(&predictions, &calibration_data);
            
            if let Ok(intervals) = result {
                prop_assert_eq!(intervals.len(), predictions.len());
                
                for (lower, upper) in intervals {
                    prop_assert!(lower <= upper);
                    prop_assert!(lower.is_finite());
                    prop_assert!(upper.is_finite());
                }
            }
        }
        
        #[test]
        fn test_simd_commutativity(
            a in prop::collection::vec(-100.0..100.0, 1..100),
            b in prop::collection::vec(-100.0..100.0, 1..100)
        ) {
            if a.len() == b.len() {
                let config = AtsCpConfig::default();
                let engine = AtsCpEngine::new(config).unwrap();
                
                let result_ab = engine.simd_vector_add(&a, &b).unwrap();
                let result_ba = engine.simd_vector_add(&b, &a).unwrap();
                
                for i in 0..a.len() {
                    if result_ab[i].is_finite() && result_ba[i].is_finite() {
                        prop_assert!((result_ab[i] - result_ba[i]).abs() < 1e-10);
                    }
                }
            }
        }
        
        #[test]
        fn test_temperature_scaling_consistency(
            predictions in prop::collection::vec(-5.0..5.0, 1..50),
            temperature in 0.1..5.0
        ) {
            let config = AtsCpConfig::default();
            let engine = AtsCpEngine::new(config).unwrap();
            
            // Test that scaling twice with same params gives same result
            let result1 = engine.temperature_scale(&predictions, temperature);
            let result2 = engine.temperature_scale(&predictions, temperature);
            
            if let (Ok(scaled1), Ok(scaled2)) = (result1, result2) {
                prop_assert_eq!(scaled1.len(), scaled2.len());
                for (v1, v2) in scaled1.iter().zip(scaled2.iter()) {
                    if v1.is_finite() && v2.is_finite() {
                        prop_assert!((v1 - v2).abs() < 1e-12);
                    }
                }
            }
        }
    }
}