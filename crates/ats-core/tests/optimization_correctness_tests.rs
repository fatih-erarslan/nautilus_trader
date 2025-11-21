//! Mathematical correctness validation tests for optimized implementations
//!
//! This test suite ensures that all performance optimizations maintain
//! mathematical correctness and numerical stability.

use ats_core::{
    config::AtsCpConfig,
    conformal::ConformalPredictor,
    conformal_optimized::{OptimizedConformalPredictor, GreenwaldKhannaQuantile},
    types::AtsCpVariant,
    memory_optimized::{CacheAlignedVec, ConformalDataLayout},
};
use approx::{assert_relative_eq, assert_abs_diff_eq};
use proptest::prelude::*;

/// Creates test configuration
fn create_test_config() -> AtsCpConfig {
    AtsCpConfig::default()
}

/// Test Greenwald-Khanna quantile algorithm correctness
#[test]
fn test_greenwald_khanna_correctness() {
    let mut gk = GreenwaldKhannaQuantile::new(0.95, 0.01);
    
    // Insert sorted data
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    for value in &data {
        gk.insert(*value);
    }
    
    let quantile = gk.query().unwrap();
    
    // For sorted data 1-10, 95th percentile should be close to 9.5
    // With epsilon=0.01, we expect reasonable accuracy
    assert!(quantile >= 9.0 && quantile <= 10.0, "Quantile {} not in expected range", quantile);
    
    // Test with larger dataset
    let mut gk_large = GreenwaldKhannaQuantile::new(0.90, 0.01);
    let large_data: Vec<f64> = (0..10000).map(|i| i as f64).collect();
    for value in &large_data {
        gk_large.insert(*value);
    }
    
    let large_quantile = gk_large.query().unwrap();
    let expected = 0.90 * 9999.0; // 90th percentile of 0-9999
    let error = (large_quantile - expected).abs() / expected;
    
    // Error should be within reasonable bounds
    assert!(error < 0.05, "Large dataset quantile error {} too high", error);
}

/// Test that optimized quantile matches exact computation within tolerance
#[test]
fn test_optimized_vs_exact_quantile() {
    let config = create_test_config();
    let mut optimized = OptimizedConformalPredictor::new(&config).unwrap();
    
    // Generate test data with known distribution
    let test_sizes = vec![100, 500, 1000, 2000];
    let confidences = vec![0.90, 0.95, 0.99];
    
    for size in test_sizes {
        for confidence in &confidences {
            // Generate uniform random data
            let mut data: Vec<f64> = (0..size).map(|i| i as f64 / size as f64).collect();
            data.sort_by(|a, b| a.partial_cmp(b).unwrap());
            
            // Compute exact quantile
            let exact_idx = ((confidence * size as f64).ceil() as usize).min(size - 1);
            let exact_quantile = data[exact_idx];
            
            // Compute GK approximation
            let gk_quantile = optimized.compute_quantile_gk(&data, *confidence).unwrap();
            
            // Check relative error is within bounds
            let relative_error = (gk_quantile - exact_quantile).abs() / exact_quantile.max(1e-10);
            assert!(relative_error < 0.1, 
                "Quantile error {} too high for size {} confidence {}", 
                relative_error, size, confidence);
        }
    }
}

/// Test softmax correctness across different implementations
#[test]
fn test_softmax_correctness_comparison() {
    let config = create_test_config();
    let original = ConformalPredictor::new(&config).unwrap();
    let mut optimized = OptimizedConformalPredictor::new(&config).unwrap();
    
    let test_cases = vec![
        vec![1.0, 2.0, 3.0],
        vec![0.0, 0.0, 0.0], // Edge case: all zeros
        vec![-5.0, 0.0, 5.0], // Large range
        vec![1e-8, 2e-8, 3e-8], // Very small values
        (0..64).map(|i| i as f64 * 0.1).collect(), // Large vector for SIMD
        vec![100.0, 101.0, 102.0], // Large values (numerical stability test)
    ];
    
    for logits in test_cases {
        let original_result = original.compute_softmax(&logits).unwrap();
        let optimized_result = optimized.softmax_avx512_optimized(&logits).unwrap();
        
        // Check same length
        assert_eq!(original_result.len(), optimized_result.len());
        
        // Check probabilities sum to 1
        let orig_sum: f64 = original_result.iter().sum();
        let opt_sum: f64 = optimized_result.iter().sum();
        assert_relative_eq!(orig_sum, 1.0, epsilon = 1e-10);
        assert_relative_eq!(opt_sum, 1.0, epsilon = 1e-10);
        
        // Check element-wise accuracy
        for (i, (&orig, &opt)) in original_result.iter().zip(&optimized_result).enumerate() {
            assert_relative_eq!(orig, opt, epsilon = 1e-12, 
                "Softmax mismatch at index {}: orig={}, opt={}", i, orig, opt);
        }
    }
}

/// Test end-to-end conformal prediction correctness
#[test]
fn test_conformal_prediction_correctness() {
    let config = create_test_config();
    let mut original = ConformalPredictor::new(&config).unwrap();
    let mut optimized = OptimizedConformalPredictor::new(&config).unwrap();
    
    // Generate test data
    let predictions = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let calibration_data: Vec<f64> = (0..500).map(|i| i as f64 * 0.002 + 0.5).collect();
    let confidence = 0.95;
    
    // Get results from both implementations
    let original_intervals = original.predict(&predictions, &calibration_data).unwrap();
    let optimized_intervals = optimized.predict_optimized(&predictions, &calibration_data, confidence).unwrap();
    
    assert_eq!(original_intervals.len(), optimized_intervals.len());
    
    // Since GK is an approximation, we allow for some tolerance in intervals
    for (i, ((orig_lower, orig_upper), (opt_lower, opt_upper))) in 
        original_intervals.iter().zip(&optimized_intervals).enumerate() {
        
        // Intervals should be reasonably close
        let lower_error = (orig_lower - opt_lower).abs() / orig_lower.abs().max(1e-6);
        let upper_error = (orig_upper - opt_upper).abs() / orig_upper.abs().max(1e-6);
        
        assert!(lower_error < 0.2, 
            "Lower bound error {} too high at index {}", lower_error, i);
        assert!(upper_error < 0.2,
            "Upper bound error {} too high at index {}", upper_error, i);
        
        // Both intervals should be valid
        assert!(orig_lower <= orig_upper, "Original interval invalid at {}", i);
        assert!(opt_lower <= opt_upper, "Optimized interval invalid at {}", i);
    }
}

/// Test ATS-CP algorithm correctness across variants
#[test] 
fn test_ats_cp_algorithm_correctness() {
    let config = create_test_config();
    let mut original = ConformalPredictor::new(&config).unwrap();
    let mut optimized = OptimizedConformalPredictor::new(&config).unwrap();
    
    // Test data
    let logits = vec![2.0, 1.0, 0.5, 1.5, 0.8];
    let calibration_logits: Vec<Vec<f64>> = (0..100).map(|i| {
        vec![
            (i as f64 * 0.02).sin() + 1.0,
            (i as f64 * 0.03).cos() + 0.5, 
            (i as f64 * 0.01) % 2.0,
            (i as f64 * 0.04).sin() * 0.5 + 1.2,
            (i as f64 * 0.02).cos() * 0.3 + 0.9
        ]
    }).collect();
    let calibration_labels: Vec<usize> = (0..100).map(|i| i % 5).collect();
    let confidence = 0.95;
    
    // Test each variant
    for variant in [AtsCpVariant::GQ, AtsCpVariant::AQ, AtsCpVariant::MGQ].iter() {
        let original_result = original.ats_cp_predict(
            &logits, 
            &calibration_logits,
            &calibration_labels,
            confidence,
            variant.clone()
        ).unwrap();
        
        let optimized_result = optimized.ats_cp_predict_optimized(
            &logits,
            &calibration_logits, 
            &calibration_labels,
            confidence,
            variant.clone()
        ).unwrap();
        
        // Check conformal sets have reasonable overlap
        let orig_set = &original_result.conformal_set;
        let opt_set = &optimized_result.conformal_set;
        
        // Due to quantile approximation, sets may differ slightly
        // but should have significant overlap for correct implementation
        let intersection: std::collections::HashSet<_> = 
            orig_set.iter().filter(|x| opt_set.contains(x)).collect();
        let union_size = orig_set.len() + opt_set.len() - intersection.len();
        let jaccard = intersection.len() as f64 / union_size as f64;
        
        // Jaccard similarity should be reasonably high
        assert!(jaccard > 0.5, 
            "Conformal sets too different for {:?}: jaccard={}", variant, jaccard);
        
        // Calibrated probabilities should be close
        assert_eq!(original_result.calibrated_probabilities.len(), 
                   optimized_result.calibrated_probabilities.len());
        
        for (i, (&orig_prob, &opt_prob)) in 
            original_result.calibrated_probabilities.iter()
            .zip(&optimized_result.calibrated_probabilities).enumerate() {
            
            assert_relative_eq!(orig_prob, opt_prob, epsilon = 1e-10,
                "Calibrated probability mismatch at {} for {:?}", i, variant);
        }
        
        // Coverage guarantee should match
        assert_eq!(original_result.coverage_guarantee, optimized_result.coverage_guarantee);
        
        // Variant should match
        assert_eq!(original_result.variant, optimized_result.variant);
    }
}

/// Test numerical stability with edge cases
#[test]
fn test_numerical_stability() {
    let config = create_test_config();
    let mut optimized = OptimizedConformalPredictor::new(&config).unwrap();
    
    // Test extreme values
    let extreme_logits = vec![
        vec![-1000.0, 0.0, 1000.0], // Very large range
        vec![1e-10, 2e-10, 3e-10],  // Very small values
        vec![std::f64::MAX / 1e6, std::f64::MAX / 1e6, std::f64::MAX / 1e6], // Large values
    ];
    
    for logits in extreme_logits {
        let result = optimized.softmax_avx512_optimized(&logits);
        assert!(result.is_ok(), "Softmax failed on extreme values: {:?}", logits);
        
        if let Ok(probs) = result {
            // Check for NaN or infinity
            for (i, &prob) in probs.iter().enumerate() {
                assert!(prob.is_finite(), "Non-finite probability at {}: {}", i, prob);
                assert!(prob >= 0.0, "Negative probability at {}: {}", i, prob);
            }
            
            // Check sum
            let sum: f64 = probs.iter().sum();
            assert!((sum - 1.0).abs() < 1e-10, "Probability sum {} != 1.0", sum);
        }
    }
}

/// Property-based testing for quantile algorithms
proptest! {
    #[test]
    fn test_quantile_properties(
        data in prop::collection::vec(0.0..1000.0, 10..1000),
        quantile in 0.1..0.99f64
    ) {
        let mut gk = GreenwaldKhannaQuantile::new(quantile, 0.01);
        
        // Insert all data
        for value in &data {
            gk.insert(*value);
        }
        
        let result = gk.query();
        prop_assert!(result.is_some());
        
        let q_value = result.unwrap();
        
        // Count values below quantile
        let below_count = data.iter().filter(|&&x| x < q_value).count();
        let actual_quantile = below_count as f64 / data.len() as f64;
        
        // Should be reasonably close to target quantile
        let error = (actual_quantile - quantile).abs();
        prop_assert!(error < 0.1, "Quantile error {} too high", error);
    }
    
    #[test]
    fn test_softmax_properties(
        logits in prop::collection::vec(-10.0..10.0f64, 2..64)
    ) {
        let config = create_test_config();
        let mut optimized = OptimizedConformalPredictor::new(&config).unwrap();
        
        let result = optimized.softmax_avx512_optimized(&logits);
        prop_assert!(result.is_ok());
        
        let probs = result.unwrap();
        prop_assert_eq!(probs.len(), logits.len());
        
        // All probabilities should be positive
        for &prob in &probs {
            prop_assert!(prob > 0.0);
            prop_assert!(prob.is_finite());
        }
        
        // Should sum to 1
        let sum: f64 = probs.iter().sum();
        prop_assert!((sum - 1.0).abs() < 1e-10);
        
        // Largest logit should correspond to largest probability
        let max_logit_idx = logits.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap().0;
        let max_prob_idx = probs.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap().0;
        prop_assert_eq!(max_logit_idx, max_prob_idx);
    }
}

/// Test memory-optimized data structures correctness
#[test]
fn test_memory_structures_correctness() {
    // Test CacheAlignedVec
    let mut vec: CacheAlignedVec<f64> = CacheAlignedVec::new(100).unwrap();
    
    // Fill with test data
    for i in 0..100 {
        vec[i] = i as f64 * 0.1;
    }
    
    // Verify data integrity
    for i in 0..100 {
        assert_eq!(vec[i], i as f64 * 0.1);
    }
    
    // Test slice operations
    let slice = vec.as_slice();
    assert_eq!(slice.len(), 100);
    assert_eq!(slice[50], 5.0);
    
    // Test resize
    vec.resize(50).unwrap();
    assert_eq!(vec.len(), 50);
    
    // Test copy_from_slice
    let source: Vec<f64> = (0..30).map(|i| i as f64 * 2.0).collect();
    vec.copy_from_slice(&source).unwrap();
    assert_eq!(vec.len(), 30);
    assert_eq!(vec[10], 20.0);
}

/// Test conformal data layout correctness
#[test]
fn test_conformal_data_layout_correctness() {
    let layout = ConformalDataLayout::new(500, 1000).unwrap();
    
    // Verify buffer sizes
    assert_eq!(layout.predictions.len(), 500);
    assert_eq!(layout.calibration_scores.len(), 1000);
    assert_eq!(layout.work_buffer.len(), 1000); // max of the two
    assert_eq!(layout.results_buffer.len(), 1000); // predictions * 2
    
    // Test cache efficiency analysis
    let efficiency = layout.validate_cache_efficiency();
    assert!(efficiency.cache_aligned);
    assert!(efficiency.cache_utilization > 0.0);
    
    // Verify memory alignment
    let pred_ptr = layout.predictions.as_ptr() as usize;
    let calib_ptr = layout.calibration_scores.as_ptr() as usize;
    
    assert_eq!(pred_ptr % 64, 0, "Predictions not cache aligned");
    assert_eq!(calib_ptr % 64, 0, "Calibration scores not cache aligned");
}

/// Test mathematical invariants are preserved
#[test]
fn test_mathematical_invariants() {
    let config = create_test_config();
    let mut optimized = OptimizedConformalPredictor::new(&config).unwrap();
    
    // Test conformal prediction mathematical properties
    let predictions = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let calibration_data = vec![0.5, 1.5, 2.5, 3.5, 4.5, 0.1, 0.9, 1.1, 1.9];
    let confidence = 0.9;
    
    let intervals = optimized.predict_optimized(&predictions, &calibration_data, confidence).unwrap();
    
    // Invariant 1: All intervals should be valid (lower ≤ upper)
    for (i, (lower, upper)) in intervals.iter().enumerate() {
        assert!(lower <= upper, "Invalid interval at {}: ({}, {})", i, lower, upper);
    }
    
    // Invariant 2: Higher confidence should lead to wider intervals
    let intervals_high = optimized.predict_optimized(&predictions, &calibration_data, 0.99).unwrap();
    
    for i in 0..predictions.len() {
        let width_90 = intervals[i].1 - intervals[i].0;
        let width_99 = intervals_high[i].1 - intervals_high[i].0;
        
        // Allow for small numerical differences due to quantile approximation
        assert!(width_99 >= width_90 * 0.8, 
            "Higher confidence should lead to wider intervals at {}: {} vs {}", 
            i, width_90, width_99);
    }
    
    // Invariant 3: Intervals should be centered around predictions (for symmetric case)
    for (i, ((lower, upper), &pred)) in intervals.iter().zip(&predictions).enumerate() {
        let center = (lower + upper) / 2.0;
        let error = (center - pred).abs();
        
        // Should be reasonably close to prediction (allowing for calibration bias)
        assert!(error < 2.0, "Interval not centered around prediction at {}: center={}, pred={}", 
            i, center, pred);
    }
}

/// Integration test combining all optimizations
#[test]
fn test_integration_all_optimizations() {
    let config = create_test_config();
    let mut optimized = OptimizedConformalPredictor::new(&config).unwrap();
    
    // Simulate realistic trading scenario
    let num_assets = 32;
    let num_calibration = 200;
    
    let logits: Vec<f64> = (0..num_assets).map(|i| {
        (i as f64 * 0.1).sin() * 2.0 + (i as f64 * 0.05).cos()
    }).collect();
    
    let calibration_logits: Vec<Vec<f64>> = (0..num_calibration).map(|i| {
        (0..num_assets).map(|j| {
            ((i + j) as f64 * 0.02).sin() + ((i - j) as f64 * 0.03).cos() * 0.5
        }).collect()
    }).collect();
    
    let calibration_labels: Vec<usize> = (0..num_calibration).map(|i| i % num_assets).collect();
    let confidence = 0.95;
    
    // Test all ATS-CP variants
    for variant in [AtsCpVariant::GQ, AtsCpVariant::AQ, AtsCpVariant::MGQ].iter() {
        let start = std::time::Instant::now();
        
        let result = optimized.ats_cp_predict_optimized(
            &logits,
            &calibration_logits,
            &calibration_labels,
            confidence,
            variant.clone(),
        );
        
        let elapsed = start.elapsed();
        
        // Should succeed
        assert!(result.is_ok(), "ATS-CP failed for variant {:?}", variant);
        
        let ats_result = result.unwrap();
        
        // Should meet latency target
        assert!(elapsed.as_micros() < 50, // Allow some margin for test environment
            "Latency target exceeded for {:?}: {}μs", variant, elapsed.as_micros());
        
        // Validate result structure
        assert!(!ats_result.conformal_set.is_empty(), "Empty conformal set");
        assert_eq!(ats_result.calibrated_probabilities.len(), logits.len());
        assert_eq!(ats_result.coverage_guarantee, confidence);
        assert!(ats_result.optimal_temperature > 0.0);
        assert!(ats_result.quantile_threshold >= 0.0);
        
        // Validate probabilities
        let prob_sum: f64 = ats_result.calibrated_probabilities.iter().sum();
        assert_relative_eq!(prob_sum, 1.0, epsilon = 1e-10);
        
        for &prob in &ats_result.calibrated_probabilities {
            assert!(prob > 0.0 && prob.is_finite());
        }
    }
}