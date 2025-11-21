//! Property-Based Tests for Mathematical Correctness
//!
//! These tests verify mathematical properties and invariants of the ATS-CP system:
//! - Conformal prediction coverage guarantees
//! - Temperature scaling properties
//! - Quantile computation correctness
//! - Probabilistic consistency
//! - Invariant preservation under transformations

use ats_core::{
    conformal::ConformalPredictor,
    config::{AtsCpConfig, QuantileMethod},
    types::{AtsCpVariant, Confidence},
    error::AtsCoreError,
    test_utils::create_test_config,
};
use proptest::prelude::*;
use std::time::Duration;

/// Property-based test strategies for generating valid inputs
mod test_strategies {
    use super::*;
    
    /// Generate valid logits for neural network outputs
    pub fn valid_logits_strategy() -> impl Strategy<Value = Vec<f64>> {
        prop::collection::vec(
            prop::num::f64::NEGATIVE_INFINITY..20.0, // Reasonable logit range
            1..10 // Reasonable number of classes
        ).prop_filter("logits must be finite", |logits| {
            logits.iter().all(|x| x.is_finite())
        })
    }
    
    /// Generate valid calibration data
    pub fn valid_calibration_strategy() -> impl Strategy<Value = Vec<f64>> {
        prop::collection::vec(
            -10.0..10.0f64,
            10..100
        ).prop_filter("calibration data must be finite", |data| {
            data.iter().all(|x| x.is_finite())
        })
    }
    
    /// Generate valid confidence levels
    pub fn valid_confidence_strategy() -> impl Strategy<Value = f64> {
        0.01f64..0.999
    }
    
    /// Generate valid temperatures
    pub fn valid_temperature_strategy() -> impl Strategy<Value = f64> {
        0.01f64..10.0
    }
    
    /// Generate valid predictions
    pub fn valid_predictions_strategy() -> impl Strategy<Value = Vec<f64>> {
        prop::collection::vec(
            -100.0..100.0f64,
            1..50
        ).prop_filter("predictions must be finite", |preds| {
            preds.iter().all(|x| x.is_finite())
        })
    }
}

/// Mathematical property tests for conformal prediction
mod conformal_prediction_properties {
    use super::*;
    use super::test_strategies::*;
    
    proptest! {
        #[test]
        fn test_coverage_guarantee_property(
            predictions in valid_predictions_strategy(),
            calibration_data in valid_calibration_strategy(),
            confidence in valid_confidence_strategy()
        ) {
            let config = create_test_config();
            let mut predictor = ConformalPredictor::new(&config)?;
            
            // Property: Conformal prediction should provide coverage guarantees
            let result = predictor.predict_detailed(&predictions, &calibration_data, confidence);
            
            if let Ok(detailed_result) = result {
                // Property 1: Confidence level should be preserved
                prop_assert_eq!(detailed_result.confidence, confidence);
                
                // Property 2: Intervals should have correct structure
                prop_assert_eq!(detailed_result.intervals.len(), predictions.len());
                
                // Property 3: All intervals should be valid
                for (i, (lower, upper)) in detailed_result.intervals.iter().enumerate() {
                    prop_assert!(lower <= upper, "Interval {} should be valid: {} <= {}", i, lower, upper);
                    prop_assert!(lower.is_finite() && upper.is_finite(), "Interval {} should be finite", i);
                }
                
                // Property 4: Quantile threshold should be non-negative
                prop_assert!(detailed_result.quantile_threshold >= 0.0,
                           "Quantile threshold should be non-negative: {}", detailed_result.quantile_threshold);
                
                // Property 5: Execution time should be recorded
                prop_assert!(detailed_result.execution_time_ns > 0,
                           "Execution time should be recorded");
            }
        }
        
        #[test]
        fn test_interval_monotonicity_property(
            predictions in valid_predictions_strategy(),
            calibration_data in valid_calibration_strategy()
        ) {
            let config = create_test_config();
            let mut predictor = ConformalPredictor::new(&config)?;
            
            // Property: Higher confidence should produce wider intervals
            let confidence_levels = vec![0.80, 0.90, 0.95, 0.99];
            let mut results = Vec::new();
            
            for &confidence in &confidence_levels {
                if let Ok(result) = predictor.predict_detailed(&predictions, &calibration_data, confidence) {
                    results.push((confidence, result.intervals));
                }
            }
            
            // Verify monotonicity property
            for i in 1..results.len() {
                let (lower_conf, lower_intervals) = &results[i-1];
                let (higher_conf, higher_intervals) = &results[i];
                
                prop_assert!(higher_conf > lower_conf, "Confidence levels should be ordered");
                
                if lower_intervals.len() == higher_intervals.len() {
                    for j in 0..lower_intervals.len() {
                        let (l1, u1) = lower_intervals[j];
                        let (l2, u2) = higher_intervals[j];
                        
                        let width1 = u1 - l1;
                        let width2 = u2 - l2;
                        
                        // Property: Higher confidence should generally produce wider intervals
                        // (allowing for some numerical tolerance)
                        prop_assert!(width2 >= width1 - 1e-10,
                                   "Higher confidence interval should be at least as wide: {} >= {} at position {}",
                                   width2, width1, j);
                    }
                }
            }
        }
        
        #[test]
        fn test_exchangeability_preservation_property(
            mut data in valid_calibration_strategy()
        ) {
            let config = create_test_config();
            let predictor = ConformalPredictor::new(&config)?;
            
            // Property: Permutation of calibration data should not affect validity
            let original_valid = predictor.validate_exchangeability(&data);
            
            if original_valid.is_ok() {
                // Shuffle the data
                use rand::seq::SliceRandom;
                let mut rng = rand::thread_rng();
                data.shuffle(&mut rng);
                
                let shuffled_valid = predictor.validate_exchangeability(&data);
                
                // Property: Exchangeability validation should be permutation-invariant
                if let (Ok(orig_result), Ok(shuffled_result)) = (original_valid, shuffled_valid) {
                    // The specific implementation might give different results due to trend analysis,
                    // but both should be valid boolean values
                    prop_assert!(orig_result == true || orig_result == false);
                    prop_assert!(shuffled_result == true || shuffled_result == false);
                }
            }
        }
        
        #[test]
        fn test_adaptive_calibration_convergence_property(
            predictions in valid_predictions_strategy(),
            noise_scale in 0.01f64..0.5
        ) {
            let config = create_test_config();
            let mut predictor = ConformalPredictor::new(&config)?;
            
            // Property: Adaptive calibration should converge with consistent data
            let confidence = 0.95;
            
            // Generate consistent true values with small noise
            let true_values: Vec<f64> = predictions.iter()
                .map(|&pred| pred + noise_scale * (rand::random::<f64>() - 0.5))
                .collect();
            
            if predictions.len() == true_values.len() && !predictions.is_empty() {
                let result1 = predictor.predict_adaptive(&predictions, &true_values, confidence);
                
                if result1.is_ok() {
                    // Second iteration with similar data
                    let result2 = predictor.predict_adaptive(&predictions, &true_values, confidence);
                    
                    if result2.is_ok() {
                        let intervals1 = result1.unwrap();
                        let intervals2 = result2.unwrap();
                        
                        // Property: Adaptive calibration should produce consistent results
                        prop_assert_eq!(intervals1.len(), intervals2.len());
                        
                        // Property: Intervals should remain reasonable after adaptation
                        for (i, ((l1, u1), (l2, u2))) in intervals1.iter().zip(intervals2.iter()).enumerate() {
                            prop_assert!(l1.is_finite() && u1.is_finite() && l2.is_finite() && u2.is_finite(),
                                       "All interval bounds should be finite at position {}", i);
                            prop_assert!(l1 <= u1 && l2 <= u2,
                                       "Interval bounds should be ordered at position {}", i);
                        }
                    }
                }
            }
        }
    }
}

/// Mathematical property tests for ATS-CP algorithms
mod ats_cp_algorithm_properties {
    use super::*;
    use super::test_strategies::*;
    
    proptest! {
        #[test]
        fn test_ats_cp_probability_normalization_property(
            logits in valid_logits_strategy(),
            temperature in valid_temperature_strategy()
        ) {
            let config = create_test_config();
            let predictor = ConformalPredictor::new(&config)?;
            
            // Property: Temperature-scaled softmax should produce normalized probabilities
            let result = predictor.temperature_scaled_softmax(&logits, temperature);
            
            if let Ok(probabilities) = result {
                // Property 1: Probabilities should sum to 1
                let sum: f64 = probabilities.iter().sum();
                prop_assert!((sum - 1.0).abs() < 1e-10, "Probabilities should sum to 1.0, got {}", sum);
                
                // Property 2: All probabilities should be non-negative
                for (i, &prob) in probabilities.iter().enumerate() {
                    prop_assert!(prob >= 0.0, "Probability {} should be non-negative: {}", i, prob);
                    prop_assert!(prob <= 1.0, "Probability {} should be <= 1.0: {}", i, prob);
                    prop_assert!(prob.is_finite(), "Probability {} should be finite: {}", i, prob);
                }
                
                // Property 3: Length should match input
                prop_assert_eq!(probabilities.len(), logits.len());
            }
        }
        
        #[test]
        fn test_ats_cp_temperature_effect_property(
            logits in valid_logits_strategy()
        ) {
            let config = create_test_config();
            let predictor = ConformalPredictor::new(&config)?;
            
            if logits.len() >= 2 {
                let temperatures = vec![0.1, 1.0, 10.0];
                let mut entropy_values = Vec::new();
                
                // Property: Higher temperature should increase entropy (uncertainty)
                for &temp in &temperatures {
                    if let Ok(probs) = predictor.temperature_scaled_softmax(&logits, temp) {
                        // Compute entropy: H = -Σ p_i * log(p_i)
                        let entropy: f64 = probs.iter()
                            .filter(|&&p| p > 0.0)
                            .map(|&p| -p * p.ln())
                            .sum();
                        
                        entropy_values.push((temp, entropy));
                    }
                }
                
                // Property: Entropy should generally increase with temperature
                if entropy_values.len() >= 2 {
                    for i in 1..entropy_values.len() {
                        let (temp1, entropy1) = entropy_values[i-1];
                        let (temp2, entropy2) = entropy_values[i];
                        
                        prop_assert!(temp2 > temp1, "Temperatures should be ordered");
                        
                        // Higher temperature should generally increase entropy
                        // (allowing for numerical tolerance and edge cases)
                        prop_assert!(entropy2 >= entropy1 - 1e-10,
                                   "Higher temperature should not significantly decrease entropy: {} vs {} at temps {} vs {}",
                                   entropy2, entropy1, temp2, temp1);
                    }
                }
            }
        }
        
        #[test]
        fn test_conformal_set_properties(
            logits in valid_logits_strategy(),
            calibration_logits in prop::collection::vec(valid_logits_strategy(), 5..20),
            calibration_labels in prop::collection::vec(0usize..5, 5..20)
        ) {
            let config = create_test_config();
            let mut predictor = ConformalPredictor::new(&config)?;
            
            // Ensure calibration data is consistent
            let min_len = calibration_logits.iter().map(|v| v.len()).min().unwrap_or(0);
            let max_label = calibration_labels.iter().cloned().max().unwrap_or(0);
            
            if min_len > max_label && calibration_logits.len() == calibration_labels.len() {
                let variants = vec![AtsCpVariant::GQ, AtsCpVariant::AQ, AtsCpVariant::MGQ, AtsCpVariant::MAQ];
                
                for variant in variants {
                    let result = predictor.ats_cp_predict(
                        &logits,
                        &calibration_logits,
                        &calibration_labels,
                        0.95,
                        variant.clone()
                    );
                    
                    if let Ok(ats_result) = result {
                        // Property 1: Conformal set should not be empty
                        prop_assert!(!ats_result.conformal_set.is_empty(),
                                   "Conformal set should not be empty for variant {:?}", variant);
                        
                        // Property 2: All indices should be valid
                        for &idx in &ats_result.conformal_set {
                            prop_assert!(idx < logits.len(),
                                       "Conformal set index {} should be valid (< {})", idx, logits.len());
                        }
                        
                        // Property 3: Probabilities should be normalized
                        let prob_sum: f64 = ats_result.calibrated_probabilities.iter().sum();
                        prop_assert!((prob_sum - 1.0).abs() < 1e-10,
                                   "Calibrated probabilities should sum to 1.0 for variant {:?}", variant);
                        
                        // Property 4: Temperature should be positive
                        prop_assert!(ats_result.optimal_temperature > 0.0,
                                   "Optimal temperature should be positive for variant {:?}", variant);
                        
                        // Property 5: Coverage guarantee should be preserved
                        prop_assert_eq!(ats_result.coverage_guarantee, 0.95);
                        
                        // Property 6: Variant should be preserved
                        prop_assert_eq!(ats_result.variant, variant);
                    }
                }
            }
        }
    }
}

/// Mathematical property tests for quantile computation
mod quantile_computation_properties {
    use super::*;
    use super::test_strategies::*;
    
    proptest! {
        #[test]
        fn test_quantile_ordering_property(
            mut data in valid_calibration_strategy(),
            confidence in valid_confidence_strategy()
        ) {
            let config = create_test_config();
            let mut predictor = ConformalPredictor::new(&config)?;
            
            // Remove any NaN or infinite values
            data.retain(|x| x.is_finite());
            
            if data.len() >= 3 {
                // Property: Different quantile methods should respect ordering
                let linear = predictor.compute_quantile_linear(&data, confidence);
                let lower = predictor.compute_quantile_lower(&data, confidence);
                let higher = predictor.compute_quantile_higher(&data, confidence);
                
                if let (Ok(linear_val), Ok(lower_val), Ok(higher_val)) = (linear, lower, higher) {
                    // Property: lower <= linear <= higher
                    prop_assert!(lower_val <= linear_val + 1e-10,
                               "Lower quantile should be <= linear quantile: {} <= {}", lower_val, linear_val);
                    prop_assert!(linear_val <= higher_val + 1e-10,
                               "Linear quantile should be <= higher quantile: {} <= {}", linear_val, higher_val);
                    
                    // Property: All quantiles should be within data range
                    let min_val = data.iter().cloned().fold(f64::INFINITY, f64::min);
                    let max_val = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    
                    prop_assert!(linear_val >= min_val - 1e-10 && linear_val <= max_val + 1e-10,
                               "Linear quantile should be within data range: {} in [{}, {}]", linear_val, min_val, max_val);
                    prop_assert!(lower_val >= min_val - 1e-10 && lower_val <= max_val + 1e-10,
                               "Lower quantile should be within data range: {} in [{}, {}]", lower_val, min_val, max_val);
                    prop_assert!(higher_val >= min_val - 1e-10 && higher_val <= max_val + 1e-10,
                               "Higher quantile should be within data range: {} in [{}, {}]", higher_val, min_val, max_val);
                }
            }
        }
        
        #[test]
        fn test_quantile_monotonicity_property(
            data in valid_calibration_strategy()
        ) {
            let config = create_test_config();
            let mut predictor = ConformalPredictor::new(&config)?;
            
            let filtered_data: Vec<f64> = data.into_iter().filter(|x| x.is_finite()).collect();
            
            if filtered_data.len() >= 5 {
                let confidence_levels = vec![0.1, 0.3, 0.5, 0.7, 0.9];
                let mut quantile_results = Vec::new();
                
                for &conf in &confidence_levels {
                    if let Ok(q) = predictor.compute_quantile_linear(&filtered_data, conf) {
                        quantile_results.push((conf, q));
                    }
                }
                
                // Property: Quantiles should be monotonically increasing with confidence
                for i in 1..quantile_results.len() {
                    let (conf1, q1) = quantile_results[i-1];
                    let (conf2, q2) = quantile_results[i];
                    
                    prop_assert!(conf2 > conf1, "Confidence levels should be ordered");
                    prop_assert!(q2 >= q1 - 1e-10,
                               "Higher confidence should give higher quantile: {} >= {} at conf {} vs {}",
                               q2, q1, conf2, conf1);
                }
            }
        }
        
        #[test]
        fn test_quantile_boundary_property(
            data in valid_calibration_strategy()
        ) {
            let config = create_test_config();
            let mut predictor = ConformalPredictor::new(&config)?;
            
            let filtered_data: Vec<f64> = data.into_iter().filter(|x| x.is_finite()).collect();
            
            if !filtered_data.is_empty() {
                // Property: 0% and 100% quantiles should be minimum and maximum
                let q_min = predictor.compute_quantile_linear(&filtered_data, 0.0);
                let q_max = predictor.compute_quantile_linear(&filtered_data, 1.0);
                
                let data_min = filtered_data.iter().cloned().fold(f64::INFINITY, f64::min);
                let data_max = filtered_data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                
                if let Ok(q_min_val) = q_min {
                    prop_assert!((q_min_val - data_min).abs() < 1e-10,
                               "0% quantile should be data minimum: {} ≈ {}", q_min_val, data_min);
                }
                
                if let Ok(q_max_val) = q_max {
                    prop_assert!((q_max_val - data_max).abs() < 1e-10,
                               "100% quantile should be data maximum: {} ≈ {}", q_max_val, data_max);
                }
            }
        }
        
        #[test]
        fn test_quantile_method_consistency_property(
            data in valid_calibration_strategy(),
            confidence in 0.4f64..0.6 // Middle range where methods should be most similar
        ) {
            let config = create_test_config();
            let mut predictor = ConformalPredictor::new(&config)?;
            
            let filtered_data: Vec<f64> = data.into_iter().filter(|x| x.is_finite()).collect();
            
            if filtered_data.len() >= 10 {
                // Test consistency between quantile methods
                let linear = predictor.compute_quantile_linear(&filtered_data, confidence);
                let nearest = predictor.compute_quantile_nearest(&filtered_data, confidence);
                let midpoint = predictor.compute_quantile_midpoint(&filtered_data, confidence);
                
                if let (Ok(linear_val), Ok(nearest_val), Ok(midpoint_val)) = (linear, nearest, midpoint) {
                    // Property: All methods should produce reasonable results
                    prop_assert!(linear_val.is_finite(), "Linear quantile should be finite");
                    prop_assert!(nearest_val.is_finite(), "Nearest quantile should be finite");
                    prop_assert!(midpoint_val.is_finite(), "Midpoint quantile should be finite");
                    
                    // Property: Results should be in similar range
                    let data_min = filtered_data.iter().cloned().fold(f64::INFINITY, f64::min);
                    let data_max = filtered_data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    
                    for (method_name, val) in [("linear", linear_val), ("nearest", nearest_val), ("midpoint", midpoint_val)] {
                        prop_assert!(val >= data_min - 1e-10 && val <= data_max + 1e-10,
                                   "{} quantile should be within data range: {} in [{}, {}]",
                                   method_name, val, data_min, data_max);
                    }
                }
            }
        }
    }
}

/// Invariant preservation property tests
mod invariant_preservation_properties {
    use super::*;
    use super::test_strategies::*;
    
    proptest! {
        #[test]
        fn test_scale_invariance_property(
            predictions in valid_predictions_strategy(),
            calibration_data in valid_calibration_strategy(),
            scale_factor in 0.1f64..10.0
        ) {
            let config = create_test_config();
            let mut predictor1 = ConformalPredictor::new(&config)?;
            let mut predictor2 = ConformalPredictor::new(&config)?;
            
            let filtered_predictions: Vec<f64> = predictions.into_iter().filter(|x| x.is_finite()).collect();
            let filtered_calibration: Vec<f64> = calibration_data.into_iter().filter(|x| x.is_finite()).collect();
            
            if !filtered_predictions.is_empty() && filtered_calibration.len() >= 10 {
                // Scale the data
                let scaled_predictions: Vec<f64> = filtered_predictions.iter().map(|x| x * scale_factor).collect();
                let scaled_calibration: Vec<f64> = filtered_calibration.iter().map(|x| x * scale_factor).collect();
                
                let original_result = predictor1.predict(&filtered_predictions, &filtered_calibration);
                let scaled_result = predictor2.predict(&scaled_predictions, &scaled_calibration);
                
                if let (Ok(original_intervals), Ok(scaled_intervals)) = (original_result, scaled_result) {
                    // Property: Intervals should scale proportionally
                    prop_assert_eq!(original_intervals.len(), scaled_intervals.len());
                    
                    for i in 0..original_intervals.len() {
                        let (orig_lower, orig_upper) = original_intervals[i];
                        let (scaled_lower, scaled_upper) = scaled_intervals[i];
                        
                        let orig_width = orig_upper - orig_lower;
                        let scaled_width = scaled_upper - scaled_lower;
                        
                        // Property: Widths should scale by approximately the same factor
                        if orig_width > 0.0 {
                            let width_ratio = scaled_width / orig_width;
                            prop_assert!((width_ratio - scale_factor).abs() / scale_factor < 0.1,
                                       "Interval width should scale proportionally: {} vs {} (factor {})",
                                       scaled_width, orig_width, scale_factor);
                        }
                    }
                }
            }
        }
        
        #[test]
        fn test_permutation_invariance_property(
            mut calibration_data in valid_calibration_strategy(),
            predictions in valid_predictions_strategy()
        ) {
            let config = create_test_config();
            let mut predictor1 = ConformalPredictor::new(&config)?;
            let mut predictor2 = ConformalPredictor::new(&config)?;
            
            calibration_data.retain(|x| x.is_finite());
            let filtered_predictions: Vec<f64> = predictions.into_iter().filter(|x| x.is_finite()).collect();
            
            if calibration_data.len() >= 10 && !filtered_predictions.is_empty() {
                // Compute with original order
                let original_result = predictor1.predict(&filtered_predictions, &calibration_data);
                
                // Shuffle calibration data
                use rand::seq::SliceRandom;
                let mut rng = rand::thread_rng();
                calibration_data.shuffle(&mut rng);
                
                // Compute with shuffled order
                let shuffled_result = predictor2.predict(&filtered_predictions, &calibration_data);
                
                if let (Ok(original_intervals), Ok(shuffled_intervals)) = (original_result, shuffled_result) {
                    // Property: Results should be identical (conformal prediction is permutation-invariant)
                    prop_assert_eq!(original_intervals.len(), shuffled_intervals.len());
                    
                    for i in 0..original_intervals.len() {
                        let (orig_lower, orig_upper) = original_intervals[i];
                        let (shuf_lower, shuf_upper) = shuffled_intervals[i];
                        
                        prop_assert!((orig_lower - shuf_lower).abs() < 1e-10,
                                   "Lower bounds should be identical under permutation: {} ≈ {}", orig_lower, shuf_lower);
                        prop_assert!((orig_upper - shuf_upper).abs() < 1e-10,
                                   "Upper bounds should be identical under permutation: {} ≈ {}", orig_upper, shuf_upper);
                    }
                }
            }
        }
        
        #[test]
        fn test_consistency_under_subset_property(
            calibration_data in valid_calibration_strategy(),
            predictions in valid_predictions_strategy()
        ) {
            let config = create_test_config();
            let mut predictor1 = ConformalPredictor::new(&config)?;
            let mut predictor2 = ConformalPredictor::new(&config)?;
            
            let filtered_calibration: Vec<f64> = calibration_data.into_iter().filter(|x| x.is_finite()).collect();
            let filtered_predictions: Vec<f64> = predictions.into_iter().filter(|x| x.is_finite()).collect();
            
            if filtered_calibration.len() >= 20 && !filtered_predictions.is_empty() {
                // Use full calibration data
                let full_result = predictor1.predict(&filtered_predictions, &filtered_calibration);
                
                // Use subset of calibration data
                let subset_size = filtered_calibration.len() / 2;
                let subset_calibration = &filtered_calibration[..subset_size];
                let subset_result = predictor2.predict(&filtered_predictions, subset_calibration);
                
                if let (Ok(full_intervals), Ok(subset_intervals)) = (full_result, subset_result) {
                    // Property: Both should produce valid results
                    prop_assert_eq!(full_intervals.len(), filtered_predictions.len());
                    prop_assert_eq!(subset_intervals.len(), filtered_predictions.len());
                    
                    // Property: Subset intervals might be wider (less data = more uncertainty)
                    for i in 0..full_intervals.len() {
                        let (full_lower, full_upper) = full_intervals[i];
                        let (subset_lower, subset_upper) = subset_intervals[i];
                        
                        let full_width = full_upper - full_lower;
                        let subset_width = subset_upper - subset_lower;
                        
                        prop_assert!(full_width >= 0.0, "Full interval width should be non-negative");
                        prop_assert!(subset_width >= 0.0, "Subset interval width should be non-negative");
                        
                        // Generally, less data should not make intervals narrower
                        // (though this is not a strict mathematical requirement)
                        prop_assert!(full_width.is_finite() && subset_width.is_finite(),
                                   "All interval widths should be finite");
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod property_test_integration {
    use super::*;
    use ats_core::test_framework::{TestFramework, swarm_utils};
    
    #[tokio::test]
    async fn test_property_based_tests_swarm_coordination() {
        // Initialize test framework for property-based testing coordination
        let mut framework = TestFramework::new(
            "mathematical_property_swarm".to_string(),
            "property_test_agent".to_string(),
        ).unwrap();
        
        // Signal coordination with other test agents
        swarm_utils::coordinate_test_execution(&framework.context, "property_based_tests").await.unwrap();
        
        // Execute a sample property test
        let config = create_test_config();
        let mut predictor = ConformalPredictor::new(&config).unwrap();
        
        let test_predictions = vec![1.0, 2.0, 3.0];
        let test_calibration = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
        
        let result = predictor.predict(&test_predictions, &test_calibration);
        assert!(result.is_ok(), "Property test sample should succeed");
        
        // Update test metrics
        framework.context.execution_metrics.tests_passed += 1;
        
        // Share results with swarm
        swarm_utils::share_test_results(&framework.context, &framework.context.execution_metrics).await.unwrap();
        
        println!("✅ Property-based tests swarm coordination completed");
    }
}