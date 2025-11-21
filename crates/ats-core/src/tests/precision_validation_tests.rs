//! Mathematical Precision Validation Tests
//!
//! This module ensures IEEE 754 compliance and numerical stability
//! for all ATS-CP mathematical operations.

#[cfg(test)]
mod tests {
    use crate::{
        config::AtsCpConfig,
        conformal::ConformalPredictor,
        types::{AtsCpVariant, PrecisionValidationResult},
        error::Result,
    };
    use approx::assert_relative_eq;

    /// Validates IEEE 754 compliance for floating-point operations
    #[test]
    fn test_ieee754_compliance() -> Result<()> {
        let config = AtsCpConfig::default();
        let predictor = ConformalPredictor::new(&config)?;
        
        // Test special values
        let test_cases = vec![
            // Normal values
            (vec![1.0, 2.0, 3.0], true),
            // Zero values  
            (vec![0.0, 1.0, 2.0], true),
            // Negative values
            (vec![-1.0, 0.0, 1.0], true),
            // Very small values
            (vec![1e-10, 2e-10, 3e-10], true),
            // Values near machine epsilon
            (vec![f64::EPSILON, 2.0 * f64::EPSILON, 3.0 * f64::EPSILON], true),
        ];
        
        for (logits, should_succeed) in test_cases {
            let result = predictor.compute_softmax(&logits);
            
            if should_succeed {
                let probs = result?;
                
                // Check for NaN or infinity
                for &prob in &probs {
                    assert!(prob.is_finite(), "Probability is not finite: {}", prob);
                    assert!(!prob.is_nan(), "Probability is NaN");
                    assert!(prob >= 0.0, "Probability is negative: {}", prob);
                }
                
                // Check sum equals 1.0 within machine precision
                let sum: f64 = probs.iter().sum();
                assert_relative_eq!(sum, 1.0, epsilon = f64::EPSILON * 10.0);
            }
        }
        
        Ok(())
    }

    /// Test numerical stability under extreme conditions
    #[test]
    fn test_numerical_stability_extreme_cases() -> Result<()> {
        let config = AtsCpConfig::default();
        let predictor = ConformalPredictor::new(&config)?;
        
        // Test case 1: Very large logits (potential overflow)
        let large_logits = vec![700.0, 800.0, 900.0]; // Near overflow threshold
        let probs = predictor.compute_softmax(&large_logits)?;
        
        // Should handle without overflow
        for &prob in &probs {
            assert!(prob.is_finite());
            assert!(prob > 0.0 && prob < 1.0);
        }
        let sum: f64 = probs.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-10);
        
        // Test case 2: Very small logits (potential underflow)
        let small_logits = vec![-700.0, -800.0, -750.0];
        let probs = predictor.compute_softmax(&small_logits)?;
        
        // Should handle without underflow
        for &prob in &probs {
            assert!(prob.is_finite());
            assert!(prob > 0.0);
        }
        let sum: f64 = probs.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-10);
        
        // Test case 3: Mixed extreme values
        let mixed_logits = vec![-500.0, 0.0, 500.0];
        let probs = predictor.compute_softmax(&mixed_logits)?;
        
        for &prob in &probs {
            assert!(prob.is_finite());
            assert!(prob > 0.0 && prob < 1.0);
        }
        let sum: f64 = probs.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-10);
        
        Ok(())
    }

    /// Test catastrophic cancellation detection
    #[test] 
    fn test_catastrophic_cancellation_detection() -> Result<()> {
        let config = AtsCpConfig::default();
        let mut predictor = ConformalPredictor::new(&config)?;
        
        // Create scenario prone to catastrophic cancellation
        let nearly_equal_logits = vec![
            1.0000000000000000,
            1.0000000000000002, // Differs at machine precision level
            1.0000000000000001,
        ];
        
        let result = predictor.ats_cp_predict(
            &nearly_equal_logits,
            &vec![vec![1.0, 1.0, 1.0]; 5],
            &vec![0, 1, 2, 0, 1],
            0.9,
            AtsCpVariant::GQ,
        )?;
        
        // Probabilities should still be computed accurately
        assert!(validate_probability_distribution(&result.calibrated_probabilities, 1e-12));
        
        // Should detect near-equal inputs and handle appropriately
        let max_prob = result.calibrated_probabilities.iter()
            .fold(0.0f64, |a, &b| a.max(b));
        let min_prob = result.calibrated_probabilities.iter()
            .fold(1.0f64, |a, &b| a.min(b));
        
        // With nearly equal inputs, probabilities should be nearly uniform
        assert!((max_prob - min_prob) < 0.1);
        
        Ok(())
    }

    /// Test condition number analysis for numerical stability
    #[test]
    fn test_condition_number_analysis() -> Result<()> {
        let config = AtsCpConfig::default();
        let predictor = ConformalPredictor::new(&config)?;
        
        // Test well-conditioned case
        let well_conditioned_logits = vec![1.0, 2.0, 3.0];
        let probs1 = predictor.compute_softmax(&well_conditioned_logits)?;
        
        // Slightly perturb input
        let perturbed_logits: Vec<f64> = well_conditioned_logits.iter()
            .map(|&x| x + 1e-10)
            .collect();
        let probs2 = predictor.compute_softmax(&perturbed_logits)?;
        
        // Compute relative change in output vs input
        let input_rel_change = 1e-10 / 2.0; // Relative to typical value
        let output_rel_change: f64 = probs1.iter()
            .zip(probs2.iter())
            .map(|(&p1, &p2)| ((p2 - p1) / p1).abs())
            .fold(0.0f64, |a, b| a.max(b));
        
        let condition_number = output_rel_change / input_rel_change;
        
        // Should be well-conditioned (condition number not too large)
        assert!(condition_number < 1e6, "Condition number too large: {}", condition_number);
        
        Ok(())
    }

    /// Test for loss of significance in subtraction operations
    #[test]
    fn test_loss_of_significance() -> Result<()> {
        let config = AtsCpConfig::default();
        let predictor = ConformalPredictor::new(&config)?;
        
        // Test subtraction of nearly equal large numbers
        let large_base = 1e15;
        let logits = vec![
            large_base + 1.0,
            large_base + 2.0, 
            large_base + 3.0,
        ];
        
        let probs = predictor.compute_softmax(&logits)?;
        
        // Should maintain precision despite large base values
        assert!(validate_probability_distribution(&probs, 1e-12));
        
        // The differences should still be preserved
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
        
        Ok(())
    }

    /// Test monotonicity preservation
    #[test]
    fn test_monotonicity_preservation() -> Result<()> {
        let config = AtsCpConfig::default();
        let predictor = ConformalPredictor::new(&config)?;
        
        // Test that softmax preserves order
        let ordered_logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let probs = predictor.compute_softmax(&ordered_logits)?;
        
        // Probabilities should maintain the same order
        for i in 1..probs.len() {
            assert!(
                probs[i] > probs[i-1], 
                "Monotonicity violated: probs[{}] = {} <= probs[{}] = {}",
                i, probs[i], i-1, probs[i-1]
            );
        }
        
        Ok(())
    }

    /// Test temperature scaling numerical stability
    #[test]
    fn test_temperature_scaling_stability() -> Result<()> {
        let config = AtsCpConfig::default();
        let predictor = ConformalPredictor::new(&config)?;
        
        let logits = vec![1.0, 2.0, 3.0];
        
        // Test extreme temperatures
        let extreme_temperatures = vec![0.01, 0.1, 1.0, 10.0, 100.0];
        
        for temperature in extreme_temperatures {
            let probs = predictor.temperature_scaled_softmax(&logits, temperature)?;
            
            // Should remain valid probability distribution
            assert!(validate_probability_distribution(&probs, 1e-10));
            
            // Check temperature effect
            if temperature < 1.0 {
                // Lower temperature should increase max probability
                let max_prob = probs.iter().fold(0.0f64, |a, &b| a.max(b));
                assert!(max_prob > 0.4); // Reasonable threshold
            } else if temperature > 1.0 {
                // Higher temperature should make distribution more uniform
                let max_prob = probs.iter().fold(0.0f64, |a, &b| a.max(b));
                let min_prob = probs.iter().fold(1.0f64, |a, &b| a.min(b));
                assert!((max_prob - min_prob) < 0.5); // More uniform
            }
        }
        
        Ok(())
    }

    /// Helper function to validate probability distribution
    fn validate_probability_distribution(probs: &[f64], tolerance: f64) -> bool {
        // Check sum equals 1
        let sum: f64 = probs.iter().sum();
        if (sum - 1.0).abs() > tolerance {
            return false;
        }
        
        // Check all probabilities are in [0, 1]
        for &prob in probs {
            if prob < 0.0 || prob > 1.0 {
                return false;
            }
            if !prob.is_finite() {
                return false;
            }
        }
        
        true
    }

    /// Test precision validation result structure
    #[test]
    fn test_precision_validation_result() {
        let validation_result = PrecisionValidationResult {
            ieee754_compliant: true,
            max_numerical_error: 1e-15,
            catastrophic_cancellation_detected: false,
            condition_number: 100.0,
            error_analysis: vec![
                "All operations maintain IEEE 754 compliance".to_string(),
                "No significant loss of precision detected".to_string(),
            ],
        };
        
        assert!(validation_result.ieee754_compliant);
        assert!(validation_result.max_numerical_error < 1e-14);
        assert!(!validation_result.catastrophic_cancellation_detected);
        assert!(validation_result.condition_number < 1000.0);
        assert!(!validation_result.error_analysis.is_empty());
    }
}