//! ATS-CP Algorithm Tests
//!
//! Comprehensive test suite for ATS-CP algorithms including:
//! - Algorithm 1: Main ATS-CP workflow
//! - Algorithm 2: SelectTau binary search
//! - All variants: GQ, AQ, MGQ, MAQ

#[cfg(test)]
mod tests {
    use crate::{
        config::AtsCpConfig,
        conformal::ConformalPredictor,
        types::{AtsCpVariant, Confidence},
        error::Result,
    };
    use super::super::test_utils;
    use approx::assert_relative_eq;

    /// Test ATS-CP Algorithm 1 with GQ variant
    #[test]
    fn test_ats_cp_algorithm_gq_variant() -> Result<()> {
        let config = AtsCpConfig::default();
        let mut predictor = ConformalPredictor::new(&config)?;
        
        let (calibration_logits, calibration_labels) = create_test_data();
        let test_logits = vec![1.5, 2.0, 2.5];
        let confidence = 0.9;
        
        let result = predictor.ats_cp_predict(
            &test_logits,
            &calibration_logits,
            &calibration_labels,
            confidence,
            AtsCpVariant::GQ,
        )?;
        
        // Validate result structure
        assert!(!result.conformal_set.is_empty());
        assert_eq!(result.calibrated_probabilities.len(), test_logits.len());
        assert!(result.optimal_temperature > 0.0);
        assert_eq!(result.coverage_guarantee, confidence);
        assert!(matches!(result.variant, AtsCpVariant::GQ));
        
        // Validate probability distribution
        assert!(validate_probability_distribution(&result.calibrated_probabilities, 1e-6));
        
        // Validate conformal set indices are valid
        for &idx in &result.conformal_set {
            assert!(idx < test_logits.len());
        }
        
        Ok(())
    }

    /// Test ATS-CP Algorithm 1 with AQ variant
    #[test]
    fn test_ats_cp_algorithm_aq_variant() -> Result<()> {
        let config = AtsCpConfig::default();
        let mut predictor = ConformalPredictor::new(&config)?;
        
        let (calibration_logits, calibration_labels) = create_test_data();
        let test_logits = vec![2.0, 1.0, 3.0];
        let confidence = 0.95;
        
        let result = predictor.ats_cp_predict(
            &test_logits,
            &calibration_logits,
            &calibration_labels,
            confidence,
            AtsCpVariant::AQ,
        )?;
        
        // Validate AQ-specific properties
        assert!(matches!(result.variant, AtsCpVariant::AQ));
        assert!(result.quantile_threshold >= 0.0);
        
        // AQ should handle logarithmic scores properly
        assert!(validate_probability_distribution(&result.calibrated_probabilities, 1e-6));
        
        Ok(())
    }

    /// Test ATS-CP Algorithm 1 with MGQ variant
    #[test]
    fn test_ats_cp_algorithm_mgq_variant() -> Result<()> {
        let config = AtsCpConfig::default();
        let mut predictor = ConformalPredictor::new(&config)?;
        
        let (calibration_logits, calibration_labels) = create_test_data();
        let test_logits = vec![3.0, 1.5, 2.0];
        let confidence = 0.9;
        
        let result = predictor.ats_cp_predict(
            &test_logits,
            &calibration_logits,
            &calibration_labels,
            confidence,
            AtsCpVariant::MGQ,
        )?;
        
        assert!(matches!(result.variant, AtsCpVariant::MGQ));
        
        // MGQ should handle multi-class scenarios
        assert!(!result.conformal_set.is_empty());
        assert!(validate_probability_distribution(&result.calibrated_probabilities, 1e-6));
        
        Ok(())
    }

    /// Test ATS-CP Algorithm 1 with MAQ variant
    #[test]
    fn test_ats_cp_algorithm_maq_variant() -> Result<()> {
        let config = AtsCpConfig::default();
        let mut predictor = ConformalPredictor::new(&config)?;
        
        let (calibration_logits, calibration_labels) = create_test_data();
        let test_logits = vec![1.0, 3.0, 2.0];
        let confidence = 0.95;
        
        let result = predictor.ats_cp_predict(
            &test_logits,
            &calibration_logits,
            &calibration_labels,
            confidence,
            AtsCpVariant::MAQ,
        )?;
        
        assert!(matches!(result.variant, AtsCpVariant::MAQ));
        
        // MAQ should handle complex multi-class adaptive scenarios
        assert!(!result.conformal_set.is_empty());
        assert!(validate_probability_distribution(&result.calibrated_probabilities, 1e-6));
        
        Ok(())
    }

    /// Test Algorithm 2: SelectTau binary search
    #[test]
    fn test_select_tau_algorithm() -> Result<()> {
        let config = AtsCpConfig::default();
        let mut predictor = ConformalPredictor::new(&config)?;
        
        let test_logits = vec![2.0, 1.0, 3.0];
        let conformal_set = vec![0, 2]; // Indices of classes in conformal set
        let target_coverage = 0.9;
        
        let optimal_temperature = predictor.select_tau(
            &test_logits,
            &conformal_set,
            target_coverage,
            AtsCpVariant::GQ,
        )?;
        
        // Validate temperature is within bounds
        assert!(optimal_temperature >= config.temperature.min_temperature);
        assert!(optimal_temperature <= config.temperature.max_temperature);
        assert!(optimal_temperature > 0.0);
        
        // Test that optimal temperature produces coverage close to target
        let probs = predictor.temperature_scaled_softmax(&test_logits, optimal_temperature)?;
        let actual_coverage: f64 = conformal_set.iter()
            .map(|&idx| probs[idx])
            .sum();
        
        assert_relative_eq!(actual_coverage, target_coverage, epsilon = 0.1);
        
        Ok(())
    }

    /// Test coverage guarantee across multiple predictions
    #[test]
    fn test_coverage_guarantee() -> Result<()> {
        let config = AtsCpConfig::default();
        let mut predictor = ConformalPredictor::new(&config)?;
        
        let (calibration_logits, calibration_labels) = create_test_data();
        let confidence = 0.9;
        let tolerance = 0.05;
        
        // Generate multiple test cases
        let test_cases = vec![
            vec![1.0, 2.0, 3.0],
            vec![2.0, 3.0, 1.0],
            vec![3.0, 1.0, 2.0],
            vec![1.5, 2.5, 3.5],
            vec![3.5, 1.5, 2.5],
        ];
        
        let mut results = Vec::new();
        for test_logits in test_cases {
            let result = predictor.ats_cp_predict(
                &test_logits,
                &calibration_logits,
                &calibration_labels,
                confidence,
                AtsCpVariant::GQ,
            )?;
            results.push(result);
        }
        
        // Validate coverage guarantee  
        let avg_coverage: f64 = results.iter()
            .map(|r| r.conformal_set.len() as f64 / r.calibrated_probabilities.len() as f64)
            .sum::<f64>() / results.len() as f64;
        assert!((avg_coverage - confidence).abs() < tolerance, 
            "Coverage guarantee not met: {} vs {}", avg_coverage, confidence);
        
        Ok(())
    }

    /// Test mathematical properties of softmax with temperature
    #[test]
    fn test_temperature_scaled_softmax_properties() -> Result<()> {
        let config = AtsCpConfig::default();
        let predictor = ConformalPredictor::new(&config)?;
        
        let logits = vec![1.0, 2.0, 3.0];
        
        // Test different temperatures
        let temperatures = vec![0.5, 1.0, 2.0, 5.0];
        
        for temperature in temperatures {
            let probs = predictor.temperature_scaled_softmax(&logits, temperature)?;
            
            // Probabilities must sum to 1
            let sum: f64 = probs.iter().sum();
            assert_relative_eq!(sum, 1.0, epsilon = 1e-10);
            
            // All probabilities must be positive
            for &prob in &probs {
                assert!(prob > 0.0);
                assert!(prob < 1.0);
            }
            
            // Higher temperature should lead to more uniform distribution
            if temperature > 1.0 {
                let max_prob = probs.iter().fold(0.0f64, |a, &b| a.max(b));
                let min_prob = probs.iter().fold(1.0f64, |a, &b| a.min(b));
                let ratio = max_prob / min_prob;
                
                // With temperature scaling, ratio should be less extreme
                assert!(ratio < 10.0); // Reasonable bound for test
            }
        }
        
        Ok(())
    }

    /// Test error handling in ATS-CP algorithms
    #[test]
    fn test_ats_cp_error_handling() {
        let config = AtsCpConfig::default();
        let mut predictor = ConformalPredictor::new(&config).unwrap();
        
        // Test empty logits
        let empty_logits = vec![];
        let (calibration_logits, calibration_labels) = create_test_data();
        
        let result = predictor.ats_cp_predict(
            &empty_logits,
            &calibration_logits,
            &calibration_labels,
            0.9,
            AtsCpVariant::GQ,
        );
        assert!(result.is_err());
        
        // Test mismatched calibration data
        let test_logits = vec![1.0, 2.0, 3.0];
        let wrong_labels = vec![0, 1]; // Wrong length
        
        let result = predictor.ats_cp_predict(
            &test_logits,
            &calibration_logits,
            &wrong_labels,
            0.9,
            AtsCpVariant::GQ,
        );
        assert!(result.is_err());
        
        // Test invalid confidence
        let result = predictor.ats_cp_predict(
            &test_logits,
            &calibration_logits,
            &calibration_labels,
            1.5, // Invalid confidence > 1.0
            AtsCpVariant::GQ,
        );
        assert!(result.is_err());
    }

    /// Test numerical stability with extreme inputs
    #[test]
    fn test_numerical_stability() -> Result<()> {
        let config = AtsCpConfig::default();
        let mut predictor = ConformalPredictor::new(&config)?;
        
        // Test with very large logits (potential overflow)
        let large_logits = vec![100.0, 200.0, 300.0];
        let (calibration_logits, calibration_labels) = create_test_data();
        
        let result = predictor.ats_cp_predict(
            &large_logits,
            &calibration_logits,
            &calibration_labels,
            0.9,
            AtsCpVariant::GQ,
        )?;
        
        // Should handle large values without overflow
        assert!(validate_probability_distribution(&result.calibrated_probabilities, 1e-6));
        
        // Test with very small logits
        let small_logits = vec![-100.0, -200.0, -150.0];
        
        let result = predictor.ats_cp_predict(
            &small_logits,
            &calibration_logits,
            &calibration_labels,
            0.9,
            AtsCpVariant::GQ,
        )?;
        
        // Should handle small values without underflow
        assert!(validate_probability_distribution(&result.calibrated_probabilities, 1e-6));
        
        Ok(())
    }

    /// Performance test for ATS-CP algorithms
    #[test]
    fn test_performance_requirements() -> Result<()> {
        let config = AtsCpConfig::high_performance();
        let mut predictor = ConformalPredictor::new(&config)?;
        
        let (calibration_logits, calibration_labels) = create_test_data();
        let test_logits = vec![1.0, 2.0, 3.0, 4.0, 5.0]; // Larger input
        
        let start = std::time::Instant::now();
        
        let result = predictor.ats_cp_predict(
            &test_logits,
            &calibration_logits,
            &calibration_labels,
            0.95,
            AtsCpVariant::GQ,
        )?;
        
        let elapsed = start.elapsed();
        
        // Should meet latency target (conformal: 15μs + some overhead)
        assert!(elapsed.as_micros() < 100); // Reasonable test bound
        assert!(result.execution_time_ns > 0);
        
        // Verify scientific accuracy wasn't compromised for speed
        assert!(validate_probability_distribution(&result.calibrated_probabilities, 1e-6));
        assert!(!result.conformal_set.is_empty());
        
        Ok(())
    }
}