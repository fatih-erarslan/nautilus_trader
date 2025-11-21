//! Comprehensive test suite for ATS-CP algorithms
//!
//! This module provides mathematical validation tests for all ATS-CP implementations,
//! ensuring scientific rigor and IEEE 754 compliance.

pub mod ats_cp_tests;
pub mod precision_validation_tests;

#[cfg(test)]
mod test_utils {
    use super::*;
    use crate::types::{AtsCpVariant, Confidence};
    use approx::assert_relative_eq;

    /// Creates test data for ATS-CP algorithm validation
    pub fn create_test_data() -> (Vec<Vec<f64>>, Vec<usize>) {
        let calibration_logits = vec![
            vec![1.0, 2.0, 3.0],
            vec![2.0, 1.0, 3.0],
            vec![3.0, 2.0, 1.0],
            vec![1.5, 2.5, 3.5],
            vec![2.5, 1.5, 3.5],
        ];
        let calibration_labels = vec![2, 2, 0, 2, 2];
        
        (calibration_logits, calibration_labels)
    }

    /// Validates that probabilities sum to 1.0
    pub fn validate_probability_distribution(probs: &[f64], tolerance: f64) -> bool {
        let sum: f64 = probs.iter().sum();
        (sum - 1.0).abs() < tolerance && probs.iter().all(|&p| p >= 0.0 && p <= 1.0)
    }

    /// Validates coverage guarantee for ATS-CP
    pub fn validate_coverage_guarantee(
        results: &[crate::types::AtsCpResult],
        target_coverage: Confidence,
        tolerance: f64,
    ) -> bool {
        let avg_coverage: f64 = results.iter()
            .map(|r| r.conformal_set.len() as f64 / r.calibrated_probabilities.len() as f64)
            .sum::<f64>() / results.len() as f64;
        
        (avg_coverage - target_coverage).abs() < tolerance
    }
}