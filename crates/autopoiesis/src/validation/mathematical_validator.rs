//! Main Mathematical Validator Integration
//! 
//! This module provides the main entry point for mathematical validation
//! and integrates all validation components.

use crate::Result;
use crate::validation::{
    MathematicalValidator, MathematicalValidationReport, ValidationConfig,
    reference_implementations::ReferenceImplementations,
    numerical_stability::NumericalStabilityTester,
    benchmark_suite::BenchmarkSuite,
    property_tests::PropertyTester,
    regression_detection::RegressionDetector,
};

/// Main validator that orchestrates all validation activities
impl MathematicalValidator {
    /// Create a new validator with custom configuration
    pub fn with_config(config: ValidationConfig) -> Result<Self> {
        Self::new(config)
    }

    /// Create a validator with default configuration
    pub fn default() -> Result<Self> {
        Self::new(ValidationConfig::default())
    }

    /// Quick validation for development use
    pub async fn quick_validate(&self) -> Result<ValidationSummary> {
        // Run a subset of tests for faster feedback
        let config = ValidationConfig {
            tolerance: 1e-8,
            test_sample_sizes: vec![100, 1000],
            enable_property_tests: true,
            enable_reference_comparisons: false,
            enable_performance_benchmarks: false,
            enable_regression_detection: false,
            random_seed: 42,
        };

        let mut quick_validator = Self::new(config)?;
        let algorithm_validation = quick_validator.validate_algorithms().await?;
        
        // Count critical issues
        let mut critical_issues = 0;
        let mut warnings = 0;
        
        for result in algorithm_validation.statistical_functions.values()
            .chain(algorithm_validation.ml_algorithms.values())
            .chain(algorithm_validation.financial_metrics.values())
            .chain(algorithm_validation.mathematical_utilities.values()) {
            
            match result.status {
                crate::validation::ValidationStatus::Critical => critical_issues += 1,
                crate::validation::ValidationStatus::Fail => critical_issues += 1,
                crate::validation::ValidationStatus::Warning => warnings += 1,
                crate::validation::ValidationStatus::Pass => {}
            }
        }

        Ok(ValidationSummary {
            total_tests: algorithm_validation.statistical_functions.len() + 
                        algorithm_validation.ml_algorithms.len() + 
                        algorithm_validation.financial_metrics.len() + 
                        algorithm_validation.mathematical_utilities.len(),
            critical_issues,
            warnings,
            overall_status: if critical_issues > 0 {
                ValidationResult::Failed
            } else if warnings > 2 {
                ValidationResult::Warning
            } else {
                ValidationResult::Passed
            },
        })
    }

    /// Comprehensive validation with full reporting
    pub async fn comprehensive_validate(&self) -> Result<MathematicalValidationReport> {
        self.validate_all_mathematics().await
    }

    /// Validate specific algorithm by name
    pub async fn validate_algorithm(&self, algorithm_name: &str) -> Result<AlgorithmValidationResult> {
        match algorithm_name {
            "ema" => Ok(AlgorithmValidationResult {
                name: "Exponential Moving Average".to_string(),
                result: self.validate_ema().await?,
                recommendations: self.get_ema_recommendations(),
            }),
            "std_dev" => Ok(AlgorithmValidationResult {
                name: "Standard Deviation".to_string(),
                result: self.validate_std_dev().await?,
                recommendations: self.get_std_dev_recommendations(),
            }),
            "correlation" => Ok(AlgorithmValidationResult {
                name: "Correlation".to_string(),
                result: self.validate_correlation().await?,
                recommendations: self.get_correlation_recommendations(),
            }),
            _ => Err(crate::error::Error::InvalidInput(
                format!("Unknown algorithm: {}", algorithm_name)
            )),
        }
    }

    /// Generate validation report for CI/CD integration
    pub async fn generate_ci_report(&self) -> Result<CiValidationReport> {
        let summary = self.quick_validate().await?;
        
        Ok(CiValidationReport {
            summary,
            exit_code: match summary.overall_status {
                ValidationResult::Passed => 0,
                ValidationResult::Warning => 1,
                ValidationResult::Failed => 2,
            },
            message: match summary.overall_status {
                ValidationResult::Passed => "All mathematical validations passed".to_string(),
                ValidationResult::Warning => format!("Validation passed with {} warnings", summary.warnings),
                ValidationResult::Failed => format!("Validation failed with {} critical issues", summary.critical_issues),
            },
        })
    }

    // Helper methods for specific algorithm recommendations
    fn get_ema_recommendations(&self) -> Vec<String> {
        vec![
            "Consider using Kahan summation for improved numerical stability".to_string(),
            "Add input validation for alpha parameter range [0,1]".to_string(),
            "Implement overflow protection for extreme input values".to_string(),
        ]
    }

    fn get_std_dev_recommendations(&self) -> Vec<String> {
        vec![
            "Use Welford's online algorithm for better numerical stability".to_string(),
            "Add protection against negative variance due to floating-point errors".to_string(),
            "Consider using higher precision for intermediate calculations".to_string(),
        ]
    }

    fn get_correlation_recommendations(&self) -> Vec<String> {
        vec![
            "Implement proper handling of zero variance cases".to_string(),
            "Add numerical stability checks for near-singular cases".to_string(),
            "Consider using Spearman correlation for non-linear relationships".to_string(),
        ]
    }

    /// Run validation benchmark for performance comparison
    pub async fn benchmark_validation_performance(&mut self) -> Result<ValidationPerformanceReport> {
        let start_time = std::time::Instant::now();
        
        // Run quick validation to measure performance
        let _summary = self.quick_validate().await?;
        let validation_time = start_time.elapsed();
        
        // Benchmark individual components
        let reference_time = {
            let start = std::time::Instant::now();
            let _ema_ref = self.reference_implementations.compute_ema_reference(&vec![1.0, 2.0, 3.0], 0.3)?;
            start.elapsed()
        };
        
        let stability_time = {
            let start = std::time::Instant::now();
            let _stability = self.stability_tester.run_all_stability_tests().await?;
            start.elapsed()
        };

        Ok(ValidationPerformanceReport {
            total_validation_time: validation_time,
            reference_computation_time: reference_time,
            stability_testing_time: stability_time,
            validation_overhead: validation_time.as_secs_f64() / reference_time.as_secs_f64(),
        })
    }
}

/// Summary of validation results
#[derive(Debug, Clone)]
pub struct ValidationSummary {
    pub total_tests: usize,
    pub critical_issues: usize,
    pub warnings: usize,
    pub overall_status: ValidationResult,
}

#[derive(Debug, Clone)]
pub enum ValidationResult {
    Passed,
    Warning,
    Failed,
}

/// Individual algorithm validation result
#[derive(Debug, Clone)]
pub struct AlgorithmValidationResult {
    pub name: String,
    pub result: crate::validation::AlgorithmResult,
    pub recommendations: Vec<String>,
}

/// CI/CD integration report
#[derive(Debug, Clone)]
pub struct CiValidationReport {
    pub summary: ValidationSummary,
    pub exit_code: i32,
    pub message: String,
}

/// Performance report for validation framework itself
#[derive(Debug, Clone)]
pub struct ValidationPerformanceReport {
    pub total_validation_time: std::time::Duration,
    pub reference_computation_time: std::time::Duration,
    pub stability_testing_time: std::time::Duration,
    pub validation_overhead: f64,
}

/// Validation utilities for specific use cases
pub struct ValidationUtils;

impl ValidationUtils {
    /// Validate a single mathematical operation
    pub fn validate_operation<F, T>(
        operation: F,
        inputs: Vec<T>,
        expected_properties: Vec<Box<dyn Fn(&T) -> bool>>,
    ) -> bool
    where
        F: Fn(&T) -> T,
        T: Clone,
    {
        for input in inputs {
            let result = operation(&input);
            for property in &expected_properties {
                if !property(&result) {
                    return false;
                }
            }
        }
        true
    }

    /// Check if a value satisfies numerical constraints
    pub fn check_numerical_constraints(value: f64, constraints: &NumericalConstraints) -> bool {
        if !value.is_finite() && !constraints.allow_infinite {
            return false;
        }
        
        if value.is_nan() && !constraints.allow_nan {
            return false;
        }
        
        if let Some(min) = constraints.min_value {
            if value < min {
                return false;
            }
        }
        
        if let Some(max) = constraints.max_value {
            if value > max {
                return false;
            }
        }
        
        true
    }

    /// Calculate relative error between two values
    pub fn relative_error(actual: f64, expected: f64) -> f64 {
        if expected == 0.0 {
            actual.abs()
        } else {
            (actual - expected).abs() / expected.abs()
        }
    }

    /// Check if two floating-point numbers are approximately equal
    pub fn approximately_equal(a: f64, b: f64, tolerance: f64) -> bool {
        (a - b).abs() <= tolerance.max(tolerance * a.abs().max(b.abs()))
    }
}

/// Numerical constraints for validation
#[derive(Debug, Clone)]
pub struct NumericalConstraints {
    pub min_value: Option<f64>,
    pub max_value: Option<f64>,
    pub allow_infinite: bool,
    pub allow_nan: bool,
}

impl Default for NumericalConstraints {
    fn default() -> Self {
        Self {
            min_value: None,
            max_value: None,
            allow_infinite: false,
            allow_nan: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_quick_validate() {
        let validator = MathematicalValidator::default().unwrap();
        let summary = validator.quick_validate().await.unwrap();
        
        assert!(summary.total_tests > 0);
        // Don't assert on specific outcomes as they depend on implementation quality
    }

    #[tokio::test]
    async fn test_algorithm_validation() {
        let validator = MathematicalValidator::default().unwrap();
        
        let ema_result = validator.validate_algorithm("ema").await.unwrap();
        assert_eq!(ema_result.name, "Exponential Moving Average");
        assert!(!ema_result.recommendations.is_empty());
        
        // Test invalid algorithm
        let invalid_result = validator.validate_algorithm("invalid").await;
        assert!(invalid_result.is_err());
    }

    #[tokio::test]
    async fn test_ci_report_generation() {
        let validator = MathematicalValidator::default().unwrap();
        let ci_report = validator.generate_ci_report().await.unwrap();
        
        assert!(ci_report.exit_code >= 0 && ci_report.exit_code <= 2);
        assert!(!ci_report.message.is_empty());
    }

    #[test]
    fn test_validation_utils() {
        // Test relative error calculation
        assert_eq!(ValidationUtils::relative_error(1.1, 1.0), 0.1);
        assert_eq!(ValidationUtils::relative_error(5.0, 0.0), 5.0);
        
        // Test approximate equality
        assert!(ValidationUtils::approximately_equal(1.0, 1.0000001, 1e-6));
        assert!(!ValidationUtils::approximately_equal(1.0, 1.1, 1e-6));
        
        // Test numerical constraints
        let constraints = NumericalConstraints {
            min_value: Some(0.0),
            max_value: Some(100.0),
            allow_infinite: false,
            allow_nan: false,
        };
        
        assert!(ValidationUtils::check_numerical_constraints(50.0, &constraints));
        assert!(!ValidationUtils::check_numerical_constraints(-1.0, &constraints));
        assert!(!ValidationUtils::check_numerical_constraints(f64::INFINITY, &constraints));
        assert!(!ValidationUtils::check_numerical_constraints(f64::NAN, &constraints));
    }

    #[test]
    fn test_operation_validation() {
        let square = |x: &f64| x * x;
        let inputs = vec![1.0, 2.0, 3.0, -1.0, -2.0];
        let non_negative = Box::new(|x: &f64| *x >= 0.0);
        let finite = Box::new(|x: &f64| x.is_finite());
        
        let properties = vec![non_negative, finite];
        
        let is_valid = ValidationUtils::validate_operation(square, inputs, properties);
        assert!(is_valid);
        
        // Test with a property that fails
        let always_positive = Box::new(|x: &f64| *x > 0.0);
        let properties_fail = vec![always_positive];
        let inputs_with_zero = vec![0.0, 1.0, 2.0];
        
        let is_valid = ValidationUtils::validate_operation(square, inputs_with_zero, properties_fail);
        assert!(!is_valid); // Should fail because square(0) = 0, which is not > 0
    }
}