//! Comprehensive Mathematical Validation Test Suite
//! 
//! This test suite validates all mathematical operations in the autopoiesis
//! system for correctness, numerical stability, and performance.

use autopoiesis::validation::{MathematicalValidator, ValidationConfig};
use autopoiesis::validation::mathematical_validator::{ValidationUtils, NumericalConstraints};
use autopoiesis::utils::MathUtils;

#[tokio::test]
async fn test_comprehensive_mathematical_validation() {
    let config = ValidationConfig {
        tolerance: 1e-12,
        test_sample_sizes: vec![10, 100, 1000],
        enable_property_tests: true,
        enable_reference_comparisons: true,
        enable_performance_benchmarks: true,
        enable_regression_detection: false, // Disable for test
        random_seed: 42,
    };
    
    let validator = MathematicalValidator::new(config).unwrap();
    let report = validator.validate_all_mathematics().await.unwrap();
    
    // Check overall validation status
    println!("Validation Status: {:?}", report.overall_status);
    println!("Timestamp: {}", report.timestamp);
    
    // Verify algorithm validations
    assert!(!report.algorithm_validation.statistical_functions.is_empty());
    assert!(!report.algorithm_validation.mathematical_utilities.is_empty());
    
    // Check for critical failures
    let mut critical_failures = 0;
    for (name, result) in &report.algorithm_validation.statistical_functions {
        if matches!(result.status, autopoiesis::validation::ValidationStatus::Critical) {
            critical_failures += 1;
            println!("CRITICAL FAILURE in {}: {}", name, result.details);
        }
    }
    
    for (name, result) in &report.algorithm_validation.ml_algorithms {
        if matches!(result.status, autopoiesis::validation::ValidationStatus::Critical) {
            critical_failures += 1;
            println!("CRITICAL FAILURE in {}: {}", name, result.details);
        }
    }
    
    for (name, result) in &report.algorithm_validation.financial_metrics {
        if matches!(result.status, autopoiesis::validation::ValidationStatus::Critical) {
            critical_failures += 1;
            println!("CRITICAL FAILURE in {}: {}", name, result.details);
        }
    }
    
    // Print recommendations
    if !report.recommendations.is_empty() {
        println!("\nValidation Recommendations:");
        for rec in &report.recommendations {
            println!("- [{:?}] {}: {}", rec.priority, rec.title, rec.description);
        }
    }
    
    // The test passes even with warnings, but fails on critical issues
    assert!(critical_failures == 0, "Found {} critical validation failures", critical_failures);
}

#[tokio::test]
async fn test_ema_mathematical_properties() {
    let validator = MathematicalValidator::default().unwrap();
    let result = validator.validate_algorithm("ema").await.unwrap();
    
    println!("EMA Validation: {} - {}", result.name, result.result.details);
    println!("Accuracy Score: {:.2}%", result.result.accuracy_score * 100.0);
    println!("Tests Passed: {}/{}", result.result.test_cases_passed, result.result.test_cases_total);
    
    // EMA should pass most tests
    assert!(result.result.accuracy_score > 0.8);
    assert!(result.result.test_cases_passed > 0);
}

#[tokio::test]
async fn test_std_dev_mathematical_properties() {
    let validator = MathematicalValidator::default().unwrap();
    let result = validator.validate_algorithm("std_dev").await.unwrap();
    
    println!("Std Dev Validation: {} - {}", result.name, result.result.details);
    println!("Max Error: {:.2e}", result.result.error_metrics.max_error);
    println!("RMSE: {:.2e}", result.result.error_metrics.rmse);
    
    // Standard deviation should be highly accurate
    assert!(result.result.accuracy_score > 0.9);
    assert!(result.result.error_metrics.max_error < 1e-10);
}

#[tokio::test]
async fn test_correlation_mathematical_properties() {
    let validator = MathematicalValidator::default().unwrap();
    let result = validator.validate_algorithm("correlation").await.unwrap();
    
    println!("Correlation Validation: {} - {}", result.name, result.result.details);
    
    // Correlation should handle various edge cases correctly
    assert!(result.result.accuracy_score > 0.7); // Allow some tolerance for edge cases
}

#[test]
fn test_mathematical_property_validation() {
    // Test EMA properties directly
    let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let alpha = 0.3;
    let ema_result = MathUtils::ema(&values, alpha);
    
    // Property 1: EMA result should have same length as input
    assert_eq!(ema_result.len(), values.len());
    
    // Property 2: First EMA value should equal first input value
    assert!((ema_result[0] - values[0]).abs() < f64::EPSILON);
    
    // Property 3: EMA values should be bounded by input range
    let min_val = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_val = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    
    for &ema_val in &ema_result {
        assert!(ema_val >= min_val && ema_val <= max_val,
                "EMA value {} not in range [{}, {}]", ema_val, min_val, max_val);
    }
    
    // Property 4: All values should be finite
    assert!(ema_result.iter().all(|x| x.is_finite()));
}

#[test]
fn test_correlation_properties() {
    // Test perfect positive correlation
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
    let corr = MathUtils::correlation(&x, &y);
    
    assert!((corr - 1.0).abs() < 1e-10, "Perfect positive correlation should be 1.0, got {}", corr);
    
    // Test perfect negative correlation
    let y_neg = vec![10.0, 8.0, 6.0, 4.0, 2.0];
    let corr_neg = MathUtils::correlation(&x, &y_neg);
    
    assert!((corr_neg + 1.0).abs() < 1e-10, "Perfect negative correlation should be -1.0, got {}", corr_neg);
    
    // Test symmetry
    let corr_xy = MathUtils::correlation(&x, &y);
    let corr_yx = MathUtils::correlation(&y, &x);
    
    assert!((corr_xy - corr_yx).abs() < f64::EPSILON, "Correlation should be symmetric");
    
    // Test bounds
    assert!(corr >= -1.0 && corr <= 1.0, "Correlation should be in [-1, 1], got {}", corr);
}

#[test]
fn test_std_dev_properties() {
    // Test non-negativity
    let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let std_dev = MathUtils::std_dev(&values);
    
    assert!(std_dev >= 0.0, "Standard deviation should be non-negative, got {}", std_dev);
    assert!(std_dev.is_finite(), "Standard deviation should be finite");
    
    // Test zero variance for constant values
    let constant_values = vec![5.0, 5.0, 5.0, 5.0];
    let constant_std = MathUtils::std_dev(&constant_values);
    
    assert!(constant_std.abs() < f64::EPSILON, "Standard deviation of constant values should be zero, got {}", constant_std);
    
    // Test scaling property: std_dev(k*X) = |k| * std_dev(X)
    let scale_factor = 3.0;
    let scaled_values: Vec<f64> = values.iter().map(|&x| scale_factor * x).collect();
    let scaled_std = MathUtils::std_dev(&scaled_values);
    let expected_scaled_std = scale_factor.abs() * std_dev;
    
    assert!((scaled_std - expected_scaled_std).abs() < 1e-10, 
            "Scaling property violated: got {}, expected {}", scaled_std, expected_scaled_std);
}

#[test]
fn test_percentile_properties() {
    let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    
    // Test that percentile is within data range
    let percentile_50 = MathUtils::percentile(&values, 0.5);
    let min_val = *values.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let max_val = *values.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    
    assert!(percentile_50 >= min_val && percentile_50 <= max_val,
            "Percentile should be within data range");
    
    // Test monotonicity
    let percentile_25 = MathUtils::percentile(&values, 0.25);
    let percentile_75 = MathUtils::percentile(&values, 0.75);
    
    assert!(percentile_25 <= percentile_50, "Percentiles should be monotonic");
    assert!(percentile_50 <= percentile_75, "Percentiles should be monotonic");
    
    // Test boundary cases
    let percentile_0 = MathUtils::percentile(&values, 0.0);
    let percentile_100 = MathUtils::percentile(&values, 1.0);
    
    assert!((percentile_0 - min_val).abs() < f64::EPSILON, "0th percentile should be minimum");
    assert!((percentile_100 - max_val).abs() < f64::EPSILON, "100th percentile should be maximum");
}

#[test]
fn test_z_score_properties() {
    let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let z_scores = MathUtils::z_score(&values);
    
    // Test length preservation
    assert_eq!(z_scores.len(), values.len());
    
    // Test mean approximately zero (for non-constant data)
    let z_mean = z_scores.iter().sum::<f64>() / z_scores.len() as f64;
    assert!(z_mean.abs() < 1e-10, "Z-score mean should be approximately zero, got {}", z_mean);
    
    // Test standard deviation approximately 1
    let z_std = MathUtils::std_dev(&z_scores);
    assert!((z_std - 1.0).abs() < 1e-10, "Z-score standard deviation should be 1, got {}", z_std);
    
    // Test that all values are finite
    assert!(z_scores.iter().all(|x| x.is_finite()), "All Z-scores should be finite");
}

#[test]
fn test_numerical_stability() {
    // Test with very large numbers
    let large_values = vec![1e15, 2e15, 3e15];
    let large_std = MathUtils::std_dev(&large_values);
    assert!(large_std.is_finite(), "Standard deviation should handle large numbers");
    
    // Test with very small numbers
    let small_values = vec![1e-15, 2e-15, 3e-15];
    let small_std = MathUtils::std_dev(&small_values);
    assert!(small_std.is_finite(), "Standard deviation should handle small numbers");
    assert!(small_std >= 0.0, "Standard deviation should be non-negative for small numbers");
    
    // Test mixed scales that could cause precision loss
    let mixed_values = vec![1e15, 1.0, 1e15];
    let mixed_std = MathUtils::std_dev(&mixed_values);
    assert!(mixed_std.is_finite(), "Standard deviation should handle mixed scales");
}

#[test]
fn test_edge_cases() {
    // Test empty arrays
    let empty_vec: Vec<f64> = vec![];
    let empty_std = MathUtils::std_dev(&empty_vec);
    assert_eq!(empty_std, 0.0, "Standard deviation of empty array should be 0");
    
    // Test single element
    let single_vec = vec![42.0];
    let single_std = MathUtils::std_dev(&single_vec);
    assert_eq!(single_std, 0.0, "Standard deviation of single element should be 0");
    
    // Test with NaN values
    let nan_vec = vec![1.0, f64::NAN, 3.0];
    let nan_std = MathUtils::std_dev(&nan_vec);
    // Implementation should handle NaN gracefully (behavior may vary)
    
    // Test with infinite values
    let inf_vec = vec![1.0, f64::INFINITY, 3.0];
    let inf_std = MathUtils::std_dev(&inf_vec);
    // Implementation should handle infinity gracefully
}

#[test]
fn test_validation_utilities() {
    // Test numerical constraints
    let constraints = NumericalConstraints {
        min_value: Some(-10.0),
        max_value: Some(10.0),
        allow_infinite: false,
        allow_nan: false,
    };
    
    assert!(ValidationUtils::check_numerical_constraints(5.0, &constraints));
    assert!(!ValidationUtils::check_numerical_constraints(15.0, &constraints));
    assert!(!ValidationUtils::check_numerical_constraints(-15.0, &constraints));
    assert!(!ValidationUtils::check_numerical_constraints(f64::INFINITY, &constraints));
    assert!(!ValidationUtils::check_numerical_constraints(f64::NAN, &constraints));
    
    // Test relative error calculation
    assert_eq!(ValidationUtils::relative_error(1.1, 1.0), 0.1);
    assert_eq!(ValidationUtils::relative_error(0.9, 1.0), 0.1);
    assert_eq!(ValidationUtils::relative_error(5.0, 0.0), 5.0);
    
    // Test approximate equality
    assert!(ValidationUtils::approximately_equal(1.0, 1.0000001, 1e-6));
    assert!(!ValidationUtils::approximately_equal(1.0, 1.1, 1e-6));
}

#[tokio::test]
async fn test_performance_benchmarks() {
    let config = ValidationConfig {
        tolerance: 1e-10,
        enable_performance_benchmarks: true,
        random_seed: 42,
        ..Default::default()
    };
    
    let mut validator = MathematicalValidator::new(config).unwrap();
    let perf_report = validator.benchmark_validation_performance().await.unwrap();
    
    println!("Validation Performance Report:");
    println!("Total validation time: {:?}", perf_report.total_validation_time);
    println!("Reference computation time: {:?}", perf_report.reference_computation_time);
    println!("Stability testing time: {:?}", perf_report.stability_testing_time);
    println!("Validation overhead: {:.2}x", perf_report.validation_overhead);
    
    // Validation should complete in reasonable time
    assert!(perf_report.total_validation_time.as_secs() < 60, "Validation taking too long");
    
    // Overhead should be reasonable
    assert!(perf_report.validation_overhead < 1000.0, "Validation overhead too high");
}

#[tokio::test]
async fn test_quick_validation_for_ci() {
    let validator = MathematicalValidator::default().unwrap();
    let ci_report = validator.generate_ci_report().await.unwrap();
    
    println!("CI Report: {}", ci_report.message);
    println!("Exit Code: {}", ci_report.exit_code);
    println!("Total Tests: {}", ci_report.summary.total_tests);
    println!("Critical Issues: {}", ci_report.summary.critical_issues);
    println!("Warnings: {}", ci_report.summary.warnings);
    
    // Quick validation should complete successfully
    assert!(ci_report.summary.total_tests > 0);
    
    // For CI, we'll allow warnings but not critical failures
    if ci_report.exit_code == 2 {
        panic!("Critical validation failures detected - CI should fail");
    }
}

#[test]
fn test_mathematical_invariants() {
    // Test that mathematical relationships hold
    
    // Variance relationship: Var(X) = E[X²] - (E[X])²
    let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance_formula1 = values.iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f64>() / (values.len() - 1) as f64;
    
    let std_dev = MathUtils::std_dev(&values);
    let variance_formula2 = std_dev.powi(2);
    
    assert!((variance_formula1 - variance_formula2).abs() < 1e-10,
            "Variance should equal standard deviation squared");
    
    // Correlation with self should be 1 (unless constant)
    let self_corr = MathUtils::correlation(&values, &values);
    assert!((self_corr - 1.0).abs() < 1e-10, 
            "Self-correlation should be 1.0, got {}", self_corr);
    
    // Test Cauchy-Schwarz inequality via correlation
    // |corr(X,Y)| <= 1
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![5.0, 1.0, 3.0, 2.0, 4.0];
    let corr = MathUtils::correlation(&x, &y);
    
    assert!(corr.abs() <= 1.0, "Correlation magnitude should not exceed 1");
}

/// Integration test that runs the full validation suite
#[tokio::test]
async fn integration_test_full_validation_suite() {
    println!("Running comprehensive mathematical validation suite...");
    
    let config = ValidationConfig {
        tolerance: 1e-12,
        test_sample_sizes: vec![100, 1000],
        enable_property_tests: true,
        enable_reference_comparisons: true,
        enable_performance_benchmarks: true,
        enable_regression_detection: false,
        random_seed: 42,
    };
    
    let validator = MathematicalValidator::new(config).unwrap();
    
    // Run comprehensive validation
    let start_time = std::time::Instant::now();
    let report = validator.validate_all_mathematics().await.unwrap();
    let validation_time = start_time.elapsed();
    
    println!("Validation completed in {:?}", validation_time);
    println!("Overall status: {:?}", report.overall_status);
    
    // Print summary of results
    let mut total_tests = 0;
    let mut passed_tests = 0;
    let mut critical_failures = Vec::new();
    
    for (name, result) in report.algorithm_validation.statistical_functions.iter()
        .chain(report.algorithm_validation.ml_algorithms.iter())
        .chain(report.algorithm_validation.financial_metrics.iter())
        .chain(report.algorithm_validation.mathematical_utilities.iter()) {
        
        total_tests += result.test_cases_total;
        passed_tests += result.test_cases_passed;
        
        if matches!(result.status, autopoiesis::validation::ValidationStatus::Critical) {
            critical_failures.push(format!("{}: {}", name, result.details));
        }
        
        println!("{}: {:.1}% accuracy ({}/{})", 
                name, 
                result.accuracy_score * 100.0,
                result.test_cases_passed, 
                result.test_cases_total);
    }
    
    println!("\nSummary:");
    println!("Total tests: {}", total_tests);
    println!("Passed tests: {}", passed_tests);
    println!("Success rate: {:.1}%", (passed_tests as f64 / total_tests as f64) * 100.0);
    
    if !critical_failures.is_empty() {
        println!("\nCritical failures:");
        for failure in &critical_failures {
            println!("- {}", failure);
        }
    }
    
    // Print recommendations
    if !report.recommendations.is_empty() {
        println!("\nRecommendations:");
        for rec in &report.recommendations {
            match rec.priority {
                autopoiesis::validation::RecommendationPriority::Critical => {
                    println!("🔴 CRITICAL: {}", rec.title);
                },
                autopoiesis::validation::RecommendationPriority::High => {
                    println!("🟡 HIGH: {}", rec.title);
                },
                _ => {
                    println!("🟢 {}: {}", rec.priority.clone(), rec.title);
                }
            }
            println!("   {}", rec.description);
        }
    }
    
    // Validation passes if no critical failures
    assert!(critical_failures.is_empty(), 
            "Critical validation failures detected: {:?}", critical_failures);
    
    // Overall success rate should be reasonable
    let success_rate = passed_tests as f64 / total_tests as f64;
    assert!(success_rate > 0.8, 
            "Overall success rate too low: {:.1}%", success_rate * 100.0);
}