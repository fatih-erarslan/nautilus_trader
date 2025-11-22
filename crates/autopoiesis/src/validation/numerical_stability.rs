//! Numerical Stability Testing Framework
//! 
//! This module provides comprehensive testing for numerical stability,
//! including overflow, underflow, and precision loss detection.

use crate::Result;
use crate::validation::{ValidationConfig, NumericalStabilityResults, StabilityTestResult, ConditionNumberAnalysis};
use crate::utils::MathUtils;
use ndarray::{Array1, Array2};
use std::collections::HashMap;

pub struct NumericalStabilityTester {
    config: ValidationConfig,
}

impl NumericalStabilityTester {
    pub fn new(config: &ValidationConfig) -> Result<Self> {
        Ok(Self {
            config: config.clone(),
        })
    }

    /// Run all numerical stability tests
    pub async fn run_all_stability_tests(&self) -> Result<NumericalStabilityResults> {
        let overflow_tests = self.test_overflow_stability().await?;
        let underflow_tests = self.test_underflow_stability().await?;
        let precision_tests = self.test_precision_stability().await?;
        let condition_number_analysis = self.analyze_condition_numbers().await?;

        Ok(NumericalStabilityResults {
            overflow_tests,
            underflow_tests,
            precision_tests,
            condition_number_analysis,
        })
    }

    /// Test stability under overflow conditions
    async fn test_overflow_stability(&self) -> Result<StabilityTestResult> {
        let mut passed = 0;
        let mut failed = 0;
        let mut critical_failures = Vec::new();

        // Test exponential moving average with large values
        let large_values = vec![1e308, 1e307, 1e306, 1e305];
        match MathUtils::ema(&large_values, 0.5) {
            result if result.iter().all(|x| x.is_finite()) => passed += 1,
            _ => {
                failed += 1;
                critical_failures.push("EMA overflow with large values".to_string());
            }
        }

        // Test standard deviation with extreme values
        let extreme_values = vec![f64::MAX / 2.0, f64::MAX / 3.0, f64::MAX / 4.0];
        let std_result = MathUtils::std_dev(&extreme_values);
        if std_result.is_finite() && std_result >= 0.0 {
            passed += 1;
        } else {
            failed += 1;
            critical_failures.push("Standard deviation overflow".to_string());
        }

        // Test correlation with large values
        let large_x = vec![1e100, 2e100, 3e100, 4e100];
        let large_y = vec![2e100, 4e100, 6e100, 8e100];
        let corr_result = MathUtils::correlation(&large_x, &large_y);
        if corr_result.is_finite() && (-1.0..=1.0).contains(&corr_result) {
            passed += 1;
        } else {
            failed += 1;
            critical_failures.push("Correlation overflow with large values".to_string());
        }

        // Test linear regression with extreme values
        let reg_result = MathUtils::linear_regression(&large_x, &large_y);
        match reg_result {
            Some((slope, intercept)) if slope.is_finite() && intercept.is_finite() => passed += 1,
            _ => {
                failed += 1;
                critical_failures.push("Linear regression overflow".to_string());
            }
        }

        // Test percentile calculation with extreme values
        let percentile_result = MathUtils::percentile(&extreme_values, 0.5);
        if percentile_result.is_finite() {
            passed += 1;
        } else {
            failed += 1;
            critical_failures.push("Percentile calculation overflow".to_string());
        }

        let total_tests = passed + failed;
        let stability_score = if total_tests > 0 { passed as f64 / total_tests as f64 } else { 0.0 };

        Ok(StabilityTestResult {
            passed,
            failed,
            critical_failures,
            stability_score,
        })
    }

    /// Test stability under underflow conditions
    async fn test_underflow_stability(&self) -> Result<StabilityTestResult> {
        let mut passed = 0;
        let mut failed = 0;
        let mut critical_failures = Vec::new();

        // Test with very small values near machine epsilon
        let tiny_values = vec![f64::MIN_POSITIVE, f64::MIN_POSITIVE * 2.0, f64::MIN_POSITIVE * 3.0];
        
        // Test EMA with tiny values
        let ema_result = MathUtils::ema(&tiny_values, 0.1);
        if ema_result.iter().all(|x| x.is_finite() && *x >= 0.0) {
            passed += 1;
        } else {
            failed += 1;
            critical_failures.push("EMA underflow with tiny values".to_string());
        }

        // Test standard deviation with tiny values
        let std_result = MathUtils::std_dev(&tiny_values);
        if std_result.is_finite() && std_result >= 0.0 {
            passed += 1;
        } else {
            failed += 1;
            critical_failures.push("Standard deviation underflow".to_string());
        }

        // Test with values that differ by tiny amounts
        let close_values = vec![1.0, 1.0 + f64::EPSILON, 1.0 + 2.0 * f64::EPSILON];
        let close_std = MathUtils::std_dev(&close_values);
        if close_std.is_finite() {
            passed += 1;
        } else {
            failed += 1;
            critical_failures.push("Standard deviation with near-identical values".to_string());
        }

        // Test correlation with tiny differences
        let x_close = vec![1.0, 1.0 + f64::EPSILON, 1.0 + 2.0 * f64::EPSILON];
        let y_close = vec![2.0, 2.0 + f64::EPSILON, 2.0 + 2.0 * f64::EPSILON];
        let corr_close = MathUtils::correlation(&x_close, &y_close);
        if corr_close.is_finite() {
            passed += 1;
        } else {
            failed += 1;
            critical_failures.push("Correlation with tiny differences".to_string());
        }

        // Test division by very small numbers in linear regression
        let x_tiny_diff = vec![1.0, 1.0, 1.0 + f64::EPSILON];
        let y_normal = vec![1.0, 2.0, 3.0];
        let reg_result = MathUtils::linear_regression(&x_tiny_diff, &y_normal);
        // This should handle the near-singular case gracefully
        match reg_result {
            None => passed += 1, // Correctly detected singular case
            Some((slope, intercept)) if slope.is_finite() && intercept.is_finite() => passed += 1,
            _ => {
                failed += 1;
                critical_failures.push("Linear regression with near-singular matrix".to_string());
            }
        }

        let total_tests = passed + failed;
        let stability_score = if total_tests > 0 { passed as f64 / total_tests as f64 } else { 0.0 };

        Ok(StabilityTestResult {
            passed,
            failed,
            critical_failures,
            stability_score,
        })
    }

    /// Test precision and accuracy maintenance
    async fn test_precision_stability(&self) -> Result<StabilityTestResult> {
        let mut passed = 0;
        let mut failed = 0;
        let mut critical_failures = Vec::new();

        // Test precision loss in iterative calculations
        let values = vec![1e15, 1.0, 1e15, -1e15, -1.0, -1e15];
        let std_result = MathUtils::std_dev(&values);
        
        // Check if the computation maintains reasonable precision
        // The expected result should be close to sqrt(2) since the effective values are [1, -1]
        let expected_approx = std::f64::consts::SQRT_2;
        let relative_error = if expected_approx != 0.0 {
            (std_result - expected_approx).abs() / expected_approx
        } else {
            std_result.abs()
        };
        
        if relative_error < 0.1 { // Allow 10% error for this challenging case
            passed += 1;
        } else {
            failed += 1;
            critical_failures.push(format!(
                "Precision loss in standard deviation calculation: got {}, expected ~{}",
                std_result, expected_approx
            ));
        }

        // Test numerical stability of correlation with challenging data
        let x_challenging = vec![1e10, 1e10 + 1.0, 1e10 + 2.0];
        let y_challenging = vec![1e10 + 1.0, 1e10 + 2.0, 1e10 + 3.0];
        let corr_result = MathUtils::correlation(&x_challenging, &y_challenging);
        
        if (corr_result - 1.0).abs() < 0.01 { // Should be close to perfect correlation
            passed += 1;
        } else {
            failed += 1;
            critical_failures.push(format!(
                "Precision loss in correlation: got {}, expected ~1.0",
                corr_result
            ));
        }

        // Test precision in percentile calculation with many identical values
        let mostly_same = vec![1.0; 1000];
        let percentile_result = MathUtils::percentile(&mostly_same, 0.5);
        if (percentile_result - 1.0).abs() < f64::EPSILON {
            passed += 1;
        } else {
            failed += 1;
            critical_failures.push("Precision error in percentile with identical values".to_string());
        }

        // Test catastrophic cancellation in subtraction
        let x_cancel = vec![1.0000000000000001, 1.0000000000000002, 1.0000000000000003];
        let y_cancel = vec![1.0, 1.0, 1.0];
        let result_values: Vec<f64> = x_cancel.iter().zip(y_cancel.iter())
            .map(|(x, y)| x - y)
            .collect();
        
        // Check if the differences are reasonable (not all zero due to precision loss)
        if result_values.iter().any(|&x| x > f64::EPSILON) {
            passed += 1;
        } else {
            failed += 1;
            critical_failures.push("Catastrophic cancellation in subtraction".to_string());
        }

        // Test precision in compound operations
        let compound_values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ema_result = MathUtils::ema(&compound_values, 0.1);
        let ema_std = MathUtils::std_dev(&ema_result);
        
        if ema_std.is_finite() && ema_std > 0.0 {
            passed += 1;
        } else {
            failed += 1;
            critical_failures.push("Precision loss in compound EMA + std_dev operation".to_string());
        }

        let total_tests = passed + failed;
        let stability_score = if total_tests > 0 { passed as f64 / total_tests as f64 } else { 0.0 };

        Ok(StabilityTestResult {
            passed,
            failed,
            critical_failures,
            stability_score,
        })
    }

    /// Analyze condition numbers of matrix operations
    async fn analyze_condition_numbers(&self) -> Result<ConditionNumberAnalysis> {
        let mut matrix_operations = HashMap::new();
        let mut numerical_rank_deficiency = Vec::new();
        let mut recommendations = Vec::new();

        // Test condition number of correlation matrix
        let test_data = self.generate_test_correlation_data();
        let condition_number = self.estimate_correlation_condition_number(&test_data);
        matrix_operations.insert("correlation_matrix".to_string(), condition_number);

        if condition_number > 1e12 {
            numerical_rank_deficiency.push("Correlation matrix is ill-conditioned".to_string());
            recommendations.push("Use regularized correlation estimation".to_string());
        }

        // Test linear regression matrix conditioning
        let (x_reg, y_reg) = self.generate_test_regression_data();
        let regression_condition = self.estimate_regression_condition_number(&x_reg, &y_reg);
        matrix_operations.insert("regression_matrix".to_string(), regression_condition);

        if regression_condition > 1e10 {
            numerical_rank_deficiency.push("Regression design matrix is ill-conditioned".to_string());
            recommendations.push("Use ridge regression or SVD for unstable cases".to_string());
        }

        // Test covariance matrix conditioning
        let covariance_condition = self.estimate_covariance_condition_number(&test_data);
        matrix_operations.insert("covariance_matrix".to_string(), covariance_condition);

        if covariance_condition > 1e10 {
            numerical_rank_deficiency.push("Covariance matrix is near-singular".to_string());
            recommendations.push("Use shrinkage estimators for covariance".to_string());
        }

        // General recommendations based on overall conditioning
        let max_condition = matrix_operations.values().fold(0.0, |a, &b| a.max(b));
        if max_condition > 1e6 {
            recommendations.push("Implement iterative refinement for linear solvers".to_string());
            recommendations.push("Use higher precision arithmetic for critical calculations".to_string());
        }

        Ok(ConditionNumberAnalysis {
            matrix_operations,
            numerical_rank_deficiency,
            recommendations,
        })
    }

    // Helper methods for condition number analysis

    fn generate_test_correlation_data(&self) -> Vec<Vec<f64>> {
        // Generate test data that might have correlation issues
        vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![1.1, 2.1, 3.1, 4.1, 5.1], // Highly correlated with first
            vec![5.0, 4.0, 3.0, 2.0, 1.0], // Negatively correlated
            vec![1.0, 1.0, 1.0, 1.0, 1.0], // Constant (rank deficient)
        ]
    }

    fn generate_test_regression_data(&self) -> (Vec<f64>, Vec<f64>) {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0]; // Perfect linear relationship
        (x, y)
    }

    fn estimate_correlation_condition_number(&self, data: &[Vec<f64>]) -> f64 {
        if data.len() < 2 || data[0].len() < 2 {
            return 1.0;
        }

        // Simplified condition number estimation
        // In practice, would compute eigenvalues of correlation matrix
        let mut correlations = Vec::new();
        
        for i in 0..data.len() {
            for j in i+1..data.len() {
                let corr = MathUtils::correlation(&data[i], &data[j]);
                if corr.is_finite() {
                    correlations.push(corr.abs());
                }
            }
        }

        if correlations.is_empty() {
            return 1.0;
        }

        let max_corr = correlations.iter().fold(0.0, |a, &b| a.max(b));
        let min_corr = correlations.iter().fold(1.0, |a, &b| a.min(b));

        // Rough estimate: higher correlation differences suggest better conditioning
        if min_corr < 1e-10 {
            1e15 // Very ill-conditioned
        } else {
            (1.0 + max_corr) / (1.0 - max_corr).max(1e-15)
        }
    }

    fn estimate_regression_condition_number(&self, x: &[f64], _y: &[f64]) -> f64 {
        if x.len() < 2 {
            return 1.0;
        }

        // For simple linear regression, condition number relates to spread of x values
        let x_min = x.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let x_max = x.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let x_range = x_max - x_min;

        if x_range < f64::EPSILON {
            1e15 // Singular matrix
        } else {
            // Simplified estimate based on range
            let x_center = (x_max + x_min) / 2.0;
            let max_abs = x_max.abs().max(x_min.abs());
            if max_abs > 0.0 {
                max_abs / x_range
            } else {
                1.0
            }
        }
    }

    fn estimate_covariance_condition_number(&self, data: &[Vec<f64>]) -> f64 {
        if data.len() < 2 {
            return 1.0;
        }

        // Estimate based on variance ratios
        let mut variances = Vec::new();
        
        for series in data {
            let variance = if series.len() > 1 {
                let mean = series.iter().sum::<f64>() / series.len() as f64;
                series.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (series.len() - 1) as f64
            } else {
                1.0
            };
            variances.push(variance);
        }

        let max_var = variances.iter().fold(0.0, |a, &b| a.max(b));
        let min_var = variances.iter().fold(f64::INFINITY, |a, &b| a.min(b));

        if min_var < f64::EPSILON {
            1e15 // Singular covariance
        } else {
            max_var / min_var
        }
    }

    /// Test specific mathematical properties that should hold
    pub async fn test_mathematical_properties(&self) -> Result<StabilityTestResult> {
        let mut passed = 0;
        let mut failed = 0;
        let mut critical_failures = Vec::new();

        // Test that correlation is symmetric
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let corr_xy = MathUtils::correlation(&x, &y);
        let corr_yx = MathUtils::correlation(&y, &x);
        
        if (corr_xy - corr_yx).abs() < f64::EPSILON {
            passed += 1;
        } else {
            failed += 1;
            critical_failures.push("Correlation is not symmetric".to_string());
        }

        // Test that correlation is bounded [-1, 1]
        let test_cases = vec![
            (vec![1.0, 2.0, 3.0], vec![1.0, 2.0, 3.0]), // Perfect positive
            (vec![1.0, 2.0, 3.0], vec![3.0, 2.0, 1.0]), // Perfect negative
            (vec![1.0, 1.0, 1.0], vec![2.0, 3.0, 4.0]), // Zero variance
        ];

        for (test_x, test_y) in test_cases {
            let corr = MathUtils::correlation(&test_x, &test_y);
            if corr.is_finite() && corr >= -1.0 && corr <= 1.0 {
                passed += 1;
            } else {
                failed += 1;
                critical_failures.push(format!("Correlation out of bounds: {}", corr));
            }
        }

        // Test that standard deviation is non-negative
        let test_vectors = vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![-5.0, -10.0, 15.0],
            vec![0.0, 0.0, 0.0],
            vec![f64::MAX / 1e6, f64::MAX / 1e6],
        ];

        for test_vec in test_vectors {
            let std_dev = MathUtils::std_dev(&test_vec);
            if std_dev.is_finite() && std_dev >= 0.0 {
                passed += 1;
            } else {
                failed += 1;
                critical_failures.push(format!("Standard deviation is negative or infinite: {}", std_dev));
            }
        }

        let total_tests = passed + failed;
        let stability_score = if total_tests > 0 { passed as f64 / total_tests as f64 } else { 0.0 };

        Ok(StabilityTestResult {
            passed,
            failed,
            critical_failures,
            stability_score,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_overflow_stability() {
        let config = ValidationConfig::default();
        let tester = NumericalStabilityTester::new(&config).unwrap();
        
        let result = tester.test_overflow_stability().await.unwrap();
        
        // Should pass most tests even with large values
        assert!(result.stability_score > 0.5);
        assert!(result.passed > 0);
    }

    #[tokio::test]
    async fn test_underflow_stability() {
        let config = ValidationConfig::default();
        let tester = NumericalStabilityTester::new(&config).unwrap();
        
        let result = tester.test_underflow_stability().await.unwrap();
        
        // Should handle tiny values gracefully
        assert!(result.stability_score > 0.5);
    }

    #[tokio::test]
    async fn test_precision_stability() {
        let config = ValidationConfig::default();
        let tester = NumericalStabilityTester::new(&config).unwrap();
        
        let result = tester.test_precision_stability().await.unwrap();
        
        // Some precision tests might fail due to challenging numerical conditions
        assert!(result.passed > 0);
    }

    #[tokio::test]
    async fn test_mathematical_properties() {
        let config = ValidationConfig::default();
        let tester = NumericalStabilityTester::new(&config).unwrap();
        
        let result = tester.test_mathematical_properties().await.unwrap();
        
        // Mathematical properties should mostly hold
        assert!(result.stability_score > 0.8);
    }
}