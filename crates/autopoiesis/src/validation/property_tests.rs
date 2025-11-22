//! Property-Based Testing for Mathematical Functions
//! 
//! This module implements property-based testing to verify mathematical
//! properties that should hold for all valid inputs.

use crate::Result;
use crate::validation::ValidationConfig;
use crate::utils::MathUtils;
use rand::{SeedableRng, Rng};
use rand_chacha::ChaCha8Rng;

pub struct PropertyTester {
    config: ValidationConfig,
    rng: ChaCha8Rng,
}

/// Property test result
#[derive(Debug, Clone)]
pub struct PropertyTestResult {
    pub property_name: String,
    pub tests_run: usize,
    pub failures: usize,
    pub success_rate: f64,
    pub failed_examples: Vec<String>,
}

impl PropertyTester {
    pub fn new(config: &ValidationConfig) -> Result<Self> {
        let rng = ChaCha8Rng::seed_from_u64(config.random_seed);
        
        Ok(Self {
            config: config.clone(),
            rng,
        })
    }

    /// Run all property-based tests
    pub async fn run_all_property_tests(&mut self) -> Result<Vec<PropertyTestResult>> {
        let mut results = Vec::new();
        
        // Test mathematical properties
        results.push(self.test_ema_properties().await?);
        results.push(self.test_sma_properties().await?);
        results.push(self.test_std_dev_properties().await?);
        results.push(self.test_correlation_properties().await?);
        results.push(self.test_linear_regression_properties().await?);
        results.push(self.test_percentile_properties().await?);
        results.push(self.test_z_score_properties().await?);
        
        Ok(results)
    }

    /// Test EMA mathematical properties
    async fn test_ema_properties(&mut self) -> Result<PropertyTestResult> {
        let mut failures = 0;
        let mut failed_examples = Vec::new();
        let tests_run = 1000;
        
        for i in 0..tests_run {
            // Generate random test data
            let size = self.rng.gen_range(2..1000);
            let values = self.generate_positive_values(size);
            let alpha = self.rng.gen_range(0.01..0.99);
            
            let result = MathUtils::ema(&values, alpha);
            
            // Property 1: EMA result should have same length as input
            if result.len() != values.len() {
                failures += 1;
                failed_examples.push(format!("Test {}: EMA length mismatch", i));
                continue;
            }
            
            // Property 2: EMA values should be bounded by min/max of input
            let min_val = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max_val = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            
            let ema_bounded = result.iter().all(|&x| x >= min_val && x <= max_val);
            if !ema_bounded {
                failures += 1;
                failed_examples.push(format!("Test {}: EMA not bounded by input range", i));
                continue;
            }
            
            // Property 3: First EMA value should equal first input value
            if (result[0] - values[0]).abs() > f64::EPSILON {
                failures += 1;
                failed_examples.push(format!("Test {}: First EMA value incorrect", i));
                continue;
            }
            
            // Property 4: EMA should be finite for finite inputs
            if !result.iter().all(|x| x.is_finite()) {
                failures += 1;
                failed_examples.push(format!("Test {}: EMA produced non-finite values", i));
            }
        }
        
        let success_rate = (tests_run - failures) as f64 / tests_run as f64;
        
        Ok(PropertyTestResult {
            property_name: "EMA Properties".to_string(),
            tests_run,
            failures,
            success_rate,
            failed_examples,
        })
    }

    /// Test SMA mathematical properties
    async fn test_sma_properties(&mut self) -> Result<PropertyTestResult> {
        let mut failures = 0;
        let mut failed_examples = Vec::new();
        let tests_run = 1000;
        
        for i in 0..tests_run {
            let size = self.rng.gen_range(10..1000);
            let window = self.rng.gen_range(2..size.min(100));
            let values = self.generate_random_values(size);
            
            let result = MathUtils::sma(&values, window);
            
            // Property 1: SMA result length should be correct
            let expected_length = if values.len() >= window { values.len() - window + 1 } else { 0 };
            if result.len() != expected_length {
                failures += 1;
                failed_examples.push(format!("Test {}: SMA length incorrect", i));
                continue;
            }
            
            // Property 2: SMA values should be bounded by input range
            if !result.is_empty() {
                let min_val = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                let max_val = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                
                let sma_bounded = result.iter().all(|&x| x >= min_val && x <= max_val);
                if !sma_bounded {
                    failures += 1;
                    failed_examples.push(format!("Test {}: SMA not bounded by input range", i));
                    continue;
                }
            }
            
            // Property 3: For constant input, SMA should equal the constant
            let constant_values = vec![42.0; window + 5];
            let constant_sma = MathUtils::sma(&constant_values, window);
            if !constant_sma.iter().all(|&x| (x - 42.0).abs() < f64::EPSILON) {
                failures += 1;
                failed_examples.push(format!("Test {}: SMA of constant values incorrect", i));
            }
        }
        
        let success_rate = (tests_run - failures) as f64 / tests_run as f64;
        
        Ok(PropertyTestResult {
            property_name: "SMA Properties".to_string(),
            tests_run,
            failures,
            success_rate,
            failed_examples,
        })
    }

    /// Test standard deviation properties
    async fn test_std_dev_properties(&mut self) -> Result<PropertyTestResult> {
        let mut failures = 0;
        let mut failed_examples = Vec::new();
        let tests_run = 1000;
        
        for i in 0..tests_run {
            let size = self.rng.gen_range(2..1000);
            let values = self.generate_random_values(size);
            
            let std_dev = MathUtils::std_dev(&values);
            
            // Property 1: Standard deviation should be non-negative
            if std_dev < 0.0 {
                failures += 1;
                failed_examples.push(format!("Test {}: Negative standard deviation", i));
                continue;
            }
            
            // Property 2: Standard deviation should be finite for finite inputs
            if !std_dev.is_finite() {
                failures += 1;
                failed_examples.push(format!("Test {}: Non-finite standard deviation", i));
                continue;
            }
            
            // Property 3: Standard deviation of constant values should be zero
            let constant_values = vec![7.5; size];
            let constant_std = MathUtils::std_dev(&constant_values);
            if constant_std.abs() > f64::EPSILON {
                failures += 1;
                failed_examples.push(format!("Test {}: Constant values std dev not zero", i));
                continue;
            }
            
            // Property 4: Scaling property - std_dev(k*X) = |k| * std_dev(X)
            let scale_factor = self.rng.gen_range(-10.0..10.0);
            let scaled_values: Vec<f64> = values.iter().map(|&x| scale_factor * x).collect();
            let scaled_std = MathUtils::std_dev(&scaled_values);
            let expected_scaled_std = scale_factor.abs() * std_dev;
            
            if (scaled_std - expected_scaled_std).abs() > 1e-10 * expected_scaled_std.max(1.0) {
                failures += 1;
                failed_examples.push(format!("Test {}: Scaling property violated", i));
            }
        }
        
        let success_rate = (tests_run - failures) as f64 / tests_run as f64;
        
        Ok(PropertyTestResult {
            property_name: "Standard Deviation Properties".to_string(),
            tests_run,
            failures,
            success_rate,
            failed_examples,
        })
    }

    /// Test correlation properties
    async fn test_correlation_properties(&mut self) -> Result<PropertyTestResult> {
        let mut failures = 0;
        let mut failed_examples = Vec::new();
        let tests_run = 1000;
        
        for i in 0..tests_run {
            let size = self.rng.gen_range(2..1000);
            let x = self.generate_random_values(size);
            let y = self.generate_random_values(size);
            
            let corr_xy = MathUtils::correlation(&x, &y);
            let corr_yx = MathUtils::correlation(&y, &x);
            
            // Property 1: Correlation should be symmetric
            if (corr_xy - corr_yx).abs() > 1e-15 {
                failures += 1;
                failed_examples.push(format!("Test {}: Correlation not symmetric", i));
                continue;
            }
            
            // Property 2: Correlation should be bounded [-1, 1]
            if corr_xy < -1.0 || corr_xy > 1.0 || !corr_xy.is_finite() {
                failures += 1;
                failed_examples.push(format!("Test {}: Correlation out of bounds: {}", i, corr_xy));
                continue;
            }
            
            // Property 3: Perfect positive correlation
            let perfect_y: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + 3.0).collect();
            let perfect_corr = MathUtils::correlation(&x, &perfect_y);
            if (perfect_corr - 1.0).abs() > 1e-10 {
                failures += 1;
                failed_examples.push(format!("Test {}: Perfect correlation not 1.0: {}", i, perfect_corr));
                continue;
            }
            
            // Property 4: Perfect negative correlation
            let negative_y: Vec<f64> = x.iter().map(|&xi| -2.0 * xi + 3.0).collect();
            let negative_corr = MathUtils::correlation(&x, &negative_y);
            if (negative_corr + 1.0).abs() > 1e-10 {
                failures += 1;
                failed_examples.push(format!("Test {}: Perfect negative correlation not -1.0: {}", i, negative_corr));
                continue;
            }
            
            // Property 5: Self-correlation should be 1 (unless constant)
            let self_corr = MathUtils::correlation(&x, &x);
            let x_std = MathUtils::std_dev(&x);
            if x_std > f64::EPSILON && (self_corr - 1.0).abs() > 1e-10 {
                failures += 1;
                failed_examples.push(format!("Test {}: Self-correlation not 1.0: {}", i, self_corr));
            }
        }
        
        let success_rate = (tests_run - failures) as f64 / tests_run as f64;
        
        Ok(PropertyTestResult {
            property_name: "Correlation Properties".to_string(),
            tests_run,
            failures,
            success_rate,
            failed_examples,
        })
    }

    /// Test linear regression properties
    async fn test_linear_regression_properties(&mut self) -> Result<PropertyTestResult> {
        let mut failures = 0;
        let mut failed_examples = Vec::new();
        let tests_run = 1000;
        
        for i in 0..tests_run {
            let size = self.rng.gen_range(2..100);
            let x = self.generate_random_values(size);
            let y = self.generate_random_values(size);
            
            if let Some((slope, intercept)) = MathUtils::linear_regression(&x, &y) {
                // Property 1: Coefficients should be finite
                if !slope.is_finite() || !intercept.is_finite() {
                    failures += 1;
                    failed_examples.push(format!("Test {}: Non-finite regression coefficients", i));
                    continue;
                }
                
                // Property 2: Perfect linear relationship
                let true_slope = self.rng.gen_range(-10.0..10.0);
                let true_intercept = self.rng.gen_range(-10.0..10.0);
                let perfect_y: Vec<f64> = x.iter().map(|&xi| true_slope * xi + true_intercept).collect();
                
                if let Some((est_slope, est_intercept)) = MathUtils::linear_regression(&x, &perfect_y) {
                    let slope_error = (est_slope - true_slope).abs();
                    let intercept_error = (est_intercept - true_intercept).abs();
                    
                    if slope_error > 1e-10 || intercept_error > 1e-10 {
                        failures += 1;
                        failed_examples.push(format!(
                            "Test {}: Perfect linear relationship not recovered. Slope error: {}, Intercept error: {}",
                            i, slope_error, intercept_error
                        ));
                    }
                }
            }
            
            // Property 3: Regression should handle identical x values gracefully
            let identical_x = vec![5.0; size];
            let result = MathUtils::linear_regression(&identical_x, &y);
            if result.is_some() {
                // Should return None for identical x values (singular matrix)
                failures += 1;
                failed_examples.push(format!("Test {}: Regression didn't handle singular case", i));
            }
        }
        
        let success_rate = (tests_run - failures) as f64 / tests_run as f64;
        
        Ok(PropertyTestResult {
            property_name: "Linear Regression Properties".to_string(),
            tests_run,
            failures,
            success_rate,
            failed_examples,
        })
    }

    /// Test percentile properties
    async fn test_percentile_properties(&mut self) -> Result<PropertyTestResult> {
        let mut failures = 0;
        let mut failed_examples = Vec::new();
        let tests_run = 1000;
        
        for i in 0..tests_run {
            let size = self.rng.gen_range(1..1000);
            let values = self.generate_random_values(size);
            let p = self.rng.gen_range(0.0..1.0);
            
            let percentile = MathUtils::percentile(&values, p);
            
            // Property 1: Percentile should be within data range
            let min_val = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max_val = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            
            if percentile < min_val || percentile > max_val {
                failures += 1;
                failed_examples.push(format!("Test {}: Percentile outside data range", i));
                continue;
            }
            
            // Property 2: Percentile should be finite for finite inputs
            if !percentile.is_finite() {
                failures += 1;
                failed_examples.push(format!("Test {}: Non-finite percentile", i));
                continue;
            }
            
            // Property 3: Monotonicity - higher percentiles should give higher values
            if p < 0.9 {
                let higher_p = p + 0.1;
                let higher_percentile = MathUtils::percentile(&values, higher_p);
                if higher_percentile < percentile - f64::EPSILON {
                    failures += 1;
                    failed_examples.push(format!("Test {}: Percentile monotonicity violated", i));
                    continue;
                }
            }
            
            // Property 4: 0th percentile should be minimum, 100th should be maximum
            let min_percentile = MathUtils::percentile(&values, 0.0);
            let max_percentile = MathUtils::percentile(&values, 1.0);
            
            if (min_percentile - min_val).abs() > f64::EPSILON {
                failures += 1;
                failed_examples.push(format!("Test {}: 0th percentile not minimum", i));
                continue;
            }
            
            if (max_percentile - max_val).abs() > f64::EPSILON {
                failures += 1;
                failed_examples.push(format!("Test {}: 100th percentile not maximum", i));
            }
        }
        
        let success_rate = (tests_run - failures) as f64 / tests_run as f64;
        
        Ok(PropertyTestResult {
            property_name: "Percentile Properties".to_string(),
            tests_run,
            failures,
            success_rate,
            failed_examples,
        })
    }

    /// Test Z-score normalization properties
    async fn test_z_score_properties(&mut self) -> Result<PropertyTestResult> {
        let mut failures = 0;
        let mut failed_examples = Vec::new();
        let tests_run = 1000;
        
        for i in 0..tests_run {
            let size = self.rng.gen_range(3..1000);
            let values = self.generate_random_values(size);
            
            let z_scores = MathUtils::z_score(&values);
            
            // Property 1: Z-scores should have same length as input
            if z_scores.len() != values.len() {
                failures += 1;
                failed_examples.push(format!("Test {}: Z-score length mismatch", i));
                continue;
            }
            
            // Property 2: Z-scores should be finite for finite inputs (unless zero variance)
            let input_std = MathUtils::std_dev(&values);
            if input_std > f64::EPSILON {
                if !z_scores.iter().all(|x| x.is_finite()) {
                    failures += 1;
                    failed_examples.push(format!("Test {}: Non-finite Z-scores", i));
                    continue;
                }
                
                // Property 3: Mean of Z-scores should be approximately zero
                let z_mean = z_scores.iter().sum::<f64>() / z_scores.len() as f64;
                if z_mean.abs() > 1e-10 {
                    failures += 1;
                    failed_examples.push(format!("Test {}: Z-score mean not zero: {}", i, z_mean));
                    continue;
                }
                
                // Property 4: Standard deviation of Z-scores should be approximately 1
                let z_std = MathUtils::std_dev(&z_scores);
                if (z_std - 1.0).abs() > 1e-10 {
                    failures += 1;
                    failed_examples.push(format!("Test {}: Z-score std dev not 1: {}", i, z_std));
                    continue;
                }
            }
            
            // Property 5: Constant values should yield zero Z-scores
            let constant_values = vec![42.0; size];
            let constant_z_scores = MathUtils::z_score(&constant_values);
            if !constant_z_scores.iter().all(|&x| x.abs() < f64::EPSILON) {
                failures += 1;
                failed_examples.push(format!("Test {}: Constant values Z-scores not zero", i));
            }
        }
        
        let success_rate = (tests_run - failures) as f64 / tests_run as f64;
        
        Ok(PropertyTestResult {
            property_name: "Z-Score Properties".to_string(),
            tests_run,
            failures,
            success_rate,
            failed_examples,
        })
    }

    // Helper methods

    /// Generate random values for testing
    fn generate_random_values(&mut self, size: usize) -> Vec<f64> {
        (0..size)
            .map(|_| self.rng.gen_range(-1000.0..1000.0))
            .collect()
    }

    /// Generate positive random values for testing
    fn generate_positive_values(&mut self, size: usize) -> Vec<f64> {
        (0..size)
            .map(|_| self.rng.gen_range(0.1..1000.0))
            .collect()
    }

    /// Test invariants under transformations
    pub async fn test_transformation_invariants(&mut self) -> Result<Vec<PropertyTestResult>> {
        let mut results = Vec::new();
        
        // Test translation invariance for correlation
        results.push(self.test_correlation_translation_invariance().await?);
        
        // Test scale invariance for correlation
        results.push(self.test_correlation_scale_invariance().await?);
        
        // Test affine invariance for linear regression
        results.push(self.test_regression_affine_invariance().await?);
        
        Ok(results)
    }

    async fn test_correlation_translation_invariance(&mut self) -> Result<PropertyTestResult> {
        let mut failures = 0;
        let mut failed_examples = Vec::new();
        let tests_run = 500;
        
        for i in 0..tests_run {
            let size = self.rng.gen_range(3..100);
            let x = self.generate_random_values(size);
            let y = self.generate_random_values(size);
            
            let original_corr = MathUtils::correlation(&x, &y);
            
            // Translate both series
            let shift = self.rng.gen_range(-100.0..100.0);
            let x_shifted: Vec<f64> = x.iter().map(|&v| v + shift).collect();
            let y_shifted: Vec<f64> = y.iter().map(|&v| v + shift).collect();
            
            let shifted_corr = MathUtils::correlation(&x_shifted, &y_shifted);
            
            if (original_corr - shifted_corr).abs() > 1e-12 {
                failures += 1;
                failed_examples.push(format!(
                    "Test {}: Translation changed correlation: {} -> {}",
                    i, original_corr, shifted_corr
                ));
            }
        }
        
        let success_rate = (tests_run - failures) as f64 / tests_run as f64;
        
        Ok(PropertyTestResult {
            property_name: "Correlation Translation Invariance".to_string(),
            tests_run,
            failures,
            success_rate,
            failed_examples,
        })
    }

    async fn test_correlation_scale_invariance(&mut self) -> Result<PropertyTestResult> {
        let mut failures = 0;
        let mut failed_examples = Vec::new();
        let tests_run = 500;
        
        for i in 0..tests_run {
            let size = self.rng.gen_range(3..100);
            let x = self.generate_random_values(size);
            let y = self.generate_random_values(size);
            
            let original_corr = MathUtils::correlation(&x, &y);
            
            // Scale both series by different factors
            let scale_x = self.rng.gen_range(0.1..10.0);
            let scale_y = self.rng.gen_range(0.1..10.0);
            let x_scaled: Vec<f64> = x.iter().map(|&v| v * scale_x).collect();
            let y_scaled: Vec<f64> = y.iter().map(|&v| v * scale_y).collect();
            
            let scaled_corr = MathUtils::correlation(&x_scaled, &y_scaled);
            
            if (original_corr - scaled_corr).abs() > 1e-12 {
                failures += 1;
                failed_examples.push(format!(
                    "Test {}: Scaling changed correlation: {} -> {}",
                    i, original_corr, scaled_corr
                ));
            }
        }
        
        let success_rate = (tests_run - failures) as f64 / tests_run as f64;
        
        Ok(PropertyTestResult {
            property_name: "Correlation Scale Invariance".to_string(),
            tests_run,
            failures,
            success_rate,
            failed_examples,
        })
    }

    async fn test_regression_affine_invariance(&mut self) -> Result<PropertyTestResult> {
        let mut failures = 0;
        let mut failed_examples = Vec::new();
        let tests_run = 500;
        
        for i in 0..tests_run {
            let size = self.rng.gen_range(3..50);
            let x = self.generate_random_values(size);
            let y = self.generate_random_values(size);
            
            if let Some((slope, intercept)) = MathUtils::linear_regression(&x, &y) {
                // Transform x by affine transformation: x' = a*x + b
                let a = self.rng.gen_range(0.1..10.0);
                let b = self.rng.gen_range(-10.0..10.0);
                let x_transformed: Vec<f64> = x.iter().map(|&v| a * v + b).collect();
                
                if let Some((new_slope, new_intercept)) = MathUtils::linear_regression(&x_transformed, &y) {
                    // Expected relationship: new_slope = slope / a
                    // new_intercept = intercept - slope * b / a
                    let expected_slope = slope / a;
                    let expected_intercept = intercept - slope * b / a;
                    
                    let slope_error = (new_slope - expected_slope).abs();
                    let intercept_error = (new_intercept - expected_intercept).abs();
                    
                    if slope_error > 1e-10 * expected_slope.abs().max(1.0) ||
                       intercept_error > 1e-10 * expected_intercept.abs().max(1.0) {
                        failures += 1;
                        failed_examples.push(format!(
                            "Test {}: Affine transformation failed. Slope error: {}, Intercept error: {}",
                            i, slope_error, intercept_error
                        ));
                    }
                }
            }
        }
        
        let success_rate = (tests_run - failures) as f64 / tests_run as f64;
        
        Ok(PropertyTestResult {
            property_name: "Linear Regression Affine Invariance".to_string(),
            tests_run,
            failures,
            success_rate,
            failed_examples,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_ema_properties() {
        let config = ValidationConfig::default();
        let mut tester = PropertyTester::new(&config).unwrap();
        
        let result = tester.test_ema_properties().await.unwrap();
        
        assert_eq!(result.property_name, "EMA Properties");
        assert!(result.success_rate > 0.95);
        assert!(result.tests_run > 0);
    }

    #[tokio::test]
    async fn test_correlation_properties() {
        let config = ValidationConfig::default();
        let mut tester = PropertyTester::new(&config).unwrap();
        
        let result = tester.test_correlation_properties().await.unwrap();
        
        assert_eq!(result.property_name, "Correlation Properties");
        assert!(result.success_rate > 0.95);
    }

    #[tokio::test]
    async fn test_all_properties() {
        let config = ValidationConfig::default();
        let mut tester = PropertyTester::new(&config).unwrap();
        
        let results = tester.run_all_property_tests().await.unwrap();
        
        assert!(!results.is_empty());
        for result in results {
            assert!(result.tests_run > 0);
            assert!(result.success_rate >= 0.0 && result.success_rate <= 1.0);
        }
    }

    #[tokio::test]
    async fn test_transformation_invariants() {
        let config = ValidationConfig::default();
        let mut tester = PropertyTester::new(&config).unwrap();
        
        let results = tester.test_transformation_invariants().await.unwrap();
        
        assert!(!results.is_empty());
        for result in results {
            assert!(result.success_rate > 0.8); // Should be quite high for these properties
        }
    }
}