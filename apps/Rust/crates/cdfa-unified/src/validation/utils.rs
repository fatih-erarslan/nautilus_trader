//! Validation utility functions and helpers

use crate::error::{CdfaError, Result};
use crate::types::Float;
use crate::validation::financial::{FinancialValidator, ValidationReport};

/// Batch validation utilities for high-throughput scenarios
pub struct BatchValidator {
    validator: FinancialValidator,
    batch_size: usize,
    error_threshold: f64,
}

impl BatchValidator {
    /// Create a new batch validator
    pub fn new(batch_size: usize, error_threshold: f64) -> Self {
        Self {
            validator: FinancialValidator::new(),
            batch_size,
            error_threshold,
        }
    }

    /// Validate a batch of price data
    pub fn validate_price_batch(&mut self, prices: &[Float], asset_type: &str) -> Result<ValidationReport> {
        if prices.len() > self.batch_size {
            return Err(CdfaError::invalid_input(format!(
                "Batch size {} exceeds maximum {}", prices.len(), self.batch_size
            )));
        }

        let report = self.validator.validate_price_series(prices, asset_type);
        
        let error_rate = (report.errors + report.critical_errors) as f64 / report.data_points_validated as f64;
        
        if error_rate > self.error_threshold {
            return Err(CdfaError::invalid_input(format!(
                "Error rate {:.2}% exceeds threshold {:.2}%", 
                error_rate * 100.0, self.error_threshold * 100.0
            )));
        }

        Ok(report)
    }

    /// Validate multiple asset types in a single batch
    pub fn validate_multi_asset_batch(
        &mut self, 
        data: &[(Vec<Float>, String)]
    ) -> Result<Vec<ValidationReport>> {
        let mut reports = Vec::new();
        
        for (prices, asset_type) in data {
            let report = self.validate_price_batch(prices, asset_type)?;
            reports.push(report);
        }

        Ok(reports)
    }
}

/// Statistical validation helpers
pub mod stats {
    use super::*;
    use std::collections::HashMap;

    /// Calculate z-scores for anomaly detection
    pub fn calculate_z_scores(values: &[Float]) -> Result<Vec<Float>> {
        if values.is_empty() {
            return Err(CdfaError::invalid_input("Cannot calculate z-scores for empty array"));
        }

        let mean = values.iter().sum::<Float>() / values.len() as Float;
        let variance = values.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<Float>() / values.len() as Float;
        let std_dev = variance.sqrt();

        if std_dev == 0.0 {
            return Ok(vec![0.0; values.len()]);
        }

        let z_scores = values.iter()
            .map(|x| (x - mean) / std_dev)
            .collect();

        Ok(z_scores)
    }

    /// Detect outliers using IQR method
    pub fn detect_outliers_iqr(values: &[Float]) -> Result<Vec<usize>> {
        if values.is_empty() {
            return Err(CdfaError::invalid_input("Cannot detect outliers in empty array"));
        }

        let mut sorted_values = values.to_vec();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let n = sorted_values.len();
        let q1_idx = n / 4;
        let q3_idx = 3 * n / 4;

        let q1 = sorted_values[q1_idx];
        let q3 = sorted_values[q3_idx];
        let iqr = q3 - q1;
        
        let lower_bound = q1 - 1.5 * iqr;
        let upper_bound = q3 + 1.5 * iqr;

        let outliers = values.iter()
            .enumerate()
            .filter(|(_, &value)| value < lower_bound || value > upper_bound)
            .map(|(idx, _)| idx)
            .collect();

        Ok(outliers)
    }

    /// Calculate volatility metrics
    pub fn calculate_volatility_metrics(prices: &[Float]) -> Result<HashMap<String, Float>> {
        if prices.len() < 2 {
            return Err(CdfaError::invalid_input("Need at least 2 prices for volatility calculation"));
        }

        let mut metrics = HashMap::new();
        
        // Calculate returns
        let returns: Vec<Float> = prices.windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect();

        // Standard deviation of returns
        let mean_return = returns.iter().sum::<Float>() / returns.len() as Float;
        let variance = returns.iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<Float>() / returns.len() as Float;
        let volatility = variance.sqrt();

        metrics.insert("volatility".to_string(), volatility);
        metrics.insert("mean_return".to_string(), mean_return);
        
        // Maximum daily move
        let max_abs_return = returns.iter()
            .map(|r| r.abs())
            .fold(0.0, |acc, x| acc.max(x));
        metrics.insert("max_daily_move".to_string(), max_abs_return);

        // Minimum and maximum returns
        let min_return = returns.iter().fold(Float::INFINITY, |acc, &x| acc.min(x));
        let max_return = returns.iter().fold(Float::NEG_INFINITY, |acc, &x| acc.max(x));
        
        metrics.insert("min_return".to_string(), min_return);
        metrics.insert("max_return".to_string(), max_return);

        Ok(metrics)
    }
}

/// Time series validation utilities
pub mod timeseries {
    use super::*;

    /// Validate timestamp gaps and frequency
    pub fn validate_timestamp_frequency(timestamps: &[i64], expected_interval_ms: i64) -> Result<()> {
        if timestamps.len() < 2 {
            return Ok(());
        }

        let tolerance = expected_interval_ms as f64 * 0.1; // 10% tolerance

        for i in 1..timestamps.len() {
            let actual_interval = timestamps[i] - timestamps[i - 1];
            let diff = (actual_interval - expected_interval_ms).abs() as f64;
            
            if diff > tolerance {
                return Err(CdfaError::invalid_input(format!(
                    "Timestamp interval {} at position {} differs from expected {} by more than tolerance {}",
                    actual_interval, i, expected_interval_ms, tolerance
                )));
            }
        }

        Ok(())
    }

    /// Detect missing data points in time series
    pub fn detect_missing_timestamps(
        timestamps: &[i64], 
        start_time: i64, 
        end_time: i64, 
        interval_ms: i64
    ) -> Vec<i64> {
        let mut expected_timestamps = Vec::new();
        let mut current_time = start_time;
        
        while current_time <= end_time {
            expected_timestamps.push(current_time);
            current_time += interval_ms;
        }

        let actual_set: std::collections::HashSet<i64> = timestamps.iter().cloned().collect();
        
        expected_timestamps.into_iter()
            .filter(|t| !actual_set.contains(t))
            .collect()
    }

    /// Validate time series continuity
    pub fn validate_continuity(timestamps: &[i64], max_gap_ms: i64) -> Result<()> {
        for i in 1..timestamps.len() {
            let gap = timestamps[i] - timestamps[i - 1];
            if gap > max_gap_ms {
                return Err(CdfaError::invalid_input(format!(
                    "Gap of {}ms between timestamps at positions {} and {} exceeds maximum allowed gap of {}ms",
                    gap, i - 1, i, max_gap_ms
                )));
            }
        }
        Ok(())
    }
}

/// Market data validation presets for different exchanges and asset types
pub mod presets {
    use super::*;
    use crate::validation::financial::{AssetValidationRules, MarketRangeBounds};

    /// Create validation rules for NYSE/NASDAQ stocks
    pub fn nyse_nasdaq_stock_rules() -> AssetValidationRules {
        AssetValidationRules {
            asset_type: "us_stock".to_string(),
            bounds: MarketRangeBounds {
                min_historical_price: 0.01,
                max_historical_price: 1e6,
                min_historical_volume: 0.0,
                max_historical_volume: 1e11,
                max_historical_change: 1.0, // 100% max daily change
            },
            enable_flash_crash_detection: true,
            enable_manipulation_detection: true,
            min_data_points: 20,
        }
    }

    /// Create validation rules for Binance cryptocurrency
    pub fn binance_crypto_rules() -> AssetValidationRules {
        AssetValidationRules {
            asset_type: "binance_crypto".to_string(),
            bounds: MarketRangeBounds {
                min_historical_price: 1e-12,
                max_historical_price: 1e8,
                min_historical_volume: 0.0,
                max_historical_volume: 1e15,
                max_historical_change: 5.0, // 500% max daily change
            },
            enable_flash_crash_detection: true,
            enable_manipulation_detection: true,
            min_data_points: 10,
        }
    }

    /// Create validation rules for major forex pairs
    pub fn major_forex_rules() -> AssetValidationRules {
        AssetValidationRules {
            asset_type: "major_forex".to_string(),
            bounds: MarketRangeBounds {
                min_historical_price: 0.0001,
                max_historical_price: 1000.0,
                min_historical_volume: 0.0,
                max_historical_volume: 1e13,
                max_historical_change: 0.2, // 20% max daily change
            },
            enable_flash_crash_detection: true,
            enable_manipulation_detection: true,
            min_data_points: 50,
        }
    }

    /// Create validation rules for commodities (gold, oil, etc.)
    pub fn commodity_rules() -> AssetValidationRules {
        AssetValidationRules {
            asset_type: "commodity".to_string(),
            bounds: MarketRangeBounds {
                min_historical_price: 0.01,
                max_historical_price: 1e5,
                min_historical_volume: 0.0,
                max_historical_volume: 1e10,
                max_historical_change: 0.5, // 50% max daily change
            },
            enable_flash_crash_detection: true,
            enable_manipulation_detection: true,
            min_data_points: 30,
        }
    }

    /// Create a validator with all preset rules
    pub fn create_preset_validator() -> FinancialValidator {
        let mut validator = FinancialValidator::new();
        
        validator.add_asset_rules("us_stock".to_string(), nyse_nasdaq_stock_rules());
        validator.add_asset_rules("binance_crypto".to_string(), binance_crypto_rules());
        validator.add_asset_rules("major_forex".to_string(), major_forex_rules());
        validator.add_asset_rules("commodity".to_string(), commodity_rules());
        
        validator
    }
}

/// Performance monitoring for validation operations
pub struct ValidationPerformanceMonitor {
    validation_count: usize,
    total_validation_time_us: u64,
    errors_detected: usize,
    data_points_processed: usize,
}

impl ValidationPerformanceMonitor {
    /// Create a new performance monitor
    pub fn new() -> Self {
        Self {
            validation_count: 0,
            total_validation_time_us: 0,
            errors_detected: 0,
            data_points_processed: 0,
        }
    }

    /// Record a validation operation
    pub fn record_validation(&mut self, duration_us: u64, data_points: usize, errors: usize) {
        self.validation_count += 1;
        self.total_validation_time_us += duration_us;
        self.data_points_processed += data_points;
        self.errors_detected += errors;
    }

    /// Get performance statistics
    pub fn get_stats(&self) -> ValidationStats {
        ValidationStats {
            total_validations: self.validation_count,
            average_time_per_validation_us: if self.validation_count > 0 {
                self.total_validation_time_us / self.validation_count as u64
            } else {
                0
            },
            total_data_points: self.data_points_processed,
            total_errors: self.errors_detected,
            error_rate: if self.data_points_processed > 0 {
                self.errors_detected as f64 / self.data_points_processed as f64
            } else {
                0.0
            },
            throughput_points_per_second: if self.total_validation_time_us > 0 {
                (self.data_points_processed as f64 * 1_000_000.0) / self.total_validation_time_us as f64
            } else {
                0.0
            },
        }
    }

    /// Reset statistics
    pub fn reset(&mut self) {
        self.validation_count = 0;
        self.total_validation_time_us = 0;
        self.errors_detected = 0;
        self.data_points_processed = 0;
    }
}

/// Validation performance statistics
#[derive(Debug, Clone)]
pub struct ValidationStats {
    pub total_validations: usize,
    pub average_time_per_validation_us: u64,
    pub total_data_points: usize,
    pub total_errors: usize,
    pub error_rate: f64,
    pub throughput_points_per_second: f64,
}

impl Default for ValidationPerformanceMonitor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_validator() {
        let mut batch_validator = BatchValidator::new(1000, 0.1); // 10% error threshold
        
        let good_prices = vec![100.0, 101.0, 99.0, 102.0, 98.0];
        let result = batch_validator.validate_price_batch(&good_prices, "stock");
        assert!(result.is_ok());

        let bad_prices = vec![Float::NAN, Float::INFINITY, -100.0, 1e20];
        let result = batch_validator.validate_price_batch(&bad_prices, "stock");
        assert!(result.is_err()); // Should exceed error threshold
    }

    #[test]
    fn test_z_score_calculation() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let z_scores = stats::calculate_z_scores(&values).unwrap();
        
        // Z-scores should sum to approximately 0
        let sum: Float = z_scores.iter().sum();
        assert!((sum).abs() < 1e-10);
    }

    #[test]
    fn test_outlier_detection() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 100.0]; // 100.0 is clear outlier
        let outliers = stats::detect_outliers_iqr(&values).unwrap();
        
        assert!(!outliers.is_empty());
        assert!(outliers.contains(&5)); // Index of outlier value
    }

    #[test]
    fn test_volatility_metrics() {
        let prices = vec![100.0, 102.0, 98.0, 105.0, 95.0];
        let metrics = stats::calculate_volatility_metrics(&prices).unwrap();
        
        assert!(metrics.contains_key("volatility"));
        assert!(metrics.contains_key("mean_return"));
        assert!(metrics.contains_key("max_daily_move"));
    }

    #[test]
    fn test_timestamp_frequency_validation() {
        let timestamps = vec![1000, 2000, 3000, 4000]; // 1000ms intervals
        let result = timeseries::validate_timestamp_frequency(&timestamps, 1000);
        assert!(result.is_ok());

        let irregular_timestamps = vec![1000, 2000, 3500, 4000]; // Irregular interval
        let result = timeseries::validate_timestamp_frequency(&irregular_timestamps, 1000);
        assert!(result.is_err());
    }

    #[test]
    fn test_missing_timestamp_detection() {
        let timestamps = vec![1000, 2000, 4000, 5000]; // Missing 3000
        let missing = timeseries::detect_missing_timestamps(&timestamps, 1000, 5000, 1000);
        
        assert_eq!(missing, vec![3000]);
    }

    #[test]
    fn test_preset_validators() {
        let validator = presets::create_preset_validator();
        
        // Test different asset types with their specific rules
        assert!(validator.validate_price(100.0, "us_stock").is_ok());
        assert!(validator.validate_price(0.000001, "binance_crypto").is_ok());
        assert!(validator.validate_price(1.1234, "major_forex").is_ok());
        assert!(validator.validate_price(2000.0, "commodity").is_ok());
    }

    #[test]
    fn test_performance_monitor() {
        let mut monitor = ValidationPerformanceMonitor::new();
        
        monitor.record_validation(1000, 100, 2);
        monitor.record_validation(1500, 150, 3);
        
        let stats = monitor.get_stats();
        assert_eq!(stats.total_validations, 2);
        assert_eq!(stats.total_data_points, 250);
        assert_eq!(stats.total_errors, 5);
        assert_eq!(stats.average_time_per_validation_us, 1250);
    }
}