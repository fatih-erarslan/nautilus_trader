//! Comprehensive financial input validation for market data
//!
//! This module provides mission-critical validation for financial data to prevent
//! invalid inputs from corrupting calculations or causing system failures.
//! Designed to handle edge cases like market crashes, flash crashes, and data manipulation.

use crate::error::{CdfaError, Result};
use crate::types::Float;
use std::collections::HashMap;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Maximum reasonable value for any financial data point (prevents overflow)
pub const MAX_FINANCIAL_VALUE: Float = 1e15;

/// Minimum reasonable price (prevents division by zero and invalid calculations)
pub const MIN_PRICE: Float = 1e-8;

/// Maximum reasonable daily price change (1000000% = 10,000x multiplier)
pub const MAX_DAILY_CHANGE_RATIO: Float = 10000.0;

/// Maximum reasonable volume (1 trillion units)
pub const MAX_REASONABLE_VOLUME: Float = 1e12;

/// Minimum reasonable correlation (allowing for perfect negative correlation)
pub const MIN_CORRELATION: Float = -1.0;

/// Maximum reasonable correlation (allowing for perfect positive correlation)
pub const MAX_CORRELATION: Float = 1.0;

/// Flash crash detection threshold (95% drop in single period)
pub const FLASH_CRASH_THRESHOLD: Float = -0.95;

/// Flash spike detection threshold (1000% increase in single period)
pub const FLASH_SPIKE_THRESHOLD: Float = 10.0;

/// Historical market range validation bounds
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct MarketRangeBounds {
    /// Minimum historical price for this asset class
    pub min_historical_price: Float,
    /// Maximum historical price for this asset class
    pub max_historical_price: Float,
    /// Minimum historical volume
    pub min_historical_volume: Float,
    /// Maximum historical volume
    pub max_historical_volume: Float,
    /// Maximum single-day price change observed historically
    pub max_historical_change: Float,
}

impl Default for MarketRangeBounds {
    fn default() -> Self {
        Self {
            min_historical_price: 1e-6,
            max_historical_price: 1e8,
            min_historical_volume: 0.0,
            max_historical_volume: 1e12,
            max_historical_change: 10.0, // 1000% change
        }
    }
}

/// Financial validation rules for different asset types
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AssetValidationRules {
    /// Asset type identifier
    pub asset_type: String,
    /// Market-specific bounds
    pub bounds: MarketRangeBounds,
    /// Enable flash crash detection
    pub enable_flash_crash_detection: bool,
    /// Enable manipulation pattern detection
    pub enable_manipulation_detection: bool,
    /// Minimum data points required for validation
    pub min_data_points: usize,
}

impl Default for AssetValidationRules {
    fn default() -> Self {
        Self {
            asset_type: "generic".to_string(),
            bounds: MarketRangeBounds::default(),
            enable_flash_crash_detection: true,
            enable_manipulation_detection: true,
            min_data_points: 10,
        }
    }
}

/// Validation severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ValidationSeverity {
    /// Critical error - must halt processing
    Critical,
    /// Error - should reject data
    Error,
    /// Warning - proceed with caution
    Warning,
    /// Info - informational only
    Info,
}

/// Validation issue with context
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ValidationIssue {
    pub severity: ValidationSeverity,
    pub code: String,
    pub message: String,
    pub value: Option<Float>,
    pub expected_range: Option<(Float, Float)>,
    pub data_point_index: Option<usize>,
    pub timestamp: Option<i64>,
}

/// Comprehensive validation report
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ValidationReport {
    pub issues: Vec<ValidationIssue>,
    pub is_valid: bool,
    pub critical_errors: usize,
    pub errors: usize,
    pub warnings: usize,
    pub data_points_validated: usize,
    pub flash_crashes_detected: usize,
    pub manipulation_patterns_detected: usize,
}

impl Default for ValidationReport {
    fn default() -> Self {
        Self {
            issues: Vec::new(),
            is_valid: true,
            critical_errors: 0,
            errors: 0,
            warnings: 0,
            data_points_validated: 0,
            flash_crashes_detected: 0,
            manipulation_patterns_detected: 0,
        }
    }
}

impl ValidationReport {
    /// Add a validation issue
    pub fn add_issue(&mut self, issue: ValidationIssue) {
        match issue.severity {
            ValidationSeverity::Critical => {
                self.critical_errors += 1;
                self.is_valid = false;
            }
            ValidationSeverity::Error => {
                self.errors += 1;
                self.is_valid = false;
            }
            ValidationSeverity::Warning => self.warnings += 1,
            ValidationSeverity::Info => {}
        }
        self.issues.push(issue);
    }

    /// Check if validation passed
    pub fn passed(&self) -> bool {
        self.is_valid && self.critical_errors == 0 && self.errors == 0
    }

    /// Get only critical and error issues
    pub fn get_blocking_issues(&self) -> Vec<&ValidationIssue> {
        self.issues
            .iter()
            .filter(|issue| matches!(issue.severity, ValidationSeverity::Critical | ValidationSeverity::Error))
            .collect()
    }
}

/// Circuit breaker for extreme values
#[derive(Debug, Clone)]
pub struct CircuitBreaker {
    /// Maximum allowed consecutive anomalies
    pub max_consecutive_anomalies: usize,
    /// Current consecutive anomaly count
    pub consecutive_anomalies: usize,
    /// Whether circuit breaker is tripped
    pub is_tripped: bool,
    /// Anomaly detection threshold (number of standard deviations)
    pub anomaly_threshold: Float,
}

impl Default for CircuitBreaker {
    fn default() -> Self {
        Self {
            max_consecutive_anomalies: 5,
            consecutive_anomalies: 0,
            is_tripped: false,
            anomaly_threshold: 6.0, // 6-sigma events
        }
    }
}

impl CircuitBreaker {
    /// Check if value triggers circuit breaker
    pub fn check_value(&mut self, value: Float, mean: Float, std_dev: Float) -> bool {
        let z_score = (value - mean).abs() / std_dev;
        
        if z_score > self.anomaly_threshold {
            self.consecutive_anomalies += 1;
            if self.consecutive_anomalies >= self.max_consecutive_anomalies {
                self.is_tripped = true;
            }
        } else {
            self.consecutive_anomalies = 0;
        }
        
        self.is_tripped
    }

    /// Reset circuit breaker
    pub fn reset(&mut self) {
        self.consecutive_anomalies = 0;
        self.is_tripped = false;
    }
}

/// Main financial data validator
#[derive(Debug, Clone)]
pub struct FinancialValidator {
    /// Validation rules by asset type
    pub rules: HashMap<String, AssetValidationRules>,
    /// Circuit breaker for extreme values
    pub circuit_breaker: CircuitBreaker,
    /// Enable strict validation mode
    pub strict_mode: bool,
}

impl Default for FinancialValidator {
    fn default() -> Self {
        let mut rules = HashMap::new();
        
        // Add default rules for common asset types
        rules.insert("stock".to_string(), AssetValidationRules {
            asset_type: "stock".to_string(),
            bounds: MarketRangeBounds {
                min_historical_price: 0.01,
                max_historical_price: 1e6,
                min_historical_volume: 0.0,
                max_historical_volume: 1e11,
                max_historical_change: 5.0, // 500% max daily change for stocks
            },
            ..Default::default()
        });
        
        rules.insert("forex".to_string(), AssetValidationRules {
            asset_type: "forex".to_string(),
            bounds: MarketRangeBounds {
                min_historical_price: 1e-6,
                max_historical_price: 1e3,
                min_historical_volume: 0.0,
                max_historical_volume: 1e13,
                max_historical_change: 0.5, // 50% max daily change for forex
            },
            ..Default::default()
        });
        
        rules.insert("crypto".to_string(), AssetValidationRules {
            asset_type: "crypto".to_string(),
            bounds: MarketRangeBounds {
                min_historical_price: 1e-12,
                max_historical_price: 1e8,
                min_historical_volume: 0.0,
                max_historical_volume: 1e12,
                max_historical_change: 20.0, // 2000% max daily change for crypto
            },
            ..Default::default()
        });
        
        rules.insert("commodity".to_string(), AssetValidationRules {
            asset_type: "commodity".to_string(),
            bounds: MarketRangeBounds {
                min_historical_price: 0.001,
                max_historical_price: 1e6,
                min_historical_volume: 0.0,
                max_historical_volume: 1e10,
                max_historical_change: 2.0, // 200% max daily change for commodities
            },
            ..Default::default()
        });
        
        Self {
            rules,
            circuit_breaker: CircuitBreaker::default(),
            strict_mode: true,
        }
    }
}

impl FinancialValidator {
    /// Create a new validator with default rules
    pub fn new() -> Self {
        Self::default()
    }

    /// Add custom validation rules for an asset type
    pub fn add_asset_rules(&mut self, asset_type: String, rules: AssetValidationRules) {
        self.rules.insert(asset_type, rules);
    }

    /// Enable or disable strict mode
    pub fn set_strict_mode(&mut self, strict: bool) {
        self.strict_mode = strict;
    }

    /// Validate a single price value
    pub fn validate_price(&self, price: Float, asset_type: &str) -> Result<()> {
        // Check for NaN and Infinite values
        if !price.is_finite() {
            return Err(CdfaError::invalid_input(format!(
                "Price contains invalid value: {} (NaN or Infinite)", price
            )));
        }

        // Check for negative prices
        if price < 0.0 {
            return Err(CdfaError::invalid_input(format!(
                "Price cannot be negative: {}", price
            )));
        }

        // Check for zero or extremely small prices
        if price < MIN_PRICE {
            return Err(CdfaError::invalid_input(format!(
                "Price {} is below minimum threshold {}", price, MIN_PRICE
            )));
        }

        // Check for unreasonably large values
        if price > MAX_FINANCIAL_VALUE {
            return Err(CdfaError::invalid_input(format!(
                "Price {} exceeds maximum reasonable value {}", price, MAX_FINANCIAL_VALUE
            )));
        }

        // Asset-specific validation
        if let Some(rules) = self.rules.get(asset_type) {
            if price < rules.bounds.min_historical_price {
                return Err(CdfaError::invalid_input(format!(
                    "Price {} below historical minimum {} for asset type {}", 
                    price, rules.bounds.min_historical_price, asset_type
                )));
            }

            if price > rules.bounds.max_historical_price {
                return Err(CdfaError::invalid_input(format!(
                    "Price {} above historical maximum {} for asset type {}", 
                    price, rules.bounds.max_historical_price, asset_type
                )));
            }
        }

        Ok(())
    }

    /// Validate a single volume value
    pub fn validate_volume(&self, volume: Float, asset_type: &str) -> Result<()> {
        // Check for NaN and Infinite values
        if !volume.is_finite() {
            return Err(CdfaError::invalid_input(format!(
                "Volume contains invalid value: {} (NaN or Infinite)", volume
            )));
        }

        // Check for negative volume
        if volume < 0.0 {
            return Err(CdfaError::invalid_input(format!(
                "Volume cannot be negative: {}", volume
            )));
        }

        // Check for unreasonably large values
        if volume > MAX_REASONABLE_VOLUME {
            return Err(CdfaError::invalid_input(format!(
                "Volume {} exceeds maximum reasonable value {}", volume, MAX_REASONABLE_VOLUME
            )));
        }

        // Asset-specific validation
        if let Some(rules) = self.rules.get(asset_type) {
            if volume > rules.bounds.max_historical_volume {
                return Err(CdfaError::invalid_input(format!(
                    "Volume {} above historical maximum {} for asset type {}", 
                    volume, rules.bounds.max_historical_volume, asset_type
                )));
            }
        }

        Ok(())
    }

    /// Validate timestamp monotonicity and reasonable dates
    pub fn validate_timestamps(&self, timestamps: &[i64]) -> Result<()> {
        if timestamps.is_empty() {
            return Err(CdfaError::invalid_input("Timestamp array cannot be empty"));
        }

        // Check for reasonable date range (between 1970 and 2100)
        let min_timestamp = 0i64; // 1970-01-01
        let max_timestamp = 4102444800000i64; // 2100-01-01 in milliseconds

        for (i, &timestamp) in timestamps.iter().enumerate() {
            if timestamp < min_timestamp || timestamp > max_timestamp {
                return Err(CdfaError::invalid_input(format!(
                    "Timestamp {} at index {} is outside reasonable range [{}, {}]",
                    timestamp, i, min_timestamp, max_timestamp
                )));
            }
        }

        // Check monotonicity
        for i in 1..timestamps.len() {
            if timestamps[i] <= timestamps[i - 1] {
                return Err(CdfaError::invalid_input(format!(
                    "Timestamps not monotonic: timestamp[{}]={} <= timestamp[{}]={}",
                    i, timestamps[i], i - 1, timestamps[i - 1]
                )));
            }
        }

        Ok(())
    }

    /// Validate percentage values (0-100 or -100 to 100)
    pub fn validate_percentage(&self, percentage: Float, allow_negative: bool) -> Result<()> {
        if !percentage.is_finite() {
            return Err(CdfaError::invalid_input(format!(
                "Percentage contains invalid value: {}", percentage
            )));
        }

        let min_val = if allow_negative { -100.0 } else { 0.0 };
        let max_val = 100.0;

        if percentage < min_val || percentage > max_val {
            return Err(CdfaError::invalid_input(format!(
                "Percentage {} outside valid range [{}, {}]", percentage, min_val, max_val
            )));
        }

        Ok(())
    }

    /// Validate correlation values (-1 to 1)
    pub fn validate_correlation(&self, correlation: Float) -> Result<()> {
        if !correlation.is_finite() {
            return Err(CdfaError::invalid_input(format!(
                "Correlation contains invalid value: {}", correlation
            )));
        }

        if correlation < MIN_CORRELATION || correlation > MAX_CORRELATION {
            return Err(CdfaError::invalid_input(format!(
                "Correlation {} outside valid range [{}, {}]", 
                correlation, MIN_CORRELATION, MAX_CORRELATION
            )));
        }

        Ok(())
    }

    /// Validate price series for flash crashes and anomalies
    pub fn validate_price_series(&mut self, prices: &[Float], asset_type: &str) -> ValidationReport {
        let mut report = ValidationReport::default();
        report.data_points_validated = prices.len();

        if prices.is_empty() {
            report.add_issue(ValidationIssue {
                severity: ValidationSeverity::Critical,
                code: "EMPTY_SERIES".to_string(),
                message: "Price series cannot be empty".to_string(),
                value: None,
                expected_range: None,
                data_point_index: None,
                timestamp: None,
            });
            return report;
        }

        // Get validation rules
        let rules = self.rules.get(asset_type).cloned().unwrap_or_default();

        // Validate individual prices
        for (i, &price) in prices.iter().enumerate() {
            if let Err(e) = self.validate_price(price, asset_type) {
                report.add_issue(ValidationIssue {
                    severity: ValidationSeverity::Error,
                    code: "INVALID_PRICE".to_string(),
                    message: e.to_string(),
                    value: Some(price),
                    expected_range: Some((rules.bounds.min_historical_price, rules.bounds.max_historical_price)),
                    data_point_index: Some(i),
                    timestamp: None,
                });
            }
        }

        // Flash crash detection
        if rules.enable_flash_crash_detection && prices.len() > 1 {
            for i in 1..prices.len() {
                let change_ratio = (prices[i] - prices[i - 1]) / prices[i - 1];
                
                if change_ratio <= FLASH_CRASH_THRESHOLD {
                    report.flash_crashes_detected += 1;
                    report.add_issue(ValidationIssue {
                        severity: ValidationSeverity::Critical,
                        code: "FLASH_CRASH".to_string(),
                        message: format!(
                            "Flash crash detected: {} drop from {} to {}", 
                            (change_ratio * 100.0).round(), prices[i - 1], prices[i]
                        ),
                        value: Some(change_ratio),
                        expected_range: Some((FLASH_CRASH_THRESHOLD, rules.bounds.max_historical_change)),
                        data_point_index: Some(i),
                        timestamp: None,
                    });
                }

                if change_ratio >= FLASH_SPIKE_THRESHOLD {
                    report.add_issue(ValidationIssue {
                        severity: ValidationSeverity::Critical,
                        code: "FLASH_SPIKE".to_string(),
                        message: format!(
                            "Flash spike detected: {} increase from {} to {}", 
                            (change_ratio * 100.0).round(), prices[i - 1], prices[i]
                        ),
                        value: Some(change_ratio),
                        expected_range: Some((-rules.bounds.max_historical_change, rules.bounds.max_historical_change)),
                        data_point_index: Some(i),
                        timestamp: None,
                    });
                }

                if change_ratio.abs() > rules.bounds.max_historical_change {
                    report.add_issue(ValidationIssue {
                        severity: ValidationSeverity::Error,
                        code: "EXCESSIVE_CHANGE".to_string(),
                        message: format!(
                            "Price change {} exceeds maximum historical change {} for asset type {}", 
                            change_ratio, rules.bounds.max_historical_change, asset_type
                        ),
                        value: Some(change_ratio),
                        expected_range: Some((-rules.bounds.max_historical_change, rules.bounds.max_historical_change)),
                        data_point_index: Some(i),
                        timestamp: None,
                    });
                }
            }
        }

        // Circuit breaker check
        if prices.len() >= 10 {
            let mean = prices.iter().sum::<Float>() / prices.len() as Float;
            let variance = prices.iter().map(|x| (x - mean).powi(2)).sum::<Float>() / prices.len() as Float;
            let std_dev = variance.sqrt();

            for (i, &price) in prices.iter().enumerate() {
                if self.circuit_breaker.check_value(price, mean, std_dev) {
                    report.add_issue(ValidationIssue {
                        severity: ValidationSeverity::Critical,
                        code: "CIRCUIT_BREAKER".to_string(),
                        message: "Circuit breaker triggered due to excessive anomalies".to_string(),
                        value: Some(price),
                        expected_range: Some((mean - 6.0 * std_dev, mean + 6.0 * std_dev)),
                        data_point_index: Some(i),
                        timestamp: None,
                    });
                    break;
                }
            }
        }

        // Data manipulation pattern detection
        if rules.enable_manipulation_detection {
            self.detect_manipulation_patterns(&mut report, prices);
        }

        report
    }

    /// Detect potential data manipulation patterns
    fn detect_manipulation_patterns(&self, report: &mut ValidationReport, prices: &[Float]) {
        if prices.len() < 10 {
            return;
        }

        // Check for unrealistic stability (prices don't change for too long)
        let mut stable_count = 0;
        let max_stable_periods = 20;
        
        for i in 1..prices.len() {
            if (prices[i] - prices[i - 1]).abs() < 1e-10 {
                stable_count += 1;
                if stable_count >= max_stable_periods {
                    report.manipulation_patterns_detected += 1;
                    report.add_issue(ValidationIssue {
                        severity: ValidationSeverity::Warning,
                        code: "SUSPICIOUS_STABILITY".to_string(),
                        message: format!("Suspiciously stable prices for {} consecutive periods", stable_count),
                        value: Some(prices[i]),
                        expected_range: None,
                        data_point_index: Some(i),
                        timestamp: None,
                    });
                    break;
                }
            } else {
                stable_count = 0;
            }
        }

        // Check for unrealistic patterns (e.g., perfect sine waves)
        let mut pattern_score = 0.0;
        for i in 2..prices.len() {
            let change1 = prices[i - 1] - prices[i - 2];
            let change2 = prices[i] - prices[i - 1];
            
            // Check for alternating pattern
            if change1 * change2 < 0.0 && change1.abs() == change2.abs() {
                pattern_score += 1.0;
            }
        }

        let pattern_ratio = pattern_score / (prices.len() - 2) as Float;
        if pattern_ratio > 0.8 {
            report.manipulation_patterns_detected += 1;
            report.add_issue(ValidationIssue {
                severity: ValidationSeverity::Warning,
                code: "ARTIFICIAL_PATTERN".to_string(),
                message: format!("Suspiciously regular pattern detected (score: {:.2})", pattern_ratio),
                value: Some(pattern_ratio),
                expected_range: Some((0.0, 0.3)),
                data_point_index: None,
                timestamp: None,
            });
        }
    }

    /// Comprehensive market data validation
    pub fn validate_market_data(
        &mut self,
        timestamps: &[i64],
        open: &[Float],
        high: &[Float],
        low: &[Float],
        close: &[Float],
        volume: &[Float],
        asset_type: &str,
    ) -> ValidationReport {
        let mut report = ValidationReport::default();

        // Check array lengths match
        let arrays = [("open", open.len()), ("high", high.len()), ("low", low.len()), 
                     ("close", close.len()), ("volume", volume.len()), ("timestamps", timestamps.len())];
        
        let expected_len = timestamps.len();
        for (name, len) in &arrays {
            if *len != expected_len {
                report.add_issue(ValidationIssue {
                    severity: ValidationSeverity::Critical,
                    code: "LENGTH_MISMATCH".to_string(),
                    message: format!("Array {} has length {} but expected {}", name, len, expected_len),
                    value: Some(*len as Float),
                    expected_range: Some((expected_len as Float, expected_len as Float)),
                    data_point_index: None,
                    timestamp: None,
                });
            }
        }

        if !report.passed() {
            return report;
        }

        // Validate timestamps
        if let Err(e) = self.validate_timestamps(timestamps) {
            report.add_issue(ValidationIssue {
                severity: ValidationSeverity::Critical,
                code: "INVALID_TIMESTAMPS".to_string(),
                message: e.to_string(),
                value: None,
                expected_range: None,
                data_point_index: None,
                timestamp: None,
            });
        }

        report.data_points_validated = timestamps.len();

        // Validate OHLCV relationships and individual values
        for i in 0..timestamps.len() {
            let timestamp = timestamps[i];

            // Validate individual prices
            for (name, &price) in [("open", open[i]), ("high", high[i]), ("low", low[i]), ("close", close[i])].iter() {
                if let Err(e) = self.validate_price(price, asset_type) {
                    report.add_issue(ValidationIssue {
                        severity: ValidationSeverity::Error,
                        code: "INVALID_PRICE".to_string(),
                        message: format!("{}: {}", name, e),
                        value: Some(price),
                        expected_range: None,
                        data_point_index: Some(i),
                        timestamp: Some(timestamp),
                    });
                }
            }

            // Validate volume
            if let Err(e) = self.validate_volume(volume[i], asset_type) {
                report.add_issue(ValidationIssue {
                    severity: ValidationSeverity::Error,
                    code: "INVALID_VOLUME".to_string(),
                    message: e.to_string(),
                    value: Some(volume[i]),
                    expected_range: None,
                    data_point_index: Some(i),
                    timestamp: Some(timestamp),
                });
            }

            // OHLC relationship validation
            if low[i] > high[i] {
                report.add_issue(ValidationIssue {
                    severity: ValidationSeverity::Critical,
                    code: "LOW_GREATER_THAN_HIGH".to_string(),
                    message: format!("Low price {} greater than high price {}", low[i], high[i]),
                    value: Some(low[i]),
                    expected_range: Some((0.0, high[i])),
                    data_point_index: Some(i),
                    timestamp: Some(timestamp),
                });
            }

            if open[i] < low[i] || open[i] > high[i] {
                report.add_issue(ValidationIssue {
                    severity: ValidationSeverity::Error,
                    code: "OPEN_OUTSIDE_RANGE".to_string(),
                    message: format!("Open price {} outside low-high range [{}, {}]", open[i], low[i], high[i]),
                    value: Some(open[i]),
                    expected_range: Some((low[i], high[i])),
                    data_point_index: Some(i),
                    timestamp: Some(timestamp),
                });
            }

            if close[i] < low[i] || close[i] > high[i] {
                report.add_issue(ValidationIssue {
                    severity: ValidationSeverity::Error,
                    code: "CLOSE_OUTSIDE_RANGE".to_string(),
                    message: format!("Close price {} outside low-high range [{}, {}]", close[i], low[i], high[i]),
                    value: Some(close[i]),
                    expected_range: Some((low[i], high[i])),
                    data_point_index: Some(i),
                    timestamp: Some(timestamp),
                });
            }
        }

        // Validate price series for patterns and anomalies
        let close_report = self.validate_price_series(close, asset_type);
        for issue in close_report.issues {
            report.add_issue(issue);
        }
        report.flash_crashes_detected += close_report.flash_crashes_detected;
        report.manipulation_patterns_detected += close_report.manipulation_patterns_detected;

        report
    }
}

/// Convenience macros for financial validation

/// Validate that a price is positive and finite
#[macro_export]
macro_rules! validate_price {
    ($price:expr) => {
        if !$price.is_finite() || $price <= 0.0 {
            return Err(CdfaError::invalid_input(format!("Invalid price: {}", $price)));
        }
    };
}

/// Validate that a volume is non-negative and finite
#[macro_export]
macro_rules! validate_volume {
    ($volume:expr) => {
        if !$volume.is_finite() || $volume < 0.0 {
            return Err(CdfaError::invalid_input(format!("Invalid volume: {}", $volume)));
        }
    };
}

/// Validate that a percentage is within bounds
#[macro_export]
macro_rules! validate_percentage {
    ($percentage:expr, $allow_negative:expr) => {
        let min_val = if $allow_negative { -100.0 } else { 0.0 };
        if !$percentage.is_finite() || $percentage < min_val || $percentage > 100.0 {
            return Err(CdfaError::invalid_input(format!("Invalid percentage: {}", $percentage)));
        }
    };
}

/// Validate that a correlation is between -1 and 1
#[macro_export]
macro_rules! validate_correlation {
    ($correlation:expr) => {
        if !$correlation.is_finite() || $correlation < -1.0 || $correlation > 1.0 {
            return Err(CdfaError::invalid_input(format!("Invalid correlation: {}", $correlation)));
        }
    };
}

/// Validate that a value is finite and within reasonable financial bounds
#[macro_export]
macro_rules! validate_financial_value {
    ($value:expr) => {
        if !$value.is_finite() || $value.abs() > MAX_FINANCIAL_VALUE {
            return Err(CdfaError::invalid_input(format!("Invalid financial value: {}", $value)));
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_price_validation() {
        let validator = FinancialValidator::new();

        // Valid prices
        assert!(validator.validate_price(100.0, "stock").is_ok());
        assert!(validator.validate_price(0.01, "stock").is_ok());

        // Invalid prices
        assert!(validator.validate_price(-10.0, "stock").is_err()); // Negative
        assert!(validator.validate_price(0.0, "stock").is_err()); // Zero
        assert!(validator.validate_price(Float::NAN, "stock").is_err()); // NaN
        assert!(validator.validate_price(Float::INFINITY, "stock").is_err()); // Infinite
        assert!(validator.validate_price(1e20, "stock").is_err()); // Too large
    }

    #[test]
    fn test_volume_validation() {
        let validator = FinancialValidator::new();

        // Valid volumes
        assert!(validator.validate_volume(1000.0, "stock").is_ok());
        assert!(validator.validate_volume(0.0, "stock").is_ok()); // Zero volume allowed

        // Invalid volumes
        assert!(validator.validate_volume(-100.0, "stock").is_err()); // Negative
        assert!(validator.validate_volume(Float::NAN, "stock").is_err()); // NaN
        assert!(validator.validate_volume(Float::INFINITY, "stock").is_err()); // Infinite
        assert!(validator.validate_volume(1e20, "stock").is_err()); // Too large
    }

    #[test]
    fn test_timestamp_validation() {
        let validator = FinancialValidator::new();

        // Valid timestamps (ascending order)
        let valid_timestamps = vec![1000, 2000, 3000, 4000];
        assert!(validator.validate_timestamps(&valid_timestamps).is_ok());

        // Invalid timestamps (not monotonic)
        let invalid_timestamps = vec![1000, 3000, 2000, 4000];
        assert!(validator.validate_timestamps(&invalid_timestamps).is_err());

        // Invalid timestamps (too old)
        let old_timestamps = vec![-1000, 1000, 2000];
        assert!(validator.validate_timestamps(&old_timestamps).is_err());
    }

    #[test]
    fn test_percentage_validation() {
        let validator = FinancialValidator::new();

        // Valid percentages
        assert!(validator.validate_percentage(50.0, false).is_ok());
        assert!(validator.validate_percentage(-10.0, true).is_ok());

        // Invalid percentages
        assert!(validator.validate_percentage(-10.0, false).is_err()); // Negative not allowed
        assert!(validator.validate_percentage(150.0, false).is_err()); // Too large
        assert!(validator.validate_percentage(Float::NAN, false).is_err()); // NaN
    }

    #[test]
    fn test_correlation_validation() {
        let validator = FinancialValidator::new();

        // Valid correlations
        assert!(validator.validate_correlation(0.5).is_ok());
        assert!(validator.validate_correlation(-0.8).is_ok());
        assert!(validator.validate_correlation(1.0).is_ok());
        assert!(validator.validate_correlation(-1.0).is_ok());

        // Invalid correlations
        assert!(validator.validate_correlation(1.5).is_err()); // Too large
        assert!(validator.validate_correlation(-1.5).is_err()); // Too small
        assert!(validator.validate_correlation(Float::NAN).is_err()); // NaN
    }

    #[test]
    fn test_flash_crash_detection() {
        let mut validator = FinancialValidator::new();
        
        // Normal price series
        let normal_prices = vec![100.0, 101.0, 99.0, 102.0, 98.0];
        let report = validator.validate_price_series(&normal_prices, "stock");
        assert_eq!(report.flash_crashes_detected, 0);

        // Flash crash series
        let crash_prices = vec![100.0, 101.0, 10.0, 102.0]; // 90% drop
        let report = validator.validate_price_series(&crash_prices, "stock");
        assert!(report.flash_crashes_detected > 0);
    }

    #[test]
    fn test_circuit_breaker() {
        let mut breaker = CircuitBreaker::default();

        // Normal values shouldn't trip breaker
        let mean = 100.0;
        let std_dev = 10.0;
        assert!(!breaker.check_value(105.0, mean, std_dev));
        assert!(!breaker.check_value(95.0, mean, std_dev));

        // Extreme values should eventually trip breaker
        for _ in 0..10 {
            breaker.check_value(200.0, mean, std_dev); // 10 sigma event
        }
        assert!(breaker.is_tripped);
    }

    #[test]
    fn test_market_data_validation() {
        let mut validator = FinancialValidator::new();

        let timestamps = vec![1000, 2000, 3000];
        let open = vec![100.0, 101.0, 102.0];
        let high = vec![105.0, 106.0, 107.0];
        let low = vec![95.0, 96.0, 97.0];
        let close = vec![102.0, 103.0, 104.0];
        let volume = vec![1000.0, 1100.0, 1200.0];

        let report = validator.validate_market_data(
            &timestamps, &open, &high, &low, &close, &volume, "stock"
        );

        assert!(report.passed());

        // Test with invalid OHLC relationship
        let invalid_low = vec![110.0, 96.0, 97.0]; // Low > High for first candle
        let report = validator.validate_market_data(
            &timestamps, &open, &high, &invalid_low, &close, &volume, "stock"
        );

        assert!(!report.passed());
        assert!(report.critical_errors > 0);
    }

    #[test]
    fn test_manipulation_detection() {
        let mut validator = FinancialValidator::new();

        // Artificially stable prices
        let stable_prices: Vec<Float> = vec![100.0; 25];
        let report = validator.validate_price_series(&stable_prices, "stock");
        assert!(report.manipulation_patterns_detected > 0);

        // Artificially regular alternating pattern
        let mut alternating_prices = Vec::new();
        for i in 0..20 {
            alternating_prices.push(if i % 2 == 0 { 100.0 } else { 101.0 });
        }
        let report = validator.validate_price_series(&alternating_prices, "stock");
        assert!(report.manipulation_patterns_detected > 0);
    }

    #[test]
    fn test_asset_specific_validation() {
        let validator = FinancialValidator::new();

        // Test crypto-specific bounds (wider range)
        assert!(validator.validate_price(1e-10, "crypto").is_ok());
        assert!(validator.validate_price(1e-10, "stock").is_err());

        // Test forex-specific bounds
        assert!(validator.validate_price(0.0001, "forex").is_ok());
        assert!(validator.validate_price(0.0001, "stock").is_err());
    }

    #[test]
    fn test_validation_macros() {
        // Test price validation macro
        let test_price = |price: Float| -> Result<()> {
            validate_price!(price);
            Ok(())
        };

        assert!(test_price(100.0).is_ok());
        assert!(test_price(-10.0).is_err());
        assert!(test_price(Float::NAN).is_err());

        // Test volume validation macro
        let test_volume = |volume: Float| -> Result<()> {
            validate_volume!(volume);
            Ok(())
        };

        assert!(test_volume(1000.0).is_ok());
        assert!(test_volume(-100.0).is_err());

        // Test correlation validation macro
        let test_correlation = |corr: Float| -> Result<()> {
            validate_correlation!(corr);
            Ok(())
        };

        assert!(test_correlation(0.5).is_ok());
        assert!(test_correlation(1.5).is_err());
    }
}