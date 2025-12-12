//! Integration example showing how to use validation in a FreqTrade strategy

use crate::validation::{FinancialValidator, ValidationSeverity, quick_validate_ohlcv};
use crate::types::Float;
use crate::error::Result;

/// Example FreqTrade strategy integration with comprehensive validation
pub struct ValidatedStrategy {
    validator: FinancialValidator,
    asset_type: String,
    strict_mode: bool,
    rejected_candles: usize,
    total_candles: usize,
}

impl ValidatedStrategy {
    /// Create a new validated strategy
    pub fn new(asset_type: String, strict_mode: bool) -> Self {
        let mut validator = FinancialValidator::new();
        validator.set_strict_mode(strict_mode);
        
        Self {
            validator,
            asset_type,
            strict_mode,
            rejected_candles: 0,
            total_candles: 0,
        }
    }

    /// Validate incoming OHLCV data before processing
    pub fn validate_candle(
        &mut self,
        timestamp: i64,
        open: Float,
        high: Float,
        low: Float,
        close: Float,
        volume: Float,
    ) -> Result<bool> {
        self.total_candles += 1;

        // Quick OHLCV validation
        if let Err(e) = quick_validate_ohlcv(open, high, low, close, volume) {
            self.rejected_candles += 1;
            log::warn!("Rejected candle at {}: {}", timestamp, e);
            return Ok(false);
        }

        // Additional price validation
        if let Err(e) = self.validator.validate_price(close, &self.asset_type) {
            self.rejected_candles += 1;
            log::warn!("Rejected close price {} at {}: {}", close, timestamp, e);
            return Ok(false);
        }

        // Volume validation
        if let Err(e) = self.validator.validate_volume(volume, &self.asset_type) {
            self.rejected_candles += 1;
            log::warn!("Rejected volume {} at {}: {}", volume, timestamp, e);
            return Ok(false);
        }

        Ok(true)
    }

    /// Validate a complete OHLCV dataset
    pub fn validate_dataset(
        &mut self,
        timestamps: &[i64],
        open: &[Float],
        high: &[Float],
        low: &[Float],
        close: &[Float],
        volume: &[Float],
    ) -> Result<ValidationReport> {
        let report = self.validator.validate_market_data(
            timestamps, open, high, low, close, volume, &self.asset_type
        );

        log::info!(
            "Dataset validation completed: {} data points, {} errors, {} warnings, {} flash crashes",
            report.data_points_validated,
            report.errors,
            report.warnings,
            report.flash_crashes_detected
        );

        if !report.passed() {
            log::error!("Dataset validation failed with {} critical errors", report.critical_errors);
            for issue in report.get_blocking_issues() {
                log::error!("  {}: {}", issue.code, issue.message);
            }
        }

        Ok(ValidationReport::from(report))
    }

    /// Get validation statistics
    pub fn get_validation_stats(&self) -> ValidationStats {
        ValidationStats {
            total_candles: self.total_candles,
            rejected_candles: self.rejected_candles,
            rejection_rate: if self.total_candles > 0 {
                self.rejected_candles as f64 / self.total_candles as f64
            } else {
                0.0
            },
            strict_mode: self.strict_mode,
            asset_type: self.asset_type.clone(),
        }
    }

    /// Reset validation statistics
    pub fn reset_stats(&mut self) {
        self.rejected_candles = 0;
        self.total_candles = 0;
    }
}

/// Simplified validation report for strategy use
#[derive(Debug, Clone)]
pub struct ValidationReport {
    pub is_valid: bool,
    pub error_count: usize,
    pub warning_count: usize,
    pub flash_crashes: usize,
    pub data_points: usize,
    pub critical_issues: Vec<String>,
}

impl From<crate::validation::financial::ValidationReport> for ValidationReport {
    fn from(report: crate::validation::financial::ValidationReport) -> Self {
        let critical_issues = report.get_blocking_issues()
            .iter()
            .map(|issue| format!("{}: {}", issue.code, issue.message))
            .collect();

        Self {
            is_valid: report.passed(),
            error_count: report.errors,
            warning_count: report.warnings,
            flash_crashes: report.flash_crashes_detected,
            data_points: report.data_points_validated,
            critical_issues,
        }
    }
}

/// Validation statistics for monitoring
#[derive(Debug, Clone)]
pub struct ValidationStats {
    pub total_candles: usize,
    pub rejected_candles: usize,
    pub rejection_rate: f64,
    pub strict_mode: bool,
    pub asset_type: String,
}

/// Utility functions for FreqTrade integration
pub mod freqtrade_utils {
    use super::*;

    /// Validate typical cryptocurrency data
    pub fn validate_crypto_data(
        open: Float, high: Float, low: Float, close: Float, volume: Float
    ) -> Result<()> {
        quick_validate_ohlcv(open, high, low, close, volume)
    }

    /// Validate typical forex data
    pub fn validate_forex_data(
        open: Float, high: Float, low: Float, close: Float, volume: Float
    ) -> Result<()> {
        // Forex volumes can be zero or very small
        if volume < 0.0 {
            return Err(crate::error::CdfaError::invalid_input("Volume cannot be negative"));
        }
        
        // Use standard OHLC validation
        if low > high {
            return Err(crate::error::CdfaError::invalid_input("Low cannot be greater than high"));
        }
        
        if open < low || open > high {
            return Err(crate::error::CdfaError::invalid_input("Open must be between low and high"));
        }
        
        if close < low || close > high {
            return Err(crate::error::CdfaError::invalid_input("Close must be between low and high"));
        }

        Ok(())
    }

    /// Check if a price movement indicates a potential flash crash
    pub fn is_flash_crash(prev_price: Float, current_price: Float, threshold: Float) -> bool {
        if prev_price <= 0.0 {
            return false;
        }
        
        let change_ratio = (current_price - prev_price) / prev_price;
        change_ratio <= -threshold
    }

    /// Sanitize price data by removing obvious errors
    pub fn sanitize_price_data(prices: &mut Vec<Float>) -> usize {
        let original_len = prices.len();
        
        prices.retain(|&price| {
            price.is_finite() && price > 0.0 && price < crate::validation::MAX_FINANCIAL_VALUE
        });
        
        original_len - prices.len()
    }

    /// Convert validation report to FreqTrade log messages
    pub fn log_validation_issues(report: &ValidationReport, pair: &str) {
        if !report.is_valid {
            log::error!("Validation failed for {}: {} errors, {} warnings", 
                       pair, report.error_count, report.warning_count);
        }
        
        if report.flash_crashes > 0 {
            log::warn!("Detected {} flash crashes in {} data", report.flash_crashes, pair);
        }
        
        for issue in &report.critical_issues {
            log::error!("Critical issue in {}: {}", pair, issue);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validated_strategy_creation() {
        let strategy = ValidatedStrategy::new("crypto".to_string(), true);
        assert_eq!(strategy.asset_type, "crypto");
        assert!(strategy.strict_mode);
        assert_eq!(strategy.total_candles, 0);
        assert_eq!(strategy.rejected_candles, 0);
    }

    #[test]
    fn test_valid_candle_validation() {
        let mut strategy = ValidatedStrategy::new("stock".to_string(), false);
        
        let result = strategy.validate_candle(
            1640995200000, // timestamp
            100.0,         // open
            105.0,         // high
            95.0,          // low
            102.0,         // close
            10000.0        // volume
        );
        
        assert!(result.is_ok());
        assert!(result.unwrap());
        assert_eq!(strategy.total_candles, 1);
        assert_eq!(strategy.rejected_candles, 0);
    }

    #[test]
    fn test_invalid_candle_validation() {
        let mut strategy = ValidatedStrategy::new("stock".to_string(), false);
        
        // Invalid: low > high
        let result = strategy.validate_candle(
            1640995200000, // timestamp
            100.0,         // open
            95.0,          // high (invalid)
            105.0,         // low (invalid)
            102.0,         // close
            10000.0        // volume
        );
        
        assert!(result.is_ok());
        assert!(!result.unwrap()); // Should reject the candle
        assert_eq!(strategy.total_candles, 1);
        assert_eq!(strategy.rejected_candles, 1);
    }

    #[test]
    fn test_validation_stats() {
        let mut strategy = ValidatedStrategy::new("crypto".to_string(), true);
        
        // Add some valid and invalid candles
        let _ = strategy.validate_candle(1000, 100.0, 105.0, 95.0, 102.0, 1000.0); // Valid
        let _ = strategy.validate_candle(2000, 100.0, 95.0, 105.0, 102.0, 1000.0); // Invalid
        let _ = strategy.validate_candle(3000, 101.0, 106.0, 96.0, 103.0, 1100.0); // Valid
        
        let stats = strategy.get_validation_stats();
        assert_eq!(stats.total_candles, 3);
        assert_eq!(stats.rejected_candles, 1);
        assert!((stats.rejection_rate - 1.0/3.0).abs() < 1e-10);
        assert!(stats.strict_mode);
        assert_eq!(stats.asset_type, "crypto");
    }

    #[test]
    fn test_freqtrade_utils_crypto_validation() {
        // Valid crypto data
        assert!(freqtrade_utils::validate_crypto_data(
            50000.0, 52000.0, 48000.0, 51000.0, 100.0
        ).is_ok());
        
        // Invalid crypto data (negative volume)
        assert!(freqtrade_utils::validate_crypto_data(
            50000.0, 52000.0, 48000.0, 51000.0, -100.0
        ).is_err());
    }

    #[test]
    fn test_flash_crash_detection() {
        // Normal price change
        assert!(!freqtrade_utils::is_flash_crash(100.0, 95.0, 0.10)); // 5% drop, 10% threshold
        
        // Flash crash
        assert!(freqtrade_utils::is_flash_crash(100.0, 80.0, 0.15)); // 20% drop, 15% threshold
        
        // Edge case: zero previous price
        assert!(!freqtrade_utils::is_flash_crash(0.0, 95.0, 0.10));
    }

    #[test]
    fn test_price_data_sanitization() {
        let mut prices = vec![
            100.0,               // Valid
            Float::NAN,          // Invalid
            -50.0,               // Invalid
            Float::INFINITY,     // Invalid
            200.0,               // Valid
            1e20,                // Invalid (too large)
            0.0,                 // Invalid (zero)
            150.0,               // Valid
        ];
        
        let removed = freqtrade_utils::sanitize_price_data(&mut prices);
        
        assert_eq!(removed, 5); // Should remove 5 invalid prices
        assert_eq!(prices.len(), 3); // Should keep 3 valid prices
        assert!(prices.iter().all(|&p| p.is_finite() && p > 0.0));
    }

    #[test]
    fn test_validation_report_conversion() {
        use crate::validation::financial;
        
        let mut original_report = financial::ValidationReport::default();
        original_report.is_valid = false;
        original_report.errors = 2;
        original_report.warnings = 1;
        original_report.flash_crashes_detected = 1;
        original_report.data_points_validated = 100;
        
        let converted_report = ValidationReport::from(original_report);
        
        assert!(!converted_report.is_valid);
        assert_eq!(converted_report.error_count, 2);
        assert_eq!(converted_report.warning_count, 1);
        assert_eq!(converted_report.flash_crashes, 1);
        assert_eq!(converted_report.data_points, 100);
    }
}