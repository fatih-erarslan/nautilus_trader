//! Comprehensive validation framework for financial data
//!
//! This module provides robust validation for financial market data, including
//! input validation, range checking, anomaly detection, and circuit breakers
//! to prevent invalid data from corrupting financial calculations.

pub mod financial;
pub mod utils;
pub mod integration_example;

use crate::error::{CdfaError, Result};
use crate::types::Float;

// Re-export main types for convenience
pub use financial::{
    FinancialValidator, ValidationReport, ValidationIssue, ValidationSeverity,
    AssetValidationRules, MarketRangeBounds, CircuitBreaker,
    MAX_FINANCIAL_VALUE, MIN_PRICE, MAX_DAILY_CHANGE_RATIO, MAX_REASONABLE_VOLUME,
    MIN_CORRELATION, MAX_CORRELATION, FLASH_CRASH_THRESHOLD, FLASH_SPIKE_THRESHOLD,
};

/// Quick validation functions for common use cases

/// Validate basic financial constraints for a single value
pub fn validate_financial_value(value: Float, name: &str) -> Result<()> {
    if !value.is_finite() {
        return Err(CdfaError::invalid_input(format!(
            "{} contains invalid value: {} (NaN or Infinite)", name, value
        )));
    }

    if value.abs() > MAX_FINANCIAL_VALUE {
        return Err(CdfaError::invalid_input(format!(
            "{} {} exceeds maximum reasonable value {}", name, value, MAX_FINANCIAL_VALUE
        )));
    }

    Ok(())
}

/// Validate that arrays have the same length
pub fn validate_array_lengths<T>(arrays: &[(&str, &[T])]) -> Result<()> {
    if arrays.is_empty() {
        return Ok(());
    }

    let expected_len = arrays[0].1.len();
    for (name, array) in arrays.iter().skip(1) {
        if array.len() != expected_len {
            return Err(CdfaError::invalid_input(format!(
                "Array {} has length {} but expected {} (based on first array '{}')",
                name, array.len(), expected_len, arrays[0].0
            )));
        }
    }

    Ok(())
}

/// Validate that an array is not empty
pub fn validate_not_empty<T>(array: &[T], name: &str) -> Result<()> {
    if array.is_empty() {
        return Err(CdfaError::invalid_input(format!("{} cannot be empty", name)));
    }
    Ok(())
}

/// Validate that a value is within a specified range
pub fn validate_range(value: Float, min: Float, max: Float, name: &str) -> Result<()> {
    validate_financial_value(value, name)?;
    
    if value < min || value > max {
        return Err(CdfaError::invalid_input(format!(
            "{} {} is outside valid range [{}, {}]", name, value, min, max
        )));
    }
    Ok(())
}

/// Validate integer overflow safety for volume calculations
pub fn validate_volume_calculation_safety(volume: Float, multiplier: Float) -> Result<()> {
    let result = volume * multiplier;
    
    if !result.is_finite() {
        return Err(CdfaError::invalid_input(format!(
            "Volume calculation would overflow: {} * {} = {}", volume, multiplier, result
        )));
    }

    if result > MAX_FINANCIAL_VALUE {
        return Err(CdfaError::invalid_input(format!(
            "Volume calculation result {} exceeds maximum safe value", result
        )));
    }

    Ok(())
}

/// Create a default financial validator for common use cases
pub fn create_default_validator() -> FinancialValidator {
    FinancialValidator::new()
}

/// Quick validation for OHLCV data
pub fn quick_validate_ohlcv(
    open: Float,
    high: Float,
    low: Float,
    close: Float,
    volume: Float,
) -> Result<()> {
    // Validate individual values
    validate_financial_value(open, "open")?;
    validate_financial_value(high, "high")?;
    validate_financial_value(low, "low")?;
    validate_financial_value(close, "close")?;
    validate_financial_value(volume, "volume")?;

    // Validate OHLC relationships
    if low > high {
        return Err(CdfaError::invalid_input(format!(
            "Low price {} cannot be greater than high price {}", low, high
        )));
    }

    if open < low || open > high {
        return Err(CdfaError::invalid_input(format!(
            "Open price {} must be between low {} and high {}", open, low, high
        )));
    }

    if close < low || close > high {
        return Err(CdfaError::invalid_input(format!(
            "Close price {} must be between low {} and high {}", close, low, high
        )));
    }

    if volume < 0.0 {
        return Err(CdfaError::invalid_input(format!(
            "Volume {} cannot be negative", volume
        )));
    }

    Ok(())
}

/// Validate market crash scenario data for testing
pub fn validate_crash_scenario_data(prices: &[Float]) -> Result<()> {
    validate_not_empty(prices, "crash scenario prices")?;

    let mut max_single_drop = 0.0;
    for i in 1..prices.len() {
        if prices[i - 1] > 0.0 {
            let change_ratio = (prices[i] - prices[i - 1]) / prices[i - 1];
            if change_ratio < max_single_drop {
                max_single_drop = change_ratio;
            }
        }
    }

    // Allow extreme drops for crash testing but validate data integrity
    if max_single_drop < -0.99 {
        // More than 99% drop - validate this is intentional test data
        for price in prices {
            validate_financial_value(*price, "crash scenario price")?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_financial_value() {
        assert!(validate_financial_value(100.0, "test").is_ok());
        assert!(validate_financial_value(Float::NAN, "test").is_err());
        assert!(validate_financial_value(Float::INFINITY, "test").is_err());
        assert!(validate_financial_value(1e20, "test").is_err());
    }

    #[test]
    fn test_validate_array_lengths() {
        let array1 = vec![1.0, 2.0, 3.0];
        let array2 = vec![4.0, 5.0, 6.0];
        let array3 = vec![7.0, 8.0]; // Different length

        let arrays = vec![("array1", array1.as_slice()), ("array2", array2.as_slice())];
        assert!(validate_array_lengths(&arrays).is_ok());

        let arrays = vec![("array1", array1.as_slice()), ("array3", array3.as_slice())];
        assert!(validate_array_lengths(&arrays).is_err());
    }

    #[test]
    fn test_validate_not_empty() {
        let non_empty = vec![1.0, 2.0, 3.0];
        let empty: Vec<Float> = vec![];

        assert!(validate_not_empty(&non_empty, "test").is_ok());
        assert!(validate_not_empty(&empty, "test").is_err());
    }

    #[test]
    fn test_validate_range() {
        assert!(validate_range(50.0, 0.0, 100.0, "test").is_ok());
        assert!(validate_range(-10.0, 0.0, 100.0, "test").is_err());
        assert!(validate_range(150.0, 0.0, 100.0, "test").is_err());
        assert!(validate_range(Float::NAN, 0.0, 100.0, "test").is_err());
    }

    #[test]
    fn test_quick_validate_ohlcv() {
        // Valid OHLCV
        assert!(quick_validate_ohlcv(100.0, 105.0, 95.0, 102.0, 1000.0).is_ok());

        // Invalid: low > high
        assert!(quick_validate_ohlcv(100.0, 95.0, 105.0, 102.0, 1000.0).is_err());

        // Invalid: open outside range
        assert!(quick_validate_ohlcv(110.0, 105.0, 95.0, 102.0, 1000.0).is_err());

        // Invalid: negative volume
        assert!(quick_validate_ohlcv(100.0, 105.0, 95.0, 102.0, -1000.0).is_err());
    }

    #[test]
    fn test_validate_volume_calculation_safety() {
        assert!(validate_volume_calculation_safety(1000.0, 100.0).is_ok());
        assert!(validate_volume_calculation_safety(1e10, 1e10).is_err()); // Would overflow
        assert!(validate_volume_calculation_safety(Float::INFINITY, 1.0).is_err());
    }

    #[test]
    fn test_validate_crash_scenario_data() {
        // Normal prices
        let normal_prices = vec![100.0, 99.0, 98.0, 97.0];
        assert!(validate_crash_scenario_data(&normal_prices).is_ok());

        // Extreme crash (but valid for testing)
        let crash_prices = vec![100.0, 50.0, 10.0, 1.0];
        assert!(validate_crash_scenario_data(&crash_prices).is_ok());

        // Invalid data (NaN)
        let invalid_prices = vec![100.0, Float::NAN, 10.0];
        assert!(validate_crash_scenario_data(&invalid_prices).is_err());

        // Empty array
        let empty_prices: Vec<Float> = vec![];
        assert!(validate_crash_scenario_data(&empty_prices).is_err());
    }

    #[test]
    fn test_create_default_validator() {
        let validator = create_default_validator();
        assert!(validator.validate_price(100.0, "stock").is_ok());
        assert!(validator.validate_price(-10.0, "stock").is_err());
    }
}