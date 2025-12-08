//! Input validation for financial data
//!
//! This module provides comprehensive validation for all financial inputs
//! to prevent malicious data from causing trading errors or system failures.

use crate::{MarketData, TalebianRiskError};
use super::{MAX_PRICE, MAX_VOLUME, MAX_PERCENTAGE, MIN_POSITION_SIZE, MAX_POSITION_SIZE};

/// Validate market data for financial safety
pub fn validate_market_data(data: &MarketData) -> Result<(), TalebianRiskError> {
    // Check for NaN and Infinity in all numeric fields
    validate_finite_positive(data.price, "price")?;
    validate_finite_non_negative(data.volume, "volume")?;
    validate_finite_positive(data.bid, "bid")?;
    validate_finite_positive(data.ask, "ask")?;
    validate_finite_non_negative(data.bid_volume, "bid_volume")?;
    validate_finite_non_negative(data.ask_volume, "ask_volume")?;
    validate_finite_non_negative(data.volatility, "volatility")?;
    
    // Range validation
    if data.price > MAX_PRICE {
        return Err(TalebianRiskError::InvalidInput(format!(
            "Price {} exceeds maximum allowed {}", data.price, MAX_PRICE
        )));
    }
    
    if data.volume > MAX_VOLUME {
        return Err(TalebianRiskError::InvalidInput(format!(
            "Volume {} exceeds maximum allowed {}", data.volume, MAX_VOLUME
        )));
    }
    
    // Bid/Ask relationship validation
    if data.bid >= data.ask {
        return Err(TalebianRiskError::InvalidInput(
            "Bid price must be less than ask price".to_string()
        ));
    }
    
    // Spread validation (prevent manipulation through extreme spreads)
    let spread = (data.ask - data.bid) / data.price;
    if spread > 0.1 {
        return Err(TalebianRiskError::InvalidInput(
            "Bid-ask spread exceeds 10% - potential manipulation".to_string()
        ));
    }
    
    // Volatility sanity check
    if data.volatility > 1.0 {
        return Err(TalebianRiskError::InvalidInput(
            "Volatility exceeds 100% - potential data error".to_string()
        ));
    }
    
    // Validate returns array
    for (i, &return_val) in data.returns.iter().enumerate() {
        if return_val.is_nan() || return_val.is_infinite() {
            return Err(TalebianRiskError::InvalidInput(format!(
                "Return value at index {} is NaN or infinite", i
            )));
        }
        
        // Sanity check for extreme returns (more than 100% in one period)
        if return_val.abs() > 1.0 {
            return Err(TalebianRiskError::InvalidInput(format!(
                "Return value {} at index {} exceeds 100% - potential error", return_val, i
            )));
        }
    }
    
    // Validate volume history
    for (i, &volume) in data.volume_history.iter().enumerate() {
        validate_finite_non_negative(volume, &format!("volume_history[{}]", i))?;
        
        if volume > MAX_VOLUME {
            return Err(TalebianRiskError::InvalidInput(format!(
                "Volume history value {} at index {} exceeds maximum", volume, i
            )));
        }
    }
    
    Ok(())
}

/// Validate percentage values (0.0 to 1.0)
pub fn validate_percentage(value: f64, name: &str) -> Result<(), TalebianRiskError> {
    validate_finite_non_negative(value, name)?;
    
    if value > MAX_PERCENTAGE {
        return Err(TalebianRiskError::InvalidInput(format!(
            "{} value {} exceeds maximum percentage {}", name, value, MAX_PERCENTAGE
        )));
    }
    
    Ok(())
}

/// Validate position size (must be within safe trading bounds)
pub fn validate_position_size(size: f64) -> Result<(), TalebianRiskError> {
    validate_percentage(size, "position_size")?;
    
    if size < MIN_POSITION_SIZE {
        return Err(TalebianRiskError::InvalidInput(format!(
            "Position size {} below minimum {}", size, MIN_POSITION_SIZE
        )));
    }
    
    if size > MAX_POSITION_SIZE {
        return Err(TalebianRiskError::InvalidInput(format!(
            "Position size {} exceeds maximum allowed {}", size, MAX_POSITION_SIZE
        )));
    }
    
    Ok(())
}

/// Validate positive finite number
pub fn validate_positive(value: f64, name: &str) -> Result<(), TalebianRiskError> {
    validate_finite_positive(value, name)
}

/// Validate finite positive number (> 0)
fn validate_finite_positive(value: f64, name: &str) -> Result<(), TalebianRiskError> {
    if value.is_nan() {
        return Err(TalebianRiskError::InvalidInput(format!("{} is NaN", name)));
    }
    
    if value.is_infinite() {
        return Err(TalebianRiskError::InvalidInput(format!("{} is infinite", name)));
    }
    
    if value <= 0.0 {
        return Err(TalebianRiskError::InvalidInput(format!(
            "{} must be positive, got {}", name, value
        )));
    }
    
    Ok(())
}

/// Validate finite non-negative number (>= 0)
fn validate_finite_non_negative(value: f64, name: &str) -> Result<(), TalebianRiskError> {
    if value.is_nan() {
        return Err(TalebianRiskError::InvalidInput(format!("{} is NaN", name)));
    }
    
    if value.is_infinite() {
        return Err(TalebianRiskError::InvalidInput(format!("{} is infinite", name)));
    }
    
    if value < 0.0 {
        return Err(TalebianRiskError::InvalidInput(format!(
            "{} must be non-negative, got {}", name, value
        )));
    }
    
    Ok(())
}

/// Validate array bounds for safe access
pub fn validate_array_bounds<T>(array: &[T], index: usize, name: &str) -> Result<(), TalebianRiskError> {
    if index >= array.len() {
        return Err(TalebianRiskError::InvalidInput(format!(
            "Index {} out of bounds for {} (length {})", index, name, array.len()
        )));
    }
    
    Ok(())
}

/// Validate array is not empty
pub fn validate_array_not_empty<T>(array: &[T], name: &str) -> Result<(), TalebianRiskError> {
    if array.is_empty() {
        return Err(TalebianRiskError::InvalidInput(format!("{} array is empty", name)));
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{DateTime, Utc};

    fn create_valid_market_data() -> MarketData {
        MarketData {
            timestamp: DateTime::from_timestamp(1640995200, 0).unwrap_or_default(),
            timestamp_unix: 1640995200,
            price: 50000.0,
            volume: 1000.0,
            bid: 49990.0,
            ask: 50010.0,
            bid_volume: 500.0,
            ask_volume: 400.0,
            volatility: 0.03,
            returns: vec![0.01, 0.015, -0.005, 0.02, 0.008],
            volume_history: vec![800.0, 900.0, 1200.0, 950.0, 1000.0],
        }
    }

    #[test]
    fn test_validate_market_data_valid() {
        let data = create_valid_market_data();
        assert!(validate_market_data(&data).is_ok());
    }

    #[test]
    fn test_validate_market_data_nan_price() {
        let mut data = create_valid_market_data();
        data.price = f64::NAN;
        assert!(validate_market_data(&data).is_err());
    }

    #[test]
    fn test_validate_market_data_infinite_volume() {
        let mut data = create_valid_market_data();
        data.volume = f64::INFINITY;
        assert!(validate_market_data(&data).is_err());
    }

    #[test]
    fn test_validate_market_data_invalid_bid_ask() {
        let mut data = create_valid_market_data();
        data.bid = 50020.0; // Higher than ask
        assert!(validate_market_data(&data).is_err());
    }

    #[test]
    fn test_validate_percentage_valid() {
        assert!(validate_percentage(0.5, "test").is_ok());
    }

    #[test]
    fn test_validate_percentage_too_high() {
        assert!(validate_percentage(1.5, "test").is_err());
    }

    #[test]
    fn test_validate_position_size_valid() {
        assert!(validate_position_size(0.1).is_ok());
    }

    #[test]
    fn test_validate_position_size_too_high() {
        assert!(validate_position_size(0.99).is_err());
    }

    #[test]
    fn test_validate_array_bounds() {
        let arr = vec![1, 2, 3];
        assert!(validate_array_bounds(&arr, 1, "test").is_ok());
        assert!(validate_array_bounds(&arr, 5, "test").is_err());
    }

    #[test]
    fn test_validate_array_not_empty() {
        let empty: Vec<i32> = vec![];
        let non_empty = vec![1, 2, 3];
        
        assert!(validate_array_not_empty(&empty, "test").is_err());
        assert!(validate_array_not_empty(&non_empty, "test").is_ok());
    }
}