//! Utility functions for Talebian risk management
//!
//! This module provides common utility functions and helpers
//! for the Talebian risk management crate.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Mathematical utilities
/// 
/// DEPRECATED: Use `crate::safe_math` module for production-grade safe operations
pub mod math {
    use crate::safe_math::safe_divide_with_fallback;
    
    /// Safe division that handles division by zero
    /// 
    /// DEPRECATED: Use `crate::safe_math::safe_divide` for better error handling
    pub fn safe_divide(numerator: f64, denominator: f64) -> f64 {
        safe_divide_with_fallback(numerator, denominator, 0.0)
    }

    /// Calculate standard deviation
    pub fn std_dev(values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance =
            values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64;

        variance.sqrt()
    }

    /// Calculate correlation between two series
    pub fn correlation(x: &[f64], y: &[f64]) -> Option<f64> {
        if x.len() != y.len() || x.len() < 2 {
            return None;
        }

        let n = x.len() as f64;
        let mean_x = x.iter().sum::<f64>() / n;
        let mean_y = y.iter().sum::<f64>() / n;

        let covariance = x
            .iter()
            .zip(y.iter())
            .map(|(&xi, &yi)| (xi - mean_x) * (yi - mean_y))
            .sum::<f64>()
            / (n - 1.0);

        let std_x = std_dev(x);
        let std_y = std_dev(y);

        if std_x == 0.0 || std_y == 0.0 {
            Some(0.0)
        } else {
            Some(covariance / (std_x * std_y))
        }
    }
}

/// Data validation utilities
pub mod validation {
    use super::*;

    /// Validate that data is finite and non-empty
    pub fn validate_data(data: &[f64]) -> Result<(), Box<dyn std::error::Error>> {
        if data.is_empty() {
            return Err("Data cannot be empty".into());
        }

        for (i, &value) in data.iter().enumerate() {
            if !value.is_finite() {
                return Err(format!("Invalid value at index {}: {}", i, value).into());
            }
        }

        Ok(())
    }

    /// Validate allocation weights sum to 1
    pub fn validate_allocation(
        allocation: &HashMap<String, f64>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let total: f64 = allocation.values().sum();

        if (total - 1.0).abs() > 0.01 {
            return Err(format!("Allocation weights sum to {} instead of 1.0", total).into());
        }

        for (asset, &weight) in allocation {
            if weight < 0.0 || weight > 1.0 {
                return Err(format!("Invalid weight for {}: {}", asset, weight).into());
            }
        }

        Ok(())
    }
}

/// Time series utilities
pub mod timeseries {
    /// Calculate returns from price series
    pub fn calculate_returns(prices: &[f64]) -> Vec<f64> {
        if prices.len() < 2 {
            return Vec::new();
        }

        prices.windows(2).map(|w| (w[1] / w[0]) - 1.0).collect()
    }

    /// Calculate log returns from price series
    pub fn calculate_log_returns(prices: &[f64]) -> Vec<f64> {
        if prices.len() < 2 {
            return Vec::new();
        }

        prices.windows(2).map(|w| (w[1] / w[0]).ln()).collect()
    }

    /// Calculate rolling window statistics
    pub fn rolling_mean(values: &[f64], window: usize) -> Vec<f64> {
        if window == 0 || values.len() < window {
            return Vec::new();
        }

        values
            .windows(window)
            .map(|w| w.iter().sum::<f64>() / window as f64)
            .collect()
    }
}

/// Configuration utilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigValidator;

impl ConfigValidator {
    /// Validate risk parameters
    pub fn validate_risk_params(
        risk_tolerance: f64,
        max_drawdown: f64,
        volatility_target: f64,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if risk_tolerance < 0.0 || risk_tolerance > 1.0 {
            return Err("Risk tolerance must be between 0 and 1".into());
        }

        if max_drawdown < 0.0 || max_drawdown > 1.0 {
            return Err("Max drawdown must be between 0 and 1".into());
        }

        if volatility_target < 0.0 || volatility_target > 2.0 {
            return Err("Volatility target must be between 0 and 2".into());
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_safe_divide() {
        assert_eq!(math::safe_divide(10.0, 2.0), 5.0);
        assert_eq!(math::safe_divide(10.0, 0.0), 0.0);
    }

    #[test]
    fn test_std_dev() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let std = math::std_dev(&values);
        assert!(std > 0.0);
    }

    #[test]
    fn test_correlation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0]; // Perfect positive correlation

        let corr = math::correlation(&x, &y)
            .expect("Correlation calculation should succeed for valid inputs");
        assert!((corr - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_calculate_returns() {
        let prices = vec![100.0, 110.0, 105.0, 115.0];
        let returns = timeseries::calculate_returns(&prices);

        assert_eq!(returns.len(), 3);
        assert!((returns[0] - 0.1).abs() < 0.001); // 10% return
    }

    #[test]
    fn test_validate_data() {
        let valid_data = vec![1.0, 2.0, 3.0];
        assert!(validation::validate_data(&valid_data).is_ok());

        let invalid_data = vec![1.0, f64::NAN, 3.0];
        assert!(validation::validate_data(&invalid_data).is_err());
    }

    #[test]
    fn test_validate_allocation() {
        let mut allocation = HashMap::new();
        allocation.insert("A".to_string(), 0.6);
        allocation.insert("B".to_string(), 0.4);

        assert!(validation::validate_allocation(&allocation).is_ok());

        allocation.insert("C".to_string(), 0.1); // Now sums to 1.1
        assert!(validation::validate_allocation(&allocation).is_err());
    }

    #[test]
    fn test_rolling_mean() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let rolling = timeseries::rolling_mean(&values, 3);

        assert_eq!(rolling.len(), 3);
        assert_eq!(rolling[0], 2.0); // (1+2+3)/3
        assert_eq!(rolling[1], 3.0); // (2+3+4)/3
        assert_eq!(rolling[2], 4.0); // (3+4+5)/3
    }
}
