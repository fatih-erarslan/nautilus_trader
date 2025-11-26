/*!
 * Precise decimal arithmetic for financial calculations
 *
 * Uses rust_decimal to avoid floating-point precision issues
 */

use rust_decimal::Decimal;
use std::str::FromStr;
use crate::error::{RustCoreError, Result};

/// Trait for decimal math operations
pub trait DecimalMath {
    fn add(&self, other: &Self) -> Result<Self> where Self: Sized;
    fn subtract(&self, other: &Self) -> Result<Self> where Self: Sized;
    fn multiply(&self, other: &Self) -> Result<Self> where Self: Sized;
    fn divide(&self, other: &Self) -> Result<Self> where Self: Sized;
}

impl DecimalMath for Decimal {
    fn add(&self, other: &Self) -> Result<Self> {
        self.checked_add(*other)
            .ok_or_else(|| RustCoreError::DecimalError("Addition overflow".to_string()))
    }

    fn subtract(&self, other: &Self) -> Result<Self> {
        self.checked_sub(*other)
            .ok_or_else(|| RustCoreError::DecimalError("Subtraction overflow".to_string()))
    }

    fn multiply(&self, other: &Self) -> Result<Self> {
        self.checked_mul(*other)
            .ok_or_else(|| RustCoreError::DecimalError("Multiplication overflow".to_string()))
    }

    fn divide(&self, other: &Self) -> Result<Self> {
        if other.is_zero() {
            return Err(RustCoreError::DecimalError("Division by zero".to_string()));
        }
        self.checked_div(*other)
            .ok_or_else(|| RustCoreError::DecimalError("Division overflow".to_string()))
    }
}

/// Add two decimal strings and return the result as a string
#[napi]
pub fn add_decimals(a: String, b: String) -> napi::Result<String> {
    let a_dec = Decimal::from_str(&a)
        .map_err(|e| napi::Error::from_reason(format!("Invalid decimal A: {}", e)))?;
    let b_dec = Decimal::from_str(&b)
        .map_err(|e| napi::Error::from_reason(format!("Invalid decimal B: {}", e)))?;

    let result = a_dec.add(&b_dec)
        .map_err(|e| napi::Error::from_reason(format!("Addition failed: {}", e)))?;

    Ok(result.to_string())
}

/// Subtract two decimal strings and return the result as a string
#[napi]
pub fn subtract_decimals(a: String, b: String) -> napi::Result<String> {
    let a_dec = Decimal::from_str(&a)
        .map_err(|e| napi::Error::from_reason(format!("Invalid decimal A: {}", e)))?;
    let b_dec = Decimal::from_str(&b)
        .map_err(|e| napi::Error::from_reason(format!("Invalid decimal B: {}", e)))?;

    let result = a_dec.subtract(&b_dec)
        .map_err(|e| napi::Error::from_reason(format!("Subtraction failed: {}", e)))?;

    Ok(result.to_string())
}

/// Multiply two decimal strings and return the result as a string
#[napi]
pub fn multiply_decimals(a: String, b: String) -> napi::Result<String> {
    let a_dec = Decimal::from_str(&a)
        .map_err(|e| napi::Error::from_reason(format!("Invalid decimal A: {}", e)))?;
    let b_dec = Decimal::from_str(&b)
        .map_err(|e| napi::Error::from_reason(format!("Invalid decimal B: {}", e)))?;

    let result = a_dec.multiply(&b_dec)
        .map_err(|e| napi::Error::from_reason(format!("Multiplication failed: {}", e)))?;

    Ok(result.to_string())
}

/// Divide two decimal strings and return the result as a string
#[napi]
pub fn divide_decimals(a: String, b: String) -> napi::Result<String> {
    let a_dec = Decimal::from_str(&a)
        .map_err(|e| napi::Error::from_reason(format!("Invalid decimal A: {}", e)))?;
    let b_dec = Decimal::from_str(&b)
        .map_err(|e| napi::Error::from_reason(format!("Invalid decimal B: {}", e)))?;

    let result = a_dec.divide(&b_dec)
        .map_err(|e| napi::Error::from_reason(format!("Division failed: {}", e)))?;

    Ok(result.to_string())
}

/// Calculate gain/loss from a disposal
#[napi]
pub fn calculate_gain_loss(
    sale_price: String,
    sale_quantity: String,
    cost_basis: String,
    cost_quantity: String,
) -> napi::Result<String> {
    let sale_price_dec = Decimal::from_str(&sale_price)
        .map_err(|e| napi::Error::from_reason(format!("Invalid sale price: {}", e)))?;
    let sale_qty_dec = Decimal::from_str(&sale_quantity)
        .map_err(|e| napi::Error::from_reason(format!("Invalid sale quantity: {}", e)))?;
    let cost_basis_dec = Decimal::from_str(&cost_basis)
        .map_err(|e| napi::Error::from_reason(format!("Invalid cost basis: {}", e)))?;
    let cost_qty_dec = Decimal::from_str(&cost_quantity)
        .map_err(|e| napi::Error::from_reason(format!("Invalid cost quantity: {}", e)))?;

    // proceeds = sale_price * sale_quantity
    let proceeds = sale_price_dec.multiply(&sale_qty_dec)
        .map_err(|e| napi::Error::from_reason(format!("Failed to calculate proceeds: {}", e)))?;

    // basis = cost_basis * cost_quantity
    let basis = cost_basis_dec.multiply(&cost_qty_dec)
        .map_err(|e| napi::Error::from_reason(format!("Failed to calculate basis: {}", e)))?;

    // gain/loss = proceeds - basis
    let gain_loss = proceeds.subtract(&basis)
        .map_err(|e| napi::Error::from_reason(format!("Failed to calculate gain/loss: {}", e)))?;

    Ok(gain_loss.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_decimals() {
        let result = add_decimals("10.50".to_string(), "5.25".to_string()).unwrap();
        assert_eq!(result, "15.75");
    }

    #[test]
    fn test_subtract_decimals() {
        let result = subtract_decimals("10.50".to_string(), "5.25".to_string()).unwrap();
        assert_eq!(result, "5.25");
    }

    #[test]
    fn test_multiply_decimals() {
        let result = multiply_decimals("10.5".to_string(), "2".to_string()).unwrap();
        assert_eq!(result, "21.0");
    }

    #[test]
    fn test_divide_decimals() {
        let result = divide_decimals("10.5".to_string(), "2".to_string()).unwrap();
        // rust_decimal preserves precision, so we get "5.250" instead of "5.25"
        assert!(result == "5.25" || result == "5.250");
    }

    #[test]
    fn test_divide_by_zero() {
        let result = divide_decimals("10.5".to_string(), "0".to_string());
        assert!(result.is_err());
    }

    #[test]
    fn test_calculate_gain_loss() {
        // Buy 1 BTC at $10,000, sell at $15,000 = $5,000 gain
        let result = calculate_gain_loss(
            "15000".to_string(),
            "1".to_string(),
            "10000".to_string(),
            "1".to_string(),
        ).unwrap();
        assert_eq!(result, "5000");
    }

    #[test]
    fn test_high_precision() {
        // Test that we maintain precision for small values
        let result = add_decimals("0.00000001".to_string(), "0.00000002".to_string()).unwrap();
        assert_eq!(result, "0.00000003");
    }
}
