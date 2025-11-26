//! Price Validation with Market Microstructure Analysis
//!
//! Validates price data against mathematical invariants, economic constraints,
//! and market microstructure theory.
//!
//! # Scientific Foundation
//!
//! Based on:
//! - OHLC mathematical invariants (high ≥ max(open, close), low ≤ min(open, close))
//! - No-arbitrage conditions (bid ≤ ask)
//! - Market microstructure theory (O'Hara, 1995)
//! - Statistical outlier detection (Tukey's fences, 1977)
//!
//! # References
//!
//! - O'Hara, M. (1995). Market Microstructure Theory
//! - Tukey, J. W. (1977). Exploratory Data Analysis
//! - Hasbrouck, J. (2007). Empirical Market Microstructure

use crate::data::Bar;
use crate::data::tick::{Quote, Tick};
use crate::error::{MarketError, MarketResult};

/// Price validator with microstructure analysis
#[derive(Debug)]
pub struct PriceValidator {
    /// Maximum price deviation threshold (percentage)
    deviation_threshold: f64,

    /// Maximum bid-ask spread percentage
    max_spread_pct: f64,
}

impl PriceValidator {
    /// Create new price validator
    pub fn new(deviation_threshold: f64, max_spread_pct: f64) -> Self {
        Self {
            deviation_threshold,
            max_spread_pct,
        }
    }

    /// Validate OHLC mathematical invariants
    pub fn validate_ohlc(&self, bar: &Bar) -> MarketResult<()> {
        // Invariant 1: high ≥ open
        if bar.high < bar.open {
            return Err(MarketError::ValidationError(format!(
                "OHLC invariant violated: high {} < open {}",
                bar.high, bar.open
            )));
        }

        // Invariant 2: high ≥ close
        if bar.high < bar.close {
            return Err(MarketError::ValidationError(format!(
                "OHLC invariant violated: high {} < close {}",
                bar.high, bar.close
            )));
        }

        // Invariant 3: low ≤ open
        if bar.low > bar.open {
            return Err(MarketError::ValidationError(format!(
                "OHLC invariant violated: low {} > open {}",
                bar.low, bar.open
            )));
        }

        // Invariant 4: low ≤ close
        if bar.low > bar.close {
            return Err(MarketError::ValidationError(format!(
                "OHLC invariant violated: low {} > close {}",
                bar.low, bar.close
            )));
        }

        // Invariant 5: high ≥ low
        if bar.high < bar.low {
            return Err(MarketError::ValidationError(format!(
                "OHLC invariant violated: high {} < low {}",
                bar.high, bar.low
            )));
        }

        // Economic constraint: positive prices
        if bar.open <= 0.0 || bar.high <= 0.0 || bar.low <= 0.0 || bar.close <= 0.0 {
            return Err(MarketError::ValidationError(
                "Non-positive price detected".to_string()
            ));
        }

        // Check for NaN or infinite values
        if !bar.open.is_finite() || !bar.high.is_finite() || !bar.low.is_finite() || !bar.close.is_finite() {
            return Err(MarketError::ValidationError(
                "NaN or infinite price detected".to_string()
            ));
        }

        // VWAP validation (if present)
        if let Some(vwap) = bar.vwap {
            if vwap < bar.low || vwap > bar.high {
                return Err(MarketError::ValidationError(format!(
                    "VWAP {} outside [low {}, high {}]",
                    vwap, bar.low, bar.high
                )));
            }
        }

        Ok(())
    }

    /// Validate tick price
    pub fn validate_tick_price(&self, tick: &Tick) -> MarketResult<()> {
        // Positive price constraint
        if tick.price <= 0.0 {
            return Err(MarketError::ValidationError(format!(
                "Non-positive tick price: {}",
                tick.price
            )));
        }

        // Check for NaN or infinite
        if !tick.price.is_finite() {
            return Err(MarketError::ValidationError(
                "NaN or infinite tick price".to_string()
            ));
        }

        // Positive size constraint
        if tick.size <= 0.0 {
            return Err(MarketError::ValidationError(format!(
                "Non-positive tick size: {}",
                tick.size
            )));
        }

        Ok(())
    }

    /// Validate quote spread (no-arbitrage condition)
    pub fn validate_quote_spread(&self, quote: &Quote) -> MarketResult<()> {
        // No-arbitrage condition: bid ≤ ask
        if quote.bid_price > quote.ask_price {
            return Err(MarketError::ValidationError(format!(
                "No-arbitrage violation: bid {} > ask {}",
                quote.bid_price, quote.ask_price
            )));
        }

        // Check spread percentage
        let mid_price = (quote.bid_price + quote.ask_price) / 2.0;
        let spread = quote.ask_price - quote.bid_price;
        let spread_pct = spread / mid_price;

        if spread_pct > self.max_spread_pct {
            return Err(MarketError::ValidationError(format!(
                "Spread {} exceeds maximum {}",
                spread_pct, self.max_spread_pct
            )));
        }

        // Positive prices
        if quote.bid_price <= 0.0 || quote.ask_price <= 0.0 {
            return Err(MarketError::ValidationError(
                "Non-positive quote price".to_string()
            ));
        }

        // Positive sizes
        if quote.bid_size == 0.0 || quote.ask_size == 0.0 {
            return Err(MarketError::ValidationError(
                "Zero quote size".to_string()
            ));
        }

        Ok(())
    }

    /// Detect price outliers using Tukey's fences
    pub fn detect_outliers(&self, prices: &[f64]) -> Vec<usize> {
        if prices.len() < 4 {
            return Vec::new();
        }

        // Calculate quartiles
        let mut sorted = prices.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let q1_idx = sorted.len() / 4;
        let q3_idx = (3 * sorted.len()) / 4;
        let q1 = sorted[q1_idx];
        let q3 = sorted[q3_idx];
        let iqr = q3 - q1;

        // Tukey's fences: outliers outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
        let lower_fence = q1 - 1.5 * iqr;
        let upper_fence = q3 + 1.5 * iqr;

        prices.iter()
            .enumerate()
            .filter(|(_, &price)| price < lower_fence || price > upper_fence)
            .map(|(i, _)| i)
            .collect()
    }

    /// Validate price continuity (no gaps > threshold)
    pub fn validate_continuity(&self, prices: &[f64]) -> MarketResult<()> {
        for i in 1..prices.len() {
            let return_pct = ((prices[i] / prices[i-1]) - 1.0).abs();
            if return_pct > self.deviation_threshold {
                return Err(MarketError::ValidationError(format!(
                    "Price discontinuity at index {}: {:.2}% change",
                    i, return_pct * 100.0
                )));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    #[test]
    fn test_valid_ohlc() {
        let validator = PriceValidator::new(0.05, 0.10);

        let bar = Bar {
            symbol: "TEST".to_string(),
            timestamp: Utc::now(),
            open: 100.0,
            high: 105.0,
            low: 98.0,
            close: 103.0,
            volume: 1000,
            vwap: Some(101.5),
            trade_count: None,
        };

        assert!(validator.validate_ohlc(&bar).is_ok());
    }

    #[test]
    fn test_invalid_high_low() {
        let validator = PriceValidator::new(0.05, 0.10);

        let bar = Bar {
            symbol: "TEST".to_string(),
            timestamp: Utc::now(),
            open: 100.0,
            high: 95.0,  // Invalid: high < low
            low: 98.0,
            close: 103.0,
            volume: 1000,
            vwap: None,
            trade_count: None,
        };

        assert!(validator.validate_ohlc(&bar).is_err());
    }

    #[test]
    fn test_no_arbitrage_condition() {
        let validator = PriceValidator::new(0.05, 0.10);

        let valid_quote = Quote {
            symbol: "TEST".to_string(),
            timestamp: Utc::now(),
            bid_price: 100.0,
            bid_size: 100.0,
            ask_price: 100.5,  // bid < ask (valid)
            ask_size: 100.0,
            exchange: None,
        };

        assert!(validator.validate_quote_spread(&valid_quote).is_ok());

        let invalid_quote = Quote {
            symbol: "TEST".to_string(),
            timestamp: Utc::now(),
            bid_price: 100.5,
            bid_size: 100.0,
            ask_price: 100.0,  // bid > ask (arbitrage!)
            ask_size: 100.0,
            exchange: None,
        };

        assert!(validator.validate_quote_spread(&invalid_quote).is_err());
    }

    #[test]
    fn test_outlier_detection() {
        let validator = PriceValidator::new(0.05, 0.10);

        let prices = vec![100.0, 101.0, 102.0, 150.0, 103.0, 104.0];
        let outliers = validator.detect_outliers(&prices);

        assert!(outliers.contains(&3));  // 150.0 is an outlier
    }
}
