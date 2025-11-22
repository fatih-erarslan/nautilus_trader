/// Core financial data types with production-grade validation
///
/// References:
/// - IEEE 754 floating-point standard for price precision
/// - FIX Protocol for order book data structures
use serde::{Deserialize, Serialize};
use std::fmt;

/// Price in USD with validation
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Price(f64);

impl Price {
    /// Create new price with validation (must be non-negative)
    pub fn new(value: f64) -> Result<Self, FinanceError> {
        if value < 0.0 || !value.is_finite() {
            return Err(FinanceError::InvalidPrice(value));
        }
        Ok(Self(value))
    }

    pub fn value(&self) -> f64 {
        self.0
    }
}

/// Quantity with validation
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Quantity(f64);

impl Quantity {
    pub fn new(value: f64) -> Result<Self, FinanceError> {
        if value < 0.0 || !value.is_finite() {
            return Err(FinanceError::InvalidQuantity(value));
        }
        Ok(Self(value))
    }

    pub fn value(&self) -> f64 {
        self.0
    }
}

/// Level-2 order book snapshot
///
/// Format follows FIX Protocol specification for market data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct L2Snapshot {
    pub symbol: String,
    pub timestamp_us: u64,
    pub bids: Vec<(Price, Quantity)>,  // Sorted descending by price
    pub asks: Vec<(Price, Quantity)>,  // Sorted ascending by price
}

impl L2Snapshot {
    /// Validate order book structure
    pub fn validate(&self) -> Result<(), FinanceError> {
        if self.bids.is_empty() && self.asks.is_empty() {
            return Err(FinanceError::EmptyOrderBook);
        }

        // Validate bids are sorted descending
        for window in self.bids.windows(2) {
            if window[0].0.value() < window[1].0.value() {
                return Err(FinanceError::InvalidOrderBookSort);
            }
        }

        // Validate asks are sorted ascending
        for window in self.asks.windows(2) {
            if window[0].0.value() > window[1].0.value() {
                return Err(FinanceError::InvalidOrderBookSort);
            }
        }

        // Ensure no bid >= ask (crossed book)
        if let (Some(best_bid), Some(best_ask)) = (self.bids.first(), self.asks.first()) {
            if best_bid.0.value() >= best_ask.0.value() {
                return Err(FinanceError::CrossedOrderBook);
            }
        }

        Ok(())
    }

    /// Get best bid price
    pub fn best_bid(&self) -> Option<Price> {
        self.bids.first().map(|(p, _)| *p)
    }

    /// Get best ask price
    pub fn best_ask(&self) -> Option<Price> {
        self.asks.first().map(|(p, _)| *p)
    }

    /// Get mid-price: (best_bid + best_ask) / 2
    pub fn mid_price(&self) -> Option<Price> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => {
                Price::new((bid.value() + ask.value()) / 2.0).ok()
            }
            _ => None,
        }
    }

    /// Get spread: best_ask - best_bid
    pub fn spread(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some(ask.value() - bid.value()),
            _ => None,
        }
    }
}

/// Financial computation errors
#[derive(Debug, thiserror::Error)]
pub enum FinanceError {
    #[error("Invalid price: {0}")]
    InvalidPrice(f64),

    #[error("Invalid quantity: {0}")]
    InvalidQuantity(f64),

    #[error("Empty order book")]
    EmptyOrderBook,

    #[error("Invalid order book sort")]
    InvalidOrderBookSort,

    #[error("Crossed order book (bid >= ask)")]
    CrossedOrderBook,

    #[error("Invalid option parameters: {0}")]
    InvalidOptionParams(String),

    #[error("Numerical error: {0}")]
    NumericalError(String),

    #[error("Insufficient data for calculation")]
    InsufficientData,
}

impl fmt::Display for Price {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "${:.2}", self.0)
    }
}

impl fmt::Display for Quantity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.6}", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_price_validation() {
        assert!(Price::new(100.0).is_ok());
        assert!(Price::new(0.0).is_ok());
        assert!(Price::new(-1.0).is_err());
        assert!(Price::new(f64::NAN).is_err());
        assert!(Price::new(f64::INFINITY).is_err());
    }

    #[test]
    fn test_l2_snapshot_validation() {
        let snapshot = L2Snapshot {
            symbol: "BTC-USD".to_string(),
            timestamp_us: 1000000,
            bids: vec![
                (Price::new(100.0).unwrap(), Quantity::new(1.0).unwrap()),
                (Price::new(99.0).unwrap(), Quantity::new(2.0).unwrap()),
            ],
            asks: vec![
                (Price::new(101.0).unwrap(), Quantity::new(1.5).unwrap()),
                (Price::new(102.0).unwrap(), Quantity::new(2.5).unwrap()),
            ],
        };

        assert!(snapshot.validate().is_ok());
        assert_eq!(snapshot.mid_price().unwrap().value(), 100.5);
        assert_eq!(snapshot.spread().unwrap(), 1.0);
    }
}
