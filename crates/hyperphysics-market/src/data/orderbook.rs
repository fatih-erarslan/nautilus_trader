//! Order book data structures

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Represents a single level in the order book
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OrderBookLevel {
    /// Price level
    pub price: f64,

    /// Total size at this level
    pub size: f64,

    /// Number of orders at this level (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub order_count: Option<u32>,
}

/// Represents a market order book
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBook {
    /// Trading symbol
    pub symbol: String,

    /// Bid levels (price -> OrderBookLevel)
    /// BTreeMap maintains sorted order
    pub bids: BTreeMap<ordered_float::OrderedFloat<f64>, OrderBookLevel>,

    /// Ask levels (price -> OrderBookLevel)
    pub asks: BTreeMap<ordered_float::OrderedFloat<f64>, OrderBookLevel>,
}

impl OrderBook {
    /// Create a new empty order book
    pub fn new(symbol: String) -> Self {
        Self {
            symbol,
            bids: BTreeMap::new(),
            asks: BTreeMap::new(),
        }
    }

    /// Get the best bid price
    pub fn best_bid(&self) -> Option<f64> {
        self.bids.keys().last().map(|k| k.into_inner())
    }

    /// Get the best ask price
    pub fn best_ask(&self) -> Option<f64> {
        self.asks.keys().next().map(|k| k.into_inner())
    }

    /// Get the bid-ask spread
    pub fn spread(&self) -> Option<f64> {
        match (self.best_ask(), self.best_bid()) {
            (Some(ask), Some(bid)) => Some(ask - bid),
            _ => None,
        }
    }

    /// Get the mid price
    pub fn mid_price(&self) -> Option<f64> {
        match (self.best_ask(), self.best_bid()) {
            (Some(ask), Some(bid)) => Some((ask + bid) / 2.0),
            _ => None,
        }
    }

    /// Calculate total bid volume up to a certain depth
    pub fn bid_volume(&self, depth: usize) -> f64 {
        self.bids
            .values()
            .rev()
            .take(depth)
            .map(|level| level.size)
            .sum()
    }

    /// Calculate total ask volume up to a certain depth
    pub fn ask_volume(&self, depth: usize) -> f64 {
        self.asks
            .values()
            .take(depth)
            .map(|level| level.size)
            .sum()
    }

    /// Calculate imbalance ratio (bid_volume / (bid_volume + ask_volume))
    pub fn imbalance(&self, depth: usize) -> Option<f64> {
        let bid_vol = self.bid_volume(depth);
        let ask_vol = self.ask_volume(depth);
        let total = bid_vol + ask_vol;

        if total > 0.0 {
            Some(bid_vol / total)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ordered_float::OrderedFloat;
    use approx::assert_relative_eq;

    #[test]
    fn test_orderbook_operations() {
        let mut book = OrderBook::new("AAPL".to_string());

        // Add bid levels
        book.bids.insert(
            OrderedFloat(150.00),
            OrderBookLevel { price: 150.00, size: 100.0, order_count: Some(5) }
        );
        book.bids.insert(
            OrderedFloat(149.99),
            OrderBookLevel { price: 149.99, size: 200.0, order_count: Some(10) }
        );

        // Add ask levels
        book.asks.insert(
            OrderedFloat(150.01),
            OrderBookLevel { price: 150.01, size: 150.0, order_count: Some(7) }
        );
        book.asks.insert(
            OrderedFloat(150.02),
            OrderBookLevel { price: 150.02, size: 100.0, order_count: Some(3) }
        );

        assert_relative_eq!(book.best_bid().unwrap(), 150.00, epsilon = 1e-10);
        assert_relative_eq!(book.best_ask().unwrap(), 150.01, epsilon = 1e-10);
        assert_relative_eq!(book.spread().unwrap(), 0.01, epsilon = 1e-10);
        assert_relative_eq!(book.mid_price().unwrap(), 150.005, epsilon = 1e-10);
        assert_relative_eq!(book.bid_volume(2), 300.0, epsilon = 1e-10);
        assert_relative_eq!(book.ask_volume(2), 250.0, epsilon = 1e-10);
    }
}
