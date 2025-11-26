//! Nautilus Trader compatibility layer.
//!
//! This module provides traits and utilities for seamless integration
//! with Nautilus Trader's type system when the full Nautilus crates
//! are available at compile time.

use crate::types::{NautilusQuoteTick, NautilusTradeTick, NautilusBar, NautilusOrderBookDelta};

/// Trait for types that can be converted from Nautilus data
pub trait FromNautilus<T> {
    /// Convert from Nautilus type
    fn from_nautilus(value: T) -> Self;
}

/// Trait for types that can be converted to Nautilus data
pub trait ToNautilus<T> {
    /// Convert to Nautilus type
    fn to_nautilus(&self) -> T;
}

/// Instrument identifier wrapper for string/hash interop
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct InstrumentId {
    /// String representation
    pub symbol: String,
    /// Venue
    pub venue: String,
    /// Precomputed hash for fast comparison
    pub hash: u64,
}

impl InstrumentId {
    /// Create a new instrument ID
    pub fn new(symbol: &str, venue: &str) -> Self {
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;

        let mut hasher = DefaultHasher::new();
        symbol.hash(&mut hasher);
        venue.hash(&mut hasher);

        Self {
            symbol: symbol.to_string(),
            venue: venue.to_string(),
            hash: hasher.finish(),
        }
    }

    /// Parse from string "SYMBOL.VENUE"
    pub fn parse(s: &str) -> Option<Self> {
        let parts: Vec<&str> = s.split('.').collect();
        if parts.len() >= 2 {
            Some(Self::new(parts[0], parts[1]))
        } else {
            None
        }
    }

    /// Convert to string representation
    pub fn to_string(&self) -> String {
        format!("{}.{}", self.symbol, self.venue)
    }
}

/// Price type with precision tracking
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Price {
    /// Raw fixed-point value
    pub raw: i64,
    /// Decimal precision
    pub precision: u8,
}

impl Price {
    /// Create from f64 with specified precision
    pub fn from_f64(value: f64, precision: u8) -> Self {
        let scale = 10_i64.pow(precision as u32);
        Self {
            raw: (value * scale as f64).round() as i64,
            precision,
        }
    }

    /// Convert to f64
    pub fn as_f64(&self) -> f64 {
        let scale = 10_i64.pow(self.precision as u32);
        self.raw as f64 / scale as f64
    }
}

/// Quantity type with precision tracking
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quantity {
    /// Raw fixed-point value
    pub raw: u64,
    /// Decimal precision
    pub precision: u8,
}

impl Quantity {
    /// Create from f64 with specified precision
    pub fn from_f64(value: f64, precision: u8) -> Self {
        let scale = 10_u64.pow(precision as u32);
        Self {
            raw: (value * scale as f64).round() as u64,
            precision,
        }
    }

    /// Convert to f64
    pub fn as_f64(&self) -> f64 {
        let scale = 10_u64.pow(self.precision as u32);
        self.raw as f64 / scale as f64
    }
}

/// Book action enumeration matching Nautilus
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum BookAction {
    /// Add order to book
    Add = 1,
    /// Update existing order
    Update = 2,
    /// Delete order from book
    Delete = 3,
    /// Clear entire book
    Clear = 4,
}

impl From<u8> for BookAction {
    fn from(value: u8) -> Self {
        match value {
            1 => BookAction::Add,
            2 => BookAction::Update,
            3 => BookAction::Delete,
            4 => BookAction::Clear,
            _ => BookAction::Clear, // Default to clear for unknown
        }
    }
}

/// Builder pattern for constructing Nautilus-compatible types
pub struct QuoteTickBuilder {
    instrument_id: u64,
    bid_price: i64,
    ask_price: i64,
    bid_size: u64,
    ask_size: u64,
    price_precision: u8,
    size_precision: u8,
    ts_event: u64,
    ts_init: u64,
}

impl QuoteTickBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            instrument_id: 0,
            bid_price: 0,
            ask_price: 0,
            bid_size: 0,
            ask_size: 0,
            price_precision: 2,
            size_precision: 0,
            ts_event: 0,
            ts_init: 0,
        }
    }

    /// Set instrument ID from InstrumentId
    pub fn instrument(mut self, id: &InstrumentId) -> Self {
        self.instrument_id = id.hash;
        self
    }

    /// Set bid price
    pub fn bid_price(mut self, price: Price) -> Self {
        self.bid_price = price.raw;
        self.price_precision = price.precision;
        self
    }

    /// Set ask price
    pub fn ask_price(mut self, price: Price) -> Self {
        self.ask_price = price.raw;
        self
    }

    /// Set bid size
    pub fn bid_size(mut self, qty: Quantity) -> Self {
        self.bid_size = qty.raw;
        self.size_precision = qty.precision;
        self
    }

    /// Set ask size
    pub fn ask_size(mut self, qty: Quantity) -> Self {
        self.ask_size = qty.raw;
        self
    }

    /// Set event timestamp
    pub fn ts_event(mut self, ts: u64) -> Self {
        self.ts_event = ts;
        self
    }

    /// Set init timestamp
    pub fn ts_init(mut self, ts: u64) -> Self {
        self.ts_init = ts;
        self
    }

    /// Build the quote tick
    pub fn build(self) -> NautilusQuoteTick {
        NautilusQuoteTick {
            instrument_id: self.instrument_id,
            bid_price: self.bid_price,
            ask_price: self.ask_price,
            bid_size: self.bid_size,
            ask_size: self.ask_size,
            price_precision: self.price_precision,
            size_precision: self.size_precision,
            ts_event: self.ts_event,
            ts_init: self.ts_init,
        }
    }
}

impl Default for QuoteTickBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_instrument_id() {
        let id = InstrumentId::new("BTCUSDT", "BINANCE");
        assert_eq!(id.symbol, "BTCUSDT");
        assert_eq!(id.venue, "BINANCE");

        let parsed = InstrumentId::parse("BTCUSDT.BINANCE").unwrap();
        assert_eq!(parsed.symbol, id.symbol);
        assert_eq!(parsed.venue, id.venue);
    }

    #[test]
    fn test_price_conversion() {
        let price = Price::from_f64(123.456, 3);
        assert!((price.as_f64() - 123.456).abs() < 1e-10);
    }

    #[test]
    fn test_quote_builder() {
        let id = InstrumentId::new("AAPL", "NASDAQ");
        let quote = QuoteTickBuilder::new()
            .instrument(&id)
            .bid_price(Price::from_f64(150.00, 2))
            .ask_price(Price::from_f64(150.05, 2))
            .bid_size(Quantity::from_f64(100.0, 0))
            .ask_size(Quantity::from_f64(150.0, 0))
            .ts_event(1000000000)
            .ts_init(1000000100)
            .build();

        assert_eq!(quote.instrument_id, id.hash);
        assert_eq!(quote.price_precision, 2);
    }
}
