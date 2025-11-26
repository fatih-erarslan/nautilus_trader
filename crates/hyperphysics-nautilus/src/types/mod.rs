//! Type definitions and conversions for Nautilus-HyperPhysics integration.
//!
//! This module provides bridge types that can be constructed from Nautilus
//! data types and converted to HyperPhysics internal representations.

mod conversions;
mod nautilus_compat;

pub use conversions::*;
pub use nautilus_compat::*;

use serde::{Deserialize, Serialize};

/// Nautilus-compatible quote tick representation
///
/// This mirrors `nautilus_model::data::QuoteTick` structure for zero-copy
/// interoperability when Nautilus crates are linked.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct NautilusQuoteTick {
    /// Instrument identifier string
    pub instrument_id: u64, // Hash of instrument ID for speed
    /// Best bid price (as fixed-point integer)
    pub bid_price: i64,
    /// Best ask price (as fixed-point integer)
    pub ask_price: i64,
    /// Best bid size (as fixed-point integer)
    pub bid_size: u64,
    /// Best ask size (as fixed-point integer)
    pub ask_size: u64,
    /// Price precision (decimal places)
    pub price_precision: u8,
    /// Size precision (decimal places)
    pub size_precision: u8,
    /// Event timestamp (nanoseconds since epoch)
    pub ts_event: u64,
    /// Init timestamp (nanoseconds since epoch)
    pub ts_init: u64,
}

/// Nautilus-compatible trade tick representation
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct NautilusTradeTick {
    /// Instrument identifier hash
    pub instrument_id: u64,
    /// Trade price (fixed-point)
    pub price: i64,
    /// Trade size (fixed-point)
    pub size: u64,
    /// Aggressor side (1 = buy, 2 = sell)
    pub aggressor_side: u8,
    /// Trade ID
    pub trade_id: u64,
    /// Price precision
    pub price_precision: u8,
    /// Size precision
    pub size_precision: u8,
    /// Event timestamp
    pub ts_event: u64,
    /// Init timestamp
    pub ts_init: u64,
}

/// Nautilus-compatible order book delta
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct NautilusOrderBookDelta {
    /// Instrument identifier hash
    pub instrument_id: u64,
    /// Book action (1=Add, 2=Update, 3=Delete, 4=Clear)
    pub action: u8,
    /// Order side (1=Bid, 2=Ask)
    pub side: u8,
    /// Price level (fixed-point)
    pub price: i64,
    /// Size at level (fixed-point)
    pub size: u64,
    /// Order ID (for L3 books)
    pub order_id: u64,
    /// Sequence number
    pub sequence: u64,
    /// Record flags
    pub flags: u8,
    /// Price precision
    pub price_precision: u8,
    /// Size precision
    pub size_precision: u8,
    /// Event timestamp
    pub ts_event: u64,
    /// Init timestamp
    pub ts_init: u64,
}

/// Nautilus-compatible bar (OHLCV)
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct NautilusBar {
    /// Instrument identifier hash
    pub instrument_id: u64,
    /// Open price (fixed-point)
    pub open: i64,
    /// High price (fixed-point)
    pub high: i64,
    /// Low price (fixed-point)
    pub low: i64,
    /// Close price (fixed-point)
    pub close: i64,
    /// Volume (fixed-point)
    pub volume: u64,
    /// Price precision
    pub price_precision: u8,
    /// Size precision
    pub size_precision: u8,
    /// Bar open timestamp
    pub ts_event: u64,
    /// Init timestamp
    pub ts_init: u64,
}

/// Order side enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum OrderSide {
    /// No side specified
    NoSide = 0,
    /// Buy order
    Buy = 1,
    /// Sell order
    Sell = 2,
}

/// Order type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum OrderType {
    /// Market order
    Market = 1,
    /// Limit order
    Limit = 2,
    /// Stop market order
    StopMarket = 3,
    /// Stop limit order
    StopLimit = 4,
}

/// Time in force enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum TimeInForce {
    /// Good till canceled
    GTC = 1,
    /// Immediate or cancel
    IOC = 2,
    /// Fill or kill
    FOK = 3,
    /// Good till date
    GTD = 4,
    /// Day order
    Day = 5,
}

/// HyperPhysics order command for Nautilus execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyperPhysicsOrderCommand {
    /// Client order ID
    pub client_order_id: String,
    /// Instrument identifier
    pub instrument_id: String,
    /// Order side
    pub side: OrderSide,
    /// Order type
    pub order_type: OrderType,
    /// Quantity (normalized 0-1, scaled by max position)
    pub quantity: f64,
    /// Limit price (if applicable)
    pub price: Option<f64>,
    /// Time in force
    pub time_in_force: TimeInForce,
    /// Reduce only flag
    pub reduce_only: bool,
    /// Post only flag
    pub post_only: bool,
    /// HyperPhysics confidence score
    pub hp_confidence: f64,
    /// HyperPhysics algorithm that generated signal
    pub hp_algorithm: String,
    /// Pipeline latency in microseconds
    pub hp_latency_us: u64,
    /// Consensus state at signal time
    pub hp_consensus_term: u64,
}

/// Market state snapshot for HyperPhysics processing
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MarketSnapshot {
    /// Current mid price
    pub mid_price: f64,
    /// Best bid price
    pub bid_price: f64,
    /// Best ask price
    pub ask_price: f64,
    /// Best bid size
    pub bid_size: f64,
    /// Best ask size
    pub ask_size: f64,
    /// Spread in price units
    pub spread: f64,
    /// Spread in basis points
    pub spread_bps: f64,
    /// Recent returns (newest first)
    pub returns: Vec<f64>,
    /// Realized volatility
    pub volatility: f64,
    /// VWAP estimate
    pub vwap: f64,
    /// Cumulative volume
    pub volume: f64,
    /// Order book imbalance (-1 to 1)
    pub book_imbalance: f64,
    /// Timestamp (seconds since epoch)
    pub timestamp: f64,
    /// Instrument identifier
    pub instrument_id: String,
}

impl MarketSnapshot {
    /// Calculate order book imbalance from bid/ask sizes
    pub fn calculate_imbalance(&mut self) {
        let total = self.bid_size + self.ask_size;
        if total > 0.0 {
            self.book_imbalance = (self.bid_size - self.ask_size) / total;
        } else {
            self.book_imbalance = 0.0;
        }
    }

    /// Update with a new return value
    pub fn push_return(&mut self, new_return: f64, max_size: usize) {
        self.returns.insert(0, new_return);
        if self.returns.len() > max_size {
            self.returns.truncate(max_size);
        }
    }

    /// Calculate volatility from returns
    pub fn update_volatility(&mut self) {
        if self.returns.len() < 2 {
            return;
        }
        let mean: f64 = self.returns.iter().sum::<f64>() / self.returns.len() as f64;
        let variance: f64 = self.returns.iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>() / (self.returns.len() - 1) as f64;
        self.volatility = variance.sqrt();
    }
}
