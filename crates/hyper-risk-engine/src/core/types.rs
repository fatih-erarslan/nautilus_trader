//! Core types for HyperRiskEngine.
//!
//! Designed for cache-line alignment and zero-copy operations
//! in the fast path.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

// ============================================================================
// Primitive Types (Cache-Aligned)
// ============================================================================

/// High-precision timestamp in nanoseconds since Unix epoch.
/// Aligned for atomic operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(transparent)]
pub struct Timestamp(pub u64);

impl Timestamp {
    /// Create timestamp from nanoseconds.
    #[inline]
    pub const fn from_nanos(nanos: u64) -> Self {
        Self(nanos)
    }

    /// Get current timestamp.
    #[inline]
    pub fn now() -> Self {
        use std::time::{SystemTime, UNIX_EPOCH};
        let duration = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards");
        Self(duration.as_nanos() as u64)
    }

    /// Get nanoseconds since epoch.
    #[inline]
    pub const fn as_nanos(&self) -> u64 {
        self.0
    }

    /// Get microseconds since epoch.
    #[inline]
    pub const fn as_micros(&self) -> u64 {
        self.0 / 1_000
    }

    /// Get milliseconds since epoch.
    #[inline]
    pub const fn as_millis(&self) -> u64 {
        self.0 / 1_000_000
    }
}

/// Price type with fixed precision (8 decimal places).
/// Uses i64 internally for exact arithmetic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(transparent)]
pub struct Price(i64);

impl Price {
    /// Precision multiplier (10^8).
    const PRECISION: i64 = 100_000_000;

    /// Create price from floating point (rounds to 8 decimal places).
    #[inline]
    pub fn from_f64(value: f64) -> Self {
        Self((value * Self::PRECISION as f64).round() as i64)
    }

    /// Convert to floating point.
    #[inline]
    pub fn as_f64(&self) -> f64 {
        self.0 as f64 / Self::PRECISION as f64
    }

    /// Zero price.
    #[inline]
    pub const fn zero() -> Self {
        Self(0)
    }

    /// Check if price is zero.
    #[inline]
    pub const fn is_zero(&self) -> bool {
        self.0 == 0
    }
}

impl fmt::Display for Price {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.8}", self.as_f64())
    }
}

/// Quantity type (can be negative for short positions).
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
#[repr(transparent)]
pub struct Quantity(f64);

impl Quantity {
    /// Create quantity from f64.
    #[inline]
    pub const fn from_f64(value: f64) -> Self {
        Self(value)
    }

    /// Get as f64.
    #[inline]
    pub const fn as_f64(&self) -> f64 {
        self.0
    }

    /// Get absolute value.
    #[inline]
    pub fn abs(&self) -> Self {
        Self(self.0.abs())
    }

    /// Check if zero.
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.0.abs() < 1e-10
    }

    /// Check if long (positive).
    #[inline]
    pub fn is_long(&self) -> bool {
        self.0 > 0.0
    }

    /// Check if short (negative).
    #[inline]
    pub fn is_short(&self) -> bool {
        self.0 < 0.0
    }
}

impl fmt::Display for Quantity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.4}", self.0)
    }
}

/// Symbol identifier (interned string for fast comparison).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Symbol {
    /// Hash of symbol string for fast comparison.
    hash: u64,
    /// Length of symbol (max 16 chars).
    len: u8,
    /// Inline storage for short symbols.
    data: [u8; 16],
}

impl Symbol {
    /// Create symbol from string.
    pub fn new(s: &str) -> Self {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        s.hash(&mut hasher);

        let bytes = s.as_bytes();
        let len = bytes.len().min(16) as u8;
        let mut data = [0u8; 16];
        data[..len as usize].copy_from_slice(&bytes[..len as usize]);

        Self {
            hash: hasher.finish(),
            len,
            data,
        }
    }

    /// Get symbol as string slice.
    pub fn as_str(&self) -> &str {
        std::str::from_utf8(&self.data[..self.len as usize]).unwrap_or("")
    }

    /// Get hash value for fast lookup.
    #[inline]
    pub fn hash_value(&self) -> u64 {
        self.hash
    }
}

impl fmt::Display for Symbol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Atomic counter for generating unique position IDs.
static POSITION_ID_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Position identifier (globally unique).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(transparent)]
pub struct PositionId(u64);

impl PositionId {
    /// Generate new unique position ID.
    #[inline]
    pub fn new() -> Self {
        Self(POSITION_ID_COUNTER.fetch_add(1, Ordering::Relaxed))
    }

    /// Create from existing ID.
    #[inline]
    pub const fn from_raw(id: u64) -> Self {
        Self(id)
    }

    /// Get raw ID value.
    #[inline]
    pub const fn as_raw(&self) -> u64 {
        self.0
    }
}

impl Default for PositionId {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Trading Types
// ============================================================================

/// Order side.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OrderSide {
    /// Buy order.
    Buy,
    /// Sell order.
    Sell,
}

impl OrderSide {
    /// Get sign for quantity calculation.
    #[inline]
    pub const fn sign(&self) -> f64 {
        match self {
            Self::Buy => 1.0,
            Self::Sell => -1.0,
        }
    }

    /// Check if this is a buy order.
    #[inline]
    pub const fn is_buy(&self) -> bool {
        matches!(self, Self::Buy)
    }

    /// Check if this is a sell order.
    #[inline]
    pub const fn is_sell(&self) -> bool {
        matches!(self, Self::Sell)
    }
}

/// Order for pre-trade risk check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Order {
    /// Symbol being traded.
    pub symbol: Symbol,
    /// Order side.
    pub side: OrderSide,
    /// Order quantity.
    pub quantity: Quantity,
    /// Limit price (None for market orders).
    pub limit_price: Option<Price>,
    /// Strategy identifier.
    pub strategy_id: u64,
    /// Timestamp when order was created.
    pub timestamp: Timestamp,
}

/// Position in portfolio.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    /// Position identifier.
    pub id: PositionId,
    /// Symbol.
    pub symbol: Symbol,
    /// Current quantity (negative for short).
    pub quantity: Quantity,
    /// Average entry price.
    pub avg_entry_price: Price,
    /// Current market price.
    pub current_price: Price,
    /// Unrealized P&L.
    pub unrealized_pnl: f64,
    /// Realized P&L.
    pub realized_pnl: f64,
    /// Position opened timestamp.
    pub opened_at: Timestamp,
    /// Last update timestamp.
    pub updated_at: Timestamp,
}

impl Position {
    /// Calculate position market value.
    #[inline]
    pub fn market_value(&self) -> f64 {
        self.quantity.as_f64() * self.current_price.as_f64()
    }

    /// Calculate total P&L.
    #[inline]
    pub fn total_pnl(&self) -> f64 {
        self.unrealized_pnl + self.realized_pnl
    }

    /// Update current price and recalculate P&L.
    pub fn update_price(&mut self, new_price: Price) {
        self.current_price = new_price;
        self.unrealized_pnl = self.quantity.as_f64()
            * (new_price.as_f64() - self.avg_entry_price.as_f64());
        self.updated_at = Timestamp::now();
    }
}

/// Portfolio state.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Portfolio {
    /// All positions.
    pub positions: Vec<Position>,
    /// Total portfolio value.
    pub total_value: f64,
    /// Cash balance.
    pub cash: f64,
    /// Peak portfolio value (for drawdown calculation).
    pub peak_value: f64,
    /// Total unrealized P&L.
    pub unrealized_pnl: f64,
    /// Total realized P&L.
    pub realized_pnl: f64,
}

impl Portfolio {
    /// Create new portfolio with initial cash.
    pub fn new(initial_cash: f64) -> Self {
        Self {
            positions: Vec::new(),
            total_value: initial_cash,
            cash: initial_cash,
            peak_value: initial_cash,
            unrealized_pnl: 0.0,
            realized_pnl: 0.0,
        }
    }

    /// Calculate current drawdown percentage.
    #[inline]
    pub fn drawdown_pct(&self) -> f64 {
        if self.peak_value > 0.0 {
            (self.peak_value - self.total_value) / self.peak_value * 100.0
        } else {
            0.0
        }
    }

    /// Get position by symbol.
    pub fn get_position(&self, symbol: &Symbol) -> Option<&Position> {
        self.positions.iter().find(|p| p.symbol == *symbol)
    }

    /// Get mutable position by symbol.
    pub fn get_position_mut(&mut self, symbol: &Symbol) -> Option<&mut Position> {
        self.positions.iter_mut().find(|p| p.symbol == *symbol)
    }

    /// Update portfolio totals.
    pub fn recalculate(&mut self) {
        self.unrealized_pnl = self.positions.iter().map(|p| p.unrealized_pnl).sum();
        self.total_value = self.cash
            + self.positions.iter().map(|p| p.market_value()).sum::<f64>();

        if self.total_value > self.peak_value {
            self.peak_value = self.total_value;
        }
    }

    /// Calculate total exposure (sum of absolute position values).
    #[inline]
    pub fn total_exposure(&self) -> f64 {
        self.positions.iter().map(|p| p.market_value().abs()).sum()
    }

    /// Get position value by symbol.
    pub fn get_position_value(&self, symbol: &Symbol) -> Option<f64> {
        self.get_position(symbol).map(|p| p.market_value())
    }
}

// ============================================================================
// Risk Types
// ============================================================================

/// Risk level classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum RiskLevel {
    /// Normal operations.
    Normal = 0,
    /// Elevated risk - increased monitoring.
    Elevated = 1,
    /// High risk - restrict new positions.
    High = 2,
    /// Critical - reduce exposure.
    Critical = 3,
    /// Emergency - halt all trading.
    Emergency = 4,
}

impl RiskLevel {
    /// Check if trading should be restricted.
    #[inline]
    pub const fn is_restricted(&self) -> bool {
        matches!(self, Self::High | Self::Critical | Self::Emergency)
    }

    /// Check if immediate action required.
    #[inline]
    pub const fn requires_immediate_action(&self) -> bool {
        matches!(self, Self::Critical | Self::Emergency)
    }
}

/// Decision from risk check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskDecision {
    /// Whether the action is allowed.
    pub allowed: bool,
    /// Current risk level.
    pub risk_level: RiskLevel,
    /// Reason for decision.
    pub reason: String,
    /// Suggested position size adjustment (1.0 = no change).
    pub size_adjustment: f64,
    /// Timestamp of decision.
    pub timestamp: Timestamp,
    /// Latency of decision in nanoseconds.
    pub latency_ns: u64,
}

impl RiskDecision {
    /// Create an approval decision.
    pub fn approve(latency_ns: u64) -> Self {
        Self {
            allowed: true,
            risk_level: RiskLevel::Normal,
            reason: "Approved".to_string(),
            size_adjustment: 1.0,
            timestamp: Timestamp::now(),
            latency_ns,
        }
    }

    /// Create a rejection decision.
    pub fn reject(reason: impl Into<String>, risk_level: RiskLevel, latency_ns: u64) -> Self {
        Self {
            allowed: false,
            risk_level,
            reason: reason.into(),
            size_adjustment: 0.0,
            timestamp: Timestamp::now(),
            latency_ns,
        }
    }
}

/// Market regime detected by regime detection agent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MarketRegime {
    /// Bullish trending market.
    BullTrending,
    /// Bearish trending market.
    BearTrending,
    /// Low volatility sideways.
    SidewaysLow,
    /// High volatility sideways.
    SidewaysHigh,
    /// Market crisis/crash.
    Crisis,
    /// Recovery from crisis.
    Recovery,
    /// Unknown/uncertain regime.
    Unknown,
}

impl MarketRegime {
    /// Get risk multiplier for this regime.
    #[inline]
    pub const fn risk_multiplier(&self) -> f64 {
        match self {
            Self::BullTrending => 1.0,
            Self::BearTrending => 0.7,
            Self::SidewaysLow => 0.9,
            Self::SidewaysHigh => 0.6,
            Self::Crisis => 0.2,
            Self::Recovery => 0.8,
            Self::Unknown => 0.5,
        }
    }

    /// Check if regime is favorable for trading.
    #[inline]
    pub const fn is_favorable(&self) -> bool {
        matches!(
            self,
            Self::BullTrending | Self::SidewaysLow | Self::Recovery
        )
    }
}

impl Default for MarketRegime {
    fn default() -> Self {
        Self::Unknown
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timestamp() {
        let ts = Timestamp::now();
        assert!(ts.as_nanos() > 0);
        assert!(ts.as_micros() < ts.as_nanos());
    }

    #[test]
    fn test_price_precision() {
        let price = Price::from_f64(123.456_789_01);
        assert!((price.as_f64() - 123.456_789_01).abs() < 1e-8);
    }

    #[test]
    fn test_symbol() {
        let sym1 = Symbol::new("AAPL");
        let sym2 = Symbol::new("AAPL");
        let sym3 = Symbol::new("GOOGL");

        assert_eq!(sym1, sym2);
        assert_ne!(sym1, sym3);
        assert_eq!(sym1.as_str(), "AAPL");
    }

    #[test]
    fn test_position_id_unique() {
        let id1 = PositionId::new();
        let id2 = PositionId::new();
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_portfolio_drawdown() {
        let mut portfolio = Portfolio::new(100_000.0);
        portfolio.total_value = 85_000.0;
        assert!((portfolio.drawdown_pct() - 15.0).abs() < 0.01);
    }

    #[test]
    fn test_risk_level_ordering() {
        assert!(RiskLevel::Normal < RiskLevel::Emergency);
        assert!(RiskLevel::Critical < RiskLevel::Emergency);
    }

    #[test]
    fn test_market_regime_multiplier() {
        assert_eq!(MarketRegime::Crisis.risk_multiplier(), 0.2);
        assert_eq!(MarketRegime::BullTrending.risk_multiplier(), 1.0);
    }
}
