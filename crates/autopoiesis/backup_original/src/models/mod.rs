//! Data models for the autopoiesis trading system

use serde::{Deserialize, Serialize};
use rust_decimal::Decimal;
use chrono::{DateTime, Utc};
use uuid::Uuid;

/// Market data point
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MarketData {
    /// Trading pair symbol
    pub symbol: String,
    
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    
    /// Best bid price
    pub bid: Decimal,
    
    /// Best ask price
    pub ask: Decimal,
    
    /// Mid price
    pub mid: Decimal,
    
    /// Last traded price
    pub last: Decimal,
    
    /// 24h volume
    pub volume_24h: Decimal,
    
    /// Bid size
    pub bid_size: Decimal,
    
    /// Ask size
    pub ask_size: Decimal,
}

/// Order representation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Order {
    /// Unique order ID
    pub id: Uuid,
    
    /// Trading pair symbol
    pub symbol: String,
    
    /// Order side
    pub side: OrderSide,
    
    /// Order type
    pub order_type: OrderType,
    
    /// Order quantity
    pub quantity: Decimal,
    
    /// Limit price (for limit orders)
    pub price: Option<Decimal>,
    
    /// Time in force
    pub time_in_force: TimeInForce,
    
    /// Order status
    pub status: OrderStatus,
    
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    
    /// Last update timestamp
    pub updated_at: DateTime<Utc>,
}

/// Order side
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum OrderSide {
    /// Buy order
    Buy,
    /// Sell order
    Sell,
}

/// Order type
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum OrderType {
    /// Market order
    Market,
    /// Limit order
    Limit,
    /// Stop order
    Stop,
    /// Stop-limit order
    StopLimit,
}

/// Time in force
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum TimeInForce {
    /// Good till canceled
    GTC,
    /// Immediate or cancel
    IOC,
    /// Fill or kill
    FOK,
    /// Good till time
    GTT(DateTime<Utc>),
}

/// Order status
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum OrderStatus {
    /// Order is pending submission
    Pending,
    /// Order has been submitted
    Submitted,
    /// Order is partially filled
    PartiallyFilled,
    /// Order is completely filled
    Filled,
    /// Order has been canceled
    Canceled,
    /// Order was rejected
    Rejected,
    /// Order has expired
    Expired,
}

/// Position representation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Position {
    /// Trading pair symbol
    pub symbol: String,
    
    /// Position side
    pub side: PositionSide,
    
    /// Position quantity
    pub quantity: Decimal,
    
    /// Average entry price
    pub entry_price: Decimal,
    
    /// Current mark price
    pub mark_price: Decimal,
    
    /// Unrealized P&L
    pub unrealized_pnl: Decimal,
    
    /// Realized P&L
    pub realized_pnl: Decimal,
    
    /// Position opened at
    pub opened_at: DateTime<Utc>,
    
    /// Last update timestamp
    pub updated_at: DateTime<Utc>,
}

/// Position side
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum PositionSide {
    /// Long position
    Long,
    /// Short position
    Short,
}

/// Trade representation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Trade {
    /// Unique trade ID
    pub id: Uuid,
    
    /// Order ID that created this trade
    pub order_id: Uuid,
    
    /// Trading pair symbol
    pub symbol: String,
    
    /// Trade side
    pub side: OrderSide,
    
    /// Trade price
    pub price: Decimal,
    
    /// Trade quantity
    pub quantity: Decimal,
    
    /// Trade fee
    pub fee: Decimal,
    
    /// Fee currency
    pub fee_currency: String,
    
    /// Trade timestamp
    pub timestamp: DateTime<Utc>,
}