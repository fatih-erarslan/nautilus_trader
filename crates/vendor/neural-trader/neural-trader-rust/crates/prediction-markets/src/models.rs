//! Prediction market data models

use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

/// Order side enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OrderSide {
    Buy,
    Sell,
}

impl OrderSide {
    pub fn opposite(&self) -> Self {
        match self {
            Self::Buy => Self::Sell,
            Self::Sell => Self::Buy,
        }
    }
}

/// Order type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OrderType {
    Market,
    Limit,
}

/// Order status enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OrderStatus {
    Pending,
    Open,
    Partial,
    Filled,
    Cancelled,
    Rejected,
    Expired,
}

impl OrderStatus {
    pub fn is_active(&self) -> bool {
        matches!(self, Self::Open | Self::Partial)
    }

    pub fn is_complete(&self) -> bool {
        matches!(
            self,
            Self::Filled | Self::Cancelled | Self::Rejected | Self::Expired
        )
    }

    pub fn can_cancel(&self) -> bool {
        matches!(self, Self::Pending | Self::Open | Self::Partial)
    }
}

/// Time in force enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum TimeInForce {
    GTC, // Good Till Cancelled
    IOC, // Immediate Or Cancel
    FOK, // Fill Or Kill
}

/// Market outcome
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Outcome {
    pub id: String,
    pub market_id: String,
    pub title: String,
    pub price: Decimal,
}

impl Outcome {
    pub fn probability(&self) -> Decimal {
        self.price
    }
}

/// Market resolution information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Resolution {
    pub outcome_id: String,
    pub resolved_at: DateTime<Utc>,
    pub resolution_source: String,
    pub disputed: bool,
    pub dispute_deadline: Option<DateTime<Utc>>,
    pub final_outcome: Option<String>,
}

/// Prediction market
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Market {
    pub id: String,
    pub question: String,
    pub description: Option<String>,
    pub outcomes: Vec<Outcome>,
    pub end_date: Option<DateTime<Utc>>,
    pub volume: Decimal,
    pub liquidity: Decimal,
    pub category: Option<String>,
    pub tags: Vec<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub resolution: Option<Resolution>,
}

impl Market {
    pub fn is_active(&self) -> bool {
        if let Some(end_date) = self.end_date {
            Utc::now() < end_date && self.resolution.is_none()
        } else {
            self.resolution.is_none()
        }
    }

    pub fn time_to_close(&self) -> Option<i64> {
        self.end_date.map(|end| (end - Utc::now()).num_seconds())
    }
}

/// Order book level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookLevel {
    pub price: Decimal,
    pub size: Decimal,
}

/// Order book
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBook {
    pub market_id: String,
    pub outcome_id: String,
    pub bids: Vec<OrderBookLevel>,
    pub asks: Vec<OrderBookLevel>,
    pub timestamp: DateTime<Utc>,
}

impl OrderBook {
    pub fn best_bid(&self) -> Option<Decimal> {
        self.bids.first().map(|level| level.price)
    }

    pub fn best_ask(&self) -> Option<Decimal> {
        self.asks.first().map(|level| level.price)
    }

    pub fn spread(&self) -> Option<Decimal> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some(ask - bid),
            _ => None,
        }
    }

    pub fn mid_price(&self) -> Option<Decimal> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some((bid + ask) / Decimal::from(2)),
            _ => None,
        }
    }

    pub fn total_bid_size(&self) -> Decimal {
        self.bids.iter().map(|level| level.size).sum()
    }

    pub fn total_ask_size(&self) -> Decimal {
        self.asks.iter().map(|level| level.size).sum()
    }

    pub fn get_depth(&self, levels: usize) -> (Vec<OrderBookLevel>, Vec<OrderBookLevel>) {
        let bids = self.bids.iter().take(levels).cloned().collect();
        let asks = self.asks.iter().take(levels).cloned().collect();
        (bids, asks)
    }

    pub fn calculate_price_impact(&self, side: OrderSide, size: Decimal) -> Option<Decimal> {
        let levels = match side {
            OrderSide::Buy => &self.asks,
            OrderSide::Sell => &self.bids,
        };

        if levels.is_empty() {
            return None;
        }

        let mut remaining_size = size;
        let mut total_cost = Decimal::ZERO;

        for level in levels {
            if remaining_size <= Decimal::ZERO {
                break;
            }

            let level_size = remaining_size.min(level.size);
            total_cost += level_size * level.price;
            remaining_size -= level_size;
        }

        if remaining_size > Decimal::ZERO {
            return None; // Insufficient liquidity
        }

        let average_price = total_cost / size;
        let best_price = match side {
            OrderSide::Buy => self.best_ask()?,
            OrderSide::Sell => self.best_bid()?,
        };

        Some((average_price - best_price).abs() / best_price)
    }
}

/// Order fill information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderFill {
    pub id: String,
    pub order_id: String,
    pub price: Decimal,
    pub size: Decimal,
    pub side: OrderSide,
    pub timestamp: DateTime<Utc>,
    pub fee: Decimal,
    pub fee_currency: String,
}

impl OrderFill {
    pub fn value(&self) -> Decimal {
        self.price * self.size
    }

    pub fn net_value(&self) -> Decimal {
        self.value() - self.fee
    }
}

/// Order representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Order {
    pub id: String,
    pub market_id: String,
    pub outcome_id: String,
    pub side: OrderSide,
    #[serde(rename = "type")]
    pub order_type: OrderType,
    pub size: Decimal,
    pub price: Option<Decimal>,
    pub filled: Decimal,
    pub remaining: Decimal,
    pub status: OrderStatus,
    pub time_in_force: TimeInForce,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub expires_at: Option<DateTime<Utc>>,
    pub fills: Vec<OrderFill>,
    pub fee_rate: Decimal,
    pub client_order_id: Option<String>,
}

impl Order {
    pub fn fill_percentage(&self) -> Decimal {
        if self.size == Decimal::ZERO {
            Decimal::ZERO
        } else {
            (self.filled / self.size) * Decimal::from(100)
        }
    }

    pub fn is_active(&self) -> bool {
        self.status.is_active()
    }

    pub fn is_complete(&self) -> bool {
        self.status.is_complete()
    }

    pub fn can_cancel(&self) -> bool {
        self.status.can_cancel()
    }

    pub fn total_fees(&self) -> Decimal {
        self.fills.iter().map(|fill| fill.fee).sum()
    }

    pub fn average_fill_price(&self) -> Option<Decimal> {
        if self.fills.is_empty() || self.filled == Decimal::ZERO {
            return None;
        }

        let total_value: Decimal = self
            .fills
            .iter()
            .map(|fill| fill.price * fill.size)
            .sum();
        Some(total_value / self.filled)
    }

    pub fn notional_value(&self) -> Decimal {
        self.price.unwrap_or(Decimal::ZERO) * self.size
    }

    pub fn remaining_value(&self) -> Decimal {
        self.price.unwrap_or(Decimal::ZERO) * self.remaining
    }
}

/// Order request for creating new orders
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderRequest {
    pub market_id: String,
    pub outcome_id: String,
    pub side: OrderSide,
    #[serde(rename = "type")]
    pub order_type: OrderType,
    pub size: Decimal,
    pub price: Option<Decimal>,
    pub time_in_force: Option<TimeInForce>,
    pub client_order_id: Option<String>,
}

impl OrderRequest {
    pub fn validate(&self) -> Result<(), String> {
        if self.size <= Decimal::ZERO {
            return Err("Order size must be positive".to_string());
        }

        if self.order_type == OrderType::Limit && self.price.is_none() {
            return Err("Limit orders require a price".to_string());
        }

        if let Some(price) = self.price {
            if price <= Decimal::ZERO || price > Decimal::ONE {
                return Err("Price must be between 0 and 1".to_string());
            }
        }

        Ok(())
    }
}

/// Order response from API
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderResponse {
    pub order: Order,
    pub success: bool,
    pub message: Option<String>,
}

/// Position in a market
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    pub market_id: String,
    pub outcome_id: String,
    pub size: Decimal,
    pub average_price: Decimal,
    pub current_price: Decimal,
    pub unrealized_pnl: Decimal,
    pub realized_pnl: Decimal,
    pub total_fees: Decimal,
}

impl Position {
    pub fn current_value(&self) -> Decimal {
        self.size * self.current_price
    }

    pub fn cost_basis(&self) -> Decimal {
        self.size * self.average_price
    }

    pub fn total_pnl(&self) -> Decimal {
        self.unrealized_pnl + self.realized_pnl - self.total_fees
    }

    pub fn pnl_percentage(&self) -> Decimal {
        let cost = self.cost_basis();
        if cost == Decimal::ZERO {
            Decimal::ZERO
        } else {
            (self.total_pnl() / cost) * Decimal::from(100)
        }
    }
}

/// Market statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketStats {
    pub volume_24h: Decimal,
    pub trades_24h: u64,
    pub liquidity: Decimal,
    pub participants: u64,
    pub price_change_24h: Decimal,
}

/// WebSocket message types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum WebSocketMessage {
    #[serde(rename = "orderbook")]
    OrderBook {
        market_id: String,
        outcome_id: String,
        bids: Vec<OrderBookLevel>,
        asks: Vec<OrderBookLevel>,
        timestamp: DateTime<Utc>,
    },
    #[serde(rename = "trade")]
    Trade {
        market_id: String,
        outcome_id: String,
        price: Decimal,
        size: Decimal,
        side: OrderSide,
        timestamp: DateTime<Utc>,
    },
    #[serde(rename = "market_update")]
    MarketUpdate {
        market_id: String,
        field: String,
        value: serde_json::Value,
        timestamp: DateTime<Utc>,
    },
    #[serde(rename = "order_update")]
    OrderUpdate {
        order_id: String,
        status: OrderStatus,
        filled: Decimal,
        remaining: Decimal,
        timestamp: DateTime<Utc>,
    },
}

/// Subscription request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Subscription {
    pub channel: String,
    pub market_id: Option<String>,
    pub outcome_id: Option<String>,
}
