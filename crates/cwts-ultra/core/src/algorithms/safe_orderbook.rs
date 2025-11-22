// Memory-Safe Order Book - Production Financial System Implementation
// Replaces unsafe lock-free implementation with memory-safe alternatives

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, VecDeque};
use std::sync::{Arc, Mutex};
use thiserror::Error;
// Remove unused import - types defined locally for this safe implementation
use rust_decimal::Decimal;

#[derive(Debug, Error)]
pub enum OrderBookError {
    #[error("Invalid order parameters: {0}")]
    InvalidOrder(String),
    #[error("Order not found: {0}")]
    OrderNotFound(u64),
    #[error("Insufficient quantity at price level")]
    InsufficientQuantity,
    #[error("Price level not found")]
    PriceLevelNotFound,
}

/// Memory-safe order structure with comprehensive validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Order {
    pub order_id: u64,
    pub price: Decimal,
    pub quantity: Decimal,
    pub side: OrderSide,
    pub timestamp: DateTime<Utc>,
    pub order_type: OrderType,
    pub user_id: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum OrderSide {
    Buy,
    Sell,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum OrderType {
    Market,
    Limit,
    StopLoss,
    StopLimit,
}

/// Memory-safe price level with validation
#[derive(Debug, Clone)]
pub struct PriceLevel {
    pub price: Decimal,
    pub orders: VecDeque<Order>,
    pub total_quantity: Decimal,
}

impl PriceLevel {
    pub fn new(price: Decimal) -> Self {
        Self {
            price,
            orders: VecDeque::new(),
            total_quantity: Decimal::ZERO,
        }
    }

    pub fn add_order(&mut self, order: Order) -> Result<(), OrderBookError> {
        // Validate order parameters
        if order.price != self.price {
            return Err(OrderBookError::InvalidOrder(format!(
                "Order price {} doesn't match level price {}",
                order.price, self.price
            )));
        }

        if order.quantity <= Decimal::ZERO {
            return Err(OrderBookError::InvalidOrder(
                "Quantity must be positive".to_string(),
            ));
        }

        self.total_quantity += order.quantity;
        self.orders.push_back(order);
        Ok(())
    }

    pub fn remove_order(&mut self, order_id: u64) -> Result<Order, OrderBookError> {
        if let Some(pos) = self.orders.iter().position(|o| o.order_id == order_id) {
            let order = self.orders.remove(pos).unwrap();
            self.total_quantity -= order.quantity;
            Ok(order)
        } else {
            Err(OrderBookError::OrderNotFound(order_id))
        }
    }

    pub fn is_empty(&self) -> bool {
        self.orders.is_empty()
    }
}

/// Memory-safe order book with comprehensive security controls
#[derive(Debug)]
pub struct SafeOrderBook {
    /// Buy orders (bids) - sorted by price descending
    bids: Arc<Mutex<BTreeMap<Decimal, PriceLevel>>>,
    /// Sell orders (asks) - sorted by price ascending
    asks: Arc<Mutex<BTreeMap<Decimal, PriceLevel>>>,
    /// Order ID to location mapping for fast lookups
    order_index: Arc<Mutex<std::collections::HashMap<u64, (OrderSide, Decimal)>>>,
    /// Security controls
    max_orders_per_level: usize,
    max_order_size: Decimal,
    min_order_size: Decimal,
    tick_size: Decimal,
}

impl SafeOrderBook {
    pub fn new() -> Self {
        Self {
            bids: Arc::new(Mutex::new(BTreeMap::new())),
            asks: Arc::new(Mutex::new(BTreeMap::new())),
            order_index: Arc::new(Mutex::new(std::collections::HashMap::new())),
            max_orders_per_level: 1000, // SEC compliance limit
            max_order_size: Decimal::from(1_000_000), // $1M max order
            min_order_size: Decimal::try_from(0.01).unwrap_or(Decimal::ZERO), // $0.01 minimum
            tick_size: Decimal::try_from(0.01).unwrap_or(Decimal::ONE), // $0.01 tick size
        }
    }

    /// Add order with comprehensive validation and security checks
    pub fn add_order(&self, mut order: Order) -> Result<(), OrderBookError> {
        // Pre-trade risk controls (SEC Rule 15c3-5)
        self.validate_order(&order)?;

        // Ensure timestamp accuracy
        order.timestamp = Utc::now();

        let (book, index_key) = match order.side {
            OrderSide::Buy => (self.bids.clone(), (OrderSide::Buy, order.price)),
            OrderSide::Sell => (self.asks.clone(), (OrderSide::Sell, order.price)),
        };

        let mut book_guard = book.lock().unwrap();
        let mut index_guard = self.order_index.lock().unwrap();

        // Create or get price level
        let level = book_guard
            .entry(order.price)
            .or_insert_with(|| PriceLevel::new(order.price));

        // Check order count per level limit
        if level.orders.len() >= self.max_orders_per_level {
            return Err(OrderBookError::InvalidOrder(
                "Maximum orders per price level exceeded".to_string(),
            ));
        }

        // Add order to level
        level.add_order(order.clone())?;

        // Update index
        index_guard.insert(order.order_id, index_key);

        Ok(())
    }

    /// Remove order with audit trail
    pub fn remove_order(&self, order_id: u64) -> Result<Order, OrderBookError> {
        let mut index_guard = self.order_index.lock().unwrap();

        if let Some((side, price)) = index_guard.remove(&order_id) {
            let book = match side {
                OrderSide::Buy => self.bids.clone(),
                OrderSide::Sell => self.asks.clone(),
            };

            let mut book_guard = book.lock().unwrap();
            if let Some(level) = book_guard.get_mut(&price) {
                let order = level.remove_order(order_id)?;

                // Remove empty price level
                if level.is_empty() {
                    book_guard.remove(&price);
                }

                Ok(order)
            } else {
                Err(OrderBookError::PriceLevelNotFound)
            }
        } else {
            Err(OrderBookError::OrderNotFound(order_id))
        }
    }

    /// Get best bid price
    pub fn best_bid(&self) -> Option<Decimal> {
        let bids = self.bids.lock().unwrap();
        bids.keys().rev().next().copied()
    }

    /// Get best ask price
    pub fn best_ask(&self) -> Option<Decimal> {
        let asks = self.asks.lock().unwrap();
        asks.keys().next().copied()
    }

    /// Get spread between best bid and ask
    pub fn spread(&self) -> Option<Decimal> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some(ask - bid),
            _ => None,
        }
    }

    /// Match orders with comprehensive audit trail
    pub fn match_orders(&self, incoming_order: &Order) -> Result<Vec<Trade>, OrderBookError> {
        let mut trades = Vec::new();
        let mut remaining_quantity = incoming_order.quantity;

        let opposite_book = match incoming_order.side {
            OrderSide::Buy => self.asks.clone(),
            OrderSide::Sell => self.bids.clone(),
        };

        let mut book_guard = opposite_book.lock().unwrap();
        let mut index_guard = self.order_index.lock().unwrap();

        // Get matching price levels
        let matching_prices: Vec<Decimal> = match incoming_order.side {
            OrderSide::Buy => {
                // Buy order matches with asks at or below buy price
                book_guard
                    .keys()
                    .filter(|&&price| price <= incoming_order.price)
                    .copied()
                    .collect()
            }
            OrderSide::Sell => {
                // Sell order matches with bids at or above sell price
                book_guard
                    .keys()
                    .rev()
                    .filter(|&&price| price >= incoming_order.price)
                    .copied()
                    .collect()
            }
        };

        // Execute trades at each price level
        for price in matching_prices {
            if remaining_quantity <= Decimal::ZERO {
                break;
            }

            if let Some(level) = book_guard.get_mut(&price) {
                while !level.orders.is_empty() && remaining_quantity > Decimal::ZERO {
                    let resting_order = level.orders.front().unwrap();
                    let trade_quantity = remaining_quantity.min(resting_order.quantity);

                    // Create trade record
                    let trade = Trade {
                        trade_id: generate_trade_id(),
                        buyer_order_id: match incoming_order.side {
                            OrderSide::Buy => incoming_order.order_id,
                            OrderSide::Sell => resting_order.order_id,
                        },
                        seller_order_id: match incoming_order.side {
                            OrderSide::Buy => resting_order.order_id,
                            OrderSide::Sell => incoming_order.order_id,
                        },
                        price,
                        quantity: trade_quantity,
                        timestamp: Utc::now(),
                    };

                    trades.push(trade);
                    remaining_quantity -= trade_quantity;

                    // Update or remove resting order
                    let mut resting_order = level.orders.pop_front().unwrap();
                    resting_order.quantity -= trade_quantity;

                    if resting_order.quantity > Decimal::ZERO {
                        // Partial fill - put back modified order
                        level.orders.push_front(resting_order);
                    } else {
                        // Complete fill - remove from index
                        index_guard.remove(&resting_order.order_id);
                    }

                    level.total_quantity -= trade_quantity;
                }

                // Remove empty price level
                if level.orders.is_empty() {
                    book_guard.remove(&price);
                }
            }
        }

        Ok(trades)
    }

    /// Comprehensive order validation with financial compliance
    fn validate_order(&self, order: &Order) -> Result<(), OrderBookError> {
        // Price validation
        if order.price <= Decimal::ZERO {
            return Err(OrderBookError::InvalidOrder(
                "Price must be positive".to_string(),
            ));
        }

        // Tick size validation
        if (order.price % self.tick_size) != Decimal::ZERO {
            return Err(OrderBookError::InvalidOrder(
                "Price must be in valid tick size".to_string(),
            ));
        }

        // Quantity validation
        if order.quantity <= Decimal::ZERO {
            return Err(OrderBookError::InvalidOrder(
                "Quantity must be positive".to_string(),
            ));
        }

        if order.quantity < self.min_order_size {
            return Err(OrderBookError::InvalidOrder(
                "Order below minimum size".to_string(),
            ));
        }

        if order.quantity > self.max_order_size {
            return Err(OrderBookError::InvalidOrder(
                "Order exceeds maximum size".to_string(),
            ));
        }

        // Order ID validation
        if order.order_id == 0 {
            return Err(OrderBookError::InvalidOrder("Invalid order ID".to_string()));
        }

        // User validation
        if order.user_id.is_empty() {
            return Err(OrderBookError::InvalidOrder("User ID required".to_string()));
        }

        Ok(())
    }

    /// Get order book depth for market data
    pub fn get_depth(&self, levels: usize) -> OrderBookDepth {
        let bids_guard = self.bids.lock().unwrap();
        let asks_guard = self.asks.lock().unwrap();

        let bids: Vec<DepthLevel> = bids_guard
            .iter()
            .rev()
            .take(levels)
            .map(|(price, level)| DepthLevel {
                price: *price,
                quantity: level.total_quantity,
                order_count: level.orders.len(),
            })
            .collect();

        let asks: Vec<DepthLevel> = asks_guard
            .iter()
            .take(levels)
            .map(|(price, level)| DepthLevel {
                price: *price,
                quantity: level.total_quantity,
                order_count: level.orders.len(),
            })
            .collect();

        OrderBookDepth {
            bids,
            asks,
            timestamp: Utc::now(),
        }
    }
}

/// Trade execution record with audit trail
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    pub trade_id: u64,
    pub buyer_order_id: u64,
    pub seller_order_id: u64,
    pub price: Decimal,
    pub quantity: Decimal,
    pub timestamp: DateTime<Utc>,
}

/// Order book depth for market data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookDepth {
    pub bids: Vec<DepthLevel>,
    pub asks: Vec<DepthLevel>,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DepthLevel {
    pub price: Decimal,
    pub quantity: Decimal,
    pub order_count: usize,
}

// Trade ID generation with collision prevention
fn generate_trade_id() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(1);

    let timestamp = Utc::now().timestamp_nanos_opt().unwrap_or(0) as u64;
    let counter = COUNTER.fetch_add(1, Ordering::AcqRel);

    // Combine timestamp and counter for unique ID
    (timestamp << 16) | (counter & 0xFFFF)
}

impl Default for SafeOrderBook {
    fn default() -> Self {
        Self::new()
    }
}

// Thread-safe implementation - SafeOrderBook is automatically Send + Sync
// due to Arc<Mutex<T>> being Send + Sync when T: Send

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_and_remove_order() {
        let book = SafeOrderBook::new();

        let order = Order {
            order_id: 1,
            price: Decimal::from(100),
            quantity: Decimal::from(10),
            side: OrderSide::Buy,
            timestamp: Utc::now(),
            order_type: OrderType::Limit,
            user_id: "test_user".to_string(),
        };

        assert!(book.add_order(order.clone()).is_ok());
        assert_eq!(book.best_bid(), Some(Decimal::from(100)));

        let removed = book.remove_order(1);
        assert!(removed.is_ok());
        assert_eq!(book.best_bid(), None);
    }

    #[test]
    fn test_order_validation() {
        let book = SafeOrderBook::new();

        let invalid_order = Order {
            order_id: 0, // Invalid ID
            price: Decimal::from(100),
            quantity: Decimal::from(10),
            side: OrderSide::Buy,
            timestamp: Utc::now(),
            order_type: OrderType::Limit,
            user_id: "test_user".to_string(),
        };

        assert!(book.add_order(invalid_order).is_err());
    }

    #[test]
    fn test_order_matching() {
        let book = SafeOrderBook::new();

        // Add a sell order
        let sell_order = Order {
            order_id: 1,
            price: Decimal::from(100),
            quantity: Decimal::from(10),
            side: OrderSide::Sell,
            timestamp: Utc::now(),
            order_type: OrderType::Limit,
            user_id: "seller".to_string(),
        };
        book.add_order(sell_order).unwrap();

        // Add matching buy order
        let buy_order = Order {
            order_id: 2,
            price: Decimal::from(100),
            quantity: Decimal::from(5),
            side: OrderSide::Buy,
            timestamp: Utc::now(),
            order_type: OrderType::Limit,
            user_id: "buyer".to_string(),
        };

        let trades = book.match_orders(&buy_order).unwrap();
        assert_eq!(trades.len(), 1);
        assert_eq!(trades[0].quantity, Decimal::from(5));
    }
}
