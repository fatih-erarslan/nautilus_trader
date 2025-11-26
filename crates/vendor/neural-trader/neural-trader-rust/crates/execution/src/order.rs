use crate::types::{ExecutionError, OrderSide, OrderStatus, OrderType, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Order structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Order {
    pub id: String,
    pub symbol: String,
    pub side: OrderSide,
    pub quantity: i32,
    pub order_type: OrderType,
    pub status: OrderStatus,
    pub filled_quantity: i32,
    pub avg_fill_price: Option<f64>,
    pub limit_price: Option<f64>,
    pub stop_price: Option<f64>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub filled_at: Option<DateTime<Utc>>,
    pub broker_order_id: Option<String>,
    pub paper_trading: bool,
}

impl Order {
    /// Create a new order
    pub fn new(
        symbol: String,
        side: OrderSide,
        quantity: i32,
        order_type: OrderType,
        limit_price: Option<f64>,
        stop_price: Option<f64>,
        paper_trading: bool,
    ) -> Result<Self> {
        // Validate inputs
        Self::validate_symbol(&symbol)?;
        Self::validate_quantity(quantity)?;
        Self::validate_prices(order_type, limit_price, stop_price)?;

        let now = Utc::now();
        Ok(Self {
            id: Uuid::new_v4().to_string(),
            symbol,
            side,
            quantity,
            order_type,
            status: OrderStatus::Pending,
            filled_quantity: 0,
            avg_fill_price: None,
            limit_price,
            stop_price,
            created_at: now,
            updated_at: now,
            filled_at: None,
            broker_order_id: None,
            paper_trading,
        })
    }

    /// Validate symbol format
    fn validate_symbol(symbol: &str) -> Result<()> {
        if symbol.is_empty() {
            return Err(ExecutionError::InvalidSymbol(
                "Symbol cannot be empty".to_string(),
            ));
        }

        // Symbol should be alphanumeric and uppercase
        if !symbol.chars().all(|c| c.is_alphanumeric() || c == '.') {
            return Err(ExecutionError::InvalidSymbol(format!(
                "Symbol contains invalid characters: {}",
                symbol
            )));
        }

        if symbol.len() > 10 {
            return Err(ExecutionError::InvalidSymbol(format!(
                "Symbol too long: {}",
                symbol
            )));
        }

        Ok(())
    }

    /// Validate quantity
    fn validate_quantity(quantity: i32) -> Result<()> {
        if quantity <= 0 {
            return Err(ExecutionError::InvalidQuantity(quantity));
        }
        Ok(())
    }

    /// Validate prices based on order type
    fn validate_prices(
        order_type: OrderType,
        limit_price: Option<f64>,
        stop_price: Option<f64>,
    ) -> Result<()> {
        match order_type {
            OrderType::Limit => {
                if limit_price.is_none() {
                    return Err(ExecutionError::LimitPriceRequired);
                }
                if let Some(price) = limit_price {
                    if price <= 0.0 {
                        return Err(ExecutionError::ValidationFailed(
                            "Limit price must be positive".to_string(),
                        ));
                    }
                }
            }
            OrderType::Stop => {
                if stop_price.is_none() {
                    return Err(ExecutionError::ValidationFailed(
                        "Stop price required for stop orders".to_string(),
                    ));
                }
                if let Some(price) = stop_price {
                    if price <= 0.0 {
                        return Err(ExecutionError::ValidationFailed(
                            "Stop price must be positive".to_string(),
                        ));
                    }
                }
            }
            OrderType::StopLimit => {
                if limit_price.is_none() || stop_price.is_none() {
                    return Err(ExecutionError::ValidationFailed(
                        "Both limit and stop prices required for stop-limit orders".to_string(),
                    ));
                }
            }
            OrderType::Market => {
                // Market orders don't require prices
            }
        }
        Ok(())
    }

    /// Update order status
    pub fn update_status(&mut self, status: OrderStatus) {
        self.status = status;
        self.updated_at = Utc::now();

        if status == OrderStatus::Filled {
            self.filled_at = Some(Utc::now());
            self.filled_quantity = self.quantity;
        }
    }

    /// Update filled quantity and average fill price
    pub fn update_fill(&mut self, filled_qty: i32, avg_price: f64) {
        self.filled_quantity = filled_qty;
        self.avg_fill_price = Some(avg_price);
        self.updated_at = Utc::now();

        if filled_qty >= self.quantity {
            self.status = OrderStatus::Filled;
            self.filled_at = Some(Utc::now());
        } else if filled_qty > 0 {
            self.status = OrderStatus::PartiallyFilled;
        }
    }

    /// Set broker order ID
    pub fn set_broker_order_id(&mut self, broker_order_id: String) {
        self.broker_order_id = Some(broker_order_id);
        self.updated_at = Utc::now();
    }

    /// Check if order is complete
    pub fn is_complete(&self) -> bool {
        matches!(
            self.status,
            OrderStatus::Filled | OrderStatus::Cancelled | OrderStatus::Rejected | OrderStatus::Expired
        )
    }

    /// Check if order is active
    pub fn is_active(&self) -> bool {
        matches!(
            self.status,
            OrderStatus::Pending | OrderStatus::Submitted | OrderStatus::PartiallyFilled
        )
    }

    /// Get remaining quantity
    pub fn remaining_quantity(&self) -> i32 {
        self.quantity - self.filled_quantity
    }

    /// Calculate total value
    pub fn total_value(&self) -> Option<f64> {
        self.avg_fill_price
            .map(|price| price * self.filled_quantity as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_market_order_creation() {
        let order = Order::new(
            "AAPL".to_string(),
            OrderSide::Buy,
            100,
            OrderType::Market,
            None,
            None,
            true,
        );

        assert!(order.is_ok());
        let order = order.unwrap();
        assert_eq!(order.symbol, "AAPL");
        assert_eq!(order.quantity, 100);
        assert_eq!(order.status, OrderStatus::Pending);
    }

    #[test]
    fn test_limit_order_requires_price() {
        let order = Order::new(
            "AAPL".to_string(),
            OrderSide::Buy,
            100,
            OrderType::Limit,
            None,
            None,
            true,
        );

        assert!(order.is_err());
    }

    #[test]
    fn test_invalid_quantity() {
        let order = Order::new(
            "AAPL".to_string(),
            OrderSide::Buy,
            -10,
            OrderType::Market,
            None,
            None,
            true,
        );

        assert!(order.is_err());
    }

    #[test]
    fn test_invalid_symbol() {
        let order = Order::new(
            "".to_string(),
            OrderSide::Buy,
            100,
            OrderType::Market,
            None,
            None,
            true,
        );

        assert!(order.is_err());
    }

    #[test]
    fn test_order_fill_update() {
        let mut order = Order::new(
            "AAPL".to_string(),
            OrderSide::Buy,
            100,
            OrderType::Market,
            None,
            None,
            true,
        )
        .unwrap();

        order.update_fill(50, 150.0);
        assert_eq!(order.filled_quantity, 50);
        assert_eq!(order.status, OrderStatus::PartiallyFilled);

        order.update_fill(100, 150.0);
        assert_eq!(order.status, OrderStatus::Filled);
        assert!(order.filled_at.is_some());
    }
}
