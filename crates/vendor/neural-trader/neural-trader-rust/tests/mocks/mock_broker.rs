//! Mock broker for testing order execution

use async_trait::async_trait;
use chrono::Utc;
use nt_execution::{
    Account, BrokerClient, BrokerError, HealthStatus, OrderFilter, OrderRequest, OrderResponse,
    OrderStatus, Position, PositionSide,
};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::sync::{Arc, Mutex};
use uuid::Uuid;

/// Mock broker client for testing
#[derive(Clone)]
pub struct MockBrokerClient {
    state: Arc<Mutex<MockBrokerState>>,
}

#[derive(Debug)]
struct MockBrokerState {
    orders: Vec<OrderResponse>,
    positions: Vec<Position>,
    account: Account,
    should_fail: bool,
    latency_ms: u64,
}

impl MockBrokerClient {
    /// Create a new mock broker with default account
    pub fn new() -> Self {
        Self {
            state: Arc::new(Mutex::new(MockBrokerState {
                orders: Vec::new(),
                positions: Vec::new(),
                account: Account {
                    account_id: "mock-account".to_string(),
                    cash: dec!(100000),
                    portfolio_value: dec!(100000),
                    buying_power: dec!(100000),
                    equity: dec!(100000),
                    last_equity: dec!(100000),
                    multiplier: "1".to_string(),
                    currency: "USD".to_string(),
                    shorting_enabled: true,
                    long_market_value: Decimal::ZERO,
                    short_market_value: Decimal::ZERO,
                    initial_margin: Decimal::ZERO,
                    maintenance_margin: Decimal::ZERO,
                    day_trading_buying_power: dec!(100000),
                    daytrade_count: 0,
                },
                should_fail: false,
                latency_ms: 0,
            })),
        }
    }

    /// Configure mock to fail on next operation
    pub fn set_failure_mode(&self, should_fail: bool) {
        self.state.lock().unwrap().should_fail = should_fail;
    }

    /// Configure mock latency
    pub fn set_latency(&self, latency_ms: u64) {
        self.state.lock().unwrap().latency_ms = latency_ms;
    }

    /// Add a position to the mock broker
    pub fn add_position(&self, position: Position) {
        self.state.lock().unwrap().positions.push(position);
    }

    /// Get number of orders placed
    pub fn orders_count(&self) -> usize {
        self.state.lock().unwrap().orders.len()
    }

    /// Get all orders
    pub fn get_all_orders(&self) -> Vec<OrderResponse> {
        self.state.lock().unwrap().orders.clone()
    }

    /// Clear all orders and positions
    pub fn reset(&self) {
        let mut state = self.state.lock().unwrap();
        state.orders.clear();
        state.positions.clear();
    }
}

impl Default for MockBrokerClient {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl BrokerClient for MockBrokerClient {
    async fn get_account(&self) -> Result<Account, BrokerError> {
        let state = self.state.lock().unwrap();

        if state.should_fail {
            return Err(BrokerError::ApiError("Mock failure".to_string()));
        }

        if state.latency_ms > 0 {
            tokio::time::sleep(tokio::time::Duration::from_millis(state.latency_ms)).await;
        }

        Ok(state.account.clone())
    }

    async fn get_positions(&self) -> Result<Vec<Position>, BrokerError> {
        let state = self.state.lock().unwrap();

        if state.should_fail {
            return Err(BrokerError::ApiError("Mock failure".to_string()));
        }

        Ok(state.positions.clone())
    }

    async fn place_order(&self, order: OrderRequest) -> Result<OrderResponse, BrokerError> {
        let mut state = self.state.lock().unwrap();

        if state.should_fail {
            return Err(BrokerError::ApiError("Mock failure".to_string()));
        }

        if state.latency_ms > 0 {
            drop(state);
            tokio::time::sleep(tokio::time::Duration::from_millis(self.state.lock().unwrap().latency_ms)).await;
            state = self.state.lock().unwrap();
        }

        let response = OrderResponse {
            id: Uuid::new_v4().to_string(),
            client_order_id: order.client_order_id.clone(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
            submitted_at: Some(Utc::now()),
            filled_at: Some(Utc::now()),
            expired_at: None,
            canceled_at: None,
            failed_at: None,
            replaced_at: None,
            replaced_by: None,
            replaces: None,
            asset_id: None,
            symbol: order.symbol.clone(),
            asset_class: "us_equity".to_string(),
            notional: None,
            qty: order.qty,
            filled_qty: order.qty,
            filled_avg_price: Some(order.limit_price.unwrap_or(dec!(100))),
            order_class: order.order_class.clone(),
            order_type: order.order_type.clone(),
            side: order.side.clone(),
            time_in_force: order.time_in_force.clone(),
            limit_price: order.limit_price,
            stop_price: order.stop_price,
            status: OrderStatus::Filled,
            extended_hours: order.extended_hours,
            legs: None,
            trail_percent: None,
            trail_price: None,
            hwm: None,
        };

        state.orders.push(response.clone());
        Ok(response)
    }

    async fn cancel_order(&self, order_id: &str) -> Result<(), BrokerError> {
        let mut state = self.state.lock().unwrap();

        if state.should_fail {
            return Err(BrokerError::ApiError("Mock failure".to_string()));
        }

        if let Some(order) = state.orders.iter_mut().find(|o| o.id == order_id) {
            order.status = OrderStatus::Canceled;
            order.canceled_at = Some(Utc::now());
            Ok(())
        } else {
            Err(BrokerError::OrderNotFound(order_id.to_string()))
        }
    }

    async fn get_order(&self, order_id: &str) -> Result<OrderResponse, BrokerError> {
        let state = self.state.lock().unwrap();

        if state.should_fail {
            return Err(BrokerError::ApiError("Mock failure".to_string()));
        }

        state
            .orders
            .iter()
            .find(|o| o.id == order_id)
            .cloned()
            .ok_or_else(|| BrokerError::OrderNotFound(order_id.to_string()))
    }

    async fn list_orders(&self, _filter: OrderFilter) -> Result<Vec<OrderResponse>, BrokerError> {
        let state = self.state.lock().unwrap();

        if state.should_fail {
            return Err(BrokerError::ApiError("Mock failure".to_string()));
        }

        Ok(state.orders.clone())
    }

    async fn health_check(&self) -> Result<HealthStatus, BrokerError> {
        let state = self.state.lock().unwrap();

        if state.should_fail {
            return Ok(HealthStatus::Unhealthy);
        }

        Ok(HealthStatus::Healthy)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nt_execution::{OrderSide, OrderType, TimeInForce};

    #[tokio::test]
    async fn test_mock_broker_place_order() {
        let broker = MockBrokerClient::new();

        let order_request = OrderRequest {
            symbol: "AAPL".to_string(),
            qty: Some(dec!(10)),
            side: OrderSide::Buy,
            order_type: OrderType::Market,
            time_in_force: TimeInForce::Day,
            limit_price: None,
            stop_price: None,
            extended_hours: false,
            client_order_id: Some("test-order-1".to_string()),
            order_class: None,
            take_profit: None,
            stop_loss: None,
        };

        let response = broker.place_order(order_request).await.unwrap();

        assert_eq!(response.symbol, "AAPL");
        assert_eq!(response.qty, Some(dec!(10)));
        assert_eq!(response.status, OrderStatus::Filled);
        assert_eq!(broker.orders_count(), 1);
    }

    #[tokio::test]
    async fn test_mock_broker_failure_mode() {
        let broker = MockBrokerClient::new();
        broker.set_failure_mode(true);

        let result = broker.get_account().await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_mock_broker_latency() {
        let broker = MockBrokerClient::new();
        broker.set_latency(50);

        let start = std::time::Instant::now();
        let _ = broker.get_account().await;
        let elapsed = start.elapsed();

        assert!(elapsed.as_millis() >= 50);
    }
}
