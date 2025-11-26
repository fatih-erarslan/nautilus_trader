//! Integration tests for order execution

use nt_execution::{OrderRequest, OrderSide, OrderType, TimeInForce};
use rust_decimal_macros::dec;

mod mocks {
    include!("../mocks/mock_broker.rs");
}

use mocks::MockBrokerClient;

#[tokio::test]
async fn test_order_execution_market_buy() {
    let broker = MockBrokerClient::new();

    let order = OrderRequest {
        symbol: "AAPL".to_string(),
        qty: Some(dec!(100)),
        side: OrderSide::Buy,
        order_type: OrderType::Market,
        time_in_force: TimeInForce::Day,
        limit_price: None,
        stop_price: None,
        extended_hours: false,
        client_order_id: Some("test-1".to_string()),
        order_class: None,
        take_profit: None,
        stop_loss: None,
    };

    let response = broker.place_order(order).await.expect("Order should succeed");

    assert_eq!(response.symbol, "AAPL");
    assert_eq!(response.qty, Some(dec!(100)));
    assert_eq!(response.side, OrderSide::Buy);
    assert_eq!(broker.orders_count(), 1);
}

#[tokio::test]
async fn test_order_execution_limit_sell() {
    let broker = MockBrokerClient::new();

    let order = OrderRequest {
        symbol: "TSLA".to_string(),
        qty: Some(dec!(50)),
        side: OrderSide::Sell,
        order_type: OrderType::Limit,
        time_in_force: TimeInForce::Day,
        limit_price: Some(dec!(250)),
        stop_price: None,
        extended_hours: false,
        client_order_id: Some("test-2".to_string()),
        order_class: None,
        take_profit: None,
        stop_loss: None,
    };

    let response = broker.place_order(order).await.expect("Order should succeed");

    assert_eq!(response.symbol, "TSLA");
    assert_eq!(response.limit_price, Some(dec!(250)));
}

#[tokio::test]
async fn test_order_cancellation() {
    let broker = MockBrokerClient::new();

    let order = OrderRequest {
        symbol: "SPY".to_string(),
        qty: Some(dec!(10)),
        side: OrderSide::Buy,
        order_type: OrderType::Market,
        time_in_force: TimeInForce::Day,
        limit_price: None,
        stop_price: None,
        extended_hours: false,
        client_order_id: Some("test-3".to_string()),
        order_class: None,
        take_profit: None,
        stop_loss: None,
    };

    let response = broker.place_order(order).await.expect("Order should succeed");
    let order_id = response.id.clone();

    broker.cancel_order(&order_id).await.expect("Cancellation should succeed");

    let canceled_order = broker.get_order(&order_id).await.expect("Should get order");
    assert_eq!(canceled_order.status, nt_execution::OrderStatus::Canceled);
}

#[tokio::test]
async fn test_order_get_account() {
    let broker = MockBrokerClient::new();

    let account = broker.get_account().await.expect("Should get account");

    assert_eq!(account.cash, dec!(100000));
    assert_eq!(account.currency, "USD");
}

#[tokio::test]
async fn test_order_execution_with_latency() {
    let broker = MockBrokerClient::new();
    broker.set_latency(100); // 100ms latency

    let order = OrderRequest {
        symbol: "AAPL".to_string(),
        qty: Some(dec!(10)),
        side: OrderSide::Buy,
        order_type: OrderType::Market,
        time_in_force: TimeInForce::Day,
        limit_price: None,
        stop_price: None,
        extended_hours: false,
        client_order_id: Some("test-latency".to_string()),
        order_class: None,
        take_profit: None,
        stop_loss: None,
    };

    let start = std::time::Instant::now();
    let _ = broker.place_order(order).await;
    let elapsed = start.elapsed();

    assert!(elapsed.as_millis() >= 100, "Should respect latency setting");
}

#[tokio::test]
async fn test_order_execution_failure_mode() {
    let broker = MockBrokerClient::new();
    broker.set_failure_mode(true);

    let order = OrderRequest {
        symbol: "AAPL".to_string(),
        qty: Some(dec!(10)),
        side: OrderSide::Buy,
        order_type: OrderType::Market,
        time_in_force: TimeInForce::Day,
        limit_price: None,
        stop_price: None,
        extended_hours: false,
        client_order_id: Some("test-fail".to_string()),
        order_class: None,
        take_profit: None,
        stop_loss: None,
    };

    let result = broker.place_order(order).await;
    assert!(result.is_err(), "Should fail when failure mode is set");
}
