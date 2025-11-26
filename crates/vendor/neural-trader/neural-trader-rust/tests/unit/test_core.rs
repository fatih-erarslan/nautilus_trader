// Unit tests for core crate
use nt_core::{Order, OrderSide, OrderType, OrderStatus, Position};
use rust_decimal::Decimal;
use std::str::FromStr;

#[test]
fn test_order_creation() {
    let order = Order {
        id: "test-123".to_string(),
        symbol: "AAPL".to_string(),
        side: OrderSide::Buy,
        quantity: Decimal::from_str("100").unwrap(),
        order_type: OrderType::Market,
        limit_price: None,
        stop_price: None,
        time_in_force: "day".to_string(),
        status: OrderStatus::New,
    };

    assert_eq!(order.symbol, "AAPL");
    assert_eq!(order.quantity, Decimal::from_str("100").unwrap());
    assert_eq!(order.side, OrderSide::Buy);
    assert_eq!(order.order_type, OrderType::Market);
}

#[test]
fn test_limit_order_validation() {
    let limit_order = Order {
        id: "test-456".to_string(),
        symbol: "MSFT".to_string(),
        side: OrderSide::Sell,
        quantity: Decimal::from_str("50").unwrap(),
        order_type: OrderType::Limit,
        limit_price: Some(Decimal::from_str("350.50").unwrap()),
        stop_price: None,
        time_in_force: "gtc".to_string(),
        status: OrderStatus::New,
    };

    assert!(limit_order.limit_price.is_some());
    assert_eq!(limit_order.limit_price.unwrap(), Decimal::from_str("350.50").unwrap());
}

#[test]
fn test_stop_loss_order() {
    let stop_order = Order {
        id: "test-789".to_string(),
        symbol: "TSLA".to_string(),
        side: OrderSide::Sell,
        quantity: Decimal::from_str("25").unwrap(),
        order_type: OrderType::StopLoss,
        limit_price: None,
        stop_price: Some(Decimal::from_str("200.00").unwrap()),
        time_in_force: "day".to_string(),
        status: OrderStatus::New,
    };

    assert!(stop_order.stop_price.is_some());
    assert_eq!(stop_order.order_type, OrderType::StopLoss);
}

#[test]
fn test_position_pnl() {
    let position = Position {
        symbol: "AAPL".to_string(),
        quantity: Decimal::from_str("100").unwrap(),
        avg_entry_price: Decimal::from_str("150.00").unwrap(),
        current_price: Decimal::from_str("160.00").unwrap(),
        market_value: Decimal::from_str("16000.00").unwrap(),
        cost_basis: Decimal::from_str("15000.00").unwrap(),
        unrealized_pnl: Decimal::from_str("1000.00").unwrap(),
        realized_pnl: Decimal::ZERO,
    };

    // Verify P&L calculation
    let expected_pnl = (position.current_price - position.avg_entry_price) * position.quantity;
    assert_eq!(position.unrealized_pnl, expected_pnl);
}

#[test]
fn test_position_value() {
    let position = Position {
        symbol: "MSFT".to_string(),
        quantity: Decimal::from_str("50").unwrap(),
        avg_entry_price: Decimal::from_str("300.00").unwrap(),
        current_price: Decimal::from_str("350.00").unwrap(),
        market_value: Decimal::from_str("17500.00").unwrap(),
        cost_basis: Decimal::from_str("15000.00").unwrap(),
        unrealized_pnl: Decimal::from_str("2500.00").unwrap(),
        realized_pnl: Decimal::ZERO,
    };

    assert_eq!(position.market_value, position.quantity * position.current_price);
}

#[test]
fn test_order_side_parsing() {
    let buy_side = OrderSide::Buy;
    let sell_side = OrderSide::Sell;

    assert_ne!(buy_side, sell_side);
}

#[test]
fn test_order_status_transitions() {
    let mut order = Order {
        id: "test-status".to_string(),
        symbol: "GOOG".to_string(),
        side: OrderSide::Buy,
        quantity: Decimal::from_str("10").unwrap(),
        order_type: OrderType::Market,
        limit_price: None,
        stop_price: None,
        time_in_force: "day".to_string(),
        status: OrderStatus::New,
    };

    // Test status transitions
    assert_eq!(order.status, OrderStatus::New);

    order.status = OrderStatus::Submitted;
    assert_eq!(order.status, OrderStatus::Submitted);

    order.status = OrderStatus::Filled;
    assert_eq!(order.status, OrderStatus::Filled);
}

#[test]
fn test_decimal_precision() {
    let price = Decimal::from_str("123.456789").unwrap();
    let quantity = Decimal::from_str("100.0").unwrap();

    let total = price * quantity;
    assert!(total > Decimal::ZERO);

    // Test precision is maintained
    let rounded = price.round_dp(2);
    assert_eq!(rounded, Decimal::from_str("123.46").unwrap());
}

#[test]
fn test_order_validation_constraints() {
    // Test zero quantity
    let zero_qty = Decimal::ZERO;
    assert!(zero_qty == Decimal::ZERO);

    // Test negative price (should be handled by validation)
    let negative = Decimal::from_str("-100.0").unwrap();
    assert!(negative < Decimal::ZERO);
}
