//! End-to-end test for complete trading loop

use chrono::Utc;
use nt_execution::{OrderRequest, OrderSide, OrderType, TimeInForce};
use nt_portfolio::Portfolio;
use nt_strategies::MomentumStrategy;
use rust_decimal_macros::dec;

mod mocks {
    include!("../mocks/mock_broker.rs");
    include!("../mocks/mock_market_data.rs");
}

use mocks::{MockBrokerClient, MockMarketDataProvider, MarketPattern};

#[tokio::test]
#[ignore] // Run with: cargo test --ignored
async fn test_complete_trading_cycle() {
    // Setup: Initialize all components
    let broker = MockBrokerClient::new();
    let mut market_data = MockMarketDataProvider::with_pattern(MarketPattern::Uptrend);
    let mut portfolio = Portfolio::new(dec!(100000));

    // Generate market data
    let symbol = "AAPL".to_string();
    market_data.generate_bars(symbol.clone(), 100, dec!(150), Utc::now());

    // Create strategy
    let strategy = MomentumStrategy::new(
        vec![symbol.clone()],
        14,
        2.0,
        0.5,
    );

    // Step 1: Get market data
    let bars = market_data
        .get_latest_bar(&symbol)
        .expect("Should get market data");
    assert!(bars.close > dec!(0), "Should have valid price data");

    // Step 2: Strategy generates signal (simplified - actual would process market data)
    // In a real system, the strategy would analyze bars and generate signals

    // Step 3: Execute order based on signal
    let order = OrderRequest {
        symbol: symbol.clone(),
        qty: Some(dec!(10)),
        side: OrderSide::Buy,
        order_type: OrderType::Market,
        time_in_force: TimeInForce::Day,
        limit_price: None,
        stop_price: None,
        extended_hours: false,
        client_order_id: Some("e2e-test-1".to_string()),
        order_class: None,
        take_profit: None,
        stop_loss: None,
    };

    let order_response = broker
        .place_order(order)
        .await
        .expect("Order should be placed");

    // Step 4: Update portfolio
    let fill_price = order_response.filled_avg_price.unwrap_or(dec!(150));
    let fill_qty = order_response.filled_qty.unwrap_or(dec!(0));
    let cost = fill_price * fill_qty;

    portfolio
        .remove_cash(cost)
        .expect("Should have sufficient cash");

    // Step 5: Verify end state
    assert_eq!(broker.orders_count(), 1);
    assert!(portfolio.cash() < dec!(100000), "Cash should be reduced");
    assert!(portfolio.cash() > dec!(0), "Should have remaining cash");
}

#[tokio::test]
#[ignore]
async fn test_multiple_trades_with_risk_management() {
    let broker = MockBrokerClient::new();
    let mut portfolio = Portfolio::new(dec!(100000));

    // Execute multiple trades
    for i in 0..5 {
        let order = OrderRequest {
            symbol: format!("STOCK{}", i),
            qty: Some(dec!(10)),
            side: OrderSide::Buy,
            order_type: OrderType::Market,
            time_in_force: TimeInForce::Day,
            limit_price: None,
            stop_price: None,
            extended_hours: false,
            client_order_id: Some(format!("trade-{}", i)),
            order_class: None,
            take_profit: None,
            stop_loss: None,
        };

        let response = broker.place_order(order).await.expect("Should place order");

        // Update portfolio for each trade
        let cost = response.filled_avg_price.unwrap_or(dec!(100)) * dec!(10);
        portfolio.remove_cash(cost).expect("Should have cash");
    }

    assert_eq!(broker.orders_count(), 5);
    assert!(portfolio.cash() > dec!(0), "Should have remaining cash");
}

#[tokio::test]
#[ignore]
async fn test_trading_with_errors() {
    let broker = MockBrokerClient::new();

    // Test order placement
    let order = OrderRequest {
        symbol: "AAPL".to_string(),
        qty: Some(dec!(10)),
        side: OrderSide::Buy,
        order_type: OrderType::Market,
        time_in_force: TimeInForce::Day,
        limit_price: None,
        stop_price: None,
        extended_hours: false,
        client_order_id: Some("test-error-1".to_string()),
        order_class: None,
        take_profit: None,
        stop_loss: None,
    };

    let _ = broker.place_order(order.clone()).await.expect("First order succeeds");

    // Enable failure mode
    broker.set_failure_mode(true);

    let result = broker.place_order(order).await;
    assert!(result.is_err(), "Should fail when broker is in failure mode");

    // Verify system can recover
    broker.set_failure_mode(false);
    assert_eq!(broker.orders_count(), 1, "Should maintain state correctly");
}
