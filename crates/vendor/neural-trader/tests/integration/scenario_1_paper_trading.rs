// Integration Test Scenario 1: Live Paper Trading
// Tests end-to-end paper trading workflow with Alpaca broker

use neural_trader_core::types::*;
use neural_trader_execution::broker::{Broker, AlpacaBroker};
use neural_trader_strategies::momentum::MomentumStrategy;
use rust_decimal_macros::dec;
use chrono::Utc;

#[tokio::test]
#[ignore] // Requires Alpaca credentials
async fn test_paper_trading_workflow() -> anyhow::Result<()> {
    // Initialize broker with paper trading
    let api_key = std::env::var("ALPACA_API_KEY")
        .expect("ALPACA_API_KEY environment variable not set");
    let api_secret = std::env::var("ALPACA_API_SECRET")
        .expect("ALPACA_API_SECRET environment variable not set");

    let broker = AlpacaBroker::new_paper(api_key, api_secret);

    // Verify connection
    let account = broker.get_account().await?;
    assert!(account.cash > dec!(0), "Account should have positive cash");

    // Create momentum strategy
    let mut strategy = MomentumStrategy::new(
        "AAPL".to_string(),
        TimeFrame::OneDay,
        20, // lookback period
    );

    // Generate signal
    let signal = strategy.generate_signal().await?;

    // Execute trade if signal is valid
    if signal.strength.abs() > dec!(0.5) {
        let order = Order {
            id: uuid::Uuid::new_v4().to_string(),
            symbol: "AAPL".to_string(),
            side: if signal.strength > dec!(0) {
                OrderSide::Buy
            } else {
                OrderSide::Sell
            },
            quantity: dec!(10),
            order_type: OrderType::Market,
            time_in_force: TimeInForce::Day,
            status: OrderStatus::Pending,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            filled_qty: None,
            filled_price: None,
            limit_price: None,
            stop_price: None,
        };

        let result = broker.place_order(order).await?;
        assert!(
            matches!(result.status, OrderStatus::Filled | OrderStatus::PartiallyFilled),
            "Order should be executed in paper trading"
        );

        println!("✅ Paper trade executed: {:?}", result);
    }

    Ok(())
}

#[tokio::test]
async fn test_strategy_signal_generation() -> anyhow::Result<()> {
    // Test strategy without broker (unit test level)
    let strategy = MomentumStrategy::new(
        "AAPL".to_string(),
        TimeFrame::OneDay,
        20,
    );

    // This should work without credentials
    assert_eq!(strategy.symbol(), "AAPL");
    assert_eq!(strategy.timeframe(), TimeFrame::OneDay);

    Ok(())
}

#[tokio::test]
async fn test_paper_trading_risk_limits() -> anyhow::Result<()> {
    // Test that risk limits are enforced
    use neural_trader_risk::manager::RiskManager;

    let risk_manager = RiskManager::new(
        dec!(100000), // initial capital
        dec!(0.02),   // max risk per trade (2%)
        dec!(0.06),   // max portfolio risk (6%)
    );

    let order_size = dec!(10000); // $10k order
    let stop_loss = dec!(9500);   // $500 risk

    let is_allowed = risk_manager.validate_order_risk(order_size, stop_loss);
    assert!(is_allowed, "Order should be within risk limits");

    // Test exceeding limits
    let large_order = dec!(50000); // $50k order
    let wide_stop = dec!(45000);   // $5k risk (5% - exceeds 2% limit)

    let is_denied = !risk_manager.validate_order_risk(large_order, wide_stop);
    assert!(is_denied, "Order should exceed risk limits");

    Ok(())
}
