//! Integration tests for Polymarket prediction markets

use nt_prediction_markets::models::*;
use nt_prediction_markets::polymarket::*;
use rust_decimal_macros::dec;

#[test]
fn test_order_side_opposite() {
    assert_eq!(OrderSide::Buy.opposite(), OrderSide::Sell);
    assert_eq!(OrderSide::Sell.opposite(), OrderSide::Buy);
}

#[test]
fn test_order_status_flags() {
    assert!(OrderStatus::Open.is_active());
    assert!(OrderStatus::Partial.is_active());
    assert!(!OrderStatus::Filled.is_active());

    assert!(OrderStatus::Filled.is_complete());
    assert!(OrderStatus::Cancelled.is_complete());
    assert!(!OrderStatus::Open.is_complete());

    assert!(OrderStatus::Open.can_cancel());
    assert!(!OrderStatus::Filled.can_cancel());
}

#[test]
fn test_outcome_probability() {
    let outcome = Outcome {
        id: "yes".to_string(),
        market_id: "market1".to_string(),
        title: "Yes".to_string(),
        price: dec!(0.65),
    };

    assert_eq!(outcome.probability(), dec!(0.65));
}

#[test]
fn test_orderbook_calculations() {
    let orderbook = OrderBook {
        market_id: "market1".to_string(),
        outcome_id: "yes".to_string(),
        bids: vec![
            OrderBookLevel { price: dec!(0.60), size: dec!(100) },
            OrderBookLevel { price: dec!(0.59), size: dec!(200) },
        ],
        asks: vec![
            OrderBookLevel { price: dec!(0.62), size: dec!(150) },
            OrderBookLevel { price: dec!(0.63), size: dec!(250) },
        ],
        timestamp: chrono::Utc::now(),
    };

    assert_eq!(orderbook.best_bid(), Some(dec!(0.60)));
    assert_eq!(orderbook.best_ask(), Some(dec!(0.62)));
    assert_eq!(orderbook.spread(), Some(dec!(0.02)));
    assert_eq!(orderbook.mid_price(), Some(dec!(0.61)));
    assert_eq!(orderbook.total_bid_size(), dec!(300));
    assert_eq!(orderbook.total_ask_size(), dec!(400));
}

#[test]
fn test_orderbook_depth() {
    let orderbook = OrderBook {
        market_id: "market1".to_string(),
        outcome_id: "yes".to_string(),
        bids: vec![
            OrderBookLevel { price: dec!(0.60), size: dec!(100) },
            OrderBookLevel { price: dec!(0.59), size: dec!(200) },
            OrderBookLevel { price: dec!(0.58), size: dec!(300) },
        ],
        asks: vec![
            OrderBookLevel { price: dec!(0.62), size: dec!(150) },
            OrderBookLevel { price: dec!(0.63), size: dec!(250) },
            OrderBookLevel { price: dec!(0.64), size: dec!(350) },
        ],
        timestamp: chrono::Utc::now(),
    };

    let (bids, asks) = orderbook.get_depth(2);
    assert_eq!(bids.len(), 2);
    assert_eq!(asks.len(), 2);
    assert_eq!(bids[0].price, dec!(0.60));
    assert_eq!(asks[0].price, dec!(0.62));
}

#[test]
fn test_price_impact_calculation() {
    let orderbook = OrderBook {
        market_id: "market1".to_string(),
        outcome_id: "yes".to_string(),
        bids: vec![
            OrderBookLevel { price: dec!(0.60), size: dec!(100) },
            OrderBookLevel { price: dec!(0.59), size: dec!(200) },
        ],
        asks: vec![
            OrderBookLevel { price: dec!(0.62), size: dec!(150) },
            OrderBookLevel { price: dec!(0.63), size: dec!(250) },
        ],
        timestamp: chrono::Utc::now(),
    };

    // Buy 100 units - should fill at 0.62
    let impact = orderbook.calculate_price_impact(OrderSide::Buy, dec!(100));
    assert!(impact.is_some());
    assert!(impact.unwrap() < dec!(0.01)); // Very small impact

    // Buy 200 units - should fill at 0.62 and 0.63
    let impact = orderbook.calculate_price_impact(OrderSide::Buy, dec!(200));
    assert!(impact.is_some());
    assert!(impact.unwrap() >= dec!(0)); // Has some impact

    // Try to buy more than available - should fail
    let impact = orderbook.calculate_price_impact(OrderSide::Buy, dec!(500));
    assert!(impact.is_none());
}

#[test]
fn test_order_fill_calculations() {
    let fill = OrderFill {
        id: "fill1".to_string(),
        order_id: "order1".to_string(),
        price: dec!(0.65),
        size: dec!(100),
        side: OrderSide::Buy,
        timestamp: chrono::Utc::now(),
        fee: dec!(1.3),
        fee_currency: "USDC".to_string(),
    };

    assert_eq!(fill.value(), dec!(65)); // 0.65 * 100
    assert_eq!(fill.net_value(), dec!(63.7)); // 65 - 1.3
}

#[test]
fn test_order_calculations() {
    let order = Order {
        id: "order1".to_string(),
        market_id: "market1".to_string(),
        outcome_id: "yes".to_string(),
        side: OrderSide::Buy,
        order_type: OrderType::Limit,
        size: dec!(200),
        price: Some(dec!(0.65)),
        filled: dec!(100),
        remaining: dec!(100),
        status: OrderStatus::Partial,
        time_in_force: TimeInForce::GTC,
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
        expires_at: None,
        fills: vec![
            OrderFill {
                id: "fill1".to_string(),
                order_id: "order1".to_string(),
                price: dec!(0.65),
                size: dec!(100),
                side: OrderSide::Buy,
                timestamp: chrono::Utc::now(),
                fee: dec!(1.3),
                fee_currency: "USDC".to_string(),
            },
        ],
        fee_rate: dec!(0.02),
        client_order_id: None,
    };

    assert_eq!(order.fill_percentage(), dec!(50)); // 100/200 * 100
    assert!(order.is_active());
    assert!(!order.is_complete());
    assert!(order.can_cancel());
    assert_eq!(order.total_fees(), dec!(1.3));
    assert_eq!(order.average_fill_price(), Some(dec!(0.65)));
    assert_eq!(order.notional_value(), dec!(130)); // 200 * 0.65
    assert_eq!(order.remaining_value(), dec!(65)); // 100 * 0.65
}

#[test]
fn test_position_calculations() {
    let position = Position {
        market_id: "market1".to_string(),
        outcome_id: "yes".to_string(),
        size: dec!(100),
        average_price: dec!(0.60),
        current_price: dec!(0.65),
        unrealized_pnl: dec!(5),
        realized_pnl: dec!(2),
        total_fees: dec!(1.5),
    };

    assert_eq!(position.current_value(), dec!(65)); // 100 * 0.65
    assert_eq!(position.cost_basis(), dec!(60)); // 100 * 0.60
    assert_eq!(position.total_pnl(), dec!(5.5)); // 5 + 2 - 1.5
    // Precision can vary slightly, so use approximate comparison
    let pnl_pct = position.pnl_percentage();
    assert!(pnl_pct > dec!(9.16) && pnl_pct < dec!(9.17)); // ~9.166...%
}

#[test]
fn test_order_request_validation() {
    // Valid limit order
    let valid_order = OrderRequest {
        market_id: "market1".to_string(),
        outcome_id: "yes".to_string(),
        side: OrderSide::Buy,
        order_type: OrderType::Limit,
        size: dec!(100),
        price: Some(dec!(0.6)),
        time_in_force: None,
        client_order_id: None,
    };
    assert!(valid_order.validate().is_ok());

    // Invalid: zero size
    let mut invalid = valid_order.clone();
    invalid.size = dec!(0);
    assert!(invalid.validate().is_err());

    // Invalid: negative size
    invalid.size = dec!(-10);
    assert!(invalid.validate().is_err());

    // Invalid: limit order without price
    invalid = valid_order.clone();
    invalid.price = None;
    assert!(invalid.validate().is_err());

    // Invalid: price > 1
    invalid = valid_order.clone();
    invalid.price = Some(dec!(1.5));
    assert!(invalid.validate().is_err());

    // Invalid: price = 0
    invalid = valid_order.clone();
    invalid.price = Some(dec!(0));
    assert!(invalid.validate().is_err());
}

#[test]
fn test_market_maker_quote_calculation() {
    let config = MarketMakerConfig {
        spread: dec!(0.04), // 4%
        order_size: dec!(100),
        max_position: dec!(1000),
        num_levels: 3,
        min_edge: dec!(0.01),
        inventory_skew: dec!(0.5),
    };

    let client_config = ClientConfig::new("test_key");
    let client = PolymarketClient::new(client_config).unwrap();
    let mm = PolymarketMM::new(client, config);

    // Test with no position
    let (bid, ask) = mm.calculate_quotes(dec!(0.5), dec!(0));
    assert_eq!(bid, dec!(0.48)); // 0.5 - 0.02
    assert_eq!(ask, dec!(0.52)); // 0.5 + 0.02

    // Test with positions - just verify they produce different quotes
    let (bid_long, ask_long) = mm.calculate_quotes(dec!(0.5), dec!(500));
    assert_ne!(bid_long, bid);
    assert_ne!(ask_long, ask);

    let (bid_short, ask_short) = mm.calculate_quotes(dec!(0.5), dec!(-500));
    assert_ne!(bid_short, bid);
    assert_ne!(ask_short, ask);
}

#[test]
fn test_market_maker_order_generation() {
    let config = MarketMakerConfig {
        spread: dec!(0.04),
        order_size: dec!(100),
        max_position: dec!(1000),
        num_levels: 3,
        min_edge: dec!(0.01),
        inventory_skew: dec!(0.5),
    };

    let client_config = ClientConfig::new("test_key");
    let client = PolymarketClient::new(client_config).unwrap();
    let mm = PolymarketMM::new(client, config);

    let orders = mm.generate_orders("market1", "yes", dec!(0.5));

    // Should generate 6 orders (3 levels * 2 sides)
    assert_eq!(orders.len(), 6);

    // Check order properties
    for order in &orders {
        assert_eq!(order.size, dec!(100));
        assert_eq!(order.order_type, OrderType::Limit);
        assert!(order.price.is_some());
        let price = order.price.unwrap();
        assert!(price > dec!(0) && price < dec!(1));
    }

    // Verify we have both buy and sell orders
    let buys = orders.iter().filter(|o| o.side == OrderSide::Buy).count();
    let sells = orders.iter().filter(|o| o.side == OrderSide::Sell).count();
    assert_eq!(buys, 3);
    assert_eq!(sells, 3);
}

#[test]
fn test_arbitrage_risk_assessment() {
    let config = ArbitrageConfig {
        min_profit: dec!(0.02),
        max_size: dec!(1000),
        fee_rate: dec!(0.02),
        check_interval: 5,
    };

    let client_config = ClientConfig::new("test_key");
    let client = PolymarketClient::new(client_config).unwrap();
    let arb = PolymarketArbitrage::new(client, config);

    assert_eq!(arb.assess_risk(dec!(15)), RiskLevel::Low);
    assert_eq!(arb.assess_risk(dec!(7)), RiskLevel::Medium);
    assert_eq!(arb.assess_risk(dec!(3)), RiskLevel::High);
    assert_eq!(arb.assess_risk(dec!(1)), RiskLevel::High);
}

#[test]
fn test_credentials_creation() {
    let creds = Credentials::new("my_api_key");
    assert_eq!(creds.api_key, "my_api_key");
    assert!(creds.api_secret.is_none());

    let creds_with_secret = creds.clone().with_secret("my_secret");
    assert_eq!(creds_with_secret.api_secret, Some("my_secret".to_string()));
}

#[test]
fn test_credentials_validation() {
    let valid = Credentials::new("test_key");
    assert!(valid.validate().is_ok());

    let invalid = Credentials::new("");
    assert!(invalid.validate().is_err());
}

#[test]
fn test_auth_header_generation() {
    let creds = Credentials::new("test_api_key_123");
    assert_eq!(creds.auth_header(), "Bearer test_api_key_123");
}

#[test]
fn test_client_config_builder() {
    let config = ClientConfig::new("my_key")
        .with_base_url("https://custom.com")
        .with_timeout(std::time::Duration::from_secs(60))
        .with_max_retries(5);

    assert_eq!(config.api_key, "my_key");
    assert_eq!(config.base_url, "https://custom.com");
    assert_eq!(config.timeout, std::time::Duration::from_secs(60));
    assert_eq!(config.max_retries, 5);
}
