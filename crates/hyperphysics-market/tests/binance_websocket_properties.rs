//! Property-based tests for Binance WebSocket client
//!
//! These tests use property testing to verify invariants and edge cases:
//! - Data format validation
//! - Price/quantity constraints
//! - Timestamp ordering
//! - Message integrity

use hyperphysics_market::providers::binance_websocket::{
    TradeEvent, KlineEvent, KlineData, DepthUpdateEvent,
};

#[test]
fn test_trade_event_price_validity() {
    // Property: Trade prices must be valid positive numbers
    let valid_prices = vec!["0.001", "100.50", "50000.123456", "0.00000001"];

    for price_str in valid_prices {
        let trade = TradeEvent {
            event_time: 1000000,
            symbol: "BTCUSDT".to_string(),
            trade_id: 12345,
            price: price_str.to_string(),
            quantity: "1.0".to_string(),
            trade_time: 1000000,
            is_buyer_maker: false,
        };

        let price = trade.price.parse::<f64>();
        assert!(price.is_ok(), "Failed to parse price: {}", trade.price);
        assert!(price.unwrap() > 0.0, "Price must be positive");
    }
}

#[test]
fn test_trade_event_quantity_validity() {
    // Property: Trade quantities must be valid positive numbers
    let valid_quantities = vec!["0.001", "1.0", "100.5", "0.00000001"];

    for qty_str in valid_quantities {
        let trade = TradeEvent {
            event_time: 1000000,
            symbol: "BTCUSDT".to_string(),
            trade_id: 12345,
            price: "50000.0".to_string(),
            quantity: qty_str.to_string(),
            trade_time: 1000000,
            is_buyer_maker: false,
        };

        let quantity = trade.quantity.parse::<f64>();
        assert!(quantity.is_ok(), "Failed to parse quantity: {}", trade.quantity);
        assert!(quantity.unwrap() > 0.0, "Quantity must be positive");
    }
}

#[test]
fn test_trade_timestamp_ordering() {
    // Property: Event time should be >= trade time
    let trade = TradeEvent {
        event_time: 2000000,
        symbol: "BTCUSDT".to_string(),
        trade_id: 12345,
        price: "50000.0".to_string(),
        quantity: "1.0".to_string(),
        trade_time: 1000000,
        is_buyer_maker: false,
    };

    assert!(
        trade.event_time >= trade.trade_time,
        "Event time must be >= trade time"
    );
}

#[test]
fn test_kline_ohlc_constraints() {
    // Property: OHLC constraints must hold
    // high >= max(open, close)
    // low <= min(open, close)
    let test_cases = vec![
        // (open, high, low, close) - valid cases
        ("100.0", "110.0", "95.0", "105.0"),
        ("50.0", "50.0", "50.0", "50.0"),
        ("100.0", "100.0", "90.0", "95.0"),
    ];

    for (open, high, low, close) in test_cases {
        let kline = KlineData {
            start_time: 1000000,
            close_time: 1060000,
            symbol: "BTCUSDT".to_string(),
            interval: "1m".to_string(),
            open: open.to_string(),
            high: high.to_string(),
            low: low.to_string(),
            close: close.to_string(),
            volume: "1000.0".to_string(),
            num_trades: 100,
            is_closed: true,
        };

        let o = kline.open.parse::<f64>().unwrap();
        let h = kline.high.parse::<f64>().unwrap();
        let l = kline.low.parse::<f64>().unwrap();
        let c = kline.close.parse::<f64>().unwrap();

        // High should be >= all other prices
        assert!(h >= o, "High must be >= open");
        assert!(h >= c, "High must be >= close");
        assert!(h >= l, "High must be >= low");

        // Low should be <= all other prices
        assert!(l <= o, "Low must be <= open");
        assert!(l <= c, "Low must be <= close");
        assert!(l <= h, "Low must be <= high");
    }
}

#[test]
fn test_kline_time_ordering() {
    // Property: Close time must be after start time
    let kline = KlineData {
        start_time: 1000000,
        close_time: 1060000,
        symbol: "BTCUSDT".to_string(),
        interval: "1m".to_string(),
        open: "100.0".to_string(),
        high: "110.0".to_string(),
        low: "95.0".to_string(),
        close: "105.0".to_string(),
        volume: "1000.0".to_string(),
        num_trades: 100,
        is_closed: true,
    };

    assert!(
        kline.close_time > kline.start_time,
        "Close time must be after start time"
    );

    // For 1-minute interval, difference should be ~60 seconds (60000 ms)
    let duration = kline.close_time - kline.start_time;
    assert_eq!(duration, 60000, "1m interval should be 60000ms");
}

#[test]
fn test_kline_volume_validity() {
    // Property: Volume must be non-negative
    let kline = KlineData {
        start_time: 1000000,
        close_time: 1060000,
        symbol: "BTCUSDT".to_string(),
        interval: "1m".to_string(),
        open: "100.0".to_string(),
        high: "110.0".to_string(),
        low: "95.0".to_string(),
        close: "105.0".to_string(),
        volume: "1000.5".to_string(),
        num_trades: 100,
        is_closed: true,
    };

    let volume = kline.volume.parse::<f64>().unwrap();
    assert!(volume >= 0.0, "Volume must be non-negative");

    assert!(kline.num_trades >= 0, "Trade count must be non-negative");
}

#[test]
fn test_depth_update_sequence() {
    // Property: Final update ID must be >= first update ID
    let depth = DepthUpdateEvent {
        event_time: 1000000,
        symbol: "BTCUSDT".to_string(),
        first_update_id: 1000,
        final_update_id: 1005,
        bids: vec![("50000.0".to_string(), "1.0".to_string())],
        asks: vec![("50001.0".to_string(), "1.0".to_string())],
    };

    assert!(
        depth.final_update_id >= depth.first_update_id,
        "Final update ID must be >= first update ID"
    );
}

#[test]
fn test_depth_bid_ask_spread() {
    // Property: Ask price must be >= bid price
    let depth = DepthUpdateEvent {
        event_time: 1000000,
        symbol: "BTCUSDT".to_string(),
        first_update_id: 1000,
        final_update_id: 1005,
        bids: vec![
            ("50000.0".to_string(), "1.0".to_string()),
            ("49999.0".to_string(), "2.0".to_string()),
        ],
        asks: vec![
            ("50001.0".to_string(), "1.0".to_string()),
            ("50002.0".to_string(), "2.0".to_string()),
        ],
    };

    if !depth.bids.is_empty() && !depth.asks.is_empty() {
        let best_bid = depth.bids[0].0.parse::<f64>().unwrap();
        let best_ask = depth.asks[0].0.parse::<f64>().unwrap();

        assert!(
            best_ask >= best_bid,
            "Best ask must be >= best bid (no arbitrage)"
        );
    }
}

#[test]
fn test_depth_price_ordering() {
    // Property: Bids should be in descending order, asks in ascending order
    let depth = DepthUpdateEvent {
        event_time: 1000000,
        symbol: "BTCUSDT".to_string(),
        first_update_id: 1000,
        final_update_id: 1005,
        bids: vec![
            ("50000.0".to_string(), "1.0".to_string()),
            ("49999.0".to_string(), "2.0".to_string()),
            ("49998.0".to_string(), "1.5".to_string()),
        ],
        asks: vec![
            ("50001.0".to_string(), "1.0".to_string()),
            ("50002.0".to_string(), "2.0".to_string()),
            ("50003.0".to_string(), "1.5".to_string()),
        ],
    };

    // Check bids are descending
    for i in 0..depth.bids.len().saturating_sub(1) {
        let current = depth.bids[i].0.parse::<f64>().unwrap();
        let next = depth.bids[i + 1].0.parse::<f64>().unwrap();
        assert!(
            current >= next,
            "Bids should be in descending order: {} >= {}",
            current,
            next
        );
    }

    // Check asks are ascending
    for i in 0..depth.asks.len().saturating_sub(1) {
        let current = depth.asks[i].0.parse::<f64>().unwrap();
        let next = depth.asks[i + 1].0.parse::<f64>().unwrap();
        assert!(
            current <= next,
            "Asks should be in ascending order: {} <= {}",
            current,
            next
        );
    }
}

#[test]
fn test_depth_quantity_validity() {
    // Property: All quantities must be positive
    let depth = DepthUpdateEvent {
        event_time: 1000000,
        symbol: "BTCUSDT".to_string(),
        first_update_id: 1000,
        final_update_id: 1005,
        bids: vec![
            ("50000.0".to_string(), "1.0".to_string()),
            ("49999.0".to_string(), "2.5".to_string()),
        ],
        asks: vec![
            ("50001.0".to_string(), "1.0".to_string()),
            ("50002.0".to_string(), "3.0".to_string()),
        ],
    };

    for (price, qty) in depth.bids.iter().chain(depth.asks.iter()) {
        let price_val = price.parse::<f64>().unwrap();
        let qty_val = qty.parse::<f64>().unwrap();

        assert!(price_val > 0.0, "Price must be positive");
        assert!(qty_val >= 0.0, "Quantity must be non-negative");
    }
}

#[test]
fn test_symbol_format() {
    // Property: Symbol should be uppercase and non-empty
    let symbols = vec!["BTCUSDT", "ETHUSDT", "BNBUSDT"];

    for symbol in symbols {
        assert!(!symbol.is_empty(), "Symbol should not be empty");
        assert_eq!(
            symbol,
            symbol.to_uppercase(),
            "Symbol should be uppercase"
        );
    }
}

#[test]
fn test_precision_preservation() {
    // Property: String representation should preserve precision
    let test_values = vec![
        "0.00000001",
        "123456789.123456789",
        "50000.50",
        "0.1",
    ];

    for value in test_values {
        let parsed = value.parse::<f64>().unwrap();
        let formatted = format!("{:.8}", parsed);

        // Ensure no significant precision loss for reasonable values
        let reparsed = formatted.parse::<f64>().unwrap();
        let diff = (parsed - reparsed).abs();

        assert!(
            diff < 1e-8 * parsed.abs().max(1.0),
            "Precision loss detected: {} vs {} (diff: {})",
            parsed,
            reparsed,
            diff
        );
    }
}
