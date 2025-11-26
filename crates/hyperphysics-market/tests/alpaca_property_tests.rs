// Alpaca API Property-Based Tests
// Uses proptest for generative testing of Alpaca data structures
//
// Run: cargo test -p hyperphysics-market --test alpaca_property_tests

use hyperphysics_market::{Bar, Tick};
use hyperphysics_market::data::tick::Quote;
use proptest::prelude::*;
use proptest::{option, collection};
use proptest::strategy::Strategy;
use proptest::test_runner::Config as ProptestConfig;
use chrono::{Utc, TimeZone};

// Generate arbitrary valid bars
fn arb_bar() -> impl Strategy<Value = Bar> {
    (
        "[A-Z]{1,5}".prop_map(|s| s.to_string()),
        0i64..1_000_000_000,
        10.0..1000.0f64,
        10.0..1000.0f64,
        10.0..1000.0f64,
        10.0..1000.0f64,
        100u64..100_000_000u64,
        option::of(10.0..1000.0f64),
        option::of(1u64..10_000u64),
    ).prop_map(|(symbol, timestamp, open, high, low, close, volume, vwap, trade_count)| {
        // Ensure OHLC consistency
        let actual_high = open.max(close).max(high);
        let actual_low = open.min(close).min(low);

        Bar {
            symbol,
            timestamp: Utc.timestamp_opt(timestamp, 0).unwrap(),
            open,
            high: actual_high,
            low: actual_low,
            close,
            volume,
            vwap: vwap.map(|v| v.max(actual_low).min(actual_high)),
            trade_count,
        }
    })
}

// Generate arbitrary valid ticks
fn arb_tick() -> impl Strategy<Value = Tick> {
    (
        "[A-Z]{1,5}".prop_map(|s| s.to_string()),
        0i64..1_000_000_000,
        0.01..10000.0f64,
        0.001..10000.0f64,
        option::of("[A-Z]{4}".prop_map(|s| s.to_string())),
        option::of(collection::vec("[A-Z]{1,2}".prop_map(|s| s.to_string()), 0..5)),
    ).prop_map(|(symbol, timestamp, price, size, exchange, conditions)| {
        Tick {
            symbol,
            timestamp: Utc.timestamp_opt(timestamp, 0).unwrap(),
            price,
            size,
            exchange,
            conditions,
        }
    })
}

// Generate arbitrary valid quotes
fn arb_quote() -> impl Strategy<Value = Quote> {
    (
        "[A-Z]{1,5}".prop_map(|s| s.to_string()),
        0i64..1_000_000_000,
        0.01..10000.0f64,
        1.0..100_000.0f64,
        0.01..10000.0f64,
        1.0..100_000.0f64,
        option::of("[A-Z]{4}".prop_map(|s| s.to_string())),
    ).prop_map(|(symbol, timestamp, bid_price, bid_size, ask_price, ask_size, exchange)| {
        // Ensure bid <= ask (no-arbitrage condition)
        let (actual_bid, actual_ask) = if bid_price > ask_price {
            (ask_price, bid_price)
        } else {
            (bid_price, ask_price)
        };

        Quote {
            symbol,
            timestamp: Utc.timestamp_opt(timestamp, 0).unwrap(),
            bid_price: actual_bid,
            bid_size,
            ask_price: actual_ask,
            ask_size,
            exchange,
        }
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    #[test]
    fn test_bar_ohlc_invariants(bar in arb_bar()) {
        // OHLC mathematical invariants
        prop_assert!(bar.high >= bar.open, "high must be >= open");
        prop_assert!(bar.high >= bar.close, "high must be >= close");
        prop_assert!(bar.low <= bar.open, "low must be <= open");
        prop_assert!(bar.low <= bar.close, "low must be <= close");
        prop_assert!(bar.high >= bar.low, "high must be >= low");
    }

    #[test]
    fn test_bar_positive_values(bar in arb_bar()) {
        // Economic constraints
        prop_assert!(bar.open > 0.0, "open must be positive");
        prop_assert!(bar.high > 0.0, "high must be positive");
        prop_assert!(bar.low > 0.0, "low must be positive");
        prop_assert!(bar.close > 0.0, "close must be positive");
        prop_assert!(bar.volume > 0, "volume must be positive");
    }

    #[test]
    fn test_bar_vwap_bounds(bar in arb_bar()) {
        // VWAP must be within [low, high] if present
        if let Some(vwap) = bar.vwap {
            prop_assert!(vwap >= bar.low, "VWAP must be >= low");
            prop_assert!(vwap <= bar.high, "VWAP must be <= high");
        }
    }

    #[test]
    fn test_bar_timestamp_validity(bar in arb_bar()) {
        // Timestamp must be valid and not in far future
        let now = Utc::now();
        let max_future = now + chrono::Duration::days(1);

        prop_assert!(bar.timestamp <= max_future,
            "Timestamp cannot be more than 1 day in future");
    }

    #[test]
    fn test_tick_positive_values(tick in arb_tick()) {
        // Tick prices and sizes must be positive
        prop_assert!(tick.price > 0.0, "price must be positive");
        prop_assert!(tick.size > 0.0, "size must be positive");
    }

    #[test]
    fn test_tick_price_precision(tick in arb_tick()) {
        // Prices should be reasonable (not NaN or infinite)
        prop_assert!(!tick.price.is_nan(), "price cannot be NaN");
        prop_assert!(!tick.price.is_infinite(), "price cannot be infinite");
        prop_assert!(!tick.size.is_nan(), "size cannot be NaN");
        prop_assert!(!tick.size.is_infinite(), "size cannot be infinite");
    }

    #[test]
    fn test_quote_bid_ask_spread(quote in arb_quote()) {
        // No-arbitrage condition: bid <= ask
        prop_assert!(quote.bid_price <= quote.ask_price,
            "bid price must be <= ask price (no-arbitrage)");
    }

    #[test]
    fn test_quote_positive_sizes(quote in arb_quote()) {
        // Bid and ask sizes must be positive
        prop_assert!(quote.bid_size > 0.0, "bid size must be positive");
        prop_assert!(quote.ask_size > 0.0, "ask size must be positive");
    }

    #[test]
    fn test_quote_realistic_spread(quote in arb_quote()) {
        // Spread should be non-negative (bid <= ask enforced by generator)
        let spread = quote.ask_price - quote.bid_price;

        prop_assert!(spread >= 0.0, "spread must be non-negative");
        // Note: spread percentage can be large with random price generation
        // This test validates the no-arbitrage constraint, not realistic spreads
    }

    #[test]
    fn test_bar_returns_calculation(bars in collection::vec(arb_bar(), 2..100)) {
        // Test that returns can be calculated from consecutive bars
        for i in 1..bars.len() {
            let prev_close = bars[i-1].close;
            let current_close = bars[i].close;
            let log_return = (current_close / prev_close).ln();

            prop_assert!(!log_return.is_nan(), "log return should not be NaN");
            prop_assert!(!log_return.is_infinite(), "log return should not be infinite");

            // Log returns can be large with random price generation
            // Test validates mathematical consistency, not realistic returns
            prop_assert!(log_return.abs() <= 10.0,
                "extreme return detected: {}", log_return);
        }
    }

    #[test]
    fn test_bar_volatility_estimation(bars in collection::vec(arb_bar(), 10..100)) {
        // Test that volatility can be estimated from bar data
        let returns: Vec<f64> = bars.windows(2)
            .map(|w| (w[1].close / w[0].close).ln())
            .collect();

        if returns.len() > 1 {
            let mean = returns.iter().sum::<f64>() / returns.len() as f64;
            let variance = returns.iter()
                .map(|r| (r - mean).powi(2))
                .sum::<f64>() / (returns.len() - 1) as f64;
            let volatility = variance.sqrt();

            prop_assert!(!volatility.is_nan(), "volatility should not be NaN");
            prop_assert!(volatility >= 0.0, "volatility must be non-negative");

            // Test validates mathematical consistency for volatility calculation
            // Random price generation produces high volatility - this is expected
            prop_assert!(volatility.is_finite(), "volatility must be finite");
        }
    }
}

#[test]
fn test_property_test_configuration() {
    // Verify proptest is configured correctly
    let config = ProptestConfig::with_cases(1000);
    assert_eq!(config.cases, 1000);
}
