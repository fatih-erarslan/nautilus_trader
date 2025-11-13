//! Integration tests for Interactive Brokers Client Portal Gateway
//!
//! # Prerequisites
//!
//! These tests require a running IBKR Client Portal Gateway instance:
//! 1. Download Client Portal Gateway from Interactive Brokers
//! 2. Start the gateway (typically on https://localhost:5000)
//! 3. Authenticate via the web interface
//! 4. Set environment variable: export IBKR_GATEWAY_URL=https://localhost:5000
//!
//! # Running Tests
//!
//! ```bash
//! # Run all IB integration tests
//! cargo test --test integration_ib -- --nocapture
//!
//! # Run specific test
//! cargo test --test integration_ib test_authentication -- --nocapture
//! ```
//!
//! # Note
//!
//! These tests use real API calls and may be rate-limited.
//! Use a paper trading account to avoid affecting real positions.

use hyperphysics_market::providers::{InteractiveBrokersProvider, MarketDataProvider};
use hyperphysics_market::data::Timeframe;
use chrono::{Duration, Utc};
use std::env;

/// Get gateway URL from environment or use default
fn get_gateway_url() -> String {
    env::var("IBKR_GATEWAY_URL").unwrap_or_else(|_| "https://localhost:5000".to_string())
}

/// Check if integration tests should run
fn should_run_integration_tests() -> bool {
    env::var("RUN_IBKR_INTEGRATION_TESTS").is_ok()
}

#[tokio::test]
#[ignore = "Requires running IBKR Client Portal Gateway"]
async fn test_provider_creation() {
    let url = get_gateway_url();
    let provider = InteractiveBrokersProvider::new(url).await;

    assert!(provider.is_ok(), "Failed to create provider");

    let provider = provider.unwrap();
    assert_eq!(
        provider.provider_name(),
        "Interactive Brokers (Client Portal Gateway)"
    );
    assert!(provider.supports_realtime());
}

#[tokio::test]
#[ignore = "Requires running IBKR Client Portal Gateway with active session"]
async fn test_authentication() {
    if !should_run_integration_tests() {
        println!("Skipping integration test - set RUN_IBKR_INTEGRATION_TESTS=1 to enable");
        return;
    }

    let url = get_gateway_url();
    let provider = InteractiveBrokersProvider::new(url).await.unwrap();

    let auth_result = provider.authenticate().await;

    match auth_result {
        Ok(_) => println!("✓ Authentication successful"),
        Err(e) => {
            eprintln!("✗ Authentication failed: {}", e);
            eprintln!("Make sure Client Portal Gateway is running and authenticated");
            panic!("Authentication test failed");
        }
    }
}

#[tokio::test]
#[ignore = "Requires running IBKR Client Portal Gateway with active session"]
async fn test_health_check() {
    if !should_run_integration_tests() {
        return;
    }

    let url = get_gateway_url();
    let provider = InteractiveBrokersProvider::new(url).await.unwrap();

    let health = provider.health_check().await;

    match health {
        Ok(_) => println!("✓ Health check passed"),
        Err(e) => {
            eprintln!("✗ Health check failed: {}", e);
            panic!("Health check test failed");
        }
    }
}

#[tokio::test]
#[ignore = "Requires running IBKR Client Portal Gateway with active session"]
async fn test_contract_search() {
    if !should_run_integration_tests() {
        return;
    }

    let url = get_gateway_url();
    let provider = InteractiveBrokersProvider::new(url).await.unwrap();

    // Authenticate first
    provider.authenticate().await.expect("Authentication failed");

    let contracts = provider.search_contract("AAPL").await;

    match contracts {
        Ok(contracts) => {
            println!("✓ Found {} contracts for AAPL", contracts.len());
            assert!(!contracts.is_empty(), "Should find at least one contract");

            for contract in contracts.iter().take(3) {
                println!(
                    "  - Contract {}: {} ({})",
                    contract.contract_id, contract.symbol, contract.sec_type
                );
            }
        }
        Err(e) => {
            eprintln!("✗ Contract search failed: {}", e);
            panic!("Contract search test failed");
        }
    }
}

#[tokio::test]
#[ignore = "Requires running IBKR Client Portal Gateway with active session and market data permissions"]
async fn test_fetch_historical_bars() {
    if !should_run_integration_tests() {
        return;
    }

    let url = get_gateway_url();
    let provider = InteractiveBrokersProvider::new(url).await.unwrap();

    // Authenticate first
    provider.authenticate().await.expect("Authentication failed");

    let end = Utc::now();
    let start = end - Duration::days(7);

    let bars = provider
        .fetch_bars("AAPL", Timeframe::Day1, start, end)
        .await;

    match bars {
        Ok(bars) => {
            println!("✓ Fetched {} bars for AAPL", bars.len());
            assert!(!bars.is_empty(), "Should fetch at least one bar");

            // Display first few bars
            for bar in bars.iter().take(3) {
                println!(
                    "  - {}: O={:.2} H={:.2} L={:.2} C={:.2} V={}",
                    bar.timestamp.format("%Y-%m-%d"),
                    bar.open,
                    bar.high,
                    bar.low,
                    bar.close,
                    bar.volume
                );

                // Validate OHLC consistency
                assert!(bar.high >= bar.low, "High must be >= Low");
                assert!(
                    bar.open >= bar.low && bar.open <= bar.high,
                    "Open must be within [Low, High]"
                );
                assert!(
                    bar.close >= bar.low && bar.close <= bar.high,
                    "Close must be within [Low, High]"
                );
            }
        }
        Err(e) => {
            eprintln!("✗ Failed to fetch bars: {}", e);
            eprintln!("Note: Market data permissions required for this test");
            panic!("Historical bars test failed");
        }
    }
}

#[tokio::test]
#[ignore = "Requires running IBKR Client Portal Gateway with active session and market data permissions"]
async fn test_fetch_latest_bar() {
    if !should_run_integration_tests() {
        return;
    }

    let url = get_gateway_url();
    let provider = InteractiveBrokersProvider::new(url).await.unwrap();

    provider.authenticate().await.expect("Authentication failed");

    let bar = provider.fetch_latest_bar("AAPL").await;

    match bar {
        Ok(bar) => {
            println!("✓ Latest bar for AAPL:");
            println!(
                "  Time: {}, Close: {:.2}, Volume: {}",
                bar.timestamp.format("%Y-%m-%d %H:%M:%S"),
                bar.close,
                bar.volume
            );

            assert!(bar.high >= bar.low, "High must be >= Low");
        }
        Err(e) => {
            eprintln!("✗ Failed to fetch latest bar: {}", e);
            panic!("Latest bar test failed");
        }
    }
}

#[tokio::test]
#[ignore = "Requires running IBKR Client Portal Gateway with active session and market data permissions"]
async fn test_fetch_snapshot() {
    if !should_run_integration_tests() {
        return;
    }

    let url = get_gateway_url();
    let provider = InteractiveBrokersProvider::new(url).await.unwrap();

    provider.authenticate().await.expect("Authentication failed");

    let snapshot = provider.fetch_snapshot("AAPL").await;

    match snapshot {
        Ok(snapshot) => {
            println!("✓ Snapshot for AAPL:");
            println!("  Contract ID: {}", snapshot.contract_id);
            if let Some(ref price) = snapshot.last_price {
                println!("  Last Price: {}", price);
            }
            if let Some(ref bid) = snapshot.bid_price {
                println!("  Bid: {}", bid);
            }
            if let Some(ref ask) = snapshot.ask_price {
                println!("  Ask: {}", ask);
            }
        }
        Err(e) => {
            eprintln!("✗ Failed to fetch snapshot: {}", e);
            panic!("Snapshot test failed");
        }
    }
}

#[tokio::test]
#[ignore = "Requires running IBKR Client Portal Gateway with active session and market data permissions"]
async fn test_fetch_quote() {
    if !should_run_integration_tests() {
        return;
    }

    let url = get_gateway_url();
    let provider = InteractiveBrokersProvider::new(url).await.unwrap();

    provider.authenticate().await.expect("Authentication failed");

    let quote = provider.fetch_quote("AAPL").await;

    match quote {
        Ok(quote) => {
            println!("✓ Quote for AAPL:");
            println!("  Bid: {:.2} x {:.0}", quote.bid_price, quote.bid_size);
            println!("  Ask: {:.2} x {:.0}", quote.ask_price, quote.ask_size);
            println!("  Mid: {:.2}", quote.mid_price());
            println!("  Spread: ${:.4}", quote.spread());

            assert!(quote.ask_price >= quote.bid_price, "Ask must be >= Bid");
            assert!(quote.spread() >= 0.0, "Spread must be non-negative");
        }
        Err(e) => {
            eprintln!("✗ Failed to fetch quote: {}", e);
            panic!("Quote test failed");
        }
    }
}

#[tokio::test]
#[ignore = "Requires running IBKR Client Portal Gateway with active session and market data permissions"]
async fn test_fetch_tick() {
    if !should_run_integration_tests() {
        return;
    }

    let url = get_gateway_url();
    let provider = InteractiveBrokersProvider::new(url).await.unwrap();

    provider.authenticate().await.expect("Authentication failed");

    let tick = provider.fetch_tick("AAPL").await;

    match tick {
        Ok(tick) => {
            println!("✓ Tick for AAPL:");
            println!("  Price: {:.2}", tick.price);
            println!("  Size: {:.0}", tick.size);
            println!("  Time: {}", tick.timestamp.format("%Y-%m-%d %H:%M:%S"));

            assert!(tick.price > 0.0, "Price must be positive");
        }
        Err(e) => {
            eprintln!("✗ Failed to fetch tick: {}", e);
            panic!("Tick test failed");
        }
    }
}

#[tokio::test]
#[ignore = "Requires running IBKR Client Portal Gateway with active session"]
async fn test_supports_symbol() {
    if !should_run_integration_tests() {
        return;
    }

    let url = get_gateway_url();
    let provider = InteractiveBrokersProvider::new(url).await.unwrap();

    provider.authenticate().await.expect("Authentication failed");

    // Test valid symbol
    let supports_aapl = provider.supports_symbol("AAPL").await.unwrap();
    assert!(supports_aapl, "Should support AAPL");

    // Test invalid symbol
    let supports_invalid = provider.supports_symbol("INVALID_SYMBOL_XYZ").await.unwrap();
    assert!(!supports_invalid, "Should not support invalid symbol");

    println!("✓ Symbol validation working correctly");
}

#[tokio::test]
#[ignore = "Requires running IBKR Client Portal Gateway with active session"]
async fn test_session_validation() {
    if !should_run_integration_tests() {
        return;
    }

    let url = get_gateway_url();
    let provider = InteractiveBrokersProvider::new(url).await.unwrap();

    provider.authenticate().await.expect("Authentication failed");

    let is_valid = provider.validate_session().await;

    match is_valid {
        Ok(valid) => {
            println!("✓ Session validation: {}", if valid { "valid" } else { "invalid" });
            assert!(valid, "Session should be valid after authentication");
        }
        Err(e) => {
            eprintln!("✗ Session validation failed: {}", e);
            panic!("Session validation test failed");
        }
    }
}

#[tokio::test]
#[ignore = "Requires running IBKR Client Portal Gateway with active session and market data permissions"]
async fn test_retry_logic() {
    if !should_run_integration_tests() {
        return;
    }

    let url = get_gateway_url();
    let provider = InteractiveBrokersProvider::new(url).await.unwrap();

    provider.authenticate().await.expect("Authentication failed");

    // Test retry with valid operation
    let result = provider
        .with_retry(3, || async { provider.fetch_latest_bar("AAPL").await })
        .await;

    match result {
        Ok(bar) => {
            println!("✓ Retry logic test passed");
            println!("  Latest AAPL close: {:.2}", bar.close);
        }
        Err(e) => {
            eprintln!("✗ Retry logic test failed: {}", e);
            panic!("Retry logic test failed");
        }
    }
}

#[tokio::test]
#[ignore = "Requires running IBKR Client Portal Gateway with active session and market data permissions"]
async fn test_multiple_timeframes() {
    if !should_run_integration_tests() {
        return;
    }

    let url = get_gateway_url();
    let provider = InteractiveBrokersProvider::new(url).await.unwrap();

    provider.authenticate().await.expect("Authentication failed");

    let timeframes = vec![
        (Timeframe::Minute1, "1 minute"),
        (Timeframe::Minute5, "5 minutes"),
        (Timeframe::Hour1, "1 hour"),
        (Timeframe::Day1, "1 day"),
    ];

    for (timeframe, name) in timeframes {
        let end = Utc::now();
        let start = end - Duration::days(1);

        match provider.fetch_bars("AAPL", timeframe, start, end).await {
            Ok(bars) => {
                println!("✓ Fetched {} {} bars", bars.len(), name);
            }
            Err(e) => {
                eprintln!("✗ Failed to fetch {} bars: {}", name, e);
                // Don't panic - some timeframes may not be available for all symbols
            }
        }
    }
}
