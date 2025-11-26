// Alpaca Markets API Integration Tests
// Tests real Alpaca API with paper trading credentials (NO MOCKS)
//
// Setup:
// export APCA_API_KEY_ID="your_paper_key"
// export APCA_API_SECRET_KEY="your_paper_secret"
//
// Run: cargo test -p hyperphysics-market --test alpaca_integration -- --ignored

use hyperphysics_market::providers::alpaca::{AlpacaProvider, AlpacaWebSocketClient};
use hyperphysics_market::{Bar, Timeframe};
use hyperphysics_market::providers::MarketDataProvider;
use chrono::{Duration, Utc};

fn init_tracing() {
    // Simple tracing initialization without EnvFilter (which requires env-filter feature)
    let _ = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .try_init();
}

#[tokio::test]
#[ignore] // Requires APCA_API_KEY_ID and APCA_API_SECRET_KEY environment variables
async fn test_alpaca_provider_from_env() {
    init_tracing();

    let provider = AlpacaProvider::from_env()
        .expect("Failed to create Alpaca provider from environment");

    assert_eq!(provider.provider_name(), "Alpaca Markets");
    assert!(provider.supports_realtime());
}

#[tokio::test]
#[ignore]
async fn test_fetch_latest_bar_aapl() {
    init_tracing();

    let provider = AlpacaProvider::from_env()
        .expect("Failed to create provider");

    let bar = provider.fetch_latest_bar("AAPL").await
        .expect("Failed to fetch latest bar for AAPL");

    // Validate bar structure
    assert_eq!(bar.symbol, "AAPL");
    assert!(bar.open > 0.0);
    assert!(bar.high > 0.0);
    assert!(bar.low > 0.0);
    assert!(bar.close > 0.0);
    assert!(bar.volume > 0);

    // OHLC consistency
    assert!(bar.high >= bar.open);
    assert!(bar.high >= bar.close);
    assert!(bar.low <= bar.open);
    assert!(bar.low <= bar.close);

    tracing::info!("AAPL latest bar: {:?}", bar);
}

#[tokio::test]
#[ignore]
async fn test_fetch_bars_multiple_symbols() {
    init_tracing();

    let provider = AlpacaProvider::from_env()
        .expect("Failed to create provider");

    let symbols = vec!["AAPL", "MSFT", "GOOGL"];
    let end = Utc::now();
    let start = end - Duration::days(5);

    for symbol in symbols {
        let bars = provider.fetch_bars(symbol, Timeframe::Day1, start, end).await
            .expect(&format!("Failed to fetch bars for {}", symbol));

        assert!(!bars.is_empty(), "No bars returned for {}", symbol);
        assert!(bars.len() <= 5, "Too many bars returned for {}", symbol);

        // Validate each bar
        for bar in &bars {
            assert_eq!(bar.symbol, symbol);
            assert!(bar.open > 0.0);
            assert!(bar.high >= bar.low);
            assert!(bar.volume > 0);
        }

        tracing::info!("{} returned {} bars", symbol, bars.len());
    }
}

#[tokio::test]
#[ignore]
async fn test_fetch_intraday_bars() {
    init_tracing();

    let provider = AlpacaProvider::from_env()
        .expect("Failed to create provider");

    let end = Utc::now();
    let start = end - Duration::hours(2);

    let bars = provider.fetch_bars("SPY", Timeframe::Minute1, start, end).await
        .expect("Failed to fetch 1-minute bars");

    assert!(!bars.is_empty(), "No intraday bars returned");

    // Validate chronological ordering
    for i in 1..bars.len() {
        assert!(bars[i].timestamp > bars[i-1].timestamp, "Bars not in chronological order");
    }

    tracing::info!("SPY returned {} 1-minute bars", bars.len());
}

#[tokio::test]
#[ignore]
async fn test_supports_symbol_valid() {
    init_tracing();

    let provider = AlpacaProvider::from_env()
        .expect("Failed to create provider");

    // Test common tradable symbols
    let valid_symbols = vec!["AAPL", "MSFT", "GOOGL", "TSLA", "SPY"];

    for symbol in valid_symbols {
        let supported = provider.supports_symbol(symbol).await
            .expect(&format!("Failed to check symbol {}", symbol));

        assert!(supported, "{} should be supported", symbol);
    }
}

#[tokio::test]
#[ignore]
async fn test_supports_symbol_invalid() {
    init_tracing();

    let provider = AlpacaProvider::from_env()
        .expect("Failed to create provider");

    // Test invalid/non-existent symbols
    let invalid_symbols = vec!["INVALIDXYZ", "NOTREAL123"];

    for symbol in invalid_symbols {
        let supported = provider.supports_symbol(symbol).await
            .expect(&format!("Failed to check symbol {}", symbol));

        assert!(!supported, "{} should not be supported", symbol);
    }
}

#[tokio::test]
#[ignore]
async fn test_rate_limiting() {
    init_tracing();

    let provider = AlpacaProvider::from_env()
        .expect("Failed to create provider");

    // Alpaca paper trading has 200 requests/minute limit
    // Send 10 rapid requests to test rate limiter
    let start_time = std::time::Instant::now();

    for i in 0..10 {
        let result = provider.fetch_latest_bar("AAPL").await;
        assert!(result.is_ok(), "Request {} failed: {:?}", i, result.err());
    }

    let elapsed = start_time.elapsed();

    // With 200/min limit, 10 requests should take ~3 seconds with rate limiter
    // (200/60 = 3.33 req/sec, so 10 req = ~3 sec)
    assert!(elapsed.as_secs() >= 2, "Rate limiter may not be working (too fast)");

    tracing::info!("10 requests completed in {:?}", elapsed);
}

#[tokio::test]
#[ignore]
async fn test_websocket_connection() {
    init_tracing();

    let api_key = std::env::var("APCA_API_KEY_ID")
        .expect("APCA_API_KEY_ID not set");
    let secret_key = std::env::var("APCA_API_SECRET_KEY")
        .expect("APCA_API_SECRET_KEY not set");

    let mut client = AlpacaWebSocketClient::new(api_key, secret_key);

    // Connect to WebSocket
    client.connect().await
        .expect("Failed to connect to Alpaca WebSocket");

    // Authenticate
    client.authenticate().await
        .expect("Failed to authenticate");

    tracing::info!("WebSocket connection and authentication successful");
}

#[tokio::test]
#[ignore]
async fn test_websocket_subscribe_trades() {
    init_tracing();

    let api_key = std::env::var("APCA_API_KEY_ID")
        .expect("APCA_API_KEY_ID not set");
    let secret_key = std::env::var("APCA_API_SECRET_KEY")
        .expect("APCA_API_SECRET_KEY not set");

    let mut client = AlpacaWebSocketClient::new(api_key, secret_key);

    client.connect().await.expect("Failed to connect");
    client.authenticate().await.expect("Failed to authenticate");

    // Subscribe to trades for AAPL and MSFT
    client.subscribe_trades(vec!["AAPL", "MSFT"]).await
        .expect("Failed to subscribe to trades");

    // Wait for a few messages (max 10 seconds)
    let timeout = tokio::time::Duration::from_secs(10);
    let mut messages_received = 0;

    tokio::select! {
        _ = tokio::time::sleep(timeout) => {
            tracing::warn!("Timeout waiting for trade messages");
        }
        _ = async {
            while messages_received < 5 {
                if let Ok(Some(msg)) = client.next_message().await {
                    tracing::info!("Received message: {:?}", msg);
                    messages_received += 1;
                }
            }
        } => {
            tracing::info!("Received {} messages", messages_received);
        }
    }

    assert!(messages_received > 0, "Should receive at least one message");
}

#[tokio::test]
#[ignore]
async fn test_websocket_subscribe_quotes() {
    init_tracing();

    let api_key = std::env::var("APCA_API_KEY_ID")
        .expect("APCA_API_KEY_ID not set");
    let secret_key = std::env::var("APCA_API_SECRET_KEY")
        .expect("APCA_API_SECRET_KEY not set");

    let mut client = AlpacaWebSocketClient::new(api_key, secret_key);

    client.connect().await.expect("Failed to connect");
    client.authenticate().await.expect("Failed to authenticate");

    // Subscribe to quotes for SPY
    client.subscribe_quotes(vec!["SPY"]).await
        .expect("Failed to subscribe to quotes");

    // Wait for quote messages (max 10 seconds)
    let timeout = tokio::time::Duration::from_secs(10);
    let mut quotes_received = 0;

    tokio::select! {
        _ = tokio::time::sleep(timeout) => {
            tracing::warn!("Timeout waiting for quote messages");
        }
        _ = async {
            while quotes_received < 5 {
                if let Ok(Some(msg)) = client.next_message().await {
                    tracing::info!("Received quote: {:?}", msg);
                    quotes_received += 1;
                }
            }
        } => {
            tracing::info!("Received {} quotes", quotes_received);
        }
    }

    assert!(quotes_received > 0, "Should receive at least one quote");
}

#[tokio::test]
#[ignore]
async fn test_bar_data_scientific_validation() {
    init_tracing();

    let provider = AlpacaProvider::from_env()
        .expect("Failed to create provider");

    let end = Utc::now();
    let start = end - Duration::days(30);

    let bars = provider.fetch_bars("SPY", Timeframe::Day1, start, end).await
        .expect("Failed to fetch bars");

    assert!(!bars.is_empty(), "No bars returned");

    // Scientific validation checks
    for (i, bar) in bars.iter().enumerate() {
        // 1. OHLC relationship (mathematical invariant)
        assert!(bar.high >= bar.open, "Bar {}: high < open", i);
        assert!(bar.high >= bar.close, "Bar {}: high < close", i);
        assert!(bar.low <= bar.open, "Bar {}: low > open", i);
        assert!(bar.low <= bar.close, "Bar {}: low > close", i);
        assert!(bar.high >= bar.low, "Bar {}: high < low", i);

        // 2. Positive values (economic constraint)
        assert!(bar.open > 0.0, "Bar {}: non-positive open", i);
        assert!(bar.volume > 0, "Bar {}: zero volume", i);

        // 3. Reasonable price range (SPY typically $200-$600)
        assert!(bar.close >= 100.0 && bar.close <= 1000.0,
            "Bar {}: unrealistic price {}", i, bar.close);

        // 4. VWAP validation (if present)
        if let Some(vwap) = bar.vwap {
            assert!(vwap >= bar.low && vwap <= bar.high,
                "Bar {}: VWAP {} outside [low, high]", i, vwap);
        }

        // 5. Timestamp validation (within query range)
        assert!(bar.timestamp >= start && bar.timestamp <= end,
            "Bar {}: timestamp outside query range", i);
    }

    tracing::info!("Scientific validation passed for {} bars", bars.len());
}

#[tokio::test]
#[ignore]
async fn test_multiple_timeframes() {
    init_tracing();

    let provider = AlpacaProvider::from_env()
        .expect("Failed to create provider");

    let end = Utc::now();
    let start = end - Duration::days(7);

    let timeframes = vec![
        (Timeframe::Minute1, "1Min"),
        (Timeframe::Minute5, "5Min"),
        (Timeframe::Hour1, "1Hour"),
        (Timeframe::Day1, "1Day"),
    ];

    for (timeframe, name) in timeframes {
        let bars = provider.fetch_bars("SPY", timeframe, start, end).await
            .expect(&format!("Failed to fetch {} bars", name));

        assert!(!bars.is_empty(), "No {} bars returned", name);
        tracing::info!("{}: {} bars", name, bars.len());
    }
}
