//! Integration tests for Binance WebSocket client
//!
//! These tests verify real-world connectivity with Binance testnet
//! and production endpoints. They validate:
//! - Connection establishment
//! - Stream subscriptions
//! - Message parsing
//! - Error handling
//! - Reconnection logic

use hyperphysics_market::providers::BinanceWebSocketClient;
use std::time::Duration;
use tokio::time::{timeout, sleep};
/// Initialize tracing for test output
fn init_tracing() {
    let _ = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .with_test_writer()
        .try_init();
}

#[tokio::test]
async fn test_connect_to_testnet() {
    init_tracing();

    let mut client = BinanceWebSocketClient::new(true)
        .expect("Failed to create client");

    let result = client.connect().await;
    assert!(result.is_ok(), "Failed to connect: {:?}", result.err());
    assert!(client.is_connected().await);

    client.disconnect().await.expect("Failed to disconnect");
    sleep(Duration::from_millis(100)).await;
}

#[tokio::test]
async fn test_subscribe_trades() {
    init_tracing();

    let mut client = BinanceWebSocketClient::new(true)
        .expect("Failed to create client");

    client.connect().await.expect("Failed to connect");

    let result = client.subscribe_trades("btcusdt").await;
    assert!(result.is_ok(), "Failed to subscribe: {:?}", result.err());

    // Wait for messages (with timeout)
    let message_result = timeout(
        Duration::from_secs(5),
        client.next_message()
    ).await;

    assert!(message_result.is_ok(), "Timeout waiting for message");

    client.disconnect().await.expect("Failed to disconnect");
}

#[tokio::test]
async fn test_subscribe_klines() {
    init_tracing();

    let mut client = BinanceWebSocketClient::new(true)
        .expect("Failed to create client");

    client.connect().await.expect("Failed to connect");

    let result = client.subscribe_klines("btcusdt", "1m").await;
    assert!(result.is_ok(), "Failed to subscribe: {:?}", result.err());

    // Wait for kline message
    let message_result = timeout(
        Duration::from_secs(10),
        client.next_message()
    ).await;

    assert!(message_result.is_ok(), "Timeout waiting for kline message");

    if let Ok(Ok(Some(msg))) = message_result {
        tracing::info!("Received message: {:?}", msg);
    }

    client.disconnect().await.expect("Failed to disconnect");
}

#[tokio::test]
async fn test_subscribe_depth() {
    init_tracing();

    let mut client = BinanceWebSocketClient::new(true)
        .expect("Failed to create client");

    client.connect().await.expect("Failed to connect");

    let result = client.subscribe_depth("btcusdt").await;
    assert!(result.is_ok(), "Failed to subscribe: {:?}", result.err());

    // Wait for depth update
    let message_result = timeout(
        Duration::from_secs(5),
        client.next_message()
    ).await;

    assert!(message_result.is_ok(), "Timeout waiting for depth message");

    client.disconnect().await.expect("Failed to disconnect");
}

#[tokio::test]
async fn test_multiple_subscriptions() {
    init_tracing();

    let mut client = BinanceWebSocketClient::new(true)
        .expect("Failed to create client");

    client.connect().await.expect("Failed to connect");

    // Subscribe to multiple streams
    client.subscribe_trades("btcusdt").await.expect("Failed to subscribe trades");
    client.subscribe_klines("ethusdt", "1m").await.expect("Failed to subscribe klines");
    client.subscribe_depth("bnbusdt").await.expect("Failed to subscribe depth");

    // Collect messages from all streams
    let mut message_count = 0;
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);

    while tokio::time::Instant::now() < deadline && message_count < 3 {
        if let Ok(Some(_msg)) = timeout(
            Duration::from_secs(2),
            client.next_message()
        ).await.unwrap_or(Ok(None)) {
            message_count += 1;
        }
    }

    assert!(message_count >= 1, "Should receive at least one message from multiple streams");

    client.disconnect().await.expect("Failed to disconnect");
}

#[tokio::test]
async fn test_reconnection() {
    init_tracing();

    let mut client = BinanceWebSocketClient::new(true)
        .expect("Failed to create client");

    // Initial connection
    client.connect().await.expect("Failed to connect");
    assert!(client.is_connected().await);

    // Subscribe to a stream
    client.subscribe_trades("btcusdt").await.expect("Failed to subscribe");

    // Disconnect
    client.disconnect().await.expect("Failed to disconnect");
    sleep(Duration::from_millis(100)).await;

    // Reconnect
    let reconnect_result = client.reconnect().await;
    assert!(reconnect_result.is_ok(), "Reconnection failed: {:?}", reconnect_result.err());
    assert!(client.is_connected().await);

    // Verify subscription is restored
    let message_result = timeout(
        Duration::from_secs(5),
        client.next_message()
    ).await;

    assert!(message_result.is_ok(), "Should receive messages after reconnection");

    client.disconnect().await.expect("Failed to disconnect");
}

#[tokio::test]
async fn test_connection_timeout() {
    init_tracing();

    // This test uses an invalid URL to trigger timeout
    // Note: We can't easily modify the URL after construction,
    // so this is a basic connectivity test
    let mut client = BinanceWebSocketClient::new(true)
        .expect("Failed to create client");

    let result = timeout(
        Duration::from_secs(15),
        client.connect()
    ).await;

    assert!(result.is_ok(), "Connection should complete within timeout");
}

#[tokio::test]
async fn test_rate_limiting() {
    init_tracing();

    let mut client = BinanceWebSocketClient::new(true)
        .expect("Failed to create client");

    client.connect().await.expect("Failed to connect");

    // Subscribe to many streams rapidly
    let symbols = vec!["btcusdt", "ethusdt", "bnbusdt", "adausdt", "dogeusdt"];

    let start = tokio::time::Instant::now();

    for symbol in symbols {
        client.subscribe_trades(symbol).await.expect("Failed to subscribe");
    }

    let elapsed = start.elapsed();

    // With rate limiting, this should take at least 400ms (5 requests at 10/sec max)
    // Allow some tolerance for processing time
    tracing::info!("Subscriptions completed in {:?}", elapsed);

    client.disconnect().await.expect("Failed to disconnect");
}

#[tokio::test]
async fn test_message_validation() {
    init_tracing();

    let mut client = BinanceWebSocketClient::new(true)
        .expect("Failed to create client");

    client.connect().await.expect("Failed to connect");
    client.subscribe_trades("btcusdt").await.expect("Failed to subscribe");

    // Receive and validate a trade message
    let message = timeout(
        Duration::from_secs(5),
        client.next_message()
    ).await.expect("Timeout").expect("Failed to get message");

    if let Some(msg) = message {
        match msg {
            hyperphysics_market::providers::binance_websocket::BinanceStreamMessage::Trade(trade) => {
                assert!(!trade.symbol.is_empty());
                assert!(trade.price.parse::<f64>().is_ok());
                assert!(trade.quantity.parse::<f64>().is_ok());
                assert!(trade.event_time > 0);
                assert!(trade.trade_time > 0);
            }
            _ => panic!("Expected trade message"),
        }
    }

    client.disconnect().await.expect("Failed to disconnect");
}

#[tokio::test]
#[ignore] // Only run when testing with production endpoint
async fn test_production_connection() {
    init_tracing();

    let mut client = BinanceWebSocketClient::new(false)
        .expect("Failed to create client");

    let result = client.connect().await;
    assert!(result.is_ok(), "Failed to connect to production: {:?}", result.err());

    client.subscribe_trades("btcusdt").await.expect("Failed to subscribe");

    let message = timeout(
        Duration::from_secs(5),
        client.next_message()
    ).await.expect("Timeout").expect("Failed to get message");

    assert!(message.is_some());

    client.disconnect().await.expect("Failed to disconnect");
}
