# Binance WebSocket Client Setup Guide

This guide provides comprehensive instructions for setting up and using the production-ready Binance WebSocket client.

## Table of Contents

1. [Overview](#overview)
2. [Configuration](#configuration)
3. [Authentication](#authentication)
4. [Usage Examples](#usage-examples)
5. [Error Handling](#error-handling)
6. [Performance Tuning](#performance-tuning)
7. [Testing](#testing)

## Overview

The Binance WebSocket client provides real-time market data streaming with:

- **Automatic Reconnection**: Exponential backoff strategy (1s to 60s)
- **Rate Limiting**: 10 requests/second (Binance limit)
- **Circuit Breaker**: Prevents cascade failures (5 failures trigger 60s cooldown)
- **Thread Safety**: Concurrent message processing with tokio channels
- **Comprehensive Error Recovery**: Network failures, timeouts, invalid data

## Configuration

### Environment Variables

While the WebSocket streams are public and don't require authentication, you can configure the client behavior:

```bash
# Use testnet (default: false)
export BINANCE_TESTNET=true

# Logging level (optional)
export RUST_LOG=hyperphysics_market=debug
```

### Endpoints

The client automatically selects endpoints based on testnet configuration:

- **Production**: `wss://stream.binance.com:9443/ws`
- **Testnet**: `wss://testnet.binance.vision/ws`

## Authentication

**Note**: WebSocket streams for market data do not require API keys. However, if you plan to use private endpoints (user data streams), you'll need credentials:

```bash
# API credentials (for private streams only)
export BINANCE_API_KEY=your_api_key_here
export BINANCE_API_SECRET=your_secret_here
```

**SECURITY WARNING**:
- Never hardcode credentials in source code
- Use environment variables or secure secret management
- Restrict API key permissions to read-only for data feeds
- Rotate keys regularly

## Usage Examples

### Basic Connection

```rust
use hyperphysics_market::providers::BinanceWebSocketClient;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create client (testnet=false for production)
    let mut client = BinanceWebSocketClient::new(true)?;

    // Connect
    client.connect().await?;

    // Subscribe to trades
    client.subscribe_trades("btcusdt").await?;

    // Process messages
    while let Ok(Some(msg)) = client.next_message().await {
        println!("Received: {:?}", msg);
    }

    Ok(())
}
```

### Multiple Stream Subscriptions

```rust
use hyperphysics_market::providers::{
    BinanceWebSocketClient,
    binance_websocket::BinanceStreamMessage
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut client = BinanceWebSocketClient::new(false)?;
    client.connect().await?;

    // Subscribe to multiple symbols and types
    client.subscribe_trades("btcusdt").await?;
    client.subscribe_klines("ethusdt", "1m").await?;
    client.subscribe_depth("bnbusdt").await?;

    // Process different message types
    loop {
        match client.next_message().await? {
            Some(BinanceStreamMessage::Trade(trade)) => {
                println!("Trade: {} @ {}", trade.symbol, trade.price);
            }
            Some(BinanceStreamMessage::Kline(kline)) => {
                println!("Kline: {} OHLC: {}/{}/{}/{}",
                    kline.symbol,
                    kline.kline.open,
                    kline.kline.high,
                    kline.kline.low,
                    kline.kline.close
                );
            }
            Some(BinanceStreamMessage::DepthUpdate(depth)) => {
                println!("Depth: {} updates: {}-{}",
                    depth.symbol,
                    depth.first_update_id,
                    depth.final_update_id
                );
            }
            None => break,
            _ => {}
        }
    }

    Ok(())
}
```

### With Automatic Reconnection

```rust
use hyperphysics_market::providers::BinanceWebSocketClient;
use std::time::Duration;
use tokio::time::sleep;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut client = BinanceWebSocketClient::new(false)?;
    client.connect().await?;
    client.subscribe_trades("btcusdt").await?;

    loop {
        match client.next_message().await {
            Ok(Some(msg)) => {
                // Process message
                println!("Message: {:?}", msg);
            }
            Ok(None) => {
                sleep(Duration::from_millis(100)).await;
            }
            Err(e) => {
                eprintln!("Error: {}", e);

                // Attempt reconnection
                match client.reconnect().await {
                    Ok(_) => println!("Reconnected successfully"),
                    Err(e) => {
                        eprintln!("Reconnection failed: {}", e);
                        break;
                    }
                }
            }
        }
    }

    Ok(())
}
```

## Error Handling

### Error Types

The client handles various error conditions:

| Error Type | Cause | Recovery Strategy |
|------------|-------|-------------------|
| `ConnectionError` | Network issues, invalid URL | Automatic reconnection with backoff |
| `TimeoutError` | Connection/message timeout | Circuit breaker + reconnection |
| `NetworkError` | WebSocket failures | Exponential backoff retry |
| `ParseError` | Invalid message format | Log + skip message |
| `RateLimitExceeded` | Too many requests | Automatic queuing + throttling |

### Circuit Breaker

The circuit breaker protects against cascade failures:

- **Threshold**: 5 consecutive failures
- **Cooldown**: 60 seconds
- **States**: Closed (normal) → Open (blocking) → Half-Open (testing)

```rust
// Circuit breaker is automatic, but you can check connection status
if !client.is_connected().await {
    println!("Not connected - circuit breaker may be open");
}
```

### Rate Limiting

Automatically enforced at 10 requests/second:

```rust
// These will be automatically throttled
for symbol in ["btcusdt", "ethusdt", "bnbusdt"] {
    client.subscribe_trades(symbol).await?;
    // Automatic delay inserted if rate limit reached
}
```

## Performance Tuning

### Configuration Constants

You can adjust these in the source code if needed:

```rust
// Connection
const CONNECTION_TIMEOUT: Duration = Duration::from_secs(10);
const MESSAGE_TIMEOUT: Duration = Duration::from_secs(30);

// Rate Limiting
const RATE_LIMIT_PER_SECOND: u32 = 10;

// Reconnection
const MAX_RECONNECT_ATTEMPTS: u32 = 10;
const INITIAL_BACKOFF_MS: u64 = 1000;
const MAX_BACKOFF_MS: u64 = 60000;

// Circuit Breaker
const CIRCUIT_BREAKER_THRESHOLD: u32 = 5;
const CIRCUIT_BREAKER_TIMEOUT: Duration = Duration::from_secs(60);
```

### Memory Optimization

The client uses unbounded channels for message passing. For high-frequency streams, consider:

1. Processing messages immediately
2. Batching database writes
3. Using async processing pools
4. Monitoring channel depth

## Testing

### Unit Tests

Run unit tests for core functionality:

```bash
cargo test --package hyperphysics-market --lib binance_websocket
```

### Property Tests

Verify data integrity constraints:

```bash
cargo test --package hyperphysics-market --test binance_websocket_properties
```

### Integration Tests

Test with real Binance testnet:

```bash
cargo test --package hyperphysics-market --test binance_websocket_integration
```

**Note**: Integration tests require network connectivity and may take longer.

### Production Testing

Run against live production (use with caution):

```bash
cargo test --package hyperphysics-market --test binance_websocket_integration -- --ignored
```

### Example Application

Run the example to see live data:

```bash
cargo run --example binance_websocket_example
```

## Best Practices

1. **Error Handling**: Always handle `Err` cases in `next_message()`
2. **Graceful Shutdown**: Call `disconnect()` before dropping client
3. **Logging**: Enable debug logging for troubleshooting
4. **Monitoring**: Track reconnection frequency and circuit breaker trips
5. **Resource Limits**: Monitor memory usage for long-running connections
6. **Security**: Never log API secrets in production
7. **Testing**: Use testnet for development, production for final validation

## Troubleshooting

### Connection Failures

```
Error: Connection error: WebSocket connection failed
```

**Solution**: Check network connectivity, firewall rules, and endpoint availability.

### Circuit Breaker Open

```
Error: Circuit breaker open - too many failures
```

**Solution**: Wait 60 seconds for automatic recovery, or investigate underlying network issues.

### Rate Limit Exceeded

```
Warning: Rate limit reached, waiting 500ms
```

**Solution**: This is automatic throttling - no action needed. Reduce subscription frequency if persistent.

### Message Parsing Errors

```
Warning: Failed to parse message: missing field `s`
```

**Solution**: Message format may have changed - check Binance API documentation for updates.

## API Reference

### `BinanceWebSocketClient`

#### Methods

- `new(testnet: bool) -> Result<Self>` - Create new client
- `connect() -> Result<()>` - Establish WebSocket connection
- `disconnect() -> Result<()>` - Close connection gracefully
- `reconnect() -> Result<()>` - Reconnect with exponential backoff
- `subscribe_trades(symbol: &str) -> Result<()>` - Subscribe to trade stream
- `subscribe_klines(symbol: &str, interval: &str) -> Result<()>` - Subscribe to kline stream
- `subscribe_depth(symbol: &str) -> Result<()>` - Subscribe to depth stream
- `next_message() -> Result<Option<BinanceStreamMessage>>` - Get next message
- `is_connected() -> bool` - Check connection status

#### Message Types

- `BinanceStreamMessage::Trade(TradeEvent)` - Individual trade
- `BinanceStreamMessage::Kline(KlineEvent)` - Candlestick update
- `BinanceStreamMessage::DepthUpdate(DepthUpdateEvent)` - Orderbook update
- `BinanceStreamMessage::Ticker24hr(Ticker24hrEvent)` - 24h ticker stats

## Resources

- [Binance WebSocket API Documentation](https://binance-docs.github.io/apidocs/spot/en/#websocket-market-streams)
- [Binance Testnet](https://testnet.binance.vision/)
- [crates.io: tokio-tungstenite](https://docs.rs/tokio-tungstenite/)
