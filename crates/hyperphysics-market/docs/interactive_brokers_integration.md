# Interactive Brokers Client Portal Gateway Integration

## Overview

This module provides production-grade integration with Interactive Brokers using the **Client Portal Gateway REST API**. The implementation supports historical market data, real-time snapshots, and contract resolution with comprehensive error handling and rate limiting.

## Architecture

### Design Principles

1. **REST API over TWS API**: Uses Client Portal Gateway REST API for better reliability and simpler deployment
2. **Session-based Authentication**: Leverages browser-based SSO authentication
3. **Rate Limiting**: Token bucket algorithm with 60 requests/minute limit
4. **Data Validation**: OHLC consistency checks and chronological ordering
5. **Connection Management**: Auto-reconnect and session validation
6. **Retry Logic**: Exponential backoff with configurable max retries

### Key Components

- **InteractiveBrokersProvider**: Main provider implementation
- **IBKRBar**: Historical OHLC bar data structure
- **IBKRSnapshot**: Real-time market snapshot
- **IBKRContract**: Contract search result
- **RateLimiter**: Token bucket rate limiting

## Setup

### Prerequisites

1. **Interactive Brokers Account**: Paper or live trading account
2. **Client Portal Gateway**: Download from IBKR
3. **Authentication**: Active browser session

### Installation

1. Download Client Portal Gateway from Interactive Brokers website
2. Extract and configure `root/conf.yaml`:
   ```yaml
   ips:
     allow:
       - 127.0.0.1

   listenPort: 5000
   listenSsl: true
   ```

3. Start the gateway:
   ```bash
   cd clientportal.gw
   bin/run.sh root/conf.yaml
   # or on Windows:
   # bin\run.bat root\conf.yaml
   ```

4. Authenticate via web browser:
   ```
   https://localhost:5000
   ```

## Usage

### Basic Example

```rust
use hyperphysics_market::providers::{InteractiveBrokersProvider, MarketDataProvider};
use hyperphysics_market::data::Timeframe;
use chrono::{Utc, Duration};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create provider
    let provider = InteractiveBrokersProvider::new(
        "https://localhost:5000".to_string(),
    ).await?;

    // Authenticate (checks existing session)
    provider.authenticate().await?;

    // Fetch historical bars
    let end = Utc::now();
    let start = end - Duration::days(7);

    let bars = provider.fetch_bars(
        "AAPL",
        Timeframe::Day1,
        start,
        end
    ).await?;

    println!("Fetched {} bars", bars.len());

    // Fetch latest bar
    let latest = provider.fetch_latest_bar("AAPL").await?;
    println!("Latest close: ${:.2}", latest.close);

    Ok(())
}
```

### Real-time Data

```rust
// Fetch real-time quote
let quote = provider.fetch_quote("AAPL").await?;
println!("Bid: ${:.2}, Ask: ${:.2}", quote.bid_price, quote.ask_price);
println!("Spread: ${:.4}", quote.spread());

// Fetch latest tick
let tick = provider.fetch_tick("AAPL").await?;
println!("Last trade: ${:.2} @ {}", tick.price, tick.timestamp);
```

### Contract Search

```rust
// Search for contracts
let contracts = provider.search_contract("AAPL").await?;

for contract in contracts {
    println!("Contract {}: {} ({})",
        contract.contract_id,
        contract.symbol,
        contract.sec_type
    );
}
```

### Connection Management

```rust
// Health check
if provider.health_check().await.is_ok() {
    println!("Gateway is healthy");
}

// Validate session
if provider.validate_session().await? {
    println!("Session is active");
} else {
    println!("Session expired, reconnecting...");
    provider.reconnect().await?;
}

// Retry logic
let result = provider.with_retry(3, || async {
    provider.fetch_latest_bar("AAPL").await
}).await?;
```

## API Features

### Historical Data

```rust
async fn fetch_bars(
    &self,
    symbol: &str,
    timeframe: Timeframe,
    start: DateTime<Utc>,
    end: DateTime<Utc>,
) -> MarketResult<Vec<Bar>>
```

**Supported Timeframes:**
- `Timeframe::Minute1` → 1-minute bars
- `Timeframe::Minute5` → 5-minute bars
- `Timeframe::Minute15` → 15-minute bars
- `Timeframe::Minute30` → 30-minute bars
- `Timeframe::Hour1` → 1-hour bars
- `Timeframe::Hour4` → 4-hour bars
- `Timeframe::Day1` → Daily bars
- `Timeframe::Week1` → Weekly bars
- `Timeframe::Month1` → Monthly bars

### Real-time Data

```rust
// Fetch latest bar (historical or snapshot-based)
async fn fetch_latest_bar(&self, symbol: &str) -> MarketResult<Bar>

// Fetch real-time quote (bid/ask)
async fn fetch_quote(&self, symbol: &str) -> MarketResult<Quote>

// Fetch latest tick (trade)
async fn fetch_tick(&self, symbol: &str) -> MarketResult<Tick>

// Fetch market snapshot
async fn fetch_snapshot(&self, symbol: &str) -> MarketResult<IBKRSnapshot>
```

### Contract Management

```rust
// Search for contracts by symbol
async fn search_contract(&self, symbol: &str) -> MarketResult<Vec<IBKRContract>>

// Check if symbol is supported
async fn supports_symbol(&self, symbol: &str) -> MarketResult<bool>
```

### Connection Management

```rust
// Authenticate with existing session
async fn authenticate(&self) -> MarketResult<()>

// Check gateway health
async fn health_check(&self) -> MarketResult<()>

// Validate session is active
async fn validate_session(&self) -> MarketResult<bool>

// Reconnect after session expiration
async fn reconnect(&self) -> MarketResult<()>

// Execute with retry logic
async fn with_retry<F, Fut, T>(&self, max_retries: u32, f: F) -> MarketResult<T>
```

## Data Validation

### OHLC Validation

All bars are validated for:
- **Non-negative volume**
- **High >= Low**
- **Open within [Low, High]**
- **Close within [Low, High]**
- **Positive timestamp**

```rust
impl IBKRBar {
    pub fn validate(&self) -> Result<(), String> {
        // Validation logic
    }
}
```

### Chronological Ordering

Bars are validated to be in chronological order:

```rust
fn validate_chronological_order(&self, bars: &[IBKRBar]) -> MarketResult<()>
```

## Rate Limiting

### Token Bucket Algorithm

- **Rate**: 60 requests per minute
- **Window**: 60 seconds
- **Strategy**: Automatic wait when limit exceeded

```rust
struct RateLimiter {
    max_requests: 60,
    window: Duration::from_secs(60),
}
```

### Usage

Rate limiting is automatic - no manual intervention needed:

```rust
// Automatically waits if rate limit exceeded
self.rate_limiter.write().await.wait_for_slot().await;
```

## Error Handling

### Error Types

```rust
pub enum MarketError {
    ApiError(String),              // API-specific errors
    NetworkError(String),           // Network failures
    ConnectionError(String),        // Connection issues
    AuthenticationError(String),    // Auth failures
    RateLimitExceeded(String),     // Rate limit hit
    DataUnavailable(String),       // No data available
    DataIntegrityError(String),    // Validation failures
    TimeoutError(String),          // Request timeout
    ConfigError(String),           // Configuration issues
}
```

### Retry Strategy

Automatic retry with exponential backoff:

1. Initial delay: 100ms
2. Exponential growth: 2^(attempt-1)
3. Auto-reconnect on session expiry
4. Configurable max retries (default: 3)

```rust
let result = provider.with_retry(3, || async {
    provider.fetch_bars("AAPL", Timeframe::Day1, start, end).await
}).await?;
```

## Testing

### Unit Tests

```bash
# Run all unit tests
cargo test -p hyperphysics-market --lib interactive_brokers

# Run specific test
cargo test -p hyperphysics-market --lib test_bar_validation
```

### Integration Tests

```bash
# Set environment variables
export RUN_IBKR_INTEGRATION_TESTS=1
export IBKR_GATEWAY_URL=https://localhost:5000

# Run integration tests
cargo test --test integration_ib -- --nocapture

# Run specific integration test
cargo test --test integration_ib test_authentication -- --nocapture
```

**Integration Test Prerequisites:**
1. Running Client Portal Gateway
2. Authenticated session
3. Market data permissions (for data tests)
4. Paper trading account (recommended)

## Performance Characteristics

### Throughput

- **Rate limit**: 60 requests/minute
- **Typical latency**: 100-500ms per request
- **Concurrent requests**: Handled via rate limiter queue

### Resource Usage

- **Memory**: Minimal (contract cache only)
- **CPU**: Low (primarily I/O bound)
- **Network**: HTTPS with TLS (self-signed cert support)

## Security Considerations

### Authentication

- **Session-based**: Browser SSO authentication
- **No credentials in code**: Session cookie managed by HTTP client
- **Self-signed certs**: Support for localhost SSL

### Best Practices

1. **Never commit credentials**: No API keys in code
2. **Use paper trading**: For testing and development
3. **Validate data**: Always check OHLC consistency
4. **Monitor rate limits**: Track request patterns
5. **Secure connections**: Use HTTPS only

## Limitations

### Known Limitations

1. **No WebSocket support**: REST API only (polling for real-time data)
2. **Session management**: Manual authentication via browser required
3. **Rate limiting**: 60 requests/minute hard limit
4. **Market data permissions**: Requires subscriptions for real-time data
5. **Paper/Live switching**: Requires gateway restart

### Workarounds

- **Real-time streaming**: Use snapshot polling with rate-aware intervals
- **Session expiry**: Implement periodic session validation
- **Rate limits**: Batch requests and cache results

## Troubleshooting

### Common Issues

#### 1. Authentication Failed

```
Error: Session not authenticated
```

**Solution**: Authenticate via browser at `https://localhost:5000`

#### 2. Gateway Not Running

```
Error: Connection refused (os error 61)
```

**Solution**: Start Client Portal Gateway

#### 3. Rate Limit Exceeded

```
Error: Rate limit exceeded
```

**Solution**: Reduce request frequency or use caching

#### 4. No Market Data

```
Error: Data unavailable
```

**Solution**: Check market data subscriptions and market hours

#### 5. SSL Certificate Error

```
Error: SSL certificate problem
```

**Solution**: Provider accepts self-signed certs by default

### Debug Mode

Enable debug logging:

```rust
use tracing_subscriber;

tracing_subscriber::fmt()
    .with_max_level(tracing::Level::DEBUG)
    .init();
```

## API References

- [IBKR Client Portal Gateway Documentation](https://www.interactivebrokers.com/api/doc.html)
- [Historical Data Endpoint](https://www.interactivebrokers.com/api/doc.html#tag/Market-Data/paths/~1hmds~1history/get)
- [Market Snapshots Endpoint](https://www.interactivebrokers.com/api/doc.html#tag/Market-Data/paths/~1iserver~1marketdata~1snapshot/get)
- [Contract Search Endpoint](https://www.interactivebrokers.com/api/doc.html#tag/Contract/paths/~1iserver~1secdef~1search/post)

## Future Enhancements

### Planned Features

1. **WebSocket streaming**: Real-time data via SSE/WebSocket
2. **Order execution**: Trading capabilities
3. **Account management**: Portfolio and position tracking
4. **Options data**: Chains and Greeks
5. **Futures support**: Contract roll and expiration handling

### Contributing

Contributions welcome! Areas for improvement:

- Enhanced error messages
- Additional data types (Level 2, time & sales)
- Performance optimizations
- More comprehensive tests
- Documentation improvements

## License

This module is part of the HyperPhysics project and follows the project's licensing terms.

## Support

For issues specific to this integration:
1. Check IBKR Client Portal Gateway logs
2. Review integration test results
3. Enable debug logging
4. Consult IBKR API documentation

For general market data provider issues, see the main hyperphysics-market documentation.
