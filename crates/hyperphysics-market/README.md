# HyperPhysics Market Data Module

Integration layer for market data providers with topological analysis capabilities.

## Overview

This crate provides:

- **Unified API** for multiple market data providers
- **Real-time and historical** data fetching
- **Topological mapping** of market structures
- **Cross-provider compatibility** through common data models

## Supported Providers

### Phase 1 (Current)
- ✅ Alpaca Markets (stocks, crypto) - Skeleton implemented
- 🚧 Interactive Brokers - Stub
- 🚧 Binance - Stub

### Phase 2 (Planned)
- Polygon.io
- Alpha Vantage
- Yahoo Finance (free tier)
- Coinbase Pro

## Usage

```rust
use hyperphysics_market::providers::{AlpacaProvider, MarketDataProvider};
use hyperphysics_market::data::Timeframe;
use chrono::{Utc, Duration};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create provider
    let provider = AlpacaProvider::new(
        std::env::var("ALPACA_API_KEY")?,
        std::env::var("ALPACA_API_SECRET")?,
        true, // paper trading
    );

    // Fetch historical data
    let end = Utc::now();
    let start = end - Duration::days(30);
    let bars = provider.fetch_bars("AAPL", Timeframe::Day1, start, end).await?;

    // Fetch latest data
    let latest = provider.fetch_latest_bar("AAPL").await?;

    println!("Latest AAPL: ${}", latest.close);

    Ok(())
}
```

## Data Models

### Bar (OHLCV)
```rust
pub struct Bar {
    pub symbol: String,
    pub timestamp: DateTime<Utc>,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: u64,
    pub vwap: Option<f64>,
    pub trade_count: Option<u64>,
}
```

### Timeframes
- `Minute1`, `Minute5`, `Minute15`, `Minute30`
- `Hour1`, `Hour4`
- `Day1`, `Week1`, `Month1`

## Topological Analysis

(Phase 2 - Integration with hyperphysics-geometry)

```rust
use hyperphysics_market::topology::MarketTopologyMapper;

let mapper = MarketTopologyMapper::new();
let point_cloud = mapper.map_bars_to_point_cloud(&bars)?;
let persistence = mapper.compute_persistence(&bars)?;
```

## Development Status

### ✅ Completed
- [x] Crate structure
- [x] Core data models (Bar, Tick, Quote, OrderBook)
- [x] Provider trait interface
- [x] Alpaca provider skeleton
- [x] Error handling
- [x] Basic tests

### 🚧 In Progress (Phase 2)
- [ ] Alpaca API implementation
- [ ] Interactive Brokers integration
- [ ] Binance integration
- [ ] Topological mapping
- [ ] Real-time streaming
- [ ] Comprehensive tests
- [ ] Documentation
- [ ] Examples

## Dependencies

- `tokio` - Async runtime
- `reqwest` - HTTP client
- `serde` - Serialization
- `chrono` - Time handling
- `thiserror` - Error types
- `async-trait` - Async trait support

## Testing

```bash
cargo test --package hyperphysics-market
```

## Environment Variables

For Alpaca provider:
```bash
export ALPACA_API_KEY="your_key"
export ALPACA_API_SECRET="your_secret"
```

## License

MIT OR Apache-2.0
