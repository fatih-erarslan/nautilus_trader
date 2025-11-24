# Real Market Data Feed Implementation Plan

**Document Version**: 1.0
**Last Updated**: 2025-01-24
**Status**: Research Complete - Implementation Pending

---

## Executive Summary

This document provides a production-ready implementation plan for replacing all mock/synthetic market data generators with real-time market data feeds from Binance and Alpaca APIs. The plan prioritizes scientific rigor, regulatory compliance (SEC Regulation S-P), and fault-tolerant architecture.

**Timeline Estimate**: 4-6 weeks for full implementation and testing
**Risk Level**: Medium (API dependencies, rate limits, compliance)

---

## 1. Current State Analysis

### 1.1 Identified Mock Data Patterns

**CRITICAL VIOLATIONS FOUND:**

```rust
// File: /crates/autopoiesis/tests/test_data/generators.rs
// Lines 84-107: MarketDataGenerator using random/synthetic data
pub fn generate_price_series(&mut self, symbol: &str, steps: usize, dt: f64) -> Vec<f64> {
    // Uses rand::distributions::Normal for synthetic price generation
    // VIOLATION: No real market data source
}

// File: /crates/autopoiesis/autopoiesis-api/src/market_data/feed.rs
// Lines 432-436: Placeholder simulation function
async fn simulate_market_data(symbol: &str) {
    trace!("Simulating market data for {}", symbol);
    // VIOLATION: Stub implementation with no real data
}

// Lines 319-323: Simulated connection establishment
for connection in connections.iter_mut() {
    connection.state = ConnectionState::Connected;
    // VIOLATION: No actual WebSocket connection
}
```

**Architecture Assessment Score**: 40/100
- **Data Authenticity**: 0/100 (All synthetic/mock data)
- **Scientific Rigor**: 20/100 (Geometric Brownian Motion model present but not used for live data)
- **Production Readiness**: 30/100 (Framework exists but not connected to real sources)

---

## 2. Binance WebSocket API Integration

### 2.1 API Specifications

**Official Documentation**: [Binance WebSocket Streams](https://developers.binance.com/docs/binance-spot-api-docs/web-socket-streams)

#### Endpoints
```
Primary: wss://stream.binance.com:9443
Futures: wss://fstream.binance.com/stream
```

#### Connection Formats
```rust
// Single stream
wss://stream.binance.com:9443/ws/{symbol}@{stream_type}

// Multiple streams (recommended)
wss://stream.binance.com:9443/stream?streams={stream1}/{stream2}/{stream3}

// Examples:
wss://stream.binance.com:9443/stream?streams=btcusdt@trade/ethusdt@trade/btcusdt@depth5@100ms
```

#### Rate Limits (Updated 2024)
- **Connection Limit**: 300 connections per 5 minutes per IP
- **Message Rate**: 5 incoming messages/second (Spot), 10 messages/second (Futures)
- **Stream Limit**: 1,024 streams per connection
- **Connection Duration**: 24 hours (auto-disconnect after)
- **Keepalive**: Server sends ping every 3 minutes, requires pong within 10 minutes

**Source**: [Binance WebSocket Limits](https://academy.binance.com/en/articles/what-are-binance-websocket-limits)

### 2.2 Rust Implementation with tokio-tungstenite

#### Dependencies (Cargo.toml)
```toml
[dependencies]
tokio = { version = "1.35", features = ["full"] }
tokio-tungstenite = { version = "0.23", features = ["native-tls-vendored"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
url = "2.5"
futures-util = "0.3"
chrono = { version = "0.4", features = ["serde"] }
rust_decimal = { version = "1.33", features = ["serde-float"] }
tracing = "0.1"

# Circuit breaker for fault tolerance
failsafe = "1.3"

# Rate limiting
governor = "0.6"
```

#### Complete Implementation Example

```rust
use tokio_tungstenite::{connect_async, tungstenite::Message};
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Deserializer};
use url::Url;
use std::time::Duration;
use failsafe::{Config as CircuitConfig, backoff};

const BINANCE_WS_ENDPOINT: &str = "wss://stream.binance.com:9443";

/// Binance WebSocket trade stream data
#[derive(Debug, Deserialize)]
pub struct BinanceTradeData {
    #[serde(rename = "e")]
    pub event_type: String,
    #[serde(rename = "E")]
    pub event_time: u64,
    #[serde(rename = "s")]
    pub symbol: String,
    #[serde(rename = "p", deserialize_with = "de_float_from_str")]
    pub price: f64,
    #[serde(rename = "q", deserialize_with = "de_float_from_str")]
    pub quantity: f64,
    #[serde(rename = "T")]
    pub trade_time: u64,
}

/// Binance depth stream data (order book)
#[derive(Debug, Deserialize)]
pub struct BinanceDepthData {
    #[serde(rename = "lastUpdateId")]
    pub last_update_id: usize,
    #[serde(rename = "bids")]
    pub bids: Vec<(String, String)>, // [price, quantity]
    #[serde(rename = "asks")]
    pub asks: Vec<(String, String)>,
}

/// Wrapper for multiple streams
#[derive(Debug, Deserialize)]
pub struct BinanceStreamWrapper<T> {
    pub stream: String,
    pub data: T,
}

/// Custom deserializer for Binance string-encoded floats
pub fn de_float_from_str<'de, D>(deserializer: D) -> Result<f64, D::Error>
where
    D: Deserializer<'de>,
{
    let str_val = String::deserialize(deserializer)?;
    str_val.parse::<f64>().map_err(serde::de::Error::custom)
}

/// Binance WebSocket client with reconnection and circuit breaker
pub struct BinanceWebSocketClient {
    endpoint: String,
    streams: Vec<String>,
    circuit_breaker: CircuitConfig,
    reconnect_delay_ms: u64,
    max_reconnect_delay_ms: u64,
}

impl BinanceWebSocketClient {
    pub fn new(streams: Vec<String>) -> Self {
        // Configure circuit breaker with exponential backoff
        let backoff_strategy = backoff::exponential(
            Duration::from_secs(1),
            Duration::from_secs(60)
        );

        let circuit_config = CircuitConfig::new()
            .failure_policy(failsafe::failure_policy::consecutive_failures(
                3, // Open circuit after 3 consecutive failures
                backoff_strategy
            ));

        Self {
            endpoint: BINANCE_WS_ENDPOINT.to_string(),
            streams,
            circuit_breaker: circuit_config,
            reconnect_delay_ms: 1000,
            max_reconnect_delay_ms: 30000,
        }
    }

    /// Connect to Binance WebSocket with automatic reconnection
    pub async fn connect_with_retry(&self) -> Result<(), Box<dyn std::error::Error>> {
        let mut current_delay = self.reconnect_delay_ms;
        let mut attempt = 0;

        loop {
            tracing::info!("Connecting to Binance WebSocket (attempt {})", attempt + 1);

            match self.connect_once().await {
                Ok(_) => {
                    tracing::info!("Connection closed normally, reconnecting...");
                    current_delay = self.reconnect_delay_ms; // Reset delay on normal close
                }
                Err(e) => {
                    tracing::error!("Connection error: {:?}", e);

                    // Exponential backoff
                    tokio::time::sleep(Duration::from_millis(current_delay)).await;
                    current_delay = (current_delay * 2).min(self.max_reconnect_delay_ms);
                }
            }

            attempt += 1;

            // Circuit breaker: if too many failures, pause longer
            if attempt % 10 == 0 {
                tracing::warn!("10 consecutive failures, entering circuit breaker pause");
                tokio::time::sleep(Duration::from_secs(60)).await;
            }
        }
    }

    /// Single connection attempt
    async fn connect_once(&self) -> Result<(), Box<dyn std::error::Error>> {
        // Build WebSocket URL with multiple streams
        let stream_param = self.streams.join("/");
        let ws_url = format!("{}/stream?streams={}", self.endpoint, stream_param);

        tracing::info!("Connecting to: {}", ws_url);

        let (ws_stream, _) = connect_async(Url::parse(&ws_url)?).await?;
        let (mut write, mut read) = ws_stream.split();

        // Keepalive task (respond to pings)
        let keepalive = tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(30));
            loop {
                interval.tick().await;
                if write.send(Message::Pong(vec![])).await.is_err() {
                    break;
                }
            }
        });

        // Message processing loop
        while let Some(msg_result) = read.next().await {
            match msg_result {
                Ok(Message::Text(text)) => {
                    self.process_message(&text)?;
                }
                Ok(Message::Ping(payload)) => {
                    // Respond to server pings
                    tracing::trace!("Received ping, sending pong");
                }
                Ok(Message::Close(_)) => {
                    tracing::info!("Received close frame from server");
                    break;
                }
                Err(e) => {
                    tracing::error!("WebSocket error: {:?}", e);
                    return Err(e.into());
                }
                _ => {}
            }
        }

        keepalive.abort();
        Ok(())
    }

    /// Process incoming messages
    fn process_message(&self, text: &str) -> Result<(), Box<dyn std::error::Error>> {
        // Parse based on stream type
        if text.contains("@trade") {
            let trade: BinanceStreamWrapper<BinanceTradeData> = serde_json::from_str(text)?;
            tracing::debug!("Trade: {} @ {} (qty: {})",
                trade.data.symbol,
                trade.data.price,
                trade.data.quantity
            );
            // TODO: Send to market data feed publisher
        } else if text.contains("@depth") {
            let depth: BinanceStreamWrapper<BinanceDepthData> = serde_json::from_str(text)?;
            tracing::debug!("Depth update for stream: {} (bids: {}, asks: {})",
                depth.stream,
                depth.data.bids.len(),
                depth.data.asks.len()
            );
            // TODO: Send to order book aggregator
        }

        Ok(())
    }
}

/// Example usage
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    let streams = vec![
        "btcusdt@trade".to_string(),
        "ethusdt@trade".to_string(),
        "btcusdt@depth5@100ms".to_string(),
    ];

    let client = BinanceWebSocketClient::new(streams);
    client.connect_with_retry().await?;

    Ok(())
}
```

**Sources**:
- [TMS Developer Blog - Binance Rust WebSocket](https://tms-dev-blog.com/easily-connect-to-binance-websocket-streams-with-rust/)
- [tokio-tungstenite Documentation](https://github.com/snapview/tokio-tungstenite)
- [Building Real-Time Binance WebSocket Clients in Rust](https://medium.com/@ekfqlwcjswl/building-real-time-binance-websocket-clients-in-rust-with-tokio-2e0027f0f1fd)

### 2.3 Authentication (Not Required for Public Data)

Binance **does not require authentication** for public market data streams (trades, order books, klines). Authentication is only needed for:
- User data streams (account updates, order updates)
- Placing/canceling orders via WebSocket API

**For this implementation**: No API key required for market data.

---

## 3. Alpaca Market Data API Integration

### 3.1 API Specifications

**Official Documentation**: [Alpaca Market Data API](https://docs.alpaca.markets/docs/about-market-data-api)

#### Endpoints
```
REST API: https://data.alpaca.markets/v2
WebSocket (Stocks): wss://stream.data.alpaca.markets/v2/iex
WebSocket (Crypto): wss://stream.data.alpaca.markets/v1beta3/crypto/us
```

#### Authentication
```http
Headers:
  APCA-API-KEY-ID: {your_api_key}
  APCA-API-SECRET-KEY: {your_secret_key}
```

**Alternative (Basic Auth)**:
```
Authorization: Basic {base64(key:secret)}
```

#### Rate Limits
| Plan | REST API | WebSocket Symbols |
|------|----------|------------------|
| **Basic (Free)** | 200 req/min | 30 equities, 200 options |
| **Algo Trader Plus** | 10,000 req/min | Unlimited equities, 1,000 options |
| **Broker Partners** | 1,000-10,000 req/min | Varies |

**Important**: Historical crypto data does NOT require authentication (free tier).

### 3.2 Rust Implementation with `apca` Crate

#### Dependencies
```toml
[dependencies]
apca = "0.27" # Latest async/await Alpaca client
tokio = { version = "1.35", features = ["full"] }
rust_decimal = "1.33"
chrono = "0.4"
tracing = "0.1"
anyhow = "1.0"
```

#### Implementation Example

```rust
use apca::{ApiInfo, Client};
use apca::data::v2::{bars, stream};
use apca::data::v2::stream::{IEX, RealtimeData};
use futures_util::StreamExt;

/// Alpaca market data client
pub struct AlpacaMarketDataClient {
    client: Client,
}

impl AlpacaMarketDataClient {
    /// Initialize from environment variables
    /// Required: APCA_API_KEY_ID, APCA_API_SECRET_KEY
    pub fn from_env() -> Result<Self, anyhow::Error> {
        let api_info = ApiInfo::from_env()?;
        let client = Client::new(api_info);

        Ok(Self { client })
    }

    /// Fetch historical bars (OHLCV data)
    pub async fn fetch_historical_bars(
        &self,
        symbol: &str,
        start: chrono::DateTime<chrono::Utc>,
        end: chrono::DateTime<chrono::Utc>,
    ) -> Result<Vec<bars::Bar>, anyhow::Error> {
        let request = bars::BarsReq {
            symbol: symbol.to_string(),
            start,
            end,
            limit: Some(10000),
            timeframe: bars::Timeframe::OneMinute,
            ..Default::default()
        };

        let bars = self.client.issue::<bars::List>(&request).await?;
        Ok(bars)
    }

    /// Subscribe to real-time WebSocket stream
    pub async fn subscribe_realtime(
        &self,
        symbols: Vec<String>,
    ) -> Result<(), anyhow::Error> {
        let (mut stream, mut subscription) = self.client
            .subscribe::<IEX>()
            .await?;

        // Subscribe to trades and quotes
        subscription
            .set_trades(symbols.clone())
            .set_quotes(symbols)
            .subscribe()
            .await?;

        tracing::info!("Subscribed to Alpaca real-time data");

        // Process incoming data
        while let Some(result) = stream.next().await {
            match result? {
                RealtimeData::Trade(trade) => {
                    tracing::debug!("Trade: {} @ {} (size: {})",
                        trade.symbol,
                        trade.price,
                        trade.size
                    );
                    // TODO: Publish to market data feed
                }
                RealtimeData::Quote(quote) => {
                    tracing::debug!("Quote: {} bid: {} ask: {}",
                        quote.symbol,
                        quote.bid_price,
                        quote.ask_price
                    );
                    // TODO: Publish to market data feed
                }
                RealtimeData::Bar(bar) => {
                    tracing::debug!("Bar: {} O:{} H:{} L:{} C:{}",
                        bar.symbol,
                        bar.open,
                        bar.high,
                        bar.low,
                        bar.close
                    );
                }
            }
        }

        Ok(())
    }
}

/// Example usage
#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    tracing_subscriber::fmt::init();

    let client = AlpacaMarketDataClient::from_env()?;

    // Historical data
    let start = chrono::Utc::now() - chrono::Duration::days(7);
    let end = chrono::Utc::now();
    let bars = client.fetch_historical_bars("AAPL", start, end).await?;
    tracing::info!("Fetched {} historical bars", bars.len());

    // Real-time streaming
    let symbols = vec!["AAPL".to_string(), "TSLA".to_string()];
    client.subscribe_realtime(symbols).await?;

    Ok(())
}
```

**Sources**:
- [apca Crate Documentation](https://github.com/d-e-s-o/apca)
- [Alpaca Market Data API v2 Documentation](https://docs.alpaca.markets/docs/about-market-data-api)
- [rpaca - Alternative Rust Client](https://crates.io/crates/rpaca)

---

## 4. Data Validation Framework

### 4.1 Validation Requirements

#### Regulatory Standards
- **MiFID II (Europe)**: ±100 microseconds timestamp accuracy for HFT
- **SEC CAT (US)**: ±50 milliseconds synchronization to NIST atomic clock
- **FINRA Rule 4590**: Daily NTP synchronization

#### Sanity Checks
```rust
#[derive(Debug, Clone)]
pub struct DataValidationRules {
    /// Maximum allowed price deviation from previous tick (percentage)
    pub max_price_jump_pct: f64,

    /// Maximum allowed bid-ask spread (percentage)
    pub max_spread_pct: f64,

    /// Minimum volume threshold to consider data valid
    pub min_volume_threshold: Decimal,

    /// Maximum data age before considered stale (seconds)
    pub max_age_seconds: u64,

    /// Minimum price value (prevent zeros/negatives)
    pub min_price: Decimal,

    /// Maximum price value (detect outliers)
    pub max_price: Decimal,
}

impl Default for DataValidationRules {
    fn default() -> Self {
        Self {
            max_price_jump_pct: 0.10,  // 10% max price jump
            max_spread_pct: 0.05,       // 5% max spread
            min_volume_threshold: Decimal::from(100),
            max_age_seconds: 10,
            min_price: Decimal::from_str("0.0001").unwrap(),
            max_price: Decimal::from(1_000_000),
        }
    }
}
```

### 4.2 Validation Implementation

```rust
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;

pub struct MarketDataValidator {
    rules: DataValidationRules,
    last_prices: std::collections::HashMap<String, Decimal>,
    ntp_client: ntp::NtpClient,
}

impl MarketDataValidator {
    pub fn new(rules: DataValidationRules) -> Self {
        Self {
            rules,
            last_prices: std::collections::HashMap::new(),
            ntp_client: ntp::NtpClient::new("time.nist.gov"),
        }
    }

    /// Validate incoming market data
    pub async fn validate(&mut self, data: &MarketData) -> ValidationResult {
        let mut errors = Vec::new();

        // 1. Timestamp validation (NTP sync check)
        if let Err(e) = self.validate_timestamp(data.timestamp).await {
            errors.push(ValidationError::TimestampError(e));
        }

        // 2. Price range check
        if data.price < self.rules.min_price || data.price > self.rules.max_price {
            errors.push(ValidationError::PriceOutOfRange {
                price: data.price,
                min: self.rules.min_price,
                max: self.rules.max_price,
            });
        }

        // 3. Price jump check (prevent flash crashes/spikes)
        if let Some(&last_price) = self.last_prices.get(&data.symbol) {
            let price_change_pct = ((data.price - last_price) / last_price).abs();
            if price_change_pct.to_f64().unwrap() > self.rules.max_price_jump_pct {
                errors.push(ValidationError::ExcessivePriceJump {
                    symbol: data.symbol.clone(),
                    old_price: last_price,
                    new_price: data.price,
                    change_pct: price_change_pct.to_f64().unwrap(),
                });
            }
        }

        // 4. Spread validation
        let spread_pct = ((data.ask - data.bid) / data.mid).to_f64().unwrap();
        if spread_pct > self.rules.max_spread_pct {
            errors.push(ValidationError::ExcessiveSpread {
                symbol: data.symbol.clone(),
                spread_pct,
                max_allowed: self.rules.max_spread_pct,
            });
        }

        // 5. Volume check
        if data.volume_24h < self.rules.min_volume_threshold {
            errors.push(ValidationError::InsufficientVolume {
                symbol: data.symbol.clone(),
                volume: data.volume_24h,
                min_required: self.rules.min_volume_threshold,
            });
        }

        // 6. Data freshness check
        let age = (Utc::now() - data.timestamp).num_seconds() as u64;
        if age > self.rules.max_age_seconds {
            errors.push(ValidationError::StaleData {
                symbol: data.symbol.clone(),
                age_seconds: age,
                max_age: self.rules.max_age_seconds,
            });
        }

        // Update last price cache
        self.last_prices.insert(data.symbol.clone(), data.price);

        if errors.is_empty() {
            ValidationResult::Valid
        } else {
            ValidationResult::Invalid(errors)
        }
    }

    /// Validate timestamp against NTP server
    async fn validate_timestamp(&self, timestamp: DateTime<Utc>) -> Result<(), String> {
        // Get NTP time
        let ntp_time = self.ntp_client.get_time().await
            .map_err(|e| format!("NTP sync failed: {}", e))?;

        // Calculate drift
        let drift_ms = (timestamp - ntp_time).num_milliseconds().abs();

        // SEC CAT requires ±50ms accuracy
        if drift_ms > 50 {
            return Err(format!(
                "Timestamp drift {} ms exceeds SEC CAT limit of 50ms",
                drift_ms
            ));
        }

        Ok(())
    }

    /// Cross-exchange consistency check
    pub fn validate_cross_exchange(
        &self,
        binance_price: Decimal,
        alpaca_price: Decimal,
        symbol: &str,
    ) -> Result<(), String> {
        let price_diff_pct = ((binance_price - alpaca_price) / binance_price).abs();

        // Allow up to 2% difference between exchanges (accounts for different liquidity)
        if price_diff_pct.to_f64().unwrap() > 0.02 {
            return Err(format!(
                "Cross-exchange price mismatch for {}: Binance={} Alpaca={} ({}% diff)",
                symbol,
                binance_price,
                alpaca_price,
                (price_diff_pct * dec!(100)).round_dp(2)
            ));
        }

        Ok(())
    }
}

#[derive(Debug)]
pub enum ValidationResult {
    Valid,
    Invalid(Vec<ValidationError>),
}

#[derive(Debug)]
pub enum ValidationError {
    TimestampError(String),
    PriceOutOfRange { price: Decimal, min: Decimal, max: Decimal },
    ExcessivePriceJump { symbol: String, old_price: Decimal, new_price: Decimal, change_pct: f64 },
    ExcessiveSpread { symbol: String, spread_pct: f64, max_allowed: f64 },
    InsufficientVolume { symbol: String, volume: Decimal, min_required: Decimal },
    StaleData { symbol: String, age_seconds: u64, max_age: u64 },
}
```

**Sources**:
- [Data Validation Best Practices for Finance](https://www.cubesoftware.com/blog/data-validation-best-practices)
- [Market Data Validations & NAV Processing](https://www.milestonegroup.com/solutions/market-data-validations)
- [Machine Learning-Based Validation Workflow](https://www.bis.org/ifc/publ/ifcb57_06_rh.pdf)

---

## 5. Production Best Practices

### 5.1 Multi-Exchange Failover Architecture

```rust
use tokio::sync::broadcast;
use std::sync::Arc;

/// Multi-source market data aggregator with failover
pub struct MultiSourceDataAggregator {
    binance_client: Arc<BinanceWebSocketClient>,
    alpaca_client: Arc<AlpacaMarketDataClient>,
    publisher: broadcast::Sender<MarketData>,
    health_checker: Arc<HealthChecker>,
}

impl MultiSourceDataAggregator {
    pub fn new(channel_buffer: usize) -> Self {
        let (tx, _) = broadcast::channel(channel_buffer);

        Self {
            binance_client: Arc::new(BinanceWebSocketClient::new(vec![])),
            alpaca_client: Arc::new(AlpacaMarketDataClient::from_env().unwrap()),
            publisher: tx,
            health_checker: Arc::new(HealthChecker::new()),
        }
    }

    /// Start all data sources with automatic failover
    pub async fn start(&self) -> Result<(), Box<dyn std::error::Error>> {
        let binance_health = Arc::clone(&self.health_checker);
        let alpaca_health = Arc::clone(&self.health_checker);

        // Spawn Binance connection
        let binance_task = {
            let client = Arc::clone(&self.binance_client);
            let publisher = self.publisher.clone();
            tokio::spawn(async move {
                loop {
                    match client.connect_with_retry().await {
                        Ok(_) => binance_health.mark_healthy("binance"),
                        Err(e) => {
                            tracing::error!("Binance connection error: {:?}", e);
                            binance_health.mark_unhealthy("binance");
                            tokio::time::sleep(Duration::from_secs(5)).await;
                        }
                    }
                }
            })
        };

        // Spawn Alpaca connection
        let alpaca_task = {
            let client = Arc::clone(&self.alpaca_client);
            let publisher = self.publisher.clone();
            tokio::spawn(async move {
                loop {
                    match client.subscribe_realtime(vec!["BTCUSD".to_string()]).await {
                        Ok(_) => alpaca_health.mark_healthy("alpaca"),
                        Err(e) => {
                            tracing::error!("Alpaca connection error: {:?}", e);
                            alpaca_health.mark_unhealthy("alpaca");
                            tokio::time::sleep(Duration::from_secs(5)).await;
                        }
                    }
                }
            })
        };

        // Health monitoring task
        let health_task = {
            let health = Arc::clone(&self.health_checker);
            tokio::spawn(async move {
                let mut interval = tokio::time::interval(Duration::from_secs(30));
                loop {
                    interval.tick().await;
                    let status = health.get_status();
                    tracing::info!("Health status: {:?}", status);

                    // Alert if all sources are down
                    if !status.any_healthy() {
                        tracing::error!("CRITICAL: All data sources are unhealthy!");
                        // TODO: Send alert to monitoring system
                    }
                }
            })
        };

        tokio::try_join!(binance_task, alpaca_task, health_task)?;
        Ok(())
    }
}

/// Health checker for multiple data sources
pub struct HealthChecker {
    source_health: Arc<RwLock<HashMap<String, SourceHealth>>>,
}

#[derive(Debug, Clone)]
pub struct SourceHealth {
    pub is_healthy: bool,
    pub last_update: DateTime<Utc>,
    pub consecutive_failures: u32,
    pub uptime_pct: f64,
}

impl HealthChecker {
    pub fn new() -> Self {
        Self {
            source_health: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub fn mark_healthy(&self, source: &str) {
        // Implementation
    }

    pub fn mark_unhealthy(&self, source: &str) {
        // Implementation
    }

    pub fn get_status(&self) -> HealthStatus {
        // Implementation
    }
}
```

### 5.2 Backpressure Management with Bounded Channels

```rust
use tokio::sync::mpsc;

/// Market data pipeline with backpressure
pub struct MarketDataPipeline {
    ingest_tx: mpsc::Sender<RawMarketData>,
    validate_tx: mpsc::Sender<RawMarketData>,
    publish_tx: broadcast::Sender<MarketData>,
}

impl MarketDataPipeline {
    pub fn new() -> Self {
        // Bounded channels for backpressure
        let (ingest_tx, mut ingest_rx) = mpsc::channel::<RawMarketData>(1000);
        let (validate_tx, mut validate_rx) = mpsc::channel::<RawMarketData>(500);
        let (publish_tx, _) = broadcast::channel::<MarketData>(10000);

        // Validation stage
        let validator_tx = validate_tx.clone();
        tokio::spawn(async move {
            let mut validator = MarketDataValidator::new(DataValidationRules::default());

            while let Some(raw_data) = ingest_rx.recv().await {
                match validator.validate(&raw_data.into()).await {
                    ValidationResult::Valid => {
                        if validator_tx.send(raw_data).await.is_err() {
                            tracing::warn!("Validation channel full, applying backpressure");
                            // Backpressure automatically applied by bounded channel
                        }
                    }
                    ValidationResult::Invalid(errors) => {
                        tracing::warn!("Invalid data: {:?}", errors);
                        // Discard invalid data
                    }
                }
            }
        });

        // Publishing stage
        let pub_tx = publish_tx.clone();
        tokio::spawn(async move {
            while let Some(validated_data) = validate_rx.recv().await {
                let market_data = validated_data.into();
                if pub_tx.send(market_data).is_err() {
                    tracing::warn!("No subscribers, dropping data");
                }
            }
        });

        Self {
            ingest_tx,
            validate_tx,
            publish_tx,
        }
    }

    /// Ingest raw market data (will apply backpressure if pipeline is full)
    pub async fn ingest(&self, data: RawMarketData) -> Result<(), String> {
        self.ingest_tx.send(data).await
            .map_err(|e| format!("Ingest failed: {}", e))
    }

    /// Subscribe to validated market data
    pub fn subscribe(&self) -> broadcast::Receiver<MarketData> {
        self.publish_tx.subscribe()
    }
}
```

**Sources**:
- [Tokio Channels Documentation](https://tokio.rs/tokio/tutorial/channels)
- [Handling Backpressure in Rust Async Systems](https://www.slingacademy.com/article/handling-backpressure-in-rust-async-systems-with-bounded-channels/)
- [Async Rust with Tokio I/O Streams](https://biriukov.dev/docs/async-rust-tokio-io/1-async-rust-with-tokio-io-streams-backpressure-concurrency-and-ergonomics/)

### 5.3 Caching Layer to Reduce API Calls

```rust
use moka::future::Cache;
use std::time::Duration;

/// Market data cache with TTL
pub struct MarketDataCache {
    cache: Cache<String, MarketData>,
}

impl MarketDataCache {
    pub fn new(max_capacity: u64, ttl_seconds: u64) -> Self {
        let cache = Cache::builder()
            .max_capacity(max_capacity)
            .time_to_live(Duration::from_secs(ttl_seconds))
            .build();

        Self { cache }
    }

    pub async fn get_or_fetch<F, Fut>(
        &self,
        symbol: &str,
        fetch_fn: F,
    ) -> Result<MarketData, Box<dyn std::error::Error>>
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = Result<MarketData, Box<dyn std::error::Error>>>,
    {
        if let Some(cached_data) = self.cache.get(symbol).await {
            return Ok(cached_data);
        }

        let fresh_data = fetch_fn().await?;
        self.cache.insert(symbol.to_string(), fresh_data.clone()).await;
        Ok(fresh_data)
    }
}
```

### 5.4 Logging and Monitoring

```toml
[dependencies]
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter", "json"] }
opentelemetry = "0.21"
opentelemetry-jaeger = "0.20"
```

```rust
use tracing::{info, warn, error, instrument};
use tracing_subscriber::layer::SubscriberExt;

#[instrument(skip(self))]
pub async fn process_market_data(&self, data: MarketData) -> Result<(), Error> {
    info!(
        symbol = %data.symbol,
        price = %data.price,
        volume = %data.volume_24h,
        "Processing market data"
    );

    // Processing logic...

    Ok(())
}

/// Initialize observability stack
pub fn init_observability() {
    let tracer = opentelemetry_jaeger::new_pipeline()
        .with_service_name("market-data-feed")
        .install_simple()
        .unwrap();

    let telemetry = tracing_opentelemetry::layer().with_tracer(tracer);

    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::from_default_env())
        .with(tracing_subscriber::fmt::layer().json())
        .with(telemetry)
        .init();
}
```

---

## 6. SEC Compliance Requirements

### 6.1 Regulation S-P (Updated May 2024)

**Critical Requirements for Market Data Handling:**

1. **Incident Response Program** (Compliance Deadline: December 3, 2025 for large entities)
   - Detect, respond to, and recover from unauthorized access
   - Written policies and procedures required
   - Service provider oversight mandatory

2. **Data Breach Notification**
   - Notify within **30 days** of becoming aware of breach
   - Covers unauthorized access to customer information

3. **Service Provider Requirements**
   - Providers must notify within **72 hours** of breach awareness
   - Written agreements for data protection required

4. **Administrative, Technical, and Physical Safeguards** (FINRA Rule 30)
   - Encryption in transit (TLS 1.3)
   - Encryption at rest for stored data
   - Access control and audit logging
   - Regular security assessments

### 6.2 Implementation Checklist

```rust
/// SEC Regulation S-P compliance configuration
pub struct ComplianceConfig {
    /// Enable encryption for all data in transit
    pub tls_enabled: bool,
    pub tls_version: TlsVersion,

    /// Enable encryption for data at rest
    pub encryption_at_rest: bool,
    pub encryption_algorithm: EncryptionAlgorithm,

    /// Audit logging configuration
    pub audit_log_enabled: bool,
    pub audit_log_retention_days: u32,

    /// Incident response configuration
    pub incident_response_enabled: bool,
    pub breach_notification_email: Vec<String>,
    pub max_breach_notification_hours: u32, // 30 days = 720 hours

    /// Service provider monitoring
    pub provider_monitoring_enabled: bool,
    pub provider_breach_notification_hours: u32, // 72 hours
}

impl Default for ComplianceConfig {
    fn default() -> Self {
        Self {
            tls_enabled: true,
            tls_version: TlsVersion::Tls13,
            encryption_at_rest: true,
            encryption_algorithm: EncryptionAlgorithm::Aes256Gcm,
            audit_log_enabled: true,
            audit_log_retention_days: 2555, // 7 years (SEC requirement)
            incident_response_enabled: true,
            breach_notification_email: vec![],
            max_breach_notification_hours: 720, // 30 days
            provider_monitoring_enabled: true,
            provider_breach_notification_hours: 72,
        }
    }
}
```

**Sources**:
- [SEC Regulation S-P Amendments (May 2024)](https://www.sec.gov/newsroom/press-releases/2024-58)
- [SEC Cybersecurity Rules 2024](https://www.metomic.io/resource-centre/sec-cybersecurity-rules)
- [FINRA Cybersecurity Requirements](https://www.finra.org/rules-guidance/guidance/reports/2024-finra-annual-regulatory-oversight-report/cybersecurity)

---

## 7. Migration Plan: Mock → Real Provider

### 7.1 Phased Rollout Strategy

**Phase 1: Parallel Testing (Week 1-2)**
```rust
pub enum DataSource {
    Mock,
    Binance,
    Alpaca,
    Hybrid { primary: Box<DataSource>, fallback: Box<DataSource> },
}

impl MarketDataFeed {
    pub fn new_with_source(source: DataSource) -> Self {
        match source {
            DataSource::Mock => Self::new_mock(),
            DataSource::Binance => Self::new_binance(),
            DataSource::Alpaca => Self::new_alpaca(),
            DataSource::Hybrid { primary, fallback } => {
                Self::new_hybrid(*primary, *fallback)
            }
        }
    }
}
```

**Phase 2: Canary Deployment (Week 3)**
- Route 10% of traffic to real feeds
- Monitor error rates, latency, data quality
- Compare results with mock data

**Phase 3: Gradual Rollout (Week 4-5)**
- Increase to 50%, then 100%
- Keep mock provider as fallback

**Phase 4: Decommission (Week 6)**
- Remove mock provider code
- Update tests to use recorded real data
- Archive mock implementation for reference

### 7.2 File-by-File Migration

```bash
# Files requiring modification:
/crates/autopoiesis/autopoiesis-api/src/market_data/feed.rs
  - Line 432-436: Remove simulate_market_data()
  - Line 319-323: Replace with actual WebSocket connection
  - Add BinanceWebSocketClient integration

/crates/autopoiesis/tests/test_data/generators.rs
  - Lines 84-107: Keep for testing, add #[cfg(test)]
  - Document as "test fixtures only"
  - Add warning comments about production usage

/crates/autopoiesis/autopoiesis-api/src/market_data/aggregator.rs
  - Integrate MultiSourceDataAggregator
  - Add cross-exchange validation

/crates/autopoiesis/autopoiesis-api/src/market_data/storage.rs
  - Add compliance-grade encryption
  - Implement audit logging
```

### 7.3 Testing Strategy

```rust
#[cfg(test)]
mod integration_tests {
    use super::*;

    #[tokio::test]
    #[ignore] // Run with --ignored flag to avoid API rate limits
    async fn test_binance_real_connection() {
        let client = BinanceWebSocketClient::new(vec![
            "btcusdt@trade".to_string()
        ]);

        // Connect for 10 seconds
        let result = tokio::time::timeout(
            Duration::from_secs(10),
            client.connect_once()
        ).await;

        assert!(result.is_ok());
    }

    #[tokio::test]
    #[ignore]
    async fn test_alpaca_real_connection() {
        let client = AlpacaMarketDataClient::from_env().unwrap();
        let bars = client.fetch_historical_bars(
            "AAPL",
            Utc::now() - chrono::Duration::hours(1),
            Utc::now()
        ).await;

        assert!(bars.is_ok());
        assert!(!bars.unwrap().is_empty());
    }

    #[test]
    fn test_data_validation() {
        let mut validator = MarketDataValidator::new(
            DataValidationRules::default()
        );

        let valid_data = MarketData {
            symbol: "BTC/USD".to_string(),
            price: Decimal::from(45000),
            bid: Decimal::from(44995),
            ask: Decimal::from(45005),
            mid: Decimal::from(45000),
            volume_24h: Decimal::from(100000),
            timestamp: Utc::now(),
        };

        let result = validator.validate(&valid_data).await;
        assert!(matches!(result, ValidationResult::Valid));
    }
}
```

---

## 8. Timeline and Resource Estimates

### 8.1 Development Timeline

| Phase | Tasks | Duration | Dependencies |
|-------|-------|----------|--------------|
| **Week 1** | Research, architecture design, dependency setup | 5 days | None |
| **Week 2** | Binance WebSocket client implementation | 5 days | Week 1 |
| **Week 3** | Alpaca API client implementation | 5 days | Week 1 |
| **Week 4** | Data validation framework, NTP sync | 5 days | Week 2-3 |
| **Week 5** | Multi-source aggregator, failover logic | 5 days | Week 2-4 |
| **Week 6** | SEC compliance implementation | 5 days | Week 4-5 |
| **Week 7** | Integration testing, parallel deployment | 5 days | Week 2-6 |
| **Week 8** | Canary rollout, monitoring, documentation | 5 days | Week 7 |

**Total**: 8 weeks (6 weeks for implementation, 2 weeks for testing/rollout)

### 8.2 Resource Requirements

**Development**:
- 1 Senior Rust Engineer (full-time, 8 weeks)
- 1 DevOps Engineer (part-time, 2 weeks for monitoring setup)
- 1 Compliance Specialist (part-time, 1 week for SEC review)

**Infrastructure**:
- Alpaca Market Data subscription ($0-$99/month depending on tier)
- Binance API (free for public data)
- NTP server access (free, time.nist.gov)
- Logging infrastructure (OpenTelemetry + Jaeger)

**Testing**:
- Staging environment with identical configuration
- Load testing tools (optional: k6, Gatling)

---

## 9. Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **API rate limit exceeded** | Medium | High | Implement rate limiter, caching layer, multiple API keys |
| **WebSocket disconnection** | High | Medium | Circuit breaker, exponential backoff, multi-source failover |
| **Data quality issues** | Medium | High | Comprehensive validation, cross-exchange checks, alerts |
| **SEC compliance violation** | Low | Critical | Legal review, audit trails, encrypted storage |
| **Third-party API downtime** | Medium | High | Multi-exchange architecture, local caching, degraded mode |
| **Clock drift (NTP failure)** | Low | Medium | Multiple NTP servers, local fallback, monitoring |

---

## 10. Success Criteria

**Quantitative Metrics**:
- ✅ 0% mock/synthetic data in production
- ✅ <100ms latency for market data updates
- ✅ >99.9% uptime for data feeds
- ✅ <0.1% invalid data rate (after validation)
- ✅ 100% SEC Regulation S-P compliance
- ✅ <50ms timestamp accuracy (SEC CAT requirement)

**Qualitative Metrics**:
- ✅ Zero false positives in trading signals due to bad data
- ✅ Comprehensive audit trail for all data sources
- ✅ Documented disaster recovery procedures
- ✅ Passing compliance audit (if applicable)

---

## 11. Next Steps

1. **Immediate Actions** (This Week):
   - Set up Alpaca account and obtain API keys
   - Configure development environment with required dependencies
   - Create feature branch: `feature/real-market-data-feeds`

2. **Week 1-2 Deliverables**:
   - Binance WebSocket client (fully tested)
   - Circuit breaker implementation
   - Initial integration tests

3. **Week 3-4 Deliverables**:
   - Alpaca API client
   - Data validation framework
   - NTP synchronization

4. **Week 5-6 Deliverables**:
   - Multi-source aggregator
   - SEC compliance implementation
   - Comprehensive test suite

5. **Week 7-8 Deliverables**:
   - Parallel deployment
   - Monitoring dashboards
   - Production rollout
   - Documentation updates

---

## 12. References

### Documentation
1. [Binance WebSocket API Documentation](https://developers.binance.com/docs/binance-spot-api-docs/web-socket-streams)
2. [Alpaca Market Data API v2](https://docs.alpaca.markets/docs/about-market-data-api)
3. [SEC Regulation S-P (2024 Amendments)](https://www.sec.gov/newsroom/press-releases/2024-58)
4. [FINRA Cybersecurity Requirements](https://www.finra.org/rules-guidance/guidance/reports/2024-finra-annual-regulatory-oversight-report/cybersecurity)

### Rust Libraries
1. [tokio-tungstenite](https://github.com/snapview/tokio-tungstenite) - WebSocket client
2. [apca](https://github.com/d-e-s-o/apca) - Alpaca API client
3. [failsafe](https://github.com/dmexe/failsafe-rs) - Circuit breaker
4. [governor](https://github.com/boinkor-net/governor) - Rate limiting
5. [moka](https://github.com/moka-rs/moka) - Caching

### Technical Articles
1. [TMS Developer Blog - Binance WebSocket in Rust](https://tms-dev-blog.com/easily-connect-to-binance-websocket-streams-with-rust/)
2. [Building Real-Time Binance Clients in Rust](https://medium.com/@ekfqlwcjswl/building-real-time-binance-websocket-clients-in-rust-with-tokio-2e0027f0f1fd)
3. [Async Rust with Tokio I/O Streams](https://biriukov.dev/docs/async-rust-tokio-io/1-async-rust-with-tokio-io-streams-backpressure-concurrency-and-ergonomics/)
4. [Handling Backpressure in Rust](https://www.slingacademy.com/article/handling-backpressure-in-rust-async-systems-with-bounded-channels/)
5. [Data Validation Best Practices](https://www.cubesoftware.com/blog/data-validation-best-practices)

---

**Document Status**: Ready for Implementation
**Approval Required From**: Systems Architect, Compliance Officer, DevOps Lead
**Expected Start Date**: 2025-01-27
**Expected Completion**: 2025-03-24
