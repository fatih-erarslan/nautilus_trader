# Exchange Adapters and Data Pipeline

## Document Purpose

This document defines the **exchange adapter architecture and market data pipeline** for the Neural Rust port. It provides abstract traits, specific implementations (Alpaca, Polygon, Yahoo), rate limiting, failover strategies, and record/replay for backtesting.

## Table of Contents

1. [Adapter Architecture](#adapter-architecture)
2. [Abstract Adapter Trait](#abstract-adapter-trait)
3. [Alpaca Integration](#alpaca-integration)
4. [Multi-Exchange Aggregation](#multi-exchange-aggregation)
5. [Rate Limiting & Retry](#rate-limiting--retry)
6. [Clock Synchronization](#clock-synchronization)
7. [Record & Replay](#record--replay)
8. [Mock Adapter](#mock-adapter)
9. [Order Routing](#order-routing)
10. [Fill Reconciliation](#fill-reconciliation)
11. [Market Data Normalization](#market-data-normalization)
12. [Troubleshooting](#troubleshooting)

---

## Adapter Architecture

### Design Principles

1. **Abstraction:** All exchange-specific code behind trait
2. **Composability:** Mix and match data providers
3. **Fault Tolerance:** Automatic failover on provider failure
4. **Performance:** Zero-copy where possible, async I/O
5. **Testability:** Mock adapters for testing

### Module Structure

```
crates/
├── core/
│   └── src/
│       └── traits.rs          # Adapter traits
├── exchanges/
│   ├── src/
│   │   ├── lib.rs
│   │   ├── adapter.rs         # Trait implementations
│   │   ├── alpaca/
│   │   │   ├── mod.rs
│   │   │   ├── rest.rs        # REST client
│   │   │   ├── websocket.rs   # WebSocket streaming
│   │   │   └── types.rs       # Alpaca-specific types
│   │   ├── polygon/
│   │   │   ├── mod.rs
│   │   │   ├── rest.rs
│   │   │   └── websocket.rs
│   │   ├── yahoo/
│   │   │   └── rest.rs        # Yahoo Finance (REST only)
│   │   ├── aggregator.rs      # Multi-provider aggregation
│   │   └── mock.rs            # Mock for testing
│   └── tests/
│       ├── alpaca_integration.rs
│       └── aggregator_test.rs
└── market-data/
    └── src/
        ├── pipeline.rs        # Data ingestion pipeline
        ├── normalization.rs   # Format normalization
        └── buffer.rs          # Buffering and batching
```

---

## Abstract Adapter Trait

### Core Trait Definition

```rust
// crates/core/src/traits.rs
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use std::error::Error;

/// Abstract market data provider
#[async_trait]
pub trait MarketDataProvider: Send + Sync {
    /// Get current quote for symbol
    async fn get_quote(&self, symbol: &str) -> Result<Quote, MarketDataError>;

    /// Get historical bars
    async fn get_bars(
        &self,
        symbol: &str,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
        timeframe: Timeframe,
    ) -> Result<Vec<Bar>, MarketDataError>;

    /// Subscribe to real-time market data stream
    async fn subscribe_quotes(
        &self,
        symbols: Vec<String>,
    ) -> Result<QuoteStream, MarketDataError>;

    /// Subscribe to trade updates
    async fn subscribe_trades(
        &self,
        symbols: Vec<String>,
    ) -> Result<TradeStream, MarketDataError>;

    /// Health check
    async fn health_check(&self) -> Result<HealthStatus, MarketDataError>;
}

/// Abstract broker/execution provider
#[async_trait]
pub trait BrokerClient: Send + Sync {
    /// Get account information
    async fn get_account(&self) -> Result<Account, BrokerError>;

    /// Get current positions
    async fn get_positions(&self) -> Result<Vec<Position>, BrokerError>;

    /// Place order
    async fn place_order(&self, order: OrderRequest) -> Result<Order, BrokerError>;

    /// Cancel order
    async fn cancel_order(&self, order_id: &str) -> Result<(), BrokerError>;

    /// Get order status
    async fn get_order(&self, order_id: &str) -> Result<Order, BrokerError>;

    /// List all orders
    async fn list_orders(&self, filter: OrderFilter) -> Result<Vec<Order>, BrokerError>;

    /// Subscribe to order updates
    async fn subscribe_order_updates(&self) -> Result<OrderUpdateStream, BrokerError>;
}

/// Data types
#[derive(Debug, Clone)]
pub struct Quote {
    pub symbol: String,
    pub timestamp: DateTime<Utc>,
    pub bid: Decimal,
    pub ask: Decimal,
    pub bid_size: u64,
    pub ask_size: u64,
}

#[derive(Debug, Clone)]
pub struct Bar {
    pub symbol: String,
    pub timestamp: DateTime<Utc>,
    pub open: Decimal,
    pub high: Decimal,
    pub low: Decimal,
    pub close: Decimal,
    pub volume: u64,
}

#[derive(Debug, Clone)]
pub struct Trade {
    pub symbol: String,
    pub timestamp: DateTime<Utc>,
    pub price: Decimal,
    pub size: u64,
    pub conditions: Vec<String>,
}

#[derive(Debug, Clone, Copy)]
pub enum Timeframe {
    Minute1,
    Minute5,
    Minute15,
    Hour1,
    Day1,
}

/// Streams (async iterators)
pub type QuoteStream = Pin<Box<dyn Stream<Item = Result<Quote, MarketDataError>> + Send>>;
pub type TradeStream = Pin<Box<dyn Stream<Item = Result<Trade, MarketDataError>> + Send>>;
pub type OrderUpdateStream = Pin<Box<dyn Stream<Item = Result<OrderUpdate, BrokerError>> + Send>>;

/// Errors
#[derive(Debug, thiserror::Error)]
pub enum MarketDataError {
    #[error("Network error: {0}")]
    Network(String),

    #[error("Authentication failed: {0}")]
    Auth(String),

    #[error("Rate limit exceeded")]
    RateLimit,

    #[error("Symbol not found: {0}")]
    SymbolNotFound(String),

    #[error("Provider unavailable: {0}")]
    ProviderUnavailable(String),

    #[error("Parse error: {0}")]
    Parse(String),
}

#[derive(Debug, thiserror::Error)]
pub enum BrokerError {
    #[error("Insufficient funds")]
    InsufficientFunds,

    #[error("Invalid order: {0}")]
    InvalidOrder(String),

    #[error("Order not found: {0}")]
    OrderNotFound(String),

    #[error("Market closed")]
    MarketClosed,

    #[error("Rate limit exceeded")]
    RateLimit,

    #[error("Network error: {0}")]
    Network(String),
}
```

---

## Alpaca Integration

### Alpaca REST Client

```rust
// crates/exchanges/src/alpaca/rest.rs
use reqwest::{Client, StatusCode};
use serde::{Deserialize, Serialize};

pub struct AlpacaRestClient {
    client: Client,
    base_url: String,
    api_key: String,
    secret_key: String,
}

impl AlpacaRestClient {
    pub fn new(api_key: String, secret_key: String, paper: bool) -> Self {
        let base_url = if paper {
            "https://paper-api.alpaca.markets".to_string()
        } else {
            "https://api.alpaca.markets".to_string()
        };

        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .unwrap();

        Self {
            client,
            base_url,
            api_key,
            secret_key,
        }
    }

    async fn request<T: DeserializeOwned>(
        &self,
        method: Method,
        path: &str,
        body: Option<impl Serialize>,
    ) -> Result<T, MarketDataError> {
        let url = format!("{}{}", self.base_url, path);

        let mut req = self.client.request(method, &url)
            .header("APCA-API-KEY-ID", &self.api_key)
            .header("APCA-API-SECRET-KEY", &self.secret_key);

        if let Some(body) = body {
            req = req.json(&body);
        }

        let response = req.send().await
            .map_err(|e| MarketDataError::Network(e.to_string()))?;

        match response.status() {
            StatusCode::OK => {
                response.json().await
                    .map_err(|e| MarketDataError::Parse(e.to_string()))
            }
            StatusCode::UNAUTHORIZED => {
                Err(MarketDataError::Auth("Invalid API keys".to_string()))
            }
            StatusCode::TOO_MANY_REQUESTS => {
                Err(MarketDataError::RateLimit)
            }
            StatusCode::NOT_FOUND => {
                Err(MarketDataError::SymbolNotFound("Unknown".to_string()))
            }
            _ => {
                let error_text = response.text().await.unwrap_or_default();
                Err(MarketDataError::Network(error_text))
            }
        }
    }
}

#[async_trait]
impl MarketDataProvider for AlpacaRestClient {
    async fn get_quote(&self, symbol: &str) -> Result<Quote, MarketDataError> {
        #[derive(Deserialize)]
        struct AlpacaQuote {
            #[serde(rename = "t")]
            timestamp: String,
            #[serde(rename = "bp")]
            bid_price: f64,
            #[serde(rename = "ap")]
            ask_price: f64,
            #[serde(rename = "bs")]
            bid_size: u64,
            #[serde(rename = "as")]
            ask_size: u64,
        }

        let path = format!("/v2/stocks/{}/quotes/latest", symbol);
        let response: AlpacaQuote = self.request(Method::GET, &path, None::<()>).await?;

        Ok(Quote {
            symbol: symbol.to_string(),
            timestamp: DateTime::parse_from_rfc3339(&response.timestamp)
                .unwrap()
                .with_timezone(&Utc),
            bid: Decimal::from_f64_retain(response.bid_price).unwrap(),
            ask: Decimal::from_f64_retain(response.ask_price).unwrap(),
            bid_size: response.bid_size,
            ask_size: response.ask_size,
        })
    }

    async fn get_bars(
        &self,
        symbol: &str,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
        timeframe: Timeframe,
    ) -> Result<Vec<Bar>, MarketDataError> {
        let timeframe_str = match timeframe {
            Timeframe::Minute1 => "1Min",
            Timeframe::Minute5 => "5Min",
            Timeframe::Minute15 => "15Min",
            Timeframe::Hour1 => "1Hour",
            Timeframe::Day1 => "1Day",
        };

        let path = format!(
            "/v2/stocks/{}/bars?start={}&end={}&timeframe={}",
            symbol,
            start.to_rfc3339(),
            end.to_rfc3339(),
            timeframe_str
        );

        #[derive(Deserialize)]
        struct AlpacaBarsResponse {
            bars: Vec<AlpacaBar>,
        }

        #[derive(Deserialize)]
        struct AlpacaBar {
            t: String,
            o: f64,
            h: f64,
            l: f64,
            c: f64,
            v: u64,
        }

        let response: AlpacaBarsResponse = self.request(Method::GET, &path, None::<()>).await?;

        Ok(response.bars.into_iter().map(|bar| Bar {
            symbol: symbol.to_string(),
            timestamp: DateTime::parse_from_rfc3339(&bar.t)
                .unwrap()
                .with_timezone(&Utc),
            open: Decimal::from_f64_retain(bar.o).unwrap(),
            high: Decimal::from_f64_retain(bar.h).unwrap(),
            low: Decimal::from_f64_retain(bar.l).unwrap(),
            close: Decimal::from_f64_retain(bar.c).unwrap(),
            volume: bar.v,
        }).collect())
    }

    async fn subscribe_quotes(&self, symbols: Vec<String>) -> Result<QuoteStream, MarketDataError> {
        // WebSocket implementation (see below)
        todo!("Use AlpacaWebSocketClient")
    }

    async fn health_check(&self) -> Result<HealthStatus, MarketDataError> {
        let path = "/v2/clock";
        let _: serde_json::Value = self.request(Method::GET, path, None::<()>).await?;

        Ok(HealthStatus::Healthy)
    }
}
```

### Alpaca WebSocket Client

```rust
// crates/exchanges/src/alpaca/websocket.rs
use tokio_tungstenite::{connect_async, tungstenite::Message};
use futures::{SinkExt, StreamExt};

pub struct AlpacaWebSocketClient {
    api_key: String,
    secret_key: String,
    url: String,
}

impl AlpacaWebSocketClient {
    pub fn new(api_key: String, secret_key: String, paper: bool) -> Self {
        let url = if paper {
            "wss://stream.data.alpaca.markets/v2/iex"
        } else {
            "wss://stream.data.alpaca.markets/v2/sip"
        }.to_string();

        Self {
            api_key,
            secret_key,
            url,
        }
    }

    pub async fn connect(&self) -> Result<QuoteStream, MarketDataError> {
        let (ws_stream, _) = connect_async(&self.url).await
            .map_err(|e| MarketDataError::Network(e.to_string()))?;

        let (mut write, mut read) = ws_stream.split();

        // Authenticate
        let auth_msg = json!({
            "action": "auth",
            "key": self.api_key,
            "secret": self.secret_key,
        });

        write.send(Message::Text(auth_msg.to_string())).await
            .map_err(|e| MarketDataError::Network(e.to_string()))?;

        // Wait for auth confirmation
        if let Some(Ok(Message::Text(msg))) = read.next().await {
            let response: serde_json::Value = serde_json::from_str(&msg).unwrap();
            if response["msg"] != "authenticated" {
                return Err(MarketDataError::Auth("WebSocket auth failed".to_string()));
            }
        }

        // Subscribe to quotes
        let subscribe_msg = json!({
            "action": "subscribe",
            "quotes": ["*"],  // All symbols
        });

        write.send(Message::Text(subscribe_msg.to_string())).await
            .map_err(|e| MarketDataError::Network(e.to_string()))?;

        // Return stream
        let stream = read.filter_map(|msg| async move {
            match msg {
                Ok(Message::Text(text)) => {
                    Self::parse_quote(&text).ok()
                }
                _ => None,
            }
        });

        Ok(Box::pin(stream))
    }

    fn parse_quote(text: &str) -> Result<Quote, MarketDataError> {
        #[derive(Deserialize)]
        struct WsQuote {
            #[serde(rename = "S")]
            symbol: String,
            #[serde(rename = "t")]
            timestamp: String,
            #[serde(rename = "bp")]
            bid_price: f64,
            #[serde(rename = "ap")]
            ask_price: f64,
            #[serde(rename = "bs")]
            bid_size: u64,
            #[serde(rename = "as")]
            ask_size: u64,
        }

        let quote: WsQuote = serde_json::from_str(text)
            .map_err(|e| MarketDataError::Parse(e.to_string()))?;

        Ok(Quote {
            symbol: quote.symbol,
            timestamp: DateTime::parse_from_rfc3339(&quote.timestamp)
                .unwrap()
                .with_timezone(&Utc),
            bid: Decimal::from_f64_retain(quote.bid_price).unwrap(),
            ask: Decimal::from_f64_retain(quote.ask_price).unwrap(),
            bid_size: quote.bid_size,
            ask_size: quote.ask_size,
        })
    }
}
```

### Alpaca Order Execution

```rust
// crates/exchanges/src/alpaca/execution.rs
#[async_trait]
impl BrokerClient for AlpacaRestClient {
    async fn place_order(&self, order: OrderRequest) -> Result<Order, BrokerError> {
        #[derive(Serialize)]
        struct AlpacaOrderRequest {
            symbol: String,
            qty: String,
            side: String,
            #[serde(rename = "type")]
            order_type: String,
            time_in_force: String,
        }

        let alpaca_order = AlpacaOrderRequest {
            symbol: order.symbol.clone(),
            qty: order.qty.to_string(),
            side: match order.side {
                OrderSide::Buy => "buy".to_string(),
                OrderSide::Sell => "sell".to_string(),
            },
            order_type: "market".to_string(),
            time_in_force: "day".to_string(),
        };

        #[derive(Deserialize)]
        struct AlpacaOrderResponse {
            id: String,
            symbol: String,
            qty: String,
            side: String,
            status: String,
            filled_qty: String,
            filled_avg_price: Option<String>,
        }

        let response: AlpacaOrderResponse = self.request(
            Method::POST,
            "/v2/orders",
            Some(alpaca_order),
        ).await.map_err(|e| match e {
            MarketDataError::RateLimit => BrokerError::RateLimit,
            MarketDataError::Network(msg) if msg.contains("insufficient") => {
                BrokerError::InsufficientFunds
            }
            MarketDataError::Network(msg) => BrokerError::Network(msg),
            _ => BrokerError::InvalidOrder(e.to_string()),
        })?;

        Ok(Order {
            id: response.id,
            symbol: response.symbol,
            qty: response.qty.parse().unwrap(),
            side: match response.side.as_str() {
                "buy" => OrderSide::Buy,
                "sell" => OrderSide::Sell,
                _ => unreachable!(),
            },
            status: match response.status.as_str() {
                "new" => OrderStatus::New,
                "filled" => OrderStatus::Filled,
                "partially_filled" => OrderStatus::PartiallyFilled,
                "canceled" => OrderStatus::Canceled,
                _ => OrderStatus::New,
            },
            filled_qty: response.filled_qty.parse().unwrap_or(0),
            filled_avg_price: response.filled_avg_price
                .and_then(|p| Decimal::from_str(&p).ok()),
        })
    }

    async fn cancel_order(&self, order_id: &str) -> Result<(), BrokerError> {
        let path = format!("/v2/orders/{}", order_id);
        let _: serde_json::Value = self.request(Method::DELETE, &path, None::<()>).await
            .map_err(|e| BrokerError::Network(e.to_string()))?;

        Ok(())
    }

    async fn get_positions(&self) -> Result<Vec<Position>, BrokerError> {
        #[derive(Deserialize)]
        struct AlpacaPosition {
            symbol: String,
            qty: String,
            avg_entry_price: String,
            market_value: String,
            unrealized_pl: String,
        }

        let positions: Vec<AlpacaPosition> = self.request(
            Method::GET,
            "/v2/positions",
            None::<()>,
        ).await.map_err(|e| BrokerError::Network(e.to_string()))?;

        Ok(positions.into_iter().map(|pos| Position {
            symbol: pos.symbol,
            qty: pos.qty.parse().unwrap(),
            avg_entry_price: Decimal::from_str(&pos.avg_entry_price).unwrap(),
            market_value: Decimal::from_str(&pos.market_value).unwrap(),
            unrealized_pl: Decimal::from_str(&pos.unrealized_pl).unwrap(),
        }).collect())
    }
}
```

---

## Multi-Exchange Aggregation

### Aggregator Implementation

```rust
// crates/exchanges/src/aggregator.rs
pub struct MarketDataAggregator {
    primary: Box<dyn MarketDataProvider>,
    fallbacks: Vec<Box<dyn MarketDataProvider>>,
    health_checker: Arc<HealthChecker>,
}

impl MarketDataAggregator {
    pub fn new(primary: Box<dyn MarketDataProvider>) -> Self {
        Self {
            primary,
            fallbacks: Vec::new(),
            health_checker: Arc::new(HealthChecker::new()),
        }
    }

    pub fn add_fallback(mut self, provider: Box<dyn MarketDataProvider>) -> Self {
        self.fallbacks.push(provider);
        self
    }
}

#[async_trait]
impl MarketDataProvider for MarketDataAggregator {
    async fn get_quote(&self, symbol: &str) -> Result<Quote, MarketDataError> {
        // Try primary first
        if self.health_checker.is_healthy("primary").await {
            match self.primary.get_quote(symbol).await {
                Ok(quote) => return Ok(quote),
                Err(e) => {
                    tracing::warn!("Primary provider failed: {}, trying fallbacks", e);
                    self.health_checker.mark_unhealthy("primary").await;
                }
            }
        }

        // Try fallbacks in order
        for (i, provider) in self.fallbacks.iter().enumerate() {
            let name = format!("fallback-{}", i);

            if !self.health_checker.is_healthy(&name).await {
                continue;
            }

            match provider.get_quote(symbol).await {
                Ok(quote) => {
                    tracing::info!("Using fallback provider {}", i);
                    return Ok(quote);
                }
                Err(e) => {
                    tracing::warn!("Fallback {} failed: {}", i, e);
                    self.health_checker.mark_unhealthy(&name).await;
                }
            }
        }

        Err(MarketDataError::ProviderUnavailable(
            "All providers failed".to_string()
        ))
    }

    async fn get_bars(
        &self,
        symbol: &str,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
        timeframe: Timeframe,
    ) -> Result<Vec<Bar>, MarketDataError> {
        // Similar fallback logic
        todo!()
    }

    async fn health_check(&self) -> Result<HealthStatus, MarketDataError> {
        // Check all providers
        let primary_healthy = self.primary.health_check().await.is_ok();

        let fallback_healthy = futures::future::join_all(
            self.fallbacks.iter().map(|p| p.health_check())
        ).await;

        if primary_healthy || fallback_healthy.iter().any(|r| r.is_ok()) {
            Ok(HealthStatus::Healthy)
        } else {
            Ok(HealthStatus::Degraded)
        }
    }
}

/// Health checker with circuit breaker pattern
pub struct HealthChecker {
    states: Arc<RwLock<HashMap<String, HealthState>>>,
}

struct HealthState {
    is_healthy: bool,
    failure_count: u32,
    last_check: Instant,
}

impl HealthChecker {
    pub fn new() -> Self {
        Self {
            states: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn is_healthy(&self, provider: &str) -> bool {
        let states = self.states.read().await;

        states.get(provider)
            .map(|state| {
                // Auto-recover after 30 seconds
                if !state.is_healthy && state.last_check.elapsed() > Duration::from_secs(30) {
                    return true;
                }
                state.is_healthy
            })
            .unwrap_or(true)  // Assume healthy if not tracked
    }

    pub async fn mark_unhealthy(&self, provider: &str) {
        let mut states = self.states.write().await;

        states.entry(provider.to_string())
            .and_modify(|state| {
                state.is_healthy = false;
                state.failure_count += 1;
                state.last_check = Instant::now();
            })
            .or_insert(HealthState {
                is_healthy: false,
                failure_count: 1,
                last_check: Instant::now(),
            });
    }
}
```

---

## Rate Limiting & Retry

### Rate Limiter

```rust
// crates/exchanges/src/rate_limiter.rs
use governor::{Quota, RateLimiter, clock::DefaultClock, state::direct::NotKeyed};
use std::num::NonZeroU32;

pub struct ApiRateLimiter {
    limiter: RateLimiter<NotKeyed, InMemoryState, DefaultClock>,
}

impl ApiRateLimiter {
    pub fn new(requests_per_minute: u32) -> Self {
        let quota = Quota::per_minute(NonZeroU32::new(requests_per_minute).unwrap());

        Self {
            limiter: RateLimiter::direct(quota),
        }
    }

    pub async fn acquire(&self) -> Result<(), RateLimitError> {
        self.limiter.until_ready().await;
        Ok(())
    }
}

/// Retry with exponential backoff
pub async fn retry_with_backoff<F, T, E>(
    mut f: F,
    max_attempts: u32,
) -> Result<T, E>
where
    F: FnMut() -> std::pin::Pin<Box<dyn Future<Output = Result<T, E>> + Send>>,
    E: std::fmt::Display,
{
    let mut attempt = 0;
    let mut delay = Duration::from_millis(100);

    loop {
        attempt += 1;

        match f().await {
            Ok(result) => return Ok(result),
            Err(e) if attempt >= max_attempts => {
                tracing::error!("All {} retry attempts failed: {}", max_attempts, e);
                return Err(e);
            }
            Err(e) => {
                tracing::warn!("Attempt {} failed: {}, retrying in {:?}", attempt, e, delay);
                tokio::time::sleep(delay).await;
                delay *= 2;  // Exponential backoff
            }
        }
    }
}

// Usage
async fn fetch_quote_with_retry(symbol: &str) -> Result<Quote, MarketDataError> {
    retry_with_backoff(
        || Box::pin(client.get_quote(symbol)),
        3  // Max 3 attempts
    ).await
}
```

---

## Clock Synchronization

### Backtest Clock

```rust
// crates/backtesting/src/clock.rs
use chrono::{DateTime, Utc, Duration};

pub trait Clock: Send + Sync {
    fn now(&self) -> DateTime<Utc>;
    fn sleep(&self, duration: Duration) -> BoxFuture<'_, ()>;
}

/// Real-time clock for live trading
pub struct RealClock;

impl Clock for RealClock {
    fn now(&self) -> DateTime<Utc> {
        Utc::now()
    }

    fn sleep(&self, duration: Duration) -> BoxFuture<'_, ()> {
        Box::pin(async move {
            tokio::time::sleep(duration.to_std().unwrap()).await;
        })
    }
}

/// Simulated clock for backtesting
pub struct SimulatedClock {
    current_time: Arc<RwLock<DateTime<Utc>>>,
}

impl SimulatedClock {
    pub fn new(start_time: DateTime<Utc>) -> Self {
        Self {
            current_time: Arc::new(RwLock::new(start_time)),
        }
    }

    pub async fn advance(&self, duration: Duration) {
        let mut time = self.current_time.write().await;
        *time = *time + duration;
    }
}

impl Clock for SimulatedClock {
    fn now(&self) -> DateTime<Utc> {
        *self.current_time.blocking_read()
    }

    fn sleep(&self, _duration: Duration) -> BoxFuture<'_, ()> {
        // No-op in simulation (instant)
        Box::pin(async move {})
    }
}
```

---

## Record & Replay

### Market Data Recorder

```rust
// crates/market-data/src/recorder.rs
pub struct MarketDataRecorder {
    file: Arc<Mutex<File>>,
}

impl MarketDataRecorder {
    pub fn new(path: &Path) -> Result<Self, std::io::Error> {
        let file = File::create(path)?;

        Ok(Self {
            file: Arc::new(Mutex::new(file)),
        })
    }

    pub async fn record_quote(&self, quote: &Quote) -> Result<(), std::io::Error> {
        let mut file = self.file.lock().await;

        // Write as CSV
        writeln!(
            file,
            "{},{},{},{},{},{}",
            quote.timestamp.to_rfc3339(),
            quote.symbol,
            quote.bid,
            quote.ask,
            quote.bid_size,
            quote.ask_size
        )?;

        Ok(())
    }
}

/// Wrap provider to record all data
pub struct RecordingAdapter<T: MarketDataProvider> {
    inner: T,
    recorder: MarketDataRecorder,
}

#[async_trait]
impl<T: MarketDataProvider> MarketDataProvider for RecordingAdapter<T> {
    async fn get_quote(&self, symbol: &str) -> Result<Quote, MarketDataError> {
        let quote = self.inner.get_quote(symbol).await?;

        // Record to file
        self.recorder.record_quote(&quote).await
            .map_err(|e| MarketDataError::Network(e.to_string()))?;

        Ok(quote)
    }

    // ... other methods
}
```

### Market Data Replayer

```rust
// crates/backtesting/src/replayer.rs
pub struct MarketDataReplayer {
    events: Vec<(DateTime<Utc>, Quote)>,
    current_index: AtomicUsize,
}

impl MarketDataReplayer {
    pub fn from_csv(path: &Path) -> Result<Self, std::io::Error> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);

        let mut events = Vec::new();

        for line in reader.lines() {
            let line = line?;
            let parts: Vec<&str> = line.split(',').collect();

            let quote = Quote {
                timestamp: DateTime::parse_from_rfc3339(parts[0])
                    .unwrap()
                    .with_timezone(&Utc),
                symbol: parts[1].to_string(),
                bid: Decimal::from_str(parts[2]).unwrap(),
                ask: Decimal::from_str(parts[3]).unwrap(),
                bid_size: parts[4].parse().unwrap(),
                ask_size: parts[5].parse().unwrap(),
            };

            events.push((quote.timestamp, quote));
        }

        // Sort by timestamp
        events.sort_by_key(|(ts, _)| *ts);

        Ok(Self {
            events,
            current_index: AtomicUsize::new(0),
        })
    }
}

#[async_trait]
impl MarketDataProvider for MarketDataReplayer {
    async fn get_quote(&self, symbol: &str) -> Result<Quote, MarketDataError> {
        let index = self.current_index.fetch_add(1, Ordering::SeqCst);

        self.events.get(index)
            .filter(|(_, quote)| quote.symbol == symbol)
            .map(|(_, quote)| quote.clone())
            .ok_or(MarketDataError::SymbolNotFound(symbol.to_string()))
    }

    // Replay subscribes by returning all events as stream
    async fn subscribe_quotes(&self, symbols: Vec<String>) -> Result<QuoteStream, MarketDataError> {
        let events = self.events.clone();
        let symbols_set: HashSet<_> = symbols.into_iter().collect();

        let stream = futures::stream::iter(events)
            .filter(move |(_, quote)| {
                let symbols = symbols_set.clone();
                async move { symbols.contains(&quote.symbol) }
            })
            .map(|(_, quote)| Ok(quote));

        Ok(Box::pin(stream))
    }
}
```

---

## Mock Adapter

### Mock for Testing

```rust
// crates/exchanges/src/mock.rs
pub struct MockMarketDataProvider {
    quotes: Arc<Mutex<HashMap<String, Quote>>>,
}

impl MockMarketDataProvider {
    pub fn new() -> Self {
        Self {
            quotes: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn set_quote(&self, symbol: &str, quote: Quote) {
        self.quotes.blocking_lock().insert(symbol.to_string(), quote);
    }
}

#[async_trait]
impl MarketDataProvider for MockMarketDataProvider {
    async fn get_quote(&self, symbol: &str) -> Result<Quote, MarketDataError> {
        self.quotes.lock().await
            .get(symbol)
            .cloned()
            .ok_or(MarketDataError::SymbolNotFound(symbol.to_string()))
    }

    async fn get_bars(
        &self,
        _symbol: &str,
        _start: DateTime<Utc>,
        _end: DateTime<Utc>,
        _timeframe: Timeframe,
    ) -> Result<Vec<Bar>, MarketDataError> {
        // Return empty for simplicity
        Ok(Vec::new())
    }

    async fn subscribe_quotes(&self, _symbols: Vec<String>) -> Result<QuoteStream, MarketDataError> {
        // Return empty stream
        let stream = futures::stream::empty();
        Ok(Box::pin(stream))
    }

    async fn health_check(&self) -> Result<HealthStatus, MarketDataError> {
        Ok(HealthStatus::Healthy)
    }
}
```

---

## Order Routing

### Smart Order Router

```rust
// crates/execution/src/router.rs
pub struct OrderRouter {
    brokers: Vec<Box<dyn BrokerClient>>,
    selector: RoutingStrategy,
}

pub enum RoutingStrategy {
    RoundRobin,
    LowestFee,
    FastestExecution,
}

impl OrderRouter {
    pub async fn route_order(&self, order: OrderRequest) -> Result<Order, BrokerError> {
        let broker_index = match self.selector {
            RoutingStrategy::RoundRobin => {
                // Simple round-robin
                rand::random::<usize>() % self.brokers.len()
            }
            RoutingStrategy::LowestFee => {
                // TODO: Compare fee structures
                0
            }
            RoutingStrategy::FastestExecution => {
                // TODO: Track latency statistics
                0
            }
        };

        let broker = &self.brokers[broker_index];
        broker.place_order(order).await
    }
}
```

---

## Fill Reconciliation

### Trade Reconciliation

```rust
// crates/execution/src/reconciliation.rs
pub struct FillReconciler {
    expected_fills: Arc<Mutex<HashMap<String, Order>>>,
}

impl FillReconciler {
    pub async fn reconcile(&self, order_id: &str, broker: &dyn BrokerClient) -> Result<(), BrokerError> {
        let expected = self.expected_fills.lock().await
            .get(order_id)
            .cloned();

        if let Some(expected_order) = expected {
            let actual_order = broker.get_order(order_id).await?;

            // Check if fill matches expectation
            if actual_order.status == OrderStatus::Filled {
                if actual_order.filled_qty != expected_order.qty {
                    tracing::warn!(
                        "Fill quantity mismatch: expected {}, got {}",
                        expected_order.qty,
                        actual_order.filled_qty
                    );
                }

                // Check fill price is reasonable
                if let Some(filled_price) = actual_order.filled_avg_price {
                    let market_price = get_current_market_price(&actual_order.symbol).await?;
                    let price_diff = (filled_price - market_price).abs();

                    if price_diff > market_price * Decimal::new(1, 2) {  // 1% tolerance
                        tracing::error!(
                            "Fill price {} deviates significantly from market {}",
                            filled_price,
                            market_price
                        );
                    }
                }
            }
        }

        Ok(())
    }
}
```

---

## Market Data Normalization

### Data Normalizer

```rust
// crates/market-data/src/normalization.rs
pub struct DataNormalizer;

impl DataNormalizer {
    /// Normalize different quote formats to common format
    pub fn normalize_quote(quote: RawQuote, source: DataSource) -> Quote {
        match source {
            DataSource::Alpaca => {
                // Alpaca uses cents, convert to dollars
                Quote {
                    symbol: quote.symbol,
                    timestamp: quote.timestamp,
                    bid: quote.bid / Decimal::new(100, 0),
                    ask: quote.ask / Decimal::new(100, 0),
                    bid_size: quote.bid_size,
                    ask_size: quote.ask_size,
                }
            }
            DataSource::Polygon => {
                // Polygon already in dollars
                quote.into()
            }
            DataSource::Yahoo => {
                // Yahoo delayed by 15 minutes
                Quote {
                    timestamp: quote.timestamp - Duration::minutes(15),
                    ..quote.into()
                }
            }
        }
    }

    /// Handle missing data
    pub fn fill_gaps(bars: &mut Vec<Bar>) {
        bars.sort_by_key(|bar| bar.timestamp);

        let mut i = 0;
        while i < bars.len() - 1 {
            let current = &bars[i];
            let next = &bars[i + 1];

            let expected_next = current.timestamp + Duration::minutes(1);

            if next.timestamp > expected_next {
                // Gap detected, fill with forward-fill
                let fill_bar = Bar {
                    timestamp: expected_next,
                    open: current.close,
                    high: current.close,
                    low: current.close,
                    close: current.close,
                    volume: 0,
                    ..current.clone()
                };

                bars.insert(i + 1, fill_bar);
            }

            i += 1;
        }
    }
}
```

---

## Troubleshooting

### Common Issues

#### 1. WebSocket Disconnects

**Symptom:** WebSocket closes unexpectedly

**Solution:**
```rust
// Implement auto-reconnect
pub async fn maintain_connection<F>(mut connect_fn: F)
where
    F: FnMut() -> BoxFuture<'static, Result<QuoteStream, MarketDataError>>,
{
    loop {
        match connect_fn().await {
            Ok(mut stream) => {
                while let Some(quote) = stream.next().await {
                    // Process quote
                }

                tracing::warn!("WebSocket closed, reconnecting in 5s");
            }
            Err(e) => {
                tracing::error!("Connection failed: {}, retrying in 10s", e);
                tokio::time::sleep(Duration::from_secs(10)).await;
            }
        }

        tokio::time::sleep(Duration::from_secs(5)).await;
    }
}
```

#### 2. Rate Limit Errors

**Symptom:** HTTP 429 errors

**Solution:**
- Implement exponential backoff
- Cache market data aggressively
- Use WebSocket instead of REST polling

#### 3. Stale Data

**Symptom:** Quotes older than expected

**Solution:**
```rust
pub fn is_stale(quote: &Quote, max_age: Duration) -> bool {
    Utc::now() - quote.timestamp > max_age
}

// Reject stale data
if is_stale(&quote, Duration::seconds(30)) {
    return Err(MarketDataError::StaleData);
}
```

---

## Acceptance Criteria

- [ ] Abstract adapter trait defined for market data and broker
- [ ] Alpaca REST and WebSocket clients implemented
- [ ] Multi-provider aggregation with failover
- [ ] Rate limiting and retry logic
- [ ] Record/replay harness for backtesting
- [ ] Mock adapter for testing
- [ ] Order routing and fill reconciliation
- [ ] Data normalization across providers
- [ ] Integration tests with live APIs

---

## Cross-References

- **Architecture:** [03_Architecture.md](./03_Architecture.md) - Module design
- **Testing:** [13_Tests_Benchmarks_CI.md](./13_Tests_Benchmarks_CI.md) - Integration tests
- **Backtesting:** [18_Simulation_Backtesting.md](./18_Simulation_Backtesting.md) - Replay harness
- **Risk:** [14_Risk_and_Fallbacks.md](./14_Risk_and_Fallbacks.md) - Provider failover

---

**Document Version:** 1.0.0
**Last Updated:** 2025-11-12
**Owner:** Backend Developer
**Status:** Complete
**Next Review:** 2025-11-19
