# Neural Trading Rust Port - Comprehensive Architecture

**Version:** 1.0.0
**Date:** 2025-11-12
**Status:** Design Complete
**Cross-References:** [Parity Requirements](02_Parity_Requirements.md) | [Interop](04_Rust_Crates_and_Node_Interop.md) | [Memory](05_Memory_and_AgentDB.md)

---

## Table of Contents

1. [Module Map (16 Crates)](#module-map-16-crates)
2. [Sequence Diagrams](#sequence-diagrams)
3. [Interface Boundaries](#interface-boundaries)
4. [Data Flow Diagrams](#data-flow-diagrams)
5. [Concurrency Model](#concurrency-model)
6. [Error Handling Patterns](#error-handling-patterns)
7. [Integration Points](#integration-points)
8. [Performance Targets Per Module](#performance-targets-per-module)

---

## Module Map (16 Crates)

### Workspace Structure

```
neural-trader-rs/
├── Cargo.toml (workspace root)
├── crates/
│   ├── nt-core/              # 1. Core types and traits
│   ├── nt-config/            # 2. Configuration management
│   ├── nt-market-data/       # 3. Market data ingestion
│   ├── nt-features/          # 4. Feature extraction
│   ├── nt-strategies/        # 5. Strategy implementations
│   ├── nt-signals/           # 6. Signal generation
│   ├── nt-portfolio/         # 7. Portfolio management
│   ├── nt-risk/              # 8. Risk management
│   ├── nt-execution/         # 9. Order execution
│   ├── nt-backtesting/       # 10. Backtesting engine
│   ├── nt-neural/            # 11. Neural forecasting
│   ├── nt-news/              # 12. News collection & sentiment
│   ├── nt-agentdb/           # 13. AgentDB integration
│   ├── nt-api/               # 14. HTTP/REST API
│   ├── nt-napi/              # 15. Node.js bindings
│   └── nt-cli/               # 16. CLI application
└── target/
```

### Crate Responsibility Matrix

| Crate | Purpose | Dependencies | Exports | LOC Est. |
|-------|---------|-------------|---------|----------|
| **nt-core** | Core types, traits, errors | None | Symbol, Decimal, Result<T> | 2,000 |
| **nt-config** | Config loading, validation | nt-core | AppConfig, StrategyConfig | 800 |
| **nt-market-data** | WebSocket, REST API clients | nt-core | MarketDataProvider trait | 3,500 |
| **nt-features** | Technical indicators | nt-core | FeatureExtractor trait | 2,500 |
| **nt-strategies** | Strategy implementations | nt-core, nt-features | Strategy trait | 8,000 |
| **nt-signals** | Signal generation, filtering | nt-core | SignalGenerator | 1,500 |
| **nt-portfolio** | Position tracking, P&L | nt-core | Portfolio | 2,000 |
| **nt-risk** | VaR, CVaR, limits | nt-core, nt-portfolio | RiskManager | 2,500 |
| **nt-execution** | Order routing, fills | nt-core | OrderManager | 3,000 |
| **nt-backtesting** | Historical simulation | All above | Backtester | 3,500 |
| **nt-neural** | NHITS, LSTM, Transformer | nt-core | NeuralForecaster | 4,000 |
| **nt-news** | News APIs, sentiment | nt-core | NewsCollector | 2,500 |
| **nt-agentdb** | Vector DB client | nt-core | AgentDBClient | 1,500 |
| **nt-api** | HTTP server | All above | start_api_server() | 3,000 |
| **nt-napi** | Node.js FFI | All above | napi bindings | 2,500 |
| **nt-cli** | Command-line interface | All above | main() | 1,000 |
| **TOTAL** | | | | **44,300** |

---

### 1. nt-core: Core Types and Traits

**Purpose:** Foundation crate with zero dependencies. Defines all core types used throughout the system.

```rust
// crates/nt-core/src/lib.rs

/// Core types
pub mod types {
    pub use rust_decimal::Decimal;
    pub use chrono::{DateTime, Utc};
    pub use uuid::Uuid;

    pub struct Symbol(String);
    pub struct OrderId(Uuid);

    pub enum Direction { Long, Short, Close }
    pub enum OrderSide { Buy, Sell }
    pub enum OrderType { Market, Limit, StopLoss, StopLimit }
    pub enum TimeInForce { Day, GTC, IOC, FOK }
}

/// Core errors
pub mod errors {
    #[derive(Debug, thiserror::Error)]
    pub enum TradingError {
        #[error("Market data error: {0}")]
        MarketData(String),

        #[error("Strategy error: {0}")]
        Strategy(String),

        #[error("Execution error: {0}")]
        Execution(String),

        #[error("Risk limit exceeded: {0}")]
        RiskLimit(String),
    }

    pub type Result<T> = std::result::Result<T, TradingError>;
}

/// Core traits
pub mod traits {
    use super::*;
    use async_trait::async_trait;

    #[async_trait]
    pub trait MarketDataProvider: Send + Sync {
        async fn subscribe(&self, symbols: &[Symbol]) -> Result<Receiver<Tick>>;
        async fn get_quote(&self, symbol: &Symbol) -> Result<Quote>;
        async fn get_bars(&self, symbol: &Symbol, start: DateTime<Utc>, end: DateTime<Utc>) -> Result<Vec<Bar>>;
    }

    #[async_trait]
    pub trait Strategy: Send + Sync {
        fn id(&self) -> &str;
        async fn on_market_data(&mut self, data: MarketData) -> Result<Vec<Signal>>;
        async fn generate_signals(&mut self) -> Result<Vec<Signal>>;
    }
}
```

**Dependencies:**
```toml
[dependencies]
rust_decimal = "1.33"
chrono = "0.4"
uuid = "1.6"
thiserror = "1.0"
async-trait = "0.1"
serde = { version = "1.0", features = ["derive"] }
```

---

### 2. nt-config: Configuration Management

**Purpose:** Load, validate, and manage configuration from env vars and TOML files.

```rust
// crates/nt-config/src/lib.rs

#[derive(Debug, Deserialize, Validate)]
pub struct AppConfig {
    #[validate(range(min = 1024, max = 65535))]
    pub port: u16,

    pub alpaca: AlpacaConfig,
    pub strategies: Vec<StrategyConfig>,
    pub risk: RiskConfig,
    pub database: DatabaseConfig,
}

impl AppConfig {
    pub fn from_env() -> Result<Self> {
        dotenvy::dotenv().ok();
        let config = envy::from_env::<Self>()?;
        config.validate()?;
        Ok(config)
    }

    pub fn from_file(path: &str) -> Result<Self> {
        let contents = std::fs::read_to_string(path)?;
        let config: Self = toml::from_str(&contents)?;
        config.validate()?;
        Ok(config)
    }
}
```

---

### 3. nt-market-data: Market Data Ingestion

**Purpose:** WebSocket and REST API clients for Alpaca, Polygon, etc.

```rust
// crates/nt-market-data/src/alpaca.rs

pub struct AlpacaMarketData {
    http_client: reqwest::Client,
    ws_client: Arc<Mutex<Option<WebSocketStream>>>,
    config: AlpacaConfig,
}

#[async_trait]
impl MarketDataProvider for AlpacaMarketData {
    async fn subscribe(&self, symbols: &[Symbol]) -> Result<Receiver<Tick>> {
        let (tx, rx) = mpsc::channel(1000);

        // Connect WebSocket
        let mut ws = self.connect_websocket().await?;

        // Send subscribe message
        let subscribe_msg = json!({
            "action": "subscribe",
            "trades": symbols.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
        });
        ws.send(Message::Text(subscribe_msg.to_string())).await?;

        // Spawn listener task
        tokio::spawn(async move {
            while let Some(Ok(msg)) = ws.next().await {
                if let Ok(tick) = parse_tick_message(msg) {
                    tx.send(tick).await.ok();
                }
            }
        });

        Ok(rx)
    }
}
```

**Performance Target:**
- WebSocket latency: <1ms
- Parse throughput: >10,000 ticks/sec
- Memory per connection: <10MB

---

### 4. nt-features: Feature Extraction

**Purpose:** Technical indicators (SMA, EMA, RSI, MACD, etc.)

```rust
// crates/nt-features/src/indicators.rs

pub struct FeatureExtractor {
    config: FeatureConfig,
}

impl FeatureExtractor {
    pub fn extract(&self, bars: &[Bar]) -> Result<DataFrame> {
        let mut df = DataFrame::new(vec![
            Series::new("close", bars.iter().map(|b| b.close).collect::<Vec<_>>()),
            Series::new("volume", bars.iter().map(|b| b.volume as f64).collect::<Vec<_>>()),
        ])?;

        // Calculate indicators
        df = df.with_column(self.calculate_sma(&df, 20)?)?;
        df = df.with_column(self.calculate_ema(&df, 12)?)?;
        df = df.with_column(self.calculate_rsi(&df, 14)?)?;
        df = df.with_column(self.calculate_macd(&df)?)?;

        Ok(df)
    }

    fn calculate_rsi(&self, df: &DataFrame, period: usize) -> Result<Series> {
        let close = df.column("close")?.f64()?;

        // Calculate price changes
        let changes: Vec<f64> = close.into_iter()
            .zip(close.into_iter().skip(1))
            .map(|(prev, curr)| curr - prev)
            .collect();

        // Separate gains and losses
        let gains: Vec<f64> = changes.iter().map(|&c| if c > 0.0 { c } else { 0.0 }).collect();
        let losses: Vec<f64> = changes.iter().map(|&c| if c < 0.0 { -c } else { 0.0 }).collect();

        // Calculate RSI
        let avg_gain = gains.windows(period).map(|w| w.iter().sum::<f64>() / period as f64).collect::<Vec<_>>();
        let avg_loss = losses.windows(period).map(|w| w.iter().sum::<f64>() / period as f64).collect::<Vec<_>>();

        let rsi: Vec<f64> = avg_gain.iter().zip(avg_loss.iter())
            .map(|(&gain, &loss)| {
                if loss == 0.0 { 100.0 }
                else { 100.0 - (100.0 / (1.0 + gain / loss)) }
            })
            .collect();

        Ok(Series::new("rsi", rsi))
    }
}
```

**Performance Target:**
- Feature extraction: <10ms for 100 bars
- Throughput: >5,000 feature vectors/sec

---

## Sequence Diagrams

### 1. Complete Trading Pipeline

```
┌──────────────────────────────────────────────────────────────────┐
│                    Trading Pipeline Flow                          │
└──────────────────────────────────────────────────────────────────┘

Market Source    MarketData     Features      Strategy    Signals    Execution    Broker
     │               │             │             │           │           │          │
     │ Tick          │             │             │           │           │          │
     ├──────────────>│             │             │           │           │          │
     │               │ Bar         │             │           │           │          │
     │               ├────────────>│             │           │           │          │
     │               │             │ FeatureVec  │           │           │          │
     │               │             ├────────────>│           │           │          │
     │               │             │             │ Signal    │           │          │
     │               │             │             ├──────────>│           │          │
     │               │             │             │           │ Validate  │          │
     │               │             │             │           ├──────────>│          │
     │               │             │             │           │           │ Order    │
     │               │             │             │           │           ├─────────>│
     │               │             │             │           │           │          │
     │               │             │             │           │           │ Fill     │
     │               │             │             │           │           │<─────────┤
     │               │             │             │           │  Update   │          │
     │               │             │             │           │<──────────┤          │
     │               │             │             │  Confirm  │           │          │
     │               │             │             │<──────────┤           │          │
     │               │             │             │           │           │          │

Latency Budget:
- Market → Bar: <1ms
- Bar → Features: <10ms
- Features → Signal: <50ms
- Signal → Validated: <5ms
- Validated → Order: <10ms
- Order → Fill: <100ms (broker)
TOTAL: <176ms (p95 target: <200ms)
```

### 2. Feature Extraction Pipeline

```
Historical Bars    FeatureExtractor              Polars DataFrame        Strategy
      │                   │                              │                   │
      │ Vec<Bar>          │                              │                   │
      ├──────────────────>│                              │                   │
      │                   │ Create DataFrame             │                   │
      │                   ├─────────────────────────────>│                   │
      │                   │                              │ Lazy Operations   │
      │                   │                              ├──────────┐        │
      │                   │                              │          │        │
      │                   │                              │ - SMA(20)│        │
      │                   │                              │ - EMA(12)│        │
      │                   │                              │ - RSI(14)│        │
      │                   │                              │ - MACD   │        │
      │                   │                              │<─────────┘        │
      │                   │                              │                   │
      │                   │                              │ Collect           │
      │                   │                              ├────────┐          │
      │                   │<─────────────────────────────┤        │          │
      │                   │ DataFrame with features      │<───────┘          │
      │                   │                              │                   │
      │                   │ Extract final row            │                   │
      │                   ├──────────────────────────────┤                   │
      │                   │<─────────────────────────────┤                   │
      │                   │ FeatureVector                │                   │
      │                   ├──────────────────────────────────────────────────>│
      │                   │                              │                   │

Performance:
- DataFrame creation: <1ms
- Indicator calculations: <5ms (lazy evaluation)
- Collect: <3ms
- Extract features: <1ms
TOTAL: <10ms for 100 bars
```

### 3. Signal Generation with Sublinear Optimization

```
Strategy     FeatureVector    SublinearSolver    AgentDB       Signal      RiskManager
   │               │                  │             │             │              │
   │ Analyze       │                  │             │             │              │
   ├──────────────>│                  │             │             │              │
   │               │ Query similar    │             │             │              │
   │               ├─────────────────────────────>│             │              │
   │               │                  │ Vec<Pattern>│             │              │
   │               │<────────────────────────────────┤             │              │
   │               │                  │             │             │              │
   │               │ Optimize(features, patterns)   │             │              │
   │               ├─────────────────>│             │             │              │
   │               │                  │ Solve in    │             │              │
   │               │                  │ o(√n) time  │             │              │
   │               │                  ├───────┐     │             │              │
   │               │                  │       │     │             │              │
   │               │                  │<──────┘     │             │              │
   │               │<─────────────────┤             │             │              │
   │ Generate      │ Optimal params   │             │             │              │
   │<──────────────┤                  │             │             │              │
   │               │                  │             │             │              │
   │ Create Signal │                  │             │             │              │
   ├────────────────────────────────────────────────────────────>│              │
   │               │                  │             │             │ Validate     │
   │               │                  │             │             ├─────────────>│
   │               │                  │             │             │ OK/Reject    │
   │               │                  │             │             │<─────────────┤
   │<────────────────────────────────────────────────────────────┤              │
   │ Signal (validated)               │             │             │              │

Latency Budget:
- Query AgentDB: <1ms
- Sublinear solve: <5ms
- Generate signal: <2ms
- Validate: <2ms
TOTAL: <10ms (well under 50ms target)
```

### 4. Order Execution Flow

```
SignalGenerator  OrderManager   RiskManager   Portfolio   Exchange   AgentDB
      │                │             │            │           │          │
      │ Signal         │             │            │           │          │
      ├───────────────>│             │            │           │          │
      │                │ Check risk  │            │           │          │
      │                ├────────────>│            │           │          │
      │                │ OK/Reject   │            │           │          │
      │                │<────────────┤            │           │          │
      │                │             │            │           │          │
      │                │ Calculate position size  │           │          │
      │                ├─────────────────────────>│           │          │
      │                │ Size        │            │           │          │
      │                │<─────────────────────────┤           │          │
      │                │             │            │           │          │
      │                │ Create Order│            │           │          │
      │                ├───────┐     │            │           │          │
      │                │       │     │            │           │          │
      │                │<──────┘     │            │           │          │
      │                │             │            │           │          │
      │                │ Submit Order│            │           │          │
      │                ├────────────────────────────────────>│          │
      │                │             │            │ Order ACK │          │
      │                │<────────────────────────────────────┤          │
      │                │             │            │           │          │
      │                │ Store decision trace     │           │          │
      │                ├────────────────────────────────────────────────>│
      │                │             │            │           │ Fill     │
      │                │<────────────────────────────────────┤          │
      │                │             │            │           │          │
      │                │ Update position          │           │          │
      │                ├─────────────────────────>│           │          │
      │                │             │            │           │          │
      │                │ Update risk metrics      │           │          │
      │                ├────────────>│            │           │          │
      │                │             │            │           │          │
      │<───────────────┤ Confirm     │            │           │          │
      │ Execution complete            │           │           │          │

Latency Budget:
- Risk check: <2ms
- Position sizing: <1ms
- Order creation: <1ms
- Submit to exchange: <5ms
- Exchange ACK: <50ms (external)
- Update state: <2ms
TOTAL: <61ms (within 100ms target)
```

### 5. AgentDB Memory Integration

```
TradingSystem   SessionMemory   LongTermMemory(AgentDB)   ReflexionEngine
     │                │                    │                      │
     │ Store obs      │                    │                      │
     ├───────────────>│                    │                      │
     │                │ TTL cache          │                      │
     │                ├──────┐             │                      │
     │                │      │             │                      │
     │                │<─────┘             │                      │
     │                │                    │                      │
     │                │ Async persist      │                      │
     │                ├───────────────────>│                      │
     │                │                    │ HNSW insert          │
     │                │                    ├───────┐              │
     │                │                    │       │              │
     │                │                    │<──────┘              │
     │                │<───────────────────┤                      │
     │                │                    │                      │
     │ Query similar  │                    │                      │
     ├───────────────────────────────────>│                      │
     │                │                    │ Vector search        │
     │                │                    ├───────┐              │
     │                │                    │       │              │
     │                │                    │<──────┘              │
     │<───────────────────────────────────┤ Results (<1ms)       │
     │                │                    │                      │
     │ Reflect on decision                 │                      │
     ├────────────────────────────────────────────────────────────>│
     │                │                    │                      │ Build trace
     │                │                    │                      ├─────────┐
     │                │                    │                      │         │
     │                │                    │                      │<────────┘
     │                │                    │                      │
     │                │                    │ Store trace          │
     │                │                    │<─────────────────────┤
     │                │                    │                      │
     │<───────────────────────────────────────────────────────────┤ Patterns
     │ Learned patterns                    │                      │

Memory Hierarchy:
- L1 (Session): <100ns lookup
- L2 (AgentDB): <1ms vector search
- L3 (Cold): >10ms (not shown)
```

---

## Interface Boundaries

### 1. Core Trait Definitions

```rust
// Strategy Interface
#[async_trait]
pub trait Strategy: Send + Sync {
    fn id(&self) -> &str;
    fn metadata(&self) -> StrategyMetadata;
    async fn on_market_data(&mut self, data: MarketData) -> Result<Vec<Signal>>;
    async fn generate_signals(&mut self) -> Result<Vec<Signal>>;
    fn risk_parameters(&self) -> RiskParameters;
}

// Market Data Provider Interface
#[async_trait]
pub trait MarketDataProvider: Send + Sync {
    async fn subscribe(&self, symbols: &[Symbol]) -> Result<Receiver<Tick>>;
    async fn get_quote(&self, symbol: &Symbol) -> Result<Quote>;
    async fn get_bars(&self, symbol: &Symbol, start: DateTime<Utc>, end: DateTime<Utc>) -> Result<Vec<Bar>>;
}

// Feature Extractor Interface
pub trait FeatureExtractor: Send + Sync {
    fn extract(&self, bars: &[Bar]) -> Result<DataFrame>;
    fn extract_single(&self, bars: &[Bar]) -> Result<FeatureVector>;
}

// Risk Manager Interface
#[async_trait]
pub trait RiskManager: Send + Sync {
    async fn validate_signal(&self, signal: &Signal, portfolio: &Portfolio) -> Result<()>;
    async fn calculate_metrics(&self, portfolio: &Portfolio) -> Result<RiskMetrics>;
    async fn check_limits(&self, portfolio: &Portfolio) -> Result<Vec<RiskViolation>>;
}

// Order Executor Interface
#[async_trait]
pub trait OrderExecutor: Send + Sync {
    async fn place_order(&self, order: &Order) -> Result<OrderResponse>;
    async fn cancel_order(&self, order_id: &str) -> Result<()>;
    async fn get_order_status(&self, order_id: &str) -> Result<OrderStatus>;
}
```

### 2. Module Communication Patterns

```rust
// Actor-based message passing
pub enum StrategyMessage {
    MarketData(MarketData),
    GenerateSignals,
    Shutdown,
}

pub struct StrategyActor {
    strategy: Box<dyn Strategy>,
    mailbox: mpsc::Receiver<StrategyMessage>,
    signal_tx: mpsc::Sender<Signal>,
}

impl StrategyActor {
    pub async fn run(mut self) {
        while let Some(msg) = self.mailbox.recv().await {
            match msg {
                StrategyMessage::MarketData(data) => {
                    if let Ok(signals) = self.strategy.on_market_data(data).await {
                        for signal in signals {
                            self.signal_tx.send(signal).await.ok();
                        }
                    }
                }
                StrategyMessage::GenerateSignals => {
                    if let Ok(signals) = self.strategy.generate_signals().await {
                        for signal in signals {
                            self.signal_tx.send(signal).await.ok();
                        }
                    }
                }
                StrategyMessage::Shutdown => break,
            }
        }
    }
}
```

### 3. Data Ownership and Borrowing

```rust
// Zero-copy market data
pub struct MarketDataFrame {
    inner: Arc<DataFrame>,  // Shared ownership
}

impl MarketDataFrame {
    // Cheap clone (Arc increment)
    pub fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }

    // Borrow for read
    pub fn view(&self) -> &DataFrame {
        &self.inner
    }

    // Expensive: Get owned copy only when needed
    pub fn to_owned(&self) -> DataFrame {
        (*self.inner).clone()
    }
}

// Message passing with ownership transfer
pub enum DataMessage {
    Tick(Box<Tick>),           // Boxed for small size
    Bars(Arc<Vec<Bar>>),       // Arc for sharing
    DataFrame(Arc<DataFrame>), // Arc for large data
}
```

---

## Data Flow Diagrams

### High-Level Data Flow

```
┌────────────────────────────────────────────────────────────────┐
│                    External Data Sources                        │
│  Alpaca WebSocket │ Polygon API │ NewsAPI │ Federal Reserve    │
└───────┬───────────┴────────┬────┴────┬────┴────────┬───────────┘
        │                    │         │             │
        ▼                    ▼         ▼             ▼
┌────────────────────────────────────────────────────────────────┐
│                   Market Data Layer (nt-market-data)            │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────────┐  │
│  │  WebSocket   │  │  REST Client │  │  News Collector     │  │
│  │  Streams     │  │  (Historical)│  │  (RSS/API)          │  │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬──────────┘  │
└─────────┼──────────────────┼──────────────────────┼─────────────┘
          │                  │                      │
          ▼                  ▼                      ▼
┌────────────────────────────────────────────────────────────────┐
│              Data Processing Layer (nt-features)                │
│  ┌──────────────────┐  ┌────────────────┐  ┌────────────────┐ │
│  │  Technical       │  │  Sentiment     │  │  Market Regime │ │
│  │  Indicators      │  │  Analysis      │  │  Detection     │ │
│  │  (Polars)        │  │  (NLP)         │  │                │ │
│  └────────┬─────────┘  └────────┬───────┘  └────────┬───────┘ │
└───────────┼──────────────────────┼──────────────────┼──────────┘
            │                      │                  │
            └──────────────────────┴──────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────┐
│              Strategy Layer (nt-strategies)                     │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌──────────┐ │
│  │ Momentum   │  │ Mean       │  │ Neural     │  │ Mirror   │ │
│  │ Strategy   │  │ Reversion  │  │ Sentiment  │  │ Trading  │ │
│  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘  └────┬─────┘ │
└────────┼────────────────┼────────────────┼──────────────┼───────┘
         │                │                │              │
         └────────────────┴────────────────┴──────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────┐
│              Signal Generation (nt-signals)                     │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  Signal Aggregation, Filtering, Prioritization        │    │
│  │  (AgentDB similarity search for pattern matching)     │    │
│  └────────────────────────────┬───────────────────────────┘    │
└─────────────────────────────────┼─────────────────────────────┘
                                  │
                                  ▼
┌────────────────────────────────────────────────────────────────┐
│              Risk Management (nt-risk)                          │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────────────┐   │
│  │ Position    │  │ VaR/CVaR    │  │ Circuit Breakers     │   │
│  │ Limits      │  │ Calculation │  │                      │   │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬───────────┘   │
└─────────┼────────────────┼──────────────────────┼───────────────┘
          │                │                      │
          └────────────────┴──────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────────────┐
│              Order Execution (nt-execution)                     │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────┐    │
│  │ Order        │  │ Fill         │  │ Settlement        │    │
│  │ Management   │  │ Tracking     │  │ Reconciliation    │    │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬────────┘    │
└─────────┼──────────────────┼──────────────────────┼─────────────┘
          │                  │                      │
          ▼                  ▼                      ▼
┌────────────────────────────────────────────────────────────────┐
│                   Broker APIs (External)                        │
│                   Alpaca / Polygon / IEX                        │
└────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│              Cross-Cutting Concerns                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │ AgentDB     │  │ Portfolio   │  │ Monitoring & Logging    │ │
│  │ (Memory)    │  │ Management  │  │ (OpenTelemetry)         │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## Concurrency Model

### Tokio Runtime Configuration

```rust
// src/main.rs

#[tokio::main]
async fn main() -> Result<()> {
    // Configure runtime
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(num_cpus::get())
        .thread_name("nt-worker")
        .enable_all()
        .build()?;

    runtime.block_on(async {
        run_trading_system().await
    })
}
```

### Actor Pattern for Strategies

```rust
pub struct TradingSystem {
    strategies: Vec<StrategyHandle>,
    market_data: MarketDataHandle,
    execution: ExecutionHandle,
}

pub struct StrategyHandle {
    id: String,
    mailbox: mpsc::Sender<StrategyMessage>,
    handle: JoinHandle<()>,
}

impl TradingSystem {
    pub async fn start(&mut self) -> Result<()> {
        // Start market data stream
        let (tick_tx, tick_rx) = mpsc::channel(1000);
        self.market_data.subscribe(tick_tx).await?;

        // Spawn strategy actors
        for strategy in &self.strategies {
            let mailbox = strategy.mailbox.clone();

            tokio::spawn(async move {
                while let Some(tick) = tick_rx.recv().await {
                    mailbox.send(StrategyMessage::MarketData(tick)).await.ok();
                }
            });
        }

        Ok(())
    }
}
```

### Channel-Based Communication

```rust
// High-throughput channels for market data
let (tick_tx, tick_rx) = mpsc::channel::<Tick>(10000);

// Broadcast for state updates
let (state_tx, _) = broadcast::channel::<PortfolioState>(100);

// Watch for configuration changes
let (config_tx, config_rx) = watch::channel(AppConfig::default());

// Oneshot for request-response
let (response_tx, response_rx) = oneshot::channel();
```

### Lock-Free Data Structures

```rust
use dashmap::DashMap;
use parking_lot::RwLock;

pub struct Portfolio {
    // Concurrent HashMap for position updates
    positions: DashMap<Symbol, Position>,

    // Fast RwLock for read-heavy data
    metrics: Arc<RwLock<PortfolioMetrics>>,
}

impl Portfolio {
    pub fn update_position(&self, symbol: Symbol, position: Position) {
        self.positions.insert(symbol, position);
    }

    pub fn get_position(&self, symbol: &Symbol) -> Option<Position> {
        self.positions.get(symbol).map(|p| p.clone())
    }
}
```

---

## Error Handling Patterns

### Error Type Hierarchy

```rust
#[derive(Debug, thiserror::Error)]
pub enum TradingError {
    #[error("Market data error: {source}")]
    MarketData {
        source: Box<dyn std::error::Error + Send + Sync>,
        context: String,
    },

    #[error("Strategy error in {strategy_id}: {message}")]
    Strategy {
        strategy_id: String,
        message: String,
    },

    #[error("Execution error: {message}")]
    Execution {
        message: String,
        order_id: Option<String>,
    },

    #[error("Risk limit exceeded: {violation}")]
    RiskLimit {
        violation: RiskViolation,
    },

    #[error("Configuration error: {0}")]
    Config(String),

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

impl TradingError {
    pub fn with_context<S: Into<String>>(self, context: S) -> Self {
        match self {
            TradingError::MarketData { source, context: old_context } => {
                TradingError::MarketData {
                    source,
                    context: format!("{}: {}", context.into(), old_context),
                }
            }
            other => other,
        }
    }
}
```

### Circuit Breaker Pattern

```rust
pub struct CircuitBreaker {
    state: Arc<RwLock<CircuitState>>,
    failure_threshold: usize,
    reset_timeout: Duration,
}

enum CircuitState {
    Closed { failure_count: usize },
    Open { opened_at: Instant },
    HalfOpen,
}

impl CircuitBreaker {
    pub async fn call<F, T>(&self, f: F) -> Result<T>
    where
        F: FnOnce() -> Pin<Box<dyn Future<Output = Result<T>> + Send>>,
    {
        // Check state
        let state = self.state.read().await;
        match *state {
            CircuitState::Open { opened_at } => {
                if opened_at.elapsed() > self.reset_timeout {
                    drop(state);
                    *self.state.write().await = CircuitState::HalfOpen;
                } else {
                    return Err(TradingError::CircuitOpen);
                }
            }
            _ => {}
        }
        drop(state);

        // Execute function
        match f().await {
            Ok(result) => {
                // Reset on success
                *self.state.write().await = CircuitState::Closed { failure_count: 0 };
                Ok(result)
            }
            Err(e) => {
                // Increment failure count
                let mut state = self.state.write().await;
                match *state {
                    CircuitState::Closed { ref mut failure_count } => {
                        *failure_count += 1;
                        if *failure_count >= self.failure_threshold {
                            *state = CircuitState::Open {
                                opened_at: Instant::now(),
                            };
                        }
                    }
                    CircuitState::HalfOpen => {
                        *state = CircuitState::Open {
                            opened_at: Instant::now(),
                        };
                    }
                    _ => {}
                }
                Err(e)
            }
        }
    }
}
```

### Retry with Exponential Backoff

```rust
pub async fn retry_with_backoff<F, T, E>(
    mut f: F,
    max_attempts: usize,
    initial_delay: Duration,
) -> Result<T, E>
where
    F: FnMut() -> Pin<Box<dyn Future<Output = Result<T, E>> + Send>>,
{
    let mut delay = initial_delay;

    for attempt in 0..max_attempts {
        match f().await {
            Ok(result) => return Ok(result),
            Err(e) if attempt == max_attempts - 1 => return Err(e),
            Err(_) => {
                tokio::time::sleep(delay).await;
                delay *= 2;
            }
        }
    }

    unreachable!()
}
```

---

## Integration Points

### 1. AgentDB Integration

**Purpose:** Vector database for memory, pattern matching, reflexion

```rust
use agentdb_client::{AgentDBClient, Query, Filter};

pub struct AgentDBMemory {
    client: AgentDBClient,
    observations_collection: String,
    signals_collection: String,
    traces_collection: String,
}

impl AgentDBMemory {
    pub async fn new() -> Result<Self> {
        let client = AgentDBClient::connect("http://localhost:8080").await?;
        Ok(Self {
            client,
            observations_collection: "observations".to_string(),
            signals_collection: "signals".to_string(),
            traces_collection: "reflexion_traces".to_string(),
        })
    }

    pub async fn store_observation(&self, obs: &Observation) -> Result<()> {
        self.client
            .insert(&self.observations_collection, obs.id.as_bytes(), &obs.embedding, Some(obs))
            .await?;
        Ok(())
    }

    pub async fn find_similar_conditions(&self, obs: &Observation, k: usize) -> Result<Vec<Observation>> {
        let query = Query::new(&obs.embedding)
            .k(k)
            .filter(Filter::eq("symbol", &obs.symbol));

        self.client.search(&self.observations_collection, query).await
    }
}
```

**Performance Target:** <1ms for k=10 similarity search

### 2. Agentic Flow Integration

**Purpose:** Multi-agent coordination and distributed execution

```rust
// Integration via Claude Flow MCP tools
pub async fn coordinate_with_agentic_flow(
    task: &str,
    agents: Vec<&str>,
) -> Result<CoordinationResult> {
    // Initialize swarm
    let swarm = mcp_claude_flow::swarm_init(SwarmConfig {
        topology: Topology::Mesh,
        max_agents: agents.len(),
    }).await?;

    // Spawn agents
    for agent in agents {
        mcp_claude_flow::agent_spawn(AgentConfig {
            agent_type: agent.to_string(),
            capabilities: vec!["trading".to_string()],
        }).await?;
    }

    // Orchestrate task
    let result = mcp_claude_flow::task_orchestrate(TaskConfig {
        task: task.to_string(),
        parallel: true,
    }).await?;

    Ok(result)
}
```

### 3. Midstreamer Event Bus

**Purpose:** Event-driven streaming and message passing

```rust
use midstreamer::{Stream, Operator};

pub struct TradingEventBus {
    tick_stream: Stream<Tick>,
    signal_stream: Stream<Signal>,
    order_stream: Stream<Order>,
}

impl TradingEventBus {
    pub fn new() -> Self {
        let tick_stream = Stream::new("ticks");
        let signal_stream = tick_stream
            .map(|tick| extract_features(tick))
            .flat_map(|features| generate_signals(features))
            .filter(|signal| signal.confidence > 0.7);

        let order_stream = signal_stream
            .map(|signal| validate_risk(signal))
            .map(|signal| create_order(signal));

        Self {
            tick_stream,
            signal_stream,
            order_stream,
        }
    }
}
```

---

## Performance Targets Per Module

| Module | Latency (p50) | Latency (p99) | Throughput | Memory | Priority |
|--------|--------------|---------------|------------|--------|----------|
| **nt-market-data** | 0.5ms | 2ms | 10K ticks/sec | <10MB/conn | P0 |
| **nt-features** | 5ms | 15ms | 5K/sec | <50MB | P0 |
| **nt-strategies** | 10ms | 30ms | 2K signals/sec | <20MB each | P0 |
| **nt-signals** | 2ms | 5ms | 5K/sec | <10MB | P0 |
| **nt-portfolio** | 0.1ms | 0.5ms | 10K ops/sec | <50MB | P0 |
| **nt-risk** | 2ms | 8ms | 5K checks/sec | <20MB | P0 |
| **nt-execution** | 8ms | 20ms | 1K orders/sec | <30MB | P0 |
| **nt-backtesting** | N/A | N/A | 100 bars/ms | <500MB | P1 |
| **nt-neural** | 40ms | 120ms (GPU) | 10 inf/sec | <2GB | P1 |
| **nt-news** | 50ms | 150ms | 100/min | <100MB | P1 |
| **nt-agentdb** | 0.5ms | 2ms | 10K queries/sec | <100MB | P0 |
| **nt-api** | 10ms | 50ms | 5K req/sec | <100MB | P0 |
| **nt-napi** | <0.1ms | 0.5ms | 1M ops/sec | <50MB | P0 |

---

## Cross-References

- **Parity Requirements:** [02_Parity_Requirements.md](02_Parity_Requirements.md)
- **Interop Strategy:** [04_Rust_Crates_and_Node_Interop.md](04_Rust_Crates_and_Node_Interop.md)
- **Memory Architecture:** [05_Memory_and_AgentDB.md](05_Memory_and_AgentDB.md)
- **Strategy Algorithms:** [06_Strategy_and_Sublinear_Solvers.md](06_Strategy_and_Sublinear_Solvers.md)
- **Event Streaming:** [07_Streaming_and_Midstreamer.md](07_Streaming_and_Midstreamer.md)

---

**Document Status:** ✅ Complete
**Last Updated:** 2025-11-12
**Next Review:** Phase 1 kickoff (Week 3)
**Owner:** System Architect
