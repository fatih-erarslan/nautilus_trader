# Module Design and Interfaces

## Workspace Structure

```
neural-trader-rs/
├── Cargo.toml                 # Workspace root
├── crates/
│   ├── nt-core/              # Core types and traits
│   ├── nt-data/              # Market data ingestion
│   ├── nt-features/          # Feature extraction
│   ├── nt-strategies/        # Strategy plugins
│   ├── nt-signals/           # Signal generation
│   ├── nt-execution/         # Order execution
│   ├── nt-risk/              # Risk management
│   ├── nt-memory/            # AgentDB integration
│   ├── nt-sublinear/         # Sublinear algorithms
│   ├── nt-streaming/         # Midstreamer integration
│   ├── nt-sandbox/           # E2B sandbox client
│   ├── nt-federation/        # Agentic Flow federation
│   ├── nt-payments/          # Cost tracking
│   ├── nt-governance/        # AIDefence integration
│   ├── nt-observability/     # Tracing and metrics
│   └── nt-napi/              # Node.js bindings
├── examples/                  # Usage examples
├── tests/                     # Integration tests
└── benches/                   # Performance benchmarks
```

## Core Module: `nt-core`

Fundamental types and traits used across the system.

### Key Types

```rust
// src/types.rs
use rust_decimal::Decimal;
use chrono::{DateTime, Utc};
use serde::{Serialize, Deserialize};

/// Symbol identifier (e.g., "AAPL", "BTC-USD")
#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct Symbol(String);

/// Monetary amount with precision
pub type Price = Decimal;
pub type Quantity = Decimal;

/// Order side
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Side {
    Buy,
    Sell,
}

/// Order type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OrderType {
    Market,
    Limit { price: Price },
    StopLoss { stop_price: Price },
    TrailingStop { trail_amount: Decimal },
}

/// Time in force
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum TimeInForce {
    Day,
    GoodTilCanceled,
    ImmediateOrCancel,
    FillOrKill,
}

/// Market data snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Quote {
    pub symbol: Symbol,
    pub timestamp: DateTime<Utc>,
    pub bid: Price,
    pub ask: Price,
    pub bid_size: Quantity,
    pub ask_size: Quantity,
}

/// Trade event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    pub symbol: Symbol,
    pub timestamp: DateTime<Utc>,
    pub price: Price,
    pub size: Quantity,
    pub side: Side,
}

/// OHLCV bar
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bar {
    pub symbol: Symbol,
    pub timestamp: DateTime<Utc>,
    pub open: Price,
    pub high: Price,
    pub low: Price,
    pub close: Price,
    pub volume: Quantity,
}
```

### Core Traits

```rust
// src/traits.rs
use async_trait::async_trait;
use std::error::Error;

/// Result type for trading operations
pub type TradingResult<T> = Result<T, Box<dyn Error + Send + Sync>>;

/// Market data provider
#[async_trait]
pub trait MarketDataProvider: Send + Sync {
    /// Subscribe to real-time quotes
    async fn subscribe_quotes(&self, symbols: &[Symbol])
        -> TradingResult<mpsc::Receiver<Quote>>;

    /// Subscribe to real-time trades
    async fn subscribe_trades(&self, symbols: &[Symbol])
        -> TradingResult<mpsc::Receiver<Trade>>;

    /// Fetch historical bars
    async fn get_bars(&self,
        symbol: &Symbol,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
        timeframe: Timeframe
    ) -> TradingResult<Vec<Bar>>;
}

/// Strategy plugin interface
#[async_trait]
pub trait Strategy: Send + Sync {
    /// Initialize strategy with configuration
    async fn initialize(&mut self, config: StrategyConfig) -> TradingResult<()>;

    /// Process market data update
    async fn on_market_data(&mut self, data: MarketData) -> TradingResult<Vec<Signal>>;

    /// Generate signals
    async fn generate_signals(&mut self) -> TradingResult<Vec<Signal>>;

    /// Get strategy metadata
    fn metadata(&self) -> StrategyMetadata;
}

/// Signal generator
#[async_trait]
pub trait SignalGenerator: Send + Sync {
    /// Evaluate features and produce trading signals
    async fn evaluate(&self, features: &FeatureSet) -> TradingResult<Vec<Signal>>;

    /// Get signal strength (0.0 to 1.0)
    fn signal_strength(&self, signal: &Signal) -> f64;
}

/// Order executor
#[async_trait]
pub trait OrderExecutor: Send + Sync {
    /// Submit order to exchange
    async fn submit_order(&self, order: Order) -> TradingResult<OrderId>;

    /// Cancel existing order
    async fn cancel_order(&self, order_id: &OrderId) -> TradingResult<()>;

    /// Get order status
    async fn get_order_status(&self, order_id: &OrderId) -> TradingResult<OrderStatus>;

    /// Subscribe to order updates
    async fn subscribe_orders(&self) -> TradingResult<mpsc::Receiver<OrderUpdate>>;
}

/// Risk management interface
#[async_trait]
pub trait RiskManager: Send + Sync {
    /// Validate order against risk limits
    async fn validate_order(&self, order: &Order) -> TradingResult<RiskDecision>;

    /// Check portfolio risk metrics
    async fn check_portfolio_risk(&self, portfolio: &Portfolio) -> TradingResult<RiskMetrics>;

    /// Calculate position size
    fn calculate_position_size(&self,
        signal: &Signal,
        account: &Account
    ) -> TradingResult<Quantity>;
}

/// Memory store interface (AgentDB)
#[async_trait]
pub trait MemoryStore: Send + Sync {
    /// Store vector embedding
    async fn store_vector(&self, key: &str, vector: Vec<f32>) -> TradingResult<()>;

    /// Query similar vectors
    async fn query_similar(&self, vector: Vec<f32>, k: usize)
        -> TradingResult<Vec<(String, f32)>>;

    /// Store key-value pair
    async fn store_kv(&self, key: &str, value: &[u8]) -> TradingResult<()>;

    /// Retrieve value by key
    async fn get_kv(&self, key: &str) -> TradingResult<Option<Vec<u8>>>;
}
```

## Data Module: `nt-data`

Market data ingestion with Polars integration.

### Module Structure

```rust
// src/providers/mod.rs
pub mod alpaca;
pub mod polygon;
pub mod coinbase;

// src/stream.rs
use polars::prelude::*;
use tokio::sync::mpsc;

/// Market data stream manager
pub struct DataStream {
    provider: Box<dyn MarketDataProvider>,
    buffer: DataBuffer,
    subscribers: Vec<mpsc::Sender<DataFrame>>,
}

impl DataStream {
    /// Create new data stream
    pub fn new(provider: Box<dyn MarketDataProvider>) -> Self {
        Self {
            provider,
            buffer: DataBuffer::new(1000), // 1000 rows
            subscribers: Vec::new(),
        }
    }

    /// Subscribe to data stream
    pub async fn subscribe(&mut self) -> mpsc::Receiver<DataFrame> {
        let (tx, rx) = mpsc::channel(100);
        self.subscribers.push(tx);
        rx
    }

    /// Start streaming data
    pub async fn start(&mut self, symbols: Vec<Symbol>) -> TradingResult<()> {
        let mut quote_rx = self.provider.subscribe_quotes(&symbols).await?;

        while let Some(quote) = quote_rx.recv().await {
            self.buffer.push(quote);

            if self.buffer.is_full() {
                let df = self.buffer.to_dataframe()?;
                self.broadcast(df).await;
                self.buffer.clear();
            }
        }

        Ok(())
    }

    async fn broadcast(&self, df: DataFrame) {
        for subscriber in &self.subscribers {
            let _ = subscriber.send(df.clone()).await;
        }
    }
}

/// Circular buffer for market data
struct DataBuffer {
    capacity: usize,
    symbols: Vec<String>,
    timestamps: Vec<i64>,
    bids: Vec<f64>,
    asks: Vec<f64>,
    bid_sizes: Vec<f64>,
    ask_sizes: Vec<f64>,
}

impl DataBuffer {
    fn new(capacity: usize) -> Self {
        Self {
            capacity,
            symbols: Vec::with_capacity(capacity),
            timestamps: Vec::with_capacity(capacity),
            bids: Vec::with_capacity(capacity),
            asks: Vec::with_capacity(capacity),
            bid_sizes: Vec::with_capacity(capacity),
            ask_sizes: Vec::with_capacity(capacity),
        }
    }

    fn push(&mut self, quote: Quote) {
        self.symbols.push(quote.symbol.to_string());
        self.timestamps.push(quote.timestamp.timestamp_nanos());
        self.bids.push(quote.bid.to_f64().unwrap());
        self.asks.push(quote.ask.to_f64().unwrap());
        self.bid_sizes.push(quote.bid_size.to_f64().unwrap());
        self.ask_sizes.push(quote.ask_size.to_f64().unwrap());
    }

    fn is_full(&self) -> bool {
        self.symbols.len() >= self.capacity
    }

    fn clear(&mut self) {
        self.symbols.clear();
        self.timestamps.clear();
        self.bids.clear();
        self.asks.clear();
        self.bid_sizes.clear();
        self.ask_sizes.clear();
    }

    fn to_dataframe(&self) -> TradingResult<DataFrame> {
        let df = DataFrame::new(vec![
            Series::new("symbol", &self.symbols),
            Series::new("timestamp", &self.timestamps),
            Series::new("bid", &self.bids),
            Series::new("ask", &self.asks),
            Series::new("bid_size", &self.bid_sizes),
            Series::new("ask_size", &self.ask_sizes),
        ])?;

        Ok(df)
    }
}
```

## Features Module: `nt-features`

Feature extraction using Polars expressions.

```rust
// src/extractors/technical.rs
use polars::prelude::*;

pub struct TechnicalFeatures;

impl TechnicalFeatures {
    /// Calculate Simple Moving Average
    pub fn sma(df: &DataFrame, column: &str, window: usize) -> PolarsResult<Series> {
        df.column(column)?
            .rolling_mean(RollingOptions {
                window_size: window,
                min_periods: window,
                ..Default::default()
            })
    }

    /// Calculate Exponential Moving Average
    pub fn ema(df: &DataFrame, column: &str, span: usize) -> PolarsResult<Series> {
        df.column(column)?
            .ewm_mean(EWMOptions {
                alpha: 2.0 / (span as f64 + 1.0),
                ..Default::default()
            })
    }

    /// Calculate RSI (Relative Strength Index)
    pub fn rsi(df: &DataFrame, column: &str, period: usize) -> PolarsResult<Series> {
        let prices = df.column(column)?;
        let changes = prices.diff(1, NullBehavior::Drop);

        let gains = changes.apply(|v| if v > 0.0 { v } else { 0.0 });
        let losses = changes.apply(|v| if v < 0.0 { -v } else { 0.0 });

        let avg_gain = gains.rolling_mean(RollingOptions {
            window_size: period,
            min_periods: period,
            ..Default::default()
        })?;

        let avg_loss = losses.rolling_mean(RollingOptions {
            window_size: period,
            min_periods: period,
            ..Default::default()
        })?;

        let rs = &avg_gain / &avg_loss;
        let rsi = lit(100.0) - (lit(100.0) / (lit(1.0) + rs));

        Ok(rsi.into_series())
    }

    /// Calculate Bollinger Bands
    pub fn bollinger_bands(
        df: &DataFrame,
        column: &str,
        window: usize,
        std_dev: f64
    ) -> PolarsResult<(Series, Series, Series)> {
        let middle = Self::sma(df, column, window)?;

        let std = df.column(column)?
            .rolling_std(RollingOptions {
                window_size: window,
                min_periods: window,
                ..Default::default()
            })?;

        let upper = &middle + (std.clone() * std_dev);
        let lower = &middle - (std * std_dev);

        Ok((upper, middle, lower))
    }

    /// Calculate MACD
    pub fn macd(
        df: &DataFrame,
        column: &str,
        fast: usize,
        slow: usize,
        signal: usize
    ) -> PolarsResult<(Series, Series, Series)> {
        let ema_fast = Self::ema(df, column, fast)?;
        let ema_slow = Self::ema(df, column, slow)?;

        let macd_line = &ema_fast - &ema_slow;

        // Signal line is EMA of MACD
        let signal_line = macd_line.ewm_mean(EWMOptions {
            alpha: 2.0 / (signal as f64 + 1.0),
            ..Default::default()
        })?;

        let histogram = &macd_line - &signal_line;

        Ok((macd_line, signal_line, histogram))
    }
}

/// Feature engineering pipeline
pub struct FeaturePipeline {
    features: Vec<Box<dyn FeatureExtractor>>,
}

#[async_trait]
pub trait FeatureExtractor: Send + Sync {
    async fn extract(&self, df: &DataFrame) -> PolarsResult<DataFrame>;
    fn feature_names(&self) -> Vec<String>;
}

impl FeaturePipeline {
    pub fn new() -> Self {
        Self {
            features: Vec::new(),
        }
    }

    pub fn add_extractor(&mut self, extractor: Box<dyn FeatureExtractor>) {
        self.features.push(extractor);
    }

    pub async fn process(&self, df: DataFrame) -> PolarsResult<DataFrame> {
        let mut result = df;

        for extractor in &self.features {
            result = extractor.extract(&result).await?;
        }

        Ok(result)
    }
}
```

## Strategy Module: `nt-strategies`

Plugin architecture for trading strategies.

```rust
// src/plugin.rs
use libloading::{Library, Symbol};
use std::path::Path;

pub struct StrategyPlugin {
    library: Library,
    strategy: Box<dyn Strategy>,
}

impl StrategyPlugin {
    /// Load strategy from dynamic library
    pub unsafe fn load<P: AsRef<Path>>(path: P) -> TradingResult<Self> {
        let library = Library::new(path.as_ref())?;

        let constructor: Symbol<fn() -> Box<dyn Strategy>> =
            library.get(b"create_strategy")?;

        let strategy = constructor();

        Ok(Self {
            library,
            strategy,
        })
    }

    pub fn strategy(&mut self) -> &mut dyn Strategy {
        self.strategy.as_mut()
    }
}

/// Strategy registry
pub struct StrategyRegistry {
    strategies: HashMap<String, Box<dyn Strategy>>,
}

impl StrategyRegistry {
    pub fn new() -> Self {
        Self {
            strategies: HashMap::new(),
        }
    }

    pub fn register(&mut self, name: String, strategy: Box<dyn Strategy>) {
        self.strategies.insert(name, strategy);
    }

    pub fn get(&self, name: &str) -> Option<&dyn Strategy> {
        self.strategies.get(name).map(|s| s.as_ref())
    }

    pub fn list(&self) -> Vec<String> {
        self.strategies.keys().cloned().collect()
    }
}

// Example strategy implementation
pub struct MomentumStrategy {
    config: MomentumConfig,
    state: StrategyState,
}

#[async_trait]
impl Strategy for MomentumStrategy {
    async fn initialize(&mut self, config: StrategyConfig) -> TradingResult<()> {
        self.config = serde_json::from_value(config.params)?;
        Ok(())
    }

    async fn on_market_data(&mut self, data: MarketData) -> TradingResult<Vec<Signal>> {
        let mut signals = Vec::new();

        // Calculate momentum
        let momentum = self.calculate_momentum(&data)?;

        if momentum > self.config.entry_threshold {
            signals.push(Signal {
                symbol: data.symbol.clone(),
                side: Side::Buy,
                strength: momentum,
                timestamp: Utc::now(),
            });
        } else if momentum < -self.config.entry_threshold {
            signals.push(Signal {
                symbol: data.symbol.clone(),
                side: Side::Sell,
                strength: -momentum,
                timestamp: Utc::now(),
            });
        }

        Ok(signals)
    }

    async fn generate_signals(&mut self) -> TradingResult<Vec<Signal>> {
        // Periodic signal generation
        Ok(Vec::new())
    }

    fn metadata(&self) -> StrategyMetadata {
        StrategyMetadata {
            name: "Momentum Strategy".to_string(),
            version: "1.0.0".to_string(),
            author: "Neural Trader".to_string(),
            description: "Trend-following momentum strategy".to_string(),
        }
    }
}

impl MomentumStrategy {
    fn calculate_momentum(&self, data: &MarketData) -> TradingResult<f64> {
        // Momentum calculation logic
        Ok(0.0)
    }
}

// Macro for strategy plugin export
#[macro_export]
macro_rules! export_strategy {
    ($strategy_type:ty) => {
        #[no_mangle]
        pub extern "C" fn create_strategy() -> Box<dyn Strategy> {
            Box::new(<$strategy_type>::default())
        }
    };
}
```

## Interface Contracts

### Version Compatibility

All modules follow semantic versioning. Breaking changes require major version bump.

```rust
#[derive(Debug, Clone)]
pub struct ApiVersion {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
}

impl ApiVersion {
    pub const CURRENT: Self = Self {
        major: 1,
        minor: 0,
        patch: 0,
    };

    pub fn is_compatible(&self, other: &Self) -> bool {
        self.major == other.major && self.minor >= other.minor
    }
}
```

### Error Handling Contract

All public functions return `TradingResult<T>` for consistent error handling.

```rust
#[derive(Debug, thiserror::Error)]
pub enum TradingError {
    #[error("Market data error: {0}")]
    MarketData(String),

    #[error("Execution error: {0}")]
    Execution(String),

    #[error("Risk limit exceeded: {0}")]
    RiskLimit(String),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}
```

---

**Next:** [03-data-flow-diagrams.md](./03-data-flow-diagrams.md)
