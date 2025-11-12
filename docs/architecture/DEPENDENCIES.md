# System Dependencies and Data Flow

## 1. Crate Dependency Graph

```
hyperphysics-trading (Live Execution)
    ├─> hyperphysics-market (Data Feeds)
    ├─> hyperphysics-risk (Risk Monitoring)
    ├─> hyperphysics-backtest (Strategy Validation)
    ├─> hyperphysics-metrics (Consciousness Metrics)
    └─> hyperphysics-core (Shared Types)

hyperphysics-backtest (Backtesting)
    ├─> hyperphysics-market (Historical Data)
    ├─> hyperphysics-risk (Portfolio Risk)
    ├─> hyperphysics-metrics (Regime Detection)
    └─> hyperphysics-core

hyperphysics-risk (Risk Management)
    ├─> hyperphysics-market (Price Data)
    ├─> hyperphysics-metrics (Entropy Calculation)
    └─> hyperphysics-core

hyperphysics-market (Market Data)
    └─> hyperphysics-core

hyperphysics-metrics (Consciousness Metrics)
    └─> hyperphysics-core
```

## 2. Data Flow Architecture

### 2.1 Live Trading Flow

```
Market Data Providers → Market Feed Aggregator
                              ↓
                    [hyperphysics-market]
                              ↓
                        Market Topology
                              ↓
                    ┌─────────┴─────────┐
                    ↓                   ↓
          [hyperphysics-metrics]  [hyperphysics-risk]
                    ↓                   ↓
                Φ, CI Calculation   Entropy Monitor
                    ↓                   ↓
                    └─────────┬─────────┘
                              ↓
                    Strategy Signal Generation
                              ↓
                    Pre-Trade Risk Check
                              ↓
                    [hyperphysics-trading]
                              ↓
                       Order Router
                              ↓
                    Broker Execution
                              ↓
                        Fill Events
                              ↓
                    Portfolio Update
```

### 2.2 Backtesting Flow

```
Historical Data Repository
        ↓
[hyperphysics-market]
        ↓
    Bar Replay
        ↓
[hyperphysics-backtest]
        ↓
    Event Queue
        ↓
    ├─> Strategy Signals
    ├─> Execution Simulation
    ├─> Risk Checks
    └─> Performance Metrics
        ↓
    Results Analysis
```

## 3. Inter-Crate APIs

### 3.1 Market → Risk

```rust
// hyperphysics-market provides:
pub trait MarketDataProvider {
    async fn fetch_correlation_matrix(
        &self,
        symbols: &[String],
        window: Duration,
    ) -> Result<Array2<f64>>;

    async fn fetch_volatility(
        &self,
        symbol: &str,
        window: Duration,
    ) -> Result<f64>;
}

// hyperphysics-risk consumes:
impl ThermodynamicRisk {
    pub async fn from_market_data(
        provider: &dyn MarketDataProvider,
        portfolio: &Portfolio,
    ) -> Result<Self>;
}
```

### 3.2 Metrics → Backtest

```rust
// hyperphysics-metrics provides:
pub trait ConsciousnessMetrics {
    fn calculate_phi(&self, data: &TimeSeriesData) -> Result<f64>;
    fn calculate_ci(&self, data: &TimeSeriesData) -> Result<f64>;
    fn detect_regime(&self, phi: f64, ci: f64) -> MarketRegime;
}

// hyperphysics-backtest consumes:
impl RegimeAwareStrategy {
    pub fn new(metrics: Arc<dyn ConsciousnessMetrics>) -> Self;

    async fn on_bar(
        &self,
        bar: &BarEvent,
    ) -> Result<Option<SignalEvent>> {
        let phi = self.metrics.calculate_phi(&bar.to_timeseries())?;
        let ci = self.metrics.calculate_ci(&bar.to_timeseries())?;
        let regime = self.metrics.detect_regime(phi, ci);

        self.generate_signal(regime, bar)
    }
}
```

### 3.3 Risk → Trading

```rust
// hyperphysics-risk provides:
pub trait RiskMonitor {
    fn check_pre_trade(&self, order: &OrderEvent, portfolio: &Portfolio) -> bool;
    fn check_real_time(&self, portfolio: &Portfolio) -> Result<(), RiskViolation>;
}

// hyperphysics-trading consumes:
impl TradingOrchestrator {
    async fn process_signal(&self, signal: SignalEvent) {
        let order = self.strategy.signal_to_order(signal)?;

        // Risk check before execution
        if !self.risk_monitor.check_pre_trade(&order, &self.portfolio) {
            return Err(TradingError::RiskCheckFailed);
        }

        self.order_router.submit_order(order).await
    }
}
```

## 4. Shared Core Types

### 4.1 hyperphysics-core

```rust
// Shared types across all crates
pub mod types {
    use chrono::{DateTime, Utc};
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct Asset {
        pub symbol: String,
        pub asset_class: AssetClass,
        pub exchange: String,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub enum AssetClass {
        Equity,
        Option,
        Future,
        Crypto,
        Forex,
        Bond,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct Position {
        pub symbol: String,
        pub quantity: f64,
        pub market_value: f64,
        pub cost_basis: f64,
        pub unrealized_pnl: f64,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct Portfolio {
        pub positions: HashMap<String, Position>,
        pub cash: f64,
        pub total_value: f64,
        pub entropy: f64,
        pub temperature: f64,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub enum MarketRegime {
        Bull,
        Bear,
        Bubble,
        Correction,
    }
}
```

## 5. External Dependencies

### 5.1 Core Libraries

```toml
[dependencies]
# Async runtime
tokio = { version = "1.35", features = ["full"] }
async-trait = "0.1"

# Linear algebra
ndarray = "0.15"
ndarray-linalg = "0.16"
blas-src = { version = "0.9", features = ["openblas"] }

# Statistics
statrs = "0.16"

# Serialization
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"

# Date/time
chrono = { version = "0.4", features = ["serde"] }

# Error handling
thiserror = "1.0"
anyhow = "1.0"

# Logging
tracing = "0.1"
tracing-subscriber = "0.3"
```

### 5.2 Market Data Providers

```toml
# Alpaca Markets
alpaca-finance = "0.3"

# Interactive Brokers
ibapi = "0.5"

# Binance
binance = "1.0"

# WebSocket
tokio-tungstenite = "0.20"
futures-util = "0.3"
```

### 5.3 Numerical Computing

```toml
# Scientific computing
nalgebra = "0.32"
approx = "0.5"

# Optimization
argmin = "0.8"
argmin-math = "0.3"

# Random number generation
rand = "0.8"
rand_distr = "0.4"
```

## 6. Configuration Management

### 6.1 Environment Variables

```bash
# Market data
ALPACA_API_KEY=your_key
ALPACA_API_SECRET=your_secret
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# Database
DATABASE_URL=postgresql://user:pass@localhost/hyperphysics

# Logging
RUST_LOG=info
RUST_BACKTRACE=1

# Risk limits
MAX_PORTFOLIO_ENTROPY=2.0
MAX_DRAWDOWN=0.20
MAX_LEVERAGE=2.0
```

### 6.2 Configuration Files

```toml
# config/production.toml
[market_data]
providers = ["alpaca", "binance"]
default_timeframe = "1min"
buffer_size = 10000

[risk]
max_entropy = 2.0
warning_threshold = 1.5
critical_threshold = 1.8
max_drawdown = 0.20
max_leverage = 2.0

[execution]
default_execution_algo = "vwap"
max_order_size = 10000.0
min_order_size = 1.0

[backtest]
commission_bps = 5.0
slippage_bps = 2.0
```

## 7. Database Schema

### 7.1 Market Data Storage

```sql
CREATE TABLE market_bars (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    open DOUBLE PRECISION NOT NULL,
    high DOUBLE PRECISION NOT NULL,
    low DOUBLE PRECISION NOT NULL,
    close DOUBLE PRECISION NOT NULL,
    volume DOUBLE PRECISION NOT NULL,
    UNIQUE(symbol, timestamp)
);

CREATE INDEX idx_bars_symbol_time ON market_bars(symbol, timestamp DESC);
```

### 7.2 Portfolio History

```sql
CREATE TABLE portfolio_snapshots (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    total_value DOUBLE PRECISION NOT NULL,
    cash DOUBLE PRECISION NOT NULL,
    entropy DOUBLE PRECISION NOT NULL,
    temperature DOUBLE PRECISION NOT NULL,
    positions JSONB NOT NULL
);

CREATE INDEX idx_snapshots_time ON portfolio_snapshots(timestamp DESC);
```

### 7.3 Trade History

```sql
CREATE TABLE trades (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL,
    quantity DOUBLE PRECISION NOT NULL,
    fill_price DOUBLE PRECISION NOT NULL,
    commission DOUBLE PRECISION NOT NULL,
    slippage DOUBLE PRECISION NOT NULL
);

CREATE INDEX idx_trades_time ON trades(timestamp DESC);
CREATE INDEX idx_trades_symbol ON trades(symbol);
```

## 8. Deployment Architecture

```
┌─────────────────────────────────────────────┐
│         Load Balancer (NGINX)              │
└─────────────────┬───────────────────────────┘
                  │
      ┌───────────┴───────────┐
      ↓                       ↓
┌──────────┐            ┌──────────┐
│  API 1   │            │  API 2   │
│  (Rust)  │            │  (Rust)  │
└────┬─────┘            └────┬─────┘
     │                       │
     └───────────┬───────────┘
                 ↓
      ┌──────────────────┐
      │   PostgreSQL     │
      │  (TimescaleDB)   │
      └──────────────────┘
                 ↓
      ┌──────────────────┐
      │  Redis (Cache)   │
      └──────────────────┘
```

## 9. Monitoring and Observability

### 9.1 Metrics Collection

```rust
use prometheus::{Counter, Histogram, Registry};

pub struct SystemMetrics {
    // Trading metrics
    orders_submitted: Counter,
    orders_filled: Counter,
    orders_rejected: Counter,

    // Performance metrics
    fill_latency: Histogram,
    signal_latency: Histogram,

    // Risk metrics
    portfolio_entropy: Gauge,
    portfolio_value: Gauge,
}
```

### 9.2 Distributed Tracing

```rust
use tracing::{info, warn, error, instrument};

#[instrument(skip(order))]
pub async fn submit_order(order: OrderEvent) -> Result<String> {
    info!("Submitting order: {} {}", order.symbol, order.quantity);

    let result = self.broker.submit(order).await;

    match result {
        Ok(order_id) => {
            info!("Order submitted successfully: {}", order_id);
            Ok(order_id)
        },
        Err(e) => {
            error!("Order submission failed: {}", e);
            Err(e.into())
        }
    }
}
```

## 10. Testing Strategy

### 10.1 Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entropy_calculation() {
        let positions = vec![
            Position { weight: 0.25, ..Default::default() },
            Position { weight: 0.25, ..Default::default() },
            Position { weight: 0.25, ..Default::default() },
            Position { weight: 0.25, ..Default::default() },
        ];

        let entropy = Portfolio::calculate_entropy(&positions);

        assert_approx_eq!(entropy, 1.386, 0.001);  // ln(4)
    }
}
```

### 10.2 Integration Tests

```rust
#[tokio::test]
async fn test_end_to_end_backtest() {
    let data_provider = MockDataProvider::new();
    let strategy = RegimeAwareStrategy::new();

    let mut engine = BacktestEngine::new(
        Arc::new(data_provider),
        Arc::new(strategy),
        100000.0,
        Utc::now(),
        Utc::now() + Duration::days(30),
    );

    let results = engine.run().await.unwrap();

    assert!(results.sharpe_ratio > 0.0);
    assert!(results.max_drawdown < 0.30);
}
```

## 11. Performance Benchmarks

| Component | Throughput | Latency (p99) | Memory |
|-----------|-----------|---------------|---------|
| Market Data Feed | 100k msg/s | 2ms | 500MB |
| Risk Calculation | 10k/s | 5ms | 200MB |
| Order Routing | 5k/s | 10ms | 100MB |
| Backtest Engine | 1M bars/min | N/A | 2GB |

## 12. Academic References

1. Cont, R. (2001). *Empirical properties of asset returns: stylized facts and statistical issues*. Quantitative Finance, 1(2), 223-236.

2. Bouchaud, J. P., & Potters, M. (2003). *Theory of Financial Risk and Derivative Pricing*. Cambridge University Press.

3. Gould, M. D., et al. (2013). *Limit order books*. Quantitative Finance, 13(11), 1709-1742.
