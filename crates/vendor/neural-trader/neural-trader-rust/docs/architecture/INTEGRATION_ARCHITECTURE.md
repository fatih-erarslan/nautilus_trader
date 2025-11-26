# Strategy Integration Architecture - Agent 5

**Version**: 1.0.0
**Date**: 2025-11-12
**Owner**: Agent 5 (Strategy Integration)

## Overview

This document defines the integration architecture for connecting trading strategies with broker APIs, neural forecasting models, risk management, and backtesting infrastructure.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Trading System                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐      ┌──────────────┐                   │
│  │ Market Data  │─────▶│  Strategies  │                   │
│  │  (Agent 3)   │      │  (Agent 5)   │                   │
│  └──────────────┘      └──────┬───────┘                   │
│         │                     │                            │
│         │                     ▼                            │
│         │              ┌──────────────┐                   │
│         │              │    Neural    │                   │
│         │              │  Forecasting │                   │
│         │              │  (Agent 4)   │                   │
│         │              └──────┬───────┘                   │
│         │                     │                            │
│         │                     ▼                            │
│         │              ┌──────────────┐                   │
│         └─────────────▶│     Risk     │                   │
│                        │  Management  │                   │
│                        │  (Agent 6)   │                   │
│                        └──────┬───────┘                   │
│                               │                            │
│                               ▼                            │
│                        ┌──────────────┐                   │
│                        │   Order      │                   │
│                        │  Execution   │                   │
│                        │  (Agent 3)   │                   │
│                        └──────────────┘                   │
└─────────────────────────────────────────────────────────────┘
```

## Core Contracts

### 1. BrokerClient Trait (Agent 3 ↔ Agent 5)

**Purpose**: Abstract broker-specific APIs for order execution and market data

**Contract Definition**:
```rust
// Location: crates/execution/src/broker.rs

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use crate::{OrderRequest, OrderResponse, OrderUpdate, Result};

/// Broker client interface for order execution
#[async_trait]
pub trait BrokerClient: Send + Sync {
    /// Place an order
    ///
    /// Target latency: <10ms
    async fn place_order(&self, request: OrderRequest) -> Result<OrderResponse>;

    /// Cancel an existing order
    async fn cancel_order(&self, order_id: &str) -> Result<()>;

    /// Get order status
    async fn get_order(&self, order_id: &str) -> Result<OrderResponse>;

    /// List all orders
    async fn list_orders(&self, symbol: Option<&str>) -> Result<Vec<OrderResponse>>;

    /// Subscribe to order updates (WebSocket)
    async fn subscribe_order_updates(
        &self,
    ) -> Result<tokio::sync::mpsc::Receiver<OrderUpdate>>;

    /// Get current account info
    async fn get_account(&self) -> Result<AccountInfo>;

    /// Get current positions
    async fn get_positions(&self) -> Result<Vec<Position>>;

    /// Get market data for symbol
    async fn get_market_data(&self, symbol: &str) -> Result<MarketData>;

    /// Subscribe to market data updates (WebSocket)
    async fn subscribe_market_data(
        &self,
        symbols: &[String],
    ) -> Result<tokio::sync::mpsc::Receiver<MarketData>>;
}

/// Account information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccountInfo {
    pub account_id: String,
    pub cash: Decimal,
    pub portfolio_value: Decimal,
    pub buying_power: Decimal,
    pub equity: Decimal,
    pub last_equity: Decimal,
    pub multiplier: f64,
    pub initial_margin: Decimal,
    pub maintenance_margin: Decimal,
    pub pattern_day_trader: bool,
    pub trading_blocked: bool,
    pub transfers_blocked: bool,
    pub account_blocked: bool,
    pub created_at: DateTime<Utc>,
}

/// Position information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    pub symbol: String,
    pub quantity: i32,
    pub side: PositionSide,
    pub avg_entry_price: Decimal,
    pub current_price: Decimal,
    pub market_value: Decimal,
    pub cost_basis: Decimal,
    pub unrealized_pl: Decimal,
    pub unrealized_plpc: Decimal,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PositionSide {
    Long,
    Short,
}
```

**Implementation Requirements (Agent 3)**:
- [ ] Alpaca broker client
- [ ] Interactive Brokers client
- [ ] Mock broker for testing
- [ ] Connection pooling
- [ ] Rate limiting
- [ ] Error retry logic

### 2. NeuralForecast Trait (Agent 4 ↔ Agent 5)

**Purpose**: Provide neural network forecasting and sentiment analysis

**Contract Definition**:
```rust
// Location: crates/neural/src/forecast.rs

use async_trait::async_trait;
use crate::Result;

/// Neural forecasting interface
#[async_trait]
pub trait NeuralForecast: Send + Sync {
    /// Predict future prices using NHITS or LSTM
    ///
    /// Target latency: <100ms with GPU
    async fn predict_prices(
        &self,
        symbol: &str,
        historical_prices: &[f64],
        horizon: usize,
    ) -> Result<NeuralForecastResult>;

    /// Predict trend direction and confidence
    async fn predict_trend(
        &self,
        symbol: &str,
        features: &[f64],
    ) -> Result<TrendPrediction>;

    /// Analyze news sentiment
    async fn analyze_sentiment(
        &self,
        symbol: &str,
        news_items: &[String],
    ) -> Result<SentimentAnalysis>;

    /// Detect arbitrage opportunities across markets
    async fn detect_arbitrage(
        &self,
        symbol: &str,
        market_prices: &[(String, f64)], // (market_id, price)
    ) -> Result<Option<ArbitrageOpportunity>>;

    /// Get model performance metrics
    async fn get_model_metrics(&self, model_id: &str) -> Result<ModelMetrics>;
}

/// Neural forecast result
#[derive(Debug, Clone)]
pub struct NeuralForecastResult {
    pub predictions: Vec<f64>,
    pub confidence_intervals: Vec<(f64, f64)>, // (lower, upper)
    pub model_confidence: f64,
    pub features_used: Vec<String>,
    pub inference_time_ms: f64,
}

/// Trend prediction
#[derive(Debug, Clone)]
pub struct TrendPrediction {
    pub direction: Direction, // Long, Short, or Neutral
    pub confidence: f64,
    pub probability_up: f64,
    pub probability_down: f64,
    pub expected_return: f64,
    pub time_horizon_hours: usize,
}

/// Sentiment analysis result
#[derive(Debug, Clone)]
pub struct SentimentAnalysis {
    pub overall_sentiment: f64, // -1.0 (bearish) to 1.0 (bullish)
    pub sentiment_scores: Vec<f64>,
    pub article_count: usize,
    pub source_diversity: f64,
    pub sentiment_momentum: f64, // Change over time
    pub confidence: f64,
}

/// Arbitrage opportunity
#[derive(Debug, Clone)]
pub struct ArbitrageOpportunity {
    pub buy_market: String,
    pub sell_market: String,
    pub buy_price: f64,
    pub sell_price: f64,
    pub expected_profit: f64,
    pub execution_probability: f64,
    pub estimated_slippage: f64,
}

/// Model performance metrics
#[derive(Debug, Clone)]
pub struct ModelMetrics {
    pub model_id: String,
    pub mse: f64,
    pub mae: f64,
    pub r2_score: f64,
    pub sharpe_ratio: f64,
    pub accuracy: f64,
    pub inference_time_ms: f64,
    pub gpu_memory_mb: f64,
}
```

**Implementation Requirements (Agent 4)**:
- [ ] NHITS time-series forecasting
- [ ] LSTM/Transformer trend prediction
- [ ] Sentiment analysis with BERT/FinBERT
- [ ] GPU-accelerated inference
- [ ] Model caching and warm-up
- [ ] News API integration (Alpha Vantage, NewsAPI)

### 3. RiskManager Trait (Agent 6 ↔ Agent 5)

**Purpose**: Validate signals against risk limits and calculate position sizes

**Contract Definition**:
```rust
// Location: crates/risk/src/manager.rs

use async_trait::async_trait;
use rust_decimal::Decimal;
use crate::{Signal, Position, Result};

/// Risk management interface
#[async_trait]
pub trait RiskManager: Send + Sync {
    /// Validate signal against risk limits
    ///
    /// Returns: Ok(adjusted_signal) or Err if signal violates limits
    async fn validate_signal(
        &self,
        signal: &Signal,
        portfolio: &Portfolio,
    ) -> Result<Signal>;

    /// Calculate position size using Kelly Criterion
    async fn calculate_position_size(
        &self,
        signal: &Signal,
        portfolio: &Portfolio,
    ) -> Result<u32>;

    /// Check portfolio-level risk limits
    async fn check_portfolio_risk(
        &self,
        portfolio: &Portfolio,
    ) -> Result<RiskAssessment>;

    /// Calculate Value at Risk (VaR)
    async fn calculate_var(
        &self,
        portfolio: &Portfolio,
        confidence: f64,
        horizon_days: usize,
    ) -> Result<Decimal>;

    /// Calculate Conditional Value at Risk (CVaR)
    async fn calculate_cvar(
        &self,
        portfolio: &Portfolio,
        confidence: f64,
        horizon_days: usize,
    ) -> Result<Decimal>;

    /// Adjust stop-loss dynamically based on volatility
    async fn adjust_stop_loss(
        &self,
        position: &Position,
        current_price: Decimal,
        volatility: f64,
    ) -> Result<Decimal>;

    /// Get risk metrics for portfolio
    async fn get_risk_metrics(
        &self,
        portfolio: &Portfolio,
    ) -> Result<RiskMetrics>;
}

/// Portfolio state
#[derive(Debug, Clone)]
pub struct Portfolio {
    pub account_id: String,
    pub cash: Decimal,
    pub equity: Decimal,
    pub positions: Vec<Position>,
    pub open_orders: Vec<OrderResponse>,
    pub daily_pnl: Decimal,
    pub total_pnl: Decimal,
    pub max_drawdown: Decimal,
}

/// Risk assessment result
#[derive(Debug, Clone)]
pub struct RiskAssessment {
    pub risk_level: RiskLevel,
    pub exposure_pct: f64,
    pub leverage: f64,
    pub concentration_risk: f64,
    pub correlation_risk: f64,
    pub warnings: Vec<String>,
    pub can_trade: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

/// Portfolio risk metrics
#[derive(Debug, Clone)]
pub struct RiskMetrics {
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
    pub max_drawdown: Decimal,
    pub current_drawdown: Decimal,
    pub var_95: Decimal,
    pub cvar_95: Decimal,
    pub beta: f64,
    pub alpha: f64,
    pub volatility: f64,
    pub correlation_to_market: f64,
}
```

**Implementation Requirements (Agent 6)**:
- [ ] Kelly Criterion position sizing
- [ ] VaR/CVaR Monte Carlo simulation
- [ ] Portfolio correlation analysis
- [ ] Dynamic stop-loss adjustment
- [ ] Drawdown monitoring
- [ ] Exposure limit enforcement

### 4. BacktestEngine (Agent 5 Internal)

**Purpose**: Validate strategies against historical data

**Contract Definition**:
```rust
// Location: crates/backtesting/src/engine.rs

use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use std::collections::HashMap;

/// Backtesting engine
pub struct BacktestEngine {
    strategies: Vec<Box<dyn Strategy>>,
    broker: Box<dyn BrokerClient>,
    risk_manager: Box<dyn RiskManager>,
    initial_capital: Decimal,
    commission_rate: f64,
    slippage_model: SlippageModel,
}

impl BacktestEngine {
    /// Run backtest on historical data
    ///
    /// Target: >10x faster than Python
    pub async fn run(
        &mut self,
        start_date: DateTime<Utc>,
        end_date: DateTime<Utc>,
        symbols: &[String],
    ) -> Result<BacktestResult>;

    /// Run walk-forward optimization
    pub async fn walk_forward(
        &mut self,
        start_date: DateTime<Utc>,
        end_date: DateTime<Utc>,
        train_window_days: usize,
        test_window_days: usize,
    ) -> Result<WalkForwardResult>;
}

/// Backtest result
#[derive(Debug, Clone)]
pub struct BacktestResult {
    pub strategy_id: String,
    pub start_date: DateTime<Utc>,
    pub end_date: DateTime<Utc>,
    pub initial_capital: Decimal,
    pub final_capital: Decimal,
    pub total_return: f64,
    pub annualized_return: f64,
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
    pub max_drawdown: f64,
    pub win_rate: f64,
    pub profit_factor: f64,
    pub total_trades: usize,
    pub winning_trades: usize,
    pub losing_trades: usize,
    pub avg_win: Decimal,
    pub avg_loss: Decimal,
    pub largest_win: Decimal,
    pub largest_loss: Decimal,
    pub equity_curve: Vec<(DateTime<Utc>, Decimal)>,
    pub trades: Vec<Trade>,
    pub benchmark_return: f64, // S&P 500
    pub alpha: f64,
    pub beta: f64,
}

/// Slippage model
#[derive(Debug, Clone, Copy)]
pub enum SlippageModel {
    Fixed(f64),              // Fixed percentage
    VolumeDependent(f64),    // Based on order size vs volume
    BidAskSpread(f64),       // Based on spread
}

/// Trade record
#[derive(Debug, Clone)]
pub struct Trade {
    pub entry_time: DateTime<Utc>,
    pub exit_time: DateTime<Utc>,
    pub symbol: String,
    pub side: OrderSide,
    pub entry_price: Decimal,
    pub exit_price: Decimal,
    pub quantity: u32,
    pub pnl: Decimal,
    pub return_pct: f64,
    pub commission: Decimal,
    pub slippage: Decimal,
    pub holding_period_hours: f64,
}
```

**Implementation Requirements (Agent 5)**:
- [ ] Historical data loader (CSV/Parquet)
- [ ] Simulated order execution
- [ ] Slippage and commission modeling
- [ ] Performance metrics calculation
- [ ] Equity curve generation
- [ ] Walk-forward validation
- [ ] Benchmark comparison

## Data Flow

### Signal Generation Flow

```rust
// Example: Mean Reversion Strategy

async fn generate_signals() -> Result<Vec<Signal>> {
    // 1. Get market data from Agent 3
    let market_data = broker_client.get_market_data("AAPL").await?;

    // 2. Strategy processes data
    let strategy = MeanReversionStrategy::new(
        vec!["AAPL".to_string()],
        20, // period
        2.0, // num_std
        14, // rsi_period
    );

    let portfolio = risk_manager.get_portfolio().await?;
    let mut signals = strategy.process(&market_data, &portfolio).await?;

    // 3. Enhance with neural forecast (Agent 4)
    for signal in &mut signals {
        let forecast = neural_forecast
            .predict_prices(
                &signal.symbol,
                &market_data.historical_prices(),
                24, // 24-hour horizon
            )
            .await?;

        // Adjust confidence based on neural prediction
        signal.confidence *= forecast.model_confidence;
    }

    // 4. Validate with risk manager (Agent 6)
    let validated_signals = Vec::new();
    for signal in signals {
        match risk_manager.validate_signal(&signal, &portfolio).await {
            Ok(validated) => validated_signals.push(validated),
            Err(e) => warn!("Signal rejected: {}", e),
        }
    }

    // 5. Execute orders (Agent 3)
    for signal in &validated_signals {
        let quantity = risk_manager
            .calculate_position_size(&signal, &portfolio)
            .await?;

        let order = OrderRequest {
            symbol: signal.symbol.clone(),
            side: match signal.direction {
                Direction::Long => OrderSide::Buy,
                Direction::Short => OrderSide::Sell,
                Direction::Close => OrderSide::Sell, // Close existing
            },
            order_type: OrderType::Market,
            quantity,
            limit_price: None,
            stop_price: signal.stop_loss,
            time_in_force: TimeInForce::Day,
        };

        let response = broker_client.place_order(order).await?;
        info!("Order placed: {:?}", response);
    }

    Ok(validated_signals)
}
```

### Backtesting Flow

```rust
// Example: Backtest Mean Reversion

async fn run_backtest() -> Result<BacktestResult> {
    let strategy = Box::new(MeanReversionStrategy::new(
        vec!["AAPL".to_string(), "GOOGL".to_string()],
        20,
        2.0,
        14,
    ));

    let mock_broker = Box::new(MockBroker::new());
    let risk_manager = Box::new(RiskManagerImpl::new());

    let mut engine = BacktestEngine::new(
        vec![strategy],
        mock_broker,
        risk_manager,
        Decimal::from(100_000), // $100k initial capital
        0.001, // 0.1% commission
        SlippageModel::Fixed(0.0005), // 0.05% slippage
    );

    let result = engine.run(
        Utc.ymd(2020, 1, 1).and_hms(0, 0, 0),
        Utc.ymd(2025, 1, 1).and_hms(0, 0, 0),
        &["AAPL".to_string(), "GOOGL".to_string()],
    ).await?;

    println!("Backtest Results:");
    println!("  Total Return: {:.2}%", result.total_return * 100.0);
    println!("  Sharpe Ratio: {:.2}", result.sharpe_ratio);
    println!("  Max Drawdown: {:.2}%", result.max_drawdown * 100.0);
    println!("  Win Rate: {:.2}%", result.win_rate * 100.0);
    println!("  Total Trades: {}", result.total_trades);

    Ok(result)
}
```

## Shared Components

### 1. Indicator Library

**Purpose**: Avoid duplicate indicator implementations across strategies

**Location**: `crates/indicators/src/`

```rust
pub mod technical {
    use statrs::statistics::Statistics;

    /// Simple Moving Average
    pub fn sma(prices: &[f64], period: usize) -> Vec<f64> {
        prices
            .windows(period)
            .map(|w| w.mean())
            .collect()
    }

    /// Exponential Moving Average
    pub fn ema(prices: &[f64], period: usize) -> Vec<f64> {
        let multiplier = 2.0 / (period as f64 + 1.0);
        let mut ema_values = Vec::with_capacity(prices.len());
        let mut ema = prices[0];

        for &price in prices {
            ema = (price - ema) * multiplier + ema;
            ema_values.push(ema);
        }

        ema_values
    }

    /// Relative Strength Index
    pub fn rsi(prices: &[f64], period: usize) -> Vec<f64> {
        // Implementation...
    }

    /// Bollinger Bands
    pub fn bollinger_bands(
        prices: &[f64],
        period: usize,
        num_std: f64,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        // Returns (upper, middle, lower)
        // Implementation...
    }

    /// MACD
    pub fn macd(
        prices: &[f64],
        fast_period: usize,
        slow_period: usize,
        signal_period: usize,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        // Returns (macd_line, signal_line, histogram)
        // Implementation...
    }
}

pub mod statistical {
    /// Z-score normalization
    pub fn zscore(values: &[f64]) -> Vec<f64> {
        // Implementation...
    }

    /// Rolling correlation
    pub fn rolling_correlation(
        series_a: &[f64],
        series_b: &[f64],
        window: usize,
    ) -> Vec<f64> {
        // Implementation...
    }

    /// Cointegration test (Engle-Granger)
    pub fn cointegration_test(
        series_a: &[f64],
        series_b: &[f64],
    ) -> (bool, f64, f64) {
        // Returns (is_cointegrated, test_statistic, p_value)
        // Implementation...
    }
}
```

### 2. Performance Monitoring

**Location**: `crates/monitoring/src/`

```rust
use prometheus::{Counter, Histogram, Registry};

pub struct PerformanceMonitor {
    signal_generation_time: Histogram,
    order_execution_time: Histogram,
    neural_inference_time: Histogram,
    signals_generated: Counter,
    orders_placed: Counter,
    orders_filled: Counter,
    orders_rejected: Counter,
}

impl PerformanceMonitor {
    pub fn record_signal_generation(&self, duration_ms: f64) {
        self.signal_generation_time.observe(duration_ms);
        self.signals_generated.inc();
    }

    pub fn record_order_execution(&self, duration_ms: f64) {
        self.order_execution_time.observe(duration_ms);
        self.orders_placed.inc();
    }

    pub fn export_metrics(&self) -> String {
        // Export Prometheus metrics
    }
}
```

## Error Handling

### Error Types

```rust
#[derive(Debug, thiserror::Error)]
pub enum TradingError {
    #[error("Broker error: {0}")]
    Broker(String),

    #[error("Risk limit exceeded: {0}")]
    RiskLimit(String),

    #[error("Insufficient data: needed {needed}, got {available}")]
    InsufficientData { needed: usize, available: usize },

    #[error("Neural inference failed: {0}")]
    NeuralInference(String),

    #[error("Order timeout")]
    Timeout,

    #[error("Invalid configuration: {0}")]
    Config(String),
}
```

## Testing Strategy

### 1. Unit Tests
- Each strategy tested in isolation
- Mock BrokerClient for deterministic tests
- Mock NeuralForecast with fixed predictions
- Test edge cases (insufficient data, extreme values)

### 2. Integration Tests
- Full pipeline: MarketData → Strategy → Risk → Execution
- Test with real market data samples
- Verify order routing and fills
- Test risk rejections

### 3. Performance Tests
- Benchmark signal generation (<15ms target)
- Benchmark order execution (<10ms target)
- Benchmark neural inference (<100ms target)
- Memory profiling (<500MB target)

### 4. Backtest Validation
- Compare Rust vs Python performance
- Verify metrics match (Sharpe, Sortino, etc.)
- Test on 5+ years of historical data
- Validate against known market regimes

## Deployment Architecture

### Development
```
Developer Machine
├── Cargo workspace
├── Mock brokers
├── Local neural models (CPU)
└── SQLite for backtesting
```

### Staging
```
Cloud VM (AWS/GCP)
├── Paper trading with Alpaca
├── GPU for neural inference
├── PostgreSQL for backtesting
├── Prometheus + Grafana monitoring
└── CI/CD with GitHub Actions
```

### Production
```
High-performance server
├── Live trading with multiple brokers
├── GPU cluster for neural inference
├── TimescaleDB for tick data
├── Distributed risk management
├── Real-time alerting
└── Disaster recovery
```

## Performance Targets

| Component | Target | Measurement |
|-----------|--------|-------------|
| Signal Generation | <15ms | 95th percentile |
| Order Placement | <10ms | 95th percentile |
| Neural Inference | <100ms | Average |
| Risk Validation | <5ms | 95th percentile |
| Backtest (5yr) | <2min | Total time |
| Memory Usage | <500MB | Peak |
| CPU Usage | <50% | Average |

## Next Steps

1. **Agent 5**: Implement missing strategies (Market Making, Portfolio Opt, Risk Parity)
2. **Agent 3**: Implement BrokerClient for Alpaca
3. **Agent 4**: Implement NeuralForecast with NHITS/LSTM
4. **Agent 6**: Implement RiskManager with Kelly Criterion
5. **Agent 5**: Build BacktestEngine
6. **All**: Integration testing and benchmarking

---

**Status**: Architecture defined, awaiting implementation coordination
