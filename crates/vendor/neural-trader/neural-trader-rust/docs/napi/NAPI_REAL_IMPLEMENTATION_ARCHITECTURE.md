# NAPI Real Implementation Architecture

**Author**: System Architecture Designer
**Date**: 2025-11-14
**Version**: 1.0
**Status**: Design Review

---

## Executive Summary

This document provides a comprehensive architecture for replacing all 103 simulation functions in `mcp_tools.rs` with real Rust implementations. The architecture leverages 35+ existing `nt-*` crates to provide production-grade trading functionality with GPU acceleration, proper error handling, and dependency injection.

### Key Metrics
- **Total Functions**: 103 NAPI exports
- **Target Crates**: 12 primary crates
- **Implementation Phases**: 4 phases over ~8-12 weeks
- **Risk Level**: Medium (careful dependency management required)
- **Performance Improvement**: 10-100x (GPU-accelerated operations)

---

## 1. Function → Crate Mapping (All 103 Functions)

### 1.1 Core Trading Tools (23 functions) → `nt-strategies`, `nt-execution`, `nt-portfolio`

| Function | Target Crate | Implementation Component | Priority |
|----------|--------------|-------------------------|----------|
| `ping()` | `nt-core` | System health check | P0 |
| `list_strategies()` | `nt-strategies` | `StrategyOrchestrator::list_strategies()` | P0 |
| `get_strategy_info()` | `nt-strategies` | `StrategyConfig` metadata | P0 |
| `execute_trade()` | `nt-execution` | `BrokerClient::execute_order()` | P1 |
| `simulate_trade()` | `nt-strategies` | `BacktestEngine::simulate_single()` | P1 |
| `quick_analysis()` | `nt-strategies` | Technical indicator calculation | P1 |
| `run_backtest()` | `nt-strategies` | `BacktestEngine::run()` | P1 |
| `optimize_strategy()` | `nt-benchoptimizer` | Genetic algorithm optimizer | P2 |
| `risk_analysis()` | `nt-risk` | `MonteCarloVaR`, `StressTest` | P1 |
| `get_portfolio_status()` | `nt-portfolio` | `PortfolioTracker::get_status()` | P0 |
| `get_market_analysis()` | `nt-strategies` | Multi-indicator analysis | P2 |
| `get_market_status()` | `nt-execution` | Market hours/session info | P2 |
| `performance_report()` | `nt-strategies` | `PerformanceMetrics` aggregation | P2 |
| `correlation_analysis()` | `nt-risk` | `correlation::calculate_matrix()` | P2 |
| `recommend_strategy()` | `nt-strategies` | ML-based strategy selection | P3 |
| `switch_active_strategy()` | `nt-strategies` | `StrategyOrchestrator::switch()` | P2 |
| `get_strategy_comparison()` | `nt-strategies` | Multi-strategy performance | P2 |
| `adaptive_strategy_selection()` | `nt-strategies` | Auto-strategy selection | P3 |
| `backtest_strategy()` | `nt-strategies` | Alias for `run_backtest()` | P2 |
| `optimize_parameters()` | `nt-benchoptimizer` | Alias for `optimize_strategy()` | P2 |
| `quick_backtest()` | `nt-strategies` | Simplified backtest | P2 |
| `monte_carlo_simulation()` | `nt-risk` | `MonteCarloVaR::calculate()` | P2 |
| `run_benchmark()` | `nt-benchoptimizer` | Performance benchmarking | P3 |

### 1.2 Neural Network Tools (7 functions) → `nt-neural`

| Function | Implementation Component | GPU Support | Priority |
|----------|-------------------------|-------------|----------|
| `neural_forecast()` | `Forecaster::predict()` | ✅ Candle+CUDA | P1 |
| `neural_train()` | `Trainer::train_model()` | ✅ Candle+CUDA | P1 |
| `neural_evaluate()` | `Evaluator::evaluate()` | ✅ Candle+CUDA | P2 |
| `neural_backtest()` | `BacktestEngine` + predictions | ✅ Candle+CUDA | P2 |
| `neural_model_status()` | Model registry + metadata | ❌ | P2 |
| `neural_optimize()` | Hyperparameter tuning (Optuna-style) | ✅ Candle+CUDA | P3 |
| `neural_predict()` | Generic prediction interface | ✅ Candle+CUDA | P1 |

**Technical Notes**:
- Use `candle-core` + `candle-nn` for neural operations
- CUDA support via `cudarc` feature flag
- Model persistence via `safetensors`
- AgentDB integration for model versioning

### 1.3 News Trading Tools (8 functions) → `nt-news-trading`

| Function | Implementation Component | Priority |
|----------|-------------------------|----------|
| `analyze_news()` | `NewsAnalyzer::analyze_sentiment()` | P2 |
| `get_news_sentiment()` | `SentimentEngine::calculate()` | P2 |
| `control_news_collection()` | `NewsCollector` lifecycle | P2 |
| `get_news_provider_status()` | Provider health monitoring | P3 |
| `fetch_filtered_news()` | `NewsCollector::fetch_filtered()` | P2 |
| `get_news_trends()` | Temporal sentiment analysis | P3 |
| `get_breaking_news()` | Real-time news stream | P3 |
| `analyze_news_impact()` | Price impact estimation | P3 |

**Implementation Strategy**:
- Use `sled` database for news storage
- Multi-provider support (Reuters, Bloomberg via APIs)
- Sentiment scoring using rule-based + ML models

### 1.4 Portfolio & Risk Tools (5 functions) → `nt-risk`, `nt-portfolio`

| Function | Target Crate | Component | GPU | Priority |
|----------|--------------|-----------|-----|----------|
| `execute_multi_asset_trade()` | `nt-execution` | Batch order execution | ❌ | P2 |
| `portfolio_rebalance()` | `nt-portfolio` | Rebalancing optimizer | ❌ | P2 |
| `cross_asset_correlation_matrix()` | `nt-risk` | Correlation calculator | ✅ | P2 |
| `get_execution_analytics()` | `nt-execution` | Execution metrics | ❌ | P3 |
| `get_system_metrics()` | `nt-core` | System monitoring | ❌ | P3 |

### 1.5 Sports Betting Tools (13 functions) → `nt-sports-betting`

| Function | Implementation Component | Priority |
|----------|-------------------------|----------|
| `get_sports_events()` | `OddsAPIClient::fetch_events()` | P2 |
| `get_sports_odds()` | `OddsAPIClient::fetch_odds()` | P2 |
| `find_sports_arbitrage()` | `ArbitrageDetector::find_opportunities()` | P2 |
| `analyze_betting_market_depth()` | Market depth analysis | P3 |
| `calculate_kelly_criterion()` | `kelly::calculate()` from `nt-risk` | P1 |
| `simulate_betting_strategy()` | Monte Carlo betting sim | P3 |
| `get_betting_portfolio_status()` | Betting portfolio tracker | P2 |
| `execute_sports_bet()` | Bet placement engine | P2 |
| `get_sports_betting_performance()` | Performance analytics | P3 |
| `compare_betting_providers()` | Provider comparison | P3 |
| `get_live_odds_updates()` | WebSocket live odds | P3 |
| `analyze_betting_trends()` | Trend analysis | P3 |
| `get_betting_history()` | Historical bet tracking | P3 |

**Dependencies**:
- The Odds API integration
- Kelly Criterion from `nt-risk`
- Syndicate integration for group betting

### 1.6 Odds API Tools (9 functions) → `nt-sports-betting` (OddsAPI module)

| Function | Implementation | Priority |
|----------|----------------|----------|
| `odds_api_get_sports()` | `OddsAPIClient::get_sports()` | P2 |
| `odds_api_get_live_odds()` | `OddsAPIClient::get_live_odds()` | P2 |
| `odds_api_get_event_odds()` | `OddsAPIClient::get_event_odds()` | P2 |
| `odds_api_find_arbitrage()` | `ArbitrageDetector::detect()` | P2 |
| `odds_api_get_bookmaker_odds()` | Bookmaker-specific odds | P3 |
| `odds_api_analyze_movement()` | Odds movement tracking | P3 |
| `odds_api_calculate_probability()` | Implied probability calc | P1 |
| `odds_api_compare_margins()` | Margin comparison | P3 |
| `odds_api_get_upcoming()` | Upcoming events filter | P2 |

### 1.7 Prediction Markets (5 functions) → `nt-prediction-markets`

| Function | Implementation | Priority |
|----------|----------------|----------|
| `get_prediction_markets()` | Market listing API | P3 |
| `analyze_market_sentiment()` | Sentiment analysis | P3 |
| `get_market_orderbook()` | Orderbook fetching | P3 |
| `place_prediction_order()` | Order placement | P3 |
| `get_prediction_positions()` | Position tracking | P3 |
| `calculate_expected_value()` | EV calculation | P2 |

### 1.8 Syndicates (15 functions) → `nt-syndicate`

| Function | Implementation Component | Priority |
|----------|-------------------------|----------|
| `create_syndicate()` | `SyndicateManager::create()` | P2 |
| `add_syndicate_member()` | Member management | P2 |
| `get_syndicate_status()` | Status tracking | P2 |
| `allocate_syndicate_funds()` | Fund allocation engine | P2 |
| `distribute_syndicate_profits()` | Profit distribution (Kelly/proportional) | P2 |
| `process_syndicate_withdrawal()` | Withdrawal processing | P2 |
| `get_syndicate_member_performance()` | Performance tracking | P3 |
| `create_syndicate_vote()` | Voting system | P3 |
| `cast_syndicate_vote()` | Vote casting | P3 |
| `get_syndicate_allocation_limits()` | Limit management | P2 |
| `update_syndicate_member_contribution()` | Contribution updates | P2 |
| `get_syndicate_profit_history()` | Historical tracking | P3 |
| `simulate_syndicate_allocation()` | Allocation simulation | P3 |
| `get_syndicate_withdrawal_history()` | Withdrawal history | P3 |
| `update_syndicate_allocation_strategy()` | Strategy updates | P3 |
| `get_syndicate_member_list()` | Member listing | P2 |
| `calculate_syndicate_tax_liability()` | Tax calculations | P3 |

**Technical Requirements**:
- `DashMap` for concurrent member access
- Integration with `nt-risk` for Kelly Criterion
- Persistent storage via `sled` or PostgreSQL

### 1.9 E2B Cloud (9 functions) → `nt-e2b-integration`

| Function | Implementation | Priority |
|----------|----------------|----------|
| `create_e2b_sandbox()` | E2B SDK wrapper | P3 |
| `run_e2b_agent()` | Agent deployment | P3 |
| `execute_e2b_process()` | Process execution | P3 |
| `list_e2b_sandboxes()` | Sandbox listing | P3 |
| `terminate_e2b_sandbox()` | Sandbox cleanup | P3 |
| `get_e2b_sandbox_status()` | Status monitoring | P3 |
| `deploy_e2b_template()` | Template deployment | P3 |
| `scale_e2b_deployment()` | Scaling management | P3 |
| `monitor_e2b_health()` | Health monitoring | P3 |
| `export_e2b_template()` | Template export | P3 |

**Note**: Lower priority - cloud execution is optional feature.

### 1.10 System & Monitoring (5 functions) → `nt-core`, `nt-utils`

| Function | Implementation | Priority |
|----------|----------------|----------|
| `monitor_strategy_health()` | Strategy health checks | P2 |
| `get_token_usage()` | Token tracking (for LLM costs) | P3 |
| `analyze_bottlenecks()` | Performance profiling | P3 |
| `get_health_status()` | System health | P1 |
| `get_api_latency()` | Latency monitoring | P3 |

---

## 2. Dependency Injection Architecture

### 2.1 Design Pattern: Service Locator + Builder

```rust
// File: napi-bindings/src/services/mod.rs

use std::sync::Arc;
use parking_lot::RwLock;
use once_cell::sync::Lazy;

/// Global service container for dependency injection
pub struct ServiceContainer {
    // Trading services
    pub broker_client: Arc<dyn BrokerClient>,
    pub portfolio_tracker: Arc<RwLock<PortfolioTracker>>,
    pub risk_manager: Arc<RiskManager>,

    // Neural services
    pub neural_engine: Arc<NeuralEngine>,
    pub model_registry: Arc<RwLock<ModelRegistry>>,

    // Market data
    pub market_data_provider: Arc<dyn MarketDataProvider>,

    // News services
    pub news_collector: Arc<NewsCollector>,

    // Sports betting
    pub odds_api_client: Arc<OddsAPIClient>,
    pub syndicate_manager: Arc<SyndicateManager>,

    // System
    pub config: Arc<SystemConfig>,
}

/// Lazy-initialized global singleton
static SERVICES: Lazy<RwLock<Option<ServiceContainer>>> =
    Lazy::new(|| RwLock::new(None));

/// Initialize services (called once at startup)
pub async fn init_services(config: SystemConfig) -> Result<()> {
    let container = ServiceContainer::new(config).await?;
    *SERVICES.write() = Some(container);
    Ok(())
}

/// Get service container (panics if not initialized)
pub fn services() -> impl std::ops::Deref<Target = ServiceContainer> {
    parking_lot::RwLockReadGuard::map(
        SERVICES.read(),
        |opt| opt.as_ref().expect("Services not initialized")
    )
}
```

### 2.2 NAPI Initialization Function

```rust
// File: napi-bindings/src/lib.rs

use napi::bindgen_prelude::*;

#[napi]
pub async fn initialize_neural_trader(config_json: String) -> Result<String> {
    // Parse configuration
    let config: SystemConfig = serde_json::from_str(&config_json)
        .map_err(|e| Error::from_reason(format!("Config parse error: {}", e)))?;

    // Initialize tracing/logging
    init_logging(&config);

    // Initialize service container
    services::init_services(config).await
        .map_err(|e| Error::from_reason(format!("Init error: {}", e)))?;

    Ok(json!({
        "status": "initialized",
        "version": env!("CARGO_PKG_VERSION"),
        "gpu_available": nt_neural::is_gpu_available(),
        "timestamp": Utc::now().to_rfc3339()
    }).to_string())
}

#[napi]
pub async fn shutdown_neural_trader() -> Result<String> {
    // Graceful shutdown
    // - Close broker connections
    // - Save state
    // - Stop background tasks
    Ok(json!({"status": "shutdown"}).to_string())
}
```

### 2.3 Configuration Structure

```rust
// File: napi-bindings/src/config.rs

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemConfig {
    // Broker configuration
    pub broker: BrokerConfig,

    // Market data
    pub market_data: MarketDataConfig,

    // Neural network
    pub neural: NeuralConfig,

    // Risk management
    pub risk: RiskConfig,

    // News trading
    pub news: Option<NewsConfig>,

    // Sports betting
    pub sports: Option<SportsConfig>,

    // System settings
    pub system: SystemSettings,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrokerConfig {
    pub provider: String, // "alpaca", "interactive_brokers", "paper"
    pub api_key: String,
    pub api_secret: String,
    pub base_url: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralConfig {
    pub use_gpu: bool,
    pub device: String, // "cuda", "cpu", "metal"
    pub model_cache_dir: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskConfig {
    pub max_position_size: f64,
    pub var_confidence: f64,
    pub use_gpu: bool,
}
```

---

## 3. Error Handling Strategy

### 3.1 Error Type Hierarchy

```rust
// File: napi-bindings/src/error.rs

use thiserror::Error;

#[derive(Error, Debug)]
pub enum NeuralTraderError {
    #[error("Broker error: {0}")]
    Broker(#[from] nt_execution::Error),

    #[error("Risk management error: {0}")]
    Risk(#[from] nt_risk::RiskError),

    #[error("Neural network error: {0}")]
    Neural(String),

    #[error("Portfolio error: {0}")]
    Portfolio(String),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Not initialized: {0}")]
    NotInitialized(String),

    #[error("Invalid argument: {0}")]
    InvalidArgument(String),
}

/// Convert Rust errors to JSON for NAPI
pub fn to_json_error(err: NeuralTraderError) -> String {
    json!({
        "error": {
            "type": error_type(&err),
            "message": err.to_string(),
            "timestamp": Utc::now().to_rfc3339()
        }
    }).to_string()
}

fn error_type(err: &NeuralTraderError) -> &'static str {
    match err {
        NeuralTraderError::Broker(_) => "BrokerError",
        NeuralTraderError::Risk(_) => "RiskError",
        NeuralTraderError::Neural(_) => "NeuralError",
        NeuralTraderError::Portfolio(_) => "PortfolioError",
        NeuralTraderError::Config(_) => "ConfigError",
        NeuralTraderError::NotInitialized(_) => "NotInitializedError",
        NeuralTraderError::InvalidArgument(_) => "InvalidArgumentError",
    }
}
```

### 3.2 NAPI Error Conversion

```rust
// Helper macro for error conversion
macro_rules! napi_result {
    ($expr:expr) => {
        match $expr {
            Ok(val) => Ok(serde_json::to_string(&val)
                .map_err(|e| Error::from_reason(format!("JSON error: {}", e)))?),
            Err(e) => Ok(to_json_error(e.into())),
        }
    };
}

// Example usage in NAPI function
#[napi]
pub async fn execute_trade(
    strategy: String,
    symbol: String,
    action: String,
    quantity: i32,
) -> Result<String> {
    napi_result! {
        let services = services();
        let order = OrderRequest {
            symbol,
            side: parse_side(&action)?,
            quantity: quantity as u32,
            order_type: OrderType::Market,
            time_in_force: TimeInForce::Day,
        };

        services.broker_client.execute_order(order).await
    }
}
```

---

## 4. GPU Integration Architecture

### 4.1 GPU Abstraction Layer

```rust
// File: napi-bindings/src/gpu.rs

use nt_neural::Device;
use nt_risk::gpu::GpuAccelerator;

pub struct GpuManager {
    neural_device: Device,
    risk_accelerator: Option<GpuAccelerator>,
    available: bool,
}

impl GpuManager {
    pub fn new(config: &NeuralConfig) -> Self {
        let available = config.use_gpu && Self::check_gpu_available();

        let neural_device = if available {
            match config.device.as_str() {
                "cuda" => Device::cuda_if_available(0).unwrap_or(Device::Cpu),
                "metal" => Device::metal_if_available(0).unwrap_or(Device::Cpu),
                _ => Device::Cpu,
            }
        } else {
            Device::Cpu
        };

        let risk_accelerator = if available && cfg!(feature = "gpu") {
            GpuAccelerator::new().ok()
        } else {
            None
        };

        Self {
            neural_device,
            risk_accelerator,
            available,
        }
    }

    fn check_gpu_available() -> bool {
        #[cfg(feature = "cuda")]
        return cudarc::driver::CudaDevice::new(0).is_ok();

        #[cfg(feature = "metal")]
        return true; // Metal availability check

        #[cfg(not(any(feature = "cuda", feature = "metal")))]
        return false;
    }
}
```

### 4.2 GPU-Accelerated Functions

Priority functions for GPU acceleration:
1. **Neural forecasting** (`neural_forecast`, `neural_train`, `neural_predict`) - 50-100x speedup
2. **Risk calculations** (`risk_analysis`, `monte_carlo_simulation`) - 10-50x speedup
3. **Correlation analysis** (`correlation_analysis`, `cross_asset_correlation_matrix`) - 5-20x speedup
4. **Backtesting** (`run_backtest`, `neural_backtest`) - 3-10x speedup

---

## 5. Phased Implementation Plan

### Phase 1: Foundation (Weeks 1-3) - Priority P0/P1

**Goal**: Core trading functionality operational

**Functions to Implement** (28 functions):
1. System initialization (`ping`, `initialize_neural_trader`, `get_health_status`)
2. Strategy basics (`list_strategies`, `get_strategy_info`, `get_portfolio_status`)
3. Order execution (`execute_trade`, `simulate_trade`)
4. Basic backtesting (`run_backtest`, `quick_analysis`)
5. Risk management (`risk_analysis`, `calculate_kelly_criterion`)
6. Neural basics (`neural_forecast`, `neural_train`, `neural_predict`)

**Deliverables**:
- Service container with DI
- Broker integration (Alpaca paper trading)
- Basic portfolio tracking
- Simple backtesting
- Neural model loading
- Configuration system
- Error handling

**Testing**:
- Unit tests for all services
- Integration test: End-to-end paper trade
- Performance benchmarks

### Phase 2: Advanced Trading (Weeks 4-6) - Priority P2

**Goal**: Production-ready trading with risk management

**Functions to Implement** (35 functions):
- Advanced backtesting (`optimize_strategy`, `performance_report`)
- Multi-asset operations (`execute_multi_asset_trade`, `portfolio_rebalance`)
- Correlation analysis (`correlation_analysis`, `cross_asset_correlation_matrix`)
- News trading (all 8 functions)
- Sports betting core (9 functions)
- Syndicates core (7 functions)

**Deliverables**:
- GPU-accelerated Monte Carlo
- News sentiment analysis
- Sports betting odds integration
- Syndicate fund management
- Advanced risk metrics

**Testing**:
- GPU vs CPU benchmarks
- Historical backtest validation
- News sentiment accuracy tests

### Phase 3: Specialized Features (Weeks 7-9) - Priority P2/P3

**Goal**: Sports betting, syndicates, prediction markets

**Functions to Implement** (25 functions):
- Full sports betting suite (13 functions)
- Full syndicate management (15 functions)
- Prediction markets (5 functions)
- Odds API integration (9 functions)

**Deliverables**:
- Complete sports betting system
- Syndicate voting and profit distribution
- Prediction market integration
- Real-time odds tracking

**Testing**:
- Syndicate allocation simulations
- Arbitrage detection validation
- Kelly Criterion accuracy

### Phase 4: Cloud & Monitoring (Weeks 10-12) - Priority P3

**Goal**: E2B integration, monitoring, optimization

**Functions to Implement** (15 functions):
- E2B cloud integration (9 functions)
- System monitoring (5 functions)
- Neural optimization (`neural_optimize`)

**Deliverables**:
- E2B sandbox deployment
- Performance monitoring dashboard
- Token usage tracking
- Bottleneck analysis

**Testing**:
- E2B deployment tests
- System load testing
- Monitoring accuracy

---

## 6. Dependencies for napi-bindings/Cargo.toml

```toml
[dependencies]
# Internal crates - core functionality
nt-core = { version = "2.0.0", path = "../core" }
nt-strategies = { version = "2.0.0", path = "../strategies" }
nt-execution = { version = "2.0.0", path = "../execution" }
nt-portfolio = { version = "2.0.0", path = "../portfolio" }
nt-risk = { version = "2.0.0", path = "../risk" }
nt-neural = { version = "2.0.0", path = "../neural" }

# Internal crates - specialized features
nt-news-trading = { version = "2.0.0", path = "../news-trading" }
nt-sports-betting = { version = "2.0.0", path = "../sports-betting" }
nt-syndicate = { version = "2.0.0", path = "../nt-syndicate" }
nt-prediction-markets = { version = "2.0.0", path = "../prediction-markets" }
nt-e2b-integration = { version = "2.0.0", path = "../e2b-integration", optional = true }
nt-benchoptimizer = { version = "2.0.0", path = "../nt-benchoptimizer" }

# NAPI
napi = "2.16"
napi-derive = "2.16"

# Async
tokio = { version = "1.35", features = ["full"] }
async-trait = "0.1"
futures = "0.3"

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
tracing-subscriber = { version = "0.3", features = ["env-filter"] }

# Concurrency
parking_lot = "0.12"
dashmap = "5.5"
once_cell = "1.19"

# Decimal precision
rust_decimal = { version = "1.33", features = ["serde-float"] }

# GPU support (optional)
candle-core = { version = "0.6", optional = true }
cudarc = { version = "0.11", optional = true }

[features]
default = []
gpu = ["nt-neural/cuda", "nt-risk/gpu", "candle-core", "cudarc"]
cuda = ["gpu", "candle-core/cuda"]
metal = ["gpu", "candle-core/metal"]
e2b = ["nt-e2b-integration"]
all = ["gpu", "e2b"]

[build-dependencies]
napi-build = "2.1"
```

---

## 7. Implementation Examples

### 7.1 Example: Replacing `execute_trade()`

**Before (Simulation)**:
```rust
#[napi]
pub async fn execute_trade(
    strategy: String,
    symbol: String,
    action: String,
    quantity: i32,
) -> ToolResult {
    Ok(json!({
        "order_id": format!("ord_{}", Utc::now().timestamp()),
        "status": "filled",
        // ... hardcoded values
    }).to_string())
}
```

**After (Real Implementation)**:
```rust
#[napi]
pub async fn execute_trade(
    strategy: String,
    symbol: String,
    action: String,
    quantity: i32,
    order_type: Option<String>,
    limit_price: Option<f64>,
) -> Result<String> {
    napi_result! {
        let services = services();

        // Parse action to OrderSide
        let side = match action.to_lowercase().as_str() {
            "buy" => OrderSide::Buy,
            "sell" => OrderSide::Sell,
            _ => return Err(NeuralTraderError::InvalidArgument(
                format!("Invalid action: {}", action)
            )),
        };

        // Parse order type
        let order_type = match order_type.as_deref() {
            Some("limit") => OrderType::Limit,
            Some("stop") => OrderType::Stop,
            Some("stop_limit") => OrderType::StopLimit,
            _ => OrderType::Market,
        };

        // Create order request
        let order = OrderRequest {
            symbol,
            side,
            quantity: quantity as u32,
            order_type,
            limit_price: limit_price.map(Decimal::from_f64_retain).flatten(),
            time_in_force: TimeInForce::Day,
        };

        // Validate with risk manager
        services.risk_manager.validate_order(&order).await?;

        // Execute via broker
        let response = services.broker_client.execute_order(order).await?;

        // Update portfolio
        services.portfolio_tracker.write().record_order(&response).await?;

        // Return response
        json!({
            "order_id": response.order_id,
            "strategy": strategy,
            "status": response.status,
            "filled_quantity": response.filled_quantity,
            "avg_fill_price": response.avg_fill_price,
            "commission": response.commission,
            "timestamp": response.timestamp,
        })
    }
}
```

### 7.2 Example: Replacing `neural_forecast()`

```rust
#[napi]
pub async fn neural_forecast(
    symbol: String,
    horizon: i32,
    model_id: Option<String>,
    use_gpu: Option<bool>,
    confidence_level: Option<f64>,
) -> Result<String> {
    napi_result! {
        let services = services();

        // Select model (default or specified)
        let model = if let Some(id) = model_id {
            services.model_registry.read()
                .get_model(&id)
                .ok_or_else(|| NeuralTraderError::Neural(
                    format!("Model not found: {}", id)
                ))?
        } else {
            services.model_registry.read().default_forecaster()?
        };

        // Fetch historical data
        let bars = services.market_data_provider
            .get_bars(&symbol, 100)
            .await?;

        // Determine device
        let device = if use_gpu.unwrap_or(true) {
            &services.gpu_manager.neural_device
        } else {
            &Device::Cpu
        };

        // Run forecast
        let predictions = services.neural_engine
            .forecast(&model, &bars, horizon as usize, device)
            .await?;

        // Calculate confidence intervals
        let confidence = confidence_level.unwrap_or(0.95);
        let intervals = predictions.confidence_intervals(confidence);

        json!({
            "forecast_id": uuid::Uuid::new_v4().to_string(),
            "symbol": symbol,
            "model_id": model.id(),
            "horizon_days": horizon,
            "predictions": predictions.to_json(),
            "confidence_intervals": intervals,
            "model_metrics": {
                "architecture": model.architecture(),
                "mae": model.metrics().mae,
                "rmse": model.metrics().rmse,
            },
            "gpu_accelerated": use_gpu.unwrap_or(true) && device.is_cuda(),
            "computation_time_ms": predictions.computation_time_ms(),
            "timestamp": Utc::now().to_rfc3339(),
        })
    }
}
```

### 7.3 Example: Replacing `risk_analysis()`

```rust
#[napi]
pub async fn risk_analysis(
    portfolio: String,
    use_gpu: Option<bool>,
    use_monte_carlo: Option<bool>,
    var_confidence: Option<f64>,
    time_horizon: Option<i32>,
) -> Result<String> {
    napi_result! {
        let services = services();

        // Parse portfolio JSON
        let portfolio: Vec<Asset> = serde_json::from_str(&portfolio)
            .map_err(|e| NeuralTraderError::InvalidArgument(
                format!("Invalid portfolio JSON: {}", e)
            ))?;

        // Configure VaR calculation
        let config = VaRConfig {
            confidence_level: var_confidence.unwrap_or(0.95),
            time_horizon_days: time_horizon.unwrap_or(1),
            num_simulations: 10_000,
            use_gpu: use_gpu.unwrap_or(true) && services.gpu_manager.available,
        };

        // Calculate VaR/CVaR
        let var_result = if use_monte_carlo.unwrap_or(true) {
            let calculator = MonteCarloVaR::new(config);
            calculator.calculate(&portfolio).await?
        } else {
            let calculator = HistoricalVaR::new(config);
            calculator.calculate(&portfolio).await?
        };

        // Run stress tests
        let stress_results = services.risk_manager
            .run_stress_tests(&portfolio)
            .await?;

        // Calculate correlations
        let correlation_matrix = services.risk_manager
            .calculate_correlations(&portfolio)
            .await?;

        json!({
            "analysis_id": uuid::Uuid::new_v4().to_string(),
            "var_metrics": {
                "var_95": var_result.var_95,
                "var_99": var_result.var_99,
                "cvar_95": var_result.cvar_95,
                "cvar_99": var_result.cvar_99,
            },
            "risk_decomposition": var_result.decomposition,
            "stress_tests": stress_results,
            "correlations": correlation_matrix,
            "monte_carlo": if use_monte_carlo.unwrap_or(true) {
                Some(json!({
                    "simulations": config.num_simulations,
                    "distribution": var_result.distribution_stats(),
                }))
            } else {
                None
            },
            "gpu_accelerated": var_result.was_gpu_accelerated,
            "computation_time_ms": var_result.computation_time_ms,
            "timestamp": Utc::now().to_rfc3339(),
        })
    }
}
```

---

## 8. Testing Strategy

### 8.1 Unit Tests
- Each NAPI function has corresponding unit tests
- Mock all external dependencies (broker, market data, etc.)
- Test error handling paths

### 8.2 Integration Tests
```rust
// tests/integration_test.rs

#[tokio::test]
async fn test_end_to_end_trade() {
    // Initialize with paper trading config
    let config = SystemConfig {
        broker: BrokerConfig {
            provider: "alpaca_paper".to_string(),
            // ... test credentials
        },
        // ...
    };

    init_services(config).await.unwrap();

    // Execute trade
    let result = execute_trade(
        "momentum".to_string(),
        "AAPL".to_string(),
        "buy".to_string(),
        10,
        None,
        None,
    ).await.unwrap();

    let response: serde_json::Value = serde_json::from_str(&result).unwrap();
    assert_eq!(response["status"], "filled");

    // Verify portfolio update
    let portfolio = get_portfolio_status(Some(true)).await.unwrap();
    let portfolio: serde_json::Value = serde_json::from_str(&portfolio).unwrap();
    assert!(portfolio["positions"].as_array().unwrap().len() > 0);
}
```

### 8.3 Performance Benchmarks
```rust
// benches/napi_benchmarks.rs

#[criterion]
fn bench_neural_forecast(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    c.bench_function("neural_forecast_cpu", |b| {
        b.iter(|| {
            rt.block_on(async {
                neural_forecast(
                    "AAPL".to_string(),
                    30,
                    None,
                    Some(false), // CPU
                    None,
                ).await
            })
        })
    });

    c.bench_function("neural_forecast_gpu", |b| {
        b.iter(|| {
            rt.block_on(async {
                neural_forecast(
                    "AAPL".to_string(),
                    30,
                    None,
                    Some(true), // GPU
                    None,
                ).await
            })
        })
    });
}
```

---

## 9. Migration Checklist

### Pre-Implementation
- [ ] Review all 103 function signatures
- [ ] Verify all `nt-*` crates compile
- [ ] Set up development environment with GPU support (optional)
- [ ] Create test broker account (Alpaca paper trading)
- [ ] Design configuration schema

### Phase 1 (Weeks 1-3)
- [ ] Implement service container and DI
- [ ] Implement `initialize_neural_trader()`
- [ ] Migrate 28 P0/P1 functions
- [ ] Write integration tests for core trading
- [ ] Set up CI/CD pipeline
- [ ] Document API changes

### Phase 2 (Weeks 4-6)
- [ ] Implement GPU acceleration for risk/neural
- [ ] Migrate 35 P2 functions
- [ ] Add news sentiment analysis
- [ ] Implement sports betting core
- [ ] Performance benchmarking

### Phase 3 (Weeks 7-9)
- [ ] Complete sports betting suite
- [ ] Implement syndicate management
- [ ] Add prediction markets
- [ ] Full Odds API integration

### Phase 4 (Weeks 10-12)
- [ ] E2B cloud integration
- [ ] System monitoring and observability
- [ ] Final performance optimization
- [ ] Production readiness review

---

## 10. Risk Mitigation

### Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| GPU initialization failures | Medium | High | Fallback to CPU, graceful degradation |
| Broker API rate limits | High | Medium | Rate limiting, request queuing |
| Neural model loading errors | Low | High | Model validation, versioning |
| Memory leaks in long-running processes | Medium | High | Thorough testing, memory profiling |
| Configuration errors | High | Medium | Schema validation, clear error messages |

### Operational Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Breaking changes in dependencies | Medium | High | Pin versions, extensive testing |
| Performance regression | Medium | High | Continuous benchmarking |
| Data loss | Low | Critical | Regular backups, transaction safety |
| Security vulnerabilities | Medium | Critical | Dependency audits, secure coding practices |

---

## 11. Performance Targets

### Latency Targets

| Function Category | Target (p50) | Target (p95) |
|------------------|--------------|--------------|
| Order execution | < 50ms | < 100ms |
| Portfolio queries | < 10ms | < 20ms |
| Risk calculations (CPU) | < 500ms | < 1s |
| Risk calculations (GPU) | < 50ms | < 100ms |
| Neural forecasting (CPU) | < 2s | < 5s |
| Neural forecasting (GPU) | < 200ms | < 500ms |
| News sentiment analysis | < 100ms | < 200ms |

### Throughput Targets
- **Orders/second**: 100+ (sustained)
- **Risk calculations/second**: 10+ (GPU), 1+ (CPU)
- **Neural forecasts/second**: 5+ (GPU), 0.5+ (CPU)

---

## 12. Monitoring & Observability

### Key Metrics to Track

```rust
// Example: Instrumenting NAPI functions with metrics

use metrics::{counter, histogram};

#[napi]
pub async fn execute_trade(...) -> Result<String> {
    let start = std::time::Instant::now();

    // Track invocation
    counter!("napi.execute_trade.calls").increment(1);

    let result = napi_result! {
        // ... implementation
    };

    // Track latency
    histogram!("napi.execute_trade.duration_ms")
        .record(start.elapsed().as_millis() as f64);

    // Track success/failure
    match &result {
        Ok(_) => counter!("napi.execute_trade.success").increment(1),
        Err(_) => counter!("napi.execute_trade.error").increment(1),
    }

    result
}
```

---

## 13. Success Criteria

### Phase 1 Success
- ✅ All 28 P0/P1 functions implemented
- ✅ End-to-end paper trading works
- ✅ Unit test coverage > 80%
- ✅ No memory leaks in 24-hour stress test

### Phase 2 Success
- ✅ GPU acceleration functional (10x+ speedup on risk/neural)
- ✅ News sentiment accuracy > 70%
- ✅ All P2 functions passing integration tests

### Phase 3 Success
- ✅ Sports betting arbitrage detection functional
- ✅ Syndicate profit distribution accurate
- ✅ Prediction market integration complete

### Phase 4 Success
- ✅ E2B deployment functional
- ✅ Monitoring dashboard operational
- ✅ Production readiness review passed

---

## 14. Conclusion

This architecture provides a comprehensive roadmap for replacing all 103 simulation functions with production-grade Rust implementations. The phased approach minimizes risk while delivering value incrementally.

### Next Steps
1. **Week 1**: Review this architecture with stakeholders
2. **Week 2**: Set up development environment and CI/CD
3. **Week 3**: Begin Phase 1 implementation
4. **Week 12**: Production deployment

### Key Success Factors
- **Incremental delivery**: Each phase delivers working functionality
- **GPU acceleration**: 10-100x performance improvements where it matters
- **Robust error handling**: No silent failures
- **Comprehensive testing**: Unit, integration, and performance tests
- **Clear documentation**: Every function documented with examples

---

**Document Status**: Ready for Review
**Last Updated**: 2025-11-14
**Contact**: System Architecture Designer
