# HyperPhysics-Nautilus Integration Architecture

## Overview

This document describes the technical architecture of the `hyperphysics-nautilus` integration crate, which bridges HyperPhysics's physics-based trading pipeline with Nautilus Trader's institutional-grade execution infrastructure.

**Version:** 1.0
**Last Updated:** November 2025

---

## 1. Architectural Goals

### 1.1 Primary Objectives

1. **Seamless Integration:** Enable HyperPhysics signal generation within Nautilus Trader's event-driven framework
2. **Type Safety:** Provide compile-time guarantees for type conversions between systems
3. **Performance:** Maintain sub-200μs total latency for signal generation
4. **Flexibility:** Support multiple deployment modes (standalone, integrated, hybrid)

### 1.2 Design Principles

- **Separation of Concerns:** Clear boundaries between data adaptation, signal generation, and order execution
- **Async-First:** Full async/await support for high-concurrency workloads
- **Testability:** Comprehensive unit and integration tests without external dependencies
- **Observability:** Detailed metrics and tracing for performance analysis

---

## 2. Module Architecture

### 2.1 Module Dependency Graph

```
┌─────────────────────────────────────────────────────────────────┐
│                    hyperphysics-nautilus                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────┐                                                    │
│  │  lib.rs  │ ← Entry point, re-exports                          │
│  └────┬─────┘                                                    │
│       │                                                          │
│  ┌────┴────┬────────────┬────────────┬────────────┐             │
│  │         │            │            │            │             │
│  ▼         ▼            ▼            ▼            ▼             │
│ error    config       types      adapter     strategy           │
│  .rs      .rs          /           /            /               │
│                    ┌───┴───┐   ┌───┴───┐   ┌───┴───┐            │
│                    │mod.rs │   │mod.rs │   │mod.rs │            │
│                    │conver-│   │data_  │   │hyper- │            │
│                    │sions  │   │adapter│   │physics│            │
│                    │naut_  │   │exec_  │   │_strat │            │
│                    │compat │   │bridge │   │egy    │            │
│                    └───────┘   └───────┘   └───────┘            │
│                                                                  │
│                              backtest/                           │
│                           ┌───────────┐                          │
│                           │  mod.rs   │                          │
│                           │  runner   │                          │
│                           │  data_    │                          │
│                           │  loader   │                          │
│                           └───────────┘                          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ depends on
                              ▼
    ┌─────────────────────────────────────────────────────────────┐
    │              External HyperPhysics Crates                    │
    ├─────────────────────────────────────────────────────────────┤
    │  hyperphysics-core     hyperphysics-hft-ecosystem           │
    │  hyperphysics-geometry hyperphysics-optimization            │
    │  hyperphysics-market                                         │
    └─────────────────────────────────────────────────────────────┘
```

### 2.2 Module Responsibilities

| Module | Responsibility |
|--------|----------------|
| `error.rs` | Custom error types and Result alias |
| `config.rs` | Integration configuration structures |
| `types/` | Type definitions and conversion layer |
| `adapter/` | Data and execution adapters |
| `strategy/` | Nautilus-compatible strategy implementation |
| `backtest/` | Standalone backtesting infrastructure |

---

## 3. Type System

### 3.1 Core Type Mappings

```
┌─────────────────────────────────────────────────────────────────┐
│                        Type Mappings                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Nautilus Types              HyperPhysics Types                  │
│  ══════════════              ═════════════════                   │
│                                                                  │
│  QuoteTick ────────────────► MarketSnapshot                      │
│  (fixed-point i64)           (f64 floating-point)                │
│                                                                  │
│  TradeTick ────────────────► MarketTick                          │
│  (fixed-point i64)           (f64 floating-point)                │
│                                                                  │
│  Bar ──────────────────────► BarData                             │
│  (fixed-point i64)           (f64 floating-point)                │
│                                                                  │
│  OrderBookDelta ───────────► OrderBookLevel                      │
│  (fixed-point i64)           (f64 floating-point)                │
│                                                                  │
│  TradingDecision ──────────► HyperPhysicsOrderCommand            │
│  (HyperPhysics internal)     (Nautilus-compatible)               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Precision Handling

The conversion layer maintains a precision scale table for efficient fixed-to-float conversion:

```rust
const PRECISION_SCALE: [f64; 17] = [
    1.0,                    // 0 decimal places
    10.0,                   // 1 decimal place
    100.0,                  // 2 decimal places
    1_000.0,                // 3 decimal places
    10_000.0,               // 4 decimal places
    100_000.0,              // 5 decimal places
    1_000_000.0,            // 6 decimal places
    10_000_000.0,           // 7 decimal places
    100_000_000.0,          // 8 decimal places (common for crypto)
    // ... up to 16 decimal places
];
```

### 3.3 Conversion Functions

```rust
/// Convert fixed-point value to f64 with precision
pub fn fixed_to_f64(value: i64, precision: u8) -> f64 {
    value as f64 / PRECISION_SCALE[precision as usize]
}

/// Convert f64 to fixed-point with precision
pub fn f64_to_fixed(value: f64, precision: u8) -> i64 {
    (value * PRECISION_SCALE[precision as usize]).round() as i64
}
```

---

## 4. Adapter Layer

### 4.1 NautilusDataAdapter

The data adapter maintains per-instrument state and converts Nautilus events to HyperPhysics MarketFeed format.

```
┌─────────────────────────────────────────────────────────────────┐
│                    NautilusDataAdapter                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input Events                       Internal State               │
│  ════════════                       ══════════════               │
│                                                                  │
│  QuoteTick ─────┐                  ┌─────────────────────┐       │
│                 │                  │  InstrumentState    │       │
│  TradeTick ─────┼──────────────────│  • last_snapshot    │       │
│                 │                  │  • bars: VecDeque   │       │
│  Bar ───────────┤                  │  • order_book       │       │
│                 │                  │  • volume_profile   │       │
│  OrderBookDelta─┘                  └─────────────────────┘       │
│                                              │                   │
│                                              ▼                   │
│                                    ┌─────────────────────┐       │
│                                    │     MarketFeed      │       │
│                                    │  • snapshot         │       │
│                                    │  • ticks            │       │
│                                    │  • bars             │       │
│                                    │  • order_book       │       │
│                                    └─────────────────────┘       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 NautilusExecBridge

The execution bridge converts HyperPhysics trading decisions to Nautilus order commands.

```
┌─────────────────────────────────────────────────────────────────┐
│                    NautilusExecBridge                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────────┐       ┌────────────────────────────┐    │
│  │   PipelineResult   │       │  HyperPhysicsOrderCommand  │    │
│  │   • decision       │  ───► │  • client_order_id         │    │
│  │   • confidence     │       │  • instrument_id           │    │
│  │   • latency_us     │       │  • side, type, quantity    │    │
│  │   • consensus_term │       │  • hp_confidence           │    │
│  └────────────────────┘       │  • hp_algorithm            │    │
│                               │  • hp_latency_us           │    │
│                               │  • hp_consensus_term       │    │
│                               └────────────────────────────┘    │
│                                                                  │
│  Validation:                                                     │
│  • Confidence ≥ min_confidence_threshold                         │
│  • Consensus (if enabled) must be reached                        │
│  • Position limits checked                                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. Strategy Architecture

### 5.1 HyperPhysicsStrategy State Machine

```
┌─────────────────────────────────────────────────────────────────┐
│              HyperPhysicsStrategy State Machine                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│      ┌───────────────────────────────────────────────────┐      │
│      │                                                   │      │
│      ▼                                                   │      │
│  ┌────────┐    start()    ┌─────────┐    stop()    ┌────────┐  │
│  │Initial-│──────────────►│ Running │─────────────►│Stopped │  │
│  │ized    │               │         │              │        │  │
│  └────────┘               └────┬────┘              └───┬────┘  │
│      ▲                         │                       │       │
│      │         reset()         │        reset()        │       │
│      └─────────────────────────┴───────────────────────┘       │
│                                                                  │
│                         │ error                                  │
│                         ▼                                        │
│                    ┌─────────┐                                   │
│                    │ Faulted │                                   │
│                    └─────────┘                                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Event Processing Flow

```rust
impl HyperPhysicsStrategy {
    pub async fn on_quote(&self, quote: &NautilusQuoteTick)
        -> Result<Option<HyperPhysicsOrderCommand>>
    {
        // 1. Check if running
        if !self.is_running().await {
            return Ok(None);
        }

        // 2. Convert to HyperPhysics format
        let feed = self.data_adapter.on_quote(quote).await?;

        // 3. Execute HyperPhysics pipeline
        let result = self.pipeline.execute(&feed).await?;

        // 4. Process result through exec bridge
        self.exec_bridge.process_result(&result).await
    }
}
```

### 5.3 Metrics Collection

```
┌─────────────────────────────────────────────────────────────────┐
│                    StrategyMetrics                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Event Counters                 Latency Tracking                 │
│  ══════════════                 ════════════════                 │
│  • quotes_processed: u64        • avg_signal_latency_us: f64     │
│  • trades_processed: u64        • max_signal_latency_us: u64     │
│  • bars_processed: u64                                           │
│                                                                  │
│  Signal Statistics              Runtime                          │
│  ═════════════════              ═══════                          │
│  • signals_generated: u64       • runtime_seconds: f64           │
│  • orders_submitted: u64                                         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. Backtest Architecture

### 6.1 BacktestRunner Design

```
┌─────────────────────────────────────────────────────────────────┐
│                       BacktestRunner                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Configuration                  Strategy                         │
│  ┌───────────────┐             ┌───────────────────────┐        │
│  │BacktestConfig │             │ HyperPhysicsStrategy  │        │
│  │• initial_cap  │             │                       │        │
│  │• commission   │             │ (internal pipeline)   │        │
│  │• slippage     │             │                       │        │
│  │• time_range   │             └───────────────────────┘        │
│  └───────────────┘                        │                      │
│                                           ▼                      │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    Event Loop                              │  │
│  │                                                            │  │
│  │  for event in events:                                      │  │
│  │    1. Filter by time range                                 │  │
│  │    2. Update current_price                                 │  │
│  │    3. strategy.on_quote/trade/bar()                        │  │
│  │    4. Execute order if generated                           │  │
│  │    5. Update equity and drawdown                           │  │
│  │                                                            │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              │                                   │
│                              ▼                                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                  BacktestResults                           │  │
│  │  • total_return      • win_rate       • final_equity       │  │
│  │  • sharpe_ratio      • profit_factor  • avg_latency_us     │  │
│  │  • max_drawdown      • total_trades   • runtime_secs       │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 Position Tracking

```rust
struct Position {
    size: f64,              // Positive = long, negative = short
    avg_entry_price: f64,   // Weighted average entry
    unrealized_pnl: f64,    // Mark-to-market
    realized_pnl: f64,      // Closed trades
}
```

### 6.3 Slippage Models

```rust
pub enum SlippageModel {
    /// No slippage (ideal execution)
    None,

    /// Fixed basis points slippage
    FixedBps(f64),

    /// Volatility-dependent slippage
    VolatilityBased { multiplier: f64 },
}
```

---

## 7. Error Handling

### 7.1 Error Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│                    IntegrationError                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Pipeline(String)                                         │    │
│  │ └── Errors from HyperPhysics UnifiedPipeline             │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ ConsensusNotReached { confidence, threshold }            │    │
│  │ └── Signal confidence below minimum threshold            │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ InvalidConversion { field, reason }                      │    │
│  │ └── Type conversion failed                               │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ StrategyNotRunning                                       │    │
│  │ └── Attempted operation on stopped strategy              │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Serialization(String)                                    │    │
│  │ └── JSON/CSV parsing errors                              │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Io(std::io::Error)                                       │    │
│  │ └── File I/O errors                                      │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 8. Configuration

### 8.1 IntegrationConfig

```rust
pub struct IntegrationConfig {
    /// Unique strategy identifier
    pub strategy_id: String,

    /// Enable Byzantine consensus validation
    pub enable_consensus: bool,

    /// Minimum confidence for signal acceptance (0.0 - 1.0)
    pub min_confidence_threshold: f64,

    /// Maximum position size per instrument
    pub max_position_size: f64,

    /// Target latency for signal generation (microseconds)
    pub target_latency_us: u64,

    /// Default order time-in-force
    pub default_tif: TimeInForce,

    /// Enable order post-only mode
    pub post_only: bool,

    /// Enable reduce-only mode
    pub reduce_only: bool,
}
```

### 8.2 BacktestConfig

```rust
pub struct BacktestConfig {
    /// Backtest start time (nanoseconds since epoch)
    pub start_time_ns: u64,

    /// Backtest end time (nanoseconds since epoch)
    pub end_time_ns: u64,

    /// Initial account capital
    pub initial_capital: f64,

    /// Commission rate (0.001 = 0.1%)
    pub commission_rate: f64,

    /// Slippage model
    pub slippage_model: SlippageModel,

    /// Enable verbose logging
    pub verbose: bool,
}
```

---

## 9. Performance Optimization

### 9.1 Memory Layout

- **Arena allocation** for order commands to reduce heap fragmentation
- **Pre-sized VecDeque** for bar history (default: 1000 bars)
- **DashMap** for concurrent instrument state access

### 9.2 Async Patterns

- **RwLock** for read-heavy data (snapshots, metrics)
- **AtomicU64** for counters (order sequence)
- **Arc** for shared ownership across async tasks

### 9.3 Zero-Copy Where Possible

- References used for event handlers (`&NautilusQuoteTick`)
- Clone only when mutation needed
- String interning via `ustr` crate for instrument IDs

---

## 10. Testing Strategy

### 10.1 Test Categories

| Category | Purpose | Location |
|----------|---------|----------|
| Unit Tests | Individual function correctness | `src/**/*.rs` |
| Integration Tests | Module interaction | `tests/integration_tests.rs` |
| Performance Tests | Latency/throughput benchmarks | `benches/` |
| Property Tests | Invariant verification | Via `proptest` |

### 10.2 Test Coverage Goals

- **Type conversions:** 100% coverage
- **Adapter logic:** 90%+ coverage
- **Strategy lifecycle:** 95%+ coverage
- **Backtest runner:** 85%+ coverage

---

## 11. Future Extensions

### 11.1 Planned Enhancements

1. **PyO3 Bindings:** Python interface for Nautilus integration
2. **WebSocket Adapter:** Direct exchange connectivity
3. **Risk Engine Integration:** Connect to Nautilus RiskEngine
4. **Order Book Reconstruction:** Full L2/L3 book support

### 11.2 Scalability Path

```
Phase 1 (Current):  Single strategy, single instrument
Phase 2:            Single strategy, multiple instruments
Phase 3:            Multiple strategies, multiple instruments
Phase 4:            Distributed deployment, co-location support
```

---

## Appendix: Code Examples

### A.1 Basic Strategy Usage

```rust
use hyperphysics_nautilus::{
    config::IntegrationConfig,
    strategy::HyperPhysicsStrategy,
    types::NautilusQuoteTick,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create configuration
    let config = IntegrationConfig {
        strategy_id: "HP-BTC-001".to_string(),
        min_confidence_threshold: 0.7,
        ..Default::default()
    };

    // Create strategy
    let strategy = HyperPhysicsStrategy::new(config).await?;

    // Set instrument and start
    strategy.set_instrument("BTCUSDT.BINANCE").await;
    strategy.start().await?;

    // Process quote (normally called by Nautilus DataEngine)
    let quote = NautilusQuoteTick {
        instrument_id: 1,
        bid_price: 5000000,
        ask_price: 5000100,
        bid_size: 100,
        ask_size: 100,
        price_precision: 2,
        size_precision: 0,
        ts_event: 1700000000_000_000_000,
        ts_init: 1700000000_000_000_000,
    };

    if let Some(order) = strategy.on_quote(&quote).await? {
        println!("Generated order: {:?}", order);
    }

    strategy.stop().await?;
    Ok(())
}
```

### A.2 Backtest Example

```rust
use hyperphysics_nautilus::{
    backtest::{BacktestConfig, BacktestRunner, DataLoader, SlippageModel},
    config::IntegrationConfig,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Configure backtest
    let bt_config = BacktestConfig {
        initial_capital: 100_000.0,
        commission_rate: 0.0005,
        slippage_model: SlippageModel::FixedBps(0.5),
        ..Default::default()
    };

    let strategy_config = IntegrationConfig::backtest();

    // Create runner
    let mut runner = BacktestRunner::new(bt_config, strategy_config).await?;

    // Generate or load market data
    let events = DataLoader::generate_synthetic_quotes(
        1,      // instrument_id
        10000,  // num_ticks
        100.0,  // start_price
        0.002,  // volatility
        1000000000,  // start_time_ns
        100000,      // interval_ns
    );

    // Run backtest
    let results = runner.run(events).await?;

    println!("Backtest Results:");
    println!("  Total Return: {:.2}%", results.total_return * 100.0);
    println!("  Sharpe Ratio: {:.2}", results.sharpe_ratio);
    println!("  Max Drawdown: {:.2}%", results.max_drawdown * 100.0);
    println!("  Total Trades: {}", results.total_trades);

    Ok(())
}
```
