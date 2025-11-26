# Neural Trading Rust Port - Testing, Benchmarking & CI/CD Strategy

## Executive Summary

This document outlines a comprehensive quality assurance strategy for the Neural Trading Rust port, ensuring high performance, reliability, and maintainability through rigorous testing, benchmarking, and continuous integration.

**Key Objectives:**
- Achieve 90%+ test coverage across all modules
- Meet strict performance targets (<100μs market data ingestion)
- Ensure cross-platform compatibility (Linux, macOS, Windows)
- Maintain parity with Python implementation
- Enable confident refactoring and feature development

---

## 1. Testing Hierarchy

### 1.1 Unit Tests (L1 - Foundation)

**Purpose:** Validate individual functions and components in isolation.

**Coverage Requirements:** >95% for core logic, 100% for financial calculations

**Location:** `tests/unit/` and inline `#[cfg(test)]` modules

**Example: Market Data Parser**
```rust
// src/market_data/parser.rs
#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal::Decimal;

    #[test]
    fn test_parse_ohlcv_valid_data() {
        let input = "1234567890,100.50,101.00,99.75,100.25,50000";
        let result = parse_ohlcv(input).unwrap();

        assert_eq!(result.timestamp, 1234567890);
        assert_eq!(result.open, Decimal::from_str("100.50").unwrap());
        assert_eq!(result.high, Decimal::from_str("101.00").unwrap());
        assert_eq!(result.low, Decimal::from_str("99.75").unwrap());
        assert_eq!(result.close, Decimal::from_str("100.25").unwrap());
        assert_eq!(result.volume, 50000);
    }

    #[test]
    fn test_parse_ohlcv_invalid_timestamp() {
        let input = "invalid,100.50,101.00,99.75,100.25,50000";
        let result = parse_ohlcv(input);

        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err().kind(),
            ErrorKind::InvalidTimestamp
        );
    }

    #[test]
    fn test_price_precision_handling() {
        // Financial calculations must maintain precision
        let price = Decimal::from_str("100.12345678").unwrap();
        let quantity = Decimal::from_str("1.5").unwrap();
        let total = calculate_total(price, quantity);

        // Expect 8 decimal places for crypto
        assert_eq!(total, Decimal::from_str("150.18518517").unwrap());
    }

    #[test]
    fn test_zero_volume_handling() {
        let input = "1234567890,100.50,101.00,99.75,100.25,0";
        let result = parse_ohlcv(input).unwrap();

        assert_eq!(result.volume, 0);
        assert!(!result.is_valid_for_trading()); // Should not trade on zero volume
    }
}
```

**Best Practices:**
- Test edge cases: zero, negative, maximum values
- Test error conditions explicitly
- Use descriptive test names: `test_<function>_<scenario>_<expected_outcome>`
- Mock external dependencies (network, filesystem, time)
- Test precision for financial calculations (use `rust_decimal`)

---

### 1.2 Property-Based Tests (L2 - Generative)

**Purpose:** Validate invariants across wide input ranges using randomized testing.

**Tool:** `proptest` crate (preferred) or `quickcheck`

**Coverage:** All pure functions with mathematical properties

**Example: Technical Indicators**
```rust
// tests/unit/indicators_properties.rs
use proptest::prelude::*;
use rust_decimal::Decimal;

proptest! {
    #[test]
    fn test_sma_invariants(
        prices in prop::collection::vec(
            any::<f64>().prop_map(|x| Decimal::from_f64(x.abs().min(1e6)).unwrap()),
            10..100
        ),
        window in 2usize..20
    ) {
        let sma = calculate_sma(&prices, window).unwrap();

        // Invariant 1: SMA should be between min and max of input
        let min_price = prices.iter().min().unwrap();
        let max_price = prices.iter().max().unwrap();
        prop_assert!(sma >= *min_price && sma <= *max_price);

        // Invariant 2: SMA of constant values equals that value
        if prices.iter().all(|&p| p == prices[0]) {
            prop_assert_eq!(sma, prices[0]);
        }

        // Invariant 3: Output length matches input length - window + 1
        let result_len = calculate_sma_series(&prices, window).unwrap().len();
        prop_assert_eq!(result_len, prices.len() - window + 1);
    }

    #[test]
    fn test_position_sizing_safety(
        capital in 1000.0..1_000_000.0,
        risk_percent in 0.01..0.10,
        entry_price in 10.0..50000.0,
        stop_loss_distance in 0.01..0.20
    ) {
        let position_size = calculate_position_size(
            Decimal::from_f64(capital).unwrap(),
            Decimal::from_f64(risk_percent).unwrap(),
            Decimal::from_f64(entry_price).unwrap(),
            Decimal::from_f64(stop_loss_distance).unwrap()
        ).unwrap();

        let max_loss = position_size * Decimal::from_f64(entry_price).unwrap()
                        * Decimal::from_f64(stop_loss_distance).unwrap();
        let expected_loss = Decimal::from_f64(capital * risk_percent).unwrap();

        // Invariant: Maximum loss should never exceed risk tolerance
        prop_assert!((max_loss - expected_loss).abs() < Decimal::from_f64(0.01).unwrap());

        // Invariant: Position size should be positive
        prop_assert!(position_size > Decimal::ZERO);

        // Invariant: Position should not exceed available capital
        prop_assert!(position_size * Decimal::from_f64(entry_price).unwrap()
                     <= Decimal::from_f64(capital).unwrap());
    }

    #[test]
    fn test_orderbook_consistency(
        bids in prop::collection::vec((1.0..10000.0, 0.1..100.0), 1..50),
        asks in prop::collection::vec((1.0..10000.0, 0.1..100.0), 1..50)
    ) {
        let mut orderbook = OrderBook::new();

        for (price, size) in bids {
            orderbook.add_bid(Decimal::from_f64(price).unwrap(),
                            Decimal::from_f64(size).unwrap());
        }
        for (price, size) in asks {
            orderbook.add_ask(Decimal::from_f64(price).unwrap(),
                            Decimal::from_f64(size).unwrap());
        }

        // Invariant 1: Best bid should be less than best ask (no crossed book)
        if let (Some(best_bid), Some(best_ask)) = (orderbook.best_bid(), orderbook.best_ask()) {
            prop_assert!(best_bid.price < best_ask.price);
        }

        // Invariant 2: Bids should be sorted descending
        let bid_prices: Vec<_> = orderbook.bids().map(|b| b.price).collect();
        prop_assert!(bid_prices.windows(2).all(|w| w[0] >= w[1]));

        // Invariant 3: Asks should be sorted ascending
        let ask_prices: Vec<_> = orderbook.asks().map(|a| a.price).collect();
        prop_assert!(ask_prices.windows(2).all(|w| w[0] <= w[1]));
    }
}
```

**Property Categories:**
1. **Idempotence:** `f(f(x)) = f(x)` for normalization functions
2. **Commutativity:** `f(a, b) = f(b, a)` for symmetric operations
3. **Boundary preservation:** Output within expected ranges
4. **Monotonicity:** Order preservation in sorted data
5. **Inverse relationships:** Encode/decode, buy/sell symmetry

---

### 1.3 Integration Tests (L3 - Component Interaction)

**Purpose:** Verify correct interaction between modules (market data → features → signals → orders).

**Location:** `tests/integration/`

**Example: Feature Pipeline Integration**
```rust
// tests/integration/feature_pipeline_test.rs
use neural_trader::{
    market_data::{MarketDataStream, Tick},
    features::{FeatureExtractor, FeatureVector},
    strategies::SignalGenerator,
    config::Config
};
use tokio::sync::mpsc;

#[tokio::test]
async fn test_tick_to_signal_pipeline() {
    // Setup: Create real components (no mocks)
    let config = Config::from_test_defaults();
    let (tick_tx, tick_rx) = mpsc::channel(100);
    let (signal_tx, mut signal_rx) = mpsc::channel(100);

    let mut market_stream = MarketDataStream::new(tick_rx);
    let mut feature_extractor = FeatureExtractor::new(config.features.clone());
    let mut signal_generator = SignalGenerator::new(config.strategy.clone());

    // Spawn pipeline tasks
    let pipeline_handle = tokio::spawn(async move {
        while let Some(tick) = market_stream.next().await {
            if let Ok(features) = feature_extractor.extract(&tick).await {
                if let Ok(signal) = signal_generator.generate(&features).await {
                    signal_tx.send(signal).await.unwrap();
                }
            }
        }
    });

    // Test: Send market ticks
    let test_ticks = vec![
        Tick::new("BTC/USD", 50000.0, 0.5, 1234567890),
        Tick::new("BTC/USD", 50100.0, 0.3, 1234567891),
        Tick::new("BTC/USD", 50200.0, 0.7, 1234567892),
    ];

    for tick in test_ticks {
        tick_tx.send(tick).await.unwrap();
    }
    drop(tick_tx); // Close channel to stop pipeline

    // Verify: Signals generated
    let mut signal_count = 0;
    while let Some(signal) = signal_rx.recv().await {
        signal_count += 1;
        assert!(signal.confidence >= 0.0 && signal.confidence <= 1.0);
        assert!(matches!(signal.action, Action::Buy | Action::Sell | Action::Hold));
    }

    assert!(signal_count > 0, "Pipeline should generate signals");
    pipeline_handle.await.unwrap();
}

#[tokio::test]
async fn test_agentdb_strategy_coordination() {
    // Test AgentDB integration for strategy memory
    let db = AgentDB::open("test_strategy.db").await.unwrap();
    let strategy = AdaptiveStrategy::new(db.clone());

    // Store historical patterns
    let pattern = TradingPattern {
        features: vec![0.5, 0.3, 0.8],
        outcome: Outcome::Profitable { pnl: 150.0 },
        timestamp: 1234567890,
    };
    db.store_pattern("momentum_strategy", &pattern).await.unwrap();

    // Generate signal using memory
    let current_features = FeatureVector::new(vec![0.52, 0.31, 0.79]);
    let signal = strategy.generate_signal(&current_features).await.unwrap();

    // Verify: Strategy adapts based on similar past patterns
    assert!(signal.confidence > 0.6, "Similar pattern should increase confidence");

    // Cleanup
    db.close().await.unwrap();
    std::fs::remove_file("test_strategy.db").ok();
}

#[tokio::test]
async fn test_risk_management_integration() {
    let config = Config::from_test_defaults();
    let mut portfolio = Portfolio::new(config.risk.clone());
    let risk_manager = RiskManager::new(config.risk.clone());

    // Add initial position
    portfolio.add_position(Position {
        symbol: "BTC/USD".to_string(),
        size: Decimal::from(1),
        entry_price: Decimal::from(50000),
        stop_loss: Some(Decimal::from(48000)),
        take_profit: Some(Decimal::from(55000)),
    }).unwrap();

    // Test: Try to add risky position
    let risky_signal = Signal {
        action: Action::Buy,
        symbol: "ETH/USD".to_string(),
        size: Decimal::from(100), // Excessive size
        confidence: 0.9,
    };

    let result = risk_manager.validate_signal(&risky_signal, &portfolio).await;

    // Verify: Risk manager rejects excessive position
    assert!(result.is_err());
    assert_eq!(result.unwrap_err().kind(), ErrorKind::RiskLimitExceeded);
}
```

**Integration Patterns:**
1. **Pipeline testing:** End-to-end data flow
2. **State management:** Shared state across components
3. **Error propagation:** Errors handled correctly across boundaries
4. **Resource management:** Connections, file handles cleaned up
5. **Timing/ordering:** Async operations complete in correct order

---

### 1.4 End-to-End Tests (L4 - Full System)

**Purpose:** Validate complete trading workflows from market data ingestion to order execution.

**Location:** `tests/e2e/`

**Example: Complete Trading Cycle**
```rust
// tests/e2e/trading_workflow_test.rs
use neural_trader::{TradingSystem, Config};
use neural_trader::mocks::{MockExchange, MockMarketData};
use std::time::Duration;

#[tokio::test]
#[ignore] // Run with `cargo test --ignored` for full e2e tests
async fn test_live_trading_workflow() {
    // Setup: Real components with mock exchange
    let config = Config::load_test_config();
    let mock_exchange = MockExchange::new();
    let mock_market = MockMarketData::new("fixtures/btc_1h_2024.csv");

    let mut trading_system = TradingSystem::builder()
        .with_config(config)
        .with_exchange(Box::new(mock_exchange.clone()))
        .with_market_data(Box::new(mock_market))
        .build()
        .await
        .unwrap();

    // Start system
    trading_system.start().await.unwrap();

    // Let system run for simulated 1 hour
    tokio::time::sleep(Duration::from_secs(60)).await;

    // Stop system
    let stats = trading_system.stop().await.unwrap();

    // Assertions: End-to-end validation
    assert!(stats.ticks_processed > 3600, "Should process 1 tick/sec minimum");
    assert!(stats.signals_generated > 0, "Should generate trading signals");
    assert!(stats.orders_placed > 0, "Should place orders");
    assert_eq!(stats.errors, 0, "Should complete without errors");

    // Verify orders on mock exchange
    let executed_orders = mock_exchange.get_executed_orders().await;
    assert!(!executed_orders.is_empty(), "Orders should be executed");

    for order in executed_orders {
        assert!(order.size > Decimal::ZERO, "Order size must be positive");
        assert!(order.price > Decimal::ZERO, "Order price must be positive");
        assert!(order.status == OrderStatus::Filled, "All orders should fill in mock");
    }

    // Verify portfolio state
    let portfolio = trading_system.portfolio().await;
    assert!(portfolio.total_value() > Decimal::ZERO, "Portfolio should have value");

    // Verify AgentDB memory persistence
    let db = trading_system.agentdb().await;
    let patterns = db.query_patterns("recent_trades", 10).await.unwrap();
    assert!(!patterns.is_empty(), "Should store trading patterns in AgentDB");
}

#[tokio::test]
#[ignore]
async fn test_multi_symbol_trading() {
    let config = Config::load_test_config();
    let symbols = vec!["BTC/USD", "ETH/USD", "SOL/USD"];

    let mut trading_system = TradingSystem::builder()
        .with_config(config)
        .with_symbols(symbols.clone())
        .with_mock_exchange()
        .build()
        .await
        .unwrap();

    trading_system.start().await.unwrap();
    tokio::time::sleep(Duration::from_secs(30)).await;
    let stats = trading_system.stop().await.unwrap();

    // Verify all symbols traded
    let portfolio = trading_system.portfolio().await;
    for symbol in symbols {
        assert!(
            portfolio.has_position(symbol) || stats.orders_for_symbol(symbol) > 0,
            "Should attempt to trade {}", symbol
        );
    }
}

#[tokio::test]
#[ignore]
async fn test_error_recovery_and_resilience() {
    let config = Config::load_test_config();
    let mut faulty_exchange = MockExchange::with_failure_rate(0.1); // 10% failure

    let mut trading_system = TradingSystem::builder()
        .with_config(config)
        .with_exchange(Box::new(faulty_exchange))
        .build()
        .await
        .unwrap();

    trading_system.start().await.unwrap();
    tokio::time::sleep(Duration::from_secs(60)).await;
    let stats = trading_system.stop().await.unwrap();

    // System should continue operating despite failures
    assert!(stats.orders_placed > 0, "Should place orders despite failures");
    assert!(stats.errors > 0, "Should encounter some errors");
    assert!(stats.orders_placed > stats.errors, "Success rate should exceed failure rate");

    // Verify error logging
    let error_logs = trading_system.error_logs().await;
    assert!(error_logs.iter().any(|e| e.contains("Exchange API error")));
}
```

**E2E Test Scenarios:**
1. **Happy path:** Normal trading cycle completes successfully
2. **Market volatility:** System handles rapid price changes
3. **Exchange failures:** Retry logic and error recovery
4. **Resource exhaustion:** System handles memory/connection limits
5. **Concurrent operations:** Multiple symbols traded simultaneously

---

### 1.5 Fuzz Tests (L5 - Security & Robustness)

**Purpose:** Discover crashes, panics, and security vulnerabilities through randomized input.

**Tool:** `cargo-fuzz` with libFuzzer

**Location:** `fuzz/fuzz_targets/`

**Example: Order Parser Fuzzing**
```rust
// fuzz/fuzz_targets/order_parser.rs
#![no_main]
use libfuzzer_sys::fuzz_target;
use neural_trader::orders::parse_order_message;

fuzz_target!(|data: &[u8]| {
    // Fuzz the order parser with arbitrary bytes
    if let Ok(s) = std::str::from_utf8(data) {
        // Should never panic, always return Result
        let _ = parse_order_message(s);
    }
});
```

```rust
// fuzz/fuzz_targets/market_data.rs
#![no_main]
use libfuzzer_sys::fuzz_target;
use neural_trader::market_data::Tick;
use serde_json;

fuzz_target!(|data: &[u8]| {
    // Fuzz JSON deserialization
    if let Ok(json_str) = std::str::from_utf8(data) {
        // Should handle malformed JSON gracefully
        let _: Result<Tick, _> = serde_json::from_str(json_str);
    }
});
```

```rust
// fuzz/fuzz_targets/agentdb_query.rs
#![no_main]
use libfuzzer_sys::fuzz_target;
use neural_trader::agentdb::AgentDB;
use arbitrary::Arbitrary;

#[derive(Arbitrary, Debug)]
struct FuzzQuery {
    collection: String,
    query_vector: Vec<f32>,
    limit: usize,
}

fuzz_target!(|input: FuzzQuery| {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        if let Ok(db) = AgentDB::open_in_memory().await {
            // Should never panic on arbitrary queries
            let _ = db.query_similar(
                &input.collection,
                &input.query_vector,
                input.limit.min(1000) // Cap to prevent OOM
            ).await;
        }
    });
});
```

**Fuzzing Campaign:**
```bash
# Run fuzzer for 1 hour per target
cargo fuzz run order_parser -- -max_total_time=3600
cargo fuzz run market_data -- -max_total_time=3600
cargo fuzz run agentdb_query -- -max_total_time=3600

# Minimize crash inputs
cargo fuzz cmin order_parser

# Reproduce crash
cargo fuzz run order_parser fuzz/corpus/order_parser/crash-da39a3ee5e6b4b0d3255bfef95601890afd80709
```

**Fuzz Targets:**
1. **Parsing:** JSON, CSV, binary protocols
2. **Network protocols:** WebSocket messages, REST responses
3. **Database operations:** Queries, inserts, transactions
4. **Arithmetic:** Financial calculations with edge cases
5. **State machines:** Order lifecycle, connection states

---

### 1.6 Parity Tests (L6 - Python Compatibility)

**Purpose:** Ensure Rust port produces identical results to Python implementation.

**Location:** `tests/parity/`

**Example: Cross-Language Validation**
```rust
// tests/parity/indicator_parity_test.rs
use neural_trader::indicators::calculate_rsi;
use pyo3::prelude::*;
use pyo3::types::PyList;
use approx::assert_relative_eq;

#[test]
fn test_rsi_parity_with_python() {
    // Load Python implementation
    Python::with_gil(|py| {
        let sys = py.import("sys").unwrap();
        sys.getattr("path").unwrap()
            .call_method1("append", ("../python_impl",))
            .unwrap();

        let python_indicators = py.import("neural_trader.indicators").unwrap();

        // Test data
        let prices = vec![
            44.34, 44.09, 43.61, 44.33, 44.83,
            45.10, 45.42, 45.84, 46.08, 45.89,
            46.03, 45.61, 46.28, 46.28, 46.00,
        ];

        // Python RSI
        let py_prices = PyList::new(py, &prices);
        let py_rsi: f64 = python_indicators
            .call_method1("calculate_rsi", (py_prices, 14))
            .unwrap()
            .extract()
            .unwrap();

        // Rust RSI
        let rust_rsi = calculate_rsi(&prices, 14).unwrap();

        // Assert parity within 0.01% (floating point tolerance)
        assert_relative_eq!(rust_rsi, py_rsi, epsilon = 1e-4);
    });
}

#[test]
fn test_feature_extraction_parity() {
    Python::with_gil(|py| {
        let python_features = py.import("neural_trader.features").unwrap();

        // Load test fixture
        let market_data = load_test_market_data("fixtures/btc_1h_sample.json");

        // Python feature extraction
        let py_market_data = pythonize::pythonize(py, &market_data).unwrap();
        let py_features = python_features
            .call_method1("extract_features", (py_market_data,))
            .unwrap();
        let py_vector: Vec<f64> = depythonize(py_features).unwrap();

        // Rust feature extraction
        let rust_extractor = FeatureExtractor::new(FeatureConfig::default());
        let rust_vector = rust_extractor.extract(&market_data).unwrap();

        // Assert parity
        assert_eq!(py_vector.len(), rust_vector.len(), "Feature vector length mismatch");
        for (i, (py_val, rust_val)) in py_vector.iter().zip(rust_vector.iter()).enumerate() {
            assert_relative_eq!(
                *py_val,
                *rust_val,
                epsilon = 1e-6,
                "Feature {} mismatch: Python={}, Rust={}", i, py_val, rust_val
            );
        }
    });
}

#[test]
fn test_order_execution_parity() {
    // Test complete order execution logic matches Python
    Python::with_gil(|py| {
        let python_orders = py.import("neural_trader.orders").unwrap();

        let signal = Signal {
            action: Action::Buy,
            symbol: "BTC/USD".to_string(),
            confidence: 0.85,
            size: Decimal::from(1),
        };

        let portfolio = Portfolio::default();
        let risk_config = RiskConfig::default();

        // Python order generation
        let py_signal = pythonize::pythonize(py, &signal).unwrap();
        let py_portfolio = pythonize::pythonize(py, &portfolio).unwrap();
        let py_risk = pythonize::pythonize(py, &risk_config).unwrap();

        let py_order = python_orders
            .call_method1("generate_order", (py_signal, py_portfolio, py_risk))
            .unwrap();
        let py_order_data: Order = depythonize(py_order).unwrap();

        // Rust order generation
        let rust_order = generate_order(&signal, &portfolio, &risk_config).unwrap();

        // Assert parity
        assert_eq!(py_order_data.symbol, rust_order.symbol);
        assert_eq!(py_order_data.side, rust_order.side);
        assert_relative_eq!(
            py_order_data.size.to_f64().unwrap(),
            rust_order.size.to_f64().unwrap(),
            epsilon = 1e-8
        );
        assert_relative_eq!(
            py_order_data.price.to_f64().unwrap(),
            rust_order.price.to_f64().unwrap(),
            epsilon = 1e-8
        );
    });
}
```

**Parity Test Matrix:**
| Component | Python Module | Rust Module | Tolerance |
|-----------|---------------|-------------|-----------|
| RSI | `indicators.calculate_rsi` | `indicators::calculate_rsi` | 0.01% |
| SMA | `indicators.calculate_sma` | `indicators::calculate_sma` | 0.001% |
| Feature extraction | `features.FeatureExtractor` | `features::FeatureExtractor` | 0.0001% |
| Position sizing | `risk.position_size` | `risk::position_size` | 0.01% |
| Order generation | `orders.generate_order` | `orders::generate_order` | Exact |

**Parity Test Automation:**
```bash
# Run full parity suite
cargo test --test parity -- --ignored

# Generate parity report
cargo test --test parity -- --ignored --format json | \
  python scripts/parity_report.py > parity_results.html
```

---

## 2. Benchmarking Plan

### 2.1 Criterion.rs Configuration

**Setup:** `benches/benchmarks.rs`
```rust
use criterion::{
    criterion_group, criterion_main,
    Criterion, BenchmarkId, Throughput,
    measurement::WallTime
};
use neural_trader::*;

// Configure criterion for high-precision measurements
fn criterion_config() -> Criterion {
    Criterion::default()
        .significance_level(0.05)
        .sample_size(100)
        .measurement_time(std::time::Duration::from_secs(10))
        .warm_up_time(std::time::Duration::from_secs(3))
}

criterion_group!(
    name = benches;
    config = criterion_config();
    targets =
        bench_market_data_ingestion,
        bench_feature_extraction,
        bench_signal_generation,
        bench_order_placement,
        bench_agentdb_queries,
);
criterion_main!(benches);
```

### 2.2 Critical Path Benchmarks

**Market Data Ingestion (<100μs target)**
```rust
fn bench_market_data_ingestion(c: &mut Criterion) {
    let mut group = c.benchmark_group("market_data_ingestion");
    group.throughput(Throughput::Elements(1)); // 1 tick per iteration

    // Benchmark JSON parsing
    group.bench_function("parse_json_tick", |b| {
        let json = r#"{"symbol":"BTC/USD","price":50000.0,"volume":1.5,"timestamp":1234567890}"#;
        b.iter(|| {
            serde_json::from_str::<Tick>(json).unwrap()
        });
    });

    // Benchmark binary parsing (msgpack)
    group.bench_function("parse_msgpack_tick", |b| {
        let msgpack_data = create_msgpack_tick();
        b.iter(|| {
            rmp_serde::from_slice::<Tick>(&msgpack_data).unwrap()
        });
    });

    // Benchmark WebSocket frame processing
    group.bench_function("websocket_frame_to_tick", |b| {
        let frame = create_websocket_frame();
        let mut parser = WebSocketParser::new();
        b.iter(|| {
            parser.parse_frame(&frame).unwrap()
        });
    });

    group.finish();
}
```

**Feature Extraction (<1ms target)**
```rust
fn bench_feature_extraction(c: &mut Criterion) {
    let mut group = c.benchmark_group("feature_extraction");

    // Setup test data
    let market_data = create_market_data_window(100); // 100 ticks
    let extractor = FeatureExtractor::new(FeatureConfig::default());

    // Benchmark full feature extraction
    group.bench_function("extract_all_features", |b| {
        b.iter(|| {
            extractor.extract(&market_data).unwrap()
        });
    });

    // Benchmark individual feature types
    for feature_type in ["price", "volume", "volatility", "momentum"] {
        group.bench_function(BenchmarkId::new("extract", feature_type), |b| {
            b.iter(|| {
                extractor.extract_specific(&market_data, feature_type).unwrap()
            });
        });
    }

    // Benchmark parallel feature extraction
    group.bench_function("extract_parallel", |b| {
        b.iter(|| {
            extractor.extract_parallel(&market_data).unwrap()
        });
    });

    group.finish();
}
```

**Signal Generation (<5ms target)**
```rust
fn bench_signal_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("signal_generation");

    let features = create_test_features();

    // Benchmark different strategies
    let strategies = vec![
        ("momentum", MomentumStrategy::new()),
        ("mean_reversion", MeanReversionStrategy::new()),
        ("ml_ensemble", MLEnsembleStrategy::new()),
    ];

    for (name, strategy) in strategies {
        group.bench_function(BenchmarkId::new("strategy", name), |b| {
            b.iter(|| {
                strategy.generate_signal(&features).unwrap()
            });
        });
    }

    group.finish();
}
```

**Order Placement (<10ms target)**
```rust
fn bench_order_placement(c: &mut Criterion) {
    let mut group = c.benchmark_group("order_placement");

    let rt = tokio::runtime::Runtime::new().unwrap();
    let mock_exchange = MockExchange::new();

    group.bench_function("place_market_order", |b| {
        b.to_async(&rt).iter(|| async {
            let order = Order::market("BTC/USD", Decimal::from(1));
            mock_exchange.place_order(order).await.unwrap()
        });
    });

    group.bench_function("place_limit_order", |b| {
        b.to_async(&rt).iter(|| async {
            let order = Order::limit("BTC/USD", Decimal::from(1), Decimal::from(50000));
            mock_exchange.place_order(order).await.unwrap()
        });
    });

    group.bench_function("cancel_order", |b| {
        b.to_async(&rt).iter(|| async {
            mock_exchange.cancel_order("test_order_id").await.unwrap()
        });
    });

    group.finish();
}
```

**AgentDB Queries (<1ms target)**
```rust
fn bench_agentdb_queries(c: &mut Criterion) {
    let mut group = c.benchmark_group("agentdb_queries");

    let rt = tokio::runtime::Runtime::new().unwrap();
    let db = rt.block_on(async {
        let db = AgentDB::open_in_memory().await.unwrap();
        // Populate with test data
        for i in 0..10000 {
            db.insert("patterns", &create_test_pattern(i)).await.unwrap();
        }
        db
    });

    // Benchmark vector similarity search
    group.bench_function("vector_search_k10", |b| {
        let query_vector = vec![0.5; 128];
        b.to_async(&rt).iter(|| async {
            db.query_similar("patterns", &query_vector, 10).await.unwrap()
        });
    });

    // Benchmark indexed lookup
    group.bench_function("indexed_lookup", |b| {
        b.to_async(&rt).iter(|| async {
            db.get_by_id("patterns", "pattern_5000").await.unwrap()
        });
    });

    // Benchmark batch insert
    group.bench_function("batch_insert_100", |b| {
        let patterns: Vec<_> = (0..100).map(create_test_pattern).collect();
        b.to_async(&rt).iter(|| async {
            db.batch_insert("patterns", &patterns).await.unwrap()
        });
    });

    group.finish();
}
```

### 2.3 Throughput Benchmarks

```rust
fn bench_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("throughput");

    // Events per second
    group.throughput(Throughput::Elements(1000));
    group.bench_function("process_1000_ticks", |b| {
        let ticks = create_test_ticks(1000);
        let mut processor = TickProcessor::new();
        b.iter(|| {
            for tick in &ticks {
                processor.process(tick).unwrap();
            }
        });
    });

    // Orders per second
    let rt = tokio::runtime::Runtime::new().unwrap();
    group.bench_function("place_100_orders_concurrent", |b| {
        let orders = create_test_orders(100);
        let exchange = MockExchange::new();
        b.to_async(&rt).iter(|| async {
            let futures = orders.iter().map(|order| exchange.place_order(order.clone()));
            futures::future::join_all(futures).await
        });
    });

    group.finish();
}
```

### 2.4 Memory Profiling

**Heaptrack Integration:**
```bash
# Profile memory usage
heaptrack target/release/neural-trader --run-duration 60s

# Analyze results
heaptrack_gui heaptrack.neural-trader.*.gz
```

**Valgrind Massif:**
```bash
# Generate memory profile
valgrind --tool=massif --massif-out-file=massif.out \
  target/release/neural-trader --run-duration 60s

# Visualize
ms_print massif.out > memory_profile.txt
```

**Rust-Specific Memory Tracking:**
```rust
// benches/memory_bench.rs
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

struct MemoryTracker;

static ALLOCATED: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for MemoryTracker {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ret = System.alloc(layout);
        if !ret.is_null() {
            ALLOCATED.fetch_add(layout.size(), Ordering::SeqCst);
        }
        ret
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
        ALLOCATED.fetch_sub(layout.size(), Ordering::SeqCst);
    }
}

#[global_allocator]
static GLOBAL: MemoryTracker = MemoryTracker;

#[test]
fn test_memory_footprint() {
    let initial = ALLOCATED.load(Ordering::SeqCst);

    // Run trading system for 60 seconds
    let mut system = TradingSystem::new(Config::default());
    system.run_for_duration(Duration::from_secs(60)).unwrap();

    let final_mem = ALLOCATED.load(Ordering::SeqCst);
    let footprint = final_mem - initial;

    assert!(
        footprint < 500 * 1024 * 1024,
        "Memory footprint {} MB exceeds 500MB limit",
        footprint / 1024 / 1024
    );
}
```

### 2.5 Comparison Baselines

**Python vs Rust Comparison:**
```rust
// benches/python_comparison.rs
use pyo3::prelude::*;
use criterion::{criterion_group, criterion_main, Criterion};

fn bench_rust_vs_python(c: &mut Criterion) {
    let mut group = c.benchmark_group("rust_vs_python");

    Python::with_gil(|py| {
        let python_indicators = py.import("neural_trader.indicators").unwrap();
        let prices = vec![44.34, 44.09, 43.61, /* ... */];

        // Python benchmark
        group.bench_function("python_rsi", |b| {
            let py_prices = pyo3::types::PyList::new(py, &prices);
            b.iter(|| {
                python_indicators
                    .call_method1("calculate_rsi", (py_prices, 14))
                    .unwrap()
            });
        });

        // Rust benchmark
        group.bench_function("rust_rsi", |b| {
            b.iter(|| {
                calculate_rsi(&prices, 14).unwrap()
            });
        });
    });

    group.finish();
}

criterion_group!(benches, bench_rust_vs_python);
criterion_main!(benches);
```

---

## 3. Performance Targets

### 3.1 Latency Targets (Percentiles)

| Operation | p50 | p95 | p99 | p99.9 |
|-----------|-----|-----|-----|-------|
| Market data ingestion | 50μs | 80μs | 100μs | 150μs |
| Feature extraction | 500μs | 800μs | 1ms | 2ms |
| Signal generation | 2ms | 4ms | 5ms | 8ms |
| Order placement | 5ms | 8ms | 10ms | 15ms |
| AgentDB indexed query | 100μs | 500μs | 1ms | 2ms |
| AgentDB vector search | 500μs | 800μs | 1ms | 3ms |
| Risk check | 100μs | 200μs | 300μs | 500μs |
| Portfolio update | 50μs | 100μs | 150μs | 200μs |

### 3.2 Throughput Targets

| Metric | Target | Measurement |
|--------|--------|-------------|
| Market ticks processed | 10,000/sec | Per symbol |
| Features extracted | 1,000/sec | Full feature vector |
| Signals generated | 500/sec | All strategies |
| Orders placed | 100/sec | Concurrent |
| AgentDB inserts | 5,000/sec | Batch operations |
| AgentDB queries | 10,000/sec | Indexed lookups |

### 3.3 Memory Targets

| Component | Target | Maximum |
|-----------|--------|---------|
| Base system | 200MB | 300MB |
| Per symbol (streaming) | 10MB | 20MB |
| AgentDB cache | 100MB | 200MB |
| Feature buffers | 50MB | 100MB |
| Order book | 20MB | 50MB |
| Total system | 400MB | 500MB |

### 3.4 CPU Utilization Targets

| Component | Target CPU% | Max CPU% |
|-----------|-------------|----------|
| Market data ingestion | 10% | 20% |
| Feature extraction | 20% | 40% |
| Signal generation | 15% | 30% |
| AgentDB operations | 10% | 20% |
| Total (4 cores) | 55% | 110% |

---

## 4. CI/CD Pipeline

### 4.1 GitHub Actions Workflow

**Main CI Pipeline:** `.github/workflows/ci.yml`
```yaml
name: Neural Trading CI/CD

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 0 * * *' # Nightly builds

env:
  CARGO_TERM_COLOR: always
  RUST_BACKTRACE: 1

jobs:
  # Job 1: Fast checks (runs first)
  quick-checks:
    name: Quick Checks
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Rust
        uses: dtolnay/rust-toolchain@stable
        with:
          components: rustfmt, clippy

      - name: Check formatting
        run: cargo fmt --all -- --check

      - name: Clippy lints
        run: cargo clippy --all-targets --all-features -- -D warnings

      - name: Check documentation
        run: cargo doc --no-deps --all-features

  # Job 2: Build matrix
  build:
    name: Build (${{ matrix.os }} / Node ${{ matrix.node }})
    needs: quick-checks
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        node: ['18.x', '20.x', '22.x']
        rust: [stable, nightly]
        exclude:
          # Reduce matrix size - only test all Node versions on Linux
          - os: macos-latest
            node: '18.x'
          - os: macos-latest
            node: '20.x'
          - os: windows-latest
            node: '18.x'
          - os: windows-latest
            node: '20.x'

    runs-on: ${{ matrix.os }}

    steps:
      - uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: ${{ matrix.node }}

      - name: Setup Rust
        uses: dtolnay/rust-toolchain@master
        with:
          toolchain: ${{ matrix.rust }}

      - name: Cache Cargo registry
        uses: actions/cache@v3
        with:
          path: ~/.cargo/registry
          key: ${{ runner.os }}-cargo-registry-${{ hashFiles('**/Cargo.lock') }}

      - name: Cache Cargo build
        uses: actions/cache@v3
        with:
          path: target
          key: ${{ runner.os }}-cargo-build-${{ matrix.rust }}-${{ hashFiles('**/Cargo.lock') }}

      - name: Build (debug)
        run: cargo build --verbose

      - name: Build (release)
        run: cargo build --release --verbose

      - name: Build with all features
        run: cargo build --all-features --verbose

      - name: Build with no default features
        run: cargo build --no-default-features --verbose

  # Job 3: Test suite
  test:
    name: Test (${{ matrix.os }} / ${{ matrix.test-suite }})
    needs: build
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        test-suite: [unit, integration, parity]
        rust: [stable]

    runs-on: ${{ matrix.os }}

    steps:
      - uses: actions/checkout@v4

      - name: Setup Rust
        uses: dtolnay/rust-toolchain@stable

      - name: Setup Python (for parity tests)
        if: matrix.test-suite == 'parity'
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install Python dependencies
        if: matrix.test-suite == 'parity'
        run: |
          pip install -r python_impl/requirements.txt

      - name: Run unit tests
        if: matrix.test-suite == 'unit'
        run: cargo test --lib --bins

      - name: Run integration tests
        if: matrix.test-suite == 'integration'
        run: cargo test --test '*' --features integration-tests

      - name: Run parity tests
        if: matrix.test-suite == 'parity'
        run: cargo test --test parity -- --ignored

      - name: Upload test results
        if: always()
        uses: actions/upload-artifact@v3
        with:
          name: test-results-${{ matrix.os }}-${{ matrix.test-suite }}
          path: target/test-results/

  # Job 4: Code coverage
  coverage:
    name: Code Coverage
    needs: test
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Setup Rust
        uses: dtolnay/rust-toolchain@stable

      - name: Install tarpaulin
        run: cargo install cargo-tarpaulin

      - name: Generate coverage
        run: |
          cargo tarpaulin --out Xml --output-dir coverage \
            --exclude-files 'tests/*' 'benches/*' \
            --ignore-panics --ignore-tests

      - name: Upload to Codecov
        uses: codecov/codecov-action@v3
        with:
          files: ./coverage/cobertura.xml
          fail_ci_if_error: true

      - name: Check coverage threshold
        run: |
          COVERAGE=$(cargo tarpaulin --output-dir /tmp | grep -oP '\d+\.\d+(?=%)')
          echo "Coverage: $COVERAGE%"
          if (( $(echo "$COVERAGE < 90.0" | bc -l) )); then
            echo "Coverage $COVERAGE% is below 90% threshold"
            exit 1
          fi

  # Job 5: Benchmarks
  benchmarks:
    name: Performance Benchmarks
    needs: test
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Setup Rust
        uses: dtolnay/rust-toolchain@stable

      - name: Run benchmarks
        run: |
          cargo bench --no-fail-fast -- --save-baseline ci-baseline

      - name: Check performance regression
        run: |
          cargo bench -- --baseline ci-baseline --load-baseline main

      - name: Upload benchmark results
        uses: actions/upload-artifact@v3
        with:
          name: benchmark-results
          path: target/criterion/

      - name: Generate performance report
        run: |
          cargo install cargo-criterion
          cargo criterion --message-format json > benchmark-results.json

      - name: Validate performance targets
        run: |
          python scripts/validate_performance.py benchmark-results.json

  # Job 6: Security audit
  security:
    name: Security Audit
    needs: quick-checks
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Setup Rust
        uses: dtolnay/rust-toolchain@stable

      - name: Install cargo-audit
        run: cargo install cargo-audit

      - name: Run security audit
        run: cargo audit --deny warnings

      - name: Install cargo-deny
        run: cargo install cargo-deny

      - name: Check licenses
        run: cargo deny check licenses

      - name: Check advisories
        run: cargo deny check advisories

      - name: Check bans
        run: cargo deny check bans

  # Job 7: Fuzz testing (nightly only)
  fuzz:
    name: Fuzz Testing
    if: github.event_name == 'schedule'
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Setup Rust nightly
        uses: dtolnay/rust-toolchain@nightly

      - name: Install cargo-fuzz
        run: cargo install cargo-fuzz

      - name: Run fuzz tests (1 hour each)
        run: |
          for target in $(cargo fuzz list); do
            echo "Fuzzing $target..."
            timeout 3600 cargo fuzz run $target -- -max_total_time=3600 || true
          done

      - name: Upload crash artifacts
        if: always()
        uses: actions/upload-artifact@v3
        with:
          name: fuzz-crashes
          path: fuzz/artifacts/

  # Job 8: E2E tests (expensive, run selectively)
  e2e:
    name: End-to-End Tests
    if: github.ref == 'refs/heads/main' || github.event_name == 'schedule'
    needs: test
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Setup Rust
        uses: dtolnay/rust-toolchain@stable

      - name: Setup test environment
        run: |
          docker-compose -f tests/e2e/docker-compose.yml up -d

      - name: Run E2E tests
        run: cargo test --test e2e -- --ignored --test-threads=1

      - name: Collect logs
        if: always()
        run: |
          docker-compose -f tests/e2e/docker-compose.yml logs > e2e-logs.txt

      - name: Upload E2E artifacts
        if: always()
        uses: actions/upload-artifact@v3
        with:
          name: e2e-artifacts
          path: |
            e2e-logs.txt
            tests/e2e/screenshots/

  # Job 9: MCP compliance check
  mcp-compliance:
    name: MCP Compliance Check
    needs: build
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '20.x'

      - name: Install MCP validator
        run: npm install -g @modelcontextprotocol/validator

      - name: Validate MCP implementation
        run: |
          cargo build --release
          mcp-validator validate target/release/neural-trader-mcp

      - name: Test MCP endpoints
        run: |
          cargo test --features mcp-integration

  # Job 10: Documentation build
  docs:
    name: Build Documentation
    needs: quick-checks
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Setup Rust
        uses: dtolnay/rust-toolchain@stable

      - name: Build rustdoc
        run: cargo doc --no-deps --all-features --document-private-items

      - name: Deploy to GitHub Pages
        if: github.ref == 'refs/heads/main'
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./target/doc

  # Job 11: Release build (on tags)
  release:
    name: Release Build
    if: startsWith(github.ref, 'refs/tags/')
    needs: [test, benchmarks, security, mcp-compliance]
    strategy:
      matrix:
        include:
          - os: ubuntu-latest
            target: x86_64-unknown-linux-gnu
          - os: macos-latest
            target: x86_64-apple-darwin
          - os: macos-latest
            target: aarch64-apple-darwin
          - os: windows-latest
            target: x86_64-pc-windows-msvc

    runs-on: ${{ matrix.os }}

    steps:
      - uses: actions/checkout@v4

      - name: Setup Rust
        uses: dtolnay/rust-toolchain@stable
        with:
          targets: ${{ matrix.target }}

      - name: Build release binary
        run: |
          cargo build --release --target ${{ matrix.target }}

      - name: Package binary
        run: |
          cd target/${{ matrix.target }}/release
          tar czf neural-trader-${{ matrix.target }}.tar.gz neural-trader*

      - name: Upload release asset
        uses: actions/upload-release-asset@v1
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        with:
          upload_url: ${{ github.event.release.upload_url }}
          asset_path: target/${{ matrix.target }}/release/neural-trader-${{ matrix.target }}.tar.gz
          asset_name: neural-trader-${{ matrix.target }}.tar.gz
          asset_content_type: application/gzip
```

### 4.2 Quality Gate Configuration

**Cargo.toml Quality Settings:**
```toml
[profile.dev]
opt-level = 0
debug = true

[profile.test]
opt-level = 1
debug = true

[profile.release]
opt-level = 3
lto = "fat"
codegen-units = 1
strip = true
panic = "abort"

[profile.bench]
inherits = "release"
debug = true

# Quality gates enforced by CI
[package.metadata.quality-gates]
min-coverage = 90.0
max-clippy-warnings = 0
require-doc-comments = true
```

**deny.toml (cargo-deny configuration):**
```toml
[advisories]
vulnerability = "deny"
unmaintained = "warn"
notice = "warn"

[licenses]
unlicensed = "deny"
allow = [
    "MIT",
    "Apache-2.0",
    "BSD-3-Clause",
]
deny = [
    "GPL-3.0",
    "AGPL-3.0",
]

[bans]
multiple-versions = "warn"
deny = [
    { name = "openssl", wrappers = ["native-tls"] }
]

[sources]
unknown-registry = "deny"
unknown-git = "deny"
```

---

## 5. Test Execution & Reporting

### 5.1 Local Test Execution

```bash
# Run all tests
cargo test

# Run specific test suite
cargo test --lib                    # Unit tests
cargo test --test integration       # Integration tests
cargo test --test e2e -- --ignored  # E2E tests (expensive)
cargo test --test parity -- --ignored  # Parity tests

# Run with coverage
cargo tarpaulin --out Html --output-dir coverage

# Run benchmarks
cargo bench

# Run fuzz tests (5 min each)
cargo +nightly fuzz run order_parser -- -max_total_time=300
```

### 5.2 Test Report Generation

**scripts/test_report.py:**
```python
#!/usr/bin/env python3
import json
import sys
from pathlib import Path

def generate_test_report(results_file):
    with open(results_file) as f:
        results = json.load(f)

    total_tests = results['test_count']
    passed = results['passed']
    failed = results['failed']
    ignored = results['ignored']

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Neural Trading Test Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .summary {{ background: #f0f0f0; padding: 20px; border-radius: 5px; }}
            .pass {{ color: green; }}
            .fail {{ color: red; }}
            .metric {{ margin: 10px 0; }}
        </style>
    </head>
    <body>
        <h1>Neural Trading Test Report</h1>
        <div class="summary">
            <div class="metric"><strong>Total Tests:</strong> {total_tests}</div>
            <div class="metric pass"><strong>Passed:</strong> {passed}</div>
            <div class="metric fail"><strong>Failed:</strong> {failed}</div>
            <div class="metric"><strong>Ignored:</strong> {ignored}</div>
            <div class="metric"><strong>Success Rate:</strong> {passed/total_tests*100:.2f}%</div>
        </div>
        <h2>Failed Tests</h2>
        <ul>
    """

    for test in results.get('failed_tests', []):
        html += f"<li><strong>{test['name']}</strong>: {test['message']}</li>"

    html += """
        </ul>
    </body>
    </html>
    """

    Path('test_report.html').write_text(html)
    print("Test report generated: test_report.html")

if __name__ == '__main__':
    generate_test_report(sys.argv[1])
```

### 5.3 Continuous Monitoring

**scripts/monitor_performance.sh:**
```bash
#!/bin/bash

# Monitor performance metrics over time
BASELINE_DIR="benchmarks/baselines"
CURRENT_RUN="benchmarks/current"

mkdir -p "$BASELINE_DIR" "$CURRENT_RUN"

# Run benchmarks
cargo bench -- --save-baseline current

# Compare with main branch baseline
if [ -f "$BASELINE_DIR/main.json" ]; then
    cargo bench -- --baseline main --load-baseline current > comparison.txt

    # Check for regressions
    if grep -q "regressed" comparison.txt; then
        echo "⚠️ Performance regression detected!"
        cat comparison.txt
        exit 1
    else
        echo "✅ No performance regressions"
    fi
fi

# Update baseline on main branch
if [ "$GITHUB_REF" == "refs/heads/main" ]; then
    cp -r target/criterion "$BASELINE_DIR/main"
fi
```

---

## 6. Best Practices & Guidelines

### 6.1 Test Writing Guidelines

1. **Naming Convention:**
   - Tests: `test_<component>_<scenario>_<expected_outcome>`
   - Benchmarks: `bench_<operation>_<variant>`

2. **Test Structure (AAA Pattern):**
   ```rust
   #[test]
   fn test_example() {
       // Arrange - Setup test data
       let input = create_test_input();

       // Act - Execute function
       let result = function_under_test(input);

       // Assert - Verify outcome
       assert_eq!(result, expected_output);
   }
   ```

3. **Error Testing:**
   ```rust
   #[test]
   fn test_error_handling() {
       let result = risky_operation();
       assert!(result.is_err());
       assert_eq!(result.unwrap_err().kind(), ErrorKind::Expected);
   }
   ```

4. **Async Testing:**
   ```rust
   #[tokio::test]
   async fn test_async_operation() {
       let result = async_function().await;
       assert!(result.is_ok());
   }
   ```

### 6.2 Mock Strategy

**Use Real Components When Possible:**
```rust
// ✅ Good: Use real FeatureExtractor with test data
let extractor = FeatureExtractor::new(test_config);
let features = extractor.extract(&test_market_data).unwrap();

// ❌ Bad: Mock when not necessary
let mut mock_extractor = MockFeatureExtractor::new();
mock_extractor.expect_extract().returning(|_| Ok(fake_features));
```

**Mock External Dependencies:**
```rust
// ✅ Good: Mock exchange API
let mock_exchange = MockExchange::new();
mock_exchange.expect_place_order()
    .returning(|order| Ok(OrderResponse::filled(order)));
```

### 6.3 Test Data Management

**Fixtures:**
```rust
// tests/fixtures/mod.rs
pub fn load_market_data_fixture(name: &str) -> Vec<Tick> {
    let path = format!("tests/fixtures/market_data/{}.json", name);
    let data = std::fs::read_to_string(path).unwrap();
    serde_json::from_str(&data).unwrap()
}

// Usage
#[test]
fn test_with_real_data() {
    let ticks = load_market_data_fixture("btc_volatile_2024");
    // Test with realistic data
}
```

### 6.4 Performance Testing Best Practices

1. **Warm-up:** Always warm up before measuring
2. **Statistical significance:** Use enough samples
3. **Isolation:** Run benchmarks on idle systems
4. **Reproducibility:** Use fixed seeds for randomness
5. **Baseline comparison:** Always compare against previous runs

---

## 7. Troubleshooting & Debugging

### 7.1 Test Failures

**Flaky Tests:**
```bash
# Detect flaky tests (run 100 times)
cargo test --test flaky_test -- --test-threads=1 --nocapture --exact || \
for i in {1..100}; do cargo test --test flaky_test; done | grep -i failed
```

**Debug Specific Test:**
```rust
#[test]
fn test_with_debug_output() {
    env_logger::init(); // Enable logging

    let result = complex_operation();
    eprintln!("Debug: {:?}", result); // Will show with --nocapture

    assert!(result.is_ok());
}
```

### 7.2 Benchmark Debugging

```bash
# Run single benchmark with output
cargo bench --bench market_data -- --verbose parse_json_tick

# Profile benchmark with flamegraph
cargo flamegraph --bench market_data -- --bench
```

### 7.3 CI Debugging

```yaml
# Add SSH access to failing CI job
- name: Setup tmate session
  if: failure()
  uses: mxschmitt/action-tmate@v3
```

---

## Appendix: Quick Reference

### Test Commands
```bash
cargo test                          # All tests
cargo test --lib                    # Unit tests only
cargo test --test integration       # Integration tests
cargo test --doc                    # Documentation tests
cargo test -- --ignored             # Ignored tests (E2E, parity)
cargo test -- --nocapture           # Show output
cargo test -- --test-threads=1      # Serial execution
```

### Benchmark Commands
```bash
cargo bench                         # Run all benchmarks
cargo bench --bench name            # Run specific benchmark
cargo bench -- --save-baseline b1   # Save baseline
cargo bench -- --baseline b1        # Compare with baseline
```

### Coverage Commands
```bash
cargo tarpaulin                     # Generate coverage
cargo tarpaulin --out Html          # HTML report
cargo tarpaulin --ignore-tests      # Exclude test code
```

### Quality Check Commands
```bash
cargo fmt --check                   # Check formatting
cargo clippy -- -D warnings         # Linting
cargo audit                         # Security audit
cargo deny check                    # License/dependency check
```

---

**Summary:** This testing strategy ensures the Neural Trading Rust port meets the highest standards of quality, performance, and reliability through comprehensive automated testing, rigorous benchmarking, and continuous integration across all target platforms.
