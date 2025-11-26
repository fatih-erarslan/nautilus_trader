# Testing, Benchmarking, and CI/CD Strategy

## Document Purpose

This document defines the **complete testing strategy, benchmark plan, and CI/CD pipeline** for the Neural Rust port. It ensures 90%+ test coverage, performance validation, and quality gates across all platforms.

## Table of Contents

1. [Test Hierarchy](#test-hierarchy)
2. [Test Organization](#test-organization)
3. [Benchmark Plan](#benchmark-plan)
4. [CI/CD Pipeline](#cicd-pipeline)
5. [Quality Gates](#quality-gates)
6. [Test Data Generation](#test-data-generation)
7. [Troubleshooting](#troubleshooting)

---

## Test Hierarchy

### 1. Unit Tests (70% of test suite)

**Purpose:** Validate individual functions and methods in isolation

**Framework:** Built-in Rust `#[test]`, `tokio::test` for async

**Coverage Target:** 95% of public APIs

**Example:**

```rust
// crates/strategies/src/momentum.rs
#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn test_calculate_momentum_uptrend() {
        let prices = vec![dec!(100), dec!(102), dec!(105), dec!(110)];
        let momentum = calculate_z_score_momentum(&prices);

        assert!(momentum > 0.0, "Uptrend should have positive momentum");
        assert!(momentum > 1.5, "Strong uptrend should exceed 1.5 sigma");
    }

    #[test]
    fn test_calculate_momentum_empty_prices() {
        let prices = vec![];
        let result = calculate_z_score_momentum(&prices);

        assert!(result.is_nan(), "Empty price series should return NaN");
    }

    #[tokio::test]
    async fn test_momentum_strategy_long_signal() {
        let strategy = MomentumStrategy::new(
            LookbackPeriod::Days(14),
            2.0, // threshold
        );

        let market_data = MarketDataBuilder::new()
            .add_uptrend(14, dec!(100), dec!(110))
            .build();

        let portfolio = Portfolio::mock_with_cash(dec!(10000));

        let signals = strategy.process(&market_data, &portfolio).await.unwrap();

        assert_eq!(signals.len(), 1);
        assert_eq!(signals[0].direction, Direction::Long);
        assert!(signals[0].confidence > 0.7);
    }
}
```

**Validation Criteria:**
- ✅ All public functions have ≥3 test cases
- ✅ Edge cases tested (empty, null, extreme values)
- ✅ Error paths validated
- ✅ Async functions use `tokio::test`

---

### 2. Property-Based Tests (10% of test suite)

**Purpose:** Validate invariants across random inputs

**Framework:** `proptest` crate

**Coverage Target:** Critical algorithms and data structures

**Example:**

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_position_size_never_exceeds_portfolio(
        confidence in 0.0..1.0f64,
        portfolio_value in 1000.0..1_000_000.0f64
    ) {
        let signal = Signal {
            direction: Direction::Long,
            confidence,
            ..Default::default()
        };

        let portfolio = Portfolio::mock_with_value(
            Decimal::from_f64_retain(portfolio_value).unwrap()
        );

        let position_size = calculate_position_size(&signal, &portfolio);

        // Invariant: Position size never exceeds portfolio value
        prop_assert!(position_size <= portfolio.total_value());

        // Invariant: Position size is non-negative
        prop_assert!(position_size >= Decimal::ZERO);
    }

    #[test]
    fn test_kelly_criterion_bounded(
        win_rate in 0.0..1.0f64,
        avg_win in 1.0..10.0f64,
        avg_loss in 1.0..10.0f64
    ) {
        let kelly = calculate_kelly_fraction(win_rate, avg_win, avg_loss);

        // Invariant: Kelly fraction between -1 and 1
        prop_assert!(kelly >= -1.0 && kelly <= 1.0);
    }
}
```

**Validation Criteria:**
- ✅ All financial calculations have property tests
- ✅ Invariants documented in comments
- ✅ Run with 1000+ random inputs
- ✅ Seed reproducibility (`PROPTEST_MAX_SHRINK_ITERS`)

---

### 3. Integration Tests (15% of test suite)

**Purpose:** Validate component interactions

**Location:** `tests/` directory (separate from `src/`)

**Coverage Target:** All major data flows

**Example:**

```rust
// tests/integration_test.rs
use neural_trader::prelude::*;
use tokio::time::{sleep, Duration};

#[tokio::test]
async fn test_market_data_to_signal_flow() {
    // Setup: Start market data manager
    let (tx, mut rx) = tokio::sync::mpsc::channel(100);
    let market_mgr = MarketDataManager::new(tx);

    // Setup: Start strategy engine
    let strategy = MomentumStrategy::new(LookbackPeriod::Minutes(5), 2.0);
    let mut engine = StrategyEngine::new(vec![Box::new(strategy)]);

    // Action: Inject market ticks
    for i in 0..100 {
        market_mgr.ingest_tick(MarketTick {
            symbol: "AAPL".to_string(),
            price: Decimal::new(150 + i, 0),
            timestamp: Utc::now().timestamp(),
            ..Default::default()
        }).await.unwrap();
    }

    // Wait for processing
    sleep(Duration::from_millis(100)).await;

    // Assert: Strategy generates signals
    let signals = rx.try_recv().ok();
    assert!(signals.is_some(), "Strategy should generate signals from market data");
}

#[tokio::test]
async fn test_signal_to_order_execution() {
    // Setup: Mock broker API
    let broker = MockAlpacaClient::new();
    let executor = OrderExecutor::new(Box::new(broker.clone()));

    // Action: Submit signal
    let signal = Signal {
        strategy_id: Uuid::new_v4(),
        symbol: "TSLA".to_string(),
        direction: Direction::Long,
        confidence: 0.85,
        position_size: dec!(1000),
        reasoning: "Strong momentum".to_string(),
    };

    let order = executor.execute_signal(signal).await.unwrap();

    // Assert: Order placed with broker
    assert_eq!(broker.orders_placed(), 1);
    assert_eq!(order.symbol, "TSLA");
    assert_eq!(order.side, OrderSide::Buy);
}
```

**Validation Criteria:**
- ✅ End-to-end data flows tested
- ✅ Component boundaries validated
- ✅ Error propagation verified
- ✅ Async coordination tested

---

### 4. End-to-End (E2E) Tests (3% of test suite)

**Purpose:** Validate complete system behavior

**Framework:** Custom test harness with mock exchanges

**Coverage Target:** Critical user scenarios

**Example:**

```rust
// tests/e2e_paper_trading.rs
#[tokio::test]
#[ignore] // Run with: cargo test --ignored
async fn test_complete_paper_trading_session() {
    // Setup: Initialize system with test config
    let config = Config::from_file("tests/fixtures/test_config.toml").unwrap();
    let mut trader = NeuralTrader::new(config).await.unwrap();

    // Use mock data source
    trader.set_data_source(Box::new(MockMarketDataSource::new(
        "tests/fixtures/historical_data.csv"
    )));

    // Use paper trading mode
    trader.set_execution_mode(ExecutionMode::Paper);

    // Action: Run for simulated 1 hour
    let start = Instant::now();
    trader.start().await.unwrap();

    // Simulate 1 hour with 1-minute ticks
    for _ in 0..60 {
        tokio::time::sleep(Duration::from_millis(100)).await; // Fast-forward
    }

    trader.stop().await.unwrap();

    // Assert: System behaves correctly
    let metrics = trader.get_metrics();

    assert!(metrics.signals_generated > 0, "Should generate signals");
    assert!(metrics.orders_placed > 0, "Should place orders");
    assert_eq!(metrics.errors, 0, "Should have no errors");
    assert!(metrics.latency_p95 < Duration::from_millis(200), "Should meet latency SLA");
}

#[tokio::test]
#[ignore]
async fn test_backtest_full_pipeline() {
    let config = BacktestConfig {
        start_date: NaiveDate::from_ymd(2024, 1, 1),
        end_date: NaiveDate::from_ymd(2024, 12, 31),
        initial_capital: dec!(100000),
        strategies: vec!["momentum", "mean_reversion"],
        ..Default::default()
    };

    let results = run_backtest(config).await.unwrap();

    // Validate results
    assert!(results.total_return > dec!(-0.5), "Should not lose >50%");
    assert!(results.sharpe_ratio > 0.0, "Should have positive Sharpe");
    assert_eq!(results.trades.len(), results.trade_count);
}
```

**Validation Criteria:**
- ✅ All user workflows tested
- ✅ Real-world scenarios covered
- ✅ Performance validated under load
- ✅ Recovery from failures tested

---

### 5. Fuzz Tests (1% of test suite)

**Purpose:** Discover crash bugs and panics

**Framework:** `cargo-fuzz` with libFuzzer

**Coverage Target:** Parsers, deserializers, untrusted inputs

**Example:**

```rust
// fuzz/fuzz_targets/market_tick_parser.rs
#![no_main]
use libfuzzer_sys::fuzz_target;
use neural_trader::market_data::MarketTick;

fuzz_target!(|data: &[u8]| {
    // Try to parse arbitrary bytes as market tick JSON
    if let Ok(s) = std::str::from_utf8(data) {
        let _ = serde_json::from_str::<MarketTick>(s);
        // Should never panic, even on malformed input
    }
});

// fuzz/fuzz_targets/order_amount.rs
fuzz_target!(|data: &[u8]| {
    if data.len() >= 8 {
        let amount = f64::from_le_bytes(data[0..8].try_into().unwrap());

        // Should never panic on any f64 value
        let _ = calculate_position_size_from_amount(amount);
    }
});
```

**Running Fuzz Tests:**

```bash
# Install cargo-fuzz
cargo install cargo-fuzz

# Run fuzzer for 10 minutes
cargo fuzz run market_tick_parser -- -max_total_time=600

# Check coverage
cargo fuzz coverage market_tick_parser
```

**Validation Criteria:**
- ✅ All parsers fuzz-tested
- ✅ No panics on malformed input
- ✅ Run for ≥1 hour in CI
- ✅ Corpus saved for regression

---

### 6. Parity Tests (1% of test suite)

**Purpose:** Ensure Rust behavior matches Python

**Framework:** Custom comparison harness

**Coverage Target:** All migrated features

**Example:**

```rust
// tests/parity_test.rs
use pyo3::prelude::*;
use pyo3::types::PyModule;

#[test]
fn test_momentum_strategy_parity() {
    Python::with_gil(|py| {
        // Load Python implementation
        let py_module = PyModule::from_code(
            py,
            include_str!("../python_reference/momentum.py"),
            "momentum.py",
            "momentum",
        ).unwrap();

        let py_strategy = py_module.getattr("MomentumStrategy").unwrap();
        let py_instance = py_strategy.call1((14, 2.0)).unwrap();

        // Create Rust implementation
        let rust_strategy = MomentumStrategy::new(LookbackPeriod::Days(14), 2.0);

        // Test data
        let prices: Vec<f64> = (0..30).map(|i| 100.0 + i as f64 * 0.5).collect();

        // Python signal
        let py_signal: f64 = py_instance
            .call_method1("calculate_signal", (prices.clone(),))
            .unwrap()
            .extract()
            .unwrap();

        // Rust signal
        let rust_signal = rust_strategy.calculate_signal(&prices);

        // Assert: Results match within tolerance
        assert!(
            (py_signal - rust_signal).abs() < 1e-6,
            "Python: {}, Rust: {} - Parity violation!",
            py_signal,
            rust_signal
        );
    });
}
```

**Validation Criteria:**
- ✅ All Python features have parity tests
- ✅ Results match within 1e-6 tolerance
- ✅ Performance is ≥10x faster in Rust
- ✅ Edge cases match Python behavior

---

## Test Organization

### Directory Structure

```
neural-trader/
├── crates/
│   ├── core/
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── types.rs
│   │   │   └── traits.rs
│   │   └── tests/           # Unit tests embedded
│   ├── strategies/
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── momentum.rs
│   │   │   └── mean_reversion.rs
│   │   └── tests/
│   │       └── strategy_tests.rs
│   └── .../
├── tests/                   # Integration tests
│   ├── integration/
│   │   ├── market_data_flow.rs
│   │   ├── execution_pipeline.rs
│   │   └── risk_management.rs
│   ├── e2e/
│   │   ├── paper_trading.rs
│   │   ├── backtesting.rs
│   │   └── live_simulation.rs
│   ├── parity/
│   │   ├── strategies.rs
│   │   ├── indicators.rs
│   │   └── risk_models.rs
│   └── fixtures/            # Test data
│       ├── historical_data.csv
│       ├── test_config.toml
│       └── mock_responses/
├── fuzz/
│   ├── Cargo.toml
│   └── fuzz_targets/
│       ├── market_tick_parser.rs
│       ├── order_amount.rs
│       └── config_parser.rs
└── benches/                 # Benchmarks (see below)
```

### Test Naming Conventions

```rust
// ✅ GOOD: Descriptive test names
#[test]
fn test_momentum_generates_long_signal_on_uptrend() { }

#[test]
fn test_position_size_respects_max_allocation_limit() { }

#[test]
fn test_order_rejected_when_insufficient_funds() { }

// ❌ BAD: Vague test names
#[test]
fn test_momentum() { }

#[test]
fn test_order() { }
```

### Test Modules

```rust
// Organize tests by feature
#[cfg(test)]
mod tests {
    use super::*;

    mod momentum {
        use super::*;

        #[test]
        fn test_uptrend() { }

        #[test]
        fn test_downtrend() { }
    }

    mod risk {
        use super::*;

        #[test]
        fn test_position_sizing() { }

        #[test]
        fn test_stop_loss() { }
    }
}
```

---

## Benchmark Plan

### Framework: criterion.rs

**Why Criterion?**
- Statistical analysis (confidence intervals)
- Outlier detection
- Historical comparison
- HTML reports with charts

### Benchmark Suite Organization

```
benches/
├── Cargo.toml
├── criterion_config.toml
├── strategies/
│   ├── momentum.rs
│   ├── mean_reversion.rs
│   └── mirror_trading.rs
├── data_pipeline/
│   ├── ingestion.rs
│   ├── feature_extraction.rs
│   └── aggregation.rs
├── execution/
│   ├── order_placement.rs
│   └── portfolio_update.rs
└── end_to_end/
    └── full_pipeline.rs
```

### Example Benchmark

```rust
// benches/strategies/momentum.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use neural_trader::strategies::MomentumStrategy;
use neural_trader::test_utils::*;

fn benchmark_momentum_strategy(c: &mut Criterion) {
    let mut group = c.benchmark_group("momentum_strategy");

    // Benchmark with different data sizes
    for size in [100, 1000, 10000].iter() {
        group.bench_with_input(
            BenchmarkId::new("process", size),
            size,
            |b, &size| {
                let strategy = MomentumStrategy::new(LookbackPeriod::Days(14), 2.0);
                let market_data = generate_market_data(size);
                let portfolio = Portfolio::mock_with_cash(dec!(10000));

                b.to_async(tokio::runtime::Runtime::new().unwrap())
                    .iter(|| async {
                        strategy.process(
                            black_box(&market_data),
                            black_box(&portfolio)
                        ).await
                    });
            },
        );
    }

    group.finish();
}

fn benchmark_feature_extraction(c: &mut Criterion) {
    c.bench_function("extract_features_1000_ticks", |b| {
        let ticks = generate_market_ticks(1000);

        b.iter(|| {
            extract_technical_indicators(black_box(&ticks))
        });
    });
}

criterion_group!(benches, benchmark_momentum_strategy, benchmark_feature_extraction);
criterion_main!(benches);
```

### Performance Targets

| Component | Metric | Target | Baseline (Python) | Improvement |
|-----------|--------|--------|-------------------|-------------|
| **Market Data Ingestion** | Latency per tick | <100μs | 5ms | **50x** |
| **Feature Extraction** | 1000 ticks | <1ms | 50ms | **50x** |
| **Signal Generation** | Per strategy | <5ms | 100ms | **20x** |
| **Order Placement** | API call | <10ms | 200ms | **20x** |
| **Portfolio Update** | Per trade | <100μs | 5ms | **50x** |
| **Risk Check** | Per signal | <500μs | 10ms | **20x** |
| **End-to-End Pipeline** | p50 latency | <50ms | 500ms | **10x** |
| **End-to-End Pipeline** | p95 latency | <200ms | 2000ms | **10x** |
| **End-to-End Pipeline** | p99 latency | <500ms | 5000ms | **10x** |
| **Throughput** | Events/sec | 100K | 10K | **10x** |
| **Memory Footprint** | Resident | <1GB | 5GB | **5x** |
| **Cold Start** | Time to ready | <500ms | 5s | **10x** |

### Running Benchmarks

```bash
# Run all benchmarks
cargo bench

# Run specific benchmark
cargo bench momentum

# Compare to baseline
cargo bench -- --save-baseline main
git checkout feature-branch
cargo bench -- --baseline main

# Generate HTML report
cargo bench -- --verbose
open target/criterion/report/index.html
```

### Benchmark Configuration

```toml
# benches/criterion_config.toml
[default]
sample_size = 100
measurement_time = 5
warm_up_time = 2
confidence_level = 0.95
significance_level = 0.05
noise_threshold = 0.02
```

---

## CI/CD Pipeline

### Platform Matrix

```yaml
# .github/workflows/ci.yml
name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

env:
  RUST_BACKTRACE: 1
  CARGO_TERM_COLOR: always

jobs:
  test:
    name: Test Suite
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        rust: [stable, beta, nightly]
        node: [18, 20, 22]
    runs-on: ${{ matrix.os }}

    steps:
      - uses: actions/checkout@v4

      - name: Install Rust ${{ matrix.rust }}
        uses: dtolnay/rust-toolchain@master
        with:
          toolchain: ${{ matrix.rust }}
          components: rustfmt, clippy

      - name: Setup Node.js ${{ matrix.node }}
        uses: actions/setup-node@v4
        with:
          node-version: ${{ matrix.node }}

      - name: Cache Cargo
        uses: actions/cache@v4
        with:
          path: |
            ~/.cargo/bin/
            ~/.cargo/registry/index/
            ~/.cargo/registry/cache/
            ~/.cargo/git/db/
            target/
          key: ${{ runner.os }}-cargo-${{ hashFiles('**/Cargo.lock') }}

      - name: Run Unit Tests
        run: cargo test --all-features --lib

      - name: Run Integration Tests
        run: cargo test --all-features --test '*'

      - name: Run Doc Tests
        run: cargo test --all-features --doc

  coverage:
    name: Code Coverage
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install Rust
        uses: dtolnay/rust-toolchain@stable

      - name: Install Tarpaulin
        run: cargo install cargo-tarpaulin

      - name: Generate Coverage
        run: |
          cargo tarpaulin --out Xml --all-features --timeout 600

      - name: Upload to Codecov
        uses: codecov/codecov-action@v4
        with:
          files: ./cobertura.xml
          fail_ci_if_error: true
          token: ${{ secrets.CODECOV_TOKEN }}

      - name: Check Coverage Threshold
        run: |
          COVERAGE=$(grep -oP 'line-rate="\K[^"]+' cobertura.xml | head -1)
          COVERAGE_PCT=$(echo "$COVERAGE * 100" | bc)
          echo "Coverage: $COVERAGE_PCT%"
          if (( $(echo "$COVERAGE_PCT < 90" | bc -l) )); then
            echo "Coverage below 90% threshold!"
            exit 1
          fi

  clippy:
    name: Clippy Lints
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with:
          components: clippy

      - name: Run Clippy
        run: cargo clippy --all-targets --all-features -- -D warnings

  fmt:
    name: Rustfmt
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with:
          components: rustfmt

      - name: Check Formatting
        run: cargo fmt --all -- --check

  security:
    name: Security Audit
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install cargo-audit
        run: cargo install cargo-audit

      - name: Run Security Audit
        run: cargo audit --deny warnings

      - name: Run cargo-deny
        run: |
          cargo install cargo-deny
          cargo deny check

  benchmarks:
    name: Performance Benchmarks
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0  # Fetch all history for comparison

      - uses: dtolnay/rust-toolchain@stable

      - name: Run Benchmarks
        run: cargo bench -- --save-baseline pr-${{ github.event.number }}

      - name: Compare to Main
        if: github.event_name == 'pull_request'
        run: |
          git checkout main
          cargo bench -- --save-baseline main
          git checkout -
          cargo bench -- --baseline main > bench_comparison.txt
          cat bench_comparison.txt

      - name: Check for Regressions
        run: |
          if grep -q "Performance regressed" bench_comparison.txt; then
            echo "Performance regression detected!"
            exit 1
          fi

      - name: Upload Benchmark Results
        uses: actions/upload-artifact@v4
        with:
          name: benchmark-results
          path: target/criterion/

  parity:
    name: Python Parity Tests
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: dtolnay/rust-toolchain@stable

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install Python Dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest

      - name: Run Parity Tests
        run: cargo test --test parity_*

  e2e:
    name: End-to-End Tests
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable

      - name: Run E2E Tests
        run: cargo test --test e2e_* --ignored
        env:
          ALPACA_API_KEY: ${{ secrets.ALPACA_API_KEY_TEST }}
          ALPACA_SECRET_KEY: ${{ secrets.ALPACA_SECRET_KEY_TEST }}
          ALPACA_BASE_URL: https://paper-api.alpaca.markets

  fuzz:
    name: Fuzz Testing
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: dtolnay/rust-toolchain@nightly

      - name: Install cargo-fuzz
        run: cargo install cargo-fuzz

      - name: Run Fuzz Tests (1 hour)
        run: |
          cd fuzz
          for target in fuzz_targets/*; do
            cargo fuzz run $(basename $target .rs) -- -max_total_time=3600
          done

  mcp_compliance:
    name: MCP Protocol Compliance
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable

      - name: Install MCP Validator
        run: npm install -g @modelcontextprotocol/validator

      - name: Build MCP Server
        run: cargo build --bin neural-trader-mcp

      - name: Run MCP Compliance Tests
        run: |
          mcp-validator test target/debug/neural-trader-mcp
```

### Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/doublify/pre-commit-rust
    rev: v1.0
    hooks:
      - id: fmt
        args: ['--all', '--', '--check']
      - id: cargo-check
      - id: clippy
        args: ['--all-targets', '--all-features', '--', '-D', 'warnings']

  - repo: local
    hooks:
      - id: test
        name: cargo test
        entry: cargo test --all-features
        language: system
        pass_filenames: false

      - id: coverage-check
        name: coverage threshold
        entry: ./scripts/check_coverage.sh
        language: system
        pass_filenames: false
```

---

## Quality Gates

### Gate 1: Code Quality

**Enforced By:** CI pipeline

**Requirements:**
- ✅ All tests pass (unit, integration, E2E)
- ✅ Code coverage ≥90%
- ✅ Clippy warnings = 0
- ✅ Rustfmt check passes
- ✅ No security vulnerabilities (cargo-audit)
- ✅ Documentation coverage ≥80%

**Blocking:** Yes (prevents merge)

### Gate 2: Performance

**Enforced By:** Benchmark CI job

**Requirements:**
- ✅ No regressions >5% vs baseline
- ✅ All targets met (see Performance Targets table)
- ✅ Memory usage within limits
- ✅ No performance degradation on hot paths

**Blocking:** Yes (fails CI if regressed)

### Gate 3: Parity

**Enforced By:** Parity test suite

**Requirements:**
- ✅ All Python features have Rust equivalent
- ✅ Results match within 1e-6 tolerance
- ✅ Edge cases match Python behavior
- ✅ Performance is ≥10x faster

**Blocking:** Yes (must maintain parity)

### Gate 4: Security

**Enforced By:** Security audit job

**Requirements:**
- ✅ Zero high/critical vulnerabilities
- ✅ All dependencies audited
- ✅ License compliance (cargo-deny)
- ✅ No secrets in code (gitleaks)

**Blocking:** Yes (critical for production)

### Gate 5: MCP Compliance

**Enforced By:** MCP validator

**Requirements:**
- ✅ All MCP tools validate
- ✅ Schema matches specification
- ✅ Error responses well-formed
- ✅ Timeout handling correct

**Blocking:** Yes (protocol compliance required)

---

## Test Data Generation

### Market Data Fixtures

```rust
// crates/test-utils/src/fixtures.rs
use rust_decimal::Decimal;
use chrono::{DateTime, Utc};

pub struct MarketDataBuilder {
    symbol: String,
    ticks: Vec<MarketTick>,
}

impl MarketDataBuilder {
    pub fn new() -> Self {
        Self {
            symbol: "TEST".to_string(),
            ticks: Vec::new(),
        }
    }

    pub fn symbol(mut self, symbol: &str) -> Self {
        self.symbol = symbol.to_string();
        self
    }

    /// Generate uptrend with specified number of ticks
    pub fn add_uptrend(mut self, num_ticks: usize, start_price: Decimal, end_price: Decimal) -> Self {
        let increment = (end_price - start_price) / Decimal::from(num_ticks);
        let start_time = Utc::now().timestamp();

        for i in 0..num_ticks {
            self.ticks.push(MarketTick {
                symbol: self.symbol.clone(),
                timestamp: start_time + (i as i64 * 60),
                price: start_price + increment * Decimal::from(i),
                volume: Decimal::new(1000, 0),
                bid: start_price + increment * Decimal::from(i) - Decimal::new(1, 2),
                ask: start_price + increment * Decimal::from(i) + Decimal::new(1, 2),
            });
        }

        self
    }

    /// Generate downtrend
    pub fn add_downtrend(self, num_ticks: usize, start_price: Decimal, end_price: Decimal) -> Self {
        self.add_uptrend(num_ticks, end_price, start_price)
    }

    /// Generate sideways movement with noise
    pub fn add_sideways(mut self, num_ticks: usize, base_price: Decimal, volatility: f64) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let start_time = Utc::now().timestamp();

        for i in 0..num_ticks {
            let noise = rng.gen_range(-volatility..volatility);
            let price = base_price + Decimal::from_f64_retain(noise).unwrap();

            self.ticks.push(MarketTick {
                symbol: self.symbol.clone(),
                timestamp: start_time + (i as i64 * 60),
                price,
                volume: Decimal::new(1000, 0),
                bid: price - Decimal::new(1, 2),
                ask: price + Decimal::new(1, 2),
            });
        }

        self
    }

    pub fn build(self) -> MarketData {
        MarketData::from_ticks(self.ticks)
    }
}

/// Generate realistic OHLCV data from CSV
pub fn load_historical_data(path: &str) -> Result<MarketData, Box<dyn std::error::Error>> {
    let mut reader = csv::Reader::from_path(path)?;
    let mut ticks = Vec::new();

    for result in reader.deserialize() {
        let record: HistoricalRecord = result?;
        ticks.push(MarketTick {
            symbol: record.symbol,
            timestamp: record.timestamp,
            price: record.close,
            volume: record.volume,
            bid: record.low,
            ask: record.high,
        });
    }

    Ok(MarketData::from_ticks(ticks))
}
```

### Mock Services

```rust
// crates/test-utils/src/mocks.rs
use async_trait::async_trait;
use std::sync::{Arc, Mutex};

#[derive(Clone)]
pub struct MockAlpacaClient {
    orders: Arc<Mutex<Vec<Order>>>,
    positions: Arc<Mutex<Vec<Position>>>,
}

impl MockAlpacaClient {
    pub fn new() -> Self {
        Self {
            orders: Arc::new(Mutex::new(Vec::new())),
            positions: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn orders_placed(&self) -> usize {
        self.orders.lock().unwrap().len()
    }

    pub fn with_position(self, symbol: &str, qty: i32) -> Self {
        self.positions.lock().unwrap().push(Position {
            symbol: symbol.to_string(),
            qty,
            ..Default::default()
        });
        self
    }
}

#[async_trait]
impl BrokerClient for MockAlpacaClient {
    async fn place_order(&self, order: OrderRequest) -> Result<Order, BrokerError> {
        let order = Order {
            id: Uuid::new_v4().to_string(),
            symbol: order.symbol,
            qty: order.qty,
            side: order.side,
            status: OrderStatus::Filled,
            ..Default::default()
        };

        self.orders.lock().unwrap().push(order.clone());
        Ok(order)
    }

    async fn get_positions(&self) -> Result<Vec<Position>, BrokerError> {
        Ok(self.positions.lock().unwrap().clone())
    }
}
```

---

## Troubleshooting

### Common Test Failures

#### 1. Flaky Async Tests

**Symptom:** Tests pass locally but fail in CI randomly

**Solution:**

```rust
// ❌ BAD: Race condition
#[tokio::test]
async fn test_async_operation() {
    let result = start_async_task();
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert_eq!(result.get(), expected);
}

// ✅ GOOD: Wait for completion
#[tokio::test]
async fn test_async_operation() {
    let result = start_async_task().await;
    assert_eq!(result, expected);
}
```

#### 2. Coverage Gaps

**Symptom:** Coverage below 90%

**Diagnosis:**

```bash
# Generate detailed coverage report
cargo tarpaulin --out Html --output-dir target/coverage
open target/coverage/index.html

# Find untested lines
cargo tarpaulin --ignored --verbose | grep "Uncovered"
```

**Solution:** Add tests for missing lines

#### 3. Benchmark Regressions

**Symptom:** CI fails with "Performance regressed"

**Diagnosis:**

```bash
# Compare locally
cargo bench -- --save-baseline main
git checkout feature-branch
cargo bench -- --baseline main

# Check specific benchmark
cargo bench momentum -- --verbose
```

**Solution:** Optimize hot paths or update baseline

#### 4. Parity Failures

**Symptom:** Rust results differ from Python

**Diagnosis:**

```bash
# Run parity test with detailed output
RUST_LOG=debug cargo test parity_momentum -- --nocapture

# Compare numerical precision
cargo test parity_momentum -- --show-output
```

**Solution:** Check floating-point precision, random seeds

---

## Acceptance Criteria

### Test Suite Completeness

- [ ] Unit tests for all public APIs (≥95% coverage)
- [ ] Property tests for financial calculations
- [ ] Integration tests for all data flows
- [ ] E2E tests for critical user scenarios
- [ ] Fuzz tests for all parsers
- [ ] Parity tests for all migrated features

### Benchmark Coverage

- [ ] All performance targets defined
- [ ] Benchmarks for all hot paths
- [ ] Regression detection enabled
- [ ] Historical comparison available

### CI/CD Pipeline

- [ ] Cross-platform builds (Linux, macOS, Windows)
- [ ] Node.js version matrix (18, 20, 22)
- [ ] Coverage reporting integrated
- [ ] Security audits automated
- [ ] MCP compliance validated

### Quality Gates

- [ ] All gates enforced in CI
- [ ] Merge blocked on failures
- [ ] Automated rollback on regression
- [ ] Performance SLA validated

---

## Cross-References

- **Architecture:** [03_Architecture.md](./03_Architecture.md) - Module design
- **Parity Requirements:** [02_Parity_Requirements.md](./02_Parity_Requirements.md) - Feature matrix
- **Risk Management:** [14_Risk_and_Fallbacks.md](./14_Risk_and_Fallbacks.md) - Test failures
- **Roadmap:** [15_Roadmap_Phases_and_Milestones.md](./15_Roadmap_Phases_and_Milestones.md) - Timeline

---

**Document Version:** 1.0.0
**Last Updated:** 2025-11-12
**Owner:** QA Lead
**Status:** Complete
**Next Review:** 2025-11-19
