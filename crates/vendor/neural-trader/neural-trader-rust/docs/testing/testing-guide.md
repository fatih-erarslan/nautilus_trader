# Testing Guide

Comprehensive guide to testing neural-trader Rust implementation.

## 🎯 Testing Philosophy

We follow Test-Driven Development (TDD) principles:

1. **Write tests first** - Define expected behavior
2. **Make tests fail** - Verify tests catch bugs
3. **Implement code** - Make tests pass
4. **Refactor** - Improve while tests stay green

## 📊 Coverage Requirements

```
Target: 90%+ overall code coverage

Per-Component Requirements:
├── Core (95%)
├── Strategies (95%)
├── Execution (95%)
├── Risk (95%)
├── Portfolio (90%)
├── Market Data (90%)
├── Neural (85%)
├── Backtesting (90%)
└── Other (85%)
```

## 🧪 Test Categories

### 1. Unit Tests

Test individual functions and methods in isolation.

**Location**: Within each crate as `#[cfg(test)]` modules

**Example**:
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_position_size() {
        let account_value = 100_000.0;
        let risk_percent = 0.02;
        let stop_loss = 0.05;

        let size = calculate_position_size(account_value, risk_percent, stop_loss);

        // Should risk $2,000 with 5% stop = 40 shares
        assert_eq!(size, 40.0);
    }

    #[test]
    fn test_position_size_handles_zero_stop_loss() {
        let result = calculate_position_size(100_000.0, 0.02, 0.0);
        assert!(result.is_err());
    }
}
```

### 2. Integration Tests

Test interactions between multiple crates.

**Location**: `tests/integration/`

**Example**:
```rust
// tests/integration/test_market_data.rs
use nt_market_data::AlpacaProvider;
use nt_strategies::PairsStrategy;

#[tokio::test]
async fn test_strategy_with_live_data() {
    let provider = AlpacaProvider::new().await.unwrap();
    let data = provider.fetch_bars("AAPL", start, end).await.unwrap();

    let strategy = PairsStrategy::new(vec!["AAPL", "MSFT"], 20, 2.0);
    let signals = strategy.generate_signals(&data).await.unwrap();

    assert!(!signals.is_empty());
    assert!(signals.iter().all(|s| s.is_valid()));
}
```

### 3. End-to-End Tests

Test complete user workflows.

**Location**: `tests/e2e/`

**Example**:
```rust
// tests/e2e/test_full_trading_loop.rs
#[tokio::test]
async fn test_complete_trading_cycle() {
    // Setup
    let system = TradingSystem::new(test_config()).await.unwrap();

    // Fetch data
    let data = system.fetch_market_data("AAPL").await.unwrap();
    assert!(!data.is_empty());

    // Generate signals
    let signals = system.strategy.analyze(&data).await.unwrap();
    assert!(signals.iter().any(|s| matches!(s, Signal::Buy)));

    // Execute orders
    for signal in signals {
        let order = system.create_order(signal).unwrap();
        let fill = system.execute(order).await.unwrap();
        assert_eq!(fill.status, OrderStatus::Filled);
    }

    // Verify P&L
    let pnl = system.portfolio.calculate_pnl().unwrap();
    assert!(pnl.total() != 0.0);
}
```

### 4. Property-Based Tests

Test properties that should hold for all inputs.

**Location**: `tests/property/`

**Example**:
```rust
// tests/property/test_position_sizing.rs
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_position_size_never_exceeds_risk(
        account_value in 1_000.0..1_000_000.0,
        risk_percent in 0.01..0.1,
        stop_loss in 0.01..0.5
    ) {
        let size = calculate_position_size(account_value, risk_percent, stop_loss);
        let max_loss = size * stop_loss;

        prop_assert!(max_loss <= account_value * risk_percent * 1.01); // 1% tolerance
    }

    #[test]
    fn test_var_calculation_is_positive(
        portfolio_value in 10_000.0..1_000_000.0,
        confidence in 0.9..0.99
    ) {
        let portfolio = create_random_portfolio(portfolio_value);
        let var = calculate_var(&portfolio, confidence, VarMethod::Historical).unwrap();

        prop_assert!(var > 0.0);
        prop_assert!(var < portfolio_value);
    }
}
```

### 5. Benchmark Tests

Measure and track performance.

**Location**: `benches/`

**Example**:
```rust
// benches/trading_benchmarks.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn benchmark_backtest(c: &mut Criterion) {
    let data = load_test_data();
    let strategy = PairsStrategy::new(vec!["AAPL", "MSFT"], 20, 2.0);

    c.bench_function("1_year_backtest", |b| {
        b.iter(|| {
            let results = run_backtest(black_box(&strategy), black_box(&data));
            black_box(results);
        });
    });
}

criterion_group!(benches, benchmark_backtest);
criterion_main!(benches);
```

## 🚀 Running Tests

### All Tests

```bash
# Run all tests
cargo test --workspace --all-features

# Run with output
cargo test -- --nocapture

# Run in parallel
cargo test --workspace -- --test-threads=8
```

### Specific Test Categories

```bash
# Unit tests only
cargo test --lib

# Integration tests only
cargo test --test '*'

# Specific test
cargo test test_position_size_calculation

# Specific module
cargo test nt_risk::tests
```

### With Coverage

```bash
# Install tarpaulin
cargo install cargo-tarpaulin

# Generate HTML coverage report
cargo tarpaulin \
  --workspace \
  --all-features \
  --timeout 300 \
  --out Html \
  --output-dir coverage/

# View coverage
open coverage/index.html
```

### Continuous Testing

```bash
# Install cargo-watch
cargo install cargo-watch

# Run tests on file changes
cargo watch -x test

# Run specific test on changes
cargo watch -x "test test_name"
```

## 🎯 Writing Quality Tests

### Test Naming Convention

```rust
// Pattern: test_[what]_[condition]_[expected_result]

#[test]
fn test_calculate_var_with_empty_portfolio_returns_error() { }

#[test]
fn test_pairs_strategy_exits_when_spread_converges() { }

#[test]
fn test_order_execution_succeeds_during_market_hours() { }
```

### Arrange-Act-Assert Pattern

```rust
#[test]
fn test_portfolio_adds_position_correctly() {
    // Arrange
    let mut portfolio = Portfolio::new(100_000.0);
    let expected_positions = 1;

    // Act
    portfolio.add_position("AAPL", 50, 180.50).unwrap();

    // Assert
    assert_eq!(portfolio.positions().len(), expected_positions);
    assert_eq!(portfolio.get_position("AAPL").unwrap().quantity(), 50);
}
```

### Test Data Fixtures

```rust
// tests/utils/fixtures.rs
pub fn create_test_portfolio() -> Portfolio {
    let mut portfolio = Portfolio::new(100_000.0);
    portfolio.add_position("AAPL", 50, 180.50);
    portfolio.add_position("MSFT", 30, 380.25);
    portfolio
}

pub fn create_test_market_data(days: usize) -> DataFrame {
    generate_mock_bars("AAPL", days)
}

pub fn test_config() -> Config {
    Config {
        alpaca_base_url: "https://paper-api.alpaca.markets".to_string(),
        alpaca_api_key: "test_key".to_string(),
        alpaca_secret_key: "test_secret".to_string(),
    }
}
```

### Mock Objects

```rust
// tests/mocks/mock_broker.rs
use mockall::*;

#[automock]
pub trait Broker {
    async fn submit_order(&mut self, order: Order) -> Result<Fill>;
    async fn cancel_order(&mut self, order_id: Uuid) -> Result<()>;
}

// Usage in tests
#[tokio::test]
async fn test_order_submission() {
    let mut mock_broker = MockBroker::new();

    mock_broker
        .expect_submit_order()
        .times(1)
        .returning(|order| {
            Ok(Fill {
                id: Uuid::new_v4(),
                order_id: order.id,
                price: 180.50,
                quantity: order.quantity,
                timestamp: Utc::now(),
            })
        });

    let result = mock_broker.submit_order(test_order()).await;
    assert!(result.is_ok());
}
```

### Custom Assertions

```rust
// tests/utils/assertions.rs
pub fn assert_approx_equal(actual: f64, expected: f64, tolerance: f64) {
    let diff = (actual - expected).abs();
    assert!(
        diff <= tolerance,
        "Expected {} to be within {} of {}, but difference was {}",
        actual,
        tolerance,
        expected,
        diff
    );
}

pub fn assert_portfolio_balanced(portfolio: &Portfolio) {
    let total_value = portfolio.total_value();
    for position in portfolio.positions() {
        let weight = position.value() / total_value;
        assert!(
            weight <= 0.3,
            "Position {} exceeds 30% weight: {:.2}%",
            position.symbol(),
            weight * 100.0
        );
    }
}
```

## 🐛 Debugging Tests

### Enable Logging

```bash
RUST_LOG=debug cargo test test_name -- --nocapture
```

### Run Single Test

```bash
cargo test test_name --exact -- --nocapture
```

### Debug with LLDB

```bash
# Build test binary
cargo test --no-run

# Find test binary
TEST_BINARY=$(find target/debug/deps -name "nt_*" -type f | head -1)

# Debug with lldb
rust-lldb $TEST_BINARY

# Set breakpoint and run
(lldb) b test_name
(lldb) run test_name
```

### Print Test Output

```rust
#[test]
fn test_with_output() {
    let result = calculate_something();
    println!("Result: {:?}", result); // Use --nocapture to see
    assert!(result > 0.0);
}
```

## 📊 CI Integration

Tests run automatically in GitHub Actions:

```yaml
# .github/workflows/rust-ci.yml
- name: Run tests
  run: cargo test --workspace --all-features

- name: Generate coverage
  run: cargo tarpaulin --out xml --output-dir coverage/

- name: Upload to Codecov
  uses: codecov/codecov-action@v4
  with:
    files: ./coverage/cobertura.xml
```

## ✅ Pre-Commit Checklist

Before committing code:

- [ ] All tests pass: `cargo test --workspace`
- [ ] Coverage ≥90%: `cargo tarpaulin`
- [ ] No clippy warnings: `cargo clippy -- -D warnings`
- [ ] Code formatted: `cargo fmt --check`
- [ ] Documentation builds: `cargo doc --no-deps`
- [ ] Benchmarks pass: `cargo bench`

## 🎓 Testing Best Practices

### DO:
✅ Test one behavior per test
✅ Use descriptive test names
✅ Keep tests independent
✅ Test edge cases and errors
✅ Use property-based testing for invariants
✅ Mock external dependencies
✅ Maintain high coverage (≥90%)

### DON'T:
❌ Test implementation details
❌ Have interdependent tests
❌ Use sleep() for timing
❌ Ignore flaky tests
❌ Test private functions directly
❌ Write slow tests (>1s for unit tests)
❌ Commit failing tests

## 📈 Measuring Success

### Coverage Metrics

```bash
# Overall coverage
cargo tarpaulin --workspace

# Per-crate coverage
cargo tarpaulin --packages nt-strategies

# Uncovered lines
cargo tarpaulin --workspace --out Stdout | grep "^nt_"
```

### Test Performance

```bash
# Time each test
cargo test -- --report-time

# Identify slow tests
cargo test -- --report-time | grep "test result" | sort -n -k 4
```

### Benchmark Comparison

```bash
# Run benchmarks
cargo bench

# Compare with baseline
cargo bench --save-baseline main
git checkout feature-branch
cargo bench --baseline main
```

## 🔍 Advanced Testing

### Fuzzing with cargo-fuzz

```bash
# Install cargo-fuzz
cargo install cargo-fuzz

# Create fuzz target
cargo fuzz init

# Run fuzzer
cargo fuzz run fuzz_target_1
```

### Mutation Testing with cargo-mutants

```bash
# Install cargo-mutants
cargo install cargo-mutants

# Run mutation testing
cargo mutants --workspace
```

### Snapshot Testing

```rust
use insta::assert_debug_snapshot;

#[test]
fn test_backtest_results() {
    let results = run_backtest();
    assert_debug_snapshot!(results);
}
```

## 📚 Resources

- [Rust Testing Book](https://rust-lang.github.io/rust-clippy/master/index.html)
- [Proptest Documentation](https://altsysrq.github.io/proptest-book/)
- [Criterion.rs Guide](https://bheisler.github.io/criterion.rs/book/)
- [Mockall Documentation](https://docs.rs/mockall/latest/mockall/)

---

Test everything. Trust nothing. Ship quality! ✅
