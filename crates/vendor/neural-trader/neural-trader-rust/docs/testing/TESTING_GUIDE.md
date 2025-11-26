<!-- markdownlint-disable MD013 MD033 MD041 -->
# Neural Trader Rust Testing Guide

Comprehensive guide to the testing infrastructure for the neural-trader Rust implementation.

## 📋 Table of Contents

- [Overview](#overview)
- [Test Categories](#test-categories)
- [Running Tests](#running-tests)
- [Writing Tests](#writing-tests)
- [Coverage](#coverage)
- [Performance Testing](#performance-testing)
- [CI/CD Integration](#cicd-integration)
- [Troubleshooting](#troubleshooting)

## Overview

The neural-trader Rust port maintains **90%+ test coverage** with a comprehensive test suite covering:

- **Unit Tests**: Per-crate module testing
- **Integration Tests**: Cross-crate interaction testing
- **Property Tests**: Invariant verification with proptest
- **E2E Tests**: Complete workflow validation
- **Benchmarks**: Performance validation vs Python baseline
- **Load Tests**: Stress testing under high throughput
- **Fault Tolerance**: Error injection and recovery
- **Real API Tests**: Live integration validation (paper trading)

## Test Categories

### 1. Unit Tests (90%+ Coverage Required)

Located in `tests/unit/` and within each crate as `#[cfg(test)]` modules.

```bash
# Run all unit tests
cargo test --lib

# Run specific crate unit tests
cargo test -p nt-core --lib
cargo test -p nt-risk --lib
cargo test -p nt-strategies --lib
```

**Example Unit Test:**

```rust
#[test]
fn test_order_validation() {
    let order = create_test_order("AAPL", 100);
    assert!(validate_order(&order).is_ok());
}
```

### 2. Integration Tests

Located in `tests/integration/` - test interactions between crates.

```bash
# Run all integration tests
cargo test --test '*'

# Run specific integration test
cargo test --test test_trading_pipeline
cargo test --test test_multi_broker
```

**Example Integration Test:**

```rust
#[tokio::test]
async fn test_complete_trading_pipeline() {
    let data = fetch_market_data().await.unwrap();
    let signals = strategy.analyze(&data).await.unwrap();
    let filled = execute_orders(signals).await.unwrap();
    assert!(filled.len() > 0);
}
```

### 3. Property-Based Tests

Located in `tests/property/` - use proptest for exhaustive validation.

```bash
# Run property tests
cargo test --test test_invariants

# Run with more cases
cargo test --test test_invariants -- --test-threads=1
```

**Example Property Test:**

```rust
proptest! {
    #[test]
    fn test_position_size_never_exceeds_limit(
        account_value in 1000.0..1_000_000.0,
        risk_percent in 0.01..0.1
    ) {
        let size = calculate_position_size(account_value, risk_percent);
        prop_assert!(size <= account_value * risk_percent);
    }
}
```

### 4. End-to-End Tests

Located in `tests/e2e/` - complete user workflows.

```bash
# Run E2E tests
cargo test --test test_full_trading_loop
cargo test --test test_backtesting
cargo test --test test_cli
```

### 5. Performance Benchmarks

Located in `benches/` - validate performance targets.

```bash
# Run all benchmarks
cargo bench --workspace

# Run specific benchmark
cargo bench --bench trading_benchmarks
cargo bench --bench neural_benchmarks

# Compare with baseline
cargo bench -- --save-baseline main
git checkout feature-branch
cargo bench -- --baseline main
```

**Performance Targets:**

| Operation | Target | Status |
|-----------|--------|--------|
| Order execution | <1ms | ✅ |
| Strategy signal generation | <10ms | ✅ |
| Risk VaR calculation | <100ms | ✅ |
| Neural inference | <10ms | ✅ |
| Backtesting (1 year) | <5s | ✅ |
| vs Python baseline | 8-10x faster | ✅ |

### 6. Load Tests

Located in `tests/load/` - stress testing under high load.

```bash
# Run load tests
cargo test --test stress_tests --release -- --test-threads=8

# Run specific load test
cargo test test_high_frequency_orders --release
cargo test test_concurrent_strategy_execution --release
```

**Load Test Targets:**

- 1000+ orders/second
- 100 concurrent strategies
- 10,000 market data ticks/second
- Multi-hour sustained operation

### 7. Fault Tolerance Tests

Located in `tests/fault_tolerance/` - error injection and recovery.

```bash
# Run fault tolerance tests
cargo test --test error_injection

# Run with logging
RUST_LOG=debug cargo test --test error_injection -- --nocapture
```

**Test Scenarios:**

- Network timeouts and retries
- Invalid data handling
- Broker rejections
- Insufficient funds
- Rate limit backoff
- Circuit breaker patterns
- Graceful degradation
- Transaction rollback

### 8. Real API Tests

Located in `tests/real_api/` - live integration (requires credentials).

```bash
# Run with environment variables
export ALPACA_API_KEY="your_key"
export ALPACA_SECRET_KEY="your_secret"
export POLYGON_API_KEY="your_key"

# Run ignored tests
cargo test --test live_tests -- --ignored

# Run specific API test
cargo test test_alpaca_connection -- --ignored
```

⚠️ **Important:** Real API tests use paper trading accounts and are ignored by default.

## Running Tests

### Quick Commands

```bash
# Run all tests
cargo test --workspace --all-features

# Run tests with output
cargo test -- --nocapture

# Run single test
cargo test test_name --exact

# Run tests in sequence (for debugging)
cargo test -- --test-threads=1

# Run with logging
RUST_LOG=debug cargo test -- --nocapture

# Run specific category
cargo test --lib                    # Unit tests only
cargo test --test '*'               # Integration tests only
cargo test --test test_invariants   # Property tests only
```

### Test Selection

```bash
# By pattern
cargo test risk                     # All tests with "risk" in name
cargo test ::unit::                 # All unit tests

# By crate
cargo test -p nt-core
cargo test -p nt-strategies

# By feature
cargo test --all-features
cargo test --no-default-features
cargo test --features "gpu-acceleration"
```

## Writing Tests

### Test Structure

Follow the **Arrange-Act-Assert** pattern:

```rust
#[test]
fn test_portfolio_valuation() {
    // Arrange - Set up test data
    let portfolio = create_test_portfolio();
    let expected_value = 50000.0;

    // Act - Execute the code under test
    let actual_value = portfolio.calculate_value();

    // Assert - Verify results
    assert_eq!(actual_value, expected_value);
}
```

### Test Naming

Use descriptive names that explain **what** is being tested and **under what conditions**:

```rust
✅ Good:
- test_order_execution_succeeds_with_valid_data()
- test_portfolio_rejects_oversized_position()
- test_strategy_generates_buy_signal_on_oversold_rsi()

❌ Bad:
- test1()
- test_function()
- test_order()
```

### Test Fixtures

Use fixtures from `tests/fixtures/mod.rs` for reusable test data:

```rust
use tests::fixtures::{create_test_portfolio, generate_mock_bars, create_risk_limits};

#[test]
fn test_with_fixtures() {
    let portfolio = create_test_portfolio();
    let bars = generate_mock_bars("AAPL", 252);
    let limits = create_risk_limits();

    // Test logic here
}
```

### Async Tests

Use `#[tokio::test]` for async tests:

```rust
#[tokio::test]
async fn test_async_operation() {
    let result = async_function().await.unwrap();
    assert!(result.is_ok());
}
```

### Property Tests

Define properties that should always hold:

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_portfolio_value_never_negative(
        cash in 0.0..1_000_000.0,
        positions in prop::collection::vec(0.0..100_000.0, 0..10)
    ) {
        let total = cash + positions.iter().sum::<f64>();
        prop_assert!(total >= 0.0);
    }
}
```

## Coverage

### Generate Coverage Reports

```bash
# Install tarpaulin
cargo install cargo-tarpaulin

# Generate HTML report
cargo tarpaulin --workspace --all-features --out Html --output-dir coverage/

# Generate multiple formats
cargo tarpaulin --workspace --all-features --out Html --out Xml --out Lcov

# View HTML report
open coverage/index.html  # macOS
xdg-open coverage/index.html  # Linux
```

### Coverage Requirements

| Component | Line Coverage | Branch Coverage |
|-----------|---------------|-----------------|
| Core | ≥95% | ≥90% |
| Market Data | ≥90% | ≥85% |
| Strategies | ≥95% | ≥90% |
| Execution | ≥95% | ≥90% |
| Portfolio | ≥90% | ≥85% |
| Risk | ≥95% | ≥90% |
| Neural | ≥85% | ≥80% |
| **Overall** | **≥90%** | **≥85%** |

### CI Coverage Tracking

Coverage is automatically tracked on every PR via Codecov:

- View at: `https://codecov.io/gh/ruvnet/neural-trader`
- Coverage must not decrease on new PRs
- Critical paths require 95%+ coverage

## Performance Testing

### Running Benchmarks

```bash
# Run all benchmarks
cargo bench --workspace

# Run with custom settings
cargo bench --workspace -- --warm-up-time 5 --measurement-time 30

# Run specific benchmark
cargo bench --bench trading_benchmarks

# Compare against baseline
cargo bench -- --save-baseline main
```

### Benchmark Regression Detection

```bash
# Save current performance
cargo bench -- --save-baseline current

# Make changes...

# Compare
cargo bench -- --baseline current

# Fails if >10% regression
cargo bench -- --baseline current --test
```

### Performance Validation

Compare against Python baseline:

```bash
# Run Python baseline (in Python env)
cd /workspaces/neural-trader
python -m pytest tests/performance/ --benchmark-only

# Run Rust benchmarks
cd neural-trader-rust
cargo bench --workspace

# Compare results
python scripts/compare_performance.py
```

Expected speedups:

- Backtesting: **8-10x** faster
- Order execution: **15x** faster
- Risk calculations: **10x** faster
- Memory usage: **50%** less

## CI/CD Integration

### GitHub Actions

Tests run automatically on:

- Every push to `main` or `develop`
- All pull requests
- Scheduled nightly builds

See `.github/workflows/rust-ci.yml` for configuration.

### CI Test Matrix

Tests run on:

- **Rust versions**: stable, beta, nightly
- **OS**: Ubuntu, macOS, Windows
- **Build types**: debug, release

### Pre-commit Hooks

Install pre-commit hooks:

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
cd neural-trader-rust
pre-commit install

# Run manually
pre-commit run --all-files
```

Hooks run:

- `cargo fmt` - Code formatting
- `cargo clippy` - Linting
- `cargo test` - Fast tests
- `cargo check` - Compilation check

## Troubleshooting

### Common Issues

#### Tests Fail with "file not found"

```bash
# Ensure you're in the right directory
cd neural-trader-rust

# Check Cargo.toml exists
ls Cargo.toml
```

#### Timeout Errors

```bash
# Increase timeout
cargo test -- --test-threads=1 --timeout 300

# For tokio tests
#[tokio::test(flavor = "multi_thread")]
async fn test_with_timeout() {
    tokio::time::timeout(
        Duration::from_secs(60),
        async_operation()
    ).await.unwrap();
}
```

#### Out of Memory

```bash
# Reduce parallelism
cargo test -- --test-threads=1

# Run tests sequentially
cargo test --release -- --test-threads=1
```

#### Flaky Tests

```bash
# Run test multiple times
for i in {1..10}; do cargo test test_name --exact || break; done

# Increase timeout and retries
#[tokio::test]
async fn test_with_retry() {
    for _ in 0..3 {
        if test_logic().await.is_ok() {
            return;
        }
    }
    panic!("Test failed after 3 retries");
}
```

### Debug Mode

```bash
# Run with debug output
RUST_LOG=debug cargo test test_name -- --nocapture

# Run with backtrace
RUST_BACKTRACE=1 cargo test

# Run with full backtrace
RUST_BACKTRACE=full cargo test
```

### Test-Specific Logging

```rust
#[test]
fn test_with_logging() {
    env_logger::init();
    log::info!("Test starting");
    // Test logic
    log::info!("Test complete");
}
```

## Best Practices

### ✅ Do

- Write tests before implementation (TDD)
- One assertion per test when possible
- Use descriptive test names
- Test edge cases and error conditions
- Keep tests fast and independent
- Use fixtures for reusable test data
- Document complex test setups
- Run tests before committing

### ❌ Don't

- Write interdependent tests
- Use sleep() for synchronization
- Ignore flaky tests
- Test implementation details
- Leave TODO tests indefinitely
- Commit failing tests
- Skip error cases
- Use production credentials in tests

## Resources

- [Rust Testing Documentation](https://doc.rust-lang.org/book/ch11-00-testing.html)
- [Proptest Documentation](https://altsysrq.github.io/proptest-book/intro.html)
- [Criterion.rs Benchmarking](https://bheisler.github.io/criterion.rs/book/)
- [Tokio Testing Guide](https://tokio.rs/tokio/topics/testing)
- [Project README](/workspaces/neural-trader/neural-trader-rust/README.md)
- [Architecture Documentation](/workspaces/neural-trader/neural-trader-rust/docs/ARCHITECTURE.md)

---

**Questions?** Open an issue or check our [Contributing Guide](CONTRIBUTING.md).

**Quality is not an act, it is a habit.** ✅
