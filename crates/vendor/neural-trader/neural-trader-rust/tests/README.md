# Neural Trader Test Suite

Comprehensive testing infrastructure for neural-trader Rust implementation.

## 📁 Test Structure

```
tests/
├── unit/                   # Unit tests (per-crate)
│   ├── core.rs
│   ├── market_data.rs
│   ├── strategies.rs
│   ├── execution.rs
│   ├── portfolio.rs
│   ├── risk.rs
│   └── ...
├── integration/            # Integration tests (cross-crate)
│   ├── test_market_data.rs
│   ├── test_strategies.rs
│   ├── test_execution.rs
│   ├── test_portfolio.rs
│   └── test_risk.rs
├── e2e/                    # End-to-end tests (full workflows)
│   ├── test_full_trading_loop.rs
│   ├── test_backtesting.rs
│   └── test_cli.rs
├── property/               # Property-based tests (proptest)
│   ├── test_position_sizing.rs
│   ├── test_risk_limits.rs
│   └── test_pnl.rs
├── mocks/                  # Mock implementations
│   ├── mock_broker.rs
│   ├── mock_market_data.rs
│   └── mod.rs
├── utils/                  # Test utilities
│   ├── fixtures.rs
│   ├── assertions.rs
│   └── mod.rs
└── README.md
```

## 🧪 Test Categories

### Unit Tests

Located in `tests/unit/` and within each crate's `src/` as `#[cfg(test)]` modules.

**Coverage Requirements**: 90%+ per crate

```rust
// Example: tests/unit/risk.rs
use nt_risk::calculate_var;

#[test]
fn test_var_calculation() {
    let portfolio = create_test_portfolio();
    let var = calculate_var(&portfolio, 0.95, VarMethod::Historical).unwrap();
    assert!(var > 0.0);
    assert!(var < portfolio.total_value());
}
```

### Integration Tests

Located in `tests/integration/` - test interactions between crates.

```rust
// Example: tests/integration/test_market_data.rs
use nt_market_data::AlpacaProvider;
use nt_strategies::PairsStrategy;

#[tokio::test]
async fn test_strategy_with_real_data() {
    let provider = AlpacaProvider::new().await.unwrap();
    let data = provider.fetch_bars("AAPL", "2024-01-01", "2024-01-31").await.unwrap();

    let strategy = PairsStrategy::new(vec!["AAPL", "MSFT"], 20, 2.0);
    let signals = strategy.generate_signals(&data).await.unwrap();

    assert!(!signals.is_empty());
}
```

### End-to-End Tests

Located in `tests/e2e/` - test complete user workflows.

```rust
// Example: tests/e2e/test_full_trading_loop.rs
#[tokio::test]
async fn test_complete_trading_cycle() {
    // 1. Initialize system
    let config = load_test_config();
    let system = TradingSystem::new(config).await.unwrap();

    // 2. Fetch market data
    let data = system.fetch_market_data("AAPL").await.unwrap();

    // 3. Generate signals
    let signals = system.strategy.analyze(&data).await.unwrap();

    // 4. Execute orders
    for signal in signals {
        let order = system.create_order(signal).unwrap();
        let filled = system.execute(order).await.unwrap();
        assert!(filled.status == OrderStatus::Filled);
    }

    // 5. Calculate P&L
    let pnl = system.portfolio.calculate_pnl().unwrap();
    assert!(pnl.realized + pnl.unrealized > 0.0);
}
```

### Property-Based Tests

Located in `tests/property/` - use proptest for exhaustive testing.

```rust
// Example: tests/property/test_position_sizing.rs
use proptest::prelude::*;
use nt_risk::calculate_position_size;

proptest! {
    #[test]
    fn test_position_size_never_exceeds_limit(
        account_value in 1000.0..1_000_000.0,
        risk_percent in 0.01..0.1,
        stop_loss in 0.01..0.5
    ) {
        let size = calculate_position_size(account_value, risk_percent, stop_loss);
        let max_loss = size * stop_loss;

        // Position size should never risk more than risk_percent
        prop_assert!(max_loss <= account_value * risk_percent);
    }
}
```

## 🚀 Running Tests

### All Tests

```bash
cargo test --workspace --all-features
```

### Specific Category

```bash
# Unit tests only
cargo test --lib

# Integration tests only
cargo test --test '*'

# Specific test file
cargo test --test test_market_data
```

### With Coverage

```bash
# Install tarpaulin
cargo install cargo-tarpaulin

# Generate coverage report
cargo tarpaulin --workspace --all-features --out Html --output-dir coverage/
```

### Performance Benchmarks

```bash
# Run all benchmarks
cargo bench --workspace

# Specific benchmark
cargo bench --bench market_data_benchmarks
```

## 🎯 Test Quality Metrics

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
| Overall | ≥90% | ≥85% |

### Performance Benchmarks

| Operation | Target | Current | Status |
|-----------|--------|---------|--------|
| Market data fetch | <500ms | 180ms | ✅ |
| Strategy signal | <50ms | 12ms | ✅ |
| Order execution | <100ms | 35ms | ✅ |
| Risk calculation | <200ms | 85ms | ✅ |
| Backtest (1 year) | <10s | 5.1s | ✅ |

## 🛠️ Test Utilities

### Fixtures

Located in `tests/utils/fixtures.rs` - reusable test data.

```rust
pub fn create_test_portfolio() -> Portfolio {
    let mut portfolio = Portfolio::new(100_000.0);
    portfolio.add_position("AAPL", 50, 180.50);
    portfolio.add_position("MSFT", 30, 380.25);
    portfolio
}

pub fn create_test_market_data(symbol: &str) -> DataFrame {
    // Returns realistic OHLCV data for testing
    generate_mock_bars(symbol, 252)
}
```

### Mocks

Located in `tests/mocks/` - mock implementations for testing.

```rust
// tests/mocks/mock_broker.rs
pub struct MockBroker {
    orders: Vec<Order>,
    fills: Vec<Fill>,
}

impl Broker for MockBroker {
    async fn submit_order(&mut self, order: Order) -> Result<Fill> {
        // Mock implementation - instant fill at market price
        let fill = Fill {
            id: Uuid::new_v4(),
            order_id: order.id,
            price: order.limit_price.unwrap_or(100.0),
            quantity: order.quantity,
            timestamp: Utc::now(),
        };
        self.fills.push(fill.clone());
        Ok(fill)
    }
}
```

### Assertions

Located in `tests/utils/assertions.rs` - custom assertions.

```rust
pub fn assert_portfolio_balanced(portfolio: &Portfolio, tolerance: f64) {
    let total_value = portfolio.total_value();
    for position in portfolio.positions() {
        let weight = position.value() / total_value;
        assert!(
            weight <= 1.0 / portfolio.positions().len() as f64 + tolerance,
            "Portfolio not balanced: {} has weight {}",
            position.symbol(),
            weight
        );
    }
}

pub fn assert_within_tolerance(actual: f64, expected: f64, tolerance: f64) {
    let diff = (actual - expected).abs();
    assert!(
        diff <= tolerance,
        "Value {} not within {} of expected {}",
        actual,
        tolerance,
        expected
    );
}
```

## 🔧 Configuration

### Test Environment Variables

Create `.env.test`:

```env
# Use paper trading for tests
ALPACA_BASE_URL=https://paper-api.alpaca.markets
ALPACA_API_KEY=test_key
ALPACA_SECRET_KEY=test_secret

# Reduce logging noise
RUST_LOG=error

# Faster test execution
TOKIO_WORKER_THREADS=1
```

### CI Configuration

Tests run automatically on:
- Every push to `main` or `develop`
- All pull requests
- Scheduled nightly builds

See `.github/workflows/rust-ci.yml` for details.

## 📊 Test Reporting

### Coverage Reports

Generated automatically by CI and uploaded to Codecov.

View at: https://codecov.io/gh/ruvnet/neural-trader

### Benchmark Tracking

Benchmark results tracked over time to detect performance regressions.

View at: https://ruvnet.github.io/neural-trader/dev/bench/

## 🐛 Debugging Tests

### Enable Verbose Logging

```bash
RUST_LOG=debug cargo test test_name -- --nocapture
```

### Run Single Test

```bash
cargo test test_name --exact
```

### Run Tests in Sequence

```bash
cargo test -- --test-threads=1
```

### Debug with rust-lldb

```bash
rust-lldb target/debug/deps/test_binary-<hash>
(lldb) b test_name
(lldb) run
```

## ✅ Test Checklist

Before merging PR, ensure:

- [ ] All tests pass locally
- [ ] New code has 90%+ coverage
- [ ] Integration tests added for new features
- [ ] Property tests added for critical logic
- [ ] Benchmarks added for performance-sensitive code
- [ ] Documentation updated
- [ ] CI passes on all platforms
- [ ] No clippy warnings
- [ ] Code formatted with rustfmt

## 🤝 Contributing Tests

### Writing Good Tests

1. **Test one thing** - Each test should verify one behavior
2. **Use descriptive names** - `test_pairs_strategy_exits_on_convergence`
3. **Arrange-Act-Assert** - Clear structure
4. **Independent** - Tests shouldn't depend on each other
5. **Fast** - Unit tests should run in milliseconds
6. **Deterministic** - Same result every time

### Example Template

```rust
#[tokio::test]
async fn test_feature_behavior_under_condition() {
    // Arrange - Set up test data
    let input = create_test_input();
    let expected = calculate_expected_output();

    // Act - Execute the code under test
    let actual = function_under_test(input).await.unwrap();

    // Assert - Verify the results
    assert_eq!(actual, expected);
}
```

## 📈 Continuous Improvement

We continuously improve test quality:
- Monitor coverage trends
- Add tests for bug fixes
- Improve property tests
- Optimize slow tests
- Update mocks as APIs evolve

---

Quality is not an act, it is a habit. Test everything! ✅
