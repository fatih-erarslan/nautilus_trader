# Testing Quick Reference - Neural Trading

## Quick Start Commands

### Run Tests
```bash
# All tests
cargo test

# Unit tests only
cargo test --lib --bins

# Integration tests
cargo test --test '*'

# Specific test
cargo test test_name

# With output
cargo test -- --nocapture

# Run ignored tests (E2E, parity)
cargo test -- --ignored
```

### Coverage
```bash
# Generate coverage report
cargo tarpaulin --out Html --output-dir coverage

# With threshold check
cargo tarpaulin | grep -oP '\d+\.\d+(?=%)'
```

### Benchmarks
```bash
# Run all benchmarks
cargo bench

# Specific benchmark
cargo bench --bench example_benchmark

# Save baseline
cargo bench -- --save-baseline my-baseline

# Compare with baseline
cargo bench -- --baseline my-baseline
```

### Quality Checks
```bash
# Format check
cargo fmt --check

# Linting
cargo clippy --all-targets --all-features -- -D warnings

# Security audit
cargo audit

# License/dependency check
cargo deny check
```

### Generate Reports
```bash
# Full test report
./scripts/generate_test_report.sh

# View HTML report
open test-reports/index.html
```

## Test Organization

```
neural-trader/
├── tests/
│   ├── unit/              # Unit tests (inline and separate files)
│   ├── integration/       # Integration tests
│   ├── e2e/              # End-to-end tests (expensive)
│   ├── parity/           # Python parity tests
│   └── fuzz/             # Fuzz tests
├── benches/              # Criterion benchmarks
├── scripts/              # Test automation scripts
└── .github/workflows/    # CI/CD pipelines
```

## Performance Targets

| Operation | Target | CI Threshold |
|-----------|--------|--------------|
| Market data ingestion | < 100μs | < 120μs |
| Feature extraction | < 1ms | < 1.2ms |
| Signal generation | < 5ms | < 6ms |
| Order placement | < 10ms | < 12ms |
| AgentDB query | < 1ms | < 1.2ms |

## Test Categories

### 1. Unit Tests
- **Location:** `tests/unit/` or `#[cfg(test)]` modules
- **Speed:** < 1ms each
- **Coverage:** > 95% for core logic

```rust
#[test]
fn test_function_valid_input() {
    // Arrange
    let input = create_test_data();

    // Act
    let result = function_under_test(input);

    // Assert
    assert_eq!(result, expected);
}
```

### 2. Property-Based Tests
- **Tool:** `proptest` crate
- **Purpose:** Test invariants with randomized inputs

```rust
proptest! {
    #[test]
    fn test_property(input in strategy) {
        prop_assert!(invariant_holds(input));
    }
}
```

### 3. Integration Tests
- **Location:** `tests/integration/`
- **Purpose:** Test component interactions

```rust
#[tokio::test]
async fn test_pipeline_integration() {
    let system = setup_test_system().await;
    let result = system.process().await.unwrap();
    assert!(result.is_valid());
}
```

### 4. End-to-End Tests
- **Location:** `tests/e2e/`
- **Run:** `cargo test --test e2e -- --ignored`
- **Purpose:** Full system validation

```rust
#[tokio::test]
#[ignore]
async fn test_full_trading_cycle() {
    let system = TradingSystem::new(test_config);
    system.start().await.unwrap();
    // ... assertions
}
```

### 5. Fuzz Tests
- **Tool:** `cargo-fuzz`
- **Run:** `cargo +nightly fuzz run target_name`

```rust
fuzz_target!(|data: &[u8]| {
    let _ = parse_input(data);  // Should never panic
});
```

### 6. Parity Tests
- **Location:** `tests/parity/`
- **Purpose:** Ensure Rust matches Python behavior

```rust
#[test]
fn test_rsi_parity() {
    let rust_result = calculate_rsi(&prices, 14);
    let python_result = call_python_rsi(&prices, 14);
    assert_relative_eq!(rust_result, python_result, epsilon = 1e-6);
}
```

## CI/CD Pipeline

### Quality Gates (All Must Pass)
1. ✅ Code formatting (`cargo fmt`)
2. ✅ Linting (`cargo clippy`)
3. ✅ Documentation complete
4. ✅ Test coverage ≥ 90%
5. ✅ All tests pass (all platforms)
6. ✅ Performance benchmarks meet targets
7. ✅ Security audit (no vulnerabilities)
8. ✅ Python parity verified
9. ✅ MCP compliance
10. ✅ Build succeeds (all targets)

### Workflow Files
- `.github/workflows/ci.yml` - Main CI pipeline
- `.github/workflows/quality-gates.yml` - Quality enforcement

## Common Issues & Solutions

### Flaky Tests
```bash
# Detect flaky test
for i in {1..100}; do cargo test test_name || break; done

# Fix: Add proper setup/teardown, avoid timing dependencies
```

### Slow Tests
```bash
# Profile tests
cargo test -- --nocapture --test-threads=1 --show-output

# Fix: Mock expensive operations, use smaller test data
```

### Coverage Not Meeting Threshold
```bash
# Find uncovered code
cargo tarpaulin --out Html
open target/tarpaulin/index.html

# Add tests for highlighted uncovered lines
```

### Benchmark Regression
```bash
# Compare with baseline
cargo bench -- --baseline main

# Profile slow benchmark
cargo flamegraph --bench benchmark_name -- --bench

# Fix: Optimize hot paths identified in flamegraph
```

## Best Practices

### Test Naming
```rust
// Pattern: test_<function>_<scenario>_<expected>
#[test]
fn test_parse_ohlcv_invalid_timestamp_returns_error() { }
```

### Assertions
```rust
// Specific assertions
assert_eq!(actual, expected);
assert!(condition, "failure message");

// Approximate equality (floating point)
assert_relative_eq!(a, b, epsilon = 1e-6);

// Error matching
assert!(result.is_err());
assert_eq!(result.unwrap_err().kind(), ErrorKind::Expected);
```

### Test Data
```rust
// Use fixtures for realistic data
let data = load_fixture("btc_volatile_2024.json");

// Use builders for complex objects
let order = OrderBuilder::new()
    .symbol("BTC/USD")
    .size(Decimal::from(1))
    .build();
```

### Async Testing
```rust
// Use tokio::test for async
#[tokio::test]
async fn test_async_operation() {
    let result = async_function().await;
    assert!(result.is_ok());
}

// Timeout for long-running tests
#[tokio::test]
#[timeout(5000)]  // 5 seconds
async fn test_with_timeout() { }
```

## Debugging

### Show Test Output
```bash
cargo test -- --nocapture
```

### Run Single Test
```bash
cargo test test_name -- --exact --nocapture
```

### Debug Test in IDE
```rust
#[test]
fn test_with_debug() {
    dbg!(&variable);  // Print debug output
    // ... test code
}
```

### Profile Test
```bash
# CPU profiling
cargo flamegraph --test integration_test -- test_name

# Memory profiling
heaptrack target/debug/deps/integration_test-*
```

## Resources

- **Main Strategy:** `/docs/testing-strategy.md`
- **Example Unit Test:** `/tests/unit/example_unit_test.rs`
- **Example Benchmark:** `/benches/example_benchmark.rs`
- **CI Configuration:** `/.github/workflows/quality-gates.yml`
- **Performance Validation:** `/scripts/validate_performance.py`
- **Report Generation:** `/scripts/generate_test_report.sh`

## Getting Help

```bash
# List all tests
cargo test -- --list

# Test help
cargo test --help

# Benchmark help
cargo bench --help

# Criterion help
cargo bench -- --help
```
