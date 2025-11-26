# Neural Trader Rust Port - Validation Instructions

This document provides step-by-step instructions for running the comprehensive validation suite once compilation issues are resolved.

## Prerequisites

### 1. Fix Compilation Errors

Before running validation, ensure all compilation errors are fixed:

```bash
# Should complete without errors
cargo build --release --all-features
```

**Current blockers (as of 2025-11-12):**
- Execution crate: 129 errors (type system issues)
- Neural crate: 20 errors (missing candle dependencies)
- Integration crate: 1 error (field mismatch)

See `/docs/VALIDATION_REPORT.md` for detailed fix instructions.

### 2. Install Dependencies

```bash
# Install required tools
cargo install cargo-tarpaulin  # For coverage
cargo install cargo-criterion  # For benchmarking
cargo install cargo-watch      # For continuous testing

# For neural models (after fixing imports)
# Add to crates/neural/Cargo.toml:
# candle-core = "0.3"
# candle-nn = "0.3"
```

### 3. Set Up Test Environment

```bash
# Copy example config
cp config.example.toml config.toml

# Set up environment variables (for broker tests)
export ALPACA_API_KEY="your_paper_key"
export ALPACA_SECRET_KEY="your_paper_secret"
export ALPACA_BASE_URL="https://paper-api.alpaca.markets"

# For sports betting tests
export ODDS_API_KEY="your_odds_api_key"

# For E2B tests
export E2B_API_KEY="your_e2b_key"
```

## Validation Phases

### Phase 1: Unit Tests (2-3 hours)

Run all unit tests for individual components:

```bash
# Test all crates
cargo test --lib --all-features

# Test specific crate
cargo test -p nt-strategies --lib
cargo test -p nt-risk --lib
cargo test -p multi-market --lib
cargo test -p mcp-server --lib

# Test with output
cargo test --lib --all-features -- --nocapture

# Test specific module
cargo test --lib strategies::pairs_trading
```

**Expected Results:**
- All unit tests pass
- No panics or crashes
- Correct behavior for edge cases

### Phase 2: Integration Tests (3-4 hours)

Run integration tests that test component interactions:

```bash
# Run all validation tests
cargo test --test '*' --all-features

# Run specific validation suite
cargo test --test validation test_strategies
cargo test --test validation test_brokers
cargo test --test validation test_risk
cargo test --test validation test_mcp

# Run with specific pattern
cargo test --test validation sports_betting
```

**Note:** Some tests require API keys and are marked with `#[ignore]`:

```bash
# Run ignored tests (requires setup)
cargo test --test validation -- --ignored

# Run all tests including ignored
cargo test --test validation -- --include-ignored
```

### Phase 3: Performance Benchmarks (2-3 hours)

Run comprehensive performance benchmarks:

```bash
# Run all benchmarks
cargo bench --all-features

# Run specific benchmark
cargo bench --bench strategy_bench
cargo bench --bench neural_bench
cargo bench --bench risk_bench

# Generate HTML reports
cargo bench --all-features -- --output-format bencher | tee output.txt
```

**Performance Targets:**
- Backtest: 2000+ bars/sec
- Neural inference: <10ms
- Risk calculation: <20ms
- API response: <50ms
- Memory usage: <200MB

### Phase 4: Coverage Analysis (1-2 hours)

Measure test coverage:

```bash
# Generate coverage report
cargo tarpaulin --all --all-features --out Html --output-dir coverage

# View report
open coverage/index.html

# Get summary
cargo tarpaulin --all --all-features
```

**Coverage Targets:**
- Line coverage: >90%
- Branch coverage: >85%
- Function coverage: >95%

### Phase 5: Stress Testing (2-3 hours)

Test system under load:

```bash
# Concurrent operations test
cargo test --test validation test_concurrent_operations -- --nocapture

# Memory leak test
cargo test --test validation benchmark_memory_usage -- --nocapture

# Long-running test
cargo test --test validation -- --nocapture --test-threads=1
```

## Test Organization

### Test Files

```
tests/validation/
├── mod.rs                    # Test utilities and helpers
├── test_strategies.rs        # 8 trading strategies
├── test_brokers.rs           # 11 broker integrations
├── test_neural.rs            # 3 neural models
├── test_multi_market.rs      # 3 market types
├── test_risk.rs              # 5 risk components
├── test_mcp.rs               # 87 MCP tools
├── test_distributed.rs       # 4 distributed systems
├── test_memory.rs            # 4 memory layers
├── test_integration.rs       # 4 integration components
└── test_performance.rs       # 5 performance benchmarks
```

### Helper Functions

All tests can use helpers from `tests/validation/mod.rs`:

```rust
use tests::validation::helpers::*;

// Generate test data
let bars = generate_sample_bars(1000);

// Assert performance
assert_performance_target(actual_ms, target_ms, tolerance);

// Calculate metrics
let sharpe = calculate_sharpe_ratio(&returns, risk_free_rate);
```

## Continuous Testing

Set up continuous testing during development:

```bash
# Watch for changes and run tests
cargo watch -x test

# Watch and run specific tests
cargo watch -x 'test --test validation test_strategies'

# Watch and run benchmarks
cargo watch -x 'bench --bench strategy_bench'
```

## Test Reporting

### Generate Validation Report

After running all tests:

```bash
# Run full validation suite
./scripts/run_validation.sh

# This will:
# 1. Run all tests
# 2. Generate coverage report
# 3. Run benchmarks
# 4. Create validation report in docs/VALIDATION_REPORT.md
```

### Report Format

The validation report includes:

1. **Executive Summary**
   - Total features validated
   - Pass/fail statistics
   - Coverage percentage

2. **Detailed Results**
   - Per-category results
   - Performance benchmarks
   - Failed tests with details

3. **Performance Metrics**
   - Comparison vs Python baseline
   - Latency measurements
   - Throughput metrics

4. **Recommendations**
   - Priority fixes
   - Optimization opportunities
   - Missing features

## Troubleshooting

### Common Issues

#### 1. Tests Fail to Compile

```bash
# Check for compilation errors
cargo check --tests --all-features

# Fix and retry
cargo test
```

#### 2. Tests Time Out

```bash
# Increase timeout
RUST_TEST_THREADS=1 cargo test -- --test-threads=1 --nocapture

# Or skip slow tests
cargo test -- --skip slow_test
```

#### 3. Missing Dependencies

```bash
# Update dependencies
cargo update

# Clean and rebuild
cargo clean
cargo build --all-features
```

#### 4. API Key Issues

```bash
# Verify environment variables
echo $ALPACA_API_KEY
echo $ODDS_API_KEY

# Run without external API tests
cargo test -- --skip api_test
```

## CI/CD Integration

### GitHub Actions

Create `.github/workflows/validation.yml`:

```yaml
name: Validation

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions-rs/toolchain@v1
        with:
          toolchain: stable
      - name: Run tests
        run: cargo test --all --all-features
      - name: Run benchmarks
        run: cargo bench --all --all-features
      - name: Generate coverage
        run: cargo tarpaulin --all --all-features --out Xml
      - name: Upload coverage
        uses: codecov/codecov-action@v1
```

## Next Steps

After successful validation:

1. **Document Results**
   - Update VALIDATION_REPORT.md
   - Create performance comparison charts
   - Document any limitations

2. **Fix Issues**
   - Prioritize failed tests
   - Optimize slow operations
   - Improve coverage

3. **Production Prep**
   - Security audit
   - Load testing
   - Deployment automation

4. **Release**
   - Version tagging
   - Changelog
   - Documentation

## Support

For issues or questions:
- Check `/docs/VALIDATION_REPORT.md` for known issues
- Review test output with `--nocapture` flag
- Run with `RUST_BACKTRACE=1` for detailed errors

---

**Last Updated:** 2025-11-12
**Status:** Ready for validation (pending compilation fixes)
