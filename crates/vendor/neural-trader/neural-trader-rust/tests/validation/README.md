# Neural Trader Validation Test Suite

Comprehensive validation tests for the Neural Trader Rust port.

---

## Quick Start

```bash
# Run all validation tests
cargo test --test '*' --all-features

# Run specific test suite
cargo test --test validation test_strategies
cargo test --test validation test_brokers
cargo test --test validation test_risk

# Run with output
cargo test --test validation -- --nocapture

# Run ignored tests (requires API keys)
cargo test --test validation -- --ignored
```

---

## Test Organization

### Test Files

| File | Category | Tests | Status |
|------|----------|-------|--------|
| `mod.rs` | Test utilities & helpers | N/A | ✅ Ready |
| `test_strategies.rs` | 8 trading strategies | 15+ | ⏸️ Pending compilation |
| `test_brokers.rs` | 11 broker integrations | 25+ | ⏸️ Pending compilation |
| `test_neural.rs` | 3 neural models | 15+ | ⏸️ Pending compilation |
| `test_risk.rs` | 5 risk components | 10+ | ⏸️ Pending compilation |
| `test_mcp.rs` | 87 MCP tools | 87+ | ⏸️ Pending compilation |
| `test_multi_market.rs` | 3 market types | 15+ | ⏸️ Pending compilation |
| `test_distributed.rs` | 4 distributed systems | 5+ | ⏸️ Pending compilation |
| `test_memory.rs` | 4 memory layers | 5+ | ⏸️ Pending compilation |
| `test_integration.rs` | 4 integration APIs | 5+ | ⏸️ Pending compilation |
| `test_performance.rs` | 5 performance benchmarks | 10+ | ⏸️ Pending compilation |

**Total:** 1,187 lines of test code, ~150+ test cases

---

## Test Categories

### 1. Trading Strategies (`test_strategies.rs`)

Tests for all 8 core trading strategies:
- Pairs Trading (cointegration, spread forecasting)
- Mean Reversion (z-score, Bollinger bands)
- Momentum (trend following, breakouts)
- Market Making (bid-ask spread, inventory)
- Arbitrage (cross-exchange, triangular)
- Portfolio Optimization (Markowitz, efficient frontier)
- Risk Parity (equal risk contribution)
- Sentiment Driven (news, social media)

**Performance Target:** 2000+ bars/sec backtesting

### 2. Broker Integrations (`test_brokers.rs`)

Tests for all 11 broker integrations:
- Interactive Brokers, Alpaca, TD Ameritrade
- CCXT (100+ exchanges), Polygon.io, Tradier
- Questrade, OANDA, Binance, Coinbase, Kraken

**Note:** Most broker tests are `#[ignore]` (require API keys)

### 3. Neural Models (`test_neural.rs`)

Tests for all 3 neural models:
- NHITS (hierarchical temporal)
- LSTM with Attention
- Transformer (time series)

**Performance Target:** <10ms inference latency

### 4. Risk Management (`test_risk.rs`)

Tests for risk components:
- Monte Carlo VaR (GPU accelerated)
- Kelly Criterion (single & multi-asset)
- Stress Testing (2008, 2020 scenarios)
- Position Limits & Enforcement
- Emergency Protocols & Circuit Breakers

**Performance Target:** <20ms risk calculations

### 5. MCP Protocol (`test_mcp.rs`)

Tests for all 87 MCP tools:
- System tools (ping, list_strategies)
- Trading tools (execute, simulate, portfolio)
- Neural tools (train, predict, optimize)
- Sports betting tools
- Risk analysis tools
- News sentiment tools

**Performance Target:** <100ms tool execution

### 6. Multi-Market (`test_multi_market.rs`)

Tests for multi-market support:
- Sports Betting (Kelly Criterion, arbitrage)
- Prediction Markets (Polymarket, EV calculation)
- Cryptocurrency (DeFi, yield farming, arbitrage)

### 7. Distributed Systems (`test_distributed.rs`)

Tests for distributed components:
- E2B Sandbox creation & execution
- Agentic-Flow Federations
- Agentic-Payments integration
- Auto-scaling & load balancing

### 8. Memory Systems (`test_memory.rs`)

Tests for memory layers:
- L1 Cache (DashMap, <1μs access)
- L2 AgentDB (vector DB, <1ms query)
- L3 Cold Storage (Sled, compression)
- ReasoningBank integration

### 9. Integration Layer (`test_integration.rs`)

Tests for integration APIs:
- REST API (Axum)
- WebSocket streaming
- CLI interface
- Configuration management

### 10. Performance (`test_performance.rs`)

Comprehensive performance benchmarks:
- Strategy backtest speed (2000+ bars/sec)
- Neural inference latency (<10ms)
- Risk calculation speed (<20ms)
- API response time (<50ms)
- Memory usage (<200MB)

---

## Test Helpers (`mod.rs`)

The `helpers` module provides utilities for all tests:

### Data Generation
```rust
use super::helpers::*;

// Generate sample bars
let bars = generate_sample_bars(1000);
```

### Performance Assertions
```rust
// Assert performance target is met
assert_performance_target(actual_ms, target_ms, tolerance);

// Example: 50ms with 20% tolerance = max 60ms
assert_performance_target(45.0, 50.0, 0.2); // ✅ Pass
assert_performance_target(65.0, 50.0, 0.2); // ❌ Fail
```

### Metrics Calculation
```rust
// Calculate Sharpe ratio
let sharpe = calculate_sharpe_ratio(&returns, risk_free_rate);
```

---

## Running Tests

### All Tests
```bash
# Run everything
cargo test --test '*' --all-features

# Run with output
cargo test --test '*' --all-features -- --nocapture

# Run with specific thread count
cargo test --test '*' --all-features -- --test-threads=4
```

### Specific Test Suite
```bash
# Strategies only
cargo test --test validation test_strategies

# Risk management only
cargo test --test validation test_risk

# Performance benchmarks only
cargo test --test validation test_performance
```

### By Pattern
```bash
# All tests containing "kelly"
cargo test --test validation kelly

# All tests containing "arbitrage"
cargo test --test validation arbitrage
```

### With API Keys (Ignored Tests)
```bash
# Run only ignored tests
cargo test --test validation -- --ignored

# Run all tests including ignored
cargo test --test validation -- --include-ignored
```

---

## Test Environment Setup

### Required for All Tests
- Rust toolchain (stable)
- All dependencies installed

### Required for Broker Tests
```bash
export ALPACA_API_KEY="your_paper_key"
export ALPACA_SECRET_KEY="your_paper_secret"
export ALPACA_BASE_URL="https://paper-api.alpaca.markets"
```

### Required for Sports Betting Tests
```bash
export ODDS_API_KEY="your_odds_api_key"
```

### Required for Distributed Tests
```bash
export E2B_API_KEY="your_e2b_key"
```

---

## Writing New Tests

### Test Structure
```rust
#[cfg(test)]
mod my_tests {
    use super::*;

    #[tokio::test]
    async fn test_my_feature() {
        // Arrange
        let input = prepare_test_data();

        // Act
        let result = my_feature(input).await;

        // Assert
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), expected);
    }

    #[tokio::test]
    #[ignore] // Requires external API
    async fn test_external_api() {
        // Test code...
    }
}
```

### Using Test Helpers
```rust
#[tokio::test]
async fn test_backtest_performance() {
    use std::time::Instant;
    use super::helpers::*;

    let bars = generate_sample_bars(10000);
    let start = Instant::now();

    // Run backtest
    strategy.backtest(&bars).await;

    let elapsed = start.elapsed().as_millis() as f64;
    assert_performance_target(elapsed, 100.0, 0.2);
}
```

---

## Test Status

### ✅ Ready to Run
- Test framework complete
- Helper utilities implemented
- Test structure defined

### ⏸️ Pending Compilation
All tests are blocked by compilation errors in:
- Execution crate (129 errors)
- Neural crate (20 errors)
- Integration crate (1 error)

Once compilation succeeds, all tests can be executed immediately.

---

## Performance Targets

All performance tests validate against these targets:

| Component | Target | Python Baseline | Speedup |
|-----------|--------|-----------------|---------|
| Backtest | 2000+ bars/sec | 500 bars/sec | 4x |
| Neural Inference | <10ms | ~50ms | 5x |
| Risk Calculation | <20ms | ~200ms | 10x |
| API Response | <50ms | 100-200ms | 2-4x |
| Memory Usage | <200MB | ~500MB | 2.5x |

---

## Coverage Goals

### Target Coverage
- **Line Coverage:** >90%
- **Branch Coverage:** >85%
- **Function Coverage:** >95%

### Measure Coverage
```bash
cargo tarpaulin --all --all-features --out Html
open coverage/index.html
```

---

## Continuous Testing

### Watch Mode
```bash
# Auto-run tests on file changes
cargo watch -x 'test --test validation'

# With clear screen
cargo watch -c -x 'test --test validation'
```

### Pre-Commit Hook
```bash
# Add to .git/hooks/pre-commit
#!/bin/bash
cargo test --test '*' --all-features
```

---

## Troubleshooting

### Tests Won't Compile
```bash
# Check compilation errors
cargo check --tests --all-features

# Fix and retry
cargo test --test validation
```

### Tests Time Out
```bash
# Increase timeout
RUST_TEST_THREADS=1 cargo test -- --test-threads=1

# Or skip slow tests
cargo test -- --skip slow_test
```

### Missing Dependencies
```bash
# Update dependencies
cargo update

# Clean and rebuild
cargo clean
cargo test --test validation
```

---

## CI/CD Integration

### GitHub Actions Example
```yaml
name: Validation Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions-rs/toolchain@v1
      - name: Run tests
        run: cargo test --test '*' --all-features
```

---

## Documentation

For more information:
- **Quick Start:** `/VALIDATION_QUICKSTART.md`
- **Full Instructions:** `/docs/VALIDATION_INSTRUCTIONS.md`
- **Complete Report:** `/docs/VALIDATION_REPORT.md`
- **Automation:** `/scripts/run_validation.sh`

---

## Support

Questions or issues?
1. Check `/docs/VALIDATION_INSTRUCTIONS.md` for detailed guide
2. Review test output with `--nocapture` flag
3. Run with `RUST_BACKTRACE=1` for detailed errors

---

**Test Suite Version:** 1.0
**Last Updated:** 2025-11-12
**Status:** Ready for execution (pending compilation)
