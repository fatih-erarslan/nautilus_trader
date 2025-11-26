# Neural Trading Rust Benchmark Suite

This directory contains comprehensive performance benchmarks for the Neural Trading Rust port using criterion.rs.

## Benchmark Files

### 1. `market_data_throughput.rs`
**Focus:** Market data ingestion and processing performance

**Benchmarks:**
- Tick ingestion pipeline (100, 1K, 10K ticks)
- JSON parsing (single and batch)
- Quote processing with spread calculations
- Bar aggregation across timeframes
- Memory allocation overhead

**Key Metrics:**
- Target: <100μs per tick
- Throughput: >10,000 ticks/sec
- Memory: <10MB per connection

### 2. `feature_extraction_latency.rs`
**Focus:** Technical indicator calculation performance

**Benchmarks:**
- Simple Moving Average (SMA) - various windows
- Exponential Moving Average (EMA)
- Relative Strength Index (RSI)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Full feature extraction pipeline
- Streaming updates

**Key Metrics:**
- Target: <1ms for 100 bars
- Full pipeline: <5ms (7 indicators)
- Streaming update: <100μs

### 3. `strategy_execution.rs`
**Focus:** Strategy signal generation and processing

**Benchmarks:**
- Momentum strategy (uptrend, downtrend, sideways)
- Mean reversion strategy
- Multi-strategy parallel execution
- Signal generation scenarios
- Strategy state updates
- Initialization overhead

**Key Metrics:**
- Target: <5ms per strategy
- Multi-strategy (10 parallel): <50ms
- State update: <500μs

### 4. `order_placement.rs`
**Focus:** Order creation, validation, and execution

**Benchmarks:**
- Order creation (market, limit, stop-loss)
- Order validation with portfolio checks
- Order placement with simulated latency
- Batch order processing
- Order routing
- Order manager operations
- Cancellation flow

**Key Metrics:**
- Order creation: <1ms
- Order validation: <2ms
- Order placement: <10ms (internal)
- End-to-end: <61ms (including broker API)

### 5. `risk_calculations.rs`
**Focus:** Risk management and portfolio risk metrics

**Benchmarks:**
- Value at Risk (VaR) - Historical and Parametric
- Conditional Value at Risk (CVaR)
- Position limit checks
- Maximum drawdown calculation
- Sharpe ratio
- Portfolio-wide risk metrics
- Stop loss calculations
- Correlation matrix
- Kelly criterion position sizing

**Key Metrics:**
- VaR calculation: <2ms (252 days)
- CVaR calculation: <3ms
- Position limit check: <500μs
- Correlation matrix (20 assets): <20ms

### 6. `portfolio_updates.rs`
**Focus:** Portfolio management and position tracking

**Benchmarks:**
- Position updates (add, update, remove)
- Unrealized P&L calculations
- Realized P&L calculations
- Total portfolio P&L
- Portfolio metrics (value, buying power, leverage)
- Trade processing (single and batch)
- Position sizing calculations
- Concurrent updates
- Portfolio snapshots
- Commission calculations

**Key Metrics:**
- Position update: <100μs
- Unrealized P&L: <50μs
- Trade processing: <200μs
- Batch trades (100): <10ms

### 7. `agentdb_queries.rs`
**Focus:** Vector database operations and memory management

**Benchmarks:**
- Vector generation and normalization (128-1536 dimensions)
- Cosine similarity calculations
- Distance calculations (Euclidean, Manhattan)
- Memory storage (single and batch)
- K-nearest neighbors search
- HNSW index operations
- Pattern storage and retrieval
- ReasoningBank operations
- Memory distillation
- Embedding cache

**Key Metrics:**
- Vector normalization: <10μs (384 dim)
- K-NN search (k=10): <1ms (10K vectors)
- HNSW search: <1ms (100K vectors)
- Pattern retrieval: <500μs

## Running Benchmarks

### Run All Benchmarks
```bash
cargo bench --workspace
```

### Run Specific Benchmark Suite
```bash
cargo bench --bench market_data_throughput
cargo bench --bench feature_extraction_latency
cargo bench --bench strategy_execution
cargo bench --bench order_placement
cargo bench --bench risk_calculations
cargo bench --bench portfolio_updates
cargo bench --bench agentdb_queries
```

### Run Specific Benchmark
```bash
cargo bench --bench market_data_throughput -- tick_ingestion
cargo bench --bench feature_extraction_latency -- sma_calculation
cargo bench --bench strategy_execution -- momentum_strategy
```

### Baseline Comparison
```bash
# Save current baseline
cargo bench -- --save-baseline main

# Make changes, then compare
cargo bench -- --baseline main
```

### View HTML Reports
```bash
cargo bench
open target/criterion/report/index.html
```

## Configuration

Benchmark configuration in `Cargo.toml`:
```toml
[profile.bench]
inherits = "release"
debug = true
```

Criterion configuration:
- Sample size: 100
- Measurement time: 5 seconds
- Warm-up time: 2 seconds
- Statistical analysis with confidence intervals
- Outlier detection enabled

## Performance Targets

| Component | Target | Priority |
|-----------|--------|----------|
| Market Data | <100μs/tick | P0 |
| Features | <1ms/100 bars | P0 |
| Strategies | <5ms/signal | P0 |
| Orders | <10ms | P0 |
| Risk | <2ms | P0 |
| Portfolio | <100μs/update | P0 |
| AgentDB | <1ms/query | P0 |

## CI/CD Integration

Benchmarks run automatically in CI:
- On every PR
- Regression detection (>5% slowdown fails CI)
- Historical trend tracking
- Automated alerts

## Profiling Tools

### Flamegraph
```bash
cargo install flamegraph
cargo flamegraph --bench market_data_throughput
```

### Memory Profiling
```bash
valgrind --tool=massif target/release/neural-trader
```

### CPU Profiling
```bash
perf record -g target/release/neural-trader
perf report
```

## Documentation

See `/docs/PERFORMANCE_REPORT.md` for detailed analysis and optimization strategies.

## Cross-References

- Performance targets: `/plans/neural-rust/13_Tests_Benchmarks_CI.md`
- Architecture: `/plans/neural-rust/03_Architecture.md`
- Performance report: `/docs/PERFORMANCE_REPORT.md`
