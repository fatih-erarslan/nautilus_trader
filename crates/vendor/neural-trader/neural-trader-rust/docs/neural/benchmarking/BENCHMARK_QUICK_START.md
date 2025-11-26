# CPU Benchmark Quick Start Guide

## Prerequisites

```bash
# Ensure you're in the neural-trader-rust directory
cd /workspaces/neural-trader/neural-trader-rust

# Verify Rust toolchain
rustc --version
cargo --version
```

## Running Benchmarks

### 1. Full Benchmark Suite (~10 minutes)

```bash
# Run all benchmarks and save baseline
cargo bench --bench cpu_benchmarks -- --save-baseline cpu-baseline

# Results will be in: target/criterion/
```

### 2. Specific Benchmark Groups

```bash
# Preprocessing only (~2 minutes)
cargo bench --bench cpu_benchmarks -- preprocessing --save-baseline preprocessing-baseline

# Feature engineering only (~3 minutes)
cargo bench --bench cpu_benchmarks -- feature_engineering --save-baseline features-baseline

# Model inference only (~2 minutes)
cargo bench --bench cpu_benchmarks -- model_inference --save-baseline inference-baseline

# Training only (~2 minutes)
cargo bench --bench cpu_benchmarks -- training --save-baseline training-baseline

# Memory only (~1 minute)
cargo bench --bench cpu_benchmarks -- memory --save-baseline memory-baseline
```

### 3. Compare Against Baseline

```bash
# After making optimizations, compare results
cargo bench --bench cpu_benchmarks -- --baseline cpu-baseline

# Criterion will show performance differences:
# - Green: Improvement
# - Red: Regression
# - Gray: No significant change
```

### 4. Individual Benchmarks

```bash
# Run specific operation
cargo bench --bench cpu_benchmarks -- preprocessing/zscore

# With baseline comparison
cargo bench --bench cpu_benchmarks -- preprocessing/zscore --baseline cpu-baseline
```

## Analyzing Results

### 1. Terminal Output

Criterion prints results directly:
```
preprocessing/zscore/100
                        time:   [198.45 ns 199.82 ns 201.31 ns]
                        change: [-2.1234% -1.0123% +0.0234%] (p = 0.05)
                        Performance has improved.
```

### 2. HTML Reports

```bash
# Generate detailed HTML report
open target/criterion/report/index.html

# Or navigate to specific benchmark
open target/criterion/preprocessing/zscore/100/report/index.html
```

### 3. Baseline Comparisons

```bash
# List saved baselines
ls target/criterion/*/base/

# View specific baseline
cat target/criterion/preprocessing/zscore/100/base/estimates.json
```

## Benchmark Structure

```
cpu_benchmarks.rs
├── 1. Preprocessing (28 benchmarks)
│   ├── Normalization
│   │   ├── zscore (4 sizes: 100, 1K, 10K, 100K)
│   │   ├── minmax (4 sizes)
│   │   └── robust (4 sizes)
│   ├── Differencing
│   │   ├── first_order (4 sizes)
│   │   └── second_order (4 sizes)
│   ├── Detrending
│   │   └── linear (4 sizes)
│   └── Outliers
│       └── iqr (4 sizes)
│
├── 2. Feature Engineering (64 benchmarks)
│   ├── Lag creation (16: 4 sizes × 4 lag counts)
│   ├── Rolling statistics (32: 3 sizes × 3 windows × 4 stats)
│   ├── Technical indicators (8)
│   └── Fourier features (8)
│
├── 3. Model Inference (20 benchmarks)
│   ├── GRU (4 batch sizes)
│   ├── TCN (4 batch sizes)
│   ├── N-BEATS (3 sequence lengths)
│   └── Prophet (3 time periods)
│
├── 4. Training (13 benchmarks)
│   ├── Single epoch (3 configurations)
│   ├── Gradient computation (3 configurations)
│   ├── Parameter update (4 feature sizes)
│   └── Full loop (2 configurations)
│
└── 5. Memory (11 benchmarks)
    ├── Allocation (3 sizes)
    ├── Clone (4 sizes)
    ├── Cache-efficient (3 sizes)
    └── Cache-inefficient (3 dimensions)

Total: 136 individual benchmarks
```

## Performance Targets

| Category | Operation | Target | Status |
|----------|-----------|--------|--------|
| Preprocessing | Normalization (100K) | < 500µs | ✅ 200µs |
| Features | Rolling stats (10K, w=100) | < 100µs | ❌ 800µs |
| Features | Fourier (10K, 10 freq) | < 1ms | ❌ 5ms |
| Inference | GRU (batch=32) | < 500µs | ❌ 2.5ms |
| Training | Epoch (10K×100) | < 10ms | ❌ 75ms |
| Memory | Allocation | < 1µs/KB | ✅ 0.25µs/KB |

## Troubleshooting

### Benchmark Compilation Errors

```bash
# Check for missing dependencies
cargo check --bench cpu_benchmarks

# Update dependencies
cargo update

# Clean and rebuild
cargo clean
cargo bench --bench cpu_benchmarks
```

### Slow Compilation

```bash
# Use release mode only (faster compilation)
cargo bench --bench cpu_benchmarks --profile bench

# Limit parallel jobs to reduce memory usage
cargo bench --bench cpu_benchmarks -- -j 1
```

### Inconsistent Results

```bash
# Increase sample size (default: 100)
cargo bench --bench cpu_benchmarks -- --sample-size 200

# Increase measurement time (default: 10s)
cargo bench --bench cpu_benchmarks -- --measurement-time 20

# Reduce system noise
sudo renice -n -20 -p $$  # Run as root for real-time priority
```

### Comparing Multiple Baselines

```bash
# Save baseline with custom name
cargo bench --bench cpu_benchmarks -- --save-baseline before-optimization

# Make changes...

# Save another baseline
cargo bench --bench cpu_benchmarks -- --save-baseline after-optimization

# Compare
cargo bench --bench cpu_benchmarks -- --baseline before-optimization
```

## Advanced Usage

### 1. Profiling with flamegraph

```bash
# Install cargo-flamegraph
cargo install flamegraph

# Profile specific benchmark
sudo cargo flamegraph --bench cpu_benchmarks -- preprocessing/zscore/100000 --profile-time 30

# Open flamegraph.svg in browser
```

### 2. Memory Profiling with Valgrind

```bash
# Install valgrind
sudo apt-get install valgrind

# Run under valgrind
valgrind --tool=massif cargo bench --bench cpu_benchmarks -- preprocessing/zscore/100000 --profile-time 10

# Analyze results
ms_print massif.out.*
```

### 3. Continuous Benchmarking

```bash
#!/bin/bash
# bench_watch.sh - Run benchmarks on code changes

while true; do
    inotifywait -r -e modify src/
    cargo bench --bench cpu_benchmarks -- --save-baseline latest
    echo "Benchmarks updated at $(date)"
done
```

### 4. CI Integration

```yaml
# .github/workflows/benchmark.yml
name: Benchmark

on:
  push:
    branches: [main]
  pull_request:

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions-rs/toolchain@v1
        with:
          toolchain: stable
      - name: Run benchmarks
        run: |
          cd neural-trader-rust
          cargo bench --bench cpu_benchmarks -- --save-baseline ci-baseline
      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: benchmark-results
          path: target/criterion/
```

## Interpreting Results

### Statistical Significance

Criterion reports confidence intervals and p-values:
- **p < 0.05**: Statistically significant change
- **p ≥ 0.05**: No significant change (noise)

### Change Indicators

- **Change > +5%**: Performance regression (investigate)
- **Change -5% to +5%**: Acceptable variation
- **Change < -5%**: Performance improvement (celebrate!)

### Regression Detection

```bash
# Set regression threshold
cargo bench --bench cpu_benchmarks -- --significance-level 0.01

# Fail on regression (CI use)
cargo bench --bench cpu_benchmarks -- --test
```

## Next Steps

After running benchmarks:

1. **Review** `/workspaces/neural-trader/docs/neural/CPU_BENCHMARK_RESULTS.md`
2. **Prioritize** bottlenecks: Critical → High → Medium
3. **Implement** optimizations from the roadmap
4. **Validate** improvements with comparison runs
5. **Update** baselines after confirmed improvements

## Resources

- **Criterion.rs Docs**: https://bheisler.github.io/criterion.rs/book/
- **Rust Performance Book**: https://nnethercote.github.io/perf-book/
- **Analysis Report**: `/workspaces/neural-trader/docs/neural/CPU_BENCHMARK_RESULTS.md`
- **Benchmark Code**: `/workspaces/neural-trader/neural-trader-rust/crates/neural/benches/cpu_benchmarks.rs`
