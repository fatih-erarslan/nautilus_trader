# Neural Trader Performance Benchmarks

Comprehensive benchmark suite for measuring and optimizing the performance of all 70+ neural-trader functions.

## 📊 What's Included

### 1. Function Performance Benchmarks
**File**: `function-performance.benchmark.js`

Measures individual function performance across all categories:
- System & Initialization
- Trading Operations
- Backtesting
- Neural Networks
- Sports Betting
- Syndicate Management
- E2B Swarm
- Security & Authentication
- Risk Analysis
- Prediction Markets

**Metrics**:
- Operations per second (throughput)
- Mean execution time (ms)
- Standard deviation
- Memory usage per operation
- Sample count

### 2. Scalability Benchmarks
**File**: `scalability.benchmark.js`

Tests performance under increasing load:
- **Concurrency**: 1, 10, 100, 500, 1000 concurrent operations
- **Portfolio Size**: 10, 100, 1K, 10K positions
- **Swarm Agents**: 1, 5, 10, 25, 50, 100 agents
- **Dataset Size**: 1-month, 3-months, 1-year, 5-years
- **Memory Growth**: Leak detection over repeated operations

**Analysis**:
- Time complexity calculation
- Scaling efficiency
- Bottleneck identification
- Memory leak detection

### 3. GPU Comparison Benchmarks
**File**: `gpu-comparison.benchmark.js`

Compares CPU vs GPU performance for all GPU-capable functions:
- Quick Analysis
- Trade Simulation
- Backtesting (multiple timeframes)
- Strategy Optimization
- Neural Forecasting
- Risk Analysis
- Correlation Analysis
- Neural Training/Evaluation

**Outputs**:
- Speedup ratios (CPU time / GPU time)
- Memory usage comparison
- Cost-benefit analysis
- Optimization recommendations

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
npm install

# Or install benchmark deps separately
npm install --save-dev benchmark microtime cli-table3 chalk ora
```

### Run All Benchmarks

```bash
# Standard run (all benchmarks)
npm run benchmark

# Or directly
node tests/benchmarks/run-all.js

# With options
node tests/benchmarks/run-all.js --export-json --export-html
```

### Run Individual Benchmarks

```bash
# Function performance only
node tests/benchmarks/function-performance.benchmark.js

# Scalability tests only
node tests/benchmarks/scalability.benchmark.js

# GPU comparison only
node tests/benchmarks/gpu-comparison.benchmark.js
```

### Memory Profiling

```bash
# Enable garbage collector access for accurate memory measurements
node --expose-gc tests/benchmarks/scalability.benchmark.js

# Increase heap size for large-scale tests
node --expose-gc --max-old-space-size=4096 tests/benchmarks/run-all.js
```

## 📈 Command Line Options

```bash
# Export results to JSON
node tests/benchmarks/run-all.js --export-json

# Generate HTML report
node tests/benchmarks/run-all.js --export-html

# Skip GPU benchmarks (CPU-only systems)
node tests/benchmarks/run-all.js --skip-gpu

# Quick mode (fewer iterations, faster results)
node tests/benchmarks/run-all.js --quick

# Combine options
node tests/benchmarks/run-all.js --export-json --export-html --skip-gpu
```

## 📊 Understanding Results

### Function Performance Output

```
╔══════════════════════════════════════════════════════════╗
║   NEURAL TRADER COMPREHENSIVE PERFORMANCE BENCHMARKS     ║
╚══════════════════════════════════════════════════════════╝

TRADING OPERATIONS

┌────────────────────┬──────────┬───────────┬────────┬─────────┬──────────────┐
│ Function           │ Ops/sec  │ Mean (ms) │ ±RME   │ Samples │ Δ Memory (MB)│
├────────────────────┼──────────┼───────────┼────────┼─────────┼──────────────┤
│ quickAnalysis_GPU  │ 238      │ 4.20      │ 2.1%   │ 50      │ +3.1         │
│ quickAnalysis      │ 80       │ 12.50     │ 1.8%   │ 50      │ +2.3         │
│ simulateTrade_GPU  │ 323      │ 3.10      │ 1.5%   │ 50      │ +2.4         │
└────────────────────┴──────────┴───────────┴────────┴─────────┴──────────────┘
```

**Interpretation**:
- **Ops/sec**: Higher is better (more operations per second)
- **Mean (ms)**: Lower is better (faster execution)
- **±RME**: Relative margin of error (lower = more consistent)
- **Δ Memory**: Memory impact (watch for large positive values)

### Scalability Output

```
CONCURRENCY SCALABILITY

┌────────────────┬────────────────┬──────────────┬──────────────┬───────────────┬─────────────┐
│ Concurrent Ops │ Total Time (ms)│ Avg Time (ms)│ Success Rate │ Throughput    │ Memory (MB) │
├────────────────┼────────────────┼──────────────┼──────────────┼───────────────┼─────────────┤
│ 1              │ 12.50          │ 12.50        │ 100.00%      │ 80            │ 45.2        │
│ 100            │ 780.00         │ 7.80         │ 99.80%       │ 128           │ 67.5        │
│ 1000           │ 6100.00        │ 6.10         │ 94.70%       │ 164           │ 145.8       │
└────────────────┴────────────────┴──────────────┴──────────────┴───────────────┴─────────────┘
```

**Key Metrics**:
- **Success Rate**: Should stay >95% (connection pool issues if lower)
- **Throughput**: Should increase with concurrency (up to a limit)
- **Memory**: Watch for excessive growth

### GPU Comparison Output

```
GPU vs CPU PERFORMANCE COMPARISON

┌──────────────────────┬─────────┬─────────┬─────────┬─────────────┬──────────────┬──────────────┬─────────────────┐
│ Operation            │ CPU (ms)│ GPU (ms)│ Speedup │ Improvement │ CPU Mem (MB) │ GPU Mem (MB) │ Recommendation  │
├──────────────────────┼─────────┼─────────┼─────────┼─────────────┼──────────────┼──────────────┼─────────────────┤
│ Neural Forecast      │ 85.00   │ 28.00   │ 3.04x   │ +204.0%     │ 45.20        │ 78.50        │ USE GPU         │
│ Backtest - 1 Year    │ 1640.00 │ 592.00  │ 2.77x   │ +177.0%     │ 12.30        │ 18.70        │ USE GPU         │
│ Quick Analysis       │ 12.50   │ 4.20    │ 2.98x   │ +198.0%     │ 2.30         │ 3.10         │ USE GPU         │
└──────────────────────┴─────────┴─────────┴─────────┴─────────────┴──────────────┴──────────────┴─────────────────┘
```

**Recommendations**:
- **USE GPU**: ≥2x speedup - always use GPU
- **GPU BENEFICIAL**: 1.5-2x speedup - prefer GPU
- **CPU SUFFICIENT**: <1.1x speedup - GPU optional

### Bottleneck Analysis

```
BOTTLENECK ANALYSIS

┌──────────┬──────────────────────┬──────────────────────────────────────┬────────────────────────────────┐
│ Severity │ Area                 │ Issue                                │ Recommendation                 │
├──────────┼──────────────────────┼──────────────────────────────────────┼────────────────────────────────┤
│ HIGH     │ Concurrency          │ Success rate drops to 94.7% at 1000  │ Increase connection pool size  │
│ MEDIUM   │ Swarm Coordination   │ Coordination overhead is 23% at 100  │ Use mesh/hierarchical topology │
│ HIGH     │ Memory Management    │ Total memory leaked: 52.3 MB         │ Implement aggressive GC        │
└──────────┴──────────────────────┴──────────────────────────────────────┴────────────────────────────────┘
```

## 📁 Output Files

All results are saved to `/workspaces/neural-trader/tests/benchmarks/results/`:

```
results/
├── function-perf-2025-11-15T10-30-00.json
├── scalability-2025-11-15T10-35-00.json
├── gpu-comparison-2025-11-15T10-40-00.json
└── performance-report-1700056800000.html
```

### JSON Schema

```json
{
  "timestamp": "2025-11-15T10:30:00.000Z",
  "system": {
    "node": "v18.17.0",
    "platform": "linux",
    "arch": "x64"
  },
  "results": {
    "Trading Operations": [
      {
        "name": "quickAnalysis",
        "hz": 80,
        "mean": 0.0125,
        "deviation": 0.00023,
        "samples": 50,
        "opsPerSec": "80",
        "meanMs": "12.5000",
        "rme": "1.8%",
        "memoryMB": "2.30"
      }
    ]
  }
}
```

## 🎯 Performance Targets

### Production SLAs

| Metric | Target | Benchmark |
|--------|--------|-----------|
| P95 Latency | <50ms | function-performance |
| P99 Latency | <100ms | function-performance |
| Throughput | >10K ops/min | scalability |
| Success Rate | >99.9% | scalability |
| GPU Speedup | >2x | gpu-comparison |
| Memory Leaks | <5MB/1K ops | scalability |

### Running SLA Tests

```bash
# Test against SLA targets
node tests/benchmarks/run-all.js --export-json

# Check results
node scripts/check-sla-compliance.js tests/benchmarks/results/latest.json
```

## 🔧 Optimization Workflow

### 1. Baseline Measurement
```bash
# Before optimization
node --expose-gc tests/benchmarks/run-all.js --export-json
cp tests/benchmarks/results/latest.json baseline.json
```

### 2. Implement Optimization
Make your performance improvements...

### 3. Re-Benchmark
```bash
# After optimization
node --expose-gc tests/benchmarks/run-all.js --export-json
cp tests/benchmarks/results/latest.json optimized.json
```

### 4. Compare Results
```bash
# Compare before/after
node scripts/compare-benchmarks.js baseline.json optimized.json
```

## 📊 Interpreting Bottlenecks

### Connection Pool Exhaustion
**Symptom**: Success rate <95% at high concurrency
**Fix**: Increase `maxConnections` in configuration

### Memory Leaks
**Symptom**: Memory growth >10MB after repeated operations
**Fix**: Add garbage collection, implement object pooling

### Coordination Overhead
**Symptom**: Swarm efficiency <80%
**Fix**: Use star/hierarchical topology, enable batching

### GPU Under-Utilization
**Symptom**: GPU utilization <60%
**Fix**: Increase batch sizes, pipeline operations

## 🐛 Troubleshooting

### Benchmark Hangs
```bash
# Increase timeout
NODE_OPTIONS="--max-old-space-size=4096" node tests/benchmarks/run-all.js
```

### Out of Memory
```bash
# Reduce sample count
# Edit benchmark files: BENCHMARK_CONFIG.minSamples = 10
node --max-old-space-size=8192 tests/benchmarks/run-all.js
```

### GPU Not Detected
```bash
# Check GPU availability
nvidia-smi

# Skip GPU tests
node tests/benchmarks/run-all.js --skip-gpu
```

### Inconsistent Results
```bash
# Ensure no other processes running
# Close applications
# Run with garbage collection
node --expose-gc tests/benchmarks/run-all.js
```

## 📚 Advanced Usage

### Custom Benchmark Configuration

Edit benchmark files to customize:

```javascript
// tests/benchmarks/function-performance.benchmark.js
const BENCHMARK_CONFIG = {
  minSamples: 50,    // Minimum benchmark iterations
  maxTime: 5,        // Maximum time per benchmark (seconds)
  initCount: 10,     // Warm-up iterations
};
```

### Programmatic Usage

```javascript
const { runAllBenchmarks } = require('./tests/benchmarks/function-performance.benchmark');
const { runScalabilityBenchmarks } = require('./tests/benchmarks/scalability.benchmark');

async function customBenchmark() {
  // Run specific benchmarks
  const functionResults = await runAllBenchmarks();
  const scalabilityResults = await runScalabilityBenchmarks();

  // Process results
  console.log('Function performance:', functionResults);
  console.log('Scalability:', scalabilityResults);
}
```

### CI/CD Integration

```yaml
# .github/workflows/benchmarks.yml
- name: Run Performance Benchmarks
  run: |
    node --expose-gc tests/benchmarks/run-all.js --export-json
    node scripts/check-performance-regression.js
```

## 🤝 Contributing

To add new benchmarks:

1. **Create test case** in appropriate suite
2. **Follow naming convention**: `categoryName_testCase`
3. **Add GPU variant** if applicable: `testCase_CPU`, `testCase_GPU`
4. **Update documentation** in performance-analysis.md
5. **Run full suite** to ensure no regressions

## 📖 Related Documentation

- [Performance Analysis Guide](../../docs/reviews/performance-analysis.md)
- [Optimization Best Practices](../../docs/performance/optimization-guide.md)
- [GPU Acceleration Guide](../../docs/performance/gpu-acceleration.md)

## 🆘 Support

**Issues**: https://github.com/your-org/neural-trader/issues
**Documentation**: https://docs.neural-trader.io/benchmarks
**Discord**: https://discord.gg/neural-trader

---

**Last Updated**: 2025-11-15
**Version**: 2.1.0
**Maintainer**: Performance Team
