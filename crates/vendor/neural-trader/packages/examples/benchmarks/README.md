# @neural-trader/benchmarks

Comprehensive benchmarking framework with statistical analysis, performance tracking, and regression detection.

## Features

- 📊 **Statistical Analysis**: Comprehensive statistical metrics (mean, median, percentiles, std dev)
- 🔍 **Regression Detection**: Automatic performance regression detection with configurable thresholds
- 📈 **Performance Tracking**: Historical performance tracking with AgentDB integration
- 💾 **Memory Leak Detection**: Identify memory leaks with statistical confidence
- ⚡ **Comparison Framework**: Compare implementations (JS vs Rust, baseline vs current)
- 📝 **Multiple Reporters**: Console, JSON, and HTML report generation
- 🎯 **Bottleneck Detection**: Profile and identify performance bottlenecks

## Installation

```bash
npm install --save-dev @neural-trader/benchmarks
```

## Quick Start

```typescript
import { BenchmarkRunner } from '@neural-trader/benchmarks';

const runner = new BenchmarkRunner({
  name: 'My Benchmark',
  iterations: 100,
  warmupIterations: 10
});

const result = await runner.run(async () => {
  // Your code to benchmark
  await expensiveOperation();
});

console.log(`Mean: ${result.mean.toFixed(2)}ms`);
console.log(`Throughput: ${result.throughput.toFixed(2)} ops/sec`);
```

## CLI Usage

### Run Benchmarks

```bash
# Run specific benchmark suite
neural-bench run ./benchmarks/my-suite.js

# Run with custom iterations
neural-bench run ./benchmarks/my-suite.js -i 1000

# Export to JSON
neural-bench run ./benchmarks/my-suite.js -o json -f results.json

# Export to HTML
neural-bench run ./benchmarks/my-suite.js -o html -f report.html
```

### Compare Implementations

```bash
# Compare two implementations
neural-bench compare ./baseline.js ./current.js -i 100
```

### Detect Memory Leaks

```bash
# Run memory leak detection
neural-bench memory-leak ./benchmarks/my-suite.js -i 100 -t 1024
```

### View History

```bash
# View benchmark history (requires AgentDB)
neural-bench history my-benchmark -l 20
```

## Benchmark Suite Format

```javascript
// benchmarks/my-suite.js
export const benchmarks = [
  {
    name: 'Array Processing',
    fn: async () => {
      const arr = Array.from({ length: 10000 }, (_, i) => i);
      return arr.map(x => x * 2).filter(x => x > 5000);
    }
  },
  {
    name: 'Object Creation',
    fn: async () => {
      const objects = [];
      for (let i = 0; i < 1000; i++) {
        objects.push({ id: i, value: Math.random() });
      }
      return objects;
    }
  }
];

export default benchmarks;
```

## API

### BenchmarkRunner

```typescript
import { BenchmarkRunner } from '@neural-trader/benchmarks';

const runner = new BenchmarkRunner({
  name: 'My Benchmark',
  iterations: 100,
  warmupIterations: 10,
  timeout: 60000,
  memoryLimit: 512 * 1024 * 1024
});

const result = await runner.run(async () => {
  // Code to benchmark
});

// Result includes:
// - mean, median, min, max
// - stdDev, p95, p99
// - throughput (ops/sec)
// - memory statistics
```

### ComparisonRunner

```typescript
import { ComparisonRunner } from '@neural-trader/benchmarks';

const runner = new ComparisonRunner();

const comparison = await runner.compare(
  async () => jsImplementation(),
  async () => rustImplementation(),
  {
    name: 'JS vs Rust',
    iterations: 100
  }
);

console.log(`Improvement: ${comparison.improvement.toFixed(2)}%`);
console.log(`Significant: ${comparison.significant}`);
console.log(`P-value: ${comparison.pValue.toFixed(4)}`);
```

### Statistical Analyzer

```typescript
import { StatisticalAnalyzer } from '@neural-trader/benchmarks';

const analyzer = new StatisticalAnalyzer();

// Analyze samples
const summary = analyzer.analyze(samples);

console.log(`Mean: ${summary.mean}`);
console.log(`Median: ${summary.median}`);
console.log(`Std Dev: ${summary.stdDev}`);
console.log(`Outliers: ${summary.outliers.length}`);

// Calculate confidence interval
const [lower, upper] = analyzer.confidenceInterval(samples, 0.95);

// Trend analysis
const trend = analyzer.analyzeTrend(historicalResults);
console.log(`Trend: ${trend.trend}`); // 'improving', 'degrading', or 'stable'

// Detect anomalies
const anomalies = analyzer.detectAnomalies(results, 2); // 2 std devs
```

### Regression Detector

```typescript
import { RegressionDetector } from '@neural-trader/benchmarks';

const detector = new RegressionDetector({
  mean: 10,        // 10% degradation threshold
  p95: 15,         // 15% for p95
  throughput: 10,  // 10% decrease threshold
  memory: 20       // 20% increase threshold
});

const alerts = detector.detect(baseline, current);

for (const alert of alerts) {
  console.log(`[${alert.severity}] ${alert.benchmark}`);
  console.log(`Metric: ${alert.metric}`);
  console.log(`Threshold: ${alert.threshold}%`);
  console.log(`Actual: ${alert.actual}%`);
}

// Calculate regression score
const score = detector.calculateRegressionScore(alerts);
console.log(`Regression score: ${score}/100`);
```

### Memory Leak Detector

```typescript
import { MemoryLeakDetector } from '@neural-trader/benchmarks';

const detector = new MemoryLeakDetector();

const result = await detector.test(
  async () => {
    // Code to test for memory leaks
    const arr = [];
    for (let i = 0; i < 1000; i++) {
      arr.push(new Array(1000));
    }
  },
  100  // iterations
);

if (result.leaked) {
  console.log(`Memory leak detected!`);
  console.log(`Leak rate: ${result.leakRate} bytes/iteration`);
  console.log(`Confidence: ${result.confidence * 100}%`);

  const prediction = detector.predictMemoryUsage(1000);
  console.log(`Predicted memory: ${prediction.predicted} bytes`);
  console.log(`Time to OOM: ${prediction.timeToOOM} iterations`);
}
```

### Performance Bottleneck Detector

```typescript
import { PerformanceBottleneckDetector } from '@neural-trader/benchmarks';

const detector = new PerformanceBottleneckDetector();

// Profile functions
await detector.profile('processData', async () => {
  await processData();
});

await detector.profile('analyzeResults', async () => {
  await analyzeResults();
});

// Generate report
const report = detector.analyze();

console.log(`Overall score: ${report.overallScore}/100`);

for (const hotspot of report.hotspots) {
  console.log(`[${hotspot.severity}] ${hotspot.name}`);
  console.log(`  Duration: ${hotspot.duration.toFixed(2)}ms`);
  console.log(`  Percentage: ${hotspot.percentage.toFixed(1)}%`);
  console.log(`  Calls: ${hotspot.callCount}`);
}

for (const recommendation of report.recommendations) {
  console.log(`💡 ${recommendation}`);
}
```

### AgentDB History

```typescript
import { AgentDBHistory } from '@neural-trader/benchmarks';
import AgentDB from 'agentdb';

const db = new AgentDB();
const history = new AgentDBHistory(db);

// Store result
await history.store(benchmarkResult);

// Get history
const perfHistory = await history.getHistory('my-benchmark', 100);

console.log(`Trend: ${perfHistory.trend}`);
console.log(`First run: ${new Date(perfHistory.firstRun)}`);
console.log(`Last run: ${new Date(perfHistory.lastRun)}`);

// Get baseline (median of last 10 runs)
const baseline = await history.getBaseline('my-benchmark', 10);

// Find similar benchmarks
const similar = await history.findSimilar(currentResult, 5);

// Export history
const exported = await history.export('my-benchmark');
```

## Reporters

### Console Reporter

```typescript
import { ConsoleReporter } from '@neural-trader/benchmarks';

const reporter = new ConsoleReporter();

// Report single result
reporter.report(result);

// Report comparison
reporter.reportComparison(comparison);

// Report regressions
reporter.reportRegressions(alerts);

// Report table of results
reporter.reportTable(results);
```

### JSON Reporter

```typescript
import { JSONReporter } from '@neural-trader/benchmarks';

const reporter = new JSONReporter();

const report = reporter.generate(results, {
  comparisons,
  regressions
});

await reporter.writeToFile(report, 'results.json');

// Read back
const loaded = await reporter.readFromFile('results.json');
```

### HTML Reporter

```typescript
import { HTMLReporter } from '@neural-trader/benchmarks';

const reporter = new HTMLReporter();

const html = reporter.generate(results, {
  title: 'Performance Report',
  comparisons
});

await reporter.writeToFile(html, 'report.html');
```

## Best Practices

### 1. Warmup Iterations

Always use warmup iterations to let JIT compile and optimize:

```typescript
const runner = new BenchmarkRunner({
  name: 'My Benchmark',
  iterations: 100,
  warmupIterations: 10  // Important!
});
```

### 2. Run with --expose-gc

For accurate memory measurements:

```bash
node --expose-gc your-benchmark.js
```

### 3. Consistent Environment

Run benchmarks in consistent environments:

```typescript
// Close other applications
// Disable CPU throttling
// Use same hardware/OS
```

### 4. Statistical Significance

Check statistical significance when comparing:

```typescript
if (comparison.significant && comparison.pValue < 0.05) {
  console.log('Performance change is statistically significant');
}
```

### 5. Track History

Use AgentDB to track performance over time:

```typescript
const history = new AgentDBHistory(db);

await history.store(result);

const perfHistory = await history.getHistory('my-benchmark');

if (perfHistory.trend === 'degrading') {
  console.warn('Performance is degrading over time!');
}
```

### 6. Set Thresholds

Configure regression thresholds based on your requirements:

```typescript
const detector = new RegressionDetector({
  mean: 5,        // Alert if 5% slower
  p95: 10,        // Alert if p95 is 10% slower
  throughput: 5,  // Alert if throughput drops 5%
  memory: 15      // Alert if memory increases 15%
});
```

## Integration with CI/CD

### GitHub Actions

```yaml
name: Performance Tests

on: [push, pull_request]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-node@v2
      - run: npm install
      - run: npm run benchmark
      - uses: actions/upload-artifact@v2
        with:
          name: benchmark-results
          path: results/
```

### Regression Alerts

```typescript
const alerts = detector.detect(baseline, current);

if (alerts.length > 0) {
  console.error('Performance regressions detected!');
  reporter.reportRegressions(alerts);
  process.exit(1); // Fail CI build
}
```

## TypeScript Support

Full TypeScript support with type definitions:

```typescript
import type {
  BenchmarkConfig,
  BenchmarkResult,
  ComparisonResult,
  RegressionAlert,
  PerformanceHistory,
  StatisticalSummary,
  MemoryStats
} from '@neural-trader/benchmarks';
```

## Contributing

See the main neural-trader repository for contribution guidelines.

## License

MIT OR Apache-2.0
