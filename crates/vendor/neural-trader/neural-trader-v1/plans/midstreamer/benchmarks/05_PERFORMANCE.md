# Midstreamer Performance Benchmarking Suite

## Overview

Comprehensive performance benchmarking framework for evaluating midstreamer's WASM-accelerated pattern matching, QUIC transport, and distributed coordination capabilities.

## 1. DTW (Dynamic Time Warping) Benchmarks

### 1.1 Pure JavaScript Baseline Implementation

```javascript
class PureJSDTW {
  /**
   * Classic O(n*m) DTW algorithm in JavaScript
   * @param {Float64Array} pattern1 - First time series
   * @param {Float64Array} pattern2 - Second time series
   * @returns {number} DTW distance
   */
  static calculate(pattern1, pattern2) {
    const n = pattern1.length;
    const m = pattern2.length;

    // Cost matrix with initialization
    const dtw = Array(n + 1).fill(null).map(() =>
      Array(m + 1).fill(Infinity)
    );
    dtw[0][0] = 0;

    // Fill cost matrix
    for (let i = 1; i <= n; i++) {
      for (let j = 1; j <= m; j++) {
        const cost = Math.abs(pattern1[i - 1] - pattern2[j - 1]);
        dtw[i][j] = cost + Math.min(
          dtw[i - 1][j],      // insertion
          dtw[i][j - 1],      // deletion
          dtw[i - 1][j - 1]   // match
        );
      }
    }

    return dtw[n][m];
  }

  /**
   * Optimized DTW with Sakoe-Chiba band constraint
   * @param {Float64Array} pattern1 - First time series
   * @param {Float64Array} pattern2 - Second time series
   * @param {number} windowSize - Warping window size
   * @returns {number} DTW distance
   */
  static calculateWithWindow(pattern1, pattern2, windowSize) {
    const n = pattern1.length;
    const m = pattern2.length;
    const w = Math.max(windowSize, Math.abs(n - m));

    const dtw = Array(n + 1).fill(null).map(() =>
      Array(m + 1).fill(Infinity)
    );
    dtw[0][0] = 0;

    for (let i = 1; i <= n; i++) {
      const jStart = Math.max(1, i - w);
      const jEnd = Math.min(m, i + w);

      for (let j = jStart; j <= jEnd; j++) {
        const cost = Math.abs(pattern1[i - 1] - pattern2[j - 1]);
        dtw[i][j] = cost + Math.min(
          dtw[i - 1][j],
          dtw[i][j - 1],
          dtw[i - 1][j - 1]
        );
      }
    }

    return dtw[n][m];
  }
}
```

### 1.2 WASM Implementation (via midstreamer)

```javascript
class WasmDTW {
  /**
   * WASM-accelerated DTW with SIMD optimizations
   * @param {Float64Array} pattern1 - First time series
   * @param {Float64Array} pattern2 - Second time series
   * @returns {Promise<number>} DTW distance
   */
  static async calculate(pattern1, pattern2) {
    const { StreamProcessor } = await import('@midstreamer/core');

    const processor = new StreamProcessor({
      mode: 'pattern_matching',
      algorithm: 'dtw',
      simd: true
    });

    return processor.computeDTW(pattern1, pattern2);
  }

  /**
   * Batch DTW computation (vectorized)
   * @param {Float64Array} reference - Reference pattern
   * @param {Float64Array[]} patterns - Array of patterns to compare
   * @returns {Promise<Float64Array>} Array of DTW distances
   */
  static async batchCalculate(reference, patterns) {
    const { StreamProcessor } = await import('@midstreamer/core');

    const processor = new StreamProcessor({
      mode: 'pattern_matching',
      algorithm: 'dtw_batch',
      simd: true,
      parallelism: 'auto'
    });

    return processor.batchDTW(reference, patterns);
  }
}
```

### 1.3 Benchmark Configuration

| Pattern Size | Iterations | Expected Speedup | Memory Limit |
|--------------|------------|------------------|--------------|
| 10           | 100,000    | 5-10x            | 1 MB         |
| 100          | 10,000     | 20-50x           | 10 MB        |
| 1,000        | 1,000      | 50-100x          | 100 MB       |
| 10,000       | 100        | 80-150x          | 1 GB         |

### 1.4 Expected Results

```
Pattern Size: 10
├─ Pure JS:    0.012 ms/op  (baseline)
├─ WASM:       0.0015 ms/op (8x faster)
└─ WASM+SIMD:  0.0008 ms/op (15x faster)

Pattern Size: 100
├─ Pure JS:    1.2 ms/op    (baseline)
├─ WASM:       0.04 ms/op   (30x faster)
└─ WASM+SIMD:  0.018 ms/op  (67x faster)

Pattern Size: 1,000
├─ Pure JS:    120 ms/op    (baseline)
├─ WASM:       2.1 ms/op    (57x faster)
└─ WASM+SIMD:  1.2 ms/op    (100x faster)

Pattern Size: 10,000
├─ Pure JS:    12,000 ms/op (baseline)
├─ WASM:       95 ms/op     (126x faster)
└─ WASM+SIMD:  80 ms/op     (150x faster)
```

## 2. LCS (Longest Common Subsequence) Benchmarks

### 2.1 Pure JavaScript Baseline

```javascript
class PureJSLCS {
  /**
   * Classic LCS algorithm for strategy correlation
   * @param {string[]} strategies1 - First strategy sequence
   * @param {string[]} strategies2 - Second strategy sequence
   * @returns {number} LCS length (correlation score)
   */
  static calculate(strategies1, strategies2) {
    const m = strategies1.length;
    const n = strategies2.length;

    const dp = Array(m + 1).fill(null).map(() =>
      Array(n + 1).fill(0)
    );

    for (let i = 1; i <= m; i++) {
      for (let j = 1; j <= n; j++) {
        if (strategies1[i - 1] === strategies2[j - 1]) {
          dp[i][j] = dp[i - 1][j - 1] + 1;
        } else {
          dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
        }
      }
    }

    return dp[m][n];
  }

  /**
   * Build correlation matrix for N strategies
   * @param {string[][]} strategies - Array of strategy sequences
   * @returns {number[][]} Correlation matrix
   */
  static buildCorrelationMatrix(strategies) {
    const n = strategies.length;
    const matrix = Array(n).fill(null).map(() => Array(n).fill(0));

    for (let i = 0; i < n; i++) {
      matrix[i][i] = 1.0; // Perfect self-correlation

      for (let j = i + 1; j < n; j++) {
        const lcs = this.calculate(strategies[i], strategies[j]);
        const maxLen = Math.max(strategies[i].length, strategies[j].length);
        const correlation = lcs / maxLen;

        matrix[i][j] = correlation;
        matrix[j][i] = correlation; // Symmetric
      }
    }

    return matrix;
  }
}
```

### 2.2 WASM LCS Implementation

```javascript
class WasmLCS {
  /**
   * WASM-accelerated LCS with bit-parallel optimizations
   * @param {string[]} strategies1 - First strategy sequence
   * @param {string[]} strategies2 - Second strategy sequence
   * @returns {Promise<number>} LCS length
   */
  static async calculate(strategies1, strategies2) {
    const { StreamProcessor } = await import('@midstreamer/core');

    const processor = new StreamProcessor({
      mode: 'correlation',
      algorithm: 'lcs',
      bitParallel: true
    });

    return processor.computeLCS(strategies1, strategies2);
  }

  /**
   * Parallel correlation matrix computation
   * @param {string[][]} strategies - Array of strategy sequences
   * @returns {Promise<Float64Array>} Flattened correlation matrix
   */
  static async buildCorrelationMatrix(strategies) {
    const { StreamProcessor } = await import('@midstreamer/core');

    const processor = new StreamProcessor({
      mode: 'correlation_matrix',
      algorithm: 'parallel_lcs',
      workers: 'auto'
    });

    return processor.correlationMatrix(strategies);
  }
}
```

### 2.3 Strategy Correlation Benchmark Configuration

| Strategy Count | Sequence Length | Matrix Operations | Expected Speedup |
|----------------|-----------------|-------------------|------------------|
| 10             | 50              | 45                | 30-60x           |
| 50             | 100             | 1,225             | 60-100x          |
| 100            | 200             | 4,950             | 80-129x          |

### 2.4 Expected Results

```
10 Strategies (50 signals each):
├─ Pure JS:    45 ms       (45 comparisons)
├─ WASM:       1.2 ms      (38x faster)
└─ WASM+Para:  0.8 ms      (56x faster)

50 Strategies (100 signals each):
├─ Pure JS:    1,850 ms    (1,225 comparisons)
├─ WASM:       28 ms       (66x faster)
└─ WASM+Para:  18 ms       (103x faster)

100 Strategies (200 signals each):
├─ Pure JS:    14,200 ms   (4,950 comparisons)
├─ WASM:       165 ms      (86x faster)
└─ WASM+Para:  110 ms      (129x faster)
```

## 3. QUIC vs WebSocket Transport Benchmarks

### 3.1 Message Round-Trip Latency

```javascript
class TransportBenchmark {
  /**
   * Measure round-trip time for different message sizes
   */
  static async measureRTT(transport, messageSizes) {
    const results = {};

    for (const size of messageSizes) {
      const message = new Uint8Array(size);
      const measurements = [];

      for (let i = 0; i < 1000; i++) {
        const start = performance.now();
        await transport.sendAndWait(message);
        const end = performance.now();
        measurements.push(end - start);
      }

      results[size] = {
        mean: this.mean(measurements),
        p50: this.percentile(measurements, 50),
        p95: this.percentile(measurements, 95),
        p99: this.percentile(measurements, 99)
      };
    }

    return results;
  }

  /**
   * Stream multiplexing overhead test
   */
  static async measureMultiplexing(transport, streamCount) {
    const streams = Array(streamCount).fill(null).map((_, i) =>
      transport.createStream(`stream-${i}`)
    );

    const start = performance.now();

    await Promise.all(streams.map(stream =>
      stream.send(new Uint8Array(1024))
    ));

    const end = performance.now();

    return {
      totalTime: end - start,
      perStream: (end - start) / streamCount
    };
  }

  /**
   * Connection resumption (0-RTT) test
   */
  static async measure0RTT(transport) {
    // Establish initial connection
    await transport.connect();
    const sessionTicket = transport.getSessionTicket();
    await transport.disconnect();

    // Measure 0-RTT resumption
    const start = performance.now();
    await transport.resume(sessionTicket);
    const end = performance.now();

    return end - start;
  }
}
```

### 3.2 Expected Results

```
Round-Trip Latency (1 KB message):
├─ WebSocket:  5.2 ms   (p50), 8.7 ms (p95), 12.1 ms (p99)
├─ QUIC:       0.8 ms   (p50), 1.3 ms (p95), 2.1 ms (p99)
└─ Speedup:    6.5x (p50), 6.7x (p95), 5.8x (p99)

Stream Multiplexing (100 streams):
├─ WebSocket:  450 ms total, 4.5 ms/stream
├─ QUIC:       85 ms total, 0.85 ms/stream
└─ Speedup:    5.3x

Connection Resumption:
├─ WebSocket:  45-60 ms  (full TLS handshake)
├─ QUIC 0-RTT: 0.5-1 ms  (immediate resumption)
└─ Speedup:    60-90x
```

## 4. End-to-End Pipeline Benchmarks

### 4.1 Pattern Matching Pipeline

```javascript
class PatternMatchingBenchmark {
  /**
   * Complete pattern matching workflow
   */
  static async benchmarkPipeline(mode) {
    const pipeline = {
      'current': async (patterns, reference) => {
        // Current pure JS implementation
        const matches = [];
        for (const pattern of patterns) {
          const distance = PureJSDTW.calculate(pattern, reference);
          matches.push({ pattern, distance });
        }
        return matches.sort((a, b) => a.distance - b.distance).slice(0, 10);
      },

      'wasm': async (patterns, reference) => {
        // WASM-accelerated implementation
        const { StreamProcessor } = await import('@midstreamer/core');
        const processor = new StreamProcessor({
          mode: 'pattern_matching',
          topK: 10,
          simd: true
        });

        return processor.findTopMatches(reference, patterns);
      }
    };

    return pipeline[mode];
  }

  /**
   * Test with realistic market data
   */
  static async testWithMarketData(patternCount, patternLength) {
    const reference = this.generateMarketPattern(patternLength);
    const patterns = Array(patternCount).fill(null).map(() =>
      this.generateMarketPattern(patternLength)
    );

    const currentPipeline = await this.benchmarkPipeline('current');
    const wasmPipeline = await this.benchmarkPipeline('wasm');

    const currentStart = performance.now();
    await currentPipeline(patterns, reference);
    const currentTime = performance.now() - currentStart;

    const wasmStart = performance.now();
    await wasmPipeline(patterns, reference);
    const wasmTime = performance.now() - wasmStart;

    return {
      current: currentTime,
      wasm: wasmTime,
      speedup: currentTime / wasmTime
    };
  }
}
```

### 4.2 Strategy Optimization Benchmark

```javascript
class StrategyOptimizationBenchmark {
  /**
   * Multi-agent strategy correlation and optimization
   */
  static async benchmarkOptimization(agentCount, signalHistory) {
    const agents = Array(agentCount).fill(null).map((_, i) => ({
      id: `agent-${i}`,
      signals: this.generateSignalHistory(signalHistory)
    }));

    // Current implementation
    const currentStart = performance.now();
    const currentMatrix = PureJSLCS.buildCorrelationMatrix(
      agents.map(a => a.signals)
    );
    const currentTime = performance.now() - currentStart;

    // WASM implementation
    const wasmStart = performance.now();
    const wasmMatrix = await WasmLCS.buildCorrelationMatrix(
      agents.map(a => a.signals)
    );
    const wasmTime = performance.now() - wasmStart;

    return {
      current: currentTime,
      wasm: wasmTime,
      speedup: currentTime / wasmTime,
      matrixSize: agentCount * agentCount
    };
  }
}
```

### 4.3 Multi-Timeframe Alignment

```javascript
class MultiTimeframeBenchmark {
  /**
   * Align patterns across multiple timeframes
   */
  static async benchmarkAlignment(timeframes, patternLength) {
    const patterns = timeframes.map(tf =>
      this.generateTimeframePattern(tf, patternLength)
    );

    // Current: Sequential DTW alignment
    const currentStart = performance.now();
    for (let i = 0; i < patterns.length - 1; i++) {
      PureJSDTW.calculate(patterns[i], patterns[i + 1]);
    }
    const currentTime = performance.now() - currentStart;

    // WASM: Parallel alignment
    const { StreamProcessor } = await import('@midstreamer/core');
    const processor = new StreamProcessor({
      mode: 'multi_timeframe',
      parallel: true
    });

    const wasmStart = performance.now();
    await processor.alignTimeframes(patterns);
    const wasmTime = performance.now() - wasmStart;

    return {
      current: currentTime,
      wasm: wasmTime,
      speedup: currentTime / wasmTime,
      timeframeCount: timeframes.length
    };
  }
}
```

### 4.4 Expected End-to-End Results

```
Pattern Matching (1000 patterns, length 500):
├─ Current:   2,400 ms
├─ WASM:      42 ms
└─ Speedup:   57x

Strategy Optimization (50 agents, 200 signals):
├─ Current:   1,850 ms
├─ WASM:      18 ms
└─ Speedup:   103x

Multi-Timeframe Alignment (5 timeframes, 1000 points):
├─ Current:   600 ms
├─ WASM:      8.5 ms
└─ Speedup:   71x
```

## 5. Load Testing & Scalability

### 5.1 Concurrent Pattern Matching

```javascript
class LoadTest {
  /**
   * Test with 1000+ concurrent pattern matches
   */
  static async testConcurrentMatches(concurrency, patternSize) {
    const { StreamProcessor } = await import('@midstreamer/core');

    const processor = new StreamProcessor({
      mode: 'high_concurrency',
      maxConcurrent: concurrency,
      queueSize: concurrency * 2
    });

    const reference = this.generatePattern(patternSize);
    const requests = Array(concurrency).fill(null).map((_, i) => ({
      id: i,
      pattern: this.generatePattern(patternSize)
    }));

    const start = performance.now();
    const startMemory = process.memoryUsage();

    const results = await Promise.all(
      requests.map(req => processor.matchPattern(reference, req.pattern))
    );

    const end = performance.now();
    const endMemory = process.memoryUsage();

    return {
      totalTime: end - start,
      throughput: concurrency / ((end - start) / 1000), // patterns/sec
      avgLatency: (end - start) / concurrency,
      memoryDelta: {
        rss: endMemory.rss - startMemory.rss,
        heapUsed: endMemory.heapUsed - startMemory.heapUsed
      }
    };
  }

  /**
   * Sustained throughput test
   */
  static async testSustainedThroughput(duration, patternSize) {
    const { StreamProcessor } = await import('@midstreamer/core');

    const processor = new StreamProcessor({
      mode: 'streaming',
      bufferSize: 10000
    });

    let completed = 0;
    const reference = this.generatePattern(patternSize);
    const endTime = Date.now() + duration;

    while (Date.now() < endTime) {
      const pattern = this.generatePattern(patternSize);
      await processor.matchPattern(reference, pattern);
      completed++;
    }

    return {
      duration: duration,
      completed: completed,
      throughput: completed / (duration / 1000) // patterns/sec
    };
  }
}
```

### 5.2 Memory Usage Under Load

```javascript
class MemoryBenchmark {
  /**
   * Profile memory usage patterns
   */
  static async profileMemory(operationCount, patternSize) {
    const snapshots = [];
    const { StreamProcessor } = await import('@midstreamer/core');

    const processor = new StreamProcessor({
      mode: 'pattern_matching',
      memoryLimit: '1GB'
    });

    const reference = this.generatePattern(patternSize);

    for (let i = 0; i < operationCount; i++) {
      if (i % 100 === 0) {
        snapshots.push({
          iteration: i,
          memory: process.memoryUsage(),
          timestamp: Date.now()
        });
      }

      const pattern = this.generatePattern(patternSize);
      await processor.matchPattern(reference, pattern);
    }

    return {
      snapshots: snapshots,
      peakRSS: Math.max(...snapshots.map(s => s.memory.rss)),
      peakHeap: Math.max(...snapshots.map(s => s.memory.heapUsed)),
      growthRate: this.calculateGrowthRate(snapshots)
    };
  }
}
```

### 5.3 Expected Load Test Results

```
Concurrent Pattern Matching (1000 concurrent, 500-point patterns):
├─ Total Time:     2,100 ms
├─ Throughput:     476 patterns/sec
├─ Avg Latency:    2.1 ms
├─ Memory Delta:   145 MB (RSS), 82 MB (Heap)
└─ Success Rate:   100%

Sustained Throughput (60 sec, 500-point patterns):
├─ Completed:      28,500 operations
├─ Throughput:     475 patterns/sec
├─ Memory Stable:  Yes (< 5% growth)
└─ CPU Usage:      45-65% (8-core system)

Memory Profile (10,000 operations, 1000-point patterns):
├─ Peak RSS:       892 MB
├─ Peak Heap:      456 MB
├─ Growth Rate:    0.8 MB/1000 ops
└─ GC Frequency:   Every 2,500 ops
```

## 6. Benchmark Execution Framework

### 6.1 Automated Benchmark Runner

```javascript
class BenchmarkRunner {
  constructor() {
    this.results = new Map();
    this.config = {
      warmupIterations: 100,
      measurementIterations: 1000,
      cooldownMs: 500
    };
  }

  /**
   * Run all benchmarks in sequence
   */
  async runAll() {
    console.log('🚀 Starting Midstreamer Performance Benchmarks\n');

    // DTW Benchmarks
    await this.runBenchmarkSuite('DTW', async () => {
      await this.benchmarkDTW(10);
      await this.benchmarkDTW(100);
      await this.benchmarkDTW(1000);
      await this.benchmarkDTW(10000);
    });

    // LCS Benchmarks
    await this.runBenchmarkSuite('LCS', async () => {
      await this.benchmarkLCS(10, 50);
      await this.benchmarkLCS(50, 100);
      await this.benchmarkLCS(100, 200);
    });

    // Transport Benchmarks
    await this.runBenchmarkSuite('Transport', async () => {
      await this.benchmarkTransport('websocket');
      await this.benchmarkTransport('quic');
    });

    // End-to-End Benchmarks
    await this.runBenchmarkSuite('E2E', async () => {
      await this.benchmarkPipeline(1000, 500);
      await this.benchmarkOptimization(50, 200);
      await this.benchmarkAlignment([1, 5, 15, 60, 240], 1000);
    });

    // Load Tests
    await this.runBenchmarkSuite('Load', async () => {
      await this.loadTest(1000, 500);
      await this.sustainedThroughput(60000, 500);
      await this.memoryProfile(10000, 1000);
    });

    return this.generateReport();
  }

  /**
   * Generate comprehensive report
   */
  generateReport() {
    const report = {
      timestamp: new Date().toISOString(),
      system: this.getSystemInfo(),
      results: Object.fromEntries(this.results),
      summary: this.generateSummary()
    };

    return report;
  }

  /**
   * Create visualizations
   */
  async generateVisualizations(report) {
    // Generate charts using Chart.js or similar
    const charts = {
      dtwSpeedup: this.createSpeedupChart('DTW', report.results.DTW),
      lcsScaling: this.createScalingChart('LCS', report.results.LCS),
      transportLatency: this.createLatencyChart(report.results.Transport),
      throughput: this.createThroughputChart(report.results.Load)
    };

    return charts;
  }
}
```

## 7. Performance Targets & Success Criteria

### 7.1 Target Metrics

| Metric | Current | Target | Stretch Goal |
|--------|---------|--------|--------------|
| DTW (1000 pts) | 120 ms | 2 ms | 1 ms |
| LCS (100 strategies) | 14,200 ms | 165 ms | 110 ms |
| QUIC RTT | N/A | < 1 ms | < 0.5 ms |
| Pattern Throughput | 50/sec | 400/sec | 500/sec |
| Memory (1000 ops) | N/A | < 200 MB | < 150 MB |

### 7.2 Success Criteria

✅ **Must Have:**
- DTW: 50x speedup for 1000-point patterns
- LCS: 60x speedup for 50 strategies
- QUIC: < 1ms RTT for 1KB messages
- Throughput: > 400 patterns/sec sustained

✅ **Should Have:**
- DTW: 100x speedup for 10000-point patterns
- LCS: 100x speedup for 100 strategies
- Memory: < 200MB for 1000 concurrent operations
- 0-RTT: < 1ms connection resumption

✅ **Nice to Have:**
- 150x speedup on all operations
- 500+ patterns/sec throughput
- < 0.5ms QUIC latency
- Linear scaling to 10,000 concurrent operations

## 8. Continuous Benchmarking

### 8.1 CI/CD Integration

```yaml
# .github/workflows/benchmark.yml
name: Performance Benchmarks
on:
  push:
    branches: [main, develop]
  pull_request:
  schedule:
    - cron: '0 0 * * 0' # Weekly

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run Benchmarks
        run: npm run benchmark:all
      - name: Compare with Baseline
        run: npm run benchmark:compare
      - name: Upload Results
        uses: actions/upload-artifact@v3
        with:
          name: benchmark-results
          path: benchmarks/results/
```

### 8.2 Performance Regression Detection

```javascript
class RegressionDetector {
  /**
   * Compare current results with baseline
   */
  static detectRegressions(current, baseline, threshold = 0.1) {
    const regressions = [];

    for (const [metric, value] of Object.entries(current)) {
      const baselineValue = baseline[metric];
      if (!baselineValue) continue;

      const change = (value - baselineValue) / baselineValue;

      if (change > threshold) {
        regressions.push({
          metric: metric,
          current: value,
          baseline: baselineValue,
          regression: `${(change * 100).toFixed(1)}%`
        });
      }
    }

    return regressions;
  }
}
```

## Conclusion

This comprehensive benchmarking suite provides:

1. **Granular Performance Metrics**: DTW, LCS, transport, and end-to-end measurements
2. **Realistic Load Testing**: 1000+ concurrent operations with memory profiling
3. **Automated Execution**: CI/CD integration and regression detection
4. **Clear Success Criteria**: Quantifiable targets and stretch goals
5. **Continuous Monitoring**: Weekly benchmarks and performance tracking

Expected overall improvements:
- **10-150x** speedup on core algorithms
- **< 1ms** QUIC latency vs 5-10ms WebSocket
- **400-500** patterns/sec sustained throughput
- **< 200MB** memory footprint under load
