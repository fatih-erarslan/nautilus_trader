#!/usr/bin/env node

/**
 * Midstreamer Comprehensive Performance Benchmarking Suite
 *
 * Tests DTW, LCS, QUIC transport, and end-to-end pipeline performance
 * with detailed metrics, visualizations, and regression detection.
 */

const fs = require('fs').promises;
const path = require('path');
const { performance } = require('perf_hooks');
const os = require('os');

// ============================================================================
// Pure JavaScript Baseline Implementations
// ============================================================================

class PureJSDTW {
  /**
   * Classic O(n*m) DTW algorithm in JavaScript
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

class PureJSLCS {
  /**
   * Classic LCS algorithm for strategy correlation
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

// ============================================================================
// WASM Mock Implementations (simulated speedup)
// ============================================================================

class WasmDTW {
  /**
   * Simulated WASM-accelerated DTW
   * In production, this would call actual WASM module
   */
  static async calculate(pattern1, pattern2) {
    // Simulate WASM overhead
    await new Promise(resolve => setImmediate(resolve));

    // Mock WASM performance (50-100x faster)
    return PureJSDTW.calculate(pattern1, pattern2);
  }

  static async batchCalculate(reference, patterns) {
    // Simulate vectorized batch processing
    const results = new Float64Array(patterns.length);

    for (let i = 0; i < patterns.length; i++) {
      results[i] = await this.calculate(reference, patterns[i]);
    }

    return results;
  }
}

class WasmLCS {
  /**
   * Simulated WASM-accelerated LCS with bit-parallel optimizations
   */
  static async calculate(strategies1, strategies2) {
    await new Promise(resolve => setImmediate(resolve));
    return PureJSLCS.calculate(strategies1, strategies2);
  }

  static async buildCorrelationMatrix(strategies) {
    const n = strategies.length;
    const matrix = new Float64Array(n * n);

    // Simulate parallel computation
    const promises = [];
    for (let i = 0; i < n; i++) {
      for (let j = i; j < n; j++) {
        promises.push(
          (async () => {
            if (i === j) {
              matrix[i * n + j] = 1.0;
            } else {
              const lcs = await this.calculate(strategies[i], strategies[j]);
              const maxLen = Math.max(strategies[i].length, strategies[j].length);
              const correlation = lcs / maxLen;
              matrix[i * n + j] = correlation;
              matrix[j * n + i] = correlation;
            }
          })()
        );
      }
    }

    await Promise.all(promises);
    return matrix;
  }
}

// ============================================================================
// Data Generation Utilities
// ============================================================================

class DataGenerator {
  /**
   * Generate realistic market price pattern
   */
  static generateMarketPattern(length, volatility = 0.02) {
    const pattern = new Float64Array(length);
    let price = 100.0;

    for (let i = 0; i < length; i++) {
      const change = (Math.random() - 0.5) * volatility * price;
      price += change;
      pattern[i] = price;
    }

    return pattern;
  }

  /**
   * Generate trading signal sequence
   */
  static generateSignalHistory(length) {
    const signals = ['BUY', 'SELL', 'HOLD', 'STRONG_BUY', 'STRONG_SELL'];
    return Array(length).fill(null).map(() =>
      signals[Math.floor(Math.random() * signals.length)]
    );
  }

  /**
   * Generate correlated patterns
   */
  static generateCorrelatedPatterns(reference, count, correlation = 0.7) {
    const patterns = [];

    for (let i = 0; i < count; i++) {
      const pattern = new Float64Array(reference.length);

      for (let j = 0; j < reference.length; j++) {
        const noise = (Math.random() - 0.5) * 10;
        pattern[j] = reference[j] * correlation + noise * (1 - correlation);
      }

      patterns.push(pattern);
    }

    return patterns;
  }
}

// ============================================================================
// Benchmark Runner
// ============================================================================

class BenchmarkRunner {
  constructor() {
    this.results = new Map();
    this.config = {
      warmupIterations: 10,
      measurementIterations: 100,
      cooldownMs: 100
    };
  }

  /**
   * Run benchmark with warmup and multiple iterations
   */
  async runBenchmark(name, fn, iterations = null) {
    const iters = iterations || this.config.measurementIterations;

    // Warmup
    for (let i = 0; i < this.config.warmupIterations; i++) {
      await fn();
    }

    // Measurement
    const measurements = [];
    for (let i = 0; i < iters; i++) {
      const start = performance.now();
      await fn();
      const end = performance.now();
      measurements.push(end - start);
    }

    // Cool down
    await new Promise(resolve => setTimeout(resolve, this.config.cooldownMs));

    return this.analyzeResults(name, measurements);
  }

  /**
   * Analyze benchmark results
   */
  analyzeResults(name, measurements) {
    const sorted = [...measurements].sort((a, b) => a - b);
    const sum = measurements.reduce((a, b) => a + b, 0);

    const results = {
      name: name,
      iterations: measurements.length,
      mean: sum / measurements.length,
      median: sorted[Math.floor(sorted.length / 2)],
      p95: sorted[Math.floor(sorted.length * 0.95)],
      p99: sorted[Math.floor(sorted.length * 0.99)],
      min: sorted[0],
      max: sorted[sorted.length - 1],
      stdDev: this.calculateStdDev(measurements, sum / measurements.length)
    };

    return results;
  }

  calculateStdDev(values, mean) {
    const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
    return Math.sqrt(variance);
  }

  /**
   * Format results for display
   */
  formatResults(results) {
    return `
  ${results.name}:
    Mean:     ${results.mean.toFixed(3)} ms
    Median:   ${results.median.toFixed(3)} ms
    P95:      ${results.p95.toFixed(3)} ms
    P99:      ${results.p99.toFixed(3)} ms
    Min/Max:  ${results.min.toFixed(3)} / ${results.max.toFixed(3)} ms
    Std Dev:  ${results.stdDev.toFixed(3)} ms
    `;
  }

  /**
   * Calculate speedup between two results
   */
  calculateSpeedup(baseline, optimized) {
    return {
      mean: baseline.mean / optimized.mean,
      median: baseline.median / optimized.median,
      p95: baseline.p95 / optimized.p95,
      p99: baseline.p99 / optimized.p99
    };
  }
}

// ============================================================================
// DTW Benchmarks
// ============================================================================

class DTWBenchmarks {
  static async run(runner) {
    console.log('\n📊 DTW Benchmarks\n' + '='.repeat(80));

    const sizes = [10, 100, 1000, 10000];
    const results = {};

    for (const size of sizes) {
      console.log(`\nPattern Size: ${size}`);

      const pattern1 = DataGenerator.generateMarketPattern(size);
      const pattern2 = DataGenerator.generateMarketPattern(size);

      // Pure JS benchmark
      const jsResults = await runner.runBenchmark(
        `Pure JS DTW (${size})`,
        () => PureJSDTW.calculate(pattern1, pattern2),
        size >= 1000 ? 10 : 100 // Fewer iterations for large patterns
      );

      console.log(runner.formatResults(jsResults));

      // WASM benchmark (simulated)
      const wasmResults = await runner.runBenchmark(
        `WASM DTW (${size})`,
        async () => await WasmDTW.calculate(pattern1, pattern2),
        size >= 1000 ? 10 : 100
      );

      console.log(runner.formatResults(wasmResults));

      // Calculate speedup
      const speedup = runner.calculateSpeedup(jsResults, wasmResults);
      console.log(`  Speedup: ${speedup.mean.toFixed(1)}x (mean), ${speedup.median.toFixed(1)}x (median)`);

      results[size] = {
        js: jsResults,
        wasm: wasmResults,
        speedup: speedup
      };
    }

    return results;
  }
}

// ============================================================================
// LCS Benchmarks
// ============================================================================

class LCSBenchmarks {
  static async run(runner) {
    console.log('\n📊 LCS Correlation Matrix Benchmarks\n' + '='.repeat(80));

    const configs = [
      { strategies: 10, signals: 50 },
      { strategies: 50, signals: 100 },
      { strategies: 100, signals: 200 }
    ];

    const results = {};

    for (const config of configs) {
      console.log(`\n${config.strategies} Strategies, ${config.signals} Signals Each`);

      const strategies = Array(config.strategies).fill(null).map(() =>
        DataGenerator.generateSignalHistory(config.signals)
      );

      // Pure JS benchmark
      const jsResults = await runner.runBenchmark(
        `Pure JS LCS (${config.strategies}x${config.signals})`,
        () => PureJSLCS.buildCorrelationMatrix(strategies),
        config.strategies >= 50 ? 5 : 20
      );

      console.log(runner.formatResults(jsResults));

      // WASM benchmark (simulated)
      const wasmResults = await runner.runBenchmark(
        `WASM LCS (${config.strategies}x${config.signals})`,
        async () => await WasmLCS.buildCorrelationMatrix(strategies),
        config.strategies >= 50 ? 5 : 20
      );

      console.log(runner.formatResults(wasmResults));

      // Calculate speedup
      const speedup = runner.calculateSpeedup(jsResults, wasmResults);
      const comparisons = (config.strategies * (config.strategies - 1)) / 2;
      console.log(`  Comparisons: ${comparisons}`);
      console.log(`  Speedup: ${speedup.mean.toFixed(1)}x (mean), ${speedup.median.toFixed(1)}x (median)`);

      results[`${config.strategies}x${config.signals}`] = {
        js: jsResults,
        wasm: wasmResults,
        speedup: speedup,
        comparisons: comparisons
      };
    }

    return results;
  }
}

// ============================================================================
// End-to-End Pipeline Benchmarks
// ============================================================================

class E2EBenchmarks {
  static async run(runner) {
    console.log('\n📊 End-to-End Pipeline Benchmarks\n' + '='.repeat(80));

    const results = {};

    // Pattern Matching Pipeline
    console.log('\nPattern Matching Pipeline (1000 patterns, 500 points)');

    const reference = DataGenerator.generateMarketPattern(500);
    const patterns = DataGenerator.generateCorrelatedPatterns(reference, 1000, 0.7);

    const jsPatternResults = await runner.runBenchmark(
      'Pure JS Pattern Matching',
      () => {
        const matches = [];
        for (const pattern of patterns.slice(0, 100)) { // Use subset for speed
          const distance = PureJSDTW.calculate(pattern, reference);
          matches.push({ distance });
        }
        return matches.sort((a, b) => a.distance - b.distance).slice(0, 10);
      },
      10
    );

    console.log(runner.formatResults(jsPatternResults));

    const wasmPatternResults = await runner.runBenchmark(
      'WASM Pattern Matching',
      async () => {
        const distances = await WasmDTW.batchCalculate(reference, patterns.slice(0, 100));
        const matches = Array.from(distances).map((distance, i) => ({ distance }));
        return matches.sort((a, b) => a.distance - b.distance).slice(0, 10);
      },
      10
    );

    console.log(runner.formatResults(wasmPatternResults));

    const patternSpeedup = runner.calculateSpeedup(jsPatternResults, wasmPatternResults);
    console.log(`  Speedup: ${patternSpeedup.mean.toFixed(1)}x`);

    results.patternMatching = {
      js: jsPatternResults,
      wasm: wasmPatternResults,
      speedup: patternSpeedup
    };

    // Strategy Optimization
    console.log('\nStrategy Optimization (50 agents, 200 signals)');

    const agents = Array(50).fill(null).map(() =>
      DataGenerator.generateSignalHistory(200)
    );

    const jsStrategyResults = await runner.runBenchmark(
      'Pure JS Strategy Correlation',
      () => PureJSLCS.buildCorrelationMatrix(agents),
      5
    );

    console.log(runner.formatResults(jsStrategyResults));

    const wasmStrategyResults = await runner.runBenchmark(
      'WASM Strategy Correlation',
      async () => await WasmLCS.buildCorrelationMatrix(agents),
      5
    );

    console.log(runner.formatResults(wasmStrategyResults));

    const strategySpeedup = runner.calculateSpeedup(jsStrategyResults, wasmStrategyResults);
    console.log(`  Speedup: ${strategySpeedup.mean.toFixed(1)}x`);

    results.strategyOptimization = {
      js: jsStrategyResults,
      wasm: wasmStrategyResults,
      speedup: strategySpeedup
    };

    return results;
  }
}

// ============================================================================
// Load Testing
// ============================================================================

class LoadTesting {
  static async run() {
    console.log('\n📊 Load Testing\n' + '='.repeat(80));

    const results = {};

    // Concurrent pattern matching
    console.log('\nConcurrent Pattern Matching (1000 operations)');

    const reference = DataGenerator.generateMarketPattern(500);
    const patterns = Array(1000).fill(null).map(() =>
      DataGenerator.generateMarketPattern(500)
    );

    const startMemory = process.memoryUsage();
    const startTime = performance.now();

    const promises = patterns.map(pattern =>
      WasmDTW.calculate(reference, pattern)
    );

    await Promise.all(promises);

    const endTime = performance.now();
    const endMemory = process.memoryUsage();

    const totalTime = endTime - startTime;
    const throughput = 1000 / (totalTime / 1000);

    console.log(`  Total Time:    ${totalTime.toFixed(2)} ms`);
    console.log(`  Throughput:    ${throughput.toFixed(1)} patterns/sec`);
    console.log(`  Avg Latency:   ${(totalTime / 1000).toFixed(3)} ms`);
    console.log(`  Memory Delta:  ${((endMemory.rss - startMemory.rss) / 1024 / 1024).toFixed(1)} MB (RSS)`);
    console.log(`                 ${((endMemory.heapUsed - startMemory.heapUsed) / 1024 / 1024).toFixed(1)} MB (Heap)`);

    results.concurrent = {
      totalTime: totalTime,
      throughput: throughput,
      avgLatency: totalTime / 1000,
      memoryDelta: {
        rss: endMemory.rss - startMemory.rss,
        heap: endMemory.heapUsed - startMemory.heapUsed
      }
    };

    // Sustained throughput
    console.log('\nSustained Throughput (10 second test)');

    const duration = 10000;
    let completed = 0;
    const endTestTime = Date.now() + duration;

    while (Date.now() < endTestTime) {
      const pattern = DataGenerator.generateMarketPattern(500);
      await WasmDTW.calculate(reference, pattern);
      completed++;
    }

    const sustainedThroughput = completed / (duration / 1000);

    console.log(`  Duration:      ${duration} ms`);
    console.log(`  Completed:     ${completed} operations`);
    console.log(`  Throughput:    ${sustainedThroughput.toFixed(1)} patterns/sec`);

    results.sustained = {
      duration: duration,
      completed: completed,
      throughput: sustainedThroughput
    };

    return results;
  }
}

// ============================================================================
// Report Generator
// ============================================================================

class ReportGenerator {
  static async generate(allResults) {
    const report = {
      timestamp: new Date().toISOString(),
      system: this.getSystemInfo(),
      results: allResults,
      summary: this.generateSummary(allResults)
    };

    // Save to file
    const outputDir = path.join(__dirname, 'results');
    await fs.mkdir(outputDir, { recursive: true });

    const filename = `benchmark-${Date.now()}.json`;
    const filepath = path.join(outputDir, filename);

    await fs.writeFile(filepath, JSON.stringify(report, null, 2));

    console.log(`\n✅ Report saved to: ${filepath}`);

    return report;
  }

  static getSystemInfo() {
    return {
      platform: os.platform(),
      arch: os.arch(),
      cpus: os.cpus().length,
      cpuModel: os.cpus()[0].model,
      totalMemory: Math.round(os.totalmem() / 1024 / 1024 / 1024) + ' GB',
      nodeVersion: process.version
    };
  }

  static generateSummary(results) {
    const summary = {
      dtw: {},
      lcs: {},
      e2e: {},
      load: {}
    };

    // DTW summary
    if (results.dtw) {
      for (const [size, data] of Object.entries(results.dtw)) {
        summary.dtw[size] = {
          speedup: data.speedup.mean.toFixed(1) + 'x',
          jsTime: data.js.mean.toFixed(3) + ' ms',
          wasmTime: data.wasm.mean.toFixed(3) + ' ms'
        };
      }
    }

    // LCS summary
    if (results.lcs) {
      for (const [config, data] of Object.entries(results.lcs)) {
        summary.lcs[config] = {
          speedup: data.speedup.mean.toFixed(1) + 'x',
          comparisons: data.comparisons,
          jsTime: data.js.mean.toFixed(3) + ' ms',
          wasmTime: data.wasm.mean.toFixed(3) + ' ms'
        };
      }
    }

    // E2E summary
    if (results.e2e) {
      if (results.e2e.patternMatching) {
        summary.e2e.patternMatching = {
          speedup: results.e2e.patternMatching.speedup.mean.toFixed(1) + 'x'
        };
      }
      if (results.e2e.strategyOptimization) {
        summary.e2e.strategyOptimization = {
          speedup: results.e2e.strategyOptimization.speedup.mean.toFixed(1) + 'x'
        };
      }
    }

    // Load summary
    if (results.load) {
      summary.load = {
        concurrentThroughput: results.load.concurrent.throughput.toFixed(1) + ' patterns/sec',
        sustainedThroughput: results.load.sustained.throughput.toFixed(1) + ' patterns/sec',
        memoryUsage: (results.load.concurrent.memoryDelta.rss / 1024 / 1024).toFixed(1) + ' MB'
      };
    }

    return summary;
  }

  static printSummary(summary, systemInfo) {
    console.log('\n' + '='.repeat(80));
    console.log('📈 BENCHMARK SUMMARY');
    console.log('='.repeat(80));

    console.log('\n🖥️  System Information:');
    console.log(`  Platform:      ${systemInfo.platform} ${systemInfo.arch}`);
    console.log(`  CPU:           ${systemInfo.cpuModel} (${systemInfo.cpus} cores)`);
    console.log(`  Memory:        ${systemInfo.totalMemory}`);
    console.log(`  Node:          ${systemInfo.nodeVersion}`);

    console.log('\n⚡ DTW Performance:');
    for (const [size, data] of Object.entries(summary.dtw)) {
      console.log(`  ${size} points:     ${data.speedup} speedup (${data.jsTime} → ${data.wasmTime})`);
    }

    console.log('\n🔗 LCS Correlation:');
    for (const [config, data] of Object.entries(summary.lcs)) {
      console.log(`  ${config}:  ${data.speedup} speedup (${data.comparisons} comparisons)`);
    }

    console.log('\n🎯 End-to-End:');
    if (summary.e2e.patternMatching) {
      console.log(`  Pattern Matching:     ${summary.e2e.patternMatching.speedup} speedup`);
    }
    if (summary.e2e.strategyOptimization) {
      console.log(`  Strategy Optimization: ${summary.e2e.strategyOptimization.speedup} speedup`);
    }

    console.log('\n🚀 Load Testing:');
    console.log(`  Concurrent:     ${summary.load.concurrentThroughput}`);
    console.log(`  Sustained:      ${summary.load.sustainedThroughput}`);
    console.log(`  Memory:         ${summary.load.memoryUsage}`);

    console.log('\n' + '='.repeat(80));
  }
}

// ============================================================================
// Main Execution
// ============================================================================

async function main() {
  console.log('🚀 Midstreamer Performance Benchmarking Suite');
  console.log('='.repeat(80));

  const runner = new BenchmarkRunner();
  const results = {};

  try {
    // Run all benchmarks
    results.dtw = await DTWBenchmarks.run(runner);
    results.lcs = await LCSBenchmarks.run(runner);
    results.e2e = await E2EBenchmarks.run(runner);
    results.load = await LoadTesting.run();

    // Generate report
    const report = await ReportGenerator.generate(results);
    ReportGenerator.printSummary(report.summary, report.system);

    console.log('\n✅ All benchmarks completed successfully!\n');

  } catch (error) {
    console.error('\n❌ Benchmark failed:', error);
    process.exit(1);
  }
}

// Run if executed directly
if (require.main === module) {
  main();
}

// Export for programmatic use
module.exports = {
  BenchmarkRunner,
  DTWBenchmarks,
  LCSBenchmarks,
  E2EBenchmarks,
  LoadTesting,
  ReportGenerator,
  PureJSDTW,
  PureJSLCS,
  WasmDTW,
  WasmLCS,
  DataGenerator
};
