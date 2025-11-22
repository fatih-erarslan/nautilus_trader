/**
 * Performance Profiler for Trading System Testing
 * Provides comprehensive performance monitoring and analysis
 */

const os = require('os');
const { performance, PerformanceObserver } = require('perf_hooks');

class PerformanceProfiler {
  constructor() {
    this.metrics = new Map();
    this.observers = new Map();
    this.cpuMonitors = new Map();
    this.memorySnapshots = [];
    this.latencyHistograms = new Map();
  }

  /**
   * Start CPU monitoring
   */
  startCpuMonitoring() {
    const monitorId = `cpu_${Date.now()}_${Math.random()}`;
    
    const monitor = {
      id: monitorId,
      startTime: Date.now(),
      samples: [],
      interval: null
    };

    // Sample CPU usage every 100ms
    monitor.interval = setInterval(() => {
      const cpuUsage = this.getCurrentCpuUsage();
      monitor.samples.push({
        timestamp: Date.now(),
        usage: cpuUsage
      });
    }, 100);

    this.cpuMonitors.set(monitorId, monitor);
    return monitorId;
  }

  /**
   * Stop CPU monitoring and return statistics
   */
  stopCpuMonitoring(monitorId) {
    const monitor = this.cpuMonitors.get(monitorId);
    if (!monitor) {
      throw new Error(`CPU monitor ${monitorId} not found`);
    }

    clearInterval(monitor.interval);
    this.cpuMonitors.delete(monitorId);

    const samples = monitor.samples.map(s => s.usage);
    
    return {
      duration: Date.now() - monitor.startTime,
      sampleCount: samples.length,
      averageCpuUsage: samples.reduce((a, b) => a + b, 0) / samples.length,
      peakCpuUsage: Math.max(...samples),
      minCpuUsage: Math.min(...samples),
      cpuVariance: this.calculateVariance(samples),
      samples: monitor.samples
    };
  }

  /**
   * Get current CPU usage percentage
   */
  getCurrentCpuUsage() {
    const cpus = os.cpus();
    let totalIdle = 0;
    let totalTick = 0;

    cpus.forEach(cpu => {
      for (let type in cpu.times) {
        totalTick += cpu.times[type];
      }
      totalIdle += cpu.times.idle;
    });

    const idle = totalIdle / cpus.length;
    const total = totalTick / cpus.length;
    const usage = 100 - ~~(100 * idle / total);

    return usage;
  }

  /**
   * Start memory monitoring
   */
  startMemoryMonitoring() {
    const monitorId = `memory_${Date.now()}_${Math.random()}`;
    
    const monitor = {
      id: monitorId,
      startTime: Date.now(),
      startMemory: process.memoryUsage(),
      snapshots: [],
      interval: null
    };

    // Take memory snapshots every 500ms
    monitor.interval = setInterval(() => {
      const memoryUsage = process.memoryUsage();
      monitor.snapshots.push({
        timestamp: Date.now(),
        memory: memoryUsage
      });
    }, 500);

    this.memorySnapshots.push(monitor);
    return monitorId;
  }

  /**
   * Stop memory monitoring and return statistics
   */
  stopMemoryMonitoring(monitorId) {
    const monitorIndex = this.memorySnapshots.findIndex(m => m.id === monitorId);
    if (monitorIndex === -1) {
      throw new Error(`Memory monitor ${monitorId} not found`);
    }

    const monitor = this.memorySnapshots[monitorIndex];
    clearInterval(monitor.interval);
    this.memorySnapshots.splice(monitorIndex, 1);

    const finalMemory = process.memoryUsage();
    const heapUsedSeries = monitor.snapshots.map(s => s.memory.heapUsed);
    
    return {
      duration: Date.now() - monitor.startTime,
      startMemory: monitor.startMemory,
      endMemory: finalMemory,
      memoryDelta: {
        heapUsed: finalMemory.heapUsed - monitor.startMemory.heapUsed,
        heapTotal: finalMemory.heapTotal - monitor.startMemory.heapTotal,
        external: finalMemory.external - monitor.startMemory.external,
        rss: finalMemory.rss - monitor.startMemory.rss
      },
      peakHeapUsed: Math.max(...heapUsedSeries),
      averageHeapUsed: heapUsedSeries.reduce((a, b) => a + b, 0) / heapUsedSeries.length,
      memoryGrowthRate: this.calculateGrowthRate(heapUsedSeries),
      snapshots: monitor.snapshots
    };
  }

  /**
   * Start latency tracking for a specific operation
   */
  startLatencyTracking(operationName) {
    if (!this.latencyHistograms.has(operationName)) {
      this.latencyHistograms.set(operationName, {
        samples: [],
        buckets: new Map(),
        startTime: Date.now()
      });
    }

    return {
      operationName,
      startTime: process.hrtime.bigint()
    };
  }

  /**
   * Record latency measurement
   */
  recordLatency(tracker) {
    const endTime = process.hrtime.bigint();
    const latencyNs = Number(endTime - tracker.startTime);
    const latencyMs = latencyNs / 1_000_000;

    const histogram = this.latencyHistograms.get(tracker.operationName);
    histogram.samples.push(latencyMs);

    // Update histogram buckets
    const bucket = this.getLatencyBucket(latencyMs);
    histogram.buckets.set(bucket, (histogram.buckets.get(bucket) || 0) + 1);

    return latencyMs;
  }

  /**
   * Get latency statistics for an operation
   */
  getLatencyStats(operationName) {
    const histogram = this.latencyHistograms.get(operationName);
    if (!histogram || histogram.samples.length === 0) {
      return null;
    }

    const sortedSamples = [...histogram.samples].sort((a, b) => a - b);
    const count = sortedSamples.length;

    return {
      operationName,
      sampleCount: count,
      duration: Date.now() - histogram.startTime,
      min: sortedSamples[0],
      max: sortedSamples[count - 1],
      mean: sortedSamples.reduce((a, b) => a + b, 0) / count,
      median: this.getPercentile(sortedSamples, 0.5),
      p95: this.getPercentile(sortedSamples, 0.95),
      p99: this.getPercentile(sortedSamples, 0.99),
      p999: this.getPercentile(sortedSamples, 0.999),
      standardDeviation: this.calculateStandardDeviation(sortedSamples),
      throughput: count / ((Date.now() - histogram.startTime) / 1000),
      histogram: Object.fromEntries(histogram.buckets)
    };
  }

  /**
   * Start comprehensive performance monitoring
   */
  startComprehensiveMonitoring(operationName) {
    const cpuId = this.startCpuMonitoring();
    const memoryId = this.startMemoryMonitoring();
    const latencyTracker = this.startLatencyTracking(operationName);

    return {
      operationName,
      cpuId,
      memoryId,
      latencyTracker,
      startTime: Date.now()
    };
  }

  /**
   * Stop comprehensive monitoring and return complete profile
   */
  stopComprehensiveMonitoring(monitor) {
    const cpuStats = this.stopCpuMonitoring(monitor.cpuId);
    const memoryStats = this.stopMemoryMonitoring(monitor.memoryId);
    const latencyMs = this.recordLatency(monitor.latencyTracker);
    const latencyStats = this.getLatencyStats(monitor.operationName);

    return {
      operationName: monitor.operationName,
      duration: Date.now() - monitor.startTime,
      latency: {
        thisOperation: latencyMs,
        statistics: latencyStats
      },
      cpu: cpuStats,
      memory: memoryStats,
      efficiency: {
        operationsPerCpuSecond: latencyStats ? latencyStats.throughput / (cpuStats.averageCpuUsage / 100) : 0,
        memoryEfficiency: memoryStats.memoryDelta.heapUsed / (latencyStats ? latencyStats.sampleCount : 1),
        resourceUtilization: (cpuStats.averageCpuUsage + (memoryStats.memoryDelta.heapUsed / (1024 * 1024))) / 2
      }
    };
  }

  /**
   * Profile a function execution
   */
  async profile(fn, operationName) {
    const monitor = this.startComprehensiveMonitoring(operationName);
    
    try {
      const result = await fn();
      const profile = this.stopComprehensiveMonitoring(monitor);
      
      return {
        result,
        profile
      };
    } catch (error) {
      const profile = this.stopComprehensiveMonitoring(monitor);
      throw new Error(`Operation failed: ${error.message}\nProfile: ${JSON.stringify(profile, null, 2)}`);
    }
  }

  /**
   * Benchmark function performance with multiple runs
   */
  async benchmark(fn, options = {}) {
    const {
      name = 'benchmark',
      iterations = 100,
      warmupIterations = 10,
      timeout = 300000 // 5 minutes
    } = options;

    console.log(`Starting benchmark: ${name} (${iterations} iterations)`);

    // Warmup
    for (let i = 0; i < warmupIterations; i++) {
      await fn();
    }

    // Force garbage collection before benchmark
    if (global.gc) global.gc();

    const monitor = this.startComprehensiveMonitoring(name);
    const iterationLatencies = [];
    const startTime = Date.now();

    for (let i = 0; i < iterations; i++) {
      if (Date.now() - startTime > timeout) {
        throw new Error(`Benchmark timeout after ${i} iterations`);
      }

      const iterationStart = process.hrtime.bigint();
      await fn();
      const iterationEnd = process.hrtime.bigint();
      
      const iterationLatencyMs = Number(iterationEnd - iterationStart) / 1_000_000;
      iterationLatencies.push(iterationLatencyMs);

      // Progress logging
      if (i % Math.max(1, Math.floor(iterations / 10)) === 0) {
        console.log(`Benchmark progress: ${i}/${iterations} (${((i/iterations)*100).toFixed(1)}%)`);
      }
    }

    const profile = this.stopComprehensiveMonitoring(monitor);

    // Calculate iteration statistics
    const sortedLatencies = iterationLatencies.sort((a, b) => a - b);
    
    return {
      name,
      iterations,
      totalDuration: profile.duration,
      iterationStats: {
        min: sortedLatencies[0],
        max: sortedLatencies[sortedLatencies.length - 1],
        mean: sortedLatencies.reduce((a, b) => a + b, 0) / sortedLatencies.length,
        median: this.getPercentile(sortedLatencies, 0.5),
        p95: this.getPercentile(sortedLatencies, 0.95),
        p99: this.getPercentile(sortedLatencies, 0.99),
        standardDeviation: this.calculateStandardDeviation(sortedLatencies)
      },
      throughput: iterations / (profile.duration / 1000),
      systemProfile: profile,
      iterationLatencies
    };
  }

  /**
   * Compare performance between two functions
   */
  async compare(fn1, fn2, options = {}) {
    const {
      name1 = 'function1',
      name2 = 'function2',
      iterations = 100
    } = options;

    console.log(`Comparing performance: ${name1} vs ${name2}`);

    const benchmark1 = await this.benchmark(fn1, { name: name1, iterations });
    const benchmark2 = await this.benchmark(fn2, { name: name2, iterations });

    const comparison = {
      function1: benchmark1,
      function2: benchmark2,
      comparison: {
        throughputRatio: benchmark1.throughput / benchmark2.throughput,
        latencyRatio: benchmark2.iterationStats.mean / benchmark1.iterationStats.mean,
        memoryRatio: benchmark1.systemProfile.memory.memoryDelta.heapUsed / 
                    benchmark2.systemProfile.memory.memoryDelta.heapUsed,
        cpuRatio: benchmark1.systemProfile.cpu.averageCpuUsage / 
                 benchmark2.systemProfile.cpu.averageCpuUsage,
        winner: benchmark1.throughput > benchmark2.throughput ? name1 : name2
      }
    };

    console.log(`Performance Winner: ${comparison.comparison.winner}`);
    console.log(`Throughput Ratio: ${comparison.comparison.throughputRatio.toFixed(2)}x`);
    console.log(`Latency Ratio: ${comparison.comparison.latencyRatio.toFixed(2)}x`);

    return comparison;
  }

  // Utility methods
  getLatencyBucket(latencyMs) {
    if (latencyMs < 1) return '< 1ms';
    if (latencyMs < 5) return '1-5ms';
    if (latencyMs < 10) return '5-10ms';
    if (latencyMs < 50) return '10-50ms';
    if (latencyMs < 100) return '50-100ms';
    if (latencyMs < 500) return '100-500ms';
    if (latencyMs < 1000) return '500ms-1s';
    return '> 1s';
  }

  getPercentile(sortedArray, percentile) {
    const index = Math.ceil(sortedArray.length * percentile) - 1;
    return sortedArray[Math.max(0, index)];
  }

  calculateVariance(values) {
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const variance = values.reduce((sum, value) => sum + Math.pow(value - mean, 2), 0) / values.length;
    return variance;
  }

  calculateStandardDeviation(values) {
    return Math.sqrt(this.calculateVariance(values));
  }

  calculateGrowthRate(series) {
    if (series.length < 2) return 0;
    
    const start = series[0];
    const end = series[series.length - 1];
    return (end - start) / start;
  }

  /**
   * Generate performance report
   */
  generateReport() {
    const report = {
      timestamp: new Date().toISOString(),
      summary: {
        totalOperations: Array.from(this.latencyHistograms.keys()).length,
        activeCpuMonitors: this.cpuMonitors.size,
        activeMemoryMonitors: this.memorySnapshots.length
      },
      operations: {}
    };

    for (const [operationName, histogram] of this.latencyHistograms) {
      report.operations[operationName] = this.getLatencyStats(operationName);
    }

    return report;
  }

  /**
   * Reset all metrics
   */
  reset() {
    // Clear all active monitors
    for (const [id, monitor] of this.cpuMonitors) {
      clearInterval(monitor.interval);
    }
    this.cpuMonitors.clear();

    for (const monitor of this.memorySnapshots) {
      clearInterval(monitor.interval);
    }
    this.memorySnapshots.length = 0;

    this.latencyHistograms.clear();
    this.metrics.clear();
  }
}

module.exports = { PerformanceProfiler };