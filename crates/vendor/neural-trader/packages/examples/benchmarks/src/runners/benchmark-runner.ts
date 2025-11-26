/**
 * Core benchmark runner
 */

import { BenchmarkConfig, BenchmarkResult, MemoryStats } from '../types';

export class BenchmarkRunner {
  private config: Required<BenchmarkConfig>;

  constructor(config: BenchmarkConfig) {
    this.config = {
      iterations: config.iterations || 100,
      warmupIterations: config.warmupIterations || 10,
      timeout: config.timeout || 60000,
      memoryLimit: config.memoryLimit || 512 * 1024 * 1024, // 512MB
      compareBaseline: config.compareBaseline ?? false,
      ...config
    };
  }

  /**
   * Run benchmark
   */
  async run(fn: () => Promise<any>): Promise<BenchmarkResult> {
    // Warmup
    await this.warmup(fn);

    // Force GC if available
    if (global.gc) {
      global.gc();
    }

    // Collect samples
    const samples: number[] = [];
    const memorySamples: MemoryStats[] = [];
    const startTime = Date.now();

    for (let i = 0; i < this.config.iterations; i++) {
      const memBefore = this.captureMemory();
      const iterStart = performance.now();

      try {
        await Promise.race([
          fn(),
          this.timeout(this.config.timeout)
        ]);
      } catch (error) {
        throw new Error(`Benchmark failed at iteration ${i}: ${error}`);
      }

      const iterDuration = performance.now() - iterStart;
      samples.push(iterDuration);

      const memAfter = this.captureMemory();
      memorySamples.push(memAfter);

      // Check memory limit
      if (memAfter.heapUsed > this.config.memoryLimit) {
        throw new Error(`Memory limit exceeded: ${memAfter.heapUsed} > ${this.config.memoryLimit}`);
      }
    }

    const totalDuration = Date.now() - startTime;

    // Calculate statistics
    const stats = this.calculateStatistics(samples);
    const memoryStats = this.aggregateMemory(memorySamples);

    return {
      name: this.config.name,
      iterations: this.config.iterations,
      duration: totalDuration,
      mean: stats.mean,
      median: stats.median,
      stdDev: stats.stdDev,
      min: stats.min,
      max: stats.max,
      p95: stats.p95,
      p99: stats.p99,
      throughput: (this.config.iterations / totalDuration) * 1000, // ops/sec
      memory: memoryStats,
      timestamp: Date.now()
    };
  }

  /**
   * Run multiple benchmarks
   */
  async runSuite(benchmarks: Array<{ name: string; fn: () => Promise<any> }>): Promise<BenchmarkResult[]> {
    const results: BenchmarkResult[] = [];

    for (const bench of benchmarks) {
      const runner = new BenchmarkRunner({
        ...this.config,
        name: bench.name
      });

      const result = await runner.run(bench.fn);
      results.push(result);

      // Cool down between benchmarks
      await this.sleep(1000);
    }

    return results;
  }

  private async warmup(fn: () => Promise<any>): Promise<void> {
    for (let i = 0; i < this.config.warmupIterations; i++) {
      await fn();
    }
  }

  private calculateStatistics(samples: number[]): {
    mean: number;
    median: number;
    stdDev: number;
    min: number;
    max: number;
    p95: number;
    p99: number;
  } {
    const sorted = [...samples].sort((a, b) => a - b);
    const mean = samples.reduce((a, b) => a + b, 0) / samples.length;
    const median = sorted[Math.floor(sorted.length / 2)];

    const variance = samples.reduce((sum, val) =>
      sum + Math.pow(val - mean, 2), 0
    ) / samples.length;
    const stdDev = Math.sqrt(variance);

    const p95Index = Math.floor(sorted.length * 0.95);
    const p99Index = Math.floor(sorted.length * 0.99);

    return {
      mean,
      median,
      stdDev,
      min: sorted[0],
      max: sorted[sorted.length - 1],
      p95: sorted[p95Index],
      p99: sorted[p99Index]
    };
  }

  private captureMemory(): MemoryStats {
    const mem = process.memoryUsage();
    return {
      heapUsed: mem.heapUsed,
      heapTotal: mem.heapTotal,
      external: mem.external,
      rss: mem.rss,
      arrayBuffers: mem.arrayBuffers,
      peak: mem.heapUsed
    };
  }

  private aggregateMemory(samples: MemoryStats[]): MemoryStats {
    const avgHeapUsed = samples.reduce((sum, s) => sum + s.heapUsed, 0) / samples.length;
    const avgHeapTotal = samples.reduce((sum, s) => sum + s.heapTotal, 0) / samples.length;
    const peak = Math.max(...samples.map(s => s.heapUsed));

    return {
      heapUsed: avgHeapUsed,
      heapTotal: avgHeapTotal,
      external: samples[samples.length - 1].external,
      rss: samples[samples.length - 1].rss,
      arrayBuffers: samples[samples.length - 1].arrayBuffers,
      peak
    };
  }

  private timeout(ms: number): Promise<never> {
    return new Promise((_, reject) =>
      setTimeout(() => reject(new Error('Benchmark timeout')), ms)
    );
  }

  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}
