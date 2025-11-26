/**
 * Memory leak detection
 */

import { MemoryStats } from '../types';

export interface LeakDetectionResult {
  leaked: boolean;
  leakRate: number; // bytes per iteration
  confidence: number; // 0-1
  samples: MemoryStats[];
  analysis: {
    heapGrowth: number;
    externalGrowth: number;
    arrayBufferGrowth: number;
    trend: 'increasing' | 'stable' | 'decreasing';
  };
}

export class MemoryLeakDetector {
  private samples: MemoryStats[] = [];
  private readonly minSamples = 10;

  /**
   * Add memory sample
   */
  addSample(stats: MemoryStats): void {
    this.samples.push(stats);
  }

  /**
   * Detect memory leaks
   */
  detect(threshold = 1024): LeakDetectionResult {
    if (this.samples.length < this.minSamples) {
      throw new Error(`Need at least ${this.minSamples} samples for leak detection`);
    }

    // Calculate memory growth rate
    const heapUsed = this.samples.map(s => s.heapUsed);
    const { slope: heapGrowth, rSquared: heapR2 } = this.linearRegression(heapUsed);

    const external = this.samples.map(s => s.external);
    const { slope: externalGrowth } = this.linearRegression(external);

    const arrayBuffers = this.samples.map(s => s.arrayBuffers);
    const { slope: arrayBufferGrowth } = this.linearRegression(arrayBuffers);

    // Determine if leaking
    const leaked = heapGrowth > threshold && heapR2 > 0.7;
    const leakRate = heapGrowth;
    const confidence = Math.min(1, heapR2);

    // Determine trend
    let trend: 'increasing' | 'stable' | 'decreasing';
    if (heapGrowth > threshold) {
      trend = 'increasing';
    } else if (heapGrowth < -threshold) {
      trend = 'decreasing';
    } else {
      trend = 'stable';
    }

    return {
      leaked,
      leakRate,
      confidence,
      samples: [...this.samples],
      analysis: {
        heapGrowth,
        externalGrowth,
        arrayBufferGrowth,
        trend
      }
    };
  }

  /**
   * Run leak detection test
   */
  async test(
    fn: () => Promise<any>,
    iterations = 100,
    cooldown = 100
  ): Promise<LeakDetectionResult> {
    this.clear();

    // Force initial GC
    if (global.gc) {
      global.gc();
      await this.sleep(cooldown);
    }

    // Run iterations and collect samples
    for (let i = 0; i < iterations; i++) {
      await fn();

      // Collect memory sample
      const stats = this.captureMemory();
      this.addSample(stats);

      // Periodic GC to avoid false positives
      if (i % 10 === 0 && global.gc) {
        global.gc();
        await this.sleep(cooldown);
      }
    }

    return this.detect();
  }

  /**
   * Clear samples
   */
  clear(): void {
    this.samples = [];
  }

  /**
   * Get growth prediction
   */
  predictMemoryUsage(iterations: number): {
    predicted: number;
    timeToOOM: number; // iterations until out of memory
  } {
    const result = this.detect();

    if (!result.leaked || result.leakRate <= 0) {
      return {
        predicted: this.samples[this.samples.length - 1].heapUsed,
        timeToOOM: Infinity
      };
    }

    const currentMemory = this.samples[this.samples.length - 1].heapUsed;
    const predicted = currentMemory + result.leakRate * iterations;

    // Estimate time to OOM (assuming 4GB heap limit)
    const heapLimit = 4 * 1024 * 1024 * 1024;
    const remainingMemory = heapLimit - currentMemory;
    const timeToOOM = Math.floor(remainingMemory / result.leakRate);

    return { predicted, timeToOOM };
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

  private linearRegression(values: number[]): {
    slope: number;
    intercept: number;
    rSquared: number;
  } {
    const n = values.length;
    const x = Array.from({ length: n }, (_, i) => i);

    const sumX = x.reduce((a, b) => a + b, 0);
    const sumY = values.reduce((a, b) => a + b, 0);
    const sumXY = x.reduce((sum, xi, i) => sum + xi * values[i], 0);
    const sumX2 = x.reduce((sum, xi) => sum + xi * xi, 0);
    const sumY2 = values.reduce((sum, yi) => sum + yi * yi, 0);

    const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
    const intercept = (sumY - slope * sumX) / n;

    const yMean = sumY / n;
    const ssTotal = values.reduce((sum, yi) => sum + Math.pow(yi - yMean, 2), 0);
    const ssResidual = values.reduce((sum, yi, i) =>
      sum + Math.pow(yi - (slope * x[i] + intercept), 2), 0
    );
    const rSquared = 1 - ssResidual / ssTotal;

    return { slope, intercept, rSquared };
  }

  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}
