/**
 * Benchmark comparison runner for A/B testing
 */

import { BenchmarkRunner } from './benchmark-runner';
import { BenchmarkConfig, BenchmarkResult, ComparisonResult } from '../types';

export class ComparisonRunner {
  /**
   * Compare two implementations
   */
  async compare(
    baselineFn: () => Promise<any>,
    currentFn: () => Promise<any>,
    config: BenchmarkConfig
  ): Promise<ComparisonResult> {
    const baselineRunner = new BenchmarkRunner({
      ...config,
      name: `${config.name}-baseline`
    });

    const currentRunner = new BenchmarkRunner({
      ...config,
      name: `${config.name}-current`
    });

    // Run benchmarks
    const baseline = await baselineRunner.run(baselineFn);
    const current = await currentRunner.run(currentFn);

    // Calculate improvement
    const improvement = ((baseline.mean - current.mean) / baseline.mean) * 100;
    const faster = current.mean < baseline.mean;

    // Statistical significance test (Welch's t-test)
    const pValue = this.welchTTest(
      this.generateSamples(baseline),
      this.generateSamples(current)
    );
    const significant = pValue < 0.05;

    // Memory comparison
    const memoryDelta = current.memory.heapUsed - baseline.memory.heapUsed;

    return {
      baseline,
      current,
      improvement,
      pValue,
      significant,
      faster,
      memoryDelta
    };
  }

  /**
   * Compare JS vs NAPI-RS implementation
   */
  async compareNativeBinding(
    jsFn: () => Promise<any>,
    rustFn: () => Promise<any>,
    config: BenchmarkConfig
  ): Promise<ComparisonResult> {
    return this.compare(jsFn, rustFn, {
      ...config,
      name: config.name || 'JS vs Rust'
    });
  }

  /**
   * Compare multiple implementations
   */
  async compareMultiple(
    implementations: Array<{ name: string; fn: () => Promise<any> }>,
    config: BenchmarkConfig
  ): Promise<Map<string, BenchmarkResult>> {
    const results = new Map<string, BenchmarkResult>();

    for (const impl of implementations) {
      const runner = new BenchmarkRunner({
        ...config,
        name: impl.name
      });

      const result = await runner.run(impl.fn);
      results.set(impl.name, result);
    }

    return results;
  }

  private welchTTest(sample1: number[], sample2: number[]): number {
    const mean1 = sample1.reduce((a, b) => a + b, 0) / sample1.length;
    const mean2 = sample2.reduce((a, b) => a + b, 0) / sample2.length;

    const variance1 = sample1.reduce((sum, val) =>
      sum + Math.pow(val - mean1, 2), 0
    ) / (sample1.length - 1);

    const variance2 = sample2.reduce((sum, val) =>
      sum + Math.pow(val - mean2, 2), 0
    ) / (sample2.length - 1);

    const tStat = (mean1 - mean2) / Math.sqrt(
      variance1 / sample1.length + variance2 / sample2.length
    );

    // Approximate degrees of freedom
    const df = Math.pow(
      variance1 / sample1.length + variance2 / sample2.length,
      2
    ) / (
      Math.pow(variance1 / sample1.length, 2) / (sample1.length - 1) +
      Math.pow(variance2 / sample2.length, 2) / (sample2.length - 1)
    );

    // Simplified p-value calculation (assumes normal distribution)
    const pValue = 2 * (1 - this.normalCDF(Math.abs(tStat)));

    return pValue;
  }

  private normalCDF(z: number): number {
    // Approximation of standard normal CDF
    const t = 1 / (1 + 0.2316419 * Math.abs(z));
    const d = 0.3989423 * Math.exp(-z * z / 2);
    const prob = d * t * (
      0.3193815 +
      t * (-0.3565638 +
        t * (1.781478 +
          t * (-1.821256 +
            t * 1.330274)))
    );

    return z > 0 ? 1 - prob : prob;
  }

  private generateSamples(result: BenchmarkResult): number[] {
    // Generate approximate sample distribution from statistics
    const samples: number[] = [];
    const count = result.iterations;

    for (let i = 0; i < count; i++) {
      // Box-Muller transform for normal distribution
      const u1 = Math.random();
      const u2 = Math.random();
      const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      const sample = result.mean + z * result.stdDev;
      samples.push(Math.max(0, sample));
    }

    return samples;
  }
}
