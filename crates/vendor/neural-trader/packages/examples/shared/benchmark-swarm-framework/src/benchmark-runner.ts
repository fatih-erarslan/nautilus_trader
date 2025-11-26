/**
 * Benchmark runner for performance testing and comparison
 * Tracks metrics and generates comprehensive reports
 */

import { SwarmCoordinator, SwarmConfig, TaskVariation, AgentTask } from './swarm-coordinator';

export interface BenchmarkConfig extends SwarmConfig {
  warmupIterations?: number;
  iterations: number;
  collectGarbage?: boolean;
}

export interface BenchmarkSuite<T = any> {
  name: string;
  description?: string;
  task: AgentTask<T>;
  variations: TaskVariation[];
}

export interface BenchmarkReport {
  suiteName: string;
  timestamp: Date;
  config: BenchmarkConfig;
  results: {
    variationId: string;
    success: boolean;
    stats: {
      mean: number;
      median: number;
      min: number;
      max: number;
      stdDev: number;
      percentile95: number;
      percentile99: number;
    };
    throughput: number; // operations per second
  }[];
  summary: {
    totalDuration: number;
    successRate: number;
    fastestVariation: string;
    slowestVariation: string;
  };
}

export class BenchmarkRunner<T = any> {
  private config: Required<BenchmarkConfig>;
  private reports: BenchmarkReport[] = [];

  constructor(config: BenchmarkConfig) {
    this.config = {
      maxAgents: config.maxAgents,
      topology: config.topology,
      communicationProtocol: config.communicationProtocol,
      timeout: config.timeout || 30000,
      retryAttempts: config.retryAttempts || 3,
      warmupIterations: config.warmupIterations || 3,
      iterations: config.iterations,
      collectGarbage: config.collectGarbage ?? true,
    };
  }

  /**
   * Run benchmark suite
   */
  async runSuite(suite: BenchmarkSuite<T>): Promise<BenchmarkReport> {
    console.log(`\n🏃 Running benchmark suite: ${suite.name}`);
    console.log(`   Variations: ${suite.variations.length}`);
    console.log(`   Iterations: ${this.config.iterations}`);
    console.log(`   Warmup: ${this.config.warmupIterations}`);

    const startTime = Date.now();

    // Warmup phase
    if (this.config.warmupIterations > 0) {
      console.log(`\n🔥 Warming up...`);
      await this.runIterations(suite, this.config.warmupIterations, true);
    }

    // Actual benchmark iterations
    console.log(`\n📊 Running benchmark iterations...`);
    const iterationResults = await this.runIterations(
      suite,
      this.config.iterations,
      false
    );

    // Aggregate results
    const results = this.aggregateResults(iterationResults);

    // Create report
    const report: BenchmarkReport = {
      suiteName: suite.name,
      timestamp: new Date(),
      config: this.config,
      results,
      summary: this.createSummary(results, Date.now() - startTime),
    };

    this.reports.push(report);
    this.printReport(report);

    return report;
  }

  /**
   * Run multiple iterations of the benchmark
   */
  private async runIterations(
    suite: BenchmarkSuite<T>,
    iterations: number,
    isWarmup: boolean
  ): Promise<Map<string, number[]>> {
    const coordinator = new SwarmCoordinator<T>({
      maxAgents: this.config.maxAgents,
      topology: this.config.topology,
      communicationProtocol: this.config.communicationProtocol,
      timeout: this.config.timeout,
      retryAttempts: this.config.retryAttempts,
    });

    const resultsByVariation = new Map<string, number[]>();
    suite.variations.forEach((v) => resultsByVariation.set(v.id, []));

    for (let i = 0; i < iterations; i++) {
      if (this.config.collectGarbage && global.gc) {
        global.gc();
      }

      const results = await coordinator.executeVariations(
        suite.variations,
        suite.task,
        (completed, total) => {
          if (!isWarmup) {
            process.stdout.write(
              `\r   Iteration ${i + 1}/${iterations}: ${completed}/${total} variations`
            );
          }
        }
      );

      results.forEach((result) => {
        if (result.success) {
          const times = resultsByVariation.get(result.variationId);
          times?.push(result.metrics.executionTime);
        }
      });

      if (!isWarmup) {
        process.stdout.write('\n');
      }

      coordinator.clear();
    }

    return resultsByVariation;
  }

  /**
   * Aggregate results and calculate statistics
   */
  private aggregateResults(resultsByVariation: Map<string, number[]>) {
    const results: BenchmarkReport['results'] = [];

    resultsByVariation.forEach((times, variationId) => {
      if (times.length === 0) {
        results.push({
          variationId,
          success: false,
          stats: {
            mean: 0,
            median: 0,
            min: 0,
            max: 0,
            stdDev: 0,
            percentile95: 0,
            percentile99: 0,
          },
          throughput: 0,
        });
        return;
      }

      const sorted = [...times].sort((a, b) => a - b);
      const mean = times.reduce((sum, t) => sum + t, 0) / times.length;
      const variance =
        times.reduce((sum, t) => sum + Math.pow(t - mean, 2), 0) / times.length;
      const stdDev = Math.sqrt(variance);

      results.push({
        variationId,
        success: true,
        stats: {
          mean,
          median: this.percentile(sorted, 50),
          min: sorted[0],
          max: sorted[sorted.length - 1],
          stdDev,
          percentile95: this.percentile(sorted, 95),
          percentile99: this.percentile(sorted, 99),
        },
        throughput: 1000 / mean, // ops/sec
      });
    });

    return results;
  }

  /**
   * Calculate percentile
   */
  private percentile(sorted: number[], p: number): number {
    const index = (p / 100) * (sorted.length - 1);
    const lower = Math.floor(index);
    const upper = Math.ceil(index);
    const weight = index - lower;

    return sorted[lower] * (1 - weight) + sorted[upper] * weight;
  }

  /**
   * Create summary statistics
   */
  private createSummary(
    results: BenchmarkReport['results'],
    totalDuration: number
  ): BenchmarkReport['summary'] {
    const successful = results.filter((r) => r.success);
    const successRate = successful.length / results.length;

    let fastest = successful[0];
    let slowest = successful[0];

    successful.forEach((result) => {
      if (result.stats.mean < fastest.stats.mean) {
        fastest = result;
      }
      if (result.stats.mean > slowest.stats.mean) {
        slowest = result;
      }
    });

    return {
      totalDuration,
      successRate,
      fastestVariation: fastest?.variationId || 'none',
      slowestVariation: slowest?.variationId || 'none',
    };
  }

  /**
   * Print benchmark report
   */
  private printReport(report: BenchmarkReport): void {
    console.log(`\n${'='.repeat(80)}`);
    console.log(`📊 Benchmark Report: ${report.suiteName}`);
    console.log(`${'='.repeat(80)}`);
    console.log(`Timestamp: ${report.timestamp.toISOString()}`);
    console.log(`Total Duration: ${(report.summary.totalDuration / 1000).toFixed(2)}s`);
    console.log(`Success Rate: ${(report.summary.successRate * 100).toFixed(2)}%`);
    console.log(`\nResults by Variation:\n`);

    // Sort by mean execution time
    const sortedResults = [...report.results].sort(
      (a, b) => a.stats.mean - b.stats.mean
    );

    sortedResults.forEach((result, index) => {
      const rank = index + 1;
      const emoji = rank === 1 ? '🥇' : rank === 2 ? '🥈' : rank === 3 ? '🥉' : '  ';

      console.log(`${emoji} ${result.variationId}`);
      console.log(`   Mean: ${result.stats.mean.toFixed(2)}ms`);
      console.log(`   Median: ${result.stats.median.toFixed(2)}ms`);
      console.log(`   Min/Max: ${result.stats.min.toFixed(2)}ms / ${result.stats.max.toFixed(2)}ms`);
      console.log(`   StdDev: ${result.stats.stdDev.toFixed(2)}ms`);
      console.log(`   P95/P99: ${result.stats.percentile95.toFixed(2)}ms / ${result.stats.percentile99.toFixed(2)}ms`);
      console.log(`   Throughput: ${result.throughput.toFixed(2)} ops/sec\n`);
    });

    console.log(`${'='.repeat(80)}\n`);
  }

  /**
   * Compare two variations
   */
  compareVariations(
    report: BenchmarkReport,
    variation1: string,
    variation2: string
  ): {
    fasterVariation: string;
    speedupFactor: number;
    percentDifference: number;
  } {
    const result1 = report.results.find((r) => r.variationId === variation1);
    const result2 = report.results.find((r) => r.variationId === variation2);

    if (!result1 || !result2) {
      throw new Error('Variation not found in report');
    }

    const speedupFactor = result2.stats.mean / result1.stats.mean;
    const percentDifference =
      ((result2.stats.mean - result1.stats.mean) / result1.stats.mean) * 100;

    return {
      fasterVariation: speedupFactor > 1 ? variation1 : variation2,
      speedupFactor: Math.abs(speedupFactor),
      percentDifference: Math.abs(percentDifference),
    };
  }

  /**
   * Export reports to JSON
   */
  exportReports(): BenchmarkReport[] {
    return [...this.reports];
  }

  /**
   * Get all reports
   */
  getReports(): BenchmarkReport[] {
    return this.reports;
  }

  /**
   * Clear all reports
   */
  clearReports(): void {
    this.reports = [];
  }
}
