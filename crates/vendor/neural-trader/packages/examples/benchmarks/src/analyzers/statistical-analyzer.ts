/**
 * Statistical analysis for benchmark results
 */

import { BenchmarkResult, StatisticalSummary } from '../types';

export class StatisticalAnalyzer {
  /**
   * Generate statistical summary from raw samples
   */
  analyze(samples: number[]): StatisticalSummary {
    if (samples.length === 0) {
      throw new Error('Cannot analyze empty sample set');
    }

    const sorted = [...samples].sort((a, b) => a - b);
    const mean = this.calculateMean(samples);
    const variance = this.calculateVariance(samples, mean);
    const stdDev = Math.sqrt(variance);

    const q1 = this.percentile(sorted, 25);
    const median = this.percentile(sorted, 50);
    const q3 = this.percentile(sorted, 75);
    const iqr = q3 - q1;

    // Detect outliers using IQR method
    const lowerBound = q1 - 1.5 * iqr;
    const upperBound = q3 + 1.5 * iqr;
    const outliers = samples.filter(x => x < lowerBound || x > upperBound);

    return {
      mean,
      median,
      stdDev,
      variance,
      min: sorted[0],
      max: sorted[sorted.length - 1],
      q1,
      q3,
      p95: this.percentile(sorted, 95),
      p99: this.percentile(sorted, 99),
      iqr,
      outliers
    };
  }

  /**
   * Calculate confidence interval
   */
  confidenceInterval(
    samples: number[],
    confidenceLevel = 0.95
  ): [number, number] {
    const mean = this.calculateMean(samples);
    const stdError = Math.sqrt(this.calculateVariance(samples, mean)) / Math.sqrt(samples.length);

    // Z-score for confidence level (approximate)
    const z = confidenceLevel === 0.95 ? 1.96 : 2.576;

    const margin = z * stdError;
    return [mean - margin, mean + margin];
  }

  /**
   * Perform trend analysis on historical results
   */
  analyzeTrend(results: BenchmarkResult[]): {
    trend: 'improving' | 'degrading' | 'stable';
    slope: number;
    rSquared: number;
  } {
    if (results.length < 2) {
      return { trend: 'stable', slope: 0, rSquared: 0 };
    }

    const n = results.length;
    const x = Array.from({ length: n }, (_, i) => i);
    const y = results.map(r => r.mean);

    const { slope, rSquared } = this.linearRegression(x, y);

    let trend: 'improving' | 'degrading' | 'stable';
    if (Math.abs(slope) < 0.01) {
      trend = 'stable';
    } else if (slope < 0) {
      trend = 'improving'; // Lower execution time is better
    } else {
      trend = 'degrading';
    }

    return { trend, slope, rSquared };
  }

  /**
   * Detect performance anomalies
   */
  detectAnomalies(
    results: BenchmarkResult[],
    threshold = 2
  ): Array<{ index: number; result: BenchmarkResult; zScore: number }> {
    const means = results.map(r => r.mean);
    const overallMean = this.calculateMean(means);
    const overallStdDev = Math.sqrt(this.calculateVariance(means, overallMean));

    const anomalies: Array<{ index: number; result: BenchmarkResult; zScore: number }> = [];

    results.forEach((result, index) => {
      const zScore = Math.abs((result.mean - overallMean) / overallStdDev);
      if (zScore > threshold) {
        anomalies.push({ index, result, zScore });
      }
    });

    return anomalies;
  }

  /**
   * Compare distributions using Kolmogorov-Smirnov test
   */
  ksTest(sample1: number[], sample2: number[]): {
    statistic: number;
    pValue: number;
    significant: boolean;
  } {
    const sorted1 = [...sample1].sort((a, b) => a - b);
    const sorted2 = [...sample2].sort((a, b) => a - b);

    let maxDiff = 0;
    let i = 0, j = 0;

    while (i < sorted1.length && j < sorted2.length) {
      const cdf1 = (i + 1) / sorted1.length;
      const cdf2 = (j + 1) / sorted2.length;
      const diff = Math.abs(cdf1 - cdf2);

      maxDiff = Math.max(maxDiff, diff);

      if (sorted1[i] < sorted2[j]) {
        i++;
      } else {
        j++;
      }
    }

    // Approximate p-value
    const n = sample1.length;
    const m = sample2.length;
    const effectiveN = Math.sqrt((n * m) / (n + m));
    const pValue = Math.exp(-2 * maxDiff * maxDiff * effectiveN);

    return {
      statistic: maxDiff,
      pValue,
      significant: pValue < 0.05
    };
  }

  private calculateMean(samples: number[]): number {
    return samples.reduce((a, b) => a + b, 0) / samples.length;
  }

  private calculateVariance(samples: number[], mean: number): number {
    return samples.reduce((sum, val) =>
      sum + Math.pow(val - mean, 2), 0
    ) / (samples.length - 1);
  }

  private percentile(sorted: number[], p: number): number {
    const index = (p / 100) * (sorted.length - 1);
    const lower = Math.floor(index);
    const upper = Math.ceil(index);
    const weight = index - lower;

    return sorted[lower] * (1 - weight) + sorted[upper] * weight;
  }

  private linearRegression(x: number[], y: number[]): {
    slope: number;
    intercept: number;
    rSquared: number;
  } {
    const n = x.length;
    const sumX = x.reduce((a, b) => a + b, 0);
    const sumY = y.reduce((a, b) => a + b, 0);
    const sumXY = x.reduce((sum, xi, i) => sum + xi * y[i], 0);
    const sumX2 = x.reduce((sum, xi) => sum + xi * xi, 0);
    const sumY2 = y.reduce((sum, yi) => sum + yi * yi, 0);

    const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
    const intercept = (sumY - slope * sumX) / n;

    // Calculate R²
    const yMean = sumY / n;
    const ssTotal = y.reduce((sum, yi) => sum + Math.pow(yi - yMean, 2), 0);
    const ssResidual = y.reduce((sum, yi, i) =>
      sum + Math.pow(yi - (slope * x[i] + intercept), 2), 0
    );
    const rSquared = 1 - ssResidual / ssTotal;

    return { slope, intercept, rSquared };
  }
}
