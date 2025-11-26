/**
 * Performance regression detection
 */

import { BenchmarkResult, RegressionAlert, PerformanceHistory } from '../types';
import { StatisticalAnalyzer } from './statistical-analyzer';

export interface RegressionThresholds {
  mean?: number;          // % degradation in mean time
  p95?: number;           // % degradation in p95 time
  throughput?: number;    // % decrease in throughput
  memory?: number;        // % increase in memory
  significance?: number;  // p-value threshold
}

export class RegressionDetector {
  private analyzer: StatisticalAnalyzer;
  private thresholds: Required<RegressionThresholds>;

  constructor(thresholds: RegressionThresholds = {}) {
    this.analyzer = new StatisticalAnalyzer();
    this.thresholds = {
      mean: thresholds.mean ?? 10,           // 10% degradation
      p95: thresholds.p95 ?? 15,             // 15% degradation
      throughput: thresholds.throughput ?? 10, // 10% decrease
      memory: thresholds.memory ?? 20,       // 20% increase
      significance: thresholds.significance ?? 0.05
    };
  }

  /**
   * Detect regressions by comparing against baseline
   */
  detect(
    baseline: BenchmarkResult,
    current: BenchmarkResult
  ): RegressionAlert[] {
    const alerts: RegressionAlert[] = [];

    // Check mean execution time
    const meanDegradation = ((current.mean - baseline.mean) / baseline.mean) * 100;
    if (meanDegradation > this.thresholds.mean) {
      alerts.push({
        benchmark: current.name,
        metric: 'mean',
        threshold: this.thresholds.mean,
        actual: meanDegradation,
        severity: this.calculateSeverity(meanDegradation, this.thresholds.mean),
        timestamp: Date.now()
      });
    }

    // Check p95 latency
    const p95Degradation = ((current.p95 - baseline.p95) / baseline.p95) * 100;
    if (p95Degradation > this.thresholds.p95) {
      alerts.push({
        benchmark: current.name,
        metric: 'p95',
        threshold: this.thresholds.p95,
        actual: p95Degradation,
        severity: this.calculateSeverity(p95Degradation, this.thresholds.p95),
        timestamp: Date.now()
      });
    }

    // Check throughput
    const throughputDecrease = ((baseline.throughput - current.throughput) / baseline.throughput) * 100;
    if (throughputDecrease > this.thresholds.throughput) {
      alerts.push({
        benchmark: current.name,
        metric: 'throughput',
        threshold: this.thresholds.throughput,
        actual: throughputDecrease,
        severity: this.calculateSeverity(throughputDecrease, this.thresholds.throughput),
        timestamp: Date.now()
      });
    }

    // Check memory usage
    const memoryIncrease = ((current.memory.peak - baseline.memory.peak) / baseline.memory.peak) * 100;
    if (memoryIncrease > this.thresholds.memory) {
      alerts.push({
        benchmark: current.name,
        metric: 'memory',
        threshold: this.thresholds.memory,
        actual: memoryIncrease,
        severity: this.calculateSeverity(memoryIncrease, this.thresholds.memory),
        timestamp: Date.now()
      });
    }

    return alerts;
  }

  /**
   * Detect regressions from historical data
   */
  detectFromHistory(history: PerformanceHistory): RegressionAlert[] {
    if (history.results.length < 2) {
      return [];
    }

    const alerts: RegressionAlert[] = [];
    const recent = history.results.slice(-5); // Last 5 runs

    // Check trend
    const trendAnalysis = this.analyzer.analyzeTrend(history.results);
    if (trendAnalysis.trend === 'degrading' && trendAnalysis.rSquared > 0.7) {
      alerts.push({
        benchmark: history.benchmark,
        metric: 'trend',
        threshold: 0,
        actual: trendAnalysis.slope,
        severity: trendAnalysis.rSquared > 0.9 ? 'high' : 'medium',
        timestamp: Date.now()
      });
    }

    // Check for sudden spike
    const anomalies = this.analyzer.detectAnomalies(recent, 2);
    if (anomalies.length > 0) {
      const latestAnomaly = anomalies[anomalies.length - 1];
      alerts.push({
        benchmark: history.benchmark,
        metric: 'anomaly',
        threshold: 2,
        actual: latestAnomaly.zScore,
        severity: latestAnomaly.zScore > 3 ? 'critical' : 'high',
        timestamp: Date.now()
      });
    }

    return alerts;
  }

  /**
   * Check if performance has improved
   */
  hasImproved(
    baseline: BenchmarkResult,
    current: BenchmarkResult,
    minImprovement = 5 // 5% improvement
  ): boolean {
    const meanImprovement = ((baseline.mean - current.mean) / baseline.mean) * 100;
    return meanImprovement >= minImprovement;
  }

  /**
   * Calculate regression score (0-100, higher is worse)
   */
  calculateRegressionScore(alerts: RegressionAlert[]): number {
    if (alerts.length === 0) return 0;

    const severityWeights = {
      low: 1,
      medium: 2,
      high: 3,
      critical: 4
    };

    const totalScore = alerts.reduce((sum, alert) =>
      sum + severityWeights[alert.severity] * (alert.actual / alert.threshold),
      0
    );

    return Math.min(100, (totalScore / alerts.length) * 10);
  }

  private calculateSeverity(
    actual: number,
    threshold: number
  ): 'low' | 'medium' | 'high' | 'critical' {
    const ratio = actual / threshold;

    if (ratio > 3) return 'critical';
    if (ratio > 2) return 'high';
    if (ratio > 1.5) return 'medium';
    return 'low';
  }
}
