/**
 * Latency Tracking and Performance Analysis
 * 
 * Monitors and analyzes request/response latencies with statistical analysis
 * and alerting for performance degradation.
 */

interface LatencyMetrics {
  averageLatency: number;
  medianLatency: number;
  p95Latency: number;
  p99Latency: number;
  minLatency: number;
  maxLatency: number;
  sampleCount: number;
}

export class LatencyTracker {
  private measurements: number[] = [];
  private maxSamples: number;
  private alertThreshold: number;
  private onLatencyAlert?: (latency: number, threshold: number) => void;

  constructor(maxSamples = 10000, alertThreshold = 25000) { // 25ms default alert
    this.maxSamples = maxSamples;
    this.alertThreshold = alertThreshold; // in microseconds
  }

  /**
   * Record a latency measurement
   */
  public record(latencyMicroseconds: number): void {
    // Keep rolling window of measurements
    if (this.measurements.length >= this.maxSamples) {
      this.measurements.shift(); // Remove oldest measurement
    }
    
    this.measurements.push(latencyMicroseconds);

    // Check for latency alerts
    if (latencyMicroseconds > this.alertThreshold && this.onLatencyAlert) {
      this.onLatencyAlert(latencyMicroseconds, this.alertThreshold);
    }
  }

  /**
   * Get current latency metrics
   */
  public getMetrics(): LatencyMetrics {
    if (this.measurements.length === 0) {
      return {
        averageLatency: 0,
        medianLatency: 0,
        p95Latency: 0,
        p99Latency: 0,
        minLatency: 0,
        maxLatency: 0,
        sampleCount: 0,
      };
    }

    const sorted = [...this.measurements].sort((a, b) => a - b);
    const len = sorted.length;

    return {
      averageLatency: this.measurements.reduce((sum, val) => sum + val, 0) / len,
      medianLatency: this.calculatePercentile(sorted, 50),
      p95Latency: this.calculatePercentile(sorted, 95),
      p99Latency: this.calculatePercentile(sorted, 99),
      minLatency: sorted[0],
      maxLatency: sorted[len - 1],
      sampleCount: len,
    };
  }

  /**
   * Calculate specific percentile
   */
  private calculatePercentile(sortedArray: number[], percentile: number): number {
    const index = (percentile / 100) * (sortedArray.length - 1);
    
    if (Number.isInteger(index)) {
      return sortedArray[index];
    }
    
    const lower = Math.floor(index);
    const upper = Math.ceil(index);
    const weight = index % 1;
    
    return sortedArray[lower] * (1 - weight) + sortedArray[upper] * weight;
  }

  /**
   * Get average latency
   */
  public getAverageLatency(): number {
    if (this.measurements.length === 0) return 0;
    
    return this.measurements.reduce((sum, val) => sum + val, 0) / this.measurements.length;
  }

  /**
   * Get recent latency trend (last N measurements)
   */
  public getRecentTrend(sampleSize = 100): {
    trend: 'improving' | 'degrading' | 'stable';
    changePercent: number;
    recentAverage: number;
    previousAverage: number;
  } {
    if (this.measurements.length < sampleSize * 2) {
      return {
        trend: 'stable',
        changePercent: 0,
        recentAverage: this.getAverageLatency(),
        previousAverage: this.getAverageLatency(),
      };
    }

    const total = this.measurements.length;
    const recent = this.measurements.slice(total - sampleSize);
    const previous = this.measurements.slice(total - sampleSize * 2, total - sampleSize);

    const recentAvg = recent.reduce((sum, val) => sum + val, 0) / recent.length;
    const previousAvg = previous.reduce((sum, val) => sum + val, 0) / previous.length;
    
    const changePercent = ((recentAvg - previousAvg) / previousAvg) * 100;
    
    let trend: 'improving' | 'degrading' | 'stable' = 'stable';
    if (Math.abs(changePercent) > 5) { // 5% threshold for trend detection
      trend = changePercent < 0 ? 'improving' : 'degrading';
    }

    return {
      trend,
      changePercent,
      recentAverage: recentAvg,
      previousAverage: previousAvg,
    };
  }

  /**
   * Check if latency is within acceptable bounds
   */
  public isPerformanceAcceptable(
    maxAverageLatency = 10000, // 10ms
    maxP95Latency = 25000       // 25ms
  ): boolean {
    const metrics = this.getMetrics();
    return metrics.averageLatency <= maxAverageLatency && 
           metrics.p95Latency <= maxP95Latency;
  }

  /**
   * Set latency alert callback
   */
  public onAlert(callback: (latency: number, threshold: number) => void): void {
    this.onLatencyAlert = callback;
  }

  /**
   * Update alert threshold
   */
  public setAlertThreshold(thresholdMicroseconds: number): void {
    this.alertThreshold = thresholdMicroseconds;
  }

  /**
   * Clear all measurements
   */
  public clear(): void {
    this.measurements = [];
  }

  /**
   * Get latency distribution histogram
   */
  public getHistogram(bucketCount = 10): Array<{ range: string; count: number; percentage: number }> {
    if (this.measurements.length === 0) return [];

    const metrics = this.getMetrics();
    const bucketSize = (metrics.maxLatency - metrics.minLatency) / bucketCount;
    const buckets: Array<{ range: string; count: number; percentage: number }> = [];

    for (let i = 0; i < bucketCount; i++) {
      const rangeStart = metrics.minLatency + (bucketSize * i);
      const rangeEnd = metrics.minLatency + (bucketSize * (i + 1));
      
      const count = this.measurements.filter(
        measurement => measurement >= rangeStart && measurement < rangeEnd
      ).length;

      // Format range in appropriate units
      const formatLatency = (us: number): string => {
        if (us < 1000) return `${us.toFixed(0)}μs`;
        if (us < 1000000) return `${(us / 1000).toFixed(1)}ms`;
        return `${(us / 1000000).toFixed(2)}s`;
      };

      buckets.push({
        range: `${formatLatency(rangeStart)} - ${formatLatency(rangeEnd)}`,
        count,
        percentage: (count / this.measurements.length) * 100,
      });
    }

    return buckets;
  }

  /**
   * Export measurements for external analysis
   */
  public exportMeasurements(): number[] {
    return [...this.measurements];
  }

  /**
   * Import measurements (e.g., for initialization from stored data)
   */
  public importMeasurements(measurements: number[]): void {
    this.measurements = measurements.slice(-this.maxSamples);
  }

  /**
   * Get performance summary
   */
  public getSummary(): string {
    const metrics = this.getMetrics();
    const trend = this.getRecentTrend();
    
    const formatLatency = (us: number): string => {
      if (us < 1000) return `${us.toFixed(0)}μs`;
      if (us < 1000000) return `${(us / 1000).toFixed(1)}ms`;
      return `${(us / 1000000).toFixed(2)}s`;
    };

    return `Performance Summary:
    • Average: ${formatLatency(metrics.averageLatency)}
    • Median: ${formatLatency(metrics.medianLatency)}
    • P95: ${formatLatency(metrics.p95Latency)}
    • P99: ${formatLatency(metrics.p99Latency)}
    • Min/Max: ${formatLatency(metrics.minLatency)} / ${formatLatency(metrics.maxLatency)}
    • Samples: ${metrics.sampleCount}
    • Trend: ${trend.trend} (${trend.changePercent.toFixed(1)}%)`;
  }
}