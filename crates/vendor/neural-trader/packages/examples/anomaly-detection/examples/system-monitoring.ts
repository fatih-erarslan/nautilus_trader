/**
 * System Monitoring Example
 *
 * Demonstrates real-time system anomaly detection for performance monitoring,
 * resource exhaustion detection, and incident prediction.
 */

import { AnomalyDetector } from '../src/detector';
import type { AnomalyPoint } from '../src/index';

interface SystemMetrics {
  timestamp: number;
  cpu: number; // 0-1
  memory: number; // 0-1
  diskIO: number; // MB/s
  networkIO: number; // MB/s
  responseTime: number; // ms
  errorRate: number; // 0-1
}

class SystemMonitor {
  private detector: AnomalyDetector;
  private baselineMetrics: SystemMetrics[] = [];
  private alertThreshold = 0.8;

  constructor() {
    this.detector = new AnomalyDetector({
      targetFalsePositiveRate: 0.02,
      featureDimensions: 6,
      useEnsemble: true,
      useConformal: true,
      windowSize: 500,
      minCalibrationSamples: 100,
      agentDbPath: './system-anomalies.db',
    });
  }

  /**
   * Initialize with baseline system metrics
   */
  async initialize(baselineData: SystemMetrics[]): Promise<void> {
    console.log('Initializing system monitor...');

    this.baselineMetrics = baselineData;

    const trainingData: AnomalyPoint[] = baselineData.map(metrics =>
      this.metricsToAnomalyPoint(metrics, 'normal')
    );

    await this.detector.calibrate(trainingData);

    console.log('System monitor ready');
    console.log('Baseline statistics:', this.getBaselineStats());
  }

  /**
   * Check current system metrics for anomalies
   */
  async checkMetrics(metrics: SystemMetrics): Promise<{
    isAnomalous: boolean;
    severity: 'low' | 'medium' | 'high' | 'critical';
    score: number;
    confidence: number;
    alerts: string[];
    recommendations: string[];
  }> {
    const point = this.metricsToAnomalyPoint(metrics);
    const result = await this.detector.detect(point);

    const alerts = this.generateAlerts(metrics, result.detection.score);
    const recommendations = this.generateRecommendations(metrics, alerts);
    const severity = this.calculateSeverity(result.detection.score, result.detection.confidence);

    return {
      isAnomalous: result.detection.isAnomaly,
      severity,
      score: result.detection.score,
      confidence: result.detection.confidence,
      alerts,
      recommendations,
    };
  }

  /**
   * Convert metrics to feature vector
   */
  private metricsToAnomalyPoint(
    metrics: SystemMetrics,
    label?: 'normal' | 'anomaly'
  ): AnomalyPoint {
    const baseline = this.getBaselineStats();

    // Normalize features by baseline
    const features = [
      // CPU deviation from baseline
      (metrics.cpu - baseline.cpu.mean) / (baseline.cpu.std || 0.1),

      // Memory deviation
      (metrics.memory - baseline.memory.mean) / (baseline.memory.std || 0.1),

      // Disk I/O deviation
      (metrics.diskIO - baseline.diskIO.mean) / (baseline.diskIO.std || 1),

      // Network I/O deviation
      (metrics.networkIO - baseline.networkIO.mean) / (baseline.networkIO.std || 1),

      // Response time deviation
      (metrics.responseTime - baseline.responseTime.mean) / (baseline.responseTime.std || 10),

      // Error rate deviation
      (metrics.errorRate - baseline.errorRate.mean) / (baseline.errorRate.std || 0.01),
    ];

    return {
      timestamp: metrics.timestamp,
      features,
      label,
      metadata: metrics,
    };
  }

  /**
   * Generate alerts based on metrics and anomaly score
   */
  private generateAlerts(metrics: SystemMetrics, score: number): string[] {
    const alerts: string[] = [];

    if (metrics.cpu > 0.9) {
      alerts.push('CRITICAL: CPU usage > 90%');
    } else if (metrics.cpu > 0.8) {
      alerts.push('WARNING: High CPU usage');
    }

    if (metrics.memory > 0.9) {
      alerts.push('CRITICAL: Memory usage > 90%');
    } else if (metrics.memory > 0.8) {
      alerts.push('WARNING: High memory usage');
    }

    if (metrics.responseTime > 1000) {
      alerts.push('CRITICAL: Response time > 1s');
    } else if (metrics.responseTime > 500) {
      alerts.push('WARNING: Elevated response time');
    }

    if (metrics.errorRate > 0.05) {
      alerts.push('CRITICAL: Error rate > 5%');
    } else if (metrics.errorRate > 0.01) {
      alerts.push('WARNING: Elevated error rate');
    }

    if (score > 5) {
      alerts.push('ANOMALY: Unusual system behavior detected');
    }

    return alerts;
  }

  /**
   * Generate actionable recommendations
   */
  private generateRecommendations(metrics: SystemMetrics, alerts: string[]): string[] {
    const recommendations: string[] = [];

    if (alerts.some(a => a.includes('CPU'))) {
      recommendations.push('Scale horizontally or optimize CPU-intensive operations');
    }

    if (alerts.some(a => a.includes('Memory'))) {
      recommendations.push('Check for memory leaks or increase instance size');
    }

    if (alerts.some(a => a.includes('Response time'))) {
      recommendations.push('Review database queries and add caching');
    }

    if (alerts.some(a => a.includes('Error rate'))) {
      recommendations.push('Check application logs and recent deployments');
    }

    if (metrics.diskIO > 100 && metrics.cpu < 0.5) {
      recommendations.push('I/O bottleneck detected - consider faster storage');
    }

    if (metrics.memory > 0.8 && metrics.cpu > 0.8) {
      recommendations.push('Resource exhaustion imminent - scale immediately');
    }

    return recommendations;
  }

  /**
   * Calculate alert severity
   */
  private calculateSeverity(
    score: number,
    confidence: number
  ): 'low' | 'medium' | 'high' | 'critical' {
    const weightedScore = score * confidence;

    if (weightedScore > 10) return 'critical';
    if (weightedScore > 5) return 'high';
    if (weightedScore > 2) return 'medium';
    return 'low';
  }

  /**
   * Get baseline statistics
   */
  private getBaselineStats() {
    if (this.baselineMetrics.length === 0) {
      return {
        cpu: { mean: 0.3, std: 0.1 },
        memory: { mean: 0.5, std: 0.1 },
        diskIO: { mean: 10, std: 5 },
        networkIO: { mean: 20, std: 10 },
        responseTime: { mean: 100, std: 50 },
        errorRate: { mean: 0.001, std: 0.001 },
      };
    }

    const stats = (key: keyof SystemMetrics) => {
      const values = this.baselineMetrics.map(m => m[key] as number);
      const mean = values.reduce((a, b) => a + b, 0) / values.length;
      const variance =
        values.reduce((sum, val) => sum + (val - mean) ** 2, 0) / values.length;
      const std = Math.sqrt(variance);
      return { mean, std };
    };

    return {
      cpu: stats('cpu'),
      memory: stats('memory'),
      diskIO: stats('diskIO'),
      networkIO: stats('networkIO'),
      responseTime: stats('responseTime'),
      errorRate: stats('errorRate'),
    };
  }

  /**
   * Get current detector statistics
   */
  getStatistics() {
    return this.detector.getStatistics();
  }
}

// Example usage
async function main() {
  const monitor = new SystemMonitor();

  // Generate baseline metrics (normal operation)
  const baselineMetrics: SystemMetrics[] = Array.from({ length: 200 }, (_, i) => ({
    timestamp: Date.now() - (200 - i) * 60 * 1000, // 1 minute intervals
    cpu: 0.3 + Math.random() * 0.2,
    memory: 0.5 + Math.random() * 0.1,
    diskIO: 10 + Math.random() * 10,
    networkIO: 20 + Math.random() * 20,
    responseTime: 80 + Math.random() * 40,
    errorRate: 0.001 + Math.random() * 0.002,
  }));

  await monitor.initialize(baselineMetrics);

  // Simulate various anomalous scenarios
  const scenarios: SystemMetrics[] = [
    {
      timestamp: Date.now(),
      cpu: 0.95,
      memory: 0.92,
      diskIO: 15,
      networkIO: 25,
      responseTime: 2000,
      errorRate: 0.08,
    },
    {
      timestamp: Date.now() + 1000,
      cpu: 0.35,
      memory: 0.55,
      diskIO: 12,
      networkIO: 22,
      responseTime: 95,
      errorRate: 0.002,
    },
    {
      timestamp: Date.now() + 2000,
      cpu: 0.6,
      memory: 0.88,
      diskIO: 200,
      networkIO: 150,
      responseTime: 500,
      errorRate: 0.015,
    },
  ];

  for (const metrics of scenarios) {
    const result = await monitor.checkMetrics(metrics);

    console.log('\n--- System Check ---');
    console.log('Timestamp:', new Date(metrics.timestamp).toISOString());
    console.log('Metrics:', {
      cpu: (metrics.cpu * 100).toFixed(1) + '%',
      memory: (metrics.memory * 100).toFixed(1) + '%',
      diskIO: metrics.diskIO.toFixed(1) + ' MB/s',
      networkIO: metrics.networkIO.toFixed(1) + ' MB/s',
      responseTime: metrics.responseTime.toFixed(0) + ' ms',
      errorRate: (metrics.errorRate * 100).toFixed(2) + '%',
    });
    console.log('Status:', result.isAnomalous ? 'ANOMALOUS' : 'NORMAL');
    console.log('Severity:', result.severity.toUpperCase());
    console.log('Anomaly Score:', result.score.toFixed(3));
    console.log('Confidence:', (result.confidence * 100).toFixed(1) + '%');

    if (result.alerts.length > 0) {
      console.log('Alerts:');
      result.alerts.forEach(alert => console.log('  -', alert));
    }

    if (result.recommendations.length > 0) {
      console.log('Recommendations:');
      result.recommendations.forEach(rec => console.log('  -', rec));
    }
  }

  // Show detector statistics
  console.log('\n--- Detector Statistics ---');
  console.log(monitor.getStatistics());
}

// Run example
if (require.main === module) {
  main().catch(console.error);
}

export { SystemMonitor };
