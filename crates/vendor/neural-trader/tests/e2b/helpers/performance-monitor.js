/**
 * Performance Monitor Helper
 *
 * Tracks and analyzes performance metrics for E2B operations
 */

class PerformanceMonitor {
  constructor() {
    this.metrics = new Map();
    this.startTimes = new Map();
  }

  /**
   * Start timing an operation
   * @param {string} operationId - Operation identifier
   */
  startOperation(operationId) {
    this.startTimes.set(operationId, Date.now());
  }

  /**
   * End timing an operation
   * @param {string} operationId - Operation identifier
   * @returns {number} Duration in milliseconds
   */
  endOperation(operationId) {
    const startTime = this.startTimes.get(operationId);
    if (!startTime) {
      console.warn(`No start time found for operation: ${operationId}`);
      return 0;
    }

    const duration = Date.now() - startTime;
    this.startTimes.delete(operationId);
    return duration;
  }

  /**
   * Record a performance metric
   * @param {string} metricName - Metric name
   * @param {Object} data - Metric data
   */
  recordMetric(metricName, data) {
    if (!this.metrics.has(metricName)) {
      this.metrics.set(metricName, []);
    }

    this.metrics.get(metricName).push({
      ...data,
      timestamp: new Date().toISOString(),
    });
  }

  /**
   * Get metrics by name
   * @param {string} metricName - Metric name
   * @returns {Array} Metric data
   */
  getMetric(metricName) {
    return this.metrics.get(metricName) || [];
  }

  /**
   * Get all metrics
   * @returns {Object} All metrics
   */
  getAllMetrics() {
    const result = {};
    this.metrics.forEach((value, key) => {
      result[key] = value;
    });
    return result;
  }

  /**
   * Calculate statistics for a metric
   * @param {string} metricName - Metric name
   * @param {string} field - Field to analyze
   * @returns {Object} Statistics
   */
  calculateStats(metricName, field) {
    const data = this.getMetric(metricName);
    if (data.length === 0) {
      return null;
    }

    const values = data.map(d => d[field]).filter(v => typeof v === 'number');
    if (values.length === 0) {
      return null;
    }

    const sum = values.reduce((a, b) => a + b, 0);
    const mean = sum / values.length;
    const sorted = [...values].sort((a, b) => a - b);
    const median = sorted[Math.floor(sorted.length / 2)];
    const min = Math.min(...values);
    const max = Math.max(...values);

    // Calculate standard deviation
    const squareDiffs = values.map(value => Math.pow(value - mean, 2));
    const avgSquareDiff = squareDiffs.reduce((a, b) => a + b, 0) / values.length;
    const stdDev = Math.sqrt(avgSquareDiff);

    return {
      count: values.length,
      mean,
      median,
      min,
      max,
      stdDev,
      sum,
    };
  }

  /**
   * Generate comprehensive performance report
   * @returns {Object} Performance report
   */
  generateReport() {
    const report = {
      summary: {
        totalMetrics: this.metrics.size,
        generatedAt: new Date().toISOString(),
      },
      metrics: {},
    };

    // Analyze each metric
    this.metrics.forEach((data, metricName) => {
      const metricReport = {
        count: data.length,
        samples: data,
      };

      // Calculate stats for common fields
      const timeFields = ['deploymentTimeMs', 'executionTimeMs', 'packageInstallTimeMs'];
      timeFields.forEach(field => {
        const stats = this.calculateStats(metricName, field);
        if (stats) {
          metricReport[`${field}_stats`] = stats;
        }
      });

      report.metrics[metricName] = metricReport;
    });

    // Add deployment summary
    const deploymentMetrics = Array.from(this.metrics.keys())
      .filter(k => k.includes('deploy'));

    if (deploymentMetrics.length > 0) {
      const totalDeployments = deploymentMetrics.reduce((sum, metric) => {
        return sum + this.getMetric(metric).length;
      }, 0);

      report.summary.totalDeployments = totalDeployments;
    }

    return report;
  }

  /**
   * Export metrics to JSON
   * @returns {string} JSON string
   */
  exportMetrics() {
    return JSON.stringify(this.generateReport(), null, 2);
  }

  /**
   * Clear all metrics
   */
  clearMetrics() {
    this.metrics.clear();
    this.startTimes.clear();
  }

  /**
   * Get performance summary
   * @returns {Object} Summary
   */
  getSummary() {
    const allMetrics = this.getAllMetrics();
    const deploymentTimes = [];
    const executionTimes = [];

    Object.values(allMetrics).forEach(metricArray => {
      metricArray.forEach(metric => {
        if (metric.deploymentTimeMs) {
          deploymentTimes.push(metric.deploymentTimeMs);
        }
        if (metric.executionTimeMs) {
          executionTimes.push(metric.executionTimeMs);
        }
      });
    });

    return {
      totalMetrics: Object.keys(allMetrics).length,
      avgDeploymentTime: deploymentTimes.length > 0
        ? deploymentTimes.reduce((a, b) => a + b) / deploymentTimes.length
        : 0,
      avgExecutionTime: executionTimes.length > 0
        ? executionTimes.reduce((a, b) => a + b) / executionTimes.length
        : 0,
    };
  }
}

module.exports = { PerformanceMonitor };
