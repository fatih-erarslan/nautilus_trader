/**
 * E2B Trading Swarm Monitor and Auto-Scaler
 *
 * Comprehensive monitoring and auto-scaling system for E2B trading swarms.
 * Features:
 * - Real-time health monitoring for all sandboxes
 * - ML-based anomaly detection
 * - Intelligent auto-scaling based on market conditions
 * - Alert system for failures and anomalies
 * - Performance metrics collection and analysis
 * - Resource optimization recommendations
 */

const EventEmitter = require('events');

/**
 * Health status types
 */
const HealthStatus = {
  HEALTHY: 'healthy',
  DEGRADED: 'degraded',
  CRITICAL: 'critical',
  FAILED: 'failed',
  UNKNOWN: 'unknown'
};

/**
 * Scaling actions
 */
const ScalingAction = {
  SCALE_UP: 'scale_up',
  SCALE_DOWN: 'scale_down',
  MAINTAIN: 'maintain',
  FAILOVER: 'failover',
  EMERGENCY_SHUTDOWN: 'emergency_shutdown'
};

/**
 * Alert severity levels
 */
const AlertSeverity = {
  INFO: 'info',
  WARNING: 'warning',
  CRITICAL: 'critical',
  EMERGENCY: 'emergency'
};

/**
 * E2B Monitor and Auto-Scaler
 */
class E2BMonitor extends EventEmitter {
  constructor(options = {}) {
    super();

    this.options = {
      monitorInterval: options.monitorInterval || 5000, // 5 seconds
      healthCheckTimeout: options.healthCheckTimeout || 3000,
      anomalyDetectionWindow: options.anomalyDetectionWindow || 100,
      scaleUpThreshold: options.scaleUpThreshold || 0.8,
      scaleDownThreshold: options.scaleDownThreshold || 0.3,
      maxSandboxes: options.maxSandboxes || 10,
      minSandboxes: options.minSandboxes || 1,
      cooldownPeriod: options.cooldownPeriod || 60000, // 1 minute
      ...options
    };

    this.sandboxes = new Map();
    this.metrics = new Map();
    this.alerts = [];
    this.anomalyHistory = [];
    this.lastScalingAction = null;
    this.lastScalingTime = 0;
    this.isMonitoring = false;
    this.monitoringInterval = null;

    // ML-based anomaly detection parameters
    this.anomalyDetector = {
      mean: new Map(),
      stdDev: new Map(),
      threshold: 3, // Standard deviations
      minSamples: 10
    };
  }

  /**
   * Start monitoring all sandboxes
   */
  async startMonitoring() {
    if (this.isMonitoring) {
      throw new Error('Monitoring is already active');
    }

    this.isMonitoring = true;
    this.emit('monitoring:started');

    this.monitoringInterval = setInterval(async () => {
      try {
        await this.monitorAllSandboxes();
      } catch (error) {
        this.emit('error', { type: 'monitoring', error: error.message });
      }
    }, this.options.monitorInterval);

    console.log('🔍 E2B monitoring started');
    return { status: 'started', interval: this.options.monitorInterval };
  }

  /**
   * Stop monitoring
   */
  async stopMonitoring() {
    if (!this.isMonitoring) {
      return { status: 'not_running' };
    }

    if (this.monitoringInterval) {
      clearInterval(this.monitoringInterval);
      this.monitoringInterval = null;
    }

    this.isMonitoring = false;
    this.emit('monitoring:stopped');

    console.log('🛑 E2B monitoring stopped');
    return { status: 'stopped' };
  }

  /**
   * Monitor all sandboxes - real-time health checks
   */
  async monitorAllSandboxes() {
    const startTime = Date.now();
    const results = {
      timestamp: new Date().toISOString(),
      totalSandboxes: this.sandboxes.size,
      healthy: 0,
      degraded: 0,
      critical: 0,
      failed: 0,
      checks: []
    };

    const checkPromises = Array.from(this.sandboxes.entries()).map(
      async ([sandboxId, sandbox]) => {
        try {
          const health = await this.checkSandboxHealth(sandboxId, sandbox);

          // Update metrics
          this.recordMetric(sandboxId, health.metrics);

          // Update status counters
          switch (health.status) {
            case HealthStatus.HEALTHY:
              results.healthy++;
              break;
            case HealthStatus.DEGRADED:
              results.degraded++;
              break;
            case HealthStatus.CRITICAL:
              results.critical++;
              break;
            case HealthStatus.FAILED:
              results.failed++;
              break;
          }

          results.checks.push(health);
          return health;
        } catch (error) {
          const failedCheck = {
            sandboxId,
            status: HealthStatus.FAILED,
            error: error.message,
            timestamp: new Date().toISOString()
          };
          results.checks.push(failedCheck);
          results.failed++;
          return failedCheck;
        }
      }
    );

    await Promise.all(checkPromises);

    results.duration = Date.now() - startTime;
    results.healthPercentage = this.sandboxes.size > 0
      ? (results.healthy / this.sandboxes.size) * 100
      : 100;

    // Detect anomalies
    const anomalies = await this.detectAnomalies();
    if (anomalies.length > 0) {
      results.anomalies = anomalies;
    }

    // Check if scaling is needed
    const scalingDecision = await this.evaluateScaling(results);
    if (scalingDecision.action !== ScalingAction.MAINTAIN) {
      results.scalingRecommendation = scalingDecision;
    }

    this.emit('monitoring:complete', results);
    return results;
  }

  /**
   * Check health of a single sandbox
   */
  async checkSandboxHealth(sandboxId, sandbox) {
    const startTime = Date.now();
    const health = {
      sandboxId,
      timestamp: new Date().toISOString(),
      status: HealthStatus.UNKNOWN,
      metrics: {},
      issues: []
    };

    try {
      // CPU usage check
      const cpuUsage = await this.getCpuUsage(sandboxId, sandbox);
      health.metrics.cpu = cpuUsage;
      if (cpuUsage > 90) {
        health.issues.push({ type: 'cpu', severity: 'critical', value: cpuUsage });
      } else if (cpuUsage > 75) {
        health.issues.push({ type: 'cpu', severity: 'warning', value: cpuUsage });
      }

      // Memory usage check
      const memoryUsage = await this.getMemoryUsage(sandboxId, sandbox);
      health.metrics.memory = memoryUsage;
      if (memoryUsage > 90) {
        health.issues.push({ type: 'memory', severity: 'critical', value: memoryUsage });
      } else if (memoryUsage > 75) {
        health.issues.push({ type: 'memory', severity: 'warning', value: memoryUsage });
      }

      // Response time check
      const responseTime = Date.now() - startTime;
      health.metrics.responseTime = responseTime;
      if (responseTime > 5000) {
        health.issues.push({ type: 'latency', severity: 'critical', value: responseTime });
      } else if (responseTime > 2000) {
        health.issues.push({ type: 'latency', severity: 'warning', value: responseTime });
      }

      // Error rate check
      const errorRate = await this.getErrorRate(sandboxId);
      health.metrics.errorRate = errorRate;
      if (errorRate > 0.1) {
        health.issues.push({ type: 'errors', severity: 'critical', value: errorRate });
      } else if (errorRate > 0.05) {
        health.issues.push({ type: 'errors', severity: 'warning', value: errorRate });
      }

      // Trade execution latency
      const tradeLatency = await this.getTradeLatency(sandboxId);
      health.metrics.tradeLatency = tradeLatency;
      if (tradeLatency > 1000) {
        health.issues.push({ type: 'trade_latency', severity: 'warning', value: tradeLatency });
      }

      // Agent response time
      const agentResponseTime = await this.getAgentResponseTime(sandboxId);
      health.metrics.agentResponseTime = agentResponseTime;

      // Determine overall status
      const criticalIssues = health.issues.filter(i => i.severity === 'critical');
      const warningIssues = health.issues.filter(i => i.severity === 'warning');

      if (criticalIssues.length > 0) {
        health.status = HealthStatus.CRITICAL;
        this.createAlert(sandboxId, AlertSeverity.CRITICAL,
          `Sandbox has ${criticalIssues.length} critical issue(s)`, criticalIssues);
      } else if (warningIssues.length > 1) {
        health.status = HealthStatus.DEGRADED;
        this.createAlert(sandboxId, AlertSeverity.WARNING,
          `Sandbox has ${warningIssues.length} warning(s)`, warningIssues);
      } else if (warningIssues.length === 1) {
        health.status = HealthStatus.DEGRADED;
      } else {
        health.status = HealthStatus.HEALTHY;
      }

      health.checkDuration = Date.now() - startTime;
      return health;

    } catch (error) {
      health.status = HealthStatus.FAILED;
      health.error = error.message;
      this.createAlert(sandboxId, AlertSeverity.CRITICAL,
        `Health check failed: ${error.message}`);
      return health;
    }
  }

  /**
   * ML-based anomaly detection
   */
  async detectAnomalies() {
    const anomalies = [];
    const metricTypes = ['cpu', 'memory', 'responseTime', 'errorRate', 'tradeLatency'];

    for (const [sandboxId, metricsHistory] of this.metrics.entries()) {
      if (metricsHistory.length < this.anomalyDetector.minSamples) {
        continue; // Not enough data
      }

      for (const metricType of metricTypes) {
        const values = metricsHistory
          .slice(-this.options.anomalyDetectionWindow)
          .map(m => m[metricType])
          .filter(v => v !== undefined);

        if (values.length === 0) continue;

        // Calculate statistics
        const mean = values.reduce((a, b) => a + b, 0) / values.length;
        const variance = values.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / values.length;
        const stdDev = Math.sqrt(variance);

        // Update detector model
        this.anomalyDetector.mean.set(`${sandboxId}:${metricType}`, mean);
        this.anomalyDetector.stdDev.set(`${sandboxId}:${metricType}`, stdDev);

        // Check current value against model
        const currentValue = values[values.length - 1];
        const zScore = stdDev > 0 ? Math.abs((currentValue - mean) / stdDev) : 0;

        if (zScore > this.anomalyDetector.threshold) {
          const anomaly = {
            sandboxId,
            metricType,
            value: currentValue,
            mean,
            stdDev,
            zScore,
            severity: zScore > 4 ? 'critical' : 'warning',
            timestamp: new Date().toISOString()
          };

          anomalies.push(anomaly);
          this.anomalyHistory.push(anomaly);

          // Keep history limited
          if (this.anomalyHistory.length > 1000) {
            this.anomalyHistory = this.anomalyHistory.slice(-500);
          }

          this.createAlert(sandboxId,
            zScore > 4 ? AlertSeverity.CRITICAL : AlertSeverity.WARNING,
            `Anomaly detected in ${metricType}: ${currentValue.toFixed(2)} (z-score: ${zScore.toFixed(2)})`,
            anomaly
          );

          this.emit('anomaly:detected', anomaly);
        }
      }
    }

    return anomalies;
  }

  /**
   * Evaluate if scaling is needed
   */
  async evaluateScaling(monitoringResults) {
    const decision = {
      action: ScalingAction.MAINTAIN,
      reason: '',
      currentCount: this.sandboxes.size,
      targetCount: this.sandboxes.size,
      confidence: 0
    };

    // Check cooldown period
    const timeSinceLastScaling = Date.now() - this.lastScalingTime;
    if (timeSinceLastScaling < this.options.cooldownPeriod) {
      decision.reason = 'In cooldown period';
      return decision;
    }

    // Calculate average resource usage
    const avgCpu = this.calculateAverageMetric('cpu');
    const avgMemory = this.calculateAverageMetric('memory');
    const avgErrorRate = this.calculateAverageMetric('errorRate');

    // Get market volatility (would be integrated with market data)
    const marketVolatility = await this.getMarketVolatility();

    // Emergency conditions
    if (monitoringResults.failed > monitoringResults.totalSandboxes * 0.3) {
      decision.action = ScalingAction.EMERGENCY_SHUTDOWN;
      decision.reason = 'More than 30% of sandboxes failed';
      decision.confidence = 1.0;
      this.createAlert('system', AlertSeverity.EMERGENCY,
        'Emergency: High failure rate detected');
      return decision;
    }

    if (monitoringResults.critical > monitoringResults.totalSandboxes * 0.5) {
      decision.action = ScalingAction.FAILOVER;
      decision.reason = 'More than 50% of sandboxes in critical state';
      decision.confidence = 0.9;
      this.createAlert('system', AlertSeverity.EMERGENCY,
        'Failover needed: High critical rate');
      return decision;
    }

    // Scale up conditions
    const shouldScaleUp = (
      avgCpu > this.options.scaleUpThreshold ||
      avgMemory > this.options.scaleUpThreshold ||
      marketVolatility > 0.7
    ) && this.sandboxes.size < this.options.maxSandboxes;

    if (shouldScaleUp) {
      const scaleUpFactor = marketVolatility > 0.8 ? 2 : 1;
      decision.action = ScalingAction.SCALE_UP;
      decision.targetCount = Math.min(
        this.sandboxes.size + scaleUpFactor,
        this.options.maxSandboxes
      );
      decision.reason = `High resource usage (CPU: ${(avgCpu * 100).toFixed(1)}%, Memory: ${(avgMemory * 100).toFixed(1)}%) or market volatility (${(marketVolatility * 100).toFixed(1)}%)`;
      decision.confidence = 0.8;
      return decision;
    }

    // Scale down conditions
    const shouldScaleDown = (
      avgCpu < this.options.scaleDownThreshold &&
      avgMemory < this.options.scaleDownThreshold &&
      avgErrorRate < 0.01 &&
      marketVolatility < 0.3
    ) && this.sandboxes.size > this.options.minSandboxes;

    if (shouldScaleDown) {
      decision.action = ScalingAction.SCALE_DOWN;
      decision.targetCount = Math.max(
        this.sandboxes.size - 1,
        this.options.minSandboxes
      );
      decision.reason = `Low resource usage (CPU: ${(avgCpu * 100).toFixed(1)}%, Memory: ${(avgMemory * 100).toFixed(1)}%) and low market volatility`;
      decision.confidence = 0.7;
      return decision;
    }

    decision.reason = 'All metrics within normal range';
    decision.confidence = 0.5;
    return decision;
  }

  /**
   * Execute scaling action
   */
  async scaleBasedOnLoad(metrics) {
    const decision = await this.evaluateScaling(metrics);

    if (decision.action === ScalingAction.MAINTAIN) {
      return { action: 'maintain', message: decision.reason };
    }

    this.lastScalingAction = decision;
    this.lastScalingTime = Date.now();

    switch (decision.action) {
      case ScalingAction.SCALE_UP:
        return await this.scaleUp(decision);

      case ScalingAction.SCALE_DOWN:
        return await this.scaleDown(decision);

      case ScalingAction.FAILOVER:
        return await this.executeFailover(decision);

      case ScalingAction.EMERGENCY_SHUTDOWN:
        return await this.emergencyShutdown(decision);

      default:
        return { action: 'unknown', message: 'Unknown scaling action' };
    }
  }

  /**
   * Scale up sandboxes
   */
  async scaleUp(decision) {
    const newSandboxCount = decision.targetCount - this.sandboxes.size;
    const results = {
      action: 'scale_up',
      created: 0,
      failed: 0,
      sandboxes: []
    };

    console.log(`⬆️  Scaling up: Creating ${newSandboxCount} new sandbox(es)`);

    for (let i = 0; i < newSandboxCount; i++) {
      try {
        const sandboxId = `sandbox-${Date.now()}-${i}`;
        // Integration point: Create actual E2B sandbox
        const sandbox = await this.createSandbox(sandboxId);

        this.sandboxes.set(sandboxId, sandbox);
        this.metrics.set(sandboxId, []);

        results.created++;
        results.sandboxes.push(sandboxId);

        this.emit('sandbox:created', { sandboxId, reason: decision.reason });
      } catch (error) {
        results.failed++;
        this.emit('error', { type: 'scale_up', error: error.message });
      }
    }

    this.createAlert('system', AlertSeverity.INFO,
      `Scaled up: Created ${results.created} sandbox(es)`);

    return results;
  }

  /**
   * Scale down sandboxes
   */
  async scaleDown(decision) {
    const removeCount = this.sandboxes.size - decision.targetCount;
    const results = {
      action: 'scale_down',
      removed: 0,
      failed: 0,
      sandboxes: []
    };

    console.log(`⬇️  Scaling down: Removing ${removeCount} sandbox(es)`);

    // Remove least active sandboxes first
    const sandboxesByActivity = this.getSandboxesByActivity();
    const toRemove = sandboxesByActivity.slice(0, removeCount);

    for (const sandboxId of toRemove) {
      try {
        // Integration point: Gracefully shutdown sandbox
        await this.shutdownSandbox(sandboxId);

        this.sandboxes.delete(sandboxId);
        this.metrics.delete(sandboxId);

        results.removed++;
        results.sandboxes.push(sandboxId);

        this.emit('sandbox:removed', { sandboxId, reason: decision.reason });
      } catch (error) {
        results.failed++;
        this.emit('error', { type: 'scale_down', error: error.message });
      }
    }

    this.createAlert('system', AlertSeverity.INFO,
      `Scaled down: Removed ${results.removed} sandbox(es)`);

    return results;
  }

  /**
   * Execute failover to backup infrastructure
   */
  async executeFailover(decision) {
    console.log('🔄 Executing failover procedure');

    const results = {
      action: 'failover',
      migrated: 0,
      failed: 0,
      sandboxes: []
    };

    // Get healthy sandboxes
    const healthySandboxes = Array.from(this.sandboxes.entries())
      .filter(([_, sandbox]) => sandbox.status === HealthStatus.HEALTHY);

    // Migrate workload from critical sandboxes to healthy ones
    for (const [sandboxId, sandbox] of this.sandboxes.entries()) {
      if (sandbox.status === HealthStatus.CRITICAL || sandbox.status === HealthStatus.FAILED) {
        try {
          // Find healthy target
          const targetSandbox = healthySandboxes[results.migrated % healthySandboxes.length];

          if (targetSandbox) {
            // Integration point: Migrate workload
            await this.migrateWorkload(sandboxId, targetSandbox[0]);

            results.migrated++;
            results.sandboxes.push({ from: sandboxId, to: targetSandbox[0] });

            this.emit('workload:migrated', {
              from: sandboxId,
              to: targetSandbox[0]
            });
          }
        } catch (error) {
          results.failed++;
          this.emit('error', { type: 'failover', error: error.message });
        }
      }
    }

    this.createAlert('system', AlertSeverity.CRITICAL,
      `Failover executed: Migrated ${results.migrated} workload(s)`);

    return results;
  }

  /**
   * Emergency shutdown of all sandboxes
   */
  async emergencyShutdown(decision) {
    console.log('🚨 EMERGENCY SHUTDOWN INITIATED');

    const results = {
      action: 'emergency_shutdown',
      shutdown: 0,
      failed: 0,
      reason: decision.reason
    };

    // Stop monitoring immediately
    await this.stopMonitoring();

    // Shutdown all sandboxes
    const shutdownPromises = Array.from(this.sandboxes.keys()).map(
      async (sandboxId) => {
        try {
          await this.shutdownSandbox(sandboxId, { force: true });
          results.shutdown++;
          this.emit('sandbox:emergency_shutdown', { sandboxId });
        } catch (error) {
          results.failed++;
          this.emit('error', {
            type: 'emergency_shutdown',
            sandboxId,
            error: error.message
          });
        }
      }
    );

    await Promise.all(shutdownPromises);

    this.sandboxes.clear();
    this.metrics.clear();

    this.createAlert('system', AlertSeverity.EMERGENCY,
      `Emergency shutdown completed: ${results.shutdown} sandbox(es) shutdown`);

    return results;
  }

  /**
   * Generate comprehensive health report
   */
  async generateHealthReport() {
    const report = {
      timestamp: new Date().toISOString(),
      summary: {
        totalSandboxes: this.sandboxes.size,
        status: {
          healthy: 0,
          degraded: 0,
          critical: 0,
          failed: 0
        }
      },
      metrics: {
        cpu: { min: 0, max: 0, avg: 0 },
        memory: { min: 0, max: 0, avg: 0 },
        responseTime: { min: 0, max: 0, avg: 0 },
        errorRate: { min: 0, max: 0, avg: 0 },
        tradeLatency: { min: 0, max: 0, avg: 0 }
      },
      alerts: {
        total: this.alerts.length,
        bySeverity: {
          info: 0,
          warning: 0,
          critical: 0,
          emergency: 0
        },
        recent: this.alerts.slice(-10)
      },
      anomalies: {
        total: this.anomalyHistory.length,
        recent: this.anomalyHistory.slice(-10)
      },
      scaling: {
        lastAction: this.lastScalingAction,
        lastActionTime: this.lastScalingTime > 0
          ? new Date(this.lastScalingTime).toISOString()
          : null
      },
      sandboxes: []
    };

    // Collect sandbox details
    for (const [sandboxId, sandbox] of this.sandboxes.entries()) {
      const metricsHistory = this.metrics.get(sandboxId) || [];
      const latestMetrics = metricsHistory[metricsHistory.length - 1] || {};

      const sandboxReport = {
        id: sandboxId,
        status: sandbox.status || HealthStatus.UNKNOWN,
        uptime: sandbox.startTime ? Date.now() - sandbox.startTime : 0,
        metrics: latestMetrics,
        recentAlerts: this.alerts
          .filter(a => a.sandboxId === sandboxId)
          .slice(-5)
      };

      report.sandboxes.push(sandboxReport);

      // Update status counters
      switch (sandboxReport.status) {
        case HealthStatus.HEALTHY:
          report.summary.status.healthy++;
          break;
        case HealthStatus.DEGRADED:
          report.summary.status.degraded++;
          break;
        case HealthStatus.CRITICAL:
          report.summary.status.critical++;
          break;
        case HealthStatus.FAILED:
          report.summary.status.failed++;
          break;
      }
    }

    // Calculate aggregate metrics
    const metricTypes = ['cpu', 'memory', 'responseTime', 'errorRate', 'tradeLatency'];
    for (const metricType of metricTypes) {
      const values = report.sandboxes
        .map(s => s.metrics[metricType])
        .filter(v => v !== undefined);

      if (values.length > 0) {
        report.metrics[metricType] = {
          min: Math.min(...values),
          max: Math.max(...values),
          avg: values.reduce((a, b) => a + b, 0) / values.length
        };
      }
    }

    // Count alerts by severity
    for (const alert of this.alerts) {
      if (report.alerts.bySeverity[alert.severity] !== undefined) {
        report.alerts.bySeverity[alert.severity]++;
      }
    }

    // Generate recommendations
    report.recommendations = this.generateRecommendations(report);

    this.emit('report:generated', report);
    return report;
  }

  /**
   * Generate resource optimization recommendations
   */
  generateRecommendations(report) {
    const recommendations = [];

    // CPU recommendations
    if (report.metrics.cpu.avg > 75) {
      recommendations.push({
        type: 'cpu',
        severity: 'warning',
        message: 'High average CPU usage detected',
        action: 'Consider scaling up or optimizing trading algorithms'
      });
    }

    // Memory recommendations
    if (report.metrics.memory.avg > 75) {
      recommendations.push({
        type: 'memory',
        severity: 'warning',
        message: 'High average memory usage detected',
        action: 'Consider increasing memory allocation or implementing memory optimization'
      });
    }

    // Error rate recommendations
    if (report.metrics.errorRate.avg > 0.05) {
      recommendations.push({
        type: 'errors',
        severity: 'critical',
        message: 'High error rate detected',
        action: 'Investigate error patterns and implement fixes'
      });
    }

    // Latency recommendations
    if (report.metrics.tradeLatency.avg > 500) {
      recommendations.push({
        type: 'latency',
        severity: 'warning',
        message: 'High trade execution latency',
        action: 'Consider optimizing network paths or trading algorithms'
      });
    }

    // Scaling recommendations
    const healthyPercentage = (report.summary.status.healthy / report.summary.totalSandboxes) * 100;
    if (healthyPercentage < 70) {
      recommendations.push({
        type: 'health',
        severity: 'critical',
        message: `Only ${healthyPercentage.toFixed(1)}% of sandboxes are healthy`,
        action: 'Review sandbox configurations and consider replacing unhealthy instances'
      });
    }

    // Alert recommendations
    if (report.alerts.bySeverity.critical > 5) {
      recommendations.push({
        type: 'alerts',
        severity: 'critical',
        message: 'High number of critical alerts',
        action: 'Review and address critical issues immediately'
      });
    }

    return recommendations;
  }

  /**
   * Register a sandbox for monitoring
   */
  registerSandbox(sandboxId, sandbox) {
    this.sandboxes.set(sandboxId, {
      ...sandbox,
      status: HealthStatus.UNKNOWN,
      startTime: Date.now()
    });
    this.metrics.set(sandboxId, []);

    this.emit('sandbox:registered', { sandboxId });
    return { success: true, sandboxId };
  }

  /**
   * Unregister a sandbox
   */
  unregisterSandbox(sandboxId) {
    const existed = this.sandboxes.has(sandboxId);
    this.sandboxes.delete(sandboxId);
    this.metrics.delete(sandboxId);

    if (existed) {
      this.emit('sandbox:unregistered', { sandboxId });
    }

    return { success: existed, sandboxId };
  }

  /**
   * Create alert
   */
  createAlert(sandboxId, severity, message, details = null) {
    const alert = {
      id: `alert-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      sandboxId,
      severity,
      message,
      details,
      timestamp: new Date().toISOString()
    };

    this.alerts.push(alert);

    // Keep alerts limited
    if (this.alerts.length > 1000) {
      this.alerts = this.alerts.slice(-500);
    }

    this.emit('alert:created', alert);

    // Log critical and emergency alerts
    if (severity === AlertSeverity.CRITICAL || severity === AlertSeverity.EMERGENCY) {
      console.error(`🚨 ${severity.toUpperCase()}: ${message}`, details || '');
    }

    return alert;
  }

  /**
   * Get alerts with filtering
   */
  getAlerts(filters = {}) {
    let filtered = [...this.alerts];

    if (filters.sandboxId) {
      filtered = filtered.filter(a => a.sandboxId === filters.sandboxId);
    }

    if (filters.severity) {
      filtered = filtered.filter(a => a.severity === filters.severity);
    }

    if (filters.since) {
      const sinceTime = new Date(filters.since).getTime();
      filtered = filtered.filter(a => new Date(a.timestamp).getTime() >= sinceTime);
    }

    if (filters.limit) {
      filtered = filtered.slice(-filters.limit);
    }

    return filtered;
  }

  /**
   * Clear alerts
   */
  clearAlerts(filters = {}) {
    const beforeCount = this.alerts.length;

    if (Object.keys(filters).length === 0) {
      this.alerts = [];
    } else {
      const toRemove = new Set(this.getAlerts(filters).map(a => a.id));
      this.alerts = this.alerts.filter(a => !toRemove.has(a.id));
    }

    const removedCount = beforeCount - this.alerts.length;
    return { removed: removedCount, remaining: this.alerts.length };
  }

  // Helper methods

  recordMetric(sandboxId, metrics) {
    const history = this.metrics.get(sandboxId) || [];
    history.push({
      ...metrics,
      timestamp: Date.now()
    });

    // Keep limited history
    if (history.length > this.options.anomalyDetectionWindow) {
      history.shift();
    }

    this.metrics.set(sandboxId, history);
  }

  calculateAverageMetric(metricType) {
    let total = 0;
    let count = 0;

    for (const history of this.metrics.values()) {
      if (history.length > 0) {
        const latest = history[history.length - 1];
        if (latest[metricType] !== undefined) {
          total += latest[metricType];
          count++;
        }
      }
    }

    return count > 0 ? total / count : 0;
  }

  getSandboxesByActivity() {
    return Array.from(this.sandboxes.entries())
      .map(([id, sandbox]) => {
        const history = this.metrics.get(id) || [];
        const activityScore = history.reduce((sum, m) =>
          sum + (m.cpu || 0) + (m.memory || 0), 0
        );
        return { id, activityScore };
      })
      .sort((a, b) => a.activityScore - b.activityScore)
      .map(s => s.id);
  }

  // Integration points - to be implemented with actual E2B sandbox manager

  async getCpuUsage(sandboxId, sandbox) {
    // Simulate CPU usage
    return Math.random() * 100;
  }

  async getMemoryUsage(sandboxId, sandbox) {
    // Simulate memory usage
    return Math.random() * 100;
  }

  async getErrorRate(sandboxId) {
    // Simulate error rate
    return Math.random() * 0.05;
  }

  async getTradeLatency(sandboxId) {
    // Simulate trade latency
    return Math.random() * 500;
  }

  async getAgentResponseTime(sandboxId) {
    // Simulate agent response time
    return Math.random() * 200;
  }

  async getMarketVolatility() {
    // Simulate market volatility
    return Math.random();
  }

  async createSandbox(sandboxId) {
    // Integration point: Create actual E2B sandbox
    return {
      id: sandboxId,
      status: HealthStatus.HEALTHY,
      startTime: Date.now()
    };
  }

  async shutdownSandbox(sandboxId, options = {}) {
    // Integration point: Shutdown actual E2B sandbox
    console.log(`Shutting down sandbox ${sandboxId}`, options);
  }

  async migrateWorkload(fromSandboxId, toSandboxId) {
    // Integration point: Migrate workload between sandboxes
    console.log(`Migrating workload from ${fromSandboxId} to ${toSandboxId}`);
  }
}

module.exports = {
  E2BMonitor,
  HealthStatus,
  ScalingAction,
  AlertSeverity
};
