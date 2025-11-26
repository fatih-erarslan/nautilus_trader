/**
 * Tests for E2B Monitor and Auto-Scaler
 */

const { E2BMonitor, HealthStatus, ScalingAction, AlertSeverity } = require('../src/e2b/monitor-and-scale');

describe('E2BMonitor', () => {
  let monitor;

  beforeEach(() => {
    monitor = new E2BMonitor({
      monitorInterval: 1000,
      scaleUpThreshold: 0.75,
      scaleDownThreshold: 0.25,
      maxSandboxes: 5,
      minSandboxes: 1
    });
  });

  afterEach(async () => {
    if (monitor.isMonitoring) {
      await monitor.stopMonitoring();
    }
  });

  describe('Monitoring Lifecycle', () => {
    test('should start monitoring', async () => {
      const result = await monitor.startMonitoring();
      expect(result.status).toBe('started');
      expect(monitor.isMonitoring).toBe(true);
    });

    test('should stop monitoring', async () => {
      await monitor.startMonitoring();
      const result = await monitor.stopMonitoring();
      expect(result.status).toBe('stopped');
      expect(monitor.isMonitoring).toBe(false);
    });

    test('should not start if already monitoring', async () => {
      await monitor.startMonitoring();
      await expect(monitor.startMonitoring()).rejects.toThrow();
    });
  });

  describe('Sandbox Registration', () => {
    test('should register sandbox', () => {
      const result = monitor.registerSandbox('test-sandbox', {
        name: 'Test Sandbox'
      });
      expect(result.success).toBe(true);
      expect(monitor.sandboxes.has('test-sandbox')).toBe(true);
    });

    test('should unregister sandbox', () => {
      monitor.registerSandbox('test-sandbox', {});
      const result = monitor.unregisterSandbox('test-sandbox');
      expect(result.success).toBe(true);
      expect(monitor.sandboxes.has('test-sandbox')).toBe(false);
    });
  });

  describe('Health Checks', () => {
    test('should check sandbox health', async () => {
      monitor.registerSandbox('test-sandbox', {});
      const health = await monitor.checkSandboxHealth('test-sandbox', {});

      expect(health).toHaveProperty('sandboxId', 'test-sandbox');
      expect(health).toHaveProperty('status');
      expect(health).toHaveProperty('metrics');
      expect(Object.values(HealthStatus)).toContain(health.status);
    });

    test('should monitor all sandboxes', async () => {
      monitor.registerSandbox('sandbox-1', {});
      monitor.registerSandbox('sandbox-2', {});

      const results = await monitor.monitorAllSandboxes();

      expect(results.totalSandboxes).toBe(2);
      expect(results.checks).toHaveLength(2);
      expect(results).toHaveProperty('healthPercentage');
    });

    test('should detect critical issues', async () => {
      monitor.registerSandbox('test-sandbox', {});

      // Mock high CPU usage
      monitor.getCpuUsage = async () => 95;

      const health = await monitor.checkSandboxHealth('test-sandbox', {});

      const criticalIssues = health.issues.filter(i => i.severity === 'critical');
      expect(criticalIssues.length).toBeGreaterThan(0);
    });
  });

  describe('Anomaly Detection', () => {
    test('should detect anomalies with sufficient data', async () => {
      monitor.registerSandbox('test-sandbox', {});

      // Generate normal metrics
      for (let i = 0; i < 20; i++) {
        monitor.recordMetric('test-sandbox', {
          cpu: 50 + Math.random() * 5,
          memory: 50 + Math.random() * 5
        });
      }

      // Add anomalous metric
      monitor.recordMetric('test-sandbox', {
        cpu: 95, // Anomaly
        memory: 50
      });

      const anomalies = await monitor.detectAnomalies();
      expect(anomalies.length).toBeGreaterThan(0);
    });

    test('should not detect anomalies with insufficient data', async () => {
      monitor.registerSandbox('test-sandbox', {});

      // Only a few metrics
      for (let i = 0; i < 5; i++) {
        monitor.recordMetric('test-sandbox', { cpu: 50, memory: 50 });
      }

      const anomalies = await monitor.detectAnomalies();
      expect(anomalies.length).toBe(0);
    });
  });

  describe('Scaling Evaluation', () => {
    test('should recommend scale up on high load', async () => {
      monitor.registerSandbox('sandbox-1', {});

      // Mock high resource usage
      monitor.calculateAverageMetric = (metric) => {
        if (metric === 'cpu' || metric === 'memory') return 0.85;
        return 0;
      };

      const decision = await monitor.evaluateScaling({
        totalSandboxes: 1,
        healthy: 1,
        degraded: 0,
        critical: 0,
        failed: 0
      });

      expect(decision.action).toBe(ScalingAction.SCALE_UP);
    });

    test('should recommend scale down on low load', async () => {
      monitor.registerSandbox('sandbox-1', {});
      monitor.registerSandbox('sandbox-2', {});

      // Mock low resource usage
      monitor.calculateAverageMetric = (metric) => 0.2;
      monitor.getMarketVolatility = async () => 0.2;

      // Bypass cooldown
      monitor.lastScalingTime = 0;

      const decision = await monitor.evaluateScaling({
        totalSandboxes: 2,
        healthy: 2,
        degraded: 0,
        critical: 0,
        failed: 0
      });

      expect(decision.action).toBe(ScalingAction.SCALE_DOWN);
    });

    test('should recommend emergency shutdown on high failures', async () => {
      const decision = await monitor.evaluateScaling({
        totalSandboxes: 5,
        healthy: 1,
        degraded: 0,
        critical: 0,
        failed: 4
      });

      expect(decision.action).toBe(ScalingAction.EMERGENCY_SHUTDOWN);
    });

    test('should respect cooldown period', async () => {
      monitor.lastScalingTime = Date.now();
      monitor.calculateAverageMetric = () => 0.9;

      const decision = await monitor.evaluateScaling({
        totalSandboxes: 1,
        healthy: 1,
        degraded: 0,
        critical: 0,
        failed: 0
      });

      expect(decision.action).toBe(ScalingAction.MAINTAIN);
      expect(decision.reason).toContain('cooldown');
    });
  });

  describe('Scaling Actions', () => {
    test('should scale up', async () => {
      monitor.registerSandbox('sandbox-1', {});

      const decision = {
        action: ScalingAction.SCALE_UP,
        targetCount: 3,
        reason: 'High load'
      };

      const result = await monitor.scaleUp(decision);

      expect(result.action).toBe('scale_up');
      expect(result.created).toBe(2);
      expect(monitor.sandboxes.size).toBe(3);
    });

    test('should scale down', async () => {
      monitor.registerSandbox('sandbox-1', {});
      monitor.registerSandbox('sandbox-2', {});
      monitor.registerSandbox('sandbox-3', {});

      const decision = {
        action: ScalingAction.SCALE_DOWN,
        targetCount: 1,
        reason: 'Low load'
      };

      const result = await monitor.scaleDown(decision);

      expect(result.action).toBe('scale_down');
      expect(result.removed).toBe(2);
      expect(monitor.sandboxes.size).toBe(1);
    });

    test('should execute failover', async () => {
      monitor.registerSandbox('sandbox-1', { status: HealthStatus.HEALTHY });
      monitor.registerSandbox('sandbox-2', { status: HealthStatus.CRITICAL });

      const decision = {
        action: ScalingAction.FAILOVER,
        reason: 'High critical rate'
      };

      const result = await monitor.executeFailover(decision);

      expect(result.action).toBe('failover');
      expect(result.migrated).toBeGreaterThanOrEqual(0);
    });

    test('should execute emergency shutdown', async () => {
      monitor.registerSandbox('sandbox-1', {});
      monitor.registerSandbox('sandbox-2', {});

      const decision = {
        action: ScalingAction.EMERGENCY_SHUTDOWN,
        reason: 'System failure'
      };

      const result = await monitor.emergencyShutdown(decision);

      expect(result.action).toBe('emergency_shutdown');
      expect(monitor.sandboxes.size).toBe(0);
      expect(monitor.isMonitoring).toBe(false);
    });
  });

  describe('Alert Management', () => {
    test('should create alert', () => {
      const alert = monitor.createAlert(
        'test-sandbox',
        AlertSeverity.WARNING,
        'Test alert'
      );

      expect(alert).toHaveProperty('id');
      expect(alert.severity).toBe(AlertSeverity.WARNING);
      expect(alert.message).toBe('Test alert');
      expect(monitor.alerts).toContain(alert);
    });

    test('should filter alerts', () => {
      monitor.createAlert('sandbox-1', AlertSeverity.INFO, 'Info 1');
      monitor.createAlert('sandbox-1', AlertSeverity.WARNING, 'Warning 1');
      monitor.createAlert('sandbox-2', AlertSeverity.CRITICAL, 'Critical 1');

      const criticalAlerts = monitor.getAlerts({ severity: AlertSeverity.CRITICAL });
      expect(criticalAlerts).toHaveLength(1);

      const sandbox1Alerts = monitor.getAlerts({ sandboxId: 'sandbox-1' });
      expect(sandbox1Alerts).toHaveLength(2);
    });

    test('should clear alerts', () => {
      monitor.createAlert('sandbox-1', AlertSeverity.INFO, 'Info 1');
      monitor.createAlert('sandbox-2', AlertSeverity.WARNING, 'Warning 1');

      const result = monitor.clearAlerts({ sandboxId: 'sandbox-1' });

      expect(result.removed).toBe(1);
      expect(result.remaining).toBe(1);
    });
  });

  describe('Health Report', () => {
    test('should generate comprehensive health report', async () => {
      monitor.registerSandbox('sandbox-1', { status: HealthStatus.HEALTHY });
      monitor.registerSandbox('sandbox-2', { status: HealthStatus.DEGRADED });

      monitor.recordMetric('sandbox-1', { cpu: 50, memory: 60 });
      monitor.recordMetric('sandbox-2', { cpu: 80, memory: 70 });

      monitor.createAlert('sandbox-2', AlertSeverity.WARNING, 'High CPU');

      const report = await monitor.generateHealthReport();

      expect(report).toHaveProperty('timestamp');
      expect(report).toHaveProperty('summary');
      expect(report).toHaveProperty('metrics');
      expect(report).toHaveProperty('alerts');
      expect(report).toHaveProperty('sandboxes');
      expect(report).toHaveProperty('recommendations');

      expect(report.summary.totalSandboxes).toBe(2);
      expect(report.sandboxes).toHaveLength(2);
    });

    test('should generate recommendations', async () => {
      monitor.registerSandbox('sandbox-1', {});

      // Simulate high CPU
      monitor.recordMetric('sandbox-1', {
        cpu: 85,
        memory: 50,
        errorRate: 0.001
      });

      const report = await monitor.generateHealthReport();

      expect(report.recommendations.length).toBeGreaterThan(0);
      const cpuRec = report.recommendations.find(r => r.type === 'cpu');
      expect(cpuRec).toBeDefined();
    });
  });

  describe('Event Emission', () => {
    test('should emit monitoring events', async () => {
      const events = [];
      monitor.on('monitoring:started', () => events.push('started'));
      monitor.on('monitoring:stopped', () => events.push('stopped'));

      await monitor.startMonitoring();
      await monitor.stopMonitoring();

      expect(events).toContain('started');
      expect(events).toContain('stopped');
    });

    test('should emit alert events', () => {
      const alerts = [];
      monitor.on('alert:created', (alert) => alerts.push(alert));

      monitor.createAlert('sandbox-1', AlertSeverity.CRITICAL, 'Test');

      expect(alerts).toHaveLength(1);
      expect(alerts[0].severity).toBe(AlertSeverity.CRITICAL);
    });

    test('should emit anomaly events', async () => {
      const anomalies = [];
      monitor.on('anomaly:detected', (anomaly) => anomalies.push(anomaly));

      monitor.registerSandbox('test-sandbox', {});

      // Generate metrics to trigger anomaly
      for (let i = 0; i < 20; i++) {
        monitor.recordMetric('test-sandbox', { cpu: 50 });
      }
      monitor.recordMetric('test-sandbox', { cpu: 95 });

      await monitor.detectAnomalies();

      expect(anomalies.length).toBeGreaterThan(0);
    });
  });
});
