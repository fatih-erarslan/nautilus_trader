/**
 * E2B Monitor Integration Example
 *
 * Demonstrates how to integrate the E2B monitoring and auto-scaling system
 * with the sandbox manager and coordinator.
 */

const { E2BMonitor, HealthStatus, ScalingAction, AlertSeverity } = require('./monitor-and-scale');

/**
 * Example: Complete integration with E2B trading swarm
 */
class E2BTradingSwarmMonitor {
  constructor(sandboxManager, coordinator) {
    this.sandboxManager = sandboxManager;
    this.coordinator = coordinator;

    // Initialize monitor with custom options
    this.monitor = new E2BMonitor({
      monitorInterval: 10000, // 10 seconds
      healthCheckTimeout: 5000,
      scaleUpThreshold: 0.75,
      scaleDownThreshold: 0.25,
      maxSandboxes: 20,
      minSandboxes: 2,
      cooldownPeriod: 120000 // 2 minutes
    });

    this.setupEventHandlers();
  }

  /**
   * Setup event handlers for monitoring events
   */
  setupEventHandlers() {
    // Monitoring lifecycle events
    this.monitor.on('monitoring:started', () => {
      console.log('📊 E2B monitoring system started');
    });

    this.monitor.on('monitoring:stopped', () => {
      console.log('📊 E2B monitoring system stopped');
    });

    this.monitor.on('monitoring:complete', (results) => {
      console.log(`✅ Monitoring complete: ${results.healthy}/${results.totalSandboxes} healthy`);
    });

    // Sandbox lifecycle events
    this.monitor.on('sandbox:created', async ({ sandboxId, reason }) => {
      console.log(`➕ New sandbox created: ${sandboxId} (${reason})`);

      // Initialize sandbox with trading agents
      await this.sandboxManager.initializeSandbox(sandboxId);
      await this.coordinator.registerAgent(sandboxId);
    });

    this.monitor.on('sandbox:removed', ({ sandboxId, reason }) => {
      console.log(`➖ Sandbox removed: ${sandboxId} (${reason})`);

      // Cleanup
      this.coordinator.unregisterAgent(sandboxId);
    });

    this.monitor.on('sandbox:emergency_shutdown', ({ sandboxId }) => {
      console.log(`🚨 Emergency shutdown: ${sandboxId}`);
    });

    // Workload events
    this.monitor.on('workload:migrated', async ({ from, to }) => {
      console.log(`🔄 Workload migrated: ${from} → ${to}`);

      // Update coordinator routing
      await this.coordinator.updateRouting(from, to);
    });

    // Alert events
    this.monitor.on('alert:created', (alert) => {
      if (alert.severity === AlertSeverity.CRITICAL || alert.severity === AlertSeverity.EMERGENCY) {
        console.error(`🚨 ${alert.severity.toUpperCase()}: ${alert.message}`);

        // Send notifications (email, Slack, etc.)
        this.sendNotification(alert);
      }
    });

    // Anomaly detection
    this.monitor.on('anomaly:detected', (anomaly) => {
      console.warn(`⚠️  Anomaly detected in ${anomaly.sandboxId}: ${anomaly.metricType} = ${anomaly.value.toFixed(2)}`);

      // Log for analysis
      this.logAnomaly(anomaly);
    });

    // Error events
    this.monitor.on('error', ({ type, error, sandboxId }) => {
      console.error(`❌ Error in ${type}${sandboxId ? ` (${sandboxId})` : ''}: ${error}`);
    });
  }

  /**
   * Start the monitoring system
   */
  async start() {
    // Register existing sandboxes
    const existingSandboxes = await this.sandboxManager.listSandboxes();
    for (const sandbox of existingSandboxes) {
      this.monitor.registerSandbox(sandbox.id, sandbox);
    }

    // Start monitoring
    await this.monitor.startMonitoring();

    console.log(`🚀 E2B Trading Swarm Monitor started with ${existingSandboxes.length} sandbox(es)`);
  }

  /**
   * Stop the monitoring system
   */
  async stop() {
    await this.monitor.stopMonitoring();
    console.log('🛑 E2B Trading Swarm Monitor stopped');
  }

  /**
   * Get comprehensive health report
   */
  async getHealthReport() {
    return await this.monitor.generateHealthReport();
  }

  /**
   * Manual scaling trigger
   */
  async triggerScaling(metrics) {
    return await this.monitor.scaleBasedOnLoad(metrics);
  }

  /**
   * Get recent alerts
   */
  getAlerts(filters = {}) {
    return this.monitor.getAlerts(filters);
  }

  /**
   * Send notification (integrate with notification service)
   */
  async sendNotification(alert) {
    // Integration point: Send to Slack, email, PagerDuty, etc.
    console.log('📧 Sending notification:', alert.message);
  }

  /**
   * Log anomaly for analysis
   */
  logAnomaly(anomaly) {
    // Integration point: Send to analytics/logging service
    console.log('📊 Logging anomaly:', anomaly);
  }
}

/**
 * Example usage
 */
async function example() {
  // Mock sandbox manager and coordinator
  const sandboxManager = {
    listSandboxes: async () => [
      { id: 'sandbox-1', status: 'running' },
      { id: 'sandbox-2', status: 'running' }
    ],
    initializeSandbox: async (id) => {
      console.log(`Initializing sandbox ${id}`);
    }
  };

  const coordinator = {
    registerAgent: async (id) => {
      console.log(`Registering agent ${id}`);
    },
    unregisterAgent: (id) => {
      console.log(`Unregistering agent ${id}`);
    },
    updateRouting: async (from, to) => {
      console.log(`Updating routing: ${from} → ${to}`);
    }
  };

  // Create monitor
  const swarmMonitor = new E2BTradingSwarmMonitor(sandboxManager, coordinator);

  // Start monitoring
  await swarmMonitor.start();

  // Simulate monitoring for 30 seconds
  await new Promise(resolve => setTimeout(resolve, 30000));

  // Get health report
  const healthReport = await swarmMonitor.getHealthReport();
  console.log('📊 Health Report:', JSON.stringify(healthReport, null, 2));

  // Check for scaling recommendations
  if (healthReport.scaling.lastAction) {
    console.log('⚖️  Scaling recommendation:', healthReport.scaling.lastAction);
  }

  // Get recent critical alerts
  const criticalAlerts = swarmMonitor.getAlerts({
    severity: AlertSeverity.CRITICAL,
    limit: 5
  });
  console.log(`🚨 Critical alerts: ${criticalAlerts.length}`);

  // Stop monitoring
  await swarmMonitor.stop();
}

// Run example if executed directly
if (require.main === module) {
  example().catch(console.error);
}

module.exports = {
  E2BTradingSwarmMonitor
};
