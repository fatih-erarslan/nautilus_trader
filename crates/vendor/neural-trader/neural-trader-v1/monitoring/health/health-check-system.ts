/**
 * Health Check System for Neural Trader Swarm
 * Monitors sandbox health, agent responsiveness, and QUIC sync
 */

import axios from 'axios';
import { EventEmitter } from 'events';

interface HealthCheckConfig {
  deploymentId: string;
  sandboxIds: string[];
  checkInterval: number;
  timeout: number;
  maxRetries: number;
  quicSyncInterval: number;
}

interface HealthStatus {
  sandboxId: string;
  agentId: string;
  isHealthy: boolean;
  lastCheck: Date;
  responseTime: number;
  consecutiveFailures: number;
  checks: {
    sandboxResponsive: boolean;
    agentResponsive: boolean;
    quicSyncActive: boolean;
    apiConnectivity: boolean;
    resourcesHealthy: boolean;
  };
  metrics: {
    cpu: number;
    memory: number;
    diskSpace: number;
    networkLatency: number;
  };
  errors: string[];
}

export class HealthCheckSystem extends EventEmitter {
  private config: HealthCheckConfig;
  private healthStatuses: Map<string, HealthStatus> = new Map();
  private checkIntervalId?: NodeJS.Timeout;
  private alertThresholds = {
    cpu: 90,
    memory: 85,
    consecutiveFailures: 3,
    responseTime: 5000
  };

  constructor(config: Partial<HealthCheckConfig> = {}) {
    super();

    this.config = {
      deploymentId: config.deploymentId || 'neural-trader-1763096012878',
      sandboxIds: config.sandboxIds || [
        'sandbox-1', 'sandbox-2', 'sandbox-3', 'sandbox-4', 'sandbox-5'
      ],
      checkInterval: config.checkInterval || 60000, // 60 seconds
      timeout: config.timeout || 10000,
      maxRetries: config.maxRetries || 3,
      quicSyncInterval: config.quicSyncInterval || 5000
    };

    this.initializeHealthStatuses();
  }

  private initializeHealthStatuses(): void {
    this.config.sandboxIds.forEach((sandboxId, index) => {
      this.healthStatuses.set(sandboxId, {
        sandboxId,
        agentId: `agent-${index + 1}`,
        isHealthy: true,
        lastCheck: new Date(),
        responseTime: 0,
        consecutiveFailures: 0,
        checks: {
          sandboxResponsive: true,
          agentResponsive: true,
          quicSyncActive: true,
          apiConnectivity: true,
          resourcesHealthy: true
        },
        metrics: {
          cpu: 0,
          memory: 0,
          diskSpace: 0,
          networkLatency: 0
        },
        errors: []
      });
    });
  }

  public async start(): Promise<void> {
    console.log(`Starting health check system for deployment: ${this.config.deploymentId}`);
    console.log(`Monitoring ${this.config.sandboxIds.length} sandboxes`);
    console.log(`Check interval: ${this.config.checkInterval}ms`);

    // Run initial check
    await this.runHealthChecks();

    // Schedule periodic checks
    this.checkIntervalId = setInterval(
      () => this.runHealthChecks(),
      this.config.checkInterval
    );

    this.emit('started', {
      deploymentId: this.config.deploymentId,
      sandboxCount: this.config.sandboxIds.length
    });
  }

  public stop(): void {
    if (this.checkIntervalId) {
      clearInterval(this.checkIntervalId);
      this.checkIntervalId = undefined;
    }

    this.emit('stopped');
    console.log('Health check system stopped');
  }

  private async runHealthChecks(): Promise<void> {
    console.log(`\n[${new Date().toISOString()}] Running health checks...`);

    const checkPromises = this.config.sandboxIds.map(sandboxId =>
      this.checkSandboxHealth(sandboxId)
    );

    await Promise.allSettled(checkPromises);

    // Emit aggregate status
    const aggregate = this.getAggregateStatus();
    this.emit('health-check-complete', aggregate);

    // Log summary
    console.log(`Health check complete: ${aggregate.healthyCount}/${aggregate.totalCount} healthy`);
  }

  private async checkSandboxHealth(sandboxId: string): Promise<void> {
    const status = this.healthStatuses.get(sandboxId);
    if (!status) return;

    const startTime = Date.now();
    status.errors = [];

    try {
      // 1. Check sandbox responsiveness
      status.checks.sandboxResponsive = await this.checkSandboxResponsiveness(sandboxId);

      // 2. Check agent responsiveness
      status.checks.agentResponsive = await this.checkAgentResponsiveness(sandboxId);

      // 3. Check QUIC sync status
      status.checks.quicSyncActive = await this.checkQuicSync(sandboxId);

      // 4. Check API connectivity
      status.checks.apiConnectivity = await this.checkApiConnectivity(sandboxId);

      // 5. Check resource health
      status.checks.resourcesHealthy = await this.checkResources(sandboxId);

      // Calculate response time
      status.responseTime = Date.now() - startTime;

      // Determine overall health
      status.isHealthy = Object.values(status.checks).every(check => check);

      if (status.isHealthy) {
        status.consecutiveFailures = 0;
        this.emit('sandbox-healthy', { sandboxId, status });
      } else {
        status.consecutiveFailures++;
        this.emit('sandbox-unhealthy', { sandboxId, status });

        // Raise alert if threshold exceeded
        if (status.consecutiveFailures >= this.alertThresholds.consecutiveFailures) {
          this.raiseAlert('critical', sandboxId,
            `Sandbox failed ${status.consecutiveFailures} consecutive health checks`);
        }
      }

    } catch (error) {
      status.isHealthy = false;
      status.consecutiveFailures++;
      status.errors.push(error instanceof Error ? error.message : String(error));

      this.emit('sandbox-error', { sandboxId, error });
    } finally {
      status.lastCheck = new Date();
      this.healthStatuses.set(sandboxId, status);
    }
  }

  private async checkSandboxResponsiveness(sandboxId: string): Promise<boolean> {
    try {
      // Simulate sandbox ping - replace with actual E2B API call
      const response = await this.makeRequest(`/sandboxes/${sandboxId}/ping`);
      return response.status === 'ok';
    } catch (error) {
      return false;
    }
  }

  private async checkAgentResponsiveness(sandboxId: string): Promise<boolean> {
    try {
      // Check if agent process is running
      const response = await this.makeRequest(`/sandboxes/${sandboxId}/processes`);
      return response.agents && response.agents.length > 0;
    } catch (error) {
      return false;
    }
  }

  private async checkQuicSync(sandboxId: string): Promise<boolean> {
    try {
      // Verify QUIC sync is active and recent
      const response = await this.makeRequest(`/sandboxes/${sandboxId}/quic-status`);
      const lastSync = new Date(response.lastSync);
      const timeSinceSync = Date.now() - lastSync.getTime();

      // Sync should happen within 2x the interval
      return timeSinceSync < (this.config.quicSyncInterval * 2);
    } catch (error) {
      return false;
    }
  }

  private async checkApiConnectivity(sandboxId: string): Promise<boolean> {
    try {
      // Test connection to trading API
      const response = await this.makeRequest(`/sandboxes/${sandboxId}/api-test`);
      return response.connected === true;
    } catch (error) {
      return false;
    }
  }

  private async checkResources(sandboxId: string): Promise<boolean> {
    try {
      const status = this.healthStatuses.get(sandboxId);
      if (!status) return false;

      // Get resource metrics
      const response = await this.makeRequest(`/sandboxes/${sandboxId}/metrics`);

      status.metrics = {
        cpu: response.cpu || 0,
        memory: response.memory || 0,
        diskSpace: response.disk || 0,
        networkLatency: response.latency || 0
      };

      // Check against thresholds
      if (status.metrics.cpu > this.alertThresholds.cpu) {
        this.raiseAlert('warning', sandboxId,
          `High CPU usage: ${status.metrics.cpu}%`);
        return false;
      }

      if (status.metrics.memory > this.alertThresholds.memory) {
        this.raiseAlert('warning', sandboxId,
          `High memory usage: ${status.metrics.memory}%`);
        return false;
      }

      return true;
    } catch (error) {
      return false;
    }
  }

  private async makeRequest(endpoint: string, retries = 0): Promise<any> {
    try {
      // Simulate API request - replace with actual implementation
      // For now, return mock data
      await new Promise(resolve => setTimeout(resolve, Math.random() * 100));

      return {
        status: 'ok',
        connected: true,
        agents: [{ id: 'agent-1', status: 'running' }],
        lastSync: new Date().toISOString(),
        cpu: Math.random() * 100,
        memory: Math.random() * 100,
        disk: Math.random() * 100,
        latency: Math.random() * 50
      };
    } catch (error) {
      if (retries < this.config.maxRetries) {
        await new Promise(resolve => setTimeout(resolve, 1000));
        return this.makeRequest(endpoint, retries + 1);
      }
      throw error;
    }
  }

  private raiseAlert(level: 'info' | 'warning' | 'critical', sandboxId: string, message: string): void {
    const alert = {
      level,
      sandboxId,
      message,
      timestamp: new Date()
    };

    this.emit('alert', alert);
    console.log(`[${level.toUpperCase()}] ${sandboxId}: ${message}`);
  }

  public getAggregateStatus() {
    const statuses = Array.from(this.healthStatuses.values());

    return {
      timestamp: new Date(),
      deploymentId: this.config.deploymentId,
      totalCount: statuses.length,
      healthyCount: statuses.filter(s => s.isHealthy).length,
      unhealthyCount: statuses.filter(s => !s.isHealthy).length,
      averageResponseTime: statuses.reduce((sum, s) => sum + s.responseTime, 0) / statuses.length,
      averageCpu: statuses.reduce((sum, s) => sum + s.metrics.cpu, 0) / statuses.length,
      averageMemory: statuses.reduce((sum, s) => sum + s.metrics.memory, 0) / statuses.length,
      criticalIssues: statuses.filter(s => s.consecutiveFailures >= this.alertThresholds.consecutiveFailures).length,
      details: statuses
    };
  }

  public getSandboxStatus(sandboxId: string): HealthStatus | undefined {
    return this.healthStatuses.get(sandboxId);
  }

  public async performManualCheck(sandboxId: string): Promise<HealthStatus | undefined> {
    await this.checkSandboxHealth(sandboxId);
    return this.getSandboxStatus(sandboxId);
  }
}

// CLI Entry Point
if (require.main === module) {
  const healthSystem = new HealthCheckSystem();

  healthSystem.on('started', (info) => {
    console.log('Health check system started:', info);
  });

  healthSystem.on('health-check-complete', (aggregate) => {
    console.log('\n=== Health Check Summary ===');
    console.log(`Healthy: ${aggregate.healthyCount}/${aggregate.totalCount}`);
    console.log(`Avg Response Time: ${aggregate.averageResponseTime.toFixed(0)}ms`);
    console.log(`Avg CPU: ${aggregate.averageCpu.toFixed(1)}%`);
    console.log(`Avg Memory: ${aggregate.averageMemory.toFixed(1)}%`);
    console.log(`Critical Issues: ${aggregate.criticalIssues}`);
  });

  healthSystem.on('alert', (alert) => {
    console.log(`\n⚠️  ALERT [${alert.level}] - ${alert.sandboxId}`);
    console.log(`   ${alert.message}`);
  });

  healthSystem.start();

  // Graceful shutdown
  process.on('SIGINT', () => {
    console.log('\nShutting down health check system...');
    healthSystem.stop();
    process.exit(0);
  });
}

export default HealthCheckSystem;
