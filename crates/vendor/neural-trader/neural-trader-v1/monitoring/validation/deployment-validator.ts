/**
 * Deployment Validation Test Suite
 * Validates neural-trader swarm deployment
 */

import { describe, it, expect, beforeAll, afterAll } from '@jest/globals';
import axios from 'axios';

interface ValidationResult {
  test: string;
  passed: boolean;
  message: string;
  duration: number;
  details?: any;
}

export class DeploymentValidator {
  private deploymentId: string;
  private sandboxIds: string[];
  private results: ValidationResult[] = [];

  constructor(
    deploymentId: string = 'neural-trader-1763096012878',
    sandboxIds: string[] = ['sandbox-1', 'sandbox-2', 'sandbox-3', 'sandbox-4', 'sandbox-5']
  ) {
    this.deploymentId = deploymentId;
    this.sandboxIds = sandboxIds;
  }

  public async runAllValidations(): Promise<{ passed: number; failed: number; results: ValidationResult[] }> {
    console.log(`\n🔍 Starting deployment validation for: ${this.deploymentId}\n`);

    await this.validateSandboxes();
    await this.validateInterAgentCommunication();
    await this.validateDistributedMemory();
    await this.validateTradingApiConnectivity();
    await this.validateQuicSync();
    await this.validateResourceLimits();
    await this.validateFailoverMechanisms();

    return this.generateReport();
  }

  private async validateSandboxes(): Promise<void> {
    console.log('📦 Validating sandboxes...');

    for (const sandboxId of this.sandboxIds) {
      const result = await this.runTest(
        `Sandbox ${sandboxId} is running`,
        async () => {
          const response = await this.makeSandboxRequest(sandboxId, '/status');

          expect(response.status).toBe('running');
          expect(response.sandboxId).toBe(sandboxId);

          return {
            status: response.status,
            uptime: response.uptime,
            version: response.version
          };
        }
      );

      this.results.push(result);
    }
  }

  private async validateInterAgentCommunication(): Promise<void> {
    console.log('🔗 Validating inter-agent communication...');

    // Test mesh topology connections
    const result = await this.runTest(
      'Mesh topology communication',
      async () => {
        const connections = await this.checkMeshConnections();

        // In mesh topology, each agent should connect to every other agent
        const expectedConnections = (this.sandboxIds.length * (this.sandboxIds.length - 1)) / 2;

        expect(connections.active).toBeGreaterThanOrEqual(expectedConnections * 0.8); // Allow 20% margin
        expect(connections.failed).toBeLessThan(3);

        return connections;
      }
    );

    this.results.push(result);

    // Test message passing
    const messagingResult = await this.runTest(
      'Agent message passing',
      async () => {
        const agent1 = this.sandboxIds[0];
        const agent2 = this.sandboxIds[1];

        const message = { type: 'test', payload: 'validation-test', timestamp: Date.now() };

        // Send message from agent1
        await this.makeSandboxRequest(agent1, '/send-message', {
          method: 'POST',
          data: { target: agent2, message }
        });

        // Check if agent2 received it
        await new Promise(resolve => setTimeout(resolve, 1000)); // Wait for propagation

        const received = await this.makeSandboxRequest(agent2, '/received-messages');

        expect(received.messages).toContainEqual(expect.objectContaining({
          type: 'test',
          payload: 'validation-test'
        }));

        return { sent: true, received: received.messages.length };
      }
    );

    this.results.push(messagingResult);
  }

  private async validateDistributedMemory(): Promise<void> {
    console.log('🧠 Validating distributed memory sync...');

    const result = await this.runTest(
      'Distributed memory synchronization',
      async () => {
        const testKey = `test-${Date.now()}`;
        const testValue = { data: 'validation', timestamp: Date.now() };

        // Write to first agent
        await this.makeSandboxRequest(this.sandboxIds[0], '/memory/write', {
          method: 'POST',
          data: { key: testKey, value: testValue }
        });

        // Wait for QUIC sync (5 seconds + buffer)
        await new Promise(resolve => setTimeout(resolve, 6000));

        // Read from all other agents
        const reads = await Promise.all(
          this.sandboxIds.slice(1).map(sandboxId =>
            this.makeSandboxRequest(sandboxId, `/memory/read?key=${testKey}`)
          )
        );

        const syncedCount = reads.filter(r =>
          r.value && r.value.data === testValue.data
        ).length;

        expect(syncedCount).toBeGreaterThanOrEqual(this.sandboxIds.length - 2); // Allow 1 failure

        return {
          totalAgents: this.sandboxIds.length - 1,
          syncedAgents: syncedCount,
          syncRate: (syncedCount / (this.sandboxIds.length - 1)) * 100
        };
      }
    );

    this.results.push(result);
  }

  private async validateTradingApiConnectivity(): Promise<void> {
    console.log('💹 Validating trading API connectivity...');

    for (const sandboxId of this.sandboxIds) {
      const result = await this.runTest(
        `${sandboxId} trading API connection`,
        async () => {
          const response = await this.makeSandboxRequest(sandboxId, '/trading-api/test');

          expect(response.connected).toBe(true);
          expect(response.latency).toBeLessThan(1000); // < 1 second latency

          return {
            connected: response.connected,
            latency: response.latency,
            provider: response.provider
          };
        }
      );

      this.results.push(result);
    }
  }

  private async validateQuicSync(): Promise<void> {
    console.log('⚡ Validating QUIC synchronization...');

    const result = await this.runTest(
      'QUIC sync interval (5 seconds)',
      async () => {
        const syncLogs = await Promise.all(
          this.sandboxIds.map(sandboxId =>
            this.makeSandboxRequest(sandboxId, '/quic/sync-logs')
          )
        );

        // Check that all agents have recent syncs
        const now = Date.now();
        const recentSyncs = syncLogs.filter(log => {
          const lastSync = new Date(log.lastSync).getTime();
          return (now - lastSync) < 10000; // Within last 10 seconds
        });

        expect(recentSyncs.length).toBe(this.sandboxIds.length);

        // Check sync intervals
        const intervals = syncLogs.map(log => log.averageInterval);
        const avgInterval = intervals.reduce((a, b) => a + b, 0) / intervals.length;

        expect(avgInterval).toBeGreaterThanOrEqual(4500); // Allow ±500ms variance
        expect(avgInterval).toBeLessThanOrEqual(5500);

        return {
          activeSyncs: recentSyncs.length,
          averageInterval: avgInterval,
          expectedInterval: 5000
        };
      }
    );

    this.results.push(result);
  }

  private async validateResourceLimits(): Promise<void> {
    console.log('📊 Validating resource limits...');

    for (const sandboxId of this.sandboxIds) {
      const result = await this.runTest(
        `${sandboxId} resource usage within limits`,
        async () => {
          const metrics = await this.makeSandboxRequest(sandboxId, '/metrics');

          expect(metrics.cpu).toBeLessThan(90); // < 90% CPU
          expect(metrics.memory).toBeLessThan(85); // < 85% memory
          expect(metrics.diskSpace).toBeGreaterThan(20); // > 20% free disk

          return metrics;
        }
      );

      this.results.push(result);
    }
  }

  private async validateFailoverMechanisms(): Promise<void> {
    console.log('🔄 Validating failover mechanisms...');

    const result = await this.runTest(
      'Agent failover and recovery',
      async () => {
        // Simulate agent failure
        const testAgent = this.sandboxIds[4]; // Last agent

        // Stop agent
        await this.makeSandboxRequest(testAgent, '/stop', { method: 'POST' });

        // Wait for detection
        await new Promise(resolve => setTimeout(resolve, 3000));

        // Check if other agents detected failure
        const healthChecks = await Promise.all(
          this.sandboxIds.slice(0, 4).map(sandboxId =>
            this.makeSandboxRequest(sandboxId, '/cluster/health')
          )
        );

        const detectedFailure = healthChecks.filter(hc =>
          hc.failedAgents && hc.failedAgents.includes(testAgent)
        );

        expect(detectedFailure.length).toBeGreaterThan(0);

        // Restart agent
        await this.makeSandboxRequest(testAgent, '/start', { method: 'POST' });

        // Wait for recovery
        await new Promise(resolve => setTimeout(resolve, 5000));

        // Verify recovery
        const recoveryStatus = await this.makeSandboxRequest(testAgent, '/status');
        expect(recoveryStatus.status).toBe('running');

        return {
          failureDetected: detectedFailure.length > 0,
          recoverySuccessful: recoveryStatus.status === 'running'
        };
      }
    );

    this.results.push(result);
  }

  private async runTest(testName: string, testFn: () => Promise<any>): Promise<ValidationResult> {
    const startTime = Date.now();

    try {
      const details = await testFn();
      const duration = Date.now() - startTime;

      console.log(`  ✅ ${testName} (${duration}ms)`);

      return {
        test: testName,
        passed: true,
        message: 'Test passed',
        duration,
        details
      };
    } catch (error) {
      const duration = Date.now() - startTime;
      const message = error instanceof Error ? error.message : String(error);

      console.log(`  ❌ ${testName} (${duration}ms)`);
      console.log(`     Error: ${message}`);

      return {
        test: testName,
        passed: false,
        message,
        duration
      };
    }
  }

  private async makeSandboxRequest(sandboxId: string, endpoint: string, options: any = {}): Promise<any> {
    try {
      // Mock implementation - replace with actual E2B API calls
      await new Promise(resolve => setTimeout(resolve, Math.random() * 100));

      // Return mock data based on endpoint
      if (endpoint === '/status') {
        return { status: 'running', sandboxId, uptime: 3600, version: '1.0.0' };
      } else if (endpoint === '/metrics') {
        return {
          cpu: Math.random() * 70,
          memory: Math.random() * 60,
          diskSpace: Math.random() * 50 + 30,
          network: Math.random() * 10
        };
      } else if (endpoint === '/trading-api/test') {
        return { connected: true, latency: Math.random() * 500, provider: 'neural-trader' };
      } else if (endpoint === '/quic/sync-logs') {
        return {
          lastSync: new Date(Date.now() - Math.random() * 5000).toISOString(),
          averageInterval: 5000 + (Math.random() - 0.5) * 200
        };
      } else if (endpoint === '/cluster/health') {
        return { failedAgents: [] };
      }

      return { success: true };
    } catch (error) {
      throw new Error(`Request to ${sandboxId}${endpoint} failed: ${error}`);
    }
  }

  private async checkMeshConnections(): Promise<{ active: number; failed: number; topology: string }> {
    // Mock mesh connection check
    await new Promise(resolve => setTimeout(resolve, 100));

    const totalPossible = (this.sandboxIds.length * (this.sandboxIds.length - 1)) / 2;

    return {
      active: totalPossible,
      failed: 0,
      topology: 'mesh'
    };
  }

  private generateReport(): { passed: number; failed: number; results: ValidationResult[] } {
    const passed = this.results.filter(r => r.passed).length;
    const failed = this.results.filter(r => !r.passed).length;

    console.log('\n' + '='.repeat(60));
    console.log('📋 VALIDATION REPORT');
    console.log('='.repeat(60));
    console.log(`Deployment ID: ${this.deploymentId}`);
    console.log(`Total Tests: ${this.results.length}`);
    console.log(`Passed: ${passed} ✅`);
    console.log(`Failed: ${failed} ❌`);
    console.log(`Success Rate: ${((passed / this.results.length) * 100).toFixed(1)}%`);
    console.log('='.repeat(60) + '\n');

    return { passed, failed, results: this.results };
  }
}

// Jest test suite
describe('Neural Trader Deployment Validation', () => {
  let validator: DeploymentValidator;

  beforeAll(() => {
    validator = new DeploymentValidator();
  });

  it('should validate all deployment components', async () => {
    const report = await validator.runAllValidations();

    expect(report.passed).toBeGreaterThan(0);
    expect(report.failed).toBe(0);
  }, 60000); // 60 second timeout
});

// CLI Entry Point
if (require.main === module) {
  const validator = new DeploymentValidator();

  validator.runAllValidations()
    .then(report => {
      process.exit(report.failed > 0 ? 1 : 0);
    })
    .catch(error => {
      console.error('Validation failed:', error);
      process.exit(1);
    });
}

export default DeploymentValidator;
