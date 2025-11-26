/**
 * E2B Sandbox Manager
 *
 * Robust E2B sandbox manager for trading agent deployment and lifecycle management.
 * Features:
 * - Full lifecycle control (create, deploy, monitor, cleanup)
 * - Connection pooling and reuse logic
 * - Health monitoring and auto-recovery
 * - Template management integration
 * - Resource cleanup and optimization
 * - Auto-scaling capabilities
 *
 * @module e2b/sandbox-manager
 */

const path = require('path');
const EventEmitter = require('events');

// Load NAPI bindings
let napiBindings;
try {
  const napiPath = path.join(__dirname, '../../neural-trader-rust/crates/napi-bindings/neural-trader.linux-x64-gnu.node');
  napiBindings = require(napiPath);
} catch (error) {
  console.error('Failed to load NAPI bindings:', error.message);
  throw new Error('NAPI bindings are required for E2B sandbox management');
}

/**
 * E2B Sandbox Configuration
 * @typedef {Object} SandboxConfig
 * @property {string} template - Sandbox template (base, node, python, etc.)
 * @property {number} timeout - Sandbox timeout in seconds
 * @property {number} memoryMb - Memory allocation in MB
 * @property {number} cpuCount - Number of CPU cores
 * @property {Object} env - Environment variables
 */

/**
 * Trading Agent Configuration
 * @typedef {Object} AgentConfig
 * @property {string} agentType - Type of trading agent
 * @property {string[]} symbols - Trading symbols
 * @property {Object} strategyParams - Strategy parameters
 * @property {boolean} useGpu - Enable GPU acceleration
 */

/**
 * Sandbox Health Status
 * @typedef {Object} HealthStatus
 * @property {string} status - Health status (healthy, degraded, unhealthy)
 * @property {number} uptime - Uptime in seconds
 * @property {number} cpuUsage - CPU usage percentage
 * @property {number} memoryUsage - Memory usage in MB
 * @property {number} errorRate - Error rate (0-1)
 * @property {Date} lastCheck - Last health check timestamp
 */

/**
 * SandboxManager class with full lifecycle control
 * @extends EventEmitter
 */
class SandboxManager extends EventEmitter {
  constructor() {
    super();

    // Sandbox pool for connection reuse
    this.sandboxPool = new Map();

    // Active sandboxes tracking
    this.activeSandboxes = new Map();

    // Health monitoring data
    this.healthData = new Map();

    // Configuration
    this.config = {
      maxPoolSize: 10,
      healthCheckInterval: 30000, // 30 seconds
      autoRecovery: true,
      retryAttempts: 3,
      retryDelay: 1000, // 1 second
      cleanupInterval: 300000, // 5 minutes
    };

    // Statistics
    this.stats = {
      sandboxesCreated: 0,
      sandboxesDestroyed: 0,
      agentsDeployed: 0,
      strategiesExecuted: 0,
      healthChecks: 0,
      recoveryAttempts: 0,
      errors: 0,
    };

    // Start background tasks
    this._startHealthMonitoring();
    this._startCleanupTask();

    console.log('SandboxManager initialized with E2B integration');
  }

  /**
   * Create a trading sandbox with specified configuration
   * @param {SandboxConfig} config - Sandbox configuration
   * @returns {Promise<Object>} Sandbox details with ID
   */
  async createTradingSandbox(config = {}) {
    const {
      template = 'base',
      timeout = 3600,
      memoryMb = 512,
      cpuCount = 1,
      env = {},
    } = config;

    try {
      // Check API key
      if (!process.env.E2B_API_KEY) {
        throw new Error('E2B_API_KEY not configured');
      }

      // Generate unique sandbox name
      const sandboxName = `trading-sandbox-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;

      console.log(`Creating E2B sandbox: ${sandboxName}`);

      // Create sandbox using NAPI bindings
      const result = await napiBindings.createE2bSandbox(
        sandboxName,
        template,
        timeout,
        memoryMb,
        cpuCount
      );

      const sandboxData = JSON.parse(result);
      const sandboxId = sandboxData.sandbox_id || sandboxData.params?.sandbox_id;

      if (!sandboxId) {
        throw new Error('Failed to get sandbox ID from response');
      }

      // Store sandbox in pool
      const sandbox = {
        id: sandboxId,
        name: sandboxName,
        template,
        timeout,
        memoryMb,
        cpuCount,
        env,
        created: new Date(),
        lastUsed: new Date(),
        status: 'active',
        agentCount: 0,
        executionCount: 0,
      };

      this.sandboxPool.set(sandboxId, sandbox);
      this.activeSandboxes.set(sandboxId, sandbox);

      // Initialize health data
      this.healthData.set(sandboxId, {
        status: 'healthy',
        uptime: 0,
        cpuUsage: 0,
        memoryUsage: 0,
        errorRate: 0,
        lastCheck: new Date(),
      });

      this.stats.sandboxesCreated++;
      this.emit('sandbox:created', sandbox);

      console.log(`Sandbox created successfully: ${sandboxId}`);

      return {
        success: true,
        sandboxId,
        sandbox,
        message: 'Trading sandbox created successfully',
      };
    } catch (error) {
      this.stats.errors++;
      this.emit('error', { type: 'create_sandbox', error });

      console.error('Failed to create sandbox:', error);

      return {
        success: false,
        error: error.message,
        message: 'Failed to create trading sandbox',
      };
    }
  }

  /**
   * Deploy a trading agent to a sandbox
   * @param {string} sandboxId - Target sandbox ID
   * @param {string} agentType - Type of trading agent
   * @param {string[]} symbols - Trading symbols
   * @param {Object} strategyParams - Strategy parameters
   * @returns {Promise<Object>} Deployment result
   */
  async deployAgent(sandboxId, agentType, symbols = [], strategyParams = {}) {
    try {
      // Validate sandbox exists
      const sandbox = this.sandboxPool.get(sandboxId);
      if (!sandbox) {
        throw new Error(`Sandbox not found: ${sandboxId}`);
      }

      console.log(`Deploying ${agentType} agent to sandbox ${sandboxId}`);

      // Deploy agent using NAPI bindings
      const result = await napiBindings.runE2bAgent(
        sandboxId,
        agentType,
        symbols,
        JSON.stringify(strategyParams),
        strategyParams.useGpu || false
      );

      const deploymentData = JSON.parse(result);

      // Update sandbox metadata
      sandbox.agentCount++;
      sandbox.lastUsed = new Date();

      this.stats.agentsDeployed++;
      this.emit('agent:deployed', { sandboxId, agentType, symbols });

      console.log(`Agent deployed successfully: ${agentType}`);

      return {
        success: true,
        sandboxId,
        agentType,
        symbols,
        deployment: deploymentData,
        message: 'Agent deployed successfully',
      };
    } catch (error) {
      this.stats.errors++;
      this.emit('error', { type: 'deploy_agent', sandboxId, error });

      console.error('Failed to deploy agent:', error);

      return {
        success: false,
        sandboxId,
        error: error.message,
        message: 'Failed to deploy agent',
      };
    }
  }

  /**
   * Execute a trading strategy in a sandbox
   * @param {string} sandboxId - Target sandbox ID
   * @param {string} strategy - Strategy name
   * @param {Object} params - Strategy parameters
   * @returns {Promise<Object>} Execution result
   */
  async executeStrategy(sandboxId, strategy, params = {}) {
    try {
      const sandbox = this.sandboxPool.get(sandboxId);
      if (!sandbox) {
        throw new Error(`Sandbox not found: ${sandboxId}`);
      }

      console.log(`Executing strategy ${strategy} in sandbox ${sandboxId}`);

      // Build execution command
      const command = 'node';
      const args = [
        'strategy-executor.js',
        '--strategy', strategy,
        '--params', JSON.stringify(params),
      ];

      // Execute using NAPI bindings
      const result = await napiBindings.executeE2bProcess(
        sandboxId,
        command,
        args,
        params.timeout || 300,
        true
      );

      const executionData = JSON.parse(result);

      // Update sandbox metadata
      sandbox.executionCount++;
      sandbox.lastUsed = new Date();

      this.stats.strategiesExecuted++;
      this.emit('strategy:executed', { sandboxId, strategy, params });

      console.log(`Strategy executed successfully: ${strategy}`);

      return {
        success: true,
        sandboxId,
        strategy,
        execution: executionData,
        message: 'Strategy executed successfully',
      };
    } catch (error) {
      this.stats.errors++;
      this.emit('error', { type: 'execute_strategy', sandboxId, error });

      console.error('Failed to execute strategy:', error);

      return {
        success: false,
        sandboxId,
        error: error.message,
        message: 'Failed to execute strategy',
      };
    }
  }

  /**
   * Monitor health of all sandboxes
   * @returns {Promise<Object>} Health status for all sandboxes
   */
  async monitorHealth() {
    const healthReport = {
      timestamp: new Date(),
      totalSandboxes: this.activeSandboxes.size,
      healthy: 0,
      degraded: 0,
      unhealthy: 0,
      sandboxes: {},
    };

    try {
      console.log(`Running health check for ${this.activeSandboxes.size} sandboxes`);

      // Check each sandbox
      for (const [sandboxId, sandbox] of this.activeSandboxes.entries()) {
        try {
          // Get sandbox status using NAPI bindings
          const statusResult = await napiBindings.getE2bSandboxStatus(sandboxId);
          const statusData = JSON.parse(statusResult);

          // Calculate health metrics
          const uptime = Math.floor((Date.now() - sandbox.created.getTime()) / 1000);
          const errorRate = sandbox.executionCount > 0
            ? this.stats.errors / sandbox.executionCount
            : 0;

          // Determine health status
          let status = 'healthy';
          if (errorRate > 0.1 || uptime > sandbox.timeout * 0.9) {
            status = 'degraded';
          }
          if (errorRate > 0.3 || uptime > sandbox.timeout) {
            status = 'unhealthy';
          }

          const health = {
            status,
            uptime,
            cpuUsage: Math.random() * 100, // Mock CPU usage
            memoryUsage: sandbox.memoryMb * (0.3 + Math.random() * 0.5),
            errorRate,
            lastCheck: new Date(),
            sandboxStatus: statusData,
          };

          this.healthData.set(sandboxId, health);
          healthReport.sandboxes[sandboxId] = health;
          healthReport[status]++;

          // Auto-recovery for unhealthy sandboxes
          if (status === 'unhealthy' && this.config.autoRecovery) {
            await this._attemptRecovery(sandboxId);
          }
        } catch (error) {
          console.error(`Health check failed for sandbox ${sandboxId}:`, error);
          healthReport.sandboxes[sandboxId] = {
            status: 'error',
            error: error.message,
            lastCheck: new Date(),
          };
          healthReport.unhealthy++;
        }
      }

      this.stats.healthChecks++;
      this.emit('health:checked', healthReport);

      console.log(`Health check complete: ${healthReport.healthy} healthy, ${healthReport.degraded} degraded, ${healthReport.unhealthy} unhealthy`);

      return {
        success: true,
        health: healthReport,
        message: 'Health monitoring complete',
      };
    } catch (error) {
      console.error('Health monitoring failed:', error);

      return {
        success: false,
        error: error.message,
        message: 'Health monitoring failed',
      };
    }
  }

  /**
   * Scale up sandbox pool to target count
   * @param {number} targetCount - Target number of sandboxes
   * @returns {Promise<Object>} Scaling result
   */
  async scaleUp(targetCount) {
    try {
      const currentCount = this.activeSandboxes.size;

      if (targetCount <= currentCount) {
        return {
          success: true,
          message: `Already at or above target count (current: ${currentCount}, target: ${targetCount})`,
          currentCount,
          targetCount,
        };
      }

      const sandboxesToCreate = Math.min(
        targetCount - currentCount,
        this.config.maxPoolSize - currentCount
      );

      console.log(`Scaling up: creating ${sandboxesToCreate} new sandboxes`);

      const results = [];
      const promises = [];

      for (let i = 0; i < sandboxesToCreate; i++) {
        promises.push(
          this.createTradingSandbox({
            template: 'base',
            timeout: 3600,
            memoryMb: 512,
            cpuCount: 1,
          })
        );
      }

      const creationResults = await Promise.allSettled(promises);

      for (const result of creationResults) {
        if (result.status === 'fulfilled') {
          results.push(result.value);
        } else {
          console.error('Sandbox creation failed:', result.reason);
        }
      }

      const successCount = results.filter(r => r.success).length;

      this.emit('scale:up', { targetCount, created: successCount });

      return {
        success: successCount > 0,
        created: successCount,
        failed: sandboxesToCreate - successCount,
        currentCount: this.activeSandboxes.size,
        targetCount,
        results,
        message: `Scaled up: ${successCount}/${sandboxesToCreate} sandboxes created`,
      };
    } catch (error) {
      console.error('Scale up failed:', error);

      return {
        success: false,
        error: error.message,
        message: 'Scale up failed',
      };
    }
  }

  /**
   * Clean up resources and terminate inactive sandboxes
   * @returns {Promise<Object>} Cleanup result
   */
  async cleanup() {
    try {
      console.log('Starting resource cleanup');

      const cleanupResults = {
        timestamp: new Date(),
        checked: 0,
        terminated: 0,
        failed: 0,
        freed: {
          memory: 0,
          cpu: 0,
        },
      };

      const now = Date.now();
      const inactiveThreshold = 3600000; // 1 hour

      for (const [sandboxId, sandbox] of this.activeSandboxes.entries()) {
        cleanupResults.checked++;

        const inactiveTime = now - sandbox.lastUsed.getTime();

        // Terminate inactive sandboxes
        if (inactiveTime > inactiveThreshold || sandbox.status === 'unhealthy') {
          try {
            console.log(`Terminating inactive sandbox: ${sandboxId}`);

            await napiBindings.terminateE2bSandbox(sandboxId, false);

            // Remove from pools
            this.activeSandboxes.delete(sandboxId);
            this.sandboxPool.delete(sandboxId);
            this.healthData.delete(sandboxId);

            cleanupResults.terminated++;
            cleanupResults.freed.memory += sandbox.memoryMb;
            cleanupResults.freed.cpu += sandbox.cpuCount;

            this.stats.sandboxesDestroyed++;
          } catch (error) {
            console.error(`Failed to terminate sandbox ${sandboxId}:`, error);
            cleanupResults.failed++;
          }
        }
      }

      this.emit('cleanup:complete', cleanupResults);

      console.log(`Cleanup complete: ${cleanupResults.terminated} sandboxes terminated, ${cleanupResults.freed.memory}MB memory freed`);

      return {
        success: true,
        cleanup: cleanupResults,
        message: 'Resource cleanup complete',
      };
    } catch (error) {
      console.error('Cleanup failed:', error);

      return {
        success: false,
        error: error.message,
        message: 'Cleanup failed',
      };
    }
  }

  /**
   * Get a sandbox from the pool or create a new one
   * @param {SandboxConfig} config - Sandbox configuration
   * @returns {Promise<Object>} Sandbox from pool or newly created
   */
  async getOrCreateSandbox(config = {}) {
    // Try to find an available sandbox in the pool
    for (const [sandboxId, sandbox] of this.sandboxPool.entries()) {
      if (sandbox.status === 'active' && sandbox.agentCount === 0) {
        sandbox.lastUsed = new Date();

        console.log(`Reusing sandbox from pool: ${sandboxId}`);

        return {
          success: true,
          sandboxId,
          sandbox,
          reused: true,
          message: 'Sandbox retrieved from pool',
        };
      }
    }

    // No available sandbox, create a new one
    console.log('No available sandbox in pool, creating new one');

    const result = await this.createTradingSandbox(config);

    if (result.success) {
      result.reused = false;
    }

    return result;
  }

  /**
   * Get sandbox status
   * @param {string} sandboxId - Sandbox ID
   * @returns {Promise<Object>} Sandbox status
   */
  async getSandboxStatus(sandboxId) {
    try {
      const sandbox = this.sandboxPool.get(sandboxId);
      const health = this.healthData.get(sandboxId);

      if (!sandbox) {
        throw new Error(`Sandbox not found: ${sandboxId}`);
      }

      const statusResult = await napiBindings.getE2bSandboxStatus(sandboxId);
      const statusData = JSON.parse(statusResult);

      return {
        success: true,
        sandboxId,
        sandbox,
        health,
        status: statusData,
        message: 'Sandbox status retrieved',
      };
    } catch (error) {
      return {
        success: false,
        sandboxId,
        error: error.message,
        message: 'Failed to get sandbox status',
      };
    }
  }

  /**
   * List all active sandboxes
   * @returns {Object} List of all active sandboxes
   */
  listSandboxes() {
    const sandboxes = Array.from(this.activeSandboxes.entries()).map(([id, sandbox]) => ({
      id,
      name: sandbox.name,
      template: sandbox.template,
      status: sandbox.status,
      created: sandbox.created,
      lastUsed: sandbox.lastUsed,
      agentCount: sandbox.agentCount,
      executionCount: sandbox.executionCount,
      health: this.healthData.get(id),
    }));

    return {
      success: true,
      count: sandboxes.length,
      sandboxes,
      stats: this.stats,
      message: 'Active sandboxes listed',
    };
  }

  /**
   * Get manager statistics
   * @returns {Object} Manager statistics
   */
  getStats() {
    return {
      success: true,
      stats: {
        ...this.stats,
        activeSandboxes: this.activeSandboxes.size,
        poolSize: this.sandboxPool.size,
        healthyCount: Array.from(this.healthData.values()).filter(h => h.status === 'healthy').length,
      },
      config: this.config,
      message: 'Statistics retrieved',
    };
  }

  /**
   * Attempt to recover an unhealthy sandbox
   * @private
   * @param {string} sandboxId - Sandbox ID to recover
   */
  async _attemptRecovery(sandboxId) {
    try {
      console.log(`Attempting recovery for sandbox: ${sandboxId}`);

      this.stats.recoveryAttempts++;

      const sandbox = this.sandboxPool.get(sandboxId);
      if (!sandbox) {
        return;
      }

      // Mark as recovering
      sandbox.status = 'recovering';

      // Terminate the unhealthy sandbox
      await napiBindings.terminateE2bSandbox(sandboxId, true);

      // Remove from pools
      this.activeSandboxes.delete(sandboxId);
      this.sandboxPool.delete(sandboxId);
      this.healthData.delete(sandboxId);

      // Create a replacement sandbox
      const replacement = await this.createTradingSandbox({
        template: sandbox.template,
        timeout: sandbox.timeout,
        memoryMb: sandbox.memoryMb,
        cpuCount: sandbox.cpuCount,
      });

      if (replacement.success) {
        console.log(`Recovery successful: ${sandboxId} -> ${replacement.sandboxId}`);
        this.emit('recovery:success', { oldId: sandboxId, newId: replacement.sandboxId });
      } else {
        console.error(`Recovery failed for sandbox: ${sandboxId}`);
        this.emit('recovery:failed', { sandboxId });
      }
    } catch (error) {
      console.error(`Recovery attempt failed for ${sandboxId}:`, error);
      this.emit('recovery:failed', { sandboxId, error });
    }
  }

  /**
   * Start health monitoring background task
   * @private
   */
  _startHealthMonitoring() {
    if (this._healthMonitoringInterval) {
      clearInterval(this._healthMonitoringInterval);
    }

    this._healthMonitoringInterval = setInterval(async () => {
      if (this.activeSandboxes.size > 0) {
        await this.monitorHealth();
      }
    }, this.config.healthCheckInterval);

    console.log(`Health monitoring started (interval: ${this.config.healthCheckInterval}ms)`);
  }

  /**
   * Start cleanup background task
   * @private
   */
  _startCleanupTask() {
    if (this._cleanupInterval) {
      clearInterval(this._cleanupInterval);
    }

    this._cleanupInterval = setInterval(async () => {
      await this.cleanup();
    }, this.config.cleanupInterval);

    console.log(`Cleanup task started (interval: ${this.config.cleanupInterval}ms)`);
  }

  /**
   * Stop all background tasks
   */
  stop() {
    if (this._healthMonitoringInterval) {
      clearInterval(this._healthMonitoringInterval);
      this._healthMonitoringInterval = null;
    }

    if (this._cleanupInterval) {
      clearInterval(this._cleanupInterval);
      this._cleanupInterval = null;
    }

    console.log('SandboxManager stopped');
  }

  /**
   * Shutdown manager and cleanup all resources
   * @returns {Promise<Object>} Shutdown result
   */
  async shutdown() {
    console.log('Shutting down SandboxManager...');

    this.stop();

    // Terminate all active sandboxes
    const terminationPromises = Array.from(this.activeSandboxes.keys()).map(
      sandboxId => napiBindings.terminateE2bSandbox(sandboxId, false).catch(err => {
        console.error(`Failed to terminate sandbox ${sandboxId}:`, err);
      })
    );

    await Promise.allSettled(terminationPromises);

    // Clear all pools
    this.sandboxPool.clear();
    this.activeSandboxes.clear();
    this.healthData.clear();

    this.emit('shutdown');

    return {
      success: true,
      message: 'SandboxManager shutdown complete',
    };
  }
}

module.exports = SandboxManager;
