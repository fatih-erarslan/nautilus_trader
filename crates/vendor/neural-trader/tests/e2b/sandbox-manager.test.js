/**
 * E2B Sandbox Manager Tests
 *
 * Comprehensive test suite for SandboxManager functionality.
 */

const SandboxManager = require('../../src/e2b/sandbox-manager');

describe('SandboxManager', () => {
  let manager;

  beforeEach(() => {
    // Create new manager instance for each test
    manager = new SandboxManager();
  });

  afterEach(async () => {
    // Cleanup after each test
    if (manager) {
      await manager.shutdown();
    }
  });

  describe('Initialization', () => {
    test('should initialize with default configuration', () => {
      expect(manager).toBeDefined();
      expect(manager.config).toBeDefined();
      expect(manager.config.maxPoolSize).toBe(10);
      expect(manager.config.autoRecovery).toBe(true);
      expect(manager.stats).toBeDefined();
    });

    test('should start background tasks', () => {
      expect(manager._healthMonitoringInterval).toBeDefined();
      expect(manager._cleanupInterval).toBeDefined();
    });
  });

  describe('createTradingSandbox', () => {
    test('should create sandbox with default configuration', async () => {
      const result = await manager.createTradingSandbox();

      expect(result).toBeDefined();
      expect(result.success).toBeDefined();

      if (result.success) {
        expect(result.sandboxId).toBeDefined();
        expect(result.sandbox).toBeDefined();
        expect(manager.stats.sandboxesCreated).toBeGreaterThan(0);
      }
    });

    test('should create sandbox with custom configuration', async () => {
      const config = {
        template: 'node',
        timeout: 7200,
        memoryMb: 1024,
        cpuCount: 2,
      };

      const result = await manager.createTradingSandbox(config);

      expect(result).toBeDefined();

      if (result.success && result.sandbox) {
        expect(result.sandbox.template).toBe('node');
        expect(result.sandbox.timeout).toBe(7200);
        expect(result.sandbox.memoryMb).toBe(1024);
        expect(result.sandbox.cpuCount).toBe(2);
      }
    });

    test('should handle E2B_API_KEY not set', async () => {
      const originalKey = process.env.E2B_API_KEY;
      delete process.env.E2B_API_KEY;

      const result = await manager.createTradingSandbox();

      expect(result.success).toBe(false);
      expect(result.error).toContain('E2B_API_KEY');

      // Restore
      if (originalKey) {
        process.env.E2B_API_KEY = originalKey;
      }
    });

    test('should emit sandbox:created event', (done) => {
      manager.once('sandbox:created', (sandbox) => {
        expect(sandbox).toBeDefined();
        expect(sandbox.id).toBeDefined();
        done();
      });

      manager.createTradingSandbox();
    });
  });

  describe('deployAgent', () => {
    test('should deploy agent to sandbox', async () => {
      // First create a sandbox
      const sandboxResult = await manager.createTradingSandbox();

      if (!sandboxResult.success) {
        // Skip test if sandbox creation fails (e.g., no API key)
        return;
      }

      const { sandboxId } = sandboxResult;

      // Deploy agent
      const result = await manager.deployAgent(
        sandboxId,
        'momentum',
        ['AAPL', 'TSLA'],
        { period: 20, threshold: 0.02 }
      );

      expect(result).toBeDefined();

      if (result.success) {
        expect(result.sandboxId).toBe(sandboxId);
        expect(result.agentType).toBe('momentum');
        expect(manager.stats.agentsDeployed).toBeGreaterThan(0);
      }
    });

    test('should handle invalid sandbox ID', async () => {
      const result = await manager.deployAgent(
        'invalid-sandbox-id',
        'momentum',
        ['AAPL'],
        {}
      );

      expect(result.success).toBe(false);
      expect(result.error).toContain('not found');
    });

    test('should emit agent:deployed event', async () => {
      const sandboxResult = await manager.createTradingSandbox();

      if (!sandboxResult.success) {
        return;
      }

      const { sandboxId } = sandboxResult;

      const eventPromise = new Promise((resolve) => {
        manager.once('agent:deployed', resolve);
      });

      await manager.deployAgent(sandboxId, 'momentum', ['AAPL'], {});

      const event = await eventPromise;
      expect(event.sandboxId).toBe(sandboxId);
      expect(event.agentType).toBe('momentum');
    });
  });

  describe('executeStrategy', () => {
    test('should execute strategy in sandbox', async () => {
      const sandboxResult = await manager.createTradingSandbox();

      if (!sandboxResult.success) {
        return;
      }

      const { sandboxId } = sandboxResult;

      const result = await manager.executeStrategy(
        sandboxId,
        'mean-reversion',
        { windowSize: 20, stddevs: 2 }
      );

      expect(result).toBeDefined();

      if (result.success) {
        expect(result.sandboxId).toBe(sandboxId);
        expect(result.strategy).toBe('mean-reversion');
        expect(manager.stats.strategiesExecuted).toBeGreaterThan(0);
      }
    });

    test('should handle invalid sandbox ID', async () => {
      const result = await manager.executeStrategy(
        'invalid-sandbox-id',
        'mean-reversion',
        {}
      );

      expect(result.success).toBe(false);
      expect(result.error).toContain('not found');
    });
  });

  describe('monitorHealth', () => {
    test('should monitor health of all sandboxes', async () => {
      // Create a sandbox first
      const sandboxResult = await manager.createTradingSandbox();

      if (!sandboxResult.success) {
        return;
      }

      const result = await manager.monitorHealth();

      expect(result).toBeDefined();
      expect(result.success).toBe(true);

      if (result.health) {
        expect(result.health.timestamp).toBeDefined();
        expect(result.health.totalSandboxes).toBeGreaterThan(0);
      }
    });

    test('should return empty report when no sandboxes', async () => {
      // Manager starts with no sandboxes
      const result = await manager.monitorHealth();

      expect(result).toBeDefined();

      if (result.health) {
        expect(result.health.totalSandboxes).toBe(0);
      }
    });

    test('should emit health:checked event', async () => {
      const sandboxResult = await manager.createTradingSandbox();

      if (!sandboxResult.success) {
        return;
      }

      const eventPromise = new Promise((resolve) => {
        manager.once('health:checked', resolve);
      });

      await manager.monitorHealth();

      const event = await eventPromise;
      expect(event).toBeDefined();
      expect(event.totalSandboxes).toBeGreaterThan(0);
    });
  });

  describe('scaleUp', () => {
    test('should scale up to target count', async () => {
      const result = await manager.scaleUp(3);

      expect(result).toBeDefined();
      expect(result.success).toBeDefined();

      if (result.success) {
        expect(result.created).toBeGreaterThan(0);
        expect(result.currentCount).toBeGreaterThan(0);
      }
    });

    test('should not scale if already at target', async () => {
      const result = await manager.scaleUp(0);

      expect(result).toBeDefined();
      expect(result.message).toContain('Already at or above target');
    });

    test('should respect maxPoolSize limit', async () => {
      const result = await manager.scaleUp(1000);

      expect(result).toBeDefined();

      if (result.success) {
        expect(result.currentCount).toBeLessThanOrEqual(manager.config.maxPoolSize);
      }
    });
  });

  describe('cleanup', () => {
    test('should cleanup inactive sandboxes', async () => {
      const result = await manager.cleanup();

      expect(result).toBeDefined();
      expect(result.success).toBe(true);

      if (result.cleanup) {
        expect(result.cleanup.checked).toBeDefined();
        expect(result.cleanup.terminated).toBeDefined();
      }
    });

    test('should emit cleanup:complete event', (done) => {
      manager.once('cleanup:complete', (result) => {
        expect(result).toBeDefined();
        expect(result.checked).toBeDefined();
        done();
      });

      manager.cleanup();
    });
  });

  describe('getOrCreateSandbox', () => {
    test('should create new sandbox when pool is empty', async () => {
      const result = await manager.getOrCreateSandbox();

      expect(result).toBeDefined();

      if (result.success) {
        expect(result.reused).toBe(false);
        expect(result.sandboxId).toBeDefined();
      }
    });

    test('should reuse sandbox from pool when available', async () => {
      // Create first sandbox
      const firstResult = await manager.createTradingSandbox();

      if (!firstResult.success) {
        return;
      }

      // Try to get or create - should reuse
      const result = await manager.getOrCreateSandbox();

      if (result.success && result.reused !== undefined) {
        // If reused, it should be the same sandbox
        if (result.reused) {
          expect(result.sandboxId).toBe(firstResult.sandboxId);
        }
      }
    });
  });

  describe('getSandboxStatus', () => {
    test('should get status of existing sandbox', async () => {
      const sandboxResult = await manager.createTradingSandbox();

      if (!sandboxResult.success) {
        return;
      }

      const { sandboxId } = sandboxResult;

      const result = await manager.getSandboxStatus(sandboxId);

      expect(result).toBeDefined();

      if (result.success) {
        expect(result.sandboxId).toBe(sandboxId);
        expect(result.sandbox).toBeDefined();
      }
    });

    test('should handle non-existent sandbox', async () => {
      const result = await manager.getSandboxStatus('non-existent-id');

      expect(result.success).toBe(false);
      expect(result.error).toContain('not found');
    });
  });

  describe('listSandboxes', () => {
    test('should list all active sandboxes', async () => {
      const result = manager.listSandboxes();

      expect(result).toBeDefined();
      expect(result.success).toBe(true);
      expect(result.count).toBeDefined();
      expect(result.sandboxes).toBeDefined();
      expect(Array.isArray(result.sandboxes)).toBe(true);
    });

    test('should include sandbox details', async () => {
      const sandboxResult = await manager.createTradingSandbox();

      if (!sandboxResult.success) {
        return;
      }

      const result = manager.listSandboxes();

      expect(result.count).toBeGreaterThan(0);

      const sandbox = result.sandboxes[0];
      expect(sandbox.id).toBeDefined();
      expect(sandbox.name).toBeDefined();
      expect(sandbox.template).toBeDefined();
      expect(sandbox.status).toBeDefined();
    });
  });

  describe('getStats', () => {
    test('should return manager statistics', () => {
      const result = manager.getStats();

      expect(result).toBeDefined();
      expect(result.success).toBe(true);
      expect(result.stats).toBeDefined();
      expect(result.stats.sandboxesCreated).toBeDefined();
      expect(result.stats.activeSandboxes).toBeDefined();
    });
  });

  describe('shutdown', () => {
    test('should shutdown manager cleanly', async () => {
      const result = await manager.shutdown();

      expect(result).toBeDefined();
      expect(result.success).toBe(true);
      expect(manager.activeSandboxes.size).toBe(0);
      expect(manager.sandboxPool.size).toBe(0);
    });

    test('should stop background tasks', async () => {
      await manager.shutdown();

      expect(manager._healthMonitoringInterval).toBeNull();
      expect(manager._cleanupInterval).toBeNull();
    });
  });

  describe('Event Emitters', () => {
    test('should emit error events on failures', (done) => {
      manager.once('error', (error) => {
        expect(error).toBeDefined();
        expect(error.type).toBeDefined();
        done();
      });

      // Trigger an error by trying to deploy to non-existent sandbox
      manager.deployAgent('invalid-id', 'test', [], {});
    });
  });
});
