/**
 * Tests for swarm coordinator
 */

import { SwarmCoordinator } from '../src/swarm-coordinator';

describe('SwarmCoordinator', () => {
  let coordinator: SwarmCoordinator;

  beforeEach(() => {
    coordinator = new SwarmCoordinator({
      maxAgents: 5,
      topology: 'star',
      communicationProtocol: 'sync',
      timeout: 5000,
    });
  });

  describe('executeVariations', () => {
    it('should execute variations successfully', async () => {
      const task = {
        execute: async (params: any) => {
          return params.value * 2;
        },
      };

      const variations = [
        { id: 'v1', parameters: { value: 1 } },
        { id: 'v2', parameters: { value: 2 } },
        { id: 'v3', parameters: { value: 3 } },
      ];

      const results = await coordinator.executeVariations(variations, task);

      expect(results).toHaveLength(3);
      expect(results[0].success).toBe(true);
      expect(results[0].result).toBe(2);
      expect(results[1].result).toBe(4);
      expect(results[2].result).toBe(6);
    });

    it('should handle task failures gracefully', async () => {
      const task = {
        execute: async (params: any) => {
          if (params.shouldFail) {
            throw new Error('Task failed');
          }
          return params.value;
        },
      };

      const variations = [
        { id: 'v1', parameters: { value: 1, shouldFail: false } },
        { id: 'v2', parameters: { value: 2, shouldFail: true } },
      ];

      const results = await coordinator.executeVariations(variations, task);

      expect(results[0].success).toBe(true);
      expect(results[1].success).toBe(false);
      expect(results[1].error).toBeDefined();
    });

    it('should respect timeout', async () => {
      const task = {
        execute: async () => {
          await new Promise(resolve => setTimeout(resolve, 10000));
          return 'done';
        },
      };

      const variations = [{ id: 'v1', parameters: {} }];

      const results = await coordinator.executeVariations(variations, task);

      expect(results[0].success).toBe(false);
      expect(results[0].error?.message).toContain('timeout');
    });
  });

  describe('getStatistics', () => {
    it('should calculate statistics correctly', async () => {
      const task = {
        execute: async (params: any) => params.value,
      };

      const variations = [
        { id: 'v1', parameters: { value: 1 } },
        { id: 'v2', parameters: { value: 2 } },
      ];

      await coordinator.executeVariations(variations, task);

      const stats = coordinator.getStatistics();

      expect(stats.totalVariations).toBe(2);
      expect(stats.successfulVariations).toBe(2);
      expect(stats.failedVariations).toBe(0);
      expect(stats.successRate).toBe(1.0);
    });
  });

  describe('getBestVariation', () => {
    it('should identify fastest variation', async () => {
      const task = {
        execute: async (params: any) => {
          await new Promise(resolve => setTimeout(resolve, params.delay));
          return 'done';
        },
      };

      const variations = [
        { id: 'fast', parameters: { delay: 10 } },
        { id: 'slow', parameters: { delay: 100 } },
      ];

      await coordinator.executeVariations(variations, task);

      const best = coordinator.getBestVariation('executionTime');

      expect(best).toBeDefined();
      expect(best!.variationId).toBe('fast');
    });
  });
});
