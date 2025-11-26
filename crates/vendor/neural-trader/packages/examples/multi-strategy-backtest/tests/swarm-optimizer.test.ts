/**
 * Test suite for SwarmOptimizer
 */

import { SwarmOptimizer } from '../src/swarm-optimizer';

describe('SwarmOptimizer', () => {
  let optimizer: SwarmOptimizer;

  beforeEach(() => {
    optimizer = new SwarmOptimizer({
      particleCount: 10,
      maxIterations: 20,
      inertia: 0.7
    });
  });

  describe('Initialization', () => {
    it('should initialize with correct configuration', () => {
      const stats = optimizer.getStats();
      expect(stats.particleCount).toBe(0); // Not initialized until optimize()
      expect(stats.iterations).toBe(0);
    });
  });

  describe('Optimization', () => {
    it('should optimize a simple function', async () => {
      // Minimize sphere function: f(x, y) = x^2 + y^2
      // Global minimum at (0, 0) = 0
      const objective = async (params: Record<string, number>) => {
        const x = params.x;
        const y = params.y;
        return -(x * x + y * y); // Negative because PSO maximizes
      };

      const bounds = {
        x: [-10, 10] as [number, number],
        y: [-10, 10] as [number, number]
      };

      const result = await optimizer.optimize(objective, bounds);

      expect(result.bestScore).toBeDefined();
      expect(result.bestParameters.x).toBeCloseTo(0, 0); // Within 1 unit
      expect(result.bestParameters.y).toBeCloseTo(0, 0);
      expect(result.evaluations).toBeGreaterThan(0);
    });

    it('should track convergence history', async () => {
      const objective = async (params: Record<string, number>) => {
        return -(params.x ** 2);
      };

      const bounds = { x: [-5, 5] as [number, number] };

      const result = await optimizer.optimize(objective, bounds);

      expect(result.convergenceHistory.length).toBeGreaterThan(0);
      expect(result.convergenceHistory.length).toBeLessThanOrEqual(20);
    });

    it('should handle multiple parameters', async () => {
      const objective = async (params: Record<string, number>) => {
        const sum = Object.values(params).reduce((a, b) => a + b ** 2, 0);
        return -sum;
      };

      const bounds = {
        a: [-5, 5] as [number, number],
        b: [-5, 5] as [number, number],
        c: [-5, 5] as [number, number]
      };

      const result = await optimizer.optimize(objective, bounds);

      expect(Object.keys(result.bestParameters).length).toBe(3);
      expect(result.bestScore).toBeLessThan(0);
    });

    it('should respect parameter bounds', async () => {
      const objective = async (params: Record<string, number>) => {
        return -params.x;
      };

      const bounds = { x: [0, 10] as [number, number] };

      const result = await optimizer.optimize(objective, bounds);

      expect(result.bestParameters.x).toBeGreaterThanOrEqual(0);
      expect(result.bestParameters.x).toBeLessThanOrEqual(10);
    });
  });

  describe('Convergence', () => {
    it('should stop early if converged', async () => {
      optimizer = new SwarmOptimizer({
        particleCount: 5,
        maxIterations: 100
      });

      const objective = async (params: Record<string, number>) => {
        return 10; // Constant function - should converge quickly
      };

      const bounds = { x: [-5, 5] as [number, number] };

      const result = await optimizer.optimize(objective, bounds);

      expect(result.convergenceHistory.length).toBeLessThan(100);
    });
  });

  describe('Performance', () => {
    it('should complete optimization in reasonable time', async () => {
      const startTime = Date.now();

      const objective = async (params: Record<string, number>) => {
        return -(params.x ** 2 + params.y ** 2);
      };

      const bounds = {
        x: [-10, 10] as [number, number],
        y: [-10, 10] as [number, number]
      };

      await optimizer.optimize(objective, bounds);

      const duration = Date.now() - startTime;
      expect(duration).toBeLessThan(5000); // Should complete in < 5 seconds
    });
  });

  describe('Edge Cases', () => {
    it('should handle single parameter optimization', async () => {
      const objective = async (params: Record<string, number>) => {
        return -Math.abs(params.x - 5);
      };

      const bounds = { x: [0, 10] as [number, number] };

      const result = await optimizer.optimize(objective, bounds);

      expect(result.bestParameters.x).toBeCloseTo(5, 0);
    });

    it('should handle narrow bounds', async () => {
      const objective = async (params: Record<string, number>) => {
        return -params.x;
      };

      const bounds = { x: [4.9, 5.1] as [number, number] };

      const result = await optimizer.optimize(objective, bounds);

      expect(result.bestParameters.x).toBeGreaterThanOrEqual(4.9);
      expect(result.bestParameters.x).toBeLessThanOrEqual(5.1);
    });
  });

  describe('Reset', () => {
    it('should reset optimizer state', async () => {
      const objective = async (params: Record<string, number>) => {
        return -params.x ** 2;
      };

      const bounds = { x: [-5, 5] as [number, number] };

      await optimizer.optimize(objective, bounds);

      const statsBeforeReset = optimizer.getStats();
      expect(statsBeforeReset.evaluations).toBeGreaterThan(0);

      optimizer.reset();

      const statsAfterReset = optimizer.getStats();
      expect(statsAfterReset.evaluations).toBe(0);
      expect(statsAfterReset.particleCount).toBe(0);
    });
  });
});
