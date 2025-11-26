/**
 * Comprehensive tests for SwarmPolicyOptimizer
 */

import { SwarmPolicyOptimizer, SwarmConfig } from '../src/swarm-policy';
import { InventoryOptimizer, OptimizerConfig } from '../src/inventory-optimizer';
import { DemandForecaster, ForecastConfig, DemandPattern } from '../src/demand-forecaster';

describe('SwarmPolicyOptimizer', () => {
  let forecaster: DemandForecaster;
  let optimizer: InventoryOptimizer;
  let swarmOptimizer: SwarmPolicyOptimizer;
  let swarmConfig: SwarmConfig;

  beforeEach(async () => {
    // Setup forecaster
    const forecastConfig: ForecastConfig = {
      alpha: 0.1,
      horizons: [1, 7, 14, 30],
      seasonalityPeriods: [7, 52],
      learningRate: 0.01,
      memoryNamespace: 'test-swarm',
    };

    forecaster = new DemandForecaster(forecastConfig);

    // Train forecaster
    const trainingData = generateTrainingData(50);
    await forecaster.train(trainingData);

    // Setup optimizer
    const optimizerConfig: OptimizerConfig = {
      targetServiceLevel: 0.95,
      planningHorizon: 30,
      reviewPeriod: 7,
      safetyFactor: 1.65,
      costWeights: {
        holding: 1,
        ordering: 1,
        shortage: 5,
      },
    };

    optimizer = new InventoryOptimizer(forecaster, optimizerConfig);
    optimizer.addNode(createTestNode('warehouse-1'));

    // Setup swarm optimizer
    swarmConfig = {
      particles: 10,
      iterations: 5, // Small for tests
      inertia: 0.7,
      cognitive: 1.5,
      social: 1.5,
      bounds: {
        reorderPoint: [50, 500],
        orderUpToLevel: [100, 1000],
        safetyFactor: [1.0, 3.0],
      },
      objectives: {
        costWeight: 0.6,
        serviceLevelWeight: 0.4,
      },
    };

    swarmOptimizer = new SwarmPolicyOptimizer(forecaster, optimizer, swarmConfig);
  });

  describe('Initialization', () => {
    it('should create swarm optimizer', () => {
      expect(swarmOptimizer).toBeDefined();
    });

    it('should initialize particles', async () => {
      const result = await swarmOptimizer.optimize('product-1', {
        dayOfWeek: 1,
        weekOfYear: 20,
        monthOfYear: 5,
        isHoliday: false,
        promotions: 0,
        priceIndex: 1.0,
      });

      expect(result.particles).toHaveLength(swarmConfig.particles);
    });
  });

  describe('Particle Swarm Optimization', () => {
    it('should optimize policy parameters', async () => {
      const result = await swarmOptimizer.optimize('product-1', {
        dayOfWeek: 1,
        weekOfYear: 20,
        monthOfYear: 5,
        isHoliday: false,
        promotions: 0,
        priceIndex: 1.0,
      });

      expect(result.bestPolicy).toBeDefined();
      expect(result.bestPolicy.reorderPoint).toBeGreaterThan(0);
      expect(result.bestPolicy.orderUpToLevel).toBeGreaterThan(
        result.bestPolicy.reorderPoint
      );
      expect(result.bestPolicy.safetyFactor).toBeGreaterThan(0);
    });

    it('should improve fitness over iterations', async () => {
      const result = await swarmOptimizer.optimize('product-1', {
        dayOfWeek: 1,
        weekOfYear: 20,
        monthOfYear: 5,
        isHoliday: false,
        promotions: 0,
        priceIndex: 1.0,
      });

      const firstFitness = result.convergenceHistory[0]!;
      const lastFitness = result.convergenceHistory[result.convergenceHistory.length - 1]!;

      // Fitness should improve (decrease)
      expect(lastFitness).toBeLessThanOrEqual(firstFitness);
    });

    it('should track convergence', async () => {
      const result = await swarmOptimizer.optimize('product-1', {
        dayOfWeek: 1,
        weekOfYear: 20,
        monthOfYear: 5,
        isHoliday: false,
        promotions: 0,
        priceIndex: 1.0,
      });

      expect(result.convergenceHistory).toHaveLength(swarmConfig.iterations);
    });

    it('should respect parameter bounds', async () => {
      const result = await swarmOptimizer.optimize('product-1', {
        dayOfWeek: 1,
        weekOfYear: 20,
        monthOfYear: 5,
        isHoliday: false,
        promotions: 0,
        priceIndex: 1.0,
      });

      for (const particle of result.particles) {
        expect(particle.position.reorderPoint).toBeGreaterThanOrEqual(
          swarmConfig.bounds.reorderPoint[0]
        );
        expect(particle.position.reorderPoint).toBeLessThanOrEqual(
          swarmConfig.bounds.reorderPoint[1]
        );

        expect(particle.position.orderUpToLevel).toBeGreaterThanOrEqual(
          swarmConfig.bounds.orderUpToLevel[0]
        );
        expect(particle.position.orderUpToLevel).toBeLessThanOrEqual(
          swarmConfig.bounds.orderUpToLevel[1]
        );

        expect(particle.position.safetyFactor).toBeGreaterThanOrEqual(
          swarmConfig.bounds.safetyFactor[0]
        );
        expect(particle.position.safetyFactor).toBeLessThanOrEqual(
          swarmConfig.bounds.safetyFactor[1]
        );
      }
    });
  });

  describe('Fitness Evaluation', () => {
    it('should evaluate particle fitness', async () => {
      const result = await swarmOptimizer.optimize('product-1', {
        dayOfWeek: 1,
        weekOfYear: 20,
        monthOfYear: 5,
        isHoliday: false,
        promotions: 0,
        priceIndex: 1.0,
      });

      expect(result.bestFitness.cost).toBeGreaterThan(0);
      expect(result.bestFitness.serviceLevel).toBeGreaterThan(0);
      expect(result.bestFitness.combined).toBeGreaterThan(0);
    });

    it('should combine multiple objectives', async () => {
      const result = await swarmOptimizer.optimize('product-1', {
        dayOfWeek: 1,
        weekOfYear: 20,
        monthOfYear: 5,
        isHoliday: false,
        promotions: 0,
        priceIndex: 1.0,
      });

      // Combined fitness should reflect both cost and service level
      const expectedCombined =
        swarmConfig.objectives.costWeight * result.bestFitness.cost +
        swarmConfig.objectives.serviceLevelWeight * (1 - result.bestFitness.serviceLevel);

      expect(result.bestFitness.combined).toBeCloseTo(expectedCombined, 5);
    });
  });

  describe('Multi-Objective Optimization', () => {
    it('should generate Pareto front', async () => {
      await swarmOptimizer.optimize('product-1', {
        dayOfWeek: 1,
        weekOfYear: 20,
        monthOfYear: 5,
        isHoliday: false,
        promotions: 0,
        priceIndex: 1.0,
      });

      const paretoFront = swarmOptimizer.getParetoFront();
      expect(paretoFront.length).toBeGreaterThan(0);
    });

    it('should identify non-dominated solutions', async () => {
      await swarmOptimizer.optimize('product-1', {
        dayOfWeek: 1,
        weekOfYear: 20,
        monthOfYear: 5,
        isHoliday: false,
        promotions: 0,
        priceIndex: 1.0,
      });

      const paretoFront = swarmOptimizer.getParetoFront();

      // No solution should dominate another in the Pareto front
      for (let i = 0; i < paretoFront.length; i++) {
        for (let j = 0; j < paretoFront.length; j++) {
          if (i === j) continue;

          const p1 = paretoFront[i]!;
          const p2 = paretoFront[j]!;

          // p2 should not strictly dominate p1
          const strictlyDominates =
            p2.fitness.cost <= p1.fitness.cost &&
            p2.fitness.serviceLevel >= p1.fitness.serviceLevel &&
            (p2.fitness.cost < p1.fitness.cost ||
              p2.fitness.serviceLevel > p1.fitness.serviceLevel);

          expect(strictlyDominates).toBe(false);
        }
      }
    });
  });

  describe('Adaptive Learning', () => {
    it('should adapt service level to target revenue', async () => {
      const targetRevenue = 10000;
      const adaptedLevel = await swarmOptimizer.adaptServiceLevel(
        'product-1',
        {
          dayOfWeek: 1,
          weekOfYear: 20,
          monthOfYear: 5,
          isHoliday: false,
          promotions: 0,
          priceIndex: 1.0,
        },
        targetRevenue
      );

      expect(adaptedLevel).toBeGreaterThan(0.8);
      expect(adaptedLevel).toBeLessThan(1.0);
    });

    it('should balance revenue and cost', async () => {
      const targetRevenue = 10000;
      const adaptedLevel = await swarmOptimizer.adaptServiceLevel(
        'product-1',
        {
          dayOfWeek: 1,
          weekOfYear: 20,
          monthOfYear: 5,
          isHoliday: false,
          promotions: 0,
          priceIndex: 1.0,
        },
        targetRevenue
      );

      // Adapted level should be reasonable (not too high or too low)
      expect(adaptedLevel).toBeGreaterThan(0.85);
      expect(adaptedLevel).toBeLessThan(0.99);
    });
  });

  describe('Policy Export', () => {
    it('should export best policy', async () => {
      await swarmOptimizer.optimize('product-1', {
        dayOfWeek: 1,
        weekOfYear: 20,
        monthOfYear: 5,
        isHoliday: false,
        promotions: 0,
        priceIndex: 1.0,
      });

      const exported = swarmOptimizer.exportPolicy();

      expect(exported.policy).toBeDefined();
      expect(exported.performance).toBeDefined();
      expect(exported.timestamp).toBeGreaterThan(0);
    });

    it('should throw if no policy available', () => {
      expect(() => swarmOptimizer.exportPolicy()).toThrow();
    });
  });

  describe('Performance', () => {
    it('should complete optimization in reasonable time', async () => {
      const start = Date.now();

      await swarmOptimizer.optimize('product-1', {
        dayOfWeek: 1,
        weekOfYear: 20,
        monthOfYear: 5,
        isHoliday: false,
        promotions: 0,
        priceIndex: 1.0,
      });

      const elapsed = Date.now() - start;

      // Should complete in under 30 seconds (generous for CI)
      expect(elapsed).toBeLessThan(30000);
    });
  });
});

/**
 * Helper functions
 */

function createTestNode(nodeId: string): any {
  return {
    nodeId,
    type: 'warehouse',
    level: 1,
    upstreamNodes: [],
    downstreamNodes: [],
    position: {
      currentStock: 500,
      onOrder: 100,
      allocated: 50,
    },
    costs: {
      holding: 0.5,
      ordering: 100,
      shortage: 50,
    },
    leadTime: {
      mean: 7,
      stdDev: 2,
      distribution: 'normal',
    },
    capacity: {
      storage: 10000,
      throughput: 1000,
    },
  };
}

function generateTrainingData(count: number): DemandPattern[] {
  const data: DemandPattern[] = [];
  const baseDate = new Date('2024-01-01');

  for (let i = 0; i < count; i++) {
    const date = new Date(baseDate.getTime() + i * 24 * 60 * 60 * 1000);
    data.push({
      productId: 'product-1',
      timestamp: date.getTime(),
      demand: 100 + Math.random() * 50,
      features: {
        dayOfWeek: date.getDay(),
        weekOfYear: Math.floor(i / 7),
        monthOfYear: date.getMonth(),
        isHoliday: false,
        promotions: 0,
        priceIndex: 1.0,
      },
    });
  }

  return data;
}
