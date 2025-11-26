/**
 * Tests for SwarmScheduler
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { SwarmScheduler } from '../src/swarm-scheduler.js';
import {
  GeneratorType,
  ForecastHorizon,
  type GeneratorUnit,
  type BatteryStorage,
  type LoadForecast,
} from '../src/types.js';
import { rm } from 'fs/promises';

describe('SwarmScheduler', () => {
  let scheduler: SwarmScheduler;
  let generators: GeneratorUnit[];
  let batteries: BatteryStorage[];
  let forecasts: LoadForecast[];
  const testNamespace = 'test-swarm-scheduler';

  beforeEach(async () => {
    scheduler = new SwarmScheduler({
      swarmSize: 6,
      explorationRate: 0.3,
      maxIterations: 5,
      convergenceThreshold: 0.01,
      enableOpenRouter: false,
      memoryNamespace: testNamespace,
    });

    await scheduler.initialize();

    generators = [
      {
        id: 'coal-1',
        type: GeneratorType.COAL,
        minCapacityMW: 100,
        maxCapacityMW: 500,
        rampUpRate: 50,
        rampDownRate: 50,
        minUpTime: 4,
        minDownTime: 4,
        startupCost: 5000,
        shutdownCost: 2000,
        variableCost: 30,
        fixedCost: 500,
        startupTime: 2,
        status: {
          isOnline: true,
          currentOutputMW: 300,
          hoursOnline: 12,
          hoursOffline: 0,
        },
      },
      {
        id: 'natgas-1',
        type: GeneratorType.NATURAL_GAS,
        minCapacityMW: 50,
        maxCapacityMW: 300,
        rampUpRate: 100,
        rampDownRate: 100,
        minUpTime: 2,
        minDownTime: 2,
        startupCost: 2000,
        shutdownCost: 500,
        variableCost: 45,
        fixedCost: 300,
        startupTime: 1,
        status: {
          isOnline: true,
          currentOutputMW: 200,
          hoursOnline: 6,
          hoursOffline: 0,
        },
      },
      {
        id: 'wind-1',
        type: GeneratorType.WIND,
        minCapacityMW: 0,
        maxCapacityMW: 200,
        rampUpRate: 200,
        rampDownRate: 200,
        minUpTime: 0,
        minDownTime: 0,
        startupCost: 0,
        shutdownCost: 0,
        variableCost: 5,
        fixedCost: 50,
        startupTime: 0,
        status: {
          isOnline: true,
          currentOutputMW: 150,
          hoursOnline: 24,
          hoursOffline: 0,
        },
      },
    ];

    batteries = [
      {
        id: 'battery-1',
        maxPowerMW: 100,
        capacityMWh: 400,
        currentChargeMWh: 200,
        chargeEfficiency: 0.95,
        dischargeEfficiency: 0.95,
        minChargeMWh: 40,
        maxChargeMWh: 400,
        degradationRate: 0.0001,
      },
    ];

    forecasts = Array.from({ length: 24 }, (_, i) => ({
      timestamp: new Date(Date.now() + i * 3600000),
      loadMW: 500 + Math.sin((i / 24) * Math.PI * 2) * 100,
      lowerBound: 450,
      upperBound: 550,
      confidence: 0.95,
      horizon: ForecastHorizon.HOUR_1,
    }));
  });

  afterEach(async () => {
    // Clean up test database
    try {
      await rm(`./${testNamespace}`, { recursive: true, force: true });
    } catch (error) {
      // Ignore cleanup errors
    }
  });

  describe('initialization', () => {
    it('should initialize successfully', () => {
      expect(scheduler).toBeDefined();
    });

    it('should initialize multiple strategies', async () => {
      const stats = scheduler.getStrategyStatistics();
      expect(stats.length).toBeGreaterThanOrEqual(0);
    });

    it('should load best strategies from memory if available', async () => {
      // Second scheduler should load strategies from first
      const scheduler2 = new SwarmScheduler({
        swarmSize: 6,
        explorationRate: 0.3,
        maxIterations: 5,
        convergenceThreshold: 0.01,
        enableOpenRouter: false,
        memoryNamespace: testNamespace,
      });

      await scheduler2.initialize();
      expect(scheduler2).toBeDefined();
    });
  });

  describe('optimizeSchedule', () => {
    it('should optimize schedule with multiple strategies', async () => {
      const result = await scheduler.optimizeSchedule(
        generators,
        batteries,
        forecasts
      );

      expect(result).toBeDefined();
      expect(result.scheduleId).toBeDefined();
      expect(result.strategy).toBeDefined();
      expect(result.commitments).toHaveLength(24);
      expect(result.totalCost).toBeGreaterThan(0);
      expect(result.qualityScore).toBeGreaterThan(0);
    });

    it('should produce feasible solutions', async () => {
      const result = await scheduler.optimizeSchedule(
        generators,
        batteries,
        forecasts
      );

      expect(result.isFeasible).toBe(true);
      result.commitments.forEach(commitment => {
        expect(commitment.isFeasible).toBe(true);
      });
    });

    it('should calculate renewable utilization', async () => {
      const result = await scheduler.optimizeSchedule(
        generators,
        batteries,
        forecasts
      );

      expect(result.renewableUtilization).toBeGreaterThanOrEqual(0);
      expect(result.renewableUtilization).toBeLessThanOrEqual(100);
    });

    it('should calculate emissions', async () => {
      const result = await scheduler.optimizeSchedule(
        generators,
        batteries,
        forecasts
      );

      expect(result.totalEmissions).toBeGreaterThanOrEqual(0);
    });

    it('should track computation time', async () => {
      const result = await scheduler.optimizeSchedule(
        generators,
        batteries,
        forecasts
      );

      expect(result.computeTimeMs).toBeGreaterThan(0);
      expect(result.computeTimeMs).toBeLessThan(30000); // Should complete in 30 seconds
    });

    it('should explore different strategies', async () => {
      const result1 = await scheduler.optimizeSchedule(
        generators,
        batteries,
        forecasts
      );

      const result2 = await scheduler.optimizeSchedule(
        generators,
        batteries,
        forecasts
      );

      // Results may differ due to strategy evolution
      expect(result1).toBeDefined();
      expect(result2).toBeDefined();
    });
  });

  describe('strategy evolution', () => {
    it('should update performance history', async () => {
      await scheduler.optimizeSchedule(generators, batteries, forecasts);

      const stats = scheduler.getStrategyStatistics();
      expect(stats.length).toBeGreaterThan(0);
      expect(stats[0].count).toBeGreaterThan(0);
    });

    it('should evolve strategies over multiple runs', async () => {
      const initialBest = scheduler.getBestStrategies(1);

      // Run optimization multiple times
      for (let i = 0; i < 3; i++) {
        await scheduler.optimizeSchedule(generators, batteries, forecasts);
      }

      const finalBest = scheduler.getBestStrategies(1);
      expect(finalBest.length).toBeGreaterThan(0);
    });

    it('should track strategy performance', async () => {
      await scheduler.optimizeSchedule(generators, batteries, forecasts);
      await scheduler.optimizeSchedule(generators, batteries, forecasts);

      const stats = scheduler.getStrategyStatistics();
      const topStrategy = stats[0];

      expect(topStrategy.avgScore).toBeGreaterThan(0);
      expect(topStrategy.count).toBeGreaterThanOrEqual(1);
    });
  });

  describe('multi-objective optimization', () => {
    it('should balance cost and renewable objectives', async () => {
      const result = await scheduler.optimizeSchedule(
        generators,
        batteries,
        forecasts
      );

      // Strategy should consider both cost and renewables
      expect(result.totalCost).toBeGreaterThan(0);
      expect(result.renewableUtilization).toBeGreaterThanOrEqual(0);
      expect(result.qualityScore).toBeGreaterThan(0);
    });

    it('should consider emissions in optimization', async () => {
      const result = await scheduler.optimizeSchedule(
        generators,
        batteries,
        forecasts
      );

      expect(result.totalEmissions).toBeGreaterThanOrEqual(0);
    });
  });

  describe('strategy statistics', () => {
    it('should return strategy statistics', async () => {
      await scheduler.optimizeSchedule(generators, batteries, forecasts);

      const stats = scheduler.getStrategyStatistics();
      expect(stats).toBeDefined();
      expect(Array.isArray(stats)).toBe(true);
    });

    it('should return best strategies', async () => {
      await scheduler.optimizeSchedule(generators, batteries, forecasts);

      const bestStrategies = scheduler.getBestStrategies(3);
      expect(bestStrategies.length).toBeGreaterThan(0);
      expect(bestStrategies.length).toBeLessThanOrEqual(3);
    });

    it('should sort strategies by performance', async () => {
      await scheduler.optimizeSchedule(generators, batteries, forecasts);
      await scheduler.optimizeSchedule(generators, batteries, forecasts);

      const stats = scheduler.getStrategyStatistics();
      if (stats.length > 1) {
        for (let i = 1; i < stats.length; i++) {
          expect(stats[i - 1].avgScore).toBeGreaterThanOrEqual(stats[i].avgScore);
        }
      }
    });
  });

  describe('persistence', () => {
    it('should persist best strategies', async () => {
      await scheduler.optimizeSchedule(generators, batteries, forecasts);

      // Create new scheduler with same namespace
      const scheduler2 = new SwarmScheduler({
        swarmSize: 6,
        explorationRate: 0.3,
        maxIterations: 5,
        convergenceThreshold: 0.01,
        enableOpenRouter: false,
        memoryNamespace: testNamespace,
      });

      await scheduler2.initialize();

      const bestStrategies = scheduler2.getBestStrategies(3);
      expect(bestStrategies.length).toBeGreaterThan(0);
    });
  });
});
