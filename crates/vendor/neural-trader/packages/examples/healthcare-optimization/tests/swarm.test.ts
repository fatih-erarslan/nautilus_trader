/**
 * Swarm Coordinator Tests
 */

import { SwarmCoordinator } from '../src/swarm';
import type { ForecastResult, ScheduleConstraints, OptimizationObjective } from '../src/types';
import { describe, it, expect, beforeEach } from '@jest/globals';
import * as fs from 'fs';
import * as path from 'path';

describe('SwarmCoordinator', () => {
  let swarm: SwarmCoordinator;
  const testDbPath = path.join(__dirname, '../.test-db/swarm.db');

  beforeEach(() => {
    // Clean up test database
    if (fs.existsSync(testDbPath)) {
      fs.unlinkSync(testDbPath);
    }

    swarm = new SwarmCoordinator(
      {
        populationSize: 10,
        maxIterations: 5, // Small for tests
        explorationRate: 0.3,
        convergenceThreshold: 0.01,
        elitismRate: 0.2
      },
      testDbPath
    );
  });

  describe('Swarm Optimization', () => {
    it('should optimize schedule using swarm intelligence', async () => {
      const forecasts = generateMockForecasts(24);
      const constraints: ScheduleConstraints = {
        minStaffPerShift: {
          physician: 2,
          nurse: 3
        },
        maxConsecutiveHours: 16,
        minRestBetweenShifts: 8,
        requiredSkillCoverage: []
      };

      const objective: OptimizationObjective = {
        minimizeWaitTime: 0.4,
        maximizeUtilization: 0.3,
        minimizeCost: 0.2,
        maximizePatientOutcomes: 0.1
      };

      const schedulerConfig = {
        planningHorizonDays: 1,
        shiftDuration: 8,
        costPerConstraintViolation: 1000
      };

      const startDate = new Date('2024-01-15T00:00:00');

      const result = await swarm.optimize(
        forecasts,
        constraints,
        objective,
        schedulerConfig,
        startDate
      );

      expect(result.bestSolution).toBeTruthy();
      expect(result.bestSolution.shifts.length).toBeGreaterThan(0);
      expect(result.iterations).toBeGreaterThan(0);
      expect(result.iterations).toBeLessThanOrEqual(5);
      expect(result.exploredSolutions).toBeGreaterThan(0);
      expect(result.convergenceHistory.length).toBe(result.iterations);
    }, 30000); // Longer timeout for swarm

    it('should improve solution over iterations', async () => {
      const forecasts = generateMockForecasts(24);
      const constraints: ScheduleConstraints = {
        minStaffPerShift: {
          nurse: 2
        },
        maxConsecutiveHours: 16,
        minRestBetweenShifts: 8,
        requiredSkillCoverage: []
      };

      const objective: OptimizationObjective = {
        minimizeWaitTime: 0.5,
        maximizeUtilization: 0.3,
        minimizeCost: 0.2,
        maximizePatientOutcomes: 0.0
      };

      const schedulerConfig = {
        planningHorizonDays: 1,
        shiftDuration: 8,
        costPerConstraintViolation: 500
      };

      const startDate = new Date('2024-01-15T00:00:00');

      const result = await swarm.optimize(
        forecasts,
        constraints,
        objective,
        schedulerConfig,
        startDate
      );

      // Fitness should generally improve (or stay same if converged early)
      const history = result.convergenceHistory;
      expect(history[history.length - 1]).toBeGreaterThanOrEqual(history[0]);
    }, 30000);

    it('should track swarm statistics', async () => {
      const forecasts = generateMockForecasts(24);
      const constraints: ScheduleConstraints = {
        minStaffPerShift: { nurse: 1 },
        maxConsecutiveHours: 16,
        minRestBetweenShifts: 8,
        requiredSkillCoverage: []
      };

      const objective: OptimizationObjective = {
        minimizeWaitTime: 0.5,
        maximizeUtilization: 0.3,
        minimizeCost: 0.2,
        maximizePatientOutcomes: 0.0
      };

      const schedulerConfig = {
        planningHorizonDays: 1,
        shiftDuration: 8,
        costPerConstraintViolation: 500
      };

      const startDate = new Date('2024-01-15T00:00:00');

      await swarm.optimize(
        forecasts,
        constraints,
        objective,
        schedulerConfig,
        startDate
      );

      const stats = swarm.getStatistics();

      expect(stats.population).toBe(10);
      expect(stats.bestFitness).toBeGreaterThanOrEqual(0);
      expect(stats.avgFitness).toBeGreaterThanOrEqual(0);
      expect(stats.worstFitness).toBeGreaterThanOrEqual(0);
      expect(stats.diversity).toBeGreaterThanOrEqual(0);
    }, 30000);
  });

  describe('Convergence', () => {
    it('should converge when improvement plateaus', async () => {
      const forecasts = generateMockForecasts(24);
      const constraints: ScheduleConstraints = {
        minStaffPerShift: { nurse: 1 },
        maxConsecutiveHours: 16,
        minRestBetweenShifts: 8,
        requiredSkillCoverage: []
      };

      const objective: OptimizationObjective = {
        minimizeWaitTime: 0.5,
        maximizeUtilization: 0.5,
        minimizeCost: 0.0,
        maximizePatientOutcomes: 0.0
      };

      const schedulerConfig = {
        planningHorizonDays: 1,
        shiftDuration: 8,
        costPerConstraintViolation: 500
      };

      const startDate = new Date('2024-01-15T00:00:00');

      // Use high convergence threshold for quick convergence
      const quickSwarm = new SwarmCoordinator(
        {
          populationSize: 5,
          maxIterations: 20,
          explorationRate: 0.2,
          convergenceThreshold: 0.1, // Easy to converge
          elitismRate: 0.4
        },
        testDbPath
      );

      const result = await quickSwarm.optimize(
        forecasts,
        constraints,
        objective,
        schedulerConfig,
        startDate
      );

      // Should converge before max iterations (usually)
      expect(result.iterations).toBeLessThan(20);
    }, 30000);
  });
});

/**
 * Generate mock forecasts
 */
function generateMockForecasts(hours: number): ForecastResult[] {
  const forecasts: ForecastResult[] = [];
  const startDate = new Date('2024-01-15T00:00:00');

  for (let i = 0; i < hours; i++) {
    const timestamp = new Date(startDate);
    timestamp.setHours(timestamp.getHours() + i);

    const hour = timestamp.getHours();
    let baseArrivals = 8;

    if ((hour >= 9 && hour <= 11) || (hour >= 18 && hour <= 20)) {
      baseArrivals = 15;
    } else if (hour >= 0 && hour <= 6) {
      baseArrivals = 4;
    }

    forecasts.push({
      timestamp,
      predictedArrivals: baseArrivals,
      lowerBound: baseArrivals - 3,
      upperBound: baseArrivals + 3,
      confidence: 0.95,
      seasonalComponent: 1.0,
      trendComponent: 0
    });
  }

  return forecasts;
}
