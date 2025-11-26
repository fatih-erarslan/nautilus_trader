/**
 * Tests for UnitCommitmentOptimizer
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { UnitCommitmentOptimizer } from '../src/unit-commitment.js';
import {
  GeneratorType,
  ForecastHorizon,
  type GeneratorUnit,
  type BatteryStorage,
  type LoadForecast,
  type RenewableForecast,
} from '../src/types.js';

describe('UnitCommitmentOptimizer', () => {
  let optimizer: UnitCommitmentOptimizer;
  let generators: GeneratorUnit[];
  let batteries: BatteryStorage[];

  beforeEach(() => {
    optimizer = new UnitCommitmentOptimizer({
      planningHorizonHours: 24,
      timeStepMinutes: 60,
      reserveMarginPercent: 10,
      maxComputeTimeMs: 5000,
      solverTolerance: 1e-6,
      enableBatteryOptimization: true,
    });

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

    optimizer.registerGenerators(generators);
    optimizer.registerBatteries(batteries);
  });

  describe('initialization', () => {
    it('should initialize successfully', () => {
      expect(optimizer).toBeDefined();
    });

    it('should accept generator registrations', () => {
      const newOptimizer = new UnitCommitmentOptimizer({
        planningHorizonHours: 24,
        timeStepMinutes: 60,
        reserveMarginPercent: 10,
        maxComputeTimeMs: 5000,
        solverTolerance: 1e-6,
        enableBatteryOptimization: false,
      });

      expect(() => newOptimizer.registerGenerators(generators)).not.toThrow();
    });

    it('should accept battery registrations', () => {
      const newOptimizer = new UnitCommitmentOptimizer({
        planningHorizonHours: 24,
        timeStepMinutes: 60,
        reserveMarginPercent: 10,
        maxComputeTimeMs: 5000,
        solverTolerance: 1e-6,
        enableBatteryOptimization: true,
      });

      expect(() => newOptimizer.registerBatteries(batteries)).not.toThrow();
    });
  });

  describe('optimize', () => {
    it('should generate commitments for load forecasts', async () => {
      const forecasts: LoadForecast[] = Array.from({ length: 24 }, (_, i) => ({
        timestamp: new Date(Date.now() + i * 3600000),
        loadMW: 500 + Math.sin(i / 24 * Math.PI * 2) * 100,
        lowerBound: 450,
        upperBound: 550,
        confidence: 0.95,
        horizon: ForecastHorizon.HOUR_1,
      }));

      const commitments = await optimizer.optimize(forecasts);

      expect(commitments).toHaveLength(24);
      commitments.forEach(commitment => {
        expect(commitment.commitments).toBeDefined();
        expect(commitment.totalGenerationMW).toBeGreaterThan(0);
        expect(commitment.totalLoadMW).toBeGreaterThan(0);
      });
    });

    it('should respect generator capacity constraints', async () => {
      const forecasts: LoadForecast[] = Array.from({ length: 24 }, (_, i) => ({
        timestamp: new Date(Date.now() + i * 3600000),
        loadMW: 600,
        lowerBound: 550,
        upperBound: 650,
        confidence: 0.95,
        horizon: ForecastHorizon.HOUR_1,
      }));

      const commitments = await optimizer.optimize(forecasts);

      commitments.forEach(commitment => {
        commitment.commitments.forEach(genCommit => {
          const generator = generators.find(g => g.id === genCommit.generatorId);
          expect(generator).toBeDefined();

          if (genCommit.isCommitted) {
            expect(genCommit.outputMW).toBeGreaterThanOrEqual(0);
            expect(genCommit.outputMW).toBeLessThanOrEqual(generator!.maxCapacityMW);
          }
        });
      });
    });

    it('should balance load and generation', async () => {
      const forecasts: LoadForecast[] = Array.from({ length: 24 }, (_, i) => ({
        timestamp: new Date(Date.now() + i * 3600000),
        loadMW: 500,
        lowerBound: 450,
        upperBound: 550,
        confidence: 0.95,
        horizon: ForecastHorizon.HOUR_1,
      }));

      const commitments = await optimizer.optimize(forecasts);

      commitments.forEach(commitment => {
        // Allow small tolerance for numerical errors and reserve
        const imbalance = Math.abs(
          commitment.totalGenerationMW - commitment.totalLoadMW
        );
        expect(imbalance).toBeLessThan(50); // 50 MW tolerance
      });
    });

    it('should include battery operations when enabled', async () => {
      const forecasts: LoadForecast[] = Array.from({ length: 24 }, (_, i) => ({
        timestamp: new Date(Date.now() + i * 3600000),
        loadMW: 500 + (i % 2 === 0 ? 100 : -100),
        lowerBound: 400,
        upperBound: 600,
        confidence: 0.95,
        horizon: ForecastHorizon.HOUR_1,
      }));

      const commitments = await optimizer.optimize(forecasts);

      const hasBatteryOperations = commitments.some(
        c => c.batteryOperations.length > 0
      );
      expect(hasBatteryOperations).toBe(true);
    });

    it('should calculate total cost', async () => {
      const forecasts: LoadForecast[] = Array.from({ length: 24 }, (_, i) => ({
        timestamp: new Date(Date.now() + i * 3600000),
        loadMW: 500,
        lowerBound: 450,
        upperBound: 550,
        confidence: 0.95,
        horizon: ForecastHorizon.HOUR_1,
      }));

      const commitments = await optimizer.optimize(forecasts);

      commitments.forEach(commitment => {
        expect(commitment.totalCost).toBeGreaterThan(0);
      });
    });
  });

  describe('renewable integration', () => {
    it('should integrate renewable forecasts', async () => {
      const forecasts: LoadForecast[] = Array.from({ length: 24 }, (_, i) => ({
        timestamp: new Date(Date.now() + i * 3600000),
        loadMW: 600,
        lowerBound: 550,
        upperBound: 650,
        confidence: 0.95,
        horizon: ForecastHorizon.HOUR_1,
      }));

      const renewableForecasts: RenewableForecast[] = Array.from(
        { length: 24 },
        (_, i) => ({
          timestamp: new Date(Date.now() + i * 3600000),
          generatorId: 'wind-1',
          expectedOutputMW: 100,
          uncertaintyMW: 20,
        })
      );

      const commitments = await optimizer.optimize(forecasts, renewableForecasts);

      expect(commitments).toHaveLength(24);
      // With renewable generation, conventional generation should be reduced
    });
  });

  describe('fallback behavior', () => {
    it('should provide fallback schedule when optimization fails', async () => {
      const forecasts: LoadForecast[] = Array.from({ length: 24 }, (_, i) => ({
        timestamp: new Date(Date.now() + i * 3600000),
        loadMW: 10000, // Impossibly high load
        lowerBound: 9500,
        upperBound: 10500,
        confidence: 0.95,
        horizon: ForecastHorizon.HOUR_1,
      }));

      const commitments = await optimizer.optimize(forecasts);

      expect(commitments).toHaveLength(24);
      // Should still return commitments, possibly marked as infeasible
    });
  });

  describe('spinning reserve', () => {
    it('should maintain spinning reserve', async () => {
      const forecasts: LoadForecast[] = Array.from({ length: 24 }, (_, i) => ({
        timestamp: new Date(Date.now() + i * 3600000),
        loadMW: 500,
        lowerBound: 450,
        upperBound: 550,
        confidence: 0.95,
        horizon: ForecastHorizon.HOUR_1,
      }));

      const commitments = await optimizer.optimize(forecasts);

      commitments.forEach(commitment => {
        const requiredReserve = commitment.totalLoadMW * 0.1; // 10%
        expect(commitment.spinningReserveMW).toBeGreaterThanOrEqual(
          requiredReserve * 0.9
        ); // Allow 10% tolerance
      });
    });
  });
});
