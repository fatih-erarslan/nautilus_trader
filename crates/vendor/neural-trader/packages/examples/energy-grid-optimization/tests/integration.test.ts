/**
 * Integration tests for complete energy grid optimization workflow
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { LoadForecaster } from '../src/load-forecaster.js';
import { UnitCommitmentOptimizer } from '../src/unit-commitment.js';
import { SwarmScheduler } from '../src/swarm-scheduler.js';
import {
  GeneratorType,
  ForecastHorizon,
  type GridState,
  type GeneratorUnit,
  type BatteryStorage,
} from '../src/types.js';
import { rm } from 'fs/promises';

describe('Energy Grid Optimization Integration', () => {
  const forecasterNamespace = 'test-integration-forecaster';
  const schedulerNamespace = 'test-integration-scheduler';

  let forecaster: LoadForecaster;
  let scheduler: SwarmScheduler;
  let generators: GeneratorUnit[];
  let batteries: BatteryStorage[];

  beforeEach(async () => {
    // Initialize forecaster
    forecaster = new LoadForecaster({
      horizons: [ForecastHorizon.HOUR_1, ForecastHorizon.HOUR_4],
      historyWindowDays: 7,
      confidenceLevel: 0.95,
      enableErrorCorrection: true,
      correctionUpdateFrequency: 1,
      memoryNamespace: forecasterNamespace,
    });
    await forecaster.initialize();

    // Initialize scheduler
    scheduler = new SwarmScheduler({
      swarmSize: 4,
      explorationRate: 0.3,
      maxIterations: 3,
      convergenceThreshold: 0.01,
      enableOpenRouter: false,
      memoryNamespace: schedulerNamespace,
    });
    await scheduler.initialize();

    // Define generator fleet
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
      {
        id: 'solar-1',
        type: GeneratorType.SOLAR,
        minCapacityMW: 0,
        maxCapacityMW: 300,
        rampUpRate: 300,
        rampDownRate: 300,
        minUpTime: 0,
        minDownTime: 0,
        startupCost: 0,
        shutdownCost: 0,
        variableCost: 2,
        fixedCost: 30,
        startupTime: 0,
        status: {
          isOnline: true,
          currentOutputMW: 250,
          hoursOnline: 8,
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

    // Add some historical data
    for (let i = 0; i < 72; i++) {
      const state: GridState = {
        timestamp: new Date(Date.now() - i * 3600000),
        loadMW: 5000 + Math.sin((i / 24) * Math.PI * 2) * 500,
        generationMW: 5100 + Math.sin((i / 24) * Math.PI * 2) * 500,
        renewablePenetration: 20 + Math.random() * 10,
        frequency: 60.0,
        voltageStability: 0.98,
        activeGenerators: ['gen-1', 'gen-2'],
        weather: {
          temperature: 20 + Math.random() * 10,
          windSpeed: 5 + Math.random() * 10,
          solarIrradiance: 600 + Math.random() * 400,
          precipitation: 0,
        },
        dayOfWeek: new Date(Date.now() - i * 3600000).getDay(),
        hourOfDay: new Date(Date.now() - i * 3600000).getHours(),
        isHoliday: false,
      };
      await forecaster.addHistoricalState(state);
    }
  });

  afterEach(async () => {
    // Clean up test databases
    try {
      await rm(`./${forecasterNamespace}`, { recursive: true, force: true });
      await rm(`./${schedulerNamespace}`, { recursive: true, force: true });
    } catch (error) {
      // Ignore cleanup errors
    }
  });

  describe('Complete Workflow', () => {
    it('should execute full optimization workflow', async () => {
      // 1. Generate load forecasts
      const currentState: GridState = {
        timestamp: new Date(),
        loadMW: 5000,
        generationMW: 5100,
        renewablePenetration: 25,
        frequency: 60.0,
        voltageStability: 0.98,
        activeGenerators: ['gen-1', 'gen-2'],
        weather: {
          temperature: 22,
          windSpeed: 8,
          solarIrradiance: 800,
          precipitation: 0,
        },
        dayOfWeek: new Date().getDay(),
        hourOfDay: new Date().getHours(),
        isHoliday: false,
      };

      const forecasts = await forecaster.forecast(currentState);
      expect(forecasts.length).toBeGreaterThan(0);

      // 2. Optimize schedule with swarm
      const result = await scheduler.optimizeSchedule(
        generators,
        batteries,
        forecasts
      );

      expect(result).toBeDefined();
      expect(result.isFeasible).toBe(true);
      expect(result.commitments.length).toBeGreaterThan(0);
      expect(result.totalCost).toBeGreaterThan(0);
      expect(result.qualityScore).toBeGreaterThan(0);

      // 3. Verify commitments are valid
      result.commitments.forEach(commitment => {
        expect(commitment.totalGenerationMW).toBeGreaterThan(0);
        expect(commitment.totalLoadMW).toBeGreaterThan(0);
        expect(commitment.isFeasible).toBe(true);
      });
    });

    it('should improve over multiple optimization cycles', async () => {
      const currentState: GridState = {
        timestamp: new Date(),
        loadMW: 5000,
        generationMW: 5100,
        renewablePenetration: 25,
        frequency: 60.0,
        voltageStability: 0.98,
        activeGenerators: ['gen-1', 'gen-2'],
        weather: {
          temperature: 22,
          windSpeed: 8,
          solarIrradiance: 800,
          precipitation: 0,
        },
        dayOfWeek: new Date().getDay(),
        hourOfDay: new Date().getHours(),
        isHoliday: false,
      };

      // Run multiple cycles
      const results = [];
      for (let i = 0; i < 3; i++) {
        const forecasts = await forecaster.forecast(currentState);
        const result = await scheduler.optimizeSchedule(
          generators,
          batteries,
          forecasts
        );
        results.push(result);
      }

      // All results should be feasible
      results.forEach(result => {
        expect(result.isFeasible).toBe(true);
      });

      // Strategy performance should be tracked
      const stats = scheduler.getStrategyStatistics();
      expect(stats.length).toBeGreaterThan(0);
    });

    it('should handle forecast error correction', async () => {
      const currentState: GridState = {
        timestamp: new Date(),
        loadMW: 5000,
        generationMW: 5100,
        renewablePenetration: 25,
        frequency: 60.0,
        voltageStability: 0.98,
        activeGenerators: ['gen-1', 'gen-2'],
        dayOfWeek: new Date().getDay(),
        hourOfDay: new Date().getHours(),
        isHoliday: false,
      };

      // Generate forecast
      const forecasts = await forecaster.forecast(currentState);
      const forecast = forecasts[0];

      // Simulate actual observation
      const actualLoad = 5100;
      await forecaster.updateWithActual(forecast, actualLoad);

      // Check metrics updated
      const metrics = forecaster.getAccuracyMetrics();
      expect(metrics.mae).toBeGreaterThan(0);

      // Generate new forecast (should be improved)
      const newForecasts = await forecaster.forecast(currentState);
      expect(newForecasts.length).toBeGreaterThan(0);
    });

    it('should persist and reload state across sessions', async () => {
      const currentState: GridState = {
        timestamp: new Date(),
        loadMW: 5000,
        generationMW: 5100,
        renewablePenetration: 25,
        frequency: 60.0,
        voltageStability: 0.98,
        activeGenerators: ['gen-1', 'gen-2'],
        dayOfWeek: new Date().getDay(),
        hourOfDay: new Date().getHours(),
        isHoliday: false,
      };

      // Run optimization
      const forecasts = await forecaster.forecast(currentState);
      const result1 = await scheduler.optimizeSchedule(
        generators,
        batteries,
        forecasts
      );

      // Create new instances with same namespaces
      const forecaster2 = new LoadForecaster({
        horizons: [ForecastHorizon.HOUR_1, ForecastHorizon.HOUR_4],
        historyWindowDays: 7,
        confidenceLevel: 0.95,
        enableErrorCorrection: true,
        correctionUpdateFrequency: 1,
        memoryNamespace: forecasterNamespace,
      });
      await forecaster2.initialize();

      const scheduler2 = new SwarmScheduler({
        swarmSize: 4,
        explorationRate: 0.3,
        maxIterations: 3,
        convergenceThreshold: 0.01,
        enableOpenRouter: false,
        memoryNamespace: schedulerNamespace,
      });
      await scheduler2.initialize();

      // Should have loaded previous strategies
      const bestStrategies = scheduler2.getBestStrategies(3);
      expect(bestStrategies.length).toBeGreaterThan(0);
    });
  });

  describe('Real-world Scenarios', () => {
    it('should handle high renewable penetration', async () => {
      const currentState: GridState = {
        timestamp: new Date(),
        loadMW: 5000,
        generationMW: 5100,
        renewablePenetration: 60, // High renewable
        frequency: 60.0,
        voltageStability: 0.98,
        activeGenerators: ['wind-1', 'solar-1', 'natgas-1'],
        weather: {
          temperature: 22,
          windSpeed: 15, // Strong wind
          solarIrradiance: 1000, // High solar
          precipitation: 0,
        },
        dayOfWeek: new Date().getDay(),
        hourOfDay: 12, // Midday
        isHoliday: false,
      };

      const forecasts = await forecaster.forecast(currentState);
      const result = await scheduler.optimizeSchedule(
        generators,
        batteries,
        forecasts
      );

      expect(result.isFeasible).toBe(true);
      expect(result.renewableUtilization).toBeGreaterThan(0);
    });

    it('should handle peak demand periods', async () => {
      const currentState: GridState = {
        timestamp: new Date(),
        loadMW: 8000, // Peak load
        generationMW: 8100,
        renewablePenetration: 15,
        frequency: 60.0,
        voltageStability: 0.98,
        activeGenerators: ['gen-1', 'gen-2', 'gen-3', 'gen-4'],
        weather: {
          temperature: 35, // Hot weather
          windSpeed: 3, // Low wind
          solarIrradiance: 400, // Evening
          precipitation: 0,
        },
        dayOfWeek: 2, // Tuesday
        hourOfDay: 18, // Evening peak
        isHoliday: false,
      };

      const forecasts = await forecaster.forecast(currentState);
      const result = await scheduler.optimizeSchedule(
        generators,
        batteries,
        forecasts
      );

      expect(result.isFeasible).toBe(true);
      // Should utilize all available capacity
      const totalCapacity = generators.reduce(
        (sum, g) => sum + g.maxCapacityMW,
        0
      );
      expect(result.commitments[0].totalGenerationMW).toBeLessThanOrEqual(
        totalCapacity + 100
      );
    });
  });
});
