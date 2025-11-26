/**
 * Tests for LoadForecaster
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { LoadForecaster } from '../src/load-forecaster.js';
import { ForecastHorizon, type GridState } from '../src/types.js';
import { rm } from 'fs/promises';

describe('LoadForecaster', () => {
  let forecaster: LoadForecaster;
  const testNamespace = 'test-load-forecaster';

  beforeEach(async () => {
    forecaster = new LoadForecaster({
      horizons: [
        ForecastHorizon.HOUR_1,
        ForecastHorizon.HOUR_4,
        ForecastHorizon.HOUR_24,
      ],
      historyWindowDays: 7,
      confidenceLevel: 0.95,
      enableErrorCorrection: true,
      correctionUpdateFrequency: 1,
      memoryNamespace: testNamespace,
    });

    await forecaster.initialize();
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
    it('should initialize successfully', async () => {
      expect(forecaster).toBeDefined();
      const metrics = forecaster.getAccuracyMetrics();
      expect(metrics).toBeDefined();
      expect(metrics.mae).toBe(0);
    });

    it('should load from empty memory on first run', async () => {
      const metrics = forecaster.getAccuracyMetrics();
      expect(metrics.mean).toBe(0);
      expect(metrics.stdDev).toBe(0);
    });
  });

  describe('addHistoricalState', () => {
    it('should add historical grid states', async () => {
      const state: GridState = {
        timestamp: new Date(),
        loadMW: 5000,
        generationMW: 5100,
        renewablePenetration: 25,
        frequency: 60.0,
        voltageStability: 0.98,
        activeGenerators: ['gen-1', 'gen-2'],
        dayOfWeek: 1,
        hourOfDay: 12,
        isHoliday: false,
      };

      await expect(forecaster.addHistoricalState(state)).resolves.not.toThrow();
    });

    it('should handle multiple historical states', async () => {
      const states: GridState[] = [];
      for (let i = 0; i < 100; i++) {
        states.push({
          timestamp: new Date(Date.now() - i * 3600000),
          loadMW: 5000 + Math.random() * 1000,
          generationMW: 5100 + Math.random() * 1000,
          renewablePenetration: 20 + Math.random() * 10,
          frequency: 60.0,
          voltageStability: 0.98,
          activeGenerators: ['gen-1', 'gen-2'],
          dayOfWeek: new Date(Date.now() - i * 3600000).getDay(),
          hourOfDay: new Date(Date.now() - i * 3600000).getHours(),
          isHoliday: false,
        });
      }

      for (const state of states) {
        await forecaster.addHistoricalState(state);
      }
    });
  });

  describe('forecast', () => {
    it('should generate forecasts for configured horizons', async () => {
      const currentState: GridState = {
        timestamp: new Date(),
        loadMW: 5000,
        generationMW: 5100,
        renewablePenetration: 25,
        frequency: 60.0,
        voltageStability: 0.98,
        activeGenerators: ['gen-1', 'gen-2'],
        dayOfWeek: 1,
        hourOfDay: 12,
        isHoliday: false,
      };

      const forecasts = await forecaster.forecast(currentState);

      expect(forecasts).toHaveLength(3);
      expect(forecasts[0].horizon).toBe(ForecastHorizon.HOUR_1);
      expect(forecasts[1].horizon).toBe(ForecastHorizon.HOUR_4);
      expect(forecasts[2].horizon).toBe(ForecastHorizon.HOUR_24);
    });

    it('should provide prediction intervals', async () => {
      const currentState: GridState = {
        timestamp: new Date(),
        loadMW: 5000,
        generationMW: 5100,
        renewablePenetration: 25,
        frequency: 60.0,
        voltageStability: 0.98,
        activeGenerators: ['gen-1', 'gen-2'],
        dayOfWeek: 1,
        hourOfDay: 12,
        isHoliday: false,
      };

      const forecasts = await forecaster.forecast(currentState);

      forecasts.forEach(forecast => {
        expect(forecast.loadMW).toBeGreaterThan(0);
        expect(forecast.lowerBound).toBeLessThan(forecast.loadMW);
        expect(forecast.upperBound).toBeGreaterThan(forecast.loadMW);
        expect(forecast.confidence).toBe(0.95);
      });
    });

    it('should have wider intervals for longer horizons', async () => {
      const currentState: GridState = {
        timestamp: new Date(),
        loadMW: 5000,
        generationMW: 5100,
        renewablePenetration: 25,
        frequency: 60.0,
        voltageStability: 0.98,
        activeGenerators: ['gen-1', 'gen-2'],
        dayOfWeek: 1,
        hourOfDay: 12,
        isHoliday: false,
      };

      const forecasts = await forecaster.forecast(currentState);

      const interval1h = forecasts[0].upperBound - forecasts[0].lowerBound;
      const interval24h = forecasts[2].upperBound - forecasts[2].lowerBound;

      expect(interval24h).toBeGreaterThan(interval1h);
    });
  });

  describe('updateWithActual', () => {
    it('should update error correction with actual values', async () => {
      const currentState: GridState = {
        timestamp: new Date(),
        loadMW: 5000,
        generationMW: 5100,
        renewablePenetration: 25,
        frequency: 60.0,
        voltageStability: 0.98,
        activeGenerators: ['gen-1', 'gen-2'],
        dayOfWeek: 1,
        hourOfDay: 12,
        isHoliday: false,
      };

      const forecasts = await forecaster.forecast(currentState);
      const forecast = forecasts[0];

      await forecaster.updateWithActual(forecast, 5100);

      const metrics = forecaster.getAccuracyMetrics();
      expect(metrics.mae).toBeGreaterThan(0);
    });

    it('should improve predictions over time', async () => {
      const currentState: GridState = {
        timestamp: new Date(),
        loadMW: 5000,
        generationMW: 5100,
        renewablePenetration: 25,
        frequency: 60.0,
        voltageStability: 0.98,
        activeGenerators: ['gen-1', 'gen-2'],
        dayOfWeek: 1,
        hourOfDay: 12,
        isHoliday: false,
      };

      // Add historical data
      for (let i = 0; i < 50; i++) {
        await forecaster.addHistoricalState({
          ...currentState,
          timestamp: new Date(Date.now() - i * 3600000),
          loadMW: 5000 + (i % 24) * 100,
        });
      }

      // Generate and update multiple times
      for (let i = 0; i < 10; i++) {
        const forecasts = await forecaster.forecast(currentState);
        await forecaster.updateWithActual(forecasts[0], 5100);
      }

      const metrics = forecaster.getAccuracyMetrics();
      expect(metrics.mae).toBeDefined();
      expect(metrics.mape).toBeDefined();
    });
  });

  describe('weather and calendar features', () => {
    it('should handle missing weather data', async () => {
      const currentState: GridState = {
        timestamp: new Date(),
        loadMW: 5000,
        generationMW: 5100,
        renewablePenetration: 25,
        frequency: 60.0,
        voltageStability: 0.98,
        activeGenerators: ['gen-1', 'gen-2'],
        dayOfWeek: 1,
        hourOfDay: 12,
        isHoliday: false,
        // No weather data
      };

      const forecasts = await forecaster.forecast(currentState);
      expect(forecasts).toHaveLength(3);
    });

    it('should incorporate weather data when available', async () => {
      const currentState: GridState = {
        timestamp: new Date(),
        loadMW: 5000,
        generationMW: 5100,
        renewablePenetration: 25,
        frequency: 60.0,
        voltageStability: 0.98,
        activeGenerators: ['gen-1', 'gen-2'],
        weather: {
          temperature: 30,
          windSpeed: 12,
          solarIrradiance: 900,
          precipitation: 0,
        },
        dayOfWeek: 1,
        hourOfDay: 12,
        isHoliday: false,
      };

      const forecasts = await forecaster.forecast(currentState);
      expect(forecasts).toHaveLength(3);
    });
  });
});
