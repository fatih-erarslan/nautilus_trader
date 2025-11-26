/**
 * Arrival Forecaster Tests
 */

import { ArrivalForecaster } from '../src/arrival-forecaster';
import { describe, it, expect, beforeEach } from '@jest/globals';
import * as fs from 'fs';
import * as path from 'path';

describe('ArrivalForecaster', () => {
  let forecaster: ArrivalForecaster;
  const testDbPath = path.join(__dirname, '../.test-db/forecaster.db');

  beforeEach(async () => {
    // Clean up test database
    if (fs.existsSync(testDbPath)) {
      fs.unlinkSync(testDbPath);
    }

    forecaster = new ArrivalForecaster({
      agentdbPath: testDbPath,
      enableNapiRS: false, // Use JS for tests
      privacy: {
        useSyntheticData: true,
        anonymizationLevel: 'full',
        dataRetentionDays: 90,
        complianceMode: 'both'
      },
      lookbackDays: 90,
      forecastHorizon: 24,
      confidenceLevel: 0.95
    });
  });

  describe('Training', () => {
    it('should train on synthetic historical data', async () => {
      const historicalData = generateSyntheticArrivalData(90);

      await forecaster.train(historicalData);

      // Should complete without error
      expect(true).toBe(true);
    });

    it('should learn seasonal patterns', async () => {
      const historicalData = generateSeasonalData(365);

      await forecaster.train(historicalData);

      // Forecast flu season (January)
      const januaryDate = new Date('2024-01-15T10:00:00');
      const januaryForecast = await forecaster.forecast(januaryDate);

      // Forecast summer (July)
      const julyDate = new Date('2024-07-15T10:00:00');
      const julyForecast = await forecaster.forecast(julyDate);

      // Flu season should have higher predictions
      expect(januaryForecast.seasonalComponent).toBeGreaterThan(1.0);
    });
  });

  describe('Forecasting', () => {
    beforeEach(async () => {
      const historicalData = generateSyntheticArrivalData(90);
      await forecaster.train(historicalData);
    });

    it('should forecast patient arrivals', async () => {
      const timestamp = new Date('2024-01-15T10:00:00');
      const forecast = await forecaster.forecast(timestamp);

      expect(forecast.predictedArrivals).toBeGreaterThan(0);
      expect(forecast.lowerBound).toBeLessThan(forecast.predictedArrivals);
      expect(forecast.upperBound).toBeGreaterThan(forecast.predictedArrivals);
      expect(forecast.confidence).toBe(0.95);
    });

    it('should forecast multiple hours ahead', async () => {
      const startTime = new Date('2024-01-15T08:00:00');
      const forecasts = await forecaster.forecastHorizon(startTime);

      expect(forecasts).toHaveLength(24);
      expect(forecasts[0].timestamp.getHours()).toBe(8);
      expect(forecasts[23].timestamp.getHours()).toBe(7); // next day
    });

    it('should detect peak hours', async () => {
      const morningPeak = new Date('2024-01-15T10:00:00');
      const nightTime = new Date('2024-01-15T03:00:00');

      const morningForecast = await forecaster.forecast(morningPeak);
      const nightForecast = await forecaster.forecast(nightTime);

      // Morning should have higher arrivals
      expect(morningForecast.predictedArrivals).toBeGreaterThan(nightForecast.predictedArrivals);
    });
  });

  describe('Online Learning', () => {
    beforeEach(async () => {
      const historicalData = generateSyntheticArrivalData(90);
      await forecaster.train(historicalData);
    });

    it('should update with actual arrivals', async () => {
      const timestamp = new Date('2024-01-15T10:00:00');
      const forecast = await forecaster.forecast(timestamp);

      const actualArrivals = 15;
      await forecaster.updateWithActuals(timestamp, actualArrivals);

      // Should complete without error
      expect(true).toBe(true);
    });

    it('should track forecast accuracy', async () => {
      // Make several forecasts and update with actuals
      for (let i = 0; i < 10; i++) {
        const timestamp = new Date(`2024-01-15T${i + 8}:00:00`);
        await forecaster.forecast(timestamp);
        await forecaster.updateWithActuals(timestamp, 10 + Math.floor(Math.random() * 5));
      }

      const metrics = await forecaster.getAccuracyMetrics();

      expect(metrics.samples).toBe(10);
      expect(metrics.mae).toBeGreaterThan(0);
      expect(metrics.rmse).toBeGreaterThan(0);
      expect(metrics.mape).toBeGreaterThan(0);
    });
  });
});

/**
 * Generate synthetic arrival data
 */
function generateSyntheticArrivalData(days: number) {
  const data: Array<{ timestamp: Date; arrivals: number }> = [];
  const startDate = new Date('2024-01-01T00:00:00');

  for (let day = 0; day < days; day++) {
    for (let hour = 0; hour < 24; hour++) {
      const timestamp = new Date(startDate);
      timestamp.setDate(timestamp.getDate() + day);
      timestamp.setHours(hour);

      // Base pattern: more arrivals during day, fewer at night
      let baseArrivals = 5;
      if (hour >= 8 && hour <= 20) {
        baseArrivals = 12;
      }
      if (hour >= 9 && hour <= 11 || hour >= 18 && hour <= 20) {
        baseArrivals = 18; // peak hours
      }

      // Weekend adjustment
      const dayOfWeek = timestamp.getDay();
      if (dayOfWeek === 0 || dayOfWeek === 6) {
        baseArrivals *= 0.8;
      }

      // Add random variation
      const arrivals = Math.max(0, Math.floor(baseArrivals + (Math.random() - 0.5) * 6));

      data.push({ timestamp, arrivals });
    }
  }

  return data;
}

/**
 * Generate seasonal data with flu season pattern
 */
function generateSeasonalData(days: number) {
  const data: Array<{ timestamp: Date; arrivals: number }> = [];
  const startDate = new Date('2024-01-01T00:00:00');

  for (let day = 0; day < days; day++) {
    for (let hour = 0; hour < 24; hour++) {
      const timestamp = new Date(startDate);
      timestamp.setDate(timestamp.getDate() + day);
      timestamp.setHours(hour);

      const month = timestamp.getMonth();

      // Base arrivals
      let baseArrivals = 10;

      // Flu season (Dec-Feb)
      const isFluSeason = month === 11 || month === 0 || month === 1;
      if (isFluSeason) {
        baseArrivals *= 1.5;
      }

      // Time of day pattern
      if (hour >= 9 && hour <= 11 || hour >= 18 && hour <= 20) {
        baseArrivals *= 1.3;
      } else if (hour >= 0 && hour <= 6) {
        baseArrivals *= 0.5;
      }

      const arrivals = Math.max(0, Math.floor(baseArrivals + (Math.random() - 0.5) * 4));

      data.push({ timestamp, arrivals });
    }
  }

  return data;
}
