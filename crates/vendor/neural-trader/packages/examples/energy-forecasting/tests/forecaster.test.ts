/**
 * Tests for main forecaster system
 */

import { EnergyForecaster } from '../src/forecaster';
import { EnergyDomain, TimeSeriesPoint, ModelType } from '../src/types';

describe('EnergyForecaster', () => {
  let testData: TimeSeriesPoint[];

  beforeEach(() => {
    // Generate comprehensive test data (7 days of hourly data)
    testData = Array.from({ length: 168 }, (_, i) => ({
      timestamp: Date.now() + i * 3600000,
      value: 100 + i * 0.2 + 20 * Math.sin((i * 2 * Math.PI) / 24)
    }));
  });

  describe('Initialization', () => {
    it('should create forecaster for solar domain', () => {
      const forecaster = new EnergyForecaster(EnergyDomain.SOLAR);
      expect(forecaster).toBeDefined();
    });

    it('should create forecaster with custom config', () => {
      const forecaster = new EnergyForecaster(EnergyDomain.WIND, {
        alpha: 0.05,
        horizon: 48,
        seasonalPeriod: 24,
        enableAdaptive: true
      });
      expect(forecaster).toBeDefined();
    });

    it('should create forecaster with ensemble config', () => {
      const forecaster = new EnergyForecaster(EnergyDomain.DEMAND, {
        ensembleConfig: {
          models: [ModelType.ARIMA, ModelType.PROPHET]
        }
      });
      expect(forecaster).toBeDefined();
    });
  });

  describe('Training', () => {
    it('should train successfully', async () => {
      const forecaster = new EnergyForecaster(EnergyDomain.SOLAR, {
        ensembleConfig: {
          models: [ModelType.ARIMA, ModelType.LSTM]
        }
      });

      await expect(forecaster.train(testData)).resolves.not.toThrow();
    }, 90000);

    it('should throw error with insufficient data', async () => {
      const forecaster = new EnergyForecaster(EnergyDomain.SOLAR);
      const smallData = testData.slice(0, 50);

      await expect(forecaster.train(smallData)).rejects.toThrow(
        'Need at least 100 data points'
      );
    });

    it('should detect seasonal patterns', async () => {
      const forecaster = new EnergyForecaster(EnergyDomain.SOLAR);
      await forecaster.train(testData);

      const stats = forecaster.getStats();
      expect(stats.seasonalPattern).not.toBeNull();
      expect(stats.seasonalPattern?.period).toBe(24);
      expect(stats.seasonalPattern?.strength).toBeGreaterThan(0);
    }, 90000);
  });

  describe('Forecasting', () => {
    let forecaster: EnergyForecaster;

    beforeEach(async () => {
      forecaster = new EnergyForecaster(EnergyDomain.SOLAR, {
        horizon: 24,
        ensembleConfig: {
          models: [ModelType.ARIMA]
        }
      });
      await forecaster.train(testData);
    }, 90000);

    it('should generate multi-step forecast', async () => {
      const forecast = await forecaster.forecast();

      expect(forecast).toHaveProperty('forecasts');
      expect(forecast).toHaveProperty('horizon');
      expect(forecast).toHaveProperty('generatedAt');
      expect(forecast).toHaveProperty('modelPerformance');

      expect(forecast.forecasts).toHaveLength(24);
    });

    it('should generate forecast with custom horizon', async () => {
      const forecast = await forecaster.forecast(48);

      expect(forecast.forecasts).toHaveLength(48);
      expect(forecast.horizon).toBe(48);
    });

    it('should include prediction intervals', async () => {
      const forecast = await forecaster.forecast(12);

      forecast.forecasts.forEach(f => {
        expect(f).toHaveProperty('interval');
        expect(f.interval.lower).toBeLessThan(f.pointForecast);
        expect(f.interval.upper).toBeGreaterThan(f.pointForecast);
        expect(f.interval.coverage()).toBeGreaterThan(0.8);
      });
    });

    it('should include seasonal components', async () => {
      const forecast = await forecaster.forecast(24);

      const hasSeasonalComponents = forecast.forecasts.some(
        f => f.seasonalComponent !== undefined
      );
      expect(hasSeasonalComponents).toBe(true);
    });

    it('should include metadata', async () => {
      const forecast = await forecaster.forecast(6);

      forecast.forecasts.forEach(f => {
        expect(f.metadata).toBeDefined();
        expect(f.metadata?.selectedModel).toBeDefined();
        expect(f.metadata?.horizon).toBeDefined();
        expect(f.metadata?.domain).toBe(EnergyDomain.SOLAR);
      });
    });

    it('should throw error if not trained', async () => {
      const untrainedForecaster = new EnergyForecaster(EnergyDomain.WIND);
      await expect(untrainedForecaster.forecast()).rejects.toThrow(
        'Forecaster not trained'
      );
    });
  });

  describe('Online Learning', () => {
    let forecaster: EnergyForecaster;

    beforeEach(async () => {
      forecaster = new EnergyForecaster(EnergyDomain.DEMAND, {
        enableAdaptive: true,
        ensembleConfig: {
          models: [ModelType.ARIMA]
        }
      });
      await forecaster.train(testData);
    }, 90000);

    it('should update with new observation', async () => {
      const newObservation: TimeSeriesPoint = {
        timestamp: Date.now() + 169 * 3600000,
        value: 150
      };

      await expect(forecaster.update(newObservation)).resolves.not.toThrow();
    });

    it('should maintain training data size', async () => {
      const statsBefore = forecaster.getStats();
      const initialSize = statsBefore.trainingPoints;

      // Add many observations
      for (let i = 0; i < 100; i++) {
        await forecaster.update({
          timestamp: Date.now() + (169 + i) * 3600000,
          value: 150 + i * 0.1
        });
      }

      const statsAfter = forecaster.getStats();
      expect(statsAfter.trainingPoints).toBeGreaterThan(initialSize);
      expect(statsAfter.trainingPoints).toBeLessThanOrEqual(10000); // Max size
    });
  });

  describe('Statistics', () => {
    it('should provide comprehensive statistics', async () => {
      const forecaster = new EnergyForecaster(EnergyDomain.TEMPERATURE, {
        ensembleConfig: {
          models: [ModelType.ARIMA, ModelType.PROPHET]
        }
      });
      await forecaster.train(testData);

      const stats = forecaster.getStats();

      expect(stats).toHaveProperty('domain');
      expect(stats).toHaveProperty('trainingPoints');
      expect(stats).toHaveProperty('ensembleStats');
      expect(stats).toHaveProperty('conformalStats');
      expect(stats).toHaveProperty('seasonalPattern');

      expect(stats.domain).toBe(EnergyDomain.TEMPERATURE);
      expect(stats.trainingPoints).toBeGreaterThan(0);
      expect(stats.conformalStats.length).toBeGreaterThan(0);
    }, 90000);

    it('should include model performance metrics', async () => {
      const forecaster = new EnergyForecaster(EnergyDomain.SOLAR, {
        ensembleConfig: {
          models: [ModelType.ARIMA]
        }
      });
      await forecaster.train(testData);

      const forecast = await forecaster.forecast(12);

      expect(forecast.modelPerformance).toBeDefined();
      expect(forecast.modelPerformance).toHaveProperty('modelName');
      expect(forecast.modelPerformance).toHaveProperty('mape');
      expect(forecast.modelPerformance).toHaveProperty('rmse');
      expect(forecast.modelPerformance).toHaveProperty('mae');
      expect(forecast.modelPerformance).toHaveProperty('coverage');
      expect(forecast.modelPerformance).toHaveProperty('intervalWidth');
    }, 90000);
  });

  describe('Reset', () => {
    it('should reset forecaster state', async () => {
      const forecaster = new EnergyForecaster(EnergyDomain.SOLAR, {
        ensembleConfig: {
          models: [ModelType.ARIMA]
        }
      });
      await forecaster.train(testData);

      const statsBefore = forecaster.getStats();
      expect(statsBefore.trainingPoints).toBeGreaterThan(0);

      await forecaster.reset();

      const statsAfter = forecaster.getStats();
      expect(statsAfter.trainingPoints).toBe(0);
    }, 90000);
  });

  describe('Multi-Domain Support', () => {
    it('should support solar generation forecasting', async () => {
      const forecaster = new EnergyForecaster(EnergyDomain.SOLAR);
      await forecaster.train(testData);

      const forecast = await forecaster.forecast(24);
      expect(forecast.forecasts[0].metadata?.domain).toBe(EnergyDomain.SOLAR);
    }, 90000);

    it('should support wind power forecasting', async () => {
      const forecaster = new EnergyForecaster(EnergyDomain.WIND);
      await forecaster.train(testData);

      const forecast = await forecaster.forecast(24);
      expect(forecast.forecasts[0].metadata?.domain).toBe(EnergyDomain.WIND);
    }, 90000);

    it('should support electricity demand forecasting', async () => {
      const forecaster = new EnergyForecaster(EnergyDomain.DEMAND);
      await forecaster.train(testData);

      const forecast = await forecaster.forecast(24);
      expect(forecast.forecasts[0].metadata?.domain).toBe(EnergyDomain.DEMAND);
    }, 90000);

    it('should support temperature prediction', async () => {
      const forecaster = new EnergyForecaster(EnergyDomain.TEMPERATURE);
      await forecaster.train(testData);

      const forecast = await forecaster.forecast(24);
      expect(forecast.forecasts[0].metadata?.domain).toBe(EnergyDomain.TEMPERATURE);
    }, 90000);
  });
});
