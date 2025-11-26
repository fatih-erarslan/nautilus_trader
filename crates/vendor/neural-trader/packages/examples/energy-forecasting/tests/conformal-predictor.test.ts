/**
 * Tests for conformal predictor wrapper
 */

import { EnergyConformalPredictor } from '../src/conformal-predictor';
import { ModelType, TimeSeriesPoint } from '../src/types';

describe('EnergyConformalPredictor', () => {
  let testData: TimeSeriesPoint[];
  let predictions: number[];

  beforeEach(() => {
    // Generate test data
    testData = Array.from({ length: 100 }, (_, i) => ({
      timestamp: Date.now() + i * 3600000,
      value: 100 + i * 0.5 + 10 * Math.sin((i * 2 * Math.PI) / 24)
    }));

    // Generate predictions (with some error)
    predictions = testData.map(d => d.value + (Math.random() - 0.5) * 5);
  });

  describe('Initialization', () => {
    it('should initialize successfully', async () => {
      const predictor = new EnergyConformalPredictor(ModelType.ARIMA);
      await expect(predictor.initialize()).resolves.not.toThrow();
      expect(predictor.isReady()).toBe(false); // Not calibrated yet
    });

    it('should initialize with adaptive mode', async () => {
      const predictor = new EnergyConformalPredictor(ModelType.LSTM, {}, true);
      await expect(predictor.initialize()).resolves.not.toThrow();
    });
  });

  describe('Calibration', () => {
    it('should calibrate with historical data', async () => {
      const predictor = new EnergyConformalPredictor(ModelType.ARIMA);
      await predictor.initialize();
      await predictor.calibrate(testData, predictions);

      expect(predictor.isReady()).toBe(true);
    });

    it('should throw error if not initialized', async () => {
      const predictor = new EnergyConformalPredictor(ModelType.ARIMA);
      await expect(predictor.calibrate(testData, predictions)).rejects.toThrow(
        'Predictor not initialized'
      );
    });

    it('should validate data length match', async () => {
      const predictor = new EnergyConformalPredictor(ModelType.ARIMA);
      await predictor.initialize();

      await expect(
        predictor.calibrate(testData, predictions.slice(0, 50))
      ).rejects.toThrow('Historical data and predictions must have same length');
    });
  });

  describe('Forecasting', () => {
    let predictor: EnergyConformalPredictor;

    beforeEach(async () => {
      predictor = new EnergyConformalPredictor(ModelType.ARIMA);
      await predictor.initialize();
      await predictor.calibrate(testData, predictions);
    });

    it('should generate forecast with prediction interval', async () => {
      const forecast = await predictor.forecast(150, Date.now());

      expect(forecast).toHaveProperty('timestamp');
      expect(forecast).toHaveProperty('pointForecast');
      expect(forecast).toHaveProperty('interval');
      expect(forecast).toHaveProperty('modelUsed');
      expect(forecast).toHaveProperty('confidence');

      expect(forecast.pointForecast).toBe(150);
      expect(forecast.interval.lower).toBeLessThan(150);
      expect(forecast.interval.upper).toBeGreaterThan(150);
      expect(forecast.confidence).toBeGreaterThan(0.8);
      expect(forecast.confidence).toBeLessThanOrEqual(1);
    });

    it('should include metadata', async () => {
      const forecast = await predictor.forecast(150, Date.now());

      expect(forecast.metadata).toBeDefined();
      expect(forecast.metadata?.implementationType).toBeDefined();
      expect(forecast.metadata?.intervalWidth).toBeDefined();
    });

    it('should update with new observation', async () => {
      const statsBefore = predictor.getStats();

      await predictor.update(150, 148, Date.now());

      const statsAfter = predictor.getStats();
      expect(statsAfter.calibrationPoints).toBe(statsBefore.calibrationPoints + 1);
    });
  });

  describe('Adaptive Forecasting', () => {
    let adaptivePredictor: EnergyConformalPredictor;

    beforeEach(async () => {
      adaptivePredictor = new EnergyConformalPredictor(ModelType.ARIMA, {}, true);
      await adaptivePredictor.initialize();
      await adaptivePredictor.calibrate(testData, predictions);
    });

    it('should adapt with actual values', async () => {
      const forecast = await adaptivePredictor.forecast(150, Date.now(), 148);

      expect(forecast).toBeDefined();
      expect(forecast.interval).toBeDefined();
    });
  });

  describe('Statistics', () => {
    it('should return null stats if not initialized', () => {
      const predictor = new EnergyConformalPredictor(ModelType.ARIMA);
      expect(predictor.getStats()).toBeNull();
    });

    it('should return comprehensive stats after calibration', async () => {
      const predictor = new EnergyConformalPredictor(ModelType.LSTM);
      await predictor.initialize();
      await predictor.calibrate(testData, predictions);

      const stats = predictor.getStats();

      expect(stats).toHaveProperty('modelType');
      expect(stats).toHaveProperty('implementationType');
      expect(stats).toHaveProperty('isAdaptive');
      expect(stats).toHaveProperty('calibrationPoints');
      expect(stats).toHaveProperty('recentPerformance');

      expect(stats.modelType).toBe(ModelType.LSTM);
      expect(stats.calibrationPoints).toBeGreaterThan(0);
    });
  });

  describe('Coverage', () => {
    it('should compute empirical coverage', async () => {
      const predictor = new EnergyConformalPredictor(ModelType.ARIMA);
      await predictor.initialize();
      await predictor.calibrate(testData, predictions);

      const actuals = testData.map(d => d.value);
      const coverage = predictor.computeCoverage(predictions, actuals);

      expect(coverage).toBeGreaterThanOrEqual(0);
      expect(coverage).toBeLessThanOrEqual(1);
      // With alpha=0.1, we expect ~90% coverage
      expect(coverage).toBeGreaterThan(0.85);
    });
  });

  describe('Reset', () => {
    it('should reset predictor state', async () => {
      const predictor = new EnergyConformalPredictor(ModelType.ARIMA);
      await predictor.initialize();
      await predictor.calibrate(testData, predictions);

      expect(predictor.isReady()).toBe(true);

      await predictor.reset();

      expect(predictor.isReady()).toBe(false);
    });
  });

  describe('Implementation Type', () => {
    it('should report implementation type', async () => {
      const predictor = new EnergyConformalPredictor(ModelType.ARIMA);
      await predictor.initialize();

      const implType = predictor.getImplementationType();
      expect(['native', 'wasm', 'pure']).toContain(implType);
    });
  });
});
