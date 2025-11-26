/**
 * Tests for forecasting models
 */

import { ARIMAModel, LSTMModel, TransformerModel, ProphetModel, createModel } from '../src/models';
import { ModelType, TimeSeriesPoint } from '../src/types';

describe('Forecasting Models', () => {
  let testData: TimeSeriesPoint[];

  beforeEach(() => {
    // Generate synthetic time series data with trend and seasonality
    testData = Array.from({ length: 200 }, (_, i) => ({
      timestamp: Date.now() + i * 3600000,
      value: 100 + i * 0.5 + 20 * Math.sin((i * 2 * Math.PI) / 24)
    }));
  });

  describe('ARIMAModel', () => {
    it('should train successfully', async () => {
      const model = new ARIMAModel(0.3, 0.1);
      await expect(model.train(testData)).resolves.not.toThrow();
    });

    it('should predict future values', async () => {
      const model = new ARIMAModel(0.3, 0.1);
      await model.train(testData);

      const predictions = await model.predict(24);
      expect(predictions).toHaveLength(24);
      expect(predictions.every(p => typeof p === 'number')).toBe(true);
    });

    it('should throw error if not trained', async () => {
      const model = new ARIMAModel();
      await expect(model.predict(10)).rejects.toThrow('Model not trained');
    });

    it('should clone successfully', async () => {
      const model = new ARIMAModel(0.3, 0.1);
      await model.train(testData);

      const cloned = model.clone();
      expect(cloned.modelType).toBe(ModelType.ARIMA);

      const originalPredictions = await model.predict(10);
      const clonedPredictions = await cloned.predict(10);

      expect(clonedPredictions).toEqual(originalPredictions);
    });
  });

  describe('LSTMModel', () => {
    it('should train successfully', async () => {
      const model = new LSTMModel(24);
      await expect(model.train(testData)).resolves.not.toThrow();
    });

    it('should predict future values', async () => {
      const model = new LSTMModel(24);
      await model.train(testData);

      const predictions = await model.predict(24);
      expect(predictions).toHaveLength(24);
      expect(predictions.every(p => typeof p === 'number')).toBe(true);
    });

    it('should require minimum lookback samples', async () => {
      const model = new LSTMModel(100);
      await expect(model.train(testData.slice(0, 50))).rejects.toThrow();
    });
  });

  describe('TransformerModel', () => {
    it('should train successfully', async () => {
      const model = new TransformerModel(48);
      await expect(model.train(testData)).resolves.not.toThrow();
    });

    it('should predict future values', async () => {
      const model = new TransformerModel(48);
      await model.train(testData);

      const predictions = await model.predict(24);
      expect(predictions).toHaveLength(24);
      expect(predictions.every(p => typeof p === 'number')).toBe(true);
    });
  });

  describe('ProphetModel', () => {
    it('should train successfully', async () => {
      const model = new ProphetModel(24);
      await expect(model.train(testData)).resolves.not.toThrow();
    });

    it('should predict future values with trend and seasonality', async () => {
      const model = new ProphetModel(24);
      await model.train(testData);

      const predictions = await model.predict(48);
      expect(predictions).toHaveLength(48);
      expect(predictions.every(p => typeof p === 'number')).toBe(true);

      // Check that predictions follow seasonal pattern
      const firstPeriod = predictions.slice(0, 24);
      const secondPeriod = predictions.slice(24, 48);

      // Seasonal patterns should be similar (allowing for trend)
      for (let i = 0; i < 24; i++) {
        const diff = Math.abs(secondPeriod[i] - firstPeriod[i]);
        expect(diff).toBeLessThan(50); // Allow for trend growth
      }
    });

    it('should detect trend correctly', async () => {
      // Create data with strong upward trend
      const trendData = Array.from({ length: 100 }, (_, i) => ({
        timestamp: Date.now() + i * 3600000,
        value: 100 + i * 2 // Strong linear trend
      }));

      const model = new ProphetModel(24);
      await model.train(trendData);

      const predictions = await model.predict(10);

      // Predictions should follow upward trend
      for (let i = 1; i < predictions.length; i++) {
        expect(predictions[i]).toBeGreaterThan(predictions[i - 1]);
      }
    });
  });

  describe('Model Factory', () => {
    it('should create ARIMA model', () => {
      const model = createModel(ModelType.ARIMA, { alpha: 0.4, beta: 0.2 });
      expect(model.modelType).toBe(ModelType.ARIMA);
    });

    it('should create LSTM model', () => {
      const model = createModel(ModelType.LSTM, { lookback: 48 });
      expect(model.modelType).toBe(ModelType.LSTM);
    });

    it('should create Transformer model', () => {
      const model = createModel(ModelType.TRANSFORMER, { sequenceLength: 72 });
      expect(model.modelType).toBe(ModelType.TRANSFORMER);
    });

    it('should create Prophet model', () => {
      const model = createModel(ModelType.PROPHET, { seasonalPeriod: 48 });
      expect(model.modelType).toBe(ModelType.PROPHET);
    });

    it('should use default hyperparameters', () => {
      const model = createModel(ModelType.ARIMA);
      expect(model.modelType).toBe(ModelType.ARIMA);
    });
  });

  describe('Model Performance', () => {
    it('should return performance metrics', async () => {
      const model = new ARIMAModel();
      await model.train(testData);

      const performance = model.getPerformance();
      expect(performance).toHaveProperty('modelName');
      expect(performance).toHaveProperty('mape');
      expect(performance).toHaveProperty('rmse');
      expect(performance).toHaveProperty('mae');
      expect(performance).toHaveProperty('coverage');
      expect(performance).toHaveProperty('lastUpdated');
    });
  });
});
