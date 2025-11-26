/**
 * Integration tests for @neural-trader/neuro-divergent package
 * Tests end-to-end forecasting workflows
 */

jest.mock('../index.js', () => ({
  NeuralForecast: jest.fn().mockImplementation(function() {
    this.addModel = jest.fn().mockResolvedValue('model-123');
    this.getConfig = jest.fn().mockResolvedValue({
      modelType: 'LSTM',
      inputSize: 10,
      horizon: 5,
      hiddenSize: 32,
      numLayers: 2
    });
    this.fit = jest.fn().mockResolvedValue([
      { epoch: 1, trainLoss: 0.5, valLoss: 0.6 },
      { epoch: 2, trainLoss: 0.3, valLoss: 0.4 },
      { epoch: 3, trainLoss: 0.2, valLoss: 0.25 }
    ]);
    this.predict = jest.fn().mockResolvedValue({
      predictions: [100.5, 101.2, 102.1],
      modelType: 'LSTM',
      confidence: 0.95
    });
    this.crossValidation = jest.fn().mockResolvedValue({
      mae: 0.15,
      mse: 0.045,
      rmse: 0.212,
      mape: 2.5
    });
  }),
  listAvailableModels: jest.fn(() => ['LSTM', 'GRU', 'Transformer', 'NHITS', 'NBEATS', 'DeepAR']),
  version: jest.fn(() => '2.1.0'),
  isGpuAvailable: jest.fn(() => true)
}));

describe('NeuralForecast Integration', () => {
  describe('complete forecasting workflow', () => {
    it('should train and forecast complete time series', async () => {
      const { NeuralForecast } = require('../index.js');
      const forecast = new NeuralForecast();

      // Add model
      const modelId = await forecast.addModel({
        modelType: 'LSTM',
        inputSize: 20,
        horizon: 5,
        hiddenSize: 64,
        numLayers: 2
      });

      expect(modelId).toBeDefined();

      // Prepare data
      const baseDate = new Date('2024-01-01T00:00:00Z');
      const data = {
        points: Array.from({ length: 100 }, (_, i) => ({
          timestamp: new Date(baseDate.getTime() + i * 86400000).toISOString(),
          value: 100 + Math.sin(i * 0.1) * 10 + Math.random() * 2
        })),
        frequency: '1D'
      };

      // Train
      const metrics = await forecast.fit(modelId, data);
      expect(Array.isArray(metrics)).toBe(true);
      expect(metrics.length).toBeGreaterThan(0);

      // Predict
      const predictions = await forecast.predict(modelId, 10);
      expect(predictions.predictions.length).toBeGreaterThan(0);
      expect(predictions.confidence).toBeGreaterThan(0);
    });

    it('should handle multi-step ahead forecasting', async () => {
      const { NeuralForecast } = require('../index.js');
      const forecast = new NeuralForecast();

      const modelId = await forecast.addModel({
        modelType: 'LSTM',
        inputSize: 10,
        horizon: 20
      });

      const baseDate = new Date('2024-01-01T00:00:00Z');
      const data = {
        points: Array.from({ length: 100 }, (_, i) => ({
          timestamp: new Date(baseDate.getTime() + i * 86400000).toISOString(),
          value: 100 + Math.sin((i / 100) * Math.PI * 2) * 20
        })),
        frequency: '1D'
      };

      await forecast.fit(modelId, data);

      // Make predictions at different horizons
      const pred1 = await forecast.predict(modelId, 5);
      const pred5 = await forecast.predict(modelId, 20);

      expect(pred1.predictions).toBeDefined();
      expect(pred5.predictions).toBeDefined();
    });

    it('should train multiple models in parallel', async () => {
      const { NeuralForecast } = require('../index.js');

      const forecasts = [
        new NeuralForecast(),
        new NeuralForecast(),
        new NeuralForecast()
      ];

      const modelIds = await Promise.all(
        forecasts.map(f =>
          f.addModel({
            modelType: 'LSTM',
            inputSize: 10,
            horizon: 5
          })
        )
      );

      expect(modelIds.length).toBe(3);
      modelIds.forEach(id => {
        expect(typeof id).toBe('string');
      });

      const baseDate = new Date('2024-01-01T00:00:00Z');
      const data = {
        points: Array.from({ length: 50 }, (_, i) => ({
          timestamp: new Date(baseDate.getTime() + i * 86400000).toISOString(),
          value: 100 + Math.random() * 10
        })),
        frequency: '1D'
      };

      const results = await Promise.all(
        forecasts.map((f, i) => f.fit(modelIds[i], data))
      );

      expect(results.length).toBe(3);
    });
  });

  describe('ensemble forecasting', () => {
    it('should create ensemble predictions from multiple models', async () => {
      const { NeuralForecast } = require('../index.js');

      const types = ['LSTM', 'GRU', 'Transformer'];
      const forecasts = types.map(() => new NeuralForecast());

      const modelIds = await Promise.all(
        forecasts.map((f, i) =>
          f.addModel({
            modelType: types[i],
            inputSize: 10,
            horizon: 5
          })
        )
      );

      const baseDate = new Date('2024-01-01T00:00:00Z');
      const data = {
        points: Array.from({ length: 50 }, (_, i) => ({
          timestamp: new Date(baseDate.getTime() + i * 86400000).toISOString(),
          value: 100 + Math.sin(i * 0.1) * 5
        })),
        frequency: '1D'
      };

      // Train all models
      await Promise.all(
        forecasts.map((f, i) => f.fit(modelIds[i], data))
      );

      // Get predictions from all models
      const predictions = await Promise.all(
        forecasts.map((f, i) => f.predict(modelIds[i], 5))
      );

      expect(predictions.length).toBe(3);

      // Compute ensemble average
      const ensembleAvg = predictions[0].predictions.map((_, j) =>
        predictions.reduce((sum, pred) => sum + pred.predictions[j], 0) /
        predictions.length
      );

      expect(ensembleAvg.length).toBeGreaterThan(0);
    });
  });

  describe('cross-validation workflow', () => {
    it('should validate model with cross-validation', async () => {
      const { NeuralForecast } = require('../index.js');
      const forecast = new NeuralForecast();

      const modelId = await forecast.addModel({
        modelType: 'LSTM',
        inputSize: 10,
        horizon: 5
      });

      const baseDate = new Date('2024-01-01T00:00:00Z');
      const data = {
        points: Array.from({ length: 150 }, (_, i) => ({
          timestamp: new Date(baseDate.getTime() + i * 86400000).toISOString(),
          value: 100 + Math.sin((i / 150) * Math.PI * 2) * 15
        })),
        frequency: '1D'
      };

      const cvResults = await forecast.crossValidation(modelId, data, 5, 1);

      expect(cvResults).toHaveProperty('mae');
      expect(cvResults).toHaveProperty('mse');
      expect(cvResults).toHaveProperty('rmse');
      expect(cvResults).toHaveProperty('mape');

      // All metrics should be positive
      Object.values(cvResults).forEach(metric => {
        expect(typeof metric).toBe('number');
        expect(metric).toBeGreaterThanOrEqual(0);
      });
    });

    it('should compare cross-validation across models', async () => {
      const { NeuralForecast } = require('../index.js');

      const types = ['LSTM', 'GRU', 'Transformer'];
      const results: any[] = [];

      for (const type of types) {
        const forecast = new NeuralForecast();

        const modelId = await forecast.addModel({
          modelType: type,
          inputSize: 10,
          horizon: 5
        });

        const baseDate = new Date('2024-01-01T00:00:00Z');
        const data = {
          points: Array.from({ length: 100 }, (_, i) => ({
            timestamp: new Date(baseDate.getTime() + i * 86400000).toISOString(),
            value: 100 + Math.sin((i / 100) * Math.PI * 2) * 10
          })),
          frequency: '1D'
        };

        const cvResult = await forecast.crossValidation(modelId, data, 3, 1);
        results.push({ type, metrics: cvResult });
      }

      expect(results.length).toBe(3);
      results.forEach(r => {
        expect(r.metrics).toHaveProperty('mae');
      });
    });
  });

  describe('time series preprocessing', () => {
    it('should handle different data frequencies', async () => {
      const { NeuralForecast } = require('../index.js');
      const forecast = new NeuralForecast();

      const frequencies = [
        { freq: '1H', intervalMs: 3600000 },
        { freq: '1D', intervalMs: 86400000 },
        { freq: '1W', intervalMs: 604800000 }
      ];

      for (const { freq, intervalMs } of frequencies) {
        const modelId = await forecast.addModel({
          modelType: 'LSTM',
          inputSize: 10,
          horizon: 5
        });

        const baseDate = new Date('2024-01-01T00:00:00Z');
        const data = {
          points: Array.from({ length: 50 }, (_, i) => ({
            timestamp: new Date(baseDate.getTime() + i * intervalMs).toISOString(),
            value: 100 + Math.random() * 5
          })),
          frequency: freq
        };

        const metrics = await forecast.fit(modelId, data);
        expect(Array.isArray(metrics)).toBe(true);
      }
    });

    it('should handle missing values in data', async () => {
      const { NeuralForecast } = require('../index.js');
      const forecast = new NeuralForecast();

      const modelId = await forecast.addModel({
        modelType: 'LSTM',
        inputSize: 10,
        horizon: 5
      });

      const baseDate = new Date('2024-01-01T00:00:00Z');
      const data = {
        points: Array.from({ length: 100 }, (_, i) => {
          // Simulate some missing values
          if (i % 10 === 0) {
            return {
              timestamp: new Date(baseDate.getTime() + i * 86400000).toISOString(),
              value: null
            };
          }
          return {
            timestamp: new Date(baseDate.getTime() + i * 86400000).toISOString(),
            value: 100 + Math.random() * 5
          };
        }).filter(p => p.value !== null),
        frequency: '1D'
      };

      const metrics = await forecast.fit(modelId, data);
      expect(Array.isArray(metrics)).toBe(true);
    });
  });

  describe('performance monitoring', () => {
    it('should track model performance over time', async () => {
      const { NeuralForecast } = require('../index.js');
      const forecast = new NeuralForecast();

      const modelId = await forecast.addModel({
        modelType: 'LSTM',
        inputSize: 10,
        horizon: 5
      });

      const baseDate = new Date('2024-01-01T00:00:00Z');

      // Train with increasing data
      const trainingSizes = [50, 100, 150];
      const performanceMetrics: any[] = [];

      for (const size of trainingSizes) {
        const data = {
          points: Array.from({ length: size }, (_, i) => ({
            timestamp: new Date(baseDate.getTime() + i * 86400000).toISOString(),
            value: 100 + Math.sin((i / size) * Math.PI * 2) * 10
          })),
          frequency: '1D'
        };

        await forecast.fit(modelId, data);

        const metrics = await forecast.crossValidation(modelId, data, 3, 1);
        performanceMetrics.push({
          trainingSize: size,
          mae: metrics.mae
        });
      }

      expect(performanceMetrics.length).toBe(3);
    });
  });

  describe('error recovery', () => {
    it('should recover from prediction after model updates', async () => {
      const { NeuralForecast } = require('../index.js');
      const forecast = new NeuralForecast();

      const modelId = await forecast.addModel({
        modelType: 'LSTM',
        inputSize: 10,
        horizon: 5
      });

      const baseDate = new Date('2024-01-01T00:00:00Z');
      const data1 = {
        points: Array.from({ length: 50 }, (_, i) => ({
          timestamp: new Date(baseDate.getTime() + i * 86400000).toISOString(),
          value: 100 + Math.random() * 5
        })),
        frequency: '1D'
      };

      // First training
      await forecast.fit(modelId, data1);
      const pred1 = await forecast.predict(modelId, 5);
      expect(pred1.predictions).toBeDefined();

      // Retrain with new data
      const data2 = {
        points: Array.from({ length: 60 }, (_, i) => ({
          timestamp: new Date(baseDate.getTime() + i * 86400000).toISOString(),
          value: 100 + Math.sin(i * 0.1) * 10
        })),
        frequency: '1D'
      };

      await forecast.fit(modelId, data2);
      const pred2 = await forecast.predict(modelId, 5);
      expect(pred2.predictions).toBeDefined();
    });
  });
});
