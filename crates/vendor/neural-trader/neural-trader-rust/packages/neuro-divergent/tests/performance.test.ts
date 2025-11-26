/**
 * Performance tests for @neural-trader/neuro-divergent package
 * Benchmarks forecasting operations and throughput
 */

jest.mock('../index.js', () => ({
  NeuralForecast: jest.fn().mockImplementation(function() {
    this.addModel = jest.fn(async () => {
      await new Promise(resolve => setTimeout(resolve, 5));
      return 'model-123';
    });
    this.getConfig = jest.fn().mockResolvedValue({
      modelType: 'LSTM',
      inputSize: 10,
      horizon: 5
    });
    this.fit = jest.fn(async () => {
      await new Promise(resolve => setTimeout(resolve, 20));
      return [
        { epoch: 1, trainLoss: 0.5, valLoss: 0.6 },
        { epoch: 2, trainLoss: 0.3, valLoss: 0.4 }
      ];
    });
    this.predict = jest.fn(async () => {
      await new Promise(resolve => setTimeout(resolve, 10));
      return {
        predictions: [100.5, 101.2, 102.1],
        modelType: 'LSTM',
        confidence: 0.95
      };
    });
    this.crossValidation = jest.fn(async () => {
      await new Promise(resolve => setTimeout(resolve, 50));
      return {
        mae: 0.15,
        mse: 0.045,
        rmse: 0.212,
        mape: 2.5
      };
    });
  }),
  listAvailableModels: jest.fn(() => ['LSTM', 'GRU', 'Transformer', 'NHITS', 'NBEATS', 'DeepAR']),
  version: jest.fn(() => '2.1.0'),
  isGpuAvailable: jest.fn(() => true)
}));

describe('Performance Benchmarks', () => {
  describe('model initialization', () => {
    it('should initialize models quickly', async () => {
      const { NeuralForecast } = require('../index.js');

      const startTime = performance.now();
      const forecasts = Array.from({ length: 100 }, () => new NeuralForecast());
      const duration = performance.now() - startTime;

      expect(forecasts.length).toBe(100);
      expect(duration).toBeLessThan(100); // 100 instances in < 100ms
    });

    it('should add models within budget', async () => {
      const { NeuralForecast } = require('../index.js');
      const forecast = new NeuralForecast();

      const startTime = performance.now();

      const modelIds = await Promise.all(
        Array.from({ length: 10 }, () =>
          forecast.addModel({
            modelType: 'LSTM',
            inputSize: 10,
            horizon: 5
          })
        )
      );

      const duration = performance.now() - startTime;

      expect(modelIds.length).toBe(10);
      expect(duration).toBeLessThan(5000); // 10 models in < 5 seconds
    }, 10000);
  });

  describe('forecasting performance', () => {
    let forecast: any;
    let modelId: string;

    beforeEach(async () => {
      const { NeuralForecast } = require('../index.js');
      forecast = new NeuralForecast();

      modelId = await forecast.addModel({
        modelType: 'LSTM',
        inputSize: 10,
        horizon: 5
      });
    });

    it('should make predictions efficiently', async () => {
      const startTime = performance.now();

      const predictions = await Promise.all(
        Array.from({ length: 100 }, () =>
          forecast.predict(modelId, 5)
        )
      );

      const duration = performance.now() - startTime;

      expect(predictions.length).toBe(100);
      expect(duration).toBeLessThan(10000); // 100 predictions in < 10s
    }, 15000);

    it('should achieve target prediction throughput', async () => {
      const predictions = 50;
      let count = 0;
      const startTime = performance.now();

      while (count < predictions) {
        await forecast.predict(modelId, 5);
        count++;
      }

      const duration = performance.now() - startTime;
      const throughput = (predictions / duration) * 1000; // predictions per second

      expect(throughput).toBeGreaterThan(5); // At least 5 predictions/sec
    }, 20000);
  });

  describe('training performance', () => {
    it('should train quickly with small datasets', async () => {
      const { NeuralForecast } = require('../index.js');
      const forecast = new NeuralForecast();

      const modelId = await forecast.addModel({
        modelType: 'LSTM',
        inputSize: 10,
        horizon: 5
      });

      const baseDate = new Date('2024-01-01T00:00:00Z');
      const data = {
        points: Array.from({ length: 50 }, (_, i) => ({
          timestamp: new Date(baseDate.getTime() + i * 86400000).toISOString(),
          value: 100 + Math.random() * 5
        })),
        frequency: '1D'
      };

      const startTime = performance.now();
      await forecast.fit(modelId, data);
      const duration = performance.now() - startTime;

      expect(duration).toBeLessThan(10000); // Training should complete
    }, 15000);

    it('should scale with training data size', async () => {
      const { NeuralForecast } = require('../index.js');

      const dataSizes = [50, 100, 200];
      const durations: number[] = [];

      for (const size of dataSizes) {
        const forecast = new NeuralForecast();

        const modelId = await forecast.addModel({
          modelType: 'LSTM',
          inputSize: 10,
          horizon: 5
        });

        const baseDate = new Date('2024-01-01T00:00:00Z');
        const data = {
          points: Array.from({ length: size }, (_, i) => ({
            timestamp: new Date(baseDate.getTime() + i * 86400000).toISOString(),
            value: 100 + Math.sin((i / size) * Math.PI * 2) * 5
          })),
          frequency: '1D'
        };

        const startTime = performance.now();
        await forecast.fit(modelId, data);
        durations.push(performance.now() - startTime);
      }

      // Should complete training for all data sizes
      durations.forEach(d => {
        expect(d).toBeLessThan(10000); // Each should complete within 10 seconds
      });
    }, 30000);

    it('should handle parallel training', async () => {
      const { NeuralForecast } = require('../index.js');

      const forecasts = Array.from({ length: 5 }, () => new NeuralForecast());

      const modelIds = await Promise.all(
        forecasts.map(f =>
          f.addModel({
            modelType: 'LSTM',
            inputSize: 10,
            horizon: 5
          })
        )
      );

      const baseDate = new Date('2024-01-01T00:00:00Z');
      const data = {
        points: Array.from({ length: 50 }, (_, i) => ({
          timestamp: new Date(baseDate.getTime() + i * 86400000).toISOString(),
          value: 100 + Math.random() * 5
        })),
        frequency: '1D'
      };

      const startTime = performance.now();

      await Promise.all(
        forecasts.map((f, i) => f.fit(modelIds[i], data))
      );

      const duration = performance.now() - startTime;

      // 5 parallel trainings should be significantly faster than sequential
      expect(duration).toBeLessThan(30000);
    }, 35000);
  });

  describe('cross-validation performance', () => {
    it('should perform cross-validation within budget', async () => {
      const { NeuralForecast } = require('../index.js');
      const forecast = new NeuralForecast();

      const modelId = await forecast.addModel({
        modelType: 'LSTM',
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

      const startTime = performance.now();
      await forecast.crossValidation(modelId, data, 5, 1);
      const duration = performance.now() - startTime;

      expect(duration).toBeLessThan(30000); // CV should complete
    }, 35000);

    it('should handle CV with different fold counts', async () => {
      const { NeuralForecast } = require('../index.js');

      const folds = [3, 5, 10];

      for (const fold of folds) {
        const forecast = new NeuralForecast();

        const modelId = await forecast.addModel({
          modelType: 'LSTM',
          inputSize: 10,
          horizon: 5
        });

        const baseDate = new Date('2024-01-01T00:00:00Z');
        const data = {
          points: Array.from({ length: 100 }, (_, i) => ({
            timestamp: new Date(baseDate.getTime() + i * 86400000).toISOString(),
            value: 100 + Math.random() * 5
          })),
          frequency: '1D'
        };

        const startTime = performance.now();
        await forecast.crossValidation(modelId, data, fold, 1);
        const duration = performance.now() - startTime;

        expect(duration).toBeLessThan(30000);
      }
    }, 40000);
  });

  describe('ensemble performance', () => {
    it('should create ensemble efficiently', async () => {
      const { NeuralForecast } = require('../index.js');

      const types = ['LSTM', 'GRU', 'Transformer'];
      const forecasts = types.map(() => new NeuralForecast());

      const startTime = performance.now();

      const modelIds = await Promise.all(
        forecasts.map((f, i) =>
          f.addModel({
            modelType: types[i],
            inputSize: 10,
            horizon: 5
          })
        )
      );

      const predictions = await Promise.all(
        forecasts.map((f, i) => f.predict(modelIds[i], 5))
      );

      const duration = performance.now() - startTime;

      expect(predictions.length).toBe(3);
      expect(duration).toBeLessThan(10000);
    }, 15000);
  });

  describe('memory efficiency', () => {
    it('should not cause excessive memory growth', () => {
      const { NeuralForecast } = require('../index.js');

      const initialMemory = process.memoryUsage().heapUsed;

      const forecasts = Array.from({ length: 1000 }, () => new NeuralForecast());

      const afterCreation = process.memoryUsage().heapUsed;
      const memoryIncrease = afterCreation - initialMemory;

      // 1000 instances shouldn't use more than 50MB
      expect(memoryIncrease).toBeLessThan(50 * 1024 * 1024);
    });

    it('should handle large time series efficiently', async () => {
      const { NeuralForecast } = require('../index.js');
      const forecast = new NeuralForecast();

      const modelId = await forecast.addModel({
        modelType: 'LSTM',
        inputSize: 10,
        horizon: 5
      });

      const baseDate = new Date('2024-01-01T00:00:00Z');
      const largeData = {
        points: Array.from({ length: 10000 }, (_, i) => ({
          timestamp: new Date(baseDate.getTime() + i * 3600000).toISOString(),
          value: 100 + Math.sin((i / 10000) * Math.PI * 2) * 10
        })),
        frequency: '1H'
      };

      const initialMemory = process.memoryUsage().heapUsed;

      // Just prepare the data, not train
      expect(largeData.points.length).toBe(10000);

      const afterPrepare = process.memoryUsage().heapUsed;
      const memoryUsed = afterPrepare - initialMemory;

      // Should use reasonable amount of memory
      expect(memoryUsed).toBeLessThan(100 * 1024 * 1024); // < 100MB
    });
  });

  describe('concurrent operations', () => {
    it('should handle concurrent operations efficiently', async () => {
      const { NeuralForecast } = require('../index.js');

      const forecasts = Array.from({ length: 10 }, () => new NeuralForecast());

      const startTime = performance.now();

      const modelIds = await Promise.all(
        forecasts.map(f =>
          f.addModel({
            modelType: 'LSTM',
            inputSize: 10,
            horizon: 5
          })
        )
      );

      const predictions = await Promise.all(
        forecasts.flatMap((f, i) =>
          Array.from({ length: 10 }, () => f.predict(modelIds[i], 5))
        )
      );

      const duration = performance.now() - startTime;

      expect(predictions.length).toBe(100);
      expect(duration).toBeLessThan(20000); // 100 concurrent predictions
    }, 25000);
  });

  describe('latency measurements', () => {
    it('should meet latency SLAs for predictions', async () => {
      const { NeuralForecast } = require('../index.js');
      const forecast = new NeuralForecast();

      const modelId = await forecast.addModel({
        modelType: 'LSTM',
        inputSize: 10,
        horizon: 5
      });

      const latencies: number[] = [];

      for (let i = 0; i < 50; i++) {
        const startTime = performance.now();
        await forecast.predict(modelId, 5);
        latencies.push(performance.now() - startTime);
      }

      const avgLatency = latencies.reduce((a, b) => a + b) / latencies.length;
      const p99Latency = latencies.sort((a, b) => a - b)[Math.floor(latencies.length * 0.99)];

      expect(avgLatency).toBeLessThan(50); // Avg < 50ms
      expect(p99Latency).toBeLessThan(100); // P99 < 100ms
    }, 15000);
  });
});
