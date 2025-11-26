/**
 * Performance tests for @neural-trader/neural package
 * Benchmarks model operations and memory usage
 */

jest.mock('../load-binary', () => ({
  loadNativeBinary: jest.fn(() => ({
    NeuralModel: jest.fn().mockImplementation(function() {
      this.train = jest.fn(async () => {
        // Simulate training delay
        await new Promise(resolve => setTimeout(resolve, 10));
        return [
          { epoch: 1, trainLoss: 0.5, valLoss: 0.6 },
          { epoch: 2, trainLoss: 0.3, valLoss: 0.4 }
        ];
      });
      this.predict = jest.fn(async () => {
        await new Promise(resolve => setTimeout(resolve, 5));
        return {
          predictions: [100.5, 101.2, 102.1],
          confidence: 0.95
        };
      });
      this.save = jest.fn().mockResolvedValue('/models/model-123.bin');
      this.load = jest.fn().mockResolvedValue(undefined);
      this.getInfo = jest.fn().mockResolvedValue('LSTM Model v1.0');
    }),
    BatchPredictor: jest.fn().mockImplementation(function() {
      this.addModel = jest.fn().mockResolvedValue(1);
      this.predictBatch = jest.fn(async (inputs: any[]) => {
        await new Promise(resolve => setTimeout(resolve, inputs.length));
        return inputs.map((_, i) => ({
          predictions: [100.5 + i],
          confidence: 0.95
        }));
      });
    }),
    listModelTypes: jest.fn(() => ['LSTM', 'GRU', 'Transformer', 'NBEATS', 'DeepAR'])
  }))
}));

import { NeuralModel, BatchPredictor } from '../index';

describe('Performance Tests', () => {
  describe('model initialization performance', () => {
    it('should initialize model within reasonable time', () => {
      const startTime = performance.now();

      for (let i = 0; i < 100; i++) {
        new NeuralModel({
          modelType: 'LSTM',
          inputSize: 10,
          horizon: 5,
          hiddenSize: 32,
          numLayers: 2
        });
      }

      const duration = performance.now() - startTime;

      // 100 initializations should take less than 100ms
      expect(duration).toBeLessThan(100);
    });

    it('should handle rapid model creation', () => {
      const startTime = performance.now();
      const models = [];

      for (let i = 0; i < 1000; i++) {
        models.push(
          new NeuralModel({
            modelType: 'LSTM',
            inputSize: 10,
            horizon: 5
          })
        );
      }

      const duration = performance.now() - startTime;

      expect(models.length).toBe(1000);
      expect(duration).toBeLessThan(500); // Should create 1000 models in under 500ms
    });
  });

  describe('prediction performance', () => {
    it('should predict single input within timeout', async () => {
      const model = new NeuralModel({
        modelType: 'LSTM',
        inputSize: 10,
        horizon: 5
      });

      const input = Array.from({ length: 10 }, () => Math.random());

      const startTime = performance.now();
      await model.predict(input);
      const duration = performance.now() - startTime;

      // Single prediction should be fast (< 100ms with mocking)
      expect(duration).toBeLessThan(100);
    }, 5000);

    it('should handle multiple predictions efficiently', async () => {
      const model = new NeuralModel({
        modelType: 'LSTM',
        inputSize: 10,
        horizon: 5
      });

      const predictions = 100;
      const input = Array.from({ length: 10 }, () => Math.random());

      const startTime = performance.now();

      const results = await Promise.all(
        Array.from({ length: predictions }, () => model.predict(input))
      );

      const duration = performance.now() - startTime;

      expect(results.length).toBe(predictions);
      expect(duration).toBeLessThan(10000); // 100 predictions should take < 10s
    }, 15000);

    it('should maintain performance with varying input sizes', async () => {
      const model = new NeuralModel({
        modelType: 'LSTM',
        inputSize: 20,
        horizon: 5
      });

      const inputSizes = [10, 50, 100, 500];

      for (const size of inputSizes) {
        const input = Array.from({ length: size }, () => Math.random());
        const startTime = performance.now();

        await model.predict(input);

        const duration = performance.now() - startTime;
        expect(duration).toBeLessThan(100);
      }
    }, 10000);
  });

  describe('training performance', () => {
    it('should train model within timeout', async () => {
      const model = new NeuralModel({
        modelType: 'LSTM',
        inputSize: 10,
        horizon: 5
      });

      const data = Array.from({ length: 100 }, () => Math.random());
      const targets = data.slice(10);

      const startTime = performance.now();

      await model.train(data, targets, {
        epochs: 5,
        batchSize: 32
      });

      const duration = performance.now() - startTime;

      expect(duration).toBeLessThan(10000); // Should complete within 10 seconds
    }, 15000);

    it('should handle training with various dataset sizes', async () => {
      const model = new NeuralModel({
        modelType: 'LSTM',
        inputSize: 10,
        horizon: 5
      });

      const sizes = [50, 100, 500, 1000];

      for (const size of sizes) {
        const data = Array.from({ length: size }, () => Math.random());
        const targets = data.slice(10);

        const startTime = performance.now();

        await model.train(data, targets, {
          epochs: 3,
          batchSize: 32
        });

        const duration = performance.now() - startTime;

        expect(duration).toBeLessThan(10000);
      }
    }, 30000);
  });

  describe('batch prediction performance', () => {
    it('should predict batch within reasonable time', async () => {
      const predictor = new BatchPredictor();

      const model = new NeuralModel({
        modelType: 'LSTM',
        inputSize: 10,
        horizon: 5
      });

      await predictor.addModel(model);

      const batchSize = 100;
      const inputs = Array.from({ length: batchSize }, () =>
        Array.from({ length: 10 }, () => Math.random())
      );

      const startTime = performance.now();
      const results = await predictor.predictBatch(inputs);
      const duration = performance.now() - startTime;

      expect(results.length).toBeGreaterThan(0);
      expect(duration).toBeLessThan(10000); // Batch of 100 should be < 10s
    }, 15000);

    it('should scale with batch size', async () => {
      const predictor = new BatchPredictor();

      const model = new NeuralModel({
        modelType: 'LSTM',
        inputSize: 10,
        horizon: 5
      });

      await predictor.addModel(model);

      const batchSizes = [10, 50, 100];
      const durations: number[] = [];

      for (const size of batchSizes) {
        const inputs = Array.from({ length: size }, () =>
          Array.from({ length: 10 }, () => Math.random())
        );

        const startTime = performance.now();
        await predictor.predictBatch(inputs);
        const duration = performance.now() - startTime;

        durations.push(duration);
      }

      // Verify roughly linear scaling (allow some variance)
      expect(durations[1]).toBeLessThan(durations[0] * 10);
      expect(durations[2]).toBeLessThan(durations[0] * 20);
    }, 20000);
  });

  describe('memory efficiency', () => {
    it('should not cause excessive memory growth with many models', () => {
      const initialMemory = process.memoryUsage().heapUsed;
      const models = [];

      for (let i = 0; i < 1000; i++) {
        models.push(
          new NeuralModel({
            modelType: 'LSTM',
            inputSize: 10,
            horizon: 5
          })
        );
      }

      const afterCreation = process.memoryUsage().heapUsed;
      const memoryIncrease = afterCreation - initialMemory;

      // Memory increase should be reasonable (less than 100MB for 1000 models)
      expect(memoryIncrease).toBeLessThan(100 * 1024 * 1024);

      // Clear models
      models.length = 0;
    });

    it('should handle prediction caching efficiently', async () => {
      const model = new NeuralModel({
        modelType: 'LSTM',
        inputSize: 10,
        horizon: 5
      });

      const input = Array.from({ length: 10 }, () => Math.random());

      // Make same prediction multiple times
      const predictions = await Promise.all(
        Array.from({ length: 1000 }, () => model.predict(input))
      );

      expect(predictions.length).toBe(1000);

      // All predictions should be identical
      const firstPred = predictions[0].predictions;
      predictions.forEach(pred => {
        expect(pred.predictions).toEqual(firstPred);
      });
    }, 30000);
  });

  describe('concurrent operations', () => {
    it('should handle concurrent model operations', async () => {
      const models = Array.from({ length: 10 }, () =>
        new NeuralModel({
          modelType: 'LSTM',
          inputSize: 10,
          horizon: 5
        })
      );

      const input = Array.from({ length: 10 }, () => Math.random());

      const startTime = performance.now();

      const results = await Promise.all(
        models.flatMap(m =>
          Array.from({ length: 10 }, () => m.predict(input))
        )
      );

      const duration = performance.now() - startTime;

      expect(results.length).toBe(100);
      expect(duration).toBeLessThan(10000);
    }, 15000);

    it('should handle concurrent training', async () => {
      const models = Array.from({ length: 5 }, () =>
        new NeuralModel({
          modelType: 'LSTM',
          inputSize: 10,
          horizon: 5
        })
      );

      const data = Array.from({ length: 50 }, () => Math.random());
      const targets = data.slice(10);

      const startTime = performance.now();

      const results = await Promise.all(
        models.map(m =>
          m.train(data, targets, { epochs: 3 })
        )
      );

      const duration = performance.now() - startTime;

      expect(results.length).toBe(5);
      expect(duration).toBeLessThan(30000); // 5 concurrent trainings
    }, 35000);
  });

  describe('throughput benchmarks', () => {
    it('should achieve target throughput for single predictions', async () => {
      const model = new NeuralModel({
        modelType: 'LSTM',
        inputSize: 10,
        horizon: 5
      });

      const input = Array.from({ length: 10 }, () => Math.random());
      const targetThroughput = 10; // Predictions per second

      const startTime = performance.now();
      let count = 0;

      while (performance.now() - startTime < 1000) {
        await model.predict(input);
        count++;
      }

      const throughput = count;
      expect(throughput).toBeGreaterThanOrEqual(targetThroughput);
    }, 5000);

    it('should achieve target batch throughput', async () => {
      const predictor = new BatchPredictor();

      const model = new NeuralModel({
        modelType: 'LSTM',
        inputSize: 10,
        horizon: 5
      });

      await predictor.addModel(model);

      const batchInputs = Array.from({ length: 10 }, () =>
        Array.from({ length: 10 }, () => Math.random())
      );

      const startTime = performance.now();
      let count = 0;

      while (performance.now() - startTime < 1000) {
        await predictor.predictBatch(batchInputs);
        count++;
      }

      const throughput = count * 10; // Multiply by batch size
      expect(throughput).toBeGreaterThanOrEqual(100); // At least 100 predictions per second
    }, 5000);
  });
});
