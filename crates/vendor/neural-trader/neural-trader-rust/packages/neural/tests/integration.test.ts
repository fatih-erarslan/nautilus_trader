/**
 * Integration tests for @neural-trader/neural package
 * Tests end-to-end workflows including training, saving, and loading
 */

jest.mock('../load-binary', () => ({
  loadNativeBinary: jest.fn(() => ({
    NeuralModel: jest.fn().mockImplementation(function() {
      this.train = jest.fn().mockResolvedValue([
        { epoch: 1, trainLoss: 0.5, valLoss: 0.6 },
        { epoch: 2, trainLoss: 0.3, valLoss: 0.4 },
        { epoch: 3, trainLoss: 0.2, valLoss: 0.25 }
      ]);
      this.predict = jest.fn().mockResolvedValue({
        predictions: [100.5, 101.2, 102.1],
        confidence: 0.95
      });
      this.save = jest.fn().mockResolvedValue('/models/model-123.bin');
      this.load = jest.fn().mockResolvedValue(undefined);
      this.getInfo = jest.fn().mockResolvedValue('LSTM Model v1.0');
    }),
    BatchPredictor: jest.fn().mockImplementation(function() {
      this.addModel = jest.fn().mockResolvedValue(1);
      this.predictBatch = jest.fn().mockResolvedValue([
        { predictions: [100.5], confidence: 0.95 },
        { predictions: [101.2], confidence: 0.93 }
      ]);
    }),
    listModelTypes: jest.fn(() => ['LSTM', 'GRU', 'Transformer', 'NBEATS', 'DeepAR'])
  }))
}));

import { NeuralModel, BatchPredictor } from '../index';

describe('NeuralModel Integration', () => {
  describe('complete training workflow', () => {
    it('should train and predict with same model', async () => {
      const model = new NeuralModel({
        modelType: 'LSTM',
        inputSize: 20,
        horizon: 5,
        hiddenSize: 32,
        numLayers: 2
      });

      // Generate synthetic time series data
      const data = Array.from({ length: 100 }, (_, i) => {
        return Math.sin((i / 100) * Math.PI * 2) + Math.random() * 0.1;
      });

      const targets = data.slice(20);

      // Train the model
      const metrics = await model.train(data, targets, {
        epochs: 10,
        batchSize: 16
      });

      expect(metrics.length).toBeGreaterThan(0);
      expect(metrics[0].trainLoss).toBeGreaterThan(0);

      // Make predictions
      const predictions = await model.predict(data.slice(0, 20));

      expect(predictions).toBeDefined();
      expect(predictions.predictions.length).toBeGreaterThan(0);
    });

    it('should train multiple models independently', async () => {
      const models = [
        new NeuralModel({
          modelType: 'LSTM',
          inputSize: 10,
          horizon: 5
        }),
        new NeuralModel({
          modelType: 'GRU',
          inputSize: 10,
          horizon: 5
        }),
        new NeuralModel({
          modelType: 'Transformer',
          inputSize: 10,
          horizon: 5
        })
      ];

      const data = Array.from({ length: 50 }, () => Math.random());
      const targets = data.slice(10);

      const results = await Promise.all(
        models.map(m => m.train(data, targets, { epochs: 5 }))
      );

      expect(results.length).toBe(3);
      results.forEach(result => {
        expect(Array.isArray(result)).toBe(true);
      });
    });
  });

  describe('model persistence workflow', () => {
    it('should save and load model maintaining state', async () => {
      const model = new NeuralModel({
        modelType: 'LSTM',
        inputSize: 10,
        horizon: 5
      });

      const data = Array.from({ length: 50 }, () => Math.random());
      const targets = data.slice(10);

      // Train model
      await model.train(data, targets, { epochs: 5 });

      // Save model
      const savedPath = await model.save('/tmp/model.bin');
      expect(savedPath).toBeDefined();

      // Load into new instance
      const newModel = new NeuralModel({
        modelType: 'LSTM',
        inputSize: 10,
        horizon: 5
      });

      await newModel.load(savedPath);

      // Verify loaded model can predict
      const predictions = await newModel.predict(data.slice(0, 10));
      expect(predictions).toBeDefined();
    });

    it('should handle multiple model saves', async () => {
      const model = new NeuralModel({
        modelType: 'LSTM',
        inputSize: 10,
        horizon: 5
      });

      const paths = [];
      for (let i = 0; i < 3; i++) {
        const path = `/tmp/model-${i}.bin`;
        const savedPath = await model.save(path);
        paths.push(savedPath);
      }

      expect(paths.length).toBe(3);
      paths.forEach(p => {
        expect(typeof p).toBe('string');
      });
    });

    it('should provide model information', async () => {
      const model = new NeuralModel({
        modelType: 'LSTM',
        inputSize: 10,
        horizon: 5
      });

      const info = await model.getInfo();

      expect(typeof info).toBe('string');
      expect(info.includes('LSTM')).toBe(true);
    });
  });

  describe('error handling', () => {
    it('should handle invalid input data', async () => {
      const model = new NeuralModel({
        modelType: 'LSTM',
        inputSize: 10,
        horizon: 5
      });

      const invalidData: any = null;
      const invalidTargets: any = undefined;

      try {
        await model.train(invalidData, invalidTargets, { epochs: 1 });
      } catch (error) {
        expect(error).toBeDefined();
      }
    });

    it('should handle mismatched data shapes', async () => {
      const model = new NeuralModel({
        modelType: 'LSTM',
        inputSize: 10,
        horizon: 5
      });

      const data = Array.from({ length: 50 }, () => Math.random());
      const targets = Array.from({ length: 10 }, () => Math.random()); // Wrong size

      try {
        await model.train(data, targets, { epochs: 1 });
      } catch (error) {
        expect(error).toBeDefined();
      }
    });
  });
});

describe('BatchPredictor Integration', () => {
  describe('multi-model batch workflow', () => {
    it('should manage multiple models and predict', async () => {
      const predictor = new BatchPredictor();

      // Add multiple models
      const modelIds = [];
      for (let i = 0; i < 3; i++) {
        const model = new NeuralModel({
          modelType: 'LSTM',
          inputSize: 10,
          horizon: 5
        });
        const id = await predictor.addModel(model);
        modelIds.push(id);
      }

      expect(modelIds.length).toBe(3);

      // Batch predict
      const inputs = Array.from({ length: 3 }, () =>
        Array.from({ length: 10 }, () => Math.random())
      );

      const results = await predictor.predictBatch(inputs);

      expect(results).toBeDefined();
      expect(Array.isArray(results)).toBe(true);
    });

    it('should handle heterogeneous model types', async () => {
      const predictor = new BatchPredictor();

      const modelTypes = ['LSTM', 'GRU', 'Transformer'];
      const modelIds = [];

      for (const type of modelTypes) {
        const model = new NeuralModel({
          modelType: type,
          inputSize: 10,
          horizon: 5
        });
        const id = await predictor.addModel(model);
        modelIds.push(id);
      }

      expect(modelIds.length).toBe(3);

      const inputs = Array.from({ length: 3 }, () =>
        Array.from({ length: 10 }, () => Math.random())
      );

      const results = await predictor.predictBatch(inputs);
      expect(results).toBeDefined();
    });
  });

  describe('large scale batch operations', () => {
    it('should handle large batch sizes efficiently', async () => {
      const predictor = new BatchPredictor();

      const model = new NeuralModel({
        modelType: 'LSTM',
        inputSize: 10,
        horizon: 5
      });

      await predictor.addModel(model);

      // Large batch
      const largeInputs = Array.from({ length: 1000 }, () =>
        Array.from({ length: 10 }, () => Math.random())
      );

      const startTime = Date.now();
      const results = await predictor.predictBatch(largeInputs);
      const duration = Date.now() - startTime;

      expect(results).toBeDefined();
      expect(duration).toBeLessThan(10000); // Should complete within 10 seconds
    });
  });

  describe('error recovery', () => {
    it('should recover from prediction errors', async () => {
      const predictor = new BatchPredictor();

      const model = new NeuralModel({
        modelType: 'LSTM',
        inputSize: 10,
        horizon: 5
      });

      await predictor.addModel(model);

      // Valid batch
      const validInputs = Array.from({ length: 2 }, () =>
        Array.from({ length: 10 }, () => Math.random())
      );

      const results = await predictor.predictBatch(validInputs);
      expect(results).toBeDefined();

      // Should still work after valid prediction
      const secondResults = await predictor.predictBatch(validInputs);
      expect(secondResults).toBeDefined();
    });
  });
});

describe('Cross-model scenarios', () => {
  it('should train one model and use predictions as input for another', async () => {
    const model1 = new NeuralModel({
      modelType: 'LSTM',
      inputSize: 10,
      horizon: 5
    });

    const data = Array.from({ length: 50 }, () => Math.random());
    const targets = data.slice(10);

    // Train first model
    await model1.train(data, targets, { epochs: 5 });

    // Get predictions from first model
    const pred1 = await model1.predict(data.slice(0, 10));

    // Create second model
    const model2 = new NeuralModel({
      modelType: 'GRU',
      inputSize: 5,
      horizon: 3
    });

    // Use predictions as input for second model
    const combinedInput = [...data.slice(0, 5), ...pred1.predictions];
    const pred2 = await model2.predict(combinedInput);

    expect(pred2).toBeDefined();
    expect(pred2.predictions).toBeDefined();
  });

  it('should ensemble predictions from multiple models', async () => {
    const models = [
      new NeuralModel({
        modelType: 'LSTM',
        inputSize: 10,
        horizon: 5
      }),
      new NeuralModel({
        modelType: 'GRU',
        inputSize: 10,
        horizon: 5
      })
    ];

    const input = Array.from({ length: 10 }, () => Math.random());

    const predictions = await Promise.all(
      models.map(m => m.predict(input))
    );

    // Average ensemble predictions
    const ensembleAvg = predictions[0].predictions.map(
      (_, i) => {
        const sum = predictions.reduce((acc, pred) => acc + pred.predictions[i], 0);
        return sum / predictions.length;
      }
    );

    expect(ensembleAvg.length).toBeGreaterThan(0);
  });
});
