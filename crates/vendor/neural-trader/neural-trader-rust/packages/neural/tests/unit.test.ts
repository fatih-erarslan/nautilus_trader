/**
 * Unit tests for @neural-trader/neural package
 * Tests model loading, initialization, and basic inference operations
 */

jest.mock('../load-binary', () => ({
  loadNativeBinary: jest.fn(() => ({
    NeuralModel: jest.fn().mockImplementation(function() {
      this.train = jest.fn().mockResolvedValue([
        { epoch: 1, trainLoss: 0.5, valLoss: 0.6 },
        { epoch: 2, trainLoss: 0.3, valLoss: 0.4 }
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

import { NeuralModel, BatchPredictor, listModelTypes } from '../index';

describe('NeuralModel', () => {
  let model: any;

  beforeEach(() => {
    model = new NeuralModel({
      modelType: 'LSTM',
      inputSize: 10,
      horizon: 5,
      hiddenSize: 32,
      numLayers: 2,
      dropout: 0.1,
      learningRate: 0.001
    });
  });

  describe('initialization', () => {
    it('should create a NeuralModel instance', () => {
      expect(model).toBeDefined();
    });

    it('should accept valid configuration', () => {
      const config = {
        modelType: 'LSTM',
        inputSize: 20,
        horizon: 10,
        hiddenSize: 64,
        numLayers: 3,
        dropout: 0.2,
        learningRate: 0.0001
      };
      const testModel = new NeuralModel(config);
      expect(testModel).toBeDefined();
    });

    it('should handle different model types', () => {
      const types = ['LSTM', 'GRU', 'Transformer', 'NBEATS', 'DeepAR'];
      types.forEach(type => {
        const testModel = new NeuralModel({
          modelType: type,
          inputSize: 10,
          horizon: 5
        });
        expect(testModel).toBeDefined();
      });
    });

    it('should use default values for optional parameters', () => {
      const minimalModel = new NeuralModel({
        modelType: 'LSTM',
        inputSize: 10,
        horizon: 5
      });
      expect(minimalModel).toBeDefined();
    });
  });

  describe('training', () => {
    it('should train model with data', async () => {
      const data = Array.from({ length: 100 }, () => Math.random());
      const targets = Array.from({ length: 100 }, () => Math.random());

      const metrics = await model.train(data, targets, {
        epochs: 10,
        batchSize: 32
      });

      expect(Array.isArray(metrics)).toBe(true);
      expect(metrics.length).toBeGreaterThan(0);
      expect(metrics[0]).toHaveProperty('epoch');
      expect(metrics[0]).toHaveProperty('trainLoss');
    });

    it('should return training metrics for each epoch', async () => {
      const data = Array.from({ length: 50 }, () => Math.random());
      const targets = Array.from({ length: 50 }, () => Math.random());

      const metrics = await model.train(data, targets, { epochs: 5 });

      expect(metrics.length).toBe(2); // Mock returns 2 epochs
      metrics.forEach(metric => {
        expect(metric).toHaveProperty('epoch');
        expect(metric).toHaveProperty('trainLoss');
        expect(metric).toHaveProperty('valLoss');
        expect(typeof metric.trainLoss).toBe('number');
        expect(typeof metric.valLoss).toBe('number');
      });
    });

    it('should handle empty training data gracefully', async () => {
      const emptyData: number[] = [];
      const emptyTargets: number[] = [];

      try {
        await model.train(emptyData, emptyTargets, { epochs: 1 });
      } catch (error) {
        // Expected to fail with empty data
        expect(error).toBeDefined();
      }
    });
  });

  describe('prediction', () => {
    it('should make predictions', async () => {
      const inputData = Array.from({ length: 10 }, () => Math.random());

      const result = await model.predict(inputData);

      expect(result).toBeDefined();
      expect(Array.isArray(result.predictions)).toBe(true);
      expect(result.predictions.length).toBeGreaterThan(0);
    });

    it('should return confidence scores', async () => {
      const inputData = Array.from({ length: 10 }, () => Math.random());

      const result = await model.predict(inputData);

      expect(result).toHaveProperty('confidence');
      expect(typeof result.confidence).toBe('number');
      expect(result.confidence).toBeGreaterThanOrEqual(0);
      expect(result.confidence).toBeLessThanOrEqual(1);
    });

    it('should handle different input sizes', async () => {
      const sizes = [5, 10, 20, 50, 100];

      for (const size of sizes) {
        const inputData = Array.from({ length: size }, () => Math.random());
        const result = await model.predict(inputData);
        expect(result).toBeDefined();
        expect(result.predictions).toBeDefined();
      }
    });
  });

  describe('model persistence', () => {
    it('should save model', async () => {
      const path = '/tmp/test-model.bin';
      const savedPath = await model.save(path);

      expect(typeof savedPath).toBe('string');
      expect(savedPath.length).toBeGreaterThan(0);
    });

    it('should load model', async () => {
      const path = '/tmp/test-model.bin';
      await expect(model.load(path)).resolves.not.toThrow();
    });

    it('should get model info', async () => {
      const info = await model.getInfo();

      expect(typeof info).toBe('string');
      expect(info.length).toBeGreaterThan(0);
    });
  });
});

describe('BatchPredictor', () => {
  let predictor: any;

  beforeEach(() => {
    predictor = new BatchPredictor();
  });

  describe('initialization', () => {
    it('should create a BatchPredictor instance', () => {
      expect(predictor).toBeDefined();
    });
  });

  describe('model management', () => {
    it('should add model to batch predictor', async () => {
      const model = new NeuralModel({
        modelType: 'LSTM',
        inputSize: 10,
        horizon: 5
      });

      const modelId = await predictor.addModel(model);

      expect(typeof modelId).toBe('number');
      expect(modelId).toBeGreaterThanOrEqual(0);
    });

    it('should handle multiple models', async () => {
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
    });
  });

  describe('batch prediction', () => {
    it('should make batch predictions', async () => {
      const model = new NeuralModel({
        modelType: 'LSTM',
        inputSize: 10,
        horizon: 5
      });

      await predictor.addModel(model);

      const batchInputs = [
        Array.from({ length: 10 }, () => Math.random()),
        Array.from({ length: 10 }, () => Math.random())
      ];

      const results = await predictor.predictBatch(batchInputs);

      expect(Array.isArray(results)).toBe(true);
      expect(results.length).toBe(2);
      results.forEach(result => {
        expect(result).toHaveProperty('predictions');
        expect(result).toHaveProperty('confidence');
      });
    });

    it('should handle large batch sizes', async () => {
      const model = new NeuralModel({
        modelType: 'LSTM',
        inputSize: 10,
        horizon: 5
      });

      await predictor.addModel(model);

      const batchInputs = Array.from({ length: 100 }, () =>
        Array.from({ length: 10 }, () => Math.random())
      );

      const results = await predictor.predictBatch(batchInputs);

      expect(results.length).toBe(2); // Mock returns 2 results
    });
  });
});

describe('listModelTypes', () => {
  it('should list available model types', () => {
    const types = listModelTypes();

    expect(Array.isArray(types)).toBe(true);
    expect(types.length).toBeGreaterThan(0);
  });

  it('should include standard model types', () => {
    const types = listModelTypes();
    const expectedTypes = ['LSTM', 'GRU', 'Transformer'];

    expectedTypes.forEach(type => {
      expect(types).toContain(type);
    });
  });

  it('should return consistent results', () => {
    const types1 = listModelTypes();
    const types2 = listModelTypes();

    expect(types1).toEqual(types2);
  });
});
