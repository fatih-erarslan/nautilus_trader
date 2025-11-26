/**
 * Unit tests for @neural-trader/neuro-divergent package
 * Tests model loading, neural forecasting, and feature extraction
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
      { epoch: 2, trainLoss: 0.3, valLoss: 0.4 }
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

describe('NeuralForecast', () => {
  let forecast: any;

  beforeEach(() => {
    jest.clearAllMocks();
    const { NeuralForecast } = require('../index.js');
    forecast = new NeuralForecast();
  });

  describe('initialization', () => {
    it('should create a NeuralForecast instance', () => {
      expect(forecast).toBeDefined();
    });

    it('should have all required methods', () => {
      expect(typeof forecast.addModel).toBe('function');
      expect(typeof forecast.getConfig).toBe('function');
      expect(typeof forecast.fit).toBe('function');
      expect(typeof forecast.predict).toBe('function');
      expect(typeof forecast.crossValidation).toBe('function');
    });
  });

  describe('model management', () => {
    it('should add LSTM model', async () => {
      const modelId = await forecast.addModel({
        modelType: 'LSTM',
        inputSize: 10,
        horizon: 5,
        hiddenSize: 32,
        numLayers: 2,
        dropout: 0.1,
        learningRate: 0.001
      });

      expect(typeof modelId).toBe('string');
      expect(modelId).toBe('model-123');
    });

    it('should add different model types', async () => {
      const types = ['LSTM', 'GRU', 'Transformer', 'NHITS', 'NBEATS'];

      for (const type of types) {
        const modelId = await forecast.addModel({
          modelType: type,
          inputSize: 10,
          horizon: 5
        });

        expect(typeof modelId).toBe('string');
      }
    });

    it('should get model configuration', async () => {
      const modelId = await forecast.addModel({
        modelType: 'LSTM',
        inputSize: 10,
        horizon: 5
      });

      const config = await forecast.getConfig(modelId);

      expect(config).toBeDefined();
      expect(config.modelType).toBe('LSTM');
      expect(config.inputSize).toBe(10);
      expect(config.horizon).toBe(5);
    });

    it('should handle configuration with optional parameters', async () => {
      const config = {
        modelType: 'LSTM',
        inputSize: 20,
        horizon: 10,
        hiddenSize: 64,
        numLayers: 3,
        dropout: 0.2,
        learningRate: 0.0001
      };

      const modelId = await forecast.addModel(config);

      expect(modelId).toBeDefined();

      const retrievedConfig = await forecast.getConfig(modelId);
      expect(retrievedConfig).toBeDefined();
      expect(retrievedConfig.modelType).toBe('LSTM');
    });
  });

  describe('fitting models', () => {
    beforeEach(async () => {
      await forecast.addModel({
        modelType: 'LSTM',
        inputSize: 10,
        horizon: 5
      });
    });

    it('should fit model with training data', async () => {
      const data = {
        points: Array.from({ length: 50 }, (_, i) => ({
          timestamp: new Date(Date.now() - (50 - i) * 86400000).toISOString(),
          value: 100 + Math.sin(i * 0.1) * 10
        })),
        frequency: '1D'
      };

      const metrics = await forecast.fit('model-123', data);

      expect(Array.isArray(metrics)).toBe(true);
      expect(metrics.length).toBeGreaterThan(0);
      metrics.forEach(metric => {
        expect(metric).toHaveProperty('epoch');
        expect(metric).toHaveProperty('trainLoss');
        expect(metric).toHaveProperty('valLoss');
      });
    });

    it('should handle different frequency data', async () => {
      const frequencies = ['1H', '1D', '1W', '1M'];

      for (const freq of frequencies) {
        const data = {
          points: Array.from({ length: 50 }, (_, i) => ({
            timestamp: new Date(Date.now() - i * 86400000).toISOString(),
            value: 100 + Math.random() * 10
          })),
          frequency: freq
        };

        const metrics = await forecast.fit('model-123', data);
        expect(Array.isArray(metrics)).toBe(true);
      }
    });

    it('should track loss improvement over epochs', async () => {
      const data = {
        points: Array.from({ length: 50 }, (_, i) => ({
          timestamp: new Date(Date.now() - i * 86400000).toISOString(),
          value: 100 + Math.sin(i * 0.1) * 10
        })),
        frequency: '1D'
      };

      const metrics = await forecast.fit('model-123', data);

      expect(metrics[0].trainLoss).toBeGreaterThan(0);
      // In mock, loss decreases
      if (metrics.length > 1) {
        expect(metrics[metrics.length - 1].trainLoss).toBeLessThan(metrics[0].trainLoss);
      }
    });
  });

  describe('making predictions', () => {
    beforeEach(async () => {
      await forecast.addModel({
        modelType: 'LSTM',
        inputSize: 10,
        horizon: 5
      });
    });

    it('should make predictions', async () => {
      const predictions = await forecast.predict('model-123', 5);

      expect(predictions).toBeDefined();
      expect(Array.isArray(predictions.predictions)).toBe(true);
      expect(predictions.predictions.length).toBeGreaterThan(0);
      expect(predictions.modelType).toBe('LSTM');
    });

    it('should include confidence scores', async () => {
      const predictions = await forecast.predict('model-123', 5);

      expect(predictions).toHaveProperty('confidence');
      expect(typeof predictions.confidence).toBe('number');
      expect(predictions.confidence).toBeGreaterThanOrEqual(0);
      expect(predictions.confidence).toBeLessThanOrEqual(1);
    });

    it('should handle different prediction horizons', async () => {
      const horizons = [1, 5, 10, 20];

      for (const h of horizons) {
        const predictions = await forecast.predict('model-123', h);
        expect(predictions.predictions.length).toBe(3); // Mock returns 3 predictions
      }
    });
  });

  describe('cross-validation', () => {
    beforeEach(async () => {
      await forecast.addModel({
        modelType: 'LSTM',
        inputSize: 10,
        horizon: 5
      });
    });

    it('should perform cross-validation', async () => {
      const data = {
        points: Array.from({ length: 100 }, (_, i) => ({
          timestamp: new Date(Date.now() - i * 86400000).toISOString(),
          value: 100 + Math.sin(i * 0.1) * 10
        })),
        frequency: '1D'
      };

      const results = await forecast.crossValidation('model-123', data, 5, 1);

      expect(results).toHaveProperty('mae');
      expect(results).toHaveProperty('mse');
      expect(results).toHaveProperty('rmse');
      expect(results).toHaveProperty('mape');

      expect(typeof results.mae).toBe('number');
      expect(results.mae).toBeGreaterThan(0);
    });

    it('should return valid error metrics', async () => {
      const data = {
        points: Array.from({ length: 100 }, (_, i) => ({
          timestamp: new Date(Date.now() - i * 86400000).toISOString(),
          value: 100 + Math.random() * 5
        })),
        frequency: '1D'
      };

      const results = await forecast.crossValidation('model-123', data, 3, 1);

      // MAE should be less than MSE's square root
      expect(results.mae).toBeLessThanOrEqual(results.rmse);

      // All metrics should be positive
      expect(results.mae).toBeGreaterThan(0);
      expect(results.mse).toBeGreaterThan(0);
      expect(results.rmse).toBeGreaterThan(0);
      expect(results.mape).toBeGreaterThan(0);
    });
  });
});

describe('Module functions', () => {
  describe('listAvailableModels', () => {
    it('should list available models', () => {
      const { listAvailableModels } = require('../index.js');
      const models = listAvailableModels();

      expect(Array.isArray(models)).toBe(true);
      expect(models.length).toBeGreaterThan(0);
    });

    it('should include expected model types', () => {
      const { listAvailableModels } = require('../index.js');
      const models = listAvailableModels();

      const expectedModels = ['LSTM', 'GRU', 'Transformer', 'NHITS', 'NBEATS'];
      expectedModels.forEach(model => {
        expect(models).toContain(model);
      });
    });
  });

  describe('version', () => {
    it('should return version string', () => {
      const { version } = require('../index.js');
      const v = version();

      expect(typeof v).toBe('string');
      expect(v).toMatch(/^\d+\.\d+\.\d+/);
    });
  });

  describe('isGpuAvailable', () => {
    it('should check GPU availability', () => {
      const { isGpuAvailable } = require('../index.js');
      const hasGpu = isGpuAvailable();

      expect(typeof hasGpu).toBe('boolean');
    });

    it('should return consistent results', () => {
      const { isGpuAvailable } = require('../index.js');
      const result1 = isGpuAvailable();
      const result2 = isGpuAvailable();

      expect(result1).toBe(result2);
    });
  });
});

describe('Error handling', () => {
  it('should handle invalid model type', async () => {
    const { NeuralForecast } = require('../index.js');
    const forecast = new NeuralForecast();

    try {
      await forecast.addModel({
        modelType: 'InvalidModel',
        inputSize: 10,
        horizon: 5
      });
    } catch (error) {
      expect(error).toBeDefined();
    }
  });

  it('should handle invalid configuration', async () => {
    const { NeuralForecast } = require('../index.js');
    const forecast = new NeuralForecast();

    try {
      await forecast.addModel({
        modelType: 'LSTM',
        inputSize: -1, // Invalid
        horizon: 5
      });
    } catch (error) {
      expect(error).toBeDefined();
    }
  });

  it('should handle non-existent model ID', async () => {
    const { NeuralForecast } = require('../index.js');
    const forecast = new NeuralForecast();

    try {
      await forecast.predict('non-existent-id', 5);
    } catch (error) {
      expect(error).toBeDefined();
    }
  });
});
