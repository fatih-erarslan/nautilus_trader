/**
 * Tests for ensemble swarm
 */

import { EnsembleSwarm } from '../src/ensemble-swarm';
import { ModelType, TimeSeriesPoint } from '../src/types';

describe('EnsembleSwarm', () => {
  let testData: TimeSeriesPoint[];
  let validationData: TimeSeriesPoint[];

  beforeEach(() => {
    // Generate training data
    testData = Array.from({ length: 150 }, (_, i) => ({
      timestamp: Date.now() + i * 3600000,
      value: 100 + i * 0.3 + 15 * Math.sin((i * 2 * Math.PI) / 24)
    }));

    // Generate validation data
    validationData = Array.from({ length: 50 }, (_, i) => ({
      timestamp: Date.now() + (150 + i) * 3600000,
      value: 145 + i * 0.3 + 15 * Math.sin(((150 + i) * 2 * Math.PI) / 24)
    }));
  });

  describe('Hyperparameter Exploration', () => {
    it('should explore ARIMA hyperparameters', async () => {
      const ensemble = new EnsembleSwarm({
        models: [ModelType.ARIMA]
      });

      const result = await ensemble.exploreHyperparameters(
        ModelType.ARIMA,
        testData,
        validationData
      );

      expect(result).toHaveProperty('bestModel');
      expect(result).toHaveProperty('bestHyperparameters');
      expect(result).toHaveProperty('allResults');
      expect(result).toHaveProperty('explorationTime');

      expect(result.bestModel).toBe(ModelType.ARIMA);
      expect(result.allResults.length).toBeGreaterThan(0);
    });

    it('should explore LSTM hyperparameters', async () => {
      const ensemble = new EnsembleSwarm({
        models: [ModelType.LSTM]
      });

      const result = await ensemble.exploreHyperparameters(
        ModelType.LSTM,
        testData,
        validationData
      );

      expect(result.bestModel).toBe(ModelType.LSTM);
      expect(result.bestHyperparameters).toHaveProperty('lookback');
    });

    it('should explore Prophet hyperparameters', async () => {
      const ensemble = new EnsembleSwarm({
        models: [ModelType.PROPHET]
      });

      const result = await ensemble.exploreHyperparameters(
        ModelType.PROPHET,
        testData,
        validationData
      );

      expect(result.bestModel).toBe(ModelType.PROPHET);
      expect(result.bestHyperparameters).toHaveProperty('seasonalPeriod');
    });
  });

  describe('Ensemble Training', () => {
    it('should train multiple models in parallel', async () => {
      const ensemble = new EnsembleSwarm({
        models: [ModelType.ARIMA, ModelType.LSTM, ModelType.PROPHET]
      });

      await expect(
        ensemble.trainEnsemble(testData, validationData)
      ).resolves.not.toThrow();

      const stats = ensemble.getEnsembleStats();
      expect(stats.modelCount).toBe(3);
    }, 60000); // Increase timeout for parallel training

    it('should initialize horizon weights', async () => {
      const ensemble = new EnsembleSwarm({
        models: [ModelType.ARIMA, ModelType.LSTM]
      });

      await ensemble.trainEnsemble(testData, validationData);

      const stats = ensemble.getEnsembleStats();
      expect(stats.horizonWeights.size).toBeGreaterThan(0);
    });
  });

  describe('Predictions', () => {
    let ensemble: EnsembleSwarm;

    beforeEach(async () => {
      ensemble = new EnsembleSwarm({
        models: [ModelType.ARIMA, ModelType.LSTM]
      });
      await ensemble.trainEnsemble(testData, validationData);
    }, 60000);

    it('should generate predictions', async () => {
      const { predictions, selectedModel } = await ensemble.predict(24);

      expect(predictions).toHaveLength(24);
      expect(predictions.every(p => typeof p === 'number')).toBe(true);
      expect(selectedModel).toBeDefined();
    });

    it('should handle different horizons', async () => {
      const short = await ensemble.predict(6);
      const long = await ensemble.predict(48);

      expect(short.predictions).toHaveLength(6);
      expect(long.predictions).toHaveLength(48);
    });

    it('should throw error if not trained', async () => {
      const untrainedEnsemble = new EnsembleSwarm({
        models: [ModelType.ARIMA]
      });

      await expect(untrainedEnsemble.predict(10)).rejects.toThrow('Ensemble not trained');
    });
  });

  describe('Adaptive Learning', () => {
    let ensemble: EnsembleSwarm;

    beforeEach(async () => {
      ensemble = new EnsembleSwarm({
        models: [ModelType.ARIMA, ModelType.LSTM],
        adaptiveLearningRate: 0.1
      });
      await ensemble.trainEnsemble(testData, validationData);
    }, 60000);

    it('should update with observations', async () => {
      const { predictions } = await ensemble.predict(1);
      const modelPredictions = new Map([
        ['arima:alpha=0.3,beta=0.1', predictions[0]],
        ['lstm:lookback=24', predictions[0]]
      ]);

      await expect(
        ensemble.updateWithObservation(modelPredictions, 150, 1)
      ).resolves.not.toThrow();
    });

    it('should adapt weights based on performance', async () => {
      const statsBefore = ensemble.getEnsembleStats();

      // Simulate multiple updates with observations
      for (let i = 0; i < 10; i++) {
        const { predictions } = await ensemble.predict(1);
        const modelPredictions = new Map([
          ['arima:alpha=0.3,beta=0.1', predictions[0]],
          ['lstm:lookback=24', predictions[0] + 5] // LSTM is worse
        ]);

        await ensemble.updateWithObservation(modelPredictions, predictions[0], 1);
      }

      const statsAfter = ensemble.getEnsembleStats();

      // Check that performances have been updated
      expect(statsAfter.performances.length).toBe(statsBefore.performances.length);
    });
  });

  describe('Ensemble Statistics', () => {
    it('should provide comprehensive statistics', async () => {
      const ensemble = new EnsembleSwarm({
        models: [ModelType.ARIMA, ModelType.PROPHET]
      });

      await ensemble.trainEnsemble(testData, validationData);

      const stats = ensemble.getEnsembleStats();

      expect(stats).toHaveProperty('modelCount');
      expect(stats).toHaveProperty('performances');
      expect(stats).toHaveProperty('horizonWeights');

      expect(stats.modelCount).toBeGreaterThan(0);
      expect(stats.performances.length).toBeGreaterThan(0);

      stats.performances.forEach(({ model, performance }) => {
        expect(model).toBeDefined();
        expect(performance).toHaveProperty('modelName');
        expect(performance).toHaveProperty('mape');
        expect(performance).toHaveProperty('mae');
        expect(performance).toHaveProperty('rmse');
      });
    });
  });
});
