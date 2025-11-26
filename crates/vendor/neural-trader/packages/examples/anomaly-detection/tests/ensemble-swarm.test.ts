import { describe, it, expect, beforeEach } from 'vitest';
import { EnsembleSwarm } from '../src/ensemble-swarm';
import type { AnomalyPoint } from '../src/index';

describe('EnsembleSwarm', () => {
  let swarm: EnsembleSwarm;

  beforeEach(() => {
    swarm = new EnsembleSwarm({
      featureDimensions: 2,
      populationSize: 20,
      maxGenerations: 50,
      crossoverRate: 0.8,
      mutationRate: 0.1,
    });
  });

  describe('Initialization', () => {
    it('should initialize with correct algorithm count', () => {
      const weights = swarm.getAlgorithmWeights();
      expect(weights.length).toBe(4); // isolation-forest, lstm-ae, vae, one-class-svm
    });

    it('should have normalized initial weights', () => {
      const weights = swarm.getAlgorithmWeights();
      const sum = weights.reduce((acc, w) => acc + w.weight, 0);
      expect(sum).toBeCloseTo(1.0, 5);
    });
  });

  describe('Training', () => {
    it('should train ensemble on labeled data', async () => {
      const trainingData = generateMixedData(200, 2);

      await swarm.train(trainingData);

      expect(swarm.getGeneration()).toBeGreaterThan(0);
      expect(swarm.getBestFitness()).toBeGreaterThan(0);
    });

    it('should improve fitness over generations', async () => {
      const trainingData = generateMixedData(150, 2);

      await swarm.train(trainingData);

      const finalFitness = swarm.getBestFitness();
      expect(finalFitness).toBeGreaterThan(0.3); // Should achieve reasonable performance
    });

    it('should optimize algorithm weights', async () => {
      const trainingData = generateMixedData(150, 2);

      await swarm.train(trainingData);

      const weights = swarm.getAlgorithmWeights();

      // All weights should be positive
      weights.forEach(w => {
        expect(w.weight).toBeGreaterThan(0);
      });

      // Weights should sum to 1
      const sum = weights.reduce((acc, w) => acc + w.weight, 0);
      expect(sum).toBeCloseTo(1.0, 5);
    });
  });

  describe('Prediction', () => {
    beforeEach(async () => {
      const trainingData = generateMixedData(150, 2);
      await swarm.train(trainingData);
    });

    it('should predict anomaly scores', () => {
      const normalPoint: AnomalyPoint = {
        timestamp: Date.now(),
        features: [0.2, 0.3],
      };

      const score = swarm.predictEnsemble(normalPoint);

      expect(score).toBeGreaterThanOrEqual(0);
      expect(score).toBeLessThanOrEqual(1);
    });

    it('should give higher scores to anomalies', () => {
      const normalPoint: AnomalyPoint = {
        timestamp: Date.now(),
        features: [0.2, 0.3],
      };

      const anomalousPoint: AnomalyPoint = {
        timestamp: Date.now(),
        features: [10, 10],
      };

      const normalScore = swarm.predictEnsemble(normalPoint);
      const anomalousScore = swarm.predictEnsemble(anomalousPoint);

      expect(anomalousScore).toBeGreaterThan(normalScore);
    });
  });

  describe('Algorithm Scores', () => {
    beforeEach(async () => {
      const trainingData = generateMixedData(100, 2);
      await swarm.train(trainingData);
    });

    it('should provide individual algorithm scores', () => {
      const point: AnomalyPoint = {
        timestamp: Date.now(),
        features: [0.5, 0.5],
      };

      const scores = swarm.getAlgorithmScores(point);

      expect(scores['isolation-forest']).toBeDefined();
      expect(scores['lstm-ae']).toBeDefined();
      expect(scores['vae']).toBeDefined();
      expect(scores['one-class-svm']).toBeDefined();
    });
  });

  describe('Evolution', () => {
    it('should evolve multiple generations', async () => {
      const trainingData = generateMixedData(100, 2);

      await swarm.train(trainingData);

      const generation = swarm.getGeneration();
      expect(generation).toBe(49); // 50 generations (0-49)
    });
  });
});

// Helper functions
function generateMixedData(count: number, dimensions: number): AnomalyPoint[] {
  const data: AnomalyPoint[] = [];

  // Generate normal data (80%)
  const normalCount = Math.floor(count * 0.8);
  for (let i = 0; i < normalCount; i++) {
    data.push({
      timestamp: Date.now(),
      features: Array.from({ length: dimensions }, () => Math.random()),
      label: 'normal',
    });
  }

  // Generate anomalies (20%)
  const anomalyCount = count - normalCount;
  for (let i = 0; i < anomalyCount; i++) {
    data.push({
      timestamp: Date.now(),
      features: Array.from({ length: dimensions }, () => 5 + Math.random() * 5),
      label: 'anomaly',
    });
  }

  return data;
}
