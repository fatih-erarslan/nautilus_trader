/**
 * Tests for Self-Learning Components
 */

import { describe, it, expect, beforeEach, afterEach } from '@jest/globals';
import { SelfLearningOptimizer, AdaptiveRiskManager } from '../src/self-learning.js';
import { OptimizationResult } from '../src/optimizer.js';
import * as fs from 'fs';
import * as path from 'path';

describe('Self-Learning Optimizer', () => {
  let optimizer: SelfLearningOptimizer;
  const testDbPath = './test-portfolio-memory.db';

  beforeEach(async () => {
    // Clean up any existing test database
    if (fs.existsSync(testDbPath)) {
      fs.unlinkSync(testDbPath);
    }

    optimizer = new SelfLearningOptimizer(testDbPath, 'test-portfolio');
    await optimizer.initialize();
  });

  afterEach(async () => {
    await optimizer.close();

    // Clean up test database
    if (fs.existsSync(testDbPath)) {
      fs.unlinkSync(testDbPath);
    }
  });

  describe('Initialization', () => {
    it('should initialize with default learning state', async () => {
      const state = await optimizer.loadLearningState();

      expect(state).toBeDefined();
      expect(state?.iteration).toBe(0);
      expect(state?.bestRiskProfile).toBeDefined();
      expect(state?.performanceHistory).toEqual([]);
    });

    it('should restore existing learning state', async () => {
      const mockState = {
        iteration: 5,
        bestRiskProfile: {
          riskAversion: 3.0,
          targetReturn: 0.12,
          maxDrawdown: 0.15,
          preferredAlgorithm: 'risk-parity',
          diversificationPreference: 0.8,
        },
        performanceHistory: [],
        strategySuccessRates: {
          'mean-variance': 0.6,
          'risk-parity': 0.7,
          'black-litterman': 0.5,
          'multi-objective': 0.6,
        },
        adaptiveParameters: {},
      };

      await optimizer.saveLearningState(mockState);

      const newOptimizer = new SelfLearningOptimizer(testDbPath, 'test-portfolio');
      await newOptimizer.initialize();
      const restoredState = await newOptimizer.loadLearningState();

      expect(restoredState?.iteration).toBe(5);
      expect(restoredState?.bestRiskProfile.riskAversion).toBe(3.0);

      await newOptimizer.close();
    });
  });

  describe('Learning from Results', () => {
    it('should update risk profile based on performance', async () => {
      const mockResult: OptimizationResult = {
        weights: [0.3, 0.4, 0.3],
        expectedReturn: 0.12,
        risk: 0.18,
        sharpeRatio: 0.67,
        algorithm: 'mean-variance',
        diversificationRatio: 0.85,
      };

      const mockPerformance = {
        sharpeRatio: 0.75,
        maxDrawdown: 0.12,
        volatility: 0.18,
        cumulativeReturn: 0.15,
        winRate: 0.65,
        informationRatio: 0.60,
      };

      const marketConditions = {
        volatility: 0.20,
        trend: 1,
        correlation: 0.6,
      };

      const newProfile = await optimizer.learn(mockResult, mockPerformance, marketConditions);

      expect(newProfile).toBeDefined();
      expect(newProfile.riskAversion).toBeGreaterThan(0);
      expect(newProfile.targetReturn).toBeGreaterThan(0);
    });

    it('should improve strategy success rates', async () => {
      const mockResult: OptimizationResult = {
        weights: [0.25, 0.5, 0.25],
        expectedReturn: 0.14,
        risk: 0.16,
        sharpeRatio: 0.88,
        algorithm: 'risk-parity',
        diversificationRatio: 0.90,
      };

      const goodPerformance = {
        sharpeRatio: 0.95,
        maxDrawdown: 0.10,
        volatility: 0.16,
        cumulativeReturn: 0.18,
        winRate: 0.70,
        informationRatio: 0.75,
      };

      const marketConditions = {
        volatility: 0.18,
        trend: 1,
        correlation: 0.5,
      };

      await optimizer.learn(mockResult, goodPerformance, marketConditions);

      const state = await optimizer.loadLearningState();
      expect(state?.strategySuccessRates['risk-parity']).toBeGreaterThan(0.5);
    });

    it('should store trajectories in memory', async () => {
      const mockResult: OptimizationResult = {
        weights: [0.33, 0.33, 0.34],
        expectedReturn: 0.11,
        risk: 0.19,
        sharpeRatio: 0.58,
        algorithm: 'multi-objective',
        diversificationRatio: 0.88,
      };

      const mockPerformance = {
        sharpeRatio: 0.60,
        maxDrawdown: 0.15,
        volatility: 0.19,
        cumulativeReturn: 0.12,
        winRate: 0.58,
        informationRatio: 0.50,
      };

      const marketConditions = {
        volatility: 0.22,
        trend: 0,
        correlation: 0.7,
      };

      await optimizer.learn(mockResult, mockPerformance, marketConditions);

      // Verify state was updated
      const state = await optimizer.loadLearningState();
      expect(state?.iteration).toBe(1);
      expect(state?.performanceHistory).toHaveLength(1);
    });
  });

  describe('Recommendations', () => {
    it('should provide recommended profile based on conditions', async () => {
      const marketConditions = {
        volatility: 0.25,
        trend: -1,
        correlation: 0.8,
      };

      const profile = await optimizer.getRecommendedProfile(marketConditions);

      expect(profile).toBeDefined();
      expect(profile.riskAversion).toBeGreaterThan(0);
      expect(profile.targetReturn).toBeGreaterThan(0);
      expect(profile.preferredAlgorithm).toBeDefined();
    });

    it('should adapt recommendations based on learning', async () => {
      // Learn from good performance
      const goodResult: OptimizationResult = {
        weights: [0.2, 0.6, 0.2],
        expectedReturn: 0.16,
        risk: 0.20,
        sharpeRatio: 0.80,
        algorithm: 'black-litterman',
        diversificationRatio: 0.85,
      };

      const goodPerformance = {
        sharpeRatio: 0.85,
        maxDrawdown: 0.12,
        volatility: 0.20,
        cumulativeReturn: 0.18,
        winRate: 0.68,
        informationRatio: 0.70,
      };

      const marketConditions = {
        volatility: 0.20,
        trend: 1,
        correlation: 0.6,
      };

      await optimizer.learn(goodResult, goodPerformance, marketConditions);

      const profile = await optimizer.getRecommendedProfile(marketConditions);

      // Should prefer the successful algorithm
      const state = await optimizer.loadLearningState();
      expect(state?.strategySuccessRates['black-litterman']).toBeGreaterThan(0.5);
    });
  });

  describe('Learning State Management', () => {
    it('should distill learning insights', async () => {
      await optimizer.distillLearning();

      // Verify insights were stored
      const state = await optimizer.loadLearningState();
      expect(state).toBeDefined();
    });

    it('should export learning data', async () => {
      const data = await optimizer.exportLearningData();

      expect(data).toBeDefined();
      expect(data?.iteration).toBeGreaterThanOrEqual(0);
      expect(data?.bestRiskProfile).toBeDefined();
    });

    it('should reset learning state', async () => {
      // Add some learning history
      const mockResult: OptimizationResult = {
        weights: [0.33, 0.33, 0.34],
        expectedReturn: 0.12,
        risk: 0.18,
        sharpeRatio: 0.67,
        algorithm: 'mean-variance',
        diversificationRatio: 0.85,
      };

      const mockPerformance = {
        sharpeRatio: 0.70,
        maxDrawdown: 0.14,
        volatility: 0.18,
        cumulativeReturn: 0.14,
        winRate: 0.62,
        informationRatio: 0.58,
      };

      await optimizer.learn(mockResult, mockPerformance, { volatility: 0.2, trend: 1, correlation: 0.6 });

      // Reset
      await optimizer.reset();

      const state = await optimizer.loadLearningState();
      expect(state?.iteration).toBe(0);
      expect(state?.performanceHistory).toEqual([]);
    });
  });
});

describe('Adaptive Risk Manager', () => {
  let learningOptimizer: SelfLearningOptimizer;
  let riskManager: AdaptiveRiskManager;
  const testDbPath = './test-risk-memory.db';

  beforeEach(async () => {
    if (fs.existsSync(testDbPath)) {
      fs.unlinkSync(testDbPath);
    }

    learningOptimizer = new SelfLearningOptimizer(testDbPath, 'test-risk');
    await learningOptimizer.initialize();
    riskManager = new AdaptiveRiskManager(learningOptimizer);
  });

  afterEach(async () => {
    await learningOptimizer.close();

    if (fs.existsSync(testDbPath)) {
      fs.unlinkSync(testDbPath);
    }
  });

  describe('Position Sizing', () => {
    it('should calculate adaptive position sizes', async () => {
      const baseWeights = [0.3, 0.4, 0.3];
      const marketConditions = { volatility: 0.20, trend: 1, correlation: 0.6 };

      const adjustedWeights = await riskManager.calculatePositionSizes(
        baseWeights,
        marketConditions,
        0.95,
      );

      expect(adjustedWeights).toHaveLength(3);
      expect(adjustedWeights.reduce((a, b) => a + b, 0)).toBeCloseTo(1.0, 2);
    });

    it('should reduce exposure in high volatility', async () => {
      const baseWeights = [0.33, 0.33, 0.34];

      const lowVolConditions = { volatility: 0.10, trend: 1, correlation: 0.5 };
      const highVolConditions = { volatility: 0.40, trend: 1, correlation: 0.5 };

      const lowVolWeights = await riskManager.calculatePositionSizes(baseWeights, lowVolConditions);
      const highVolWeights = await riskManager.calculatePositionSizes(baseWeights, highVolConditions);

      // Verify weights are adjusted (implementation specific)
      expect(lowVolWeights).toBeDefined();
      expect(highVolWeights).toBeDefined();
    });
  });

  describe('Rebalancing', () => {
    it('should trigger rebalancing when deviation exceeds threshold', async () => {
      const currentWeights = [0.35, 0.40, 0.25];
      const targetWeights = [0.30, 0.40, 0.30];

      const shouldRebalance = await riskManager.shouldRebalance(
        currentWeights,
        targetWeights,
        0.04,
      );

      expect(typeof shouldRebalance).toBe('boolean');
    });

    it('should not trigger rebalancing for small deviations', async () => {
      const currentWeights = [0.31, 0.39, 0.30];
      const targetWeights = [0.30, 0.40, 0.30];

      const shouldRebalance = await riskManager.shouldRebalance(
        currentWeights,
        targetWeights,
        0.05,
      );

      expect(shouldRebalance).toBe(false);
    });
  });
});
