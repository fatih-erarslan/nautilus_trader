/**
 * Tests for PatternLearner
 */

import { PatternLearner, PatternFeatures, Pattern } from '../src/pattern-learner';
import { MicrostructureMetrics } from '../src/order-book-analyzer';
import * as fs from 'fs';
import * as path from 'path';

describe('PatternLearner', () => {
  let learner: PatternLearner;
  const testDbPath = path.join(__dirname, 'test-patterns.db');

  beforeEach(async () => {
    // Clean up any existing test database
    if (fs.existsSync(testDbPath)) {
      fs.unlinkSync(testDbPath);
    }

    learner = new PatternLearner({
      agentDbPath: testDbPath,
      minConfidence: 0.7,
      maxPatterns: 100,
      useNeuralPredictor: false // Disable for faster tests
    });

    await learner.initialize();
  });

  afterEach(async () => {
    await learner.close();

    // Clean up test database
    if (fs.existsSync(testDbPath)) {
      fs.unlinkSync(testDbPath);
    }
  });

  const createMockMetricsHistory = (count: number = 10): MicrostructureMetrics[] => {
    const metrics: MicrostructureMetrics[] = [];
    const baseTime = Date.now();

    for (let i = 0; i < count; i++) {
      metrics.push({
        bidAskSpread: 0.1 + Math.random() * 0.05,
        spreadBps: 10 + Math.random() * 5,
        effectiveSpread: 0.12 + Math.random() * 0.03,
        bidDepth: 1000 + Math.random() * 500,
        askDepth: 1000 + Math.random() * 500,
        imbalance: (Math.random() - 0.5) * 0.4,
        vpin: Math.random() * 0.5,
        orderFlowToxicity: Math.random() * 0.6,
        adverseSelection: Math.random() * 0.1,
        buyPressure: 0.4 + Math.random() * 0.2,
        sellPressure: 0.4 + Math.random() * 0.2,
        netFlow: (Math.random() - 0.5) * 0.3,
        midPrice: 100 + i * 0.1 + (Math.random() - 0.5) * 0.2,
        microPrice: 100 + i * 0.1 + (Math.random() - 0.5) * 0.15,
        priceImpact: 0.001 + Math.random() * 0.002,
        liquidityScore: 0.6 + Math.random() * 0.3,
        resilienceTime: 100 + Math.random() * 200,
        timestamp: baseTime + i * 1000
      });
    }

    return metrics;
  };

  const createMockOutcome = (): Pattern['outcome'] => ({
    priceMove: (Math.random() - 0.5) * 2,
    spreadChange: (Math.random() - 0.5) * 0.1,
    liquidityChange: (Math.random() - 0.5) * 0.3,
    timeHorizon: 5000
  });

  describe('Feature Extraction', () => {
    test('should extract features from metrics history', () => {
      const metricsHistory = createMockMetricsHistory(10);
      const features = learner.extractFeatures(metricsHistory);

      expect(features).toHaveProperty('spreadTrend');
      expect(features).toHaveProperty('spreadVolatility');
      expect(features).toHaveProperty('depthImbalance');
      expect(features).toHaveProperty('depthTrend');
      expect(features).toHaveProperty('flowPersistence');
      expect(features).toHaveProperty('flowReversal');
      expect(features).toHaveProperty('toxicityLevel');
      expect(features).toHaveProperty('informedTradingProbability');
      expect(features).toHaveProperty('priceEfficiency');
      expect(features).toHaveProperty('microPriceDivergence');
      expect(features).toHaveProperty('timestamp');
    });

    test('should calculate spread trend', () => {
      const metricsHistory = createMockMetricsHistory(10);
      // Make spreads increasing
      metricsHistory.forEach((m, i) => {
        m.bidAskSpread = 0.1 + i * 0.01;
      });

      const features = learner.extractFeatures(metricsHistory);

      expect(features.spreadTrend).toBeGreaterThan(0);
    });

    test('should calculate spread volatility', () => {
      const metricsHistory = createMockMetricsHistory(10);
      const features = learner.extractFeatures(metricsHistory);

      expect(features.spreadVolatility).toBeGreaterThanOrEqual(0);
    });

    test('should calculate flow persistence', () => {
      const metricsHistory = createMockMetricsHistory(10);
      // Make flow persistent
      metricsHistory.forEach(m => {
        m.netFlow = 0.3;
      });

      const features = learner.extractFeatures(metricsHistory);

      expect(features.flowPersistence).toBeGreaterThan(0.8);
    });

    test('should detect flow reversal', () => {
      const metricsHistory = createMockMetricsHistory(10);
      // Create alternating flow
      metricsHistory.forEach((m, i) => {
        m.netFlow = i % 2 === 0 ? 0.3 : -0.3;
      });

      const features = learner.extractFeatures(metricsHistory);

      expect(features.flowReversal).toBeGreaterThan(0);
    });

    test('should throw error with insufficient data', () => {
      const metricsHistory = createMockMetricsHistory(1);

      expect(() => {
        learner.extractFeatures(metricsHistory);
      }).toThrow('Need at least 2 metrics points');
    });
  });

  describe('Pattern Learning', () => {
    test('should learn new pattern', async () => {
      const metricsHistory = createMockMetricsHistory(10);
      const features = learner.extractFeatures(metricsHistory);
      const outcome = createMockOutcome();

      const pattern = await learner.learnPattern(features, outcome, 'test_pattern');

      expect(pattern).toHaveProperty('id');
      expect(pattern).toHaveProperty('features');
      expect(pattern).toHaveProperty('label');
      expect(pattern).toHaveProperty('confidence');
      expect(pattern).toHaveProperty('outcome');
      expect(pattern).toHaveProperty('metadata');

      expect(pattern.label).toBe('test_pattern');
      expect(pattern.confidence).toBe(0.5); // Initial confidence
      expect(pattern.metadata.occurrences).toBe(1);
    });

    test('should update existing pattern', async () => {
      const metricsHistory = createMockMetricsHistory(10);
      const features = learner.extractFeatures(metricsHistory);
      const outcome1 = createMockOutcome();

      // Learn pattern first time
      const pattern1 = await learner.learnPattern(features, outcome1, 'test_pattern');
      const initialConfidence = pattern1.confidence;

      // Learn same pattern again
      const pattern2 = await learner.learnPattern(features, outcome1, 'test_pattern');

      expect(pattern2.confidence).toBeGreaterThan(initialConfidence);
      expect(pattern2.metadata.occurrences).toBe(2);
    });

    test('should auto-label patterns', async () => {
      const metricsHistory = createMockMetricsHistory(10);
      const features = learner.extractFeatures(metricsHistory);

      // Set specific feature values to trigger auto-labeling
      features.spreadTrend = 0.2;
      features.depthImbalance = 0.4;
      features.toxicityLevel = 0.7;

      const outcome = createMockOutcome();
      const pattern = await learner.learnPattern(features, outcome);

      expect(pattern.label).toContain('widening_spread');
      expect(pattern.label).toContain('buy_pressure');
      expect(pattern.label).toContain('toxic_flow');
    });

    test('should store patterns in AgentDB', async () => {
      const metricsHistory = createMockMetricsHistory(10);
      const features = learner.extractFeatures(metricsHistory);
      const outcome = createMockOutcome();

      await learner.learnPattern(features, outcome, 'stored_pattern');

      const patterns = learner.getPatterns();
      expect(patterns.length).toBeGreaterThan(0);
      expect(patterns[0].label).toBe('stored_pattern');
    });
  });

  describe('Pattern Recognition', () => {
    test('should recognize similar patterns', async () => {
      const metricsHistory = createMockMetricsHistory(10);
      const features = learner.extractFeatures(metricsHistory);
      const outcome = createMockOutcome();

      // Learn a pattern
      const learnedPattern = await learner.learnPattern(features, outcome, 'recognizable');

      // Try to recognize the same pattern
      const recognized = await learner.recognizePattern(features);

      expect(recognized).not.toBeNull();
      if (recognized) {
        expect(recognized.id).toBe(learnedPattern.id);
      }
    });

    test('should return null for unknown patterns', async () => {
      const metricsHistory = createMockMetricsHistory(10);
      const features = learner.extractFeatures(metricsHistory);

      const recognized = await learner.recognizePattern(features);

      expect(recognized).toBeNull();
    });

    test('should respect confidence threshold', async () => {
      const highConfidenceLearner = new PatternLearner({
        agentDbPath: testDbPath + '.high',
        minConfidence: 0.95
      });

      await highConfidenceLearner.initialize();

      const metricsHistory = createMockMetricsHistory(10);
      const features = learner.extractFeatures(metricsHistory);
      const outcome = createMockOutcome();

      // Learn pattern (will have confidence 0.5)
      await highConfidenceLearner.learnPattern(features, outcome);

      // Should not recognize due to low confidence
      const recognized = await highConfidenceLearner.recognizePattern(features);

      expect(recognized).toBeNull();

      await highConfidenceLearner.close();
    });
  });

  describe('Statistics', () => {
    test('should provide pattern statistics', async () => {
      const metricsHistory = createMockMetricsHistory(10);

      // Learn multiple patterns
      for (let i = 0; i < 5; i++) {
        const features = learner.extractFeatures(metricsHistory);
        features.spreadTrend += i * 0.1; // Make features different
        const outcome = createMockOutcome();

        await learner.learnPattern(features, outcome);
      }

      const stats = learner.getStatistics();

      expect(stats.totalPatterns).toBe(5);
      expect(stats).toHaveProperty('highConfidencePatterns');
      expect(stats).toHaveProperty('avgConfidence');
      expect(stats).toHaveProperty('mostCommonLabels');
      expect(Array.isArray(stats.mostCommonLabels)).toBe(true);
    });

    test('should track most common labels', async () => {
      const metricsHistory = createMockMetricsHistory(10);

      // Learn multiple patterns with same label
      for (let i = 0; i < 3; i++) {
        const features = learner.extractFeatures(metricsHistory);
        features.spreadTrend += i * 0.1;
        const outcome = createMockOutcome();

        await learner.learnPattern(features, outcome, 'common_pattern');
      }

      const stats = learner.getStatistics();

      expect(stats.mostCommonLabels.length).toBeGreaterThan(0);
      expect(stats.mostCommonLabels[0].label).toBe('common_pattern');
      expect(stats.mostCommonLabels[0].count).toBe(3);
    });
  });

  describe('Pattern Persistence', () => {
    test('should persist patterns to database', async () => {
      const metricsHistory = createMockMetricsHistory(10);
      const features = learner.extractFeatures(metricsHistory);
      const outcome = createMockOutcome();

      await learner.learnPattern(features, outcome, 'persistent_pattern');
      await learner.close();

      // Create new learner and load patterns
      const newLearner = new PatternLearner({ agentDbPath: testDbPath });
      await newLearner.initialize();

      const patterns = newLearner.getPatterns();
      expect(patterns.length).toBeGreaterThan(0);
      expect(patterns[0].label).toBe('persistent_pattern');

      await newLearner.close();
    });
  });

  describe('Pattern Pruning', () => {
    test('should prune patterns when limit exceeded', async () => {
      const smallLearner = new PatternLearner({
        agentDbPath: testDbPath + '.small',
        maxPatterns: 10
      });

      await smallLearner.initialize();

      const metricsHistory = createMockMetricsHistory(10);

      // Learn more patterns than limit
      for (let i = 0; i < 15; i++) {
        const features = learner.extractFeatures(metricsHistory);
        features.spreadTrend += i * 0.1; // Make each unique
        const outcome = createMockOutcome();

        await smallLearner.learnPattern(features, outcome);
      }

      const patterns = smallLearner.getPatterns();
      expect(patterns.length).toBeLessThanOrEqual(10);

      await smallLearner.close();
    });
  });
});
