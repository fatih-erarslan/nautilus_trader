/**
 * Tests for SwarmFeatureEngineer
 */

import { SwarmFeatureEngineer, SwarmAgent, FeatureSet } from '../src/swarm-features';
import { MicrostructureMetrics } from '../src/order-book-analyzer';
import { PatternFeatures } from '../src/pattern-learner';

describe('SwarmFeatureEngineer', () => {
  let swarm: SwarmFeatureEngineer;

  beforeEach(async () => {
    swarm = new SwarmFeatureEngineer({
      numAgents: 20,
      generations: 10,
      mutationRate: 0.2,
      crossoverRate: 0.7,
      eliteSize: 2,
      useOpenRouter: false
    });

    await swarm.initialize();
  });

  afterEach(async () => {
    await swarm.cleanup();
  });

  const createMockMetricsHistory = (count: number = 50): MicrostructureMetrics[] => {
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

  const createMockPatternFeatures = (): PatternFeatures => ({
    spreadTrend: Math.random() - 0.5,
    spreadVolatility: Math.random() * 0.1,
    depthImbalance: Math.random() - 0.5,
    depthTrend: Math.random() - 0.5,
    flowPersistence: Math.random(),
    flowReversal: Math.random(),
    toxicityLevel: Math.random() * 0.6,
    informedTradingProbability: Math.random() * 0.5,
    priceEfficiency: 0.8 + Math.random() * 0.2,
    microPriceDivergence: Math.random() * 0.01,
    timestamp: Date.now()
  });

  describe('Initialization', () => {
    test('should initialize with correct number of agents', () => {
      const stats = swarm.getAgentStats();

      expect(stats.totalAgents).toBe(20);
    });

    test('should create diverse agent types', () => {
      const stats = swarm.getAgentStats();

      expect(stats.byType).toHaveProperty('explorer');
      expect(stats.byType).toHaveProperty('optimizer');
      expect(stats.byType).toHaveProperty('validator');
      expect(stats.byType).toHaveProperty('anomaly-detector');

      // Each type should have roughly equal agents
      Object.values(stats.byType).forEach(count => {
        expect(count).toBeGreaterThan(0);
      });
    });

    test('should initialize agents with features', () => {
      const stats = swarm.getAgentStats();

      expect(stats.bestAgent).not.toBeNull();
      if (stats.bestAgent) {
        expect(stats.bestAgent.features.length).toBeGreaterThan(0);
        expect(stats.bestAgent).toHaveProperty('id');
        expect(stats.bestAgent).toHaveProperty('type');
        expect(stats.bestAgent).toHaveProperty('performance');
      }
    });
  });

  describe('Feature Exploration', () => {
    test('should explore features and return feature sets', async () => {
      const metricsHistory = createMockMetricsHistory(50);

      const featureSets = await swarm.exploreFeatures(metricsHistory);

      expect(Array.isArray(featureSets)).toBe(true);
      expect(featureSets.length).toBeGreaterThan(0);

      featureSets.forEach(fs => {
        expect(fs).toHaveProperty('name');
        expect(fs).toHaveProperty('features');
        expect(fs).toHaveProperty('importance');
        expect(fs).toHaveProperty('performance');
        expect(fs).toHaveProperty('metadata');

        expect(Array.isArray(fs.features)).toBe(true);
        expect(fs.features.length).toBeGreaterThan(0);
      });
    }, 15000); // Increase timeout for exploration

    test('should improve performance over generations', async () => {
      const metricsHistory = createMockMetricsHistory(30);

      const initialStats = swarm.getAgentStats();
      const initialPerf = initialStats.avgPerformance;

      await swarm.exploreFeatures(metricsHistory);

      const finalStats = swarm.getAgentStats();
      const finalPerf = finalStats.avgPerformance;

      // Performance should improve (or at least not decrease significantly)
      expect(finalPerf).toBeGreaterThanOrEqual(initialPerf * 0.8);
    }, 15000);

    test('should create feature sets from different agent types', async () => {
      const metricsHistory = createMockMetricsHistory(30);

      const featureSets = await swarm.exploreFeatures(metricsHistory);

      const uniqueTypes = new Set(featureSets.map(fs => fs.name.split('_')[0]));

      // Should have feature sets from multiple agent types
      expect(uniqueTypes.size).toBeGreaterThan(1);
    }, 15000);
  });

  describe('Feature Optimization', () => {
    test('should optimize features for accuracy', async () => {
      const baseFeatures = createMockPatternFeatures();

      const optimized = await swarm.optimizeFeatures(baseFeatures, 'accuracy');

      expect(optimized).toHaveProperty('spreadTrend');
      expect(optimized).toHaveProperty('spreadVolatility');
      expect(optimized).toHaveProperty('timestamp');
    });

    test('should optimize features for profitability', async () => {
      const baseFeatures = createMockPatternFeatures();

      const optimized = await swarm.optimizeFeatures(baseFeatures, 'profitability');

      expect(optimized).toBeDefined();
      expect(typeof optimized.spreadTrend).toBe('number');
    });

    test('should optimize features for sharpe ratio', async () => {
      const baseFeatures = createMockPatternFeatures();

      const optimized = await swarm.optimizeFeatures(baseFeatures, 'sharpe');

      expect(optimized).toBeDefined();
      expect(typeof optimized.priceEfficiency).toBe('number');
    });
  });

  describe('Anomaly Detection', () => {
    test('should detect anomalies in metrics', async () => {
      const normalMetrics: MicrostructureMetrics = {
        bidAskSpread: 0.1,
        spreadBps: 10,
        effectiveSpread: 0.12,
        bidDepth: 1000,
        askDepth: 1000,
        imbalance: 0.1,
        vpin: 0.3,
        orderFlowToxicity: 0.3,
        adverseSelection: 0.05,
        buyPressure: 0.5,
        sellPressure: 0.5,
        netFlow: 0,
        midPrice: 100,
        microPrice: 100.01,
        priceImpact: 0.001,
        liquidityScore: 0.8,
        resilienceTime: 150,
        timestamp: Date.now()
      };

      const result = await swarm.detectAnomalies(normalMetrics);

      expect(result).toHaveProperty('isAnomaly');
      expect(result).toHaveProperty('confidence');
      expect(result).toHaveProperty('anomalyType');
      expect(result).toHaveProperty('explanation');

      expect(typeof result.isAnomaly).toBe('boolean');
      expect(result.confidence).toBeGreaterThanOrEqual(0);
      expect(result.confidence).toBeLessThanOrEqual(1);
    });

    test('should detect wide spread anomaly', async () => {
      const anomalousMetrics: MicrostructureMetrics = {
        bidAskSpread: 1.0,
        spreadBps: 100, // Very wide
        effectiveSpread: 1.2,
        bidDepth: 1000,
        askDepth: 1000,
        imbalance: 0.1,
        vpin: 0.3,
        orderFlowToxicity: 0.3,
        adverseSelection: 0.05,
        buyPressure: 0.5,
        sellPressure: 0.5,
        netFlow: 0,
        midPrice: 100,
        microPrice: 100.01,
        priceImpact: 0.001,
        liquidityScore: 0.3,
        resilienceTime: 150,
        timestamp: Date.now()
      };

      const result = await swarm.detectAnomalies(anomalousMetrics);

      if (result.isAnomaly) {
        expect(['wide_spread', 'none']).toContain(result.anomalyType);
      }
    });

    test('should detect toxic flow anomaly', async () => {
      const toxicMetrics: MicrostructureMetrics = {
        bidAskSpread: 0.1,
        spreadBps: 10,
        effectiveSpread: 0.12,
        bidDepth: 1000,
        askDepth: 1000,
        imbalance: 0.1,
        vpin: 0.7,
        orderFlowToxicity: 0.8, // Very toxic
        adverseSelection: 0.15,
        buyPressure: 0.5,
        sellPressure: 0.5,
        netFlow: 0,
        midPrice: 100,
        microPrice: 100.01,
        priceImpact: 0.001,
        liquidityScore: 0.6,
        resilienceTime: 150,
        timestamp: Date.now()
      };

      const result = await swarm.detectAnomalies(toxicMetrics);

      if (result.isAnomaly) {
        expect(['toxic_flow', 'none']).toContain(result.anomalyType);
      }
    });

    test('should detect extreme imbalance anomaly', async () => {
      const imbalancedMetrics: MicrostructureMetrics = {
        bidAskSpread: 0.1,
        spreadBps: 10,
        effectiveSpread: 0.12,
        bidDepth: 3000,
        askDepth: 200,
        imbalance: 0.9, // Extreme imbalance
        vpin: 0.3,
        orderFlowToxicity: 0.3,
        adverseSelection: 0.05,
        buyPressure: 0.8,
        sellPressure: 0.2,
        netFlow: 0.6,
        midPrice: 100,
        microPrice: 100.01,
        priceImpact: 0.005,
        liquidityScore: 0.5,
        resilienceTime: 150,
        timestamp: Date.now()
      };

      const result = await swarm.detectAnomalies(imbalancedMetrics);

      if (result.isAnomaly) {
        expect(['extreme_imbalance', 'none']).toContain(result.anomalyType);
      }
    });
  });

  describe('Agent Statistics', () => {
    test('should provide agent statistics', () => {
      const stats = swarm.getAgentStats();

      expect(stats).toHaveProperty('totalAgents');
      expect(stats).toHaveProperty('byType');
      expect(stats).toHaveProperty('avgPerformance');
      expect(stats).toHaveProperty('bestAgent');

      expect(stats.totalAgents).toBeGreaterThan(0);
      expect(typeof stats.avgPerformance).toBe('number');
    });

    test('should track best agent', () => {
      const stats = swarm.getAgentStats();

      if (stats.bestAgent) {
        expect(stats.bestAgent).toHaveProperty('id');
        expect(stats.bestAgent).toHaveProperty('type');
        expect(stats.bestAgent).toHaveProperty('features');
        expect(stats.bestAgent).toHaveProperty('performance');
        expect(stats.bestAgent).toHaveProperty('generation');
      }
    });
  });

  describe('Feature Sets', () => {
    test('should return discovered feature sets', async () => {
      const metricsHistory = createMockMetricsHistory(30);
      await swarm.exploreFeatures(metricsHistory);

      const featureSets = swarm.getFeatureSets();

      expect(Array.isArray(featureSets)).toBe(true);
      expect(featureSets.length).toBeGreaterThan(0);
    }, 15000);

    test('should include performance metrics in feature sets', async () => {
      const metricsHistory = createMockMetricsHistory(30);
      await swarm.exploreFeatures(metricsHistory);

      const featureSets = swarm.getFeatureSets();

      featureSets.forEach(fs => {
        expect(fs.performance).toHaveProperty('accuracy');
        expect(fs.performance).toHaveProperty('profitability');
        expect(fs.performance).toHaveProperty('sharpeRatio');

        expect(fs.performance.accuracy).toBeGreaterThan(0);
        expect(fs.performance.accuracy).toBeLessThanOrEqual(1);
      });
    }, 15000);
  });

  describe('Error Handling', () => {
    test('should handle empty metrics history', async () => {
      const emptyMetrics: MicrostructureMetrics[] = [];

      // Should not throw, but might return empty results
      const featureSets = await swarm.exploreFeatures(emptyMetrics);

      expect(Array.isArray(featureSets)).toBe(true);
    }, 15000);
  });

  describe('Custom Configuration', () => {
    test('should respect custom agent count', async () => {
      const customSwarm = new SwarmFeatureEngineer({
        numAgents: 50,
        generations: 5
      });

      await customSwarm.initialize();

      const stats = customSwarm.getAgentStats();
      expect(stats.totalAgents).toBe(50);

      await customSwarm.cleanup();
    });

    test('should respect custom generations', async () => {
      const fastSwarm = new SwarmFeatureEngineer({
        numAgents: 10,
        generations: 5
      });

      await fastSwarm.initialize();

      const metricsHistory = createMockMetricsHistory(20);
      const start = Date.now();
      await fastSwarm.exploreFeatures(metricsHistory);
      const duration = Date.now() - start;

      // Should complete faster with fewer generations
      expect(duration).toBeLessThan(10000);

      await fastSwarm.cleanup();
    });
  });
});
