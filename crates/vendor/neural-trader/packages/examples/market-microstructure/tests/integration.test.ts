/**
 * Integration tests for MarketMicrostructure
 */

import {
  MarketMicrostructure,
  createMarketMicrostructure,
  OrderBook,
  OrderFlow
} from '../src';
import * as fs from 'fs';
import * as path from 'path';

describe('MarketMicrostructure Integration', () => {
  let mm: MarketMicrostructure;
  const testDbPath = path.join(__dirname, 'integration-test.db');

  beforeEach(() => {
    // Clean up any existing test database
    if (fs.existsSync(testDbPath)) {
      fs.unlinkSync(testDbPath);
    }
  });

  afterEach(async () => {
    if (mm) {
      await mm.close();
    }

    // Clean up test database
    if (fs.existsSync(testDbPath)) {
      fs.unlinkSync(testDbPath);
    }
  });

  const createMockOrderBook = (timestamp?: number): OrderBook => ({
    bids: [
      { price: 100.0, size: 1000, orders: 10 },
      { price: 99.9, size: 800, orders: 8 },
      { price: 99.8, size: 600, orders: 6 }
    ],
    asks: [
      { price: 100.1, size: 1000, orders: 10 },
      { price: 100.2, size: 800, orders: 8 },
      { price: 100.3, size: 600, orders: 6 }
    ],
    timestamp: timestamp || Date.now(),
    symbol: 'BTCUSD'
  });

  const createMockTrades = (count: number = 10): OrderFlow[] => {
    const trades: OrderFlow[] = [];

    for (let i = 0; i < count; i++) {
      trades.push({
        type: i % 2 === 0 ? 'buy' : 'sell',
        price: 100 + (Math.random() - 0.5) * 0.5,
        size: Math.random() * 100 + 50,
        aggressor: i % 2 === 0 ? 'buyer' : 'seller',
        timestamp: Date.now() + i * 1000
      });
    }

    return trades;
  };

  describe('Initialization', () => {
    test('should create and initialize MarketMicrostructure', async () => {
      mm = new MarketMicrostructure({
        agentDbPath: testDbPath,
        useSwarm: false
      });

      await mm.initialize();

      expect(mm).toBeDefined();
    });

    test('should create using convenience function', async () => {
      mm = await createMarketMicrostructure({
        agentDbPath: testDbPath,
        useSwarm: false
      });

      expect(mm).toBeDefined();
    });

    test('should initialize with swarm enabled', async () => {
      mm = await createMarketMicrostructure({
        agentDbPath: testDbPath,
        useSwarm: true,
        swarmConfig: {
          numAgents: 10,
          generations: 5
        }
      });

      expect(mm).toBeDefined();
    });

    test('should throw error if analyzing before initialization', async () => {
      mm = new MarketMicrostructure({ agentDbPath: testDbPath });

      const orderBook = createMockOrderBook();

      await expect(mm.analyze(orderBook)).rejects.toThrow('not initialized');
    });
  });

  describe('Order Book Analysis', () => {
    beforeEach(async () => {
      mm = await createMarketMicrostructure({
        agentDbPath: testDbPath,
        useSwarm: false
      });
    });

    test('should analyze order book and return metrics', async () => {
      const orderBook = createMockOrderBook();
      const result = await mm.analyze(orderBook);

      expect(result).toHaveProperty('metrics');
      expect(result).toHaveProperty('pattern');

      expect(result.metrics).toHaveProperty('bidAskSpread');
      expect(result.metrics).toHaveProperty('spreadBps');
      expect(result.metrics).toHaveProperty('imbalance');
      expect(result.metrics).toHaveProperty('liquidityScore');
    });

    test('should analyze order book with trades', async () => {
      const orderBook = createMockOrderBook();
      const trades = createMockTrades(10);

      const result = await mm.analyze(orderBook, trades);

      expect(result.metrics).toHaveProperty('buyPressure');
      expect(result.metrics).toHaveProperty('sellPressure');
      expect(result.metrics).toHaveProperty('netFlow');
    });

    test('should build pattern over multiple analyses', async () => {
      // First analysis - no pattern yet
      const orderBook1 = createMockOrderBook();
      const result1 = await mm.analyze(orderBook1);

      expect(result1.pattern).toBeNull();

      // Second analysis - should have features
      const orderBook2 = createMockOrderBook(Date.now() + 1000);
      const result2 = await mm.analyze(orderBook2);

      expect(result2.pattern).not.toBeNull();
      expect(result2.pattern).toHaveProperty('features');
    });
  });

  describe('Pattern Learning', () => {
    beforeEach(async () => {
      mm = await createMarketMicrostructure({
        agentDbPath: testDbPath,
        useSwarm: false
      });
    });

    test('should learn from observed outcomes', async () => {
      // Build history
      for (let i = 0; i < 5; i++) {
        const orderBook = createMockOrderBook(Date.now() + i * 1000);
        await mm.analyze(orderBook);
      }

      // Learn outcome
      const outcome = {
        priceMove: 0.5,
        spreadChange: -0.01,
        liquidityChange: 0.1,
        timeHorizon: 5000
      };

      await mm.learn(outcome, 'profitable_pattern');

      const patterns = mm.getPatterns();
      expect(patterns.length).toBeGreaterThan(0);
      expect(patterns[0].label).toBe('profitable_pattern');
    });

    test('should recognize learned patterns', async () => {
      // Build history and learn pattern
      for (let i = 0; i < 5; i++) {
        const orderBook = createMockOrderBook(Date.now() + i * 1000);
        await mm.analyze(orderBook);
      }

      const outcome = {
        priceMove: 1.0,
        spreadChange: 0.02,
        liquidityChange: -0.2,
        timeHorizon: 5000
      };

      await mm.learn(outcome, 'known_pattern');

      // Analyze again with similar conditions
      for (let i = 0; i < 3; i++) {
        const orderBook = createMockOrderBook(Date.now() + 10000 + i * 1000);
        await mm.analyze(orderBook);
      }

      const result = await mm.analyze(createMockOrderBook());

      // Should recognize pattern or have prediction
      expect(result.pattern).not.toBeNull();
    });
  });

  describe('Swarm Features', () => {
    beforeEach(async () => {
      mm = await createMarketMicrostructure({
        agentDbPath: testDbPath,
        useSwarm: true,
        swarmConfig: {
          numAgents: 10,
          generations: 5
        }
      });
    });

    test('should detect anomalies with swarm', async () => {
      const normalBook = createMockOrderBook();
      const result = await mm.analyze(normalBook);

      expect(result).toHaveProperty('anomaly');
      expect(result.anomaly).toHaveProperty('isAnomaly');
      expect(result.anomaly).toHaveProperty('confidence');
      expect(result.anomaly).toHaveProperty('anomalyType');
    });

    test('should explore features with sufficient history', async () => {
      // Build history
      for (let i = 0; i < 15; i++) {
        const orderBook = createMockOrderBook(Date.now() + i * 1000);
        await mm.analyze(orderBook);
      }

      const featureSets = await mm.exploreFeatures();

      expect(Array.isArray(featureSets)).toBe(true);
      expect(featureSets.length).toBeGreaterThan(0);
    }, 20000);

    test('should optimize features', async () => {
      // Build history
      for (let i = 0; i < 5; i++) {
        const orderBook = createMockOrderBook(Date.now() + i * 1000);
        await mm.analyze(orderBook);
      }

      const optimized = await mm.optimizeFeatures('profitability');

      expect(optimized).toBeDefined();
      expect(optimized).toHaveProperty('spreadTrend');
    });

    test('should throw error when exploring without sufficient history', async () => {
      // Only analyze once
      const orderBook = createMockOrderBook();
      await mm.analyze(orderBook);

      await expect(mm.exploreFeatures()).rejects.toThrow('at least 10 metrics points');
    });
  });

  describe('Statistics', () => {
    beforeEach(async () => {
      mm = await createMarketMicrostructure({
        agentDbPath: testDbPath,
        useSwarm: true,
        swarmConfig: {
          numAgents: 10,
          generations: 5
        }
      });
    });

    test('should provide comprehensive statistics', async () => {
      // Build some history
      for (let i = 0; i < 5; i++) {
        const orderBook = createMockOrderBook(Date.now() + i * 1000);
        await mm.analyze(orderBook);
      }

      const stats = mm.getStatistics();

      expect(stats).toHaveProperty('analyzer');
      expect(stats).toHaveProperty('learner');
      expect(stats).toHaveProperty('swarm');

      expect(stats.analyzer.metricsCount).toBeGreaterThan(0);
      expect(stats.learner).toHaveProperty('totalPatterns');
      expect(stats.swarm).toHaveProperty('totalAgents');
    });

    test('should track metrics count', async () => {
      const initialStats = mm.getStatistics();
      expect(initialStats.analyzer.metricsCount).toBe(0);

      // Analyze multiple times
      for (let i = 0; i < 10; i++) {
        const orderBook = createMockOrderBook(Date.now() + i * 1000);
        await mm.analyze(orderBook);
      }

      const finalStats = mm.getStatistics();
      expect(finalStats.analyzer.metricsCount).toBe(10);
    });
  });

  describe('State Management', () => {
    beforeEach(async () => {
      mm = await createMarketMicrostructure({
        agentDbPath: testDbPath,
        useSwarm: false
      });
    });

    test('should reset analyzer state', async () => {
      // Build history
      for (let i = 0; i < 5; i++) {
        const orderBook = createMockOrderBook(Date.now() + i * 1000);
        await mm.analyze(orderBook);
      }

      const statsBeforeReset = mm.getStatistics();
      expect(statsBeforeReset.analyzer.metricsCount).toBeGreaterThan(0);

      mm.reset();

      const statsAfterReset = mm.getStatistics();
      expect(statsAfterReset.analyzer.metricsCount).toBe(0);
    });

    test('should preserve learned patterns after reset', async () => {
      // Build history and learn
      for (let i = 0; i < 5; i++) {
        const orderBook = createMockOrderBook(Date.now() + i * 1000);
        await mm.analyze(orderBook);
      }

      await mm.learn({
        priceMove: 0.5,
        spreadChange: 0.01,
        liquidityChange: 0.1,
        timeHorizon: 5000
      }, 'preserved_pattern');

      const patternsBeforeReset = mm.getPatterns();
      expect(patternsBeforeReset.length).toBeGreaterThan(0);

      mm.reset();

      const patternsAfterReset = mm.getPatterns();
      expect(patternsAfterReset.length).toBe(patternsBeforeReset.length);
    });
  });

  describe('End-to-End Workflow', () => {
    test('should complete full market microstructure analysis workflow', async () => {
      mm = await createMarketMicrostructure({
        agentDbPath: testDbPath,
        useSwarm: true,
        swarmConfig: {
          numAgents: 10,
          generations: 5
        }
      });

      // Phase 1: Collect market data
      const orderBooks: OrderBook[] = [];
      for (let i = 0; i < 20; i++) {
        const book = createMockOrderBook(Date.now() + i * 1000);
        book.bids[0].price = 100 + i * 0.1;
        book.asks[0].price = 100.1 + i * 0.1;
        orderBooks.push(book);
      }

      // Phase 2: Analyze all order books
      for (const book of orderBooks) {
        const trades = createMockTrades(5);
        await mm.analyze(book, trades);
      }

      // Phase 3: Learn from outcomes
      const outcome = {
        priceMove: 1.5,
        spreadChange: 0.05,
        liquidityChange: -0.1,
        timeHorizon: 5000
      };
      await mm.learn(outcome, 'uptrend_pattern');

      // Phase 4: Explore features
      const featureSets = await mm.exploreFeatures();
      expect(featureSets.length).toBeGreaterThan(0);

      // Phase 5: Optimize features
      const optimized = await mm.optimizeFeatures('sharpe');
      expect(optimized).toBeDefined();

      // Phase 6: Get final statistics
      const stats = mm.getStatistics();
      expect(stats.analyzer.metricsCount).toBe(20);
      expect(stats.learner.totalPatterns).toBeGreaterThan(0);
      expect(stats.swarm.totalAgents).toBe(10);

      // Phase 7: Verify patterns learned
      const patterns = mm.getPatterns();
      expect(patterns.length).toBeGreaterThan(0);
      expect(patterns.some(p => p.label === 'uptrend_pattern')).toBe(true);
    }, 30000);
  });
});
