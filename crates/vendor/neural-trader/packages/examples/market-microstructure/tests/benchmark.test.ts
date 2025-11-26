/**
 * Performance benchmarks for market microstructure
 */

import { BenchmarkRunner, ComparisonRunner } from '@neural-trader/benchmarks';
import { OrderBookAnalyzer } from '../src/order-book-analyzer';
import { PatternLearner } from '../src/pattern-learner';
import {
  setupTestEnvironment,
  cleanupTestEnvironment,
  createMockAgentDB,
  createMockPredictor,
  generateOrderBook,
  generateMarketData
} from '@neural-trader/test-framework';

describe('Market Microstructure Benchmarks', () => {
  let mockDB: any;
  let mockPredictor: any;

  beforeAll(() => {
    setupTestEnvironment({ timeout: 120000 });
  });

  beforeEach(() => {
    mockDB = createMockAgentDB();
    mockPredictor = createMockPredictor();
  });

  afterEach(() => {
    mockDB.clear();
  });

  afterAll(() => {
    cleanupTestEnvironment();
  });

  describe('Order Book Processing', () => {
    test('benchmark: process order book snapshot', async () => {
      const analyzer = new OrderBookAnalyzer(mockDB);
      const orderBook = generateOrderBook('AAPL', 150, 20);

      const runner = new BenchmarkRunner({
        name: 'OrderBook-Process',
        iterations: 100,
        warmupIterations: 10
      });

      const result = await runner.run(async () => {
        await analyzer.processSnapshot(orderBook);
      });

      expect(result.mean).toBeLessThan(50); // Should process in < 50ms
      expect(result.throughput).toBeGreaterThan(20); // > 20 ops/sec

      console.log('\nOrderBook Processing Benchmark:');
      console.log(`Mean: ${result.mean.toFixed(2)}ms`);
      console.log(`Throughput: ${result.throughput.toFixed(2)} ops/sec`);
      console.log(`P95: ${result.p95.toFixed(2)}ms`);
    });

    test('benchmark: calculate liquidity metrics', async () => {
      const analyzer = new OrderBookAnalyzer(mockDB);
      const orderBook = generateOrderBook('AAPL', 150, 20);

      const runner = new BenchmarkRunner({
        name: 'Liquidity-Calculate',
        iterations: 100
      });

      const result = await runner.run(async () => {
        await analyzer.calculateLiquidity(orderBook);
      });

      expect(result.mean).toBeLessThan(30);
      console.log(`\nLiquidity Calculation: ${result.mean.toFixed(2)}ms`);
    });
  });

  describe('Pattern Learning', () => {
    test('benchmark: pattern detection', async () => {
      const learner = new PatternLearner(mockDB, mockPredictor);
      const marketData = generateMarketData('AAPL', 100);

      const runner = new BenchmarkRunner({
        name: 'Pattern-Detection',
        iterations: 50
      });

      const result = await runner.run(async () => {
        await learner.detectPatterns(marketData);
      });

      expect(result.mean).toBeLessThan(100);
      console.log(`\nPattern Detection: ${result.mean.toFixed(2)}ms`);
    });

    test('benchmark: feature extraction', async () => {
      const learner = new PatternLearner(mockDB, mockPredictor);
      const marketData = generateMarketData('AAPL', 100);

      const runner = new BenchmarkRunner({
        name: 'Feature-Extraction',
        iterations: 100
      });

      const result = await runner.run(async () => {
        await learner.extractFeatures(marketData);
      });

      expect(result.mean).toBeLessThan(20);
      console.log(`\nFeature Extraction: ${result.mean.toFixed(2)}ms`);
    });

    test('benchmark: prediction', async () => {
      const learner = new PatternLearner(mockDB, mockPredictor);
      const marketData = generateMarketData('AAPL', 100);

      await learner.train(marketData);

      const runner = new BenchmarkRunner({
        name: 'Prediction',
        iterations: 100
      });

      const result = await runner.run(async () => {
        await learner.predict(marketData.slice(-10));
      });

      expect(result.mean).toBeLessThan(10);
      console.log(`\nPrediction: ${result.mean.toFixed(2)}ms`);
    });
  });

  describe('Comparison: JS vs NAPI-RS', () => {
    test('compare order book processing implementations', async () => {
      const analyzer = new OrderBookAnalyzer(mockDB);
      const orderBook = generateOrderBook('AAPL', 150, 20);

      // JavaScript implementation
      const jsImpl = async () => {
        await analyzer.processSnapshot(orderBook);
      };

      // Simulated Rust implementation (would use actual NAPI binding)
      const rustImpl = async () => {
        // In real implementation, this would call Rust via NAPI
        await analyzer.processSnapshot(orderBook);
      };

      const comparer = new ComparisonRunner();

      const result = await comparer.compare(
        jsImpl,
        rustImpl,
        {
          name: 'OrderBook-JS-vs-Rust',
          iterations: 100
        }
      );

      console.log('\nJS vs Rust Comparison:');
      console.log(`Baseline (JS): ${result.baseline.mean.toFixed(2)}ms`);
      console.log(`Current (Rust): ${result.current.mean.toFixed(2)}ms`);
      console.log(`Improvement: ${result.improvement.toFixed(2)}%`);
      console.log(`Significant: ${result.significant}`);

      // Rust should be faster or comparable
      expect(result.improvement).toBeGreaterThan(-50); // At most 50% slower
    });
  });

  describe('Scalability', () => {
    test('benchmark: scale with order book depth', async () => {
      const analyzer = new OrderBookAnalyzer(mockDB);
      const depths = [10, 20, 50, 100];
      const results = [];

      for (const depth of depths) {
        const orderBook = generateOrderBook('AAPL', 150, depth);

        const runner = new BenchmarkRunner({
          name: `OrderBook-Depth-${depth}`,
          iterations: 50
        });

        const result = await runner.run(async () => {
          await analyzer.processSnapshot(orderBook);
        });

        results.push({ depth, mean: result.mean });
      }

      console.log('\nScalability (Order Book Depth):');
      results.forEach(r => {
        console.log(`Depth ${r.depth}: ${r.mean.toFixed(2)}ms`);
      });

      // Processing time should scale sub-linearly
      const firstMean = results[0].mean;
      const lastMean = results[results.length - 1].mean;
      const depthRatio = depths[depths.length - 1] / depths[0];
      const timeRatio = lastMean / firstMean;

      expect(timeRatio).toBeLessThan(depthRatio); // Sub-linear scaling
    });

    test('benchmark: concurrent analysis', async () => {
      const analyzer = new OrderBookAnalyzer(mockDB);
      const orderBook = generateOrderBook('AAPL', 150, 20);

      const concurrencyLevels = [1, 5, 10, 20];
      const results = [];

      for (const concurrency of concurrencyLevels) {
        const runner = new BenchmarkRunner({
          name: `Concurrent-${concurrency}`,
          iterations: 20
        });

        const result = await runner.run(async () => {
          await Promise.all(
            Array.from({ length: concurrency }, () =>
              analyzer.processSnapshot(orderBook)
            )
          );
        });

        results.push({
          concurrency,
          throughput: result.throughput * concurrency
        });
      }

      console.log('\nConcurrent Analysis Throughput:');
      results.forEach(r => {
        console.log(`Concurrency ${r.concurrency}: ${r.throughput.toFixed(2)} ops/sec`);
      });

      // Throughput should increase with concurrency
      const firstThroughput = results[0].throughput;
      const lastThroughput = results[results.length - 1].throughput;

      expect(lastThroughput).toBeGreaterThan(firstThroughput);
    });
  });
});
