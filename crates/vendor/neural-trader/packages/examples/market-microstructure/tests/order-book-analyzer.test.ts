/**
 * Tests for OrderBookAnalyzer
 */

import {
  OrderBookAnalyzer,
  OrderBook,
  OrderBookLevel,
  OrderFlow,
  MicrostructureMetrics
} from '../src/order-book-analyzer';

describe('OrderBookAnalyzer', () => {
  let analyzer: OrderBookAnalyzer;

  beforeEach(() => {
    analyzer = new OrderBookAnalyzer();
  });

  afterEach(() => {
    analyzer.reset();
  });

  const createMockOrderBook = (
    bidPrice: number = 100,
    askPrice: number = 100.1,
    depth: number = 1000
  ): OrderBook => ({
    bids: [
      { price: bidPrice, size: depth, orders: 10 },
      { price: bidPrice - 0.1, size: depth * 0.8, orders: 8 },
      { price: bidPrice - 0.2, size: depth * 0.6, orders: 6 }
    ],
    asks: [
      { price: askPrice, size: depth, orders: 10 },
      { price: askPrice + 0.1, size: depth * 0.8, orders: 8 },
      { price: askPrice + 0.2, size: depth * 0.6, orders: 6 }
    ],
    timestamp: Date.now(),
    symbol: 'BTCUSD'
  });

  const createMockTrades = (count: number = 10): OrderFlow[] => {
    const trades: OrderFlow[] = [];
    const basePrice = 100;

    for (let i = 0; i < count; i++) {
      trades.push({
        type: i % 2 === 0 ? 'buy' : 'sell',
        price: basePrice + (Math.random() - 0.5) * 0.5,
        size: Math.random() * 100 + 50,
        aggressor: i % 2 === 0 ? 'buyer' : 'seller',
        timestamp: Date.now() + i * 1000
      });
    }

    return trades;
  };

  describe('Basic Metrics', () => {
    test('should calculate bid-ask spread', () => {
      const orderBook = createMockOrderBook(100, 100.1);
      const metrics = analyzer.analyzeOrderBook(orderBook);

      expect(metrics.bidAskSpread).toBe(0.1);
      expect(metrics.bidAskSpread).toBeGreaterThan(0);
    });

    test('should calculate spread in basis points', () => {
      const orderBook = createMockOrderBook(100, 100.1);
      const metrics = analyzer.analyzeOrderBook(orderBook);

      // 0.1 / 100.05 * 10000 ≈ 10 bps
      expect(metrics.spreadBps).toBeCloseTo(10, 0);
    });

    test('should calculate mid price', () => {
      const orderBook = createMockOrderBook(100, 100.1);
      const metrics = analyzer.analyzeOrderBook(orderBook);

      expect(metrics.midPrice).toBe(100.05);
    });

    test('should calculate micro price', () => {
      const orderBook = createMockOrderBook(100, 100.1, 1000);
      const metrics = analyzer.analyzeOrderBook(orderBook);

      // Volume-weighted mid price
      expect(metrics.microPrice).toBeGreaterThan(0);
      expect(metrics.microPrice).toBeCloseTo(100.05, 1);
    });
  });

  describe('Depth Metrics', () => {
    test('should calculate bid and ask depth', () => {
      const orderBook = createMockOrderBook(100, 100.1, 1000);
      const metrics = analyzer.analyzeOrderBook(orderBook);

      expect(metrics.bidDepth).toBeGreaterThan(0);
      expect(metrics.askDepth).toBeGreaterThan(0);
      expect(metrics.bidDepth).toBeCloseTo(2400, 0); // 1000 + 800 + 600
      expect(metrics.askDepth).toBeCloseTo(2400, 0);
    });

    test('should calculate imbalance', () => {
      const orderBook = createMockOrderBook(100, 100.1, 1000);
      const metrics = analyzer.analyzeOrderBook(orderBook);

      // Balanced book
      expect(metrics.imbalance).toBeCloseTo(0, 1);
    });

    test('should detect buy-side imbalance', () => {
      const orderBook = createMockOrderBook(100, 100.1, 2000);
      orderBook.asks[0].size = 500;
      orderBook.asks[1].size = 400;
      orderBook.asks[2].size = 300;

      const metrics = analyzer.analyzeOrderBook(orderBook);

      // More bid depth than ask depth
      expect(metrics.imbalance).toBeGreaterThan(0);
    });

    test('should detect sell-side imbalance', () => {
      const orderBook = createMockOrderBook(100, 100.1, 500);
      orderBook.bids[0].size = 100;
      orderBook.bids[1].size = 80;
      orderBook.bids[2].size = 60;

      const metrics = analyzer.analyzeOrderBook(orderBook);

      // More ask depth than bid depth
      expect(metrics.imbalance).toBeLessThan(0);
    });
  });

  describe('Order Flow Metrics', () => {
    test('should track order flow history', () => {
      const orderBook = createMockOrderBook();
      const trades = createMockTrades(10);

      analyzer.analyzeOrderBook(orderBook, trades);

      const history = analyzer.getOrderFlowHistory();
      expect(history.length).toBe(10);
    });

    test('should calculate buy pressure', () => {
      const orderBook = createMockOrderBook();
      const trades: OrderFlow[] = [];

      // Create more buy trades
      for (let i = 0; i < 15; i++) {
        trades.push({
          type: 'buy',
          price: 100 + Math.random(),
          size: 100,
          aggressor: 'buyer',
          timestamp: Date.now() + i * 1000
        });
      }

      // Add some sell trades
      for (let i = 0; i < 5; i++) {
        trades.push({
          type: 'sell',
          price: 100 + Math.random(),
          size: 100,
          aggressor: 'seller',
          timestamp: Date.now() + (15 + i) * 1000
        });
      }

      analyzer.analyzeOrderBook(orderBook, trades);
      const metrics = analyzer.analyzeOrderBook(orderBook);

      expect(metrics.buyPressure).toBeGreaterThan(0.5);
      expect(metrics.sellPressure).toBeLessThan(0.5);
      expect(metrics.netFlow).toBeGreaterThan(0);
    });

    test('should calculate sell pressure', () => {
      const orderBook = createMockOrderBook();
      const trades: OrderFlow[] = [];

      // Create more sell trades
      for (let i = 0; i < 15; i++) {
        trades.push({
          type: 'sell',
          price: 100 + Math.random(),
          size: 100,
          aggressor: 'seller',
          timestamp: Date.now() + i * 1000
        });
      }

      // Add some buy trades
      for (let i = 0; i < 5; i++) {
        trades.push({
          type: 'buy',
          price: 100 + Math.random(),
          size: 100,
          aggressor: 'buyer',
          timestamp: Date.now() + (15 + i) * 1000
        });
      }

      analyzer.analyzeOrderBook(orderBook, trades);
      const metrics = analyzer.analyzeOrderBook(orderBook);

      expect(metrics.sellPressure).toBeGreaterThan(0.5);
      expect(metrics.buyPressure).toBeLessThan(0.5);
      expect(metrics.netFlow).toBeLessThan(0);
    });
  });

  describe('Toxicity Metrics', () => {
    test('should calculate VPIN', () => {
      const orderBook = createMockOrderBook();
      const trades = createMockTrades(50);

      analyzer.analyzeOrderBook(orderBook, trades);
      const metrics = analyzer.analyzeOrderBook(orderBook);

      expect(metrics.vpin).toBeGreaterThanOrEqual(0);
      expect(metrics.vpin).toBeLessThanOrEqual(1);
    });

    test('should calculate order flow toxicity', () => {
      const orderBook = createMockOrderBook();
      const trades = createMockTrades(20);

      analyzer.analyzeOrderBook(orderBook, trades);
      const metrics = analyzer.analyzeOrderBook(orderBook);

      expect(metrics.orderFlowToxicity).toBeGreaterThanOrEqual(0);
    });

    test('should calculate adverse selection', () => {
      const orderBook = createMockOrderBook();

      // Analyze multiple times to build history
      for (let i = 0; i < 15; i++) {
        const ob = createMockOrderBook(100, 100.1 + i * 0.01);
        analyzer.analyzeOrderBook(ob);
      }

      const metrics = analyzer.analyzeOrderBook(orderBook);

      expect(metrics.adverseSelection).toBeGreaterThanOrEqual(0);
    });
  });

  describe('Liquidity Metrics', () => {
    test('should calculate liquidity score', () => {
      const orderBook = createMockOrderBook(100, 100.1, 1000);
      const metrics = analyzer.analyzeOrderBook(orderBook);

      expect(metrics.liquidityScore).toBeGreaterThan(0);
      expect(metrics.liquidityScore).toBeLessThanOrEqual(1);
    });

    test('should rate tight spread as more liquid', () => {
      const tightBook = createMockOrderBook(100, 100.05, 1000);
      const wideBook = createMockOrderBook(100, 100.5, 1000);

      const tightMetrics = analyzer.analyzeOrderBook(tightBook);
      analyzer.reset();
      const wideMetrics = analyzer.analyzeOrderBook(wideBook);

      expect(tightMetrics.liquidityScore).toBeGreaterThan(wideMetrics.liquidityScore);
    });

    test('should calculate price impact', () => {
      const orderBook = createMockOrderBook(100, 100.1, 1000);
      const metrics = analyzer.analyzeOrderBook(orderBook);

      expect(metrics.priceImpact).toBeGreaterThanOrEqual(0);
    });

    test('should estimate resilience time', () => {
      const orderBook = createMockOrderBook();

      // Build history
      for (let i = 0; i < 25; i++) {
        const ob = createMockOrderBook(100, 100.1 + Math.random() * 0.1);
        analyzer.analyzeOrderBook(ob);
      }

      const metrics = analyzer.analyzeOrderBook(orderBook);

      expect(metrics.resilienceTime).toBeGreaterThanOrEqual(0);
    });
  });

  describe('Metrics History', () => {
    test('should maintain metrics history', () => {
      const orderBook = createMockOrderBook();

      for (let i = 0; i < 10; i++) {
        analyzer.analyzeOrderBook(orderBook);
      }

      const history = analyzer.getMetricsHistory();
      expect(history.length).toBe(10);
    });

    test('should limit history length', () => {
      const shortAnalyzer = new OrderBookAnalyzer(100);
      const orderBook = createMockOrderBook();

      for (let i = 0; i < 150; i++) {
        shortAnalyzer.analyzeOrderBook(orderBook);
      }

      const history = shortAnalyzer.getMetricsHistory();
      expect(history.length).toBe(100);
    });

    test('should reset clear all history', () => {
      const orderBook = createMockOrderBook();
      const trades = createMockTrades(10);

      analyzer.analyzeOrderBook(orderBook, trades);
      analyzer.reset();

      expect(analyzer.getMetricsHistory().length).toBe(0);
      expect(analyzer.getOrderFlowHistory().length).toBe(0);
    });
  });

  describe('Edge Cases', () => {
    test('should handle empty order book', () => {
      const emptyBook: OrderBook = {
        bids: [],
        asks: [],
        timestamp: Date.now(),
        symbol: 'BTCUSD'
      };

      const metrics = analyzer.analyzeOrderBook(emptyBook);

      expect(metrics.bidAskSpread).toBe(Infinity);
      expect(metrics.midPrice).toBe(0);
      expect(metrics.imbalance).toBe(0);
    });

    test('should handle missing trades', () => {
      const orderBook = createMockOrderBook();
      const metrics = analyzer.analyzeOrderBook(orderBook);

      expect(metrics.buyPressure).toBe(0);
      expect(metrics.sellPressure).toBe(0);
      expect(metrics.netFlow).toBe(0);
    });

    test('should handle insufficient history for some metrics', () => {
      const orderBook = createMockOrderBook();
      const metrics = analyzer.analyzeOrderBook(orderBook);

      // Should return 0 for metrics requiring history
      expect(metrics.vpin).toBe(0);
      expect(metrics.orderFlowToxicity).toBe(0);
      expect(metrics.adverseSelection).toBe(0);
    });
  });
});
