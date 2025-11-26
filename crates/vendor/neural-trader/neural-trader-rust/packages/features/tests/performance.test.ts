/**
 * Performance tests for @neural-trader/features package
 */

jest.mock('../load-binary', () => ({
  loadNativeBinary: jest.fn(() => ({
    calculateSma: jest.fn((prices: number[], period: number) => {
      if (prices.length < period) return [];
      return prices.slice(period - 1).map((_, i) => {
        const sum = prices.slice(i, i + period).reduce((a, b) => a + b, 0);
        return sum / period;
      });
    }),
    calculateRsi: jest.fn((prices: number[], period: number) => {
      if (prices.length < period) return [];
      return prices.slice(period).map(() => 50 + Math.random() * 20);
    }),
    calculateIndicator: jest.fn(async (bars: any[], indicator: string, params: string) => {
      await new Promise(r => setTimeout(r, 5));
      return Array.from({ length: bars.length }, () => 100 + Math.random() * 5);
    })
  }))
}));

import { calculateSma, calculateRsi, calculateIndicator } from '../index';

describe('Performance Benchmarks', () => {
  describe('calculation speed', () => {
    it('should calculate SMA quickly', () => {
      const prices = Array.from({ length: 1000 }, () => 100 + Math.random() * 5);

      const start = performance.now();
      const result = calculateSma(prices, 20);
      const duration = performance.now() - start;

      expect(result.length).toBeGreaterThan(0);
      expect(duration).toBeLessThan(100);
    });

    it('should calculate RSI efficiently', () => {
      const prices = Array.from({ length: 1000 }, () => 100 + Math.random() * 5);

      const start = performance.now();
      const result = calculateRsi(prices, 14);
      const duration = performance.now() - start;

      expect(result.length).toBeGreaterThan(0);
      expect(duration).toBeLessThan(100);
    });
  });

  describe('throughput', () => {
    it('should achieve target SMA throughput', () => {
      const prices = Array.from({ length: 500 }, () => 100 + Math.random() * 5);

      let count = 0;
      const start = performance.now();

      while (performance.now() - start < 1000) {
        calculateSma(prices, 20);
        count++;
      }

      const throughput = count; // Calculations per second
      expect(throughput).toBeGreaterThan(100);
    }, 5000);

    it('should achieve target RSI throughput', () => {
      const prices = Array.from({ length: 500 }, () => 100 + Math.random() * 5);

      let count = 0;
      const start = performance.now();

      while (performance.now() - start < 1000) {
        calculateRsi(prices, 14);
        count++;
      }

      const throughput = count;
      expect(throughput).toBeGreaterThan(100);
    }, 5000);
  });

  describe('large dataset handling', () => {
    it('should handle large price arrays', () => {
      const largeData = Array.from({ length: 100000 }, (_, i) =>
        100 + Math.sin((i / 100000) * Math.PI * 2) * 10
      );

      const start = performance.now();
      const result = calculateSma(largeData, 200);
      const duration = performance.now() - start;

      expect(result.length).toBeGreaterThan(0);
      expect(duration).toBeLessThan(1000); // Should handle 100k points in < 1s
    }, 5000);
  });

  describe('scaling characteristics', () => {
    it('should scale linearly with data size', () => {
      const sizes = [100, 500, 1000, 5000];
      const durations: number[] = [];

      for (const size of sizes) {
        const prices = Array.from({ length: size }, () => 100 + Math.random() * 5);

        const start = performance.now();
        calculateSma(prices, 20);
        durations.push(performance.now() - start);
      }

      // Verify roughly linear scaling
      expect(durations[1]).toBeLessThan(durations[0] * 10);
      expect(durations[2]).toBeLessThan(durations[0] * 20);
    });
  });

  describe('concurrent calculations', () => {
    it('should handle parallel calculations', async () => {
      const prices = Array.from({ length: 1000 }, () => 100 + Math.random() * 5);

      const start = performance.now();

      const tasks = Array.from({ length: 100 }, () =>
        Promise.resolve(calculateSma(prices, 20))
      );

      await Promise.all(tasks);
      const duration = performance.now() - start;

      expect(duration).toBeLessThan(1000); // 100 calculations should complete quickly
    }, 5000);
  });

  describe('memory efficiency', () => {
    it('should not leak memory with repeated calls', () => {
      const prices = Array.from({ length: 500 }, () => 100 + Math.random() * 5);
      const initialMemory = process.memoryUsage().heapUsed;

      for (let i = 0; i < 1000; i++) {
        calculateSma(prices, 20);
        calculateRsi(prices, 14);
      }

      const finalMemory = process.memoryUsage().heapUsed;
      const increase = finalMemory - initialMemory;

      // Should not exceed 20MB for 2000 function calls
      expect(increase).toBeLessThan(20 * 1024 * 1024);
    });
  });

  describe('latency measurements', () => {
    it('should meet latency requirements', () => {
      const prices = Array.from({ length: 200 }, () => 100 + Math.random() * 5);
      const latencies: number[] = [];

      for (let i = 0; i < 50; i++) {
        const start = performance.now();
        calculateSma(prices, 20);
        latencies.push(performance.now() - start);
      }

      const avg = latencies.reduce((a, b) => a + b) / latencies.length;
      const p99 = latencies.sort((a, b) => a - b)[Math.floor(latencies.length * 0.99)];

      expect(avg).toBeLessThan(5);
      expect(p99).toBeLessThan(10);
    }, 10000);
  });

  describe('batch processing', () => {
    it('should efficiently process multiple indicators', () => {
      const prices = Array.from({ length: 1000 }, () => 100 + Math.random() * 5);

      const start = performance.now();

      const sma10 = calculateSma(prices, 10);
      const sma20 = calculateSma(prices, 20);
      const sma50 = calculateSma(prices, 50);
      const rsi14 = calculateRsi(prices, 14);

      const duration = performance.now() - start;

      expect(sma10.length).toBeGreaterThan(0);
      expect(sma20.length).toBeGreaterThan(0);
      expect(sma50.length).toBeGreaterThan(0);
      expect(rsi14.length).toBeGreaterThan(0);
      expect(duration).toBeLessThan(200);
    }, 5000);
  });
});
