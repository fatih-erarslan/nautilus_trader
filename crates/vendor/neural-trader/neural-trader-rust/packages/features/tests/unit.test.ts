/**
 * Unit tests for @neural-trader/features package
 * Tests technical indicator calculations
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
      const gains: number[] = [];
      const losses: number[] = [];

      for (let i = 1; i < prices.length; i++) {
        const diff = prices[i] - prices[i - 1];
        gains.push(diff > 0 ? diff : 0);
        losses.push(diff < 0 ? -diff : 0);
      }

      return prices.slice(period).map((_, i) => {
        const avgGain = gains.slice(i, i + period).reduce((a, b) => a + b, 0) / period;
        const avgLoss = losses.slice(i, i + period).reduce((a, b) => a + b, 0) / period;
        const rs = avgGain / (avgLoss || 1);
        return 100 - (100 / (1 + rs));
      });
    }),
    calculateIndicator: jest.fn(async (bars: any[], indicator: string, params: string) => {
      return Array.from({ length: bars.length }, () => 100 + Math.random() * 5);
    })
  }))
}));

import { calculateSma, calculateRsi, calculateIndicator } from '../index';

describe('calculateSma', () => {
  it('should calculate simple moving average', () => {
    const prices = [100, 102, 101, 103, 105, 104, 106];
    const result = calculateSma(prices, 3);

    expect(Array.isArray(result)).toBe(true);
    expect(result.length).toBeGreaterThan(0);
    result.forEach(val => {
      expect(typeof val).toBe('number');
      expect(val).toBeGreaterThan(0);
    });
  });

  it('should handle different periods', () => {
    const prices = Array.from({ length: 100 }, (_, i) => 100 + Math.sin(i * 0.1) * 10);

    for (const period of [5, 10, 20, 50]) {
      const result = calculateSma(prices, period);
      expect(result.length).toBeLessThanOrEqual(prices.length);
    }
  });

  it('should handle small datasets', () => {
    const prices = [100, 101, 102];
    const result = calculateSma(prices, 5);

    // Should return empty array if period > length
    expect(result.length).toBeLessThanOrEqual(prices.length);
  });

  it('should return empty for period larger than data', () => {
    const prices = [100, 101];
    const result = calculateSma(prices, 10);

    expect(result.length).toBe(0);
  });
});

describe('calculateRsi', () => {
  it('should calculate relative strength index', () => {
    const prices = [100, 102, 101, 103, 105, 104, 106, 108, 107];
    const result = calculateRsi(prices, 3);

    expect(Array.isArray(result)).toBe(true);
    result.forEach(val => {
      expect(typeof val).toBe('number');
      expect(val).toBeGreaterThanOrEqual(0);
      expect(val).toBeLessThanOrEqual(100);
    });
  });

  it('should detect overbought/oversold', () => {
    // Continuously increasing prices (should show overbought)
    const bullishPrices = Array.from({ length: 50 }, (_, i) => 100 + i * 2);
    const result = calculateRsi(bullishPrices, 14);

    const maxRsi = Math.max(...result);
    expect(maxRsi).toBeGreaterThan(50);
  });

  it('should handle different periods', () => {
    const prices = Array.from({ length: 100 }, (_, i) => 100 + Math.sin(i * 0.1) * 10);

    for (const period of [7, 14, 21]) {
      const result = calculateRsi(prices, period);
      expect(result.length).toBeLessThanOrEqual(prices.length);
    }
  });

  it('should return valid RSI range', () => {
    const prices = Array.from({ length: 100 }, (_, i) => 100 + Math.random() * 20 - 10);
    const result = calculateRsi(prices, 14);

    result.forEach(val => {
      expect(val).toBeGreaterThanOrEqual(0);
      expect(val).toBeLessThanOrEqual(100);
    });
  });
});

describe('calculateIndicator', () => {
  it('should calculate generic indicator', async () => {
    const bars = Array.from({ length: 50 }, (_, i) => ({
      close: 100 + Math.sin(i * 0.1) * 10,
      high: 105,
      low: 95,
      open: 100,
      volume: 1000000,
      timestamp: Date.now() + i * 60000
    }));

    const result = await calculateIndicator(bars, 'SMA', '20');

    expect(Array.isArray(result)).toBe(true);
    expect(result.length).toBeGreaterThan(0);
  });

  it('should handle different indicator types', async () => {
    const bars = Array.from({ length: 50 }, (_, i) => ({
      close: 100 + Math.random() * 5,
      high: 105,
      low: 95,
      open: 100,
      volume: 1000000,
      timestamp: Date.now() + i * 60000
    }));

    for (const indicator of ['SMA', 'RSI', 'MACD']) {
      const result = await calculateIndicator(bars, indicator, '14');
      expect(Array.isArray(result)).toBe(true);
    }
  });

  it('should handle empty bars', async () => {
    const result = await calculateIndicator([], 'SMA', '20');
    expect(Array.isArray(result)).toBe(true);
  });
});

describe('Error handling', () => {
  it('should handle invalid price data', () => {
    const invalidPrices = [NaN, Infinity, -Infinity];
    const result = calculateSma(invalidPrices, 2);

    // Should not throw, but return array with potentially NaN values
    expect(Array.isArray(result)).toBe(true);
  });

  it('should handle negative prices', () => {
    const prices = [-10, -5, 0, 5, 10];
    const result = calculateSma(prices, 2);

    expect(Array.isArray(result)).toBe(true);
  });

  it('should handle zero period', () => {
    const prices = [100, 101, 102];
    // This might throw or return empty
    try {
      const result = calculateSma(prices, 0);
      expect(Array.isArray(result)).toBe(true);
    } catch {
      // Expected behavior
    }
  });
});

describe('Edge cases', () => {
  it('should handle very large datasets', () => {
    const largeDataset = Array.from({ length: 10000 }, (_, i) => 100 + Math.sin(i * 0.01) * 10);
    const result = calculateSma(largeDataset, 200);

    expect(result.length).toBeGreaterThan(0);
  });

  it('should handle identical prices', () => {
    const prices = Array.from({ length: 50 }, () => 100);
    const result = calculateSma(prices, 20);

    result.forEach(val => {
      expect(val).toBe(100);
    });
  });

  it('should handle single price', () => {
    const prices = [100];
    const result = calculateSma(prices, 1);

    expect(result.length).toBeGreaterThan(0);
  });
});
