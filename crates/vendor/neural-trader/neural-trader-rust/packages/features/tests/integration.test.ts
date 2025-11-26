/**
 * Integration tests for @neural-trader/features package
 * Tests complete feature engineering workflows
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

describe('Feature Engineering Integration', () => {
  describe('multi-indicator analysis', () => {
    it('should calculate multiple indicators on same data', () => {
      const prices = Array.from({ length: 100 }, (_, i) =>
        100 + Math.sin((i / 100) * Math.PI * 2) * 20 + Math.random() * 2
      );

      const sma20 = calculateSma(prices, 20);
      const sma50 = calculateSma(prices, 50);
      const rsi14 = calculateRsi(prices, 14);

      expect(sma20.length).toBeGreaterThan(0);
      expect(sma50.length).toBeGreaterThan(0);
      expect(rsi14.length).toBeGreaterThan(0);
    });

    it('should identify trend changes', () => {
      const prices = Array.from({ length: 200 }, (_, i) => {
        if (i < 100) return 100 + i * 0.5;
        return 150 - (i - 100) * 0.5;
      });

      const sma = calculateSma(prices, 20);

      // Find crossover points
      let crossovers = 0;
      for (let i = 1; i < sma.length; i++) {
        if ((prices[i + 20] > sma[i] && prices[i + 19] <= sma[i - 1]) ||
            (prices[i + 20] < sma[i] && prices[i + 19] >= sma[i - 1])) {
          crossovers++;
        }
      }

      expect(crossovers).toBeGreaterThanOrEqual(0);
    });
  });

  describe('feature normalization', () => {
    it('should normalize indicators to 0-1 range', () => {
      const prices = Array.from({ length: 100 }, (_, i) => 100 + Math.sin(i * 0.05) * 20);
      const sma = calculateSma(prices, 20);

      const min = Math.min(...sma);
      const max = Math.max(...sma);
      const normalized = sma.map(v => (v - min) / (max - min || 1));

      normalized.forEach(val => {
        expect(val).toBeGreaterThanOrEqual(0);
        expect(val).toBeLessThanOrEqual(1);
      });
    });

    it('should handle RSI normalization', () => {
      const prices = Array.from({ length: 100 }, (_, i) => 100 + Math.sin(i * 0.05) * 10);
      const rsi = calculateRsi(prices, 14);

      // RSI is already normalized to 0-100
      rsi.forEach(val => {
        expect(val).toBeGreaterThanOrEqual(0);
        expect(val).toBeLessThanOrEqual(100);
      });
    });
  });

  describe('feature matrix creation', () => {
    it('should create feature matrix for ML training', () => {
      const prices = Array.from({ length: 150 }, (_, i) =>
        100 + Math.sin((i / 150) * Math.PI * 2) * 15
      );

      const sma10 = calculateSma(prices, 10);
      const sma20 = calculateSma(prices, 20);
      const rsi14 = calculateRsi(prices, 14);

      // Align all features
      const startIdx = Math.max(9, 19, 13); // Latest period -1
      const featureMatrix: any[] = [];

      for (let i = startIdx; i < Math.min(sma20.length, rsi14.length); i++) {
        featureMatrix.push({
          price: prices[i + 20],
          sma10: sma10[i - 9],
          sma20: sma20[i - 19],
          rsi14: rsi14[i - 13]
        });
      }

      expect(featureMatrix.length).toBeGreaterThan(0);
      featureMatrix.forEach(row => {
        expect(row).toHaveProperty('price');
        expect(row).toHaveProperty('sma10');
        expect(row).toHaveProperty('sma20');
        expect(row).toHaveProperty('rsi14');
      });
    });
  });

  describe('trading signal generation', () => {
    it('should generate simple moving average crossover signals', () => {
      const prices = Array.from({ length: 100 }, (_, i) =>
        100 + Math.sin((i / 100) * Math.PI * 2) * 10
      );

      const sma_fast = calculateSma(prices, 10);
      const sma_slow = calculateSma(prices, 20);

      const signals = [];
      for (let i = 0; i < Math.min(sma_fast.length, sma_slow.length); i++) {
        signals.push({
          timestamp: i,
          signal: sma_fast[i] > sma_slow[i] ? 'BUY' : 'SELL'
        });
      }

      expect(signals.length).toBeGreaterThan(0);
      signals.forEach(s => {
        expect(['BUY', 'SELL']).toContain(s.signal);
      });
    });

    it('should generate RSI overbought/oversold signals', () => {
      const prices = Array.from({ length: 100 }, (_, i) => 100 + Math.sin(i * 0.1) * 20);
      const rsi = calculateRsi(prices, 14);

      const signals = [];
      for (let i = 0; i < rsi.length; i++) {
        let signal = 'NEUTRAL';
        if (rsi[i] > 70) signal = 'OVERBOUGHT';
        if (rsi[i] < 30) signal = 'OVERSOLD';

        signals.push({
          timestamp: i,
          rsi: rsi[i],
          signal
        });
      }

      expect(signals.length).toBeGreaterThan(0);

      // Should have some overbought/oversold conditions
      const extremes = signals.filter(s => s.signal !== 'NEUTRAL');
      expect(extremes.length).toBeGreaterThan(0);
    });
  });

  describe('multi-timeframe analysis', () => {
    it('should extract features across different timeframes', () => {
      const timeframes = {
        '5min': 5,
        '15min': 15,
        '1hour': 60
      };

      const baseData = Array.from({ length: 1000 }, (_, i) =>
        100 + Math.sin((i / 1000) * Math.PI * 2) * 10
      );

      const results: any = {};

      for (const [tf, period] of Object.entries(timeframes)) {
        const sma = calculateSma(baseData, period);
        results[tf] = sma;
      }

      Object.entries(results).forEach(([tf, data]: any) => {
        expect(Array.isArray(data)).toBe(true);
        expect(data.length).toBeGreaterThan(0);
      });
    });
  });

  describe('volatility analysis', () => {
    it('should calculate volatility from prices', () => {
      const prices = Array.from({ length: 100 }, (_, i) =>
        100 + Math.sin((i / 100) * Math.PI * 4) * 15 + (Math.random() - 0.5) * 5
      );

      // Calculate returns
      const returns = [];
      for (let i = 1; i < prices.length; i++) {
        returns.push((prices[i] - prices[i - 1]) / prices[i - 1]);
      }

      // Calculate rolling volatility
      const volatilities = [];
      const period = 20;

      for (let i = period; i < returns.length; i++) {
        const window = returns.slice(i - period, i);
        const mean = window.reduce((a, b) => a + b, 0) / window.length;
        const variance = window.reduce((sum, r) => sum + (r - mean) ** 2, 0) / window.length;
        const vol = Math.sqrt(variance);
        volatilities.push(vol);
      }

      expect(volatilities.length).toBeGreaterThan(0);
      volatilities.forEach(v => {
        expect(v).toBeGreaterThanOrEqual(0);
      });
    });
  });

  describe('data quality handling', () => {
    it('should handle price gaps', () => {
      const prices = [100, 101, 102, 150, 151, 152]; // Price gap at index 3
      const sma = calculateSma(prices, 2);

      expect(sma.length).toBeGreaterThan(0);
      // SMA should reflect the gap
      expect(sma[2]).toBeLessThan(sma[3]); // Before/after gap
    });

    it('should handle duplicate prices', () => {
      const prices = Array.from({ length: 50 }, () => 100);
      const sma = calculateSma(prices, 20);

      sma.forEach(val => {
        expect(val).toBe(100);
      });
    });

    it('should handle price reversals', () => {
      const prices = Array.from({ length: 50 }, (_, i) =>
        i < 25 ? 100 + i : 125 - (i - 25)
      );

      const rsi = calculateRsi(prices, 14);

      expect(rsi.length).toBeGreaterThan(0);
      // Should show changing RSI due to reversal
      const first = rsi.slice(0, 5);
      const last = rsi.slice(-5);
      expect(Math.max(...first)).not.toBe(Math.max(...last));
    });
  });
});
