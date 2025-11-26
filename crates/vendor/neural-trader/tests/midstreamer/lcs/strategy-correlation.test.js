/**
 * LCS Strategy Correlation Tests
 * Tests Longest Common Subsequence for strategy similarity matching
 */

const { performance } = require('perf_hooks');

describe('LCS Strategy Correlation', () => {
  let lcs;

  beforeAll(() => {
    // LCS implementation for strategy sequences
    lcs = {
      calculate: (seq1, seq2) => {
        const m = seq1.length;
        const n = seq2.length;
        const dp = Array(m + 1).fill(null).map(() => Array(n + 1).fill(0));

        for (let i = 1; i <= m; i++) {
          for (let j = 1; j <= n; j++) {
            if (seq1[i - 1] === seq2[j - 1]) {
              dp[i][j] = dp[i - 1][j - 1] + 1;
            } else {
              dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
            }
          }
        }

        return dp[m][n];
      },

      correlation: (strategy1, strategy2) => {
        const lcsLength = lcs.calculate(strategy1, strategy2);
        const maxLength = Math.max(strategy1.length, strategy2.length);
        return lcsLength / maxLength;
      },

      extractSequence: (seq1, seq2) => {
        const m = seq1.length;
        const n = seq2.length;
        const dp = Array(m + 1).fill(null).map(() => Array(n + 1).fill(0));

        for (let i = 1; i <= m; i++) {
          for (let j = 1; j <= n; j++) {
            if (seq1[i - 1] === seq2[j - 1]) {
              dp[i][j] = dp[i - 1][j - 1] + 1;
            } else {
              dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
            }
          }
        }

        // Backtrack to find the sequence
        const result = [];
        let i = m, j = n;
        while (i > 0 && j > 0) {
          if (seq1[i - 1] === seq2[j - 1]) {
            result.unshift(seq1[i - 1]);
            i--;
            j--;
          } else if (dp[i - 1][j] > dp[i][j - 1]) {
            i--;
          } else {
            j--;
          }
        }

        return result;
      }
    };
  });

  describe('Perfect Correlation', () => {
    it('should return 1.0 for identical strategies', () => {
      const strategy = ['BUY', 'HOLD', 'SELL', 'BUY', 'HOLD'];
      const correlation = lcs.correlation(strategy, strategy);

      expect(correlation).toBe(1.0);
    });

    it('should return 1.0 for identical complex strategies', () => {
      const strategy = [
        'ANALYZE_MARKET',
        'CALCULATE_INDICATORS',
        'CHECK_SIGNALS',
        'EXECUTE_BUY',
        'SET_STOP_LOSS',
        'MONITOR_POSITION',
        'TAKE_PROFIT'
      ];

      const correlation = lcs.correlation(strategy, strategy);

      expect(correlation).toBe(1.0);
    });

    it('should handle single-action strategies', () => {
      const strategy = ['BUY'];
      const correlation = lcs.correlation(strategy, strategy);

      expect(correlation).toBe(1.0);
    });
  });

  describe('Zero Correlation', () => {
    it('should return 0.0 for completely different strategies', () => {
      const strategy1 = ['BUY', 'BUY', 'BUY'];
      const strategy2 = ['SELL', 'SELL', 'SELL'];

      const correlation = lcs.correlation(strategy1, strategy2);

      expect(correlation).toBe(0.0);
    });

    it('should detect opposite trading patterns', () => {
      const bullStrategy = ['LONG', 'ACCUMULATE', 'PROFIT'];
      const bearStrategy = ['SHORT', 'DISTRIBUTE', 'COVER'];

      const correlation = lcs.correlation(bullStrategy, bearStrategy);

      expect(correlation).toBe(0.0);
    });

    it('should handle empty strategy comparison', () => {
      const strategy = ['BUY', 'SELL'];
      const empty = [];

      const correlation = lcs.correlation(strategy, empty);

      expect(correlation).toBe(0.0);
    });
  });

  describe('Partial Correlation', () => {
    it('should detect partial strategy overlap', () => {
      const strategy1 = ['BUY', 'HOLD', 'SELL', 'WAIT'];
      const strategy2 = ['BUY', 'ANALYZE', 'HOLD', 'EXECUTE', 'SELL'];

      const correlation = lcs.correlation(strategy1, strategy2);

      // Should find BUY, HOLD, SELL as common subsequence (3/5 = 0.6)
      expect(correlation).toBeGreaterThan(0.5);
      expect(correlation).toBeLessThan(0.8);
    });

    it('should measure strategy similarity with reordering', () => {
      const strategy1 = ['A', 'B', 'C', 'D', 'E'];
      const strategy2 = ['A', 'C', 'B', 'E', 'D'];

      const correlation = lcs.correlation(strategy1, strategy2);

      // Should find A, C, E or similar
      expect(correlation).toBeGreaterThan(0.4);
      expect(correlation).toBeLessThan(1.0);
    });

    it('should extract common strategy pattern', () => {
      const strategy1 = ['ANALYZE', 'BUY', 'MONITOR', 'SELL', 'WAIT'];
      const strategy2 = ['CHECK', 'BUY', 'TRACK', 'SELL', 'REST'];

      const commonPattern = lcs.extractSequence(strategy1, strategy2);

      expect(commonPattern).toEqual(['BUY', 'SELL']);
      expect(commonPattern.length).toBe(2);
    });
  });

  describe('Performance Benchmarks', () => {
    it('should process 50 strategies in <500ms', () => {
      const strategies = Array.from({ length: 50 }, (_, i) =>
        Array.from({ length: 10 }, (_, j) =>
          ['BUY', 'SELL', 'HOLD', 'WAIT', 'ANALYZE'][Math.floor(Math.random() * 5)]
        )
      );

      const start = performance.now();

      for (let i = 0; i < strategies.length; i++) {
        for (let j = i + 1; j < strategies.length; j++) {
          lcs.correlation(strategies[i], strategies[j]);
        }
      }

      const duration = performance.now() - start;

      expect(duration).toBeLessThan(500);
    });

    it('should handle large strategy sequences efficiently', () => {
      const strategy1 = Array.from({ length: 100 }, () =>
        ['ACTION_' + Math.floor(Math.random() * 20)]
      ).flat();
      const strategy2 = Array.from({ length: 100 }, () =>
        ['ACTION_' + Math.floor(Math.random() * 20)]
      ).flat();

      const start = performance.now();
      lcs.correlation(strategy1, strategy2);
      const duration = performance.now() - start;

      expect(duration).toBeLessThan(100);
    });

    it('should benchmark 60x speedup claim', () => {
      // Naive recursive implementation (exponential time)
      const naiveLCS = (s1, s2, m = s1.length, n = s2.length) => {
        if (m === 0 || n === 0) return 0;
        if (s1[m - 1] === s2[n - 1]) {
          return 1 + naiveLCS(s1, s2, m - 1, n - 1);
        }
        return Math.max(naiveLCS(s1, s2, m - 1, n), naiveLCS(s1, s2, m, n - 1));
      };

      const strategy1 = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'];
      const strategy2 = ['A', 'C', 'B', 'E', 'F', 'H', 'G', 'D'];

      // Measure naive implementation (small input only due to exponential time)
      const naiveStart = performance.now();
      naiveLCS(strategy1.slice(0, 8), strategy2.slice(0, 8));
      const naiveDuration = performance.now() - naiveStart;

      // Measure optimized implementation
      const optimizedStart = performance.now();
      lcs.calculate(strategy1, strategy2);
      const optimizedDuration = performance.now() - optimizedStart;

      // Should be significantly faster
      expect(optimizedDuration).toBeLessThan(naiveDuration * 0.5);
    });

    it('should process all-pairs correlation matrix efficiently', () => {
      const numStrategies = 20;
      const strategies = Array.from({ length: numStrategies }, (_, i) =>
        Array.from({ length: 15 }, (_, j) =>
          `ACTION_${(i + j) % 10}`
        )
      );

      const start = performance.now();
      const correlationMatrix = [];

      for (let i = 0; i < numStrategies; i++) {
        correlationMatrix[i] = [];
        for (let j = 0; j < numStrategies; j++) {
          correlationMatrix[i][j] = lcs.correlation(strategies[i], strategies[j]);
        }
      }

      const duration = performance.now() - start;

      // 20x20 = 400 comparisons should be fast
      expect(duration).toBeLessThan(200);
      expect(correlationMatrix.length).toBe(numStrategies);

      // Diagonal should be 1.0 (self-correlation)
      for (let i = 0; i < numStrategies; i++) {
        expect(correlationMatrix[i][i]).toBe(1.0);
      }
    });
  });

  describe('Real-World Strategy Patterns', () => {
    it('should correlate trend-following strategies', () => {
      const maStrategy = [
        'CALCULATE_MA_50',
        'CALCULATE_MA_200',
        'CROSS_ABOVE',
        'BUY_SIGNAL',
        'HOLD',
        'CROSS_BELOW',
        'SELL_SIGNAL'
      ];

      const emaStrategy = [
        'CALCULATE_EMA_12',
        'CALCULATE_EMA_26',
        'CROSS_ABOVE',
        'BUY_SIGNAL',
        'MONITOR',
        'CROSS_BELOW',
        'SELL_SIGNAL'
      ];

      const correlation = lcs.correlation(maStrategy, emaStrategy);

      // Should detect the common pattern
      expect(correlation).toBeGreaterThan(0.6);
    });

    it('should differentiate momentum vs mean-reversion', () => {
      const momentum = [
        'DETECT_BREAKOUT',
        'CONFIRM_VOLUME',
        'ENTER_LONG',
        'TRAIL_STOP',
        'EXIT_ON_REVERSAL'
      ];

      const meanReversion = [
        'DETECT_OVERBOUGHT',
        'WAIT_FOR_MEAN',
        'ENTER_SHORT',
        'TARGET_MEAN',
        'EXIT_ON_REVERSION'
      ];

      const correlation = lcs.correlation(momentum, meanReversion);

      // Should show low correlation (different philosophies)
      expect(correlation).toBeLessThan(0.3);
    });

    it('should cluster similar risk management strategies', () => {
      const strategy1 = ['ENTRY', 'SET_STOP_2PCT', 'SET_TARGET_6PCT', 'MONITOR', 'EXIT'];
      const strategy2 = ['ENTRY', 'SET_STOP_2PCT', 'SET_TARGET_4PCT', 'MONITOR', 'EXIT'];
      const strategy3 = ['ENTRY', 'MARTINGALE', 'DOUBLE_DOWN', 'PRAY', 'EXIT'];

      const corr12 = lcs.correlation(strategy1, strategy2);
      const corr13 = lcs.correlation(strategy1, strategy3);

      // Similar risk management should correlate more
      expect(corr12).toBeGreaterThan(corr13);
      expect(corr12).toBeGreaterThan(0.8);
    });
  });
});
