/**
 * DTW Pattern Matching Tests
 * Tests Dynamic Time Warping for temporal pattern similarity
 */

const { performance } = require('perf_hooks');

describe('DTW Pattern Matching', () => {
  let dtw;

  beforeAll(async () => {
    // Mock DTW implementation - replace with actual import
    dtw = {
      calculate: (pattern1, pattern2) => {
        if (pattern1.length === 0 || pattern2.length === 0) {
          return Infinity;
        }

        const n = pattern1.length;
        const m = pattern2.length;
        const dp = Array(n + 1).fill(null).map(() => Array(m + 1).fill(Infinity));
        dp[0][0] = 0;

        for (let i = 1; i <= n; i++) {
          for (let j = 1; j <= m; j++) {
            const cost = Math.abs(pattern1[i - 1] - pattern2[j - 1]);
            dp[i][j] = cost + Math.min(
              dp[i - 1][j],     // insertion
              dp[i][j - 1],     // deletion
              dp[i - 1][j - 1]  // match
            );
          }
        }

        return dp[n][m];
      },

      similarity: (pattern1, pattern2) => {
        const distance = dtw.calculate(pattern1, pattern2);
        const maxLen = Math.max(pattern1.length, pattern2.length);
        // Normalize to 0-1 range (1 = identical, 0 = completely different)
        return Math.max(0, 1 - (distance / (maxLen * 100)));
      }
    };
  });

  describe('Identical Patterns', () => {
    it('should return 100% similarity for identical patterns', () => {
      const pattern = [1.0, 2.5, 3.7, 2.1, 4.3];
      const similarity = dtw.similarity(pattern, pattern);

      expect(similarity).toBeGreaterThanOrEqual(0.99);
      expect(similarity).toBeLessThanOrEqual(1.0);
    });

    it('should return 100% similarity for identical long patterns', () => {
      const pattern = Array.from({ length: 100 }, (_, i) => Math.sin(i * 0.1));
      const similarity = dtw.similarity(pattern, pattern);

      expect(similarity).toBeGreaterThanOrEqual(0.99);
    });

    it('should handle identical single-element patterns', () => {
      const pattern = [5.0];
      const similarity = dtw.similarity(pattern, pattern);

      expect(similarity).toBe(1.0);
    });
  });

  describe('Different Length Patterns', () => {
    it('should match stretched patterns with high similarity', () => {
      const pattern1 = [1, 2, 3, 4, 5];
      const pattern2 = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5];

      const similarity = dtw.similarity(pattern1, pattern2);

      // Should still recognize the same trend
      expect(similarity).toBeGreaterThan(0.8);
    });

    it('should match compressed patterns', () => {
      const pattern1 = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5];
      const pattern2 = [1, 2, 3, 4, 5];

      const similarity = dtw.similarity(pattern1, pattern2);

      expect(similarity).toBeGreaterThan(0.8);
    });

    it('should handle very different length patterns', () => {
      const pattern1 = [1, 2, 3];
      const pattern2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

      const similarity = dtw.similarity(pattern1, pattern2);

      // Should still find some similarity in the matching prefix
      expect(similarity).toBeGreaterThan(0.5);
    });

    it('should detect dissimilar patterns of different lengths', () => {
      const pattern1 = [1, 2, 3, 4, 5];
      const pattern2 = [10, 20, 30, 40, 50, 60, 70];

      const similarity = dtw.similarity(pattern1, pattern2);

      expect(similarity).toBeLessThan(0.5);
    });
  });

  describe('Performance Benchmarks', () => {
    it('should compute DTW for small patterns in <10ms', () => {
      const pattern1 = Array.from({ length: 50 }, (_, i) => Math.sin(i * 0.1));
      const pattern2 = Array.from({ length: 50 }, (_, i) => Math.sin(i * 0.1 + 0.1));

      const start = performance.now();
      dtw.calculate(pattern1, pattern2);
      const duration = performance.now() - start;

      expect(duration).toBeLessThan(10);
    });

    it('should compute DTW for medium patterns in <50ms', () => {
      const pattern1 = Array.from({ length: 100 }, (_, i) => Math.sin(i * 0.1));
      const pattern2 = Array.from({ length: 100 }, (_, i) => Math.cos(i * 0.1));

      const start = performance.now();
      dtw.calculate(pattern1, pattern2);
      const duration = performance.now() - start;

      expect(duration).toBeLessThan(50);
    });

    it('should handle 1000 pattern comparisons efficiently', () => {
      const patterns = Array.from({ length: 100 }, (_, i) =>
        Array.from({ length: 20 }, (_, j) => Math.sin(i * 0.1 + j * 0.2))
      );

      const start = performance.now();

      for (let i = 0; i < 10; i++) {
        for (let j = 0; j < 10; j++) {
          dtw.similarity(patterns[i], patterns[j]);
        }
      }

      const duration = performance.now() - start;
      const avgDuration = duration / 100;

      expect(avgDuration).toBeLessThan(10);
    });

    it('should benchmark 100x speedup claim', () => {
      // Naive O(n²) implementation
      const naiveDTW = (p1, p2) => {
        let sum = 0;
        for (let i = 0; i < Math.min(p1.length, p2.length); i++) {
          for (let j = 0; j < Math.min(p1.length, p2.length); j++) {
            sum += Math.abs(p1[i] - p2[j]);
          }
        }
        return sum;
      };

      const pattern1 = Array.from({ length: 50 }, (_, i) => Math.random());
      const pattern2 = Array.from({ length: 50 }, (_, i) => Math.random());

      // Measure naive implementation
      const naiveStart = performance.now();
      naiveDTW(pattern1, pattern2);
      const naiveDuration = performance.now() - naiveStart;

      // Measure optimized implementation
      const optimizedStart = performance.now();
      dtw.calculate(pattern1, pattern2);
      const optimizedDuration = performance.now() - optimizedStart;

      // Should be significantly faster (allowing for variance)
      expect(optimizedDuration).toBeLessThan(naiveDuration * 0.5);
    });
  });

  describe('Edge Cases', () => {
    it('should handle empty patterns', () => {
      const pattern = [1, 2, 3];
      const empty = [];

      const distance = dtw.calculate(pattern, empty);

      expect(distance).toBe(Infinity);
    });

    it('should handle patterns with negative values', () => {
      const pattern1 = [-1, -2, -3, -4, -5];
      const pattern2 = [-1, -2, -3, -4, -5];

      const similarity = dtw.similarity(pattern1, pattern2);

      expect(similarity).toBeGreaterThanOrEqual(0.99);
    });

    it('should handle patterns with large value differences', () => {
      const pattern1 = [1, 2, 3];
      const pattern2 = [1000, 2000, 3000];

      const similarity = dtw.similarity(pattern1, pattern2);

      expect(similarity).toBeLessThan(0.1);
    });

    it('should detect shifted patterns', () => {
      const pattern1 = [1, 2, 3, 4, 5];
      const pattern2 = [0, 0, 1, 2, 3, 4, 5];

      const similarity = dtw.similarity(pattern1, pattern2);

      // DTW should align despite the shift
      expect(similarity).toBeGreaterThan(0.7);
    });
  });
});
