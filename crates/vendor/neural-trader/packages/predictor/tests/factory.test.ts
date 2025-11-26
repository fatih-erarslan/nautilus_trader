/**
 * Factory pattern tests
 * Tests the factory functions for creating predictors and scores
 */

import { describe, it, expect, beforeEach } from 'vitest';
import {
  AbsoluteScore,
  NormalizedScore,
  QuantileScore,
  PredictionIntervalImpl,
  defaultPredictorConfig,
  defaultAdaptiveConfig,
} from '../src/index';

describe('Score Factories', () => {
  describe('AbsoluteScore Factory', () => {
    it('should create AbsoluteScore instance', () => {
      const score = new AbsoluteScore();
      expect(score).toBeDefined();
      expect(score.score(100, 105)).toBe(5);
    });

    it('should create multiple instances independently', () => {
      const score1 = new AbsoluteScore();
      const score2 = new AbsoluteScore();

      expect(score1.score(100, 105)).toBe(score2.score(100, 105));
    });

    it('should have consistent behavior across instances', () => {
      const scores = [
        new AbsoluteScore(),
        new AbsoluteScore(),
        new AbsoluteScore(),
      ];

      const result1 = scores[0].score(100, 105);
      const result2 = scores[1].score(100, 105);
      const result3 = scores[2].score(100, 105);

      expect(result1).toBe(result2);
      expect(result2).toBe(result3);
    });

    it('should create scores with correct interval computation', () => {
      const score = new AbsoluteScore();
      const [lower, upper] = score.interval(100, 10);

      expect(lower).toBe(90);
      expect(upper).toBe(110);
    });
  });

  describe('NormalizedScore Factory', () => {
    it('should create NormalizedScore with default stddev', () => {
      const score = new NormalizedScore();
      expect(score).toBeDefined();
    });

    it('should create NormalizedScore with custom stddev', () => {
      const score = new NormalizedScore(2.0);
      expect(score.score(100, 104)).toBeCloseTo(2);
    });

    it('should allow stddev updates', () => {
      const score = new NormalizedScore(1.0);
      score.updateStdDev(2.0);

      const s = score.score(100, 104);
      expect(s).toBeCloseTo(2);
    });

    it('should create multiple instances with different stddevs', () => {
      const scores = [
        new NormalizedScore(1.0),
        new NormalizedScore(2.0),
        new NormalizedScore(0.5),
      ];

      const pred = 100;
      const actual = 105;

      const s1 = scores[0].score(pred, actual);
      const s2 = scores[1].score(pred, actual);
      const s3 = scores[2].score(pred, actual);

      expect(s1).toBeCloseTo(5);
      expect(s2).toBeCloseTo(2.5);
      expect(s3).toBeCloseTo(10);
    });

    it('should handle stddev updates gracefully', () => {
      const score = new NormalizedScore(1.0);

      score.updateStdDev(0.5);
      expect(score.score(100, 105)).toBeCloseTo(10);

      score.updateStdDev(5.0);
      expect(score.score(100, 105)).toBeCloseTo(1);
    });
  });

  describe('QuantileScore Factory', () => {
    it('should create QuantileScore with valid quantiles', () => {
      const score = new QuantileScore(0.05, 0.95);
      expect(score).toBeDefined();
    });

    it('should throw on invalid quantiles', () => {
      expect(() => new QuantileScore(0.95, 0.05)).toThrow();
      expect(() => new QuantileScore(0.5, 0.5)).toThrow();
    });

    it('should create multiple instances with different quantiles', () => {
      const scores = [
        new QuantileScore(0.05, 0.95),
        new QuantileScore(0.1, 0.9),
        new QuantileScore(0.25, 0.75),
      ];

      expect(scores).toHaveLength(3);
      for (const score of scores) {
        expect(score.score(100, 105)).toBeGreaterThanOrEqual(0);
      }
    });

    it('should compute quantile scores correctly', () => {
      const score = new QuantileScore(0.05, 0.95);

      // When actual is within bounds
      expect(score.scoreQuantiles(90, 110, 100)).toBe(0);

      // When actual is below lower bound
      expect(score.scoreQuantiles(90, 110, 85)).toBe(5);

      // When actual is above upper bound
      expect(score.scoreQuantiles(90, 110, 115)).toBe(5);
    });
  });
});

describe('PredictionInterval Factory', () => {
  it('should create PredictionInterval with all parameters', () => {
    const interval = new PredictionIntervalImpl(100, 90, 110, 0.1, 10, 1000);

    expect(interval.point).toBe(100);
    expect(interval.lower).toBe(90);
    expect(interval.upper).toBe(110);
    expect(interval.alpha).toBe(0.1);
    expect(interval.quantile).toBe(10);
  });

  it('should create PredictionInterval with default timestamp', () => {
    const before = Date.now();
    const interval = new PredictionIntervalImpl(100, 90, 110, 0.1, 10);
    const after = Date.now();

    expect(interval.timestamp).toBeGreaterThanOrEqual(before);
    expect(interval.timestamp).toBeLessThanOrEqual(after);
  });

  it('should create multiple intervals independently', () => {
    const intervals = [
      new PredictionIntervalImpl(100, 90, 110, 0.1, 10),
      new PredictionIntervalImpl(105, 95, 115, 0.1, 10),
      new PredictionIntervalImpl(95, 85, 105, 0.1, 10),
    ];

    expect(intervals).toHaveLength(3);
    expect(intervals[0].point).toBe(100);
    expect(intervals[1].point).toBe(105);
    expect(intervals[2].point).toBe(95);
  });

  it('should handle intervals with varying alpha values', () => {
    const intervals = [
      new PredictionIntervalImpl(100, 90, 110, 0.05, 10), // 95% coverage
      new PredictionIntervalImpl(100, 90, 110, 0.1, 10),  // 90% coverage
      new PredictionIntervalImpl(100, 90, 110, 0.2, 10),  // 80% coverage
    ];

    expect(intervals[0].coverage()).toBeCloseTo(0.95);
    expect(intervals[1].coverage()).toBeCloseTo(0.9);
    expect(intervals[2].coverage()).toBeCloseTo(0.8);
  });

  it('should compute metrics for created intervals', () => {
    const intervals = [];

    for (let i = 0; i < 5; i++) {
      const point = 100 + i * 10;
      intervals.push(new PredictionIntervalImpl(point, point - 5, point + 5, 0.1, 5));
    }

    let totalWidth = 0;
    let avgCoverage = 0;

    for (const interval of intervals) {
      totalWidth += interval.width();
      avgCoverage += interval.coverage();
    }

    expect(totalWidth / intervals.length).toBe(10);
    expect(avgCoverage / intervals.length).toBeCloseTo(0.9);
  });
});

describe('Configuration Factories', () => {
  it('should use default predictor config', () => {
    expect(defaultPredictorConfig.alpha).toBe(0.1);
    expect(defaultPredictorConfig.calibrationSize).toBe(2000);
    expect(defaultPredictorConfig.maxIntervalWidthPct).toBe(5.0);
    expect(defaultPredictorConfig.recalibrationFreq).toBe(100);
  });

  it('should use default adaptive config', () => {
    expect(defaultAdaptiveConfig.targetCoverage).toBe(0.9);
    expect(defaultAdaptiveConfig.gamma).toBe(0.02);
    expect(defaultAdaptiveConfig.coverageWindow).toBe(200);
    expect(defaultAdaptiveConfig.alphaMin).toBe(0.01);
    expect(defaultAdaptiveConfig.alphaMax).toBe(0.3);
  });

  it('should allow custom config creation', () => {
    const customConfig = {
      alpha: 0.05,
      calibrationSize: 1000,
      maxIntervalWidthPct: 3.0,
      recalibrationFreq: 50,
    };

    expect(customConfig.alpha).toBe(0.05);
    expect(customConfig.calibrationSize).toBe(1000);
  });

  it('should allow custom adaptive config creation', () => {
    const customConfig = {
      targetCoverage: 0.95,
      gamma: 0.01,
      coverageWindow: 100,
      alphaMin: 0.02,
      alphaMax: 0.25,
    };

    expect(customConfig.targetCoverage).toBe(0.95);
    expect(customConfig.gamma).toBe(0.01);
  });
});

describe('Composite Factory Patterns', () => {
  it('should create predictor with score', () => {
    const score = new AbsoluteScore();
    const interval = new PredictionIntervalImpl(100, 90, 110, 0.1, 10);

    expect(score.score(interval.point, 105)).toBe(5);
  });

  it('should create multiple predictors with different scores', () => {
    const configs = [
      { score: new AbsoluteScore(), alpha: 0.1 },
      { score: new NormalizedScore(1.0), alpha: 0.1 },
      { score: new QuantileScore(0.05, 0.95), alpha: 0.1 },
    ];

    expect(configs).toHaveLength(3);
    for (const config of configs) {
      expect(config.score).toBeDefined();
    }
  });

  it('should create interval batch from scores', () => {
    const score = new AbsoluteScore();
    const points = [100, 105, 95, 110, 90];
    const intervals = [];

    for (const point of points) {
      const [lower, upper] = score.interval(point, 5);
      intervals.push(new PredictionIntervalImpl(point, lower, upper, 0.1, 5));
    }

    expect(intervals).toHaveLength(5);
    for (let i = 0; i < intervals.length; i++) {
      expect(intervals[i].point).toBe(points[i]);
    }
  });

  it('should support score polymorphism', () => {
    const scores = [
      new AbsoluteScore(),
      new NormalizedScore(2.0),
      new QuantileScore(0.05, 0.95),
    ] as const;

    // All should work with the same interface
    for (const score of scores) {
      const interval = score.interval(100, 5);
      expect(interval).toHaveLength(2);
      expect(interval[0]).toBeLessThan(interval[1]);
    }
  });
});

describe('Factory Error Handling', () => {
  it('should handle invalid score parameters gracefully', () => {
    expect(() => new QuantileScore(1.5, 0.5)).toThrow();
    expect(() => new QuantileScore(-0.1, 0.9)).toThrow();
  });

  it('should handle extreme interval values', () => {
    // Very large values
    const largeInterval = new PredictionIntervalImpl(1e10, 1e10 - 1e9, 1e10 + 1e9, 0.1, 1e9);
    expect(largeInterval.width()).toBe(2e9);

    // Very small values
    const smallInterval = new PredictionIntervalImpl(1e-10, 0, 2e-10, 0.1, 1e-10);
    expect(smallInterval.width()).toBe(2e-10);
  });

  it('should handle edge case configurations', () => {
    const edgeConfig1 = {
      alpha: 0.001,
      calibrationSize: 10,
      maxIntervalWidthPct: 0.1,
      recalibrationFreq: 1,
    };

    const edgeConfig2 = {
      alpha: 0.99,
      calibrationSize: 100000,
      maxIntervalWidthPct: 100.0,
      recalibrationFreq: 10000,
    };

    expect(edgeConfig1.alpha).toBe(0.001);
    expect(edgeConfig2.alpha).toBe(0.99);
  });
});

describe('Factory Batch Operations', () => {
  it('should create batch of scores', () => {
    const scoreTypes = [
      () => new AbsoluteScore(),
      () => new NormalizedScore(1.0),
      () => new QuantileScore(0.05, 0.95),
    ];

    const scores = scoreTypes.map(factory => factory());
    expect(scores).toHaveLength(3);
  });

  it('should create batch of intervals', () => {
    const score = new AbsoluteScore();
    const batchSize = 100;
    const intervals = [];

    for (let i = 0; i < batchSize; i++) {
      const point = 100 + i;
      const [lower, upper] = score.interval(point, 5);
      intervals.push(new PredictionIntervalImpl(point, lower, upper, 0.1, 5));
    }

    expect(intervals).toHaveLength(batchSize);
  });

  it('should batch create with different alphas', () => {
    const alphas = [0.01, 0.05, 0.1, 0.2, 0.3];
    const intervals = [];

    for (const alpha of alphas) {
      intervals.push(new PredictionIntervalImpl(100, 90, 110, alpha, 10));
    }

    expect(intervals).toHaveLength(5);
    for (let i = 0; i < intervals.length; i++) {
      expect(intervals[i].alpha).toBe(alphas[i]);
      expect(intervals[i].coverage()).toBeCloseTo(1 - alphas[i]);
    }
  });
});

describe('Factory Memory and Performance', () => {
  it('should efficiently create many scores', () => {
    const startTime = performance.now();

    for (let i = 0; i < 1000; i++) {
      new AbsoluteScore();
    }

    const endTime = performance.now();
    expect(endTime - startTime).toBeLessThan(1000); // Should complete in < 1 second
  });

  it('should efficiently create many intervals', () => {
    const score = new AbsoluteScore();
    const startTime = performance.now();

    for (let i = 0; i < 1000; i++) {
      new PredictionIntervalImpl(100 + i, 90 + i, 110 + i, 0.1, 10);
    }

    const endTime = performance.now();
    expect(endTime - startTime).toBeLessThan(1000); // Should complete in < 1 second
  });

  it('should reuse score instances efficiently', () => {
    const score = new AbsoluteScore();
    const intervals = [];

    for (let i = 0; i < 100; i++) {
      const [lower, upper] = score.interval(100 + i, 5);
      intervals.push(new PredictionIntervalImpl(100 + i, lower, upper, 0.1, 5));
    }

    expect(intervals).toHaveLength(100);
  });
});
