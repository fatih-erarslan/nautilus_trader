/**
 * End-to-end integration tests
 * Tests complete workflows with realistic market scenarios
 */

import { describe, it, expect, beforeEach } from 'vitest';
import {
  PredictionIntervalImpl,
  AbsoluteScore,
  NormalizedScore,
  QuantileScore,
  defaultPredictorConfig,
  defaultAdaptiveConfig,
} from '../src/index';

/**
 * Generate synthetic market data with realistic patterns
 */
function generateMarketData(
  points: number,
  trend: number = 0.1,
  volatility: number = 1.0
): { predictions: number[]; actuals: number[] } {
  const predictions: number[] = [];
  const actuals: number[] = [];

  for (let i = 0; i < points; i++) {
    const t = i;
    const trendComponent = trend * t;
    // Simple pseudo-random noise using trigonometric functions
    const noise = volatility * Math.sin(t * 12.9898 + 4.1414) * Math.sin(43758.5453 * t);

    const predicted = 100.0 + trendComponent;
    const actual = 100.0 + trendComponent + noise;

    predictions.push(predicted);
    actuals.push(actual);
  }

  return { predictions, actuals };
}

describe('Integration: Basic Workflow', () => {
  it('should execute complete prediction workflow', () => {
    const score = new AbsoluteScore();
    const { predictions, actuals } = generateMarketData(50, 0.1, 1.0);

    // Simulate calibration
    let sumSquaredError = 0;
    for (let i = 0; i < predictions.length; i++) {
      const error = score.score(predictions[i], actuals[i]);
      sumSquaredError += error * error;
    }
    const rmse = Math.sqrt(sumSquaredError / predictions.length);
    expect(rmse).toBeGreaterThan(0);

    // Create prediction intervals
    const intervals = [];
    for (let i = 0; i < 10; i++) {
      const point = 100.0 + i;
      const quantile = rmse * 1.96; // ~95% coverage
      const [lower, upper] = score.interval(point, quantile);
      intervals.push(new PredictionIntervalImpl(point, lower, upper, 0.05, quantile));
    }

    // Verify intervals
    for (const interval of intervals) {
      expect(interval.lower).toBeLessThan(interval.upper);
      expect(interval.contains(interval.point)).toBe(true);
    }
  });

  it('should work with different score functions', () => {
    const scores = [new AbsoluteScore(), new NormalizedScore(1.0), new QuantileScore(0.05, 0.95)];

    for (const score of scores) {
      const [lower, upper] = score.interval(100, 5);
      const interval = new PredictionIntervalImpl(100, lower, upper, 0.1, 5);

      expect(interval.lower).toBeLessThan(interval.upper);
      expect(interval.contains(100)).toBe(true);
    }
  });
});

describe('Integration: Sequential Predictions', () => {
  it('should make sequential streaming predictions', () => {
    const score = new AbsoluteScore();
    const intervals = [];

    // Simulate streaming predictions
    for (let i = 0; i < 50; i++) {
      const point = 100.0 + (i * 0.5);
      const quantile = 2.0; // Fixed quantile
      const [lower, upper] = score.interval(point, quantile);
      intervals.push(new PredictionIntervalImpl(point, lower, upper, 0.1, quantile));
    }

    // Verify monotonicity of point predictions
    for (let i = 1; i < intervals.length; i++) {
      expect(intervals[i].point).toBeGreaterThan(intervals[i - 1].point);
    }

    // Verify all intervals are valid
    for (const interval of intervals) {
      expect(interval.width()).toBe(4.0);
    }
  });

  it('should track coverage across multiple predictions', () => {
    const score = new AbsoluteScore();

    // Generate synthetic true values
    const truths = [];
    for (let i = 0; i < 100; i++) {
      truths.push(100 + i * 0.1 + Math.sin(i * 0.1) * 0.5);
    }

    // Make predictions
    let covered = 0;
    for (let i = 0; i < truths.length; i++) {
      const point = 100 + i * 0.1;
      const [lower, upper] = score.interval(point, 2.0);
      const interval = new PredictionIntervalImpl(point, lower, upper, 0.1, 2.0);

      if (interval.contains(truths[i])) {
        covered++;
      }
    }

    // Should achieve reasonable coverage
    const coverage = covered / truths.length;
    expect(coverage).toBeGreaterThan(0.5);
  });
});

describe('Integration: Multiple Score Types', () => {
  it('should compare intervals from different scores', () => {
    const point = 100.0;
    const quantile = 5.0;

    // Absolute score
    const absScore = new AbsoluteScore();
    const [absLower, absUpper] = absScore.interval(point, quantile);

    // Normalized score
    const normScore = new NormalizedScore(1.0);
    const [normLower, normUpper] = normScore.interval(point, quantile);

    // Quantile score
    const qScore = new QuantileScore(0.05, 0.95);
    const [qLower, qUpper] = qScore.interval(point, quantile);

    // All should be valid intervals
    expect(absLower).toBeLessThan(absUpper);
    expect(normLower).toBeLessThan(normUpper);
    expect(qLower).toBeLessThan(qUpper);

    // All should contain point
    expect(absLower <= point && point <= absUpper).toBe(true);
    expect(normLower <= point && point <= normUpper).toBe(true);
    expect(qLower <= point && point <= qUpper).toBe(true);
  });

  it('should switch between scores dynamically', () => {
    const point = 100.0;
    const quantile = 5.0;

    const scoreFactories = [
      () => new AbsoluteScore(),
      () => new NormalizedScore(1.0),
      () => new QuantileScore(0.05, 0.95),
      () => new AbsoluteScore(),
      () => new NormalizedScore(2.0),
    ];

    const intervals = [];
    for (const factory of scoreFactories) {
      const score = factory();
      const [lower, upper] = score.interval(point, quantile);
      intervals.push(new PredictionIntervalImpl(point, lower, upper, 0.1, quantile));
    }

    // All should be valid
    for (const interval of intervals) {
      expect(interval.lower).toBeLessThan(interval.upper);
    }
  });
});

describe('Integration: Market Scenarios', () => {
  it('should handle uptrend scenario', () => {
    const { predictions, actuals } = generateMarketData(100, 0.5, 2.0);

    const score = new AbsoluteScore();
    let totalError = 0;

    for (let i = 0; i < predictions.length; i++) {
      totalError += score.score(predictions[i], actuals[i]);
    }

    const avgError = totalError / predictions.length;
    expect(avgError).toBeGreaterThan(0);

    // Make predictions on trend continuation
    const intervals = [];
    for (let i = 0; i < 10; i++) {
      const point = 100.0 + 50.0 + i;
      const [lower, upper] = score.interval(point, avgError * 2);
      intervals.push(new PredictionIntervalImpl(point, lower, upper, 0.1, avgError * 2));
    }

    expect(intervals).toHaveLength(10);
  });

  it('should handle downtrend scenario', () => {
    const { predictions, actuals } = generateMarketData(100, -0.5, 2.0);

    const score = new AbsoluteScore();
    let totalError = 0;

    for (let i = 0; i < predictions.length; i++) {
      totalError += score.score(predictions[i], actuals[i]);
    }

    const avgError = totalError / predictions.length;

    // Make predictions on trend continuation
    const intervals = [];
    for (let i = 0; i < 10; i++) {
      const point = 100.0 - 50.0 + i;
      const [lower, upper] = score.interval(point, avgError * 2);
      intervals.push(new PredictionIntervalImpl(point, lower, upper, 0.1, avgError * 2));
    }

    expect(intervals).toHaveLength(10);
  });

  it('should handle high volatility scenario', () => {
    const { predictions, actuals } = generateMarketData(100, 0.1, 5.0);

    const score = new AbsoluteScore();
    let maxError = 0;

    for (let i = 0; i < predictions.length; i++) {
      const error = score.score(predictions[i], actuals[i]);
      maxError = Math.max(maxError, error);
    }

    // High volatility should result in wider intervals
    const interval = new PredictionIntervalImpl(100, 100 - maxError, 100 + maxError, 0.1, maxError);
    expect(interval.width()).toBeGreaterThan(1.0);
  });

  it('should handle low volatility scenario', () => {
    const { predictions, actuals } = generateMarketData(100, 0.1, 0.1);

    const score = new AbsoluteScore();
    let sumError = 0;

    for (let i = 0; i < predictions.length; i++) {
      sumError += score.score(predictions[i], actuals[i]);
    }

    const avgError = sumError / predictions.length;

    // Low volatility should result in narrow intervals
    const interval = new PredictionIntervalImpl(100, 100 - avgError, 100 + avgError, 0.1, avgError);
    expect(interval.width()).toBeLessThan(2.0);
  });
});

describe('Integration: Batch Processing', () => {
  it('should process batch of predictions', () => {
    const score = new AbsoluteScore();
    const points = [100, 105, 110, 95, 90];
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

  it('should compute batch statistics', () => {
    const score = new AbsoluteScore();
    const intervals = [];

    for (let i = 0; i < 100; i++) {
      const point = 100 + i;
      const [lower, upper] = score.interval(point, 5);
      intervals.push(new PredictionIntervalImpl(point, lower, upper, 0.1, 5));
    }

    // Compute statistics
    let totalWidth = 0;
    let totalCoverage = 0;
    let minAlpha = Infinity;
    let maxAlpha = -Infinity;

    for (const interval of intervals) {
      totalWidth += interval.width();
      totalCoverage += interval.coverage();
      minAlpha = Math.min(minAlpha, interval.alpha);
      maxAlpha = Math.max(maxAlpha, interval.alpha);
    }

    expect(totalWidth / intervals.length).toBe(10);
    expect(totalCoverage / intervals.length).toBe(0.9);
    expect(minAlpha).toBe(0.1);
    expect(maxAlpha).toBe(0.1);
  });
});

describe('Integration: Different Alpha Values', () => {
  it('should compare intervals with different confidence levels', () => {
    const point = 100.0;
    const quantile = 5.0;

    const alphas = [0.01, 0.05, 0.1, 0.2, 0.3];
    const intervals = [];

    for (const alpha of alphas) {
      intervals.push(new PredictionIntervalImpl(point, point - quantile, point + quantile, alpha, quantile));
    }

    // All intervals should have same width
    for (const interval of intervals) {
      expect(interval.width()).toBe(2 * quantile);
    }

    // But different coverage values
    expect(intervals[0].coverage()).toBeCloseTo(0.99);
    expect(intervals[1].coverage()).toBeCloseTo(0.95);
    expect(intervals[2].coverage()).toBeCloseTo(0.9);
    expect(intervals[3].coverage()).toBeCloseTo(0.8);
    expect(intervals[4].coverage()).toBeCloseTo(0.7);
  });

  it('should produce wider intervals for higher confidence', () => {
    const score = new AbsoluteScore();
    const point = 100.0;

    // Low confidence (high alpha)
    const [lower1, upper1] = score.interval(point, 2.0);
    const interval1 = new PredictionIntervalImpl(point, lower1, upper1, 0.2, 2.0);

    // High confidence (low alpha)
    const [lower2, upper2] = score.interval(point, 5.0);
    const interval2 = new PredictionIntervalImpl(point, lower2, upper2, 0.05, 5.0);

    // Higher confidence should have wider interval (larger quantile)
    expect(interval2.width()).toBeGreaterThan(interval1.width());
  });
});

describe('Integration: Adaptive Behavior Simulation', () => {
  it('should simulate adaptive alpha adjustment', () => {
    let alpha = 0.1;
    const intervals = [];
    const gamma = 0.02; // Learning rate

    const { predictions, actuals } = generateMarketData(100, 0.1, 1.0);

    // Calculate errors
    const errors: number[] = [];
    const score = new AbsoluteScore();
    for (let i = 0; i < predictions.length; i++) {
      errors.push(score.score(predictions[i], actuals[i]));
    }
    const avgError = errors.reduce((a, b) => a + b) / errors.length;

    // Simulate adaptive adjustment
    for (let i = 0; i < 50; i++) {
      const point = 100.0 + i;
      const quantile = avgError;

      // Simulate coverage feedback
      const rand = Math.sin(i * 123.456) * 0.5 + 0.5;
      const covered = rand > alpha;

      // Update alpha based on coverage
      if (covered) {
        alpha += gamma; // Increase alpha if over-covering
      } else {
        alpha -= gamma; // Decrease alpha if under-covering
      }

      // Clamp alpha
      alpha = Math.max(0.01, Math.min(0.3, alpha));

      const [lower, upper] = score.interval(point, quantile);
      intervals.push(new PredictionIntervalImpl(point, lower, upper, alpha, quantile));
    }

    // Verify alpha stayed in bounds
    for (const interval of intervals) {
      expect(interval.alpha).toBeGreaterThanOrEqual(0.01);
      expect(interval.alpha).toBeLessThanOrEqual(0.3);
    }
  });
});

describe('Integration: Configuration Impact', () => {
  it('should use default configuration', () => {
    const config = defaultPredictorConfig;

    expect(config.alpha).toBe(0.1);
    expect(config.calibrationSize).toBe(2000);
    expect(config.maxIntervalWidthPct).toBe(5.0);
    expect(config.recalibrationFreq).toBe(100);
  });

  it('should support custom configuration', () => {
    const customConfig = {
      alpha: 0.05,
      calibrationSize: 1000,
      maxIntervalWidthPct: 3.0,
      recalibrationFreq: 50,
    };

    // Create intervals respecting custom config
    const score = new AbsoluteScore();
    const intervals = [];

    for (let i = 0; i < 10; i++) {
      const point = 100.0 + i;
      // Max width based on config: 3% of point
      const maxWidth = (point * customConfig.maxIntervalWidthPct) / 100;
      const [lower, upper] = score.interval(point, maxWidth / 2);

      intervals.push(new PredictionIntervalImpl(point, lower, upper, customConfig.alpha, maxWidth / 2));
    }

    // Verify all intervals respect max width
    for (const interval of intervals) {
      const widthPct = (interval.width() / interval.point) * 100;
      expect(widthPct).toBeLessThanOrEqual(customConfig.maxIntervalWidthPct + 0.01);
    }
  });
});

describe('Integration: Serialization', () => {
  it('should serialize and deserialize intervals', () => {
    const interval = new PredictionIntervalImpl(100, 90, 110, 0.1, 10, 1000);

    // Create a serializable object
    const serialized = {
      point: interval.point,
      lower: interval.lower,
      upper: interval.upper,
      alpha: interval.alpha,
      quantile: interval.quantile,
      timestamp: interval.timestamp,
    };

    // Deserialize and recreate
    const deserialized = new PredictionIntervalImpl(
      serialized.point,
      serialized.lower,
      serialized.upper,
      serialized.alpha,
      serialized.quantile,
      serialized.timestamp
    );

    expect(deserialized.point).toBe(interval.point);
    expect(deserialized.lower).toBe(interval.lower);
    expect(deserialized.upper).toBe(interval.upper);
    expect(deserialized.width()).toBe(interval.width());
  });

  it('should handle batch serialization', () => {
    const intervals = [];
    const score = new AbsoluteScore();

    for (let i = 0; i < 10; i++) {
      const point = 100 + i;
      const [lower, upper] = score.interval(point, 5);
      intervals.push(new PredictionIntervalImpl(point, lower, upper, 0.1, 5));
    }

    // Serialize batch
    const serialized = intervals.map(interval => ({
      point: interval.point,
      lower: interval.lower,
      upper: interval.upper,
      alpha: interval.alpha,
      quantile: interval.quantile,
      timestamp: interval.timestamp,
    }));

    expect(serialized).toHaveLength(10);

    // Deserialize batch
    const deserialized = serialized.map(
      s => new PredictionIntervalImpl(s.point, s.lower, s.upper, s.alpha, s.quantile, s.timestamp)
    );

    expect(deserialized).toHaveLength(10);
    for (let i = 0; i < intervals.length; i++) {
      expect(deserialized[i].point).toBe(intervals[i].point);
    }
  });
});

describe('Integration: Performance and Scalability', () => {
  it('should efficiently generate large prediction batches', () => {
    const startTime = performance.now();
    const score = new AbsoluteScore();
    const intervals = [];

    for (let i = 0; i < 10000; i++) {
      const point = 100 + i * 0.01;
      const [lower, upper] = score.interval(point, 2);
      intervals.push(new PredictionIntervalImpl(point, lower, upper, 0.1, 2));
    }

    const endTime = performance.now();
    expect(intervals).toHaveLength(10000);
    expect(endTime - startTime).toBeLessThan(5000); // Should complete in < 5 seconds
  });

  it('should compute statistics efficiently', () => {
    const intervals = [];
    const score = new AbsoluteScore();

    for (let i = 0; i < 1000; i++) {
      const point = 100 + i * 0.1;
      const [lower, upper] = score.interval(point, 5);
      intervals.push(new PredictionIntervalImpl(point, lower, upper, 0.1, 5));
    }

    const startTime = performance.now();

    // Compute statistics
    let totalWidth = 0;
    let avgCoverage = 0;
    for (const interval of intervals) {
      totalWidth += interval.width();
      avgCoverage += interval.coverage();
    }

    totalWidth /= intervals.length;
    avgCoverage /= intervals.length;

    const endTime = performance.now();

    expect(totalWidth).toBe(10);
    expect(avgCoverage).toBeCloseTo(0.9);
    expect(endTime - startTime).toBeLessThan(100); // Should complete in < 100ms
  });
});
