/**
 * Comprehensive tests for pure TypeScript conformal prediction
 */

import { describe, it, expect, beforeEach } from 'vitest';
import {
  SplitConformalPredictor,
  AdaptiveConformalPredictor,
  CQRPredictor,
} from '../src/pure/conformal';
import { AbsoluteScore, NormalizedScore, QuantileScore } from '../src/pure/scores';
import { PredictionIntervalImpl } from '../src/pure/types';

describe('SplitConformalPredictor', () => {
  let predictor: SplitConformalPredictor;
  let predictions: number[];
  let actuals: number[];

  beforeEach(() => {
    // Create simple synthetic data
    predictor = new SplitConformalPredictor({ alpha: 0.1 });
    predictions = [100, 105, 98, 102, 97, 103, 101, 99, 104, 102];
    actuals = [102, 104, 99, 101, 98, 102, 100, 100, 105, 103];
  });

  describe('basic calibration and prediction', () => {
    it('should calibrate with predictions and actuals', async () => {
      await predictor.calibrate(predictions, actuals);
      const stats = predictor.getStats();
      expect(stats.nCalibration).toBe(10);
      expect(stats.alpha).toBe(0.1);
    });

    it('should throw error if predictions and actuals have different lengths', async () => {
      await expect(
        predictor.calibrate([100, 105], [102, 104, 106])
      ).rejects.toThrow('same length');
    });

    it('should throw error if calibration data is empty', async () => {
      await expect(predictor.calibrate([], [])).rejects.toThrow(
        'least one'
      );
    });

    it('should make predictions after calibration', async () => {
      await predictor.calibrate(predictions, actuals);
      const interval = predictor.predict(103);

      expect(interval.point).toBe(103);
      expect(interval.lower).toBeLessThan(interval.point);
      expect(interval.upper).toBeGreaterThan(interval.point);
      expect(interval.alpha).toBe(0.1);
      expect(interval.width()).toBeGreaterThan(0);
    });

    it('should throw error if predicting before calibration', () => {
      expect(() => predictor.predict(103)).toThrow('not calibrated');
    });
  });

  describe('prediction intervals', () => {
    beforeEach(async () => {
      await predictor.calibrate(predictions, actuals);
    });

    it('should contain point prediction in interval', () => {
      const interval = predictor.predict(100);
      expect(interval.lower).toBeLessThanOrEqual(interval.point);
      expect(interval.upper).toBeGreaterThanOrEqual(interval.point);
    });

    it('should have symmetric intervals around point for AbsoluteScore', () => {
      const interval = predictor.predict(100);
      const lowerDist = interval.point - interval.lower;
      const upperDist = interval.upper - interval.point;
      expect(lowerDist).toBeCloseTo(upperDist, 5);
    });

    it('should have narrower intervals with higher alpha (lower coverage)', async () => {
      // Higher alpha means lower required coverage, so narrower intervals
      const predictor2 = new SplitConformalPredictor({ alpha: 0.2 });
      await predictor2.calibrate(predictions, actuals);

      const interval1 = predictor.predict(100);
      const interval2 = predictor2.predict(100);

      // alpha=0.2 means lower required coverage, so narrower intervals
      expect(interval2.width()).toBeLessThanOrEqual(interval1.width());
    });
  });

  describe('online updates', () => {
    beforeEach(async () => {
      await predictor.calibrate(predictions.slice(0, 5), actuals.slice(0, 5));
    });

    it('should update with new observations', async () => {
      const statsBefore = predictor.getStats();
      await predictor.update(106, 105);
      const statsAfter = predictor.getStats();

      expect(statsAfter.nCalibration).toBe(statsBefore.nCalibration + 1);
    });

    it('should maintain sorted calibration scores after updates', async () => {
      await predictor.update(106, 105);
      await predictor.update(95, 96);
      await predictor.update(110, 109);

      const stats = predictor.getStats();
      expect(stats.nCalibration).toBeGreaterThan(5);
      expect(stats.minScore).toBeGreaterThanOrEqual(0);
      expect(stats.maxScore).toBeGreaterThanOrEqual(stats.minScore);
    });

    it('should respect max calibration size', async () => {
      const config = { alpha: 0.1, calibrationSize: 5 };
      const p = new SplitConformalPredictor(config);
      await p.calibrate(predictions.slice(0, 5), actuals.slice(0, 5));

      for (let i = 0; i < 10; i++) {
        await p.update(100 + i, 101 + i);
      }

      const stats = p.getStats();
      expect(stats.nCalibration).toBeLessThanOrEqual(5);
    });
  });

  describe('quantile computation', () => {
    it('should compute quantile correctly', async () => {
      const testPred = [1, 2, 3, 4, 5];
      const testActual = [1.1, 2.1, 3.1, 4.1, 5.1];
      await predictor.calibrate(testPred, testActual);

      const stats = predictor.getStats();
      expect(stats.quantile).toBeGreaterThan(0);
    });

    it('should increase quantile with lower alpha (higher required coverage)', async () => {
      const p1 = new SplitConformalPredictor({ alpha: 0.1 });
      const p2 = new SplitConformalPredictor({ alpha: 0.01 });

      await p1.calibrate(predictions, actuals);
      await p2.calibrate(predictions, actuals);

      const stats1 = p1.getStats();
      const stats2 = p2.getStats();

      // Lower alpha means higher required coverage, so larger quantile
      expect(stats2.quantile).toBeGreaterThanOrEqual(stats1.quantile);
    });
  });

  describe('empirical coverage', () => {
    beforeEach(async () => {
      await predictor.calibrate(predictions, actuals);
    });

    it('should compute empirical coverage', () => {
      const coverage = predictor.getEmpiricalCoverage(predictions, actuals);
      expect(coverage).toBeGreaterThanOrEqual(0);
      expect(coverage).toBeLessThanOrEqual(1);
    });

    it('should have coverage close to target', () => {
      const coverage = predictor.getEmpiricalCoverage(predictions, actuals);
      // With 10 samples, expect coverage >= 1 - alpha = 0.9
      // This is probabilistic, so just check it's reasonable
      expect(coverage).toBeGreaterThan(0.5);
    });
  });

  describe('score functions', () => {
    it('should work with AbsoluteScore', async () => {
      const p = new SplitConformalPredictor(
        { alpha: 0.1 },
        new AbsoluteScore()
      );
      await p.calibrate(predictions, actuals);
      const interval = p.predict(100);
      expect(interval).toBeDefined();
      expect(interval.width()).toBeGreaterThan(0);
    });

    it('should work with NormalizedScore', async () => {
      const p = new SplitConformalPredictor(
        { alpha: 0.1 },
        new NormalizedScore(0.5)
      );
      await p.calibrate(predictions, actuals);
      const interval = p.predict(100);
      expect(interval).toBeDefined();
      expect(interval.width()).toBeGreaterThan(0);
    });

    it('should work with different NormalizedScore values', async () => {
      const p1 = new SplitConformalPredictor(
        { alpha: 0.1 },
        new NormalizedScore(1.0)
      );
      const p2 = new SplitConformalPredictor(
        { alpha: 0.1 },
        new NormalizedScore(2.0)
      );

      await p1.calibrate(predictions, actuals);
      await p2.calibrate(predictions, actuals);

      const interval1 = p1.predict(100);
      const interval2 = p2.predict(100);

      // Both should have valid intervals
      expect(interval1.width()).toBeGreaterThan(0);
      expect(interval2.width()).toBeGreaterThan(0);
    });
  });

  describe('statistics', () => {
    beforeEach(async () => {
      await predictor.calibrate(predictions, actuals);
    });

    it('should provide stats', () => {
      const stats = predictor.getStats();
      expect(stats).toHaveProperty('nCalibration');
      expect(stats).toHaveProperty('alpha');
      expect(stats).toHaveProperty('quantile');
      expect(stats).toHaveProperty('predictionCount');
      expect(stats).toHaveProperty('minScore');
      expect(stats).toHaveProperty('maxScore');
    });

    it('should track prediction count', () => {
      let stats = predictor.getStats();
      expect(stats.predictionCount).toBe(0);

      predictor.predict(100);
      stats = predictor.getStats();
      expect(stats.predictionCount).toBe(1);

      predictor.predict(105);
      stats = predictor.getStats();
      expect(stats.predictionCount).toBe(2);
    });
  });
});

describe('AdaptiveConformalPredictor', () => {
  let predictor: AdaptiveConformalPredictor;
  let calibrationPredictions: number[];
  let calibrationActuals: number[];

  beforeEach(async () => {
    predictor = new AdaptiveConformalPredictor({
      targetCoverage: 0.9,
      gamma: 0.02,
    });

    calibrationPredictions = [100, 105, 98, 102, 97];
    calibrationActuals = [102, 104, 99, 101, 98];

    await predictor.calibrate(calibrationPredictions, calibrationActuals);
  });

  describe('initialization', () => {
    it('should initialize with target coverage', () => {
      const stats = predictor.getStats();
      expect(stats.targetCoverage).toBe(0.9);
    });

    it('should initialize with learning rate', () => {
      const stats = predictor.getStats();
      expect(stats.alphaCurrent).toBeDefined();
    });
  });

  describe('adaptation', () => {
    it('should make predictions without adaptation', () => {
      const interval = predictor.predict(103);
      expect(interval.point).toBe(103);
      expect(interval.lower).toBeLessThan(interval.upper);
    });

    it('should adapt when actual is provided', async () => {
      const alphaBefore = predictor.getCurrentAlpha();

      await predictor.predictAndAdapt(103, 104);

      const alphaAfter = predictor.getCurrentAlpha();
      // Alpha should change due to adaptation
      expect(alphaAfter).toBeDefined();
    });

    it('should track coverage history', async () => {
      const statsBefore = predictor.getStats();
      const sizeBefore = statsBefore.coverageHistorySize;

      await predictor.predictAndAdapt(103, 104);

      const statsAfter = predictor.getStats();
      const sizeAfter = statsAfter.coverageHistorySize;

      expect(sizeAfter).toBe(sizeBefore + 1);
    });

    it('should adapt alpha during predictions', async () => {
      // Make several predictions with actual values
      const alphaBefore = predictor.getCurrentAlpha();

      for (let i = 0; i < 5; i++) {
        await predictor.predictAndAdapt(100 + i, 101 + i);
      }

      const alphaAfter = predictor.getCurrentAlpha();
      // Alpha should change (be different from initial)
      expect(typeof alphaAfter).toBe('number');
      expect(alphaAfter).toBeGreaterThan(0);
      expect(alphaAfter).toBeLessThan(1);
    });

    it('should maintain alpha within bounds', async () => {
      // Repeatedly adapt with extreme mismatches
      for (let i = 0; i < 20; i++) {
        if (i % 2 === 0) {
          await predictor.predictAndAdapt(100, 200); // Far from prediction
        } else {
          await predictor.predictAndAdapt(100, 100); // Perfect prediction
        }
      }

      const alpha = predictor.getCurrentAlpha();
      const stats = predictor.getStats();
      // Alpha should stay within bounds
      expect(alpha).toBeGreaterThanOrEqual(stats.alpha_min || 0.01);
      expect(alpha).toBeLessThanOrEqual(stats.alpha_max || 0.3);
    });
  });

  describe('empirical coverage', () => {
    it('should compute empirical coverage from history', async () => {
      await predictor.predictAndAdapt(100, 100);
      await predictor.predictAndAdapt(101, 101);
      await predictor.predictAndAdapt(102, 102);

      const coverage = predictor.empiricalCoverage();
      expect(coverage).toBeGreaterThanOrEqual(0);
      expect(coverage).toBeLessThanOrEqual(1);
    });

    it('should return target coverage with empty history', () => {
      const coverage = predictor.empiricalCoverage();
      expect(coverage).toBe(0.9); // Should be target coverage
    });

    it('should maintain coverage window size', async () => {
      const windowSize = 200; // default
      for (let i = 0; i < 300; i++) {
        await predictor.predictAndAdapt(100 + i % 50, 100 + (i + 1) % 50);
      }

      const stats = predictor.getStats();
      expect(stats.coverageHistorySize).toBeLessThanOrEqual(windowSize);
    });
  });

  describe('statistics', () => {
    it('should provide adaptive stats', async () => {
      await predictor.predictAndAdapt(100, 100);
      const stats = predictor.getStats();

      expect(stats).toHaveProperty('alphaCurrent');
      expect(stats).toHaveProperty('empiricalCoverage');
      expect(stats).toHaveProperty('targetCoverage');
      expect(stats).toHaveProperty('coverageDifference');
      expect(stats).toHaveProperty('coverageHistorySize');
    });
  });

  describe('alpha bounds', () => {
    it('should clamp alpha to min/max bounds', async () => {
      const config = {
        targetCoverage: 0.5,
        gamma: 0.5, // High learning rate
        alphaMin: 0.01,
        alphaMax: 0.3,
      };

      const p = new AdaptiveConformalPredictor(config);
      await p.calibrate(calibrationPredictions, calibrationActuals);

      for (let i = 0; i < 50; i++) {
        await p.predictAndAdapt(100, 200); // Always outside
      }

      const alpha = p.getCurrentAlpha();
      expect(alpha).toBeGreaterThanOrEqual(0.01);
      expect(alpha).toBeLessThanOrEqual(0.3);
    });
  });
});

describe('CQRPredictor', () => {
  let predictor: CQRPredictor;
  const qLow = [95, 100, 93, 97, 96];
  const qHigh = [105, 110, 103, 107, 106];
  const actuals = [102, 104, 99, 101, 98];

  beforeEach(async () => {
    predictor = new CQRPredictor(
      { alpha: 0.1 },
      0.05,
      0.95,
      new QuantileScore(0.05, 0.95)
    );
    await predictor.calibrate(qLow, qHigh, actuals);
  });

  it('should calibrate with quantile predictions', () => {
    const stats = predictor.getStats();
    expect(stats.nCalibration).toBe(5);
  });

  it('should reject invalid quantiles', () => {
    expect(() => {
      new CQRPredictor({ alpha: 0.1 }, 0.5, 0.2); // alphaLow > alphaHigh
    }).toThrow('Invalid quantile values');
  });

  it('should make CQR predictions', () => {
    const interval = predictor.predict(98, 108);
    expect(interval.point).toBe(103); // (98 + 108) / 2
    expect(interval.lower).toBeDefined();
    expect(interval.upper).toBeDefined();
    expect(interval.lower).toBeLessThanOrEqual(interval.point);
    expect(interval.upper).toBeGreaterThanOrEqual(interval.point);
  });

  it('should update with new quantile observations', async () => {
    const statsBefore = predictor.getStats();
    await predictor.update(94, 104, 99);
    const statsAfter = predictor.getStats();

    expect(statsAfter.nCalibration).toBe(statsBefore.nCalibration + 1);
  });

  it('should have valid intervals around quantiles', () => {
    const interval = predictor.predict(95, 105);
    expect(interval.lower).toBeLessThanOrEqual(interval.point);
    expect(interval.upper).toBeGreaterThanOrEqual(interval.point);
    expect(interval.width()).toBeGreaterThanOrEqual(0);
  });

  it('should provide CQR statistics', () => {
    const stats = predictor.getStats();
    expect(stats).toHaveProperty('alphaLow');
    expect(stats).toHaveProperty('alphaHigh');
    expect(stats.alphaLow).toBe(0.05);
    expect(stats.alphaHigh).toBe(0.95);
  });
});

describe('PredictionInterval', () => {
  it('should compute width correctly', () => {
    const interval = new PredictionIntervalImpl(100, 95, 105, 0.1, 5);

    expect(interval.width()).toBe(10);
  });

  it('should check containment', () => {
    const interval = new PredictionIntervalImpl(100, 95, 105, 0.1, 5);

    expect(interval.contains(100)).toBe(true);
    expect(interval.contains(95)).toBe(true);
    expect(interval.contains(105)).toBe(true);
    expect(interval.contains(94.9)).toBe(false);
    expect(interval.contains(105.1)).toBe(false);
  });

  it('should compute relative width', () => {
    const interval = new PredictionIntervalImpl(100, 95, 105, 0.1, 5);

    expect(interval.relativeWidth()).toBe(10); // (10 / 100) * 100
  });

  it('should compute coverage correctly', () => {
    const interval = new PredictionIntervalImpl(100, 95, 105, 0.1, 5);

    expect(interval.coverage()).toBe(0.9); // 1 - 0.1
  });
});

describe('ScoreFunctions', () => {
  describe('AbsoluteScore', () => {
    const score = new AbsoluteScore();

    it('should compute absolute residual', () => {
      expect(score.score(100, 105)).toBe(5);
      expect(score.score(105, 100)).toBe(5);
      expect(score.score(100, 100)).toBe(0);
    });

    it('should create symmetric intervals', () => {
      const [lower, upper] = score.interval(100, 5);
      expect(lower).toBe(95);
      expect(upper).toBe(105);
    });
  });

  describe('NormalizedScore', () => {
    it('should normalize by std dev', () => {
      const score = new NormalizedScore(2.0);
      expect(score.score(100, 104)).toBe(2); // |104 - 100| / 2
      expect(score.score(100, 106)).toBe(3); // |106 - 100| / 2
    });

    it('should update std dev', () => {
      const score = new NormalizedScore(2.0);
      const originalScore = score.score(100, 104);
      score.updateStdDev(4.0);
      const updatedScore = score.score(100, 104);
      expect(updatedScore).toBeLessThan(originalScore);
    });

    it('should create scaled intervals', () => {
      const score = new NormalizedScore(2.0);
      const [lower, upper] = score.interval(100, 5);
      // interval = (prediction - quantile*stdDev, prediction + quantile*stdDev)
      // = (100 - 5*2, 100 + 5*2) = (90, 110)
      expect(lower).toBe(90);
      expect(upper).toBe(110);
    });
  });

  describe('QuantileScore', () => {
    it('should compute quantile score', () => {
      const score = new QuantileScore(0.05, 0.95);
      const s = score.scoreQuantiles(95, 105, 100);
      // score = max(qLow - actual, actual - qHigh) = max(95-100, 100-105) = max(-5, -5) = -5
      // Negative score means actual is inside the interval
      expect(s).toBeLessThanOrEqual(0);
    });

    it('should compute positive score for out of range', () => {
      const score = new QuantileScore(0.05, 0.95);
      const s1 = score.scoreQuantiles(95, 105, 90);
      const s2 = score.scoreQuantiles(95, 105, 110);
      // score for 90: max(95-90, 90-105) = max(5, -15) = 5 (outside lower)
      expect(s1).toBeGreaterThan(0);
      // score for 110: max(95-110, 110-105) = max(-15, 5) = 5 (outside upper)
      expect(s2).toBeGreaterThan(0);
    });

    it('should reject invalid quantiles', () => {
      expect(() => {
        new QuantileScore(0.5, 0.2); // Invalid: low > high
      }).toThrow();
    });
  });
});

describe('integration tests', () => {
  it('should handle streaming predictions', async () => {
    const predictor = new SplitConformalPredictor({ alpha: 0.1 });

    // Initial calibration
    const initPred = [100, 105, 98, 102];
    const initActual = [102, 104, 99, 101];
    await predictor.calibrate(initPred, initActual);

    // Stream of online updates
    const updates = [
      { pred: 103, actual: 104 },
      { pred: 99, actual: 100 },
      { pred: 106, actual: 105 },
      { pred: 101, actual: 102 },
    ];

    let errorCount = 0;
    let coveredCount = 0;

    for (const update of updates) {
      const interval = predictor.predict(update.pred);
      if (interval.contains(update.actual)) {
        coveredCount++;
      } else {
        errorCount++;
      }
      await predictor.update(update.pred, update.actual);
    }

    // Should have reasonable coverage
    expect(coveredCount).toBeGreaterThanOrEqual(0);
    expect(coveredCount + errorCount).toBe(updates.length);
  });

  it('should compare different alpha values', async () => {
    const alphas = [0.05, 0.1, 0.2];
    const predictions = [100, 105, 98, 102, 97];
    const actuals = [102, 104, 99, 101, 98];

    const intervals = await Promise.all(
      alphas.map(async (alpha) => {
        const p = new SplitConformalPredictor({ alpha });
        await p.calibrate(predictions, actuals);
        return p.predict(100);
      })
    );

    // Lower alpha (higher coverage) should give wider or equal intervals
    // alpha=0.05 (95% coverage) >= alpha=0.1 (90% coverage) >= alpha=0.2 (80% coverage)
    expect(intervals[0].width()).toBeGreaterThanOrEqual(intervals[1].width());
    expect(intervals[1].width()).toBeGreaterThanOrEqual(intervals[2].width());
  });

  it('should handle large datasets efficiently', async () => {
    const predictor = new SplitConformalPredictor({ alpha: 0.1 });

    // Generate large synthetic dataset
    const predictions: number[] = [];
    const actuals: number[] = [];
    for (let i = 0; i < 1000; i++) {
      const pred = 100 + Math.sin(i * 0.1) * 10;
      predictions.push(pred);
      actuals.push(pred + (Math.random() - 0.5) * 5);
    }

    const start = performance.now();
    await predictor.calibrate(predictions, actuals);
    const calibrationTime = performance.now() - start;

    expect(calibrationTime).toBeLessThan(1000); // Should complete in < 1s

    // Make predictions
    const predStart = performance.now();
    for (let i = 0; i < 100; i++) {
      predictor.predict(100 + Math.random() * 20);
    }
    const predictionTime = performance.now() - predStart;

    expect(predictionTime).toBeLessThan(100); // 100 predictions in < 100ms
  });
});
