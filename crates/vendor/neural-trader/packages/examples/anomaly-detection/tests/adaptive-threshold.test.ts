import { describe, it, expect, beforeEach } from 'vitest';
import { AdaptiveThreshold } from '../src/adaptive-threshold';

describe('AdaptiveThreshold', () => {
  let threshold: AdaptiveThreshold;

  beforeEach(() => {
    threshold = new AdaptiveThreshold({
      targetFalsePositiveRate: 0.05,
      windowSize: 100,
      adaptationRate: 0.1,
      initialThreshold: 1.0,
    });
  });

  describe('Initialization', () => {
    it('should initialize with default threshold', () => {
      expect(threshold.getThreshold()).toBe(1.0);
    });

    it('should start with zero FPR', () => {
      expect(threshold.getFalsePositiveRate()).toBe(0);
    });
  });

  describe('Threshold Adaptation', () => {
    it('should increase threshold when FPR is too high', () => {
      const initialThreshold = threshold.getThreshold();

      // Simulate high false positive rate
      for (let i = 0; i < 20; i++) {
        threshold.update(1.5, false); // Predicted positive, but actually negative
      }

      const newThreshold = threshold.getThreshold();
      expect(newThreshold).toBeGreaterThan(initialThreshold);
    });

    it('should decrease threshold when FPR is too low', () => {
      // First create some history
      for (let i = 0; i < 50; i++) {
        threshold.update(0.5, false); // Predicted negative, actually negative
      }

      const initialThreshold = threshold.getThreshold();

      // Now add some true positives to lower FPR
      for (let i = 0; i < 20; i++) {
        threshold.update(1.5, true); // Predicted positive, actually positive
      }

      const newThreshold = threshold.getThreshold();
      expect(newThreshold).toBeLessThan(initialThreshold);
    });

    it('should respect threshold bounds', () => {
      const boundedThreshold = new AdaptiveThreshold({
        targetFalsePositiveRate: 0.05,
        windowSize: 100,
        adaptationRate: 0.5,
        initialThreshold: 1.0,
        minThreshold: 0.5,
        maxThreshold: 2.0,
      });

      // Try to push threshold below minimum
      for (let i = 0; i < 100; i++) {
        boundedThreshold.update(0.1, true);
      }

      expect(boundedThreshold.getThreshold()).toBeGreaterThanOrEqual(0.5);

      // Try to push threshold above maximum
      for (let i = 0; i < 100; i++) {
        boundedThreshold.update(3.0, false);
      }

      expect(boundedThreshold.getThreshold()).toBeLessThanOrEqual(2.0);
    });
  });

  describe('Metrics Computation', () => {
    beforeEach(() => {
      // Create labeled dataset
      // True positives: score > 1.0, label = true
      for (let i = 0; i < 30; i++) {
        threshold.update(1.5, true);
      }

      // True negatives: score < 1.0, label = false
      for (let i = 0; i < 60; i++) {
        threshold.update(0.5, false);
      }

      // False positives: score > 1.0, label = false
      for (let i = 0; i < 5; i++) {
        threshold.update(1.5, false);
      }

      // False negatives: score < 1.0, label = true
      for (let i = 0; i < 5; i++) {
        threshold.update(0.5, true);
      }
    });

    it('should compute precision correctly', () => {
      const precision = threshold.getPrecision();
      // TP / (TP + FP) = 30 / (30 + 5) = 0.857
      expect(precision).toBeCloseTo(0.857, 2);
    });

    it('should compute recall correctly', () => {
      const recall = threshold.getRecall();
      // TP / (TP + FN) = 30 / (30 + 5) = 0.857
      expect(recall).toBeCloseTo(0.857, 2);
    });

    it('should compute F1 score correctly', () => {
      const f1 = threshold.getF1Score();
      const precision = threshold.getPrecision();
      const recall = threshold.getRecall();
      const expectedF1 = 2 * (precision * recall) / (precision + recall);

      expect(f1).toBeCloseTo(expectedF1, 3);
    });
  });

  describe('Sliding Window', () => {
    it('should maintain window size', () => {
      const windowSize = 50;
      const smallWindowThreshold = new AdaptiveThreshold({
        targetFalsePositiveRate: 0.05,
        windowSize,
        adaptationRate: 0.1,
      });

      // Add more than window size
      for (let i = 0; i < 100; i++) {
        smallWindowThreshold.update(Math.random(), Math.random() > 0.5);
      }

      const metrics = smallWindowThreshold.getMetrics();
      expect(metrics.totalDetections).toBe(windowSize);
    });
  });

  describe('Reset', () => {
    it('should reset to initial state', () => {
      // Add some data
      for (let i = 0; i < 50; i++) {
        threshold.update(1.5, false);
      }

      threshold.reset(2.0);

      expect(threshold.getThreshold()).toBe(2.0);
      expect(threshold.getFalsePositiveRate()).toBe(0);
      expect(threshold.getMetrics().totalDetections).toBe(0);
    });
  });
});
