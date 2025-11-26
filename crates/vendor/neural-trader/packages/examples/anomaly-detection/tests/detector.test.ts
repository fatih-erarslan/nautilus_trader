import { describe, it, expect, beforeEach } from 'vitest';
import { AnomalyDetector } from '../src/detector';
import type { AnomalyPoint } from '../src/index';

describe('AnomalyDetector', () => {
  let detector: AnomalyDetector;

  beforeEach(() => {
    detector = new AnomalyDetector({
      targetFalsePositiveRate: 0.05,
      featureDimensions: 2,
      useEnsemble: false,
      useConformal: false,
      windowSize: 100,
      minCalibrationSamples: 50,
    });
  });

  describe('Calibration', () => {
    it('should calibrate with sufficient training data', async () => {
      const trainingData: AnomalyPoint[] = generateNormalData(100, 2);

      await detector.calibrate(trainingData);

      const stats = detector.getStatistics();
      expect(stats.isCalibrated).toBe(true);
    });

    it('should reject insufficient calibration data', async () => {
      const trainingData: AnomalyPoint[] = generateNormalData(30, 2);

      await expect(detector.calibrate(trainingData)).rejects.toThrow('Insufficient calibration data');
    });
  });

  describe('Detection', () => {
    beforeEach(async () => {
      const trainingData = generateNormalData(100, 2);
      await detector.calibrate(trainingData);
    });

    it('should detect normal points as normal', async () => {
      const normalPoint: AnomalyPoint = {
        timestamp: Date.now(),
        features: [0.1, 0.2],
      };

      const result = await detector.detect(normalPoint);

      expect(result.detection.isAnomaly).toBe(false);
      expect(result.detection.score).toBeLessThan(2);
    });

    it('should detect anomalous points as anomalies', async () => {
      const anomalousPoint: AnomalyPoint = {
        timestamp: Date.now(),
        features: [10, 10], // Far from normal distribution
      };

      const result = await detector.detect(anomalousPoint);

      expect(result.detection.isAnomaly).toBe(true);
      expect(result.detection.score).toBeGreaterThan(2);
    });

    it('should provide confidence scores', async () => {
      const point: AnomalyPoint = {
        timestamp: Date.now(),
        features: [0.5, 0.5],
      };

      const result = await detector.detect(point);

      expect(result.detection.confidence).toBeGreaterThanOrEqual(0);
      expect(result.detection.confidence).toBeLessThanOrEqual(1);
    });
  });

  describe('Feedback Learning', () => {
    beforeEach(async () => {
      const trainingData = generateNormalData(100, 2);
      await detector.calibrate(trainingData);
    });

    it('should improve with feedback', async () => {
      const point: AnomalyPoint = {
        timestamp: Date.now(),
        features: [3, 3],
      };

      const result = await detector.detect(point);
      const initialThreshold = detector.getStatistics().currentThreshold;

      // Provide feedback that this was a false positive
      await detector.provideFeedback(result.timestamp, false);

      const newThreshold = detector.getStatistics().currentThreshold;

      // Threshold should adjust (may increase or decrease based on adaptation)
      expect(newThreshold).not.toBe(initialThreshold);
    });
  });

  describe('Streaming Detection', () => {
    beforeEach(async () => {
      const trainingData = generateNormalData(100, 2);
      await detector.calibrate(trainingData);
    });

    it('should handle streaming data efficiently', async () => {
      const streamSize = 200;
      let anomalyCount = 0;

      for (let i = 0; i < streamSize; i++) {
        const point: AnomalyPoint = {
          timestamp: Date.now(),
          features: i % 20 === 0 ? [10, 10] : [Math.random(), Math.random()],
        };

        const result = await detector.detect(point);
        if (result.detection.isAnomaly) {
          anomalyCount++;
        }
      }

      // Should detect injected anomalies
      expect(anomalyCount).toBeGreaterThan(0);
      expect(anomalyCount).toBeLessThan(streamSize * 0.2); // Reasonable anomaly rate
    });
  });

  describe('Statistics', () => {
    it('should track detection statistics', async () => {
      const trainingData = generateNormalData(100, 2);
      await detector.calibrate(trainingData);

      const stats = detector.getStatistics();

      expect(stats.isCalibrated).toBe(true);
      expect(stats.currentThreshold).toBeGreaterThan(0);
      expect(stats.totalDetections).toBeGreaterThanOrEqual(0);
    });
  });
});

// Helper functions
function generateNormalData(count: number, dimensions: number): AnomalyPoint[] {
  return Array.from({ length: count }, () => ({
    timestamp: Date.now(),
    features: Array.from({ length: dimensions }, () => Math.random()),
    label: 'normal' as const,
  }));
}
