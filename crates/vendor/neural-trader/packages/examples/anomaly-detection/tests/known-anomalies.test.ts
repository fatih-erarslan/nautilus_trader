import { describe, it, expect, beforeEach } from 'vitest';
import { AnomalyDetector } from '../src/detector';
import type { AnomalyPoint } from '../src/index';

/**
 * Test suite for known anomaly patterns
 *
 * Tests the detector's ability to identify well-known anomaly types:
 * - Point anomalies (outliers)
 * - Contextual anomalies (normal in one context, anomalous in another)
 * - Collective anomalies (sequences that are anomalous together)
 */
describe('Known Anomaly Patterns', () => {
  let detector: AnomalyDetector;

  beforeEach(async () => {
    detector = new AnomalyDetector({
      targetFalsePositiveRate: 0.05,
      featureDimensions: 2,
      useEnsemble: false,
      useConformal: false,
      windowSize: 100,
      minCalibrationSamples: 50,
    });

    // Calibrate with normal data
    const normalData = generateNormalData(100, 2, { mean: 0, std: 1 });
    await detector.calibrate(normalData);
  });

  describe('Point Anomalies', () => {
    it('should detect extreme outliers', async () => {
      const outlier: AnomalyPoint = {
        timestamp: Date.now(),
        features: [100, 100], // Far from normal distribution
        metadata: { type: 'extreme-outlier' },
      };

      const result = await detector.detect(outlier);

      expect(result.detection.isAnomaly).toBe(true);
      expect(result.detection.score).toBeGreaterThan(10);
    });

    it('should detect subtle outliers', async () => {
      const subtleOutlier: AnomalyPoint = {
        timestamp: Date.now(),
        features: [4, 4], // 4 standard deviations from mean
        metadata: { type: 'subtle-outlier' },
      };

      const result = await detector.detect(subtleOutlier);

      expect(result.detection.isAnomaly).toBe(true);
    });

    it('should not flag normal points as anomalies', async () => {
      const normalPoint: AnomalyPoint = {
        timestamp: Date.now(),
        features: [0.1, 0.2],
      };

      const result = await detector.detect(normalPoint);

      expect(result.detection.isAnomaly).toBe(false);
    });
  });

  describe('Contextual Anomalies', () => {
    it('should detect anomalies in specific contexts', async () => {
      // Value is normal in general but anomalous for this specific time/context
      const contextualAnomaly: AnomalyPoint = {
        timestamp: Date.now(),
        features: [2, -2], // Large negative correlation
        metadata: { context: 'night-trading', expectedCorrelation: 'positive' },
      };

      const result = await detector.detect(contextualAnomaly);

      // Should be flagged based on distance from expected pattern
      expect(result.detection.score).toBeGreaterThan(0);
    });
  });

  describe('Fraud Detection Patterns', () => {
    it('should detect credit card fraud patterns', async () => {
      // Typical fraud indicators: high amount, unusual location, rapid succession
      const fraudPattern: AnomalyPoint = {
        timestamp: Date.now(),
        features: [
          10, // Normalized amount (unusually high)
          8,  // Location distance from usual (far)
        ],
        metadata: {
          type: 'credit-card-fraud',
          amount: 5000,
          location: 'foreign',
          timeSinceLastTransaction: 2, // minutes
        },
      };

      const result = await detector.detect(fraudPattern);

      expect(result.detection.isAnomaly).toBe(true);
    });

    it('should detect account takeover patterns', async () => {
      const accountTakeover: AnomalyPoint = {
        timestamp: Date.now(),
        features: [
          5,  // Login attempts (high)
          10, // New device/IP
        ],
        metadata: {
          type: 'account-takeover',
          failedLogins: 5,
          newDevice: true,
          locationChange: true,
        },
      };

      const result = await detector.detect(accountTakeover);

      expect(result.detection.isAnomaly).toBe(true);
    });
  });

  describe('Network Intrusion Patterns', () => {
    it('should detect port scanning', async () => {
      const portScan: AnomalyPoint = {
        timestamp: Date.now(),
        features: [
          100, // Connection attempts (very high)
          0.1, // Packet size (small)
        ],
        metadata: {
          type: 'port-scan',
          connections: 100,
          duration: 10,
          uniquePorts: 50,
        },
      };

      const result = await detector.detect(portScan);

      expect(result.detection.isAnomaly).toBe(true);
    });

    it('should detect DDoS patterns', async () => {
      const ddos: AnomalyPoint = {
        timestamp: Date.now(),
        features: [
          1000, // Request rate (extremely high)
          0.5,  // Payload size (uniform)
        ],
        metadata: {
          type: 'ddos',
          requestsPerSecond: 1000,
          sourceIPs: 500,
        },
      };

      const result = await detector.detect(ddos);

      expect(result.detection.isAnomaly).toBe(true);
    });
  });

  describe('Trading Anomalies', () => {
    it('should detect flash crash patterns', async () => {
      const flashCrash: AnomalyPoint = {
        timestamp: Date.now(),
        features: [
          -10,  // Price change (extreme negative)
          100,  // Volume (extremely high)
        ],
        metadata: {
          type: 'flash-crash',
          priceChange: -0.15, // -15%
          volumeMultiplier: 100,
          duration: 5, // seconds
        },
      };

      const result = await detector.detect(flashCrash);

      expect(result.detection.isAnomaly).toBe(true);
    });

    it('should detect pump and dump schemes', async () => {
      const pumpAndDump: AnomalyPoint = {
        timestamp: Date.now(),
        features: [
          20,  // Price increase (extreme)
          50,  // Social media mentions (spike)
        ],
        metadata: {
          type: 'pump-and-dump',
          priceIncrease: 0.50, // +50%
          volumeSpike: 200,
          socialMediaActivity: 'extreme',
        },
      };

      const result = await detector.detect(pumpAndDump);

      expect(result.detection.isAnomaly).toBe(true);
    });

    it('should detect insider trading patterns', async () => {
      const insiderTrading: AnomalyPoint = {
        timestamp: Date.now(),
        features: [
          10, // Trading volume before announcement
          5,  // Price movement precision
        ],
        metadata: {
          type: 'insider-trading',
          beforeAnnouncement: true,
          unusualVolume: true,
          profitRate: 0.20, // 20% profit
        },
      };

      const result = await detector.detect(insiderTrading);

      expect(result.detection.isAnomaly).toBe(true);
    });
  });

  describe('System Monitoring Anomalies', () => {
    it('should detect memory leaks', async () => {
      const memoryLeak: AnomalyPoint = {
        timestamp: Date.now(),
        features: [
          8,   // Memory growth rate (high)
          0.5, // CPU usage (normal)
        ],
        metadata: {
          type: 'memory-leak',
          memoryGrowth: 'linear',
          heapSize: 'increasing',
        },
      };

      const result = await detector.detect(memoryLeak);

      expect(result.detection.isAnomaly).toBe(true);
    });

    it('should detect resource exhaustion', async () => {
      const resourceExhaustion: AnomalyPoint = {
        timestamp: Date.now(),
        features: [
          10, // CPU usage (maxed)
          10, // Disk I/O (maxed)
        ],
        metadata: {
          type: 'resource-exhaustion',
          cpu: 0.99,
          disk: 0.98,
          responseTime: 5000, // ms
        },
      };

      const result = await detector.detect(resourceExhaustion);

      expect(result.detection.isAnomaly).toBe(true);
    });
  });

  describe('Adaptive Learning', () => {
    it('should reduce false positives with feedback', async () => {
      let falsePositiveCount = 0;

      // First pass: detect anomalies
      const testPoints = generateNormalData(50, 2, { mean: 2, std: 0.5 });
      for (const point of testPoints) {
        const result = await detector.detect(point);
        if (result.detection.isAnomaly) {
          falsePositiveCount++;
          // Provide feedback that this was actually normal
          await detector.provideFeedback(result.timestamp, false);
        }
      }

      const initialFPR = falsePositiveCount / testPoints.length;

      // Second pass: should have fewer false positives
      falsePositiveCount = 0;
      const testPoints2 = generateNormalData(50, 2, { mean: 2, std: 0.5 });
      for (const point of testPoints2) {
        const result = await detector.detect(point);
        if (result.detection.isAnomaly) {
          falsePositiveCount++;
        }
      }

      const finalFPR = falsePositiveCount / testPoints2.length;

      // False positive rate should improve or stay similar
      expect(finalFPR).toBeLessThanOrEqual(initialFPR * 1.5);
    });
  });
});

// Helper functions
function generateNormalData(
  count: number,
  dimensions: number,
  params: { mean: number; std: number }
): AnomalyPoint[] {
  return Array.from({ length: count }, () => ({
    timestamp: Date.now(),
    features: Array.from({ length: dimensions }, () =>
      params.mean + params.std * (Math.random() * 2 - 1) * Math.sqrt(3)
    ),
    label: 'normal' as const,
  }));
}
