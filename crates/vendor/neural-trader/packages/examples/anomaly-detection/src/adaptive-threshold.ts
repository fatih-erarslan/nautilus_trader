/**
 * Adaptive threshold learning based on false positive rate
 *
 * Automatically adjusts detection thresholds to maintain target false positive rate
 * while maximizing true positive detection.
 */

export interface ThresholdConfig {
  /**
   * Target false positive rate (0-1)
   */
  targetFalsePositiveRate: number;

  /**
   * Sliding window size for computing metrics
   */
  windowSize: number;

  /**
   * Learning rate for threshold adaptation (0-1)
   */
  adaptationRate: number;

  /**
   * Initial threshold value
   */
  initialThreshold?: number;

  /**
   * Minimum threshold value
   */
  minThreshold?: number;

  /**
   * Maximum threshold value
   */
  maxThreshold?: number;
}

interface DetectionRecord {
  score: number;
  predicted: boolean;
  actual?: boolean;
  timestamp: number;
}

/**
 * Self-learning adaptive threshold system
 */
export class AdaptiveThreshold {
  private threshold: number;
  private detectionHistory: DetectionRecord[] = [];
  private falsePositiveCount = 0;
  private totalPositives = 0;

  constructor(private config: ThresholdConfig) {
    this.threshold = config.initialThreshold ?? 1.0;
  }

  /**
   * Update threshold based on new detection
   */
  update(score: number, isActualAnomaly?: boolean, weight = 1.0): void {
    const predicted = score > this.threshold;

    const record: DetectionRecord = {
      score,
      predicted,
      actual: isActualAnomaly,
      timestamp: Date.now(),
    };

    this.detectionHistory.push(record);

    // Maintain sliding window
    if (this.detectionHistory.length > this.config.windowSize) {
      const removed = this.detectionHistory.shift()!;
      if (removed.actual !== undefined && removed.predicted && !removed.actual) {
        this.falsePositiveCount--;
      }
      if (removed.predicted) {
        this.totalPositives--;
      }
    }

    // Update counts
    if (predicted) {
      this.totalPositives++;
    }

    if (isActualAnomaly !== undefined) {
      if (predicted && !isActualAnomaly) {
        this.falsePositiveCount++;
      }

      // Adapt threshold
      this.adaptThreshold(weight);
    }
  }

  /**
   * Adapt threshold to maintain target false positive rate
   */
  private adaptThreshold(weight: number): void {
    if (this.totalPositives === 0) return;

    const currentFPR = this.falsePositiveCount / this.totalPositives;
    const targetFPR = this.config.targetFalsePositiveRate;

    // Compute adjustment
    const error = currentFPR - targetFPR;
    const adjustment = error * this.config.adaptationRate * weight;

    // If FPR too high, increase threshold (be stricter)
    // If FPR too low, decrease threshold (be more sensitive)
    this.threshold += adjustment;

    // Apply bounds
    if (this.config.minThreshold !== undefined) {
      this.threshold = Math.max(this.threshold, this.config.minThreshold);
    }
    if (this.config.maxThreshold !== undefined) {
      this.threshold = Math.min(this.threshold, this.config.maxThreshold);
    }
  }

  /**
   * Get current threshold value
   */
  getThreshold(): number {
    return this.threshold;
  }

  /**
   * Get current false positive rate
   */
  getFalsePositiveRate(): number {
    if (this.totalPositives === 0) return 0;
    return this.falsePositiveCount / this.totalPositives;
  }

  /**
   * Get precision (1 - FPR) for labeled data
   */
  getPrecision(): number {
    const labeledDetections = this.detectionHistory.filter(r =>
      r.actual !== undefined && r.predicted
    );

    if (labeledDetections.length === 0) return 0;

    const truePositives = labeledDetections.filter(r => r.actual === true).length;
    return truePositives / labeledDetections.length;
  }

  /**
   * Get recall (true positive rate) for labeled data
   */
  getRecall(): number {
    const labeledActualAnomalies = this.detectionHistory.filter(r =>
      r.actual === true
    );

    if (labeledActualAnomalies.length === 0) return 0;

    const truePositives = labeledActualAnomalies.filter(r => r.predicted).length;
    return truePositives / labeledActualAnomalies.length;
  }

  /**
   * Get F1 score (harmonic mean of precision and recall)
   */
  getF1Score(): number {
    const precision = this.getPrecision();
    const recall = this.getRecall();

    if (precision + recall === 0) return 0;
    return 2 * (precision * recall) / (precision + recall);
  }

  /**
   * Get comprehensive metrics
   */
  getMetrics() {
    return {
      threshold: this.threshold,
      falsePositiveRate: this.getFalsePositiveRate(),
      precision: this.getPrecision(),
      recall: this.getRecall(),
      f1Score: this.getF1Score(),
      totalDetections: this.detectionHistory.length,
      totalPositives: this.totalPositives,
      falsePositives: this.falsePositiveCount,
    };
  }

  /**
   * Reset threshold and history
   */
  reset(newThreshold?: number): void {
    this.threshold = newThreshold ?? this.config.initialThreshold ?? 1.0;
    this.detectionHistory = [];
    this.falsePositiveCount = 0;
    this.totalPositives = 0;
  }
}
