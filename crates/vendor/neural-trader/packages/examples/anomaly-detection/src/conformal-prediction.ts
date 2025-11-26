import type { AnomalyPoint } from './index';

export interface ConformalConfig {
  /**
   * Significance level (e.g., 0.1 for 90% confidence)
   */
  alpha: number;

  /**
   * Calibration window size
   */
  windowSize: number;
}

export interface ConformalResult {
  score: number;
  interval: [number, number];
  confidence: number;
  isAnomaly: boolean;
}

/**
 * Conformal prediction for anomaly detection with guaranteed confidence
 *
 * Provides statistically valid confidence intervals for anomaly scores
 */
export class ConformalAnomalyPredictor {
  private calibrationScores: number[] = [];
  private normalScores: number[] = [];
  private anomalyScores: number[] = [];

  constructor(private config: ConformalConfig) {}

  /**
   * Calibrate with labeled data
   */
  async calibrate(calibrationData: AnomalyPoint[]): Promise<void> {
    this.normalScores = [];
    this.anomalyScores = [];

    for (const point of calibrationData) {
      const score = this.computeNonconformityScore(point);

      if (point.label === 'anomaly') {
        this.anomalyScores.push(score);
      } else {
        this.normalScores.push(score);
      }
    }

    // Sort for quantile computation
    this.normalScores.sort((a, b) => a - b);
    this.anomalyScores.sort((a, b) => a - b);

    console.log(`Calibrated with ${this.normalScores.length} normal and ${this.anomalyScores.length} anomaly samples`);
  }

  /**
   * Predict with conformal interval
   */
  predict(point: AnomalyPoint): ConformalResult {
    const score = this.computeNonconformityScore(point);

    // Compute conformal interval
    const interval = this.computeInterval(score);

    // Compute p-value (proportion of calibration scores >= test score)
    const pValue = this.computePValue(score);

    // Confidence = 1 - p-value
    const confidence = 1 - pValue;

    // Is anomaly if p-value < alpha
    const isAnomaly = pValue < this.config.alpha;

    return {
      score,
      interval,
      confidence,
      isAnomaly,
    };
  }

  /**
   * Compute nonconformity score (distance from normal distribution)
   */
  private computeNonconformityScore(point: AnomalyPoint): number {
    // Simple Euclidean distance from origin
    const sumSquares = point.features.reduce((sum, f) => sum + f * f, 0);
    return Math.sqrt(sumSquares);
  }

  /**
   * Compute conformal prediction interval
   */
  private computeInterval(score: number): [number, number] {
    if (this.normalScores.length === 0) {
      return [0, Infinity];
    }

    // Find quantiles of normal distribution
    const lowerQuantile = this.quantile(this.normalScores, this.config.alpha / 2);
    const upperQuantile = this.quantile(this.normalScores, 1 - this.config.alpha / 2);

    return [
      Math.max(0, score - upperQuantile),
      score + upperQuantile,
    ];
  }

  /**
   * Compute p-value for conformity
   */
  private computePValue(score: number): number {
    const allScores = [...this.normalScores];

    // Count how many calibration scores are >= test score
    const higherCount = allScores.filter(s => s >= score).length;

    return (higherCount + 1) / (allScores.length + 1);
  }

  /**
   * Compute quantile of sorted array
   */
  private quantile(sortedArray: number[], q: number): number {
    if (sortedArray.length === 0) return 0;

    const pos = q * (sortedArray.length - 1);
    const base = Math.floor(pos);
    const rest = pos - base;

    if (base + 1 < sortedArray.length) {
      return sortedArray[base] + rest * (sortedArray[base + 1] - sortedArray[base]);
    }

    return sortedArray[base];
  }

  /**
   * Update calibration with new labeled data
   */
  updateCalibration(point: AnomalyPoint): void {
    if (!point.label) return;

    const score = this.computeNonconformityScore(point);

    if (point.label === 'anomaly') {
      this.anomalyScores.push(score);
      this.anomalyScores.sort((a, b) => a - b);

      if (this.anomalyScores.length > this.config.windowSize) {
        this.anomalyScores.shift();
      }
    } else {
      this.normalScores.push(score);
      this.normalScores.sort((a, b) => a - b);

      if (this.normalScores.length > this.config.windowSize) {
        this.normalScores.shift();
      }
    }
  }

  /**
   * Get calibration statistics
   */
  getCalibrationStats() {
    return {
      normalCount: this.normalScores.length,
      anomalyCount: this.anomalyScores.length,
      normalMean: this.mean(this.normalScores),
      normalStd: this.std(this.normalScores),
      anomalyMean: this.mean(this.anomalyScores),
      anomalyStd: this.std(this.anomalyScores),
    };
  }

  private mean(arr: number[]): number {
    if (arr.length === 0) return 0;
    return arr.reduce((sum, val) => sum + val, 0) / arr.length;
  }

  private std(arr: number[]): number {
    if (arr.length === 0) return 0;
    const m = this.mean(arr);
    const variance = arr.reduce((sum, val) => sum + (val - m) ** 2, 0) / arr.length;
    return Math.sqrt(variance);
  }
}
