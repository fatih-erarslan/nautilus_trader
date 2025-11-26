/**
 * Nonconformity score functions
 */

export interface NonconformityScore {
  /** Compute nonconformity score */
  score(prediction: number, actual: number): number;

  /** Compute prediction interval given quantile */
  interval(prediction: number, quantile: number): [number, number];
}

/**
 * Absolute residual score: |actual - prediction|
 */
export class AbsoluteScore implements NonconformityScore {
  score(prediction: number, actual: number): number {
    return Math.abs(actual - prediction);
  }

  interval(prediction: number, quantile: number): [number, number] {
    return [prediction - quantile, prediction + quantile];
  }
}

/**
 * Normalized score: residual divided by model uncertainty
 */
export class NormalizedScore implements NonconformityScore {
  constructor(private stdDev: number = 1.0) {}

  score(prediction: number, actual: number): number {
    return Math.abs(actual - prediction) / Math.max(this.stdDev, 1e-6);
  }

  interval(prediction: number, quantile: number): [number, number] {
    const width = quantile * this.stdDev;
    return [prediction - width, prediction + width];
  }

  /** Update standard deviation estimate */
  updateStdDev(stdDev: number): void {
    this.stdDev = Math.max(stdDev, 1e-6);
  }
}

/**
 * Quantile-based score for CQR
 */
export class QuantileScore implements NonconformityScore {
  constructor(alphaLow: number = 0.05, alphaHigh: number = 0.95) {
    if (alphaLow < 0 || alphaLow >= alphaHigh || alphaHigh > 1) {
      throw new Error('Invalid quantile values');
    }
    // Validation complete - parameters validated but not stored as they're only used for validation
  }

  score(prediction: number, actual: number): number {
    // For CQR, prediction should be [lower, upper] quantiles
    // But for compatibility, we use symmetric assumption
    const half = prediction * 0.05; // Approximate quantile range
    return Math.max(prediction - half - actual, actual - (prediction + half));
  }

  interval(prediction: number, quantile: number): [number, number] {
    return [prediction - quantile, prediction + quantile];
  }

  /**
   * Compute score for quantile predictions
   */
  scoreQuantiles(qLow: number, qHigh: number, actual: number): number {
    return Math.max(qLow - actual, actual - qHigh);
  }
}
