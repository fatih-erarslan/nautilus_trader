/**
 * Conformal prediction for demand uncertainty quantification
 */

import { ConformalPrediction } from './types';

export class ConformalPredictor {
  private calibrationScores: number[];
  private alpha: number; // Significance level (e.g., 0.1 for 90% coverage)

  constructor(alpha: number = 0.1) {
    this.alpha = alpha;
    this.calibrationScores = [];
  }

  /**
   * Calibrate predictor with historical data
   */
  calibrate(predictions: number[], actuals: number[]): void {
    if (predictions.length !== actuals.length) {
      throw new Error('Predictions and actuals must have same length');
    }

    // Calculate non-conformity scores (absolute residuals)
    this.calibrationScores = predictions.map((pred, i) => Math.abs(pred - actuals[i]));
    this.calibrationScores.sort((a, b) => a - b);
  }

  /**
   * Make conformal prediction with uncertainty bounds
   */
  predict(pointPrediction: number): ConformalPrediction {
    if (this.calibrationScores.length === 0) {
      // No calibration data: use heuristic bounds
      const defaultMargin = pointPrediction * 0.2; // 20% margin
      return {
        point: pointPrediction,
        lower: Math.max(0, pointPrediction - defaultMargin),
        upper: pointPrediction + defaultMargin,
        coverage: 1 - this.alpha,
      };
    }

    // Calculate quantile of calibration scores
    const n = this.calibrationScores.length;
    const quantileIdx = Math.ceil((n + 1) * (1 - this.alpha)) - 1;
    const conformalScore = this.calibrationScores[Math.min(quantileIdx, n - 1)];

    // Prediction interval
    const lower = Math.max(0, pointPrediction - conformalScore);
    const upper = pointPrediction + conformalScore;

    return {
      point: pointPrediction,
      lower,
      upper,
      coverage: 1 - this.alpha,
    };
  }

  /**
   * Adaptive conformal prediction (updates as new data arrives)
   */
  adaptivePredict(
    pointPrediction: number,
    recentPredictions: number[],
    recentActuals: number[]
  ): ConformalPrediction {
    // Use only recent data for calibration (more adaptive)
    if (recentPredictions.length >= 30) {
      this.calibrate(recentPredictions.slice(-30), recentActuals.slice(-30));
    }

    return this.predict(pointPrediction);
  }

  /**
   * Multi-step ahead conformal prediction
   */
  multiStepPredict(pointPredictions: number[], horizon: number): ConformalPrediction[] {
    return pointPredictions.map((pred, step) => {
      // Uncertainty typically increases with horizon
      const horizonFactor = 1 + step * 0.1;
      const adjusted = this.predict(pred);

      return {
        point: adjusted.point,
        lower: adjusted.point - (adjusted.point - adjusted.lower) * horizonFactor,
        upper: adjusted.point + (adjusted.upper - adjusted.point) * horizonFactor,
        coverage: adjusted.coverage * Math.pow(0.95, step),
      };
    });
  }

  /**
   * Check if actual value falls within prediction interval
   */
  isValid(prediction: ConformalPrediction, actual: number): boolean {
    return actual >= prediction.lower && actual <= prediction.upper;
  }

  /**
   * Calculate empirical coverage
   */
  calculateCoverage(predictions: ConformalPrediction[], actuals: number[]): number {
    if (predictions.length !== actuals.length || predictions.length === 0) {
      return 0;
    }

    const covered = predictions.filter((pred, i) => this.isValid(pred, actuals[i])).length;
    return covered / predictions.length;
  }

  /**
   * Get calibration statistics
   */
  getCalibrationStats(): {
    numSamples: number;
    medianScore: number;
    quantile95: number;
    quantile90: number;
  } {
    if (this.calibrationScores.length === 0) {
      return {
        numSamples: 0,
        medianScore: 0,
        quantile95: 0,
        quantile90: 0,
      };
    }

    const n = this.calibrationScores.length;
    const sorted = [...this.calibrationScores].sort((a, b) => a - b);

    return {
      numSamples: n,
      medianScore: sorted[Math.floor(n / 2)],
      quantile95: sorted[Math.floor(n * 0.95)],
      quantile90: sorted[Math.floor(n * 0.90)],
    };
  }
}
