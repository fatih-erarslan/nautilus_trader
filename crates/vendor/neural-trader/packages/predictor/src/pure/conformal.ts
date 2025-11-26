/**
 * Pure TypeScript implementation of conformal prediction algorithms
 * Ports Rust algorithms with efficient sorting and binary search
 */

import {
  PredictionInterval,
  PredictionIntervalImpl,
  PredictorConfig,
  AdaptiveConfig,
  defaultPredictorConfig,
  defaultAdaptiveConfig,
} from './types';
import { NonconformityScore, AbsoluteScore } from './scores';

/**
 * Split Conformal Predictor
 * Provides distribution-free prediction intervals with guaranteed coverage
 *
 * Mathematical guarantee: P(y ∈ [lower, upper]) ≥ 1 - α
 */
export class SplitConformalPredictor {
  private alpha: number;
  private calibrationSize: number;
  private recalibrationFreq: number;
  private scoreFunction: NonconformityScore;

  private calibrationScores: number[] = [];
  private quantile: number = 0;
  private nCalibration: number = 0;
  private predictionCount: number = 0;

  constructor(config: Partial<PredictorConfig> = {}, scoreFunction?: NonconformityScore) {
    const fullConfig = { ...defaultPredictorConfig, ...config };
    this.alpha = fullConfig.alpha;
    this.calibrationSize = fullConfig.calibrationSize;
    // maxIntervalWidthPct from config reserved for future interval width constraints
    void fullConfig.maxIntervalWidthPct;
    this.recalibrationFreq = fullConfig.recalibrationFreq;
    this.scoreFunction = scoreFunction || new AbsoluteScore();
  }

  /**
   * Calibrate the predictor with historical data
   * O(n log n) due to sorting
   *
   * @param predictions - Model's point predictions
   * @param actuals - Actual observed values
   */
  async calibrate(predictions: number[], actuals: number[]): Promise<void> {
    if (predictions.length !== actuals.length) {
      throw new Error('Predictions and actuals must have same length');
    }

    if (predictions.length === 0) {
      throw new Error('At least one calibration sample required');
    }

    // Compute nonconformity scores
    const scores: number[] = [];
    for (let i = 0; i < predictions.length; i++) {
      const score = this.scoreFunction.score(predictions[i], actuals[i]);
      scores.push(score);
    }

    // Sort scores for quantile computation
    scores.sort((a, b) => a - b);

    this.calibrationScores = scores;
    this.nCalibration = scores.length;
    this.updateQuantile();
  }

  /**
   * Make a prediction with a confidence interval
   * O(1) time after calibration
   *
   * @param pointPrediction - Model's point prediction
   * @returns PredictionInterval with bounds
   */
  predict(pointPrediction: number): PredictionInterval {
    if (this.nCalibration === 0) {
      throw new Error('Predictor not calibrated');
    }

    const [lower, upper] = this.scoreFunction.interval(pointPrediction, this.quantile);

    const interval = new PredictionIntervalImpl(
      pointPrediction,
      lower,
      upper,
      this.alpha,
      this.quantile
    );

    this.predictionCount++;
    return interval;
  }

  /**
   * Update predictor with new observation
   * O(log n) via binary search insertion
   *
   * @param prediction - Model's point prediction
   * @param actual - Actual observed value
   */
  async update(prediction: number, actual: number): Promise<void> {
    const score = this.scoreFunction.score(prediction, actual);

    // Binary search for insertion point
    const insertPos = this.binarySearchInsertPosition(score);
    this.calibrationScores.splice(insertPos, 0, score);

    // Maintain maximum window size
    if (this.calibrationScores.length > this.calibrationSize) {
      this.calibrationScores.shift();
    }

    this.nCalibration = this.calibrationScores.length;
    this.updateQuantile();
  }

  /**
   * Trigger full recalibration if needed
   */
  async recalibrate(predictions: number[], actuals: number[]): Promise<void> {
    if (this.predictionCount % this.recalibrationFreq === 0) {
      await this.calibrate(predictions, actuals);
      this.predictionCount = 0;
    }
  }

  /**
   * Get empirical coverage from calibration set
   */
  getEmpiricalCoverage(predictions: number[], actuals: number[]): number {
    if (predictions.length === 0) return 0;

    let covered = 0;
    for (let i = 0; i < predictions.length; i++) {
      const interval = this.predict(predictions[i]);
      if (interval.contains(actuals[i])) {
        covered++;
      }
    }

    return covered / predictions.length;
  }

  /**
   * Get calibration statistics
   */
  getStats() {
    return {
      nCalibration: this.nCalibration,
      alpha: this.alpha,
      quantile: this.quantile,
      predictionCount: this.predictionCount,
      minScore: this.calibrationScores[0] ?? 0,
      maxScore: this.calibrationScores[this.nCalibration - 1] ?? 0,
    };
  }

  /**
   * Update the quantile threshold based on sorted scores
   * Follows: q = ceil((n+1)(1-alpha))/n
   * @private
   */
  private updateQuantile(): void {
    if (this.nCalibration === 0) {
      this.quantile = 0;
      return;
    }

    // Compute quantile index: ceil((n+1)(1-alpha))/n
    const index = Math.ceil((this.nCalibration + 1) * (1 - this.alpha)) - 1;
    const clampedIndex = Math.max(0, Math.min(index, this.nCalibration - 1));
    this.quantile = this.calibrationScores[clampedIndex];
  }

  /**
   * Find binary search insertion position
   * @private
   */
  private binarySearchInsertPosition(score: number): number {
    let left = 0;
    let right = this.calibrationScores.length;

    while (left < right) {
      const mid = Math.floor((left + right) / 2);
      if (this.calibrationScores[mid] < score) {
        left = mid + 1;
      } else {
        right = mid;
      }
    }

    return left;
  }
}

/**
 * Adaptive Conformal Inference (ACI)
 * Dynamically adjusts alpha using PID control to track target coverage
 *
 * Maintains empirical coverage close to target by adapting alpha during streaming
 */
export class AdaptiveConformalPredictor {
  private targetCoverage: number;
  private gamma: number;
  private coverageWindow: number;
  private alphaMin: number;
  private alphaMax: number;

  private basePredictorConfig: Partial<PredictorConfig>;
  private basePredictor: SplitConformalPredictor;
  private scoreFunction: NonconformityScore;

  private coverageHistory: number[] = [];
  private alphaCurrent: number;

  constructor(
    config: Partial<AdaptiveConfig> = {},
    scoreFunction?: NonconformityScore
  ) {
    const fullConfig = { ...defaultAdaptiveConfig, ...config };
    this.targetCoverage = fullConfig.targetCoverage;
    this.gamma = fullConfig.gamma;
    this.coverageWindow = fullConfig.coverageWindow;
    this.alphaMin = fullConfig.alphaMin;
    this.alphaMax = fullConfig.alphaMax;

    this.scoreFunction = scoreFunction || new AbsoluteScore();
    this.alphaCurrent = 1 - this.targetCoverage;

    this.basePredictorConfig = {
      alpha: this.alphaCurrent,
    };

    this.basePredictor = new SplitConformalPredictor(
      this.basePredictorConfig,
      this.scoreFunction
    );
  }

  /**
   * Initialize with calibration data
   *
   * @param predictions - Initial predictions for calibration
   * @param actuals - Actual values for calibration
   */
  async calibrate(predictions: number[], actuals: number[]): Promise<void> {
    await this.basePredictor.calibrate(predictions, actuals);
  }

  /**
   * Make prediction and adapt alpha based on coverage
   * O(log n) with binary search
   *
   * @param pointPrediction - Model's point prediction
   * @param actual - Optional actual value for adaptation
   * @returns PredictionInterval
   */
  async predictAndAdapt(pointPrediction: number, actual?: number): Promise<PredictionInterval> {
    // Make prediction with current alpha
    const interval = this.basePredictor.predict(pointPrediction);

    // If actual is provided, adapt alpha
    if (actual !== undefined) {
      const covered = interval.contains(actual) ? 1 : 0;
      this.coverageHistory.push(covered);

      // Maintain window size
      if (this.coverageHistory.length > this.coverageWindow) {
        this.coverageHistory.shift();
      }

      // PID control: adjust alpha based on coverage error
      const empirical = this.empiricalCoverage();
      const error = this.targetCoverage - empirical;

      // Update alpha: alpha -= gamma * error
      this.alphaCurrent -= this.gamma * error;

      // Clamp alpha to valid range
      this.alphaCurrent = Math.max(this.alphaMin, Math.min(this.alphaMax, this.alphaCurrent));

      // Update base predictor's alpha
      const updatedConfig = { ...this.basePredictorConfig, alpha: this.alphaCurrent };
      this.basePredictor = new SplitConformalPredictor(updatedConfig, this.scoreFunction);

      // Update base predictor with new observation
      await this.basePredictor.update(pointPrediction, actual);
    }

    return interval;
  }

  /**
   * Standard prediction without adaptation
   *
   * @param pointPrediction - Model's point prediction
   * @returns PredictionInterval
   */
  predict(pointPrediction: number): PredictionInterval {
    return this.basePredictor.predict(pointPrediction);
  }

  /**
   * Update predictor with new observation
   *
   * @param prediction - Model's point prediction
   * @param actual - Actual observed value
   */
  async update(prediction: number, actual: number): Promise<void> {
    await this.basePredictor.update(prediction, actual);
  }

  /**
   * Compute empirical coverage from history
   * Simple average of coverage indicator in the window
   */
  empiricalCoverage(): number {
    if (this.coverageHistory.length === 0) {
      return this.targetCoverage; // Default to target if no history
    }

    const sum = this.coverageHistory.reduce((a, b) => a + b, 0);
    return sum / this.coverageHistory.length;
  }

  /**
   * Get current alpha value
   */
  getCurrentAlpha(): number {
    return this.alphaCurrent;
  }

  /**
   * Get statistics including coverage metrics
   */
  getStats() {
    const empirical = this.empiricalCoverage();
    return {
      ...this.basePredictor.getStats(),
      alphaCurrent: this.alphaCurrent,
      empiricalCoverage: empirical,
      targetCoverage: this.targetCoverage,
      coverageDifference: this.targetCoverage - empirical,
      coverageHistorySize: this.coverageHistory.length,
    };
  }
}

/**
 * Conformalized Quantile Regression (CQR) Predictor
 * Uses quantile predictions from model for prediction intervals
 */
export class CQRPredictor {
  private alpha: number;
  private calibrationSize: number;
  // @ts-ignore scoreFunction reserved for future quantile-based extensions
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  private scoreFunction: NonconformityScore;

  private calibrationScores: number[] = [];
  private quantile: number = 0;
  private nCalibration: number = 0;

  private alphaLow: number;
  private alphaHigh: number;

  constructor(
    config: Partial<PredictorConfig> = {},
    alphaLow: number = 0.05,
    alphaHigh: number = 0.95,
    scoreFunction?: NonconformityScore
  ) {
    if (alphaLow < 0 || alphaLow >= alphaHigh || alphaHigh > 1) {
      throw new Error('Invalid quantile values');
    }

    const fullConfig = { ...defaultPredictorConfig, ...config };
    this.alpha = fullConfig.alpha;
    this.calibrationSize = fullConfig.calibrationSize;
    // scoreFunction reserved for future quantile-based extensions
    this.scoreFunction = scoreFunction || new AbsoluteScore();
    this.alphaLow = alphaLow;
    this.alphaHigh = alphaHigh;
  }

  /**
   * Calibrate with quantile predictions
   *
   * @param qLow - Lower quantile predictions
   * @param qHigh - Upper quantile predictions
   * @param actuals - Actual observed values
   */
  async calibrate(qLow: number[], qHigh: number[], actuals: number[]): Promise<void> {
    if (qLow.length !== qHigh.length || qLow.length !== actuals.length) {
      throw new Error('All arrays must have same length');
    }

    if (qLow.length === 0) {
      throw new Error('At least one calibration sample required');
    }

    // Compute nonconformity scores for quantile predictions
    const scores: number[] = [];
    for (let i = 0; i < qLow.length; i++) {
      const score = Math.max(qLow[i] - actuals[i], actuals[i] - qHigh[i]);
      scores.push(score);
    }

    // Sort for quantile
    scores.sort((a, b) => a - b);
    this.calibrationScores = scores;
    this.nCalibration = scores.length;
    this.updateQuantile();
  }

  /**
   * Make CQR prediction with adjusted quantile bounds
   *
   * @param qLow - Lower quantile prediction from model
   * @param qHigh - Upper quantile prediction from model
   * @returns PredictionInterval with adjusted bounds
   */
  predict(qLow: number, qHigh: number): PredictionInterval {
    if (this.nCalibration === 0) {
      throw new Error('Predictor not calibrated');
    }

    // Adjust quantiles by computed quantile threshold
    const lower = qLow - this.quantile;
    const upper = qHigh + this.quantile;
    const point = (qLow + qHigh) / 2;

    return new PredictionIntervalImpl(point, lower, upper, this.alpha, this.quantile);
  }

  /**
   * Update with new observation
   *
   * @param qLow - Lower quantile prediction
   * @param qHigh - Upper quantile prediction
   * @param actual - Actual observed value
   */
  async update(qLow: number, qHigh: number, actual: number): Promise<void> {
    const score = Math.max(qLow - actual, actual - qHigh);
    const insertPos = this.binarySearchInsertPosition(score);
    this.calibrationScores.splice(insertPos, 0, score);

    if (this.calibrationScores.length > this.calibrationSize) {
      this.calibrationScores.shift();
    }

    this.nCalibration = this.calibrationScores.length;
    this.updateQuantile();
  }

  /**
   * Get statistics
   */
  getStats() {
    return {
      nCalibration: this.nCalibration,
      alpha: this.alpha,
      alphaLow: this.alphaLow,
      alphaHigh: this.alphaHigh,
      quantile: this.quantile,
      minScore: this.calibrationScores[0] ?? 0,
      maxScore: this.calibrationScores[this.nCalibration - 1] ?? 0,
    };
  }

  /**
   * Update quantile threshold
   * @private
   */
  private updateQuantile(): void {
    if (this.nCalibration === 0) {
      this.quantile = 0;
      return;
    }

    const index = Math.ceil((this.nCalibration + 1) * (1 - this.alpha)) - 1;
    const clampedIndex = Math.max(0, Math.min(index, this.nCalibration - 1));
    this.quantile = this.calibrationScores[clampedIndex];
  }

  /**
   * Binary search insertion position
   * @private
   */
  private binarySearchInsertPosition(score: number): number {
    let left = 0;
    let right = this.calibrationScores.length;

    while (left < right) {
      const mid = Math.floor((left + right) / 2);
      if (this.calibrationScores[mid] < score) {
        left = mid + 1;
      } else {
        right = mid;
      }
    }

    return left;
  }
}
