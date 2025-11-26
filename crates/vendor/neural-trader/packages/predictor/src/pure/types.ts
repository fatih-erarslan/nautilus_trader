/**
 * Core types for conformal prediction
 */

export interface PredictionInterval {
  /** Point prediction from base model */
  point: number;

  /** Lower bound of prediction interval */
  lower: number;

  /** Upper bound of prediction interval */
  upper: number;

  /** Miscoverage rate (1 - coverage) */
  alpha: number;

  /** Computed quantile threshold */
  quantile: number;

  /** Timestamp of prediction */
  timestamp: number;

  /** Width of the interval */
  width(): number;

  /** Check if value is in interval */
  contains(value: number): boolean;

  /** Relative width as percentage */
  relativeWidth(): number;

  /** Expected coverage (1 - alpha) */
  coverage(): number;
}

export class PredictionIntervalImpl implements PredictionInterval {
  constructor(
    public point: number,
    public lower: number,
    public upper: number,
    public alpha: number,
    public quantile: number,
    public timestamp: number = Date.now()
  ) {}

  width(): number {
    return this.upper - this.lower;
  }

  contains(value: number): boolean {
    return value >= this.lower && value <= this.upper;
  }

  relativeWidth(): number {
    if (Math.abs(this.point) < Number.EPSILON) {
      return Infinity;
    }
    return (this.width() / Math.abs(this.point)) * 100;
  }

  coverage(): number {
    return 1 - this.alpha;
  }
}

export interface PredictorConfig {
  /** Miscoverage rate (e.g., 0.1 for 90% coverage) */
  alpha: number;

  /** Maximum calibration set size */
  calibrationSize?: number;

  /** Maximum interval width as percentage */
  maxIntervalWidthPct?: number;

  /** Recalibration frequency (number of predictions) */
  recalibrationFreq?: number;
}

export interface AdaptiveConfig {
  /** Target coverage (e.g., 0.90 for 90%) */
  targetCoverage: number;

  /** Learning rate for PID control */
  gamma: number;

  /** Window size for coverage tracking */
  coverageWindow?: number;

  /** Minimum alpha value */
  alphaMin?: number;

  /** Maximum alpha value */
  alphaMax?: number;
}

export const defaultPredictorConfig: Required<PredictorConfig> = {
  alpha: 0.1,
  calibrationSize: 2000,
  maxIntervalWidthPct: 5.0,
  recalibrationFreq: 100,
};

export const defaultAdaptiveConfig: Required<AdaptiveConfig> = {
  targetCoverage: 0.90,
  gamma: 0.02,
  coverageWindow: 200,
  alphaMin: 0.01,
  alphaMax: 0.30,
};
