/**
 * Conformal prediction wrapper for energy forecasting
 * Provides uncertainty quantification with guaranteed coverage
 */

import {
  SplitConformalPredictor,
  AdaptiveConformalPredictor,
  createPredictor,
  createAdaptivePredictor,
  PredictionInterval,
  type PredictorConfig,
  type AdaptiveConfig
} from '@neural-trader/predictor';
import { TimeSeriesPoint, ForecastResult, ModelType } from './types';

/**
 * Energy forecasting conformal predictor
 * Wraps @neural-trader/predictor for time series forecasting
 */
export class EnergyConformalPredictor {
  private predictor: SplitConformalPredictor | AdaptiveConformalPredictor | null = null;
  private implementationType: string = 'pure';
  private isAdaptive: boolean;
  private modelType: ModelType;
  private calibrationHistory: Array<{ prediction: number; actual: number; timestamp: number }> = [];

  constructor(
    modelType: ModelType,
    config: Partial<PredictorConfig> = {},
    adaptive: boolean = false
  ) {
    this.modelType = modelType;
    this.isAdaptive = adaptive;
  }

  /**
   * Initialize the predictor (auto-detects best implementation)
   */
  async initialize(): Promise<void> {
    if (this.isAdaptive) {
      const { predictor, type } = await createAdaptivePredictor({
        targetCoverage: 0.90,
        gamma: 0.02,
        implementation: 'auto',
        fallbackToPure: true
      });
      this.predictor = predictor;
      this.implementationType = type;
    } else {
      const { predictor, type } = await createPredictor({
        alpha: 0.1,
        implementation: 'auto',
        fallbackToPure: true
      });
      this.predictor = predictor;
      this.implementationType = type;
    }
  }

  /**
   * Calibrate predictor with historical data
   */
  async calibrate(historicalData: TimeSeriesPoint[], modelPredictions: number[]): Promise<void> {
    if (!this.predictor) {
      throw new Error('Predictor not initialized. Call initialize() first.');
    }

    if (historicalData.length !== modelPredictions.length) {
      throw new Error('Historical data and predictions must have same length');
    }

    const actuals = historicalData.map(d => d.value);
    await this.predictor.calibrate(modelPredictions, actuals);

    // Store calibration history
    for (let i = 0; i < historicalData.length; i++) {
      this.calibrationHistory.push({
        prediction: modelPredictions[i],
        actual: actuals[i],
        timestamp: historicalData[i].timestamp
      });
    }
  }

  /**
   * Make a forecast with conformal prediction interval
   */
  async forecast(
    pointPrediction: number,
    timestamp: number,
    actualValue?: number
  ): Promise<ForecastResult> {
    if (!this.predictor) {
      throw new Error('Predictor not initialized. Call initialize() first.');
    }

    let interval: PredictionInterval;

    if (this.isAdaptive && actualValue !== undefined) {
      // Adaptive predictor updates with actual value
      const adaptivePredictor = this.predictor as AdaptiveConformalPredictor;
      interval = await adaptivePredictor.predictAndAdapt(pointPrediction, actualValue);
    } else {
      interval = this.predictor.predict(pointPrediction);
    }

    return {
      timestamp,
      pointForecast: pointPrediction,
      interval,
      modelUsed: this.modelType,
      confidence: interval.coverage(),
      metadata: {
        implementationType: this.implementationType,
        intervalWidth: interval.width(),
        relativeWidth: interval.relativeWidth()
      }
    };
  }

  /**
   * Update predictor with new observation
   */
  async update(prediction: number, actual: number, timestamp: number): Promise<void> {
    if (!this.predictor) {
      throw new Error('Predictor not initialized');
    }

    await this.predictor.update(prediction, actual);

    this.calibrationHistory.push({
      prediction,
      actual,
      timestamp
    });

    // Maintain maximum history size
    if (this.calibrationHistory.length > 10000) {
      this.calibrationHistory.shift();
    }
  }

  /**
   * Get predictor statistics
   */
  getStats(): any {
    if (!this.predictor) {
      return null;
    }

    const baseStats = this.predictor.getStats();
    const recentHistory = this.calibrationHistory.slice(-100);

    let mape = 0;
    let mae = 0;
    let rmse = 0;

    if (recentHistory.length > 0) {
      let sumApe = 0;
      let sumAe = 0;
      let sumSe = 0;

      for (const point of recentHistory) {
        const error = Math.abs(point.prediction - point.actual);
        const ape = (error / Math.abs(point.actual)) * 100;
        sumApe += ape;
        sumAe += error;
        sumSe += error * error;
      }

      mape = sumApe / recentHistory.length;
      mae = sumAe / recentHistory.length;
      rmse = Math.sqrt(sumSe / recentHistory.length);
    }

    return {
      ...baseStats,
      modelType: this.modelType,
      implementationType: this.implementationType,
      isAdaptive: this.isAdaptive,
      calibrationPoints: this.calibrationHistory.length,
      recentPerformance: {
        mape,
        mae,
        rmse
      }
    };
  }

  /**
   * Compute empirical coverage on validation set
   */
  computeCoverage(predictions: number[], actuals: number[]): number {
    if (!this.predictor) {
      return 0;
    }

    let covered = 0;
    for (let i = 0; i < predictions.length; i++) {
      const interval = this.predictor.predict(predictions[i]);
      if (interval.contains(actuals[i])) {
        covered++;
      }
    }

    return covered / predictions.length;
  }

  /**
   * Reset predictor state
   */
  async reset(): Promise<void> {
    this.calibrationHistory = [];
    await this.initialize();
  }

  /**
   * Get implementation type
   */
  getImplementationType(): string {
    return this.implementationType;
  }

  /**
   * Check if predictor is ready
   */
  isReady(): boolean {
    return this.predictor !== null && this.calibrationHistory.length > 0;
  }
}
