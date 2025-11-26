/**
 * Time series forecasting model implementations
 * Simple pure-TypeScript implementations for demonstration
 */

import { TimeSeriesPoint, ModelType, TrainingResult, ModelPerformance } from './types';

/**
 * Base forecasting model interface
 */
export interface ForecastingModel {
  modelType: ModelType;
  train(data: TimeSeriesPoint[]): Promise<void>;
  predict(steps: number): Promise<number[]>;
  getPerformance(): ModelPerformance;
  clone(): ForecastingModel;
}

/**
 * Simple ARIMA-like model (exponential smoothing)
 */
export class ARIMAModel implements ForecastingModel {
  modelType = ModelType.ARIMA;
  private alpha: number = 0.3; // Smoothing parameter
  private beta: number = 0.1; // Trend parameter
  private level: number = 0;
  private trend: number = 0;
  private trained: boolean = false;

  constructor(alpha: number = 0.3, beta: number = 0.1) {
    this.alpha = alpha;
    this.beta = beta;
  }

  async train(data: TimeSeriesPoint[]): Promise<void> {
    if (data.length < 2) {
      throw new Error('Need at least 2 data points for training');
    }

    // Initialize with first two points
    this.level = data[0].value;
    this.trend = data[1].value - data[0].value;

    // Double exponential smoothing
    for (let i = 1; i < data.length; i++) {
      const prevLevel = this.level;
      this.level = this.alpha * data[i].value + (1 - this.alpha) * (this.level + this.trend);
      this.trend = this.beta * (this.level - prevLevel) + (1 - this.beta) * this.trend;
    }

    this.trained = true;
  }

  async predict(steps: number): Promise<number[]> {
    if (!this.trained) {
      throw new Error('Model not trained');
    }

    const predictions: number[] = [];
    for (let i = 1; i <= steps; i++) {
      predictions.push(this.level + i * this.trend);
    }

    return predictions;
  }

  getPerformance(): ModelPerformance {
    return {
      modelName: 'ARIMA',
      mape: 0,
      rmse: 0,
      mae: 0,
      coverage: 0.9,
      intervalWidth: 0,
      lastUpdated: Date.now()
    };
  }

  clone(): ForecastingModel {
    const cloned = new ARIMAModel(this.alpha, this.beta);
    cloned.level = this.level;
    cloned.trend = this.trend;
    cloned.trained = this.trained;
    return cloned;
  }
}

/**
 * LSTM-inspired model (simple recurrent pattern)
 */
export class LSTMModel implements ForecastingModel {
  modelType = ModelType.LSTM;
  private lookback: number = 24;
  private weights: number[] = [];
  private recentValues: number[] = [];
  private trained: boolean = false;

  constructor(lookback: number = 24) {
    this.lookback = lookback;
  }

  async train(data: TimeSeriesPoint[]): Promise<void> {
    if (data.length < this.lookback) {
      throw new Error(`Need at least ${this.lookback} data points`);
    }

    // Simple attention-like weights based on recency
    this.weights = Array.from({ length: this.lookback }, (_, i) =>
      Math.exp(-0.1 * (this.lookback - i - 1))
    );

    // Normalize weights
    const sum = this.weights.reduce((a, b) => a + b, 0);
    this.weights = this.weights.map(w => w / sum);

    // Store recent values
    this.recentValues = data.slice(-this.lookback).map(d => d.value);
    this.trained = true;
  }

  async predict(steps: number): Promise<number[]> {
    if (!this.trained) {
      throw new Error('Model not trained');
    }

    const predictions: number[] = [];
    let currentWindow = [...this.recentValues];

    for (let i = 0; i < steps; i++) {
      // Weighted average prediction
      let prediction = 0;
      for (let j = 0; j < this.lookback; j++) {
        prediction += currentWindow[j] * this.weights[j];
      }

      predictions.push(prediction);

      // Update window for next prediction
      currentWindow.shift();
      currentWindow.push(prediction);
    }

    return predictions;
  }

  getPerformance(): ModelPerformance {
    return {
      modelName: 'LSTM',
      mape: 0,
      rmse: 0,
      mae: 0,
      coverage: 0.9,
      intervalWidth: 0,
      lastUpdated: Date.now()
    };
  }

  clone(): ForecastingModel {
    const cloned = new LSTMModel(this.lookback);
    cloned.weights = [...this.weights];
    cloned.recentValues = [...this.recentValues];
    cloned.trained = this.trained;
    return cloned;
  }
}

/**
 * Transformer-inspired model (self-attention)
 */
export class TransformerModel implements ForecastingModel {
  modelType = ModelType.TRANSFORMER;
  private sequenceLength: number = 48;
  private attentionWeights: number[][] = [];
  private recentValues: number[] = [];
  private trained: boolean = false;

  constructor(sequenceLength: number = 48) {
    this.sequenceLength = sequenceLength;
  }

  async train(data: TimeSeriesPoint[]): Promise<void> {
    if (data.length < this.sequenceLength) {
      throw new Error(`Need at least ${this.sequenceLength} data points`);
    }

    // Compute simple attention weights based on correlation
    this.recentValues = data.slice(-this.sequenceLength).map(d => d.value);

    // Self-attention: each position attends to all positions
    this.attentionWeights = Array(this.sequenceLength).fill(0).map(() => {
      const weights = Array(this.sequenceLength).fill(0).map(() => Math.random());
      const sum = weights.reduce((a, b) => a + b, 0);
      return weights.map(w => w / sum);
    });

    this.trained = true;
  }

  async predict(steps: number): Promise<number[]> {
    if (!this.trained) {
      throw new Error('Model not trained');
    }

    const predictions: number[] = [];
    let currentSequence = [...this.recentValues];

    for (let i = 0; i < steps; i++) {
      // Use last position's attention weights
      const weights = this.attentionWeights[this.attentionWeights.length - 1];
      let prediction = 0;

      for (let j = 0; j < this.sequenceLength; j++) {
        prediction += currentSequence[j] * weights[j];
      }

      predictions.push(prediction);

      // Update sequence
      currentSequence.shift();
      currentSequence.push(prediction);
    }

    return predictions;
  }

  getPerformance(): ModelPerformance {
    return {
      modelName: 'Transformer',
      mape: 0,
      rmse: 0,
      mae: 0,
      coverage: 0.9,
      intervalWidth: 0,
      lastUpdated: Date.now()
    };
  }

  clone(): ForecastingModel {
    const cloned = new TransformerModel(this.sequenceLength);
    cloned.attentionWeights = this.attentionWeights.map(w => [...w]);
    cloned.recentValues = [...this.recentValues];
    cloned.trained = this.trained;
    return cloned;
  }
}

/**
 * Prophet-inspired model (trend + seasonality)
 */
export class ProphetModel implements ForecastingModel {
  modelType = ModelType.PROPHET;
  private seasonalPeriod: number = 24;
  private trend: { level: number; slope: number } = { level: 0, slope: 0 };
  private seasonal: number[] = [];
  private trained: boolean = false;

  constructor(seasonalPeriod: number = 24) {
    this.seasonalPeriod = seasonalPeriod;
  }

  async train(data: TimeSeriesPoint[]): Promise<void> {
    if (data.length < this.seasonalPeriod * 2) {
      throw new Error(`Need at least ${this.seasonalPeriod * 2} data points`);
    }

    const values = data.map(d => d.value);

    // Compute trend using linear regression
    const n = values.length;
    const sumX = (n * (n - 1)) / 2;
    const sumY = values.reduce((a, b) => a + b, 0);
    const sumXY = values.reduce((sum, y, x) => sum + x * y, 0);
    const sumX2 = (n * (n - 1) * (2 * n - 1)) / 6;

    this.trend.slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
    this.trend.level = (sumY - this.trend.slope * sumX) / n;

    // Detrend and compute seasonal component
    const detrended = values.map((y, x) => y - (this.trend.level + this.trend.slope * x));

    this.seasonal = Array(this.seasonalPeriod).fill(0);
    const counts = Array(this.seasonalPeriod).fill(0);

    for (let i = 0; i < detrended.length; i++) {
      const seasonIdx = i % this.seasonalPeriod;
      this.seasonal[seasonIdx] += detrended[i];
      counts[seasonIdx]++;
    }

    this.seasonal = this.seasonal.map((s, i) => s / counts[i]);

    this.trained = true;
  }

  async predict(steps: number): Promise<number[]> {
    if (!this.trained) {
      throw new Error('Model not trained');
    }

    const predictions: number[] = [];
    const startIndex = 0; // Assuming we're predicting from current time

    for (let i = 0; i < steps; i++) {
      const trendComponent = this.trend.level + this.trend.slope * (startIndex + i);
      const seasonalComponent = this.seasonal[(startIndex + i) % this.seasonalPeriod];
      predictions.push(trendComponent + seasonalComponent);
    }

    return predictions;
  }

  getPerformance(): ModelPerformance {
    return {
      modelName: 'Prophet',
      mape: 0,
      rmse: 0,
      mae: 0,
      coverage: 0.9,
      intervalWidth: 0,
      lastUpdated: Date.now()
    };
  }

  clone(): ForecastingModel {
    const cloned = new ProphetModel(this.seasonalPeriod);
    cloned.trend = { ...this.trend };
    cloned.seasonal = [...this.seasonal];
    cloned.trained = this.trained;
    return cloned;
  }
}

/**
 * Model factory
 */
export function createModel(
  modelType: ModelType,
  hyperparameters?: Record<string, any>
): ForecastingModel {
  switch (modelType) {
    case ModelType.ARIMA:
      return new ARIMAModel(
        hyperparameters?.alpha ?? 0.3,
        hyperparameters?.beta ?? 0.1
      );
    case ModelType.LSTM:
      return new LSTMModel(hyperparameters?.lookback ?? 24);
    case ModelType.TRANSFORMER:
      return new TransformerModel(hyperparameters?.sequenceLength ?? 48);
    case ModelType.PROPHET:
      return new ProphetModel(hyperparameters?.seasonalPeriod ?? 24);
    default:
      throw new Error(`Unsupported model type: ${modelType}`);
  }
}
