/**
 * Patient Arrival Forecaster
 *
 * Self-learning prediction of patient arrivals with uncertainty quantification.
 * Uses @neural-trader/predictor with seasonal pattern detection.
 */

import { Predictor, type PredictorConfig } from '@neural-trader/predictor';
import { AgentDB } from 'agentdb';
import type {
  ArrivalPattern,
  ForecastResult,
  HospitalMemory,
  PrivacyConfig
} from './types.js';

export interface ArrivalForecasterConfig {
  agentdbPath: string;
  enableNapiRS: boolean;
  privacy: PrivacyConfig;
  lookbackDays: number;
  forecastHorizon: number; // hours
  confidenceLevel: number; // 0-1
}

export class ArrivalForecaster {
  private predictor: Predictor;
  private memory: AgentDB;
  private config: ArrivalForecasterConfig;
  private patterns: Map<string, ArrivalPattern>;

  constructor(config: ArrivalForecasterConfig) {
    this.config = config;
    this.patterns = new Map();

    // Initialize predictor with NAPI-RS if enabled
    const predictorConfig: PredictorConfig = {
      enableNapiRS: config.enableNapiRS,
      modelType: 'gradient_boosting',
      hyperparameters: {
        n_estimators: 100,
        max_depth: 7,
        learning_rate: 0.05,
        subsample: 0.8
      }
    };
    this.predictor = new Predictor(predictorConfig);

    // Initialize AgentDB for memory persistence
    this.memory = new AgentDB({
      dbPath: config.agentdbPath,
      enableQuantization: true,
      quantizationBits: 8
    });
  }

  /**
   * Train forecaster on historical arrival data
   */
  async train(historicalData: Array<{ timestamp: Date; arrivals: number }>): Promise<void> {
    // Extract features for training
    const features = historicalData.map(d => this.extractFeatures(d.timestamp));
    const labels = historicalData.map(d => d.arrivals);

    // Train predictor
    await this.predictor.train(features, labels);

    // Learn seasonal patterns
    await this.learnSeasonalPatterns(historicalData);

    // Store training metadata in memory
    await this.memory.store('arrival_forecaster', {
      trainedAt: new Date().toISOString(),
      dataPoints: historicalData.length,
      dateRange: {
        start: historicalData[0].timestamp,
        end: historicalData[historicalData.length - 1].timestamp
      },
      patterns: Array.from(this.patterns.entries())
    });

    console.log(`✅ Trained arrival forecaster on ${historicalData.length} data points`);
  }

  /**
   * Forecast patient arrivals with uncertainty
   */
  async forecast(timestamp: Date): Promise<ForecastResult> {
    const features = this.extractFeatures(timestamp);

    // Get prediction with uncertainty
    const prediction = await this.predictor.predict([features]);
    const baselineValue = prediction[0];

    // Get seasonal adjustment
    const seasonalFactor = this.getSeasonalFactor(timestamp);
    const adjustedValue = baselineValue * seasonalFactor;

    // Get trend component
    const trendComponent = await this.getTrendComponent(timestamp);

    // Calculate confidence intervals using historical variance
    const variance = await this.getHistoricalVariance(timestamp);
    const zScore = this.getZScore(this.config.confidenceLevel);
    const stdDev = Math.sqrt(variance);

    const lowerBound = Math.max(0, adjustedValue - zScore * stdDev);
    const upperBound = adjustedValue + zScore * stdDev;

    // Store forecast for learning
    await this.storeForecast(timestamp, adjustedValue);

    return {
      timestamp,
      predictedArrivals: Math.round(adjustedValue + trendComponent),
      lowerBound: Math.round(lowerBound),
      upperBound: Math.round(upperBound),
      confidence: this.config.confidenceLevel,
      seasonalComponent: seasonalFactor,
      trendComponent
    };
  }

  /**
   * Forecast multiple time periods ahead
   */
  async forecastHorizon(startTime: Date): Promise<ForecastResult[]> {
    const forecasts: ForecastResult[] = [];
    const hoursAhead = this.config.forecastHorizon;

    for (let i = 0; i < hoursAhead; i++) {
      const timestamp = new Date(startTime.getTime() + i * 60 * 60 * 1000);
      const forecast = await this.forecast(timestamp);
      forecasts.push(forecast);
    }

    return forecasts;
  }

  /**
   * Update forecaster with actual arrivals (online learning)
   */
  async updateWithActuals(timestamp: Date, actualArrivals: number): Promise<void> {
    // Get previous forecast
    const forecast = await this.memory.retrieve(`forecast:${timestamp.toISOString()}`);

    if (forecast) {
      const error = actualArrivals - forecast.predictedArrivals;
      const errorPct = Math.abs(error) / actualArrivals;

      // Store forecast accuracy
      await this.memory.store(`accuracy:${timestamp.toISOString()}`, {
        timestamp,
        forecast: forecast.predictedArrivals,
        actual: actualArrivals,
        error,
        errorPct
      });

      // Update seasonal patterns if significant deviation
      if (errorPct > 0.2) {
        await this.updateSeasonalPattern(timestamp, actualArrivals);
      }
    }

    // Retrain incrementally every N updates
    const updateCount = await this.memory.retrieve('update_count') || 0;
    await this.memory.store('update_count', updateCount + 1);

    if (updateCount % 100 === 0) {
      console.log(`🔄 Incremental retrain after ${updateCount} updates`);
      await this.incrementalRetrain();
    }
  }

  /**
   * Extract temporal features from timestamp
   */
  private extractFeatures(timestamp: Date): number[] {
    const hour = timestamp.getHours();
    const dayOfWeek = timestamp.getDay();
    const dayOfMonth = timestamp.getDate();
    const month = timestamp.getMonth();
    const quarter = Math.floor(month / 3);

    // Cyclical encoding for periodic features
    const hourSin = Math.sin(2 * Math.PI * hour / 24);
    const hourCos = Math.cos(2 * Math.PI * hour / 24);
    const dowSin = Math.sin(2 * Math.PI * dayOfWeek / 7);
    const dowCos = Math.cos(2 * Math.PI * dayOfWeek / 7);
    const monthSin = Math.sin(2 * Math.PI * month / 12);
    const monthCos = Math.cos(2 * Math.PI * month / 12);

    // Binary features
    const isWeekend = dayOfWeek === 0 || dayOfWeek === 6 ? 1 : 0;
    const isNight = (hour >= 22 || hour < 6) ? 1 : 0;
    const isPeakHours = (hour >= 9 && hour <= 11) || (hour >= 18 && hour <= 20) ? 1 : 0;

    return [
      hour,
      dayOfWeek,
      dayOfMonth,
      month,
      quarter,
      hourSin,
      hourCos,
      dowSin,
      dowCos,
      monthSin,
      monthCos,
      isWeekend,
      isNight,
      isPeakHours
    ];
  }

  /**
   * Learn seasonal patterns from historical data
   */
  private async learnSeasonalPatterns(
    historicalData: Array<{ timestamp: Date; arrivals: number }>
  ): Promise<void> {
    // Group by hour of day, day of week, and month
    const patterns: Map<string, number[]> = new Map();

    for (const data of historicalData) {
      const hour = data.timestamp.getHours();
      const dow = data.timestamp.getDay();
      const month = data.timestamp.getMonth();
      const key = `${hour}-${dow}-${month}`;

      if (!patterns.has(key)) {
        patterns.set(key, []);
      }
      patterns.get(key)!.push(data.arrivals);
    }

    // Calculate statistics for each pattern
    for (const [key, values] of patterns.entries()) {
      const [hour, dow, month] = key.split('-').map(Number);
      const mean = values.reduce((a, b) => a + b, 0) / values.length;
      const variance = values.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / values.length;

      // Detect flu season (typically Dec-Feb in northern hemisphere)
      const isFluSeason = month === 11 || month === 0 || month === 1;
      const seasonalFactor = isFluSeason ? 1.3 : 1.0;

      this.patterns.set(key, {
        hourOfDay: hour,
        dayOfWeek: dow,
        month,
        expectedArrivals: mean,
        variance,
        seasonalFactor
      });
    }

    console.log(`📊 Learned ${this.patterns.size} seasonal patterns`);
  }

  /**
   * Get seasonal adjustment factor
   */
  private getSeasonalFactor(timestamp: Date): number {
    const hour = timestamp.getHours();
    const dow = timestamp.getDay();
    const month = timestamp.getMonth();
    const key = `${hour}-${dow}-${month}`;

    const pattern = this.patterns.get(key);
    return pattern?.seasonalFactor || 1.0;
  }

  /**
   * Get trend component
   */
  private async getTrendComponent(timestamp: Date): Promise<number> {
    // Simple linear trend from recent data
    const recentForecasts = await this.memory.retrieve('recent_forecasts') || [];

    if (recentForecasts.length < 2) {
      return 0;
    }

    // Calculate simple moving average trend
    const values = recentForecasts.map((f: any) => f.actual || f.predicted);
    const trend = (values[values.length - 1] - values[0]) / values.length;

    return trend;
  }

  /**
   * Get historical variance for confidence intervals
   */
  private async getHistoricalVariance(timestamp: Date): Promise<number> {
    const hour = timestamp.getHours();
    const dow = timestamp.getDay();
    const month = timestamp.getMonth();
    const key = `${hour}-${dow}-${month}`;

    const pattern = this.patterns.get(key);
    return pattern?.variance || 10; // default variance
  }

  /**
   * Get z-score for confidence level
   */
  private getZScore(confidenceLevel: number): number {
    // Approximate z-scores
    const zScores: Record<number, number> = {
      0.90: 1.645,
      0.95: 1.96,
      0.99: 2.576
    };

    return zScores[confidenceLevel] || 1.96;
  }

  /**
   * Store forecast for learning
   */
  private async storeForecast(timestamp: Date, value: number): Promise<void> {
    await this.memory.store(`forecast:${timestamp.toISOString()}`, {
      timestamp,
      predictedArrivals: value,
      storedAt: new Date().toISOString()
    });
  }

  /**
   * Update seasonal pattern based on new data
   */
  private async updateSeasonalPattern(timestamp: Date, actualArrivals: number): Promise<void> {
    const hour = timestamp.getHours();
    const dow = timestamp.getDay();
    const month = timestamp.getMonth();
    const key = `${hour}-${dow}-${month}`;

    const pattern = this.patterns.get(key);
    if (pattern) {
      // Exponential moving average update
      const alpha = 0.1;
      pattern.expectedArrivals = alpha * actualArrivals + (1 - alpha) * pattern.expectedArrivals;
      this.patterns.set(key, pattern);
    }
  }

  /**
   * Incremental retrain with recent data
   */
  private async incrementalRetrain(): Promise<void> {
    // In production, implement incremental learning
    // For now, just log the event
    console.log('🔄 Incremental retrain triggered');
  }

  /**
   * Get forecast accuracy metrics
   */
  async getAccuracyMetrics(): Promise<{
    mae: number;
    rmse: number;
    mape: number;
    samples: number;
  }> {
    const accuracyData = await this.memory.retrieve('accuracy:*') || [];

    if (accuracyData.length === 0) {
      return { mae: 0, rmse: 0, mape: 0, samples: 0 };
    }

    const errors = accuracyData.map((d: any) => Math.abs(d.error));
    const errorsPct = accuracyData.map((d: any) => d.errorPct);
    const errorsSquared = errors.map((e: number) => e * e);

    const mae = errors.reduce((a: number, b: number) => a + b, 0) / errors.length;
    const rmse = Math.sqrt(errorsSquared.reduce((a: number, b: number) => a + b, 0) / errorsSquared.length);
    const mape = errorsPct.reduce((a: number, b: number) => a + b, 0) / errorsPct.length;

    return {
      mae,
      rmse,
      mape: mape * 100,
      samples: accuracyData.length
    };
  }
}
