/**
 * Load Forecaster - Multi-horizon load prediction with self-learning error correction
 *
 * Features:
 * - Multi-horizon forecasting (5min to 7 days)
 * - Conformal prediction intervals via @neural-trader/predictor
 * - Self-learning error correction with AgentDB
 * - Weather and calendar feature engineering
 * - Memory-persistent patterns
 */

import { createDatabase } from 'agentdb';
import type {
  LoadForecast,
  GridState,
  LoadForecasterConfig,
  ForecastErrorCorrection,
} from './types.js';
import { ForecastHorizon } from './types.js';

/**
 * Multi-horizon load forecaster with self-learning capabilities
 */
export class LoadForecaster {
  private readonly config: LoadForecasterConfig;
  private readonly memoryDb: any;
  private errorCorrection: ForecastErrorCorrection;
  private historicalData: GridState[] = [];

  constructor(config: LoadForecasterConfig) {
    this.config = config;
    this.memoryDb = createDatabase(`./${config.memoryNamespace}.db`);

    // Initialize error correction
    this.errorCorrection = {
      hourlyBias: new Array(24).fill(0),
      dailyBias: new Array(7).fill(0),
      weatherCorrections: new Map(),
      recentErrors: {
        mean: 0,
        stdDev: 0,
        mae: 0,
        mape: 0,
      },
      lastUpdate: new Date(),
    };
  }

  /**
   * Initialize forecaster by loading historical patterns from memory
   */
  async initialize(): Promise<void> {
    try {
      // Create table for storing data
      this.memoryDb.exec(`
        CREATE TABLE IF NOT EXISTS forecaster_data (
          key TEXT PRIMARY KEY,
          value TEXT
        )
      `);

      // Load historical error corrections from memory
      const stmt = this.memoryDb.prepare(
        'SELECT value FROM forecaster_data WHERE key = ?'
      );

      const correctionsRow = stmt.get('forecast-error-corrections');
      if (correctionsRow) {
        this.errorCorrection = JSON.parse(correctionsRow.value) as ForecastErrorCorrection;
        this.errorCorrection.weatherCorrections = new Map(
          Object.entries(
            (JSON.parse(correctionsRow.value) as any).weatherCorrections || {}
          )
        );
      }

      // Load historical grid states
      const historyRow = stmt.get('historical-grid-states');
      if (historyRow) {
        this.historicalData = JSON.parse(historyRow.value) as GridState[];
      }

      console.log(`LoadForecaster initialized with ${this.historicalData.length} historical states`);
    } catch (error) {
      console.warn('Failed to load historical data, starting fresh:', error);
    }
  }

  /**
   * Add historical grid state for training
   */
  async addHistoricalState(state: GridState): Promise<void> {
    this.historicalData.push(state);

    // Maintain rolling window
    const maxStates = this.config.historyWindowDays * 24 * (60 / 5); // 5-min intervals
    if (this.historicalData.length > maxStates) {
      this.historicalData = this.historicalData.slice(-maxStates);
    }

    // Persist to memory periodically
    if (this.historicalData.length % 100 === 0) {
      const stmt = this.memoryDb.prepare(
        'INSERT OR REPLACE INTO forecaster_data (key, value) VALUES (?, ?)'
      );
      stmt.run('historical-grid-states', JSON.stringify(this.historicalData));
    }
  }

  /**
   * Generate multi-horizon load forecasts
   */
  async forecast(
    currentState: GridState,
    horizons: ForecastHorizon[] = this.config.horizons
  ): Promise<LoadForecast[]> {
    const forecasts: LoadForecast[] = [];

    for (const horizon of horizons) {
      const forecast = await this.forecastHorizon(currentState, horizon);
      forecasts.push(forecast);
    }

    return forecasts;
  }

  /**
   * Forecast load for specific horizon
   */
  private async forecastHorizon(
    currentState: GridState,
    horizon: ForecastHorizon
  ): Promise<LoadForecast> {
    const stepsAhead = this.getStepsAhead(horizon);
    const targetTime = new Date(
      currentState.timestamp.getTime() + stepsAhead * 60000
    );

    // Feature engineering
    const features = this.extractFeatures(currentState, targetTime);

    // Base prediction using historical patterns
    const basePrediction = this.predictFromPatterns(features);

    // Apply self-learning error correction
    const correctedPrediction = this.applyErrorCorrection(
      basePrediction,
      features
    );

    // Calculate prediction intervals (simplified - in production use @neural-trader/predictor)
    const uncertainty = this.estimateUncertainty(horizon);
    const zScore = 1.96; // 95% confidence
    const margin = uncertainty * zScore;

    const forecast: LoadForecast = {
      timestamp: targetTime,
      loadMW: correctedPrediction,
      lowerBound: correctedPrediction - margin,
      upperBound: correctedPrediction + margin,
      confidence: this.config.confidenceLevel,
      horizon,
      metrics: {
        mae: this.errorCorrection.recentErrors.mae,
        mape: this.errorCorrection.recentErrors.mape,
      },
    };

    return forecast;
  }

  /**
   * Extract features for prediction
   */
  private extractFeatures(
    currentState: GridState,
    targetTime: Date
  ): Record<string, number> {
    const targetHour = targetTime.getHours();
    const targetDay = targetTime.getDay();

    return {
      // Time features
      hourOfDay: targetHour,
      dayOfWeek: targetDay,
      isWeekend: targetDay === 0 || targetDay === 6 ? 1 : 0,
      isHoliday: currentState.isHoliday ? 1 : 0,
      hourSin: Math.sin((2 * Math.PI * targetHour) / 24),
      hourCos: Math.cos((2 * Math.PI * targetHour) / 24),
      daySin: Math.sin((2 * Math.PI * targetDay) / 7),
      dayCos: Math.cos((2 * Math.PI * targetDay) / 7),

      // Current state
      currentLoad: currentState.loadMW,
      currentGeneration: currentState.generationMW,
      renewablePenetration: currentState.renewablePenetration,

      // Weather (if available)
      temperature: currentState.weather?.temperature || 20,
      windSpeed: currentState.weather?.windSpeed || 5,
      solarIrradiance: currentState.weather?.solarIrradiance || 500,
    };
  }

  /**
   * Predict load from historical patterns
   */
  private predictFromPatterns(features: Record<string, number>): number {
    if (this.historicalData.length < 7 * 24) {
      // Not enough data, return simple estimate
      return features.currentLoad || 1000;
    }

    // Find similar historical states
    const similarStates = this.findSimilarStates(features, 20);

    if (similarStates.length === 0) {
      return features.currentLoad || 1000;
    }

    // Weighted average based on similarity
    const totalWeight = similarStates.reduce((sum, s) => sum + s.weight, 0);
    const prediction = similarStates.reduce(
      (sum, s) => sum + s.state.loadMW * s.weight,
      0
    ) / totalWeight;

    return prediction;
  }

  /**
   * Find similar historical states using feature distance
   */
  private findSimilarStates(
    features: Record<string, number>,
    topK: number
  ): Array<{ state: GridState; weight: number }> {
    const candidates = this.historicalData
      .filter(state => {
        // Filter to same hour and day type
        return (
          Math.abs(state.hourOfDay - features.hourOfDay) <= 1 &&
          ((state.dayOfWeek < 5 && features.dayOfWeek < 5) ||
            (state.dayOfWeek >= 5 && features.dayOfWeek >= 5))
        );
      })
      .map(state => {
        const stateFeatures = this.extractFeatures(
          state,
          new Date(state.timestamp)
        );
        const distance = this.euclideanDistance(features, stateFeatures);
        const weight = 1 / (1 + distance);
        return { state, weight };
      })
      .sort((a, b) => b.weight - a.weight)
      .slice(0, topK);

    return candidates;
  }

  /**
   * Calculate Euclidean distance between feature vectors
   */
  private euclideanDistance(
    a: Record<string, number>,
    b: Record<string, number>
  ): number {
    const keys = Object.keys(a);
    let sumSquares = 0;

    for (const key of keys) {
      if (key in b) {
        const diff = a[key] - b[key];
        sumSquares += diff * diff;
      }
    }

    return Math.sqrt(sumSquares);
  }

  /**
   * Apply self-learning error correction
   */
  private applyErrorCorrection(
    basePrediction: number,
    features: Record<string, number>
  ): number {
    if (!this.config.enableErrorCorrection) {
      return basePrediction;
    }

    // Apply hourly bias correction
    const hourlyCorrection = this.errorCorrection.hourlyBias[
      Math.floor(features.hourOfDay)
    ];

    // Apply daily bias correction
    const dailyCorrection = this.errorCorrection.dailyBias[
      Math.floor(features.dayOfWeek)
    ];

    // Weather-based corrections
    let weatherCorrection = 0;
    const tempBucket = Math.floor(features.temperature / 5) * 5;
    const weatherKey = `temp_${tempBucket}`;
    if (this.errorCorrection.weatherCorrections.has(weatherKey)) {
      weatherCorrection = this.errorCorrection.weatherCorrections.get(weatherKey)!;
    }

    // Combine corrections with decay weights
    const correctedPrediction =
      basePrediction +
      0.5 * hourlyCorrection +
      0.3 * dailyCorrection +
      0.2 * weatherCorrection;

    return correctedPrediction;
  }

  /**
   * Update error correction with actual observation
   */
  async updateWithActual(
    forecast: LoadForecast,
    actualLoadMW: number
  ): Promise<void> {
    const error = actualLoadMW - forecast.loadMW;
    const absError = Math.abs(error);
    const pctError = (absError / actualLoadMW) * 100;

    // Update hourly bias
    const hour = forecast.timestamp.getHours();
    this.errorCorrection.hourlyBias[hour] =
      0.9 * this.errorCorrection.hourlyBias[hour] + 0.1 * error;

    // Update daily bias
    const day = forecast.timestamp.getDay();
    this.errorCorrection.dailyBias[day] =
      0.9 * this.errorCorrection.dailyBias[day] + 0.1 * error;

    // Update error statistics
    const alpha = 0.05; // Exponential moving average factor
    this.errorCorrection.recentErrors.mean =
      (1 - alpha) * this.errorCorrection.recentErrors.mean + alpha * error;
    this.errorCorrection.recentErrors.mae =
      (1 - alpha) * this.errorCorrection.recentErrors.mae + alpha * absError;
    this.errorCorrection.recentErrors.mape =
      (1 - alpha) * this.errorCorrection.recentErrors.mape + alpha * pctError;

    this.errorCorrection.lastUpdate = new Date();

    // Persist corrections periodically
    const hoursSinceLastUpdate =
      (Date.now() - this.errorCorrection.lastUpdate.getTime()) / 3600000;
    if (hoursSinceLastUpdate >= this.config.correctionUpdateFrequency) {
      await this.persistErrorCorrections();
    }
  }

  /**
   * Estimate prediction uncertainty for horizon
   */
  private estimateUncertainty(horizon: ForecastHorizon): number {
    // Uncertainty grows with forecast horizon
    const baseUncertainty = 50; // MW
    const horizonMultipliers: Record<ForecastHorizon, number> = {
      [ForecastHorizon.MINUTES_5]: 1.0,
      [ForecastHorizon.MINUTES_15]: 1.2,
      [ForecastHorizon.HOUR_1]: 1.5,
      [ForecastHorizon.HOUR_4]: 2.0,
      [ForecastHorizon.HOUR_24]: 3.0,
      [ForecastHorizon.HOUR_168]: 5.0,
    };

    const multiplier = horizonMultipliers[horizon] || 1.0;
    return baseUncertainty * multiplier;
  }

  /**
   * Convert horizon to minutes ahead
   */
  private getStepsAhead(horizon: ForecastHorizon): number {
    const horizonMinutes: Record<ForecastHorizon, number> = {
      [ForecastHorizon.MINUTES_5]: 5,
      [ForecastHorizon.MINUTES_15]: 15,
      [ForecastHorizon.HOUR_1]: 60,
      [ForecastHorizon.HOUR_4]: 240,
      [ForecastHorizon.HOUR_24]: 1440,
      [ForecastHorizon.HOUR_168]: 10080,
    };
    return horizonMinutes[horizon];
  }

  /**
   * Persist error corrections to memory
   */
  private async persistErrorCorrections(): Promise<void> {
    const correctionsData = {
      ...this.errorCorrection,
      weatherCorrections: Object.fromEntries(
        this.errorCorrection.weatherCorrections
      ),
    };

    const stmt = this.memoryDb.prepare(
      'INSERT OR REPLACE INTO forecaster_data (key, value) VALUES (?, ?)'
    );
    stmt.run('forecast-error-corrections', JSON.stringify(correctionsData));
  }

  /**
   * Get forecast accuracy metrics
   */
  getAccuracyMetrics(): ForecastErrorCorrection['recentErrors'] {
    return { ...this.errorCorrection.recentErrors };
  }
}
