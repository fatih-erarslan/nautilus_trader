/**
 * Core types for energy forecasting system
 */

import type { PredictionInterval } from '@neural-trader/predictor';

/**
 * Time series data point
 */
export interface TimeSeriesPoint {
  timestamp: number;
  value: number;
  metadata?: Record<string, any>;
}

/**
 * Forecasting result with conformal prediction intervals
 */
export interface ForecastResult {
  timestamp: number;
  pointForecast: number;
  interval: PredictionInterval;
  modelUsed: string;
  confidence: number;
  seasonalComponent?: number;
  trendComponent?: number;
  metadata?: Record<string, any>;
}

/**
 * Multi-step ahead forecast
 */
export interface MultiStepForecast {
  forecasts: ForecastResult[];
  horizon: number;
  generatedAt: number;
  modelPerformance: ModelPerformance;
}

/**
 * Model performance metrics
 */
export interface ModelPerformance {
  modelName: string;
  mape: number; // Mean Absolute Percentage Error
  rmse: number; // Root Mean Squared Error
  mae: number; // Mean Absolute Error
  coverage: number; // Empirical coverage rate
  intervalWidth: number; // Average prediction interval width
  lastUpdated: number;
}

/**
 * Ensemble model configuration
 */
export interface EnsembleConfig {
  models: ModelType[];
  horizonWeights?: Map<number, Map<ModelType, number>>;
  adaptiveLearningRate?: number;
  retrainFrequency?: number;
  minCalibrationSamples?: number;
}

/**
 * Supported model types
 */
export enum ModelType {
  ARIMA = 'arima',
  LSTM = 'lstm',
  TRANSFORMER = 'transformer',
  PROPHET = 'prophet',
  ENSEMBLE = 'ensemble'
}

/**
 * Forecaster configuration
 */
export interface ForecasterConfig {
  alpha?: number; // Conformal prediction miscoverage rate
  calibrationSize?: number;
  horizon?: number; // Default forecast horizon
  seasonalPeriod?: number; // Seasonal period (e.g., 24 for hourly with daily seasonality)
  enableAdaptive?: boolean;
  ensembleConfig?: EnsembleConfig;
  weatherIntegration?: WeatherConfig;
}

/**
 * Weather integration configuration
 */
export interface WeatherConfig {
  enabled: boolean;
  openRouterApiKey?: string;
  features?: string[]; // e.g., ['temperature', 'cloudCover', 'windSpeed']
  location?: {
    latitude: number;
    longitude: number;
  };
}

/**
 * Seasonal pattern information
 */
export interface SeasonalPattern {
  period: number;
  strength: number; // 0-1 indicating how strong the pattern is
  components: number[]; // Seasonal component values
}

/**
 * Energy forecasting domain types
 */
export enum EnergyDomain {
  SOLAR = 'solar_generation',
  WIND = 'wind_power',
  DEMAND = 'electricity_demand',
  TEMPERATURE = 'temperature'
}

/**
 * Domain-specific metadata
 */
export interface DomainMetadata {
  domain: EnergyDomain;
  unit: string; // e.g., 'kW', 'MWh', '°C'
  capacity?: number; // For generation: installed capacity
  location?: string;
  additionalFeatures?: Record<string, number>;
}

/**
 * Model training result
 */
export interface TrainingResult {
  modelType: ModelType;
  trainDuration: number;
  samples: number;
  performance: ModelPerformance;
  hyperparameters?: Record<string, any>;
}

/**
 * Swarm exploration result
 */
export interface SwarmExplorationResult {
  bestModel: ModelType;
  bestHyperparameters: Record<string, any>;
  allResults: TrainingResult[];
  explorationTime: number;
  convergenceMetric: number;
}
