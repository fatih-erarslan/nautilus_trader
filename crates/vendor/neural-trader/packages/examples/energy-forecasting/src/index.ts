/**
 * @neural-trader/example-energy-forecasting
 *
 * Self-learning energy forecasting with conformal prediction and swarm-based ensemble models
 *
 * @module @neural-trader/example-energy-forecasting
 */

// Core exports
export { EnergyForecaster } from './forecaster';
export { EnergyConformalPredictor } from './conformal-predictor';
export { EnsembleSwarm } from './ensemble-swarm';

// Model exports
export { ARIMAModel, LSTMModel, TransformerModel, ProphetModel, createModel } from './models';
export type { ForecastingModel } from './models';

// Type exports
export * from './types';

// Re-export predictor types for convenience
export type {
  PredictionInterval,
  PredictorConfig,
  AdaptiveConfig
} from '@neural-trader/predictor';
