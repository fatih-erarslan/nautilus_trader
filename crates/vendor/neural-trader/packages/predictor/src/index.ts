/**
 * @neural-trader/predictor
 *
 * Conformal prediction for neural trading with guaranteed intervals
 *
 * @module @neural-trader/predictor
 */

export * from './pure/types';
export * from './pure/scores';
export * from './pure/conformal';
export * from './factory';

// Re-export for convenience
export { createPredictor } from './factory';
export type { PredictorImplementation } from './factory';
