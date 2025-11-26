/**
 * TypeScript wrapper for WASM bindings
 * Provides async initialization and type-safe API
 */

// Import the WASM module
import * as wasm from '../../wasm-pkg/neural_trader_predictor_wasm';

// Track initialization state
let wasmInitialized = false;

/**
 * Initialize the WASM module
 * Safe to call multiple times - only initializes once
 */
export async function initWasm(): Promise<void> {
  if (wasmInitialized) {
    return;
  }
  wasmInitialized = true;
}

/**
 * Prediction interval with bounds and metadata
 */
export interface IPredictionInterval {
  readonly point: number;
  readonly lower: number;
  readonly upper: number;
  readonly alpha: number;
  readonly quantile: number;
  readonly timestamp: number;

  width(): number;
  contains(value: number): boolean;
  relativeWidth(): number;
  coverage(): number;
}

/**
 * Configuration for Split Conformal Predictor
 */
export interface PredictorConfig {
  alpha?: number;
  calibrationSize?: number;
  maxIntervalWidthPct?: number;
  recalibrationFreq?: number;
}

/**
 * Configuration for Adaptive Conformal Predictor
 */
export interface AdaptiveConfig {
  targetCoverage?: number;
  gamma?: number;
  coverageWindow?: number;
  alphaMin?: number;
  alphaMax?: number;
}

/**
 * Statistics from predictor
 */
export interface PredictorStats {
  [key: string]: any;
}

/**
 * Wrapper for WASM-based Split Conformal Predictor
 */
export class WasmConformalPredictor {
  private predictor: wasm.WasmConformalPredictor;

  constructor(config?: PredictorConfig) {
    // Initialize WASM on construction
    initWasm().catch(err => console.error('Failed to initialize WASM:', err));

    let wasmConfig: any = undefined;
    if (config) {
      wasmConfig = new wasm.WasmPredictorConfig(config.alpha);
      if (config.calibrationSize !== undefined) {
        wasmConfig = wasmConfig.with_calibration_size(config.calibrationSize);
      }
    }

    this.predictor = new wasm.WasmConformalPredictor(wasmConfig);
  }

  /**
   * Calibrate with predictions and actuals
   */
  async calibrate(predictions: number[], actuals: number[]): Promise<void> {
    return this.predictor.calibrate(predictions, actuals);
  }

  /**
   * Make a prediction with confidence interval
   */
  predict(pointPrediction: number): IPredictionInterval {
    return this.predictor.predict(pointPrediction);
  }

  /**
   * Update with new observation
   */
  async update(prediction: number, actual: number): Promise<void> {
    return this.predictor.update(prediction, actual);
  }

  /**
   * Get empirical coverage
   */
  async getEmpiricalCoverage(
    predictions: number[],
    actuals: number[]
  ): Promise<number> {
    return this.predictor.get_empirical_coverage(predictions, actuals);
  }

  /**
   * Get number of calibration samples
   */
  nCalibration(): number {
    return this.predictor.n_calibration();
  }

  /**
   * Get number of predictions made
   */
  nPredictions(): number {
    return this.predictor.n_predictions();
  }

  /**
   * Get current quantile
   */
  getQuantile(): number {
    return this.predictor.get_quantile();
  }

  /**
   * Get alpha value
   */
  getAlpha(): number {
    return this.predictor.get_alpha();
  }

  /**
   * Check if calibrated
   */
  isCalibrated(): boolean {
    return this.predictor.is_calibrated();
  }

  /**
   * Get statistics
   */
  getStats(): PredictorStats {
    const json = this.predictor.get_stats();
    return JSON.parse(json);
  }

  /**
   * Reset the predictor
   */
  reset(): void {
    this.predictor.reset();
  }
}

/**
 * Wrapper for WASM-based Adaptive Conformal Predictor
 */
export class WasmAdaptivePredictor {
  private predictor: wasm.WasmAdaptivePredictor;

  constructor(config?: AdaptiveConfig) {
    // Initialize WASM on construction
    initWasm().catch(err => console.error('Failed to initialize WASM:', err));

    let wasmConfig: any = undefined;
    if (config) {
      wasmConfig = new wasm.WasmAdaptiveConfig(
        config.targetCoverage,
        config.gamma
      );
    } else {
      wasmConfig = new wasm.WasmAdaptiveConfig(undefined, undefined);
    }

    this.predictor = new wasm.WasmAdaptivePredictor(wasmConfig);
  }

  /**
   * Calibrate with predictions and actuals
   */
  async calibrate(predictions: number[], actuals: number[]): Promise<void> {
    return this.predictor.calibrate(predictions, actuals);
  }

  /**
   * Make a prediction
   */
  predict(pointPrediction: number): IPredictionInterval {
    return this.predictor.predict(pointPrediction);
  }

  /**
   * Make prediction and adapt alpha
   */
  predictAndAdapt(
    pointPrediction: number,
    actual?: number
  ): IPredictionInterval {
    return this.predictor.predict_and_adapt(pointPrediction, actual);
  }

  /**
   * Observe actual value and adapt alpha
   */
  observeAndAdapt(interval: IPredictionInterval, actual: number): number {
    return this.predictor.observe_and_adapt(interval as any, actual);
  }

  /**
   * Get empirical coverage
   */
  getEmpiricalCoverage(): number {
    return this.predictor.empirical_coverage();
  }

  /**
   * Get current alpha
   */
  getCurrentAlpha(): number {
    return this.predictor.get_current_alpha();
  }

  /**
   * Get target coverage
   */
  getTargetCoverage(): number {
    return this.predictor.get_target_coverage();
  }

  /**
   * Get coverage error
   */
  getCoverageError(): number {
    return this.predictor.get_coverage_error();
  }

  /**
   * Get number of adaptations
   */
  getNAdaptations(): number {
    return this.predictor.get_n_adaptations();
  }

  /**
   * Get history size
   */
  getHistorySize(): number {
    return this.predictor.get_history_size();
  }

  /**
   * Get statistics
   */
  getStats(): PredictorStats {
    const json = this.predictor.get_stats();
    return JSON.parse(json);
  }

  /**
   * Reset the adaptive state
   */
  reset(): void {
    this.predictor.reset();
  }
}

/**
 * Export pure JS implementations as fallback
 */
export {
  SplitConformalPredictor,
  AdaptiveConformalPredictor,
  CQRPredictor,
} from '../pure/conformal';
export type { PredictionInterval } from '../pure/types';

/**
 * Factory function to create predictor with WASM if available, fallback to pure JS
 */
export async function createConformalPredictor(
  config?: PredictorConfig
): Promise<WasmConformalPredictor> {
  try {
    await initWasm();
    return new WasmConformalPredictor(config);
  } catch (error) {
    console.warn('Failed to initialize WASM, consider using pure JS implementation', error);
    throw error;
  }
}

/**
 * Factory function to create adaptive predictor with WASM if available
 */
export async function createAdaptivePredictor(
  config?: AdaptiveConfig
): Promise<WasmAdaptivePredictor> {
  try {
    await initWasm();
    return new WasmAdaptivePredictor(config);
  } catch (error) {
    console.warn('Failed to initialize WASM, consider using pure JS implementation', error);
    throw error;
  }
}
