/**
 * Factory pattern for automatic implementation selection
 * Detects and uses best available implementation: native > WASM > pure JS
 */

import { SplitConformalPredictor, AdaptiveConformalPredictor } from './pure/conformal';
import type { NonconformityScore } from './pure/scores';

export type ImplementationType = 'native' | 'wasm' | 'pure';

export interface PredictorImplementation {
  type: ImplementationType;
  predictor: SplitConformalPredictor | AdaptiveConformalPredictor;
}

/**
 * Factory configuration options
 */
export interface FactoryConfig {
  alpha?: number;
  scoreFunction?: NonconformityScore;
  implementation?: 'auto' | 'native' | 'wasm' | 'pure';
  preferNative?: boolean;
  fallbackToWasm?: boolean;
  fallbackToPure?: boolean;
}

export interface AdaptiveFactoryConfig extends FactoryConfig {
  targetCoverage?: number;
  gamma?: number;
}

/**
 * Detect available implementations
 * @internal
 */
async function detectImplementations(): Promise<Set<ImplementationType>> {
  const available = new Set<ImplementationType>();

  // Always available
  available.add('pure');

  // Try to detect WASM (optional)
  try {
    // Check if wasm module can be imported
    if (typeof globalThis !== 'undefined') {
      // Would load WASM package if available
      // For now, mark as potentially available but we won't use it
      // available.add('wasm');
    }
  } catch (e) {
    // WASM not available
  }

  // Try to detect native addon (optional)
  try {
    // Check for native binding through optional dependency
    // eslint-disable-next-line global-require, @typescript-eslint/no-var-requires
    const nativeModule = require('@neural-trader/predictor-native');
    if (nativeModule) {
      available.add('native');
    }
  } catch (e) {
    // Native binding not available
  }

  return available;
}

/**
 * Select best available implementation
 * Priority: native > wasm > pure
 * @internal
 */
async function selectImplementation(
  options: FactoryConfig
): Promise<ImplementationType> {
  if (options.implementation && options.implementation !== 'auto') {
    return options.implementation;
  }

  const available = await detectImplementations();

  // Explicit preference
  if (options.preferNative && available.has('native')) {
    return 'native';
  }

  if (options.fallbackToWasm && available.has('wasm')) {
    return 'wasm';
  }

  // Default priority: native > wasm > pure
  if (available.has('native')) {
    return 'native';
  }

  if (available.has('wasm')) {
    return 'wasm';
  }

  return 'pure';
}

/**
 * Create a SplitConformalPredictor with automatic implementation selection
 *
 * Automatically detects and uses the best available implementation:
 * - Native (NAPI-rs): Fastest, requires compilation
 * - WASM: Good performance, smaller bundle size (requires wasm-pack)
 * - Pure JS: Always available, works everywhere
 *
 * @param config - Configuration options
 * @param scoreFunction - Nonconformity score function
 * @returns Promise resolving to predictor and implementation type
 *
 * @example
 * ```typescript
 * const { predictor, type } = await createPredictor({
 *   alpha: 0.1,
 *   preferNative: true,
 * });
 *
 * console.log(`Using ${type} implementation`);
 * await predictor.calibrate(predictions, actuals);
 * ```
 */
export async function createPredictor(
  config: FactoryConfig = {},
  scoreFunction?: NonconformityScore
): Promise<{ predictor: SplitConformalPredictor; type: ImplementationType }> {
  const implementation = await selectImplementation(config);

  let predictor: SplitConformalPredictor;

  if (implementation === 'native') {
    try {
      // Lazy load native implementation
      const { NativeConformalPredictor } = await lazyLoadNative();
      predictor = new NativeConformalPredictor(
        { alpha: config.alpha },
        scoreFunction
      );
      return { predictor, type: 'native' };
    } catch (e) {
      console.warn('Failed to load native implementation, falling back to WASM', e);
    }
  }

  if (implementation === 'wasm' || implementation === 'native') {
    try {
      // Lazy load WASM implementation
      const { WasmConformalPredictor } = await lazyLoadWasm();
      predictor = new WasmConformalPredictor(
        { alpha: config.alpha },
        scoreFunction
      );
      return { predictor, type: 'wasm' };
    } catch (e) {
      console.warn('Failed to load WASM implementation, falling back to pure JS', e);
    }
  }

  // Always available pure JS fallback
  predictor = new SplitConformalPredictor(
    { alpha: config.alpha },
    scoreFunction
  );
  return { predictor, type: 'pure' };
}

/**
 * Create an AdaptiveConformalPredictor with automatic implementation selection
 *
 * Same as createPredictor but for adaptive variant
 *
 * @param config - Configuration options
 * @param scoreFunction - Nonconformity score function
 * @returns Promise resolving to adaptive predictor and implementation type
 */
export async function createAdaptivePredictor(
  config: AdaptiveFactoryConfig = {},
  scoreFunction?: NonconformityScore
): Promise<{ predictor: AdaptiveConformalPredictor; type: ImplementationType }> {
  const implementation = await selectImplementation(config);

  let predictor: AdaptiveConformalPredictor;

  if (implementation === 'native') {
    try {
      // Lazy load native implementation
      const { NativeAdaptiveConformalPredictor } = await lazyLoadNative();
      predictor = new NativeAdaptiveConformalPredictor(
        {
          targetCoverage: config.targetCoverage,
          gamma: config.gamma,
        },
        scoreFunction
      );
      return { predictor, type: 'native' };
    } catch (e) {
      console.warn('Failed to load native implementation, falling back to WASM', e);
    }
  }

  if (implementation === 'wasm' || implementation === 'native') {
    try {
      // Lazy load WASM implementation
      const { WasmAdaptiveConformalPredictor } = await lazyLoadWasm();
      predictor = new WasmAdaptiveConformalPredictor(
        {
          targetCoverage: config.targetCoverage,
          gamma: config.gamma,
        },
        scoreFunction
      );
      return { predictor, type: 'wasm' };
    } catch (e) {
      console.warn('Failed to load WASM implementation, falling back to pure JS', e);
    }
  }

  // Always available pure JS fallback
  predictor = new AdaptiveConformalPredictor(
    {
      targetCoverage: config.targetCoverage,
      gamma: config.gamma,
    },
    scoreFunction
  );
  return { predictor, type: 'pure' };
}

/**
 * Lazy load native NAPI implementation
 * @internal
 */
async function lazyLoadNative(): Promise<any> {
  // Dynamic import for lazy loading
  if (typeof globalThis !== 'undefined' && typeof require !== 'undefined') {
    try {
      // @ts-expect-error - Optional native dependency may not be installed
      return await import('@neural-trader/predictor-native');
    } catch (e) {
      throw new Error('Native implementation not available');
    }
  }
  throw new Error('Native implementation not available in this environment');
}

/**
 * Lazy load WASM implementation
 * @internal
 */
async function lazyLoadWasm(): Promise<any> {
  // Dynamic import for lazy loading
  try {
    if (typeof globalThis !== 'undefined') {
      // Would load WASM module here
      // @ts-expect-error optional WASM dependency
      const wasmModule = await import('../wasm-pkg/index.js');
      return wasmModule;
    }
    throw new Error('WASM not available in this environment');
  } catch (e) {
    throw new Error('WASM implementation not available');
  }
}

/**
 * Detect current implementation type
 * Useful for logging and debugging
 *
 * @returns Promise resolving to available implementation types
 *
 * @example
 * ```typescript
 * const available = await detectAvailableImplementations();
 * console.log('Available implementations:', available);
 * ```
 */
export async function detectAvailableImplementations(): Promise<ImplementationType[]> {
  const available = await detectImplementations();
  return Array.from(available);
}

/**
 * Get implementation information
 * @internal
 */
export function getImplementationInfo(type: ImplementationType): {
  name: string;
  description: string;
  performance: string;
} {
  const info: Record<ImplementationType, any> = {
    native: {
      name: 'Native (NAPI-rs)',
      description: 'High-performance Rust implementation via Node.js native addon',
      performance: '~1x (baseline, fastest)',
    },
    wasm: {
      name: 'WebAssembly (Rust compiled to WASM)',
      description: 'Good performance with smaller bundle size than native',
      performance: '~1-2x slower than native',
    },
    pure: {
      name: 'Pure TypeScript',
      description: 'Pure JavaScript implementation with no external dependencies',
      performance: '~5-10x slower than native',
    },
  };

  return info[type];
}
