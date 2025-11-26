/**
 * Neural Trader - Complete AI-powered algorithmic trading platform
 *
 * This meta package exports all Neural Trader functionality in one place.
 * It includes core types, backtesting, neural models, risk management,
 * trading strategies, and more.
 *
 * @packageDocumentation
 */

// Core types and interfaces
export * from '@neural-trader/core';

// Backtesting engine (Rust-powered)
export * from '@neural-trader/backtesting';

// Neural network models (Rust-powered with SIMD)
export * from '@neural-trader/neural';

// Risk management (Rust-powered)
export * from '@neural-trader/risk';

// Trading strategies
export * from '@neural-trader/strategies';

// Portfolio management
export * from '@neural-trader/portfolio';

// Order execution
export * from '@neural-trader/execution';

// Broker integrations
export * from '@neural-trader/brokers';

// Market data feeds
export * from '@neural-trader/market-data';

// Technical indicators and features
export * from '@neural-trader/features';

// Sports betting strategies
export * from '@neural-trader/sports-betting';

// Prediction market trading
export * from '@neural-trader/prediction-markets';

// News-based trading
export * from '@neural-trader/news-trading';

/**
 * Platform information
 */
export interface PlatformInfo {
  /** Operating system platform */
  platform: string;
  /** CPU architecture */
  arch: string;
}

/**
 * Package availability status
 */
export interface DependencyStatus {
  [packageName: string]: boolean;
}

/**
 * Version information
 */
export interface VersionInfo {
  /** Neural Trader version */
  version: string;
  /** Package dependencies */
  dependencies: Record<string, string>;
  /** Platform information */
  platform: PlatformInfo;
}

/**
 * Lazy-loaded package references
 */
export interface Packages {
  /** Core types and interfaces */
  readonly core: any;
  /** Backtesting engine */
  readonly backtesting: any;
  /** Neural network models */
  readonly neural: any;
  /** Risk management */
  readonly risk: any;
  /** Trading strategies */
  readonly strategies: any;
  /** Portfolio management */
  readonly portfolio: any;
  /** Order execution */
  readonly execution: any;
  /** Broker integrations */
  readonly brokers: any;
  /** Market data feeds */
  readonly marketData: any;
  /** Technical indicators */
  readonly features: any;
  /** Sports betting */
  readonly sportsBetting: any;
  /** Prediction markets */
  readonly predictionMarkets: any;
  /** News trading */
  readonly newsTrading: any;
}

/**
 * Platform detection information
 */
export const platform: PlatformInfo;

/**
 * Lazy-loaded package references
 * Access individual packages via this object to avoid loading all dependencies
 *
 * @example
 * ```typescript
 * import { packages } from 'neural-trader';
 *
 * // Only load backtesting package
 * const backtesting = packages.backtesting;
 * ```
 */
export const packages: Packages;

/**
 * Check availability of all dependencies
 * Useful for debugging installation issues
 *
 * @returns Object with availability status for each package
 *
 * @example
 * ```typescript
 * import { checkDependencies } from 'neural-trader';
 *
 * const status = checkDependencies();
 * if (!status.backtesting) {
 *   console.log('Backtesting package not available');
 * }
 * ```
 */
export function checkDependencies(): DependencyStatus;

/**
 * Get version and platform information
 *
 * @returns Version information object
 *
 * @example
 * ```typescript
 * import { getVersionInfo } from 'neural-trader';
 *
 * const info = getVersionInfo();
 * console.log(`Neural Trader v${info.version}`);
 * console.log(`Platform: ${info.platform.platform} ${info.platform.arch}`);
 * ```
 */
export function getVersionInfo(): VersionInfo;

/**
 * Display quick start guide
 * Shows basic usage examples and CLI commands
 *
 * @example
 * ```typescript
 * import { quickStart } from 'neural-trader';
 *
 * quickStart(); // Prints quick start guide
 * ```
 */
export function quickStart(): void;
