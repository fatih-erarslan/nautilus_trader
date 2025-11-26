/**
 * Validated wrapper for StrategyRunner class
 * Enforces input validation before delegating to native Rust layer
 */

import { StrategyRunner as NativeStrategyRunner, SubscriptionHandle } from './index';
import {
  validateStrategyConfig,
  validateMomentumParameters,
  validateMeanReversionParameters,
  validateArbitrageParameters,
  validatePairsTradingParameters,
  validateStrategyId,
  ValidationError,
} from './validation';
import type { StrategyConfig, Signal } from '@neural-trader/core';

export class ValidatedStrategyRunner {
  private readonly native: NativeStrategyRunner;

  constructor() {
    this.native = new NativeStrategyRunner();
  }

  /**
   * Add momentum strategy with validated configuration
   * @throws {ValidationError} If configuration is invalid
   */
  async addMomentumStrategy(config: StrategyConfig): Promise<string> {
    try {
      // Validate base configuration
      const validatedConfig = validateStrategyConfig(config);

      // Parse and validate momentum-specific parameters
      const parameters = JSON.parse(validatedConfig.parameters);
      validateMomentumParameters(parameters);

      return await this.native.addMomentumStrategy(validatedConfig);
    } catch (error) {
      if (error instanceof ValidationError) {
        throw error;
      }
      throw new ValidationError(
        `Failed to add momentum strategy: ${error instanceof Error ? error.message : String(error)}`,
        error
      );
    }
  }

  /**
   * Add mean reversion strategy with validated configuration
   * @throws {ValidationError} If configuration is invalid
   */
  async addMeanReversionStrategy(config: StrategyConfig): Promise<string> {
    try {
      // Validate base configuration
      const validatedConfig = validateStrategyConfig(config);

      // Parse and validate mean reversion-specific parameters
      const parameters = JSON.parse(validatedConfig.parameters);
      validateMeanReversionParameters(parameters);

      return await this.native.addMeanReversionStrategy(validatedConfig);
    } catch (error) {
      if (error instanceof ValidationError) {
        throw error;
      }
      throw new ValidationError(
        `Failed to add mean reversion strategy: ${error instanceof Error ? error.message : String(error)}`,
        error
      );
    }
  }

  /**
   * Add arbitrage strategy with validated configuration
   * @throws {ValidationError} If configuration is invalid
   */
  async addArbitrageStrategy(config: StrategyConfig): Promise<string> {
    try {
      // Validate base configuration
      const validatedConfig = validateStrategyConfig(config);

      // Parse and validate arbitrage-specific parameters
      const parameters = JSON.parse(validatedConfig.parameters);
      validateArbitrageParameters(parameters);

      return await this.native.addArbitrageStrategy(validatedConfig);
    } catch (error) {
      if (error instanceof ValidationError) {
        throw error;
      }
      throw new ValidationError(
        `Failed to add arbitrage strategy: ${error instanceof Error ? error.message : String(error)}`,
        error
      );
    }
  }

  /**
   * Add pairs trading strategy with validated configuration
   * @throws {ValidationError} If configuration is invalid
   */
  async addPairsTradingStrategy(config: StrategyConfig): Promise<string> {
    try {
      // Validate base configuration
      const validatedConfig = validateStrategyConfig(config);

      // Parse and validate pairs trading-specific parameters
      const parameters = JSON.parse(validatedConfig.parameters);
      validatePairsTradingParameters(parameters);

      return await this.native.addArbitrageStrategy(validatedConfig); // Reuse arbitrage endpoint
    } catch (error) {
      if (error instanceof ValidationError) {
        throw error;
      }
      throw new ValidationError(
        `Failed to add pairs trading strategy: ${error instanceof Error ? error.message : String(error)}`,
        error
      );
    }
  }

  /**
   * Generate signals from all active strategies
   */
  async generateSignals(): Promise<Signal[]> {
    try {
      return await this.native.generateSignals();
    } catch (error) {
      throw new ValidationError(
        `Failed to generate signals: ${error instanceof Error ? error.message : String(error)}`,
        error
      );
    }
  }

  /**
   * Subscribe to signal updates with validation
   */
  subscribeSignals(callback: (signal: Signal) => void): SubscriptionHandle {
    if (typeof callback !== 'function') {
      throw new ValidationError('Callback must be a function');
    }

    return this.native.subscribeSignals(callback);
  }

  /**
   * List all active strategies
   */
  async listStrategies(): Promise<string[]> {
    try {
      return await this.native.listStrategies();
    } catch (error) {
      throw new ValidationError(
        `Failed to list strategies: ${error instanceof Error ? error.message : String(error)}`,
        error
      );
    }
  }

  /**
   * Remove strategy with ID validation
   * @throws {ValidationError} If strategy ID is invalid
   */
  async removeStrategy(strategyId: string): Promise<boolean> {
    try {
      const validatedId = validateStrategyId(strategyId);
      return await this.native.removeStrategy(validatedId);
    } catch (error) {
      if (error instanceof ValidationError) {
        throw error;
      }
      throw new ValidationError(
        `Failed to remove strategy: ${error instanceof Error ? error.message : String(error)}`,
        error
      );
    }
  }
}

// Export for use
export { SubscriptionHandle };
