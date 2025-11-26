/**
 * Validated wrappers for Portfolio management classes
 * Enforces input validation before delegating to native Rust layer
 */

import {
  PortfolioManager as NativePortfolioManager,
  PortfolioOptimizer as NativePortfolioOptimizer,
} from './index';
import {
  validatePortfolioConfig,
  validatePositionUpdate,
  validateOptimizerConfig,
  validateOptimizationInput,
  validateSymbol,
  validateRiskMetricsInput,
  ValidationError,
} from './validation';
import type {
  Position,
  PortfolioOptimization,
  RiskMetrics,
  OptimizerConfig,
} from '@neural-trader/core';

export class ValidatedPortfolioManager {
  private readonly native: NativePortfolioManager;

  /**
   * Initialize with validated initial cash
   * @throws {ValidationError} If configuration is invalid
   */
  constructor(initialCash: number) {
    try {
      // Validate initial cash
      if (!Number.isFinite(initialCash) || initialCash <= 0) {
        throw new ValidationError('Initial cash must be a positive finite number');
      }

      this.native = new NativePortfolioManager(initialCash);
    } catch (error) {
      if (error instanceof ValidationError) {
        throw error;
      }
      throw new ValidationError(
        `Failed to initialize PortfolioManager: ${error instanceof Error ? error.message : String(error)}`,
        error
      );
    }
  }

  /**
   * Get all positions
   */
  async getPositions(): Promise<Position[]> {
    try {
      return await this.native.getPositions();
    } catch (error) {
      throw new ValidationError(
        `Failed to get positions: ${error instanceof Error ? error.message : String(error)}`,
        error
      );
    }
  }

  /**
   * Get specific position with validated symbol
   * @throws {ValidationError} If symbol is invalid
   */
  async getPosition(symbol: string): Promise<Position | null> {
    try {
      const validatedSymbol = validateSymbol(symbol);
      return await this.native.getPosition(validatedSymbol);
    } catch (error) {
      if (error instanceof ValidationError) {
        throw error;
      }
      throw new ValidationError(
        `Failed to get position: ${error instanceof Error ? error.message : String(error)}`,
        error
      );
    }
  }

  /**
   * Update position with validated inputs
   * @throws {ValidationError} If inputs are invalid
   */
  async updatePosition(symbol: string, quantity: number, price: number): Promise<Position> {
    try {
      const validatedSymbol = validateSymbol(symbol);

      if (!Number.isFinite(quantity) || quantity === 0) {
        throw new ValidationError('Quantity must be a non-zero finite number');
      }

      if (!Number.isFinite(price) || price <= 0) {
        throw new ValidationError('Price must be a positive finite number');
      }

      return await this.native.updatePosition(validatedSymbol, quantity, price);
    } catch (error) {
      if (error instanceof ValidationError) {
        throw error;
      }
      throw new ValidationError(
        `Failed to update position: ${error instanceof Error ? error.message : String(error)}`,
        error
      );
    }
  }

  /**
   * Get current cash balance
   */
  async getCash(): Promise<number> {
    try {
      return await this.native.getCash();
    } catch (error) {
      throw new ValidationError(
        `Failed to get cash: ${error instanceof Error ? error.message : String(error)}`,
        error
      );
    }
  }

  /**
   * Get total portfolio value
   */
  async getTotalValue(): Promise<number> {
    try {
      return await this.native.getTotalValue();
    } catch (error) {
      throw new ValidationError(
        `Failed to get total value: ${error instanceof Error ? error.message : String(error)}`,
        error
      );
    }
  }

  /**
   * Get total profit/loss
   */
  async getTotalPnl(): Promise<number> {
    try {
      return await this.native.getTotalPnl();
    } catch (error) {
      throw new ValidationError(
        `Failed to get total PnL: ${error instanceof Error ? error.message : String(error)}`,
        error
      );
    }
  }
}

export class ValidatedPortfolioOptimizer {
  private readonly native: NativePortfolioOptimizer;

  /**
   * Initialize with validated configuration
   * @throws {ValidationError} If configuration is invalid
   */
  constructor(config: OptimizerConfig) {
    try {
      const validatedConfig = validateOptimizerConfig(config);
      this.native = new NativePortfolioOptimizer(validatedConfig);
    } catch (error) {
      if (error instanceof ValidationError) {
        throw error;
      }
      throw new ValidationError(
        `Failed to initialize PortfolioOptimizer: ${error instanceof Error ? error.message : String(error)}`,
        error
      );
    }
  }

  /**
   * Optimize portfolio with validated inputs
   * @throws {ValidationError} If inputs are invalid
   */
  async optimize(
    symbols: string[],
    returns: number[],
    covariance: number[][]
  ): Promise<PortfolioOptimization> {
    try {
      const validatedInput = validateOptimizationInput({
        symbols,
        returns,
        covariance,
      });

      return await this.native.optimize(
        validatedInput.symbols,
        validatedInput.returns,
        validatedInput.covariance
      );
    } catch (error) {
      if (error instanceof ValidationError) {
        throw error;
      }
      throw new ValidationError(
        `Failed to optimize portfolio: ${error instanceof Error ? error.message : String(error)}`,
        error
      );
    }
  }

  /**
   * Calculate risk metrics for positions with validated inputs
   * @throws {ValidationError} If inputs are invalid
   */
  calculateRisk(positions: Record<string, number>): RiskMetrics {
    try {
      const validatedMetrics = validateRiskMetricsInput({
        positions,
        prices: {}, // Will be populated by the native layer
      });

      return this.native.calculateRisk(positions);
    } catch (error) {
      if (error instanceof ValidationError) {
        throw error;
      }
      throw new ValidationError(
        `Failed to calculate risk: ${error instanceof Error ? error.message : String(error)}`,
        error
      );
    }
  }
}
