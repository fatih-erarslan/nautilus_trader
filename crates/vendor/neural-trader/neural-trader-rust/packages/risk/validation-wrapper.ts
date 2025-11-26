/**
 * Validated wrapper for Risk management classes and functions
 * Enforces input validation before delegating to native Rust layer
 */

import { RiskManager as NativeRiskManager } from './index';
import {
  validateRiskConfig,
  validateVarInput,
  validateCvarInput,
  validateKellyInput,
  validateDrawdownInput,
  validatePositionSizeInput,
  validatePositionValue,
  validateSharpeRatioInput,
  validateSortinoRatioInput,
  validateMaxLeverageInput,
  ValidationError,
} from './validation';
import type {
  RiskConfig,
  VaRResult,
  CVaRResult,
  DrawdownMetrics,
  KellyResult,
  PositionSize,
} from '@neural-trader/core';

export class ValidatedRiskManager {
  private readonly native: NativeRiskManager;

  /**
   * Initialize with validated configuration
   * @throws {ValidationError} If configuration is invalid
   */
  constructor(config: RiskConfig) {
    try {
      const validatedConfig = validateRiskConfig(config);
      this.native = new NativeRiskManager(validatedConfig);
    } catch (error) {
      if (error instanceof ValidationError) {
        throw error;
      }
      throw new ValidationError(
        `Failed to initialize RiskManager: ${error instanceof Error ? error.message : String(error)}`,
        error
      );
    }
  }

  /**
   * Calculate Value at Risk with validated inputs
   * @throws {ValidationError} If inputs are invalid
   */
  calculateVar(returns: number[], portfolioValue: number): VaRResult {
    try {
      const validatedInput = validateVarInput({
        returns,
        portfolioValue,
      });

      return this.native.calculateVar(validatedInput.returns, validatedInput.portfolioValue);
    } catch (error) {
      if (error instanceof ValidationError) {
        throw error;
      }
      throw new ValidationError(
        `Failed to calculate VaR: ${error instanceof Error ? error.message : String(error)}`,
        error
      );
    }
  }

  /**
   * Calculate Conditional Value at Risk with validated inputs
   * @throws {ValidationError} If inputs are invalid
   */
  calculateCvar(returns: number[], portfolioValue: number): CVaRResult {
    try {
      const validatedInput = validateCvarInput({
        returns,
        portfolioValue,
      });

      return this.native.calculateCvar(validatedInput.returns, validatedInput.portfolioValue);
    } catch (error) {
      if (error instanceof ValidationError) {
        throw error;
      }
      throw new ValidationError(
        `Failed to calculate CVaR: ${error instanceof Error ? error.message : String(error)}`,
        error
      );
    }
  }

  /**
   * Calculate Kelly Criterion with validated inputs
   * @throws {ValidationError} If inputs are invalid
   */
  calculateKelly(winRate: number, avgWin: number, avgLoss: number): KellyResult {
    try {
      const validatedInput = validateKellyInput({
        winRate,
        avgWin,
        avgLoss,
      });

      return this.native.calculateKelly(
        validatedInput.winRate,
        validatedInput.avgWin,
        validatedInput.avgLoss
      );
    } catch (error) {
      if (error instanceof ValidationError) {
        throw error;
      }
      throw new ValidationError(
        `Failed to calculate Kelly: ${error instanceof Error ? error.message : String(error)}`,
        error
      );
    }
  }

  /**
   * Calculate drawdown metrics with validated inputs
   * @throws {ValidationError} If inputs are invalid
   */
  calculateDrawdown(equityCurve: number[]): DrawdownMetrics {
    try {
      const validatedInput = validateDrawdownInput({
        equityCurve,
      });

      return this.native.calculateDrawdown(validatedInput.equityCurve);
    } catch (error) {
      if (error instanceof ValidationError) {
        throw error;
      }
      throw new ValidationError(
        `Failed to calculate drawdown: ${error instanceof Error ? error.message : String(error)}`,
        error
      );
    }
  }

  /**
   * Calculate position size with validated inputs
   * @throws {ValidationError} If inputs are invalid
   */
  calculatePositionSize(
    portfolioValue: number,
    pricePerShare: number,
    riskPerTrade: number,
    stopLossDistance: number
  ): PositionSize {
    try {
      const validatedInput = validatePositionSizeInput({
        portfolioValue,
        pricePerShare,
        riskPerTrade,
        stopLossDistance,
      });

      return this.native.calculatePositionSize(
        validatedInput.portfolioValue,
        validatedInput.pricePerShare,
        validatedInput.riskPerTrade,
        validatedInput.stopLossDistance
      );
    } catch (error) {
      if (error instanceof ValidationError) {
        throw error;
      }
      throw new ValidationError(
        `Failed to calculate position size: ${error instanceof Error ? error.message : String(error)}`,
        error
      );
    }
  }

  /**
   * Validate position with size constraints
   * @throws {ValidationError} If position is invalid
   */
  validatePosition(
    positionSize: number,
    portfolioValue: number,
    maxPositionPercentage: number
  ): boolean {
    try {
      const validatedInput = validatePositionValue({
        positionSize,
        portfolioValue,
        maxPositionPercentage,
      });

      return this.native.validatePosition(
        validatedInput.positionSize,
        validatedInput.portfolioValue,
        validatedInput.maxPositionPercentage
      );
    } catch (error) {
      if (error instanceof ValidationError) {
        throw error;
      }
      throw new ValidationError(
        `Failed to validate position: ${error instanceof Error ? error.message : String(error)}`,
        error
      );
    }
  }
}

/**
 * Calculate Sharpe Ratio with validated inputs
 * @throws {ValidationError} If inputs are invalid
 */
export function calculateSharpeRatio(
  returns: number[],
  riskFreeRate: number,
  annualizationFactor: number
): number {
  try {
    const validatedInput = validateSharpeRatioInput({
      returns,
      riskFreeRate,
      annualizationFactor,
    });

    // Import the native function
    const { calculateSharpeRatio: nativeCalculateSharpeRatio } = require('./index');

    return nativeCalculateSharpeRatio(
      validatedInput.returns,
      validatedInput.riskFreeRate,
      validatedInput.annualizationFactor
    );
  } catch (error) {
    if (error instanceof ValidationError) {
      throw error;
    }
    throw new ValidationError(
      `Failed to calculate Sharpe Ratio: ${error instanceof Error ? error.message : String(error)}`,
      error
    );
  }
}

/**
 * Calculate Sortino Ratio with validated inputs
 * @throws {ValidationError} If inputs are invalid
 */
export function calculateSortinoRatio(
  returns: number[],
  targetReturn: number,
  annualizationFactor: number
): number {
  try {
    const validatedInput = validateSortinoRatioInput({
      returns,
      targetReturn,
      annualizationFactor,
    });

    // Import the native function
    const { calculateSortinoRatio: nativeCalculateSortinoRatio } = require('./index');

    return nativeCalculateSortinoRatio(
      validatedInput.returns,
      validatedInput.targetReturn,
      validatedInput.annualizationFactor
    );
  } catch (error) {
    if (error instanceof ValidationError) {
      throw error;
    }
    throw new ValidationError(
      `Failed to calculate Sortino Ratio: ${error instanceof Error ? error.message : String(error)}`,
      error
    );
  }
}

/**
 * Calculate maximum leverage with validated inputs
 * @throws {ValidationError} If inputs are invalid
 */
export function calculateMaxLeverage(
  portfolioValue: number,
  volatility: number,
  maxVolatilityTarget: number
): number {
  try {
    const validatedInput = validateMaxLeverageInput({
      portfolioValue,
      volatility,
      maxVolatilityTarget,
    });

    // Import the native function
    const { calculateMaxLeverage: nativeCalculateMaxLeverage } = require('./index');

    return nativeCalculateMaxLeverage(
      validatedInput.portfolioValue,
      validatedInput.volatility,
      validatedInput.maxVolatilityTarget
    );
  } catch (error) {
    if (error instanceof ValidationError) {
      throw error;
    }
    throw new ValidationError(
      `Failed to calculate max leverage: ${error instanceof Error ? error.message : String(error)}`,
      error
    );
  }
}
