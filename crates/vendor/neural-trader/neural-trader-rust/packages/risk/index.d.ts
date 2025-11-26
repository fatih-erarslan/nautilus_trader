// Type definitions for @neural-trader/risk
import type {
  RiskConfig,
  VaRResult,
  CVaRResult,
  DrawdownMetrics,
  KellyResult,
  PositionSize
} from '@neural-trader/core';

export { RiskConfig, VaRResult, CVaRResult, DrawdownMetrics, KellyResult, PositionSize };

export class RiskManager {
  constructor(config: RiskConfig);
  calculateVar(returns: number[], portfolioValue: number): VaRResult;
  calculateCvar(returns: number[], portfolioValue: number): CVaRResult;
  calculateKelly(winRate: number, avgWin: number, avgLoss: number): KellyResult;
  calculateDrawdown(equityCurve: number[]): DrawdownMetrics;
  calculatePositionSize(
    portfolioValue: number,
    pricePerShare: number,
    riskPerTrade: number,
    stopLossDistance: number
  ): PositionSize;
  validatePosition(
    positionSize: number,
    portfolioValue: number,
    maxPositionPercentage: number
  ): boolean;
}

export function calculateSharpeRatio(
  returns: number[],
  riskFreeRate: number,
  annualizationFactor: number
): number;

export function calculateSortinoRatio(
  returns: number[],
  targetReturn: number,
  annualizationFactor: number
): number;

export function calculateMaxLeverage(
  portfolioValue: number,
  volatility: number,
  maxVolatilityTarget: number
): number;
