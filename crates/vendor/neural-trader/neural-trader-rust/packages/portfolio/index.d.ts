// Type definitions for @neural-trader/portfolio
import type {
  Position,
  PortfolioOptimization,
  RiskMetrics,
  OptimizerConfig
} from '@neural-trader/core';

export { Position, PortfolioOptimization, RiskMetrics, OptimizerConfig };

export class PortfolioManager {
  constructor(initialCash: number);
  getPositions(): Promise<Position[]>;
  getPosition(symbol: string): Promise<Position | null>;
  updatePosition(symbol: string, quantity: number, price: number): Promise<Position>;
  getCash(): Promise<number>;
  getTotalValue(): Promise<number>;
  getTotalPnl(): Promise<number>;
}

export class PortfolioOptimizer {
  constructor(config: OptimizerConfig);
  optimize(
    symbols: string[],
    returns: number[],
    covariance: number[]
  ): Promise<PortfolioOptimization>;
  calculateRisk(positions: Record<string, number>): RiskMetrics;
}
