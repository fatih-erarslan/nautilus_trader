// Type definitions for @neural-trader/backtesting
import type {
  BacktestConfig,
  BacktestResult,
  BacktestMetrics,
  Signal,
  Trade
} from '@neural-trader/core';

export { BacktestConfig, BacktestResult, BacktestMetrics, Signal, Trade };

export class BacktestEngine {
  constructor(config: BacktestConfig);
  run(signals: Signal[], marketData: string): Promise<BacktestResult>;
  calculateMetrics(equityCurve: number[]): BacktestMetrics;
  exportTradesCsv(trades: Trade[]): string;
}

export function compareBacktests(results: BacktestResult[]): string;
