// Type definitions for @neural-trader/sports-betting
import type { RiskConfig, KellyResult } from '@neural-trader/core';

export { RiskConfig, KellyResult };

export class RiskManager {
  constructor(config: RiskConfig);
  calculateKelly(winRate: number, avgWin: number, avgLoss: number): KellyResult;
}
