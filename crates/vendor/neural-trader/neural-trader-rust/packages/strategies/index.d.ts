// Type definitions for @neural-trader/strategies
import type { Signal, StrategyConfig } from '@neural-trader/core';

export { Signal, StrategyConfig };

export class StrategyRunner {
  constructor();
  addMomentumStrategy(config: StrategyConfig): Promise<string>;
  addMeanReversionStrategy(config: StrategyConfig): Promise<string>;
  addArbitrageStrategy(config: StrategyConfig): Promise<string>;
  generateSignals(): Promise<Signal[]>;
  subscribeSignals(callback: (signal: Signal) => void): SubscriptionHandle;
  listStrategies(): Promise<string[]>;
  removeStrategy(strategyId: string): Promise<boolean>;
}

export class SubscriptionHandle {
  unsubscribe(): Promise<void>;
}
