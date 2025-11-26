/**
 * Market making strategy implementation
 * Provides liquidity by placing bid/ask orders
 */

import { Strategy, MarketData, StrategySignal } from '../types';

export class MarketMakingStrategy implements Strategy {
  name = 'market-making';
  type = 'market-making';
  private parameters: {
    spreadBps: number;
    inventoryLimit: number;
    volatilityAdjustment: boolean;
    minVolume: number;
  };

  constructor(parameters?: Partial<MarketMakingStrategy['parameters']>) {
    this.parameters = {
      spreadBps: 10, // 0.1%
      inventoryLimit: 1000,
      volatilityAdjustment: true,
      minVolume: 100000,
      ...parameters
    };
  }

  generateSignal(data: MarketData[], parameters: Record<string, any>): StrategySignal {
    if (data.length < 20) {
      return this.neutralSignal(data[data.length - 1]);
    }

    const currentBar = data[data.length - 1];

    // Check minimum volume requirement
    if (currentBar.volume < this.parameters.minVolume) {
      return this.neutralSignal(currentBar);
    }

    // Calculate volatility if adjustment is enabled
    let spreadAdjustment = 1.0;
    if (this.parameters.volatilityAdjustment) {
      const returns = this.calculateReturns(data.slice(-20));
      const volatility = this.calculateStdDev(returns);
      spreadAdjustment = 1.0 + (volatility * 100); // Scale by volatility
    }

    // Calculate bid/ask spread
    const spread = (this.parameters.spreadBps / 10000) * spreadAdjustment;
    const midPrice = (currentBar.high + currentBar.low) / 2;
    const bidPrice = midPrice * (1 - spread / 2);
    const askPrice = midPrice * (1 + spread / 2);

    // Determine action based on current price relative to mid
    let action: 'buy' | 'sell' | 'hold' = 'hold';
    let strength = 0;
    let confidence = 0;

    if (currentBar.close < midPrice) {
      // Price below mid - buy to provide liquidity
      action = 'buy';
      strength = (midPrice - currentBar.close) / (spread * midPrice);
      confidence = 0.65;
    } else if (currentBar.close > midPrice) {
      // Price above mid - sell to provide liquidity
      action = 'sell';
      strength = (currentBar.close - midPrice) / (spread * midPrice);
      confidence = 0.65;
    }

    // Limit strength to reasonable range
    strength = Math.max(0, Math.min(1, strength));

    return {
      symbol: currentBar.symbol,
      action,
      strength,
      confidence,
      strategy: this.name,
      timestamp: currentBar.timestamp,
      metadata: {
        midPrice,
        bidPrice,
        askPrice,
        spread: spread * 10000, // In basis points
        spreadAdjustment
      }
    };
  }

  updateParameters(parameters: Record<string, any>): void {
    this.parameters = { ...this.parameters, ...parameters };
  }

  getParameters(): Record<string, any> {
    return { ...this.parameters };
  }

  private calculateReturns(data: MarketData[]): number[] {
    const returns: number[] = [];
    for (let i = 1; i < data.length; i++) {
      returns.push((data[i].close - data[i - 1].close) / data[i - 1].close);
    }
    return returns;
  }

  private calculateStdDev(values: number[]): number {
    if (values.length === 0) return 0;
    const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
    const squaredDiffs = values.map(val => Math.pow(val - mean, 2));
    const variance = squaredDiffs.reduce((sum, val) => sum + val, 0) / values.length;
    return Math.sqrt(variance);
  }

  private neutralSignal(bar: MarketData): StrategySignal {
    return {
      symbol: bar.symbol,
      action: 'hold',
      strength: 0,
      confidence: 0,
      strategy: this.name,
      timestamp: bar.timestamp
    };
  }
}
