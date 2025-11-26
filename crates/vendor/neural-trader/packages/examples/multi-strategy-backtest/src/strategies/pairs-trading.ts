/**
 * Pairs trading strategy implementation
 * Trades based on spread between correlated assets
 */

import { Strategy, MarketData, StrategySignal } from '../types';

export class PairsTradingStrategy implements Strategy {
  name = 'pairs-trading';
  type = 'pairs-trading';
  private parameters: {
    lookbackPeriod: number;
    entryZScore: number;
    exitZScore: number;
    minCorrelation: number;
  };

  constructor(parameters?: Partial<PairsTradingStrategy['parameters']>) {
    this.parameters = {
      lookbackPeriod: 60,
      entryZScore: 2.0,
      exitZScore: 0.5,
      minCorrelation: 0.7,
      ...parameters
    };
  }

  generateSignal(data: MarketData[], parameters: Record<string, any>): StrategySignal {
    // Note: Pairs trading requires two assets, this is a simplified version
    if (data.length < this.parameters.lookbackPeriod) {
      return this.neutralSignal(data[data.length - 1]);
    }

    const currentBar = data[data.length - 1];
    const historicalData = data.slice(-this.parameters.lookbackPeriod);

    // Calculate price ratio (simplified - would need second asset)
    const prices = historicalData.map(d => d.close);
    const meanPrice = prices.reduce((sum, p) => sum + p, 0) / prices.length;

    // Calculate spread z-score
    const squaredDiffs = prices.map(p => Math.pow(p - meanPrice, 2));
    const variance = squaredDiffs.reduce((sum, d) => sum + d, 0) / prices.length;
    const stdDev = Math.sqrt(variance);

    const zScore = (currentBar.close - meanPrice) / stdDev;

    // Generate signal
    let action: 'buy' | 'sell' | 'hold' = 'hold';
    let strength = 0;
    let confidence = 0;

    if (zScore < -this.parameters.entryZScore) {
      // Spread too low - buy
      action = 'buy';
      strength = Math.min(Math.abs(zScore) / this.parameters.entryZScore, 1);
      confidence = 0.7;
    } else if (zScore > this.parameters.entryZScore) {
      // Spread too high - sell
      action = 'sell';
      strength = Math.min(Math.abs(zScore) / this.parameters.entryZScore, 1);
      confidence = 0.7;
    } else if (Math.abs(zScore) < this.parameters.exitZScore) {
      // Spread normalized - exit
      action = 'sell';
      strength = 0.3;
      confidence = 0.6;
    }

    return {
      symbol: currentBar.symbol,
      action,
      strength,
      confidence,
      strategy: this.name,
      timestamp: currentBar.timestamp,
      metadata: {
        zScore,
        meanPrice,
        stdDev
      }
    };
  }

  updateParameters(parameters: Record<string, any>): void {
    this.parameters = { ...this.parameters, ...parameters };
  }

  getParameters(): Record<string, any> {
    return { ...this.parameters };
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
