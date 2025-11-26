/**
 * Mean reversion strategy implementation
 * Trades when price deviates significantly from moving average
 */

import { Strategy, MarketData, StrategySignal } from '../types';

export class MeanReversionStrategy implements Strategy {
  name = 'mean-reversion';
  type = 'mean-reversion';
  private parameters: {
    maPeriod: number;
    stdDevMultiplier: number;
    minDeviation: number;
    exitAtMean: boolean;
  };

  constructor(parameters?: Partial<MeanReversionStrategy['parameters']>) {
    this.parameters = {
      maPeriod: 20,
      stdDevMultiplier: 2.0,
      minDeviation: 0.015,
      exitAtMean: true,
      ...parameters
    };
  }

  generateSignal(data: MarketData[], parameters: Record<string, any>): StrategySignal {
    if (data.length < this.parameters.maPeriod) {
      return this.neutralSignal(data[data.length - 1]);
    }

    const currentBar = data[data.length - 1];
    const closes = data.slice(-this.parameters.maPeriod).map(d => d.close);

    // Calculate moving average
    const ma = closes.reduce((sum, price) => sum + price, 0) / closes.length;

    // Calculate standard deviation
    const squaredDiffs = closes.map(price => Math.pow(price - ma, 2));
    const variance = squaredDiffs.reduce((sum, diff) => sum + diff, 0) / closes.length;
    const stdDev = Math.sqrt(variance);

    // Calculate deviation from mean
    const deviation = (currentBar.close - ma) / ma;
    const zScore = (currentBar.close - ma) / stdDev;

    // Generate signal
    let action: 'buy' | 'sell' | 'hold' = 'hold';
    let strength = 0;
    let confidence = 0;

    const threshold = this.parameters.stdDevMultiplier;

    if (zScore < -threshold && Math.abs(deviation) > this.parameters.minDeviation) {
      // Price significantly below mean - buy
      action = 'buy';
      strength = Math.min(Math.abs(zScore) / threshold, 1);
      confidence = 0.75;
    } else if (zScore > threshold && Math.abs(deviation) > this.parameters.minDeviation) {
      // Price significantly above mean - sell
      action = 'sell';
      strength = Math.min(Math.abs(zScore) / threshold, 1);
      confidence = 0.75;
    } else if (this.parameters.exitAtMean && Math.abs(zScore) < 0.5) {
      // Price near mean - exit position
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
        ma,
        stdDev,
        zScore,
        deviation
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
