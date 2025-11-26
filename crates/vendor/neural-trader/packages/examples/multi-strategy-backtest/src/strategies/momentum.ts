/**
 * Momentum strategy implementation
 * Buys assets with strong upward momentum
 */

import { Strategy, MarketData, StrategySignal } from '../types';

export class MomentumStrategy implements Strategy {
  name = 'momentum';
  type = 'momentum';
  private parameters: {
    lookbackPeriod: number;
    entryThreshold: number;
    exitThreshold: number;
    volumeConfirmation: boolean;
  };

  constructor(parameters?: Partial<MomentumStrategy['parameters']>) {
    this.parameters = {
      lookbackPeriod: 20,
      entryThreshold: 0.02,
      exitThreshold: -0.01,
      volumeConfirmation: true,
      ...parameters
    };
  }

  generateSignal(data: MarketData[], parameters: Record<string, any>): StrategySignal {
    if (data.length < this.parameters.lookbackPeriod) {
      return this.neutralSignal(data[data.length - 1]);
    }

    const currentBar = data[data.length - 1];
    const lookbackBar = data[data.length - this.parameters.lookbackPeriod];

    // Calculate momentum
    const momentum = (currentBar.close - lookbackBar.close) / lookbackBar.close;

    // Calculate volume confirmation
    let volumeConfirmed = true;
    if (this.parameters.volumeConfirmation) {
      const avgVolume = data.slice(-this.parameters.lookbackPeriod)
        .reduce((sum, bar) => sum + bar.volume, 0) / this.parameters.lookbackPeriod;
      volumeConfirmed = currentBar.volume > avgVolume * 1.2;
    }

    // Generate signal
    let action: 'buy' | 'sell' | 'hold' = 'hold';
    let strength = 0;
    let confidence = 0;

    if (momentum > this.parameters.entryThreshold && volumeConfirmed) {
      action = 'buy';
      strength = Math.min(momentum / this.parameters.entryThreshold, 1);
      confidence = volumeConfirmed ? 0.8 : 0.6;
    } else if (momentum < this.parameters.exitThreshold) {
      action = 'sell';
      strength = Math.min(Math.abs(momentum) / Math.abs(this.parameters.exitThreshold), 1);
      confidence = 0.7;
    }

    return {
      symbol: currentBar.symbol,
      action,
      strength,
      confidence,
      strategy: this.name,
      timestamp: currentBar.timestamp,
      metadata: {
        momentum,
        volumeConfirmed,
        lookbackPeriod: this.parameters.lookbackPeriod
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
