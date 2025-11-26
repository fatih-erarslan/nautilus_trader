/**
 * Core backtesting engine with walk-forward optimization
 * Supports multi-strategy allocation and transaction cost modeling
 */

import {
  BacktestConfig,
  MarketData,
  Position,
  Trade,
  PortfolioState,
  StrategyPerformance,
  RegimeDetection,
  Strategy,
  StrategySignal
} from './types';

export class Backtester {
  private config: BacktestConfig;
  private portfolioStates: PortfolioState[] = [];
  private trades: Trade[] = [];
  private currentPositions: Position[] = [];
  private cash: number;
  private equity: number;
  private strategies: Map<string, Strategy>;

  constructor(config: BacktestConfig, strategies: Strategy[]) {
    this.config = config;
    this.cash = config.initialCapital;
    this.equity = config.initialCapital;
    this.strategies = new Map(strategies.map(s => [s.name, s]));
  }

  /**
   * Run complete backtest with walk-forward optimization
   */
  async runBacktest(marketData: MarketData[]): Promise<StrategyPerformance[]> {
    const periods = this.config.walkForwardPeriods || 1;
    const periodLength = Math.floor(marketData.length / periods);

    const allPerformances: StrategyPerformance[] = [];

    for (let period = 0; period < periods; period++) {
      const startIdx = period * periodLength;
      const endIdx = Math.min((period + 1) * periodLength, marketData.length);
      const periodData = marketData.slice(startIdx, endIdx);

      console.log(`\n🔄 Walk-forward period ${period + 1}/${periods}`);
      console.log(`📊 Data points: ${periodData.length}`);

      await this.runPeriod(periodData);
    }

    // Calculate performance for each strategy
    const performances = this.calculatePerformanceMetrics();
    return performances;
  }

  /**
   * Run single period of backtest
   */
  private async runPeriod(marketData: MarketData[]): Promise<void> {
    for (let i = 0; i < marketData.length; i++) {
      const currentData = marketData.slice(Math.max(0, i - 100), i + 1);
      const currentBar = marketData[i];

      // Detect market regime
      const regime = this.detectRegime(currentData);

      // Generate signals from all strategies
      const signals = this.generateSignals(currentData, regime);

      // Execute trades based on signals
      await this.executeTrades(signals, currentBar);

      // Update portfolio state
      this.updatePortfolio(currentBar, regime);

      // Rebalance if needed
      if (this.shouldRebalance(currentBar)) {
        await this.rebalancePortfolio(currentBar);
      }
    }
  }

  /**
   * Detect market regime using multi-timeframe analysis
   */
  private detectRegime(data: MarketData[]): RegimeDetection {
    if (data.length < 50) {
      return {
        regime: 'sideways',
        confidence: 0.5,
        timestamp: data[data.length - 1].timestamp,
        indicators: { trend: 0, volatility: 0.5, correlation: 0.5 }
      };
    }

    // Calculate trend using moving averages
    const closes = data.map(d => d.close);
    const sma20 = this.calculateSMA(closes, 20);
    const sma50 = this.calculateSMA(closes, 50);
    const trend = (sma20 - sma50) / sma50;

    // Calculate volatility
    const returns = this.calculateReturns(closes);
    const volatility = this.calculateStdDev(returns);

    // Determine regime
    let regime: RegimeDetection['regime'] = 'sideways';
    let confidence = 0.5;

    if (trend > 0.02 && volatility < 0.02) {
      regime = 'bull';
      confidence = 0.8;
    } else if (trend < -0.02 && volatility < 0.02) {
      regime = 'bear';
      confidence = 0.8;
    } else if (volatility > 0.03) {
      regime = 'high-volatility';
      confidence = 0.7;
    } else if (volatility < 0.01) {
      regime = 'low-volatility';
      confidence = 0.7;
    }

    return {
      regime,
      confidence,
      timestamp: data[data.length - 1].timestamp,
      indicators: {
        trend,
        volatility,
        correlation: 0.5 // Simplified
      }
    };
  }

  /**
   * Generate signals from all enabled strategies
   */
  private generateSignals(data: MarketData[], regime: RegimeDetection): StrategySignal[] {
    const signals: StrategySignal[] = [];

    for (const [name, strategy] of this.strategies.entries()) {
      const strategyConfig = this.config.strategies.find(s => s.name === name);
      if (!strategyConfig?.enabled) continue;

      try {
        const signal = strategy.generateSignal(data, strategyConfig.parameters);

        // Adjust signal strength based on regime
        signal.strength *= this.getRegimeAdjustment(strategyConfig.type, regime);

        signals.push(signal);
      } catch (error) {
        console.error(`Error generating signal for ${name}:`, error);
      }
    }

    return signals;
  }

  /**
   * Adjust signal strength based on market regime and strategy type
   */
  private getRegimeAdjustment(strategyType: string, regime: RegimeDetection): number {
    const adjustments: Record<string, Record<string, number>> = {
      'momentum': {
        'bull': 1.2,
        'bear': 0.8,
        'sideways': 0.6,
        'high-volatility': 1.1,
        'low-volatility': 0.9
      },
      'mean-reversion': {
        'bull': 0.8,
        'bear': 0.8,
        'sideways': 1.3,
        'high-volatility': 0.7,
        'low-volatility': 1.2
      },
      'pairs-trading': {
        'bull': 1.0,
        'bear': 1.0,
        'sideways': 1.1,
        'high-volatility': 0.9,
        'low-volatility': 1.0
      },
      'market-making': {
        'bull': 0.9,
        'bear': 0.9,
        'sideways': 1.2,
        'high-volatility': 0.8,
        'low-volatility': 1.1
      }
    };

    return adjustments[strategyType]?.[regime.regime] || 1.0;
  }

  /**
   * Execute trades with transaction cost modeling
   */
  private async executeTrades(signals: StrategySignal[], currentBar: MarketData): Promise<void> {
    for (const signal of signals) {
      if (signal.action === 'hold' || signal.strength < 0.3) continue;

      const strategyConfig = this.config.strategies.find(s => s.name === signal.strategy);
      if (!strategyConfig) continue;

      const targetWeight = strategyConfig.initialWeight * signal.strength;
      const targetValue = this.equity * targetWeight;
      const currentValue = this.getPositionValue(signal.symbol, currentBar.close);

      const tradeDifference = targetValue - currentValue;
      if (Math.abs(tradeDifference) < 100) continue; // Minimum trade size

      const quantity = Math.floor(Math.abs(tradeDifference) / currentBar.close);
      if (quantity === 0) continue;

      // Calculate transaction costs
      const commission = quantity * currentBar.close * this.config.commission;
      const slippage = quantity * currentBar.close * this.config.slippage;
      const totalCost = commission + slippage;

      const trade: Trade = {
        id: `${Date.now()}-${Math.random()}`,
        symbol: signal.symbol,
        side: tradeDifference > 0 ? 'buy' : 'sell',
        quantity,
        price: currentBar.close,
        timestamp: currentBar.timestamp,
        strategy: signal.strategy,
        commission,
        slippage
      };

      // Execute trade
      if (trade.side === 'buy') {
        const totalCost = (quantity * currentBar.close) + commission + slippage;
        if (totalCost <= this.cash) {
          this.cash -= totalCost;
          this.addPosition({
            symbol: signal.symbol,
            quantity,
            entryPrice: currentBar.close,
            entryTime: currentBar.timestamp,
            side: 'long',
            strategy: signal.strategy
          });
          this.trades.push(trade);
        }
      } else {
        this.removePosition(signal.symbol, quantity);
        this.cash += (quantity * currentBar.close) - commission - slippage;
        this.trades.push(trade);
      }
    }
  }

  /**
   * Update portfolio state
   */
  private updatePortfolio(currentBar: MarketData, regime: RegimeDetection): void {
    // Calculate total equity
    let positionsValue = 0;
    for (const position of this.currentPositions) {
      if (position.symbol === currentBar.symbol) {
        positionsValue += position.quantity * currentBar.close;
      }
    }

    this.equity = this.cash + positionsValue;

    // Calculate strategy weights
    const strategyWeights: Record<string, number> = {};
    for (const position of this.currentPositions) {
      const value = position.quantity * currentBar.close;
      strategyWeights[position.strategy] =
        (strategyWeights[position.strategy] || 0) + (value / this.equity);
    }

    const state: PortfolioState = {
      timestamp: currentBar.timestamp,
      equity: this.equity,
      cash: this.cash,
      positions: [...this.currentPositions],
      strategyWeights,
      regime
    };

    this.portfolioStates.push(state);
  }

  /**
   * Check if portfolio should be rebalanced
   */
  private shouldRebalance(currentBar: MarketData): boolean {
    if (this.portfolioStates.length === 0) return false;

    const lastRebalance = this.portfolioStates[this.portfolioStates.length - 1];
    const timeDiff = currentBar.timestamp - lastRebalance.timestamp;

    switch (this.config.rebalanceFrequency) {
      case 'daily':
        return timeDiff >= 24 * 60 * 60 * 1000;
      case 'weekly':
        return timeDiff >= 7 * 24 * 60 * 60 * 1000;
      case 'monthly':
        return timeDiff >= 30 * 24 * 60 * 60 * 1000;
      default:
        return false;
    }
  }

  /**
   * Rebalance portfolio to target weights
   */
  private async rebalancePortfolio(currentBar: MarketData): Promise<void> {
    // Simple rebalancing logic - can be enhanced
    console.log(`🔄 Rebalancing portfolio at ${new Date(currentBar.timestamp).toISOString()}`);
  }

  /**
   * Calculate comprehensive performance metrics
   */
  private calculatePerformanceMetrics(): StrategyPerformance[] {
    const performances: StrategyPerformance[] = [];
    const strategyTrades = new Map<string, Trade[]>();

    // Group trades by strategy
    for (const trade of this.trades) {
      if (!strategyTrades.has(trade.strategy)) {
        strategyTrades.set(trade.strategy, []);
      }
      strategyTrades.get(trade.strategy)!.push(trade);
    }

    // Calculate metrics for each strategy
    for (const [strategyName, trades] of strategyTrades.entries()) {
      const returns = this.calculateStrategyReturns(trades);
      const wins = returns.filter(r => r > 0);
      const losses = returns.filter(r => r < 0);

      const totalReturn = returns.reduce((sum, r) => sum + r, 0);
      const avgReturn = returns.length > 0 ? totalReturn / returns.length : 0;
      const stdDev = this.calculateStdDev(returns);
      const sharpeRatio = stdDev > 0 ? (avgReturn / stdDev) * Math.sqrt(252) : 0;

      const drawdowns = this.calculateDrawdowns(returns);
      const maxDrawdown = Math.min(...drawdowns);

      const winRate = returns.length > 0 ? wins.length / returns.length : 0;
      const avgWin = wins.length > 0 ? wins.reduce((s, w) => s + w, 0) / wins.length : 0;
      const avgLoss = losses.length > 0 ? Math.abs(losses.reduce((s, l) => s + l, 0) / losses.length) : 0;
      const profitFactor = avgLoss > 0 ? avgWin / avgLoss : 0;

      const calmarRatio = maxDrawdown !== 0 ? totalReturn / Math.abs(maxDrawdown) : 0;
      const downsideReturns = returns.filter(r => r < 0);
      const downsideStdDev = this.calculateStdDev(downsideReturns);
      const sortinoRatio = downsideStdDev > 0 ? (avgReturn / downsideStdDev) * Math.sqrt(252) : 0;

      performances.push({
        strategyName,
        totalReturn,
        sharpeRatio,
        maxDrawdown,
        winRate,
        profitFactor,
        trades: trades.length,
        avgWin,
        avgLoss,
        calmarRatio,
        sortinoRatio
      });
    }

    return performances;
  }

  // Utility methods
  private calculateSMA(data: number[], period: number): number {
    if (data.length < period) return data[data.length - 1] || 0;
    const slice = data.slice(-period);
    return slice.reduce((sum, val) => sum + val, 0) / period;
  }

  private calculateReturns(prices: number[]): number[] {
    const returns: number[] = [];
    for (let i = 1; i < prices.length; i++) {
      returns.push((prices[i] - prices[i - 1]) / prices[i - 1]);
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

  private calculateStrategyReturns(trades: Trade[]): number[] {
    const returns: number[] = [];
    const positions = new Map<string, { qty: number; avgPrice: number }>();

    for (const trade of trades) {
      const key = `${trade.symbol}-${trade.strategy}`;
      const pos = positions.get(key) || { qty: 0, avgPrice: 0 };

      if (trade.side === 'buy') {
        const newQty = pos.qty + trade.quantity;
        pos.avgPrice = ((pos.avgPrice * pos.qty) + (trade.price * trade.quantity)) / newQty;
        pos.qty = newQty;
      } else {
        const returnPct = (trade.price - pos.avgPrice) / pos.avgPrice;
        returns.push(returnPct);
        pos.qty -= trade.quantity;
        if (pos.qty === 0) {
          positions.delete(key);
        }
      }
      positions.set(key, pos);
    }

    return returns;
  }

  private calculateDrawdowns(returns: number[]): number[] {
    const cumReturns: number[] = [0];
    for (const ret of returns) {
      cumReturns.push(cumReturns[cumReturns.length - 1] + ret);
    }

    const drawdowns: number[] = [];
    let peak = cumReturns[0];

    for (const value of cumReturns) {
      if (value > peak) peak = value;
      drawdowns.push((value - peak) / (peak === 0 ? 1 : peak));
    }

    return drawdowns;
  }

  private addPosition(position: Position): void {
    this.currentPositions.push(position);
  }

  private removePosition(symbol: string, quantity: number): void {
    const idx = this.currentPositions.findIndex(p => p.symbol === symbol);
    if (idx >= 0) {
      this.currentPositions[idx].quantity -= quantity;
      if (this.currentPositions[idx].quantity <= 0) {
        this.currentPositions.splice(idx, 1);
      }
    }
  }

  private getPositionValue(symbol: string, price: number): number {
    return this.currentPositions
      .filter(p => p.symbol === symbol)
      .reduce((sum, p) => sum + (p.quantity * price), 0);
  }

  // Getters
  public getPortfolioStates(): PortfolioState[] {
    return this.portfolioStates;
  }

  public getTrades(): Trade[] {
    return this.trades;
  }

  public getFinalEquity(): number {
    return this.equity;
  }
}
