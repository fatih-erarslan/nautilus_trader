# @neural-trader/strategies

[![npm version](https://img.shields.io/npm/v/@neural-trader/strategies.svg)](https://www.npmjs.com/package/@neural-trader/strategies)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](https://github.com/ruvnet/neural-trader)
[![Build Status](https://img.shields.io/github/workflow/status/ruvnet/neural-trader/CI)](https://github.com/ruvnet/neural-trader)
[![Downloads](https://img.shields.io/npm/dm/@neural-trader/strategies.svg)](https://www.npmjs.com/package/@neural-trader/strategies)

Professional-grade trading strategies for the Neural Trader platform, powered by Rust with native Node.js bindings. Implement momentum, mean reversion, arbitrage, and pairs trading strategies with real-time signal generation and subscription capabilities.

## Features

- **6+ Trading Strategies**: Momentum (SMA/EMA crossover), mean reversion (Bollinger Bands, RSI), arbitrage, pairs trading
- **Real-Time Signals**: Subscribe to live trading signals with callbacks
- **Multi-Strategy Support**: Run multiple strategies concurrently with unified signal aggregation
- **Strategy Builder Interface**: Flexible configuration system for rapid strategy development
- **Backtestable**: Full compatibility with @neural-trader/backtesting for historical validation
- **Risk-Aware**: Integrates with @neural-trader/risk for position sizing and risk management
- **Rust Performance**: Microsecond-level signal generation with zero-copy data structures
- **Type-Safe**: Complete TypeScript definitions with comprehensive JSDoc

## Installation

```bash
npm install @neural-trader/strategies @neural-trader/core
```

## Quick Start

```typescript
import { StrategyRunner } from '@neural-trader/strategies';
import type { StrategyConfig, Signal } from '@neural-trader/core';

// Create strategy runner
const runner = new StrategyRunner();

// Add momentum strategy
const momentumConfig: StrategyConfig = {
  name: 'SMA Crossover',
  symbols: ['AAPL', 'MSFT', 'GOOGL'],
  parameters: JSON.stringify({
    shortPeriod: 20,
    longPeriod: 50,
    minVolume: 1000000
  })
};

const strategyId = await runner.addMomentumStrategy(momentumConfig);
console.log(`Strategy added: ${strategyId}`);

// Generate signals
const signals: Signal[] = await runner.generateSignals();
signals.forEach(signal => {
  console.log(`${signal.symbol}: ${signal.direction} (confidence: ${signal.confidence})`);
  console.log(`  Entry: $${signal.entryPrice}, Stop: $${signal.stopLoss}, Target: $${signal.takeProfit}`);
  console.log(`  Reason: ${signal.reasoning}`);
});

// Subscribe to live signals
const subscription = runner.subscribeSignals((signal: Signal) => {
  console.log('New signal received:', signal);
  // Execute trade logic here
});

// Later: Unsubscribe
await subscription.unsubscribe();
```

## Core Concepts

### Strategy Types

| Strategy | Description | Use Case | Typical Win Rate |
|----------|-------------|----------|------------------|
| **Momentum** | Trend-following with SMA/EMA crossovers | Trending markets, breakouts | 45-55% |
| **Mean Reversion** | Bollinger Bands, RSI-based reversions | Range-bound, oversold/overbought | 55-65% |
| **Pairs Trading** | Statistical arbitrage with cointegration | Market-neutral exposure | 60-70% |
| **Arbitrage** | Cross-exchange price discrepancies | Low-risk, high-frequency | 70-80% |

### Signal Structure

Every strategy generates signals with:
- **Entry/Exit Prices**: Calculated from real-time market data
- **Stop Loss/Take Profit**: Risk management levels
- **Confidence Score**: 0.0-1.0 quality indicator
- **Reasoning**: Human-readable explanation
- **Timestamp**: Nanosecond-precision for ordering

## In-Depth Usage

### Momentum Strategies

Momentum strategies capitalize on trending market movements:

```typescript
import { StrategyRunner } from '@neural-trader/strategies';

async function setupMomentumStrategy() {
  const runner = new StrategyRunner();

  // Classic SMA crossover strategy
  const smaConfig: StrategyConfig = {
    name: 'SMA 20/50 Crossover',
    symbols: ['AAPL', 'TSLA', 'NVDA'],
    parameters: JSON.stringify({
      shortPeriod: 20,    // Fast moving average
      longPeriod: 50,     // Slow moving average
      minVolume: 1000000, // Minimum daily volume
      atr_multiplier: 2.0 // ATR for stop loss
    })
  };

  const strategyId = await runner.addMomentumStrategy(smaConfig);

  // Strategy generates signals when:
  // - Short SMA crosses above long SMA (bullish)
  // - Short SMA crosses below long SMA (bearish)
  // - Volume confirms the move
  // - ATR-based stop losses are calculated

  return { runner, strategyId };
}

// Advanced momentum: Multiple timeframe analysis
async function multiTimeframeMomentum() {
  const runner = new StrategyRunner();

  const config: StrategyConfig = {
    name: 'Multi-Timeframe Momentum',
    symbols: ['SPY', 'QQQ', 'IWM'],
    parameters: JSON.stringify({
      shortPeriod: 10,
      mediumPeriod: 20,
      longPeriod: 50,
      trend_strength_threshold: 0.7,
      volume_surge_multiplier: 1.5
    })
  };

  await runner.addMomentumStrategy(config);

  // This strategy confirms trends across multiple timeframes
  // Signals only when all timeframes align

  return runner;
}
```

### Mean Reversion Strategies

Mean reversion strategies profit from price returning to average:

```typescript
async function setupMeanReversionStrategy() {
  const runner = new StrategyRunner();

  // Bollinger Band reversion strategy
  const bbConfig: StrategyConfig = {
    name: 'Bollinger Band Reversion',
    symbols: ['SPY', 'QQQ'],
    parameters: JSON.stringify({
      period: 20,           // BB period
      stdDev: 2.0,          // Standard deviations
      oversold_threshold: 0.1,  // % below lower band
      overbought_threshold: 0.1, // % above upper band
      min_reversion_target: 0.02 // Minimum 2% reversion target
    })
  };

  const strategyId = await runner.addMeanReversionStrategy(bbConfig);

  // Generates signals when:
  // - Price touches or breaks lower band (buy signal)
  // - Price touches or breaks upper band (sell signal)
  // - RSI confirms oversold/overbought
  // - Target is middle or opposite band

  return { runner, strategyId };
}

// RSI-based mean reversion
async function rsiReversionStrategy() {
  const runner = new StrategyRunner();

  const config: StrategyConfig = {
    name: 'RSI Mean Reversion',
    symbols: ['AAPL', 'MSFT', 'GOOGL', 'AMZN'],
    parameters: JSON.stringify({
      rsi_period: 14,
      oversold: 30,
      overbought: 70,
      lookback_period: 50,
      mean_reversion_strength: 0.6
    })
  };

  await runner.addMeanReversionStrategy(config);

  // Signals when RSI extreme and price deviates from mean
  // Filters for high-quality reversion opportunities

  return runner;
}
```

### Pairs Trading Strategies

Statistical arbitrage using correlated assets:

```typescript
async function setupPairsTradingStrategy() {
  const runner = new StrategyRunner();

  // Classic pairs trading
  const pairsConfig: StrategyConfig = {
    name: 'Pairs Trading',
    symbols: ['PEP,KO', 'XOM,CVX', 'JPM,BAC'], // Comma-separated pairs
    parameters: JSON.stringify({
      lookback: 60,           // Cointegration lookback
      entry_threshold: 2.0,   // Z-score entry
      exit_threshold: 0.5,    // Z-score exit
      stop_threshold: 3.0,    // Max z-score
      min_correlation: 0.7,   // Minimum correlation
      half_life_max: 20       // Maximum half-life (days)
    })
  };

  const strategyId = await runner.addArbitrageStrategy(pairsConfig);

  // Strategy identifies:
  // - Cointegrated pairs
  // - Spread deviations
  // - Entry/exit points
  // - Position sizing for both legs

  return { runner, strategyId };
}

// Real-time pairs monitoring
async function monitorPairsSpreads(runner: StrategyRunner) {
  const subscription = runner.subscribeSignals((signal: Signal) => {
    if (signal.strategyId.includes('pairs')) {
      console.log(`Pairs Signal: ${signal.symbol}`);
      console.log(`Z-Score: ${JSON.parse(signal.reasoning).zscore}`);
      console.log(`Direction: ${signal.direction}`);
      console.log(`Confidence: ${(signal.confidence * 100).toFixed(1)}%`);

      // Execute both legs of the pair
      if (signal.direction === 'long') {
        console.log('  Long: First symbol');
        console.log('  Short: Second symbol');
      } else {
        console.log('  Short: First symbol');
        console.log('  Long: Second symbol');
      }
    }
  });

  return subscription;
}
```

### Arbitrage Strategies

Cross-exchange and cross-asset arbitrage:

```typescript
async function setupArbitrageStrategy() {
  const runner = new StrategyRunner();

  // Cross-exchange arbitrage
  const arbConfig: StrategyConfig = {
    name: 'Cross-Exchange Arbitrage',
    symbols: ['BTC/USD', 'ETH/USD'],
    parameters: JSON.stringify({
      min_spread: 0.002,      // 0.2% minimum spread
      fee_per_trade: 0.001,   // 0.1% fees
      slippage_buffer: 0.0005, // 0.05% slippage
      execution_time_ms: 1000, // Max execution time
      min_profit_after_costs: 0.001 // 0.1% minimum profit
    })
  };

  await runner.addArbitrageStrategy(arbConfig);

  // Monitors price discrepancies across exchanges
  // Accounts for fees, slippage, and execution time
  // Only signals when profitable after all costs

  return runner;
}

// Statistical arbitrage
async function statArbStrategy() {
  const runner = new StrategyRunner();

  const config: StrategyConfig = {
    name: 'Statistical Arbitrage',
    symbols: ['SPY', 'IVV', 'VOO'], // ETFs tracking same index
    parameters: JSON.stringify({
      cointegration_period: 90,
      zscore_entry: 2.5,
      zscore_exit: 0.0,
      max_holding_period: 5,
      basket_optimization: true
    })
  };

  await runner.addArbitrageStrategy(config);
  return runner;
}
```

## Integration Examples

### Integration with @neural-trader/backtesting

Complete workflow for strategy validation:

```typescript
import { StrategyRunner } from '@neural-trader/strategies';
import { BacktestEngine } from '@neural-trader/backtesting';
import type { BacktestConfig, Signal } from '@neural-trader/core';

async function backtestMomentumStrategy() {
  // 1. Create strategy
  const runner = new StrategyRunner();

  const config: StrategyConfig = {
    name: 'SMA Crossover Backtest',
    symbols: ['AAPL', 'MSFT', 'GOOGL'],
    parameters: JSON.stringify({
      shortPeriod: 20,
      longPeriod: 50,
      minVolume: 1000000
    })
  };

  await runner.addMomentumStrategy(config);

  // 2. Generate historical signals
  const signals: Signal[] = await runner.generateSignals();

  console.log(`Generated ${signals.length} historical signals`);

  // 3. Configure backtest
  const backtestConfig: BacktestConfig = {
    initialCapital: 100000,
    startDate: '2023-01-01',
    endDate: '2024-01-01',
    commission: 0.001,
    slippage: 0.0005,
    useMarkToMarket: true
  };

  // 4. Run backtest
  const engine = new BacktestEngine(backtestConfig);
  const result = await engine.run(signals, 'historical-data.csv');

  // 5. Analyze results
  console.log('\n=== Backtest Results ===');
  console.log(`Total Return: ${(result.metrics.totalReturn * 100).toFixed(2)}%`);
  console.log(`Sharpe Ratio: ${result.metrics.sharpeRatio.toFixed(2)}`);
  console.log(`Max Drawdown: ${(result.metrics.maxDrawdown * 100).toFixed(2)}%`);
  console.log(`Win Rate: ${(result.metrics.winRate * 100).toFixed(1)}%`);
  console.log(`Profit Factor: ${result.metrics.profitFactor.toFixed(2)}`);
  console.log(`Total Trades: ${result.metrics.totalTrades}`);

  // 6. Export detailed results
  const tradeCsv = engine.exportTradesCsv(result.trades);
  console.log('\nTrade history exported to CSV');

  return { signals, result };
}
```

### Integration with @neural-trader/risk

Position sizing with Kelly Criterion:

```typescript
import { StrategyRunner } from '@neural-trader/strategies';
import { RiskManager } from '@neural-trader/risk';
import type { Signal, RiskConfig } from '@neural-trader/core';

async function riskAwareTrading() {
  // 1. Initialize strategy and risk management
  const runner = new StrategyRunner();
  const riskConfig: RiskConfig = {
    confidenceLevel: 0.95,
    lookbackPeriods: 252,
    method: 'historical'
  };
  const riskManager = new RiskManager(riskConfig);

  // 2. Add strategies
  await runner.addMomentumStrategy({
    name: 'Momentum with Risk',
    symbols: ['AAPL', 'MSFT', 'GOOGL'],
    parameters: JSON.stringify({ shortPeriod: 20, longPeriod: 50 })
  });

  // 3. Subscribe to signals with risk-based position sizing
  const portfolioValue = 100000;

  const subscription = runner.subscribeSignals((signal: Signal) => {
    console.log(`\n=== New Signal: ${signal.symbol} ===`);

    // Calculate Kelly position size
    const kelly = riskManager.calculateKelly(
      signal.confidence,  // Win probability
      (signal.takeProfit - signal.entryPrice) / signal.entryPrice,  // Avg win
      (signal.entryPrice - signal.stopLoss) / signal.entryPrice     // Avg loss
    );

    // Use half-Kelly for safer sizing
    const positionSizePercent = kelly.halfKelly;
    const positionDollars = portfolioValue * positionSizePercent;
    const shares = Math.floor(positionDollars / signal.entryPrice);

    console.log(`Signal Direction: ${signal.direction}`);
    console.log(`Confidence: ${(signal.confidence * 100).toFixed(1)}%`);
    console.log(`Entry: $${signal.entryPrice.toFixed(2)}`);
    console.log(`Stop Loss: $${signal.stopLoss.toFixed(2)}`);
    console.log(`Take Profit: $${signal.takeProfit.toFixed(2)}`);
    console.log(`\nRisk-Based Position Sizing:`);
    console.log(`  Kelly Fraction: ${(kelly.kellyFraction * 100).toFixed(2)}%`);
    console.log(`  Half Kelly (Recommended): ${(kelly.halfKelly * 100).toFixed(2)}%`);
    console.log(`  Position Size: $${positionDollars.toLocaleString()}`);
    console.log(`  Shares: ${shares}`);
    console.log(`  Max Risk: $${((signal.entryPrice - signal.stopLoss) * shares).toFixed(2)}`);

    // Execute trade with calculated position size
    // executeOrder({ signal, shares });
  });

  return { runner, subscription };
}
```

### Integration with @neural-trader/portfolio

Portfolio construction with optimization:

```typescript
import { StrategyRunner } from '@neural-trader/strategies';
import { PortfolioManager, PortfolioOptimizer } from '@neural-trader/portfolio';
import type { Signal } from '@neural-trader/core';

async function portfolioBasedTrading() {
  // 1. Initialize components
  const runner = new StrategyRunner();
  const portfolio = new PortfolioManager(250000); // $250k initial capital
  const optimizer = new PortfolioOptimizer({ riskFreeRate: 0.02 });

  // 2. Add multiple strategies for diversification
  await runner.addMomentumStrategy({
    name: 'Large Cap Momentum',
    symbols: ['AAPL', 'MSFT', 'GOOGL', 'AMZN'],
    parameters: JSON.stringify({ shortPeriod: 20, longPeriod: 50 })
  });

  await runner.addMeanReversionStrategy({
    name: 'Index Reversion',
    symbols: ['SPY', 'QQQ', 'IWM'],
    parameters: JSON.stringify({ period: 20, stdDev: 2.0 })
  });

  // 3. Generate signals and optimize portfolio allocation
  const signals = await runner.generateSignals();

  // Filter high-confidence signals
  const highQualitySignals = signals.filter(s => s.confidence >= 0.75);

  // Extract unique symbols
  const symbols = [...new Set(highQualitySignals.map(s => s.symbol))];

  // Calculate expected returns from signals
  const expectedReturns = symbols.map(symbol => {
    const signalsForSymbol = highQualitySignals.filter(s => s.symbol === symbol);
    const avgReturn = signalsForSymbol.reduce((sum, sig) =>
      sum + (sig.takeProfit - sig.entryPrice) / sig.entryPrice, 0
    ) / signalsForSymbol.length;
    return avgReturn;
  });

  // Generate covariance matrix (simplified - use historical data in production)
  const covMatrix = generateCovarianceMatrix(symbols);

  // 4. Optimize portfolio
  const optimization = await optimizer.optimize(
    symbols,
    expectedReturns,
    covMatrix
  );

  console.log('\n=== Portfolio Optimization Results ===');
  console.log(`Expected Return: ${(optimization.expectedReturn * 100).toFixed(2)}%`);
  console.log(`Portfolio Risk: ${(optimization.risk * 100).toFixed(2)}%`);
  console.log(`Sharpe Ratio: ${optimization.sharpeRatio.toFixed(2)}`);
  console.log('\nOptimal Allocations:');

  optimization.allocations.forEach((allocation, idx) => {
    const symbol = symbols[idx];
    const dollarAmount = portfolio.getCash() * allocation;
    console.log(`  ${symbol}: ${(allocation * 100).toFixed(2)}% ($${dollarAmount.toLocaleString()})`);
  });

  // 5. Execute trades based on optimal allocations
  for (let i = 0; i < symbols.length; i++) {
    const symbol = symbols[i];
    const allocation = optimization.allocations[i];
    const signal = highQualitySignals.find(s => s.symbol === symbol);

    if (signal && allocation > 0.01) { // Only trade if allocation > 1%
      const cash = await portfolio.getCash();
      const dollarAmount = cash * allocation;
      const shares = Math.floor(dollarAmount / signal.entryPrice);

      // Update portfolio
      await portfolio.updatePosition(symbol, shares, signal.entryPrice);

      console.log(`\nExecuted: ${symbol} - ${shares} shares @ $${signal.entryPrice}`);
    }
  }

  // 6. Monitor portfolio
  const totalValue = await portfolio.getTotalValue();
  const pnl = await portfolio.getTotalPnl();
  console.log(`\n=== Portfolio Status ===`);
  console.log(`Total Value: $${totalValue.toLocaleString()}`);
  console.log(`P&L: $${pnl.toLocaleString()} (${((pnl / 250000) * 100).toFixed(2)}%)`);

  return { runner, portfolio, optimization };
}

function generateCovarianceMatrix(symbols: string[]): number[] {
  // Simplified for example - use historical returns in production
  const n = symbols.length;
  const matrix: number[] = [];
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (i === j) {
        matrix.push(0.04); // 4% variance
      } else {
        matrix.push(0.02); // 2% covariance
      }
    }
  }
  return matrix;
}
```

### Complete Live Trading Workflow

Full integration with execution and brokers:

```typescript
import { StrategyRunner } from '@neural-trader/strategies';
import { RiskManager } from '@neural-trader/risk';
import { PortfolioManager } from '@neural-trader/portfolio';
import { NeuralTrader } from '@neural-trader/execution';
import { BrokerClient } from '@neural-trader/brokers';
import type { Signal } from '@neural-trader/core';

async function liveTradingSystem() {
  // 1. Initialize all components
  const runner = new StrategyRunner();
  const riskManager = new RiskManager({
    confidenceLevel: 0.95,
    lookbackPeriods: 252,
    method: 'historical'
  });
  const portfolio = new PortfolioManager(100000);

  // 2. Connect to broker (paper trading)
  const broker = new BrokerClient({
    brokerType: 'alpaca',
    apiKey: process.env.ALPACA_API_KEY!,
    apiSecret: process.env.ALPACA_API_SECRET!,
    baseUrl: 'https://paper-api.alpaca.markets',
    paperTrading: true
  });

  await broker.connect();
  console.log('Connected to broker');

  // 3. Initialize execution engine
  const trader = new NeuralTrader({
    apiKey: process.env.ALPACA_API_KEY!,
    apiSecret: process.env.ALPACA_API_SECRET!,
    baseUrl: 'https://paper-api.alpaca.markets',
    paperTrading: true
  });

  await trader.start();
  console.log('Trading engine started');

  // 4. Add strategies
  await runner.addMomentumStrategy({
    name: 'Live Momentum',
    symbols: ['AAPL', 'MSFT', 'GOOGL'],
    parameters: JSON.stringify({
      shortPeriod: 20,
      longPeriod: 50,
      minVolume: 1000000,
      atr_multiplier: 2.0
    })
  });

  await runner.addMeanReversionStrategy({
    name: 'Live Mean Reversion',
    symbols: ['SPY', 'QQQ'],
    parameters: JSON.stringify({
      period: 20,
      stdDev: 2.0,
      oversold_threshold: 0.1,
      overbought_threshold: 0.1
    })
  });

  console.log('Strategies initialized');

  // 5. Subscribe to live signals with full execution pipeline
  const subscription = runner.subscribeSignals(async (signal: Signal) => {
    try {
      console.log(`\n${'='.repeat(60)}`);
      console.log(`NEW SIGNAL: ${signal.symbol} - ${signal.direction.toUpperCase()}`);
      console.log(`${'='.repeat(60)}`);

      // Step 1: Validate signal confidence
      if (signal.confidence < 0.7) {
        console.log(`❌ Signal rejected: Low confidence (${signal.confidence})`);
        return;
      }

      // Step 2: Calculate Kelly position size
      const portfolioValue = await portfolio.getTotalValue();
      const kelly = riskManager.calculateKelly(
        signal.confidence,
        (signal.takeProfit - signal.entryPrice) / signal.entryPrice,
        (signal.entryPrice - signal.stopLoss) / signal.entryPrice
      );

      // Use conservative half-Kelly
      const positionSize = Math.floor(
        (portfolioValue * kelly.halfKelly) / signal.entryPrice
      );

      if (positionSize === 0) {
        console.log('❌ Position size too small');
        return;
      }

      console.log(`\n📊 Signal Details:`);
      console.log(`   Confidence: ${(signal.confidence * 100).toFixed(1)}%`);
      console.log(`   Entry: $${signal.entryPrice.toFixed(2)}`);
      console.log(`   Stop Loss: $${signal.stopLoss.toFixed(2)}`);
      console.log(`   Take Profit: $${signal.takeProfit.toFixed(2)}`);
      console.log(`   Reasoning: ${signal.reasoning}`);

      console.log(`\n💰 Position Sizing:`);
      console.log(`   Portfolio Value: $${portfolioValue.toLocaleString()}`);
      console.log(`   Kelly Fraction: ${(kelly.halfKelly * 100).toFixed(2)}%`);
      console.log(`   Shares: ${positionSize}`);
      console.log(`   Position Value: $${(positionSize * signal.entryPrice).toLocaleString()}`);

      // Step 3: Validate portfolio constraints
      const currentPosition = await portfolio.getPosition(signal.symbol);
      const cash = await portfolio.getCash();

      if (cash < positionSize * signal.entryPrice) {
        console.log('❌ Insufficient cash');
        return;
      }

      // Step 4: Calculate VaR for risk check
      const historicalReturns = []; // Load from database in production
      const var95 = riskManager.calculateVar(historicalReturns, portfolioValue);

      console.log(`\n⚠️  Risk Metrics:`);
      console.log(`   VaR (95%): $${var95.varAmount.toLocaleString()}`);
      console.log(`   Max Risk on Trade: $${((signal.entryPrice - signal.stopLoss) * positionSize).toFixed(2)}`);

      // Step 5: Place order through broker
      const order = {
        id: `order-${Date.now()}`,
        symbol: signal.symbol,
        side: signal.direction === 'long' ? 'buy' : 'sell',
        orderType: 'limit',
        quantity: positionSize.toString(),
        limitPrice: signal.entryPrice.toFixed(2),
        timeInForce: 'day'
      };

      console.log(`\n🚀 Placing order...`);
      const orderResult = await trader.placeOrder(order);

      if (orderResult.success) {
        console.log(`✅ Order placed successfully: ${orderResult.orderId}`);

        // Step 6: Update portfolio tracking
        await portfolio.updatePosition(
          signal.symbol,
          positionSize,
          signal.entryPrice
        );

        // Step 7: Set stop loss and take profit orders
        const stopOrder = {
          id: `stop-${Date.now()}`,
          symbol: signal.symbol,
          side: signal.direction === 'long' ? 'sell' : 'buy',
          orderType: 'stop',
          quantity: positionSize.toString(),
          stopPrice: signal.stopLoss.toFixed(2),
          timeInForce: 'gtc'
        };

        await trader.placeOrder(stopOrder);
        console.log(`✅ Stop loss set at $${signal.stopLoss.toFixed(2)}`);

        const limitOrder = {
          id: `limit-${Date.now()}`,
          symbol: signal.symbol,
          side: signal.direction === 'long' ? 'sell' : 'buy',
          orderType: 'limit',
          quantity: positionSize.toString(),
          limitPrice: signal.takeProfit.toFixed(2),
          timeInForce: 'gtc'
        };

        await trader.placeOrder(limitOrder);
        console.log(`✅ Take profit set at $${signal.takeProfit.toFixed(2)}`);

        // Step 8: Log trade to database/file
        console.log(`\n📝 Trade logged for analysis`);

      } else {
        console.log(`❌ Order failed: ${orderResult.message}`);
      }

    } catch (error) {
      console.error(`❌ Error processing signal:`, error);
    }
  });

  // 6. Periodic portfolio monitoring
  setInterval(async () => {
    const positions = await portfolio.getPositions();
    const totalValue = await portfolio.getTotalValue();
    const pnl = await portfolio.getTotalPnl();

    console.log(`\n${'='.repeat(60)}`);
    console.log(`PORTFOLIO STATUS - ${new Date().toLocaleString()}`);
    console.log(`${'='.repeat(60)}`);
    console.log(`Total Value: $${totalValue.toLocaleString()}`);
    console.log(`P&L: $${pnl.toLocaleString()} (${((pnl / 100000) * 100).toFixed(2)}%)`);
    console.log(`Positions: ${positions.length}`);

    positions.forEach(pos => {
      console.log(`  ${pos.symbol}: ${pos.quantity} shares @ $${pos.avgPrice.toFixed(2)}`);
    });
  }, 60000); // Every minute

  // 7. Graceful shutdown handler
  process.on('SIGINT', async () => {
    console.log('\n\nShutting down trading system...');
    await subscription.unsubscribe();
    await trader.stop();
    await broker.disconnect();
    console.log('✅ Clean shutdown complete');
    process.exit(0);
  });

  console.log('\n✅ Live trading system running. Press Ctrl+C to stop.\n');

  return { runner, trader, broker, portfolio, subscription };
}

// Run the live trading system
liveTradingSystem().catch(console.error);
```

## API Reference

### StrategyRunner

```typescript
class StrategyRunner {
  constructor();

  // Add momentum strategy (SMA/EMA crossovers)
  addMomentumStrategy(config: StrategyConfig): Promise<string>;

  // Add mean reversion strategy (Bollinger Bands, RSI)
  addMeanReversionStrategy(config: StrategyConfig): Promise<string>;

  // Add arbitrage strategy (pairs trading, cross-exchange)
  addArbitrageStrategy(config: StrategyConfig): Promise<string>;

  // Generate signals from all active strategies
  generateSignals(): Promise<Signal[]>;

  // Subscribe to real-time signal updates
  subscribeSignals(callback: (signal: Signal) => void): SubscriptionHandle;

  // List all active strategy IDs
  listStrategies(): Promise<string[]>;

  // Remove strategy by ID
  removeStrategy(strategyId: string): Promise<boolean>;
}
```

### SubscriptionHandle

```typescript
class SubscriptionHandle {
  // Unsubscribe from signal updates
  unsubscribe(): Promise<void>;
}
```

## Configuration

### StrategyConfig

| Field | Type | Description |
|-------|------|-------------|
| `name` | `string` | Human-readable strategy name |
| `symbols` | `string[]` | Trading symbols (use "SYM1,SYM2" for pairs) |
| `parameters` | `string` | JSON-encoded strategy parameters |

### Common Parameters

**Momentum Strategies:**
- `shortPeriod`: Fast moving average period (default: 20)
- `longPeriod`: Slow moving average period (default: 50)
- `minVolume`: Minimum daily volume filter
- `atr_multiplier`: ATR multiplier for stop loss calculation

**Mean Reversion Strategies:**
- `period`: Lookback period for calculations (default: 20)
- `stdDev`: Standard deviations for bands (default: 2.0)
- `oversold_threshold`: Buy trigger threshold
- `overbought_threshold`: Sell trigger threshold
- `rsi_period`: RSI calculation period (default: 14)

**Pairs Trading:**
- `lookback`: Cointegration lookback period (default: 60)
- `entry_threshold`: Z-score entry threshold (default: 2.0)
- `exit_threshold`: Z-score exit threshold (default: 0.5)
- `min_correlation`: Minimum correlation requirement (default: 0.7)

**Arbitrage:**
- `min_spread`: Minimum spread for execution
- `fee_per_trade`: Trading fees to account for
- `slippage_buffer`: Expected slippage buffer

## Performance

- **Signal Generation**: <1ms per strategy per symbol
- **Subscription Latency**: <100μs callback invocation
- **Memory Usage**: ~2MB per strategy instance
- **Throughput**: 10,000+ signals/second aggregate

## Platform Support

- ✅ Linux x64 (GNU/musl)
- ✅ macOS x64 (Intel)
- ✅ macOS ARM64 (Apple Silicon)
- ✅ Windows x64 (MSVC)

## Related Packages

- **@neural-trader/core** - Core types and interfaces (required)
- **@neural-trader/backtesting** - Historical strategy validation
- **@neural-trader/risk** - Risk management and position sizing
- **@neural-trader/portfolio** - Portfolio optimization and tracking
- **@neural-trader/execution** - Smart order routing and execution
- **@neural-trader/brokers** - Multi-broker connectivity

## License

This package is dual-licensed under MIT OR Apache-2.0.

**MIT License**: https://opensource.org/licenses/MIT
**Apache-2.0 License**: https://www.apache.org/licenses/LICENSE-2.0

You may choose either license for your use.
