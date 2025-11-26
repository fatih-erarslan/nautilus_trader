# NAPI Bindings Complete Reference

## Overview

This document describes the complete NAPI-RS bindings for the Neural Trader Rust core, providing comprehensive Node.js/TypeScript access to all Rust functionality.

**Package**: `@neural-trader/rust-core@0.1.0`
**Location**: `/workspaces/neural-trader/neural-trader-rust/crates/napi-bindings/`

---

## Table of Contents

1. [Core Trading System](#core-trading-system)
2. [Broker Integrations](#broker-integrations)
3. [Neural Networks](#neural-networks)
4. [Risk Management](#risk-management)
5. [Backtesting](#backtesting)
6. [Market Data](#market-data)
7. [Strategy System](#strategy-system)
8. [Portfolio Management](#portfolio-management)
9. [Usage Examples](#usage-examples)
10. [Build and Deployment](#build-and-deployment)

---

## Core Trading System

### `NeuralTrader` Class

Main entry point for the trading system.

```typescript
import { NeuralTrader } from '@neural-trader/rust-core';

const trader = new NeuralTrader({
  apiKey: process.env.ALPACA_API_KEY,
  apiSecret: process.env.ALPACA_API_SECRET,
  paperTrading: true
});

await trader.start();
const positions = await trader.getPositions();
await trader.stop();
```

### Version Information

```typescript
import { getVersionInfo } from '@neural-trader/rust-core';

const version = getVersionInfo();
console.log(version.rustCore); // "0.1.0"
console.log(version.napiBindings); // "0.1.0"
console.log(version.rustCompiler); // "1.76.0"
```

---

## Broker Integrations

### Supported Brokers

- **Alpaca** - US stocks and options
- **Interactive Brokers (IBKR)** - Global multi-asset
- **CCXT** - 100+ cryptocurrency exchanges
- **Oanda** - Forex and CFDs
- **Questrade** - Canadian equities
- **Lime Trading** - Professional trading

### `BrokerClient` Class

```typescript
import { BrokerClient, listBrokerTypes } from '@neural-trader/rust-core';

// List available brokers
const brokers = listBrokerTypes();
// ['alpaca', 'ibkr', 'ccxt', 'oanda', 'questrade', 'lime']

// Create broker client
const broker = new BrokerClient({
  brokerType: 'alpaca',
  apiKey: process.env.ALPACA_API_KEY,
  apiSecret: process.env.ALPACA_API_SECRET,
  paperTrading: true
});

await broker.connect();

// Place order
const order = await broker.placeOrder({
  symbol: 'AAPL',
  side: 'buy',
  orderType: 'limit',
  quantity: 100,
  limitPrice: 150.00,
  timeInForce: 'day'
});

console.log(`Order placed: ${order.orderId}`);

// Check order status
const status = await broker.getOrderStatus(order.orderId);
console.log(`Status: ${status.status}`);

// Get account balance
const balance = await broker.getAccountBalance();
console.log(`Cash: $${balance.cash}`);
console.log(`Equity: $${balance.equity}`);

await broker.disconnect();
```

### CCXT Crypto Trading

```typescript
const cryptoBroker = new BrokerClient({
  brokerType: 'ccxt',
  exchange: 'binance',
  apiKey: process.env.BINANCE_API_KEY,
  apiSecret: process.env.BINANCE_API_SECRET,
  paperTrading: false
});

await cryptoBroker.connect();

const order = await cryptoBroker.placeOrder({
  symbol: 'BTC/USDT',
  side: 'buy',
  orderType: 'market',
  quantity: 0.01,
  timeInForce: 'ioc'
});
```

---

## Neural Networks

### Model Types

- **NHITS** - Neural Hierarchical Interpolation for Time Series
- **LSTM-Attention** - LSTM with multi-head attention
- **Transformer** - Transformer architecture for time series

### `NeuralModel` Class

```typescript
import { NeuralModel, listModelTypes } from '@neural-trader/rust-core';

// List available models
const models = listModelTypes();
// ['nhits', 'lstm_attention', 'transformer']

// Create model
const model = new NeuralModel({
  modelType: 'nhits',
  inputSize: 168,  // 1 week of hourly data
  horizon: 24,     // 24 hour forecast
  hiddenSize: 512,
  numLayers: 3,
  dropout: 0.1,
  learningRate: 0.001
});

// Training data (simplified)
const prices = [/* 168 historical prices */];
const targets = [/* 24 future prices for training */];

// Train model
const metrics = await model.train(prices, targets, {
  epochs: 100,
  batchSize: 32,
  validationSplit: 0.2,
  earlyStoppingPatience: 10,
  useGpu: true
});

console.log(`Final validation loss: ${metrics[metrics.length - 1].valLoss}`);

// Make predictions
const input = [/* 168 recent prices */];
const prediction = await model.predict(input);

console.log('Predicted prices:', prediction.predictions);
console.log('95% confidence interval:', [
  prediction.lowerBound,
  prediction.upperBound
]);

// Save model
await model.save('./models/nhits-btc.safetensors');

// Load model
await model.load('./models/nhits-btc.safetensors');
```

### Batch Prediction

```typescript
import { BatchPredictor } from '@neural-trader/rust-core';

const batchPredictor = new BatchPredictor();

// Add multiple models
await batchPredictor.addModel(model1);
await batchPredictor.addModel(model2);
await batchPredictor.addModel(model3);

// Predict in parallel
const inputs = [input1, input2, input3];
const results = await batchPredictor.predictBatch(inputs);
```

---

## Risk Management

### `RiskManager` Class

```typescript
import { RiskManager, calculateSharpeRatio } from '@neural-trader/rust-core';

const riskMgr = new RiskManager({
  confidenceLevel: 0.95,
  lookbackPeriods: 252,
  method: 'historical'
});

// Calculate Value at Risk
const varResult = riskMgr.calculateVar(
  returns,  // Array of historical returns
  100000    // Portfolio value
);

console.log(`VaR (95%): $${varResult.varAmount}`);
console.log(`Maximum 1-day loss: ${varResult.varPercentage * 100}%`);

// Calculate Conditional VaR (Expected Shortfall)
const cvarResult = riskMgr.calculateCvar(returns, 100000);
console.log(`CVaR (95%): $${cvarResult.cvarAmount}`);

// Kelly Criterion for position sizing
const kelly = riskMgr.calculateKelly(
  0.55,  // 55% win rate
  1000,  // Average win
  500    // Average loss
);

console.log(`Kelly fraction: ${kelly.kellyFraction}`);
console.log(`Recommended: ${kelly.halfKelly} (half Kelly)`);

// Drawdown analysis
const drawdown = riskMgr.calculateDrawdown(equityCurve);
console.log(`Max drawdown: ${drawdown.maxDrawdown * 100}%`);
console.log(`Recovery factor: ${drawdown.recoveryFactor}`);

// Position sizing
const posSize = riskMgr.calculatePositionSize(
  100000,  // Portfolio value
  150,     // Price per share
  0.02,    // Risk 2% per trade
  5        // Stop loss $5 away
);

console.log(`Buy ${posSize.shares} shares`);
console.log(`Risk: $${posSize.maxLoss}`);

// Sharpe ratio
const sharpe = calculateSharpeRatio(
  returns,
  0.02,  // 2% risk-free rate
  252    // Annualization factor (daily data)
);

console.log(`Sharpe ratio: ${sharpe.toFixed(2)}`);
```

---

## Backtesting

### `BacktestEngine` Class

```typescript
import { BacktestEngine, compareBacktests } from '@neural-trader/rust-core';

const backtest = new BacktestEngine({
  initialCapital: 100000,
  startDate: '2023-01-01',
  endDate: '2023-12-31',
  commission: 0.001,  // $0.001 per share
  slippage: 0.0005,   // 0.05%
  useMarkToMarket: true
});

// Run backtest with strategy signals
const result = await backtest.run(signals, marketData);

console.log('Performance Metrics:');
console.log(`Total return: ${result.metrics.totalReturn * 100}%`);
console.log(`Sharpe ratio: ${result.metrics.sharpeRatio}`);
console.log(`Max drawdown: ${result.metrics.maxDrawdown * 100}%`);
console.log(`Win rate: ${result.metrics.winRate * 100}%`);
console.log(`Total trades: ${result.metrics.totalTrades}`);

// Export trades to CSV
const csv = backtest.exportTradesCsv(result.trades);
fs.writeFileSync('trades.csv', csv);

// Compare multiple strategies
const comparison = compareBacktests([result1, result2, result3]);
console.log(comparison);
```

---

## Market Data

### `MarketDataProvider` Class

```typescript
import { MarketDataProvider, calculateSma, calculateRsi, listDataProviders } from '@neural-trader/rust-core';

// List providers
const providers = listDataProviders();
// ['alpaca', 'polygon', 'yahoo', 'binance', 'coinbase']

const dataProvider = new MarketDataProvider({
  provider: 'alpaca',
  apiKey: process.env.ALPACA_API_KEY,
  apiSecret: process.env.ALPACA_API_SECRET,
  websocketEnabled: true
});

await dataProvider.connect();

// Fetch historical bars
const bars = await dataProvider.fetchBars(
  'AAPL',
  '2023-01-01T00:00:00Z',
  '2023-12-31T23:59:59Z',
  '1Day'
);

// Get latest quote
const quote = await dataProvider.getQuote('AAPL');
console.log(`Bid: ${quote.bid}, Ask: ${quote.ask}`);

// Subscribe to real-time quotes
const subscription = dataProvider.subscribeQuotes(
  ['AAPL', 'GOOGL', 'MSFT'],
  (quote) => {
    console.log(`${quote.symbol}: ${quote.last} @ ${quote.timestamp}`);
  }
);

// Unsubscribe later
await subscription.unsubscribe();

// Technical indicators
const prices = bars.map(b => parseFloat(b.close));
const sma20 = calculateSma(prices, 20);
const rsi14 = calculateRsi(prices, 14);

console.log(`SMA(20): ${sma20[sma20.length - 1]}`);
console.log(`RSI(14): ${rsi14[rsi14.length - 1]}`);
```

---

## Strategy System

### `StrategyRunner` Class

```typescript
import { StrategyRunner } from '@neural-trader/rust-core';

const runner = new StrategyRunner();

// Add strategies
const momentumId = await runner.addMomentumStrategy({
  name: 'Momentum 20/50',
  symbols: ['AAPL', 'GOOGL', 'MSFT'],
  parameters: {
    fastPeriod: 20,
    slowPeriod: 50,
    threshold: 0.02
  }
});

const meanRevId = await runner.addMeanReversionStrategy({
  name: 'Mean Reversion RSI',
  symbols: ['SPY', 'QQQ'],
  parameters: {
    rsiPeriod: 14,
    oversold: 30,
    overbought: 70
  }
});

// Generate signals
const signals = await runner.generateSignals();

for (const signal of signals) {
  console.log(`${signal.symbol}: ${signal.direction} @ ${signal.confidence}`);
  console.log(`Reasoning: ${signal.reasoning}`);
}

// Subscribe to real-time signals
const sub = runner.subscribeSignals((signal) => {
  console.log(`New signal: ${signal.symbol} ${signal.direction}`);
});

// List active strategies
const strategies = await runner.listStrategies();
console.log('Active strategies:', strategies);

// Remove strategy
await runner.removeStrategy(momentumId);
```

---

## Portfolio Management

### `PortfolioManager` Class

```typescript
import { PortfolioManager } from '@neural-trader/rust-core';

const portfolio = new PortfolioManager(100000); // $100k initial cash

// Update position after trade
const position = await portfolio.updatePosition(
  'AAPL',
  100,   // Quantity
  150.50 // Price
);

console.log(`Position: ${position.quantity} shares @ $${position.avgCost}`);
console.log(`Unrealized P&L: $${position.unrealizedPnl}`);

// Get all positions
const positions = await portfolio.getPositions();

// Get total portfolio value
const totalValue = await portfolio.getTotalValue();
const cash = await portfolio.getCash();

console.log(`Total value: $${totalValue}`);
console.log(`Cash: $${cash}`);
```

### `PortfolioOptimizer` Class

```typescript
import { PortfolioOptimizer } from '@neural-trader/rust-core';

const optimizer = new PortfolioOptimizer({
  riskFreeRate: 0.02,
  maxPositionSize: 0.25,  // 25% max per position
  minPositionSize: 0.05   // 5% minimum
});

// Optimize allocation
const optimization = await optimizer.optimize(
  ['AAPL', 'GOOGL', 'MSFT', 'TSLA'],
  expectedReturns,
  covarianceMatrix
);

console.log('Optimal allocations:', optimization.allocations);
console.log(`Expected return: ${optimization.expectedReturn * 100}%`);
console.log(`Risk: ${optimization.risk * 100}%`);
console.log(`Sharpe ratio: ${optimization.sharpeRatio}`);

// Calculate risk metrics
const riskMetrics = optimizer.calculateRisk(positions);
console.log(`VaR (95%): ${riskMetrics.var95}`);
console.log(`Beta: ${riskMetrics.beta}`);
```

---

## Usage Examples

### Complete Trading System

```typescript
import {
  NeuralTrader,
  BrokerClient,
  StrategyRunner,
  RiskManager,
  PortfolioManager,
  MarketDataProvider
} from '@neural-trader/rust-core';

async function main() {
  // Initialize components
  const broker = new BrokerClient({
    brokerType: 'alpaca',
    apiKey: process.env.ALPACA_API_KEY,
    apiSecret: process.env.ALPACA_API_SECRET,
    paperTrading: true
  });

  const dataProvider = new MarketDataProvider({
    provider: 'alpaca',
    apiKey: process.env.ALPACA_API_KEY,
    apiSecret: process.env.ALPACA_API_SECRET,
    websocketEnabled: true
  });

  const strategyRunner = new StrategyRunner();
  const riskMgr = new RiskManager({
    confidenceLevel: 0.95,
    lookbackPeriods: 252,
    method: 'historical'
  });

  const portfolio = new PortfolioManager(100000);

  // Connect
  await broker.connect();
  await dataProvider.connect();

  // Add strategy
  await strategyRunner.addMomentumStrategy({
    name: 'Main Momentum',
    symbols: ['AAPL', 'GOOGL'],
    parameters: { fastPeriod: 20, slowPeriod: 50 }
  });

  // Subscribe to signals
  strategyRunner.subscribeSignals(async (signal) => {
    console.log(`Signal received: ${signal.symbol} ${signal.direction}`);

    // Calculate position size
    const balance = await broker.getAccountBalance();
    const quote = await dataProvider.getQuote(signal.symbol);

    const posSize = riskMgr.calculatePositionSize(
      balance.equity,
      parseFloat(quote.last),
      0.02,  // Risk 2%
      parseFloat(quote.last) * 0.05  // 5% stop loss
    );

    // Place order
    const order = await broker.placeOrder({
      symbol: signal.symbol,
      side: signal.direction === 'long' ? 'buy' : 'sell',
      orderType: 'market',
      quantity: posSize.shares,
      timeInForce: 'day'
    });

    console.log(`Order placed: ${order.orderId}`);

    // Update portfolio
    await portfolio.updatePosition(
      signal.symbol,
      posSize.shares,
      parseFloat(quote.last)
    );
  });

  // Keep running
  process.on('SIGINT', async () => {
    await broker.disconnect();
    await dataProvider.disconnect();
    process.exit(0);
  });
}

main().catch(console.error);
```

---

## Build and Deployment

### Building from Source

```bash
cd /workspaces/neural-trader/neural-trader-rust/crates/napi-bindings

# Install dependencies
npm install

# Build for current platform
npm run build

# Build for all platforms
npm run build -- --target aarch64-apple-darwin
npm run build -- --target x86_64-apple-darwin
npm run build -- --target x86_64-unknown-linux-gnu
npm run build -- --target x86_64-pc-windows-msvc

# Package for NPM
npm run prepublishOnly
```

### Publishing to NPM

```bash
# Set version
npm version patch/minor/major

# Publish
npm publish --access public
```

### Installing in Projects

```bash
npm install @neural-trader/rust-core
```

---

## Performance Characteristics

- **Zero-copy buffers**: Large datasets transferred without serialization overhead
- **Async operations**: All I/O operations return Promises
- **Thread-safe**: Concurrent operations using Arc and Mutex
- **SIMD acceleration**: Auto-vectorized calculations on supported platforms
- **GPU support**: Neural models can use CUDA/Metal when available

---

## Error Handling

All functions that can fail return `Promise<T>` and throw proper JavaScript errors:

```typescript
try {
  const result = await broker.placeOrder(order);
} catch (error) {
  console.error('Order failed:', error.message);
  // Error messages include context from Rust
}
```

---

## Memory Management

- **Automatic cleanup**: All resources use RAII pattern
- **Subscription handles**: Must call `unsubscribe()` to cleanup
- **Connection lifecycle**: Always call `disconnect()` on shutdown

---

## Testing

```bash
# Run Node.js integration tests
npm run test:node

# Run Rust unit tests
cargo test --package nt-napi-bindings
```

---

## Support and Documentation

- **Repository**: https://github.com/ruvnet/neural-trader
- **Issues**: https://github.com/ruvnet/neural-trader/issues
- **API Docs**: Auto-generated TypeScript definitions in `index.d.ts`

---

## License

MIT OR Apache-2.0
