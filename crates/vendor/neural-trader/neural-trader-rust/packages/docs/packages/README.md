# Package API Documentation

API reference documentation for all @neural-trader packages.

## 📦 Core Packages

### @neural-trader/core
**Type definitions and interfaces**

```typescript
import { Strategy, BacktestResult, Position } from '@neural-trader/core';
```

**Key Types:**
- `Strategy` - Trading strategy interface
- `BacktestResult` - Backtest output format
- `Position` - Open position representation
- `Order` - Order structure
- `Portfolio` - Portfolio state

**Dependencies:** 0
**Status:** ✅ Stable

---

### @neural-trader/execution
**Order execution engine**

```typescript
import { NeuralTrader } from '@neural-trader/execution';

const trader = new NeuralTrader();
```

**Features:**
- Order placement and management
- Position tracking
- Broker abstraction layer

**Dependencies:** @neural-trader/core
**Status:** ⚠️ Fix needed (hardcoded path issue #69)

---

### @neural-trader/features
**Technical indicators**

```typescript
import { rsi, macd, sma, ema } from '@neural-trader/features';

const rsiValues = rsi(prices, 14);
```

**Indicators:**
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- SMA (Simple Moving Average)
- EMA (Exponential Moving Average)
- Bollinger Bands
- And more...

**Dependencies:** @neural-trader/core
**Status:** ⚠️ Fix needed (RSI bug issue #70)

## 🎯 Trading Packages

### @neural-trader/strategies
**Trading strategy runners**

```typescript
import { StrategyRunner, SubscriptionHandle } from '@neural-trader/strategies';

const runner = new StrategyRunner('momentum');
const handle = await runner.run({ symbol: 'SPY' });
```

**Strategies:**
- Momentum
- Mean Reversion
- Pairs Trading
- Arbitrage

**Dependencies:** @neural-trader/core
**Status:** ✅ Stable

---

### @neural-trader/backtesting
**Backtesting engine**

```typescript
import { BacktestEngine, compareBacktests } from '@neural-trader/backtesting';

const engine = new BacktestEngine(strategy);
const results = await engine.run({
  symbol: 'AAPL',
  start: '2020-01-01',
  end: '2024-01-01'
});
```

**Features:**
- Historical backtesting
- Performance metrics (Sharpe, Sortino, max drawdown)
- Strategy comparison

**Dependencies:** @neural-trader/core
**Status:** ✅ Stable

---

### @neural-trader/benchoptimizer
**Performance optimization CLI**

```bash
# CLI commands
npx benchoptimizer validate
npx benchoptimizer benchmark
npx benchoptimizer optimize
```

**Features:**
- Package validation
- Performance benchmarking
- Optimization suggestions
- Comprehensive reports

**Dependencies:** 12 CLI tools
**Status:** ✅ Stable

## 🧠 Neural Networks

### @neural-trader/neural
**Neural network models**

```typescript
import { NeuralModel, BatchPredictor, listModelTypes } from '@neural-trader/neural';

const model = new NeuralModel('lstm');
await model.train({ symbol: 'AAPL', epochs: 100 });

const forecast = await model.predict({
  horizon: 24,
  confidence: 0.95
});
```

**Models:**
- LSTM (with attention)
- Transformer
- N-HiTS

**Features:**
- Training and inference
- Batch prediction
- GPU acceleration (via Rust NAPI)

**Dependencies:** @neural-trader/core
**Status:** ✅ Stable

## 💼 Portfolio & Risk

### @neural-trader/portfolio
**Portfolio management and optimization**

```typescript
import { PortfolioManager, PortfolioOptimizer } from '@neural-trader/portfolio';

const manager = new PortfolioManager({ initialCash: 100000 });
const optimizer = new PortfolioOptimizer();

const allocation = await optimizer.optimize({
  assets: ['SPY', 'TLT', 'GLD'],
  constraints: {
    maxDrawdown: 0.15,
    minSharpe: 1.5
  }
});
```

**Features:**
- Mean-variance optimization
- Risk parity
- Portfolio rebalancing

**Dependencies:** @neural-trader/core
**Status:** ✅ Stable

---

### @neural-trader/risk
**Risk management**

```typescript
import { RiskManager, calculateSharpe, calculateSortino, calculateMaxLeverage } from '@neural-trader/risk';

const manager = new RiskManager({
  maxDrawdown: 0.2,
  maxPositionSize: 0.1
});

const sharpe = calculateSharpe(returns);
```

**Features:**
- VaR (Value at Risk)
- CVaR (Conditional VaR)
- Kelly Criterion
- Sharpe/Sortino ratios
- Maximum leverage calculation

**Dependencies:** @neural-trader/core
**Status:** ✅ Stable

## 📊 Market Data

### @neural-trader/market-data
**Market data feeds**

```typescript
import { MarketData } from '@neural-trader/market-data';

// TypeScript definitions only
```

**Dependencies:** 0
**Status:** ✅ Stable

---

### @neural-trader/brokers
**Broker integrations**

```typescript
import { Broker } from '@neural-trader/brokers';

// TypeScript definitions only
```

**Supported:**
- Alpaca
- Interactive Brokers
- Binance
- Coinbase

**Dependencies:** 0
**Status:** ✅ Stable

---

### @neural-trader/news-trading
**News analysis (placeholder)**

```typescript
import { NewsTrading } from '@neural-trader/news-trading';

// Module loads but has no exports
```

**Dependencies:** 7 (needs cleanup issue #71)
**Status:** ⚠️ Placeholder

## 🎲 Specialized Markets

### @neural-trader/sports-betting
**Sports betting (partial implementation)**

```typescript
import { createSportsBetting, calculateKellyCriterion, findArbitrageOpportunities } from '@neural-trader/sports-betting';

const betting = createSportsBetting();
const kelly = calculateKellyCriterion(probability, odds, bankroll);
```

**Dependencies:** @neural-trader/core
**Status:** ⚠️ Partial (30% implemented)

---

### @neural-trader/prediction-markets
**Prediction markets (placeholder)**

```typescript
import { PredictionMarkets } from '@neural-trader/prediction-markets';

// Module loads but is empty
```

**Dependencies:** @neural-trader/core
**Status:** ❌ Empty (issue #72)

---

### @neural-trader/syndicate
**Collaborative trading syndicates**

```typescript
import { SyndicateManager } from '@neural-trader/syndicate';

const syndicate = new SyndicateManager({
  name: 'Pro Traders',
  capital: 100000
});

await syndicate.addMember({ name: 'John', contribution: 25000 });
const allocation = await syndicate.allocate({ opportunities, strategy: 'kelly' });
```

**Features:**
- Member management
- Kelly Criterion allocation
- Profit distribution
- Governance system
- 20+ CLI commands

**Dependencies:** @neural-trader/core + 3 CLI tools
**Status:** ✅ Stable (exemplary implementation)

## 🔌 Integration

### @neural-trader/mcp
**MCP server**

```bash
# Start MCP server
npx neural-trader-mcp

# Or via main package
npx neural-trader mcp
```

**Features:**
- 15 MCP tools for AI assistants
- Syndicate management tools
- Trading strategy execution
- Portfolio optimization

**Dependencies:** @neural-trader/core, @neural-trader/mcp-protocol
**Status:** ✅ Stable

---

### @neural-trader/mcp-protocol
**MCP protocol types**

```typescript
import { createRequest, createResponse, createNotification } from '@neural-trader/mcp-protocol';

const request = createRequest('initialize', params);
```

**Features:**
- JSON-RPC 2.0 protocol
- Type-safe message creation
- Request/response/notification types

**Dependencies:** @neural-trader/core
**Status:** ✅ Stable

## 📦 Meta Package

### neural-trader
**All-in-one package**

```bash
npm install neural-trader
```

**Includes:**
- All 17 @neural-trader packages
- Complete CLI with all commands
- MCP server
- Examples and documentation

**Dependencies:** All @neural-trader packages + chalk + commander
**Status:** ✅ Stable

## 🔗 Quick Links

- [Getting Started Guide](../guides/getting-started.md)
- [Package Selection Guide](../guides/package-selection.md)
- [Development Guide](../development/)
- [Main Documentation](../../../../../docs/)

---

[← Back to Packages Docs](../README.md)
