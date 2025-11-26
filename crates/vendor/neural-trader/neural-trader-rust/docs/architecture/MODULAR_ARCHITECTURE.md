# Neural Trader - Modular Plugin Architecture

**Vision**: Install only what you need, or get the complete platform.

---

## 🎯 Architecture Overview

### Plugin-Style Approach

```bash
# Install only what you need
npm install @neural-trader/core           # Core types only (minimal)
npm install @neural-trader/backtesting    # Just backtesting
npm install @neural-trader/neural         # Just AI models
npm install @neural-trader/risk           # Just risk management

# Or get everything
npm install neural-trader                 # Full platform
```

---

## 📦 Package Structure

### Core Package (Required Base)

**`@neural-trader/core`** - Foundation package
- Core types and interfaces
- No NAPI bindings (pure TypeScript types)
- Minimal dependencies
- Size: ~50 KB

```typescript
// Exports: Types only
export type { Bar, Signal, Order, Position, TradeDirection };
export type { BrokerConfig, StrategyConfig, RiskConfig };
export interface INeuralTraderPlugin {
  name: string;
  version: string;
  initialize(): Promise<void>;
}
```

### Functional Packages (Optional Plugins)

Each package maps to a Rust crate with NAPI bindings:

#### 1. **@neural-trader/backtesting**
**Rust Crate**: `nt-backtesting`
**Size**: ~300 KB (with NAPI binary)
**Dependencies**: `@neural-trader/core`, `@neural-trader/execution`

```typescript
import { BacktestEngine, compareBacktests } from '@neural-trader/backtesting';

const engine = new BacktestEngine({
  startDate: '2024-01-01',
  endDate: '2024-12-31',
  initialCapital: 100000
});

const results = await engine.run(strategy, historicalData);
```

**Features**:
- Backtesting engine
- Walk-forward analysis
- Monte Carlo simulation
- Performance metrics
- Comparison tools

---

#### 2. **@neural-trader/neural**
**Rust Crate**: `nt-neural`
**Size**: ~1.2 MB (with ML models)
**Dependencies**: `@neural-trader/core`

```typescript
import { NeuralModel, BatchPredictor, ModelType } from '@neural-trader/neural';

const model = new NeuralModel({
  modelType: ModelType.LSTM,
  inputSize: 10,
  hiddenSize: 128
});

await model.train(trainingData);
const predictions = await model.predict(testData);
```

**Features**:
- LSTM, GRU, TCN models
- Transformer models
- DeepAR forecasting
- N-BEATS
- Prophet integration
- Batch prediction
- Model versioning

---

#### 3. **@neural-trader/risk**
**Rust Crate**: `nt-risk`
**Size**: ~250 KB
**Dependencies**: `@neural-trader/core`, `@neural-trader/portfolio`

```typescript
import { RiskManager, calculateVaR, kellyCriterion } from '@neural-trader/risk';

const riskManager = new RiskManager({
  maxDrawdown: 0.15,
  maxLeverage: 3.0,
  confidenceLevel: 0.95
});

const var95 = await riskManager.calculateVaR(portfolio, method: 'historical');
const kellySize = kellyCriterion(winRate, avgWin, avgLoss);
```

**Features**:
- VaR calculation (historical, parametric, Monte Carlo)
- CVaR/Expected Shortfall
- Kelly Criterion
- Position sizing
- Drawdown management
- Leverage limits
- Risk-adjusted returns (Sharpe, Sortino, Calmar)

---

#### 4. **@neural-trader/strategies**
**Rust Crate**: `nt-strategies`
**Size**: ~400 KB
**Dependencies**: `@neural-trader/core`, `@neural-trader/risk`

```typescript
import {
  MomentumStrategy,
  MeanReversionStrategy,
  ArbitrageStrategy,
  MarketMakingStrategy,
  PairsTradingStrategy
} from '@neural-trader/strategies';

const momentum = new MomentumStrategy({
  lookbackPeriod: 20,
  entryThreshold: 0.02,
  exitThreshold: 0.01
});

const signals = await momentum.generateSignals(marketData);
```

**Features**:
- Momentum strategies
- Mean reversion
- Arbitrage detection
- Market making
- Pairs trading
- Custom strategy framework

---

#### 5. **@neural-trader/sports-betting**
**Rust Crate**: `nt-sports-betting`
**Size**: ~350 KB
**Dependencies**: `@neural-trader/core`, `@neural-trader/risk`

```typescript
import {
  SportsBettingEngine,
  kellyCriterionBetting,
  findArbitrage,
  SyndicateManager
} from '@neural-trader/sports-betting';

const engine = new SportsBettingEngine({
  sports: ['soccer', 'basketball'],
  bookmakers: ['pinnacle', 'betfair']
});

const opportunities = await engine.findArbitrage();
const syndicate = new SyndicateManager({ members: 5 });
await syndicate.allocateFunds(opportunities);
```

**Features**:
- Sports odds aggregation
- Arbitrage detection
- Kelly Criterion betting
- Syndicate management
- Profit distribution
- Market depth analysis

---

#### 6. **@neural-trader/prediction-markets**
**Rust Crate**: `nt-prediction-markets`
**Size**: ~300 KB
**Dependencies**: `@neural-trader/core`

```typescript
import {
  PredictionMarketClient,
  calculateExpectedValue,
  analyzeSentiment
} from '@neural-trader/prediction-markets';

const client = new PredictionMarketClient({
  platforms: ['polymarket', 'augur']
});

const markets = await client.getMarkets({ category: 'politics' });
const ev = calculateExpectedValue(market, probability);
```

**Features**:
- Polymarket integration
- Augur support
- Sentiment analysis
- Expected value calculation
- Market orderbook
- Automated trading

---

#### 7. **@neural-trader/news-trading**
**Rust Crate**: `nt-news-trading`
**Size**: ~400 KB
**Dependencies**: `@neural-trader/core`, `@neural-trader/strategies`

```typescript
import { NewsTrader, sentimentAnalysis } from '@neural-trader/news-trading';

const newsTrader = new NewsTrader({
  sources: ['reuters', 'bloomberg', 'twitter'],
  symbols: ['AAPL', 'TSLA'],
  sentimentThreshold: 0.7
});

await newsTrader.subscribe(async (signal) => {
  console.log('News signal:', signal);
});
```

**Features**:
- Multi-source news aggregation
- Real-time sentiment analysis
- Event-driven trading
- News trend analysis

---

#### 8. **@neural-trader/portfolio**
**Rust Crate**: `nt-portfolio`
**Size**: ~300 KB
**Dependencies**: `@neural-trader/core`

```typescript
import {
  PortfolioManager,
  PortfolioOptimizer,
  performanceMetrics
} from '@neural-trader/portfolio';

const portfolio = new PortfolioManager({ initialCash: 100000 });
await portfolio.buy('AAPL', 100, 150.00);

const optimizer = new PortfolioOptimizer({
  method: 'mean-variance',
  constraints: { maxWeight: 0.2 }
});

const allocation = await optimizer.optimize(assets, returns);
```

**Features**:
- Position tracking
- P&L calculation
- Portfolio optimization
- Rebalancing
- Performance attribution

---

#### 9. **@neural-trader/execution**
**Rust Crate**: `nt-execution`
**Size**: ~250 KB
**Dependencies**: `@neural-trader/core`

```typescript
import { ExecutionEngine, OrderRouter } from '@neural-trader/execution';

const executor = new ExecutionEngine({
  slippageModel: 'linear',
  commissionModel: 'tiered'
});

await executor.executeOrder({
  symbol: 'AAPL',
  quantity: 100,
  side: 'buy',
  orderType: 'limit',
  limitPrice: 150.00
});
```

**Features**:
- Order execution simulation
- Slippage modeling
- Commission calculation
- Smart order routing
- TWAP/VWAP execution

---

#### 10. **@neural-trader/market-data**
**Rust Crate**: `nt-market-data`
**Size**: ~350 KB
**Dependencies**: `@neural-trader/core`

```typescript
import { MarketDataProvider, StreamingClient } from '@neural-trader/market-data';

const provider = new MarketDataProvider({
  source: 'alpaca',
  apiKey: process.env.ALPACA_KEY
});

const bars = await provider.getBars('AAPL', {
  start: '2024-01-01',
  end: '2024-12-31',
  timeframe: '1Day'
});

// Real-time streaming
const stream = new StreamingClient({ symbols: ['AAPL', 'TSLA'] });
await stream.subscribe((bar) => console.log(bar));
```

**Features**:
- Historical data fetching
- Real-time streaming
- Multiple data sources (Alpaca, Polygon, IEX)
- Data normalization
- Caching

---

#### 11. **@neural-trader/brokers**
**Rust Crate**: `nt-brokers` (combines broker integrations)
**Size**: ~500 KB
**Dependencies**: `@neural-trader/core`, `@neural-trader/execution`

```typescript
import { AlpacaClient, IBKRClient, CCXTClient } from '@neural-trader/brokers';

const alpaca = new AlpacaClient({
  apiKey: process.env.ALPACA_KEY,
  apiSecret: process.env.ALPACA_SECRET,
  paper: true
});

await alpaca.placeOrder({
  symbol: 'AAPL',
  qty: 100,
  side: 'buy',
  type: 'market'
});
```

**Features**:
- Alpaca integration
- Interactive Brokers (IBKR)
- CCXT (crypto exchanges)
- Oanda (forex)
- Questrade (Canadian)
- Lime (Brazilian)

---

#### 12. **@neural-trader/features**
**Rust Crate**: `nt-features`
**Size**: ~200 KB
**Dependencies**: `@neural-trader/core`

```typescript
import {
  calculateSMA,
  calculateEMA,
  calculateRSI,
  calculateMACD,
  calculateBollingerBands
} from '@neural-trader/features';

const sma20 = calculateSMA(prices, 20);
const rsi14 = calculateRSI(prices, 14);
const { macd, signal, histogram } = calculateMACD(prices);
```

**Features**:
- Technical indicators (50+)
- Moving averages
- Oscillators
- Volume indicators
- Custom indicator framework

---

#### 13. **@neural-trader/distributed**
**Rust Crate**: `neural-trader-distributed`
**Size**: ~400 KB
**Dependencies**: `@neural-trader/core`

```typescript
import { DistributedBacktest, ClusterManager } from '@neural-trader/distributed';

const cluster = new ClusterManager({
  nodes: 4,
  coordinationType: 'raft'
});

const backtest = new DistributedBacktest({
  cluster,
  parallelism: 4
});

const results = await backtest.run(strategies, data);
```

**Features**:
- Distributed backtesting
- Multi-agent coordination
- Cluster management
- Result aggregation

---

### Meta Package (Full Platform)

#### **`neural-trader`** - Complete Platform
**Size**: ~5 MB (all binaries)
**Dependencies**: All `@neural-trader/*` packages

```typescript
// Single import for everything
import {
  NeuralTrader,
  BacktestEngine,
  RiskManager,
  PortfolioOptimizer,
  // ... everything
} from 'neural-trader';

// Or import specific modules
import * as Backtesting from 'neural-trader/backtesting';
import * as Neural from 'neural-trader/neural';
import * as Risk from 'neural-trader/risk';
```

---

## 🏗️ Implementation Strategy

### Phase 1: Core Package Structure

1. **Create base types package**:
   ```bash
   /packages/core/                # TypeScript types only
   /packages/backtesting/         # NAPI bindings for nt-backtesting
   /packages/neural/              # NAPI bindings for nt-neural
   /packages/risk/                # NAPI bindings for nt-risk
   ```

2. **Each package has**:
   ```
   package/
   ├── package.json
   ├── index.js               # Main exports
   ├── index.d.ts             # TypeScript types
   ├── Cargo.toml             # Rust build config
   ├── src/
   │   └── lib.rs             # NAPI bindings
   └── README.md
   ```

3. **Package naming convention**:
   ```json
   {
     "name": "@neural-trader/backtesting",
     "version": "1.0.0",
     "main": "index.js",
     "types": "index.d.ts",
     "napi": {
       "name": "backtesting",
       "package": {
         "name": "@neural-trader/backtesting"
       }
     },
     "peerDependencies": {
       "@neural-trader/core": "^1.0.0"
     }
   }
   ```

### Phase 2: NAPI Bindings per Package

Each package builds its own `.node` binary:

```toml
# packages/backtesting/Cargo.toml
[package]
name = "nt-backtesting-napi"
version = "1.0.0"

[lib]
crate-type = ["cdylib"]
name = "backtesting"

[dependencies]
nt-backtesting = { path = "../../crates/backtesting" }
nt-core = { path = "../../crates/core" }
napi = "2.16"
napi-derive = "2.16"

[package.metadata.napi]
name = "backtesting"
```

Build command:
```bash
cd packages/backtesting
npm run build  # → backtesting.linux-x64-gnu.node
```

### Phase 3: Monorepo Structure

```
neural-trader/
├── packages/
│   ├── core/                    # @neural-trader/core
│   ├── backtesting/             # @neural-trader/backtesting
│   ├── neural/                  # @neural-trader/neural
│   ├── risk/                    # @neural-trader/risk
│   ├── strategies/              # @neural-trader/strategies
│   ├── sports-betting/          # @neural-trader/sports-betting
│   ├── prediction-markets/      # @neural-trader/prediction-markets
│   ├── news-trading/            # @neural-trader/news-trading
│   ├── portfolio/               # @neural-trader/portfolio
│   ├── execution/               # @neural-trader/execution
│   ├── market-data/             # @neural-trader/market-data
│   ├── brokers/                 # @neural-trader/brokers
│   ├── features/                # @neural-trader/features
│   └── all/                     # neural-trader (meta package)
├── crates/                      # Rust crates (existing)
└── package.json                 # Workspace root
```

### Phase 4: Workspace Configuration

```json
// Root package.json
{
  "name": "neural-trader-workspace",
  "private": true,
  "workspaces": [
    "packages/*"
  ],
  "scripts": {
    "build": "npm run build --workspaces",
    "build:backtesting": "npm run build -w @neural-trader/backtesting",
    "build:neural": "npm run build -w @neural-trader/neural",
    "build:all": "npm run build -w neural-trader"
  }
}
```

---

## 💡 Usage Examples

### Minimal Installation (Types Only)

```bash
npm install @neural-trader/core
```

```typescript
import type { Bar, Signal, Order } from '@neural-trader/core';

// Use types for your own implementations
function processBar(bar: Bar): void {
  console.log(bar.symbol, bar.close);
}
```

### Backtesting Only

```bash
npm install @neural-trader/core @neural-trader/backtesting
```

```typescript
import { BacktestEngine } from '@neural-trader/backtesting';

const engine = new BacktestEngine(config);
const results = await engine.run(strategy, data);
```

### AI Trading

```bash
npm install @neural-trader/core @neural-trader/neural @neural-trader/strategies
```

```typescript
import { NeuralModel } from '@neural-trader/neural';
import { MomentumStrategy } from '@neural-trader/strategies';

const model = new NeuralModel({ modelType: 'LSTM' });
const strategy = new MomentumStrategy({ model });
```

### Sports Betting

```bash
npm install @neural-trader/core @neural-trader/sports-betting @neural-trader/risk
```

```typescript
import { SportsBettingEngine } from '@neural-trader/sports-betting';
import { kellyCriterion } from '@neural-trader/risk';

const engine = new SportsBettingEngine(config);
const opportunities = await engine.findArbitrage();
```

### Full Platform

```bash
npm install neural-trader
```

```typescript
import {
  BacktestEngine,
  NeuralModel,
  RiskManager,
  PortfolioOptimizer
} from 'neural-trader';

// Everything available
```

---

## 📊 Package Comparison

| Package | Size | Use Case | Dependencies |
|---------|------|----------|--------------|
| `@neural-trader/core` | 50 KB | Types only | None |
| `@neural-trader/backtesting` | 300 KB | Historical testing | core, execution |
| `@neural-trader/neural` | 1.2 MB | AI/ML models | core |
| `@neural-trader/risk` | 250 KB | Risk management | core, portfolio |
| `@neural-trader/strategies` | 400 KB | Trading strategies | core, risk |
| `@neural-trader/sports-betting` | 350 KB | Sports betting | core, risk |
| `@neural-trader/prediction-markets` | 300 KB | Prediction markets | core |
| `@neural-trader/news-trading` | 400 KB | News-driven trading | core, strategies |
| `@neural-trader/portfolio` | 300 KB | Portfolio management | core |
| `@neural-trader/execution` | 250 KB | Order execution | core |
| `@neural-trader/market-data` | 350 KB | Market data | core |
| `@neural-trader/brokers` | 500 KB | Broker integrations | core, execution |
| `@neural-trader/features` | 200 KB | Technical indicators | core |
| `neural-trader` (all) | ~5 MB | Full platform | All above |

---

## 🔧 Development Workflow

### Adding a New Plugin

1. **Create package directory**:
   ```bash
   mkdir -p packages/my-plugin
   cd packages/my-plugin
   ```

2. **Initialize package**:
   ```bash
   npm init -y
   napi init
   ```

3. **Create NAPI bindings**:
   ```rust
   // src/lib.rs
   use napi::bindgen_prelude::*;
   use nt_my_crate::*;

   #[napi]
   pub fn my_function() -> String {
     "Hello from my plugin".to_string()
   }
   ```

4. **Build**:
   ```bash
   npm run build
   ```

5. **Test**:
   ```typescript
   import { myFunction } from '@neural-trader/my-plugin';
   console.log(myFunction());
   ```

---

## 🚀 Benefits

### For Users

✅ **Install only what you need** - Smaller bundle sizes
✅ **Faster installation** - Only download required binaries
✅ **Clear dependencies** - Know exactly what you're using
✅ **Better tree-shaking** - Dead code elimination
✅ **Version independence** - Update packages separately

### For Developers

✅ **Modular codebase** - Easier to maintain
✅ **Independent releases** - Update packages separately
✅ **Clear boundaries** - Well-defined APIs
✅ **Parallel development** - Multiple teams
✅ **Better testing** - Test packages in isolation

---

## 📝 Migration Path

### Current (Monolithic)

```typescript
import { BacktestEngine, NeuralModel, RiskManager } from '@neural-trader/core';
```

Bundle size: 5 MB (everything)

### Future (Modular)

```typescript
import { BacktestEngine } from '@neural-trader/backtesting';
import { NeuralModel } from '@neural-trader/neural';
import { RiskManager } from '@neural-trader/risk';
```

Bundle size: 1.75 MB (only what you need)

### Backwards Compatibility

Keep `neural-trader` meta package that re-exports everything:

```typescript
// Still works
import { BacktestEngine } from 'neural-trader';

// Also works
import { BacktestEngine } from '@neural-trader/backtesting';
```

---

## 🎯 Next Steps

1. ✅ Create `@neural-trader/core` with TypeScript types
2. ⏳ Migrate current NAPI bindings to `@neural-trader/backtesting`
3. ⏳ Create `@neural-trader/neural` package
4. ⏳ Create `@neural-trader/risk` package
5. ⏳ Create remaining packages
6. ⏳ Create `neural-trader` meta package
7. ⏳ Update documentation
8. ⏳ Publish to npm

---

**Target Launch**: Q1 2025
**Current Status**: Phase 1 - Architecture design complete
