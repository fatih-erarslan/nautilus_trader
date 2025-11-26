# Getting Started with Neural Trader Packages

This guide helps you choose and install the right packages for your use case.

## 📦 Package Options

Neural Trader offers two installation approaches:

### Option 1: Meta Package (Recommended)

Install everything with a single package:

```bash
npm install neural-trader
```

**Includes:**
- All 17 @neural-trader packages
- Complete CLI with all commands
- MCP server with 15 tools
- Examples and documentation

**Use when:**
- You want all features available
- Exploring Neural Trader capabilities
- Building a full trading platform

### Option 2: Individual Packages (Minimal)

Install only what you need:

```bash
npm install @neural-trader/core @neural-trader/strategies
```

**Use when:**
- Minimizing bundle size
- Specific feature requirements
- Integrating into existing system

## 🎯 Common Use Cases

### 1. Trading Bot with Backtesting

```bash
npm install @neural-trader/core \
            @neural-trader/strategies \
            @neural-trader/backtesting \
            @neural-trader/execution \
            @neural-trader/risk
```

**Example:**
```typescript
import { StrategyRunner } from '@neural-trader/strategies';
import { BacktestEngine } from '@neural-trader/backtesting';
import { RiskManager } from '@neural-trader/risk';

// Run momentum strategy backtest
const strategy = new StrategyRunner('momentum');
const backtest = new BacktestEngine(strategy);
const results = await backtest.run({
  symbol: 'SPY',
  start: '2020-01-01',
  end: '2024-01-01'
});
```

### 2. Neural Network Forecasting

```bash
npm install @neural-trader/core \
            @neural-trader/neural \
            @neural-trader/market-data
```

**Example:**
```typescript
import { NeuralModel } from '@neural-trader/neural';

// Train LSTM model
const model = new NeuralModel('lstm');
await model.train({
  symbol: 'AAPL',
  epochs: 100
});

// Generate forecast
const forecast = await model.predict({
  horizon: 24,
  confidence: 0.95
});
```

### 3. Portfolio Optimization

```bash
npm install @neural-trader/core \
            @neural-trader/portfolio \
            @neural-trader/risk
```

**Example:**
```typescript
import { PortfolioOptimizer } from '@neural-trader/portfolio';
import { RiskManager } from '@neural-trader/risk';

const optimizer = new PortfolioOptimizer();
const allocation = await optimizer.optimize({
  assets: ['SPY', 'TLT', 'GLD'],
  constraints: {
    maxDrawdown: 0.15,
    minSharpe: 1.5
  }
});
```

### 4. Sports Betting Syndicate

```bash
npm install @neural-trader/syndicate \
            @neural-trader/sports-betting
```

**Example:**
```typescript
import { SyndicateManager } from '@neural-trader/syndicate';

// Create syndicate
const syndicate = new SyndicateManager({
  name: 'Pro Sports Bettors',
  capital: 100000
});

// Add members
await syndicate.addMember({
  name: 'John',
  contribution: 25000,
  role: 'analyst'
});

// Allocate funds using Kelly Criterion
const allocation = await syndicate.allocate({
  opportunities: sportsOdds,
  strategy: 'kelly_criterion'
});
```

### 5. MCP Server for AI Assistants

```bash
npm install neural-trader
```

**Start server:**
```bash
npx neural-trader mcp
```

**Configure in Claude Desktop:**
```json
{
  "mcpServers": {
    "neural-trader": {
      "command": "npx",
      "args": ["neural-trader", "mcp"]
    }
  }
}
```

## 📚 Package Dependencies

### Dependency Tree

```
neural-trader (meta package)
├── @neural-trader/core (0 deps)
├── @neural-trader/strategies (→ core)
├── @neural-trader/neural (→ core)
├── @neural-trader/portfolio (→ core)
├── @neural-trader/risk (→ core)
├── @neural-trader/backtesting (→ core)
├── @neural-trader/execution (→ core)
├── @neural-trader/features (→ core)
├── @neural-trader/market-data (0 deps)
├── @neural-trader/brokers (0 deps)
├── @neural-trader/news-trading (→ core)
├── @neural-trader/sports-betting (→ core)
├── @neural-trader/prediction-markets (→ core)
├── @neural-trader/syndicate (→ core + 3 CLI deps)
├── @neural-trader/mcp (→ core + mcp-protocol)
├── @neural-trader/mcp-protocol (→ core)
└── @neural-trader/benchoptimizer (→ core + 11 tools)
```

### Peer Dependencies

All packages (except core, market-data, brokers) have a peer dependency on:
```json
{
  "peerDependencies": {
    "@neural-trader/core": "^1.0.0"
  }
}
```

## ⚙️ TypeScript Configuration

All packages include TypeScript definitions:

```typescript
// Types are automatically available
import { Strategy, BacktestResult } from '@neural-trader/core';
import { StrategyRunner } from '@neural-trader/strategies';

const runner: StrategyRunner = new StrategyRunner('momentum');
const result: BacktestResult = await runner.backtest(/*...*/);
```

**tsconfig.json:**
```json
{
  "compilerOptions": {
    "moduleResolution": "node",
    "esModuleInterop": true,
    "resolveJsonModule": true
  }
}
```

## 🔧 Platform-Specific Bindings

Some packages use native Rust bindings (NAPI):

**Packages with native bindings:**
- @neural-trader/strategies
- @neural-trader/neural
- @neural-trader/portfolio
- @neural-trader/risk
- @neural-trader/backtesting
- @neural-trader/execution
- @neural-trader/features

**Supported platforms:**
- Linux x64 (GNU)
- macOS x64/ARM64
- Windows x64 (MSVC)

**Note:** Native bindings are included automatically during install.

## 📖 Next Steps

1. [Package Selection Guide](./package-selection.md) - Choose the right packages
2. [Integration Patterns](./integration.md) - Best practices
3. [Testing Packages](./testing.md) - Verify your setup
4. [Package API Reference](../packages/) - Detailed API docs

## 🆘 Troubleshooting

**Issue: Module not found**
```bash
# Ensure peer dependencies are installed
npm install @neural-trader/core
```

**Issue: Native binding errors**
```bash
# Rebuild native modules
npm rebuild
```

**Issue: TypeScript errors**
```bash
# Install type definitions
npm install --save-dev @types/node
```

See [Troubleshooting Guide](./troubleshooting.md) for more help.

---

[← Back to Packages Docs](../README.md) | [Package Selection →](./package-selection.md)
