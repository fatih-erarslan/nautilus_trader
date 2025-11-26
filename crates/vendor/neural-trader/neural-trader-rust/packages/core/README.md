# @neural-trader/core

[![CI Status](https://github.com/ruvnet/neural-trader/workflows/Rust%20CI/badge.svg)](https://github.com/ruvnet/neural-trader/actions)
[![codecov](https://codecov.io/gh/ruvnet/neural-trader/branch/main/graph/badge.svg)](https://codecov.io/gh/ruvnet/neural-trader)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](../../LICENSE)
[![npm version](https://badge.fury.io/js/%40neural-trader%2Fcore.svg)](https://www.npmjs.com/package/@neural-trader/core)

**Zero-dependency TypeScript types and interfaces for Neural Trader - The foundation of AI-first algorithmic trading**

## Introduction

`@neural-trader/core` is the foundational types package for the entire Neural Trader ecosystem. It provides **type definitions only** with **absolutely zero runtime dependencies**, making it the perfect base for type-safe trading applications.

This package serves as the single source of truth for all type definitions across Neural Trader packages, ensuring type safety and consistency from market data ingestion to neural model predictions to risk management and execution.

Every Neural Trader package depends on `@neural-trader/core` for its type definitions, creating a unified, type-safe development experience across the entire platform. Whether you're building strategies, backtesting systems, neural forecasters, or risk engines, these types ensure correctness at compile time.

## Features

- **Zero Dependencies**: Pure TypeScript types with no runtime overhead
- **Comprehensive Coverage**: 50+ interfaces covering every trading domain
- **Strict Type Safety**: Built with strict TypeScript compiler settings
- **Self-Documenting**: Rich JSDoc comments for every type
- **Neural-First Design**: Native support for ML model types and predictions
- **Risk-Aware Types**: Built-in risk metrics, VaR, CVaR, and Kelly Criterion
- **Backtest-Ready**: Complete backtest configuration and results types
- **Multi-Asset Support**: Types for stocks, crypto, forex, and more
- **Real-Time Ready**: WebSocket-compatible quote and bar types
- **Portfolio-Focused**: Position, account, and optimization types

## Installation

### Via npm (Node.js)

```bash
# Core package (required by all other packages)
npm install @neural-trader/core

# With related packages
npm install @neural-trader/core @neural-trader/backtesting @neural-trader/neural

# Full platform (all features)
npm install neural-trader
```

### Via Cargo (Rust)

```bash
# Add to Cargo.toml
[dependencies]
neural-trader-core = "1.0.0"
```

**Package Size**: ~50 KB (types only, zero runtime dependencies)

See [main packages documentation](../README.md) for all available packages.

## Quick Start

**30-second example** showing basic usage:

```typescript
import {
  BacktestConfig,
  Signal,
  Position,
  RiskConfig,
  ModelConfig
} from '@neural-trader/core';

// Define backtest configuration
const backtestConfig: BacktestConfig = {
  initialCapital: 100000,
  startDate: '2024-01-01',
  endDate: '2024-12-31',
  commission: 0.001,      // 0.1%
  slippage: 0.0005,       // 0.05%
  useMarkToMarket: true
};

// Create a trading signal
const signal: Signal = {
  symbol: 'AAPL',
  timestamp: new Date().toISOString(),
  type: 'BUY',
  strength: 0.85,
  price: 175.50,
  reason: 'Neural momentum indicator crossover'
};

// Define risk parameters
const riskConfig: RiskConfig = {
  confidenceLevel: 0.95,
  lookbackPeriods: 252,
  method: 'historical'
};

console.log('Backtest Config:', backtestConfig);
console.log('Signal:', signal);
console.log('Risk Config:', riskConfig);
```

**Expected Output:**
```
Backtest Config: { initialCapital: 100000, startDate: '2024-01-01', ... }
Signal: { symbol: 'AAPL', type: 'BUY', strength: 0.85, ... }
Risk Config: { confidenceLevel: 0.95, lookbackPeriods: 252, method: 'historical' }
```

## Core Concepts

### Market Data Types

Market data is represented through standardized interfaces that work with any data provider.

```typescript
import { Bar, Quote, MarketDataConfig } from '@neural-trader/core';

// OHLCV bar/candle
const bar: Bar = {
  symbol: 'AAPL',
  timestamp: '2024-01-15T09:30:00Z',
  open: 175.20,
  high: 176.50,
  low: 174.80,
  close: 176.20,
  volume: 1250000
};

// Real-time quote
const quote: Quote = {
  symbol: 'AAPL',
  bid: 176.18,
  ask: 176.22,
  bidSize: 100,
  askSize: 150,
  last: 176.20,
  lastSize: 50,
  timestamp: '2024-01-15T14:35:42.123Z'
};

// Market data provider configuration
const config: MarketDataConfig = {
  provider: 'alpaca',
  apiKey: 'your-key',
  websocketEnabled: true
};
```

### Trading Signals and Orders

Trading signals represent strategy decisions, while orders handle execution.

```typescript
import { Signal, OrderRequest, OrderResponse, Position } from '@neural-trader/core';

// Trading signal from strategy
const signal: Signal = {
  symbol: 'TSLA',
  timestamp: new Date().toISOString(),
  type: 'SELL',
  strength: 0.72,
  price: 245.80,
  reason: 'Mean reversion signal at resistance'
};

// Order placement request
const order: OrderRequest = {
  symbol: 'TSLA',
  side: 'SELL',
  orderType: 'LIMIT',
  quantity: 100,
  limitPrice: 246.00,
  timeInForce: 'GTC'
};

// Order response from broker
const response: OrderResponse = {
  orderId: 'ord_123456',
  brokerOrderId: 'broker_789',
  status: 'FILLED',
  filledQuantity: 100,
  filledPrice: 245.95,
  timestamp: '2024-01-15T14:36:00Z'
};

// Position tracking
const position: Position = {
  symbol: 'TSLA',
  quantity: -100,           // Negative = short
  avgEntryPrice: 245.95,
  currentPrice: 244.50,
  unrealizedPnl: 145.00,    // Profit on short
  realizedPnl: 0,
  marketValue: -24450.00
};
```

### Neural Network Types

Native support for neural forecasting models with configuration and predictions.

```typescript
import {
  ModelConfig,
  TrainingConfig,
  PredictionResult,
  TrainingMetrics,
  ModelType
} from '@neural-trader/core';

// Model configuration
const modelConfig: ModelConfig = {
  modelType: 'NHITS',
  inputSize: 30,      // 30 days lookback
  horizon: 5,         // 5 days forecast
  hiddenSize: 128,
  numLayers: 3,
  dropout: 0.2,
  learningRate: 0.001
};

// Training configuration
const trainingConfig: TrainingConfig = {
  epochs: 100,
  batchSize: 32,
  validationSplit: 0.2,
  earlyStoppingPatience: 10,
  useGpu: true
};

// Prediction result with confidence intervals
const prediction: PredictionResult = {
  predictions: [176.5, 177.2, 178.0, 177.8, 179.1],
  lowerBound: [174.2, 174.8, 175.3, 175.1, 176.2],
  upperBound: [178.8, 179.6, 180.7, 180.5, 182.0],
  timestamp: '2024-01-15T15:00:00Z'
};

// Training progress metrics
const metrics: TrainingMetrics = {
  epoch: 45,
  trainLoss: 0.0234,
  valLoss: 0.0289,
  trainMae: 1.45,
  valMae: 1.82
};
```

### Key Terminology

- **Bar**: OHLCV market data candle/bar for a specific time period
- **Signal**: Trading decision with direction, strength, and reasoning
- **Position**: Current holdings in a symbol with PnL tracking
- **VaR**: Value at Risk - maximum expected loss at a confidence level
- **Kelly Criterion**: Optimal position sizing based on win rate and payoff ratio
- **Drawdown**: Peak-to-trough decline in portfolio value
- **Sharpe Ratio**: Risk-adjusted return metric (return/volatility)

### Architecture Overview

```
┌─────────────────────────────────────────────────┐
│         @neural-trader/core (Types Only)         │
│                                                  │
│  ┌──────────────┐  ┌──────────────┐            │
│  │ Market Data  │  │   Trading    │            │
│  │   Types      │  │    Types     │            │
│  └──────────────┘  └──────────────┘            │
│                                                  │
│  ┌──────────────┐  ┌──────────────┐            │
│  │   Neural     │  │     Risk     │            │
│  │  ML Types    │  │  Management  │            │
│  └──────────────┘  └──────────────┘            │
│                                                  │
│  ┌──────────────┐  ┌──────────────┐            │
│  │  Backtest    │  │  Portfolio   │            │
│  │    Types     │  │   Optimizer  │            │
│  └──────────────┘  └──────────────┘            │
└─────────────────────────────────────────────────┘
           │                │               │
           ▼                ▼               ▼
    ┌─────────┐      ┌──────────┐   ┌──────────┐
    │Backtesting│    │  Neural  │   │   Risk   │
    └─────────┘      └──────────┘   └──────────┘
```

## API Reference

### Market Data Types

#### Bar

OHLCV market data bar/candle.

```typescript
interface Bar {
  symbol: string;
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}
```

**Example:**

```typescript
const bars: Bar[] = [
  {
    symbol: 'AAPL',
    timestamp: '2024-01-15T09:30:00Z',
    open: 175.20,
    high: 176.50,
    low: 174.80,
    close: 176.20,
    volume: 1250000
  },
  {
    symbol: 'AAPL',
    timestamp: '2024-01-15T10:30:00Z',
    open: 176.20,
    high: 177.10,
    low: 175.90,
    close: 176.85,
    volume: 980000
  }
];
```

#### Quote

Real-time bid/ask quote data.

```typescript
interface Quote {
  symbol: string;
  bid: number;
  ask: number;
  bidSize: number;
  askSize: number;
  last: number;
  lastSize: number;
  timestamp: string;
}
```

#### MarketDataConfig

Configuration for market data providers.

```typescript
interface MarketDataConfig {
  provider: string;
  apiKey?: string;
  apiSecret?: string;
  websocketEnabled: boolean;
}
```

### Trading Types

#### Signal

Trading signal from a strategy.

```typescript
interface Signal {
  symbol: string;
  timestamp: string;
  type: 'BUY' | 'SELL' | 'HOLD';
  strength: number;      // 0.0 to 1.0
  price: number;
  reason: string;
}
```

**Example:**

```typescript
const signals: Signal[] = [
  {
    symbol: 'NVDA',
    timestamp: '2024-01-15T14:35:00Z',
    type: 'BUY',
    strength: 0.92,
    price: 495.50,
    reason: 'Neural momentum + volume breakout'
  },
  {
    symbol: 'AMD',
    timestamp: '2024-01-15T14:36:00Z',
    type: 'SELL',
    strength: 0.78,
    price: 142.30,
    reason: 'Overbought on RSI, take profit'
  }
];
```

#### OrderRequest

Order placement request.

```typescript
interface OrderRequest {
  symbol: string;
  side: string;          // 'BUY' or 'SELL'
  orderType: string;     // 'MARKET', 'LIMIT', 'STOP', 'STOP_LIMIT'
  quantity: number;
  limitPrice?: number;
  stopPrice?: number;
  timeInForce: string;   // 'GTC', 'DAY', 'IOC', 'FOK'
}
```

#### Position

Current position in a symbol.

```typescript
interface Position {
  symbol: string;
  quantity: number;           // Positive = long, negative = short
  avgEntryPrice: number;
  currentPrice: number;
  unrealizedPnl: number;
  realizedPnl: number;
  marketValue: number;
}
```

### Risk Management Types

#### RiskConfig

Risk calculation configuration.

```typescript
interface RiskConfig {
  confidenceLevel: number;    // e.g., 0.95 for 95%
  lookbackPeriods: number;    // e.g., 252 for 1 year daily
  method: string;             // 'historical', 'parametric', 'monte_carlo'
}
```

**Example:**

```typescript
const riskConfigs = {
  conservative: {
    confidenceLevel: 0.99,
    lookbackPeriods: 504,    // 2 years
    method: 'historical'
  },
  moderate: {
    confidenceLevel: 0.95,
    lookbackPeriods: 252,    // 1 year
    method: 'parametric'
  },
  aggressive: {
    confidenceLevel: 0.90,
    lookbackPeriods: 126,    // 6 months
    method: 'monte_carlo'
  }
};
```

#### VaRResult

Value at Risk calculation result.

```typescript
interface VaRResult {
  varAmount: number;         // Dollar amount
  varPercentage: number;     // Percentage of portfolio
  confidenceLevel: number;
  method: string;
  portfolioValue: number;
}
```

#### CVaRResult

Conditional Value at Risk (Expected Shortfall).

```typescript
interface CVaRResult {
  cvarAmount: number;        // Expected loss beyond VaR
  cvarPercentage: number;
  varAmount: number;
  confidenceLevel: number;
}
```

#### KellyResult

Kelly Criterion position sizing result.

```typescript
interface KellyResult {
  kellyFraction: number;     // Full Kelly percentage
  halfKelly: number;         // Conservative half-Kelly
  quarterKelly: number;      // Very conservative quarter-Kelly
  winRate: number;
  avgWin: number;
  avgLoss: number;
}
```

**Example:**

```typescript
const kelly: KellyResult = {
  kellyFraction: 0.125,      // 12.5% of capital
  halfKelly: 0.0625,         // 6.25% of capital
  quarterKelly: 0.03125,     // 3.125% of capital
  winRate: 0.58,             // 58% win rate
  avgWin: 250.00,            // $250 average win
  avgLoss: -150.00           // $150 average loss
};

// Position sizing recommendation
const positionSize: PositionSize = {
  shares: 50,
  dollarAmount: 8780.00,
  percentageOfPortfolio: 0.0625,  // Using half-Kelly
  maxLoss: -438.00,
  reasoning: 'Half-Kelly sizing for 58% win rate strategy'
};
```

#### DrawdownMetrics

Portfolio drawdown metrics.

```typescript
interface DrawdownMetrics {
  maxDrawdown: number;           // Maximum peak-to-trough decline
  maxDrawdownDuration: number;   // Days underwater
  currentDrawdown: number;       // Current decline from peak
  recoveryFactor: number;        // Net profit / max drawdown
}
```

### Neural Network Types

#### ModelConfig

Neural network model configuration.

```typescript
interface ModelConfig {
  modelType: string;        // 'NHITS', 'LSTMAttention', 'Transformer'
  inputSize: number;        // Lookback window size
  horizon: number;          // Forecast horizon
  hiddenSize: number;       // Hidden layer size
  numLayers: number;        // Number of layers
  dropout: number;          // Dropout rate
  learningRate: number;     // Learning rate
}
```

#### TrainingConfig

Training parameters.

```typescript
interface TrainingConfig {
  epochs: number;
  batchSize: number;
  validationSplit: number;
  earlyStoppingPatience: number;
  useGpu: boolean;
}
```

#### PredictionResult

Model prediction with confidence intervals.

```typescript
interface PredictionResult {
  predictions: number[];    // Point forecasts
  lowerBound: number[];     // Lower confidence bound
  upperBound: number[];     // Upper confidence bound
  timestamp: string;
}
```

**Example:**

```typescript
const forecast: PredictionResult = {
  predictions: [176.5, 177.2, 178.0, 177.8, 179.1],
  lowerBound: [174.2, 174.8, 175.3, 175.1, 176.2],
  upperBound: [178.8, 179.6, 180.7, 180.5, 182.0],
  timestamp: '2024-01-15T15:00:00Z'
};

console.log('5-day forecast for AAPL:');
forecast.predictions.forEach((price, i) => {
  console.log(`Day ${i+1}: $${price} (${forecast.lowerBound[i]} - ${forecast.upperBound[i]})`);
});
```

### Backtest Types

#### BacktestConfig

Backtest configuration parameters.

```typescript
interface BacktestConfig {
  initialCapital: number;
  startDate: string;
  endDate: string;
  commission: number;       // Per trade commission
  slippage: number;         // Slippage factor
  useMarkToMarket: boolean;
}
```

#### Trade

Individual trade record.

```typescript
interface Trade {
  symbol: string;
  entryDate: string;
  exitDate: string;
  entryPrice: number;
  exitPrice: number;
  quantity: number;
  pnl: number;
  pnlPercentage: number;
  commissionPaid: number;
}
```

#### BacktestMetrics

Performance metrics from backtest.

```typescript
interface BacktestMetrics {
  totalReturn: number;
  annualReturn: number;
  sharpeRatio: number;
  sortinoRatio: number;
  maxDrawdown: number;
  winRate: number;
  profitFactor: number;
  totalTrades: number;
  winningTrades: number;
  losingTrades: number;
  avgWin: number;
  avgLoss: number;
  largestWin: number;
  largestLoss: number;
  finalEquity: number;
}
```

## Detailed Tutorials

### Tutorial 1: Building Type-Safe Trading Strategies

**Goal:** Create a momentum strategy with full type safety using core types.

**Prerequisites:**
- TypeScript 5.0+
- Basic understanding of trading concepts

**Step 1: Define Strategy Configuration**

Use core types to ensure type-safe configuration.

```typescript
import { Bar, Signal, BacktestConfig } from '@neural-trader/core';

interface MomentumConfig {
  symbol: string;
  lookbackPeriod: number;
  threshold: number;
}

const config: MomentumConfig = {
  symbol: 'AAPL',
  lookbackPeriod: 20,
  threshold: 0.02  // 2% threshold
};
```

**Step 2: Process Market Data**

Type-safe market data handling.

```typescript
function calculateMomentum(bars: Bar[]): number {
  if (bars.length < 2) return 0;

  const latest = bars[bars.length - 1];
  const previous = bars[bars.length - 2];

  return (latest.close - previous.close) / previous.close;
}

function generateSignal(bars: Bar[], config: MomentumConfig): Signal {
  const momentum = calculateMomentum(bars);
  const latest = bars[bars.length - 1];

  return {
    symbol: config.symbol,
    timestamp: latest.timestamp,
    type: momentum > config.threshold ? 'BUY' :
          momentum < -config.threshold ? 'SELL' : 'HOLD',
    strength: Math.abs(momentum) / config.threshold,
    price: latest.close,
    reason: `Momentum ${momentum.toFixed(4)} vs threshold ${config.threshold}`
  };
}
```

**Step 3: Run Type-Safe Backtest**

Configure backtest with proper types.

```typescript
const backtestConfig: BacktestConfig = {
  initialCapital: 100000,
  startDate: '2023-01-01',
  endDate: '2023-12-31',
  commission: 0.001,
  slippage: 0.0005,
  useMarkToMarket: true
};

async function runBacktest(
  config: MomentumConfig,
  backtestConfig: BacktestConfig
): Promise<void> {
  // Implementation would use @neural-trader/backtesting
  // All types are enforced at compile time
  console.log('Running backtest with config:', backtestConfig);
}
```

**Complete Example:**

```typescript
import { Bar, Signal, BacktestConfig, Position } from '@neural-trader/core';

interface MomentumStrategy {
  config: MomentumConfig;
  backtestConfig: BacktestConfig;
  bars: Bar[];
  positions: Map<string, Position>;
}

class TypeSafeMomentumStrategy implements MomentumStrategy {
  config: MomentumConfig;
  backtestConfig: BacktestConfig;
  bars: Bar[] = [];
  positions = new Map<string, Position>();

  constructor(config: MomentumConfig, backtestConfig: BacktestConfig) {
    this.config = config;
    this.backtestConfig = backtestConfig;
  }

  addBar(bar: Bar): void {
    this.bars.push(bar);
    if (this.bars.length > this.config.lookbackPeriod) {
      this.bars.shift();
    }
  }

  generateSignal(): Signal | null {
    if (this.bars.length < 2) return null;
    return generateSignal(this.bars, this.config);
  }
}

// Usage
const strategy = new TypeSafeMomentumStrategy(
  { symbol: 'AAPL', lookbackPeriod: 20, threshold: 0.02 },
  { initialCapital: 100000, startDate: '2023-01-01', endDate: '2023-12-31',
    commission: 0.001, slippage: 0.0005, useMarkToMarket: true }
);
```

---

### Tutorial 2: Neural Forecasting with Type Safety

**Goal:** Build a neural forecasting pipeline with complete type safety.

**Prerequisites:**
- Understanding of neural networks
- Familiarity with time series forecasting

**Step 1: Configure Neural Model**

```typescript
import {
  ModelConfig,
  TrainingConfig,
  PredictionResult
} from '@neural-trader/core';

const modelConfig: ModelConfig = {
  modelType: 'NHITS',
  inputSize: 60,      // 60 days of history
  horizon: 10,        // 10-day forecast
  hiddenSize: 256,
  numLayers: 4,
  dropout: 0.2,
  learningRate: 0.001
};

const trainingConfig: TrainingConfig = {
  epochs: 100,
  batchSize: 32,
  validationSplit: 0.2,
  earlyStoppingPatience: 15,
  useGpu: true
};
```

**Step 2: Process Training Data**

```typescript
import { Bar } from '@neural-trader/core';

function prepareForecastData(bars: Bar[]): number[][] {
  return bars.map(bar => [
    bar.open,
    bar.high,
    bar.low,
    bar.close,
    bar.volume
  ]);
}

// Type-safe data preparation
const historicalBars: Bar[] = [/* ... */];
const trainingData = prepareForecastData(historicalBars);
```

**Step 3: Generate Predictions**

```typescript
function createPrediction(
  forecastValues: number[],
  confidenceInterval: number = 0.95
): PredictionResult {
  const std = calculateStd(forecastValues);
  const zScore = 1.96; // 95% confidence

  return {
    predictions: forecastValues,
    lowerBound: forecastValues.map(v => v - zScore * std),
    upperBound: forecastValues.map(v => v + zScore * std),
    timestamp: new Date().toISOString()
  };
}

// Type-safe prediction
const prediction: PredictionResult = createPrediction([
  176.5, 177.2, 178.0, 177.8, 179.1,
  179.5, 180.2, 180.8, 181.2, 181.8
]);
```

**Complete Example:**

```typescript
import {
  ModelConfig,
  TrainingConfig,
  TrainingMetrics,
  PredictionResult,
  Bar
} from '@neural-trader/core';

interface NeuralForecaster {
  modelConfig: ModelConfig;
  trainingConfig: TrainingConfig;
  train(data: Bar[]): Promise<TrainingMetrics[]>;
  predict(data: Bar[]): Promise<PredictionResult>;
}

class TypeSafeNeuralForecaster implements NeuralForecaster {
  modelConfig: ModelConfig;
  trainingConfig: TrainingConfig;
  private trained = false;

  constructor(modelConfig: ModelConfig, trainingConfig: TrainingConfig) {
    this.modelConfig = modelConfig;
    this.trainingConfig = trainingConfig;
  }

  async train(data: Bar[]): Promise<TrainingMetrics[]> {
    const metrics: TrainingMetrics[] = [];

    // Training loop with type-safe metrics
    for (let epoch = 0; epoch < this.trainingConfig.epochs; epoch++) {
      // Simulate training
      metrics.push({
        epoch,
        trainLoss: 0.05 - (epoch * 0.0004),
        valLoss: 0.06 - (epoch * 0.0003),
        trainMae: 2.0 - (epoch * 0.015),
        valMae: 2.5 - (epoch * 0.012)
      });
    }

    this.trained = true;
    return metrics;
  }

  async predict(data: Bar[]): Promise<PredictionResult> {
    if (!this.trained) {
      throw new Error('Model must be trained before prediction');
    }

    // Generate forecast
    const lastClose = data[data.length - 1].close;
    const predictions = Array.from(
      { length: this.modelConfig.horizon },
      (_, i) => lastClose * (1 + 0.001 * i)
    );

    return createPrediction(predictions);
  }
}

// Usage
const forecaster = new TypeSafeNeuralForecaster(modelConfig, trainingConfig);
const trainingMetrics = await forecaster.train(historicalBars);
const forecast = await forecaster.predict(historicalBars);

console.log('Training completed:', trainingMetrics.length, 'epochs');
console.log('Forecast:', forecast.predictions);
```

---

### Tutorial 3: Risk Management System

**Goal:** Implement comprehensive risk management with type-safe calculations.

**Complete Implementation:**

```typescript
import {
  RiskConfig,
  VaRResult,
  CVaRResult,
  KellyResult,
  DrawdownMetrics,
  Position,
  PositionSize
} from '@neural-trader/core';

class RiskManager {
  private config: RiskConfig;
  private portfolioValue: number;
  private positions: Map<string, Position>;

  constructor(config: RiskConfig, portfolioValue: number) {
    this.config = config;
    this.portfolioValue = portfolioValue;
    this.positions = new Map();
  }

  // Calculate Value at Risk
  calculateVaR(returns: number[]): VaRResult {
    const sortedReturns = [...returns].sort((a, b) => a - b);
    const varIndex = Math.floor((1 - this.config.confidenceLevel) * returns.length);
    const varReturn = sortedReturns[varIndex];

    return {
      varAmount: Math.abs(varReturn * this.portfolioValue),
      varPercentage: Math.abs(varReturn),
      confidenceLevel: this.config.confidenceLevel,
      method: this.config.method,
      portfolioValue: this.portfolioValue
    };
  }

  // Calculate Conditional VaR (Expected Shortfall)
  calculateCVaR(returns: number[]): CVaRResult {
    const var95 = this.calculateVaR(returns);
    const sortedReturns = [...returns].sort((a, b) => a - b);
    const varIndex = Math.floor((1 - this.config.confidenceLevel) * returns.length);

    const tailReturns = sortedReturns.slice(0, varIndex);
    const cvarReturn = tailReturns.reduce((a, b) => a + b, 0) / tailReturns.length;

    return {
      cvarAmount: Math.abs(cvarReturn * this.portfolioValue),
      cvarPercentage: Math.abs(cvarReturn),
      varAmount: var95.varAmount,
      confidenceLevel: this.config.confidenceLevel
    };
  }

  // Kelly Criterion position sizing
  calculateKelly(
    winRate: number,
    avgWin: number,
    avgLoss: number
  ): KellyResult {
    const b = avgWin / Math.abs(avgLoss);
    const q = 1 - winRate;
    const kellyFraction = (winRate * b - q) / b;

    return {
      kellyFraction: Math.max(0, Math.min(kellyFraction, 0.25)), // Cap at 25%
      halfKelly: kellyFraction / 2,
      quarterKelly: kellyFraction / 4,
      winRate,
      avgWin,
      avgLoss
    };
  }

  // Position sizing recommendation
  recommendPositionSize(
    symbol: string,
    currentPrice: number,
    kelly: KellyResult
  ): PositionSize {
    const allocation = this.portfolioValue * kelly.halfKelly;
    const shares = Math.floor(allocation / currentPrice);

    return {
      shares,
      dollarAmount: shares * currentPrice,
      percentageOfPortfolio: kelly.halfKelly,
      maxLoss: -allocation * 0.02,  // 2% stop loss
      reasoning: `Half-Kelly sizing based on ${(kelly.winRate * 100).toFixed(1)}% win rate`
    };
  }

  // Calculate drawdown metrics
  calculateDrawdown(equityCurve: number[]): DrawdownMetrics {
    let maxEquity = equityCurve[0];
    let maxDrawdown = 0;
    let maxDrawdownDuration = 0;
    let currentDuration = 0;

    for (const equity of equityCurve) {
      if (equity > maxEquity) {
        maxEquity = equity;
        currentDuration = 0;
      } else {
        currentDuration++;
        const drawdown = (maxEquity - equity) / maxEquity;
        if (drawdown > maxDrawdown) {
          maxDrawdown = drawdown;
          maxDrawdownDuration = currentDuration;
        }
      }
    }

    const currentEquity = equityCurve[equityCurve.length - 1];
    const currentDrawdown = (maxEquity - currentEquity) / maxEquity;
    const netProfit = currentEquity - equityCurve[0];

    return {
      maxDrawdown,
      maxDrawdownDuration,
      currentDrawdown,
      recoveryFactor: netProfit / (maxDrawdown * equityCurve[0])
    };
  }
}

// Usage example
const riskConfig: RiskConfig = {
  confidenceLevel: 0.95,
  lookbackPeriods: 252,
  method: 'historical'
};

const riskManager = new RiskManager(riskConfig, 100000);

// Calculate VaR
const returns = [0.02, -0.01, 0.015, -0.025, 0.01 /* ... */];
const var95 = riskManager.calculateVaR(returns);
console.log(`95% VaR: $${var95.varAmount.toFixed(2)}`);

// Calculate CVaR
const cvar = riskManager.calculateCVaR(returns);
console.log(`Expected Shortfall: $${cvar.cvarAmount.toFixed(2)}`);

// Kelly Criterion sizing
const kelly = riskManager.calculateKelly(0.58, 250, -150);
const positionSize = riskManager.recommendPositionSize('AAPL', 175.50, kelly);
console.log(`Recommended position: ${positionSize.shares} shares ($${positionSize.dollarAmount.toFixed(2)})`);

// Drawdown analysis
const equityCurve = [100000, 102000, 101500, 103000, 99000, 101000 /* ... */];
const drawdown = riskManager.calculateDrawdown(equityCurve);
console.log(`Max Drawdown: ${(drawdown.maxDrawdown * 100).toFixed(2)}%`);
console.log(`Recovery Factor: ${drawdown.recoveryFactor.toFixed(2)}`);
```

---

### Tutorial 4: Multi-Asset Portfolio Type Safety

**Goal:** Manage a multi-asset portfolio with complete type safety.

**Implementation:**

```typescript
import {
  Position,
  Bar,
  Signal,
  OrderRequest,
  AccountBalance,
  RiskConfig
} from '@neural-trader/core';

interface PortfolioState {
  cash: number;
  positions: Map<string, Position>;
  accountBalance: AccountBalance;
}

class Portfolio {
  private state: PortfolioState;
  private riskConfig: RiskConfig;

  constructor(initialCash: number, riskConfig: RiskConfig) {
    this.state = {
      cash: initialCash,
      positions: new Map(),
      accountBalance: {
        cash: initialCash,
        equity: initialCash,
        buyingPower: initialCash,
        currency: 'USD'
      }
    };
    this.riskConfig = riskConfig;
  }

  // Add position
  addPosition(symbol: string, quantity: number, entryPrice: number): void {
    const existing = this.state.positions.get(symbol);

    if (existing) {
      // Average up/down
      const totalQuantity = existing.quantity + quantity;
      const avgPrice = (
        (existing.avgEntryPrice * existing.quantity) +
        (entryPrice * quantity)
      ) / totalQuantity;

      existing.quantity = totalQuantity;
      existing.avgEntryPrice = avgPrice;
    } else {
      this.state.positions.set(symbol, {
        symbol,
        quantity,
        avgEntryPrice: entryPrice,
        currentPrice: entryPrice,
        unrealizedPnl: 0,
        realizedPnl: 0,
        marketValue: quantity * entryPrice
      });
    }

    this.state.cash -= quantity * entryPrice;
    this.updateAccountBalance();
  }

  // Update position with current price
  updatePosition(symbol: string, currentPrice: number): void {
    const position = this.state.positions.get(symbol);
    if (!position) return;

    position.currentPrice = currentPrice;
    position.marketValue = position.quantity * currentPrice;
    position.unrealizedPnl =
      (currentPrice - position.avgEntryPrice) * position.quantity;

    this.updateAccountBalance();
  }

  // Generate order from signal
  signalToOrder(signal: Signal, positionSize: number): OrderRequest | null {
    if (signal.type === 'HOLD') return null;

    return {
      symbol: signal.symbol,
      side: signal.type === 'BUY' ? 'BUY' : 'SELL',
      orderType: 'LIMIT',
      quantity: positionSize,
      limitPrice: signal.price,
      timeInForce: 'GTC'
    };
  }

  // Calculate portfolio metrics
  getMetrics(): {
    totalValue: number;
    totalPnl: number;
    positionCount: number;
    largestPosition: string;
  } {
    let totalPnl = 0;
    let largestValue = 0;
    let largestPosition = '';

    for (const [symbol, position] of this.state.positions) {
      totalPnl += position.unrealizedPnl + position.realizedPnl;

      if (Math.abs(position.marketValue) > largestValue) {
        largestValue = Math.abs(position.marketValue);
        largestPosition = symbol;
      }
    }

    return {
      totalValue: this.state.accountBalance.equity,
      totalPnl,
      positionCount: this.state.positions.size,
      largestPosition
    };
  }

  private updateAccountBalance(): void {
    let totalMarketValue = 0;
    for (const position of this.state.positions.values()) {
      totalMarketValue += position.marketValue;
    }

    this.state.accountBalance = {
      cash: this.state.cash,
      equity: this.state.cash + totalMarketValue,
      buyingPower: this.state.cash,
      currency: 'USD'
    };
  }

  getState(): PortfolioState {
    return this.state;
  }
}

// Usage
const riskConfig: RiskConfig = {
  confidenceLevel: 0.95,
  lookbackPeriods: 252,
  method: 'historical'
};

const portfolio = new Portfolio(100000, riskConfig);

// Add positions
portfolio.addPosition('AAPL', 100, 175.50);
portfolio.addPosition('NVDA', 50, 495.80);
portfolio.addPosition('MSFT', 75, 380.20);

// Update prices
portfolio.updatePosition('AAPL', 178.30);
portfolio.updatePosition('NVDA', 502.10);
portfolio.updatePosition('MSFT', 378.90);

// Get metrics
const metrics = portfolio.getMetrics();
console.log('Portfolio Value:', metrics.totalValue);
console.log('Total P&L:', metrics.totalPnl);
console.log('Positions:', metrics.positionCount);
console.log('Largest Position:', metrics.largestPosition);

// Process signal
const signal: Signal = {
  symbol: 'TSLA',
  timestamp: new Date().toISOString(),
  type: 'BUY',
  strength: 0.85,
  price: 245.80,
  reason: 'Neural breakout signal'
};

const order = portfolio.signalToOrder(signal, 50);
if (order) {
  console.log('Generated order:', order);
}
```

---

### Tutorial 5: Complete Trading System

**Goal:** Build a complete end-to-end trading system with all core types.

**Advanced Example:**

```typescript
import {
  Bar,
  Quote,
  Signal,
  Position,
  OrderRequest,
  OrderResponse,
  BacktestConfig,
  BacktestResult,
  ModelConfig,
  PredictionResult,
  RiskConfig,
  VaRResult,
  KellyResult,
  PositionSize
} from '@neural-trader/core';

interface TradingSystem {
  // Market data
  bars: Map<string, Bar[]>;
  quotes: Map<string, Quote>;

  // Models and predictions
  modelConfig: ModelConfig;
  predictions: Map<string, PredictionResult>;

  // Risk management
  riskConfig: RiskConfig;
  var: VaRResult | null;
  kelly: KellyResult | null;

  // Portfolio
  positions: Map<string, Position>;
  orders: OrderRequest[];
  fills: OrderResponse[];

  // Performance
  backtestConfig: BacktestConfig;
  results: BacktestResult | null;
}

class CompleteTradingSystem implements TradingSystem {
  bars = new Map<string, Bar[]>();
  quotes = new Map<string, Quote>();
  modelConfig: ModelConfig;
  predictions = new Map<string, PredictionResult>();
  riskConfig: RiskConfig;
  var: VaRResult | null = null;
  kelly: KellyResult | null = null;
  positions = new Map<string, Position>();
  orders: OrderRequest[] = [];
  fills: OrderResponse[] = [];
  backtestConfig: BacktestConfig;
  results: BacktestResult | null = null;

  constructor(
    modelConfig: ModelConfig,
    riskConfig: RiskConfig,
    backtestConfig: BacktestConfig
  ) {
    this.modelConfig = modelConfig;
    this.riskConfig = riskConfig;
    this.backtestConfig = backtestConfig;
  }

  // Process market data
  addBar(bar: Bar): void {
    if (!this.bars.has(bar.symbol)) {
      this.bars.set(bar.symbol, []);
    }
    this.bars.get(bar.symbol)!.push(bar);
  }

  updateQuote(quote: Quote): void {
    this.quotes.set(quote.symbol, quote);
  }

  // Generate predictions
  predict(symbol: string): PredictionResult | null {
    const bars = this.bars.get(symbol);
    if (!bars || bars.length < this.modelConfig.inputSize) {
      return null;
    }

    // Neural forecast would happen here
    const lastClose = bars[bars.length - 1].close;
    const predictions = Array.from(
      { length: this.modelConfig.horizon },
      (_, i) => lastClose * (1 + 0.002 * i)
    );

    const prediction: PredictionResult = {
      predictions,
      lowerBound: predictions.map(p => p * 0.98),
      upperBound: predictions.map(p => p * 1.02),
      timestamp: new Date().toISOString()
    };

    this.predictions.set(symbol, prediction);
    return prediction;
  }

  // Generate trading signals
  generateSignal(symbol: string): Signal | null {
    const prediction = this.predictions.get(symbol);
    const bars = this.bars.get(symbol);

    if (!prediction || !bars || bars.length === 0) {
      return null;
    }

    const currentPrice = bars[bars.length - 1].close;
    const forecastPrice = prediction.predictions[0];
    const priceChange = (forecastPrice - currentPrice) / currentPrice;

    return {
      symbol,
      timestamp: new Date().toISOString(),
      type: priceChange > 0.01 ? 'BUY' :
            priceChange < -0.01 ? 'SELL' : 'HOLD',
      strength: Math.abs(priceChange) * 10,
      price: currentPrice,
      reason: `Neural forecast: ${forecastPrice.toFixed(2)} vs current ${currentPrice.toFixed(2)}`
    };
  }

  // Execute complete trading cycle
  async executeTradingCycle(symbols: string[]): Promise<void> {
    for (const symbol of symbols) {
      // 1. Generate prediction
      const prediction = this.predict(symbol);
      if (!prediction) continue;

      // 2. Generate signal
      const signal = this.generateSignal(symbol);
      if (!signal || signal.type === 'HOLD') continue;

      // 3. Calculate position size (Kelly Criterion)
      if (!this.kelly) {
        this.kelly = {
          kellyFraction: 0.125,
          halfKelly: 0.0625,
          quarterKelly: 0.03125,
          winRate: 0.58,
          avgWin: 250,
          avgLoss: -150
        };
      }

      const positionSize: PositionSize = {
        shares: Math.floor((100000 * this.kelly.halfKelly) / signal.price),
        dollarAmount: 0,
        percentageOfPortfolio: this.kelly.halfKelly,
        maxLoss: -100000 * this.kelly.halfKelly * 0.02,
        reasoning: 'Half-Kelly sizing'
      };
      positionSize.dollarAmount = positionSize.shares * signal.price;

      // 4. Create order
      const order: OrderRequest = {
        symbol,
        side: signal.type,
        orderType: 'LIMIT',
        quantity: positionSize.shares,
        limitPrice: signal.price,
        timeInForce: 'GTC'
      };

      this.orders.push(order);
      console.log(`Order placed: ${order.side} ${order.quantity} ${order.symbol} @ $${order.limitPrice}`);
    }
  }

  // Get system status
  getStatus(): string {
    return `
Trading System Status:
- Symbols tracked: ${this.bars.size}
- Active predictions: ${this.predictions.size}
- Open positions: ${this.positions.size}
- Orders placed: ${this.orders.length}
- Orders filled: ${this.fills.length}
- Portfolio Value: ${this.backtestConfig.initialCapital}
    `.trim();
  }
}

// Complete usage
const system = new CompleteTradingSystem(
  {
    modelType: 'NHITS',
    inputSize: 60,
    horizon: 10,
    hiddenSize: 256,
    numLayers: 4,
    dropout: 0.2,
    learningRate: 0.001
  },
  {
    confidenceLevel: 0.95,
    lookbackPeriods: 252,
    method: 'historical'
  },
  {
    initialCapital: 100000,
    startDate: '2024-01-01',
    endDate: '2024-12-31',
    commission: 0.001,
    slippage: 0.0005,
    useMarkToMarket: true
  }
);

// Add market data
const symbols = ['AAPL', 'NVDA', 'MSFT', 'TSLA'];
symbols.forEach(symbol => {
  for (let i = 0; i < 100; i++) {
    system.addBar({
      symbol,
      timestamp: new Date(2024, 0, i + 1).toISOString(),
      open: 175 + Math.random() * 10,
      high: 177 + Math.random() * 10,
      low: 174 + Math.random() * 10,
      close: 176 + Math.random() * 10,
      volume: 1000000 + Math.random() * 500000
    });
  }
});

// Run trading cycle
await system.executeTradingCycle(symbols);
console.log(system.getStatus());
```

## Integration Examples

### Integration with @neural-trader/backtesting

The backtesting package uses core types for configuration and results.

```typescript
import { BacktestConfig, BacktestResult, Trade } from '@neural-trader/core';
// @neural-trader/backtesting would be imported here

async function runBacktestExample() {
  const config: BacktestConfig = {
    initialCapital: 100000,
    startDate: '2023-01-01',
    endDate: '2023-12-31',
    commission: 0.001,
    slippage: 0.0005,
    useMarkToMarket: true
  };

  // Backtesting engine uses core types
  // const engine = new BacktestEngine(config);
  // const result: BacktestResult = await engine.run(strategy);

  // All types are from @neural-trader/core
  console.log('Backtest Config:', config);
}
```

### Integration with @neural-trader/neural

The neural package uses core model and prediction types.

```typescript
import {
  ModelConfig,
  TrainingConfig,
  PredictionResult,
  Bar
} from '@neural-trader/core';
// @neural-trader/neural would be imported here

async function neuralForecastExample() {
  const modelConfig: ModelConfig = {
    modelType: 'NHITS',
    inputSize: 60,
    horizon: 10,
    hiddenSize: 256,
    numLayers: 4,
    dropout: 0.2,
    learningRate: 0.001
  };

  const trainingConfig: TrainingConfig = {
    epochs: 100,
    batchSize: 32,
    validationSplit: 0.2,
    earlyStoppingPatience: 15,
    useGpu: true
  };

  // Neural model uses core types
  // const model = new NeuralForecaster(modelConfig);
  // await model.train(historicalData, trainingConfig);
  // const prediction: PredictionResult = await model.predict(recentData);

  console.log('Model Config:', modelConfig);
}
```

### Integration with @neural-trader/risk

Risk management uses core risk calculation types.

```typescript
import {
  RiskConfig,
  VaRResult,
  CVaRResult,
  KellyResult,
  DrawdownMetrics,
  Position
} from '@neural-trader/core';
// @neural-trader/risk would be imported here

function riskManagementExample() {
  const riskConfig: RiskConfig = {
    confidenceLevel: 0.95,
    lookbackPeriods: 252,
    method: 'historical'
  };

  const positions: Position[] = [
    {
      symbol: 'AAPL',
      quantity: 100,
      avgEntryPrice: 175.50,
      currentPrice: 178.30,
      unrealizedPnl: 280,
      realizedPnl: 0,
      marketValue: 17830
    }
  ];

  // Risk engine uses core types
  // const riskEngine = new RiskEngine(riskConfig);
  // const var95: VaRResult = riskEngine.calculateVaR(returns);
  // const cvar: CVaRResult = riskEngine.calculateCVaR(returns);
  // const kelly: KellyResult = riskEngine.calculateKelly(trades);

  console.log('Risk Config:', riskConfig);
  console.log('Positions:', positions);
}
```

### Full Platform Integration

Combining multiple packages with core types as the foundation:

```typescript
import {
  // Market data
  Bar,
  Quote,

  // Trading
  Signal,
  Position,
  OrderRequest,

  // Neural
  ModelConfig,
  PredictionResult,

  // Risk
  RiskConfig,
  KellyResult,

  // Backtest
  BacktestConfig,
  BacktestResult
} from '@neural-trader/core';

// Other packages would be imported:
// import { BacktestEngine } from '@neural-trader/backtesting';
// import { NeuralForecaster } from '@neural-trader/neural';
// import { RiskManager } from '@neural-trader/risk';
// import { Strategy } from '@neural-trader/strategies';

async function fullPlatformExample() {
  // 1. Configure neural model
  const modelConfig: ModelConfig = {
    modelType: 'NHITS',
    inputSize: 60,
    horizon: 10,
    hiddenSize: 256,
    numLayers: 4,
    dropout: 0.2,
    learningRate: 0.001
  };

  // 2. Configure risk management
  const riskConfig: RiskConfig = {
    confidenceLevel: 0.95,
    lookbackPeriods: 252,
    method: 'historical'
  };

  // 3. Configure backtest
  const backtestConfig: BacktestConfig = {
    initialCapital: 100000,
    startDate: '2024-01-01',
    endDate: '2024-12-31',
    commission: 0.001,
    slippage: 0.0005,
    useMarkToMarket: true
  };

  // 4. Initialize components (would use real packages)
  // const forecaster = new NeuralForecaster(modelConfig);
  // const riskManager = new RiskManager(riskConfig);
  // const backtester = new BacktestEngine(backtestConfig);

  // 5. Run complete workflow
  console.log('Complete trading system configured');
  console.log('Model:', modelConfig);
  console.log('Risk:', riskConfig);
  console.log('Backtest:', backtestConfig);

  // All types flow seamlessly between packages
}

fullPlatformExample();
```

## Configuration Options

### TypeScript Configuration

This package requires strict TypeScript settings for maximum type safety:

```json
{
  "compilerOptions": {
    "strict": true,
    "target": "ES2020",
    "module": "commonjs",
    "declaration": true,
    "declarationMap": true,
    "sourceMap": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true,
    "resolveJsonModule": true
  }
}
```

### Import Configuration

Multiple ways to import types:

```typescript
// Named imports (recommended)
import { Bar, Signal, Position } from '@neural-trader/core';

// Namespace import
import * as NeuralTrader from '@neural-trader/core';
type Bar = NeuralTrader.Bar;

// Selective imports
import type { BacktestConfig, BacktestResult } from '@neural-trader/core';
```

### Type Augmentation

Extend core types in your application:

```typescript
import { Signal } from '@neural-trader/core';

// Extend Signal with custom properties
interface CustomSignal extends Signal {
  strategyId: string;
  confidence: number;
  metadata: Record<string, any>;
}

// Use extended type
const signal: CustomSignal = {
  symbol: 'AAPL',
  timestamp: new Date().toISOString(),
  type: 'BUY',
  strength: 0.85,
  price: 175.50,
  reason: 'Neural signal',
  strategyId: 'momentum-v2',
  confidence: 0.92,
  metadata: { source: 'neural-network' }
};
```

## Performance Tips

### 1. Zero Runtime Cost

Core types have absolutely zero runtime overhead - they're compiled away.

**Performance Gain:** No runtime cost, only compile-time type checking

### 2. Tree Shaking

Import only what you need for optimal bundle size.

**❌ Inefficient:**
```typescript
import * as Core from '@neural-trader/core';
const config: Core.BacktestConfig = { /* ... */ };
```

**✅ Optimized:**
```typescript
import { BacktestConfig } from '@neural-trader/core';
const config: BacktestConfig = { /* ... */ };
```

**Performance Gain:** Smaller bundle size in applications

### 3. Type Inference

Let TypeScript infer types when possible to reduce code verbosity.

```typescript
import { Bar } from '@neural-trader/core';

// Explicit type annotation
const bars1: Bar[] = [
  { symbol: 'AAPL', timestamp: '2024-01-15', open: 175, high: 176, low: 174, close: 175.5, volume: 1000000 }
];

// Type inference (cleaner)
const bars2 = [
  { symbol: 'AAPL', timestamp: '2024-01-15', open: 175, high: 176, low: 174, close: 175.5, volume: 1000000 }
] satisfies Bar[];
```

### 4. Const Assertions

Use `as const` for literal types and readonly arrays.

```typescript
const config = {
  confidenceLevel: 0.95,
  lookbackPeriods: 252,
  method: 'historical'
} as const satisfies Partial<RiskConfig>;

// Type is { readonly confidenceLevel: 0.95, ... }
```

### 5. Utility Types

Leverage TypeScript utility types with core types.

**Best Practices:**
- Use `Partial<T>` for optional configuration updates
- Use `Required<T>` to ensure all properties are set
- Use `Pick<T, K>` to select specific properties
- Use `Omit<T, K>` to exclude properties
- Use `ReadonlyArray<T>` for immutable arrays

```typescript
import { BacktestConfig, Signal } from '@neural-trader/core';

// Partial configuration
type PartialConfig = Partial<BacktestConfig>;
const updates: PartialConfig = { commission: 0.002 };

// Pick specific fields
type SignalSummary = Pick<Signal, 'symbol' | 'type' | 'price'>;
const summary: SignalSummary = { symbol: 'AAPL', type: 'BUY', price: 175.50 };

// Omit timestamp for internal use
type SignalWithoutTimestamp = Omit<Signal, 'timestamp'>;
```

## Troubleshooting

### Common Issue 1: Type Mismatch Errors

**Problem:** TypeScript complains about type mismatches between packages.

**Symptoms:**
```
Type 'Bar' is not assignable to type 'Bar'.
  Types have separate declarations of a private property 'symbol'.
```

**Cause:** Multiple versions of @neural-trader/core installed.

**Solution:**

```bash
# Check for duplicate installations
npm ls @neural-trader/core

# Remove node_modules and reinstall
rm -rf node_modules package-lock.json
npm install

# Or use npm dedupe
npm dedupe @neural-trader/core
```

**Prevention:** Pin @neural-trader/core version in package.json dependencies.

---

### Common Issue 2: Missing Type Definitions

**Problem:** Types not found or not recognized by IDE.

**Error:**
```
Cannot find module '@neural-trader/core' or its corresponding type declarations.
```

**Solution:**

**Step 1:** Verify installation

```bash
npm list @neural-trader/core
```

**Step 2:** Check TypeScript configuration

```json
{
  "compilerOptions": {
    "types": ["@neural-trader/core"],
    "typeRoots": ["./node_modules/@types"]
  }
}
```

**Step 3:** Restart TypeScript server in your IDE

- VS Code: Cmd/Ctrl + Shift + P → "TypeScript: Restart TS Server"
- WebStorm: File → Invalidate Caches / Restart

---

### Common Issue 3: Import Path Resolution

**Problem:** Cannot resolve import paths.

**Quick Fix:**
```typescript
// Use exact package name
import { Bar } from '@neural-trader/core';

// Not relative paths
// import { Bar } from '@neural-trader/core/dist/index';
```

---

### Common Issue 4: Strict Mode Errors

**Problem:** Strict TypeScript mode causes type errors in existing code.

**Diagnosis:**

1. Check if strict mode is enabled in tsconfig.json
2. Review error messages for null/undefined issues
3. Check for missing required properties

**Solution:** Add proper type guards and null checks

```typescript
import { Position } from '@neural-trader/core';

function getPositionValue(position: Position | undefined): number {
  // Type guard
  if (!position) return 0;

  return position.marketValue;
}
```

---

### Common Issue 5: Version Conflicts

**Problem:** Incompatible versions between @neural-trader packages.

**Solution:** Ensure all @neural-trader packages use compatible versions

```json
{
  "dependencies": {
    "@neural-trader/core": "^1.0.0",
    "@neural-trader/backtesting": "^1.0.0",
    "@neural-trader/neural": "^1.0.0",
    "@neural-trader/risk": "^1.0.0"
  }
}
```

---

### Debugging Tips

**Check Type Definitions:**

```typescript
import type { Bar } from '@neural-trader/core';

// Hover over 'Bar' in IDE to see full definition
const bar: Bar = {
  symbol: 'AAPL',
  timestamp: '2024-01-15',
  open: 175,
  high: 176,
  low: 174,
  close: 175.5,
  volume: 1000000
};
```

**Verify Package Contents:**

```bash
# Check package files
npm pack @neural-trader/core --dry-run

# View type definitions
cat node_modules/@neural-trader/core/dist/index.d.ts
```

### Getting Help

If you encounter issues not covered here:

1. Check the [main documentation](../../README.md)
2. Search [GitHub Issues](https://github.com/ruvnet/neural-trader/issues)
3. Join our [Discord community](https://discord.gg/neural-trader)
4. Email support at support@neural-trader.io

## Related Packages

### Core Packages

- **[@neural-trader/core](https://www.npmjs.com/package/@neural-trader/core)** - Core types and interfaces (this package)
- **[@neural-trader/mcp-protocol](https://www.npmjs.com/package/@neural-trader/mcp-protocol)** - JSON-RPC 2.0 protocol types
- **[@neural-trader/mcp](https://www.npmjs.com/package/@neural-trader/mcp)** - MCP server with 102+ AI tools

### Recommended Combinations

**For Neural Trading:**
```bash
npm install @neural-trader/core @neural-trader/neural @neural-trader/strategies
```

**For Risk Management:**
```bash
npm install @neural-trader/core @neural-trader/risk @neural-trader/portfolio
```

**For Backtesting:**
```bash
npm install @neural-trader/core @neural-trader/backtesting @neural-trader/data
```

### Complementary Packages

- **[@neural-trader/backtesting](https://www.npmjs.com/package/@neural-trader/backtesting)** - High-performance backtesting engine
- **[@neural-trader/neural](https://www.npmjs.com/package/@neural-trader/neural)** - Neural forecasting models (NHITS, LSTM, Transformer)
- **[@neural-trader/risk](https://www.npmjs.com/package/@neural-trader/risk)** - VaR, CVaR, Kelly Criterion, drawdown analysis
- **[@neural-trader/strategies](https://www.npmjs.com/package/@neural-trader/strategies)** - Pre-built trading strategies
- **[@neural-trader/data](https://www.npmjs.com/package/@neural-trader/data)** - Market data providers and feeds
- **[@neural-trader/execution](https://www.npmjs.com/package/@neural-trader/execution)** - Order execution and broker integration

### Full Package List

See [packages/README.md](../README.md) for all 18 available packages.

## Contributing

Contributions are welcome! We follow the main Neural Trader contribution guidelines.

### Development Setup

```bash
# Clone repository
git clone https://github.com/ruvnet/neural-trader.git
cd neural-trader/neural-trader-rust

# Install dependencies
npm install

# Build core types
cd packages/core
npm run build

# Run type checking
npm run typecheck
```

### Adding New Types

When adding new types to @neural-trader/core:

1. **Edit source files** in `src/`
2. **Add JSDoc comments** with descriptions and examples
3. **Run type checker**: `npm run typecheck`
4. **Build package**: `npm run build`
5. **Update README** with new type documentation
6. **Add examples** showing usage

### Code Style

This package follows strict TypeScript standards:

- **TypeScript**: Strict mode enabled, ES2020 target
- **Formatting**: Prettier with 2-space indentation
- **Documentation**: JSDoc comments for every exported type
- **Naming**: PascalCase for types, camelCase for properties

```bash
# Format code
npm run format

# Type check
npm run typecheck
```

### Contribution Workflow

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feat/new-types`
3. **Add** new types with documentation
4. **Run** type checking: `npm run typecheck`
5. **Commit** changes: `git commit -m 'feat(core): add new trading types'`
6. **Push** to branch: `git push origin feat/new-types`
7. **Open** a Pull Request

### Pull Request Guidelines

- Add JSDoc comments for all new types
- Update README with examples
- Ensure backward compatibility
- Follow existing naming conventions
- Update CHANGELOG.md if applicable

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for detailed guidelines.

## License

This project is dual-licensed under **MIT OR Apache-2.0**.

You may choose either license at your option:

- **MIT License**: See [LICENSE-MIT](../../LICENSE-MIT)
- **Apache License 2.0**: See [LICENSE-APACHE](../../LICENSE-APACHE)

### Third-Party Licenses

This package has **zero dependencies** and includes no third-party code.

## Support

### Documentation

- **Package Docs**: This README
- **Type Reference**: [index.d.ts](./dist/index.d.ts)
- **Examples**: See tutorials above
- **Main Docs**: https://neural-trader.ruv.io

### Community

- **GitHub Issues**: https://github.com/ruvnet/neural-trader/issues
- **Discord**: https://discord.gg/neural-trader
- **Twitter**: [@neural_trader](https://twitter.com/neural_trader)

### Professional Support

For enterprise support, training, or custom development:

- **Email**: support@neural-trader.io
- **Website**: https://neural-trader.io/support

## Changelog

See [CHANGELOG.md](./CHANGELOG.md) for version history and release notes.

## Security

For security issues, please see [SECURITY.md](../../SECURITY.md) for our security policy and reporting process.

**Do not report security vulnerabilities through public GitHub issues.**

## Acknowledgments

- Built with [TypeScript](https://www.typescriptlang.org/) for type safety
- Inspired by [QuantConnect](https://www.quantconnect.com/) and [Backtrader](https://www.backtrader.com/)
- Neural Trader community contributors
- Zero dependencies by design

---

**Disclaimer**: This software is for educational and research purposes only. Trading financial instruments carries risk. Past performance does not guarantee future results. Use at your own risk.
