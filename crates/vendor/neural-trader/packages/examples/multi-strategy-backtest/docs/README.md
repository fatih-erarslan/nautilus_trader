# @neural-trader/example-multi-strategy-backtest

Self-learning multi-strategy backtesting system with reinforcement learning and swarm optimization.

## Features

- 🧠 **Reinforcement Learning**: Adaptive strategy allocation using Q-Learning
- 🐝 **Swarm Optimization**: Parameter tuning with Particle Swarm Optimization
- 📊 **Walk-Forward Analysis**: Out-of-sample validation with multiple periods
- 💰 **Transaction Cost Modeling**: Realistic commission and slippage simulation
- 🌊 **Regime Detection**: Adaptive strategy weighting based on market conditions
- 🤖 **AI Strategy Discovery**: OpenRouter integration for strategy suggestions
- ⚡ **NAPI-RS Acceleration**: Optional native module for 10-100x speedup
- 💾 **Memory Persistence**: AgentDB integration for cross-session learning

## Installation

```bash
npm install @neural-trader/example-multi-strategy-backtest
```

## Quick Start

```typescript
import { MultiStrategyBacktestSystem } from '@neural-trader/example-multi-strategy-backtest';

// Configure backtest
const config = {
  startDate: new Date('2023-01-01'),
  endDate: new Date('2024-01-01'),
  initialCapital: 100000,
  symbols: ['AAPL'],
  strategies: [
    {
      name: 'momentum',
      type: 'momentum',
      initialWeight: 0.5,
      parameters: { lookbackPeriod: 20 },
      enabled: true
    },
    {
      name: 'mean-reversion',
      type: 'mean-reversion',
      initialWeight: 0.5,
      parameters: { maPeriod: 20 },
      enabled: true
    }
  ],
  commission: 0.001,
  slippage: 0.0005,
  walkForwardPeriods: 3
};

// Initialize system
const system = new MultiStrategyBacktestSystem(config, process.env.OPENROUTER_API_KEY);
await system.initialize();

// Run complete workflow
const results = await system.runCompleteWorkflow(marketData);

console.log('Best Strategy:', results.performances[0]);
```

## Architecture

### Core Components

1. **Backtester** (`backtester.ts`)
   - Walk-forward optimization
   - Transaction cost modeling
   - Regime detection
   - Multi-timeframe analysis

2. **StrategyLearner** (`strategy-learner.ts`)
   - Q-Learning implementation
   - Experience replay
   - Epsilon-greedy exploration
   - AgentDB persistence

3. **SwarmOptimizer** (`swarm-optimizer.ts`)
   - Particle Swarm Optimization
   - Adaptive inertia
   - Parallel evaluation
   - Constraint handling

4. **Strategies**
   - Momentum (`strategies/momentum.ts`)
   - Mean Reversion (`strategies/mean-reversion.ts`)
   - Pairs Trading (`strategies/pairs-trading.ts`)
   - Market Making (`strategies/market-making.ts`)

### Workflow Phases

1. **Initial Backtest**: Baseline performance with default parameters
2. **Reinforcement Learning**: Learn optimal strategy allocation
3. **Swarm Optimization**: Tune strategy parameters
4. **Final Backtest**: Validate with optimized configuration
5. **Continuous Learning**: Update models with new results
6. **AI Discovery** (optional): Generate strategy suggestions with OpenRouter

## Configuration

### Backtest Config

```typescript
interface BacktestConfig {
  startDate: Date;
  endDate: Date;
  initialCapital: number;
  symbols: string[];
  strategies: StrategyConfig[];
  commission: number;
  slippage: number;
  walkForwardPeriods?: number;
  rebalanceFrequency?: 'daily' | 'weekly' | 'monthly';
}
```

### Strategy Config

```typescript
interface StrategyConfig {
  name: string;
  type: 'momentum' | 'mean-reversion' | 'pairs-trading' | 'market-making';
  initialWeight: number;
  parameters: Record<string, any>;
  enabled: boolean;
}
```

### Learner Config

```typescript
interface LearnerConfig {
  learningRate: number;           // Default: 0.1
  discountFactor: number;          // Default: 0.95
  explorationRate: number;         // Default: 1.0
  explorationDecay: number;        // Default: 0.995
  minExplorationRate: number;      // Default: 0.01
  experienceBufferSize: number;    // Default: 10000
  batchSize: number;               // Default: 32
  updateFrequency: number;         // Default: 100
}
```

### Swarm Config

```typescript
interface SwarmConfig {
  particleCount: number;           // Default: 30
  maxIterations: number;           // Default: 100
  inertia: number;                 // Default: 0.7
  cognitiveWeight: number;         // Default: 1.5
  socialWeight: number;            // Default: 1.5
  bounds: Record<string, [number, number]>;
}
```

## Performance Metrics

The system calculates comprehensive performance metrics:

- **Total Return**: Cumulative return over period
- **Sharpe Ratio**: Risk-adjusted return
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Ratio of gross profit to gross loss
- **Calmar Ratio**: Return divided by max drawdown
- **Sortino Ratio**: Return divided by downside deviation

## NAPI-RS Acceleration

For performance-critical operations, compile the native module:

```bash
cd packages/examples/multi-strategy-backtest
npm run native:build
```

The system automatically uses native implementations when available, falling back to pure JS otherwise.

### Performance Comparison

| Operation | Pure JS | NAPI-RS | Speedup |
|-----------|---------|---------|---------|
| SMA Calculation | 100ms | 2ms | 50x |
| Sharpe Ratio | 50ms | 1ms | 50x |
| Backtest Execution | 10s | 0.5s | 20x |

## Memory Bank Integration

The system uses AgentDB for persistent learning:

```typescript
// Learning state is automatically saved
await learner.initialize();  // Loads previous state
await learner.learnFromBacktest(states, performances);  // Updates and saves

// Access learning statistics
const stats = learner.getStats();
console.log(`Episodes: ${stats.episodes}`);
console.log(`Q-Table Size: ${stats.qTableSize}`);
```

## OpenRouter Integration

Enable AI strategy discovery with OpenRouter:

```typescript
const system = new MultiStrategyBacktestSystem(
  config,
  process.env.OPENROUTER_API_KEY  // Set your API key
);

// AI suggestions are automatically generated during workflow
const results = await system.runCompleteWorkflow(marketData);
```

## Examples

### Basic Backtest

```bash
npm run build
node dist/examples/basic-backtest.js
```

### Custom Strategy

```typescript
import { Strategy, MarketData, StrategySignal } from '@neural-trader/example-multi-strategy-backtest';

class CustomStrategy implements Strategy {
  name = 'custom';
  type = 'custom';

  generateSignal(data: MarketData[], parameters: Record<string, any>): StrategySignal {
    // Implement your strategy logic
    return {
      symbol: data[data.length - 1].symbol,
      action: 'buy',
      strength: 0.8,
      confidence: 0.7,
      strategy: this.name,
      timestamp: data[data.length - 1].timestamp
    };
  }

  updateParameters(parameters: Record<string, any>): void {
    // Update strategy parameters
  }

  getParameters(): Record<string, any> {
    return {};
  }
}
```

### Hooks Integration

```bash
# Pre-task hook
npx claude-flow@alpha hooks pre-task --description "Multi-strategy backtest"

# Post-edit hook
npx claude-flow@alpha hooks post-edit --file "results.json" --memory-key "backtest/results"

# Post-task hook
npx claude-flow@alpha hooks post-task --task-id "backtest-123"
```

## Testing

```bash
# Run all tests
npm test

# Run with coverage
npm run test:coverage

# Watch mode
npm run test:watch
```

## Benchmarks

```bash
# Run benchmarks
npm run benchmark
```

Expected performance on modern hardware:
- 1000 bars backtest: ~100ms (pure JS), ~5ms (NAPI-RS)
- Strategy optimization: ~2-5 minutes
- Complete workflow: ~5-10 minutes

## Best Practices

1. **Start Simple**: Begin with 2-3 strategies and short backtests
2. **Walk-Forward**: Always use out-of-sample validation
3. **Transaction Costs**: Include realistic commission and slippage
4. **Regime Awareness**: Let the learner adjust weights dynamically
5. **Parameter Bounds**: Set reasonable optimization bounds
6. **Persistence**: Let the learner accumulate experience across runs
7. **Native Module**: Compile NAPI-RS for production use

## Limitations

- Assumes perfect fills at signal prices (adjusted by slippage)
- Simplified pairs trading (single-asset mode)
- No short selling constraints (can be added)
- Market impact not modeled separately
- Assumes continuous trading (no gaps)

## Contributing

See the main neural-trader repository for contribution guidelines.

## License

MIT

## Related Packages

- `@neural-trader/predictor` - Neural network predictions
- `@neural-trader/core` - Core trading utilities
- `agentdb` - Vector database for persistent memory
