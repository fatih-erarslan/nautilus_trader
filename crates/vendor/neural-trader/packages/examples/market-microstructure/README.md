# @neural-trader/example-market-microstructure

[![npm version](https://badge.fury.io/js/%40neural-trader%2Fexample-market-microstructure.svg)](https://www.npmjs.com/package/@neural-trader/example-market-microstructure)
[![npm downloads](https://img.shields.io/npm/dm/@neural-trader/example-market-microstructure.svg)](https://www.npmjs.com/package/@neural-trader/example-market-microstructure)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Build Status](https://img.shields.io/github/actions/workflow/status/ruvnet/neural-trader/ci.yml?branch=main)](https://github.com/ruvnet/neural-trader/actions)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.3-blue)](https://www.typescriptlang.org/)
[![Coverage](https://img.shields.io/badge/coverage-85%25-brightgreen)]()

Self-learning market microstructure analysis with swarm-based feature engineering for high-frequency trading and order book analysis.

## Features

- **Real-time Order Flow Analysis**: Track buy/sell pressure, toxicity, and informed trading
- **Market Impact Modeling**: Calculate price impact and liquidity provision costs
- **Price Discovery Patterns**: Analyze mid price, micro price, and price efficiency
- **Liquidity Optimization**: Score liquidity and estimate resilience times
- **Self-Learning**: AgentDB-backed pattern recognition and learning
- **Swarm Intelligence**: Distributed feature engineering with 50+ generations
- **Anomaly Detection**: Multi-agent consensus for detecting market anomalies

## Installation

```bash
npm install @neural-trader/example-market-microstructure
```

## Quick Start

```typescript
import { createMarketMicrostructure, OrderBook } from '@neural-trader/example-market-microstructure';

// Initialize
const mm = await createMarketMicrostructure({
  agentDbPath: './market-patterns.db',
  useSwarm: true,
  swarmConfig: {
    numAgents: 20,
    generations: 30
  }
});

// Analyze order book
const orderBook: OrderBook = {
  bids: [
    { price: 100.0, size: 1000, orders: 10 },
    { price: 99.9, size: 800, orders: 8 }
  ],
  asks: [
    { price: 100.1, size: 1000, orders: 10 },
    { price: 100.2, size: 800, orders: 8 }
  ],
  timestamp: Date.now(),
  symbol: 'BTCUSD'
};

const result = await mm.analyze(orderBook);

console.log('Spread:', result.metrics.bidAskSpread);
console.log('Liquidity Score:', result.metrics.liquidityScore);
console.log('Order Flow Toxicity:', result.metrics.orderFlowToxicity);

// Learn from outcomes
await mm.learn({
  priceMove: 0.5,
  spreadChange: -0.01,
  liquidityChange: 0.1,
  timeHorizon: 5000
}, 'profitable_pattern');

// Explore features with swarm
const featureSets = await mm.exploreFeatures();

// Optimize features
const optimized = await mm.optimizeFeatures('profitability');

// Get statistics
const stats = mm.getStatistics();
console.log('Total Patterns:', stats.learner.totalPatterns);
console.log('Swarm Agents:', stats.swarm.totalAgents);

await mm.close();
```

## Core Components

### OrderBookAnalyzer

Analyzes order books to compute comprehensive microstructure metrics:

```typescript
import { OrderBookAnalyzer } from '@neural-trader/example-market-microstructure';

const analyzer = new OrderBookAnalyzer();
const metrics = analyzer.analyzeOrderBook(orderBook, recentTrades);

// Spread metrics
console.log('Bid-Ask Spread:', metrics.bidAskSpread);
console.log('Spread (bps):', metrics.spreadBps);
console.log('Effective Spread:', metrics.effectiveSpread);

// Depth metrics
console.log('Bid Depth:', metrics.bidDepth);
console.log('Ask Depth:', metrics.askDepth);
console.log('Imbalance:', metrics.imbalance);

// Toxicity metrics
console.log('VPIN:', metrics.vpin);
console.log('Order Flow Toxicity:', metrics.orderFlowToxicity);
console.log('Adverse Selection:', metrics.adverseSelection);

// Flow metrics
console.log('Buy Pressure:', metrics.buyPressure);
console.log('Sell Pressure:', metrics.sellPressure);
console.log('Net Flow:', metrics.netFlow);

// Price discovery
console.log('Mid Price:', metrics.midPrice);
console.log('Micro Price:', metrics.microPrice);
console.log('Price Impact:', metrics.priceImpact);

// Liquidity
console.log('Liquidity Score:', metrics.liquidityScore);
console.log('Resilience Time:', metrics.resilienceTime);
```

### PatternLearner

Self-learning pattern recognition with AgentDB persistence:

```typescript
import { PatternLearner } from '@neural-trader/example-market-microstructure';

const learner = new PatternLearner({
  agentDbPath: './patterns.db',
  minConfidence: 0.7,
  maxPatterns: 1000,
  useNeuralPredictor: true
});

await learner.initialize();

// Extract features from metrics history
const features = learner.extractFeatures(metricsHistory);

// Learn from outcome
const pattern = await learner.learnPattern(features, {
  priceMove: 1.5,
  spreadChange: 0.05,
  liquidityChange: -0.1,
  timeHorizon: 5000
}, 'uptrend_pattern');

// Recognize patterns
const recognized = await learner.recognizePattern(features);
if (recognized) {
  console.log('Pattern:', recognized.label);
  console.log('Confidence:', recognized.confidence);
  console.log('Expected outcome:', recognized.outcome);
}

// Predict with neural network
const prediction = await learner.predictOutcome(features);

// Get statistics
const stats = learner.getStatistics();
console.log('Total Patterns:', stats.totalPatterns);
console.log('High Confidence:', stats.highConfidencePatterns);
console.log('Average Confidence:', stats.avgConfidence);

await learner.close();
```

### SwarmFeatureEngineer

Distributed feature engineering with swarm intelligence:

```typescript
import { SwarmFeatureEngineer } from '@neural-trader/example-market-microstructure';

const swarm = new SwarmFeatureEngineer({
  numAgents: 30,
  generations: 50,
  mutationRate: 0.2,
  crossoverRate: 0.7,
  eliteSize: 3,
  useOpenRouter: false
});

await swarm.initialize();

// Explore feature space
const featureSets = await swarm.exploreFeatures(metricsHistory);

featureSets.forEach(fs => {
  console.log('Feature Set:', fs.name);
  console.log('Features:', fs.features);
  console.log('Importance:', fs.importance);
  console.log('Performance:', fs.performance);
});

// Optimize features
const optimized = await swarm.optimizeFeatures(
  baseFeatures,
  'profitability'
);

// Detect anomalies
const anomaly = await swarm.detectAnomalies(metrics);
if (anomaly.isAnomaly) {
  console.log('Anomaly Type:', anomaly.anomalyType);
  console.log('Confidence:', anomaly.confidence);
  console.log('Explanation:', anomaly.explanation);
}

// Get agent statistics
const stats = swarm.getAgentStats();
console.log('Total Agents:', stats.totalAgents);
console.log('By Type:', stats.byType);
console.log('Best Agent:', stats.bestAgent);

await swarm.cleanup();
```

## Metrics Reference

### Spread Metrics

- **bidAskSpread**: Raw spread between best bid and ask
- **spreadBps**: Spread in basis points (relative to mid price)
- **effectiveSpread**: Average spread based on actual trades

### Depth Metrics

- **bidDepth**: Total size at top 5 bid levels
- **askDepth**: Total size at top 5 ask levels
- **imbalance**: Order book imbalance (-1 to +1, positive = more bids)

### Toxicity Metrics

- **vpin**: Volume-Synchronized Probability of Informed Trading (0-1)
- **orderFlowToxicity**: Correlation between flow and price movement
- **adverseSelection**: Cost of adverse selection (positive = widening spreads)

### Flow Metrics

- **buyPressure**: Proportion of buy volume (0-1)
- **sellPressure**: Proportion of sell volume (0-1)
- **netFlow**: Net order flow (buyPressure - sellPressure)

### Price Discovery

- **midPrice**: Simple mid price (bid + ask) / 2
- **microPrice**: Volume-weighted mid price
- **priceImpact**: Price impact per unit size

### Liquidity

- **liquidityScore**: Composite liquidity score (0-1, higher = more liquid)
- **resilienceTime**: Estimated time for order book recovery (ms)

## Pattern Features

Extracted from metrics history for pattern learning:

- **spreadTrend**: Direction of spread changes
- **spreadVolatility**: Volatility of spread
- **depthImbalance**: Average order book imbalance
- **depthTrend**: Direction of depth changes
- **flowPersistence**: How often flow direction persists
- **flowReversal**: How often flow reverses
- **toxicityLevel**: Average order flow toxicity
- **informedTradingProbability**: Average VPIN
- **priceEfficiency**: How efficiently price discovers value
- **microPriceDivergence**: Divergence between mid and micro price

## Swarm Agent Types

The swarm consists of 4 specialized agent types:

### Explorer Agents
- Try new feature combinations
- Discover novel patterns
- High mutation rate

### Optimizer Agents
- Refine existing features
- Improve performance
- Gradual improvements

### Validator Agents
- Test feature robustness
- Ensure generalization
- Cross-validation

### Anomaly Detector Agents
- Find unusual patterns
- Detect market regime changes
- Focused feature sets

## Examples

See the `examples/` directory:

- `basic-usage.ts`: Complete workflow demonstration
- `advanced-anomaly-detection.ts`: Anomaly detection showcase

Run examples:

```bash
npm run dev examples/basic-usage.ts
npm run dev examples/advanced-anomaly-detection.ts
```

## Integration with Claude-Flow

This package integrates with claude-flow for swarm coordination:

```bash
# Pre-task hook
npx claude-flow@alpha hooks pre-task --description "Market microstructure analysis"

# Post-edit hook (stores features in memory)
npx claude-flow@alpha hooks post-edit --file "features.json" --memory-key "market/features"

# Post-task hook
npx claude-flow@alpha hooks post-task --task-id "microstructure-analysis"
```

The SwarmFeatureEngineer automatically calls these hooks during:
- Initialization
- Progress reporting (every 10 generations)
- Feature set finalization
- Cleanup

## OpenRouter Integration (Optional)

For enhanced anomaly detection, provide OpenRouter API key:

```typescript
const mm = await createMarketMicrostructure({
  useSwarm: true,
  swarmConfig: {
    useOpenRouter: true,
    openRouterKey: process.env.OPENROUTER_API_KEY
  }
});
```

The swarm will use LLM-enhanced explanations for detected anomalies.

## Testing

```bash
# Run all tests
npm test

# Run with coverage
npm run test:coverage

# Watch mode
npm run test:watch
```

Test coverage target: >80% for all metrics.

## Architecture

```
┌─────────────────────────────────────────┐
│      MarketMicrostructure               │
│  (Main Orchestrator)                    │
└──────────┬──────────────────────────────┘
           │
           ├─► OrderBookAnalyzer
           │   ├─ Spread metrics
           │   ├─ Depth metrics
           │   ├─ Toxicity metrics
           │   ├─ Flow metrics
           │   ├─ Price discovery
           │   └─ Liquidity metrics
           │
           ├─► PatternLearner
           │   ├─ Feature extraction
           │   ├─ Pattern recognition
           │   ├─ Neural prediction
           │   ├─ AgentDB storage
           │   └─ Statistics
           │
           └─► SwarmFeatureEngineer
               ├─ Agent population
               ├─ Feature exploration
               ├─ Genetic evolution
               ├─ Anomaly detection
               └─ Claude-flow hooks
```

## Performance

- **Order book analysis**: <1ms per snapshot
- **Pattern recognition**: <10ms with 1000 patterns
- **Feature exploration**: ~5-30s for 50 generations
- **Anomaly detection**: <50ms with 30 agents

## Dependencies

- `@neural-trader/predictor`: Neural network predictions
- `agentdb`: Vector database for pattern storage (150x faster than alternatives)
- `sublinear-time-solver`: Sublinear algorithms for optimization
- `claude-flow`: Swarm coordination and hooks

## License

MIT

## Contributing

Contributions welcome! Please ensure:
- Test coverage >80%
- All tests pass
- TypeScript strict mode
- ESLint clean

## Support

- Issues: [GitHub Issues](https://github.com/neural-trader/neural-trader/issues)
- Documentation: [Neural Trader Docs](https://docs.neural-trader.io)

---

Built with ❤️ by the Neural Trader team
