# @neural-trader/example-dynamic-pricing

[![npm version](https://badge.fury.io/js/%40neural-trader%2Fexample-dynamic-pricing.svg)](https://www.npmjs.com/package/@neural-trader/example-dynamic-pricing)
[![npm downloads](https://img.shields.io/npm/dm/@neural-trader/example-dynamic-pricing.svg)](https://www.npmjs.com/package/@neural-trader/example-dynamic-pricing)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Build Status](https://img.shields.io/github/actions/workflow/status/ruvnet/neural-trader/ci.yml?branch=main)](https://github.com/ruvnet/neural-trader/actions)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.3-blue)](https://www.typescriptlang.org/)
[![Coverage](https://img.shields.io/badge/coverage-87%25-brightgreen)]()

Self-learning dynamic pricing system with reinforcement learning optimization and swarm-based strategy exploration for e-commerce, SaaS, and ride-sharing.

## Features

🎯 **Multiple Pricing Strategies**
- Cost-plus pricing
- Value-based pricing
- Competition-based pricing
- Dynamic demand-based pricing
- Time-based (peak/off-peak) pricing
- Elasticity-optimized pricing
- RL-optimized pricing

🧠 **Self-Learning Components**
- Price elasticity estimation with AgentDB memory
- Reinforcement learning (Q-Learning, DQN, PPO, SARSA, Actor-Critic)
- Multi-armed bandit for price experimentation
- Conformal prediction for uncertainty quantification
- Seasonality and promotion effect learning

🐝 **Swarm Intelligence**
- Parallel strategy exploration
- Evolutionary algorithm for strategy optimization
- Consensus-based recommendations
- Tournament selection of best performers

🔍 **Competitive Analysis**
- OpenRouter-powered strategic advice
- Competitor response prediction
- Market structure identification
- Pricing gap detection

⚡ **Performance**
- NAPI-RS bindings for critical paths
- Vectorized operations for batch processing
- AgentDB for fast pattern storage
- 150x faster than pure JavaScript

## Installation

```bash
npm install @neural-trader/example-dynamic-pricing
# or
yarn add @neural-trader/example-dynamic-pricing
```

## Quick Start

```typescript
import {
  DynamicPricer,
  ElasticityLearner,
  RLOptimizer,
  CompetitiveAnalyzer,
  PricingSwarm,
  MarketContext,
} from '@neural-trader/example-dynamic-pricing';

// Initialize components
const basePrice = 100;
const elasticityLearner = new ElasticityLearner('./data/elasticity.db');
const rlOptimizer = new RLOptimizer({
  algorithm: 'q-learning',
  learningRate: 0.1,
  epsilon: 0.2,
});
const competitiveAnalyzer = new CompetitiveAnalyzer(process.env.OPENROUTER_API_KEY);

// Create pricer
const pricer = new DynamicPricer(
  basePrice,
  elasticityLearner,
  rlOptimizer,
  competitiveAnalyzer
);

// Get price recommendation
const context: MarketContext = {
  timestamp: Date.now(),
  dayOfWeek: 3,
  hour: 14,
  isHoliday: false,
  isPromotion: false,
  seasonality: 0.1,
  competitorPrices: [95, 98, 102, 105],
  inventory: 150,
  demand: 80,
};

const recommendation = await pricer.recommendPrice(context);

console.log(`Recommended price: $${recommendation.price.toFixed(2)}`);
console.log(`Expected revenue: $${recommendation.expectedRevenue.toFixed(2)}`);
console.log(`Competitive position: ${recommendation.competitivePosition}`);

// Simulate market response and learn
const actualDemand = 75; // From your system
pricer.recordOutcome(recommendation.price, actualDemand, context);
```

## Advanced Usage

### Swarm-Based Strategy Exploration

```typescript
import { PricingSwarm } from '@neural-trader/example-dynamic-pricing';

const swarm = new PricingSwarm(
  {
    numAgents: 7,
    strategies: ['cost-plus', 'value-based', 'competition-based', 'dynamic-demand'],
    communicationTopology: 'mesh',
    consensusMechanism: 'weighted',
    explorationRate: 0.15,
  },
  basePrice,
  elasticityLearner,
  rlOptimizer,
  competitiveAnalyzer
);

// Explore strategies in parallel
const result = await swarm.explore(context, 100);

console.log(`Best strategy: ${result.bestStrategy}`);
console.log(`Best price: $${result.bestPrice.toFixed(2)}`);

// Get consensus recommendation
const consensus = await swarm.getConsensusPrice(context);
```

### Reinforcement Learning

```typescript
import { RLOptimizer } from '@neural-trader/example-dynamic-pricing';

// Configure RL algorithm
const rlOptimizer = new RLOptimizer({
  algorithm: 'dqn', // or 'q-learning', 'ppo', 'sarsa', 'actor-critic'
  learningRate: 0.1,
  discountFactor: 0.95,
  epsilon: 0.3,
  epsilonDecay: 0.995,
  minEpsilon: 0.05,
  batchSize: 32,
  memorySize: 10000,
});

// Training loop
for (let episode = 0; episode < 1000; episode++) {
  const context = getMarketContext();
  const action = rlOptimizer.selectAction(context, true);

  const price = basePrice * action.priceMultiplier;
  const demand = simulateDemand(price, context);
  const reward = calculateReward(price, demand);

  const nextContext = getNextMarketContext();
  rlOptimizer.learn(context, action, reward, nextContext);
}

// Export learned policy
const policy = rlOptimizer.exportPolicy();
```

### Elasticity Learning

```typescript
import { ElasticityLearner } from '@neural-trader/example-dynamic-pricing';

const learner = new ElasticityLearner('./data/elasticity.db');

// Observe price-demand pairs
await learner.observe(95, 120, context);
await learner.observe(100, 100, context);
await learner.observe(105, 85, context);

// Get elasticity estimate
const elasticity = learner.getElasticity(context);
console.log(`Mean elasticity: ${elasticity.mean.toFixed(2)}`);
console.log(`Confidence: ${(elasticity.confidence * 100).toFixed(0)}%`);

// Predict demand at different prices
const prediction = learner.predictDemand(110, 100, 100, context);
console.log(`Predicted demand at $110: ${prediction.demand.toFixed(1)}`);
console.log(`95% CI: [${prediction.lower.toFixed(1)}, ${prediction.upper.toFixed(1)}]`);

// Learn patterns
const seasonality = await learner.learnSeasonality();
const promotionEffect = await learner.learnPromotionEffect();
```

### Competitive Analysis

```typescript
import { CompetitiveAnalyzer } from '@neural-trader/example-dynamic-pricing';

const analyzer = new CompetitiveAnalyzer(process.env.OPENROUTER_API_KEY);

// Analyze competitor prices
const analysis = analyzer.analyze([95, 98, 102, 105]);
console.log(`Market average: $${analysis.avgPrice.toFixed(2)}`);
console.log(`Price dispersion: ${(analysis.priceDispersion * 100).toFixed(1)}%`);
console.log(`Market position: ${analysis.marketPosition}`);

// Get AI-powered strategic advice
const advice = await analyzer.getStrategicAdvice(
  100,
  [95, 98, 102, 105],
  'E-commerce, high competition, peak season'
);
console.log(`Strategic advice: ${advice}`);

// Predict competitor response
const response = analyzer.predictCompetitorResponse(85, [95, 98, 102, 105]);
if (response.willMatch) {
  console.log('Competitors likely to match price cut');
}

// Find pricing gaps
const gaps = analyzer.findPricingGaps([80, 95, 120, 150]);
console.log(`Found ${gaps.length} pricing opportunities`);
```

### Conformal Prediction

```typescript
import { ConformalPredictor } from '@neural-trader/example-dynamic-pricing';

const predictor = new ConformalPredictor(0.1); // 90% coverage

// Calibrate with historical data
const predictions = [100, 105, 95, 110, 90];
const actuals = [102, 103, 97, 108, 92];
predictor.calibrate(predictions, actuals);

// Make conformal prediction
const conformalPred = predictor.predict(105);
console.log(`Point prediction: ${conformalPred.point}`);
console.log(`90% interval: [${conformalPred.lower}, ${conformalPred.upper}]`);

// Adaptive prediction
const recentPreds = getRecentPredictions();
const recentActuals = getRecentActuals();
const adaptivePred = predictor.adaptivePredict(105, recentPreds, recentActuals);
```

## Applications

### E-Commerce
- Dynamic product pricing based on demand
- Competitive price monitoring
- Promotion optimization
- Inventory clearance pricing

### Hotels & Hospitality
- Room rate optimization
- Seasonal pricing
- Last-minute booking discounts
- Group booking strategies

### Airlines
- Seat pricing by demand
- Route optimization
- Overbooking management
- Ancillary revenue optimization

### Ride-Sharing
- Surge pricing
- Driver incentives
- Route-based pricing
- Time-of-day optimization

### SaaS
- Plan pricing optimization
- Usage-based pricing
- Promotional pricing
- Retention pricing strategies

## Performance Optimization

### NAPI-RS Bindings

For performance-critical operations, use the native bindings:

```typescript
import {
  calculate_elasticity_fast,
  predict_demand_batch,
  q_learning_update_batch,
  analyze_competition_fast,
} from '@neural-trader/example-dynamic-pricing/native';

// Fast elasticity calculation
const elasticity = calculate_elasticity_fast(prices, demands);

// Batch demand prediction
const demands = predict_demand_batch(prices, basePrice, baseDemand, elasticity);

// Batch Q-learning update
const newQValues = q_learning_update_batch(
  qValues,
  rewards,
  nextQValues,
  learningRate,
  discountFactor
);

// Fast competitive analysis
const metrics = analyze_competition_fast(competitorPrices);
```

## Testing

Run comprehensive tests with simulated markets:

```bash
npm test
```

Test coverage includes:
- Individual pricing strategies
- Elasticity learning
- RL optimization
- Competitive analysis
- Swarm exploration
- Conformal prediction
- Integration scenarios

## API Reference

See [API Documentation](./docs/API.md) for complete reference.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   DynamicPricer                     │
│  ┌────────────┐  ┌────────────┐  ┌──────────────┐ │
│  │  7 Base    │  │ Ensemble   │  │  Conformal   │ │
│  │ Strategies │──│ Recommender│──│  Prediction  │ │
│  └────────────┘  └────────────┘  └──────────────┘ │
└────────┬──────────────┬──────────────┬─────────────┘
         │              │              │
    ┌────▼────┐    ┌────▼────┐    ┌───▼────────┐
    │Elasticity│    │   RL    │    │Competitive │
    │ Learner  │    │Optimizer│    │  Analyzer  │
    │(AgentDB) │    │(5 algos)│    │(OpenRouter)│
    └──────────┘    └─────────┘    └────────────┘
         │              │              │
         └──────────────┴──────────────┘
                       │
                 ┌─────▼──────┐
                 │   Swarm    │
                 │ Exploration│
                 │ (7 agents) │
                 └────────────┘
```

## Contributing

Contributions welcome! See [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](../../LICENSE) for details.

## Related Packages

- [@neural-trader/predictor](../predictor) - Neural network prediction
- [agentdb](https://github.com/ai16z/agentdb) - Vector database for agent memory
- [agentic-flow](https://github.com/ruvnet/agentic-flow) - Multi-agent orchestration

## Support

- GitHub Issues: https://github.com/neural-trader/neural-trader/issues
- Discord: https://discord.gg/neural-trader
- Documentation: https://docs.neural-trader.ai

---

Built with ❤️ by the Neural Trader team
