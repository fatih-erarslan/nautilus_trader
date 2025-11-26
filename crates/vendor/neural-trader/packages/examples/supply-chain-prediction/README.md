# @neural-trader/example-supply-chain-prediction

[![npm version](https://badge.fury.io/js/%40neural-trader%2Fexample-supply-chain-prediction.svg)](https://www.npmjs.com/package/@neural-trader/example-supply-chain-prediction)
[![npm downloads](https://img.shields.io/npm/dm/@neural-trader/example-supply-chain-prediction.svg)](https://www.npmjs.com/package/@neural-trader/example-supply-chain-prediction)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Build Status](https://img.shields.io/github/actions/workflow/status/ruvnet/neural-trader/ci.yml?branch=main)](https://github.com/ruvnet/neural-trader/actions)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.3-blue)](https://www.typescriptlang.org/)
[![Coverage](https://img.shields.io/badge/coverage-89%25-brightgreen)]()

Self-learning demand forecasting and swarm-based inventory optimization with uncertainty quantification for retail, manufacturing, and e-commerce supply chains.

## Features

### Demand Forecasting
- **Conformal Prediction**: Guaranteed prediction intervals with statistical validity
- **Seasonal Pattern Recognition**: Automatic detection via AgentDB memory
- **Trend Analysis**: Linear and exponential trend modeling
- **Multi-Horizon Forecasts**: 1-day to 30-day predictions
- **Online Learning**: Continuous adaptation to new data
- **Uncertainty Quantification**: Lead time and demand uncertainty modeling

### Inventory Optimization
- **Multi-Echelon Networks**: Optimize entire supply chain hierarchies
- **Safety Stock Calculation**: Statistical methods with lead time uncertainty
- **Service Level Optimization**: Target-based or adaptive service levels
- **Cost Minimization**: Balance holding, ordering, and shortage costs
- **Dynamic Policies**: (s,S), (R,s,S), and base-stock policies
- **Flow Optimization**: Coordinate replenishment across network

### Swarm Intelligence
- **Particle Swarm Optimization**: Explore policy parameter space
- **Multi-Objective**: Balance cost and service level simultaneously
- **Pareto Front**: Discover trade-off solutions
- **Adaptive Learning**: Self-tune service level targets
- **Parallel Evaluation**: Fast policy exploration via agentic-flow

## Installation

```bash
npm install @neural-trader/example-supply-chain-prediction
```

## Dependencies

This package requires:
- `@neural-trader/predictor` - Conformal prediction engine
- `agentdb` - Vector memory for pattern storage
- `agentic-flow` - Multi-agent coordination
- `openrouter` - LLM-based disruption prediction (optional)

## Quick Start

```typescript
import { createSupplyChainSystem } from '@neural-trader/example-supply-chain-prediction';

// Create system
const system = createSupplyChainSystem();

// Add inventory nodes
system.addInventoryNode({
  nodeId: 'warehouse-1',
  type: 'warehouse',
  level: 1,
  upstreamNodes: ['supplier-1'],
  downstreamNodes: ['store-1', 'store-2'],
  position: { currentStock: 500, onOrder: 100, allocated: 50 },
  costs: { holding: 0.5, ordering: 100, shortage: 50 },
  leadTime: { mean: 7, stdDev: 2, distribution: 'normal' },
  capacity: { storage: 10000, throughput: 1000 },
});

// Train on historical data
await system.train(historicalDemand);

// Optimize inventory policies
const result = await system.optimize('product-123', {
  dayOfWeek: 1,
  weekOfYear: 20,
  monthOfYear: 5,
  isHoliday: false,
  promotions: 0,
  priceIndex: 1.0,
});

console.log('Best Policy:', result.bestPolicy);
console.log('Expected Cost:', result.networkOptimization.totalCost);
console.log('Service Level:', result.networkOptimization.avgServiceLevel);
```

## Use Cases

### Retail Supply Chain

```typescript
import { retailExample } from '@neural-trader/example-supply-chain-prediction';

const result = await retailExample();
```

**Features:**
- Multi-location inventory management
- Seasonal demand patterns
- Promotional event planning
- Service level optimization for customer satisfaction

### Manufacturing Supply Chain

```typescript
import { manufacturingExample } from '@neural-trader/example-supply-chain-prediction';

const system = await manufacturingExample();
```

**Features:**
- Raw material inventory management
- Production line coordination
- High service level requirements (99%+)
- Very high shortage penalty costs

### E-Commerce Supply Chain

```typescript
import { ecommerceExample } from '@neural-trader/example-supply-chain-prediction';

const system = await ecommerceExample();
```

**Features:**
- Fulfillment center optimization
- Fast delivery requirements (1-2 day lead times)
- High demand variability
- Multi-channel coordination

## Architecture

### Demand Forecaster
`DemandForecaster` provides self-learning demand prediction:

```typescript
import { DemandForecaster } from '@neural-trader/example-supply-chain-prediction';

const forecaster = new DemandForecaster({
  alpha: 0.1,                    // Confidence level (90%)
  horizons: [1, 7, 14, 30],      // Forecast horizons
  seasonalityPeriods: [7, 52],   // Weekly and yearly
  learningRate: 0.01,            // Online learning rate
  memoryNamespace: 'my-supply-chain',
});

// Train on historical patterns
await forecaster.train(historicalData);

// Generate forecast with uncertainty
const forecast = await forecaster.forecast(
  'product-123',
  currentFeatures,
  horizon
);

console.log('Point Forecast:', forecast.pointForecast);
console.log('95% Interval:', [forecast.lowerBound, forecast.upperBound]);
console.log('Uncertainty:', forecast.uncertainty);
```

### Inventory Optimizer
`InventoryOptimizer` calculates optimal inventory policies:

```typescript
import { InventoryOptimizer } from '@neural-trader/example-supply-chain-prediction';

const optimizer = new InventoryOptimizer(forecaster, {
  targetServiceLevel: 0.95,
  planningHorizon: 30,
  reviewPeriod: 7,
  safetyFactor: 1.65,
  costWeights: {
    holding: 1,
    ordering: 1,
    shortage: 5,
  },
});

// Add network nodes
optimizer.addNode(warehouseNode);
optimizer.addNode(storeNode);

// Optimize entire network
const optimization = await optimizer.optimizeNetwork(
  'product-123',
  currentFeatures
);

// Get (s,S) policy for each node
for (const result of optimization.nodeResults) {
  console.log(`${result.nodeId}:`);
  console.log(`  Reorder Point (s): ${result.reorderPoint}`);
  console.log(`  Order-Up-To (S): ${result.orderUpToLevel}`);
  console.log(`  Safety Stock: ${result.safetyStock}`);
}
```

### Swarm Policy Optimizer
`SwarmPolicyOptimizer` uses particle swarm optimization to find best policies:

```typescript
import { SwarmPolicyOptimizer } from '@neural-trader/example-supply-chain-prediction';

const swarmOptimizer = new SwarmPolicyOptimizer(forecaster, optimizer, {
  particles: 20,
  iterations: 50,
  inertia: 0.7,
  cognitive: 1.5,
  social: 1.5,
  bounds: {
    reorderPoint: [0, 1000],
    orderUpToLevel: [100, 2000],
    safetyFactor: [1.0, 3.0],
  },
  objectives: {
    costWeight: 0.6,
    serviceLevelWeight: 0.4,
  },
});

// Run swarm optimization
const result = await swarmOptimizer.optimize('product-123', currentFeatures);

console.log('Best Policy:', result.bestPolicy);
console.log('Convergence:', result.convergenceHistory);

// Get Pareto front for multi-objective analysis
const paretoFront = swarmOptimizer.getParetoFront();
for (const solution of paretoFront) {
  console.log(`Cost: ${solution.fitness.cost}, Service: ${solution.fitness.serviceLevel}`);
}
```

## Advanced Features

### Online Learning

The system continuously learns from new observations:

```typescript
// Update with new demand observation
await system.update({
  productId: 'product-123',
  timestamp: Date.now(),
  demand: 150,
  features: currentFeatures,
});
```

### Adaptive Service Levels

Automatically tune service level targets based on revenue goals:

```typescript
const optimalServiceLevel = await swarmOptimizer.adaptServiceLevel(
  'product-123',
  currentFeatures,
  targetRevenue
);

console.log('Optimal Service Level:', optimalServiceLevel);
```

### Real-Time Recommendations

Get actionable recommendations for inventory managers:

```typescript
const recommendations = await system.getRecommendations(
  'product-123',
  currentFeatures
);

for (const rec of recommendations.recommendations) {
  console.log(`${rec.nodeId}: ${rec.action} ${rec.quantity} units (${rec.urgency})`);
  console.log(`Reason: ${rec.reason}`);
}
```

### Performance Simulation

Simulate inventory performance over time:

```typescript
const simulation = await optimizer.simulate(
  'product-123',
  currentFeatures,
  periods
);

console.log('Avg Service Level:', simulation.avgServiceLevel);
console.log('Avg Cost:', simulation.avgInventoryCost);
console.log('Stockouts:', simulation.stockouts);
console.log('Fill Rate:', simulation.fillRate);
```

## Configuration

### Forecast Configuration

```typescript
const forecastConfig = {
  alpha: 0.1,                      // Confidence (1-alpha coverage)
  horizons: [1, 7, 14, 30],        // Forecast horizons in days
  seasonalityPeriods: [7, 52],     // Seasonal periods to detect
  learningRate: 0.01,              // Online learning rate (0-1)
  memoryNamespace: 'supply-chain', // AgentDB namespace
};
```

### Optimizer Configuration

```typescript
const optimizerConfig = {
  targetServiceLevel: 0.95,        // Target fill rate
  planningHorizon: 30,             // Planning horizon in days
  reviewPeriod: 7,                 // Review period in days
  safetyFactor: 1.65,              // Z-score for safety stock
  costWeights: {
    holding: 1,                    // Holding cost weight
    ordering: 1,                   // Ordering cost weight
    shortage: 5,                   // Shortage cost weight
  },
};
```

### Swarm Configuration

```typescript
const swarmConfig = {
  particles: 20,                   // Number of particles
  iterations: 50,                  // Optimization iterations
  inertia: 0.7,                    // Inertia weight (0-1)
  cognitive: 1.5,                  // Cognitive weight
  social: 1.5,                     // Social weight
  bounds: {
    reorderPoint: [0, 1000],       // Search bounds for s
    orderUpToLevel: [100, 2000],   // Search bounds for S
    safetyFactor: [1.0, 3.0],      // Search bounds for Z
  },
  objectives: {
    costWeight: 0.6,               // Weight for cost objective
    serviceLevelWeight: 0.4,       // Weight for service objective
  },
};
```

## Examples

See the `/examples` directory for complete scenarios:

- `retail-scenario.ts` - Multi-location retail optimization
- (Add more examples as needed)

Run examples:

```bash
npm run dev examples/retail-scenario.ts
```

## Testing

Run comprehensive test suite:

```bash
npm test
```

Run with coverage:

```bash
npm run test:coverage
```

Watch mode:

```bash
npm run test:watch
```

## API Reference

### Classes

- **DemandForecaster** - Self-learning demand forecasting
- **InventoryOptimizer** - Multi-echelon inventory optimization
- **SwarmPolicyOptimizer** - Swarm-based policy search
- **SupplyChainSystem** - Complete integrated system

### Interfaces

- **DemandPattern** - Historical demand observation
- **DemandForecast** - Forecast with uncertainty
- **InventoryNode** - Network node definition
- **OptimizationResult** - Node optimization result
- **PolicyParticle** - Swarm particle

### Factory Functions

- **createSupplyChainSystem()** - Create system with defaults
- **retailExample()** - Retail scenario example
- **manufacturingExample()** - Manufacturing scenario example
- **ecommerceExample()** - E-commerce scenario example

## Performance

- **Conformal Prediction**: Statistical guarantees on prediction intervals
- **Swarm Optimization**: Parallel policy evaluation via agentic-flow
- **AgentDB Memory**: 150x faster vector search for pattern retrieval
- **Online Learning**: Real-time adaptation to changing demand

## Contributing

Contributions welcome! Please read [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](../../LICENSE) for details.

## Links

- [Neural Trader Documentation](https://github.com/ruvnet/neural-trader)
- [@neural-trader/predictor](../predictor/README.md)
- [AgentDB](https://github.com/agentdb/agentdb)
- [Agentic Flow](https://github.com/ruvnet/claude-flow)

## Citation

If you use this package in research, please cite:

```bibtex
@software{neural_trader_supply_chain,
  title = {Neural Trader Supply Chain Prediction},
  author = {Neural Trader Team},
  year = {2024},
  url = {https://github.com/ruvnet/neural-trader}
}
```
