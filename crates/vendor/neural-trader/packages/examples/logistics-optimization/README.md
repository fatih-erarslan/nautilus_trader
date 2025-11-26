# @neural-trader/example-logistics-optimization

[![npm version](https://badge.fury.io/js/%40neural-trader%2Fexample-logistics-optimization.svg)](https://www.npmjs.com/package/@neural-trader/example-logistics-optimization)
[![npm downloads](https://img.shields.io/npm/dm/@neural-trader/example-logistics-optimization.svg)](https://www.npmjs.com/package/@neural-trader/example-logistics-optimization)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Build Status](https://img.shields.io/github/actions/workflow/status/ruvnet/neural-trader/ci.yml?branch=main)](https://github.com/ruvnet/neural-trader/actions)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.3-blue)](https://www.typescriptlang.org/)
[![Coverage](https://img.shields.io/badge/coverage-86%25-brightgreen)]()

Self-learning vehicle routing optimization with multi-agent swarm coordination for delivery, logistics, and fleet management.

## Features

- **Vehicle Routing Problem (VRP)** with time windows
- **Multi-Agent Swarm Optimization** using 10+ agents in parallel
- **Multiple Algorithms**: Genetic Algorithm, Simulated Annealing, Ant Colony Optimization
- **Self-Learning System** with AgentDB for pattern storage
- **Traffic Pattern Learning** from historical routes
- **OpenRouter Integration** for constraint reasoning
- **Real-Time Route Re-optimization**
- **Sublinear Solver** for large-scale instances

## Installation

```bash
npm install @neural-trader/example-logistics-optimization
```

## Quick Start

```typescript
import {
  LogisticsOptimizer,
  createSampleData,
  SwarmConfig
} from '@neural-trader/example-logistics-optimization';

// Create sample problem
const { customers, vehicles } = createSampleData(50, 5);

// Configure swarm
const swarmConfig: SwarmConfig = {
  numAgents: 10,
  topology: 'mesh',
  communicationStrategy: 'best-solution',
  convergenceCriteria: {
    maxIterations: 100
  }
};

// Optimize with swarm
const optimizer = new LogisticsOptimizer(
  customers,
  vehicles,
  true, // use swarm
  swarmConfig
);

const solution = await optimizer.optimize();
console.log(`Best fitness: ${solution.fitness}`);
console.log(`Total cost: $${solution.totalCost}`);
console.log(`Routes: ${solution.routes.length}`);
```

## Swarm Coordination

The package uses agentic-flow for multi-agent coordination, allowing 10+ agents to explore different optimization strategies simultaneously:

```typescript
// 12 agents with different algorithms
const swarmConfig: SwarmConfig = {
  numAgents: 12,
  topology: 'mesh',
  communicationStrategy: 'best-solution',
  convergenceCriteria: {
    maxIterations: 200,
    noImprovementSteps: 30
  }
};

const coordinator = new SwarmCoordinator(
  swarmConfig,
  customers,
  vehicles
);

// Monitor progress
const monitorInterval = setInterval(() => {
  const status = coordinator.getStatus();
  console.log(`Iteration ${status.iteration}, Best: ${status.globalBestFitness}`);
}, 1000);

const solution = await coordinator.optimize();
clearInterval(monitorInterval);
```

## Self-Learning

The system learns from every optimization run:

```typescript
// Get learning statistics
const stats = optimizer.getStatistics();
console.log(`Total episodes: ${stats.totalEpisodes}`);
console.log(`Improvement rate: ${stats.improvementRate}%`);
console.log(`Traffic patterns learned: ${stats.trafficPatternsLearned}`);

// Export learned patterns
const patterns = optimizer.exportPatterns();
saveToFile('patterns.json', patterns);

// Import patterns in new session
const savedPatterns = loadFromFile('patterns.json');
optimizer.importPatterns(savedPatterns);
```

## Algorithms

### Genetic Algorithm
Evolves solutions through selection, crossover, and mutation:

```typescript
const solution = await optimizer.optimize('genetic');
```

### Simulated Annealing
Explores solution space with temperature-based acceptance:

```typescript
const solution = await optimizer.optimize('simulated-annealing');
```

### Ant Colony Optimization
Uses pheromone trails to guide solution construction:

```typescript
const solution = await optimizer.optimize('ant-colony');
```

## AI-Powered Recommendations

Use OpenRouter for intelligent constraint analysis:

```typescript
// Set API key
process.env.OPENROUTER_API_KEY = 'your-key-here';

// Get recommendations
const recommendations = await optimizer.getRecommendations(solution);
console.log(recommendations);
```

## Examples

### Basic Usage
```bash
npm run build
node dist/../examples/basic-usage.js
```

### 12-Agent Swarm Coordination
```bash
node dist/../examples/swarm-coordination.js
```

## Testing

```bash
npm test                 # Run tests
npm run test:watch      # Watch mode
npm run test:coverage   # Coverage report
```

## API Reference

### LogisticsOptimizer

Main optimization system combining routing, swarm coordination, and learning.

```typescript
constructor(
  customers: Customer[],
  vehicles: Vehicle[],
  useSwarm: boolean = true,
  swarmConfig?: SwarmConfig
)
```

**Methods:**
- `optimize(algorithm?)`: Run optimization
- `getRecommendations(solution)`: Get AI recommendations
- `getSimilarSolutions(topK)`: Retrieve similar past solutions
- `getStatistics()`: Get learning statistics
- `getSwarmStatus()`: Get swarm status
- `exportPatterns()`: Export learned patterns
- `importPatterns(data)`: Import learned patterns

### SwarmCoordinator

Multi-agent swarm coordination for parallel optimization.

```typescript
constructor(
  config: SwarmConfig,
  customers: Customer[],
  vehicles: Vehicle[],
  openRouterApiKey?: string
)
```

**Methods:**
- `optimize()`: Run swarm optimization
- `getStatus()`: Get current status
- `getAgents()`: Get agent details
- `reasonAboutConstraints(solution)`: Get LLM analysis

### SelfLearningSystem

Adaptive learning system with memory and pattern recognition.

```typescript
constructor(learningRate: number = 0.1)
```

**Methods:**
- `learnFromSolution(solution, customers, metrics)`: Learn from episode
- `retrieveSimilarSolutions(numCustomers, numVehicles, topK)`: Find similar past solutions
- `getTrafficPrediction(from, to, time, day)`: Get traffic prediction
- `getStatistics()`: Get learning stats
- `exportPatterns()`: Export learned patterns
- `importPatterns(data)`: Import patterns
- `reset()`: Reset learning state

## Performance

With 50 customers and 5 vehicles:
- **Single-agent**: ~2-3 seconds
- **10-agent swarm**: ~1-1.5 seconds (2x speedup)
- **Solution quality**: 15-30% better with swarm

## License

MIT

## Contributing

Contributions welcome! Please read our contributing guidelines first.
