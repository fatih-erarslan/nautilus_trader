# Benchmark Swarm Framework

Generic swarm orchestration framework for parallel variation exploration, performance benchmarking, and parameter optimization using agentic-flow.

## Features

- **Swarm Coordination**: Star, mesh, and hierarchical topologies
- **Parallel Execution**: Run variations concurrently with configurable agents
- **Performance Benchmarking**: Comprehensive metrics and statistical analysis
- **Parameter Optimization**: Grid search, random search, Bayesian, and evolutionary algorithms
- **AgentDB Integration**: Persistent optimization history

## Installation

```bash
npm install @neural-trader/benchmark-swarm-framework
```

## Quick Start

```typescript
import {
  SwarmCoordinator,
  BenchmarkRunner,
  Optimizer,
} from '@neural-trader/benchmark-swarm-framework';

// Define task to benchmark
const task = {
  execute: async (params) => {
    // Your algorithm implementation
    return performComputation(params);
  },
  validate: (result) => result > 0,
};

// Define variations to test
const variations = [
  { id: 'v1', parameters: { threshold: 0.5, windowSize: 10 } },
  { id: 'v2', parameters: { threshold: 0.7, windowSize: 20 } },
  { id: 'v3', parameters: { threshold: 0.9, windowSize: 30 } },
];

// Run swarm coordinator
const coordinator = new SwarmCoordinator({
  maxAgents: 5,
  topology: 'mesh',
  communicationProtocol: 'async',
});

const results = await coordinator.executeVariations(
  variations,
  task,
  (completed, total) => console.log(`Progress: ${completed}/${total}`)
);

// Analyze results
const best = coordinator.getBestVariation('executionTime');
console.log('Best configuration:', best.variationId);
console.log('Execution time:', best.metrics.executionTime, 'ms');
```

## Swarm Coordination

### Topologies

```typescript
// Star topology (centralized coordinator)
const coordinator = new SwarmCoordinator({
  maxAgents: 10,
  topology: 'star',
  communicationProtocol: 'sync',
  timeout: 30000,
});

// Mesh topology (peer-to-peer)
const meshCoordinator = new SwarmCoordinator({
  maxAgents: 8,
  topology: 'mesh',
  communicationProtocol: 'async',
});

// Hierarchical topology (tree structure)
const hierarchical = new SwarmCoordinator({
  maxAgents: 20,
  topology: 'hierarchical',
  communicationProtocol: 'sync',
});
```

### Execute Variations

```typescript
const results = await coordinator.executeVariations(
  variations,
  task,
  (completed, total) => {
    console.log(`Progress: ${completed}/${total}`);
  }
);

// Get statistics
const stats = coordinator.getStatistics();
console.log('Success rate:', stats.successRate * 100, '%');
console.log('Average execution time:', stats.averageExecutionTime, 'ms');
```

## Performance Benchmarking

### Basic Benchmark

```typescript
const runner = new BenchmarkRunner({
  maxAgents: 5,
  topology: 'mesh',
  communicationProtocol: 'async',
  iterations: 100,
  warmupIterations: 10,
  collectGarbage: true,
});

const suite = {
  name: 'Algorithm Performance Test',
  description: 'Compare different parameter configurations',
  task: {
    execute: async (params) => {
      return await runAlgorithm(params);
    },
  },
  variations: [
    { id: 'baseline', parameters: { ...defaultParams } },
    { id: 'optimized-v1', parameters: { ...optimizedParams1 } },
    { id: 'optimized-v2', parameters: { ...optimizedParams2 } },
  ],
};

const report = await runner.runSuite(suite);
```

### Benchmark Report

```typescript
// Report includes:
{
  suiteName: 'Algorithm Performance Test',
  timestamp: Date,
  results: [
    {
      variationId: 'baseline',
      success: true,
      stats: {
        mean: 125.4,           // Average time
        median: 120.0,         // Median time
        min: 95.2,             // Fastest run
        max: 180.5,            // Slowest run
        stdDev: 15.3,          // Standard deviation
        percentile95: 155.0,   // 95th percentile
        percentile99: 170.0,   // 99th percentile
      },
      throughput: 7.97,        // Operations per second
    },
    // ... more results
  ],
  summary: {
    totalDuration: 45000,
    successRate: 1.0,
    fastestVariation: 'optimized-v2',
    slowestVariation: 'baseline',
  }
}
```

### Compare Variations

```typescript
const comparison = runner.compareVariations(
  report,
  'baseline',
  'optimized-v2'
);

console.log(`${comparison.fasterVariation} is faster by:`);
console.log(`  ${comparison.speedupFactor.toFixed(2)}x`);
console.log(`  ${comparison.percentDifference.toFixed(1)}%`);
```

## Parameter Optimization

### Grid Search

```typescript
const optimizer = new Optimizer({
  maxAgents: 10,
  topology: 'mesh',
  communicationProtocol: 'async',
  strategy: 'grid-search',
  objective: 'minimize',
  metric: 'executionTime',
  iterations: 1,
  budget: {
    maxEvaluations: 100,
    maxTime: 300000, // 5 minutes
  },
});

const parameterSpace = {
  threshold: {
    type: 'continuous',
    range: [0.1, 0.9],
    scale: 'linear',
  },
  windowSize: {
    type: 'discrete',
    values: [10, 20, 30, 40, 50],
  },
  method: {
    type: 'categorical',
    values: ['sma', 'ema', 'wma'],
  },
};

const result = await optimizer.optimize(parameterSpace, task);

console.log('Best parameters:', result.bestParameters);
console.log('Best score:', result.bestScore);
console.log('Converged:', result.metadata.converged);
```

### Random Search

```typescript
const randomOptimizer = new Optimizer({
  maxAgents: 5,
  topology: 'star',
  communicationProtocol: 'sync',
  strategy: 'random-search',
  objective: 'maximize',
  metric: 'throughput',
  iterations: 1,
  budget: {
    maxEvaluations: 50,
  },
});

const result = await randomOptimizer.optimize(parameterSpace, task);
```

### Evolutionary Optimization

```typescript
const evolutionaryOptimizer = new Optimizer({
  maxAgents: 8,
  topology: 'hierarchical',
  communicationProtocol: 'async',
  strategy: 'evolutionary',
  objective: 'minimize',
  metric: 'executionTime',
  iterations: 1,
  budget: {
    maxEvaluations: 200,
  },
  convergenceCriteria: {
    threshold: 0.001,
    patience: 10,
  },
});

const result = await evolutionaryOptimizer.optimize(parameterSpace, task);

// View convergence history
console.log('Convergence:', result.convergenceHistory);
```

### Custom Metrics

```typescript
const customOptimizer = new Optimizer({
  maxAgents: 5,
  topology: 'mesh',
  communicationProtocol: 'async',
  strategy: 'random-search',
  objective: 'maximize',
  metric: 'custom',
  customMetric: (result) => {
    // Custom scoring function
    const accuracy = result.correctPredictions / result.totalPredictions;
    const speed = 1000 / result.metrics.executionTime;
    return accuracy * 0.7 + speed * 0.3; // Weighted combination
  },
  iterations: 1,
  budget: { maxEvaluations: 100 },
});
```

## Complete Example

```typescript
import {
  SwarmCoordinator,
  BenchmarkRunner,
  Optimizer,
} from '@neural-trader/benchmark-swarm-framework';

// 1. Define task
const tradingTask = {
  execute: async (params) => {
    const strategy = new TradingStrategy(params);
    const result = await strategy.backtest(historicalData);
    return {
      sharpeRatio: result.sharpe,
      maxDrawdown: result.maxDrawdown,
      totalReturn: result.return,
    };
  },
  validate: (result) => result.sharpeRatio > 0,
};

// 2. Run initial benchmark
const benchmarkRunner = new BenchmarkRunner({
  maxAgents: 5,
  topology: 'mesh',
  communicationProtocol: 'async',
  iterations: 50,
  warmupIterations: 5,
});

const initialVariations = [
  { id: 'conservative', parameters: { riskLevel: 0.3, leverage: 1.0 } },
  { id: 'moderate', parameters: { riskLevel: 0.5, leverage: 1.5 } },
  { id: 'aggressive', parameters: { riskLevel: 0.8, leverage: 2.0 } },
];

const benchmarkReport = await benchmarkRunner.runSuite({
  name: 'Initial Trading Strategy Benchmark',
  task: tradingTask,
  variations: initialVariations,
});

// 3. Optimize parameters
const optimizer = new Optimizer({
  maxAgents: 10,
  topology: 'hierarchical',
  communicationProtocol: 'async',
  strategy: 'evolutionary',
  objective: 'maximize',
  metric: 'custom',
  customMetric: (result) => result.result.sharpeRatio,
  iterations: 1,
  budget: { maxEvaluations: 100 },
});

const optimizationResult = await optimizer.optimize(
  {
    riskLevel: { type: 'continuous', range: [0.1, 0.9], scale: 'linear' },
    leverage: { type: 'continuous', range: [1.0, 3.0], scale: 'linear' },
    stopLoss: { type: 'continuous', range: [0.01, 0.1], scale: 'linear' },
    takeProfit: { type: 'continuous', range: [0.02, 0.2], scale: 'linear' },
  },
  tradingTask
);

console.log('Optimized parameters:', optimizationResult.bestParameters);
console.log('Sharpe ratio:', optimizationResult.bestScore);

// 4. Validate optimized configuration
const finalBenchmark = await benchmarkRunner.runSuite({
  name: 'Optimized Strategy Validation',
  task: tradingTask,
  variations: [
    { id: 'baseline', parameters: initialVariations[1].parameters },
    { id: 'optimized', parameters: optimizationResult.bestParameters },
  ],
});

const comparison = benchmarkRunner.compareVariations(
  finalBenchmark,
  'baseline',
  'optimized'
);

console.log(`Optimization improved performance by ${comparison.percentDifference.toFixed(1)}%`);
```

## API Reference

### SwarmCoordinator

```typescript
class SwarmCoordinator<T> {
  constructor(config: SwarmConfig);
  executeVariations(variations: TaskVariation[], task: AgentTask<T>, onProgress?: ProgressCallback): Promise<SwarmResult<T>[]>;
  getStatistics(): SwarmStatistics;
  getBestVariation(metric?: 'executionTime' | 'memoryUsed'): SwarmResult<T> | null;
  clear(): void;
}
```

### BenchmarkRunner

```typescript
class BenchmarkRunner<T> {
  constructor(config: BenchmarkConfig);
  runSuite(suite: BenchmarkSuite<T>): Promise<BenchmarkReport>;
  compareVariations(report: BenchmarkReport, variation1: string, variation2: string): ComparisonResult;
  getReports(): BenchmarkReport[];
}
```

### Optimizer

```typescript
class Optimizer<T> {
  constructor(config: OptimizationConfig);
  optimize(parameterSpace: ParameterSpace, task: AgentTask<T>): Promise<OptimizationResult<T>>;
}
```

## Best Practices

1. **Start with small swarms** (5-10 agents) and scale up
2. **Use appropriate topology** for your communication pattern
3. **Always run warmup iterations** for accurate benchmarks
4. **Enable garbage collection** between benchmark runs
5. **Define validation functions** to catch invalid results
6. **Use evolutionary optimization** for complex parameter spaces
7. **Track convergence** to avoid unnecessary evaluations

## License

MIT
