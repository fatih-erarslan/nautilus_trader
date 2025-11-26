# Architecture Overview

This document describes the architectural patterns and design principles used across all Neural Trader examples.

## Core Architecture Principles

### 1. Layered Architecture

All examples follow a consistent 3-layer architecture:

```
┌─────────────────────────────────────────┐
│         Public API Layer                │
│  (index.ts - exported interfaces)       │
└─────────────────────────────────────────┘
                    │
┌─────────────────────────────────────────┐
│      Business Logic Layer               │
│  (core algorithms, optimizers)          │
└─────────────────────────────────────────┘
                    │
┌─────────────────────────────────────────┐
│      Data/Learning Layer                │
│  (AgentDB, swarm coordination)          │
└─────────────────────────────────────────┘
```

### 2. Component Composition

Each example consists of 3-5 core components:

#### Core Algorithm Component
- Pure implementation of the main algorithm
- No external dependencies on learning/swarm
- Testable in isolation
- Example: `OrderBookAnalyzer`, `MeanVarianceOptimizer`

#### Self-Learning Component
- AgentDB integration for persistent memory
- Experience replay and pattern recognition
- Adaptive parameter optimization
- Example: `PatternLearner`, `SelfLearningOptimizer`

#### Swarm Component
- Multi-agent coordination for exploration
- Parallel strategy evaluation
- Consensus-based decision making
- Example: `SwarmFeatureEngineer`, `BenchmarkSwarm`

#### Orchestrator Component
- Coordinates all components
- Provides unified API
- Manages lifecycle (init, run, cleanup)
- Example: `MarketMicrostructure`, `PortfolioOptimizationSwarm`

#### Types Component
- Shared interfaces and types
- Configuration structures
- Result types
- Example: `types.ts` in each package

---

## Example: Market Microstructure Architecture

```typescript
┌────────────────────────────────────────────────┐
│      MarketMicrostructure (Orchestrator)       │
│  - initialize()                                │
│  - analyze(orderBook)                          │
│  - learn(outcome)                              │
│  - exploreFeatures()                           │
└───────────┬────────────────────────────────────┘
            │
     ┌──────┴───────┬──────────────┐
     │              │              │
┌────▼────┐  ┌─────▼─────┐  ┌────▼──────┐
│ Order   │  │  Pattern  │  │   Swarm   │
│  Book   │  │  Learner  │  │  Feature  │
│Analyzer │  │           │  │ Engineer  │
└─────────┘  └─────┬─────┘  └─────┬─────┘
                   │              │
                   │              │
                ┌──▼──────────────▼──┐
                │     AgentDB        │
                │  (Persistent       │
                │   Memory)          │
                └────────────────────┘
```

### Data Flow

1. **Analysis Phase**:
   ```
   OrderBook → Analyzer → Metrics → Learner → Pattern Recognition
                                  ↓
                                Swarm → Anomaly Detection
   ```

2. **Learning Phase**:
   ```
   Outcome → Learner → Feature Extraction → AgentDB Storage
                                          ↓
                                    Experience Replay
   ```

3. **Exploration Phase**:
   ```
   Metrics History → Swarm → Feature Evolution → Best Features
                             ↑                        ↓
                             └────────────────────────┘
                               (50+ generations)
   ```

---

## Example: Portfolio Optimization Architecture

```typescript
┌────────────────────────────────────────────────┐
│   PortfolioOptimizationSwarm (Orchestrator)    │
│  - runBenchmark()                              │
│  - exploreConstraints()                        │
│  - compareMarketRegimes()                      │
└───────────┬────────────────────────────────────┘
            │
     ┌──────┴────────┬────────────────┬────────────┐
     │               │                │            │
┌────▼────┐  ┌──────▼──────┐  ┌─────▼──┐  ┌─────▼────┐
│  Mean   │  │    Risk     │  │ Black  │  │  Multi   │
│Variance │  │   Parity    │  │Litterman│ │Objective │
└─────────┘  └─────────────┘  └────────┘  └──────────┘
     │               │                │            │
     └───────────────┴────────────────┴────────────┘
                     │
          ┌──────────▼──────────┐
          │  SelfLearning       │
          │   Optimizer         │
          │  (AgentDB Memory)   │
          └─────────────────────┘
```

### Parallel Execution

```
Benchmark Request
        │
        ├─► Algorithm 1 (Mean-Variance)
        ├─► Algorithm 2 (Risk Parity)
        ├─► Algorithm 3 (Black-Litterman)
        └─► Algorithm 4 (Multi-Objective)
                │
        Aggregate Results
                │
        Rank by Performance
                │
        Return Best Strategy
```

---

## Common Patterns

### 1. Initialization Pattern

All components follow consistent initialization:

```typescript
class Component {
  private initialized: boolean = false;

  async initialize(): Promise<void> {
    if (this.initialized) return;

    // Setup resources
    await this.setupResources();

    // Load from memory if available
    await this.loadFromMemory();

    this.initialized = true;
  }

  async operation(): Promise<Result> {
    if (!this.initialized) {
      throw new Error('Not initialized');
    }
    // ... operation logic
  }
}
```

### 2. Builder Pattern

Complex configurations use builder pattern:

```typescript
const mm = await createMarketMicrostructure({
  agentDbPath: './memory.db',
  useSwarm: true,
  swarmConfig: {
    numAgents: 30,
    generations: 50,
    mutationRate: 0.2
  }
});
```

### 3. Strategy Pattern

Algorithms are swappable:

```typescript
interface OptimizationStrategy {
  optimize(assets: Asset[], constraints: Constraints): Result;
}

class MeanVarianceOptimizer implements OptimizationStrategy { }
class RiskParityOptimizer implements OptimizationStrategy { }
class BlackLittermanOptimizer implements OptimizationStrategy { }

// Swap strategies at runtime
const strategy: OptimizationStrategy = getStrategy(type);
const result = strategy.optimize(assets, constraints);
```

### 4. Observer Pattern

Learning updates use event-driven pattern:

```typescript
class PatternLearner {
  private listeners: Array<(pattern: Pattern) => void> = [];

  onPatternLearned(callback: (pattern: Pattern) => void) {
    this.listeners.push(callback);
  }

  async learnPattern(features: Features, outcome: Outcome) {
    const pattern = await this.storePattern(features, outcome);
    // Notify all observers
    this.listeners.forEach(listener => listener(pattern));
  }
}
```

### 5. Factory Pattern

Component creation centralized:

```typescript
export async function createMarketMicrostructure(
  config?: MarketMicrostructureConfig
): Promise<MarketMicrostructure> {
  const mm = new MarketMicrostructure(config);
  await mm.initialize();
  return mm;
}
```

---

## AgentDB Integration Architecture

### Memory Hierarchy

```
┌─────────────────────────────────────────────┐
│          Application Layer                  │
│  (Business Logic)                           │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────▼───────────────────────────┐
│       Learning Coordinator                  │
│  - Pattern recognition                      │
│  - Experience replay                        │
│  - Memory distillation                      │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────▼───────────────────────────┐
│           AgentDB Layer                     │
│  ┌────────────────────────────────────┐    │
│  │  Vector Store (HNSW Index)         │    │
│  │  - Embeddings                      │    │
│  │  - Similarity search (150x faster) │    │
│  └────────────────────────────────────┘    │
│  ┌────────────────────────────────────┐    │
│  │  RL Algorithms                     │    │
│  │  - Decision Transformer            │    │
│  │  - Q-Learning, SARSA, Actor-Critic │    │
│  └────────────────────────────────────┘    │
│  ┌────────────────────────────────────┐    │
│  │  Memory Management                 │    │
│  │  - Trajectory storage              │    │
│  │  - Verdict judgment                │    │
│  │  - Distillation                    │    │
│  └────────────────────────────────────┘    │
└─────────────────────────────────────────────┘
```

### Learning Workflow

```typescript
// 1. Store experience
await agentdb.storeTrajectory({
  state: features,
  action: decision,
  reward: outcome,
  nextState: nextFeatures
});

// 2. Retrieve similar experiences
const similar = await agentdb.similaritySearch(currentFeatures, 10);

// 3. Learn from experiences
const trainedModel = await agentdb.trainRL({
  algorithm: 'decision_transformer',
  trajectories: similar
});

// 4. Distill insights
const insights = await agentdb.distillMemory(trajectories);
```

---

## Swarm Coordination Architecture

### Swarm Topology

```
                  ┌──────────┐
                  │ Swarm    │
                  │Coordinator│
                  └─────┬────┘
                        │
      ┌─────────────────┼─────────────────┐
      │                 │                 │
┌─────▼──────┐   ┌─────▼──────┐   ┌─────▼──────┐
│ Explorer   │   │ Optimizer  │   │ Validator  │
│  Agents    │   │  Agents    │   │  Agents    │
│  (30%)     │   │  (40%)     │   │  (30%)     │
└────────────┘   └────────────┘   └────────────┘
      │                 │                 │
      └─────────────────┼─────────────────┘
                        │
                  ┌─────▼────┐
                  │ Shared   │
                  │ Memory   │
                  └──────────┘
```

### Agent Lifecycle

```
Initialize Agents
       │
       ├─► Assign Roles (Explorer, Optimizer, Validator)
       │
       ├─► Generation Loop:
       │      │
       │      ├─► Evaluate Fitness
       │      ├─► Select Parents (Tournament)
       │      ├─► Crossover & Mutation
       │      ├─► Create Offspring
       │      └─► Replace Worst
       │
       └─► Extract Best Solutions
                  │
            Consensus Building
                  │
            Return Results
```

---

## Performance Optimization Strategies

### 1. Lazy Initialization

```typescript
class Optimizer {
  private _swarm?: SwarmFeatureEngineer;

  get swarm(): SwarmFeatureEngineer {
    if (!this._swarm) {
      this._swarm = new SwarmFeatureEngineer(this.config);
    }
    return this._swarm;
  }
}
```

### 2. Caching

```typescript
class PatternLearner {
  private patternCache = new Map<string, Pattern>();

  async recognizePattern(features: Features): Promise<Pattern | null> {
    const key = this.featuresToKey(features);

    if (this.patternCache.has(key)) {
      return this.patternCache.get(key)!;
    }

    const pattern = await this.searchPatterns(features);
    this.patternCache.set(key, pattern);
    return pattern;
  }
}
```

### 3. Parallel Processing

```typescript
async runBenchmark(config: BenchmarkConfig): Promise<Insights> {
  // Run all algorithms in parallel
  const results = await Promise.all([
    this.runMeanVariance(config),
    this.runRiskParity(config),
    this.runBlackLitterman(config),
    this.runMultiObjective(config)
  ]);

  return this.aggregateResults(results);
}
```

### 4. Memory Quantization

```typescript
// 4-32x memory reduction with AgentDB quantization
const learner = new SelfLearningOptimizer('./memory.db', {
  quantization: '8bit'  // 4bit, 8bit, 16bit, 32bit
});
```

### 5. HNSW Indexing

```typescript
// 150x faster similarity search
await agentdb.initialize({
  indexType: 'hnsw',
  M: 16,              // Connections per layer
  efConstruction: 200 // Build quality
});
```

---

## Error Handling Architecture

### Layered Error Handling

```typescript
// Base error types
class NeuralTraderError extends Error { }
class InitializationError extends NeuralTraderError { }
class OptimizationError extends NeuralTraderError { }
class LearningError extends NeuralTraderError { }

// Component-level error handling
class Component {
  async operation(): Promise<Result> {
    try {
      return await this.executeOperation();
    } catch (error) {
      if (error instanceof KnownError) {
        return this.handleKnownError(error);
      }
      throw new ComponentError('Operation failed', { cause: error });
    }
  }
}

// Top-level error handling
async function main() {
  try {
    const result = await component.operation();
    return result;
  } catch (error) {
    logger.error('Operation failed', error);
    await cleanup();
    throw error;
  }
}
```

---

## Testing Architecture

### Test Pyramid

```
                    ┌──────────┐
                    │  E2E     │  (5%)
                    │  Tests   │
                    └──────────┘
              ┌────────────────────┐
              │  Integration Tests │  (15%)
              └────────────────────┘
        ┌──────────────────────────────┐
        │       Unit Tests             │  (80%)
        └──────────────────────────────┘
```

### Test Structure

```typescript
describe('OrderBookAnalyzer', () => {
  describe('Unit Tests', () => {
    it('calculates bid-ask spread correctly', () => { });
    it('handles empty order book', () => { });
    it('detects toxic order flow', () => { });
  });

  describe('Integration Tests', () => {
    it('analyzes order book with real data', () => { });
    it('integrates with PatternLearner', () => { });
  });

  describe('Performance Tests', () => {
    it('analyzes 1000 order books in <1s', () => { });
  });
});
```

---

## Deployment Architecture

### Package Structure

```
@neural-trader/example-*
├── dist/              # Compiled output
│   ├── index.js       # CommonJS
│   ├── index.mjs      # ES Module
│   └── index.d.ts     # Type definitions
├── src/               # Source code
├── tests/             # Test suites
├── examples/          # Usage examples
├── package.json       # Package manifest
├── tsconfig.json      # TypeScript config
└── README.md          # Documentation
```

### Build Pipeline

```
Source (TypeScript)
       │
       ├─► TypeScript Compiler → dist/*.js, dist/*.d.ts
       ├─► ESLint → Code quality checks
       ├─► Jest/Vitest → Test execution
       └─► tsup/tsc → Bundle & optimize
                │
          Publishing to npm
```

---

## Security Architecture

### Input Validation

```typescript
import { z } from 'zod';

const AssetSchema = z.object({
  symbol: z.string().min(1).max(10),
  expectedReturn: z.number().min(-1).max(10),
  volatility: z.number().min(0).max(10)
});

function validateAssets(assets: unknown): Asset[] {
  return z.array(AssetSchema).parse(assets);
}
```

### API Key Management

```typescript
// Never hardcode API keys
const apiKey = process.env.OPENROUTER_API_KEY;

if (!apiKey && config.useOpenRouter) {
  throw new Error('OPENROUTER_API_KEY not set');
}
```

### Database Security

```typescript
// Use parameterized queries
await db.query('SELECT * FROM patterns WHERE id = ?', [patternId]);

// Not: await db.query(`SELECT * FROM patterns WHERE id = ${patternId}`);
```

---

## Scalability Considerations

### Horizontal Scaling

```typescript
// Partition agents across workers
const workers = Array.from({ length: numCPUs }, () =>
  new Worker('./agent-worker.js')
);

const results = await Promise.all(
  workers.map(worker => worker.runAgents(subset))
);
```

### Vertical Scaling

```typescript
// Use streaming for large datasets
async function* processLargeDataset(filePath: string) {
  const stream = createReadStream(filePath);
  for await (const chunk of stream) {
    yield processChunk(chunk);
  }
}
```

### Memory Management

```typescript
// Limit in-memory history
class Analyzer {
  private maxHistory = 1000;

  addMetric(metric: Metric) {
    this.history.push(metric);

    if (this.history.length > this.maxHistory) {
      // Offload to AgentDB
      await this.persistOldHistory(this.history.slice(0, 500));
      this.history = this.history.slice(500);
    }
  }
}
```

---

## Monitoring Architecture

### Metrics Collection

```typescript
class MetricsCollector {
  recordLatency(operation: string, duration: number) { }
  recordThroughput(operation: string, count: number) { }
  recordError(operation: string, error: Error) { }
  recordMemoryUsage() { }
}

// Usage
const metrics = new MetricsCollector();

const start = Date.now();
const result = await operation();
metrics.recordLatency('operation', Date.now() - start);
```

### Logging Strategy

```typescript
// Structured logging
logger.info('Operation started', {
  operation: 'portfolio_optimization',
  algorithm: 'mean_variance',
  assets: assets.length,
  constraints: constraints
});

logger.error('Operation failed', {
  operation: 'portfolio_optimization',
  error: error.message,
  stack: error.stack
});
```

---

## Future Architecture Enhancements

1. **Distributed Coordination**: Multi-node swarm coordination via message queues
2. **GPU Acceleration**: CUDA kernels for optimization algorithms
3. **Real-time Streaming**: WebSocket integration for live data
4. **Federated Learning**: Privacy-preserving cross-organization learning
5. **Quantum Integration**: Hybrid quantum-classical algorithms

---

## References

- [Design Patterns](./DESIGN_PATTERNS.md)
- [Best Practices](./BEST_PRACTICES.md)
- [AgentDB Integration](./AGENTDB_GUIDE.md)
- [Swarm Coordination](./SWARM_PATTERNS.md)

---

Built with ❤️ by the Neural Trader team
