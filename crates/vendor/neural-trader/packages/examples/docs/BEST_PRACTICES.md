# Best Practices

Comprehensive guide to development best practices for Neural Trader examples.

## Table of Contents

- [Code Organization](#code-organization)
- [TypeScript Guidelines](#typescript-guidelines)
- [AgentDB Best Practices](#agentdb-best-practices)
- [Swarm Optimization](#swarm-optimization)
- [Performance Optimization](#performance-optimization)
- [Testing Strategy](#testing-strategy)
- [Error Handling](#error-handling)
- [Security](#security)
- [Documentation](#documentation)

---

## Code Organization

### File Structure

Follow consistent file organization:

```
src/
├── index.ts              # Public API exports
├── types.ts              # Shared types and interfaces
├── core-algorithm.ts     # Main algorithm implementation
├── self-learning.ts      # AgentDB integration
├── swarm-*.ts            # Swarm components
└── utils.ts              # Helper functions

tests/
├── unit/
│   ├── core-algorithm.test.ts
│   ├── self-learning.test.ts
│   └── swarm.test.ts
├── integration/
│   └── integration.test.ts
└── benchmarks/
    └── performance.bench.ts

examples/
├── basic-usage.ts        # Quick start example
└── advanced-*.ts         # Advanced demos
```

### Module Organization

```typescript
// ✅ GOOD: Single responsibility
export class OrderBookAnalyzer {
  analyzeOrderBook(orderBook: OrderBook): Metrics {
    // Only order book analysis logic
  }
}

export class PatternLearner {
  learnPattern(features: Features, outcome: Outcome): Promise<Pattern> {
    // Only learning logic
  }
}

// ❌ BAD: Mixed responsibilities
export class OrderBookAnalyzerAndLearner {
  analyzeAndLearn(orderBook: OrderBook, outcome: Outcome) {
    // Doing too much
  }
}
```

### Dependency Injection

```typescript
// ✅ GOOD: Inject dependencies
export class MarketMicrostructure {
  constructor(
    private analyzer: OrderBookAnalyzer,
    private learner: PatternLearner,
    private swarm?: SwarmFeatureEngineer
  ) {}
}

// ❌ BAD: Create dependencies internally
export class MarketMicrostructure {
  private analyzer = new OrderBookAnalyzer();
  private learner = new PatternLearner();
}
```

---

## TypeScript Guidelines

### Type Safety

```typescript
// ✅ GOOD: Explicit types
export interface OrderBook {
  bids: OrderBookLevel[];
  asks: OrderBookLevel[];
  timestamp: number;
  symbol: string;
}

export function analyzeOrderBook(orderBook: OrderBook): Metrics {
  // Type-safe implementation
}

// ❌ BAD: Using any
export function analyzeOrderBook(orderBook: any): any {
  // No type safety
}
```

### Generics

```typescript
// ✅ GOOD: Generic swarm components
export class SwarmOptimizer<T extends Agent> {
  async optimize(agents: T[], fitness: (agent: T) => number): Promise<T> {
    // Type-safe optimization
  }
}

// Usage with specific agent type
const optimizer = new SwarmOptimizer<TradingAgent>();
const best = await optimizer.optimize(agents, agent => agent.sharpeRatio);
```

### Discriminated Unions

```typescript
// ✅ GOOD: Type-safe pattern matching
export type OptimizationResult =
  | { status: 'success'; result: Portfolio; sharpeRatio: number }
  | { status: 'failure'; error: string; reason: string };

function handleResult(result: OptimizationResult) {
  if (result.status === 'success') {
    console.log(result.sharpeRatio); // Type-safe access
  } else {
    console.error(result.error); // Type-safe access
  }
}
```

### Strict Null Checks

```typescript
// ✅ GOOD: Handle null/undefined explicitly
function getPattern(id: string): Pattern | null {
  const pattern = this.patterns.get(id);
  return pattern ?? null;
}

const pattern = getPattern('123');
if (pattern) {
  console.log(pattern.confidence); // Type-safe
}

// ❌ BAD: Implicit null handling
function getPattern(id: string): Pattern {
  return this.patterns.get(id); // Might be undefined!
}
```

---

## AgentDB Best Practices

### Initialization

```typescript
// ✅ GOOD: Initialize once, check before use
export class SelfLearningOptimizer {
  private db: AgentDB;
  private initialized = false;

  async initialize(): Promise<void> {
    if (this.initialized) return;

    this.db = new AgentDB(this.dbPath);
    await this.db.initialize({
      indexType: 'hnsw',
      quantization: '8bit'
    });

    this.initialized = true;
  }

  async learn(data: TrajectoryData): Promise<void> {
    if (!this.initialized) {
      throw new Error('Not initialized. Call initialize() first.');
    }

    await this.db.storeTrajectory(data);
  }
}
```

### Memory Management

```typescript
// ✅ GOOD: Limit in-memory history, persist to AgentDB
export class PatternLearner {
  private maxInMemory = 1000;
  private patterns: Pattern[] = [];

  async learnPattern(features: Features, outcome: Outcome): Promise<Pattern> {
    const pattern = this.createPattern(features, outcome);

    // Store in AgentDB first
    await this.db.store({
      collection: 'patterns',
      data: pattern
    });

    // Add to in-memory cache
    this.patterns.push(pattern);

    // Limit in-memory size
    if (this.patterns.length > this.maxInMemory) {
      this.patterns = this.patterns.slice(-this.maxInMemory);
    }

    return pattern;
  }
}
```

### Similarity Search

```typescript
// ✅ GOOD: Use HNSW indexing for fast search
const similar = await this.db.similaritySearch({
  collection: 'patterns',
  query: currentFeatures,
  k: 10,
  indexType: 'hnsw' // 150x faster than brute force
});

// ❌ BAD: Brute force search on large datasets
const similar = await this.db.similaritySearch({
  collection: 'patterns',
  query: currentFeatures,
  k: 10
  // No indexType specified - slow on large datasets
});
```

### Quantization

```typescript
// ✅ GOOD: Use quantization for memory reduction
await this.db.initialize({
  quantization: '8bit' // 4-8x memory reduction
});

// For even more memory savings:
await this.db.initialize({
  quantization: '4bit' // 8-32x memory reduction
});
```

### Cleanup

```typescript
// ✅ GOOD: Always cleanup resources
export class Component {
  async close(): Promise<void> {
    if (this.db) {
      await this.db.close();
      this.db = null;
    }
  }
}

// Usage with try-finally
const component = new Component();
try {
  await component.initialize();
  await component.run();
} finally {
  await component.close();
}
```

---

## Swarm Optimization

### Agent Population

```typescript
// ✅ GOOD: Adjust swarm size based on problem complexity
function getSwarmSize(problemComplexity: number): number {
  if (problemComplexity < 100) return 20;
  if (problemComplexity < 1000) return 50;
  return 100;
}

const swarm = new SwarmFeatureEngineer({
  numAgents: getSwarmSize(features.length),
  generations: Math.min(100, features.length * 2)
});
```

### Diversity Management

```typescript
// ✅ GOOD: Maintain genetic diversity
export class SwarmOptimizer {
  private mutationRate = 0.2;
  private crossoverRate = 0.7;
  private eliteSize = 3;

  evolve(population: Agent[]): Agent[] {
    // Keep elite agents
    const elite = this.selectElite(population, this.eliteSize);

    // Generate offspring with diversity
    const offspring = [];
    while (offspring.length < population.length - this.eliteSize) {
      const parent1 = this.tournamentSelect(population);
      const parent2 = this.tournamentSelect(population);

      if (Math.random() < this.crossoverRate) {
        offspring.push(this.crossover(parent1, parent2));
      } else {
        offspring.push(parent1);
      }

      if (Math.random() < this.mutationRate) {
        offspring[offspring.length - 1] = this.mutate(offspring[offspring.length - 1]);
      }
    }

    return [...elite, ...offspring];
  }
}
```

### Parallel Evaluation

```typescript
// ✅ GOOD: Evaluate agents in parallel
async evaluatePopulation(agents: Agent[]): Promise<void> {
  await Promise.all(
    agents.map(agent => this.evaluateFitness(agent))
  );
}

// ❌ BAD: Sequential evaluation
async evaluatePopulation(agents: Agent[]): Promise<void> {
  for (const agent of agents) {
    await this.evaluateFitness(agent);
  }
}
```

### Early Stopping

```typescript
// ✅ GOOD: Stop when converged
export class SwarmOptimizer {
  private convergenceThreshold = 0.001;
  private noImprovementLimit = 10;

  async optimize(): Promise<Solution> {
    let bestFitness = -Infinity;
    let noImprovementCount = 0;

    for (let gen = 0; gen < this.maxGenerations; gen++) {
      const currentBest = this.getBestAgent();

      if (currentBest.fitness - bestFitness < this.convergenceThreshold) {
        noImprovementCount++;
      } else {
        noImprovementCount = 0;
        bestFitness = currentBest.fitness;
      }

      if (noImprovementCount >= this.noImprovementLimit) {
        console.log(`Converged at generation ${gen}`);
        break;
      }

      this.evolve();
    }

    return this.getBestAgent();
  }
}
```

---

## Performance Optimization

### Caching

```typescript
// ✅ GOOD: Cache expensive computations
export class OrderBookAnalyzer {
  private metricsCache = new Map<string, Metrics>();

  analyzeOrderBook(orderBook: OrderBook): Metrics {
    const key = this.getCacheKey(orderBook);

    if (this.metricsCache.has(key)) {
      return this.metricsCache.get(key)!;
    }

    const metrics = this.computeMetrics(orderBook);
    this.metricsCache.set(key, metrics);

    // Limit cache size
    if (this.metricsCache.size > 1000) {
      const firstKey = this.metricsCache.keys().next().value;
      this.metricsCache.delete(firstKey);
    }

    return metrics;
  }
}
```

### Batch Operations

```typescript
// ✅ GOOD: Batch database operations
async storeBatch(patterns: Pattern[]): Promise<void> {
  await this.db.storeBatch({
    collection: 'patterns',
    data: patterns
  });
}

// ❌ BAD: Individual operations
async storePatterns(patterns: Pattern[]): Promise<void> {
  for (const pattern of patterns) {
    await this.db.store({
      collection: 'patterns',
      data: pattern
    });
  }
}
```

### Lazy Loading

```typescript
// ✅ GOOD: Lazy initialization
export class MarketMicrostructure {
  private _swarm?: SwarmFeatureEngineer;

  get swarm(): SwarmFeatureEngineer {
    if (!this._swarm) {
      this._swarm = new SwarmFeatureEngineer(this.swarmConfig);
      this._swarm.initialize();
    }
    return this._swarm;
  }
}
```

### Memory Profiling

```typescript
// ✅ GOOD: Monitor memory usage
export class Component {
  private checkMemoryUsage(): void {
    const usage = process.memoryUsage();
    const heapUsedMB = usage.heapUsed / 1024 / 1024;

    if (heapUsedMB > 1000) {
      console.warn(`High memory usage: ${heapUsedMB.toFixed(2)}MB`);
      this.cleanup();
    }
  }

  private cleanup(): void {
    this.cache.clear();
    this.history = this.history.slice(-100);
    global.gc?.(); // Force garbage collection if available
  }
}
```

---

## Testing Strategy

### Unit Tests

```typescript
// ✅ GOOD: Test individual components
describe('OrderBookAnalyzer', () => {
  let analyzer: OrderBookAnalyzer;

  beforeEach(() => {
    analyzer = new OrderBookAnalyzer();
  });

  it('calculates bid-ask spread correctly', () => {
    const orderBook = createTestOrderBook({
      bestBid: 100.0,
      bestAsk: 100.1
    });

    const metrics = analyzer.analyzeOrderBook(orderBook);

    expect(metrics.bidAskSpread).toBe(0.1);
    expect(metrics.spreadBps).toBeCloseTo(10, 1);
  });

  it('handles empty order book', () => {
    const orderBook = createTestOrderBook({
      bids: [],
      asks: []
    });

    expect(() => analyzer.analyzeOrderBook(orderBook))
      .toThrow('Empty order book');
  });
});
```

### Integration Tests

```typescript
// ✅ GOOD: Test component interactions
describe('MarketMicrostructure Integration', () => {
  let mm: MarketMicrostructure;

  beforeAll(async () => {
    mm = await createMarketMicrostructure({
      agentDbPath: './test-memory.db',
      useSwarm: true
    });
  });

  afterAll(async () => {
    await mm.close();
    await fs.rm('./test-memory.db', { force: true });
  });

  it('analyzes order book and learns patterns', async () => {
    const orderBook = createTestOrderBook();
    const analysis = await mm.analyze(orderBook);

    expect(analysis.metrics).toBeDefined();
    expect(analysis.pattern).toBeDefined();

    // Learn from outcome
    await mm.learn({
      priceMove: 0.5,
      spreadChange: -0.01,
      liquidityChange: 0.1,
      timeHorizon: 5000
    });

    const stats = mm.getStatistics();
    expect(stats.learner.totalPatterns).toBeGreaterThan(0);
  });
});
```

### Performance Tests

```typescript
// ✅ GOOD: Benchmark performance
describe('Performance', () => {
  it('analyzes 1000 order books in <1s', async () => {
    const analyzer = new OrderBookAnalyzer();
    const orderBooks = Array.from({ length: 1000 }, () =>
      createTestOrderBook()
    );

    const start = Date.now();

    for (const orderBook of orderBooks) {
      analyzer.analyzeOrderBook(orderBook);
    }

    const duration = Date.now() - start;
    expect(duration).toBeLessThan(1000);
  });
});
```

### Mock External Dependencies

```typescript
// ✅ GOOD: Mock external APIs
jest.mock('openai', () => ({
  OpenAI: jest.fn().mockImplementation(() => ({
    chat: {
      completions: {
        create: jest.fn().mockResolvedValue({
          choices: [{ message: { content: 'Mocked response' } }]
        })
      }
    }
  }))
}));

describe('SwarmFeatureEngineer with OpenRouter', () => {
  it('gets anomaly explanation', async () => {
    const swarm = new SwarmFeatureEngineer({
      useOpenRouter: true,
      openRouterKey: 'test-key'
    });

    const anomaly = await swarm.detectAnomalies(metrics);
    expect(anomaly.explanation).toBe('Mocked response');
  });
});
```

---

## Error Handling

### Custom Error Types

```typescript
// ✅ GOOD: Specific error types
export class NeuralTraderError extends Error {
  constructor(message: string, public readonly cause?: Error) {
    super(message);
    this.name = 'NeuralTraderError';
  }
}

export class InitializationError extends NeuralTraderError {
  constructor(component: string, cause?: Error) {
    super(`Failed to initialize ${component}`, cause);
    this.name = 'InitializationError';
  }
}

export class OptimizationError extends NeuralTraderError {
  constructor(algorithm: string, cause?: Error) {
    super(`Optimization failed: ${algorithm}`, cause);
    this.name = 'OptimizationError';
  }
}
```

### Error Recovery

```typescript
// ✅ GOOD: Graceful degradation
export class Component {
  async operationWithFallback(): Promise<Result> {
    try {
      return await this.primaryOperation();
    } catch (error) {
      console.warn('Primary operation failed, using fallback', error);
      return await this.fallbackOperation();
    }
  }

  async retryableOperation(maxRetries = 3): Promise<Result> {
    let lastError: Error;

    for (let i = 0; i < maxRetries; i++) {
      try {
        return await this.operation();
      } catch (error) {
        lastError = error;
        console.warn(`Attempt ${i + 1} failed, retrying...`);
        await this.delay(Math.pow(2, i) * 1000); // Exponential backoff
      }
    }

    throw new NeuralTraderError('Max retries exceeded', lastError);
  }

  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}
```

### Validation

```typescript
// ✅ GOOD: Input validation
import { z } from 'zod';

const OrderBookSchema = z.object({
  bids: z.array(z.object({
    price: z.number().positive(),
    size: z.number().positive(),
    orders: z.number().int().positive()
  })).min(1),
  asks: z.array(z.object({
    price: z.number().positive(),
    size: z.number().positive(),
    orders: z.number().int().positive()
  })).min(1),
  timestamp: z.number().int().positive(),
  symbol: z.string().min(1)
});

export function validateOrderBook(data: unknown): OrderBook {
  try {
    return OrderBookSchema.parse(data);
  } catch (error) {
    throw new ValidationError('Invalid order book data', error);
  }
}
```

---

## Security

### API Key Management

```typescript
// ✅ GOOD: Environment variables
const openRouterKey = process.env.OPENROUTER_API_KEY;

if (!openRouterKey && config.useOpenRouter) {
  throw new Error(
    'OPENROUTER_API_KEY environment variable required when useOpenRouter is true'
  );
}

// ❌ BAD: Hardcoded keys
const openRouterKey = 'sk-or-v1-abc123'; // NEVER DO THIS!
```

### Input Sanitization

```typescript
// ✅ GOOD: Sanitize inputs
function sanitizeSymbol(symbol: string): string {
  return symbol.replace(/[^A-Z0-9]/gi, '').toUpperCase();
}

// ❌ BAD: Direct use of user input
const query = `SELECT * FROM assets WHERE symbol = '${userInput}'`;
```

### Rate Limiting

```typescript
// ✅ GOOD: Rate limit external API calls
export class RateLimiter {
  private requests: number[] = [];
  private limit: number;
  private window: number;

  constructor(limit: number, window: number) {
    this.limit = limit;
    this.window = window;
  }

  async checkLimit(): Promise<void> {
    const now = Date.now();
    this.requests = this.requests.filter(time => now - time < this.window);

    if (this.requests.length >= this.limit) {
      const oldestRequest = Math.min(...this.requests);
      const waitTime = this.window - (now - oldestRequest);
      await this.delay(waitTime);
    }

    this.requests.push(now);
  }

  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}
```

---

## Documentation

### JSDoc Comments

```typescript
/**
 * Analyzes order book to compute comprehensive microstructure metrics.
 *
 * @param orderBook - Order book snapshot with bids, asks, and metadata
 * @param recentTrades - Optional array of recent trades for flow analysis
 * @returns Comprehensive metrics including spread, depth, toxicity, and liquidity
 *
 * @example
 * ```typescript
 * const analyzer = new OrderBookAnalyzer();
 * const metrics = analyzer.analyzeOrderBook({
 *   bids: [{ price: 100, size: 1000, orders: 10 }],
 *   asks: [{ price: 101, size: 1000, orders: 10 }],
 *   timestamp: Date.now(),
 *   symbol: 'BTCUSD'
 * });
 * console.log('Spread:', metrics.bidAskSpread);
 * ```
 *
 * @throws {Error} If order book is empty or invalid
 *
 * @see {@link MicrostructureMetrics} for metric definitions
 * @see {@link OrderBook} for input format
 */
export function analyzeOrderBook(
  orderBook: OrderBook,
  recentTrades?: OrderFlow[]
): MicrostructureMetrics {
  // Implementation
}
```

### README Structure

```markdown
# Package Name

Brief description (1-2 sentences).

## Features

- Feature 1
- Feature 2

## Installation

\```bash
npm install
\```

## Quick Start

\```typescript
// Minimal working example
\```

## Usage

### Basic Usage

\```typescript
// Common use case
\```

### Advanced Usage

\```typescript
// Complex scenario
\```

## API Reference

### Class: Component

#### Methods

##### method()

Description

**Parameters**:
- `param1` (type): Description

**Returns**: Description

**Example**:
\```typescript
// Usage example
\```

## Testing

\```bash
npm test
\```

## License

MIT
```

---

## References

- [Architecture Overview](./ARCHITECTURE.md)
- [Integration Guide](./INTEGRATION_GUIDE.md)
- [AgentDB Guide](./AGENTDB_GUIDE.md)
- [Swarm Patterns](./SWARM_PATTERNS.md)

---

Built with ❤️ by the Neural Trader team
