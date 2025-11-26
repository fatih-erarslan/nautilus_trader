# Design Patterns

Common design patterns used across Neural Trader examples.

## Table of Contents

- [Creational Patterns](#creational-patterns)
- [Structural Patterns](#structural-patterns)
- [Behavioral Patterns](#behavioral-patterns)
- [Concurrency Patterns](#concurrency-patterns)
- [Neural Trader Specific Patterns](#neural-trader-specific-patterns)

---

## Creational Patterns

### Factory Pattern

Create objects without specifying exact classes.

```typescript
// Factory for creating optimizers
export class OptimizerFactory {
  static create(type: string, assets: Asset[], correlationMatrix: number[][]): Optimizer {
    switch (type) {
      case 'mean-variance':
        return new MeanVarianceOptimizer(assets, correlationMatrix);
      case 'risk-parity':
        return new RiskParityOptimizer(assets, correlationMatrix);
      case 'black-litterman':
        return new BlackLittermanOptimizer(assets, correlationMatrix, marketCapWeights);
      case 'multi-objective':
        return new MultiObjectiveOptimizer(assets, correlationMatrix, historicalReturns);
      default:
        throw new Error(`Unknown optimizer type: ${type}`);
    }
  }
}

// Usage
const optimizer = OptimizerFactory.create('mean-variance', assets, correlationMatrix);
```

### Builder Pattern

Construct complex objects step by step.

```typescript
// Builder for swarm configuration
export class SwarmBuilder {
  private config: Partial<SwarmConfig> = {};

  withAgents(numAgents: number): this {
    this.config.numAgents = numAgents;
    return this;
  }

  withGenerations(generations: number): this {
    this.config.generations = generations;
    return this;
  }

  withMutation(rate: number): this {
    this.config.mutationRate = rate;
    return this;
  }

  withOpenRouter(apiKey: string): this {
    this.config.useOpenRouter = true;
    this.config.openRouterKey = apiKey;
    return this;
  }

  build(): SwarmConfig {
    return {
      numAgents: this.config.numAgents ?? 30,
      generations: this.config.generations ?? 50,
      mutationRate: this.config.mutationRate ?? 0.2,
      crossoverRate: this.config.crossoverRate ?? 0.7,
      eliteSize: this.config.eliteSize ?? 3,
      useOpenRouter: this.config.useOpenRouter ?? false,
      openRouterKey: this.config.openRouterKey
    };
  }
}

// Usage
const config = new SwarmBuilder()
  .withAgents(50)
  .withGenerations(100)
  .withMutation(0.3)
  .withOpenRouter(process.env.OPENROUTER_API_KEY)
  .build();

const swarm = new SwarmFeatureEngineer(config);
```

### Singleton Pattern

Ensure a class has only one instance.

```typescript
// Singleton for AgentDB connection
export class AgentDBConnection {
  private static instance: AgentDB;
  private static dbPath: string;

  private constructor() {}

  static async getInstance(dbPath: string = './memory.db'): Promise<AgentDB> {
    if (!this.instance || this.dbPath !== dbPath) {
      this.instance = new AgentDB(dbPath);
      await this.instance.initialize();
      this.dbPath = dbPath;
    }
    return this.instance;
  }

  static async close(): Promise<void> {
    if (this.instance) {
      await this.instance.close();
      this.instance = null;
    }
  }
}

// Usage
const db1 = await AgentDBConnection.getInstance();
const db2 = await AgentDBConnection.getInstance(); // Same instance
console.log(db1 === db2); // true
```

---

## Structural Patterns

### Adapter Pattern

Convert interface of a class into another interface.

```typescript
// Adapter for external trading API
export class BinanceAdapter implements TradingPlatformInterface {
  private exchange: ccxt.Exchange;

  constructor(apiKey: string, secret: string) {
    this.exchange = new ccxt.binance({ apiKey, secret });
  }

  async fetchOrderBook(symbol: string): Promise<OrderBook> {
    const externalOrderBook = await this.exchange.fetchOrderBook(symbol);

    // Adapt to our format
    return {
      bids: externalOrderBook.bids.map(([price, size]) => ({
        price,
        size,
        orders: 1
      })),
      asks: externalOrderBook.asks.map(([price, size]) => ({
        price,
        size,
        orders: 1
      })),
      timestamp: Date.now(),
      symbol
    };
  }
}

// Usage
const adapter = new BinanceAdapter(apiKey, secret);
const orderBook = await adapter.fetchOrderBook('BTC/USD');
const metrics = analyzer.analyzeOrderBook(orderBook);
```

### Decorator Pattern

Add responsibilities to objects dynamically.

```typescript
// Decorator for caching
export class CachedOptimizer implements Optimizer {
  private cache = new Map<string, OptimizationResult>();

  constructor(private optimizer: Optimizer) {}

  optimize(constraints: Constraints): OptimizationResult {
    const key = JSON.stringify(constraints);

    if (this.cache.has(key)) {
      return this.cache.get(key)!;
    }

    const result = this.optimizer.optimize(constraints);
    this.cache.set(key, result);
    return result;
  }
}

// Decorator for logging
export class LoggedOptimizer implements Optimizer {
  constructor(private optimizer: Optimizer) {}

  optimize(constraints: Constraints): OptimizationResult {
    console.log('Optimization started', constraints);
    const start = Date.now();

    const result = this.optimizer.optimize(constraints);

    console.log(`Optimization completed in ${Date.now() - start}ms`, result);
    return result;
  }
}

// Usage with multiple decorators
let optimizer: Optimizer = new MeanVarianceOptimizer(assets, correlationMatrix);
optimizer = new CachedOptimizer(optimizer);
optimizer = new LoggedOptimizer(optimizer);

const result = optimizer.optimize(constraints);
```

### Proxy Pattern

Provide a surrogate to control access to an object.

```typescript
// Lazy loading proxy
export class LazySwarmProxy implements SwarmInterface {
  private _swarm?: SwarmFeatureEngineer;

  constructor(private config: SwarmConfig) {}

  private get swarm(): SwarmFeatureEngineer {
    if (!this._swarm) {
      console.log('Initializing swarm...');
      this._swarm = new SwarmFeatureEngineer(this.config);
    }
    return this._swarm;
  }

  async exploreFeatures(data: any[]): Promise<FeatureSet[]> {
    return await this.swarm.exploreFeatures(data);
  }

  async detectAnomalies(metrics: any): Promise<AnomalyResult> {
    return await this.swarm.detectAnomalies(metrics);
  }
}

// Usage
const swarm = new LazySwarmProxy(config); // Not initialized yet
const features = await swarm.exploreFeatures(data); // Initialized on first use
```

---

## Behavioral Patterns

### Strategy Pattern

Define a family of algorithms, encapsulate each one, and make them interchangeable.

```typescript
// Strategy interface
export interface OptimizationStrategy {
  optimize(assets: Asset[], constraints: Constraints): OptimizationResult;
}

// Concrete strategies
export class MeanVarianceStrategy implements OptimizationStrategy {
  optimize(assets: Asset[], constraints: Constraints): OptimizationResult {
    // Mean-Variance implementation
  }
}

export class RiskParityStrategy implements OptimizationStrategy {
  optimize(assets: Asset[], constraints: Constraints): OptimizationResult {
    // Risk Parity implementation
  }
}

// Context
export class PortfolioOptimizer {
  constructor(private strategy: OptimizationStrategy) {}

  setStrategy(strategy: OptimizationStrategy): void {
    this.strategy = strategy;
  }

  optimize(assets: Asset[], constraints: Constraints): OptimizationResult {
    return this.strategy.optimize(assets, constraints);
  }
}

// Usage
const optimizer = new PortfolioOptimizer(new MeanVarianceStrategy());
let result = optimizer.optimize(assets, constraints);

// Switch strategy at runtime
optimizer.setStrategy(new RiskParityStrategy());
result = optimizer.optimize(assets, constraints);
```

### Observer Pattern

Define one-to-many dependency for notifications.

```typescript
// Observer interface
export interface PatternObserver {
  onPatternLearned(pattern: Pattern): void;
  onAnomalyDetected(anomaly: Anomaly): void;
}

// Subject
export class PatternLearner {
  private observers: PatternObserver[] = [];

  subscribe(observer: PatternObserver): void {
    this.observers.push(observer);
  }

  unsubscribe(observer: PatternObserver): void {
    const index = this.observers.indexOf(observer);
    if (index > -1) {
      this.observers.splice(index, 1);
    }
  }

  private notifyPatternLearned(pattern: Pattern): void {
    this.observers.forEach(observer => observer.onPatternLearned(pattern));
  }

  private notifyAnomalyDetected(anomaly: Anomaly): void {
    this.observers.forEach(observer => observer.onAnomalyDetected(anomaly));
  }

  async learnPattern(features: Features, outcome: Outcome): Promise<Pattern> {
    const pattern = await this.createPattern(features, outcome);
    this.notifyPatternLearned(pattern);
    return pattern;
  }
}

// Observer implementation
export class TradingBot implements PatternObserver {
  onPatternLearned(pattern: Pattern): void {
    console.log('New pattern learned:', pattern.label);
    this.adjustStrategy(pattern);
  }

  onAnomalyDetected(anomaly: Anomaly): void {
    console.warn('Anomaly detected:', anomaly.type);
    this.reduceRisk();
  }
}

// Usage
const learner = new PatternLearner();
const bot = new TradingBot();
learner.subscribe(bot);
```

### Template Method Pattern

Define skeleton of algorithm, deferring some steps to subclasses.

```typescript
// Abstract base class
export abstract class BaseOptimizer {
  // Template method
  optimize(assets: Asset[], constraints: Constraints): OptimizationResult {
    this.validateInputs(assets, constraints);

    const initialWeights = this.generateInitialWeights(assets.length);
    const optimizedWeights = this.optimizeWeights(initialWeights, assets, constraints);
    const metrics = this.calculateMetrics(optimizedWeights, assets);

    this.logResults(metrics);

    return { weights: optimizedWeights, ...metrics };
  }

  // Common implementation
  private validateInputs(assets: Asset[], constraints: Constraints): void {
    if (assets.length === 0) {
      throw new Error('No assets provided');
    }
    // More validation...
  }

  private logResults(metrics: Metrics): void {
    console.log('Optimization complete:', metrics);
  }

  // Abstract methods - subclasses must implement
  protected abstract generateInitialWeights(numAssets: number): number[];
  protected abstract optimizeWeights(
    initial: number[],
    assets: Asset[],
    constraints: Constraints
  ): number[];
  protected abstract calculateMetrics(weights: number[], assets: Asset[]): Metrics;
}

// Concrete implementation
export class MeanVarianceOptimizer extends BaseOptimizer {
  protected generateInitialWeights(numAssets: number): number[] {
    return Array(numAssets).fill(1 / numAssets);
  }

  protected optimizeWeights(
    initial: number[],
    assets: Asset[],
    constraints: Constraints
  ): number[] {
    // Mean-Variance specific optimization
  }

  protected calculateMetrics(weights: number[], assets: Asset[]): Metrics {
    // Calculate Sharpe ratio, risk, return, etc.
  }
}
```

### Chain of Responsibility Pattern

Pass requests along a chain of handlers.

```typescript
// Handler interface
export abstract class AnomalyDetector {
  protected next?: AnomalyDetector;

  setNext(detector: AnomalyDetector): AnomalyDetector {
    this.next = detector;
    return detector;
  }

  async detect(data: any): Promise<AnomalyResult | null> {
    const result = await this.doDetect(data);

    if (result) {
      return result;
    }

    if (this.next) {
      return await this.next.detect(data);
    }

    return null;
  }

  protected abstract doDetect(data: any): Promise<AnomalyResult | null>;
}

// Concrete handlers
export class StatisticalAnomalyDetector extends AnomalyDetector {
  protected async doDetect(data: any): Promise<AnomalyResult | null> {
    const zScore = this.calculateZScore(data);

    if (Math.abs(zScore) > 3) {
      return {
        isAnomaly: true,
        type: 'statistical',
        confidence: Math.min(Math.abs(zScore) / 5, 1),
        explanation: `Z-score: ${zScore.toFixed(2)}`
      };
    }

    return null;
  }
}

export class MLAnomalyDetector extends AnomalyDetector {
  protected async doDetect(data: any): Promise<AnomalyResult | null> {
    const score = await this.mlModel.predict(data);

    if (score > 0.9) {
      return {
        isAnomaly: true,
        type: 'ml-based',
        confidence: score,
        explanation: 'ML model detected anomaly'
      };
    }

    return null;
  }
}

// Usage
const detector = new StatisticalAnomalyDetector();
detector.setNext(new MLAnomalyDetector());

const result = await detector.detect(data);
```

---

## Concurrency Patterns

### Promise Pool Pattern

Limit concurrent async operations.

```typescript
export class PromisePool<T> {
  constructor(private concurrency: number) {}

  async map<R>(
    items: T[],
    fn: (item: T) => Promise<R>
  ): Promise<R[]> {
    const results: R[] = [];
    const executing: Promise<void>[] = [];

    for (const item of items) {
      const promise = fn(item).then(result => {
        results.push(result);
      });

      executing.push(promise);

      if (executing.length >= this.concurrency) {
        await Promise.race(executing);
        executing.splice(
          executing.findIndex(p => p === promise),
          1
        );
      }
    }

    await Promise.all(executing);
    return results;
  }
}

// Usage
const pool = new PromisePool<Agent>(10); // Max 10 concurrent

const results = await pool.map(agents, async agent => {
  return await evaluateAgent(agent);
});
```

### Producer-Consumer Pattern

Decouple production and consumption of data.

```typescript
export class DataQueue<T> {
  private queue: T[] = [];
  private consumers: Array<(item: T) => Promise<void>> = [];

  async produce(item: T): Promise<void> {
    this.queue.push(item);

    if (this.consumers.length > 0) {
      const consumer = this.consumers.shift()!;
      const item = this.queue.shift()!;
      await consumer(item);
    }
  }

  async consume(): Promise<T> {
    if (this.queue.length > 0) {
      return this.queue.shift()!;
    }

    return new Promise<T>(resolve => {
      this.consumers.push(async (item: T) => {
        resolve(item);
      });
    });
  }
}

// Usage
const queue = new DataQueue<OrderBook>();

// Producer
setInterval(() => {
  const orderBook = fetchOrderBook();
  queue.produce(orderBook);
}, 1000);

// Consumer
while (true) {
  const orderBook = await queue.consume();
  await processOrderBook(orderBook);
}
```

---

## Neural Trader Specific Patterns

### Self-Learning Pattern

Components that learn from experience via AgentDB.

```typescript
export class SelfLearningComponent {
  private db: AgentDB;
  private memory: Map<string, any> = new Map();

  async initialize(): Promise<void> {
    this.db = new AgentDB(this.dbPath);
    await this.db.initialize();

    // Load previous learnings
    await this.loadMemory();
  }

  async learn(experience: Experience): Promise<void> {
    // Store in AgentDB
    await this.db.storeTrajectory({
      state: experience.state,
      action: experience.action,
      reward: experience.reward,
      nextState: experience.nextState
    });

    // Update in-memory cache
    this.memory.set(experience.id, experience);

    // Train RL model
    await this.trainModel();
  }

  async recall(query: any): Promise<Experience[]> {
    // Search similar experiences
    const similar = await this.db.similaritySearch({
      collection: 'experiences',
      query,
      k: 10
    });

    return similar;
  }

  private async loadMemory(): Promise<void> {
    const stored = await this.db.query({
      collection: 'experiences',
      limit: 1000
    });

    stored.forEach(exp => {
      this.memory.set(exp.id, exp);
    });
  }

  private async trainModel(): Promise<void> {
    await this.db.trainRL({
      algorithm: 'decision_transformer',
      trajectories: Array.from(this.memory.values())
    });
  }
}
```

### Swarm Coordination Pattern

Multi-agent systems with genetic evolution.

```typescript
export class SwarmCoordinator<T extends Agent> {
  private population: T[] = [];

  async initialize(populationSize: number): Promise<void> {
    this.population = Array.from({ length: populationSize }, () =>
      this.createRandomAgent()
    );

    await this.evaluatePopulation();
  }

  async evolve(generations: number): Promise<T> {
    for (let gen = 0; gen < generations; gen++) {
      // Selection
      const parents = this.selectParents();

      // Crossover
      const offspring = this.crossover(parents);

      // Mutation
      this.mutate(offspring);

      // Evaluation
      await this.evaluatePopulation();

      // Replacement
      this.replaceWorst(offspring);

      // Report progress
      if (gen % 10 === 0) {
        this.reportProgress(gen);
      }
    }

    return this.getBestAgent();
  }

  private async evaluatePopulation(): Promise<void> {
    await Promise.all(
      this.population.map(agent => this.evaluateFitness(agent))
    );
  }

  private selectParents(): T[] {
    // Tournament selection
    return this.population
      .sort((a, b) => b.fitness - a.fitness)
      .slice(0, Math.floor(this.population.length * 0.3));
  }

  private crossover(parents: T[]): T[] {
    const offspring: T[] = [];

    for (let i = 0; i < parents.length - 1; i += 2) {
      const child1 = this.crossoverPair(parents[i], parents[i + 1]);
      const child2 = this.crossoverPair(parents[i + 1], parents[i]);
      offspring.push(child1, child2);
    }

    return offspring;
  }

  private mutate(offspring: T[]): void {
    offspring.forEach(agent => {
      if (Math.random() < this.mutationRate) {
        this.mutateAgent(agent);
      }
    });
  }

  protected abstract createRandomAgent(): T;
  protected abstract evaluateFitness(agent: T): Promise<void>;
  protected abstract crossoverPair(parent1: T, parent2: T): T;
  protected abstract mutateAgent(agent: T): void;
}
```

### Pipeline Pattern

Sequential processing with stages.

```typescript
export interface PipelineStage<T, R> {
  name: string;
  process(input: T): Promise<R>;
}

export class Pipeline<T> {
  private stages: PipelineStage<any, any>[] = [];

  addStage<R>(stage: PipelineStage<T, R>): Pipeline<R> {
    this.stages.push(stage);
    return this as any;
  }

  async execute(input: T): Promise<any> {
    let result: any = input;

    for (const stage of this.stages) {
      console.log(`Executing stage: ${stage.name}`);
      result = await stage.process(result);
    }

    return result;
  }
}

// Usage
const pipeline = new Pipeline<OrderBook>()
  .addStage({
    name: 'Analysis',
    process: async (orderBook) => analyzer.analyzeOrderBook(orderBook)
  })
  .addStage({
    name: 'Pattern Recognition',
    process: async (metrics) => learner.recognizePattern(metrics)
  })
  .addStage({
    name: 'Anomaly Detection',
    process: async (pattern) => swarm.detectAnomalies(pattern)
  });

const result = await pipeline.execute(orderBook);
```

---

## References

- [Architecture Overview](./ARCHITECTURE.md)
- [Best Practices](./BEST_PRACTICES.md)
- [Integration Guide](./INTEGRATION_GUIDE.md)

---

Built with ❤️ by the Neural Trader team
