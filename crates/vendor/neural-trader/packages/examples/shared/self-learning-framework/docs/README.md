# Self-Learning Framework

Unified self-learning framework with AgentDB integration for experience replay, pattern recognition, and adaptive parameter tuning.

## Features

- **Experience Replay**: Store and retrieve past experiences with AgentDB vector search
- **Pattern Learning**: Automatic pattern recognition from historical data
- **Adaptive Parameters**: Self-tuning parameters based on performance feedback
- **Memory Persistence**: Cross-session learning with AgentDB
- **Transfer Learning**: Apply learned patterns to new scenarios

## Installation

```bash
npm install @neural-trader/self-learning-framework
```

## Quick Start

```typescript
import { createSelfLearningSystem } from '@neural-trader/self-learning-framework';

// Create self-learning system
const system = createSelfLearningSystem({
  replay: {
    maxSize: 10000,
    prioritization: 'prioritized',
  },
  learner: {
    minOccurrences: 5,
    similarityThreshold: 0.8,
    confidenceThreshold: 0.7,
  },
  adaptation: {
    learningRate: 0.1,
    adaptationInterval: 100,
    explorationRate: 0.3,
    decayRate: 0.995,
    minExplorationRate: 0.05,
  },
});

// Register parameters to adapt
system.adaptation.registerParameters([
  {
    name: 'threshold',
    type: 'continuous',
    range: [0.1, 0.9],
    default: 0.5,
    current: 0.5,
  },
  {
    name: 'windowSize',
    type: 'discrete',
    values: [10, 20, 30, 40, 50],
    default: 20,
    current: 20,
  },
]);

// Execute and learn
async function executeStrategy() {
  const params = system.getParameters();
  const result = await runStrategy(params);

  // Record experience
  await system.learn({
    id: `exp-${Date.now()}`,
    timestamp: new Date(),
    state: { marketCondition: 'bullish', volatility: 0.15 },
    action: params,
    result: result,
    reward: result.profitLoss,
  });
}

// Run multiple iterations
for (let i = 0; i < 1000; i++) {
  await executeStrategy();
}

// Get learned patterns
const patterns = system.learner.getTopPatterns(10);
console.log('Top patterns:', patterns);
```

## Experience Replay

### Store Experiences

```typescript
import { ExperienceReplay } from '@neural-trader/self-learning-framework';

const replay = new ExperienceReplay({
  maxSize: 10000,
  prioritization: 'prioritized',
  dbPath: './data/experiences',
  namespace: 'trading-bot',
});

// Store single experience
await replay.store({
  id: 'exp-1',
  timestamp: new Date(),
  state: { price: 150.0, volume: 1000000 },
  action: { signal: 'buy', quantity: 100 },
  result: { executed: true, fillPrice: 150.5 },
  reward: 50.0,
  metadata: { strategy: 'momentum' },
});

// Store batch
await replay.storeBatch(experiences);
```

### Sample Experiences

```typescript
// Uniform sampling
const batch = await replay.sample(32);

// Prioritized sampling (higher reward = higher probability)
const prioritizedBatch = await replay.samplePrioritized(32, 0.6);

// Query similar experiences
const similar = await replay.querySimilar(currentExperience, 10);

// Get high-reward experiences
const successful = await replay.getHighReward(100);
```

### Statistics

```typescript
const stats = replay.getStats();
console.log(`Buffer: ${stats.size}/${stats.maxSize}`);
console.log(`Utilization: ${stats.utilizationPercent.toFixed(1)}%`);
```

## Pattern Learning

### Learn from Experience

```typescript
import { PatternLearner } from '@neural-trader/self-learning-framework';

const learner = new PatternLearner(replay, {
  minOccurrences: 5,
  similarityThreshold: 0.8,
  confidenceThreshold: 0.7,
});

// Learn patterns from replay buffer
const newPatterns = await learner.learnPatterns();
console.log(`Learned ${newPatterns.length} new patterns`);

// Get top patterns
const topPatterns = learner.getTopPatterns(10);
topPatterns.forEach(pattern => {
  console.log(`${pattern.name}:`);
  console.log(`  Success rate: ${(pattern.successRate * 100).toFixed(1)}%`);
  console.log(`  Average reward: ${pattern.avgReward.toFixed(2)}`);
  console.log(`  Confidence: ${(pattern.confidence * 100).toFixed(1)}%`);
});
```

### Match Patterns

```typescript
// Match current state against known patterns
const matches = await learner.matchPatterns(currentState, 5);

if (matches.length > 0) {
  const best = matches[0];
  console.log(`Matched pattern: ${best.pattern.name}`);
  console.log(`Similarity: ${(best.similarity * 100).toFixed(1)}%`);
  console.log(`Expected reward: ${best.pattern.avgReward.toFixed(2)}`);

  // Use pattern template
  const recommendedAction = best.pattern.template;
}
```

### Update Patterns

```typescript
// Update pattern with new occurrence
await learner.updatePattern('pattern-123', reward, success);

// Prune low-confidence patterns
const pruned = await learner.prunePatterns(0.5);
console.log(`Removed ${pruned} low-confidence patterns`);
```

### Export/Import

```typescript
// Export patterns
const patterns = learner.export();
fs.writeFileSync('patterns.json', JSON.stringify(patterns));

// Import patterns
const loadedPatterns = JSON.parse(fs.readFileSync('patterns.json'));
await learner.import(loadedPatterns);
```

## Adaptive Parameters

### Register Parameters

```typescript
import { AdaptiveParameters } from '@neural-trader/self-learning-framework';

const adaptation = new AdaptiveParameters(replay, learner, {
  learningRate: 0.1,
  adaptationInterval: 100,
  explorationRate: 0.3,
  decayRate: 0.995,
  minExplorationRate: 0.05,
});

// Register continuous parameter
adaptation.registerParameter({
  name: 'riskLevel',
  type: 'continuous',
  range: [0.1, 0.9],
  default: 0.5,
  current: 0.5,
});

// Register discrete parameter
adaptation.registerParameter({
  name: 'lookbackPeriod',
  type: 'discrete',
  values: [5, 10, 15, 20, 25, 30],
  default: 20,
  current: 20,
});

// Register categorical parameter
adaptation.registerParameter({
  name: 'strategy',
  type: 'categorical',
  values: ['momentum', 'mean-reversion', 'breakout'],
  default: 'momentum',
  current: 'momentum',
});
```

### Automatic Adaptation

```typescript
// Record experiences and adapt automatically
for (const experience of experiences) {
  const adapted = await adaptation.recordExperience(experience);

  if (adapted) {
    console.log('Parameters adapted!');
    console.log('New parameters:', adaptation.getAllParameters());
    console.log('Exploration rate:', adaptation.getExplorationRate());
  }
}
```

### Manual Adaptation

```typescript
// Trigger adaptation manually
await adaptation.adapt();

// Get current parameters
const params = adaptation.getAllParameters();
console.log('Risk level:', params.riskLevel);
console.log('Lookback period:', params.lookbackPeriod);
console.log('Strategy:', params.strategy);
```

### Control Exploration

```typescript
// Increase exploration
adaptation.setExplorationRate(0.5);

// Decrease exploration
adaptation.setExplorationRate(0.1);

// Reset to default
adaptation.resetAllParameters();
```

### Statistics

```typescript
const stats = adaptation.getStats();
console.log('Parameters:', stats.totalParameters);
console.log('Exploration rate:', stats.explorationRate.toFixed(3));
console.log('Avg performance:', stats.avgPerformance.toFixed(3));
console.log('Trend:', stats.performanceTrend);
```

## Complete Trading Bot Example

```typescript
import { createSelfLearningSystem } from '@neural-trader/self-learning-framework';

// Create system
const system = createSelfLearningSystem({
  replay: {
    maxSize: 50000,
    prioritization: 'prioritized',
    dbPath: './data/trading-bot',
  },
  learner: {
    minOccurrences: 10,
    similarityThreshold: 0.85,
    confidenceThreshold: 0.75,
  },
  adaptation: {
    learningRate: 0.05,
    adaptationInterval: 50,
    explorationRate: 0.2,
    decayRate: 0.998,
    minExplorationRate: 0.01,
  },
});

// Register trading parameters
system.adaptation.registerParameters([
  { name: 'entryThreshold', type: 'continuous', range: [0.01, 0.1], default: 0.05, current: 0.05 },
  { name: 'exitThreshold', type: 'continuous', range: [0.02, 0.15], default: 0.08, current: 0.08 },
  { name: 'stopLoss', type: 'continuous', range: [0.01, 0.05], default: 0.02, current: 0.02 },
  { name: 'positionSize', type: 'discrete', values: [0.1, 0.2, 0.3, 0.4, 0.5], default: 0.2, current: 0.2 },
]);

// Trading loop
async function tradingLoop() {
  while (true) {
    // Get current market state
    const marketState = await getMarketState();

    // Check if similar patterns exist
    const matches = await system.learner.matchPatterns(marketState, 3);

    // Get adaptive parameters
    const params = system.getParameters();

    // Generate trading signal
    const signal = generateSignal(marketState, matches, params);

    // Execute trade
    const result = await executeTrade(signal);

    // Calculate reward (profit/loss + risk-adjusted return)
    const reward = calculateReward(result, params);

    // Record experience and learn
    await system.learn({
      id: `trade-${Date.now()}`,
      timestamp: new Date(),
      state: marketState,
      action: { signal, params },
      result: result,
      reward: reward,
      metadata: {
        matchedPatterns: matches.map(m => m.pattern.id),
      },
    });

    // Periodic pattern learning
    if (Math.random() < 0.1) { // 10% chance each iteration
      const newPatterns = await system.learner.learnPatterns();
      if (newPatterns.length > 0) {
        console.log(`🧠 Learned ${newPatterns.length} new patterns`);
      }

      // Prune low-confidence patterns
      await system.learner.prunePatterns();
    }

    await sleep(60000); // Wait 1 minute
  }
}

// Save system state periodically
setInterval(async () => {
  const state = await system.exportState();
  fs.writeFileSync('bot-state.json', JSON.stringify(state));
  console.log('💾 System state saved');
}, 3600000); // Every hour

// Load previous state on startup
if (fs.existsSync('bot-state.json')) {
  const state = JSON.parse(fs.readFileSync('bot-state.json'));
  await system.importState(state);
  console.log('📂 Previous state restored');
}

// Start trading
tradingLoop();
```

## API Reference

### ExperienceReplay

```typescript
class ExperienceReplay<T> {
  constructor(config: ReplayConfig);
  store(experience: Experience): Promise<void>;
  storeBatch(experiences: Experience[]): Promise<void>;
  sample(batchSize: number): Promise<ReplayBatch>;
  samplePrioritized(batchSize: number, alpha?: number): Promise<ReplayBatch>;
  querySimilar(experience: Experience, k: number): Promise<Experience[]>;
  getHighReward(threshold: number): Promise<Experience[]>;
  export(): Promise<Experience[]>;
  import(experiences: Experience[]): Promise<void>;
}
```

### PatternLearner

```typescript
class PatternLearner<T> {
  constructor(replay: ExperienceReplay, config: LearnerConfig);
  learnPatterns(): Promise<Pattern<T>[]>;
  matchPatterns(currentState: any, k?: number): Promise<PatternMatch<T>[]>;
  updatePattern(patternId: string, reward: number, success: boolean): Promise<void>;
  prunePatterns(minConfidence?: number): Promise<number>;
  getTopPatterns(k?: number): Pattern<T>[];
  export(): Pattern<T>[];
  import(patterns: Pattern<T>[]): Promise<void>;
}
```

### AdaptiveParameters

```typescript
class AdaptiveParameters {
  constructor(replay: ExperienceReplay, learner: PatternLearner, config: AdaptationConfig);
  registerParameter(parameter: Parameter): void;
  getParameter(name: string): any;
  getAllParameters(): Record<string, any>;
  recordExperience(experience: Experience): Promise<boolean>;
  adapt(): Promise<void>;
  setExplorationRate(rate: number): void;
  resetAllParameters(): void;
  export(): { parameters: Record<string, any>; config: AdaptationConfig; performanceHistory: number[] };
  import(data: ReturnType<AdaptiveParameters['export']>): void;
}
```

## Best Practices

1. **Start with large replay buffers** (10,000+ experiences)
2. **Use prioritized sampling** for faster learning
3. **Set appropriate similarity thresholds** (0.7-0.9)
4. **Decay exploration rate** over time
5. **Prune patterns regularly** to remove outdated ones
6. **Save system state** frequently
7. **Monitor performance trends** for degradation
8. **Use transfer learning** across similar tasks

## Performance Tips

- AgentDB provides 150x faster vector search than naive approaches
- Batch operations when possible (storeBatch vs individual store)
- Adjust buffer size based on available memory
- Use continuous parameters for smooth optimization surfaces
- Implement custom reward functions for better learning signals

## License

MIT
