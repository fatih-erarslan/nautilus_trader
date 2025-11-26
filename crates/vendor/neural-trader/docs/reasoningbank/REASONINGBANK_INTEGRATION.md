# ReasoningBank Learning Integration for E2B Trading Swarms

Complete adaptive learning system that learns from trading experience and continuously improves agent performance.

## Overview

ReasoningBank provides E2B trading swarms with:

- **Trajectory Tracking**: Records all trading decisions with full context
- **Verdict Judgment**: Evaluates decision quality based on multiple factors
- **Memory Distillation**: Learns successful patterns and compresses knowledge
- **Pattern Recognition**: 150x faster vector similarity search via AgentDB
- **Strategy Adaptation**: Continuously adapts agent behavior based on learnings

## Architecture

```
ReasoningBankSwarmLearner
├── TrajectoryTracker         # Records decision → outcome pipeline
│   ├── Records all decisions with context
│   ├── Tracks execution and outcomes
│   └── Stores in AgentDB for distributed access
│
├── VerdictJudge              # Evaluates decision quality
│   ├── Multi-factor evaluation (P&L, risk, timing, market alignment)
│   ├── Assigns verdict scores (0-1)
│   └── Identifies strengths and weaknesses
│
├── MemoryDistiller           # Compresses learning patterns
│   ├── Extracts successful patterns
│   ├── Deduplicates and merges similar patterns
│   ├── Prunes low-value patterns
│   └── Aggregates learnings for adaptation
│
└── PatternRecognizer         # Vector-based similarity search
    ├── Creates embeddings from patterns
    ├── HNSW indexing for ultra-fast retrieval
    ├── Quantization for 4-32x memory reduction
    └── Semantic similarity search
```

## Quick Start

### 1. Basic Usage

```javascript
const { ReasoningBankSwarmLearner, LearningMode } = require('./src/reasoningbank');

// Create learner
const learner = new ReasoningBankSwarmLearner('swarm-001', {
  learningMode: LearningMode.ONLINE,  // Learn from every trade immediately
  quicUrl: 'quic://localhost:8443',   // AgentDB connection
  enableHNSW: true,                    // 150x faster vector search
  enableQuantization: true             // 4-32x memory reduction
});

// Initialize
await learner.initialize();

// Start trading episode
learner.startEpisode({ strategy: 'momentum' });

// Record decision
const trajectory = await learner.recordTrajectory({
  id: 'dec-001',
  agentId: 'agent-momentum-1',
  type: 'momentum',
  action: 'buy',
  symbol: 'AAPL',
  quantity: 100,
  price: 150.25,
  reasoning: {
    confidence: 0.85,
    factors: ['strong_momentum', 'high_volume', 'breakout'],
    riskLevel: 'medium'
  },
  marketState: {
    volatility: 22,
    trend: 'up',
    volume: 2000000,
    expectedDirection: 'up'
  }
}, {
  // Outcome (can be null initially, added later)
  executed: true,
  fillPrice: 150.30,
  fillQuantity: 100,
  slippage: 0.033,
  executionTime: 125,
  pnl: 250,
  pnlPercent: 1.66,
  riskAdjustedReturn: 1.42
});

// Verdict is automatically judged in ONLINE mode
console.log(trajectory.verdict);
// { score: 0.82, quality: 'good', analysis: {...} }

// Query similar past decisions
const similar = await learner.querySimilarDecisions({
  marketState: {
    volatility: 25,
    trend: 'up',
    volume: 1800000
  }
}, { topK: 5, minSimilarity: 0.7 });

console.log(similar);
// [
//   { similarity: 0.91, trajectory: {...}, recommendation: { action: 'follow', confidence: 0.82 } },
//   { similarity: 0.85, trajectory: {...}, recommendation: { action: 'follow', confidence: 0.78 } },
//   ...
// ]

// Adapt agent based on learnings
await learner.adaptAgentStrategy('agent-momentum-1', similar);

// End episode and learn
await learner.endEpisode({ totalPnL: 1200, tradesCount: 15 });

// Shutdown
await learner.shutdown();
```

### 2. SwarmCoordinator Integration

```javascript
const { SwarmCoordinator, TOPOLOGY } = require('./src/e2b/swarm-coordinator');
const { addLearningCapabilities } = require('./src/reasoningbank/swarm-coordinator-integration');

// Create coordinator
const coordinator = new SwarmCoordinator({
  swarmId: 'production-swarm',
  topology: TOPOLOGY.MESH,
  maxAgents: 10
});

// Add learning capabilities
addLearningCapabilities(coordinator, {
  mode: LearningMode.CONTINUOUS,  // Background learning every 30s
  enableHNSW: true,
  enableQuantization: true
});

// Initialize swarm and learning
await coordinator.initializeSwarm();
await coordinator.initializeLearning();

// Record trading decision
const trajectory = await coordinator.recordDecision('agent-001', {
  action: 'buy',
  symbol: 'TSLA',
  quantity: 50,
  price: 200,
  reasoning: { confidence: 0.9 },
  marketState: { volatility: 30, trend: 'up' }
});

// Update with outcome
await coordinator.recordOutcome(trajectory.id, {
  executed: true,
  pnlPercent: 3.5,
  riskAdjustedReturn: 2.8
});

// Get recommendations
const recommendations = await coordinator.getRecommendations({
  volatility: 32,
  trend: 'up',
  volume: 5000000
});

// Adapt agent
await coordinator.adaptAgent('agent-001');

// Get learning stats
const stats = coordinator.getLearningStats();
console.log(stats);
```

### 3. E2BMonitor Integration

```javascript
const { E2BMonitor } = require('./src/e2b/monitor-and-scale');
const { addLearningMetrics } = require('./src/reasoningbank/e2b-monitor-integration');

// Create monitor
const monitor = new E2BMonitor({
  monitorInterval: 5000
});

// Add learning metrics
addLearningMetrics(monitor, coordinator);

// Start monitoring
await monitor.startMonitoring();

// Check learning health
const learningHealth = monitor.checkLearningHealth();
console.log(learningHealth);
// {
//   status: 'healthy',
//   issues: [],
//   stats: { avgVerdictScore: 0.75, patternsLearned: 42, ... }
// }

// Generate health report (includes learning section)
const report = await monitor.generateHealthReport();
console.log(report.learning);
// {
//   enabled: true,
//   mode: 'continuous',
//   stats: { totalDecisions: 156, avgVerdictScore: 0.73, ... },
//   components: { trajectoryTracker: {...}, verdictJudge: {...}, ... }
// }
```

## Learning Modes

### ONLINE Mode
```javascript
learningMode: LearningMode.ONLINE
```
- Learns from every trade immediately
- Judges verdict as soon as outcome is available
- Best for real-time adaptation
- Higher computational overhead

### BATCH Mode
```javascript
learningMode: LearningMode.BATCH
```
- Processes pending trajectories every minute
- Batches verdict judgments for efficiency
- Good balance between learning speed and overhead
- Recommended for most production scenarios

### EPISODE Mode
```javascript
learningMode: LearningMode.EPISODE
```
- Learns at the end of each trading episode
- Analyzes complete episode for patterns
- Best for episodic trading strategies
- Lower overhead, delayed adaptation

### CONTINUOUS Mode
```javascript
learningMode: LearningMode.CONTINUOUS
```
- Background distillation every 30 seconds
- Continuous pattern extraction
- Best for long-running swarms
- Balances freshness with efficiency

## Verdict Evaluation Factors

The VerdictJudge evaluates decisions based on 5 weighted factors:

| Factor | Weight | Description |
|--------|--------|-------------|
| **P&L Outcome** | 35% | Profitability (0% = 0.4, +5% = 0.85, +10% = 1.0) |
| **Risk-Adjusted Return** | 25% | Risk-adjusted performance (Sharpe-like metric) |
| **Timing Quality** | 20% | Slippage and execution timing (low slippage = high score) |
| **Market Alignment** | 15% | Did market move as expected? |
| **Reasoning Quality** | 5% | Was decision logic structured and sound? |

### Quality Classifications

- **Excellent** (0.9+): Outstanding decision
- **Good** (0.7-0.9): Positive outcome
- **Neutral** (0.4-0.7): No clear outcome
- **Poor** (0.2-0.4): Negative outcome
- **Terrible** (<0.2): Very bad outcome

## Pattern Types

### Market Condition Patterns
```javascript
{
  type: 'market_condition',
  conditions: {
    volatility: 22.5,
    trend: 'up',
    volume: 1500000
  },
  score: 0.82,
  confidence: 0.75
}
```

### Timing Patterns
```javascript
{
  type: 'timing',
  hour: 10,  // 10 AM
  avgSlippage: 0.02,
  score: 0.78,
  confidence: 0.68
}
```

### Risk Management Patterns
```javascript
{
  type: 'risk_management',
  avgQuantity: 85,
  avgRiskAdjusted: 1.45,
  score: 0.80,
  confidence: 0.72
}
```

### Strategy Patterns
```javascript
{
  type: 'strategy',
  strategyType: 'momentum',
  avgScore: 0.74,
  successRate: 0.65,
  confidence: 0.70
}
```

### Composite Patterns
```javascript
{
  type: 'composite',
  factors: [
    { type: 'market_condition', importance: 'high' },
    { type: 'timing', hour: 14, importance: 'medium' }
  ],
  score: 0.85,
  confidence: 0.80
}
```

## Vector Embeddings

Pattern embeddings are 128-dimensional vectors:

- **Market Features** (32 dims): Volatility, volume, trend, price, MAs
- **Decision Features** (64 dims): Action, type, quantity, price, reasoning
- **Outcome Features** (16 dims): P&L, risk-adjusted, slippage, verdict
- **Total**: 128 dimensions, normalized to unit vectors

### Similarity Search

```javascript
// Find 5 most similar patterns with 70%+ similarity
const similar = await learner.querySimilarDecisions(currentState, {
  topK: 5,
  minSimilarity: 0.7,
  includeContext: true
});

// Results sorted by similarity (cosine distance)
similar.forEach(result => {
  console.log(`Similarity: ${result.similarity.toFixed(3)}`);
  console.log(`Recommendation: ${result.recommendation.action}`);
  console.log(`Confidence: ${result.recommendation.confidence.toFixed(3)}`);
});
```

## Performance Optimization

### HNSW Indexing (150x Faster)
```javascript
enableHNSW: true  // Hierarchical Navigable Small World indexing
```
- **Without HNSW**: O(n) linear search
- **With HNSW**: O(log n) approximate nearest neighbor
- **Speedup**: 150x for 10,000+ patterns

### Quantization (4-32x Memory Reduction)
```javascript
enableQuantization: true
```
- Compresses 32-bit floats to 8-bit integers
- 4x memory reduction with minimal accuracy loss
- Can achieve 32x with aggressive quantization

## Statistics and Monitoring

```javascript
const stats = learner.getStats();

console.log(stats);
// {
//   totalDecisions: 1250,
//   totalOutcomes: 1198,
//   avgVerdictScore: 0.68,
//   patternsLearned: 87,
//   adaptationEvents: 42,
//   learningRate: 0.145,  // patterns/second
//   decisionsPerSecond: 2.08,
//   adaptationRate: 0.07,
//   uptime: '600.00s',
//   currentEpisode: 'episode-1763163540545-abc123',
//   totalEpisodes: 12
// }
```

## Best Practices

### 1. Choose the Right Learning Mode
- **High-frequency trading**: `ONLINE` or `BATCH`
- **Medium-frequency**: `CONTINUOUS`
- **Episodic strategies**: `EPISODE`

### 2. Set Appropriate Thresholds
```javascript
const learner = new ReasoningBankSwarmLearner(swarmId, {
  minPatternOccurrence: 3,      // Require 3+ occurrences
  similarityThreshold: 0.8,     // 80%+ similarity to merge
  pruneThreshold: 0.3,          // Prune patterns below 0.3 score
  maxPatternsPerType: 50        // Keep top 50 per type
});
```

### 3. Monitor Learning Health
```javascript
// Regular health checks
setInterval(() => {
  const health = monitor.checkLearningHealth();

  if (health.status === 'critical') {
    console.error('Learning system critical!', health.issues);
    // Take corrective action
  }
}, 60000);
```

### 4. Adapt Agents Periodically
```javascript
// Adapt all agents every 5 minutes
setInterval(async () => {
  for (const [agentId, agent] of coordinator.agents.entries()) {
    await coordinator.adaptAgent(agentId);
  }
}, 300000);
```

### 5. Clean Old Trajectories
```javascript
// Weekly cleanup of learned trajectories
setInterval(() => {
  const cleared = learner.trajectoryTracker.clearOldTrajectories(
    7 * 24 * 60 * 60 * 1000  // 7 days
  );
  console.log(`Cleared ${cleared} old trajectories`);
}, 86400000);  // Daily
```

## Troubleshooting

### Low Verdict Scores
```javascript
if (stats.avgVerdictScore < 0.5) {
  // Review trading strategies
  const poorDecisions = await learner.trajectoryTracker.getTrajectorysByStatus('judged');
  const poorest = poorDecisions
    .filter(t => t.verdict.score < 0.4)
    .sort((a, b) => a.verdict.score - b.verdict.score)
    .slice(0, 10);

  // Analyze common weaknesses
  poorest.forEach(t => {
    console.log('Weaknesses:', t.verdict.analysis.weaknesses);
  });
}
```

### No Patterns Learned
```javascript
if (stats.patternsLearned === 0 && stats.totalOutcomes > 20) {
  // Check verdict quality distribution
  const verdictStats = learner.verdictJudge.getStats();
  console.log('Quality distribution:', verdictStats.qualityDistribution);

  // Lower thresholds if needed
  learner.memoryDistiller.options.minPatternOccurrence = 2;
  learner.memoryDistiller.options.similarityThreshold = 0.7;
}
```

### Slow Pattern Recognition
```javascript
const recognizerStats = learner.patternRecognizer.getStats();

if (recognizerStats.avgQueryTime > 100) {
  console.log('Slow queries detected');
  console.log('Cache hit rate:', recognizerStats.cacheHitRate);

  // Enable optimizations
  learner.patternRecognizer.options.enableHNSW = true;
  learner.patternRecognizer.options.enableQuantization = true;
}
```

## File Structure

```
src/reasoningbank/
├── swarm-learning.js                    # Main learner class
├── trajectory-tracker.js                # Decision recording
├── verdict-judge.js                     # Quality evaluation
├── memory-distiller.js                  # Pattern learning
├── pattern-recognizer.js                # Vector similarity
├── index.js                             # Exports
├── swarm-coordinator-integration.js     # SwarmCoordinator extension
└── e2b-monitor-integration.js           # E2BMonitor extension

tests/reasoningbank/
└── reasoningbank.test.js                # Comprehensive tests

docs/
└── REASONINGBANK_INTEGRATION.md         # This file
```

## API Reference

### ReasoningBankSwarmLearner

#### Constructor
```javascript
new ReasoningBankSwarmLearner(swarmId, config)
```

#### Methods
- `async initialize()` - Initialize learning system
- `async recordTrajectory(decision, outcome, metadata)` - Record decision
- `async judgeVerdict(trajectoryId)` - Evaluate decision quality
- `async learnFromExperience(episodeId)` - Learn from episode
- `async querySimilarDecisions(state, options)` - Find similar patterns
- `async adaptAgentStrategy(agentId, learnings)` - Adapt agent
- `startEpisode(config)` - Start new episode
- `async endEpisode(result)` - End episode and learn
- `getStats()` - Get learning statistics
- `async shutdown()` - Cleanup resources

### SwarmCoordinator Extensions

- `async initializeLearning()` - Initialize learning
- `async recordDecision(agentId, decision, metadata)` - Record agent decision
- `async recordOutcome(trajectoryId, outcome)` - Update with outcome
- `async getRecommendations(marketState, options)` - Get recommendations
- `async adaptAgent(agentId)` - Adapt agent strategy
- `async endTradingEpisode(result)` - End episode
- `getLearningStats()` - Get statistics

### E2BMonitor Extensions

- `checkLearningHealth()` - Check learning system health
- `async generateHealthReport()` - Generate report with learning metrics

## License

MIT

## Support

For issues or questions, please file an issue on GitHub or contact the development team.
