# Agentic Jujutsu + E2B Integration Guide

## Overview

This guide explains how to use **agentic-jujutsu** for multi-agent coordination with **E2B sandboxes** for distributed trading strategy execution.

### What is Agentic Jujutsu?

**Agentic Jujutsu** is a quantum-resistant, self-learning version control system designed for AI agents. It provides:

- 🤖 **Multi-Agent Coordination**: Zero-conflict collaboration (23x faster than Git)
- 🧠 **Self-Learning AI**: Learns from successful operations via ReasoningBank
- 🔒 **Quantum-Resistant Security**: SHA3-512 fingerprints + HQC-128 encryption
- 📊 **Pattern Recognition**: Automatically discovers optimal operation sequences
- ⚡ **Lock-Free Operations**: No blocking, perfect for distributed systems

### What are E2B Sandboxes?

**E2B (Environment2 Boxes)** provides isolated, ephemeral sandboxes for code execution:

- 🔒 **Isolated Environments**: Each strategy runs in its own sandbox
- ⚡ **Fast Startup**: Sandboxes create in <2 seconds
- 🐳 **Docker-Based**: Built on container technology
- 🔄 **Automatic Cleanup**: Resources released after execution
- 📊 **Monitoring**: Built-in performance tracking

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Swarm Coordinator                        │
│  ┌────────────────────────┐  ┌──────────────────────────┐  │
│  │  Agentic Jujutsu      │  │   E2B Sandbox Manager    │  │
│  │  - Learning           │  │   - Resource Allocation  │  │
│  │  - Pattern Recognition│  │   - Isolation            │  │
│  │  - Suggestions        │  │   - Cleanup              │  │
│  └────────────────────────┘  └──────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
            ┌─────────────────┼─────────────────┐
            │                 │                 │
    ┌───────▼──────┐  ┌───────▼──────┐  ┌───────▼──────┐
    │  E2B Sandbox │  │  E2B Sandbox │  │  E2B Sandbox │
    │              │  │              │  │              │
    │  Strategy A  │  │  Strategy B  │  │  Strategy C  │
    │  Agent 1     │  │  Agent 2     │  │  Agent 3     │
    └──────────────┘  └──────────────┘  └──────────────┘
```

---

## Quick Start

### Installation

Package is already installed with:
```bash
npm install @neural-trader/e2b-strategies
```

Dependencies included:
- `agentic-jujutsu@2.3.3` - AI coordination
- `@e2b/sdk@0.12.5` - Sandbox management
- `commander@11.1.0` - CLI interface

### Basic Usage

#### 1. Deploy Single Strategy

```javascript
const { SwarmCoordinator } = require('@neural-trader/e2b-strategies/swarm');

async function deploy() {
    // Initialize coordinator with learning enabled
    const coordinator = new SwarmCoordinator({
        maxAgents: 10,
        learningEnabled: true,
        autoOptimize: true
    });

    // Register strategy
    coordinator.registerStrategy('momentum', {
        type: 'momentum',
        symbols: ['SPY', 'QQQ', 'IWM'],
        threshold: 0.02,
        positionSize: 10
    });

    // Deploy to E2B sandbox
    const result = await coordinator.deployStrategy('momentum', {
        symbol: 'SPY',
        threshold: 0.02
    });

    console.log('Result:', result);

    // Cleanup
    await coordinator.cleanup();
}

deploy();
```

#### 2. Deploy Multiple Agents (Swarm)

```javascript
const { SwarmCoordinator } = require('@neural-trader/e2b-strategies/swarm');

async function deploySwarm() {
    const coordinator = new SwarmCoordinator({
        maxAgents: 20,
        learningEnabled: true
    });

    // Register multiple strategies
    coordinator.registerStrategy('momentum', { /* config */ });
    coordinator.registerStrategy('mean-reversion', { /* config */ });
    coordinator.registerStrategy('neural-forecast', { /* config */ });

    // Create swarm deployments
    const deployments = [
        { strategyName: 'momentum', params: { symbol: 'SPY' } },
        { strategyName: 'momentum', params: { symbol: 'QQQ' } },
        { strategyName: 'mean-reversion', params: { symbol: 'IWM' } },
        { strategyName: 'neural-forecast', params: { symbol: 'TSLA' } }
    ];

    // Deploy all concurrently
    const results = await coordinator.deploySwarm(deployments);

    console.log(`Success: ${results.filter(r => r.status === 'fulfilled').length}/${deployments.length}`);

    await coordinator.cleanup();
}

deploySwarm();
```

---

## Self-Learning Features

### Learning from Executions

Agentic Jujutsu automatically learns from each deployment:

```javascript
const coordinator = new SwarmCoordinator({
    learningEnabled: true
});

// Each deployment is tracked as a "trajectory"
for (let i = 0; i < 10; i++) {
    await coordinator.deployStrategy('momentum', { symbol: 'SPY' });
}

// View learning statistics
const stats = coordinator.getLearningStats();
console.log('Total Trajectories:', stats.totalTrajectories);
console.log('Average Success:', (stats.avgSuccessRate * 100).toFixed(1) + '%');
console.log('Improvement Rate:', (stats.improvementRate * 100).toFixed(1) + '%');
console.log('Prediction Accuracy:', (stats.predictionAccuracy * 100).toFixed(1) + '%');
```

### Getting AI Suggestions

```javascript
// Get AI recommendation for deployment
const suggestion = coordinator.getSuggestion('momentum', {
    symbol: 'SPY',
    threshold: 0.02
});

console.log('Confidence:', (suggestion.confidence * 100).toFixed(1) + '%');
console.log('Expected Success:', (suggestion.expectedSuccessRate * 100).toFixed(1) + '%');
console.log('Reasoning:', suggestion.reasoning);

if (suggestion.confidence > 0.7) {
    // High confidence - proceed with suggested approach
    console.log('Recommended operations:');
    suggestion.recommendedOperations.forEach(op => console.log('  -', op));
}
```

### Discovering Patterns

```javascript
// Get discovered patterns
const patterns = coordinator.getPatterns();

patterns.forEach(pattern => {
    console.log(`\nPattern: ${pattern.name}`);
    console.log(`  Success Rate: ${(pattern.successRate * 100).toFixed(1)}%`);
    console.log(`  Observations: ${pattern.observationCount}`);
    console.log(`  Confidence: ${(pattern.confidence * 100).toFixed(1)}%`);
    console.log(`  Operations: ${pattern.operationSequence.join(' → ')}`);
});
```

---

## CLI Usage

### Deploy Strategies

```bash
# Deploy single strategy with 5 agents
npm run swarm:deploy -- -s momentum -a 5 -p '{"symbols":["SPY","QQQ","IWM"]}'

# Deploy with learning enabled (default)
npm run swarm:deploy -- -s mean-reversion -a 3 --learning

# Deploy with encryption
npm run swarm:deploy -- -s neural-forecast -a 2 --encryption-key "your-key"
```

### Run Benchmarks

```bash
# Run all benchmark scenarios
npm run swarm:benchmark

# Run specific scenario
npm run swarm:benchmark -- -s light-load

# Custom output directory
npm run swarm:benchmark -- -o ./my-results

# Benchmark specific strategies
npm run swarm:benchmark -- --strategies momentum,mean-reversion
```

### View Status

```bash
# Get current metrics
npm run swarm:status

# View learning statistics and patterns
node scripts/deploy-swarm.js status --learning --patterns
```

### View Patterns

```bash
# View all discovered patterns
npm run swarm:patterns

# Get AI suggestion for specific task
node scripts/deploy-swarm.js patterns -t "Deploy momentum strategy with high volatility"
```

### Export State

```bash
# Export learning data and metrics
npm run swarm:export

# Custom output file
node scripts/deploy-swarm.js export -o ./swarm-backup.json
```

---

## Benchmarking

### Run Complete Benchmark Suite

```javascript
const { E2BBenchmark } = require('@neural-trader/e2b-strategies/benchmark');

async function runBenchmarks() {
    const benchmark = new E2BBenchmark({
        scenarios: [
            { name: 'light-load', agents: 1, iterations: 10 },
            { name: 'medium-load', agents: 5, iterations: 20 },
            { name: 'heavy-load', agents: 10, iterations: 30 }
        ],
        strategies: ['momentum', 'mean-reversion', 'neural-forecast'],
        outputDir: './benchmark-results',
        thresholds: {
            maxLatencyMs: 5000,
            minThroughput: 10,
            maxErrorRate: 0.05,
            minSuccessRate: 0.95
        }
    });

    const report = await benchmark.run();

    console.log('Benchmark Results:');
    console.log('  Total Executions:', report.summary.totalExecutions);
    console.log('  Success Rate:', (report.summary.successRate * 100).toFixed(2) + '%');
    console.log('  Duration:', (report.summary.totalDuration / 1000).toFixed(2) + 's');
}

runBenchmarks();
```

### Benchmark Metrics

The benchmark framework tracks:

| Metric | Description |
|--------|-------------|
| **Average Duration** | Mean execution time across all runs |
| **P95 Latency** | 95th percentile latency (performance SLA) |
| **P99 Latency** | 99th percentile latency (tail performance) |
| **Success Rate** | Percentage of successful executions |
| **Throughput** | Operations per second |
| **Error Rate** | Percentage of failed executions |

### Benchmark Reports

After running benchmarks, you'll get:

1. **JSON Report** - Complete data for analysis
2. **Text Report** - Human-readable summary
3. **CSV Export** - For spreadsheet analysis
4. **Coordinator Report** - Learning statistics and patterns

Example output structure:
```
benchmark-results/
├── benchmark-2025-11-15T23-00-00-000Z.json
├── benchmark-2025-11-15T23-00-00-000Z.txt
├── benchmark-2025-11-15T23-00-00-000Z.csv
└── coordinator-2025-11-15T23-00-00-000Z.txt
```

---

## Optimization Recommendations

The system automatically generates optimization recommendations based on:

### 1. Reliability Issues
```
❌ Success rate (85.2%) is below 90%
💡 Recommendation: Add retry logic or circuit breakers
```

### 2. Performance Issues
```
❌ P95 latency (4500ms) is high
💡 Recommendation: Optimize strategy code or increase resources
```

### 3. Scalability Issues
```
❌ Low throughput (3.5 ops/sec)
💡 Recommendation: Consider parallel execution or caching
```

### 4. Learning Insights
```
✅ 12 high-success patterns discovered
💡 Recommendation: Review patterns for best practices
```

---

## Advanced Configuration

### Coordinator Options

```javascript
const coordinator = new SwarmCoordinator({
    // Agent Configuration
    maxAgents: 20,                    // Maximum concurrent agents
    sandboxTimeout: 300000,           // 5 minutes per execution

    // Learning Configuration
    learningEnabled: true,            // Enable ReasoningBank learning
    autoOptimize: true,              // Automatic optimization

    // Security Configuration
    encryptionKey: process.env.ENCRYPTION_KEY,  // HQC-128 encryption

    // Resource Configuration
    memoryLimit: 512,                // MB per sandbox
    cpuLimit: 1.0                    // CPU cores per sandbox
});
```

### Encryption for Sensitive Data

```javascript
// Enable quantum-resistant encryption
const encryptionKey = crypto.randomBytes(32).toString('base64');

const coordinator = new SwarmCoordinator({
    encryptionKey: encryptionKey,
    learningEnabled: true
});

// All trajectory data is now encrypted with HQC-128
```

### Custom Benchmark Scenarios

```javascript
const benchmark = new E2BBenchmark({
    scenarios: [
        {
            name: 'production-load',
            agents: 15,
            iterations: 100
        },
        {
            name: 'burst-load',
            agents: 50,
            iterations: 10
        }
    ],

    thresholds: {
        maxLatencyMs: 3000,
        minThroughput: 20,
        maxErrorRate: 0.02,
        minSuccessRate: 0.98
    }
});
```

---

## Performance Characteristics

### Agentic Jujutsu vs Traditional Git

| Metric | Git | Agentic Jujutsu |
|--------|-----|-----------------|
| Concurrent commits | 15 ops/s | 350 ops/s (23x) |
| Context switching | 500-1000ms | 50-100ms (10x) |
| Conflict resolution | 30-40% auto | 87% auto (2.5x) |
| Lock waiting | 50 min/day | 0 min (∞) |

### E2B Sandbox Performance

| Metric | Value |
|--------|-------|
| Startup time | <2s |
| Execution overhead | ~50ms |
| Memory per sandbox | 256-512 MB |
| Max concurrent sandboxes | 100+ |
| Cleanup time | <500ms |

---

## Best Practices

### 1. Learning Trajectory Management

```javascript
// ✅ Good: Meaningful descriptions
coordinator.deployStrategy('momentum', {
    symbol: 'SPY',
    threshold: 0.02,
    description: 'Deploy momentum strategy for SPY with 2% threshold'
});

// ❌ Bad: Vague descriptions
coordinator.deployStrategy('momentum', {});
```

### 2. Pattern Recognition

```javascript
// ✅ Good: Let patterns emerge naturally
for (let i = 0; i < 10; i++) {
    await coordinator.deployStrategy('momentum', params);
}

// Then review patterns
const patterns = coordinator.getPatterns();
```

### 3. Resource Management

```javascript
// ✅ Good: Always cleanup
try {
    const result = await coordinator.deployStrategy('momentum', params);
} finally {
    await coordinator.cleanup();
}

// ❌ Bad: Forgetting cleanup
const result = await coordinator.deployStrategy('momentum', params);
// Sandboxes leak!
```

### 4. Error Handling

```javascript
// ✅ Good: Record failures with details
try {
    await coordinator.deploySwarm(deployments);
} catch (error) {
    console.error('Deployment failed:', error.message);
    // Learning system automatically records failure
}
```

---

## Troubleshooting

### Issue: Low Confidence Suggestions

```javascript
const suggestion = coordinator.getSuggestion('momentum', params);

if (suggestion.confidence < 0.5) {
    console.log('Not enough learning data');
    console.log('Current trajectories:', stats.totalTrajectories);
    // Recommend: Run 5-10 deployments to build learning data
}
```

### Issue: Sandbox Timeout

```javascript
// Increase timeout for long-running strategies
const coordinator = new SwarmCoordinator({
    sandboxTimeout: 600000  // 10 minutes
});
```

### Issue: Out of Resources

```javascript
// Reduce concurrent agents
const coordinator = new SwarmCoordinator({
    maxAgents: 5  // Lower limit
});

// Or deploy in batches
const batch1 = deployments.slice(0, 5);
const batch2 = deployments.slice(5, 10);

await coordinator.deploySwarm(batch1);
await coordinator.deploySwarm(batch2);
```

---

## Examples

### Example 1: Production Deployment

```javascript
const { SwarmCoordinator } = require('@neural-trader/e2b-strategies/swarm');

async function productionDeploy() {
    const coordinator = new SwarmCoordinator({
        maxAgents: 10,
        learningEnabled: true,
        encryptionKey: process.env.ENCRYPTION_KEY,
        autoOptimize: true
    });

    // Register all strategies
    ['momentum', 'mean-reversion', 'neural-forecast'].forEach(strategy => {
        coordinator.registerStrategy(strategy, {
            type: strategy,
            symbols: ['SPY', 'QQQ', 'IWM'],
            interval: '1min'
        });
    });

    // Get AI suggestions before deploying
    const momentumSuggestion = coordinator.getSuggestion('momentum', { symbol: 'SPY' });

    if (momentumSuggestion.confidence > 0.8) {
        console.log('High confidence deployment');
        const result = await coordinator.deployStrategy('momentum', { symbol: 'SPY' });
        console.log('Result:', result);
    } else {
        console.log('Low confidence - needs more learning data');
    }

    await coordinator.cleanup();
}
```

### Example 2: Continuous Learning Loop

```javascript
async function learningLoop() {
    const coordinator = new SwarmCoordinator({
        learningEnabled: true
    });

    coordinator.registerStrategy('momentum', { /* config */ });

    // Run 20 iterations to build learning data
    for (let i = 0; i < 20; i++) {
        console.log(`\n--- Iteration ${i + 1} ---`);

        const result = await coordinator.deployStrategy('momentum', {
            symbol: 'SPY',
            threshold: 0.02 + (i * 0.001)  // Vary parameters
        });

        const stats = coordinator.getLearningStats();
        console.log(`Improvement Rate: ${(stats.improvementRate * 100).toFixed(1)}%`);
        console.log(`Prediction Accuracy: ${(stats.predictionAccuracy * 100).toFixed(1)}%`);
    }

    // Review learned patterns
    const patterns = coordinator.getPatterns();
    console.log(`\nDiscovered ${patterns.length} patterns`);

    await coordinator.cleanup();
}
```

---

## API Reference

See [API.md](./API.md) for complete API documentation.

## Support

- **GitHub Issues**: https://github.com/ruvnet/neural-trader/issues
- **NPM Package**: https://www.npmjs.com/package/@neural-trader/e2b-strategies
- **Agentic Jujutsu Docs**: https://www.npmjs.com/package/agentic-jujutsu
- **E2B Documentation**: https://e2b.dev/docs

---

**Version**: 1.1.0
**License**: MIT
**Status**: Production Ready ✅
