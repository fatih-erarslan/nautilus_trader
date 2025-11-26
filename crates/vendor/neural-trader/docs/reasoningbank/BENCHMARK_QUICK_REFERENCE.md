# ReasoningBank E2B Swarm Benchmarks - Quick Reference

## Overview

Comprehensive benchmark suite comparing traditional rule-based trading swarms vs ReasoningBank-enhanced self-learning swarms on E2B sandboxes.

## Running Benchmarks

### Full Suite
```bash
# Run all benchmarks
npm test -- tests/reasoningbank/learning-benchmarks.test.js

# Run with detailed output
npm test -- tests/reasoningbank/learning-benchmarks.test.js --verbose

# Run specific category
npm test -- tests/reasoningbank/learning-benchmarks.test.js -t "Learning Effectiveness"
```

### Using Benchmark Runner
```bash
# Automated execution with report generation
node tests/reasoningbank/benchmark-runner.js
```

## Test Categories

### 1. Learning Effectiveness (5 tests)
- **Decision quality improvement** - 100 episodes
- **Convergence rate** - Episodes to 80% accuracy
- **Pattern recognition** - Accuracy of learned patterns
- **Strategy adaptation** - Speed of adaptation to conditions

**Duration:** ~10 minutes each

### 2. Topology Comparison (4 tests)
- **Mesh topology** - Distributed learning
- **Hierarchical topology** - Centralized learning
- **Ring topology** - Sequential learning
- **Efficiency comparison** - Best topology analysis

**Duration:** ~10 minutes each

### 3. Traditional vs Self-Learning (4 tests)
- **P&L comparison** - Profitability
- **Decision latency** - Performance overhead
- **Resource overhead** - Memory/CPU usage
- **Sharpe ratio** - Risk-adjusted returns

**Duration:** ~15 minutes each

### 4. Memory & Performance (3 tests)
- **AgentDB query performance** - Vector search speed
- **Memory usage** - Trajectory storage overhead
- **Learning throughput** - Decisions per second

**Duration:** ~5 minutes each

### 5. Adaptive Learning (3 tests)
- **Market adaptation** - Response to conditions
- **Strategy switching** - Dynamic strategy changes
- **Knowledge sharing** - Multi-agent coordination

**Duration:** ~10 minutes each

## Key Metrics Tracked

### Trading Performance
- **P&L** - Profit and Loss
- **Sharpe Ratio** - Risk-adjusted returns
- **Win Rate** - Percentage of profitable trades
- **Max Drawdown** - Largest peak-to-trough decline

### Learning Metrics
- **Accuracy** - Prediction correctness
- **Convergence Rate** - Episodes to 80% accuracy
- **Pattern Count** - Number of learned patterns
- **Improvement Rate** - Accuracy increase over time

### Performance Metrics
- **Decision Latency** - Time per decision (ms)
- **Throughput** - Decisions per second
- **Memory Usage** - RAM consumption (MB)
- **CPU Usage** - Processor utilization (%)

### Efficiency Metrics
- **Learning Efficiency** - Accuracy improvement per time
- **Resource Efficiency** - Performance per resource unit
- **Knowledge Sharing** - Pattern distribution effectiveness

## Results Structure

```
tests/reasoningbank/results/
├── learning-effectiveness.json       # Learning curve data
├── topology-mesh-learning.json       # Mesh topology results
├── topology-hierarchical-learning.json
├── topology-ring-learning.json
├── topology-comparison.json          # Topology rankings
├── traditional-vs-reasoningbank.json # Head-to-head comparison
├── latency-comparison.json           # Performance overhead
├── resource-overhead.json            # Memory/CPU overhead
├── sharpe-comparison.json            # Risk-adjusted performance
├── agentdb-query-performance.json    # Vector search metrics
├── memory-usage-trajectory.json      # Memory growth
├── learning-throughput.json          # Decision throughput
├── market-adaptation.json            # Adaptation events
├── strategy-switching.json           # Strategy changes
├── knowledge-sharing.json            # Multi-agent coordination
└── learning-curve-data.json          # Visualization data
```

## Report Location

**Primary Report:**
```
docs/reasoningbank/LEARNING_BENCHMARKS_REPORT.md
```

**Contents:**
- Executive Summary
- Key Findings
- Detailed Results
- Learning Curves
- Topology Comparison
- Traditional vs ReasoningBank
- Resource Analysis
- Conclusions & Recommendations

## Configuration

Edit `BENCHMARK_CONFIG` in `learning-benchmarks.test.js`:

```javascript
const BENCHMARK_CONFIG = {
  episodeCount: 100,           // Number of trading episodes
  warmupEpisodes: 10,          // Warmup before measurement
  tradingSymbols: ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA'],
  initialCapital: 100000,      // Starting capital ($)
  maxPositionSize: 0.2,        // 20% max position size
  topologies: ['mesh', 'hierarchical', 'ring', 'star'],
  learningRates: [0.001, 0.01, 0.1],
  trajectoryBatchSize: 32      // Batch size for learning
};
```

## Market Scenarios

Test against different market conditions:

```javascript
const { MARKET_SCENARIOS } = require('./scenarios/market-conditions');

// Available scenarios:
- bullMarket          // Strong uptrend
- bearMarket          // Strong downtrend
- sidewaysMarket      // Range-bound
- highVolatility      // Choppy market
- marketCrash         // Rapid decline
- recoveryPhase       // Post-crash recovery
- newsEvent           // Sudden spike
- sectorRotation      // Changing leadership
```

## Expected Results

### Learning Effectiveness
- **Accuracy Improvement:** 15-30%
- **Convergence Rate:** 50-80 episodes
- **Final Accuracy:** 65-80%

### Performance Comparison
- **P&L Improvement:** 10-25% (vs traditional)
- **Sharpe Improvement:** 0.2-0.5
- **Win Rate:** 55-65% (after convergence)

### Resource Overhead
- **Memory Increase:** 50-150%
- **CPU Increase:** 20-80%
- **Time Overhead:** 15-40%

### Query Performance
- **Avg Query Time:** <50ms
- **P95 Query Time:** <100ms
- **Throughput:** >1 decision/sec

## Quick Commands

```bash
# Run single test category
npm test -- -t "Learning Effectiveness"

# Run with coverage
npm test -- --coverage tests/reasoningbank

# Generate report only (if results exist)
node tests/reasoningbank/benchmark-runner.js --report-only

# Clean results
rm -rf tests/reasoningbank/results/*.json

# View latest report
cat docs/reasoningbank/LEARNING_BENCHMARKS_REPORT.md
```

## Interpreting Results

### Good Performance Indicators
✅ Accuracy improvement >15%
✅ Convergence <80 episodes
✅ P&L improvement >10%
✅ Resource overhead <150%
✅ Query latency <100ms

### Warning Signs
⚠️ Accuracy improvement <5%
⚠️ Convergence >100 episodes
⚠️ P&L degradation >10%
⚠️ Resource overhead >200%
⚠️ Query latency >200ms

### When to Use ReasoningBank

**✅ Recommended:**
- Long-term strategies (>100 episodes)
- Complex market conditions
- Multi-agent scenarios
- Value learning from experience

**❌ Not Recommended:**
- Short-term trading (<20 episodes)
- Resource-constrained environments
- Simple, stable markets
- When rules are well-defined

## Troubleshooting

### Tests Timing Out
- Reduce `episodeCount` in config
- Increase Jest timeout: `jest.setTimeout(900000)`
- Run fewer topologies

### High Memory Usage
- Reduce `trajectoryBatchSize`
- Lower `episodeCount`
- Run tests sequentially

### E2B Sandbox Errors
- Check E2B API key: `E2B_API_KEY`
- Verify sandbox quota
- Check network connectivity

### Inconsistent Results
- Increase `warmupEpisodes`
- Use fixed random seed
- Run multiple iterations

## Advanced Usage

### Custom Market Scenarios
```javascript
const customScenario = {
  name: 'Custom Market',
  pricePattern: (basePrice, episode) => {
    // Your custom price generation
  },
  indicators: { /* ... */ }
};

await executor.runEpisode(i, symbols, customScenario);
```

### Custom Learning Rates
```javascript
const executor = new ReasoningBankSwarmExecutor('mesh', {
  ...BENCHMARK_CONFIG,
  learningRate: 0.005,
  explorationDecay: 0.99
});
```

### Export Results for Analysis
```javascript
// Results are automatically saved as JSON
const results = require('./results/learning-effectiveness.json');

// Analyze with your tools
analyze(results.episodes);
```

## CI/CD Integration

```yaml
# .github/workflows/benchmarks.yml
- name: Run ReasoningBank Benchmarks
  run: npm test -- tests/reasoningbank/learning-benchmarks.test.js
  timeout-minutes: 120

- name: Upload Results
  uses: actions/upload-artifact@v3
  with:
    name: benchmark-results
    path: tests/reasoningbank/results/
```

## Support

- **Issues:** GitHub Issues
- **Documentation:** `docs/reasoningbank/`
- **Examples:** `tests/reasoningbank/scenarios/`

---

**Last Updated:** 2024-01-14
**Version:** 1.0.0
