# E2B Trading Swarm Benchmark Guide

## Overview

This guide explains how to run and interpret the comprehensive E2B trading swarm performance benchmarks.

## Prerequisites

```bash
# Install dependencies
npm install

# Set E2B API key
export E2B_API_KEY="your-api-key-here"

# Verify E2B connection
npx e2b-code-interpreter@latest --version
```

## Running Benchmarks

### Full Benchmark Suite

Run all benchmarks (takes ~45 minutes):

```bash
npm test -- tests/e2b/swarm-benchmarks.test.js
```

### Individual Benchmark Categories

Run specific benchmark categories:

```bash
# Creation performance only
npm test -- tests/e2b/swarm-benchmarks.test.js -t "Creation Performance"

# Scalability only
npm test -- tests/e2b/swarm-benchmarks.test.js -t "Scalability"

# Trading operations only
npm test -- tests/e2b/swarm-benchmarks.test.js -t "Trading Operations"

# Communication only
npm test -- tests/e2b/swarm-benchmarks.test.js -t "Communication"

# Resource usage only
npm test -- tests/e2b/swarm-benchmarks.test.js -t "Resource Usage"

# Cost analysis only
npm test -- tests/e2b/swarm-benchmarks.test.js -t "Cost Analysis"
```

### Custom Test Runs

Run with custom Jest options:

```bash
# Verbose output
npm test -- tests/e2b/swarm-benchmarks.test.js --verbose

# Run specific test
npm test -- tests/e2b/swarm-benchmarks.test.js -t "Swarm initialization time"

# Generate coverage
npm test -- tests/e2b/swarm-benchmarks.test.js --coverage

# Run in watch mode for development
npm test -- tests/e2b/swarm-benchmarks.test.js --watch
```

## Benchmark Categories

### 1. Creation Performance

**Tests:**
- Swarm initialization time
- Agent deployment latency
- Parallel vs sequential deployment

**What it measures:**
- How quickly swarms can be created
- Agent deployment efficiency
- Parallelization benefits

**Expected results:**
- Swarm init: < 5s
- Agent deployment: < 3s per agent
- Parallel speedup: > 1.5x

### 2. Scalability

**Tests:**
- Agent count scaling (1, 5, 10 agents)
- Linear scaling (2-20 agents)
- Topology performance comparison

**What it measures:**
- How the system scales with agent count
- Resource efficiency at scale
- Optimal topology selection

**Expected results:**
- Sub-linear time scaling
- Constant memory per agent
- Scaling to 10 agents: < 30s

### 3. Trading Operations

**Tests:**
- Strategy execution throughput
- Task distribution efficiency
- Consensus decision latency

**What it measures:**
- Trading strategy performance
- Parallel task execution
- Decision-making speed

**Expected results:**
- Strategy execution: < 100ms
- Throughput: > 10 strategies/sec
- Consensus: < 200ms

### 4. Communication

**Tests:**
- Inter-agent communication latency
- State synchronization overhead
- Message passing throughput

**What it measures:**
- Network performance
- Agent coordination efficiency
- Communication overhead

**Expected results:**
- Inter-agent latency: < 50ms
- Message throughput: > 50 msg/sec
- State sync: < 1s

### 5. Resource Usage

**Tests:**
- Memory usage per agent
- CPU utilization per topology
- Network bandwidth usage

**What it measures:**
- Resource consumption patterns
- Topology efficiency
- Infrastructure requirements

**Expected results:**
- Memory per agent: < 200MB
- CPU per agent: < 100%
- Reasonable network usage

### 6. Cost Analysis

**Tests:**
- Cost per trading operation
- Cost comparison across topologies
- Scalability cost efficiency

**What it measures:**
- Operational costs
- Cost-effectiveness of topologies
- ROI at different scales

**Expected results:**
- Cost per operation: < $0.01
- Hourly rate: < $2
- Sub-linear cost scaling

## Performance Targets

| Metric | Target | Importance |
|--------|--------|------------|
| Swarm Initialization | < 5s | High |
| Agent Deployment | < 3s | High |
| Strategy Execution | < 100ms | Critical |
| Inter-Agent Latency | < 50ms | High |
| Scaling to 10 Agents | < 30s | Medium |
| Cost per Hour | < $2 | High |

## Interpreting Results

### Report Location

After running benchmarks, find the detailed report at:
```
/docs/e2b/SWARM_BENCHMARKS_REPORT.md
```

### Key Metrics to Review

1. **Mean Latency**: Average performance across all runs
2. **P95/P99 Latency**: Tail latency (worst-case scenarios)
3. **Throughput**: Operations per second
4. **Cost per Operation**: Financial efficiency
5. **Scalability Factor**: How well it scales with agents

### Performance Status

- ✅ **Pass**: Meets or exceeds target
- ❌ **Fail**: Below target, needs optimization
- ⏳ **Pending**: Test not yet run

### Cost Analysis

Review the cost section to understand:
- Current operational costs
- Cost-optimal topology for your use case
- Projected monthly expenses
- Operations per dollar (ROI)

## Optimization Recommendations

Based on benchmark results, the report includes:

### High Priority
- Critical performance issues
- Cost optimization opportunities
- Scalability bottlenecks

### Medium Priority
- Efficiency improvements
- Configuration optimizations
- Resource tuning

### Low Priority
- Nice-to-have enhancements
- Future considerations
- Advanced optimizations

## Topology Selection Guide

### Mesh Topology

**Best for:**
- 2-8 agents
- Low-latency requirements
- Simple coordination

**Characteristics:**
- Fastest consensus
- Higher network overhead
- Best for small swarms

### Hierarchical Topology

**Best for:**
- 10+ agents
- Cost efficiency
- Structured coordination

**Characteristics:**
- Better scalability
- Lower network overhead
- Predictable performance

### Ring Topology

**Best for:**
- Balanced requirements
- Medium-sized swarms
- Simple message passing

**Characteristics:**
- Moderate performance
- Low complexity
- Good cost-efficiency

## Troubleshooting

### E2B Connection Issues

```bash
# Check E2B API key
echo $E2B_API_KEY

# Test E2B connection
node -e "require('@e2b/code-interpreter').Sandbox.create().then(s => { console.log('Connected'); s.close(); })"
```

### Timeout Issues

If tests timeout, increase Jest timeout:

```javascript
// In test file
jest.setTimeout(600000); // 10 minutes
```

Or run with increased timeout:
```bash
npm test -- tests/e2b/swarm-benchmarks.test.js --testTimeout=600000
```

### Memory Issues

For memory-constrained environments:

```bash
# Increase Node.js memory
NODE_OPTIONS=--max-old-space-size=4096 npm test -- tests/e2b/swarm-benchmarks.test.js
```

### Cost Concerns

To limit costs during testing:

1. Reduce iteration counts in tests
2. Run specific test categories
3. Use smaller agent counts
4. Monitor E2B dashboard for usage

## CI/CD Integration

### GitHub Actions

```yaml
name: E2B Benchmarks

on:
  schedule:
    - cron: '0 0 * * 0'  # Weekly on Sunday
  workflow_dispatch:  # Manual trigger

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
        with:
          node-version: '18'
      - run: npm ci
      - name: Run Benchmarks
        env:
          E2B_API_KEY: ${{ secrets.E2B_API_KEY }}
        run: npm test -- tests/e2b/swarm-benchmarks.test.js
      - name: Upload Report
        uses: actions/upload-artifact@v3
        with:
          name: benchmark-report
          path: docs/e2b/SWARM_BENCHMARKS_REPORT.md
```

### GitLab CI

```yaml
benchmark:
  stage: test
  script:
    - npm ci
    - npm test -- tests/e2b/swarm-benchmarks.test.js
  artifacts:
    paths:
      - docs/e2b/SWARM_BENCHMARKS_REPORT.md
  only:
    - schedules
```

## Best Practices

### 1. Regular Benchmarking

Run benchmarks:
- Weekly for trend analysis
- Before major releases
- After infrastructure changes
- When optimizing performance

### 2. Baseline Comparison

Keep historical reports to:
- Track performance trends
- Identify regressions
- Validate optimizations
- Plan capacity

### 3. Cost Monitoring

- Review cost analysis regularly
- Set budget alerts
- Optimize for cost-efficiency
- Monitor usage patterns

### 4. Performance Tuning

Use benchmark data to:
- Select optimal topology
- Configure agent count
- Tune resource allocation
- Optimize strategies

## Advanced Usage

### Custom Benchmark Scenarios

Extend benchmarks with custom scenarios:

```javascript
test('Custom: High-frequency trading scenario', async () => {
  const { swarmId } = await swarmManager.createSwarm('mesh', 10);

  // Your custom benchmark logic
  const results = await customBenchmark(swarmId);

  expect(results.latency).toBeLessThan(10); // < 10ms
});
```

### Performance Profiling

Enable detailed profiling:

```bash
# CPU profiling
node --prof tests/e2b/swarm-benchmarks.test.js

# Memory profiling
node --inspect tests/e2b/swarm-benchmarks.test.js
```

### Load Testing

For production load testing:

```javascript
// Increase iteration counts
const iterations = process.env.LOAD_TEST ? 1000 : 50;

// Increase agent counts
const maxAgents = process.env.LOAD_TEST ? 50 : 20;

// Extended duration
const testDuration = process.env.LOAD_TEST ? 3600000 : 60000;
```

## Support

For issues or questions:

1. Check [SWARM_BENCHMARKS_REPORT.md](./SWARM_BENCHMARKS_REPORT.md) for detailed results
2. Review [E2B documentation](https://e2b.dev/docs)
3. Check [GitHub issues](https://github.com/your-repo/neural-trader/issues)
4. Contact support team

## Next Steps

After running benchmarks:

1. ✅ Review the comprehensive report
2. ✅ Identify optimization opportunities
3. ✅ Select optimal topology for your use case
4. ✅ Configure production deployment
5. ✅ Set up monitoring and alerts
6. ✅ Schedule regular benchmark runs

---

**Last Updated**: 2025-11-14
**Version**: 1.0.0
**Author**: Neural Trader Team
