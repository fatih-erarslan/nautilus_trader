# E2B Trading Swarm Benchmarks

Comprehensive performance benchmarking suite for distributed trading swarms on E2B cloud infrastructure.

## 📋 Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Benchmark Categories](#benchmark-categories)
- [Running Benchmarks](#running-benchmarks)
- [Understanding Results](#understanding-results)
- [Optimization Guide](#optimization-guide)
- [Documentation](#documentation)

## Overview

This benchmark suite provides comprehensive performance testing for trading swarms, measuring:

- **Creation Performance**: Swarm initialization and agent deployment
- **Scalability**: Performance from 1 to 20+ agents
- **Trading Operations**: Strategy execution and consensus
- **Communication**: Inter-agent latency and throughput
- **Resource Usage**: Memory, CPU, and network utilization
- **Cost Analysis**: Operational costs and ROI

### Why Benchmark?

1. **Validate Performance**: Ensure system meets performance targets
2. **Identify Bottlenecks**: Find and fix performance issues
3. **Optimize Costs**: Reduce operational expenses
4. **Guide Scaling**: Make informed scaling decisions
5. **Track Trends**: Monitor performance over time

## Quick Start

### 1. Setup (30 seconds)

```bash
# Set E2B API key
export E2B_API_KEY="your-api-key-here"

# Install dependencies
npm install
```

### 2. Run Benchmarks (5-45 minutes)

```bash
# Quick benchmarks (5 minutes, ~$0.02)
npm run bench:swarm:fast

# Standard benchmarks (20 minutes, ~$0.08)
npm run bench:swarm

# Full benchmarks (45 minutes, ~$0.20)
npm run bench:swarm:full
```

### 3. View Results

```bash
# View comprehensive report
cat docs/e2b/SWARM_BENCHMARKS_REPORT.md

# Or run and view in one command
npm run bench:swarm:report
```

## Benchmark Categories

### 1. Creation Performance

**Tests:**
- Swarm initialization time
- Agent deployment latency
- Parallel vs sequential deployment

**Duration:** ~5 minutes
**Cost:** ~$0.02

**Measures:**
- How quickly swarms can be created
- Agent deployment efficiency
- Benefits of parallelization

### 2. Scalability

**Tests:**
- Agent count scaling (1, 5, 10 agents)
- Linear scaling (2-20 agents)
- Topology performance comparison

**Duration:** ~10 minutes
**Cost:** ~$0.04

**Measures:**
- How system scales with agent count
- Resource efficiency at scale
- Optimal topology selection

### 3. Trading Operations

**Tests:**
- Strategy execution throughput
- Task distribution efficiency
- Consensus decision latency

**Duration:** ~8 minutes
**Cost:** ~$0.03

**Measures:**
- Trading strategy performance
- Parallel task execution
- Decision-making speed

### 4. Communication

**Tests:**
- Inter-agent communication latency
- State synchronization overhead
- Message passing throughput

**Duration:** ~6 minutes
**Cost:** ~$0.02

**Measures:**
- Network performance
- Agent coordination efficiency
- Communication overhead

### 5. Resource Usage

**Tests:**
- Memory usage per agent
- CPU utilization per topology
- Network bandwidth usage

**Duration:** ~8 minutes
**Cost:** ~$0.03

**Measures:**
- Resource consumption patterns
- Topology efficiency
- Infrastructure requirements

### 6. Cost Analysis

**Tests:**
- Cost per trading operation
- Cost comparison across topologies
- Scalability cost efficiency

**Duration:** ~8 minutes
**Cost:** ~$0.03

**Measures:**
- Operational costs
- Cost-effectiveness of topologies
- ROI at different scales

## Running Benchmarks

### Basic Commands

```bash
# Run all benchmarks
npm run bench:swarm:full

# Run specific category
npm run bench:swarm -- -t "Creation Performance"
npm run bench:swarm -- -t "Scalability"
npm run bench:swarm -- -t "Trading Operations"

# Run specific test
npm run bench:swarm -- -t "Swarm initialization time"

# Run with verbose output
npm run bench:swarm -- --verbose
```

### Advanced Options

```bash
# Increase timeout for slower systems
npm run bench:swarm -- --testTimeout=900000  # 15 minutes

# Run with coverage
npm run bench:swarm -- --coverage

# Run in watch mode (development)
npm run bench:swarm -- --watch

# Generate JSON output
npm run bench:swarm -- --json --outputFile=results.json
```

### Environment Variables

```bash
# Set E2B API key
export E2B_API_KEY="your-key"

# Increase Node.js memory (if needed)
export NODE_OPTIONS="--max-old-space-size=4096"

# Set test environment
export NODE_ENV="test"
```

## Understanding Results

### Performance Metrics

#### Mean Latency
Average performance across all runs. Good indicator of typical performance.

**Example:** Mean strategy execution: 68ms

#### P95/P99 Latency
95th/99th percentile latency. Critical for production readiness.

**Example:** P95: 89ms, P99: 96ms

#### Throughput
Operations per second. Higher is better.

**Example:** 14.7 strategies/second

#### Cost per Operation
Financial efficiency metric.

**Example:** $0.000234 per trading operation

### Report Sections

1. **Executive Summary**: Quick overview of all targets
2. **Creation Performance**: Initialization metrics
3. **Scalability Analysis**: Scaling characteristics
4. **Trading Operations**: Strategy execution performance
5. **Communication Performance**: Latency and throughput
6. **Resource Utilization**: Memory, CPU, network
7. **Cost Analysis**: Operational costs and ROI
8. **Optimization Recommendations**: Prioritized improvements
9. **Comparison Charts**: Visual analysis
10. **Conclusions**: Key findings and next steps

### Performance Targets

| Metric | Target | Good | Excellent |
|--------|--------|------|-----------|
| Swarm Init | < 5s | < 4s | < 3s |
| Agent Deploy | < 3s | < 2s | < 1s |
| Strategy Exec | < 100ms | < 75ms | < 50ms |
| Inter-Agent Lat | < 50ms | < 40ms | < 30ms |
| Cost/Hour | < $2 | < $1.50 | < $1 |

### Status Indicators

- ✅ **Pass**: Meets or exceeds target
- ⚠️ **Warning**: Close to target, needs monitoring
- ❌ **Fail**: Below target, requires optimization
- ⏳ **Pending**: Test not yet run

## Optimization Guide

### Topology Selection

#### Mesh Topology
**Use for:** 2-8 agents

**Pros:**
- Fastest consensus
- Lowest latency
- Simple coordination

**Cons:**
- Higher network overhead
- Doesn't scale beyond 10 agents

#### Hierarchical Topology
**Use for:** 10+ agents

**Pros:**
- Best scalability
- Lower network overhead
- Cost-efficient at scale

**Cons:**
- Higher latency than mesh
- More complex coordination

#### Ring Topology
**Use for:** 5-15 agents

**Pros:**
- Balanced performance
- Moderate overhead
- Simple to understand

**Cons:**
- Not optimal for any specific use case

### Cost Optimization

1. **Choose Right Topology**
   - Mesh for < 10 agents
   - Hierarchical for 10+ agents

2. **Connection Pooling**
   - Reuse sandbox connections
   - Reduce initialization overhead

3. **Result Caching**
   - Cache frequently used data
   - Reduce redundant computation

4. **Batch Processing**
   - Process tasks in batches
   - Reduce per-task overhead

### Performance Tuning

1. **Strategy Execution**
   - Optimize strategy logic
   - Use efficient data structures
   - Minimize network calls

2. **Communication**
   - Use message batching
   - Compress large payloads
   - Implement request pipelining

3. **Resource Usage**
   - Monitor memory leaks
   - Optimize CPU usage
   - Reduce network bandwidth

## Documentation

### Core Documentation

- **[Quick Start](./BENCHMARK_QUICK_START.md)** - Get started in 3 steps
- **[Full Guide](./BENCHMARK_GUIDE.md)** - Complete documentation
- **[Example Report](./SWARM_BENCHMARKS_REPORT_EXAMPLE.md)** - Sample output

### Test Files

- **[Benchmark Suite](../../tests/e2b/swarm-benchmarks.test.js)** - Full implementation
- **[Test Setup](../../tests/setup.js)** - Jest configuration

### Related Documentation

- **[E2B Integration](../E2B_INTEGRATION.md)** - E2B setup guide
- **[Trading Swarms](../TRADING_SWARMS.md)** - Swarm architecture
- **[Performance Guide](../PERFORMANCE.md)** - Optimization tips

## Cost Estimates

### By Duration

| Duration | Agent Count | Estimated Cost |
|----------|-------------|----------------|
| 5 min | 3 agents | ~$0.02 |
| 20 min | 8 agents | ~$0.08 |
| 45 min | 20 agents | ~$0.20 |

### By Category

| Category | Duration | Cost |
|----------|----------|------|
| Creation | ~5 min | ~$0.02 |
| Scalability | ~10 min | ~$0.04 |
| Trading | ~8 min | ~$0.03 |
| Communication | ~6 min | ~$0.02 |
| Resources | ~8 min | ~$0.03 |
| Cost Analysis | ~8 min | ~$0.03 |

**Full Suite:** ~$0.20 (45 minutes)

## Best Practices

### 1. Run Regularly

```bash
# Weekly benchmarks
0 0 * * 0 npm run bench:swarm:full

# Pre-release benchmarks
npm run bench:swarm:full && git add docs/e2b/SWARM_BENCHMARKS_REPORT.md
```

### 2. Compare Results

```bash
# Save results with timestamp
cp docs/e2b/SWARM_BENCHMARKS_REPORT.md \
   benchmarks/results-$(date +%Y%m%d).md

# Compare with previous
diff benchmarks/results-20251101.md \
     benchmarks/results-20251108.md
```

### 3. Monitor Trends

Track key metrics over time:
- Mean latency trends
- P95/P99 trends
- Cost per operation
- Resource usage patterns

### 4. Set Alerts

```yaml
# Example GitHub Actions alert
- name: Check Performance
  run: |
    npm run bench:swarm
    if grep -q "❌ Fail" docs/e2b/SWARM_BENCHMARKS_REPORT.md; then
      echo "Performance regression detected!"
      exit 1
    fi
```

## Troubleshooting

### E2B Connection Issues

```bash
# Verify API key
echo $E2B_API_KEY

# Test connection
node -e "require('@e2b/code-interpreter').Sandbox.create().then(s => { console.log('Connected'); s.close(); })"
```

### Timeout Issues

```bash
# Increase timeout
npm run bench:swarm -- --testTimeout=900000

# Or in jest.config.js
testTimeout: 900000  // 15 minutes
```

### Memory Issues

```bash
# Increase Node.js memory
NODE_OPTIONS=--max-old-space-size=4096 npm run bench:swarm

# Monitor memory usage
node --expose-gc --inspect tests/e2b/swarm-benchmarks.test.js
```

### Cost Concerns

To reduce benchmark costs:

1. Run fewer iterations
2. Use smaller agent counts
3. Run specific categories only
4. Test during off-peak hours

## CI/CD Integration

### GitHub Actions

```yaml
name: Performance Benchmarks

on:
  schedule:
    - cron: '0 0 * * 0'  # Weekly
  workflow_dispatch:

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
        run: npm run bench:swarm:full
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
    - npm run bench:swarm:full
  artifacts:
    paths:
      - docs/e2b/SWARM_BENCHMARKS_REPORT.md
  only:
    - schedules
```

## Support

### Getting Help

1. **Documentation**: Check guides in this directory
2. **Issues**: [GitHub Issues](https://github.com/your-repo/neural-trader/issues)
3. **E2B Support**: [E2B Documentation](https://e2b.dev/docs)
4. **Community**: [Discord Server](https://discord.gg/your-server)

### Reporting Issues

When reporting benchmark issues, include:

1. Full error message
2. Benchmark category/test name
3. E2B API key status (set/not set)
4. Node.js version
5. Operating system
6. Recent changes to code/config

## Contributing

To contribute new benchmarks:

1. Add test to `swarm-benchmarks.test.js`
2. Update documentation
3. Run full suite to verify
4. Submit PR with results

### Example New Benchmark

```javascript
test('Benchmark: Your new test', async () => {
  const { swarmId } = await swarmManager.createSwarm('mesh', 5);

  const measurements = [];
  for (let i = 0; i < 10; i++) {
    const { duration } = await measureTime(async () => {
      // Your benchmark logic
    });
    measurements.push(duration);
  }

  const stats = calculateStatistics(measurements);
  benchmarkResults.yourCategory.push({
    name: 'Your Test Name',
    stats,
    target: 1000, // Your target
    passed: stats.mean < 1000
  });

  await swarmManager.cleanup(swarmId);
  expect(stats.mean).toBeLessThan(1000);
}, 60000);
```

## License

MIT OR Apache-2.0

## Acknowledgments

- **E2B**: Cloud sandbox infrastructure
- **Jest**: Testing framework
- **Neural Trader Team**: Development and maintenance

---

**Ready to benchmark?** Start with: `npm run bench:swarm:fast`

**Questions?** Check the [Quick Start Guide](./BENCHMARK_QUICK_START.md)
