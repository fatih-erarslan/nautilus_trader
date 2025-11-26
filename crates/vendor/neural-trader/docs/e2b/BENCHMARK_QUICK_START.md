# E2B Swarm Benchmarks - Quick Start

## 🚀 Quick Start (3 Steps)

### 1. Setup

```bash
# Set E2B API key
export E2B_API_KEY="your-api-key-here"

# Install dependencies
npm install
```

### 2. Run Benchmarks

```bash
# Full benchmark suite (~45 minutes)
npm run bench:swarm:full

# Quick benchmarks (~5 minutes)
npm run bench:swarm:fast

# Standard benchmarks (~20 minutes)
npm run bench:swarm
```

### 3. View Results

```bash
# View comprehensive report
cat docs/e2b/SWARM_BENCHMARKS_REPORT.md

# Or run and view in one command
npm run bench:swarm:report
```

## 📊 What Gets Benchmarked

| Category | Tests | Duration |
|----------|-------|----------|
| **Creation** | Swarm init, agent deployment, parallel vs sequential | ~5 min |
| **Scalability** | 1-20 agents, topology comparison | ~10 min |
| **Trading** | Strategy execution, task distribution, consensus | ~8 min |
| **Communication** | Latency, state sync, message passing | ~6 min |
| **Resources** | Memory, CPU, network bandwidth | ~8 min |
| **Cost** | Per operation, topology comparison, scaling | ~8 min |

## 🎯 Performance Targets

```
✅ Swarm Init:          < 5s
✅ Agent Deployment:    < 3s
✅ Strategy Execution:  < 100ms
✅ Inter-Agent Latency: < 50ms
✅ Scaling to 10:       < 30s
✅ Cost per Hour:       < $2
```

## 📈 Example Output

```
 PASS  tests/e2b/swarm-benchmarks.test.js (45.23s)
  E2B Swarm Benchmarks
    Creation Performance Benchmarks
      ✓ Benchmark: Swarm initialization time (1234ms)
      ✓ Benchmark: Agent deployment latency (2345ms)
      ✓ Benchmark: Parallel vs sequential deployment (3456ms)
    Scalability Benchmarks
      ✓ Benchmark: 1 agent vs 5 agents vs 10 agents (4567ms)
      ✓ Benchmark: Scaling from 2 to 20 agents (15678ms)
      ✓ Benchmark: Mesh vs Hierarchical topology (5678ms)
    Trading Operations Benchmarks
      ✓ Benchmark: Strategy execution throughput (6789ms)
      ✓ Benchmark: Task distribution efficiency (7890ms)
      ✓ Benchmark: Consensus decision latency (2345ms)

Test Suites: 1 passed, 1 total
Tests:       18 passed, 18 total
Time:        45.23s

📄 Report saved to: /docs/e2b/SWARM_BENCHMARKS_REPORT.md
```

## 📋 Report Sections

The generated report includes:

1. **Executive Summary** - Performance targets and status
2. **Creation Performance** - Initialization and deployment metrics
3. **Scalability Analysis** - Scaling characteristics and topology comparison
4. **Trading Operations** - Strategy execution and consensus performance
5. **Communication Performance** - Latency and throughput analysis
6. **Resource Utilization** - Memory, CPU, and network usage
7. **Cost Analysis** - Operational costs and ROI
8. **Optimization Recommendations** - Prioritized improvement suggestions
9. **Comparison Charts** - Visual performance analysis
10. **Conclusions** - Key findings and next steps

## 🔧 Common Commands

```bash
# Run specific category
npm run bench:swarm -- -t "Creation Performance"
npm run bench:swarm -- -t "Scalability"
npm run bench:swarm -- -t "Trading Operations"
npm run bench:swarm -- -t "Communication"
npm run bench:swarm -- -t "Resource Usage"
npm run bench:swarm -- -t "Cost Analysis"

# Run specific test
npm run bench:swarm -- -t "Swarm initialization time"

# Verbose output
npm run bench:swarm:full

# Quick creation tests only
npm run bench:swarm:fast
```

## 💰 Cost Estimates

| Agent Count | Topology | Test Duration | Estimated Cost |
|-------------|----------|---------------|----------------|
| 3 agents | Mesh | 5 minutes | ~$0.01 |
| 8 agents | Hierarchical | 20 minutes | ~$0.05 |
| 20 agents | Hierarchical | 45 minutes | ~$0.15 |

**Full suite**: ~$0.20 for all benchmarks

## 🏆 Best Practices

### 1. Run Regularly
- Weekly for trend analysis
- Before releases
- After infrastructure changes

### 2. Compare Results
- Keep historical reports
- Track performance trends
- Identify regressions early

### 3. Optimize Based on Data
- Review recommendations
- Test optimizations
- Measure improvements

### 4. Monitor Costs
- Set budget alerts
- Track usage patterns
- Optimize for efficiency

## 🐛 Troubleshooting

### E2B Not Connected

```bash
# Check API key
echo $E2B_API_KEY

# Test connection
node -e "require('@e2b/code-interpreter').Sandbox.create().then(s => { console.log('✅ Connected'); s.close(); }).catch(e => console.error('❌ Failed:', e.message))"
```

### Tests Timeout

```bash
# Increase timeout
npm run bench:swarm -- --testTimeout=900000  # 15 minutes
```

### Memory Issues

```bash
# Increase Node memory
NODE_OPTIONS=--max-old-space-size=4096 npm run bench:swarm
```

## 📚 Additional Resources

- **[Full Guide](./BENCHMARK_GUIDE.md)** - Detailed documentation
- **[Report Template](./SWARM_BENCHMARKS_REPORT.md)** - Example report
- **[E2B Docs](https://e2b.dev/docs)** - E2B documentation
- **[Test Code](../../tests/e2b/swarm-benchmarks.test.js)** - Benchmark implementation

## 🎓 Understanding Results

### Mean vs P95 vs P99

- **Mean**: Average performance (typical case)
- **P95**: 95th percentile (worst 5% cases)
- **P99**: 99th percentile (worst 1% cases)

Focus on P95/P99 for production readiness.

### Topology Selection

- **Mesh (2-8 agents)**: Fastest, best for small swarms
- **Hierarchical (10+ agents)**: Most scalable, cost-efficient
- **Ring (5-15 agents)**: Balanced performance

### Cost Optimization

1. Choose appropriate topology for scale
2. Use connection pooling
3. Implement result caching
4. Monitor and optimize resource usage

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-repo/neural-trader/issues)
- **Docs**: [Full Documentation](./BENCHMARK_GUIDE.md)
- **E2B Support**: [E2B Help](https://e2b.dev/docs)

---

**Ready to start?** Run: `npm run bench:swarm:fast`
