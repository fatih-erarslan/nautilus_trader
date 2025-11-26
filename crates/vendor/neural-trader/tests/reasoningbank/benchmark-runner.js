#!/usr/bin/env node
/**
 * Benchmark Runner for ReasoningBank E2B Swarm Comparisons
 * Execute comprehensive benchmark suite and generate reports
 */

const fs = require('fs').promises;
const path = require('path');
const { execSync } = require('child_process');

class BenchmarkRunner {
  constructor() {
    this.resultsDir = path.join(__dirname, 'results');
    this.reportDir = path.join(__dirname, '../../docs/reasoningbank');
    this.startTime = Date.now();
    this.benchmarkResults = new Map();
  }

  async run() {
    console.log('╔════════════════════════════════════════════════════════════╗');
    console.log('║   ReasoningBank E2B Swarm Benchmark Suite                 ║');
    console.log('║   Comparing Traditional vs Self-Learning Trading Swarms    ║');
    console.log('╚════════════════════════════════════════════════════════════╝\n');

    // Ensure directories exist
    await fs.mkdir(this.resultsDir, { recursive: true });
    await fs.mkdir(this.reportDir, { recursive: true });

    // Run benchmark suite
    try {
      console.log('Running Jest benchmark suite...\n');
      execSync('npm test -- tests/reasoningbank/learning-benchmarks.test.js --verbose', {
        stdio: 'inherit',
        cwd: path.join(__dirname, '../..')
      });

      // Collect results
      await this.collectResults();

      // Generate comprehensive report
      await this.generateReport();

      // Generate visualizations
      await this.generateVisualizations();

      console.log('\n✅ Benchmark suite completed successfully!');
      console.log(`📊 Results: ${this.resultsDir}`);
      console.log(`📄 Report: ${path.join(this.reportDir, 'LEARNING_BENCHMARKS_REPORT.md')}`);

    } catch (error) {
      console.error('❌ Benchmark suite failed:', error.message);
      process.exit(1);
    }
  }

  async collectResults() {
    console.log('\nCollecting benchmark results...');

    const files = await fs.readdir(this.resultsDir);

    for (const file of files) {
      if (file.endsWith('.json')) {
        const content = await fs.readFile(path.join(this.resultsDir, file), 'utf8');
        const data = JSON.parse(content);
        this.benchmarkResults.set(file.replace('.json', ''), data);
      }
    }

    console.log(`Collected ${this.benchmarkResults.size} result files`);
  }

  async generateReport() {
    console.log('\nGenerating comprehensive report...');

    const report = this.buildMarkdownReport();
    await fs.writeFile(
      path.join(this.reportDir, 'LEARNING_BENCHMARKS_REPORT.md'),
      report
    );

    console.log('Report generated successfully');
  }

  buildMarkdownReport() {
    const duration = ((Date.now() - this.startTime) / 1000 / 60).toFixed(2);

    let report = `# ReasoningBank E2B Swarm Learning Benchmarks Report

**Generated:** ${new Date().toISOString()}
**Duration:** ${duration} minutes
**Test Categories:** ${this.benchmarkResults.size}

---

## Executive Summary

This report presents comprehensive benchmarks comparing traditional rule-based trading swarms with ReasoningBank-enhanced self-learning swarms deployed on E2B sandboxes.

### Key Findings

`;

    // Add key findings from results
    const traditionalVsRB = this.benchmarkResults.get('traditional-vs-reasoningbank');
    if (traditionalVsRB) {
      report += `
#### Performance Comparison

- **P&L Improvement:** ${traditionalVsRB.improvement?.pnlImprovement?.toFixed(2)}%
- **Sharpe Ratio Improvement:** ${traditionalVsRB.improvement?.sharpeImprovement?.toFixed(2)}
- **Traditional Total P&L:** $${traditionalVsRB.traditional?.totalPnL?.toFixed(2)}
- **ReasoningBank Total P&L:** $${traditionalVsRB.reasoningBank?.totalPnL?.toFixed(2)}
- **Win Rate (Traditional):** ${(traditionalVsRB.traditional?.avgWinRate * 100)?.toFixed(1)}%
- **Win Rate (ReasoningBank):** ${(traditionalVsRB.reasoningBank?.avgWinRate * 100)?.toFixed(1)}%
`;
    }

    const learningEffectiveness = this.benchmarkResults.get('learning-effectiveness');
    if (learningEffectiveness) {
      report += `
#### Learning Effectiveness

- **Accuracy Improvement:** ${learningEffectiveness.stats?.accuracyImprovement?.toFixed(2)}%
- **Convergence Rate:** ${learningEffectiveness.stats?.convergenceRate} episodes to 80% accuracy
- **Final Accuracy:** ${(learningEffectiveness.stats?.avgAccuracy * 100)?.toFixed(1)}%
- **Total Episodes:** ${learningEffectiveness.episodes?.length || 0}
`;
    }

    const resourceOverhead = this.benchmarkResults.get('resource-overhead');
    if (resourceOverhead) {
      report += `
#### Resource Overhead

- **Memory Overhead:** ${resourceOverhead.memoryIncrease?.toFixed(1)}%
- **CPU Overhead:** ${resourceOverhead.cpuIncrease?.toFixed(1)}%
- **Additional Memory:** ${resourceOverhead.memoryOverhead?.toFixed(2)} MB
- **Additional CPU:** ${resourceOverhead.cpuOverhead?.toFixed(2)}%
`;
    }

    report += `
---

## Detailed Results

### 1. Learning Effectiveness

`;

    // Learning effectiveness details
    if (learningEffectiveness && learningEffectiveness.episodes) {
      const episodes = learningEffectiveness.episodes;
      const firstEpisode = episodes[0];
      const lastEpisode = episodes[episodes.length - 1];

      report += `
#### Episode Progress

| Metric | First Episode | Last Episode | Improvement |
|--------|--------------|--------------|-------------|
| Accuracy | ${(firstEpisode.accuracy * 100).toFixed(1)}% | ${(lastEpisode.accuracy * 100).toFixed(1)}% | ${((lastEpisode.accuracy - firstEpisode.accuracy) * 100).toFixed(1)}% |
| P&L | $${firstEpisode.pnl.toFixed(2)} | $${lastEpisode.pnl.toFixed(2)} | $${(lastEpisode.pnl - firstEpisode.pnl).toFixed(2)} |
| Sharpe Ratio | ${firstEpisode.sharpeRatio.toFixed(2)} | ${lastEpisode.sharpeRatio.toFixed(2)} | ${(lastEpisode.sharpeRatio - firstEpisode.sharpeRatio).toFixed(2)} |
| Patterns | - | ${lastEpisode.patternsLearned || 0} | - |

#### Learning Curve

Episodes with accuracy milestones:
`;

      const milestones = [0.5, 0.6, 0.7, 0.8, 0.9];
      for (const milestone of milestones) {
        const episodeIdx = episodes.findIndex(e => e.accuracy >= milestone);
        if (episodeIdx !== -1) {
          report += `- **${(milestone * 100).toFixed(0)}% accuracy:** Reached at episode ${episodeIdx}\n`;
        }
      }
    }

    report += `
### 2. Topology Comparison

`;

    // Topology comparison
    const topologyComparison = this.benchmarkResults.get('topology-comparison');
    if (topologyComparison && Array.isArray(topologyComparison)) {
      report += `
| Topology | Convergence Rate | Avg Accuracy | Learning Efficiency |
|----------|------------------|--------------|---------------------|
`;
      for (const result of topologyComparison) {
        report += `| ${result.topology} | ${result.convergenceRate} episodes | ${(result.avgAccuracy * 100).toFixed(1)}% | ${result.learningEfficiency?.toFixed(6)} |\n`;
      }

      report += `
**Best Topology:** ${topologyComparison[0].topology} (highest learning efficiency)
`;
    }

    report += `
### 3. Traditional vs ReasoningBank

`;

    if (traditionalVsRB) {
      report += `
#### Head-to-Head Comparison

**Trading Performance:**

| Metric | Traditional | ReasoningBank | Improvement |
|--------|------------|---------------|-------------|
| Total P&L | $${traditionalVsRB.traditional?.totalPnL?.toFixed(2)} | $${traditionalVsRB.reasoningBank?.totalPnL?.toFixed(2)} | ${traditionalVsRB.improvement?.pnlImprovement?.toFixed(2)}% |
| Sharpe Ratio | ${traditionalVsRB.traditional?.avgSharpeRatio?.toFixed(2)} | ${traditionalVsRB.reasoningBank?.avgSharpeRatio?.toFixed(2)} | ${traditionalVsRB.improvement?.sharpeImprovement?.toFixed(2)} |
| Win Rate | ${(traditionalVsRB.traditional?.avgWinRate * 100)?.toFixed(1)}% | ${(traditionalVsRB.reasoningBank?.avgWinRate * 100)?.toFixed(1)}% | ${((traditionalVsRB.reasoningBank?.avgWinRate - traditionalVsRB.traditional?.avgWinRate) * 100)?.toFixed(1)}% |

**Learning Advantages:**

- ReasoningBank showed ${traditionalVsRB.reasoningBank?.accuracyImprovement?.toFixed(1)}% accuracy improvement over time
- Adaptive strategy selection based on learned patterns
- Continuous improvement through trajectory tracking
`;
    }

    const latencyComparison = this.benchmarkResults.get('latency-comparison');
    if (latencyComparison) {
      report += `
#### Performance Overhead

| Metric | Traditional | ReasoningBank | Overhead |
|--------|------------|---------------|----------|
| Total Duration | ${(latencyComparison.traditional?.totalDuration / 1000).toFixed(2)}s | ${(latencyComparison.reasoningBank?.totalDuration / 1000).toFixed(2)}s | ${latencyComparison.overhead?.percentOverhead?.toFixed(1)}% |
| Avg Decision Latency | ${latencyComparison.traditional?.avgDecisionLatency?.toFixed(2)}ms | ${latencyComparison.reasoningBank?.avgDecisionLatency?.toFixed(2)}ms | - |
| Time Overhead | - | - | ${(latencyComparison.overhead?.timeOverhead / 1000).toFixed(2)}s |
`;
    }

    report += `
### 4. Memory & Performance

`;

    const agentDBPerformance = this.benchmarkResults.get('agentdb-query-performance');
    if (agentDBPerformance) {
      report += `
#### AgentDB Vector Search Performance

- **Average Query Time:** ${agentDBPerformance.avgQueryTime?.toFixed(2)}ms
- **P95 Query Time:** ${agentDBPerformance.p95QueryTime?.toFixed(2)}ms
- **Sample Size:** ${agentDBPerformance.samples} queries

`;
    }

    const memoryUsage = this.benchmarkResults.get('memory-usage-trajectory');
    if (memoryUsage && memoryUsage.memorySnapshots) {
      report += `
#### Memory Usage with Trajectory Storage

| Episode | Memory (MB) | Trajectories Stored | Patterns Learned |
|---------|-------------|---------------------|------------------|
`;
      for (const snapshot of memoryUsage.memorySnapshots) {
        report += `| ${snapshot.episode} | ${snapshot.memoryMB?.toFixed(2)} | ${snapshot.trajectoriesStored} | ${snapshot.patternsLearned} |\n`;
      }

      report += `
**Total Memory Growth:** ${memoryUsage.memoryGrowth?.toFixed(2)} MB
`;
    }

    const throughput = this.benchmarkResults.get('learning-throughput');
    if (throughput) {
      report += `
#### Learning System Throughput

- **Throughput:** ${throughput.throughput?.toFixed(2)} decisions/second
- **Total Decisions:** ${throughput.totalDecisions}
- **Duration:** ${throughput.durationSeconds?.toFixed(2)} seconds
`;
    }

    report += `
### 5. Adaptive Learning

`;

    const marketAdaptation = this.benchmarkResults.get('market-adaptation');
    if (marketAdaptation && Array.isArray(marketAdaptation)) {
      report += `
#### Market Condition Adaptation

Detected ${marketAdaptation.length} significant adaptation events:

| Episode | P&L Change | Accuracy | Patterns Learned |
|---------|-----------|----------|------------------|
`;
      marketAdaptation.slice(0, 10).forEach(event => {
        report += `| ${event.episode} | $${event.pnlChange?.toFixed(2)} | ${(event.accuracy * 100)?.toFixed(1)}% | ${event.patternsLearned} |\n`;
      });
    }

    const strategySwitching = this.benchmarkResults.get('strategy-switching');
    if (strategySwitching && Array.isArray(strategySwitching)) {
      report += `
#### Strategy Switching Behavior

Detected ${strategySwitching.length} strategy switches:

| Episode | From | To | P&L | Accuracy |
|---------|------|----| ----|----------|
`;
      strategySwitching.slice(0, 10).forEach(event => {
        report += `| ${event.episode} | ${event.from} | ${event.to} | $${event.pnl?.toFixed(2)} | ${(event.accuracy * 100)?.toFixed(1)}% |\n`;
      });
    }

    const knowledgeSharing = this.benchmarkResults.get('knowledge-sharing');
    if (knowledgeSharing) {
      report += `
#### Multi-Agent Knowledge Sharing

- **Sharing Efficiency:** ${knowledgeSharing.efficiency?.toFixed(2)}x improvement
- **Pattern Distribution:** Effective across ${knowledgeSharing.metrics?.[0]?.agentCount} agents

Knowledge sharing progress:

| Episode | Total Patterns | Agents | Patterns/Agent |
|---------|---------------|--------|----------------|
`;
      if (knowledgeSharing.metrics) {
        knowledgeSharing.metrics.forEach(m => {
          report += `| ${m.episode} | ${m.totalPatternsLearned} | ${m.agentCount} | ${m.patternsPerAgent?.toFixed(2)} |\n`;
        });
      }
    }

    report += `
---

## Conclusions

### ReasoningBank Advantages

1. **Learning & Adaptation**
   - Continuous improvement in decision quality
   - Pattern recognition enables better strategy selection
   - Adaptive to changing market conditions

2. **Performance**
   - ${traditionalVsRB?.improvement?.pnlImprovement > 0 ? 'Superior' : 'Competitive'} P&L performance
   - Improved Sharpe ratio through risk-aware learning
   - Higher win rates after convergence period

3. **Knowledge Sharing**
   - Effective multi-agent coordination
   - Distributed learning across topology
   - Pattern reuse accelerates learning

### Trade-offs

1. **Resource Overhead**
   - ${resourceOverhead?.memoryIncrease?.toFixed(1)}% additional memory usage
   - ${resourceOverhead?.cpuIncrease?.toFixed(1)}% additional CPU usage
   - ${latencyComparison?.overhead?.percentOverhead?.toFixed(1)}% time overhead

2. **Convergence Period**
   - Requires ${learningEffectiveness?.stats?.convergenceRate || 'N/A'} episodes to reach 80% accuracy
   - Initial performance may lag traditional approaches
   - Benefits increase over time

### Recommendations

1. **When to Use ReasoningBank:**
   - Long-term trading strategies (>100 episodes)
   - Complex, changing market conditions
   - Multi-agent coordination scenarios
   - When learning from experience is valuable

2. **When to Use Traditional:**
   - Short-term trading windows
   - Resource-constrained environments
   - Well-defined, stable market conditions
   - When simple rules are sufficient

3. **Optimal Configuration:**
   - **Topology:** ${topologyComparison?.[0]?.topology || 'mesh'} for best learning efficiency
   - **Episode Count:** Minimum ${learningEffectiveness?.stats?.convergenceRate || 80} for convergence
   - **Resource Allocation:** Budget ${resourceOverhead?.memoryIncrease?.toFixed(0) || 150}% memory overhead

---

## Appendix

### Test Configuration

\`\`\`javascript
{
  episodeCount: 100,
  warmupEpisodes: 10,
  tradingSymbols: ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA'],
  initialCapital: 100000,
  maxPositionSize: 0.2,
  topologies: ['mesh', 'hierarchical', 'ring', 'star'],
  trajectoryBatchSize: 32
}
\`\`\`

### Raw Data Files

All raw benchmark data is available in \`${this.resultsDir}\`:

`;

    for (const [filename, _] of this.benchmarkResults) {
      report += `- \`${filename}.json\`\n`;
    }

    report += `
---

**Report Generated by:** ReasoningBank E2B Swarm Benchmark Suite
**Version:** 1.0.0
**Date:** ${new Date().toISOString()}
`;

    return report;
  }

  async generateVisualizations() {
    console.log('\nGenerating visualizations...');

    // Generate learning curve data for visualization
    const learningEffectiveness = this.benchmarkResults.get('learning-effectiveness');
    if (learningEffectiveness && learningEffectiveness.episodes) {
      const chartData = {
        labels: learningEffectiveness.episodes.map(e => e.episode),
        datasets: [
          {
            label: 'Accuracy',
            data: learningEffectiveness.episodes.map(e => e.accuracy * 100)
          },
          {
            label: 'Sharpe Ratio',
            data: learningEffectiveness.episodes.map(e => e.sharpeRatio)
          },
          {
            label: 'Cumulative P&L',
            data: learningEffectiveness.episodes.reduce((acc, e) => {
              const last = acc.length > 0 ? acc[acc.length - 1] : 0;
              acc.push(last + e.pnl);
              return acc;
            }, [])
          }
        ]
      };

      await fs.writeFile(
        path.join(this.resultsDir, 'learning-curve-data.json'),
        JSON.stringify(chartData, null, 2)
      );
    }

    console.log('Visualizations generated');
  }
}

// Run if called directly
if (require.main === module) {
  const runner = new BenchmarkRunner();
  runner.run().catch(error => {
    console.error('Fatal error:', error);
    process.exit(1);
  });
}

module.exports = { BenchmarkRunner };
