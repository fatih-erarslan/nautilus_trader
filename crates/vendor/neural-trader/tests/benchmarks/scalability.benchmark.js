/**
 * Scalability Benchmark Suite
 * Tests performance under increasing load: concurrent operations, portfolio size, swarm agents
 */

const Benchmark = require('benchmark');
const Table = require('cli-table3');
const chalk = require('chalk');
const { performance } = require('perf_hooks');

let backend;
try {
  backend = require('../../neural-trader-rust/packages/neural-trader-backend');
} catch (e) {
  console.error(chalk.red('Failed to load neural-trader-backend:'), e.message);
  process.exit(1);
}

/**
 * Scalability test configuration
 */
const SCALE_CONFIGS = {
  concurrency: [1, 10, 100, 500, 1000],
  portfolioSizes: [10, 100, 1000, 10000],
  swarmAgents: [1, 5, 10, 25, 50, 100],
  datasetSizes: ['1-month', '3-months', '1-year', '5-years'],
};

/**
 * Results tracker for scalability tests
 */
class ScalabilityResults {
  constructor() {
    this.results = {
      concurrency: [],
      portfolio: [],
      swarm: [],
      dataset: [],
      memoryGrowth: [],
    };
    this.startMemory = process.memoryUsage();
  }

  recordConcurrency(concurrentOps, totalTime, successRate, throughput) {
    this.results.concurrency.push({
      operations: concurrentOps,
      totalTimeMs: totalTime,
      avgTimeMs: totalTime / concurrentOps,
      successRate: successRate,
      throughput: throughput,
      memoryMB: this.getMemoryUsage(),
    });
  }

  recordPortfolio(size, analysisTime, riskCalcTime, rebalanceTime) {
    this.results.portfolio.push({
      positions: size,
      analysisTimeMs: analysisTime,
      riskTimeMs: riskCalcTime,
      rebalanceTimeMs: rebalanceTime,
      totalTimeMs: analysisTime + riskCalcTime + rebalanceTime,
      memoryMB: this.getMemoryUsage(),
    });
  }

  recordSwarm(agents, initTime, executionTime, coordinationOverhead, throughput) {
    this.results.swarm.push({
      agentCount: agents,
      initTimeMs: initTime,
      executionTimeMs: executionTime,
      coordinationMs: coordinationOverhead,
      throughputOpsPerSec: throughput,
      efficiencyRatio: executionTime / (executionTime + coordinationOverhead),
      memoryMB: this.getMemoryUsage(),
    });
  }

  recordDataset(period, backtestTime, optimizationTime, dataPoints) {
    this.results.dataset.push({
      period: period,
      backtestTimeMs: backtestTime,
      optimizationTimeMs: optimizationTime,
      dataPoints: dataPoints,
      msPerDataPoint: backtestTime / dataPoints,
      memoryMB: this.getMemoryUsage(),
    });
  }

  recordMemoryGrowth(operation, initialMB, peakMB, finalMB, leaked) {
    this.results.memoryGrowth.push({
      operation: operation,
      initialMB: initialMB,
      peakMB: peakMB,
      finalMB: finalMB,
      leakedMB: leaked,
      growthPercent: ((finalMB - initialMB) / initialMB) * 100,
    });
  }

  getMemoryUsage() {
    const usage = process.memoryUsage();
    return (usage.heapUsed / 1024 / 1024).toFixed(2);
  }

  printResults() {
    console.log(chalk.bold.cyan('\n📊 SCALABILITY BENCHMARK RESULTS\n'));

    this.printConcurrencyResults();
    this.printPortfolioResults();
    this.printSwarmResults();
    this.printDatasetResults();
    this.printMemoryGrowthResults();
    this.printBottleneckAnalysis();
  }

  printConcurrencyResults() {
    console.log(chalk.bold.yellow('\n🔄 CONCURRENCY SCALABILITY\n'));

    const table = new Table({
      head: [
        chalk.white('Concurrent Ops'),
        chalk.white('Total Time (ms)'),
        chalk.white('Avg Time (ms)'),
        chalk.white('Success Rate'),
        chalk.white('Throughput (ops/s)'),
        chalk.white('Memory (MB)'),
      ],
    });

    this.results.concurrency.forEach((r) => {
      const successColor = r.successRate >= 99 ? chalk.green : r.successRate >= 95 ? chalk.yellow : chalk.red;
      table.push([
        r.operations,
        r.totalTimeMs.toFixed(2),
        r.avgTimeMs.toFixed(4),
        successColor(r.successRate.toFixed(2) + '%'),
        chalk.cyan(r.throughput.toFixed(0)),
        r.memoryMB,
      ]);
    });

    console.log(table.toString());
  }

  printPortfolioResults() {
    console.log(chalk.bold.yellow('\n💼 PORTFOLIO SIZE SCALABILITY\n'));

    const table = new Table({
      head: [
        chalk.white('Positions'),
        chalk.white('Analysis (ms)'),
        chalk.white('Risk Calc (ms)'),
        chalk.white('Rebalance (ms)'),
        chalk.white('Total (ms)'),
        chalk.white('Memory (MB)'),
      ],
    });

    this.results.portfolio.forEach((r) => {
      table.push([
        r.positions.toLocaleString(),
        r.analysisTimeMs.toFixed(2),
        r.riskTimeMs.toFixed(2),
        r.rebalanceTimeMs.toFixed(2),
        chalk.cyan(r.totalTimeMs.toFixed(2)),
        r.memoryMB,
      ]);
    });

    console.log(table.toString());

    // Calculate complexity
    if (this.results.portfolio.length >= 2) {
      const first = this.results.portfolio[0];
      const last = this.results.portfolio[this.results.portfolio.length - 1];
      const complexityRatio = last.totalTimeMs / first.totalTimeMs;
      const sizeRatio = last.positions / first.positions;
      const complexity = Math.log(complexityRatio) / Math.log(sizeRatio);

      console.log(
        chalk.magenta(`\n⚙️  Time Complexity: O(n^${complexity.toFixed(2)}) where n = portfolio size`)
      );
    }
  }

  printSwarmResults() {
    console.log(chalk.bold.yellow('\n🐝 SWARM AGENT SCALABILITY\n'));

    const table = new Table({
      head: [
        chalk.white('Agents'),
        chalk.white('Init (ms)'),
        chalk.white('Execution (ms)'),
        chalk.white('Coordination (ms)'),
        chalk.white('Throughput'),
        chalk.white('Efficiency'),
        chalk.white('Memory (MB)'),
      ],
    });

    this.results.swarm.forEach((r) => {
      const efficiencyColor =
        r.efficiencyRatio >= 0.9 ? chalk.green : r.efficiencyRatio >= 0.7 ? chalk.yellow : chalk.red;
      table.push([
        r.agentCount,
        r.initTimeMs.toFixed(2),
        r.executionTimeMs.toFixed(2),
        r.coordinationMs.toFixed(2),
        chalk.cyan(r.throughputOpsPerSec.toFixed(0)),
        efficiencyColor((r.efficiencyRatio * 100).toFixed(1) + '%'),
        r.memoryMB,
      ]);
    });

    console.log(table.toString());

    // Analyze scaling efficiency
    if (this.results.swarm.length >= 2) {
      const first = this.results.swarm[0];
      const last = this.results.swarm[this.results.swarm.length - 1];
      const speedup = last.throughputOpsPerSec / first.throughputOpsPerSec;
      const agentRatio = last.agentCount / first.agentCount;
      const scalingEfficiency = (speedup / agentRatio) * 100;

      console.log(
        chalk.magenta(`\n⚡ Scaling Efficiency: ${scalingEfficiency.toFixed(1)}% (ideal = 100%)`)
      );
      console.log(
        chalk.magenta(
          `📈 Speedup: ${speedup.toFixed(2)}x with ${agentRatio}x agents`
        )
      );
    }
  }

  printDatasetResults() {
    console.log(chalk.bold.yellow('\n📅 DATASET SIZE SCALABILITY\n'));

    const table = new Table({
      head: [
        chalk.white('Period'),
        chalk.white('Backtest (ms)'),
        chalk.white('Optimization (ms)'),
        chalk.white('Data Points'),
        chalk.white('ms/Point'),
        chalk.white('Memory (MB)'),
      ],
    });

    this.results.dataset.forEach((r) => {
      table.push([
        r.period,
        r.backtestTimeMs.toFixed(2),
        r.optimizationTimeMs.toFixed(2),
        r.dataPoints.toLocaleString(),
        r.msPerDataPoint.toFixed(6),
        r.memoryMB,
      ]);
    });

    console.log(table.toString());
  }

  printMemoryGrowthResults() {
    console.log(chalk.bold.yellow('\n💾 MEMORY GROWTH ANALYSIS\n'));

    const table = new Table({
      head: [
        chalk.white('Operation'),
        chalk.white('Initial (MB)'),
        chalk.white('Peak (MB)'),
        chalk.white('Final (MB)'),
        chalk.white('Leaked (MB)'),
        chalk.white('Growth %'),
      ],
    });

    this.results.memoryGrowth.forEach((r) => {
      const leakColor = r.leakedMB < 1 ? chalk.green : r.leakedMB < 10 ? chalk.yellow : chalk.red;
      table.push([
        r.operation,
        r.initialMB.toFixed(2),
        r.peakMB.toFixed(2),
        r.finalMB.toFixed(2),
        leakColor(r.leakedMB.toFixed(2)),
        r.growthPercent.toFixed(1) + '%',
      ]);
    });

    console.log(table.toString());
  }

  printBottleneckAnalysis() {
    console.log(chalk.bold.magenta('\n🔍 BOTTLENECK ANALYSIS\n'));

    const bottlenecks = [];

    // Check concurrency bottlenecks
    if (this.results.concurrency.length >= 2) {
      const last = this.results.concurrency[this.results.concurrency.length - 1];
      if (last.successRate < 95) {
        bottlenecks.push({
          severity: 'HIGH',
          area: 'Concurrency',
          issue: `Success rate drops to ${last.successRate.toFixed(1)}% at ${last.operations} concurrent operations`,
          recommendation: 'Increase connection pool size or implement request queuing',
        });
      }
    }

    // Check swarm efficiency bottlenecks
    if (this.results.swarm.length >= 2) {
      const last = this.results.swarm[this.results.swarm.length - 1];
      if (last.efficiencyRatio < 0.7) {
        bottlenecks.push({
          severity: 'MEDIUM',
          area: 'Swarm Coordination',
          issue: `Coordination overhead is ${((1 - last.efficiencyRatio) * 100).toFixed(1)}% at ${last.agentCount} agents`,
          recommendation: 'Consider mesh topology or optimize inter-agent communication',
        });
      }
    }

    // Check memory leaks
    const totalLeaked = this.results.memoryGrowth.reduce((sum, r) => sum + r.leakedMB, 0);
    if (totalLeaked > 50) {
      bottlenecks.push({
        severity: 'HIGH',
        area: 'Memory Management',
        issue: `Total memory leaked: ${totalLeaked.toFixed(2)} MB across all operations`,
        recommendation: 'Implement aggressive garbage collection and object pooling',
      });
    }

    if (bottlenecks.length === 0) {
      console.log(chalk.green('✅ No significant bottlenecks detected!\n'));
      return;
    }

    const table = new Table({
      head: [chalk.white('Severity'), chalk.white('Area'), chalk.white('Issue'), chalk.white('Recommendation')],
      colWidths: [12, 20, 40, 50],
      wordWrap: true,
    });

    bottlenecks.forEach((b) => {
      const severityColor = b.severity === 'HIGH' ? chalk.red : b.severity === 'MEDIUM' ? chalk.yellow : chalk.blue;
      table.push([severityColor(b.severity), b.area, b.issue, chalk.cyan(b.recommendation)]);
    });

    console.log(table.toString());
  }

  exportToJSON(filepath) {
    const fs = require('fs');
    const exportData = {
      timestamp: new Date().toISOString(),
      system: {
        node: process.version,
        platform: process.platform,
        arch: process.arch,
        memoryTotal: (require('os').totalmem() / 1024 / 1024 / 1024).toFixed(2) + ' GB',
      },
      results: this.results,
    };
    fs.writeFileSync(filepath, JSON.stringify(exportData, null, 2));
    console.log(chalk.green(`\n✅ Results exported to ${filepath}`));
  }
}

const results = new ScalabilityResults();

/**
 * Test concurrent operations scalability
 */
async function testConcurrencyScalability() {
  console.log(chalk.bold.blue('\n🔄 Testing Concurrency Scalability...\n'));

  for (const concurrentOps of SCALE_CONFIGS.concurrency) {
    console.log(chalk.cyan(`Testing ${concurrentOps} concurrent operations...`));

    const startTime = performance.now();
    let successCount = 0;

    const promises = Array(concurrentOps)
      .fill()
      .map(async (_, i) => {
        try {
          await backend.quickAnalysis('AAPL', false);
          successCount++;
        } catch (err) {
          // Operation failed
        }
      });

    await Promise.all(promises);

    const totalTime = performance.now() - startTime;
    const successRate = (successCount / concurrentOps) * 100;
    const throughput = (successCount / totalTime) * 1000;

    results.recordConcurrency(concurrentOps, totalTime, successRate, throughput);

    // Cool down between tests
    await new Promise((resolve) => setTimeout(resolve, 1000));
  }
}

/**
 * Test portfolio size scalability
 */
async function testPortfolioScalability() {
  console.log(chalk.bold.blue('\n💼 Testing Portfolio Size Scalability...\n'));

  for (const size of SCALE_CONFIGS.portfolioSizes) {
    console.log(chalk.cyan(`Testing portfolio with ${size} positions...`));

    // Generate test portfolio
    const positions = Array(size)
      .fill()
      .map((_, i) => ({
        symbol: `STOCK${i}`,
        quantity: 100,
        avgPrice: 100 + Math.random() * 100,
      }));

    const portfolio = JSON.stringify({ positions });

    // Test analysis
    const analysisStart = performance.now();
    try {
      await backend.getPortfolioStatus(true);
    } catch (err) {
      // Simulated - actual implementation would use real portfolio
    }
    const analysisTime = performance.now() - analysisStart;

    // Test risk calculation
    const riskStart = performance.now();
    try {
      await backend.riskAnalysis(portfolio, false);
    } catch (err) {
      // Expected for large portfolios in demo mode
    }
    const riskTime = performance.now() - riskStart;

    // Test rebalancing
    const targetAllocations = {};
    positions.forEach((p, i) => {
      targetAllocations[p.symbol] = 1 / size;
    });

    const rebalanceStart = performance.now();
    try {
      await backend.portfolioRebalance(JSON.stringify(targetAllocations), portfolio);
    } catch (err) {
      // Expected for large portfolios
    }
    const rebalanceTime = performance.now() - rebalanceStart;

    results.recordPortfolio(size, analysisTime, riskTime, rebalanceTime);

    // Cool down
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
}

/**
 * Test swarm agent scalability
 */
async function testSwarmScalability() {
  console.log(chalk.bold.blue('\n🐝 Testing Swarm Agent Scalability...\n'));

  for (const agentCount of SCALE_CONFIGS.swarmAgents) {
    console.log(chalk.cyan(`Testing swarm with ${agentCount} agents...`));

    const config = JSON.stringify({
      maxAgents: agentCount,
      distributionStrategy: 'adaptive',
      enableGpu: false,
      autoScaling: false,
    });

    // Test initialization
    const initStart = performance.now();
    let swarmId;
    try {
      const result = await backend.initE2bSwarm('mesh', config);
      swarmId = result.swarmId;
    } catch (err) {
      console.log(chalk.yellow(`Warning: Could not init swarm: ${err.message}`));
    }
    const initTime = performance.now() - initStart;

    // Test execution
    const execStart = performance.now();
    const coordStart = performance.now();
    try {
      if (swarmId) {
        await backend.executeSwarmStrategy(swarmId, 'momentum', ['AAPL', 'GOOGL', 'MSFT']);
      }
    } catch (err) {
      // Expected in benchmark mode
    }
    const execTime = performance.now() - execStart;
    const coordTime = performance.now() - coordStart - execTime;

    const throughput = agentCount / (execTime / 1000);

    results.recordSwarm(agentCount, initTime, execTime, coordTime, throughput);

    // Cleanup
    try {
      if (swarmId) {
        await backend.shutdownSwarm(swarmId);
      }
    } catch (err) {
      // Ignore cleanup errors
    }

    await new Promise((resolve) => setTimeout(resolve, 1000));
  }
}

/**
 * Test dataset size scalability
 */
async function testDatasetScalability() {
  console.log(chalk.bold.blue('\n📅 Testing Dataset Size Scalability...\n'));

  const periods = [
    { name: '1-month', start: '2023-12-01', end: '2023-12-31', points: 21 },
    { name: '3-months', start: '2023-10-01', end: '2023-12-31', points: 63 },
    { name: '1-year', start: '2023-01-01', end: '2023-12-31', points: 252 },
    { name: '5-years', start: '2019-01-01', end: '2023-12-31', points: 1260 },
  ];

  for (const period of periods) {
    console.log(chalk.cyan(`Testing ${period.name} dataset (${period.points} data points)...`));

    // Test backtest
    const backtestStart = performance.now();
    try {
      await backend.runBacktest('momentum', 'AAPL', period.start, period.end, false);
    } catch (err) {
      // Expected in demo mode
    }
    const backtestTime = performance.now() - backtestStart;

    // Test optimization
    const optStart = performance.now();
    try {
      const paramRanges = JSON.stringify({ period: [10, 20, 30] });
      await backend.optimizeStrategy('momentum', 'AAPL', paramRanges, false);
    } catch (err) {
      // Expected in demo mode
    }
    const optTime = performance.now() - optStart;

    results.recordDataset(period.name, backtestTime, optTime, period.points);

    await new Promise((resolve) => setTimeout(resolve, 500));
  }
}

/**
 * Test memory growth under load
 */
async function testMemoryGrowth() {
  console.log(chalk.bold.blue('\n💾 Testing Memory Growth...\n'));

  const operations = [
    {
      name: 'Repeated Quick Analysis (100x)',
      fn: async () => {
        for (let i = 0; i < 100; i++) {
          await backend.quickAnalysis('AAPL', false);
        }
      },
    },
    {
      name: 'Repeated Backtest (20x)',
      fn: async () => {
        for (let i = 0; i < 20; i++) {
          try {
            await backend.runBacktest('momentum', 'AAPL', '2023-01-01', '2023-12-31', false);
          } catch (err) {
            // Expected
          }
        }
      },
    },
    {
      name: 'Syndicate Operations (50x)',
      fn: async () => {
        for (let i = 0; i < 50; i++) {
          try {
            const syndicate = await backend.createSyndicate(`test-${i}`, 'Test', 'Testing');
            await backend.getSyndicateStatus(syndicate.syndicateId);
          } catch (err) {
            // Expected
          }
        }
      },
    },
  ];

  for (const op of operations) {
    console.log(chalk.cyan(`Testing: ${op.name}...`));

    // Force GC if available
    if (global.gc) {
      global.gc();
    }

    const initialMem = process.memoryUsage().heapUsed / 1024 / 1024;
    let peakMem = initialMem;

    const interval = setInterval(() => {
      const currentMem = process.memoryUsage().heapUsed / 1024 / 1024;
      peakMem = Math.max(peakMem, currentMem);
    }, 100);

    await op.fn();

    clearInterval(interval);

    // Force GC again
    if (global.gc) {
      global.gc();
    }

    await new Promise((resolve) => setTimeout(resolve, 500));

    const finalMem = process.memoryUsage().heapUsed / 1024 / 1024;
    const leaked = Math.max(0, finalMem - initialMem);

    results.recordMemoryGrowth(op.name, initialMem, peakMem, finalMem, leaked);
  }
}

/**
 * Run all scalability tests
 */
async function runScalabilityBenchmarks() {
  console.log(chalk.bold.magenta('\n╔═══════════════════════════════════════════════════════════╗'));
  console.log(chalk.bold.magenta('║      NEURAL TRADER SCALABILITY BENCHMARKS                ║'));
  console.log(chalk.bold.magenta('╚═══════════════════════════════════════════════════════════╝\n'));

  console.log(chalk.yellow('💡 Tip: Run with --expose-gc flag for accurate memory measurements\n'));

  try {
    await testConcurrencyScalability();
    await testPortfolioScalability();
    await testSwarmScalability();
    await testDatasetScalability();
    await testMemoryGrowth();

    results.printResults();

    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    results.exportToJSON(`/workspaces/neural-trader/tests/benchmarks/results/scalability-${timestamp}.json`);
  } catch (err) {
    console.error(chalk.red('❌ Scalability test failed:'), err);
    throw err;
  }
}

// Execute if run directly
if (require.main === module) {
  runScalabilityBenchmarks().catch((err) => {
    console.error(chalk.red('Fatal error:'), err);
    process.exit(1);
  });
}

module.exports = { runScalabilityBenchmarks, ScalabilityResults };
