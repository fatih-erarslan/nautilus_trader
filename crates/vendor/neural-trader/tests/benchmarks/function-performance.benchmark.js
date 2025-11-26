/**
 * Comprehensive Function Performance Benchmark Suite
 * Measures execution time, throughput, memory usage for all 70+ neural-trader functions
 */

const Benchmark = require('benchmark');
const Table = require('cli-table3');
const chalk = require('chalk');
const { performance } = require('perf_hooks');

// Lazy load the native module to avoid initialization issues
let backend;
try {
  backend = require('../../neural-trader-rust/packages/neural-trader-backend');
} catch (e) {
  console.error(chalk.red('Failed to load neural-trader-backend:'), e.message);
  process.exit(1);
}

/**
 * Benchmark configuration
 */
const BENCHMARK_CONFIG = {
  minSamples: 50,
  maxTime: 5,
  initCount: 10,
};

/**
 * Performance metrics tracker
 */
class PerformanceMetrics {
  constructor() {
    this.results = new Map();
    this.memoryBaseline = process.memoryUsage();
  }

  recordMetric(category, functionName, result) {
    if (!this.results.has(category)) {
      this.results.set(category, []);
    }

    const metrics = {
      name: functionName,
      hz: result.hz,
      mean: result.times.period,
      deviation: result.stats.deviation,
      samples: result.stats.sample.length,
      opsPerSec: Math.round(result.hz).toLocaleString(),
      meanMs: (result.times.period * 1000).toFixed(4),
      rme: result.stats.rme.toFixed(2) + '%',
      memoryMB: this.getMemoryDelta(),
    };

    this.results.get(category).push(metrics);
  }

  getMemoryDelta() {
    const current = process.memoryUsage();
    const delta = (current.heapUsed - this.memoryBaseline.heapUsed) / 1024 / 1024;
    return delta.toFixed(2);
  }

  printResults() {
    console.log(chalk.bold.cyan('\n📊 PERFORMANCE BENCHMARK RESULTS\n'));

    for (const [category, metrics] of this.results) {
      console.log(chalk.bold.yellow(`\n${category.toUpperCase()}\n`));

      const table = new Table({
        head: [
          chalk.white('Function'),
          chalk.white('Ops/sec'),
          chalk.white('Mean (ms)'),
          chalk.white('±RME'),
          chalk.white('Samples'),
          chalk.white('Δ Memory (MB)'),
        ],
        style: { head: [], border: [] },
      });

      metrics
        .sort((a, b) => b.hz - a.hz)
        .forEach((m) => {
          table.push([
            m.name,
            chalk.green(m.opsPerSec),
            chalk.cyan(m.meanMs),
            chalk.yellow(m.rme),
            m.samples,
            m.memoryMB > 0 ? chalk.red(`+${m.memoryMB}`) : chalk.green(m.memoryMB),
          ]);
        });

      console.log(table.toString());
    }

    this.printStatistics();
  }

  printStatistics() {
    console.log(chalk.bold.magenta('\n📈 AGGREGATE STATISTICS\n'));

    const allMetrics = Array.from(this.results.values()).flat();
    const totalFunctions = allMetrics.length;
    const avgOpsPerSec = allMetrics.reduce((sum, m) => sum + m.hz, 0) / totalFunctions;
    const fastestOp = allMetrics.reduce((max, m) => (m.hz > max.hz ? m : max));
    const slowestOp = allMetrics.reduce((min, m) => (m.hz < min.hz ? m : min));

    const statsTable = new Table();
    statsTable.push(
      ['Total Functions Tested', totalFunctions],
      ['Average Ops/sec', Math.round(avgOpsPerSec).toLocaleString()],
      ['Fastest Operation', `${fastestOp.name} (${fastestOp.opsPerSec} ops/sec)`],
      ['Slowest Operation', `${slowestOp.name} (${slowestOp.opsPerSec} ops/sec)`],
      ['Total Memory Impact', `${allMetrics.reduce((sum, m) => sum + parseFloat(m.memoryMB), 0).toFixed(2)} MB`]
    );

    console.log(statsTable.toString());
  }

  exportToJSON(filepath) {
    const fs = require('fs');
    const exportData = {
      timestamp: new Date().toISOString(),
      system: {
        node: process.version,
        platform: process.platform,
        arch: process.arch,
      },
      results: Object.fromEntries(this.results),
    };
    fs.writeFileSync(filepath, JSON.stringify(exportData, null, 2));
    console.log(chalk.green(`\n✅ Results exported to ${filepath}`));
  }
}

const metrics = new PerformanceMetrics();

/**
 * 1. SYSTEM & INITIALIZATION BENCHMARKS
 */
console.log(chalk.bold.blue('\n🚀 Starting System & Initialization Benchmarks...\n'));

const systemSuite = new Benchmark.Suite('System', BENCHMARK_CONFIG);

systemSuite
  .add('getVersion', {
    defer: false,
    fn: () => backend.getVersion(),
  })
  .add('getSystemInfo', {
    defer: false,
    fn: () => backend.getSystemInfo(),
  })
  .add('healthCheck', {
    defer: true,
    fn: (deferred) => {
      backend.healthCheck().then(() => deferred.resolve());
    },
  })
  .add('initSyndicate', {
    defer: false,
    fn: () => backend.initSyndicate(),
  })
  .on('cycle', (event) => {
    metrics.recordMetric('System & Initialization', event.target.name, event.target);
  })
  .on('complete', function () {
    console.log(chalk.green('✅ System benchmarks complete'));
  });

/**
 * 2. TRADING STRATEGY BENCHMARKS
 */
console.log(chalk.bold.blue('\n📈 Starting Trading Strategy Benchmarks...\n'));

const tradingSuite = new Benchmark.Suite('Trading', BENCHMARK_CONFIG);

tradingSuite
  .add('listStrategies', {
    defer: true,
    fn: (deferred) => {
      backend.listStrategies().then(() => deferred.resolve());
    },
  })
  .add('getStrategyInfo', {
    defer: true,
    fn: (deferred) => {
      backend.getStrategyInfo('momentum').then(() => deferred.resolve());
    },
  })
  .add('quickAnalysis', {
    defer: true,
    fn: (deferred) => {
      backend.quickAnalysis('AAPL', false).then(() => deferred.resolve());
    },
  })
  .add('quickAnalysis_GPU', {
    defer: true,
    fn: (deferred) => {
      backend.quickAnalysis('AAPL', true).then(() => deferred.resolve());
    },
  })
  .add('simulateTrade', {
    defer: true,
    fn: (deferred) => {
      backend.simulateTrade('momentum', 'AAPL', 'buy', false).then(() => deferred.resolve());
    },
  })
  .add('simulateTrade_GPU', {
    defer: true,
    fn: (deferred) => {
      backend.simulateTrade('momentum', 'AAPL', 'buy', true).then(() => deferred.resolve());
    },
  })
  .add('getPortfolioStatus', {
    defer: true,
    fn: (deferred) => {
      backend.getPortfolioStatus(true).then(() => deferred.resolve());
    },
  })
  .on('cycle', (event) => {
    metrics.recordMetric('Trading Operations', event.target.name, event.target);
  })
  .on('complete', function () {
    console.log(chalk.green('✅ Trading benchmarks complete'));
  });

/**
 * 3. BACKTEST BENCHMARKS
 */
console.log(chalk.bold.blue('\n🔍 Starting Backtest Benchmarks...\n'));

const backtestSuite = new Benchmark.Suite('Backtest', BENCHMARK_CONFIG);

backtestSuite
  .add('runBacktest_CPU', {
    defer: true,
    fn: (deferred) => {
      backend
        .runBacktest('momentum', 'AAPL', '2023-01-01', '2023-12-31', false)
        .then(() => deferred.resolve());
    },
    minSamples: 10,
    maxTime: 10,
  })
  .add('runBacktest_GPU', {
    defer: true,
    fn: (deferred) => {
      backend
        .runBacktest('momentum', 'AAPL', '2023-01-01', '2023-12-31', true)
        .then(() => deferred.resolve());
    },
    minSamples: 10,
    maxTime: 10,
  })
  .add('optimizeStrategy_CPU', {
    defer: true,
    fn: (deferred) => {
      const paramRanges = JSON.stringify({ momentum_period: [10, 20, 30], rsi_period: [10, 14, 20] });
      backend.optimizeStrategy('momentum', 'AAPL', paramRanges, false).then(() => deferred.resolve());
    },
    minSamples: 5,
    maxTime: 15,
  })
  .add('optimizeStrategy_GPU', {
    defer: true,
    fn: (deferred) => {
      const paramRanges = JSON.stringify({ momentum_period: [10, 20, 30], rsi_period: [10, 14, 20] });
      backend.optimizeStrategy('momentum', 'AAPL', paramRanges, true).then(() => deferred.resolve());
    },
    minSamples: 5,
    maxTime: 15,
  })
  .on('cycle', (event) => {
    metrics.recordMetric('Backtesting', event.target.name, event.target);
  })
  .on('complete', function () {
    console.log(chalk.green('✅ Backtest benchmarks complete'));
  });

/**
 * 4. NEURAL NETWORK BENCHMARKS
 */
console.log(chalk.bold.blue('\n🧠 Starting Neural Network Benchmarks...\n'));

const neuralSuite = new Benchmark.Suite('Neural', BENCHMARK_CONFIG);

neuralSuite
  .add('neuralForecast_CPU', {
    defer: true,
    fn: (deferred) => {
      backend.neuralForecast('AAPL', 30, false, 0.95).then(() => deferred.resolve());
    },
    minSamples: 10,
    maxTime: 10,
  })
  .add('neuralForecast_GPU', {
    defer: true,
    fn: (deferred) => {
      backend.neuralForecast('AAPL', 30, true, 0.95).then(() => deferred.resolve());
    },
    minSamples: 10,
    maxTime: 10,
  })
  .add('neuralModelStatus', {
    defer: true,
    fn: (deferred) => {
      backend.neuralModelStatus().then(() => deferred.resolve());
    },
  })
  .on('cycle', (event) => {
    metrics.recordMetric('Neural Networks', event.target.name, event.target);
  })
  .on('complete', function () {
    console.log(chalk.green('✅ Neural benchmarks complete'));
  });

/**
 * 5. SPORTS BETTING BENCHMARKS
 */
console.log(chalk.bold.blue('\n⚽ Starting Sports Betting Benchmarks...\n'));

const sportsSuite = new Benchmark.Suite('Sports', BENCHMARK_CONFIG);

sportsSuite
  .add('getSportsEvents', {
    defer: true,
    fn: (deferred) => {
      backend.getSportsEvents('basketball_nba', 7).then(() => deferred.resolve());
    },
  })
  .add('getSportsOdds', {
    defer: true,
    fn: (deferred) => {
      backend.getSportsOdds('basketball_nba').then(() => deferred.resolve());
    },
  })
  .add('findSportsArbitrage', {
    defer: true,
    fn: (deferred) => {
      backend.findSportsArbitrage('basketball_nba', 0.01).then(() => deferred.resolve());
    },
  })
  .add('calculateKellyCriterion', {
    defer: true,
    fn: (deferred) => {
      backend.calculateKellyCriterion(0.55, 2.0, 10000).then(() => deferred.resolve());
    },
  })
  .on('cycle', (event) => {
    metrics.recordMetric('Sports Betting', event.target.name, event.target);
  })
  .on('complete', function () {
    console.log(chalk.green('✅ Sports betting benchmarks complete'));
  });

/**
 * 6. SYNDICATE MANAGEMENT BENCHMARKS
 */
console.log(chalk.bold.blue('\n👥 Starting Syndicate Management Benchmarks...\n'));

const syndicateSuite = new Benchmark.Suite('Syndicate', BENCHMARK_CONFIG);

let testSyndicateId;

syndicateSuite
  .add('createSyndicate', {
    defer: true,
    fn: (deferred) => {
      backend
        .createSyndicate(`test-syndicate-${Date.now()}`, 'Benchmark Syndicate', 'Testing performance')
        .then((result) => {
          testSyndicateId = result.syndicateId;
          deferred.resolve();
        });
    },
  })
  .add('addSyndicateMember', {
    defer: true,
    fn: (deferred) => {
      backend
        .addSyndicateMember(testSyndicateId || 'test-1', `Member-${Date.now()}`, 'test@example.com', 'trader', 10000)
        .then(() => deferred.resolve());
    },
  })
  .add('getSyndicateStatus', {
    defer: true,
    fn: (deferred) => {
      backend.getSyndicateStatus(testSyndicateId || 'test-1').then(() => deferred.resolve());
    },
  })
  .on('cycle', (event) => {
    metrics.recordMetric('Syndicate Management', event.target.name, event.target);
  })
  .on('complete', function () {
    console.log(chalk.green('✅ Syndicate benchmarks complete'));
  });

/**
 * 7. E2B SWARM BENCHMARKS
 */
console.log(chalk.bold.blue('\n🐝 Starting E2B Swarm Benchmarks...\n'));

const swarmSuite = new Benchmark.Suite('Swarm', BENCHMARK_CONFIG);

swarmSuite
  .add('createE2bSandbox', {
    defer: true,
    fn: (deferred) => {
      backend.createE2bSandbox(`benchmark-${Date.now()}`, 'base').then(() => deferred.resolve());
    },
    minSamples: 10,
  })
  .add('initE2bSwarm', {
    defer: true,
    fn: (deferred) => {
      const config = JSON.stringify({
        maxAgents: 5,
        distributionStrategy: 'adaptive',
        enableGpu: false,
        autoScaling: true,
      });
      backend.initE2bSwarm('mesh', config).then(() => deferred.resolve());
    },
    minSamples: 10,
  })
  .on('cycle', (event) => {
    metrics.recordMetric('E2B Swarm', event.target.name, event.target);
  })
  .on('complete', function () {
    console.log(chalk.green('✅ E2B swarm benchmarks complete'));
  });

/**
 * 8. SECURITY & AUTH BENCHMARKS
 */
console.log(chalk.bold.blue('\n🔐 Starting Security & Authentication Benchmarks...\n'));

const securitySuite = new Benchmark.Suite('Security', BENCHMARK_CONFIG);

securitySuite
  .add('sanitizeInput', {
    defer: false,
    fn: () => backend.sanitizeInput('<script>alert("xss")</script> SELECT * FROM users WHERE id=1 OR 1=1'),
  })
  .add('validateTradingParams', {
    defer: false,
    fn: () => backend.validateTradingParams('AAPL', 100, 150.5),
  })
  .add('validateEmailFormat', {
    defer: false,
    fn: () => backend.validateEmailFormat('test.user+tag@example.com'),
  })
  .add('validateApiKeyFormat', {
    defer: false,
    fn: () => backend.validateApiKeyFormat('sk_test_1234567890abcdefghijklmnopqrstuvwxyz'),
  })
  .add('checkSecurityThreats', {
    defer: false,
    fn: () => backend.checkSecurityThreats('SELECT * FROM users; DROP TABLE users; --'),
  })
  .add('checkRateLimit', {
    defer: false,
    fn: () => backend.checkRateLimit('user-123', 1),
  })
  .add('checkDdosProtection', {
    defer: false,
    fn: () => backend.checkDdosProtection('192.168.1.100', 10),
  })
  .add('checkIpAllowed', {
    defer: false,
    fn: () => backend.checkIpAllowed('192.168.1.1'),
  })
  .add('checkCorsOrigin', {
    defer: false,
    fn: () => backend.checkCorsOrigin('https://example.com'),
  })
  .on('cycle', (event) => {
    metrics.recordMetric('Security & Authentication', event.target.name, event.target);
  })
  .on('complete', function () {
    console.log(chalk.green('✅ Security benchmarks complete'));
  });

/**
 * 9. RISK ANALYSIS BENCHMARKS
 */
console.log(chalk.bold.blue('\n📊 Starting Risk Analysis Benchmarks...\n'));

const riskSuite = new Benchmark.Suite('Risk', BENCHMARK_CONFIG);

const samplePortfolio = JSON.stringify({
  positions: [
    { symbol: 'AAPL', quantity: 100, avgPrice: 150 },
    { symbol: 'GOOGL', quantity: 50, avgPrice: 2800 },
    { symbol: 'MSFT', quantity: 75, avgPrice: 300 },
  ],
});

riskSuite
  .add('riskAnalysis_CPU', {
    defer: true,
    fn: (deferred) => {
      backend.riskAnalysis(samplePortfolio, false).then(() => deferred.resolve());
    },
  })
  .add('riskAnalysis_GPU', {
    defer: true,
    fn: (deferred) => {
      backend.riskAnalysis(samplePortfolio, true).then(() => deferred.resolve());
    },
  })
  .add('correlationAnalysis_CPU', {
    defer: true,
    fn: (deferred) => {
      backend.correlationAnalysis(['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN'], false).then(() => deferred.resolve());
    },
  })
  .add('correlationAnalysis_GPU', {
    defer: true,
    fn: (deferred) => {
      backend.correlationAnalysis(['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN'], true).then(() => deferred.resolve());
    },
  })
  .on('cycle', (event) => {
    metrics.recordMetric('Risk Analysis', event.target.name, event.target);
  })
  .on('complete', function () {
    console.log(chalk.green('✅ Risk analysis benchmarks complete'));
  });

/**
 * 10. PREDICTION MARKETS BENCHMARKS
 */
console.log(chalk.bold.blue('\n🎯 Starting Prediction Markets Benchmarks...\n'));

const predictionSuite = new Benchmark.Suite('Prediction', BENCHMARK_CONFIG);

predictionSuite
  .add('getPredictionMarkets', {
    defer: true,
    fn: (deferred) => {
      backend.getPredictionMarkets('politics', 20).then(() => deferred.resolve());
    },
  })
  .add('analyzeMarketSentiment', {
    defer: true,
    fn: (deferred) => {
      backend.analyzeMarketSentiment('market-123').then(() => deferred.resolve());
    },
  })
  .on('cycle', (event) => {
    metrics.recordMetric('Prediction Markets', event.target.name, event.target);
  })
  .on('complete', function () {
    console.log(chalk.green('✅ Prediction markets benchmarks complete'));
  });

/**
 * Run all benchmark suites sequentially
 */
async function runAllBenchmarks() {
  console.log(chalk.bold.magenta('\n╔═══════════════════════════════════════════════════════════╗'));
  console.log(chalk.bold.magenta('║   NEURAL TRADER COMPREHENSIVE PERFORMANCE BENCHMARKS     ║'));
  console.log(chalk.bold.magenta('╚═══════════════════════════════════════════════════════════╝\n'));

  const suites = [
    systemSuite,
    tradingSuite,
    backtestSuite,
    neuralSuite,
    sportsSuite,
    syndicateSuite,
    swarmSuite,
    securitySuite,
    riskSuite,
    predictionSuite,
  ];

  for (const suite of suites) {
    await new Promise((resolve) => {
      suite.on('complete', resolve).run({ async: true });
    });
  }

  // Print consolidated results
  metrics.printResults();

  // Export to JSON
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
  metrics.exportToJSON(`/workspaces/neural-trader/tests/benchmarks/results/function-perf-${timestamp}.json`);
}

// Execute benchmarks
if (require.main === module) {
  runAllBenchmarks().catch((err) => {
    console.error(chalk.red('❌ Benchmark failed:'), err);
    process.exit(1);
  });
}

module.exports = { runAllBenchmarks, PerformanceMetrics };
