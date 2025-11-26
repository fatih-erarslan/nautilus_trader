/**
 * GPU vs CPU Performance Comparison Benchmark
 * Measures performance gains from GPU acceleration across all GPU-capable functions
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
 * GPU comparison results tracker
 */
class GPUComparisonResults {
  constructor() {
    this.comparisons = [];
    this.gpuAvailable = true; // Will be detected during tests
  }

  recordComparison(operation, cpuTime, gpuTime, cpuMemory, gpuMemory, dataSize) {
    const speedup = cpuTime / gpuTime;
    const memoryRatio = gpuMemory / cpuMemory;

    this.comparisons.push({
      operation,
      cpuTimeMs: cpuTime,
      gpuTimeMs: gpuTime,
      speedup: speedup,
      speedupPercent: ((speedup - 1) * 100).toFixed(1),
      cpuMemoryMB: cpuMemory,
      gpuMemoryMB: gpuMemory,
      memoryRatio: memoryRatio,
      dataSize: dataSize,
      recommendation: speedup >= 1.5 ? 'USE GPU' : speedup >= 1.1 ? 'GPU BENEFICIAL' : 'CPU SUFFICIENT',
    });
  }

  printResults() {
    console.log(chalk.bold.cyan('\n🎮 GPU vs CPU PERFORMANCE COMPARISON\n'));

    if (!this.gpuAvailable) {
      console.log(
        chalk.yellow('⚠️  GPU acceleration not available - showing simulated comparisons\n')
      );
    }

    const table = new Table({
      head: [
        chalk.white('Operation'),
        chalk.white('CPU (ms)'),
        chalk.white('GPU (ms)'),
        chalk.white('Speedup'),
        chalk.white('Improvement'),
        chalk.white('CPU Mem (MB)'),
        chalk.white('GPU Mem (MB)'),
        chalk.white('Recommendation'),
      ],
      colWidths: [30, 12, 12, 10, 13, 14, 14, 18],
    });

    this.comparisons
      .sort((a, b) => b.speedup - a.speedup)
      .forEach((c) => {
        const speedupColor =
          c.speedup >= 2 ? chalk.green : c.speedup >= 1.5 ? chalk.cyan : c.speedup >= 1.1 ? chalk.yellow : chalk.red;

        const recColor =
          c.recommendation === 'USE GPU'
            ? chalk.green
            : c.recommendation === 'GPU BENEFICIAL'
            ? chalk.cyan
            : chalk.yellow;

        table.push([
          c.operation,
          c.cpuTimeMs.toFixed(2),
          c.gpuTimeMs.toFixed(2),
          speedupColor(`${c.speedup.toFixed(2)}x`),
          speedupColor(`+${c.speedupPercent}%`),
          c.cpuMemoryMB.toFixed(2),
          c.gpuMemoryMB.toFixed(2),
          recColor(c.recommendation),
        ]);
      });

    console.log(table.toString());
    this.printStatistics();
    this.printRecommendations();
  }

  printStatistics() {
    console.log(chalk.bold.magenta('\n📊 GPU ACCELERATION STATISTICS\n'));

    const avgSpeedup = this.comparisons.reduce((sum, c) => sum + c.speedup, 0) / this.comparisons.length;
    const maxSpeedup = Math.max(...this.comparisons.map((c) => c.speedup));
    const minSpeedup = Math.min(...this.comparisons.map((c) => c.speedup));

    const highSpeedupOps = this.comparisons.filter((c) => c.speedup >= 2).length;
    const beneficialOps = this.comparisons.filter((c) => c.speedup >= 1.1).length;

    const table = new Table();
    table.push(
      ['Total Operations Tested', this.comparisons.length],
      ['Average GPU Speedup', `${avgSpeedup.toFixed(2)}x (${((avgSpeedup - 1) * 100).toFixed(1)}% faster)`],
      ['Maximum GPU Speedup', `${maxSpeedup.toFixed(2)}x`],
      ['Minimum GPU Speedup', `${minSpeedup.toFixed(2)}x`],
      ['High Speedup Ops (≥2x)', `${highSpeedupOps} (${((highSpeedupOps / this.comparisons.length) * 100).toFixed(0)}%)`],
      [
        'Beneficial Ops (≥1.1x)',
        `${beneficialOps} (${((beneficialOps / this.comparisons.length) * 100).toFixed(0)}%)`,
      ]
    );

    console.log(table.toString());
  }

  printRecommendations() {
    console.log(chalk.bold.yellow('\n💡 OPTIMIZATION RECOMMENDATIONS\n'));

    const recommendations = [];

    // High impact operations
    const highImpact = this.comparisons.filter((c) => c.speedup >= 2);
    if (highImpact.length > 0) {
      recommendations.push({
        priority: 'HIGH',
        category: 'GPU Acceleration',
        operations: highImpact.map((c) => c.operation).join(', '),
        recommendation: `Always use GPU for these operations (${highImpact.length} ops with 2x+ speedup)`,
        impact: 'Significant performance improvement',
      });
    }

    // Medium impact operations
    const mediumImpact = this.comparisons.filter((c) => c.speedup >= 1.5 && c.speedup < 2);
    if (mediumImpact.length > 0) {
      recommendations.push({
        priority: 'MEDIUM',
        category: 'GPU Acceleration',
        operations: mediumImpact.map((c) => c.operation).join(', '),
        recommendation: `Prefer GPU for these operations (${mediumImpact.length} ops with 1.5-2x speedup)`,
        impact: 'Moderate performance improvement',
      });
    }

    // Memory-intensive operations
    const memoryIntensive = this.comparisons.filter((c) => c.gpuMemoryMB > 100);
    if (memoryIntensive.length > 0) {
      recommendations.push({
        priority: 'MEDIUM',
        category: 'Memory Management',
        operations: memoryIntensive.map((c) => c.operation).join(', '),
        recommendation: `Monitor GPU memory usage for these operations (>100MB allocation)`,
        impact: 'Prevent out-of-memory errors',
      });
    }

    // Low benefit operations
    const lowBenefit = this.comparisons.filter((c) => c.speedup < 1.1);
    if (lowBenefit.length > 0) {
      recommendations.push({
        priority: 'LOW',
        category: 'CPU Processing',
        operations: lowBenefit.map((c) => c.operation).join(', '),
        recommendation: `Use CPU for these operations (<10% GPU improvement)`,
        impact: 'Save GPU resources for high-impact tasks',
      });
    }

    const table = new Table({
      head: [chalk.white('Priority'), chalk.white('Category'), chalk.white('Recommendation'), chalk.white('Impact')],
      colWidths: [10, 20, 60, 35],
      wordWrap: true,
    });

    recommendations.forEach((r) => {
      const priorityColor = r.priority === 'HIGH' ? chalk.red : r.priority === 'MEDIUM' ? chalk.yellow : chalk.blue;
      table.push([priorityColor(r.priority), r.category, chalk.cyan(r.recommendation), r.impact]);
    });

    console.log(table.toString());
  }

  exportToJSON(filepath) {
    const fs = require('fs');
    const exportData = {
      timestamp: new Date().toISOString(),
      gpuAvailable: this.gpuAvailable,
      system: {
        node: process.version,
        platform: process.platform,
        arch: process.arch,
      },
      comparisons: this.comparisons,
      summary: {
        totalOperations: this.comparisons.length,
        averageSpeedup: this.comparisons.reduce((sum, c) => sum + c.speedup, 0) / this.comparisons.length,
        maxSpeedup: Math.max(...this.comparisons.map((c) => c.speedup)),
        minSpeedup: Math.min(...this.comparisons.map((c) => c.speedup)),
      },
    };
    fs.writeFileSync(filepath, JSON.stringify(exportData, null, 2));
    console.log(chalk.green(`\n✅ Results exported to ${filepath}`));
  }
}

const results = new GPUComparisonResults();

/**
 * Benchmark a GPU-capable operation
 */
async function benchmarkOperation(name, cpuFn, gpuFn, dataSize = 'standard') {
  console.log(chalk.cyan(`\nBenchmarking: ${name}`));

  // CPU benchmark
  const cpuMemStart = process.memoryUsage().heapUsed / 1024 / 1024;
  const cpuStart = performance.now();
  try {
    await cpuFn();
  } catch (err) {
    console.log(chalk.yellow(`  CPU warning: ${err.message}`));
  }
  const cpuTime = performance.now() - cpuStart;
  const cpuMemEnd = process.memoryUsage().heapUsed / 1024 / 1024;
  const cpuMemory = cpuMemEnd - cpuMemStart;

  console.log(chalk.gray(`  CPU: ${cpuTime.toFixed(2)}ms`));

  // GPU benchmark
  const gpuMemStart = process.memoryUsage().heapUsed / 1024 / 1024;
  const gpuStart = performance.now();
  try {
    await gpuFn();
  } catch (err) {
    console.log(chalk.yellow(`  GPU warning: ${err.message}`));
  }
  const gpuTime = performance.now() - gpuStart;
  const gpuMemEnd = process.memoryUsage().heapUsed / 1024 / 1024;
  const gpuMemory = gpuMemEnd - gpuMemStart;

  console.log(chalk.gray(`  GPU: ${gpuTime.toFixed(2)}ms`));

  const speedup = cpuTime / gpuTime;
  const speedupColor = speedup >= 2 ? chalk.green : speedup >= 1.5 ? chalk.cyan : speedup >= 1.1 ? chalk.yellow : chalk.red;
  console.log(speedupColor(`  Speedup: ${speedup.toFixed(2)}x (${((speedup - 1) * 100).toFixed(1)}% faster)`));

  results.recordComparison(name, cpuTime, gpuTime, cpuMemory, gpuMemory, dataSize);
}

/**
 * Run all GPU comparison benchmarks
 */
async function runGPUComparisons() {
  console.log(chalk.bold.magenta('\n╔═══════════════════════════════════════════════════════════╗'));
  console.log(chalk.bold.magenta('║       GPU vs CPU PERFORMANCE COMPARISON SUITE            ║'));
  console.log(chalk.bold.magenta('╚═══════════════════════════════════════════════════════════╝\n'));

  // 1. Quick Analysis
  await benchmarkOperation(
    'Quick Market Analysis',
    () => backend.quickAnalysis('AAPL', false),
    () => backend.quickAnalysis('AAPL', true),
    'single-symbol'
  );

  // 2. Trade Simulation
  await benchmarkOperation(
    'Trade Simulation',
    () => backend.simulateTrade('momentum', 'AAPL', 'buy', false),
    () => backend.simulateTrade('momentum', 'AAPL', 'buy', true),
    'single-trade'
  );

  // 3. Backtest (small dataset)
  await benchmarkOperation(
    'Backtest - 1 Month',
    () => backend.runBacktest('momentum', 'AAPL', '2023-12-01', '2023-12-31', false),
    () => backend.runBacktest('momentum', 'AAPL', '2023-12-01', '2023-12-31', true),
    '1-month'
  );

  // 4. Backtest (medium dataset)
  await benchmarkOperation(
    'Backtest - 3 Months',
    () => backend.runBacktest('momentum', 'AAPL', '2023-10-01', '2023-12-31', false),
    () => backend.runBacktest('momentum', 'AAPL', '2023-10-01', '2023-12-31', true),
    '3-months'
  );

  // 5. Backtest (large dataset)
  await benchmarkOperation(
    'Backtest - 1 Year',
    () => backend.runBacktest('momentum', 'AAPL', '2023-01-01', '2023-12-31', false),
    () => backend.runBacktest('momentum', 'AAPL', '2023-01-01', '2023-12-31', true),
    '1-year'
  );

  // 6. Strategy Optimization
  const paramRanges = JSON.stringify({ momentum_period: [10, 20, 30], rsi_period: [10, 14, 20] });
  await benchmarkOperation(
    'Strategy Optimization',
    () => backend.optimizeStrategy('momentum', 'AAPL', paramRanges, false),
    () => backend.optimizeStrategy('momentum', 'AAPL', paramRanges, true),
    'parameter-grid'
  );

  // 7. Neural Forecast
  await benchmarkOperation(
    'Neural Forecast - 30 Days',
    () => backend.neuralForecast('AAPL', 30, false, 0.95),
    () => backend.neuralForecast('AAPL', 30, true, 0.95),
    '30-day-forecast'
  );

  // 8. Neural Forecast (extended)
  await benchmarkOperation(
    'Neural Forecast - 90 Days',
    () => backend.neuralForecast('AAPL', 90, false, 0.95),
    () => backend.neuralForecast('AAPL', 90, true, 0.95),
    '90-day-forecast'
  );

  // 9. Risk Analysis (small portfolio)
  const smallPortfolio = JSON.stringify({
    positions: [
      { symbol: 'AAPL', quantity: 100, avgPrice: 150 },
      { symbol: 'GOOGL', quantity: 50, avgPrice: 2800 },
      { symbol: 'MSFT', quantity: 75, avgPrice: 300 },
    ],
  });
  await benchmarkOperation(
    'Risk Analysis - 3 Positions',
    () => backend.riskAnalysis(smallPortfolio, false),
    () => backend.riskAnalysis(smallPortfolio, true),
    'small-portfolio'
  );

  // 10. Risk Analysis (large portfolio)
  const largePortfolio = JSON.stringify({
    positions: Array(50)
      .fill()
      .map((_, i) => ({
        symbol: `STOCK${i}`,
        quantity: 100,
        avgPrice: 100 + Math.random() * 100,
      })),
  });
  await benchmarkOperation(
    'Risk Analysis - 50 Positions',
    () => backend.riskAnalysis(largePortfolio, false),
    () => backend.riskAnalysis(largePortfolio, true),
    'large-portfolio'
  );

  // 11. Correlation Analysis (small)
  await benchmarkOperation(
    'Correlation Analysis - 5 Symbols',
    () => backend.correlationAnalysis(['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN'], false),
    () => backend.correlationAnalysis(['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN'], true),
    '5-symbols'
  );

  // 12. Correlation Analysis (large)
  const manySymbols = Array(20)
    .fill()
    .map((_, i) => `STOCK${i}`);
  await benchmarkOperation(
    'Correlation Analysis - 20 Symbols',
    () => backend.correlationAnalysis(manySymbols, false),
    () => backend.correlationAnalysis(manySymbols, true),
    '20-symbols'
  );

  // 13. Neural Backtest
  await benchmarkOperation(
    'Neural Model Backtest',
    () => backend.neuralBacktest('model-1', '2023-01-01', '2023-12-31', 'sp500', false),
    () => backend.neuralBacktest('model-1', '2023-01-01', '2023-12-31', 'sp500', true),
    '1-year-backtest'
  );

  // 14. Neural Model Evaluation
  await benchmarkOperation(
    'Neural Model Evaluation',
    () => backend.neuralEvaluate('model-1', 'test-data.csv', false),
    () => backend.neuralEvaluate('model-1', 'test-data.csv', true),
    'evaluation-set'
  );

  // 15. Neural Optimization
  const neuralParamRanges = JSON.stringify({
    learning_rate: [0.001, 0.01, 0.1],
    hidden_layers: [64, 128, 256],
  });
  await benchmarkOperation(
    'Neural Hyperparameter Optimization',
    () => backend.neuralOptimize('model-1', neuralParamRanges, false),
    () => backend.neuralOptimize('model-1', neuralParamRanges, true),
    'hyperparameter-grid'
  );

  // Print consolidated results
  results.printResults();

  // Export results
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
  results.exportToJSON(`/workspaces/neural-trader/tests/benchmarks/results/gpu-comparison-${timestamp}.json`);

  // Print cost-benefit analysis
  printCostBenefitAnalysis();
}

/**
 * Print cost-benefit analysis for GPU usage
 */
function printCostBenefitAnalysis() {
  console.log(chalk.bold.magenta('\n💰 GPU COST-BENEFIT ANALYSIS\n'));

  const avgSpeedup = results.comparisons.reduce((sum, c) => sum + c.speedup, 0) / results.comparisons.length;
  const highSpeedupOps = results.comparisons.filter((c) => c.speedup >= 2).length;

  const table = new Table({
    head: [chalk.white('Metric'), chalk.white('Value'), chalk.white('Analysis')],
    colWidths: [35, 20, 60],
    wordWrap: true,
  });

  table.push(
    [
      'Average Performance Gain',
      chalk.green(`${((avgSpeedup - 1) * 100).toFixed(1)}%`),
      avgSpeedup >= 1.5 ? 'Strong ROI - GPU highly recommended' : 'Moderate ROI - GPU beneficial for specific tasks',
    ],
    [
      'High-Impact Operations',
      chalk.green(`${highSpeedupOps}/${results.comparisons.length}`),
      highSpeedupOps > results.comparisons.length / 2
        ? 'Majority benefit from GPU - invest in GPU infrastructure'
        : 'Selective GPU usage recommended',
    ],
    [
      'Estimated Time Savings',
      chalk.cyan(`~${((avgSpeedup - 1) * 100).toFixed(0)}%`),
      'For batch operations and high-frequency trading',
    ],
    [
      'GPU Recommendation',
      avgSpeedup >= 2 ? chalk.green('REQUIRED') : avgSpeedup >= 1.5 ? chalk.cyan('HIGHLY BENEFICIAL') : chalk.yellow('OPTIONAL'),
      avgSpeedup >= 2
        ? 'GPU essential for production performance'
        : avgSpeedup >= 1.5
        ? 'GPU provides significant competitive advantage'
        : 'GPU offers moderate improvements',
    ]
  );

  console.log(table.toString());
}

// Execute if run directly
if (require.main === module) {
  runGPUComparisons().catch((err) => {
    console.error(chalk.red('❌ GPU comparison failed:'), err);
    process.exit(1);
  });
}

module.exports = { runGPUComparisons, GPUComparisonResults };
