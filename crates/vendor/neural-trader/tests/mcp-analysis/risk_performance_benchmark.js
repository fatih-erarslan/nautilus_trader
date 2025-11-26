#!/usr/bin/env node
/**
 * Comprehensive Risk & Performance MCP Tools Benchmarking Suite
 *
 * Tests 8 MCP tools:
 * 1. risk_analysis - VaR/CVaR calculations
 * 2. optimize_strategy - Parameter optimization
 * 3. portfolio_rebalance - Rebalancing calculations
 * 4. correlation_analysis - Asset correlations
 * 5. run_backtest - Historical testing
 * 6. get_system_metrics - Performance metrics
 * 7. monitor_strategy_health - Strategy monitoring
 * 8. get_execution_analytics - Execution analysis
 */

const { spawn } = require('child_process');
const fs = require('fs').promises;
const path = require('path');

// ANSI colors for terminal output
const colors = {
  reset: '\x1b[0m',
  bright: '\x1b[1m',
  green: '\x1b[32m',
  red: '\x1b[31m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  magenta: '\x1b[35m',
  cyan: '\x1b[36m',
};

class RiskPerformanceBenchmark {
  constructor() {
    this.results = {
      timestamp: new Date().toISOString(),
      environment: {
        node_version: process.version,
        platform: process.platform,
        arch: process.arch,
        cuda_available: process.env.CUDA_VISIBLE_DEVICES !== undefined,
        cuda_devices: process.env.CUDA_VISIBLE_DEVICES || 'none',
      },
      tests: [],
      summary: {},
    };

    this.testData = this.generateTestData();
  }

  log(message, color = 'reset') {
    console.log(`${colors[color]}${message}${colors.reset}`);
  }

  generateTestData() {
    return {
      // Portfolio data for risk analysis
      portfolio: [
        { symbol: 'AAPL', quantity: 100, value: 17500 },
        { symbol: 'GOOGL', quantity: 50, value: 7000 },
        { symbol: 'MSFT', quantity: 75, value: 25875 },
        { symbol: 'TSLA', quantity: 40, value: 10000 },
        { symbol: 'NVDA', quantity: 30, value: 15000 },
      ],

      // Symbols for correlation analysis
      symbols: ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA', 'AMZN', 'META', 'NFLX'],

      // Target allocations for rebalancing
      targetAllocations: {
        'AAPL': 0.25,
        'GOOGL': 0.20,
        'MSFT': 0.25,
        'TSLA': 0.15,
        'NVDA': 0.15,
      },

      // Current portfolio for rebalancing
      currentPortfolio: {
        'AAPL': 17500,
        'GOOGL': 7000,
        'MSFT': 25875,
        'TSLA': 10000,
        'NVDA': 15000,
      },

      // Strategy parameters for optimization
      parameterRanges: {
        'short_window': { min: 5, max: 20 },
        'long_window': { min: 20, max: 50 },
        'threshold': { min: 0.01, max: 0.05 },
      },
    };
  }

  async runMCPTool(toolName, params) {
    const startTime = Date.now();

    return new Promise((resolve, reject) => {
      const mcp = spawn('node', [
        path.join(__dirname, '../../neural-trader-rust/packages/mcp/bin/mcp-server.js'),
      ], {
        env: { ...process.env },
      });

      let stdout = '';
      let stderr = '';

      mcp.stdout.on('data', (data) => {
        stdout += data.toString();
      });

      mcp.stderr.on('data', (data) => {
        stderr += data.toString();
      });

      // Send JSONRPC request
      const request = {
        jsonrpc: '2.0',
        id: 1,
        method: 'tools/call',
        params: {
          name: toolName,
          arguments: params,
        },
      };

      mcp.stdin.write(JSON.stringify(request) + '\n');

      setTimeout(() => {
        const elapsed = Date.now() - startTime;
        mcp.kill();

        resolve({
          success: stdout.includes('result'),
          elapsed,
          stdout,
          stderr,
        });
      }, 30000); // 30 second timeout
    });
  }

  async testRiskAnalysis() {
    this.log('\n=== Testing risk_analysis (VaR/CVaR) ===', 'cyan');

    const tests = [
      {
        name: 'Monte Carlo VaR with GPU',
        params: {
          portfolio: JSON.stringify(this.testData.portfolio),
          use_gpu: true,
          use_monte_carlo: true,
          var_confidence: 0.05,
          time_horizon: 1,
        },
      },
      {
        name: 'Monte Carlo VaR without GPU',
        params: {
          portfolio: JSON.stringify(this.testData.portfolio),
          use_gpu: false,
          use_monte_carlo: true,
          var_confidence: 0.05,
          time_horizon: 1,
        },
      },
      {
        name: 'Parametric VaR',
        params: {
          portfolio: JSON.stringify(this.testData.portfolio),
          use_gpu: false,
          use_monte_carlo: false,
          var_confidence: 0.05,
          time_horizon: 1,
        },
      },
    ];

    const results = [];

    for (const test of tests) {
      this.log(`  Testing: ${test.name}`, 'yellow');

      const result = await this.runMCPTool('risk_analysis', test.params);

      this.log(`    ✓ Elapsed: ${result.elapsed}ms`, result.success ? 'green' : 'red');

      results.push({
        test: test.name,
        ...result,
      });
    }

    // Calculate GPU speedup
    const gpuTime = results.find(r => r.test.includes('with GPU'))?.elapsed || 0;
    const cpuTime = results.find(r => r.test.includes('without GPU'))?.elapsed || 0;
    const speedup = cpuTime > 0 ? (cpuTime / gpuTime).toFixed(2) : 'N/A';

    this.log(`\n  GPU Speedup: ${speedup}x`, 'magenta');

    return {
      tool: 'risk_analysis',
      tests: results,
      gpuSpeedup: speedup,
    };
  }

  async testCorrelationAnalysis() {
    this.log('\n=== Testing correlation_analysis ===', 'cyan');

    const tests = [
      {
        name: 'Small matrix (8 symbols) with GPU',
        params: {
          symbols: this.testData.symbols,
          period_days: 90,
          use_gpu: true,
        },
      },
      {
        name: 'Small matrix (8 symbols) without GPU',
        params: {
          symbols: this.testData.symbols,
          period_days: 90,
          use_gpu: false,
        },
      },
      {
        name: 'Large matrix (30 symbols) with GPU',
        params: {
          symbols: Array.from({ length: 30 }, (_, i) => `SYM${i}`),
          period_days: 90,
          use_gpu: true,
        },
      },
    ];

    const results = [];

    for (const test of tests) {
      this.log(`  Testing: ${test.name}`, 'yellow');

      const result = await this.runMCPTool('correlation_analysis', test.params);

      this.log(`    ✓ Elapsed: ${result.elapsed}ms`, result.success ? 'green' : 'red');

      results.push({
        test: test.name,
        ...result,
      });
    }

    return {
      tool: 'correlation_analysis',
      tests: results,
    };
  }

  async testPortfolioRebalance() {
    this.log('\n=== Testing portfolio_rebalance ===', 'cyan');

    const tests = [
      {
        name: 'Standard rebalancing',
        params: {
          target_allocations: JSON.stringify(this.testData.targetAllocations),
          current_portfolio: JSON.stringify(this.testData.currentPortfolio),
          rebalance_threshold: 0.05,
        },
      },
      {
        name: 'Aggressive rebalancing (low threshold)',
        params: {
          target_allocations: JSON.stringify(this.testData.targetAllocations),
          current_portfolio: JSON.stringify(this.testData.currentPortfolio),
          rebalance_threshold: 0.01,
        },
      },
    ];

    const results = [];

    for (const test of tests) {
      this.log(`  Testing: ${test.name}`, 'yellow');

      const result = await this.runMCPTool('portfolio_rebalance', test.params);

      this.log(`    ✓ Elapsed: ${result.elapsed}ms`, result.success ? 'green' : 'red');

      results.push({
        test: test.name,
        ...result,
      });
    }

    return {
      tool: 'portfolio_rebalance',
      tests: results,
    };
  }

  async testOptimizeStrategy() {
    this.log('\n=== Testing optimize_strategy ===', 'cyan');

    const tests = [
      {
        name: 'Quick optimization (100 iterations)',
        params: {
          strategy: 'neural_trend',
          symbol: 'AAPL',
          parameter_ranges: this.testData.parameterRanges,
          max_iterations: 100,
          optimization_metric: 'sharpe_ratio',
          use_gpu: true,
        },
      },
      {
        name: 'Full optimization (1000 iterations)',
        params: {
          strategy: 'neural_trend',
          symbol: 'AAPL',
          parameter_ranges: this.testData.parameterRanges,
          max_iterations: 1000,
          optimization_metric: 'sharpe_ratio',
          use_gpu: true,
        },
      },
    ];

    const results = [];

    for (const test of tests) {
      this.log(`  Testing: ${test.name}`, 'yellow');

      const result = await this.runMCPTool('optimize_strategy', test.params);

      this.log(`    ✓ Elapsed: ${result.elapsed}ms`, result.success ? 'green' : 'red');

      results.push({
        test: test.name,
        ...result,
      });
    }

    return {
      tool: 'optimize_strategy',
      tests: results,
    };
  }

  async testBacktest() {
    this.log('\n=== Testing run_backtest ===', 'cyan');

    const tests = [
      {
        name: 'Short backtest (30 days)',
        params: {
          strategy: 'neural_trend',
          symbol: 'AAPL',
          start_date: '2024-01-01',
          end_date: '2024-01-31',
          use_gpu: true,
        },
      },
      {
        name: 'Long backtest (1 year)',
        params: {
          strategy: 'neural_trend',
          symbol: 'AAPL',
          start_date: '2023-01-01',
          end_date: '2024-01-01',
          use_gpu: true,
        },
      },
    ];

    const results = [];

    for (const test of tests) {
      this.log(`  Testing: ${test.name}`, 'yellow');

      const result = await this.runMCPTool('run_backtest', test.params);

      this.log(`    ✓ Elapsed: ${result.elapsed}ms`, result.success ? 'green' : 'red');

      results.push({
        test: test.name,
        ...result,
      });
    }

    return {
      tool: 'run_backtest',
      tests: results,
    };
  }

  async testSystemMetrics() {
    this.log('\n=== Testing get_system_metrics ===', 'cyan');

    const result = await this.runMCPTool('get_system_metrics', {
      include_history: true,
      metrics: ['cpu', 'memory', 'latency', 'throughput'],
      time_range_minutes: 60,
    });

    this.log(`  ✓ Elapsed: ${result.elapsed}ms`, result.success ? 'green' : 'red');

    return {
      tool: 'get_system_metrics',
      tests: [result],
    };
  }

  async testStrategyHealth() {
    this.log('\n=== Testing monitor_strategy_health ===', 'cyan');

    const result = await this.runMCPTool('monitor_strategy_health', {
      strategy: 'neural_trend',
    });

    this.log(`  ✓ Elapsed: ${result.elapsed}ms`, result.success ? 'green' : 'red');

    return {
      tool: 'monitor_strategy_health',
      tests: [result],
    };
  }

  async testExecutionAnalytics() {
    this.log('\n=== Testing get_execution_analytics ===', 'cyan');

    const result = await this.runMCPTool('get_execution_analytics', {
      time_period: '1h',
    });

    this.log(`  ✓ Elapsed: ${result.elapsed}ms`, result.success ? 'green' : 'red');

    return {
      tool: 'get_execution_analytics',
      tests: [result],
    };
  }

  async runAllTests() {
    this.log('\n' + '='.repeat(80), 'bright');
    this.log('Risk & Performance MCP Tools Benchmark Suite', 'bright');
    this.log('='.repeat(80) + '\n', 'bright');

    const toolTests = [
      this.testRiskAnalysis.bind(this),
      this.testCorrelationAnalysis.bind(this),
      this.testPortfolioRebalance.bind(this),
      this.testOptimizeStrategy.bind(this),
      this.testBacktest.bind(this),
      this.testSystemMetrics.bind(this),
      this.testStrategyHealth.bind(this),
      this.testExecutionAnalytics.bind(this),
    ];

    for (const test of toolTests) {
      const result = await test();
      this.results.tests.push(result);
    }

    this.generateSummary();
    await this.saveResults();
  }

  generateSummary() {
    const totalTests = this.results.tests.reduce((sum, t) => sum + t.tests.length, 0);
    const passedTests = this.results.tests.reduce(
      (sum, t) => sum + t.tests.filter(test => test.success).length,
      0
    );

    const avgElapsed = this.results.tests.reduce(
      (sum, t) => sum + t.tests.reduce((s, test) => s + test.elapsed, 0),
      0
    ) / totalTests;

    this.results.summary = {
      total_tests: totalTests,
      passed: passedTests,
      failed: totalTests - passedTests,
      success_rate: ((passedTests / totalTests) * 100).toFixed(2) + '%',
      avg_execution_time_ms: avgElapsed.toFixed(2),
    };

    this.log('\n' + '='.repeat(80), 'bright');
    this.log('Summary', 'bright');
    this.log('='.repeat(80), 'bright');
    this.log(`Total Tests: ${totalTests}`, 'cyan');
    this.log(`Passed: ${passedTests}`, 'green');
    this.log(`Failed: ${totalTests - passedTests}`, 'red');
    this.log(`Success Rate: ${this.results.summary.success_rate}`, 'magenta');
    this.log(`Avg Execution Time: ${this.results.summary.avg_execution_time_ms}ms`, 'yellow');
  }

  async saveResults() {
    const outputPath = path.join(__dirname, '../../docs/mcp-analysis/benchmark_results.json');
    await fs.writeFile(outputPath, JSON.stringify(this.results, null, 2));
    this.log(`\nResults saved to: ${outputPath}`, 'green');
  }
}

// Run the benchmark
if (require.main === module) {
  const benchmark = new RiskPerformanceBenchmark();
  benchmark.runAllTests().catch(err => {
    console.error('Benchmark failed:', err);
    process.exit(1);
  });
}

module.exports = RiskPerformanceBenchmark;
