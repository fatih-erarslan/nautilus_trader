#!/usr/bin/env node

/**
 * Comprehensive Alpaca API Benchmark for @neural-trader/backend
 * Tests: API connectivity, trading, neural forecasting, portfolio management, performance
 */

const os = require('os');
const fs = require('fs');
const path = require('path');

// Load environment variables from .env file
try {
  const dotenv = require('dotenv');
  const envPath = path.join(__dirname, '..', '.env');
  if (fs.existsSync(envPath)) {
    dotenv.config({ path: envPath });
    console.log('✓ Loaded environment variables from .env file\n');
  }
} catch (err) {
  // dotenv not installed, that's okay - will use system environment variables
}

// Performance tracking
const performanceMetrics = {
  apiCalls: [],
  trades: [],
  forecasts: [],
  errors: [],
  startTime: Date.now()
};

function logMetric(category, operation, duration, success = true, details = {}) {
  const metric = {
    timestamp: Date.now(),
    category,
    operation,
    duration,
    success,
    ...details
  };
  performanceMetrics[category].push(metric);
}

function formatDuration(ms) {
  if (ms < 1000) return `${ms.toFixed(2)}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
}

function printSection(title) {
  console.log(`\n${'='.repeat(70)}`);
  console.log(`  ${title}`);
  console.log('='.repeat(70));
}

function printSubSection(title) {
  console.log(`\n--- ${title} ---`);
}

async function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// Main benchmark execution
async function runBenchmark() {
  console.log('╔═══════════════════════════════════════════════════════════════════╗');
  console.log('║      Neural Trader Backend - Alpaca API Benchmark Suite          ║');
  console.log('╚═══════════════════════════════════════════════════════════════════╝');
  console.log(`\nPlatform: ${os.platform()}`);
  console.log(`Architecture: ${os.arch()}`);
  console.log(`Node.js: ${process.version}`);
  console.log(`CPUs: ${os.cpus().length}`);
  console.log(`Memory: ${(os.totalmem() / 1024 / 1024 / 1024).toFixed(2)} GB`);
  console.log(`Start Time: ${new Date().toISOString()}\n`);

  let backend;

  // ========================================================================
  // STEP 1: Module Loading & Initialization
  // ========================================================================
  printSection('STEP 1: Module Loading & Initialization');

  try {
    const loadStart = Date.now();
    backend = require('../index.js');
    const loadDuration = Date.now() - loadStart;
    console.log(`✓ Module loaded in ${formatDuration(loadDuration)}`);
    console.log(`✓ Exports: ${Object.keys(backend).length} functions available`);

    // Initialize the neural trader system
    const initStart = Date.now();
    if (backend.initNeuralTrader) {
      await backend.initNeuralTrader();
      const initDuration = Date.now() - initStart;
      console.log(`✓ System initialized in ${formatDuration(initDuration)}`);
    }
  } catch (err) {
    console.error('✗ Failed to load module:', err.message);
    process.exit(1);
  }

  // ========================================================================
  // STEP 2: Alpaca API Configuration
  // ========================================================================
  printSection('STEP 2: Alpaca API Configuration');

  // Check for Alpaca credentials in environment
  const alpacaKey = process.env.ALPACA_API_KEY || 'DEMO_API_KEY';
  const alpacaSecret = process.env.ALPACA_API_SECRET || 'DEMO_SECRET';
  const alpacaPaper = process.env.ALPACA_PAPER_TRADING !== 'false';

  console.log(`API Key: ${alpacaKey.substring(0, 8)}...`);
  console.log(`Mode: ${alpacaPaper ? 'Paper Trading' : 'Live Trading'}`);

  if (alpacaKey === 'DEMO_API_KEY') {
    console.log('\n⚠ Using DEMO credentials - set ALPACA_API_KEY and ALPACA_API_SECRET for real testing');
    console.log('⚠ Tests will run in simulation mode\n');
  }

  // ========================================================================
  // STEP 3: System Information & Health Check
  // ========================================================================
  printSection('STEP 3: System Information & Health Check');

  try {
    if (backend.getSystemInfo) {
      const infoStart = Date.now();
      const sysInfo = await backend.getSystemInfo();
      const infoDuration = Date.now() - infoStart;
      logMetric('apiCalls', 'getSystemInfo', infoDuration, true);
      console.log(`✓ System info retrieved in ${formatDuration(infoDuration)}`);
      console.log(`  Version: ${sysInfo.version || 'N/A'}`);
      console.log(`  Features: ${sysInfo.features ? sysInfo.features.join(', ') : 'N/A'}`);
    }

    if (backend.healthCheck) {
      const healthStart = Date.now();
      const health = await backend.healthCheck();
      const healthDuration = Date.now() - healthStart;
      logMetric('apiCalls', 'healthCheck', healthDuration, true);
      console.log(`✓ Health check passed in ${formatDuration(healthDuration)}`);
      console.log(`  Status: ${health.status || 'OK'}`);
    }
  } catch (err) {
    console.error('⚠ System checks failed:', err.message);
    performanceMetrics.errors.push({ step: 'health', error: err.message });
  }

  // ========================================================================
  // STEP 4: Strategy Listing & Analysis
  // ========================================================================
  printSection('STEP 4: Strategy Listing & Analysis');

  try {
    if (backend.listStrategies) {
      const listStart = Date.now();
      const strategies = await backend.listStrategies();
      const listDuration = Date.now() - listStart;
      logMetric('apiCalls', 'listStrategies', listDuration, true);

      console.log(`✓ Retrieved ${strategies.length} strategies in ${formatDuration(listDuration)}`);
      console.log(`  Available: ${strategies.join(', ')}`);

      // Get details for first few strategies
      for (let i = 0; i < Math.min(3, strategies.length); i++) {
        const strategy = strategies[i];
        if (backend.getStrategyInfo) {
          try {
            const detailStart = Date.now();
            const info = await backend.getStrategyInfo(strategy);
            const detailDuration = Date.now() - detailStart;
            logMetric('apiCalls', 'getStrategyInfo', detailDuration, true, { strategy });
            // Handle both string and object responses
            const description = typeof info === 'string' ? info : (info.description || 'No description');
            console.log(`  - ${strategy.name || strategy}: ${description}`);
          } catch (err) {
            console.log(`  - ${strategy.name || strategy}: ${err.message}`);
          }
        }
      }
    }
  } catch (err) {
    console.error('⚠ Strategy listing failed:', err.message);
    performanceMetrics.errors.push({ step: 'strategies', error: err.message });
  }

  // ========================================================================
  // STEP 5: Quick Market Analysis Benchmark
  // ========================================================================
  printSection('STEP 5: Quick Market Analysis Benchmark');

  const testSymbols = ['AAPL', 'TSLA', 'SPY'];

  if (backend.quickAnalysis) {
    for (const symbol of testSymbols) {
      try {
        const analysisStart = Date.now();
        const analysis = await backend.quickAnalysis(symbol, alpacaPaper);
        const analysisDuration = Date.now() - analysisStart;
        logMetric('apiCalls', 'quickAnalysis', analysisDuration, true, { symbol });

        console.log(`\n✓ ${symbol} analysis completed in ${formatDuration(analysisDuration)}`);
        if (analysis.price) console.log(`  Price: $${analysis.price}`);
        if (analysis.trend) console.log(`  Trend: ${analysis.trend}`);
        if (analysis.signal) console.log(`  Signal: ${analysis.signal}`);
      } catch (err) {
        console.error(`✗ ${symbol} analysis failed: ${err.message}`);
        performanceMetrics.errors.push({ step: 'analysis', symbol, error: err.message });
      }

      await sleep(100); // Rate limiting
    }
  } else {
    console.log('⚠ quickAnalysis not available');
  }

  // ========================================================================
  // STEP 6: Trading Simulation Benchmark
  // ========================================================================
  printSection('STEP 6: Trading Simulation Benchmark');

  if (backend.simulateTrade) {
    const tradeTests = [
      { strategy: 'momentum_trading', symbol: 'AAPL', action: 'buy' },
      { strategy: 'mean_reversion', symbol: 'TSLA', action: 'sell' },
      { strategy: 'trend_following', symbol: 'SPY', action: 'buy' }
    ];

    for (const test of tradeTests) {
      try {
        const tradeStart = Date.now();
        const result = await backend.simulateTrade(
          test.strategy,
          test.symbol,
          test.action,
          true // use GPU if available
        );
        const tradeDuration = Date.now() - tradeStart;
        logMetric('trades', 'simulateTrade', tradeDuration, true, test);

        console.log(`\n✓ Trade simulation: ${test.action} ${test.symbol} (${test.strategy})`);
        console.log(`  Duration: ${formatDuration(tradeDuration)}`);
        if (result.expectedReturn) console.log(`  Expected Return: ${result.expectedReturn}%`);
        if (result.confidence) console.log(`  Confidence: ${result.confidence}%`);
      } catch (err) {
        console.error(`✗ Trade simulation failed: ${err.message}`);
        performanceMetrics.errors.push({ step: 'trade', ...test, error: err.message });
      }

      await sleep(100);
    }
  } else {
    console.log('⚠ simulateTrade not available');
  }

  // ========================================================================
  // STEP 7: Neural Forecasting Benchmark
  // ========================================================================
  printSection('STEP 7: Neural Forecasting Benchmark');

  if (backend.neuralForecast) {
    const forecastSymbols = ['AAPL', 'SPY'];
    const horizons = [1, 5, 10]; // days ahead

    for (const symbol of forecastSymbols) {
      for (const horizon of horizons) {
        try {
          const forecastStart = Date.now();
          // Correct signature: neuralForecast(symbol, horizon, useGpu, confidenceLevel)
          const forecast = await backend.neuralForecast(
            symbol,
            horizon,
            true,  // use GPU
            0.95   // confidence level
          );
          const forecastDuration = Date.now() - forecastStart;
          logMetric('forecasts', 'neuralForecast', forecastDuration, true, { symbol, horizon });

          console.log(`\n✓ ${symbol} forecast (${horizon}d ahead): ${formatDuration(forecastDuration)}`);
          if (forecast.predicted_price) console.log(`  Predicted: $${forecast.predicted_price.toFixed(2)}`);
          if (forecast.confidence_interval) {
            console.log(`  Range: $${forecast.confidence_interval.lower.toFixed(2)} - $${forecast.confidence_interval.upper.toFixed(2)}`);
          }
        } catch (err) {
          console.error(`✗ Forecast failed (${symbol}, ${horizon}d): ${err.message}`);
          performanceMetrics.errors.push({ step: 'forecast', symbol, horizon, error: err.message });
        }

        await sleep(100);
      }
    }
  } else {
    console.log('⚠ neuralForecast not available');
  }

  // ========================================================================
  // STEP 8: Portfolio Management Benchmark
  // ========================================================================
  printSection('STEP 8: Portfolio Management Benchmark');

  if (backend.getPortfolioStatus) {
    try {
      const portfolioStart = Date.now();
      const portfolio = await backend.getPortfolioStatus(true); // include analytics
      const portfolioDuration = Date.now() - portfolioStart;
      logMetric('apiCalls', 'getPortfolioStatus', portfolioDuration, true);

      console.log(`✓ Portfolio retrieved in ${formatDuration(portfolioDuration)}`);
      if (portfolio.total_value) console.log(`  Total Value: $${portfolio.total_value.toFixed(2)}`);
      if (portfolio.positions) console.log(`  Positions: ${portfolio.positions.length}`);
      if (portfolio.cash) console.log(`  Cash: $${portfolio.cash.toFixed(2)}`);
    } catch (err) {
      console.error('✗ Portfolio retrieval failed:', err.message);
      performanceMetrics.errors.push({ step: 'portfolio', error: err.message });
    }
  }

  if (backend.riskAnalysis) {
    // Correct portfolio format with positions array and cash
    const testPortfolio = {
      positions: [
        { symbol: 'AAPL', quantity: 10, avg_entry_price: 150, current_price: 155, side: 'long' },
        { symbol: 'TSLA', quantity: 5, avg_entry_price: 200, current_price: 210, side: 'long' },
        { symbol: 'SPY', quantity: 20, avg_entry_price: 400, current_price: 405, side: 'long' }
      ],
      cash: 42000,
      returns: [],
      equity_curve: [],
      trade_pnls: []
    };

    try {
      const riskStart = Date.now();
      // Correct signature: riskAnalysis(portfolio: string, useGpu)
      const risk = await backend.riskAnalysis(
        JSON.stringify(testPortfolio),
        true  // use GPU
      );
      const riskDuration = Date.now() - riskStart;
      logMetric('apiCalls', 'riskAnalysis', riskDuration, true);

      console.log(`\n✓ Risk analysis completed in ${formatDuration(riskDuration)}`);
      if (risk.var) console.log(`  VaR (95%): $${risk.var.toFixed(2)}`);
      if (risk.cvar) console.log(`  CVaR (95%): $${risk.cvar.toFixed(2)}`);
      if (risk.sharpe_ratio) console.log(`  Sharpe Ratio: ${risk.sharpe_ratio.toFixed(2)}`);
    } catch (err) {
      console.error('✗ Risk analysis failed:', err.message);
      performanceMetrics.errors.push({ step: 'risk', error: err.message });
    }
  }

  // ========================================================================
  // STEP 9: Backtesting Benchmark
  // ========================================================================
  printSection('STEP 9: Backtesting Benchmark');

  if (backend.runBacktest) {
    const backtestConfigs = [
      { strategy: 'momentum_trading', symbol: 'AAPL', days: 30 },
      { strategy: 'mean_reversion', symbol: 'SPY', days: 60 }
    ];

    for (const config of backtestConfigs) {
      try {
        const endDate = new Date();
        const startDate = new Date(endDate);
        startDate.setDate(startDate.getDate() - config.days);

        const backtestStart = Date.now();
        // Correct signature: runBacktest(strategy, symbol, startDate, endDate, useGpu)
        const results = await backend.runBacktest(
          config.strategy,
          config.symbol,
          startDate.toISOString().split('T')[0],
          endDate.toISOString().split('T')[0],
          true     // use GPU
        );
        const backtestDuration = Date.now() - backtestStart;
        logMetric('apiCalls', 'runBacktest', backtestDuration, true, config);

        console.log(`\n✓ Backtest: ${config.strategy} on ${config.symbol} (${config.days}d)`);
        console.log(`  Duration: ${formatDuration(backtestDuration)}`);
        if (results.total_return) console.log(`  Return: ${results.total_return.toFixed(2)}%`);
        if (results.sharpe_ratio) console.log(`  Sharpe: ${results.sharpe_ratio.toFixed(2)}`);
        if (results.max_drawdown) console.log(`  Max Drawdown: ${results.max_drawdown.toFixed(2)}%`);
      } catch (err) {
        console.error(`✗ Backtest failed (${config.strategy}, ${config.symbol}): ${err.message}`);
        performanceMetrics.errors.push({ step: 'backtest', ...config, error: err.message });
      }

      await sleep(200);
    }
  } else {
    console.log('⚠ runBacktest not available');
  }

  // ========================================================================
  // STEP 10: Performance Summary & Report
  // ========================================================================
  printSection('STEP 10: Performance Summary & Report');

  const totalDuration = Date.now() - performanceMetrics.startTime;

  console.log(`\nTotal Duration: ${formatDuration(totalDuration)}`);
  console.log(`End Time: ${new Date().toISOString()}\n`);

  // Calculate statistics for each category
  function calculateStats(metrics) {
    if (metrics.length === 0) return null;

    const durations = metrics.map(m => m.duration);
    const sum = durations.reduce((a, b) => a + b, 0);
    const avg = sum / durations.length;
    const sorted = durations.sort((a, b) => a - b);
    const median = sorted[Math.floor(sorted.length / 2)];
    const min = sorted[0];
    const max = sorted[sorted.length - 1];
    const successful = metrics.filter(m => m.success).length;

    return { count: metrics.length, avg, median, min, max, successful };
  }

  printSubSection('API Calls Performance');
  const apiStats = calculateStats(performanceMetrics.apiCalls);
  if (apiStats) {
    console.log(`  Total Calls: ${apiStats.count}`);
    console.log(`  Successful: ${apiStats.successful}/${apiStats.count}`);
    console.log(`  Average: ${formatDuration(apiStats.avg)}`);
    console.log(`  Median: ${formatDuration(apiStats.median)}`);
    console.log(`  Min: ${formatDuration(apiStats.min)}`);
    console.log(`  Max: ${formatDuration(apiStats.max)}`);
  } else {
    console.log('  No API calls recorded');
  }

  printSubSection('Trade Simulations Performance');
  const tradeStats = calculateStats(performanceMetrics.trades);
  if (tradeStats) {
    console.log(`  Total Trades: ${tradeStats.count}`);
    console.log(`  Successful: ${tradeStats.successful}/${tradeStats.count}`);
    console.log(`  Average: ${formatDuration(tradeStats.avg)}`);
    console.log(`  Median: ${formatDuration(tradeStats.median)}`);
    console.log(`  Min: ${formatDuration(tradeStats.min)}`);
    console.log(`  Max: ${formatDuration(tradeStats.max)}`);
  } else {
    console.log('  No trades recorded');
  }

  printSubSection('Neural Forecasts Performance');
  const forecastStats = calculateStats(performanceMetrics.forecasts);
  if (forecastStats) {
    console.log(`  Total Forecasts: ${forecastStats.count}`);
    console.log(`  Successful: ${forecastStats.successful}/${forecastStats.count}`);
    console.log(`  Average: ${formatDuration(forecastStats.avg)}`);
    console.log(`  Median: ${formatDuration(forecastStats.median)}`);
    console.log(`  Min: ${formatDuration(forecastStats.min)}`);
    console.log(`  Max: ${formatDuration(forecastStats.max)}`);
  } else {
    console.log('  No forecasts recorded');
  }

  printSubSection('Errors Summary');
  console.log(`  Total Errors: ${performanceMetrics.errors.length}`);
  if (performanceMetrics.errors.length > 0) {
    console.log('  Error Details:');
    performanceMetrics.errors.forEach((err, idx) => {
      console.log(`    ${idx + 1}. [${err.step}] ${err.error}`);
    });
  }

  // Overall performance metrics
  printSubSection('Overall Performance Metrics');
  const totalOps = apiStats?.count || 0 + tradeStats?.count || 0 + forecastStats?.count || 0;
  const throughput = totalOps / (totalDuration / 1000);
  console.log(`  Total Operations: ${totalOps}`);
  console.log(`  Throughput: ${throughput.toFixed(2)} ops/sec`);
  console.log(`  Success Rate: ${((totalOps - performanceMetrics.errors.length) / totalOps * 100).toFixed(2)}%`);

  // Save detailed report
  printSubSection('Saving Detailed Report');
  const reportPath = path.join(__dirname, 'alpaca-benchmark-report.json');
  try {
    fs.writeFileSync(reportPath, JSON.stringify({
      metadata: {
        platform: os.platform(),
        arch: os.arch(),
        nodeVersion: process.version,
        cpus: os.cpus().length,
        totalMemory: os.totalmem(),
        timestamp: new Date().toISOString(),
        duration: totalDuration
      },
      metrics: performanceMetrics,
      summary: {
        apiCalls: apiStats,
        trades: tradeStats,
        forecasts: forecastStats,
        errors: performanceMetrics.errors,
        totalOperations: totalOps,
        throughput,
        successRate: (totalOps - performanceMetrics.errors.length) / totalOps
      }
    }, null, 2));
    console.log(`✓ Report saved to: ${reportPath}`);
  } catch (err) {
    console.error(`✗ Failed to save report: ${err.message}`);
  }

  // Final summary
  console.log('\n' + '='.repeat(70));
  console.log('                    BENCHMARK COMPLETE                           ');
  console.log('='.repeat(70));
  console.log(`\nStatus: ${performanceMetrics.errors.length === 0 ? '✓ ALL TESTS PASSED' : `⚠ ${performanceMetrics.errors.length} ERRORS`}`);
  console.log(`Duration: ${formatDuration(totalDuration)}`);
  console.log(`Operations: ${totalOps}`);
  console.log(`Throughput: ${throughput.toFixed(2)} ops/sec\n`);

  // Cleanup
  if (backend.shutdown) {
    await backend.shutdown();
    console.log('✓ System shutdown complete\n');
  }

  return performanceMetrics.errors.length === 0 ? 0 : 1;
}

// Run the benchmark
runBenchmark()
  .then(exitCode => process.exit(exitCode))
  .catch(err => {
    console.error('\n✗ Benchmark failed with error:', err);
    console.error(err.stack);
    process.exit(1);
  });
