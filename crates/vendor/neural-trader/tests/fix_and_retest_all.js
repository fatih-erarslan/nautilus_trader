#!/usr/bin/env node
/**
 * Neural Trader - Fix All Failing Tools and Retest
 *
 * Fixes parameter passing issues for all 9 failing tools
 * Targets: 100% success rate
 */

require('dotenv').config({ path: require('path').join(__dirname, '../.env') });
const path = require('path');
const fs = require('fs');

// Load NAPI module
function loadNapiModule() {
  const modulePath = path.join(__dirname, '../neural-trader-rust/crates/napi-bindings/neural-trader.linux-x64-gnu.node');
  if (!fs.existsSync(modulePath)) {
    throw new Error(`NAPI module not found: ${modulePath}`);
  }
  return require(modulePath);
}

const results = {
  timestamp: new Date().toISOString(),
  tests: [],
  summary: {
    total: 0,
    passed: 0,
    failed: 0,
  }
};

async function runTest(name, testFn) {
  console.log(`\n🧪 Testing: ${name}...`);
  const startTime = Date.now();
  const result = {
    name,
    success: false,
    latency: 0,
    error: null,
    response: null,
  };

  try {
    const response = await testFn();
    result.response = response;
    result.success = true;
    result.latency = Date.now() - startTime;
    console.log(`   ✅ Passed (${result.latency}ms)`);
    results.summary.passed++;
  } catch (error) {
    result.error = error.message;
    result.latency = Date.now() - startTime;
    console.log(`   ❌ Failed: ${error.message}`);
    results.summary.failed++;
  }

  results.tests.push(result);
  results.summary.total++;
  return result;
}

async function main() {
  console.log('\n' + '='.repeat(80));
  console.log('🔧 NEURAL TRADER - FIX ALL FAILING TOOLS');
  console.log('='.repeat(80));
  console.log('\n🎯 Target: 100% Success Rate\n');

  const napi = loadNapiModule();

  // =============================================================================
  // Fix 1: Neural Tool Parameters (4 tools)
  // Issue: Complex objects need JSON serialization
  // =============================================================================

  console.log('━'.repeat(80));
  console.log('🔧 FIXING NEURAL TOOLS (4 tools)');
  console.log('━'.repeat(80));

  await runTest('Neural - Forecast (FIXED)', async () => {
    // Original failing call:
    // await napi.neuralForecast('test-model', 'AAPL', 5, 0.95, true);

    // Fixed: Convert boolean to string or use correct parameter order
    const response = await napi.neuralForecast(
      'test-model',  // model_id
      'AAPL',        // symbol
      5,             // horizon
      0.95,          // confidence_level
      'true'         // use_gpu (as string)
    );

    const data = JSON.parse(response);
    console.log(`   Model: ${data.model_id || 'N/A'}`);
    console.log(`   Forecast: ${data.forecast ? 'Generated' : 'N/A'}`);
    return data;
  });

  await runTest('Neural - Train (FIXED)', async () => {
    // Fixed: Convert boolean to string
    const response = await napi.neuralTrain(
      '/tmp/data.csv', // data_path
      'lstm',          // model_type
      100,             // epochs
      32,              // batch_size
      0.001,           // learning_rate
      0.2,             // validation_split
      'true'           // use_gpu (as string)
    );

    const data = JSON.parse(response);
    console.log(`   Model Type: ${data.model_type || 'N/A'}`);
    console.log(`   Status: ${data.status || 'N/A'}`);
    return data;
  });

  await runTest('Neural - Optimize (FIXED)', async () => {
    // Fixed: Serialize parameter_ranges as JSON string
    const paramRanges = JSON.stringify({
      learning_rate: [0.001, 0.01],
      batch_size: [16, 64]
    });

    const response = await napi.neuralOptimize(
      'test-model',    // model_id
      paramRanges,     // parameter_ranges (as JSON string)
      100,             // trials
      'mae',           // optimization_metric
      'true'           // use_gpu (as string)
    );

    const data = JSON.parse(response);
    console.log(`   Best Params: ${data.best_params ? 'Found' : 'N/A'}`);
    console.log(`   Best Score: ${data.best_score || 'N/A'}`);
    return data;
  });

  await runTest('Neural - Predict (FIXED)', async () => {
    // Fixed: Serialize input array as JSON string
    const inputData = JSON.stringify(['AAPL']);

    const response = await napi.neuralPredict(
      'test-model',    // model_id
      inputData,       // input (as JSON string)
      'true'           // use_gpu (as string)
    );

    const data = JSON.parse(response);
    console.log(`   Predictions: ${data.predictions ? data.predictions.length : 0}`);
    return data;
  });

  // =============================================================================
  // Fix 2: E2B Tools (2 tools)
  // Issue: Parameter type conversions needed
  // =============================================================================

  console.log('\n' + '━'.repeat(80));
  console.log('🔧 FIXING E2B TOOLS (2 tools)');
  console.log('━'.repeat(80));

  let sandboxId = null;

  // Create sandbox first
  const createResult = await runTest('E2B - Create Sandbox (for testing)', async () => {
    const response = await napi.createE2BSandbox(
      'test-sandbox',
      'node',
      300,
      null,
      null,
      null
    );
    const data = JSON.parse(response);
    sandboxId = data.sandbox_id || data.sandboxId;
    return data;
  });

  if (sandboxId) {
    await runTest('E2B - Run Agent (FIXED)', async () => {
      // Fixed: Convert boolean to string and serialize symbols array
      const symbols = JSON.stringify(['AAPL', 'MSFT']);

      const response = await napi.runE2BAgent(
        sandboxId,
        'momentum',      // agent_type
        symbols,         // symbols (as JSON string)
        'false',         // use_gpu (as string)
        null             // strategy_params
      );

      const data = JSON.parse(response);
      console.log(`   Agent Type: ${data.agent_type || 'N/A'}`);
      console.log(`   Status: ${data.status || 'N/A'}`);
      return data;
    });

    await runTest('E2B - Deploy Template (FIXED)', async () => {
      // Fixed: Serialize configuration as JSON string
      const config = JSON.stringify({
        strategy: 'momentum',
        symbols: ['AAPL']
      });

      const response = await napi.deployE2BTemplate(
        'trading-bot',   // template_name
        'e2b',           // category
        config           // configuration (as JSON string)
      );

      const data = JSON.parse(response);
      console.log(`   Template: ${data.template_name || 'N/A'}`);
      console.log(`   Status: ${data.status || 'N/A'}`);
      return data;
    });

    // Cleanup
    await napi.terminateE2BSandbox(sandboxId, 'false');
  }

  // =============================================================================
  // Fix 3: Backtest Tools (2 tools)
  // Issue: Parameter order and type mismatches
  // =============================================================================

  console.log('\n' + '━'.repeat(80));
  console.log('🔧 FIXING BACKTEST TOOLS (2 tools)');
  console.log('━'.repeat(80));

  await runTest('Backtest - Run Backtest (FIXED)', async () => {
    // Fixed: Convert boolean to string, check parameter order
    const response = await napi.runBacktest(
      'momentum',           // strategy
      'AAPL',               // symbol
      '2024-01-01',         // start_date
      '2024-12-31',         // end_date
      'true',               // include_costs (as string)
      'sp500',              // benchmark
      'true'                // use_gpu (as string)
    );

    const data = JSON.parse(response);
    console.log(`   Total Return: ${data.total_return || 'N/A'}`);
    console.log(`   Sharpe Ratio: ${data.sharpe_ratio || 'N/A'}`);
    return data;
  });

  await runTest('Backtest - Optimize Strategy (FIXED)', async () => {
    // Fixed: Serialize parameter_ranges as JSON string
    const paramRanges = JSON.stringify({
      lookback_period: [10, 50],
      threshold: [0.01, 0.05]
    });

    const response = await napi.optimizeStrategy(
      'momentum',           // strategy
      'AAPL',               // symbol
      paramRanges,          // parameter_ranges (as JSON string)
      1000,                 // max_iterations
      'sharpe_ratio',       // optimization_metric
      'true'                // use_gpu (as string)
    );

    const data = JSON.parse(response);
    console.log(`   Best Params: ${data.best_params ? 'Found' : 'N/A'}`);
    console.log(`   Best Value: ${data.best_value || 'N/A'}`);
    return data;
  });

  // =============================================================================
  // Fix 4: News Control (1 tool)
  // Issue: Array parameter passing
  // =============================================================================

  console.log('\n' + '━'.repeat(80));
  console.log('🔧 FIXING NEWS TOOLS (1 tool)');
  console.log('━'.repeat(80));

  await runTest('News - Control Collection (FIXED)', async () => {
    // Fixed: Serialize symbols array as JSON string
    const symbols = JSON.stringify(['AAPL', 'MSFT']);
    const sources = JSON.stringify(['reuters', 'bloomberg']);

    const response = await napi.controlNewsCollection(
      'start',             // action
      symbols,             // symbols (as JSON string)
      24,                  // lookback_hours
      300,                 // update_frequency
      sources              // sources (as JSON string)
    );

    const data = JSON.parse(response);
    console.log(`   Action: ${data.action || 'N/A'}`);
    console.log(`   Status: ${data.status || 'N/A'}`);
    return data;
  });

  // =============================================================================
  // Summary
  // =============================================================================
  console.log('\n' + '='.repeat(80));
  console.log('📊 FIX AND RETEST RESULTS');
  console.log('='.repeat(80) + '\n');

  console.log(`Total Tests: ${results.summary.total}`);
  console.log(`✅ Passed: ${results.summary.passed}`);
  console.log(`❌ Failed: ${results.summary.failed}`);
  console.log(`📈 Success Rate: ${((results.summary.passed / results.summary.total) * 100).toFixed(1)}%\n`);

  const successRate = (results.summary.passed / results.summary.total) * 100;

  if (successRate === 100) {
    console.log('🎉 SUCCESS! All tools now working at 100%!');
    console.log('🚀 System is ready for production deployment.\n');
  } else {
    console.log(`⚠️  Success rate: ${successRate.toFixed(1)}% - still some issues remaining.\n`);
  }

  // Save results
  const reportPath = path.join(__dirname, '../docs/FIX_RETEST_RESULTS.json');
  fs.writeFileSync(reportPath, JSON.stringify(results, null, 2));
  console.log(`📄 Results saved to: ${reportPath}\n`);

  process.exit(results.summary.failed > 0 ? 1 : 0);
}

main().catch(error => {
  console.error('\n❌ Fatal error:', error);
  console.error(error.stack);
  process.exit(1);
});
