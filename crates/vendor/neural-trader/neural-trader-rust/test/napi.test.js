#!/usr/bin/env node

/**
 * NAPI Bindings Integration Test Suite
 *
 * Tests all NAPI bindings for the Neural Trader Rust core
 */

const assert = require('assert');
const path = require('path');

// Import the NAPI bindings
// Note: Adjust path based on build location
let bindings;
try {
  bindings = require('../crates/napi-bindings/index.js');
} catch (err) {
  console.error('Failed to load NAPI bindings. Have you run `npm run build`?');
  console.error(err.message);
  process.exit(1);
}

console.log('NAPI Bindings Test Suite\n');
console.log('='.repeat(60));

// Test results tracking
const results = {
  passed: 0,
  failed: 0,
  skipped: 0
};

/**
 * Test helper function
 */
async function test(name, fn) {
  try {
    await fn();
    console.log(`✓ ${name}`);
    results.passed++;
  } catch (error) {
    console.error(`✗ ${name}`);
    console.error(`  Error: ${error.message}`);
    results.failed++;
  }
}

/**
 * Test suite runner
 */
async function runTests() {
  console.log('\n## Core System Tests\n');

  await test('getVersionInfo returns version information', () => {
    const version = bindings.getVersionInfo();
    assert(version.rustCore, 'Should have rustCore version');
    assert(version.napiBindings, 'Should have napiBindings version');
    assert(version.rustCompiler, 'Should have rustCompiler version');
    console.log(`    Rust Core: ${version.rustCore}`);
    console.log(`    NAPI Bindings: ${version.napiBindings}`);
  });

  await test('initRuntime initializes without error', () => {
    bindings.initRuntime(4);
  });

  await test('NeuralTrader constructor works', () => {
    const trader = new bindings.NeuralTrader({
      apiKey: 'test-key',
      apiSecret: 'test-secret',
      paperTrading: true
    });
    assert(trader, 'Should create trader instance');
  });

  console.log('\n## Broker Tests\n');

  await test('listBrokerTypes returns available brokers', () => {
    const brokers = bindings.listBrokerTypes();
    assert(Array.isArray(brokers), 'Should return array');
    assert(brokers.includes('alpaca'), 'Should include alpaca');
    assert(brokers.includes('ibkr'), 'Should include ibkr');
    console.log(`    Available brokers: ${brokers.join(', ')}`);
  });

  await test('BrokerClient constructor validates config', () => {
    const client = new bindings.BrokerClient({
      brokerType: 'alpaca',
      apiKey: 'test-key',
      apiSecret: 'test-secret',
      paperTrading: true
    });
    assert(client, 'Should create broker client');
  });

  await test('BrokerClient can connect and disconnect', async () => {
    const client = new bindings.BrokerClient({
      brokerType: 'alpaca',
      apiKey: 'test-key',
      apiSecret: 'test-secret',
      paperTrading: true
    });

    const connected = await client.connect();
    assert(connected === true, 'Should connect successfully');

    await client.disconnect();
  });

  await test('validateBrokerConfig validates configuration', () => {
    const valid = bindings.validateBrokerConfig({
      brokerType: 'alpaca',
      apiKey: 'test-key',
      apiSecret: 'test-secret',
      paperTrading: true
    });
    assert(valid === true, 'Should validate correct config');
  });

  console.log('\n## Neural Network Tests\n');

  await test('listModelTypes returns available models', () => {
    const models = bindings.listModelTypes();
    assert(Array.isArray(models), 'Should return array');
    assert(models.includes('nhits'), 'Should include NHITS');
    console.log(`    Available models: ${models.join(', ')}`);
  });

  await test('NeuralModel constructor works', () => {
    const model = new bindings.NeuralModel({
      modelType: 'nhits',
      inputSize: 168,
      horizon: 24,
      hiddenSize: 512,
      numLayers: 3,
      dropout: 0.1,
      learningRate: 0.001
    });
    assert(model, 'Should create model instance');
  });

  await test('NeuralModel can get info', async () => {
    const model = new bindings.NeuralModel({
      modelType: 'nhits',
      inputSize: 168,
      horizon: 24,
      hiddenSize: 512,
      numLayers: 3,
      dropout: 0.1,
      learningRate: 0.001
    });

    const info = await model.getInfo();
    assert(info.model_type === 'nhits', 'Should have correct model type');
    assert(info.input_size === 168, 'Should have correct input size');
  });

  await test('BatchPredictor can be created', () => {
    const predictor = new bindings.BatchPredictor();
    assert(predictor, 'Should create batch predictor');
  });

  console.log('\n## Risk Management Tests\n');

  await test('RiskManager constructor works', () => {
    const riskMgr = new bindings.RiskManager({
      confidenceLevel: 0.95,
      lookbackPeriods: 252,
      method: 'historical'
    });
    assert(riskMgr, 'Should create risk manager');
  });

  await test('RiskManager calculates VaR', () => {
    const riskMgr = new bindings.RiskManager({
      confidenceLevel: 0.95,
      lookbackPeriods: 252,
      method: 'historical'
    });

    // Generate some random returns
    const returns = Array.from({ length: 252 }, () => (Math.random() - 0.5) * 0.02);

    const varResult = riskMgr.calculateVar(returns, 100000);
    assert(varResult.varAmount > 0, 'Should calculate VaR amount');
    assert(varResult.confidenceLevel === 0.95, 'Should have correct confidence level');
    console.log(`    VaR (95%): $${varResult.varAmount.toFixed(2)}`);
  });

  await test('RiskManager calculates Kelly criterion', () => {
    const riskMgr = new bindings.RiskManager({
      confidenceLevel: 0.95,
      lookbackPeriods: 252,
      method: 'historical'
    });

    const kelly = riskMgr.calculateKelly(0.55, 1000, 500);
    assert(kelly.kellyFraction >= 0, 'Should have non-negative Kelly fraction');
    assert(kelly.halfKelly === kelly.kellyFraction / 2, 'Half Kelly should be half of Kelly');
    console.log(`    Kelly fraction: ${kelly.kellyFraction.toFixed(4)}`);
  });

  await test('calculateSharpeRatio works', () => {
    const returns = Array.from({ length: 252 }, () => (Math.random() - 0.4) * 0.02);
    const sharpe = bindings.calculateSharpeRatio(returns, 0.02, 252);
    assert(typeof sharpe === 'number', 'Should return a number');
    console.log(`    Sharpe ratio: ${sharpe.toFixed(2)}`);
  });

  console.log('\n## Backtesting Tests\n');

  await test('BacktestEngine constructor works', () => {
    const engine = new bindings.BacktestEngine({
      initialCapital: 100000,
      startDate: '2023-01-01',
      endDate: '2023-12-31',
      commission: 0.001,
      slippage: 0.0005,
      useMarkToMarket: true
    });
    assert(engine, 'Should create backtest engine');
  });

  await test('BacktestEngine calculates metrics from equity curve', () => {
    const engine = new bindings.BacktestEngine({
      initialCapital: 100000,
      startDate: '2023-01-01',
      endDate: '2023-12-31',
      commission: 0.001,
      slippage: 0.0005,
      useMarkToMarket: true
    });

    const equityCurve = [100000, 101000, 102000, 101500, 103000, 104000];
    const metrics = engine.calculateMetrics(equityCurve);

    assert(metrics.totalReturn > 0, 'Should have positive return');
    assert(metrics.finalEquity === 104000, 'Should match final equity');
    console.log(`    Total return: ${(metrics.totalReturn * 100).toFixed(2)}%`);
  });

  console.log('\n## Market Data Tests\n');

  await test('listDataProviders returns available providers', () => {
    const providers = bindings.listDataProviders();
    assert(Array.isArray(providers), 'Should return array');
    assert(providers.includes('alpaca'), 'Should include alpaca');
    console.log(`    Available providers: ${providers.join(', ')}`);
  });

  await test('MarketDataProvider constructor works', () => {
    const provider = new bindings.MarketDataProvider({
      provider: 'alpaca',
      apiKey: 'test-key',
      apiSecret: 'test-secret',
      websocketEnabled: true
    });
    assert(provider, 'Should create market data provider');
  });

  await test('MarketDataProvider can connect', async () => {
    const provider = new bindings.MarketDataProvider({
      provider: 'alpaca',
      websocketEnabled: false
    });

    const connected = await provider.connect();
    assert(connected === true, 'Should connect successfully');

    await provider.disconnect();
  });

  await test('calculateSma works', () => {
    const prices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    const sma = bindings.calculateSma(prices, 3);

    assert(Array.isArray(sma), 'Should return array');
    assert(sma.length === prices.length, 'Should match input length');
    // Average of [3,4,5] should be 4
    assert(Math.abs(sma[4] - 4) < 0.01, 'Should calculate correct SMA');
  });

  await test('calculateRsi works', () => {
    const prices = [44, 44.34, 44.09, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08, 45.89, 46.03, 45.61, 46.28, 46.28, 46.00, 46.03, 46.41, 46.22, 45.64];
    const rsi = bindings.calculateRsi(prices, 14);

    assert(Array.isArray(rsi), 'Should return array');
    const lastRsi = rsi[rsi.length - 1];
    assert(lastRsi >= 0 && lastRsi <= 100, 'RSI should be between 0 and 100');
    console.log(`    Last RSI value: ${lastRsi.toFixed(2)}`);
  });

  console.log('\n## Strategy Tests\n');

  await test('StrategyRunner constructor works', () => {
    const runner = new bindings.StrategyRunner();
    assert(runner, 'Should create strategy runner');
  });

  await test('StrategyRunner can add momentum strategy', async () => {
    const runner = new bindings.StrategyRunner();
    const strategyId = await runner.addMomentumStrategy({
      name: 'Test Momentum',
      symbols: ['AAPL', 'GOOGL'],
      parameters: {}
    });
    assert(typeof strategyId === 'string', 'Should return strategy ID');
    assert(strategyId.startsWith('momentum-'), 'Should have correct prefix');
  });

  await test('StrategyRunner can list strategies', async () => {
    const runner = new bindings.StrategyRunner();
    await runner.addMomentumStrategy({
      name: 'Test1',
      symbols: ['AAPL'],
      parameters: {}
    });

    const strategies = await runner.listStrategies();
    assert(Array.isArray(strategies), 'Should return array');
    assert(strategies.length > 0, 'Should have strategies');
  });

  console.log('\n## Portfolio Tests\n');

  await test('PortfolioManager constructor works', () => {
    const portfolio = new bindings.PortfolioManager(100000);
    assert(portfolio, 'Should create portfolio manager');
  });

  await test('PortfolioManager can get cash balance', async () => {
    const portfolio = new bindings.PortfolioManager(100000);
    const cash = await portfolio.getCash();
    assert(cash === 100000, 'Should have initial cash');
  });

  await test('PortfolioOptimizer constructor works', () => {
    const optimizer = new bindings.PortfolioOptimizer({
      riskFreeRate: 0.02,
      maxPositionSize: 0.25,
      minPositionSize: 0.05
    });
    assert(optimizer, 'Should create portfolio optimizer');
  });

  // Print results
  console.log('\n' + '='.repeat(60));
  console.log('\nTest Results:\n');
  console.log(`✓ Passed: ${results.passed}`);
  console.log(`✗ Failed: ${results.failed}`);
  console.log(`○ Skipped: ${results.skipped}`);
  console.log(`\nTotal: ${results.passed + results.failed + results.skipped} tests`);

  if (results.failed > 0) {
    console.log('\n⚠️  Some tests failed!');
    process.exit(1);
  } else {
    console.log('\n✅ All tests passed!');
    process.exit(0);
  }
}

// Run the tests
runTests().catch(error => {
  console.error('\nUnexpected error running tests:');
  console.error(error);
  process.exit(1);
});
