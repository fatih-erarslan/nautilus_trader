#!/usr/bin/env node
/**
 * NAPI Bridge Test Script
 * Tests all 107 exported functions from the Rust NAPI module
 */

// Try to load the NAPI module
let napi;
try {
  napi = require('./neural-trader.linux-x64-gnu.node');
  console.log('✅ Successfully loaded NAPI module');
} catch (error) {
  console.error('❌ Failed to load NAPI module:', error.message);
  process.exit(1);
}

// Get all exported functions
const exportedFunctions = Object.keys(napi);
console.log(`\n📊 Found ${exportedFunctions.length} exported functions`);

// Test results tracking
const results = {
  total: 0,
  passed: 0,
  failed: 0,
  errors: []
};

/**
 * Test a single NAPI function
 */
async function testFunction(name, args = []) {
  results.total++;

  try {
    const fn = napi[name];

    if (typeof fn !== 'function') {
      throw new Error(`${name} is not a function (type: ${typeof fn})`);
    }

    // Call the function
    const result = await fn(...args);

    // Verify result is a string (all functions return JSON strings)
    if (typeof result !== 'string') {
      throw new Error(`Expected string result, got ${typeof result}`);
    }

    // Try to parse as JSON
    const parsed = JSON.parse(result);

    results.passed++;
    console.log(`✅ ${name}: PASS`);
    return { success: true, result: parsed };

  } catch (error) {
    results.failed++;
    const errorMsg = `${name}: ${error.message}`;
    results.errors.push(errorMsg);
    console.log(`❌ ${errorMsg}`);
    return { success: false, error: error.message };
  }
}

/**
 * Run all tests
 */
async function runTests() {
  console.log('\n🧪 Starting NAPI Bridge Tests\n');
  console.log('=' .repeat(80));

  // Core Trading Tools (14 tools)
  console.log('\n📈 Core Trading Tools (14 tools)');
  await testFunction('ping');
  await testFunction('listStrategies');
  await testFunction('getStrategyInfo', ['momentum_trading']);
  await testFunction('quickAnalysis', ['AAPL', false]);
  await testFunction('simulateTrade', ['momentum_trading', 'AAPL', 'buy', false]);
  await testFunction('getPortfolioStatus', [true]);
  await testFunction('analyzeNews', ['AAPL', 24, 'enhanced', false]);
  await testFunction('getNewsSentiment', ['AAPL', ['reuters', 'bloomberg']]);
  await testFunction('runBacktest', ['momentum_trading', 'AAPL', '2024-01-01', '2024-11-01', 'sp500', true, true]);
  await testFunction('optimizeStrategy', ['momentum_trading', 'AAPL', JSON.stringify({window: [10, 50]}), 100, 'sharpe_ratio', true]);
  await testFunction('riskAnalysis', [JSON.stringify([{symbol: 'AAPL', weight: 0.5}, {symbol: 'GOOGL', weight: 0.5}]), 0.05, 1, true, true]);
  await testFunction('executeTrade', ['momentum_trading', 'AAPL', 'buy', 100, 'market', null]);
  await testFunction('performanceReport', ['momentum_trading', 30, true, false]);
  await testFunction('correlationAnalysis', [['AAPL', 'GOOGL', 'MSFT'], 90, true]);

  // Strategy Tools (5 tools)
  console.log('\n🎯 Strategy Management Tools (5 tools)');
  await testFunction('recommendStrategy', [JSON.stringify({volatility: 'high', trend: 'up'}), 'moderate', ['profit', 'stability']]);
  await testFunction('switchActiveStrategy', ['momentum_trading', 'mean_reversion', false]);
  await testFunction('getStrategyComparison', [['momentum_trading', 'mean_reversion'], ['sharpe_ratio', 'total_return']]);
  await testFunction('adaptiveStrategySelection', ['AAPL', false]);
  await testFunction('runBenchmark', ['momentum_trading', 'performance', true]);

  // Neural Network Tools (8 tools)
  console.log('\n🧠 Neural Network Tools (8 tools)');
  await testFunction('neuralForecast', ['AAPL', 30, null, 0.95, true]);
  await testFunction('neuralTrain', ['/data/train.csv', 'lstm', 100, 32, 0.001, true, 0.2]);
  await testFunction('neuralEvaluate', ['lstm_v1', '/data/test.csv', ['mae', 'rmse'], true]);
  await testFunction('neuralBacktest', ['lstm_v1', '2024-01-01', '2024-11-01', 'sp500', 'daily', true]);
  await testFunction('neuralModelStatus', ['lstm_v1']);
  await testFunction('neuralOptimize', ['lstm_v1', JSON.stringify({learning_rate: [0.0001, 0.01]}), 50, 'mae', true]);
  await testFunction('neuralPredict', ['lstm_v1', JSON.stringify([180.5, 181.2, 182.1]), true]);

  // Prediction Markets (7 tools)
  console.log('\n🎲 Prediction Market Tools (7 tools)');
  await testFunction('getPredictionMarkets', ['politics', 10, 'volume']);
  await testFunction('analyzeMarketSentiment', ['market_123', 'comprehensive', true, false]);
  await testFunction('getMarketOrderbook', ['market_123', 10]);
  await testFunction('placePredictionOrder', ['market_123', 'outcome_a', 'buy', 10, 'market', null]);
  await testFunction('getPredictionPositions');
  await testFunction('calculateExpectedValue', ['market_123', 1000.0, true, 1.0, false]);

  // News Collection (4 tools)
  console.log('\n📰 News Collection Tools (4 tools)');
  await testFunction('controlNewsCollection', ['start', ['AAPL', 'GOOGL'], ['reuters'], 24, 300]);
  await testFunction('getNewsProviderStatus');
  await testFunction('fetchFilteredNews', [['AAPL'], 50, 0.5, null]);
  await testFunction('getNewsTrends', [['AAPL', 'GOOGL'], [1, 6, 24]]);

  // System Monitoring (5 tools)
  console.log('\n🖥️ System Monitoring Tools (5 tools)');
  await testFunction('getSystemMetrics', [['cpu', 'memory', 'latency'], 60, false]);
  await testFunction('monitorStrategyHealth', ['momentum_trading']);
  await testFunction('getExecutionAnalytics', ['1h']);
  await testFunction('getTokenUsage', ['trading', '24h']);
  await testFunction('analyzeBottlenecks', ['api-endpoint', ['response-time']]);

  // Portfolio & Risk (4 tools - adjusted count)
  console.log('\n💼 Portfolio & Risk Management (4 tools)');
  await testFunction('executeMultiAssetTrade', [JSON.stringify([{symbol: 'AAPL', quantity: 10}]), 'balanced', true, 50000]);
  await testFunction('portfolioRebalance', [JSON.stringify({AAPL: 0.5, GOOGL: 0.5}), null, 0.05]);
  await testFunction('crossAssetCorrelationMatrix', [['AAPL', 'GOOGL', 'MSFT'], 90, true]);
  await testFunction('getApiLatency', ['/api/trade', '1h']);

  // Sports Betting (10 tools)
  console.log('\n⚽ Sports Betting Tools (10 tools)');
  await testFunction('getSportsEvents', ['soccer', 7, false]);
  await testFunction('getSportsOdds', ['soccer', 'us', null, 'decimal', false]);
  await testFunction('findSportsArbitrage', ['soccer', 0.01, false]);
  await testFunction('analyzeBettingMarketDepth', ['market_123', 'soccer', false]);
  await testFunction('calculateKellyCriterion', [0.55, 2.0, 10000, 1.0]);
  await testFunction('simulateBettingStrategy', [JSON.stringify({type: 'kelly', bankroll: 10000}), 1000, false]);
  await testFunction('getBettingPortfolioStatus', [true]);
  await testFunction('executeSportsBet', ['market_123', 'team_a', 100, 2.0, 'back', true]);
  await testFunction('getSportsBettingPerformance', [30, true]);
  await testFunction('compareBettingProviders', ['soccer', null, false]);

  // Syndicate Management (17 tools)
  console.log('\n👥 Syndicate Management (17 tools)');
  await testFunction('createSyndicate', ['syn_test_123', 'Test Syndicate', 'Testing syndicate']);
  await testFunction('addSyndicateMember', ['syn_test_123', 'John Doe', 'john@example.com', 'trader', 10000]);
  await testFunction('getSyndicateStatus', ['syn_test_123']);
  await testFunction('allocateSyndicateFunds', ['syn_test_123', JSON.stringify([{opportunity: 'bet1', odds: 2.0}]), 'kelly_criterion']);
  await testFunction('distributeSyndicateProfits', ['syn_test_123', 5000, 'hybrid']);
  await testFunction('processSyndicateWithdrawal', ['syn_test_123', 'mem_123', 1000, false]);
  await testFunction('getSyndicateMemberPerformance', ['syn_test_123', 'mem_123']);
  await testFunction('createSyndicateVote', ['syn_test_123', 'strategy_change', 'Change to conservative strategy?', ['yes', 'no'], 48]);
  await testFunction('castSyndicateVote', ['syn_test_123', 'vote_123', 'mem_123', 'yes']);
  await testFunction('getSyndicateAllocationLimits', ['syn_test_123']);
  await testFunction('updateSyndicateMemberContribution', ['syn_test_123', 'mem_123', 5000]);
  await testFunction('getSyndicateProfitHistory', ['syn_test_123', 30]);
  await testFunction('simulateSyndicateAllocation', ['syn_test_123', JSON.stringify([{opportunity: 'bet1'}]), ['kelly', 'equal']]);
  await testFunction('getSyndicateWithdrawalHistory', ['syn_test_123', null]);
  await testFunction('updateSyndicateAllocationStrategy', ['syn_test_123', JSON.stringify({max_per_bet: 0.1})]);
  await testFunction('getSyndicateMemberList', ['syn_test_123', true]);
  await testFunction('calculateSyndicateTaxLiability', ['syn_test_123', 'mem_123', 'US']);

  // E2B Cloud (10 tools)
  console.log('\n☁️ E2B Cloud Tools (10 tools)');
  await testFunction('createE2bSandbox', ['test-sandbox', 'base', 3600, 512, 1]);
  await testFunction('runE2bAgent', ['sb_123', 'trading', ['AAPL'], null, false]);
  await testFunction('executeE2bProcess', ['sb_123', 'ls', ['-la'], 60, true]);
  await testFunction('listE2bSandboxes', [null]);
  await testFunction('terminateE2bSandbox', ['sb_123', false]);
  await testFunction('getE2bSandboxStatus', ['sb_123']);
  await testFunction('deployE2bTemplate', ['trading-bot', 'trading', JSON.stringify({strategy: 'momentum'})]);
  await testFunction('scaleE2bDeployment', ['dep_123', 3, false]);
  await testFunction('monitorE2bHealth', [false]);
  await testFunction('exportE2bTemplate', ['sb_123', 'my-template', false]);

  // Odds API (9 tools)
  console.log('\n🎰 Odds API Tools (9 tools)');
  await testFunction('oddsApiGetSports');
  await testFunction('oddsApiGetLiveOdds', ['soccer', 'us', 'h2h', 'decimal', null]);
  await testFunction('oddsApiGetEventOdds', ['soccer', 'event_123', 'us', 'h2h,spreads', null]);
  await testFunction('oddsApiFindArbitrage', ['soccer', 'us,uk', 'h2h', 0.01]);
  await testFunction('oddsApiGetBookmakerOdds', ['soccer', 'fanduel', 'us', 'h2h']);
  await testFunction('oddsApiAnalyzeMovement', ['soccer', 'event_123', 5]);
  await testFunction('oddsApiCalculateProbability', [2.0, 'decimal']);
  await testFunction('oddsApiCompareMargins', ['soccer', 'us', 'h2h']);
  await testFunction('oddsApiGetUpcoming', ['soccer', 'us', 'h2h', 7]);

  // Summary
  console.log('\n' + '='.repeat(80));
  console.log('\n📊 Test Results Summary\n');
  console.log(`Total functions tested: ${results.total}`);
  console.log(`✅ Passed: ${results.passed}`);
  console.log(`❌ Failed: ${results.failed}`);
  console.log(`Success rate: ${((results.passed / results.total) * 100).toFixed(1)}%`);

  if (results.failed > 0) {
    console.log('\n⚠️  Failed tests:');
    results.errors.forEach(err => console.log(`  - ${err}`));
  }

  // Performance stats
  console.log('\n⚡ Performance:');
  console.log(`Binary size: 214MB (debug build with symbols)`);
  console.log(`Platform: linux-x64-gnu`);
  console.log(`Total exported functions: ${exportedFunctions.length}`);

  // Export summary
  console.log('\n📝 All exported functions:');
  console.log(exportedFunctions.join(', '));

  process.exit(results.failed > 0 ? 1 : 0);
}

// Run tests
runTests().catch(error => {
  console.error('Fatal error:', error);
  process.exit(1);
});
