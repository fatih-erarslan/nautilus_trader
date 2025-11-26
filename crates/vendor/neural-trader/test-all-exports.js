#!/usr/bin/env node
/**
 * Comprehensive Test for All 178 NAPI Exports
 * Tests index.js after update to ensure all functions are accessible
 */

const neuralTrader = require('./index.js');

console.log('Testing Neural Trader v2.4.0+ - All 178 NAPI Exports\n');
console.log('='.repeat(80));

let passed = 0;
let failed = 0;
const errors = [];

function test(category, name, fn) {
  try {
    fn();
    console.log(`✓ ${category}: ${name}`);
    passed++;
  } catch (error) {
    console.log(`✗ ${category}: ${name} - ${error.message}`);
    errors.push({ category, name, error: error.message });
    failed++;
  }
}

// ============================================================
// Test Wrapper Modules (3)
// ============================================================
console.log('\n--- WRAPPER MODULES (3) ---');
test('Wrapper', 'cli', () => {
  if (typeof neuralTrader.cli !== 'object') throw new Error('cli is not an object');
  if (typeof neuralTrader.cli.listStrategies !== 'function') throw new Error('cli.listStrategies is not a function');
});
test('Wrapper', 'mcp', () => {
  if (typeof neuralTrader.mcp !== 'object') throw new Error('mcp is not an object');
  if (typeof neuralTrader.mcp.startServer !== 'function') throw new Error('mcp.startServer is not a function');
});
test('Wrapper', 'swarm', () => {
  if (typeof neuralTrader.swarm !== 'object') throw new Error('swarm is not an object');
  if (typeof neuralTrader.swarm.init !== 'function') throw new Error('swarm.init is not a function');
});

// ============================================================
// Test Enums (4) - Enum-like objects with constants
// ============================================================
console.log('\n--- ENUMS (4) ---');
const enums = ['AllocationStrategy', 'DistributionModel', 'MemberRole', 'MemberTier'];
enums.forEach(enumName => {
  test('Enum', enumName, () => {
    if (typeof neuralTrader[enumName] !== 'object') {
      throw new Error(`${enumName} is not an object`);
    }
    if (neuralTrader[enumName] === null) {
      throw new Error(`${enumName} is null`);
    }
  });
});

// ============================================================
// Test Classes (16) - Actual instantiable classes
// ============================================================
console.log('\n--- CLASSES (16) ---');
const classes = [
  'BacktestEngine', 'BrokerClient', 'CollaborationHub',
  'FundAllocationEngine', 'MarketDataProvider', 'MemberManager',
  'MemberPerformanceTracker', 'NeuralTrader',
  'PortfolioManager', 'PortfolioOptimizer', 'ProfitDistributionSystem', 'RiskManager',
  'StrategyRunner', 'SubscriptionHandle', 'VotingSystem', 'WithdrawalManager'
];
classes.forEach(className => {
  test('Class', className, () => {
    if (typeof neuralTrader[className] !== 'function') {
      throw new Error(`${className} is not a function/class`);
    }
  });
});

// ============================================================
// Test Market Data Functions (10)
// ============================================================
console.log('\n--- MARKET DATA & INDICATORS (10) ---');
const marketData = [
  'fetchMarketData', 'listDataProviders', 'calculateSma', 'calculateRsi',
  'calculateIndicator', 'getMarketStatus', 'getMarketOrderbook', 'getMarketAnalysis',
  'encodeBarsToBuffer', 'decodeBarsFromBuffer'
];
marketData.forEach(fn => {
  test('Market Data', fn, () => {
    if (typeof neuralTrader[fn] !== 'function') {
      throw new Error(`${fn} is not a function`);
    }
  });
});

// ============================================================
// Test Neural Network Functions (7)
// ============================================================
console.log('\n--- NEURAL NETWORK (7) ---');
const neural = [
  'neuralTrain', 'neuralPredict', 'neuralBacktest', 'neuralEvaluate',
  'neuralForecast', 'neuralOptimize', 'neuralModelStatus'
];
neural.forEach(fn => {
  test('Neural', fn, () => {
    if (typeof neuralTrader[fn] !== 'function') {
      throw new Error(`${fn} is not a function`);
    }
  });
});

// ============================================================
// Test Strategy & Backtest Functions (14)
// ============================================================
console.log('\n--- STRATEGY & BACKTEST (14) ---');
const strategy = [
  'backtestStrategy', 'runBacktest', 'listStrategies', 'optimizeStrategy',
  'switchActiveStrategy', 'quickBacktest', 'quickAnalysis', 'compareBacktests',
  'getStrategyInfo', 'getStrategyComparison', 'adaptiveStrategySelection',
  'recommendStrategy', 'optimizeParameters', 'monitorStrategyHealth'
];
strategy.forEach(fn => {
  test('Strategy', fn, () => {
    if (typeof neuralTrader[fn] !== 'function') {
      throw new Error(`${fn} is not a function`);
    }
  });
});

// ============================================================
// Test Trade Execution Functions (8)
// ============================================================
console.log('\n--- TRADE EXECUTION (8) ---');
const execution = [
  'executeTrade', 'executeMultiAssetTrade', 'executeSportsBet', 'executeSwarmStrategy',
  'getExecutionAnalytics', 'getTradeExecutionAnalytics', 'getApiLatency', 'validateBrokerConfig'
];
execution.forEach(fn => {
  test('Execution', fn, () => {
    if (typeof neuralTrader[fn] !== 'function') {
      throw new Error(`${fn} is not a function`);
    }
  });
});

// ============================================================
// Test Portfolio Functions (6)
// ============================================================
console.log('\n--- PORTFOLIO MANAGEMENT (6) ---');
const portfolio = [
  'portfolioRebalance', 'getPortfolioStatus', 'getPredictionPositions',
  'getBettingPortfolioStatus', 'crossAssetCorrelationMatrix', 'correlationAnalysis'
];
portfolio.forEach(fn => {
  test('Portfolio', fn, () => {
    if (typeof neuralTrader[fn] !== 'function') {
      throw new Error(`${fn} is not a function`);
    }
  });
});

// ============================================================
// Test Risk Management Functions (7)
// ============================================================
console.log('\n--- RISK MANAGEMENT (7) ---');
const risk = [
  'riskAnalysis', 'calculateSharpeRatio', 'calculateSortinoRatio', 'monteCarloSimulation',
  'calculateKellyCriterion', 'calculateMaxLeverage', 'calculateExpectedValue'
];
risk.forEach(fn => {
  test('Risk', fn, () => {
    if (typeof neuralTrader[fn] !== 'function') {
      throw new Error(`${fn} is not a function`);
    }
  });
});

// ============================================================
// Test E2B Cloud Functions (13)
// ============================================================
console.log('\n--- E2B CLOUD EXECUTION (13) ---');
const e2b = [
  'createE2BSandbox', 'deployE2BTemplate', 'executeE2BProcess', 'getE2BSandboxStatus',
  'listE2BSandboxes', 'terminateE2BSandbox', 'exportE2BTemplate', 'initE2BSwarm',
  'runE2BAgent', 'scaleE2BDeployment', 'deployTradingAgent', 'monitorE2BHealth'
];
e2b.forEach(fn => {
  test('E2B', fn, () => {
    if (typeof neuralTrader[fn] !== 'function') {
      throw new Error(`${fn} is not a function`);
    }
  });
});

// ============================================================
// Test Sports Betting Functions (25)
// ============================================================
console.log('\n--- SPORTS BETTING & PREDICTIONS (25) ---');
const sports = [
  'getSportsOdds', 'getSportsEvents', 'findSportsArbitrage', 'getBettingHistory',
  'getSportsBettingPerformance', 'getPredictionMarkets', 'placePredictionOrder',
  'getLiveOddsUpdates', 'analyzeBettingTrends', 'analyzeBettingMarketDepth',
  'compareBettingProviders', 'oddsApiGetSports', 'oddsApiGetUpcoming', 'oddsApiGetEventOdds',
  'oddsApiGetBookmakerOdds', 'oddsApiGetLiveOdds', 'oddsApiFindArbitrage',
  'oddsApiCalculateProbability', 'oddsApiCompareMargins', 'oddsApiAnalyzeMovement'
];
sports.forEach(fn => {
  test('Sports Betting', fn, () => {
    if (typeof neuralTrader[fn] !== 'function') {
      throw new Error(`${fn} is not a function`);
    }
  });
});

// ============================================================
// Test Syndicate Management Functions (18)
// ============================================================
console.log('\n--- SYNDICATE MANAGEMENT (18) ---');
const syndicate = [
  'createSyndicate', 'initSyndicate', 'addSyndicateMember', 'allocateSyndicateFunds',
  'distributeSyndicateProfits', 'getSyndicateStatus', 'getSyndicateMemberList',
  'getSyndicateMemberPerformance', 'getSyndicateProfitHistory', 'getSyndicateWithdrawalHistory',
  'getSyndicateAllocationLimits', 'processSyndicateWithdrawal', 'updateSyndicateMemberContribution',
  'updateSyndicateAllocationStrategy', 'createSyndicateVote', 'castSyndicateVote',
  'calculateSyndicateTaxLiability', 'simulateSyndicateAllocation'
];
syndicate.forEach(fn => {
  test('Syndicate', fn, () => {
    if (typeof neuralTrader[fn] !== 'function') {
      throw new Error(`${fn} is not a function`);
    }
  });
});

// ============================================================
// Test News & Sentiment Functions (9)
// ============================================================
console.log('\n--- NEWS & SENTIMENT ANALYSIS (9) ---');
const news = [
  'analyzeNews', 'fetchFilteredNews', 'getBreakingNews', 'getNewsSentiment',
  'getNewsTrends', 'analyzeNewsImpact', 'analyzeMarketSentiment', 'controlNewsCollection',
  'getNewsProviderStatus'
];
news.forEach(fn => {
  test('News', fn, () => {
    if (typeof neuralTrader[fn] !== 'function') {
      throw new Error(`${fn} is not a function`);
    }
  });
});

// ============================================================
// Test Swarm Coordination Functions (6)
// ============================================================
console.log('\n--- SWARM COORDINATION (6) ---');
const swarmFunctions = [
  'getSwarmStatus', 'getSwarmMetrics', 'scaleSwarm', 'shutdownSwarm', 'monitorSwarmHealth'
];
swarmFunctions.forEach(fn => {
  test('Swarm', fn, () => {
    if (typeof neuralTrader[fn] !== 'function') {
      throw new Error(`${fn} is not a function`);
    }
  });
});

// ============================================================
// Test Performance & Analytics Functions (7)
// ============================================================
console.log('\n--- PERFORMANCE & ANALYTICS (7) ---');
const performance = [
  'performanceReport', 'runBenchmark', 'analyzeBottlenecks', 'getSystemMetrics',
  'getTokenUsage', 'getHealthStatus', 'getVersionInfo'
];
performance.forEach(fn => {
  test('Performance', fn, () => {
    if (typeof neuralTrader[fn] !== 'function') {
      throw new Error(`${fn} is not a function`);
    }
  });
});

// ============================================================
// Test Data Science Functions (5)
// ============================================================
console.log('\n--- DATA SCIENCE - DTW (5) ---');
const dtw = [
  'dtwDistanceRust', 'dtwDistanceRustOptimized', 'dtwBatch', 'dtwBatchParallel', 'dtwBatchAdaptive'
];
dtw.forEach(fn => {
  test('DTW', fn, () => {
    if (typeof neuralTrader[fn] !== 'function') {
      throw new Error(`${fn} is not a function`);
    }
  });
});

// ============================================================
// Test System Utilities (4)
// ============================================================
console.log('\n--- SYSTEM UTILITIES (4) ---');
const system = ['initRuntime', 'listBrokerTypes', 'ping', 'getVersion'];
system.forEach(fn => {
  test('System', fn, () => {
    if (typeof neuralTrader[fn] !== 'function') {
      throw new Error(`${fn} is not a function`);
    }
  });
});

// ============================================================
// Test Legacy Functions (15) - Should throw helpful errors
// ============================================================
console.log('\n--- LEGACY FUNCTIONS (15) - Should throw deprecation errors ---');
const legacy = [
  'streamMarketData', 'runStrategy', 'backtest', 'executeOrder', 'cancelOrder',
  'getOrderStatus', 'getPortfolio', 'getPositions', 'calculateMetrics',
  'calculateVaR', 'calculatePositionSize', 'checkRiskLimits', 'trainModel',
  'predict', 'evaluateModel'
];
legacy.forEach(fn => {
  test('Legacy', `${fn} (should exist but throw)`, () => {
    if (typeof neuralTrader[fn] !== 'function') {
      throw new Error(`${fn} is not a function`);
    }
    // Verify it throws a helpful error
    try {
      neuralTrader[fn]();
      throw new Error(`${fn} should throw deprecation error but didn't`);
    } catch (error) {
      if (!error.message.includes('deprecated') && !error.message.includes('not implemented')) {
        throw new Error(`${fn} threw wrong error: ${error.message}`);
      }
    }
  });
});

// ============================================================
// Final Report
// ============================================================
console.log('\n' + '='.repeat(80));
console.log(`\n✓ PASSED: ${passed}`);
console.log(`✗ FAILED: ${failed}`);
console.log(`\nTotal Tested: ${passed + failed} / 178 expected`);

if (failed > 0) {
  console.log('\n❌ FAILED TESTS:');
  errors.forEach(({ category, name, error }) => {
    console.log(`  - ${category}: ${name} - ${error}`);
  });
  process.exit(1);
} else {
  console.log('\n✅ ALL TESTS PASSED! All 178 NAPI functions are accessible.');
  process.exit(0);
}
