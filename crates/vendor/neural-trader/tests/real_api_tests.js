#!/usr/bin/env node
/**
 * Neural Trader - Real API Integration Tests
 *
 * Tests critical tools with REAL API calls to verify:
 * - Alpaca trading API integration
 * - The Odds API sports betting integration
 * - News API integration
 * - E2B sandbox integration
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

// Test results
const results = {
  timestamp: new Date().toISOString(),
  tests: [],
  summary: {
    total: 0,
    passed: 0,
    failed: 0,
    realApiCalls: 0,
  }
};

// Helper to run test
async function runTest(name, testFn) {
  console.log(`\n🧪 Testing: ${name}...`);
  const startTime = Date.now();
  const result = {
    name,
    success: false,
    latency: 0,
    error: null,
    response: null,
    realApiCall: false,
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
  console.log('🚀 NEURAL TRADER - REAL API INTEGRATION TESTS');
  console.log('='.repeat(80) + '\n');

  const napi = loadNapiModule();
  console.log('✅ NAPI module loaded\n');

  // =============================================================================
  // Test 1: Alpaca Trading API - Quick Analysis with Real Market Data
  // =============================================================================
  await runTest('Alpaca - Quick Analysis (AAPL)', async () => {
    const response = await napi.quickAnalysis('AAPL', false);
    const data = JSON.parse(response);

    console.log(`   Symbol: ${data.symbol || 'N/A'}`);
    console.log(`   Status: ${data.status || 'N/A'}`);

    if (data.status === 'no_broker_configured') {
      console.log('   ℹ️  Broker not configured - expected for this test');
    } else if (data.current_price) {
      console.log(`   Price: $${data.current_price}`);
      results.summary.realApiCalls++;
    }

    return data;
  });

  // =============================================================================
  // Test 2: The Odds API - Get Available Sports
  // =============================================================================
  await runTest('The Odds API - Get Sports List', async () => {
    const response = await napi.oddsApiGetSports();
    const data = JSON.parse(response);

    console.log(`   Status: ${data.status || 'N/A'}`);

    if (data.sports && Array.isArray(data.sports)) {
      console.log(`   Sports Available: ${data.sports.length}`);
      results.summary.realApiCalls++;

      if (data.sports.length > 0) {
        console.log(`   Sample: ${data.sports.slice(0, 3).map(s => s.title || s.key).join(', ')}`);
      }
    }

    return data;
  });

  // =============================================================================
  // Test 3: The Odds API - Get Live NFL Odds
  // =============================================================================
  await runTest('The Odds API - Get NFL Odds', async () => {
    const response = await napi.oddsApiGetLiveOdds('americanfootball_nfl', 'us', 'h2h', 'decimal', null);
    const data = JSON.parse(response);

    console.log(`   Status: ${data.status || 'N/A'}`);

    if (data.events && Array.isArray(data.events)) {
      console.log(`   Events Found: ${data.events.length}`);
      results.summary.realApiCalls++;

      if (data.events.length > 0) {
        const event = data.events[0];
        console.log(`   Sample Event: ${event.home_team} vs ${event.away_team}`);
      }
    } else if (data.message) {
      console.log(`   Message: ${data.message}`);
    }

    return data;
  });

  // =============================================================================
  // Test 4: Kelly Criterion Calculation (Mathematical, no API)
  // =============================================================================
  await runTest('Kelly Criterion - Optimal Bet Sizing', async () => {
    // Example: 55% win probability, 2.0 odds, $10,000 bankroll
    const response = await napi.calculateKellyCriterion(0.55, 2.0, 10000, 1.0);
    const data = JSON.parse(response);

    console.log(`   Probability: ${data.probability || 'N/A'}`);
    console.log(`   Odds: ${data.odds || 'N/A'}`);
    console.log(`   Optimal Bet: $${data.optimal_bet || 'N/A'}`);
    console.log(`   Kelly Fraction: ${data.kelly_fraction || 'N/A'}`);

    return data;
  });

  // =============================================================================
  // Test 5: News Sentiment Analysis
  // =============================================================================
  await runTest('News - Sentiment Analysis (AAPL)', async () => {
    const response = await napi.getNewsSentiment('AAPL', null);
    const data = JSON.parse(response);

    console.log(`   Symbol: ${data.symbol || 'N/A'}`);
    console.log(`   Status: ${data.status || 'N/A'}`);

    if (data.sentiment) {
      console.log(`   Sentiment: ${data.sentiment.overall || 'N/A'}`);
      results.summary.realApiCalls++;
    }

    return data;
  });

  // =============================================================================
  // Test 6: Syndicate Creation
  // =============================================================================
  await runTest('Syndicate - Create New Syndicate', async () => {
    const syndicateId = 'test-syndicate-' + Date.now();
    const response = await napi.createSyndicate(syndicateId, 'Performance Test Syndicate', 'Created during API testing');
    const data = JSON.parse(response);

    console.log(`   Syndicate ID: ${data.syndicate_id || 'N/A'}`);
    console.log(`   Status: ${data.status || 'N/A'}`);
    console.log(`   Created: ${data.created_at || 'N/A'}`);

    return data;
  });

  // =============================================================================
  // Test 7: Prediction Market Listing
  // =============================================================================
  await runTest('Prediction Markets - List Available Markets', async () => {
    const response = await napi.getPredictionMarkets(null, 10, 'volume');
    const data = JSON.parse(response);

    console.log(`   Status: ${data.status || 'N/A'}`);

    if (data.markets && Array.isArray(data.markets)) {
      console.log(`   Markets Found: ${data.markets.length}`);

      if (data.markets.length > 0) {
        console.log(`   Sample Market: ${data.markets[0].title || data.markets[0].question}`);
      }
    }

    return data;
  });

  // =============================================================================
  // Test 8: System Metrics
  // =============================================================================
  await runTest('System - Get Performance Metrics', async () => {
    const response = await napi.getSystemMetrics(['cpu', 'memory'], 60, false);
    const data = JSON.parse(response);

    console.log(`   Metrics Type: ${data.metrics_type || 'N/A'}`);

    if (data.cpu) {
      console.log(`   CPU Usage: ${data.cpu.usage || 'N/A'}`);
    }
    if (data.memory) {
      console.log(`   Memory Usage: ${data.memory.used || 'N/A'}`);
    }

    return data;
  });

  // =============================================================================
  // Test 9: E2B Sandbox Listing
  // =============================================================================
  await runTest('E2B - List Sandboxes', async () => {
    const response = await napi.listE2bSandboxes(null);
    const data = JSON.parse(response);

    console.log(`   Status: ${data.status || 'N/A'}`);

    if (data.sandboxes && Array.isArray(data.sandboxes)) {
      console.log(`   Active Sandboxes: ${data.sandboxes.length}`);
    } else if (data.message) {
      console.log(`   Message: ${data.message}`);
    }

    return data;
  });

  // =============================================================================
  // Test 10: Sports Events Lookup
  // =============================================================================
  await runTest('Sports Betting - Get Upcoming Events', async () => {
    const response = await napi.getSportsEvents('americanfootball_nfl', 7, false);
    const data = JSON.parse(response);

    console.log(`   Sport: ${data.sport || 'N/A'}`);
    console.log(`   Status: ${data.status || 'N/A'}`);

    if (data.events && Array.isArray(data.events)) {
      console.log(`   Upcoming Events: ${data.events.length}`);
      results.summary.realApiCalls++;
    }

    return data;
  });

  // =============================================================================
  // Summary
  // =============================================================================
  console.log('\n' + '='.repeat(80));
  console.log('📊 TEST RESULTS SUMMARY');
  console.log('='.repeat(80) + '\n');

  console.log(`Total Tests: ${results.summary.total}`);
  console.log(`✅ Passed: ${results.summary.passed}`);
  console.log(`❌ Failed: ${results.summary.failed}`);
  console.log(`🌐 Real API Calls: ${results.summary.realApiCalls}`);
  console.log(`📈 Success Rate: ${((results.summary.passed / results.summary.total) * 100).toFixed(1)}%\n`);

  // Save detailed results
  const reportPath = path.join(__dirname, '../docs/REAL_API_TEST_RESULTS.json');
  fs.writeFileSync(reportPath, JSON.stringify(results, null, 2));
  console.log(`📄 Detailed results saved to: ${reportPath}\n`);

  process.exit(results.summary.failed > 0 ? 1 : 0);
}

main().catch(error => {
  console.error('Fatal error:', error);
  process.exit(1);
});
