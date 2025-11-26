#!/usr/bin/env node
/**
 * Core Trading MCP Tools - Comprehensive Benchmark Suite
 *
 * Tests 6 core trading tools:
 * 1. ping - Connectivity verification
 * 2. list_strategies - Strategy enumeration
 * 3. get_strategy_info - Strategy details
 * 4. quick_analysis - Market analysis
 * 5. simulate_trade - Trade simulation (REMOVED - using real execution)
 * 6. get_portfolio_status - Portfolio retrieval
 */

const { loadNativeBinding } = require('@napi-rs/load');
const { performance } = require('perf_hooks');
const path = require('path');

// Load NAPI bindings
const nativeBinding = loadNativeBinding(
  path.join(__dirname, '../neural-trader-rust/packages/neural-trader-backend'),
  'neural-trader-backend',
  '@neural-trader/backend'
);

// Configuration
const CONCURRENCY_LEVELS = [1, 10, 50, 100];
const ITERATIONS_PER_LEVEL = 100;
const SLA_TARGETS = {
  ping: 50,                     // 50ms
  list_strategies: 200,         // 200ms
  get_strategy_info: 150,       // 150ms
  quick_analysis: 300,          // 300ms (with data fetching)
  get_portfolio_status: 250     // 250ms
};

// Benchmark results storage
const results = {
  timestamp: new Date().toISOString(),
  toolResults: {},
  securityFindings: [],
  optimizationRecommendations: []
};

/**
 * Measure latency for a single tool call
 */
async function measureLatency(toolFn, ...args) {
  const start = performance.now();
  try {
    const result = await toolFn(...args);
    const end = performance.now();
    return {
      success: true,
      latency: end - start,
      result: JSON.parse(result)
    };
  } catch (error) {
    const end = performance.now();
    return {
      success: false,
      latency: end - start,
      error: error.message
    };
  }
}

/**
 * Run benchmark for a specific tool
 */
async function benchmarkTool(toolName, toolFn, args = []) {
  console.log(`\n🔬 Benchmarking: ${toolName}`);
  console.log('━'.repeat(60));

  const toolResults = {
    name: toolName,
    slaTarget: SLA_TARGETS[toolName] || 200,
    measurements: [],
    concurrencyTests: []
  };

  // Single call baseline
  console.log('📊 Baseline measurement...');
  const baseline = await measureLatency(toolFn, ...args);
  toolResults.baseline = baseline;

  if (!baseline.success) {
    console.error(`❌ Baseline failed: ${baseline.error}`);
    return toolResults;
  }

  console.log(`✅ Baseline latency: ${baseline.latency.toFixed(2)}ms`);
  console.log(`📋 Response sample:`, JSON.stringify(baseline.result, null, 2).substring(0, 200) + '...');

  // Sequential iterations for statistical significance
  console.log(`\n📈 Running ${ITERATIONS_PER_LEVEL} sequential iterations...`);
  const latencies = [];

  for (let i = 0; i < ITERATIONS_PER_LEVEL; i++) {
    const measurement = await measureLatency(toolFn, ...args);
    if (measurement.success) {
      latencies.push(measurement.latency);
    }
    if ((i + 1) % 20 === 0) {
      process.stdout.write(`  Progress: ${i + 1}/${ITERATIONS_PER_LEVEL}\r`);
    }
  }

  console.log(`\n  Completed: ${latencies.length}/${ITERATIONS_PER_LEVEL}`);

  // Calculate statistics
  latencies.sort((a, b) => a - b);
  const stats = {
    count: latencies.length,
    min: latencies[0],
    max: latencies[latencies.length - 1],
    mean: latencies.reduce((a, b) => a + b, 0) / latencies.length,
    p50: latencies[Math.floor(latencies.length * 0.50)],
    p95: latencies[Math.floor(latencies.length * 0.95)],
    p99: latencies[Math.floor(latencies.length * 0.99)],
    slaCompliance: (latencies.filter(l => l <= toolResults.slaTarget).length / latencies.length) * 100
  };

  toolResults.statistics = stats;

  console.log(`\n📊 Statistics:`);
  console.log(`  Mean:    ${stats.mean.toFixed(2)}ms`);
  console.log(`  Median:  ${stats.p50.toFixed(2)}ms`);
  console.log(`  P95:     ${stats.p95.toFixed(2)}ms`);
  console.log(`  P99:     ${stats.p99.toFixed(2)}ms`);
  console.log(`  Range:   ${stats.min.toFixed(2)}ms - ${stats.max.toFixed(2)}ms`);
  console.log(`  SLA:     ${stats.slaCompliance.toFixed(1)}% (target: <${toolResults.slaTarget}ms)`);

  // Concurrency testing
  for (const concurrency of CONCURRENCY_LEVELS) {
    console.log(`\n⚡ Testing with ${concurrency} concurrent requests...`);

    const startTime = performance.now();
    const promises = Array(concurrency).fill().map(() => measureLatency(toolFn, ...args));
    const concurrentResults = await Promise.all(promises);
    const endTime = performance.now();

    const successfulResults = concurrentResults.filter(r => r.success);
    const successRate = (successfulResults.length / concurrency) * 100;
    const avgLatency = successfulResults.reduce((sum, r) => sum + r.latency, 0) / successfulResults.length;
    const throughput = (concurrency / (endTime - startTime)) * 1000; // requests per second

    const concurrencyTest = {
      concurrency,
      successRate,
      avgLatency,
      throughput,
      totalTime: endTime - startTime
    };

    toolResults.concurrencyTests.push(concurrencyTest);

    console.log(`  Success:     ${successRate.toFixed(1)}%`);
    console.log(`  Avg Latency: ${avgLatency.toFixed(2)}ms`);
    console.log(`  Throughput:  ${throughput.toFixed(0)} req/sec`);
  }

  return toolResults;
}

/**
 * Test input validation and error handling
 */
async function testInputValidation(toolName, toolFn, testCases) {
  console.log(`\n🛡️  Testing input validation for ${toolName}...`);

  const validationResults = [];

  for (const testCase of testCases) {
    try {
      const result = await toolFn(...testCase.args);
      validationResults.push({
        test: testCase.name,
        expected: 'error',
        actual: 'success',
        passed: false,
        result
      });
    } catch (error) {
      validationResults.push({
        test: testCase.name,
        expected: 'error',
        actual: 'error',
        passed: true,
        errorMessage: error.message
      });
    }
  }

  return validationResults;
}

/**
 * Analyze security aspects
 */
function analyzeSecurityFindings(toolResults) {
  const findings = [];

  // Check for missing rate limiting
  for (const tool of Object.values(toolResults)) {
    const maxThroughput = Math.max(...tool.concurrencyTests.map(t => t.throughput));
    if (maxThroughput > 1000) {
      findings.push({
        severity: 'MEDIUM',
        tool: tool.name,
        issue: 'High throughput without apparent rate limiting',
        detail: `Tool achieved ${maxThroughput.toFixed(0)} req/sec`,
        recommendation: 'Implement rate limiting at API gateway or tool level'
      });
    }
  }

  return findings;
}

/**
 * Generate optimization recommendations
 */
function generateOptimizations(toolResults) {
  const recommendations = [];

  for (const tool of Object.values(toolResults)) {
    // Check if caching could help
    if (tool.statistics && tool.statistics.mean > 100) {
      const stdDev = Math.sqrt(
        tool.measurements
          .map(m => Math.pow(m - tool.statistics.mean, 2))
          .reduce((a, b) => a + b, 0) / tool.measurements.length
      );

      if (stdDev < tool.statistics.mean * 0.3) {
        // Low variance suggests consistent computation - good caching candidate
        recommendations.push({
          priority: 'HIGH',
          tool: tool.name,
          optimization: 'Response caching',
          expectedImpact: `Reduce latency by ~${(tool.statistics.mean * 0.8).toFixed(0)}ms`,
          implementation: 'Cache results with TTL based on data freshness requirements'
        });
      }
    }

    // Check for SLA violations
    if (tool.statistics && tool.statistics.slaCompliance < 95) {
      recommendations.push({
        priority: 'CRITICAL',
        tool: tool.name,
        optimization: 'Performance improvement required',
        expectedImpact: `Improve SLA compliance from ${tool.statistics.slaCompliance.toFixed(1)}% to >95%`,
        implementation: 'Profile code, optimize database queries, consider parallel processing'
      });
    }

    // Check for concurrency bottlenecks
    if (tool.concurrencyTests && tool.concurrencyTests.length >= 2) {
      const lowConcurrency = tool.concurrencyTests[1]; // 10 concurrent
      const highConcurrency = tool.concurrencyTests[3]; // 100 concurrent

      if (highConcurrency.avgLatency > lowConcurrency.avgLatency * 3) {
        recommendations.push({
          priority: 'HIGH',
          tool: tool.name,
          optimization: 'Improve concurrency handling',
          expectedImpact: `Reduce latency degradation under load`,
          implementation: 'Implement connection pooling, async processing, or resource limits'
        });
      }
    }
  }

  return recommendations;
}

/**
 * Main benchmark execution
 */
async function main() {
  console.log('╔════════════════════════════════════════════════════════════╗');
  console.log('║     Core Trading MCP Tools - Benchmark Suite              ║');
  console.log('║     Testing 6 tools with performance & security analysis  ║');
  console.log('╚════════════════════════════════════════════════════════════╝');
  console.log('');
  console.log(`Start time: ${new Date().toISOString()}`);
  console.log(`SLA Targets: ${JSON.stringify(SLA_TARGETS, null, 2)}`);

  try {
    // 1. ping
    results.toolResults.ping = await benchmarkTool(
      'ping',
      nativeBinding.ping
    );

    // 2. list_strategies
    results.toolResults.list_strategies = await benchmarkTool(
      'list_strategies',
      nativeBinding.listStrategies
    );

    // 3. get_strategy_info
    results.toolResults.get_strategy_info = await benchmarkTool(
      'get_strategy_info',
      nativeBinding.getStrategyInfo,
      ['momentum']
    );

    // 4. quick_analysis
    results.toolResults.quick_analysis = await benchmarkTool(
      'quick_analysis',
      nativeBinding.quickAnalysis,
      ['AAPL', false]
    );

    // 5. get_portfolio_status
    results.toolResults.get_portfolio_status = await benchmarkTool(
      'get_portfolio_status',
      nativeBinding.getPortfolioStatus,
      [true]
    );

    // Security analysis
    console.log('\n\n🔒 Security Analysis');
    console.log('━'.repeat(60));
    results.securityFindings = analyzeSecurityFindings(results.toolResults);

    if (results.securityFindings.length === 0) {
      console.log('✅ No security concerns identified');
    } else {
      results.securityFindings.forEach(finding => {
        console.log(`\n[${finding.severity}] ${finding.tool}`);
        console.log(`  Issue: ${finding.issue}`);
        console.log(`  Detail: ${finding.detail}`);
        console.log(`  Fix: ${finding.recommendation}`);
      });
    }

    // Optimization recommendations
    console.log('\n\n⚡ Optimization Recommendations');
    console.log('━'.repeat(60));
    results.optimizationRecommendations = generateOptimizations(results.toolResults);

    if (results.optimizationRecommendations.length === 0) {
      console.log('✅ All tools performing within acceptable parameters');
    } else {
      results.optimizationRecommendations
        .sort((a, b) => {
          const priority = { CRITICAL: 0, HIGH: 1, MEDIUM: 2, LOW: 3 };
          return priority[a.priority] - priority[b.priority];
        })
        .forEach((rec, idx) => {
          console.log(`\n${idx + 1}. [${rec.priority}] ${rec.tool}`);
          console.log(`   Optimization: ${rec.optimization}`);
          console.log(`   Impact: ${rec.expectedImpact}`);
          console.log(`   How: ${rec.implementation}`);
        });
    }

    // Summary
    console.log('\n\n📊 Summary');
    console.log('━'.repeat(60));

    const toolNames = Object.keys(results.toolResults);
    const avgCompliance = toolNames.reduce((sum, name) => {
      return sum + (results.toolResults[name].statistics?.slaCompliance || 0);
    }, 0) / toolNames.length;

    console.log(`Tools tested: ${toolNames.length}`);
    console.log(`Overall SLA compliance: ${avgCompliance.toFixed(1)}%`);
    console.log(`Security findings: ${results.securityFindings.length}`);
    console.log(`Optimization opportunities: ${results.optimizationRecommendations.length}`);

    // Save detailed results
    const fs = require('fs');
    const outputPath = path.join(__dirname, '../docs/mcp-analysis/benchmark_results.json');
    fs.mkdirSync(path.dirname(outputPath), { recursive: true });
    fs.writeFileSync(outputPath, JSON.stringify(results, null, 2));

    console.log(`\n✅ Detailed results saved to: ${outputPath}`);
    console.log(`\nBenchmark completed: ${new Date().toISOString()}`);

  } catch (error) {
    console.error('\n❌ Benchmark failed:', error);
    process.exit(1);
  }
}

// Run if called directly
if (require.main === module) {
  main().catch(console.error);
}

module.exports = { benchmarkTool, measureLatency };
