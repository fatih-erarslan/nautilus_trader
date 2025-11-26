#!/usr/bin/env node
/**
 * Neural Trader - E2B Final Integration Test
 *
 * Tests E2B functionality using the correct NAPI function names
 * Found 10 E2B functions: runE2BAgent, createE2BSandbox, listE2BSandboxes, etc.
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
  e2bCredentials: {
    apiKey: process.env.E2B_API_KEY ? '✅ Configured' : '❌ Missing',
    accessToken: process.env.E2B_ACCESS_TOKEN ? '✅ Configured' : '❌ Missing',
  },
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
  console.log('🚀 NEURAL TRADER - E2B FINAL INTEGRATION TEST');
  console.log('='.repeat(80));
  console.log('\n⚠️  TESTING E2B TOOLS WITH REAL DATA - NO MOCKS\n');

  const startTime = Date.now();

  // Validate credentials
  console.log('📋 E2B Credentials Status:');
  console.log(`   API Key: ${results.e2bCredentials.apiKey}`);
  console.log(`   Access Token: ${results.e2bCredentials.accessToken}`);

  if (!process.env.E2B_API_KEY && !process.env.E2B_ACCESS_TOKEN) {
    console.error('\n❌ No E2B credentials found in .env');
    console.error('   Set E2B_API_KEY or E2B_ACCESS_TOKEN to run E2B tests\n');
    process.exit(1);
  }

  const napi = loadNapiModule();
  console.log(`\n✅ NAPI module loaded`);
  console.log(`📦 Available functions: ${Object.keys(napi).length}\n`);

  let sandboxId = null;

  // =============================================================================
  // Test 1: Create E2B Sandbox (camelCase: createE2BSandbox)
  // =============================================================================
  await runTest('E2B - Create Sandbox', async () => {
    const response = await napi.createE2BSandbox(
      'neural-trader-final-test', // name
      'node',                       // template
      300,                          // timeout
      null,                         // api_key (uses .env)
      null,                         // env_vars
      null                          // metadata
    );

    const data = JSON.parse(response);
    console.log(`   Sandbox ID: ${data.sandbox_id || data.sandboxId || 'N/A'}`);
    console.log(`   Status: ${data.status || 'N/A'}`);
    console.log(`   Message: ${data.message || 'N/A'}`);

    if (data.sandbox_id || data.sandboxId) {
      sandboxId = data.sandbox_id || data.sandboxId;
      if (data.status === 'success') {
        results.summary.realApiCalls++;
      }
    }

    return data;
  });

  // =============================================================================
  // Test 2: List E2B Sandboxes
  // =============================================================================
  await runTest('E2B - List Sandboxes', async () => {
    const response = await napi.listE2BSandboxes(null);
    const data = JSON.parse(response);

    console.log(`   Status: ${data.status || 'N/A'}`);

    if (data.sandboxes && Array.isArray(data.sandboxes)) {
      console.log(`   Active Sandboxes: ${data.sandboxes.length}`);
      if (data.status === 'success') {
        results.summary.realApiCalls++;
      }
    } else if (data.message) {
      console.log(`   Message: ${data.message}`);
    }

    return data;
  });

  // =============================================================================
  // Test 3: Get Sandbox Status
  // =============================================================================
  if (sandboxId) {
    await runTest('E2B - Get Sandbox Status', async () => {
      const response = await napi.getE2BSandboxStatus(sandboxId);
      const data = JSON.parse(response);

      console.log(`   Sandbox ID: ${data.sandbox_id || sandboxId}`);
      console.log(`   Status: ${data.status || 'N/A'}`);

      if (data.uptime !== undefined) {
        console.log(`   Uptime: ${data.uptime}s`);
        console.log(`   Memory: ${data.memory_mb || 'N/A'}MB`);
        console.log(`   CPU: ${data.cpu_usage || 'N/A'}%`);
        if (data.status === 'running') {
          results.summary.realApiCalls++;
        }
      }

      return data;
    });
  }

  // =============================================================================
  // Test 4: Execute Code in Sandbox
  // =============================================================================
  if (sandboxId) {
    await runTest('E2B - Execute JavaScript Code', async () => {
      const code = `
console.log("Hello from Neural Trader E2B!");
console.log("Testing sandbox execution...");

// Kelly Criterion calculation
function kellyFraction(winProb, odds) {
  return (winProb * odds - (1 - winProb)) / odds;
}

const prob = 0.55;
const odds = 2.0;
const bankroll = 10000;

const kelly = kellyFraction(prob, odds);
const optimalBet = bankroll * Math.max(0, kelly);

console.log(\`Win Probability: \${(prob * 100).toFixed(1)}%\`);
console.log(\`Odds: \${odds}x\`);
console.log(\`Kelly Fraction: \${kelly.toFixed(4)}\`);
console.log(\`Optimal Bet: $\${optimalBet.toFixed(2)}\`);

"Kelly calculation complete"
`;

      const response = await napi.executeE2BProcess(
        sandboxId,
        code,
        null,  // args
        60,    // timeout
        true,  // capture_output
        null   // working_dir
      );

      const data = JSON.parse(response);

      console.log(`   Status: ${data.status || 'N/A'}`);

      if (data.output) {
        const stdout = data.output.stdout || '';
        console.log(`   Output Lines: ${stdout.split('\n').filter(l => l.trim()).length}`);
        if (stdout.includes('Kelly')) {
          console.log(`   ✅ Kelly Criterion calculation executed`);
        }
        if (data.status === 'success') {
          results.summary.realApiCalls++;
        }
      }

      return data;
    });
  }

  // =============================================================================
  // Test 5: Run Trading Agent in Sandbox
  // =============================================================================
  if (sandboxId) {
    await runTest('E2B - Run Trading Agent', async () => {
      const response = await napi.runE2BAgent(
        sandboxId,
        'momentum',       // agent_type
        ['AAPL', 'MSFT'], // symbols
        false,            // use_gpu
        null              // strategy_params
      );

      const data = JSON.parse(response);

      console.log(`   Agent Type: ${data.agent_type || 'N/A'}`);
      console.log(`   Symbols: ${data.symbols ? data.symbols.join(', ') : 'N/A'}`);
      console.log(`   Status: ${data.status || 'N/A'}`);

      if (data.status === 'success' || data.status === 'running') {
        results.summary.realApiCalls++;
      }

      return data;
    });
  }

  // =============================================================================
  // Test 6: Monitor E2B Health
  // =============================================================================
  await runTest('E2B - Monitor Health', async () => {
    const response = await napi.monitorE2BHealth(false);
    const data = JSON.parse(response);

    console.log(`   Status: ${data.status || 'N/A'}`);

    if (data.health) {
      console.log(`   Active Sandboxes: ${data.health.active_sandboxes || 0}`);
      console.log(`   Total Deployments: ${data.health.total_deployments || 0}`);
    }

    return data;
  });

  // =============================================================================
  // Test 7: Deploy E2B Template
  // =============================================================================
  await runTest('E2B - Deploy Template', async () => {
    const response = await napi.deployE2BTemplate(
      'trading-bot',
      'e2b',
      { strategy: 'momentum', symbols: ['AAPL'] }
    );

    const data = JSON.parse(response);

    console.log(`   Template: ${data.template_name || 'N/A'}`);
    console.log(`   Category: ${data.category || 'N/A'}`);
    console.log(`   Status: ${data.status || 'N/A'}`);

    if (data.deployment_id) {
      console.log(`   Deployment ID: ${data.deployment_id}`);
    }

    return data;
  });

  // =============================================================================
  // Test 8: Scale E2B Deployment
  // =============================================================================
  await runTest('E2B - Scale Deployment', async () => {
    const response = await napi.scaleE2BDeployment(
      'deployment-test',
      3,
      false
    );

    const data = JSON.parse(response);

    console.log(`   Deployment ID: ${data.deployment_id || 'N/A'}`);
    console.log(`   Target Instances: ${data.instance_count || data.target_instances || 'N/A'}`);
    console.log(`   Status: ${data.status || 'N/A'}`);

    return data;
  });

  // =============================================================================
  // Test 9: Export E2B Template
  // =============================================================================
  if (sandboxId) {
    await runTest('E2B - Export Template', async () => {
      const response = await napi.exportE2BTemplate(
        sandboxId,
        'neural-trader-custom',
        false
      );

      const data = JSON.parse(response);

      console.log(`   Template Name: ${data.template_name || 'N/A'}`);
      console.log(`   Sandbox ID: ${data.sandbox_id || 'N/A'}`);
      console.log(`   Status: ${data.status || 'N/A'}`);

      return data;
    });
  }

  // =============================================================================
  // Test 10: Terminate Sandbox
  // =============================================================================
  if (sandboxId) {
    await runTest('E2B - Terminate Sandbox', async () => {
      const response = await napi.terminateE2BSandbox(sandboxId, false);
      const data = JSON.parse(response);

      console.log(`   Sandbox ID: ${data.sandbox_id || sandboxId}`);
      console.log(`   Status: ${data.status || 'N/A'}`);
      console.log(`   Message: ${data.message || 'N/A'}`);

      if (data.status === 'terminated' || data.status === 'success') {
        results.summary.realApiCalls++;
      }

      return data;
    });
  }

  // =============================================================================
  // Summary
  // =============================================================================
  const duration = Date.now() - startTime;

  console.log('\n' + '='.repeat(80));
  console.log('📊 E2B TEST RESULTS SUMMARY');
  console.log('='.repeat(80) + '\n');

  console.log(`Total Tests: ${results.summary.total}`);
  console.log(`✅ Passed: ${results.summary.passed}`);
  console.log(`❌ Failed: ${results.summary.failed}`);
  console.log(`🌐 Real API Calls: ${results.summary.realApiCalls}`);
  console.log(`⏱️  Total Duration: ${duration}ms`);
  console.log(`📈 Success Rate: ${((results.summary.passed / results.summary.total) * 100).toFixed(1)}%\n`);

  // Save detailed results
  const reportPath = path.join(__dirname, '../docs/E2B_FINAL_TEST_RESULTS.json');
  fs.writeFileSync(reportPath, JSON.stringify(results, null, 2));
  console.log(`📄 Detailed results saved to: ${reportPath}\n`);

  // Generate markdown report
  const mdReportPath = path.join(__dirname, '../docs/E2B_FINAL_TEST_REPORT.md');
  generateMarkdownReport(mdReportPath, results);
  console.log(`📄 Markdown report saved to: ${mdReportPath}\n`);

  process.exit(results.summary.failed > 0 ? 1 : 0);
}

function generateMarkdownReport(filepath, results) {
  const lines = [];

  lines.push('# Neural Trader - E2B Final Integration Test Report\n');
  lines.push(`**Generated:** ${results.timestamp}\n`);
  lines.push('**Test Method:** NAPI module with real E2B integration\n');

  lines.push('## E2B Credentials Status\n');
  lines.push('| Credential | Status |');
  lines.push('|------------|--------|');
  lines.push(`| API Key | ${results.e2bCredentials.apiKey} |`);
  lines.push(`| Access Token | ${results.e2bCredentials.accessToken} |\n`);

  lines.push('## Test Summary\n');
  lines.push('| Metric | Value |');
  lines.push('|--------|-------|');
  lines.push(`| Total Tests | ${results.summary.total} |`);
  lines.push(`| Passed | ${results.summary.passed} |`);
  lines.push(`| Failed | ${results.summary.failed} |`);
  lines.push(`| Real API Calls | ${results.summary.realApiCalls} |`);
  lines.push(`| Success Rate | ${((results.summary.passed / results.summary.total) * 100).toFixed(1)}% |\n`);

  lines.push('## Test Results\n');
  lines.push('| Test | Status | Latency | Details |');
  lines.push('|------|--------|---------|---------|');

  for (const test of results.tests) {
    const status = test.success ? '✅ Pass' : '❌ Fail';
    const details = test.error || 'Success';
    lines.push(`| ${test.name} | ${status} | ${test.latency}ms | ${details} |`);
  }
  lines.push('');

  lines.push('## Detailed Test Results\n');
  for (const test of results.tests) {
    lines.push(`### ${test.name}\n`);
    lines.push(`**Status:** ${test.success ? '✅ Passed' : '❌ Failed'}  `);
    lines.push(`**Latency:** ${test.latency}ms\n`);

    if (test.response) {
      lines.push('**Response:**');
      lines.push('```json');
      lines.push(JSON.stringify(test.response, null, 2));
      lines.push('```\n');
    }

    if (test.error) {
      lines.push(`**Error:** ${test.error}\n`);
    }
  }

  lines.push('---\n');
  lines.push('**Test Type:** Real E2B Integration via NAPI  ');
  lines.push('**Module:** neural-trader.linux-x64-gnu.node  ');
  lines.push(`**Timestamp:** ${results.timestamp}\n`);

  lines.push('## E2B Functions Tested\n');
  lines.push('1. `createE2BSandbox` - Create isolated execution environment');
  lines.push('2. `listE2BSandboxes` - List all active sandboxes');
  lines.push('3. `getE2BSandboxStatus` - Get sandbox runtime status');
  lines.push('4. `executeE2BProcess` - Execute code in sandbox');
  lines.push('5. `runE2BAgent` - Run trading agent in sandbox');
  lines.push('6. `monitorE2BHealth` - Monitor E2B infrastructure health');
  lines.push('7. `deployE2BTemplate` - Deploy pre-configured template');
  lines.push('8. `scaleE2BDeployment` - Scale deployment instances');
  lines.push('9. `exportE2BTemplate` - Export sandbox as template');
  lines.push('10. `terminateE2BSandbox` - Stop and cleanup sandbox\n');

  fs.writeFileSync(filepath, lines.join('\n'));
}

// Run tests
main().catch(error => {
  console.error('\n❌ Fatal error:', error);
  console.error(error.stack);
  process.exit(1);
});
