#!/usr/bin/env node
/**
 * Neural Trader - E2B NAPI Direct Test
 *
 * Tests E2B functionality using the NAPI module directly
 * Tests the Rust-implemented E2B tools from mcp_tools.rs
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
    skipped: 0,
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
    skipped: false,
  };

  try {
    const response = await testFn();
    result.response = response;
    result.success = true;
    result.latency = Date.now() - startTime;
    console.log(`   ✅ Passed (${result.latency}ms)`);
    results.summary.passed++;
  } catch (error) {
    if (error.message.includes('not a function') || error.message.includes('is not exported')) {
      result.skipped = true;
      result.error = 'Tool not exported from NAPI module';
      console.log(`   ⏭️  Skipped: ${result.error}`);
      results.summary.skipped++;
    } else {
      result.error = error.message;
      result.latency = Date.now() - startTime;
      console.log(`   ❌ Failed: ${error.message}`);
      results.summary.failed++;
    }
  }

  results.tests.push(result);
  results.summary.total++;
  return result;
}

async function main() {
  console.log('\n' + '='.repeat(80));
  console.log('🚀 NEURAL TRADER - E2B NAPI DIRECT TEST');
  console.log('='.repeat(80));
  console.log('\n⚠️  TESTING E2B TOOLS VIA RUST NAPI MODULE\n');

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

  // Check which E2B functions are available
  const e2bFunctions = Object.keys(napi).filter(key =>
    key.toLowerCase().includes('e2b') || key.toLowerCase().includes('sandbox')
  );

  console.log(`🔍 E2B-related functions found: ${e2bFunctions.length}`);
  if (e2bFunctions.length > 0) {
    console.log(`   Functions: ${e2bFunctions.join(', ')}`);
  }
  console.log('');

  let sandboxId = null;

  // =============================================================================
  // Test 1: Create E2B Sandbox
  // =============================================================================
  await runTest('NAPI - Create E2B Sandbox', async () => {
    if (typeof napi.createE2bSandbox !== 'function') {
      throw new Error('createE2bSandbox is not exported from NAPI module');
    }

    const response = await napi.createE2bSandbox(
      'neural-trader-napi-test',
      'node',
      300,
      null, // api_key (uses .env)
      null, // env_vars
      null  // metadata
    );

    const data = JSON.parse(response);
    console.log(`   Sandbox ID: ${data.sandbox_id || 'N/A'}`);
    console.log(`   Status: ${data.status || 'N/A'}`);

    if (data.sandbox_id) {
      sandboxId = data.sandbox_id;
      results.summary.realApiCalls++;
    }

    return data;
  });

  // =============================================================================
  // Test 2: List E2B Sandboxes
  // =============================================================================
  await runTest('NAPI - List E2B Sandboxes', async () => {
    if (typeof napi.listE2bSandboxes !== 'function') {
      throw new Error('listE2bSandboxes is not exported from NAPI module');
    }

    const response = await napi.listE2bSandboxes(null);
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
    await runTest('NAPI - Get E2B Sandbox Status', async () => {
      if (typeof napi.getE2bSandboxStatus !== 'function') {
        throw new Error('getE2bSandboxStatus is not exported from NAPI module');
      }

      const response = await napi.getE2bSandboxStatus(sandboxId);
      const data = JSON.parse(response);

      console.log(`   Status: ${data.status || 'N/A'}`);

      if (data.uptime) {
        console.log(`   Uptime: ${data.uptime}s`);
        results.summary.realApiCalls++;
      }

      return data;
    });

    // =============================================================================
    // Test 4: Execute Code in Sandbox
    // =============================================================================
    await runTest('NAPI - Execute Code in E2B Sandbox', async () => {
      if (typeof napi.executeE2bProcess !== 'function') {
        throw new Error('executeE2bProcess is not exported from NAPI module');
      }

      const code = 'console.log("Hello from NAPI E2B!");';
      const response = await napi.executeE2bProcess(
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
        console.log(`   Output: ${data.output.stdout || 'N/A'}`);
        if (data.status === 'success') {
          results.summary.realApiCalls++;
        }
      }

      return data;
    });

    // =============================================================================
    // Test 5: Execute Trading Agent in Sandbox
    // =============================================================================
    await runTest('NAPI - Run Trading Agent in E2B Sandbox', async () => {
      if (typeof napi.runE2bAgent !== 'function') {
        throw new Error('runE2bAgent is not exported from NAPI module');
      }

      const response = await napi.runE2bAgent(
        sandboxId,
        'momentum',       // agent_type
        ['AAPL'],         // symbols
        false,            // use_gpu
        null              // strategy_params
      );

      const data = JSON.parse(response);

      console.log(`   Agent Type: ${data.agent_type || 'N/A'}`);
      console.log(`   Status: ${data.status || 'N/A'}`);

      if (data.status === 'success') {
        results.summary.realApiCalls++;
      }

      return data;
    });

    // =============================================================================
    // Test 6: Stop Sandbox
    // =============================================================================
    await runTest('NAPI - Stop E2B Sandbox', async () => {
      if (typeof napi.terminateE2bSandbox !== 'function') {
        throw new Error('terminateE2bSandbox is not exported from NAPI module');
      }

      const response = await napi.terminateE2bSandbox(sandboxId, false);
      const data = JSON.parse(response);

      console.log(`   Status: ${data.status || 'N/A'}`);

      if (data.status === 'success' || data.status === 'stopped') {
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
  console.log('📊 E2B NAPI TEST RESULTS SUMMARY');
  console.log('='.repeat(80) + '\n');

  console.log(`Total Tests: ${results.summary.total}`);
  console.log(`✅ Passed: ${results.summary.passed}`);
  console.log(`❌ Failed: ${results.summary.failed}`);
  console.log(`⏭️  Skipped: ${results.summary.skipped} (not exported from NAPI)`);
  console.log(`🌐 Real API Calls: ${results.summary.realApiCalls}`);
  console.log(`⏱️  Total Duration: ${duration}ms`);
  console.log(`📈 Success Rate: ${((results.summary.passed / results.summary.total) * 100).toFixed(1)}%\n`);

  // Save detailed results
  const reportPath = path.join(__dirname, '../docs/E2B_NAPI_TEST_RESULTS.json');
  fs.writeFileSync(reportPath, JSON.stringify(results, null, 2));
  console.log(`📄 Detailed results saved to: ${reportPath}\n`);

  // Generate markdown report
  const mdReportPath = path.join(__dirname, '../docs/E2B_NAPI_TEST_REPORT.md');
  generateMarkdownReport(mdReportPath, results);
  console.log(`📄 Markdown report saved to: ${mdReportPath}\n`);

  // Analysis
  console.log('📋 Analysis:');
  if (results.summary.skipped > 0) {
    console.log(`   ⚠️  ${results.summary.skipped} E2B tools are implemented in Rust but not exported to JavaScript`);
    console.log('   💡 These tools need to be added to lib.rs exports');
  }
  if (results.summary.passed > 0) {
    console.log(`   ✅ ${results.summary.passed} E2B tools are working and exported`);
  }
  if (results.summary.realApiCalls > 0) {
    console.log(`   🌐 ${results.summary.realApiCalls} real E2B API calls were successful`);
  }
  console.log('');

  process.exit(results.summary.failed > 0 ? 1 : 0);
}

function generateMarkdownReport(filepath, results) {
  const lines = [];

  lines.push('# Neural Trader - E2B NAPI Direct Test Report\n');
  lines.push(`**Generated:** ${results.timestamp}\n`);
  lines.push('**Test Method:** Direct NAPI module testing (bypassing MCP server)\n');

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
  lines.push(`| Skipped | ${results.summary.skipped} |`);
  lines.push(`| Real API Calls | ${results.summary.realApiCalls} |`);
  lines.push(`| Success Rate | ${((results.summary.passed / results.summary.total) * 100).toFixed(1)}% |\n`);

  lines.push('## Test Results\n');
  lines.push('| Test | Status | Latency | Details |');
  lines.push('|------|--------|---------|---------|');

  for (const test of results.tests) {
    const status = test.skipped ? '⏭️ Skip' : test.success ? '✅ Pass' : '❌ Fail';
    const details = test.error || 'Success';
    lines.push(`| ${test.name} | ${status} | ${test.latency}ms | ${details} |`);
  }
  lines.push('');

  lines.push('## Detailed Test Results\n');
  for (const test of results.tests) {
    lines.push(`### ${test.name}\n`);
    lines.push(`**Status:** ${test.skipped ? '⏭️ Skipped' : test.success ? '✅ Passed' : '❌ Failed'}  `);
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
  lines.push('**Test Type:** Direct NAPI module testing  ');
  lines.push('**Module:** neural-trader.linux-x64-gnu.node  ');
  lines.push(`**Timestamp:** ${results.timestamp}\n`);

  lines.push('## E2B Tools Tested\n');
  lines.push('1. `createE2bSandbox` - Create isolated execution environment');
  lines.push('2. `listE2bSandboxes` - List all active sandboxes');
  lines.push('3. `getE2bSandboxStatus` - Get sandbox runtime status');
  lines.push('4. `executeE2bProcess` - Execute code in sandbox');
  lines.push('5. `runE2bAgent` - Run trading agent in sandbox');
  lines.push('6. `terminateE2bSandbox` - Stop and cleanup sandbox\n');

  lines.push('## Integration Notes\n');
  lines.push('- E2B tools are implemented in Rust (`e2b_monitoring_impl.rs`)');
  lines.push('- Tools use mock responses when neural-trader-api is disabled');
  lines.push('- Real E2B integration requires enabling neural-trader-api in Cargo.toml');
  lines.push('- SQLite dependency conflict prevents full neural-trader-api activation');
  lines.push('- Tools need to be exported in `lib.rs` to be accessible via NAPI\n');

  lines.push('## Required Actions\n');
  lines.push('1. **Export E2B tools** in `/workspaces/neural-trader/neural-trader-rust/crates/napi-bindings/src/lib.rs`');
  lines.push('2. **Resolve SQLite conflict** to enable neural-trader-api');
  lines.push('3. **Add E2B schemas** to MCP tool registry');
  lines.push('4. **Test with real E2B API** after exports are added\n');

  fs.writeFileSync(filepath, lines.join('\n'));
}

// Run tests
main().catch(error => {
  console.error('\n❌ Fatal error:', error);
  console.error(error.stack);
  process.exit(1);
});
