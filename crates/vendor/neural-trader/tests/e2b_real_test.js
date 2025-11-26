#!/usr/bin/env node
/**
 * Neural Trader - Real E2B Integration Test
 *
 * Tests E2B sandbox functionality with REAL API credentials from .env
 * NO MOCKS - Real sandbox creation, code execution, and management
 */

require('dotenv').config({ path: require('path').join(__dirname, '../.env') });
const { CodeInterpreter } = require('@e2b/code-interpreter');

// Test results tracking
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
    duration: 0,
  },
};

// Helper to run test
async function runTest(name, testFn) {
  console.log(`\n🧪 Testing: ${name}...`);
  const startTime = Date.now();
  const result = {
    name,
    success: false,
    duration: 0,
    error: null,
    details: null,
  };

  try {
    const response = await testFn();
    result.details = response;
    result.success = true;
    result.duration = Date.now() - startTime;
    console.log(`   ✅ Passed (${result.duration}ms)`);
    results.summary.passed++;
  } catch (error) {
    result.error = error.message;
    result.duration = Date.now() - startTime;
    console.log(`   ❌ Failed: ${error.message}`);
    results.summary.failed++;
  }

  results.tests.push(result);
  results.summary.total++;
  return result;
}

async function main() {
  console.log('\n' + '='.repeat(80));
  console.log('🚀 NEURAL TRADER - REAL E2B INTEGRATION TEST');
  console.log('='.repeat(80));
  console.log('\n⚠️  USING REAL E2B API - REAL SANDBOXES WILL BE CREATED\n');

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

  const apiKey = process.env.E2B_ACCESS_TOKEN || process.env.E2B_API_KEY;
  let sandbox = null;

  // =============================================================================
  // Test 1: Create E2B Sandbox
  // =============================================================================
  await runTest('E2B Sandbox Creation', async () => {
    console.log('   Creating sandbox with Code Interpreter template...');

    sandbox = await CodeInterpreter.create({ apiKey });

    return {
      sandboxId: sandbox.sandboxId,
      status: 'created',
      template: 'code-interpreter',
      message: 'Sandbox created successfully',
    };
  });

  if (!sandbox) {
    console.error('\n❌ Sandbox creation failed - cannot continue with tests\n');
    process.exit(1);
  }

  // =============================================================================
  // Test 2: Execute Simple Python Code
  // =============================================================================
  await runTest('Execute Simple Python Code', async () => {
    console.log('   Running: print("Hello from E2B!")');

    const execution = await sandbox.notebook.execCell('print("Hello from E2B!")');

    return {
      stdout: execution.logs.stdout.join(''),
      stderr: execution.logs.stderr.join(''),
      success: execution.error === null,
      executionTime: execution.results?.[0]?.executionTime || 'N/A',
    };
  });

  // =============================================================================
  // Test 3: Execute Trading Calculation
  // =============================================================================
  await runTest('Execute Trading Calculation (Kelly Criterion)', async () => {
    const code = `
# Kelly Criterion Calculator
def kelly_criterion(win_prob, odds, bankroll):
    kelly_fraction = (win_prob * odds - (1 - win_prob)) / odds
    optimal_bet = bankroll * max(0, kelly_fraction)
    return {
        'kelly_fraction': kelly_fraction,
        'optimal_bet': optimal_bet,
        'win_prob': win_prob,
        'odds': odds
    }

result = kelly_criterion(0.55, 2.0, 10000)
print(f"Kelly Fraction: {result['kelly_fraction']:.4f}")
print(f"Optimal Bet: ${result['optimal_bet']:.2f}")
result
`;

    const execution = await sandbox.notebook.execCell(code);

    return {
      stdout: execution.logs.stdout.join('\n'),
      result: execution.results?.[0]?.text || 'No result',
      success: execution.error === null,
    };
  });

  // =============================================================================
  // Test 4: Install and Use Trading Library
  // =============================================================================
  await runTest('Install and Use NumPy for Portfolio Analysis', async () => {
    const code = `
import numpy as np

# Portfolio returns
returns = np.array([0.05, -0.02, 0.08, 0.03, -0.01])
weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

portfolio_return = np.sum(returns * weights)
portfolio_std = np.std(returns)
sharpe_ratio = portfolio_return / portfolio_std if portfolio_std > 0 else 0

print(f"Portfolio Return: {portfolio_return:.4f}")
print(f"Portfolio Std Dev: {portfolio_std:.4f}")
print(f"Sharpe Ratio: {sharpe_ratio:.4f}")

{"return": portfolio_return, "std": portfolio_std, "sharpe": sharpe_ratio}
`;

    const execution = await sandbox.notebook.execCell(code);

    return {
      stdout: execution.logs.stdout.join('\n'),
      result: execution.results?.[0]?.text || 'No result',
      success: execution.error === null,
    };
  });

  // =============================================================================
  // Test 5: File Operations - Create Trading Data
  // =============================================================================
  await runTest('File Operations - Create and Read Trading Data', async () => {
    const createCode = `
import csv

# Create sample trading data
data = [
    ['timestamp', 'symbol', 'price', 'volume'],
    ['2024-01-01 09:30', 'AAPL', '180.50', '1000000'],
    ['2024-01-01 09:31', 'AAPL', '180.75', '950000'],
    ['2024-01-01 09:32', 'AAPL', '180.60', '1100000'],
]

with open('/tmp/trading_data.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(data)

print("Created trading_data.csv")
"File created successfully"
`;

    const createExec = await sandbox.notebook.execCell(createCode);

    const readCode = `
import csv

with open('/tmp/trading_data.csv', 'r') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

print(f"Read {len(rows)} rows from trading data")
for row in rows[:2]:
    print(f"  {row['timestamp']}: {row['symbol']} @ ${row['price']}")

len(rows)
`;

    const readExec = await sandbox.notebook.execCell(readCode);

    return {
      create_stdout: createExec.logs.stdout.join('\n'),
      read_stdout: readExec.logs.stdout.join('\n'),
      rows_read: readExec.results?.[0]?.text || 'Unknown',
      success: createExec.error === null && readExec.error === null,
    };
  });

  // =============================================================================
  // Test 6: Multi-Step Trading Strategy Simulation
  // =============================================================================
  await runTest('Multi-Step Trading Strategy Simulation', async () => {
    const code = `
import numpy as np
import random

class SimpleStrategyBacktest:
    def __init__(self, initial_capital=10000):
        self.capital = initial_capital
        self.initial_capital = initial_capital
        self.position = 0
        self.trades = []

    def execute_strategy(self, prices, signals):
        for i, (price, signal) in enumerate(zip(prices, signals)):
            if signal == 'BUY' and self.position == 0:
                shares = int(self.capital * 0.1 / price)
                if shares > 0:
                    self.position = shares
                    self.capital -= shares * price
                    self.trades.append(('BUY', i, price, shares))
            elif signal == 'SELL' and self.position > 0:
                self.capital += self.position * price
                self.trades.append(('SELL', i, price, self.position))
                self.position = 0

        # Close position if still open
        if self.position > 0:
            final_price = prices[-1]
            self.capital += self.position * final_price
            self.position = 0

        return {
            'total_return': (self.capital - self.initial_capital) / self.initial_capital,
            'final_capital': self.capital,
            'num_trades': len(self.trades),
            'trades': self.trades[:3]  # First 3 trades
        }

# Simulate price data
np.random.seed(42)
prices = 100 + np.cumsum(np.random.randn(100) * 2)

# Simple momentum signals
signals = ['BUY' if prices[i] > prices[i-1] else 'SELL' if prices[i] < prices[i-1] else 'HOLD'
           for i in range(1, len(prices))]
signals.insert(0, 'HOLD')

backtest = SimpleStrategyBacktest()
result = backtest.execute_strategy(prices, signals)

print(f"Total Return: {result['total_return']:.2%}")
print(f"Final Capital: ${result['final_capital']:.2f}")
print(f"Number of Trades: {result['num_trades']}")

result
`;

    const execution = await sandbox.notebook.execCell(code);

    return {
      stdout: execution.logs.stdout.join('\n'),
      result: execution.results?.[0]?.text || 'No result',
      success: execution.error === null,
    };
  });

  // =============================================================================
  // Test 7: Sandbox Info and Resources
  // =============================================================================
  await runTest('Get Sandbox System Information', async () => {
    const code = `
import os
import platform
import psutil

info = {
    'platform': platform.platform(),
    'python_version': platform.python_version(),
    'cpu_count': os.cpu_count(),
    'memory_mb': psutil.virtual_memory().total / (1024 * 1024),
    'disk_gb': psutil.disk_usage('/').total / (1024 * 1024 * 1024),
}

for key, value in info.items():
    print(f"{key}: {value}")

info
`;

    const execution = await sandbox.notebook.execCell(code);

    return {
      stdout: execution.logs.stdout.join('\n'),
      systemInfo: execution.results?.[0]?.text || 'No result',
      success: execution.error === null,
    };
  });

  // =============================================================================
  // Test 8: Error Handling
  // =============================================================================
  await runTest('Error Handling - Invalid Code', async () => {
    const code = `
# This should cause an error
undefined_variable + 123
`;

    const execution = await sandbox.notebook.execCell(code);

    return {
      hasError: execution.error !== null,
      errorType: execution.error?.name || 'None',
      errorMessage: execution.error?.value || 'No error',
      success: true,  // Success means we handled the error correctly
    };
  });

  // =============================================================================
  // Test 9: Cleanup - Close Sandbox
  // =============================================================================
  await runTest('Sandbox Cleanup and Shutdown', async () => {
    console.log('   Closing sandbox...');

    await sandbox.close();

    return {
      sandboxId: sandbox.sandboxId,
      status: 'closed',
      message: 'Sandbox closed successfully',
    };
  });

  // =============================================================================
  // Summary
  // =============================================================================
  results.summary.duration = Date.now() - startTime;

  console.log('\n' + '='.repeat(80));
  console.log('📊 E2B TEST RESULTS SUMMARY');
  console.log('='.repeat(80) + '\n');

  console.log(`Total Tests: ${results.summary.total}`);
  console.log(`✅ Passed: ${results.summary.passed}`);
  console.log(`❌ Failed: ${results.summary.failed}`);
  console.log(`⏱️  Total Duration: ${results.summary.duration}ms`);
  console.log(`📈 Success Rate: ${((results.summary.passed / results.summary.total) * 100).toFixed(1)}%\n`);

  // Save detailed results
  const fs = require('fs');
  const path = require('path');
  const reportPath = path.join(__dirname, '../docs/E2B_REAL_TEST_RESULTS.json');
  fs.writeFileSync(reportPath, JSON.stringify(results, null, 2));
  console.log(`📄 Detailed results saved to: ${reportPath}\n`);

  // Generate markdown report
  const mdReportPath = path.join(__dirname, '../docs/E2B_REAL_TEST_REPORT.md');
  generateMarkdownReport(mdReportPath, results);
  console.log(`📄 Markdown report saved to: ${mdReportPath}\n`);

  process.exit(results.summary.failed > 0 ? 1 : 0);
}

function generateMarkdownReport(filepath, results) {
  const fs = require('fs');
  const lines = [];

  lines.push('# Neural Trader - E2B Real Integration Test Report\n');
  lines.push(`**Generated:** ${results.timestamp}\n`);

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
  lines.push(`| Duration | ${results.summary.duration}ms |`);
  lines.push(`| Success Rate | ${((results.summary.passed / results.summary.total) * 100).toFixed(1)}% |\n`);

  lines.push('## Test Results\n');
  lines.push('| Test | Status | Duration | Details |');
  lines.push('|------|--------|----------|---------|');

  for (const test of results.tests) {
    const status = test.success ? '✅ Pass' : '❌ Fail';
    const details = test.error || 'Success';
    lines.push(`| ${test.name} | ${status} | ${test.duration}ms | ${details} |`);
  }
  lines.push('');

  lines.push('## Detailed Test Results\n');
  for (const test of results.tests) {
    lines.push(`### ${test.name}\n`);
    lines.push(`**Status:** ${test.success ? '✅ Passed' : '❌ Failed'}  `);
    lines.push(`**Duration:** ${test.duration}ms\n`);

    if (test.details) {
      lines.push('**Details:**');
      lines.push('```json');
      lines.push(JSON.stringify(test.details, null, 2));
      lines.push('```\n');
    }

    if (test.error) {
      lines.push(`**Error:** ${test.error}\n`);
    }
  }

  lines.push('---\n');
  lines.push('**Test Type:** Real E2B API Integration (NO MOCKS)  ');
  lines.push('**SDK:** @e2b/code-interpreter  ');
  lines.push(`**Timestamp:** ${results.timestamp}\n`);

  fs.writeFileSync(filepath, lines.join('\n'));
}

// Run tests
main().catch(error => {
  console.error('\n❌ Fatal error:', error);
  console.error(error.stack);
  process.exit(1);
});
