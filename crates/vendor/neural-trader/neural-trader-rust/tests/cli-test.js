#!/usr/bin/env node
/**
 * CLI Test Suite
 *
 * Tests the neural-trader CLI functionality
 */

const { execSync } = require('child_process');
const path = require('path');

const CLI_PATH = path.join(__dirname, '..', 'bin', 'cli.js');

console.log('='.repeat(80));
console.log('Neural Trader CLI Test Suite');
console.log('='.repeat(80));

let passedTests = 0;
let failedTests = 0;

function runTest(name, command, expectedPattern) {
  try {
    console.log(`\nTest: ${name}`);
    console.log(`Command: ${command}`);

    const output = execSync(command, {
      encoding: 'utf-8',
      stdio: ['pipe', 'pipe', 'pipe']
    });

    const passed = expectedPattern.test(output);

    if (passed) {
      console.log('✅ PASSED');
      console.log('Output:', output.split('\n').slice(0, 3).join('\n'));
      passedTests++;
    } else {
      console.log('❌ FAILED');
      console.log('Expected pattern:', expectedPattern);
      console.log('Actual output:', output);
      failedTests++;
    }

    return passed;
  } catch (error) {
    console.log('❌ FAILED');
    console.log('Error:', error.message);
    failedTests++;
    return false;
  }
}

// Test 1: --version
runTest(
  'CLI --version',
  `node ${CLI_PATH} --version`,
  /Neural Trader v\d+\.\d+\.\d+/
);

// Test 2: --help
runTest(
  'CLI --help',
  `node ${CLI_PATH} --help`,
  /USAGE:.*COMMANDS:.*OPTIONS:/s
);

// Test 3: version command
runTest(
  'CLI version command',
  `node ${CLI_PATH} version`,
  /Neural Trader v\d+\.\d+\.\d+/
);

// Test 4: help command
runTest(
  'CLI help command',
  `node ${CLI_PATH} help`,
  /Neural Trader.*AI-Powered Trading Platform/
);

// Test 5: init command
runTest(
  'CLI init command (placeholder)',
  `node ${CLI_PATH} init test-project`,
  /Initializing Neural Trader project/
);

// Test 6: npx neural-trader --version
runTest(
  'npx neural-trader --version',
  `npx neural-trader --version`,
  /Neural Trader v\d+\.\d+\.\d+/
);

// Test 7: Unknown command
try {
  console.log('\nTest: Unknown command handling');
  execSync(`node ${CLI_PATH} invalid-command`, {
    encoding: 'utf-8',
    stdio: ['pipe', 'pipe', 'pipe']
  });
  console.log('❌ FAILED - Should have exited with error');
  failedTests++;
} catch (error) {
  if (error.status === 1 && error.stderr.includes('Unknown command')) {
    console.log('✅ PASSED - Correctly rejected unknown command');
    passedTests++;
  } else {
    console.log('❌ FAILED - Wrong error handling');
    failedTests++;
  }
}

// Summary
console.log('\n' + '='.repeat(80));
console.log('Test Summary');
console.log('='.repeat(80));
console.log(`Total Tests: ${passedTests + failedTests}`);
console.log(`Passed: ${passedTests}`);
console.log(`Failed: ${failedTests}`);
console.log(`Success Rate: ${((passedTests / (passedTests + failedTests)) * 100).toFixed(1)}%`);
console.log('='.repeat(80));

process.exit(failedTests > 0 ? 1 : 0);
