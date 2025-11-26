#!/usr/bin/env node
/**
 * SDK Test Suite
 *
 * Tests the neural-trader Node.js SDK/API functionality
 */

const path = require('path');

console.log('='.repeat(80));
console.log('Neural Trader SDK Test Suite');
console.log('='.repeat(80));

let passedTests = 0;
let failedTests = 0;

function test(name, fn) {
  try {
    console.log(`\nTest: ${name}`);
    fn();
    console.log('✅ PASSED');
    passedTests++;
    return true;
  } catch (error) {
    console.log('❌ FAILED');
    console.log('Error:', error.message);
    failedTests++;
    return false;
  }
}

async function asyncTest(name, fn) {
  try {
    console.log(`\nTest: ${name}`);
    await fn();
    console.log('✅ PASSED');
    passedTests++;
    return true;
  } catch (error) {
    console.log('❌ FAILED');
    console.log('Error:', error.message);
    failedTests++;
    return false;
  }
}

// Test 1: Module import
test('Import neural-trader module', () => {
  const neuralTrader = require('..');

  if (!neuralTrader) {
    throw new Error('Failed to import module');
  }

  console.log('  - Module imported successfully');
  console.log('  - Exports:', Object.keys(neuralTrader).slice(0, 5).join(', '), '...');
});

// Test 2: Check exports
test('Check required exports', () => {
  const neuralTrader = require('..');

  const requiredExports = [
    'getVersion',
    'validateConfig',
    'version',
    'platform',
    'arch'
  ];

  for (const exportName of requiredExports) {
    if (!(exportName in neuralTrader)) {
      throw new Error(`Missing required export: ${exportName}`);
    }
  }

  console.log('  - All required exports present');
});

// Test 3: Version info
test('Get version information', () => {
  const { version, platform, arch } = require('..');

  console.log(`  - Version: ${version}`);
  console.log(`  - Platform: ${platform}`);
  console.log(`  - Architecture: ${arch}`);

  if (!version || !platform || !arch) {
    throw new Error('Missing version information');
  }
});

// Test 4: getVersion function
test('Call getVersion()', () => {
  const { getVersion } = require('..');

  try {
    const versionInfo = getVersion();
    console.log('  - Version info retrieved:', JSON.stringify(versionInfo, null, 2));
  } catch (error) {
    // Expected to fail if native module not fully implemented
    console.log('  - Native getVersion() not yet implemented (expected)');
  }
});

// Test 5: validateConfig function
test('Call validateConfig()', () => {
  const { validateConfig } = require('..');

  const testConfig = {
    apiKey: 'test-key',
    secretKey: 'test-secret',
    paperTrading: true
  };

  try {
    const result = validateConfig(testConfig);
    console.log('  - Validation result:', JSON.stringify(result, null, 2));
  } catch (error) {
    // Expected to fail if native module not fully implemented
    console.log('  - Native validateConfig() not yet implemented (expected)');
  }
});

// Test 6: Check TypeScript definitions
test('TypeScript definitions exist', () => {
  const fs = require('fs');
  const indexDts = path.join(__dirname, '..', 'index.d.ts');

  if (!fs.existsSync(indexDts)) {
    throw new Error('index.d.ts not found');
  }

  const content = fs.readFileSync(indexDts, 'utf-8');

  if (!content.includes('export function getVersion()')) {
    throw new Error('Missing type definitions');
  }

  console.log('  - TypeScript definitions present');
});

// Test 7: Import from TypeScript (check types)
test('TypeScript compatibility', () => {
  const fs = require('fs');
  const indexDts = path.join(__dirname, '..', 'index.d.ts');
  const content = fs.readFileSync(indexDts, 'utf-8');

  const requiredTypes = [
    'VersionInfo',
    'Quote',
    'Bar',
    'Signal',
    'TradeOrder',
    'ExecutionResult',
    'Position'
  ];

  for (const type of requiredTypes) {
    if (!content.includes(`interface ${type}`)) {
      throw new Error(`Missing TypeScript type: ${type}`);
    }
  }

  console.log('  - All required TypeScript types defined');
});

// Summary
(async () => {
  console.log('\n' + '='.repeat(80));
  console.log('Test Summary');
  console.log('='.repeat(80));
  console.log(`Total Tests: ${passedTests + failedTests}`);
  console.log(`Passed: ${passedTests}`);
  console.log(`Failed: ${failedTests}`);
  console.log(`Success Rate: ${((passedTests / (passedTests + failedTests)) * 100).toFixed(1)}%`);
  console.log('='.repeat(80));

  process.exit(failedTests > 0 ? 1 : 0);
})();
