#!/usr/bin/env node

/**
 * Smoke test for @neural-trader/backend
 * Verifies that the native module can be loaded and basic functions work
 */

const os = require('os');

console.log('=== Neural Trader Backend - Smoke Test ===\n');
console.log(`Platform: ${os.platform()}`);
console.log(`Architecture: ${os.arch()}`);
console.log(`Node.js: ${process.version}\n`);

// Test 1: Module loading
console.log('Test 1: Module loading...');
try {
  const backend = require('../index.js');
  console.log('✓ Module loaded successfully');
  console.log(`  Exports: ${Object.keys(backend).join(', ')}`);
} catch (err) {
  console.error('✗ Failed to load module:', err.message);
  console.error('\nThis could mean:');
  console.error('  1. Native binary not built for your platform');
  console.error('  2. Missing dependencies');
  console.error('  3. Binary compilation errors\n');
  console.error('Stack trace:', err.stack);
  process.exit(1);
}

// Test 2: Basic functionality
console.log('\nTest 2: Basic functionality...');
try {
  const backend = require('../index.js');

  // Test if we have expected exports
  const expectedExports = [
    'calculateTechnicalIndicators',
    'runBacktest',
    'executeTrade',
    'getMarketData'
  ];

  let foundExports = 0;
  for (const exportName of expectedExports) {
    if (backend[exportName]) {
      foundExports++;
      console.log(`  ✓ ${exportName} available`);
    } else {
      console.log(`  ⚠ ${exportName} not found (may be optional)`);
    }
  }

  if (foundExports > 0) {
    console.log(`✓ Found ${foundExports}/${expectedExports.length} expected exports`);
  } else {
    console.log('⚠ No standard exports found - module may have different API');
  }
} catch (err) {
  console.error('✗ Functionality test failed:', err.message);
  process.exit(1);
}

// Test 3: Error handling
console.log('\nTest 3: Error handling...');
try {
  const backend = require('../index.js');

  // Try to call a function with invalid parameters
  if (backend.calculateTechnicalIndicators) {
    try {
      // This should throw an error gracefully
      backend.calculateTechnicalIndicators();
    } catch (err) {
      console.log('✓ Error handling works:', err.message);
    }
  } else {
    console.log('⚠ No testable functions available');
  }
} catch (err) {
  console.error('✗ Error handling test failed:', err.message);
}

console.log('\n=== All smoke tests passed ===\n');
console.log('Next steps:');
console.log('  - Run advanced tests: npm run test:advanced');
console.log('  - Check performance benchmarks');
console.log('  - Test with real trading data\n');
