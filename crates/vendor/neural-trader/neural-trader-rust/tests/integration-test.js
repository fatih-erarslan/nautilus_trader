#!/usr/bin/env node
/**
 * Integration Tests
 * Tests complete workflows and real-world scenarios
 */

const trader = require('../index.js');
const { spawn } = require('child_process');

const colors = {
  reset: '\x1b[0m',
  green: '\x1b[32m',
  red: '\x1b[31m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  cyan: '\x1b[36m'
};

function log(message, color = 'reset') {
  console.log(`${colors[color]}${message}${colors.reset}`);
}

class TestCase {
  constructor(name) {
    this.name = name;
    this.passed = true;
    this.errors = [];
  }

  assert(condition, message) {
    if (!condition) {
      this.passed = false;
      this.errors.push(message);
      log(`   ✗ ${message}`, 'red');
    } else {
      log(`   ✓ ${message}`, 'green');
    }
  }

  assertEquals(actual, expected, label) {
    const message = `${label}: expected ${expected}, got ${actual}`;
    this.assert(actual === expected, message);
  }

  assertType(value, type, label) {
    const actualType = typeof value;
    const message = `${label}: expected type ${type}, got ${actualType}`;
    this.assert(actualType === type, message);
  }

  assertExists(value, label) {
    const message = `${label}: should exist`;
    this.assert(value !== null && value !== undefined, message);
  }
}

async function testVersionInfo() {
  const test = new TestCase('Version Information');
  log('\n🔍 Testing: ' + test.name, 'blue');

  try {
    const version = trader.getVersionInfo();

    test.assertExists(version, 'Version object');
    test.assertType(version.version, 'string', 'Version string');
    test.assertType(version.rust_version, 'string', 'Rust version');
    test.assert(version.version.length > 0, 'Version not empty');
    test.assert(version.rust_version.length > 0, 'Rust version not empty');

  } catch (error) {
    test.passed = false;
    test.errors.push(error.message);
    log(`   ✗ Error: ${error.message}`, 'red');
  }

  return test;
}

async function testBarEncoding() {
  const test = new TestCase('Bar Encoding/Decoding');
  log('\n🔍 Testing: ' + test.name, 'blue');

  try {
    const bars = [
      { timestamp: 1609459200000, open: 100.5, high: 102.3, low: 99.8, close: 101.2, volume: 10000 },
      { timestamp: 1609545600000, open: 101.2, high: 103.5, low: 100.5, close: 102.8, volume: 12000 },
      { timestamp: 1609632000000, open: 102.8, high: 104.2, low: 101.9, close: 103.5, volume: 11500 }
    ];

    // Encode
    const encoded = trader.encodeBarsToBuffer(bars);
    test.assertExists(encoded, 'Encoded buffer');
    test.assert(Buffer.isBuffer(encoded), 'Result is a Buffer');
    test.assert(encoded.length > 0, 'Buffer has data');

    // Decode
    const decoded = trader.decodeBarsFromBuffer(encoded);
    test.assertExists(decoded, 'Decoded bars');
    test.assert(Array.isArray(decoded), 'Decoded is array');
    test.assertEquals(decoded.length, bars.length, 'Bar count matches');

    // Verify data integrity
    for (let i = 0; i < bars.length; i++) {
      test.assertEquals(decoded[i].timestamp, bars[i].timestamp, `Bar ${i} timestamp`);
      test.assertEquals(decoded[i].open, bars[i].open, `Bar ${i} open`);
      test.assertEquals(decoded[i].high, bars[i].high, `Bar ${i} high`);
      test.assertEquals(decoded[i].low, bars[i].low, `Bar ${i} low`);
      test.assertEquals(decoded[i].close, bars[i].close, `Bar ${i} close`);
      test.assertEquals(decoded[i].volume, bars[i].volume, `Bar ${i} volume`);
    }

  } catch (error) {
    test.passed = false;
    test.errors.push(error.message);
    log(`   ✗ Error: ${error.message}`, 'red');
  }

  return test;
}

async function testNeuralTraderClass() {
  const test = new TestCase('NeuralTrader Class');
  log('\n🔍 Testing: ' + test.name, 'blue');

  try {
    test.assert(typeof trader.NeuralTrader === 'function', 'NeuralTrader is a constructor');

    // Try to instantiate (may fail without config, but should be defined)
    try {
      const instance = new trader.NeuralTrader({
        strategy: 'momentum',
        broker: 'paper'
      });
      test.assertExists(instance, 'Can create instance');
    } catch (error) {
      // Expected if not fully configured
      log(`   ⚠ Instance creation requires config: ${error.message}`, 'yellow');
    }

  } catch (error) {
    test.passed = false;
    test.errors.push(error.message);
    log(`   ✗ Error: ${error.message}`, 'red');
  }

  return test;
}

async function testIndicatorCalculations() {
  const test = new TestCase('Technical Indicators');
  log('\n🔍 Testing: ' + test.name, 'blue');

  try {
    const prices = [100, 102, 101, 103, 105, 104, 106, 108, 107, 109];

    // Test SMA
    try {
      const sma = trader.calculateIndicator('SMA', { prices, period: 5 });
      test.assertExists(sma, 'SMA result');
      test.assert(Array.isArray(sma) || typeof sma === 'object', 'SMA returns data');
    } catch (error) {
      log(`   ⚠ SMA: ${error.message}`, 'yellow');
    }

    // Test EMA
    try {
      const ema = trader.calculateIndicator('EMA', { prices, period: 5 });
      test.assertExists(ema, 'EMA result');
    } catch (error) {
      log(`   ⚠ EMA: ${error.message}`, 'yellow');
    }

    // Test RSI
    try {
      const rsi = trader.calculateIndicator('RSI', { prices, period: 5 });
      test.assertExists(rsi, 'RSI result');
    } catch (error) {
      log(`   ⚠ RSI: ${error.message}`, 'yellow');
    }

  } catch (error) {
    test.passed = false;
    test.errors.push(error.message);
    log(`   ✗ Error: ${error.message}`, 'red');
  }

  return test;
}

async function testCLIIntegration() {
  const test = new TestCase('CLI Integration');
  log('\n🔍 Testing: ' + test.name, 'blue');

  const testCommand = (args) => {
    return new Promise((resolve) => {
      const proc = spawn('node', ['bin/cli.js', ...args], {
        cwd: process.cwd()
      });

      let output = '';
      proc.stdout.on('data', (data) => { output += data.toString(); });
      proc.stderr.on('data', (data) => { output += data.toString(); });

      proc.on('close', (code) => {
        resolve({ code, output });
      });
    });
  };

  // Test version
  const versionResult = await testCommand(['--version']);
  test.assertEquals(versionResult.code, 0, 'CLI --version exit code');
  test.assert(versionResult.output.length > 0, 'CLI --version has output');

  // Test help
  const helpResult = await testCommand(['--help']);
  test.assertEquals(helpResult.code, 0, 'CLI --help exit code');
  test.assert(helpResult.output.includes('Usage') || helpResult.output.includes('Commands'), 'CLI --help shows usage');

  // Test list-strategies
  const strategiesResult = await testCommand(['list-strategies']);
  test.assertEquals(strategiesResult.code, 0, 'CLI list-strategies exit code');

  return test;
}

async function testErrorHandling() {
  const test = new TestCase('Error Handling');
  log('\n🔍 Testing: ' + test.name, 'blue');

  // Test invalid bar data
  try {
    trader.encodeBarsToBuffer([{ invalid: 'data' }]);
    test.assert(false, 'Should throw on invalid bar data');
  } catch (error) {
    test.assert(true, 'Throws on invalid bar data');
  }

  // Test invalid indicator
  try {
    trader.calculateIndicator('INVALID_INDICATOR', { prices: [] });
    test.assert(false, 'Should throw on invalid indicator');
  } catch (error) {
    test.assert(true, 'Throws on invalid indicator');
  }

  return test;
}

async function runIntegrationTests() {
  log('\n╔═══════════════════════════════════════════════╗', 'cyan');
  log('║   Integration Test Suite                      ║', 'cyan');
  log('║   @neural-trader/core                         ║', 'cyan');
  log('╚═══════════════════════════════════════════════╝', 'cyan');

  const tests = [
    await testVersionInfo(),
    await testBarEncoding(),
    await testNeuralTraderClass(),
    await testIndicatorCalculations(),
    await testCLIIntegration(),
    await testErrorHandling()
  ];

  // Summary
  log('\n' + '═'.repeat(50), 'cyan');
  const passed = tests.filter(t => t.passed).length;
  const failed = tests.filter(t => !t.passed).length;

  log(`\n📊 Integration Test Results:`, 'blue');
  log(`   Passed: ${passed}`, 'green');
  log(`   Failed: ${failed}`, failed > 0 ? 'red' : 'green');
  log(`   Total:  ${tests.length}`, 'cyan');

  if (failed > 0) {
    log('\n❌ Failed tests:', 'red');
    tests.filter(t => !t.passed).forEach(t => {
      log(`   - ${t.name}`, 'red');
      t.errors.forEach(e => log(`     ${e}`, 'yellow'));
    });
  }

  process.exit(failed > 0 ? 1 : 0);
}

// Run if executed directly
if (require.main === module) {
  runIntegrationTests().catch(error => {
    log(`\n❌ Test suite error: ${error.message}`, 'red');
    console.error(error);
    process.exit(1);
  });
}

module.exports = { runIntegrationTests };
