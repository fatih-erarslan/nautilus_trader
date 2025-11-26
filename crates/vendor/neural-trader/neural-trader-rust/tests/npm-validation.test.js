#!/usr/bin/env node
/**
 * Comprehensive NPM Package Validation Suite
 * Tests @neural-trader/core package functionality
 */

const { spawn } = require('child_process');
const { existsSync } = require('fs');
const { join } = require('path');

// Test results tracker
const results = {
  passed: 0,
  failed: 0,
  tests: []
};

// Color codes for terminal output
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

function testResult(name, passed, details = '') {
  const status = passed ? '✓' : '✗';
  const color = passed ? 'green' : 'red';
  log(`${status} ${name}`, color);
  if (details) {
    log(`  ${details}`, 'cyan');
  }

  results.tests.push({ name, passed, details });
  if (passed) results.passed++;
  else results.failed++;
}

// Execute command and capture output
function execCommand(command, args = []) {
  return new Promise((resolve, reject) => {
    const proc = spawn(command, args, {
      cwd: process.cwd(),
      shell: true
    });

    let stdout = '';
    let stderr = '';

    proc.stdout.on('data', (data) => {
      stdout += data.toString();
    });

    proc.stderr.on('data', (data) => {
      stderr += data.toString();
    });

    proc.on('close', (code) => {
      resolve({ code, stdout, stderr });
    });

    proc.on('error', (error) => {
      reject(error);
    });
  });
}

// Test 1: Package.json validation
async function testPackageJson() {
  log('\n📦 Testing package.json configuration...', 'blue');

  const pkgPath = join(process.cwd(), 'package.json');
  const pkgExists = existsSync(pkgPath);
  testResult('package.json exists', pkgExists);

  if (!pkgExists) return;

  const pkg = require(pkgPath);

  // Required fields
  testResult('Has name field', !!pkg.name, pkg.name);
  testResult('Has version field', !!pkg.version, pkg.version);
  testResult('Has description', !!pkg.description);
  testResult('Has main entry point', !!pkg.main, pkg.main);
  testResult('Has bin commands', !!pkg.bin, JSON.stringify(pkg.bin));
  testResult('Has repository', !!pkg.repository);
  testResult('Has license', !!pkg.license, pkg.license);

  // Files configuration
  testResult('Has files array', Array.isArray(pkg.files));
  testResult('Files includes index.js', pkg.files?.includes('index.js'));
  testResult('Files includes bin/', pkg.files?.some(f => f.includes('bin')));

  // Scripts
  testResult('Has build script', !!pkg.scripts?.build);
  testResult('Has test scripts', !!pkg.scripts?.test);

  // Keywords for discoverability
  testResult('Has keywords', Array.isArray(pkg.keywords) && pkg.keywords.length > 0);
}

// Test 2: File structure validation
async function testFileStructure() {
  log('\n📁 Testing file structure...', 'blue');

  const requiredFiles = [
    'index.js',
    'index.d.ts',
    'bin/cli.js',
    'package.json',
    'README.md'
  ];

  for (const file of requiredFiles) {
    const path = join(process.cwd(), file);
    testResult(`File exists: ${file}`, existsSync(path));
  }
}

// Test 3: Programmatic API testing
async function testProgrammaticAPI() {
  log('\n🔌 Testing programmatic API...', 'blue');

  try {
    const trader = require('../index.js');

    // Check exports
    testResult('Exports NeuralTrader', typeof trader.NeuralTrader !== 'undefined');
    testResult('Exports getVersionInfo', typeof trader.getVersionInfo === 'function');
    testResult('Exports fetchMarketData', typeof trader.fetchMarketData === 'function');
    testResult('Exports calculateIndicator', typeof trader.calculateIndicator === 'function');
    testResult('Exports encodeBarsToBuffer', typeof trader.encodeBarsToBuffer === 'function');
    testResult('Exports decodeBarsFromBuffer', typeof trader.decodeBarsFromBuffer === 'function');
    testResult('Exports initRuntime', typeof trader.initRuntime === 'function');

    // Test getVersionInfo
    try {
      const version = trader.getVersionInfo();
      testResult('getVersionInfo() returns object', typeof version === 'object');
      testResult('Version has version field', typeof version.version === 'string', version.version);
      testResult('Version has rust_version field', typeof version.rust_version === 'string', version.rust_version);
    } catch (error) {
      testResult('getVersionInfo() works', false, error.message);
    }

  } catch (error) {
    testResult('Can require index.js', false, error.message);
  }
}

// Test 4: CLI commands
async function testCLI() {
  log('\n⌨️  Testing CLI commands...', 'blue');

  const cliPath = './bin/cli.js';

  // Test --version
  try {
    const { code, stdout } = await execCommand('node', [cliPath, '--version']);
    testResult('CLI --version works', code === 0, stdout.trim());
  } catch (error) {
    testResult('CLI --version works', false, error.message);
  }

  // Test --help
  try {
    const { code, stdout } = await execCommand('node', [cliPath, '--help']);
    const hasCommands = stdout.includes('Commands:') || stdout.includes('Usage:');
    testResult('CLI --help works', code === 0 && hasCommands);
  } catch (error) {
    testResult('CLI --help works', false, error.message);
  }

  // Test list-strategies
  try {
    const { code, stdout } = await execCommand('node', [cliPath, 'list-strategies']);
    const hasStrategies = stdout.includes('momentum') ||
                         stdout.includes('mean-reversion') ||
                         stdout.includes('pairs');
    testResult('CLI list-strategies works', code === 0 && hasStrategies);
  } catch (error) {
    testResult('CLI list-strategies works', false, error.message);
  }

  // Test list-brokers
  try {
    const { code, stdout } = await execCommand('node', [cliPath, 'list-brokers']);
    const hasBrokers = stdout.includes('alpaca') ||
                      stdout.includes('ibkr') ||
                      stdout.includes('paper');
    testResult('CLI list-brokers works', code === 0 && hasBrokers);
  } catch (error) {
    testResult('CLI list-brokers works', false, error.message);
  }
}

// Test 5: Native binding
async function testNativeBinding() {
  log('\n🦀 Testing native Rust bindings...', 'blue');

  const { platform, arch } = process;
  log(`  Platform: ${platform} ${arch}`, 'cyan');

  // Expected .node file based on platform
  let expectedBinding = '';
  if (platform === 'linux' && arch === 'x64') {
    expectedBinding = 'neural-trader.linux-x64-gnu.node';
  } else if (platform === 'darwin' && arch === 'x64') {
    expectedBinding = 'neural-trader.darwin-x64.node';
  } else if (platform === 'darwin' && arch === 'arm64') {
    expectedBinding = 'neural-trader.darwin-arm64.node';
  } else if (platform === 'win32' && arch === 'x64') {
    expectedBinding = 'neural-trader.win32-x64-msvc.node';
  }

  if (expectedBinding) {
    const bindingPath = join(process.cwd(), expectedBinding);
    testResult(`Native binding exists: ${expectedBinding}`, existsSync(bindingPath));
  }
}

// Test 6: TypeScript definitions
async function testTypeScriptDefs() {
  log('\n📘 Testing TypeScript definitions...', 'blue');

  const dtsPath = join(process.cwd(), 'index.d.ts');
  const dtsExists = existsSync(dtsPath);
  testResult('index.d.ts exists', dtsExists);

  if (dtsExists) {
    const fs = require('fs');
    const content = fs.readFileSync(dtsPath, 'utf8');

    testResult('Declares NeuralTrader class', content.includes('export class NeuralTrader'));
    testResult('Declares getVersionInfo', content.includes('getVersionInfo'));
    testResult('Declares fetchMarketData', content.includes('fetchMarketData'));
    testResult('Has TypeScript export syntax', content.includes('export'));
  }
}

// Test 7: Dependencies
async function testDependencies() {
  log('\n📚 Testing dependencies...', 'blue');

  try {
    const { code } = await execCommand('npm', ['list', '--depth=0']);
    testResult('All dependencies installed', code === 0);
  } catch (error) {
    testResult('All dependencies installed', false, error.message);
  }
}

// Main test runner
async function runTests() {
  log('\n╔═══════════════════════════════════════════════╗', 'cyan');
  log('║   NPM Package Validation Suite               ║', 'cyan');
  log('║   @neural-trader/core                         ║', 'cyan');
  log('╚═══════════════════════════════════════════════╝', 'cyan');

  await testPackageJson();
  await testFileStructure();
  await testProgrammaticAPI();
  await testCLI();
  await testNativeBinding();
  await testTypeScriptDefs();
  await testDependencies();

  // Summary
  log('\n' + '═'.repeat(50), 'cyan');
  log(`\n📊 Test Results Summary:`, 'blue');
  log(`   Passed: ${results.passed}`, 'green');
  log(`   Failed: ${results.failed}`, results.failed > 0 ? 'red' : 'green');
  log(`   Total:  ${results.passed + results.failed}`, 'cyan');

  const passRate = ((results.passed / (results.passed + results.failed)) * 100).toFixed(1);
  log(`   Success Rate: ${passRate}%`, passRate >= 90 ? 'green' : 'yellow');

  // Exit with appropriate code
  process.exit(results.failed > 0 ? 1 : 0);
}

// Run if executed directly
if (require.main === module) {
  runTests().catch(error => {
    log(`\n❌ Test suite error: ${error.message}`, 'red');
    console.error(error);
    process.exit(1);
  });
}

module.exports = { runTests, results };
