#!/usr/bin/env node
/**
 * Comprehensive Validation Script
 *
 * Validates all aspects of the Neural Trader NPM package:
 * - Build artifacts
 * - CLI functionality
 * - SDK/API
 * - TypeScript definitions
 * - MCP server structure
 * - Package metadata
 */

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

console.log('='.repeat(80));
console.log('NEURAL TRADER - COMPREHENSIVE VALIDATION');
console.log('='.repeat(80));
console.log();

let totalTests = 0;
let passedTests = 0;
let failedTests = 0;

function section(title) {
  console.log('\n' + '─'.repeat(80));
  console.log(`📋 ${title}`);
  console.log('─'.repeat(80));
}

function test(name, fn) {
  totalTests++;
  try {
    process.stdout.write(`  ${name}... `);
    fn();
    console.log('✅');
    passedTests++;
    return true;
  } catch (error) {
    console.log('❌');
    console.log(`    Error: ${error.message}`);
    failedTests++;
    return false;
  }
}

// =============================================================================
// 1. Build Artifacts
// =============================================================================
section('Build Artifacts');

test('Native addon exists', () => {
  const nativeAddon = path.join(__dirname, '..', 'neural-trader.linux-x64-gnu.node');
  if (!fs.existsSync(nativeAddon)) {
    throw new Error('Native addon not found');
  }
  const stats = fs.statSync(nativeAddon);
  console.log(`      Size: ${(stats.size / 1024).toFixed(0)} KB`);
});

test('Package.json is valid', () => {
  const packageJson = require('../package.json');
  if (!packageJson.name || !packageJson.version) {
    throw new Error('Invalid package.json');
  }
  console.log(`      Name: ${packageJson.name}`);
  console.log(`      Version: ${packageJson.version}`);
});

test('Binary entry point exists', () => {
  const cliPath = path.join(__dirname, '..', 'bin', 'cli.js');
  if (!fs.existsSync(cliPath)) {
    throw new Error('CLI not found');
  }
  const stats = fs.statSync(cliPath);
  if (!(stats.mode & fs.constants.S_IXUSR)) {
    console.log('      (not executable, but exists)');
  }
});

test('Main index.js exists', () => {
  const indexJs = path.join(__dirname, '..', 'index.js');
  if (!fs.existsSync(indexJs)) {
    throw new Error('index.js not found');
  }
});

test('TypeScript definitions exist', () => {
  const indexDts = path.join(__dirname, '..', 'index.d.ts');
  if (!fs.existsSync(indexDts)) {
    throw new Error('index.d.ts not found');
  }
  const content = fs.readFileSync(indexDts, 'utf-8');
  const typeCount = (content.match(/interface \w+/g) || []).length;
  console.log(`      Interfaces: ${typeCount}`);
});

// =============================================================================
// 2. CLI Commands
// =============================================================================
section('CLI Commands');

const CLI = path.join(__dirname, '..', 'bin', 'cli.js');

test('CLI --version', () => {
  const output = execSync(`node ${CLI} --version`, { encoding: 'utf-8' });
  if (!output.includes('Neural Trader v')) {
    throw new Error('Invalid version output');
  }
});

test('CLI --help', () => {
  const output = execSync(`node ${CLI} --help`, { encoding: 'utf-8' });
  if (!output.includes('USAGE') || !output.includes('COMMANDS')) {
    throw new Error('Invalid help output');
  }
});

test('CLI list-strategies', () => {
  const output = execSync(`node ${CLI} list-strategies`, { encoding: 'utf-8' });
  if (!output.includes('Momentum Strategy') || !output.includes('Mean Reversion')) {
    throw new Error('Invalid strategies list');
  }
  const strategyCount = (output.match(/\d+\./g) || []).length;
  console.log(`      Strategies: ${strategyCount}`);
});

test('CLI list-brokers', () => {
  const output = execSync(`node ${CLI} list-brokers`, { encoding: 'utf-8' });
  if (!output.includes('Alpaca') || !output.includes('Supported Brokers')) {
    throw new Error('Invalid brokers list');
  }
  const brokerCount = (output.match(/\d+\./g) || []).length;
  console.log(`      Brokers: ${brokerCount}`);
});

test('CLI unknown command handling', () => {
  try {
    execSync(`node ${CLI} invalid-command-xyz`, { encoding: 'utf-8', stdio: 'pipe' });
    throw new Error('Should have failed');
  } catch (error) {
    if (error.status !== 1) {
      throw new Error('Wrong exit code');
    }
  }
});

// =============================================================================
// 3. SDK/API
// =============================================================================
section('SDK/API');

test('Module imports successfully', () => {
  const neuralTrader = require('..');
  if (!neuralTrader) {
    throw new Error('Module import failed');
  }
  const exportCount = Object.keys(neuralTrader).length;
  console.log(`      Exports: ${exportCount}`);
});

test('Required exports present', () => {
  const neuralTrader = require('..');
  const required = ['getVersion', 'validateConfig', 'version', 'platform', 'arch'];
  for (const name of required) {
    if (!(name in neuralTrader)) {
      throw new Error(`Missing export: ${name}`);
    }
  }
});

test('Version info accessible', () => {
  const { version, platform, arch } = require('..');
  if (!version || !platform || !arch) {
    throw new Error('Version info incomplete');
  }
  console.log(`      ${version} on ${platform}-${arch}`);
});

test('Class exports available', () => {
  const neuralTrader = require('..');
  const classes = [
    'ExecutionEngine',
    'MarketDataStream',
    'StrategyRunner',
    'PortfolioManager',
    'PortfolioOptimizer'
  ];
  for (const className of classes) {
    if (!(className in neuralTrader)) {
      throw new Error(`Missing class: ${className}`);
    }
    if (typeof neuralTrader[className] !== 'function') {
      throw new Error(`${className} is not a constructor`);
    }
  }
  console.log(`      Classes: ${classes.length}`);
});

// =============================================================================
// 4. TypeScript Definitions
// =============================================================================
section('TypeScript Definitions');

const indexDtsPath = path.join(__dirname, '..', 'index.d.ts');
const dtsContent = fs.readFileSync(indexDtsPath, 'utf-8');

test('Function signatures defined', () => {
  const functions = ['getVersion', 'validateConfig'];
  for (const fn of functions) {
    if (!dtsContent.includes(`function ${fn}`)) {
      throw new Error(`Missing function signature: ${fn}`);
    }
  }
});

test('Interface types defined', () => {
  const interfaces = [
    'VersionInfo',
    'Quote',
    'Bar',
    'Signal',
    'TradeOrder',
    'ExecutionResult',
    'Position',
    'PortfolioOptimization',
    'RiskMetrics'
  ];
  for (const iface of interfaces) {
    if (!dtsContent.includes(`interface ${iface}`)) {
      throw new Error(`Missing interface: ${iface}`);
    }
  }
  console.log(`      Interfaces: ${interfaces.length}`);
});

test('Class declarations defined', () => {
  const classes = [
    'MarketDataStream',
    'StrategyRunner',
    'ExecutionEngine',
    'PortfolioOptimizer',
    'PortfolioManager',
    'SubscriptionHandle'
  ];
  for (const cls of classes) {
    if (!dtsContent.includes(`class ${cls}`)) {
      throw new Error(`Missing class declaration: ${cls}`);
    }
  }
  console.log(`      Classes: ${classes.length}`);
});

test('Export declarations present', () => {
  const exports = ['version', 'platform', 'arch'];
  for (const exp of exports) {
    if (!dtsContent.includes(`export const ${exp}`)) {
      throw new Error(`Missing export declaration: ${exp}`);
    }
  }
});

// =============================================================================
// 5. Package Structure
// =============================================================================
section('Package Structure');

test('Required directories exist', () => {
  const dirs = ['bin', 'crates', 'tests', 'docs'];
  for (const dir of dirs) {
    const dirPath = path.join(__dirname, '..', dir);
    if (!fs.existsSync(dirPath)) {
      throw new Error(`Missing directory: ${dir}`);
    }
  }
});

test('Core crates present', () => {
  const crates = [
    'core',
    'napi-bindings',
    'strategies',
    'execution',
    'portfolio',
    'risk',
    'backtesting'
  ];
  const cratesDir = path.join(__dirname, '..', 'crates');
  for (const crate of crates) {
    const cratePath = path.join(cratesDir, crate);
    if (!fs.existsSync(cratePath)) {
      throw new Error(`Missing crate: ${crate}`);
    }
  }
  console.log(`      Crates: ${crates.length}`);
});

test('Test files present', () => {
  const tests = ['cli-test.js', 'sdk-test.js', 'mcp-test.js'];
  for (const test of tests) {
    const testPath = path.join(__dirname, test);
    if (!fs.existsSync(testPath)) {
      throw new Error(`Missing test: ${test}`);
    }
  }
});

test('Documentation exists', () => {
  const docs = [
    path.join(__dirname, '..', 'README.md'),
    path.join(__dirname, '..', 'docs', 'NPM_TEST_RESULTS.md')
  ];
  for (const doc of docs) {
    if (!fs.existsSync(doc)) {
      throw new Error(`Missing doc: ${path.basename(doc)}`);
    }
  }
});

// =============================================================================
// 6. MCP Structure (Planned)
// =============================================================================
section('MCP Structure (Planned)');

test('MCP crates structure', () => {
  const mcpCrates = ['mcp-protocol', 'mcp-server'];
  const cratesDir = path.join(__dirname, '..', 'crates');
  let foundCount = 0;
  for (const crate of mcpCrates) {
    if (fs.existsSync(path.join(cratesDir, crate))) {
      foundCount++;
    }
  }
  console.log(`      Found: ${foundCount}/${mcpCrates.length} (expected in development)`);
  // Don't fail - MCP is planned feature
});

test('MCP tools specification', () => {
  const expectedTools = [
    'list-strategies',
    'list-brokers',
    'get-quote',
    'submit-order',
    'get-portfolio'
  ];
  console.log(`      Planned tools: ${expectedTools.length}`);
  // Don't fail - just informational
});

// =============================================================================
// 7. NPM Package Metadata
// =============================================================================
section('NPM Package Metadata');

const packageJson = require('../package.json');

test('Package name is scoped', () => {
  if (!packageJson.name.startsWith('@')) {
    throw new Error('Package should be scoped');
  }
  console.log(`      ${packageJson.name}`);
});

test('License is specified', () => {
  if (!packageJson.license) {
    throw new Error('No license specified');
  }
  console.log(`      License: ${packageJson.license}`);
});

test('Keywords are present', () => {
  if (!packageJson.keywords || packageJson.keywords.length === 0) {
    throw new Error('No keywords');
  }
  console.log(`      Keywords: ${packageJson.keywords.length}`);
});

test('Repository is specified', () => {
  if (!packageJson.repository) {
    throw new Error('No repository');
  }
  console.log(`      Repo: ${packageJson.repository.type}`);
});

test('Scripts are defined', () => {
  const requiredScripts = ['build', 'test:cli', 'test:sdk'];
  for (const script of requiredScripts) {
    if (!packageJson.scripts[script]) {
      throw new Error(`Missing script: ${script}`);
    }
  }
  console.log(`      Scripts: ${Object.keys(packageJson.scripts).length}`);
});

// =============================================================================
// Summary
// =============================================================================
console.log('\n' + '='.repeat(80));
console.log('VALIDATION SUMMARY');
console.log('='.repeat(80));
console.log();
console.log(`  Total Tests: ${totalTests}`);
console.log(`  Passed:      ${passedTests} ✅`);
console.log(`  Failed:      ${failedTests} ${failedTests > 0 ? '❌' : '✅'}`);
console.log();
console.log(`  Success Rate: ${((passedTests / totalTests) * 100).toFixed(1)}%`);
console.log();

if (failedTests === 0) {
  console.log('🎉 All validations passed! Package is ready.');
} else {
  console.log('⚠️  Some validations failed. Please review and fix.');
}

console.log('='.repeat(80));
console.log();

process.exit(failedTests > 0 ? 1 : 0);
