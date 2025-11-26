/**
 * Test Interactive CLI Mode
 * Validates REPL, configuration, and command functionality
 */

const fs = require('fs').promises;
const path = require('path');
const os = require('os');

// Import modules to test
const ConfigManager = require('../../src/cli/lib/config-manager');
const HistoryManager = require('../../src/cli/lib/history-manager');
const AutoComplete = require('../../src/cli/lib/auto-complete');
const { validateConfig, defaultConfig } = require('../../src/cli/lib/config-schema');

const chalk = require('chalk');

// Test results
const results = {
  passed: 0,
  failed: 0,
  tests: []
};

/**
 * Test helper
 */
function test(name, fn) {
  return async () => {
    try {
      await fn();
      results.passed++;
      results.tests.push({ name, status: 'passed' });
      console.log(chalk.green('✓'), name);
    } catch (error) {
      results.failed++;
      results.tests.push({ name, status: 'failed', error: error.message });
      console.log(chalk.red('✗'), name);
      console.log(chalk.red('  Error:'), error.message);
      if (process.env.DEBUG) {
        console.log(error.stack);
      }
    }
  };
}

/**
 * Run all tests
 */
async function runTests() {
  console.log(chalk.cyan.bold('\n🧪 Testing Interactive CLI Mode\n'));

  // Create temporary test directory
  const testDir = path.join(os.tmpdir(), 'neural-trader-test-' + Date.now());
  await fs.mkdir(testDir, { recursive: true });
  process.chdir(testDir);

  console.log(chalk.dim(`Test directory: ${testDir}\n`));

  // Test 1: ConfigManager initialization
  await test('ConfigManager initialization', async () => {
    const manager = new ConfigManager();
    if (!manager) throw new Error('Failed to create ConfigManager');
  })();

  // Test 2: Default configuration validation
  await test('Default configuration validation', async () => {
    const validated = validateConfig(defaultConfig);
    if (!validated) throw new Error('Default config validation failed');
  })();

  // Test 3: Configuration schema validation
  await test('Configuration schema validation', async () => {
    const testConfig = {
      ...defaultConfig,
      trading: {
        ...defaultConfig.trading,
        symbols: ['AAPL']
      }
    };
    const validated = validateConfig(testConfig);
    if (!validated.trading.symbols[0] === 'AAPL') {
      throw new Error('Config validation failed');
    }
  })();

  // Test 4: ConfigManager save and load
  await test('ConfigManager save and load', async () => {
    const manager = new ConfigManager();
    const testConfig = { ...defaultConfig };

    const savedPath = await manager.saveProjectConfig(testConfig);
    if (!savedPath) throw new Error('Failed to save config');

    const { config } = await manager.loadProjectConfig();
    if (!config) throw new Error('Failed to load config');
    if (config.name !== testConfig.name) throw new Error('Config mismatch');
  })();

  // Test 5: ConfigManager get/set
  await test('ConfigManager get and set', async () => {
    const manager = new ConfigManager();
    await manager.loadProjectConfig();

    await manager.set('trading.risk.maxPositionSize', 25000);
    const value = manager.get('trading.risk.maxPositionSize');

    if (value !== 25000) {
      throw new Error(`Expected 25000, got ${value}`);
    }
  })();

  // Test 6: ConfigManager export
  await test('ConfigManager export', async () => {
    const manager = new ConfigManager();
    await manager.loadProjectConfig();

    const exportPath = path.join(testDir, 'exported-config.json');
    await manager.exportConfig(exportPath);

    const exists = await fs.access(exportPath).then(() => true).catch(() => false);
    if (!exists) throw new Error('Export file not created');
  })();

  // Test 7: ConfigManager import
  await test('ConfigManager import', async () => {
    const manager = new ConfigManager();

    const importPath = path.join(testDir, 'exported-config.json');
    const imported = await manager.importConfig(importPath);

    if (!imported) throw new Error('Import failed');
  })();

  // Test 8: HistoryManager initialization
  await test('HistoryManager initialization', async () => {
    const historyFile = path.join(testDir, '.test-history');
    const manager = new HistoryManager({ historyFile });

    await manager.load();
    if (manager.size() !== 0) {
      throw new Error('History should be empty initially');
    }
  })();

  // Test 9: HistoryManager add and retrieve
  await test('HistoryManager add and retrieve', async () => {
    const historyFile = path.join(testDir, '.test-history');
    const manager = new HistoryManager({ historyFile });

    manager.add('test command 1');
    manager.add('test command 2');

    if (manager.size() !== 2) {
      throw new Error(`Expected 2 entries, got ${manager.size()}`);
    }

    const recent = manager.getRecent(2);
    if (recent[0] !== 'test command 2') {
      throw new Error('Most recent command mismatch');
    }
  })();

  // Test 10: HistoryManager persistence
  await test('HistoryManager persistence', async () => {
    const historyFile = path.join(testDir, '.test-history');

    // Save history
    const manager1 = new HistoryManager({ historyFile });
    manager1.add('persistent command');
    await manager1.save();

    // Load in new instance
    const manager2 = new HistoryManager({ historyFile });
    await manager2.load();

    if (manager2.size() === 0) {
      throw new Error('History not persisted');
    }

    const all = manager2.getAll();
    if (!all.includes('persistent command')) {
      throw new Error('Saved command not found');
    }
  })();

  // Test 11: HistoryManager search
  await test('HistoryManager search', async () => {
    const historyFile = path.join(testDir, '.test-history');
    const manager = new HistoryManager({ historyFile });

    manager.add('config get test');
    manager.add('version');
    manager.add('config set key value');

    const results = manager.search('config');

    if (results.length !== 2) {
      throw new Error(`Expected 2 results, got ${results.length}`);
    }
  })();

  // Test 12: AutoComplete initialization
  await test('AutoComplete initialization', async () => {
    const ac = new AutoComplete();
    ac.registerCommand('test', {
      description: 'Test command',
      options: ['--flag']
    });

    const suggestions = ac.getSuggestions('test');
    if (!suggestions || suggestions.command !== 'test') {
      throw new Error('Command registration failed');
    }
  })();

  // Test 13: AutoComplete command completion
  await test('AutoComplete command completion', async () => {
    const ac = AutoComplete.createDefaultAutoComplete();

    const [completions] = ac.complete('ver');

    if (!completions.includes('version')) {
      throw new Error('Command completion failed');
    }
  })();

  // Test 14: Invalid configuration rejection
  await test('Invalid configuration rejection', async () => {
    try {
      validateConfig({
        name: 'test',
        version: 'invalid-version', // Should fail
        trading: {}
      });
      throw new Error('Should have thrown validation error');
    } catch (error) {
      if (!error.message.includes('validation')) {
        throw new Error('Wrong error type');
      }
    }
  })();

  // Test 15: Configuration reset
  await test('Configuration reset', async () => {
    const manager = new ConfigManager();
    await manager.loadProjectConfig();

    // Change a value
    await manager.set('trading.risk.maxPositionSize', 99999);

    // Reset
    await manager.resetProjectConfig();

    // Verify reset
    const value = manager.get('trading.risk.maxPositionSize');
    if (value === 99999) {
      throw new Error('Configuration not reset');
    }
  })();

  // Print results
  console.log(chalk.cyan.bold('\n📊 Test Results\n'));
  console.log(chalk.white(`Total:  ${results.passed + results.failed}`));
  console.log(chalk.green(`Passed: ${results.passed}`));
  console.log(chalk.red(`Failed: ${results.failed}`));
  console.log();

  if (results.failed > 0) {
    console.log(chalk.red.bold('❌ Some tests failed\n'));
    process.exit(1);
  } else {
    console.log(chalk.green.bold('✅ All tests passed!\n'));
  }

  // Cleanup
  try {
    await fs.rm(testDir, { recursive: true, force: true });
  } catch (error) {
    console.log(chalk.yellow(`Warning: Could not cleanup test directory: ${testDir}`));
  }
}

// Run tests
if (require.main === module) {
  runTests().catch(error => {
    console.error(chalk.red('Fatal error:'), error);
    process.exit(1);
  });
}

module.exports = runTests;
