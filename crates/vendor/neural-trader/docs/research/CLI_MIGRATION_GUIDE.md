# Neural Trader CLI Migration Guide
## Step-by-Step Refactoring from Manual to Modern Framework

**Version:** 1.0
**Date:** 2025-11-17
**Estimated Timeline:** 4 weeks
**Risk Level:** Low (phased approach with backward compatibility)

---

## Overview

This guide provides a practical, step-by-step plan to modernize the neural-trader CLI from manual implementation to a framework-based architecture while maintaining 100% backward compatibility.

---

## Phase 1: Foundation Setup (Week 1)

### Step 1.1: Install Core Dependencies

```bash
# Core framework
npm install commander

# Already installed (verify)
npm list chalk ora cli-table3

# If missing:
npm install chalk ora cli-table3
```

**Verify Installation:**
```bash
node -e "console.log(require('commander').Command ? '✓ commander' : '✗ commander')"
node -e "console.log(require('chalk') ? '✓ chalk' : '✗ chalk')"
```

---

### Step 1.2: Create New Directory Structure

**Create directories:**

```bash
mkdir -p src/cli/{commands,ui,lib,data}
mkdir -p src/cli/plugins
mkdir -p src/completion
```

**Directory structure:**
```
src/
├── cli/
│   ├── program.js          # Main Commander program
│   ├── commands/           # Command implementations
│   │   ├── init.js
│   │   ├── list.js
│   │   ├── info.js
│   │   ├── install.js
│   │   ├── test.js
│   │   └── doctor.js
│   ├── ui/                 # UI components
│   │   ├── banner.js
│   │   ├── table.js
│   │   └── spinner.js
│   ├── lib/                # Utilities
│   │   ├── config.js
│   │   ├── validator.js
│   │   └── templates.js
│   ├── data/               # Data files
│   │   └── packages.js
│   └── plugins/            # Plugin system
│       └── manager.js
└── completion/
    └── setup.js
```

---

### Step 1.3: Extract Package Registry

**File:** `src/cli/data/packages.js`

Move the PACKAGES object from `bin/cli.js` to a dedicated module:

```javascript
// src/cli/data/packages.js
export const PACKAGES = {
  trading: {
    name: 'Trading Strategy System',
    icon: '💹',
    category: 'trading',
    description: 'Complete algorithmic trading with strategies, execution, and risk management',
    packages: ['neural-trader', '@neural-trader/core', '@neural-trader/strategies'],
    features: [
      'Real-time execution',
      'Multiple strategies (momentum, mean-reversion, pairs)',
      'Risk management',
      'Live market data',
    ],
    templates: ['momentum', 'mean-reversion', 'pairs-trading'],
    hasExamples: true,
  },
  // ... rest of packages (copy from bin/cli.js lines 36-205)
};

// Helper functions
export function getPackagesByCategory(category) {
  return Object.entries(PACKAGES)
    .filter(([, pkg]) => pkg.category === category)
    .reduce((acc, [key, pkg]) => ({ ...acc, [key]: pkg }), {});
}

export function getCategories() {
  return [...new Set(Object.values(PACKAGES).map(p => p.category))];
}

export function searchPackages(query) {
  const q = query.toLowerCase();
  return Object.entries(PACKAGES)
    .filter(([key, pkg]) =>
      key.includes(q) ||
      pkg.name.toLowerCase().includes(q) ||
      pkg.description.toLowerCase().includes(q)
    )
    .reduce((acc, [key, pkg]) => ({ ...acc, [key]: pkg }), {});
}
```

**Test:**
```bash
node -e "import('./src/cli/data/packages.js').then(m => console.log(Object.keys(m.PACKAGES).length, 'packages'))"
```

---

### Step 1.4: Extract UI Components

**File:** `src/cli/ui/banner.js`

```javascript
import chalk from 'chalk';

export function printBanner() {
  console.log();
  console.log(chalk.cyan('╔══════════════════════════════════════════════════════════════╗'));
  console.log(chalk.cyan('║  Neural Trader - High-Performance Trading & Analytics       ║'));
  console.log(chalk.cyan('║  GPU-Accelerated • Real-Time • Self-Learning • 30+ Packages  ║'));
  console.log(chalk.cyan('╚══════════════════════════════════════════════════════════════╝'));
  console.log();
}

export function printSuccess(message) {
  console.log(chalk.green('✓'), message);
}

export function printError(message) {
  console.error(chalk.red('✗'), message);
}

export function printWarning(message) {
  console.warn(chalk.yellow('⚠'), message);
}

export function printInfo(message) {
  console.log(chalk.blue('ℹ'), message);
}
```

**File:** `src/cli/ui/table.js`

```javascript
import Table from 'cli-table3';
import chalk from 'chalk';

export function createPackageTable(packages) {
  const table = new Table({
    head: [
      chalk.cyan.bold('Package'),
      chalk.cyan.bold('Category'),
      chalk.cyan.bold('Description'),
    ],
    colWidths: [30, 15, 55],
    style: {
      head: [],
      border: ['dim'],
    },
    wordWrap: true,
  });

  for (const [key, pkg] of Object.entries(packages)) {
    table.push([
      chalk.green(key),
      chalk.yellow(pkg.category),
      chalk.dim(pkg.description.slice(0, 52) + '...'),
    ]);
  }

  return table.toString();
}

export function createInfoTable(data) {
  const table = new Table({
    colWidths: [20, 60],
    style: { border: ['dim'] },
    wordWrap: true,
  });

  for (const [key, value] of Object.entries(data)) {
    table.push([chalk.cyan(key), value]);
  }

  return table.toString();
}
```

---

### Step 1.5: Create Main Program File

**File:** `src/cli/program.js`

```javascript
import { Command } from 'commander';
import chalk from 'chalk';

export async function run() {
  const program = new Command();

  // Configure program
  program
    .name('neural-trader')
    .description('High-Performance Trading & Analytics')
    .version('2.3.15')
    .option('-d, --debug', 'Enable debug mode')
    .option('--no-color', 'Disable colors')
    .hook('preAction', (thisCommand) => {
      const opts = thisCommand.opts();

      if (opts.debug) {
        process.env.DEBUG = '1';
      }

      if (!opts.color) {
        chalk.level = 0;
      }
    });

  // Register commands (lazy-loaded for performance)
  registerCommands(program);

  // Parse arguments
  try {
    await program.parseAsync(process.argv);
  } catch (err) {
    console.error(chalk.red('Error:'), err.message);

    if (process.env.DEBUG) {
      console.error(err.stack);
    }

    process.exit(1);
  }
}

function registerCommands(program) {
  // Version command (built-in, but we'll enhance it)
  program
    .command('version')
    .description('Show version and system info')
    .action(async () => {
      const { versionCommand } = await import('./commands/version.js');
      await versionCommand();
    });

  // Help command (built-in, but we'll enhance it)
  program
    .command('help')
    .description('Show help information')
    .action(() => {
      program.help();
    });

  // Init command
  program
    .command('init [type]')
    .description('Initialize a new project')
    .option('-t, --template <name>', 'Use specific template')
    .option('--skip-install', 'Skip npm install')
    .option('-i, --interactive', 'Interactive mode')
    .action(async (type, options) => {
      const { initCommand } = await import('./commands/init.js');
      await initCommand(type, options);
    });

  // List command
  program
    .command('list')
    .description('List available packages')
    .option('-c, --category <name>', 'Filter by category')
    .option('-f, --format <type>', 'Output format (table|json)', 'table')
    .action(async (options) => {
      const { listCommand } = await import('./commands/list.js');
      await listCommand(options);
    });

  // Info command
  program
    .command('info <package>')
    .description('Show package details')
    .action(async (packageName) => {
      const { infoCommand } = await import('./commands/info.js');
      await infoCommand(packageName);
    });

  // Install command
  program
    .command('install <package>')
    .description('Install a sub-package')
    .option('--save-dev', 'Install as devDependency')
    .action(async (packageName, options) => {
      const { installCommand } = await import('./commands/install.js');
      await installCommand(packageName, options);
    });

  // Test command
  program
    .command('test')
    .description('Test NAPI bindings and installations')
    .action(async () => {
      const { testCommand } = await import('./commands/test.js');
      await testCommand();
    });

  // Doctor command
  program
    .command('doctor')
    .description('Run health checks')
    .action(async () => {
      const { doctorCommand } = await import('./commands/doctor.js');
      await doctorCommand();
    });
}
```

---

### Step 1.6: Migrate First Command (Version)

**File:** `src/cli/commands/version.js`

```javascript
import chalk from 'chalk';
import { printBanner } from '../ui/banner.js';
import { PACKAGES } from '../data/packages.js';

export async function versionCommand() {
  printBanner();

  console.log(chalk.bold('  Version:'), chalk.cyan('2.3.15'));
  console.log(chalk.bold('  Node:'), chalk.dim(process.version));
  console.log();

  // Check NAPI bindings
  let nt;
  try {
    nt = await import('../../../index.js');
  } catch (err) {
    // NAPI not available
  }

  if (nt) {
    console.log(chalk.green('  ✓ NAPI Bindings:'), chalk.bold('Available'));
    console.log(chalk.green('  ✓ Core Functions:'), chalk.bold(Object.keys(nt).length));
  } else {
    console.log(chalk.yellow('  ⚠ NAPI Bindings:'), chalk.dim('Not loaded (CLI-only mode)'));
  }

  console.log();

  // Package statistics
  const categories = {};
  Object.values(PACKAGES).forEach(pkg => {
    categories[pkg.category] = (categories[pkg.category] || 0) + 1;
  });

  console.log(chalk.bold('  Available Packages:'), chalk.cyan(Object.keys(PACKAGES).length));
  console.log(chalk.bold('  Categories:'));

  Object.entries(categories).forEach(([cat, count]) => {
    console.log(chalk.dim(`    • ${cat}:`), chalk.bold(count), 'packages');
  });

  console.log();
}
```

---

### Step 1.7: Update Entry Point

**File:** `bin/cli.js` (minimal entry point)

```javascript
#!/usr/bin/env node

/**
 * Neural Trader CLI - Entry Point
 * Version: 2.3.15
 */

// Minimal entry point - all logic in src/cli/program.js
import('../src/cli/program.js')
  .then((module) => module.run())
  .catch((err) => {
    console.error('Fatal error:', err.message);

    if (process.env.DEBUG) {
      console.error(err.stack);
    }

    process.exit(1);
  });
```

---

### Step 1.8: Test Phase 1

**Create test script:** `tests/cli/phase1.test.js`

```javascript
import { execa } from 'execa';
import { describe, it, expect } from 'vitest';

describe('Phase 1: Foundation', () => {
  it('should show version', async () => {
    const { stdout } = await execa('node', ['bin/cli.js', 'version']);
    expect(stdout).toContain('Neural Trader');
    expect(stdout).toContain('2.3.15');
  });

  it('should show help', async () => {
    const { stdout } = await execa('node', ['bin/cli.js', '--help']);
    expect(stdout).toContain('Usage:');
    expect(stdout).toContain('neural-trader');
  });

  it('should handle invalid command', async () => {
    try {
      await execa('node', ['bin/cli.js', 'invalid-command']);
    } catch (err) {
      expect(err.exitCode).toBe(1);
    }
  });
});
```

**Run tests:**
```bash
npm test tests/cli/phase1.test.js
```

---

### Step 1.9: Backup and Commit

**Backup current implementation:**
```bash
cp bin/cli.js bin/cli-v2.3.15-backup.js
```

**Git commit:**
```bash
git add .
git commit -m "refactor(cli): Phase 1 - foundation with commander.js

- Add commander.js framework
- Extract package registry to data module
- Create UI components (banner, table)
- Migrate version command
- Update entry point to minimal loader
- Maintain backward compatibility"
```

---

## Phase 2: Command Migration (Week 2)

### Step 2.1: Migrate List Command

**File:** `src/cli/commands/list.js`

```javascript
import chalk from 'chalk';
import { PACKAGES, getPackagesByCategory, getCategories } from '../data/packages.js';
import { createPackageTable } from '../ui/table.js';

export async function listCommand(options) {
  let packages = PACKAGES;

  // Filter by category
  if (options.category) {
    packages = getPackagesByCategory(options.category);

    if (Object.keys(packages).length === 0) {
      console.error(chalk.red(`No packages found in category: ${options.category}`));
      console.log('Available categories:', getCategories().join(', '));
      process.exit(1);
    }
  }

  // Format output
  if (options.format === 'json') {
    console.log(JSON.stringify(packages, null, 2));
    return;
  }

  if (options.format === 'simple') {
    Object.keys(packages).forEach(key => console.log(key));
    return;
  }

  // Table format (default)
  console.log();
  console.log(chalk.cyan.bold('📦 Available Neural Trader Packages'));
  console.log();
  console.log(createPackageTable(packages));
  console.log();
  console.log(chalk.dim('Use'), chalk.cyan('neural-trader info <package>'), chalk.dim('for details'));
  console.log();
}
```

**Test:**
```bash
node bin/cli.js list
node bin/cli.js list --category trading
node bin/cli.js list --format json
```

---

### Step 2.2: Migrate Info Command

**File:** `src/cli/commands/info.js`

```javascript
import chalk from 'chalk';
import { PACKAGES } from '../data/packages.js';

export async function infoCommand(packageName) {
  if (!packageName) {
    console.error(chalk.red('Package name required'));
    console.error('Usage: neural-trader info <package>');
    process.exit(1);
  }

  const pkg = PACKAGES[packageName];

  if (!pkg) {
    console.error(chalk.red(`Unknown package: ${packageName}`));
    console.error('Run', chalk.cyan('neural-trader list'), 'to see all packages');
    process.exit(1);
  }

  // Display package info
  console.log();
  console.log(chalk.cyan.bold(pkg.name));
  console.log(chalk.dim(`Category: ${pkg.category}`));
  console.log();

  console.log(chalk.bold('Description:'));
  console.log(`  ${pkg.description}`);
  console.log();

  if (pkg.features?.length > 0) {
    console.log(chalk.bold('Features:'));
    pkg.features.forEach(f => console.log(chalk.green('  •'), f));
    console.log();
  }

  if (pkg.packages?.length > 0) {
    console.log(chalk.bold('NPM Packages:'));
    pkg.packages.forEach(p => console.log(chalk.dim('  •'), p));
    console.log();
  }

  if (pkg.isExample) {
    console.log(chalk.yellow('⚡ This is an example package'));
    console.log();
  }

  console.log(chalk.bold('Initialize:'));
  console.log(chalk.cyan(`  neural-trader init ${packageName}`));
  console.log();
}
```

---

### Step 2.3: Migrate Init Command (Basic)

**File:** `src/cli/commands/init.js`

```javascript
import fs from 'fs/promises';
import path from 'path';
import chalk from 'chalk';
import ora from 'ora';
import { PACKAGES } from '../data/packages.js';
import { getConfigTemplate, getExampleCode, getReadmeTemplate } from '../lib/templates.js';

export async function initCommand(type, options) {
  if (!type) {
    console.error(chalk.red('Project type required'));
    console.error('Usage: neural-trader init <type>');
    console.error('Run', chalk.cyan('neural-trader list'), 'to see available types');
    process.exit(1);
  }

  type = type.toLowerCase();

  // Handle example: prefix
  if (type.startsWith('example:')) {
    const exampleName = type.split(':')[1];
    return initExample(exampleName, options);
  }

  // Validate package type
  if (!PACKAGES[type]) {
    console.error(chalk.red(`Unknown project type: ${type}`));
    console.error('Run', chalk.cyan('neural-trader list'), 'to see available types');
    process.exit(1);
  }

  console.log();
  console.log(chalk.cyan('🚀 Initializing'), chalk.bold(PACKAGES[type].name), chalk.cyan('project...'));
  console.log();

  // Create directories
  const spinner = ora('Creating directories...').start();

  const dirs = ['src', 'data', 'config'];
  if (type === 'trading') {
    dirs.push('strategies', 'backtest-results');
  } else if (type === 'accounting') {
    dirs.push('reports', 'tax-lots');
  } else if (type === 'predictor') {
    dirs.push('models', 'predictions');
  }

  for (const dir of dirs) {
    await fs.mkdir(dir, { recursive: true });
  }

  spinner.succeed('Directories created');

  // Generate files
  spinner.start('Generating project files...');

  const config = getConfigTemplate(type, options.template);
  await fs.writeFile('config.json', JSON.stringify(config, null, 2));

  const pkgJson = {
    name: `my-${type}-project`,
    version: '1.0.0',
    private: true,
    dependencies: {
      'neural-trader': '^2.3.15',
    },
  };
  await fs.writeFile('package.json', JSON.stringify(pkgJson, null, 2));

  const example = getExampleCode(type, options.template);
  await fs.writeFile('src/main.js', example);

  const readme = getReadmeTemplate(type);
  await fs.writeFile('README.md', readme);

  spinner.succeed('Project files generated');

  // Install dependencies
  if (!options.skipInstall) {
    spinner.start('Installing dependencies...');

    try {
      const { execa } = await import('execa');
      await execa('npm', ['install'], { stdio: 'pipe' });
      spinner.succeed('Dependencies installed');
    } catch (err) {
      spinner.fail('Dependency installation failed');
      console.error(chalk.dim(err.message));
    }
  }

  // Success message
  console.log();
  console.log(chalk.green.bold('✅ Project initialized!'));
  console.log();
  console.log('Next steps:');
  if (options.skipInstall) {
    console.log(chalk.cyan('  1. npm install'));
    console.log(chalk.cyan('  2. Edit config.json'));
    console.log(chalk.cyan('  3. node src/main.js'));
  } else {
    console.log(chalk.cyan('  1. Edit config.json'));
    console.log(chalk.cyan('  2. node src/main.js'));
  }
  console.log();
}

async function initExample(exampleName, options) {
  // Similar implementation for examples
  console.log(chalk.cyan('Initializing example:'), exampleName);
  // ... implementation
}
```

**File:** `src/cli/lib/templates.js`

```javascript
// Move helper functions from bin/cli.js here
export function getConfigTemplate(type, template) {
  // Copy from bin/cli.js lines 608-658
  if (type === 'trading') {
    return {
      trading: {
        provider: "alpaca",
        symbols: ["AAPL", "MSFT", "GOOGL"],
        strategy: template || "momentum",
        parameters: {
          threshold: 0.02,
          lookback: 20,
          stop_loss: 0.05,
        },
      },
      risk: {
        max_position_size: 10000,
        max_portfolio_risk: 0.02,
        stop_loss_pct: 0.05,
      },
    };
  }
  // ... rest of implementations
  return {};
}

export function getExampleCode(type, template) {
  // Copy from bin/cli.js lines 678-746
  if (type === 'trading') {
    return `const nt = require('neural-trader');
const config = require('../config.json');

async function main() {
  console.log('Starting ${template || 'momentum'} trading strategy...');

  const data = await nt.fetchMarketData(
    config.trading.symbols[0],
    '2024-01-01',
    '2024-12-31',
    config.trading.provider
  );

  console.log(\`Fetched \${data.length} data points\`);
}

main().catch(console.error);
`;
  }
  // ... rest of implementations
  return `console.log('Hello from Neural Trader!');`;
}

export function getReadmeTemplate(type) {
  // Copy from bin/cli.js lines 748-774
  const pkg = PACKAGES[type];
  return `# Neural Trader ${pkg?.name || 'Project'}

${pkg?.description || 'High-performance trading and analytics'}

## Getting Started

\`\`\`bash
npm install
node src/main.js
\`\`\`

## Configuration

Edit \`config.json\` to customize your settings.

## Documentation

- [Neural Trader](https://github.com/ruvnet/neural-trader)
`;
}
```

---

### Step 2.4: Test Phase 2

```bash
# Test all commands
node bin/cli.js list
node bin/cli.js list --category trading
node bin/cli.js info trading
node bin/cli.js init trading --skip-install

# Cleanup test
rm -rf my-trading-project
```

---

### Step 2.5: Commit Phase 2

```bash
git add .
git commit -m "refactor(cli): Phase 2 - migrate list, info, init commands

- Migrate list command with filtering
- Migrate info command
- Migrate init command (basic)
- Extract template helpers
- All commands working with commander.js"
```

---

## Phase 3: Enhanced Features (Week 3)

### Step 3.1: Add Validation Library

```bash
npm install zod
```

### Step 3.2: Add Interactive Prompts

```bash
npm install inquirer@latest
```

### Step 3.3: Add Enhanced Init with Inquirer

**Update:** `src/cli/commands/init.js`

Add interactive mode:

```javascript
import inquirer from 'inquirer';

// Add at top of initCommand
if (options.interactive || !type) {
  const answers = await inquirer.prompt([
    {
      type: 'list',
      name: 'projectType',
      message: 'What type of project do you want to create?',
      choices: Object.entries(PACKAGES).map(([key, pkg]) => ({
        name: `${pkg.icon || '📦'} ${pkg.name}`,
        value: key,
      })),
    },
    {
      type: 'input',
      name: 'projectName',
      message: 'Project name:',
      default: (answers) => `my-${answers.projectType}-project`,
      validate: (input) => /^[a-z0-9-]+$/.test(input) || 'Use lowercase and hyphens only',
    },
    {
      type: 'confirm',
      name: 'installDeps',
      message: 'Install dependencies now?',
      default: true,
    },
  ]);

  type = answers.projectType;
  options.skipInstall = !answers.installDeps;
}
```

---

### Step 3.4: Add Configuration Management

```bash
npm install cosmiconfig conf
```

**File:** `src/cli/lib/config.js`

```javascript
import { cosmiconfig } from 'cosmiconfig';
import Conf from 'conf';

const explorer = cosmiconfig('neuraltrader');

const userConfig = new Conf({
  projectName: 'neural-trader',
  schema: {
    defaultProvider: { type: 'string', default: 'alpaca' },
    apiKeys: { type: 'object', default: {} },
    preferences: {
      type: 'object',
      properties: {
        autoInstall: { type: 'boolean', default: true },
        colorOutput: { type: 'boolean', default: true },
      },
    },
  },
});

export async function loadConfig() {
  const result = await explorer.search();
  return result ? result.config : {};
}

export function getUserConfig(key) {
  return userConfig.get(key);
}

export function setUserConfig(key, value) {
  userConfig.set(key, value);
}
```

---

### Step 3.5: Add Config Command

**File:** `src/cli/commands/config.js`

```javascript
import inquirer from 'inquirer';
import chalk from 'chalk';
import { getUserConfig, setUserConfig } from '../lib/config.js';

export async function configCommand(options) {
  if (options.wizard) {
    const answers = await inquirer.prompt([
      {
        type: 'list',
        name: 'provider',
        message: 'Default data provider:',
        choices: ['alpaca', 'binance', 'coinbase', 'polygon'],
        default: getUserConfig('defaultProvider'),
      },
      {
        type: 'confirm',
        name: 'autoInstall',
        message: 'Auto-install dependencies?',
        default: getUserConfig('preferences.autoInstall'),
      },
    ]);

    setUserConfig('defaultProvider', answers.provider);
    setUserConfig('preferences.autoInstall', answers.autoInstall);

    console.log(chalk.green('✓ Configuration saved'));
  } else {
    // Show current config
    console.log(JSON.stringify(getUserConfig(), null, 2));
  }
}
```

**Register in:** `src/cli/program.js`

```javascript
program
  .command('config')
  .description('Manage configuration')
  .option('--wizard', 'Interactive configuration wizard')
  .action(async (options) => {
    const { configCommand } = await import('./commands/config.js');
    await configCommand(options);
  });
```

---

### Step 3.6: Test Phase 3

```bash
node bin/cli.js init --interactive
node bin/cli.js config --wizard
```

---

### Step 3.7: Commit Phase 3

```bash
git add .
git commit -m "refactor(cli): Phase 3 - enhanced features

- Add interactive mode with inquirer
- Add configuration management (cosmiconfig + conf)
- Add config command
- Improve user experience"
```

---

## Phase 4: Extensibility (Week 4)

### Step 4.1: Add Final Dependencies

```bash
npm install tabtab execa listr2 boxen gradient-string
```

### Step 4.2: Implement Plugin System

Already covered in CLI_REFACTORING_EXAMPLES.md

### Step 4.3: Add Auto-completion

Already covered in CLI_REFACTORING_EXAMPLES.md

### Step 4.4: Final Polish

- Add update checker
- Add better error messages
- Add --version command enhancement
- Add search command

---

## Testing Strategy

### Unit Tests

```bash
npm install --save-dev vitest
```

**File:** `tests/cli/commands/list.test.js`

```javascript
import { describe, it, expect } from 'vitest';
import { listCommand } from '../../../src/cli/commands/list.js';

describe('list command', () => {
  it('should list all packages', async () => {
    await listCommand({ format: 'json' });
    // Assertions
  });

  it('should filter by category', async () => {
    await listCommand({ category: 'trading', format: 'json' });
    // Assertions
  });
});
```

### Integration Tests

```javascript
import { execa } from 'execa';

describe('CLI integration', () => {
  it('should execute version command', async () => {
    const { stdout } = await execa('node', ['bin/cli.js', 'version']);
    expect(stdout).toContain('2.3.15');
  });
});
```

---

## Rollback Plan

If issues arise:

1. **Immediate Rollback:**
   ```bash
   git checkout bin/cli-v2.3.15-backup.js bin/cli.js
   ```

2. **Keep Both Versions:**
   ```bash
   # Old CLI
   neural-trader-legacy <command>

   # New CLI
   neural-trader <command>
   ```

3. **Feature Flags:**
   ```bash
   # Use legacy mode
   NEURAL_TRADER_LEGACY=1 neural-trader init trading
   ```

---

## Success Criteria

- ✅ All existing commands work identically
- ✅ 100% backward compatibility
- ✅ All tests pass
- ✅ Performance: startup time < 50ms
- ✅ Documentation updated
- ✅ No breaking changes

---

## Timeline Summary

| Week | Phase | Key Deliverables |
|------|-------|------------------|
| 1 | Foundation | Commander.js integration, basic commands |
| 2 | Migration | All commands migrated, templates extracted |
| 3 | Enhancement | Interactive mode, configuration, validation |
| 4 | Extensibility | Plugins, auto-completion, polish |

---

## Maintenance Checklist

After migration:

- [ ] Update README with new commands
- [ ] Update documentation
- [ ] Create plugin development guide
- [ ] Add CI/CD tests
- [ ] Monitor user feedback
- [ ] Plan for 3.0.0 features

---

## Conclusion

This migration guide provides a safe, incremental path to modernizing the neural-trader CLI. Each phase builds on the previous one while maintaining full backward compatibility. The phased approach allows for early feedback and reduces risk.

**Next Steps:**
1. Review with team
2. Begin Phase 1 implementation
3. Test thoroughly after each phase
4. Gather user feedback
5. Iterate based on learnings

**Questions or Issues:**
- Create GitHub issue
- Tag: `cli-refactoring`
- Reference this migration guide
