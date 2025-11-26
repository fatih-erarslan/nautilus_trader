# CLI Refactoring Code Examples
## Practical Implementation Guide for Neural Trader CLI

---

## Example 1: Current vs. Refactored Init Command

### Current Implementation (Manual)

**File:** `bin/cli.js` (lines 289-351)

```javascript
// ❌ Current: Mixed concerns, manual prompts, 62 lines
init: async (type = 'trading', template) => {
  type = type.toLowerCase();

  if (type.startsWith('example:')) {
    const exampleName = type.split(':')[1];
    return commands.initExample(exampleName);
  }

  console.log(`🚀 Initializing ${PACKAGES[type]?.name || 'Trading'} project...`);
  console.log('');

  const dirs = ['src', 'data', 'config'];
  if (type === 'trading') {
    dirs.push('strategies', 'backtest-results');
  } else if (type === 'accounting') {
    dirs.push('reports', 'tax-lots');
  } else if (type === 'predictor') {
    dirs.push('models', 'predictions');
  }

  dirs.forEach(dir => {
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
      console.log(`✓ Created ${dir}/`);
    }
  });

  const config = getConfigTemplate(type, template);
  fs.writeFileSync('config.json', JSON.stringify(config, null, 2));
  console.log('✓ Created config.json');

  // ... more file operations

  console.log('');
  console.log('✅ Project initialized!');
  console.log('');
  console.log('Next steps:');
  console.log('  1. npm install');
  console.log('  2. Edit config.json with your settings');
  console.log('  3. node src/main.js');
  console.log('');
}
```

### Refactored Implementation (Modern)

**File:** `src/cli/commands/init.js`

```javascript
// ✅ Refactored: Separated concerns, type-safe, interactive
import inquirer from 'inquirer';
import ora from 'ora';
import chalk from 'chalk';
import { Listr } from 'listr2';
import { execa } from 'execa';
import fs from 'fs/promises';
import path from 'path';
import { z } from 'zod';
import { PACKAGES } from '../data/packages.js';
import { ProjectGenerator } from '../lib/project-generator.js';
import { getUserConfig } from '../lib/config.js';

// Input validation schema
const InitOptionsSchema = z.object({
  type: z.string(),
  template: z.string().optional(),
  skipInstall: z.boolean().default(false),
  interactive: z.boolean().default(false),
  projectName: z.string().optional(),
});

export async function initCommand(type, options) {
  // Validate options
  const validated = InitOptionsSchema.parse({ type, ...options });

  // Interactive mode
  if (validated.interactive || !validated.type) {
    const interactive = await runInteractiveSetup(validated.type);
    Object.assign(validated, interactive);
  }

  // Validate package type
  const packageInfo = PACKAGES[validated.type];
  if (!packageInfo) {
    throw new Error(
      `Unknown package type: ${validated.type}\n` +
      `Run ${chalk.cyan('neural-trader list')} to see available types`
    );
  }

  // Generate project
  const generator = new ProjectGenerator(packageInfo, validated);

  const tasks = new Listr([
    {
      title: 'Validating project name',
      task: async () => await generator.validateProjectName(),
    },
    {
      title: 'Creating directory structure',
      task: async () => await generator.createDirectories(),
    },
    {
      title: 'Generating configuration',
      task: async () => await generator.generateConfig(),
    },
    {
      title: 'Creating source files',
      task: async () => await generator.generateSourceFiles(),
    },
    {
      title: 'Writing README',
      task: async () => await generator.generateReadme(),
    },
    {
      title: 'Installing dependencies',
      enabled: () => !validated.skipInstall,
      task: async (ctx, task) => {
        task.output = 'Running npm install...';
        await generator.installDependencies();
      },
    },
  ]);

  try {
    await tasks.run();

    // Success message
    console.log();
    console.log(chalk.green.bold('✅ Project created successfully!'));
    console.log();
    console.log('Next steps:');
    console.log(chalk.cyan(`  cd ${generator.projectName}`));
    if (validated.skipInstall) {
      console.log(chalk.cyan('  npm install'));
    }
    console.log(chalk.cyan('  npm start'));
    console.log();
  } catch (err) {
    console.error(chalk.red('Failed to create project:'), err.message);
    throw err;
  }
}

async function runInteractiveSetup(initialType) {
  const categories = groupPackagesByCategory();

  const answers = await inquirer.prompt([
    {
      type: 'list',
      name: 'category',
      message: 'What type of project?',
      choices: [
        { name: '💹 Trading & Execution', value: 'trading' },
        { name: '⚽ Sports & Betting', value: 'betting' },
        { name: '🧮 Accounting & Tax', value: 'accounting' },
        { name: '🔮 Prediction & Forecasting', value: 'prediction' },
        { name: '🧪 Example Projects', value: 'example' },
      ],
      default: initialType ? getPackageCategory(initialType) : 'trading',
    },
    {
      type: 'list',
      name: 'type',
      message: 'Choose a package:',
      choices: (answers) => {
        return categories[answers.category].map(([key, pkg]) => ({
          name: `${pkg.icon || '📦'} ${pkg.name}`,
          value: key,
          short: key,
        }));
      },
    },
    {
      type: 'input',
      name: 'projectName',
      message: 'Project name:',
      default: (answers) => `my-${answers.type}-project`,
      validate: (input) => {
        if (!/^[a-z0-9-]+$/.test(input)) {
          return 'Project name must be lowercase with hyphens only';
        }
        if (input.length < 3) {
          return 'Project name must be at least 3 characters';
        }
        return true;
      },
      filter: (input) => input.toLowerCase().trim(),
    },
    {
      type: 'list',
      name: 'template',
      message: 'Choose a template:',
      when: (answers) => PACKAGES[answers.type]?.templates?.length > 0,
      choices: (answers) => {
        const templates = PACKAGES[answers.type].templates;
        return [
          { name: 'Default', value: null },
          ...templates.map(t => ({ name: t, value: t })),
        ];
      },
    },
    {
      type: 'confirm',
      name: 'installDependencies',
      message: 'Install dependencies now?',
      default: getUserConfig('preferences.autoInstall') ?? true,
    },
  ]);

  return {
    type: answers.type,
    projectName: answers.projectName,
    template: answers.template,
    skipInstall: !answers.installDependencies,
  };
}

function groupPackagesByCategory() {
  const groups = {};

  for (const [key, pkg] of Object.entries(PACKAGES)) {
    const category = pkg.category;
    if (!groups[category]) {
      groups[category] = [];
    }
    groups[category].push([key, pkg]);
  }

  return groups;
}
```

**File:** `src/cli/lib/project-generator.js`

```javascript
// ProjectGenerator class: Encapsulates project creation logic
import fs from 'fs/promises';
import path from 'path';
import { execa } from 'execa';
import { getTemplate } from './templates.js';

export class ProjectGenerator {
  constructor(packageInfo, options) {
    this.packageInfo = packageInfo;
    this.options = options;
    this.projectName = options.projectName || `my-${options.type}-project`;
    this.projectDir = path.join(process.cwd(), this.projectName);
  }

  async validateProjectName() {
    try {
      await fs.access(this.projectDir);
      throw new Error(`Directory already exists: ${this.projectName}`);
    } catch (err) {
      if (err.code !== 'ENOENT') throw err;
      // Directory doesn't exist - good!
    }
  }

  async createDirectories() {
    const dirs = this.getDirectoryStructure();

    for (const dir of dirs) {
      await fs.mkdir(path.join(this.projectDir, dir), { recursive: true });
    }
  }

  getDirectoryStructure() {
    const base = ['src', 'data', 'config'];

    // Type-specific directories
    const typeSpecific = {
      trading: ['strategies', 'backtest-results'],
      accounting: ['reports', 'tax-lots'],
      predictor: ['models', 'predictions'],
      backtesting: ['results', 'reports'],
    };

    return [...base, ...(typeSpecific[this.options.type] || [])];
  }

  async generateConfig() {
    const template = await getTemplate(this.options.type, this.options.template);

    await fs.writeFile(
      path.join(this.projectDir, 'config.json'),
      JSON.stringify(template.config, null, 2)
    );
  }

  async generateSourceFiles() {
    const template = await getTemplate(this.options.type, this.options.template);

    // Main entry point
    await fs.writeFile(
      path.join(this.projectDir, 'src', 'main.js'),
      template.mainCode
    );

    // Additional files based on template
    if (template.additionalFiles) {
      for (const [filename, content] of Object.entries(template.additionalFiles)) {
        await fs.writeFile(
          path.join(this.projectDir, filename),
          content
        );
      }
    }

    // package.json
    await fs.writeFile(
      path.join(this.projectDir, 'package.json'),
      JSON.stringify(template.packageJson, null, 2)
    );
  }

  async generateReadme() {
    const template = await getTemplate(this.options.type, this.options.template);

    await fs.writeFile(
      path.join(this.projectDir, 'README.md'),
      template.readme
    );
  }

  async installDependencies() {
    await execa('npm', ['install'], {
      cwd: this.projectDir,
      stdio: 'pipe',
    });
  }
}
```

---

## Example 2: List Command Comparison

### Current Implementation

```javascript
// ❌ Current: Basic, no filtering, hardcoded format
list: () => {
  console.log('');
  console.log('📦 Available Neural Trader Packages:');
  console.log('');

  Object.entries(PACKAGES).forEach(([key, pkg]) => {
    console.log(`  ${key.padEnd(15)} ${pkg.name}`);
    console.log(`  ${' '.repeat(15)} ${pkg.description}`);
    if (pkg.packages) {
      console.log(`  ${' '.repeat(15)} Packages: ${pkg.packages.join(', ')}`);
    }
    console.log('');
  });

  console.log('Use "neural-trader init <type>" to create a project');
  console.log('');
}
```

### Refactored Implementation

```javascript
// ✅ Refactored: Filtered, formatted, sortable
import chalk from 'chalk';
import Table from 'cli-table3';
import { PACKAGES, getPackagesByCategory, getCategories } from '../data/packages.js';

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
  displayPackagesTable(packages, options);
}

function displayPackagesTable(packages, options) {
  console.log();
  console.log(chalk.cyan.bold('📦 Available Neural Trader Packages'));
  console.log();

  // Group by category
  const grouped = {};
  for (const [key, pkg] of Object.entries(packages)) {
    const cat = pkg.category;
    if (!grouped[cat]) grouped[cat] = [];
    grouped[cat].push([key, pkg]);
  }

  // Display each category
  for (const [category, items] of Object.entries(grouped)) {
    console.log(chalk.yellow.bold(`\n${categoryIcon(category)} ${category.toUpperCase()}\n`));

    const table = new Table({
      head: [
        chalk.cyan('Package'),
        chalk.cyan('Name'),
        chalk.cyan('Description'),
      ],
      colWidths: [30, 35, 55],
      style: {
        head: [],
        border: ['dim'],
      },
      wordWrap: true,
    });

    for (const [key, pkg] of items) {
      table.push([
        chalk.green(key),
        pkg.name,
        chalk.dim(pkg.description),
      ]);
    }

    console.log(table.toString());
  }

  // Footer
  console.log();
  console.log(chalk.dim('Use'), chalk.cyan('neural-trader info <package>'), chalk.dim('for details'));
  console.log(chalk.dim('Use'), chalk.cyan('neural-trader init <package>'), chalk.dim('to create a project'));
  console.log();
}

function categoryIcon(category) {
  const icons = {
    trading: '💹',
    betting: '⚽',
    accounting: '🧮',
    prediction: '🔮',
    example: '🧪',
    data: '📊',
    markets: '🏛️',
  };
  return icons[category] || '📦';
}
```

---

## Example 3: Info Command with Rich UI

### Current Implementation

```javascript
// ❌ Current: Basic, plain text
info: (packageName) => {
  if (!packageName) {
    console.error('❌ Package name required');
    process.exit(1);
  }

  const pkg = PACKAGES[packageName];
  if (!pkg) {
    console.error(`❌ Unknown package: ${packageName}`);
    process.exit(1);
  }

  console.log('');
  console.log(`${c.cyan}${c.bright}${pkg.name}${c.reset}`);
  console.log(`${c.dim}Category: ${pkg.category}${c.reset}`);
  console.log('');
  console.log(`${c.bright}Description:${c.reset}`);
  console.log(`  ${pkg.description}`);
  // ... more output
}
```

### Refactored Implementation

```javascript
// ✅ Refactored: Rich UI with boxes, colors, examples
import chalk from 'chalk';
import boxen from 'boxen';
import Table from 'cli-table3';
import { PACKAGES } from '../data/packages.js';

export async function infoCommand(packageName, options) {
  const pkg = PACKAGES[packageName];

  if (!pkg) {
    console.error(chalk.red(`Package not found: ${packageName}`));
    console.log();
    console.log('Did you mean:');
    const suggestions = findSimilarPackages(packageName);
    suggestions.forEach(s => console.log(chalk.cyan(`  • ${s}`)));
    process.exit(1);
  }

  // Header box
  console.log(boxen(
    `${chalk.cyan.bold(pkg.name)}\n\n` +
    `${chalk.dim(pkg.description)}`,
    {
      padding: 1,
      margin: 1,
      borderStyle: 'round',
      borderColor: 'cyan',
      title: categoryIcon(pkg.category) + ' ' + pkg.category.toUpperCase(),
      titleAlignment: 'center',
    }
  ));

  // Features table
  if (pkg.features?.length > 0) {
    console.log(chalk.bold('\n✨ Features:\n'));

    const table = new Table({
      colWidths: [80],
      style: { border: ['dim'] },
      wordWrap: true,
    });

    pkg.features.forEach(feature => {
      table.push([`${chalk.green('•')} ${feature}`]);
    });

    console.log(table.toString());
  }

  // NPM packages
  if (pkg.packages?.length > 0) {
    console.log(chalk.bold('\n📦 NPM Packages:\n'));

    for (const p of pkg.packages) {
      console.log(`  ${chalk.cyan(p)}`);
    }
  }

  // Templates
  if (pkg.templates?.length > 0) {
    console.log(chalk.bold('\n📋 Available Templates:\n'));

    for (const t of pkg.templates) {
      console.log(`  ${chalk.yellow(t)}`);
    }
  }

  // Example usage
  console.log(chalk.bold('\n🚀 Quick Start:\n'));
  console.log(chalk.cyan(`  neural-trader init ${packageName}`));

  if (pkg.templates?.length > 0) {
    console.log(chalk.cyan(`  neural-trader init ${packageName} --template ${pkg.templates[0]}`));
  }

  console.log(chalk.cyan(`  neural-trader init ${packageName} --interactive`));

  // Links
  if (pkg.documentation || pkg.examples) {
    console.log(chalk.bold('\n📚 Resources:\n'));

    if (pkg.documentation) {
      console.log(`  ${chalk.blue('Documentation:')} ${pkg.documentation}`);
    }
    if (pkg.examples) {
      console.log(`  ${chalk.blue('Examples:')} ${pkg.examples}`);
    }
  }

  console.log();
}

function findSimilarPackages(query) {
  // Fuzzy matching implementation
  const keys = Object.keys(PACKAGES);
  return keys
    .map(key => ({
      key,
      distance: levenshteinDistance(query.toLowerCase(), key.toLowerCase()),
    }))
    .sort((a, b) => a.distance - b.distance)
    .slice(0, 3)
    .map(item => item.key);
}

function levenshteinDistance(a, b) {
  const matrix = [];

  for (let i = 0; i <= b.length; i++) {
    matrix[i] = [i];
  }

  for (let j = 0; j <= a.length; j++) {
    matrix[0][j] = j;
  }

  for (let i = 1; i <= b.length; i++) {
    for (let j = 1; j <= a.length; j++) {
      if (b.charAt(i - 1) === a.charAt(j - 1)) {
        matrix[i][j] = matrix[i - 1][j - 1];
      } else {
        matrix[i][j] = Math.min(
          matrix[i - 1][j - 1] + 1,
          matrix[i][j - 1] + 1,
          matrix[i - 1][j] + 1
        );
      }
    }
  }

  return matrix[b.length][a.length];
}
```

---

## Example 4: Doctor Command with Health Checks

### Refactored Implementation (Complete)

```javascript
// ✅ Modern: Comprehensive health checks with async tests
import chalk from 'chalk';
import ora from 'ora';
import Table from 'cli-table3';
import { execa } from 'execa';
import fs from 'fs/promises';
import { z } from 'zod';

export async function doctorCommand(options) {
  console.log(chalk.cyan.bold('\n🔍 Running Neural Trader health check...\n'));

  const checks = [
    {
      name: 'Node.js Version',
      check: checkNodeVersion,
    },
    {
      name: 'NPM Version',
      check: checkNpmVersion,
    },
    {
      name: 'NAPI Bindings',
      check: checkNapiBindings,
    },
    {
      name: 'Required Dependencies',
      check: checkDependencies,
    },
    {
      name: 'Project Configuration',
      check: checkConfiguration,
    },
    {
      name: 'Network Connectivity',
      check: checkNetwork,
    },
    {
      name: 'Disk Space',
      check: checkDiskSpace,
    },
  ];

  const results = [];

  for (const { name, check } of checks) {
    const spinner = ora(name).start();

    try {
      const result = await check();
      results.push({ name, ...result });

      if (result.status === 'ok') {
        spinner.succeed(chalk.green(name));
      } else if (result.status === 'warning') {
        spinner.warn(chalk.yellow(`${name}: ${result.message}`));
      } else {
        spinner.fail(chalk.red(`${name}: ${result.message}`));
      }
    } catch (err) {
      results.push({
        name,
        status: 'error',
        message: err.message,
      });
      spinner.fail(chalk.red(`${name}: ${err.message}`));
    }
  }

  // Summary table
  console.log();
  displaySummaryTable(results);

  // Recommendations
  const issues = results.filter(r => r.status !== 'ok');
  if (issues.length > 0) {
    console.log();
    displayRecommendations(issues);
  } else {
    console.log(chalk.green.bold('\n✅ All checks passed! System is healthy.\n'));
  }

  // Exit with appropriate code
  const hasErrors = results.some(r => r.status === 'error');
  if (hasErrors && !options.ignoreErrors) {
    process.exit(1);
  }
}

async function checkNodeVersion() {
  const version = process.version;
  const major = parseInt(version.slice(1).split('.')[0]);

  if (major >= 18) {
    return {
      status: 'ok',
      message: version,
      details: `Node.js ${version} (>= 18.0.0 required)`,
    };
  } else {
    return {
      status: 'error',
      message: `${version} (requires >= 18.0.0)`,
      fix: 'Update Node.js to version 18 or higher',
    };
  }
}

async function checkNpmVersion() {
  try {
    const { stdout } = await execa('npm', ['--version']);
    const version = stdout.trim();

    return {
      status: 'ok',
      message: version,
      details: `npm ${version}`,
    };
  } catch (err) {
    return {
      status: 'error',
      message: 'npm not found',
      fix: 'Install npm',
    };
  }
}

async function checkNapiBindings() {
  try {
    const nt = await import('../../../index.js');

    if (nt && typeof nt.fetchMarketData === 'function') {
      return {
        status: 'ok',
        message: 'Available',
        details: `${Object.keys(nt).length} functions loaded`,
      };
    } else {
      return {
        status: 'warning',
        message: 'Partially loaded',
        fix: 'Some NAPI functions may be unavailable',
      };
    }
  } catch (err) {
    return {
      status: 'warning',
      message: 'Not loaded (CLI-only mode)',
      details: 'NAPI bindings are optional for CLI usage',
    };
  }
}

async function checkDependencies() {
  try {
    const pkgPath = './package.json';
    const content = await fs.readFile(pkgPath, 'utf8');
    const pkg = JSON.parse(content);

    const deps = { ...pkg.dependencies, ...pkg.devDependencies };
    const neuralDeps = Object.keys(deps).filter(
      d => d.includes('neural-trader') || d.includes('@neural-trader')
    );

    if (neuralDeps.length > 0) {
      return {
        status: 'ok',
        message: `${neuralDeps.length} packages installed`,
        details: neuralDeps.join(', '),
      };
    } else {
      return {
        status: 'warning',
        message: 'No neural-trader packages found',
        fix: 'Run: neural-trader install <package>',
      };
    }
  } catch (err) {
    return {
      status: 'warning',
      message: 'No package.json found',
      details: 'Not in a neural-trader project',
    };
  }
}

async function checkConfiguration() {
  try {
    await fs.access('./config.json');

    const content = await fs.readFile('./config.json', 'utf8');
    JSON.parse(content); // Validate JSON

    return {
      status: 'ok',
      message: 'Valid configuration found',
    };
  } catch (err) {
    if (err.code === 'ENOENT') {
      return {
        status: 'info',
        message: 'No config.json (optional)',
      };
    }

    return {
      status: 'error',
      message: 'Invalid JSON in config.json',
      fix: 'Check config.json for syntax errors',
    };
  }
}

async function checkNetwork() {
  try {
    const { stdout } = await execa('npm', ['ping'], { timeout: 5000 });

    return {
      status: 'ok',
      message: 'NPM registry reachable',
    };
  } catch (err) {
    return {
      status: 'warning',
      message: 'Cannot reach NPM registry',
      details: 'Package installation may fail',
    };
  }
}

async function checkDiskSpace() {
  try {
    const { stdout } = await execa('df', ['-h', '.']);
    const lines = stdout.trim().split('\n');
    const data = lines[1].split(/\s+/);
    const available = data[3];
    const usedPercent = parseInt(data[4]);

    if (usedPercent < 90) {
      return {
        status: 'ok',
        message: `${available} available`,
        details: `${usedPercent}% used`,
      };
    } else {
      return {
        status: 'warning',
        message: `Low disk space (${available} available)`,
        fix: 'Free up disk space',
      };
    }
  } catch (err) {
    return {
      status: 'info',
      message: 'Could not check disk space',
    };
  }
}

function displaySummaryTable(results) {
  const table = new Table({
    head: [
      chalk.cyan('Check'),
      chalk.cyan('Status'),
      chalk.cyan('Details'),
    ],
    colWidths: [30, 15, 55],
    style: {
      head: [],
      border: ['dim'],
    },
    wordWrap: true,
  });

  for (const result of results) {
    const statusIcon = {
      ok: chalk.green('✓'),
      warning: chalk.yellow('⚠'),
      error: chalk.red('✗'),
      info: chalk.blue('ℹ'),
    }[result.status];

    table.push([
      result.name,
      `${statusIcon} ${result.status}`,
      result.message,
    ]);
  }

  console.log(table.toString());
}

function displayRecommendations(issues) {
  console.log(chalk.yellow.bold('⚠️  Recommendations:\n'));

  for (const issue of issues) {
    if (issue.fix) {
      console.log(`${chalk.yellow('•')} ${chalk.bold(issue.name)}:`);
      console.log(`  ${chalk.dim(issue.fix)}\n`);
    }
  }
}
```

---

## Example 5: Plugin System

```javascript
// src/cli/plugins/manager.js
import fs from 'fs/promises';
import path from 'path';
import chalk from 'chalk';
import { z } from 'zod';

// Plugin manifest schema
const PluginManifestSchema = z.object({
  name: z.string(),
  version: z.string(),
  description: z.string().optional(),
  author: z.string().optional(),
  dependencies: z.array(z.string()).optional(),
});

export class PluginManager {
  constructor(pluginDir = './plugins') {
    this.pluginDir = pluginDir;
    this.plugins = new Map();
    this.loaded = false;
  }

  async discover() {
    try {
      const files = await fs.readdir(this.pluginDir);

      for (const file of files) {
        if (file.startsWith('plugin-') && file.endsWith('.js')) {
          await this.loadPlugin(path.join(this.pluginDir, file));
        }
      }

      this.loaded = true;
      return this.plugins.size;
    } catch (err) {
      if (err.code === 'ENOENT') {
        // No plugins directory - that's ok
        return 0;
      }
      throw err;
    }
  }

  async loadPlugin(pluginPath) {
    try {
      const plugin = await import(pluginPath);

      if (!plugin.default) {
        throw new Error(`Plugin ${pluginPath} has no default export`);
      }

      const manifest = plugin.default;

      // Validate manifest
      PluginManifestSchema.parse(manifest);

      // Check for register function
      if (typeof manifest.register !== 'function') {
        throw new Error(`Plugin ${manifest.name} has no register() function`);
      }

      // Check dependencies
      if (manifest.dependencies) {
        for (const dep of manifest.dependencies) {
          if (!this.plugins.has(dep)) {
            console.warn(
              chalk.yellow(
                `Plugin ${manifest.name} depends on ${dep} which is not loaded`
              )
            );
          }
        }
      }

      this.plugins.set(manifest.name, manifest);

      if (process.env.DEBUG) {
        console.log(chalk.dim(`Loaded plugin: ${manifest.name} v${manifest.version}`));
      }
    } catch (err) {
      console.error(chalk.red(`Failed to load plugin ${pluginPath}:`), err.message);
    }
  }

  register(program) {
    if (!this.loaded) {
      throw new Error('Plugins not loaded. Call discover() first.');
    }

    let count = 0;

    for (const [name, plugin] of this.plugins) {
      try {
        plugin.register(program);
        count++;

        if (process.env.DEBUG) {
          console.log(chalk.green(`✓ Registered plugin: ${name}`));
        }
      } catch (err) {
        console.error(chalk.red(`Failed to register plugin ${name}:`), err.message);
      }
    }

    return count;
  }

  getPlugin(name) {
    return this.plugins.get(name);
  }

  hasPlugin(name) {
    return this.plugins.has(name);
  }

  listPlugins() {
    return Array.from(this.plugins.values()).map(p => ({
      name: p.name,
      version: p.version,
      description: p.description,
    }));
  }
}
```

**Example Plugin:**

```javascript
// plugins/plugin-backtest.js
export default {
  name: 'backtest',
  version: '1.0.0',
  description: 'Advanced backtesting commands',
  author: 'Neural Trader Team',

  register(program) {
    const cmd = program
      .command('backtest')
      .description('Run backtesting strategies');

    cmd
      .command('run <strategy>')
      .description('Run a backtest')
      .option('-s, --start <date>', 'Start date')
      .option('-e, --end <date>', 'End date')
      .option('-d, --data <path>', 'Historical data path')
      .option('--walk-forward', 'Enable walk-forward analysis')
      .action(async (strategy, options) => {
        const { runBacktest } = await import('./backtest-runner.js');
        await runBacktest(strategy, options);
      });

    cmd
      .command('report <id>')
      .description('View backtest report')
      .option('-f, --format <type>', 'Output format (table|json|html)', 'table')
      .action(async (id, options) => {
        const { generateReport } = await import('./backtest-reporter.js');
        await generateReport(id, options);
      });

    cmd
      .command('optimize <strategy>')
      .description('Optimize strategy parameters')
      .option('--method <type>', 'Optimization method (grid|genetic)', 'grid')
      .option('--params <json>', 'Parameter ranges')
      .action(async (strategy, options) => {
        const { optimizeStrategy } = await import('./backtest-optimizer.js');
        await optimizeStrategy(strategy, options);
      });
  },
};
```

---

## Example 6: Auto-completion Setup

```javascript
// src/completion/setup.js
import tabtab from 'tabtab';
import chalk from 'chalk';
import { PACKAGES } from '../cli/data/packages.js';

export async function setupCompletion() {
  try {
    await tabtab.install({
      name: 'neural-trader',
      completer: 'neural-trader-completion',
    });

    console.log(chalk.green('✓ Shell completion installed'));
    console.log();
    console.log('Restart your shell or run:');
    console.log(chalk.cyan('  source ~/.bashrc'));
    console.log(chalk.cyan('  source ~/.zshrc'));
    console.log();
  } catch (err) {
    console.error(chalk.red('Failed to install completion:'), err.message);
  }
}

export function handleCompletion() {
  if (!process.env.COMP_LINE) {
    return false;
  }

  const env = tabtab.parseEnv(process.env);

  // Commands after "neural-trader"
  if (env.prev === 'neural-trader') {
    tabtab.log([
      'init',
      'list',
      'info',
      'install',
      'test',
      'doctor',
      'search',
      'config',
      'completion',
    ]);
    return true;
  }

  // Package types after "init"
  if (env.prev === 'init') {
    tabtab.log(Object.keys(PACKAGES));
    return true;
  }

  // Package names after "info" or "install"
  if (env.prev === 'info' || env.prev === 'install') {
    tabtab.log(Object.keys(PACKAGES));
    return true;
  }

  // Categories after "list --category"
  if (env.prev === '--category' || env.prev === '-c') {
    const categories = [...new Set(Object.values(PACKAGES).map(p => p.category))];
    tabtab.log(categories);
    return true;
  }

  // Formats after "list --format"
  if (env.prev === '--format' || env.prev === '-f') {
    tabtab.log(['table', 'json', 'simple']);
    return true;
  }

  return false;
}
```

---

## Summary: Key Improvements

1. **Separation of Concerns:** Commands are in separate files
2. **Type Safety:** Zod schemas for validation
3. **Async/Await:** Modern promise handling
4. **Error Handling:** Proper try-catch and error messages
5. **User Experience:** Interactive prompts, progress bars, rich UI
6. **Performance:** Lazy loading, caching
7. **Extensibility:** Plugin system
8. **Testing:** Testable units
9. **Configuration:** Persistent user config
10. **Auto-completion:** Shell tab completion

**Code Reduction:**
- Current monolithic: 799 lines
- Refactored (total across modules): ~1200 lines
- BUT: Better organized, maintainable, testable, and extensible

**User Experience Improvement:**
- Interactive mode with inquirer
- Beautiful progress indicators
- Helpful error messages with suggestions
- Rich formatting with tables and boxes
- Shell auto-completion
- Plugin support for extensions
