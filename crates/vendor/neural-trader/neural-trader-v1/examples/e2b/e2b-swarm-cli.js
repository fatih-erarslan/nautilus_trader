#!/usr/bin/env node

/**
 * E2B Trading Swarm CLI
 *
 * Comprehensive command-line interface for managing E2B trading swarms.
 * Provides sandbox management, agent deployment, swarm coordination,
 * and real-time monitoring capabilities.
 *
 * Usage:
 *   e2b-swarm create --template trading-bot --count 3
 *   e2b-swarm deploy --agent momentum --symbols AAPL,MSFT
 *   e2b-swarm monitor --interval 5s
 *   e2b-swarm scale --count 5
 */

require('dotenv').config();
const { Command } = require('commander');
const chalk = require('chalk');
const fs = require('fs');
const path = require('path');

// Constants
const CLI_VERSION = '2.1.1';
const STATE_FILE = path.join(process.cwd(), '.swarm', 'cli-state.json');
const LOG_FILE = path.join(process.cwd(), '.swarm', 'cli.log');

/**
 * CLI State Manager
 * Handles persistent state for sandboxes, agents, and deployments
 */
class CLIStateManager {
  constructor() {
    this.state = this.loadState();
  }

  loadState() {
    try {
      if (fs.existsSync(STATE_FILE)) {
        return JSON.parse(fs.readFileSync(STATE_FILE, 'utf8'));
      }
    } catch (error) {
      this.log(`Warning: Failed to load state: ${error.message}`, 'warning');
    }

    return {
      sandboxes: [],
      agents: [],
      deployments: [],
      lastUpdate: null,
      version: CLI_VERSION
    };
  }

  saveState() {
    try {
      const dir = path.dirname(STATE_FILE);
      if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
      }
      this.state.lastUpdate = new Date().toISOString();
      fs.writeFileSync(STATE_FILE, JSON.stringify(this.state, null, 2));
    } catch (error) {
      this.log(`Error saving state: ${error.message}`, 'error');
    }
  }

  addSandbox(sandbox) {
    this.state.sandboxes.push(sandbox);
    this.saveState();
  }

  updateSandbox(id, updates) {
    const index = this.state.sandboxes.findIndex(s => s.id === id);
    if (index !== -1) {
      this.state.sandboxes[index] = { ...this.state.sandboxes[index], ...updates };
      this.saveState();
    }
  }

  removeSandbox(id) {
    this.state.sandboxes = this.state.sandboxes.filter(s => s.id !== id);
    this.saveState();
  }

  getSandboxes(status = null) {
    if (status) {
      return this.state.sandboxes.filter(s => s.status === status);
    }
    return this.state.sandboxes;
  }

  addAgent(agent) {
    this.state.agents.push(agent);
    this.saveState();
  }

  addDeployment(deployment) {
    this.state.deployments.push(deployment);
    this.saveState();
  }

  log(message, level = 'info') {
    const timestamp = new Date().toISOString();
    const logEntry = `[${timestamp}] [${level.toUpperCase()}] ${message}\n`;

    try {
      const dir = path.dirname(LOG_FILE);
      if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
      }
      fs.appendFileSync(LOG_FILE, logEntry);
    } catch (error) {
      // Silent fail for logging
    }
  }
}

/**
 * Output Formatter
 * Handles colored output and formatting
 */
class OutputFormatter {
  constructor(jsonMode = false) {
    this.jsonMode = jsonMode;
  }

  success(message) {
    if (this.jsonMode) return;
    console.log(chalk.green('✓'), message);
  }

  error(message) {
    if (this.jsonMode) return;
    console.error(chalk.red('✗'), message);
  }

  warning(message) {
    if (this.jsonMode) return;
    console.log(chalk.yellow('⚠'), message);
  }

  info(message) {
    if (this.jsonMode) return;
    console.log(chalk.blue('ℹ'), message);
  }

  json(data) {
    console.log(JSON.stringify(data, null, 2));
  }

  table(headers, rows) {
    if (this.jsonMode) {
      this.json({ headers, rows });
      return;
    }

    const columnWidths = headers.map((h, i) =>
      Math.max(h.length, ...rows.map(r => String(r[i] || '').length))
    );

    const separator = columnWidths.map(w => '─'.repeat(w + 2)).join('┼');
    const headerRow = headers.map((h, i) =>
      ` ${h.padEnd(columnWidths[i])} `
    ).join('│');

    console.log('┌' + separator.replace(/┼/g, '┬') + '┐');
    console.log('│' + headerRow + '│');
    console.log('├' + separator + '┤');

    rows.forEach(row => {
      const rowStr = row.map((cell, i) =>
        ` ${String(cell || '').padEnd(columnWidths[i])} `
      ).join('│');
      console.log('│' + rowStr + '│');
    });

    console.log('└' + separator.replace(/┼/g, '┴') + '┘');
  }

  progressBar(current, total, label = '') {
    if (this.jsonMode) return;

    const percentage = Math.round((current / total) * 100);
    const filled = Math.round((current / total) * 40);
    const empty = 40 - filled;

    const bar = '█'.repeat(filled) + '░'.repeat(empty);
    const progress = `${label} ${bar} ${percentage}% (${current}/${total})`;

    process.stdout.write(`\r${progress}`);
    if (current === total) {
      process.stdout.write('\n');
    }
  }

  banner(title) {
    if (this.jsonMode) return;

    const width = 60;
    const titlePadding = Math.floor((width - title.length - 2) / 2);

    console.log('\n' + chalk.cyan('═'.repeat(width)));
    console.log(chalk.cyan('║') + ' '.repeat(titlePadding) + chalk.bold(title) + ' '.repeat(width - titlePadding - title.length - 2) + chalk.cyan('║'));
    console.log(chalk.cyan('═'.repeat(width)) + '\n');
  }
}

/**
 * E2B Sandbox Manager
 * Handles sandbox lifecycle operations
 */
class SandboxManager {
  constructor(state, formatter) {
    this.state = state;
    this.formatter = formatter;
    this.validateEnvironment();
  }

  validateEnvironment() {
    const required = ['E2B_API_KEY', 'E2B_ACCESS_TOKEN'];
    const missing = required.filter(key => !process.env[key]);

    if (missing.length > 0) {
      this.formatter.error(`Missing required environment variables: ${missing.join(', ')}`);
      this.formatter.info('Please configure them in your .env file');
      process.exit(1);
    }
  }

  async create(options) {
    const { template = 'base', count = 1, name } = options;

    this.formatter.banner('Creating E2B Sandboxes');
    this.formatter.info(`Template: ${template}`);
    this.formatter.info(`Count: ${count}`);
    this.formatter.info(`Name: ${name || 'auto-generated'}\n`);

    const sandboxes = [];

    for (let i = 0; i < count; i++) {
      this.formatter.progressBar(i + 1, count, 'Creating sandboxes');

      try {
        const sandbox = await this.createSingle(template, name ? `${name}-${i + 1}` : null);
        sandboxes.push(sandbox);
        this.state.addSandbox(sandbox);
      } catch (error) {
        this.formatter.error(`Failed to create sandbox ${i + 1}: ${error.message}`);
      }

      // Rate limiting
      if (i < count - 1) {
        await this.sleep(1000);
      }
    }

    this.formatter.success(`\nCreated ${sandboxes.length} of ${count} sandboxes`);

    if (this.formatter.jsonMode) {
      this.formatter.json({ sandboxes });
    } else {
      this.displaySandboxes(sandboxes);
    }

    return sandboxes;
  }

  async createSingle(template, name) {
    const sandboxId = `sb-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;

    // Simulate E2B sandbox creation
    // In production, this would use the actual E2B SDK
    const sandbox = {
      id: sandboxId,
      name: name || sandboxId,
      template,
      status: 'running',
      created_at: new Date().toISOString(),
      resources: {
        cpu: 2,
        memory_mb: 1024
      },
      url: `https://e2b.dev/sandboxes/${sandboxId}`
    };

    this.state.log(`Created sandbox: ${sandboxId}`, 'info');
    return sandbox;
  }

  list(options) {
    const { status } = options;
    const sandboxes = this.state.getSandboxes(status);

    if (this.formatter.jsonMode) {
      this.formatter.json({ sandboxes, total: sandboxes.length });
      return;
    }

    this.formatter.banner('E2B Sandboxes');

    if (sandboxes.length === 0) {
      this.formatter.info('No sandboxes found');
      if (status) {
        this.formatter.info(`Filter: status=${status}`);
      }
      return;
    }

    const rows = sandboxes.map(s => [
      s.id.substring(0, 20) + '...',
      s.name || 'N/A',
      s.template,
      this.getStatusColor(s.status),
      new Date(s.created_at).toLocaleString()
    ]);

    this.formatter.table(
      ['ID', 'Name', 'Template', 'Status', 'Created'],
      rows
    );

    this.formatter.info(`\nTotal: ${sandboxes.length} sandbox${sandboxes.length !== 1 ? 'es' : ''}`);
  }

  async status(sandboxId) {
    const sandbox = this.state.getSandboxes().find(s => s.id === sandboxId);

    if (!sandbox) {
      this.formatter.error(`Sandbox not found: ${sandboxId}`);
      return;
    }

    if (this.formatter.jsonMode) {
      this.formatter.json(sandbox);
      return;
    }

    this.formatter.banner(`Sandbox Status: ${sandbox.name || sandbox.id}`);

    console.log(chalk.bold('ID:'), sandbox.id);
    console.log(chalk.bold('Name:'), sandbox.name || 'N/A');
    console.log(chalk.bold('Template:'), sandbox.template);
    console.log(chalk.bold('Status:'), this.getStatusColor(sandbox.status));
    console.log(chalk.bold('Created:'), new Date(sandbox.created_at).toLocaleString());

    if (sandbox.resources) {
      console.log(chalk.bold('\nResources:'));
      console.log(`  CPU: ${sandbox.resources.cpu} cores`);
      console.log(`  Memory: ${sandbox.resources.memory_mb} MB`);
    }

    if (sandbox.url) {
      console.log(chalk.bold('\nURL:'), chalk.cyan(sandbox.url));
    }
  }

  async destroy(sandboxId, options) {
    const { force = false } = options;
    const sandbox = this.state.getSandboxes().find(s => s.id === sandboxId);

    if (!sandbox) {
      this.formatter.error(`Sandbox not found: ${sandboxId}`);
      return;
    }

    if (!force) {
      this.formatter.warning('Destroying sandbox. Use --force to skip confirmation.');
      // In a real implementation, prompt for confirmation
    }

    this.formatter.info(`Destroying sandbox: ${sandboxId}`);

    try {
      // Simulate destruction
      await this.sleep(1000);
      this.state.removeSandbox(sandboxId);
      this.state.log(`Destroyed sandbox: ${sandboxId}`, 'info');
      this.formatter.success('Sandbox destroyed successfully');
    } catch (error) {
      this.formatter.error(`Failed to destroy sandbox: ${error.message}`);
    }
  }

  getStatusColor(status) {
    const colors = {
      running: chalk.green('●') + ' running',
      stopped: chalk.yellow('●') + ' stopped',
      failed: chalk.red('●') + ' failed',
      pending: chalk.blue('●') + ' pending'
    };
    return colors[status] || status;
  }

  sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  displaySandboxes(sandboxes) {
    console.log();
    sandboxes.forEach(s => {
      console.log(chalk.green('✓'), `${s.name || s.id}`);
      console.log(`  ID: ${s.id}`);
      console.log(`  Status: ${this.getStatusColor(s.status)}`);
      console.log(`  URL: ${chalk.cyan(s.url)}`);
      console.log();
    });
  }
}

/**
 * Agent Deployment Manager
 * Handles agent deployment and coordination
 */
class AgentManager {
  constructor(state, formatter, sandboxManager) {
    this.state = state;
    this.formatter = formatter;
    this.sandboxManager = sandboxManager;
  }

  async deploy(options) {
    const { agent, symbols, sandbox: sandboxId } = options;
    const symbolList = symbols ? symbols.split(',') : ['SPY'];

    this.formatter.banner('Deploying Trading Agent');
    this.formatter.info(`Agent Type: ${agent}`);
    this.formatter.info(`Symbols: ${symbolList.join(', ')}`);

    let sandbox;
    if (sandboxId) {
      sandbox = this.state.getSandboxes().find(s => s.id === sandboxId);
      if (!sandbox) {
        this.formatter.error(`Sandbox not found: ${sandboxId}`);
        return;
      }
    } else {
      // Create new sandbox
      this.formatter.info('Creating new sandbox for agent...');
      const sandboxes = await this.sandboxManager.create({
        template: 'trading-bot',
        count: 1,
        name: `${agent}-agent`
      });
      sandbox = sandboxes[0];
    }

    const deployment = await this.deployAgent(agent, symbolList, sandbox);
    this.state.addAgent(deployment);

    if (this.formatter.jsonMode) {
      this.formatter.json(deployment);
    } else {
      this.formatter.success('\nAgent deployed successfully');
      console.log(chalk.bold('Agent ID:'), deployment.id);
      console.log(chalk.bold('Sandbox:'), deployment.sandbox_id);
      console.log(chalk.bold('Strategy:'), deployment.strategy);
      console.log(chalk.bold('Symbols:'), deployment.symbols.join(', '));
    }

    return deployment;
  }

  async deployAgent(agentType, symbols, sandbox) {
    const agentId = `agent-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;

    const strategies = {
      momentum: 'Momentum Trading',
      pairs: 'Pairs Trading',
      neural: 'Neural Forecasting',
      mean_reversion: 'Mean Reversion',
      arbitrage: 'Statistical Arbitrage'
    };

    const agent = {
      id: agentId,
      type: agentType,
      strategy: strategies[agentType] || agentType,
      sandbox_id: sandbox.id,
      symbols,
      status: 'deployed',
      deployed_at: new Date().toISOString(),
      resources: sandbox.resources
    };

    this.state.log(`Deployed agent: ${agentId} (${agentType})`, 'info');
    return agent;
  }

  list() {
    const agents = this.state.state.agents;

    if (this.formatter.jsonMode) {
      this.formatter.json({ agents, total: agents.length });
      return;
    }

    this.formatter.banner('Deployed Agents');

    if (agents.length === 0) {
      this.formatter.info('No agents deployed');
      return;
    }

    const rows = agents.map(a => [
      a.id.substring(0, 20) + '...',
      a.strategy,
      a.symbols.join(', ').substring(0, 20),
      this.sandboxManager.getStatusColor(a.status),
      new Date(a.deployed_at).toLocaleString()
    ]);

    this.formatter.table(
      ['ID', 'Strategy', 'Symbols', 'Status', 'Deployed'],
      rows
    );

    this.formatter.info(`\nTotal: ${agents.length} agent${agents.length !== 1 ? 's' : ''}`);
  }
}

/**
 * Swarm Coordinator
 * Handles swarm-level operations
 */
class SwarmCoordinator {
  constructor(state, formatter, sandboxManager, agentManager) {
    this.state = state;
    this.formatter = formatter;
    this.sandboxManager = sandboxManager;
    this.agentManager = agentManager;
  }

  async scale(options) {
    const { count } = options;
    const current = this.state.getSandboxes('running').length;

    this.formatter.banner('Scaling Swarm');
    this.formatter.info(`Current sandboxes: ${current}`);
    this.formatter.info(`Target sandboxes: ${count}`);

    if (count > current) {
      // Scale up
      const toCreate = count - current;
      this.formatter.info(`Scaling up: creating ${toCreate} new sandbox${toCreate !== 1 ? 'es' : ''}`);
      await this.sandboxManager.create({ count: toCreate, template: 'trading-bot' });
    } else if (count < current) {
      // Scale down
      const toRemove = current - count;
      this.formatter.warning(`Scaling down: removing ${toRemove} sandbox${toRemove !== 1 ? 'es' : ''}`);

      const sandboxes = this.state.getSandboxes('running');
      for (let i = 0; i < toRemove; i++) {
        await this.sandboxManager.destroy(sandboxes[i].id, { force: true });
      }
    } else {
      this.formatter.info('Already at target scale');
    }

    this.formatter.success('Scaling complete');
  }

  async monitor(options) {
    const { interval = '5s', duration } = options;
    const intervalMs = this.parseInterval(interval);
    const startTime = Date.now();

    this.formatter.banner('Monitoring Swarm');
    this.formatter.info(`Interval: ${interval}`);
    if (duration) {
      this.formatter.info(`Duration: ${duration}`);
    }
    this.formatter.info('Press Ctrl+C to stop\n');

    let iteration = 0;
    const monitorLoop = setInterval(() => {
      iteration++;
      this.displayStatus(iteration);

      if (duration) {
        const elapsed = Date.now() - startTime;
        const durationMs = this.parseInterval(duration);
        if (elapsed >= durationMs) {
          clearInterval(monitorLoop);
          this.formatter.success('\nMonitoring complete');
        }
      }
    }, intervalMs);

    // Handle Ctrl+C
    process.on('SIGINT', () => {
      clearInterval(monitorLoop);
      this.formatter.info('\nMonitoring stopped');
      process.exit(0);
    });
  }

  displayStatus(iteration) {
    const sandboxes = this.state.getSandboxes();
    const agents = this.state.state.agents;
    const running = sandboxes.filter(s => s.status === 'running').length;

    if (!this.formatter.jsonMode) {
      console.clear();
      this.formatter.banner('Swarm Status');
      console.log(chalk.bold('Update:'), `#${iteration}`, chalk.gray(`(${new Date().toLocaleTimeString()})`));
      console.log(chalk.bold('Sandboxes:'), running, '/', sandboxes.length);
      console.log(chalk.bold('Agents:'), agents.length);
      console.log();

      if (sandboxes.length > 0) {
        const rows = sandboxes.slice(0, 10).map(s => [
          s.name || s.id.substring(0, 15),
          this.sandboxManager.getStatusColor(s.status),
          `${s.resources?.cpu || 0} CPU`,
          `${s.resources?.memory_mb || 0} MB`
        ]);

        this.formatter.table(['Name', 'Status', 'CPU', 'Memory'], rows);
      }
    } else {
      this.formatter.json({
        iteration,
        timestamp: new Date().toISOString(),
        sandboxes: { total: sandboxes.length, running },
        agents: agents.length
      });
    }
  }

  async health(options) {
    const { detailed = false } = options;

    this.formatter.banner('Swarm Health Check');

    const sandboxes = this.state.getSandboxes();
    const agents = this.state.state.agents;
    const running = sandboxes.filter(s => s.status === 'running').length;
    const failed = sandboxes.filter(s => s.status === 'failed').length;

    const health = {
      status: failed === 0 ? 'healthy' : 'degraded',
      sandboxes: {
        total: sandboxes.length,
        running,
        failed
      },
      agents: {
        total: agents.length,
        active: agents.filter(a => a.status === 'deployed').length
      },
      resources: {
        total_cpu: sandboxes.reduce((sum, s) => sum + (s.resources?.cpu || 0), 0),
        total_memory_mb: sandboxes.reduce((sum, s) => sum + (s.resources?.memory_mb || 0), 0)
      }
    };

    if (this.formatter.jsonMode) {
      this.formatter.json(health);
      return;
    }

    console.log(chalk.bold('Status:'), health.status === 'healthy' ? chalk.green('● Healthy') : chalk.yellow('● Degraded'));
    console.log();
    console.log(chalk.bold('Sandboxes:'));
    console.log(`  Total: ${health.sandboxes.total}`);
    console.log(`  Running: ${chalk.green(health.sandboxes.running)}`);
    if (health.sandboxes.failed > 0) {
      console.log(`  Failed: ${chalk.red(health.sandboxes.failed)}`);
    }
    console.log();
    console.log(chalk.bold('Agents:'));
    console.log(`  Total: ${health.agents.total}`);
    console.log(`  Active: ${chalk.green(health.agents.active)}`);
    console.log();
    console.log(chalk.bold('Resources:'));
    console.log(`  CPU Cores: ${health.resources.total_cpu}`);
    console.log(`  Memory: ${health.resources.total_memory_mb} MB (${(health.resources.total_memory_mb / 1024).toFixed(2)} GB)`);

    if (detailed) {
      console.log();
      console.log(chalk.bold('Detailed Status:'));
      this.sandboxManager.list({});
    }
  }

  parseInterval(interval) {
    const match = interval.match(/^(\d+)(ms|s|m|h)?$/);
    if (!match) return 5000;

    const value = parseInt(match[1]);
    const unit = match[2] || 's';

    const multipliers = { ms: 1, s: 1000, m: 60000, h: 3600000 };
    return value * multipliers[unit];
  }
}

/**
 * Strategy Executor
 * Handles strategy execution and backtesting
 */
class StrategyExecutor {
  constructor(state, formatter, agentManager) {
    this.state = state;
    this.formatter = formatter;
    this.agentManager = agentManager;
  }

  async execute(options) {
    const { strategy, symbols, sandbox: sandboxId } = options;
    const symbolList = symbols ? symbols.split(',') : ['SPY'];

    this.formatter.banner('Executing Strategy');
    this.formatter.info(`Strategy: ${strategy}`);
    this.formatter.info(`Symbols: ${symbolList.join(', ')}`);

    // Deploy agent if needed
    const agent = await this.agentManager.deploy({
      agent: strategy,
      symbols: symbolList.join(','),
      sandbox: sandboxId
    });

    // Simulate strategy execution
    this.formatter.info('\nStarting strategy execution...');
    await this.sleep(2000);

    const result = {
      execution_id: `exec-${Date.now()}`,
      agent_id: agent.id,
      strategy,
      symbols: symbolList,
      started_at: new Date().toISOString(),
      status: 'running'
    };

    this.formatter.success('Strategy execution started');

    if (this.formatter.jsonMode) {
      this.formatter.json(result);
    } else {
      console.log(chalk.bold('Execution ID:'), result.execution_id);
      console.log(chalk.bold('Agent ID:'), result.agent_id);
      console.log(chalk.bold('Status:'), chalk.green('running'));
    }

    return result;
  }

  async backtest(options) {
    const { strategy, start, end = new Date().toISOString().split('T')[0], symbols = 'SPY' } = options;
    const symbolList = symbols.split(',');

    this.formatter.banner('Running Backtest');
    this.formatter.info(`Strategy: ${strategy}`);
    this.formatter.info(`Symbols: ${symbolList.join(', ')}`);
    this.formatter.info(`Period: ${start} to ${end}`);
    this.formatter.info('\nSimulating backtest...');

    // Simulate backtest
    for (let i = 1; i <= 10; i++) {
      this.formatter.progressBar(i, 10, 'Processing');
      await this.sleep(500);
    }

    const result = {
      backtest_id: `bt-${Date.now()}`,
      strategy,
      symbols: symbolList,
      period: { start, end },
      metrics: {
        total_return: '12.34%',
        sharpe_ratio: 1.45,
        max_drawdown: '-8.21%',
        win_rate: '58.3%',
        total_trades: 147
      },
      completed_at: new Date().toISOString()
    };

    this.formatter.success('\nBacktest complete');

    if (this.formatter.jsonMode) {
      this.formatter.json(result);
    } else {
      console.log();
      console.log(chalk.bold('Results:'));
      console.log(`  Total Return: ${chalk.green(result.metrics.total_return)}`);
      console.log(`  Sharpe Ratio: ${result.metrics.sharpe_ratio}`);
      console.log(`  Max Drawdown: ${chalk.red(result.metrics.max_drawdown)}`);
      console.log(`  Win Rate: ${result.metrics.win_rate}`);
      console.log(`  Total Trades: ${result.metrics.total_trades}`);
    }

    return result;
  }

  sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

/**
 * Main CLI Program
 */
function createCLI() {
  const program = new Command();
  const state = new CLIStateManager();

  program
    .name('e2b-swarm')
    .description('E2B Trading Swarm Management CLI')
    .version(CLI_VERSION)
    .option('-j, --json', 'output in JSON format');

  // Initialize managers
  const getManagers = (options) => {
    const formatter = new OutputFormatter(options.json);
    const sandboxManager = new SandboxManager(state, formatter);
    const agentManager = new AgentManager(state, formatter, sandboxManager);
    const swarmCoordinator = new SwarmCoordinator(state, formatter, sandboxManager, agentManager);
    const strategyExecutor = new StrategyExecutor(state, formatter, agentManager);
    return { formatter, sandboxManager, agentManager, swarmCoordinator, strategyExecutor };
  };

  // Sandbox commands
  const sandbox = program.command('create').description('Create E2B sandboxes');
  sandbox
    .option('-t, --template <template>', 'sandbox template', 'base')
    .option('-c, --count <count>', 'number of sandboxes', '1')
    .option('-n, --name <name>', 'sandbox name prefix')
    .action(async (options) => {
      const { sandboxManager } = getManagers(program.opts());
      await sandboxManager.create({
        template: options.template,
        count: parseInt(options.count),
        name: options.name
      });
    });

  program.command('list').description('List all sandboxes')
    .option('-s, --status <status>', 'filter by status')
    .action((options) => {
      const { sandboxManager } = getManagers(program.opts());
      sandboxManager.list(options);
    });

  program.command('status <sandbox-id>').description('Get sandbox status')
    .action(async (sandboxId) => {
      const { sandboxManager } = getManagers(program.opts());
      await sandboxManager.status(sandboxId);
    });

  program.command('destroy <sandbox-id>').description('Destroy a sandbox')
    .option('-f, --force', 'skip confirmation')
    .action(async (sandboxId, options) => {
      const { sandboxManager } = getManagers(program.opts());
      await sandboxManager.destroy(sandboxId, options);
    });

  // Agent commands
  program.command('deploy').description('Deploy trading agent')
    .requiredOption('-a, --agent <type>', 'agent type (momentum, pairs, neural, mean_reversion, arbitrage)')
    .option('-s, --symbols <symbols>', 'comma-separated symbol list', 'SPY')
    .option('--sandbox <id>', 'target sandbox ID')
    .action(async (options) => {
      const { agentManager } = getManagers(program.opts());
      await agentManager.deploy(options);
    });

  program.command('agents').description('List deployed agents')
    .action(() => {
      const { agentManager } = getManagers(program.opts());
      agentManager.list();
    });

  // Swarm commands
  program.command('scale').description('Scale swarm size')
    .requiredOption('-c, --count <count>', 'target sandbox count')
    .action(async (options) => {
      const { swarmCoordinator } = getManagers(program.opts());
      await swarmCoordinator.scale({ count: parseInt(options.count) });
    });

  program.command('monitor').description('Monitor swarm in real-time')
    .option('-i, --interval <interval>', 'update interval (e.g., 5s, 1m)', '5s')
    .option('-d, --duration <duration>', 'monitoring duration')
    .action(async (options) => {
      const { swarmCoordinator } = getManagers(program.opts());
      await swarmCoordinator.monitor(options);
    });

  program.command('health').description('Check swarm health')
    .option('--detailed', 'show detailed health information')
    .action(async (options) => {
      const { swarmCoordinator } = getManagers(program.opts());
      await swarmCoordinator.health(options);
    });

  // Strategy commands
  program.command('execute').description('Execute trading strategy')
    .requiredOption('-s, --strategy <strategy>', 'strategy type')
    .option('--symbols <symbols>', 'comma-separated symbol list', 'SPY')
    .option('--sandbox <id>', 'target sandbox ID')
    .action(async (options) => {
      const { strategyExecutor } = getManagers(program.opts());
      await strategyExecutor.execute(options);
    });

  program.command('backtest').description('Run strategy backtest')
    .requiredOption('-s, --strategy <strategy>', 'strategy type')
    .requiredOption('--start <date>', 'start date (YYYY-MM-DD)')
    .option('--end <date>', 'end date (YYYY-MM-DD)', new Date().toISOString().split('T')[0])
    .option('--symbols <symbols>', 'comma-separated symbol list', 'SPY')
    .action(async (options) => {
      const { strategyExecutor } = getManagers(program.opts());
      await strategyExecutor.backtest(options);
    });

  return program;
}

/**
 * ReasoningBank Learning Dashboard Commands
 */
function addLearningCommands(program, state) {
  const { DashboardCLI } = require('../src/reasoningbank/dashboard-cli');

  const learning = program.command('learning')
    .description('ReasoningBank learning visualization and analytics');

  learning.command('dashboard')
    .description('Display live learning dashboard')
    .option('--live', 'live update mode')
    .option('-i, --interval <ms>', 'update interval in milliseconds', '2000')
    .option('-d, --duration <ms>', 'display duration in milliseconds', '60000')
    .option('-s, --source <file>', 'load data from file')
    .action(async (options) => {
      const formatter = new OutputFormatter(program.opts().json);
      const dashboard = new DashboardCLI(state, formatter);

      if (options.live) {
        await dashboard.displayLive({
          interval: parseInt(options.interval),
          duration: parseInt(options.duration),
          source: options.source
        });
      } else {
        await dashboard.generateHTML({
          source: options.source,
          open: true
        });
      }
    });

  learning.command('report')
    .description('Generate learning analytics report')
    .option('-f, --format <format>', 'output format (html, markdown, json)', 'html')
    .option('-o, --output <path>', 'output file path')
    .option('-s, --source <file>', 'load data from file')
    .action(async (options) => {
      const formatter = new OutputFormatter(program.opts().json);
      const dashboard = new DashboardCLI(state, formatter);

      await dashboard.generateReport({
        format: options.format,
        output: options.output,
        source: options.source
      });
    });

  learning.command('stats')
    .description('Show learning statistics')
    .option('-a, --agent <id>', 'agent ID for detailed stats')
    .option('-s, --source <file>', 'load data from file')
    .action(async (options) => {
      const formatter = new OutputFormatter(program.opts().json);
      const dashboard = new DashboardCLI(state, formatter);

      if (options.agent) {
        await dashboard.showAgentStats({
          agent: options.agent,
          source: options.source
        });
      } else {
        await dashboard.quickStats({ source: options.source });
      }
    });

  learning.command('analytics')
    .description('Display learning analytics and recommendations')
    .option('-s, --source <file>', 'load data from file')
    .action(async (options) => {
      const formatter = new OutputFormatter(program.opts().json);
      const dashboard = new DashboardCLI(state, formatter);

      await dashboard.showAnalytics({ source: options.source });
    });

  learning.command('export')
    .description('Export learning data')
    .option('-f, --format <format>', 'export format (json)', 'json')
    .option('-o, --output <path>', 'output file path')
    .option('-s, --source <file>', 'load data from file')
    .action(async (options) => {
      const formatter = new OutputFormatter(program.opts().json);
      const dashboard = new DashboardCLI(state, formatter);

      await dashboard.exportData({
        format: options.format,
        output: options.output,
        source: options.source
      });
    });

  return learning;
}

// Main execution
if (require.main === module) {
  const program = createCLI();

  // Add learning dashboard commands
  addLearningCommands(program, new CLIStateManager());

  program.parse(process.argv);

  // Show help if no command
  if (process.argv.length === 2) {
    program.help();
  }
}

module.exports = { createCLI, CLIStateManager, OutputFormatter, addLearningCommands };
