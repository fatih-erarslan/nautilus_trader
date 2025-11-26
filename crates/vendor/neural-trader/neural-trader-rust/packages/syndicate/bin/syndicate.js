#!/usr/bin/env node

/**
 * Syndicate CLI - Comprehensive Investment Syndicate Management Tool
 *
 * Features:
 * - Syndicate creation and management
 * - Member management (add, remove, update, stats)
 * - Fund allocation with multiple strategies
 * - Profit distribution with various models
 * - Withdrawal processing
 * - Voting and governance
 * - Analytics and reporting
 * - Configuration management
 */

const yargs = require('yargs/yargs');
const { hideBin } = require('yargs/helpers');
const chalk = require('chalk');
const ora = require('ora');
const Table = require('cli-table3');
const fs = require('fs').promises;
const path = require('path');

// Configuration
const CONFIG_DIR = path.join(process.env.HOME || process.env.USERPROFILE, '.syndicate');
const CONFIG_FILE = path.join(CONFIG_DIR, 'config.json');
const DATA_DIR = path.join(CONFIG_DIR, 'data');

// Utility Functions
const log = {
  success: (msg) => console.log(chalk.green('✓'), msg),
  error: (msg) => console.error(chalk.red('✗'), msg),
  info: (msg) => console.log(chalk.blue('ℹ'), msg),
  warning: (msg) => console.log(chalk.yellow('⚠'), msg),
  header: (msg) => console.log(chalk.bold.cyan(`\n${msg}\n`)),
};

async function ensureDir(dir) {
  try {
    await fs.mkdir(dir, { recursive: true });
  } catch (error) {
    if (error.code !== 'EEXIST') throw error;
  }
}

async function loadConfig() {
  try {
    await ensureDir(CONFIG_DIR);
    await ensureDir(DATA_DIR);
    const data = await fs.readFile(CONFIG_FILE, 'utf8');
    return JSON.parse(data);
  } catch (error) {
    if (error.code === 'ENOENT') {
      return { syndicates: {}, members: {}, config: {} };
    }
    throw error;
  }
}

async function saveConfig(config) {
  await ensureDir(CONFIG_DIR);
  await fs.writeFile(CONFIG_FILE, JSON.stringify(config, null, 2));
}

async function loadSyndicateData(syndicateId) {
  try {
    const dataPath = path.join(DATA_DIR, `${syndicateId}.json`);
    const data = await fs.readFile(dataPath, 'utf8');
    return JSON.parse(data);
  } catch (error) {
    if (error.code === 'ENOENT') {
      return {
        id: syndicateId,
        members: [],
        allocations: [],
        distributions: [],
        withdrawals: [],
        votes: [],
        created: new Date().toISOString(),
      };
    }
    throw error;
  }
}

async function saveSyndicateData(syndicateId, data) {
  const dataPath = path.join(DATA_DIR, `${syndicateId}.json`);
  await fs.writeFile(dataPath, JSON.stringify(data, null, 2));
}

function formatCurrency(amount) {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
  }).format(amount);
}

function formatDate(dateStr) {
  return new Date(dateStr).toLocaleString();
}

function createTable(headers, options = {}) {
  return new Table({
    head: headers.map(h => chalk.cyan.bold(h)),
    style: { head: [], border: [] },
    ...options,
  });
}

// Command Handlers

// 1. CREATE COMMAND
async function handleCreate(argv) {
  const spinner = ora('Creating syndicate...').start();

  try {
    const config = await loadConfig();

    if (config.syndicates[argv.id]) {
      spinner.fail(`Syndicate '${argv.id}' already exists`);
      return;
    }

    const syndicate = {
      id: argv.id,
      bankroll: argv.bankroll,
      created: new Date().toISOString(),
      rules: null,
    };

    if (argv.rules) {
      try {
        const rulesContent = await fs.readFile(argv.rules, 'utf8');
        syndicate.rules = JSON.parse(rulesContent);
      } catch (error) {
        spinner.fail(`Failed to load rules file: ${error.message}`);
        return;
      }
    }

    config.syndicates[argv.id] = syndicate;
    await saveConfig(config);
    await saveSyndicateData(argv.id, {
      id: argv.id,
      members: [],
      allocations: [],
      distributions: [],
      withdrawals: [],
      votes: [],
      created: syndicate.created,
    });

    spinner.succeed(`Syndicate '${argv.id}' created successfully`);

    if (argv.json) {
      console.log(JSON.stringify(syndicate, null, 2));
    } else {
      log.info(`Bankroll: ${formatCurrency(argv.bankroll)}`);
      log.info(`Created: ${formatDate(syndicate.created)}`);
      if (syndicate.rules) {
        log.info(`Rules loaded from: ${argv.rules}`);
      }
    }
  } catch (error) {
    spinner.fail(`Failed to create syndicate: ${error.message}`);
    if (argv.verbose) {
      console.error(error);
    }
  }
}

// 2. MEMBER COMMANDS
async function handleMemberAdd(argv) {
  const spinner = ora('Adding member...').start();

  try {
    const config = await loadConfig();
    const syndicateId = argv.syndicate || Object.keys(config.syndicates)[0];

    if (!syndicateId) {
      spinner.fail('No syndicate found. Create one first.');
      return;
    }

    const data = await loadSyndicateData(syndicateId);

    const memberId = `mem-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    const member = {
      id: memberId,
      name: argv.name,
      email: argv.email,
      role: argv.role,
      capital: argv.capital,
      joined: new Date().toISOString(),
      active: true,
      totalProfit: 0,
      totalWithdrawals: 0,
    };

    data.members.push(member);
    await saveSyndicateData(syndicateId, data);

    spinner.succeed(`Member '${argv.name}' added successfully`);

    if (argv.json) {
      console.log(JSON.stringify(member, null, 2));
    } else {
      log.info(`Member ID: ${memberId}`);
      log.info(`Role: ${argv.role}`);
      log.info(`Capital: ${formatCurrency(argv.capital)}`);
    }
  } catch (error) {
    spinner.fail(`Failed to add member: ${error.message}`);
    if (argv.verbose) {
      console.error(error);
    }
  }
}

async function handleMemberList(argv) {
  const spinner = ora('Loading members...').start();

  try {
    const config = await loadConfig();
    const syndicateId = argv.syndicate || Object.keys(config.syndicates)[0];

    if (!syndicateId) {
      spinner.fail('No syndicate found');
      return;
    }

    const data = await loadSyndicateData(syndicateId);
    spinner.succeed(`Found ${data.members.length} members`);

    if (argv.json) {
      console.log(JSON.stringify(data.members, null, 2));
    } else {
      log.header(`Members of Syndicate: ${syndicateId}`);

      const table = createTable(['ID', 'Name', 'Email', 'Role', 'Capital', 'Profit', 'Status']);

      data.members.forEach(member => {
        table.push([
          chalk.yellow(member.id.substr(0, 12)),
          member.name,
          member.email,
          chalk.blue(member.role),
          formatCurrency(member.capital),
          formatCurrency(member.totalProfit),
          member.active ? chalk.green('Active') : chalk.red('Inactive'),
        ]);
      });

      console.log(table.toString());
    }
  } catch (error) {
    spinner.fail(`Failed to list members: ${error.message}`);
    if (argv.verbose) {
      console.error(error);
    }
  }
}

async function handleMemberStats(argv) {
  const spinner = ora('Loading member statistics...').start();

  try {
    const config = await loadConfig();
    const syndicates = Object.keys(config.syndicates);

    let member = null;
    let syndicateId = null;

    for (const id of syndicates) {
      const data = await loadSyndicateData(id);
      member = data.members.find(m => m.id === argv.memberId);
      if (member) {
        syndicateId = id;
        break;
      }
    }

    if (!member) {
      spinner.fail(`Member '${argv.memberId}' not found`);
      return;
    }

    const data = await loadSyndicateData(syndicateId);

    // Calculate statistics
    const memberDistributions = data.distributions.filter(d => d.memberId === argv.memberId);
    const memberWithdrawals = data.withdrawals.filter(w => w.memberId === argv.memberId);
    const memberAllocations = data.allocations.filter(a => a.memberId === argv.memberId);

    const totalDistributed = memberDistributions.reduce((sum, d) => sum + d.amount, 0);
    const totalWithdrawn = memberWithdrawals
      .filter(w => w.status === 'completed')
      .reduce((sum, w) => sum + w.amount, 0);
    const pendingWithdrawals = memberWithdrawals
      .filter(w => w.status === 'pending')
      .reduce((sum, w) => sum + w.amount, 0);

    const roi = member.capital > 0 ? ((totalDistributed / member.capital) * 100).toFixed(2) : 0;

    spinner.succeed('Member statistics loaded');

    if (argv.json) {
      console.log(JSON.stringify({
        member,
        statistics: {
          totalDistributed,
          totalWithdrawn,
          pendingWithdrawals,
          roi,
          allocationsCount: memberAllocations.length,
          distributionsCount: memberDistributions.length,
          withdrawalsCount: memberWithdrawals.length,
        },
      }, null, 2));
    } else {
      log.header(`Member Statistics: ${member.name}`);

      const table = createTable(['Metric', 'Value']);
      table.push(
        ['Member ID', chalk.yellow(member.id)],
        ['Email', member.email],
        ['Role', chalk.blue(member.role)],
        ['Status', member.active ? chalk.green('Active') : chalk.red('Inactive')],
        ['Joined', formatDate(member.joined)],
        ['', ''],
        [chalk.bold('Capital'), formatCurrency(member.capital)],
        [chalk.bold('Total Distributed'), formatCurrency(totalDistributed)],
        [chalk.bold('Total Withdrawn'), formatCurrency(totalWithdrawn)],
        [chalk.bold('Pending Withdrawals'), formatCurrency(pendingWithdrawals)],
        [chalk.bold('Current Balance'), formatCurrency(totalDistributed - totalWithdrawn - pendingWithdrawals)],
        [chalk.bold('ROI'), `${roi}%`],
        ['', ''],
        ['Allocations', memberAllocations.length],
        ['Distributions', memberDistributions.length],
        ['Withdrawals', memberWithdrawals.length]
      );

      console.log(table.toString());
    }
  } catch (error) {
    spinner.fail(`Failed to load member stats: ${error.message}`);
    if (argv.verbose) {
      console.error(error);
    }
  }
}

async function handleMemberUpdate(argv) {
  const spinner = ora('Updating member...').start();

  try {
    const config = await loadConfig();
    const syndicates = Object.keys(config.syndicates);

    let found = false;
    for (const id of syndicates) {
      const data = await loadSyndicateData(id);
      const member = data.members.find(m => m.id === argv.memberId);

      if (member) {
        if (argv.role) member.role = argv.role;
        if (argv.active !== undefined) member.active = argv.active;

        await saveSyndicateData(id, data);
        found = true;

        spinner.succeed(`Member '${member.name}' updated successfully`);

        if (argv.json) {
          console.log(JSON.stringify(member, null, 2));
        }
        break;
      }
    }

    if (!found) {
      spinner.fail(`Member '${argv.memberId}' not found`);
    }
  } catch (error) {
    spinner.fail(`Failed to update member: ${error.message}`);
    if (argv.verbose) {
      console.error(error);
    }
  }
}

async function handleMemberRemove(argv) {
  const spinner = ora('Removing member...').start();

  try {
    const config = await loadConfig();
    const syndicates = Object.keys(config.syndicates);

    let found = false;
    for (const id of syndicates) {
      const data = await loadSyndicateData(id);
      const memberIndex = data.members.findIndex(m => m.id === argv.memberId);

      if (memberIndex !== -1) {
        const member = data.members[memberIndex];
        member.active = false;
        member.removed = new Date().toISOString();

        await saveSyndicateData(id, data);
        found = true;

        spinner.succeed(`Member '${member.name}' removed successfully`);
        break;
      }
    }

    if (!found) {
      spinner.fail(`Member '${argv.memberId}' not found`);
    }
  } catch (error) {
    spinner.fail(`Failed to remove member: ${error.message}`);
    if (argv.verbose) {
      console.error(error);
    }
  }
}

// 3. ALLOCATE COMMANDS
async function handleAllocate(argv) {
  const spinner = ora('Processing allocation...').start();

  try {
    const opportunityData = JSON.parse(await fs.readFile(argv.opportunityFile, 'utf8'));

    const config = await loadConfig();
    const syndicateId = argv.syndicate || Object.keys(config.syndicates)[0];

    if (!syndicateId) {
      spinner.fail('No syndicate found');
      return;
    }

    const data = await loadSyndicateData(syndicateId);
    const syndicate = config.syndicates[syndicateId];

    // Calculate allocations based on strategy
    let allocations = [];
    const totalCapital = data.members.reduce((sum, m) => sum + m.capital, 0);

    switch (argv.strategy) {
      case 'kelly':
        // Kelly Criterion allocation
        allocations = data.members.map(member => ({
          memberId: member.id,
          memberName: member.name,
          allocation: calculateKellyAllocation(member, opportunityData, totalCapital),
          strategy: 'kelly',
        }));
        break;

      case 'fixed':
        // Fixed percentage allocation
        allocations = data.members.map(member => ({
          memberId: member.id,
          memberName: member.name,
          allocation: (member.capital / totalCapital) * opportunityData.totalAmount,
          strategy: 'fixed',
        }));
        break;

      case 'dynamic':
        // Dynamic based on performance
        allocations = data.members.map(member => ({
          memberId: member.id,
          memberName: member.name,
          allocation: calculateDynamicAllocation(member, data, opportunityData),
          strategy: 'dynamic',
        }));
        break;

      case 'risk-parity':
        // Risk parity allocation
        allocations = data.members.map(member => ({
          memberId: member.id,
          memberName: member.name,
          allocation: calculateRiskParityAllocation(member, opportunityData, totalCapital),
          strategy: 'risk-parity',
        }));
        break;

      default:
        spinner.fail(`Unknown strategy: ${argv.strategy}`);
        return;
    }

    const allocationRecord = {
      id: `alloc-${Date.now()}`,
      timestamp: new Date().toISOString(),
      opportunity: opportunityData,
      strategy: argv.strategy,
      allocations,
      totalAllocated: allocations.reduce((sum, a) => sum + a.allocation, 0),
    };

    data.allocations.push(allocationRecord);
    await saveSyndicateData(syndicateId, data);

    spinner.succeed('Allocation completed successfully');

    if (argv.json) {
      console.log(JSON.stringify(allocationRecord, null, 2));
    } else {
      log.header(`Allocation Summary - Strategy: ${argv.strategy}`);

      const table = createTable(['Member', 'Capital', 'Allocation', 'Percentage']);

      allocations.forEach(alloc => {
        const member = data.members.find(m => m.id === alloc.memberId);
        const percentage = ((alloc.allocation / member.capital) * 100).toFixed(2);

        table.push([
          alloc.memberName,
          formatCurrency(member.capital),
          formatCurrency(alloc.allocation),
          `${percentage}%`,
        ]);
      });

      console.log(table.toString());
      log.info(`Total Allocated: ${formatCurrency(allocationRecord.totalAllocated)}`);
    }
  } catch (error) {
    spinner.fail(`Failed to process allocation: ${error.message}`);
    if (argv.verbose) {
      console.error(error);
    }
  }
}

function calculateKellyAllocation(member, opportunity, totalCapital) {
  const p = opportunity.probability || 0.5;
  const odds = opportunity.odds || 2;
  const b = odds - 1;
  const q = 1 - p;
  const kelly = (p * b - q) / b;
  const kellyFraction = Math.max(0, Math.min(kelly, 0.25)); // Cap at 25%
  return (member.capital / totalCapital) * opportunity.totalAmount * kellyFraction;
}

function calculateDynamicAllocation(member, data, opportunity) {
  const recentDistributions = data.distributions
    .filter(d => d.memberId === member.id)
    .slice(-10);

  const avgProfit = recentDistributions.length > 0
    ? recentDistributions.reduce((sum, d) => sum + d.amount, 0) / recentDistributions.length
    : member.capital * 0.1;

  const performanceMultiplier = Math.min(2, Math.max(0.5, avgProfit / (member.capital * 0.1)));
  return (member.capital * 0.1 * performanceMultiplier);
}

function calculateRiskParityAllocation(member, opportunity, totalCapital) {
  const riskLevel = opportunity.riskLevel || 'medium';
  const riskFactors = { low: 1.2, medium: 1.0, high: 0.8 };
  const factor = riskFactors[riskLevel] || 1.0;
  return (member.capital / totalCapital) * opportunity.totalAmount * factor;
}

async function handleAllocateList(argv) {
  const spinner = ora('Loading allocations...').start();

  try {
    const config = await loadConfig();
    const syndicateId = argv.syndicate || Object.keys(config.syndicates)[0];

    if (!syndicateId) {
      spinner.fail('No syndicate found');
      return;
    }

    const data = await loadSyndicateData(syndicateId);
    spinner.succeed(`Found ${data.allocations.length} allocations`);

    if (argv.json) {
      console.log(JSON.stringify(data.allocations, null, 2));
    } else {
      log.header(`Allocations for Syndicate: ${syndicateId}`);

      const table = createTable(['ID', 'Date', 'Strategy', 'Total', 'Members']);

      data.allocations.slice(-20).reverse().forEach(alloc => {
        table.push([
          chalk.yellow(alloc.id.substr(0, 12)),
          formatDate(alloc.timestamp),
          chalk.blue(alloc.strategy),
          formatCurrency(alloc.totalAllocated),
          alloc.allocations.length,
        ]);
      });

      console.log(table.toString());
    }
  } catch (error) {
    spinner.fail(`Failed to list allocations: ${error.message}`);
    if (argv.verbose) {
      console.error(error);
    }
  }
}

async function handleAllocateHistory(argv) {
  const spinner = ora('Loading allocation history...').start();

  try {
    const config = await loadConfig();
    const syndicates = Object.keys(config.syndicates);

    let history = [];

    for (const id of syndicates) {
      const data = await loadSyndicateData(id);

      if (argv.member) {
        data.allocations.forEach(alloc => {
          const memberAlloc = alloc.allocations.find(a => a.memberId === argv.member);
          if (memberAlloc) {
            history.push({
              syndicate: id,
              ...alloc,
              memberAllocation: memberAlloc,
            });
          }
        });
      } else {
        history.push(...data.allocations.map(a => ({ syndicate: id, ...a })));
      }
    }

    spinner.succeed(`Found ${history.length} allocation records`);

    if (argv.json) {
      console.log(JSON.stringify(history, null, 2));
    } else {
      log.header('Allocation History');

      const table = createTable(['Date', 'Syndicate', 'Strategy', 'Amount']);

      history.slice(-20).reverse().forEach(item => {
        const amount = argv.member
          ? item.memberAllocation.allocation
          : item.totalAllocated;

        table.push([
          formatDate(item.timestamp),
          item.syndicate,
          chalk.blue(item.strategy),
          formatCurrency(amount),
        ]);
      });

      console.log(table.toString());
    }
  } catch (error) {
    spinner.fail(`Failed to load history: ${error.message}`);
    if (argv.verbose) {
      console.error(error);
    }
  }
}

// 4. DISTRIBUTE COMMANDS
async function handleDistribute(argv) {
  const spinner = ora('Processing profit distribution...').start();

  try {
    const config = await loadConfig();
    const syndicateId = argv.syndicate || Object.keys(config.syndicates)[0];

    if (!syndicateId) {
      spinner.fail('No syndicate found');
      return;
    }

    const data = await loadSyndicateData(syndicateId);
    const totalCapital = data.members.reduce((sum, m) => sum + m.capital, 0);

    let distributions = [];

    switch (argv.model) {
      case 'proportional':
        distributions = data.members.map(member => ({
          memberId: member.id,
          memberName: member.name,
          amount: (member.capital / totalCapital) * argv.profit,
          model: 'proportional',
        }));
        break;

      case 'performance':
        distributions = calculatePerformanceDistribution(data, argv.profit);
        break;

      case 'tiered':
        distributions = calculateTieredDistribution(data, argv.profit);
        break;

      case 'hybrid':
        distributions = calculateHybridDistribution(data, argv.profit, totalCapital);
        break;

      default:
        spinner.fail(`Unknown distribution model: ${argv.model}`);
        return;
    }

    const distributionRecord = {
      id: `dist-${Date.now()}`,
      timestamp: new Date().toISOString(),
      totalProfit: argv.profit,
      model: argv.model,
      distributions,
      totalDistributed: distributions.reduce((sum, d) => sum + d.amount, 0),
    };

    // Update member totals
    distributions.forEach(dist => {
      const member = data.members.find(m => m.id === dist.memberId);
      if (member) {
        member.totalProfit += dist.amount;
      }
    });

    data.distributions.push(distributionRecord);
    await saveSyndicateData(syndicateId, data);

    spinner.succeed('Distribution completed successfully');

    if (argv.json) {
      console.log(JSON.stringify(distributionRecord, null, 2));
    } else {
      log.header(`Distribution Summary - Model: ${argv.model}`);

      const table = createTable(['Member', 'Capital', 'Distribution', 'Share %']);

      distributions.forEach(dist => {
        const member = data.members.find(m => m.id === dist.memberId);
        const sharePercent = ((dist.amount / argv.profit) * 100).toFixed(2);

        table.push([
          dist.memberName,
          formatCurrency(member.capital),
          formatCurrency(dist.amount),
          `${sharePercent}%`,
        ]);
      });

      console.log(table.toString());
      log.info(`Total Distributed: ${formatCurrency(distributionRecord.totalDistributed)}`);
    }
  } catch (error) {
    spinner.fail(`Failed to process distribution: ${error.message}`);
    if (argv.verbose) {
      console.error(error);
    }
  }
}

function calculatePerformanceDistribution(data, totalProfit) {
  const recentAllocations = data.allocations.slice(-10);

  return data.members.map(member => {
    const memberAllocations = recentAllocations
      .flatMap(a => a.allocations)
      .filter(a => a.memberId === member.id);

    const avgAllocation = memberAllocations.length > 0
      ? memberAllocations.reduce((sum, a) => sum + a.allocation, 0) / memberAllocations.length
      : member.capital * 0.1;

    const performanceScore = avgAllocation / member.capital;

    return {
      memberId: member.id,
      memberName: member.name,
      amount: totalProfit * (performanceScore / data.members.length),
      model: 'performance',
    };
  });
}

function calculateTieredDistribution(data, totalProfit) {
  const tiers = [
    { threshold: 100000, rate: 0.4 },
    { threshold: 50000, rate: 0.3 },
    { threshold: 0, rate: 0.3 },
  ];

  return data.members.map(member => {
    const tier = tiers.find(t => member.capital >= t.threshold) || tiers[tiers.length - 1];
    const baseAmount = totalProfit / data.members.length;

    return {
      memberId: member.id,
      memberName: member.name,
      amount: baseAmount * tier.rate,
      model: 'tiered',
    };
  });
}

function calculateHybridDistribution(data, totalProfit, totalCapital) {
  const proportionalWeight = 0.6;
  const performanceWeight = 0.4;

  const proportional = data.members.map(m => ({
    memberId: m.id,
    amount: (m.capital / totalCapital) * totalProfit * proportionalWeight,
  }));

  const performance = calculatePerformanceDistribution(data, totalProfit * performanceWeight);

  return data.members.map(member => {
    const propAmount = proportional.find(p => p.memberId === member.id)?.amount || 0;
    const perfAmount = performance.find(p => p.memberId === member.id)?.amount || 0;

    return {
      memberId: member.id,
      memberName: member.name,
      amount: propAmount + perfAmount,
      model: 'hybrid',
    };
  });
}

async function handleDistributeHistory(argv) {
  const spinner = ora('Loading distribution history...').start();

  try {
    const config = await loadConfig();
    const syndicates = Object.keys(config.syndicates);

    let history = [];

    for (const id of syndicates) {
      const data = await loadSyndicateData(id);
      history.push(...data.distributions.map(d => ({ syndicate: id, ...d })));
    }

    spinner.succeed(`Found ${history.length} distributions`);

    if (argv.json) {
      console.log(JSON.stringify(history, null, 2));
    } else {
      log.header('Distribution History');

      const table = createTable(['Date', 'Syndicate', 'Model', 'Profit', 'Distributed']);

      history.slice(-20).reverse().forEach(item => {
        table.push([
          formatDate(item.timestamp),
          item.syndicate,
          chalk.blue(item.model),
          formatCurrency(item.totalProfit),
          formatCurrency(item.totalDistributed),
        ]);
      });

      console.log(table.toString());
    }
  } catch (error) {
    spinner.fail(`Failed to load history: ${error.message}`);
    if (argv.verbose) {
      console.error(error);
    }
  }
}

async function handleDistributePreview(argv) {
  const spinner = ora('Calculating distribution preview...').start();

  try {
    const config = await loadConfig();
    const syndicateId = argv.syndicate || Object.keys(config.syndicates)[0];

    if (!syndicateId) {
      spinner.fail('No syndicate found');
      return;
    }

    const data = await loadSyndicateData(syndicateId);
    const totalCapital = data.members.reduce((sum, m) => sum + m.capital, 0);

    let distributions = [];

    switch (argv.model) {
      case 'proportional':
        distributions = data.members.map(member => ({
          memberName: member.name,
          amount: (member.capital / totalCapital) * argv.profit,
        }));
        break;
      case 'performance':
        distributions = calculatePerformanceDistribution(data, argv.profit);
        break;
      case 'tiered':
        distributions = calculateTieredDistribution(data, argv.profit);
        break;
      case 'hybrid':
        distributions = calculateHybridDistribution(data, argv.profit, totalCapital);
        break;
      default:
        spinner.fail(`Unknown model: ${argv.model}`);
        return;
    }

    spinner.succeed('Distribution preview calculated');

    if (argv.json) {
      console.log(JSON.stringify(distributions, null, 2));
    } else {
      log.header(`Distribution Preview - Model: ${argv.model}`);

      const table = createTable(['Member', 'Distribution', 'Share %']);

      distributions.forEach(dist => {
        const sharePercent = ((dist.amount / argv.profit) * 100).toFixed(2);
        table.push([
          dist.memberName,
          formatCurrency(dist.amount),
          `${sharePercent}%`,
        ]);
      });

      console.log(table.toString());
      log.warning('This is a preview only. No changes have been made.');
    }
  } catch (error) {
    spinner.fail(`Failed to preview distribution: ${error.message}`);
    if (argv.verbose) {
      console.error(error);
    }
  }
}

// 5. WITHDRAW COMMANDS
async function handleWithdrawRequest(argv) {
  const spinner = ora('Processing withdrawal request...').start();

  try {
    const config = await loadConfig();
    const syndicates = Object.keys(config.syndicates);

    let found = false;
    for (const id of syndicates) {
      const data = await loadSyndicateData(id);
      const member = data.members.find(m => m.id === argv.memberId);

      if (member) {
        const withdrawal = {
          id: `with-${Date.now()}`,
          memberId: argv.memberId,
          memberName: member.name,
          amount: argv.amount,
          requested: new Date().toISOString(),
          status: 'pending',
          isEmergency: argv.emergency || false,
        };

        data.withdrawals.push(withdrawal);
        await saveSyndicateData(id, data);
        found = true;

        spinner.succeed('Withdrawal request submitted');

        if (argv.json) {
          console.log(JSON.stringify(withdrawal, null, 2));
        } else {
          log.info(`Request ID: ${withdrawal.id}`);
          log.info(`Amount: ${formatCurrency(argv.amount)}`);
          log.info(`Status: ${withdrawal.status}`);
          if (withdrawal.isEmergency) {
            log.warning('Marked as emergency withdrawal');
          }
        }
        break;
      }
    }

    if (!found) {
      spinner.fail(`Member '${argv.memberId}' not found`);
    }
  } catch (error) {
    spinner.fail(`Failed to submit withdrawal request: ${error.message}`);
    if (argv.verbose) {
      console.error(error);
    }
  }
}

async function handleWithdrawApprove(argv) {
  const spinner = ora('Approving withdrawal...').start();

  try {
    const config = await loadConfig();
    const syndicates = Object.keys(config.syndicates);

    let found = false;
    for (const id of syndicates) {
      const data = await loadSyndicateData(id);
      const withdrawal = data.withdrawals.find(w => w.id === argv.requestId);

      if (withdrawal) {
        withdrawal.status = 'approved';
        withdrawal.approved = new Date().toISOString();

        await saveSyndicateData(id, data);
        found = true;

        spinner.succeed('Withdrawal approved');

        if (argv.json) {
          console.log(JSON.stringify(withdrawal, null, 2));
        }
        break;
      }
    }

    if (!found) {
      spinner.fail(`Withdrawal '${argv.requestId}' not found`);
    }
  } catch (error) {
    spinner.fail(`Failed to approve withdrawal: ${error.message}`);
    if (argv.verbose) {
      console.error(error);
    }
  }
}

async function handleWithdrawProcess(argv) {
  const spinner = ora('Processing withdrawal...').start();

  try {
    const config = await loadConfig();
    const syndicates = Object.keys(config.syndicates);

    let found = false;
    for (const id of syndicates) {
      const data = await loadSyndicateData(id);
      const withdrawal = data.withdrawals.find(w => w.id === argv.requestId);

      if (withdrawal) {
        if (withdrawal.status !== 'approved') {
          spinner.fail('Withdrawal must be approved first');
          return;
        }

        withdrawal.status = 'completed';
        withdrawal.processed = new Date().toISOString();

        const member = data.members.find(m => m.id === withdrawal.memberId);
        if (member) {
          member.totalWithdrawals += withdrawal.amount;
        }

        await saveSyndicateData(id, data);
        found = true;

        spinner.succeed('Withdrawal processed successfully');

        if (argv.json) {
          console.log(JSON.stringify(withdrawal, null, 2));
        } else {
          log.info(`Member: ${withdrawal.memberName}`);
          log.info(`Amount: ${formatCurrency(withdrawal.amount)}`);
          log.info(`Processed: ${formatDate(withdrawal.processed)}`);
        }
        break;
      }
    }

    if (!found) {
      spinner.fail(`Withdrawal '${argv.requestId}' not found`);
    }
  } catch (error) {
    spinner.fail(`Failed to process withdrawal: ${error.message}`);
    if (argv.verbose) {
      console.error(error);
    }
  }
}

async function handleWithdrawList(argv) {
  const spinner = ora('Loading withdrawals...').start();

  try {
    const config = await loadConfig();
    const syndicates = Object.keys(config.syndicates);

    let withdrawals = [];

    for (const id of syndicates) {
      const data = await loadSyndicateData(id);
      let items = data.withdrawals.map(w => ({ syndicate: id, ...w }));

      if (argv.pending) {
        items = items.filter(w => w.status === 'pending');
      }

      withdrawals.push(...items);
    }

    spinner.succeed(`Found ${withdrawals.length} withdrawals`);

    if (argv.json) {
      console.log(JSON.stringify(withdrawals, null, 2));
    } else {
      log.header('Withdrawals');

      const table = createTable(['ID', 'Member', 'Amount', 'Status', 'Requested']);

      withdrawals.slice(-20).reverse().forEach(item => {
        const statusColor = item.status === 'completed' ? chalk.green
          : item.status === 'approved' ? chalk.blue
          : chalk.yellow;

        table.push([
          chalk.yellow(item.id.substr(0, 12)),
          item.memberName,
          formatCurrency(item.amount),
          statusColor(item.status),
          formatDate(item.requested),
        ]);
      });

      console.log(table.toString());
    }
  } catch (error) {
    spinner.fail(`Failed to list withdrawals: ${error.message}`);
    if (argv.verbose) {
      console.error(error);
    }
  }
}

// 6. VOTE COMMANDS
async function handleVoteCreate(argv) {
  const spinner = ora('Creating vote...').start();

  try {
    const config = await loadConfig();
    const syndicateId = argv.syndicate || Object.keys(config.syndicates)[0];

    if (!syndicateId) {
      spinner.fail('No syndicate found');
      return;
    }

    const data = await loadSyndicateData(syndicateId);

    const options = argv.options
      ? argv.options.split(',').map(o => o.trim())
      : ['Yes', 'No', 'Abstain'];

    const vote = {
      id: `vote-${Date.now()}`,
      proposal: argv.proposal,
      options,
      votes: {},
      created: new Date().toISOString(),
      active: true,
      votesCount: 0,
    };

    data.votes.push(vote);
    await saveSyndicateData(syndicateId, data);

    spinner.succeed('Vote created successfully');

    if (argv.json) {
      console.log(JSON.stringify(vote, null, 2));
    } else {
      log.info(`Vote ID: ${vote.id}`);
      log.info(`Proposal: ${vote.proposal}`);
      log.info(`Options: ${options.join(', ')}`);
    }
  } catch (error) {
    spinner.fail(`Failed to create vote: ${error.message}`);
    if (argv.verbose) {
      console.error(error);
    }
  }
}

async function handleVoteCast(argv) {
  const spinner = ora('Casting vote...').start();

  try {
    const config = await loadConfig();
    const syndicates = Object.keys(config.syndicates);

    let found = false;
    for (const id of syndicates) {
      const data = await loadSyndicateData(id);
      const vote = data.votes.find(v => v.id === argv.proposalId);

      if (vote) {
        if (!vote.active) {
          spinner.fail('Vote is no longer active');
          return;
        }

        if (!vote.options.includes(argv.option)) {
          spinner.fail(`Invalid option. Valid options: ${vote.options.join(', ')}`);
          return;
        }

        vote.votes[argv.memberId] = {
          option: argv.option,
          timestamp: new Date().toISOString(),
        };
        vote.votesCount = Object.keys(vote.votes).length;

        await saveSyndicateData(id, data);
        found = true;

        spinner.succeed('Vote cast successfully');

        if (argv.json) {
          console.log(JSON.stringify(vote.votes[argv.memberId], null, 2));
        }
        break;
      }
    }

    if (!found) {
      spinner.fail(`Vote '${argv.proposalId}' not found`);
    }
  } catch (error) {
    spinner.fail(`Failed to cast vote: ${error.message}`);
    if (argv.verbose) {
      console.error(error);
    }
  }
}

async function handleVoteResults(argv) {
  const spinner = ora('Calculating results...').start();

  try {
    const config = await loadConfig();
    const syndicates = Object.keys(config.syndicates);

    let found = false;
    for (const id of syndicates) {
      const data = await loadSyndicateData(id);
      const vote = data.votes.find(v => v.id === argv.proposalId);

      if (vote) {
        const results = {};
        vote.options.forEach(opt => {
          results[opt] = 0;
        });

        Object.values(vote.votes).forEach(v => {
          results[v.option] = (results[v.option] || 0) + 1;
        });

        const totalVotes = Object.keys(vote.votes).length;
        const totalMembers = data.members.filter(m => m.active).length;
        const participation = totalMembers > 0
          ? ((totalVotes / totalMembers) * 100).toFixed(2)
          : 0;

        spinner.succeed('Results calculated');

        if (argv.json) {
          console.log(JSON.stringify({
            vote,
            results,
            totalVotes,
            totalMembers,
            participation,
          }, null, 2));
        } else {
          log.header(`Vote Results: ${vote.proposal}`);

          const table = createTable(['Option', 'Votes', 'Percentage']);

          Object.entries(results).forEach(([option, count]) => {
            const percentage = totalVotes > 0
              ? ((count / totalVotes) * 100).toFixed(2)
              : 0;

            table.push([
              option,
              count,
              `${percentage}%`,
            ]);
          });

          console.log(table.toString());
          log.info(`Total Votes: ${totalVotes} / ${totalMembers}`);
          log.info(`Participation: ${participation}%`);
        }

        found = true;
        break;
      }
    }

    if (!found) {
      spinner.fail(`Vote '${argv.proposalId}' not found`);
    }
  } catch (error) {
    spinner.fail(`Failed to calculate results: ${error.message}`);
    if (argv.verbose) {
      console.error(error);
    }
  }
}

async function handleVoteList(argv) {
  const spinner = ora('Loading votes...').start();

  try {
    const config = await loadConfig();
    const syndicates = Object.keys(config.syndicates);

    let votes = [];

    for (const id of syndicates) {
      const data = await loadSyndicateData(id);
      let items = data.votes.map(v => ({ syndicate: id, ...v }));

      if (argv.active) {
        items = items.filter(v => v.active);
      }

      votes.push(...items);
    }

    spinner.succeed(`Found ${votes.length} votes`);

    if (argv.json) {
      console.log(JSON.stringify(votes, null, 2));
    } else {
      log.header('Votes');

      const table = createTable(['ID', 'Proposal', 'Options', 'Votes', 'Status', 'Created']);

      votes.slice(-20).reverse().forEach(item => {
        table.push([
          chalk.yellow(item.id.substr(0, 12)),
          item.proposal.substr(0, 40) + (item.proposal.length > 40 ? '...' : ''),
          item.options.length,
          item.votesCount,
          item.active ? chalk.green('Active') : chalk.gray('Closed'),
          formatDate(item.created),
        ]);
      });

      console.log(table.toString());
    }
  } catch (error) {
    spinner.fail(`Failed to list votes: ${error.message}`);
    if (argv.verbose) {
      console.error(error);
    }
  }
}

// 7. STATS COMMANDS
async function handleStats(argv) {
  const spinner = ora('Calculating statistics...').start();

  try {
    const config = await loadConfig();

    if (argv.syndicate) {
      const data = await loadSyndicateData(argv.syndicate);
      const syndicate = config.syndicates[argv.syndicate];

      const totalCapital = data.members.reduce((sum, m) => sum + m.capital, 0);
      const totalProfit = data.distributions.reduce((sum, d) => sum + d.totalDistributed, 0);
      const totalWithdrawals = data.withdrawals
        .filter(w => w.status === 'completed')
        .reduce((sum, w) => sum + w.amount, 0);

      const roi = totalCapital > 0 ? ((totalProfit / totalCapital) * 100).toFixed(2) : 0;

      spinner.succeed('Statistics calculated');

      if (argv.json) {
        console.log(JSON.stringify({
          syndicate,
          statistics: {
            members: data.members.length,
            activeMembers: data.members.filter(m => m.active).length,
            totalCapital,
            totalProfit,
            totalWithdrawals,
            roi,
            allocations: data.allocations.length,
            distributions: data.distributions.length,
            activeVotes: data.votes.filter(v => v.active).length,
          },
        }, null, 2));
      } else {
        log.header(`Syndicate Statistics: ${argv.syndicate}`);

        const table = createTable(['Metric', 'Value']);
        table.push(
          ['Created', formatDate(syndicate.created)],
          ['Initial Bankroll', formatCurrency(syndicate.bankroll)],
          ['', ''],
          [chalk.bold('Members'), data.members.length],
          [chalk.bold('Active Members'), data.members.filter(m => m.active).length],
          ['', ''],
          [chalk.bold('Total Capital'), formatCurrency(totalCapital)],
          [chalk.bold('Total Profit'), formatCurrency(totalProfit)],
          [chalk.bold('Total Withdrawals'), formatCurrency(totalWithdrawals)],
          [chalk.bold('ROI'), `${roi}%`],
          ['', ''],
          ['Allocations', data.allocations.length],
          ['Distributions', data.distributions.length],
          ['Withdrawals', data.withdrawals.length],
          ['Active Votes', data.votes.filter(v => v.active).length]
        );

        console.log(table.toString());
      }
    } else if (argv.member) {
      await handleMemberStats({ memberId: argv.member, json: argv.json, verbose: argv.verbose });
    } else if (argv.performance) {
      const syndicates = Object.keys(config.syndicates);
      let allData = [];

      for (const id of syndicates) {
        const data = await loadSyndicateData(id);
        allData.push({ id, data });
      }

      spinner.succeed('Performance statistics calculated');

      log.header('Performance Overview');

      const table = createTable(['Syndicate', 'Members', 'Capital', 'Profit', 'ROI']);

      allData.forEach(({ id, data }) => {
        const totalCapital = data.members.reduce((sum, m) => sum + m.capital, 0);
        const totalProfit = data.distributions.reduce((sum, d) => sum + d.totalDistributed, 0);
        const roi = totalCapital > 0 ? ((totalProfit / totalCapital) * 100).toFixed(2) : 0;

        table.push([
          id,
          data.members.filter(m => m.active).length,
          formatCurrency(totalCapital),
          formatCurrency(totalProfit),
          `${roi}%`,
        ]);
      });

      console.log(table.toString());
    }
  } catch (error) {
    spinner.fail(`Failed to calculate statistics: ${error.message}`);
    if (argv.verbose) {
      console.error(error);
    }
  }
}

// 8. CONFIG COMMANDS
async function handleConfigSet(argv) {
  const spinner = ora('Updating configuration...').start();

  try {
    const config = await loadConfig();

    if (!config.config) {
      config.config = {};
    }

    config.config[argv.key] = argv.value;
    await saveConfig(config);

    spinner.succeed(`Configuration updated: ${argv.key} = ${argv.value}`);
  } catch (error) {
    spinner.fail(`Failed to update configuration: ${error.message}`);
    if (argv.verbose) {
      console.error(error);
    }
  }
}

async function handleConfigGet(argv) {
  const spinner = ora('Loading configuration...').start();

  try {
    const config = await loadConfig();
    const value = config.config?.[argv.key];

    spinner.succeed('Configuration loaded');

    if (argv.json) {
      console.log(JSON.stringify({ [argv.key]: value }, null, 2));
    } else {
      if (value !== undefined) {
        log.info(`${argv.key}: ${value}`);
      } else {
        log.warning(`Configuration key '${argv.key}' not found`);
      }
    }
  } catch (error) {
    spinner.fail(`Failed to load configuration: ${error.message}`);
    if (argv.verbose) {
      console.error(error);
    }
  }
}

async function handleConfigRules(argv) {
  const spinner = ora('Loading syndicate rules...').start();

  try {
    const config = await loadConfig();
    const syndicateId = argv.syndicate || Object.keys(config.syndicates)[0];

    if (!syndicateId) {
      spinner.fail('No syndicate found');
      return;
    }

    const syndicate = config.syndicates[syndicateId];

    if (argv.file) {
      const rulesContent = await fs.readFile(argv.file, 'utf8');
      syndicate.rules = JSON.parse(rulesContent);
      await saveConfig(config);
      spinner.succeed('Rules updated successfully');
    } else {
      spinner.succeed('Rules loaded');

      if (argv.json) {
        console.log(JSON.stringify(syndicate.rules, null, 2));
      } else {
        if (syndicate.rules) {
          log.header(`Rules for Syndicate: ${syndicateId}`);
          console.log(JSON.stringify(syndicate.rules, null, 2));
        } else {
          log.warning('No rules configured for this syndicate');
        }
      }
    }
  } catch (error) {
    spinner.fail(`Failed to manage rules: ${error.message}`);
    if (argv.verbose) {
      console.error(error);
    }
  }
}

// Main CLI Setup
const cli = yargs(hideBin(process.argv))
  .scriptName('syndicate')
  .usage('$0 <command> [options]')
  .version('1.0.0')
  .option('json', {
    alias: 'j',
    describe: 'Output in JSON format',
    type: 'boolean',
    global: true,
  })
  .option('verbose', {
    alias: 'v',
    describe: 'Verbose output with error details',
    type: 'boolean',
    global: true,
  })
  .option('syndicate', {
    alias: 's',
    describe: 'Syndicate ID (uses first available if not specified)',
    type: 'string',
    global: true,
  })
  .command('create <id>', 'Create a new syndicate', (yargs) => {
    return yargs
      .positional('id', {
        describe: 'Syndicate identifier',
        type: 'string',
      })
      .option('bankroll', {
        alias: 'b',
        describe: 'Initial bankroll amount',
        type: 'number',
        demandOption: true,
      })
      .option('rules', {
        alias: 'r',
        describe: 'Path to rules JSON file',
        type: 'string',
      });
  }, handleCreate)
  .command('member <action>', 'Member management commands', (yargs) => {
    return yargs
      .command('add <name> <email> <role>', 'Add a new member', (yargs) => {
        return yargs
          .positional('name', { type: 'string' })
          .positional('email', { type: 'string' })
          .positional('role', { type: 'string' })
          .option('capital', {
            alias: 'c',
            describe: 'Initial capital contribution',
            type: 'number',
            demandOption: true,
          });
      }, handleMemberAdd)
      .command('list', 'List all members', {}, handleMemberList)
      .command('stats <member-id>', 'Show member statistics', (yargs) => {
        return yargs.positional('member-id', { type: 'string' });
      }, handleMemberStats)
      .command('update <member-id>', 'Update member information', (yargs) => {
        return yargs
          .positional('member-id', { type: 'string' })
          .option('role', { type: 'string' })
          .option('active', { type: 'boolean' });
      }, handleMemberUpdate)
      .command('remove <member-id>', 'Remove a member', (yargs) => {
        return yargs.positional('member-id', { type: 'string' });
      }, handleMemberRemove)
      .demandCommand();
  })
  .command('allocate [action]', 'Fund allocation commands', (yargs) => {
    return yargs
      .command('$0 <opportunity-file>', 'Allocate funds based on strategy', (yargs) => {
        return yargs
          .positional('opportunity-file', {
            describe: 'Path to opportunity JSON file',
            type: 'string',
          })
          .option('strategy', {
            describe: 'Allocation strategy',
            choices: ['kelly', 'fixed', 'dynamic', 'risk-parity'],
            default: 'kelly',
          });
      }, handleAllocate)
      .command('list', 'List all allocations', {}, handleAllocateList)
      .command('history', 'Show allocation history', (yargs) => {
        return yargs.option('member', {
          describe: 'Filter by member ID',
          type: 'string',
        });
      }, handleAllocateHistory);
  })
  .command('distribute [action]', 'Profit distribution commands', (yargs) => {
    return yargs
      .command('$0 <profit>', 'Distribute profits to members', (yargs) => {
        return yargs
          .positional('profit', {
            describe: 'Total profit amount to distribute',
            type: 'number',
          })
          .option('model', {
            describe: 'Distribution model',
            choices: ['proportional', 'performance', 'tiered', 'hybrid'],
            default: 'proportional',
          });
      }, handleDistribute)
      .command('history', 'Show distribution history', {}, handleDistributeHistory)
      .command('preview <profit>', 'Preview distribution without applying', (yargs) => {
        return yargs
          .positional('profit', { type: 'number' })
          .option('model', {
            choices: ['proportional', 'performance', 'tiered', 'hybrid'],
            default: 'proportional',
          });
      }, handleDistributePreview);
  })
  .command('withdraw <action>', 'Withdrawal management commands', (yargs) => {
    return yargs
      .command('request <member-id> <amount>', 'Request a withdrawal', (yargs) => {
        return yargs
          .positional('member-id', { type: 'string' })
          .positional('amount', { type: 'number' })
          .option('emergency', {
            alias: 'e',
            describe: 'Mark as emergency withdrawal',
            type: 'boolean',
            default: false,
          });
      }, handleWithdrawRequest)
      .command('approve <request-id>', 'Approve a withdrawal', (yargs) => {
        return yargs.positional('request-id', { type: 'string' });
      }, handleWithdrawApprove)
      .command('process <request-id>', 'Process an approved withdrawal', (yargs) => {
        return yargs.positional('request-id', { type: 'string' });
      }, handleWithdrawProcess)
      .command('list', 'List all withdrawals', (yargs) => {
        return yargs.option('pending', {
          describe: 'Show only pending withdrawals',
          type: 'boolean',
        });
      }, handleWithdrawList)
      .demandCommand();
  })
  .command('vote <action>', 'Voting and governance commands', (yargs) => {
    return yargs
      .command('create <proposal>', 'Create a new vote', (yargs) => {
        return yargs
          .positional('proposal', {
            describe: 'Vote proposal text',
            type: 'string',
          })
          .option('options', {
            describe: 'Comma-separated voting options',
            type: 'string',
          });
      }, handleVoteCreate)
      .command('cast <proposal-id> <option>', 'Cast a vote', (yargs) => {
        return yargs
          .positional('proposal-id', { type: 'string' })
          .positional('option', { type: 'string' })
          .option('member', {
            alias: 'm',
            describe: 'Member ID casting the vote',
            type: 'string',
            demandOption: true,
          });
      }, (argv) => handleVoteCast({ ...argv, memberId: argv.member }))
      .command('results <proposal-id>', 'Show vote results', (yargs) => {
        return yargs.positional('proposal-id', { type: 'string' });
      }, handleVoteResults)
      .command('list', 'List all votes', (yargs) => {
        return yargs.option('active', {
          describe: 'Show only active votes',
          type: 'boolean',
        });
      }, handleVoteList)
      .demandCommand();
  })
  .command('stats', 'Show statistics and analytics', (yargs) => {
    return yargs
      .option('syndicate', {
        describe: 'Show syndicate statistics',
        type: 'string',
      })
      .option('member', {
        describe: 'Show member statistics',
        type: 'string',
      })
      .option('performance', {
        describe: 'Show performance overview',
        type: 'boolean',
      });
  }, handleStats)
  .command('config <action>', 'Configuration management', (yargs) => {
    return yargs
      .command('set <key> <value>', 'Set configuration value', (yargs) => {
        return yargs
          .positional('key', { type: 'string' })
          .positional('value', { type: 'string' });
      }, handleConfigSet)
      .command('get <key>', 'Get configuration value', (yargs) => {
        return yargs.positional('key', { type: 'string' });
      }, handleConfigGet)
      .command('rules', 'Manage syndicate rules', (yargs) => {
        return yargs.option('file', {
          describe: 'Path to rules JSON file',
          type: 'string',
        });
      }, handleConfigRules)
      .demandCommand();
  })
  .demandCommand()
  .help()
  .alias('help', 'h')
  .epilogue('For more information, visit https://github.com/neural-trader')
  .wrap(null);

// Execute CLI
cli.parse();
