# E2B Swarm Deployment Examples

Production-ready examples for deploying distributed trading swarms with E2B sandboxes.

## Table of Contents

1. [Basic Swarm Initialization](#basic-swarm-initialization)
2. [Multi-Strategy Swarm](#multi-strategy-swarm)
3. [Auto-Scaling Swarm](#auto-scaling-swarm)
4. [High-Frequency Trading Swarm](#high-frequency-trading-swarm)
5. [Fault-Tolerant Swarm](#fault-tolerant-swarm)
6. [Global Market Coverage](#global-market-coverage)
7. [Complete Production Swarm](#complete-production-swarm)

---

## 1. Basic Swarm Initialization

Simple swarm setup with multiple trading agents.

```javascript
const {
  initE2bSwarm,
  deployTradingAgent,
  createE2bSandbox,
  getSwarmStatus,
  SwarmTopology,
  DistributionStrategy,
  AgentType
} = require('@rUv/neural-trader-backend');

async function basicSwarmSetup() {
  console.log('=== Basic E2B Swarm Setup ===\n');

  // Step 1: Initialize swarm with mesh topology
  console.log('1. Initializing swarm...');

  const config = JSON.stringify({
    topology: SwarmTopology.Mesh,
    maxAgents: 5,
    distributionStrategy: DistributionStrategy.Adaptive,
    enableGpu: true,
    autoScaling: false,
    minAgents: 3,
    maxMemoryMb: 512,
    timeoutSecs: 300
  });

  const swarm = await initE2bSwarm('mesh', config);

  console.log(`   Swarm ID: ${swarm.swarmId}`);
  console.log(`   Topology: ${swarm.topology}`);
  console.log(`   Agents: ${swarm.agentCount}`);
  console.log(`   Status: ${swarm.status}\n`);

  // Step 2: Create sandboxes for agents
  console.log('2. Creating E2B sandboxes...');

  const sandboxes = [];

  for (let i = 0; i < 3; i++) {
    const sandbox = await createE2bSandbox(
      `trading-sandbox-${i+1}`,
      'nodejs'
    );

    console.log(`   ✓ Created: ${sandbox.sandboxId}`);
    sandboxes.push(sandbox);
  }

  console.log('');

  // Step 3: Deploy trading agents
  console.log('3. Deploying trading agents...\n');

  const agents = [
    {
      type: 'momentum',
      symbols: ['AAPL', 'GOOGL', 'MSFT'],
      params: { lookback: 20, threshold: 0.02 }
    },
    {
      type: 'mean_reversion',
      symbols: ['AMZN', 'TSLA', 'NVDA'],
      params: { window: 50, bands: 2.0 }
    },
    {
      type: 'neural',
      symbols: ['SPY', 'QQQ', 'IWM'],
      params: { model_type: 'lstm', horizon: 5 }
    }
  ];

  const deployedAgents = [];

  for (let i = 0; i < agents.length; i++) {
    const agent = agents[i];
    const sandbox = sandboxes[i];

    const deployment = await deployTradingAgent(
      sandbox.sandboxId,
      agent.type,
      agent.symbols,
      JSON.stringify(agent.params)
    );

    console.log(`   Agent ${i+1}:`);
    console.log(`     Type: ${deployment.agentType}`);
    console.log(`     Sandbox: ${deployment.sandboxId}`);
    console.log(`     Symbols: ${deployment.symbols.join(', ')}`);
    console.log(`     Status: ${deployment.status}\n`);

    deployedAgents.push(deployment);
  }

  // Step 4: Check swarm status
  console.log('4. Swarm Status:');

  const status = await getSwarmStatus(swarm.swarmId);

  console.log(`   Active Agents: ${status.activeAgents}`);
  console.log(`   Idle Agents: ${status.idleAgents}`);
  console.log(`   Total Trades: ${status.totalTrades}`);
  console.log(`   Total P&L: $${status.totalPnl.toLocaleString()}`);
  console.log(`   Uptime: ${status.uptimeSecs}s`);

  return { swarm, sandboxes, deployedAgents };
}

basicSwarmSetup();
```

---

## 2. Multi-Strategy Swarm

Deploy multiple trading strategies across swarm.

```javascript
const {
  initE2bSwarm,
  deployTradingAgent,
  executeSwarmStrategy,
  getSwarmPerformance,
  createE2bSandbox
} = require('@rUv/neural-trader-backend');

async function multiStrategySwarm() {
  console.log('=== Multi-Strategy Trading Swarm ===\n');

  // Initialize hierarchical swarm for strategy coordination
  const config = JSON.stringify({
    topology: 1,  // Hierarchical
    maxAgents: 10,
    distributionStrategy: 2,  // Specialized
    enableGpu: true,
    autoScaling: true,
    minAgents: 5
  });

  const swarm = await initE2bSwarm('hierarchical', config);
  console.log(`Swarm initialized: ${swarm.swarmId}\n`);

  // Define strategies and their target symbols
  const strategies = [
    {
      name: 'momentum',
      symbols: ['AAPL', 'MSFT', 'GOOGL'],
      count: 2  // Deploy 2 agents
    },
    {
      name: 'mean_reversion',
      symbols: ['AMZN', 'TSLA', 'NVDA', 'AMD'],
      count: 2
    },
    {
      name: 'pairs',
      symbols: ['KO', 'PEP', 'WMT', 'TGT'],
      count: 2
    },
    {
      name: 'neural',
      symbols: ['SPY', 'QQQ', 'DIA', 'IWM'],
      count: 2
    },
    {
      name: 'arbitrage',
      symbols: ['BTC-USD', 'ETH-USD'],
      count: 2
    }
  ];

  console.log('Deploying strategy agents...\n');

  const deployments = [];

  for (const strategy of strategies) {
    console.log(`${strategy.name.toUpperCase()} Strategy:`);

    for (let i = 0; i < strategy.count; i++) {
      // Create sandbox
      const sandbox = await createE2bSandbox(
        `${strategy.name}-${i+1}`,
        'nodejs'
      );

      // Deploy agent
      const agent = await deployTradingAgent(
        sandbox.sandboxId,
        strategy.name,
        strategy.symbols,
        JSON.stringify({
          instance_id: i + 1,
          enable_gpu: true
        })
      );

      console.log(`  Agent ${i+1}: ${agent.agentId} on ${sandbox.sandboxId}`);
      deployments.push({ strategy: strategy.name, agent, sandbox });
    }

    console.log('');
  }

  // Execute each strategy across its agents
  console.log('\nExecuting strategies...\n');

  for (const strategy of strategies) {
    const execution = await executeSwarmStrategy(
      swarm.swarmId,
      strategy.name,
      strategy.symbols
    );

    console.log(`${strategy.name.toUpperCase()}:`);
    console.log(`  Execution ID: ${execution.executionId}`);
    console.log(`  Agents Used: ${execution.agentsUsed}`);
    console.log(`  Expected Return: ${execution.expectedReturn.toFixed(2)}%`);
    console.log(`  Risk Score: ${execution.riskScore.toFixed(2)}`);
    console.log('');
  }

  // Get performance metrics
  console.log('Overall Swarm Performance:\n');

  const performance = await getSwarmPerformance(swarm.swarmId);

  console.log(`  Total Return: ${performance.totalReturn.toFixed(2)}%`);
  console.log(`  Sharpe Ratio: ${performance.sharpeRatio.toFixed(2)}`);
  console.log(`  Max Drawdown: ${performance.maxDrawdown.toFixed(2)}%`);
  console.log(`  Win Rate: ${performance.winRate.toFixed(2)}%`);
  console.log(`  Total Trades: ${performance.totalTrades}`);
  console.log(`  Profit Factor: ${performance.profitFactor.toFixed(2)}`);

  return { swarm, deployments, performance };
}

multiStrategySwarm();
```

---

## 3. Auto-Scaling Swarm

Dynamic swarm that scales based on market conditions.

```javascript
const {
  initE2bSwarm,
  scaleSwarm,
  getSwarmStatus,
  monitorSwarmHealth,
  getSwarmMetrics
} = require('@rUv/neural-trader-backend');

class AutoScalingSwarm {
  constructor(minAgents = 5, maxAgents = 20) {
    this.minAgents = minAgents;
    this.maxAgents = maxAgents;
    this.swarmId = null;
    this.isRunning = false;
  }

  async initialize() {
    console.log('=== Auto-Scaling Swarm ===\n');

    const config = JSON.stringify({
      topology: 0,  // Mesh
      maxAgents: this.maxAgents,
      distributionStrategy: 4,  // Adaptive
      enableGpu: true,
      autoScaling: true,
      minAgents: this.minAgents,
      maxMemoryMb: 1024
    });

    const swarm = await initE2bSwarm('mesh', config);
    this.swarmId = swarm.swarmId;

    console.log(`Swarm initialized: ${this.swarmId}`);
    console.log(`Scaling range: ${this.minAgents} - ${this.maxAgents} agents\n`);
  }

  async checkAndScale() {
    // Get current metrics
    const health = await monitorSwarmHealth();
    const status = await getSwarmStatus(this.swarmId);
    const metrics = await getSwarmMetrics(this.swarmId);

    console.log(`\n[${new Date().toISOString()}] Swarm Health Check:`);
    console.log(`  Active Agents: ${status.activeAgents}`);
    console.log(`  CPU Usage: ${health.cpuUsage.toFixed(2)}%`);
    console.log(`  Memory Usage: ${health.memoryUsage.toFixed(2)}%`);
    console.log(`  Avg Response Time: ${health.avgResponseTime.toFixed(2)}ms`);
    console.log(`  Throughput: ${metrics.throughput.toFixed(2)} trades/hour`);
    console.log(`  Success Rate: ${metrics.successRate.toFixed(2)}%`);

    // Scaling logic
    let targetAgents = status.activeAgents;

    // Scale up conditions
    if (health.cpuUsage > 80 || health.avgResponseTime > 1000) {
      targetAgents = Math.min(status.activeAgents + 2, this.maxAgents);
      console.log(`  ⚠ High load detected - scaling UP to ${targetAgents} agents`);
    }

    // Scale down conditions
    else if (health.cpuUsage < 30 && status.activeAgents > this.minAgents) {
      targetAgents = Math.max(status.activeAgents - 1, this.minAgents);
      console.log(`  → Low load detected - scaling DOWN to ${targetAgents} agents`);
    }

    // Performance-based scaling
    else if (metrics.throughput > 1000 && status.activeAgents < this.maxAgents) {
      targetAgents = status.activeAgents + 1;
      console.log(`  📈 High throughput - scaling UP to ${targetAgents} agents`);
    }

    // Execute scaling if needed
    if (targetAgents !== status.activeAgents) {
      const scaleResult = await scaleSwarm(this.swarmId, targetAgents);

      console.log(`\n  Scaling Complete:`);
      console.log(`    Previous: ${scaleResult.previousCount} agents`);
      console.log(`    New: ${scaleResult.newCount} agents`);
      console.log(`    Added: ${scaleResult.agentsAdded}`);
      console.log(`    Removed: ${scaleResult.agentsRemoved}`);
    }
  }

  async start(intervalSeconds = 60) {
    await this.initialize();
    this.isRunning = true;

    console.log(`Starting auto-scaling monitor (${intervalSeconds}s intervals)...\n`);

    // Initial check
    await this.checkAndScale();

    // Periodic checks
    this.intervalId = setInterval(async () => {
      if (this.isRunning) {
        await this.checkAndScale();
      }
    }, intervalSeconds * 1000);
  }

  stop() {
    this.isRunning = false;
    if (this.intervalId) {
      clearInterval(this.intervalId);
    }
    console.log('\nAuto-scaling stopped');
  }
}

// Run auto-scaling swarm
const swarm = new AutoScalingSwarm(5, 20);
swarm.start(60);

// Stop after 10 minutes (for demo)
setTimeout(() => swarm.stop(), 600000);
```

---

## 4. High-Frequency Trading Swarm

Low-latency swarm optimized for HFT.

```javascript
const {
  initE2bSwarm,
  deployTradingAgent,
  executeSwarmStrategy,
  getSwarmMetrics
} = require('@rUv/neural-trader-backend');

async function highFrequencySwarm() {
  console.log('=== High-Frequency Trading Swarm ===\n');

  // Star topology for centralized coordination (lowest latency)
  const config = JSON.stringify({
    topology: 3,  // Star
    maxAgents: 15,
    distributionStrategy: 0,  // Round-robin for balanced load
    enableGpu: true,
    autoScaling: false,
    maxMemoryMb: 2048,
    timeoutSecs: 60  // Short timeout for HFT
  });

  const swarm = await initE2bSwarm('star', config);
  console.log(`HFT Swarm: ${swarm.swarmId}\n`);

  // Deploy agents for liquid instruments
  const instruments = [
    'SPY',   // S&P 500 ETF
    'QQQ',   // Nasdaq 100 ETF
    'IWM',   // Russell 2000 ETF
    'AAPL', // Apple
    'MSFT', // Microsoft
    'GOOGL' // Google
  ];

  console.log('Deploying HFT agents...\n');

  const agents = [];

  for (const symbol of instruments) {
    const sandbox = await createE2bSandbox(
      `hft-${symbol.toLowerCase()}`,
      'nodejs'
    );

    const agent = await deployTradingAgent(
      sandbox.sandboxId,
      'momentum',  // Fast momentum strategy
      [symbol],
      JSON.stringify({
        lookback: 5,      // Very short lookback
        threshold: 0.001,  // Small threshold for HFT
        max_hold_time: 300, // 5 minutes max hold
        enable_gpu: true
      })
    );

    console.log(`  ${symbol}: ${agent.agentId}`);
    agents.push({ symbol, agent });
  }

  // Execute HFT strategy
  console.log('\n\nExecuting HFT strategy...\n');

  const execution = await executeSwarmStrategy(
    swarm.swarmId,
    'momentum',
    instruments
  );

  console.log(`Execution ID: ${execution.executionId}`);
  console.log(`Agents Used: ${execution.agentsUsed}`);
  console.log(`Status: ${execution.status}\n`);

  // Monitor performance
  console.log('Performance Metrics:\n');

  const metrics = await getSwarmMetrics(swarm.swarmId);

  console.log(`  Throughput: ${metrics.throughput.toFixed(2)} trades/hour`);
  console.log(`  Avg Latency: ${metrics.avgLatency.toFixed(2)}ms`);
  console.log(`  Success Rate: ${metrics.successRate.toFixed(2)}%`);
  console.log(`  Resource Utilization: ${metrics.resourceUtilization.toFixed(2)}%`);

  if (metrics.avgLatency < 100) {
    console.log('\n  ✓ Excellent latency for HFT (<100ms)');
  } else if (metrics.avgLatency < 500) {
    console.log('\n  → Good latency (100-500ms)');
  } else {
    console.log('\n  ⚠ High latency - optimization needed');
  }

  return { swarm, agents, metrics };
}

highFrequencySwarm();
```

---

## 5. Fault-Tolerant Swarm

Resilient swarm with automatic recovery.

```javascript
const {
  initE2bSwarm,
  deployTradingAgent,
  listSwarmAgents,
  getAgentStatus,
  stopSwarmAgent,
  restartSwarmAgent
} = require('@rUv/neural-trader-backend');

class FaultTolerantSwarm {
  constructor(swarmId) {
    this.swarmId = swarmId;
    this.failedAgents = new Set();
    this.restartAttempts = new Map();
    this.maxRestartAttempts = 3;
  }

  async initialize() {
    console.log('=== Fault-Tolerant Swarm ===\n');

    const config = JSON.stringify({
      topology: 1,  // Hierarchical for redundancy
      maxAgents: 12,
      distributionStrategy: 4,  // Adaptive
      enableGpu: true,
      autoScaling: true,
      minAgents: 6
    });

    const swarm = await initE2bSwarm('hierarchical', config);
    this.swarmId = swarm.swarmId;

    console.log(`Swarm initialized: ${this.swarmId}\n`);
  }

  async monitorAgents() {
    console.log(`\n[${new Date().toISOString()}] Health Check:`);

    const agents = await listSwarmAgents(this.swarmId);

    for (const agent of agents) {
      const status = await getAgentStatus(agent.agentId);

      console.log(`\n  Agent: ${agent.agentId}`);
      console.log(`    Status: ${status.status}`);
      console.log(`    CPU: ${status.cpuUsage.toFixed(2)}%`);
      console.log(`    Memory: ${status.memoryUsageMb.toFixed(2)}MB`);
      console.log(`    Active Trades: ${status.activeTrades}`);
      console.log(`    Errors: ${status.errorCount}`);

      // Check for failed agents
      if (status.status === 'failed' || status.errorCount > 10) {
        console.log(`    ⚠ AGENT FAILED - Attempting recovery`);
        await this.recoverAgent(agent.agentId);
      }

      // Check for degraded performance
      else if (status.cpuUsage > 90 || status.memoryUsageMb > 1800) {
        console.log(`    ⚠ DEGRADED PERFORMANCE - Restarting`);
        await this.recoverAgent(agent.agentId);
      }

      // Check for hung agents
      else if (status.activeTrades === 0 && status.status === 'busy') {
        console.log(`    ⚠ POSSIBLE HUNG STATE - Restarting`);
        await this.recoverAgent(agent.agentId);
      }
    }
  }

  async recoverAgent(agentId) {
    // Check restart attempts
    const attempts = this.restartAttempts.get(agentId) || 0;

    if (attempts >= this.maxRestartAttempts) {
      console.log(`      ✗ Max restart attempts reached - marking as permanently failed`);
      this.failedAgents.add(agentId);
      return;
    }

    try {
      // Stop the agent
      console.log(`      Stopping agent...`);
      await stopSwarmAgent(agentId);

      // Wait a bit
      await new Promise(resolve => setTimeout(resolve, 2000));

      // Restart
      console.log(`      Restarting agent...`);
      const deployment = await restartSwarmAgent(agentId);

      console.log(`      ✓ Agent restarted: ${deployment.agentId}`);
      console.log(`        Status: ${deployment.status}`);

      // Increment restart counter
      this.restartAttempts.set(agentId, attempts + 1);

    } catch (error) {
      console.log(`      ✗ Recovery failed: ${error.message}`);
      this.failedAgents.add(agentId);
    }
  }

  async getFailureReport() {
    console.log('\n\n=== Failure Report ===\n');

    console.log(`Failed Agents: ${this.failedAgents.size}`);

    if (this.failedAgents.size > 0) {
      console.log('\nPermanently Failed:');
      this.failedAgents.forEach(id => {
        const attempts = this.restartAttempts.get(id);
        console.log(`  - ${id} (${attempts} restart attempts)`);
      });
    }

    console.log('\nRestart Statistics:');
    this.restartAttempts.forEach((attempts, agentId) => {
      if (!this.failedAgents.has(agentId)) {
        console.log(`  - ${agentId}: ${attempts} restarts (recovered)`);
      }
    });
  }

  async start(intervalSeconds = 30) {
    await this.initialize();

    console.log(`Starting fault monitoring (${intervalSeconds}s intervals)...\n`);

    // Continuous monitoring
    this.intervalId = setInterval(async () => {
      try {
        await this.monitorAgents();
      } catch (error) {
        console.error(`Monitor error: ${error.message}`);
      }
    }, intervalSeconds * 1000);
  }

  async stop() {
    if (this.intervalId) {
      clearInterval(this.intervalId);
    }
    await this.getFailureReport();
    console.log('\nFault monitoring stopped');
  }
}

// Run fault-tolerant swarm
const swarm = new FaultTolerantSwarm();
swarm.start(30);

// Stop after 10 minutes
setTimeout(() => swarm.stop(), 600000);
```

---

## 6. Global Market Coverage

24/7 swarm covering global markets.

```javascript
const {
  initE2bSwarm,
  deployTradingAgent,
  executeSwarmStrategy
} = require('@rUv/neural-trader-backend');

async function globalMarketSwarm() {
  console.log('=== Global 24/7 Market Coverage ===\n');

  // Initialize large mesh swarm
  const config = JSON.stringify({
    topology: 0,  // Mesh
    maxAgents: 30,
    distributionStrategy: 2,  // Specialized
    enableGpu: true,
    autoScaling: true,
    minAgents: 15
  });

  const swarm = await initE2bSwarm('mesh', config);
  console.log(`Global Swarm: ${swarm.swarmId}\n`);

  // Define market regions and trading hours (UTC)
  const markets = {
    'Asia': {
      symbols: ['2330.TW', '9984.T', '000001.SS', 'BABA', 'TCEHY'],
      hours: '00:00-08:00 UTC',
      agents: 5
    },
    'Europe': {
      symbols: ['VOD.L', 'SAP.DE', 'MC.PA', 'ASML.AS'],
      hours: '07:00-16:00 UTC',
      agents: 5
    },
    'Americas': {
      symbols: ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA'],
      hours: '13:00-21:00 UTC',
      agents: 5
    },
    'Crypto': {
      symbols: ['BTC-USD', 'ETH-USD', 'BNB-USD'],
      hours: '24/7',
      agents: 3
    },
    'Forex': {
      symbols: ['EUR/USD', 'GBP/USD', 'USD/JPY'],
      hours: '24/7',
      agents: 3
    }
  };

  console.log('Deploying regional agents...\n');

  const deployments = {};

  for (const [region, config] of Object.entries(markets)) {
    console.log(`${region} Market:`);
    console.log(`  Trading Hours: ${config.hours}`);
    console.log(`  Symbols: ${config.symbols.join(', ')}`);

    deployments[region] = [];

    for (let i = 0; i < config.agents; i++) {
      const sandbox = await createE2bSandbox(
        `${region.toLowerCase()}-${i+1}`,
        'nodejs'
      );

      const agent = await deployTradingAgent(
        sandbox.sandboxId,
        'momentum',
        config.symbols,
        JSON.stringify({
          region: region,
          timezone: 'UTC'
        })
      );

      console.log(`  Agent ${i+1}: ${agent.agentId}`);
      deployments[region].push(agent);
    }

    console.log('');
  }

  // Schedule execution based on current time
  const currentHour = new Date().getUTCHours();

  console.log(`\nCurrent UTC Hour: ${currentHour}\n`);
  console.log('Active Markets:\n');

  // Determine which markets are active
  const activeMarkets = [];

  if (currentHour >= 0 && currentHour < 8) {
    activeMarkets.push('Asia', 'Crypto', 'Forex');
  } else if (currentHour >= 7 && currentHour < 13) {
    activeMarkets.push('Europe', 'Crypto', 'Forex');
  } else if (currentHour >= 13 && currentHour < 21) {
    activeMarkets.push('Americas', 'Crypto', 'Forex');
  } else {
    activeMarkets.push('Crypto', 'Forex');
  }

  for (const market of activeMarkets) {
    const marketConfig = markets[market];
    console.log(`  ✓ ${market}: ${marketConfig.symbols.join(', ')}`);

    const execution = await executeSwarmStrategy(
      swarm.swarmId,
      'momentum',
      marketConfig.symbols
    );

    console.log(`    Execution ID: ${execution.executionId}`);
    console.log(`    Agents Used: ${execution.agentsUsed}\n`);
  }

  return { swarm, markets, deployments };
}

globalMarketSwarm();
```

---

## 7. Complete Production Swarm

Full-featured production swarm system.

```javascript
const {
  initE2bSwarm,
  deployTradingAgent,
  executeSwarmStrategy,
  getSwarmStatus,
  getSwarmPerformance,
  monitorSwarmHealth,
  scaleSwarm,
  shutdownSwarm
} = require('@rUv/neural-trader-backend');

class ProductionSwarm {
  constructor() {
    this.swarmId = null;
    this.agents = [];
    this.isRunning = false;
  }

  async initialize() {
    console.log('=== Production Trading Swarm ===\n');

    const config = JSON.stringify({
      topology: 1,  // Hierarchical
      maxAgents: 25,
      distributionStrategy: 4,  // Adaptive
      enableGpu: true,
      autoScaling: true,
      minAgents: 10,
      maxMemoryMb: 2048,
      timeoutSecs: 300
    });

    const swarm = await initE2bSwarm('hierarchical', config);
    this.swarmId = swarm.swarmId;

    console.log(`✓ Swarm initialized: ${this.swarmId}\n`);
  }

  async deployStrategies() {
    console.log('Deploying production strategies...\n');

    const strategies = [
      { name: 'momentum', symbols: ['AAPL', 'MSFT', 'GOOGL'], count: 3 },
      { name: 'mean_reversion', symbols: ['AMZN', 'TSLA'], count: 2 },
      { name: 'pairs', symbols: ['KO', 'PEP'], count: 2 },
      { name: 'neural', symbols: ['SPY', 'QQQ'], count: 2 },
      { name: 'arbitrage', symbols: ['BTC-USD'], count: 1 }
    ];

    for (const strat of strategies) {
      for (let i = 0; i < strat.count; i++) {
        const sandbox = await createE2bSandbox(
          `${strat.name}-${i+1}`,
          'nodejs'
        );

        const agent = await deployTradingAgent(
          sandbox.sandboxId,
          strat.name,
          strat.symbols,
          JSON.stringify({ enable_gpu: true })
        );

        this.agents.push({ strategy: strat.name, agent, sandbox });
        console.log(`✓ ${strat.name} agent ${i+1}: ${agent.agentId}`);
      }
    }

    console.log(`\n✓ Deployed ${this.agents.length} agents\n`);
  }

  async monitorPerformance() {
    const status = await getSwarmStatus(this.swarmId);
    const health = await monitorSwarmHealth();
    const performance = await getSwarmPerformance(this.swarmId);

    console.log(`\n=== Performance Report [${new Date().toISOString()}] ===\n`);

    console.log('Status:');
    console.log(`  Active Agents: ${status.activeAgents}`);
    console.log(`  Total Trades: ${status.totalTrades}`);
    console.log(`  Total P&L: $${status.totalPnl.toLocaleString()}`);

    console.log('\nHealth:');
    console.log(`  System Status: ${health.status}`);
    console.log(`  CPU Usage: ${health.cpuUsage.toFixed(2)}%`);
    console.log(`  Memory Usage: ${health.memoryUsage.toFixed(2)}%`);
    console.log(`  Error Rate: ${health.errorRate.toFixed(2)}/min`);

    console.log('\nPerformance:');
    console.log(`  Total Return: ${performance.totalReturn.toFixed(2)}%`);
    console.log(`  Sharpe Ratio: ${performance.sharpeRatio.toFixed(2)}`);
    console.log(`  Max Drawdown: ${performance.maxDrawdown.toFixed(2)}%`);
    console.log(`  Win Rate: ${performance.winRate.toFixed(2)}%`);
    console.log(`  Profit Factor: ${performance.profitFactor.toFixed(2)}`);

    return { status, health, performance };
  }

  async run() {
    await this.initialize();
    await this.deployStrategies();

    this.isRunning = true;

    console.log('✓ Swarm is operational\n');

    // Monitor every 5 minutes
    this.monitorInterval = setInterval(async () => {
      if (this.isRunning) {
        await this.monitorPerformance();
      }
    }, 300000);

    // Initial performance check
    await this.monitorPerformance();
  }

  async shutdown() {
    console.log('\n\n=== Shutting Down Swarm ===\n');

    this.isRunning = false;

    if (this.monitorInterval) {
      clearInterval(this.monitorInterval);
    }

    // Final performance report
    await this.monitorPerformance();

    // Shutdown swarm
    const result = await shutdownSwarm(this.swarmId);
    console.log(`\n${result}`);
  }
}

// Run production swarm
const swarm = new ProductionSwarm();
swarm.run();

// Graceful shutdown handler
process.on('SIGINT', async () => {
  await swarm.shutdown();
  process.exit(0);
});
```

---

## Best Practices

1. **Choose appropriate topology** for your use case
2. **Enable auto-scaling** for variable loads
3. **Monitor health continuously**
4. **Implement fault tolerance** with automatic recovery
5. **Use GPU acceleration** for compute-intensive strategies
6. **Distribute strategies** across multiple agents
7. **Set resource limits** to prevent runaway costs
8. **Log all executions** for audit trails
9. **Test thoroughly** before production deployment
10. **Have shutdown procedures** for emergencies

---

**Next Steps:**
- Review [Best Practices Guide](../guides/best-practices.md)
- Check [Getting Started Guide](../guides/getting-started.md)
- Explore [Security Guidelines](../guides/security.md)
