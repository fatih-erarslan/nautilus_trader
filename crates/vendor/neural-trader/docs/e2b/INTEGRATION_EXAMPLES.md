# SwarmCoordinator Integration Examples

## Table of Contents

1. [Basic Setup](#basic-setup)
2. [Production Deployment](#production-deployment)
3. [Integration with Claude-Flow MCP](#integration-with-claude-flow-mcp)
4. [Real Trading Scenario](#real-trading-scenario)
5. [Advanced Patterns](#advanced-patterns)

## Basic Setup

### Minimal Configuration

```javascript
const { SwarmCoordinator, TOPOLOGY } = require('./src/e2b/swarm-coordinator');

const coordinator = new SwarmCoordinator({
  topology: TOPOLOGY.MESH,
  e2bApiKey: process.env.E2B_API_KEY
});

await coordinator.initializeSwarm({
  agents: [
    {
      name: 'trader1',
      agent_type: 'momentum_trader',
      symbols: ['SPY'],
      resources: { cpu: 2, memory_mb: 1024, timeout: 3600 }
    }
  ]
});

// Distribute task
await coordinator.distributeTask({
  type: 'analyze',
  symbol: 'SPY',
  data: { timeframe: '1h' }
});
```

## Production Deployment

### Complete Production Setup

```javascript
const { SwarmCoordinator, TOPOLOGY, DISTRIBUTION_STRATEGY } = require('./src/e2b/swarm-coordinator');
const { deploySwarm } = require('./deploy-e2b-swarm');

async function deployProductionSwarm() {
  // Step 1: Prepare deployment configuration
  const deploymentConfig = await deploySwarm();

  // Step 2: Create production coordinator
  const coordinator = new SwarmCoordinator({
    swarmId: deploymentConfig.deployment_id,
    topology: TOPOLOGY.HIERARCHICAL,
    maxAgents: 10,
    distributionStrategy: DISTRIBUTION_STRATEGY.ADAPTIVE,
    e2bApiKey: process.env.E2B_API_KEY,
    quicEnabled: true,
    agentDBUrl: 'quic://production.agentdb.io:8443',

    // Production settings
    consensusThreshold: 0.75,
    syncInterval: 3000,
    healthCheckInterval: 5000,
    rebalanceThreshold: 0.25
  });

  // Step 3: Initialize with production agents
  await coordinator.initializeSwarm({
    agents: [
      // Coordinator tier
      {
        name: 'risk_coordinator',
        agent_type: 'risk_manager',
        symbols: ['ALL'],
        resources: { cpu: 4, memory_mb: 2048, timeout: 14400 }
      },

      // Trading tier
      {
        name: 'momentum_spy',
        agent_type: 'momentum_trader',
        symbols: ['SPY', 'QQQ', 'IWM'],
        resources: { cpu: 2, memory_mb: 1024, timeout: 3600 }
      },
      {
        name: 'neural_tech',
        agent_type: 'neural_forecaster',
        symbols: ['AAPL', 'TSLA', 'NVDA', 'MSFT'],
        resources: { cpu: 8, memory_mb: 4096, timeout: 3600 }
      },
      {
        name: 'mean_reversion_bonds',
        agent_type: 'mean_reversion_trader',
        symbols: ['TLT', 'GLD', 'SLV', 'UUP'],
        resources: { cpu: 2, memory_mb: 1024, timeout: 3600 }
      },

      // Optimization tier
      {
        name: 'portfolio_optimizer',
        agent_type: 'portfolio_optimizer',
        symbols: ['ALL'],
        resources: { cpu: 8, memory_mb: 4096, timeout: 7200 }
      }
    ]
  });

  // Step 4: Set up monitoring
  coordinator.on('agent-offline', async (agent) => {
    console.error(`ALERT: Agent ${agent.id} offline`);
    // Send alert to monitoring system
    await sendAlert({
      type: 'agent_offline',
      agent: agent.id,
      timestamp: Date.now()
    });

    // Trigger rebalancing
    await coordinator.rebalance();
  });

  coordinator.on('rebalanced', (result) => {
    console.log(`Swarm rebalanced: ${result.adjustedAgents} agents`);
  });

  // Step 5: Start production trading loop
  await runProductionTradingLoop(coordinator);

  return coordinator;
}

async function runProductionTradingLoop(coordinator) {
  console.log('Starting production trading loop...');

  while (true) {
    try {
      // Market analysis cycle
      const analysisTask = {
        type: 'market_analysis',
        symbols: ['SPY', 'QQQ', 'AAPL', 'TSLA', 'TLT'],
        data: {
          timeframe: '5m',
          indicators: ['RSI', 'MACD', 'BB', 'VWAP']
        }
      };

      await coordinator.distributeTask(analysisTask, DISTRIBUTION_STRATEGY.ADAPTIVE);

      // Risk assessment
      const riskTask = {
        type: 'risk_assessment',
        requiredCapability: 'risk_assessment',
        data: {
          portfolio: await getCurrentPortfolio(),
          var_confidence: 0.99
        }
      };

      await coordinator.distributeTask(riskTask, DISTRIBUTION_STRATEGY.SPECIALIZED);

      // Check for trading opportunities
      const opportunities = await checkTradingOpportunities();

      if (opportunities.length > 0) {
        // Require consensus for trade decisions
        for (const opp of opportunities) {
          const tradeTask = {
            type: 'trade_decision',
            requireConsensus: true,
            symbol: opp.symbol,
            data: {
              action: opp.action,
              quantity: opp.quantity,
              currentPrice: opp.price,
              confidence_threshold: 0.80
            }
          };

          const result = await coordinator.distributeTask(tradeTask, DISTRIBUTION_STRATEGY.CONSENSUS);

          // Wait for consensus
          await sleep(5000);

          const consensus = await coordinator.collectResults(result.taskId);

          if (consensus.consensus.achieved) {
            console.log(`✅ Consensus reached for ${opp.symbol}: ${consensus.consensus.decision}`);
            // Execute trade
            await executeTrade(consensus);
          } else {
            console.log(`❌ No consensus for ${opp.symbol}, skipping trade`);
          }
        }
      }

      // Performance monitoring
      const status = coordinator.getStatus();
      console.log(`Status: ${status.agents.ready}/${status.agents.total} agents ready, ${status.tasks.completed} tasks completed`);

      // Wait for next cycle
      await sleep(60000); // 1 minute

    } catch (error) {
      console.error('Trading loop error:', error);
      await sleep(5000); // Wait before retry
    }
  }
}
```

## Integration with Claude-Flow MCP

### Using MCP Tools with SwarmCoordinator

```javascript
const { SwarmCoordinator, TOPOLOGY } = require('./src/e2b/swarm-coordinator');

async function integrateWithClaudeFlow() {
  // Step 1: Initialize swarm via MCP
  const swarmInit = await mcp__claude_flow__swarm_init({
    topology: 'hierarchical',
    maxAgents: 8,
    strategy: 'auto'
  });

  console.log('MCP swarm initialized:', swarmInit);

  // Step 2: Create SwarmCoordinator with same topology
  const coordinator = new SwarmCoordinator({
    swarmId: swarmInit.swarm_id,
    topology: TOPOLOGY.HIERARCHICAL,
    maxAgents: 8,
    e2bApiKey: process.env.E2B_API_KEY,
    quicEnabled: true
  });

  // Step 3: Spawn agents via MCP for coordination
  await mcp__claude_flow__agent_spawn({
    type: 'coordinator',
    swarmId: swarmInit.swarm_id
  });

  await mcp__claude_flow__agent_spawn({
    type: 'coder',
    swarmId: swarmInit.swarm_id
  });

  // Step 4: Initialize SwarmCoordinator with actual E2B agents
  await coordinator.initializeSwarm({
    agents: [
      {
        name: 'coordinator_agent',
        agent_type: 'risk_manager',
        symbols: ['ALL'],
        resources: { cpu: 4, memory_mb: 2048, timeout: 7200 }
      },
      {
        name: 'trader_agent',
        agent_type: 'momentum_trader',
        symbols: ['SPY', 'QQQ'],
        resources: { cpu: 2, memory_mb: 1024, timeout: 3600 }
      }
    ]
  });

  // Step 5: Orchestrate task via MCP
  const taskOrchestration = await mcp__claude_flow__task_orchestrate({
    task: 'Analyze market conditions and execute trades',
    strategy: 'adaptive',
    priority: 'high'
  });

  // Step 6: Distribute to actual agents via SwarmCoordinator
  const result = await coordinator.distributeTask({
    type: 'market_analysis',
    mcpTaskId: taskOrchestration.task_id,
    data: { timeframe: '1h', indicators: ['RSI', 'MACD'] }
  });

  // Step 7: Store coordination state in MCP memory
  await mcp__claude_flow__memory_usage({
    action: 'store',
    key: 'swarm/e2b/state',
    namespace: 'coordination',
    value: JSON.stringify(coordinator.getStatus())
  });

  // Step 8: Monitor via MCP
  const swarmStatus = await mcp__claude_flow__swarm_status({
    swarmId: swarmInit.swarm_id
  });

  console.log('MCP swarm status:', swarmStatus);

  return { coordinator, mcpSwarmId: swarmInit.swarm_id };
}
```

### Using Hooks for Coordination

```bash
#!/bin/bash

# Pre-task: Initialize coordination
npx claude-flow@alpha hooks pre-task \
  --description "Deploying multi-agent trading swarm to E2B"

# During deployment: Report progress
npx claude-flow@alpha hooks notify \
  --message "Swarm initialized with 5 agents" \
  --level "info"

# After each agent deployment
npx claude-flow@alpha hooks post-edit \
  --file "src/e2b/swarm-coordinator.js" \
  --memory-key "swarm/e2b/coordinator/agent-1"

# Post-task: Complete coordination
npx claude-flow@alpha hooks post-task \
  --task-id "swarm-deployment-123" \
  --export-metrics true

# Session management
npx claude-flow@alpha hooks session-end \
  --session-id "trading-session-1" \
  --export-metrics true
```

## Real Trading Scenario

### Day Trading Strategy with Consensus

```javascript
const { SwarmCoordinator, TOPOLOGY, DISTRIBUTION_STRATEGY } = require('./src/e2b/swarm-coordinator');

async function runDayTradingStrategy() {
  const coordinator = new SwarmCoordinator({
    topology: TOPOLOGY.MESH,
    distributionStrategy: DISTRIBUTION_STRATEGY.CONSENSUS,
    consensusThreshold: 0.75,
    e2bApiKey: process.env.E2B_API_KEY
  });

  await coordinator.initializeSwarm({
    agents: [
      {
        name: 'momentum_1',
        agent_type: 'momentum_trader',
        symbols: ['SPY'],
        resources: { cpu: 2, memory_mb: 1024, timeout: 3600 }
      },
      {
        name: 'neural_1',
        agent_type: 'neural_forecaster',
        symbols: ['SPY'],
        resources: { cpu: 4, memory_mb: 2048, timeout: 3600 }
      },
      {
        name: 'mean_rev_1',
        agent_type: 'mean_reversion_trader',
        symbols: ['SPY'],
        resources: { cpu: 2, memory_mb: 1024, timeout: 3600 }
      }
    ]
  });

  // Market hours: 9:30 AM - 4:00 PM EST
  const marketOpen = 9.5 * 60 * 60 * 1000;  // 9:30 AM in ms
  const marketClose = 16 * 60 * 60 * 1000;  // 4:00 PM in ms

  while (isMarketHours(marketOpen, marketClose)) {
    // Analyze current market
    const analysis = {
      type: 'intraday_analysis',
      symbol: 'SPY',
      data: {
        timeframe: '5m',
        lookback: 50,
        indicators: ['RSI', 'MACD', 'VWAP', 'Volume']
      }
    };

    await coordinator.distributeTask(analysis);

    // Get signals from all strategies
    const signals = await getSignalsFromAgents(coordinator);

    // Check for consensus
    if (signals.length >= 2 && allAgree(signals)) {
      const tradeDecision = {
        type: 'trade_decision',
        requireConsensus: true,
        symbol: 'SPY',
        data: {
          signal: signals[0].action,
          price: await getCurrentPrice('SPY'),
          quantity: calculatePositionSize()
        }
      };

      const result = await coordinator.distributeTask(
        tradeDecision,
        DISTRIBUTION_STRATEGY.CONSENSUS
      );

      await sleep(3000);

      const consensus = await coordinator.collectResults(result.taskId);

      if (consensus.consensus.achieved) {
        console.log(`Executing trade: ${consensus.consensus.decision}`);
        await executeIntraday Trade(consensus);
      }
    }

    // Wait 5 minutes
    await sleep(5 * 60 * 1000);
  }

  // End of day: Close all positions
  await closeAllPositions(coordinator);

  // Generate daily report
  const report = coordinator.getStatus();
  console.log('Daily Trading Report:', report);

  await coordinator.shutdown();
}
```

## Advanced Patterns

### Multi-Stage Pipeline

```javascript
async function runMultiStagePipeline(coordinator) {
  // Stage 1: Data collection
  const dataTask = {
    type: 'data_collection',
    data: {
      symbols: ['SPY', 'QQQ', 'AAPL'],
      sources: ['market_data', 'news', 'social_sentiment']
    }
  };

  const dataResult = await coordinator.distributeTask(
    dataTask,
    DISTRIBUTION_STRATEGY.SPECIALIZED
  );

  // Wait for completion
  await sleep(5000);
  const collectedData = await coordinator.collectResults(dataResult.taskId);

  // Stage 2: Feature engineering
  const featureTask = {
    type: 'feature_engineering',
    requiredCapability: 'pattern_recognition',
    data: {
      rawData: collectedData,
      features: ['technical', 'sentiment', 'fundamental']
    }
  };

  const featureResult = await coordinator.distributeTask(
    featureTask,
    DISTRIBUTION_STRATEGY.SPECIALIZED
  );

  await sleep(5000);
  const features = await coordinator.collectResults(featureResult.taskId);

  // Stage 3: Prediction (parallel from multiple models)
  const predictionTask = {
    type: 'prediction',
    requireConsensus: true,
    data: {
      features: features,
      models: ['lstm', 'transformer', 'xgboost']
    }
  };

  const predictionResult = await coordinator.distributeTask(
    predictionTask,
    DISTRIBUTION_STRATEGY.CONSENSUS
  );

  await sleep(10000);
  const predictions = await coordinator.collectResults(predictionResult.taskId);

  // Stage 4: Risk assessment
  const riskTask = {
    type: 'risk_assessment',
    requiredCapability: 'risk_assessment',
    data: {
      predictions: predictions,
      portfolio: await getCurrentPortfolio()
    }
  };

  const riskResult = await coordinator.distributeTask(
    riskTask,
    DISTRIBUTION_STRATEGY.SPECIALIZED
  );

  await sleep(3000);
  const riskAssessment = await coordinator.collectResults(riskResult.taskId);

  // Stage 5: Execution (if approved)
  if (riskAssessment.approved) {
    const executionTask = {
      type: 'trade_execution',
      data: {
        trades: riskAssessment.recommendedTrades
      }
    };

    await coordinator.distributeTask(
      executionTask,
      DISTRIBUTION_STRATEGY.LEAST_LOADED
    );
  }
}
```

### Adaptive Rebalancing with Triggers

```javascript
async function setupAdaptiveRebalancing(coordinator) {
  // Monitor load continuously
  coordinator.on('state-synchronized', async (state) => {
    const agents = state.agents;

    // Calculate load statistics
    const loads = agents.map(a => a.load);
    const avgLoad = loads.reduce((a, b) => a + b, 0) / loads.length;
    const maxLoad = Math.max(...loads);
    const minLoad = Math.min(...loads);

    const imbalance = maxLoad - minLoad;

    // Trigger rebalancing if needed
    if (imbalance > 0.3) {
      console.log(`Load imbalance detected: ${(imbalance * 100).toFixed(1)}%`);
      await coordinator.rebalance();
    }

    // Scale up if all agents are busy
    const busyRatio = agents.filter(a => a.state === 'busy').length / agents.length;

    if (busyRatio > 0.8 && agents.length < coordinator.maxAgents) {
      console.log('High utilization detected, scaling up...');
      await scaleUpSwarm(coordinator);
    }

    // Scale down if underutilized
    if (avgLoad < 0.2 && agents.length > 3) {
      console.log('Low utilization detected, scaling down...');
      await scaleDownSwarm(coordinator);
    }
  });
}

async function scaleUpSwarm(coordinator) {
  // Add new agent dynamically
  await coordinator.deployAgent({
    name: `dynamic_agent_${Date.now()}`,
    agent_type: 'momentum_trader',
    symbols: ['SPY'],
    resources: { cpu: 2, memory_mb: 1024, timeout: 3600 }
  });

  // Re-establish topology
  await coordinator.establishTopology();
}
```

## Helper Functions

```javascript
function isMarketHours(open, close) {
  const now = new Date();
  const currentTime = now.getHours() * 60 * 60 * 1000 + now.getMinutes() * 60 * 1000;
  return currentTime >= open && currentTime <= close;
}

function allAgree(signals) {
  if (signals.length < 2) return false;
  const firstAction = signals[0].action;
  return signals.every(s => s.action === firstAction);
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function getCurrentPrice(symbol) {
  // Implementation to fetch current price
  return 450.00; // Mock
}

function calculatePositionSize() {
  // Implementation for position sizing
  return 10; // Mock
}
```

## Best Practices Summary

1. **Always use consensus for critical trading decisions**
2. **Monitor agent health continuously**
3. **Store coordination state in MCP memory**
4. **Use specialized routing for capability-specific tasks**
5. **Implement proper error handling and recovery**
6. **Set appropriate timeouts for different operation types**
7. **Use adaptive strategies for dynamic workloads**
8. **Track metrics for continuous optimization**

## Troubleshooting

See [SWARM_COORDINATION_GUIDE.md](./SWARM_COORDINATION_GUIDE.md#troubleshooting) for detailed troubleshooting steps.
