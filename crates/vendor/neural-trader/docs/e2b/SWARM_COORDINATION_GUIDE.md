# SwarmCoordinator - Multi-Agent Trading Orchestration Guide

## Overview

The `SwarmCoordinator` provides enterprise-grade orchestration for distributed trading agents across E2B sandboxes. It implements advanced coordination patterns including mesh/hierarchical topologies, consensus mechanisms, intelligent load balancing, and self-healing capabilities.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    SwarmCoordinator                          │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Topology   │  │    Task      │  │   Consensus  │     │
│  │  Management  │  │ Distribution │  │  Mechanisms  │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Health     │  │  Performance │  │    Memory    │     │
│  │  Monitoring  │  │   Tracking   │  │ Coordination │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
           │                   │                   │
           ▼                   ▼                   ▼
    ┌──────────┐        ┌──────────┐        ┌──────────┐
    │ E2B      │        │ AgentDB  │        │ Claude   │
    │ Sandbox  │        │ QUIC     │        │ Flow MCP │
    └──────────┘        └──────────┘        └──────────┘
```

## Key Features

### 1. **Topology Management**

Support for multiple coordination patterns:

- **Mesh**: Full connectivity for maximum coordination
- **Hierarchical**: Tree structure with coordinator nodes
- **Ring**: Circular communication for sequential processing
- **Star**: Central hub for centralized control

### 2. **Intelligent Task Distribution**

Multiple distribution strategies:

- **Round Robin**: Simple load distribution
- **Least Loaded**: Route to agents with lowest load
- **Specialized**: Capability-based routing
- **Consensus**: Multi-agent decision making
- **Adaptive**: ML-based dynamic routing

### 3. **Consensus Mechanisms**

Trade decision consensus with configurable thresholds:
- Majority voting
- Weighted confidence scoring
- Byzantine fault tolerance
- Quorum-based decisions

### 4. **Self-Healing**

Automatic recovery and optimization:
- Agent health monitoring
- Automatic rebalancing
- Failed task redistribution
- Dynamic topology adaptation

## Quick Start

### Basic Usage

```javascript
const { SwarmCoordinator, TOPOLOGY, DISTRIBUTION_STRATEGY } = require('./src/e2b/swarm-coordinator');

// Create coordinator
const coordinator = new SwarmCoordinator({
  swarmId: 'my-trading-swarm',
  topology: TOPOLOGY.MESH,
  maxAgents: 5,
  distributionStrategy: DISTRIBUTION_STRATEGY.ADAPTIVE,
  e2bApiKey: process.env.E2B_API_KEY
});

// Initialize swarm
await coordinator.initializeSwarm({
  agents: [
    {
      name: 'momentum_trader',
      agent_type: 'momentum_trader',
      symbols: ['SPY', 'QQQ'],
      resources: { cpu: 2, memory_mb: 1024, timeout: 3600 }
    },
    {
      name: 'neural_forecaster',
      agent_type: 'neural_forecaster',
      symbols: ['AAPL', 'TSLA'],
      resources: { cpu: 4, memory_mb: 2048, timeout: 3600 }
    }
  ]
});

// Distribute task
const task = {
  type: 'analyze',
  symbol: 'SPY',
  data: { timeframe: '1h', indicators: ['RSI', 'MACD'] }
};

const result = await coordinator.distributeTask(task);
console.log('Task assigned to:', result.assignedAgents);

// Collect results
const results = await coordinator.collectResults(result.taskId);
console.log('Results:', results);

// Get status
const status = coordinator.getStatus();
console.log('Swarm status:', status);

// Shutdown
await coordinator.shutdown();
```

### Advanced: Consensus-Based Trading

```javascript
const coordinator = new SwarmCoordinator({
  topology: TOPOLOGY.MESH,
  distributionStrategy: DISTRIBUTION_STRATEGY.CONSENSUS,
  consensusThreshold: 0.75  // 75% agreement required
});

await coordinator.initializeSwarm({ agents: [...] });

// Distribute trade decision to multiple agents
const tradeTask = {
  type: 'trade_decision',
  requireConsensus: true,
  symbol: 'AAPL',
  data: {
    currentPrice: 185.50,
    action: 'buy',
    quantity: 100
  }
};

const result = await coordinator.distributeTask(tradeTask);

// Wait for consensus
const consensus = await coordinator.collectResults(result.taskId);

if (consensus.consensus.achieved) {
  console.log('Consensus reached:', consensus.consensus.decision);
  console.log('Agreement:', consensus.consensus.agreement);
  // Execute trade
} else {
  console.log('No consensus, holding position');
}
```

### Integration with AgentDB

```javascript
const coordinator = new SwarmCoordinator({
  topology: TOPOLOGY.MESH,
  quicEnabled: true,  // Enable QUIC synchronization
  agentDBUrl: 'quic://localhost:8443'
});

await coordinator.initializeSwarm({ agents: [...] });

// State automatically syncs to AgentDB every 5 seconds
// Agents can query coordination actions via RL
```

### Event-Driven Coordination

```javascript
const coordinator = new SwarmCoordinator({ ... });

// Listen to events
coordinator.on('initialized', (info) => {
  console.log('Swarm initialized:', info);
});

coordinator.on('task-distributed', (result) => {
  console.log('Task distributed:', result.taskId);
});

coordinator.on('agent-offline', (agent) => {
  console.warn('Agent offline:', agent.id);
  // Trigger recovery
});

coordinator.on('rebalanced', (result) => {
  console.log('Swarm rebalanced:', result);
});

coordinator.on('state-synchronized', (state) => {
  console.log('State synced:', state.timestamp);
});

await coordinator.initializeSwarm({ agents: [...] });
```

## Configuration Options

```javascript
{
  // Core settings
  swarmId: 'custom-swarm-id',
  topology: TOPOLOGY.MESH,
  maxAgents: 10,
  distributionStrategy: DISTRIBUTION_STRATEGY.ADAPTIVE,

  // E2B integration
  e2bApiKey: process.env.E2B_API_KEY,

  // AgentDB integration
  quicEnabled: true,
  agentDBUrl: 'quic://localhost:8443',

  // Coordination settings
  consensusThreshold: 0.66,      // 66% agreement for consensus
  syncInterval: 5000,            // State sync every 5 seconds
  healthCheckInterval: 10000,    // Health check every 10 seconds
  rebalanceThreshold: 0.3        // Rebalance at 30% load imbalance
}
```

## API Reference

### Core Methods

#### `initializeSwarm(config)`
Initialize swarm with agent configurations.

```javascript
await coordinator.initializeSwarm({
  agents: [
    {
      name: 'agent_name',
      agent_type: 'momentum_trader',
      symbols: ['SPY'],
      resources: { cpu: 2, memory_mb: 1024, timeout: 3600 }
    }
  ]
});
```

#### `distributeTask(task, strategy)`
Distribute task to agents using specified strategy.

```javascript
const result = await coordinator.distributeTask({
  type: 'analyze',
  symbol: 'SPY',
  data: { ... }
}, DISTRIBUTION_STRATEGY.ADAPTIVE);
```

#### `collectResults(taskId)`
Collect and aggregate results from agents.

```javascript
const results = await coordinator.collectResults(taskId);
```

#### `synchronizeState()`
Manually trigger state synchronization.

```javascript
const snapshot = await coordinator.synchronizeState();
```

#### `rebalance()`
Manually trigger load rebalancing.

```javascript
const result = await coordinator.rebalance();
```

#### `getStatus()`
Get current swarm status and metrics.

```javascript
const status = coordinator.getStatus();
console.log(status.agents.ready, 'ready agents');
console.log(status.performance.throughput, 'tasks/sec');
```

#### `shutdown()`
Gracefully shutdown swarm.

```javascript
await coordinator.shutdown();
```

## Agent Types

The coordinator supports these specialized agent types:

- **neural_forecaster**: Neural network-based forecasting
- **momentum_trader**: Momentum-based trading
- **mean_reversion_trader**: Mean reversion strategies
- **risk_manager**: Portfolio risk management
- **portfolio_optimizer**: Portfolio optimization

Each agent type has specific capabilities used for intelligent task routing.

## Performance Monitoring

The coordinator tracks comprehensive metrics:

```javascript
const status = coordinator.getStatus();

// Agent metrics
console.log('Total agents:', status.agents.total);
console.log('Ready agents:', status.agents.ready);
console.log('Busy agents:', status.agents.busy);

// Task metrics
console.log('Tasks distributed:', status.tasks.distributed);
console.log('Tasks completed:', status.tasks.completed);
console.log('Success rate:', status.tasks.successRate);

// Performance metrics
console.log('Avg latency:', status.performance.avgLatency, 'ms');
console.log('Throughput:', status.performance.throughput);
console.log('Uptime:', status.performance.uptime);
console.log('Rebalance events:', status.performance.rebalanceEvents);
```

## Integration with Claude-Flow Hooks

```bash
# Before task
npx claude-flow@alpha hooks pre-task --description "Initializing swarm coordination"

# After task updates
npx claude-flow@alpha hooks post-edit --file "src/e2b/swarm-coordinator.js" --memory-key "swarm/e2b/coordinator"

# Store coordination state
npx claude-flow@alpha hooks notify --message "Swarm initialized with 5 agents"

# After task completion
npx claude-flow@alpha hooks post-task --task-id "swarm-deployment"
```

## Best Practices

### 1. **Topology Selection**

- **Mesh**: Best for small swarms (<10 agents) requiring high coordination
- **Hierarchical**: Best for large swarms (>10 agents) with coordinator/worker pattern
- **Ring**: Best for sequential processing pipelines
- **Star**: Best for centralized control and monitoring

### 2. **Distribution Strategy**

- **Round Robin**: Simple, low overhead, predictable
- **Least Loaded**: Best for varying task complexity
- **Specialized**: Best when agents have distinct capabilities
- **Consensus**: Required for critical trading decisions
- **Adaptive**: Best for dynamic workloads (ML-based)

### 3. **Consensus Configuration**

- **Low threshold (0.51-0.60)**: Fast decisions, lower confidence
- **Medium threshold (0.66-0.75)**: Balanced approach (recommended)
- **High threshold (0.80-1.0)**: High confidence, slower decisions

### 4. **Resource Management**

- Set appropriate `syncInterval` based on coordination needs
- Use `healthCheckInterval` to detect failures quickly
- Configure `rebalanceThreshold` to avoid over-rebalancing

## Troubleshooting

### Agent Not Responding

```javascript
// Check agent status
const status = coordinator.getStatus();
console.log('Offline agents:', status.agents.offline);

// Manual rebalance
await coordinator.rebalance();
```

### Task Distribution Failing

```javascript
// Check ready agents
const readyAgents = Array.from(coordinator.agents.values())
  .filter(a => a.state === 'ready');

if (readyAgents.length === 0) {
  console.error('No agents available');
  // Wait for agents to initialize
}
```

### High Latency

```javascript
// Check performance metrics
const status = coordinator.getStatus();
console.log('Avg latency:', status.performance.avgLatency);

// Trigger rebalancing
if (status.performance.avgLatency > 5000) {
  await coordinator.rebalance();
}
```

## Examples

See `/tests/e2b/swarm-coordinator.test.js` for comprehensive examples.

## Support

For issues and questions:
- GitHub Issues: https://github.com/ruvnet/neural-trader/issues
- Documentation: /docs/e2b/

## License

MIT OR Apache-2.0
