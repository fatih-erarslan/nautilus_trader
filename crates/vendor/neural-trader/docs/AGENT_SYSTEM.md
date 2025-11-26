# Neural Trader - Multi-Agent Coordination System

A comprehensive multi-agent coordination system with swarm intelligence for algorithmic trading strategies.

## Overview

The Neural Trader agent system enables you to deploy and coordinate multiple trading agents that work together using swarm intelligence. Each agent specializes in a specific trading strategy or function, and they can communicate, share resources, and make collective decisions.

## Features

- **7 Agent Types**: Momentum, pairs trading, mean reversion, portfolio optimization, risk management, news trading, market making
- **4 Swarm Strategies**: Multi-strategy, adaptive portfolio, high-frequency, risk-aware
- **Inter-Agent Communication**: Message passing, consensus mechanisms, and coordination protocols
- **Resource Management**: Load balancing, auto-scaling, and intelligent task distribution
- **Health Monitoring**: Auto-restart, fault tolerance, and performance tracking
- **Real-Time Dashboard**: Terminal-based coordination monitoring

## Quick Start

### 1. Spawn a Single Agent

```bash
# Spawn a momentum trading agent
neural-trader agent spawn momentum

# Spawn with custom configuration
neural-trader agent spawn pairs-trading --config '{"entry_threshold": 2.5}'
```

### 2. Deploy a Multi-Agent Swarm

```bash
# Deploy multi-strategy swarm (momentum + mean-reversion + pairs-trading)
neural-trader agent swarm multi-strategy

# Deploy adaptive portfolio swarm with risk management
neural-trader agent swarm adaptive-portfolio

# Deploy high-frequency trading swarm
neural-trader agent swarm high-frequency
```

### 3. Monitor and Manage

```bash
# View all running agents
neural-trader agent list

# Show coordination dashboard
neural-trader agent coordinate

# Get detailed agent status
neural-trader agent status <agent-id>

# View agent logs
neural-trader agent logs <agent-id>

# Stop specific agent
neural-trader agent stop <agent-id>

# Stop all agents
neural-trader agent stopall --force
```

## Agent Types

### Trading Agents

#### Momentum Trading
Trades based on price momentum and trend strength.

**Configuration:**
```json
{
  "lookback_period": 20,
  "momentum_threshold": 0.02,
  "stop_loss": 0.05,
  "take_profit": 0.10,
  "position_size": 0.1
}
```

#### Pairs Trading
Statistical arbitrage between correlated pairs.

**Configuration:**
```json
{
  "lookback_period": 60,
  "entry_threshold": 2.0,
  "exit_threshold": 0.5,
  "cointegration_test": "adf",
  "hedge_ratio_method": "ols"
}
```

#### Mean Reversion
Trades mean reversion opportunities.

**Configuration:**
```json
{
  "lookback_period": 20,
  "entry_std": 2.0,
  "exit_std": 0.5,
  "bollinger_bands": true,
  "rsi_threshold": 30
}
```

#### News Trading
Sentiment-driven trading with NLP analysis.

**Configuration:**
```json
{
  "sentiment_threshold": 0.6,
  "news_sources": ["bloomberg", "reuters", "twitter"],
  "event_types": ["earnings", "economic", "political"],
  "reaction_time": 5000,
  "sentiment_model": "transformer"
}
```

#### Market Making
Provides liquidity and captures spread.

**Configuration:**
```json
{
  "spread_target": 0.001,
  "inventory_limit": 1000,
  "quote_size": 100,
  "skew_factor": 0.5,
  "quote_frequency": 1000
}
```

### Portfolio & Risk Agents

#### Portfolio Optimization
Optimizes portfolio allocation and rebalancing.

**Configuration:**
```json
{
  "optimization_method": "mean-variance",
  "rebalance_frequency": "weekly",
  "max_position_weight": 0.2,
  "min_position_weight": 0.05,
  "risk_target": 0.15
}
```

#### Risk Management
Monitors and manages portfolio risk.

**Configuration:**
```json
{
  "max_portfolio_var": 0.02,
  "max_position_size": 0.1,
  "max_correlation": 0.7,
  "stress_test_frequency": "daily",
  "var_confidence": 0.95
}
```

## Swarm Strategies

### Multi-Strategy
Coordinates multiple trading strategies with weighted voting.

**Agents:** Momentum, Mean Reversion, Pairs Trading
**Topology:** Hierarchical
**Coordination:** Weighted voting

```bash
neural-trader agent swarm multi-strategy
```

### Adaptive Portfolio
Self-optimizing portfolio with comprehensive risk management.

**Agents:** Portfolio, Risk Manager, Momentum, Mean Reversion
**Topology:** Mesh
**Coordination:** Consensus

```bash
neural-trader agent swarm adaptive-portfolio
```

### High-Frequency
Ultra-fast trading with market making and news analysis.

**Agents:** Market Maker, Momentum, News Trader
**Topology:** Pipeline
**Coordination:** Streaming

```bash
neural-trader agent swarm high-frequency
```

### Risk-Aware
Conservative trading with comprehensive risk controls.

**Agents:** Risk Manager, Momentum, Mean Reversion, Portfolio
**Topology:** Hierarchical
**Coordination:** Guardian (risk-first)

```bash
neural-trader agent swarm risk-aware
```

## Architecture

### Core Components

1. **Agent Manager** (`src/cli/lib/agent-manager.js`)
   - Lifecycle management (spawn, stop, restart)
   - Health monitoring
   - Auto-restart on failures
   - Metrics tracking

2. **Agent Coordinator** (`src/cli/lib/agent-coordinator.js`)
   - Inter-agent communication
   - Message passing
   - Consensus mechanisms
   - Channel management

3. **Swarm Orchestrator** (`src/cli/lib/swarm-orchestrator.js`)
   - Swarm deployment
   - Topology management
   - Agent scaling
   - Coordination protocols

4. **Load Balancer** (`src/cli/lib/load-balancer.js`)
   - Task distribution
   - Resource allocation
   - Auto-scaling
   - Performance optimization

5. **Agent Registry** (`src/cli/lib/agent-registry.js`)
   - Agent type definitions
   - Configuration schemas
   - Swarm strategy templates

## Communication Patterns

### Direct Messaging
```javascript
// Agent A sends message to Agent B
coordinator.sendDirect(agentA, agentB, {
  type: 'signal',
  data: { action: 'buy', symbol: 'AAPL', quantity: 100 }
});
```

### Broadcast
```javascript
// Agent broadcasts to all agents
coordinator.broadcast(agentId, {
  type: 'alert',
  data: { message: 'High volatility detected' }
});
```

### Consensus
```javascript
// Request consensus from swarm
const result = await coordinator.requestConsensus(
  'trade-proposal-123',
  { action: 'buy', symbol: 'AAPL', quantity: 1000 },
  [agent1, agent2, agent3]
);
```

## Coordination Topologies

### Mesh Topology
All agents are fully connected. Best for small swarms requiring high coordination.

```
    A --- B
    |  X  |
    C --- D
```

### Hierarchical Topology
One coordinator agent manages worker agents. Best for centralized control.

```
       A (Coordinator)
      /|\
     B C D (Workers)
```

### Pipeline Topology
Sequential processing chain. Best for streaming data processing.

```
A → B → C → D
```

## Integration

### With agentic-flow
```javascript
// Use agentic-flow for enhanced coordination
const { AgenticFlow } = require('agentic-flow');

const flow = new AgenticFlow({
  orchestrator: orchestrator,
  coordination: 'consensus'
});
```

### With AgentDB
```javascript
// Persist agent state with AgentDB
const { AgentDB } = require('agentdb');

const db = new AgentDB();
await db.store('agent-state', agentId, agent.state);
```

### With MCP Tools
```bash
# Use MCP tools for advanced operations
neural-trader mcp agent-spawn momentum
neural-trader mcp swarm-deploy multi-strategy
```

## Configuration

### Agent Configuration File
Create `agent-config.json`:

```json
{
  "agent": {
    "type": "momentum",
    "name": "momentum-trader-1",
    "config": {
      "lookback_period": 20,
      "momentum_threshold": 0.02,
      "stop_loss": 0.05,
      "take_profit": 0.10,
      "position_size": 0.1
    }
  },
  "metadata": {
    "version": "1.0.0",
    "created": "2025-11-17T00:00:00Z"
  }
}
```

Load configuration:
```bash
neural-trader agent spawn momentum --config-file agent-config.json
```

## Monitoring

### Real-Time Dashboard
```bash
neural-trader agent coordinate
```

Displays:
- Active swarms and status
- Running agents by type
- Coordination statistics
- Performance metrics
- Recent activity

### Agent Status
```bash
neural-trader agent status <agent-id>
```

Shows:
- Basic information
- Runtime and uptime
- Performance metrics
- Last errors
- Success rate

### Agent Logs
```bash
neural-trader agent logs <agent-id> --limit 100 --level error
```

Options:
- `--limit <n>`: Show last n entries
- `--level <lvl>`: Filter by level (error, warn, info, debug)
- `--verbose`: Show full stack traces

## Best Practices

1. **Start Small**: Begin with single agents before deploying swarms
2. **Monitor Health**: Use the coordinate command to monitor agent health
3. **Set Limits**: Configure appropriate resource limits for agents
4. **Use Risk Management**: Always include a risk-manager agent in production
5. **Test Strategies**: Backtest individual agents before swarm deployment
6. **Handle Failures**: Configure auto-restart and fault tolerance
7. **Scale Gradually**: Add agents incrementally to swarms
8. **Log Everything**: Enable detailed logging for debugging

## Troubleshooting

### Agent Won't Start
```bash
# Check agent configuration
neural-trader agent info <type>

# View system status
neural-trader doctor

# Enable debug mode
DEBUG=true neural-trader agent spawn <type>
```

### Swarm Communication Issues
```bash
# Check coordinator status
neural-trader agent coordinate

# Restart failed agents
neural-trader agent stop <id>
neural-trader agent spawn <type>
```

### High Memory Usage
```bash
# Monitor resource usage
neural-trader agent status <id>

# Adjust agent limits
neural-trader agent spawn <type> --config '{"max_memory": 512}'

# Scale down swarm
neural-trader agent stop <id>
```

## Examples

### Example 1: Simple Momentum Trading
```bash
# Spawn momentum agent
neural-trader agent spawn momentum

# Check status
neural-trader agent list

# View performance
neural-trader agent status <momentum-agent-id>
```

### Example 2: Pairs Trading Swarm
```bash
# Deploy multi-strategy swarm with pairs trading
neural-trader agent swarm multi-strategy

# Monitor coordination
neural-trader agent coordinate

# View specific agent
neural-trader agent status <pairs-agent-id>
```

### Example 3: Risk-Managed Portfolio
```bash
# Deploy risk-aware swarm
neural-trader agent swarm risk-aware

# Check risk metrics
neural-trader agent status <risk-manager-id>

# View portfolio allocation
neural-trader agent status <portfolio-agent-id>
```

## API Reference

See individual module documentation:
- [Agent Manager API](./api/agent-manager.md)
- [Agent Coordinator API](./api/agent-coordinator.md)
- [Swarm Orchestrator API](./api/swarm-orchestrator.md)
- [Load Balancer API](./api/load-balancer.md)

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

## License

MIT OR Apache-2.0

## Support

- Issues: https://github.com/ruvnet/neural-trader/issues
- Documentation: https://github.com/ruvnet/neural-trader
- Discord: https://discord.gg/neural-trader
