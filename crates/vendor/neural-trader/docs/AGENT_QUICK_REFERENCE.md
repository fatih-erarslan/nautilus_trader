# Agent System - Quick Reference Card

## Agent Commands

### Spawn Agent
```bash
neural-trader agent spawn <type>
neural-trader agent spawn momentum
neural-trader agent spawn pairs-trading --config '{"entry_threshold": 2.5}'
```

### List Agents
```bash
neural-trader agent list
neural-trader agent list --type momentum
neural-trader agent list --status running
```

### Agent Status
```bash
neural-trader agent status <agent-id>
```

### Agent Logs
```bash
neural-trader agent logs <agent-id>
neural-trader agent logs <agent-id> --limit 100 --level error
```

### Stop Agent
```bash
neural-trader agent stop <agent-id>
neural-trader agent stopall --force
```

### Coordination Dashboard
```bash
neural-trader agent coordinate
```

### Deploy Swarm
```bash
neural-trader agent swarm <strategy>
neural-trader agent swarm multi-strategy
neural-trader agent swarm adaptive-portfolio
```

## Agent Types

| Type | Description | Use Case |
|------|-------------|----------|
| `momentum` | Trend-following | Trending markets |
| `pairs-trading` | Statistical arbitrage | Correlated pairs |
| `mean-reversion` | Mean reversion | Range-bound markets |
| `portfolio` | Portfolio optimization | Asset allocation |
| `risk-manager` | Risk monitoring | Risk control |
| `news-trader` | News sentiment | Event-driven |
| `market-maker` | Liquidity provision | Market making |

## Swarm Strategies

| Strategy | Agents | Topology | Use Case |
|----------|--------|----------|----------|
| `multi-strategy` | 3 agents | Hierarchical | Diversified trading |
| `adaptive-portfolio` | 4 agents | Mesh | Self-optimizing portfolio |
| `high-frequency` | 3 agents | Pipeline | Ultra-fast trading |
| `risk-aware` | 4 agents | Hierarchical | Conservative trading |

## Quick Start

### 1. Single Agent
```bash
# Spawn
neural-trader agent spawn momentum

# Check
neural-trader agent list

# Monitor
neural-trader agent status <id>

# Stop
neural-trader agent stop <id>
```

### 2. Multi-Agent Swarm
```bash
# Deploy
neural-trader agent swarm multi-strategy

# Monitor
neural-trader agent coordinate

# Stop all
neural-trader agent stopall --force
```

## Configuration

### Momentum Agent
```json
{
  "lookback_period": 20,
  "momentum_threshold": 0.02,
  "stop_loss": 0.05,
  "take_profit": 0.10,
  "position_size": 0.1
}
```

### Pairs Trading Agent
```json
{
  "lookback_period": 60,
  "entry_threshold": 2.0,
  "exit_threshold": 0.5,
  "cointegration_test": "adf",
  "hedge_ratio_method": "ols"
}
```

### Risk Manager Agent
```json
{
  "max_portfolio_var": 0.02,
  "max_position_size": 0.1,
  "max_correlation": 0.7,
  "stress_test_frequency": "daily",
  "var_confidence": 0.95
}
```

## Status Indicators

| Icon | Status | Meaning |
|------|--------|---------|
| 🟢 | Running | Agent active and healthy |
| 🔴 | Stopped | Agent stopped |
| 🟡 | Starting/Stopping | Transitioning |
| ❌ | Failed | Agent failed |
| ✓ | Healthy | Passing health checks |
| ✗ | Unhealthy | Failing health checks |

## Troubleshooting

### Agent won't start
```bash
neural-trader agent info <type>
neural-trader doctor
DEBUG=true neural-trader agent spawn <type>
```

### View agent errors
```bash
neural-trader agent logs <id> --level error --verbose
```

### Reset all agents
```bash
neural-trader agent stopall --force
neural-trader agent list  # Verify all stopped
```

## Advanced

### Custom Configuration File
```bash
# Create config
echo '{"lookback_period": 30}' > momentum-config.json

# Use config
neural-trader agent spawn momentum --config-file momentum-config.json
```

### Filter Agents
```bash
# By type
neural-trader agent list --type momentum

# By status
neural-trader agent list --status running

# By health
neural-trader agent list --health unhealthy
```

### Watch Mode
```bash
# Real-time dashboard
neural-trader agent coordinate --watch
```

## Integration

### With agentic-flow
```javascript
const { AgenticFlow } = require('agentic-flow');
const orchestrator = global.__agentOrchestrator;
```

### With AgentDB
```javascript
const { AgentDB } = require('agentdb');
const db = new AgentDB();
```

### With MCP
```bash
neural-trader mcp agent-spawn momentum
neural-trader mcp swarm-deploy multi-strategy
```

## Limits

- **Max Agents**: 50 per manager
- **Max Swarms**: 10 per orchestrator
- **Task Queue**: 10,000 tasks
- **Health Check**: Every 5 seconds
- **Auto-Restart**: Max 3 attempts

## Support

- Docs: https://github.com/ruvnet/neural-trader
- Issues: https://github.com/ruvnet/neural-trader/issues
- Guide: docs/AGENT_SYSTEM.md
