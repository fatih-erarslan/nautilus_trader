# API Reference

Complete API and CLI documentation for Neural Trader.

## 📚 Contents

### Command-Line Interface (CLI)
- [CLI Reference](./cli_reference.md) - Complete command-line interface documentation
- [Neural Forecast CLI](./neural_forecast.md) - Neural network forecasting commands

### MCP Server
Model Context Protocol server for AI assistant integration.

**Location:** [`/docs/api-reference/mcp/`](./mcp/)

- [MCP Server README](./mcp/MCP_SERVER_README.md) - Main MCP server documentation
- [MCP Implementation](./mcp/MCP_IMPLEMENTATION_COMPLETE.md) - Implementation details
- [MCP Deployment Guide](./mcp/MCP_DEPLOYMENT_GUIDE.md) - Production deployment
- [Syndicate Tools](./mcp/SYNDICATE_TOOLS.md) - Collaborative trading tools
- [Best Practices & Security](./mcp/MCP_BEST_PRACTICES_SECURITY.md)

### Swarm API
Multi-agent orchestration and coordination.

- [Swarm API Guide](./SWARM_API_GUIDE.md) - Complete swarm orchestration API
- [MCP Tools Reference](./mcp_tools.md) - MCP tool catalog

## 🚀 Quick Reference

### CLI Commands

```bash
# Show all commands
npx neural-trader --help

# Strategy backtesting
npx neural-trader strategy --strategy momentum --symbol SPY --backtest

# Neural network forecasting
npx neural-trader neural --model lstm --train --symbol AAPL

# Multi-agent swarm
npx neural-trader swarm --swarm hierarchical --agents 12 --e2b

# Risk analysis
npx neural-trader risk --var --monte-carlo --scenarios 10000

# MCP server (for AI assistants)
npx neural-trader mcp
```

### MCP Tools

The MCP server provides 99+ tools accessible via Claude, Cursor, or Copilot:

**Categories:**
- Trading Strategies (momentum, mean-reversion, pairs, arbitrage)
- Neural Networks (LSTM, Transformer, N-HiTS)
- Risk Management (VaR, CVaR, stress testing)
- Portfolio Optimization
- Sports Betting
- Prediction Markets
- Swarm Orchestration

See [MCP Server README](./mcp/MCP_SERVER_README.md) for complete tool catalog.

### Swarm Orchestration

```typescript
// Initialize swarm
await swarm_init({
  topology: "hierarchical",
  maxAgents: 8
});

// Spawn agents
await agent_spawn({
  type: "researcher",
  capabilities: ["analysis", "reporting"]
});

// Orchestrate task
await task_orchestrate({
  task: "Analyze market trends and generate report",
  strategy: "parallel"
});
```

See [Swarm API Guide](./SWARM_API_GUIDE.md) for full API documentation.

## 📖 Integration Examples

### Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "neural-trader": {
      "command": "npx",
      "args": ["neural-trader", "mcp"]
    }
  }
}
```

### Cursor / Copilot

Similar MCP configuration in their respective settings.

## 🔗 Related Documentation

- [Getting Started](../getting-started/) - Installation and setup
- [Integrations](../integrations/) - Third-party integrations
- [Features](../features/) - Platform features
- [Deployment](../deployment/) - Production deployment

---

[← Back to Main Docs](../README.md) | [MCP Server →](./mcp/)
