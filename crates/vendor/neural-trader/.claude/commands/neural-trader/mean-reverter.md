# Neural Mean Reverter Command

Execute statistical arbitrage and mean-reversion strategies using the neural-mean-reverter agent.

## Agent Location
`.claude/agents/neural-trader/mean-reversion/neural-mean-reverter.md`

## Command

```javascript
// Spawn the neural mean reverter agent
Task(
  "Mean Reversion Analysis",
  "Use the agent at .claude/agents/neural-trader/mean-reversion/neural-mean-reverter.md to identify mean-reversion opportunities in {symbol} using z-score analysis and neural pattern validation. The agent should execute contrarian trades at statistical extremes with dynamic position sizing.",
  "researcher"
)
```

## Direct Agent Invocation

```bash
# Using Claude's agent system
claude agent run .claude/agents/neural-trader/mean-reversion/neural-mean-reverter.md --symbol XLF

# Or via Task with explicit path
Task(
  "Mean Reverter",
  "Load and execute the neural-mean-reverter agent from .claude/agents/neural-trader/mean-reversion/neural-mean-reverter.md for XLF analysis",
  "base-template-generator"
)
```

## Parameters

- `{symbol}` - Single asset or pair for mean-reversion (e.g., "AAPL", "XLF/XLE")

## Full Workflow with Agent

```javascript
// Step 1: Initialize for mean-reversion
mcp__claude-flow__swarm_init({
  topology: "mesh",
  maxAgents: 4,
  strategy: "specialized"
})

// Step 2: Load the neural-mean-reverter agent template
Task(
  "Load Mean Reversion Agent",
  "Read the agent configuration from .claude/agents/neural-trader/mean-reversion/neural-mean-reverter.md and initialize a mean-reversion specialist",
  "base-template-generator"
)

// Step 3: Spawn agent with configuration
mcp__claude-flow__agent_spawn({
  type: "optimizer",
  name: "neural-mean-reverter",
  capabilities: ["statistical_arbitrage", "pairs_trading", "z_score_analysis"],
  config_path: ".claude/agents/neural-trader/mean-reversion/neural-mean-reverter.md"
})

// Step 4: Execute using agent's strategy
mcp__claude-flow__task_orchestrate({
  task: "Execute mean-reversion strategy on XLF using the neural-mean-reverter agent template",
  agent_config: ".claude/agents/neural-trader/mean-reversion/neural-mean-reverter.md",
  strategy: "parallel",
  priority: "high"
})
```

## Agent Configuration Reference

The neural-mean-reverter agent at `.claude/agents/neural-trader/mean-reversion/neural-mean-reverter.md` includes:
- Statistical arbitrage with neural validation
- Pairs trading with cointegration analysis
- Z-score calculations with volatility adjustments
- Neural pattern recognition for mean-reversion signals
- Regime detection through sentiment analysis

## Using Agent's Trading Strategies

```javascript
// The agent implements these specific strategies:

// 1. Single Asset Mean-Reversion (from agent)
Task(
  "Single Asset MR",
  "Using .claude/agents/neural-trader/mean-reversion/neural-mean-reverter.md, trade SPY when z-score exceeds ±2",
  "neural-mean-reverter"
)

// 2. Pairs Trading (from agent)
Task(
  "Pairs Strategy",
  "Apply neural-mean-reverter agent's pairs trading logic from .claude/agents/neural-trader/mean-reversion/ on JPM/BAC",
  "neural-mean-reverter"
)

// 3. Sector Rotation (from agent)
Task(
  "Sector MR",
  "Use neural-mean-reverter agent to trade sector rotation in SPDR ETFs",
  "neural-mean-reverter"
)
```

## Agent's Entry Criteria

The agent at `.claude/agents/neural-trader/mean-reversion/neural-mean-reverter.md` uses:
- Z-score beyond ±2 standard deviations
- Neural pattern confidence > 0.75
- No major news events or catalysts
- Liquidity sufficient for position size
- Correlation stable (for pairs trading)

## Agent's Position Sizing

```javascript
// Using the agent's Kelly Criterion implementation
Task(
  "Position Sizing",
  "Apply neural-mean-reverter agent's position sizing: Kelly Criterion with 0.25 fraction, max 3% per position",
  "neural-mean-reverter"
)
```

## Agent's Risk Management

```javascript
// Using the agent's risk rules
Task(
  "Risk Management",
  "Implement neural-mean-reverter agent's risk controls: stop at z-score ±3, time stop after 5 days, exit on regime change",
  "neural-mean-reverter"
)
```

## Pairs Trading with Agent

```javascript
// Find pairs using agent's methodology
Task(
  "Find Pairs",
  "Use .claude/agents/neural-trader/mean-reversion/neural-mean-reverter.md to identify cointegrated pairs in financials",
  "neural-mean-reverter"
)

// Monitor spread using agent
Task(
  "Monitor Spread",
  "Apply neural-mean-reverter agent's spread monitoring logic to JPM/BAC pair",
  "neural-mean-reverter"
)

// Execute pairs trade with agent
Task(
  "Execute Pairs",
  "Use neural-mean-reverter agent to execute market-neutral pairs trade on JPM/BAC",
  "neural-mean-reverter"
)
```

## Performance Monitoring

```javascript
// Track agent's performance
Task(
  "Agent Metrics",
  "Monitor performance of neural-mean-reverter agent from .claude/agents/neural-trader/mean-reversion/",
  "performance-analyzer"
)

// Backtest agent's strategy
Task(
  "Backtest Agent",
  "Backtest the neural-mean-reverter agent's strategy using historical data",
  "backtester"
)
```

## Agent Success Metrics

The neural-mean-reverter agent targets:
- Win Rate: > 65%
- Average Reversion Time: < 5 days
- Sharpe Ratio: > 2.0
- Max Drawdown: < 10%
- Correlation Stability: > 0.7

## Example Usage

### Basic Mean-Reversion with Agent
```javascript
Task(
  "Basic MR",
  "Load .claude/agents/neural-trader/mean-reversion/neural-mean-reverter.md and trade SPY mean-reversion",
  "researcher"
)
```

### Pairs Trading with Agent
```javascript
Task(
  "Pairs Trade",
  "Execute .claude/agents/neural-trader/mean-reversion/neural-mean-reverter.md pairs trading on XLF/XLE",
  "researcher"
)
```

### Sector Rotation with Agent
```javascript
Task(
  "Sector Rotation",
  "Use neural-mean-reverter agent from .claude/agents/neural-trader/mean-reversion/ for sector ETF rotation",
  "researcher"
)
```

## Tips for Using the Agent

1. **Agent's Regime Detection**: The agent monitors for regime changes - trust its signals
2. **Use Agent's Z-Score Logic**: The agent has sophisticated z-score calculations
3. **Follow Agent's Pairs Selection**: The agent tests for cointegration properly
4. **Respect Agent's Time Stops**: The agent exits after 5 days for a reason
5. **Monitor Agent's Confidence**: The agent provides confidence scores - use them

## Related Agents

- `.claude/agents/neural-trader/pairs-trading/neural-pairs-trader.md`
- `.claude/agents/neural-trader/arbitrage/neural-arbitrageur.md`
- `.claude/agents/neural-trader/risk-manager/neural-risk-manager.md`