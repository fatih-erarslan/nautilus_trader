# Neural Trend Follower Command

Execute multi-timeframe trend following strategies using the neural-trend-follower agent.

## Agent Location
`.claude/agents/neural-trader/trend-following/neural-trend-follower.md`

## Command

```javascript
// Spawn the neural trend follower agent
Task(
  "Trend Following Analysis", 
  "Use the agent at .claude/agents/neural-trader/trend-following/neural-trend-follower.md to analyze {symbol} for trend-following opportunities using neural forecasting across 1h, 4h, and 24h timeframes. The agent should generate entry signals when trends align and execute trades with proper risk management.",
  "researcher"  // Base agent type that will load the neural-trend-follower template
)
```

## Direct Agent Invocation

```bash
# Using Claude's agent system
claude agent run .claude/agents/neural-trader/trend-following/neural-trend-follower.md --symbol SPY

# Or via Task with explicit path
Task(
  "Trend Follower",
  "Load and execute the neural-trend-follower agent from .claude/agents/neural-trader/trend-following/neural-trend-follower.md for SPY analysis",
  "base-template-generator"
)
```

## Parameters

- `{symbol}` - Stock ticker to analyze (e.g., "SPY", "AAPL", "TSLA")

## Full Workflow with Agent

```javascript
// Step 1: Initialize swarm
mcp__claude-flow__swarm_init({
  topology: "hierarchical",
  maxAgents: 3,
  strategy: "specialized"
})

// Step 2: Load the neural-trend-follower agent template
Task(
  "Load Trend Agent",
  "Read the agent configuration from .claude/agents/neural-trader/trend-following/neural-trend-follower.md and initialize a trend-following specialist",
  "base-template-generator"
)

// Step 3: Spawn agent with loaded configuration
mcp__claude-flow__agent_spawn({
  type: "analyst",
  name: "neural-trend-follower",
  capabilities: ["trend_analysis", "neural_forecasting", "multi_timeframe"],
  config_path: ".claude/agents/neural-trader/trend-following/neural-trend-follower.md"
})

// Step 4: Execute trend analysis using agent's logic
mcp__claude-flow__task_orchestrate({
  task: "Execute trend-following strategy on SPY using the neural-trend-follower agent template",
  agent_config: ".claude/agents/neural-trader/trend-following/neural-trend-follower.md",
  strategy: "parallel",
  priority: "high"
})
```

## Agent Configuration Reference

The neural-trend-follower agent at `.claude/agents/neural-trader/trend-following/neural-trend-follower.md` includes:
- Multi-timeframe analysis (1h, 4h, 24h)
- Neural forecasting with GPU acceleration
- Trend strength measurement
- Dynamic position sizing
- Trailing stop management

## Using Agent's Trading Logic

```javascript
// The agent implements these specific methods from its template:

// 1. Multi-Timeframe Analysis (from agent)
Task(
  "Timeframe Analysis",
  "Using .claude/agents/neural-trader/trend-following/neural-trend-follower.md, scan 1h, 4h, and 24h timeframes for SPY",
  "neural-trend-follower"
)

// 2. Neural Validation (from agent)
Task(
  "Neural Validation",
  "Apply neural-trend-follower agent's validation logic using mcp__ai-news-trader__neural_forecast",
  "neural-trend-follower"
)

// 3. Risk Assessment (from agent)
Task(
  "Risk Check",
  "Use neural-trend-follower agent's risk management rules: max 5% position, 2% stop loss, 0.6 correlation limit",
  "neural-trend-follower"
)
```

## Example Usage

### Basic Trend Analysis with Agent
```javascript
Task(
  "Trend Analysis", 
  "Load .claude/agents/neural-trader/trend-following/neural-trend-follower.md and analyze AAPL for trend signals",
  "researcher"
)
```

### Multi-Symbol Screening with Agent
```javascript
const symbols = ["AAPL", "MSFT", "GOOGL", "NVDA"]
for (const symbol of symbols) {
  Task(
    `Screen ${symbol}`,
    `Use neural-trend-follower agent from .claude/agents/neural-trader/trend-following/ to analyze ${symbol}`,
    "researcher"
  )
}
```

### Agent with Custom Parameters
```javascript
Task(
  "Conservative Trend",
  "Execute .claude/agents/neural-trader/trend-following/neural-trend-follower.md on SPY with modified parameters: max 2% risk per trade and 10% portfolio heat (override agent defaults)",
  "researcher"
)
```

## Monitoring Agent Performance

```javascript
// Check specific agent metrics
mcp__claude-flow__agent_metrics({
  agentId: "neural-trend-follower",
  config_source: ".claude/agents/neural-trader/trend-following/neural-trend-follower.md"
})

// Get agent's trading performance
Task(
  "Agent Performance",
  "Analyze performance of trades executed by neural-trend-follower agent from .claude/agents/neural-trader/trend-following/",
  "performance-analyzer"
)
```

## Agent Success Metrics

The neural-trend-follower agent targets:
- Win Rate: > 55%
- Average Win/Loss: > 1.5
- Sharpe Ratio: > 1.5
- Max Drawdown: < 15%
- Trend Capture: > 60%

## Tips for Using the Agent

1. **Load Agent First**: Always reference the full path to the agent template
2. **Use Agent's Logic**: The agent has predefined trading rules - use them
3. **Monitor Agent State**: Track the agent's internal state and decisions
4. **Override When Needed**: You can override agent parameters if needed
5. **Let Agent Run**: The agent is designed for autonomous operation

## Related Agents

- `.claude/agents/neural-trader/momentum/neural-momentum-trader.md`
- `.claude/agents/neural-trader/mean-reversion/neural-mean-reverter.md`
- `.claude/agents/neural-trader/risk-manager/neural-risk-manager.md`