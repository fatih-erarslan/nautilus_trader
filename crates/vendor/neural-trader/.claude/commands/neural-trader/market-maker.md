# Neural Market Maker Command

High-frequency liquidity provision and spread capture using the neural-market-maker agent.

## Agent Location
`.claude/agents/neural-trader/market-maker/neural-market-maker.md`

## Command

```javascript
// Spawn the neural market maker agent
Task(
  "Market Making",
  "Use the agent at .claude/agents/neural-trader/market-maker/neural-market-maker.md to provide liquidity in {symbol} with dynamic spread adjustment and inventory management. The agent should maintain delta neutrality and manage adverse selection.",
  "researcher"
)
```

## Direct Agent Invocation

```bash
# Using Claude's agent system
claude agent run .claude/agents/neural-trader/market-maker/neural-market-maker.md --symbol AAPL

# Or via Task with explicit path
Task(
  "Market Maker",
  "Load and execute the neural-market-maker agent from .claude/agents/neural-trader/market-maker/neural-market-maker.md for AAPL",
  "base-template-generator"
)
```

## Parameters

- `{symbol}` - Asset for market making (e.g., "AAPL", "SPY", "BTC-USD")

## Full Workflow with Agent

```javascript
// Step 1: Initialize market making
mcp__claude-flow__swarm_init({
  topology: "star",  // Centralized for speed
  maxAgents: 10,
  strategy: "specialized"
})

// Step 2: Load the neural-market-maker agent template
Task(
  "Load MM Agent",
  "Read the agent configuration from .claude/agents/neural-trader/market-maker/neural-market-maker.md and initialize a market making specialist",
  "base-template-generator"
)

// Step 3: Spawn agent with configuration
mcp__claude-flow__agent_spawn({
  type: "specialist",
  name: "neural-market-maker",
  capabilities: ["hft", "liquidity_provision", "spread_management", "inventory_control"],
  config_path: ".claude/agents/neural-trader/market-maker/neural-market-maker.md"
})

// Step 4: Execute using agent's market making framework
mcp__claude-flow__task_orchestrate({
  task: "Execute market making on AAPL using the neural-market-maker agent template",
  agent_config: ".claude/agents/neural-trader/market-maker/neural-market-maker.md",
  strategy: "parallel",
  priority: "critical"
})

// Place bid/ask quotes
// Bid side
mcp__ai-news-trader__execute_trade({
  strategy: "market_making",
  symbol: "AAPL",
  action: "buy",
  quantity: 100,
  order_type: "limit",
  limit_price: 149.95  // Below mid
})

// Ask side
mcp__ai-news-trader__execute_trade({
  strategy: "market_making",
  symbol: "AAPL",
  action: "sell",
  quantity: 100,
  order_type: "limit",
  limit_price: 150.05  // Above mid
})

// Monitor order flow
setInterval(() => {
  mcp__ai-news-trader__get_execution_analytics({
    time_period: "1m"
  })
}, 1000)

// Adjust spreads dynamically
mcp__claude-flow__neural_patterns({
  action: "analyze",
  pattern_type: "order_flow_imbalance"
})
```

## Agent Configuration Reference

The neural-market-maker agent at `.claude/agents/neural-trader/market-maker/neural-market-maker.md` includes:
- High-frequency quote updates (<100ms)
- Dynamic spread adjustment based on volatility
- Inventory risk management with skewing
- Adverse selection detection and mitigation
- Order flow toxicity analysis
- Rebate optimization strategies

## Configuration

```yaml
market_maker_config:
  quote_update_ms: 100
  min_spread_bps: 5
  max_spread_bps: 50
  inventory_limits:
    max_long: 10000
    max_short: 10000
    target: 0
  risk_parameters:
    max_position_time: 300  # seconds
    adverse_selection_threshold: 0.6
    toxic_flow_threshold: 0.7
```

## Using Agent's Market Making Strategies

```javascript
// The agent implements these specific strategies:

// 1. Basic Quote Management (from agent)
Task(
  "Quote Management",
  "Using .claude/agents/neural-trader/market-maker/neural-market-maker.md, maintain continuous bid/ask quotes",
  "neural-market-maker"
)

// 2. Inventory Skewing (from agent)
Task(
  "Inventory Skew",
  "Apply neural-market-maker agent's inventory skewing from .claude/agents/neural-trader/market-maker/",
  "neural-market-maker"
)

// 3. Adverse Selection (from agent)
Task(
  "Adverse Selection",
  "Use neural-market-maker agent to detect and avoid toxic order flow",
  "neural-market-maker"
)

// 4. Spread Optimization (from agent)
Task(
  "Spread Optimize",
  "Execute neural-market-maker agent's dynamic spread optimization",
  "neural-market-maker"
)
```

## Agent's Quote Logic

The agent at `.claude/agents/neural-trader/market-maker/neural-market-maker.md` uses:
- Base spread from volatility (σ × √t)
- Inventory adjustment (skew quotes when imbalanced)
- Order flow imbalance adjustment
- Adverse selection premium
- Competition-based tightening

## Agent's Spread Calculation

```javascript
// Using the agent's spread formula
Task(
  "Calculate Spread",
  "Apply neural-market-maker agent's spread calculation: base + inventory + adverse + competition",
  "neural-market-maker"
)

// Dynamic adjustment
Task(
  "Dynamic Spread",
  "Use neural-market-maker agent to adjust spreads every 100ms based on market conditions",
  "neural-market-maker"
)
```

## Agent's Inventory Management

```javascript
// Inventory control with agent
Task(
  "Inventory Control",
  "Use .claude/agents/neural-trader/market-maker/neural-market-maker.md to maintain neutral inventory",
  "neural-market-maker"
)

// Skewing quotes
Task(
  "Quote Skewing",
  "Apply neural-market-maker agent's quote skewing when inventory > 50% of limit",
  "neural-market-maker"
)

// Emergency liquidation
Task(
  "Emergency Flat",
  "Execute neural-market-maker agent's emergency inventory liquidation",
  "neural-market-maker"
)
```

## Order Flow Analysis

```javascript
// Toxicity detection with agent
Task(
  "Toxicity Check",
  "Use neural-market-maker agent from .claude/agents/neural-trader/market-maker/ to detect toxic flow",
  "neural-market-maker"
)

// Informed trader detection
Task(
  "Informed Flow",
  "Apply neural-market-maker agent's informed trader detection algorithm",
  "neural-market-maker"
)
```

## Rebate Optimization

```javascript
// Maximize rebates with agent
Task(
  "Rebate Optimize",
  "Use .claude/agents/neural-trader/market-maker/neural-market-maker.md to maximize maker rebates",
  "neural-market-maker"
)

// Venue selection
Task(
  "Venue Select",
  "Apply neural-market-maker agent's venue selection for best rebates",
  "neural-market-maker"
)
```

## Performance Monitoring

```javascript
// Track agent's performance
Task(
  "Agent Metrics",
  "Monitor performance of neural-market-maker agent from .claude/agents/neural-trader/market-maker/",
  "performance-analyzer"
)

// Spread capture analysis
Task(
  "Spread Analysis",
  "Analyze spread capture efficiency of neural-market-maker agent",
  "analyzer"
)
```

## Agent Success Metrics

The neural-market-maker agent targets:
- Spread Capture Rate: > 60%
- Inventory Half-Life: < 60 seconds
- Adverse Selection: < 30%
- Daily Volume: > $1M notional
- Sharpe Ratio: > 3.0

## Example Usage

### Basic Market Making with Agent
```javascript
Task(
  "Basic MM",
  "Load .claude/agents/neural-trader/market-maker/neural-market-maker.md and make markets in SPY",
  "researcher"
)
```

### High-Frequency MM with Agent
```javascript
Task(
  "HFT MM",
  "Execute .claude/agents/neural-trader/market-maker/neural-market-maker.md with 10ms quote updates",
  "researcher"
)
```

### Multi-Asset MM with Agent
```javascript
Task(
  "Multi MM",
  "Use neural-market-maker agent from .claude/agents/neural-trader/market-maker/ for AAPL, MSFT, GOOGL",
  "researcher"
)
```

## Agent's Execution Framework

The agent implements sophisticated execution:
1. **Quote Placement**: Optimal bid/ask positioning
2. **Order Matching**: Smart order routing
3. **Fill Management**: Partial fill handling
4. **Cancel/Replace**: Dynamic quote updates
5. **Latency Optimization**: Sub-millisecond execution

## Agent's Risk Controls

```javascript
// Position limits
Task(
  "Position Limits",
  "Apply neural-market-maker agent's position limits: max 10,000 shares long/short",
  "neural-market-maker"
)

// Loss limits
Task(
  "Loss Control",
  "Use neural-market-maker agent's daily loss limit of $10,000",
  "neural-market-maker"
)

// Volatility adjustment
Task(
  "Vol Adjust",
  "Apply neural-market-maker agent's volatility-based position reduction",
  "neural-market-maker"
)
```

## Tips for Using the Agent

1. **Trust Agent's Spreads**: The agent optimizes spread width dynamically
2. **Use Agent's Inventory Logic**: The agent manages inventory efficiently
3. **Follow Agent's Risk Limits**: The agent protects against adverse selection
4. **Monitor Agent's Toxicity Signals**: The agent detects informed traders
5. **Let Agent Execute**: The agent handles high-frequency updates optimally

## Related Agents

- `.claude/agents/neural-trader/arbitrage/neural-arbitrageur.md`
- `.claude/agents/neural-trader/risk-manager/neural-risk-manager.md`
- `.claude/agents/neural-trader/pairs-trading/neural-pairs-trader.md`