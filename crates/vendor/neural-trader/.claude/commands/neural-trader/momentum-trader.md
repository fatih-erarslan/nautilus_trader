# Neural Momentum Trader Command

Execute momentum breakout strategies using the neural-momentum-trader agent.

## Agent Location
`.claude/agents/neural-trader/momentum/neural-momentum-trader.md`

## Command

```javascript
// Spawn the neural momentum trader agent
Task(
  "Momentum Trading",
  "Use the agent at .claude/agents/neural-trader/momentum/neural-momentum-trader.md to identify and trade momentum breakouts in {symbol} using neural predictions and sentiment analysis. The agent should implement pyramiding with dynamic position scaling and trailing stops.",
  "researcher"
)
```

## Direct Agent Invocation

```bash
# Using Claude's agent system
claude agent run .claude/agents/neural-trader/momentum/neural-momentum-trader.md --symbol NVDA

# Or via Task with explicit path
Task(
  "Momentum Trader",
  "Load and execute the neural-momentum-trader agent from .claude/agents/neural-trader/momentum/neural-momentum-trader.md for NVDA momentum trading",
  "base-template-generator"
)
```

## Parameters

- `{symbol}` - Stock ticker for momentum trading (e.g., "TSLA", "NVDA", "AMD")

## Full Workflow with Agent

```javascript
// Step 1: Initialize momentum trading
mcp__claude-flow__swarm_init({
  topology: "hierarchical",
  maxAgents: 5,
  strategy: "specialized"
})

// Step 2: Load the neural-momentum-trader agent
Task(
  "Load Momentum Agent",
  "Read the agent configuration from .claude/agents/neural-trader/momentum/neural-momentum-trader.md and initialize a momentum specialist",
  "base-template-generator"
)

// Step 3: Spawn agent with configuration
mcp__claude-flow__agent_spawn({
  type: "analyst",
  name: "neural-momentum-trader",
  capabilities: ["momentum_detection", "breakout_analysis", "news_sentiment"],
  config_path: ".claude/agents/neural-trader/momentum/neural-momentum-trader.md"
})

// Step 4: Execute using agent's momentum framework
mcp__claude-flow__task_orchestrate({
  task: "Execute momentum trading on NVDA using the neural-momentum-trader agent template",
  agent_config: ".claude/agents/neural-trader/momentum/neural-momentum-trader.md",
  strategy: "parallel",
  priority: "high"
})
```

## Agent Configuration Reference

The neural-momentum-trader agent at `.claude/agents/neural-trader/momentum/neural-momentum-trader.md` includes:
- Momentum breakout detection with neural validation
- News-driven momentum trading with sentiment analysis
- Dynamic position pyramiding and scaling strategies
- Trailing stop management with profit protection
- Multi-timeframe momentum confirmation

## Using Agent's Momentum Strategies

```javascript
// The agent implements these specific strategies:

// 1. Breakout Momentum (from agent)
Task(
  "Breakout Trading",
  "Using .claude/agents/neural-trader/momentum/neural-momentum-trader.md, trade range breakouts with volume confirmation",
  "neural-momentum-trader"
)

// 2. News Momentum (from agent)
Task(
  "News Momentum",
  "Apply neural-momentum-trader agent's news-driven momentum strategy from .claude/agents/neural-trader/momentum/",
  "neural-momentum-trader"
)

// 3. Earnings Momentum (from agent)
Task(
  "Earnings Play",
  "Use neural-momentum-trader agent to trade post-earnings momentum continuation",
  "neural-momentum-trader"
)

// 4. Sector Momentum (from agent)
Task(
  "Sector Rotation",
  "Execute neural-momentum-trader agent's sector rotation momentum strategy",
  "neural-momentum-trader"
)
```

## Agent's Entry Criteria

The agent at `.claude/agents/neural-trader/momentum/neural-momentum-trader.md` uses:
- 20-day high breakout with volume > 1.5x average
- Neural forecast confidence > 0.8
- Positive news sentiment or catalyst
- Relative strength vs market > 1.2
- No resistance within 5% of entry

## Agent's Position Management

```javascript
// Using the agent's pyramiding strategy
Task(
  "Pyramiding",
  "Apply neural-momentum-trader agent's pyramiding: 2% initial, add 1% at +2%, add 0.5% at +4%, max 5%",
  "neural-momentum-trader"
)
```

## Agent's Trailing Stop System

```javascript
// Using the agent's trailing stop methodology
Task(
  "Trailing Stops",
  "Implement neural-momentum-trader agent's stops: 2% initial, breakeven at +2%, trail 3% after +5%",
  "neural-momentum-trader"
)
```

## News Integration with Agent

```javascript
// Monitor sentiment using agent's framework
Task(
  "Sentiment Monitor",
  "Use .claude/agents/neural-trader/momentum/neural-momentum-trader.md to monitor news sentiment for NVDA",
  "neural-momentum-trader"
)

// Trade on catalyst using agent
Task(
  "Catalyst Trade",
  "Apply neural-momentum-trader agent's catalyst detection and trading logic",
  "neural-momentum-trader"
)
```

## Multi-Timeframe Confirmation

```javascript
// Use agent's multi-timeframe logic
Task(
  "Timeframe Check",
  "Apply neural-momentum-trader agent's multi-timeframe confirmation from .claude/agents/neural-trader/momentum/",
  "neural-momentum-trader"
)
```

## Performance Monitoring

```javascript
// Track agent's momentum trades
Task(
  "Agent Performance",
  "Monitor performance of neural-momentum-trader agent from .claude/agents/neural-trader/momentum/",
  "performance-analyzer"
)

// Optimize agent's parameters
Task(
  "Optimize Agent",
  "Optimize neural-momentum-trader agent's entry and exit parameters",
  "optimizer"
)
```

## Agent Success Metrics

The neural-momentum-trader agent targets:
- Win Rate: > 45%
- Average Win/Loss: > 2.5
- Profit Factor: > 2.0
- Max Drawdown: < 20%
- Momentum Capture: > 70%

## Example Usage

### Basic Momentum Trade with Agent
```javascript
Task(
  "Basic Momentum",
  "Load .claude/agents/neural-trader/momentum/neural-momentum-trader.md and trade TSLA momentum",
  "researcher"
)
```

### News-Driven Momentum with Agent
```javascript
Task(
  "News Momentum",
  "Execute .claude/agents/neural-trader/momentum/neural-momentum-trader.md on AAPL earnings momentum",
  "researcher"
)
```

### Sector Rotation with Agent
```javascript
Task(
  "Sector Momentum",
  "Use neural-momentum-trader agent from .claude/agents/neural-trader/momentum/ for XLK sector momentum",
  "researcher"
)
```

## Agent's Momentum Detection Framework

The agent uses a 5-point framework:
1. **Price Acceleration**: Rate of change increasing
2. **Volume Confirmation**: Above-average volume on moves
3. **Neural Validation**: `mcp__ai-news-trader__neural_forecast` confirms direction
4. **Sentiment Support**: `mcp__ai-news-trader__analyze_news` shows alignment
5. **Technical Strength**: RSI > 50 and rising, MACD positive

## Tips for Using the Agent

1. **Trust Agent's Breakout Detection**: The agent has sophisticated breakout logic
2. **Use Agent's Pyramiding Rules**: The agent scales positions optimally
3. **Follow Agent's Stop Management**: The agent's trailing stops protect profits
4. **Monitor Agent's Sentiment Signals**: The agent integrates news effectively
5. **Let Agent Run Trends**: The agent knows when to hold and when to exit

## Related Agents

- `.claude/agents/neural-trader/trend-following/neural-trend-follower.md`
- `.claude/agents/neural-trader/sentiment/neural-sentiment-trader.md`
- `.claude/agents/neural-trader/risk-manager/neural-risk-manager.md`