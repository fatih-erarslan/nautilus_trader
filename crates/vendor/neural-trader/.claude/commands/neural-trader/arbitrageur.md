# Neural Arbitrageur Command

Execute high-frequency arbitrage strategies using the neural-arbitrageur agent.

## Agent Location
`.claude/agents/neural-trader/arbitrage/neural-arbitrageur.md`

## Command

```javascript
// Spawn the neural arbitrageur agent
Task(
  "Arbitrage Detection",
  "Use the agent at .claude/agents/neural-trader/arbitrage/neural-arbitrageur.md to monitor and execute arbitrage opportunities across {markets} with neural spread convergence prediction. The agent should target sub-10ms execution with risk-neutral positions.",
  "researcher"
)
```

## Direct Agent Invocation

```bash
# Using Claude's agent system
claude agent run .claude/agents/neural-trader/arbitrage/neural-arbitrageur.md --markets NYSE/NASDAQ

# Or via Task with explicit path
Task(
  "Arbitrageur",
  "Load and execute the neural-arbitrageur agent from .claude/agents/neural-trader/arbitrage/neural-arbitrageur.md for market arbitrage",
  "base-template-generator"
)
```

## Parameters

- `{markets}` - Markets/exchanges to monitor (e.g., "NYSE/NASDAQ", "SPY/ES", "BTC/ETH")

## Full Workflow with Agent

```javascript
// Step 1: Initialize HFT infrastructure
mcp__claude-flow__swarm_init({
  topology: "star",  // Centralized for speed
  maxAgents: 8,
  strategy: "specialized"
})

// Step 2: Load the neural-arbitrageur agent template
Task(
  "Load Arbitrage Agent",
  "Read the agent configuration from .claude/agents/neural-trader/arbitrage/neural-arbitrageur.md and initialize an HFT arbitrage specialist",
  "base-template-generator"
)

// Step 3: Spawn agent with configuration
mcp__claude-flow__agent_spawn({
  type: "specialist",
  name: "neural-arbitrageur",
  capabilities: ["hft", "cross_market", "latency_sensitive"],
  config_path: ".claude/agents/neural-trader/arbitrage/neural-arbitrageur.md"
})

// Step 4: Execute using agent's arbitrage framework
mcp__claude-flow__task_orchestrate({
  task: "Monitor SPY/ES/SPX for arbitrage using the neural-arbitrageur agent template",
  agent_config: ".claude/agents/neural-trader/arbitrage/neural-arbitrageur.md",
  strategy: "parallel",
  priority: "critical"
})

// Correlation analysis for stat arb
mcp__ai-news-trader__correlation_analysis({
  symbols: ["SPY", "IVV", "VOO"],  // Similar ETFs
  period_days: 30,
  use_gpu: true
})

// Neural spread prediction
mcp__ai-news-trader__neural_forecast({
  symbol: "SPY-IVV",  // Spread
  horizon: 1,  // 1 hour
  confidence_level: 0.99,
  use_gpu: true
})

// Execute arbitrage trade
// Buy underpriced
mcp__ai-news-trader__execute_trade({
  strategy: "arbitrage",
  symbol: "SPY",
  action: "buy",
  quantity: 1000,
  order_type: "market"
})

// Sell overpriced (simultaneous)
mcp__ai-news-trader__execute_trade({
  strategy: "arbitrage",
  symbol: "IVV",
  action: "sell",
  quantity: 1000,
  order_type: "market"
})
```

## Agent Configuration Reference

The neural-arbitrageur agent at `.claude/agents/neural-trader/arbitrage/neural-arbitrageur.md` includes:
- Sub-10ms latency execution targets
- Cross-exchange and cross-market monitoring
- Statistical arbitrage with neural validation
- Triangular and index arbitrage strategies
- Real-time spread convergence prediction

## Configuration

```yaml
arbitrage_config:
  latency_target_ms: 10
  min_spread_bps: 5
  max_position: 100000
  concurrent_positions: 10
  stop_loss_bps: 50
  confidence_threshold: 0.8
```

## Using Agent's Arbitrage Strategies

```javascript
// The agent implements these specific strategies:

// 1. Cross-Exchange Arbitrage (from agent)
Task(
  "Cross-Exchange",
  "Using .claude/agents/neural-trader/arbitrage/neural-arbitrageur.md, monitor AAPL across NYSE, NASDAQ, BATS",
  "neural-arbitrageur"
)

// 2. Index Arbitrage (from agent)
Task(
  "Index Arb",
  "Apply neural-arbitrageur agent's index arbitrage from .claude/agents/neural-trader/arbitrage/ on SPY vs ES",
  "neural-arbitrageur"
)

// 3. Statistical Arbitrage (from agent)
Task(
  "Stat Arb",
  "Use neural-arbitrageur agent to trade mean-reversion spreads with neural predictions",
  "neural-arbitrageur"
)

// 4. Triangular Arbitrage (from agent)
Task(
  "Triangular",
  "Execute neural-arbitrageur agent's three-way arbitrage strategy on EUR/USD/GBP",
  "neural-arbitrageur"
)
```

## Agent's Entry Criteria

The agent at `.claude/agents/neural-trader/arbitrage/neural-arbitrageur.md` uses:
- Spread > 5 basis points after costs
- Neural confidence > 0.8
- Liquidity sufficient on both sides
- Latency < 10ms to both venues
- No regulatory restrictions

## Execution Management

```javascript
// Pre-trade validation
function validate_arbitrage(opportunity) {
  // Check spread after costs
  const net_spread = opportunity.spread - transaction_costs
  if (net_spread < min_profit) return false
  
  // Check position limits
  if (current_exposure + opportunity.size > max_exposure) return false
  
  // Verify both legs executable
  if (!check_liquidity(opportunity.buy_side)) return false
  if (!check_liquidity(opportunity.sell_side)) return false
  
  return true
}

// Simultaneous execution
async function execute_arbitrage(opportunity) {
  const orders = [
    execute_order(opportunity.buy_side),
    execute_order(opportunity.sell_side)
  ]
  
  const results = await Promise.all(orders)
  
  // Handle partial fills
  if (results[0].filled !== results[1].filled) {
    hedge_imbalance(results)
  }
}
```

## Risk Management

```javascript
// Position limits
const risk_limits = {
  max_single_position: 100000,
  max_total_exposure: 500000,
  max_correlation: 0.3,
  daily_loss_limit: 10000
}

// Circuit breakers
if (consecutive_losses >= 3) {
  stop_all_trading()
  alert("Circuit breaker triggered")
}

// Latency monitoring
if (execution_latency > 15) {  // ms
  reduce_position_sizes()
  alert("Latency degradation detected")
}
```

## Performance Optimization

```javascript
// Latency optimization
mcp__claude-flow__wasm_optimize({
  operation: "arbitrage_detection"
})

// Neural model optimization
mcp__ai-news-trader__neural_optimize({
  model_id: "spread_predictor",
  parameter_ranges: {
    lookback_window: [10, 100],
    prediction_horizon: [1, 10],
    confidence_threshold: [0.7, 0.95]
  },
  trials: 200,
  optimization_metric: "sharpe_ratio",
  use_gpu: true
})
```

## Monitoring Dashboard

```javascript
// Real-time metrics
mcp__claude-flow__swarm_monitor({
  interval: 1,  // 1 second updates
  swarmId: "arbitrage_swarm"
})

// Execution analytics
mcp__ai-news-trader__get_execution_analytics({
  time_period: "1h"
})

// P&L tracking
mcp__ai-news-trader__get_portfolio_status({
  include_analytics: true
})
```

## Success Metrics

- Sharpe Ratio: Target > 5.0
- Win Rate: Target > 75%
- Average Profit: 10-50 bps
- Execution Latency: < 10ms
- Daily Trades: 100-500

## Infrastructure Requirements

```yaml
infrastructure:
  servers: co-located
  network: direct_market_access
  redundancy: hot_backup
  monitoring: microsecond_precision
  data_feeds: 
    - direct_exchange
    - normalized_consolidated
  execution:
    - smart_order_router
    - dark_pool_access
```

## Agent Success Metrics

The neural-arbitrageur agent targets:
- Sharpe Ratio: > 5.0
- Win Rate: > 75%
- Average Profit: 10-50 bps
- Execution Latency: < 10ms
- Daily Trades: 100-500

## Performance Monitoring

```javascript
// Track agent's performance
Task(
  "Agent Metrics",
  "Monitor performance of neural-arbitrageur agent from .claude/agents/neural-trader/arbitrage/",
  "performance-analyzer"
)

// Optimize agent's parameters
Task(
  "Optimize Agent",
  "Optimize neural-arbitrageur agent's spread detection and execution parameters",
  "optimizer"
)
```

## Tips for Using the Agent

1. **Trust Agent's Speed**: The agent optimizes for microsecond-level execution
2. **Use Agent's Risk Controls**: The agent implements sophisticated risk management
3. **Follow Agent's Signals**: The agent validates opportunities with neural models
4. **Respect Agent's Limits**: The agent enforces position and exposure limits
5. **Monitor Agent's Compliance**: The agent ensures regulatory compliance

## Related Agents

- `.claude/agents/neural-trader/pairs-trading/neural-pairs-trader.md`
- `.claude/agents/neural-trader/market-maker/neural-market-maker.md`
- `.claude/agents/neural-trader/risk-manager/neural-risk-manager.md`