# Neural Pairs Trader Command

Execute market-neutral pairs trading using the neural-pairs-trader agent.

## Agent Location
`.claude/agents/neural-trader/pairs-trading/neural-pairs-trader.md`

## Command

```javascript
// Spawn the neural pairs trader agent
Task(
  "Pairs Trading",
  "Use the agent at .claude/agents/neural-trader/pairs-trading/neural-pairs-trader.md to identify and trade cointegrated pairs in {sector} using neural spread predictions and dynamic hedge ratios. The agent should maintain market neutrality with systematic rebalancing.",
  "researcher"
)
```

## Direct Agent Invocation

```bash
# Using Claude's agent system
claude agent run .claude/agents/neural-trader/pairs-trading/neural-pairs-trader.md --sector financials

# Or via Task with explicit path
Task(
  "Pairs Trader",
  "Load and execute the neural-pairs-trader agent from .claude/agents/neural-trader/pairs-trading/neural-pairs-trader.md for pairs trading",
  "base-template-generator"
)
```

## Parameters

- `{sector}` - Sector or asset class for pairs (e.g., "financials", "energy", "tech")

## Full Workflow with Agent

```javascript
// Step 1: Initialize pairs trading
mcp__claude-flow__swarm_init({
  topology: "mesh",
  maxAgents: 6,
  strategy: "specialized"
})

// Step 2: Load the neural-pairs-trader agent template
Task(
  "Load Pairs Agent",
  "Read the agent configuration from .claude/agents/neural-trader/pairs-trading/neural-pairs-trader.md and initialize a pairs trading specialist",
  "base-template-generator"
)

// Step 3: Spawn agent with configuration
mcp__claude-flow__agent_spawn({
  type: "specialist",
  name: "neural-pairs-trader",
  capabilities: ["cointegration", "spread_trading", "market_neutral"],
  config_path: ".claude/agents/neural-trader/pairs-trading/neural-pairs-trader.md"
})

// Find cointegrated pairs
mcp__ai-news-trader__correlation_analysis({
  symbols: ["JPM", "BAC", "WFC", "C", "GS", "MS"],
  period_days: 90,
  use_gpu: true
})

// Step 4: Execute using agent's cointegration framework
mcp__claude-flow__task_orchestrate({
  task: "Test JPM/BAC for cointegration using the neural-pairs-trader agent template",
  agent_config: ".claude/agents/neural-trader/pairs-trading/neural-pairs-trader.md",
  strategy: "parallel",
  priority: "high"
})

// Test cointegration (via agent)
Task(
  "Cointegration Test",
  "Apply neural-pairs-trader agent from .claude/agents/neural-trader/pairs-trading/ to test JPM/BAC cointegration",
  "neural-pairs-trader"
)

// Train spread model
mcp__ai-news-trader__neural_train({
  data_path: "/data/jpm_bac_spread.csv",
  model_type: "lstm",
  epochs: 100,
  batch_size: 32,
  learning_rate: 0.001,
  validation_split: 0.2,
  use_gpu: true
})

// Forecast spread
mcp__ai-news-trader__neural_forecast({
  symbol: "JPM-BAC",  // Spread
  horizon: 24,
  confidence_level: 0.95,
  model_id: "pairs_spread_lstm",
  use_gpu: true
})

// Execute pairs trade
// Long leg
mcp__ai-news-trader__execute_trade({
  strategy: "pairs_trading",
  symbol: "JPM",
  action: "buy",
  quantity: 100,
  order_type: "limit"
})

// Short leg (hedge ratio adjusted)
mcp__ai-news-trader__execute_trade({
  strategy: "pairs_trading",
  symbol: "BAC",
  action: "sell",
  quantity: 150,  // 1.5 hedge ratio
  order_type: "limit"
})
```

## Agent Configuration Reference

The neural-pairs-trader agent at `.claude/agents/neural-trader/pairs-trading/neural-pairs-trader.md` includes:
- Cointegration testing with ADF and Johansen tests
- Dynamic hedge ratio calculation with Kalman filters
- Neural spread forecasting and convergence prediction
- Market-neutral position management
- Pair breakdown detection and risk controls

## Configuration

```yaml
pairs_config:
  min_correlation: 0.75
  cointegration_pvalue: 0.05
  half_life_range: [5, 20]
  z_score_entry: 2.0
  z_score_exit: 0.0
  z_score_stop: 3.5
  max_pairs: 15
  position_size_per_pair: 0.05
```

## Using Agent's Trading Logic

```javascript
// The agent implements these specific methods:

// 1. Pair Selection (from agent)
Task(
  "Find Pairs",
  "Using .claude/agents/neural-trader/pairs-trading/neural-pairs-trader.md, identify cointegrated pairs in financials",
  "neural-pairs-trader"
)

// 2. Spread Trading (from agent)
Task(
  "Trade Spread",
  "Apply neural-pairs-trader agent's spread trading logic from .claude/agents/neural-trader/pairs-trading/",
  "neural-pairs-trader"
)

// 3. Dynamic Hedging (from agent)
Task(
  "Update Hedge",
  "Use neural-pairs-trader agent to dynamically adjust hedge ratios",
  "neural-pairs-trader"
)
```

## Agent's Entry Criteria

The agent at `.claude/agents/neural-trader/pairs-trading/neural-pairs-trader.md` uses:
- Correlation > 0.75 over 90 days
- Cointegration p-value < 0.05
- Half-life between 5-20 days
- Z-score beyond ±2 standard deviations
- Neural spread forecast confidence > 0.7

## Pair Selection Process

```javascript
// Step 1: Correlation screen
const candidates = symbols.filter(pair => 
  correlation(pair[0], pair[1]) > 0.75
)

// Step 2: Cointegration test
const cointegrated = candidates.filter(pair =>
  adf_test(pair).pvalue < 0.05
)

// Step 3: Half-life analysis
const tradeable = cointegrated.filter(pair => {
  const half_life = calculate_half_life(pair)
  return half_life >= 5 && half_life <= 20
})

// Step 4: Liquidity check
const final_pairs = tradeable.filter(pair =>
  min_daily_volume(pair) > 1000000
)
```

## Dynamic Hedge Ratio

```javascript
// Calculate using multiple methods
function calculate_hedge_ratio(pair) {
  // Method 1: OLS regression
  const ols_ratio = linear_regression(pair[0], pair[1]).beta
  
  // Method 2: Total least squares
  const tls_ratio = total_least_squares(pair[0], pair[1])
  
  // Method 3: Kalman filter (dynamic)
  const kalman_ratio = kalman_filter(pair[0], pair[1])
  
  // Ensemble average
  return (ols_ratio + tls_ratio + kalman_ratio) / 3
}

// Update hedge ratio periodically
setInterval(() => {
  const new_ratio = calculate_hedge_ratio(active_pair)
  if (Math.abs(new_ratio - current_ratio) > 0.1) {
    rebalance_pair(new_ratio)
  }
}, 3600000)  // Hourly
```

## Spread Trading Signals

```javascript
// Monitor spread z-score
function calculate_z_score(spread) {
  const mean = moving_average(spread, 20)
  const std = standard_deviation(spread, 20)
  return (spread[spread.length - 1] - mean) / std
}

// Generate signals
function generate_signal(z_score, neural_prediction) {
  // Entry signals
  if (z_score > 2 && neural_prediction.direction === "mean_revert") {
    return "short_spread"  // Sell expensive, buy cheap
  }
  if (z_score < -2 && neural_prediction.direction === "mean_revert") {
    return "long_spread"   // Buy expensive, sell cheap
  }
  
  // Exit signals
  if (Math.abs(z_score) < 0.5) {
    return "close_position"
  }
  
  // Stop loss
  if (Math.abs(z_score) > 3.5) {
    return "stop_loss"
  }
  
  return "hold"
}
```

## Portfolio Management

```javascript
// Manage multiple pairs
const pairs_portfolio = {
  "JPM-BAC": { weight: 0.15, z_score: 2.1 },
  "XOM-CVX": { weight: 0.15, z_score: -1.8 },
  "GOOGL-META": { weight: 0.10, z_score: 1.5 },
  // ... more pairs
}

// Diversification constraints
function check_concentration() {
  // Sector concentration
  const sector_exposure = calculate_sector_exposure(pairs_portfolio)
  if (Math.max(...Object.values(sector_exposure)) > 0.4) {
    rebalance_sectors()
  }
  
  // Correlation between pairs
  const pair_correlations = calculate_pair_correlations(pairs_portfolio)
  if (average_correlation(pair_correlations) > 0.3) {
    reduce_correlated_pairs()
  }
}
```

## Risk Management

```javascript
// Pair breakdown detection
function detect_breakdown(pair) {
  // Check correlation
  const rolling_corr = correlation(pair[0], pair[1], 20)
  if (rolling_corr < 0.5) return true
  
  // Check cointegration
  const rolling_adf = adf_test(pair, 60)
  if (rolling_adf.pvalue > 0.10) return true
  
  // Check half-life extension
  const current_half_life = calculate_half_life(pair)
  if (current_half_life > 30) return true
  
  return false
}

// Emergency exit
if (detect_breakdown(active_pair)) {
  close_pair_position(active_pair)
  alert("Pair breakdown detected")
}
```

## Performance Monitoring

```javascript
// Backtest pairs strategy
mcp__ai-news-trader__run_backtest({
  strategy: "pairs_trading",
  symbol: "JPM-BAC",
  start_date: "2023-01-01",
  end_date: "2024-12-31",
  benchmark: "risk_free_rate",
  include_costs: true,
  use_gpu: true
})

// Track metrics
const metrics = {
  win_rate: count_winning_trades() / total_trades,
  avg_days_in_trade: average_holding_period(),
  sharpe_ratio: calculate_sharpe_ratio(),
  max_drawdown: calculate_max_drawdown(),
  correlation_stability: average_correlation_over_time()
}
```

## Success Metrics

- Win Rate: Target > 70%
- Sharpe Ratio: Target > 2.5
- Max Drawdown: Limit < 8%
- Average Holding: 5-15 days
- Correlation Stability: > 0.7

## Agent Success Metrics

The neural-pairs-trader agent targets:
- Win Rate: > 70%
- Sharpe Ratio: > 2.5
- Max Drawdown: < 8%
- Average Holding: 5-15 days
- Correlation Stability: > 0.7

## Performance Monitoring

```javascript
// Track agent's performance
Task(
  "Agent Metrics",
  "Monitor performance of neural-pairs-trader agent from .claude/agents/neural-trader/pairs-trading/",
  "performance-analyzer"
)

// Optimize agent's parameters
Task(
  "Optimize Agent",
  "Optimize neural-pairs-trader agent's cointegration and hedge ratio parameters",
  "optimizer"
)
```

## Tips for Using the Agent

1. **Use Agent's Sector Logic**: The agent selects pairs within sectors for stability
2. **Trust Agent's Selection**: The agent tests fundamental similarity
3. **Follow Agent's Liquidity Rules**: The agent ensures adequate liquidity
4. **Let Agent Rebalance**: The agent updates hedge ratios optimally
5. **Respect Agent's Exit Signals**: The agent detects correlation breakdowns

## Related Agents

- `.claude/agents/neural-trader/mean-reversion/neural-mean-reverter.md`
- `.claude/agents/neural-trader/arbitrage/neural-arbitrageur.md`
- `.claude/agents/neural-trader/risk-manager/neural-risk-manager.md`