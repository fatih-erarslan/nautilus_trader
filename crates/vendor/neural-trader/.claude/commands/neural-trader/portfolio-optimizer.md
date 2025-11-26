# Neural Portfolio Optimizer Command

Execute portfolio optimization with neural return predictions and modern portfolio theory.

## Command

```javascript
Task(
  "Portfolio Optimization",
  "Optimize portfolio allocation for {assets} using neural return predictions, efficient frontier analysis, and risk parity. Implement tax-efficient rebalancing with dynamic triggers.",
  "neural-portfolio-optimizer"
)
```

## Parameters

- `{assets}` - Asset universe for optimization (e.g., "SPY,QQQ,TLT,GLD,VNQ")

## Full Workflow

```javascript
// Initialize portfolio optimizer
mcp__claude-flow__swarm_init({
  topology: "hierarchical",
  maxAgents: 4,
  strategy: "specialized"
})

// Spawn optimizer agent
mcp__claude-flow__agent_spawn({
  type: "optimizer",
  name: "neural-portfolio-optimizer",
  capabilities: ["portfolio_optimization", "risk_parity", "rebalancing"]
})

// Generate return predictions
const assets = ["SPY", "QQQ", "TLT", "GLD", "VNQ"]
for (const asset of assets) {
  mcp__ai-news-trader__neural_forecast({
    symbol: asset,
    horizon: 252,  // 1 year
    confidence_level: 0.95,
    use_gpu: true
  })
}

// Correlation analysis
mcp__ai-news-trader__correlation_analysis({
  symbols: assets,
  period_days: 252,
  use_gpu: true
})

// Risk analysis
mcp__ai-news-trader__risk_analysis({
  portfolio: current_positions,
  var_confidence: 0.05,
  time_horizon: 20,
  use_monte_carlo: true,
  use_gpu: true
})

// Optimize strategy
mcp__ai-news-trader__optimize_strategy({
  strategy: "portfolio_optimization",
  symbol: "MULTI",
  parameter_ranges: {
    target_return: [0.08, 0.15],
    max_volatility: [0.10, 0.18],
    max_drawdown: [0.10, 0.20]
  },
  optimization_metric: "sharpe_ratio",
  max_iterations: 1000,
  use_gpu: true
})

// Execute rebalancing trades
const rebalancing_trades = calculate_rebalancing_trades()
for (const trade of rebalancing_trades) {
  mcp__ai-news-trader__execute_trade({
    strategy: "portfolio_rebalance",
    symbol: trade.symbol,
    action: trade.action,
    quantity: trade.quantity,
    order_type: "limit"
  })
}
```

## Configuration

```yaml
optimizer_config:
  optimization_method: "mean_variance"
  constraints:
    long_only: true
    max_position: 0.30
    min_position: 0.02
    max_sectors: 5
  risk_limits:
    max_var_95: 0.02
    max_drawdown: 0.15
    min_sharpe: 1.0
  rebalancing:
    frequency: "monthly"
    threshold: 0.05
    tax_harvest: true
```

## Optimization Methods

### Mean-Variance Optimization
```javascript
Task("MVO", "Run Markowitz mean-variance optimization with neural return forecasts", "neural-portfolio-optimizer")

function mean_variance_optimization(returns, covariance, target_return) {
  // Quadratic programming problem
  // Minimize: w' * Σ * w (portfolio variance)
  // Subject to: w' * μ >= target_return (return constraint)
  //            Σw = 1 (weights sum to 1)
  //            w >= 0 (long only)
  
  return solve_qp(covariance, returns, target_return)
}
```

### Risk Parity
```javascript
Task("Risk Parity", "Implement equal risk contribution across all assets", "neural-portfolio-optimizer")

function risk_parity_optimization(covariance) {
  // Each asset contributes equally to portfolio risk
  const n = covariance.length
  const target_risk_contribution = 1 / n
  
  // Iterative solution
  let weights = Array(n).fill(1/n)
  
  for (let iter = 0; iter < 100; iter++) {
    const risk_contributions = calculate_risk_contributions(weights, covariance)
    weights = adjust_weights_for_equal_risk(weights, risk_contributions)
  }
  
  return weights
}
```

### Black-Litterman
```javascript
Task("Black-Litterman", "Combine market equilibrium with neural views", "neural-portfolio-optimizer")

function black_litterman(market_cap_weights, views, view_confidence) {
  // Equilibrium returns
  const equilibrium_returns = reverse_optimization(market_cap_weights)
  
  // Combine with views
  const posterior_returns = combine_equilibrium_and_views(
    equilibrium_returns,
    views,
    view_confidence
  )
  
  return mean_variance_optimization(posterior_returns)
}
```

## Neural Return Prediction

```javascript
// Ensemble prediction
async function predict_returns(assets) {
  const predictions = {}
  
  for (const asset of assets) {
    // Multiple horizons
    const horizons = [20, 60, 252]  // 1m, 3m, 1y
    const forecasts = []
    
    for (const horizon of horizons) {
      const forecast = await mcp__ai-news-trader__neural_forecast({
        symbol: asset,
        horizon: horizon,
        confidence_level: 0.95,
        use_gpu: true
      })
      forecasts.push(forecast)
    }
    
    // Weight by horizon
    predictions[asset] = {
      expected_return: weighted_average(forecasts, [0.2, 0.3, 0.5]),
      confidence: average(forecasts.map(f => f.confidence))
    }
  }
  
  return predictions
}

// Adjust for prediction confidence
function confidence_adjusted_weights(raw_weights, confidences) {
  const adjusted = {}
  let total = 0
  
  for (const [asset, weight] of Object.entries(raw_weights)) {
    adjusted[asset] = weight * confidences[asset]
    total += adjusted[asset]
  }
  
  // Renormalize
  for (const asset in adjusted) {
    adjusted[asset] /= total
  }
  
  return adjusted
}
```

## Dynamic Rebalancing

```javascript
// Rebalancing triggers
function check_rebalancing_triggers(current_weights, target_weights) {
  const triggers = {
    drift: false,
    risk: false,
    opportunity: false,
    time: false,
    tax: false
  }
  
  // Drift trigger
  const max_drift = Math.max(...Object.keys(target_weights).map(asset =>
    Math.abs(current_weights[asset] - target_weights[asset])
  ))
  triggers.drift = max_drift > 0.05
  
  // Risk trigger
  const current_var = calculate_var(current_weights)
  triggers.risk = current_var > risk_limit
  
  // Opportunity trigger
  const improvement = calculate_sharpe_improvement(target_weights, current_weights)
  triggers.opportunity = improvement > 0.2
  
  // Time trigger
  const days_since_rebalance = get_days_since_last_rebalance()
  triggers.time = days_since_rebalance >= 30
  
  // Tax trigger
  const harvestable_losses = calculate_harvestable_losses()
  triggers.tax = harvestable_losses > 3000
  
  return triggers
}

// Tax-efficient rebalancing
function tax_efficient_rebalancing(current, target) {
  const trades = []
  
  // First, harvest losses
  for (const [asset, position] of Object.entries(current)) {
    if (position.unrealized_loss > 0 && !has_wash_sale_risk(asset)) {
      trades.push({
        symbol: asset,
        action: "sell",
        quantity: position.quantity,
        reason: "tax_harvest"
      })
    }
  }
  
  // Then rebalance remaining
  for (const [asset, target_weight] of Object.entries(target)) {
    const current_weight = current[asset]?.weight || 0
    const diff = target_weight - current_weight
    
    if (Math.abs(diff) > 0.01) {
      trades.push({
        symbol: asset,
        action: diff > 0 ? "buy" : "sell",
        quantity: calculate_shares(diff * portfolio_value, asset),
        reason: "rebalance"
      })
    }
  }
  
  return trades
}
```

## Risk Management

```javascript
// Multi-factor risk model
function calculate_portfolio_risk(weights, returns, factors) {
  // Systematic risk
  const factor_exposure = calculate_factor_exposure(weights, factors)
  const systematic_risk = factor_exposure.variance
  
  // Idiosyncratic risk
  const specific_risk = calculate_specific_risk(weights, returns)
  
  // Total risk
  const total_risk = Math.sqrt(systematic_risk + specific_risk)
  
  return {
    total: total_risk,
    systematic: Math.sqrt(systematic_risk),
    specific: Math.sqrt(specific_risk),
    var_95: calculate_var(weights, 0.05),
    cvar_95: calculate_cvar(weights, 0.05),
    max_drawdown: estimate_max_drawdown(weights)
  }
}

// Stress testing
async function stress_test_portfolio(weights) {
  const scenarios = [
    { name: "Market Crash", spy: -0.20, vix: 3.0 },
    { name: "Rate Spike", tlt: -0.15, spy: -0.10 },
    { name: "Inflation Surge", gld: 0.20, tlt: -0.10 },
    { name: "Tech Bubble", qqq: -0.30, spy: -0.15 }
  ]
  
  const results = []
  
  for (const scenario of scenarios) {
    const impact = calculate_scenario_impact(weights, scenario)
    results.push({
      scenario: scenario.name,
      portfolio_impact: impact,
      var_breach: impact < -var_limit
    })
  }
  
  return results
}
```

## Performance Attribution

```javascript
// Analyze portfolio performance
function performance_attribution(portfolio, benchmark) {
  const attribution = {
    total_return: portfolio.total_return,
    benchmark_return: benchmark.total_return,
    excess_return: portfolio.total_return - benchmark.total_return
  }
  
  // Asset allocation effect
  attribution.asset_allocation = calculate_allocation_effect(
    portfolio.weights,
    benchmark.weights,
    benchmark.returns
  )
  
  // Security selection effect
  attribution.security_selection = calculate_selection_effect(
    portfolio.weights,
    portfolio.returns,
    benchmark.returns
  )
  
  // Interaction effect
  attribution.interaction = attribution.excess_return - 
    attribution.asset_allocation - 
    attribution.security_selection
  
  return attribution
}
```

## Success Metrics

- Sharpe Ratio: Target > 1.5
- Information Ratio: Target > 0.5
- Max Drawdown: Limit < 15%
- Tracking Error: < 5% (if benchmarked)
- Rebalancing Frequency: 6-12 per year

## Tips

1. **Prediction Quality**: Better forecasts = better optimization
2. **Constraint Reality**: Don't over-constrain the optimization
3. **Transaction Costs**: Include in rebalancing decisions
4. **Tax Awareness**: Harvest losses, defer gains
5. **Regular Review**: Re-optimize monthly, rebalance as needed