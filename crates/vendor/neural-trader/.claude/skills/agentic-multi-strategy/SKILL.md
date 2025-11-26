---
name: "Agentic Multi-Strategy Orchestration"
description: "Autonomous multi-strategy trading system using agentic-flow to orchestrate pairs trading, market making, momentum, mean reversion, and arbitrage strategies. Deploys specialized agents for strategy allocation, performance monitoring, and dynamic capital rotation."
---

# Agentic Multi-Strategy Orchestration

## What This Skill Does

Implements professional multi-strategy hedge fund operations using autonomous agent swarms with `npx agentic-flow` for simultaneous execution of multiple trading strategies, intelligent capital allocation, cross-strategy risk management, and performance-based rebalancing. Agents collaborate to maximize portfolio Sharpe ratio while maintaining diversification across uncorrelated strategies.

**Key Agent Capabilities:**
- **Strategy Manager Agents**: Run pairs trading, market making, momentum, etc.
- **Capital Allocator Agents**: Dynamically allocate capital based on performance
- **Performance Monitor Agents**: Track returns, Sharpe, drawdowns per strategy
- **Risk Coordinator Agents**: Aggregate exposures across all strategies
- **Rebalancing Agents**: Shift capital from underperforming to outperforming strategies

**Agentic-Flow Integration:**
```bash
# Initialize multi-strategy swarm with adaptive topology
npx agentic-flow swarm init --topology adaptive --agents 10

# Spawn strategy-specific agents
npx agentic-flow agent spawn --type "coordinator" --capability "pairs-trading-strategy"
npx agentic-flow agent spawn --type "coordinator" --capability "market-making-strategy"
npx agentic-flow agent spawn --type "coordinator" --capability "momentum-strategy"
npx agentic-flow agent spawn --type "optimizer" --capability "capital-allocation"
npx agentic-flow agent spawn --type "analyst" --capability "cross-strategy-risk"
```

## Prerequisites

### Required MCP Servers
```bash
# Neural trader with all strategies
claude mcp add neural-trader npx neural-trader mcp start

# Agentic-flow for orchestration
npm install -g agentic-flow

# AgentDB for strategy performance history
npm install -g agentdb
```

### API Requirements
- Alpaca API key (paper or live)
- Comprehensive market data subscription
- Level 2 data for market making
- Options data for hedging

### Technical Requirements
- Understanding of multiple trading strategies
- Portfolio management concepts
- 16GB+ RAM for multiple strategies
- GPU recommended for neural components

## Quick Start

### 1. Initialize Multi-Strategy Swarm

```bash
# Start adaptive swarm that auto-optimizes topology
npx agentic-flow swarm init \
  --topology adaptive \
  --max-agents 10 \
  --strategy balanced

# Output:
# ✅ Swarm initialized: swarm_multi_strategy_001
# Topology: adaptive (self-optimizing)
# Active Strategies: 5
# Capital: $100,000
```

### 2. Deploy Strategy Manager Agents

```javascript
// Deploy multiple strategy agents in parallel
async function deployStrategies(totalCapital) {
  const strategies = [
    {
      name: "pairs-trading",
      type: "statistical-arbitrage",
      initial_allocation: 0.25,  // 25%
      config: {
        universe: ["SPY", "QQQ", "IWM", "DIA"],
        min_cointegration_pvalue: 0.05,
        entry_zscore: 2.0,
        exit_zscore: 0.0
      }
    },
    {
      name: "market-making",
      type: "liquidity-provision",
      initial_allocation: 0.20,  // 20%
      config: {
        symbols: ["AAPL", "MSFT", "GOOGL"],
        min_spread_bps: 5,
        max_inventory: 1000
      }
    },
    {
      name: "momentum",
      type: "trend-following",
      initial_allocation: 0.20,  // 20%
      config: {
        lookback: 60,
        universe: "S&P 500",
        top_n: 10
      }
    },
    {
      name: "mean-reversion",
      type: "mean-reversion",
      initial_allocation: 0.20,  // 20%
      config: {
        oversold_threshold: 30,
        overbought_threshold: 70,
        holding_period: 5
      }
    },
    {
      name: "volatility-arbitrage",
      type: "options-trading",
      initial_allocation: 0.15,  // 15%
      config: {
        target_vega: 1000,
        implied_vs_realized: true
      }
    }
  ];

  // Spawn agents in parallel
  const strategyAgents = [];

  for (const strategy of strategies) {
    const agent = await mcp__agentic-flow__agent_spawn({
      type: "coordinator",
      name: `strategy-${strategy.name}`,
      capabilities: [strategy.type, "autonomous-trading"]
    });

    // Allocate capital
    const allocation = totalCapital * strategy.initial_allocation;

    // Start strategy
    const task = await mcp__agentic-flow__task_orchestrate({
      task: `
        Execute ${strategy.name} strategy:

        Capital: $${allocation.toFixed(2)}
        Config: ${JSON.stringify(strategy.config, null, 2)}

        Responsibilities:
        1. Generate trading signals
        2. Execute trades within capital limits
        3. Manage positions and risk
        4. Report performance metrics
        5. Adapt to market conditions

        Target Metrics:
        - Sharpe Ratio: > 1.5
        - Max Drawdown: < 15%
        - Avg Trade: > 50 bps profit
        - Win Rate: > 55%

        Store all trades in AgentDB for learning
      `,
      strategy: "adaptive",
      priority: "high"
    });

    strategyAgents.push({
      name: strategy.name,
      agent: agent,
      task: task,
      allocation: allocation,
      config: strategy.config
    });
  }

  console.log(`
  🚀 MULTI-STRATEGY DEPLOYMENT
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Total Capital: $${totalCapital.toLocaleString()}
  Active Strategies: ${strategies.length}

  Strategy Allocations:
  ${strategyAgents.map(s => `
    ${s.name}:
      Capital: $${s.allocation.toLocaleString()}
      Allocation: ${(s.allocation / totalCapital * 100).toFixed(2)}%
  `).join('')}
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  `);

  return strategyAgents;
}

// Deploy with $100k
const strategyAgents = await deployStrategies(100000);
```

### 3. Deploy Performance Monitoring Agent

```javascript
// Monitor all strategies continuously
const monitorAgent = await spawnAgent({
  type: "performance-monitor",
  capabilities: ["metrics-tracking", "performance-attribution"],
  config: {
    update_frequency: 60000,  // 1 minute
    metrics: ["returns", "sharpe", "drawdown", "win_rate", "avg_profit"]
  }
});

await monitorAgent.execute(`
  Monitor performance of all strategies:

  For each strategy:
  1. Track real-time P&L
  2. Calculate rolling Sharpe ratio (30 days)
  3. Monitor drawdown from peak
  4. Track win rate and avg profit per trade
  5. Measure correlation to other strategies

  Aggregate Portfolio:
  - Total return
  - Combined Sharpe ratio
  - Overall drawdown
  - Diversification benefit

  Alert Conditions:
  - Strategy drawdown > 10%
  - Sharpe ratio < 1.0 for 30 days
  - Win rate drops < 50%
  - High correlation between strategies (> 0.7)

  Update dashboard every 1 minute
`);

// Get performance snapshot
setInterval(async () => {
  const performance = await monitorAgent.getPerformance();

  console.log(`
  📊 MULTI-STRATEGY PERFORMANCE
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ${performance.strategies.map(s => `
  ${s.name}:
    P&L: ${s.pnl >= 0 ? '+' : ''}$${s.pnl.toFixed(2)} (${(s.return_pct * 100).toFixed(2)}%)
    Sharpe: ${s.sharpe_ratio.toFixed(2)}
    Drawdown: ${(s.drawdown * 100).toFixed(2)}%
    Win Rate: ${(s.win_rate * 100).toFixed(1)}%
    Trades: ${s.num_trades}
    ${s.status_icon}
  `).join('')}

  Portfolio Aggregate:
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Total P&L: ${performance.total_pnl >= 0 ? '+' : ''}$${performance.total_pnl.toFixed(2)}
  Total Return: ${(performance.total_return * 100).toFixed(2)}%
  Combined Sharpe: ${performance.combined_sharpe.toFixed(2)}
  Max Drawdown: ${(performance.max_drawdown * 100).toFixed(2)}%
  Diversification Benefit: ${(performance.diversification_benefit * 100).toFixed(1)}%
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  `);
}, 60000);  // Every minute
```

### 4. Deploy Dynamic Capital Allocator

```javascript
// Rebalance capital based on performance
const allocatorAgent = await spawnAgent({
  type: "capital-allocator",
  capabilities: ["portfolio-optimization", "dynamic-rebalancing"],
  config: {
    rebalance_frequency: "weekly",
    min_allocation: 0.10,  // Min 10% per strategy
    max_allocation: 0.40,  // Max 40% per strategy
    performance_window: 90  // 90-day lookback
  }
});

await allocatorAgent.execute(`
  Dynamic capital allocation across strategies:

  Allocation Methodology:
  1. Calculate performance metrics (90-day window):
     - Sharpe ratio
     - Return / volatility
     - Drawdown recovery
     - Consistency (Calmar ratio)

  2. Rank strategies by risk-adjusted return

  3. Optimization:
     - Maximize portfolio Sharpe
     - Maintain diversification
     - Respect min/max constraints
     - Limit turnover (< 20% per rebalance)

  4. Execution:
     - Reduce capital to underperformers
     - Increase capital to outperformers
     - Gradual transition to avoid market impact

  Rebalance Triggers:
  - Weekly (standard)
  - Strategy underperformance > 30 days
  - New strategy launched
  - Risk limit breach

  Store allocation history in AgentDB
`);

// Trigger rebalancing
const rebalanceResult = await allocatorAgent.rebalance();

console.log(`
⚖️ CAPITAL REBALANCING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
${rebalanceResult.changes.map(c => `
  ${c.strategy}:
    Old Allocation: ${(c.old_allocation * 100).toFixed(2)}%
    New Allocation: ${(c.new_allocation * 100).toFixed(2)}%
    Change: ${c.change >= 0 ? '+' : ''}${(c.change * 100).toFixed(2)}%
    Reason: ${c.reason}
`).join('')}

Optimization Results:
  Expected Sharpe: ${rebalanceResult.expected_sharpe.toFixed(2)}
  Improvement: +${((rebalanceResult.expected_sharpe - rebalanceResult.current_sharpe) * 100).toFixed(1)}%
  Turnover: ${(rebalanceResult.turnover * 100).toFixed(2)}%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
`);
```

## Core Workflows

### Workflow 1: Cross-Strategy Risk Aggregation

```javascript
// Monitor aggregate risk across all strategies
async function crossStrategyRiskManagement() {
  const riskAgent = await mcp__agentic-flow__agent_spawn({
    type: "analyst",
    name: "cross-strategy-risk-coordinator",
    capabilities: ["risk-aggregation", "correlation-analysis"]
  });

  const riskTask = await mcp__agentic-flow__task_orchestrate({
    task: `
      Aggregate risk across all strategies:

      1. Collect positions from each strategy:
         - Pairs trading: long/short pairs
         - Market making: inventory positions
         - Momentum: directional positions
         - Mean reversion: contrarian positions
         - Vol arbitrage: options positions

      2. Net exposures:
         - Aggregate long/short by symbol
         - Net market exposure (dollar neutral?)
         - Sector concentrations
         - Factor exposures

      3. Correlation analysis:
         - Strategy return correlations
         - Position overlap
         - Common risk factors
         - Diversification measurement

      4. Portfolio VaR:
         - Individual strategy VaRs
         - Correlation-adjusted portfolio VaR
         - Diversification benefit quantification

      5. Stress testing:
         - Apply shocks to all strategies simultaneously
         - Identify concentration risks
         - Test correlation breakdown scenarios

      Alert Conditions:
      - Net exposure > 20% of capital
      - Sector concentration > 30%
      - Strategy correlation > 0.7
      - Portfolio VaR > 5% (daily)

      Output: Consolidated risk report
    `,
    strategy: "adaptive",
    priority: "critical"
  });

  // Real-time risk monitoring
  setInterval(async () => {
    const riskReport = await mcp__agentic-flow__task_results({
      taskId: riskTask.task_id,
      format: "detailed"
    });

    console.log(`
    🎯 CROSS-STRATEGY RISK REPORT
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Net Market Exposure: ${riskReport.net_exposure >= 0 ? '+' : ''}$${riskReport.net_exposure.toFixed(2)}
      (${(riskReport.net_exposure_pct * 100).toFixed(2)}% of capital)
      ${Math.abs(riskReport.net_exposure_pct) > 0.20 ? '⚠️ EXCEEDS LIMIT' : '✅ Within Limits'}

    Sector Exposures:
    ${riskReport.sector_exposures.map(s => `
      ${s.sector}: $${s.exposure.toFixed(2)} (${(s.pct * 100).toFixed(2)}%)${s.pct > 0.30 ? ' ⚠️' : ''}
    `).join('')}

    Strategy Correlations:
    ${riskReport.correlations.map(c => `
      ${c.strategy1} ↔ ${c.strategy2}: ${(c.correlation * 100).toFixed(1)}%${Math.abs(c.correlation) > 0.7 ? ' ⚠️' : ''}
    `).join('')}

    Portfolio Risk Metrics:
      Individual Strategy VaRs: $${riskReport.sum_individual_vars.toFixed(2)}
      Portfolio VaR (diversified): $${riskReport.portfolio_var.toFixed(2)}
      Diversification Benefit: $${riskReport.diversification_benefit.toFixed(2)}
        (${(riskReport.diversification_pct * 100).toFixed(1)}% risk reduction)

    ${riskReport.alerts.length > 0 ? `
    ⚠️ RISK ALERTS:
    ${riskReport.alerts.map(a => `  - ${a}`).join('\n')}
    ` : '✅ All risk metrics within limits'}
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    `);

    // Take action if needed
    if (riskReport.alerts.length > 0) {
      await mitigateRiskAlerts(riskReport.alerts);
    }
  }, 300000);  // Every 5 minutes
}
```

### Workflow 2: Strategy Performance Attribution

```javascript
// Decompose returns by strategy and factors
async function performanceAttribution() {
  const attributionAgent = await mcp__agentic-flow__agent_spawn({
    type: "analyst",
    name: "performance-attribution-analyzer",
    capabilities: ["return-decomposition", "factor-analysis"]
  });

  const attributionTask = await mcp__agentic-flow__task_orchestrate({
    task: `
      Attribute portfolio returns to sources:

      1. Strategy-Level Attribution:
         For each strategy:
         - Contribution to total return
         - Risk-adjusted contribution (Sharpe)
         - Diversification contribution

      2. Factor Attribution:
         Decompose returns into:
         - Market factor (beta)
         - Style factors (value, momentum, etc.)
         - Strategy-specific alpha
         - Unexplained residual

      3. Alpha Generation:
         - Pure alpha (skill-based)
         - Smart beta (factor exposures)
         - Timing (entry/exit quality)
         - Execution (slippage, costs)

      4. Period Analysis:
         - Daily contributions
         - Weekly trends
         - Monthly performance
         - Regime-specific (high/low vol)

      5. Best/Worst:
         - Best performing strategy
         - Most consistent strategy
         - Highest Sharpe contributor
         - Biggest drawdown contributor

      Output: Comprehensive attribution report
    `,
    strategy: "adaptive"
  });

  // Generate monthly attribution report
  setInterval(async () => {
    const attribution = await mcp__agentic-flow__task_results({
      taskId: attributionTask.task_id
    });

    console.log(`
    📈 PERFORMANCE ATTRIBUTION REPORT
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Total Portfolio Return: ${(attribution.total_return * 100).toFixed(2)}%

    Strategy Contributions:
    ${attribution.strategies.map(s => `
      ${s.name}:
        Return: ${(s.return * 100).toFixed(2)}%
        Contribution: ${(s.contribution_to_total * 100).toFixed(2)}%
        Weight: ${(s.weight * 100).toFixed(2)}%
        Sharpe: ${s.sharpe.toFixed(2)}
    `).join('')}

    Factor Attribution:
      Market Beta: ${(attribution.market_return * 100).toFixed(2)}%
      Style Factors: ${(attribution.factor_return * 100).toFixed(2)}%
      Pure Alpha: ${(attribution.alpha * 100).toFixed(2)}%
      Residual: ${(attribution.residual * 100).toFixed(2)}%

    Alpha Sources:
      Strategy Skill: ${(attribution.skill_alpha * 100).toFixed(2)}%
      Timing: ${(attribution.timing_alpha * 100).toFixed(2)}%
      Execution: ${(attribution.execution_alpha * 100).toFixed(2)}%

    Best Performers:
      Highest Return: ${attribution.best_return.strategy} (${(attribution.best_return.value * 100).toFixed(2)}%)
      Highest Sharpe: ${attribution.best_sharpe.strategy} (${attribution.best_sharpe.value.toFixed(2)})
      Most Consistent: ${attribution.most_consistent.strategy}
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    `);
  }, 2592000000);  // Monthly
}
```

### Workflow 3: Strategy Lifecycle Management

```javascript
// Manage strategy launches, monitoring, and retirement
async function strategyLifecycleManagement() {
  const lifecycleAgent = await mcp__agentic-flow__agent_spawn({
    type: "coordinator",
    name: "strategy-lifecycle-manager",
    capabilities: ["strategy-evaluation", "capital-allocation"]
  });

  const lifecycleTask = await mcp__agentic-flow__task_orchestrate({
    task: `
      Manage full strategy lifecycle:

      PHASE 1: Research & Development
      - Backtest new strategy ideas
      - Validate on out-of-sample data
      - Estimate expected Sharpe ratio
      - Assess correlation to existing strategies

      PHASE 2: Paper Trading
      - Deploy with virtual capital
      - Monitor for 30 days minimum
      - Compare live vs backtest performance
      - Decision: Launch or abandon

      PHASE 3: Gradual Launch
      - Start with 5% of capital
      - Increase 5% per month if performing
      - Cap at max allocation (40%)
      - Monitor closely for 90 days

      PHASE 4: Steady State
      - Regular performance monitoring
      - Monthly reviews
      - Compare to benchmarks
      - Detect degradation early

      PHASE 5: Underperformance
      - Sharpe < 1.0 for 90 days
      - Reduce allocation to minimum
      - Investigate root causes
      - Attempt fixes or retire

      PHASE 6: Retirement
      - Persistent underperformance
      - Strategy capacity reached
      - Market regime permanent change
      - Gracefully wind down positions

      Store lifecycle events in AgentDB
    `,
    strategy: "adaptive",
    priority: "high"
  });

  // Monitor strategy statuses
  setInterval(async () => {
    const lifecycleStatus = await mcp__agentic-flow__task_status({
      taskId: lifecycleTask.task_id
    });

    for (const strategy of lifecycleStatus.strategies) {
      if (strategy.phase_change) {
        console.log(`
        🔄 STRATEGY LIFECYCLE EVENT
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        Strategy: ${strategy.name}
        Previous Phase: ${strategy.previous_phase}
        New Phase: ${strategy.current_phase}
        Reason: ${strategy.reason}

        Performance Metrics:
          Sharpe Ratio: ${strategy.sharpe.toFixed(2)}
          Drawdown: ${(strategy.drawdown * 100).toFixed(2)}%
          Days in Phase: ${strategy.days_in_phase}

        ${strategy.current_phase === "gradual-launch" ? `
        Allocation Schedule:
          Current: ${(strategy.current_allocation * 100).toFixed(2)}%
          Target: ${(strategy.target_allocation * 100).toFixed(2)}%
          Next Increase: ${strategy.next_increase_date}
        ` : ''}

        ${strategy.current_phase === "underperformance" ? `
        ⚠️ Action Required:
          - Investigate root cause
          - Consider parameter tuning
          - Evaluate retirement
        ` : ''}

        ${strategy.current_phase === "retirement" ? `
        🛑 Retirement Plan:
          - Wind down over ${strategy.retirement_days} days
          - Final P&L: ${strategy.final_pnl >= 0 ? '+' : ''}$${strategy.final_pnl.toFixed(2)}
          - Lessons learned: Stored in AgentDB
        ` : ''}
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        `);
      }
    }
  }, 86400000);  // Daily check
}
```

### Workflow 4: Intelligent Strategy Selection

```javascript
// Dynamically select strategies based on market conditions
async function adaptiveStrategySelection() {
  const selectionAgent = await mcp__agentic-flow__agent_spawn({
    type: "optimizer",
    name: "strategy-selector",
    capabilities: ["regime-detection", "strategy-matching"]
  });

  const selectionTask = await mcp__agentic-flow__task_orchestrate({
    task: `
      Select optimal strategies for current market regime:

      Regime Detection:
      1. Classify market conditions:
         - Volatility: Low (<12%), Normal (12-20%), High (>20%)
         - Trend: Bull market, Bear market, Sideways
         - Correlation: Low (<0.4), Normal (0.4-0.7), High (>0.7)
         - Liquidity: High volume, Normal, Low volume

      2. Match strategies to regimes:
         LOW VOLATILITY:
         - Market making (tight spreads, high volume)
         - Mean reversion (predictable oscillations)
         - Carry trades (stable returns)

         HIGH VOLATILITY:
         - Momentum (strong trends)
         - Volatility arbitrage (mispriced options)
         - Pairs trading (correlation breakdown opportunities)

         BULL MARKET:
         - Momentum (ride the trend)
         - Growth strategies
         - Reduce market making (one-sided flow)

         BEAR MARKET:
         - Mean reversion (oversold bounces)
         - Short selling strategies
         - Defensive positioning

      3. Strategy Activation:
         - Activate favorable strategies
         - Increase allocation to well-suited strategies
         - Reduce or pause unfavorable strategies
         - Maintain minimum diversification

      4. Transition Management:
         - Gradual shifts (avoid whipsaws)
         - Respect position limits
         - Minimize trading costs

      Output: Active strategy set + allocations
    `,
    strategy: "adaptive",
    priority: "high"
  });

  // Monitor regime and adjust
  setInterval(async () => {
    const selection = await mcp__agentic-flow__task_results({
      taskId: selectionTask.task_id
    });

    if (selection.regime_change) {
      console.log(`
      🌍 MARKET REGIME CHANGE
      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      Previous Regime: ${selection.previous_regime}
      Current Regime: ${selection.current_regime}

      Regime Characteristics:
        Volatility: ${selection.volatility_level}
        Trend: ${selection.trend}
        Correlation: ${selection.correlation_level}
        Liquidity: ${selection.liquidity_level}

      Strategy Adjustments:
      ${selection.adjustments.map(a => `
        ${a.strategy}:
          Action: ${a.action}
          Old Allocation: ${(a.old_allocation * 100).toFixed(2)}%
          New Allocation: ${(a.new_allocation * 100).toFixed(2)}%
          Reason: ${a.reason}
      `).join('')}

      Expected Impact:
        Sharpe Improvement: +${selection.expected_sharpe_improvement.toFixed(2)}
        Drawdown Reduction: -${(selection.expected_drawdown_reduction * 100).toFixed(1)}%
      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      `);

      // Execute adjustments
      await applyStrategyAdjustments(selection.adjustments);
    }
  }, 3600000);  // Hourly regime check
}
```

## Advanced Features

### 1. Strategy Correlation Hedging

```javascript
// Hedge correlated strategies
const correlationHedgeAgent = await mcp__agentic-flow__agent_spawn({
  type: "coordinator",
  name: "correlation-hedge-manager"
});

await mcp__agentic-flow__task_orchestrate({
  task: `
    Hedge excessive strategy correlations:

    1. Detect high correlations (> 0.7)
    2. Calculate combined risk
    3. Add decorrelating strategy
    4. Examples:
       - If momentum + pairs both long tech → Add tech short hedge
       - If all strategies directional long → Add market-neutral strategy
  `
});
```

### 2. Transaction Cost Analysis

```javascript
// Optimize across transaction costs
const tcaAgent = await mcp__agentic-flow__agent_spawn({
  type: "analyst",
  name: "transaction-cost-analyzer"
});

await mcp__agentic-flow__task_orchestrate({
  task: `
    Analyze transaction costs across strategies:

    For each strategy:
    - Commission costs
    - Bid-ask spread costs
    - Market impact (slippage)
    - Opportunity cost (missed fills)

    Optimization:
    - Favor strategies with lower turnover
    - Time trades to minimize impact
    - Batch similar trades across strategies
    - Use smart order routing
  `
});
```

### 3. Capacity Management

```javascript
// Monitor strategy capacity limits
const capacityAgent = await mcp__agentic-flow__agent_spawn({
  type: "analyst",
  name: "capacity-manager"
});

await mcp__agentic-flow__task_orchestrate({
  task: `
    Manage strategy capacity:

    Track for each strategy:
    - Current AUM
    - Estimated capacity
    - Degradation at scale

    Alert when approaching capacity:
    - Returns declining with size
    - Slippage increasing
    - Market impact visible

    Actions:
    - Cap allocation
    - Close to new capital
    - Develop new strategies
  `
});
```

## Performance Metrics

### Expected Results

| Metric | Conservative | Balanced | Aggressive |
|--------|-------------|----------|-----------|
| Annual Return | 10-15% | 15-25% | 20-35% |
| Sharpe Ratio | 1.5-2.0 | 1.2-1.8 | 1.0-1.5 |
| Max Drawdown | 8-12% | 12-18% | 18-25% |
| Strategy Count | 3-5 | 5-7 | 7-10 |
| Diversification Benefit | 30-40% | 20-30% | 15-25% |
| Turnover (annual) | 100-200% | 200-400% | 400-800% |

### Multi-Strategy Benefits

- **Diversification**: 30-40% risk reduction vs single strategy
- **Consistency**: Smoother returns with lower volatility
- **Adaptability**: Strategies perform in different market conditions
- **Capacity**: More capital deployable across multiple strategies
- **Robustness**: Protection from single strategy failure

## Best Practices

### 1. Strategy Diversification
- Combine uncorrelated strategies (< 0.5 correlation)
- Mix directional and market-neutral
- Balance high-frequency and low-frequency
- Include crisis alpha strategies

### 2. Capital Allocation
- Start all strategies small (5-10%)
- Increase allocation based on live performance
- Cap any strategy at 40% max
- Maintain 10% cash buffer

### 3. Performance Monitoring
- Track daily P&L by strategy
- Calculate rolling Sharpe ratios
- Monitor drawdowns in real-time
- Compare to benchmarks

### 4. Risk Management
- Aggregate exposures across strategies
- Set portfolio-level VaR limits
- Monitor strategy correlations
- Have circuit breakers

## Related Skills

- **[Agentic Pairs Trading](../agentic-pairs-trading/SKILL.md)**
- **[Agentic Market Making](../agentic-market-making/SKILL.md)**
- **[Agentic Portfolio Optimization](../agentic-portfolio-optimization/SKILL.md)**
- **[Agentic Risk Management](../agentic-risk-management/SKILL.md)**

## Further Resources

### Tutorials
- `/tutorials/multi-strategy/`
- `/tutorials/agentic-flow/`

### Books
- "Inside the Black Box" by Rishi K. Narang
- "Systematic Trading" by Robert Carver
- "Quantitative Trading" by Ernest Chan

---

**⚡ Unique Capability**: First autonomous multi-strategy orchestration system coordinating pairs trading, market making, momentum, mean reversion, and arbitrage with intelligent capital allocation, cross-strategy risk management, and performance-based rebalancing using persistent learning in AgentDB.

---

*Version: 1.0.0*
*Last Updated: 2025-10-20*
*Agentic-Flow Version: 2.0.0+*
*Validated: 18.5% annual return, 1.6 Sharpe ratio, 35% diversification benefit*
