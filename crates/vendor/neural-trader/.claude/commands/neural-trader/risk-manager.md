# Neural Risk Manager Command

Real-time portfolio risk monitoring and automated hedging using the neural-risk-manager agent.

## Agent Location
`.claude/agents/neural-trader/risk-manager/neural-risk-manager.md`

## Command

```javascript
// Spawn the neural risk manager agent
Task(
  "Risk Management",
  "Use the agent at .claude/agents/neural-trader/risk-manager/neural-risk-manager.md to monitor portfolio risk for {portfolio} with real-time VaR/CVaR calculations and automated hedging. The agent should implement emergency protocols and compliance monitoring.",
  "researcher"
)
```

## Direct Agent Invocation

```bash
# Using Claude's agent system
claude agent run .claude/agents/neural-trader/risk-manager/neural-risk-manager.md --portfolio ALL

# Or via Task with explicit path
Task(
  "Risk Manager",
  "Load and execute the neural-risk-manager agent from .claude/agents/neural-trader/risk-manager/neural-risk-manager.md for portfolio monitoring",
  "base-template-generator"
)
```

## Parameters

- `{portfolio}` - Portfolio or assets to monitor (e.g., "ALL", "TECH_SECTOR", "OPTIONS")

## Full Workflow with Agent

```javascript
// Step 1: Initialize risk management
mcp__claude-flow__swarm_init({
  topology: "hierarchical",
  maxAgents: 8,
  strategy: "specialized"
})

// Step 2: Load the neural-risk-manager agent template
Task(
  "Load Risk Agent",
  "Read the agent configuration from .claude/agents/neural-trader/risk-manager/neural-risk-manager.md and initialize a risk management specialist",
  "base-template-generator"
)

// Step 3: Spawn agent with configuration
mcp__claude-flow__agent_spawn({
  type: "monitor",
  name: "neural-risk-manager",
  capabilities: ["risk_monitoring", "var_calculation", "hedging", "compliance"],
  config_path: ".claude/agents/neural-trader/risk-manager/neural-risk-manager.md"
})

// Step 4: Execute using agent's risk framework
mcp__claude-flow__task_orchestrate({
  task: "Monitor portfolio risk using the neural-risk-manager agent template",
  agent_config: ".claude/agents/neural-trader/risk-manager/neural-risk-manager.md",
  strategy: "parallel",
  priority: "critical"
})

// Real-time risk monitoring (1-second updates)
setInterval(() => {
  mcp__ai-news-trader__risk_analysis({
    portfolio: current_positions,
    var_confidence: 0.01,  // 99% VaR
    time_horizon: 1,  // 1 day
    use_monte_carlo: true,
    use_gpu: true
  })
}, 1000)

// Portfolio stress testing
mcp__ai-news-trader__risk_analysis({
  portfolio: current_positions,
  var_confidence: 0.01,
  time_horizon: 10,
  use_monte_carlo: true,
  use_gpu: true
})

// Get system health
mcp__ai-news-trader__monitor_strategy_health({
  strategy: "portfolio"
})
```

## Agent Configuration Reference

The neural-risk-manager agent at `.claude/agents/neural-trader/risk-manager/neural-risk-manager.md` includes:
- Real-time risk monitoring with 1-second updates
- VaR and CVaR calculations with GPU acceleration
- Monte Carlo stress testing with scenarios
- Automated hedging with options and futures
- Emergency stop-loss and circuit breakers
- Regulatory compliance monitoring

## Configuration

```yaml
risk_config:
  monitoring_interval_ms: 1000
  var_confidence: 0.01  # 99%
  cvar_confidence: 0.05  # 95%
  stress_scenarios: 20
  limits:
    max_var_daily: 0.02
    max_drawdown: 0.15
    max_correlation: 0.8
    max_leverage: 2.0
  hedging:
    automatic: true
    instruments: ["options", "futures", "etfs"]
    cost_threshold: 0.002
```

## Using Agent's Risk Strategies

```javascript
// The agent implements these specific strategies:

// 1. VaR Monitoring (from agent)
Task(
  "VaR Monitor",
  "Using .claude/agents/neural-trader/risk-manager/neural-risk-manager.md, calculate 99% VaR every second",
  "neural-risk-manager"
)

// 2. Stress Testing (from agent)
Task(
  "Stress Test",
  "Apply neural-risk-manager agent's stress scenarios from .claude/agents/neural-trader/risk-manager/",
  "neural-risk-manager"
)

// 3. Automated Hedging (from agent)
Task(
  "Auto Hedge",
  "Use neural-risk-manager agent to automatically hedge portfolio delta",
  "neural-risk-manager"
)

// 4. Emergency Protocol (from agent)
Task(
  "Emergency Stop",
  "Execute neural-risk-manager agent's emergency stop-loss protocol",
  "neural-risk-manager"
)
```

## Agent's Risk Limits

The agent at `.claude/agents/neural-trader/risk-manager/neural-risk-manager.md` enforces:
- Daily VaR limit: 2% of portfolio
- Maximum drawdown: 15%
- Position concentration: < 10% per asset
- Sector concentration: < 30% per sector
- Correlation limit: < 0.8 between positions
- Leverage limit: 2x maximum

## Agent's Emergency Protocols

```javascript
// Using the agent's emergency system
Task(
  "Emergency Protocol",
  "Apply neural-risk-manager agent's emergency protocols: reduce all positions by 50% if VaR > 3%",
  "neural-risk-manager"
)

// Circuit breaker activation
Task(
  "Circuit Breaker",
  "Implement neural-risk-manager agent's circuit breaker: stop all trading if loss > 5% in 1 hour",
  "neural-risk-manager"
)
```

## Agent's Hedging Framework

```javascript
// Delta hedging with agent
Task(
  "Delta Hedge",
  "Use .claude/agents/neural-trader/risk-manager/neural-risk-manager.md to maintain delta-neutral portfolio",
  "neural-risk-manager"
)

// Tail risk hedging
Task(
  "Tail Hedge",
  "Apply neural-risk-manager agent's tail risk hedging with OTM puts",
  "neural-risk-manager"
)

// Dynamic hedging
Task(
  "Dynamic Hedge",
  "Execute neural-risk-manager agent's dynamic hedging based on market volatility",
  "neural-risk-manager"
)
```

## Real-Time Monitoring

```javascript
// Monitor with agent's framework
Task(
  "Real-Time Monitor",
  "Use neural-risk-manager agent from .claude/agents/neural-trader/risk-manager/ for continuous monitoring",
  "neural-risk-manager"
)

// Alert system
Task(
  "Risk Alerts",
  "Apply neural-risk-manager agent's alert thresholds and notification system",
  "neural-risk-manager"
)
```

## Compliance Monitoring

```javascript
// Regulatory compliance with agent
Task(
  "Compliance Check",
  "Use .claude/agents/neural-trader/risk-manager/neural-risk-manager.md to verify regulatory compliance",
  "neural-risk-manager"
)

// Position limits
Task(
  "Position Limits",
  "Apply neural-risk-manager agent's position limit enforcement",
  "neural-risk-manager"
)
```

## Performance Monitoring

```javascript
// Track agent's effectiveness
Task(
  "Agent Performance",
  "Monitor effectiveness of neural-risk-manager agent from .claude/agents/neural-trader/risk-manager/",
  "performance-analyzer"
)

// Risk-adjusted returns
Task(
  "Risk Metrics",
  "Calculate Sharpe, Sortino, and Calmar ratios using agent's framework",
  "neural-risk-manager"
)
```

## Agent Success Metrics

The neural-risk-manager agent targets:
- VaR Breaches: < 1% of days
- Drawdown Events: < 2 per year
- Hedging Efficiency: > 85%
- Alert Accuracy: > 90%
- Response Time: < 1 second

## Example Usage

### Basic Risk Monitoring with Agent
```javascript
Task(
  "Basic Monitor",
  "Load .claude/agents/neural-trader/risk-manager/neural-risk-manager.md and monitor portfolio risk",
  "researcher"
)
```

### Stress Testing with Agent
```javascript
Task(
  "Stress Portfolio",
  "Execute .claude/agents/neural-trader/risk-manager/neural-risk-manager.md stress testing scenarios",
  "researcher"
)
```

### Automated Hedging with Agent
```javascript
Task(
  "Auto Hedge",
  "Use neural-risk-manager agent from .claude/agents/neural-trader/risk-manager/ for automated hedging",
  "researcher"
)
```

## Agent's Risk Calculation Framework

The agent uses sophisticated risk metrics:
1. **Value at Risk (VaR)**: Historical, parametric, and Monte Carlo
2. **Conditional VaR (CVaR)**: Expected shortfall beyond VaR
3. **Maximum Drawdown**: Peak-to-trough analysis
4. **Greeks Management**: Delta, gamma, vega, theta monitoring
5. **Correlation Risk**: Cross-asset correlation monitoring

## Tips for Using the Agent

1. **Trust Agent's Limits**: The agent enforces risk limits for protection
2. **Use Agent's Alerts**: The agent provides early warning signals
3. **Follow Agent's Hedging**: The agent optimizes hedge ratios
4. **Respect Agent's Emergency Protocols**: The agent protects capital
5. **Monitor Agent's Compliance**: The agent ensures regulatory adherence

## Related Agents

- `.claude/agents/neural-trader/portfolio-optimizer/neural-portfolio-optimizer.md`
- `.claude/agents/neural-trader/arbitrage/neural-arbitrageur.md`
- `.claude/agents/neural-trader/market-maker/neural-market-maker.md`