# Risk Analysis MCP Tool

## Overview
The `mcp__ai-news-trader__risk_analysis` tool provides comprehensive portfolio risk assessment using advanced quantitative methods including Value at Risk (VaR), Conditional VaR, Monte Carlo simulation, and stress testing. GPU acceleration enables real-time risk computation for complex portfolios.

## Tool Specifications

### Tool Name
`mcp__ai-news-trader__risk_analysis`

### Purpose
- Calculate portfolio risk metrics (VaR, CVaR, volatility, beta)
- Run Monte Carlo simulations for tail risk assessment
- Perform stress testing under various market scenarios
- Analyze correlation and diversification benefits
- Identify concentration risks and hidden exposures
- Generate risk-adjusted performance metrics

## Parameters

### Required Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `portfolio` | array | Array of portfolio holdings with symbols and weights |

### Optional Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `time_horizon` | integer | 1 | Risk assessment time horizon in years |
| `use_gpu` | boolean | true | Enable GPU acceleration for faster computation |
| `use_monte_carlo` | boolean | true | Use Monte Carlo simulation for risk metrics |
| `var_confidence` | number | 0.05 | VaR confidence level (0.05 = 95% VaR) |

### Portfolio Format

```python
portfolio = [
    {"symbol": "AAPL", "weight": 0.3, "quantity": 100},
    {"symbol": "MSFT", "weight": 0.25, "quantity": 75},
    {"symbol": "GOOGL", "weight": 0.25, "quantity": 30},
    {"symbol": "AMZN", "weight": 0.2, "quantity": 40}
]
```

## Return Value Structure

```json
{
  "portfolio_summary": {
    "total_value": 125000,
    "number_of_positions": 4,
    "effective_diversification": 3.21,
    "concentration_risk": 0.3
  },
  "risk_metrics": {
    "portfolio_volatility": 0.1823,
    "portfolio_beta": 1.05,
    "value_at_risk": {
      "var_95": -12500,
      "var_99": -18750,
      "expected_shortfall": -21250
    },
    "conditional_var": {
      "cvar_95": -15625,
      "cvar_99": -23125
    },
    "downside_risk": {
      "downside_deviation": 0.1234,
      "sortino_ratio": 1.87,
      "omega_ratio": 1.45,
      "gain_loss_ratio": 1.32
    }
  },
  "correlation_analysis": {
    "correlation_matrix": [
      [1.00, 0.82, 0.75, 0.78],
      [0.82, 1.00, 0.79, 0.81],
      [0.75, 0.79, 1.00, 0.77],
      [0.78, 0.81, 0.77, 1.00]
    ],
    "average_correlation": 0.79,
    "diversification_ratio": 1.18,
    "principal_components": {
      "pc1_variance": 0.85,
      "pc2_variance": 0.08,
      "pc3_variance": 0.05
    }
  },
  "stress_testing": {
    "scenarios": {
      "market_crash": {
        "portfolio_loss": -0.35,
        "worst_performer": "AAPL",
        "best_performer": "MSFT"
      },
      "tech_selloff": {
        "portfolio_loss": -0.28,
        "sector_impact": -0.32
      },
      "interest_rate_rise": {
        "portfolio_loss": -0.12,
        "duration_impact": -0.08
      },
      "volatility_spike": {
        "portfolio_loss": -0.18,
        "vix_sensitivity": 1.23
      }
    }
  },
  "monte_carlo_results": {
    "simulations": 10000,
    "percentiles": {
      "p1": -0.2834,
      "p5": -0.1823,
      "p25": -0.0543,
      "p50": 0.0234,
      "p75": 0.0987,
      "p95": 0.2134,
      "p99": 0.3421
    },
    "probability_of_loss": 0.3421,
    "expected_maximum_drawdown": -0.2156,
    "tail_risk_metrics": {
      "left_tail_index": 2.34,
      "right_tail_index": 2.87,
      "kurtosis": 3.45
    }
  },
  "risk_contribution": {
    "marginal_var": {
      "AAPL": -4875,
      "MSFT": -3750,
      "GOOGL": -3125,
      "AMZN": -750
    },
    "component_var": {
      "AAPL": 0.39,
      "MSFT": 0.30,
      "GOOGL": 0.25,
      "AMZN": 0.06
    },
    "risk_parity_weights": {
      "AAPL": 0.20,
      "MSFT": 0.25,
      "GOOGL": 0.27,
      "AMZN": 0.28
    }
  },
  "recommendations": {
    "risk_level": "MEDIUM-HIGH",
    "suggested_hedges": ["SPY puts", "VIX calls"],
    "rebalancing_suggestions": [
      "Reduce AAPL allocation by 10%",
      "Increase defensive assets"
    ]
  },
  "execution_time": {
    "total_ms": 342,
    "monte_carlo_ms": 234,
    "gpu_speedup": "12.5x"
  }
}
```

## Advanced Usage Examples

### Basic Portfolio Risk Assessment
```python
# Simple portfolio risk analysis
result = await mcp.call_tool(
    "mcp__ai-news-trader__risk_analysis",
    {
        "portfolio": [
            {"symbol": "SPY", "weight": 0.6},
            {"symbol": "TLT", "weight": 0.4}
        ]
    }
)
```

### Multi-Asset Class Portfolio Analysis
```python
# Analyze diversified portfolio across asset classes
portfolio = [
    {"symbol": "SPY", "weight": 0.35, "asset_class": "equity"},
    {"symbol": "QQQ", "weight": 0.15, "asset_class": "equity"},
    {"symbol": "TLT", "weight": 0.20, "asset_class": "bonds"},
    {"symbol": "GLD", "weight": 0.15, "asset_class": "commodities"},
    {"symbol": "REIT", "weight": 0.10, "asset_class": "real_estate"},
    {"symbol": "BTC-USD", "weight": 0.05, "asset_class": "crypto"}
]

result = await mcp.call_tool(
    "mcp__ai-news-trader__risk_analysis",
    {
        "portfolio": portfolio,
        "time_horizon": 1,
        "use_monte_carlo": true,
        "var_confidence": 0.01  # 99% VaR
    }
)

# Analyze by asset class
asset_class_risk = {}
for asset_class in ["equity", "bonds", "commodities", "real_estate", "crypto"]:
    class_holdings = [h for h in portfolio if h.get("asset_class") == asset_class]
    if class_holdings:
        asset_class_risk[asset_class] = result["risk_contribution"]
```

### Dynamic Risk Monitoring
```python
# Real-time portfolio risk monitoring
async def monitor_portfolio_risk(portfolio, alert_thresholds):
    while True:
        # Get current risk metrics
        risk_analysis = await mcp.call_tool(
            "mcp__ai-news-trader__risk_analysis",
            {
                "portfolio": portfolio,
                "time_horizon": 1/252,  # Daily risk
                "use_gpu": true
            }
        )
        
        # Check alert conditions
        current_var = risk_analysis["risk_metrics"]["value_at_risk"]["var_95"]
        if abs(current_var) > alert_thresholds["var_limit"]:
            send_alert(f"VaR breach: {current_var}")
        
        current_volatility = risk_analysis["risk_metrics"]["portfolio_volatility"]
        if current_volatility > alert_thresholds["vol_limit"]:
            send_alert(f"Volatility spike: {current_volatility}")
        
        await asyncio.sleep(300)  # Check every 5 minutes
```

### Risk Parity Portfolio Construction
```python
# Build risk parity portfolio
initial_portfolio = [
    {"symbol": "SPY", "weight": 0.25},
    {"symbol": "TLT", "weight": 0.25},
    {"symbol": "GLD", "weight": 0.25},
    {"symbol": "REIT", "weight": 0.25}
]

# Iterate to find risk parity weights
for iteration in range(10):
    result = await mcp.call_tool(
        "mcp__ai-news-trader__risk_analysis",
        {
            "portfolio": initial_portfolio,
            "use_gpu": true
        }
    )
    
    # Update weights based on risk contribution
    new_weights = result["risk_contribution"]["risk_parity_weights"]
    
    for i, holding in enumerate(initial_portfolio):
        holding["weight"] = new_weights[holding["symbol"]]
    
    # Check convergence
    if max(result["risk_contribution"]["component_var"].values()) < 0.26:
        break

print(f"Risk parity achieved after {iteration + 1} iterations")
```

## Integration with Other Tools

### 1. Risk-Adjusted Strategy Optimization
```python
# Optimize strategy with risk constraints
optimization = await mcp.call_tool(
    "mcp__ai-news-trader__optimize_strategy",
    {
        "strategy": "portfolio_optimization",
        "symbol": "PORTFOLIO",
        "parameter_ranges": {
            "spy_weight": [0.2, 0.6],
            "tlt_weight": [0.2, 0.5],
            "gld_weight": [0.1, 0.3],
            "cash_weight": [0.0, 0.2]
        },
        "optimization_metric": "sharpe_ratio"
    }
)

# Verify risk constraints are met
portfolio = [
    {"symbol": "SPY", "weight": optimization["best_parameters"]["spy_weight"]},
    {"symbol": "TLT", "weight": optimization["best_parameters"]["tlt_weight"]},
    {"symbol": "GLD", "weight": optimization["best_parameters"]["gld_weight"]}
]

risk_check = await mcp.call_tool(
    "mcp__ai-news-trader__risk_analysis",
    {
        "portfolio": portfolio,
        "var_confidence": 0.05
    }
)

if abs(risk_check["risk_metrics"]["value_at_risk"]["var_95"]) > 50000:
    print("Risk limit exceeded, adjusting portfolio...")
```

### 2. Neural Forecast Risk Integration
```python
# Combine neural forecasts with risk analysis
symbols = ["AAPL", "MSFT", "GOOGL", "AMZN"]
forecasts = {}

# Get neural forecasts
for symbol in symbols:
    forecast = await mcp.call_tool(
        "mcp__ai-news-trader__neural_forecast",
        {
            "symbol": symbol,
            "horizon": 24,
            "confidence_level": 0.95
        }
    )
    forecasts[symbol] = forecast

# Build forward-looking portfolio
portfolio = []
for symbol in symbols:
    expected_return = forecasts[symbol]["forecast"]["expected_return"]
    confidence = forecasts[symbol]["forecast"]["confidence"]
    
    # Weight by expected return and confidence
    weight = (expected_return * confidence) / sum([f["forecast"]["expected_return"] * f["forecast"]["confidence"] for f in forecasts.values()])
    
    portfolio.append({
        "symbol": symbol,
        "weight": weight,
        "expected_return": expected_return
    })

# Analyze forward-looking risk
risk_analysis = await mcp.call_tool(
    "mcp__ai-news-trader__risk_analysis",
    {
        "portfolio": portfolio,
        "time_horizon": 1/12,  # Monthly
        "use_monte_carlo": true
    }
)
```

### 3. Correlation-Based Diversification
```python
# Find uncorrelated assets for diversification
base_portfolio = [
    {"symbol": "SPY", "weight": 0.5},
    {"symbol": "QQQ", "weight": 0.3},
    {"symbol": "IWM", "weight": 0.2}
]

# Analyze current correlations
base_risk = await mcp.call_tool(
    "mcp__ai-news-trader__risk_analysis",
    {
        "portfolio": base_portfolio
    }
)

high_correlation = base_risk["correlation_analysis"]["average_correlation"]

# Test adding diversifying assets
diversifiers = ["TLT", "GLD", "UUP", "VIX", "REIT"]
best_diversifier = None
lowest_correlation = high_correlation

for asset in diversifiers:
    test_portfolio = base_portfolio + [{"symbol": asset, "weight": 0.1}]
    
    # Renormalize weights
    total_weight = 1.1
    for holding in test_portfolio:
        holding["weight"] = holding["weight"] / total_weight
    
    test_risk = await mcp.call_tool(
        "mcp__ai-news-trader__risk_analysis",
        {
            "portfolio": test_portfolio
        }
    )
    
    if test_risk["correlation_analysis"]["average_correlation"] < lowest_correlation:
        lowest_correlation = test_risk["correlation_analysis"]["average_correlation"]
        best_diversifier = asset

print(f"Best diversifier: {best_diversifier} (reduces correlation to {lowest_correlation:.3f})")
```

## Performance Optimization Tips

### 1. GPU Acceleration for Monte Carlo
```python
# Maximize GPU efficiency for large simulations
large_portfolio = [{"symbol": f"STOCK_{i}", "weight": 1/100} for i in range(100)]

# Warm up GPU
warmup = await mcp.call_tool(
    "mcp__ai-news-trader__risk_analysis",
    {
        "portfolio": large_portfolio[:10],
        "use_monte_carlo": true,
        "use_gpu": true
    }
)

# Run full analysis with pre-warmed GPU
result = await mcp.call_tool(
    "mcp__ai-news-trader__risk_analysis",
    {
        "portfolio": large_portfolio,
        "use_monte_carlo": true,
        "use_gpu": true,
        "time_horizon": 1
    }
)

print(f"GPU speedup: {result['execution_time']['gpu_speedup']}")
```

### 2. Batch Risk Analysis
```python
# Analyze multiple portfolios efficiently
portfolios = {
    "conservative": [
        {"symbol": "TLT", "weight": 0.6},
        {"symbol": "SPY", "weight": 0.3},
        {"symbol": "GLD", "weight": 0.1}
    ],
    "moderate": [
        {"symbol": "SPY", "weight": 0.5},
        {"symbol": "TLT", "weight": 0.3},
        {"symbol": "REIT", "weight": 0.2}
    ],
    "aggressive": [
        {"symbol": "QQQ", "weight": 0.6},
        {"symbol": "ARKK", "weight": 0.3},
        {"symbol": "BTC-USD", "weight": 0.1}
    ]
}

# Parallel risk analysis
import asyncio

async def analyze_all_portfolios():
    tasks = []
    for name, portfolio in portfolios.items():
        task = mcp.call_tool(
            "mcp__ai-news-trader__risk_analysis",
            {
                "portfolio": portfolio,
                "use_gpu": true
            }
        )
        tasks.append((name, task))
    
    results = {}
    for name, task in tasks:
        results[name] = await task
    
    return results

all_risks = await analyze_all_portfolios()
```

### 3. Incremental Risk Updates
```python
# Efficiently update risk metrics for portfolio changes
base_portfolio = [
    {"symbol": "SPY", "weight": 0.6},
    {"symbol": "TLT", "weight": 0.4}
]

# Initial analysis
base_risk = await mcp.call_tool(
    "mcp__ai-news-trader__risk_analysis",
    {
        "portfolio": base_portfolio,
        "use_monte_carlo": true
    }
)

# Test incremental changes
changes = [
    {"action": "add", "symbol": "GLD", "weight": 0.1},
    {"action": "reweight", "symbol": "SPY", "new_weight": 0.5},
    {"action": "remove", "symbol": "TLT"}
]

for change in changes:
    # Apply change to portfolio
    modified_portfolio = apply_portfolio_change(base_portfolio, change)
    
    # Quick risk update (no Monte Carlo for speed)
    quick_risk = await mcp.call_tool(
        "mcp__ai-news-trader__risk_analysis",
        {
            "portfolio": modified_portfolio,
            "use_monte_carlo": false,
            "use_gpu": true
        }
    )
    
    print(f"Change: {change['action']} - New VaR: {quick_risk['risk_metrics']['value_at_risk']['var_95']}")
```

## Risk Management Best Practices

### 1. Stress Testing Framework
```python
# Comprehensive stress testing
stress_scenarios = {
    "2008_crisis": {
        "SPY": -0.37,
        "TLT": 0.12,
        "GLD": 0.24,
        "REIT": -0.42
    },
    "covid_crash": {
        "SPY": -0.34,
        "TLT": 0.08,
        "GLD": 0.07,
        "REIT": -0.38
    },
    "tech_bubble": {
        "QQQ": -0.78,
        "SPY": -0.49,
        "TLT": 0.15,
        "GLD": 0.02
    }
}

portfolio = [
    {"symbol": "SPY", "weight": 0.4},
    {"symbol": "QQQ", "weight": 0.2},
    {"symbol": "TLT", "weight": 0.2},
    {"symbol": "GLD", "weight": 0.1},
    {"symbol": "REIT", "weight": 0.1}
]

# Run stress tests
for scenario_name, scenario_returns in stress_scenarios.items():
    scenario_impact = sum(
        holding["weight"] * scenario_returns.get(holding["symbol"], -0.20)
        for holding in portfolio
    )
    
    print(f"{scenario_name}: Portfolio impact = {scenario_impact:.1%}")
    
    # Get detailed risk metrics under stress
    stress_risk = await mcp.call_tool(
        "mcp__ai-news-trader__risk_analysis",
        {
            "portfolio": portfolio,
            "use_monte_carlo": true
            # Stress parameters applied internally
        }
    )
```

### 2. Dynamic Hedging
```python
# Implement dynamic portfolio hedging
async def calculate_hedge_ratios(portfolio):
    # Analyze current risk
    risk_analysis = await mcp.call_tool(
        "mcp__ai-news-trader__risk_analysis",
        {
            "portfolio": portfolio,
            "use_monte_carlo": true
        }
    )
    
    # Calculate hedge ratios based on risk metrics
    portfolio_beta = risk_analysis["risk_metrics"]["portfolio_beta"]
    portfolio_value = risk_analysis["portfolio_summary"]["total_value"]
    
    # SPY put hedge
    spy_hedge_ratio = portfolio_beta * portfolio_value / 100  # SPY price ~$500
    
    # VIX call hedge for tail risk
    tail_risk = risk_analysis["monte_carlo_results"]["tail_risk_metrics"]["left_tail_index"]
    vix_hedge_ratio = (tail_risk - 2) * portfolio_value / 1000 if tail_risk > 2 else 0
    
    return {
        "spy_puts": int(spy_hedge_ratio),
        "vix_calls": int(vix_hedge_ratio),
        "hedge_cost": spy_hedge_ratio * 5 + vix_hedge_ratio * 2  # Rough option costs
    }

# Monitor and adjust hedges
portfolio = [{"symbol": "SPY", "weight": 0.7}, {"symbol": "QQQ", "weight": 0.3}]
hedges = await calculate_hedge_ratios(portfolio)
print(f"Recommended hedges: {hedges}")
```

### 3. Risk Budgeting
```python
# Implement risk budgeting framework
risk_budget = {
    "total_var_limit": 50000,
    "max_single_position_var": 15000,
    "max_sector_concentration": 0.40,
    "max_correlation": 0.85
}

async def check_risk_budget(portfolio, risk_budget):
    # Analyze portfolio risk
    risk_analysis = await mcp.call_tool(
        "mcp__ai-news-trader__risk_analysis",
        {
            "portfolio": portfolio,
            "var_confidence": 0.05
        }
    )
    
    violations = []
    
    # Check total VaR limit
    total_var = abs(risk_analysis["risk_metrics"]["value_at_risk"]["var_95"])
    if total_var > risk_budget["total_var_limit"]:
        violations.append(f"Total VaR ({total_var}) exceeds limit ({risk_budget['total_var_limit']})")
    
    # Check position concentration
    for symbol, marginal_var in risk_analysis["risk_contribution"]["marginal_var"].items():
        if abs(marginal_var) > risk_budget["max_single_position_var"]:
            violations.append(f"{symbol} VaR ({abs(marginal_var)}) exceeds position limit")
    
    # Check correlation
    avg_correlation = risk_analysis["correlation_analysis"]["average_correlation"]
    if avg_correlation > risk_budget["max_correlation"]:
        violations.append(f"Average correlation ({avg_correlation:.2f}) exceeds limit")
    
    return {
        "compliant": len(violations) == 0,
        "violations": violations,
        "risk_metrics": risk_analysis["risk_metrics"]
    }

# Example usage
tech_portfolio = [
    {"symbol": "AAPL", "weight": 0.3},
    {"symbol": "MSFT", "weight": 0.3},
    {"symbol": "GOOGL", "weight": 0.2},
    {"symbol": "AMZN", "weight": 0.2}
]

budget_check = await check_risk_budget(tech_portfolio, risk_budget)
if not budget_check["compliant"]:
    print("Risk budget violations:")
    for violation in budget_check["violations"]:
        print(f"  - {violation}")
```

## Common Issues and Solutions

### Issue: Monte Carlo Simulation Too Slow
**Solution**: Use GPU acceleration and optimize simulation parameters
```python
# Optimize Monte Carlo parameters
result = await mcp.call_tool(
    "mcp__ai-news-trader__risk_analysis",
    {
        "portfolio": large_portfolio,
        "use_monte_carlo": true,
        "use_gpu": true,
        # Internal optimizations:
        # - Reduced simulation count for initial assessment
        # - Adaptive sampling for tail events
        # - Parallel random number generation
    }
)
```

### Issue: Correlation Matrix Not Positive Definite
**Solution**: Use correlation matrix repair techniques
```python
# Handle correlation matrix issues
try:
    result = await mcp.call_tool(
        "mcp__ai-news-trader__risk_analysis",
        {
            "portfolio": complex_portfolio,
            "use_monte_carlo": true
        }
    )
except Exception as e:
    if "correlation matrix" in str(e):
        # Retry with correlation repair
        result = await mcp.call_tool(
            "mcp__ai-news-trader__risk_analysis",
            {
                "portfolio": complex_portfolio,
                "use_monte_carlo": false  # Use analytical methods
            }
        )
```

### Issue: Risk Metrics Unstable for Small Portfolios
**Solution**: Use appropriate time horizons and confidence levels
```python
# Adjust parameters for small portfolios
small_portfolio = [
    {"symbol": "SPY", "weight": 0.6},
    {"symbol": "TLT", "weight": 0.4}
]

# Use longer time horizon for stability
result = await mcp.call_tool(
    "mcp__ai-news-trader__risk_analysis",
    {
        "portfolio": small_portfolio,
        "time_horizon": 1,  # Annual instead of daily
        "var_confidence": 0.05,  # 95% instead of 99%
        "use_monte_carlo": true
    }
)
```

## See Also
- [Run Backtest Tool](run-backtest.md) - Historical risk validation
- [Optimize Strategy Tool](optimize-strategy.md) - Risk-constrained optimization
- [Correlation Analysis Tool](correlation-analysis.md) - Detailed correlation studies
- [Performance Report Tool](performance-report.md) - Risk-adjusted performance metrics