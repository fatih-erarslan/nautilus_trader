#!/bin/bash
# Claude Flow Command: Demo - Risk Management

echo "⚠️ Claude Flow: Risk Management Demo"
echo "===================================="
echo ""
echo "Advanced portfolio risk analysis with Monte Carlo simulations."
echo ""

cat << 'EOF'
### Risk Management Workflow

#### 1. Portfolio Overview
```
Use: mcp__ai-news-trader__get_portfolio_status
Parameters:
  include_analytics: true
```
Returns:
- Current positions and P&L
- Portfolio value and cash
- Performance metrics (Sharpe, Sortino)
- Risk metrics (VaR, Beta, Volatility)

#### 2. Correlation Analysis
```
Use: mcp__ai-news-trader__cross_asset_correlation_matrix
Parameters:
  assets: ["AAPL", "NVDA", "TSLA", "GOOGL", "MSFT"]
  lookback_days: 90
  include_prediction_confidence: true
```
Purpose: Identify concentration risk and natural hedges

#### 3. Monte Carlo Risk Simulation
```
Use: mcp__ai-news-trader__risk_analysis
Parameters:
  portfolio: [
    {"symbol": "AAPL", "shares": 100, "entry_price": 185.0},
    {"symbol": "NVDA", "shares": 50, "entry_price": 450.0},
    {"symbol": "GOOGL", "shares": 75, "entry_price": 140.0}
  ]
  time_horizon: 5        # days
  var_confidence: 0.05   # 95% VaR
  use_monte_carlo: true
  use_gpu: true         # 500x faster
```

#### 4. Optimal Rebalancing
```
Use: mcp__ai-news-trader__portfolio_rebalance
Parameters:
  target_allocations: {
    "AAPL": 0.25,
    "NVDA": 0.20,
    "GOOGL": 0.20,
    "MSFT": 0.15,
    "TSLA": 0.10,
    "CASH": 0.10
  }
  rebalance_threshold: 0.05  # 5% deviation triggers
```

#### 5. Strategy Health Monitoring
```
Use: mcp__ai-news-trader__monitor_strategy_health
Parameters:
  strategy: "momentum_trading_optimized"
```

### Risk Metrics Explained

**Value at Risk (VaR)**
- Definition: Maximum expected loss at confidence level
- Example: $10,000 VaR (95%) = 5% chance of losing >$10,000
- Calculation: Historical, Parametric, or Monte Carlo

**Conditional VaR (CVaR)**
- Definition: Expected loss beyond VaR threshold
- More conservative than VaR
- Better for tail risk assessment

**Sharpe Ratio**
- Risk-adjusted returns
- Formula: (Return - Risk-free rate) / Volatility
- Target: >1.5 for active strategies

**Maximum Drawdown**
- Largest peak-to-trough decline
- Measures downside risk
- Psychology: Can you handle -20%?

**Beta**
- Market correlation
- β > 1: More volatile than market
- β < 1: Less volatile than market
- β < 0: Inverse correlation (hedge)

### Risk Management Rules

1. **Position Sizing**
   - Kelly Criterion for optimal size
   - Never >5% in single position
   - Sector concentration <25%

2. **Stop Losses**
   - Technical: 2-3x ATR
   - Fundamental: -8% from entry
   - Time-based: Review after 30 days

3. **Portfolio Hedging**
   - Maintain negative correlations
   - Consider options for tail risk
   - Keep 10-20% cash reserve

### Advanced Risk Analysis

**Stress Testing Scenarios**
```python
scenarios = [
    "market_crash",      # -20% equity move
    "interest_spike",    # +200bps rates
    "volatility_surge",  # VIX to 40
    "sector_rotation",   # Tech to value
    "black_swan"        # 6-sigma event
]

for scenario in scenarios:
    impact = stress_test(portfolio, scenario)
    print(f"{scenario}: ${impact:,.0f} loss")
```

**Dynamic Risk Adjustment**
```python
# Adjust position sizes based on volatility
if market_volatility > threshold:
    reduce_position_sizes(factor=0.7)
    increase_cash_allocation(target=0.25)
    enable_protective_stops()
```

**Risk Parity Allocation**
- Equal risk contribution from each asset
- Volatility-weighted positions
- Better diversification than equal weight

### Real-Time Risk Monitoring

```bash
# Continuous risk monitoring loop
while market_open:
    # Check portfolio risk metrics
    var = calculate_var(portfolio)
    if var > risk_limit:
        send_alert("VaR breach: ${var}")
    
    # Monitor correlations
    correlations = get_correlations()
    if max(correlations) > 0.9:
        send_alert("High correlation warning")
    
    # Strategy health check
    health = check_strategy_health()
    if health.score < 80:
        reduce_exposure()
    
    sleep(60)  # Check every minute
```

### Risk Dashboard Metrics
- Portfolio VaR: Real-time value at risk
- Correlation Heatmap: Asset relationships
- Drawdown Chart: Historical losses
- Risk Attribution: By position/sector
- Stress Test Results: Scenario impacts

EOF

echo ""
echo "📊 Typical Risk Metrics for Balanced Portfolio:"
echo "- VaR (95%): 2-3% of portfolio value"
echo "- Sharpe Ratio: 1.5-2.0"
echo "- Max Drawdown: 10-15%"
echo "- Beta: 0.8-1.2"
echo "- Correlation: <0.7 between positions"
echo ""
echo "⚡ GPU Acceleration Benefits:"
echo "- Monte Carlo (10k scenarios): 0.2s vs 100s"
echo "- Correlation matrix: Instant vs 10s"
echo "- Optimization: 1s vs 2 minutes"
echo ""
echo "📚 Full guide: /workspaces/ai-news-trader/demo/guides/risk_management_demo.md"