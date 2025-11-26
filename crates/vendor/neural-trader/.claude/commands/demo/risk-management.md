# Claude Code Demo: Risk Management

Master portfolio risk analysis and management using MCP tools.

## Portfolio Overview

### Current Portfolio Status
```
Show my complete portfolio status:
Use mcp__ai-news-trader__get_portfolio_status with:
- include_analytics: true

Display positions, P&L, risk metrics, and performance analytics.
```

## Correlation Analysis

### Basic Correlation Matrix
```
Analyze correlations in my portfolio:
Use mcp__ai-news-trader__cross_asset_correlation_matrix with:
- assets: ["AAPL", "NVDA", "TSLA", "GOOGL", "MSFT"]
- lookback_days: 90
- include_prediction_confidence: true

Show correlation matrix and identify concentration risks.
```

### Sector Correlation Analysis
```
Check correlations across sectors:
Use mcp__ai-news-trader__cross_asset_correlation_matrix with:
- assets: ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY"]  # Sector ETFs
- lookback_days: 180
- include_prediction_confidence: true

Find uncorrelated sectors for diversification.
```

### Dynamic Correlation Monitoring
```
Track changing correlations:
Use mcp__ai-news-trader__correlation_analysis with:
- symbols: ["AAPL", "NVDA", "TSLA", "GOOGL", "MSFT"]
- period_days: 30
- use_gpu: true

Compare to 90-day correlations and identify regime changes.
```

## Risk Analysis

### Portfolio VaR Calculation
```
Calculate my portfolio risk:
Use mcp__ai-news-trader__risk_analysis with:
- portfolio: [
    {"symbol": "AAPL", "shares": 100, "entry_price": 185.0},
    {"symbol": "NVDA", "shares": 50, "entry_price": 450.0},
    {"symbol": "GOOGL", "shares": 75, "entry_price": 140.0},
    {"symbol": "MSFT", "shares": 80, "entry_price": 380.0}
  ]
- time_horizon: 5
- var_confidence: 0.05
- use_monte_carlo: true
- use_gpu: true

Show VaR, CVaR, and worst-case scenarios.
```

### Stress Testing
```
Run stress tests on portfolio:
Use mcp__ai-news-trader__risk_analysis with:
- portfolio: [current positions]
- time_horizon: 20
- var_confidence: 0.01  # 99% confidence for extreme events
- use_monte_carlo: true
- use_gpu: true

Simulate market crash scenarios and show impact.
```

### Risk Decomposition
```
Break down risk by position:
1. Calculate individual position VaR
2. Show risk contribution by position
3. Identify positions adding most risk
4. Suggest risk reduction trades

Use risk_analysis for each position and aggregate results.
```

## Portfolio Optimization

### Rebalancing Calculation
```
Calculate optimal rebalancing:
Use mcp__ai-news-trader__portfolio_rebalance with:
- target_allocations: {
    "AAPL": 0.20,
    "NVDA": 0.20,
    "GOOGL": 0.20,
    "MSFT": 0.20,
    "TSLA": 0.10,
    "CASH": 0.10
  }
- current_portfolio: null  # Auto-detect
- rebalance_threshold: 0.05

Show required trades and estimated costs.
```

### Risk Parity Rebalancing
```
Create risk-balanced portfolio:
Use mcp__ai-news-trader__portfolio_rebalance with:
- target_allocations: {
    "AAPL": 0.20,
    "BONDS": 0.30,  # TLT
    "GOLD": 0.15,   # GLD
    "REITS": 0.15,  # VNQ
    "INTL": 0.15,   # VEA
    "CASH": 0.05
  }
- rebalance_threshold: 0.03

Design portfolio where each asset contributes equally to risk.
```

## Strategy Health Monitoring

### Single Strategy Check
```
Monitor my active strategy:
Use mcp__ai-news-trader__monitor_strategy_health with:
- strategy: "momentum_trading_optimized"

Show health score, issues, and recommendations.
```

### Multi-Strategy Monitoring
```
Check all active strategies:
For each strategy in [momentum, swing, mean_reversion]:
Use mcp__ai-news-trader__monitor_strategy_health

Create health dashboard with alerts for any issues.
```

## Advanced Risk Workflows

### Position Sizing Calculator
```
Calculate optimal position sizes:
1. Define max portfolio risk (e.g., 2% per trade)
2. For each potential trade:
   - Calculate position volatility
   - Determine stop loss level
   - Size position based on risk budget

Example for AAPL:
- Account: $100,000
- Max risk: 2% = $2,000
- Stop loss: 3% below entry
- Position size: $2,000 / 0.03 = $66,666 (667 shares)
```

### Hedge Ratio Calculation
```
Find optimal hedges:
1. Identify portfolio beta to SPY
2. Calculate negative correlation assets
3. Determine hedge ratios
4. Monitor hedge effectiveness

Use correlation_analysis and risk_analysis tools.
```

### Risk Budget Allocation
```
Allocate risk budget across strategies:
Total risk budget: 20% annual volatility
- Momentum: 8% volatility allocation
- Swing: 6% volatility allocation  
- Mean reversion: 4% volatility allocation
- Cash buffer: 2% volatility

Size positions to stay within budgets.
```

## Real-Time Risk Monitoring

### Risk Dashboard
```
Create live risk monitoring dashboard:
Every 30 minutes check:
1. Portfolio VaR and changes
2. Correlation shifts
3. Strategy health scores
4. Position concentration
5. Margin usage

Alert if any metric exceeds thresholds.
```

### Drawdown Monitoring
```
Track and limit drawdowns:
- Warning at -5% drawdown
- Reduce exposure at -10%
- Stop trading at -15%
- Review and reset monthly

Use portfolio_status to track continuously.
```

## Integration Examples

### Risk-Adjusted Trading
```
Only take trades with positive risk/reward:
1. Calculate expected return from forecast
2. Calculate downside risk (VaR)
3. Require 2:1 reward/risk minimum
4. Size position based on confidence

Example: 
Expected gain: 5%
Potential loss: 2%
Ratio: 2.5:1 ✅ Proceed with trade
```

### Dynamic Risk Management
```
Adjust risk based on market conditions:
If VIX > 25: Reduce all positions by 50%
If correlations > 0.8: Increase diversification
If drawdown > 5%: Tighten stop losses
If winning streak: Take profits, don't increase risk
```

## Output Templates

### Risk Report Card
```
Portfolio Risk Summary
─────────────────────
Total Value: $125,000
Number of Positions: 8
Cash Allocation: 15%

Risk Metrics:
- VaR (95%, 1-day): $2,500 (2.0%)
- CVaR (95%, 1-day): $3,750 (3.0%)
- Beta to Market: 1.15
- Sharpe Ratio: 1.85
- Max Drawdown: -12.5%

Risk Alerts:
⚠️ High correlation between AAPL-MSFT (0.82)
⚠️ NVDA position exceeds 25% of portfolio
✅ Overall risk within acceptable limits
```

### Rebalancing Plan
```
Current vs Target Allocation:
Symbol | Current | Target | Action
AAPL   | 23.5%  | 20.0%  | Sell 50 shares
NVDA   | 28.2%  | 20.0%  | Sell 35 shares  
GOOGL  | 18.3%  | 20.0%  | Buy 15 shares
MSFT   | 16.5%  | 20.0%  | Buy 20 shares
TSLA   | 8.5%   | 10.0%  | Buy 10 shares
CASH   | 5.0%   | 10.0%  | Add $6,250

Estimated Costs: $125 (0.1%)
Risk Reduction: -18% volatility
```

## Best Practices

1. **Check correlations weekly** - They change over time
2. **Run Monte Carlo with 10,000+ scenarios** for accuracy
3. **Rebalance monthly** unless deviation >10%
4. **Keep 10-20% cash** for opportunities and risk buffer
5. **Monitor strategy health daily** during volatile periods
6. **Use trailing stops** to protect profits
7. **Diversify across strategies** not just assets

## Risk Limits

### Recommended Limits
- Single position: Max 25% of portfolio
- Sector exposure: Max 40%
- Daily VaR: Max 3% of portfolio
- Monthly drawdown: Max 10%
- Correlation between positions: Max 0.7
- Leverage: Max 1.5x (150% gross exposure)

### Emergency Procedures
1. If daily loss > 5%: Stop all trading
2. If drawdown > 15%: Reduce all positions by 50%
3. If margin call risk: Liquidate highest risk positions first
4. If correlation > 0.9: Immediately diversify
5. If strategy health < 70: Switch to cash