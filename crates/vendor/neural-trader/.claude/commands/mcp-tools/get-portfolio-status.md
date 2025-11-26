# MCP Tool: get_portfolio_status

## Overview
Get current portfolio status with optional advanced analytics. This tool provides comprehensive insights into your trading portfolio including positions, performance metrics, and risk analytics.

## Tool Details
- **Name**: `mcp__ai-news-trader__get_portfolio_status`
- **Category**: Trading Strategy Tools
- **GPU Support**: Analytics calculations can leverage GPU acceleration

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `include_analytics` | boolean | `true` | Include advanced analytics calculations in the response |

## Return Value Structure
The tool returns a comprehensive portfolio status object containing:

```json
{
  "portfolio": {
    "total_value": 125432.50,
    "cash_balance": 25432.50,
    "positions": [
      {
        "symbol": "AAPL",
        "quantity": 100,
        "current_price": 195.50,
        "cost_basis": 180.25,
        "current_value": 19550.00,
        "unrealized_pnl": 1525.00,
        "pnl_percentage": 8.46
      }
    ],
    "performance": {
      "daily_return": 0.023,
      "weekly_return": 0.056,
      "monthly_return": 0.124,
      "ytd_return": 0.285
    }
  },
  "analytics": {
    "sharpe_ratio": 1.85,
    "sortino_ratio": 2.15,
    "max_drawdown": -0.125,
    "volatility": 0.186,
    "beta": 1.12,
    "alpha": 0.045,
    "risk_metrics": {
      "var_95": -2340.50,
      "cvar_95": -3125.75,
      "kelly_criterion": 0.22
    }
  },
  "timestamp": "2024-12-27T10:30:00Z"
}
```

## Examples

### Example 1: Basic Portfolio Status
```bash
# Get portfolio status with full analytics
claude --mcp ai-news-trader "Get my current portfolio status"

# The tool will be called as:
mcp__ai-news-trader__get_portfolio_status({
  "include_analytics": true
})
```

### Example 2: Quick Portfolio Check (No Analytics)
```bash
# Get portfolio positions without advanced analytics
claude --mcp ai-news-trader "Show my portfolio positions without analytics"

# The tool will be called as:
mcp__ai-news-trader__get_portfolio_status({
  "include_analytics": false
})
```

### Example 3: Risk Assessment Focus
```bash
# Check portfolio with focus on risk metrics
claude --mcp ai-news-trader "Analyze my portfolio risk exposure"

# The tool will be called as:
mcp__ai-news-trader__get_portfolio_status({
  "include_analytics": true
})
# Claude will then focus on the risk_metrics section
```

### Example 4: Performance Review
```bash
# Review portfolio performance metrics
claude --mcp ai-news-trader "How has my portfolio performed this month?"

# The tool will be called as:
mcp__ai-news-trader__get_portfolio_status({
  "include_analytics": true
})
# Claude will analyze the performance section
```

### Example 5: Integration with Trading Decisions
```bash
# Use portfolio status to inform trading decisions
claude --mcp ai-news-trader "Check my portfolio and suggest rebalancing opportunities"

# The tool will be called as:
mcp__ai-news-trader__get_portfolio_status({
  "include_analytics": true
})
# Claude will analyze positions and suggest rebalancing based on analytics
```

## GPU Acceleration Notes
- When `include_analytics` is `true`, GPU acceleration can speed up:
  - Sharpe ratio calculations by ~50x
  - VaR/CVaR computations by ~100x
  - Correlation matrix calculations by ~200x
- GPU is automatically used when available
- Fallback to CPU ensures compatibility

## Performance Benchmarks
| Operation | CPU Time | GPU Time | Speedup |
|-----------|----------|----------|---------|
| Basic Portfolio Status | 5ms | 5ms | 1x |
| Full Analytics (10 assets) | 150ms | 8ms | 18.75x |
| Full Analytics (100 assets) | 2500ms | 25ms | 100x |
| Risk Metrics Calculation | 500ms | 10ms | 50x |

## Best Practices
1. **Regular Monitoring**: Check portfolio status at market open and close
2. **Analytics Usage**: Enable analytics for end-of-day reviews, disable for quick checks
3. **Risk Management**: Monitor VaR and drawdown metrics daily
4. **Performance Tracking**: Compare returns against benchmarks regularly
5. **Integration**: Combine with neural forecasts for forward-looking analysis

## Related Tools
- `simulate_trade`: Test trades before execution
- `execute_trade`: Execute portfolio changes
- `risk_analysis`: Deep dive into portfolio risk
- `performance_report`: Detailed performance analytics
- `correlation_analysis`: Understand asset relationships