# Risk Management Demo

## Portfolio Risk Analysis Suite

### Step 1: Get Portfolio Status
```
Use tool: mcp__ai-news-trader__get_portfolio_status
Parameters:
  include_analytics: true
```

### Step 2: Generate Correlation Matrix
```
Use tool: mcp__ai-news-trader__cross_asset_correlation_matrix
Parameters:
  assets: ["AAPL", "NVDA", "TSLA", "GOOGL", "MSFT"]
  lookback_days: 90
  include_prediction_confidence: true
```

### Step 3: Run Monte Carlo Risk Analysis
```
Use tool: mcp__ai-news-trader__risk_analysis
Parameters:
  portfolio: [
    {"symbol": "AAPL", "shares": 100, "entry_price": 185.0},
    {"symbol": "NVDA", "shares": 50, "entry_price": 450.0},
    {"symbol": "GOOGL", "shares": 75, "entry_price": 140.0}
  ]
  time_horizon: 5
  var_confidence: 0.05
  use_monte_carlo: true
  use_gpu: true
```

### Step 4: Calculate Rebalancing
```
Use tool: mcp__ai-news-trader__portfolio_rebalance
Parameters:
  target_allocations: {
    "AAPL": 0.25,
    "NVDA": 0.25,
    "GOOGL": 0.20,
    "MSFT": 0.20,
    "CASH": 0.10
  }
  rebalance_threshold: 0.05
```

## Expected Results:
- Current portfolio value and P&L
- 5x5 correlation matrix with ML confidence
- VaR and CVaR from 10,000 simulations
- Specific rebalancing trades needed
