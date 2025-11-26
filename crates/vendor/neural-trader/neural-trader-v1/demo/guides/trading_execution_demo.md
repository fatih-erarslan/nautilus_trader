# Trading Execution Demo

## Complete Trading Workflow

### Step 1: Simulate Trade
```
Use tool: mcp__ai-news-trader__simulate_trade
Parameters:
  strategy: "momentum_trading_optimized"
  symbol: "NVDA"
  action: "buy"
  use_gpu: true
```

### Step 2: Get Prediction Markets
```
Use tool: mcp__ai-news-trader__get_prediction_markets_tool
Parameters:
  category: "Crypto"
  sort_by: "volume"
  limit: 5
```

### Step 3: Analyze Market Sentiment
```
Use tool: mcp__ai-news-trader__analyze_market_sentiment_tool
Parameters:
  market_id: "crypto_btc_100k"
  analysis_depth: "gpu_enhanced"
  include_correlations: true
  use_gpu: true
```

### Step 4: Execute Multi-Asset Trade
```
Use tool: mcp__ai-news-trader__execute_multi_asset_trade
Parameters:
  trades: [
    {"symbol": "AAPL", "action": "buy", "quantity": 10},
    {"symbol": "NVDA", "action": "buy", "quantity": 5},
    {"symbol": "GOOGL", "action": "sell", "quantity": 3}
  ]
  strategy: "momentum_trading_optimized"
  risk_limit: 50000
  execute_parallel: true
```

### Step 5: Generate Performance Report
```
Use tool: mcp__ai-news-trader__performance_report
Parameters:
  strategy: "momentum_trading_optimized"
  period_days: 30
  include_benchmark: true
  use_gpu: true
```

## Expected Results:
- Trade simulation with expected P&L
- Top crypto prediction markets by volume
- Advanced market analysis with Kelly Criterion
- Multi-asset execution confirmation
- 30-day performance attribution report
