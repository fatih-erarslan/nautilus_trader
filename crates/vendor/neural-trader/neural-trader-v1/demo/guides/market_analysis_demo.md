# Market Analysis Agent Demo

## Quick Market Analysis for Multiple Symbols

### Step 1: Analyze AAPL
```
Use tool: mcp__ai-news-trader__quick_analysis
Parameters:
  symbol: "AAPL"
  use_gpu: true
```

### Step 2: Analyze NVDA
```
Use tool: mcp__ai-news-trader__quick_analysis
Parameters:
  symbol: "NVDA"
  use_gpu: true
```

### Step 3: Generate Neural Forecast
```
Use tool: mcp__ai-news-trader__neural_forecast
Parameters:
  symbol: "NVDA"
  horizon: 7
  confidence_level: 0.95
  use_gpu: true
```

### Step 4: Check System Performance
```
Use tool: mcp__ai-news-trader__get_system_metrics
Parameters:
  metrics: ["cpu", "memory", "latency", "throughput"]
  include_history: true
  time_range_minutes: 60
```

## Expected Results:
- Current prices, trends, and technical indicators
- 7-day price predictions with confidence bands
- Buy/sell recommendations based on indicators
- System resource utilization and performance metrics
