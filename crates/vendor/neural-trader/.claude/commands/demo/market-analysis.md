# Claude Code Demo: Market Analysis

Learn how to perform real-time market analysis with neural forecasting using MCP tools.

## Basic Market Analysis

### Quick Analysis for Single Symbol
```
Please analyze Apple stock using the following:
Use mcp__ai-news-trader__quick_analysis with parameters:
- symbol: "AAPL"
- use_gpu: true

Explain the current price, trend, technical indicators, and trading recommendation.
```

### Multiple Symbol Analysis
```
Analyze the top tech stocks for me. For each of these symbols: AAPL, NVDA, MSFT, GOOGL, TSLA
Use mcp__ai-news-trader__quick_analysis with:
- symbol: [each symbol]
- use_gpu: true

Create a comparison table showing price, trend, RSI, MACD, and recommendations.
```

## Neural Price Forecasting

### 7-Day Forecast
```
Generate a 7-day price forecast for NVIDIA:
Use mcp__ai-news-trader__neural_forecast with:
- symbol: "NVDA"
- horizon: 7
- confidence_level: 0.95
- use_gpu: true

Show the daily predictions with confidence intervals and overall trend analysis.
```

### Extended 30-Day Forecast
```
I need a month-long forecast for my portfolio. Please forecast these symbols:
1. AAPL - 30 days
2. MSFT - 30 days
3. GOOGL - 30 days

Use mcp__ai-news-trader__neural_forecast for each with:
- horizon: 30
- confidence_level: 0.95
- use_gpu: true

Identify which stock has the most bullish forecast.
```

## Model Performance Check

### Neural Model Status
```
Check the health and performance of all neural forecasting models:
Use mcp__ai-news-trader__neural_model_status with:
- model_id: null (to check all models)

Report on model accuracy, last training date, and any issues.
```

### System Performance Monitoring
```
Monitor the trading system performance:
Use mcp__ai-news-trader__get_system_metrics with:
- metrics: ["cpu", "memory", "latency", "throughput", "gpu_utilization"]
- include_history: true
- time_range_minutes: 60

Create a performance dashboard showing current stats and trends.
```

## Advanced Analysis Workflows

### Pre-Market Analysis
```
Perform a comprehensive pre-market analysis:

1. Quick analysis for SPY, QQQ, DIA (market indices)
2. Neural forecast for top movers from yesterday
3. System health check
4. GPU utilization status

Use the appropriate mcp__ai-news-trader__ tools and summarize market conditions.
```

### Sector Rotation Analysis
```
Analyze sector rotation opportunities:

1. Use quick_analysis for sector ETFs: XLK (Tech), XLF (Financial), XLE (Energy), XLV (Healthcare)
2. Generate 7-day forecasts for the top 2 performing sectors
3. Compare momentum indicators across sectors

Recommend sector allocation based on the analysis.
```

### Volatility Analysis
```
Analyze market volatility:

1. Quick analysis for VIX (volatility index)
2. Check correlation between VIX and major indices
3. Forecast VIX for next 7 days
4. Identify stocks with lowest correlation to VIX

Provide volatility-based trading recommendations.
```

## Integration Examples

### Combine with News Sentiment
```
For TSLA, I need both technical and sentiment analysis:

1. Use mcp__ai-news-trader__quick_analysis for technical indicators
2. Use mcp__ai-news-trader__analyze_news for sentiment (last 48 hours)
3. Use mcp__ai-news-trader__neural_forecast for 7-day prediction

Generate a composite buy/sell signal based on all three factors.
```

### Risk-Adjusted Analysis
```
Analyze AAPL with risk considerations:

1. Quick analysis for current status
2. 30-day neural forecast
3. Calculate potential drawdown scenarios
4. Compare risk/reward to SPY benchmark

Provide risk-adjusted recommendation with position sizing suggestion.
```

## Output Formats

### Trading Signal Format
```
Request: "Give me clear trading signals"

Symbol: AAPL
Signal: BUY
Confidence: 85%
Entry: $189.50
Target: $195.00 (7 days)
Stop Loss: $186.00
Risk/Reward: 1:2.2
```

### Dashboard Format
```
Request: "Create a market dashboard"

| Symbol | Price  | Trend | RSI  | Signal | 7-Day Forecast |
|--------|--------|-------|------|--------|----------------|
| AAPL   | $189.50| ↑ Up  | 58.5 | BUY    | +2.8% ±1.5%   |
| NVDA   | $487.25| ↔ Flat| 51.2 | HOLD   | +0.5% ±2.1%   |
```

## Best Practices

1. **Always use GPU acceleration** for neural forecasts (1000x faster)
2. **Check system metrics** before running intensive analysis
3. **Combine multiple timeframes** (1, 7, 30 days) for better insights
4. **Verify model status** weekly to ensure accuracy
5. **Cross-reference** technical indicators with AI forecasts

## Common Issues

- If GPU is unavailable, processing will be slower but still functional
- Mock data is used when market is closed (for testing)
- Forecasts are probabilistic - always use confidence intervals
- System metrics help identify performance bottlenecks