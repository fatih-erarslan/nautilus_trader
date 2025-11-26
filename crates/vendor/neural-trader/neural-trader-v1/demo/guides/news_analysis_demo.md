# News Sentiment Analysis Demo

## Comprehensive News Analysis Workflow

### Step 1: Start News Collection
```
Use tool: mcp__ai-news-trader__control_news_collection
Parameters:
  action: "start"
  symbols: ["AAPL", "NVDA", "TSLA", "GOOGL", "MSFT"]
  update_frequency: 300
  lookback_hours: 48
```

### Step 2: Analyze TSLA News Sentiment
```
Use tool: mcp__ai-news-trader__analyze_news
Parameters:
  symbol: "TSLA"
  lookback_hours: 48
  sentiment_model: "enhanced"
  use_gpu: true
```

### Step 3: Get Sentiment Trends
```
Use tool: mcp__ai-news-trader__get_news_trends
Parameters:
  symbols: ["AAPL", "TSLA", "NVDA"]
  time_intervals: [1, 6, 24, 48]
```

### Step 4: Filter High-Impact News
```
Use tool: mcp__ai-news-trader__fetch_filtered_news
Parameters:
  symbols: ["AAPL", "NVDA", "TSLA", "GOOGL", "MSFT"]
  sentiment_filter: "positive"
  relevance_threshold: 0.8
  limit: 10
```

## Expected Results:
- Multi-source news aggregation activated
- Sentiment scores from -1 (bearish) to +1 (bullish)
- Trend analysis showing momentum changes
- Filtered list of high-impact positive news
