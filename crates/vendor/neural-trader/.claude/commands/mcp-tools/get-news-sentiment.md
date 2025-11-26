# MCP Tool: get_news_sentiment

## Overview
The `get_news_sentiment` tool provides real-time news sentiment data for trading symbols with source filtering capabilities. It offers a lightweight, fast alternative to `analyze_news` for quick sentiment checks and real-time monitoring.

## Tool Details
- **Full Name**: `mcp__ai-news-trader__get_news_sentiment`
- **Category**: News & Sentiment Analysis
- **GPU Support**: No (optimized for speed)
- **AI Models**: Pre-computed sentiment scores with real-time aggregation

## Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `symbol` | string | Yes | - | Trading symbol to analyze (e.g., "AAPL", "BTC-USD") |
| `sources` | array[string] | No | null | Filter by specific news sources (e.g., ["reuters", "bloomberg"]) |

## Key Differences from analyze_news

| Feature | get_news_sentiment | analyze_news |
|---------|-------------------|--------------|
| Processing Speed | < 100ms | 1-5 seconds |
| GPU Required | No | Optional |
| Sentiment Depth | Pre-computed scores | Deep neural analysis |
| Historical Data | Last 24 hours | Configurable lookback |
| Best Use Case | Real-time monitoring | Detailed analysis |

## Sentiment Data Structure

### Response Format
```json
{
  "symbol": "AAPL",
  "timestamp": "2024-01-15T10:30:00Z",
  "sentiment_score": 0.65,
  "sentiment_label": "bullish",
  "article_count": 42,
  "sources": {
    "reuters": 0.72,
    "bloomberg": 0.58,
    "cnbc": 0.65
  },
  "top_headlines": [
    {
      "headline": "Apple Announces Record Q4 Earnings",
      "sentiment": 0.85,
      "source": "reuters",
      "timestamp": "2024-01-15T09:15:00Z"
    }
  ],
  "sentiment_velocity": 0.12,
  "data_freshness": "2 minutes ago"
}
```

### Sentiment Classifications
- **Strong Bullish**: > 0.7
- **Bullish**: 0.3 to 0.7
- **Neutral**: -0.3 to 0.3
- **Bearish**: -0.7 to -0.3
- **Strong Bearish**: < -0.7

## Integration with Trading Decisions

### High-Frequency Trading Integration
```python
# Real-time sentiment monitoring for HFT
def hft_sentiment_signal(symbol):
    sentiment = get_news_sentiment(symbol)
    
    # Fast decision based on sentiment
    if sentiment['sentiment_score'] > 0.7 and sentiment['sentiment_velocity'] > 0.1:
        return "BUY_SIGNAL"
    elif sentiment['sentiment_score'] < -0.7 and sentiment['sentiment_velocity'] < -0.1:
        return "SELL_SIGNAL"
    else:
        return "NO_ACTION"
```

### News-Driven Alerts
```python
# Set up sentiment alerts
def sentiment_alert_system(watchlist):
    alerts = []
    for symbol in watchlist:
        data = get_news_sentiment(symbol)
        
        # Check for significant sentiment shifts
        if abs(data['sentiment_velocity']) > 0.2:
            alerts.append({
                'symbol': symbol,
                'alert_type': 'SENTIMENT_SHIFT',
                'score': data['sentiment_score'],
                'velocity': data['sentiment_velocity']
            })
    
    return alerts
```

## Multiple Timeframe Examples

### Scalping (Real-time monitoring)
```bash
# Ultra-fast sentiment checks for scalping
while true; do
  mcp call get_news_sentiment '{"symbol": "TSLA"}'
  sleep 30
done
```

### Day Trading (Periodic checks)
```bash
# Check sentiment every 5 minutes during market hours
mcp call get_news_sentiment '{
  "symbol": "AAPL",
  "sources": ["reuters", "bloomberg", "wsj"]
}'
```

### Pre-Market Analysis
```bash
# Check overnight news sentiment before market open
symbols=("AAPL" "GOOGL" "MSFT" "AMZN")
for symbol in "${symbols[@]}"; do
  echo "Checking $symbol..."
  mcp call get_news_sentiment "{\"symbol\": \"$symbol\"}"
done
```

### Multi-Source Validation
```bash
# Compare sentiment across different news sources
mcp call get_news_sentiment '{
  "symbol": "NVDA",
  "sources": ["reuters", "bloomberg", "marketwatch", "seekingalpha"]
}'
```

## Advanced Usage Patterns

### Source Reliability Weighting
```python
# Weight sentiment by source reliability
def weighted_sentiment(symbol):
    # Define source weights based on accuracy
    source_weights = {
        'reuters': 0.9,
        'bloomberg': 0.85,
        'wsj': 0.8,
        'marketwatch': 0.7,
        'seekingalpha': 0.6
    }
    
    result = get_news_sentiment(symbol)
    
    # Calculate weighted sentiment
    weighted_score = 0
    total_weight = 0
    
    for source, score in result['sources'].items():
        if source in source_weights:
            weight = source_weights[source]
            weighted_score += score * weight
            total_weight += weight
    
    return weighted_score / total_weight if total_weight > 0 else result['sentiment_score']
```

### Sentiment Divergence Detection
```python
# Detect when sources disagree on sentiment
def detect_sentiment_divergence(symbol):
    result = get_news_sentiment(symbol)
    
    if not result['sources']:
        return None
    
    scores = list(result['sources'].values())
    
    # Calculate standard deviation
    mean = sum(scores) / len(scores)
    variance = sum((x - mean) ** 2 for x in scores) / len(scores)
    std_dev = variance ** 0.5
    
    # High divergence indicates uncertainty
    if std_dev > 0.3:
        return {
            'divergence': 'high',
            'std_dev': std_dev,
            'recommendation': 'avoid_trading'
        }
    
    return {
        'divergence': 'low',
        'std_dev': std_dev,
        'recommendation': 'normal_trading'
    }
```

### Rapid Multi-Symbol Screening
```python
# Fast sentiment screening for 100+ symbols
import asyncio

async def rapid_sentiment_screen(symbols):
    results = {}
    
    # Process in batches for efficiency
    batch_size = 10
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i+batch_size]
        
        # Parallel processing
        tasks = [get_news_sentiment(symbol) for symbol in batch]
        batch_results = await asyncio.gather(*tasks)
        
        for symbol, result in zip(batch, batch_results):
            results[symbol] = {
                'score': result['sentiment_score'],
                'velocity': result['sentiment_velocity'],
                'articles': result['article_count']
            }
    
    # Return top movers
    return sorted(
        results.items(),
        key=lambda x: abs(x[1]['velocity']),
        reverse=True
    )[:10]
```

## Best Practices

### Performance Optimization
1. **Use for real-time monitoring** - Designed for <100ms response
2. **Avoid excessive polling** - Cache results for at least 30 seconds
3. **Batch symbol requests** when screening multiple stocks

### Source Selection Strategy
- **Major moves**: Focus on ["reuters", "bloomberg"]
- **Crypto**: Include ["coindesk", "cointelegraph"]
- **Retail sentiment**: Add ["reddit", "stocktwits"]
- **Professional**: Stick to ["wsj", "ft", "bloomberg"]

### Integration Guidelines
1. **Combine with `analyze_news`** for confirmation on significant moves
2. **Use sentiment velocity** as an early warning system
3. **Monitor source disagreement** as a volatility indicator
4. **Set thresholds** based on historical accuracy for each source

## Common Use Cases

### Pre-Market Gaps
```python
# Predict opening gaps based on overnight sentiment
def predict_gap(symbol):
    sentiment = get_news_sentiment(symbol)
    
    if sentiment['sentiment_score'] > 0.6:
        gap_prediction = "UP"
        confidence = min(sentiment['sentiment_score'], 0.9)
    elif sentiment['sentiment_score'] < -0.6:
        gap_prediction = "DOWN"
        confidence = min(abs(sentiment['sentiment_score']), 0.9)
    else:
        gap_prediction = "FLAT"
        confidence = 0.5
    
    return {
        'prediction': gap_prediction,
        'confidence': confidence,
        'articles': sentiment['article_count']
    }
```

### News Catalyst Detection
```python
# Identify potential catalysts from headlines
def detect_catalyst(symbol):
    data = get_news_sentiment(symbol)
    
    catalyst_keywords = [
        'earnings', 'merger', 'acquisition', 'fda',
        'lawsuit', 'guidance', 'upgrade', 'downgrade'
    ]
    
    catalysts = []
    for headline in data['top_headlines']:
        for keyword in catalyst_keywords:
            if keyword in headline['headline'].lower():
                catalysts.append({
                    'type': keyword,
                    'headline': headline['headline'],
                    'sentiment': headline['sentiment']
                })
    
    return catalysts
```

## Error Handling

### No Data Available
```python
# Handle cases with no recent news
result = get_news_sentiment("OBSCURE_SYMBOL")
if result['article_count'] == 0:
    # Fall back to sector sentiment or skip
    pass
```

### Source Filtering Issues
```python
# Handle invalid source names
try:
    result = get_news_sentiment(
        symbol="AAPL",
        sources=["invalid_source"]
    )
except ValueError as e:
    # Use default sources
    result = get_news_sentiment(symbol="AAPL")
```

## Related Tools
- [`analyze_news`](analyze-news.md): Deep sentiment analysis with AI
- [`quick_analysis`](quick-analysis.md): Combined technical and sentiment
- [`neural_forecast`](../neural-forecast.md): Price predictions
- [`execute_trade`](execute-trade.md): Act on sentiment signals