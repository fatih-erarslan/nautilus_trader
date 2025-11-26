# MCP Tool: analyze_news

## Overview
The `analyze_news` tool provides AI-powered sentiment analysis of market news for any trading symbol. It uses advanced neural models to process news articles and extract trading-relevant sentiment signals with GPU acceleration support.

## Tool Details
- **Full Name**: `mcp__ai-news-trader__analyze_news`
- **Category**: News & Sentiment Analysis
- **GPU Support**: Yes (optional)
- **AI Models**: Enhanced sentiment analysis with neural language models

## Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `symbol` | string | Yes | - | Trading symbol to analyze (e.g., "AAPL", "BTC-USD") |
| `lookback_hours` | integer | No | 24 | Number of hours to look back for news articles |
| `sentiment_model` | string | No | "enhanced" | Sentiment analysis model ("enhanced", "basic", "financial") |
| `use_gpu` | boolean | No | false | Enable GPU acceleration for faster processing |

## Sentiment Scoring

The tool returns comprehensive sentiment metrics:

### Sentiment Scale
- **Bullish**: +0.5 to +1.0 (Strong positive sentiment)
- **Moderately Bullish**: +0.2 to +0.5 (Positive sentiment)
- **Neutral**: -0.2 to +0.2 (Mixed or neutral sentiment)
- **Moderately Bearish**: -0.5 to -0.2 (Negative sentiment)
- **Bearish**: -1.0 to -0.5 (Strong negative sentiment)

### Metrics Returned
- `overall_sentiment`: Weighted average sentiment score
- `sentiment_distribution`: Percentage breakdown by sentiment category
- `key_topics`: Most discussed topics in the news
- `sentiment_trend`: How sentiment has changed over the lookback period
- `confidence_score`: Model's confidence in the analysis (0-1)
- `article_count`: Number of articles analyzed
- `processing_time`: Time taken for analysis (useful for benchmarking)

## Integration with Trading Decisions

### Signal Generation
```python
# Example trading signal based on sentiment
sentiment_result = analyze_news("AAPL", lookback_hours=12, use_gpu=True)

if sentiment_result['overall_sentiment'] > 0.5:
    signal = "STRONG_BUY"
elif sentiment_result['overall_sentiment'] > 0.2:
    signal = "BUY"
elif sentiment_result['overall_sentiment'] < -0.5:
    signal = "STRONG_SELL"
elif sentiment_result['overall_sentiment'] < -0.2:
    signal = "SELL"
else:
    signal = "HOLD"
```

### Risk Adjustment
- High confidence scores (>0.8) can increase position sizes
- Low confidence scores (<0.5) suggest reduced positions
- Extreme sentiments (>0.7 or <-0.7) may indicate overreaction opportunities

## Multiple Timeframe Examples

### Intraday Trading (4-hour lookback)
```bash
# Quick sentiment check for day trading
mcp call analyze_news '{
  "symbol": "TSLA",
  "lookback_hours": 4,
  "sentiment_model": "enhanced",
  "use_gpu": true
}'
```

### Swing Trading (24-hour lookback)
```bash
# Daily sentiment analysis for swing positions
mcp call analyze_news '{
  "symbol": "AAPL",
  "lookback_hours": 24,
  "sentiment_model": "financial",
  "use_gpu": true
}'
```

### Position Trading (72-hour lookback)
```bash
# Multi-day sentiment trend analysis
mcp call analyze_news '{
  "symbol": "MSFT",
  "lookback_hours": 72,
  "sentiment_model": "enhanced",
  "use_gpu": true
}'
```

### Crypto 24/7 Analysis (12-hour intervals)
```bash
# Cryptocurrency sentiment monitoring
mcp call analyze_news '{
  "symbol": "BTC-USD",
  "lookback_hours": 12,
  "sentiment_model": "enhanced",
  "use_gpu": true
}'
```

## Advanced Usage Patterns

### Combined with Neural Forecasting
```python
# Combine sentiment with price predictions
sentiment = analyze_news("AAPL", lookback_hours=24)
forecast = neural_forecast("AAPL", horizon=24)

# Weight predictions by sentiment
if sentiment['overall_sentiment'] > 0.3:
    adjusted_forecast = forecast['prediction'] * 1.1  # Bullish bias
elif sentiment['overall_sentiment'] < -0.3:
    adjusted_forecast = forecast['prediction'] * 0.9  # Bearish bias
```

### Multi-Symbol Sentiment Screening
```python
# Screen multiple stocks for sentiment opportunities
symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
sentiment_scores = {}

for symbol in symbols:
    result = analyze_news(symbol, lookback_hours=24, use_gpu=True)
    sentiment_scores[symbol] = {
        'score': result['overall_sentiment'],
        'confidence': result['confidence_score']
    }

# Find most bullish stocks
bullish_stocks = sorted(
    sentiment_scores.items(), 
    key=lambda x: x[1]['score'], 
    reverse=True
)[:3]
```

### Event-Driven Trading
```python
# Monitor sentiment changes for events
def monitor_sentiment_shift(symbol, baseline_hours=72, current_hours=4):
    # Get baseline sentiment
    baseline = analyze_news(symbol, lookback_hours=baseline_hours)
    
    # Get recent sentiment
    current = analyze_news(symbol, lookback_hours=current_hours)
    
    # Calculate sentiment shift
    sentiment_shift = current['overall_sentiment'] - baseline['overall_sentiment']
    
    if abs(sentiment_shift) > 0.4:
        return {
            'alert': True,
            'direction': 'bullish' if sentiment_shift > 0 else 'bearish',
            'magnitude': abs(sentiment_shift)
        }
```

## Best Practices

### Performance Optimization
1. **Use GPU acceleration** for real-time analysis of multiple symbols
2. **Cache results** for frequently accessed symbols (results valid for 5-15 minutes)
3. **Batch analysis** when screening multiple symbols

### Sentiment Model Selection
- **"enhanced"**: Best for general market analysis with deep learning
- **"financial"**: Optimized for financial news with domain-specific training
- **"basic"**: Faster processing for high-frequency applications

### Integration Tips
1. **Combine with technical indicators** for confirmation
2. **Use sentiment extremes** as contrarian indicators
3. **Monitor sentiment trends** not just absolute values
4. **Validate with volume** - high sentiment + high volume = stronger signal

## Common Issues and Solutions

### Low Confidence Scores
- **Issue**: Confidence score below 0.5
- **Solution**: Increase lookback period or wait for more news coverage

### Conflicting Sentiments
- **Issue**: Mixed positive/negative articles
- **Solution**: Check sentiment_distribution for detailed breakdown

### Processing Delays
- **Issue**: Slow analysis without GPU
- **Solution**: Enable GPU acceleration or reduce lookback period

## Related Tools
- [`get_news_sentiment`](get-news-sentiment.md): Real-time sentiment data
- [`neural_forecast`](../neural-forecast.md): Price predictions
- [`quick_analysis`](quick-analysis.md): Market analysis with sentiment
- [`simulate_trade`](simulate-trade.md): Test sentiment-based strategies