# Claude Code Demo: News Sentiment Analysis

Master AI-powered news analysis for trading signals using MCP tools.

## Getting Started with News Collection

### Start News Monitoring
```
Start collecting news for my watchlist:
Use mcp__ai-news-trader__control_news_collection with:
- action: "start"
- symbols: ["AAPL", "NVDA", "TSLA", "GOOGL", "MSFT"]
- sources: ["alpha_vantage", "newsapi", "finnhub"]
- update_frequency: 300
- lookback_hours: 48

Confirm the collection has started and show initial statistics.
```

### Check News Provider Status
```
Check the health of all news providers:
Use mcp__ai-news-trader__get_news_provider_status

Show which providers are active, rate limits, and any errors.
```

## Sentiment Analysis

### Basic Sentiment Analysis
```
Analyze Tesla's news sentiment:
Use mcp__ai-news-trader__analyze_news with:
- symbol: "TSLA"
- lookback_hours: 24
- sentiment_model: "enhanced"
- use_gpu: true

Provide sentiment score, key articles, and trading implications.
```

### Multi-Symbol Sentiment Comparison
```
Compare news sentiment across tech giants:
For each symbol in [AAPL, GOOGL, MSFT, META, AMZN]:
Use mcp__ai-news-trader__analyze_news with:
- lookback_hours: 48
- sentiment_model: "enhanced"
- use_gpu: true

Create a sentiment ranking from most bullish to most bearish.
```

## Sentiment Trends

### Trend Analysis
```
Analyze sentiment trends for my portfolio:
Use mcp__ai-news-trader__get_news_trends with:
- symbols: ["AAPL", "TSLA", "NVDA"]
- time_intervals: [1, 6, 24, 48]

Show momentum changes and identify improving/deteriorating sentiment.
```

### Intraday Sentiment Monitoring
```
Track intraday sentiment changes:
Use mcp__ai-news-trader__get_news_trends with:
- symbols: ["SPY", "QQQ"]
- time_intervals: [1, 2, 4, 8]

Alert me to any sudden sentiment shifts in the last hour.
```

## News Filtering

### High-Impact Positive News
```
Find bullish opportunities:
Use mcp__ai-news-trader__fetch_filtered_news with:
- symbols: ["AAPL", "NVDA", "TSLA", "GOOGL", "MSFT"]
- sentiment_filter: "positive"
- relevance_threshold: 0.8
- limit: 20

Show only high-confidence positive news that could drive prices up.
```

### Breaking News Alert
```
Get the most recent high-relevance news:
Use mcp__ai-news-trader__fetch_filtered_news with:
- symbols: ["AAPL", "NVDA", "TSLA", "GOOGL", "MSFT"]
- sentiment_filter: null (all sentiments)
- relevance_threshold: 0.9
- limit: 10

Focus on news from the last 2 hours with highest market impact.
```

## Advanced Sentiment Strategies

### Earnings Season Analysis
```
Analyze earnings-related sentiment:
1. Start collection focusing on earnings keywords
2. For each symbol with earnings this week:
   - Analyze sentiment trend over past 5 days
   - Compare to historical earnings sentiment
   - Identify unusual sentiment patterns

Use appropriate mcp__ai-news-trader__ tools and provide trading recommendations.
```

### Sector Sentiment Rotation
```
Identify sector rotation through sentiment:
1. Analyze sentiment for sector leaders:
   - Tech: AAPL, MSFT, GOOGL
   - Finance: JPM, BAC, GS
   - Energy: XOM, CVX, COP
   
2. Compare sector sentiment trends over [1, 7, 30] days
3. Identify sectors with improving sentiment momentum

Recommend sector allocation based on sentiment shifts.
```

### Sentiment Divergence Trading
```
Find price-sentiment divergences:
1. Get quick_analysis for technical indicators
2. Get news sentiment for same symbols
3. Identify where price and sentiment disagree:
   - Positive sentiment + declining price = potential buy
   - Negative sentiment + rising price = potential sell

List the top 3 divergence opportunities.
```

## Real-Time Monitoring

### Live Sentiment Dashboard
```
Create a real-time sentiment monitor:
Every 15 minutes, check:
1. Overall market sentiment (SPY, QQQ, DIA)
2. Top movers by sentiment change
3. Breaking news alerts
4. Sentiment momentum indicators

Set up alerts for sentiment score changes >0.3.
```

### Event-Driven Analysis
```
Monitor for specific events:
1. Product launches
2. Executive changes
3. Regulatory news
4. Partnership announcements

Use filtered news to catch these events early and analyze immediate sentiment impact.
```

## Integration Workflows

### Sentiment + Technical Analysis
```
Complete analysis for AAPL:
1. Use mcp__ai-news-trader__analyze_news for sentiment
2. Use mcp__ai-news-trader__quick_analysis for technicals
3. Weight signals: 40% sentiment, 60% technical
4. Generate composite buy/sell recommendation

Show how sentiment confirms or contradicts technical signals.
```

### Sentiment-Based Position Sizing
```
Adjust position sizes based on sentiment confidence:
- Very Positive (>0.7): 100% position size
- Positive (0.3-0.7): 75% position size
- Neutral (-0.3-0.3): 50% position size
- Negative (<-0.3): 0% or short position

Apply this to current opportunities.
```

## Output Templates

### Sentiment Report Card
```
Symbol: TSLA
Overall Sentiment: +0.65 (Bullish)
Trend: Improving ↑
Key Drivers: 
- Positive: Production numbers beat (0.85)
- Positive: New factory announcement (0.72)
- Negative: Regulatory concerns (-0.45)
Recommendation: BUY with 75% position size
```

### Market Sentiment Heatmap
```
         1hr   6hr   24hr  48hr
AAPL    +0.2  +0.3  +0.4  +0.5  ↑↑
NVDA    +0.7  +0.6  +0.5  +0.3  ↓
TSLA    -0.1  +0.1  +0.2  +0.4  ↑↑
GOOGL   +0.3  +0.3  +0.3  +0.3  →
MSFT    +0.5  +0.4  +0.4  +0.4  →
```

## Best Practices

1. **Update frequency**: 5 minutes for active trading, 15 minutes for position trading
2. **Lookback periods**: 24-48 hours for swing trading, 1-6 hours for day trading
3. **Relevance threshold**: 0.7+ for quality, 0.9+ for high conviction
4. **Sentiment confidence**: Only trade when confidence >80%
5. **Multiple sources**: Require 2+ sources for confirmation

## Troubleshooting

- **No news found**: Check if symbols are correct, market hours, provider status
- **Slow analysis**: Ensure GPU is enabled for enhanced sentiment model
- **Conflicting sentiments**: Weight by source reliability and recency
- **API limits**: Reduce update frequency or symbol count if hitting limits