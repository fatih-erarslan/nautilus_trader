# Phase 2 API Usage Examples

## News Trading Functions

### 1. Analyze News Sentiment

```javascript
const { analyze_news } = require('neural-trader');

// Analyze news for AAPL over last 24 hours
const result = await analyze_news(
  "AAPL",              // symbol
  24,                  // lookback_hours
  "enhanced",          // sentiment_model
  false                // use_gpu
);

console.log(JSON.parse(result));
// {
//   "symbol": "AAPL",
//   "sentiment": {
//     "overall": 0.72,
//     "positive": 0.78,
//     "negative": 0.22,
//     "confidence": 0.85
//   },
//   "articles_analyzed": 42,
//   "processing_time_ms": 245.3
// }
```

### 2. Get Multi-Source Sentiment

```javascript
const { get_news_sentiment } = require('neural-trader');

const sentiment = await get_news_sentiment(
  "TSLA",
  ["newsapi", "alpha_vantage"]
);
// Aggregated sentiment from multiple sources
```

### 3. Fetch Filtered News

```javascript
const { fetch_filtered_news } = require('neural-trader');

const news = await fetch_filtered_news(
  ["AAPL", "MSFT"],   // symbols
  50,                  // limit
  0.7,                 // relevance_threshold
  "positive"           // sentiment_filter
);
// Returns filtered news articles with sentiment
```

### 4. Get Sentiment Trends

```javascript
const { get_news_trends } = require('neural-trader');

const trends = await get_news_trends(
  ["AAPL"],
  [1, 6, 24]  // time intervals in hours
);
// {
//   "1h": {"sentiment": 0.65, "volume": 12},
//   "6h": {"sentiment": 0.58, "volume": 45},
//   "24h": {"sentiment": 0.72, "volume": 124}
// }
```

## Strategy Functions

### 5. Run Backtest

```javascript
const { run_backtest } = require('neural-trader');

const backtest = await run_backtest(
  "momentum",           // strategy
  "AAPL",              // symbol
  "2023-01-01",        // start_date
  "2023-12-31",        // end_date
  true,                // use_gpu
  "sp500",             // benchmark
  true                 // include_costs
);

const result = JSON.parse(backtest);
console.log(result.performance);
// {
//   "total_return": 0.453,
//   "sharpe_ratio": 2.84,
//   "max_drawdown": 0.12,
//   "win_rate": 0.68
// }
```

### 6. Optimize Strategy Parameters

```javascript
const { optimize_strategy } = require('neural-trader');

const optimization = await optimize_strategy(
  "momentum",
  "AAPL",
  JSON.stringify({
    lookback_period: [10, 50],
    threshold: [0.01, 0.05]
  }),
  true,                // use_gpu
  1000,                // max_iterations
  "sharpe_ratio"       // optimization_metric
);
// Returns optimal parameters
```

### 7. Monitor Strategy Health

```javascript
const { monitor_strategy_health } = require('neural-trader');

const health = await monitor_strategy_health("momentum");
// {
//   "health_score": 0.92,
//   "status": "healthy",
//   "degradation_alerts": []
// }
```

## Environment Setup

### Required API Keys

```bash
# NewsAPI (required for news functions)
export NEWS_API_KEY=your_newsapi_key

# Alpha Vantage (optional, for additional news sources)
export ALPHA_VANTAGE_KEY=your_alphavantage_key

# Broker (optional, for live trading)
export BROKER_API_KEY=your_broker_key
export BROKER_API_SECRET=your_broker_secret
```

### Getting API Keys

1. **NewsAPI**: https://newsapi.org (free tier: 100 requests/day)
2. **Alpha Vantage**: https://www.alphavantage.co (free tier: 5 requests/min)

## Error Handling

```javascript
const { analyze_news } = require('neural-trader');

try {
  const result = await analyze_news("AAPL");
  const data = JSON.parse(result);
  
  if (data.status === 'configuration_required') {
    console.log('Setup required:', data.configure);
  } else if (data.status === 'error') {
    console.error('Error:', data.error);
  } else {
    console.log('Success:', data.sentiment);
  }
} catch (error) {
  console.error('Failed:', error);
}
```

## Performance Tips

1. **Use GPU** for neural strategies and large-scale analysis
2. **Cache news** to avoid rate limits
3. **Batch operations** when analyzing multiple symbols
4. **Set appropriate timeframes** to balance accuracy and performance

## Integration Example

```javascript
// Full news-driven trading workflow
const {
  analyze_news,
  recommend_strategy,
  run_backtest,
  execute_trade
} = require('neural-trader');

async function newsTradingWorkflow(symbol) {
  // 1. Analyze sentiment
  const newsData = await analyze_news(symbol, 24);
  const sentiment = JSON.parse(newsData).sentiment.overall;
  
  // 2. Recommend strategy based on conditions
  const marketConditions = JSON.stringify({
    sentiment_score: sentiment,
    volatility: "medium"
  });
  const strategyRec = await recommend_strategy(marketConditions);
  const strategy = JSON.parse(strategyRec).recommended_strategy;
  
  // 3. Backtest strategy
  const backtest = await run_backtest(
    strategy,
    symbol,
    "2023-01-01",
    "2023-12-31"
  );
  const metrics = JSON.parse(backtest).performance;
  
  // 4. Execute if criteria met
  if (metrics.sharpe_ratio > 2.0 && sentiment > 0.6) {
    console.log(`Executing ${strategy} trade on ${symbol}`);
    // await execute_trade(...);
  }
}
```
