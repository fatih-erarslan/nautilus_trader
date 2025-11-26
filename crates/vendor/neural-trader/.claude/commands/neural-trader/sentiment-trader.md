# Neural Sentiment Trader Command

Execute sentiment-based trading strategies using multi-source news and social media analysis.

## Command

```javascript
Task(
  "Sentiment Trading",
  "Analyze sentiment for {symbol} across news and social media sources. Generate contrarian signals at extremes and momentum signals on sentiment acceleration.",
  "neural-sentiment-trader"
)
```

## Parameters

- `{symbol}` - Stock ticker for sentiment analysis (e.g., "TSLA", "GME", "AMC")

## Full Workflow

```javascript
// Initialize sentiment trading
mcp__claude-flow__swarm_init({
  topology: "hierarchical",
  maxAgents: 6,
  strategy: "specialized"
})

// Spawn sentiment trader
mcp__claude-flow__agent_spawn({
  type: "analyst",
  name: "neural-sentiment-trader",
  capabilities: ["sentiment_analysis", "news_aggregation", "event_driven"]
})

// Get multi-source sentiment
mcp__ai-news-trader__get_news_sentiment({
  symbol: "TSLA",
  sources: ["reuters", "bloomberg", "wsj", "cnbc"]
})

// Deep sentiment analysis
mcp__ai-news-trader__analyze_news({
  symbol: "TSLA",
  lookback_hours: 24,
  sentiment_model: "enhanced",
  use_gpu: true
})

// Fetch filtered news
mcp__ai-news-trader__fetch_filtered_news({
  symbols: ["TSLA"],
  relevance_threshold: 0.7,
  sentiment_filter: "extreme",  // Only extreme sentiment
  limit: 50
})

// Get news trends
mcp__ai-news-trader__get_news_trends({
  symbols: ["TSLA"],
  time_intervals: [1, 6, 24]  // 1hr, 6hr, 24hr
})

// Neural sentiment prediction
mcp__claude-flow__neural_patterns({
  action: "analyze",
  pattern_type: "sentiment_momentum"
})

// Execute sentiment trade
mcp__ai-news-trader__execute_trade({
  strategy: "sentiment_contrarian",
  symbol: "TSLA",
  action: "buy",  // Contrarian at negative extreme
  quantity: 50,
  order_type: "limit"
})
```

## Configuration

```yaml
sentiment_config:
  sources:
    news: ["reuters", "bloomberg", "wsj", "cnbc"]
    social: ["twitter", "reddit", "stocktwits"]
    market: ["options_flow", "insider_trading"]
  thresholds:
    extreme_bullish: 0.8
    extreme_bearish: -0.8
    momentum_threshold: 0.5
  position_sizing:
    extreme: 0.03
    strong: 0.02
    moderate: 0.01
```

## Sentiment Strategies

### Contrarian Trading
```javascript
Task("Contrarian", "Trade against extreme sentiment in AAPL when sentiment < -0.8 or > 0.8", "neural-sentiment-trader")

// Logic
if (sentiment_score < -0.8) {
  // Extreme bearish = contrarian buy
  execute_buy_order()
} else if (sentiment_score > 0.8) {
  // Extreme bullish = contrarian sell
  execute_sell_order()
}
```

### Momentum Trading
```javascript
Task("Sentiment Momentum", "Follow sentiment acceleration in NVDA when rate of change > 0.5/day", "neural-sentiment-trader")

// Logic
const sentiment_acceleration = (current_sentiment - yesterday_sentiment) / 1
if (sentiment_acceleration > 0.5) {
  // Positive acceleration = momentum buy
  execute_buy_order()
}
```

### Event-Driven Trading
```javascript
Task("Event Trading", "Trade on high-impact events for TSLA including earnings, product launches, regulatory news", "neural-sentiment-trader")

// Event categorization
const event_impact = {
  earnings: "high",
  product_launch: "high",
  analyst_upgrade: "medium",
  executive_change: "low"
}
```

## Multi-Source Aggregation

```javascript
// Aggregate sentiment scores
function aggregate_sentiment(sources) {
  const weights = {
    reuters: 0.25,
    bloomberg: 0.25,
    wsj: 0.20,
    cnbc: 0.15,
    twitter: 0.10,
    reddit: 0.05
  }
  
  let weighted_sum = 0
  let total_weight = 0
  
  for (const [source, score] of Object.entries(sources)) {
    if (weights[source]) {
      weighted_sum += score * weights[source]
      total_weight += weights[source]
    }
  }
  
  return weighted_sum / total_weight
}

// Calculate confidence
function sentiment_confidence(sources) {
  const scores = Object.values(sources)
  const std_dev = standard_deviation(scores)
  
  // Low std dev = high agreement = high confidence
  return 1 - Math.min(std_dev, 1)
}
```

## Sentiment Indicators

```javascript
// Sentiment RSI
function sentiment_rsi(sentiment_history, period = 14) {
  const gains = []
  const losses = []
  
  for (let i = 1; i < sentiment_history.length; i++) {
    const change = sentiment_history[i] - sentiment_history[i-1]
    if (change > 0) {
      gains.push(change)
      losses.push(0)
    } else {
      gains.push(0)
      losses.push(Math.abs(change))
    }
  }
  
  const avg_gain = average(gains.slice(-period))
  const avg_loss = average(losses.slice(-period))
  const rs = avg_gain / avg_loss
  
  return 100 - (100 / (1 + rs))
}

// Sentiment MACD
function sentiment_macd(sentiment_history) {
  const ema12 = exponential_moving_average(sentiment_history, 12)
  const ema26 = exponential_moving_average(sentiment_history, 26)
  const macd_line = ema12 - ema26
  const signal_line = exponential_moving_average([macd_line], 9)
  
  return {
    macd: macd_line,
    signal: signal_line,
    histogram: macd_line - signal_line
  }
}
```

## Event Detection

```javascript
// Monitor for events
async function detect_events(symbol) {
  // Check earnings calendar
  const earnings_date = await get_earnings_date(symbol)
  const days_to_earnings = days_until(earnings_date)
  
  // Check news volume spike
  const news_volume = await get_news_count(symbol, 1)  // Last hour
  const avg_volume = await get_avg_news_count(symbol)
  const volume_spike = news_volume / avg_volume
  
  // Check sentiment shift
  const current_sentiment = await get_current_sentiment(symbol)
  const prev_sentiment = await get_prev_sentiment(symbol)
  const sentiment_shift = Math.abs(current_sentiment - prev_sentiment)
  
  return {
    earnings_approaching: days_to_earnings < 3,
    news_spike: volume_spike > 3,
    sentiment_shift: sentiment_shift > 0.5
  }
}
```

## Risk Management

```javascript
// Position sizing by sentiment confidence
function calculate_position_size(sentiment_score, confidence) {
  const base_size = portfolio_value * 0.02
  
  // Adjust for confidence
  const confidence_multiplier = 0.5 + (confidence * 0.5)  // 0.5x to 1x
  
  // Adjust for extremity
  const extremity = Math.abs(sentiment_score)
  const extremity_multiplier = 0.5 + (extremity * 0.5)  // 0.5x to 1x
  
  return base_size * confidence_multiplier * extremity_multiplier
}

// Volatility adjustment
function adjust_for_volatility(position_size, symbol) {
  const implied_volatility = get_implied_volatility(symbol)
  const vix = get_vix_level()
  
  if (implied_volatility > 50 || vix > 30) {
    return position_size * 0.5  // Reduce by half in high volatility
  }
  
  return position_size
}
```

## Performance Monitoring

```javascript
// Track sentiment accuracy
const sentiment_trades = []

function track_sentiment_trade(trade) {
  sentiment_trades.push({
    symbol: trade.symbol,
    sentiment_score: trade.sentiment_score,
    action: trade.action,
    entry_price: trade.price,
    timestamp: Date.now()
  })
}

function analyze_sentiment_performance() {
  const results = sentiment_trades.map(trade => {
    const exit_price = get_exit_price(trade)
    const return_pct = (exit_price - trade.entry_price) / trade.entry_price
    
    return {
      ...trade,
      return_pct,
      correct_direction: 
        (trade.action === "buy" && return_pct > 0) ||
        (trade.action === "sell" && return_pct < 0)
    }
  })
  
  return {
    accuracy: results.filter(r => r.correct_direction).length / results.length,
    avg_return: average(results.map(r => r.return_pct)),
    sharpe_ratio: calculate_sharpe(results.map(r => r.return_pct))
  }
}
```

## Success Metrics

- Signal Accuracy: Target > 60%
- Sharpe Ratio: Target > 1.8
- Average Holding: 1-5 days
- Win Rate: Target > 55%
- Sentiment Correlation: > 0.6

## Tips

1. **Source Credibility**: Weight established sources higher
2. **Confirmation Required**: Don't trade on single source
3. **Volume Matters**: High news volume = higher confidence
4. **Fade Retail**: Contrarian against retail sentiment extremes
5. **Event Awareness**: Reduce size before binary events