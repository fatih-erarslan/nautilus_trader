# Quick Start Guide - AI Trading with Claude Code

Get trading in 5 minutes with these essential MCP tool commands.

## 🏃 Instant Trading Analysis

### The One-Command Trading Decision
```
Analyze NVDA for trading opportunity:
1. Use mcp__ai-news-trader__quick_analysis with symbol: "NVDA", use_gpu: true
2. Use mcp__ai-news-trader__analyze_news with symbol: "NVDA", lookback_hours: 24, sentiment_model: "enhanced", use_gpu: true  
3. Use mcp__ai-news-trader__neural_forecast with symbol: "NVDA", horizon: 7, use_gpu: true

Give me a clear BUY/SELL/HOLD signal with confidence percentage.
```

## 🎯 5 Essential Commands

### 1. Check Any Stock
```
What's happening with Apple stock?
Use mcp__ai-news-trader__quick_analysis with:
- symbol: "AAPL"
- use_gpu: true
```

### 2. Get AI Price Prediction
```
Where will Tesla be in a week?
Use mcp__ai-news-trader__neural_forecast with:
- symbol: "TSLA"
- horizon: 7
- confidence_level: 0.95
- use_gpu: true
```

### 3. Check Market Sentiment
```
What's the news saying about Microsoft?
Use mcp__ai-news-trader__analyze_news with:
- symbol: "MSFT"
- lookback_hours: 48
- sentiment_model: "enhanced"
- use_gpu: true
```

### 4. Compare Trading Strategies
```
Which strategy is best right now?
Use mcp__ai-news-trader__get_strategy_comparison with:
- strategies: ["momentum_trading_optimized", "swing_trading_optimized", "mean_reversion_optimized"]
- metrics: ["sharpe_ratio", "total_return", "max_drawdown", "win_rate"]
```

### 5. Check Portfolio Risk
```
How risky is my portfolio?
Use mcp__ai-news-trader__get_portfolio_status with:
- include_analytics: true
```

## 🔥 Power User Combos

### The Day Trader Special
```
I'm day trading tech stocks today. For AAPL, NVDA, and TSLA:
1. Quick analysis on each
2. Check news sentiment 
3. Show 1-day forecasts
4. Highlight the best opportunity
5. Calculate position size for $10,000 account with 2% risk

Use appropriate mcp__ai-news-trader__ tools with use_gpu: true.
```

### The Swing Trader Setup
```
Find me the best swing trade for this week:
1. Analyze top 10 S&P 500 stocks by momentum
2. Check sentiment trends over 7 days
3. Generate 7-day forecasts
4. Backtest swing strategy on top 3
5. Recommend best entry with stop loss and target

Focus on risk/reward ratio above 2:1.
```

### The Risk Manager's Dashboard
```
Give me a complete risk report:
1. Portfolio correlation matrix
2. Value at Risk (95% confidence)
3. Stress test -20% market crash
4. Rebalancing recommendations
5. Strategy health check

Alert me to any critical issues.
```

## 📱 Mobile-Friendly Commands

### Quick Check
```
AAPL quick check - price, trend, signal
```
Claude knows to use: `mcp__ai-news-trader__quick_analysis`

### Fast Forecast
```
TSLA 7 day prediction with confidence
```
Claude knows to use: `mcp__ai-news-trader__neural_forecast`

### Sentiment Pulse
```
NVDA news sentiment now
```
Claude knows to use: `mcp__ai-news-trader__analyze_news`

## 🎮 Interactive Workflows

### Build Your Watchlist
```
Help me build a watchlist:
1. Start with tech giants: AAPL, MSFT, GOOGL, NVDA, META
2. Add top performers from last week
3. Include stocks with positive sentiment
4. Remove anything with high correlation (>0.8)
5. Final list should have 8-10 stocks

Analyze each and rank by opportunity score.
```

### Portfolio Builder
```
I have $50,000 to invest. Build me a portfolio:
1. Risk tolerance: Moderate
2. Time horizon: 6 months
3. Prefer tech and healthcare
4. Maximum 8 positions
5. Include 10-20% cash

Use risk analysis to optimize allocations.
```

## ⚡ Speed Commands

Copy these for instant results:

**Best Stock Right Now**
```
Which stock has the best setup today? Check momentum leaders, analyze sentiment, forecast returns, pick the winner.
```

**Risk Check**
```
Is my portfolio too risky? Run full risk analysis and give me one number: safe/caution/danger.
```

**Strategy Pick**
```
Which strategy for volatile markets? Compare all strategies for high volatility performance.
```

**News Flash**
```
Any breaking news affecting my stocks? Check AAPL, NVDA, TSLA for urgent news.
```

**Exit Signal**
```
Should I sell NVDA? Check technicals, sentiment, and forecast - give clear YES/NO.
```

## 🏆 Pro Templates

### The Opening Bell Ritual
```
It's 9:30 AM. Run my opening bell checklist:
1. Pre-market movers analysis
2. Overnight news sentiment
3. Update all forecasts
4. Risk check on open positions
5. Top 3 trades for today

Execute all checks in parallel for speed.
```

### The Closing Bell Review
```
Market just closed. Daily review time:
1. Performance report for today
2. Winners and losers analysis
3. Tomorrow's forecast
4. Strategy health check
5. Any positions to adjust?

Include lessons learned.
```

## 🚨 Emergency Commands

### Crash Protocol
```
EMERGENCY: Market crashing! 
1. Calculate current portfolio loss
2. Run crash stress test
3. Show defensive positions
4. Generate hedge recommendations
5. Safe haven analysis (gold, bonds, dollar)

Need answers in 30 seconds!
```

### Recovery Mode
```
Big loss today. Recovery plan needed:
1. Analyze what went wrong
2. Check if strategy is broken
3. Position size recalculation
4. Find mean reversion opportunities
5. Set strict risk limits

Focus on capital preservation.
```

## 🎯 Your First Trade

Ready to make your first AI-powered trade? Copy this:

```
I want to make my first AI trade with $1,000. Find me:
1. The safest stock with positive momentum
2. Clear entry and exit points
3. Position size with max $50 risk
4. Expected return in 7 days
5. Simple monitoring plan

Keep it simple and safe!
```

---

**Remember**: All trades are simulated unless you connect a real broker. Perfect for learning!

**Need Help?** Just ask: "How do I analyze stocks with Claude?" or "Show me trading examples"