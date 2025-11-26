# MCP Trading Automation Examples

## Overview
These examples show how to automate trading workflows using Claude Code with MCP tools. Each automation can be triggered by schedule, events, or conditions.

## 1. Daily Trading Automation

### Morning Market Scan
**Request to Claude Code:**
```
"Every morning at 9:00 AM, run my market scan:
1. Check portfolio status
2. Analyze overnight news for my holdings
3. Generate neural forecasts for top movers
4. Identify trading opportunities
5. Send me a summary"
```

**MCP Tool Chain:**
```python
# 1. Portfolio check
ai-news-trader:get_portfolio_status (MCP)(include_analytics: true)

# 2. News analysis for each holding
for symbol in portfolio_holdings:
    ai-news-trader:analyze_news (MCP)(symbol: symbol, lookback_hours: 12)

# 3. Neural forecasts for volatile stocks
for symbol in high_volatility_stocks:
    ai-news-trader:neural_forecast (MCP)(symbol: symbol, horizon: 8)

# 4. Technical analysis
for opportunity in opportunities:
    ai-news-trader:quick_analysis (MCP)(symbol: opportunity)
```

### Automated Position Sizing
```
"Calculate position sizes based on:
- Current volatility
- Portfolio risk limits
- Kelly criterion
- Maximum 5% per position"
```

## 2. Alert-Based Automation

### Price Breakout Alerts
```
"Monitor these stocks and alert me when:
- Price breaks 20-day high with volume
- Neural forecast shows >3% move
- News sentiment is positive
Stocks: AAPL, TSLA, NVDA, SPY"
```

### Sentiment Spike Notifications
```
"Alert me immediately when:
- Any stock has >90% positive sentiment
- News volume is 3x normal
- Multiple sources confirm
Then run full analysis automatically"
```

## 3. Strategy Automation

### Automated Backtesting Cycles
```
"Every weekend, backtest all strategies:
1. Test on last month's data
2. Compare to benchmark
3. Identify underperformers
4. Suggest optimizations"
```

**Weekly Automation:**
```python
strategies = ['momentum', 'swing', 'mean_reversion']
for strategy in strategies:
    # Backtest
    ai-news-trader:run_backtest (MCP)(
        strategy: strategy,
        symbol: "SPY",
        start_date: last_month,
        end_date: today
    )
    
    # Optimize if underperforming
    if performance < benchmark:
        ai-news-trader:optimize_strategy (MCP)(
            strategy: strategy,
            optimization_metric: "sharpe_ratio"
        )
```

### Strategy Rotation
```
"Switch strategies based on market regime:
- VIX < 15: Use momentum
- VIX 15-25: Use swing trading
- VIX > 25: Use mean reversion
Check daily and adjust"
```

## 4. Risk Automation

### Dynamic Hedge Adjustments
```
"Monitor portfolio risk continuously:
- If portfolio beta > 1.2, add hedges
- If correlation spike detected, reduce positions
- If VaR exceeds limit, force risk reduction"
```

### Volatility-Based Position Sizing
```
"Adjust all position sizes based on:
- 20-day ATR
- Current VIX level
- Position correlation
Update daily at market open"
```

## 5. Research Automation

### Sector Analysis Cycles
```
"Every Monday, analyze all 11 sectors:
1. Performance vs SPY
2. Neural forecasts for ETFs
3. Top 3 stocks per sector
4. Correlation changes
Create report and store in memory"
```

### Alpha Factor Testing
```
"Test new alpha factors automatically:
1. Define factor (e.g., sentiment + momentum)
2. Backtest on universe
3. Calculate Information Ratio
4. Add to factor library if IR > 0.5"
```

## 6. Reporting Automation

### Daily P&L Report
```
"At 4:30 PM each day:
1. Calculate daily P&L
2. Attribution by strategy
3. Compare to targets
4. Risk metrics update
5. Email summary"
```

**Report Template:**
```
Daily Trading Report - [Date]
========================
P&L: $X,XXX (X.X%)
Sharpe: X.XX
Winners: X/X trades

By Strategy:
- Momentum: +X.X%
- Swing: +X.X%
- Arbitrage: +X.X%

Risk Metrics:
- VaR: $X,XXX
- Beta: X.XX
- Max Drawdown: X.X%
```

### Weekly Strategy Summary
```
"Every Friday at 5 PM:
1. Performance by strategy
2. Best/worst trades
3. Risk analysis
4. Optimization suggestions
5. Next week outlook"
```

## Implementation Examples

### Schedule-Based Trigger
```python
# Cron-style scheduling
schedule = {
    "morning_scan": "0 9 * * 1-5",      # 9 AM weekdays
    "risk_check": "*/30 * * * 1-5",     # Every 30 min
    "eod_report": "30 16 * * 1-5",      # 4:30 PM
    "weekly_optimization": "0 18 * * 5"  # 6 PM Friday
}
```

### Event-Based Trigger
```python
# React to market events
triggers = {
    "vix_spike": lambda: get_vix() > 30,
    "correlation_break": lambda: check_correlation() < 0.5,
    "news_event": lambda: check_news_volume() > 3 * average,
    "drawdown": lambda: get_drawdown() > 0.05
}
```

### Condition-Based Chain
```python
# Multi-condition automation
if sentiment > 0.8 and technical_score > 7:
    if neural_forecast > 0.03:
        execute_trade()
        set_stops()
        log_decision()
```

## Error Handling

### Retry Logic
```
"If any automation fails:
1. Retry 3 times with 30s delay
2. Use fallback data source
3. Alert me if still failing
4. Log all errors"
```

### Graceful Degradation
```
"If neural forecast unavailable:
- Use technical analysis
- Increase position size threshold
- Add confirmation requirement"
```

## Monitoring Setup

### Health Checks
```
"Monitor automation health:
- Tool response times
- Success rates
- Data freshness
- System resources"
```

### Performance Tracking
```
"Track automation effectiveness:
- Signals generated vs executed
- Automation P&L attribution
- Time saved
- Error rates"
```

## Quick Start Templates

### 1. Basic Daily Automation
```
"Set up my daily routine:
Morning: Portfolio check, news scan, forecasts
Midday: Risk check, position adjustments
Evening: P&L report, next day planning"
```

### 2. Alert System
```
"Create alerts for:
- 3% moves in my positions
- High news sentiment (>85%)
- Risk limit breaches
- Unusual options activity"
```

### 3. Research Pipeline
```
"Automate weekly research:
- Screen for momentum stocks
- Run neural forecasts
- Backtest top candidates
- Generate trade ideas"
```

## Best Practices

1. **Start Simple**: Begin with daily reports before complex automation
2. **Test Thoroughly**: Run automations in simulation first
3. **Monitor Closely**: Check automation logs daily
4. **Version Control**: Track automation changes
5. **Fail-Safe**: Always have manual override
6. **Documentation**: Document all automation logic

Remember: Claude Code handles the orchestration - just describe what you want automated!