# Advanced MCP Trading Patterns for Claude Code

This guide presents advanced trading patterns using MCP tools through Claude Code, with practical examples for common trading workflows.

## 1. Sequential Trading Patterns

### Morning Trading Routine
**Natural Language Request:**
"Run my morning trading routine: check portfolio, analyze overnight news for my positions, generate forecasts, and create a trading plan for today."

**MCP Tools Called (in sequence):**
1. `mcp__ai-news-trader__get_portfolio_status` → Current holdings
2. `mcp__ai-news-trader__analyze_news` → Overnight news sentiment
3. `mcp__ai-news-trader__neural_forecast` → Price predictions
4. `mcp__ai-news-trader__quick_analysis` → Market conditions

**Example Conversation:**
```
User: "Run my morning trading analysis for my portfolio"

Claude: I'll run your morning trading routine. Let me start by checking your portfolio status...

[Calls get_portfolio_status]
Current Portfolio:
- AAPL: 100 shares @ $185.50
- TSLA: 50 shares @ $245.00
- SPY: 200 shares @ $450.25

[Calls analyze_news for each position]
Overnight News Analysis:
- AAPL: Positive sentiment (+0.72) - New product launch rumors
- TSLA: Neutral sentiment (+0.15) - Q4 delivery numbers pending
- SPY: Slightly bearish (-0.25) - Fed minutes release today

[Calls neural_forecast for each]
24-Hour Forecasts:
- AAPL: $188.50 (±$2.10) - 1.6% upside potential
- TSLA: $242.00 (±$5.50) - 1.2% downside risk
- SPY: $449.00 (±$1.75) - Consolidation expected

Trading Plan:
1. Hold AAPL - positive momentum building
2. Set stop-loss on TSLA at $240
3. Watch SPY for Fed reaction at 2PM
```

### End-of-Day Review
**Natural Language Request:**
"Review today's trading performance, check portfolio risk levels, and suggest any position adjustments for tomorrow."

**MCP Tools Called:**
1. `mcp__ai-news-trader__performance_report` → Daily P&L
2. `mcp__ai-news-trader__risk_analysis` → Portfolio risk metrics
3. `mcp__ai-news-trader__correlation_analysis` → Position correlations
4. `mcp__ai-news-trader__optimize_strategy` → Adjustment recommendations

### Weekly Optimization Cycle
**Natural Language Request:**
"Run my weekly strategy optimization: backtest current parameters, optimize based on recent performance, and update trading rules."

**MCP Tools Called:**
1. `mcp__ai-news-trader__run_backtest` → Test current strategy
2. `mcp__ai-news-trader__optimize_strategy` → Find better parameters
3. `mcp__ai-news-trader__neural_optimize` → Tune neural models
4. `mcp__ai-news-trader__performance_report` → Compare results

## 2. Parallel Analysis Patterns

### Multi-Symbol Simultaneous Analysis
**Natural Language Request:**
"Analyze AAPL, MSFT, GOOGL, and AMZN simultaneously for the best opportunity today."

**MCP Tools Called (in parallel):**
- 4x `mcp__ai-news-trader__quick_analysis`
- 4x `mcp__ai-news-trader__neural_forecast`
- 4x `mcp__ai-news-trader__get_news_sentiment`
- 1x `mcp__ai-news-trader__correlation_analysis`

**Example Conversation:**
```
User: "Compare trading opportunities in big tech stocks: AAPL, MSFT, GOOGL, AMZN"

Claude: I'll analyze all four tech giants simultaneously to find the best opportunity...

[Parallel execution of analysis tools]

Comparative Analysis:
Symbol | Price | 24h Forecast | News Sentiment | Signal Strength
-------|-------|--------------|----------------|----------------
AAPL   | $185  | $188 (+1.6%) | +0.72 Bullish  | Strong Buy
MSFT   | $420  | $422 (+0.5%) | +0.45 Positive | Moderate Buy
GOOGL  | $155  | $154 (-0.6%) | -0.15 Neutral  | Hold
AMZN   | $175  | $173 (-1.1%) | -0.38 Bearish  | Avoid

Correlation Matrix shows AAPL has lowest correlation (0.65) with others.

Recommendation: AAPL presents the best risk/reward with strong technicals and positive sentiment.
```

### Cross-Timeframe Forecasting
**Natural Language Request:**
"Generate forecasts for SPY across multiple timeframes: 1 hour, 4 hours, daily, and weekly."

**MCP Tools Called:**
- `mcp__ai-news-trader__neural_forecast` (horizon=1)
- `mcp__ai-news-trader__neural_forecast` (horizon=4)
- `mcp__ai-news-trader__neural_forecast` (horizon=24)
- `mcp__ai-news-trader__neural_forecast` (horizon=168)

**Expected Results:**
```
SPY Multi-Timeframe Analysis:
- 1 Hour: $451.25 → $451.50 (+0.05%) - Ranging
- 4 Hour: $451.25 → $452.75 (+0.33%) - Bullish
- Daily: $451.25 → $454.00 (+0.61%) - Strong Bullish
- Weekly: $451.25 → $458.50 (+1.60%) - Trend Continuation
```

### Sector-Wide Correlation Analysis
**Natural Language Request:**
"Analyze correlations between tech sector ETFs and major tech stocks to find diversification opportunities."

**MCP Tools Called:**
- `mcp__ai-news-trader__correlation_analysis` (symbols=["XLK", "QQQ", "AAPL", "MSFT", "NVDA"])
- `mcp__ai-news-trader__risk_analysis` → Portfolio impact

## 3. Conditional Trading Patterns

### If-Then Trading Logic
**Natural Language Request:**
"If TSLA breaks above $250 with positive news sentiment, simulate a momentum trade. Otherwise, check for mean reversion opportunity below $240."

**MCP Tools Called (conditional flow):**
```python
if price > 250:
    - analyze_news(symbol="TSLA")
    - if sentiment > 0.5:
        - simulate_trade(strategy="momentum", symbol="TSLA", action="buy")
else if price < 240:
    - neural_forecast(symbol="TSLA", horizon=24)
    - if forecast > current_price * 1.02:
        - simulate_trade(strategy="mean_reversion", symbol="TSLA", action="buy")
```

### Multi-Condition Entry Signals
**Natural Language Request:**
"Only suggest a trade if: 1) Neural forecast shows >2% upside, 2) News sentiment is positive, 3) Risk is below 5% of portfolio."

**Example Implementation:**
```
User: "Find trades meeting all my criteria: 2%+ upside, positive news, <5% portfolio risk"

Claude: I'll scan for opportunities meeting all three criteria...

[Checking AAPL]
✓ Neural forecast: +2.3% in 24h
✓ News sentiment: +0.68 (positive)
✓ Risk assessment: 3.2% of portfolio
→ AAPL meets all criteria

[Checking NVDA]
✓ Neural forecast: +3.1% in 24h
✗ News sentiment: -0.22 (negative)
✓ Risk assessment: 4.5% of portfolio
→ NVDA fails sentiment criteria

Qualified trades: AAPL only
```

### Dynamic Stop-Loss Adjustments
**Natural Language Request:**
"Monitor my TSLA position and adjust stop-loss based on volatility and neural forecasts every 4 hours."

**MCP Tools Pattern:**
```
Every 4 hours:
1. get_portfolio_status() → Current position
2. risk_analysis() → Current volatility
3. neural_forecast() → Price trajectory
4. calculate new stop = max(entry - 2*volatility, forecast_support)
5. Log adjustment recommendation
```

## 4. Risk Management Patterns

### Pre-Trade Risk Assessment
**Natural Language Request:**
"Before entering any trade, run a full risk assessment including position sizing, portfolio impact, and correlation risk."

**MCP Tools Sequence:**
1. `mcp__ai-news-trader__risk_analysis` → Portfolio VaR
2. `mcp__ai-news-trader__correlation_analysis` → Existing correlations
3. Calculate position size based on Kelly Criterion
4. `mcp__ai-news-trader__simulate_trade` → Test impact

**Example Output:**
```
Risk Assessment for AAPL Trade:
- Current Portfolio VaR (95%): $2,450
- Adding 100 shares AAPL increases VaR to: $2,875
- Correlation with existing holdings: 0.72 (high)
- Recommended position size: 75 shares (reduced due to correlation)
- Max drawdown scenario: -$1,350 (-2.8% of portfolio)
```

### Position Sizing Based on Volatility
**Natural Language Request:**
"Calculate position sizes for my watchlist based on equal risk allocation of $1,000 per position."

**Implementation:**
```
For each symbol in [AAPL, TSLA, SPY, GLD]:
1. risk_analysis() → Get volatility
2. Position size = $1,000 / (price × daily_volatility × sqrt(holding_period))
3. Round to nearest share

Results:
- AAPL: 28 shares (volatility: 1.8%)
- TSLA: 16 shares (volatility: 3.2%)
- SPY: 44 shares (volatility: 0.9%)
- GLD: 52 shares (volatility: 0.7%)
```

## 5. Optimization Patterns

### A/B Strategy Testing
**Natural Language Request:**
"Test momentum strategy vs mean reversion on TSLA over the last 3 months and compare performance."

**MCP Tools Used:**
```
Parallel execution:
- run_backtest(strategy="momentum", symbol="TSLA", start="2024-10-01", end="2024-12-31")
- run_backtest(strategy="mean_reversion", symbol="TSLA", start="2024-10-01", end="2024-12-31")
- performance_report(strategy="momentum")
- performance_report(strategy="mean_reversion")
```

**Comparison Results:**
```
Strategy Comparison for TSLA:
                  Momentum | Mean Reversion
Returns:            18.5% | 12.3%
Sharpe Ratio:        1.85 | 2.15
Max Drawdown:       -8.2% | -5.1%
Win Rate:           58.0% | 72.0%
Avg Trade:          +1.2% | +0.6%

Mean reversion shows better risk-adjusted returns.
```

### Parameter Sweep Optimization
**Natural Language Request:**
"Optimize the moving average crossover strategy by testing MA periods from 10-50 for fast and 20-200 for slow."

**MCP Implementation:**
```python
optimize_strategy(
    strategy="ma_crossover",
    symbol="SPY",
    parameter_ranges={
        "fast_ma": [10, 50],
        "slow_ma": [20, 200],
        "stop_loss": [0.01, 0.05]
    },
    optimization_metric="sharpe_ratio",
    use_gpu=True
)
```

### Walk-Forward Analysis
**Natural Language Request:**
"Perform walk-forward analysis on my swing trading strategy: optimize on 3 months, test on 1 month, repeat for the last year."

**Pattern:**
```
For each quarter:
1. optimize_strategy() on 3-month training window
2. run_backtest() on 1-month test window
3. Store out-of-sample results
4. Move window forward 1 month

Aggregate results show true strategy robustness
```

## 6. News-Driven Patterns

### Event-Based Trading
**Natural Language Request:**
"Monitor for earnings announcements in my watchlist and prepare pre-earnings analysis 24 hours before."

**MCP Pattern:**
```
Daily scan:
1. Check earnings calendar for watchlist
2. For companies reporting tomorrow:
   - analyze_news(lookback_hours=168) → Week sentiment trend
   - neural_forecast(horizon=48) → Through earnings
   - risk_analysis() → Event volatility
   - Generate pre-earnings report
```

### Sentiment Threshold Triggers
**Natural Language Request:**
"Alert me when news sentiment for any of my positions drops below -0.5 or rises above +0.7."

**Implementation:**
```
Continuous monitoring:
while market_open:
    for position in portfolio:
        sentiment = get_news_sentiment(position)
        if sentiment < -0.5:
            alert(f"Negative sentiment alert: {position} at {sentiment}")
            suggest_action = simulate_trade(action="sell", symbol=position)
        elif sentiment > 0.7:
            alert(f"Positive sentiment surge: {position} at {sentiment}")
            suggest_action = simulate_trade(action="add", symbol=position)
```

### News-to-Trade Automation
**Natural Language Request:**
"When breaking news appears for NVDA with sentiment >0.8, automatically simulate a momentum trade and show me the results."

**Automated Flow:**
```
1. analyze_news(symbol="NVDA", lookback_hours=1)
2. If sentiment > 0.8 and news_count > 3:
   - neural_forecast(symbol="NVDA", horizon=4)
   - quick_analysis(symbol="NVDA")
   - simulate_trade(strategy="momentum", action="buy")
   - risk_analysis(portfolio + new_position)
3. Present complete analysis for decision
```

## Complex Pattern Examples

### Portfolio Rebalancing Trigger
**Natural Language Request:**
"Check if my portfolio needs rebalancing when any position exceeds 30% of total value or falls below 5%."

```
User: "Should I rebalance my portfolio?"

Claude: Analyzing portfolio allocation...

Current Allocation:
- AAPL: 38.2% (TRIGGER: >30%)
- MSFT: 25.1%
- GOOGL: 22.3%
- Cash: 14.4%

Rebalancing Plan:
1. Sell 125 shares AAPL → Reduce to 25%
2. Buy 40 shares GOOGL → Increase to 25%
3. Buy 30 shares MSFT → Increase to 25%
4. Result: Equal 25% allocation across positions
```

### Pairs Trading Opportunity Scanner
**Natural Language Request:**
"Find pairs trading opportunities in the S&P 500 technology sector with correlation >0.8 and current price divergence >2 standard deviations."

```
MCP Flow:
1. correlation_analysis(symbols=[tech_stocks])
2. For each highly correlated pair:
   - Calculate historical spread
   - Identify current deviation
   - neural_forecast() for both symbols
   - simulate_trade() for convergence play
```

## Best Practices for Pattern Implementation

1. **Always validate patterns with backtesting** before live implementation
2. **Use GPU acceleration** for complex patterns involving multiple neural forecasts
3. **Implement circuit breakers** for automated patterns (max trades/day, max position size)
4. **Log all pattern executions** for performance analysis
5. **Regular pattern review** - optimize patterns based on live performance
6. **Combine patterns** for more sophisticated strategies

## Quick Reference: Common Pattern Combinations

| Pattern Type | Common Combinations | Best For |
|-------------|-------------------|----------|
| Morning Routine | Portfolio Status + News + Forecasts | Daily Planning |
| Risk Check | Risk Analysis + Correlation + Position Sizing | Pre-Trade |
| Optimization | Backtest + Optimize + Neural Tune | Weekly/Monthly |
| News Trading | Sentiment + Forecast + Quick Analysis | Event-Driven |
| Rebalancing | Portfolio Status + Risk + Correlation | Portfolio Management |