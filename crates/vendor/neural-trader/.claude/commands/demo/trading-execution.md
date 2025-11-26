# Claude Code Demo: Trading Execution

Learn how to simulate trades, execute orders, and analyze prediction markets using MCP tools.

## Trade Simulation

### Single Trade Simulation
```
Simulate a buy trade for NVIDIA:
Use mcp__ai-news-trader__simulate_trade with:
- strategy: "momentum_trading_optimized"
- symbol: "NVDA"
- action: "buy"
- use_gpu: true

Show expected entry price, position size, and profit targets.
```

### Multiple Trade Simulations
```
Simulate trades for my watchlist:
For each symbol in [AAPL, NVDA, TSLA, GOOGL]:
Use mcp__ai-news-trader__simulate_trade with:
- strategy: "swing_trading_optimized"
- action: "buy"
- use_gpu: true

Compare expected returns and select best opportunity.
```

### Short Selling Simulation
```
Simulate short positions:
Use mcp__ai-news-trader__simulate_trade with:
- strategy: "mean_reversion_optimized"
- symbol: "TSLA"
- action: "sell"
- use_gpu: true

Show borrowing costs and risk metrics for short position.
```

## Multi-Asset Trading

### Batch Order Execution
```
Execute multiple trades simultaneously:
Use mcp__ai-news-trader__execute_multi_asset_trade with:
- trades: [
    {"symbol": "AAPL", "action": "buy", "quantity": 100},
    {"symbol": "NVDA", "action": "buy", "quantity": 50},
    {"symbol": "TSLA", "action": "sell", "quantity": 25},
    {"symbol": "GOOGL", "action": "buy", "quantity": 40}
  ]
- strategy: "momentum_trading_optimized"
- risk_limit: 100000
- execute_parallel: true

Show execution summary and total portfolio impact.
```

### Pairs Trading Execution
```
Execute a pairs trade:
Use mcp__ai-news-trader__execute_multi_asset_trade with:
- trades: [
    {"symbol": "GM", "action": "buy", "quantity": 200},
    {"symbol": "F", "action": "sell", "quantity": 400}
  ]
- strategy: "mean_reversion_optimized"
- risk_limit: 20000
- execute_parallel: true

Monitor spread and set exit conditions.
```

### Portfolio Rebalancing Trade
```
Execute rebalancing trades:
First, calculate required trades with portfolio_rebalance
Then use mcp__ai-news-trader__execute_multi_asset_trade with:
- trades: [calculated rebalancing trades]
- strategy: "passive_rebalancing"
- execute_parallel: true

Confirm new allocations match targets.
```

## Prediction Markets

### Discover Markets
```
Find active prediction markets:
Use mcp__ai-news-trader__get_prediction_markets_tool with:
- category: "Crypto"
- sort_by: "volume"
- limit: 20

Show markets with highest liquidity and interesting odds.
```

### Market Analysis
```
Analyze Bitcoin 100k prediction market:
Use mcp__ai-news-trader__analyze_market_sentiment_tool with:
- market_id: "crypto_btc_100k"
- analysis_depth: "gpu_enhanced"
- include_correlations: true
- use_gpu: true

Provide detailed analysis with Kelly criterion sizing.
```

### Order Book Analysis
```
Check market depth:
Use mcp__ai-news-trader__get_market_orderbook_tool with:
- market_id: "crypto_btc_100k"
- depth: 20

Identify liquidity levels and optimal entry points.
```

### Expected Value Calculation
```
Calculate betting opportunity:
Use mcp__ai-news-trader__calculate_expected_value_tool with:
- market_id: "crypto_btc_100k"
- investment_amount: 5000
- confidence_adjustment: 1.2
- include_fees: true
- use_gpu: true

Show expected returns and optimal bet size.
```

### Place Prediction Order
```
Place a prediction market bet:
Use mcp__ai-news-trader__place_prediction_order_tool with:
- market_id: "crypto_btc_100k"
- outcome: "yes"
- side: "buy"
- quantity: 1000
- order_type: "limit"
- limit_price: 0.45

Track order status and potential profit.
```

### Monitor Positions
```
Check all prediction positions:
Use mcp__ai-news-trader__get_prediction_positions_tool

Show current P&L and exit strategies for each position.
```

## Performance Tracking

### Strategy Performance Report
```
Generate performance report:
Use mcp__ai-news-trader__performance_report with:
- strategy: "momentum_trading_optimized"
- period_days: 30
- include_benchmark: true
- use_gpu: true

Include win rate, Sharpe ratio, and attribution analysis.
```

### Execution Analytics
```
Analyze execution quality:
Use mcp__ai-news-trader__get_execution_analytics with:
- time_period: "1d"

Show slippage, fill rates, and latency statistics.
```

### Custom Performance Periods
```
Compare weekly performance:
For each week in the last month:
Use mcp__ai-news-trader__performance_report with:
- strategy: "swing_trading_optimized"
- period_days: 7
- include_benchmark: true

Identify best and worst performing weeks.
```

## Advanced Execution Strategies

### TWAP Execution
```
Execute large order over time:
Break 10,000 share order into 20 chunks of 500
Execute every 15 minutes using:
Use mcp__ai-news-trader__execute_multi_asset_trade

This minimizes market impact for large positions.
```

### Smart Order Routing
```
Find best execution venue:
1. Check multiple exchanges for best price
2. Split order across venues
3. Use limit orders to capture spread
4. Monitor fill quality

Optimize for price improvement over speed.
```

### Conditional Orders
```
Set up conditional trades:
If AAPL breaks above 190:
  - Buy 200 shares
  - Set stop loss at 187
  - Set profit target at 195

If NVDA sentiment turns negative:
  - Reduce position by 50%
  - Move stop loss higher
```

## Real-Time Trading Workflows

### Day Trading Setup
```
Morning routine:
1. Check pre-market movers
2. Run sentiment analysis on news
3. Simulate trades for top 5 opportunities
4. Set alerts for entry signals
5. Execute trades with tight stops

Use all relevant MCP tools in sequence.
```

### Swing Trading Management
```
Position management:
1. Daily: Check sentiment trends
2. Daily: Update stop losses
3. Weekly: Review position sizing
4. On signal: Add to winners
5. On reversal: Exit positions

Maintain trading journal with results.
```

### Automated Trading Logic
```
Create systematic execution:
Every 30 minutes:
1. Scan for momentum signals
2. Check risk limits
3. Simulate potential trades
4. Execute if criteria met:
   - Positive expected value
   - Risk within limits
   - Sentiment confirms
   - Technical setup valid
```

## Output Templates

### Trade Execution Summary
```
Trade Execution Report
────────────────────
Strategy: Momentum Trading
Symbol: NVDA
Action: BUY
Quantity: 100 shares
Entry Price: $487.50
Total Value: $48,750

Risk Metrics:
- Stop Loss: $475.00 (-2.6%)
- Target: $510.00 (+4.6%)
- Risk/Reward: 1:1.8
- Position Size: 4.9% of portfolio

Status: ✅ EXECUTED (Demo Mode)
```

### Prediction Market Analysis
```
Market: Will BTC reach $100k by Dec 2024?
Current Odds: YES: 42% | NO: 58%
Your Analysis: 55% probability of YES

Expected Value Calculation:
- Bet $1,000 on YES at 42%
- Win: $2,380 (profit $1,380)
- Expected Value: +$209 (+20.9%)
- Kelly Criterion: Bet 13% of bankroll
- Recommended: $650 position

Risk: Market resolves in 45 days
```

### Performance Dashboard
```
30-Day Performance Summary
─────────────────────────
Strategy: Momentum Trading
Total Return: +18.5%
Benchmark (SPY): +3.2%
Alpha: +15.3%

Trade Statistics:
- Total Trades: 47
- Win Rate: 62%
- Avg Win: +2.8%
- Avg Loss: -1.4%
- Profit Factor: 2.4

Risk Metrics:
- Sharpe Ratio: 2.15
- Max Drawdown: -7.3%
- Daily VaR: $1,250
```

## Best Practices

### Order Execution
1. **Use limit orders** for entries to control price
2. **Split large orders** to reduce market impact
3. **Trade liquid hours** (first and last hour)
4. **Set stops immediately** after entry
5. **Monitor slippage** and adjust tactics

### Risk Management
1. **Size positions** based on volatility
2. **Never risk >2%** per trade
3. **Use correlation** to avoid concentration
4. **Take profits** at predetermined levels
5. **Review trades** weekly for improvement

### Prediction Markets
1. **Focus on liquidity** - trade active markets
2. **Understand resolution** criteria clearly
3. **Diversify bets** across markets
4. **Use Kelly criterion** for sizing
5. **Track record** predictions vs outcomes

## Common Issues

### Execution Problems
- **Slippage**: Use limit orders or trade smaller size
- **Partial fills**: Set "all or none" for full execution
- **Rejected orders**: Check buying power and restrictions
- **Wide spreads**: Wait for better liquidity

### Performance Tracking
- **Missing trades**: Ensure all executions are logged
- **Wrong benchmark**: Use appropriate comparison index
- **Period mismatch**: Align dates for fair comparison
- **Cost inclusion**: Always include fees and slippage