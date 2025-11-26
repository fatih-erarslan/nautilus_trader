#!/bin/bash
# Claude Flow Command: Demo - Trading Execution

echo "💹 Claude Flow: Trading Execution Demo"
echo "====================================="
echo ""
echo "Execute trades, manage orders, and track performance."
echo ""

cat << 'EOF'
### Trading Execution Workflow

#### 1. Trade Simulation
```
Use: mcp__ai-news-trader__simulate_trade
Parameters:
  strategy: "momentum_trading_optimized"
  symbol: "NVDA"
  action: "buy"
  use_gpu: true
```
Returns:
- Expected entry price
- Position size (shares)
- Risk metrics
- Profit/loss estimates

#### 2. Multi-Asset Batch Trading
```
Use: mcp__ai-news-trader__execute_multi_asset_trade
Parameters:
  trades: [
    {"symbol": "AAPL", "action": "buy", "quantity": 100},
    {"symbol": "NVDA", "action": "buy", "quantity": 50},
    {"symbol": "TSLA", "action": "sell", "quantity": 25}
  ]
  strategy: "momentum_trading_optimized"
  risk_limit: 50000      # Max total exposure
  execute_parallel: true  # Simultaneous execution
```

#### 3. Prediction Market Trading
```
# List available markets
Use: mcp__ai-news-trader__get_prediction_markets_tool
Parameters:
  category: "Crypto"
  sort_by: "volume"
  limit: 10

# Analyze specific market
Use: mcp__ai-news-trader__analyze_market_sentiment_tool
Parameters:
  market_id: "crypto_btc_100k"
  analysis_depth: "gpu_enhanced"
  include_correlations: true
  use_gpu: true

# Calculate expected value
Use: mcp__ai-news-trader__calculate_expected_value_tool
Parameters:
  market_id: "crypto_btc_100k"
  investment_amount: 1000
  confidence_adjustment: 1.1
  include_fees: true
  use_gpu: true

# Place prediction order
Use: mcp__ai-news-trader__place_prediction_order_tool
Parameters:
  market_id: "crypto_btc_100k"
  outcome: "yes"
  side: "buy"
  quantity: 500
  order_type: "limit"
  limit_price: 0.35
```

#### 4. Performance Tracking
```
Use: mcp__ai-news-trader__performance_report
Parameters:
  strategy: "momentum_trading_optimized"
  period_days: 30
  include_benchmark: true
  use_gpu: true
```

#### 5. Execution Analytics
```
Use: mcp__ai-news-trader__get_execution_analytics
Parameters:
  time_period: "1d"  # 1h, 1d, 1w available
```

### Order Types & Execution

**Market Orders**
- Immediate execution at best price
- Use for: Liquid stocks, urgent trades
- Risk: Slippage in volatile markets

**Limit Orders**
- Execute at specific price or better
- Use for: Large positions, illiquid assets
- Risk: May not fill

**Stop-Loss Orders**
- Trigger market order at stop price
- Use for: Risk management
- Types: Fixed, trailing, time-based

**Advanced Orders**
- TWAP: Time-weighted average price
- VWAP: Volume-weighted average price
- Iceberg: Hide large order size

### Execution Best Practices

#### 1. Pre-Trade Analysis
```python
# Check liquidity
orderbook = get_market_depth(symbol)
spread = orderbook.ask - orderbook.bid
if spread > max_spread:
    use_limit_order()

# Verify signals
signals = {
    "technical": get_technical_signal(),
    "sentiment": get_news_sentiment(),
    "neural": get_neural_forecast()
}
if sum(signals.values()) >= 2:
    proceed_with_trade()
```

#### 2. Smart Order Routing
```python
# Split large orders
if order_size > daily_volume * 0.01:
    chunks = split_order(order_size, n=10)
    execute_over_time(chunks, minutes=30)

# Choose execution venue
venues = ["NYSE", "NASDAQ", "ARCA", "IEX"]
best_venue = find_best_liquidity(symbol, venues)
route_order_to(best_venue)
```

#### 3. Real-Time Monitoring
```python
# Track execution quality
while order.status != "filled":
    check_partial_fills()
    monitor_slippage()
    adjust_limit_if_needed()
    
# Post-trade analysis
actual_price = order.average_fill_price
expected_price = order.expected_price
slippage = (actual_price - expected_price) / expected_price
log_execution_metrics(slippage, fill_time)
```

### Performance Metrics

**Trade Execution**
- Fill Rate: % of orders completed
- Slippage: Difference from expected price
- Speed: Milliseconds to execution
- Price Improvement: Better than expected

**Strategy Performance**
- Win Rate: Profitable trades %
- Profit Factor: Gross profit / Gross loss
- Average Win/Loss: Risk/reward ratio
- Recovery Factor: Net profit / Max drawdown

**Risk-Adjusted Returns**
- Sharpe Ratio: Return per unit of risk
- Sortino Ratio: Downside risk focus
- Calmar Ratio: Return / Max drawdown
- Information Ratio: Active return / Tracking error

### Live Trading Checklist

✅ Pre-Market
- [ ] Check news and sentiment
- [ ] Review overnight moves
- [ ] Update neural forecasts
- [ ] Set daily risk limits

✅ Market Hours
- [ ] Monitor open positions
- [ ] Execute new signals
- [ ] Adjust stop losses
- [ ] Track performance

✅ Post-Market
- [ ] Review executions
- [ ] Update P&L
- [ ] Analyze mistakes
- [ ] Plan next day

### Prediction Market Strategies

**Kelly Criterion Sizing**
```
Optimal Bet = (p(win) × odds - 1) / (odds - 1)
Where p(win) = your estimated probability
```

**Arbitrage Opportunities**
- Cross-market spreads
- Time decay exploitation
- Correlated market hedging

**Risk Management**
- Never bet > 25% of bankroll
- Diversify across markets
- Monitor correlation risk

EOF

echo ""
echo "📈 Typical Execution Metrics:"
echo "- Fill Rate: 98%+"
echo "- Average Slippage: 0.05%"
echo "- Execution Speed: <100ms"
echo "- Daily Volume: 100-500 trades"
echo ""
echo "💰 Prediction Market Stats:"
echo "- Average ROI: 15-25% (skilled traders)"
echo "- Win Rate: 55-65%"
echo "- Kelly Sizing: 1-5% per bet"
echo ""
echo "📚 Full guide: /workspaces/ai-news-trader/demo/guides/trading_execution_demo.md"