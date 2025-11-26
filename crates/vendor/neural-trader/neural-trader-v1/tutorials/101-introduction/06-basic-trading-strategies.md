# Part 6: Basic Trading Strategies
**Duration**: 10 minutes | **Difficulty**: Intermediate

## 📈 Available Trading Strategies

Neural Trader includes 15+ pre-built strategies, each optimized with neural enhancement:

### Core Strategies

| Strategy | Type | Best For | Risk Level |
|----------|------|----------|------------|
| Momentum | Trend Following | Trending Markets | Medium |
| Mean Reversion | Counter-Trend | Range-Bound Markets | Low-Medium |
| Swing Trading | Multi-Day | Volatile Stocks | Medium-High |
| Arbitrage | Market Neutral | Price Discrepancies | Low |
| News Sentiment | Event-Driven | News-Heavy Stocks | High |

## 🎯 Strategy Deep Dive

### 1. Momentum Trading

**Concept**: Buy assets showing upward momentum, sell on weakness

```bash
# Basic momentum strategy
claude "Run momentum strategy on SPY"

# With parameters
claude "Run momentum strategy with:
- Symbol: QQQ
- RSI threshold: 70/30
- Volume filter: 2x average
- Position size: $5000"
```

**Configuration**:
```python
momentum_config = {
    "lookback_period": 20,      # Days for momentum calculation
    "rsi_overbought": 70,       # Sell signal
    "rsi_oversold": 30,         # Buy signal
    "volume_multiplier": 1.5,   # Volume confirmation
    "stop_loss": 0.02,          # 2% stop loss
    "take_profit": 0.05         # 5% take profit
}
```

### 2. Mean Reversion

**Concept**: Trade on assumption prices return to average

```bash
# Simple mean reversion
claude "Execute mean reversion on AAPL"

# Advanced with Bollinger Bands
claude "Mean reversion using:
- Bollinger Bands (20, 2)
- Z-score threshold: 2
- Position scaling
- Risk limit: 1% per trade"
```

**Key Signals**:
```python
# Entry conditions
if price < lower_bollinger_band and z_score < -2:
    signal = "BUY"
elif price > upper_bollinger_band and z_score > 2:
    signal = "SELL"
```

### 3. Swing Trading

**Concept**: Capture multi-day price swings

```bash
# Swing trade setup
claude "Find swing trading setups in tech stocks"

# Automated swing trader
claude "Deploy swing trader:
- Scan top 50 liquid stocks
- Entry: Support bounce + RSI oversold
- Exit: Resistance or 5-day hold
- Risk: 1% per trade"
```

**Pattern Recognition**:
- Support/Resistance levels
- Chart patterns (flags, triangles)
- Fibonacci retracements
- Moving average crossovers

### 4. News Sentiment Trading

**Concept**: Trade on news sentiment analysis

```bash
# Real-time news trading
claude "Monitor news sentiment for TSLA and trade on extremes"

# Multi-source sentiment
claude "Aggregate sentiment from:
- Reuters, Bloomberg, Twitter
- Weight by source credibility
- Trade when consensus > 80%"
```

**Sentiment Scoring**:
```python
sentiment_sources = {
    "reuters": 0.3,      # 30% weight
    "bloomberg": 0.3,    # 30% weight
    "twitter": 0.2,      # 20% weight
    "reddit": 0.2        # 20% weight
}
```

### 5. Neural Enhanced Arbitrage

**Concept**: Exploit price differences with AI prediction

```bash
# Crypto arbitrage
claude "Find arbitrage opportunities between BTC exchanges"

# Statistical arbitrage
claude "Run pairs trading on correlated stocks:
- Scan for cointegrated pairs
- Trade divergences > 2 std dev
- Neural prediction for convergence timing"
```

## 💰 Position Sizing

### Kelly Criterion
```bash
# Optimal position sizing
claude "Calculate Kelly Criterion position size for AAPL trade"
```

Formula: `f = (p * b - q) / b`
- f = fraction of capital to bet
- p = probability of winning
- b = odds (profit/loss ratio)
- q = probability of losing (1-p)

### Fixed Percentage
```bash
# Risk 2% per trade
claude "Set position sizing to 2% portfolio risk per trade"
```

### Volatility-Based
```bash
# Adjust by volatility
claude "Use ATR-based position sizing with 1% risk"
```

## 🛡 Risk Management

### Stop Loss Types

1. **Fixed Percentage**
```bash
claude "Set 2% stop loss on all positions"
```

2. **Trailing Stop**
```bash
claude "Apply 5% trailing stop to winning positions"
```

3. **ATR-Based**
```bash
claude "Set stops at 2x ATR from entry"
```

4. **Support/Resistance**
```bash
claude "Place stops below key support levels"
```

### Risk Metrics
```bash
# Portfolio risk check
claude "Calculate current portfolio risk metrics:
- Value at Risk (VaR)
- Maximum drawdown
- Sharpe ratio
- Beta exposure"
```

## 📊 Strategy Combination

### Multi-Strategy Portfolio
```bash
# Diversified approach
claude "Create multi-strategy portfolio:
30% - Momentum (tech stocks)
30% - Mean reversion (SPY)
20% - Swing trading (small caps)
20% - News sentiment (crypto)"
```

### Adaptive Strategy Selection
```bash
# Market regime detection
claude "Auto-select strategy based on market conditions:
- Trending → Momentum
- Ranging → Mean reversion
- High volatility → Swing trading
- News-driven → Sentiment"
```

## 🧪 Backtesting

### Quick Backtest
```bash
# Last 30 days
claude "Backtest momentum strategy on AAPL for 30 days"
```

### Comprehensive Test
```bash
# Full analysis
claude "Run comprehensive backtest:
- Strategy: Mean reversion
- Period: 2 years
- Include transaction costs
- Monte Carlo simulation
- Walk-forward analysis"
```

### Performance Metrics
```python
backtest_results = {
    "total_return": "23.4%",
    "sharpe_ratio": 1.82,
    "max_drawdown": "-8.3%",
    "win_rate": "58%",
    "profit_factor": 1.65,
    "avg_trade": "$234"
}
```

## 🔬 Strategy Optimization

### Parameter Optimization
```bash
# Find optimal parameters
claude "Optimize momentum strategy parameters:
- RSI period: 10-30
- MA period: 20-50
- Stop loss: 1-5%
Use genetic algorithm with 1000 iterations"
```

### Walk-Forward Analysis
```bash
# Robust testing
claude "Run walk-forward optimization:
- In-sample: 6 months
- Out-sample: 2 months
- Step forward: 1 month
- Total period: 2 years"
```

## 📈 Live Trading

### Paper Trading First
```bash
# Test with paper money
claude "Start paper trading momentum strategy on QQQ"
```

### Go Live Checklist
- [ ] Backtest profitable over 1+ year
- [ ] Paper trade for 30 days minimum
- [ ] Risk controls in place
- [ ] Position sizing defined
- [ ] Emergency stop configured

### Start Live Trading
```bash
# Deploy with safeguards
claude "Deploy live momentum strategy:
- Symbol: SPY
- Capital: $10,000
- Max daily loss: $200
- Auto-stop if down 5%
- Paper trade parallel for comparison"
```

## 🎯 Quick Strategy Selection Guide

| Market Condition | Recommended Strategy | Risk Level |
|-----------------|---------------------|------------|
| Strong Uptrend | Momentum | Medium |
| Ranging/Sideways | Mean Reversion | Low |
| High Volatility | Swing Trading | Medium-High |
| Breaking News | Sentiment Trading | High |
| Calm Markets | Arbitrage | Low |

## ✅ Key Takeaways

- [ ] Each strategy suits different market conditions
- [ ] Risk management is crucial for all strategies
- [ ] Backtest before live trading
- [ ] Start with paper trading
- [ ] Combine strategies for diversification

## ⏭ Next Steps

Ready for advanced trading? Continue to [Advanced Polymarket Trading](07-advanced-polymarket.md)

---

**Progress**: 50 min / 2 hours | [← Previous: Claude Code UI](05-claude-code-ui.md) | [Back to Contents](README.md) | [Next: Polymarket →](07-advanced-polymarket.md)