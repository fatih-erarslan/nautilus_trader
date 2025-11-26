# Part 7: Advanced Polymarket Trading
**Duration**: 10 minutes | **Difficulty**: Intermediate-Advanced

## 🎲 What is Polymarket?

Polymarket is a decentralized prediction market where you trade on real-world event outcomes. Neural Trader adds AI-powered analysis for better predictions.

## 🚀 Getting Started with Polymarket

### List Available Markets
```bash
# View active prediction markets
claude "Show top Polymarket opportunities"

# Filter by category
claude "Show political prediction markets with >$100k volume"
```

### Market Categories
- **Politics**: Elections, policy decisions
- **Sports**: Game outcomes, championships  
- **Crypto**: Price predictions, protocol events
- **Economics**: Fed decisions, GDP, inflation
- **Technology**: Product launches, adoption metrics
- **Entertainment**: Awards, box office

## 📊 Market Analysis

### 1. Sentiment Analysis
```bash
# Analyze market sentiment
claude "Analyze Polymarket sentiment for 'Will BTC reach $100k by EOY?'"
```

Returns:
```python
{
    "market_id": "btc-100k-eoy",
    "current_probability": 0.42,
    "sentiment_score": 0.65,
    "volume_24h": "$2.3M",
    "liquidity": "$450K",
    "correlation_with_spot": 0.78,
    "news_impact": "positive",
    "recommendation": "BUY YES @ 0.42"
}
```

### 2. Order Book Analysis
```bash
# Get market depth
claude "Show orderbook for 'Fed rate cut by March'"
```

Displays:
```
YES Side          | NO Side
$5,000 @ 0.65    | $3,000 @ 0.36
$8,000 @ 0.64    | $6,000 @ 0.37
$12,000 @ 0.63   | $10,000 @ 0.38
```

### 3. Expected Value Calculation
```bash
# Calculate EV with fees
claude "Calculate expected value:
- Market: 'Trump wins 2024'
- YES price: 0.45
- My probability: 0.55
- Investment: $1000"
```

## 💰 Trading Strategies

### 1. Information Arbitrage
```python
# Trade on superior information processing
strategy = {
    "name": "news_arbitrage",
    "steps": [
        "Monitor breaking news",
        "Analyze impact faster than market",
        "Trade before probability adjusts",
        "Exit when market catches up"
    ]
}

# Execute
claude "Run information arbitrage on political markets"
```

### 2. Market Making
```bash
# Provide liquidity for profit
claude "Deploy market maker on high-volume markets:
- Spread: 2%
- Inventory limit: $5000
- Rebalance frequency: 5 min
- Risk limit: $500 daily loss"
```

### 3. Correlation Trading
```bash
# Trade correlated events
claude "Find correlated Polymarket events:
Example: 'Fed raises rates' correlates with 'Recession in 2024'
Trade the spread when correlation breaks"
```

### 4. Event-Driven Trading
```bash
# Trade around known events
claude "Set up event trading:
- Event: Presidential debate tonight
- Strategy: Fade extreme moves
- Size: Scale in over 2 hours
- Exit: Close by morning"
```

## 🤖 Automated Trading

### Setup Auto-Trader
```bash
# Configure automated system
claude "Create Polymarket auto-trader:
- Markets: Top 10 by volume
- Strategy: Mean reversion on overreactions
- Position limit: $500 per market
- Daily limit: $2000
- Stop if down 10%"
```

### Real-time Monitoring
```bash
# Live monitoring
claude "Monitor my Polymarket positions:
- Alert if any position moves >10%
- Show unrealized P&L
- Calculate portfolio Greeks
- Suggest hedges"
```

## 📈 Advanced Features

### 1. Multi-Market Portfolios
```bash
# Diversified prediction portfolio
claude "Build diversified Polymarket portfolio:
- 30% politics (low correlation)
- 30% sports (quick resolution)
- 20% crypto (high volatility)
- 20% economics (hedge inflation)"
```

### 2. Statistical Arbitrage
```python
# Find mispriced markets
arbitrage_scan = {
    "method": "statistical",
    "confidence": 0.95,
    "min_edge": 0.05,  # 5% minimum edge
    "max_exposure": 0.1,  # 10% of capital
    "markets": "all"
}

claude "Scan for statistical arbitrage opportunities"
```

### 3. Neural Prediction Models
```bash
# Use AI for probability estimation
claude "Train neural model on historical outcomes:
- Features: News sentiment, volume, momentum
- Target: Actual outcome
- Validation: Last 100 resolved markets
- Deploy on: Active markets"
```

## 🔧 Risk Management

### Position Sizing
```python
# Kelly Criterion for binary outcomes
def kelly_binary(probability, odds):
    """
    probability: Your estimated probability
    odds: Decimal odds offered
    """
    edge = probability * odds - 1
    return max(0, edge / (odds - 1))

# Example
claude "Calculate Kelly size:
My probability: 0.6
Market price: 0.5 (odds = 2.0)"
# Result: Bet 20% of bankroll
```

### Hedging Strategies
```bash
# Hedge correlated risks
claude "Hedge my Polymarket portfolio:
- Long 'Recession': $1000
- Hedge with: Short stocks, long bonds
- Correlation target: -0.5"
```

### Stop Loss Rules
```bash
# Risk controls
claude "Set Polymarket risk limits:
- Single market max: 5% of portfolio
- Daily loss limit: 10%
- Correlation limit: 0.7 between positions
- Auto-close if probability swings >30%"
```

## 💡 Tips & Tricks

### 1. Liquidity Timing
- Trade during US market hours
- Avoid thin markets (<$10k liquidity)
- Check spread before large trades

### 2. Resolution Risk
- Read resolution criteria carefully
- Avoid ambiguous markets
- Factor in settlement time

### 3. Information Sources
```bash
# Configure news sources
claude "Setup Polymarket news feeds:
- Twitter: Key accounts for each market
- News: Reuters, Bloomberg, AP
- Prediction sites: Metaculus, GJOpen
- Weight by accuracy history"
```

## 📊 Performance Tracking

```bash
# Track your edge
claude "Analyze my Polymarket performance:
- Win rate by category
- Average edge captured
- Sharpe ratio
- Compare to market probabilities"
```

## 🧪 Practice Exercise

### Exercise 1: Find Opportunity
```bash
claude "Find best Polymarket opportunity right now based on:
- Volume > $50k
- My edge > 5%
- Resolution < 30 days"
```

### Exercise 2: Build Strategy
```bash
claude "Create strategy for election markets:
- Use polling aggregation
- Sentiment analysis
- Historical accuracy weighting
- Position sizing by confidence"
```

### Exercise 3: Risk Analysis
```bash
claude "Analyze risk for position:
- Market: 'BTC above $75k'
- Position: $1000 YES @ 0.4
- Show scenarios and hedges"
```

## ✅ Polymarket Checklist

- [ ] Understand binary outcome structure
- [ ] Can calculate expected value
- [ ] Know position sizing methods
- [ ] Have risk management plan
- [ ] Set up monitoring alerts

## ⏭ Next Steps

Ready for sports betting? Continue to [Sports Betting & Syndicates](08-sports-betting-syndicates.md)

---

**Progress**: 60 min / 2 hours | [← Previous: Trading Strategies](06-basic-trading-strategies.md) | [Back to Contents](README.md) | [Next: Sports Betting →](08-sports-betting-syndicates.md)