# Part 8: Sports Betting & Syndicates
**Duration**: 10 minutes | **Difficulty**: Advanced

## ⚽ Neural Sports Betting

Neural Trader transforms sports betting with AI-powered analysis, arbitrage detection, and syndicate management for collaborative betting.

## 🏆 Sports Markets Overview

### Available Sports
```bash
# List upcoming events
claude "Show sports events for next 7 days"

# Focus on specific sport
claude "Get NFL games with betting opportunities"
```

Supported sports:
- **Football**: NFL, NCAA, Soccer
- **Basketball**: NBA, NCAA
- **Baseball**: MLB
- **Hockey**: NHL
- **Tennis**: ATP, WTA
- **MMA/Boxing**: UFC, major fights
- **Esports**: LoL, CS:GO, Dota 2

## 📊 Odds Analysis

### 1. Multi-Book Comparison
```bash
# Compare odds across bookmakers
claude "Compare odds for Lakers vs Warriors:
- DraftKings
- FanDuel
- BetMGM
- Caesars
Show best prices for each outcome"
```

Returns:
```python
{
    "Lakers ML": {
        "best_odds": +150,
        "book": "DraftKings",
        "implied_prob": 0.40
    },
    "Warriors ML": {
        "best_odds": -170,
        "book": "FanDuel",
        "implied_prob": 0.63
    },
    "arbitrage_available": False,
    "edge": -0.03  # 3% vig
}
```

### 2. Arbitrage Detection
```bash
# Find arbitrage opportunities
claude "Scan all NFL games for arbitrage with min 1% profit"
```

Example opportunity:
```
Game: Chiefs vs Bills
Bet $618.18 on Chiefs +2.5 @ -110 (Book A)
Bet $381.82 on Bills -2.5 @ +180 (Book B)
Guaranteed profit: $11.82 (1.18%)
```

### 3. Line Movement Analysis
```bash
# Track line movements
claude "Show line movement for Super Bowl:
- Opening lines
- Current lines
- Sharp money indicators
- Public betting percentage"
```

## 💰 Kelly Criterion Betting

### Optimal Bet Sizing
```bash
# Calculate optimal bet size
claude "Calculate Kelly bet for:
- Bankroll: $10,000
- Odds: +150
- Win probability: 45%
- Kelly fraction: 0.25 (quarter Kelly for safety)"
```

Formula: `f = (p * (b + 1) - 1) / b`
- Recommended bet: $312.50

### Multi-Bet Kelly
```bash
# Size multiple bets optimally
claude "Calculate Kelly for 5-game parlay:
Show individual and combined sizing"
```

## 🤝 Syndicate Management

### 1. Create Syndicate
```bash
# Initialize betting syndicate
claude "Create sports betting syndicate:
- Name: 'Sharp Shooters'
- Initial capital: $50,000
- Members: 10
- Profit split: 60% performance, 40% equal"
```

### 2. Add Members
```bash
# Add syndicate member
claude "Add member to syndicate:
- Name: John Doe
- Email: john@example.com
- Initial contribution: $5,000
- Role: Analyst
- Voting power: 1 share"
```

### 3. Fund Allocation
```bash
# Distribute funds across opportunities
claude "Allocate syndicate funds:
- NFL: 40% ($20,000)
- NBA: 30% ($15,000)
- MLB: 20% ($10,000)
- Reserve: 10% ($5,000)
Use Kelly sizing within each sport"
```

### 4. Profit Distribution
```bash
# Calculate member payouts
claude "Distribute syndicate profits:
- Total profit: $15,000
- Performance bonus pool: $9,000 (60%)
- Equal distribution: $6,000 (40%)
- Show individual member earnings"
```

Member earnings example:
```
Member A (20% of wins): $1,800 + $600 = $2,400
Member B (15% of wins): $1,350 + $600 = $1,950
Member C (10% of wins): $900 + $600 = $1,500
```

## 📈 Advanced Strategies

### 1. Model-Based Betting
```python
# Neural model predictions
model_config = {
    "features": [
        "team_elo_rating",
        "recent_performance",
        "injury_report",
        "weather_conditions",
        "rest_days",
        "home_advantage"
    ],
    "model_type": "neural_ensemble",
    "confidence_threshold": 0.65
}

claude "Generate predictions using neural model for all NBA games"
```

### 2. Live Betting Automation
```bash
# In-play betting bot
claude "Create live betting bot:
- Sport: NBA
- Strategy: Momentum shifts
- Entry: 10+ point swings
- Bet size: 1% of bankroll
- Exit: End of quarter"
```

### 3. Correlated Parlays
```bash
# Smart parlay construction
claude "Build correlated parlay:
- If Chiefs win, likely high scoring
- Bet: Chiefs ML + Over
- Show correlation coefficient
- Calculate true odds vs offered"
```

## 🛡 Risk Management

### Bankroll Management
```bash
# Set up controls
claude "Configure bankroll management:
- Daily limit: 5% of bankroll
- Single bet max: 2%
- Stop if down 10% in week
- Reduce size after 3 losses"
```

### Tracking & Analytics
```bash
# Performance analysis
claude "Analyze betting performance:
- ROI by sport
- Win rate by bet type
- Average odds taken
- Closing line value
- Sharp vs square analysis"
```

Returns:
```
Overall ROI: +8.3%
NFL: +12.1% (58% win rate)
NBA: +5.2% (54% win rate)
MLB: +6.8% (52% win rate)
CLV: +2.3% (beating closing line)
```

## 🤖 Automated Systems

### 1. Value Betting Bot
```bash
# Deploy value hunter
claude "Start value betting bot:
- Scan all major sports
- Minimum edge: 3%
- Max bets per day: 20
- Notification threshold: 5% edge
- Auto-bet if edge > 7%"
```

### 2. Arbitrage Bot
```bash
# Arbitrage automation
claude "Run arbitrage bot:
- Check every 30 seconds
- Minimum profit: 0.5%
- Maximum stake: $5,000
- Books to monitor: 10
- Alert and execute automatically"
```

## 📊 Syndicate Dashboard

```bash
# View syndicate status
claude "Show syndicate dashboard"
```

Displays:
```
Syndicate: Sharp Shooters
═════════════════════════
Capital: $65,000 (+30%)
Active Bets: 23
Today's P&L: +$2,340

Members: 10
Top Performer: Member A (+45%)
Pending Votes: 2

Recent Wins:
- Lakers +4.5 ✓ (+$1,200)
- Over 218.5 ✓ (+$800)
- Chiefs ML ✓ (+$650)
```

## 🎯 Quick Tools

### Line Shopping
```bash
claude "Find best line for Celtics -5.5"
```

### Hedge Calculator
```bash
claude "Calculate hedge for:
Original bet: $500 on +300
Current odds: -150
Show cashout value"
```

### Parlay Builder
```bash
claude "Build optimal 3-leg parlay from tonight's games"
```

## 🧪 Practice Exercises

### Exercise 1: Find Value
```bash
claude "Find tonight's best value bet:
- Minimum edge: 4%
- Maximum odds: +200
- Include reasoning"
```

### Exercise 2: Syndicate Simulation
```bash
claude "Simulate 30-day syndicate:
- Starting capital: $10,000
- Strategy: Value betting
- Show ending balance and ROI"
```

### Exercise 3: Risk Analysis
```bash
claude "Analyze risk for 10-game betting card:
- Total exposure: $2,000
- Show best/worst case
- Probability distribution"
```

## ✅ Key Takeaways

- [ ] Sports betting requires disciplined bankroll management
- [ ] Syndicates enable larger, diversified betting
- [ ] Kelly Criterion optimizes long-term growth
- [ ] Arbitrage provides risk-free profit
- [ ] Automation increases efficiency

## ⏭ Next Steps

Learn about sandbox workflows in [Sandbox Workflows](09-sandbox-workflows.md)

---

**Progress**: 70 min / 2 hours | [← Previous: Polymarket](07-advanced-polymarket.md) | [Back to Contents](README.md) | [Next: Sandboxes →](09-sandbox-workflows.md)