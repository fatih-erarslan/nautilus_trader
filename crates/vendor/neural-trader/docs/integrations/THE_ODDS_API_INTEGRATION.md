# 🎯 The Odds API Integration Guide

## Overview

The Neural Trading System now includes comprehensive integration with **The Odds API** - the leading sports betting data provider with access to 250+ bookmakers worldwide. This integration provides real-time odds, arbitrage detection, and advanced betting analytics.

## 🚀 Key Features

### ✅ **Live Sports Data**
- Real-time odds from 250+ bookmakers
- Coverage of 25+ sports including NFL, NBA, MLB, Premier League
- Multiple betting markets: moneyline, spreads, totals, outrights
- Support for US, UK, AU, and EU regions

### ✅ **Advanced Analytics**
- **Arbitrage Detection**: Automatically find risk-free betting opportunities
- **Margin Comparison**: Compare bookmaker margins to find best value
- **Implied Probability**: Convert odds to probabilities with Kelly Criterion
- **Odds Movement**: Track price changes over time

### ✅ **MCP Tool Integration**
- 9 specialized MCP tools for betting analysis
- Rate limiting and error handling
- Caching for frequently accessed data
- GPU acceleration support

## 🔧 Setup Instructions

### Step 1: Get Your API Key

1. Visit [The Odds API](https://the-odds-api.com/)
2. Sign up for a free account (500 requests/month)
3. Navigate to your dashboard and copy your API key

### Step 2: Configure Environment

Add your API key to your `.env` file:

```bash
# The Odds API Configuration
THE_ODDS_API_KEY=your-api-key-here
```

⚠️ **Never commit your API key to version control!**

### Step 3: Verify Integration

Test the integration:

```bash
# Test basic connection
python -c "
from src.odds_api.client import OddsAPIClient
client = OddsAPIClient()
result = client.validate_connection()
print(result)
"
```

Expected output:
```json
{
  "status": "success",
  "api_key_valid": true,
  "sports_available": 25,
  "requests_remaining": 499,
  "message": "Successfully connected to The Odds API"
}
```

## 📊 Available MCP Tools

### 1. **odds_api_get_sports**
Get list of available sports and leagues.

**Usage:**
```python
# Get all available sports
sports = odds_api_get_sports()
```

**Response:**
```json
{
  "status": "success",
  "total_sports": 25,
  "active_sports": 15,
  "sports_by_group": {
    "American Football": [...],
    "Basketball": [...],
    "Soccer": [...]
  },
  "popular_sports": [
    {"key": "americanfootball_nfl", "title": "NFL"},
    {"key": "basketball_nba", "title": "NBA"}
  ]
}
```

### 2. **odds_api_get_live_odds**
Get live odds for a specific sport.

**Parameters:**
- `sport`: Sport key (e.g., "americanfootball_nfl")
- `regions`: Regions ("us", "uk", "au", "eu")
- `markets`: Markets ("h2h", "spreads", "totals")
- `bookmakers`: Optional bookmaker filter
- `odds_format`: "decimal" or "american"

**Usage:**
```python
# Get NFL odds from US bookmakers
odds = odds_api_get_live_odds(
    sport="americanfootball_nfl",
    regions="us",
    markets="h2h,spreads,totals"
)
```

### 3. **odds_api_find_arbitrage**
Find arbitrage opportunities across bookmakers.

**Parameters:**
- `sport`: Sport key
- `regions`: Multiple regions for coverage
- `markets`: Markets to analyze
- `min_profit_margin`: Minimum profit (default: 0.01 = 1%)

**Usage:**
```python
# Find NBA arbitrage opportunities
arb = odds_api_find_arbitrage(
    sport="basketball_nba",
    regions="us,uk,au",
    min_profit_margin=0.02  # 2% minimum profit
)
```

**Response:**
```json
{
  "status": "success",
  "arbitrage_opportunities": 3,
  "opportunities": [
    {
      "event": {
        "home_team": "Lakers",
        "away_team": "Warriors",
        "commence_time": "2025-01-15T02:00:00Z"
      },
      "arbitrage": {
        "has_arbitrage": true,
        "profit_margin": 0.025,
        "profit_percentage": 2.5,
        "best_odds": {"Lakers": 2.10, "Warriors": 1.95},
        "best_bookmakers": {"Lakers": "draftkings", "Warriors": "fanduel"}
      }
    }
  ]
}
```

### 4. **odds_api_get_event_odds**
Get detailed odds for a specific event.

**Usage:**
```python
# Get detailed odds for specific game
event_odds = odds_api_get_event_odds(
    sport="basketball_nba",
    event_id="event_12345",
    markets="h2h,spreads,totals"
)
```

### 5. **odds_api_calculate_probability**
Convert odds to implied probability.

**Usage:**
```python
# Calculate probability from decimal odds
prob = odds_api_calculate_probability(
    odds=2.50,
    odds_format="decimal"
)
# Returns: {"implied_probability": 0.40, "implied_probability_percent": 40.0}
```

### 6. **odds_api_compare_margins**
Compare bookmaker margins to find best value.

**Usage:**
```python
# Compare margins across bookmakers
margins = odds_api_compare_margins(
    sport="americanfootball_nfl",
    regions="us"
)
```

### 7. **odds_api_get_upcoming**
Get upcoming events with odds.

**Usage:**
```python
# Get upcoming NBA games for next 7 days
upcoming = odds_api_get_upcoming(
    sport="basketball_nba",
    days_ahead=7
)
```

### 8. **odds_api_get_bookmaker_odds**
Get odds from a specific bookmaker.

**Usage:**
```python
# Get DraftKings odds only
dk_odds = odds_api_get_bookmaker_odds(
    sport="americanfootball_nfl",
    bookmaker="draftkings"
)
```

### 9. **odds_api_analyze_movement**
Analyze odds movement over time.

**Usage:**
```python
# Track odds movement for specific event
movement = odds_api_analyze_movement(
    sport="basketball_nba",
    event_id="event_12345"
)
```

## 🎯 Practical Use Cases

### Arbitrage Trading Bot

```python
# Find and execute arbitrage opportunities
def find_arbitrage_opportunities():
    sports = ["americanfootball_nfl", "basketball_nba", "baseball_mlb"]

    for sport in sports:
        arb_data = odds_api_find_arbitrage(
            sport=sport,
            regions="us,uk,au",
            min_profit_margin=0.015  # 1.5% minimum
        )

        if arb_data["arbitrage_opportunities"] > 0:
            print(f"Found {arb_data['arbitrage_opportunities']} opportunities in {sport}")
            # Execute trades...
```

### Kelly Criterion Betting

```python
# Calculate optimal bet size using Kelly Criterion
def kelly_bet_sizing(odds, win_probability):
    prob_data = odds_api_calculate_probability(odds, "decimal")
    implied_prob = prob_data["implied_probability"]

    # Kelly formula: f = (bp - q) / b
    # where b = odds - 1, p = win prob, q = lose prob
    b = odds - 1
    p = win_probability
    q = 1 - p

    kelly_fraction = (b * p - q) / b
    return max(0, kelly_fraction)  # Don't bet if negative
```

### Market Monitoring

```python
# Monitor specific markets for value
def monitor_nfl_lines():
    odds = odds_api_get_live_odds(
        sport="americanfootball_nfl",
        markets="spreads",
        regions="us"
    )

    for event in odds["events"]:
        # Analyze spread movements
        margins = odds_api_compare_margins(
            sport="americanfootball_nfl"
        )

        # Alert on favorable conditions
        if margins["best_value_bookmaker"]:
            print(f"Best value at {margins['best_value_bookmaker']}")
```

## 📈 Rate Limits & Usage

### Free Tier Limits
- **500 requests/month**
- **Rate limit**: 30 requests/second
- **Historical data**: None (live only)

### Paid Tiers
- **$10/month**: 10,000 requests
- **$50/month**: 100,000 requests
- **$200/month**: 1,000,000 requests

### Rate Limit Handling
The integration includes automatic rate limiting:

```python
# Client automatically manages rate limits
client = OddsAPIClient()

# Check usage
usage = client.get_usage_info()
print(f"Requests remaining: {usage['requests_remaining']}")
```

## 🛡️ Error Handling

The integration includes comprehensive error handling:

```python
try:
    odds = odds_api_get_live_odds("basketball_nba")
    if odds["status"] == "error":
        print(f"API Error: {odds['error']}")
except Exception as e:
    print(f"Connection error: {e}")
```

**Common Error Codes:**
- `401`: Invalid API key
- `422`: Invalid parameters
- `429`: Rate limit exceeded
- `500`: Server error

## 🔄 Data Refresh Rates

**Live Events**: Updated every 5-10 seconds
**Upcoming Events**: Updated every 30 minutes
**Outrights**: Updated every 6 hours

## 📊 Supported Sports

### Major Sports (High Coverage)
- **NFL** (`americanfootball_nfl`)
- **NBA** (`basketball_nba`)
- **MLB** (`baseball_mlb`)
- **NHL** (`icehockey_nhl`)
- **Premier League** (`soccer_epl`)
- **Champions League** (`soccer_uefa_champs_league`)

### Additional Sports
- College Football/Basketball
- Tennis (ATP/WTA)
- Golf (PGA/European Tour)
- MMA/Boxing
- Cricket
- Rugby
- And 15+ more

## 🔧 Advanced Configuration

### Custom Client Settings

```python
from src.odds_api.client import OddsAPIClient

# Custom configuration
client = OddsAPIClient(api_key="your-key")
client.min_request_interval = 2.0  # 2 seconds between requests
client._sports_cache_duration = 7200  # 2 hour cache
```

### Bulk Data Collection

```python
# Collect odds for multiple sports efficiently
sports = ["americanfootball_nfl", "basketball_nba", "baseball_mlb"]
all_odds = {}

for sport in sports:
    all_odds[sport] = odds_api_get_live_odds(sport, "us", "h2h")
    time.sleep(1)  # Respect rate limits
```

## 🚨 Best Practices

### 1. **Rate Limit Management**
- Cache frequently accessed data
- Batch requests when possible
- Monitor usage with `get_usage_info()`

### 2. **Data Storage**
- Store historical odds for trend analysis
- Implement database for arbitrage tracking
- Cache sports list (changes infrequently)

### 3. **Error Recovery**
- Implement retry logic with exponential backoff
- Handle network timeouts gracefully
- Log errors for debugging

### 4. **Security**
- Never log API keys
- Use environment variables only
- Rotate keys regularly

## 🔗 Integration with Neural Trading

The Odds API integrates seamlessly with the neural trading system:

```python
# Use with existing neural models
from src.neural_models import TradingModel

model = TradingModel()
odds = odds_api_get_live_odds("americanfootball_nfl")

# Feed odds data to neural model for predictions
predictions = model.predict_betting_value(odds["events"])
```

## 📞 Support & Resources

- **API Documentation**: https://the-odds-api.com/liveapi/guides/v4/
- **Support Email**: support@the-odds-api.com
- **Status Page**: https://status.the-odds-api.com/
- **Rate Limit Calculator**: https://the-odds-api.com/pricing

## 🎉 Getting Started

1. **Sign up** for The Odds API free account
2. **Add your API key** to `.env` file
3. **Test connection** with `odds_api_get_sports()`
4. **Start building** your betting strategies!

---

**Ready to dominate the sports betting markets with AI-powered analytics? Let's get started! 🚀**