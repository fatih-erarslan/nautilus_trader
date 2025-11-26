# Part 12: Supported APIs
**Duration**: 7 minutes | **Difficulty**: Beginner-Intermediate

## 🌐 API Ecosystem Overview

Neural Trader integrates with 20+ APIs for data, execution, and analysis. Here's your complete guide to available APIs and their capabilities.

## 📊 Market Data APIs

### 1. Alpha Vantage
**Free Tier Available**
```bash
# Setup
claude "Configure Alpha Vantage API:
- Key: YOUR_API_KEY
- Rate limit: 5 calls/min (free)
- Data: Stocks, forex, crypto"
```

Capabilities:
- Real-time quotes
- Historical data (20+ years)
- Technical indicators
- Fundamental data

### 2. Yahoo Finance
**Free - No API Key Required**
```python
# Direct usage via yfinance
import yfinance as yf
ticker = yf.Ticker("AAPL")
data = ticker.history(period="1mo")
```

Features:
- Real-time prices
- Historical OHLCV
- Options chains
- Fundamental data

### 3. Polygon.io
**Free Tier: 5 API calls/min**
```bash
claude "Connect to Polygon API:
- Real-time WebSocket
- Tick-level data
- All US exchanges"
```

### 4. IEX Cloud
**Free Tier: 50,000 messages/month**
```bash
claude "Setup IEX Cloud for:
- Real-time quotes
- Company fundamentals
- Market statistics"
```

## 📰 News & Sentiment APIs

### 1. NewsAPI
**Free: 100 requests/day**
```bash
claude "Configure NewsAPI:
- Sources: 80,000+ publishers
- Languages: 14
- Historical: 1 month"
```

### 2. Bloomberg API
**Professional License Required**
```python
# Terminal users only
from bloomberg import blpapi
# Real-time news, data, analytics
```

### 3. Twitter/X API
**Free Tier Limited**
```bash
claude "Setup Twitter sentiment:
- Track keywords
- User sentiment
- Trend analysis"
```

### 4. Reddit API
**Free with Rate Limits**
```bash
claude "Monitor Reddit for:
- r/wallstreetbets
- r/stocks
- Sentiment scoring"
```

## 💹 Trading Execution APIs

### 1. Alpaca
**Free Paper Trading**
```bash
claude "Configure Alpaca:
- Paper trading: Unlimited
- Live trading: $0 commissions
- Crypto: 24/7 trading"
```

Features:
- REST & WebSocket
- Fractional shares
- Extended hours
- Crypto trading

### 2. Interactive Brokers
**Professional Platform**
```python
from ib_insync import IB
ib = IB()
ib.connect('127.0.0.1', 7497)
```

Capabilities:
- Global markets
- All asset classes
- Advanced orders
- Portfolio margin

### 3. TD Ameritrade
**Free API Access**
```bash
claude "Setup TD Ameritrade:
- OAuth authentication
- Real-time streaming
- Options trading"
```

### 4. Robinhood (Unofficial)
```python
# Via robin_stocks library
import robin_stocks as rs
rs.login(username, password)
```

## 🪙 Cryptocurrency APIs

### 1. Binance
**Largest Crypto Exchange**
```bash
claude "Connect to Binance:
- Spot trading
- Futures
- 1000+ pairs
- 0.1% fees"
```

### 2. Coinbase Pro
**US Regulated**
```bash
claude "Setup Coinbase Pro:
- USD pairs
- Institutional grade
- 0.5% fees"
```

### 3. Kraken
**Advanced Features**
```python
import krakenex
api = krakenex.API()
api.load_key('kraken.key')
```

## 🎲 Prediction & Betting APIs

### 1. Polymarket
**Prediction Markets**
```bash
claude "Connect to Polymarket:
- Political events
- Sports outcomes
- Economic indicators"
```

### 2. DraftKings
**Sports Betting**
```bash
claude "Access DraftKings odds:
- Live lines
- Player props
- Arbitrage detection"
```

### 3. FanDuel
**Sports Betting**
```bash
claude "Monitor FanDuel:
- Odds comparison
- Line movements
- Best prices"
```

## 🤖 AI/ML APIs

### 1. OpenRouter
**Multiple LLMs**
```bash
claude "Configure OpenRouter:
- GPT-4, Claude, Llama
- Sentiment analysis
- Trade reasoning"
```

### 2. Anthropic Claude
**Advanced AI**
```python
from anthropic import Anthropic
client = Anthropic(api_key="...")
```

### 3. OpenAI
**GPT Models**
```bash
claude "Setup OpenAI:
- GPT-4 for analysis
- Embeddings for similarity
- Fine-tuning available"
```

## 📈 Technical Analysis APIs

### 1. TradingView
**Charting & Indicators**
```bash
claude "Connect TradingView:
- Pine Script strategies
- 100+ indicators
- Chart patterns"
```

### 2. TA-Lib
**Technical Indicators**
```python
import talib
rsi = talib.RSI(prices, timeperiod=14)
```

## 🏦 Alternative Data APIs

### 1. Quandl
**Financial & Alternative Data**
```bash
claude "Access Quandl datasets:
- Economic data
- Futures
- Alternative data"
```

### 2. Federal Reserve (FRED)
**Economic Data - Free**
```bash
claude "Connect to FRED:
- Interest rates
- GDP, inflation
- 800,000+ series"
```

## 🔗 API Configuration

### Environment Setup
```bash
# .env file
ALPHA_VANTAGE_API_KEY=your_key
NEWS_API_KEY=your_key
ALPACA_API_KEY=your_key
ALPACA_SECRET_KEY=your_secret
OPENROUTER_API_KEY=your_key
POLYGON_API_KEY=your_key
```

### Rate Limit Management
```python
# Built-in rate limiting
from ratelimit import limits, sleep_and_retry

@sleep_and_retry
@limits(calls=5, period=60)  # 5 calls per minute
def call_api():
    return api.get_quote("AAPL")
```

## 🛠 API Integration Examples

### Example 1: Multi-Source Data Aggregation
```bash
claude "Aggregate data from:
- Alpha Vantage: Prices
- NewsAPI: Sentiment
- Reddit: Social sentiment
For symbol: TSLA"
```

### Example 2: Smart Order Routing
```bash
claude "Execute trade with best price:
- Check Alpaca
- Check IBKR
- Check TD Ameritrade
- Route to best"
```

### Example 3: Real-time Pipeline
```python
# WebSocket streaming
async def stream_pipeline():
    # Connect to multiple sources
    alpaca_stream = AlpacaStream()
    polygon_stream = PolygonStream()
    
    # Process in real-time
    async for data in merge_streams():
        signal = process_signal(data)
        if signal.strength > threshold:
            execute_trade(signal)
```

## 📊 API Costs

### Free Tier Limits
| API | Free Tier | Paid Starting |
|-----|-----------|---------------|
| Alpha Vantage | 5 req/min | $50/month |
| NewsAPI | 100 req/day | $449/month |
| Polygon | 5 req/min | $29/month |
| IEX Cloud | 50k msg/mo | $9/month |

### Cost Optimization Tips
1. Cache frequently used data
2. Batch requests when possible
3. Use WebSocket for real-time
4. Implement fallback APIs
5. Monitor usage closely

## 🧪 Testing APIs

### Test Connectivity
```bash
claude "Test all configured APIs:
- Check authentication
- Verify rate limits
- Test data quality
- Measure latency"
```

### API Health Monitor
```bash
claude "Create API monitor:
- Check every 5 minutes
- Alert on failures
- Auto-switch to backup
- Log response times"
```

## ✅ API Checklist

- [ ] Market data source configured
- [ ] News API connected
- [ ] Execution API ready
- [ ] Rate limits understood
- [ ] Error handling implemented
- [ ] Backup APIs configured
- [ ] Costs calculated

## 🎯 Quick Reference

### Most Important APIs
1. **Data**: Alpha Vantage or Yahoo Finance
2. **News**: NewsAPI or Reddit
3. **Execution**: Alpaca (free paper trading)
4. **AI**: OpenRouter or Anthropic
5. **Crypto**: Binance or Coinbase

### Minimal Setup (Free)
```bash
# Only free APIs
claude "Setup free API stack:
- Yahoo Finance (data)
- Reddit API (sentiment)
- Alpaca paper (testing)
Total cost: $0"
```

### Professional Setup
```bash
# Production-ready
claude "Setup pro API stack:
- Polygon (real-time data)
- Multiple news sources
- IBKR (execution)
- OpenRouter (AI)
Estimated: $200-500/month"
```

## ⏭ Next Steps

Ready to optimize your strategies? Continue to [Optimization Strategies](13-optimization-strategies.md)

---

**Progress**: 112 min / 2 hours | [← Previous: Hello World Bot](11-hello-world-bot.md) | [Back to Contents](README.md) | [Next: Optimization →](13-optimization-strategies.md)