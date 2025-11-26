# 📈 Alpaca Trading Integration Guide

## Overview

The AI News Trader platform includes full integration with Alpaca Markets for paper and live trading. The existing Alpaca module provides WebSocket streaming, order execution, and portfolio management capabilities.

## 🔧 Configuration

### Step 1: Set Up Environment Variables

1. Copy the example environment file:
```bash
cp example.env .env
```

2. Edit `.env` and add your Alpaca credentials:
```env
# Alpaca Trading API Configuration
ALPACA_API_ENDPOINT=https://paper-api.alpaca.markets  # For paper trading
ALPACA_API_KEY=your-alpaca-api-key-here
ALPACA_API_SECRET=your-alpaca-api-secret-here
ALPACA_API_VERSION=v2
```

### Step 2: Get Your API Credentials

1. Sign up at [Alpaca Markets](https://alpaca.markets)
2. Navigate to your Paper Trading dashboard
3. Generate API keys from the API section
4. Copy the API Key ID and Secret Key to your `.env` file

⚠️ **Security Notes:**
- Never commit `.env` with real credentials
- The `.env` file is gitignored by default
- For production, use environment variables or secret management systems

## 🚀 Quick Start

### Test Your Configuration
```bash
# Install required dependencies
pip install python-dotenv msgpack websockets

# Test the configuration
python scripts/alpaca_trading.py --test
```

Expected output:
```
✅ Alpaca credentials loaded
📍 Endpoint: https://paper-api.alpaca.markets
📊 API Version: v2
✅ Configuration test successful!
```

### Start Real-Time Trading
```bash
# Start trading with default symbols (SPY, QQQ)
python scripts/alpaca_trading.py

# Trade specific symbols
python scripts/alpaca_trading.py --symbols AAPL MSFT TSLA
```

## 📊 Using the Existing Alpaca Module

### WebSocket Streaming
```python
from src.alpaca_trading.websocket.alpaca_client import AlpacaWebSocketClient
import os

# Initialize client with environment variables
client = AlpacaWebSocketClient(
    api_key=os.getenv('ALPACA_API_KEY'),
    api_secret=os.getenv('ALPACA_API_SECRET'),
    stream_type="data",
    feed="sip"  # Use SIP feed for best quality
)

# Connect and subscribe to symbols
await client.connect()
await client.subscribe(
    trades=["AAPL", "GOOGL"],
    quotes=["AAPL", "GOOGL"],
    bars=["AAPL", "GOOGL"]
)
```

### Order Execution
```python
from src.alpaca_trading.execution.execution_engine import ExecutionEngine

# Initialize execution engine
engine = ExecutionEngine(
    api_key=os.getenv('ALPACA_API_KEY'),
    api_secret=os.getenv('ALPACA_API_SECRET')
)

# Place an order
order = await engine.place_order(
    symbol="AAPL",
    qty=100,
    side="buy",
    order_type="limit",
    limit_price=150.00
)
```

### Stream Manager
```python
from src.alpaca_trading.websocket.stream_manager import StreamManager

# Create stream manager for easier handling
manager = StreamManager(client)

# Set up handlers
manager.on_trade = handle_trades
manager.on_quote = handle_quotes
manager.on_bar = handle_bars

# Subscribe to multiple symbols
await manager.subscribe(
    trades=["SPY", "QQQ"],
    quotes=["SPY", "QQQ"],
    bars=["SPY", "QQQ"]
)
```

## 🔗 Integration with Trading Strategies

### Crypto Momentum Strategy Integration
```python
from src.strategies.crypto_momentum_strategy import CryptoMomentumStrategy
from src.alpaca_trading.websocket.alpaca_client import AlpacaWebSocketClient

class AlpacaMomentumTrader:
    def __init__(self):
        self.strategy = CryptoMomentumStrategy()
        self.alpaca_client = AlpacaWebSocketClient(
            api_key=os.getenv('ALPACA_API_KEY'),
            api_secret=os.getenv('ALPACA_API_SECRET')
        )
    
    async def handle_bar(self, bar):
        # Convert Alpaca bar to strategy format
        signal = self.strategy.generate_signal(
            symbol=bar.symbol,
            price_data=bar_to_dataframe(bar)
        )
        
        if signal and signal.fee_efficiency_ratio > 7:
            # Execute trade via Alpaca
            await self.execute_trade(signal)
```

## 📁 Module Structure

```
src/alpaca_trading/
├── websocket/
│   ├── alpaca_client.py      # Main WebSocket client
│   ├── stream_manager.py     # Stream management
│   ├── message_handler.py    # Message processing
│   ├── connection_pool.py    # Connection pooling
│   └── example_usage.py      # Usage examples
├── execution/
│   ├── execution_engine.py   # Order execution
│   └── example_usage.py      # Execution examples
└── README.md                  # Module documentation
```

## 🎯 Features

### WebSocket Features
- ✅ Real-time data streaming (trades, quotes, bars)
- ✅ Automatic reconnection with exponential backoff
- ✅ MessagePack encoding for efficiency
- ✅ Connection pooling for multiple streams
- ✅ Health checks and heartbeat monitoring
- ✅ Latency measurement

### Execution Features
- ✅ Market, limit, stop, and stop-limit orders
- ✅ Order status tracking
- ✅ Position management
- ✅ Risk controls
- ✅ Paper and live trading support

## 📊 Supported Data Types

### Trade Data
- Symbol
- Price
- Size
- Exchange
- Timestamp
- Conditions

### Quote Data
- Bid price/size
- Ask price/size
- Bid/Ask exchange
- Timestamp

### Bar Data
- Open, High, Low, Close
- Volume
- VWAP
- Trade count
- Timestamp

## 🛠️ Advanced Usage

### Connection Pooling
```python
from src.alpaca_trading.websocket.connection_pool import ConnectionPool

# Create connection pool for better performance
pool = ConnectionPool(
    api_key=os.getenv('ALPACA_API_KEY'),
    api_secret=os.getenv('ALPACA_API_SECRET'),
    pool_size=5  # Number of connections
)

# Get connections for different purposes
data_conn = await pool.get_connection(stream_type="data")
trading_conn = await pool.get_connection(stream_type="trading")
```

### Custom Message Handlers
```python
from src.alpaca_trading.websocket.message_handler import MessageHandler

class CustomHandler(MessageHandler):
    async def handle_trade(self, trade):
        # Custom trade processing
        if trade.price > self.threshold:
            await self.alert_high_price(trade)
    
    async def handle_quote(self, quote):
        # Custom quote processing
        spread = quote.ask_price - quote.bid_price
        if spread > self.max_spread:
            await self.alert_wide_spread(quote)
```

## 🔍 Monitoring & Debugging

### Enable Debug Logging
```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('alpaca_trading')
```

### Connection Health Check
```python
# Check connection status
if client.is_connected():
    latency = client.get_latency()
    logger.info(f"Connected with {latency}ms latency")
```

### Error Handling
```python
try:
    await client.connect()
except ConnectionError as e:
    logger.error(f"Failed to connect: {e}")
    # Implement retry logic
```

## 📈 Trading Endpoints

### Paper Trading (Default)
- Endpoint: `https://paper-api.alpaca.markets`
- WebSocket Data: `wss://stream.data.alpaca.markets/v2/sip`
- WebSocket Trading: `wss://paper-trading.alpaca.markets/stream`

### Live Trading
- Endpoint: `https://api.alpaca.markets`
- WebSocket Data: `wss://stream.data.alpaca.markets/v2/sip`
- WebSocket Trading: `wss://api.alpaca.markets/stream`

To switch to live trading, update your `.env`:
```env
ALPACA_API_ENDPOINT=https://api.alpaca.markets  # Live trading
```

## ⚠️ Important Notes

1. **Paper Trading First**: Always test strategies in paper trading before going live
2. **Rate Limits**: Respect Alpaca's rate limits (200 requests/minute for most endpoints)
3. **Market Hours**: US market hours are 9:30 AM - 4:00 PM ET
4. **Extended Hours**: Available 4:00 AM - 9:30 AM and 4:00 PM - 8:00 PM ET
5. **Crypto Trading**: Alpaca supports crypto trading 24/7 for select pairs

## 🔗 Resources

- [Alpaca API Documentation](https://alpaca.markets/docs/api-documentation/)
- [WebSocket Streaming Docs](https://alpaca.markets/docs/api-documentation/api-v2/market-data/streaming/)
- [Order Types Guide](https://alpaca.markets/docs/trading-on-alpaca/orders/)
- [Paper Trading Guide](https://alpaca.markets/docs/trading-on-alpaca/paper-trading/)

## 🆘 Troubleshooting

### Connection Issues
```bash
# Check if credentials are loaded
python -c "import os; print('API Key:', os.getenv('ALPACA_API_KEY', 'Not Set'))"

# Test WebSocket connection
python scripts/alpaca_trading.py --test
```

### Common Errors

| Error | Solution |
|-------|----------|
| "Invalid API credentials" | Check API key and secret in `.env` |
| "Connection refused" | Verify endpoint URL and network connectivity |
| "Rate limit exceeded" | Implement backoff or reduce request frequency |
| "Insufficient permissions" | Check API key permissions in Alpaca dashboard |

## 🎉 Next Steps

1. **Test Your Setup**: Run the test script to verify configuration
2. **Paper Trade**: Start with paper trading to test strategies
3. **Integrate Strategies**: Connect your trading strategies to Alpaca
4. **Monitor Performance**: Use the monitoring tools to track execution
5. **Go Live**: When ready, switch to live trading with real funds

The Alpaca integration provides a complete trading infrastructure for executing your AI-powered trading strategies in real markets!