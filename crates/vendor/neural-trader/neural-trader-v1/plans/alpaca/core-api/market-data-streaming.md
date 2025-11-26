# Alpaca API Market Data Streaming Research

## Overview
Alpaca provides WebSocket-based streaming for real-time market data including trades, quotes, minute bars, and news across stocks, crypto, and options. The streaming service supports both JSON and MessagePack protocols for optimal performance.

## WebSocket Stream Architecture

### Connection URLs
- **Paper Trading**: `wss://paper-api.alpaca.markets/stream`
- **Live Trading**: `wss://api.alpaca.markets/stream`
- **Market Data**: `wss://stream.data.alpaca.markets/v2/{source}` (where source is 'iex' or 'sip')

### Supported Data Types
1. **Trades**: Real-time trade executions
2. **Quotes**: Bid/ask price updates
3. **Minute Bars**: OHLCV data aggregated by minute
4. **Daily Bars**: OHLCV data aggregated by day
5. **News**: Real-time financial news
6. **Account Updates**: Portfolio and order status changes

## Authentication Protocol

### Initial Handshake
```python
import websocket
import json

def on_open(ws):
    # Must authenticate within 10 seconds
    auth_data = {
        "action": "auth",
        "key": "YOUR_API_KEY",
        "secret": "YOUR_SECRET_KEY"
    }
    ws.send(json.dumps(auth_data))

def on_message(ws, message):
    data = json.loads(message)
    if data.get('T') == 'success' and data.get('msg') == 'authenticated':
        # Subscribe to data streams
        subscribe_data = {
            "action": "subscribe",
            "trades": ["AAPL", "TSLA"],
            "quotes": ["AAPL", "TSLA"],
            "bars": ["AAPL", "TSLA"]
        }
        ws.send(json.dumps(subscribe_data))
```

## Python SDK Implementation (2024)

### Basic Live Data Client
```python
from alpaca.data.live.stock import StockDataStream
from alpaca.data.live.crypto import CryptoDataStream

# Stock data streaming
stock_stream = StockDataStream(
    api_key="YOUR_API_KEY",
    secret_key="YOUR_SECRET_KEY",
    feed="iex"  # or "sip" for full market data
)

# Crypto data streaming
crypto_stream = CryptoDataStream(
    api_key="YOUR_API_KEY",
    secret_key="YOUR_SECRET_KEY"
)

# Event handlers
@stock_stream.on_trade("AAPL")
async def on_trade(trade):
    print(f"Trade: {trade.symbol} - Price: {trade.price}, Size: {trade.size}")

@stock_stream.on_quote("AAPL")
async def on_quote(quote):
    print(f"Quote: {quote.symbol} - Bid: {quote.bid_price}, Ask: {quote.ask_price}")

@stock_stream.on_bar("AAPL")
async def on_bar(bar):
    print(f"Bar: {bar.symbol} - OHLCV: {bar.open}, {bar.high}, {bar.low}, {bar.close}, {bar.volume}")

# Start streaming
stock_stream.subscribe_trades("AAPL", "TSLA", "GOOGL")
stock_stream.subscribe_quotes("AAPL", "TSLA", "GOOGL")
stock_stream.subscribe_bars("AAPL", "TSLA", "GOOGL")

stock_stream.run()
```

### Advanced Streaming with Multiple Assets
```python
import asyncio
from alpaca.data.live.stock import StockDataStream

class AdvancedStreamHandler:
    def __init__(self):
        self.stream = StockDataStream(
            api_key="YOUR_API_KEY",
            secret_key="YOUR_SECRET_KEY",
            feed="sip"  # Full market data
        )
        self.setup_handlers()

    def setup_handlers(self):
        # Trade handlers
        @self.stream.on_trade("*")  # All symbols
        async def trade_handler(trade):
            await self.process_trade(trade)

        # Quote handlers
        @self.stream.on_quote("*")
        async def quote_handler(quote):
            await self.process_quote(quote)

        # Bar handlers
        @self.stream.on_bar("*")
        async def bar_handler(bar):
            await self.process_bar(bar)

    async def process_trade(self, trade):
        # Custom trade processing logic
        print(f"Processing trade: {trade.symbol} @ {trade.price}")
        # Add to database, trigger algorithms, etc.

    async def process_quote(self, quote):
        # Custom quote processing logic
        spread = quote.ask_price - quote.bid_price
        print(f"Quote update: {quote.symbol} spread: {spread}")

    async def process_bar(self, bar):
        # Custom bar processing logic
        print(f"New bar: {bar.symbol} volume: {bar.volume}")

    def start_streaming(self, symbols):
        self.stream.subscribe_trades(*symbols)
        self.stream.subscribe_quotes(*symbols)
        self.stream.subscribe_bars(*symbols)
        self.stream.run()

# Usage
handler = AdvancedStreamHandler()
symbols = ["AAPL", "TSLA", "GOOGL", "MSFT", "AMZN"]
handler.start_streaming(symbols)
```

## Crypto Streaming Implementation

### Crypto-Specific Features
```python
from alpaca.data.live.crypto import CryptoDataStream

crypto_stream = CryptoDataStream(
    api_key="YOUR_API_KEY",
    secret_key="YOUR_SECRET_KEY"
)

@crypto_stream.on_trade("BTC/USD")
async def crypto_trade_handler(trade):
    print(f"Crypto Trade: {trade.symbol} - Price: {trade.price}")

@crypto_stream.on_quote("BTC/USD", "ETH/USD")
async def crypto_quote_handler(quote):
    print(f"Crypto Quote: {quote.symbol} - Bid: {quote.bid_price}")

@crypto_stream.on_bar("BTC/USD")
async def crypto_bar_handler(bar):
    print(f"Crypto Bar: {bar.symbol} - Volume: {bar.volume}")

# 24/7 crypto streaming
crypto_stream.subscribe_trades("BTC/USD", "ETH/USD", "LTC/USD")
crypto_stream.run()
```

## Performance Optimization

### Connection Management
```python
import websocket
import threading
import time

class OptimizedStreamManager:
    def __init__(self):
        self.connection = None
        self.is_connected = False
        self.heartbeat_interval = 30
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5

    def create_connection(self):
        self.connection = websocket.WebSocketApp(
            "wss://stream.data.alpaca.markets/v2/iex",
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close
        )

    def on_open(self, ws):
        print("WebSocket connection opened")
        self.is_connected = True
        self.reconnect_attempts = 0
        self.authenticate()
        self.start_heartbeat()

    def on_message(self, ws, message):
        # Process incoming messages
        data = json.loads(message)
        self.handle_message(data)

    def on_error(self, ws, error):
        print(f"WebSocket error: {error}")
        self.is_connected = False

    def on_close(self, ws, close_status_code, close_msg):
        print("WebSocket connection closed")
        self.is_connected = False
        self.attempt_reconnect()

    def start_heartbeat(self):
        def heartbeat():
            while self.is_connected:
                time.sleep(self.heartbeat_interval)
                if self.is_connected:
                    self.connection.send(json.dumps({"action": "ping"}))

        threading.Thread(target=heartbeat, daemon=True).start()

    def attempt_reconnect(self):
        if self.reconnect_attempts < self.max_reconnect_attempts:
            self.reconnect_attempts += 1
            print(f"Attempting reconnect #{self.reconnect_attempts}")
            time.sleep(2 ** self.reconnect_attempts)  # Exponential backoff
            self.create_connection()
            self.connection.run_forever()
```

### Message Filtering and Processing
```python
class StreamProcessor:
    def __init__(self):
        self.subscribed_symbols = set()
        self.message_handlers = {
            't': self.handle_trade,
            'q': self.handle_quote,
            'b': self.handle_bar,
            'n': self.handle_news
        }

    def handle_message(self, data):
        if isinstance(data, list):
            for message in data:
                self.process_single_message(message)
        else:
            self.process_single_message(data)

    def process_single_message(self, message):
        msg_type = message.get('T')
        if msg_type in self.message_handlers:
            symbol = message.get('S')
            if symbol in self.subscribed_symbols:
                self.message_handlers[msg_type](message)

    def handle_trade(self, trade_data):
        # Process trade data
        symbol = trade_data['S']
        price = trade_data['p']
        size = trade_data['s']
        timestamp = trade_data['t']
        # Custom processing logic

    def handle_quote(self, quote_data):
        # Process quote data
        symbol = quote_data['S']
        bid_price = quote_data['bp']
        ask_price = quote_data['ap']
        # Custom processing logic

    def handle_bar(self, bar_data):
        # Process bar data
        symbol = bar_data['S']
        open_price = bar_data['o']
        high_price = bar_data['h']
        low_price = bar_data['l']
        close_price = bar_data['c']
        volume = bar_data['v']
        # Custom processing logic
```

## Real-Time Trading Integration

### Order Execution Based on Streaming Data
```python
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

class StreamBasedTrader:
    def __init__(self):
        self.trading_client = TradingClient(
            api_key="YOUR_API_KEY",
            secret_key="YOUR_SECRET_KEY",
            paper=True
        )

        self.stream = StockDataStream(
            api_key="YOUR_API_KEY",
            secret_key="YOUR_SECRET_KEY"
        )

        self.setup_stream_handlers()

    def setup_stream_handlers(self):
        @self.stream.on_trade("AAPL")
        async def trade_signal(trade):
            if self.should_buy(trade):
                await self.place_buy_order(trade.symbol, 10)
            elif self.should_sell(trade):
                await self.place_sell_order(trade.symbol, 10)

    def should_buy(self, trade):
        # Implement buy signal logic
        return trade.price > 150  # Example condition

    def should_sell(self, trade):
        # Implement sell signal logic
        return trade.price < 140  # Example condition

    async def place_buy_order(self, symbol, qty):
        order_request = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY
        )

        try:
            order = self.trading_client.submit_order(order_request)
            print(f"Buy order placed: {order.id}")
        except Exception as e:
            print(f"Error placing buy order: {e}")

    async def place_sell_order(self, symbol, qty):
        order_request = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY
        )

        try:
            order = self.trading_client.submit_order(order_request)
            print(f"Sell order placed: {order.id}")
        except Exception as e:
            print(f"Error placing sell order: {e}")
```

## Data Sources and Pricing

### IEX vs SIP Data
- **IEX**: Free for basic use, includes major exchanges
- **SIP**: Full market data, requires subscription, includes all venues
- **Latency**: SIP provides lower latency and more comprehensive data

### Data Feeds Available
1. **IEX**: Basic real-time data from IEX exchange
2. **SIP**: Securities Information Processor - full market data
3. **Crypto**: Real-time crypto data (included with trading account)

## Best Practices for Streaming

### Connection Reliability
- Implement automatic reconnection with exponential backoff
- Use heartbeat/ping messages to maintain connection
- Handle connection timeouts gracefully
- Store API credentials securely

### Performance Optimization
- Filter messages at the source (subscribe only to needed symbols)
- Use efficient message processing (avoid blocking operations)
- Implement proper error handling and logging
- Consider using multiple connections for different data types

### Memory Management
- Limit message queue sizes
- Implement data retention policies
- Use efficient data structures for high-frequency updates
- Monitor memory usage in long-running applications

## Integration with Trading Strategies

### Event-Driven Architecture
```python
import asyncio
from collections import deque

class EventDrivenStrategy:
    def __init__(self):
        self.price_history = deque(maxlen=100)
        self.order_queue = asyncio.Queue()
        self.stream = StockDataStream(api_key="KEY", secret_key="SECRET")

    @self.stream.on_trade("AAPL")
    async def on_price_update(self, trade):
        self.price_history.append(trade.price)
        signal = self.generate_signal()
        if signal:
            await self.order_queue.put(signal)

    def generate_signal(self):
        if len(self.price_history) >= 20:
            ma_20 = sum(list(self.price_history)[-20:]) / 20
            current_price = self.price_history[-1]

            if current_price > ma_20 * 1.02:  # 2% above MA
                return {"action": "buy", "symbol": "AAPL", "qty": 10}
            elif current_price < ma_20 * 0.98:  # 2% below MA
                return {"action": "sell", "symbol": "AAPL", "qty": 10}
        return None

    async def process_orders(self):
        while True:
            signal = await self.order_queue.get()
            # Execute order based on signal
            print(f"Processing signal: {signal}")
```

## Error Handling and Monitoring

### Connection Monitoring
- Track connection status and uptime
- Monitor message latency and processing time
- Implement alerting for connection failures
- Log all error conditions and recovery actions

### Data Quality Checks
- Validate message formats and data integrity
- Check for missing or delayed data
- Implement data consistency checks
- Monitor for abnormal trading patterns