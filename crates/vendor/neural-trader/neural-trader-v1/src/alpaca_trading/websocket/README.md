# Alpaca WebSocket Trading Client

A high-performance WebSocket client for real-time market data streaming from Alpaca Markets.

## Features

- **Real-time Market Data**: Stream trades, quotes, bars, and more from Alpaca's SIP feed
- **MessagePack Encoding**: Efficient binary protocol for low-latency data transfer
- **Automatic Reconnection**: Exponential backoff strategy for connection resilience
- **Connection Pooling**: Manage multiple connections for high-throughput scenarios
- **Async Processing Pipeline**: Multi-worker architecture for parallel message processing
- **Comprehensive Metrics**: Latency tracking, throughput monitoring, and error rates
- **Flexible Subscription Management**: Batch operations and symbol routing

## Components

### 1. AlpacaWebSocketClient (`alpaca_client.py`)
Core WebSocket client with:
- Authentication handling
- Message encoding/decoding (MessagePack and JSON)
- Automatic reconnection with exponential backoff
- Heartbeat monitoring
- Subscription management

### 2. StreamManager (`stream_manager.py`)
Manages subscriptions with:
- Batch subscription operations
- Symbol-specific routing
- Rate limiting for API calls
- Dynamic subscription updates

### 3. MessageHandler (`message_handler.py`)
Async message processing with:
- Multi-worker processing pipeline
- Message type routing
- Performance metrics
- Error handling and recovery
- Configurable buffering and batching

### 4. ConnectionPool (`connection_pool.py`)
Connection management with:
- Multiple connection support
- Health monitoring
- Automatic failover
- Load balancing
- Connection rotation

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
import asyncio
import os
from alpaca_trading.websocket import AlpacaWebSocketClient

async def main():
    # Initialize client
    client = AlpacaWebSocketClient(
        api_key=os.getenv("ALPACA_API_KEY"),
        api_secret=os.getenv("ALPACA_API_SECRET"),
        stream_type="data",
        feed="sip"
    )
    
    # Register handlers
    async def handle_trade(message):
        print(f"Trade: {message['S']} @ ${message['p']}")
    
    client.register_handler("t", handle_trade)
    
    # Connect and subscribe
    await client.connect()
    await client.subscribe(trades=["AAPL", "MSFT", "GOOGL"])
    
    # Stream for 60 seconds
    await asyncio.sleep(60)
    
    # Disconnect
    await client.disconnect()

asyncio.run(main())
```

## Advanced Usage

### Using Stream Manager

```python
from alpaca_trading.websocket import AlpacaWebSocketClient, StreamManager

# Create client and manager
client = AlpacaWebSocketClient(api_key, api_secret)
stream_manager = StreamManager(client)

# Start streaming
await client.connect()
stream_manager.start()

# Subscribe with specific handler
async def handle_tech_stocks(message):
    print(f"Tech update: {message}")

await stream_manager.subscribe(
    symbols=["AAPL", "MSFT", "GOOGL"],
    data_types=["trades", "quotes"],
    handler=handle_tech_stocks
)
```

### Using Message Handler

```python
from alpaca_trading.websocket import MessageHandler

# Create handler with parallel workers
handler = MessageHandler(
    buffer_size=10000,
    batch_size=100,
    worker_count=4
)

# Register processors
async def process_trades(trades):
    for trade in trades:
        print(f"{trade.symbol}: ${trade.price}")

handler.register_processor("trades", process_trades)
handler.start()
```

### Using Connection Pool

```python
from alpaca_trading.websocket import ConnectionPool

# Create pool for high-throughput
pool = ConnectionPool(
    api_key=api_key,
    api_secret=api_secret,
    pool_size=3,
    max_subscriptions_per_connection=200
)

await pool.start()

# Get connections for different symbol groups
conn1 = await pool.get_connection(["AAPL", "MSFT"])
conn2 = await pool.get_connection(["SPY", "QQQ"])
```

## Configuration

### Environment Variables

```bash
export ALPACA_API_KEY="your-api-key"
export ALPACA_API_SECRET="your-api-secret"
```

### Client Options

- `stream_type`: "data" for market data, "trading" for account updates
- `feed`: "sip" (paid), "iex" (free), or "otc"
- `raw_data`: Use JSON instead of MessagePack
- `max_reconnect_attempts`: Maximum reconnection attempts
- `heartbeat_interval`: Connection health check interval

### Performance Tuning

- **Buffer Size**: Increase for high-volume streams
- **Batch Size**: Larger batches reduce overhead but increase latency
- **Worker Count**: More workers for CPU-intensive processing
- **Connection Pool Size**: Balance between redundancy and resource usage

## Metrics

Get real-time performance metrics:

```python
# Client metrics
client_metrics = client.get_metrics()
print(f"Average latency: {client_metrics['avg_latency_ms']}ms")

# Handler metrics
handler_metrics = handler.get_metrics()
print(f"Messages/sec: {handler_metrics['by_type']['trades']['rate']}")

# Pool metrics
pool_metrics = pool.get_pool_metrics()
print(f"Healthy connections: {pool_metrics['healthy_connections']}")
```

## Error Handling

```python
# Set error handler
async def handle_error(error):
    logger.error(f"WebSocket error: {error}")
    # Implement recovery logic

client.set_error_handler(handle_error)
handler.set_error_handler(handle_error)
```

## Best Practices

1. **Use Connection Pooling** for production environments
2. **Implement Error Handlers** for robust operation
3. **Monitor Metrics** to detect issues early
4. **Batch Subscriptions** to reduce API calls
5. **Use Appropriate Buffer Sizes** based on data volume
6. **Implement Circuit Breakers** for downstream systems

## Testing

```bash
# Run unit tests
pytest tests/

# Run with coverage
pytest --cov=alpaca_trading.websocket tests/

# Run integration tests (requires API credentials)
pytest tests/integration/ --integration
```

## Troubleshooting

### Connection Issues
- Check API credentials
- Verify network connectivity
- Check Alpaca service status

### High Latency
- Use SIP feed for lowest latency
- Increase worker count
- Optimize message processors

### Memory Usage
- Reduce buffer sizes
- Implement message filtering
- Use connection pooling

## License

See main project LICENSE file.