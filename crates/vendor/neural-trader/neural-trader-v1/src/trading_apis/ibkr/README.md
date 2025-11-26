# Interactive Brokers Trading API Integration

This module provides a high-performance, low-latency integration with Interactive Brokers TWS and Gateway APIs, optimized for algorithmic and high-frequency trading.

## Features

- **Ultra-Low Latency**: Sub-100ms order submission and data processing
- **Async/Await Support**: Non-blocking operations throughout
- **Connection Resilience**: Automatic reconnection and failover
- **Real-Time Data Streaming**: Optimized market data handling with batching and conflation
- **Load Balancing**: Multiple gateway connections for improved throughput
- **Performance Monitoring**: Built-in latency tracking and statistics

## Components

### IBKRClient (`ibkr_client.py`)
Main client for TWS API interaction with async wrappers around all IB API calls.

**Key Features:**
- Async order placement and management
- Real-time position and account tracking
- Automatic reconnection on disconnection
- Latency metrics for all operations
- Thread-safe operation

**Usage Example:**
```python
from ibkr_client import IBKRClient, ConnectionConfig

# Configure connection
config = ConnectionConfig(
    host="127.0.0.1",
    port=7497,  # Paper trading port
    client_id=1,
    auto_reconnect=True
)

# Create client
client = IBKRClient(config)

# Connect
await client.connect()

# Place order
order_id = await client.place_order(
    symbol="AAPL",
    quantity=100,
    order_type="LMT",
    side="BUY",
    price=150.00
)

# Get positions
positions = await client.get_positions()

# Get latency stats
latency_report = client.get_latency_report()
```

### IBKRGateway (`ibkr_gateway.py`)
Low-level gateway connection manager with support for multiple connection types and load balancing.

**Key Features:**
- Multiple connection modes (Direct, SSL, WebSocket, FIX)
- Connection pooling and load balancing
- Automatic failover to backup gateways
- Optimized socket settings for low latency
- Health monitoring and auto-recovery

**Usage Example:**
```python
from ibkr_gateway import IBKRGateway, GatewayConfig, ConnectionMode

# Configure gateway
config = GatewayConfig(
    primary_host="127.0.0.1",
    primary_port=4001,
    backup_hosts=[("backup1.ib.com", 4001), ("backup2.ib.com", 4001)],
    connection_mode=ConnectionMode.DIRECT,
    load_balance=True,
    max_connections=5
)

# Create gateway manager
gateway = IBKRGateway(config)

# Connect with failover
await gateway.connect()

# Send message
await gateway.send_message(message_bytes)

# Receive message
response = await gateway.receive_message(timeout=1.0)

# Get statistics
stats = await gateway.get_statistics()
```

### IBKRDataStream (`ibkr_data_stream.py`)
High-performance market data streaming handler optimized for low latency.

**Key Features:**
- Sub-millisecond tick processing
- Automatic batching and conflation
- Memory-efficient circular buffers
- Real-time filtering and aggregation
- Support for trades, quotes, and market depth

**Usage Example:**
```python
from ibkr_data_stream import IBKRDataStream, StreamConfig, DataType

# Configure streaming
config = StreamConfig(
    buffer_size=10000,
    batch_size=100,
    batch_timeout_ms=10.0,
    conflation_ms=0,  # No conflation for lowest latency
    use_native_parsing=True
)

# Create stream handler
stream = IBKRDataStream(client, config)

# Define callback
async def on_market_data(data):
    if data['type'] == 'tick':
        print(f"Price update: {data['updates']}")
    elif data['type'] == 'snapshot':
        snapshot = data['snapshot']
        print(f"Bid: {snapshot.bid}, Ask: {snapshot.ask}, Spread: {snapshot.spread}")

# Subscribe to data
await stream.subscribe(
    symbol="AAPL",
    data_types=[DataType.TRADES, DataType.QUOTES, DataType.DEPTH],
    callback=on_market_data
)

# Get current snapshot
snapshot = stream.get_snapshot("AAPL")

# Get statistics
stats = stream.get_statistics()
```

## Performance Optimization

### Latency Targets
- Order submission: < 100ms
- Market data reception: < 1ms
- Tick-to-trade: < 10ms

### Optimization Techniques
1. **TCP_NODELAY**: Disabled Nagle's algorithm for immediate packet transmission
2. **Large Socket Buffers**: 256KB buffers for high-throughput scenarios
3. **Pre-allocated Buffers**: Reduces memory allocation overhead
4. **Native Parsing**: Optimized binary protocol parsing
5. **Async Everything**: Non-blocking operations throughout

### Monitoring
All components include comprehensive latency tracking:
- Order submission latency
- Order fill latency
- Market data latency
- Network round-trip time
- Processing time per batch

## Installation

```bash
# Install required dependencies
pip install ib_insync aiohttp numpy

# For production use, also install:
pip install uvloop  # Faster event loop
pip install orjson  # Faster JSON parsing
```

## Configuration

### Environment Variables
```bash
# TWS/Gateway settings
IB_HOST=127.0.0.1
IB_PORT=7497  # 7496 for live trading
IB_CLIENT_ID=1

# Performance settings
IB_MAX_CONNECTIONS=5
IB_BATCH_SIZE=100
IB_BUFFER_SIZE=10000
```

### Connection Ports
- **TWS Live**: 7496
- **TWS Paper**: 7497
- **Gateway Live**: 4001
- **Gateway Paper**: 4002

## Best Practices

1. **Connection Management**
   - Always use auto-reconnect in production
   - Implement proper error handling for all operations
   - Monitor connection health regularly

2. **Order Management**
   - Use unique order IDs
   - Implement order tracking and reconciliation
   - Handle partial fills appropriately

3. **Market Data**
   - Subscribe only to needed data types
   - Use conflation in high-volume scenarios
   - Monitor buffer usage to prevent overflow

4. **Performance**
   - Pre-warm connections before trading
   - Use connection pooling for high throughput
   - Monitor latency metrics continuously

## Troubleshooting

### Common Issues

1. **Connection Refused**
   - Ensure TWS/Gateway is running
   - Check firewall settings
   - Verify correct port numbers

2. **High Latency**
   - Check network connectivity
   - Verify TCP_NODELAY is enabled
   - Monitor system resources

3. **Data Overflow**
   - Increase buffer sizes
   - Enable conflation
   - Reduce subscription count

### Debug Mode
Enable detailed logging:
```python
import logging
logging.getLogger('ibkr').setLevel(logging.DEBUG)
```

## License

This module is part of the AI News Trader project. See main project license for details.