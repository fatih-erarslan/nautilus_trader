# Low-Latency Trading APIs Integration

This documentation covers the implementation of high-performance, low-latency trading APIs using Polygon.io for market data and Alpaca for trade execution.

## Overview

The trading system is designed for minimal latency with the following features:
- Sub-millisecond market data processing
- Optimized WebSocket connections
- CPU affinity and priority scheduling
- Lock-free data structures
- Memory-mapped buffers
- Hardware-optimized network stack

## Architecture

### Component Overview

```
┌─────────────────────┐     ┌──────────────────────┐
│   Polygon.io API    │     │     Alpaca API       │
│  (Market Data)      │     │  (Trade Execution)   │
└──────────┬──────────┘     └──────────┬───────────┘
           │                           │
           ▼                           ▼
┌─────────────────────────────────────────────────┐
│            Low-Latency Gateway                   │
│  ┌─────────────┐    ┌─────────────┐            │
│  │  WebSocket  │    │   REST API  │            │
│  │  Handler    │    │   Handler   │            │
│  └─────────────┘    └─────────────┘            │
└──────────────────────┬──────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────┐
│           Message Processing Engine              │
│  ┌─────────────┐    ┌─────────────┐            │
│  │Lock-Free    │    │  Strategy   │            │
│  │Queue        │    │  Engine     │            │
│  └─────────────┘    └─────────────┘            │
└─────────────────────────────────────────────────┘
```

## API Credentials Setup

### Polygon.io
1. Sign up at https://polygon.io
2. Get your API key from the dashboard
3. Add to `.env` file:
   ```
   POLYGON_API_KEY=your_polygon_api_key_here
   ```

### Alpaca
1. Sign up at https://alpaca.markets
2. Get your API credentials from the dashboard
3. Add to `.env` file:
   ```
   ALPACA_API_KEY=your_alpaca_key_here
   ALPACA_API_SECRET=your_alpaca_secret_here
   ```

## Installation

### Using Docker (Recommended)

```bash
# Build and run with Docker Compose
docker-compose up -d

# Or run with optimization script
sudo ./run_low_latency.sh
```

### Manual Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run with optimizations
sudo ./run_low_latency.sh
```

## Configuration

All configuration is managed through `config/trading_apis.yaml`:

### Key Settings

- **WebSocket Compression**: Disabled for lower latency
- **CPU Affinity**: Polygon on cores 0-1, Alpaca on cores 2-3
- **Process Priority**: Set to -20 (highest)
- **Network Buffers**: 64KB for optimal throughput
- **Message Protocol**: MessagePack for binary efficiency

## Usage Examples

### Market Data Streaming (Polygon)

```python
from src.market_data.polygon_client import PolygonClient

# Initialize client
client = PolygonClient()

# Subscribe to real-time data
async def on_quote(quote):
    print(f"Quote: {quote.symbol} @ {quote.price}")

await client.subscribe_quotes(['AAPL', 'GOOGL'], on_quote)
```

### Trade Execution (Alpaca)

```python
from src.trading.alpaca_client import AlpacaClient

# Initialize client
client = AlpacaClient()

# Place a limit order
order = await client.place_order(
    symbol='AAPL',
    qty=100,
    side='buy',
    order_type='limit',
    limit_price=150.00
)
```

## Latency Optimization Techniques

### 1. CPU Optimization
- **CPU Affinity**: Pin processes to specific cores
- **Frequency Scaling**: Disable to maintain peak performance
- **C-States**: Disable CPU idle states
- **Priority**: Real-time scheduling with SCHED_FIFO

### 2. Memory Optimization
- **Huge Pages**: Use 2MB pages for reduced TLB misses
- **NUMA Awareness**: Allocate memory on same NUMA node
- **Pre-allocation**: Reserve memory upfront
- **Lock Pages**: Prevent swapping with mlockall()

### 3. Network Optimization
- **TCP_NODELAY**: Disable Nagle's algorithm
- **SO_REUSEPORT**: Multiple threads on same port
- **Interrupt Coalescing**: Reduce interrupt overhead
- **Kernel Bypass**: Consider DPDK for ultra-low latency

### 4. Code Optimization
- **Lock-Free Structures**: SPSC/MPSC queues
- **Branch Prediction**: Optimize hot paths
- **SIMD Instructions**: Vectorized operations
- **JIT Compilation**: Numba for numerical code

## Performance Metrics

### Target Latencies
- Market Data Processing: < 100 microseconds
- Order Submission: < 10 milliseconds
- Quote-to-Trade: < 1 millisecond

### Monitoring
Access metrics at http://localhost:9090/metrics

Key metrics:
- `trading_latency_microseconds` - End-to-end latency
- `market_data_rate_per_second` - Message throughput
- `order_success_rate` - Execution success rate

## Troubleshooting

### High Latency Issues
1. Check CPU governor: `cpupower frequency-info`
2. Verify process priority: `ps -eo pid,nice,pri,cmd | grep python`
3. Monitor network buffers: `ss -tm`

### Connection Issues
1. Verify API credentials in `.env`
2. Check firewall rules for WebSocket ports
3. Test connectivity: `wscat -c wss://socket.polygon.io`

### Memory Issues
1. Check huge pages: `cat /proc/meminfo | grep Huge`
2. Monitor memory usage: `free -h`
3. Review GC metrics in logs

## Security Considerations

1. **API Keys**: Never commit credentials to version control
2. **Network**: Use VPN or direct connect for production
3. **Access Control**: Implement IP whitelisting
4. **Audit Logs**: Monitor all trading activity

## Testing

Run the test suite:
```bash
# Unit tests
pytest tests/unit/

# Integration tests (requires API keys)
pytest tests/integration/

# Performance benchmarks
pytest tests/performance/ -v
```

## Production Deployment

### Recommended Hardware
- CPU: Intel Xeon with high single-thread performance
- RAM: 32GB minimum, ECC preferred
- Network: 10Gbps NIC with kernel bypass support
- Storage: NVMe SSD for logs and data

### Deployment Checklist
- [ ] Configure CPU isolation in kernel boot parameters
- [ ] Enable huge pages support
- [ ] Disable unnecessary services
- [ ] Configure network interrupts affinity
- [ ] Set up monitoring and alerting
- [ ] Implement circuit breakers
- [ ] Test failover procedures

## Support

For issues and questions:
- Polygon.io Support: https://polygon.io/support
- Alpaca Support: https://alpaca.markets/support
- Project Issues: Create an issue in this repository

## License

This implementation is provided under the MIT License. See LICENSE file for details.