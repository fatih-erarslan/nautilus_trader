# Trading API Base Infrastructure

Ultra-low latency connection management and monitoring infrastructure for trading APIs.

## Overview

This base infrastructure provides the foundation for building high-performance trading systems with:

- **Microsecond-precision latency monitoring**
- **Persistent connection pooling** with automatic failover
- **Standardized API interface** for all trading providers
- **Configuration management** with hot reloading and encryption
- **Circuit breaker pattern** for resilience
- **Health monitoring** and alerting

## Components

### 1. API Interface (`api_interface.py`)

Abstract base class providing a standardized interface for all trading APIs:

```python
from trading_apis.base import TradingAPIInterface, OrderRequest

class MyTradingAPI(TradingAPIInterface):
    async def place_order(self, order: OrderRequest) -> OrderResponse:
        # Implementation here
        pass
```

**Key Features:**
- Standardized order and market data structures
- Built-in latency tracking
- Validation and error handling
- Callback system for events
- Context manager support

### 2. Connection Pool (`connection_pool.py`)

High-performance connection pooling with intelligent load balancing:

```python
from trading_apis.base import ConnectionPool

pool = ConnectionPool(
    api_class=MyTradingAPI,
    config=api_config,
    min_connections=3,
    max_connections=10
)

await pool.initialize()

# Execute operations using pooled connections
result = await pool.execute_with_connection(
    lambda api: api.place_order(order)
)
```

**Key Features:**
- Health-based connection routing
- Automatic connection recovery
- Circuit breaker protection
- Real-time metrics
- Connection warming

### 3. Latency Monitor (`latency_monitor.py`)

Microsecond-precision latency monitoring and alerting:

```python
from trading_apis.base import LatencyMonitor

monitor = LatencyMonitor(
    alert_thresholds={
        'order_placement': 5.0,  # 5ms threshold
        'market_data': 2.0       # 2ms threshold
    }
)

# Measure operation latency
measurement = monitor.measure('place_order')
# ... perform operation ...
measurement.stop()
monitor.record(measurement)
```

**Key Features:**
- Microsecond precision timing
- Statistical analysis (percentiles, trends)
- Threshold-based alerting
- Trend detection
- Export capabilities

### 4. Configuration Loader (`config_loader.py`)

Advanced configuration management with encryption and hot reloading:

```python
from trading_apis.base import ConfigLoader

loader = ConfigLoader(
    config_path="config/trading_apis.yaml",
    enable_hot_reload=True
)

config = await loader.load_config()
```

**Key Features:**
- YAML/JSON configuration files
- Environment variable interpolation
- Encrypted secrets support
- Hot reloading
- Validation

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Set up configuration
cp config/trading_apis.yaml.example config/trading_apis.yaml
```

## Configuration

### Environment Variables

```bash
# API credentials
export ALPACA_API_KEY="your_key_here"
export ALPACA_SECRET_KEY="your_secret_here"

# Connection settings
export TRADING_CONNECTION_MIN_CONNECTIONS=3
export TRADING_CONNECTION_MAX_CONNECTIONS=10
```

### Configuration File

```yaml
# trading_apis.yaml
global_settings:
  timezone: "America/New_York"
  enable_paper_trading: false
  default_timeout_ms: 5000

connection_pool:
  min_connections: 3
  max_connections: 10
  health_check_interval: 30

apis:
  alpaca:
    provider: "alpaca"
    enabled: true
    credentials:
      api_key: "${ALPACA_API_KEY}"
      secret_key: "${ALPACA_SECRET_KEY}"
    settings:
      use_paper_trading: false
```

## Usage Examples

### Basic Usage

```python
import asyncio
from trading_apis.base import ConfigLoader, ConnectionPool, LatencyMonitor
from my_trading_api import MyTradingAPI

async def main():
    # Load configuration
    config_loader = ConfigLoader("config/trading_apis.yaml")
    config = await config_loader.load_config()
    
    # Initialize monitoring
    monitor = LatencyMonitor()
    await monitor.start_monitoring()
    
    # Create connection pool
    pool = ConnectionPool(
        api_class=MyTradingAPI,
        config=config.apis['alpaca'].dict(),
        min_connections=3,
        max_connections=10
    )
    await pool.initialize()
    
    # Place an order
    order = OrderRequest(
        symbol="AAPL",
        quantity=100,
        side="buy",
        order_type="limit",
        price=150.00
    )
    
    # Execute with monitoring
    measurement = monitor.measure('place_order')
    try:
        response = await pool.execute_with_connection(
            lambda api: api.place_order(order)
        )
        measurement.stop()
        monitor.record(measurement)
        print(f"Order placed: {response.order_id}")
    except Exception as e:
        measurement.stop()
        monitor.record(measurement)
        print(f"Error: {e}")
    
    # Cleanup
    await pool.shutdown()
    await monitor.stop_monitoring()

if __name__ == "__main__":
    asyncio.run(main())
```

### Advanced Usage with Error Handling

```python
async def robust_trading_session():
    """Example of robust trading session with full error handling"""
    
    # Initialize components
    config_loader = ConfigLoader("config/trading_apis.yaml")
    config = await config_loader.load_config()
    
    monitor = LatencyMonitor()
    monitor.add_alert_callback(handle_latency_alert)
    
    pool = ConnectionPool(
        api_class=MyTradingAPI,
        config=config.apis['alpaca'].dict()
    )
    pool.set_error_handler(handle_connection_error)
    
    try:
        await pool.initialize()
        await monitor.start_monitoring()
        
        # Trading operations...
        
    except Exception as e:
        print(f"Session error: {e}")
    finally:
        await pool.shutdown()
        await monitor.stop_monitoring()

async def handle_latency_alert(alert):
    """Handle latency alerts"""
    if alert.level == LatencyLevel.CRITICAL:
        # Switch to backup connection
        pass
    elif alert.level == LatencyLevel.WARNING:
        # Log warning
        pass

async def handle_connection_error(error, metrics):
    """Handle connection errors"""
    print(f"Connection error: {error}")
    # Implement retry logic, failover, etc.
```

## Performance Optimization

### Connection Pool Tuning

```python
# High-frequency trading setup
pool = ConnectionPool(
    api_class=MyTradingAPI,
    config=config,
    min_connections=5,      # Keep more connections warm
    max_connections=20,     # Allow more concurrent operations
    health_check_interval=10  # Check health more frequently
)
```

### Latency Monitoring Tuning

```python
# Aggressive latency monitoring
monitor = LatencyMonitor(
    alert_thresholds={
        'order_placement': 2.0,  # 2ms threshold
        'market_data': 1.0       # 1ms threshold
    }
)
```

## Monitoring and Metrics

### Pool Statistics

```python
stats = pool.get_pool_stats()
print(f"Active connections: {stats['total_connections']}")
print(f"Requests handled: {stats['total_requests']}")
print(f"Average health: {stats['average_health_score']}")
```

### Latency Statistics

```python
latency_stats = monitor.get_profile_stats()
for operation, stats in latency_stats.items():
    print(f"{operation}: P95={stats['p95_ms']:.2f}ms")
```

### Exporting Metrics

```python
# Export to JSON
monitor.export_metrics("metrics/latency_report.json")

# Get summary report
summary = monitor.get_summary_report()
print(f"Total measurements: {summary['total_measurements']}")
```

## Security

### Encrypted Secrets

```python
# Encrypt a secret
loader = ConfigLoader()
encrypted = loader.encrypt_value("my_secret_key")
print(f"Encrypted: {encrypted}")  # ENC[encrypted_value]
```

### Environment Variables

Always use environment variables for sensitive data:

```yaml
credentials:
  api_key: "${ALPACA_API_KEY}"
  secret_key: "${ALPACA_SECRET_KEY}"
```

## Testing

```bash
# Run tests
pytest tests/

# Run with coverage
pytest --cov=trading_apis tests/

# Run specific test
pytest tests/test_connection_pool.py::test_connection_acquisition
```

## Best Practices

1. **Always use connection pooling** for production systems
2. **Monitor latency continuously** and set appropriate thresholds
3. **Use circuit breakers** to prevent cascade failures
4. **Implement proper error handling** and retry logic
5. **Keep connections warm** with regular health checks
6. **Use encrypted configuration** for sensitive data
7. **Monitor pool health** and connection metrics

## Troubleshooting

### High Latency Issues

1. Check connection pool health
2. Verify network connectivity
3. Review API rate limits
4. Check system resources

### Connection Pool Issues

1. Monitor pool statistics
2. Check health check logs
3. Verify API credentials
4. Review circuit breaker status

### Configuration Issues

1. Validate configuration syntax
2. Check environment variables
3. Verify file permissions
4. Review encryption keys

## Contributing

1. Follow the existing code style
2. Add tests for new features
3. Update documentation
4. Consider performance impact

## License

This infrastructure is part of the AI News Trader project.