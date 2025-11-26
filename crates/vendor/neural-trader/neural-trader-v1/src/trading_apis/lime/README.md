# Lime Trading Ultra-Low Latency API

High-performance trading API integration for Lime Trading with sub-10 microsecond latency optimizations.

## Features

### Core Capabilities
- **FIX 4.4 Protocol**: Full implementation with Lime Trading
- **Risk Management**: Microsecond pre-trade risk checks
- **Order Management**: Lock-free order tracking and lifecycle
- **Memory Optimization**: Pre-allocated pools to eliminate GC pauses
- **Performance Monitoring**: Real-time latency and throughput metrics

### Performance Optimizations
- **Zero-Copy Operations**: Pre-allocated message buffers
- **Lock-Free Data Structures**: Atomic operations for order tracking  
- **CPU Affinity**: Pin processes to specific cores
- **Memory Alignment**: Cache-line aligned data structures
- **Hardware Timestamps**: Nanosecond precision timing
- **SIMD Vectorization**: Accelerated portfolio calculations

### Target Performance
- **Order Latency**: < 10μs (microseconds)
- **Risk Checks**: < 1μs 
- **Order Rate**: 1000+ orders/second
- **Memory Allocation**: Zero on hot path
- **GC Pauses**: Eliminated through pooling

## Architecture

```
LimeTradingAPI
├── FIX Client (lime_client.py)
│   ├── QuickFIX integration
│   ├── Message pools
│   └── Hardware timestamps
├── Order Manager (lime_order_manager.py)  
│   ├── Lock-free order book
│   ├── Atomic state transitions
│   └── Pre-allocated objects
├── Risk Engine (lime_risk_engine.py)
│   ├── Vectorized calculations
│   ├── SIMD operations
│   └── GPU acceleration
├── Memory Pools (memory_pool.py)
│   ├── Object pools
│   ├── Buffer management
│   └── NUMA awareness
└── Performance Monitor (performance_monitor.py)
    ├── Latency tracking
    ├── System metrics
    └── Hardware profiling
```

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install QuickFIX (may require compilation)
pip install quickfix

# Optional: Install GPU support
pip install cupy-cuda11x  # or appropriate CUDA version
```

## Configuration

1. **FIX Configuration**: Update `config/lime_fix.cfg` with your credentials:
```ini
[SESSION]
SenderCompID=YOUR_SENDER_ID
TargetCompID=LIME
Username=YOUR_USERNAME
Password=YOUR_PASSWORD
Account=YOUR_ACCOUNT
SocketConnectHost=fix.lime.com
SocketConnectPort=4001
```

2. **Performance Settings**: Configure CPU affinity and memory pools:
```python
config = LimeConfig(
    fix_config_file="config/lime_fix.cfg",
    cpu_core=1,  # Pin to CPU core 1
    use_hardware_timestamps=True,
    enable_memory_pools=True,
    max_orders=100000,
    order_pool_size=10000
)
```

## Usage

### Basic Trading
```python
import asyncio
from src.trading_apis.lime import LimeTradingAPI, LimeConfig

async def main():
    config = LimeConfig(
        fix_config_file="config/lime_fix.cfg",
        sender_comp_id="TRADERCLIENT",
        target_comp_id="LIME",
        host="fix.lime.com",
        port=4001,
        cpu_core=1  # Pin to CPU core for low latency
    )
    
    async with LimeTradingAPI(config).trading_session() as api:
        # Send market order
        success, order_id, error = api.send_order(
            symbol="AAPL",
            side="BUY", 
            quantity=100,
            order_type="MARKET"
        )
        
        if success:
            print(f"Order sent: {order_id}")
            
            # Monitor order status
            order = api.get_order(order_id)
            print(f"Status: {order.status}")
            
        # Get performance metrics
        metrics = api.get_metrics()
        print(f"Avg latency: {metrics.avg_order_latency:.2f}μs")

if __name__ == "__main__":
    asyncio.run(main())
```

### Advanced Risk Management
```python
from src.trading_apis.lime import RiskLimits

# Configure risk limits
risk_limits = RiskLimits(
    max_position_size=100000,
    max_position_value=10_000_000,
    max_single_order_size=10000,
    max_single_order_value=1_000_000,
    max_daily_loss=500_000,
    max_orders_per_second=100,
    max_leverage=4.0
)

config = LimeConfig(
    # ... other settings
    risk_limits=risk_limits
)
```

### Event Handlers
```python
def on_execution(message, order_id, exec_type, status):
    print(f"Execution: {order_id} {exec_type} {status}")
    
def on_order_update(order_id, status):
    print(f"Order update: {order_id} {status}")
    
api.set_execution_handler(on_execution)
api.set_order_update_handler(on_order_update)
```

## Performance Tuning

### System Optimization
```bash
# Set CPU governor to performance
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Disable CPU frequency scaling
echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo

# Set process priority
sudo chrt -f 99 python your_trading_app.py

# Pin to specific CPU cores
taskset -c 1-2 python your_trading_app.py
```

### Memory Optimization
```python
# Configure large memory pools
config = LimeConfig(
    max_orders=100000,          # Pre-allocate for 100k orders
    order_pool_size=10000,      # Message pool size
    enable_memory_pools=True,   # Use object pooling
    use_hardware_timestamps=True # Hardware timing
)
```

### Network Optimization
```bash
# Optimize network stack
echo 'net.core.rmem_max = 134217728' | sudo tee -a /etc/sysctl.conf
echo 'net.core.wmem_max = 134217728' | sudo tee -a /etc/sysctl.conf
echo 'net.ipv4.tcp_rmem = 4096 87380 134217728' | sudo tee -a /etc/sysctl.conf
echo 'net.ipv4.tcp_wmem = 4096 65536 134217728' | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

## Monitoring

### Real-time Metrics
```python
# Get current performance stats
stats = api.get_metrics()
print(f"""
Orders sent: {stats.orders_sent}
Orders filled: {stats.orders_filled}
Avg order latency: {stats.avg_order_latency:.2f}μs
P99 latency: {stats.p99_order_latency:.2f}μs
CPU usage: {stats.cpu_usage:.1f}%
Memory usage: {stats.memory_usage:.1f}%
""")
```

### Latency Analysis
```python
# Get detailed latency breakdown
if api.performance_monitor:
    latency_summary = api.performance_monitor.get_latency_summary()
    print("Latency Distribution (μs):")
    for metric, values in latency_summary.items():
        print(f"{metric}:")
        print(f"  Mean: {values['mean']:.2f}")
        print(f"  P99: {values['p99']:.2f}")
        print(f"  P99.9: {values['p999']:.2f}")
```

### Export Metrics
```python
# Export performance data
if api.performance_monitor:
    api.performance_monitor.export_metrics("performance_data.json")
    
    # Generate report
    report = api.performance_monitor.get_performance_report()
    print(report)
```

## Architecture Details

### Memory Management
- **Object Pools**: Pre-allocated orders, messages, and buffers
- **Zero-Copy**: Direct memory access for message construction
- **NUMA Awareness**: Optimal memory allocation for multi-socket systems
- **Cache Alignment**: Data structures aligned to cache line boundaries

### Lock-Free Design
- **Atomic Operations**: Lock-free counters and state transitions
- **Memory Ordering**: Proper memory barriers for consistency
- **Wait-Free Reads**: Non-blocking data access on hot paths
- **RCU Patterns**: Read-copy-update for complex data structures

### Hardware Optimization
- **CPU Affinity**: Process pinning to avoid context switches
- **Real-time Priority**: SCHED_FIFO scheduling for deterministic latency
- **Hardware Timestamps**: TSC-based timing when available
- **SIMD Instructions**: Vectorized calculations for portfolio risk

### Risk Engine
- **Vectorized Checks**: SIMD operations for parallel validation
- **Pre-computed Limits**: Cached risk parameters
- **Incremental Updates**: Minimal recalculation on position changes
- **GPU Acceleration**: CUDA kernels for complex portfolio math

## Troubleshooting

### Common Issues

1. **High Latency**
   - Check CPU governor settings
   - Verify process is pinned to isolated CPU cores
   - Monitor for GC pauses (should be zero)
   - Check network configuration

2. **Connection Issues**
   - Verify FIX configuration settings
   - Check firewall rules for FIX port
   - Validate credentials with Lime Trading
   - Monitor session logs

3. **Risk Rejections**
   - Review risk limit configuration
   - Check position calculations
   - Verify order parameters
   - Monitor portfolio exposure

### Performance Debugging
```python
# Enable detailed monitoring
config.enable_monitoring = True
config.metrics_interval = 100  # 100ms intervals

# Check system resources
import psutil
print(f"CPU cores: {psutil.cpu_count()}")
print(f"Memory: {psutil.virtual_memory().total / 1e9:.1f} GB")
print(f"CPU frequency: {psutil.cpu_freq().current:.0f} MHz")
```

## Security Considerations

- Store credentials securely (use environment variables)
- Implement proper authentication with Lime Trading
- Use encrypted connections (TLS/SSL)
- Monitor for unauthorized access attempts
- Implement rate limiting and circuit breakers

## License

This implementation is for educational and research purposes. Ensure compliance with Lime Trading's API terms of service and applicable regulations.

## Support

For issues related to:
- **Lime Trading API**: Contact Lime Trading support
- **QuickFIX**: See QuickFIX documentation
- **Performance**: Review system optimization guides
- **Implementation**: Check code comments and examples