# Performance Metrics Specification

## Overview

This document defines the comprehensive performance metrics framework for the AI News Trading platform. The metrics are categorized into four primary domains: Latency, Throughput, Strategy Performance, and Resource Utilization. Each metric includes measurement methodology, acceptance criteria, and optimization targets.

## Metrics Architecture

### Collection Framework

```
┌─────────────────────────────────────────────────────────────────┐
│                      Metrics Collection System                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────────┐  ┌──────────────────┐  ┌───────────────┐ │
│  │ Instrumentation  │  │   Aggregation    │  │   Storage     │ │
│  │   - Decorators   │  │   - Time Series  │  │   - InfluxDB  │ │
│  │   - Hooks        │  │   - Statistics   │  │   - Redis     │ │
│  │   - Tracers      │  │   - Percentiles  │  │   - Files     │ │
│  └────────┬─────────┘  └────────┬──────────┘  └──────┬────────┘│
│           │                     │                      │         │
│  ┌────────┴─────────────────────────────────────────────┘       │
│  │                    Metrics Pipeline                           │
│  │  • Real-time Processing  • Batch Aggregation  • Alerting     │
│  └───────────────────────────────────────────────────────────────┘
└─────────────────────────────────────────────────────────────────┘
```

## 1. Latency Metrics

### 1.1 Signal Generation Latency

**Definition**: Time from market data receipt to trading signal generation

```python
class SignalLatencyMetric:
    """Measure signal generation latency with nanosecond precision"""
    
    def __init__(self):
        self.histogram = HdrHistogram(1, 1000000, 3)  # 1μs to 1s
        self.window = SlidingTimeWindow(60)  # 60-second window
        
    @instrument_latency
    async def measure_signal_generation(self, market_data):
        start = time.perf_counter_ns()
        
        signal = await self.strategy.generate_signal(market_data)
        
        latency_ns = time.perf_counter_ns() - start
        self.histogram.record(latency_ns / 1000)  # Convert to μs
        self.window.add(latency_ns)
        
        return signal
```

**Targets**:
- P50: < 20ms
- P90: < 50ms
- P95: < 75ms
- P99: < 100ms ✓ (Primary KPI)
- P99.9: < 150ms

**Measurement Points**:
1. Data ingestion → Preprocessing
2. Preprocessing → Feature extraction
3. Feature extraction → Model inference
4. Model inference → Signal generation
5. End-to-end total

### 1.2 Order Execution Latency

**Definition**: Time from signal generation to order acknowledgment

```python
class OrderExecutionLatencyMetric:
    """Track order execution latency across multiple venues"""
    
    def __init__(self):
        self.venue_metrics = defaultdict(HdrHistogram)
        self.order_types = defaultdict(HdrHistogram)
        
    async def measure_execution(self, order):
        start = time.perf_counter_ns()
        
        # Submit order
        ack = await self.broker.submit_order(order)
        
        latency_ns = time.perf_counter_ns() - start
        
        # Record by venue and order type
        self.venue_metrics[order.venue].record(latency_ns / 1000)
        self.order_types[order.type].record(latency_ns / 1000)
        
        return ack
```

**Targets**:
- Market Orders: < 30ms (P99)
- Limit Orders: < 40ms (P99)
- Stop Orders: < 45ms (P99)
- Complex Orders: < 50ms (P99) ✓

### 1.3 Data Processing Latency

**Definition**: Time to process incoming market data

```python
class DataProcessingLatencyMetric:
    """Measure data processing pipeline latency"""
    
    stages = [
        'ingestion',
        'validation',
        'normalization',
        'enrichment',
        'distribution'
    ]
    
    def __init__(self):
        self.stage_metrics = {stage: HdrHistogram() for stage in self.stages}
        self.end_to_end = HdrHistogram()
```

**Targets**:
- Tick Data: < 5ms (P95)
- OHLCV: < 10ms (P95)
- News Data: < 20ms (P95) ✓
- Level 2 Data: < 15ms (P95)

### 1.4 End-to-End Latency

**Definition**: Total time from market event to executed order

```python
class EndToEndLatencyMetric:
    """Comprehensive end-to-end latency tracking"""
    
    def __init__(self):
        self.trace_spans = {}
        self.latency_breakdown = defaultdict(list)
        
    async def trace_full_cycle(self, market_event):
        trace_id = generate_trace_id()
        
        with TraceContext(trace_id) as trace:
            # Each component adds its span
            data = await self.process_data(market_event)
            signal = await self.generate_signal(data)
            order = await self.create_order(signal)
            ack = await self.execute_order(order)
            
        return self.analyze_trace(trace)
```

**Targets**:
- Normal Market: < 200ms (P99) ✓
- Volatile Market: < 300ms (P99)
- News Events: < 250ms (P99)

## 2. Throughput Metrics

### 2.1 Signal Generation Throughput

**Definition**: Number of trading signals generated per second

```python
class SignalThroughputMetric:
    """Measure signal generation rate and capacity"""
    
    def __init__(self):
        self.rate_counter = RateCounter(window=1.0)  # 1-second window
        self.burst_detector = BurstDetector(threshold=2.0)
        
    async def measure_throughput(self):
        while self.running:
            current_rate = self.rate_counter.get_rate()
            
            # Detect bursts
            if self.burst_detector.is_burst(current_rate):
                await self.handle_burst(current_rate)
            
            # Record metrics
            self.metrics.record('signals_per_second', current_rate)
            
            await asyncio.sleep(0.1)
```

**Targets**:
- Sustained: 10,000+ signals/second ✓
- Burst: 50,000+ signals/second
- Per Strategy: 1,000+ signals/second

### 2.2 Order Throughput

**Definition**: Number of orders processed per second

```python
class OrderThroughputMetric:
    """Track order processing capacity"""
    
    def __init__(self):
        self.order_counter = AtomicCounter()
        self.venue_counters = defaultdict(AtomicCounter)
        self.order_type_counters = defaultdict(AtomicCounter)
        
    def record_order(self, order):
        self.order_counter.increment()
        self.venue_counters[order.venue].increment()
        self.order_type_counters[order.type].increment()
```

**Targets**:
- Total Orders: 5,000+ orders/second ✓
- Per Venue: 1,000+ orders/second
- Per Strategy: 500+ orders/second

### 2.3 Data Ingestion Rate

**Definition**: Market data points processed per second

```python
class DataIngestionMetric:
    """Monitor data ingestion capacity"""
    
    def __init__(self):
        self.data_counters = {
            'ticks': RateCounter(),
            'quotes': RateCounter(),
            'trades': RateCounter(),
            'news': RateCounter()
        }
        
    async def measure_ingestion(self):
        rates = {}
        for data_type, counter in self.data_counters.items():
            rates[data_type] = counter.get_rate()
        return rates
```

**Targets**:
- Tick Data: 100,000+ points/second ✓
- Quote Data: 50,000+ quotes/second
- Trade Data: 25,000+ trades/second
- News Items: 100+ items/second

### 2.4 Concurrent Operations

**Definition**: Number of parallel operations supported

```python
class ConcurrencyMetric:
    """Track concurrent operation capacity"""
    
    def __init__(self):
        self.active_simulations = AtomicGauge()
        self.active_strategies = AtomicGauge()
        self.active_connections = AtomicGauge()
        
    async def monitor_concurrency(self):
        return {
            'simulations': self.active_simulations.value(),
            'strategies': self.active_strategies.value(),
            'connections': self.active_connections.value()
        }
```

**Targets**:
- Concurrent Simulations: 1,000+ ✓
- Active Strategies: 100+
- Market Connections: 50+

## 3. Strategy Performance Metrics

### 3.1 Risk-Adjusted Returns

```python
class StrategyPerformanceMetrics:
    """Comprehensive strategy performance measurement"""
    
    def calculate_sharpe_ratio(self, returns, risk_free_rate=0.02):
        """Calculate Sharpe ratio with proper annualization"""
        excess_returns = returns - risk_free_rate / 252
        if len(returns) < 2:
            return 0.0
        
        return np.sqrt(252) * (excess_returns.mean() / excess_returns.std())
    
    def calculate_sortino_ratio(self, returns, mar=0.0):
        """Calculate Sortino ratio (downside deviation)"""
        excess_returns = returns - mar / 252
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) < 2:
            return float('inf')
            
        downside_std = np.sqrt(np.mean(downside_returns ** 2))
        return np.sqrt(252) * (excess_returns.mean() / downside_std)
```

**Targets**:
- Sharpe Ratio: > 2.0 ✓
- Sortino Ratio: > 3.0
- Calmar Ratio: > 2.5
- Information Ratio: > 1.5

### 3.2 Win Rate and Profit Factor

```python
class TradingMetrics:
    """Track trade-level performance metrics"""
    
    def calculate_win_rate(self, trades):
        """Calculate percentage of profitable trades"""
        winning_trades = sum(1 for t in trades if t.pnl > 0)
        return winning_trades / len(trades) if trades else 0.0
    
    def calculate_profit_factor(self, trades):
        """Calculate ratio of gross profits to gross losses"""
        gross_profits = sum(t.pnl for t in trades if t.pnl > 0)
        gross_losses = abs(sum(t.pnl for t in trades if t.pnl < 0))
        
        return gross_profits / gross_losses if gross_losses > 0 else float('inf')
```

**Targets**:
- Win Rate: > 60% ✓
- Profit Factor: > 1.5 ✓
- Average Win/Loss Ratio: > 1.2
- Consecutive Wins: Track distribution

### 3.3 Drawdown Analysis

```python
class DrawdownMetrics:
    """Analyze drawdown characteristics"""
    
    def calculate_max_drawdown(self, equity_curve):
        """Calculate maximum peak-to-trough drawdown"""
        cumulative = (1 + equity_curve).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def calculate_drawdown_duration(self, equity_curve):
        """Calculate longest drawdown period"""
        cumulative = (1 + equity_curve).cumprod()
        running_max = cumulative.cummax()
        
        # Find drawdown periods
        in_drawdown = cumulative < running_max
        
        # Calculate consecutive periods
        return self._max_consecutive_true(in_drawdown)
```

**Targets**:
- Maximum Drawdown: < 15% ✓
- Average Drawdown: < 5%
- Drawdown Duration: < 30 days
- Recovery Time: < 15 days

### 3.4 Market Regime Performance

```python
class RegimePerformanceMetrics:
    """Analyze performance across market regimes"""
    
    def segment_by_regime(self, returns, market_data):
        """Segment performance by market regime"""
        regimes = self.identify_regimes(market_data)
        
        regime_performance = {}
        for regime in ['bull', 'bear', 'sideways', 'volatile']:
            mask = regimes == regime
            regime_returns = returns[mask]
            
            regime_performance[regime] = {
                'returns': regime_returns.mean() * 252,
                'volatility': regime_returns.std() * np.sqrt(252),
                'sharpe': self.calculate_sharpe_ratio(regime_returns),
                'max_drawdown': self.calculate_max_drawdown(regime_returns)
            }
            
        return regime_performance
```

**Targets**:
- Bull Market Sharpe: > 2.5
- Bear Market Sharpe: > 1.0
- Sideways Market Sharpe: > 1.5
- High Volatility Sharpe: > 1.8

## 4. Resource Utilization Metrics

### 4.1 CPU Usage

```python
class CPUMetrics:
    """Monitor CPU utilization patterns"""
    
    def __init__(self):
        self.cpu_monitor = CPUMonitor()
        self.process_monitor = ProcessMonitor()
        
    async def collect_cpu_metrics(self):
        return {
            'total_usage': self.cpu_monitor.get_usage(),
            'per_core': self.cpu_monitor.get_per_core_usage(),
            'process_usage': self.process_monitor.get_cpu_percent(),
            'thread_count': self.process_monitor.get_thread_count(),
            'context_switches': self.cpu_monitor.get_context_switches()
        }
```

**Targets**:
- Average CPU: < 60%
- Peak CPU: < 80% ✓
- Per-Core Balance: < 20% variance
- Context Switches: < 10,000/sec

### 4.2 Memory Usage

```python
class MemoryMetrics:
    """Track memory consumption and patterns"""
    
    def __init__(self):
        self.memory_tracker = MemoryTracker()
        self.gc_stats = GCStats()
        
    async def analyze_memory(self):
        return {
            'heap_size': self.memory_tracker.get_heap_size(),
            'resident_set': self.memory_tracker.get_rss(),
            'virtual_memory': self.memory_tracker.get_vms(),
            'gc_collections': self.gc_stats.get_collection_stats(),
            'object_counts': self.memory_tracker.get_object_counts(),
            'memory_leaks': self.memory_tracker.detect_leaks()
        }
```

**Targets**:
- Heap Size: < 2GB per simulation
- Total Memory: < 4GB per simulation ✓
- GC Pause Time: < 10ms (P99)
- Memory Growth: < 1MB/hour

### 4.3 I/O Performance

```python
class IOMetrics:
    """Monitor I/O operations and performance"""
    
    def __init__(self):
        self.disk_monitor = DiskMonitor()
        self.network_monitor = NetworkMonitor()
        
    async def measure_io(self):
        return {
            'disk_read_rate': self.disk_monitor.get_read_rate(),
            'disk_write_rate': self.disk_monitor.get_write_rate(),
            'disk_queue_depth': self.disk_monitor.get_queue_depth(),
            'network_rx_rate': self.network_monitor.get_rx_rate(),
            'network_tx_rate': self.network_monitor.get_tx_rate(),
            'network_latency': self.network_monitor.get_latency()
        }
```

**Targets**:
- Disk Read: < 50MB/s
- Disk Write: < 100MB/s ✓
- Network RX: < 10Mbps ✓
- Network TX: < 5Mbps
- Network Latency: < 1ms

### 4.4 Thread and Connection Pools

```python
class PoolMetrics:
    """Monitor resource pool utilization"""
    
    def __init__(self):
        self.thread_pool_monitor = ThreadPoolMonitor()
        self.connection_pool_monitor = ConnectionPoolMonitor()
        
    def get_pool_stats(self):
        return {
            'thread_pool': {
                'active': self.thread_pool_monitor.active_threads,
                'idle': self.thread_pool_monitor.idle_threads,
                'queue_size': self.thread_pool_monitor.queue_size,
                'rejected': self.thread_pool_monitor.rejected_tasks
            },
            'connection_pool': {
                'active': self.connection_pool_monitor.active_connections,
                'idle': self.connection_pool_monitor.idle_connections,
                'wait_time': self.connection_pool_monitor.avg_wait_time
            }
        }
```

**Targets**:
- Thread Pool Utilization: 50-80%
- Connection Pool Utilization: 40-70%
- Queue Wait Time: < 5ms
- Rejected Tasks: < 0.1%

## Measurement Implementation

### Decorator-Based Instrumentation

```python
def measure_latency(metric_name):
    """Decorator for automatic latency measurement"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start = time.perf_counter_ns()
            
            try:
                result = await func(*args, **kwargs)
                latency_us = (time.perf_counter_ns() - start) / 1000
                
                metrics.histogram(
                    f'{metric_name}_latency_us',
                    latency_us,
                    tags={'status': 'success'}
                )
                
                return result
                
            except Exception as e:
                latency_us = (time.perf_counter_ns() - start) / 1000
                
                metrics.histogram(
                    f'{metric_name}_latency_us',
                    latency_us,
                    tags={'status': 'error', 'error_type': type(e).__name__}
                )
                
                raise
                
        return wrapper
    return decorator
```

### Continuous Monitoring

```python
class MetricsCollector:
    """Continuous metrics collection and aggregation"""
    
    def __init__(self, interval=1.0):
        self.interval = interval
        self.collectors = []
        self.aggregators = []
        self.exporters = []
        
    async def run(self):
        """Main collection loop"""
        while self.running:
            # Collect metrics
            raw_metrics = await self.collect_all()
            
            # Aggregate
            aggregated = await self.aggregate(raw_metrics)
            
            # Export
            await self.export(aggregated)
            
            # Check alerts
            await self.check_alerts(aggregated)
            
            await asyncio.sleep(self.interval)
```

## Testing Strategy

### Performance Test Suite

```python
# test_performance_requirements.py
import pytest
from metrics import LatencyMetric, ThroughputMetric

class TestPerformanceRequirements:
    """Test that all performance requirements are met"""
    
    @pytest.mark.performance
    async def test_signal_generation_latency(self, benchmark_harness):
        """Verify signal generation meets latency requirements"""
        metric = LatencyMetric()
        
        # Run benchmark
        await benchmark_harness.run_signal_generation_test(
            duration=300,  # 5 minutes
            load_profile='production'
        )
        
        # Check requirements
        assert metric.get_percentile(99) < 100  # 100ms P99
        assert metric.get_percentile(95) < 75   # 75ms P95
        assert metric.get_percentile(90) < 50   # 50ms P90
    
    @pytest.mark.performance
    async def test_throughput_requirements(self, benchmark_harness):
        """Verify throughput meets requirements"""
        metric = ThroughputMetric()
        
        # Run benchmark
        await benchmark_harness.run_throughput_test(
            duration=300,
            target_rate=10000  # 10k signals/sec
        )
        
        # Check sustained throughput
        assert metric.get_sustained_rate() > 10000
        assert metric.get_burst_rate() > 50000
```

## Visualization and Reporting

### Real-Time Dashboard

```python
class MetricsDashboard:
    """Real-time metrics visualization"""
    
    def __init__(self):
        self.app = dash.Dash(__name__)
        self.setup_layout()
        self.setup_callbacks()
        
    def setup_layout(self):
        """Create dashboard layout"""
        self.app.layout = html.Div([
            # Latency gauges
            dcc.Graph(id='latency-gauges'),
            
            # Throughput time series
            dcc.Graph(id='throughput-chart'),
            
            # Strategy performance heatmap
            dcc.Graph(id='strategy-heatmap'),
            
            # Resource utilization
            dcc.Graph(id='resource-meters'),
            
            # Update interval
            dcc.Interval(id='interval', interval=1000)
        ])
```

### Performance Reports

```python
class PerformanceReporter:
    """Generate comprehensive performance reports"""
    
    def generate_report(self, metrics_data):
        """Create detailed performance report"""
        report = {
            'executive_summary': self.create_summary(metrics_data),
            'latency_analysis': self.analyze_latency(metrics_data),
            'throughput_analysis': self.analyze_throughput(metrics_data),
            'strategy_performance': self.analyze_strategies(metrics_data),
            'resource_utilization': self.analyze_resources(metrics_data),
            'recommendations': self.generate_recommendations(metrics_data)
        }
        
        return self.format_report(report)
```

---
*Document Version: 1.0*  
*Last Updated: 2025-06-20*  
*Status: Specification Phase*