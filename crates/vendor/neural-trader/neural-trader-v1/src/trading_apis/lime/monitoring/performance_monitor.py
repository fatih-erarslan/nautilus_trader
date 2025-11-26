"""
Performance Monitor for Ultra-Low Latency Trading

Features:
- Real-time performance metrics
- Hardware-level monitoring
- Latency distribution tracking
- Memory usage optimization
- CPU utilization analysis
"""

import time
import threading
import psutil
import gc
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import deque
import numpy as np
from datetime import datetime, timedelta
import json

# Try to import optional performance libraries
try:
    import py_cpuinfo
    CPUINFO_AVAILABLE = True
except ImportError:
    CPUINFO_AVAILABLE = False


@dataclass
class LatencyMetrics:
    """Latency metrics with distribution analysis"""
    count: int = 0
    sum_ns: int = 0
    min_ns: int = float('inf')
    max_ns: int = 0
    p50_ns: int = 0
    p95_ns: int = 0
    p99_ns: int = 0
    p999_ns: int = 0
    
    def add_sample(self, latency_ns: int):
        """Add latency sample"""
        self.count += 1
        self.sum_ns += latency_ns
        self.min_ns = min(self.min_ns, latency_ns)
        self.max_ns = max(self.max_ns, latency_ns)
        
    def get_mean_us(self) -> float:
        """Get mean latency in microseconds"""
        return (self.sum_ns / self.count) / 1000 if self.count > 0 else 0
        
    def update_percentiles(self, samples: List[int]):
        """Update percentiles from samples"""
        if not samples:
            return
            
        sorted_samples = sorted(samples)
        n = len(sorted_samples)
        
        self.p50_ns = sorted_samples[int(n * 0.5)]
        self.p95_ns = sorted_samples[int(n * 0.95)]
        self.p99_ns = sorted_samples[int(n * 0.99)]
        self.p999_ns = sorted_samples[int(n * 0.999)]


@dataclass
class SystemMetrics:
    """System performance metrics"""
    timestamp: float = field(default_factory=time.time)
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_rss_mb: float = 0.0
    memory_vms_mb: float = 0.0
    gc_collections: int = 0
    gc_collected: int = 0
    gc_uncollectable: int = 0
    thread_count: int = 0
    fd_count: int = 0
    context_switches: int = 0
    page_faults: int = 0
    cache_misses: int = 0
    instructions_per_cycle: float = 0.0


@dataclass
class TradingMetrics:
    """Trading-specific performance metrics"""
    orders_per_second: float = 0.0
    fills_per_second: float = 0.0
    messages_per_second: float = 0.0
    
    # Latency metrics
    order_latency: LatencyMetrics = field(default_factory=LatencyMetrics)
    fill_latency: LatencyMetrics = field(default_factory=LatencyMetrics)
    risk_check_latency: LatencyMetrics = field(default_factory=LatencyMetrics)
    
    # Throughput metrics
    total_orders: int = 0
    total_fills: int = 0
    total_messages: int = 0
    
    # Error metrics
    risk_rejections: int = 0
    order_rejections: int = 0
    connection_errors: int = 0


class PerformanceMonitor:
    """
    High-performance monitoring system for trading applications
    """
    
    def __init__(self, interval_ms: int = 1000, history_size: int = 3600):
        self.interval_ms = interval_ms
        self.history_size = history_size
        
        # Metrics storage
        self.system_metrics = deque(maxlen=history_size)
        self.trading_metrics = deque(maxlen=history_size)
        
        # Current metrics
        self.current_system = SystemMetrics()
        self.current_trading = TradingMetrics()
        
        # Latency sample buffers
        self.order_latency_samples = deque(maxlen=10000)
        self.fill_latency_samples = deque(maxlen=10000)
        self.risk_check_samples = deque(maxlen=10000)
        
        # Process reference
        self.process = psutil.Process()
        
        # Hardware info
        self.hardware_info = self._get_hardware_info()
        
        # Monitoring thread
        self.monitoring_thread = None
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        
        # Performance counters
        self.last_context_switches = 0
        self.last_page_faults = 0
        self.last_timestamp = time.time()
        
    def _get_hardware_info(self) -> Dict[str, Any]:
        """Get hardware information"""
        info = {
            'cpu_count': psutil.cpu_count(),
            'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else {},
            'memory_total': psutil.virtual_memory().total,
            'platform': psutil.LINUX if hasattr(psutil, 'LINUX') else 'unknown'
        }
        
        if CPUINFO_AVAILABLE:
            try:
                cpu_info = py_cpuinfo.get_cpu_info()
                info.update({
                    'cpu_brand': cpu_info.get('brand_raw', 'unknown'),
                    'cpu_arch': cpu_info.get('arch', 'unknown'),
                    'cpu_cache_l1': cpu_info.get('l1_cache_size', 'unknown'),
                    'cpu_cache_l2': cpu_info.get('l2_cache_size', 'unknown'),
                    'cpu_cache_l3': cpu_info.get('l3_cache_size', 'unknown')
                })
            except:
                pass
                
        return info
        
    def start(self):
        """Start performance monitoring"""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            return
            
        self.stop_event.clear()
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        
    def stop(self):
        """Stop performance monitoring"""
        if self.monitoring_thread:
            self.stop_event.set()
            self.monitoring_thread.join(timeout=1.0)
            
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while not self.stop_event.is_set():
            try:
                # Collect system metrics
                self._collect_system_metrics()
                
                # Update latency percentiles
                self._update_latency_percentiles()
                
                # Store metrics
                with self.lock:
                    self.system_metrics.append(self.current_system)
                    self.trading_metrics.append(self.current_trading)
                    
                # Reset counters
                self.current_trading = TradingMetrics()
                
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                
            # Wait for next interval
            self.stop_event.wait(self.interval_ms / 1000.0)
            
    def _collect_system_metrics(self):
        """Collect system performance metrics"""
        try:
            # CPU metrics
            self.current_system.cpu_percent = self.process.cpu_percent()
            
            # Memory metrics
            memory_info = self.process.memory_info()
            self.current_system.memory_rss_mb = memory_info.rss / 1024 / 1024
            self.current_system.memory_vms_mb = memory_info.vms / 1024 / 1024
            
            # System memory
            sys_memory = psutil.virtual_memory()
            self.current_system.memory_percent = sys_memory.percent
            
            # GC metrics
            gc_stats = gc.get_stats()
            if gc_stats:
                self.current_system.gc_collections = sum(stat['collections'] for stat in gc_stats)
                self.current_system.gc_collected = sum(stat['collected'] for stat in gc_stats)
                self.current_system.gc_uncollectable = sum(stat['uncollectable'] for stat in gc_stats)
                
            # Thread count
            self.current_system.thread_count = self.process.num_threads()
            
            # File descriptor count
            try:
                self.current_system.fd_count = self.process.num_fds()
            except:
                self.current_system.fd_count = 0
                
            # Context switches and page faults
            try:
                ctx_switches = self.process.num_ctx_switches()
                current_ctx_switches = ctx_switches.voluntary + ctx_switches.involuntary
                
                if self.last_context_switches > 0:
                    self.current_system.context_switches = current_ctx_switches - self.last_context_switches
                self.last_context_switches = current_ctx_switches
                
            except:
                self.current_system.context_switches = 0
                
            self.current_system.timestamp = time.time()
            
        except Exception as e:
            print(f"Error collecting system metrics: {e}")
            
    def _update_latency_percentiles(self):
        """Update latency percentiles from samples"""
        # Order latency
        if self.order_latency_samples:
            samples = list(self.order_latency_samples)
            self.current_trading.order_latency.update_percentiles(samples)
            
        # Fill latency
        if self.fill_latency_samples:
            samples = list(self.fill_latency_samples)
            self.current_trading.fill_latency.update_percentiles(samples)
            
        # Risk check latency
        if self.risk_check_samples:
            samples = list(self.risk_check_samples)
            self.current_trading.risk_check_latency.update_percentiles(samples)
            
    def record_order_latency(self, latency_ns: int):
        """Record order latency sample"""
        with self.lock:
            self.current_trading.order_latency.add_sample(latency_ns)
            self.order_latency_samples.append(latency_ns)
            
    def record_fill_latency(self, latency_ns: int):
        """Record fill latency sample"""
        with self.lock:
            self.current_trading.fill_latency.add_sample(latency_ns)
            self.fill_latency_samples.append(latency_ns)
            
    def record_risk_check_latency(self, latency_ns: int):
        """Record risk check latency sample"""
        with self.lock:
            self.current_trading.risk_check_latency.add_sample(latency_ns)
            self.risk_check_samples.append(latency_ns)
            
    def increment_orders(self):
        """Increment order counter"""
        with self.lock:
            self.current_trading.total_orders += 1
            
    def increment_fills(self):
        """Increment fill counter"""
        with self.lock:
            self.current_trading.total_fills += 1
            
    def increment_messages(self):
        """Increment message counter"""
        with self.lock:
            self.current_trading.total_messages += 1
            
    def increment_risk_rejections(self):
        """Increment risk rejection counter"""
        with self.lock:
            self.current_trading.risk_rejections += 1
            
    def increment_order_rejections(self):
        """Increment order rejection counter"""
        with self.lock:
            self.current_trading.order_rejections += 1
            
    def get_current_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        with self.lock:
            latest_system = self.system_metrics[-1] if self.system_metrics else self.current_system
            latest_trading = self.trading_metrics[-1] if self.trading_metrics else self.current_trading
            
            return {
                'system': {
                    'cpu_percent': latest_system.cpu_percent,
                    'memory_percent': latest_system.memory_percent,
                    'memory_rss_mb': latest_system.memory_rss_mb,
                    'memory_vms_mb': latest_system.memory_vms_mb,
                    'gc_collections': latest_system.gc_collections,
                    'thread_count': latest_system.thread_count,
                    'fd_count': latest_system.fd_count,
                    'context_switches': latest_system.context_switches
                },
                'trading': {
                    'orders_per_second': latest_trading.orders_per_second,
                    'fills_per_second': latest_trading.fills_per_second,
                    'total_orders': latest_trading.total_orders,
                    'total_fills': latest_trading.total_fills,
                    'risk_rejections': latest_trading.risk_rejections,
                    'order_rejections': latest_trading.order_rejections,
                    'order_latency_p99_us': latest_trading.order_latency.p99_ns / 1000,
                    'fill_latency_p99_us': latest_trading.fill_latency.p99_ns / 1000,
                    'risk_check_latency_p99_us': latest_trading.risk_check_latency.p99_ns / 1000
                }
            }
            
    def get_latency_summary(self) -> Dict[str, Dict[str, float]]:
        """Get latency summary statistics"""
        with self.lock:
            latest_trading = self.trading_metrics[-1] if self.trading_metrics else self.current_trading
            
            return {
                'order_latency_us': {
                    'mean': latest_trading.order_latency.get_mean_us(),
                    'min': latest_trading.order_latency.min_ns / 1000,
                    'max': latest_trading.order_latency.max_ns / 1000,
                    'p50': latest_trading.order_latency.p50_ns / 1000,
                    'p95': latest_trading.order_latency.p95_ns / 1000,
                    'p99': latest_trading.order_latency.p99_ns / 1000,
                    'p999': latest_trading.order_latency.p999_ns / 1000
                },
                'fill_latency_us': {
                    'mean': latest_trading.fill_latency.get_mean_us(),
                    'min': latest_trading.fill_latency.min_ns / 1000,
                    'max': latest_trading.fill_latency.max_ns / 1000,
                    'p50': latest_trading.fill_latency.p50_ns / 1000,
                    'p95': latest_trading.fill_latency.p95_ns / 1000,
                    'p99': latest_trading.fill_latency.p99_ns / 1000,
                    'p999': latest_trading.fill_latency.p999_ns / 1000
                },
                'risk_check_latency_us': {
                    'mean': latest_trading.risk_check_latency.get_mean_us(),
                    'min': latest_trading.risk_check_latency.min_ns / 1000,
                    'max': latest_trading.risk_check_latency.max_ns / 1000,
                    'p50': latest_trading.risk_check_latency.p50_ns / 1000,
                    'p95': latest_trading.risk_check_latency.p95_ns / 1000,
                    'p99': latest_trading.risk_check_latency.p99_ns / 1000,
                    'p999': latest_trading.risk_check_latency.p999_ns / 1000
                }
            }
            
    def export_metrics(self, file_path: str):
        """Export metrics to JSON file"""
        with self.lock:
            data = {
                'hardware_info': self.hardware_info,
                'system_metrics': [
                    {
                        'timestamp': m.timestamp,
                        'cpu_percent': m.cpu_percent,
                        'memory_percent': m.memory_percent,
                        'memory_rss_mb': m.memory_rss_mb,
                        'gc_collections': m.gc_collections,
                        'thread_count': m.thread_count,
                        'context_switches': m.context_switches
                    }
                    for m in self.system_metrics
                ],
                'trading_metrics': [
                    {
                        'total_orders': m.total_orders,
                        'total_fills': m.total_fills,
                        'risk_rejections': m.risk_rejections,
                        'order_rejections': m.order_rejections,
                        'order_latency_p99_us': m.order_latency.p99_ns / 1000,
                        'fill_latency_p99_us': m.fill_latency.p99_ns / 1000
                    }
                    for m in self.trading_metrics
                ]
            }
            
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
            
    def get_hardware_info(self) -> Dict[str, Any]:
        """Get hardware information"""
        return self.hardware_info.copy()
        
    def get_performance_report(self) -> str:
        """Generate performance report"""
        stats = self.get_current_stats()
        latency = self.get_latency_summary()
        
        report = f"""
=== Performance Report ===
Timestamp: {datetime.now().isoformat()}

System Metrics:
- CPU Usage: {stats['system']['cpu_percent']:.1f}%
- Memory Usage: {stats['system']['memory_percent']:.1f}% ({stats['system']['memory_rss_mb']:.1f} MB RSS)
- GC Collections: {stats['system']['gc_collections']}
- Thread Count: {stats['system']['thread_count']}
- File Descriptors: {stats['system']['fd_count']}
- Context Switches: {stats['system']['context_switches']}

Trading Metrics:
- Orders/sec: {stats['trading']['orders_per_second']:.1f}
- Fills/sec: {stats['trading']['fills_per_second']:.1f}
- Total Orders: {stats['trading']['total_orders']}
- Total Fills: {stats['trading']['total_fills']}
- Risk Rejections: {stats['trading']['risk_rejections']}
- Order Rejections: {stats['trading']['order_rejections']}

Latency Summary (microseconds):
Order Latency:
- Mean: {latency['order_latency_us']['mean']:.2f}
- P99: {latency['order_latency_us']['p99']:.2f}
- P99.9: {latency['order_latency_us']['p999']:.2f}

Fill Latency:
- Mean: {latency['fill_latency_us']['mean']:.2f}
- P99: {latency['fill_latency_us']['p99']:.2f}
- P99.9: {latency['fill_latency_us']['p999']:.2f}

Risk Check Latency:
- Mean: {latency['risk_check_latency_us']['mean']:.2f}
- P99: {latency['risk_check_latency_us']['p99']:.2f}
- P99.9: {latency['risk_check_latency_us']['p999']:.2f}

Hardware Info:
- CPU: {self.hardware_info.get('cpu_brand', 'unknown')}
- CPU Cores: {self.hardware_info['cpu_count']}
- Total Memory: {self.hardware_info['memory_total'] / 1024 / 1024 / 1024:.1f} GB
"""
        
        return report