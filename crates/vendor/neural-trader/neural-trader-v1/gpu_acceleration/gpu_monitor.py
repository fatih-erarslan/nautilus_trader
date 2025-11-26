"""
Comprehensive GPU Monitoring and Profiling System
Advanced monitoring, profiling, and performance analysis for GPU operations.
"""

import cupy as cp
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
import logging
from dataclasses import dataclass, field
from enum import Enum
import threading
import time
import json
import os
import subprocess
import psutil
from collections import deque, defaultdict
import warnings

from .flyio_gpu_config import get_gpu_config_manager
from .gpu_memory_manager import get_gpu_memory_manager

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics to monitor."""
    GPU_UTILIZATION = "gpu_utilization"
    MEMORY_USAGE = "memory_usage"
    TEMPERATURE = "temperature"
    POWER_CONSUMPTION = "power_consumption"
    THROUGHPUT = "throughput"
    LATENCY = "latency"
    ERROR_RATE = "error_rate"
    KERNEL_EXECUTION = "kernel_execution"


@dataclass
class GPUMetrics:
    """GPU performance metrics."""
    timestamp: float
    gpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    memory_used_mb: float = 0.0
    memory_total_mb: float = 0.0
    temperature_c: float = 0.0
    power_draw_w: float = 0.0
    power_limit_w: float = 0.0
    clock_sm_mhz: float = 0.0
    clock_memory_mhz: float = 0.0
    fan_speed_pct: float = 0.0
    throughput_ops_sec: float = 0.0
    avg_latency_ms: float = 0.0
    error_count: int = 0
    active_kernels: int = 0


@dataclass 
class KernelProfile:
    """Kernel execution profile."""
    kernel_name: str
    execution_time_ms: float
    grid_size: Tuple[int, int, int]
    block_size: Tuple[int, int, int]
    shared_memory_bytes: int
    registers_per_thread: int
    occupancy_pct: float
    memory_throughput_gb_s: float
    compute_throughput_gflops: float
    timestamp: float


@dataclass
class PerformanceProfile:
    """Comprehensive performance profile."""
    duration_seconds: float
    total_operations: int
    avg_throughput: float
    peak_throughput: float
    avg_latency: float
    p95_latency: float
    p99_latency: float
    memory_efficiency: float
    compute_efficiency: float
    energy_efficiency: float
    error_rate: float
    kernel_profiles: List[KernelProfile] = field(default_factory=list)


class NVMLInterface:
    """Interface to NVIDIA Management Library for detailed GPU metrics."""
    
    def __init__(self):
        """Initialize NVML interface."""
        self.nvml_available = False
        self.device_count = 0
        self.device_handles = []
        
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nvml_available = True
            self.device_count = pynvml.nvmlDeviceGetCount()
            
            for i in range(self.device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                self.device_handles.append(handle)
                
            logger.info(f"NVML initialized with {self.device_count} devices")
            
        except ImportError:
            logger.warning("pynvml not available, using fallback methods")
        except Exception as e:
            logger.warning(f"NVML initialization failed: {str(e)}")
            
    def get_device_metrics(self, device_id: int = 0) -> Dict[str, Any]:
        """Get detailed device metrics."""
        if not self.nvml_available or device_id >= len(self.device_handles):
            return self._get_fallback_metrics(device_id)
            
        try:
            import pynvml
            handle = self.device_handles[device_id]
            
            # Get utilization
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            
            # Get memory info
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            
            # Get temperature
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            
            # Get power info
            power_draw = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
            power_limit = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)[1] / 1000.0
            
            # Get clock speeds
            sm_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
            mem_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
            
            # Get fan speed (if available)
            try:
                fan_speed = pynvml.nvmlDeviceGetFanSpeed(handle)
            except:
                fan_speed = 0
                
            return {
                'gpu_utilization': util.gpu,
                'memory_utilization': (mem_info.used / mem_info.total) * 100,
                'memory_used_mb': mem_info.used / (1024**2),
                'memory_total_mb': mem_info.total / (1024**2),
                'temperature_c': temp,
                'power_draw_w': power_draw,
                'power_limit_w': power_limit,
                'clock_sm_mhz': sm_clock,
                'clock_memory_mhz': mem_clock,
                'fan_speed_pct': fan_speed
            }
            
        except Exception as e:
            logger.warning(f"NVML metrics collection failed: {str(e)}")
            return self._get_fallback_metrics(device_id)
            
    def _get_fallback_metrics(self, device_id: int = 0) -> Dict[str, Any]:
        """Get basic metrics using fallback methods."""
        try:
            # Use nvidia-smi for basic metrics
            result = subprocess.run([
                'nvidia-smi', '--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw',
                '--format=csv,noheader,nounits', f'--id={device_id}'
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                values = result.stdout.strip().split(', ')
                return {
                    'gpu_utilization': float(values[0]) if values[0] != '[Not Supported]' else 0.0,
                    'memory_utilization': float(values[1]) if values[1] != '[Not Supported]' else 0.0,
                    'memory_used_mb': float(values[2]) if values[2] != '[Not Supported]' else 0.0,
                    'memory_total_mb': float(values[3]) if values[3] != '[Not Supported]' else 0.0,
                    'temperature_c': float(values[4]) if values[4] != '[Not Supported]' else 0.0,
                    'power_draw_w': float(values[5]) if values[5] != '[Not Supported]' else 0.0,
                    'power_limit_w': 0.0,
                    'clock_sm_mhz': 0.0,
                    'clock_memory_mhz': 0.0,
                    'fan_speed_pct': 0.0
                }
        except:
            pass
            
        # Last resort: use CuPy memory info
        try:
            device = cp.cuda.Device(device_id)
            mem_info = device.mem_info
            
            return {
                'gpu_utilization': 0.0,
                'memory_utilization': ((mem_info[1] - mem_info[0]) / mem_info[1]) * 100,
                'memory_used_mb': (mem_info[1] - mem_info[0]) / (1024**2),
                'memory_total_mb': mem_info[1] / (1024**2),
                'temperature_c': 0.0,
                'power_draw_w': 0.0,
                'power_limit_w': 0.0,
                'clock_sm_mhz': 0.0,
                'clock_memory_mhz': 0.0,
                'fan_speed_pct': 0.0
            }
        except:
            return {key: 0.0 for key in [
                'gpu_utilization', 'memory_utilization', 'memory_used_mb', 
                'memory_total_mb', 'temperature_c', 'power_draw_w',
                'power_limit_w', 'clock_sm_mhz', 'clock_memory_mhz', 'fan_speed_pct'
            ]}


class KernelProfiler:
    """Profiles CUDA kernel execution."""
    
    def __init__(self):
        """Initialize kernel profiler."""
        self.profiling_enabled = False
        self.kernel_profiles = deque(maxlen=1000)
        self.active_profiles = {}
        
    def start_profiling(self):
        """Start kernel profiling."""
        self.profiling_enabled = True
        logger.info("Kernel profiling enabled")
        
    def stop_profiling(self):
        """Stop kernel profiling."""
        self.profiling_enabled = False
        logger.info("Kernel profiling disabled")
        
    def profile_kernel(self, kernel_name: str):
        """Decorator for profiling kernel execution."""
        def decorator(func):
            def wrapper(*args, **kwargs):
                if not self.profiling_enabled:
                    return func(*args, **kwargs)
                    
                # Start timing
                start_event = cp.cuda.Event()
                end_event = cp.cuda.Event()
                
                start_event.record()
                result = func(*args, **kwargs)
                end_event.record()
                end_event.synchronize()
                
                # Calculate execution time
                execution_time = cp.cuda.get_elapsed_time(start_event, end_event)
                
                # Create profile (simplified - in practice would use CUPTI)
                profile = KernelProfile(
                    kernel_name=kernel_name,
                    execution_time_ms=execution_time,
                    grid_size=(1, 1, 1),  # Would need actual grid size
                    block_size=(256, 1, 1),  # Would need actual block size
                    shared_memory_bytes=0,
                    registers_per_thread=0,
                    occupancy_pct=0.0,
                    memory_throughput_gb_s=0.0,
                    compute_throughput_gflops=0.0,
                    timestamp=time.time()
                )
                
                self.kernel_profiles.append(profile)
                return result
                
            return wrapper
        return decorator
        
    def get_kernel_stats(self) -> Dict[str, Any]:
        """Get kernel execution statistics."""
        if not self.kernel_profiles:
            return {}
            
        profiles_by_kernel = defaultdict(list)
        for profile in self.kernel_profiles:
            profiles_by_kernel[profile.kernel_name].append(profile)
            
        stats = {}
        for kernel_name, profiles in profiles_by_kernel.items():
            execution_times = [p.execution_time_ms for p in profiles]
            
            stats[kernel_name] = {
                'call_count': len(profiles),
                'total_time_ms': sum(execution_times),
                'avg_time_ms': np.mean(execution_times),
                'min_time_ms': min(execution_times),
                'max_time_ms': max(execution_times),
                'std_time_ms': np.std(execution_times)
            }
            
        return stats


class GPUMonitor:
    """Comprehensive GPU monitoring system."""
    
    def __init__(self, sampling_interval: float = 1.0, history_size: int = 3600):
        """Initialize GPU monitor."""
        self.sampling_interval = sampling_interval
        self.history_size = history_size
        
        # Initialize interfaces
        self.nvml = NVMLInterface()
        self.kernel_profiler = KernelProfiler()
        self.memory_manager = get_gpu_memory_manager()
        
        # Monitoring state
        self.is_monitoring = False
        self.monitor_thread = None
        self.metrics_history = deque(maxlen=history_size)
        self.performance_counters = defaultdict(float)
        self.alert_thresholds = {}
        
        # Performance tracking
        self.operation_count = 0
        self.error_count = 0
        self.start_time = time.time()
        
        # Set default alert thresholds
        self._set_default_thresholds()
        
    def _set_default_thresholds(self):
        """Set default alert thresholds."""
        self.alert_thresholds = {
            'gpu_utilization': {'min': 10.0, 'max': 95.0},
            'memory_utilization': {'min': 5.0, 'max': 90.0},
            'temperature_c': {'min': 30.0, 'max': 80.0},
            'power_draw_w': {'min': 50.0, 'max': 350.0},
            'error_rate': {'min': 0.0, 'max': 0.01}
        }
        
    def start_monitoring(self):
        """Start GPU monitoring."""
        if self.is_monitoring:
            return  
            
        self.is_monitoring = True
        self.start_time = time.time()
        
        self.monitor_thread = threading.Thread(target=self._monitoring_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        self.kernel_profiler.start_profiling()
        
        logger.info(f"GPU monitoring started (interval: {self.sampling_interval}s)")
        
    def stop_monitoring(self):
        """Stop GPU monitoring."""
        self.is_monitoring = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
            
        self.kernel_profiler.stop_profiling()
        
        logger.info("GPU monitoring stopped")
        
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                # Collect metrics
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # Check alerts
                self._check_alerts(metrics)
                
                # Sleep until next sample
                time.sleep(self.sampling_interval)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {str(e)}")
                time.sleep(self.sampling_interval)
                
    def _collect_metrics(self) -> GPUMetrics:
        """Collect comprehensive GPU metrics."""
        timestamp = time.time()
        
        # Get NVML metrics
        nvml_metrics = self.nvml.get_device_metrics(0)
        
        # Get memory manager metrics
        memory_info = self.memory_manager.get_memory_info()
        
        # Calculate derived metrics
        runtime = timestamp - self.start_time
        throughput = self.operation_count / runtime if runtime > 0 else 0.0
        error_rate = self.error_count / max(self.operation_count, 1)
        
        # Get recent latency data (simplified)
        recent_metrics = list(self.metrics_history)[-10:]
        avg_latency = np.mean([m.avg_latency_ms for m in recent_metrics]) if recent_metrics else 0.0
        
        metrics = GPUMetrics(
            timestamp=timestamp,
            gpu_utilization=nvml_metrics['gpu_utilization'],
            memory_utilization=nvml_metrics['memory_utilization'],
            memory_used_mb=nvml_metrics['memory_used_mb'],
            memory_total_mb=nvml_metrics['memory_total_mb'],
            temperature_c=nvml_metrics['temperature_c'],
            power_draw_w=nvml_metrics['power_draw_w'],
            power_limit_w=nvml_metrics['power_limit_w'],
            clock_sm_mhz=nvml_metrics['clock_sm_mhz'],
            clock_memory_mhz=nvml_metrics['clock_memory_mhz'],
            fan_speed_pct=nvml_metrics['fan_speed_pct'],
            throughput_ops_sec=throughput,
            avg_latency_ms=avg_latency,
            error_count=self.error_count,
            active_kernels=0  # Would need CUPTI for accurate count
        )
        
        return metrics
        
    def _check_alerts(self, metrics: GPUMetrics):
        """Check metrics against alert thresholds."""
        alerts = []
        
        for metric_name, thresholds in self.alert_thresholds.items():
            value = getattr(metrics, metric_name, 0.0)
            
            if value < thresholds['min']:
                alerts.append(f"{metric_name} too low: {value:.2f} < {thresholds['min']}")
            elif value > thresholds['max']:
                alerts.append(f"{metric_name} too high: {value:.2f} > {thresholds['max']}")
                
        if alerts:
            logger.warning(f"GPU alerts: {', '.join(alerts)}")
            
    def record_operation(self, success: bool = True, latency_ms: float = 0.0):
        """Record operation for performance tracking."""
        self.operation_count += 1
        if not success:
            self.error_count += 1
            
        # Update performance counters
        self.performance_counters['total_operations'] = self.operation_count
        self.performance_counters['total_errors'] = self.error_count
        self.performance_counters['last_latency_ms'] = latency_ms
        
    def get_current_metrics(self) -> Optional[GPUMetrics]:
        """Get the most recent metrics."""
        if self.metrics_history:
            return self.metrics_history[-1]
        return None
        
    def get_metrics_summary(self, window_minutes: int = 5) -> Dict[str, Any]:
        """Get summary statistics for recent metrics."""
        if not self.metrics_history:
            return {}
            
        # Filter to recent window
        cutoff_time = time.time() - (window_minutes * 60)
        recent_metrics = [m for m in self.metrics_history if m.timestamp >= cutoff_time]
        
        if not recent_metrics:
            return {}
            
        # Calculate summary statistics
        summary = {
            'window_minutes': window_minutes,
            'sample_count': len(recent_metrics),
            'gpu_utilization': {
                'avg': np.mean([m.gpu_utilization for m in recent_metrics]),
                'min': np.min([m.gpu_utilization for m in recent_metrics]),
                'max': np.max([m.gpu_utilization for m in recent_metrics]),
                'std': np.std([m.gpu_utilization for m in recent_metrics])
            },
            'memory_utilization': {
                'avg': np.mean([m.memory_utilization for m in recent_metrics]),
                'min': np.min([m.memory_utilization for m in recent_metrics]),
                'max': np.max([m.memory_utilization for m in recent_metrics]),
                'std': np.std([m.memory_utilization for m in recent_metrics])
            },
            'temperature': {
                'avg': np.mean([m.temperature_c for m in recent_metrics]),
                'min': np.min([m.temperature_c for m in recent_metrics]),
                'max': np.max([m.temperature_c for m in recent_metrics])
            },
            'power_draw': {
                'avg': np.mean([m.power_draw_w for m in recent_metrics]),
                'min': np.min([m.power_draw_w for m in recent_metrics]),
                'max': np.max([m.power_draw_w for m in recent_metrics])
            },
            'throughput': {
                'avg': np.mean([m.throughput_ops_sec for m in recent_metrics]),
                'max': np.max([m.throughput_ops_sec for m in recent_metrics])
            }
        }
        
        return summary
        
    def generate_performance_report(self) -> PerformanceProfile:
        """Generate comprehensive performance report."""
        if not self.metrics_history:
            return PerformanceProfile(
                duration_seconds=0.0,
                total_operations=0,
                avg_throughput=0.0,
                peak_throughput=0.0,
                avg_latency=0.0,
                p95_latency=0.0,
                p99_latency=0.0,
                memory_efficiency=0.0,
                compute_efficiency=0.0,
                energy_efficiency=0.0,
                error_rate=0.0
            )
            
        duration = time.time() - self.start_time
        
        # Calculate throughput statistics
        throughputs = [m.throughput_ops_sec for m in self.metrics_history]
        avg_throughput = np.mean(throughputs)
        peak_throughput = np.max(throughputs)
        
        # Calculate latency statistics
        latencies = [m.avg_latency_ms for m in self.metrics_history if m.avg_latency_ms > 0]
        avg_latency = np.mean(latencies) if latencies else 0.0
        p95_latency = np.percentile(latencies, 95) if latencies else 0.0
        p99_latency = np.percentile(latencies, 99) if latencies else 0.0
        
        # Calculate efficiency metrics
        gpu_utils = [m.gpu_utilization for m in self.metrics_history]
        memory_utils = [m.memory_utilization for m in self.metrics_history]
        power_draws = [m.power_draw_w for m in self.metrics_history if m.power_draw_w > 0]
        
        compute_efficiency = np.mean(gpu_utils) / 100.0
        memory_efficiency = np.mean(memory_utils) / 100.0
        
        # Energy efficiency (operations per watt-hour)
        avg_power = np.mean(power_draws) if power_draws else 0.0
        energy_efficiency = (avg_throughput / avg_power) if avg_power > 0 else 0.0
        
        # Error rate
        error_rate = self.error_count / max(self.operation_count, 1)
        
        # Get kernel profiles
        kernel_profiles = list(self.kernel_profiler.kernel_profiles)
        
        return PerformanceProfile(
            duration_seconds=duration,
            total_operations=self.operation_count,
            avg_throughput=avg_throughput,
            peak_throughput=peak_throughput,
            avg_latency=avg_latency,
            p95_latency=p95_latency,
            p99_latency=p99_latency,
            memory_efficiency=memory_efficiency,
            compute_efficiency=compute_efficiency,
            energy_efficiency=energy_efficiency,
            error_rate=error_rate,
            kernel_profiles=kernel_profiles
        )
        
    def export_metrics(self, filepath: str, format: str = "json"):
        """Export metrics to file."""
        if format == "json":
            data = {
                'metadata': {
                    'export_time': time.time(),
                    'duration_seconds': time.time() - self.start_time,
                    'total_samples': len(self.metrics_history)
                },
                'metrics': [
                    {
                        'timestamp': m.timestamp,
                        'gpu_utilization': m.gpu_utilization,
                        'memory_utilization': m.memory_utilization,
                        'temperature_c': m.temperature_c,
                        'power_draw_w': m.power_draw_w,
                        'throughput_ops_sec': m.throughput_ops_sec,
                        'avg_latency_ms': m.avg_latency_ms,
                        'error_count': m.error_count
                    }
                    for m in self.metrics_history
                ],
                'performance_profile': self.generate_performance_report().__dict__,
                'kernel_stats': self.kernel_profiler.get_kernel_stats()
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
        logger.info(f"Metrics exported to {filepath}")
        
    def create_dashboard_data(self) -> Dict[str, Any]:
        """Create data for monitoring dashboard."""
        current_metrics = self.get_current_metrics()
        summary = self.get_metrics_summary(5)  # Last 5 minutes
        performance_profile = self.generate_performance_report()
        
        return {
            'current': current_metrics.__dict__ if current_metrics else {},
            'summary': summary,
            'performance': performance_profile.__dict__,
            'kernel_stats': self.kernel_profiler.get_kernel_stats(),
            'memory_info': self.memory_manager.get_memory_info(),
            'alerts_active': len([m for m in self.metrics_history[-10:] 
                                if self._has_alerts(m)]) if self.metrics_history else 0
        }
        
    def _has_alerts(self, metrics: GPUMetrics) -> bool:
        """Check if metrics have any alerts."""
        for metric_name, thresholds in self.alert_thresholds.items():
            value = getattr(metrics, metric_name, 0.0)
            if value < thresholds['min'] or value > thresholds['max']:
                return True
        return False


# Global monitor instance
_global_gpu_monitor = None


def get_gpu_monitor() -> GPUMonitor:
    """Get the global GPU monitor."""
    global _global_gpu_monitor
    if _global_gpu_monitor is None:
        _global_gpu_monitor = GPUMonitor()
    return _global_gpu_monitor


def monitor_gpu_operation(operation_name: str):
    """Decorator for monitoring GPU operations."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            monitor = get_gpu_monitor()
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                end_time = time.time()
                latency = (end_time - start_time) * 1000  # Convert to ms
                
                monitor.record_operation(success=True, latency_ms=latency)
                return result
                
            except Exception as e:
                end_time = time.time()
                latency = (end_time - start_time) * 1000
                
                monitor.record_operation(success=False, latency_ms=latency)
                raise
                
        return wrapper
    return decorator


if __name__ == "__main__":
    # Test GPU monitor
    logger.info("Testing GPU monitor...")
    
    # Create monitor
    monitor = get_gpu_monitor()
    
    # Start monitoring
    monitor.start_monitoring()
    
    # Simulate some operations
    @monitor_gpu_operation("test_operation")
    def test_operation():
        # Simulate GPU work
        a = cp.random.rand(1000, 1000)
        b = cp.random.rand(1000, 1000)
        c = cp.matmul(a, b)
        cp.cuda.Stream.null.synchronize()
        return c
        
    # Run test operations
    for i in range(10):
        test_operation()
        time.sleep(0.1)
        
    # Wait for metrics collection
    time.sleep(2.0)
    
    # Get metrics
    current = monitor.get_current_metrics()
    if current:
        logger.info(f"Current GPU utilization: {current.gpu_utilization}%")
        logger.info(f"Current memory utilization: {current.memory_utilization}%")
        
    # Get summary
    summary = monitor.get_metrics_summary(1)  # Last 1 minute
    logger.info(f"Summary: {summary}")
    
    # Generate report
    report = monitor.generate_performance_report()
    logger.info(f"Performance report: avg_throughput={report.avg_throughput:.2f} ops/sec")
    
    # Export metrics
    monitor.export_metrics("/tmp/gpu_metrics.json")
    
    # Stop monitoring
    monitor.stop_monitoring()
    
    print("GPU monitor tested successfully!")