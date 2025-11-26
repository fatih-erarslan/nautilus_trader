"""
Performance Monitor - Real-time system monitoring for the benchmark system.

This module provides comprehensive performance monitoring including resource usage,
throughput metrics, latency tracking, and system health indicators.
"""

import asyncio
import logging
import time
import psutil
import json
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
from pathlib import Path
import threading
import numpy as np


class MetricType(Enum):
    """Types of performance metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class MetricValue:
    """A single metric value with metadata."""
    name: str
    value: float
    timestamp: float
    labels: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE


@dataclass
class Alert:
    """Performance alert."""
    level: AlertLevel
    message: str
    timestamp: float
    metric_name: str
    threshold: float
    current_value: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemSnapshot:
    """System performance snapshot."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    disk_usage: float
    network_io: Dict[str, int]
    process_count: int
    thread_count: int
    open_files: int
    connections: int


class PerformanceMonitor:
    """
    Comprehensive performance monitoring system.
    
    Features:
    - Real-time system resource monitoring
    - Custom application metrics
    - Alert system with configurable thresholds
    - Historical data storage and analysis
    - Performance trend analysis
    - Resource usage predictions
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = self._setup_logging()
        
        # Monitoring state
        self.is_monitoring = False
        self.monitoring_interval = self.config.get('monitoring_interval', 1.0)  # seconds
        
        # Metrics storage
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = defaultdict(float)
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self.timers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # System snapshots
        self.system_snapshots: deque = deque(maxlen=1000)
        
        # Alert system
        self.alerts: deque = deque(maxlen=500)
        self.alert_thresholds: Dict[str, Dict[str, float]] = {}
        self.alert_handlers: List[Callable[[Alert], None]] = []
        
        # Monitoring tasks
        self.monitoring_tasks: List[asyncio.Task] = []
        
        # Performance analysis
        self.analysis_window = 300  # 5 minutes
        self.trend_analysis_enabled = True
        
        # Persistence
        self.persistence_enabled = self.config.get('persistence_enabled', True)
        self.data_dir = Path(self.config.get('data_dir', 'monitoring_data'))
        self.data_dir.mkdir(exist_ok=True)
        
        # Default alert thresholds
        self._setup_default_thresholds()
        
        self.logger.info("Performance Monitor initialized")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup performance monitor logging."""
        logger = logging.getLogger('performance_monitor')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _setup_default_thresholds(self):
        """Setup default alert thresholds."""
        self.alert_thresholds = {
            'cpu_percent': {
                'warning': 80.0,
                'critical': 90.0,
                'emergency': 95.0
            },
            'memory_percent': {
                'warning': 80.0,
                'critical': 90.0,
                'emergency': 95.0
            },
            'disk_usage': {
                'warning': 80.0,
                'critical': 90.0,
                'emergency': 95.0
            },
            'error_rate': {
                'warning': 5.0,
                'critical': 10.0,
                'emergency': 20.0
            },
            'response_time': {
                'warning': 1000.0,  # ms
                'critical': 5000.0,
                'emergency': 10000.0
            }
        }
    
    async def start(self) -> bool:
        """
        Start performance monitoring.
        
        Returns:
            bool: True if monitoring started successfully
        """
        if self.is_monitoring:
            self.logger.warning("Performance monitoring already running")
            return True
        
        try:
            self.logger.info("Starting performance monitoring...")
            
            # Start monitoring tasks
            self.monitoring_tasks = [
                asyncio.create_task(self._system_monitor()),
                asyncio.create_task(self._metrics_processor()),
                asyncio.create_task(self._alert_processor()),
                asyncio.create_task(self._data_persister())
            ]
            
            if self.trend_analysis_enabled:
                self.monitoring_tasks.append(
                    asyncio.create_task(self._trend_analyzer())
                )
            
            self.is_monitoring = True
            self.logger.info("Performance monitoring started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start performance monitoring: {e}")
            await self.stop()
            return False
    
    async def stop(self) -> bool:
        """
        Stop performance monitoring.
        
        Returns:
            bool: True if monitoring stopped successfully
        """
        if not self.is_monitoring:
            return True
        
        self.logger.info("Stopping performance monitoring...")
        
        try:
            self.is_monitoring = False
            
            # Cancel monitoring tasks
            for task in self.monitoring_tasks:
                task.cancel()
            
            # Wait for tasks to complete
            if self.monitoring_tasks:
                await asyncio.gather(*self.monitoring_tasks, return_exceptions=True)
            
            # Save final data
            await self._save_final_data()
            
            self.logger.info("Performance monitoring stopped successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping performance monitoring: {e}")
            return False
    
    async def _system_monitor(self):
        """Monitor system resources continuously."""
        self.logger.info("System monitor started")
        
        while self.is_monitoring:
            try:
                # Collect system metrics
                snapshot = await self._collect_system_snapshot()
                self.system_snapshots.append(snapshot)
                
                # Update gauge metrics
                self._update_gauge('cpu_percent', snapshot.cpu_percent)
                self._update_gauge('memory_percent', snapshot.memory_percent)
                self._update_gauge('disk_usage', snapshot.disk_usage)
                self._update_gauge('process_count', snapshot.process_count)
                self._update_gauge('thread_count', snapshot.thread_count)
                
                # Check for alerts
                await self._check_system_alerts(snapshot)
                
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"System monitor error: {e}")
                await asyncio.sleep(self.monitoring_interval)
    
    async def _collect_system_snapshot(self) -> SystemSnapshot:
        """Collect a system performance snapshot."""
        def collect_sync():
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=None)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_usage = (disk.used / disk.total) * 100
            
            # Network I/O
            network = psutil.net_io_counters()
            network_io = {
                'bytes_sent': network.bytes_sent,
                'bytes_recv': network.bytes_recv,
                'packets_sent': network.packets_sent,
                'packets_recv': network.packets_recv
            }
            
            # Process info
            process_count = len(psutil.pids())
            
            # Current process info
            current_process = psutil.Process()
            thread_count = current_process.num_threads()
            open_files = len(current_process.open_files())
            connections = len(current_process.connections())
            
            return SystemSnapshot(
                timestamp=time.time(),
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                disk_usage=disk_usage,
                network_io=network_io,
                process_count=process_count,
                thread_count=thread_count,
                open_files=open_files,
                connections=connections
            )
        
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, collect_sync)
    
    async def _check_system_alerts(self, snapshot: SystemSnapshot):
        """Check system metrics against alert thresholds."""
        checks = [
            ('cpu_percent', snapshot.cpu_percent),
            ('memory_percent', snapshot.memory_percent),
            ('disk_usage', snapshot.disk_usage)
        ]
        
        for metric_name, value in checks:
            if metric_name in self.alert_thresholds:
                await self._check_metric_threshold(metric_name, value)
    
    async def _check_metric_threshold(self, metric_name: str, value: float):
        """Check a metric against its alert thresholds."""
        thresholds = self.alert_thresholds.get(metric_name, {})
        
        for level_name, threshold in thresholds.items():
            if value >= threshold:
                alert_level = AlertLevel(level_name.lower())
                alert = Alert(
                    level=alert_level,
                    message=f"{metric_name} is {value:.2f}% (threshold: {threshold:.2f}%)",
                    timestamp=time.time(),
                    metric_name=metric_name,
                    threshold=threshold,
                    current_value=value
                )
                
                await self._emit_alert(alert)
                break  # Only emit the highest level alert
    
    async def _emit_alert(self, alert: Alert):
        """Emit an alert to all registered handlers."""
        self.alerts.append(alert)
        
        # Log alert
        log_level = {
            AlertLevel.INFO: logging.INFO,
            AlertLevel.WARNING: logging.WARNING,
            AlertLevel.CRITICAL: logging.CRITICAL,
            AlertLevel.EMERGENCY: logging.CRITICAL
        }.get(alert.level, logging.INFO)
        
        self.logger.log(log_level, f"ALERT [{alert.level.value.upper()}]: {alert.message}")
        
        # Call alert handlers
        for handler in self.alert_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(alert)
                else:
                    handler(alert)
            except Exception as e:
                self.logger.error(f"Alert handler error: {e}")
    
    async def _metrics_processor(self):
        """Process and aggregate metrics."""
        self.logger.info("Metrics processor started")
        
        while self.is_monitoring:
            try:
                # Process histograms
                for name, values in self.histograms.items():
                    if values:
                        self._update_gauge(f"{name}_mean", np.mean(values))
                        self._update_gauge(f"{name}_p50", np.percentile(values, 50))
                        self._update_gauge(f"{name}_p95", np.percentile(values, 95))
                        self._update_gauge(f"{name}_p99", np.percentile(values, 99))
                
                # Calculate rates for counters
                for name, value in self.counters.items():
                    rate_name = f"{name}_rate"
                    if rate_name in self.gauges:
                        # Calculate rate over last interval
                        prev_value = self.gauges.get(f"{name}_prev", 0)
                        rate = (value - prev_value) / self.monitoring_interval
                        self._update_gauge(rate_name, rate)
                    
                    self.gauges[f"{name}_prev"] = value
                
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Metrics processor error: {e}")
                await asyncio.sleep(self.monitoring_interval)
    
    async def _alert_processor(self):
        """Process alerts and manage alert lifecycle."""
        self.logger.info("Alert processor started")
        
        while self.is_monitoring:
            try:
                # Check for duplicate alerts and suppress if needed
                await self._deduplicate_alerts()
                
                # Auto-resolve alerts if conditions improve
                await self._auto_resolve_alerts()
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Alert processor error: {e}")
                await asyncio.sleep(10)
    
    async def _deduplicate_alerts(self):
        """Remove duplicate alerts within a time window."""
        if len(self.alerts) < 2:
            return
        
        current_time = time.time()
        dedupe_window = 300  # 5 minutes
        
        # Group alerts by metric and level
        alert_groups = defaultdict(list)
        for alert in self.alerts:
            if current_time - alert.timestamp <= dedupe_window:
                key = (alert.metric_name, alert.level)
                alert_groups[key].append(alert)
        
        # Keep only the most recent alert per group
        unique_alerts = []
        for alerts in alert_groups.values():
            if alerts:
                unique_alerts.append(max(alerts, key=lambda a: a.timestamp))
        
        # Add older alerts outside the dedupe window
        for alert in self.alerts:
            if current_time - alert.timestamp > dedupe_window:
                unique_alerts.append(alert)
        
        # Update alerts deque
        self.alerts.clear()
        self.alerts.extend(sorted(unique_alerts, key=lambda a: a.timestamp))
    
    async def _auto_resolve_alerts(self):
        """Auto-resolve alerts if conditions have improved."""
        current_time = time.time()
        resolution_window = 300  # 5 minutes
        
        # Check recent alerts for auto-resolution
        for alert in list(self.alerts):
            if current_time - alert.timestamp > resolution_window:
                continue
            
            # Check if condition has improved
            current_value = self.gauges.get(alert.metric_name, 0)
            if current_value < alert.threshold * 0.9:  # 10% buffer
                self.logger.info(f"Auto-resolved alert: {alert.message}")
                # We don't remove the alert, just log the resolution
    
    async def _trend_analyzer(self):
        """Analyze performance trends and predict issues."""
        self.logger.info("Trend analyzer started")
        
        while self.is_monitoring:
            try:
                await self._analyze_trends()
                await asyncio.sleep(60)  # Analyze every minute
                
            except Exception as e:
                self.logger.error(f"Trend analyzer error: {e}")
                await asyncio.sleep(60)
    
    async def _analyze_trends(self):
        """Analyze performance trends."""
        if len(self.system_snapshots) < 10:
            return  # Need enough data points
        
        # Analyze CPU trend
        cpu_values = [s.cpu_percent for s in list(self.system_snapshots)[-20:]]
        cpu_trend = self._calculate_trend(cpu_values)
        
        if cpu_trend > 5:  # Increasing trend
            self.logger.warning(f"CPU usage trending upward: {cpu_trend:.2f}%/min")
        
        # Analyze memory trend
        memory_values = [s.memory_percent for s in list(self.system_snapshots)[-20:]]
        memory_trend = self._calculate_trend(memory_values)
        
        if memory_trend > 2:  # Increasing trend
            self.logger.warning(f"Memory usage trending upward: {memory_trend:.2f}%/min")
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend slope for a series of values."""
        if len(values) < 2:
            return 0.0
        
        n = len(values)
        x = list(range(n))
        
        # Simple linear regression
        x_mean = sum(x) / n
        y_mean = sum(values) / n
        
        numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return 0.0
        
        slope = numerator / denominator
        return slope * 60 / self.monitoring_interval  # Convert to per-minute rate
    
    async def _data_persister(self):
        """Persist monitoring data to disk."""
        if not self.persistence_enabled:
            return
        
        self.logger.info("Data persister started")
        
        while self.is_monitoring:
            try:
                await self._persist_data()
                await asyncio.sleep(300)  # Persist every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Data persister error: {e}")
                await asyncio.sleep(300)
    
    async def _persist_data(self):
        """Persist current monitoring data."""
        timestamp = int(time.time())
        
        # Save system snapshots
        snapshots_file = self.data_dir / f"system_snapshots_{timestamp}.json"
        snapshots_data = [
            {
                'timestamp': s.timestamp,
                'cpu_percent': s.cpu_percent,
                'memory_percent': s.memory_percent,
                'disk_usage': s.disk_usage,
                'network_io': s.network_io,
                'process_count': s.process_count,
                'thread_count': s.thread_count,
                'open_files': s.open_files,
                'connections': s.connections
            }
            for s in list(self.system_snapshots)
        ]
        
        with open(snapshots_file, 'w') as f:
            json.dump(snapshots_data, f, indent=2)
        
        # Save metrics
        metrics_file = self.data_dir / f"metrics_{timestamp}.json"
        metrics_data = {
            'counters': dict(self.counters),
            'gauges': dict(self.gauges),
            'timestamp': timestamp
        }
        
        with open(metrics_file, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        # Save alerts
        alerts_file = self.data_dir / f"alerts_{timestamp}.json"
        alerts_data = [
            {
                'level': a.level.value,
                'message': a.message,
                'timestamp': a.timestamp,
                'metric_name': a.metric_name,
                'threshold': a.threshold,
                'current_value': a.current_value,
                'metadata': a.metadata
            }
            for a in list(self.alerts)
        ]
        
        with open(alerts_file, 'w') as f:
            json.dump(alerts_data, f, indent=2)
    
    async def _save_final_data(self):
        """Save final data before shutdown."""
        if self.persistence_enabled:
            await self._persist_data()
    
    # Public API methods
    
    def increment_counter(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None):
        """Increment a counter metric."""
        self.counters[name] += value
    
    def _update_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Update a gauge metric."""
        self.gauges[name] = value
        
        # Store in time series
        metric_value = MetricValue(
            name=name,
            value=value,
            timestamp=time.time(),
            labels=labels or {},
            metric_type=MetricType.GAUGE
        )
        self.metrics[name].append(metric_value)
    
    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Set a gauge metric value."""
        self._update_gauge(name, value, labels)
    
    def record_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a value in a histogram."""
        self.histograms[name].append(value)
        
        # Keep only recent values
        if len(self.histograms[name]) > 1000:
            self.histograms[name] = self.histograms[name][-1000:]
    
    def start_timer(self, name: str) -> Callable[[], None]:
        """Start a timer and return a function to stop it."""
        start_time = time.time()
        
        def stop_timer():
            duration = (time.time() - start_time) * 1000  # Convert to milliseconds
            self.timers[name].append(duration)
            self.record_histogram(f"{name}_duration", duration)
        
        return stop_timer
    
    def add_alert_handler(self, handler: Callable[[Alert], None]):
        """Add an alert handler."""
        self.alert_handlers.append(handler)
    
    def set_alert_threshold(self, metric_name: str, level: str, threshold: float):
        """Set an alert threshold for a metric."""
        if metric_name not in self.alert_thresholds:
            self.alert_thresholds[metric_name] = {}
        self.alert_thresholds[metric_name][level] = threshold
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current metric values."""
        return {
            'counters': dict(self.counters),
            'gauges': dict(self.gauges),
            'timestamp': time.time()
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        if not self.system_snapshots:
            return {}
        
        latest = self.system_snapshots[-1]
        return {
            'cpu_percent': latest.cpu_percent,
            'memory_percent': latest.memory_percent,
            'disk_usage': latest.disk_usage,
            'process_count': latest.process_count,
            'thread_count': latest.thread_count,
            'open_files': latest.open_files,
            'connections': latest.connections,
            'network_io': latest.network_io,
            'timestamp': latest.timestamp
        }
    
    def get_recent_alerts(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent alerts."""
        recent_alerts = list(self.alerts)[-limit:]
        return [
            {
                'level': alert.level.value,
                'message': alert.message,
                'timestamp': alert.timestamp,
                'metric_name': alert.metric_name,
                'threshold': alert.threshold,
                'current_value': alert.current_value
            }
            for alert in recent_alerts
        ]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance monitoring summary."""
        return {
            'monitoring': self.is_monitoring,
            'metrics_count': len(self.metrics),
            'counters_count': len(self.counters),
            'gauges_count': len(self.gauges),
            'alerts_count': len(self.alerts),
            'system_snapshots_count': len(self.system_snapshots),
            'monitoring_interval': self.monitoring_interval,
            'persistence_enabled': self.persistence_enabled
        }