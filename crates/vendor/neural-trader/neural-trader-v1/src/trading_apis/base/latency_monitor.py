"""
Latency Monitor for Trading APIs

Provides microsecond-precision latency monitoring, profiling, and alerting
for ultra-low latency trading operations.
"""

import time
import asyncio
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, defaultdict
import numpy as np
import threading
from enum import Enum
import json
import statistics
from concurrent.futures import ThreadPoolExecutor


class LatencyLevel(Enum):
    """Latency threshold levels"""
    EXCELLENT = "excellent"      # < 1ms
    GOOD = "good"               # 1-5ms
    ACCEPTABLE = "acceptable"   # 5-20ms
    WARNING = "warning"         # 20-100ms
    CRITICAL = "critical"       # > 100ms


@dataclass
class LatencyMeasurement:
    """Single latency measurement"""
    operation: str
    start_time: float
    end_time: float
    latency_us: float  # microseconds
    latency_ms: float  # milliseconds
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def start(cls, operation: str, metadata: Optional[Dict[str, Any]] = None) -> 'LatencyMeasurement':
        """Start a new measurement"""
        return cls(
            operation=operation,
            start_time=time.perf_counter(),
            end_time=0.0,
            latency_us=0.0,
            latency_ms=0.0,
            metadata=metadata or {}
        )
    
    def stop(self) -> 'LatencyMeasurement':
        """Stop the measurement and calculate latency"""
        self.end_time = time.perf_counter()
        self.latency_us = (self.end_time - self.start_time) * 1_000_000
        self.latency_ms = self.latency_us / 1000
        return self


@dataclass
class LatencyProfile:
    """Latency profile for a specific operation type"""
    operation: str
    measurements: deque = field(default_factory=lambda: deque(maxlen=10000))
    
    def add_measurement(self, measurement: LatencyMeasurement) -> None:
        """Add a measurement to the profile"""
        self.measurements.append(measurement)
    
    def get_stats(self) -> Dict[str, float]:
        """Calculate statistics for this profile"""
        if not self.measurements:
            return {
                'count': 0,
                'mean_ms': 0.0,
                'median_ms': 0.0,
                'std_ms': 0.0,
                'min_ms': 0.0,
                'max_ms': 0.0,
                'p50_ms': 0.0,
                'p95_ms': 0.0,
                'p99_ms': 0.0,
                'p999_ms': 0.0
            }
        
        latencies_ms = [m.latency_ms for m in self.measurements]
        latencies_array = np.array(latencies_ms)
        
        return {
            'count': len(latencies_ms),
            'mean_ms': np.mean(latencies_array),
            'median_ms': np.median(latencies_array),
            'std_ms': np.std(latencies_array),
            'min_ms': np.min(latencies_array),
            'max_ms': np.max(latencies_array),
            'p50_ms': np.percentile(latencies_array, 50),
            'p95_ms': np.percentile(latencies_array, 95),
            'p99_ms': np.percentile(latencies_array, 99),
            'p999_ms': np.percentile(latencies_array, 99.9)
        }


@dataclass
class LatencyAlert:
    """Alert for latency threshold violations"""
    timestamp: datetime
    operation: str
    latency_ms: float
    threshold_ms: float
    level: LatencyLevel
    message: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class LatencyMonitor:
    """
    High-precision latency monitoring system for trading operations.
    
    Features:
    - Microsecond precision timing
    - Operation-specific profiling
    - Real-time alerting
    - Statistical analysis
    - Trend detection
    - Export capabilities
    """
    
    def __init__(self, 
                 alert_thresholds: Optional[Dict[str, float]] = None,
                 enable_auto_profiling: bool = True):
        """
        Initialize latency monitor.
        
        Args:
            alert_thresholds: Custom alert thresholds by operation type
            enable_auto_profiling: Automatically profile all operations
        """
        self.profiles: Dict[str, LatencyProfile] = {}
        self.alerts: deque = deque(maxlen=1000)
        self.enable_auto_profiling = enable_auto_profiling
        
        # Default thresholds in milliseconds
        self.default_thresholds = {
            'order_placement': 5.0,
            'order_cancel': 5.0,
            'market_data': 2.0,
            'account_query': 10.0,
            'connection': 50.0
        }
        
        self.alert_thresholds = alert_thresholds or self.default_thresholds
        
        # Alert callbacks
        self.alert_callbacks: List[Callable[[LatencyAlert], None]] = []
        
        # Trend detection
        self.trend_window = 100  # measurements
        self.trend_threshold = 1.5  # 50% increase triggers alert
        
        # Thread safety
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=2)
        
        # Continuous monitoring
        self._monitoring_task: Optional[asyncio.Task] = None
        self._running = False
    
    def measure(self, operation: str, metadata: Optional[Dict[str, Any]] = None) -> LatencyMeasurement:
        """
        Start a latency measurement.
        
        Args:
            operation: Name of the operation being measured
            metadata: Additional context for the measurement
            
        Returns:
            LatencyMeasurement object (call .stop() when operation completes)
        """
        return LatencyMeasurement.start(operation, metadata)
    
    def record(self, measurement: LatencyMeasurement) -> None:
        """
        Record a completed measurement.
        
        Args:
            measurement: Completed LatencyMeasurement
        """
        with self._lock:
            # Get or create profile
            if measurement.operation not in self.profiles:
                self.profiles[measurement.operation] = LatencyProfile(measurement.operation)
            
            profile = self.profiles[measurement.operation]
            profile.add_measurement(measurement)
            
            # Check thresholds
            self._check_thresholds(measurement)
            
            # Check trends
            self._check_trends(profile)
    
    def _check_thresholds(self, measurement: LatencyMeasurement) -> None:
        """Check if measurement violates thresholds."""
        threshold = self.alert_thresholds.get(
            measurement.operation,
            self.default_thresholds.get('order_placement', 10.0)
        )
        
        if measurement.latency_ms > threshold:
            level = self._get_latency_level(measurement.latency_ms)
            
            alert = LatencyAlert(
                timestamp=datetime.now(),
                operation=measurement.operation,
                latency_ms=measurement.latency_ms,
                threshold_ms=threshold,
                level=level,
                message=f"Latency {measurement.latency_ms:.2f}ms exceeds threshold {threshold}ms",
                metadata=measurement.metadata
            )
            
            self.alerts.append(alert)
            
            # Trigger callbacks
            for callback in self.alert_callbacks:
                self._executor.submit(callback, alert)
    
    def _check_trends(self, profile: LatencyProfile) -> None:
        """Check for latency trends."""
        if len(profile.measurements) < self.trend_window:
            return
        
        # Get recent and historical windows
        recent = list(profile.measurements)[-self.trend_window//2:]
        historical = list(profile.measurements)[-self.trend_window:-self.trend_window//2]
        
        recent_mean = statistics.mean(m.latency_ms for m in recent)
        historical_mean = statistics.mean(m.latency_ms for m in historical)
        
        # Check if trend is increasing
        if recent_mean > historical_mean * self.trend_threshold:
            alert = LatencyAlert(
                timestamp=datetime.now(),
                operation=profile.operation,
                latency_ms=recent_mean,
                threshold_ms=historical_mean * self.trend_threshold,
                level=LatencyLevel.WARNING,
                message=f"Latency trend increasing: {recent_mean:.2f}ms vs {historical_mean:.2f}ms historical",
                metadata={'trend_ratio': recent_mean / historical_mean}
            )
            
            self.alerts.append(alert)
            
            # Trigger callbacks
            for callback in self.alert_callbacks:
                self._executor.submit(callback, alert)
    
    def _get_latency_level(self, latency_ms: float) -> LatencyLevel:
        """Determine latency level based on value."""
        if latency_ms < 1:
            return LatencyLevel.EXCELLENT
        elif latency_ms < 5:
            return LatencyLevel.GOOD
        elif latency_ms < 20:
            return LatencyLevel.ACCEPTABLE
        elif latency_ms < 100:
            return LatencyLevel.WARNING
        else:
            return LatencyLevel.CRITICAL
    
    def add_alert_callback(self, callback: Callable[[LatencyAlert], None]) -> None:
        """Add a callback for latency alerts."""
        self.alert_callbacks.append(callback)
    
    def get_profile_stats(self, operation: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics for operation profiles.
        
        Args:
            operation: Specific operation or None for all
            
        Returns:
            Dictionary of statistics
        """
        with self._lock:
            if operation:
                if operation in self.profiles:
                    return {operation: self.profiles[operation].get_stats()}
                else:
                    return {}
            else:
                return {
                    op: profile.get_stats()
                    for op, profile in self.profiles.items()
                }
    
    def get_recent_measurements(self, 
                               operation: Optional[str] = None,
                               limit: int = 100) -> List[LatencyMeasurement]:
        """Get recent measurements."""
        with self._lock:
            if operation and operation in self.profiles:
                measurements = list(self.profiles[operation].measurements)
                return measurements[-limit:]
            else:
                # Combine all measurements
                all_measurements = []
                for profile in self.profiles.values():
                    all_measurements.extend(list(profile.measurements))
                
                # Sort by time and return most recent
                all_measurements.sort(key=lambda m: m.end_time)
                return all_measurements[-limit:]
    
    def get_alerts(self, 
                   since: Optional[datetime] = None,
                   level: Optional[LatencyLevel] = None) -> List[LatencyAlert]:
        """Get alerts with optional filtering."""
        alerts = list(self.alerts)
        
        if since:
            alerts = [a for a in alerts if a.timestamp >= since]
        
        if level:
            alerts = [a for a in alerts if a.level == level]
        
        return alerts
    
    def export_metrics(self, filepath: str) -> None:
        """Export metrics to JSON file."""
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'profiles': self.get_profile_stats(),
            'recent_alerts': [
                {
                    'timestamp': alert.timestamp.isoformat(),
                    'operation': alert.operation,
                    'latency_ms': alert.latency_ms,
                    'threshold_ms': alert.threshold_ms,
                    'level': alert.level.value,
                    'message': alert.message
                }
                for alert in self.get_alerts(
                    since=datetime.now() - timedelta(hours=1)
                )
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def get_summary_report(self) -> Dict[str, Any]:
        """Generate a summary report of latency metrics."""
        with self._lock:
            total_measurements = sum(
                len(p.measurements) for p in self.profiles.values()
            )
            
            # Calculate overall statistics
            all_latencies = []
            for profile in self.profiles.values():
                all_latencies.extend([m.latency_ms for m in profile.measurements])
            
            if all_latencies:
                overall_stats = {
                    'mean_ms': statistics.mean(all_latencies),
                    'median_ms': statistics.median(all_latencies),
                    'p95_ms': np.percentile(all_latencies, 95),
                    'p99_ms': np.percentile(all_latencies, 99)
                }
            else:
                overall_stats = {}
            
            # Count alerts by level
            alert_counts = defaultdict(int)
            for alert in self.alerts:
                alert_counts[alert.level.value] += 1
            
            return {
                'total_measurements': total_measurements,
                'operations_tracked': len(self.profiles),
                'overall_stats': overall_stats,
                'operation_stats': self.get_profile_stats(),
                'alert_counts': dict(alert_counts),
                'recent_alerts': len([
                    a for a in self.alerts 
                    if a.timestamp > datetime.now() - timedelta(minutes=5)
                ])
            }
    
    async def start_monitoring(self) -> None:
        """Start continuous monitoring tasks."""
        self._running = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
    
    async def stop_monitoring(self) -> None:
        """Stop continuous monitoring."""
        self._running = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
    
    async def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while self._running:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                # Generate periodic summary
                summary = self.get_summary_report()
                
                # Check for concerning patterns
                if summary.get('recent_alerts', 0) > 10:
                    alert = LatencyAlert(
                        timestamp=datetime.now(),
                        operation='system',
                        latency_ms=0,
                        threshold_ms=0,
                        level=LatencyLevel.WARNING,
                        message=f"High alert rate: {summary['recent_alerts']} alerts in last 5 minutes",
                        metadata={'summary': summary}
                    )
                    
                    for callback in self.alert_callbacks:
                        self._executor.submit(callback, alert)
                        
            except Exception as e:
                # Log error but continue monitoring
                pass
    
    def __enter__(self):
        """Context manager entry."""
        asyncio.create_task(self.start_monitoring())
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        asyncio.create_task(self.stop_monitoring())
        self._executor.shutdown(wait=True)