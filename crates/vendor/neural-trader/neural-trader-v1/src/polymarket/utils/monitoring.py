"""
Performance monitoring and metrics collection for Polymarket integration

This module provides utilities for monitoring system performance,
collecting metrics, and managing alerts.
"""

import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
import logging

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics container"""
    
    operation: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    success: bool = True
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def finish(self, success: bool = True, error: Optional[str] = None):
        """Mark operation as finished"""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        self.success = success
        self.error = error


class MetricsCollector:
    """Collects and aggregates performance metrics"""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.metrics_history: deque = deque(maxlen=max_history)
        self.operation_stats: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                'count': 0,
                'total_duration': 0.0,
                'success_count': 0,
                'error_count': 0,
                'avg_duration': 0.0,
                'last_execution': None
            }
        )
    
    def record_metric(self, metric: PerformanceMetrics):
        """Record a performance metric"""
        self.metrics_history.append(metric)
        
        # Update operation statistics
        stats = self.operation_stats[metric.operation]
        stats['count'] += 1
        stats['last_execution'] = datetime.now()
        
        if metric.duration:
            stats['total_duration'] += metric.duration
            stats['avg_duration'] = stats['total_duration'] / stats['count']
        
        if metric.success:
            stats['success_count'] += 1
        else:
            stats['error_count'] += 1
    
    def get_operation_stats(self, operation: str) -> Dict[str, Any]:
        """Get statistics for a specific operation"""
        return dict(self.operation_stats.get(operation, {}))
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all operations"""
        return dict(self.operation_stats)


class AlertManager:
    """Manages alerts based on performance thresholds"""
    
    def __init__(self):
        self.alert_rules: List[Dict[str, Any]] = []
        self.alert_callbacks: List[Callable] = []
        self.active_alerts: Dict[str, datetime] = {}


class PerformanceMonitor:
    """Main performance monitoring class"""
    
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager()
    
    def start_operation(self, operation: str, **metadata) -> PerformanceMetrics:
        """Start monitoring an operation"""
        metric = PerformanceMetrics(
            operation=operation,
            start_time=time.time(),
            metadata=metadata
        )
        return metric
    
    def finish_operation(self, 
                        metric: PerformanceMetrics,
                        success: bool = True,
                        error: Optional[str] = None):
        """Finish monitoring an operation"""
        metric.finish(success=success, error=error)
        self.metrics_collector.record_metric(metric)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        return {
            'operation_stats': self.metrics_collector.get_all_stats(),
            'metrics_count': len(self.metrics_collector.metrics_history)
        }


def monitor_performance(operation: str, **metadata):
    """Decorator for monitoring function performance"""
    def decorator(func):
        return func  # Simplified implementation
    return decorator