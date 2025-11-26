"""
Performance Monitoring Client
===========================

Client for monitoring and tracking performance metrics across the trading system.
Handles metrics collection, aggregation, and reporting for all components.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import UUID, uuid4
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import statistics
import time

from ..client import AsyncSupabaseClient, SupabaseError
from ..models.database_models import *

logger = logging.getLogger(__name__)

class MetricType(str, Enum):
    """Metric type enumeration."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"

class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class MetricData:
    """Metric data structure."""
    name: str
    value: float
    metric_type: MetricType
    tags: Dict[str, str] = field(default_factory=dict)
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

@dataclass
class PerformanceThreshold:
    """Performance threshold configuration."""
    metric_name: str
    warning_threshold: Optional[float] = None
    error_threshold: Optional[float] = None
    critical_threshold: Optional[float] = None
    operator: str = ">"  # >, <, >=, <=, ==, !=

@dataclass
class SystemHealthStatus:
    """System health status."""
    overall_status: str
    component_statuses: Dict[str, str]
    active_alerts: List[Dict[str, Any]]
    performance_summary: Dict[str, Any]
    timestamp: datetime

class PerformanceMonitor:
    """Client for performance monitoring and metrics collection."""
    
    def __init__(self, supabase_client: AsyncSupabaseClient):
        """
        Initialize performance monitor.
        
        Args:
            supabase_client: Async Supabase client instance
        """
        self.supabase = supabase_client
        self._metric_buffer: List[MetricData] = []
        self._buffer_size = 100
        self._flush_interval = 30  # seconds
        self._last_flush = time.time()
        self._thresholds: Dict[str, PerformanceThreshold] = {}
        
    async def record_metric(
        self, 
        metric: MetricData
    ) -> Tuple[bool, Optional[str]]:
        """
        Record a single metric.
        
        Args:
            metric: Metric data to record
            
        Returns:
            Tuple of (success, error_message)
        """
        try:
            # Add to buffer
            self._metric_buffer.append(metric)
            
            # Check if we need to flush
            if (len(self._metric_buffer) >= self._buffer_size or 
                time.time() - self._last_flush >= self._flush_interval):
                await self._flush_metrics()
            
            # Check thresholds for alerts
            await self._check_metric_thresholds(metric)
            
            return True, None
            
        except Exception as e:
            logger.error(f"Failed to record metric: {e}")
            return False, f"Failed to record metric: {str(e)}"
    
    async def record_batch_metrics(
        self, 
        metrics: List[MetricData]
    ) -> Tuple[int, Optional[str]]:
        """
        Record multiple metrics in batch.
        
        Args:
            metrics: List of metrics to record
            
        Returns:
            Tuple of (recorded_count, error_message)
        """
        try:
            recorded_count = 0
            
            for metric in metrics:
                self._metric_buffer.append(metric)
                await self._check_metric_thresholds(metric)
                recorded_count += 1
            
            # Flush if buffer is full
            if len(self._metric_buffer) >= self._buffer_size:
                await self._flush_metrics()
            
            return recorded_count, None
            
        except Exception as e:
            logger.error(f"Failed to record batch metrics: {e}")
            return 0, f"Failed to record batch metrics: {str(e)}"
    
    async def get_metrics(
        self,
        metric_names: Optional[List[str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        tags: Optional[Dict[str, str]] = None,
        limit: int = 1000
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """
        Retrieve metrics with filtering.
        
        Args:
            metric_names: Optional list of metric names to filter
            start_time: Optional start time filter
            end_time: Optional end time filter
            tags: Optional tag filters
            limit: Maximum number of results
            
        Returns:
            Tuple of (metrics_list, error_message)
        """
        try:
            filter_dict = {}
            
            if metric_names:
                filter_dict["name"] = metric_names
            
            if start_time:
                # In a real implementation, you'd handle time range filtering
                pass
                
            if end_time:
                # In a real implementation, you'd handle time range filtering
                pass
            
            metrics = await self.supabase.select(
                "performance_metrics",
                filter_dict=filter_dict,
                order_by="-timestamp",
                limit=limit
            )
            
            # Filter by tags if specified
            if tags:
                filtered_metrics = []
                for metric in metrics:
                    metric_tags = metric.get("tags", {})
                    if all(metric_tags.get(k) == v for k, v in tags.items()):
                        filtered_metrics.append(metric)
                metrics = filtered_metrics
            
            return metrics, None
            
        except Exception as e:
            logger.error(f"Failed to get metrics: {e}")
            return [], f"Failed to get metrics: {str(e)}"
    
    async def calculate_aggregates(
        self,
        metric_name: str,
        aggregation: str = "avg",  # avg, sum, min, max, count
        window_minutes: int = 60,
        group_by_tags: Optional[List[str]] = None
    ) -> Tuple[Dict[str, Any], Optional[str]]:
        """
        Calculate metric aggregates over time windows.
        
        Args:
            metric_name: Name of metric to aggregate
            aggregation: Type of aggregation (avg, sum, min, max, count)
            window_minutes: Time window in minutes
            group_by_tags: Optional tags to group by
            
        Returns:
            Tuple of (aggregation_result, error_message)
        """
        try:
            # Get metrics for the window
            start_time = datetime.utcnow() - timedelta(minutes=window_minutes)
            
            metrics = await self.supabase.select(
                "performance_metrics",
                filter_dict={"name": metric_name},
                order_by="-timestamp",
                limit=10000  # Large limit for aggregation
            )
            
            # Filter by time window (simplified)
            windowed_metrics = [
                m for m in metrics 
                if datetime.fromisoformat(m["timestamp"].replace("Z", "+00:00")) >= start_time
            ]
            
            if not windowed_metrics:
                return {"result": None, "count": 0}, None
            
            # Group by tags if specified
            groups = {"all": windowed_metrics}
            if group_by_tags:
                groups = {}
                for metric in windowed_metrics:
                    tags = metric.get("tags", {})
                    key = tuple(tags.get(tag, "") for tag in group_by_tags)
                    if key not in groups:
                        groups[key] = []
                    groups[key].append(metric)
            
            # Calculate aggregations
            results = {}
            for group_key, group_metrics in groups.items():
                values = [float(m["value"]) for m in group_metrics]
                
                if aggregation == "avg":
                    result = statistics.mean(values)
                elif aggregation == "sum":
                    result = sum(values)
                elif aggregation == "min":
                    result = min(values)
                elif aggregation == "max":
                    result = max(values)
                elif aggregation == "count":
                    result = len(values)
                else:
                    result = statistics.mean(values)  # Default to average
                
                results[str(group_key)] = {
                    "value": result,
                    "count": len(values),
                    "window_minutes": window_minutes
                }
            
            return results, None
            
        except Exception as e:
            logger.error(f"Failed to calculate aggregates: {e}")
            return {}, f"Failed to calculate aggregates: {str(e)}"
    
    async def get_system_health(self) -> Tuple[SystemHealthStatus, Optional[str]]:
        """
        Get overall system health status.
        
        Returns:
            Tuple of (health_status, error_message)
        """
        try:
            # Get recent metrics for health check
            recent_metrics = await self.supabase.select(
                "performance_metrics",
                order_by="-timestamp",
                limit=1000
            )
            
            # Get active alerts
            active_alerts = await self.supabase.select(
                "performance_alerts",
                filter_dict={"status": "active"},
                order_by="-created_at",
                limit=50
            )
            
            # Calculate component statuses
            component_statuses = await self._calculate_component_health(recent_metrics)
            
            # Calculate overall status
            overall_status = "healthy"
            if any(status == "critical" for status in component_statuses.values()):
                overall_status = "critical"
            elif any(status == "warning" for status in component_statuses.values()):
                overall_status = "warning"
            elif any(status == "error" for status in component_statuses.values()):
                overall_status = "error"
            
            # Performance summary
            performance_summary = await self._calculate_performance_summary(recent_metrics)
            
            health_status = SystemHealthStatus(
                overall_status=overall_status,
                component_statuses=component_statuses,
                active_alerts=active_alerts,
                performance_summary=performance_summary,
                timestamp=datetime.utcnow()
            )
            
            return health_status, None
            
        except Exception as e:
            logger.error(f"Failed to get system health: {e}")
            return None, f"Failed to get system health: {str(e)}"
    
    async def create_alert(
        self,
        metric_name: str,
        severity: AlertSeverity,
        message: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Create a performance alert.
        
        Args:
            metric_name: Associated metric name
            severity: Alert severity level
            message: Alert message
            metadata: Additional alert metadata
            
        Returns:
            Tuple of (alert_data, error_message)
        """
        try:
            alert_data = {
                "id": str(uuid4()),
                "metric_name": metric_name,
                "severity": severity.value,
                "message": message,
                "metadata": metadata or {},
                "status": "active",
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }
            
            result = await self.supabase.insert("performance_alerts", alert_data)
            
            logger.warning(f"Performance alert created: {metric_name} - {message}")
            return result[0], None
            
        except Exception as e:
            logger.error(f"Failed to create alert: {e}")
            return None, f"Failed to create alert: {str(e)}"
    
    async def resolve_alert(
        self,
        alert_id: str,
        resolution_note: Optional[str] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Resolve a performance alert.
        
        Args:
            alert_id: Alert ID to resolve
            resolution_note: Optional resolution note
            
        Returns:
            Tuple of (success, error_message)
        """
        try:
            update_data = {
                "status": "resolved",
                "resolved_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }
            
            if resolution_note:
                update_data["resolution_note"] = resolution_note
            
            await self.supabase.update(
                "performance_alerts",
                update_data,
                {"id": alert_id}
            )
            
            logger.info(f"Alert resolved: {alert_id}")
            return True, None
            
        except Exception as e:
            logger.error(f"Failed to resolve alert: {e}")
            return False, f"Failed to resolve alert: {str(e)}"
    
    async def set_threshold(
        self,
        threshold: PerformanceThreshold
    ) -> Tuple[bool, Optional[str]]:
        """
        Set performance threshold for monitoring.
        
        Args:
            threshold: Threshold configuration
            
        Returns:
            Tuple of (success, error_message)
        """
        try:
            self._thresholds[threshold.metric_name] = threshold
            
            # Store in database
            threshold_data = {
                "id": str(uuid4()),
                "metric_name": threshold.metric_name,
                "warning_threshold": threshold.warning_threshold,
                "error_threshold": threshold.error_threshold,
                "critical_threshold": threshold.critical_threshold,
                "operator": threshold.operator,
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }
            
            # Upsert threshold
            await self.supabase.upsert(
                "performance_thresholds",
                threshold_data,
                on_conflict="metric_name"
            )
            
            logger.info(f"Threshold set for metric: {threshold.metric_name}")
            return True, None
            
        except Exception as e:
            logger.error(f"Failed to set threshold: {e}")
            return False, f"Failed to set threshold: {str(e)}"
    
    async def start_monitoring(self):
        """Start the background monitoring task."""
        try:
            # Load existing thresholds
            await self._load_thresholds()
            
            # Start flush timer
            asyncio.create_task(self._periodic_flush())
            
            logger.info("Performance monitoring started")
            
        except Exception as e:
            logger.error(f"Failed to start monitoring: {e}")
    
    async def stop_monitoring(self):
        """Stop monitoring and flush remaining metrics."""
        try:
            await self._flush_metrics()
            logger.info("Performance monitoring stopped")
            
        except Exception as e:
            logger.error(f"Failed to stop monitoring: {e}")
    
    async def _flush_metrics(self):
        """Flush buffered metrics to database."""
        if not self._metric_buffer:
            return
            
        try:
            metrics_data = []
            for metric in self._metric_buffer:
                metrics_data.append({
                    "id": str(uuid4()),
                    "name": metric.name,
                    "value": metric.value,
                    "metric_type": metric.metric_type.value,
                    "tags": metric.tags,
                    "timestamp": metric.timestamp.isoformat()
                })
            
            await self.supabase.bulk_insert("performance_metrics", metrics_data)
            
            logger.debug(f"Flushed {len(metrics_data)} metrics to database")
            self._metric_buffer.clear()
            self._last_flush = time.time()
            
        except Exception as e:
            logger.error(f"Failed to flush metrics: {e}")
    
    async def _periodic_flush(self):
        """Periodic flush task."""
        while True:
            try:
                await asyncio.sleep(self._flush_interval)
                await self._flush_metrics()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic flush: {e}")
    
    async def _load_thresholds(self):
        """Load existing thresholds from database."""
        try:
            thresholds = await self.supabase.select("performance_thresholds")
            
            for threshold_data in thresholds:
                threshold = PerformanceThreshold(
                    metric_name=threshold_data["metric_name"],
                    warning_threshold=threshold_data.get("warning_threshold"),
                    error_threshold=threshold_data.get("error_threshold"),
                    critical_threshold=threshold_data.get("critical_threshold"),
                    operator=threshold_data.get("operator", ">")
                )
                self._thresholds[threshold.metric_name] = threshold
            
            logger.info(f"Loaded {len(self._thresholds)} thresholds")
            
        except Exception as e:
            logger.error(f"Failed to load thresholds: {e}")
    
    async def _check_metric_thresholds(self, metric: MetricData):
        """Check if metric exceeds thresholds."""
        threshold = self._thresholds.get(metric.name)
        if not threshold:
            return
            
        try:
            value = metric.value
            
            # Check critical threshold first
            if threshold.critical_threshold is not None:
                if self._evaluate_threshold(value, threshold.critical_threshold, threshold.operator):
                    await self.create_alert(
                        metric.name,
                        AlertSeverity.CRITICAL,
                        f"Critical threshold exceeded: {value} {threshold.operator} {threshold.critical_threshold}",
                        {"metric_value": value, "threshold": threshold.critical_threshold}
                    )
                    return
            
            # Check error threshold
            if threshold.error_threshold is not None:
                if self._evaluate_threshold(value, threshold.error_threshold, threshold.operator):
                    await self.create_alert(
                        metric.name,
                        AlertSeverity.ERROR,
                        f"Error threshold exceeded: {value} {threshold.operator} {threshold.error_threshold}",
                        {"metric_value": value, "threshold": threshold.error_threshold}
                    )
                    return
            
            # Check warning threshold
            if threshold.warning_threshold is not None:
                if self._evaluate_threshold(value, threshold.warning_threshold, threshold.operator):
                    await self.create_alert(
                        metric.name,
                        AlertSeverity.WARNING,
                        f"Warning threshold exceeded: {value} {threshold.operator} {threshold.warning_threshold}",
                        {"metric_value": value, "threshold": threshold.warning_threshold}
                    )
            
        except Exception as e:
            logger.error(f"Failed to check thresholds for {metric.name}: {e}")
    
    def _evaluate_threshold(self, value: float, threshold: float, operator: str) -> bool:
        """Evaluate if value meets threshold condition."""
        if operator == ">":
            return value > threshold
        elif operator == "<":
            return value < threshold
        elif operator == ">=":
            return value >= threshold
        elif operator == "<=":
            return value <= threshold
        elif operator == "==":
            return value == threshold
        elif operator == "!=":
            return value != threshold
        else:
            return value > threshold  # Default
    
    async def _calculate_component_health(self, metrics: List[Dict[str, Any]]) -> Dict[str, str]:
        """Calculate health status for each component."""
        component_health = {}
        
        # Group metrics by component (using tags)
        components = set()
        for metric in metrics:
            tags = metric.get("tags", {})
            component = tags.get("component", "unknown")
            components.add(component)
        
        for component in components:
            # For each component, check if there are any error metrics
            component_metrics = [
                m for m in metrics 
                if m.get("tags", {}).get("component") == component
            ]
            
            # Simple health logic based on error rates
            error_count = sum(1 for m in component_metrics if "error" in m["name"].lower())
            total_count = len(component_metrics)
            
            if total_count == 0:
                component_health[component] = "unknown"
            elif error_count / total_count > 0.1:  # More than 10% errors
                component_health[component] = "critical"
            elif error_count / total_count > 0.05:  # More than 5% errors
                component_health[component] = "warning"
            else:
                component_health[component] = "healthy"
        
        return component_health
    
    async def _calculate_performance_summary(self, metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate performance summary from recent metrics."""
        if not metrics:
            return {}
        
        # Calculate basic statistics
        latency_metrics = [m for m in metrics if "latency" in m["name"].lower()]
        throughput_metrics = [m for m in metrics if "throughput" in m["name"].lower()]
        error_metrics = [m for m in metrics if "error" in m["name"].lower()]
        
        summary = {
            "total_metrics": len(metrics),
            "metric_types": len(set(m["name"] for m in metrics)),
            "time_range_minutes": 60  # Assuming 1 hour of data
        }
        
        if latency_metrics:
            latencies = [float(m["value"]) for m in latency_metrics]
            summary["avg_latency_ms"] = statistics.mean(latencies)
            summary["max_latency_ms"] = max(latencies)
        
        if throughput_metrics:
            throughputs = [float(m["value"]) for m in throughput_metrics]
            summary["avg_throughput"] = statistics.mean(throughputs)
        
        if error_metrics:
            errors = [float(m["value"]) for m in error_metrics]
            summary["total_errors"] = sum(errors)
            summary["error_rate"] = sum(errors) / len(metrics) if metrics else 0
        
        return summary