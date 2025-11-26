"""
Monitoring and Logging System for Neural Forecasting.

This module provides comprehensive monitoring, logging, and observability
for the NHITS forecasting system including performance metrics, alerting,
and real-time monitoring capabilities.
"""

import asyncio
import logging
import json
import time
import psutil
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
import threading
from pathlib import Path

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class PerformanceMetric:
    """Performance metric data structure."""
    timestamp: str
    metric_name: str
    value: float
    unit: str
    component: str
    metadata: Dict[str, Any] = None


@dataclass
class SystemAlert:
    """System alert data structure."""
    timestamp: str
    alert_id: str
    severity: str  # 'info', 'warning', 'error', 'critical'
    component: str
    message: str
    metric_value: Optional[float] = None
    threshold: Optional[float] = None
    resolved: bool = False
    resolution_time: Optional[str] = None


class NeuralForecastMonitor:
    """
    Comprehensive monitoring and observability system for neural forecasting.
    
    Features:
    - Real-time performance monitoring
    - Resource utilization tracking
    - Custom metrics collection
    - Alerting and threshold monitoring
    - Log aggregation and analysis
    - Export capabilities for external monitoring systems
    """
    
    def __init__(
        self,
        enable_performance_monitoring: bool = True,
        enable_resource_monitoring: bool = True,
        enable_model_monitoring: bool = True,
        enable_alerting: bool = True,
        metrics_retention_hours: int = 24,
        log_level: str = 'INFO',
        log_file_path: Optional[str] = None,
        export_interval_seconds: int = 60
    ):
        """
        Initialize monitoring system.
        
        Args:
            enable_performance_monitoring: Enable performance metric collection
            enable_resource_monitoring: Enable system resource monitoring
            enable_model_monitoring: Enable model-specific monitoring
            enable_alerting: Enable alerting system
            metrics_retention_hours: Hours to retain metrics in memory
            log_level: Logging level
            log_file_path: Optional log file path
            export_interval_seconds: Interval for metric exports
        """
        self.enable_performance_monitoring = enable_performance_monitoring
        self.enable_resource_monitoring = enable_resource_monitoring
        self.enable_model_monitoring = enable_model_monitoring
        self.enable_alerting = enable_alerting
        self.metrics_retention_hours = metrics_retention_hours
        self.export_interval_seconds = export_interval_seconds
        
        # Metrics storage
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.alerts: List[SystemAlert] = []
        self.alert_thresholds: Dict[str, Dict[str, float]] = {}
        
        # Performance tracking
        self.operation_times: Dict[str, List[float]] = defaultdict(list)
        self.operation_counts: Dict[str, int] = defaultdict(int)
        self.error_counts: Dict[str, int] = defaultdict(int)
        
        # Resource monitoring
        self.resource_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Model monitoring
        self.model_metrics: Dict[str, Dict] = {}
        self.prediction_history: deque = deque(maxlen=1000)
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_thread = None
        self.last_cleanup = datetime.now()
        
        # Logging setup
        self.logger = self._setup_logging(log_level, log_file_path)
        
        # Default alert thresholds
        self._setup_default_thresholds()
        
        self.logger.info("Neural Forecast Monitor initialized")
    
    def _setup_logging(self, log_level: str, log_file_path: Optional[str]) -> logging.Logger:
        """Setup comprehensive logging configuration."""
        logger = logging.getLogger('neural_forecast_monitor')
        logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear any existing handlers
        logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # File handler if specified
        if log_file_path:
            try:
                file_path = Path(log_file_path)
                file_path.parent.mkdir(parents=True, exist_ok=True)
                
                file_handler = logging.FileHandler(file_path)
                file_formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(component)s - %(message)s'
                )
                file_handler.setFormatter(file_formatter)
                logger.addHandler(file_handler)
                
            except Exception as e:
                logger.warning(f"Failed to setup file logging: {str(e)}")
        
        return logger
    
    def _setup_default_thresholds(self):
        """Setup default alert thresholds."""
        self.alert_thresholds = {
            'cpu_usage': {'warning': 80.0, 'critical': 95.0},
            'memory_usage': {'warning': 85.0, 'critical': 95.0},
            'gpu_memory_usage': {'warning': 90.0, 'critical': 98.0},
            'prediction_latency': {'warning': 5.0, 'critical': 10.0},
            'error_rate': {'warning': 0.05, 'critical': 0.1},
            'model_accuracy': {'warning': 0.7, 'critical': 0.5}  # Lower is worse
        }
    
    def start_monitoring(self):
        """Start continuous monitoring in background thread."""
        if self.monitoring_active:
            self.logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        self.logger.info("Monitoring started")
    
    def stop_monitoring(self):
        """Stop continuous monitoring."""
        self.monitoring_active = False
        
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5.0)
        
        self.logger.info("Monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop running in background thread."""
        while self.monitoring_active:
            try:
                # Collect system metrics
                if self.enable_resource_monitoring:
                    self._collect_system_metrics()
                
                # Check alert thresholds
                if self.enable_alerting:
                    self._check_alert_thresholds()
                
                # Cleanup old metrics
                self._cleanup_old_metrics()
                
                # Sleep until next collection
                time.sleep(self.export_interval_seconds)
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {str(e)}")
                time.sleep(10)  # Longer sleep on error
    
    def _collect_system_metrics(self):
        """Collect system resource metrics."""
        try:
            timestamp = datetime.now().isoformat()
            
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            self._record_metric('cpu_usage', cpu_percent, '%', 'system', timestamp)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            self._record_metric('memory_usage', memory.percent, '%', 'system', timestamp)
            self._record_metric('memory_available_gb', memory.available / (1024**3), 'GB', 'system', timestamp)
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            self._record_metric('disk_usage', (disk.used / disk.total) * 100, '%', 'system', timestamp)
            
            # GPU metrics if available
            if TORCH_AVAILABLE and torch.cuda.is_available():
                try:
                    gpu_memory_allocated = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100
                    self._record_metric('gpu_memory_usage', gpu_memory_allocated, '%', 'gpu', timestamp)
                    
                    gpu_utilization = self._get_gpu_utilization()
                    if gpu_utilization is not None:
                        self._record_metric('gpu_utilization', gpu_utilization, '%', 'gpu', timestamp)
                        
                except Exception as e:
                    self.logger.debug(f"GPU metrics collection failed: {str(e)}")
            
            # Process-specific metrics
            process = psutil.Process()
            self._record_metric('process_memory_mb', process.memory_info().rss / (1024**2), 'MB', 'process', timestamp)
            self._record_metric('process_cpu_percent', process.cpu_percent(), '%', 'process', timestamp)
            
        except Exception as e:
            self.logger.error(f"System metrics collection failed: {str(e)}")
    
    def _get_gpu_utilization(self) -> Optional[float]:
        """Get GPU utilization percentage."""
        try:
            # Try nvidia-ml-py3 first
            try:
                import nvidia_ml_py3 as nvml
                nvml.nvmlInit()
                handle = nvml.nvmlDeviceGetHandleByIndex(0)
                util = nvml.nvmlDeviceGetUtilizationRates(handle)
                return float(util.gpu)
            except ImportError:
                pass
            
            # Fallback: try nvidia-smi
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                return float(result.stdout.strip())
                
        except Exception:
            pass
        
        return None
    
    def _record_metric(self, name: str, value: float, unit: str, component: str, timestamp: str):
        """Record a performance metric."""
        metric = PerformanceMetric(
            timestamp=timestamp,
            metric_name=name,
            value=value,
            unit=unit,
            component=component
        )
        
        self.metrics[name].append(metric)
        self.resource_metrics[name].append(value)
    
    def _check_alert_thresholds(self):
        """Check metrics against alert thresholds."""
        try:
            for metric_name, thresholds in self.alert_thresholds.items():
                if metric_name in self.resource_metrics and self.resource_metrics[metric_name]:
                    current_value = self.resource_metrics[metric_name][-1]
                    
                    # Check critical threshold
                    if 'critical' in thresholds and current_value >= thresholds['critical']:
                        self._create_alert(
                            metric_name, 'critical', current_value, thresholds['critical']
                        )
                    # Check warning threshold
                    elif 'warning' in thresholds and current_value >= thresholds['warning']:
                        self._create_alert(
                            metric_name, 'warning', current_value, thresholds['warning']
                        )
                        
        except Exception as e:
            self.logger.error(f"Alert threshold checking failed: {str(e)}")
    
    def _create_alert(self, metric_name: str, severity: str, value: float, threshold: float):
        """Create and record an alert."""
        try:
            alert_id = f"{metric_name}_{severity}_{int(time.time())}"
            
            # Check if similar alert already exists and is unresolved
            existing_alert = None
            for alert in self.alerts:
                if (alert.component == metric_name and 
                    alert.severity == severity and 
                    not alert.resolved):
                    existing_alert = alert
                    break
            
            if existing_alert:
                return  # Don't create duplicate alerts
            
            alert = SystemAlert(
                timestamp=datetime.now().isoformat(),
                alert_id=alert_id,
                severity=severity,
                component=metric_name,
                message=f"{metric_name} {severity}: {value:.2f} exceeds threshold {threshold:.2f}",
                metric_value=value,
                threshold=threshold,
                resolved=False
            )
            
            self.alerts.append(alert)
            
            # Log based on severity
            if severity == 'critical':
                self.logger.critical(alert.message)
            elif severity == 'warning':
                self.logger.warning(alert.message)
            else:
                self.logger.info(alert.message)
                
        except Exception as e:
            self.logger.error(f"Alert creation failed: {str(e)}")
    
    def _cleanup_old_metrics(self):
        """Clean up old metrics to manage memory usage."""
        try:
            if datetime.now() - self.last_cleanup < timedelta(hours=1):
                return  # Only cleanup once per hour
            
            cutoff_time = datetime.now() - timedelta(hours=self.metrics_retention_hours)
            
            # Clean up metrics
            for metric_name, metric_deque in self.metrics.items():
                # Remove metrics older than retention period
                while (metric_deque and 
                       datetime.fromisoformat(metric_deque[0].timestamp) < cutoff_time):
                    metric_deque.popleft()
            
            # Clean up alerts older than 7 days
            alert_cutoff = datetime.now() - timedelta(days=7)
            self.alerts = [
                alert for alert in self.alerts
                if datetime.fromisoformat(alert.timestamp) > alert_cutoff
            ]
            
            self.last_cleanup = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Metrics cleanup failed: {str(e)}")
    
    def record_operation(self, operation_name: str, duration: float, success: bool = True):
        """Record operation performance metrics."""
        try:
            timestamp = datetime.now().isoformat()
            
            # Record duration
            self.operation_times[operation_name].append(duration)
            self._record_metric(f"{operation_name}_duration", duration, 'seconds', 'operation', timestamp)
            
            # Record count
            self.operation_counts[operation_name] += 1
            
            # Record error if failed
            if not success:
                self.error_counts[operation_name] += 1
                self._record_metric(f"{operation_name}_error", 1, 'count', 'operation', timestamp)
            
            # Calculate and record error rate
            total_ops = self.operation_counts[operation_name]
            error_rate = self.error_counts[operation_name] / total_ops if total_ops > 0 else 0
            self._record_metric(f"{operation_name}_error_rate", error_rate, 'ratio', 'operation', timestamp)
            
            # Keep operation times manageable
            if len(self.operation_times[operation_name]) > 1000:
                self.operation_times[operation_name] = self.operation_times[operation_name][-1000:]
                
        except Exception as e:
            self.logger.error(f"Operation recording failed: {str(e)}")
    
    def record_prediction(
        self,
        model_name: str,
        input_data: Dict[str, Any],
        prediction: Dict[str, Any],
        duration: float,
        success: bool = True
    ):
        """Record prediction-specific metrics."""
        try:
            timestamp = datetime.now().isoformat()
            
            # Record prediction event
            prediction_event = {
                'timestamp': timestamp,
                'model_name': model_name,
                'input_size': len(input_data.get('y', [])) if isinstance(input_data, dict) else 0,
                'prediction_size': len(prediction.get('point_forecast', [])) if isinstance(prediction, dict) else 0,
                'duration': duration,
                'success': success,
                'confidence': prediction.get('confidence', 0.0) if isinstance(prediction, dict) else 0.0
            }
            
            self.prediction_history.append(prediction_event)
            
            # Record metrics
            self._record_metric('prediction_duration', duration, 'seconds', f'model_{model_name}', timestamp)
            self._record_metric('prediction_success', 1 if success else 0, 'binary', f'model_{model_name}', timestamp)
            
            if success and isinstance(prediction, dict):
                confidence = prediction.get('confidence', 0.0)
                self._record_metric('prediction_confidence', confidence, 'ratio', f'model_{model_name}', timestamp)
            
            # Update model-specific metrics
            if model_name not in self.model_metrics:
                self.model_metrics[model_name] = {
                    'total_predictions': 0,
                    'successful_predictions': 0,
                    'avg_duration': 0.0,
                    'avg_confidence': 0.0
                }
            
            model_stats = self.model_metrics[model_name]
            model_stats['total_predictions'] += 1
            
            if success:
                model_stats['successful_predictions'] += 1
                
                # Update rolling averages
                total_successful = model_stats['successful_predictions']
                old_avg_duration = model_stats['avg_duration']
                model_stats['avg_duration'] = (old_avg_duration * (total_successful - 1) + duration) / total_successful
                
                if isinstance(prediction, dict) and 'confidence' in prediction:
                    old_avg_confidence = model_stats['avg_confidence']
                    confidence = prediction['confidence']
                    model_stats['avg_confidence'] = (old_avg_confidence * (total_successful - 1) + confidence) / total_successful
            
        except Exception as e:
            self.logger.error(f"Prediction recording failed: {str(e)}")
    
    def monitor_operation(self, operation_name: str):
        """Decorator to automatically monitor operation performance."""
        def decorator(func: Callable):
            if asyncio.iscoroutinefunction(func):
                @functools.wraps(func)
                async def async_wrapper(*args, **kwargs):
                    start_time = time.time()
                    success = True
                    try:
                        result = await func(*args, **kwargs)
                        return result
                    except Exception as e:
                        success = False
                        raise
                    finally:
                        duration = time.time() - start_time
                        self.record_operation(operation_name, duration, success)
                return async_wrapper
            else:
                @functools.wraps(func)
                def sync_wrapper(*args, **kwargs):
                    start_time = time.time()
                    success = True
                    try:
                        result = func(*args, **kwargs)
                        return result
                    except Exception as e:
                        success = False
                        raise
                    finally:
                        duration = time.time() - start_time
                        self.record_operation(operation_name, duration, success)
                return sync_wrapper
        return decorator
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary."""
        try:
            summary = {
                'timestamp': datetime.now().isoformat(),
                'system_metrics': {},
                'operation_metrics': {},
                'model_metrics': self.model_metrics.copy(),
                'alert_summary': {},
                'resource_usage': {}
            }
            
            # Current system metrics
            for metric_name, metric_deque in self.resource_metrics.items():
                if metric_deque:
                    current_value = metric_deque[-1]
                    if NUMPY_AVAILABLE and len(metric_deque) > 1:
                        values = list(metric_deque)
                        summary['system_metrics'][metric_name] = {
                            'current': current_value,
                            'avg_1h': np.mean(values[-60:]) if len(values) >= 60 else np.mean(values),
                            'max_1h': np.max(values[-60:]) if len(values) >= 60 else np.max(values),
                            'min_1h': np.min(values[-60:]) if len(values) >= 60 else np.min(values)
                        }
                    else:
                        summary['system_metrics'][metric_name] = {'current': current_value}
            
            # Operation metrics
            for op_name, times in self.operation_times.items():
                if times:
                    if NUMPY_AVAILABLE:
                        summary['operation_metrics'][op_name] = {
                            'total_count': self.operation_counts[op_name],
                            'error_count': self.error_counts[op_name],
                            'error_rate': self.error_counts[op_name] / self.operation_counts[op_name] if self.operation_counts[op_name] > 0 else 0,
                            'avg_duration': np.mean(times),
                            'p95_duration': np.percentile(times, 95) if len(times) > 1 else times[0],
                            'p99_duration': np.percentile(times, 99) if len(times) > 1 else times[0]
                        }
            
            # Alert summary
            active_alerts = [alert for alert in self.alerts if not alert.resolved]
            summary['alert_summary'] = {
                'total_alerts': len(self.alerts),
                'active_alerts': len(active_alerts),
                'critical_alerts': len([a for a in active_alerts if a.severity == 'critical']),
                'warning_alerts': len([a for a in active_alerts if a.severity == 'warning'])
            }
            
            # Recent prediction summary
            if self.prediction_history:
                recent_predictions = list(self.prediction_history)[-100:]  # Last 100 predictions
                successful_predictions = [p for p in recent_predictions if p['success']]
                
                if successful_predictions:
                    if NUMPY_AVAILABLE:
                        durations = [p['duration'] for p in successful_predictions]
                        confidences = [p['confidence'] for p in successful_predictions if p['confidence'] > 0]
                        
                        summary['prediction_summary'] = {
                            'total_predictions': len(recent_predictions),
                            'successful_predictions': len(successful_predictions),
                            'success_rate': len(successful_predictions) / len(recent_predictions),
                            'avg_duration': np.mean(durations),
                            'avg_confidence': np.mean(confidences) if confidences else 0.0
                        }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to generate metrics summary: {str(e)}")
            return {'error': str(e)}
    
    def get_alerts(self, severity: Optional[str] = None, active_only: bool = True) -> List[Dict[str, Any]]:
        """Get alerts with optional filtering."""
        try:
            alerts = self.alerts
            
            if active_only:
                alerts = [alert for alert in alerts if not alert.resolved]
            
            if severity:
                alerts = [alert for alert in alerts if alert.severity == severity]
            
            return [asdict(alert) for alert in alerts]
            
        except Exception as e:
            self.logger.error(f"Failed to get alerts: {str(e)}")
            return []
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Manually resolve an alert."""
        try:
            for alert in self.alerts:
                if alert.alert_id == alert_id and not alert.resolved:
                    alert.resolved = True
                    alert.resolution_time = datetime.now().isoformat()
                    self.logger.info(f"Alert {alert_id} resolved manually")
                    return True
            return False
        except Exception as e:
            self.logger.error(f"Failed to resolve alert: {str(e)}")
            return False
    
    def export_metrics(self, format_type: str = 'json') -> Dict[str, Any]:
        """Export metrics in specified format."""
        try:
            if format_type == 'json':
                return self._export_json_metrics()
            elif format_type == 'prometheus':
                return self._export_prometheus_metrics()
            else:
                raise ValueError(f"Unsupported export format: {format_type}")
                
        except Exception as e:
            self.logger.error(f"Metrics export failed: {str(e)}")
            return {'error': str(e)}
    
    def _export_json_metrics(self) -> Dict[str, Any]:
        """Export metrics in JSON format."""
        return {
            'timestamp': datetime.now().isoformat(),
            'metrics': {
                name: [asdict(metric) for metric in metric_deque]
                for name, metric_deque in self.metrics.items()
            },
            'alerts': [asdict(alert) for alert in self.alerts],
            'model_metrics': self.model_metrics,
            'summary': self.get_metrics_summary()
        }
    
    def _export_prometheus_metrics(self) -> Dict[str, Any]:
        """Export metrics in Prometheus format."""
        # This would generate Prometheus-compatible metrics
        # For now, return a simplified version
        prometheus_metrics = []
        
        for metric_name, metric_deque in self.metrics.items():
            if metric_deque:
                latest_metric = metric_deque[-1]
                prometheus_metrics.append(
                    f"neural_forecast_{metric_name.replace('-', '_')} "
                    f"{{component=\"{latest_metric.component}\"}} {latest_metric.value}"
                )
        
        return {
            'format': 'prometheus',
            'metrics': prometheus_metrics,
            'timestamp': datetime.now().isoformat()
        }
    
    def set_alert_threshold(self, metric_name: str, warning: Optional[float] = None, critical: Optional[float] = None):
        """Set custom alert thresholds for a metric."""
        if metric_name not in self.alert_thresholds:
            self.alert_thresholds[metric_name] = {}
        
        if warning is not None:
            self.alert_thresholds[metric_name]['warning'] = warning
        
        if critical is not None:
            self.alert_thresholds[metric_name]['critical'] = critical
        
        self.logger.info(f"Alert thresholds updated for {metric_name}: {self.alert_thresholds[metric_name]}")
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check of monitoring system."""
        try:
            active_alerts = [alert for alert in self.alerts if not alert.resolved]
            critical_alerts = [alert for alert in active_alerts if alert.severity == 'critical']
            
            # Determine health status
            if critical_alerts:
                status = 'critical'
            elif len(active_alerts) > 5:
                status = 'warning'
            elif not self.monitoring_active:
                status = 'warning'
            else:
                status = 'healthy'
            
            return {
                'status': status,
                'timestamp': datetime.now().isoformat(),
                'monitoring_active': self.monitoring_active,
                'metrics_count': sum(len(deque) for deque in self.metrics.values()),
                'active_alerts': len(active_alerts),
                'critical_alerts': len(critical_alerts),
                'models_monitored': len(self.model_metrics),
                'predictions_recorded': len(self.prediction_history),
                'uptime': 'monitoring_thread_active' if self.monitoring_active else 'stopped'
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def __del__(self):
        """Cleanup monitoring resources."""
        try:
            self.stop_monitoring()
        except Exception:
            pass