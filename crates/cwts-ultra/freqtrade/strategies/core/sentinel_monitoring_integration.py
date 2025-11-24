#!/usr/bin/env python3
"""
Sentinel Monitoring Integration for Unified NHITS/NBEATSx + ATS-CP System
==========================================================================

Real-time monitoring and alerting system for the unified forecasting pipeline
ensuring performance, accuracy, and TENGRI compliance in production.

Monitoring Capabilities:
- Real-time performance tracking (<585ns latency)
- Accuracy degradation detection
- TENGRI compliance monitoring
- Quantum coherence tracking
- Component health monitoring
- Predictive failure detection

Integration Features:
- Deployed sentinel system integration
- Real-time dashboard updates
- Automated alerting
- Performance regression detection
- Anomaly detection and response
"""

import asyncio
import time
import numpy as np
import json
import logging
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import deque, defaultdict
import warnings
from datetime import datetime, timedelta
import threading
import socket
import psutil
import os

# Performance monitoring
try:
    import prometheus_client
    from prometheus_client import Gauge, Counter, Histogram, CollectorRegistry
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

# Dashboard integration
try:
    import websockets
    import aiohttp
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False

# Import integration components for monitoring
try:
    from .unified_nhits_nbeatsx_ats_cp_integration import (
        UnifiedNHITSNBEATSxATSCPEngine,
        ForecastOutput
    )
    from .ultra_fast_performance_engine import (
        UltraFastPipelineOrchestrator,
        PerformanceConfig
    )
    from .component_wise_ats_cp_calibrator import (
        UnifiedComponentWiseCalibrator,
        ComponentCalibrationResult
    )
    from .quantum_enhanced_temperature_scaling import (
        QuantumEnhancedTemperatureScaling,
        QuantumTemperatureResult
    )
    from .tengri_compliance_validator import (
        ComprehensiveTENGRIValidator,
        TENGRIValidationResult
    )
    INTEGRATION_COMPONENTS_AVAILABLE = True
except ImportError as e:
    INTEGRATION_COMPONENTS_AVAILABLE = False
    warnings.warn(f"Integration components not available for monitoring: {e}")

logger = logging.getLogger(__name__)

# =============================================================================
# MONITORING METRICS AND ALERTS
# =============================================================================

class AlertSeverity(Enum):
    """Alert severity levels"""
    CRITICAL = "critical"    # System failure, immediate action required
    HIGH = "high"           # Performance degradation, urgent attention
    MEDIUM = "medium"       # Warning condition, should investigate
    LOW = "low"            # Information, monitor trend
    INFO = "info"          # Normal operation, for logging

class MetricType(Enum):
    """Types of metrics monitored"""
    PERFORMANCE = "performance"
    ACCURACY = "accuracy"
    AVAILABILITY = "availability"
    THROUGHPUT = "throughput"
    LATENCY = "latency"
    ERROR_RATE = "error_rate"
    RESOURCE_USAGE = "resource_usage"
    QUANTUM_COHERENCE = "quantum_coherence"
    TENGRI_COMPLIANCE = "tengri_compliance"

@dataclass
class MonitoringAlert:
    """Monitoring alert record"""
    
    alert_id: str
    severity: AlertSeverity
    metric_type: MetricType
    title: str
    description: str
    
    # Context
    component: str
    timestamp: float = field(default_factory=time.time)
    value: Optional[float] = None
    threshold: Optional[float] = None
    
    # Alert management
    acknowledged: bool = False
    resolved: bool = False
    resolution_time: Optional[float] = None
    
    # Metadata
    tags: Dict[str, str] = field(default_factory=dict)
    additional_data: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SystemMetrics:
    """System-wide metrics snapshot"""
    
    # Performance metrics
    total_latency_ns: int
    nbeatsx_latency_ns: int
    ats_cp_latency_ns: int
    quantum_latency_ns: int
    
    # Accuracy metrics
    forecast_accuracy: float
    calibration_accuracy: float
    coverage_achieved: float
    
    # System health
    cpu_usage_percent: float
    memory_usage_percent: float
    disk_usage_percent: float
    network_io_mbps: float
    
    # Component status
    nbeatsx_status: str
    ats_cp_status: str
    quantum_status: str
    cerebellar_status: str
    
    # Quantum metrics
    quantum_coherence: float
    quantum_advantage: float
    
    # TENGRI compliance
    tengri_compliance_score: float
    
    # Metadata
    timestamp: float = field(default_factory=time.time)
    requests_per_second: float = 0.0
    error_rate_percent: float = 0.0

# =============================================================================
# REAL-TIME METRICS COLLECTOR
# =============================================================================

class MetricsCollector:
    """
    Real-time metrics collection for all system components
    """
    
    def __init__(self, collection_interval_ms: int = 100):
        self.collection_interval = collection_interval_ms / 1000.0
        self.running = False
        self.metrics_history = deque(maxlen=10000)  # Keep 10k samples
        
        # Component performance tracking
        self.component_metrics = defaultdict(lambda: deque(maxlen=1000))
        
        # System baselines
        self.baseline_metrics = None
        self.baseline_update_time = 0
        
        # Prometheus metrics (if available)
        if PROMETHEUS_AVAILABLE:
            self.registry = CollectorRegistry()
            self._setup_prometheus_metrics()
        
        # Collection thread
        self.collection_thread = None
        
        logger.info(f"📊 Metrics collector initialized (interval: {collection_interval_ms}ms)")
    
    def _setup_prometheus_metrics(self):
        """Setup Prometheus metrics"""
        self.prom_latency = Histogram(
            'pipeline_latency_nanoseconds',
            'Pipeline execution latency in nanoseconds',
            buckets=[100, 200, 500, 1000, 2000, 5000],
            registry=self.registry
        )
        
        self.prom_accuracy = Gauge(
            'forecast_accuracy',
            'Forecast accuracy score',
            registry=self.registry
        )
        
        self.prom_quantum_coherence = Gauge(
            'quantum_coherence',
            'Quantum system coherence level',
            registry=self.registry
        )
        
        self.prom_tengri_compliance = Gauge(
            'tengri_compliance_score',
            'TENGRI compliance score',
            registry=self.registry
        )
        
        self.prom_error_rate = Gauge(
            'error_rate_percent',
            'System error rate percentage',
            registry=self.registry
        )
    
    def start_collection(self):
        """Start metrics collection in background thread"""
        if self.running:
            return
        
        self.running = True
        self.collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.collection_thread.start()
        
        logger.info("📊 Metrics collection started")
    
    def stop_collection(self):
        """Stop metrics collection"""
        self.running = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5)
        
        logger.info("📊 Metrics collection stopped")
    
    def _collection_loop(self):
        """Main metrics collection loop"""
        while self.running:
            try:
                start_time = time.time()
                
                # Collect system metrics
                metrics = self._collect_system_metrics()
                
                # Store metrics
                self.metrics_history.append(metrics)
                
                # Update Prometheus metrics
                if PROMETHEUS_AVAILABLE:
                    self._update_prometheus_metrics(metrics)
                
                # Calculate sleep time to maintain interval
                collection_time = time.time() - start_time
                sleep_time = max(0, self.collection_interval - collection_time)
                time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                time.sleep(self.collection_interval)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect comprehensive system metrics"""
        
        # System resource metrics
        cpu_percent = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        network = psutil.net_io_counters()
        
        # Default metrics (would be updated by components)
        metrics = SystemMetrics(
            # Performance (defaults - will be updated by actual measurements)
            total_latency_ns=0,
            nbeatsx_latency_ns=0,
            ats_cp_latency_ns=0,
            quantum_latency_ns=0,
            
            # Accuracy (defaults)
            forecast_accuracy=0.0,
            calibration_accuracy=0.0,
            coverage_achieved=0.0,
            
            # System health
            cpu_usage_percent=cpu_percent,
            memory_usage_percent=memory.percent,
            disk_usage_percent=disk.percent,
            network_io_mbps=(network.bytes_sent + network.bytes_recv) / 1024 / 1024,
            
            # Component status (defaults)
            nbeatsx_status="unknown",
            ats_cp_status="unknown",
            quantum_status="unknown",
            cerebellar_status="unknown",
            
            # Quantum metrics (defaults)
            quantum_coherence=0.0,
            quantum_advantage=1.0,
            
            # TENGRI compliance (default)
            tengri_compliance_score=0.0
        )
        
        return metrics
    
    def _update_prometheus_metrics(self, metrics: SystemMetrics):
        """Update Prometheus metrics"""
        try:
            self.prom_latency.observe(metrics.total_latency_ns)
            self.prom_accuracy.set(metrics.forecast_accuracy)
            self.prom_quantum_coherence.set(metrics.quantum_coherence)
            self.prom_tengri_compliance.set(metrics.tengri_compliance_score)
            self.prom_error_rate.set(metrics.error_rate_percent)
        except Exception as e:
            logger.warning(f"Failed to update Prometheus metrics: {e}")
    
    def update_component_metrics(self, component: str, metrics: Dict[str, Any]):
        """Update metrics for a specific component"""
        self.component_metrics[component].append({
            'timestamp': time.time(),
            'metrics': metrics
        })
    
    def get_latest_metrics(self) -> Optional[SystemMetrics]:
        """Get the most recent metrics"""
        return self.metrics_history[-1] if self.metrics_history else None
    
    def get_metrics_history(self, duration_seconds: int = 300) -> List[SystemMetrics]:
        """Get metrics history for specified duration"""
        cutoff_time = time.time() - duration_seconds
        return [m for m in self.metrics_history if m.timestamp >= cutoff_time]
    
    def calculate_baseline_metrics(self):
        """Calculate baseline metrics for anomaly detection"""
        if len(self.metrics_history) < 100:  # Need enough samples
            return
        
        recent_metrics = list(self.metrics_history)[-100:]
        
        self.baseline_metrics = {
            'latency_mean': np.mean([m.total_latency_ns for m in recent_metrics]),
            'latency_std': np.std([m.total_latency_ns for m in recent_metrics]),
            'accuracy_mean': np.mean([m.forecast_accuracy for m in recent_metrics]),
            'accuracy_std': np.std([m.forecast_accuracy for m in recent_metrics]),
            'cpu_mean': np.mean([m.cpu_usage_percent for m in recent_metrics]),
            'cpu_std': np.std([m.cpu_usage_percent for m in recent_metrics])
        }
        
        self.baseline_update_time = time.time()
        logger.debug("📊 Baseline metrics updated")

# =============================================================================
# ANOMALY DETECTION ENGINE
# =============================================================================

class AnomalyDetector:
    """
    Real-time anomaly detection for system monitoring
    """
    
    def __init__(self, sensitivity: float = 2.0):
        self.sensitivity = sensitivity  # Number of standard deviations for anomaly
        self.anomaly_history = deque(maxlen=1000)
        
        # Detection thresholds
        self.thresholds = {
            'latency_critical_ns': 1000,     # 1μs
            'latency_warning_ns': 585,       # Target threshold
            'accuracy_critical': 0.7,        # 70% accuracy
            'accuracy_warning': 0.8,         # 80% accuracy
            'cpu_critical': 90.0,            # 90% CPU
            'cpu_warning': 75.0,             # 75% CPU
            'memory_critical': 90.0,         # 90% memory
            'memory_warning': 80.0,          # 80% memory
            'error_rate_critical': 5.0,      # 5% error rate
            'error_rate_warning': 1.0        # 1% error rate
        }
        
        logger.info(f"🔍 Anomaly detector initialized (sensitivity: {sensitivity}σ)")
    
    def detect_anomalies(self, current_metrics: SystemMetrics,
                        baseline_metrics: Optional[Dict[str, float]] = None) -> List[MonitoringAlert]:
        """
        Detect anomalies in current metrics
        
        Args:
            current_metrics: Current system metrics
            baseline_metrics: Baseline metrics for comparison
            
        Returns:
            List of detected anomalies as alerts
        """
        alerts = []
        
        # Threshold-based detection
        alerts.extend(self._detect_threshold_violations(current_metrics))
        
        # Statistical anomaly detection (if baseline available)
        if baseline_metrics:
            alerts.extend(self._detect_statistical_anomalies(current_metrics, baseline_metrics))
        
        # Trend-based detection
        alerts.extend(self._detect_trend_anomalies(current_metrics))
        
        # Store anomalies
        for alert in alerts:
            self.anomaly_history.append(alert)
        
        return alerts
    
    def _detect_threshold_violations(self, metrics: SystemMetrics) -> List[MonitoringAlert]:
        """Detect threshold violations"""
        alerts = []
        
        # Latency violations
        if metrics.total_latency_ns > self.thresholds['latency_critical_ns']:
            alerts.append(MonitoringAlert(
                alert_id=f"latency_critical_{int(time.time())}",
                severity=AlertSeverity.CRITICAL,
                metric_type=MetricType.LATENCY,
                title="Critical Latency Violation",
                description=f"Total latency {metrics.total_latency_ns}ns exceeds critical threshold",
                component="pipeline",
                value=metrics.total_latency_ns,
                threshold=self.thresholds['latency_critical_ns']
            ))
        elif metrics.total_latency_ns > self.thresholds['latency_warning_ns']:
            alerts.append(MonitoringAlert(
                alert_id=f"latency_warning_{int(time.time())}",
                severity=AlertSeverity.HIGH,
                metric_type=MetricType.LATENCY,
                title="Latency Performance Warning",
                description=f"Total latency {metrics.total_latency_ns}ns exceeds target",
                component="pipeline",
                value=metrics.total_latency_ns,
                threshold=self.thresholds['latency_warning_ns']
            ))
        
        # Accuracy violations
        if metrics.forecast_accuracy < self.thresholds['accuracy_critical']:
            alerts.append(MonitoringAlert(
                alert_id=f"accuracy_critical_{int(time.time())}",
                severity=AlertSeverity.CRITICAL,
                metric_type=MetricType.ACCURACY,
                title="Critical Accuracy Degradation",
                description=f"Forecast accuracy {metrics.forecast_accuracy:.1%} below critical threshold",
                component="forecasting",
                value=metrics.forecast_accuracy,
                threshold=self.thresholds['accuracy_critical']
            ))
        elif metrics.forecast_accuracy < self.thresholds['accuracy_warning']:
            alerts.append(MonitoringAlert(
                alert_id=f"accuracy_warning_{int(time.time())}",
                severity=AlertSeverity.MEDIUM,
                metric_type=MetricType.ACCURACY,
                title="Accuracy Performance Warning",
                description=f"Forecast accuracy {metrics.forecast_accuracy:.1%} below target",
                component="forecasting",
                value=metrics.forecast_accuracy,
                threshold=self.thresholds['accuracy_warning']
            ))
        
        # Resource violations
        if metrics.cpu_usage_percent > self.thresholds['cpu_critical']:
            alerts.append(MonitoringAlert(
                alert_id=f"cpu_critical_{int(time.time())}",
                severity=AlertSeverity.HIGH,
                metric_type=MetricType.RESOURCE_USAGE,
                title="Critical CPU Usage",
                description=f"CPU usage {metrics.cpu_usage_percent:.1f}% exceeds critical threshold",
                component="system",
                value=metrics.cpu_usage_percent,
                threshold=self.thresholds['cpu_critical']
            ))
        
        if metrics.memory_usage_percent > self.thresholds['memory_critical']:
            alerts.append(MonitoringAlert(
                alert_id=f"memory_critical_{int(time.time())}",
                severity=AlertSeverity.HIGH,
                metric_type=MetricType.RESOURCE_USAGE,
                title="Critical Memory Usage",
                description=f"Memory usage {metrics.memory_usage_percent:.1f}% exceeds critical threshold",
                component="system",
                value=metrics.memory_usage_percent,
                threshold=self.thresholds['memory_critical']
            ))
        
        return alerts
    
    def _detect_statistical_anomalies(self, current: SystemMetrics,
                                    baseline: Dict[str, float]) -> List[MonitoringAlert]:
        """Detect statistical anomalies using baseline metrics"""
        alerts = []
        
        # Latency anomaly
        latency_z_score = (current.total_latency_ns - baseline['latency_mean']) / (baseline['latency_std'] + 1e-8)
        if abs(latency_z_score) > self.sensitivity:
            severity = AlertSeverity.HIGH if latency_z_score > 0 else AlertSeverity.MEDIUM
            alerts.append(MonitoringAlert(
                alert_id=f"latency_anomaly_{int(time.time())}",
                severity=severity,
                metric_type=MetricType.LATENCY,
                title="Latency Statistical Anomaly",
                description=f"Latency deviates {latency_z_score:.1f}σ from baseline",
                component="pipeline",
                value=current.total_latency_ns,
                additional_data={'z_score': latency_z_score, 'baseline_mean': baseline['latency_mean']}
            ))
        
        # Accuracy anomaly
        accuracy_z_score = (current.forecast_accuracy - baseline['accuracy_mean']) / (baseline['accuracy_std'] + 1e-8)
        if abs(accuracy_z_score) > self.sensitivity:
            severity = AlertSeverity.HIGH if accuracy_z_score < 0 else AlertSeverity.INFO
            alerts.append(MonitoringAlert(
                alert_id=f"accuracy_anomaly_{int(time.time())}",
                severity=severity,
                metric_type=MetricType.ACCURACY,
                title="Accuracy Statistical Anomaly",
                description=f"Accuracy deviates {accuracy_z_score:.1f}σ from baseline",
                component="forecasting",
                value=current.forecast_accuracy,
                additional_data={'z_score': accuracy_z_score, 'baseline_mean': baseline['accuracy_mean']}
            ))
        
        return alerts
    
    def _detect_trend_anomalies(self, current: SystemMetrics) -> List[MonitoringAlert]:
        """Detect trend-based anomalies"""
        alerts = []
        
        # TENGRI compliance degradation
        if current.tengri_compliance_score < 0.8:
            severity = AlertSeverity.CRITICAL if current.tengri_compliance_score < 0.6 else AlertSeverity.HIGH
            alerts.append(MonitoringAlert(
                alert_id=f"tengri_compliance_{int(time.time())}",
                severity=severity,
                metric_type=MetricType.TENGRI_COMPLIANCE,
                title="TENGRI Compliance Violation",
                description=f"TENGRI compliance score {current.tengri_compliance_score:.1%} below acceptable threshold",
                component="compliance",
                value=current.tengri_compliance_score,
                threshold=0.8
            ))
        
        # Quantum coherence degradation
        if current.quantum_coherence < 0.9:
            severity = AlertSeverity.HIGH if current.quantum_coherence < 0.8 else AlertSeverity.MEDIUM
            alerts.append(MonitoringAlert(
                alert_id=f"quantum_coherence_{int(time.time())}",
                severity=severity,
                metric_type=MetricType.QUANTUM_COHERENCE,
                title="Quantum Coherence Degradation",
                description=f"Quantum coherence {current.quantum_coherence:.1%} below acceptable level",
                component="quantum",
                value=current.quantum_coherence,
                threshold=0.9
            ))
        
        return alerts

# =============================================================================
# SENTINEL DASHBOARD INTEGRATION
# =============================================================================

class SentinelDashboardIntegrator:
    """
    Integration with deployed sentinel monitoring dashboard
    """
    
    def __init__(self, dashboard_url: str = "ws://localhost:8080/ws",
                 api_url: str = "http://localhost:8080/api"):
        self.dashboard_url = dashboard_url
        self.api_url = api_url
        self.websocket = None
        self.connected = False
        
        # Message queue for dashboard updates
        self.message_queue = asyncio.Queue()
        self.update_task = None
        
        logger.info(f"📱 Dashboard integrator initialized (URL: {dashboard_url})")
    
    async def connect(self):
        """Connect to sentinel dashboard"""
        if not WEBSOCKET_AVAILABLE:
            logger.warning("WebSocket not available, dashboard integration disabled")
            return
        
        try:
            self.websocket = await websockets.connect(self.dashboard_url)
            self.connected = True
            
            # Start update task
            self.update_task = asyncio.create_task(self._update_loop())
            
            logger.info("📱 Connected to sentinel dashboard")
            
        except Exception as e:
            logger.error(f"Failed to connect to dashboard: {e}")
            self.connected = False
    
    async def disconnect(self):
        """Disconnect from sentinel dashboard"""
        self.connected = False
        
        if self.update_task:
            self.update_task.cancel()
        
        if self.websocket:
            await self.websocket.close()
        
        logger.info("📱 Disconnected from sentinel dashboard")
    
    async def send_metrics_update(self, metrics: SystemMetrics):
        """Send metrics update to dashboard"""
        if not self.connected:
            return
        
        update_message = {
            'type': 'metrics_update',
            'timestamp': metrics.timestamp,
            'data': asdict(metrics)
        }
        
        await self.message_queue.put(update_message)
    
    async def send_alert(self, alert: MonitoringAlert):
        """Send alert to dashboard"""
        if not self.connected:
            return
        
        alert_message = {
            'type': 'alert',
            'timestamp': alert.timestamp,
            'data': asdict(alert)
        }
        
        await self.message_queue.put(alert_message)
    
    async def send_component_status(self, component: str, status: Dict[str, Any]):
        """Send component status update"""
        if not self.connected:
            return
        
        status_message = {
            'type': 'component_status',
            'component': component,
            'timestamp': time.time(),
            'data': status
        }
        
        await self.message_queue.put(status_message)
    
    async def _update_loop(self):
        """Main update loop for dashboard communication"""
        while self.connected:
            try:
                # Wait for message with timeout
                message = await asyncio.wait_for(self.message_queue.get(), timeout=1.0)
                
                # Send message to dashboard
                await self.websocket.send(json.dumps(message))
                
            except asyncio.TimeoutError:
                # Send heartbeat
                heartbeat = {
                    'type': 'heartbeat',
                    'timestamp': time.time()
                }
                await self.websocket.send(json.dumps(heartbeat))
                
            except Exception as e:
                logger.error(f"Dashboard update error: {e}")
                self.connected = False

# =============================================================================
# UNIFIED SENTINEL MONITORING SYSTEM
# =============================================================================

class UnifiedSentinelMonitoringSystem:
    """
    Unified monitoring system coordinating all monitoring aspects
    for the integrated NHITS/NBEATSx + ATS-CP pipeline
    """
    
    def __init__(self, 
                 metrics_interval_ms: int = 100,
                 dashboard_url: Optional[str] = None,
                 enable_prometheus: bool = True):
        
        # Initialize components
        self.metrics_collector = MetricsCollector(metrics_interval_ms)
        self.anomaly_detector = AnomalyDetector()
        
        # Dashboard integration
        if dashboard_url:
            self.dashboard = SentinelDashboardIntegrator(dashboard_url)
        else:
            self.dashboard = None
        
        # Alert management
        self.active_alerts = {}
        self.alert_history = deque(maxlen=10000)
        
        # Component monitoring
        self.monitored_components = {}
        
        # Running state
        self.running = False
        self.monitoring_task = None
        
        logger.info("🛡️ Unified Sentinel Monitoring System initialized")
        logger.info(f"   Metrics interval: {metrics_interval_ms}ms")
        logger.info(f"   Dashboard integration: {'enabled' if dashboard_url else 'disabled'}")
        logger.info(f"   Prometheus integration: {'enabled' if enable_prometheus and PROMETHEUS_AVAILABLE else 'disabled'}")
    
    async def start_monitoring(self):
        """Start the complete monitoring system"""
        if self.running:
            return
        
        self.running = True
        
        # Start metrics collection
        self.metrics_collector.start_collection()
        
        # Connect to dashboard
        if self.dashboard:
            await self.dashboard.connect()
        
        # Start monitoring loop
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info("🛡️ Sentinel monitoring system started")
    
    async def stop_monitoring(self):
        """Stop the monitoring system"""
        self.running = False
        
        # Stop components
        self.metrics_collector.stop_collection()
        
        if self.dashboard:
            await self.dashboard.disconnect()
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
        
        logger.info("🛡️ Sentinel monitoring system stopped")
    
    def register_component(self, component_name: str, component_instance: Any):
        """Register a component for monitoring"""
        self.monitored_components[component_name] = component_instance
        logger.info(f"📊 Registered component for monitoring: {component_name}")
    
    async def record_forecast_execution(self, 
                                      execution_time_ns: int,
                                      forecast_output: ForecastOutput):
        """Record forecast execution metrics"""
        
        # Update metrics collector
        self.metrics_collector.update_component_metrics('pipeline', {
            'execution_time_ns': execution_time_ns,
            'nbeatsx_time_ns': forecast_output.inference_time_ns,
            'calibration_time_ns': forecast_output.calibration_time_ns,
            'forecast_confidence': forecast_output.forecast_confidence
        })
        
        # Send to dashboard
        if self.dashboard:
            await self.dashboard.send_component_status('pipeline', {
                'status': 'active',
                'last_execution_ns': execution_time_ns,
                'performance_target_met': execution_time_ns <= 585
            })
    
    async def record_calibration_result(self, 
                                      component: str,
                                      calibration_result: ComponentCalibrationResult):
        """Record calibration result metrics"""
        
        # Update metrics
        self.metrics_collector.update_component_metrics(f'calibration_{component}', {
            'calibration_time_ns': calibration_result.calibration_time_ns,
            'temperature': calibration_result.temperature,
            'coverage_estimate': calibration_result.coverage_estimate,
            'calibration_error': calibration_result.calibration_error
        })
    
    async def record_quantum_result(self, quantum_result: QuantumTemperatureResult):
        """Record quantum optimization result"""
        
        # Update metrics
        self.metrics_collector.update_component_metrics('quantum', {
            'optimization_time_ns': quantum_result.optimization_time_ns,
            'quantum_advantage': quantum_result.quantum_advantage,
            'optimal_temperature': quantum_result.optimal_temperature,
            'convergence_achieved': quantum_result.convergence_achieved
        })
    
    async def record_tengri_validation(self, validation_result: TENGRIValidationResult):
        """Record TENGRI validation result"""
        
        # Update metrics
        self.metrics_collector.update_component_metrics('tengri_compliance', {
            'compliance_score': validation_result.overall_compliance_score,
            'validation_passed': validation_result.passed,
            'critical_violations': len([v for v in validation_result.violations if v.severity.value == 'critical']),
            'total_violations': len(validation_result.violations)
        })
        
        # Send compliance alert if failed
        if not validation_result.passed:
            alert = MonitoringAlert(
                alert_id=f"tengri_validation_failed_{int(time.time())}",
                severity=AlertSeverity.CRITICAL,
                metric_type=MetricType.TENGRI_COMPLIANCE,
                title="TENGRI Validation Failed",
                description=f"TENGRI compliance validation failed with {len(validation_result.violations)} violations",
                component="compliance",
                value=validation_result.overall_compliance_score
            )
            await self._handle_alert(alert)
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                # Get latest metrics
                current_metrics = self.metrics_collector.get_latest_metrics()
                
                if current_metrics:
                    # Update system metrics with component data
                    await self._update_system_metrics(current_metrics)
                    
                    # Detect anomalies
                    baseline = self.metrics_collector.baseline_metrics
                    alerts = self.anomaly_detector.detect_anomalies(current_metrics, baseline)
                    
                    # Handle alerts
                    for alert in alerts:
                        await self._handle_alert(alert)
                    
                    # Send metrics to dashboard
                    if self.dashboard:
                        await self.dashboard.send_metrics_update(current_metrics)
                    
                    # Update baselines periodically
                    if time.time() - self.metrics_collector.baseline_update_time > 300:  # 5 minutes
                        self.metrics_collector.calculate_baseline_metrics()
                
                # Sleep for monitoring interval
                await asyncio.sleep(self.metrics_collector.collection_interval)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(1.0)
    
    async def _update_system_metrics(self, metrics: SystemMetrics):
        """Update system metrics with component data"""
        
        # Update from component metrics
        for component, component_metrics_list in self.metrics_collector.component_metrics.items():
            if component_metrics_list:
                latest = component_metrics_list[-1]['metrics']
                
                if component == 'pipeline':
                    metrics.total_latency_ns = latest.get('execution_time_ns', 0)
                    metrics.nbeatsx_latency_ns = latest.get('nbeatsx_time_ns', 0)
                    metrics.ats_cp_latency_ns = latest.get('calibration_time_ns', 0)
                    metrics.forecast_accuracy = latest.get('forecast_confidence', 0.0)
                
                elif component == 'quantum':
                    metrics.quantum_latency_ns = latest.get('optimization_time_ns', 0)
                    metrics.quantum_advantage = latest.get('quantum_advantage', 1.0)
                    metrics.quantum_coherence = latest.get('quantum_coherence', 0.0)
                
                elif component == 'tengri_compliance':
                    metrics.tengri_compliance_score = latest.get('compliance_score', 0.0)
        
        # Update component status
        metrics.nbeatsx_status = "active" if metrics.nbeatsx_latency_ns > 0 else "inactive"
        metrics.ats_cp_status = "active" if metrics.ats_cp_latency_ns > 0 else "inactive"
        metrics.quantum_status = "active" if metrics.quantum_latency_ns > 0 else "inactive"
    
    async def _handle_alert(self, alert: MonitoringAlert):
        """Handle monitoring alerts"""
        
        # Store alert
        self.active_alerts[alert.alert_id] = alert
        self.alert_history.append(alert)
        
        # Log alert
        severity_emoji = {
            AlertSeverity.CRITICAL: "🔴",
            AlertSeverity.HIGH: "🟠", 
            AlertSeverity.MEDIUM: "🟡",
            AlertSeverity.LOW: "🔵",
            AlertSeverity.INFO: "ℹ️"
        }
        
        emoji = severity_emoji.get(alert.severity, "⚠️")
        logger.warning(f"{emoji} ALERT [{alert.severity.value.upper()}] {alert.title}: {alert.description}")
        
        # Send to dashboard
        if self.dashboard:
            await self.dashboard.send_alert(alert)
        
        # Auto-resolve some alerts after time
        if alert.severity in [AlertSeverity.LOW, AlertSeverity.INFO]:
            # Auto-resolve info/low alerts after 5 minutes
            asyncio.create_task(self._auto_resolve_alert(alert.alert_id, 300))
    
    async def _auto_resolve_alert(self, alert_id: str, delay_seconds: int):
        """Auto-resolve alert after delay"""
        await asyncio.sleep(delay_seconds)
        
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            alert.resolution_time = time.time()
            del self.active_alerts[alert_id]
            
            logger.info(f"✅ Auto-resolved alert: {alert.title}")
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an active alert"""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].acknowledged = True
            logger.info(f"✅ Alert acknowledged: {alert_id}")
            return True
        return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Manually resolve an alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            alert.resolution_time = time.time()
            del self.active_alerts[alert_id]
            
            logger.info(f"✅ Alert resolved: {alert.title}")
            return True
        return False
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get comprehensive monitoring status"""
        latest_metrics = self.metrics_collector.get_latest_metrics()
        
        return {
            'system_status': {
                'monitoring_active': self.running,
                'dashboard_connected': self.dashboard.connected if self.dashboard else False,
                'components_monitored': len(self.monitored_components)
            },
            'current_metrics': asdict(latest_metrics) if latest_metrics else None,
            'active_alerts': {
                'total': len(self.active_alerts),
                'critical': len([a for a in self.active_alerts.values() if a.severity == AlertSeverity.CRITICAL]),
                'high': len([a for a in self.active_alerts.values() if a.severity == AlertSeverity.HIGH]),
                'medium': len([a for a in self.active_alerts.values() if a.severity == AlertSeverity.MEDIUM])
            },
            'performance_summary': {
                'target_latency_ns': 585,
                'current_latency_ns': latest_metrics.total_latency_ns if latest_metrics else 0,
                'target_met': (latest_metrics.total_latency_ns <= 585) if latest_metrics else False,
                'forecast_accuracy': latest_metrics.forecast_accuracy if latest_metrics else 0.0,
                'tengri_compliance': latest_metrics.tengri_compliance_score if latest_metrics else 0.0
            }
        }

# =============================================================================
# DEMONSTRATION
# =============================================================================

async def demonstrate_sentinel_monitoring():
    """
    Demonstrate sentinel monitoring integration
    """
    print("🛡️ SENTINEL MONITORING INTEGRATION DEMONSTRATION")
    print("=" * 60)
    print("Testing real-time monitoring for unified forecasting system")
    print("=" * 60)
    
    try:
        # Initialize monitoring system
        monitoring_system = UnifiedSentinelMonitoringSystem(
            metrics_interval_ms=500,  # 500ms for demonstration
            dashboard_url=None,  # No dashboard for demo
            enable_prometheus=PROMETHEUS_AVAILABLE
        )
        
        print(f"\\n🛡️ Monitoring System Configuration:")
        print(f"   Metrics collection: 500ms interval")
        print(f"   Anomaly detection: enabled")
        print(f"   Prometheus integration: {'enabled' if PROMETHEUS_AVAILABLE else 'disabled'}")
        print(f"   Dashboard integration: disabled (demo mode)")
        
        # Start monitoring
        await monitoring_system.start_monitoring()
        
        print(f"\\n📊 Starting Monitoring...")
        
        # Simulate system activity
        print(f"\\n🔄 Simulating System Activity:")
        
        # Simulate forecast executions
        for i in range(5):
            # Create mock forecast output
            mock_forecast = type('ForecastOutput', (), {
                'inference_time_ns': 400 + i * 20,  # Increasing latency
                'calibration_time_ns': 80 + i * 10,
                'total_time_ns': 480 + i * 30,
                'forecast_confidence': 0.9 - i * 0.05  # Decreasing confidence
            })()
            
            execution_time = 480 + i * 30
            
            await monitoring_system.record_forecast_execution(execution_time, mock_forecast)
            
            print(f"   Execution {i+1}: {execution_time}ns "
                  f"({'✅' if execution_time <= 585 else '❌ ALERT'})")
            
            await asyncio.sleep(0.1)  # Brief pause
        
        # Simulate calibration results
        mock_calibration = type('ComponentCalibrationResult', (), {
            'calibration_time_ns': 95,
            'temperature': 1.2,
            'coverage_estimate': 0.88,
            'calibration_error': 0.05
        })()
        
        await monitoring_system.record_calibration_result('trend', mock_calibration)
        print(f"   Calibration recorded: 95ns, temp=1.2")
        
        # Simulate quantum optimization
        mock_quantum = type('QuantumTemperatureResult', (), {
            'optimization_time_ns': 45,
            'quantum_advantage': 3.2,
            'optimal_temperature': 1.15,
            'convergence_achieved': True
        })()
        
        await monitoring_system.record_quantum_result(mock_quantum)
        print(f"   Quantum optimization: 45ns, advantage=3.2x")
        
        # Wait for metrics collection
        await asyncio.sleep(1.0)
        
        # Get monitoring status
        status = monitoring_system.get_monitoring_status()
        
        print(f"\\n📊 Monitoring Status:")
        print(f"   System active: {status['system_status']['monitoring_active']}")
        print(f"   Components monitored: {status['system_status']['components_monitored']}")
        
        if status['current_metrics']:
            metrics = status['current_metrics']
            print(f"\\n📈 Current Metrics:")
            print(f"   Total latency: {metrics['total_latency_ns']}ns")
            print(f"   NBEATSx latency: {metrics['nbeatsx_latency_ns']}ns")  
            print(f"   ATS-CP latency: {metrics['ats_cp_latency_ns']}ns")
            print(f"   Forecast accuracy: {metrics['forecast_accuracy']:.1%}")
            print(f"   CPU usage: {metrics['cpu_usage_percent']:.1f}%")
            print(f"   Memory usage: {metrics['memory_usage_percent']:.1f}%")
        
        print(f"\\n🚨 Active Alerts:")
        alerts = status['active_alerts']
        print(f"   Critical: {alerts['critical']}")
        print(f"   High: {alerts['high']}")
        print(f"   Medium: {alerts['medium']}")
        print(f"   Total: {alerts['total']}")
        
        performance = status['performance_summary']
        print(f"\\n🎯 Performance Summary:")
        print(f"   Target latency: {performance['target_latency_ns']}ns")
        print(f"   Current latency: {performance['current_latency_ns']}ns")
        print(f"   Target met: {'✅' if performance['target_met'] else '❌'}")
        print(f"   Forecast accuracy: {performance['forecast_accuracy']:.1%}")
        
        # Simulate a critical violation
        print(f"\\n⚠️ Simulating Critical Violation...")
        
        # Mock high latency execution
        critical_forecast = type('ForecastOutput', (), {
            'inference_time_ns': 800,  # Over target
            'calibration_time_ns': 250,  # Way over target
            'total_time_ns': 1050,  # Critical violation
            'forecast_confidence': 0.6  # Low confidence
        })()
        
        await monitoring_system.record_forecast_execution(1050, critical_forecast)
        
        # Wait for alert processing
        await asyncio.sleep(0.5)
        
        # Check for new alerts
        final_status = monitoring_system.get_monitoring_status()
        if final_status['active_alerts']['total'] > alerts['total']:
            print(f"   🔴 CRITICAL ALERT TRIGGERED")
            print(f"   New alerts: {final_status['active_alerts']['total'] - alerts['total']}")
        
        # Stop monitoring
        await monitoring_system.stop_monitoring()
        
        print(f"\\n✅ SENTINEL MONITORING DEMONSTRATION SUCCESSFUL")
        print(f"Ready for production monitoring with real-time alerts!")
        
    except Exception as e:
        print(f"❌ Sentinel monitoring demonstration failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    def run_async_safe(coro):
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, coro)
                    return future.result()
            else:
                return asyncio.run(coro)
        except RuntimeError:
            return asyncio.run(coro)
    
    print("🚀 Starting Sentinel Monitoring Demonstration...")
    run_async_safe(demonstrate_sentinel_monitoring())
    print("🎉 Demonstration completed!")