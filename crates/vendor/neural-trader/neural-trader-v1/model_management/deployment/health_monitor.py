"""Health Monitoring System for Deployed AI Trading Models."""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import time
import requests
import statistics
from collections import deque, defaultdict
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class HealthMetric:
    """Individual health metric."""
    name: str
    value: float
    unit: str
    timestamp: datetime
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    
    def get_status(self) -> HealthStatus:
        """Get status based on thresholds."""
        if self.threshold_critical is not None and self.value >= self.threshold_critical:
            return HealthStatus.CRITICAL
        elif self.threshold_warning is not None and self.value >= self.threshold_warning:
            return HealthStatus.WARNING
        else:
            return HealthStatus.HEALTHY


@dataclass
class HealthReport:
    """Health report for a deployment."""
    deployment_id: str
    model_id: str
    timestamp: datetime
    overall_status: HealthStatus
    metrics: Dict[str, HealthMetric]
    endpoints_status: Dict[str, bool]
    response_times: Dict[str, float]
    error_count: int
    uptime_percentage: float
    alerts: List[Dict[str, Any]]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'deployment_id': self.deployment_id,
            'model_id': self.model_id,
            'timestamp': self.timestamp.isoformat(),
            'overall_status': self.overall_status.value,
            'metrics': {k: asdict(v) for k, v in self.metrics.items()},
            'endpoints_status': self.endpoints_status,
            'response_times': self.response_times,
            'error_count': self.error_count,
            'uptime_percentage': self.uptime_percentage,
            'alerts': self.alerts
        }


@dataclass
class AlertRule:
    """Alert rule configuration."""
    name: str
    metric_name: str
    condition: str  # 'greater_than', 'less_than', 'equals'
    threshold: float
    severity: AlertSeverity
    duration_minutes: int = 5  # How long condition must persist
    cooldown_minutes: int = 30  # Minimum time between alerts
    enabled: bool = True


@dataclass
class Alert:
    """System alert."""
    alert_id: str
    rule_name: str
    deployment_id: str
    model_id: str
    severity: AlertSeverity
    message: str
    triggered_at: datetime
    resolved_at: Optional[datetime] = None
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


class HealthMonitor:
    """Comprehensive health monitoring system for deployed models."""
    
    def __init__(self, storage_path: str = "model_management/monitoring"):
        """
        Initialize health monitor.
        
        Args:
            storage_path: Path for monitoring data storage
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Monitoring data
        self.health_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_rules: Dict[str, AlertRule] = {}
        self.deployment_endpoints: Dict[str, Dict[str, str]] = {}
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_tasks: List[asyncio.Task] = []
        self.check_interval = 60  # seconds
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Alert callbacks
        self.alert_callbacks: List[Callable] = []
        
        # Load configuration
        self._load_configuration()
        self._setup_default_rules()
        
        logger.info("Health Monitor initialized")
    
    def _load_configuration(self):
        """Load monitoring configuration."""
        config_file = self.storage_path / "config.json"
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                
                self.check_interval = config.get('check_interval', 60)
                
                # Load alert rules
                for rule_data in config.get('alert_rules', []):
                    rule = AlertRule(
                        name=rule_data['name'],
                        metric_name=rule_data['metric_name'],
                        condition=rule_data['condition'],
                        threshold=rule_data['threshold'],
                        severity=AlertSeverity(rule_data['severity']),
                        duration_minutes=rule_data.get('duration_minutes', 5),
                        cooldown_minutes=rule_data.get('cooldown_minutes', 30),
                        enabled=rule_data.get('enabled', True)
                    )
                    self.alert_rules[rule.name] = rule
                
                logger.info(f"Loaded {len(self.alert_rules)} alert rules")
                
            except Exception as e:
                logger.error(f"Failed to load configuration: {e}")
    
    def _setup_default_rules(self):
        """Setup default alert rules."""
        default_rules = [
            AlertRule(
                name="high_response_time",
                metric_name="avg_response_time",
                condition="greater_than",
                threshold=5000.0,  # 5 seconds
                severity=AlertSeverity.WARNING,
                duration_minutes=3
            ),
            AlertRule(
                name="critical_response_time",
                metric_name="avg_response_time",
                condition="greater_than",
                threshold=10000.0,  # 10 seconds
                severity=AlertSeverity.CRITICAL,
                duration_minutes=2
            ),
            AlertRule(
                name="low_uptime",
                metric_name="uptime_percentage",
                condition="less_than",
                threshold=95.0,
                severity=AlertSeverity.WARNING,
                duration_minutes=10
            ),
            AlertRule(
                name="critical_uptime",
                metric_name="uptime_percentage",
                condition="less_than",
                threshold=90.0,
                severity=AlertSeverity.CRITICAL,
                duration_minutes=5
            ),
            AlertRule(
                name="high_error_rate",
                metric_name="error_rate",
                condition="greater_than",
                threshold=5.0,  # 5%
                severity=AlertSeverity.WARNING,
                duration_minutes=5
            ),
            AlertRule(
                name="critical_error_rate",
                metric_name="error_rate",
                condition="greater_than",
                threshold=15.0,  # 15%
                severity=AlertSeverity.CRITICAL,
                duration_minutes=3
            )
        ]
        
        for rule in default_rules:
            if rule.name not in self.alert_rules:
                self.alert_rules[rule.name] = rule
    
    def register_deployment(self, deployment_id: str, model_id: str, 
                          endpoints: Dict[str, str]):
        """Register a deployment for monitoring."""
        with self.lock:
            self.deployment_endpoints[deployment_id] = {
                'model_id': model_id,
                **endpoints
            }
        
        logger.info(f"Registered deployment {deployment_id} for monitoring")
    
    def unregister_deployment(self, deployment_id: str):
        """Unregister a deployment from monitoring."""
        with self.lock:
            self.deployment_endpoints.pop(deployment_id, None)
            self.health_history.pop(deployment_id, None)
            
            # Remove related alerts
            alerts_to_remove = [
                alert_id for alert_id, alert in self.active_alerts.items()
                if alert.deployment_id == deployment_id
            ]
            for alert_id in alerts_to_remove:
                del self.active_alerts[alert_id]
        
        logger.info(f"Unregistered deployment {deployment_id} from monitoring")
    
    async def start_monitoring(self):
        """Start health monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        
        # Start monitoring tasks
        self.monitoring_tasks = [
            asyncio.create_task(self._health_check_loop()),
            asyncio.create_task(self._alert_processor()),
            asyncio.create_task(self._cleanup_old_data())
        ]
        
        logger.info("Health monitoring started")
    
    async def stop_monitoring(self):
        """Stop health monitoring."""
        self.monitoring_active = False
        
        # Cancel all tasks
        for task in self.monitoring_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.monitoring_tasks, return_exceptions=True)
        
        self.monitoring_tasks.clear()
        logger.info("Health monitoring stopped")
    
    async def _health_check_loop(self):
        """Main health check loop."""
        while self.monitoring_active:
            try:
                # Check health of all registered deployments
                for deployment_id, deployment_info in self.deployment_endpoints.items():
                    await self._check_deployment_health(deployment_id, deployment_info)
                
                await asyncio.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Health check loop error: {e}")
                await asyncio.sleep(self.check_interval)
    
    async def _check_deployment_health(self, deployment_id: str, 
                                     deployment_info: Dict[str, str]):
        """Check health of a specific deployment."""
        try:
            model_id = deployment_info['model_id']
            api_endpoint = deployment_info.get('api')
            
            if not api_endpoint:
                logger.warning(f"No API endpoint for deployment {deployment_id}")
                return
            
            # Perform health checks
            health_data = await self._perform_health_checks(api_endpoint)
            
            # Calculate metrics
            metrics = await self._calculate_health_metrics(deployment_id, health_data)
            
            # Determine overall status
            overall_status = self._determine_overall_status(metrics)
            
            # Calculate uptime
            uptime_percentage = self._calculate_uptime(deployment_id)
            
            # Create health report
            report = HealthReport(
                deployment_id=deployment_id,
                model_id=model_id,
                timestamp=datetime.now(),
                overall_status=overall_status,
                metrics=metrics,
                endpoints_status=health_data['endpoints_status'],
                response_times=health_data['response_times'],
                error_count=health_data['error_count'],
                uptime_percentage=uptime_percentage,
                alerts=[]
            )
            
            # Store health report
            with self.lock:
                self.health_history[deployment_id].append(report)
            
            # Check alert rules
            await self._check_alert_rules(report)
            
        except Exception as e:
            logger.error(f"Health check failed for deployment {deployment_id}: {e}")
            
            # Create error health report
            error_report = HealthReport(
                deployment_id=deployment_id,
                model_id=deployment_info['model_id'],
                timestamp=datetime.now(),
                overall_status=HealthStatus.CRITICAL,
                metrics={},
                endpoints_status={'api': False},
                response_times={},
                error_count=1,
                uptime_percentage=0.0,
                alerts=[]
            )
            
            with self.lock:
                self.health_history[deployment_id].append(error_report)
    
    async def _perform_health_checks(self, api_endpoint: str) -> Dict[str, Any]:
        """Perform various health checks on an endpoint."""
        health_data = {
            'endpoints_status': {},
            'response_times': {},
            'error_count': 0
        }
        
        # Health endpoint check
        try:
            start_time = time.time()
            response = requests.get(f"{api_endpoint}/health", timeout=10)
            response_time = (time.time() - start_time) * 1000  # ms
            
            if response.status_code == 200:
                health_data['endpoints_status']['health'] = True
                health_data['response_times']['health'] = response_time
                
                # Parse health response
                try:
                    health_response = response.json()
                    health_data['health_response'] = health_response
                except:
                    pass
            else:
                health_data['endpoints_status']['health'] = False
                health_data['error_count'] += 1
                
        except Exception as e:
            health_data['endpoints_status']['health'] = False
            health_data['error_count'] += 1
            logger.debug(f"Health check failed: {e}")
        
        # Metadata endpoint check
        try:
            start_time = time.time()
            response = requests.get(f"{api_endpoint}/metadata", timeout=10)
            response_time = (time.time() - start_time) * 1000  # ms
            
            if response.status_code == 200:
                health_data['endpoints_status']['metadata'] = True
                health_data['response_times']['metadata'] = response_time
            else:
                health_data['endpoints_status']['metadata'] = False
                health_data['error_count'] += 1
                
        except Exception as e:
            health_data['endpoints_status']['metadata'] = False
            health_data['error_count'] += 1
        
        # Prediction endpoint check (lightweight)
        try:
            start_time = time.time()
            test_data = {"input_data": {"test": "ping"}}
            response = requests.post(
                f"{api_endpoint}/predict", 
                json=test_data, 
                timeout=10
            )
            response_time = (time.time() - start_time) * 1000  # ms
            
            if response.status_code in [200, 422]:  # 422 for validation error is OK
                health_data['endpoints_status']['predict'] = True
                health_data['response_times']['predict'] = response_time
            else:
                health_data['endpoints_status']['predict'] = False
                health_data['error_count'] += 1
                
        except Exception as e:
            health_data['endpoints_status']['predict'] = False
            health_data['error_count'] += 1
        
        return health_data
    
    async def _calculate_health_metrics(self, deployment_id: str, 
                                      health_data: Dict[str, Any]) -> Dict[str, HealthMetric]:
        """Calculate health metrics from check data."""
        metrics = {}
        timestamp = datetime.now()
        
        # Average response time
        response_times = list(health_data['response_times'].values())
        if response_times:
            avg_response_time = statistics.mean(response_times)
            metrics['avg_response_time'] = HealthMetric(
                name="Average Response Time",
                value=avg_response_time,
                unit="ms",
                timestamp=timestamp,
                threshold_warning=3000.0,
                threshold_critical=8000.0
            )
        
        # Error rate (from recent history)
        error_rate = self._calculate_error_rate(deployment_id)
        metrics['error_rate'] = HealthMetric(
            name="Error Rate",
            value=error_rate,
            unit="%",
            timestamp=timestamp,
            threshold_warning=5.0,
            threshold_critical=15.0
        )
        
        # Endpoint availability
        available_endpoints = sum(health_data['endpoints_status'].values())
        total_endpoints = len(health_data['endpoints_status'])
        
        if total_endpoints > 0:
            availability_percentage = (available_endpoints / total_endpoints) * 100
            metrics['endpoint_availability'] = HealthMetric(
                name="Endpoint Availability",
                value=availability_percentage,
                unit="%",
                timestamp=timestamp,
                threshold_warning=90.0,
                threshold_critical=70.0
            )
        
        return metrics
    
    def _calculate_error_rate(self, deployment_id: str) -> float:
        """Calculate error rate from recent history."""
        with self.lock:
            recent_reports = list(self.health_history[deployment_id])[-10:]  # Last 10 reports
        
        if not recent_reports:
            return 0.0
        
        total_checks = len(recent_reports)
        error_checks = sum(1 for report in recent_reports 
                          if report.overall_status in [HealthStatus.CRITICAL, HealthStatus.UNKNOWN])
        
        return (error_checks / total_checks) * 100
    
    def _calculate_uptime(self, deployment_id: str) -> float:
        """Calculate uptime percentage from recent history."""
        with self.lock:
            recent_reports = list(self.health_history[deployment_id])[-100:]  # Last 100 reports
        
        if not recent_reports:
            return 100.0
        
        healthy_reports = sum(1 for report in recent_reports 
                             if report.overall_status in [HealthStatus.HEALTHY, HealthStatus.WARNING])
        
        return (healthy_reports / len(recent_reports)) * 100
    
    def _determine_overall_status(self, metrics: Dict[str, HealthMetric]) -> HealthStatus:
        """Determine overall health status from metrics."""
        if not metrics:
            return HealthStatus.UNKNOWN
        
        statuses = [metric.get_status() for metric in metrics.values()]
        
        if HealthStatus.CRITICAL in statuses:
            return HealthStatus.CRITICAL
        elif HealthStatus.WARNING in statuses:
            return HealthStatus.WARNING
        else:
            return HealthStatus.HEALTHY
    
    async def _check_alert_rules(self, report: HealthReport):
        """Check alert rules against health report."""
        for rule_name, rule in self.alert_rules.items():
            if not rule.enabled:
                continue
            
            # Check if metric exists
            if rule.metric_name in report.metrics:
                metric = report.metrics[rule.metric_name]
                await self._evaluate_alert_rule(rule, metric, report)
            elif rule.metric_name == 'uptime_percentage':
                # Special case for uptime percentage
                fake_metric = HealthMetric(
                    name="Uptime Percentage",
                    value=report.uptime_percentage,
                    unit="%",
                    timestamp=report.timestamp
                )
                await self._evaluate_alert_rule(rule, fake_metric, report)
    
    async def _evaluate_alert_rule(self, rule: AlertRule, metric: HealthMetric, 
                                 report: HealthReport):
        """Evaluate an alert rule against a metric."""
        condition_met = False
        
        if rule.condition == "greater_than":
            condition_met = metric.value > rule.threshold
        elif rule.condition == "less_than":
            condition_met = metric.value < rule.threshold
        elif rule.condition == "equals":
            condition_met = abs(metric.value - rule.threshold) < 0.001
        
        alert_key = f"{rule.name}_{report.deployment_id}"
        
        if condition_met:
            # Check if alert already exists
            if alert_key not in self.active_alerts:
                # Create new alert
                alert = Alert(
                    alert_id=alert_key,
                    rule_name=rule.name,
                    deployment_id=report.deployment_id,
                    model_id=report.model_id,
                    severity=rule.severity,
                    message=f"{rule.name}: {metric.name} is {metric.value:.2f} {metric.unit} (threshold: {rule.threshold:.2f})",
                    triggered_at=datetime.now()
                )
                
                self.active_alerts[alert_key] = alert
                await self._trigger_alert(alert)
        else:
            # Condition not met, resolve alert if it exists
            if alert_key in self.active_alerts:
                alert = self.active_alerts[alert_key]
                alert.resolved_at = datetime.now()
                await self._resolve_alert(alert)
                del self.active_alerts[alert_key]
    
    async def _trigger_alert(self, alert: Alert):
        """Trigger an alert."""
        logger.warning(f"ALERT TRIGGERED: {alert.message}")
        
        # Call alert callbacks
        for callback in self.alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert)
                else:
                    callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
    
    async def _resolve_alert(self, alert: Alert):
        """Resolve an alert."""
        logger.info(f"ALERT RESOLVED: {alert.rule_name} for {alert.deployment_id}")
        
        # Could trigger resolution callbacks here
    
    async def _alert_processor(self):
        """Process and manage alerts."""
        while self.monitoring_active:
            try:
                # Clean up old resolved alerts
                current_time = datetime.now()
                alerts_to_clean = []
                
                for alert_id, alert in self.active_alerts.items():
                    if (alert.resolved_at and 
                        current_time - alert.resolved_at > timedelta(hours=24)):
                        alerts_to_clean.append(alert_id)
                
                for alert_id in alerts_to_clean:
                    del self.active_alerts[alert_id]
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Alert processor error: {e}")
                await asyncio.sleep(300)
    
    async def _cleanup_old_data(self):
        """Clean up old monitoring data."""
        while self.monitoring_active:
            try:
                # Clean up old health history (keep last 24 hours)
                cutoff_time = datetime.now() - timedelta(hours=24)
                
                with self.lock:
                    for deployment_id, history in self.health_history.items():
                        # Filter out old reports
                        recent_reports = deque(
                            [report for report in history if report.timestamp > cutoff_time],
                            maxlen=1000
                        )
                        self.health_history[deployment_id] = recent_reports
                
                await asyncio.sleep(3600)  # Clean every hour
                
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
                await asyncio.sleep(3600)
    
    def add_alert_callback(self, callback: Callable):
        """Add callback for alert notifications."""
        self.alert_callbacks.append(callback)
    
    def get_deployment_health(self, deployment_id: str) -> Optional[HealthReport]:
        """Get latest health report for a deployment."""
        with self.lock:
            history = self.health_history.get(deployment_id)
            if history:
                return history[-1]  # Latest report
        return None
    
    def get_deployment_history(self, deployment_id: str, 
                             hours: int = 24) -> List[HealthReport]:
        """Get health history for a deployment."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self.lock:
            history = self.health_history.get(deployment_id, deque())
            return [report for report in history if report.timestamp > cutoff_time]
    
    def get_active_alerts(self, deployment_id: str = None, 
                         severity: AlertSeverity = None) -> List[Alert]:
        """Get active alerts with optional filtering."""
        alerts = []
        
        for alert in self.active_alerts.values():
            if deployment_id and alert.deployment_id != deployment_id:
                continue
            if severity and alert.severity != severity:
                continue
            if alert.resolved_at is None:  # Only active alerts
                alerts.append(alert)
        
        # Sort by severity and time
        severity_order = {
            AlertSeverity.EMERGENCY: 0,
            AlertSeverity.CRITICAL: 1,
            AlertSeverity.WARNING: 2,
            AlertSeverity.INFO: 3
        }
        
        alerts.sort(key=lambda a: (severity_order[a.severity], a.triggered_at))
        return alerts
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.acknowledged = True
            alert.acknowledged_by = acknowledged_by
            alert.acknowledged_at = datetime.now()
            
            logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
            return True
        
        return False
    
    def get_system_overview(self) -> Dict[str, Any]:
        """Get system-wide health overview."""
        total_deployments = len(self.deployment_endpoints)
        healthy_deployments = 0
        warning_deployments = 0
        critical_deployments = 0
        
        for deployment_id in self.deployment_endpoints:
            latest_health = self.get_deployment_health(deployment_id)
            if latest_health:
                if latest_health.overall_status == HealthStatus.HEALTHY:
                    healthy_deployments += 1
                elif latest_health.overall_status == HealthStatus.WARNING:
                    warning_deployments += 1
                elif latest_health.overall_status == HealthStatus.CRITICAL:
                    critical_deployments += 1
        
        active_alerts = self.get_active_alerts()
        critical_alerts = len([a for a in active_alerts if a.severity == AlertSeverity.CRITICAL])
        warning_alerts = len([a for a in active_alerts if a.severity == AlertSeverity.WARNING])
        
        return {
            'total_deployments': total_deployments,
            'healthy_deployments': healthy_deployments,
            'warning_deployments': warning_deployments,
            'critical_deployments': critical_deployments,
            'total_active_alerts': len(active_alerts),
            'critical_alerts': critical_alerts,
            'warning_alerts': warning_alerts,
            'monitoring_active': self.monitoring_active,
            'last_check': datetime.now().isoformat()
        }


# Factory function
def create_health_monitor(storage_path: str = "model_management/monitoring") -> HealthMonitor:
    """Create health monitor instance."""
    return HealthMonitor(storage_path)