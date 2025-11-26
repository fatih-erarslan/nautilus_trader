"""
Alert system for trading monitoring.
Handles latency alerts, risk breaches, and performance degradation.
"""

import asyncio
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
from collections import deque
import json

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertType(Enum):
    """Types of alerts."""
    LATENCY = "latency"
    RISK_BREACH = "risk_breach"
    CONNECTION_ISSUE = "connection_issue"
    PERFORMANCE = "performance"
    DATA_QUALITY = "data_quality"
    SYSTEM = "system"


@dataclass
class Alert:
    """Represents a single alert."""
    id: str
    timestamp: datetime
    type: AlertType
    severity: AlertSeverity
    title: str
    message: str
    data: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    resolution_message: Optional[str] = None
    
    def resolve(self, message: str = "Alert resolved"):
        """Mark alert as resolved."""
        self.resolved = True
        self.resolved_at = datetime.now()
        self.resolution_message = message
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'type': self.type.value,
            'severity': self.severity.value,
            'title': self.title,
            'message': self.message,
            'data': self.data,
            'resolved': self.resolved,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None,
            'resolution_message': self.resolution_message
        }


@dataclass
class AlertRule:
    """Defines conditions for triggering alerts."""
    name: str
    type: AlertType
    condition: Callable[[Any], bool]
    severity: AlertSeverity
    title_template: str
    message_template: str
    cooldown_seconds: int = 300  # 5 minutes default
    auto_resolve: bool = False
    auto_resolve_condition: Optional[Callable[[Any], bool]] = None


class AlertSystem:
    """
    Comprehensive alert management system.
    """
    
    def __init__(self, max_alerts: int = 1000):
        """
        Initialize alert system.
        
        Args:
            max_alerts: Maximum number of alerts to keep in memory
        """
        self.max_alerts = max_alerts
        self.alerts: deque = deque(maxlen=max_alerts)
        self.active_alerts: Dict[str, Alert] = {}
        self.rules: Dict[str, AlertRule] = {}
        self.last_alert_times: Dict[str, datetime] = {}
        
        # Alert handlers
        self.handlers: List[Callable] = []
        
        # Statistics
        self.alert_counts: Dict[AlertType, int] = {t: 0 for t in AlertType}
        self.severity_counts: Dict[AlertSeverity, int] = {s: 0 for s in AlertSeverity}
        
        self._lock = asyncio.Lock()
        self._setup_default_rules()
    
    def _setup_default_rules(self):
        """Set up default alert rules."""
        # Latency alerts
        self.add_rule(AlertRule(
            name="high_latency",
            type=AlertType.LATENCY,
            condition=lambda data: data.get('latency_ms', 0) > 100,
            severity=AlertSeverity.WARNING,
            title_template="High Latency Detected",
            message_template="Latency exceeded threshold: {latency_ms:.1f}ms",
            cooldown_seconds=60
        ))
        
        self.add_rule(AlertRule(
            name="critical_latency",
            type=AlertType.LATENCY,
            condition=lambda data: data.get('latency_ms', 0) > 500,
            severity=AlertSeverity.CRITICAL,
            title_template="Critical Latency",
            message_template="Critical latency detected: {latency_ms:.1f}ms",
            cooldown_seconds=30
        ))
        
        # Risk alerts
        self.add_rule(AlertRule(
            name="position_limit_breach",
            type=AlertType.RISK_BREACH,
            condition=lambda data: data.get('position_size', 0) > data.get('max_position_size', float('inf')),
            severity=AlertSeverity.ERROR,
            title_template="Position Limit Breach",
            message_template="Position size {position_size} exceeds limit {max_position_size}",
            cooldown_seconds=0  # No cooldown for risk breaches
        ))
        
        self.add_rule(AlertRule(
            name="drawdown_limit",
            type=AlertType.RISK_BREACH,
            condition=lambda data: data.get('drawdown_pct', 0) > data.get('max_drawdown_pct', 10),
            severity=AlertSeverity.ERROR,
            title_template="Drawdown Limit Exceeded",
            message_template="Drawdown {drawdown_pct:.1f}% exceeds limit {max_drawdown_pct}%",
            cooldown_seconds=300
        ))
        
        # Connection alerts
        self.add_rule(AlertRule(
            name="connection_lost",
            type=AlertType.CONNECTION_ISSUE,
            condition=lambda data: data.get('status') == 'disconnected',
            severity=AlertSeverity.ERROR,
            title_template="Connection Lost",
            message_template="WebSocket connection lost: {error}",
            cooldown_seconds=60,
            auto_resolve=True,
            auto_resolve_condition=lambda data: data.get('status') == 'connected'
        ))
        
        # Performance alerts
        self.add_rule(AlertRule(
            name="low_message_rate",
            type=AlertType.PERFORMANCE,
            condition=lambda data: data.get('message_rate', 0) < 1 and data.get('expected_rate', 0) > 0,
            severity=AlertSeverity.WARNING,
            title_template="Low Message Rate",
            message_template="Message rate {message_rate:.1f}/s below expected",
            cooldown_seconds=300
        ))
    
    def add_rule(self, rule: AlertRule):
        """Add an alert rule."""
        self.rules[rule.name] = rule
    
    def remove_rule(self, rule_name: str):
        """Remove an alert rule."""
        if rule_name in self.rules:
            del self.rules[rule_name]
    
    async def check_condition(self, rule_name: str, data: Dict[str, Any]) -> Optional[Alert]:
        """
        Check if a specific rule's condition is met.
        
        Args:
            rule_name: Name of the rule to check
            data: Data to check against the condition
            
        Returns:
            Alert if condition is met and cooldown passed, None otherwise
        """
        async with self._lock:
            if rule_name not in self.rules:
                return None
            
            rule = self.rules[rule_name]
            
            # Check condition
            try:
                if not rule.condition(data):
                    # Check auto-resolve
                    if rule.auto_resolve and rule_name in self.active_alerts:
                        if rule.auto_resolve_condition and rule.auto_resolve_condition(data):
                            alert = self.active_alerts[rule_name]
                            alert.resolve("Condition no longer met")
                            del self.active_alerts[rule_name]
                            await self._trigger_handlers(alert, resolved=True)
                    return None
            except Exception as e:
                logger.error(f"Error checking condition for rule {rule_name}: {e}")
                return None
            
            # Check cooldown
            if rule_name in self.last_alert_times:
                time_since_last = (datetime.now() - self.last_alert_times[rule_name]).total_seconds()
                if time_since_last < rule.cooldown_seconds:
                    return None
            
            # Create alert
            alert = await self._create_alert(rule, data)
            return alert
    
    async def _create_alert(self, rule: AlertRule, data: Dict[str, Any]) -> Alert:
        """Create and register an alert."""
        alert_id = f"{rule.type.value}_{datetime.now().timestamp()}"
        
        # Format message with data
        try:
            title = rule.title_template.format(**data)
            message = rule.message_template.format(**data)
        except Exception as e:
            title = rule.title_template
            message = f"{rule.message_template} (formatting error: {e})"
        
        alert = Alert(
            id=alert_id,
            timestamp=datetime.now(),
            type=rule.type,
            severity=rule.severity,
            title=title,
            message=message,
            data=data
        )
        
        # Register alert
        self.alerts.append(alert)
        if rule.auto_resolve:
            self.active_alerts[rule.name] = alert
        
        # Update statistics
        self.alert_counts[rule.type] += 1
        self.severity_counts[rule.severity] += 1
        
        # Update cooldown
        self.last_alert_times[rule.name] = datetime.now()
        
        # Trigger handlers
        await self._trigger_handlers(alert)
        
        return alert
    
    async def trigger_custom_alert(
        self,
        type: AlertType,
        severity: AlertSeverity,
        title: str,
        message: str,
        data: Optional[Dict[str, Any]] = None
    ) -> Alert:
        """
        Trigger a custom alert without a predefined rule.
        
        Args:
            type: Alert type
            severity: Alert severity
            title: Alert title
            message: Alert message
            data: Optional alert data
            
        Returns:
            Created alert
        """
        async with self._lock:
            alert = Alert(
                id=f"{type.value}_{datetime.now().timestamp()}",
                timestamp=datetime.now(),
                type=type,
                severity=severity,
                title=title,
                message=message,
                data=data or {}
            )
            
            self.alerts.append(alert)
            self.alert_counts[type] += 1
            self.severity_counts[severity] += 1
            
            await self._trigger_handlers(alert)
            
            return alert
    
    async def check_latency(self, operation: str, latency_ms: float):
        """Check latency and trigger alerts if needed."""
        data = {
            'operation': operation,
            'latency_ms': latency_ms
        }
        
        await self.check_condition('high_latency', data)
        await self.check_condition('critical_latency', data)
    
    async def check_risk(self, metrics: Dict[str, float]):
        """Check risk metrics and trigger alerts if needed."""
        # Check position limits
        if 'position_size' in metrics and 'max_position_size' in metrics:
            await self.check_condition('position_limit_breach', metrics)
        
        # Check drawdown
        if 'drawdown_pct' in metrics and 'max_drawdown_pct' in metrics:
            await self.check_condition('drawdown_limit', metrics)
    
    async def check_connection(self, status: str, error: Optional[str] = None):
        """Check connection status and trigger alerts if needed."""
        data = {
            'status': status,
            'error': error or 'Unknown error'
        }
        
        await self.check_condition('connection_lost', data)
    
    async def check_performance(self, metrics: Dict[str, float]):
        """Check performance metrics and trigger alerts if needed."""
        if 'message_rate' in metrics:
            await self.check_condition('low_message_rate', metrics)
    
    def add_handler(self, handler: Callable):
        """
        Add an alert handler.
        
        Handler signature: handler(alert: Alert, resolved: bool = False)
        """
        self.handlers.append(handler)
    
    async def _trigger_handlers(self, alert: Alert, resolved: bool = False):
        """Trigger all alert handlers."""
        for handler in self.handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(alert, resolved)
                else:
                    handler(alert, resolved)
            except Exception as e:
                logger.error(f"Error in alert handler: {e}")
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active (unresolved) alerts."""
        return [a for a in self.alerts if not a.resolved]
    
    def get_alerts_by_type(self, alert_type: AlertType) -> List[Alert]:
        """Get alerts of a specific type."""
        return [a for a in self.alerts if a.type == alert_type]
    
    def get_alerts_by_severity(self, severity: AlertSeverity) -> List[Alert]:
        """Get alerts of a specific severity."""
        return [a for a in self.alerts if a.severity == severity]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get alert statistics."""
        active_alerts = self.get_active_alerts()
        
        return {
            'total_alerts': len(self.alerts),
            'active_alerts': len(active_alerts),
            'by_type': dict(self.alert_counts),
            'by_severity': dict(self.severity_counts),
            'active_by_severity': {
                s.value: sum(1 for a in active_alerts if a.severity == s)
                for s in AlertSeverity
            }
        }
    
    def export_alerts(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Export alerts as list of dictionaries."""
        alerts = list(self.alerts)
        if limit:
            alerts = alerts[-limit:]
        
        return [alert.to_dict() for alert in alerts]
    
    def to_json(self) -> str:
        """Export alert data as JSON."""
        data = {
            'statistics': self.get_statistics(),
            'active_alerts': [a.to_dict() for a in self.get_active_alerts()],
            'recent_alerts': self.export_alerts(limit=50)
        }
        return json.dumps(data, indent=2)