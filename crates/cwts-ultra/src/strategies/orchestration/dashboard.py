"""
PHASE 9 HIVE MIND ORCHESTRATION - SYSTEM ORCHESTRATION DASHBOARD
Real-time monitoring, coordination, and executive oversight of collective intelligence
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

@dataclass
class MetricSnapshot:
    """Snapshot of system metrics at a point in time"""
    timestamp: datetime
    metric_type: str
    metric_name: str
    value: float
    agent_id: Optional[str] = None
    additional_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'metric_type': self.metric_type,
            'metric_name': self.metric_name,
            'value': self.value,
            'agent_id': self.agent_id,
            'additional_data': self.additional_data
        }

@dataclass
class SystemAlert:
    """System alert for critical events"""
    alert_id: str
    alert_type: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    message: str
    source: str
    timestamp: datetime
    acknowledged: bool = False
    resolution_time: Optional[datetime] = None
    additional_context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CoordinationMetrics:
    """Metrics for agent coordination effectiveness"""
    coordination_id: str
    participating_agents: List[str]
    coordination_type: str
    start_time: datetime
    end_time: Optional[datetime]
    success_rate: float
    message_count: int
    response_times: List[float]
    coordination_quality: float
    bottlenecks: List[str] = field(default_factory=list)

class RealTimeMetricsCollector:
    """Collects and processes real-time metrics from all system components"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics_buffer = deque(maxlen=100000)  # Large buffer for high-frequency metrics
        self.aggregation_intervals = {
            'second': deque(maxlen=3600),    # Last hour by second
            'minute': deque(maxlen=1440),    # Last day by minute
            'hour': deque(maxlen=168),       # Last week by hour
            'day': deque(maxlen=30)          # Last month by day
        }
        
        # Metric aggregators
        self.metric_aggregators = {
            'performance': self._aggregate_performance_metrics,
            'coordination': self._aggregate_coordination_metrics,
            'system': self._aggregate_system_metrics,
            'agent': self._aggregate_agent_metrics
        }
        
        # Real-time computations
        self.running_averages = defaultdict(lambda: deque(maxlen=100))
        self.metric_trends = defaultdict(list)
        self.anomaly_detectors = {}
        
        self.collection_lock = threading.RLock()
        self.last_aggregation = datetime.now()
        
    async def collect_metric(self, metric: MetricSnapshot):
        """Collect a single metric snapshot"""
        
        with self.collection_lock:
            self.metrics_buffer.append(metric)
            
            # Update running averages
            metric_key = f"{metric.metric_type}_{metric.metric_name}"
            self.running_averages[metric_key].append(metric.value)
            
            # Detect anomalies
            await self._detect_anomalies(metric)
            
            # Trigger aggregation if needed
            if (datetime.now() - self.last_aggregation).total_seconds() > 60:  # Every minute
                await self._aggregate_metrics()
                
    async def _detect_anomalies(self, metric: MetricSnapshot):
        """Detect anomalies in real-time metrics"""
        
        metric_key = f"{metric.metric_type}_{metric.metric_name}"
        history = list(self.running_averages[metric_key])
        
        if len(history) >= 10:  # Need minimum history
            mean = np.mean(history[:-1])  # Exclude current value
            std = np.std(history[:-1])
            
            # Z-score anomaly detection
            z_score = abs(metric.value - mean) / (std + 0.001)  # Avoid division by zero
            
            if z_score > 3.0:  # 3-sigma rule
                anomaly_alert = SystemAlert(
                    alert_id=f"anomaly_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
                    alert_type='anomaly_detection',
                    severity='medium' if z_score < 4.0 else 'high',
                    message=f"Anomaly detected in {metric_key}: value={metric.value:.4f}, z-score={z_score:.2f}",
                    source='metrics_collector',
                    timestamp=datetime.now(),
                    additional_context={
                        'metric': metric.to_dict(),
                        'z_score': z_score,
                        'historical_mean': mean,
                        'historical_std': std
                    }
                )
                
                # This would typically trigger an alert to the dashboard
                # For now, we'll store it for later processing
                
    async def _aggregate_metrics(self):
        """Aggregate metrics across different time intervals"""
        
        current_time = datetime.now()
        
        with self.collection_lock:
            # Get metrics from last aggregation
            recent_metrics = [m for m in self.metrics_buffer 
                            if m.timestamp > self.last_aggregation]
            
            if not recent_metrics:
                return
                
            # Aggregate by type
            for metric_type, aggregator in self.metric_aggregators.items():
                type_metrics = [m for m in recent_metrics if m.metric_type == metric_type]
                if type_metrics:
                    aggregated = await aggregator(type_metrics, current_time)
                    
                    # Store in appropriate interval buckets
                    for interval in self.aggregation_intervals:
                        self.aggregation_intervals[interval].append({
                            'timestamp': current_time,
                            'metric_type': metric_type,
                            'aggregated_data': aggregated
                        })
                        
            self.last_aggregation = current_time
            
    async def _aggregate_performance_metrics(self, metrics: List[MetricSnapshot], 
                                           timestamp: datetime) -> Dict[str, Any]:
        """Aggregate performance-related metrics"""
        
        values = [m.value for m in metrics]
        
        return {
            'count': len(metrics),
            'mean': np.mean(values),
            'median': np.median(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'p95': np.percentile(values, 95),
            'p99': np.percentile(values, 99),
            'trend': self._calculate_trend(values),
            'quality_score': np.mean([v for v in values if v <= 1.0])  # Assuming normalized values
        }
        
    async def _aggregate_coordination_metrics(self, metrics: List[MetricSnapshot], 
                                            timestamp: datetime) -> Dict[str, Any]:
        """Aggregate coordination-related metrics"""
        
        # Group by agent for coordination analysis
        agent_metrics = defaultdict(list)
        for metric in metrics:
            if metric.agent_id:
                agent_metrics[metric.agent_id].append(metric.value)
                
        return {
            'total_coordination_events': len(metrics),
            'participating_agents': len(agent_metrics),
            'average_coordination_quality': np.mean([m.value for m in metrics]),
            'agent_participation': {
                agent: {
                    'event_count': len(values),
                    'average_quality': np.mean(values)
                }
                for agent, values in agent_metrics.items()
            },
            'coordination_balance': self._calculate_coordination_balance(agent_metrics),
            'system_sync_level': np.std([np.mean(values) for values in agent_metrics.values()]) if agent_metrics else 0
        }
        
    async def _aggregate_system_metrics(self, metrics: List[MetricSnapshot], 
                                      timestamp: datetime) -> Dict[str, Any]:
        """Aggregate system-level metrics"""
        
        metric_groups = defaultdict(list)
        for metric in metrics:
            metric_groups[metric.metric_name].append(metric.value)
            
        return {
            'system_health_score': np.mean([np.mean(values) for values in metric_groups.values()]),
            'metric_diversity': len(metric_groups),
            'stability_score': 1.0 - np.mean([np.std(values) for values in metric_groups.values()]),
            'resource_utilization': metric_groups.get('resource_utilization', [0.5])[0] if 'resource_utilization' in metric_groups else 0.5,
            'throughput': sum(metric_groups.get('throughput', [])),
            'error_rate': np.mean(metric_groups.get('error_rate', [0.0]))
        }
        
    async def _aggregate_agent_metrics(self, metrics: List[MetricSnapshot], 
                                     timestamp: datetime) -> Dict[str, Any]:
        """Aggregate agent-specific metrics"""
        
        agent_performance = defaultdict(lambda: defaultdict(list))
        
        for metric in metrics:
            if metric.agent_id:
                agent_performance[metric.agent_id][metric.metric_name].append(metric.value)
                
        aggregated_agents = {}
        for agent_id, agent_metrics in agent_performance.items():
            agent_summary = {}
            for metric_name, values in agent_metrics.items():
                agent_summary[metric_name] = {
                    'mean': np.mean(values),
                    'count': len(values),
                    'trend': self._calculate_trend(values)
                }
                
            aggregated_agents[agent_id] = {
                'metrics_summary': agent_summary,
                'overall_performance': np.mean([np.mean(values) for values in agent_metrics.values()]),
                'activity_level': sum([len(values) for values in agent_metrics.values()]),
                'consistency_score': 1.0 - np.mean([np.std(values) for values in agent_metrics.values()])
            }
            
        return aggregated_agents
        
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from values"""
        if len(values) < 2:
            return 'stable'
            
        # Simple linear regression slope
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope > 0.01:
            return 'increasing'
        elif slope < -0.01:
            return 'decreasing'
        else:
            return 'stable'
            
    def _calculate_coordination_balance(self, agent_metrics: Dict[str, List[float]]) -> float:
        """Calculate how balanced coordination is across agents"""
        if not agent_metrics:
            return 1.0
            
        agent_averages = [np.mean(values) for values in agent_metrics.values()]
        return 1.0 - (np.std(agent_averages) / (np.mean(agent_averages) + 0.001))

class AlertManager:
    """Manages system alerts and notifications"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.active_alerts: Dict[str, SystemAlert] = {}
        self.alert_history = deque(maxlen=10000)
        self.alert_rules = self._initialize_alert_rules()
        self.notification_channels = []
        self.alert_lock = threading.RLock()
        
    def _initialize_alert_rules(self) -> Dict[str, Dict[str, Any]]:
        """Initialize alert rules and thresholds"""
        return {
            'system_health': {
                'threshold': 0.7,
                'operator': 'less_than',
                'severity': 'high',
                'message_template': 'System health below threshold: {value:.3f}'
            },
            'agent_failure': {
                'threshold': 0.5,
                'operator': 'less_than',
                'severity': 'critical',
                'message_template': 'Agent {agent_id} performance critical: {value:.3f}'
            },
            'coordination_breakdown': {
                'threshold': 0.6,
                'operator': 'less_than',
                'severity': 'high',
                'message_template': 'Coordination effectiveness low: {value:.3f}'
            },
            'memory_usage': {
                'threshold': 0.9,
                'operator': 'greater_than',
                'severity': 'medium',
                'message_template': 'High memory usage: {value:.1%}'
            },
            'response_time': {
                'threshold': 5.0,  # 5 seconds
                'operator': 'greater_than',
                'severity': 'medium',
                'message_template': 'High response time: {value:.2f}s'
            }
        }
        
    async def evaluate_metric_for_alerts(self, metric: MetricSnapshot):
        """Evaluate metric against alert rules"""
        
        # Check if metric triggers any alert rules
        for rule_name, rule_config in self.alert_rules.items():
            if await self._metric_matches_rule(metric, rule_name, rule_config):
                await self._trigger_alert(rule_name, rule_config, metric)
                
    async def _metric_matches_rule(self, metric: MetricSnapshot, 
                                 rule_name: str, rule_config: Dict[str, Any]) -> bool:
        """Check if metric matches alert rule conditions"""
        
        # Simple rule matching - in production, this would be more sophisticated
        threshold = rule_config['threshold']
        operator = rule_config['operator']
        value = metric.value
        
        if operator == 'less_than':
            return value < threshold
        elif operator == 'greater_than':
            return value > threshold
        elif operator == 'equals':
            return abs(value - threshold) < 0.001
        else:
            return False
            
    async def _trigger_alert(self, rule_name: str, rule_config: Dict[str, Any], 
                           metric: MetricSnapshot):
        """Trigger an alert based on rule violation"""
        
        alert_id = f"alert_{rule_name}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        # Check for duplicate alerts (debouncing)
        existing_similar = [
            alert for alert in self.active_alerts.values()
            if (alert.alert_type == rule_name and 
                not alert.acknowledged and
                (datetime.now() - alert.timestamp).total_seconds() < 300)  # 5 minutes
        ]
        
        if existing_similar:
            return  # Don't spam similar alerts
            
        # Create alert
        message = rule_config['message_template'].format(
            value=metric.value,
            agent_id=metric.agent_id or 'system'
        )
        
        alert = SystemAlert(
            alert_id=alert_id,
            alert_type=rule_name,
            severity=rule_config['severity'],
            message=message,
            source='alert_manager',
            timestamp=datetime.now(),
            additional_context={
                'metric': metric.to_dict(),
                'rule': rule_config,
                'rule_name': rule_name
            }
        )
        
        with self.alert_lock:
            self.active_alerts[alert_id] = alert
            self.alert_history.append(alert)
            
        # Send notifications
        await self._send_alert_notifications(alert)
        
    async def _send_alert_notifications(self, alert: SystemAlert):
        """Send alert notifications through configured channels"""
        
        # In a real implementation, this would send to various channels
        # (email, Slack, SMS, etc.)
        notification_message = {
            'alert_id': alert.alert_id,
            'severity': alert.severity,
            'message': alert.message,
            'timestamp': alert.timestamp.isoformat(),
            'source': alert.source
        }
        
        # Log for now (in production, send to external systems)
        logging.getLogger('AlertManager').warning(f"ALERT [{alert.severity.upper()}]: {alert.message}")
        
    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an alert"""
        
        with self.alert_lock:
            alert = self.active_alerts.get(alert_id)
            if alert and not alert.acknowledged:
                alert.acknowledged = True
                alert.resolution_time = datetime.now()
                alert.additional_context['acknowledged_by'] = acknowledged_by
                return True
                
        return False
        
    def get_active_alerts(self) -> List[SystemAlert]:
        """Get all active (unacknowledged) alerts"""
        with self.alert_lock:
            return [alert for alert in self.active_alerts.values() if not alert.acknowledged]
            
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert summary statistics"""
        
        with self.alert_lock:
            active_alerts = [alert for alert in self.active_alerts.values() if not alert.acknowledged]
            
            severity_counts = defaultdict(int)
            for alert in active_alerts:
                severity_counts[alert.severity] += 1
                
            recent_alerts = [alert for alert in self.alert_history 
                           if (datetime.now() - alert.timestamp).total_seconds() < 3600]  # Last hour
                           
            return {
                'active_alerts_count': len(active_alerts),
                'severity_breakdown': dict(severity_counts),
                'recent_alerts_count': len(recent_alerts),
                'total_alerts_today': len([
                    alert for alert in self.alert_history
                    if alert.timestamp.date() == datetime.now().date()
                ]),
                'most_frequent_alert_type': max(
                    [alert.alert_type for alert in recent_alerts],
                    key=[alert.alert_type for alert in recent_alerts].count
                ) if recent_alerts else None
            }

class SystemOrchestrationDashboard:
    """
    Main dashboard for real-time monitoring and coordination of the hive mind system
    Provides executive-level oversight and operational intelligence
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = self._setup_logging()
        
        # Core components
        self.metrics_collector = RealTimeMetricsCollector(config.get('metrics', {}))
        self.alert_manager = AlertManager(config.get('alerts', {}))
        
        # Dashboard state
        self.system_status = {
            'overall_health': 1.0,
            'coordination_effectiveness': 1.0,
            'performance_score': 1.0,
            'agent_status_summary': {},
            'active_coordination_sessions': 0,
            'last_updated': datetime.now()
        }
        
        # Real-time data streams
        self.live_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.coordination_sessions: Dict[str, CoordinationMetrics] = {}
        
        # Dashboard configuration
        self.update_interval = config.get('update_interval_seconds', 30)
        self.dashboard_lock = threading.RLock()
        
        # Start background monitoring
        self.monitoring_active = True
        self.monitoring_task = None
        
    def _setup_logging(self) -> logging.Logger:
        """Setup dashboard logging"""
        logger = logging.getLogger('SystemOrchestrationDashboard')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
        
    async def start_monitoring(self):
        """Start real-time monitoring"""
        if not self.monitoring_task:
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            self.logger.info("Dashboard monitoring started")
            
    async def stop_monitoring(self):
        """Stop real-time monitoring"""
        self.monitoring_active = False
        if self.monitoring_task:
            await self.monitoring_task
            self.monitoring_task = None
            self.logger.info("Dashboard monitoring stopped")
            
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                await self._update_dashboard_state()
                await asyncio.sleep(self.update_interval)
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {str(e)}")
                await asyncio.sleep(5)  # Brief pause before retrying
                
    async def _update_dashboard_state(self):
        """Update overall dashboard state"""
        
        with self.dashboard_lock:
            # Calculate current system metrics
            current_time = datetime.now()
            
            # Get recent metrics for analysis
            recent_metrics = [m for m in self.metrics_collector.metrics_buffer 
                            if (current_time - m.timestamp).total_seconds() < 300]  # Last 5 minutes
            
            if recent_metrics:
                # Update system health
                health_metrics = [m for m in recent_metrics if 'health' in m.metric_name]
                if health_metrics:
                    self.system_status['overall_health'] = np.mean([m.value for m in health_metrics])
                    
                # Update coordination effectiveness
                coord_metrics = [m for m in recent_metrics if 'coordination' in m.metric_type]
                if coord_metrics:
                    self.system_status['coordination_effectiveness'] = np.mean([m.value for m in coord_metrics])
                    
                # Update performance score
                perf_metrics = [m for m in recent_metrics if 'performance' in m.metric_type]
                if perf_metrics:
                    self.system_status['performance_score'] = np.mean([m.value for m in perf_metrics])
                    
                # Update agent status summary
                agent_metrics = defaultdict(list)
                for metric in recent_metrics:
                    if metric.agent_id:
                        agent_metrics[metric.agent_id].append(metric.value)
                        
                self.system_status['agent_status_summary'] = {
                    agent_id: {
                        'status': 'active',
                        'performance': np.mean(values),
                        'activity_level': len(values)
                    }
                    for agent_id, values in agent_metrics.items()
                }
                
            # Update coordination sessions count
            active_sessions = sum(1 for session in self.coordination_sessions.values() 
                                if session.end_time is None)
            self.system_status['active_coordination_sessions'] = active_sessions
            
            # Update timestamp
            self.system_status['last_updated'] = current_time
            
    async def update_metrics(self, execution_result: Dict[str, Any]):
        """Update dashboard with execution results"""
        
        try:
            current_time = datetime.now()
            
            # Create metric snapshots from execution results
            metrics = []
            
            # Overall performance metric
            overall_metric = MetricSnapshot(
                timestamp=current_time,
                metric_type='performance',
                metric_name='execution_success_rate',
                value=execution_result.get('success_rate', 0.5)
            )
            metrics.append(overall_metric)
            
            # Coordination effectiveness metric
            coord_metric = MetricSnapshot(
                timestamp=current_time,
                metric_type='coordination',
                metric_name='coordination_effectiveness',
                value=execution_result.get('coordination_effectiveness', 0.8)
            )
            metrics.append(coord_metric)
            
            # Agent-specific metrics
            agent_results = execution_result.get('agent_results', {})
            for agent_id, agent_result in agent_results.items():
                agent_metric = MetricSnapshot(
                    timestamp=current_time,
                    metric_type='agent',
                    metric_name='task_success',
                    value=1.0 if agent_result.get('success', False) else 0.0,
                    agent_id=agent_id,
                    additional_data=agent_result
                )
                metrics.append(agent_metric)
                
                # Quality metric
                if 'result_quality' in agent_result:
                    quality_metric = MetricSnapshot(
                        timestamp=current_time,
                        metric_type='agent',
                        metric_name='result_quality',
                        value=agent_result['result_quality'],
                        agent_id=agent_id
                    )
                    metrics.append(quality_metric)
                    
            # Collect all metrics
            for metric in metrics:
                await self.metrics_collector.collect_metric(metric)
                await self.alert_manager.evaluate_metric_for_alerts(metric)
                
                # Update live streams
                metric_key = f"{metric.metric_type}_{metric.metric_name}"
                self.live_metrics[metric_key].append({
                    'timestamp': metric.timestamp.isoformat(),
                    'value': metric.value,
                    'agent_id': metric.agent_id
                })
                
            self.logger.info(f"Updated dashboard with {len(metrics)} metrics")
            
        except Exception as e:
            self.logger.error(f"Failed to update dashboard metrics: {str(e)}")
            
    async def record_coordination_session(self, session_data: Dict[str, Any]):
        """Record coordination session metrics"""
        
        try:
            session_id = session_data.get('session_id', f"session_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}")
            
            coordination_metrics = CoordinationMetrics(
                coordination_id=session_id,
                participating_agents=session_data.get('participating_agents', []),
                coordination_type=session_data.get('type', 'general'),
                start_time=datetime.fromisoformat(session_data['start_time']) if isinstance(session_data.get('start_time'), str) else session_data.get('start_time', datetime.now()),
                end_time=datetime.fromisoformat(session_data['end_time']) if session_data.get('end_time') and isinstance(session_data['end_time'], str) else session_data.get('end_time'),
                success_rate=session_data.get('success_rate', 0.8),
                message_count=session_data.get('message_count', 0),
                response_times=session_data.get('response_times', []),
                coordination_quality=session_data.get('coordination_quality', 0.8)
            )
            
            self.coordination_sessions[session_id] = coordination_metrics
            
            # Create coordination metric
            coord_metric = MetricSnapshot(
                timestamp=datetime.now(),
                metric_type='coordination',
                metric_name='session_quality',
                value=coordination_metrics.coordination_quality,
                additional_data=session_data
            )
            
            await self.metrics_collector.collect_metric(coord_metric)
            
            self.logger.info(f"Recorded coordination session {session_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to record coordination session: {str(e)}")
            
    async def emergency_alert(self, message: str):
        """Issue emergency alert"""
        
        emergency_alert = SystemAlert(
            alert_id=f"emergency_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
            alert_type='emergency',
            severity='critical',
            message=message,
            source='system',
            timestamp=datetime.now()
        )
        
        with self.alert_manager.alert_lock:
            self.alert_manager.active_alerts[emergency_alert.alert_id] = emergency_alert
            self.alert_manager.alert_history.append(emergency_alert)
            
        await self.alert_manager._send_alert_notifications(emergency_alert)
        
        self.logger.critical(f"EMERGENCY ALERT: {message}")
        
    def get_real_time_dashboard(self) -> Dict[str, Any]:
        """Get real-time dashboard data"""
        
        with self.dashboard_lock:
            # Get current alerts
            alert_summary = self.alert_manager.get_alert_summary()
            
            # Get recent performance trends
            performance_trends = {}
            for metric_name, metric_data in self.live_metrics.items():
                if len(metric_data) >= 2:
                    recent_values = [item['value'] for item in list(metric_data)[-10:]]
                    performance_trends[metric_name] = {
                        'current': recent_values[-1] if recent_values else 0,
                        'trend': self.metrics_collector._calculate_trend(recent_values),
                        'average': np.mean(recent_values),
                        'change_pct': ((recent_values[-1] - recent_values[0]) / max(recent_values[0], 0.001) * 100) if len(recent_values) >= 2 else 0
                    }
                    
            # Get coordination status
            active_coordinations = [session for session in self.coordination_sessions.values() 
                                  if session.end_time is None]
            
            coordination_summary = {
                'active_sessions': len(active_coordinations),
                'average_quality': np.mean([s.coordination_quality for s in active_coordinations]) if active_coordinations else 0,
                'total_agents_coordinating': len(set().union(*[s.participating_agents for s in active_coordinations])) if active_coordinations else 0
            }
            
            # System health indicators
            health_indicators = {
                'overall_health': self.system_status['overall_health'],
                'coordination_effectiveness': self.system_status['coordination_effectiveness'], 
                'performance_score': self.system_status['performance_score'],
                'system_load': len(self.live_metrics),
                'uptime_hours': (datetime.now() - self.system_status['last_updated']).total_seconds() / 3600
            }
            
            return {
                'timestamp': datetime.now().isoformat(),
                'system_status': self.system_status,
                'health_indicators': health_indicators,
                'performance_trends': performance_trends,
                'coordination_summary': coordination_summary,
                'alert_summary': alert_summary,
                'active_alerts': [asdict(alert) for alert in self.alert_manager.get_active_alerts()[:10]],  # Top 10
                'metrics_summary': {
                    'total_metrics_collected': len(self.metrics_collector.metrics_buffer),
                    'metrics_per_second': len([m for m in self.metrics_collector.metrics_buffer 
                                             if (datetime.now() - m.timestamp).total_seconds() < 60]) / 60,
                    'unique_metric_types': len(set(m.metric_type for m in self.metrics_collector.metrics_buffer))
                }
            }
            
    def get_agent_dashboard(self, agent_id: str) -> Dict[str, Any]:
        """Get agent-specific dashboard"""
        
        # Filter metrics for specific agent
        agent_metrics = [m for m in self.metrics_collector.metrics_buffer 
                        if m.agent_id == agent_id]
                        
        if not agent_metrics:
            return {'agent_id': agent_id, 'status': 'no_data'}
            
        # Calculate agent-specific statistics
        recent_metrics = [m for m in agent_metrics 
                         if (datetime.now() - m.timestamp).total_seconds() < 3600]  # Last hour
                         
        performance_values = [m.value for m in recent_metrics if 'performance' in m.metric_name]
        quality_values = [m.value for m in recent_metrics if 'quality' in m.metric_name]
        
        # Get agent coordination participation
        agent_coordinations = [session for session in self.coordination_sessions.values() 
                              if agent_id in session.participating_agents]
        
        return {
            'agent_id': agent_id,
            'status': 'active' if recent_metrics else 'inactive',
            'performance_summary': {
                'average_performance': np.mean(performance_values) if performance_values else 0,
                'performance_trend': self.metrics_collector._calculate_trend(performance_values) if len(performance_values) >= 2 else 'stable',
                'quality_score': np.mean(quality_values) if quality_values else 0,
                'activity_level': len(recent_metrics)
            },
            'coordination_summary': {
                'sessions_participated': len(agent_coordinations),
                'average_coordination_quality': np.mean([s.coordination_quality for s in agent_coordinations]) if agent_coordinations else 0,
                'coordination_frequency': len([s for s in agent_coordinations if s.start_time > datetime.now() - timedelta(days=1)])
            },
            'recent_metrics': [m.to_dict() for m in recent_metrics[-20:]],  # Last 20 metrics
            'alerts': [asdict(alert) for alert in self.alert_manager.get_active_alerts() 
                      if alert.additional_context.get('metric', {}).get('agent_id') == agent_id]
        }
        
    def get_executive_summary(self) -> Dict[str, Any]:
        """Get executive-level system summary"""
        
        current_time = datetime.now()
        
        # Calculate key performance indicators
        recent_metrics = [m for m in self.metrics_collector.metrics_buffer 
                         if (current_time - m.timestamp).total_seconds() < 3600]  # Last hour
        
        kpis = {
            'system_availability': 1.0 if recent_metrics else 0.0,  # Simplified calculation
            'average_performance': np.mean([m.value for m in recent_metrics if 'performance' in m.metric_type]) if recent_metrics else 0,
            'coordination_success_rate': np.mean([m.value for m in recent_metrics if 'coordination' in m.metric_type]) if recent_metrics else 0,
            'agent_utilization': len(set(m.agent_id for m in recent_metrics if m.agent_id)) / max(1, len(self.system_status['agent_status_summary'])),
            'system_efficiency': self.system_status['performance_score'] * self.system_status['coordination_effectiveness']
        }
        
        # Risk assessment
        critical_alerts = len([alert for alert in self.alert_manager.get_active_alerts() if alert.severity == 'critical'])
        high_alerts = len([alert for alert in self.alert_manager.get_active_alerts() if alert.severity == 'high'])
        
        risk_level = 'low'
        if critical_alerts > 0:
            risk_level = 'critical'
        elif high_alerts > 2:
            risk_level = 'high'
        elif high_alerts > 0:
            risk_level = 'medium'
            
        # Operational insights
        insights = []
        if kpis['average_performance'] < 0.7:
            insights.append("System performance below optimal levels")
        if kpis['coordination_success_rate'] < 0.8:
            insights.append("Coordination effectiveness requires attention")
        if kpis['agent_utilization'] < 0.6:
            insights.append("Low agent utilization detected")
        if len(insights) == 0:
            insights.append("System operating within normal parameters")
            
        return {
            'report_timestamp': current_time.isoformat(),
            'system_status': 'operational' if kpis['system_availability'] > 0.9 else 'degraded',
            'key_performance_indicators': kpis,
            'risk_assessment': {
                'current_risk_level': risk_level,
                'critical_alerts': critical_alerts,
                'high_priority_alerts': high_alerts,
                'risk_factors': [alert.message for alert in self.alert_manager.get_active_alerts()[:5]]
            },
            'operational_insights': insights,
            'recommendations': self._generate_recommendations(kpis, risk_level),
            'next_review': (current_time + timedelta(hours=1)).isoformat()
        }
        
    def _generate_recommendations(self, kpis: Dict[str, float], risk_level: str) -> List[str]:
        """Generate operational recommendations based on current state"""
        
        recommendations = []
        
        if kpis['average_performance'] < 0.7:
            recommendations.append("Consider scaling up computational resources or optimizing agent algorithms")
            
        if kpis['coordination_success_rate'] < 0.8:
            recommendations.append("Review coordination protocols and consider implementing backup coordination mechanisms")
            
        if kpis['agent_utilization'] < 0.6:
            recommendations.append("Optimize task distribution to improve agent utilization")
            
        if risk_level in ['high', 'critical']:
            recommendations.append("Immediate attention required for critical system components")
            
        if kpis['system_efficiency'] > 0.9:
            recommendations.append("System performing optimally - consider expanding capacity or capabilities")
            
        if not recommendations:
            recommendations.append("System operating normally - maintain current operational procedures")
            
        return recommendations[:5]  # Limit to top 5 recommendations