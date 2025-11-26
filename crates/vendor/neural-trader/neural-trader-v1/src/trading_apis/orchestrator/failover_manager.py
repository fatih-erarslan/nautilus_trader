"""
High Availability Failover Manager for Trading APIs

Features:
- Automatic failover and recovery
- Health monitoring with predictive analysis
- Graceful degradation
- Real-time status tracking
- Load balancing during failover
- Circuit breaker pattern
"""

import asyncio
import time
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from collections import deque, defaultdict
import numpy as np

from .api_selector import APISelector
from .execution_router import ExecutionRouter

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class FailoverTrigger(Enum):
    MANUAL = "manual"
    HEALTH_CHECK = "health_check"
    ERROR_THRESHOLD = "error_threshold"
    LATENCY_THRESHOLD = "latency_threshold"
    TIMEOUT = "timeout"
    CIRCUIT_BREAKER = "circuit_breaker"


@dataclass
class HealthCheckResult:
    """Result of an API health check"""
    api: str
    status: HealthStatus
    latency_us: float
    timestamp: datetime
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FailoverEvent:
    """Failover event record"""
    api: str
    trigger: FailoverTrigger
    timestamp: datetime
    from_status: HealthStatus
    to_status: HealthStatus
    recovery_time_seconds: Optional[float] = None
    error_details: Optional[str] = None


@dataclass
class APIHealthMetrics:
    """Comprehensive health metrics for an API"""
    api: str
    status: HealthStatus = HealthStatus.UNKNOWN
    last_check: Optional[datetime] = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    
    # Response time metrics
    response_times: deque = field(default_factory=lambda: deque(maxlen=100))
    avg_response_time_us: float = 0.0
    p95_response_time_us: float = 0.0
    p99_response_time_us: float = 0.0
    
    # Error tracking
    error_count: int = 0
    error_rate: float = 0.0
    last_error: Optional[datetime] = None
    
    # Availability metrics
    uptime_percentage: float = 100.0
    last_healthy_time: Optional[datetime] = None
    downtime_duration: timedelta = timedelta(0)
    
    # Throughput metrics
    requests_per_second: float = 0.0
    max_concurrent_requests: int = 0
    
    def update_response_time(self, latency_us: float):
        """Update response time metrics"""
        self.response_times.append(latency_us)
        if self.response_times:
            self.avg_response_time_us = np.mean(self.response_times)
            self.p95_response_time_us = np.percentile(self.response_times, 95)
            self.p99_response_time_us = np.percentile(self.response_times, 99)


class FailoverManager:
    """
    High availability failover manager for trading APIs
    """
    
    def __init__(self,
                 api_selector: APISelector,
                 execution_router: ExecutionRouter,
                 apis: Dict[str, Any],
                 health_check_interval: float = 5.0,
                 failure_threshold: int = 3,
                 recovery_threshold: int = 2,
                 latency_threshold_us: float = 5000,
                 error_rate_threshold: float = 0.05):
        """
        Initialize failover manager
        
        Args:
            api_selector: API selector instance
            execution_router: Execution router instance
            apis: Dictionary of API instances
            health_check_interval: Health check interval in seconds
            failure_threshold: Consecutive failures before marking unhealthy
            recovery_threshold: Consecutive successes before marking healthy
            latency_threshold_us: Latency threshold in microseconds
            error_rate_threshold: Error rate threshold (0-1)
        """
        self.api_selector = api_selector
        self.execution_router = execution_router
        self.apis = apis
        self.health_check_interval = health_check_interval
        self.failure_threshold = failure_threshold
        self.recovery_threshold = recovery_threshold
        self.latency_threshold_us = latency_threshold_us
        self.error_rate_threshold = error_rate_threshold
        
        # Health monitoring
        self.health_metrics: Dict[str, APIHealthMetrics] = {
            api: APIHealthMetrics(api=api) for api in apis.keys()
        }
        
        # Event tracking
        self.failover_events: List[FailoverEvent] = []
        self.health_check_history: Dict[str, List[HealthCheckResult]] = defaultdict(list)
        
        # Control flags
        self.monitoring_active = False
        self.monitor_task: Optional[asyncio.Task] = None
        
        # Failover callbacks
        self.failover_callbacks: List[Callable] = []
        self.recovery_callbacks: List[Callable] = []
        
        # Performance optimization
        self.executor = ThreadPoolExecutor(max_workers=len(apis))
        
        # Predictive analysis
        self.failure_predictions: Dict[str, float] = {}  # API -> failure probability
        self.prediction_window = 300  # 5 minutes
        
    async def start_monitoring(self):
        """Start continuous health monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_task = asyncio.create_task(self._monitor_loop())
        
        logger.info("Failover monitoring started")
    
    async def stop_monitoring(self):
        """Stop health monitoring"""
        self.monitoring_active = False
        
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Failover monitoring stopped")
    
    async def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Perform health checks
                await self._perform_health_checks()
                
                # Update predictions
                await self._update_failure_predictions()
                
                # Check for failover conditions
                await self._check_failover_conditions()
                
                # Wait for next check
                await asyncio.sleep(self.health_check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(1)  # Brief pause before retry
    
    async def _perform_health_checks(self):
        """Perform health checks on all APIs"""
        tasks = []
        
        for api_name, api_instance in self.apis.items():
            task = self._check_api_health(api_name, api_instance)
            tasks.append(task)
        
        # Execute all health checks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for result in results:
            if isinstance(result, HealthCheckResult):
                await self._process_health_check_result(result)
            elif isinstance(result, Exception):
                logger.error(f"Health check failed: {result}")
    
    async def _check_api_health(self, api_name: str, api_instance: Any) -> HealthCheckResult:
        """Check health of a single API"""
        start_time = time.perf_counter()
        
        try:
            # Perform lightweight health check
            if hasattr(api_instance, 'health_check'):
                health_data = await api_instance.health_check()
            elif hasattr(api_instance, 'get_server_time'):
                health_data = await api_instance.get_server_time()
            else:
                # Fallback to basic connectivity test
                health_data = await api_instance.get_account_info()
            
            latency_us = (time.perf_counter() - start_time) * 1_000_000
            
            # Determine health status
            if latency_us > self.latency_threshold_us:
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.HEALTHY
            
            return HealthCheckResult(
                api=api_name,
                status=status,
                latency_us=latency_us,
                timestamp=datetime.now(),
                metrics={
                    'server_time': health_data.get('server_time'),
                    'status': health_data.get('status', 'ok')
                }
            )
            
        except Exception as e:
            latency_us = (time.perf_counter() - start_time) * 1_000_000
            
            return HealthCheckResult(
                api=api_name,
                status=HealthStatus.UNHEALTHY,
                latency_us=latency_us,
                timestamp=datetime.now(),
                error_message=str(e)
            )
    
    async def _process_health_check_result(self, result: HealthCheckResult):
        """Process a health check result and update metrics"""
        api_name = result.api
        metrics = self.health_metrics[api_name]
        
        # Update response time metrics
        metrics.update_response_time(result.latency_us)
        metrics.last_check = result.timestamp
        
        # Store check history
        self.health_check_history[api_name].append(result)
        
        # Keep only recent history
        if len(self.health_check_history[api_name]) > 1000:
            self.health_check_history[api_name] = self.health_check_history[api_name][-1000:]
        
        # Update health status
        old_status = metrics.status
        
        if result.status == HealthStatus.HEALTHY:
            metrics.consecutive_successes += 1
            metrics.consecutive_failures = 0
            
            # Check if API has recovered
            if old_status != HealthStatus.HEALTHY and metrics.consecutive_successes >= self.recovery_threshold:
                metrics.status = HealthStatus.HEALTHY
                metrics.last_healthy_time = result.timestamp
                
                # Calculate recovery time
                recovery_time = None
                if metrics.last_error:
                    recovery_time = (result.timestamp - metrics.last_error).total_seconds()
                
                # Log recovery event
                event = FailoverEvent(
                    api=api_name,
                    trigger=FailoverTrigger.HEALTH_CHECK,
                    timestamp=result.timestamp,
                    from_status=old_status,
                    to_status=HealthStatus.HEALTHY,
                    recovery_time_seconds=recovery_time
                )
                self.failover_events.append(event)
                
                # Notify callbacks
                for callback in self.recovery_callbacks:
                    try:
                        await callback(api_name, old_status, HealthStatus.HEALTHY)
                    except Exception as e:
                        logger.error(f"Recovery callback failed: {e}")
                
                logger.info(f"API {api_name} recovered (recovery time: {recovery_time:.2f}s)")
        
        else:
            metrics.consecutive_failures += 1
            metrics.consecutive_successes = 0
            
            if result.status == HealthStatus.UNHEALTHY:
                metrics.error_count += 1
                metrics.last_error = result.timestamp
            
            # Check if API should be marked unhealthy
            if metrics.consecutive_failures >= self.failure_threshold:
                if old_status == HealthStatus.HEALTHY:
                    metrics.status = HealthStatus.UNHEALTHY
                    
                    # Log failover event
                    event = FailoverEvent(
                        api=api_name,
                        trigger=FailoverTrigger.HEALTH_CHECK,
                        timestamp=result.timestamp,
                        from_status=old_status,
                        to_status=HealthStatus.UNHEALTHY,
                        error_details=result.error_message
                    )
                    self.failover_events.append(event)
                    
                    # Notify callbacks
                    for callback in self.failover_callbacks:
                        try:
                            await callback(api_name, old_status, HealthStatus.UNHEALTHY)
                        except Exception as e:
                            logger.error(f"Failover callback failed: {e}")
                    
                    logger.warning(f"API {api_name} marked unhealthy after {metrics.consecutive_failures} failures")
        
        # Update error rate
        recent_checks = self.health_check_history[api_name][-50:]  # Last 50 checks
        if recent_checks:
            error_count = sum(1 for check in recent_checks if check.status == HealthStatus.UNHEALTHY)
            metrics.error_rate = error_count / len(recent_checks)
    
    async def _update_failure_predictions(self):
        """Update failure predictions using historical data"""
        for api_name, metrics in self.health_metrics.items():
            recent_checks = self.health_check_history[api_name][-60:]  # Last 60 checks
            
            if len(recent_checks) < 10:
                continue
            
            # Calculate trend in response times
            response_times = [check.latency_us for check in recent_checks]
            if len(response_times) >= 2:
                trend = np.polyfit(range(len(response_times)), response_times, 1)[0]
                
                # Predict failure probability
                failure_prob = 0.0
                
                # Factor 1: Response time trend
                if trend > 100:  # Increasing latency
                    failure_prob += min(trend / 1000, 0.3)  # Max 30% from trend
                
                # Factor 2: Error rate
                failure_prob += metrics.error_rate * 0.5  # Max 50% from error rate
                
                # Factor 3: Consecutive failures
                failure_prob += (metrics.consecutive_failures / self.failure_threshold) * 0.2
                
                # Factor 4: Time since last healthy
                if metrics.last_healthy_time:
                    time_since_healthy = (datetime.now() - metrics.last_healthy_time).total_seconds()
                    if time_since_healthy > 60:  # More than 1 minute
                        failure_prob += min(time_since_healthy / 600, 0.3)  # Max 30% from time
                
                self.failure_predictions[api_name] = min(failure_prob, 1.0)
    
    async def _check_failover_conditions(self):
        """Check for conditions that require failover"""
        for api_name, metrics in self.health_metrics.items():
            # Check error rate threshold
            if metrics.error_rate > self.error_rate_threshold:
                if metrics.status == HealthStatus.HEALTHY:
                    await self._trigger_failover(
                        api_name,
                        FailoverTrigger.ERROR_THRESHOLD,
                        HealthStatus.DEGRADED,
                        f"Error rate {metrics.error_rate:.2%} exceeds threshold"
                    )
            
            # Check latency threshold
            if metrics.p95_response_time_us > self.latency_threshold_us:
                if metrics.status == HealthStatus.HEALTHY:
                    await self._trigger_failover(
                        api_name,
                        FailoverTrigger.LATENCY_THRESHOLD,
                        HealthStatus.DEGRADED,
                        f"P95 latency {metrics.p95_response_time_us:.0f}μs exceeds threshold"
                    )
            
            # Check prediction-based early warning
            prediction = self.failure_predictions.get(api_name, 0.0)
            if prediction > 0.8 and metrics.status == HealthStatus.HEALTHY:
                await self._trigger_failover(
                    api_name,
                    FailoverTrigger.HEALTH_CHECK,
                    HealthStatus.DEGRADED,
                    f"High failure probability: {prediction:.2%}"
                )
    
    async def _trigger_failover(self,
                               api_name: str,
                               trigger: FailoverTrigger,
                               new_status: HealthStatus,
                               details: str):
        """Trigger failover for an API"""
        old_status = self.health_metrics[api_name].status
        self.health_metrics[api_name].status = new_status
        
        # Log failover event
        event = FailoverEvent(
            api=api_name,
            trigger=trigger,
            timestamp=datetime.now(),
            from_status=old_status,
            to_status=new_status,
            error_details=details
        )
        self.failover_events.append(event)
        
        # Notify callbacks
        for callback in self.failover_callbacks:
            try:
                await callback(api_name, old_status, new_status)
            except Exception as e:
                logger.error(f"Failover callback failed: {e}")
        
        logger.warning(f"Failover triggered for {api_name}: {details}")
    
    def get_api_health_status(self, api_name: str) -> Optional[APIHealthMetrics]:
        """Get health status for a specific API"""
        return self.health_metrics.get(api_name)
    
    def get_all_health_status(self) -> Dict[str, APIHealthMetrics]:
        """Get health status for all APIs"""
        return self.health_metrics.copy()
    
    def get_healthy_apis(self) -> List[str]:
        """Get list of healthy APIs"""
        return [
            api for api, metrics in self.health_metrics.items()
            if metrics.status == HealthStatus.HEALTHY
        ]
    
    def get_failover_history(self, api_name: Optional[str] = None) -> List[FailoverEvent]:
        """Get failover event history"""
        if api_name:
            return [event for event in self.failover_events if event.api == api_name]
        return self.failover_events.copy()
    
    def add_failover_callback(self, callback: Callable):
        """Add callback for failover events"""
        self.failover_callbacks.append(callback)
    
    def add_recovery_callback(self, callback: Callable):
        """Add callback for recovery events"""
        self.recovery_callbacks.append(callback)
    
    async def manual_failover(self, api_name: str, reason: str = "Manual failover"):
        """Manually trigger failover for an API"""
        if api_name not in self.apis:
            raise ValueError(f"API {api_name} not found")
        
        await self._trigger_failover(
            api_name,
            FailoverTrigger.MANUAL,
            HealthStatus.UNHEALTHY,
            reason
        )
    
    async def manual_recovery(self, api_name: str, reason: str = "Manual recovery"):
        """Manually trigger recovery for an API"""
        if api_name not in self.apis:
            raise ValueError(f"API {api_name} not found")
        
        old_status = self.health_metrics[api_name].status
        self.health_metrics[api_name].status = HealthStatus.HEALTHY
        self.health_metrics[api_name].consecutive_failures = 0
        self.health_metrics[api_name].consecutive_successes = self.recovery_threshold
        
        # Log recovery event
        event = FailoverEvent(
            api=api_name,
            trigger=FailoverTrigger.MANUAL,
            timestamp=datetime.now(),
            from_status=old_status,
            to_status=HealthStatus.HEALTHY,
            error_details=reason
        )
        self.failover_events.append(event)
        
        # Notify callbacks
        for callback in self.recovery_callbacks:
            try:
                await callback(api_name, old_status, HealthStatus.HEALTHY)
            except Exception as e:
                logger.error(f"Recovery callback failed: {e}")
        
        logger.info(f"Manual recovery triggered for {api_name}: {reason}")
    
    def get_availability_report(self) -> Dict[str, Dict[str, Any]]:
        """Generate availability report for all APIs"""
        report = {}
        
        for api_name, metrics in self.health_metrics.items():
            # Calculate uptime from check history
            checks = self.health_check_history[api_name]
            if not checks:
                continue
            
            healthy_checks = sum(1 for check in checks if check.status == HealthStatus.HEALTHY)
            uptime_pct = (healthy_checks / len(checks)) * 100
            
            # Calculate MTTR (Mean Time To Recovery)
            recovery_times = [
                event.recovery_time_seconds 
                for event in self.failover_events 
                if event.api == api_name and event.recovery_time_seconds
            ]
            mttr = np.mean(recovery_times) if recovery_times else 0
            
            # Calculate MTBF (Mean Time Between Failures)
            failure_events = [
                event for event in self.failover_events 
                if event.api == api_name and event.to_status == HealthStatus.UNHEALTHY
            ]
            
            if len(failure_events) > 1:
                failure_intervals = []
                for i in range(1, len(failure_events)):
                    interval = (failure_events[i].timestamp - failure_events[i-1].timestamp).total_seconds()
                    failure_intervals.append(interval)
                mtbf = np.mean(failure_intervals)
            else:
                mtbf = 0
            
            report[api_name] = {
                'current_status': metrics.status.value,
                'uptime_percentage': uptime_pct,
                'avg_response_time_us': metrics.avg_response_time_us,
                'p95_response_time_us': metrics.p95_response_time_us,
                'p99_response_time_us': metrics.p99_response_time_us,
                'error_rate': metrics.error_rate,
                'failure_prediction': self.failure_predictions.get(api_name, 0.0),
                'mttr_seconds': mttr,
                'mtbf_seconds': mtbf,
                'total_failovers': len([e for e in self.failover_events if e.api == api_name]),
                'last_healthy': metrics.last_healthy_time.isoformat() if metrics.last_healthy_time else None
            }
        
        return report
    
    def export_metrics(self, filepath: str):
        """Export metrics to JSON file"""
        data = {
            'health_metrics': {
                api: {
                    'status': metrics.status.value,
                    'last_check': metrics.last_check.isoformat() if metrics.last_check else None,
                    'consecutive_failures': metrics.consecutive_failures,
                    'consecutive_successes': metrics.consecutive_successes,
                    'avg_response_time_us': metrics.avg_response_time_us,
                    'p95_response_time_us': metrics.p95_response_time_us,
                    'p99_response_time_us': metrics.p99_response_time_us,
                    'error_count': metrics.error_count,
                    'error_rate': metrics.error_rate,
                    'uptime_percentage': metrics.uptime_percentage
                }
                for api, metrics in self.health_metrics.items()
            },
            'failover_events': [
                {
                    'api': event.api,
                    'trigger': event.trigger.value,
                    'timestamp': event.timestamp.isoformat(),
                    'from_status': event.from_status.value,
                    'to_status': event.to_status.value,
                    'recovery_time_seconds': event.recovery_time_seconds,
                    'error_details': event.error_details
                }
                for event in self.failover_events
            ],
            'failure_predictions': self.failure_predictions,
            'export_timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Metrics exported to {filepath}")