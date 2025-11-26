"""
Unit tests for Failover Manager component
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
from collections import deque
import numpy as np

from src.trading_apis.orchestrator.failover_manager import (
    FailoverManager, HealthStatus, FailoverTrigger, HealthCheckResult,
    FailoverEvent, APIHealthMetrics
)
from src.trading_apis.orchestrator.api_selector import APISelector
from src.trading_apis.orchestrator.execution_router import ExecutionRouter


class MockAPI:
    """Mock API for testing"""
    
    def __init__(self, name: str, fail_rate: float = 0.0, latency_ms: float = 1.0):
        self.name = name
        self.fail_rate = fail_rate
        self.latency_ms = latency_ms
        self.call_count = 0
        self.healthy = True
    
    async def health_check(self):
        """Mock health check"""
        self.call_count += 1
        await asyncio.sleep(self.latency_ms / 1000)
        
        if not self.healthy or np.random.random() < self.fail_rate:
            raise Exception(f"API {self.name} health check failed")
        
        return {
            'status': 'ok',
            'server_time': datetime.now().isoformat()
        }
    
    async def get_server_time(self):
        """Mock server time"""
        return await self.health_check()


class TestAPIHealthMetrics:
    """Test API health metrics"""
    
    def test_initialization(self):
        """Test metrics initialization"""
        metrics = APIHealthMetrics(api="test_api")
        
        assert metrics.api == "test_api"
        assert metrics.status == HealthStatus.UNKNOWN
        assert metrics.consecutive_failures == 0
        assert metrics.consecutive_successes == 0
        assert metrics.error_count == 0
        assert metrics.uptime_percentage == 100.0
    
    def test_update_response_time(self):
        """Test response time updates"""
        metrics = APIHealthMetrics(api="test_api")
        
        # Add some response times
        response_times = [100, 200, 300, 150, 250]
        for rt in response_times:
            metrics.update_response_time(rt)
        
        assert metrics.avg_response_time_us == 200.0
        assert metrics.p95_response_time_us == 300.0
        assert metrics.p99_response_time_us == 300.0
    
    def test_response_time_window(self):
        """Test response time window management"""
        metrics = APIHealthMetrics(api="test_api")
        
        # Add more than window size
        for i in range(150):
            metrics.update_response_time(i)
        
        # Should only keep last 100
        assert len(metrics.response_times) == 100
        assert metrics.avg_response_time_us == np.mean(range(50, 150))


class TestHealthCheckResult:
    """Test health check result"""
    
    def test_healthy_result(self):
        """Test healthy result creation"""
        result = HealthCheckResult(
            api="test_api",
            status=HealthStatus.HEALTHY,
            latency_us=500.0,
            timestamp=datetime.now()
        )
        
        assert result.api == "test_api"
        assert result.status == HealthStatus.HEALTHY
        assert result.latency_us == 500.0
        assert result.error_message is None
    
    def test_unhealthy_result(self):
        """Test unhealthy result creation"""
        result = HealthCheckResult(
            api="test_api",
            status=HealthStatus.UNHEALTHY,
            latency_us=10000.0,
            timestamp=datetime.now(),
            error_message="Connection timeout"
        )
        
        assert result.status == HealthStatus.UNHEALTHY
        assert result.error_message == "Connection timeout"


class TestFailoverEvent:
    """Test failover event"""
    
    def test_failover_event_creation(self):
        """Test failover event creation"""
        event = FailoverEvent(
            api="test_api",
            trigger=FailoverTrigger.HEALTH_CHECK,
            timestamp=datetime.now(),
            from_status=HealthStatus.HEALTHY,
            to_status=HealthStatus.UNHEALTHY,
            error_details="Too many errors"
        )
        
        assert event.api == "test_api"
        assert event.trigger == FailoverTrigger.HEALTH_CHECK
        assert event.from_status == HealthStatus.HEALTHY
        assert event.to_status == HealthStatus.UNHEALTHY
        assert event.error_details == "Too many errors"


class TestFailoverManager:
    """Test FailoverManager functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.apis = {
            "primary": MockAPI("primary", latency_ms=1.0),
            "secondary": MockAPI("secondary", latency_ms=3.0),
            "backup": MockAPI("backup", latency_ms=10.0)
        }
        
        self.api_selector = APISelector(list(self.apis.keys()))
        self.execution_router = ExecutionRouter(self.api_selector, self.apis)
        self.failover_manager = FailoverManager(
            self.api_selector,
            self.execution_router,
            self.apis,
            health_check_interval=0.1  # Fast for testing
        )
    
    def test_initialization(self):
        """Test proper initialization"""
        assert self.failover_manager.api_selector == self.api_selector
        assert self.failover_manager.execution_router == self.execution_router
        assert self.failover_manager.apis == self.apis
        assert len(self.failover_manager.health_metrics) == 3
        assert not self.failover_manager.monitoring_active
    
    @pytest.mark.asyncio
    async def test_check_api_health_success(self):
        """Test successful health check"""
        api = self.apis["primary"]
        result = await self.failover_manager._check_api_health("primary", api)
        
        assert result.api == "primary"
        assert result.status == HealthStatus.HEALTHY
        assert result.latency_us > 0
        assert result.error_message is None
    
    @pytest.mark.asyncio
    async def test_check_api_health_failure(self):
        """Test failed health check"""
        api = self.apis["primary"]
        api.healthy = False
        
        result = await self.failover_manager._check_api_health("primary", api)
        
        assert result.api == "primary"
        assert result.status == HealthStatus.UNHEALTHY
        assert result.error_message is not None
    
    @pytest.mark.asyncio
    async def test_check_api_health_latency_threshold(self):
        """Test health check with high latency"""
        api = MockAPI("slow_api", latency_ms=10.0)  # 10ms latency
        result = await self.failover_manager._check_api_health("slow_api", api)
        
        # Should be degraded due to high latency
        assert result.status == HealthStatus.DEGRADED
    
    @pytest.mark.asyncio
    async def test_process_health_check_result_success(self):
        """Test processing successful health check"""
        result = HealthCheckResult(
            api="primary",
            status=HealthStatus.HEALTHY,
            latency_us=500.0,
            timestamp=datetime.now()
        )
        
        await self.failover_manager._process_health_check_result(result)
        
        metrics = self.failover_manager.health_metrics["primary"]
        assert metrics.consecutive_successes == 1
        assert metrics.consecutive_failures == 0
        assert len(metrics.response_times) == 1
    
    @pytest.mark.asyncio
    async def test_process_health_check_result_failure(self):
        """Test processing failed health check"""
        result = HealthCheckResult(
            api="primary",
            status=HealthStatus.UNHEALTHY,
            latency_us=10000.0,
            timestamp=datetime.now(),
            error_message="Connection failed"
        )
        
        await self.failover_manager._process_health_check_result(result)
        
        metrics = self.failover_manager.health_metrics["primary"]
        assert metrics.consecutive_failures == 1
        assert metrics.consecutive_successes == 0
        assert metrics.error_count == 1
    
    @pytest.mark.asyncio
    async def test_failover_trigger_after_threshold(self):
        """Test failover trigger after failure threshold"""
        # Generate enough failures to trigger failover
        for i in range(self.failover_manager.failure_threshold):
            result = HealthCheckResult(
                api="primary",
                status=HealthStatus.UNHEALTHY,
                latency_us=10000.0,
                timestamp=datetime.now()
            )
            await self.failover_manager._process_health_check_result(result)
        
        # Should trigger failover
        metrics = self.failover_manager.health_metrics["primary"]
        assert metrics.status == HealthStatus.UNHEALTHY
        assert len(self.failover_manager.failover_events) > 0
    
    @pytest.mark.asyncio
    async def test_recovery_after_success_threshold(self):
        """Test recovery after success threshold"""
        # First make API unhealthy
        metrics = self.failover_manager.health_metrics["primary"]
        metrics.status = HealthStatus.UNHEALTHY
        
        # Generate enough successes to trigger recovery
        for i in range(self.failover_manager.recovery_threshold):
            result = HealthCheckResult(
                api="primary",
                status=HealthStatus.HEALTHY,
                latency_us=500.0,
                timestamp=datetime.now()
            )
            await self.failover_manager._process_health_check_result(result)
        
        # Should recover
        assert metrics.status == HealthStatus.HEALTHY
        assert len(self.failover_manager.failover_events) > 0
    
    @pytest.mark.asyncio
    async def test_perform_health_checks(self):
        """Test health checks on all APIs"""
        await self.failover_manager._perform_health_checks()
        
        # All APIs should have been checked
        for api_name in self.apis:
            assert api_name in self.failover_manager.health_check_history
            assert len(self.failover_manager.health_check_history[api_name]) > 0
    
    @pytest.mark.asyncio
    async def test_update_failure_predictions(self):
        """Test failure prediction updates"""
        # Add some health check history
        api_name = "primary"
        
        # Add failing trend
        for i in range(20):
            result = HealthCheckResult(
                api=api_name,
                status=HealthStatus.HEALTHY if i < 10 else HealthStatus.UNHEALTHY,
                latency_us=500.0 + i * 100,  # Increasing latency
                timestamp=datetime.now()
            )
            self.failover_manager.health_check_history[api_name].append(result)
        
        await self.failover_manager._update_failure_predictions()
        
        # Should predict higher failure probability
        prediction = self.failover_manager.failure_predictions.get(api_name, 0.0)
        assert prediction > 0.0
    
    @pytest.mark.asyncio
    async def test_check_failover_conditions(self):
        """Test failover condition checking"""
        # Set high error rate
        metrics = self.failover_manager.health_metrics["primary"]
        metrics.error_rate = 0.1  # 10% error rate
        metrics.status = HealthStatus.HEALTHY
        
        await self.failover_manager._check_failover_conditions()
        
        # Should trigger failover due to error rate
        assert metrics.status == HealthStatus.DEGRADED
        assert len(self.failover_manager.failover_events) > 0
    
    @pytest.mark.asyncio
    async def test_trigger_failover(self):
        """Test manual failover trigger"""
        await self.failover_manager._trigger_failover(
            "primary",
            FailoverTrigger.MANUAL,
            HealthStatus.UNHEALTHY,
            "Test failover"
        )
        
        # Should update status and create event
        metrics = self.failover_manager.health_metrics["primary"]
        assert metrics.status == HealthStatus.UNHEALTHY
        assert len(self.failover_manager.failover_events) == 1
        
        event = self.failover_manager.failover_events[0]
        assert event.api == "primary"
        assert event.trigger == FailoverTrigger.MANUAL
        assert event.to_status == HealthStatus.UNHEALTHY
    
    @pytest.mark.asyncio
    async def test_start_stop_monitoring(self):
        """Test monitoring lifecycle"""
        # Start monitoring
        await self.failover_manager.start_monitoring()
        assert self.failover_manager.monitoring_active
        assert self.failover_manager.monitor_task is not None
        
        # Let it run briefly
        await asyncio.sleep(0.2)
        
        # Stop monitoring
        await self.failover_manager.stop_monitoring()
        assert not self.failover_manager.monitoring_active
    
    def test_get_api_health_status(self):
        """Test health status retrieval"""
        status = self.failover_manager.get_api_health_status("primary")
        assert status is not None
        assert status.api == "primary"
        
        # Non-existent API
        status = self.failover_manager.get_api_health_status("nonexistent")
        assert status is None
    
    def test_get_all_health_status(self):
        """Test all health status retrieval"""
        all_status = self.failover_manager.get_all_health_status()
        
        assert len(all_status) == 3
        assert "primary" in all_status
        assert "secondary" in all_status
        assert "backup" in all_status
    
    def test_get_healthy_apis(self):
        """Test healthy API retrieval"""
        # Initially all should be healthy (unknown = healthy)
        healthy = self.failover_manager.get_healthy_apis()
        assert len(healthy) == 3
        
        # Make one unhealthy
        self.failover_manager.health_metrics["primary"].status = HealthStatus.UNHEALTHY
        healthy = self.failover_manager.get_healthy_apis()
        assert len(healthy) == 2
        assert "primary" not in healthy
    
    def test_get_failover_history(self):
        """Test failover history retrieval"""
        # Add some events
        events = [
            FailoverEvent(
                api="primary",
                trigger=FailoverTrigger.HEALTH_CHECK,
                timestamp=datetime.now(),
                from_status=HealthStatus.HEALTHY,
                to_status=HealthStatus.UNHEALTHY
            ),
            FailoverEvent(
                api="secondary",
                trigger=FailoverTrigger.LATENCY_THRESHOLD,
                timestamp=datetime.now(),
                from_status=HealthStatus.HEALTHY,
                to_status=HealthStatus.DEGRADED
            )
        ]
        
        self.failover_manager.failover_events.extend(events)
        
        # Get all history
        all_history = self.failover_manager.get_failover_history()
        assert len(all_history) == 2
        
        # Get specific API history
        primary_history = self.failover_manager.get_failover_history("primary")
        assert len(primary_history) == 1
        assert primary_history[0].api == "primary"
    
    @pytest.mark.asyncio
    async def test_manual_failover(self):
        """Test manual failover"""
        await self.failover_manager.manual_failover("primary", "Test manual failover")
        
        metrics = self.failover_manager.health_metrics["primary"]
        assert metrics.status == HealthStatus.UNHEALTHY
        
        events = self.failover_manager.get_failover_history("primary")
        assert len(events) == 1
        assert events[0].trigger == FailoverTrigger.MANUAL
    
    @pytest.mark.asyncio
    async def test_manual_recovery(self):
        """Test manual recovery"""
        # First make it unhealthy
        self.failover_manager.health_metrics["primary"].status = HealthStatus.UNHEALTHY
        
        await self.failover_manager.manual_recovery("primary", "Test manual recovery")
        
        metrics = self.failover_manager.health_metrics["primary"]
        assert metrics.status == HealthStatus.HEALTHY
        assert metrics.consecutive_successes == self.failover_manager.recovery_threshold
    
    @pytest.mark.asyncio
    async def test_invalid_api_operations(self):
        """Test operations on invalid APIs"""
        with pytest.raises(ValueError):
            await self.failover_manager.manual_failover("nonexistent", "Test")
        
        with pytest.raises(ValueError):
            await self.failover_manager.manual_recovery("nonexistent", "Test")
    
    def test_callback_registration(self):
        """Test callback registration"""
        failover_callback = Mock()
        recovery_callback = Mock()
        
        self.failover_manager.add_failover_callback(failover_callback)
        self.failover_manager.add_recovery_callback(recovery_callback)
        
        assert failover_callback in self.failover_manager.failover_callbacks
        assert recovery_callback in self.failover_manager.recovery_callbacks
    
    def test_get_availability_report(self):
        """Test availability report generation"""
        # Add some test data
        api_name = "primary"
        
        # Add health check history
        for i in range(10):
            result = HealthCheckResult(
                api=api_name,
                status=HealthStatus.HEALTHY if i < 8 else HealthStatus.UNHEALTHY,
                latency_us=500.0,
                timestamp=datetime.now()
            )
            self.failover_manager.health_check_history[api_name].append(result)
        
        # Add failover event
        event = FailoverEvent(
            api=api_name,
            trigger=FailoverTrigger.HEALTH_CHECK,
            timestamp=datetime.now(),
            from_status=HealthStatus.HEALTHY,
            to_status=HealthStatus.UNHEALTHY,
            recovery_time_seconds=30.0
        )
        self.failover_manager.failover_events.append(event)
        
        report = self.failover_manager.get_availability_report()
        
        assert api_name in report
        api_report = report[api_name]
        assert "uptime_percentage" in api_report
        assert "avg_response_time_us" in api_report
        assert "mttr_seconds" in api_report
        assert "total_failovers" in api_report
        
        # Check calculated values
        assert api_report["uptime_percentage"] == 80.0  # 8/10 healthy
        assert api_report["mttr_seconds"] == 30.0
        assert api_report["total_failovers"] == 1
    
    def test_export_metrics(self, tmp_path):
        """Test metrics export"""
        # Add some test data
        self.failover_manager.health_metrics["primary"].error_count = 5
        self.failover_manager.health_metrics["primary"].avg_response_time_us = 1000.0
        
        # Export to file
        export_file = tmp_path / "metrics.json"
        self.failover_manager.export_metrics(str(export_file))
        
        # Verify file exists and has content
        assert export_file.exists()
        
        import json
        with open(export_file) as f:
            data = json.load(f)
        
        assert "health_metrics" in data
        assert "failover_events" in data
        assert "failure_predictions" in data
        assert "export_timestamp" in data
        
        # Check specific data
        assert "primary" in data["health_metrics"]
        assert data["health_metrics"]["primary"]["error_count"] == 5


@pytest.mark.asyncio
class TestFailoverManagerIntegration:
    """Integration tests for FailoverManager"""
    
    def setup_method(self):
        """Setup integration test environment"""
        self.apis = {
            "stable": MockAPI("stable", fail_rate=0.0, latency_ms=1.0),
            "unstable": MockAPI("unstable", fail_rate=0.3, latency_ms=2.0),
            "slow": MockAPI("slow", fail_rate=0.0, latency_ms=20.0)
        }
        
        self.api_selector = APISelector(list(self.apis.keys()))
        self.execution_router = ExecutionRouter(self.api_selector, self.apis)
        self.failover_manager = FailoverManager(
            self.api_selector,
            self.execution_router,
            self.apis,
            health_check_interval=0.05,  # 50ms for fast testing
            failure_threshold=2,
            recovery_threshold=2
        )
    
    async def test_full_failover_cycle(self):
        """Test complete failover and recovery cycle"""
        # Start monitoring
        await self.failover_manager.start_monitoring()
        
        # Let it run to establish baseline
        await asyncio.sleep(0.2)
        
        # Make an API unhealthy
        self.apis["unstable"].healthy = False
        
        # Wait for failover detection
        await asyncio.sleep(0.3)
        
        # Should have detected failure
        metrics = self.failover_manager.health_metrics["unstable"]
        assert metrics.status == HealthStatus.UNHEALTHY
        
        # Make API healthy again
        self.apis["unstable"].healthy = True
        
        # Wait for recovery detection
        await asyncio.sleep(0.3)
        
        # Should have recovered
        assert metrics.status == HealthStatus.HEALTHY
        
        # Stop monitoring
        await self.failover_manager.stop_monitoring()
        
        # Check events were recorded
        events = self.failover_manager.get_failover_history("unstable")
        assert len(events) >= 2  # At least failover and recovery
    
    async def test_callback_notifications(self):
        """Test callback notifications during failover"""
        failover_calls = []
        recovery_calls = []
        
        async def failover_callback(api, from_status, to_status):
            failover_calls.append((api, from_status, to_status))
        
        async def recovery_callback(api, from_status, to_status):
            recovery_calls.append((api, from_status, to_status))
        
        self.failover_manager.add_failover_callback(failover_callback)
        self.failover_manager.add_recovery_callback(recovery_callback)
        
        # Start monitoring
        await self.failover_manager.start_monitoring()
        await asyncio.sleep(0.1)
        
        # Trigger failover
        await self.failover_manager.manual_failover("stable", "Test failover")
        
        # Trigger recovery
        await self.failover_manager.manual_recovery("stable", "Test recovery")
        
        # Stop monitoring
        await self.failover_manager.stop_monitoring()
        
        # Check callbacks were called
        assert len(failover_calls) > 0
        assert len(recovery_calls) > 0
        
        # Check callback parameters
        assert failover_calls[0][0] == "stable"
        assert recovery_calls[0][0] == "stable"
    
    async def test_predictive_failover(self):
        """Test predictive failover based on trends"""
        # Start monitoring
        await self.failover_manager.start_monitoring()
        
        # Let it establish baseline
        await asyncio.sleep(0.1)
        
        # Gradually increase fail rate to simulate degradation
        api = self.apis["unstable"]
        for i in range(5):
            api.fail_rate = i * 0.2  # Increase from 0% to 80%
            await asyncio.sleep(0.1)
        
        # Should eventually trigger failover
        await asyncio.sleep(0.3)
        
        # Stop monitoring
        await self.failover_manager.stop_monitoring()
        
        # Check if predictive metrics were updated
        predictions = self.failover_manager.failure_predictions
        assert "unstable" in predictions
        assert predictions["unstable"] > 0.5  # High failure probability
    
    async def test_multiple_api_failures(self):
        """Test handling of multiple API failures"""
        # Start monitoring
        await self.failover_manager.start_monitoring()
        await asyncio.sleep(0.1)
        
        # Make multiple APIs fail
        self.apis["unstable"].healthy = False
        self.apis["slow"].healthy = False
        
        # Wait for detection
        await asyncio.sleep(0.3)
        
        # Check multiple failures detected
        unhealthy_apis = [
            api for api, metrics in self.failover_manager.health_metrics.items()
            if metrics.status == HealthStatus.UNHEALTHY
        ]
        
        assert len(unhealthy_apis) >= 2
        assert "unstable" in unhealthy_apis
        assert "slow" in unhealthy_apis
        
        # Stop monitoring
        await self.failover_manager.stop_monitoring()
    
    async def test_performance_monitoring(self):
        """Test performance monitoring and metrics"""
        # Start monitoring
        await self.failover_manager.start_monitoring()
        
        # Run for a while to collect metrics
        await asyncio.sleep(0.5)
        
        # Stop monitoring
        await self.failover_manager.stop_monitoring()
        
        # Check metrics were collected
        for api_name in self.apis:
            metrics = self.failover_manager.health_metrics[api_name]
            assert len(metrics.response_times) > 0
            assert metrics.avg_response_time_us > 0
        
        # Check availability report
        report = self.failover_manager.get_availability_report()
        assert len(report) == 3
        
        for api_name, api_report in report.items():
            assert api_report["uptime_percentage"] >= 0
            assert api_report["avg_response_time_us"] > 0
    
    async def test_concurrent_health_checks(self):
        """Test concurrent health check handling"""
        # Start monitoring
        await self.failover_manager.start_monitoring()
        
        # Perform manual health checks concurrently
        tasks = []
        for api_name, api_instance in self.apis.items():
            task = self.failover_manager._check_api_health(api_name, api_instance)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        # All should complete successfully
        assert len(results) == 3
        assert all(isinstance(result, HealthCheckResult) for result in results)
        
        # Stop monitoring
        await self.failover_manager.stop_monitoring()
    
    async def test_error_rate_threshold_failover(self):
        """Test failover triggered by error rate threshold"""
        # Start monitoring
        await self.failover_manager.start_monitoring()
        
        # Let it run to establish baseline
        await asyncio.sleep(0.1)
        
        # Manually set high error rate
        metrics = self.failover_manager.health_metrics["stable"]
        metrics.error_rate = 0.1  # 10% error rate (above 5% threshold)
        
        # Wait for condition check
        await asyncio.sleep(0.2)
        
        # Should trigger failover
        assert metrics.status == HealthStatus.DEGRADED
        
        # Stop monitoring
        await self.failover_manager.stop_monitoring()
        
        # Check failover event was recorded
        events = self.failover_manager.get_failover_history("stable")
        assert len(events) > 0
        assert events[0].trigger == FailoverTrigger.ERROR_THRESHOLD