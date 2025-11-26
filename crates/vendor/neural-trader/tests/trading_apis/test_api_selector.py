"""
Unit tests for API Selector component
"""

import pytest
import asyncio
from unittest.mock import Mock, patch
from datetime import datetime, timedelta
from collections import deque
import numpy as np

from src.trading_apis.orchestrator.api_selector import APISelector, APIMetrics


class TestAPIMetrics:
    """Test API metrics calculations"""
    
    def test_empty_metrics(self):
        """Test metrics with no data"""
        metrics = APIMetrics(api_name="test_api")
        
        assert metrics.avg_latency == float('inf')
        assert metrics.p99_latency == float('inf')
        assert metrics.success_rate == 0.0
        assert metrics.availability_score == 1.0
    
    def test_latency_calculations(self):
        """Test latency metric calculations"""
        metrics = APIMetrics(api_name="test_api")
        
        # Add some latency samples
        latencies = [100, 200, 300, 150, 250]
        for latency in latencies:
            metrics.update_latency(latency)
        
        assert metrics.avg_latency == 200.0
        assert metrics.p99_latency == 300.0
    
    def test_success_rate_calculation(self):
        """Test success rate calculation"""
        metrics = APIMetrics(api_name="test_api")
        
        # Record some results
        metrics.record_success(100)
        metrics.record_success(200)
        metrics.record_error()
        
        assert metrics.success_count == 2
        assert metrics.error_count == 1
        assert metrics.success_rate == 2/3
    
    def test_availability_score_decay(self):
        """Test availability score decay over time"""
        metrics = APIMetrics(api_name="test_api")
        
        # Fresh error should reduce availability
        metrics.record_error()
        assert metrics.availability_score < 1.0
        
        # No errors should give full availability
        metrics.last_error_time = None
        assert metrics.availability_score == 1.0


class TestAPISelector:
    """Test API Selector functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.apis = ["api1", "api2", "api3"]
        self.selector = APISelector(self.apis)
    
    def test_initialization(self):
        """Test proper initialization"""
        assert len(self.selector.apis) == 3
        assert len(self.selector.metrics) == 3
        assert all(api in self.selector.metrics for api in self.apis)
    
    def test_normalize_latency_score(self):
        """Test latency score normalization"""
        # Perfect latency
        assert self.selector._normalize_latency_score(50) == 1.0
        
        # Decent latency
        score_1ms = self.selector._normalize_latency_score(1000)
        assert 0.5 < score_1ms < 1.0
        
        # Poor latency
        assert self.selector._normalize_latency_score(15000) == 0.0
        
        # Infinite latency
        assert self.selector._normalize_latency_score(float('inf')) == 0.0
    
    def test_normalize_cost_score(self):
        """Test cost score normalization"""
        metrics = APIMetrics(api_name="test", cost_per_request=0.01)
        
        # Free
        assert self.selector._normalize_cost_score(metrics, None) == 1.0
        
        # Reasonable cost
        score_5 = self.selector._normalize_cost_score(metrics, 5.0)
        assert 0.5 < score_5 < 1.0
        
        # Expensive
        score_expensive = self.selector._normalize_cost_score(metrics, 1000.0)
        assert score_expensive == 0.0
    
    def test_normalize_rate_limit_score(self):
        """Test rate limit score normalization"""
        metrics = APIMetrics(api_name="test")
        
        # No limit info
        assert self.selector._normalize_rate_limit_score(metrics) == 1.0
        
        # No requests remaining
        metrics.rate_limit_remaining = 0
        assert self.selector._normalize_rate_limit_score(metrics) == 0.0
        
        # Plenty of requests
        metrics.rate_limit_remaining = 2000
        assert self.selector._normalize_rate_limit_score(metrics) == 1.0
        
        # Medium availability
        metrics.rate_limit_remaining = 500
        assert self.selector._normalize_rate_limit_score(metrics) == 0.5
    
    def test_select_api_no_data(self):
        """Test API selection with no historical data"""
        # Should select first API by default
        selected = self.selector.select_api()
        assert selected in self.apis
    
    def test_select_api_with_metrics(self):
        """Test API selection with performance metrics"""
        # Make api1 fastest
        self.selector.metrics["api1"].record_success(100)
        self.selector.metrics["api1"].record_success(120)
        
        # Make api2 slower
        self.selector.metrics["api2"].record_success(1000)
        self.selector.metrics["api2"].record_success(1200)
        
        # Make api3 unhealthy
        self.selector.metrics["api3"].record_error()
        self.selector.metrics["api3"].record_error()
        
        # Should prefer api1
        selected = self.selector.select_api(priority="latency")
        assert selected == "api1"
    
    def test_circuit_breaker(self):
        """Test circuit breaker functionality"""
        # Generate enough errors to open circuit
        for _ in range(10):
            self.selector.metrics["api1"].record_error()
        
        # Trigger health check
        self.selector._check_api_health()
        
        # Should be circuit broken
        available = self.selector._get_available_apis()
        assert "api1" not in available
    
    def test_update_metrics_success(self):
        """Test metrics update on success"""
        self.selector.update_metrics("api1", True, 500, {"remaining": 1000})
        
        metrics = self.selector.metrics["api1"]
        assert metrics.success_count == 1
        assert metrics.error_count == 0
        assert len(metrics.latency_samples) == 1
        assert metrics.rate_limit_remaining == 1000
    
    def test_update_metrics_failure(self):
        """Test metrics update on failure"""
        self.selector.update_metrics("api1", False)
        
        metrics = self.selector.metrics["api1"]
        assert metrics.success_count == 0
        assert metrics.error_count == 1
        assert metrics.last_error_time is not None
    
    def test_get_metrics_summary(self):
        """Test metrics summary generation"""
        # Add some test data
        self.selector.update_metrics("api1", True, 100)
        self.selector.update_metrics("api1", False)
        
        summary = self.selector.get_metrics_summary()
        
        assert "api1" in summary
        assert "avg_latency_us" in summary["api1"]
        assert "success_rate" in summary["api1"]
        assert "circuit_state" in summary["api1"]
    
    @pytest.mark.asyncio
    async def test_benchmark_apis(self):
        """Test API benchmarking"""
        # Mock test operation
        async def mock_test_operation(api):
            await asyncio.sleep(0.001)  # 1ms delay
            if api == "api3":
                raise Exception("API3 failed")
        
        results = await self.selector.benchmark_apis(mock_test_operation)
        
        assert len(results) == 3
        assert "api1" in results
        assert "api2" in results
        assert results["api1"] > 0  # Should have some latency
        assert results["api3"] == float('inf')  # Should be infinite for failed API
    
    def test_priority_weight_adjustment(self):
        """Test weight adjustment for different priorities"""
        # Test latency priority
        weights = self.selector._adjust_weights_for_priority("latency")
        assert weights['latency'] > weights['cost']
        
        # Test cost priority
        weights = self.selector._adjust_weights_for_priority("cost")
        assert weights['cost'] > weights['latency']
        
        # Test balanced priority
        weights = self.selector._adjust_weights_for_priority("balanced")
        assert weights['latency'] == self.selector.latency_weight
    
    def test_api_score_calculation(self):
        """Test API score calculation"""
        # Setup some metrics
        self.selector.metrics["api1"].record_success(100)
        self.selector.metrics["api1"].record_success(120)
        
        score = self.selector._calculate_api_score("api1", "order", 100.0, "balanced")
        
        assert 0.0 <= score <= 1.2  # Score should be reasonable
        assert isinstance(score, float)


class TestAPIMetricsIntegration:
    """Integration tests for API metrics"""
    
    def test_metrics_over_time(self):
        """Test metrics behavior over time"""
        metrics = APIMetrics(api_name="test_api")
        
        # Simulate API calls over time
        for i in range(100):
            if i % 10 == 0:  # 10% error rate
                metrics.record_error()
            else:
                # Gradually increasing latency
                latency = 100 + i * 2
                metrics.record_success(latency)
        
        # Check final metrics
        assert metrics.success_rate == 0.9
        assert metrics.avg_latency > 100  # Should have increased
        assert metrics.error_count == 10
    
    def test_availability_over_time(self):
        """Test availability score changes over time"""
        metrics = APIMetrics(api_name="test_api")
        
        # Start with error
        metrics.record_error()
        initial_score = metrics.availability_score
        
        # Simulate time passing (mock last_error_time)
        metrics.last_error_time = datetime.now() - timedelta(minutes=10)
        later_score = metrics.availability_score
        
        # Should have recovered somewhat
        assert later_score > initial_score


@pytest.mark.asyncio
class TestAPISelectionScenarios:
    """Test various API selection scenarios"""
    
    def setup_method(self):
        """Setup test scenario"""
        self.apis = ["fast_api", "cheap_api", "reliable_api"]
        self.selector = APISelector(self.apis)
        
        # Setup different API characteristics
        # Fast API: Low latency, high cost
        for _ in range(10):
            self.selector.update_metrics("fast_api", True, 50)
        self.selector.metrics["fast_api"].cost_per_request = 0.05
        
        # Cheap API: High latency, low cost
        for _ in range(10):
            self.selector.update_metrics("cheap_api", True, 2000)
        self.selector.metrics["cheap_api"].cost_per_request = 0.001
        
        # Reliable API: Medium latency, medium cost, high availability
        for _ in range(10):
            self.selector.update_metrics("reliable_api", True, 500)
        self.selector.metrics["reliable_api"].cost_per_request = 0.01
    
    def test_latency_priority_selection(self):
        """Test selection with latency priority"""
        selected = self.selector.select_api(priority="latency")
        assert selected == "fast_api"
    
    def test_cost_priority_selection(self):
        """Test selection with cost priority"""
        selected = self.selector.select_api(priority="cost", order_size=1000)
        assert selected == "cheap_api"
    
    def test_balanced_selection(self):
        """Test balanced selection"""
        selected = self.selector.select_api(priority="balanced")
        # Should likely be reliable_api as it's balanced
        assert selected in self.apis
    
    def test_failover_scenario(self):
        """Test selection during failover"""
        # Make fast_api fail
        for _ in range(10):
            self.selector.update_metrics("fast_api", False)
        
        # Force circuit breaker
        self.selector._check_api_health()
        
        # Should select another API
        selected = self.selector.select_api(priority="latency")
        assert selected != "fast_api"
    
    def test_rate_limit_scenario(self):
        """Test selection with rate limits"""
        # Set rate limits
        self.selector.metrics["fast_api"].rate_limit_remaining = 0
        self.selector.metrics["cheap_api"].rate_limit_remaining = 1000
        self.selector.metrics["reliable_api"].rate_limit_remaining = 100
        
        # Should avoid rate limited API
        selected = self.selector.select_api()
        assert selected != "fast_api"
    
    async def test_concurrent_selection(self):
        """Test concurrent API selection"""
        # Run multiple selections concurrently
        tasks = [
            self.selector.select_api(priority="latency"),
            self.selector.select_api(priority="cost"),
            self.selector.select_api(priority="balanced")
        ]
        
        results = await asyncio.gather(*tasks)
        
        # All should complete successfully
        assert len(results) == 3
        assert all(result in self.apis for result in results)