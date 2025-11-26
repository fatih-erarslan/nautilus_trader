"""
Dynamic API Selector for Low Latency Trading

Intelligently selects the best API based on:
- Current latency metrics
- API availability
- Rate limits
- Cost optimization
- Geographic proximity
"""

import time
import asyncio
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime, timedelta
import heapq
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import logging

logger = logging.getLogger(__name__)


@dataclass
class APIMetrics:
    """Real-time metrics for API performance"""
    api_name: str
    latency_samples: deque = field(default_factory=lambda: deque(maxlen=100))
    error_count: int = 0
    success_count: int = 0
    last_error_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    rate_limit_remaining: Optional[int] = None
    rate_limit_reset: Optional[datetime] = None
    cost_per_request: float = 0.0
    geographic_distance: float = 0.0  # km from user
    
    @property
    def avg_latency(self) -> float:
        """Calculate average latency in microseconds"""
        if not self.latency_samples:
            return float('inf')
        return np.mean(self.latency_samples)
    
    @property
    def p99_latency(self) -> float:
        """Calculate 99th percentile latency"""
        if not self.latency_samples:
            return float('inf')
        return np.percentile(self.latency_samples, 99)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        total = self.success_count + self.error_count
        if total == 0:
            return 0.0
        return self.success_count / total
    
    @property
    def availability_score(self) -> float:
        """Calculate availability score based on recent errors"""
        if not self.last_error_time:
            return 1.0
        
        time_since_error = datetime.now() - self.last_error_time
        # Exponential decay: errors become less relevant over time
        decay_factor = 0.5 ** (time_since_error.total_seconds() / 300)  # 5 min half-life
        return 1.0 - (0.5 * decay_factor)
    
    def update_latency(self, latency_us: float):
        """Update latency samples"""
        self.latency_samples.append(latency_us)
    
    def record_success(self, latency_us: float):
        """Record successful API call"""
        self.update_latency(latency_us)
        self.success_count += 1
        self.last_success_time = datetime.now()
    
    def record_error(self):
        """Record API error"""
        self.error_count += 1
        self.last_error_time = datetime.now()


class APISelector:
    """
    Intelligent API selector using multi-factor scoring
    """
    
    def __init__(self, 
                 apis: List[str],
                 latency_weight: float = 0.4,
                 availability_weight: float = 0.3,
                 cost_weight: float = 0.2,
                 rate_limit_weight: float = 0.1):
        """
        Initialize API selector
        
        Args:
            apis: List of available API names
            latency_weight: Weight for latency in scoring (0-1)
            availability_weight: Weight for availability (0-1)
            cost_weight: Weight for cost optimization (0-1)
            rate_limit_weight: Weight for rate limit headroom (0-1)
        """
        self.apis = apis
        self.metrics: Dict[str, APIMetrics] = {
            api: APIMetrics(api_name=api) for api in apis
        }
        
        # Scoring weights (should sum to 1.0)
        self.latency_weight = latency_weight
        self.availability_weight = availability_weight
        self.cost_weight = cost_weight
        self.rate_limit_weight = rate_limit_weight
        
        # Circuit breaker configuration
        self.circuit_breaker_threshold = 5  # errors before opening
        self.circuit_breaker_timeout = 30  # seconds before half-open
        self.circuit_states: Dict[str, Tuple[str, datetime]] = {}
        
        # Performance optimization
        self.executor = ThreadPoolExecutor(max_workers=len(apis))
        self._last_health_check = datetime.now()
        self._health_check_interval = timedelta(seconds=5)
    
    def select_api(self, 
                   operation_type: str = "order",
                   order_size: Optional[float] = None,
                   priority: str = "balanced") -> str:
        """
        Select the best API for the operation
        
        Args:
            operation_type: Type of operation (order, quote, etc.)
            order_size: Size of order for cost calculation
            priority: Selection priority (latency, cost, balanced)
        
        Returns:
            Selected API name
        """
        # Run periodic health checks
        self._check_api_health()
        
        # Get available APIs (not circuit broken)
        available_apis = self._get_available_apis()
        if not available_apis:
            raise RuntimeError("No APIs available")
        
        # Calculate scores for each API
        scores = []
        for api in available_apis:
            score = self._calculate_api_score(api, operation_type, order_size, priority)
            heapq.heappush(scores, (-score, api))  # Max heap
        
        # Select best API
        _, selected_api = heapq.heappop(scores)
        
        logger.info(f"Selected API: {selected_api} for {operation_type} "
                   f"(priority: {priority})")
        
        return selected_api
    
    def _calculate_api_score(self, 
                            api: str, 
                            operation_type: str,
                            order_size: Optional[float],
                            priority: str) -> float:
        """Calculate composite score for API selection"""
        metrics = self.metrics[api]
        
        # Adjust weights based on priority
        weights = self._adjust_weights_for_priority(priority)
        
        # Normalize metrics to 0-1 scale
        latency_score = self._normalize_latency_score(metrics.avg_latency)
        availability_score = metrics.availability_score
        cost_score = self._normalize_cost_score(metrics, order_size)
        rate_limit_score = self._normalize_rate_limit_score(metrics)
        
        # Calculate weighted score
        score = (
            weights['latency'] * latency_score +
            weights['availability'] * availability_score +
            weights['cost'] * cost_score +
            weights['rate_limit'] * rate_limit_score
        )
        
        # Apply operation-specific adjustments
        if operation_type == "order" and metrics.p99_latency < 1000:  # <1ms p99
            score *= 1.2  # Boost fast APIs for orders
        
        return score
    
    def _adjust_weights_for_priority(self, priority: str) -> Dict[str, float]:
        """Adjust scoring weights based on priority"""
        if priority == "latency":
            return {
                'latency': 0.7,
                'availability': 0.2,
                'cost': 0.05,
                'rate_limit': 0.05
            }
        elif priority == "cost":
            return {
                'latency': 0.2,
                'availability': 0.2,
                'cost': 0.5,
                'rate_limit': 0.1
            }
        else:  # balanced
            return {
                'latency': self.latency_weight,
                'availability': self.availability_weight,
                'cost': self.cost_weight,
                'rate_limit': self.rate_limit_weight
            }
    
    def _normalize_latency_score(self, latency_us: float) -> float:
        """Normalize latency to 0-1 score (lower is better)"""
        if latency_us == float('inf'):
            return 0.0
        
        # Ideal latency: <100us = 1.0, >10ms = 0.0
        if latency_us < 100:
            return 1.0
        elif latency_us > 10000:
            return 0.0
        else:
            # Logarithmic scale
            return 1.0 - (np.log10(latency_us) - 2) / 2
    
    def _normalize_cost_score(self, 
                             metrics: APIMetrics, 
                             order_size: Optional[float]) -> float:
        """Normalize cost to 0-1 score (lower is better)"""
        if not order_size or metrics.cost_per_request == 0:
            return 1.0
        
        # Calculate total cost
        total_cost = metrics.cost_per_request * order_size
        
        # Normalize: $0 = 1.0, >$10 = 0.0
        if total_cost == 0:
            return 1.0
        elif total_cost > 10:
            return 0.0
        else:
            return 1.0 - (total_cost / 10)
    
    def _normalize_rate_limit_score(self, metrics: APIMetrics) -> float:
        """Normalize rate limit headroom to 0-1 score"""
        if metrics.rate_limit_remaining is None:
            return 1.0
        
        if metrics.rate_limit_remaining == 0:
            return 0.0
        elif metrics.rate_limit_remaining > 1000:
            return 1.0
        else:
            return metrics.rate_limit_remaining / 1000
    
    def _get_available_apis(self) -> List[str]:
        """Get APIs that are not circuit broken"""
        available = []
        now = datetime.now()
        
        for api in self.apis:
            state, state_time = self.circuit_states.get(api, ("closed", now))
            
            if state == "closed":
                available.append(api)
            elif state == "open":
                # Check if timeout expired (move to half-open)
                if (now - state_time).total_seconds() > self.circuit_breaker_timeout:
                    self.circuit_states[api] = ("half-open", now)
                    available.append(api)
            elif state == "half-open":
                # Allow one request through
                available.append(api)
        
        return available
    
    def _check_api_health(self):
        """Periodic health check for all APIs"""
        now = datetime.now()
        if (now - self._last_health_check) < self._health_check_interval:
            return
        
        self._last_health_check = now
        
        # Check circuit breaker states
        for api in self.apis:
            metrics = self.metrics[api]
            
            # Open circuit if too many recent errors
            if metrics.error_count > self.circuit_breaker_threshold:
                if metrics.last_error_time and \
                   (now - metrics.last_error_time).total_seconds() < 60:
                    self.circuit_states[api] = ("open", now)
                    logger.warning(f"Circuit breaker opened for {api}")
    
    def update_metrics(self, 
                      api: str, 
                      success: bool, 
                      latency_us: Optional[float] = None,
                      rate_limit_info: Optional[Dict[str, Any]] = None):
        """
        Update API metrics after a request
        
        Args:
            api: API name
            success: Whether request succeeded
            latency_us: Request latency in microseconds
            rate_limit_info: Rate limit headers from response
        """
        if api not in self.metrics:
            return
        
        metrics = self.metrics[api]
        
        if success:
            if latency_us:
                metrics.record_success(latency_us)
            
            # Update circuit breaker state
            if self.circuit_states.get(api, ("closed", None))[0] == "half-open":
                self.circuit_states[api] = ("closed", datetime.now())
                logger.info(f"Circuit breaker closed for {api}")
        else:
            metrics.record_error()
            
            # Update circuit breaker state
            state, _ = self.circuit_states.get(api, ("closed", None))
            if state == "half-open":
                self.circuit_states[api] = ("open", datetime.now())
                logger.warning(f"Circuit breaker re-opened for {api}")
        
        # Update rate limit info
        if rate_limit_info:
            metrics.rate_limit_remaining = rate_limit_info.get('remaining')
            reset_time = rate_limit_info.get('reset')
            if reset_time:
                metrics.rate_limit_reset = datetime.fromtimestamp(reset_time)
    
    def get_metrics_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get summary of all API metrics"""
        summary = {}
        
        for api, metrics in self.metrics.items():
            circuit_state = self.circuit_states.get(api, ("closed", None))[0]
            
            summary[api] = {
                'avg_latency_us': metrics.avg_latency,
                'p99_latency_us': metrics.p99_latency,
                'success_rate': metrics.success_rate,
                'availability_score': metrics.availability_score,
                'error_count': metrics.error_count,
                'success_count': metrics.success_count,
                'circuit_state': circuit_state,
                'rate_limit_remaining': metrics.rate_limit_remaining
            }
        
        return summary
    
    async def benchmark_apis(self, test_operation: callable) -> Dict[str, float]:
        """
        Benchmark all APIs with a test operation
        
        Args:
            test_operation: Async function to test each API
        
        Returns:
            Latency results for each API
        """
        results = {}
        
        async def benchmark_single(api: str):
            try:
                start = time.perf_counter()
                await test_operation(api)
                latency_us = (time.perf_counter() - start) * 1_000_000
                
                self.update_metrics(api, True, latency_us)
                return api, latency_us
            except Exception as e:
                logger.error(f"Benchmark failed for {api}: {e}")
                self.update_metrics(api, False)
                return api, float('inf')
        
        # Run benchmarks concurrently
        tasks = [benchmark_single(api) for api in self.apis]
        benchmark_results = await asyncio.gather(*tasks)
        
        for api, latency in benchmark_results:
            results[api] = latency
        
        return results