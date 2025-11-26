"""
Rate Limiter Implementation

Implements token bucket algorithm for API rate limiting with:
- Per-endpoint rate limiting  
- Exponential backoff
- Concurrent access handling
- Metrics collection
"""

import asyncio
import time
import threading
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class RateLimitExceeded(Exception):
    """Exception raised when rate limit is exceeded."""
    
    def __init__(
        self,
        endpoint: str,
        limit: int,
        window: int,
        retry_after: float,
        backoff_multiplier: float = 1.0
    ):
        self.endpoint = endpoint
        self.limit = limit
        self.window = window
        self.retry_after = retry_after
        self.backoff_multiplier = backoff_multiplier
        self.timestamp = datetime.now()
        
        message = (
            f"Rate limit exceeded for {endpoint}. "
            f"Limit: {limit} requests per {window}s. "
            f"Retry after {retry_after:.2f}s"
        )
        super().__init__(message)


class TokenBucket:
    """
    Token bucket implementation for rate limiting.
    
    Thread-safe token bucket that refills at a constant rate.
    """
    
    def __init__(self, capacity: int, refill_rate: float):
        """
        Initialize token bucket.
        
        Args:
            capacity: Maximum number of tokens
            refill_rate: Tokens added per second
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = float(capacity)  # Start with full bucket
        self.last_refill = time.time()
        self._lock = threading.Lock()
    
    def consume(self, tokens: int = 1) -> bool:
        """
        Try to consume tokens from bucket.
        
        Args:
            tokens: Number of tokens to consume
            
        Returns:
            True if tokens were consumed, False if insufficient tokens
        """
        with self._lock:
            self._refill()
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False
    
    def _refill(self):
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill
        
        if elapsed > 0:
            tokens_to_add = elapsed * self.refill_rate
            self.tokens = min(self.capacity, self.tokens + tokens_to_add)
            self.last_refill = now
    
    def get_tokens(self) -> float:
        """Get current token count."""
        with self._lock:
            self._refill()
            return self.tokens
    
    def reset(self):
        """Reset bucket to full capacity."""
        with self._lock:
            self.tokens = float(self.capacity)
            self.last_refill = time.time()


@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""
    
    default_limit: int = 100  # Default requests per window
    default_window: int = 60  # Default window in seconds
    burst_limit: Optional[int] = None  # Maximum burst limit
    endpoint_limits: Dict[str, int] = field(default_factory=dict)  # Per-endpoint limits
    
    def get_limit(self, endpoint: str) -> int:
        """Get rate limit for specific endpoint."""
        return self.endpoint_limits.get(endpoint, self.default_limit)
    
    def get_burst_limit(self) -> int:
        """Get burst limit (defaults to 1.5x normal limit)."""
        if self.burst_limit is not None:
            return self.burst_limit
        return int(self.default_limit * 1.5)


class RateLimiter:
    """
    Rate limiter using token bucket algorithm.
    
    Provides per-endpoint rate limiting with exponential backoff.
    """
    
    def __init__(self, config: RateLimitConfig):
        """
        Initialize rate limiter.
        
        Args:
            config: Rate limiting configuration
        """
        self.config = config
        self._buckets: Dict[str, TokenBucket] = {}
        self._backoff_multipliers: Dict[str, float] = {}
        self._lock = asyncio.Lock()
        
        # Metrics
        self._metrics = {
            'total_requests': 0,
            'rate_limited_requests': 0,
            'endpoints': {}
        }
        
        logger.info(f"Initialized RateLimiter with default limit {config.default_limit}")
    
    async def acquire(self, endpoint: str, tokens: int = 1) -> None:
        """
        Acquire permission to make request.
        
        Args:
            endpoint: API endpoint
            tokens: Number of tokens to consume
            
        Raises:
            RateLimitExceeded: If rate limit exceeded
        """
        async with self._lock:
            self._metrics['total_requests'] += 1
            
            # Get or create bucket for endpoint
            if endpoint not in self._buckets:
                limit = self.config.get_limit(endpoint)
                refill_rate = limit / self.config.default_window
                self._buckets[endpoint] = TokenBucket(
                    capacity=self.config.get_burst_limit(),
                    refill_rate=refill_rate
                )
                self._backoff_multipliers[endpoint] = 1.0
                self._metrics['endpoints'][endpoint] = {
                    'requests': 0,
                    'rate_limited': 0
                }
            
            bucket = self._buckets[endpoint]
            endpoint_metrics = self._metrics['endpoints'][endpoint]
            endpoint_metrics['requests'] += 1
            
            # Try to consume tokens
            if bucket.consume(tokens):
                # Reset backoff multiplier on successful request
                self._backoff_multipliers[endpoint] = 1.0
                return
            
            # Rate limit exceeded
            self._metrics['rate_limited_requests'] += 1
            endpoint_metrics['rate_limited'] += 1
            
            # Calculate retry delay with exponential backoff
            base_delay = self.config.default_window / self.config.get_limit(endpoint)
            backoff_multiplier = self._backoff_multipliers[endpoint]
            retry_after = base_delay * backoff_multiplier
            
            # Increase backoff multiplier for next time
            self._backoff_multipliers[endpoint] = min(backoff_multiplier * 2.0, 16.0)
            
            logger.warning(
                f"Rate limit exceeded for {endpoint}. "
                f"Retry after {retry_after:.2f}s (backoff: {backoff_multiplier:.1f}x)"
            )
            
            raise RateLimitExceeded(
                endpoint=endpoint,
                limit=self.config.get_limit(endpoint),
                window=self.config.default_window,
                retry_after=retry_after,
                backoff_multiplier=backoff_multiplier
            )
    
    def get_remaining(self, endpoint: str) -> int:
        """Get remaining tokens for endpoint."""
        if endpoint not in self._buckets:
            return self.config.get_limit(endpoint)
        
        return int(self._buckets[endpoint].get_tokens())
    
    def get_retry_after(self, endpoint: str) -> float:
        """Calculate retry-after time for endpoint."""
        if endpoint not in self._buckets:
            return 0.0
        
        bucket = self._buckets[endpoint]
        tokens_needed = 1
        tokens_available = bucket.get_tokens()
        
        if tokens_available >= tokens_needed:
            return 0.0
        
        # Time to refill enough tokens
        tokens_deficit = tokens_needed - tokens_available
        refill_time = tokens_deficit / bucket.refill_rate
        return refill_time
    
    def reset(self, endpoint: Optional[str] = None):
        """
        Reset rate limits.
        
        Args:
            endpoint: Specific endpoint to reset, or None for all
        """
        if endpoint is None:
            # Reset all endpoints
            for bucket in self._buckets.values():
                bucket.reset()
            self._backoff_multipliers.clear()
            logger.info("Reset all rate limits")
        else:
            # Reset specific endpoint
            if endpoint in self._buckets:
                self._buckets[endpoint].reset()
                self._backoff_multipliers[endpoint] = 1.0
                logger.info(f"Reset rate limit for {endpoint}")
    
    def adjust_limit(self, endpoint: str, new_limit: int):
        """
        Dynamically adjust rate limit for endpoint.
        
        Args:
            endpoint: Endpoint to adjust
            new_limit: New rate limit
        """
        self.config.endpoint_limits[endpoint] = new_limit
        
        # Recreate bucket with new limit if it exists
        if endpoint in self._buckets:
            refill_rate = new_limit / self.config.default_window
            self._buckets[endpoint] = TokenBucket(
                capacity=self.config.get_burst_limit(),
                refill_rate=refill_rate
            )
        
        logger.info(f"Adjusted rate limit for {endpoint} to {new_limit}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get rate limiter metrics."""
        return {
            'total_requests': self._metrics['total_requests'],
            'rate_limited_requests': self._metrics['rate_limited_requests'],
            'rate_limit_percentage': (
                self._metrics['rate_limited_requests'] / max(self._metrics['total_requests'], 1) * 100
            ),
            'active_endpoints': len(self._buckets),
            'endpoints': dict(self._metrics['endpoints']),
            'backoff_multipliers': dict(self._backoff_multipliers)
        }
    
    def clear_metrics(self):
        """Clear collected metrics."""
        self._metrics = {
            'total_requests': 0,
            'rate_limited_requests': 0,
            'endpoints': {}
        }
        logger.info("Cleared rate limiter metrics")
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        # Cleanup if needed
        pass