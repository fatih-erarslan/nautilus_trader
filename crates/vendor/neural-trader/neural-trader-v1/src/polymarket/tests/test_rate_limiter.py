"""
Test suite for the RateLimiter implementation.

Tests the token bucket rate limiting algorithm with:
- Token bucket mechanics
- Per-endpoint rate limiting
- Exponential backoff
- Concurrent access handling
"""

import pytest
import asyncio
import time
from unittest.mock import patch, AsyncMock, MagicMock
from datetime import datetime, timedelta

from src.polymarket.api.rate_limiter import (
    RateLimiter,
    RateLimitConfig,
    RateLimitExceeded,
    TokenBucket,
)


class TestTokenBucket:
    """Test cases for the TokenBucket implementation."""
    
    @pytest.fixture
    def bucket(self):
        """Create a test token bucket."""
        return TokenBucket(capacity=10, refill_rate=5.0)  # 5 tokens per second
    
    def test_bucket_initialization(self):
        """Test token bucket initialization."""
        bucket = TokenBucket(capacity=100, refill_rate=10.0)
        assert bucket.capacity == 100
        assert bucket.refill_rate == 10.0
        assert bucket.tokens == 100  # Should start full
        assert bucket.last_refill is not None
    
    def test_consume_tokens_success(self, bucket):
        """Test successful token consumption."""
        # Should have 10 tokens initially
        assert bucket.consume(5) is True
        assert bucket.tokens <= 5  # Allow for small refill amount
        
        # Should be able to consume remaining tokens
        assert bucket.consume(5) is True
        assert bucket.tokens < 0.1  # Should be close to 0
    
    def test_consume_tokens_insufficient(self, bucket):
        """Test token consumption when insufficient tokens available."""
        # Consume all tokens
        bucket.consume(10)
        assert bucket.tokens < 0.1  # Should be close to 0
        
        # Should fail to consume more
        assert bucket.consume(1) is False
        assert bucket.tokens < 0.1  # Should still be close to 0
    
    def test_token_refill(self, bucket):
        """Test token bucket refill over time."""
        # Consume all tokens
        bucket.consume(10)
        assert bucket.tokens == 0
        
        # Mock time advancement (1 second = 5 tokens)
        with patch('time.time') as mock_time:
            # Advance time by 1 second
            mock_time.return_value = bucket.last_refill + 1.0
            
            # Should refill 5 tokens
            bucket._refill()
            assert bucket.tokens == 5
            
            # Advance another second
            mock_time.return_value = bucket.last_refill + 2.0
            bucket._refill()
            assert bucket.tokens == 10  # Should cap at capacity
    
    def test_partial_refill(self, bucket):
        """Test partial token refill with fractional time."""
        bucket.consume(10)  # Empty bucket
        
        with patch('time.time') as mock_time:
            # Advance time by 0.5 seconds (should add 2.5 tokens)
            mock_time.return_value = bucket.last_refill + 0.5
            bucket._refill()
            assert bucket.tokens == 2.5
    
    def test_concurrent_access(self, bucket):
        """Test thread safety of token bucket."""
        import threading
        import concurrent.futures
        
        def consume_token():
            return bucket.consume(1)
        
        # Try to consume 15 tokens with 15 threads (only 10 should succeed)
        with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
            futures = [executor.submit(consume_token) for _ in range(15)]
            results = [future.result() for future in futures]
        
        # Exactly 10 should succeed
        successes = sum(1 for r in results if r)
        assert successes == 10
        assert bucket.tokens < 0.1  # Should be close to 0 with small refill


class TestRateLimitConfig:
    """Test cases for RateLimitConfig."""
    
    def test_config_initialization(self):
        """Test rate limit configuration initialization."""
        config = RateLimitConfig(
            default_limit=100,
            default_window=60,
            burst_limit=150
        )
        assert config.default_limit == 100
        assert config.default_window == 60
        assert config.burst_limit == 150
        assert len(config.endpoint_limits) == 0
    
    def test_endpoint_specific_limits(self):
        """Test endpoint-specific rate limits."""
        config = RateLimitConfig(
            default_limit=100,
            endpoint_limits={
                '/orders': 10,
                '/markets': 50,
                '/trades': 25
            }
        )
        
        assert config.get_limit('/orders') == 10
        assert config.get_limit('/markets') == 50
        assert config.get_limit('/trades') == 25
        assert config.get_limit('/unknown') == 100  # Default


class TestRateLimiter:
    """Test cases for the RateLimiter implementation."""
    
    @pytest.fixture
    def rate_limiter(self):
        """Create a test rate limiter."""
        config = RateLimitConfig(
            default_limit=10,
            default_window=60,
            endpoint_limits={
                '/orders': 5,
                '/high-frequency': 2
            }
        )
        return RateLimiter(config)
    
    @pytest.mark.asyncio
    async def test_acquire_within_limit(self, rate_limiter):
        """Test acquiring permits within rate limit."""
        # Should succeed for requests within limit
        for i in range(5):
            await rate_limiter.acquire('/orders')
        
        # Check that bucket has correct remaining tokens
        bucket = rate_limiter._buckets['/orders']
        # Bucket starts with burst_limit (15) tokens, we consumed 5, should have ~10 left
        assert 9 <= bucket.tokens <= 11  # Allow for small refill during test
    
    @pytest.mark.asyncio
    async def test_acquire_exceeds_limit(self, rate_limiter):
        """Test rate limit exceeded scenario."""
        # Consume all tokens for /orders endpoint (burst_limit = 15)
        for i in range(15):
            await rate_limiter.acquire('/orders')
        
        # Next request should raise exception
        with pytest.raises(RateLimitExceeded) as exc_info:
            await rate_limiter.acquire('/orders')
        
        assert exc_info.value.endpoint == '/orders'
        assert exc_info.value.retry_after > 0
    
    @pytest.mark.asyncio
    async def test_different_endpoints_independent(self, rate_limiter):
        """Test that different endpoints have independent rate limits."""
        # Consume all tokens for /orders (15 tokens)
        for i in range(15):
            await rate_limiter.acquire('/orders')
        
        # Should still be able to access other endpoints
        await rate_limiter.acquire('/markets')  # Uses default limit
        await rate_limiter.acquire('/high-frequency')
    
    @pytest.mark.asyncio
    async def test_exponential_backoff(self, rate_limiter):
        """Test exponential backoff mechanism."""
        endpoint = '/test-backoff'
        
        # Fill up the bucket first to trigger rate limiting
        for i in range(15):  # Consume all burst_limit tokens
            await rate_limiter.acquire(endpoint)
        
        # First rate limited request
        with pytest.raises(RateLimitExceeded) as exc_info:
            await rate_limiter.acquire(endpoint)
        first_retry_after = exc_info.value.retry_after
        
        # Second rate limited request should have higher backoff
        with pytest.raises(RateLimitExceeded) as exc_info:
            await rate_limiter.acquire(endpoint)
        second_retry_after = exc_info.value.retry_after
        
        # Exponential backoff should increase delay
        assert second_retry_after > first_retry_after
    
    @pytest.mark.asyncio
    async def test_burst_handling(self):
        """Test burst limit handling."""
        config = RateLimitConfig(
            default_limit=10,
            burst_limit=15,
            default_window=60
        )
        rate_limiter = RateLimiter(config)
        
        # Should handle burst up to burst_limit
        for i in range(15):
            await rate_limiter.acquire('/test')
        
        # 16th request should fail
        with pytest.raises(RateLimitExceeded):
            await rate_limiter.acquire('/test')
    
    @pytest.mark.asyncio
    async def test_auto_reset_after_window(self, rate_limiter):
        """Test automatic reset after time window."""
        endpoint = '/test-reset'
        
        # Consume all tokens
        for i in range(10):  # Default limit
            await rate_limiter.acquire(endpoint)
        
        # Should be rate limited
        with pytest.raises(RateLimitExceeded):
            await rate_limiter.acquire(endpoint)
        
        # Mock time advancement beyond window
        with patch('time.time') as mock_time:
            bucket = rate_limiter._buckets[endpoint]
            mock_time.return_value = bucket.last_refill + 61  # Beyond 60s window
            
            # Should work again after refill
            await rate_limiter.acquire(endpoint)
    
    def test_get_remaining_tokens(self, rate_limiter):
        """Test getting remaining token count."""
        endpoint = '/test-remaining'
        
        # Initially should have full capacity
        remaining = rate_limiter.get_remaining(endpoint)
        assert remaining == 10  # Default limit
        
        # After consuming some tokens
        asyncio.run(rate_limiter.acquire(endpoint))
        remaining = rate_limiter.get_remaining(endpoint)
        assert remaining == 9
    
    def test_get_retry_after(self, rate_limiter):
        """Test calculating retry-after time."""
        endpoint = '/test-retry'
        
        # Consume all tokens
        for i in range(10):
            asyncio.run(rate_limiter.acquire(endpoint))
        
        retry_after = rate_limiter.get_retry_after(endpoint)
        assert retry_after > 0
        assert retry_after <= 60  # Should be within window
    
    @pytest.mark.asyncio
    async def test_reset_endpoint_limits(self, rate_limiter):
        """Test manually resetting endpoint limits."""
        endpoint = '/test-reset'
        
        # Consume tokens
        for i in range(5):
            await rate_limiter.acquire(endpoint)
        
        remaining_before = rate_limiter.get_remaining(endpoint)
        assert remaining_before < 10
        
        # Reset
        rate_limiter.reset(endpoint)
        remaining_after = rate_limiter.get_remaining(endpoint)
        assert remaining_after == 10
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, rate_limiter):
        """Test handling concurrent requests to same endpoint."""
        endpoint = '/concurrent-test'
        
        async def make_request():
            try:
                await rate_limiter.acquire(endpoint)
                return True
            except RateLimitExceeded:
                return False
        
        # Make 15 concurrent requests (limit is 10)
        tasks = [make_request() for _ in range(15)]
        results = await asyncio.gather(*tasks)
        
        # Only 10 should succeed
        successes = sum(1 for r in results if r)
        assert successes == 10
    
    def test_metrics_collection(self, rate_limiter):
        """Test rate limiter metrics collection."""
        metrics = rate_limiter.get_metrics()
        
        assert 'total_requests' in metrics
        assert 'rate_limited_requests' in metrics
        assert 'endpoints' in metrics
        assert isinstance(metrics['total_requests'], int)
        assert isinstance(metrics['rate_limited_requests'], int)
        assert isinstance(metrics['endpoints'], dict)
    
    @pytest.mark.asyncio
    async def test_dynamic_limit_adjustment(self):
        """Test dynamic rate limit adjustment."""
        config = RateLimitConfig(default_limit=10)
        rate_limiter = RateLimiter(config)
        
        endpoint = '/dynamic'
        
        # Initial limit
        assert rate_limiter.config.get_limit(endpoint) == 10
        
        # Adjust limit
        rate_limiter.adjust_limit(endpoint, 5)
        assert rate_limiter.config.get_limit(endpoint) == 5
        
        # Should only allow 5 requests now
        for i in range(5):
            await rate_limiter.acquire(endpoint)
        
        with pytest.raises(RateLimitExceeded):
            await rate_limiter.acquire(endpoint)


class TestRateLimitExceeded:
    """Test cases for RateLimitExceeded exception."""
    
    def test_exception_attributes(self):
        """Test exception contains expected attributes."""
        exc = RateLimitExceeded(
            endpoint='/test',
            limit=100,
            window=60,
            retry_after=30
        )
        
        assert exc.endpoint == '/test'
        assert exc.limit == 100
        assert exc.window == 60
        assert exc.retry_after == 30
        assert 'Rate limit exceeded' in str(exc)
    
    def test_exception_with_backoff_multiplier(self):
        """Test exception with exponential backoff multiplier."""
        exc = RateLimitExceeded(
            endpoint='/test',
            limit=10,
            window=60,
            retry_after=5,
            backoff_multiplier=2.0
        )
        
        assert exc.backoff_multiplier == 2.0
        assert exc.retry_after == 5


@pytest.mark.integration
class TestRateLimiterIntegration:
    """Integration tests for rate limiter with realistic scenarios."""
    
    @pytest.mark.asyncio
    async def test_api_client_integration(self):
        """Test rate limiter integration with API client pattern."""
        config = RateLimitConfig(
            default_limit=5,
            default_window=60,
            endpoint_limits={'/orders': 2}
        )
        rate_limiter = RateLimiter(config)
        
        # Simulate API client making requests
        async def simulate_api_call(endpoint):
            await rate_limiter.acquire(endpoint)
            # Simulate API processing time
            await asyncio.sleep(0.01)
            return f"Response from {endpoint}"
        
        # Test mixed endpoint usage
        results = []
        
        # Make requests to different endpoints
        for i in range(3):
            results.append(await simulate_api_call('/markets'))
        
        for i in range(2):
            results.append(await simulate_api_call('/orders'))
        
        assert len(results) == 5
        
        # Next order request should fail
        with pytest.raises(RateLimitExceeded):
            await simulate_api_call('/orders')
        
        # But markets should still work
        results.append(await simulate_api_call('/markets'))
        assert len(results) == 6
    
    @pytest.mark.asyncio
    async def test_high_load_scenario(self):
        """Test rate limiter under high load."""
        config = RateLimitConfig(default_limit=50, default_window=60)
        rate_limiter = RateLimiter(config)
        
        async def worker(worker_id, endpoint):
            success_count = 0
            rate_limited_count = 0
            
            for i in range(20):  # Each worker tries 20 requests
                try:
                    await rate_limiter.acquire(endpoint)
                    success_count += 1
                    await asyncio.sleep(0.001)  # Small delay
                except RateLimitExceeded:
                    rate_limited_count += 1
            
            return success_count, rate_limited_count
        
        # Create 10 workers, each making 20 requests (200 total)
        # Only 50 should succeed
        tasks = [worker(i, '/high-load') for i in range(10)]
        results = await asyncio.gather(*tasks)
        
        total_success = sum(r[0] for r in results)
        total_rate_limited = sum(r[1] for r in results)
        
        assert total_success == 50  # Only 50 should succeed
        assert total_rate_limited == 150  # 150 should be rate limited
        assert total_success + total_rate_limited == 200