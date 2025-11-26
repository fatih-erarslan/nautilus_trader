"""
Async Utility Functions
======================

Helper functions for async operations and concurrency control.
"""

import asyncio
import functools
import logging
from typing import Any, Awaitable, Callable, List, Optional, TypeVar, Union
from concurrent.futures import ThreadPoolExecutor
import time

logger = logging.getLogger(__name__)

T = TypeVar('T')

def run_sync(async_func: Callable[..., Awaitable[T]], *args, **kwargs) -> T:
    """
    Run an async function synchronously.
    
    Args:
        async_func: Async function to run
        *args: Positional arguments
        **kwargs: Keyword arguments
        
    Returns:
        Function result
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If we're already in an event loop, use a thread executor
            with ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, async_func(*args, **kwargs))
                return future.result()
        else:
            return loop.run_until_complete(async_func(*args, **kwargs))
    except RuntimeError:
        # No event loop running
        return asyncio.run(async_func(*args, **kwargs))

def async_retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """
    Decorator for retrying async functions with exponential backoff.
    
    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries
        backoff: Backoff multiplier
        exceptions: Exceptions to catch and retry on
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_attempts - 1:
                        logger.error(f"Function {func.__name__} failed after {max_attempts} attempts: {e}")
                        raise
                    
                    logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying in {current_delay}s...")
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff
            
            # This should never be reached, but just in case
            if last_exception:
                raise last_exception
                
        return wrapper
    return decorator

async def timeout_after(
    seconds: float,
    coro: Awaitable[T],
    timeout_result: Optional[T] = None
) -> T:
    """
    Execute a coroutine with a timeout.
    
    Args:
        seconds: Timeout in seconds
        coro: Coroutine to execute
        timeout_result: Result to return on timeout (if None, raises TimeoutError)
        
    Returns:
        Coroutine result or timeout_result
        
    Raises:
        TimeoutError: If timeout occurs and timeout_result is None
    """
    try:
        return await asyncio.wait_for(coro, timeout=seconds)
    except asyncio.TimeoutError:
        if timeout_result is not None:
            return timeout_result
        raise TimeoutError(f"Operation timed out after {seconds} seconds")

async def gather_with_limit(
    *coros: Awaitable[T],
    limit: int = 10,
    return_exceptions: bool = False
) -> List[Union[T, Exception]]:
    """
    Execute multiple coroutines with a concurrency limit.
    
    Args:
        *coros: Coroutines to execute
        limit: Maximum number of concurrent executions
        return_exceptions: If True, exceptions are returned instead of raised
        
    Returns:
        List of results
    """
    semaphore = asyncio.Semaphore(limit)
    
    async def _limited_coro(coro: Awaitable[T]) -> T:
        async with semaphore:
            return await coro
    
    limited_coros = [_limited_coro(coro) for coro in coros]
    return await asyncio.gather(*limited_coros, return_exceptions=return_exceptions)

class AsyncCache:
    """Simple async cache with TTL support."""
    
    def __init__(self, default_ttl: float = 300.0):
        """
        Initialize async cache.
        
        Args:
            default_ttl: Default time-to-live in seconds
        """
        self._cache = {}
        self._default_ttl = default_ttl
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key in self._cache:
            value, expiry = self._cache[key]
            if time.time() < expiry:
                return value
            else:
                del self._cache[key]
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set value in cache."""
        if ttl is None:
            ttl = self._default_ttl
        expiry = time.time() + ttl
        self._cache[key] = (value, expiry)
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        if key in self._cache:
            del self._cache[key]
            return True
        return False
    
    async def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
    
    async def cleanup_expired(self) -> int:
        """Remove expired entries and return count."""
        current_time = time.time()
        expired_keys = [
            key for key, (_, expiry) in self._cache.items()
            if current_time >= expiry
        ]
        
        for key in expired_keys:
            del self._cache[key]
        
        return len(expired_keys)

class RateLimiter:
    """Async rate limiter using token bucket algorithm."""
    
    def __init__(self, rate: float, capacity: int = None):
        """
        Initialize rate limiter.
        
        Args:
            rate: Tokens per second
            capacity: Maximum token capacity (default: rate)
        """
        self.rate = rate
        self.capacity = capacity or int(rate)
        self.tokens = self.capacity
        self.last_update = time.time()
        self._lock = asyncio.Lock()
    
    async def acquire(self, tokens: int = 1) -> bool:
        """
        Acquire tokens from the bucket.
        
        Args:
            tokens: Number of tokens to acquire
            
        Returns:
            True if tokens were acquired, False otherwise
        """
        async with self._lock:
            now = time.time()
            elapsed = now - self.last_update
            
            # Add tokens based on elapsed time
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            self.last_update = now
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            
            return False
    
    async def wait_for_tokens(self, tokens: int = 1) -> None:
        """
        Wait until tokens are available.
        
        Args:
            tokens: Number of tokens needed
        """
        while not await self.acquire(tokens):
            wait_time = tokens / self.rate
            await asyncio.sleep(wait_time)

class AsyncBatchProcessor:
    """Process items in batches asynchronously."""
    
    def __init__(
        self,
        processor: Callable[[List[Any]], Awaitable[List[Any]]],
        batch_size: int = 10,
        max_concurrent_batches: int = 3
    ):
        """
        Initialize batch processor.
        
        Args:
            processor: Async function to process batches
            batch_size: Size of each batch
            max_concurrent_batches: Maximum concurrent batch processing
        """
        self.processor = processor
        self.batch_size = batch_size
        self.semaphore = asyncio.Semaphore(max_concurrent_batches)
    
    async def process(self, items: List[Any]) -> List[Any]:
        """
        Process items in batches.
        
        Args:
            items: Items to process
            
        Returns:
            Processed items
        """
        if not items:
            return []
        
        # Split into batches
        batches = [
            items[i:i + self.batch_size]
            for i in range(0, len(items), self.batch_size)
        ]
        
        # Process batches concurrently
        async def _process_batch(batch):
            async with self.semaphore:
                return await self.processor(batch)
        
        batch_results = await asyncio.gather(
            *[_process_batch(batch) for batch in batches],
            return_exceptions=True
        )
        
        # Flatten results
        results = []
        for batch_result in batch_results:
            if isinstance(batch_result, Exception):
                logger.error(f"Batch processing error: {batch_result}")
                continue
            results.extend(batch_result)
        
        return results

class AsyncDebouncer:
    """Debounce async function calls."""
    
    def __init__(self, delay: float):
        """
        Initialize debouncer.
        
        Args:
            delay: Debounce delay in seconds
        """
        self.delay = delay
        self._tasks = {}
    
    async def debounce(self, key: str, coro: Awaitable[T]) -> Optional[T]:
        """
        Debounce a coroutine call.
        
        Args:
            key: Unique key for the operation
            coro: Coroutine to debounce
            
        Returns:
            Result if executed, None if debounced
        """
        # Cancel existing task for this key
        if key in self._tasks:
            self._tasks[key].cancel()
        
        # Create new task
        async def _delayed_execution():
            await asyncio.sleep(self.delay)
            try:
                result = await coro
                return result
            finally:
                if key in self._tasks:
                    del self._tasks[key]
        
        task = asyncio.create_task(_delayed_execution())
        self._tasks[key] = task
        
        try:
            return await task
        except asyncio.CancelledError:
            return None

# Global instances for convenience
default_cache = AsyncCache()
default_rate_limiter = RateLimiter(rate=10.0)  # 10 requests per second