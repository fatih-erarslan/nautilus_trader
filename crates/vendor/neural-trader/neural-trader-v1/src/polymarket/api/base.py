"""
Base Polymarket API client with common functionality

This module provides the abstract base class for all Polymarket API clients,
including common functionality like authentication, error handling, rate limiting,
caching, and metrics collection.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union
import aiohttp
from cachetools import TTLCache
import time

from ..utils import PolymarketConfig, authenticate


logger = logging.getLogger(__name__)


class PolymarketAPIError(Exception):
    """Base exception for all Polymarket API errors"""
    
    def __init__(self, message: str, status_code: Optional[int] = None, response_data: Optional[Dict] = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data
        self.timestamp = datetime.now()


class RateLimitError(PolymarketAPIError):
    """Rate limit exceeded error"""
    
    def __init__(self, retry_after: Optional[int] = None, **kwargs):
        super().__init__("Rate limit exceeded", **kwargs)
        self.retry_after = retry_after


class OrderError(PolymarketAPIError):
    """Order-related errors (placement, execution, cancellation)"""
    pass


class MarketClosedError(PolymarketAPIError):
    """Market is closed for trading"""
    pass


class AuthenticationError(PolymarketAPIError):
    """Authentication or authorization failed"""
    pass


class ValidationError(PolymarketAPIError):
    """Input validation failed"""
    pass


class PolymarketClient(ABC):
    """
    Abstract base class for all Polymarket API clients
    
    Provides common functionality:
    - Authentication management
    - HTTP request handling with retries
    - Rate limiting
    - Response caching
    - Error handling and logging
    - Performance metrics
    """
    
    def __init__(
        self,
        config: Optional[PolymarketConfig] = None,
        cache_ttl: int = 300,
        cache_maxsize: int = 1000,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Initialize the Polymarket client
        
        Args:
            config: Client configuration, uses default if None
            cache_ttl: Cache time-to-live in seconds
            cache_maxsize: Maximum cache size
            max_retries: Maximum number of retry attempts
            retry_delay: Base delay between retries in seconds
        """
        self.config = config or PolymarketConfig()
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Caching
        self._cache = TTLCache(maxsize=cache_maxsize, ttl=cache_ttl)
        
        # Rate limiting and retry configuration
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Metrics tracking
        self._metrics = {
            'requests_total': 0,
            'requests_successful': 0,
            'requests_failed': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'rate_limits_hit': 0,
            'average_response_time': 0.0,
            'last_request_time': None,
        }
        
        # Authentication
        self._auth_headers: Optional[Dict[str, str]] = None
        
        logger.info(f"Initialized {self.__class__.__name__} with cache TTL={cache_ttl}s")
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self._ensure_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
    
    async def _ensure_session(self):
        """Ensure HTTP session exists"""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                headers=self._get_default_headers()
            )
    
    def _get_default_headers(self) -> Dict[str, str]:
        """Get default HTTP headers for requests"""
        headers = {
            'User-Agent': f'ai-news-trader-polymarket/{self.__class__.__name__}',
            'Accept': 'application/json',
            'Content-Type': 'application/json',
        }
        
        # Add authentication headers if available
        if self._auth_headers:
            headers.update(self._auth_headers)
            
        return headers
    
    async def authenticate(self) -> bool:
        """
        Authenticate with the Polymarket API
        
        Returns:
            True if authentication successful, False otherwise
        """
        try:
            self._auth_headers = await authenticate(
                api_key=self.config.api_key,
                private_key=self.config.private_key
            )
            logger.info("Successfully authenticated with Polymarket API")
            return True
            
        except Exception as e:
            logger.error(f"Authentication failed: {str(e)}")
            raise AuthenticationError(f"Authentication failed: {str(e)}") from e
    
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        data: Optional[Dict] = None,
        use_cache: bool = True,
        cache_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Make HTTP request with error handling, retries, and caching
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            params: Query parameters
            data: Request body data
            use_cache: Whether to use response caching
            cache_key: Custom cache key, auto-generated if None
            
        Returns:
            Response data as dictionary
            
        Raises:
            PolymarketAPIError: On API errors
        """
        await self._ensure_session()
        
        # Generate cache key
        if use_cache and method.upper() == 'GET':
            if cache_key is None:
                cache_key = f"{endpoint}:{hash(str(params) if params else '')}"
            
            # Check cache first
            if cache_key in self._cache:
                self._metrics['cache_hits'] += 1
                logger.debug(f"Cache hit for {cache_key}")
                return self._cache[cache_key]
        
        # Prepare request
        url = f"{self._get_base_url()}/{endpoint.lstrip('/')}"
        request_kwargs = {
            'method': method,
            'url': url,
            'params': params,
        }
        
        if data:
            request_kwargs['json'] = data
        
        # Execute request with retries
        last_exception = None
        for attempt in range(self.max_retries + 1):
            try:
                start_time = time.time()
                self._metrics['requests_total'] += 1
                
                async with self.session.request(**request_kwargs) as response:
                    response_time = time.time() - start_time
                    self._update_response_time(response_time)
                    self._metrics['last_request_time'] = datetime.now()
                    
                    # Handle response
                    response_data = await self._handle_response(response)
                    
                    # Cache successful GET responses
                    if use_cache and method.upper() == 'GET' and cache_key:
                        self._cache[cache_key] = response_data
                        self._metrics['cache_misses'] += 1
                    
                    self._metrics['requests_successful'] += 1
                    return response_data
                    
            except RateLimitError as e:
                self._metrics['rate_limits_hit'] += 1
                if attempt < self.max_retries:
                    wait_time = e.retry_after or (self.retry_delay * (2 ** attempt))
                    logger.warning(f"Rate limited, waiting {wait_time}s before retry {attempt + 1}")
                    await asyncio.sleep(wait_time)
                    continue
                last_exception = e
                
            except PolymarketAPIError as e:
                # Don't retry client errors (4xx)
                if e.status_code and 400 <= e.status_code < 500:
                    last_exception = e
                    break
                    
                # Retry server errors (5xx)
                if attempt < self.max_retries:
                    wait_time = self.retry_delay * (2 ** attempt)
                    logger.warning(f"Server error, retrying in {wait_time}s: {str(e)}")
                    await asyncio.sleep(wait_time)
                    continue
                last_exception = e
                
            except Exception as e:
                last_exception = PolymarketAPIError(f"Unexpected error: {str(e)}")
                if attempt < self.max_retries:
                    wait_time = self.retry_delay * (2 ** attempt)
                    logger.warning(f"Unexpected error, retrying in {wait_time}s: {str(e)}")
                    await asyncio.sleep(wait_time)
                    continue
        
        # All retries exhausted
        self._metrics['requests_failed'] += 1
        logger.error(f"Request failed after {self.max_retries + 1} attempts: {str(last_exception)}")
        raise last_exception
    
    async def _handle_response(self, response: aiohttp.ClientResponse) -> Dict[str, Any]:
        """
        Handle HTTP response and convert to appropriate exception if needed
        
        Args:
            response: HTTP response object
            
        Returns:
            Parsed response data
            
        Raises:
            PolymarketAPIError: On API errors
        """
        try:
            response_data = await response.json()
        except Exception:
            response_data = {"message": await response.text()}
        
        # Handle successful responses
        if 200 <= response.status < 300:
            return response_data
        
        # Handle specific error codes
        error_message = response_data.get('message', f'HTTP {response.status}')
        
        if response.status == 429:
            retry_after = int(response.headers.get('Retry-After', 60))
            raise RateLimitError(retry_after=retry_after, status_code=response.status, response_data=response_data)
        
        elif response.status == 401:
            raise AuthenticationError(error_message, status_code=response.status, response_data=response_data)
        
        elif response.status == 400:
            raise ValidationError(error_message, status_code=response.status, response_data=response_data)
        
        elif 400 <= response.status < 500:
            raise PolymarketAPIError(error_message, status_code=response.status, response_data=response_data)
        
        else:
            raise PolymarketAPIError(f"Server error: {error_message}", status_code=response.status, response_data=response_data)
    
    def _update_response_time(self, response_time: float):
        """Update average response time metric"""
        current_avg = self._metrics['average_response_time']
        total_requests = self._metrics['requests_total']
        
        if total_requests == 1:
            self._metrics['average_response_time'] = response_time
        else:
            # Exponential moving average
            alpha = 0.1
            self._metrics['average_response_time'] = alpha * response_time + (1 - alpha) * current_avg
    
    @abstractmethod
    def _get_base_url(self) -> str:
        """Get the base URL for this client's API"""
        pass
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get client performance metrics"""
        return {
            'client': self.__class__.__name__,
            **self._metrics,
            'cache_info': {
                'size': len(self._cache),
                'maxsize': self._cache.maxsize,
                'ttl': self._cache.ttl,
                'hit_rate': (
                    self._metrics['cache_hits'] / 
                    (self._metrics['cache_hits'] + self._metrics['cache_misses'])
                    if (self._metrics['cache_hits'] + self._metrics['cache_misses']) > 0 
                    else 0.0
                )
            }
        }
    
    def clear_cache(self):
        """Clear the response cache"""
        self._cache.clear()
        logger.info(f"Cleared cache for {self.__class__.__name__}")
    
    async def close(self):
        """Close the HTTP session"""
        if self.session and not self.session.closed:
            await self.session.close()
            logger.info(f"Closed HTTP session for {self.__class__.__name__}")
    
    # Abstract methods that subclasses must implement
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the API is healthy and accessible"""
        pass