# Alpaca API Error Handling & Retry Strategies

## Overview
Robust error handling and retry strategies are essential for building reliable trading applications. Alpaca's Python SDK includes built-in retry mechanisms, but implementing comprehensive error handling ensures your trading strategies can handle network issues, API limits, and unexpected market conditions gracefully.

## Alpaca SDK Built-in Error Handling

### Default Retry Configuration
The Alpaca Python SDK automatically retries certain HTTP status codes:
- **429**: Too Many Requests (rate limiting)
- **504**: Gateway Timeout
- **Default Retries**: 3 attempts
- **Default Wait Time**: 3 seconds between retries

### Environment Configuration
```bash
# Configure retry behavior via environment variables
APCA_RETRY_MAX=5           # Maximum retry attempts
APCA_RETRY_WAIT=2          # Seconds between retries
APCA_RETRY_CODES=429,504,502,503  # HTTP codes to retry
```

### SDK Exception Types
```python
from alpaca.common.exceptions import APIError
from alpaca.trading.client import TradingClient

# Common exception types
try:
    trading_client = TradingClient(api_key="key", secret_key="secret", paper=True)
    account = trading_client.get_account()
except APIError as e:
    print(f"API Error: {e}")
    print(f"Status Code: {e.status_code}")
    print(f"Response: {e.response}")
```

## Comprehensive Error Handling Framework

### Custom Exception Classes
```python
from enum import Enum
import time
import logging
from typing import Optional, Callable, Any

class AlpacaErrorType(Enum):
    RATE_LIMIT = "rate_limit"
    AUTHENTICATION = "authentication"
    INSUFFICIENT_FUNDS = "insufficient_funds"
    MARKET_CLOSED = "market_closed"
    INVALID_ORDER = "invalid_order"
    NETWORK_ERROR = "network_error"
    SERVER_ERROR = "server_error"
    UNKNOWN = "unknown"

class AlpacaException(Exception):
    def __init__(self, message: str, error_type: AlpacaErrorType, status_code: Optional[int] = None,
                 retry_after: Optional[int] = None, original_exception: Optional[Exception] = None):
        super().__init__(message)
        self.error_type = error_type
        self.status_code = status_code
        self.retry_after = retry_after
        self.original_exception = original_exception
        self.timestamp = time.time()

class RateLimitException(AlpacaException):
    def __init__(self, message: str, retry_after: int = 60):
        super().__init__(message, AlpacaErrorType.RATE_LIMIT, 429, retry_after)

class InsufficientFundsException(AlpacaException):
    def __init__(self, message: str):
        super().__init__(message, AlpacaErrorType.INSUFFICIENT_FUNDS, 422)

class MarketClosedException(AlpacaException):
    def __init__(self, message: str):
        super().__init__(message, AlpacaErrorType.MARKET_CLOSED, 422)

class AuthenticationException(AlpacaException):
    def __init__(self, message: str):
        super().__init__(message, AlpacaErrorType.AUTHENTICATION, 401)

class NetworkException(AlpacaException):
    def __init__(self, message: str, original_exception: Exception):
        super().__init__(message, AlpacaErrorType.NETWORK_ERROR, None, None, original_exception)
```

### Error Classification and Handler
```python
import re
import requests
from alpaca.common.exceptions import APIError

class AlpacaErrorHandler:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.setup_error_patterns()

    def setup_error_patterns(self):
        """Define patterns for classifying errors"""
        self.error_patterns = {
            AlpacaErrorType.RATE_LIMIT: [
                r"rate limit",
                r"too many requests",
                r"429"
            ],
            AlpacaErrorType.INSUFFICIENT_FUNDS: [
                r"insufficient.*fund",
                r"buying power",
                r"not enough.*cash",
                r"insufficient.*balance"
            ],
            AlpacaErrorType.MARKET_CLOSED: [
                r"market.*closed",
                r"outside.*trading.*hours",
                r"market not open"
            ],
            AlpacaErrorType.INVALID_ORDER: [
                r"invalid.*order",
                r"order.*rejected",
                r"invalid.*symbol",
                r"invalid.*quantity"
            ],
            AlpacaErrorType.AUTHENTICATION: [
                r"unauthorized",
                r"invalid.*credentials",
                r"authentication.*failed",
                r"forbidden"
            ]
        }

    def classify_error(self, error: Exception) -> AlpacaErrorType:
        """Classify error based on message and status code"""
        error_message = str(error).lower()

        # Check status code first
        if hasattr(error, 'status_code'):
            if error.status_code == 429:
                return AlpacaErrorType.RATE_LIMIT
            elif error.status_code == 401:
                return AlpacaErrorType.AUTHENTICATION
            elif error.status_code in [500, 502, 503, 504]:
                return AlpacaErrorType.SERVER_ERROR

        # Check error message patterns
        for error_type, patterns in self.error_patterns.items():
            for pattern in patterns:
                if re.search(pattern, error_message):
                    return error_type

        # Check for network-related errors
        if isinstance(error, (requests.exceptions.ConnectionError,
                             requests.exceptions.Timeout,
                             requests.exceptions.RequestException)):
            return AlpacaErrorType.NETWORK_ERROR

        return AlpacaErrorType.UNKNOWN

    def create_alpaca_exception(self, error: Exception) -> AlpacaException:
        """Convert generic exception to AlpacaException"""
        error_type = self.classify_error(error)
        message = str(error)

        status_code = getattr(error, 'status_code', None)

        if error_type == AlpacaErrorType.RATE_LIMIT:
            retry_after = self.extract_retry_after(error)
            return RateLimitException(message, retry_after)
        elif error_type == AlpacaErrorType.INSUFFICIENT_FUNDS:
            return InsufficientFundsException(message)
        elif error_type == AlpacaErrorType.MARKET_CLOSED:
            return MarketClosedException(message)
        elif error_type == AlpacaErrorType.AUTHENTICATION:
            return AuthenticationException(message)
        elif error_type == AlpacaErrorType.NETWORK_ERROR:
            return NetworkException(message, error)
        else:
            return AlpacaException(message, error_type, status_code, original_exception=error)

    def extract_retry_after(self, error: Exception) -> int:
        """Extract retry-after header or estimate wait time"""
        if hasattr(error, 'response') and error.response:
            retry_after = error.response.headers.get('Retry-After')
            if retry_after:
                return int(retry_after)

        # Default retry times based on error type
        return 60  # Default 1 minute for rate limits
```

## Advanced Retry Strategies

### Exponential Backoff with Jitter
```python
import random
import asyncio
from functools import wraps
from typing import Union, Tuple

class RetryStrategy:
    def __init__(self, max_attempts: int = 3, base_delay: float = 1.0,
                 max_delay: float = 60.0, exponential_base: float = 2.0,
                 jitter: bool = True):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter

    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt with exponential backoff"""
        delay = self.base_delay * (self.exponential_base ** attempt)
        delay = min(delay, self.max_delay)

        if self.jitter:
            # Add jitter to prevent thundering herd
            jitter_range = delay * 0.1
            delay += random.uniform(-jitter_range, jitter_range)

        return max(0, delay)

    def should_retry(self, exception: AlpacaException, attempt: int) -> bool:
        """Determine if operation should be retried"""
        if attempt >= self.max_attempts:
            return False

        # Don't retry authentication errors
        if exception.error_type == AlpacaErrorType.AUTHENTICATION:
            return False

        # Don't retry business logic errors
        if exception.error_type in [AlpacaErrorType.INSUFFICIENT_FUNDS,
                                   AlpacaErrorType.INVALID_ORDER]:
            return False

        # Retry network, server, and rate limit errors
        if exception.error_type in [AlpacaErrorType.NETWORK_ERROR,
                                   AlpacaErrorType.SERVER_ERROR,
                                   AlpacaErrorType.RATE_LIMIT]:
            return True

        return False

class RetryableClient:
    def __init__(self, trading_client, retry_strategy: RetryStrategy = None,
                 error_handler: AlpacaErrorHandler = None):
        self.trading_client = trading_client
        self.retry_strategy = retry_strategy or RetryStrategy()
        self.error_handler = error_handler or AlpacaErrorHandler()
        self.logger = logging.getLogger(__name__)

    def retry_on_error(self, func: Callable) -> Callable:
        """Decorator for adding retry logic to methods"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(self.retry_strategy.max_attempts):
                try:
                    return func(*args, **kwargs)

                except Exception as e:
                    alpaca_exception = self.error_handler.create_alpaca_exception(e)
                    last_exception = alpaca_exception

                    if not self.retry_strategy.should_retry(alpaca_exception, attempt):
                        self.logger.error(f"Not retrying {func.__name__}: {alpaca_exception}")
                        raise alpaca_exception

                    delay = self.retry_strategy.calculate_delay(attempt)

                    # Special handling for rate limits
                    if alpaca_exception.error_type == AlpacaErrorType.RATE_LIMIT and alpaca_exception.retry_after:
                        delay = alpaca_exception.retry_after

                    self.logger.warning(
                        f"Attempt {attempt + 1} failed for {func.__name__}: {alpaca_exception}. "
                        f"Retrying in {delay:.2f} seconds..."
                    )

                    time.sleep(delay)

            # All attempts failed
            self.logger.error(f"All retry attempts failed for {func.__name__}")
            raise last_exception

        return wrapper

    def get_account(self):
        @self.retry_on_error
        def _get_account():
            return self.trading_client.get_account()
        return _get_account()

    def submit_order(self, order_request):
        @self.retry_on_error
        def _submit_order():
            return self.trading_client.submit_order(order_request)
        return _submit_order()

    def get_positions(self):
        @self.retry_on_error
        def _get_positions():
            return self.trading_client.get_all_positions()
        return _get_positions()

    def cancel_order(self, order_id):
        @self.retry_on_error
        def _cancel_order():
            return self.trading_client.cancel_order_by_id(order_id)
        return _cancel_order()
```

### Async Retry Implementation
```python
import asyncio
from typing import AsyncCallable

class AsyncRetryableClient:
    def __init__(self, retry_strategy: RetryStrategy = None,
                 error_handler: AlpacaErrorHandler = None):
        self.retry_strategy = retry_strategy or RetryStrategy()
        self.error_handler = error_handler or AlpacaErrorHandler()
        self.logger = logging.getLogger(__name__)

    def async_retry(self, func: AsyncCallable) -> AsyncCallable:
        """Async decorator for retry logic"""
        async def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(self.retry_strategy.max_attempts):
                try:
                    return await func(*args, **kwargs)

                except Exception as e:
                    alpaca_exception = self.error_handler.create_alpaca_exception(e)
                    last_exception = alpaca_exception

                    if not self.retry_strategy.should_retry(alpaca_exception, attempt):
                        self.logger.error(f"Not retrying {func.__name__}: {alpaca_exception}")
                        raise alpaca_exception

                    delay = self.retry_strategy.calculate_delay(attempt)

                    if alpaca_exception.error_type == AlpacaErrorType.RATE_LIMIT and alpaca_exception.retry_after:
                        delay = alpaca_exception.retry_after

                    self.logger.warning(
                        f"Async attempt {attempt + 1} failed for {func.__name__}: {alpaca_exception}. "
                        f"Retrying in {delay:.2f} seconds..."
                    )

                    await asyncio.sleep(delay)

            self.logger.error(f"All async retry attempts failed for {func.__name__}")
            raise last_exception

        return wrapper

    @async_retry
    async def async_get_account(self, trading_client):
        """Async wrapper for get_account"""
        return trading_client.get_account()

    @async_retry
    async def async_submit_order(self, trading_client, order_request):
        """Async wrapper for submit_order"""
        return trading_client.submit_order(order_request)
```

## Circuit Breaker Pattern

### Trading Circuit Breaker
```python
from datetime import datetime, timedelta
from enum import Enum

class CircuitState(Enum):
    CLOSED = "closed"    # Normal operation
    OPEN = "open"        # Failure threshold reached, blocking requests
    HALF_OPEN = "half_open"  # Testing if service recovered

class TradingCircuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 300,
                 expected_exception: type = AlpacaException):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
        self.logger = logging.getLogger(__name__)

    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                self.logger.info("Circuit breaker moved to HALF_OPEN state")
            else:
                raise AlpacaException(
                    "Circuit breaker is OPEN - trading operations blocked",
                    AlpacaErrorType.SERVER_ERROR
                )

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result

        except self.expected_exception as e:
            self._on_failure(e)
            raise

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        return (self.last_failure_time and
                time.time() - self.last_failure_time >= self.recovery_timeout)

    def _on_success(self):
        """Handle successful operation"""
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.logger.info("Circuit breaker reset to CLOSED state")

    def _on_failure(self, exception: AlpacaException):
        """Handle failed operation"""
        self.failure_count += 1
        self.last_failure_time = time.time()

        # Don't trip circuit breaker for business logic errors
        if exception.error_type in [AlpacaErrorType.INSUFFICIENT_FUNDS,
                                   AlpacaErrorType.INVALID_ORDER,
                                   AlpacaErrorType.MARKET_CLOSED]:
            return

        if (self.failure_count >= self.failure_threshold and
            self.state != CircuitState.OPEN):
            self.state = CircuitState.OPEN
            self.logger.error(
                f"Circuit breaker OPENED after {self.failure_count} failures. "
                f"Trading operations blocked for {self.recovery_timeout} seconds."
            )

class CircuitBreakerTradingClient:
    def __init__(self, trading_client, circuit_breaker: TradingCircuitBreaker = None):
        self.trading_client = trading_client
        self.circuit_breaker = circuit_breaker or TradingCircuitBreaker()

    def submit_order(self, order_request):
        """Submit order with circuit breaker protection"""
        return self.circuit_breaker.call(
            self.trading_client.submit_order,
            order_request
        )

    def cancel_order(self, order_id):
        """Cancel order with circuit breaker protection"""
        return self.circuit_breaker.call(
            self.trading_client.cancel_order_by_id,
            order_id
        )

    def get_account(self):
        """Get account with circuit breaker protection"""
        return self.circuit_breaker.call(
            self.trading_client.get_account
        )
```

## Complete Error Handling System

### Production-Ready Trading Client
```python
class RobustTradingClient:
    def __init__(self, api_key: str, secret_key: str, paper: bool = True,
                 retry_strategy: RetryStrategy = None,
                 circuit_breaker: TradingCircuitBreaker = None):

        # Base client
        self.base_client = TradingClient(
            api_key=api_key,
            secret_key=secret_key,
            paper=paper
        )

        # Error handling components
        self.error_handler = AlpacaErrorHandler()
        self.retry_strategy = retry_strategy or RetryStrategy()
        self.circuit_breaker = circuit_breaker or TradingCircuitBreaker()

        # Monitoring
        self.metrics = TradingMetrics()
        self.logger = logging.getLogger(__name__)

    def execute_with_protection(self, operation_name: str, func: Callable, *args, **kwargs):
        """Execute operation with full error protection"""
        start_time = time.time()

        try:
            # Execute with circuit breaker and retry
            retryable_func = self._make_retryable(func)
            result = self.circuit_breaker.call(retryable_func, *args, **kwargs)

            # Record success metrics
            duration = time.time() - start_time
            self.metrics.record_success(operation_name, duration)

            return result

        except AlpacaException as e:
            # Record failure metrics
            duration = time.time() - start_time
            self.metrics.record_failure(operation_name, e.error_type, duration)

            # Log structured error information
            self.logger.error(
                f"Operation failed: {operation_name}",
                extra={
                    'error_type': e.error_type.value,
                    'status_code': e.status_code,
                    'duration': duration,
                    'retry_after': e.retry_after
                }
            )

            raise

    def _make_retryable(self, func: Callable) -> Callable:
        """Add retry logic to function"""
        def retryable_func(*args, **kwargs):
            last_exception = None

            for attempt in range(self.retry_strategy.max_attempts):
                try:
                    return func(*args, **kwargs)

                except Exception as e:
                    alpaca_exception = self.error_handler.create_alpaca_exception(e)
                    last_exception = alpaca_exception

                    if not self.retry_strategy.should_retry(alpaca_exception, attempt):
                        raise alpaca_exception

                    delay = self.retry_strategy.calculate_delay(attempt)
                    if alpaca_exception.error_type == AlpacaErrorType.RATE_LIMIT and alpaca_exception.retry_after:
                        delay = alpaca_exception.retry_after

                    time.sleep(delay)

            raise last_exception

        return retryable_func

    # Trading operations with protection
    def submit_order(self, order_request):
        return self.execute_with_protection(
            "submit_order",
            self.base_client.submit_order,
            order_request
        )

    def get_account(self):
        return self.execute_with_protection(
            "get_account",
            self.base_client.get_account
        )

    def get_positions(self):
        return self.execute_with_protection(
            "get_positions",
            self.base_client.get_all_positions
        )

    def cancel_order(self, order_id):
        return self.execute_with_protection(
            "cancel_order",
            self.base_client.cancel_order_by_id,
            order_id
        )

    def get_orders(self, status=None, limit=None):
        return self.execute_with_protection(
            "get_orders",
            self.base_client.get_orders,
            status=status,
            limit=limit
        )

class TradingMetrics:
    def __init__(self):
        self.success_count = 0
        self.failure_count = 0
        self.total_duration = 0
        self.failure_by_type = {}
        self.operation_stats = {}

    def record_success(self, operation: str, duration: float):
        self.success_count += 1
        self.total_duration += duration
        self._update_operation_stats(operation, True, duration)

    def record_failure(self, operation: str, error_type: AlpacaErrorType, duration: float):
        self.failure_count += 1
        self.total_duration += duration

        if error_type not in self.failure_by_type:
            self.failure_by_type[error_type] = 0
        self.failure_by_type[error_type] += 1

        self._update_operation_stats(operation, False, duration)

    def _update_operation_stats(self, operation: str, success: bool, duration: float):
        if operation not in self.operation_stats:
            self.operation_stats[operation] = {
                'success_count': 0,
                'failure_count': 0,
                'total_duration': 0,
                'avg_duration': 0
            }

        stats = self.operation_stats[operation]
        if success:
            stats['success_count'] += 1
        else:
            stats['failure_count'] += 1

        stats['total_duration'] += duration
        total_calls = stats['success_count'] + stats['failure_count']
        stats['avg_duration'] = stats['total_duration'] / total_calls

    def get_summary(self):
        total_operations = self.success_count + self.failure_count
        success_rate = self.success_count / total_operations if total_operations > 0 else 0
        avg_duration = self.total_duration / total_operations if total_operations > 0 else 0

        return {
            'total_operations': total_operations,
            'success_rate': success_rate,
            'avg_duration': avg_duration,
            'failure_by_type': dict(self.failure_by_type),
            'operation_stats': dict(self.operation_stats)
        }

# Usage example
def main():
    # Initialize robust client
    client = RobustTradingClient(
        api_key="your_api_key",
        secret_key="your_secret_key",
        paper=True,
        retry_strategy=RetryStrategy(max_attempts=5, base_delay=1.0),
        circuit_breaker=TradingCircuitBreaker(failure_threshold=3, recovery_timeout=300)
    )

    try:
        # All operations now have comprehensive error handling
        account = client.get_account()
        print(f"Account: {account.id}")

        # Submit order with automatic retry and error handling
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        order_request = MarketOrderRequest(
            symbol="AAPL",
            qty=1,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY
        )

        order = client.submit_order(order_request)
        print(f"Order submitted: {order.id}")

    except AlpacaException as e:
        print(f"Trading operation failed: {e.error_type.value} - {e}")

    # Print metrics summary
    metrics_summary = client.metrics.get_summary()
    print(f"Trading Metrics: {metrics_summary}")

if __name__ == "__main__":
    main()
```

## Error Monitoring and Alerting

### Error Monitoring System
```python
import smtplib
from email.mime.text import MimeText
from datetime import datetime

class ErrorMonitor:
    def __init__(self, alert_threshold: int = 10, time_window: int = 300):
        self.alert_threshold = alert_threshold
        self.time_window = time_window
        self.error_history = []
        self.last_alert_time = 0

    def record_error(self, error: AlpacaException):
        """Record error for monitoring"""
        current_time = time.time()

        # Clean old errors outside time window
        self.error_history = [
            (timestamp, err) for timestamp, err in self.error_history
            if current_time - timestamp <= self.time_window
        ]

        # Add new error
        self.error_history.append((current_time, error))

        # Check if alert threshold reached
        if (len(self.error_history) >= self.alert_threshold and
            current_time - self.last_alert_time > self.time_window):
            self.send_alert()

    def send_alert(self):
        """Send alert notification"""
        error_summary = self.generate_error_summary()

        # Send email alert (implement based on your requirements)
        self.send_email_alert(error_summary)

        # Send Slack alert (implement based on your requirements)
        self.send_slack_alert(error_summary)

        self.last_alert_time = time.time()

    def generate_error_summary(self):
        """Generate error summary for alerts"""
        error_types = {}
        for _, error in self.error_history:
            error_type = error.error_type.value
            if error_type not in error_types:
                error_types[error_type] = 0
            error_types[error_type] += 1

        return {
            'total_errors': len(self.error_history),
            'time_window': self.time_window,
            'error_breakdown': error_types,
            'timestamp': datetime.now().isoformat()
        }

    def send_email_alert(self, error_summary):
        """Send email alert (implement based on your email setup)"""
        pass

    def send_slack_alert(self, error_summary):
        """Send Slack alert (implement based on your Slack setup)"""
        pass
```

This comprehensive error handling and retry system provides robust protection for trading applications, ensuring they can handle various failure scenarios while maintaining system stability and providing detailed error reporting.