# Alpaca API Rate Limiting & Connection Management

## Overview
Alpaca enforces specific rate limits to ensure fair API usage and system stability. Understanding and properly implementing rate limiting strategies is crucial for building robust trading applications that can scale and handle high-frequency operations.

## Rate Limits (2024)

### Standard Rate Limits
- **REST API**: 200 requests per minute (all accounts)
- **Burst Limit**: 10 requests per second
- **WebSocket**: 1 connection per account (base version)
- **Unlimited Plan**: 1,000 requests per minute

### Rate Limit Headers
Alpaca returns rate limit information in response headers:
- `X-RateLimit-Limit`: Maximum requests per window
- `X-RateLimit-Remaining`: Remaining requests in current window
- `X-RateLimit-Reset`: Timestamp when window resets

## Rate Limiting Implementation

### Basic Rate Limiter
```python
import time
import threading
from collections import deque
from functools import wraps

class AlpacaRateLimiter:
    def __init__(self, max_requests_per_minute=200, max_burst=10):
        self.max_requests_per_minute = max_requests_per_minute
        self.max_burst = max_burst
        self.requests_per_minute = deque()
        self.last_request_time = 0
        self.lock = threading.Lock()

        # Calculate minimum time between requests
        self.min_interval = 60.0 / max_requests_per_minute  # 0.3 seconds for 200/min
        self.burst_interval = 1.0 / max_burst  # 0.1 seconds for 10/sec

    def wait_if_needed(self):
        """Wait if rate limit would be exceeded"""
        with self.lock:
            current_time = time.time()

            # Check burst limit (10 requests/second)
            time_since_last = current_time - self.last_request_time
            if time_since_last < self.burst_interval:
                sleep_time = self.burst_interval - time_since_last
                time.sleep(sleep_time)
                current_time = time.time()

            # Check per-minute limit
            # Remove requests older than 1 minute
            while self.requests_per_minute and current_time - self.requests_per_minute[0] > 60:
                self.requests_per_minute.popleft()

            # If we've made too many requests in the last minute, wait
            if len(self.requests_per_minute) >= self.max_requests_per_minute:
                wait_time = 60 - (current_time - self.requests_per_minute[0])
                if wait_time > 0:
                    time.sleep(wait_time)
                    current_time = time.time()
                    # Clean up old requests after waiting
                    while self.requests_per_minute and current_time - self.requests_per_minute[0] > 60:
                        self.requests_per_minute.popleft()

            # Record this request
            self.requests_per_minute.append(current_time)
            self.last_request_time = current_time

    def rate_limited(self, func):
        """Decorator to add rate limiting to functions"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            self.wait_if_needed()
            return func(*args, **kwargs)
        return wrapper

# Usage example
rate_limiter = AlpacaRateLimiter()

@rate_limiter.rate_limited
def make_api_call():
    # Your API call here
    pass
```

### Advanced Rate Limiter with Backoff
```python
import random
import logging
from datetime import datetime, timedelta

class AdvancedRateLimiter:
    def __init__(self, max_requests_per_minute=200, max_burst=10, enable_backoff=True):
        self.max_requests_per_minute = max_requests_per_minute
        self.max_burst = max_burst
        self.enable_backoff = enable_backoff

        self.request_history = deque()
        self.consecutive_limits = 0
        self.backoff_until = None
        self.lock = threading.Lock()

        # Setup logging
        self.logger = logging.getLogger(__name__)

    def exponential_backoff(self, attempt):
        """Calculate exponential backoff delay"""
        base_delay = 1.0  # Start with 1 second
        max_delay = 300.0  # Maximum 5 minutes

        delay = min(base_delay * (2 ** attempt), max_delay)
        # Add jitter to prevent thundering herd
        jitter = random.uniform(0.1, 0.3) * delay
        return delay + jitter

    def check_rate_limit_response(self, response):
        """Check response for rate limit information"""
        if hasattr(response, 'status_code') and response.status_code == 429:
            self.consecutive_limits += 1
            self.logger.warning(f"Rate limit exceeded. Consecutive limits: {self.consecutive_limits}")

            if self.enable_backoff:
                backoff_delay = self.exponential_backoff(self.consecutive_limits)
                self.backoff_until = datetime.now() + timedelta(seconds=backoff_delay)
                self.logger.info(f"Backing off for {backoff_delay:.2f} seconds")

            return True
        else:
            # Reset consecutive limits on successful request
            self.consecutive_limits = 0
            self.backoff_until = None
            return False

    def wait_for_rate_limit(self):
        """Wait if rate limiting is needed"""
        with self.lock:
            current_time = time.time()

            # Check if we're in backoff period
            if self.backoff_until and datetime.now() < self.backoff_until:
                wait_time = (self.backoff_until - datetime.now()).total_seconds()
                self.logger.info(f"In backoff period, waiting {wait_time:.2f} seconds")
                time.sleep(wait_time)
                current_time = time.time()

            # Standard rate limiting logic
            # Remove old requests (older than 1 minute)
            while self.request_history and current_time - self.request_history[0] > 60:
                self.request_history.popleft()

            # Check if we need to wait
            if len(self.request_history) >= self.max_requests_per_minute:
                wait_time = 60 - (current_time - self.request_history[0]) + 0.1  # Small buffer
                self.logger.info(f"Rate limit approached, waiting {wait_time:.2f} seconds")
                time.sleep(wait_time)
                current_time = time.time()

            # Add current request to history
            self.request_history.append(current_time)

    def make_request_with_retry(self, request_func, max_retries=3, *args, **kwargs):
        """Make request with automatic retry and rate limiting"""
        for attempt in range(max_retries + 1):
            try:
                self.wait_for_rate_limit()
                response = request_func(*args, **kwargs)

                if self.check_rate_limit_response(response):
                    if attempt < max_retries:
                        continue  # Retry on rate limit
                    else:
                        raise Exception("Max retries reached due to rate limiting")

                return response

            except Exception as e:
                if attempt < max_retries:
                    wait_time = self.exponential_backoff(attempt)
                    self.logger.warning(f"Request failed (attempt {attempt + 1}), retrying in {wait_time:.2f}s: {e}")
                    time.sleep(wait_time)
                else:
                    self.logger.error(f"Request failed after {max_retries} retries: {e}")
                    raise
```

## Connection Management

### Robust HTTP Client
```python
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

class RobustAlpacaClient:
    def __init__(self, api_key, secret_key, base_url, rate_limiter=None):
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = base_url
        self.rate_limiter = rate_limiter or AdvancedRateLimiter()

        # Setup session with retry strategy
        self.session = requests.Session()
        self.setup_session()

    def setup_session(self):
        """Configure session with retry strategy and connection pooling"""
        # Retry strategy
        retry_strategy = Retry(
            total=3,
            status_forcelist=[429, 500, 502, 503, 504],  # HTTP status codes to retry
            method_whitelist=["HEAD", "GET", "OPTIONS"],  # Only retry safe methods
            backoff_factor=1,  # Wait 1, 2, 4 seconds between retries
            raise_on_status=False
        )

        # HTTP adapter with retry strategy
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=10,  # Number of connection pools
            pool_maxsize=10,      # Maximum connections in pool
            pool_block=True       # Block when pool is full
        )

        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        # Set default headers
        self.session.headers.update({
            'APCA-API-KEY-ID': self.api_key,
            'APCA-API-SECRET-KEY': self.secret_key,
            'Content-Type': 'application/json',
            'User-Agent': 'AlpacaRobustClient/1.0'
        })

    def make_request(self, method, endpoint, **kwargs):
        """Make HTTP request with rate limiting and error handling"""
        url = f"{self.base_url}{endpoint}"

        # Set timeout if not provided
        kwargs.setdefault('timeout', (5, 30))  # (connect, read) timeout

        try:
            response = self.rate_limiter.make_request_with_retry(
                self.session.request,
                method=method,
                url=url,
                **kwargs
            )

            response.raise_for_status()
            return response.json() if response.content else {}

        except requests.exceptions.RequestException as e:
            self.handle_request_error(e, method, endpoint)
            raise

    def handle_request_error(self, error, method, endpoint):
        """Handle and log request errors"""
        error_info = {
            'method': method,
            'endpoint': endpoint,
            'error_type': type(error).__name__,
            'error_message': str(error)
        }

        if hasattr(error, 'response') and error.response is not None:
            error_info['status_code'] = error.response.status_code
            error_info['response_text'] = error.response.text[:500]  # Limit log size

        self.rate_limiter.logger.error(f"Request error: {error_info}")

    def get(self, endpoint, **kwargs):
        return self.make_request('GET', endpoint, **kwargs)

    def post(self, endpoint, **kwargs):
        return self.make_request('POST', endpoint, **kwargs)

    def put(self, endpoint, **kwargs):
        return self.make_request('PUT', endpoint, **kwargs)

    def delete(self, endpoint, **kwargs):
        return self.make_request('DELETE', endpoint, **kwargs)
```

### WebSocket Connection Management
```python
import websocket
import json
import threading
import ssl

class RobustWebSocketManager:
    def __init__(self, url, api_key, secret_key, reconnect_interval=5, max_reconnect_attempts=10):
        self.url = url
        self.api_key = api_key
        self.secret_key = secret_key
        self.reconnect_interval = reconnect_interval
        self.max_reconnect_attempts = max_reconnect_attempts

        self.ws = None
        self.is_connected = False
        self.authenticated = False
        self.reconnect_attempts = 0
        self.should_reconnect = True

        self.subscriptions = set()
        self.message_handlers = {}
        self.heartbeat_thread = None
        self.heartbeat_interval = 30

        self.logger = logging.getLogger(__name__)

    def connect(self):
        """Establish WebSocket connection"""
        try:
            self.ws = websocket.WebSocketApp(
                self.url,
                on_open=self.on_open,
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close
            )

            # Run with SSL context for security
            ssl_context = ssl.create_default_context()
            self.ws.run_forever(sslopt={"context": ssl_context})

        except Exception as e:
            self.logger.error(f"Failed to establish WebSocket connection: {e}")
            self.attempt_reconnect()

    def on_open(self, ws):
        """Handle WebSocket connection open"""
        self.logger.info("WebSocket connection opened")
        self.is_connected = True
        self.reconnect_attempts = 0

        # Authenticate immediately
        self.authenticate()

        # Start heartbeat
        self.start_heartbeat()

    def authenticate(self):
        """Authenticate WebSocket connection"""
        auth_message = {
            "action": "auth",
            "key": self.api_key,
            "secret": self.secret_key
        }
        self.send_message(auth_message)

    def on_message(self, ws, message):
        """Handle incoming WebSocket messages"""
        try:
            data = json.loads(message)

            # Handle authentication response
            if isinstance(data, list):
                for msg in data:
                    self.process_message(msg)
            else:
                self.process_message(data)

        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to decode WebSocket message: {e}")

    def process_message(self, message):
        """Process individual WebSocket message"""
        msg_type = message.get('T')

        if msg_type == 'success' and message.get('msg') == 'authenticated':
            self.authenticated = True
            self.logger.info("WebSocket authenticated successfully")
            # Resubscribe to previous subscriptions
            self.resubscribe()

        elif msg_type == 'error':
            self.logger.error(f"WebSocket error: {message.get('msg')}")

        elif msg_type in self.message_handlers:
            # Route message to appropriate handler
            try:
                self.message_handlers[msg_type](message)
            except Exception as e:
                self.logger.error(f"Error in message handler for {msg_type}: {e}")

    def on_error(self, ws, error):
        """Handle WebSocket errors"""
        self.logger.error(f"WebSocket error: {error}")
        self.is_connected = False
        self.authenticated = False

    def on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket connection close"""
        self.logger.info(f"WebSocket connection closed: {close_status_code} - {close_msg}")
        self.is_connected = False
        self.authenticated = False

        if self.heartbeat_thread:
            self.heartbeat_thread = None

        if self.should_reconnect:
            self.attempt_reconnect()

    def attempt_reconnect(self):
        """Attempt to reconnect with exponential backoff"""
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            self.logger.error("Max reconnection attempts reached, giving up")
            return

        self.reconnect_attempts += 1
        wait_time = min(self.reconnect_interval * (2 ** (self.reconnect_attempts - 1)), 300)

        self.logger.info(f"Reconnection attempt {self.reconnect_attempts} in {wait_time} seconds")
        time.sleep(wait_time)

        if self.should_reconnect:
            self.connect()

    def send_message(self, message):
        """Send message to WebSocket"""
        if self.ws and self.is_connected:
            try:
                self.ws.send(json.dumps(message))
            except Exception as e:
                self.logger.error(f"Failed to send WebSocket message: {e}")

    def subscribe(self, subscription_data):
        """Subscribe to data streams"""
        if self.authenticated:
            self.send_message(subscription_data)
            # Store subscription for reconnection
            self.subscriptions.add(json.dumps(subscription_data))
        else:
            self.logger.warning("Cannot subscribe: not authenticated")

    def resubscribe(self):
        """Resubscribe to all previous subscriptions"""
        for subscription in self.subscriptions:
            subscription_data = json.loads(subscription)
            self.send_message(subscription_data)

    def start_heartbeat(self):
        """Start heartbeat thread to keep connection alive"""
        def heartbeat():
            while self.is_connected:
                time.sleep(self.heartbeat_interval)
                if self.is_connected:
                    self.send_message({"action": "ping"})

        self.heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
        self.heartbeat_thread.start()

    def add_message_handler(self, message_type, handler):
        """Add handler for specific message type"""
        self.message_handlers[message_type] = handler

    def disconnect(self):
        """Gracefully disconnect WebSocket"""
        self.should_reconnect = False
        if self.ws:
            self.ws.close()
```

## Best Practices Implementation

### Production-Ready Client
```python
class ProductionAlpacaClient:
    def __init__(self, config):
        self.config = config
        self.rate_limiter = AdvancedRateLimiter(
            max_requests_per_minute=config.get('rate_limit', 200),
            enable_backoff=True
        )

        self.http_client = RobustAlpacaClient(
            api_key=config['api_key'],
            secret_key=config['secret_key'],
            base_url=config['base_url'],
            rate_limiter=self.rate_limiter
        )

        self.ws_manager = None
        self.circuit_breaker = CircuitBreaker()

    def initialize_streaming(self):
        """Initialize WebSocket streaming with proper error handling"""
        self.ws_manager = RobustWebSocketManager(
            url=self.config['stream_url'],
            api_key=self.config['api_key'],
            secret_key=self.config['secret_key']
        )

        # Add message handlers
        self.ws_manager.add_message_handler('t', self.handle_trade)
        self.ws_manager.add_message_handler('q', self.handle_quote)
        self.ws_manager.add_message_handler('b', self.handle_bar)

        # Start connection in separate thread
        threading.Thread(target=self.ws_manager.connect, daemon=True).start()

    def handle_trade(self, trade_data):
        """Handle trade messages with error boundaries"""
        try:
            # Process trade data
            symbol = trade_data['S']
            price = trade_data['p']
            size = trade_data['s']

            # Your trading logic here
            self.process_trade_signal(symbol, price, size)

        except Exception as e:
            self.rate_limiter.logger.error(f"Error processing trade: {e}")

    def process_trade_signal(self, symbol, price, size):
        """Process trade signal with circuit breaker pattern"""
        @self.circuit_breaker.call
        def execute_trade():
            # Your trade execution logic
            pass

        try:
            execute_trade()
        except Exception as e:
            self.rate_limiter.logger.error(f"Trade execution failed: {e}")

class CircuitBreaker:
    def __init__(self, failure_threshold=5, recovery_timeout=60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN

    def call(self, func):
        """Decorator to implement circuit breaker pattern"""
        def wrapper(*args, **kwargs):
            if self.state == 'OPEN':
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = 'HALF_OPEN'
                else:
                    raise Exception("Circuit breaker is OPEN")

            try:
                result = func(*args, **kwargs)
                if self.state == 'HALF_OPEN':
                    self.state = 'CLOSED'
                    self.failure_count = 0
                return result

            except Exception as e:
                self.failure_count += 1
                self.last_failure_time = time.time()

                if self.failure_count >= self.failure_threshold:
                    self.state = 'OPEN'

                raise e

        return wrapper
```

## Monitoring and Metrics

### Rate Limit Monitoring
```python
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge

class AlpacaMetrics:
    def __init__(self):
        # Counters
        self.api_requests_total = Counter('alpaca_api_requests_total', 'Total API requests', ['endpoint', 'method', 'status'])
        self.rate_limit_hits = Counter('alpaca_rate_limit_hits_total', 'Rate limit hits')
        self.websocket_reconnects = Counter('alpaca_websocket_reconnects_total', 'WebSocket reconnections')

        # Histograms
        self.request_duration = Histogram('alpaca_request_duration_seconds', 'Request duration')
        self.rate_limit_wait_time = Histogram('alpaca_rate_limit_wait_seconds', 'Rate limit wait time')

        # Gauges
        self.current_connections = Gauge('alpaca_current_connections', 'Current WebSocket connections')
        self.requests_per_minute = Gauge('alpaca_requests_per_minute', 'Current requests per minute')

    def record_api_request(self, endpoint, method, status_code, duration):
        self.api_requests_total.labels(endpoint=endpoint, method=method, status=status_code).inc()
        self.request_duration.observe(duration)

    def record_rate_limit_hit(self, wait_time):
        self.rate_limit_hits.inc()
        self.rate_limit_wait_time.observe(wait_time)

    def record_websocket_reconnect(self):
        self.websocket_reconnects.inc()

# Usage in monitoring system
metrics = AlpacaMetrics()

# Integrate with your existing monitoring setup
def start_metrics_server(port=8000):
    prometheus_client.start_http_server(port)
```

This comprehensive guide provides robust rate limiting and connection management strategies for building production-ready Alpaca API integrations that can handle high-frequency trading scenarios while respecting API limits and maintaining system stability.