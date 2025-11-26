# Alpaca API Authentication & Security Best Practices

## Authentication Overview
Alpaca API uses API key-based authentication with separate credentials for paper and live trading environments. The authentication method involves passing key-secret pairs in HTTP headers for REST API calls and initial WebSocket handshake.

## API Key Management

### Key Generation and Storage
```python
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Secure credential storage
ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
ALPACA_BASE_URL = os.getenv('ALPACA_BASE_URL')  # Paper or live URL

# Never hardcode credentials in source code
# ❌ BAD
api_key = "PKTEST12345..."  # Never do this

# ✅ GOOD
api_key = os.getenv('ALPACA_API_KEY')
```

### Environment Configuration (.env file)
```bash
# Paper Trading Configuration
ALPACA_API_KEY=PKTEST_your_paper_key_here
ALPACA_SECRET_KEY=your_paper_secret_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# Live Trading Configuration (separate file recommended)
# ALPACA_API_KEY=AKFZ_your_live_key_here
# ALPACA_SECRET_KEY=your_live_secret_here
# ALPACA_BASE_URL=https://api.alpaca.markets

# Market Data Configuration
ALPACA_MARKET_DATA_URL=https://data.alpaca.markets
ALPACA_STREAM_URL=wss://stream.data.alpaca.markets/v2/iex
```

## Paper vs Live Trading Configuration

### Alpaca-py SDK Configuration
```python
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.live.stock import StockDataStream

class AlpacaClientManager:
    def __init__(self, use_paper=True):
        self.api_key = os.getenv('ALPACA_API_KEY')
        self.secret_key = os.getenv('ALPACA_SECRET_KEY')
        self.use_paper = use_paper

        # Trading client
        self.trading_client = TradingClient(
            api_key=self.api_key,
            secret_key=self.secret_key,
            paper=use_paper  # True for paper trading, False for live
        )

        # Historical data client (same for both paper and live)
        self.data_client = StockHistoricalDataClient(
            api_key=self.api_key,
            secret_key=self.secret_key
        )

        # Live streaming client
        self.stream_client = StockDataStream(
            api_key=self.api_key,
            secret_key=self.secret_key,
            feed="iex"  # or "sip" for premium data
        )

    def switch_to_live_trading(self):
        """Switch to live trading - use with extreme caution"""
        if not self.confirm_live_trading():
            raise ValueError("Live trading not confirmed")

        self.use_paper = False
        self.trading_client = TradingClient(
            api_key=self.api_key,
            secret_key=self.secret_key,
            paper=False
        )

    def confirm_live_trading(self):
        """Safety check before switching to live trading"""
        confirmation = input("Are you sure you want to switch to LIVE trading with REAL money? (type 'CONFIRM'): ")
        return confirmation == "CONFIRM"
```

## REST API Authentication

### Header-Based Authentication
```python
import requests

class SecureAPIClient:
    def __init__(self):
        self.api_key = os.getenv('ALPACA_API_KEY')
        self.secret_key = os.getenv('ALPACA_SECRET_KEY')
        self.base_url = os.getenv('ALPACA_BASE_URL')

        self.headers = {
            'APCA-API-KEY-ID': self.api_key,
            'APCA-API-SECRET-KEY': self.secret_key,
            'Content-Type': 'application/json'
        }

    def make_authenticated_request(self, endpoint, method='GET', data=None):
        url = f"{self.base_url}{endpoint}"

        try:
            response = requests.request(
                method=method,
                url=url,
                headers=self.headers,
                json=data,
                timeout=30  # Always set timeouts
            )
            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")
            raise

    def get_account_info(self):
        return self.make_authenticated_request('/v2/account')

    def place_order(self, order_data):
        return self.make_authenticated_request('/v2/orders', 'POST', order_data)
```

## WebSocket Authentication

### Secure WebSocket Connection
```python
import json
import websocket
import threading

class SecureWebSocketClient:
    def __init__(self):
        self.api_key = os.getenv('ALPACA_API_KEY')
        self.secret_key = os.getenv('ALPACA_SECRET_KEY')
        self.ws_url = "wss://stream.data.alpaca.markets/v2/iex"
        self.connection = None
        self.authenticated = False

    def create_connection(self):
        self.connection = websocket.WebSocketApp(
            self.ws_url,
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close
        )

    def on_open(self, ws):
        print("WebSocket connection opened")
        # Must authenticate within 10 seconds
        auth_message = {
            "action": "auth",
            "key": self.api_key,
            "secret": self.secret_key
        }
        ws.send(json.dumps(auth_message))

    def on_message(self, ws, message):
        data = json.loads(message)

        if data.get('T') == 'success' and data.get('msg') == 'authenticated':
            print("Successfully authenticated")
            self.authenticated = True
            self.subscribe_to_data()
        elif data.get('T') == 'error':
            print(f"Authentication error: {data.get('msg')}")
            self.authenticated = False

    def subscribe_to_data(self):
        if self.authenticated:
            subscribe_message = {
                "action": "subscribe",
                "trades": ["AAPL", "TSLA"],
                "quotes": ["AAPL", "TSLA"]
            }
            self.connection.send(json.dumps(subscribe_message))

    def on_error(self, ws, error):
        print(f"WebSocket error: {error}")
        self.authenticated = False

    def on_close(self, ws, close_status_code, close_msg):
        print("WebSocket connection closed")
        self.authenticated = False
```

## Security Best Practices

### 1. Credential Management
```python
class CredentialManager:
    def __init__(self):
        self.validate_credentials()

    def validate_credentials(self):
        """Validate that all required credentials are present"""
        required_vars = ['ALPACA_API_KEY', 'ALPACA_SECRET_KEY', 'ALPACA_BASE_URL']
        missing_vars = []

        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)

        if missing_vars:
            raise ValueError(f"Missing required environment variables: {missing_vars}")

    def mask_credentials(self, credential):
        """Mask credentials for logging purposes"""
        if len(credential) <= 8:
            return "*" * len(credential)
        return credential[:4] + "*" * (len(credential) - 8) + credential[-4:]

    def log_safe_credentials(self):
        """Log credentials safely for debugging"""
        api_key = os.getenv('ALPACA_API_KEY')
        print(f"Using API Key: {self.mask_credentials(api_key)}")
```

### 2. Environment Separation
```python
class EnvironmentManager:
    def __init__(self):
        self.environment = self.detect_environment()
        self.load_environment_config()

    def detect_environment(self):
        """Detect current environment"""
        env = os.getenv('TRADING_ENVIRONMENT', 'paper').lower()
        if env not in ['paper', 'live']:
            raise ValueError("TRADING_ENVIRONMENT must be 'paper' or 'live'")
        return env

    def load_environment_config(self):
        """Load environment-specific configuration"""
        if self.environment == 'paper':
            self.load_paper_config()
        else:
            self.load_live_config()

    def load_paper_config(self):
        """Load paper trading configuration"""
        self.base_url = "https://paper-api.alpaca.markets"
        self.stream_url = "wss://paper-api.alpaca.markets/stream"
        print("🟡 PAPER TRADING MODE - No real money at risk")

    def load_live_config(self):
        """Load live trading configuration with extra warnings"""
        self.base_url = "https://api.alpaca.markets"
        self.stream_url = "wss://api.alpaca.markets/stream"
        print("🔴 LIVE TRADING MODE - REAL MONEY AT RISK!")

        # Additional confirmation for live trading
        if not self.confirm_live_mode():
            raise ValueError("Live trading mode not confirmed")

    def confirm_live_mode(self):
        """Require explicit confirmation for live trading"""
        return os.getenv('CONFIRM_LIVE_TRADING') == 'true'
```

### 3. Rate Limiting and Connection Security
```python
import time
from functools import wraps

class RateLimitedClient:
    def __init__(self):
        self.last_request_time = 0
        self.min_request_interval = 0.3  # 200 requests/minute = 0.3s interval
        self.request_count = 0
        self.request_window_start = time.time()

    def rate_limit(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_time = time.time()

            # Check per-second rate limit (10 requests/second)
            time_since_last = current_time - self.last_request_time
            if time_since_last < 0.1:  # 100ms minimum between requests
                time.sleep(0.1 - time_since_last)

            # Check per-minute rate limit (200 requests/minute)
            if current_time - self.request_window_start >= 60:
                self.request_count = 0
                self.request_window_start = current_time

            if self.request_count >= 200:
                wait_time = 60 - (current_time - self.request_window_start)
                if wait_time > 0:
                    print(f"Rate limit reached, waiting {wait_time:.1f}s")
                    time.sleep(wait_time)
                    self.request_count = 0
                    self.request_window_start = time.time()

            self.request_count += 1
            self.last_request_time = time.time()

            return func(*args, **kwargs)
        return wrapper
```

### 4. Secure Configuration Class
```python
class SecureAlpacaConfig:
    def __init__(self):
        self.environment = os.getenv('TRADING_ENVIRONMENT', 'paper')
        self.validate_environment()
        self.setup_credentials()
        self.setup_endpoints()

    def validate_environment(self):
        """Validate trading environment"""
        if self.environment not in ['paper', 'live']:
            raise ValueError("Invalid environment. Must be 'paper' or 'live'")

        if self.environment == 'live':
            self.confirm_live_trading()

    def confirm_live_trading(self):
        """Require multiple confirmations for live trading"""
        confirmations = [
            os.getenv('CONFIRM_LIVE_TRADING') == 'true',
            os.getenv('LIVE_TRADING_APPROVED') == 'true',
            os.getenv('REAL_MONEY_ACKNOWLEDGED') == 'true'
        ]

        if not all(confirmations):
            raise ValueError("Live trading requires explicit confirmation")

    def setup_credentials(self):
        """Setup and validate credentials"""
        self.api_key = os.getenv('ALPACA_API_KEY')
        self.secret_key = os.getenv('ALPACA_SECRET_KEY')

        if not self.api_key or not self.secret_key:
            raise ValueError("Missing API credentials")

        # Validate key format
        if self.environment == 'paper' and not self.api_key.startswith('PK'):
            raise ValueError("Paper trading key should start with 'PK'")
        elif self.environment == 'live' and not self.api_key.startswith('AK'):
            raise ValueError("Live trading key should start with 'AK'")

    def setup_endpoints(self):
        """Setup appropriate endpoints"""
        if self.environment == 'paper':
            self.base_url = "https://paper-api.alpaca.markets"
            self.stream_url = "wss://paper-api.alpaca.markets/stream"
        else:
            self.base_url = "https://api.alpaca.markets"
            self.stream_url = "wss://api.alpaca.markets/stream"
```

## Error Handling and Security

### Secure Error Handling
```python
import logging

class SecureErrorHandler:
    def __init__(self):
        self.setup_logging()

    def setup_logging(self):
        """Setup secure logging that doesn't leak credentials"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('alpaca_trading.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def log_api_error(self, error, endpoint, sanitized_params=None):
        """Log API errors without exposing sensitive data"""
        error_message = f"API Error on {endpoint}: {str(error)}"
        if sanitized_params:
            error_message += f" | Params: {sanitized_params}"
        self.logger.error(error_message)

    def sanitize_for_logging(self, data):
        """Remove sensitive data from logging"""
        if isinstance(data, dict):
            sanitized = {}
            for key, value in data.items():
                if any(sensitive in key.lower() for sensitive in ['key', 'secret', 'token', 'auth']):
                    sanitized[key] = "*" * len(str(value))
                else:
                    sanitized[key] = value
            return sanitized
        return data
```

## Testing and Validation

### Credential Testing
```python
class CredentialTester:
    def __init__(self, config):
        self.config = config
        self.trading_client = TradingClient(
            api_key=config.api_key,
            secret_key=config.secret_key,
            paper=config.environment == 'paper'
        )

    def test_authentication(self):
        """Test if credentials are valid"""
        try:
            account = self.trading_client.get_account()
            print(f"✅ Authentication successful")
            print(f"Account ID: {account.id}")
            print(f"Account Status: {account.status}")
            print(f"Trading Environment: {self.config.environment}")
            return True
        except Exception as e:
            print(f"❌ Authentication failed: {e}")
            return False

    def test_permissions(self):
        """Test API permissions"""
        tests = [
            ("Account Info", self.test_account_access),
            ("Positions", self.test_positions_access),
            ("Orders", self.test_orders_access),
            ("Assets", self.test_assets_access)
        ]

        results = {}
        for test_name, test_func in tests:
            try:
                test_func()
                results[test_name] = "✅ Pass"
            except Exception as e:
                results[test_name] = f"❌ Fail: {e}"

        return results

    def test_account_access(self):
        account = self.trading_client.get_account()
        return account is not None

    def test_positions_access(self):
        positions = self.trading_client.get_all_positions()
        return positions is not None

    def test_orders_access(self):
        orders = self.trading_client.get_orders()
        return orders is not None

    def test_assets_access(self):
        assets = self.trading_client.get_all_assets()
        return assets is not None
```

## Production Deployment Security

### Docker Environment Security
```dockerfile
# Dockerfile
FROM python:3.11-slim

# Create non-root user
RUN useradd -m -u 1001 trader
USER trader

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set secure defaults
ENV PYTHONUNBUFFERED=1
ENV TRADING_ENVIRONMENT=paper

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

CMD ["python", "main.py"]
```

### Kubernetes Secrets Management
```yaml
# k8s-secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: alpaca-credentials
type: Opaque
stringData:
  api-key: "your-api-key"
  secret-key: "your-secret-key"
  base-url: "https://paper-api.alpaca.markets"
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: trading-bot
spec:
  replicas: 1
  selector:
    matchLabels:
      app: trading-bot
  template:
    metadata:
      labels:
        app: trading-bot
    spec:
      containers:
      - name: trading-bot
        image: trading-bot:latest
        env:
        - name: ALPACA_API_KEY
          valueFrom:
            secretKeyRef:
              name: alpaca-credentials
              key: api-key
        - name: ALPACA_SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: alpaca-credentials
              key: secret-key
        - name: ALPACA_BASE_URL
          valueFrom:
            secretKeyRef:
              name: alpaca-credentials
              key: base-url
```

## Monitoring and Alerting

### Security Monitoring
```python
class SecurityMonitor:
    def __init__(self):
        self.failed_auth_attempts = 0
        self.last_auth_time = time.time()
        self.alert_threshold = 5

    def monitor_authentication(self, success, error_msg=None):
        """Monitor authentication attempts"""
        current_time = time.time()

        if success:
            self.failed_auth_attempts = 0
            self.last_auth_time = current_time
            self.logger.info("Successful authentication")
        else:
            self.failed_auth_attempts += 1
            self.logger.warning(f"Failed authentication attempt #{self.failed_auth_attempts}")

            if self.failed_auth_attempts >= self.alert_threshold:
                self.send_security_alert("Multiple failed authentication attempts")

    def send_security_alert(self, message):
        """Send security alert (implement your preferred alerting method)"""
        self.logger.critical(f"SECURITY ALERT: {message}")
        # Implement email/SMS/Slack notification here
```

This comprehensive guide provides the foundation for secure Alpaca API integration with proper authentication, environment management, and security best practices for both development and production environments.