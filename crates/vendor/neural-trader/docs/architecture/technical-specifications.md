# AI News Trader: Technical Specifications

## 1. Broker API Integration Specifications

### 1.1 Interactive Brokers Integration

#### Connection Configuration
```python
# src/brokers/interactive_brokers.py
class InteractiveBrokersConfig:
    """Configuration for Interactive Brokers TWS API"""
    host: str = "127.0.0.1"
    port: int = 7497  # Paper trading: 7497, Live: 7496
    client_id: int = 1
    account_id: str
    timeout: int = 10
    max_requests_per_second: int = 50
    enable_logging: bool = True
```

#### API Methods
```python
class InteractiveBrokersAdapter(BrokerAdapter):
    """Interactive Brokers API adapter implementation"""
    
    async def connect(self) -> ConnectionResult:
        """Establish connection to TWS/IB Gateway"""
        
    async def place_order(self, order: Order) -> OrderResult:
        """Place order via IB API"""
        
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel existing order"""
        
    async def get_positions(self) -> List[Position]:
        """Retrieve current positions"""
        
    async def get_account_info(self) -> AccountInfo:
        """Get account details and balances"""
        
    async def stream_market_data(self, symbols: List[str]) -> AsyncIterator[MarketData]:
        """Stream real-time market data"""
        
    async def get_order_book(self, symbol: str, depth: int = 5) -> OrderBook:
        """Get order book depth"""
        
    async def get_historical_data(self, symbol: str, timeframe: str, 
                                 start: datetime, end: datetime) -> List[OHLCV]:
        """Retrieve historical price data"""
```

### 1.2 Alpaca Integration

#### Configuration
```python
class AlpacaConfig:
    """Configuration for Alpaca API"""
    api_key: str
    secret_key: str
    base_url: str = "https://paper-api.alpaca.markets"  # Paper trading
    # base_url: str = "https://api.alpaca.markets"  # Live trading
    api_version: str = "v2"
    timeout: int = 30
    max_requests_per_minute: int = 200
```

#### API Implementation
```python
class AlpacaAdapter(BrokerAdapter):
    """Alpaca API adapter implementation"""
    
    async def place_order(self, order: Order) -> OrderResult:
        """Place order via Alpaca API"""
        payload = {
            "symbol": order.symbol,
            "qty": str(order.quantity),
            "side": order.side.lower(),
            "type": order.order_type.lower(),
            "time_in_force": order.time_in_force,
            "limit_price": str(order.limit_price) if order.limit_price else None,
            "stop_price": str(order.stop_price) if order.stop_price else None
        }
        
        response = await self._post("/orders", payload)
        return OrderResult(
            order_id=response["id"],
            status=response["status"],
            filled_qty=float(response["filled_qty"]),
            avg_fill_price=float(response["filled_avg_price"]) if response["filled_avg_price"] else None
        )
```

### 1.3 TD Ameritrade Integration

#### OAuth2 Authentication
```python
class TDAmeritradAuthManager:
    """Handle TD Ameritrade OAuth2 authentication"""
    
    def __init__(self, client_id: str, redirect_uri: str):
        self.client_id = client_id
        self.redirect_uri = redirect_uri
        self.access_token = None
        self.refresh_token = None
        self.token_expires = None
    
    async def get_authorization_url(self) -> str:
        """Generate authorization URL"""
        
    async def exchange_code_for_token(self, code: str) -> TokenResponse:
        """Exchange authorization code for access token"""
        
    async def refresh_access_token(self) -> TokenResponse:
        """Refresh expired access token"""
```

## 2. News API Integration Specifications

### 2.1 Bloomberg API Integration

#### Configuration
```python
class BloombergConfig:
    """Configuration for Bloomberg API"""
    application_name: str = "ai-news-trader"
    session_options: Dict[str, Any] = {
        "serverHost": "localhost",
        "serverPort": 8194,
        "authenticationOptions": {
            "user": "your_username",
            "application": "your_application"
        }
    }
    subscription_options: Dict[str, str] = {
        "interval": "1.0",  # seconds
        "conflateEvents": "false"
    }
```

#### News Retrieval
```python
class BloombergNewsAdapter(NewsSource):
    """Bloomberg News API adapter"""
    
    async def fetch_latest(self, limit: int = 100) -> List[NewsItem]:
        """Fetch latest news from Bloomberg"""
        
    async def fetch_by_symbol(self, symbol: str, limit: int = 50) -> List[NewsItem]:
        """Fetch news for specific symbol"""
        
    async def stream(self) -> AsyncIterator[NewsItem]:
        """Stream real-time news updates"""
        
    async def search_news(self, query: str, start_date: datetime, 
                         end_date: datetime) -> List[NewsItem]:
        """Search historical news"""
```

### 2.2 Alpha Vantage Integration

#### Configuration
```python
class AlphaVantageConfig:
    """Configuration for Alpha Vantage API"""
    api_key: str
    base_url: str = "https://www.alphavantage.co/query"
    timeout: int = 30
    max_requests_per_minute: int = 5  # Free tier limit
    news_sentiment_endpoint: str = "NEWS_SENTIMENT"
    time_series_endpoint: str = "TIME_SERIES_INTRADAY"
```

#### Implementation
```python
class AlphaVantageAdapter(NewsSource):
    """Alpha Vantage News & Sentiment API adapter"""
    
    async def get_news_sentiment(self, tickers: List[str], 
                               time_from: Optional[str] = None,
                               time_to: Optional[str] = None,
                               sort: str = "LATEST") -> NewsSentiment:
        """Get news sentiment for tickers"""
        
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": ",".join(tickers),
            "apikey": self.config.api_key
        }
        
        if time_from:
            params["time_from"] = time_from
        if time_to:
            params["time_to"] = time_to
            
        response = await self._get(params)
        return self._parse_sentiment_response(response)
```

### 2.3 NewsAPI Integration

#### Configuration
```python
class NewsAPIConfig:
    """Configuration for NewsAPI"""
    api_key: str
    base_url: str = "https://newsapi.org/v2"
    timeout: int = 30
    max_requests_per_day: int = 1000  # Developer plan limit
    language: str = "en"
    country: str = "us"
    sources: List[str] = [
        "bloomberg", "reuters", "financial-times", 
        "wall-street-journal", "cnbc", "marketwatch"
    ]
```

## 3. Data Models and Schemas

### 3.1 Order Models

```python
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class OrderStatus(Enum):
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

class TimeInForce(Enum):
    DAY = "day"
    GTC = "gtc"  # Good Till Cancelled
    IOC = "ioc"  # Immediate Or Cancel
    FOK = "fok"  # Fill Or Kill

@dataclass
class Order:
    """Order data model"""
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType
    time_in_force: TimeInForce = TimeInForce.DAY
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    order_id: Optional[str] = None
    created_at: Optional[datetime] = None
    metadata: Dict[str, Any] = None
```

### 3.2 Position Models

```python
@dataclass
class Position:
    """Position data model"""
    symbol: str
    quantity: float
    avg_cost: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float
    side: str  # "long" or "short"
    account_id: str
    last_updated: datetime
    
    @property
    def market_price(self) -> float:
        """Calculate current market price"""
        return self.market_value / abs(self.quantity) if self.quantity != 0 else 0.0
    
    @property
    def pnl_percentage(self) -> float:
        """Calculate P&L percentage"""
        cost_basis = abs(self.quantity) * self.avg_cost
        return (self.unrealized_pnl / cost_basis) * 100 if cost_basis != 0 else 0.0
```

### 3.3 Market Data Models

```python
@dataclass
class Quote:
    """Real-time quote data"""
    symbol: str
    bid: float
    ask: float
    bid_size: int
    ask_size: int
    last: float
    last_size: int
    volume: int
    timestamp: datetime
    
    @property
    def spread(self) -> float:
        """Calculate bid-ask spread"""
        return self.ask - self.bid
    
    @property
    def mid_price(self) -> float:
        """Calculate mid price"""
        return (self.bid + self.ask) / 2

@dataclass
class OHLCV:
    """OHLCV candle data"""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    
    @property
    def typical_price(self) -> float:
        """Calculate typical price (HLC/3)"""
        return (self.high + self.low + self.close) / 3
```

### 3.4 News Data Models

```python
@dataclass
class NewsItem:
    """Enhanced news item model"""
    id: str
    title: str
    content: str
    source: str
    author: Optional[str]
    timestamp: datetime
    url: str
    entities: List[str]
    symbols: List[str]  # Extracted stock symbols
    sentiment_score: Optional[float] = None
    impact_score: Optional[float] = None
    relevance_score: Optional[float] = None
    category: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class SentimentAnalysis:
    """Sentiment analysis result"""
    score: float  # -1.0 to 1.0
    magnitude: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    label: str  # "positive", "negative", "neutral"
    keywords: List[str]
    timestamp: datetime
```

## 4. Configuration Management

### 4.1 Environment Configuration

```yaml
# config/environments/development.yaml
environment: development
debug: true
log_level: DEBUG

database:
  url: postgresql://localhost:5432/ai_news_trader_dev
  pool_size: 10
  max_overflow: 20
  echo: true

redis:
  url: redis://localhost:6379/0
  max_connections: 10

brokers:
  interactive_brokers:
    enabled: true
    host: 127.0.0.1
    port: 7497  # Paper trading
    client_id: 1
    timeout: 10
  
  alpaca:
    enabled: true
    base_url: https://paper-api.alpaca.markets
    timeout: 30

news_sources:
  alpha_vantage:
    enabled: true
    max_requests_per_minute: 5
  
  newsapi:
    enabled: true
    max_requests_per_day: 1000
    language: en
    country: us

features:
  live_trading: false
  paper_trading: true
  real_time_news: true
  neural_forecasting: true
  gpu_acceleration: true
```

### 4.2 Broker Configuration Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "brokers": {
      "type": "object",
      "properties": {
        "interactive_brokers": {
          "type": "object",
          "properties": {
            "enabled": {"type": "boolean"},
            "host": {"type": "string"},
            "port": {"type": "integer", "minimum": 1, "maximum": 65535},
            "client_id": {"type": "integer"},
            "account_id": {"type": "string"},
            "timeout": {"type": "integer", "minimum": 1},
            "max_requests_per_second": {"type": "integer", "minimum": 1}
          },
          "required": ["enabled", "host", "port", "client_id"],
          "additionalProperties": false
        },
        "alpaca": {
          "type": "object",
          "properties": {
            "enabled": {"type": "boolean"},
            "api_key": {"type": "string"},
            "secret_key": {"type": "string"},
            "base_url": {"type": "string", "format": "uri"},
            "timeout": {"type": "integer", "minimum": 1}
          },
          "required": ["enabled", "api_key", "secret_key"],
          "additionalProperties": false
        }
      },
      "additionalProperties": false
    }
  },
  "required": ["brokers"],
  "additionalProperties": false
}
```

## 5. Event System Specifications

### 5.1 Event Types

```python
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

class EventType(Enum):
    MARKET_DATA = "market_data"
    NEWS_ITEM = "news_item"
    ORDER_UPDATE = "order_update"
    POSITION_UPDATE = "position_update"
    ACCOUNT_UPDATE = "account_update"
    ALERT = "alert"
    SYSTEM_EVENT = "system_event"

@dataclass
class Event:
    """Base event class"""
    event_type: EventType
    timestamp: datetime
    source: str
    data: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class MarketDataEvent(Event):
    """Market data event"""
    symbol: str
    price: float
    volume: int
    bid: Optional[float] = None
    ask: Optional[float] = None
    
    def __post_init__(self):
        super().__post_init__()
        self.event_type = EventType.MARKET_DATA
        self.data.update({
            "symbol": self.symbol,
            "price": self.price,
            "volume": self.volume,
            "bid": self.bid,
            "ask": self.ask
        })
```

### 5.2 Event Bus Implementation

```python
import asyncio
from typing import Callable, Dict, List, Any
from collections import defaultdict

class EventBus:
    """Asynchronous event bus for real-time data distribution"""
    
    def __init__(self):
        self._subscribers: Dict[EventType, List[Callable]] = defaultdict(list)
        self._middleware: List[Callable] = []
        self._running = False
        self._queue = asyncio.Queue()
        
    def subscribe(self, event_type: EventType, handler: Callable):
        """Subscribe to specific event type"""
        self._subscribers[event_type].append(handler)
        
    def unsubscribe(self, event_type: EventType, handler: Callable):
        """Unsubscribe from event type"""
        if handler in self._subscribers[event_type]:
            self._subscribers[event_type].remove(handler)
    
    def add_middleware(self, middleware: Callable):
        """Add middleware for event processing"""
        self._middleware.append(middleware)
        
    async def publish(self, event: Event):
        """Publish event to all subscribers"""
        await self._queue.put(event)
        
    async def start(self):
        """Start event processing loop"""
        self._running = True
        while self._running:
            try:
                event = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                await self._process_event(event)
            except asyncio.TimeoutError:
                continue
                
    async def stop(self):
        """Stop event processing"""
        self._running = False
        
    async def _process_event(self, event: Event):
        """Process event through middleware and subscribers"""
        # Apply middleware
        for middleware in self._middleware:
            event = await middleware(event)
            if event is None:
                return  # Event was filtered out
        
        # Notify subscribers
        subscribers = self._subscribers[event.event_type]
        if subscribers:
            tasks = [handler(event) for handler in subscribers]
            await asyncio.gather(*tasks, return_exceptions=True)
```

## 6. Database Schema

### 6.1 SQLAlchemy Models

```python
from sqlalchemy import Column, String, Numeric, DateTime, Boolean, Text, JSON, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from datetime import datetime

Base = declarative_base()

class Account(Base):
    """Account information table"""
    __tablename__ = "accounts"
    
    account_id = Column(String, primary_key=True)
    broker_name = Column(String, nullable=False)
    account_type = Column(String, nullable=False)  # "paper", "live"
    balance = Column(Numeric(15, 2), nullable=False)
    buying_power = Column(Numeric(15, 2), nullable=False)
    day_trading_buying_power = Column(Numeric(15, 2))
    maintenance_margin = Column(Numeric(15, 2))
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

class Position(Base):
    """Positions table"""
    __tablename__ = "positions"
    
    position_id = Column(String, primary_key=True)
    account_id = Column(String, nullable=False)
    symbol = Column(String, nullable=False)
    quantity = Column(Numeric(15, 4), nullable=False)
    avg_cost = Column(Numeric(15, 4), nullable=False)
    market_value = Column(Numeric(15, 2), nullable=False)
    unrealized_pnl = Column(Numeric(15, 2), nullable=False)
    realized_pnl = Column(Numeric(15, 2), nullable=False)
    side = Column(String, nullable=False)  # "long", "short"
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

class Order(Base):
    """Orders table"""
    __tablename__ = "orders"
    
    order_id = Column(String, primary_key=True)
    account_id = Column(String, nullable=False)
    symbol = Column(String, nullable=False)
    side = Column(String, nullable=False)  # "buy", "sell"
    quantity = Column(Numeric(15, 4), nullable=False)
    order_type = Column(String, nullable=False)
    time_in_force = Column(String, nullable=False)
    limit_price = Column(Numeric(15, 4))
    stop_price = Column(Numeric(15, 4))
    status = Column(String, nullable=False)
    filled_quantity = Column(Numeric(15, 4), default=0)
    avg_fill_price = Column(Numeric(15, 4))
    commission = Column(Numeric(10, 4))
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

class NewsItem(Base):
    """News items table"""
    __tablename__ = "news_items"
    
    news_id = Column(String, primary_key=True)
    title = Column(Text, nullable=False)
    content = Column(Text, nullable=False)
    source = Column(String, nullable=False)
    author = Column(String)
    url = Column(Text, nullable=False)
    entities = Column(JSON)
    symbols = Column(JSON)
    sentiment_score = Column(Numeric(5, 4))
    impact_score = Column(Numeric(5, 4))
    relevance_score = Column(Numeric(5, 4))
    category = Column(String)
    published_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=func.now())
```

### 6.2 Time Series Data Schema

```sql
-- Market data time series (using TimescaleDB)
CREATE TABLE market_data (
    time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    price NUMERIC(15, 4) NOT NULL,
    volume BIGINT NOT NULL,
    bid NUMERIC(15, 4),
    ask NUMERIC(15, 4),
    bid_size INTEGER,
    ask_size INTEGER
);

-- Create hypertable for time series data
SELECT create_hypertable('market_data', 'time');

-- Create indexes
CREATE INDEX idx_market_data_symbol_time ON market_data (symbol, time DESC);
CREATE INDEX idx_market_data_time ON market_data (time DESC);

-- Predictions table
CREATE TABLE predictions (
    prediction_id UUID PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    model_name VARCHAR(50) NOT NULL,
    predicted_price NUMERIC(15, 4) NOT NULL,
    confidence NUMERIC(5, 4) NOT NULL,
    horizon_days INTEGER NOT NULL,
    prediction_time TIMESTAMPTZ NOT NULL,
    target_time TIMESTAMPTZ NOT NULL,
    actual_price NUMERIC(15, 4),
    accuracy NUMERIC(5, 4),
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

## 7. API Specifications

### 7.1 MCP Tool Enhancements

```python
# Enhanced MCP tool signatures
@mcp_tool
async def execute_trade(
    strategy: str,
    symbol: str,
    action: str,  # "buy" or "sell"
    quantity: int,
    order_type: str = "market",
    limit_price: Optional[float] = None,
    broker: Optional[str] = None,  # NEW: Specify broker
    dry_run: bool = False  # NEW: Dry run mode
) -> Dict[str, Any]:
    """Execute live trade with enhanced parameters"""

@mcp_tool
async def get_real_time_quotes(
    symbols: List[str],
    fields: List[str] = ["bid", "ask", "last", "volume"]  # NEW: Customizable fields
) -> Dict[str, Any]:
    """Get real-time market quotes"""

@mcp_tool
async def stream_market_data(
    symbols: List[str],
    duration_minutes: int = 60,
    callback_url: Optional[str] = None  # NEW: Webhook callback
) -> Dict[str, Any]:
    """Stream real-time market data"""

@mcp_tool
async def analyze_news_impact(
    symbol: str,
    lookback_hours: int = 24,
    impact_threshold: float = 0.5,  # NEW: Impact scoring threshold
    correlation_analysis: bool = True  # NEW: Price correlation analysis
) -> Dict[str, Any]:
    """Analyze news impact on stock price"""
```

### 7.2 WebSocket API Specifications

```python
# WebSocket message formats
class WSMessageType(Enum):
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    MARKET_DATA = "market_data"
    NEWS_UPDATE = "news_update"
    ORDER_UPDATE = "order_update"
    ERROR = "error"

@dataclass
class WSMessage:
    """WebSocket message format"""
    type: WSMessageType
    data: Dict[str, Any]
    timestamp: datetime
    request_id: Optional[str] = None

# Market data subscription
{
    "type": "subscribe",
    "data": {
        "channel": "market_data",
        "symbols": ["AAPL", "GOOGL", "MSFT"],
        "fields": ["bid", "ask", "last", "volume"]
    },
    "request_id": "req_123"
}

# Market data update
{
    "type": "market_data",
    "data": {
        "symbol": "AAPL",
        "bid": 150.25,
        "ask": 150.27,
        "last": 150.26,
        "volume": 1000,
        "timestamp": "2024-01-15T10:30:00Z"
    },
    "timestamp": "2024-01-15T10:30:00.123Z"
}
```

## 8. Testing Specifications

### 8.1 Unit Test Structure

```python
import pytest
from unittest.mock import AsyncMock, Mock
from src.brokers.alpaca import AlpacaAdapter
from src.models.order import Order, OrderSide, OrderType

class TestAlpacaAdapter:
    """Unit tests for Alpaca adapter"""
    
    @pytest.fixture
    def alpaca_adapter(self):
        """Create Alpaca adapter instance for testing"""
        config = Mock()
        config.api_key = "test_key"
        config.secret_key = "test_secret"
        config.base_url = "https://paper-api.alpaca.markets"
        return AlpacaAdapter(config)
    
    @pytest.mark.asyncio
    async def test_place_order_success(self, alpaca_adapter):
        """Test successful order placement"""
        # Mock HTTP response
        alpaca_adapter._post = AsyncMock(return_value={
            "id": "order_123",
            "status": "accepted",
            "filled_qty": "0",
            "filled_avg_price": None
        })
        
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=10,
            order_type=OrderType.MARKET
        )
        
        result = await alpaca_adapter.place_order(order)
        
        assert result.order_id == "order_123"
        assert result.status == "accepted"
        assert result.filled_qty == 0.0
```

### 8.2 Integration Test Structure

```python
import pytest
from src.brokers.factory import BrokerFactory
from src.config.config_manager import ConfigManager

class TestBrokerIntegration:
    """Integration tests for broker functionality"""
    
    @pytest.fixture
    async def broker_manager(self):
        """Create broker manager with test configuration"""
        config = ConfigManager(environment="test")
        factory = BrokerFactory(config)
        return await factory.create_broker_manager()
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_end_to_end_trading_flow(self, broker_manager):
        """Test complete trading flow from order to position"""
        # Place order
        order_result = await broker_manager.place_order(
            symbol="AAPL",
            side="buy",
            quantity=1,
            order_type="market"
        )
        
        # Wait for fill
        await broker_manager.wait_for_fill(order_result.order_id, timeout=30)
        
        # Verify position
        positions = await broker_manager.get_positions()
        aapl_position = next((p for p in positions if p.symbol == "AAPL"), None)
        
        assert aapl_position is not None
        assert aapl_position.quantity == 1
```

## 9. Deployment Specifications

### 9.1 Docker Configuration

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ src/
COPY config/ config/

# Create non-root user
RUN useradd -m -u 1000 trader
USER trader

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8080/health')"

# Start application
CMD ["python", "src/mcp/mcp_server_enhanced.py"]
```

### 9.2 Kubernetes Deployment

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-news-trader
  labels:
    app: ai-news-trader
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ai-news-trader
  template:
    metadata:
      labels:
        app: ai-news-trader
    spec:
      containers:
      - name: ai-news-trader
        image: ai-news-trader:latest
        ports:
        - containerPort: 8080
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: ai-news-trader-secrets
              key: database-url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: ai-news-trader-secrets
              key: redis-url
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
```

This technical specification provides the detailed implementation requirements for integrating broker and news APIs with the AI News Trader system, covering all aspects from API interfaces to deployment configurations.