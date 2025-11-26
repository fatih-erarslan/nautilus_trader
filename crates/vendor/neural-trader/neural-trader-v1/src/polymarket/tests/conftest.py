"""
Pytest configuration for Polymarket tests

This module provides fixtures and configuration for testing the Polymarket
integration module.
"""

import asyncio
import os
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, Any, List
import pytest
from unittest.mock import AsyncMock, MagicMock

# Import models and clients for test fixtures
from ..models import Market, MarketStatus, Order, OrderSide, OrderStatus, OrderType
from ..api import PolymarketClient, CLOBClient
# from ..api import GammaClient  # TODO: Implement GammaClient
from ..utils import PolymarketConfig


@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_config():
    """Mock Polymarket configuration for testing"""
    return PolymarketConfig(
        api_key="test_api_key",
        private_key="test_private_key",
        clob_url="https://test-clob.polymarket.com",
        gamma_url="https://test-gamma.polymarket.com",
        ws_url="wss://test-ws.polymarket.com",
        environment="testing",
        debug=True,
        timeout=10,
        rate_limit=1000,
    )


@pytest.fixture
def sample_market():
    """Sample market for testing"""
    return Market(
        id="test-market-1",
        question="Will Bitcoin reach $100,000 by end of 2024?",
        outcomes=["Yes", "No"],
        end_date=datetime.now() + timedelta(days=30),
        status=MarketStatus.ACTIVE,
        current_prices={"Yes": Decimal("0.65"), "No": Decimal("0.35")},
        created_at=datetime.now() - timedelta(days=1),
        updated_at=datetime.now(),
    )


@pytest.fixture
def sample_markets():
    """Multiple sample markets for testing"""
    base_time = datetime.now()
    return [
        Market(
            id=f"test-market-{i}",
            question=f"Test market question {i}",
            outcomes=["Yes", "No"],
            end_date=base_time + timedelta(days=30 + i),
            status=MarketStatus.ACTIVE,
            current_prices={"Yes": Decimal(f"0.{50 + i}"), "No": Decimal(f"0.{50 - i}")},
            created_at=base_time - timedelta(days=i),
            updated_at=base_time,
        )
        for i in range(1, 6)
    ]


@pytest.fixture
def sample_order():
    """Sample order for testing"""
    return Order(
        id="test-order-1",
        market_id="test-market-1",
        outcome="Yes",
        side=OrderSide.BUY,
        price=Decimal("0.60"),
        size=Decimal("10.0"),
        status=OrderStatus.OPEN,
        created_at=datetime.now(),
    )


@pytest.fixture
def mock_clob_client(mock_config):
    """Mock CLOB client for testing"""
    client = AsyncMock(spec=CLOBClient)
    client.config = mock_config
    client.get_markets = AsyncMock(return_value=[])
    client.get_market = AsyncMock(return_value=None)
    client.place_order = AsyncMock(return_value=None)
    client.cancel_order = AsyncMock(return_value=True)
    client.get_orders = AsyncMock(return_value=[])
    client.health_check = AsyncMock(return_value=True)
    return client


# @pytest.fixture
# def mock_gamma_client(mock_config):
#     """Mock Gamma client for testing"""
#     client = AsyncMock(spec=GammaClient)
#     client.config = mock_config
#     client.get_events = AsyncMock(return_value=[])
#     client.get_market_data = AsyncMock(return_value={})
#     client.health_check = AsyncMock(return_value=True)
#     return client


@pytest.fixture
def mock_strategy(mock_clob_client):
    """Mock strategy for testing"""
    strategy = AsyncMock()
    strategy.client = mock_clob_client
    strategy.name = "TestStrategy"
    strategy.analyze_market = AsyncMock(return_value=None)
    strategy.should_trade_market = AsyncMock(return_value=True)
    return strategy


@pytest.fixture
def mock_api_response():
    """Mock API response data"""
    return {
        "status": "success",
        "data": {
            "markets": [
                {
                    "id": "test-market-1",
                    "question": "Test market question",
                    "outcomes": ["Yes", "No"],
                    "end_date": "2024-12-31T23:59:59Z",
                    "status": "active",
                    "current_prices": {"Yes": 0.65, "No": 0.35}
                }
            ]
        },
        "timestamp": datetime.now().isoformat()
    }


@pytest.fixture
def mock_environment_variables():
    """Mock environment variables for testing"""
    env_vars = {
        "POLYMARKET_API_KEY": "test_api_key",
        "POLYMARKET_PRIVATE_KEY": "test_private_key",
        "POLYMARKET_ENVIRONMENT": "testing",
        "POLYMARKET_DEBUG": "true",
    }
    
    # Set environment variables
    for key, value in env_vars.items():
        os.environ[key] = value
    
    yield env_vars
    
    # Clean up environment variables
    for key in env_vars:
        if key in os.environ:
            del os.environ[key]


# Async test utilities

@pytest.fixture
async def async_mock_clob_client(mock_config):
    """Async mock CLOB client for testing"""
    client = AsyncMock(spec=CLOBClient)
    client.config = mock_config
    return client


# Enhanced fixture collections
# TODO: Implement these fixture classes when needed


# Enhanced mock fixtures

@pytest.fixture
def mock_aiohttp_session():
    """Mock aiohttp session for HTTP client testing"""
    from aioresponses import aioresponses
    with aioresponses() as mock:
        yield mock


@pytest.fixture
def mock_websocket_connection():
    """Mock WebSocket connection"""
    mock_ws = AsyncMock()
    mock_ws.send_str = AsyncMock()
    mock_ws.receive_str = AsyncMock()
    mock_ws.close = AsyncMock()
    mock_ws.closed = False
    return mock_ws


@pytest.fixture
def mock_rate_limiter():
    """Mock rate limiter for API testing"""
    mock_limiter = AsyncMock()
    mock_limiter.acquire = AsyncMock()
    mock_limiter.release = AsyncMock()
    mock_limiter.wait_if_needed = AsyncMock()
    return mock_limiter


@pytest.fixture
async def mock_polymarket_client(mock_config):
    """Mock complete Polymarket client"""
    client = AsyncMock(spec=PolymarketClient)
    client.config = mock_config
    
    # Mock all client methods
    client.get_markets = AsyncMock(return_value=[])
    client.get_market = AsyncMock(return_value=None)
    client.get_order_book = AsyncMock(return_value=None)
    client.place_order = AsyncMock(return_value=None)
    client.cancel_order = AsyncMock(return_value=True)
    client.get_orders = AsyncMock(return_value=[])
    client.get_positions = AsyncMock(return_value=[])
    client.get_portfolio = AsyncMock(return_value=None)
    client.connect_websocket = AsyncMock()
    client.disconnect_websocket = AsyncMock()
    client.subscribe_to_market = AsyncMock()
    client.health_check = AsyncMock(return_value=True)
    
    return client


# TODO: Implement fixture classes and then uncomment these fixtures


# Async test fixtures

@pytest.fixture
async def async_test_client():
    """Async HTTP test client"""
    import aiohttp
    async with aiohttp.ClientSession() as session:
        yield session


# Database/Storage fixtures (if needed)

@pytest.fixture
def mock_redis():
    """Mock Redis client for caching tests"""
    import fakeredis
    return fakeredis.FakeStrictRedis()


@pytest.fixture
def temporary_file():
    """Temporary file for testing file operations"""
    import tempfile
    import os
    
    fd, path = tempfile.mkstemp()
    yield path
    os.close(fd)
    os.unlink(path)


# TODO: Implement performance testing fixtures when test data generator is available


# Test data helpers

def create_test_market(
    market_id: str = "test-market",
    question: str = "Test question?",
    outcomes: List[str] = None,
    status: MarketStatus = MarketStatus.ACTIVE,
    prices: Dict[str, float] = None,
) -> Market:
    """Helper to create test market with custom parameters"""
    if outcomes is None:
        outcomes = ["Yes", "No"]
    if prices is None:
        prices = {"Yes": 0.6, "No": 0.4}
    
    return Market(
        id=market_id,
        question=question,
        outcomes=outcomes,
        end_date=datetime.now() + timedelta(days=30),
        status=status,
        current_prices={k: Decimal(str(v)) for k, v in prices.items()},
        created_at=datetime.now() - timedelta(hours=1),
        updated_at=datetime.now(),
    )


def create_test_order(
    order_id: str = "test-order",
    market_id: str = "test-market",
    outcome: str = "Yes",
    side: OrderSide = OrderSide.BUY,
    price: float = 0.6,
    size: float = 10.0,
    status: OrderStatus = OrderStatus.OPEN,
) -> Order:
    """Helper to create test order with custom parameters"""
    return Order(
        id=order_id,
        market_id=market_id,
        outcome_id=outcome,
        side=side,
        type=OrderType.LIMIT,
        size=size,
        price=price,
        status=status,
        created_at=datetime.now(),
    )


# Pytest marks for test categorization

pytest_plugins = []

# Custom markers
def pytest_configure(config):
    """Configure custom pytest markers"""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "api: mark test as requiring API access"
    )
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU acceleration"
    )