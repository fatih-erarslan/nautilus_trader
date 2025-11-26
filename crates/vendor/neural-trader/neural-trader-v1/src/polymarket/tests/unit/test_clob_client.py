"""
Comprehensive unit tests for the Polymarket CLOB API client

This module implements Test-Driven Development (TDD) by defining the expected 
behavior of the CLOB client through comprehensive test cases BEFORE implementation.

Test Categories:
- Market data retrieval (markets, market details, order books)
- Order management (place, cancel, get orders)  
- Trade history and portfolio
- Authentication and request signing
- Error handling and rate limiting
- Async operations and caching
"""

import asyncio
import json
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Any
from unittest.mock import AsyncMock, Mock, patch

import pytest
import aiohttp
from aioresponses import aioresponses

from polymarket.api.clob_client import CLOBClient
from polymarket.api.base import PolymarketAPIError, RateLimitError, AuthenticationError, ValidationError
from polymarket.models import Market, Order, OrderBook, OrderSide, OrderStatus, OrderType, MarketStatus
from polymarket.utils import PolymarketConfig


class TestCLOBClientInitialization:
    """Test CLOB client initialization and configuration"""
    
    def test_clob_client_init_default_config(self):
        """Test client initialization with default configuration"""
        # Use testing environment to avoid API key requirements
        test_config = PolymarketConfig(environment="testing")
        client = CLOBClient(config=test_config)
        
        assert client.config is not None
        assert isinstance(client.config, PolymarketConfig)
        assert client.session is None
        assert client._cache is not None
        assert client.max_retries == 3
        assert client.retry_delay == 1.0
    
    def test_clob_client_init_custom_config(self, mock_config):
        """Test client initialization with custom configuration"""
        client = CLOBClient(config=mock_config, max_retries=5, retry_delay=2.0)
        
        assert client.config == mock_config
        assert client.max_retries == 5
        assert client.retry_delay == 2.0
    
    def test_clob_client_base_url(self):
        """Test CLOB client base URL configuration"""
        test_config = PolymarketConfig(environment="testing")
        client = CLOBClient(config=test_config)
        
        # This should return the CLOB API base URL
        expected_url = client.config.clob_url or "https://clob.polymarket.com"
        assert client._get_base_url() == expected_url


class TestCLOBClientMarketRetrieval:
    """Test market data retrieval methods"""
    
    @pytest.mark.asyncio
    async def test_get_markets_success(self, mock_config):
        """Test successful retrieval of all markets"""
        client = CLOBClient(config=mock_config)
        
        # Mock API response
        mock_response = {
            "markets": [
                {
                    "id": "market-1",
                    "question": "Will Bitcoin reach $100k by end of 2024?",
                    "outcomes": ["Yes", "No"],
                    "end_date": "2024-12-31T23:59:59Z",
                    "status": "active",
                    "current_prices": {"Yes": 0.65, "No": 0.35},
                    "created_at": "2024-01-01T00:00:00Z",
                    "updated_at": "2024-06-01T12:00:00Z"
                },
                {
                    "id": "market-2", 
                    "question": "Will Ethereum reach $5k by end of 2024?",
                    "outcomes": ["Yes", "No"],
                    "end_date": "2024-12-31T23:59:59Z",
                    "status": "active",
                    "current_prices": {"Yes": 0.45, "No": 0.55},
                    "created_at": "2024-01-01T00:00:00Z",
                    "updated_at": "2024-06-01T12:00:00Z"
                }
            ]
        }
        
        with aioresponses() as m:
            m.get(
                f"{client._get_base_url()}/markets",
                payload=mock_response,
                status=200
            )
            
            markets = await client.get_markets()
            
            assert len(markets) == 2
            assert all(isinstance(market, Market) for market in markets)
            assert markets[0].id == "market-1"
            assert markets[0].question == "Will Bitcoin reach $100k by end of 2024?"
            assert markets[1].id == "market-2"
    
    @pytest.mark.asyncio
    async def test_get_markets_with_filters(self, mock_config):
        """Test market retrieval with filtering parameters"""
        client = CLOBClient(config=mock_config)
        
        mock_response = {"markets": []}
        
        with aioresponses() as m:
            m.get(
                f"{client._get_base_url()}/markets",
                payload=mock_response,
                status=200
            )
            
            await client.get_markets(
                limit=10,
                offset=0,
                status="active",
                category="crypto"
            )
            
            # Verify the request was made with correct parameters
            call = m.requests[('GET', f"{client._get_base_url()}/markets")][0]
            assert 'limit=10' in str(call.url)
            assert 'offset=0' in str(call.url)
            assert 'status=active' in str(call.url)
            assert 'category=crypto' in str(call.url)
    
    @pytest.mark.asyncio
    async def test_get_market_by_id_success(self, mock_config):
        """Test successful retrieval of specific market"""
        client = CLOBClient(config=mock_config)
        market_id = "market-123"
        
        mock_response = {
            "market": {
                "id": market_id,
                "question": "Test market question?",
                "outcomes": ["Yes", "No"],
                "end_date": "2024-12-31T23:59:59Z",
                "status": "active",
                "current_prices": {"Yes": 0.60, "No": 0.40},
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-06-01T12:00:00Z"
            }
        }
        
        with aioresponses() as m:
            m.get(
                f"{client._get_base_url()}/markets/{market_id}",
                payload=mock_response,
                status=200
            )
            
            market = await client.get_market_by_id(market_id)
            
            assert isinstance(market, Market)
            assert market.id == market_id
            assert market.question == "Test market question?"
            assert market.status == MarketStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_get_market_by_id_not_found(self, mock_config):
        """Test market retrieval for non-existent market"""
        client = CLOBClient(config=mock_config)
        market_id = "nonexistent-market"
        
        with aioresponses() as m:
            m.get(
                f"{client._get_base_url()}/markets/{market_id}",
                status=404,
                payload={"error": "Market not found"}
            )
            
            with pytest.raises(PolymarketAPIError) as exc_info:
                await client.get_market_by_id(market_id)
            
            assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_get_order_book_success(self, mock_config):
        """Test successful order book retrieval"""
        client = CLOBClient(config=mock_config)
        market_id = "market-123"
        outcome_id = "outcome-1"
        
        mock_response = {
            "order_book": {
                "market_id": market_id,
                "outcome_id": outcome_id,
                "bids": [
                    {"price": 0.60, "size": 100.0},
                    {"price": 0.59, "size": 200.0}
                ],
                "asks": [
                    {"price": 0.65, "size": 150.0},
                    {"price": 0.66, "size": 250.0}
                ],
                "timestamp": "2024-06-01T12:00:00Z"
            }
        }
        
        with aioresponses() as m:
            m.get(
                f"{client._get_base_url()}/book",
                payload=mock_response,
                status=200
            )
            
            order_book = await client.get_order_book(market_id, outcome_id)
            
            assert isinstance(order_book, OrderBook)
            assert order_book.market_id == market_id
            assert order_book.outcome_id == outcome_id
            assert len(order_book.bids) == 2
            assert len(order_book.asks) == 2


class TestCLOBClientOrderManagement:
    """Test order placement, cancellation, and retrieval"""
    
    @pytest.mark.asyncio
    async def test_place_order_success(self, mock_config):
        """Test successful order placement"""
        client = CLOBClient(config=mock_config)
        
        # Mock authentication
        with patch.object(client, '_sign_order_request') as mock_sign:
            mock_sign.return_value = {"signature": "test_signature"}
            
            order_data = {
                "market_id": "market-123",
                "outcome_id": "outcome-1",
                "side": "buy",
                "type": "limit",
                "size": 10.0,
                "price": 0.60
            }
            
            mock_response = {
                "order": {
                    "id": "order-456",
                    "market_id": "market-123",
                    "outcome_id": "outcome-1",
                    "side": "buy",
                    "type": "limit",
                    "size": 10.0,
                    "price": 0.60,
                    "status": "open",
                    "created_at": "2024-06-01T12:00:00Z"
                }
            }
            
            with aioresponses() as m:
                m.post(
                    f"{client._get_base_url()}/orders",
                    payload=mock_response,
                    status=201
                )
                
                order = await client.place_order(**order_data)
                
                assert isinstance(order, Order)
                assert order.id == "order-456"
                assert order.market_id == "market-123"
                assert order.side == OrderSide.BUY
                assert order.type == OrderType.LIMIT
                assert order.status == OrderStatus.OPEN
    
    @pytest.mark.asyncio
    async def test_place_order_validation_error(self, mock_config):
        """Test order placement with validation error"""
        client = CLOBClient(config=mock_config)
        
        with aioresponses() as m:
            m.post(
                f"{client._get_base_url()}/orders",
                status=400,
                payload={"error": "Invalid order size"}
            )
            
            with pytest.raises(ValidationError):
                await client.place_order(
                    market_id="market-123",
                    outcome_id="outcome-1",
                    side="buy",
                    type="limit",
                    size=-10.0,  # Invalid negative size
                    price=0.60
                )
    
    @pytest.mark.asyncio
    async def test_cancel_order_success(self, mock_config):
        """Test successful order cancellation"""
        client = CLOBClient(config=mock_config)
        order_id = "order-456"
        
        with aioresponses() as m:
            m.delete(
                f"{client._get_base_url()}/orders/{order_id}",
                payload={"cancelled": True},
                status=200
            )
            
            result = await client.cancel_order(order_id)
            
            assert result is True
    
    @pytest.mark.asyncio
    async def test_cancel_order_not_found(self, mock_config):
        """Test cancellation of non-existent order"""
        client = CLOBClient(config=mock_config)
        order_id = "nonexistent-order"
        
        with aioresponses() as m:
            m.delete(
                f"{client._get_base_url()}/orders/{order_id}",
                status=404,
                payload={"error": "Order not found"}
            )
            
            with pytest.raises(PolymarketAPIError) as exc_info:
                await client.cancel_order(order_id)
            
            assert exc_info.value.status_code == 404
    
    @pytest.mark.asyncio
    async def test_get_orders_success(self, mock_config):
        """Test successful retrieval of user orders"""
        client = CLOBClient(config=mock_config)
        
        mock_response = {
            "orders": [
                {
                    "id": "order-1",
                    "market_id": "market-123",
                    "outcome_id": "outcome-1",
                    "side": "buy",
                    "type": "limit",
                    "size": 10.0,
                    "price": 0.60,
                    "status": "open",
                    "created_at": "2024-06-01T12:00:00Z"
                },
                {
                    "id": "order-2",
                    "market_id": "market-124",
                    "outcome_id": "outcome-2",
                    "side": "sell",
                    "type": "limit",
                    "size": 5.0,
                    "price": 0.70,
                    "status": "filled",
                    "created_at": "2024-06-01T11:00:00Z"
                }
            ]
        }
        
        with aioresponses() as m:
            m.get(
                f"{client._get_base_url()}/orders",
                payload=mock_response,
                status=200
            )
            
            orders = await client.get_orders()
            
            assert len(orders) == 2
            assert all(isinstance(order, Order) for order in orders)
            assert orders[0].id == "order-1"
            assert orders[0].status == OrderStatus.OPEN
            assert orders[1].id == "order-2"
            assert orders[1].status == OrderStatus.FILLED
    
    @pytest.mark.asyncio
    async def test_get_orders_with_filters(self, mock_config):
        """Test order retrieval with filtering parameters"""
        client = CLOBClient(config=mock_config)
        
        mock_response = {"orders": []}
        
        with aioresponses() as m:
            m.get(
                f"{client._get_base_url()}/orders",
                payload=mock_response,
                status=200
            )
            
            await client.get_orders(
                market_id="market-123",
                status="open",
                limit=50
            )
            
            # Verify request parameters
            call = m.requests[('GET', f"{client._get_base_url()}/orders")][0]
            assert 'market_id=market-123' in str(call.url)
            assert 'status=open' in str(call.url)
            assert 'limit=50' in str(call.url)


class TestCLOBClientTradeHistory:
    """Test trade history and portfolio retrieval"""
    
    @pytest.mark.asyncio
    async def test_get_trades_success(self, mock_config):
        """Test successful trade history retrieval"""
        client = CLOBClient(config=mock_config)
        
        mock_response = {
            "trades": [
                {
                    "id": "trade-1",
                    "market_id": "market-123",
                    "outcome_id": "outcome-1",
                    "side": "buy",
                    "price": 0.60,
                    "size": 10.0,
                    "timestamp": "2024-06-01T12:00:00Z",
                    "fee": 0.12
                },
                {
                    "id": "trade-2",
                    "market_id": "market-124",
                    "outcome_id": "outcome-2",
                    "side": "sell",
                    "price": 0.70,
                    "size": 5.0,
                    "timestamp": "2024-06-01T11:00:00Z",
                    "fee": 0.07
                }
            ]
        }
        
        with aioresponses() as m:
            m.get(
                f"{client._get_base_url()}/trades",
                payload=mock_response,
                status=200
            )
            
            trades = await client.get_trades()
            
            assert len(trades) == 2
            assert trades[0]["id"] == "trade-1"
            assert trades[0]["price"] == 0.60
            assert trades[1]["id"] == "trade-2"
    
    @pytest.mark.asyncio
    async def test_get_trades_with_pagination(self, mock_config):
        """Test trade retrieval with pagination"""
        client = CLOBClient(config=mock_config)
        
        mock_response = {"trades": []}
        
        with aioresponses() as m:
            m.get(
                f"{client._get_base_url()}/trades",
                payload=mock_response,
                status=200
            )
            
            await client.get_trades(
                limit=100,
                offset=50,
                start_date="2024-01-01",
                end_date="2024-06-01"
            )
            
            # Verify pagination parameters
            call = m.requests[('GET', f"{client._get_base_url()}/trades")][0]
            assert 'limit=100' in str(call.url)
            assert 'offset=50' in str(call.url)
            assert 'start_date=2024-01-01' in str(call.url)
            assert 'end_date=2024-06-01' in str(call.url)


class TestCLOBClientAuthentication:
    """Test authentication and request signing"""
    
    @pytest.mark.asyncio
    async def test_authenticate_success(self, mock_config):
        """Test successful authentication"""
        client = CLOBClient(config=mock_config)
        
        with patch('polymarket.utils.authenticate') as mock_auth:
            mock_auth.return_value = {
                'Authorization': 'Bearer test_token',
                'X-Signature': 'test_signature'
            }
            
            result = await client.authenticate()
            
            assert result is True
            assert client._auth_headers is not None
            assert 'Authorization' in client._auth_headers
    
    @pytest.mark.asyncio
    async def test_authenticate_failure(self, mock_config):
        """Test authentication failure"""
        client = CLOBClient(config=mock_config)
        
        with patch('polymarket.utils.authenticate') as mock_auth:
            mock_auth.side_effect = Exception("Invalid credentials")
            
            with pytest.raises(AuthenticationError):
                await client.authenticate()
    
    def test_sign_order_request(self, mock_config):
        """Test order request signing"""
        client = CLOBClient(config=mock_config)
        
        order_data = {
            "market_id": "market-123",
            "outcome_id": "outcome-1",
            "side": "buy",
            "size": 10.0,
            "price": 0.60
        }
        
        with patch('polymarket.utils.sign_order') as mock_sign:
            mock_sign.return_value = "test_signature"
            
            signature = client._sign_order_request(order_data)
            
            assert signature == "test_signature"
            mock_sign.assert_called_once()


class TestCLOBClientErrorHandling:
    """Test error handling and edge cases"""
    
    @pytest.mark.asyncio
    async def test_rate_limit_error_handling(self, mock_config):
        """Test rate limit error handling with retry"""
        client = CLOBClient(config=mock_config, max_retries=2, retry_delay=0.1)
        
        with aioresponses() as m:
            # First request hits rate limit
            m.get(
                f"{client._get_base_url()}/markets",
                status=429,
                headers={'Retry-After': '1'},
                payload={"error": "Rate limit exceeded"}
            )
            # Second request succeeds
            m.get(
                f"{client._get_base_url()}/markets",
                payload={"markets": []},
                status=200
            )
            
            # Should succeed after retry
            markets = await client.get_markets()
            assert markets == []
    
    @pytest.mark.asyncio
    async def test_rate_limit_exhausted_retries(self, mock_config):
        """Test rate limit with exhausted retries"""
        client = CLOBClient(config=mock_config, max_retries=1, retry_delay=0.1)
        
        with aioresponses() as m:
            # All requests hit rate limit
            for _ in range(3):
                m.get(
                    f"{client._get_base_url()}/markets",
                    status=429,
                    headers={'Retry-After': '1'},
                    payload={"error": "Rate limit exceeded"}
                )
            
            with pytest.raises(RateLimitError):
                await client.get_markets()
    
    @pytest.mark.asyncio
    async def test_server_error_with_retry(self, mock_config):
        """Test server error handling with retry"""
        client = CLOBClient(config=mock_config, max_retries=2, retry_delay=0.1)
        
        with aioresponses() as m:
            # First request fails with server error
            m.get(
                f"{client._get_base_url()}/markets",
                status=500,
                payload={"error": "Internal server error"}
            )
            # Second request succeeds
            m.get(
                f"{client._get_base_url()}/markets",
                payload={"markets": []},
                status=200
            )
            
            markets = await client.get_markets()
            assert markets == []
    
    @pytest.mark.asyncio
    async def test_network_error_handling(self, mock_config):
        """Test network error handling"""
        client = CLOBClient(config=mock_config)
        
        with aioresponses() as m:
            m.get(
                f"{client._get_base_url()}/markets",
                exception=aiohttp.ClientError("Connection failed")
            )
            
            with pytest.raises(PolymarketAPIError):
                await client.get_markets()
    
    @pytest.mark.asyncio
    async def test_invalid_json_response(self, mock_config):
        """Test handling of invalid JSON responses"""
        client = CLOBClient(config=mock_config)
        
        with aioresponses() as m:
            m.get(
                f"{client._get_base_url()}/markets",
                body="invalid json",
                status=200,
                content_type='application/json'
            )
            
            with pytest.raises(PolymarketAPIError):
                await client.get_markets()


class TestCLOBClientCaching:
    """Test response caching functionality"""
    
    @pytest.mark.asyncio
    async def test_cache_hit_for_get_requests(self, mock_config):
        """Test cache hit for GET requests"""
        client = CLOBClient(config=mock_config, cache_ttl=60)
        
        mock_response = {"markets": []}
        
        with aioresponses() as m:
            m.get(
                f"{client._get_base_url()}/markets",
                payload=mock_response,
                status=200
            )
            
            # First request - cache miss
            await client.get_markets()
            
            # Second request - cache hit (no HTTP request)
            await client.get_markets()
            
            # Should only have made one HTTP request
            assert len(m.requests) == 1
    
    @pytest.mark.asyncio
    async def test_cache_bypass_for_post_requests(self, mock_config):
        """Test cache bypass for POST requests"""
        client = CLOBClient(config=mock_config)
        
        with patch.object(client, '_sign_order_request') as mock_sign:
            mock_sign.return_value = {"signature": "test_signature"}
            
            mock_response = {"order": {"id": "order-123"}}
            
            with aioresponses() as m:
                for _ in range(2):
                    m.post(
                        f"{client._get_base_url()}/orders",
                        payload=mock_response,
                        status=201
                    )
                
                # Two identical POST requests
                await client.place_order(
                    market_id="market-123",
                    outcome_id="outcome-1",
                    side="buy",
                    type="limit",
                    size=10.0,
                    price=0.60
                )
                await client.place_order(
                    market_id="market-123",
                    outcome_id="outcome-1",
                    side="buy",
                    type="limit",
                    size=10.0,
                    price=0.60
                )
                
                # Should make both HTTP requests (no caching for POST)
                assert len(m.requests) == 2
    
    def test_cache_clear(self, mock_config):
        """Test cache clearing functionality"""
        client = CLOBClient(config=mock_config)
        
        # Add something to cache
        client._cache["test_key"] = "test_value"
        assert len(client._cache) == 1
        
        # Clear cache
        client.clear_cache()
        assert len(client._cache) == 0


class TestCLOBClientMetrics:
    """Test performance metrics collection"""
    
    @pytest.mark.asyncio
    async def test_metrics_collection(self, mock_config):
        """Test metrics are collected during requests"""
        client = CLOBClient(config=mock_config)
        
        with aioresponses() as m:
            m.get(
                f"{client._get_base_url()}/markets",
                payload={"markets": []},
                status=200
            )
            
            await client.get_markets()
            
            metrics = client.get_metrics()
            
            assert metrics['requests_total'] == 1
            assert metrics['requests_successful'] == 1
            assert metrics['requests_failed'] == 0
            assert metrics['cache_hits'] == 0
            assert metrics['cache_misses'] == 1
            assert metrics['average_response_time'] > 0
    
    @pytest.mark.asyncio
    async def test_error_metrics(self, mock_config):
        """Test error metrics collection"""
        client = CLOBClient(config=mock_config)
        
        with aioresponses() as m:
            m.get(
                f"{client._get_base_url()}/markets",
                status=500,
                payload={"error": "Server error"}
            )
            
            with pytest.raises(PolymarketAPIError):
                await client.get_markets()
            
            metrics = client.get_metrics()
            
            assert metrics['requests_total'] == 1
            assert metrics['requests_successful'] == 0
            assert metrics['requests_failed'] == 1


class TestCLOBClientAsyncContextManager:
    """Test async context manager functionality"""
    
    @pytest.mark.asyncio
    async def test_async_context_manager(self, mock_config):
        """Test client as async context manager"""
        async with CLOBClient(config=mock_config) as client:
            assert client.session is not None
            assert not client.session.closed
        
        # Session should be closed after context exit
        assert client.session.closed
    
    @pytest.mark.asyncio
    async def test_health_check(self, mock_config):
        """Test health check functionality"""
        client = CLOBClient(config=mock_config)
        
        with aioresponses() as m:
            m.get(
                f"{client._get_base_url()}/health",
                payload={"status": "healthy"},
                status=200
            )
            
            is_healthy = await client.health_check()
            assert is_healthy is True
    
    @pytest.mark.asyncio
    async def test_health_check_failure(self, mock_config):
        """Test health check failure"""
        client = CLOBClient(config=mock_config)
        
        with aioresponses() as m:
            m.get(
                f"{client._get_base_url()}/health",
                status=503,
                payload={"status": "unhealthy"}
            )
            
            is_healthy = await client.health_check()
            assert is_healthy is False


# Fixtures for async testing
@pytest.fixture
async def clob_client(mock_config):
    """Fixture providing initialized CLOB client"""
    async with CLOBClient(config=mock_config) as client:
        yield client


# Integration test markers
pytestmark = pytest.mark.unit