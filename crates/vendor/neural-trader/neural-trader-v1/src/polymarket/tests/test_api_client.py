"""
Polymarket API Client Tests

Following TDD principles - these tests are written before implementation.
All tests should fail initially until the actual API client is implemented.
"""

import pytest
import asyncio
import aiohttp
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
import json
from typing import Dict, List, Any
import time


class TestPolymarketAPIClient:
    """Test Polymarket API client functionality."""
    
    @pytest.mark.unit
    @pytest.mark.api
    async def test_client_initialization(self, polymarket_config):
        """Test API client initialization with configuration."""
        # This should fail until PolymarketAPIClient is implemented
        from src.polymarket.api_client import PolymarketAPIClient
        
        client = PolymarketAPIClient(config=polymarket_config)
        
        assert client.api_key == polymarket_config["api_key"]
        assert client.api_secret == polymarket_config["api_secret"]
        assert client.base_url == polymarket_config["base_url"]
        assert client.rate_limiter is not None
        assert client.session is None  # Session created on demand
    
    @pytest.mark.unit
    @pytest.mark.api
    async def test_client_context_manager(self, polymarket_config):
        """Test API client as async context manager."""
        from src.polymarket.api_client import PolymarketAPIClient
        
        async with PolymarketAPIClient(config=polymarket_config) as client:
            assert client.session is not None
            assert isinstance(client.session, aiohttp.ClientSession)
        
        # Session should be closed after context
        assert client.session.closed
    
    @pytest.mark.unit
    @pytest.mark.auth
    async def test_authentication_headers(self, polymarket_config):
        """Test authentication header generation."""
        from src.polymarket.api_client import PolymarketAPIClient
        
        client = PolymarketAPIClient(config=polymarket_config)
        
        # Test API key authentication
        headers = client._get_auth_headers()
        assert "Authorization" in headers
        assert headers["Authorization"].startswith("Bearer ")
        assert "X-API-Key" in headers
        assert headers["X-API-Key"] == polymarket_config["api_key"]
    
    @pytest.mark.unit
    @pytest.mark.auth
    async def test_signature_generation(self, polymarket_config):
        """Test request signature generation for authenticated endpoints."""
        from src.polymarket.api_client import PolymarketAPIClient
        
        client = PolymarketAPIClient(config=polymarket_config)
        
        # Test signature for order placement
        timestamp = int(time.time())
        order_data = {
            "market_id": "0x123",
            "side": "buy",
            "outcome": "Yes",
            "size": "100",
            "price": "0.65"
        }
        
        signature = client._generate_signature(
            method="POST",
            path="/orders",
            timestamp=timestamp,
            body=json.dumps(order_data)
        )
        
        assert isinstance(signature, str)
        assert len(signature) > 0
        assert signature.startswith("0x")  # Ethereum-style signature
    
    @pytest.mark.unit
    @pytest.mark.api
    async def test_get_markets(self, polymarket_config, mock_http_session, mock_api_responses):
        """Test fetching markets from API."""
        from src.polymarket.api_client import PolymarketAPIClient
        
        # Configure mock response
        mock_http_session.get.return_value.json.return_value = mock_api_responses["markets"]
        
        client = PolymarketAPIClient(config=polymarket_config)
        client.session = mock_http_session
        
        # Test basic market fetch
        markets = await client.get_markets()
        
        assert isinstance(markets, dict)
        assert "markets" in markets
        assert len(markets["markets"]) == 2
        assert markets["markets"][0]["id"] == "0x" + "a" * 40
        
        # Verify API call
        mock_http_session.get.assert_called_once()
        call_args = mock_http_session.get.call_args
        assert call_args[0][0].endswith("/markets")
    
    @pytest.mark.unit
    @pytest.mark.api
    async def test_get_markets_with_filters(self, polymarket_config, mock_http_session):
        """Test fetching markets with various filters."""
        from src.polymarket.api_client import PolymarketAPIClient
        
        mock_http_session.get.return_value.json.return_value = {"markets": []}
        
        client = PolymarketAPIClient(config=polymarket_config)
        client.session = mock_http_session
        
        # Test with filters
        filters = {
            "status": "active",
            "tag": "crypto",
            "liquidity_min": 10000,
            "created_after": "2024-01-01",
            "limit": 50,
            "cursor": "cursor_123"
        }
        
        await client.get_markets(**filters)
        
        # Verify query parameters
        call_args = mock_http_session.get.call_args
        assert "params" in call_args[1]
        params = call_args[1]["params"]
        assert params["status"] == "active"
        assert params["tag"] == "crypto"
        assert params["liquidity_min"] == 10000
    
    @pytest.mark.unit
    @pytest.mark.api
    async def test_get_market_by_id(self, polymarket_config, mock_http_session, mock_market_data):
        """Test fetching a specific market by ID."""
        from src.polymarket.api_client import PolymarketAPIClient
        
        mock_http_session.get.return_value.json.return_value = mock_market_data
        
        client = PolymarketAPIClient(config=polymarket_config)
        client.session = mock_http_session
        
        market_id = "0x" + "a" * 40
        market = await client.get_market(market_id)
        
        assert market["market_id"] == market_id
        assert "question" in market
        assert "outcomes" in market
        assert "order_book" in market
        
        # Verify API call
        call_args = mock_http_session.get.call_args
        assert call_args[0][0].endswith(f"/markets/{market_id}")
    
    @pytest.mark.unit
    @pytest.mark.api
    async def test_get_order_book(self, polymarket_config, mock_http_session, mock_market_data):
        """Test fetching order book for a market."""
        from src.polymarket.api_client import PolymarketAPIClient
        
        mock_http_session.get.return_value.json.return_value = mock_market_data["order_book"]
        
        client = PolymarketAPIClient(config=polymarket_config)
        client.session = mock_http_session
        
        market_id = "0x" + "a" * 40
        order_book = await client.get_order_book(market_id, outcome="Yes")
        
        assert "bids" in order_book
        assert "asks" in order_book
        assert len(order_book["bids"]) > 0
        assert len(order_book["asks"]) > 0
        
        # Verify bid/ask structure
        bid = order_book["bids"][0]
        assert "price" in bid
        assert "size" in bid
        assert bid["price"] < order_book["asks"][0]["price"]  # Bid < Ask
    
    @pytest.mark.unit
    @pytest.mark.api
    async def test_place_order(self, polymarket_config, mock_http_session, mock_api_responses):
        """Test placing an order."""
        from src.polymarket.api_client import PolymarketAPIClient
        
        mock_http_session.post.return_value.json.return_value = mock_api_responses["order_created"]
        
        client = PolymarketAPIClient(config=polymarket_config)
        client.session = mock_http_session
        
        order_params = {
            "market_id": "0x" + "a" * 40,
            "side": "buy",
            "outcome": "Yes",
            "size": "100",
            "price": "0.65",
            "type": "limit"
        }
        
        result = await client.place_order(**order_params)
        
        assert "order_id" in result
        assert result["status"] == "pending"
        
        # Verify API call
        mock_http_session.post.assert_called_once()
        call_args = mock_http_session.post.call_args
        assert call_args[0][0].endswith("/orders")
        assert call_args[1]["json"] == order_params
    
    @pytest.mark.unit
    @pytest.mark.api
    async def test_cancel_order(self, polymarket_config, mock_http_session, mock_api_responses):
        """Test canceling an order."""
        from src.polymarket.api_client import PolymarketAPIClient
        
        mock_http_session.delete.return_value.json.return_value = mock_api_responses["order_canceled"]
        
        client = PolymarketAPIClient(config=polymarket_config)
        client.session = mock_http_session
        
        order_id = "order_123"
        result = await client.cancel_order(order_id)
        
        assert result["order_id"] == order_id
        assert result["status"] == "canceled"
        
        # Verify API call
        call_args = mock_http_session.delete.call_args
        assert call_args[0][0].endswith(f"/orders/{order_id}")
    
    @pytest.mark.unit
    @pytest.mark.api
    async def test_get_open_orders(self, polymarket_config, mock_http_session):
        """Test fetching open orders."""
        from src.polymarket.api_client import PolymarketAPIClient
        
        mock_orders = [
            {"order_id": "order_1", "status": "open"},
            {"order_id": "order_2", "status": "open"}
        ]
        mock_http_session.get.return_value.json.return_value = {"orders": mock_orders}
        
        client = PolymarketAPIClient(config=polymarket_config)
        client.session = mock_http_session
        
        orders = await client.get_open_orders()
        
        assert len(orders) == 2
        assert all(order["status"] == "open" for order in orders)
        
        # Test with market filter
        await client.get_open_orders(market_id="0x123")
        call_args = mock_http_session.get.call_args
        assert "params" in call_args[1]
        assert call_args[1]["params"]["market_id"] == "0x123"
    
    @pytest.mark.unit
    @pytest.mark.api
    async def test_get_positions(self, polymarket_config, mock_http_session, mock_position_data):
        """Test fetching user positions."""
        from src.polymarket.api_client import PolymarketAPIClient
        
        mock_http_session.get.return_value.json.return_value = {"positions": [mock_position_data]}
        
        client = PolymarketAPIClient(config=polymarket_config)
        client.session = mock_http_session
        
        positions = await client.get_positions()
        
        assert len(positions) == 1
        position = positions[0]
        assert position["market_id"] == mock_position_data["market_id"]
        assert "unrealized_pnl" in position
        assert "realized_pnl" in position
    
    @pytest.mark.unit
    @pytest.mark.api
    async def test_get_trade_history(self, polymarket_config, mock_http_session, test_data_generator):
        """Test fetching trade history."""
        from src.polymarket.api_client import PolymarketAPIClient
        
        trades = test_data_generator.generate_trade_history(5)
        mock_http_session.get.return_value.json.return_value = {"trades": trades}
        
        client = PolymarketAPIClient(config=polymarket_config)
        client.session = mock_http_session
        
        # Test basic fetch
        history = await client.get_trade_history()
        assert len(history) == 5
        
        # Test with date range
        start_date = datetime.utcnow() - timedelta(days=7)
        end_date = datetime.utcnow()
        
        await client.get_trade_history(start_date=start_date, end_date=end_date)
        
        call_args = mock_http_session.get.call_args
        assert "params" in call_args[1]
        assert "start_date" in call_args[1]["params"]
        assert "end_date" in call_args[1]["params"]
    
    @pytest.mark.unit
    @pytest.mark.rate_limit
    async def test_rate_limiting(self, polymarket_config, mock_http_session):
        """Test rate limiting functionality."""
        from src.polymarket.api_client import PolymarketAPIClient
        from src.polymarket.exceptions import RateLimitError
        
        client = PolymarketAPIClient(config=polymarket_config)
        client.session = mock_http_session
        
        # Simulate rapid requests
        start_time = time.time()
        request_count = 0
        
        with pytest.raises(RateLimitError):
            # Try to exceed rate limit
            for _ in range(polymarket_config["rate_limit"]["calls_per_second"] + 5):
                await client.get_markets()
                request_count += 1
        
        elapsed_time = time.time() - start_time
        
        # Should have hit rate limit before completing all requests
        assert request_count <= polymarket_config["rate_limit"]["calls_per_second"]
        assert elapsed_time < 1.5  # Should fail quickly, not wait
    
    @pytest.mark.unit
    @pytest.mark.api
    async def test_retry_mechanism(self, polymarket_config, mock_http_session):
        """Test automatic retry on transient failures."""
        from src.polymarket.api_client import PolymarketAPIClient
        
        # Configure mock to fail twice then succeed
        mock_http_session.get.side_effect = [
            aiohttp.ClientError("Connection error"),
            aiohttp.ClientError("Timeout"),
            AsyncMock(json=AsyncMock(return_value={"markets": []}))()
        ]
        
        client = PolymarketAPIClient(config=polymarket_config)
        client.session = mock_http_session
        
        # Should succeed after retries
        result = await client.get_markets()
        assert "markets" in result
        
        # Verify retry attempts
        assert mock_http_session.get.call_count == 3
    
    @pytest.mark.unit
    @pytest.mark.api
    async def test_error_handling(self, polymarket_config, mock_http_session, mock_api_responses):
        """Test error handling for various API errors."""
        from src.polymarket.api_client import PolymarketAPIClient
        from src.polymarket.exceptions import (
            PolymarketAPIError, AuthenticationError, MarketNotFoundError,
            InsufficientFundsError, OrderNotFoundError
        )
        
        client = PolymarketAPIClient(config=polymarket_config)
        client.session = mock_http_session
        
        # Test 401 Unauthorized
        mock_http_session.get.return_value.status = 401
        mock_http_session.get.return_value.json.return_value = {"error": "Unauthorized"}
        
        with pytest.raises(AuthenticationError):
            await client.get_markets()
        
        # Test 404 Not Found
        mock_http_session.get.return_value.status = 404
        
        with pytest.raises(MarketNotFoundError):
            await client.get_market("invalid_id")
        
        # Test 400 Bad Request with insufficient funds
        mock_http_session.post.return_value.status = 400
        mock_http_session.post.return_value.json.return_value = {
            "error": "Insufficient funds",
            "code": "INSUFFICIENT_FUNDS"
        }
        
        with pytest.raises(InsufficientFundsError):
            await client.place_order(market_id="0x123", side="buy", size="1000000")
    
    @pytest.mark.unit
    @pytest.mark.api
    async def test_pagination_handling(self, polymarket_config, mock_http_session):
        """Test automatic pagination for large result sets."""
        from src.polymarket.api_client import PolymarketAPIClient
        
        # Mock paginated responses
        page1 = {
            "markets": [{"id": f"market_{i}"} for i in range(100)],
            "next_cursor": "cursor_page2"
        }
        page2 = {
            "markets": [{"id": f"market_{i}"} for i in range(100, 150)],
            "next_cursor": None
        }
        
        mock_http_session.get.side_effect = [
            AsyncMock(json=AsyncMock(return_value=page1))(),
            AsyncMock(json=AsyncMock(return_value=page2))()
        ]
        
        client = PolymarketAPIClient(config=polymarket_config)
        client.session = mock_http_session
        
        # Test auto-pagination
        all_markets = await client.get_all_markets()
        
        assert len(all_markets) == 150
        assert mock_http_session.get.call_count == 2
        
        # Verify cursor was used
        second_call = mock_http_session.get.call_args_list[1]
        assert second_call[1]["params"]["cursor"] == "cursor_page2"
    
    @pytest.mark.unit
    @pytest.mark.api
    async def test_response_caching(self, polymarket_config, mock_http_session, temp_cache_dir):
        """Test response caching for frequently accessed data."""
        from src.polymarket.api_client import PolymarketAPIClient
        
        mock_http_session.get.return_value.json.return_value = {"markets": [{"id": "cached_market"}]}
        
        # Create client with caching enabled
        config = {**polymarket_config, "cache_dir": str(temp_cache_dir), "cache_ttl": 60}
        client = PolymarketAPIClient(config=config)
        client.session = mock_http_session
        
        # First call should hit API
        result1 = await client.get_markets(use_cache=True)
        assert mock_http_session.get.call_count == 1
        
        # Second call should use cache
        result2 = await client.get_markets(use_cache=True)
        assert mock_http_session.get.call_count == 1  # No additional API call
        assert result1 == result2
        
        # Force refresh should bypass cache
        result3 = await client.get_markets(use_cache=False)
        assert mock_http_session.get.call_count == 2
    
    @pytest.mark.unit
    @pytest.mark.api
    async def test_bulk_operations(self, polymarket_config, mock_http_session):
        """Test bulk order operations."""
        from src.polymarket.api_client import PolymarketAPIClient
        
        mock_http_session.post.return_value.json.return_value = {
            "orders": [
                {"order_id": "order_1", "status": "pending"},
                {"order_id": "order_2", "status": "pending"},
                {"order_id": "order_3", "status": "failed", "error": "Invalid price"}
            ]
        }
        
        client = PolymarketAPIClient(config=polymarket_config)
        client.session = mock_http_session
        
        orders = [
            {"market_id": "0x1", "side": "buy", "size": "100", "price": "0.60"},
            {"market_id": "0x2", "side": "sell", "size": "50", "price": "0.70"},
            {"market_id": "0x3", "side": "buy", "size": "200", "price": "1.50"}  # Invalid
        ]
        
        results = await client.place_bulk_orders(orders)
        
        assert len(results["orders"]) == 3
        assert sum(1 for o in results["orders"] if o["status"] == "pending") == 2
        assert sum(1 for o in results["orders"] if o["status"] == "failed") == 1
    
    @pytest.mark.integration
    @pytest.mark.slow
    async def test_concurrent_requests(self, polymarket_config, mock_http_session):
        """Test handling of concurrent API requests."""
        from src.polymarket.api_client import PolymarketAPIClient
        
        mock_http_session.get.return_value.json.return_value = {"markets": []}
        
        client = PolymarketAPIClient(config=polymarket_config)
        client.session = mock_http_session
        
        # Launch multiple concurrent requests
        tasks = [
            client.get_markets(),
            client.get_open_orders(),
            client.get_positions(),
            client.get_trade_history()
        ]
        
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 4
        assert all(isinstance(r, (dict, list)) for r in results)
        
        # Should respect rate limits even with concurrent requests
        assert mock_http_session.get.call_count == 4
    
    @pytest.mark.unit
    @pytest.mark.websocket
    async def test_websocket_connection(self, polymarket_config, mock_websocket):
        """Test WebSocket connection for real-time updates."""
        from src.polymarket.api_client import PolymarketAPIClient
        
        client = PolymarketAPIClient(config=polymarket_config)
        
        # Test connection establishment
        ws = await client.connect_websocket()
        assert ws is not None
        
        # Test subscription
        await client.subscribe_market_updates("0x123")
        mock_websocket.send.assert_called()
        
        # Verify subscription message
        call_args = mock_websocket.send.call_args
        message = json.loads(call_args[0][0])
        assert message["type"] == "subscribe"
        assert message["channel"] == "market_updates"
        assert message["market_id"] == "0x123"