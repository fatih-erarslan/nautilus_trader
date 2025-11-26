"""
Comprehensive API Integration Tests for Polymarket

This module contains full API client integration tests with real endpoints,
testing the complete API lifecycle including authentication, rate limiting,
data retrieval, order placement, and WebSocket streaming.
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
import aiohttp
import websockets

from polymarket.api import (
    CLOBClient, GammaClient, PolymarketClient, 
    RateLimiter, WebSocketClient
)
from polymarket.api.base import APIError, RateLimitError
from polymarket.models import (
    Market, MarketStatus, Order, OrderSide, OrderStatus, OrderType,
    Position, TokenInfo, Trade, OrderBook
)
from polymarket.utils import generate_signature, format_number
from polymarket.utils.config import PolymarketConfig


class TestAPIIntegration:
    """Full API client integration tests with real endpoints."""

    @pytest.fixture
    async def api_client(self):
        """Create API client instance with test credentials."""
        config = PolymarketConfig.from_env()
        client = PolymarketClient(config)
        yield client
        await client.close()

    @pytest.fixture
    async def clob_client(self):
        """Create CLOB client instance."""
        config = PolymarketConfig.from_env()
        client = CLOBClient(config)
        yield client
        await client.close()

    @pytest.fixture
    async def gamma_client(self):
        """Create Gamma client instance."""
        config = PolymarketConfig.from_env()
        client = GammaClient(config)
        yield client
        await client.close()

    @pytest.fixture
    async def websocket_client(self):
        """Create WebSocket client instance."""
        config = PolymarketConfig.from_env()
        client = WebSocketClient(config)
        yield client
        await client.disconnect()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_full_api_lifecycle(self, api_client):
        """Test complete API lifecycle: auth → markets → orders → positions."""
        # Test authentication
        assert api_client.is_authenticated()
        
        # Fetch markets
        markets = await api_client.get_markets(limit=10)
        assert len(markets) > 0
        assert all(isinstance(m, Market) for m in markets)
        
        # Get specific market details
        market = markets[0]
        market_details = await api_client.get_market(market.id)
        assert market_details.id == market.id
        assert market_details.outcome_prices is not None
        
        # Get orderbook
        orderbook = await api_client.get_orderbook(market.id)
        assert isinstance(orderbook, OrderBook)
        assert len(orderbook.bids) >= 0
        assert len(orderbook.asks) >= 0
        
        # Get user positions
        positions = await api_client.get_positions()
        assert isinstance(positions, list)
        
        # Get user orders
        orders = await api_client.get_orders(market_id=market.id)
        assert isinstance(orders, list)

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_market_data_retrieval(self, clob_client):
        """Test comprehensive market data retrieval."""
        # Get active markets with different filters
        active_markets = await clob_client.get_markets(
            status=MarketStatus.ACTIVE,
            limit=20
        )
        assert len(active_markets) > 0
        
        # Test pagination
        page1 = await clob_client.get_markets(limit=5, offset=0)
        page2 = await clob_client.get_markets(limit=5, offset=5)
        assert page1[0].id != page2[0].id if len(page2) > 0 else True
        
        # Get market by slug
        if active_markets:
            market = active_markets[0]
            market_by_slug = await clob_client.get_market_by_slug(market.slug)
            assert market_by_slug.id == market.id
        
        # Get markets by category
        crypto_markets = await clob_client.get_markets_by_category("Crypto")
        assert all("crypto" in m.tags or "Crypto" in m.category 
                  for m in crypto_markets)

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_order_placement_flow(self, clob_client):
        """Test complete order placement flow with validation."""
        # Get a liquid market
        markets = await clob_client.get_markets(
            status=MarketStatus.ACTIVE,
            min_liquidity=1000,
            limit=1
        )
        
        if not markets:
            pytest.skip("No liquid markets available for testing")
        
        market = markets[0]
        
        # Get current orderbook
        orderbook = await clob_client.get_orderbook(market.id)
        
        # Calculate safe order price (outside spread)
        best_bid = Decimal(orderbook.bids[0]["price"]) if orderbook.bids else Decimal("0.1")
        safe_price = best_bid * Decimal("0.9")  # 10% below best bid
        
        # Place limit order
        order = await clob_client.place_order(
            market_id=market.id,
            outcome="Yes",
            side=OrderSide.BUY,
            size=Decimal("1.0"),
            price=safe_price,
            order_type=OrderType.LIMIT
        )
        
        assert isinstance(order, Order)
        assert order.market_id == market.id
        assert order.side == OrderSide.BUY
        assert order.status in [OrderStatus.OPEN, OrderStatus.PENDING]
        
        # Get order status
        order_status = await clob_client.get_order(order.id)
        assert order_status.id == order.id
        
        # Cancel order
        cancelled = await clob_client.cancel_order(order.id)
        assert cancelled.status == OrderStatus.CANCELLED
        
        # Verify cancellation
        final_status = await clob_client.get_order(order.id)
        assert final_status.status == OrderStatus.CANCELLED

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_websocket_streaming(self, websocket_client):
        """Test WebSocket real-time data streaming."""
        received_messages = []
        
        async def message_handler(message: Dict[str, Any]):
            received_messages.append(message)
        
        # Connect to WebSocket
        await websocket_client.connect()
        
        # Subscribe to market updates
        markets = ["market_1", "market_2"]  # Test market IDs
        await websocket_client.subscribe_markets(markets, message_handler)
        
        # Wait for messages
        await asyncio.sleep(5)
        
        # Verify messages received
        assert len(received_messages) > 0
        assert all("type" in msg for msg in received_messages)
        
        # Unsubscribe
        await websocket_client.unsubscribe_markets(markets)
        
        # Disconnect
        await websocket_client.disconnect()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_rate_limiting(self, clob_client):
        """Test rate limiting behavior and recovery."""
        # Configure aggressive rate limiter for testing
        rate_limiter = RateLimiter(
            calls_per_second=2,
            burst_size=5,
            backoff_factor=2.0
        )
        clob_client._rate_limiter = rate_limiter
        
        # Make rapid requests
        start_time = time.time()
        request_times = []
        
        for i in range(10):
            try:
                await clob_client.get_markets(limit=1)
                request_times.append(time.time() - start_time)
            except RateLimitError:
                # Expected behavior
                pass
        
        # Verify rate limiting enforced
        assert len(request_times) < 10  # Some requests should be rate limited
        
        # Verify proper spacing between successful requests
        if len(request_times) > 1:
            intervals = [request_times[i+1] - request_times[i] 
                        for i in range(len(request_times)-1)]
            avg_interval = sum(intervals) / len(intervals)
            assert avg_interval >= 0.4  # ~2 requests per second

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_gamma_integration(self, gamma_client):
        """Test Gamma market operations and approvals."""
        # Get token info
        usdc_info = await gamma_client.get_token_info("USDC")
        assert isinstance(usdc_info, TokenInfo)
        assert usdc_info.symbol == "USDC"
        
        # Check allowances
        allowance = await gamma_client.get_allowance(
            token_address=usdc_info.address,
            spender_address="0x..."  # Exchange address
        )
        assert isinstance(allowance, Decimal)
        
        # Test approval flow (dry run)
        with patch.object(gamma_client, '_send_transaction') as mock_tx:
            mock_tx.return_value = "0x123..."
            
            tx_hash = await gamma_client.approve_token(
                token_address=usdc_info.address,
                amount=Decimal("100.0")
            )
            assert tx_hash == "0x123..."
            mock_tx.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_error_handling_and_recovery(self, api_client):
        """Test API error handling and automatic recovery."""
        # Test invalid market ID
        with pytest.raises(APIError) as exc_info:
            await api_client.get_market("invalid_market_id")
        assert exc_info.value.status_code in [400, 404]
        
        # Test network error recovery
        with patch.object(api_client._session, 'get') as mock_get:
            # Simulate network error then success
            mock_get.side_effect = [
                aiohttp.ClientError("Network error"),
                AsyncMock(status=200, json=AsyncMock(return_value={"markets": []}))
            ]
            
            # Should retry and succeed
            result = await api_client.get_markets()
            assert result == []
            assert mock_get.call_count == 2

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.benchmark
    async def test_api_performance(self, clob_client, benchmark):
        """Benchmark API response times."""
        async def fetch_markets():
            return await clob_client.get_markets(limit=10)
        
        # Benchmark market fetching
        markets = await benchmark.pedantic(
            fetch_markets,
            rounds=10,
            iterations=3,
            warmup_rounds=2
        )
        
        assert len(markets) > 0
        # Verify reasonable response time (< 1 second)
        assert benchmark.stats["mean"] < 1.0

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_concurrent_requests(self, api_client):
        """Test handling of concurrent API requests."""
        # Create multiple concurrent requests
        tasks = []
        for i in range(20):
            if i % 4 == 0:
                task = api_client.get_markets(limit=5)
            elif i % 4 == 1:
                task = api_client.get_positions()
            elif i % 4 == 2:
                task = api_client.get_orders()
            else:
                task = api_client.get_trades(limit=10)
            tasks.append(task)
        
        # Execute concurrently
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        elapsed = time.time() - start_time
        
        # Verify results
        successful = [r for r in results if not isinstance(r, Exception)]
        errors = [r for r in results if isinstance(r, Exception)]
        
        assert len(successful) >= 15  # At least 75% success rate
        assert elapsed < 5.0  # Should complete within 5 seconds
        
        # Check rate limit errors are handled properly
        rate_limit_errors = [e for e in errors if isinstance(e, RateLimitError)]
        assert len(rate_limit_errors) < 10  # Not all requests should be rate limited

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_data_consistency(self, api_client):
        """Test data consistency across different API endpoints."""
        # Get market from different endpoints
        markets = await api_client.get_markets(limit=1)
        if not markets:
            pytest.skip("No markets available")
        
        market_id = markets[0].id
        
        # Fetch same market data from different endpoints
        market_detail = await api_client.get_market(market_id)
        orderbook = await api_client.get_orderbook(market_id)
        trades = await api_client.get_trades(market_id=market_id, limit=10)
        
        # Verify consistency
        assert market_detail.id == market_id
        assert orderbook.market_id == market_id
        assert all(t.market_id == market_id for t in trades)
        
        # Verify price consistency
        if orderbook.bids and orderbook.asks:
            spread = Decimal(orderbook.asks[0]["price"]) - Decimal(orderbook.bids[0]["price"])
            assert spread >= 0  # Ask should be >= bid
            assert spread < Decimal("0.1")  # Reasonable spread

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_authentication_flow(self, api_client):
        """Test complete authentication and session management."""
        # Test initial authentication
        assert api_client.is_authenticated()
        
        # Test signature generation
        timestamp = int(time.time() * 1000)
        message = f"GET/markets{timestamp}"
        signature = generate_signature(message, api_client.config.private_key)
        assert len(signature) > 0
        
        # Test authenticated request
        headers = api_client._get_auth_headers("GET", "/markets")
        assert "X-API-KEY" in headers
        assert "X-SIGNATURE" in headers
        assert "X-TIMESTAMP" in headers
        
        # Test session refresh
        old_session = api_client._session
        await api_client.close()
        await api_client._ensure_session()
        assert api_client._session != old_session


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])