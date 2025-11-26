"""Test suite for Polymarket API client."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import json
import time
from requests.exceptions import HTTPError, ConnectionError, Timeout

from src.polymarket.api.client import PolymarketClient
from src.polymarket.models.market import Market, MarketOutcome, MarketStatus
from src.polymarket.models.order import Order, OrderType, OrderSide, OrderStatus
from src.polymarket.models.position import Position
from src.polymarket.models.common import TokenInfo, Resolution


class TestPolymarketClient:
    """Test cases for Polymarket API client."""
    
    @pytest.fixture
    def client(self):
        """Create a test client instance."""
        return PolymarketClient(
            api_key="test_api_key",
            api_secret="test_api_secret",
            base_url="https://api.polymarket.com",
            testnet=True
        )
    
    @pytest.fixture
    def mock_response(self):
        """Create a mock response object."""
        response = Mock()
        response.status_code = 200
        response.headers = {"X-RateLimit-Remaining": "100", "X-RateLimit-Reset": str(int(time.time()) + 3600)}
        return response
    
    def test_client_initialization(self):
        """Test client initialization with different configurations."""
        # Test production client
        client = PolymarketClient(
            api_key="prod_key",
            api_secret="prod_secret",
            testnet=False
        )
        assert client.api_key == "prod_key"
        assert client.base_url == "https://api.polymarket.com"
        assert not client.testnet
        
        # Test testnet client
        testnet_client = PolymarketClient(
            api_key="test_key",
            api_secret="test_secret",
            testnet=True
        )
        assert testnet_client.testnet
        assert testnet_client.base_url == "https://api.testnet.polymarket.com"
    
    @patch('requests.Session.request')
    def test_authentication_headers(self, mock_request, client, mock_response):
        """Test that authentication headers are properly set."""
        mock_response.json.return_value = {"data": []}
        mock_request.return_value = mock_response
        
        client._request("GET", "/markets")
        
        # Verify authentication headers were set
        call_args = mock_request.call_args
        headers = call_args[1]["headers"]
        assert "Authorization" in headers
        assert headers["Authorization"].startswith("Bearer ")
        assert "X-API-Key" in headers
        assert headers["X-API-Key"] == "test_api_key"
    
    @patch('requests.Session.request')
    def test_get_markets(self, mock_request, client, mock_response):
        """Test getting market list."""
        market_data = {
            "data": [
                {
                    "id": "0x123",
                    "question": "Will BTC reach $100k by end of 2024?",
                    "slug": "btc-100k-2024",
                    "status": "active",
                    "outcomes": [
                        {"id": "0x456", "name": "Yes", "price": 0.65},
                        {"id": "0x789", "name": "No", "price": 0.35}
                    ],
                    "volume": 1000000.0,
                    "liquidity": 500000.0,
                    "created_at": "2024-01-01T00:00:00Z",
                    "end_date": "2024-12-31T23:59:59Z",
                    "resolution": None
                }
            ]
        }
        mock_response.json.return_value = market_data
        mock_request.return_value = mock_response
        
        markets = client.get_markets(limit=10, active_only=True)
        
        assert len(markets) == 1
        market = markets[0]
        assert isinstance(market, Market)
        assert market.id == "0x123"
        assert market.question == "Will BTC reach $100k by end of 2024?"
        assert market.status == MarketStatus.ACTIVE
        assert len(market.outcomes) == 2
        assert market.outcomes[0].price == 0.65
        
        # Verify request parameters
        mock_request.assert_called_once()
        call_args = mock_request.call_args
        assert call_args[0][0] == "GET"
        assert "/markets" in call_args[0][1]
    
    @patch('requests.Session.request')
    def test_get_market_details(self, mock_request, client, mock_response):
        """Test getting detailed market information."""
        market_data = {
            "data": {
                "id": "0x123",
                "question": "Will ETH merge happen in 2024?",
                "slug": "eth-merge-2024",
                "status": "active",
                "outcomes": [
                    {"id": "0x456", "name": "Yes", "price": 0.80, "liquidity": 100000},
                    {"id": "0x789", "name": "No", "price": 0.20, "liquidity": 50000}
                ],
                "volume": 2000000.0,
                "volume_24h": 50000.0,
                "liquidity": 150000.0,
                "created_at": "2024-01-01T00:00:00Z",
                "end_date": "2024-06-30T23:59:59Z",
                "resolution": None,
                "category": "Crypto",
                "tags": ["ethereum", "merge", "technology"],
                "description": "This market resolves to Yes if..."
            }
        }
        mock_response.json.return_value = market_data
        mock_request.return_value = mock_response
        
        market = client.get_market_details("0x123")
        
        assert isinstance(market, Market)
        assert market.id == "0x123"
        assert market.category == "Crypto"
        assert "ethereum" in market.tags
        assert market.volume_24h == 50000.0
    
    @patch('requests.Session.request')
    def test_get_orderbook(self, mock_request, client, mock_response):
        """Test getting market orderbook."""
        orderbook_data = {
            "data": {
                "market_id": "0x123",
                "outcome_id": "0x456",
                "bids": [
                    {"price": 0.65, "size": 1000, "total": 650},
                    {"price": 0.64, "size": 2000, "total": 1280},
                    {"price": 0.63, "size": 3000, "total": 1890}
                ],
                "asks": [
                    {"price": 0.66, "size": 1500, "total": 990},
                    {"price": 0.67, "size": 2500, "total": 1675},
                    {"price": 0.68, "size": 3500, "total": 2380}
                ],
                "spread": 0.01,
                "mid_price": 0.655,
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }
        mock_response.json.return_value = orderbook_data
        mock_request.return_value = mock_response
        
        orderbook = client.get_orderbook("0x123", "0x456", depth=3)
        
        assert orderbook["market_id"] == "0x123"
        assert len(orderbook["bids"]) == 3
        assert len(orderbook["asks"]) == 3
        assert orderbook["bids"][0]["price"] == 0.65
        assert orderbook["spread"] == 0.01
        assert orderbook["mid_price"] == 0.655
    
    @patch('requests.Session.request')
    def test_place_order(self, mock_request, client, mock_response):
        """Test placing orders."""
        order_response = {
            "data": {
                "id": "order_123",
                "market_id": "0x123",
                "outcome_id": "0x456",
                "side": "buy",
                "type": "limit",
                "price": 0.65,
                "size": 100,
                "filled": 0,
                "remaining": 100,
                "status": "open",
                "created_at": "2024-01-15T10:30:00Z",
                "updated_at": "2024-01-15T10:30:00Z"
            }
        }
        mock_response.json.return_value = order_response
        mock_request.return_value = mock_response
        
        order = client.place_order(
            market_id="0x123",
            outcome_id="0x456",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            size=100,
            price=0.65
        )
        
        assert isinstance(order, Order)
        assert order.id == "order_123"
        assert order.side == OrderSide.BUY
        assert order.type == OrderType.LIMIT
        assert order.price == 0.65
        assert order.size == 100
        assert order.status == OrderStatus.OPEN
        
        # Verify request payload
        mock_request.assert_called_once()
        call_args = mock_request.call_args
        payload = json.loads(call_args[1]["data"])
        assert payload["market_id"] == "0x123"
        assert payload["side"] == "buy"
        assert payload["price"] == 0.65
    
    @patch('requests.Session.request')
    def test_cancel_order(self, mock_request, client, mock_response):
        """Test canceling an order."""
        cancel_response = {
            "data": {
                "id": "order_123",
                "status": "cancelled",
                "cancelled_at": "2024-01-15T10:35:00Z"
            }
        }
        mock_response.json.return_value = cancel_response
        mock_request.return_value = mock_response
        
        result = client.cancel_order("order_123")
        
        assert result["id"] == "order_123"
        assert result["status"] == "cancelled"
        
        mock_request.assert_called_once()
        call_args = mock_request.call_args
        assert call_args[0][0] == "DELETE"
        assert "orders/order_123" in call_args[0][1]
    
    @patch('requests.Session.request')
    def test_get_positions(self, mock_request, client, mock_response):
        """Test getting user positions."""
        positions_data = {
            "data": [
                {
                    "id": "pos_123",
                    "market_id": "0x123",
                    "outcome_id": "0x456",
                    "size": 1000,
                    "avg_price": 0.60,
                    "current_price": 0.65,
                    "pnl": 50.0,
                    "pnl_percent": 8.33,
                    "market": {
                        "question": "Will BTC reach $100k?",
                        "status": "active"
                    }
                }
            ]
        }
        mock_response.json.return_value = positions_data
        mock_request.return_value = mock_response
        
        positions = client.get_positions()
        
        assert len(positions) == 1
        position = positions[0]
        assert isinstance(position, Position)
        assert position.id == "pos_123"
        assert position.size == 1000
        assert position.avg_price == 0.60
        assert position.pnl == 50.0
    
    @patch('requests.Session.request')
    def test_get_balances(self, mock_request, client, mock_response):
        """Test getting account balances."""
        balances_data = {
            "data": {
                "USDC": {
                    "total": 10000.0,
                    "available": 8000.0,
                    "locked": 2000.0
                },
                "ETH": {
                    "total": 5.0,
                    "available": 4.5,
                    "locked": 0.5
                }
            }
        }
        mock_response.json.return_value = balances_data
        mock_request.return_value = mock_response
        
        balances = client.get_balances()
        
        assert "USDC" in balances
        assert balances["USDC"]["total"] == 10000.0
        assert balances["USDC"]["available"] == 8000.0
        assert "ETH" in balances
        assert balances["ETH"]["total"] == 5.0
    
    @patch('requests.Session.request')
    def test_rate_limiting(self, mock_request, client):
        """Test rate limiting functionality."""
        # First request succeeds
        response1 = Mock()
        response1.status_code = 200
        response1.headers = {"X-RateLimit-Remaining": "1", "X-RateLimit-Reset": str(int(time.time()) + 60)}
        response1.json.return_value = {"data": []}
        
        # Second request should be rate limited
        response2 = Mock()
        response2.status_code = 429
        response2.headers = {"Retry-After": "60"}
        response2.json.return_value = {"error": "Rate limit exceeded"}
        
        mock_request.side_effect = [response1, response2]
        
        # First request should succeed
        client._request("GET", "/markets")
        
        # Second request should raise rate limit error
        with pytest.raises(HTTPError) as exc_info:
            client._request("GET", "/markets")
        
        assert exc_info.value.response.status_code == 429
    
    @patch('requests.Session.request')
    def test_retry_on_network_error(self, mock_request, client, mock_response):
        """Test retry logic on network errors."""
        # First two attempts fail, third succeeds
        mock_request.side_effect = [
            ConnectionError("Network error"),
            Timeout("Request timeout"),
            mock_response
        ]
        mock_response.json.return_value = {"data": []}
        
        # Should succeed after retries
        result = client._request("GET", "/markets", max_retries=3)
        
        assert mock_request.call_count == 3
        assert result == {"data": []}
    
    @patch('requests.Session.request')
    def test_error_handling(self, mock_request, client):
        """Test error handling for various HTTP errors."""
        # Test 400 Bad Request
        response_400 = Mock()
        response_400.status_code = 400
        response_400.json.return_value = {"error": "Invalid parameters"}
        response_400.raise_for_status.side_effect = HTTPError(response=response_400)
        mock_request.return_value = response_400
        
        with pytest.raises(HTTPError) as exc_info:
            client._request("GET", "/invalid")
        assert exc_info.value.response.status_code == 400
        
        # Test 401 Unauthorized
        response_401 = Mock()
        response_401.status_code = 401
        response_401.json.return_value = {"error": "Invalid API key"}
        response_401.raise_for_status.side_effect = HTTPError(response=response_401)
        mock_request.return_value = response_401
        
        with pytest.raises(HTTPError) as exc_info:
            client._request("GET", "/protected")
        assert exc_info.value.response.status_code == 401
        
        # Test 500 Server Error
        response_500 = Mock()
        response_500.status_code = 500
        response_500.json.return_value = {"error": "Internal server error"}
        response_500.raise_for_status.side_effect = HTTPError(response=response_500)
        mock_request.return_value = response_500
        
        with pytest.raises(HTTPError) as exc_info:
            client._request("GET", "/error")
        assert exc_info.value.response.status_code == 500
    
    def test_signature_generation(self, client):
        """Test API signature generation for authenticated requests."""
        timestamp = "1234567890"
        method = "POST"
        path = "/orders"
        body = '{"market_id": "0x123", "size": 100}'
        
        signature = client._generate_signature(timestamp, method, path, body)
        
        assert signature is not None
        assert len(signature) > 0
        # Signature should be deterministic for same inputs
        signature2 = client._generate_signature(timestamp, method, path, body)
        assert signature == signature2
    
    @patch('requests.Session.request')
    def test_websocket_connection(self, mock_request, client):
        """Test WebSocket connection setup for real-time data."""
        # Note: This is a placeholder for WebSocket testing
        # Actual implementation would use websocket-client library
        ws_url = client._get_websocket_url()
        assert ws_url.startswith("wss://")
        assert "polymarket.com" in ws_url
        if client.testnet:
            assert "testnet" in ws_url