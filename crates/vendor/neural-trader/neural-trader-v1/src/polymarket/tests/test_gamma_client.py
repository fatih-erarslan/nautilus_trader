"""
Test suite for the GammaClient implementation.

Tests the Gamma Markets API client for:
- Market metadata retrieval
- Historical data fetching
- Market statistics and analytics
- Error handling and caching
- Data transformation and validation
"""

import pytest
import json
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from typing import Dict, List, Any

import aiohttp

from src.polymarket.api.gamma_client import (
    GammaClient,
    GammaAPIError,
    MarketNotFoundError,
    InvalidDateRangeError,
)
from src.polymarket.api.base import (
    PolymarketAPIError,
    RateLimitError,
    AuthenticationError,
    ValidationError,
)
from src.polymarket.models.market import Market, MarketStatus, Outcome
from src.polymarket.models.event import Event, EventCategory, EventStatus
from src.polymarket.models.analytics import (
    MarketAnalytics,
    PriceHistory,
    VolumeData,
    LiquidityMetrics,
)
from src.polymarket.utils import PolymarketConfig


class TestGammaClient:
    """Test cases for the GammaClient implementation."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return PolymarketConfig(
            gamma_url="https://gamma-api.polymarket.com",
            api_key="test_api_key",
            private_key="test_private_key",
            timeout=30
        )
    
    @pytest.fixture
    def gamma_client(self, config):
        """Create test Gamma client."""
        return GammaClient(config=config, cache_ttl=60, max_retries=2)
    
    @pytest.fixture
    def mock_session(self):
        """Create mock aiohttp session."""
        session = AsyncMock(spec=aiohttp.ClientSession)
        session.closed = False
        return session
    
    @pytest.fixture
    def sample_market_data(self):
        """Sample market data for testing."""
        return {
            "id": "0x1234567890",
            "question": "Will Bitcoin reach $100,000 by end of 2024?",
            "slug": "bitcoin-100k-2024",
            "description": "This market will resolve to Yes if...",
            "outcomes": [
                {
                    "id": "0xabc123",
                    "name": "Yes",
                    "price": "0.6500",
                    "volume": "1000000.50",
                    "liquidity": "250000.75"
                },
                {
                    "id": "0xdef456",
                    "name": "No", 
                    "price": "0.3500",
                    "volume": "750000.25",
                    "liquidity": "180000.00"
                }
            ],
            "status": "active",
            "category": "Crypto",
            "tags": ["bitcoin", "cryptocurrency", "price"],
            "created_at": "2024-01-01T00:00:00Z",
            "end_date": "2024-12-31T23:59:59Z",
            "resolution_date": None,
            "volume": "1750000.75",
            "volume_24h": "50000.00",
            "liquidity": "430000.75",
            "participants": 1250,
            "event_id": "event_123",
            "creator": "0x9876543210",
            "fee_rate": "0.02"
        }
    
    @pytest.fixture
    def sample_event_data(self):
        """Sample event data for testing."""
        return {
            "id": "event_123",
            "title": "Bitcoin Price Predictions 2024",
            "description": "Markets related to Bitcoin price in 2024",
            "category": "Crypto",
            "status": "active",
            "created_at": "2024-01-01T00:00:00Z",
            "end_date": "2024-12-31T23:59:59Z",
            "market_count": 5,
            "total_volume": "10000000.00",
            "tags": ["bitcoin", "cryptocurrency", "2024"],
            "image_url": "https://example.com/bitcoin.jpg"
        }
    
    def test_client_initialization(self, config):
        """Test Gamma client initialization."""
        client = GammaClient(config=config, cache_ttl=120, max_retries=5)
        assert client.config == config
        assert client._cache.ttl == 120
        assert client.max_retries == 5
        assert client._get_base_url() == "https://gamma-api.polymarket.com"
    
    def test_client_default_initialization(self):
        """Test client with default configuration."""
        client = GammaClient()
        assert client.config is not None
        assert client._cache.ttl == 300  # Default cache TTL
        assert client.max_retries == 3
    
    @pytest.mark.asyncio
    async def test_health_check_success(self, gamma_client, mock_session):
        """Test successful health check."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {"status": "healthy", "timestamp": "2024-01-01T12:00:00Z"}
        
        mock_session.request.return_value.__aenter__.return_value = mock_response
        gamma_client.session = mock_session
        
        result = await gamma_client.health_check()
        assert result is True
        
        mock_session.request.assert_called_once()
        call_args = mock_session.request.call_args
        assert call_args[1]['method'] == 'GET'
        assert '/health' in call_args[1]['url']
    
    @pytest.mark.asyncio
    async def test_health_check_failure(self, gamma_client, mock_session):
        """Test health check failure."""
        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.json.return_value = {"error": "Internal server error"}
        
        mock_session.request.return_value.__aenter__.return_value = mock_response
        gamma_client.session = mock_session
        
        result = await gamma_client.health_check()
        assert result is False
    
    @pytest.mark.asyncio
    async def test_get_markets_success(self, gamma_client, mock_session, sample_market_data):
        """Test successful market retrieval."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "data": [sample_market_data],
            "total": 1,
            "page": 1,
            "per_page": 10
        }
        
        mock_session.request.return_value.__aenter__.return_value = mock_response
        gamma_client.session = mock_session
        
        markets = await gamma_client.get_markets(limit=10, category="Crypto")
        
        assert len(markets) == 1
        market = markets[0]
        assert isinstance(market, Market)
        assert market.id == "0x1234567890"
        assert market.question == "Will Bitcoin reach $100,000 by end of 2024?"
        assert market.status == MarketStatus.ACTIVE
        assert len(market.outcomes) == 2
        assert market.outcomes[0].price == Decimal("0.6500")
        assert market.volume == Decimal("1750000.75")
        
        # Verify request parameters
        mock_session.request.assert_called_once()
        call_args = mock_session.request.call_args
        assert call_args[1]['params']['limit'] == 10
        assert call_args[1]['params']['category'] == "Crypto"
    
    @pytest.mark.asyncio
    async def test_get_markets_with_filters(self, gamma_client, mock_session, sample_market_data):
        """Test market retrieval with various filters."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {"data": [sample_market_data]}
        
        mock_session.request.return_value.__aenter__.return_value = mock_response
        gamma_client.session = mock_session
        
        await gamma_client.get_markets(
            limit=50,
            offset=20,
            category="Politics",
            status="active",
            tags=["election", "2024"],
            min_liquidity=1000.0,
            sort_by="volume",
            sort_order="desc"
        )
        
        call_args = mock_session.request.call_args
        params = call_args[1]['params']
        assert params['limit'] == 50
        assert params['offset'] == 20
        assert params['category'] == "Politics"
        assert params['status'] == "active"
        assert params['tags'] == "election,2024"
        assert params['min_liquidity'] == 1000.0
        assert params['sort_by'] == "volume"
        assert params['sort_order'] == "desc"
    
    @pytest.mark.asyncio
    async def test_get_market_by_id_success(self, gamma_client, mock_session, sample_market_data):
        """Test successful single market retrieval."""
        market_id = "0x1234567890"
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {"data": sample_market_data}
        
        mock_session.request.return_value.__aenter__.return_value = mock_response
        gamma_client.session = mock_session
        
        market = await gamma_client.get_market_by_id(market_id)
        
        assert isinstance(market, Market)
        assert market.id == market_id
        assert market.question == sample_market_data["question"]
        
        # Verify correct endpoint called
        call_args = mock_session.request.call_args
        assert f'/markets/{market_id}' in call_args[1]['url']
    
    @pytest.mark.asyncio
    async def test_get_market_by_id_not_found(self, gamma_client, mock_session):
        """Test market not found scenario."""
        market_id = "0xnonexistent"
        mock_response = AsyncMock()
        mock_response.status = 404
        mock_response.json.return_value = {"error": "Market not found"}
        
        mock_session.request.return_value.__aenter__.return_value = mock_response
        gamma_client.session = mock_session
        
        with patch.object(gamma_client, '_handle_response') as mock_handle:
            mock_handle.side_effect = MarketNotFoundError("Market not found")
            
            with pytest.raises(MarketNotFoundError):
                await gamma_client.get_market_by_id(market_id)
    
    @pytest.mark.asyncio
    async def test_get_events_success(self, gamma_client, mock_session, sample_event_data):
        """Test successful event retrieval."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "data": [sample_event_data],
            "total": 1
        }
        
        mock_session.request.return_value.__aenter__.return_value = mock_response
        gamma_client.session = mock_session
        
        events = await gamma_client.get_events(category="Crypto")
        
        assert len(events) == 1
        event = events[0]
        assert isinstance(event, Event)
        assert event.id == "event_123"
        assert event.title == "Bitcoin Price Predictions 2024"
        assert event.category == EventCategory.CRYPTO
        assert event.status == EventStatus.ACTIVE
    
    @pytest.mark.asyncio
    async def test_get_event_by_id_success(self, gamma_client, mock_session, sample_event_data):
        """Test successful single event retrieval."""
        event_id = "event_123"
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {"data": sample_event_data}
        
        mock_session.request.return_value.__aenter__.return_value = mock_response
        gamma_client.session = mock_session
        
        event = await gamma_client.get_event_by_id(event_id)
        
        assert isinstance(event, Event)
        assert event.id == event_id
        assert event.title == sample_event_data["title"]
    
    @pytest.mark.asyncio
    async def test_get_market_history_success(self, gamma_client, mock_session):
        """Test successful market price history retrieval."""
        market_id = "0x1234567890"
        history_data = {
            "data": {
                "market_id": market_id,
                "outcome_id": "0xabc123",
                "prices": [
                    {"timestamp": "2024-01-01T00:00:00Z", "price": "0.50", "volume": "1000.0"},
                    {"timestamp": "2024-01-01T01:00:00Z", "price": "0.52", "volume": "1500.0"},
                    {"timestamp": "2024-01-01T02:00:00Z", "price": "0.55", "volume": "2000.0"}
                ],
                "start_date": "2024-01-01T00:00:00Z",
                "end_date": "2024-01-01T02:00:00Z",
                "interval": "1h"
            }
        }
        
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = history_data
        
        mock_session.request.return_value.__aenter__.return_value = mock_response
        gamma_client.session = mock_session
        
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 2)
        
        price_history = await gamma_client.get_market_history(
            market_id=market_id,
            outcome_id="0xabc123",
            start_date=start_date,
            end_date=end_date,
            interval="1h"
        )
        
        assert isinstance(price_history, PriceHistory)
        assert price_history.market_id == market_id
        assert len(price_history.prices) == 3
        assert price_history.prices[0].price == Decimal("0.50")
        assert price_history.interval == "1h"
        
        # Verify request parameters
        call_args = mock_session.request.call_args
        params = call_args[1]['params']
        assert params['start_date'] == start_date.isoformat()
        assert params['end_date'] == end_date.isoformat()
        assert params['interval'] == "1h"
    
    @pytest.mark.asyncio
    async def test_get_market_history_invalid_date_range(self, gamma_client):
        """Test invalid date range handling."""
        start_date = datetime(2024, 1, 2)
        end_date = datetime(2024, 1, 1)  # End before start
        
        with pytest.raises(InvalidDateRangeError):
            await gamma_client.get_market_history(
                market_id="0x123",
                start_date=start_date,
                end_date=end_date
            )
    
    @pytest.mark.asyncio
    async def test_get_market_analytics_success(self, gamma_client, mock_session):
        """Test successful market analytics retrieval."""
        market_id = "0x1234567890"
        analytics_data = {
            "data": {
                "market_id": market_id,
                "volume_24h": "50000.00",
                "volume_7d": "300000.00",
                "volume_30d": "1200000.00",
                "price_change_24h": "0.05",
                "price_change_7d": "0.15",
                "liquidity": "250000.00",
                "spread": "0.02",
                "participants": 1250,
                "trades_24h": 150,
                "last_trade_price": "0.65",
                "last_updated": "2024-01-15T12:00:00Z"
            }
        }
        
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = analytics_data
        
        mock_session.request.return_value.__aenter__.return_value = mock_response
        gamma_client.session = mock_session
        
        analytics = await gamma_client.get_market_analytics(market_id)
        
        assert isinstance(analytics, MarketAnalytics)
        assert analytics.market_id == market_id
        assert analytics.volume_24h == Decimal("50000.00")
        assert analytics.liquidity == Decimal("250000.00")
        assert analytics.participants == 1250
    
    @pytest.mark.asyncio
    async def test_get_market_trades_success(self, gamma_client, mock_session):
        """Test successful market trades retrieval."""
        market_id = "0x1234567890"
        trades_data = {
            "data": [
                {
                    "id": "trade_1",
                    "market_id": market_id,
                    "outcome_id": "0xabc123",
                    "side": "buy",
                    "price": "0.65",
                    "size": "100.0",
                    "timestamp": "2024-01-15T12:00:00Z",
                    "trader": "0x9876543210",
                    "fee": "1.30"
                },
                {
                    "id": "trade_2", 
                    "market_id": market_id,
                    "outcome_id": "0xdef456",
                    "side": "sell",
                    "price": "0.35",
                    "size": "200.0",
                    "timestamp": "2024-01-15T12:01:00Z",
                    "trader": "0x1234567890",
                    "fee": "1.40"
                }
            ]
        }
        
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = trades_data
        
        mock_session.request.return_value.__aenter__.return_value = mock_response
        gamma_client.session = mock_session
        
        trades = await gamma_client.get_market_trades(market_id, limit=100)
        
        assert len(trades) == 2
        assert trades[0].id == "trade_1"
        assert trades[0].price == Decimal("0.65")
        assert trades[0].size == Decimal("100.0")
    
    @pytest.mark.asyncio
    async def test_search_markets_success(self, gamma_client, mock_session, sample_market_data):
        """Test successful market search."""
        search_query = "Bitcoin"
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "data": [sample_market_data],
            "total": 1,
            "query": search_query
        }
        
        mock_session.request.return_value.__aenter__.return_value = mock_response
        gamma_client.session = mock_session
        
        markets = await gamma_client.search_markets(query=search_query)
        
        assert len(markets) == 1
        assert isinstance(markets[0], Market)
        
        # Verify search parameters
        call_args = mock_session.request.call_args
        assert call_args[1]['params']['q'] == search_query
    
    @pytest.mark.asyncio
    async def test_get_popular_markets_success(self, gamma_client, mock_session, sample_market_data):
        """Test successful popular markets retrieval."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "data": [sample_market_data],
            "period": "24h",
            "sort_by": "volume"
        }
        
        mock_session.request.return_value.__aenter__.return_value = mock_response
        gamma_client.session = mock_session
        
        markets = await gamma_client.get_popular_markets(period="24h", limit=10)
        
        assert len(markets) == 1
        assert isinstance(markets[0], Market)
        
        # Verify endpoint and parameters
        call_args = mock_session.request.call_args
        assert '/markets/popular' in call_args[1]['url']
        assert call_args[1]['params']['period'] == "24h"
        assert call_args[1]['params']['limit'] == 10
    
    @pytest.mark.asyncio
    async def test_caching_behavior(self, gamma_client, mock_session, sample_market_data):
        """Test response caching behavior."""
        market_id = "0x1234567890"
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {"data": sample_market_data}
        
        mock_session.request.return_value.__aenter__.return_value = mock_response
        gamma_client.session = mock_session
        
        # First call should hit the API
        market1 = await gamma_client.get_market_by_id(market_id)
        assert mock_session.request.call_count == 1
        
        # Second call should use cache
        market2 = await gamma_client.get_market_by_id(market_id)
        assert mock_session.request.call_count == 1  # No additional call
        
        assert market1.id == market2.id
        
        # Check cache hit metrics
        metrics = gamma_client.get_metrics()
        assert metrics['cache_hits'] > 0
    
    @pytest.mark.asyncio
    async def test_rate_limit_handling(self, gamma_client, mock_session):
        """Test rate limit error handling."""
        mock_response = AsyncMock()
        mock_response.status = 429
        mock_response.headers = {'Retry-After': '60'}
        mock_response.json.return_value = {"error": "Rate limit exceeded"}
        
        mock_session.request.return_value.__aenter__.return_value = mock_response
        gamma_client.session = mock_session
        
        with patch.object(gamma_client, '_handle_response') as mock_handle:
            mock_handle.side_effect = RateLimitError(retry_after=60)
            
            with pytest.raises(RateLimitError) as exc_info:
                await gamma_client.get_markets()
            
            assert exc_info.value.retry_after == 60
    
    @pytest.mark.asyncio
    async def test_authentication_error_handling(self, gamma_client, mock_session):
        """Test authentication error handling."""
        mock_response = AsyncMock()
        mock_response.status = 401
        mock_response.json.return_value = {"error": "Invalid API key"}
        
        mock_session.request.return_value.__aenter__.return_value = mock_response
        gamma_client.session = mock_session
        
        with patch.object(gamma_client, '_handle_response') as mock_handle:
            mock_handle.side_effect = AuthenticationError("Invalid API key")
            
            with pytest.raises(AuthenticationError):
                await gamma_client.get_markets()
    
    @pytest.mark.asyncio
    async def test_validation_error_handling(self, gamma_client, mock_session):
        """Test validation error handling."""
        mock_response = AsyncMock()
        mock_response.status = 400
        mock_response.json.return_value = {"error": "Invalid parameters"}
        
        mock_session.request.return_value.__aenter__.return_value = mock_response
        gamma_client.session = mock_session
        
        with patch.object(gamma_client, '_handle_response') as mock_handle:
            mock_handle.side_effect = ValidationError("Invalid parameters")
            
            with pytest.raises(ValidationError):
                await gamma_client.get_markets(limit=-1)  # Invalid limit
    
    @pytest.mark.asyncio
    async def test_network_error_retry(self, gamma_client, mock_session, sample_market_data):
        """Test network error retry logic."""
        # First two attempts fail, third succeeds
        network_error = aiohttp.ClientError("Network error")
        success_response = AsyncMock()
        success_response.status = 200
        success_response.json.return_value = {"data": [sample_market_data]}
        
        mock_session.request.side_effect = [
            asyncio.coroutine(lambda: None)(),  # First call fails
            asyncio.coroutine(lambda: None)(),  # Second call fails  
            success_response  # Third call succeeds
        ]
        
        # Mock the context manager behavior
        mock_session.request.return_value.__aenter__.side_effect = [
            network_error,
            network_error,
            success_response
        ]
        
        gamma_client.session = mock_session
        
        with patch('asyncio.sleep'):  # Speed up test
            markets = await gamma_client.get_markets()
            
        assert len(markets) == 1
        assert mock_session.request.call_count == 3
    
    def test_metrics_collection(self, gamma_client):
        """Test metrics collection and reporting."""
        metrics = gamma_client.get_metrics()
        
        assert 'client' in metrics
        assert 'requests_total' in metrics
        assert 'requests_successful' in metrics
        assert 'requests_failed' in metrics
        assert 'cache_hits' in metrics
        assert 'cache_misses' in metrics
        assert 'average_response_time' in metrics
        assert 'cache_info' in metrics
        
        assert metrics['client'] == 'GammaClient'
        assert isinstance(metrics['requests_total'], int)
        assert isinstance(metrics['cache_info'], dict)
    
    def test_cache_management(self, gamma_client):
        """Test cache clearing and management."""
        # Add some data to cache
        gamma_client._cache['test_key'] = {'data': 'test_value'}
        assert len(gamma_client._cache) == 1
        
        # Clear cache
        gamma_client.clear_cache()
        assert len(gamma_client._cache) == 0
    
    @pytest.mark.asyncio
    async def test_session_management(self, gamma_client):
        """Test HTTP session lifecycle management."""
        # Session should be None initially
        assert gamma_client.session is None
        
        # Ensure session gets created
        await gamma_client._ensure_session()
        assert gamma_client.session is not None
        assert not gamma_client.session.closed
        
        # Close session
        await gamma_client.close()
        assert gamma_client.session.closed
    
    @pytest.mark.asyncio
    async def test_context_manager_usage(self, config):
        """Test using client as async context manager."""
        async with GammaClient(config=config) as client:
            assert client.session is not None
            assert not client.session.closed
        
        # Session should be closed after context exit
        assert client.session.closed


class TestGammaClientDataTransformation:
    """Test data transformation and validation in GammaClient."""
    
    @pytest.fixture
    def gamma_client(self):
        """Create test client."""
        return GammaClient()
    
    def test_market_data_transformation(self, gamma_client, sample_market_data):
        """Test transformation of raw market data to Market objects."""
        market = gamma_client._transform_market_data(sample_market_data)
        
        assert isinstance(market, Market)
        assert market.id == sample_market_data["id"]
        assert market.question == sample_market_data["question"]
        assert market.status == MarketStatus.ACTIVE
        assert len(market.outcomes) == 2
        assert market.volume == Decimal(sample_market_data["volume"])
        assert market.liquidity == Decimal(sample_market_data["liquidity"])
    
    def test_event_data_transformation(self, gamma_client, sample_event_data):
        """Test transformation of raw event data to Event objects."""
        event = gamma_client._transform_event_data(sample_event_data)
        
        assert isinstance(event, Event)
        assert event.id == sample_event_data["id"]
        assert event.title == sample_event_data["title"]
        assert event.category == EventCategory.CRYPTO
        assert event.status == EventStatus.ACTIVE
    
    def test_outcome_data_transformation(self, gamma_client):
        """Test transformation of outcome data."""
        outcome_data = {
            "id": "0xabc123",
            "name": "Yes",
            "price": "0.6500",
            "volume": "1000000.50",
            "liquidity": "250000.75"
        }
        
        outcome = gamma_client._transform_outcome_data(outcome_data)
        
        assert isinstance(outcome, Outcome)
        assert outcome.id == "0xabc123"
        assert outcome.name == "Yes"
        assert outcome.price == Decimal("0.6500")
        assert outcome.volume == Decimal("1000000.50")
        assert outcome.liquidity == Decimal("250000.75")
    
    def test_invalid_market_data_handling(self, gamma_client):
        """Test handling of invalid market data."""
        invalid_data = {
            "id": "0x123",
            # Missing required fields
        }
        
        with pytest.raises(ValidationError):
            gamma_client._transform_market_data(invalid_data)
    
    def test_decimal_conversion_validation(self, gamma_client):
        """Test decimal conversion and validation."""
        # Valid decimal strings
        assert gamma_client._to_decimal("123.456") == Decimal("123.456")
        assert gamma_client._to_decimal("0") == Decimal("0")
        
        # Invalid decimal strings should raise ValidationError
        with pytest.raises(ValidationError):
            gamma_client._to_decimal("invalid")
        
        with pytest.raises(ValidationError):
            gamma_client._to_decimal("")
    
    def test_datetime_parsing(self, gamma_client):
        """Test datetime parsing from ISO strings."""
        iso_string = "2024-01-15T12:30:45Z"
        parsed_dt = gamma_client._parse_datetime(iso_string)
        
        assert isinstance(parsed_dt, datetime)
        assert parsed_dt.year == 2024
        assert parsed_dt.month == 1
        assert parsed_dt.day == 15
        
        # Invalid datetime should raise ValidationError
        with pytest.raises(ValidationError):
            gamma_client._parse_datetime("invalid-date")


@pytest.mark.integration
class TestGammaClientIntegration:
    """Integration tests for GammaClient."""
    
    @pytest.fixture
    def gamma_client(self):
        """Create client for integration tests."""
        config = PolymarketConfig(
            gamma_url="https://gamma-api.polymarket.com",
            api_key="test_key",
            private_key="test_private_key"
        )
        return GammaClient(config=config)
    
    @pytest.mark.asyncio
    async def test_full_market_workflow(self, gamma_client):
        """Test complete market discovery and analysis workflow."""
        with patch.object(gamma_client, '_make_request') as mock_request:
            # Mock market discovery
            mock_request.return_value = {
                "data": [
                    {
                        "id": "0x123",
                        "question": "Test market",
                        "status": "active",
                        "outcomes": [
                            {"id": "0xabc", "name": "Yes", "price": "0.5", "volume": "1000", "liquidity": "500"},
                            {"id": "0xdef", "name": "No", "price": "0.5", "volume": "1000", "liquidity": "500"}
                        ],
                        "volume": "2000",
                        "volume_24h": "100",
                        "liquidity": "1000",
                        "category": "Other",
                        "tags": ["test"],
                        "created_at": "2024-01-01T00:00:00Z",
                        "end_date": "2024-12-31T23:59:59Z",
                        "participants": 100,
                        "event_id": "event_1",
                        "creator": "0x999",
                        "fee_rate": "0.02"
                    }
                ]
            }
            
            # Discover markets
            markets = await gamma_client.get_markets(category="Other")
            assert len(markets) == 1
            
            market = markets[0]
            assert market.id == "0x123"
            
            # Mock detailed market data
            mock_request.return_value = {
                "data": {
                    "market_id": "0x123",
                    "volume_24h": "100.00",
                    "volume_7d": "700.00", 
                    "volume_30d": "3000.00",
                    "price_change_24h": "0.05",
                    "liquidity": "1000.00",
                    "spread": "0.02",
                    "participants": 100,
                    "trades_24h": 10,
                    "last_trade_price": "0.52",
                    "last_updated": "2024-01-15T12:00:00Z"
                }
            }
            
            # Get analytics
            analytics = await gamma_client.get_market_analytics(market.id)
            assert analytics.market_id == "0x123"
            assert analytics.volume_24h == Decimal("100.00")
    
    @pytest.mark.asyncio
    async def test_error_resilience(self, gamma_client):
        """Test client resilience to various error conditions."""
        with patch.object(gamma_client, '_make_request') as mock_request:
            # Test rate limiting
            mock_request.side_effect = RateLimitError(retry_after=1)
            
            with pytest.raises(RateLimitError):
                await gamma_client.get_markets()
            
            # Test recovery after error
            mock_request.side_effect = None
            mock_request.return_value = {"data": []}
            
            markets = await gamma_client.get_markets()
            assert markets == []