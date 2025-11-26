"""
Gamma Markets API Client

Client for Polymarket's Gamma Markets API providing enhanced market data:
- Market metadata retrieval
- Historical data fetching
- Market statistics and analytics
- Event information
- Data transformation and validation
"""

import asyncio
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Any, Optional

from .base import PolymarketClient
from .base import (
    PolymarketAPIError,
    RateLimitError,
    AuthenticationError,
    ValidationError,
)
from ..models.market import Market, MarketStatus, Outcome
from ..models.event import Event, EventCategory, EventStatus
from ..models.analytics import (
    MarketAnalytics,
    PriceHistory,
    PriceDataPoint,
    VolumeData,
    LiquidityMetrics,
)
from ..models.order import Trade, OrderSide
from ..utils import PolymarketConfig

logger = logging.getLogger(__name__)


class GammaAPIError(PolymarketAPIError):
    """Gamma API specific error"""
    pass


class MarketNotFoundError(GammaAPIError):
    """Market not found error"""
    pass


class InvalidDateRangeError(GammaAPIError):
    """Invalid date range error"""
    pass


class GammaClient(PolymarketClient):
    """
    Gamma Markets API client for enhanced market data
    
    Provides access to market discovery, historical data, analytics,
    and event information from Polymarket's Gamma API.
    """
    
    def __init__(
        self,
        config: Optional[PolymarketConfig] = None,
        cache_ttl: int = 300,
        cache_maxsize: int = 1000,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Initialize Gamma client
        
        Args:
            config: Client configuration
            cache_ttl: Cache time-to-live in seconds
            cache_maxsize: Maximum cache size
            max_retries: Maximum retry attempts
            retry_delay: Base retry delay in seconds
        """
        super().__init__(config, cache_ttl, cache_maxsize, max_retries, retry_delay)
        logger.info("Initialized GammaClient")
    
    def _get_base_url(self) -> str:
        """Get Gamma API base URL"""
        return self.config.gamma_url
    
    async def health_check(self) -> bool:
        """
        Check if Gamma API is healthy
        
        Returns:
            True if API is healthy, False otherwise
        """
        try:
            response = await self._make_request('GET', '/health', use_cache=False)
            return response.get('status') == 'healthy'
        except Exception as e:
            logger.error(f"Gamma API health check failed: {str(e)}")
            return False
    
    async def get_markets(
        self,
        limit: int = 10,
        offset: int = 0,
        category: Optional[str] = None,
        status: Optional[str] = None,
        tags: Optional[List[str]] = None,
        min_liquidity: Optional[float] = None,
        sort_by: str = "created_at",
        sort_order: str = "desc"
    ) -> List[Market]:
        """
        Get list of markets with filtering options
        
        Args:
            limit: Maximum number of markets to return
            offset: Number of markets to skip
            category: Filter by category
            status: Filter by status
            tags: Filter by tags
            min_liquidity: Minimum liquidity filter
            sort_by: Field to sort by
            sort_order: Sort order (asc/desc)
            
        Returns:
            List of Market objects
        """
        params = {
            'limit': limit,
            'offset': offset,
            'sort_by': sort_by,
            'sort_order': sort_order
        }
        
        if category:
            params['category'] = category
        if status:
            params['status'] = status
        if tags:
            params['tags'] = ','.join(tags)
        if min_liquidity is not None:
            params['min_liquidity'] = min_liquidity
        
        try:
            response = await self._make_request('GET', '/markets', params=params)
            markets_data = response.get('data', [])
            
            markets = []
            for market_data in markets_data:
                market = self._transform_market_data(market_data)
                markets.append(market)
            
            logger.info(f"Retrieved {len(markets)} markets from Gamma API")
            return markets
            
        except Exception as e:
            logger.error(f"Failed to get markets: {str(e)}")
            raise GammaAPIError(f"Failed to get markets: {str(e)}") from e
    
    async def get_market_by_id(self, market_id: str) -> Market:
        """
        Get detailed market information by ID
        
        Args:
            market_id: Market identifier
            
        Returns:
            Market object
            
        Raises:
            MarketNotFoundError: If market not found
        """
        try:
            response = await self._make_request('GET', f'/markets/{market_id}')
            market_data = response.get('data')
            
            if not market_data:
                raise MarketNotFoundError(f"Market {market_id} not found")
            
            market = self._transform_market_data(market_data)
            logger.debug(f"Retrieved market {market_id}")
            return market
            
        except PolymarketAPIError:
            raise
        except Exception as e:
            logger.error(f"Failed to get market {market_id}: {str(e)}")
            raise GammaAPIError(f"Failed to get market: {str(e)}") from e
    
    async def get_events(
        self,
        limit: int = 10,
        offset: int = 0,
        category: Optional[str] = None,
        status: Optional[str] = None
    ) -> List[Event]:
        """
        Get list of events
        
        Args:
            limit: Maximum number of events to return
            offset: Number of events to skip
            category: Filter by category
            status: Filter by status
            
        Returns:
            List of Event objects
        """
        params = {
            'limit': limit,
            'offset': offset
        }
        
        if category:
            params['category'] = category
        if status:
            params['status'] = status
        
        try:
            response = await self._make_request('GET', '/events', params=params)
            events_data = response.get('data', [])
            
            events = []
            for event_data in events_data:
                event = self._transform_event_data(event_data)
                events.append(event)
            
            logger.info(f"Retrieved {len(events)} events from Gamma API")
            return events
            
        except Exception as e:
            logger.error(f"Failed to get events: {str(e)}")
            raise GammaAPIError(f"Failed to get events: {str(e)}") from e
    
    async def get_event_by_id(self, event_id: str) -> Event:
        """
        Get detailed event information by ID
        
        Args:
            event_id: Event identifier
            
        Returns:
            Event object
        """
        try:
            response = await self._make_request('GET', f'/events/{event_id}')
            event_data = response.get('data')
            
            if not event_data:
                raise GammaAPIError(f"Event {event_id} not found")
            
            event = self._transform_event_data(event_data)
            logger.debug(f"Retrieved event {event_id}")
            return event
            
        except Exception as e:
            logger.error(f"Failed to get event {event_id}: {str(e)}")
            raise GammaAPIError(f"Failed to get event: {str(e)}") from e
    
    async def get_market_history(
        self,
        market_id: str,
        outcome_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        interval: str = "1h"
    ) -> PriceHistory:
        """
        Get historical price data for a market
        
        Args:
            market_id: Market identifier
            outcome_id: Specific outcome to get data for
            start_date: Start date for historical data
            end_date: End date for historical data
            interval: Data interval (1h, 4h, 1d, etc.)
            
        Returns:
            PriceHistory object
            
        Raises:
            InvalidDateRangeError: If date range is invalid
        """
        # Validate date range
        if start_date and end_date and start_date >= end_date:
            raise InvalidDateRangeError("Start date must be before end date")
        
        params = {'interval': interval}
        
        if start_date:
            params['start_date'] = start_date.isoformat()
        if end_date:
            params['end_date'] = end_date.isoformat()
        if outcome_id:
            params['outcome_id'] = outcome_id
        
        try:
            endpoint = f'/markets/{market_id}/prices'
            response = await self._make_request('GET', endpoint, params=params)
            
            history_data = response.get('data')
            if not history_data:
                raise GammaAPIError(f"No price history found for market {market_id}")
            
            # Transform price data points
            price_points = []
            for price_data in history_data.get('prices', []):
                point = PriceDataPoint(
                    timestamp=self._parse_datetime(price_data['timestamp']),
                    price=self._to_decimal(price_data['price']),
                    volume=self._to_decimal(price_data['volume'])
                )
                price_points.append(point)
            
            price_history = PriceHistory(
                market_id=history_data['market_id'],
                outcome_id=history_data.get('outcome_id', ''),
                prices=price_points,
                start_date=self._parse_datetime(history_data['start_date']),
                end_date=self._parse_datetime(history_data['end_date']),
                interval=history_data['interval']
            )
            
            logger.debug(f"Retrieved price history for market {market_id}")
            return price_history
            
        except PolymarketAPIError:
            raise
        except Exception as e:
            logger.error(f"Failed to get market history: {str(e)}")
            raise GammaAPIError(f"Failed to get market history: {str(e)}") from e
    
    async def get_market_analytics(self, market_id: str) -> MarketAnalytics:
        """
        Get comprehensive analytics for a market
        
        Args:
            market_id: Market identifier
            
        Returns:
            MarketAnalytics object
        """
        try:
            endpoint = f'/markets/{market_id}/analytics'
            response = await self._make_request('GET', endpoint)
            
            analytics_data = response.get('data')
            if not analytics_data:
                raise GammaAPIError(f"No analytics found for market {market_id}")
            
            analytics = MarketAnalytics(
                market_id=analytics_data['market_id'],
                volume_24h=self._to_decimal(analytics_data['volume_24h']),
                volume_7d=self._to_decimal(analytics_data.get('volume_7d', '0')),
                volume_30d=self._to_decimal(analytics_data.get('volume_30d', '0')),
                price_change_24h=self._to_decimal(analytics_data.get('price_change_24h', '0')),
                price_change_7d=self._to_decimal(analytics_data.get('price_change_7d', '0')),
                liquidity=self._to_decimal(analytics_data['liquidity']),
                spread=self._to_decimal(analytics_data['spread']),
                participants=analytics_data['participants'],
                trades_24h=analytics_data['trades_24h'],
                last_trade_price=self._to_decimal(analytics_data['last_trade_price']),
                last_updated=self._parse_datetime(analytics_data['last_updated'])
            )
            
            logger.debug(f"Retrieved analytics for market {market_id}")
            return analytics
            
        except Exception as e:
            logger.error(f"Failed to get market analytics: {str(e)}")
            raise GammaAPIError(f"Failed to get market analytics: {str(e)}") from e
    
    async def get_market_trades(
        self,
        market_id: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[Trade]:
        """
        Get recent trades for a market
        
        Args:
            market_id: Market identifier
            limit: Maximum number of trades to return
            offset: Number of trades to skip
            
        Returns:
            List of Trade objects
        """
        params = {
            'limit': limit,
            'offset': offset
        }
        
        try:
            endpoint = f'/markets/{market_id}/trades'
            response = await self._make_request('GET', endpoint, params=params)
            
            trades_data = response.get('data', [])
            trades = []
            
            for trade_data in trades_data:
                trade = Trade(
                    id=trade_data['id'],
                    market_id=trade_data['market_id'],
                    outcome_id=trade_data['outcome_id'],
                    order_id=trade_data.get('order_id'),
                    side=OrderSide(trade_data['side']),
                    price=float(trade_data['price']),
                    size=float(trade_data['size']),
                    timestamp=self._parse_datetime(trade_data['timestamp']),
                    fee=float(trade_data.get('fee', 0))
                )
                trades.append(trade)
            
            logger.debug(f"Retrieved {len(trades)} trades for market {market_id}")
            return trades
            
        except Exception as e:
            logger.error(f"Failed to get market trades: {str(e)}")
            raise GammaAPIError(f"Failed to get market trades: {str(e)}") from e
    
    async def search_markets(
        self,
        query: str,
        limit: int = 10,
        category: Optional[str] = None
    ) -> List[Market]:
        """
        Search markets by query string
        
        Args:
            query: Search query
            limit: Maximum number of results
            category: Filter by category
            
        Returns:
            List of Market objects
        """
        params = {
            'q': query,
            'limit': limit
        }
        
        if category:
            params['category'] = category
        
        try:
            response = await self._make_request('GET', '/markets/search', params=params)
            markets_data = response.get('data', [])
            
            markets = []
            for market_data in markets_data:
                market = self._transform_market_data(market_data)
                markets.append(market)
            
            logger.debug(f"Found {len(markets)} markets for query: {query}")
            return markets
            
        except Exception as e:
            logger.error(f"Failed to search markets: {str(e)}")
            raise GammaAPIError(f"Failed to search markets: {str(e)}") from e
    
    async def get_popular_markets(
        self,
        period: str = "24h",
        limit: int = 10
    ) -> List[Market]:
        """
        Get popular markets by volume or activity
        
        Args:
            period: Time period (24h, 7d, 30d)
            limit: Maximum number of markets
            
        Returns:
            List of Market objects
        """
        params = {
            'period': period,
            'limit': limit
        }
        
        try:
            response = await self._make_request('GET', '/markets/popular', params=params)
            markets_data = response.get('data', [])
            
            markets = []
            for market_data in markets_data:
                market = self._transform_market_data(market_data)
                markets.append(market)
            
            logger.debug(f"Retrieved {len(markets)} popular markets")
            return markets
            
        except Exception as e:
            logger.error(f"Failed to get popular markets: {str(e)}")
            raise GammaAPIError(f"Failed to get popular markets: {str(e)}") from e
    
    def _transform_market_data(self, data: Dict[str, Any]) -> Market:
        """
        Transform raw market data to Market object
        
        Args:
            data: Raw market data from API
            
        Returns:
            Market object
            
        Raises:
            ValidationError: If data is invalid
        """
        try:
            # Required fields
            market_id = data.get('id')
            question = data.get('question')
            if not market_id or not question:
                raise ValidationError("Market ID and question are required")
            
            # Parse outcomes
            outcomes = []
            outcomes_data = data.get('outcomes', [])
            if not outcomes_data:
                raise ValidationError("Market must have at least one outcome")
            
            for outcome_data in outcomes_data:
                outcome = self._transform_outcome_data(outcome_data)
                outcomes.append(outcome)
            
            # Parse dates
            created_at = self._parse_datetime(data.get('created_at', datetime.now().isoformat()))
            end_date = self._parse_datetime(data['end_date'])
            
            # Parse status
            status = MarketStatus(data.get('status', 'active'))
            
            # Create market
            market = Market(
                id=market_id,
                question=question,
                outcomes=[outcome.name for outcome in outcomes],
                end_date=end_date,
                status=status,
                created_at=created_at,
                volume=self._to_decimal(data.get('volume', '0')),
                volume_24h=self._to_decimal(data.get('volume_24h', '0'))
            )
            
            # Add pricing data
            for outcome in outcomes:
                market.current_prices[outcome.name] = outcome.price
            
            return market
            
        except Exception as e:
            logger.error(f"Failed to transform market data: {str(e)}")
            raise ValidationError(f"Invalid market data: {str(e)}") from e
    
    def _transform_outcome_data(self, data: Dict[str, Any]) -> Outcome:
        """
        Transform raw outcome data to Outcome object
        
        Args:
            data: Raw outcome data
            
        Returns:
            Outcome object
            
        Raises:
            ValidationError: If data is invalid
        """
        try:
            outcome = Outcome(
                id=data['id'],
                name=data['name'],
                price=self._to_decimal(data['price']),
                volume=self._to_decimal(data.get('volume', '0')),
                liquidity=self._to_decimal(data.get('liquidity', '0'))
            )
            return outcome
            
        except Exception as e:
            logger.error(f"Failed to transform outcome data: {str(e)}")
            raise ValidationError(f"Invalid outcome data: {str(e)}") from e
    
    def _transform_event_data(self, data: Dict[str, Any]) -> Event:
        """
        Transform raw event data to Event object
        
        Args:
            data: Raw event data
            
        Returns:
            Event object
            
        Raises:
            ValidationError: If data is invalid
        """
        try:
            # Parse dates
            created_at = self._parse_datetime(data['created_at'])
            end_date = None
            if data.get('end_date'):
                end_date = self._parse_datetime(data['end_date'])
            
            # Parse enums
            category = EventCategory(data.get('category', 'Other'))
            status = EventStatus(data.get('status', 'active'))
            
            event = Event(
                id=data['id'],
                title=data['title'],
                description=data['description'],
                category=category,
                status=status,
                created_at=created_at,
                end_date=end_date,
                tags=data.get('tags', []),
                image_url=data.get('image_url'),
                market_ids=data.get('market_ids', [])
            )
            
            return event
            
        except Exception as e:
            logger.error(f"Failed to transform event data: {str(e)}")
            raise ValidationError(f"Invalid event data: {str(e)}") from e
    
    def _to_decimal(self, value: Any) -> Decimal:
        """
        Convert value to Decimal with validation
        
        Args:
            value: Value to convert
            
        Returns:
            Decimal value
            
        Raises:
            ValidationError: If conversion fails
        """
        try:
            if isinstance(value, Decimal):
                return value
            return Decimal(str(value))
        except Exception as e:
            raise ValidationError(f"Invalid decimal value: {value}") from e
    
    def _parse_datetime(self, value: str) -> datetime:
        """
        Parse ISO datetime string
        
        Args:
            value: ISO datetime string
            
        Returns:
            datetime object
            
        Raises:
            ValidationError: If parsing fails
        """
        try:
            return datetime.fromisoformat(value.replace('Z', '+00:00'))
        except Exception as e:
            raise ValidationError(f"Invalid datetime format: {value}") from e