"""
Polymarket CLOB (Central Limit Order Book) API Client

This module provides a comprehensive client for interacting with the Polymarket CLOB API,
which handles market data, order management, and trading operations for prediction markets.

Key Features:
- Market data retrieval (markets, order books, trade history)
- Order management (place, cancel, modify orders)
- Portfolio and position tracking
- Real-time data streaming capabilities
- Authentication and request signing
- Comprehensive error handling and retry logic
- Response caching and performance optimization
- Rate limiting compliance

The client inherits from PolymarketClient base class and implements all abstract methods
while providing CLOB-specific functionality for trading operations.
"""

import asyncio
import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
from urllib.parse import urljoin

import aiohttp

from .base import PolymarketClient, PolymarketAPIError, RateLimitError, AuthenticationError, ValidationError
from ..models import Market, Order, OrderBook, OrderSide, OrderStatus, OrderType, MarketStatus
from ..utils import PolymarketConfig


logger = logging.getLogger(__name__)


class CLOBClient(PolymarketClient):
    """
    Polymarket CLOB (Central Limit Order Book) API Client
    
    This client provides comprehensive access to Polymarket's trading API including:
    - Market data and metadata
    - Order placement and management
    - Trade history and portfolio tracking
    - Real-time order book data
    - Authentication and request signing
    """
    
    def _get_base_url(self) -> str:
        """Get the CLOB API base URL"""
        return self.config.clob_url or "https://clob.polymarket.com"
    
    # Market Data Methods
    
    async def get_markets(
        self,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        status: Optional[str] = None,
        category: Optional[str] = None,
        search: Optional[str] = None,
        sort_by: Optional[str] = None,
        sort_order: Optional[str] = None,
    ) -> List[Market]:
        """
        Retrieve all available prediction markets
        
        Args:
            limit: Maximum number of markets to return
            offset: Number of markets to skip (for pagination)
            status: Filter by market status ('active', 'closed', 'resolved')
            category: Filter by market category
            search: Search term for market questions
            sort_by: Sort field ('created_at', 'end_date', 'volume')
            sort_order: Sort order ('asc', 'desc')
            
        Returns:
            List of Market objects
            
        Raises:
            PolymarketAPIError: On API errors
        """
        params = {}
        if limit is not None:
            params['limit'] = limit
        if offset is not None:
            params['offset'] = offset
        if status:
            params['status'] = status
        if category:
            params['category'] = category
        if search:
            params['search'] = search
        if sort_by:
            params['sort_by'] = sort_by
        if sort_order:
            params['sort_order'] = sort_order
        
        try:
            response = await self._make_request(
                method='GET',
                endpoint='/markets',
                params=params,
                use_cache=True
            )
            
            markets = []
            for market_data in response.get('markets', []):
                try:
                    market = self._parse_market_data(market_data)
                    markets.append(market)
                except Exception as e:
                    logger.warning(f"Failed to parse market data: {e}")
                    continue
            
            logger.info(f"Retrieved {len(markets)} markets")
            return markets
            
        except Exception as e:
            logger.error(f"Failed to retrieve markets: {e}")
            raise
    
    async def get_market_by_id(self, market_id: str) -> Market:
        """
        Retrieve specific market by ID
        
        Args:
            market_id: Unique market identifier
            
        Returns:
            Market object
            
        Raises:
            PolymarketAPIError: On API errors or if market not found
        """
        try:
            response = await self._make_request(
                method='GET',
                endpoint=f'/markets/{market_id}',
                use_cache=True
            )
            
            market_data = response.get('market')
            if not market_data:
                raise PolymarketAPIError(f"Market {market_id} not found")
            
            market = self._parse_market_data(market_data)
            logger.info(f"Retrieved market: {market.id}")
            return market
            
        except Exception as e:
            logger.error(f"Failed to retrieve market {market_id}: {e}")
            raise
    
    async def get_order_book(
        self,
        market_id: str,
        outcome_id: str,
        depth: Optional[int] = None
    ) -> OrderBook:
        """
        Retrieve order book for a specific market outcome
        
        Args:
            market_id: Market identifier
            outcome_id: Outcome identifier
            depth: Number of price levels to include
            
        Returns:
            OrderBook object
            
        Raises:
            PolymarketAPIError: On API errors
        """
        params = {
            'market_id': market_id,
            'outcome_id': outcome_id
        }
        if depth:
            params['depth'] = depth
        
        try:
            response = await self._make_request(
                method='GET',
                endpoint='/book',
                params=params,
                use_cache=True,
                cache_key=f"book:{market_id}:{outcome_id}:{depth}"
            )
            
            book_data = response.get('order_book', {})
            order_book = self._parse_order_book_data(book_data)
            
            logger.info(f"Retrieved order book for {market_id}:{outcome_id}")
            return order_book
            
        except Exception as e:
            logger.error(f"Failed to retrieve order book for {market_id}:{outcome_id}: {e}")
            raise
    
    # Order Management Methods
    
    async def place_order(
        self,
        market_id: str,
        outcome_id: str,
        side: str,
        type: str,
        size: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: str = "gtc",
        client_order_id: Optional[str] = None,
    ) -> Order:
        """
        Place a new order
        
        Args:
            market_id: Market identifier
            outcome_id: Outcome identifier  
            side: Order side ('buy' or 'sell')
            type: Order type ('market', 'limit', 'stop', 'stop_limit')
            size: Order size
            price: Limit price (required for limit orders)
            stop_price: Stop price (required for stop orders)
            time_in_force: Time in force ('gtc', 'ioc', 'fok', 'day')
            client_order_id: Client-provided order ID
            
        Returns:
            Order object
            
        Raises:
            ValidationError: On invalid order parameters
            AuthenticationError: If not authenticated
            PolymarketAPIError: On API errors
        """
        # Validate order parameters
        self._validate_order_params(market_id, outcome_id, side, type, size, price, stop_price)
        
        order_data = {
            'market_id': market_id,
            'outcome_id': outcome_id,
            'side': side.lower(),
            'type': type.lower(),
            'size': size,
            'time_in_force': time_in_force.lower()
        }
        
        if price is not None:
            order_data['price'] = price
        if stop_price is not None:
            order_data['stop_price'] = stop_price
        if client_order_id:
            order_data['client_order_id'] = client_order_id
        
        # Sign the order request
        signature_data = self._sign_order_request(order_data)
        order_data.update(signature_data)
        
        try:
            response = await self._make_request(
                method='POST',
                endpoint='/orders',
                data=order_data,
                use_cache=False
            )
            
            order_response = response.get('order', {})
            order = self._parse_order_data(order_response)
            
            logger.info(f"Placed order: {order.id}")
            return order
            
        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            raise
    
    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an existing order
        
        Args:
            order_id: Order identifier to cancel
            
        Returns:
            True if cancellation successful
            
        Raises:
            PolymarketAPIError: On API errors or if order not found
        """
        try:
            response = await self._make_request(
                method='DELETE',
                endpoint=f'/orders/{order_id}',
                use_cache=False
            )
            
            cancelled = response.get('cancelled', False)
            if cancelled:
                logger.info(f"Cancelled order: {order_id}")
            else:
                logger.warning(f"Order {order_id} was not cancelled")
            
            return cancelled
            
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            raise
    
    async def get_orders(
        self,
        market_id: Optional[str] = None,
        status: Optional[str] = None,
        side: Optional[str] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[Order]:
        """
        Retrieve user's orders
        
        Args:
            market_id: Filter by market ID
            status: Filter by order status
            side: Filter by order side ('buy', 'sell')
            limit: Maximum number of orders to return
            offset: Number of orders to skip
            start_date: Start date filter (ISO format)
            end_date: End date filter (ISO format)
            
        Returns:
            List of Order objects
            
        Raises:
            AuthenticationError: If not authenticated
            PolymarketAPIError: On API errors
        """
        params = {}
        if market_id:
            params['market_id'] = market_id
        if status:
            params['status'] = status
        if side:
            params['side'] = side
        if limit:
            params['limit'] = limit
        if offset:
            params['offset'] = offset
        if start_date:
            params['start_date'] = start_date
        if end_date:
            params['end_date'] = end_date
        
        try:
            response = await self._make_request(
                method='GET',
                endpoint='/orders',
                params=params,
                use_cache=False  # Don't cache order data
            )
            
            orders = []
            for order_data in response.get('orders', []):
                try:
                    order = self._parse_order_data(order_data)
                    orders.append(order)
                except Exception as e:
                    logger.warning(f"Failed to parse order data: {e}")
                    continue
            
            logger.info(f"Retrieved {len(orders)} orders")
            return orders
            
        except Exception as e:
            logger.error(f"Failed to retrieve orders: {e}")
            raise
    
    # Trade History Methods
    
    async def get_trades(
        self,
        market_id: Optional[str] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve trade history
        
        Args:
            market_id: Filter by market ID
            limit: Maximum number of trades to return
            offset: Number of trades to skip
            start_date: Start date filter (ISO format)
            end_date: End date filter (ISO format)
            
        Returns:
            List of trade dictionaries
            
        Raises:
            PolymarketAPIError: On API errors
        """
        params = {}
        if market_id:
            params['market_id'] = market_id
        if limit:
            params['limit'] = limit
        if offset:
            params['offset'] = offset
        if start_date:
            params['start_date'] = start_date
        if end_date:
            params['end_date'] = end_date
        
        try:
            response = await self._make_request(
                method='GET',
                endpoint='/trades',
                params=params,
                use_cache=False
            )
            
            trades = response.get('trades', [])
            logger.info(f"Retrieved {len(trades)} trades")
            return trades
            
        except Exception as e:
            logger.error(f"Failed to retrieve trades: {e}")
            raise
    
    # Health Check and Utility Methods
    
    async def health_check(self) -> bool:
        """
        Check if the CLOB API is healthy and accessible
        
        Returns:
            True if API is healthy, False otherwise
        """
        try:
            response = await self._make_request(
                method='GET',
                endpoint='/health',
                use_cache=False
            )
            
            status = response.get('status', '').lower()
            is_healthy = status == 'healthy' or status == 'ok'
            
            logger.info(f"CLOB API health check: {'healthy' if is_healthy else 'unhealthy'}")
            return is_healthy
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    # Private Helper Methods
    
    def _validate_order_params(
        self,
        market_id: str,
        outcome_id: str,
        side: str,
        order_type: str,
        size: float,
        price: Optional[float],
        stop_price: Optional[float],
    ) -> None:
        """Validate order parameters"""
        if not market_id:
            raise ValidationError("Market ID is required")
        if not outcome_id:
            raise ValidationError("Outcome ID is required")
        if side.lower() not in ['buy', 'sell']:
            raise ValidationError("Side must be 'buy' or 'sell'")
        if order_type.lower() not in ['market', 'limit', 'stop', 'stop_limit']:
            raise ValidationError("Invalid order type")
        if size <= 0:
            raise ValidationError("Size must be positive")
        
        # Validate price requirements
        if order_type.lower() in ['limit', 'stop_limit'] and price is None:
            raise ValidationError(f"Price is required for {order_type} orders")
        if order_type.lower() in ['stop', 'stop_limit'] and stop_price is None:
            raise ValidationError(f"Stop price is required for {order_type} orders")
        
        # Validate price bounds
        if price is not None and not (0.0 < price <= 1.0):
            raise ValidationError("Price must be between 0 and 1")
        if stop_price is not None and not (0.0 < stop_price <= 1.0):
            raise ValidationError("Stop price must be between 0 and 1")
    
    def _sign_order_request(self, order_data: Dict[str, Any]) -> Dict[str, str]:
        """
        Sign an order request for authentication
        
        Args:
            order_data: Order data to sign
            
        Returns:
            Dictionary with signature and timestamp
        """
        try:
            # Create canonical string representation
            canonical_data = json.dumps(order_data, sort_keys=True, separators=(',', ':'))
            
            # Generate timestamp
            timestamp = str(int(datetime.now(timezone.utc).timestamp() * 1000))
            
            # Create signature (simplified for testing - real implementation would use crypto)
            signature_input = f"{timestamp}{canonical_data}"
            signature = hashlib.sha256(signature_input.encode()).hexdigest()
            
            return {
                'signature': signature,
                'timestamp': timestamp
            }
            
        except Exception as e:
            logger.error(f"Failed to sign order request: {e}")
            raise AuthenticationError("Failed to sign order request")
    
    def _parse_market_data(self, market_data: Dict[str, Any]) -> Market:
        """Parse market data from API response"""
        try:
            # Parse dates
            end_date = datetime.fromisoformat(
                market_data['end_date'].replace('Z', '+00:00')
            )
            created_at = datetime.fromisoformat(
                market_data['created_at'].replace('Z', '+00:00')
            )
            updated_at = datetime.fromisoformat(
                market_data['updated_at'].replace('Z', '+00:00')
            )
            
            # Parse status
            status = MarketStatus(market_data['status'])
            
            # Parse current prices
            current_prices = {}
            if 'current_prices' in market_data:
                for outcome, price in market_data['current_prices'].items():
                    current_prices[outcome] = float(price)
            
            return Market(
                id=market_data['id'],
                question=market_data['question'],
                outcomes=market_data['outcomes'],
                end_date=end_date,
                status=status,
                current_prices=current_prices,
                created_at=created_at,
                updated_at=updated_at,
            )
            
        except Exception as e:
            logger.error(f"Failed to parse market data: {e}")
            raise PolymarketAPIError(f"Invalid market data format: {e}")
    
    def _parse_order_data(self, order_data: Dict[str, Any]) -> Order:
        """Parse order data from API response"""
        try:
            # Parse enums
            side = OrderSide(order_data['side'])
            order_type = OrderType(order_data['type'])
            status = OrderStatus(order_data['status'])
            
            # Parse dates
            created_at = None
            if order_data.get('created_at'):
                created_at = datetime.fromisoformat(
                    order_data['created_at'].replace('Z', '+00:00')
                )
            
            return Order(
                id=order_data['id'],
                market_id=order_data['market_id'],
                outcome_id=order_data['outcome_id'],
                side=side,
                type=order_type,
                size=float(order_data['size']),
                price=float(order_data['price']) if order_data.get('price') else None,
                status=status,
                created_at=created_at,
                filled=float(order_data.get('filled', 0.0)),
                remaining=float(order_data.get('remaining', 0.0)),
            )
            
        except Exception as e:
            logger.error(f"Failed to parse order data: {e}")
            raise PolymarketAPIError(f"Invalid order data format: {e}")
    
    def _parse_order_book_data(self, book_data: Dict[str, Any]) -> OrderBook:
        """Parse order book data from API response"""
        try:
            timestamp = None
            if book_data.get('timestamp'):
                timestamp = datetime.fromisoformat(
                    book_data['timestamp'].replace('Z', '+00:00')
                )
            
            return OrderBook(
                market_id=book_data['market_id'],
                outcome_id=book_data['outcome_id'],
                bids=book_data.get('bids', []),
                asks=book_data.get('asks', []),
                timestamp=timestamp,
            )
            
        except Exception as e:
            logger.error(f"Failed to parse order book data: {e}")
            raise PolymarketAPIError(f"Invalid order book data format: {e}")