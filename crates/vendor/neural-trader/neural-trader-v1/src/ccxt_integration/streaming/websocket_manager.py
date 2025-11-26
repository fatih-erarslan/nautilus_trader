"""
WebSocket Streaming Manager for CCXT Pro

Handles real-time data streaming from cryptocurrency exchanges using CCXT Pro.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json

try:
    import ccxt.pro as ccxt_pro
except ImportError:
    ccxt_pro = None
    logging.warning("CCXT Pro not installed. WebSocket features will be limited.")

logger = logging.getLogger(__name__)


class StreamType(Enum):
    """Types of data streams"""
    TICKER = "ticker"
    ORDERBOOK = "orderbook"
    TRADES = "trades"
    OHLCV = "ohlcv"
    BALANCE = "balance"
    ORDERS = "orders"
    POSITIONS = "positions"


@dataclass
class StreamSubscription:
    """Represents a stream subscription"""
    stream_type: StreamType
    symbol: str
    callback: Callable[[Dict[str, Any]], None]
    params: Dict[str, Any] = field(default_factory=dict)
    active: bool = True
    last_update: Optional[datetime] = None
    error_count: int = 0


class WebSocketManager:
    """
    Manages WebSocket connections for real-time data streaming using CCXT Pro.
    """
    
    def __init__(self, exchange_name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize WebSocket manager.
        
        Args:
            exchange_name: Name of the exchange (e.g., 'binance', 'coinbase')
            config: Optional configuration including API credentials
        """
        if not ccxt_pro:
            raise ImportError("CCXT Pro is required for WebSocket functionality")
            
        self.exchange_name = exchange_name
        self.config = config or {}
        self.exchange: Optional[ccxt_pro.Exchange] = None
        self.subscriptions: Dict[str, StreamSubscription] = {}
        self.active_streams: Set[asyncio.Task] = set()
        self._running = False
        self._reconnect_delay = 5  # seconds
        self._max_reconnect_attempts = 10
        
    async def initialize(self) -> None:
        """Initialize the WebSocket connection."""
        try:
            # Get exchange class from ccxt.pro
            exchange_class = getattr(ccxt_pro, self.exchange_name)
            
            # Create exchange instance with config
            self.exchange = exchange_class(self.config)
            
            # Enable WebSocket
            self.exchange.enableRateLimit = True
            
            self._running = True
            logger.info(f"Initialized WebSocket manager for {self.exchange_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize WebSocket for {self.exchange_name}: {str(e)}")
            raise
            
    async def close(self) -> None:
        """Close all WebSocket connections."""
        self._running = False
        
        # Cancel all active streams
        for task in self.active_streams:
            task.cancel()
            
        # Wait for all tasks to complete
        if self.active_streams:
            await asyncio.gather(*self.active_streams, return_exceptions=True)
            
        # Close exchange connection
        if self.exchange:
            await self.exchange.close()
            
        logger.info(f"Closed WebSocket manager for {self.exchange_name}")
        
    async def subscribe_ticker(
        self,
        symbol: str,
        callback: Callable[[Dict[str, Any]], None]
    ) -> str:
        """
        Subscribe to ticker updates for a symbol.
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            callback: Function to call with ticker data
            
        Returns:
            Subscription ID
        """
        subscription_id = f"ticker_{symbol}_{id(callback)}"
        
        subscription = StreamSubscription(
            stream_type=StreamType.TICKER,
            symbol=symbol,
            callback=callback
        )
        
        self.subscriptions[subscription_id] = subscription
        
        # Start streaming task
        task = asyncio.create_task(self._stream_ticker(subscription_id))
        self.active_streams.add(task)
        task.add_done_callback(self.active_streams.discard)
        
        logger.info(f"Subscribed to ticker for {symbol}")
        return subscription_id
        
    async def subscribe_orderbook(
        self,
        symbol: str,
        callback: Callable[[Dict[str, Any]], None],
        limit: int = 100
    ) -> str:
        """
        Subscribe to order book updates for a symbol.
        
        Args:
            symbol: Trading pair
            callback: Function to call with orderbook data
            limit: Depth of order book
            
        Returns:
            Subscription ID
        """
        subscription_id = f"orderbook_{symbol}_{id(callback)}"
        
        subscription = StreamSubscription(
            stream_type=StreamType.ORDERBOOK,
            symbol=symbol,
            callback=callback,
            params={'limit': limit}
        )
        
        self.subscriptions[subscription_id] = subscription
        
        # Start streaming task
        task = asyncio.create_task(self._stream_orderbook(subscription_id))
        self.active_streams.add(task)
        task.add_done_callback(self.active_streams.discard)
        
        logger.info(f"Subscribed to orderbook for {symbol}")
        return subscription_id
        
    async def subscribe_trades(
        self,
        symbol: str,
        callback: Callable[[Dict[str, Any]], None]
    ) -> str:
        """
        Subscribe to trade updates for a symbol.
        
        Args:
            symbol: Trading pair
            callback: Function to call with trade data
            
        Returns:
            Subscription ID
        """
        subscription_id = f"trades_{symbol}_{id(callback)}"
        
        subscription = StreamSubscription(
            stream_type=StreamType.TRADES,
            symbol=symbol,
            callback=callback
        )
        
        self.subscriptions[subscription_id] = subscription
        
        # Start streaming task
        task = asyncio.create_task(self._stream_trades(subscription_id))
        self.active_streams.add(task)
        task.add_done_callback(self.active_streams.discard)
        
        logger.info(f"Subscribed to trades for {symbol}")
        return subscription_id
        
    async def subscribe_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        callback: Callable[[Dict[str, Any]], None]
    ) -> str:
        """
        Subscribe to OHLCV candle updates.
        
        Args:
            symbol: Trading pair
            timeframe: Candle timeframe (e.g., '1m', '5m', '1h')
            callback: Function to call with OHLCV data
            
        Returns:
            Subscription ID
        """
        subscription_id = f"ohlcv_{symbol}_{timeframe}_{id(callback)}"
        
        subscription = StreamSubscription(
            stream_type=StreamType.OHLCV,
            symbol=symbol,
            callback=callback,
            params={'timeframe': timeframe}
        )
        
        self.subscriptions[subscription_id] = subscription
        
        # Start streaming task
        task = asyncio.create_task(self._stream_ohlcv(subscription_id))
        self.active_streams.add(task)
        task.add_done_callback(self.active_streams.discard)
        
        logger.info(f"Subscribed to OHLCV for {symbol} {timeframe}")
        return subscription_id
        
    async def subscribe_balance(
        self,
        callback: Callable[[Dict[str, Any]], None]
    ) -> str:
        """
        Subscribe to balance updates.
        
        Args:
            callback: Function to call with balance data
            
        Returns:
            Subscription ID
        """
        subscription_id = f"balance_{id(callback)}"
        
        subscription = StreamSubscription(
            stream_type=StreamType.BALANCE,
            symbol='*',  # All symbols
            callback=callback
        )
        
        self.subscriptions[subscription_id] = subscription
        
        # Start streaming task
        task = asyncio.create_task(self._stream_balance(subscription_id))
        self.active_streams.add(task)
        task.add_done_callback(self.active_streams.discard)
        
        logger.info("Subscribed to balance updates")
        return subscription_id
        
    async def subscribe_orders(
        self,
        symbol: Optional[str],
        callback: Callable[[Dict[str, Any]], None]
    ) -> str:
        """
        Subscribe to order updates.
        
        Args:
            symbol: Trading pair or None for all symbols
            callback: Function to call with order data
            
        Returns:
            Subscription ID
        """
        subscription_id = f"orders_{symbol or 'all'}_{id(callback)}"
        
        subscription = StreamSubscription(
            stream_type=StreamType.ORDERS,
            symbol=symbol or '*',
            callback=callback
        )
        
        self.subscriptions[subscription_id] = subscription
        
        # Start streaming task
        task = asyncio.create_task(self._stream_orders(subscription_id))
        self.active_streams.add(task)
        task.add_done_callback(self.active_streams.discard)
        
        logger.info(f"Subscribed to order updates for {symbol or 'all'}")
        return subscription_id
        
    async def unsubscribe(self, subscription_id: str) -> None:
        """
        Unsubscribe from a stream.
        
        Args:
            subscription_id: ID of the subscription to cancel
        """
        if subscription_id in self.subscriptions:
            self.subscriptions[subscription_id].active = False
            del self.subscriptions[subscription_id]
            logger.info(f"Unsubscribed from {subscription_id}")
            
    async def _stream_ticker(self, subscription_id: str) -> None:
        """Stream ticker data."""
        subscription = self.subscriptions.get(subscription_id)
        if not subscription:
            return
            
        reconnect_attempts = 0
        
        while self._running and subscription.active:
            try:
                ticker = await self.exchange.watch_ticker(subscription.symbol)
                
                # Update subscription
                subscription.last_update = datetime.now()
                subscription.error_count = 0
                reconnect_attempts = 0
                
                # Call callback
                subscription.callback(ticker)
                
            except Exception as e:
                subscription.error_count += 1
                reconnect_attempts += 1
                
                logger.error(f"Error in ticker stream for {subscription.symbol}: {str(e)}")
                
                if reconnect_attempts >= self._max_reconnect_attempts:
                    logger.error(f"Max reconnect attempts reached for ticker {subscription.symbol}")
                    subscription.active = False
                    break
                    
                await asyncio.sleep(self._reconnect_delay * reconnect_attempts)
                
    async def _stream_orderbook(self, subscription_id: str) -> None:
        """Stream orderbook data."""
        subscription = self.subscriptions.get(subscription_id)
        if not subscription:
            return
            
        reconnect_attempts = 0
        limit = subscription.params.get('limit', 100)
        
        while self._running and subscription.active:
            try:
                orderbook = await self.exchange.watch_order_book(
                    subscription.symbol,
                    limit
                )
                
                # Update subscription
                subscription.last_update = datetime.now()
                subscription.error_count = 0
                reconnect_attempts = 0
                
                # Call callback
                subscription.callback(orderbook)
                
            except Exception as e:
                subscription.error_count += 1
                reconnect_attempts += 1
                
                logger.error(f"Error in orderbook stream for {subscription.symbol}: {str(e)}")
                
                if reconnect_attempts >= self._max_reconnect_attempts:
                    logger.error(f"Max reconnect attempts reached for orderbook {subscription.symbol}")
                    subscription.active = False
                    break
                    
                await asyncio.sleep(self._reconnect_delay * reconnect_attempts)
                
    async def _stream_trades(self, subscription_id: str) -> None:
        """Stream trades data."""
        subscription = self.subscriptions.get(subscription_id)
        if not subscription:
            return
            
        reconnect_attempts = 0
        
        while self._running and subscription.active:
            try:
                trades = await self.exchange.watch_trades(subscription.symbol)
                
                # Update subscription
                subscription.last_update = datetime.now()
                subscription.error_count = 0
                reconnect_attempts = 0
                
                # Call callback with each trade
                for trade in trades:
                    subscription.callback(trade)
                    
            except Exception as e:
                subscription.error_count += 1
                reconnect_attempts += 1
                
                logger.error(f"Error in trades stream for {subscription.symbol}: {str(e)}")
                
                if reconnect_attempts >= self._max_reconnect_attempts:
                    logger.error(f"Max reconnect attempts reached for trades {subscription.symbol}")
                    subscription.active = False
                    break
                    
                await asyncio.sleep(self._reconnect_delay * reconnect_attempts)
                
    async def _stream_ohlcv(self, subscription_id: str) -> None:
        """Stream OHLCV data."""
        subscription = self.subscriptions.get(subscription_id)
        if not subscription:
            return
            
        reconnect_attempts = 0
        timeframe = subscription.params.get('timeframe', '1m')
        
        while self._running and subscription.active:
            try:
                ohlcv = await self.exchange.watch_ohlcv(
                    subscription.symbol,
                    timeframe
                )
                
                # Update subscription
                subscription.last_update = datetime.now()
                subscription.error_count = 0
                reconnect_attempts = 0
                
                # Call callback
                subscription.callback(ohlcv)
                
            except Exception as e:
                subscription.error_count += 1
                reconnect_attempts += 1
                
                logger.error(f"Error in OHLCV stream for {subscription.symbol}: {str(e)}")
                
                if reconnect_attempts >= self._max_reconnect_attempts:
                    logger.error(f"Max reconnect attempts reached for OHLCV {subscription.symbol}")
                    subscription.active = False
                    break
                    
                await asyncio.sleep(self._reconnect_delay * reconnect_attempts)
                
    async def _stream_balance(self, subscription_id: str) -> None:
        """Stream balance updates."""
        subscription = self.subscriptions.get(subscription_id)
        if not subscription:
            return
            
        reconnect_attempts = 0
        
        while self._running and subscription.active:
            try:
                balance = await self.exchange.watch_balance()
                
                # Update subscription
                subscription.last_update = datetime.now()
                subscription.error_count = 0
                reconnect_attempts = 0
                
                # Call callback
                subscription.callback(balance)
                
            except Exception as e:
                subscription.error_count += 1
                reconnect_attempts += 1
                
                logger.error(f"Error in balance stream: {str(e)}")
                
                if reconnect_attempts >= self._max_reconnect_attempts:
                    logger.error("Max reconnect attempts reached for balance stream")
                    subscription.active = False
                    break
                    
                await asyncio.sleep(self._reconnect_delay * reconnect_attempts)
                
    async def _stream_orders(self, subscription_id: str) -> None:
        """Stream order updates."""
        subscription = self.subscriptions.get(subscription_id)
        if not subscription:
            return
            
        reconnect_attempts = 0
        
        while self._running and subscription.active:
            try:
                if subscription.symbol == '*':
                    orders = await self.exchange.watch_orders()
                else:
                    orders = await self.exchange.watch_orders(subscription.symbol)
                    
                # Update subscription
                subscription.last_update = datetime.now()
                subscription.error_count = 0
                reconnect_attempts = 0
                
                # Call callback
                subscription.callback(orders)
                
            except Exception as e:
                subscription.error_count += 1
                reconnect_attempts += 1
                
                logger.error(f"Error in orders stream: {str(e)}")
                
                if reconnect_attempts >= self._max_reconnect_attempts:
                    logger.error("Max reconnect attempts reached for orders stream")
                    subscription.active = False
                    break
                    
                await asyncio.sleep(self._reconnect_delay * reconnect_attempts)
                
    def get_subscription_status(self) -> Dict[str, Any]:
        """
        Get status of all subscriptions.
        
        Returns:
            Dictionary with subscription information
        """
        status = {
            'exchange': self.exchange_name,
            'running': self._running,
            'active_streams': len(self.active_streams),
            'subscriptions': {}
        }
        
        for sub_id, subscription in self.subscriptions.items():
            status['subscriptions'][sub_id] = {
                'type': subscription.stream_type.value,
                'symbol': subscription.symbol,
                'active': subscription.active,
                'last_update': subscription.last_update.isoformat() if subscription.last_update else None,
                'error_count': subscription.error_count
            }
            
        return status