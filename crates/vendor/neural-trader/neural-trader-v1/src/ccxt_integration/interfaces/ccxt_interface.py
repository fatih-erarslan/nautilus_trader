"""
CCXT Interface Implementation

Implements the TradingAPIInterface for CCXT exchanges, providing a unified
interface for cryptocurrency trading across multiple exchanges.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import ccxt
import ccxt.async_support as ccxt_async
from dataclasses import dataclass, field
from enum import Enum

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from crypto_trading.interfaces import TradingAPIInterface, MarketDataInterface, AccountInterface

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order types supported by CCXT"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class TimeInForce(Enum):
    """Time in force options"""
    GTC = "GTC"  # Good Till Cancelled
    IOC = "IOC"  # Immediate Or Cancel
    FOK = "FOK"  # Fill Or Kill
    DAY = "DAY"  # Day order


@dataclass
class ExchangeConfig:
    """Configuration for a specific exchange"""
    name: str
    api_key: Optional[str] = None
    secret: Optional[str] = None
    password: Optional[str] = None  # For exchanges that require password
    sandbox: bool = False
    enable_rate_limit: bool = True
    options: Dict[str, Any] = field(default_factory=dict)


class CCXTInterface(TradingAPIInterface, MarketDataInterface, AccountInterface):
    """
    CCXT implementation of the TradingAPIInterface.
    
    Provides unified access to 100+ cryptocurrency exchanges through CCXT.
    """
    
    def __init__(self, config: ExchangeConfig):
        """
        Initialize CCXT interface with exchange configuration.
        
        Args:
            config: ExchangeConfig with exchange details
        """
        self.config = config
        self.exchange: Optional[ccxt_async.Exchange] = None
        self.sync_exchange: Optional[ccxt.Exchange] = None
        self._initialized = False
        self._market_cache: Dict[str, Any] = {}
        self._last_market_update: Optional[datetime] = None
        
    async def initialize(self) -> None:
        """Initialize the exchange connection"""
        if self._initialized:
            return
            
        try:
            # Get exchange class
            exchange_class = getattr(ccxt_async, self.config.name)
            
            # Prepare credentials
            credentials = {
                'enableRateLimit': self.config.enable_rate_limit,
                'options': self.config.options
            }
            
            if self.config.api_key:
                credentials['apiKey'] = self.config.api_key
            if self.config.secret:
                credentials['secret'] = self.config.secret
            if self.config.password:
                credentials['password'] = self.config.password
                
            # Set sandbox mode if enabled
            if self.config.sandbox:
                if 'test' in exchange_class.urls:
                    credentials['urls'] = {'api': exchange_class.urls['test']}
                    
            # Create exchange instance
            self.exchange = exchange_class(credentials)
            
            # Load markets
            await self.exchange.load_markets()
            self._market_cache = self.exchange.markets
            self._last_market_update = datetime.now()
            
            # Also create sync exchange for certain operations
            sync_class = getattr(ccxt, self.config.name)
            self.sync_exchange = sync_class(credentials)
            
            self._initialized = True
            logger.info(f"Initialized {self.config.name} exchange (sandbox: {self.config.sandbox})")
            
        except Exception as e:
            logger.error(f"Failed to initialize {self.config.name}: {str(e)}")
            raise
            
    async def close(self) -> None:
        """Close exchange connection"""
        if self.exchange:
            await self.exchange.close()
            self._initialized = False
            
    async def _ensure_initialized(self) -> None:
        """Ensure exchange is initialized"""
        if not self._initialized:
            await self.initialize()
            
    # TradingAPIInterface Implementation
    
    async def get_balance(self, asset: Optional[str] = None) -> Dict[str, Any]:
        """
        Get balance for a specific asset or all assets.
        
        Args:
            asset: Asset symbol (e.g., 'BTC', 'ETH') or None for all
            
        Returns:
            Dictionary with balance information
        """
        await self._ensure_initialized()
        
        try:
            balance = await self.exchange.fetch_balance()
            
            if asset:
                asset = asset.upper()
                if asset in balance:
                    return {
                        'asset': asset,
                        'free': balance[asset]['free'],
                        'used': balance[asset]['used'],
                        'total': balance[asset]['total']
                    }
                else:
                    return {
                        'asset': asset,
                        'free': 0,
                        'used': 0,
                        'total': 0
                    }
            
            # Return all non-zero balances
            result = {}
            for currency, bal in balance.items():
                if isinstance(bal, dict) and bal.get('total', 0) > 0:
                    result[currency] = {
                        'free': bal['free'],
                        'used': bal['used'],
                        'total': bal['total']
                    }
            
            return result
            
        except Exception as e:
            logger.error(f"Error fetching balance: {str(e)}")
            raise
            
    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Get current price ticker for a trading pair.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            
        Returns:
            Dictionary with ticker information
        """
        await self._ensure_initialized()
        
        try:
            ticker = await self.exchange.fetch_ticker(symbol)
            
            return {
                'symbol': ticker['symbol'],
                'bid': ticker['bid'],
                'ask': ticker['ask'],
                'last': ticker['last'],
                'volume': ticker['baseVolume'],
                'quote_volume': ticker['quoteVolume'],
                'timestamp': ticker['timestamp'],
                'datetime': ticker['datetime'],
                'high': ticker['high'],
                'low': ticker['low'],
                'open': ticker['open'],
                'close': ticker['close'],
                'change': ticker['change'],
                'percentage': ticker['percentage']
            }
            
        except Exception as e:
            logger.error(f"Error fetching ticker for {symbol}: {str(e)}")
            raise
            
    async def place_order(self, order_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Place a new order.
        
        Args:
            order_params: Dictionary with order parameters:
                - symbol: Trading pair (e.g., 'BTC/USDT')
                - type: Order type ('market', 'limit', etc.)
                - side: 'buy' or 'sell'
                - amount: Order amount
                - price: Limit price (for limit orders)
                - params: Additional exchange-specific parameters
                
        Returns:
            Dictionary with order result
        """
        await self._ensure_initialized()
        
        try:
            symbol = order_params['symbol']
            order_type = order_params['type']
            side = order_params['side']
            amount = order_params['amount']
            price = order_params.get('price')
            params = order_params.get('params', {})
            
            # Place order based on type
            if order_type == 'market':
                order = await self.exchange.create_market_order(
                    symbol, side, amount, params
                )
            elif order_type == 'limit':
                if not price:
                    raise ValueError("Price required for limit order")
                order = await self.exchange.create_limit_order(
                    symbol, side, amount, price, params
                )
            elif order_type == 'stop':
                if 'stopPrice' not in params:
                    raise ValueError("stopPrice required in params for stop order")
                order = await self.exchange.create_order(
                    symbol, 'stop', side, amount, None, params
                )
            elif order_type == 'stop_limit':
                if not price or 'stopPrice' not in params:
                    raise ValueError("Price and stopPrice required for stop-limit order")
                order = await self.exchange.create_order(
                    symbol, 'stop_limit', side, amount, price, params
                )
            else:
                # Generic order creation for other types
                order = await self.exchange.create_order(
                    symbol, order_type, side, amount, price, params
                )
                
            return {
                'id': order['id'],
                'symbol': order['symbol'],
                'type': order['type'],
                'side': order['side'],
                'amount': order['amount'],
                'price': order.get('price'),
                'status': order['status'],
                'filled': order.get('filled', 0),
                'remaining': order.get('remaining', order['amount']),
                'timestamp': order['timestamp'],
                'datetime': order['datetime'],
                'fee': order.get('fee'),
                'trades': order.get('trades', [])
            }
            
        except Exception as e:
            logger.error(f"Error placing order: {str(e)}")
            raise
            
    async def cancel_order(self, order_id: str, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Cancel an existing order.
        
        Args:
            order_id: Order identifier
            symbol: Trading pair symbol (required for some exchanges)
            
        Returns:
            Dictionary with cancellation result
        """
        await self._ensure_initialized()
        
        try:
            # Some exchanges require symbol for cancellation
            if symbol:
                result = await self.exchange.cancel_order(order_id, symbol)
            else:
                # Try to cancel without symbol (not all exchanges support this)
                result = await self.exchange.cancel_order(order_id)
                
            return {
                'id': result.get('id', order_id),
                'status': 'cancelled',
                'info': result.get('info', {})
            }
            
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {str(e)}")
            raise
            
    async def get_order_status(self, order_id: str, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Get status of an order.
        
        Args:
            order_id: Order identifier
            symbol: Trading pair symbol (required for some exchanges)
            
        Returns:
            Dictionary with order status
        """
        await self._ensure_initialized()
        
        try:
            if symbol:
                order = await self.exchange.fetch_order(order_id, symbol)
            else:
                # Try to fetch without symbol
                order = await self.exchange.fetch_order(order_id)
                
            return {
                'id': order['id'],
                'symbol': order['symbol'],
                'type': order['type'],
                'side': order['side'],
                'status': order['status'],
                'amount': order['amount'],
                'filled': order.get('filled', 0),
                'remaining': order.get('remaining'),
                'price': order.get('price'),
                'average': order.get('average'),
                'timestamp': order['timestamp'],
                'datetime': order['datetime']
            }
            
        except Exception as e:
            logger.error(f"Error fetching order status for {order_id}: {str(e)}")
            raise
            
    # MarketDataInterface Implementation
    
    async def get_orderbook(self, symbol: str, limit: int = 100) -> Dict[str, Any]:
        """Get order book data."""
        await self._ensure_initialized()
        
        try:
            orderbook = await self.exchange.fetch_order_book(symbol, limit)
            
            return {
                'symbol': symbol,
                'bids': orderbook['bids'],
                'asks': orderbook['asks'],
                'timestamp': orderbook['timestamp'],
                'datetime': orderbook['datetime']
            }
            
        except Exception as e:
            logger.error(f"Error fetching orderbook for {symbol}: {str(e)}")
            raise
            
    async def get_trades(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent trades."""
        await self._ensure_initialized()
        
        try:
            trades = await self.exchange.fetch_trades(symbol, limit=limit)
            
            return [
                {
                    'id': trade.get('id'),
                    'timestamp': trade['timestamp'],
                    'datetime': trade['datetime'],
                    'symbol': trade['symbol'],
                    'type': trade.get('type'),
                    'side': trade['side'],
                    'price': trade['price'],
                    'amount': trade['amount'],
                    'cost': trade.get('cost', trade['price'] * trade['amount'])
                }
                for trade in trades
            ]
            
        except Exception as e:
            logger.error(f"Error fetching trades for {symbol}: {str(e)}")
            raise
            
    async def get_klines(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 100
    ) -> List[List[Any]]:
        """
        Get candlestick/kline data.
        
        Args:
            symbol: Trading pair
            interval: Timeframe (e.g., '1m', '5m', '1h', '1d')
            start_time: Start timestamp in milliseconds
            end_time: End timestamp in milliseconds
            limit: Number of candles to fetch
            
        Returns:
            List of OHLCV data
        """
        await self._ensure_initialized()
        
        try:
            ohlcv = await self.exchange.fetch_ohlcv(
                symbol, 
                interval, 
                since=start_time,
                limit=limit
            )
            
            return ohlcv
            
        except Exception as e:
            logger.error(f"Error fetching klines for {symbol}: {str(e)}")
            raise
            
    # AccountInterface Implementation
    
    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information."""
        await self._ensure_initialized()
        
        try:
            # Fetch account status if available
            if hasattr(self.exchange, 'fetch_status'):
                status = await self.exchange.fetch_status()
            else:
                status = {'status': 'ok'}
                
            # Fetch trading fees
            if hasattr(self.exchange, 'fetch_trading_fees'):
                fees = await self.exchange.fetch_trading_fees()
            else:
                fees = {}
                
            return {
                'exchange': self.config.name,
                'status': status,
                'fees': fees,
                'sandbox': self.config.sandbox,
                'rate_limit': self.config.enable_rate_limit
            }
            
        except Exception as e:
            logger.error(f"Error fetching account info: {str(e)}")
            raise
            
    async def get_trade_history(
        self,
        symbol: Optional[str] = None,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get trade history."""
        await self._ensure_initialized()
        
        try:
            params = {}
            if start_time:
                params['since'] = start_time
                
            if symbol:
                trades = await self.exchange.fetch_my_trades(symbol, since=start_time, limit=limit, params=params)
            else:
                # Fetch trades for all symbols (if supported)
                trades = await self.exchange.fetch_my_trades(since=start_time, limit=limit, params=params)
                
            return [
                {
                    'id': trade.get('id'),
                    'order': trade.get('order'),
                    'symbol': trade['symbol'],
                    'type': trade.get('type'),
                    'side': trade['side'],
                    'price': trade['price'],
                    'amount': trade['amount'],
                    'cost': trade.get('cost'),
                    'fee': trade.get('fee'),
                    'timestamp': trade['timestamp'],
                    'datetime': trade['datetime']
                }
                for trade in trades
            ]
            
        except Exception as e:
            logger.error(f"Error fetching trade history: {str(e)}")
            raise
            
    async def get_order_history(
        self,
        symbol: Optional[str] = None,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get order history."""
        await self._ensure_initialized()
        
        try:
            params = {}
            if start_time:
                params['since'] = start_time
                
            if symbol:
                orders = await self.exchange.fetch_closed_orders(symbol, since=start_time, limit=limit, params=params)
            else:
                orders = await self.exchange.fetch_closed_orders(since=start_time, limit=limit, params=params)
                
            return [
                {
                    'id': order['id'],
                    'symbol': order['symbol'],
                    'type': order['type'],
                    'side': order['side'],
                    'status': order['status'],
                    'amount': order['amount'],
                    'filled': order.get('filled'),
                    'remaining': order.get('remaining'),
                    'price': order.get('price'),
                    'average': order.get('average'),
                    'timestamp': order['timestamp'],
                    'datetime': order['datetime']
                }
                for order in orders
            ]
            
        except Exception as e:
            logger.error(f"Error fetching order history: {str(e)}")
            raise
            
    # Additional CCXT-specific methods
    
    async def get_markets(self) -> Dict[str, Any]:
        """Get all available markets on the exchange."""
        await self._ensure_initialized()
        return self._market_cache
        
    async def get_currencies(self) -> Dict[str, Any]:
        """Get all available currencies on the exchange."""
        await self._ensure_initialized()
        
        if hasattr(self.exchange, 'currencies'):
            return self.exchange.currencies
        return {}
        
    async def fetch_funding_rate(self, symbol: str) -> Dict[str, Any]:
        """Fetch funding rate for perpetual contracts."""
        await self._ensure_initialized()
        
        if hasattr(self.exchange, 'fetch_funding_rate'):
            return await self.exchange.fetch_funding_rate(symbol)
        raise NotImplementedError(f"{self.config.name} does not support funding rates")
        
    async def fetch_positions(self) -> List[Dict[str, Any]]:
        """Fetch open positions (for derivatives)."""
        await self._ensure_initialized()
        
        if hasattr(self.exchange, 'fetch_positions'):
            return await self.exchange.fetch_positions()
        raise NotImplementedError(f"{self.config.name} does not support position fetching")