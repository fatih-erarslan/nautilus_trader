"""
Alpha Vantage Trading API Implementation for German Stocks
Implements the TradingAPIInterface for Alpha Vantage with German market support
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime

from ..base.api_interface import (
    TradingAPIInterface, OrderRequest, OrderResponse, 
    MarketData, AccountBalance
)
from .alpha_vantage_client import AlphaVantageClient, AlphaVantageConfig
from .german_stock_processor import GermanStockProcessor, GermanStockData

logger = logging.getLogger(__name__)


class AlphaVantageTradingAPI(TradingAPIInterface):
    """
    Alpha Vantage Trading API implementation focused on German stocks
    Note: Alpha Vantage is primarily a data provider, not a trading platform
    This implementation focuses on market data and analysis
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Initialize Alpha Vantage client
        av_config = AlphaVantageConfig(
            api_key=config['credentials']['api_key'],
            tier=config['settings'].get('tier', 'free'),
            timeout=config.get('timeout', 30),
            max_retries=config.get('max_retries', 3)
        )
        
        self.client = AlphaVantageClient(av_config)
        self.processor = GermanStockProcessor(config)
        
        # German market specific configuration
        self.german_exchanges = config.get('german_exchanges', [])
        self.base_currency = 'EUR'
        self.timezone = config['settings'].get('timezone', 'Europe/Berlin')
        
        # Market data subscriptions
        self.subscriptions: Dict[str, List[Callable]] = {}
        self.subscription_task: Optional[asyncio.Task] = None
        
        # Cache for recent data
        self.data_cache: Dict[str, GermanStockData] = {}
        self.cache_ttl = 60  # 1 minute cache
        
        logger.info("Alpha Vantage Trading API initialized for German stocks")
    
    async def connect(self) -> bool:
        """Connect to Alpha Vantage API"""
        try:
            success = await self.client.connect()
            if success:
                self._connected = True
                logger.info("Connected to Alpha Vantage API")
                return True
            else:
                logger.error("Failed to connect to Alpha Vantage API")
                return False
        except Exception as e:
            logger.error(f"Connection error: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from Alpha Vantage API"""
        try:
            # Stop subscription task
            if self.subscription_task and not self.subscription_task.done():
                self.subscription_task.cancel()
            
            success = await self.client.disconnect()
            self._connected = False
            logger.info("Disconnected from Alpha Vantage API")
            return success
        except Exception as e:
            logger.error(f"Disconnection error: {e}")
            return False
    
    async def place_order(self, order: OrderRequest) -> OrderResponse:
        """
        Alpha Vantage doesn't support trading, only market data
        This method raises NotImplementedError
        """
        raise NotImplementedError(
            "Alpha Vantage is a market data provider, not a trading platform. "
            "Use this API for market data analysis only."
        )
    
    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """
        Alpha Vantage doesn't support trading
        This method raises NotImplementedError
        """
        raise NotImplementedError(
            "Alpha Vantage is a market data provider, not a trading platform."
        )
    
    async def get_order_status(self, order_id: str) -> OrderResponse:
        """
        Alpha Vantage doesn't support trading
        This method raises NotImplementedError
        """
        raise NotImplementedError(
            "Alpha Vantage is a market data provider, not a trading platform."
        )
    
    async def get_market_data(self, symbols: List[str]) -> List[MarketData]:
        """Get real-time market data for German stocks"""
        start_time = time.perf_counter()
        market_data_list = []
        
        try:
            # Normalize symbols for German exchanges
            normalized_symbols = []
            for symbol in symbols:
                normalized_symbol = self.client.normalize_german_symbol(symbol)
                normalized_symbols.append(normalized_symbol)
            
            # Get batch quotes
            quotes = await self.client.get_german_stocks_batch(normalized_symbols)
            
            # Process quotes
            processed_data = await self.processor.process_batch_quotes(quotes)
            
            # Convert to MarketData format
            for data in processed_data:
                # Alpha Vantage doesn't provide bid/ask, so we estimate
                bid = data.price * 0.999  # Estimate bid as 0.1% below price
                ask = data.price * 1.001  # Estimate ask as 0.1% above price
                
                market_data = MarketData(
                    symbol=data.symbol,
                    bid=bid,
                    ask=ask,
                    last=data.price,
                    volume=data.volume,
                    timestamp=data.timestamp,
                    latency_ms=self.measure_latency(start_time),
                    raw_data=data.raw_data
                )
                
                market_data_list.append(market_data)
                
                # Cache the data
                self.data_cache[data.symbol] = data
            
            logger.info(f"Retrieved market data for {len(market_data_list)} German stocks")
            return market_data_list
            
        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return []
    
    async def get_account_balance(self) -> AccountBalance:
        """
        Alpha Vantage doesn't provide account information
        Returns a placeholder AccountBalance
        """
        return AccountBalance(
            cash=0.0,
            portfolio_value=0.0,
            buying_power=0.0,
            margin_used=0.0,
            positions=[],
            timestamp=datetime.now(),
            raw_data={'note': 'Alpha Vantage is a data provider, not a broker'}
        )
    
    async def subscribe_market_data(self, symbols: List[str], 
                                   callback: Callable[[MarketData], None]) -> bool:
        """Subscribe to market data updates"""
        try:
            for symbol in symbols:
                if symbol not in self.subscriptions:
                    self.subscriptions[symbol] = []
                self.subscriptions[symbol].append(callback)
            
            # Start subscription task if not already running
            if not self.subscription_task or self.subscription_task.done():
                self.subscription_task = asyncio.create_task(self._subscription_loop())
            
            logger.info(f"Subscribed to market data for {len(symbols)} symbols")
            return True
            
        except Exception as e:
            logger.error(f"Error subscribing to market data: {e}")
            return False
    
    async def unsubscribe_market_data(self, symbols: List[str]) -> bool:
        """Unsubscribe from market data updates"""
        try:
            for symbol in symbols:
                if symbol in self.subscriptions:
                    del self.subscriptions[symbol]
            
            # Stop subscription task if no more subscriptions
            if not self.subscriptions and self.subscription_task:
                self.subscription_task.cancel()
                self.subscription_task = None
            
            logger.info(f"Unsubscribed from market data for {len(symbols)} symbols")
            return True
            
        except Exception as e:
            logger.error(f"Error unsubscribing from market data: {e}")
            return False
    
    async def _subscription_loop(self):
        """Background task for market data subscriptions"""
        while self.subscriptions and self._connected:
            try:
                # Get all subscribed symbols
                symbols = list(self.subscriptions.keys())
                
                # Get market data
                market_data_list = await self.get_market_data(symbols)
                
                # Trigger callbacks
                for market_data in market_data_list:
                    if market_data.symbol in self.subscriptions:
                        for callback in self.subscriptions[market_data.symbol]:
                            try:
                                self.trigger_callbacks('market_data', market_data)
                            except Exception as e:
                                logger.error(f"Error in market data callback: {e}")
                
                # Wait before next update (respect rate limits)
                await asyncio.sleep(60)  # 1 minute interval for free tier
                
            except asyncio.CancelledError:
                logger.info("Subscription loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in subscription loop: {e}")
                await asyncio.sleep(30)  # Wait before retry
    
    async def get_german_stock_fundamentals(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get fundamental data for German stock"""
        try:
            normalized_symbol = self.client.normalize_german_symbol(symbol)
            return await self.client.get_company_overview(normalized_symbol)
        except Exception as e:
            logger.error(f"Error getting fundamentals for {symbol}: {e}")
            return None
    
    async def get_german_stock_news(self, symbols: List[str]) -> Optional[Dict[str, Any]]:
        """Get news sentiment for German stocks"""
        try:
            normalized_symbols = [
                self.client.normalize_german_symbol(symbol) for symbol in symbols
            ]
            return await self.client.get_news_sentiment(normalized_symbols)
        except Exception as e:
            logger.error(f"Error getting news for {symbols}: {e}")
            return None
    
    async def get_eur_usd_rate(self) -> Optional[float]:
        """Get EUR/USD exchange rate"""
        try:
            rate_data = await self.client.get_exchange_rate('EUR', 'USD')
            if rate_data:
                return float(rate_data.get('5. Exchange Rate', 1.0))
            return None
        except Exception as e:
            logger.error(f"Error getting EUR/USD rate: {e}")
            return None
    
    async def get_intraday_data(self, symbol: str, interval: str = '1min') -> Optional[Any]:
        """Get intraday data for German stock"""
        try:
            normalized_symbol = self.client.normalize_german_symbol(symbol)
            raw_data = await self.client.get_intraday_data(normalized_symbol, interval)
            if raw_data:
                return self.processor.process_intraday_data(raw_data, symbol)
            return None
        except Exception as e:
            logger.error(f"Error getting intraday data for {symbol}: {e}")
            return None
    
    async def get_daily_data(self, symbol: str, outputsize: str = 'compact') -> Optional[Any]:
        """Get daily data for German stock"""
        try:
            normalized_symbol = self.client.normalize_german_symbol(symbol)
            return await self.client.get_daily_data(normalized_symbol, outputsize)
        except Exception as e:
            logger.error(f"Error getting daily data for {symbol}: {e}")
            return None
    
    def get_dax_components(self) -> List[str]:
        """Get DAX component symbols"""
        return self.processor.get_dax_symbols()
    
    def get_rate_limit_info(self) -> Dict[str, Any]:
        """Get current rate limit information"""
        return self.client.get_rate_limit_info()
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        try:
            start_time = time.perf_counter()
            
            # Test connection with a DAX stock
            test_symbol = 'SAP.DE'
            quote = await self.client.get_quote(test_symbol)
            
            latency_ms = self.measure_latency(start_time)
            
            if quote:
                return {
                    'status': 'healthy',
                    'connected': self.is_connected,
                    'latency_ms': latency_ms,
                    'test_symbol': test_symbol,
                    'rate_limit_info': self.get_rate_limit_info(),
                    'timestamp': datetime.now()
                }
            else:
                return {
                    'status': 'unhealthy',
                    'connected': self.is_connected,
                    'error': 'Failed to get test quote',
                    'timestamp': datetime.now()
                }
                
        except Exception as e:
            return {
                'status': 'unhealthy',
                'connected': self.is_connected,
                'error': str(e),
                'timestamp': datetime.now()
            }
    
    def get_cached_data(self, symbol: str) -> Optional[GermanStockData]:
        """Get cached data for symbol"""
        cached_data = self.data_cache.get(symbol)
        if cached_data:
            # Check if cache is still valid
            age = (datetime.now() - cached_data.timestamp).total_seconds()
            if age < self.cache_ttl:
                return cached_data
            else:
                # Remove stale data
                del self.data_cache[symbol]
        return None
    
    def get_trading_session_info(self) -> Dict[str, Any]:
        """Get German trading session information"""
        return {
            'timezone': self.timezone,
            'base_currency': self.base_currency,
            'exchanges': self.german_exchanges,
            'current_time': datetime.now().isoformat(),
            'trading_calendar': {
                'XETRA': {
                    'regular_hours': '09:00-17:30',
                    'extended_hours': '08:00-22:00'
                },
                'STUTTGART': {
                    'regular_hours': '08:00-22:00'
                }
            }
        }