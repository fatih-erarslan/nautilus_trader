"""
Stock feed handler for equity market data
Aggregates data from multiple sources (Yahoo, Finnhub, Alpha Vantage)
"""
import asyncio
import logging
from typing import Dict, List, Optional, Set, Any
from datetime import datetime
from collections import defaultdict

from ..realtime_manager import DataFeed, DataPoint, RealtimeManager
from ..yahoo_realtime import YahooRealtimeSource
from ..finnhub_client import FinnhubClient
from ..alpha_vantage import AlphaVantageSource

logger = logging.getLogger(__name__)


class StockFeed(DataFeed):
    """Stock feed handler with multi-source aggregation"""
    
    def __init__(self, symbols: List[str], config: Dict[str, Any] = None):
        self.symbols = set(symbols)
        self.config = config or {}
        
        # Data sources
        self.sources: Dict[str, Any] = {}
        self.manager = RealtimeManager()
        
        # Feed state
        self.is_running = False
        self.tasks: List[asyncio.Task] = []
        
        # Data tracking
        self.latest_data: Dict[str, DataPoint] = {}
        self.data_callbacks: List[callable] = []
        
        # Source priorities (higher number = higher priority)
        self.source_priorities = {
            'finnhub': 3,        # Real-time WebSocket
            'yahoo_realtime': 2,  # Near real-time REST
            'alpha_vantage': 1    # Rate-limited REST
        }
        
        # Configuration
        self.enable_yahoo = self.config.get('enable_yahoo', True)
        self.enable_finnhub = self.config.get('enable_finnhub', True)
        self.enable_alpha_vantage = self.config.get('enable_alpha_vantage', False)  # Requires API key
        
        self.yahoo_update_interval = self.config.get('yahoo_update_interval', 1.0)
        self.finnhub_api_key = self.config.get('finnhub_api_key')
        self.alpha_vantage_api_key = self.config.get('alpha_vantage_api_key')
    
    async def start(self) -> None:
        """Start the stock feed"""
        if self.is_running:
            return
        
        logger.info(f"Starting stock feed for {len(self.symbols)} symbols")
        
        # Initialize sources
        await self._initialize_sources()
        
        # Connect sources
        await self._connect_sources()
        
        # Subscribe to symbols
        await self._subscribe_to_symbols()
        
        self.is_running = True
        logger.info("Stock feed started successfully")
    
    async def stop(self) -> None:
        """Stop the stock feed"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Cancel tasks
        for task in self.tasks:
            task.cancel()
        
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        
        # Shutdown manager
        await self.manager.shutdown()
        
        logger.info("Stock feed stopped")
    
    async def subscribe(self, symbols: List[str]) -> None:
        """Subscribe to additional symbols"""
        new_symbols = set(symbols) - self.symbols
        if not new_symbols:
            return
        
        self.symbols.update(new_symbols)
        
        # Subscribe to new symbols on active sources
        for source_name, source in self.sources.items():
            if hasattr(source, 'subscribe'):
                await source.subscribe(list(new_symbols))
        
        logger.info(f"Subscribed to {len(new_symbols)} new symbols")
    
    async def unsubscribe(self, symbols: List[str]) -> None:
        """Unsubscribe from symbols"""
        symbols_to_remove = set(symbols) & self.symbols
        if not symbols_to_remove:
            return
        
        self.symbols -= symbols_to_remove
        
        # Unsubscribe from sources
        for source_name, source in self.sources.items():
            if hasattr(source, 'unsubscribe'):
                await source.unsubscribe(list(symbols_to_remove))
        
        # Clean up data
        for symbol in symbols_to_remove:
            self.latest_data.pop(symbol, None)
        
        logger.info(f"Unsubscribed from {len(symbols_to_remove)} symbols")
    
    async def _initialize_sources(self) -> None:
        """Initialize data sources"""
        # Yahoo Finance
        if self.enable_yahoo:
            yahoo_source = YahooRealtimeSource(
                use_websocket=False,  # Yahoo doesn't have free WebSocket
                update_interval=self.yahoo_update_interval
            )
            yahoo_source.set_data_callback(self._handle_data_point)
            self.sources['yahoo_realtime'] = yahoo_source
            self.manager.add_source(yahoo_source)
        
        # Finnhub
        if self.enable_finnhub and self.finnhub_api_key:
            finnhub_source = FinnhubClient(
                api_key=self.finnhub_api_key,
                use_websocket=True
            )
            finnhub_source.set_data_callback(self._handle_data_point)
            self.sources['finnhub'] = finnhub_source
            self.manager.add_source(finnhub_source)
        
        # Alpha Vantage
        if self.enable_alpha_vantage and self.alpha_vantage_api_key:
            alpha_source = AlphaVantageSource(
                api_key=self.alpha_vantage_api_key,
                rate_limit=5
            )
            alpha_source.set_data_callback(self._handle_data_point)
            self.sources['alpha_vantage'] = alpha_source
            self.manager.add_source(alpha_source)
        
        logger.info(f"Initialized {len(self.sources)} data sources")
    
    async def _connect_sources(self) -> None:
        """Connect to all data sources"""
        results = await self.manager.connect_all()
        
        connected_sources = [name for name, success in results.items() if success]
        failed_sources = [name for name, success in results.items() if not success]
        
        if connected_sources:
            logger.info(f"Connected to sources: {connected_sources}")
        
        if failed_sources:
            logger.warning(f"Failed to connect to sources: {failed_sources}")
        
        if not connected_sources:
            raise RuntimeError("Failed to connect to any data sources")
    
    async def _subscribe_to_symbols(self) -> None:
        """Subscribe to symbols on all sources"""
        symbols_list = list(self.symbols)
        
        for source_name, source in self.sources.items():
            try:
                await source.subscribe(symbols_list)
                logger.info(f"Subscribed to {len(symbols_list)} symbols on {source_name}")
            except Exception as e:
                logger.error(f"Failed to subscribe to {source_name}: {e}")
    
    async def _handle_data_point(self, data_point: DataPoint) -> None:
        """Handle incoming data point"""
        try:
            symbol = data_point.symbol
            source_name = data_point.source
            
            # Update latest data if this source has higher priority
            if (symbol not in self.latest_data or 
                self._get_source_priority(source_name) > self._get_source_priority(self.latest_data[symbol].source)):
                
                self.latest_data[symbol] = data_point
            
            # Notify callbacks
            for callback in self.data_callbacks:
                try:
                    await callback(data_point)
                except Exception as e:
                    logger.error(f"Callback error: {e}")
            
        except Exception as e:
            logger.error(f"Error handling data point: {e}")
    
    def _get_source_priority(self, source_name: str) -> int:
        """Get priority for a data source"""
        return self.source_priorities.get(source_name, 0)
    
    def add_callback(self, callback: callable) -> None:
        """Add data callback"""
        self.data_callbacks.append(callback)
    
    def remove_callback(self, callback: callable) -> None:
        """Remove data callback"""
        if callback in self.data_callbacks:
            self.data_callbacks.remove(callback)
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get latest price for a symbol"""
        data_point = self.latest_data.get(symbol)
        return data_point.price if data_point else None
    
    def get_latest_data(self, symbol: str) -> Optional[DataPoint]:
        """Get latest data point for a symbol"""
        return self.latest_data.get(symbol)
    
    def get_all_latest_data(self) -> Dict[str, DataPoint]:
        """Get latest data for all symbols"""
        return self.latest_data.copy()
    
    async def get_quote(self, symbol: str) -> Optional[DataPoint]:
        """Get real-time quote for a symbol"""
        # Try to get from cache first
        if symbol in self.latest_data:
            data_point = self.latest_data[symbol]
            # Return if recent (within 5 seconds)
            if (datetime.now() - data_point.timestamp).total_seconds() < 5:
                return data_point
        
        # Fetch fresh data from highest priority source
        for source_name in sorted(self.sources.keys(), 
                                 key=lambda x: self.source_priorities.get(x, 0), 
                                 reverse=True):
            
            source = self.sources[source_name]
            if hasattr(source, 'fetch_quote'):
                try:
                    quote = await source.fetch_quote(symbol)
                    if quote:
                        self.latest_data[symbol] = quote
                        return quote
                except Exception as e:
                    logger.error(f"Error fetching quote from {source_name}: {e}")
        
        return None
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get feed metrics"""
        metrics = {
            'feed_type': 'stock',
            'is_running': self.is_running,
            'symbols_count': len(self.symbols),
            'symbols': list(self.symbols),
            'sources_count': len(self.sources),
            'data_points_count': len(self.latest_data),
            'callbacks_count': len(self.data_callbacks)
        }
        
        # Add source metrics
        for source_name, source in self.sources.items():
            if hasattr(source, 'get_metrics'):
                metrics[f'{source_name}_metrics'] = source.get_metrics()
        
        # Add manager metrics
        if self.manager:
            metrics['manager_metrics'] = asyncio.create_task(self.manager.get_stats())
        
        return metrics
    
    def get_symbols(self) -> List[str]:
        """Get list of subscribed symbols"""
        return list(self.symbols)
    
    def is_symbol_subscribed(self, symbol: str) -> bool:
        """Check if symbol is subscribed"""
        return symbol in self.symbols
    
    async def validate_symbols(self, symbols: List[str]) -> Dict[str, bool]:
        """Validate if symbols exist and are tradeable"""
        results = {}
        
        # Try to get quotes for each symbol
        for symbol in symbols:
            try:
                quote = await self.get_quote(symbol)
                results[symbol] = quote is not None and quote.price > 0
            except Exception:
                results[symbol] = False
        
        return results