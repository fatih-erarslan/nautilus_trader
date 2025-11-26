"""
Crypto feed handler for cryptocurrency market data
Aggregates data from Coinbase and other crypto sources
"""
import asyncio
import logging
from typing import Dict, List, Optional, Set, Any
from datetime import datetime
from collections import defaultdict

from ..realtime_manager import DataFeed, DataPoint, RealtimeManager
from ..coinbase_feed import CoinbaseFeed

logger = logging.getLogger(__name__)


class CryptoFeed(DataFeed):
    """Crypto feed handler with multi-source aggregation"""
    
    # Common crypto pairs
    POPULAR_PAIRS = [
        "BTC-USD", "ETH-USD", "SOL-USD", "DOGE-USD",
        "MATIC-USD", "LINK-USD", "DOT-USD", "UNI-USD",
        "ADA-USD", "XRP-USD", "AVAX-USD", "ATOM-USD"
    ]
    
    def __init__(self, symbols: List[str] = None, config: Dict[str, Any] = None):
        self.symbols = set(symbols or self.POPULAR_PAIRS)
        self.config = config or {}
        
        # Data sources
        self.sources: Dict[str, Any] = {}
        self.manager = RealtimeManager()
        
        # Feed state
        self.is_running = False
        self.tasks: List[asyncio.Task] = []
        
        # Data tracking
        self.latest_data: Dict[str, DataPoint] = {}
        self.order_books: Dict[str, Dict[str, Any]] = {}
        self.data_callbacks: List[callable] = []
        
        # Source priorities
        self.source_priorities = {
            'coinbase': 3,       # Real-time WebSocket
            'binance': 2,        # Future implementation
            'coingecko': 1       # Future implementation
        }
        
        # Configuration
        self.enable_coinbase = self.config.get('enable_coinbase', True)
        self.coinbase_sandbox = self.config.get('coinbase_sandbox', False)
        self.enable_order_books = self.config.get('enable_order_books', True)
        
        # Market hours (crypto markets are 24/7)
        self.market_hours = "24/7"
    
    async def start(self) -> None:
        """Start the crypto feed"""
        if self.is_running:
            return
        
        logger.info(f"Starting crypto feed for {len(self.symbols)} pairs")
        
        # Initialize sources
        await self._initialize_sources()
        
        # Connect sources
        await self._connect_sources()
        
        # Subscribe to symbols
        await self._subscribe_to_symbols()
        
        # Start order book tracking if enabled
        if self.enable_order_books:
            self.tasks.append(asyncio.create_task(self._order_book_updater()))
        
        self.is_running = True
        logger.info("Crypto feed started successfully")
    
    async def stop(self) -> None:
        """Stop the crypto feed"""
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
        
        logger.info("Crypto feed stopped")
    
    async def subscribe(self, symbols: List[str]) -> None:
        """Subscribe to additional crypto pairs"""
        # Normalize symbols to exchange format
        normalized_symbols = []
        for symbol in symbols:
            normalized = self._normalize_symbol(symbol)
            normalized_symbols.append(normalized)
        
        new_symbols = set(normalized_symbols) - self.symbols
        if not new_symbols:
            return
        
        self.symbols.update(new_symbols)
        
        # Subscribe to new symbols on active sources
        for source_name, source in self.sources.items():
            if hasattr(source, 'subscribe'):
                await source.subscribe(list(new_symbols))
        
        logger.info(f"Subscribed to {len(new_symbols)} new crypto pairs")
    
    async def unsubscribe(self, symbols: List[str]) -> None:
        """Unsubscribe from crypto pairs"""
        normalized_symbols = [self._normalize_symbol(s) for s in symbols]
        symbols_to_remove = set(normalized_symbols) & self.symbols
        
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
            self.order_books.pop(symbol, None)
        
        logger.info(f"Unsubscribed from {len(symbols_to_remove)} crypto pairs")
    
    def _normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol to standard format (e.g., BTC-USD)"""
        # Handle different input formats
        if '/' in symbol:
            # BTC/USD -> BTC-USD
            return symbol.replace('/', '-')
        elif '-' in symbol:
            # Already in correct format
            return symbol.upper()
        else:
            # Assume USD pairing for single currency
            return f"{symbol.upper()}-USD"
    
    async def _initialize_sources(self) -> None:
        """Initialize crypto data sources"""
        # Coinbase
        if self.enable_coinbase:
            coinbase_source = CoinbaseFeed(
                use_websocket=True,
                sandbox=self.coinbase_sandbox
            )
            coinbase_source.set_data_callback(self._handle_data_point)
            self.sources['coinbase'] = coinbase_source
            self.manager.add_source(coinbase_source)
        
        # Future: Add Binance, CoinGecko, etc.
        
        logger.info(f"Initialized {len(self.sources)} crypto data sources")
    
    async def _connect_sources(self) -> None:
        """Connect to all crypto data sources"""
        results = await self.manager.connect_all()
        
        connected_sources = [name for name, success in results.items() if success]
        failed_sources = [name for name, success in results.items() if not success]
        
        if connected_sources:
            logger.info(f"Connected to crypto sources: {connected_sources}")
        
        if failed_sources:
            logger.warning(f"Failed to connect to crypto sources: {failed_sources}")
        
        if not connected_sources:
            raise RuntimeError("Failed to connect to any crypto data sources")
    
    async def _subscribe_to_symbols(self) -> None:
        """Subscribe to crypto pairs on all sources"""
        symbols_list = list(self.symbols)
        
        for source_name, source in self.sources.items():
            try:
                await source.subscribe(symbols_list)
                logger.info(f"Subscribed to {len(symbols_list)} pairs on {source_name}")
            except Exception as e:
                logger.error(f"Failed to subscribe to {source_name}: {e}")
    
    async def _handle_data_point(self, data_point: DataPoint) -> None:
        """Handle incoming crypto data point"""
        try:
            symbol = data_point.symbol
            source_name = data_point.source
            
            # Update order book if bid/ask data available
            if data_point.bid and data_point.ask and self.enable_order_books:
                self._update_order_book(symbol, data_point)
            
            # Update latest data if this source has higher priority
            if (symbol not in self.latest_data or 
                self._get_source_priority(source_name) > self._get_source_priority(self.latest_data[symbol].source)):
                
                self.latest_data[symbol] = data_point
            
            # Calculate spread if available
            if data_point.bid and data_point.ask:
                spread = data_point.ask - data_point.bid
                spread_pct = (spread / data_point.price) * 100 if data_point.price > 0 else 0
                
                # Add spread info to metadata
                if data_point.metadata is None:
                    data_point.metadata = {}
                data_point.metadata.update({
                    'spread': spread,
                    'spread_percent': spread_pct
                })
            
            # Notify callbacks
            for callback in self.data_callbacks:
                try:
                    await callback(data_point)
                except Exception as e:
                    logger.error(f"Callback error: {e}")
            
        except Exception as e:
            logger.error(f"Error handling crypto data point: {e}")
    
    def _update_order_book(self, symbol: str, data_point: DataPoint) -> None:
        """Update order book with bid/ask data"""
        if symbol not in self.order_books:
            self.order_books[symbol] = {
                'bids': {},
                'asks': {},
                'last_update': datetime.now()
            }
        
        book = self.order_books[symbol]
        
        # Update best bid/ask
        if data_point.bid:
            bid_size = data_point.metadata.get('best_bid_size', 0) if data_point.metadata else 0
            book['bids'][data_point.bid] = bid_size
        
        if data_point.ask:
            ask_size = data_point.metadata.get('best_ask_size', 0) if data_point.metadata else 0
            book['asks'][data_point.ask] = ask_size
        
        book['last_update'] = datetime.now()
    
    async def _order_book_updater(self) -> None:
        """Periodically clean up old order book data"""
        while self.is_running:
            try:
                now = datetime.now()
                
                # Clean up order books older than 5 minutes
                for symbol, book in list(self.order_books.items()):
                    if (now - book['last_update']).total_seconds() > 300:
                        del self.order_books[symbol]
                
                await asyncio.sleep(60)  # Clean up every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Order book updater error: {e}")
    
    def _get_source_priority(self, source_name: str) -> int:
        """Get priority for a crypto data source"""
        return self.source_priorities.get(source_name, 0)
    
    def add_callback(self, callback: callable) -> None:
        """Add data callback"""
        self.data_callbacks.append(callback)
    
    def remove_callback(self, callback: callable) -> None:
        """Remove data callback"""
        if callback in self.data_callbacks:
            self.data_callbacks.remove(callback)
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get latest price for a crypto pair"""
        normalized_symbol = self._normalize_symbol(symbol)
        data_point = self.latest_data.get(normalized_symbol)
        return data_point.price if data_point else None
    
    def get_latest_data(self, symbol: str) -> Optional[DataPoint]:
        """Get latest data point for a crypto pair"""
        normalized_symbol = self._normalize_symbol(symbol)
        return self.latest_data.get(normalized_symbol)
    
    def get_order_book(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get order book for a crypto pair"""
        normalized_symbol = self._normalize_symbol(symbol)
        return self.order_books.get(normalized_symbol)
    
    def get_best_bid_ask(self, symbol: str) -> tuple[Optional[float], Optional[float]]:
        """Get best bid and ask for a crypto pair"""
        data_point = self.get_latest_data(symbol)
        if data_point:
            return data_point.bid, data_point.ask
        return None, None
    
    def get_spread(self, symbol: str) -> Optional[Dict[str, float]]:
        """Get spread information for a crypto pair"""
        data_point = self.get_latest_data(symbol)
        if data_point and data_point.bid and data_point.ask:
            spread = data_point.ask - data_point.bid
            spread_pct = (spread / data_point.price) * 100 if data_point.price > 0 else 0
            
            return {
                'spread': spread,
                'spread_percent': spread_pct,
                'bid': data_point.bid,
                'ask': data_point.ask,
                'mid': (data_point.bid + data_point.ask) / 2
            }
        return None
    
    async def get_quote(self, symbol: str) -> Optional[DataPoint]:
        """Get real-time quote for a crypto pair"""
        normalized_symbol = self._normalize_symbol(symbol)
        
        # Try to get from cache first
        if normalized_symbol in self.latest_data:
            data_point = self.latest_data[normalized_symbol]
            # Return if recent (within 2 seconds for crypto)
            if (datetime.now() - data_point.timestamp).total_seconds() < 2:
                return data_point
        
        # Fetch fresh data from highest priority source
        for source_name in sorted(self.sources.keys(), 
                                 key=lambda x: self.source_priorities.get(x, 0), 
                                 reverse=True):
            
            source = self.sources[source_name]
            if hasattr(source, 'fetch_quote'):
                try:
                    quote = await source.fetch_quote(normalized_symbol)
                    if quote:
                        self.latest_data[normalized_symbol] = quote
                        return quote
                except Exception as e:
                    logger.error(f"Error fetching crypto quote from {source_name}: {e}")
        
        return None
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get crypto feed metrics"""
        metrics = {
            'feed_type': 'crypto',
            'is_running': self.is_running,
            'pairs_count': len(self.symbols),
            'pairs': list(self.symbols),
            'sources_count': len(self.sources),
            'data_points_count': len(self.latest_data),
            'order_books_count': len(self.order_books),
            'callbacks_count': len(self.data_callbacks),
            'market_hours': self.market_hours
        }
        
        # Add source metrics
        for source_name, source in self.sources.items():
            if hasattr(source, 'get_metrics'):
                metrics[f'{source_name}_metrics'] = source.get_metrics()
        
        # Calculate average spread across all pairs
        spreads = []
        for symbol in self.symbols:
            spread_info = self.get_spread(symbol)
            if spread_info:
                spreads.append(spread_info['spread_percent'])
        
        if spreads:
            metrics['average_spread_percent'] = sum(spreads) / len(spreads)
            metrics['max_spread_percent'] = max(spreads)
            metrics['min_spread_percent'] = min(spreads)
        
        return metrics
    
    def get_symbols(self) -> List[str]:
        """Get list of subscribed crypto pairs"""
        return list(self.symbols)
    
    def is_symbol_subscribed(self, symbol: str) -> bool:
        """Check if crypto pair is subscribed"""
        normalized_symbol = self._normalize_symbol(symbol)
        return normalized_symbol in self.symbols
    
    def is_market_open(self) -> bool:
        """Check if crypto market is open (always True - 24/7)"""
        return True
    
    async def get_24hr_stats(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get 24-hour statistics for a crypto pair"""
        normalized_symbol = self._normalize_symbol(symbol)
        
        # Try to get from Coinbase source
        if 'coinbase' in self.sources:
            coinbase_source = self.sources['coinbase']
            if hasattr(coinbase_source, 'get_24hr_stats'):
                try:
                    return await coinbase_source.get_24hr_stats(normalized_symbol)
                except Exception as e:
                    logger.error(f"Error fetching 24hr stats: {e}")
        
        return None