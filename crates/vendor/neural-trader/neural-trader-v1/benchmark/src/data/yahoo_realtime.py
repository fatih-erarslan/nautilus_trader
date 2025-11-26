"""
Yahoo Finance real-time data source with WebSocket fallback to REST
"""
import asyncio
import aiohttp
import time
import logging
from typing import Dict, List, Optional, Set, Any
from datetime import datetime
import json

from .realtime_manager import DataSource, DataPoint
from .sources.yahoo_finance_adapter import YahooFinanceAdapter, YahooFinanceConfig

logger = logging.getLogger(__name__)


class YahooRealtimeSource(DataSource):
    """Yahoo Finance real-time data source"""
    
    def __init__(self, use_websocket: bool = True, update_interval: float = 0.5):
        super().__init__("yahoo_realtime")
        self.use_websocket = use_websocket
        self.update_interval = update_interval
        
        # REST API adapter
        self.rest_adapter: Optional[YahooFinanceAdapter] = None
        self.subscribed_symbols: Set[str] = set()
        
        # WebSocket connection (for future implementation)
        self.ws_session: Optional[aiohttp.ClientSession] = None
        self.ws_connection = None
        
        # Callbacks for data delivery
        self.data_callback = None
        
        # Connection tracking
        self.last_heartbeat = time.time()
        self.connection_start_time = None
        
        # Performance tracking
        self.request_count = 0
        self.error_count = 0
    
    async def connect(self) -> bool:
        """Connect to Yahoo Finance"""
        try:
            self.connection_start_time = time.time()
            
            # Try WebSocket first if requested
            if self.use_websocket and hasattr(self, 'connect_websocket'):
                try:
                    await self.connect_websocket()
                    self.connection_type = "WebSocket"
                    self.is_connected = True
                    return True
                except Exception as e:
                    logger.warning(f"WebSocket connection failed: {e}, falling back to REST")
            
            # Fallback to REST API
            config = YahooFinanceConfig(
                symbols=[],
                update_interval=self.update_interval,
                batch_size=50,
                cache_ttl=1  # Very short cache for real-time
            )
            
            self.rest_adapter = YahooFinanceAdapter(config)
            self.rest_adapter.add_callback(self._handle_rest_update)
            await self.rest_adapter.start()
            
            self.connection_type = "REST"
            self.is_connected = True
            logger.info(f"Connected to Yahoo Finance via {self.connection_type}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Yahoo Finance: {e}")
            self.is_connected = False
            return False
    
    async def connect_websocket(self) -> None:
        """Connect via WebSocket (placeholder for future implementation)"""
        # Yahoo Finance doesn't provide free WebSocket API
        # This is a placeholder for potential future implementation
        # or for using alternative WebSocket sources
        raise NotImplementedError("Yahoo Finance WebSocket not available")
    
    async def disconnect(self) -> None:
        """Disconnect from Yahoo Finance"""
        try:
            if self.rest_adapter:
                await self.rest_adapter.stop()
                self.rest_adapter = None
            
            if self.ws_connection:
                await self.ws_connection.close()
                self.ws_connection = None
            
            if self.ws_session:
                await self.ws_session.close()
                self.ws_session = None
            
            self.is_connected = False
            logger.info("Disconnected from Yahoo Finance")
            
        except Exception as e:
            logger.error(f"Error during disconnect: {e}")
    
    async def subscribe(self, symbols: List[str]) -> None:
        """Subscribe to symbols"""
        if not self.is_connected:
            raise RuntimeError("Not connected to Yahoo Finance")
        
        # Format symbols for Yahoo
        formatted_symbols = []
        for symbol in symbols:
            # Auto-detect and format symbol types
            if symbol in ['BTC', 'ETH', 'SOL', 'DOGE']:  # Common crypto
                formatted_symbols.append(f"{symbol}-USD")
            elif symbol.startswith('^'):  # Index
                formatted_symbols.append(symbol)
            elif '/' in symbol:  # Forex pair
                formatted_symbols.append(f"{symbol.replace('/', '')}=X")
            else:  # Regular stock
                formatted_symbols.append(symbol)
        
        self.subscribed_symbols.update(formatted_symbols)
        
        if self.rest_adapter:
            # Update REST adapter symbols
            self.rest_adapter.config.symbols = list(self.subscribed_symbols)
            logger.info(f"Subscribed to {len(formatted_symbols)} symbols on Yahoo Finance")
    
    async def unsubscribe(self, symbols: List[str]) -> None:
        """Unsubscribe from symbols"""
        for symbol in symbols:
            self.subscribed_symbols.discard(symbol)
            # Also try formatted versions
            self.subscribed_symbols.discard(f"{symbol}-USD")
            self.subscribed_symbols.discard(f"{symbol}=X")
        
        if self.rest_adapter:
            self.rest_adapter.config.symbols = list(self.subscribed_symbols)
    
    async def _handle_rest_update(self, update) -> None:
        """Handle REST API data update"""
        try:
            # Calculate latency
            latency_ms = (time.time() - update.timestamp) * 1000
            
            # Extract data from update
            metadata = update.metadata or {}
            
            # Create DataPoint
            data_point = DataPoint(
                source=self.name,
                symbol=update.symbol,
                timestamp=datetime.fromtimestamp(update.timestamp),
                price=update.price,
                volume=metadata.get('volume', 0),
                bid=metadata.get('bid'),
                ask=metadata.get('ask'),
                latency_ms=latency_ms,
                metadata={
                    'change': metadata.get('change'),
                    'change_percent': metadata.get('change_percent'),
                    'bid_size': metadata.get('bid_size'),
                    'ask_size': metadata.get('ask_size')
                }
            )
            
            # Deliver to callback
            if self.data_callback:
                await self.data_callback(data_point)
            
            # Update metrics
            self.request_count += 1
            self.last_heartbeat = time.time()
            
        except Exception as e:
            logger.error(f"Error handling REST update: {e}")
            self.error_count += 1
    
    async def fetch_quote(self, symbol: str) -> Optional[DataPoint]:
        """Fetch a single quote on demand"""
        if not self.rest_adapter:
            return None
        
        try:
            start_time = time.time()
            quote = await self.rest_adapter.get_quote(symbol)
            
            if quote:
                latency_ms = (time.time() - start_time) * 1000
                
                return DataPoint(
                    source=self.name,
                    symbol=symbol,
                    timestamp=datetime.now(),
                    price=float(quote.get('regularMarketPrice', 0)),
                    volume=int(quote.get('regularMarketVolume', 0)),
                    bid=quote.get('bid'),
                    ask=quote.get('ask'),
                    latency_ms=latency_ms,
                    metadata={
                        'change': quote.get('regularMarketChange'),
                        'change_percent': quote.get('regularMarketChangePercent')
                    }
                )
        except Exception as e:
            logger.error(f"Error fetching quote for {symbol}: {e}")
        
        return None
    
    def set_data_callback(self, callback) -> None:
        """Set callback for data delivery"""
        self.data_callback = callback
    
    async def get_historical_data(self, symbol: str, period: str = "1d") -> Optional[Dict[str, Any]]:
        """Get historical data for backtesting"""
        if self.rest_adapter:
            return await self.rest_adapter.get_historical_data(symbol, period)
        return None
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get source metrics"""
        uptime = time.time() - self.connection_start_time if self.connection_start_time else 0
        success_rate = (self.request_count - self.error_count) / self.request_count if self.request_count > 0 else 0
        
        metrics = {
            'source': self.name,
            'connection_type': self.connection_type,
            'is_connected': self.is_connected,
            'symbols_subscribed': len(self.subscribed_symbols),
            'request_count': self.request_count,
            'error_count': self.error_count,
            'success_rate': success_rate,
            'uptime_seconds': uptime,
            'last_heartbeat': self.last_heartbeat
        }
        
        # Add REST adapter metrics if available
        if self.rest_adapter:
            metrics['rest_metrics'] = self.rest_adapter.get_metrics()
        
        return metrics
    
    def is_market_open(self, market: str = "US") -> bool:
        """Check if market is currently open"""
        from datetime import time
        now = datetime.now()
        
        # Simple US market hours check (9:30 AM - 4:00 PM ET)
        # Note: This doesn't account for holidays or time zones properly
        if market == "US":
            current_hour = now.hour
            current_minute = now.minute
            
            # Convert to ET (assuming UTC-5 for simplicity)
            et_hour = (current_hour - 5) % 24
            
            if et_hour >= 9 and et_hour < 16:
                if et_hour == 9 and current_minute < 30:
                    return False
                return True
        
        # Crypto markets are always open
        elif market == "CRYPTO":
            return True
        
        # Forex markets (Sunday 5 PM ET to Friday 5 PM ET)
        elif market == "FX":
            weekday = now.weekday()
            return weekday < 5  # Monday-Friday
        
        return False