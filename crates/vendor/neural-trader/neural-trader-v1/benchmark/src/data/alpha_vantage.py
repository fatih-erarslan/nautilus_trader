"""
Alpha Vantage data source with strict rate limiting for free tier
"""
import asyncio
import aiohttp
import time
import logging
from typing import Dict, List, Optional, Set, Any
from datetime import datetime
from collections import deque

from .realtime_manager import DataSource, DataPoint
from .sources.alpha_vantage_adapter import AlphaVantageAdapter, AlphaVantageConfig, RateLimiter

logger = logging.getLogger(__name__)


class AlphaVantageSource(DataSource):
    """Alpha Vantage data source with rate limiting"""
    
    def __init__(self, api_key: str, rate_limit: int = 5):
        super().__init__("alpha_vantage")
        self.api_key = api_key
        self.rate_limit = rate_limit  # Calls per minute
        
        # Adapter
        self.adapter: Optional[AlphaVantageAdapter] = None
        self.subscribed_symbols: Set[str] = set()
        
        # Rate limiting
        self.rate_limiter = RateLimiter(calls_per_minute=rate_limit)
        self.last_request_time = 0
        
        # Callbacks
        self.data_callback = None
        
        # Metrics
        self.request_count = 0
        self.error_count = 0
        self.rate_limit_hits = 0
        
        # Connection tracking
        self.connection_start_time = None
    
    async def connect(self) -> bool:
        """Connect to Alpha Vantage API"""
        try:
            self.connection_start_time = time.time()
            
            # Create adapter configuration
            config = AlphaVantageConfig(
                api_key=self.api_key,
                symbols=[],
                update_interval=12.0,  # 5 calls/min = 1 call every 12 seconds
                functions=["GLOBAL_QUOTE"],
                enable_intraday=False,  # To conserve API calls
                outputsize="compact"
            )
            
            # Create and start adapter
            self.adapter = AlphaVantageAdapter(config)
            self.adapter.add_callback(self._handle_data_update)
            await self.adapter.start()
            
            self.is_connected = True
            self.connection_type = "REST"
            logger.info(f"Connected to Alpha Vantage API (rate limit: {self.rate_limit} calls/min)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Alpha Vantage: {e}")
            self.is_connected = False
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from Alpha Vantage"""
        try:
            if self.adapter:
                await self.adapter.stop()
                self.adapter = None
            
            self.is_connected = False
            logger.info("Disconnected from Alpha Vantage")
            
        except Exception as e:
            logger.error(f"Error during disconnect: {e}")
    
    async def subscribe(self, symbols: List[str]) -> None:
        """Subscribe to symbols"""
        if not self.is_connected:
            raise RuntimeError("Not connected to Alpha Vantage")
        
        # Alpha Vantage doesn't have real-time subscriptions
        # We'll poll these symbols based on rate limits
        self.subscribed_symbols.update(symbols)
        
        if self.adapter:
            self.adapter.config.symbols = list(self.subscribed_symbols)
            logger.info(f"Subscribed to {len(symbols)} symbols on Alpha Vantage")
    
    async def unsubscribe(self, symbols: List[str]) -> None:
        """Unsubscribe from symbols"""
        for symbol in symbols:
            self.subscribed_symbols.discard(symbol)
        
        if self.adapter:
            self.adapter.config.symbols = list(self.subscribed_symbols)
    
    async def _handle_data_update(self, update) -> None:
        """Handle data update from adapter"""
        try:
            # Calculate latency (Alpha Vantage doesn't provide real-time timestamps)
            latency_ms = 50.0  # Estimated latency for REST API
            
            # Extract metadata
            metadata = update.metadata or {}
            
            # Create DataPoint
            data_point = DataPoint(
                source=self.name,
                symbol=update.symbol,
                timestamp=datetime.fromtimestamp(update.timestamp),
                price=update.price,
                volume=metadata.get('volume', 0),
                bid=None,  # Alpha Vantage doesn't provide bid/ask in free tier
                ask=None,
                latency_ms=latency_ms,
                metadata={
                    'open': metadata.get('open'),
                    'high': metadata.get('high'),
                    'low': metadata.get('low'),
                    'previous_close': metadata.get('previous_close'),
                    'change': metadata.get('change'),
                    'change_percent': metadata.get('change_percent')
                }
            )
            
            # Deliver to callback
            if self.data_callback:
                await self.data_callback(data_point)
            
            # Update metrics
            self.request_count += 1
            
        except Exception as e:
            logger.error(f"Error handling data update: {e}")
            self.error_count += 1
    
    async def fetch_quote(self, symbol: str) -> Optional[DataPoint]:
        """Fetch a single quote with rate limiting"""
        # Enforce rate limiting
        await self.rate_limiter.acquire()
        
        if not self.adapter:
            return None
        
        try:
            # Use the adapter's direct fetch method
            await self.adapter._fetch_quote(symbol)
            
            # Get from cache
            if symbol in self.adapter._quote_cache:
                quote_data, _ = self.adapter._quote_cache[symbol]
                
                # Convert to DataPoint
                price = float(quote_data.get("05. price", 0))
                volume = int(quote_data.get("06. volume", 0))
                change = float(quote_data.get("09. change", 0))
                change_percent = quote_data.get("10. change percent", "0%").rstrip("%")
                
                return DataPoint(
                    source=self.name,
                    symbol=symbol,
                    timestamp=datetime.now(),
                    price=price,
                    volume=volume,
                    latency_ms=50.0,
                    metadata={
                        'open': float(quote_data.get("02. open", 0)),
                        'high': float(quote_data.get("03. high", 0)),
                        'low': float(quote_data.get("04. low", 0)),
                        'previous_close': float(quote_data.get("08. previous close", 0)),
                        'change': change,
                        'change_percent': float(change_percent) if change_percent else 0
                    }
                )
                
        except Exception as e:
            logger.error(f"Error fetching quote for {symbol}: {e}")
            if "rate limit" in str(e).lower():
                self.rate_limit_hits += 1
        
        return None
    
    def set_data_callback(self, callback) -> None:
        """Set callback for data delivery"""
        self.data_callback = callback
    
    async def get_technical_indicator(self, symbol: str, indicator: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Get technical indicator data"""
        # Enforce rate limiting
        await self.rate_limiter.acquire()
        
        if self.adapter:
            return await self.adapter.get_technical_indicators(symbol, indicator, **kwargs)
        return None
    
    async def get_intraday_data(self, symbol: str, interval: str = "5min") -> Optional[Dict[str, Any]]:
        """Get intraday time series data"""
        # Enforce rate limiting
        await self.rate_limiter.acquire()
        
        if not self.adapter:
            return None
        
        try:
            # Temporarily switch to intraday function
            original_functions = self.adapter.config.functions
            self.adapter.config.functions = ["TIME_SERIES_INTRADAY"]
            
            await self.adapter._fetch_intraday(symbol, interval)
            
            # Restore original functions
            self.adapter.config.functions = original_functions
            
            return True
            
        except Exception as e:
            logger.error(f"Error fetching intraday data: {e}")
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
            'rate_limit_hits': self.rate_limit_hits,
            'success_rate': success_rate,
            'uptime_seconds': uptime,
            'rate_limit': f"{self.rate_limit} calls/min"
        }
        
        # Add adapter metrics if available
        if self.adapter:
            metrics['adapter_metrics'] = self.adapter.get_metrics()
        
        return metrics
    
    def get_remaining_calls(self) -> int:
        """Get estimated remaining API calls this minute"""
        if not hasattr(self, 'rate_limiter'):
            return 0
        
        # Count recent calls
        now = time.time()
        recent_calls = sum(1 for t in self.rate_limiter.call_times if t > now - 60)
        return max(0, self.rate_limit - recent_calls)
    
    async def batch_fetch_quotes(self, symbols: List[str]) -> Dict[str, Optional[DataPoint]]:
        """Batch fetch quotes with rate limiting"""
        results = {}
        
        for symbol in symbols:
            # Check if we have remaining calls
            if self.get_remaining_calls() == 0:
                logger.warning(f"Rate limit reached, skipping remaining {len(symbols) - len(results)} symbols")
                break
            
            # Fetch quote
            quote = await self.fetch_quote(symbol)
            results[symbol] = quote
            
            # Small delay between requests to be respectful
            await asyncio.sleep(0.1)
        
        return results