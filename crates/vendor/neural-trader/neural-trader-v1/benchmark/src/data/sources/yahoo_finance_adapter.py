"""Yahoo Finance data adapter for real-time stock quotes.

Provides free real-time and historical stock data using Yahoo Finance API.
Note: This uses the unofficial yfinance library which may have rate limits.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
import logging
import aiohttp
import json

from ..realtime_feed import DataUpdate, DataSource

logger = logging.getLogger(__name__)


@dataclass
class YahooFinanceConfig:
    """Configuration for Yahoo Finance adapter."""
    symbols: List[str]
    update_interval: float = 1.0  # Seconds between updates
    batch_size: int = 50  # Max symbols per request
    enable_websocket: bool = False  # Yahoo doesn't provide free WebSocket
    cache_ttl: int = 60  # Cache results for 60 seconds
    retry_attempts: int = 3
    retry_delay: float = 1.0
    headers: Dict[str, str] = None


class YahooFinanceAdapter:
    """Yahoo Finance data adapter."""
    
    # Yahoo Finance endpoints
    QUOTE_URL = "https://query2.finance.yahoo.com/v7/finance/quote"
    CHART_URL = "https://query2.finance.yahoo.com/v8/finance/chart"
    SPARK_URL = "https://query2.finance.yahoo.com/v7/finance/spark"
    
    def __init__(self, config: YahooFinanceConfig):
        self.config = config
        self._session: Optional[aiohttp.ClientSession] = None
        self._running = False
        self._update_task = None
        self._callbacks: List[Callable] = []
        
        # Default headers to avoid rate limiting
        self._headers = config.headers or {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.9",
        }
        
        # Cache for rate limiting
        self._cache: Dict[str, tuple] = {}  # symbol -> (data, timestamp)
        
        # Metrics
        self._total_requests = 0
        self._failed_requests = 0
        self._last_update_time = 0
    
    async def start(self):
        """Start the adapter."""
        if self._running:
            return
        
        self._session = aiohttp.ClientSession(headers=self._headers)
        self._running = True
        self._update_task = asyncio.create_task(self._update_loop())
        logger.info(f"Yahoo Finance adapter started for {len(self.config.symbols)} symbols")
    
    async def stop(self):
        """Stop the adapter."""
        self._running = False
        
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
        
        if self._session:
            await self._session.close()
        
        logger.info("Yahoo Finance adapter stopped")
    
    def add_callback(self, callback: Callable[[DataUpdate], None]):
        """Add callback for data updates."""
        self._callbacks.append(callback)
    
    async def _update_loop(self):
        """Main update loop."""
        while self._running:
            try:
                start_time = time.time()
                
                # Fetch updates for all symbols in batches
                for i in range(0, len(self.config.symbols), self.config.batch_size):
                    batch = self.config.symbols[i:i + self.config.batch_size]
                    await self._fetch_batch(batch)
                
                # Calculate sleep time to maintain update interval
                elapsed = time.time() - start_time
                sleep_time = max(0, self.config.update_interval - elapsed)
                
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Update loop error: {e}")
                await asyncio.sleep(self.config.retry_delay)
    
    async def _fetch_batch(self, symbols: List[str]):
        """Fetch data for a batch of symbols."""
        # Check cache first
        uncached_symbols = []
        current_time = time.time()
        
        for symbol in symbols:
            if symbol in self._cache:
                data, timestamp = self._cache[symbol]
                if current_time - timestamp < self.config.cache_ttl:
                    # Use cached data
                    await self._process_quote(data)
                else:
                    uncached_symbols.append(symbol)
            else:
                uncached_symbols.append(symbol)
        
        if not uncached_symbols:
            return
        
        # Fetch uncached symbols
        params = {
            "symbols": ",".join(uncached_symbols),
            "fields": "symbol,regularMarketPrice,regularMarketTime,regularMarketChange,"
                     "regularMarketChangePercent,regularMarketVolume,bid,ask,bidSize,askSize"
        }
        
        for attempt in range(self.config.retry_attempts):
            try:
                self._total_requests += 1
                
                async with self._session.get(self.QUOTE_URL, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if "quoteResponse" in data and "result" in data["quoteResponse"]:
                            quotes = data["quoteResponse"]["result"]
                            
                            for quote in quotes:
                                # Cache the quote
                                symbol = quote.get("symbol")
                                if symbol:
                                    self._cache[symbol] = (quote, current_time)
                                
                                # Process the quote
                                await self._process_quote(quote)
                        
                        self._last_update_time = time.time()
                        return
                    else:
                        logger.warning(f"Yahoo Finance returned status {response.status}")
                        
            except Exception as e:
                logger.error(f"Fetch error (attempt {attempt + 1}): {e}")
                self._failed_requests += 1
                
                if attempt < self.config.retry_attempts - 1:
                    await asyncio.sleep(self.config.retry_delay * (attempt + 1))
    
    async def _process_quote(self, quote: Dict[str, Any]):
        """Process a quote and create data update."""
        try:
            symbol = quote.get("symbol")
            price = quote.get("regularMarketPrice")
            timestamp = quote.get("regularMarketTime", time.time())
            
            if not symbol or price is None:
                return
            
            # Create metadata
            metadata = {
                "source": "yahoo_finance",
                "change": quote.get("regularMarketChange"),
                "change_percent": quote.get("regularMarketChangePercent"),
                "volume": quote.get("regularMarketVolume"),
                "bid": quote.get("bid"),
                "ask": quote.get("ask"),
                "bid_size": quote.get("bidSize"),
                "ask_size": quote.get("askSize"),
            }
            
            # Create data update
            update = DataUpdate(
                symbol=symbol,
                price=float(price),
                timestamp=float(timestamp),
                source=DataSource.REST,
                metadata=metadata
            )
            
            # Notify callbacks
            for callback in self._callbacks:
                try:
                    await callback(update)
                except Exception as e:
                    logger.error(f"Callback error: {e}")
                    
        except Exception as e:
            logger.error(f"Error processing quote: {e}")
    
    async def get_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get single quote on demand."""
        params = {
            "symbols": symbol,
            "fields": "symbol,regularMarketPrice,regularMarketTime,regularMarketChange,"
                     "regularMarketChangePercent,regularMarketVolume,bid,ask"
        }
        
        try:
            async with self._session.get(self.QUOTE_URL, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if "quoteResponse" in data and "result" in data["quoteResponse"]:
                        quotes = data["quoteResponse"]["result"]
                        return quotes[0] if quotes else None
                        
        except Exception as e:
            logger.error(f"Error getting quote for {symbol}: {e}")
        
        return None
    
    async def get_historical_data(self, symbol: str, period: str = "1d", 
                                interval: str = "1m") -> Optional[Dict[str, Any]]:
        """Get historical data for a symbol."""
        params = {
            "period1": 0,  # Will be replaced by period
            "period2": int(time.time()),
            "interval": interval,
            "includePrePost": "true",
            "events": "div,splits",
        }
        
        # Convert period to timestamps
        period_map = {
            "1d": 86400,
            "5d": 432000,
            "1mo": 2592000,
            "3mo": 7776000,
            "6mo": 15552000,
            "1y": 31536000,
        }
        
        if period in period_map:
            params["period1"] = params["period2"] - period_map[period]
        
        try:
            url = f"{self.CHART_URL}/{symbol}"
            async with self._session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("chart", {}).get("result", [None])[0]
                    
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {e}")
        
        return None
    
    async def get_options_chain(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get options chain for a symbol."""
        # Note: This requires additional implementation
        # Yahoo Finance options API is more complex
        logger.warning("Options chain not yet implemented")
        return None
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get adapter metrics."""
        success_rate = 1 - (self._failed_requests / self._total_requests) if self._total_requests > 0 else 0
        
        return {
            "total_requests": self._total_requests,
            "failed_requests": self._failed_requests,
            "success_rate": success_rate,
            "cache_size": len(self._cache),
            "symbols_tracked": len(self.config.symbols),
            "last_update": self._last_update_time,
            "update_interval": self.config.update_interval,
        }
    
    async def add_symbol(self, symbol: str):
        """Add a symbol to track."""
        if symbol not in self.config.symbols:
            self.config.symbols.append(symbol)
            logger.info(f"Added symbol: {symbol}")
    
    async def remove_symbol(self, symbol: str):
        """Remove a symbol from tracking."""
        if symbol in self.config.symbols:
            self.config.symbols.remove(symbol)
            self._cache.pop(symbol, None)
            logger.info(f"Removed symbol: {symbol}")
    
    def get_supported_markets(self) -> List[str]:
        """Get list of supported markets."""
        return [
            "US",      # US stocks
            "CRYPTO",  # Cryptocurrencies (symbol-USD)
            "FX",      # Forex pairs
            "INDEX",   # Market indices
            "FUTURES", # Futures contracts
            "ETF",     # Exchange-traded funds
        ]
    
    def format_symbol(self, symbol: str, market: str = "US") -> str:
        """Format symbol for Yahoo Finance."""
        # Yahoo Finance symbol formatting
        if market == "CRYPTO":
            # Crypto symbols need -USD suffix
            if not symbol.endswith("-USD"):
                return f"{symbol}-USD"
        elif market == "FX":
            # Forex pairs need =X suffix
            if not symbol.endswith("=X"):
                return f"{symbol}=X"
        elif market == "INDEX":
            # Indices need ^ prefix
            if not symbol.startswith("^"):
                return f"^{symbol}"
        elif market == "FUTURES":
            # Futures need =F suffix
            if not symbol.endswith("=F"):
                return f"{symbol}=F"
        
        return symbol