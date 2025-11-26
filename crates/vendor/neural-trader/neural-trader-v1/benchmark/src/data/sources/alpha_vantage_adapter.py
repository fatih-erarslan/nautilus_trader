"""Alpha Vantage data adapter for free market data API.

Provides real-time and historical stock data using Alpha Vantage's free tier.
Note: Free tier has rate limits of 5 API requests per minute.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from collections import deque
import logging
import aiohttp

from ..realtime_feed import DataUpdate, DataSource

logger = logging.getLogger(__name__)


@dataclass
class AlphaVantageConfig:
    """Configuration for Alpha Vantage adapter."""
    api_key: str
    symbols: List[str]
    update_interval: float = 15.0  # 15 seconds to respect rate limits
    functions: List[str] = None  # Which functions to use
    enable_intraday: bool = True
    enable_daily: bool = False
    outputsize: str = "compact"  # compact or full
    datatype: str = "json"  # json or csv


class RateLimiter:
    """Rate limiter for API calls."""
    
    def __init__(self, calls_per_minute: int = 5):
        self.calls_per_minute = calls_per_minute
        self.call_times: deque = deque(maxlen=calls_per_minute)
        self._lock = asyncio.Lock()
    
    async def acquire(self):
        """Wait if necessary to respect rate limit."""
        async with self._lock:
            now = time.time()
            
            # Remove calls older than 1 minute
            while self.call_times and self.call_times[0] < now - 60:
                self.call_times.popleft()
            
            # If at limit, wait
            if len(self.call_times) >= self.calls_per_minute:
                sleep_time = 60 - (now - self.call_times[0]) + 0.1
                if sleep_time > 0:
                    logger.info(f"Rate limit reached, waiting {sleep_time:.1f}s")
                    await asyncio.sleep(sleep_time)
            
            # Record this call
            self.call_times.append(now)


class AlphaVantageAdapter:
    """Alpha Vantage data adapter."""
    
    BASE_URL = "https://www.alphavantage.co/query"
    
    # Available functions
    FUNCTIONS = {
        "TIME_SERIES_INTRADAY": "intraday",
        "GLOBAL_QUOTE": "quote",
        "TIME_SERIES_DAILY": "daily",
        "TIME_SERIES_WEEKLY": "weekly",
        "TIME_SERIES_MONTHLY": "monthly",
    }
    
    def __init__(self, config: AlphaVantageConfig):
        self.config = config
        self._session: Optional[aiohttp.ClientSession] = None
        self._running = False
        self._update_task = None
        self._callbacks: List[Callable] = []
        
        # Rate limiter (5 calls per minute for free tier)
        self._rate_limiter = RateLimiter(calls_per_minute=5)
        
        # Cache to avoid redundant API calls
        self._quote_cache: Dict[str, tuple] = {}  # symbol -> (data, timestamp)
        self._cache_ttl = 60  # 1 minute cache
        
        # Metrics
        self._total_requests = 0
        self._failed_requests = 0
        self._api_calls_remaining = 5
        
        # Set default functions if not specified
        if not config.functions:
            config.functions = ["GLOBAL_QUOTE"] if not config.enable_intraday else ["TIME_SERIES_INTRADAY"]
    
    async def start(self):
        """Start the adapter."""
        if self._running:
            return
        
        self._session = aiohttp.ClientSession()
        self._running = True
        self._update_task = asyncio.create_task(self._update_loop())
        logger.info(f"Alpha Vantage adapter started for {len(self.config.symbols)} symbols")
    
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
        
        logger.info("Alpha Vantage adapter stopped")
    
    def add_callback(self, callback: Callable[[DataUpdate], None]):
        """Add callback for data updates."""
        self._callbacks.append(callback)
    
    async def _update_loop(self):
        """Main update loop with rate limiting."""
        symbol_index = 0
        
        while self._running:
            try:
                # Cycle through symbols
                if self.config.symbols:
                    symbol = self.config.symbols[symbol_index]
                    
                    # Check cache first
                    if not self._is_cache_valid(symbol):
                        # Fetch fresh data
                        for function in self.config.functions:
                            if function == "GLOBAL_QUOTE":
                                await self._fetch_quote(symbol)
                            elif function == "TIME_SERIES_INTRADAY":
                                await self._fetch_intraday(symbol)
                    else:
                        # Use cached data
                        cached_data, _ = self._quote_cache.get(symbol, (None, 0))
                        if cached_data:
                            await self._process_cached_quote(symbol, cached_data)
                    
                    # Move to next symbol
                    symbol_index = (symbol_index + 1) % len(self.config.symbols)
                
                # Sleep to maintain update interval
                await asyncio.sleep(self.config.update_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Update loop error: {e}")
                await asyncio.sleep(5)
    
    def _is_cache_valid(self, symbol: str) -> bool:
        """Check if cached data is still valid."""
        if symbol not in self._quote_cache:
            return False
        
        _, timestamp = self._quote_cache[symbol]
        return time.time() - timestamp < self._cache_ttl
    
    async def _fetch_quote(self, symbol: str):
        """Fetch global quote for a symbol."""
        await self._rate_limiter.acquire()
        
        params = {
            "function": "GLOBAL_QUOTE",
            "symbol": symbol,
            "apikey": self.config.api_key,
            "datatype": self.config.datatype,
        }
        
        try:
            self._total_requests += 1
            
            async with self._session.get(self.BASE_URL, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Check for API errors
                    if "Error Message" in data:
                        logger.error(f"API error: {data['Error Message']}")
                        self._failed_requests += 1
                        return
                    
                    if "Note" in data:
                        logger.warning(f"API note: {data['Note']}")
                        # Rate limit hit
                        self._api_calls_remaining = 0
                        return
                    
                    if "Global Quote" in data:
                        quote_data = data["Global Quote"]
                        
                        # Cache the data
                        self._quote_cache[symbol] = (quote_data, time.time())
                        
                        # Process the quote
                        await self._process_quote(symbol, quote_data)
                else:
                    logger.error(f"HTTP error {response.status}")
                    self._failed_requests += 1
                    
        except Exception as e:
            logger.error(f"Fetch quote error for {symbol}: {e}")
            self._failed_requests += 1
    
    async def _fetch_intraday(self, symbol: str, interval: str = "1min"):
        """Fetch intraday time series data."""
        await self._rate_limiter.acquire()
        
        params = {
            "function": "TIME_SERIES_INTRADAY",
            "symbol": symbol,
            "interval": interval,
            "apikey": self.config.api_key,
            "outputsize": self.config.outputsize,
            "datatype": self.config.datatype,
        }
        
        try:
            self._total_requests += 1
            
            async with self._session.get(self.BASE_URL, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Check for errors
                    if "Error Message" in data:
                        logger.error(f"API error: {data['Error Message']}")
                        self._failed_requests += 1
                        return
                    
                    # Find the time series key
                    time_series_key = f"Time Series ({interval})"
                    if time_series_key in data:
                        time_series = data[time_series_key]
                        
                        # Get the most recent data point
                        if time_series:
                            latest_time = max(time_series.keys())
                            latest_data = time_series[latest_time]
                            
                            # Process as update
                            await self._process_intraday_data(symbol, latest_time, latest_data)
                else:
                    logger.error(f"HTTP error {response.status}")
                    self._failed_requests += 1
                    
        except Exception as e:
            logger.error(f"Fetch intraday error for {symbol}: {e}")
            self._failed_requests += 1
    
    async def _process_quote(self, symbol: str, quote_data: Dict[str, Any]):
        """Process global quote data."""
        try:
            # Extract fields (Alpha Vantage uses numbered keys)
            price = float(quote_data.get("05. price", 0))
            volume = int(quote_data.get("06. volume", 0))
            latest_day = quote_data.get("07. latest trading day", "")
            change = float(quote_data.get("09. change", 0))
            change_percent = quote_data.get("10. change percent", "0%").rstrip("%")
            
            if price <= 0:
                return
            
            # Create metadata
            metadata = {
                "source": "alpha_vantage",
                "open": float(quote_data.get("02. open", 0)),
                "high": float(quote_data.get("03. high", 0)),
                "low": float(quote_data.get("04. low", 0)),
                "volume": volume,
                "latest_trading_day": latest_day,
                "previous_close": float(quote_data.get("08. previous close", 0)),
                "change": change,
                "change_percent": float(change_percent) if change_percent else 0,
            }
            
            # Create data update
            update = DataUpdate(
                symbol=symbol,
                price=price,
                timestamp=time.time(),  # Use current time as AV doesn't provide exact timestamp
                source=DataSource.REST,
                metadata=metadata
            )
            
            # Notify callbacks
            await self._notify_callbacks(update)
            
        except Exception as e:
            logger.error(f"Error processing quote for {symbol}: {e}")
    
    async def _process_cached_quote(self, symbol: str, quote_data: Dict[str, Any]):
        """Process cached quote data."""
        await self._process_quote(symbol, quote_data)
    
    async def _process_intraday_data(self, symbol: str, timestamp_str: str, data: Dict[str, Any]):
        """Process intraday time series data."""
        try:
            # Parse timestamp
            timestamp = time.mktime(time.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S"))
            
            # Extract OHLCV data
            open_price = float(data.get("1. open", 0))
            high = float(data.get("2. high", 0))
            low = float(data.get("3. low", 0))
            close = float(data.get("4. close", 0))
            volume = int(data.get("5. volume", 0))
            
            if close <= 0:
                return
            
            # Create metadata
            metadata = {
                "source": "alpha_vantage_intraday",
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "timestamp_str": timestamp_str,
            }
            
            # Create data update
            update = DataUpdate(
                symbol=symbol,
                price=close,
                timestamp=timestamp,
                source=DataSource.REST,
                metadata=metadata
            )
            
            # Notify callbacks
            await self._notify_callbacks(update)
            
        except Exception as e:
            logger.error(f"Error processing intraday data for {symbol}: {e}")
    
    async def _notify_callbacks(self, update: DataUpdate):
        """Notify all callbacks of new data."""
        for callback in self._callbacks:
            try:
                await callback(update)
            except Exception as e:
                logger.error(f"Callback error: {e}")
    
    async def get_technical_indicators(self, symbol: str, indicator: str, 
                                     interval: str = "daily", **kwargs) -> Optional[Dict[str, Any]]:
        """Get technical indicators for a symbol."""
        await self._rate_limiter.acquire()
        
        params = {
            "function": indicator.upper(),
            "symbol": symbol,
            "interval": interval,
            "apikey": self.config.api_key,
            **kwargs  # Additional indicator-specific parameters
        }
        
        try:
            async with self._session.get(self.BASE_URL, params=params) as response:
                if response.status == 200:
                    return await response.json()
                    
        except Exception as e:
            logger.error(f"Error getting {indicator} for {symbol}: {e}")
        
        return None
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get adapter metrics."""
        success_rate = 1 - (self._failed_requests / self._total_requests) if self._total_requests > 0 else 0
        
        return {
            "total_requests": self._total_requests,
            "failed_requests": self._failed_requests,
            "success_rate": success_rate,
            "api_calls_remaining": self._api_calls_remaining,
            "cache_size": len(self._quote_cache),
            "symbols_tracked": len(self.config.symbols),
            "rate_limit": f"{self._rate_limiter.calls_per_minute} calls/min",
        }
    
    def get_supported_indicators(self) -> List[str]:
        """Get list of supported technical indicators."""
        return [
            "SMA",      # Simple Moving Average
            "EMA",      # Exponential Moving Average
            "RSI",      # Relative Strength Index
            "MACD",     # Moving Average Convergence Divergence
            "STOCH",    # Stochastic Oscillator
            "ADX",      # Average Directional Index
            "CCI",      # Commodity Channel Index
            "AROON",    # Aroon Indicator
            "BBANDS",   # Bollinger Bands
            "AD",       # Chaikin A/D Line
            "OBV",      # On Balance Volume
        ]