"""
Alpha Vantage API Client for German Stocks
High-performance client with German market support and rate limiting
"""

import asyncio
import aiohttp
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import json
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class AlphaVantageConfig:
    """Configuration for Alpha Vantage API client"""
    api_key: str
    tier: str = "free"
    timeout: int = 30
    max_retries: int = 3
    base_url: str = "https://www.alphavantage.co/query"
    
    # Rate limits by tier
    rate_limits: Dict[str, Dict[str, int]] = None
    
    def __post_init__(self):
        if self.rate_limits is None:
            self.rate_limits = {
                'free': {'calls_per_minute': 5, 'daily_limit': 500},
                'starter': {'calls_per_minute': 30, 'daily_limit': 5000},
                'professional': {'calls_per_minute': 60, 'daily_limit': 10000},
                'enterprise': {'calls_per_minute': 120, 'daily_limit': 25000}
            }


class AlphaVantageClient:
    """
    High-performance Alpha Vantage API client with German stock support
    """
    
    def __init__(self, config: AlphaVantageConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Rate limiting
        self.current_limit = config.rate_limits.get(
            config.tier, 
            config.rate_limits['free']
        )
        
        # Request tracking
        self.requests_made = 0
        self.daily_requests = 0
        self.last_request_time = None
        self.request_times = []  # Track request times for rate limiting
        
        # German stock mapping
        self.german_exchanges = {
            'XETRA': {'suffix': '.DE', 'currency': 'EUR'},
            'STUTTGART': {'suffix': '.STU', 'currency': 'EUR'},
            'MUNICH': {'suffix': '.MUN', 'currency': 'EUR'},
            'BERLIN': {'suffix': '.BER', 'currency': 'EUR'}
        }
        
        # DAX 40 symbols for validation
        self.dax_symbols = [
            'SAP.DE', 'ASML.DE', 'LVMH.DE', 'NESN.DE', 'MSFT.DE',
            'AAPL.DE', 'GOOGL.DE', 'AMZN.DE', 'TSLA.DE', 'META.DE',
            'NVDA.DE', 'UNH.DE', 'JNJ.DE', 'V.DE', 'PG.DE',
            'HD.DE', 'MA.DE', 'ABBV.DE', 'BAC.DE', 'ADBE.DE',
            'KO.DE', 'AVGO.DE', 'PEP.DE', 'TMO.DE', 'COST.DE',
            'DHR.DE', 'ABT.DE', 'VZ.DE', 'CRM.DE', 'ACHN.DE',
            'BMW.DE', 'SIE.DE', 'ALV.DE', 'BAS.DE', 'VOW3.DE',
            'DTE.DE', 'MUV2.DE', 'DAI.DE', 'DB1.DE', 'HEN3.DE'
        ]
        
        logger.info(f"Initialized Alpha Vantage client (tier: {config.tier})")
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.disconnect()
    
    async def connect(self) -> bool:
        """Create HTTP session"""
        try:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.config.timeout)
            )
            
            # Test connection with health check
            health_ok = await self.health_check()
            if health_ok:
                logger.info("Alpha Vantage client connected successfully")
                return True
            else:
                logger.error("Alpha Vantage health check failed")
                return False
                
        except Exception as e:
            logger.error(f"Failed to connect to Alpha Vantage: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """Close HTTP session"""
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None
            logger.info("Alpha Vantage client disconnected")
        return True
    
    async def _rate_limit(self):
        """Implement smart rate limiting"""
        current_time = time.time()
        
        # Clean old requests (older than 1 minute)
        minute_ago = current_time - 60
        self.request_times = [t for t in self.request_times if t > minute_ago]
        
        # Check if we've hit the per-minute limit
        if len(self.request_times) >= self.current_limit['calls_per_minute']:
            # Wait until the oldest request is more than 1 minute old
            oldest_request = min(self.request_times)
            wait_time = 60 - (current_time - oldest_request)
            if wait_time > 0:
                logger.debug(f"Rate limiting: waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
        
        # Check daily limit
        if self.daily_requests >= self.current_limit['daily_limit']:
            logger.warning("Daily API limit exceeded")
            raise Exception("Daily API limit exceeded")
    
    async def _make_request(self, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Make API request with rate limiting and error handling"""
        if not self.session:
            raise Exception("Client not connected")
        
        # Add API key
        params['apikey'] = self.config.api_key
        
        # Rate limiting
        await self._rate_limit()
        
        start_time = time.perf_counter()
        
        for attempt in range(self.config.max_retries):
            try:
                async with self.session.get(self.config.base_url, params=params) as response:
                    # Track request
                    self.requests_made += 1
                    self.daily_requests += 1
                    self.last_request_time = time.time()
                    self.request_times.append(self.last_request_time)
                    
                    # Measure latency
                    latency_ms = (time.perf_counter() - start_time) * 1000
                    
                    if response.status == 200:
                        data = await response.json()
                        
                        # Check for API error messages
                        if 'Error Message' in data:
                            logger.error(f"Alpha Vantage API error: {data['Error Message']}")
                            return None
                        
                        if 'Note' in data:
                            logger.warning(f"Alpha Vantage rate limit: {data['Note']}")
                            await asyncio.sleep(60)  # Wait a minute on rate limit
                            continue
                        
                        logger.debug(f"API request successful (latency: {latency_ms:.2f}ms)")
                        return data
                    
                    elif response.status == 429:
                        # Rate limit exceeded
                        wait_time = 2 ** attempt
                        logger.warning(f"Rate limited, waiting {wait_time}s (attempt {attempt + 1})")
                        await asyncio.sleep(wait_time)
                        continue
                    
                    else:
                        error_text = await response.text()
                        logger.error(f"HTTP error {response.status}: {error_text}")
                        return None
                        
            except asyncio.TimeoutError:
                logger.warning(f"Request timeout (attempt {attempt + 1})")
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                continue
                
            except Exception as e:
                logger.error(f"Request error: {e} (attempt {attempt + 1})")
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                continue
        
        logger.error("All retry attempts failed")
        return None
    
    async def get_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get real-time quote for a symbol"""
        params = {
            'function': 'GLOBAL_QUOTE',
            'symbol': symbol
        }
        
        data = await self._make_request(params)
        if data and 'Global Quote' in data:
            return data['Global Quote']
        return None
    
    async def get_intraday_data(self, symbol: str, interval: str = '1min') -> Optional[Dict[str, Any]]:
        """Get intraday data for a symbol"""
        params = {
            'function': 'TIME_SERIES_INTRADAY',
            'symbol': symbol,
            'interval': interval,
            'outputsize': 'compact'
        }
        
        data = await self._make_request(params)
        if data:
            # Return the time series data
            key = f'Time Series ({interval})'
            if key in data:
                return {
                    'metadata': data.get('Meta Data', {}),
                    'time_series': data[key]
                }
        return None
    
    async def get_daily_data(self, symbol: str, outputsize: str = 'compact') -> Optional[Dict[str, Any]]:
        """Get daily data for a symbol"""
        params = {
            'function': 'TIME_SERIES_DAILY',
            'symbol': symbol,
            'outputsize': outputsize
        }
        
        data = await self._make_request(params)
        if data and 'Time Series (Daily)' in data:
            return {
                'metadata': data.get('Meta Data', {}),
                'time_series': data['Time Series (Daily)']
            }
        return None
    
    async def get_company_overview(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get company overview/fundamentals"""
        params = {
            'function': 'OVERVIEW',
            'symbol': symbol
        }
        
        return await self._make_request(params)
    
    async def get_news_sentiment(self, tickers: List[str], limit: int = 50) -> Optional[Dict[str, Any]]:
        """Get news sentiment for tickers"""
        params = {
            'function': 'NEWS_SENTIMENT',
            'tickers': ','.join(tickers[:10]),  # Limit to 10 symbols
            'limit': min(limit, 1000),
            'sort': 'LATEST'
        }
        
        return await self._make_request(params)
    
    async def get_exchange_rate(self, from_currency: str, to_currency: str) -> Optional[Dict[str, Any]]:
        """Get exchange rate between currencies"""
        params = {
            'function': 'CURRENCY_EXCHANGE_RATE',
            'from_currency': from_currency,
            'to_currency': to_currency
        }
        
        data = await self._make_request(params)
        if data and 'Realtime Currency Exchange Rate' in data:
            return data['Realtime Currency Exchange Rate']
        return None
    
    def normalize_german_symbol(self, symbol: str, exchange: str = 'XETRA') -> str:
        """Normalize German stock symbol for Alpha Vantage"""
        if exchange in self.german_exchanges:
            suffix = self.german_exchanges[exchange]['suffix']
            if not symbol.endswith(suffix):
                return f"{symbol}{suffix}"
        return symbol
    
    async def get_german_stocks_batch(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get quotes for multiple German stocks"""
        results = {}
        
        for symbol in symbols:
            normalized_symbol = self.normalize_german_symbol(symbol)
            try:
                quote = await self.get_quote(normalized_symbol)
                if quote:
                    results[symbol] = quote
                    
                # Small delay to avoid hitting rate limits
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {e}")
                results[symbol] = {'error': str(e)}
        
        return results
    
    async def health_check(self) -> bool:
        """Check if Alpha Vantage API is accessible"""
        try:
            # Test with a simple quote request
            quote = await self.get_quote('AAPL')
            return quote is not None
            
        except Exception as e:
            logger.error(f"Alpha Vantage health check failed: {e}")
            return False
    
    def get_rate_limit_info(self) -> Dict[str, Any]:
        """Get current rate limit status"""
        current_time = time.time()
        minute_ago = current_time - 60
        recent_requests = [t for t in self.request_times if t > minute_ago]
        
        return {
            'tier': self.config.tier,
            'calls_per_minute': self.current_limit['calls_per_minute'],
            'calls_remaining_this_minute': max(0, self.current_limit['calls_per_minute'] - len(recent_requests)),
            'daily_limit': self.current_limit['daily_limit'],
            'daily_remaining': max(0, self.current_limit['daily_limit'] - self.daily_requests),
            'requests_made_today': self.daily_requests,
            'last_request_time': self.last_request_time
        }