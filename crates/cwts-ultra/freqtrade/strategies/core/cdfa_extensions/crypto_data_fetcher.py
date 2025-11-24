#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Data Fetcher for CDFA Extensions

Provides data fetching capabilities from multiple sources, with special support for:
- Yahoo Finance for stocks and crypto
- Cryptocurrency specific features
- Integration with CDFA analysis modules

Author: Created on May 6, 2025
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
import datetime
import time
from functools import lru_cache
import os
import json

# Import yfinance
try:
    import yfinance as yf
    import inspect
    # Check yfinance version and available parameters
    YFINANCE_AVAILABLE = True
    # Check if 'progress' is a valid parameter
    HISTORY_PARAMS = inspect.signature(yf.Ticker.history).parameters
    SUPPORTS_PROGRESS = 'progress' in HISTORY_PARAMS
except ImportError:
    YFINANCE_AVAILABLE = False
    SUPPORTS_PROGRESS = False
    logging.warning("yfinance not available. Install with 'pip install yfinance'")

# Try to import CCXT for additional crypto exchange support
try:
    import ccxt
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False
    logging.warning("ccxt not available. Install with 'pip install ccxt' for additional crypto exchange support")

class MarketDataFetcher:
    """
    Enhanced data fetcher for retrieving market data from various sources
    including Yahoo Finance, with special support for cryptocurrency markets.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger(f"{__name__}.MarketDataFetcher")
        
        # Default configuration
        self.default_config = {
            # General settings
            "cache_ttl": 3600,  # 1 hour
            "max_retries": 3,
            "retry_wait": 5,  # Seconds between retries
            "data_cache_dir": os.path.expanduser("~/.cdfa/cache"),
            "use_disk_cache": True,
            
            # Yahoo Finance settings
            "auto_adjust": True,  # Auto-adjust data from Yahoo Finance
            "back_adjust": False,
            "proxy": None,
            "default_timeframe": "1d",
            "default_period": "1y",
            
            # Crypto settings
            "default_crypto_exchange": "binance",
            "default_quote_currency": "USD",
            "alternative_quote_currencies": ["USDT", "USDC", "BTC", "ETH"],
            "top_crypto_pairs": ["BTC/USD", "ETH/USD", "BNB/USD", "SOL/USD", "XRP/USD"],
            "crypto_timeframes": {
                "1m": "1 minute",
                "5m": "5 minutes",
                "15m": "15 minutes",
                "1h": "1 hour",
                "4h": "4 hour",
                "1d": "1 day"
            },
            
            # CCXT settings
            "ccxt_timeout": 30000,
            "ccxt_enableRateLimit": True,
            "ccxt_exchanges": ["binance", "coinbase", "kraken", "kucoin"]
        }
        
        # Merge with provided config
        self.config = {**self.default_config, **(config or {})}
        
        # Create cache directory if it doesn't exist
        if self.config["use_disk_cache"]:
            os.makedirs(self.config["data_cache_dir"], exist_ok=True)
        
        # Initialize CCXT exchange connections if available
        self.exchanges = {}
        if CCXT_AVAILABLE:
            self._initialize_exchanges()
        
        # Memory cache for data
        self._data_cache = {}
        
    def _initialize_exchanges(self):
        """Initialize connections to cryptocurrency exchanges using CCXT."""
        if not CCXT_AVAILABLE:
            return
            
        for exchange_id in self.config["ccxt_exchanges"]:
            try:
                # Get exchange class by id
                exchange_class = getattr(ccxt, exchange_id)
                
                # Instantiate exchange with config
                self.exchanges[exchange_id] = exchange_class({
                    'timeout': self.config["ccxt_timeout"],
                    'enableRateLimit': self.config["ccxt_enableRateLimit"]
                })
                self.logger.info(f"Initialized connection to {exchange_id}")
            except Exception as e:
                self.logger.warning(f"Failed to initialize {exchange_id}: {e}")
                
    def _get_cache_key(self, symbols, period, interval, source, **kwargs):
        """Generate a cache key for the data request."""
        if isinstance(symbols, list):
            symbols_str = '_'.join(sorted(symbols))
        else:
            symbols_str = symbols
            
        return f"{source}_{symbols_str}_{period}_{interval}_{hash(str(kwargs))}"
    
    def _get_disk_cache_path(self, cache_key):
        """Get disk cache file path for the cache key."""
        # Create a safe filename from the cache key
        safe_key = ''.join(c if c.isalnum() else '_' for c in cache_key)
        return os.path.join(self.config["data_cache_dir"], f"{safe_key}.parquet")
    
    def _load_from_cache(self, cache_key):
        """Load data from cache if available and not expired."""
        # Check memory cache first
        if cache_key in self._data_cache:
            data, timestamp = self._data_cache[cache_key]
            if time.time() - timestamp < self.config["cache_ttl"]:
                return data
                
        # Check disk cache if enabled
        if self.config["use_disk_cache"]:
            cache_path = self._get_disk_cache_path(cache_key)
            if os.path.exists(cache_path):
                # Check cache file modification time
                mtime = os.path.getmtime(cache_path)
                if time.time() - mtime < self.config["cache_ttl"]:
                    try:
                        # Load from disk cache
                        data_dict = {}
                        df = pd.read_parquet(cache_path)
                        
                        # Split multi-symbol dataframe if needed
                        if 'symbol' in df.columns:
                            for symbol in df['symbol'].unique():
                                data_dict[symbol] = df[df['symbol'] == symbol].drop('symbol', axis=1)
                        else:
                            # Single symbol data
                            symbol = cache_key.split('_')[1]
                            data_dict[symbol] = df
                            
                        # Store in memory cache and return
                        self._data_cache[cache_key] = (data_dict, time.time())
                        return data_dict
                    except Exception as e:
                        self.logger.warning(f"Error loading from disk cache: {e}")
                        
        return None
    
    def _save_to_cache(self, cache_key, data_dict):
        """Save data to cache."""
        # Save to memory cache
        self._data_cache[cache_key] = (data_dict, time.time())
        
        # Save to disk cache if enabled
        if self.config["use_disk_cache"] and data_dict:
            cache_path = self._get_disk_cache_path(cache_key)
            try:
                # Convert to a single dataframe with symbol column if multiple symbols
                if len(data_dict) > 1:
                    dfs = []
                    for symbol, df in data_dict.items():
                        df_copy = df.copy()
                        df_copy['symbol'] = symbol
                        dfs.append(df_copy)
                    combined_df = pd.concat(dfs)
                    combined_df.to_parquet(cache_path, index=True)
                else:
                    # Single symbol data
                    symbol, df = next(iter(data_dict.items()))
                    df.to_parquet(cache_path, index=True)
            except Exception as e:
                self.logger.warning(f"Error saving to disk cache: {e}")
                
    def fetch_yahoo_data(self, 
                        symbols: Union[str, List[str]], 
                        start: Optional[Union[str, datetime.datetime]] = None,
                        end: Optional[Union[str, datetime.datetime]] = None,
                        period: Optional[str] = None,
                        interval: Optional[str] = None,
                        **kwargs) -> Dict[str, pd.DataFrame]:
        """
        Fetch data from Yahoo Finance using yfinance.
        
        Args:
            symbols: Symbol or list of symbols
            start: Start date (or use period)
            end: End date (default: today)
            period: Period string (default: 1y)
            interval: Data interval (default: 1d)
            **kwargs: Additional parameters for yfinance
            
        Returns:
            Dictionary of symbol -> dataframe
        """
        if not YFINANCE_AVAILABLE:
            self.logger.error("yfinance is not available. Install with 'pip install yfinance'")
            return {}
            
        # Set defaults
        interval = interval or self.config["default_timeframe"]
        period = period or self.config["default_period"]
        
        # Process parameters
        if isinstance(symbols, str):
            symbols = [symbols]
        
        # Convert cryptocurrency format for Yahoo Finance
        yahoo_symbols = []
        symbol_map = {}  # Map to track original symbol to yahoo symbol
        
        for symbol in symbols:
            # Debug log original symbol
            self.logger.info(f"Processing symbol: {symbol}")
            # Handle cryptocurrency format conversion
            if '/' in symbol:
                base, quote = symbol.split('/')
                
                # Try different formats based on currency type
                if quote in ["USDT", "BUSD", "USDC"]:
                    # For stablecoins, try both Base-USD and Base-Quote
                    yahoo_symbol = f"{base}-USD"
                    yahoo_symbols.append(yahoo_symbol)
                    symbol_map[yahoo_symbol] = symbol
                    
                    yahoo_symbol_alt = f"{base}-{quote}"
                    yahoo_symbols.append(yahoo_symbol_alt)
                    symbol_map[yahoo_symbol_alt] = symbol
                else:
                    # For other quote currencies
                    yahoo_symbol = f"{base}-{quote}"
                    yahoo_symbols.append(yahoo_symbol)
                    symbol_map[yahoo_symbol] = symbol
                    
                    # Try traditional forex format as fallback
                    if quote in ["USD", "EUR", "JPY", "GBP"]:
                        yahoo_symbol_alt = f"{base}{quote}=X"
                        yahoo_symbols.append(yahoo_symbol_alt)
                        symbol_map[yahoo_symbol_alt] = symbol
            else:
                # Not a cryptocurrency pair, use as is
                yahoo_symbols.append(symbol)
                symbol_map[symbol] = symbol
        
        # Debug logging
        self.logger.info(f"Converted symbols {symbols} to yahoo symbols: {yahoo_symbols}")
        
        # Generate cache key using original symbols
        cache_key = self._get_cache_key(
            symbols, period, interval, "yahoo", 
            start=start, end=end, **kwargs
        )
        
        # Check cache
        cached_data = self._load_from_cache(cache_key)
        if cached_data is not None:
            return cached_data
            
        # Process parameters for yfinance
        params = {
            "auto_adjust": kwargs.get("auto_adjust", self.config["auto_adjust"]),
            "back_adjust": kwargs.get("back_adjust", self.config["back_adjust"]),
            "proxy": kwargs.get("proxy", self.config["proxy"]),
        }
        
        # Only add 'progress' parameter if supported in this yfinance version
        if SUPPORTS_PROGRESS:
            params["progress"] = kwargs.get("progress", False)
        
        # Initialize result dictionary
        result = {}
        
        # Fetch data for each symbol with retry logic
        for yahoo_symbol in yahoo_symbols:
            # Skip if we already have data for the original symbol
            original_symbol = symbol_map[yahoo_symbol]
            if original_symbol in result:
                continue
                
            for retry in range(self.config["max_retries"]):
                try:
                    self.logger.info(f"Fetching data for {yahoo_symbol} (attempt {retry+1})")
                    
                    # Create Ticker object
                    ticker = yf.Ticker(yahoo_symbol)
                    
                    # Determine how to fetch the data
                    if start is not None and end is not None:
                        # Fetch with date range
                        df = ticker.history(start=start, end=end, interval=interval, **params)
                    else:
                        # Fetch with period
                        df = ticker.history(period=period, interval=interval, **params)
                    
                    # Process the dataframe
                    if not df.empty:
                        # Convert column names to lowercase for consistency with CDFA
                        df.columns = [col.lower() for col in df.columns]
                        
                        # Rename columns to match CDFA format
                        column_mapping = {
                            'adj close': 'adj_close',
                            'stock splits': 'splits',
                            'dividends': 'dividends'
                        }
                        df = df.rename(columns=column_mapping)
                        
                        # Ensure index is datetime
                        if not isinstance(df.index, pd.DatetimeIndex):
                            df.index = pd.to_datetime(df.index)
                        
                        # Add to result using original symbol
                        result[original_symbol] = df
                        
                        # Log success
                        self.logger.info(f"Successfully fetched data for {original_symbol} using {yahoo_symbol}")
                        
                        break  # Success, exit retry loop
                    else:
                        self.logger.warning(f"No data returned for {yahoo_symbol}")
                        # Try again if this is not the last attempt
                        if retry < self.config["max_retries"] - 1:
                            time.sleep(self.config["retry_wait"])
                except Exception as e:
                    self.logger.error(f"Error fetching data for {yahoo_symbol}: {e}")
                    # Try again if this is not the last attempt
                    if retry < self.config["max_retries"] - 1:
                        time.sleep(self.config["retry_wait"])
                        
        # Cache the result
        if result:
            self._save_to_cache(cache_key, result)
            
        return result
    
    
    def fetch_crypto_data(self, 
                        cryptos: Union[str, List[str]], 
                        period: str = '1y', 
                        interval: str = '1d',
                        quote_currencies: Optional[List[str]] = None,
                        exchange: Optional[str] = None,
                        prefer_ccxt: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Fetch cryptocurrency data from Yahoo Finance or CCXT.
        
        Args:
            cryptos: Cryptocurrency symbol or list of symbols (e.g., 'BTC', 'ETH')
            period: Time period to analyze
            interval: Data interval
            quote_currencies: Quote currencies to try (default: from config)
            exchange: Specific exchange to use (default: from config)
            prefer_ccxt: Whether to prefer CCXT over Yahoo Finance
            
        Returns:
            Dictionary of crypto symbol -> dataframe
        """
        # Handle single symbol
        if isinstance(cryptos, str):
            cryptos = [cryptos]
            
        # Set default quote currencies
        if quote_currencies is None:
            quote_currencies = [self.config["default_quote_currency"]] + self.config["alternative_quote_currencies"]
            
        # Set default exchange
        exchange = exchange or self.config["default_crypto_exchange"]
        
        # Determine data source
        if prefer_ccxt and CCXT_AVAILABLE and exchange in self.exchanges:
            self.logger.info(f"Using CCXT with {exchange} for crypto data")
            return self._fetch_crypto_data_ccxt(cryptos, period, interval, quote_currencies, exchange)
        else:
            self.logger.info("Using Yahoo Finance for crypto data")
            return self._fetch_crypto_data_yahoo(cryptos, period, interval, quote_currencies)
    
    def _fetch_crypto_data_yahoo(self, 
                              cryptos: List[str], 
                              period: str,
                              interval: str,
                              quote_currencies: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Fetch cryptocurrency data from Yahoo Finance.
        """
        if not YFINANCE_AVAILABLE:
            self.logger.error("yfinance is not available. Install with 'pip install yfinance'")
            return {}
            
        # Generate cache key
        cache_key = self._get_cache_key(
            cryptos, period, interval, "yahoo_crypto", 
            quote_currencies=quote_currencies
        )
        
        # Check cache
        cached_data = self._load_from_cache(cache_key)
        if cached_data is not None:
            return cached_data
            
        # Initialize result dictionary
        result = {}
        
        # Try different quote currencies for each crypto
        for crypto in cryptos:
            for quote in quote_currencies:
                symbol = f"{crypto}-{quote}"
                
                try:
                    # Fetch data for this pair
                    ticker_data = self.fetch_yahoo_data(
                        symbol, 
                        period=period, 
                        interval=interval
                    )
                    
                    # Check if we got data
                    if symbol in ticker_data and not ticker_data[symbol].empty:
                        # Store with original crypto symbol (without quote currency)
                        if crypto not in result:
                            # Add base symbol only once (preferring the first successful quote currency)
                            df = ticker_data[symbol].copy()
                            
                            # Add quote currency as metadata
                            df.attrs['quote_currency'] = quote
                            df.attrs['source'] = 'yahoo'
                            
                            # Add to result
                            result[crypto] = df
                            self.logger.info(f"Successfully fetched {crypto} data quoted in {quote}")
                            
                            # Break the quote currency loop for this crypto
                            break
                except Exception as e:
                    self.logger.warning(f"Failed to fetch {symbol}: {e}")
                    
        # Cache the result
        if result:
            self._save_to_cache(cache_key, result)
            
        return result
    
    def _fetch_crypto_data_ccxt(self, 
                             cryptos: List[str], 
                             period: str,
                             interval: str,
                             quote_currencies: List[str],
                             exchange_id: str) -> Dict[str, pd.DataFrame]:
        """
        Fetch cryptocurrency data from CCXT.
        """
        if not CCXT_AVAILABLE:
            self.logger.error("CCXT is not available. Install with 'pip install ccxt'")
            return {}
            
        if exchange_id not in self.exchanges:
            self.logger.error(f"Exchange {exchange_id} not initialized")
            return {}
            
        # Get exchange instance
        exchange = self.exchanges[exchange_id]
        
        # Generate cache key
        cache_key = self._get_cache_key(
            cryptos, period, interval, f"ccxt_{exchange_id}", 
            quote_currencies=quote_currencies
        )
        
        # Check cache
        cached_data = self._load_from_cache(cache_key)
        if cached_data is not None:
            return cached_data
            
        # Convert period to timeframe that CCXT understands
        if interval in self.config["crypto_timeframes"]:
            timeframe = interval
        else:
            # Default to daily
            timeframe = "1d"
            
        # Convert period to number of candles
        period_map = {
            "1d": 1, "5d": 5, "1mo": 30, "3mo": 90, 
            "6mo": 180, "1y": 365, "2y": 730, "5y": 1825,
            "max": 10000  # Default to a large number
        }
        limit = period_map.get(period, 365)
        
        # Initialize result dictionary
        result = {}
        
        # Try different quote currencies for each crypto
        for crypto in cryptos:
            for quote in quote_currencies:
                symbol = f"{crypto}/{quote}"
                
                try:
                    # Check if the symbol exists on the exchange
                    markets = exchange.load_markets()
                    if symbol not in markets:
                        continue
                        
                    # Fetch OHLCV data
                    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
                    
                    # Convert to dataframe
                    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    
                    # Convert timestamp to datetime index
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df.set_index('timestamp', inplace=True)
                    
                    # Add metadata
                    df.attrs['quote_currency'] = quote
                    df.attrs['source'] = f'ccxt_{exchange_id}'
                    
                    # Store with original crypto symbol (without quote currency)
                    if crypto not in result:
                        # Add base symbol only once (preferring the first successful quote currency)
                        result[crypto] = df
                        self.logger.info(f"Successfully fetched {crypto} data from {exchange_id} quoted in {quote}")
                        
                        # Break the quote currency loop for this crypto
                        break
                except Exception as e:
                    self.logger.warning(f"Failed to fetch {symbol} from {exchange_id}: {e}")
                    
        # Cache the result
        if result:
            self._save_to_cache(cache_key, result)
            
        return result
    
    def fetch_data_for_cdfa(self, 
                          symbols: Union[str, List[str]],
                          source: str = 'yahoo',
                          **kwargs) -> Dict[str, pd.DataFrame]:
        """
        Fetch data from specified source and format it for CDFA processing.
        
        Args:
            symbols: Symbol or list of symbols
            source: Data source ('yahoo', 'crypto', 'ccxt', etc.)
            **kwargs: Additional parameters for the specific source
            
        Returns:
            Dictionary of symbol -> dataframe ready for CDFA
        """
        if source.lower() == 'yahoo':
            data_dict = self.fetch_yahoo_data(symbols, **kwargs)
        elif source.lower() == 'crypto':
            data_dict = self.fetch_crypto_data(symbols, **kwargs)
        elif source.lower().startswith('ccxt'):
            exchange = source.split('_')[1] if '_' in source else self.config["default_crypto_exchange"]
            data_dict = self.fetch_crypto_data(
                symbols, 
                prefer_ccxt=True, 
                exchange=exchange, 
                **kwargs
            )
        else:
            self.logger.error(f"Unsupported data source: {source}")
            return {}
            
        # Process data for CDFA compatibility
        for symbol, df in data_dict.items():
            # Add any additional processing needed for CDFA
            # For example, ensuring required columns exist
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in required_cols:
                if col not in df.columns:
                    self.logger.warning(f"Missing required column {col} for {symbol}")
                    
            # Add any additional features that CDFA expects
            # For example, calculate returns if needed
            if 'close' in df.columns:
                if 'returns' not in df.columns:
                    df['returns'] = df['close'].pct_change().fillna(0)
                
                # Add log returns
                if 'log_returns' not in df.columns:
                    df['log_returns'] = np.log1p(df['returns']).fillna(0)
                    
                # Add rolling volatility
                if 'volatility_20d' not in df.columns:
                    df['volatility_20d'] = df['returns'].rolling(20).std().fillna(0)
                
                # Add Bollinger Bands
                if all(col not in df.columns for col in ['bb_upper', 'bb_lower', 'bb_middle']):
                    df['bb_middle'] = df['close'].rolling(20).mean()
                    stddev = df['close'].rolling(20).std()
                    df['bb_upper'] = df['bb_middle'] + 2 * stddev
                    df['bb_lower'] = df['bb_middle'] - 2 * stddev
                
                # For cryptocurrencies, add additional metrics
                if symbol in kwargs.get('cryptos', []) or source.lower() in ['crypto', 'ccxt']:
                    # Add VWAP (Volume Weighted Average Price)
                    if 'vwap' not in df.columns and 'volume' in df.columns:
                        df['vwap'] = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
                    
                    # Add market cap estimate if available
                    if 'market_cap' not in df.columns and hasattr(df, 'attrs') and 'supply' in df.attrs:
                        df['market_cap'] = df['close'] * df.attrs['supply']
                
        return data_dict
    
    def get_crypto_market_info(self) -> Dict[str, Any]:
        """
        Get information about available cryptocurrency markets.
        
        Returns:
            Dictionary with market information
        """
        result = {
            "top_pairs": [],
            "exchanges": {},
            "timestamps": {}
        }
        
        # Get top pairs from Yahoo Finance
        if YFINANCE_AVAILABLE:
            try:
                for pair in self.config["top_crypto_pairs"]:
                    crypto = pair.split('/')[0]
                    quote = pair.split('/')[1]
                    ticker_symbol = f"{crypto}-{quote}"
                    
                    # Just check if data is available
                    ticker = yf.Ticker(ticker_symbol)
                    info = ticker.info
                    
                    if info and 'regularMarketPrice' in info:
                        result["top_pairs"].append({
                            "symbol": pair,
                            "price": info.get('regularMarketPrice'),
                            "change_24h": info.get('regularMarketChangePercent', 0),
                            "source": "yahoo"
                        })
            except Exception as e:
                self.logger.warning(f"Error getting crypto market info from Yahoo: {e}")
                
        # Get market info from CCXT
        if CCXT_AVAILABLE:
            for exchange_id, exchange in self.exchanges.items():
                try:
                    # Get exchange info
                    exchange.load_markets()
                    
                    # Get ticker info for top pairs
                    markets = []
                    for pair in self.config["top_crypto_pairs"]:
                        if pair in exchange.markets:
                            ticker = exchange.fetch_ticker(pair)
                            markets.append({
                                "symbol": pair,
                                "price": ticker.get('last'),
                                "change_24h": ticker.get('percentage') if 'percentage' in ticker else ticker.get('change'),
                                "volume_24h": ticker.get('quoteVolume') or ticker.get('baseVolume'),
                                "source": f"ccxt_{exchange_id}"
                            })
                            
                    result["exchanges"][exchange_id] = {
                        "pairs_count": len(exchange.markets),
                        "top_markets": markets,
                        "has_websocket": hasattr(exchange, 'has') and exchange.has.get('ws', False),
                        "maker_fee": exchange.fees.get('trading', {}).get('maker', None) if hasattr(exchange, 'fees') else None,
                        "taker_fee": exchange.fees.get('trading', {}).get('taker', None) if hasattr(exchange, 'fees') else None
                    }
                    
                    result["timestamps"][exchange_id] = exchange.milliseconds()
                except Exception as e:
                    self.logger.warning(f"Error getting market info from {exchange_id}: {e}")
                    
        return result
    
    def get_market_holidays(self, country: str = 'US') -> List[Dict[str, Any]]:
        """
        Get market holidays for a given country.
        
        Args:
            country: Country code
            
        Returns:
            List of holiday information dictionaries
        """
        if not YFINANCE_AVAILABLE:
            self.logger.error("yfinance is not available. Install with 'pip install yfinance'")
            return []
            
        try:
            # Using pandas_market_calendars if available
            try:
                import pandas_market_calendars as mcal
                calendar = mcal.get_calendar(f"NYSE" if country == "US" else country)
                holidays = calendar.holidays().holidays
                
                return [{"date": holiday.strftime("%Y-%m-%d"), "name": str(holiday)} for holiday in holidays]
            except ImportError:
                # Fallback to manual holiday checking by testing market open status
                today = datetime.datetime.now().date()
                holidays = []
                
                # Check the next 30 days
                for i in range(30):
                    check_date = today + datetime.timedelta(days=i)
                    
                    # Skip weekends
                    if check_date.weekday() >= 5:
                        continue
                        
                    # Format date for yfinance
                    date_str = check_date.strftime("%Y-%m-%d")
                    
                    # Use SPY as a proxy for US market
                    symbol = "SPY" if country == "US" else "^FTSE" if country == "UK" else None
                    
                    if symbol:
                        # Try to get data for this date
                        ticker = yf.Ticker(symbol)
                        history = ticker.history(start=date_str, end=date_str)
                        
                        # If no data, it might be a holiday
                        if history.empty:
                            holidays.append({"date": date_str, "name": "Market Holiday"})
                            
                return holidays
        except Exception as e:
            self.logger.error(f"Error getting market holidays: {e}")
            return []
