"""Stock data collection from free sources"""

import yfinance as yf
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


class StockDataCollector:
    """Collects stock market data from free sources"""
    
    def __init__(self):
        """Initialize the stock data collector"""
        self.cache = {}
        self.cache_duration = timedelta(minutes=5)
        self.sector_etfs = {
            "XLK": "Technology",
            "XLF": "Financials", 
            "XLE": "Energy",
            "XLV": "Healthcare",
            "XLI": "Industrials",
            "XLY": "Consumer Discretionary",
            "XLP": "Consumer Staples",
            "XLU": "Utilities",
            "XLRE": "Real Estate"
        }
        
    def get_stock_data(self, symbol: str, period: str = "1mo") -> pd.DataFrame:
        """
        Fetch stock data from Yahoo Finance (free)
        
        Args:
            symbol: Stock ticker symbol
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            
        Returns:
            DataFrame with OHLCV data
        """
        cache_key = f"{symbol}_{period}"
        
        # Check cache
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if datetime.now() - timestamp < self.cache_duration:
                logger.debug(f"Returning cached data for {symbol}")
                return cached_data
                
        try:
            # Fetch fresh data
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            if data.empty:
                raise ValueError(f"No data returned for symbol {symbol}")
            
            # Cache the data
            self.cache[cache_key] = (data, datetime.now())
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            raise ValueError(f"Failed to fetch data for {symbol}: {str(e)}")
    
    def get_intraday_data(self, symbol: str, interval: str = "5m") -> pd.DataFrame:
        """
        Fetch intraday data for day trading
        
        Args:
            symbol: Stock ticker symbol
            interval: Time interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h)
            
        Returns:
            DataFrame with intraday OHLCV data
        """
        try:
            ticker = yf.Ticker(symbol)
            
            # Get today's data with specified interval
            data = ticker.history(period="1d", interval=interval)
            
            if data.empty:
                # Try last 5 days if today has no data (weekend/holiday)
                data = ticker.history(period="5d", interval=interval)
                
            return data
            
        except Exception as e:
            logger.error(f"Error fetching intraday data for {symbol}: {str(e)}")
            raise ValueError(f"Failed to fetch intraday data for {symbol}: {str(e)}")
    
    def get_sector_data(self) -> Dict[str, float]:
        """
        Get sector performance data
        
        Returns:
            Dictionary with sector names and performance percentages
        """
        performance = {}
        
        for ticker, name in self.sector_etfs.items():
            try:
                data = self.get_stock_data(ticker, period="5d")
                if len(data) >= 2:
                    # Calculate percentage change
                    perf = (data['Close'].iloc[-1] / data['Close'].iloc[0] - 1) * 100
                    performance[name] = round(perf, 2)
                else:
                    logger.warning(f"Insufficient data for sector {name}")
                    performance[name] = 0.0
                    
            except Exception as e:
                logger.error(f"Error fetching sector data for {ticker}: {str(e)}")
                performance[name] = 0.0
                
        return performance
    
    def get_multiple_stocks(self, symbols: List[str], period: str = "5d") -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple stocks in parallel
        
        Args:
            symbols: List of stock ticker symbols
            period: Time period for data
            
        Returns:
            Dictionary mapping symbols to DataFrames
        """
        results = {}
        
        def fetch_single_stock(symbol):
            try:
                return symbol, self.get_stock_data(symbol, period)
            except Exception as e:
                logger.error(f"Error fetching {symbol}: {str(e)}")
                return symbol, None
        
        # Use thread pool for parallel fetching
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(fetch_single_stock, symbol) for symbol in symbols]
            
            for future in as_completed(futures):
                symbol, data = future.result()
                if data is not None and not data.empty:
                    results[symbol] = data
                    
        return results
    
    def get_quote(self, symbol: str) -> Dict[str, float]:
        """
        Get real-time quote for a stock
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Dictionary with current price, change, volume, etc.
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Extract key quote data
            quote = {
                'symbol': symbol,
                'price': info.get('regularMarketPrice', info.get('currentPrice', 0)),
                'previous_close': info.get('regularMarketPreviousClose', 0),
                'open': info.get('regularMarketOpen', 0),
                'day_high': info.get('regularMarketDayHigh', 0),
                'day_low': info.get('regularMarketDayLow', 0),
                'volume': info.get('regularMarketVolume', 0),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'dividend_yield': info.get('dividendYield', 0),
                '52_week_high': info.get('fiftyTwoWeekHigh', 0),
                '52_week_low': info.get('fiftyTwoWeekLow', 0)
            }
            
            # Calculate change and change percentage
            if quote['previous_close'] > 0:
                quote['change'] = quote['price'] - quote['previous_close']
                quote['change_percent'] = (quote['change'] / quote['previous_close']) * 100
            else:
                quote['change'] = 0
                quote['change_percent'] = 0
                
            return quote
            
        except Exception as e:
            logger.error(f"Error fetching quote for {symbol}: {str(e)}")
            return {}
    
    def get_market_indices(self) -> Dict[str, Dict[str, float]]:
        """
        Get major market indices data
        
        Returns:
            Dictionary with index data
        """
        indices = {
            '^GSPC': 'S&P 500',
            '^DJI': 'Dow Jones',
            '^IXIC': 'NASDAQ',
            '^VIX': 'VIX',
            '^TNX': '10-Year Treasury'
        }
        
        results = {}
        
        for symbol, name in indices.items():
            try:
                quote = self.get_quote(symbol)
                if quote:
                    results[name] = quote
            except Exception as e:
                logger.error(f"Error fetching index {name}: {str(e)}")
                
        return results
    
    def clear_cache(self):
        """Clear the data cache"""
        self.cache.clear()
        logger.info("Data cache cleared")