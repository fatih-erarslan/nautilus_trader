"""Stock data collection from free sources."""

import yfinance as yf
import pandas as pd
from typing import Dict, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class StockDataCollector:
    """Collects stock data from free APIs."""
    
    def __init__(self):
        """Initialize the stock data collector."""
        self.cache = {}
        self.cache_duration = timedelta(minutes=5)
        
    def get_stock_data(self, symbol: str, period: str = "1mo") -> pd.DataFrame:
        """Fetch stock data from Yahoo Finance (free).
        
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
                return cached_data
                
        try:
            # Fetch fresh data
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            # Validate data
            if data.empty:
                logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame()
            
            # Cache the data
            self.cache[cache_key] = (data, datetime.now())
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
        
    def get_intraday_data(self, symbol: str, interval: str = "5m") -> pd.DataFrame:
        """Fetch intraday data for day trading.
        
        Args:
            symbol: Stock ticker symbol
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m)
            
        Returns:
            DataFrame with intraday OHLCV data
        """
        try:
            ticker = yf.Ticker(symbol)
            
            # Get today's data
            data = ticker.history(period="1d", interval=interval)
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching intraday data for {symbol}: {e}")
            return pd.DataFrame()
        
    def get_sector_data(self) -> Dict[str, float]:
        """Get sector performance data.
        
        Returns:
            Dictionary of sector name to performance percentage
        """
        sectors = {
            "XLK": "Technology",
            "XLF": "Financials", 
            "XLE": "Energy",
            "XLV": "Healthcare",
            "XLI": "Industrials",
            "XLY": "Consumer Discretionary",
            "XLP": "Consumer Staples",
            "XLU": "Utilities",
            "XLRE": "Real Estate",
        }
        
        performance = {}
        
        for ticker, name in sectors.items():
            try:
                data = self.get_stock_data(ticker, period="5d")
                if len(data) >= 2:
                    # Calculate percentage change
                    pct_change = (data['Close'].iloc[-1] / data['Close'].iloc[0] - 1) * 100
                    performance[name] = round(pct_change, 2)
            except Exception as e:
                logger.error(f"Error getting sector data for {ticker}: {e}")
                performance[name] = 0.0
                
        return performance
    
    def get_market_internals(self) -> Dict[str, any]:
        """Get market breadth and internal indicators.
        
        Returns:
            Dictionary of market internal metrics
        """
        try:
            # Get major indices
            spy_data = self.get_stock_data("SPY", period="1d")
            vix_data = self.get_stock_data("^VIX", period="1d")
            
            internals = {
                "spy_change": 0.0,
                "vix_level": 0.0,
                "vix_change": 0.0,
            }
            
            if len(spy_data) > 0:
                internals["spy_change"] = (
                    (spy_data['Close'].iloc[-1] / spy_data['Open'].iloc[0] - 1) * 100
                )
            
            if len(vix_data) > 0:
                internals["vix_level"] = vix_data['Close'].iloc[-1]
                if len(vix_data) >= 2:
                    internals["vix_change"] = (
                        vix_data['Close'].iloc[-1] - vix_data['Close'].iloc[-2]
                    )
            
            return internals
            
        except Exception as e:
            logger.error(f"Error getting market internals: {e}")
            return {}