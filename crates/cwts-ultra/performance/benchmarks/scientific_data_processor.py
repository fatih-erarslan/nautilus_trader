#!/usr/bin/env python3
"""
Scientific Data Processor - Real Data Integration
Replaces mock data generators with scientifically-validated real data processing
Implements high-fidelity market data simulation and historical data analysis
"""

import numpy as np
import pandas as pd
import requests
import json
import time
import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass, field
import concurrent.futures
from pathlib import Path
import h5py
import sqlite3
import yfinance as yf
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
import websocket
import threading
import queue
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings('ignore')

@dataclass
class MarketDataPoint:
    """Scientific market data point with validation"""
    timestamp: int
    symbol: str
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: int
    bid_price: float = 0.0
    ask_price: float = 0.0
    bid_volume: int = 0
    ask_volume: int = 0
    trades_count: int = 0
    vwap: float = 0.0
    
    def __post_init__(self):
        """Validate data integrity"""
        if self.high_price < max(self.open_price, self.close_price, self.low_price):
            raise ValueError(f"Invalid high price for {self.symbol} at {self.timestamp}")
        if self.low_price > min(self.open_price, self.close_price, self.high_price):
            raise ValueError(f"Invalid low price for {self.symbol} at {self.timestamp}")
        if self.volume < 0:
            raise ValueError(f"Invalid volume for {self.symbol} at {self.timestamp}")

@dataclass
class MarketStatistics:
    """Comprehensive market statistics"""
    symbol: str
    period_start: int
    period_end: int
    total_volume: int
    avg_price: float
    volatility: float
    skewness: float
    kurtosis: float
    returns_std: float
    max_drawdown: float
    sharpe_ratio: float
    var_95: float  # Value at Risk 95%
    var_99: float  # Value at Risk 99%
    liquidity_score: float
    microstructure_metrics: Dict[str, float] = field(default_factory=dict)

class RealDataProvider:
    """Real market data provider with multiple sources"""
    
    def __init__(self, alpha_vantage_key: Optional[str] = None):
        self.alpha_vantage_key = alpha_vantage_key
        self.logger = logging.getLogger(__name__)
        self.session_pool = self._create_session_pool()
        self.cache = {}
        self.rate_limits = {
            'alpha_vantage': {'calls': 0, 'reset_time': time.time() + 60},
            'yahoo': {'calls': 0, 'reset_time': time.time() + 60}
        }
        
    def _create_session_pool(self) -> concurrent.futures.ThreadPoolExecutor:
        """Create thread pool for concurrent data fetching"""
        return concurrent.futures.ThreadPoolExecutor(max_workers=8)
        
    async def fetch_real_time_data(self, symbols: List[str]) -> Dict[str, MarketDataPoint]:
        """Fetch real-time market data for symbols"""
        async with aiohttp.ClientSession() as session:
            tasks = []
            for symbol in symbols:
                task = self._fetch_symbol_realtime(session, symbol)
                tasks.append(task)
                
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
        data = {}
        for symbol, result in zip(symbols, results):
            if not isinstance(result, Exception) and result is not None:
                data[symbol] = result
                
        return data
        
    async def _fetch_symbol_realtime(self, session: aiohttp.ClientSession, 
                                   symbol: str) -> Optional[MarketDataPoint]:
        """Fetch real-time data for a single symbol"""
        try:
            # Try Yahoo Finance API first
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_yahoo_realtime(data, symbol)
                    
        except Exception as e:
            self.logger.warning(f"Failed to fetch real-time data for {symbol}: {e}")
            
        # Fallback to Alpha Vantage if available
        if self.alpha_vantage_key:
            return await self._fetch_alpha_vantage_realtime(symbol)
            
        return None
        
    def _parse_yahoo_realtime(self, data: Dict, symbol: str) -> Optional[MarketDataPoint]:
        """Parse Yahoo Finance real-time data"""
        try:
            result = data['chart']['result'][0]
            meta = result['meta']
            
            current_price = meta.get('regularMarketPrice', 0.0)
            previous_close = meta.get('previousClose', current_price)
            volume = meta.get('regularMarketVolume', 0)
            
            # Get latest OHLC data
            quotes = result.get('indicators', {}).get('quote', [{}])[0]
            timestamps = result.get('timestamp', [])
            
            if timestamps:
                latest_timestamp = timestamps[-1]
                open_prices = quotes.get('open', [current_price])
                high_prices = quotes.get('high', [current_price])
                low_prices = quotes.get('low', [current_price])
                close_prices = quotes.get('close', [current_price])
                volumes = quotes.get('volume', [volume])
                
                return MarketDataPoint(
                    timestamp=latest_timestamp * 1000,  # Convert to milliseconds
                    symbol=symbol,
                    open_price=open_prices[-1] or current_price,
                    high_price=high_prices[-1] or current_price,
                    low_price=low_prices[-1] or current_price,
                    close_price=current_price,
                    volume=volumes[-1] or volume,
                    vwap=current_price  # Simplified VWAP
                )
                
        except Exception as e:
            self.logger.error(f"Error parsing Yahoo data for {symbol}: {e}")
            
        return None
        
    def fetch_historical_data(self, symbol: str, period: str = "1y", 
                            interval: str = "1d") -> pd.DataFrame:
        """Fetch historical market data with scientific validation"""
        try:
            # Use yfinance for reliable historical data
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                self.logger.warning(f"No historical data found for {symbol}")
                return pd.DataFrame()
                
            # Validate data integrity
            data = self._validate_historical_data(data, symbol)
            
            # Add derived metrics
            data = self._enhance_historical_data(data)
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching historical data for {symbol}: {e}")
            return pd.DataFrame()
            
    def _validate_historical_data(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Validate historical data for scientific accuracy"""
        # Remove rows with invalid OHLC relationships
        invalid_high = data['High'] < data[['Open', 'Close', 'Low']].max(axis=1)
        invalid_low = data['Low'] > data[['Open', 'Close', 'High']].min(axis=1)
        invalid_volume = data['Volume'] < 0
        
        invalid_rows = invalid_high | invalid_low | invalid_volume
        if invalid_rows.any():
            self.logger.warning(f"Removed {invalid_rows.sum()} invalid rows from {symbol} data")
            data = data[~invalid_rows]
            
        # Remove outliers using IQR method
        for col in ['Open', 'High', 'Low', 'Close']:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            outliers = (data[col] < lower_bound) | (data[col] > upper_bound)
            if outliers.any():
                self.logger.info(f"Found {outliers.sum()} price outliers in {symbol} {col} data")
                # Don't remove, just log - extreme moves can be legitimate
                
        return data
        
    def _enhance_historical_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Enhance historical data with derived metrics"""
        # Calculate returns
        data['Returns'] = data['Close'].pct_change()
        data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
        
        # Calculate VWAP (simplified daily VWAP)
        data['VWAP'] = (data['Close'] * data['Volume']).rolling(window=20).sum() / \
                       data['Volume'].rolling(window=20).sum()
        
        # Calculate typical price and true range
        data['Typical_Price'] = (data['High'] + data['Low'] + data['Close']) / 3
        data['True_Range'] = np.maximum(
            data['High'] - data['Low'],
            np.maximum(
                abs(data['High'] - data['Close'].shift(1)),
                abs(data['Low'] - data['Close'].shift(1))
            )
        )
        
        # Calculate volatility (rolling 20-day)
        data['Volatility'] = data['Returns'].rolling(window=20).std() * np.sqrt(252)
        
        # Calculate RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # Calculate Bollinger Bands
        sma_20 = data['Close'].rolling(window=20).mean()
        std_20 = data['Close'].rolling(window=20).std()
        data['BB_Upper'] = sma_20 + (2 * std_20)
        data['BB_Lower'] = sma_20 - (2 * std_20)
        data['BB_Width'] = (data['BB_Upper'] - data['BB_Lower']) / sma_20
        
        return data

class ScientificDataProcessor:
    """Scientific data processing with statistical validation"""
    
    def __init__(self):
        self.data_provider = RealDataProvider()
        self.processed_cache = {}
        self.statistics_cache = {}
        
    def calculate_market_statistics(self, data: pd.DataFrame, symbol: str) -> MarketStatistics:
        """Calculate comprehensive market statistics"""
        if data.empty:
            raise ValueError(f"No data provided for {symbol}")
            
        returns = data['Returns'].dropna()
        prices = data['Close']
        volumes = data['Volume']
        
        # Basic statistics
        total_volume = int(volumes.sum())
        avg_price = float(prices.mean())
        volatility = float(returns.std() * np.sqrt(252))
        
        # Distribution statistics
        skewness = float(stats.skew(returns))
        kurtosis = float(stats.kurtosis(returns))
        returns_std = float(returns.std())
        
        # Risk metrics
        var_95 = float(np.percentile(returns, 5))  # 5th percentile for 95% VaR
        var_99 = float(np.percentile(returns, 1))  # 1st percentile for 99% VaR
        
        # Maximum drawdown
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = float(drawdown.min())
        
        # Sharpe ratio (assuming 0% risk-free rate)
        sharpe_ratio = float(returns.mean() / returns_std * np.sqrt(252)) if returns_std > 0 else 0.0
        
        # Liquidity score (based on volume consistency and bid-ask spread proxy)
        volume_cv = volumes.std() / volumes.mean() if volumes.mean() > 0 else float('inf')
        liquidity_score = 1.0 / (1.0 + volume_cv)  # Higher score = more liquid
        
        # Microstructure metrics
        microstructure = self._calculate_microstructure_metrics(data)
        
        return MarketStatistics(
            symbol=symbol,
            period_start=int(data.index[0].timestamp()),
            period_end=int(data.index[-1].timestamp()),
            total_volume=total_volume,
            avg_price=avg_price,
            volatility=volatility,
            skewness=skewness,
            kurtosis=kurtosis,
            returns_std=returns_std,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            var_95=var_95,
            var_99=var_99,
            liquidity_score=liquidity_score,
            microstructure_metrics=microstructure
        )
        
    def _calculate_microstructure_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate market microstructure metrics"""
        metrics = {}
        
        # Price impact proxy (return-to-volume correlation)
        if 'Returns' in data.columns and 'Volume' in data.columns:
            returns = data['Returns'].dropna()
            volumes = data['Volume'][returns.index]
            if len(returns) > 20:
                correlation, p_value = stats.pearsonr(abs(returns), volumes)
                metrics['price_impact_correlation'] = float(correlation) if not np.isnan(correlation) else 0.0
                metrics['price_impact_p_value'] = float(p_value) if not np.isnan(p_value) else 1.0
                
        # Effective spread proxy (high-low to close ratio)
        if all(col in data.columns for col in ['High', 'Low', 'Close']):
            spread_proxy = (data['High'] - data['Low']) / data['Close']
            metrics['effective_spread_proxy'] = float(spread_proxy.mean())
            metrics['spread_volatility'] = float(spread_proxy.std())
            
        # Order flow imbalance proxy (using volume patterns)
        if 'Volume' in data.columns:
            volume_changes = data['Volume'].pct_change().dropna()
            price_changes = data['Close'].pct_change().dropna()
            
            # Align indices
            common_idx = volume_changes.index.intersection(price_changes.index)
            if len(common_idx) > 20:
                vol_chg = volume_changes[common_idx]
                price_chg = price_changes[common_idx]
                
                # Order flow imbalance = correlation between volume change and price change
                imbalance_corr, _ = stats.pearsonr(vol_chg, price_chg)
                metrics['order_flow_imbalance'] = float(imbalance_corr) if not np.isnan(imbalance_corr) else 0.0
                
        # Market efficiency metric (autocorrelation of returns)
        if 'Returns' in data.columns:
            returns = data['Returns'].dropna()
            if len(returns) > 50:
                # First-order autocorrelation
                autocorr = returns.autocorr(lag=1)
                metrics['return_autocorrelation'] = float(autocorr) if not np.isnan(autocorr) else 0.0
                
                # Variance ratio test for random walk
                var_1 = returns.var()
                returns_2 = returns.rolling(window=2).sum()[1::2]  # Non-overlapping 2-period returns
                var_2 = returns_2.var() / 2
                variance_ratio = var_2 / var_1 if var_1 > 0 else 1.0
                metrics['variance_ratio'] = float(variance_ratio)
                
        return metrics
        
    def generate_realistic_synthetic_data(self, symbol: str, days: int = 252, 
                                        base_price: float = 100.0) -> pd.DataFrame:
        """Generate realistic synthetic market data based on statistical properties"""
        # Try to get real data for calibration
        real_data = self.data_provider.fetch_historical_data(symbol, period="2y")
        
        if not real_data.empty:
            # Calibrate from real data
            returns = real_data['Returns'].dropna()
            volatility = returns.std()
            drift = returns.mean()
            skew = stats.skew(returns)
            kurt = stats.kurtosis(returns)
        else:
            # Use reasonable defaults for financial data
            volatility = 0.02  # 2% daily volatility
            drift = 0.0005     # 0.05% daily drift
            skew = -0.5        # Slight negative skew
            kurt = 3.0         # Excess kurtosis
            
        # Generate dates
        dates = pd.date_range(start=datetime.now() - timedelta(days=days), 
                             periods=days, freq='D')
        
        # Generate returns using skewed Student's t-distribution for realism
        np.random.seed(int(time.time()) % 2**32)  # Reproducible but time-varying
        
        # Use Student's t-distribution with finite variance
        df = 5  # degrees of freedom for heavy tails
        raw_returns = stats.t.rvs(df=df, size=days) * volatility + drift
        
        # Apply skewness transformation
        if abs(skew) > 0.1:
            raw_returns = self._apply_skewness(raw_returns, skew)
            
        # Generate price series
        prices = [base_price]
        for ret in raw_returns[1:]:
            prices.append(prices[-1] * (1 + ret))
            
        # Generate OHLC data with realistic relationships
        data = []
        for i, (date, price, ret) in enumerate(zip(dates, prices, raw_returns)):
            # Intraday volatility (fraction of daily volatility)
            intraday_vol = volatility * np.random.uniform(0.3, 0.8)
            
            # Generate realistic OHLC
            if i == 0:
                open_price = price
            else:
                open_price = prices[i-1] * (1 + np.random.normal(0, volatility * 0.1))
                
            close_price = price
            
            # High and low based on intraday volatility
            high_low_range = abs(ret) * np.random.uniform(1.2, 2.5) + intraday_vol
            high_price = max(open_price, close_price) * (1 + high_low_range * np.random.uniform(0.3, 1.0))
            low_price = min(open_price, close_price) * (1 - high_low_range * np.random.uniform(0.3, 1.0))
            
            # Ensure OHLC relationships are valid
            high_price = max(high_price, open_price, close_price)
            low_price = min(low_price, open_price, close_price)
            
            # Volume with realistic patterns
            base_volume = 1000000
            volume_multiplier = 1 + abs(ret) * 5  # Higher volume on big moves
            volume_noise = np.random.lognormal(0, 0.5)  # Log-normal volume distribution
            volume = int(base_volume * volume_multiplier * volume_noise)
            
            data.append({
                'Open': open_price,
                'High': high_price,
                'Low': low_price,
                'Close': close_price,
                'Volume': volume
            })
            
        df = pd.DataFrame(data, index=dates)
        
        # Add enhanced metrics
        df = self.data_provider._enhance_historical_data(df)
        
        return df
        
    def _apply_skewness(self, returns: np.ndarray, target_skew: float) -> np.ndarray:
        """Apply skewness transformation to return series"""
        if abs(target_skew) < 0.1:
            return returns
            
        # Use Johnson SU transformation to achieve target skewness
        current_skew = stats.skew(returns)
        
        if abs(current_skew - target_skew) > 0.1:
            # Simple transformation: cube root for negative skew, cube for positive
            if target_skew < 0:
                factor = min(abs(target_skew), 1.0)
                transformed = np.sign(returns) * np.power(np.abs(returns), 1 - factor * 0.3)
            else:
                factor = min(target_skew, 1.0)
                transformed = np.sign(returns) * np.power(np.abs(returns), 1 + factor * 0.3)
                
            # Scale to maintain original variance
            original_std = returns.std()
            transformed = transformed * (original_std / transformed.std())
            return transformed
            
        return returns
        
    def save_processed_data(self, data: pd.DataFrame, symbol: str, filepath: str):
        """Save processed data to HDF5 for efficient access"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with h5py.File(filepath, 'w') as f:
            # Save main data
            grp = f.create_group(symbol)
            grp.create_dataset('timestamps', data=data.index.astype(np.int64))
            
            for column in data.columns:
                grp.create_dataset(column, data=data[column].values)
                
            # Save metadata
            grp.attrs['start_date'] = str(data.index[0])
            grp.attrs['end_date'] = str(data.index[-1])
            grp.attrs['rows'] = len(data)
            grp.attrs['columns'] = len(data.columns)
            
    def load_processed_data(self, symbol: str, filepath: str) -> pd.DataFrame:
        """Load processed data from HDF5"""
        try:
            with h5py.File(filepath, 'r') as f:
                grp = f[symbol]
                
                # Load timestamps
                timestamps = pd.to_datetime(grp['timestamps'][:])
                
                # Load data columns
                data = {}
                for key in grp.keys():
                    if key != 'timestamps':
                        data[key] = grp[key][:]
                        
                return pd.DataFrame(data, index=timestamps)
                
        except (KeyError, OSError) as e:
            raise FileNotFoundError(f"Could not load data for {symbol} from {filepath}: {e}")

def main():
    """Demonstrate scientific data processing capabilities"""
    print("🔬 CWTS Scientific Data Processor")
    print("=" * 50)
    
    processor = ScientificDataProcessor()
    
    # Test symbols
    symbols = ['AAPL', 'GOOGL', 'TSLA', 'SPY']
    
    print("\n📊 Fetching and processing real market data...")
    
    for symbol in symbols:
        try:
            print(f"\nProcessing {symbol}...")
            
            # Get historical data
            historical_data = processor.data_provider.fetch_historical_data(symbol, period="6mo")
            
            if not historical_data.empty:
                # Calculate statistics
                stats = processor.calculate_market_statistics(historical_data, symbol)
                
                print(f"  Average Price: ${stats.avg_price:.2f}")
                print(f"  Volatility: {stats.volatility*100:.1f}%")
                print(f"  Sharpe Ratio: {stats.sharpe_ratio:.2f}")
                print(f"  Max Drawdown: {stats.max_drawdown*100:.1f}%")
                print(f"  Liquidity Score: {stats.liquidity_score:.3f}")
                
                # Save processed data
                filepath = f"/home/kutlu/CWTS/cwts-ultra/performance/benchmarks/processed_data_{symbol}.h5"
                processor.save_processed_data(historical_data, symbol, filepath)
                
            else:
                print(f"  No data available for {symbol}")
                
                # Generate synthetic data as fallback
                print(f"  Generating synthetic data for {symbol}...")
                synthetic_data = processor.generate_realistic_synthetic_data(symbol)
                stats = processor.calculate_market_statistics(synthetic_data, symbol)
                
                print(f"  Synthetic - Volatility: {stats.volatility*100:.1f}%")
                print(f"  Synthetic - Sharpe Ratio: {stats.sharpe_ratio:.2f}")
                
        except Exception as e:
            print(f"  Error processing {symbol}: {e}")
            
    print("\n✅ Scientific data processing demonstration complete")

if __name__ == "__main__":
    main()