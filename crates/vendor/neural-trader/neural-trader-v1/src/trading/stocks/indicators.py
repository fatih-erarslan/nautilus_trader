"""Technical indicators for stock analysis"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """Calculate technical indicators for stock analysis"""
    
    def calculate_sma(self, prices: List[float], period: int) -> List[float]:
        """
        Calculate Simple Moving Average
        
        Args:
            prices: List of prices
            period: Period for SMA calculation
            
        Returns:
            List of SMA values
        """
        if len(prices) < period:
            return []
            
        sma_values = []
        for i in range(period - 1, len(prices)):
            window = prices[i - period + 1:i + 1]
            sma_values.append(sum(window) / period)
            
        return sma_values
    
    def calculate_ema(self, prices: List[float], period: int) -> List[float]:
        """
        Calculate Exponential Moving Average
        
        Args:
            prices: List of prices
            period: Period for EMA calculation
            
        Returns:
            List of EMA values
        """
        if len(prices) < period:
            return []
            
        # Calculate multiplier (alpha)
        alpha = 2.0 / (period + 1)
        
        # Initialize EMA with first price (more responsive)
        ema_values = [prices[0]]
        
        # Calculate EMA for each subsequent price
        for i in range(1, len(prices)):
            ema = alpha * prices[i] + (1 - alpha) * ema_values[-1]
            ema_values.append(ema)
            
        return ema_values
    
    def calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """
        Calculate Relative Strength Index
        
        Args:
            prices: List of prices
            period: Period for RSI calculation (default 14)
            
        Returns:
            RSI value (0-100)
        """
        if len(prices) < period + 1:
            return 50.0  # Neutral RSI if not enough data
            
        # Calculate price changes
        deltas = np.diff(prices)
        gains = deltas.copy()
        losses = deltas.copy()
        
        gains[gains < 0] = 0
        losses[losses > 0] = 0
        losses = abs(losses)
        
        # Calculate average gains and losses
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        
        # Handle division by zero
        if avg_loss == 0:
            return 100.0
            
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_macd(self, prices: List[float], fast_period: int = 12, 
                      slow_period: int = 26, signal_period: int = 9) -> Tuple[List[float], List[float], List[float]]:
        """
        Calculate MACD (Moving Average Convergence Divergence)
        
        Args:
            prices: List of prices
            fast_period: Fast EMA period (default 12)
            slow_period: Slow EMA period (default 26)
            signal_period: Signal line EMA period (default 9)
            
        Returns:
            Tuple of (MACD line, Signal line, Histogram)
        """
        if len(prices) < slow_period:
            return [], [], []
            
        # Calculate EMAs
        ema_fast = self.calculate_ema(prices, fast_period)
        ema_slow = self.calculate_ema(prices, slow_period)
        
        # Calculate MACD line starting from slow_period-1 (when slow EMA becomes valid)
        macd_line = []
        for i in range(slow_period - 1, len(prices)):
            macd_line.append(ema_fast[i] - ema_slow[i])
                
        # Calculate Signal line (EMA of MACD)
        signal_line = []
        if len(macd_line) >= signal_period:
            signal_multiplier = 2 / (signal_period + 1)
            signal_sma = sum(macd_line[:signal_period]) / signal_period
            signal_line.append(signal_sma)
            
            for i in range(signal_period, len(macd_line)):
                signal = (macd_line[i] - signal_line[-1]) * signal_multiplier + signal_line[-1]
                signal_line.append(signal)
                
        # Calculate Histogram
        histogram = []
        offset = len(macd_line) - len(signal_line)
        for i in range(len(signal_line)):
            histogram.append(macd_line[offset + i] - signal_line[i])
            
        return macd_line, signal_line, histogram
    
    def calculate_bollinger_bands(self, prices: List[float], period: int = 20, 
                                 std_dev: float = 2) -> Tuple[List[float], List[float], List[float]]:
        """
        Calculate Bollinger Bands
        
        Args:
            prices: List of prices
            period: Period for moving average (default 20)
            std_dev: Number of standard deviations (default 2)
            
        Returns:
            Tuple of (Upper band, Middle band, Lower band)
        """
        if len(prices) < period:
            return [], [], []
            
        # Calculate SMA (middle band)
        middle_band = self.calculate_sma(prices, period)
        
        # Calculate standard deviation and bands
        upper_band = []
        lower_band = []
        
        for i in range(period - 1, len(prices)):
            window = prices[i - period + 1:i + 1]
            std = np.std(window)
            upper_band.append(middle_band[i - period + 1] + (std_dev * std))
            lower_band.append(middle_band[i - period + 1] - (std_dev * std))
            
        return upper_band, middle_band, lower_band
    
    def calculate_vwap(self, data: pd.DataFrame) -> np.ndarray:
        """
        Calculate Volume Weighted Average Price
        
        Args:
            data: DataFrame with 'Close' and 'Volume' columns
            
        Returns:
            Array of VWAP values
        """
        vwap_values = []
        cumulative_volume = 0
        cumulative_pv = 0
        
        for i in range(len(data)):
            price = data['Close'].iloc[i]
            volume = data['Volume'].iloc[i]
            
            cumulative_pv += price * volume
            cumulative_volume += volume
            
            if cumulative_volume > 0:
                vwap = cumulative_pv / cumulative_volume
            else:
                vwap = price
                
            vwap_values.append(vwap)
            
        return np.array(vwap_values)
    
    def calculate_obv(self, data: pd.DataFrame) -> np.ndarray:
        """
        Calculate On-Balance Volume
        
        Args:
            data: DataFrame with 'Close' and 'Volume' columns
            
        Returns:
            Array of OBV values
        """
        obv_values = [0]  # Start with 0
        
        for i in range(1, len(data)):
            if data['Close'].iloc[i] > data['Close'].iloc[i-1]:
                # Price up, add volume
                obv = obv_values[-1] + data['Volume'].iloc[i]
            elif data['Close'].iloc[i] < data['Close'].iloc[i-1]:
                # Price down, subtract volume
                obv = obv_values[-1] - data['Volume'].iloc[i]
            else:
                # Price unchanged, OBV stays same
                obv = obv_values[-1]
                
            obv_values.append(obv)
            
        return np.array(obv_values)
    
    def calculate_atr(self, data: pd.DataFrame, period: int = 14) -> np.ndarray:
        """
        Calculate Average True Range
        
        Args:
            data: DataFrame with 'High', 'Low', 'Close' columns
            period: Period for ATR calculation
            
        Returns:
            Array of ATR values
        """
        true_ranges = []
        
        # First TR is just high - low
        true_ranges.append(data['High'].iloc[0] - data['Low'].iloc[0])
        
        # Calculate true ranges for remaining periods
        for i in range(1, len(data)):
            high = data['High'].iloc[i]
            low = data['Low'].iloc[i]
            prev_close = data['Close'].iloc[i-1]
            
            # True Range is max of:
            # 1. Current High - Current Low
            # 2. |Current High - Previous Close|
            # 3. |Current Low - Previous Close|
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            true_ranges.append(tr)
            
        # Calculate ATR as EMA of True Range
        atr_values = []
        for i in range(period - 1, len(true_ranges)):
            if i == period - 1:
                # First ATR is simple average
                atr = sum(true_ranges[:period]) / period
            else:
                # Subsequent ATRs use EMA formula
                atr = (atr_values[-1] * (period - 1) + true_ranges[i]) / period
            atr_values.append(atr)
                
        return np.array(atr_values)
    
    def calculate_stochastic(self, data: pd.DataFrame, period: int = 14, 
                           smooth_k: int = 3, smooth_d: int = 3) -> Tuple[List[float], List[float]]:
        """
        Calculate Stochastic Oscillator
        
        Args:
            data: DataFrame with 'High', 'Low', 'Close' columns
            period: Period for calculation
            smooth_k: Smoothing period for %K
            smooth_d: Smoothing period for %D
            
        Returns:
            Tuple of (%K values, %D values)
        """
        raw_k_values = []
        
        for i in range(period - 1, len(data)):
            # Get period high and low
            period_high = data['High'].iloc[i - period + 1:i + 1].max()
            period_low = data['Low'].iloc[i - period + 1:i + 1].min()
            current_close = data['Close'].iloc[i]
            
            # Calculate %K
            if period_high != period_low:
                k = ((current_close - period_low) / (period_high - period_low)) * 100
            else:
                k = 50  # Default to middle if no range
                
            raw_k_values.append(k)
            
        # Smooth %K if requested and enough data
        if smooth_k > 1 and len(raw_k_values) >= smooth_k:
            k_values = self.calculate_sma(raw_k_values, smooth_k)
        else:
            # If not enough data for smoothing, return raw values
            k_values = raw_k_values
            
        # Calculate %D as SMA of %K
        d_values = []
        if smooth_d > 1 and len(k_values) >= smooth_d:
            d_values = self.calculate_sma(k_values, smooth_d)
        elif len(k_values) >= 2:
            # If not enough for full smoothing but have some data, use simpler smoothing
            # Use a basic average of available values to make D% smoother than K%
            d_values = []
            for i in range(len(k_values)):
                if i == 0:
                    d_values.append(k_values[i])
                else:
                    # Simple 2-period average to smooth
                    d_values.append((k_values[i] + k_values[i-1]) / 2)
        else:
            # Very limited data - return k_values but make D slightly smoother
            d_values = k_values.copy() if k_values else []
            
        return k_values, d_values
    
    def calculate_fibonacci_levels(self, high: float, low: float) -> Dict[str, float]:
        """
        Calculate Fibonacci retracement levels
        
        Args:
            high: Recent high price
            low: Recent low price
            
        Returns:
            Dictionary with Fibonacci levels
        """
        diff = high - low
        
        levels = {
            '0.0%': low,
            '23.6%': low + (diff * 0.236),
            '38.2%': low + (diff * 0.382),
            '50.0%': low + (diff * 0.500),
            '61.8%': low + (diff * 0.618),
            '78.6%': low + (diff * 0.786),
            '100.0%': high,
            '127.2%': high + (diff * 0.272),  # Extension
            '161.8%': high + (diff * 0.618)   # Extension
        }
        
        return levels
    
    def calculate_pivot_points(self, high: float, low: float, close: float) -> Dict[str, float]:
        """
        Calculate pivot points for support and resistance
        
        Args:
            high: Previous period high
            low: Previous period low
            close: Previous period close
            
        Returns:
            Dictionary with pivot levels
        """
        # Standard pivot point
        pivot = (high + low + close) / 3
        
        # Support and resistance levels
        levels = {
            'pivot': pivot,
            's1': (2 * pivot) - high,
            's2': pivot - (high - low),
            's3': low - 2 * (high - pivot),
            'r1': (2 * pivot) - low,
            'r2': pivot + (high - low),
            'r3': high + 2 * (pivot - low)
        }
        
        return levels