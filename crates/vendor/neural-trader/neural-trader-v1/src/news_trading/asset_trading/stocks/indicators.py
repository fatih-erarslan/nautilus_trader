"""Technical indicators for stock trading."""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional


class TechnicalIndicators:
    """Calculate technical indicators for swing trading."""
    
    def calculate_sma(self, prices: List[float], period: int) -> List[float]:
        """Calculate Simple Moving Average.
        
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
            sma = sum(window) / period
            sma_values.append(sma)
            
        return sma_values
    
    def calculate_ema(self, prices: List[float], period: int) -> List[float]:
        """Calculate Exponential Moving Average.
        
        Args:
            prices: List of prices
            period: Period for EMA calculation
            
        Returns:
            List of EMA values
        """
        if len(prices) < period:
            return []
        
        multiplier = 2 / (period + 1)
        ema_values = []
        
        # Start with SMA for first value
        sma = sum(prices[:period]) / period
        ema_values.append(sma)
        
        # Calculate EMA for remaining values
        for i in range(period, len(prices)):
            ema = (prices[i] - ema_values[-1]) * multiplier + ema_values[-1]
            ema_values.append(ema)
            
        return ema_values
    
    def calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Calculate Relative Strength Index.
        
        Args:
            prices: List of prices
            period: Period for RSI calculation
            
        Returns:
            RSI value (0-100)
        """
        if len(prices) < period + 1:
            return 50.0  # Neutral
        
        # Calculate price changes
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        
        # Separate gains and losses
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]
        
        # Calculate average gain/loss
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_macd(
        self, 
        prices: List[float],
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ) -> Tuple[List[float], List[float], List[float]]:
        """Calculate MACD (Moving Average Convergence Divergence).
        
        Args:
            prices: List of prices
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line EMA period
            
        Returns:
            Tuple of (MACD line, Signal line, Histogram)
        """
        if len(prices) < slow_period + signal_period:
            return [], [], []
        
        # Calculate EMAs
        ema_fast = self.calculate_ema(prices, fast_period)
        ema_slow = self.calculate_ema(prices, slow_period)
        
        # Calculate MACD line
        macd_line = []
        for i in range(len(ema_slow)):
            # Align indices
            fast_idx = len(ema_fast) - len(ema_slow) + i
            if fast_idx >= 0:
                macd_line.append(ema_fast[fast_idx] - ema_slow[i])
        
        # Calculate signal line
        signal_line = self.calculate_ema(macd_line, signal_period)
        
        # Calculate histogram
        histogram = []
        for i in range(len(signal_line)):
            macd_idx = len(macd_line) - len(signal_line) + i
            if macd_idx >= 0:
                histogram.append(macd_line[macd_idx] - signal_line[i])
        
        return macd_line, signal_line, histogram
    
    def calculate_bollinger_bands(
        self,
        prices: List[float],
        period: int = 20,
        std_dev: float = 2
    ) -> Tuple[List[float], List[float], List[float]]:
        """Calculate Bollinger Bands.
        
        Args:
            prices: List of prices
            period: Period for moving average
            std_dev: Number of standard deviations
            
        Returns:
            Tuple of (Upper band, Middle band (SMA), Lower band)
        """
        if len(prices) < period:
            return [], [], []
        
        sma = self.calculate_sma(prices, period)
        upper_band = []
        lower_band = []
        
        for i in range(len(sma)):
            # Get price window
            start_idx = len(prices) - len(sma) + i - period + 1
            end_idx = start_idx + period
            window = prices[start_idx:end_idx]
            
            # Calculate standard deviation
            std = np.std(window)
            
            # Calculate bands
            upper_band.append(sma[i] + (std_dev * std))
            lower_band.append(sma[i] - (std_dev * std))
        
        return upper_band, sma, lower_band
    
    def calculate_atr(
        self,
        high_prices: List[float],
        low_prices: List[float],
        close_prices: List[float],
        period: int = 14
    ) -> float:
        """Calculate Average True Range.
        
        Args:
            high_prices: List of high prices
            low_prices: List of low prices
            close_prices: List of close prices
            period: Period for ATR calculation
            
        Returns:
            ATR value
        """
        if len(high_prices) < period + 1:
            return 0.0
        
        true_ranges = []
        
        for i in range(1, len(high_prices)):
            high_low = high_prices[i] - low_prices[i]
            high_close = abs(high_prices[i] - close_prices[i-1])
            low_close = abs(low_prices[i] - close_prices[i-1])
            
            true_range = max(high_low, high_close, low_close)
            true_ranges.append(true_range)
        
        # Calculate ATR
        atr = sum(true_ranges[-period:]) / period
        
        return atr
    
    def calculate_stochastic(
        self,
        high_prices: List[float],
        low_prices: List[float],
        close_prices: List[float],
        period: int = 14,
        smooth_k: int = 3,
        smooth_d: int = 3
    ) -> Tuple[float, float]:
        """Calculate Stochastic Oscillator.
        
        Args:
            high_prices: List of high prices
            low_prices: List of low prices
            close_prices: List of close prices
            period: Look-back period
            smooth_k: Smoothing for %K
            smooth_d: Smoothing for %D
            
        Returns:
            Tuple of (%K, %D)
        """
        if len(close_prices) < period:
            return 50.0, 50.0
        
        # Calculate raw %K values
        k_values = []
        
        for i in range(period - 1, len(close_prices)):
            window_high = max(high_prices[i - period + 1:i + 1])
            window_low = min(low_prices[i - period + 1:i + 1])
            
            if window_high == window_low:
                k = 50.0
            else:
                k = ((close_prices[i] - window_low) / (window_high - window_low)) * 100
            
            k_values.append(k)
        
        # Smooth %K
        if len(k_values) >= smooth_k:
            smooth_k_value = sum(k_values[-smooth_k:]) / smooth_k
        else:
            smooth_k_value = k_values[-1] if k_values else 50.0
        
        # Calculate %D (moving average of %K)
        if len(k_values) >= smooth_d:
            d_value = sum(k_values[-smooth_d:]) / smooth_d
        else:
            d_value = smooth_k_value
        
        return smooth_k_value, d_value