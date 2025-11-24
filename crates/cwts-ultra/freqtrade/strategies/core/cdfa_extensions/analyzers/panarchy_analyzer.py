#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 27 10:40:29 2025

@author: ashina
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from enum import Enum
from dataclasses import dataclass, field
from functools import lru_cache
import time
from numba import njit, float64, int64, boolean

# Optional - will use if available
try:
    import talib.abstract as ta
    TALIB_AVAILABLE = True
except ImportError:
    ta = None
    TALIB_AVAILABLE = False

# PennyLane Catalyst JIT support - keeping for backward compatibility
try:
    from catalyst import qjit
    CATALYST_AVAILABLE = True
except ImportError:
    # Fallback decorator when Catalyst isn't available
    def qjit(func):
        return func
    CATALYST_AVAILABLE = False


class MarketPhase(Enum):
    """Enum representing the four phases of Panarchy"""
    GROWTH = "growth"
    CONSERVATION = "conservation"
    RELEASE = "release"
    REORGANIZATION = "reorganization"
    UNKNOWN = "unknown"
    
    @classmethod
    def from_string(cls, phase_str: str) -> 'MarketPhase':
        """Convert string to enum value, with validation"""
        phase_str = phase_str.lower()
        for phase in cls:
            if phase.value == phase_str:
                return phase
        return cls.UNKNOWN


@dataclass
class PanarchyParameters:
    """Parameters for Panarchy calculations"""
    autocorr_lag: int = 1
    adx_period: int = 14
    n_regimes: int = 4
    p_momentum_weight: float = 0.5
    regime_smoothing_window: int = 3
    hysteresis_min_score_threshold: float = 0.35
    hysteresis_min_score_diff: float = 0.10
    
    # Weight configurations for phase determination
    weights_growth: Dict[str, float] = field(default_factory=lambda: {
        "r_high_c_low": 0.4,  # High resilience, low connectedness
        "momentum_pos": 0.4,  # Positive momentum
        "potential_rising": 0.2  # Rising potential
    })
    
    weights_conservation: Dict[str, float] = field(default_factory=lambda: {
        "p_high_c_high_r_low": 0.5,  # High potential, high connectedness, low resilience
        "momentum_stable": 0.3,  # Stable momentum
        "potential_stable": 0.2  # Stable potential
    })
    
    weights_release: Dict[str, float] = field(default_factory=lambda: {
        "r_low_c_high": 0.3,  # Low resilience, high connectedness
        "momentum_neg": 0.4,  # Negative momentum
        "potential_falling": 0.3  # Falling potential
    })
    
    weights_reorganization: Dict[str, float] = field(default_factory=lambda: {
        "p_low_c_low_r_high": 0.4,  # Low potential, low connectedness, high resilience
        "momentum_improving": 0.4,  # Improving momentum
        "potential_low": 0.2  # Low potential
    })


# Numba helper functions
@njit(float64(float64[:], int64), cache=True) 
def _autocorrelation_impl(series, lag=1):
    """Numba implementation of autocorrelation calculation"""
    n = len(series)
    if n <= lag or n < 4:
        return 0.0
    
    # Calculate mean
    series_mean = 0.0
    for i in range(n):
        series_mean += series[i]
    series_mean /= n
    
    # Calculate normalized series
    series_norm = np.zeros(n)
    for i in range(n):
        series_norm[i] = series[i] - series_mean
    
    # Calculate autocorrelation
    numerator = 0.0
    for i in range(lag, n):
        numerator += series_norm[i] * series_norm[i-lag]
    
    # Calculate denominator
    denominator = 0.0
    for i in range(n):
        denominator += series_norm[i] * series_norm[i]
    
    if denominator < 1e-10:
        return 0.0
        
    return numerator / denominator

@njit(cache=True)
def _pcr_calculation_impl(prices, returns, highs, lows, volatility, period, autocorr_lag=1):
    """Numba implementation of PCR calculation"""
    n = len(prices)
    P = np.full(n, 0.5)
    C = np.full(n, 0.5)
    R = np.full(n, 0.5)
    
    if n < period + 1:
        return P, C, R
    
    # Calculate P (Potential) - normalized within min/max
    roll_max = np.zeros(n)
    roll_min = np.zeros(n)
    
    for i in range(period, n):
        # Calculate rolling max/min
        roll_max[i] = prices[i-period]
        roll_min[i] = prices[i-period]
        
        for j in range(i-period+1, i+1):
            if prices[j] > roll_max[i]:
                roll_max[i] = prices[j]
            if prices[j] < roll_min[i]:
                roll_min[i] = prices[j]
        
        # Calculate normalized potential
        if roll_max[i] - roll_min[i] > 1e-9:
            P[i] = (prices[i] - roll_min[i]) / (roll_max[i] - roll_min[i])
    
    # Calculate C (Connectedness) using windowed autocorrelation
    for i in range(period, n):
        # Extract window
        window_returns = returns[i-period:i]
        
        # Calculate standard deviation of window
        std_val = 0.0
        mean_val = 0.0
        
        for j in range(len(window_returns)):
            mean_val += window_returns[j]
        
        if len(window_returns) > 0:
            mean_val /= len(window_returns)
        
        for j in range(len(window_returns)):
            std_val += (window_returns[j] - mean_val) ** 2
        
        if len(window_returns) > 1:
            std_val = np.sqrt(std_val / (len(window_returns) - 1))
        
        if std_val > 1e-9 and len(window_returns) >= autocorr_lag + 1:
            # Calculate autocorrelation for this window
            shifted = window_returns[:-autocorr_lag]
            shifted_lag = window_returns[autocorr_lag:]
            
            if len(shifted) == len(shifted_lag) and len(shifted) > 0:
                # Calculate means
                shifted_mean = 0.0
                shifted_lag_mean = 0.0
                
                for j in range(len(shifted)):
                    shifted_mean += shifted[j]
                    shifted_lag_mean += shifted_lag[j]
                
                shifted_mean /= len(shifted)
                shifted_lag_mean /= len(shifted_lag)
                
                # Calculate standard deviations
                shifted_std = 0.0
                shifted_lag_std = 0.0
                
                for j in range(len(shifted)):
                    shifted_std += (shifted[j] - shifted_mean) ** 2
                    shifted_lag_std += (shifted_lag[j] - shifted_lag_mean) ** 2
                
                shifted_std = np.sqrt(shifted_std / len(shifted))
                shifted_lag_std = np.sqrt(shifted_lag_std / len(shifted_lag))
                
                # Calculate covariance
                numerator = 0.0
                for j in range(len(shifted)):
                    numerator += (shifted[j] - shifted_mean) * (shifted_lag[j] - shifted_lag_mean)
                
                numerator /= len(shifted)
                
                # Calculate correlation
                denominator = shifted_std * shifted_lag_std
                if denominator > 1e-9:
                    autocorr = numerator / denominator
                    C[i] = (autocorr + 1) / 2  # Normalize to [0, 1]
    
    # Calculate R (Resilience) - inverse of volatility
    for i in range(n):
        R[i] = 1.0 - volatility[i]
    
    # Ensure values are in [0, 1] range
    for i in range(n):
        P[i] = max(0, min(1, P[i]))
        C[i] = max(0, min(1, C[i]))
        R[i] = max(0, min(1, R[i]))
    
    return P, C, R


class PanarchyAnalyzer:
    """
    Analyzer for Panarchy cycles in market data.
    
    This class implements methods to detect adaptive cycles (growth, conservation,
    release, reorganization) and measure Potential, Connectedness, and Resilience
    in financial time series.
    """
    
    def __init__(self, cache_size: int = 100, use_jit: bool = True,
                 params: Optional[PanarchyParameters] = None,
                 log_level: str = "INFO"):
        """
        Initialize the Panarchy analyzer.
        
        Args:
            cache_size: Size of the LRU cache for expensive calculations
            use_jit: Whether to use JIT compilation via numba
            params: Optional custom parameters for Panarchy calculations
            log_level: Logging level (default: INFO)
        """
        # Configure logger
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            self.logger.setLevel(getattr(logging, log_level))
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        else:
            self.logger.setLevel(getattr(logging, log_level))
        
        # Set up parameters and caching
        self.cache_size = cache_size
        self.use_jit = use_jit and CATALYST_AVAILABLE
        self.params = params or PanarchyParameters()
        
        # Set up caching via decorators
        self._setup_cached_methods()
        
        self.logger.info(f"Initialized PanarchyAnalyzer (JIT: {self.use_jit}, Cache: {self.cache_size})")
    
    def _setup_cached_methods(self):
        """Setup method caching using LRU cache decorator"""
        # Apply caching to expensive calculations
        self._cached_calculate_pcr = lru_cache(maxsize=self.cache_size)(self._calculate_pcr)
        
        # Keep backward compatibility while using Numba internally
        self._jit_autocorrelation = self._autocorrelation
        self._jit_pcr_calculation = self._pcr_calculation
    
    def _autocorrelation(self, series: np.ndarray, lag: int = 1) -> float:
        """
        Calculate autocorrelation of a series at specified lag.
        
        Args:
            series: Input time series as numpy array
            lag: Lag periods for autocorrelation
            
        Returns:
            Autocorrelation value
        """
        n = len(series)
        if n <= lag or n < 4:
            return 0.0
        
        try:
            # Use Numba implementation
            return _autocorrelation_impl(series, lag)
        except Exception as e:
            self.logger.warning(f"Numba autocorrelation failed: {e}, falling back to Python")
            
            # Fallback to Python implementation
            # Normalize data
            series_norm = series - np.mean(series)
            
            # Calculate autocorrelation
            numerator = np.sum(series_norm[lag:] * series_norm[:-lag])
            denominator = np.sum(series_norm**2)
            
            if denominator < 1e-10:
                return 0.0
                
            return numerator / denominator
    
    def _pcr_calculation(self, prices: np.ndarray, 
                         returns: np.ndarray, 
                         highs: np.ndarray, 
                         lows: np.ndarray,
                         volatility: np.ndarray,
                         period: int,
                         autocorr_lag: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate P, C, R components using vectorized operations.
        
        Args:
            prices: Close price array
            returns: Returns array
            highs: High price array
            lows: Low price array
            volatility: Volatility array
            period: Calculation period
            autocorr_lag: Lag for autocorrelation
            
        Returns:
            Tuple of arrays (P, C, R)
        """
        n = len(prices)
        P = np.full(n, 0.5)
        C = np.full(n, 0.5)
        R = np.full(n, 0.5)
        
        if n < period + 1:
            return P, C, R
        
        try:
            # Use Numba implementation
            return _pcr_calculation_impl(prices, returns, highs, lows, volatility, period, autocorr_lag)
        except Exception as e:
            self.logger.warning(f"Numba PCR calculation failed: {e}, falling back to Python")
            
            # Fallback to Python implementation
            # Calculate P (Potential) - normalized within min/max
            roll_max = np.zeros(n)
            roll_min = np.zeros(n)
            
            for i in range(period, n):
                roll_max[i] = np.max(prices[i-period:i])
                roll_min[i] = np.min(prices[i-period:i])
                
                # Calculate normalized potential
                if roll_max[i] - roll_min[i] > 1e-9:
                    P[i] = (prices[i] - roll_min[i]) / (roll_max[i] - roll_min[i])
            
            # Calculate C (Connectedness) using windowed autocorrelation
            # We need to calculate autocorr for each window
            for i in range(period, n):
                window_returns = returns[i-period:i]
                if len(window_returns) >= autocorr_lag + 1:
                    std_val = np.std(window_returns)
                    if std_val > 1e-9:
                        shifted = window_returns[:-autocorr_lag]
                        shifted_lag = window_returns[autocorr_lag:]
                        if len(shifted) == len(shifted_lag) and len(shifted) > 0:
                            numerator = np.mean((shifted - np.mean(shifted)) * 
                                              (shifted_lag - np.mean(shifted_lag)))
                            denominator = np.std(shifted) * np.std(shifted_lag)
                            if denominator > 1e-9:
                                autocorr = numerator / denominator
                                C[i] = (autocorr + 1) / 2  # Normalize to [0, 1]
            
            # Calculate R (Resilience) - inverse of volatility
            R = 1.0 - volatility
            
            # Ensure values are in [0, 1] range
            P = np.clip(P, 0, 1)
            C = np.clip(C, 0, 1)
            R = np.clip(R, 0, 1)
            
            return P, C, R
    
    def calculate_pcr_components(self, dataframe: pd.DataFrame, period: int = 50) -> pd.DataFrame:
        """
        Calculate Potential, Connectedness, and Resilience components.
        
        Args:
            dataframe (pd.DataFrame): Market data
            period (int): Period for calculations
            
        Returns:
            pd.DataFrame: Market data with PCR components added
        """
        try:
            # Create a copy to avoid modifying the original
            df = dataframe.copy()
            
            if len(df) < period:
                self.logger.warning(f"Insufficient data for P, C, R calculation: {len(df)} < {period}")
                # Create default columns
                df['panarchy_P'] = 0.5
                df['panarchy_C'] = 0.5
                df['panarchy_R'] = 0.5
                return df
            
            # Extract required series and convert to numpy for performance
            close = df['close'].values
            high = df['high'].values if 'high' in df.columns else close
            low = df['low'].values if 'low' in df.columns else close
            
            # Calculate log returns
            log_returns = np.zeros_like(close)
            log_returns[1:] = np.log(close[1:] / np.maximum(close[:-1], 1e-9))
            
            # Calculate volatility proxy for R
            if 'volatility_regime' in df:
                volatility = df['volatility_regime'].values
            else:
                # Calculate rolling volatility if not provided
                volatility = np.zeros_like(close)
                returns = df['close'].pct_change().fillna(0).values
                for i in range(period, len(returns)):
                    volatility[i] = np.std(returns[i-period:i])
                # Normalize to [0, 1]
                max_vol = np.max(volatility)
                if max_vol > 1e-9:
                    volatility = volatility / max_vol
            
            # Use calculated PCR values or cached values if available
            try:
                # Try to use cached values for this specific input
                # Hash key based on input data to ensure cache correctness
                data_key = hash(str(period) + "_PCR_hash")
                try:
                    # Try to include a small sample of the data in the hash
                    if len(close) > 10:
                        data_key = hash(str(period) + "_" + str(hash(close[:10].tobytes())))
                except:
                    # Fallback hash method if tobytes fails
                    pass
                    
                P, C, R = self._cached_calculate_pcr(data_key, period)
                
                # Check if cached values are valid (not empty)
                if len(P) <= 1 or len(P) != len(close):
                    # Cache miss or invalid cached data, calculate directly
                    P, C, R = self._jit_pcr_calculation(
                        close, log_returns, high, low, volatility, period, self.params.autocorr_lag
                    )
            except Exception as e:
                # Calculate PCR directly if caching fails
                self.logger.debug(f"Cache access failed: {e}, calculating directly")
                P, C, R = self._jit_pcr_calculation(
                    close, log_returns, high, low, volatility, period, self.params.autocorr_lag
                )
                    
            # Assign results to dataframe
            df['panarchy_P'] = P
            df['panarchy_C'] = C
            df['panarchy_R'] = R
            
            return df
            
        except Exception as e:
            self.logger.error(f"PCR calculation error: {str(e)}", exc_info=True)
            # Return original dataframe with default values
            df = dataframe.copy()
            df['panarchy_P'] = 0.5
            df['panarchy_C'] = 0.5
            df['panarchy_R'] = 0.5
            return df
    
    def _calculate_pcr(self, data_key: int, period: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Cached PCR calculation wrapper - used by cached_calculate_pcr.
        This is an internal method that works with the LRU cache.
        """
        # This is a placeholder that signals to the caller to perform direct calculation
        # Log as debug rather than warning since this is an expected case
        self.logger.debug(f"Cache miss for key {data_key}, caller should perform direct calculation")
        return np.array([]), np.array([]), np.array([])
    
    def _calculate_potential(self, dataframe: pd.DataFrame, period: int) -> pd.Series:
        """
        Calculate Potential component.
        
        Args:
            dataframe (pd.DataFrame): Market data
            period (int): Period for calculation
            
        Returns:
            pd.Series: Potential values
        """
        try:
            # Calculate trend strength using ADX-like indicator
            # Get price and returns
            close = dataframe['close']
            returns = close.pct_change(1).fillna(0)
            
            # Calculate directional movement
            pos_dm = np.maximum(close.diff(1), 0)
            neg_dm = np.maximum(-close.diff(1), 0)
            
            # Calculate directional indicators
            pos_di = pos_dm.rolling(window=period).mean() / close.rolling(window=period).std()
            neg_di = neg_dm.rolling(window=period).mean() / close.rolling(window=period).std()
            
            # Calculate directional movement index
            dx = abs(pos_di - neg_di) / (pos_di + neg_di) * 100
            
            # Smooth with ADX-like calculation
            adx = dx.rolling(window=period).mean()
            
            # Normalize to [0, 1]
            potential = adx / 100
            potential = potential.clip(0, 1)
            
            # Fill NaN values
            potential = potential.fillna(0.5)
            
            return potential
            
        except Exception as e:
            self.logger.error(f"Potential calculation error: {str(e)}", exc_info=True)
            # Return default values
            return pd.Series(0.5, index=dataframe.index)
    
    def _calculate_connectedness(self, dataframe: pd.DataFrame, period: int) -> pd.Series:
        """
        Calculate Connectedness component.
        
        Args:
            dataframe (pd.DataFrame): Market data
            period (int): Period for calculation
            
        Returns:
            pd.Series: Connectedness values
        """
        try:
            # Calculate autocorrelation
            returns = dataframe['close'].pct_change(1).fillna(0)
            
            # Calculate autocorrelation for multiple lags
            ac1 = returns.rolling(window=period).apply(
                lambda x: pd.Series(x).autocorr(lag=1) if len(x) > 1 else 0)
            ac2 = returns.rolling(window=period).apply(
                lambda x: pd.Series(x).autocorr(lag=2) if len(x) > 2 else 0)
            ac3 = returns.rolling(window=period).apply(
                lambda x: pd.Series(x).autocorr(lag=3) if len(x) > 3 else 0)
            
            # Combine autocorrelations with decreasing weights
            weighted_ac = (0.5 * ac1 + 0.3 * ac2 + 0.2 * ac3)
            
            # Map to [0, 1] range
            connectedness = (weighted_ac + 1) / 2
            
            # Fill NaN values
            connectedness = connectedness.fillna(0.5)
            
            return connectedness
            
        except Exception as e:
            self.logger.error(f"Connectedness calculation error: {str(e)}", exc_info=True)
            # Return default values
            return pd.Series(0.5, index=dataframe.index)
    
    def _calculate_resilience(self, dataframe: pd.DataFrame, period: int) -> pd.Series:
        """
        Calculate Resilience component.
        
        Args:
            dataframe (pd.DataFrame): Market data
            period (int): Period for calculation
            
        Returns:
            pd.Series: Resilience values
        """
        try:
            # In market systems, resilience is related to:
            # 1. Inverse of volatility
            # 2. Market depth and liquidity
            # 3. Diversity of participants
            
            # Calculate inverse volatility component
            returns = dataframe['close'].pct_change(1).fillna(0)
            volatility = returns.rolling(window=period).std()
            inv_volatility = np.exp(-10 * volatility)  # Exponential scaling
            
            # Use volume information if available for liquidity proxy
            if 'volume' in dataframe.columns:
                volume = dataframe['volume']
                rel_volume = volume / volume.rolling(window=period).mean()
                volume_factor = 1 - np.exp(-rel_volume)  # Scale 0-1
            else:
                volume_factor = pd.Series(0.5, index=dataframe.index)
            
            # Combine components
            resilience = 0.7 * inv_volatility + 0.3 * volume_factor
            
            # Clip to [0, 1] range
            resilience = resilience.clip(0, 1)
            
            # Fill NaN values
            resilience = resilience.fillna(0.5)
            
            return resilience
            
        except Exception as e:
            self.logger.error(f"Resilience calculation error: {str(e)}", exc_info=True)
            # Return default values
            return pd.Series(0.5, index=dataframe.index)
    
    def identify_regime(self, dataframe: pd.DataFrame, period: int = 50) -> pd.DataFrame:
        """
        Identify Panarchy regime (adaptive cycle phase).
        
        Args:
            dataframe (pd.DataFrame): Market data
            period (int): Period for calculations
            
        Returns:
            pd.DataFrame: Market data with regime information added
        """
        df = dataframe.copy()

        # Check for required columns and add defaults if missing
        required_columns = {
            'soc_regime': 'unknown',
            'volatility_regime': 0.5,
            'soc_fragility': 0.5,
            'adx': 25.0
        }
        
        missing_columns = []
        for col, default_value in required_columns.items():
            if col not in df.columns:
                missing_columns.append(col)
                df[col] = default_value
                
        if missing_columns:
            self.logger.warning(f"Regime Score Calc: Missing columns: {missing_columns}. Assigning default.")
    
        try:
            # Create a copy to avoid modifying the original
            df = dataframe.copy()
            
            # Ensure PCR components exist
            if not all(c in df.columns for c in ['panarchy_P', 'panarchy_C', 'panarchy_R']):
                self.logger.warning("PCR components not found, calculating them first")
                df = self.calculate_pcr_components(df, period)
            
            # Extract PCR components
            P = df['panarchy_P']
            C = df['panarchy_C']
            R = df['panarchy_R']
            
            # Calculate and add momentum if it doesn't exist
            if 'momentum' not in df.columns:
                if 'close' in df.columns:
                    close = df['close']
                    df['momentum'] = close.pct_change(period // 4).fillna(0)
                    df['momentum_change'] = df['momentum'].diff(period // 4).fillna(0)
                else:
                    df['momentum'] = 0.0
                    df['momentum_change'] = 0.0
            
            # Calculate phase likelihoods
            # Growth phase: increasing P, low C, increasing R
            growth_score = (
                P * (1.1 - C) * R
            ).clip(lower=0)
            
            # Conservation phase: high P, increasing C, decreasing R
            conservation_score = (
                P * C * (1.1 - R)
            ).clip(lower=0)
            
            # Release phase: decreasing P, high C, low R
            release_score = (
                (1.1 - P) * C * (1.1 - R)
            ).clip(lower=0)
            
            # Reorganization phase: low P, decreasing C, increasing R
            reorganization_score = (
                (1.1 - P) * (1.1 - C) * R
            ).clip(lower=0)
            
            # Normalize scores
            scores_df = pd.DataFrame({
                'growth': growth_score,
                'conservation': conservation_score,
                'release': release_score,
                'reorganization': reorganization_score
            }, index=df.index)
            
            # Normalize scores row-wise
            total_scores = scores_df.sum(axis=1)
            safe_total_scores = total_scores.replace(0, 1.0)  # Avoid division by zero
            norm_scores = scores_df.div(safe_total_scores, axis=0)
            
            # Determine phase with highest score
            df['panarchy_phase'] = norm_scores.idxmax(axis=1)
            
            # Calculate numerical score based on phase weights
            regime_score = (
                (norm_scores['reorganization'] * 0.10) +
                (norm_scores['growth'] * 0.35) +
                (norm_scores['conservation'] * 0.65) +
                (norm_scores['release'] * 0.90)
            ).fillna(0.5)
            
            df['panarchy_regime_score'] = regime_score.clip(0, 1)
            
            # Apply hysteresis to prevent phase oscillation
            if 'panarchy_phase_prev' not in df.columns:
                df['panarchy_phase_prev'] = df['panarchy_phase'].shift(1).fillna('conservation')
            
            # Calculate minimum score difference for phase change
            min_score_threshold = self.params.hysteresis_min_score_threshold
            min_score_diff = self.params.hysteresis_min_score_diff
            
            # Apply hysteresis logic
            for i in range(1, len(df)):
                current_phase = df['panarchy_phase'].iloc[i]
                prev_phase = df['panarchy_phase_prev'].iloc[i]
                
                if current_phase != prev_phase:
                    # Phase changed, check if the change is significant
                    current_score = norm_scores.loc[df.index[i], current_phase]
                    prev_phase_score = norm_scores.loc[df.index[i], prev_phase]
                    
                    if (current_score < min_score_threshold or 
                        current_score < prev_phase_score + min_score_diff):
                        # Not a significant change, revert to previous phase
                        df.at[df.index[i], 'panarchy_phase'] = prev_phase
                
                # Update previous phase
                df.at[df.index[i], 'panarchy_phase_prev'] = df['panarchy_phase'].iloc[i]
            
            return df
            
        except Exception as e:
            self.logger.error(f"Regime identification error: {str(e)}", exc_info=True)
            # Return original dataframe with default values
            df = dataframe.copy()
            df['panarchy_regime_score'] = 0.5
            df['panarchy_phase'] = 'unknown'
            return df
    
    def _calculate_regime(self, dataframe: pd.DataFrame, period: int) -> Dict[str, Union[pd.Series, List[str]]]:
        """
        Calculate Panarchy regime.
        
        Args:
            dataframe (pd.DataFrame): Market data
            period (int): Period for calculation
            
        Returns:
            Dict: Regime information
        """
        try:
            # Ensure PCR components exist
            if not all(c in dataframe.columns for c in ['panarchy_P', 'panarchy_C', 'panarchy_R']):
                self.logger.warning("PCR components not found, calculating them first")
                dataframe = self.calculate_pcr_components(dataframe, period)
            
            # Extract PCR components
            potential = dataframe['panarchy_P']
            connectedness = dataframe['panarchy_C']
            resilience = dataframe['panarchy_R']
            
            # Calculate regime score
            # This is a continuous measure through the adaptive cycle
            # 0.0-0.25: Growth (r) phase
            # 0.25-0.5: Conservation (K) phase
            # 0.5-0.75: Release (Ω) phase
            # 0.75-1.0: Reorganization (α) phase
            
            # Initialize score
            score = pd.Series(index=dataframe.index, dtype=float)
            
            # Calculate score based on PCR values
            for i in range(len(dataframe)):
                p = potential.iloc[i]
                c = connectedness.iloc[i]
                r = resilience.iloc[i]
                
                # Growth phase: increasing P, low C, increasing R
                growth_score = (p * (1 - c) * r)
                
                # Conservation phase: high P, increasing C, decreasing R
                conservation_score = (p * c * (1 - r))
                
                # Release phase: decreasing P, high C, low R
                release_score = ((1 - p) * c * (1 - r))
                
                # Reorganization phase: low P, decreasing C, increasing R
                reorganization_score = ((1 - p) * (1 - c) * r)
                
                # Normalize scores
                total = growth_score + conservation_score + release_score + reorganization_score
                if total > 0:
                    growth_score /= total
                    conservation_score /= total
                    release_score /= total
                    reorganization_score /= total
                
                # Calculate final score (weighted position in cycle)
                score.iloc[i] = (0.125 * growth_score + 
                                0.375 * conservation_score + 
                                0.625 * release_score + 
                                0.875 * reorganization_score)
            
            # Determine phase
            phase = []
            for s in score:
                if s < 0.25:
                    phase.append('growth')
                elif s < 0.5:
                    phase.append('conservation')
                elif s < 0.75:
                    phase.append('release')
                else:
                    phase.append('reorganization')
            
            return {
                'score': score.fillna(0.5),
                'phase': phase
            }
            
        except Exception as e:
            self.logger.error(f"Regime calculation error: {str(e)}", exc_info=True)
            # Return default values
            score = pd.Series(0.5, index=dataframe.index)
            phase = ['unknown'] * len(dataframe)
            return {
                'score': score,
                'phase': phase
            }

    def calculate_regime_score(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate a numeric regime score (0-100) indicating market character.
        
        Args:
            dataframe: Input dataframe with required indicators
            
        Returns:
            DataFrame with 'regime_score' column added (0-100 scale)
        """
        required_cols = ['panarchy_phase', 'soc_regime', 'volatility_regime', 'soc_fragility', 'adx']
        default_score = 50.0  # Neutral default
        
        if not all(col in dataframe.columns for col in required_cols):
            missing = [col for col in required_cols if col not in dataframe.columns]
            self.logger.warning(f"Regime Score Calc: Missing columns: {missing}. Assigning default.")
            dataframe['regime_score'] = default_score
            return dataframe
            
        try:
            # Create a copy to avoid modifying the original
            df = dataframe.copy()
            
            # Initialize score Series
            regime_score = pd.Series(default_score, index=df.index)
            
            # 1. Panarchy Phase Influence
            phase = df['panarchy_phase']
            pan_score = pd.Series(default_score, index=df.index)
            pan_score[phase == 'release'] = 90.0
            pan_score[phase == 'reorganization'] = 75.0
            pan_score[phase == 'conservation'] = 50.0
            pan_score[phase == 'growth'] = 25.0
            pan_score[phase == 'unknown'] = default_score
            
            # 2. SOC Regime Influence
            soc_phase = df['soc_regime']
            soc_score = pd.Series(default_score, index=df.index)
            soc_score[soc_phase.isin(['critical', 'unstable'])] = 85.0
            soc_score[soc_phase == 'release'] = 90.0
            soc_score[soc_phase == 'stable'] = 20.0
            soc_score[soc_phase == 'normal'] = 40.0
            
            # 3. Volatility Regime Influence
            vol_score = df['volatility_regime'].clip(0, 1) * 100
            
            # 4. SOC Fragility Influence
            fragility_factor = (df['soc_fragility'] * (1 + (soc_score - 50) / 50 * 0.5)).clip(0, 1.5)
            frag_score = fragility_factor * 100
            
            # 5. Trend Strength (ADX) Influence
            adx = df['adx'].fillna(25)
            adx_score = pd.Series(default_score, index=df.index)
            adx_score[adx < 18] = 70.0  # Low ADX = more unstable/choppy
            adx_score[(adx >= 18) & (adx < 40)] = 30.0  # Medium ADX = smoother trend
            adx_score[adx >= 40] = 60.0  # High ADX = strong but potentially volatile
            
            # Combine Scores with Weights
            weights = {
                'panarchy': 0.30,
                'soc': 0.25,
                'volatility': 0.25,
                'fragility': 0.10,
                'adx': 0.10
            }
            
            combined_score = (
                pan_score * weights['panarchy'] +
                soc_score * weights['soc'] +
                vol_score * weights['volatility'] +
                frag_score * weights['fragility'] +
                adx_score * weights['adx']
            )
            
            # Smooth the final score
            final_score = combined_score.ewm(span=5, min_periods=3).mean()
            
            # Assign final clipped score
            df['regime_score'] = final_score.fillna(default_score).clip(0, 100)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating regime score: {e}", exc_info=True)
            dataframe['regime_score'] = default_score
            return dataframe

    def calculate_panarchy_cycle(self, dataframe: pd.DataFrame, period: int = 50) -> pd.DataFrame:
        """
        Calculate complete panarchy cycle analysis.
        This is the missing method that analyze() calls internally.
        
        Args:
            dataframe: Input dataframe with OHLC data
            period: Period for calculations
            
        Returns:
            DataFrame with complete panarchy analysis
        """
        try:
            # First calculate PCR components if not present
            df_with_pcr = self.calculate_pcr_components(dataframe, period)
            
            # Then identify the panarchy regime/phase
            df_with_regime = self.identify_regime(df_with_pcr, period)
            
            # Add any missing columns needed for regime score calculation
            if 'soc_regime' not in df_with_regime.columns:
                df_with_regime['soc_regime'] = 'normal'
            if 'volatility_regime' not in df_with_regime.columns:
                # Simple volatility regime based on rolling std
                returns = df_with_regime['close'].pct_change().fillna(0)
                rolling_vol = returns.rolling(window=period//2).std()
                vol_threshold = rolling_vol.quantile(0.7)
                df_with_regime['volatility_regime'] = (rolling_vol > vol_threshold).astype(float)
            if 'soc_fragility' not in df_with_regime.columns:
                df_with_regime['soc_fragility'] = 0.5
            if 'adx' not in df_with_regime.columns:
                df_with_regime['adx'] = 25.0
            
            return df_with_regime
            
        except Exception as e:
            self.logger.error(f"Error calculating panarchy cycle: {e}")
            # Return original dataframe with basic columns
            df = dataframe.copy()
            df['panarchy_P'] = 0.5
            df['panarchy_C'] = 0.5
            df['panarchy_R'] = 0.5
            df['panarchy_phase'] = 'unknown'
            df['regime_score'] = 50.0
            return df

    def analyze(self, prices: np.ndarray, volumes: np.ndarray) -> Dict[str, Any]:
        """
        Analyze price and volume data to generate panarchy-based signal.
        Expected interface for CDFA server integration.
        
        Args:
            prices: Array of price values
            volumes: Array of volume values
            
        Returns:
            Dict containing signal, confidence, and analysis details
        """
        try:
            # Convert arrays to DataFrame for internal processing
            df = pd.DataFrame({
                'close': prices,
                'open': prices,  # Approximation
                'high': prices,  # Approximation
                'low': prices,   # Approximation
                'volume': volumes
            })
            
            # Calculate panarchy cycle
            cycle_df = self.calculate_panarchy_cycle(df)
            
            # Calculate regime score
            regime_df = self.calculate_regime_score(cycle_df)
            
            # Get latest regime score and convert to 0-1 signal
            latest_regime = regime_df['regime_score'].iloc[-1] if len(regime_df) > 0 else 50.0
            signal = latest_regime / 100.0  # Convert 0-100 to 0-1
            
            # Get panarchy phase for confidence calculation
            latest_phase = regime_df['panarchy_phase'].iloc[-1] if 'panarchy_phase' in regime_df.columns else 'unknown'
            
            # Calculate confidence based on phase clarity and stability
            if len(regime_df) >= 10:
                recent_scores = regime_df['regime_score'].tail(10)
                score_stability = 1.0 - (np.std(recent_scores) / 100.0)  # Lower std = higher confidence
                
                # Adjust confidence based on phase
                phase_confidence_map = {
                    'growth': 0.8,
                    'conservation': 0.9,
                    'release': 0.7,
                    'reorganization': 0.6,
                    'unknown': 0.5
                }
                phase_confidence = phase_confidence_map.get(latest_phase, 0.5)
                
                # Combine stability and phase confidence
                confidence = (score_stability * 0.6 + phase_confidence * 0.4)
                confidence = max(0.1, min(1.0, confidence))
            else:
                confidence = 0.5  # Default confidence for insufficient data
            
            return {
                "signal": float(signal),
                "confidence": float(confidence),
                "regime_score": float(latest_regime),
                "panarchy_phase": str(latest_phase),
                "analysis_type": "panarchy",
                "data_points": len(prices)
            }
            
        except Exception as e:
            self.logger.error(f"Error in Panarchy analyze method: {e}")
            return {
                "signal": 0.5,
                "confidence": 0.0,
                "error": str(e),
                "analysis_type": "panarchy"
            }