import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Union, Any, Callable, Set
from functools import lru_cache
import time
from dataclasses import dataclass
import numba as nb
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


@dataclass
class FibonacciParameters:
    """Parameters for Fibonacci calculations"""
    # Fibonacci retracement levels
    retracement_levels: Dict[str, float] = None
    
    # Fibonacci extension levels
    extension_levels: Dict[str, float] = None
    
    # Tolerance for alignment score calculation
    alignment_tolerance: float = 0.006
    
    # Default tolerance for regime score adaptation
    base_tolerance: float = 0.006
    max_tolerance_factor: float = 2.0
    
    # ATR period for volatility bands
    atr_period: int = 14
    
    # Hysteresis parameters for trend stability
    trend_hysteresis_threshold: float = 0.01
    
    def __post_init__(self):
        """Initialize default Fibonacci levels if not provided"""
        if self.retracement_levels is None:
            self.retracement_levels = {
                "0.0": 0.0,
                "23.6": 0.236,
                "38.2": 0.382,
                "50.0": 0.5,
                "61.8": 0.618,
                "78.6": 0.786,
                "100.0": 1.0
            }
        
        if self.extension_levels is None:
            self.extension_levels = {
                "100.0": 1.0,
                "127.2": 1.272,
                "161.8": 1.618,
                "261.8": 2.618,
                "361.8": 3.618
            }


# Numba helper functions
@njit(cache=True)
def _distance_calculation_impl(prices, levels):
    """Numba implementation of distance calculation"""
    n_prices = len(prices)
    n_levels = len(levels)
    min_distances = np.ones(n_prices)
    
    for i in range(n_prices):
        if prices[i] <= 0:
            continue
            
        min_dist = 1.0
        for j in range(n_levels):
            dist = abs(levels[j] - prices[i]) / prices[i]
            if dist < min_dist:
                min_dist = dist
                
        min_distances[i] = min_dist
    
    return min_distances

@njit(cache=True)
def _find_swing_points_impl(high, low, period):
    """
    Numba implementation to find swing high and low points
    
    Returns:
        Tuple of (swing_high_values, swing_low_values, new_swing_high_mask, new_swing_low_mask)
    """
    n = len(high)
    if n < period:
        return np.zeros(n), np.zeros(n), np.zeros(n, dtype=np.bool_), np.zeros(n, dtype=np.bool_)
    
    swing_high_values = np.zeros(n)
    swing_low_values = np.zeros(n)
    new_swing_high_mask = np.zeros(n, dtype=np.bool_)
    new_swing_low_mask = np.zeros(n, dtype=np.bool_)
    
    # Calculate rolling max/min
    for i in range(period-1, n):
        # Calculate rolling max for swing highs
        high_max = high[i]
        for j in range(i-period+1, i+1):
            if high[j] > high_max:
                high_max = high[j]
        swing_high_values[i] = high_max
        
        # Calculate rolling min for swing lows
        low_min = low[i]
        for j in range(i-period+1, i+1):
            if low[j] < low_min:
                low_min = low[j]
        swing_low_values[i] = low_min
    
    # Detect swing points
    for i in range(period, n):
        # New swing high
        if (swing_high_values[i] != swing_high_values[i-1]) and (swing_high_values[i] == high[i]):
            new_swing_high_mask[i] = True
        
        # New swing low
        if (swing_low_values[i] != swing_low_values[i-1]) and (swing_low_values[i] == low[i]):
            new_swing_low_mask[i] = True
    
    return swing_high_values, swing_low_values, new_swing_high_mask, new_swing_low_mask

@njit(cache=True)
def _calculate_alignment_scores(current_price, levels):
    """Calculate alignment scores using Numba"""
    n_levels = len(levels)
    if n_levels == 0 or current_price <= 0:
        return 1.0
    
    min_distance = 1.0
    for i in range(n_levels):
        if levels[i] <= 0:
            continue
        distance = abs(current_price - levels[i]) / current_price
        if distance < min_distance:
            min_distance = distance
    
    return min_distance


class FibonacciAnalyzer:
    """
    Optimized Fibonacci analysis with vectorized operations, caching, and JIT.
    
    This class implements vectorized algorithms for:
    - Fibonacci retracement/extension identification
    - Swing point detection
    - Alignment scoring
    - Multi-timeframe confluence analysis
    - Volatility-based Fibonacci bands
    """
    
    def __init__(self, 
                 cache_size: int = 100, 
                 use_jit: bool = True,
                 params: Optional[FibonacciParameters] = None,
                 log_level: str = "INFO"):
        """
        Initialize FibonacciAnalyzer with caching and JIT optimization.
        
        Args:
            cache_size: Size of the LRU cache for expensive calculations
            use_jit: Whether to use JIT compilation via PennyLane Catalyst
            params: Optional custom parameters for Fibonacci calculations
            log_level: Logging level (default: INFO)
        """
        self.cache_size = cache_size
        self.use_jit = use_jit and CATALYST_AVAILABLE
        self.params = params or FibonacciParameters()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(getattr(logging, log_level))
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        # Apply caching
        self._setup_cached_methods()
        
        # Setup result cache for frequently accessed calculations
        self._calculation_cache = {}
        
        # Log initialization status
        self.logger.info(f"Initialized FibonacciAnalyzer (JIT: use_numba=True, Cache: {self.cache_size})")
    
    def _setup_cached_methods(self):
        """Setup method caching using LRU cache decorator"""
        # Apply caching to expensive calculations
        self._cached_find_swing_points = lru_cache(maxsize=self.cache_size)(self._find_swing_points)
        
        # Keep backward compatibility while using Numba internally
        self._jit_distance_calculation = self._distance_calculation
    
    def _distance_calculation(self, prices: np.ndarray, levels: np.ndarray) -> np.ndarray:
        """
        Calculate distance between prices and Fibonacci levels.
        
        Args:
            prices: Price array
            levels: Fibonacci levels array
            
        Returns:
            Array of minimum distances (as percentage of price)
        """
        n_prices = len(prices)
        n_levels = len(levels)
        if n_prices == 0 or n_levels == 0:
            return np.ones(n_prices)
        
        try:
            # Use Numba implementation
            return _distance_calculation_impl(prices, levels)
        except Exception as e:
            self.logger.warning(f"Numba implementation failed: {e}, falling back to Python")
            
            # Fallback to Python implementation
            min_distances = np.ones(n_prices)
            
            for i in range(n_prices):
                if prices[i] <= 0:
                    continue
                    
                distances = np.abs(levels - prices[i]) / prices[i]
                min_dist = np.min(distances) if len(distances) > 0 else 1.0
                min_distances[i] = min_dist
            
            return min_distances
    
    def _find_swing_points(self, data_key: str, high: np.ndarray, low: np.ndarray, period: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Find swing high and low points using rolling max/min (for LRU cache).
        
        Args:
            data_key: Cache key
            high: High prices array
            low: Low prices array
            period: Lookback period
            
        Returns:
            Tuple of (swing_high_values, swing_low_values, new_swing_high_mask, new_swing_low_mask)
        """
        try:
            # Use Numba implementation
            return _find_swing_points_impl(high, low, period)
        except Exception as e:
            self.logger.warning(f"Numba swing point detection failed: {e}, using Python implementation")
            # Fallback to empty arrays 
            # In practice, identify_swing_points will handle this case appropriately
            return (np.array([]), np.array([]), np.array([]), np.array([]))
    
    def identify_swing_points(self, dataframe: pd.DataFrame, period: int) -> pd.DataFrame:
        """
        Identify swing high/low points using rolling max/min with vectorized operations.
        
        Args:
            dataframe: Input dataframe with high/low prices
            period: Lookback period for swing point detection
            
        Returns:
            DataFrame with swing point columns added
        """
        if len(dataframe) < period:
            self.logger.debug(f"Insufficient data for swing points: {len(dataframe)} < {period}")
            return dataframe
        
        try:
            # Create column names based on period
            high_col = f'swing_high_{period}'
            low_col = f'swing_low_{period}'
            new_high_col = f'new_swing_high_{period}'
            new_low_col = f'new_swing_low_{period}'
            
            # Check if already in cache
            cache_key = f"swing_{...}_{hash(dataframe['high'].iloc[-period:].to_numpy().tobytes() + dataframe['low'].iloc[-period:].to_numpy().tobytes())}"
            
            if cache_key in self._calculation_cache:
                # Retrieve cached calculations
                cached = self._calculation_cache[cache_key]
                dataframe[high_col] = cached['high']
                dataframe[low_col] = cached['low']
                dataframe[new_high_col] = cached['new_high']
                dataframe[new_low_col] = cached['new_low']
                return dataframe
            
            # Get high/low values as numpy arrays for faster processing
            high = dataframe['high'].values
            low = dataframe['low'].values
            
            # Use numba accelerated function with fallback
            try:
                # First try direct calculation for better performance
                swing_high_values, swing_low_values, new_swing_high_mask, new_swing_low_mask = _find_swing_points_impl(high, low, period)
                
                # Convert results back to pandas series
                dataframe[high_col] = pd.Series(swing_high_values, index=dataframe.index)
                dataframe[low_col] = pd.Series(swing_low_values, index=dataframe.index)
                dataframe[new_high_col] = pd.Series(new_swing_high_mask, index=dataframe.index)
                dataframe[new_low_col] = pd.Series(new_swing_low_mask, index=dataframe.index)
            except Exception:
                # Fallback to pandas operations if Numba fails
                self.logger.debug("Using pandas operations for swing point detection")
                # Calculate rolling max/min
                dataframe[high_col] = dataframe['high'].rolling(window=period).max()
                dataframe[low_col] = dataframe['low'].rolling(window=period).min()
                
                # Initialize new swing point columns
                dataframe[new_high_col] = False
                dataframe[new_low_col] = False
                
                # Detect new swing points using vectorized operations
                if len(dataframe) > period:
                    # Shift to identify when the max/min changes
                    prev_high = dataframe[high_col].shift(1)
                    prev_low = dataframe[low_col].shift(1)
                    
                    # Conditions for new swing points
                    high_cond = (dataframe[high_col] != prev_high) & (dataframe[high_col] == dataframe['high'])
                    low_cond = (dataframe[low_col] != prev_low) & (dataframe[low_col] == dataframe['low'])
                    
                    # Apply conditions
                    dataframe.loc[high_cond.index, new_high_col] = high_cond
                    dataframe.loc[low_cond.index, new_low_col] = low_cond
            
            # Cache the results
            self._calculation_cache[cache_key] = {
                'high': dataframe[high_col],
                'low': dataframe[low_col],
                'new_high': dataframe[new_high_col],
                'new_low': dataframe[new_low_col]
            }
            
            return dataframe
            
        except Exception as e:
            self.logger.error(f"Error identifying swing points: {e}", exc_info=True)
            # Ensure columns exist even on error
            dataframe[f'swing_high_{period}'] = dataframe['high']
            dataframe[f'swing_low_{period}'] = dataframe['low']
            dataframe[f'new_swing_high_{period}'] = False
            dataframe[f'new_swing_low_{period}'] = False
            return dataframe
    
    def calculate_retracements(self, dataframe: pd.DataFrame, period: int) -> pd.DataFrame:
        """
        Calculate Fibonacci retracement levels based on most recent swings.
        
        Args:
            dataframe: Input dataframe with swing points already identified
            period: The period used for swing point detection
            
        Returns:
            DataFrame with Fibonacci retracement levels added
        """
        # Check prerequisites
        swing_high_col = f'new_swing_high_{period}'
        swing_low_col = f'new_swing_low_{period}'
        
        if swing_high_col not in dataframe.columns or swing_low_col not in dataframe.columns:
            self.logger.warning(f"Swing points not found. Call identify_swing_points first.")
            # Initialize necessary columns
            fib_levels = list(self.params.retracement_levels.keys())
            for level in fib_levels:
                dataframe[f'fib_retr_{level}'] = np.nan
            dataframe['fib_trend'] = 'unknown'
            return dataframe
        
        if len(dataframe) < period:
            self.logger.debug(f"Insufficient data for retracements: {len(dataframe)} < {period}")
            # Initialize necessary columns
            fib_levels = list(self.params.retracement_levels.keys())
            for level in fib_levels:
                dataframe[f'fib_retr_{level}'] = np.nan
            dataframe['fib_trend'] = 'unknown'
            return dataframe
        
        try:
            # Create empty columns for each Fibonacci level
            fib_levels = list(self.params.retracement_levels.keys())
            for level in fib_levels:
                dataframe[f'fib_retr_{level}'] = np.nan
            
            # Initialize trend column
            dataframe['fib_trend'] = 'unknown'
            
            # Find the most recent swing high/low indices
            high_series = dataframe[swing_high_col]
            low_series = dataframe[swing_low_col]
            
            last_high_idx = high_series[high_series].last_valid_index()
            last_low_idx = low_series[low_series].last_valid_index()
            
            if last_high_idx is None or last_low_idx is None:
                self.logger.debug("No valid swing points found for retracements")
                return dataframe
            
            # Get prices at these indices
            recent_swing_high_price = dataframe.loc[last_high_idx, 'high']
            recent_swing_low_price = dataframe.loc[last_low_idx, 'low']
            
            # Check for valid price difference
            diff = abs(recent_swing_high_price - recent_swing_low_price)
            if diff < 1e-9:
                self.logger.debug(f"Swing points too close: diff={diff}")
                return dataframe
            
            # Determine trend based on which swing was more recent
            trend = 'up' if last_high_idx > last_low_idx else 'down'
            dataframe['fib_trend'] = trend
            
            # Calculate retracement levels
            if trend == 'up':
                high_val = recent_swing_high_price
                low_val = recent_swing_low_price
                
                # Vectorized calculation
                for level_key, level_pct in self.params.retracement_levels.items():
                    retr_val = high_val - level_pct * diff
                    dataframe[f'fib_retr_{level_key}'] = retr_val
            else:  # downtrend
                high_val = recent_swing_high_price
                low_val = recent_swing_low_price
                
                # Vectorized calculation
                for level_key, level_pct in self.params.retracement_levels.items():
                    retr_val = low_val + level_pct * diff
                    dataframe[f'fib_retr_{level_key}'] = retr_val
            
            # Forward fill for all subsequent rows
            fib_columns = [f'fib_retr_{l}' for l in fib_levels] + ['fib_trend']
            dataframe[fib_columns] = dataframe[fib_columns].ffill()
            
            # Log latest retracement levels
            if not dataframe.empty:
                last_idx = dataframe.index[-1]
                level_values = {
                    level_key: dataframe.loc[last_idx, f'fib_retr_{level_key}'] 
                    for level_key in fib_levels
                    if f'fib_retr_{level_key}' in dataframe.columns
                }
                self.logger.debug(f"Retracement levels calculated. Trend: {dataframe.loc[last_idx, 'fib_trend']}")
            
            return dataframe
            
        except Exception as e:
            self.logger.error(f"Error calculating retracements: {e}", exc_info=True)
            # Ensure columns exist even on error
            for level in fib_levels:
                dataframe[f'fib_retr_{level}'] = np.nan
            dataframe['fib_trend'] = 'unknown'
            return dataframe
    
    def calculate_extensions(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Fibonacci extension levels based on the latest retracement.
        
        Args:
            dataframe: Input dataframe with retracement levels already calculated
            
        Returns:
            DataFrame with Fibonacci extension levels added
        """
        # Check prerequisites
        if 'fib_trend' not in dataframe.columns or 'fib_retr_0.0' not in dataframe.columns:
            self.logger.warning("Retracement levels not found. Call calculate_retracements first.")
            # Initialize extension columns
            ext_levels = list(self.params.extension_levels.keys())
            for level in ext_levels:
                dataframe[f'fib_ext_{level}'] = np.nan
            return dataframe
        
        try:
            # Create columns for extension levels
            ext_levels = list(self.params.extension_levels.keys())
            for level in ext_levels:
                dataframe[f'fib_ext_{level}'] = np.nan
            
            # Extract data for vectorized calculation
            trends = dataframe['fib_trend'].values
            zeros = dataframe['fib_retr_0.0'].values
            hundreds = dataframe['fib_retr_100.0'].values
            
            # Arrays to store calculated extensions
            ext_arrays = {f'fib_ext_{level}': np.full(len(dataframe), np.nan) for level in ext_levels}
            
            # Calculate extensions using Numba for core calculation
            for i in range(1, len(dataframe)):
                trend = trends[i]
                if trend == 'unknown' or pd.isna(zeros[i]) or pd.isna(hundreds[i]):
                    continue
                
                # Determine swing high/low based on trend
                swing_high = zeros[i] if trend == 'up' else hundreds[i]
                swing_low = hundreds[i] if trend == 'up' else zeros[i]
                
                # Calculate price difference
                diff = abs(swing_high - swing_low)
                if diff < 1e-9:
                    continue
                
                # Calculate extension levels
                if trend == 'up':
                    for level_key, ext_factor in self.params.extension_levels.items():
                        ext_val = swing_high + (ext_factor - 1.0) * diff
                        ext_arrays[f'fib_ext_{level_key}'][i] = ext_val
                else:  # downtrend
                    for level_key, ext_factor in self.params.extension_levels.items():
                        ext_val = swing_low - (ext_factor - 1.0) * diff
                        ext_arrays[f'fib_ext_{level_key}'][i] = ext_val
            
            # Assign calculated columns back to dataframe
            for col_name, values in ext_arrays.items():
                dataframe[col_name] = values
            
            # Forward fill extension values
            for level in ext_levels:
                dataframe[f'fib_ext_{level}'] = dataframe[f'fib_ext_{level}'].ffill()
            
            return dataframe
            
        except Exception as e:
            self.logger.error(f"Error calculating extensions: {e}", exc_info=True)
            # Ensure columns exist even on error
            for level in ext_levels:
                dataframe[f'fib_ext_{level}'] = np.nan
            return dataframe
    
    def calculate_volatility_bands(self, dataframe: pd.DataFrame, base_period: int = 20, atr_period: int = 14) -> pd.DataFrame:
        """
        Create adaptive bands based on Fibonacci and volatility.
        
        Args:
            dataframe: Input dataframe with OHLC data
            base_period: Period for base price calculation
            atr_period: Period for ATR calculation
            
        Returns:
            DataFrame with Fibonacci volatility bands added
        """
        if not TALIB_AVAILABLE:
            self.logger.warning("TA-Lib not available for Fibonacci Volatility Bands")
            return dataframe
        
        try:
            # Calculate ATR
            dataframe['atr'] = ta.ATR(dataframe, timeperiod=atr_period)
            
            # Calculate base price (EMA)
            dataframe['fib_base'] = ta.EMA(dataframe, timeperiod=base_period)
            
            # Define Fibonacci ratios for bands
            fib_ratios = [0.618, 1.0, 1.618, 2.618]
            
            # Calculate bands with vectorized operations
            for ratio in fib_ratios:
                # Ensure ATR values are not NaN
                atr_values = dataframe['atr'].fillna(0)
                
                # Calculate upper and lower bands
                dataframe[f'fib_upper_{ratio}'] = dataframe['fib_base'] + (atr_values * ratio)
                dataframe[f'fib_lower_{ratio}'] = dataframe['fib_base'] - (atr_values * ratio)
            
            return dataframe
            
        except Exception as e:
            self.logger.error(f"Error calculating Fibonacci volatility bands: {e}", exc_info=True)
            # Ensure columns exist even on error
            dataframe['fib_base'] = dataframe['close']
            for ratio in [0.618, 1.0, 1.618, 2.618]:
                dataframe[f'fib_upper_{ratio}'] = dataframe['close']
                dataframe[f'fib_lower_{ratio}'] = dataframe['close']
            return dataframe
    
    def calculate_alignment_score(self, dataframe: pd.DataFrame, period: int, tolerance: Optional[float] = None) -> pd.DataFrame:
        """
        Calculate proximity score (0-1) to recent Fibonacci levels.
        
        Args:
            dataframe: Input dataframe with price data
            period: Period for rolling high/low calculation
            tolerance: Distance threshold as fraction of price (default: from params)
            
        Returns:
            DataFrame with 'fib_alignment_score' column added
        """
        col_name = 'fib_alignment_score'
        default_score = 0.0
        
        # Use provided tolerance or default from parameters
        if tolerance is None:
            tolerance = self.params.alignment_tolerance
        
        try:
            if len(dataframe) < period:
                dataframe[col_name] = default_score
                return dataframe
            
            # Check for cached result
            cache_key = f"alignment_{period}_{tolerance}_{len(dataframe)}_{hash(dataframe['close'].iloc[-period:].to_numpy().tobytes())}"
            
            if cache_key in self._calculation_cache:
                dataframe[col_name] = self._calculation_cache[cache_key]
                return dataframe
            
            # Calculate rolling high/low and price range
            roll_high = dataframe['high'].rolling(period, min_periods=period//2).max()
            roll_low = dataframe['low'].rolling(period, min_periods=period//2).min()
            price_range = roll_high - roll_low
            current_price = dataframe['close']
            
            # Initialize minimum distance percentage
            min_distance_pct = pd.Series(1.0, index=dataframe.index)
            
            # Fibonacci levels to check
            fib_levels_pct = np.array([0.236, 0.382, 0.5, 0.618, 0.786])
            
            # Create valid mask for calculation
            valid_mask = (price_range > 1e-9) & (current_price > 1e-9) & (~current_price.isna())
            
            # Use Numba for core calculation on valid rows
            for i in dataframe[valid_mask].index:
                # Get values for this row
                idx = dataframe.index.get_loc(i)
                price = current_price.iloc[idx]
                high_val = roll_high.iloc[idx]
                low_val = roll_low.iloc[idx]
                
                if price <= 0 or high_val <= low_val:
                    continue
                
                # Calculate levels from high to low
                levels_down = high_val - fib_levels_pct * (high_val - low_val)
                # Calculate levels from low to high
                levels_up = low_val + fib_levels_pct * (high_val - low_val)
                
                # Combine all levels
                all_levels = np.concatenate((levels_down, levels_up))
                
                # Calculate minimum distance using Numba
                min_dist = _calculate_alignment_scores(price, all_levels)
                min_distance_pct.iloc[idx] = min_dist
            
            # Ensure minimum distance is valid
            min_distance_pct = min_distance_pct.fillna(1.0)
            
            # Calculate score: closer = higher score
            alignment_score = (1.0 - (min_distance_pct / tolerance)).clip(0.0, 1.0)
            
            # Cache the result
            self._calculation_cache[cache_key] = alignment_score
            
            # Assign to dataframe
            dataframe[col_name] = alignment_score
            
            self.logger.debug(f"Calculated alignment score. Last value: {alignment_score.iloc[-1]:.4f}")
            
            return dataframe
            
        except Exception as e:
            self.logger.error(f"Error calculating {col_name}: {e}", exc_info=True)
            dataframe[col_name] = default_score
            return dataframe
    
    def calculate_mtf_confluence(self, dataframe: pd.DataFrame, timeframes: List[str], metadata: dict, tolerance: Optional[float] = None) -> pd.DataFrame:
        """
        Calculate Fibonacci confluence score based on proximity to levels from multiple timeframes.
        
        Args:
            dataframe: Input dataframe with merged MTF Fibonacci levels
            timeframes: List of timeframes to consider for confluence
            metadata: Pair metadata
            tolerance: Distance threshold as fraction of price (default: from params)
            
        Returns:
            DataFrame with 'fib_mtf_confluence' column added
        """
        pair = metadata.get('pair', 'unknown_pair')
        col_name = 'fib_mtf_confluence'
        dataframe[col_name] = 0  # Initialize with zero
        
        # Use provided tolerance or default from parameters
        if tolerance is None:
            tolerance = self.params.alignment_tolerance
        
        try:
            # Validate current price
            if dataframe.empty:
                self.logger.debug(f"[{pair}] Empty dataframe. Skipping confluence.")
                return dataframe
                
            last_idx = dataframe.index[-1]
            current_price = dataframe.loc[last_idx, 'close']
            
            if pd.isna(current_price) or current_price <= 1e-9:
                self.logger.debug(f"[{pair}] Invalid current_price ({current_price}). Skipping confluence.")
                return dataframe
            
            # Check for cached result
            cache_key = f"confluence_{tolerance}_{pair}_{last_idx}"
            
            if cache_key in self._calculation_cache:
                dataframe.loc[last_idx, col_name] = self._calculation_cache[cache_key]
                return dataframe
            
            # Count nearby Fibonacci levels from all timeframes
            nearby_levels_count = 0
            nearby_levels_details = []
            
            # Define key retracement levels to check
            fib_levels_retr = ['23.6', '38.2', '50.0', '61.8', '78.6']
            
            # Iterate through timeframes and check all Fibonacci levels
            for tf in timeframes:
                # Determine suffix for column names
                suffix = f"_{tf}" if tf != dataframe.attrs.get('timeframe', '') else ""
                
                # Check retracement levels from this timeframe
                for level in fib_levels_retr:
                    # Construct the column name for the level
                    col_raw = f'fib_retr_{level}{suffix}'
                    
                    # Skip if column doesn't exist in the merged dataframe
                    if col_raw not in dataframe.columns:
                        continue
                    
                    # Get the level price from the last row
                    level_price = dataframe.loc[last_idx, col_raw]
                    
                    # Skip invalid prices
                    if pd.isna(level_price):
                        continue
                    
                    # Calculate percentage distance
                    distance_pct = abs(current_price - level_price) / current_price
                    
                    # Check if within tolerance
                    if distance_pct <= tolerance:
                        nearby_levels_count += 1
                        nearby_levels_details.append(f"{tf}@{level}% ({level_price:.4f})")
                        self.logger.debug(f"[{pair}] Confluence hit: Current {current_price:.4f} near {col_raw} ({level_price:.4f}), dist={distance_pct:.4f}")
            
            # Store the confluence count
            dataframe.loc[last_idx, col_name] = nearby_levels_count
            
            # Cache the result
            self._calculation_cache[cache_key] = nearby_levels_count
            
            # Log if there's confluence
            if nearby_levels_count > 0:
                self.logger.info(f"[{pair}] MTF Confluence score: {nearby_levels_count}. Nearby levels: {', '.join(nearby_levels_details)}")
            
            return dataframe
            
        except KeyError as e:
            self.logger.warning(f"[{pair}] Error calculating MTF confluence: Missing column {e}.")
            dataframe.loc[dataframe.index[-1] if not dataframe.empty else 0, col_name] = 0
            return dataframe
            
        except Exception as e:
            self.logger.error(f"[{pair}] Unexpected error calculating MTF confluence: {e}", exc_info=True)
            dataframe.loc[dataframe.index[-1] if not dataframe.empty else 0, col_name] = 0
            return dataframe
       
 
    def adjust_fibonacci_by_regime(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Adjust Fibonacci level tolerance based on regime scores.
        
        Args:
            dataframe: Input dataframe with Fibonacci levels and regime_score
            
        Returns:
            DataFrame with adjusted Fibonacci bands
        """
        fib_levels = ['0.0', '23.6', '38.2', '50.0', '61.8', '78.6', '100.0']
        
        if 'regime_score' not in dataframe.columns:
            self.logger.warning("Cannot adjust Fib by regime: 'regime_score' column missing.")
            # Initialize bands equal to levels
            for level in fib_levels:
                col = f'fib_retr_{level}'
                if col in dataframe.columns:
                    dataframe[f'{col}_upper'] = dataframe[col]
                    dataframe[f'{col}_lower'] = dataframe[col]
                else:
                    dataframe[f'{col}_upper'] = np.nan
                    dataframe[f'{col}_lower'] = np.nan
            return dataframe
        
        try:
            # Calculate tolerance based on regime score
            base_tolerance = self.params.base_tolerance
            
            # Scale tolerance: regime=0 -> base, regime=50 -> 1.5*base, regime=100 -> max_factor*base
            # Linear scaling formula
            max_factor = self.params.max_tolerance_factor
            scaling = 1.0 + (dataframe['regime_score'].fillna(50).clip(0, 100) / 100.0) * (max_factor - 1.0)
            
            adj_tolerance = base_tolerance * scaling
            
            # Vectorized calculation of upper/lower bands
            for level in fib_levels:
                col = f'fib_retr_{level}'
                if col not in dataframe.columns:
                    continue
                
                level_price = dataframe[col]
                
                # Calculate upper/lower bands based on adjusted tolerance
                dataframe[f'{col}_upper'] = level_price * (1 + adj_tolerance)
                dataframe[f'{col}_lower'] = level_price * (1 - adj_tolerance)
            
            return dataframe
            
        except Exception as e:
            self.logger.error(f"Error adjusting Fib levels by regime: {e}", exc_info=True)
            # Initialize bands equal to levels on error
            for level in fib_levels:
                col = f'fib_retr_{level}'
                df_col = dataframe.get(col)
                dataframe[f'{col}_upper'] = df_col if df_col is not None else np.nan
                dataframe[f'{col}_lower'] = df_col if df_col is not None else np.nan
            return dataframe

    def calculate_fibonacci_retracements(self, dataframe: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        Calculate Fibonacci retracement levels based on swing points.
        This is the missing method that analyze() calls internally.
        
        Args:
            dataframe: Input dataframe with OHLC data
            period: Period for swing point detection
            
        Returns:
            DataFrame with Fibonacci retracement levels added
        """
        try:
            # First identify swing points
            df_with_swings = self.identify_swing_points(dataframe, period)
            
            # Then calculate retracements based on those swing points
            df_with_retracements = self.calculate_retracements(df_with_swings, period)
            
            return df_with_retracements
            
        except Exception as e:
            self.logger.error(f"Error calculating Fibonacci retracements: {e}")
            # Return original dataframe with basic Fibonacci columns
            fib_levels = list(self.params.retracement_levels.keys())
            for level in fib_levels:
                dataframe[f'fib_retr_{level}'] = np.nan
            return dataframe

    def calculate_fibonacci_alignment_score(self, dataframe: pd.DataFrame, period: int = 20) -> pd.Series:
        """
        Calculate Fibonacci alignment score series.
        This is the missing method that analyze() calls internally.
        
        Args:
            dataframe: Input dataframe with Fibonacci levels
            period: Period for calculation
            
        Returns:
            Series with alignment scores
        """
        try:
            df_with_score = self.calculate_alignment_score(dataframe, period)
            return df_with_score['fib_alignment_score']
            
        except Exception as e:
            self.logger.error(f"Error calculating Fibonacci alignment score: {e}")
            # Return default score series
            return pd.Series(0.5, index=dataframe.index)

    def analyze(self, prices: np.ndarray, volumes: np.ndarray) -> Dict[str, Any]:
        """
        Analyze price and volume data to generate Fibonacci-based signal.
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
            
            # Calculate Fibonacci retracements and extensions
            fib_df = self.calculate_fibonacci_retracements(df)
            
            # Calculate alignment score
            alignment_series = self.calculate_fibonacci_alignment_score(fib_df)
            
            # Get latest alignment score as signal
            latest_alignment = alignment_series.iloc[-1] if len(alignment_series) > 0 else 0.5
            
            # Calculate confidence based on price proximity to multiple Fibonacci levels
            current_price = prices[-1] if len(prices) > 0 else 0
            confidence = 0.5  # Default confidence
            
            if current_price > 0 and len(alignment_series) > 0:
                # Count how many Fibonacci levels are close to current price
                fib_levels = ['23.6', '38.2', '50.0', '61.8', '78.6']
                close_levels = 0
                total_levels = 0
                
                for level in fib_levels:
                    col = f'fib_retr_{level}'
                    if col in fib_df.columns:
                        fib_price = fib_df[col].iloc[-1]
                        if not pd.isna(fib_price) and fib_price > 0:
                            total_levels += 1
                            # Check if current price is within 2% of Fibonacci level
                            if abs(current_price - fib_price) / fib_price < 0.02:
                                close_levels += 1
                
                if total_levels > 0:
                    # Higher confidence when multiple levels align
                    level_confidence = close_levels / total_levels
                    confidence = 0.5 + (level_confidence * 0.4)  # Scale to 0.5-0.9 range
            
            return {
                "signal": float(latest_alignment),
                "confidence": float(confidence),
                "fibonacci_alignment": float(latest_alignment),
                "analysis_type": "fibonacci",
                "data_points": len(prices),
                "current_price": float(current_price)
            }
            
        except Exception as e:
            self.logger.error(f"Error in Fibonacci analyze method: {e}")
            return {
                "signal": 0.5,
                "confidence": 0.0,
                "error": str(e),
                "analysis_type": "fibonacci"
            }
