import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from functools import lru_cache
import time
from dataclasses import dataclass
import numba as nb
from numba import njit, float64, int64, boolean
import sys
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
class AntifragilityParameters:
    """Parameters for Antifragility calculations"""
    # Component weights
    convexity_weight: float = 0.40
    asymmetry_weight: float = 0.20
    recovery_weight: float = 0.25
    benefit_ratio_weight: float = 0.15

    # Volatility estimation parameters
    yz_volatility_k: float = 0.34
    garch_alpha_base: float = 0.05
    parkinson_factor: float = 4 * np.log(2)

    # Signal processing parameters
    recovery_horizon_factor: float = 0.5  # Future horizon = vol_period * this factor
    vol_lookback_factor: float = 3  # Long period = period * this factor


# Numba helper functions
@njit(cache=True)
def _convexity_calculation_impl(perf_acceleration, vol_roc_smoothed, window):
    """Numba implementation of convexity calculation"""
    n = len(perf_acceleration)
    convexity = np.full(n, 0.0)

    for i in range(window, n):
        perf_window = perf_acceleration[i-window:i]
        vol_window = vol_roc_smoothed[i-window:i]

        # Calculate means
        perf_mean = 0.0
        vol_mean = 0.0
        for j in range(len(perf_window)):
            perf_mean += perf_window[j]
            vol_mean += vol_window[j]

        if len(perf_window) > 0:
            perf_mean /= len(perf_window)
            vol_mean /= len(vol_window)

        # Calculate standard deviations
        perf_std = 0.0
        vol_std = 0.0
        for j in range(len(perf_window)):
            perf_std += (perf_window[j] - perf_mean) ** 2
            vol_std += (vol_window[j] - vol_mean) ** 2

        if len(perf_window) > 1:
            perf_std = np.sqrt(perf_std / (len(perf_window) - 1))
            vol_std = np.sqrt(vol_std / (len(vol_window) - 1))

        # Calculate covariance
        cov = 0.0
        for j in range(len(perf_window)):
            cov += (perf_window[j] - perf_mean) * (vol_window[j] - vol_mean)

        if len(perf_window) > 0:
            cov /= len(perf_window)

        # Calculate correlation
        if perf_std > 1e-9 and vol_std > 1e-9:
            corr = cov / (perf_std * vol_std)
            convexity[i] = corr

    return convexity

@njit(cache=True)
def _asymmetry_calculation_impl(returns, window, vol_regime):
    """Numba implementation of asymmetry calculation"""
    n = len(returns)
    rolling_skew = np.full(n, 0.0)
    rolling_kurt = np.full(n, 3.0)

    for i in range(window, n):
        window_returns = returns[i-window:i]

        # Calculate mean
        mean_ret = 0.0
        for j in range(len(window_returns)):
            mean_ret += window_returns[j]
        mean_ret /= len(window_returns)

        # Calculate standard deviation
        std_ret = 0.0
        for j in range(len(window_returns)):
            std_ret += (window_returns[j] - mean_ret) ** 2

        if len(window_returns) > 1:
            std_ret = np.sqrt(std_ret / (len(window_returns) - 1))

        if std_ret > 1e-9:
            # Calculate skewness
            skew_sum = 0.0
            for j in range(len(window_returns)):
                skew_sum += ((window_returns[j] - mean_ret) / std_ret) ** 3

            skew = skew_sum / len(window_returns)
            rolling_skew[i] = skew

            # Calculate kurtosis
            kurt_sum = 0.0
            for j in range(len(window_returns)):
                kurt_sum += ((window_returns[j] - mean_ret) / std_ret) ** 4

            kurt = kurt_sum / len(window_returns)
            rolling_kurt[i] = kurt

    # Weight skewness by volatility regime
    weighted_skew = np.zeros(n)
    for i in range(n):
        weighted_skew[i] = rolling_skew[i] * vol_regime[i]

    return weighted_skew, rolling_kurt

@njit(cache=True)
def _calc_robust_volatility_impl(high, low, close, open_price, overnight_returns, volume_period):
    """Numba implementation of volatility calculation components"""
    n = len(close)
    # Initialize arrays
    yz_variance = np.zeros(n)
    rs_term = np.zeros(n)

    # Calculate intraday volatility component (Rogers-Satchell)
    for i in range(n):
        if high[i] > 0 and low[i] > 0 and open_price[i] > 0 and close[i] > 0:
            log_hl = np.log(high[i] / low[i])
            log_co = np.log(close[i] / open_price[i])
            rs_term[i] = log_hl * (log_hl - log_co) / 2

    # Calculate rolling means for overnight and intraday
    for i in range(volume_period, n):
        # Rogers-Satchell rolling mean
        rs_sum = 0.0
        count = 0
        for j in range(i - volume_period, i):
            if rs_term[j] >= 0:  # Skip negative values (should be rare but possible due to precision errors)
                rs_sum += rs_term[j]
                count += 1

        intraday_var = rs_sum / max(1, count)

        # Overnight variance
        if i >= volume_period + 1:  # Need at least 1 previous value for overnight
            overnight_sum = 0.0
            count = 0
            for j in range(i - volume_period, i):
                # Use the overnight returns array
                if j < len(overnight_returns):
                    overnight_val = overnight_returns[j]
                    overnight_sum += overnight_val * overnight_val
                    count += 1

            overnight_var = overnight_sum / max(1, count)

            # Yang-Zhang k parameter
            k = 0.34 / (1.34 + (volume_period + 1) / (volume_period - 1)) if volume_period > 1 else 0.34

            # Combine into YZ variance
            yz_variance[i] = overnight_var + k * intraday_var

    return yz_variance, rs_term

@njit(cache=True)
def _garch_volatility_impl(returns, alpha_base):
    """Numba implementation of GARCH-like volatility calculation"""
    n = len(returns)
    if n <= 1:
        return np.zeros(n)

    # Calculate initial variance
    var_sum = 0.0
    for i in range(n):
        var_sum += returns[i] * returns[i]

    init_var = var_sum / n
    init_var = max(init_var, 1e-9)

    # Initialize GARCH variance
    garch_var = np.full(n, init_var)

    # Calculate standard deviation for dynamic alpha
    ret_std = 0.0
    for i in range(n):
        ret_std += (returns[i] - returns.mean()) ** 2
    ret_std = np.sqrt(ret_std / (n - 1)) if n > 1 else 1.0
    ret_std_safe = max(ret_std, 1e-9)

    # Calculate exponentially weighted recursive GARCH
    for i in range(1, n):
        # Dynamic alpha based on return magnitude
        ret_ratio = abs(returns[i-1]) / ret_std_safe
        ret_ratio = min(ret_ratio, 3.0)  # Clip for stability
        dynamic_alpha = alpha_base * (1 + ret_ratio)

        # Update variance
        garch_var[i] = dynamic_alpha * returns[i-1]**2 + (1 - dynamic_alpha) * garch_var[i-1]

    return np.sqrt(garch_var)

@njit(float64[:](float64[:], int64), cache=True)
def _calculate_rolling_zscore_numba(data: np.ndarray, window: int) -> np.ndarray:
    """
    Calculates rolling Z-score using Numba.

    Args:
        data (np.ndarray): Input 1D array (e.g., raw convexity values).
        window (int): Rolling window size.

    Returns:
        np.ndarray: Array of rolling Z-scores (clipped), same length as input.
                    Defaults to 0.0 for initial points or where std dev is zero.
    """
    n = len(data)
    zscores = np.full(n, 0.0, dtype=np.float64) # Initialize with neutral 0.0

    if n < window:
        return zscores # Not enough data for a single window

    # Loop through the data starting from the first point where a full window is available
    for i in range(window - 1, n):
        # Define the window slice [inclusive:exclusive]
        start_idx = i - window + 1
        end_idx = i + 1
        window_data = data[start_idx:end_idx]

        # Filter out NaNs/Infs within the window for robust stats
        finite_window = window_data[np.isfinite(window_data)]

        # Need at least 2 finite points to calculate standard deviation
        if len(finite_window) > 1:
            mean = np.mean(finite_window)
            std_dev = np.std(finite_window)

            # Get the current value to normalize (the last one in the window slice)
            current_val = data[i]

            if np.isfinite(current_val):
                if std_dev > 1e-9: # Check for non-zero standard deviation
                    z = (current_val - mean) / std_dev
                    # Clip the Z-score to a reasonable range (-5 to 5)
                    zscores[i] = max(-5.0, min(5.0, z))
                # else: zscores[i] remains 0.0 (neutral for zero std dev)
            # else: zscores[i] remains 0.0 (neutral if current value is NaN/Inf)
        # else: zscores[i] remains 0.0 (neutral if not enough finite points in window)

    return zscores

class AntifragilityAnalyzer:
    """
    Optimized Antifragility analyzer based on Taleb's antifragility concept.

    This class implements vectorized calculations with caching and JIT
    for measuring how systems benefit from volatility and stress.
    """

    def __init__(self,
                 cache_size: int = 100,
                 use_jit: bool = True,
                 params: Optional[AntifragilityParameters] = None,
                 log_level: str = "INFO"):
        """
        Initialize AntifragilityAnalyzer with caching and JIT optimization.

        Args:
            cache_size: Size of the LRU cache for expensive calculations
            use_jit: Whether to use JIT compilation via PennyLane Catalyst
            params: Optional custom parameters for Antifragility calculations
            log_level: Logging level (default: INFO)
        """
        self.cache_size = cache_size
        self.use_jit = use_jit and CATALYST_AVAILABLE
        self.params = params or AntifragilityParameters()

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

        # Initialize result cache
        self._calculation_cache = {}

        # Log initialization status
        self.logger.info(f"Initialized AntifragilityAnalyzer (JIT: use_numba=True, Cache: {self.cache_size})")

    def _setup_cached_methods(self):
        """Setup method caching using LRU cache decorator"""
        # Apply caching to expensive calculations
        #self._cached_calc_robust_volatility = lru_cache(maxsize=self.cache_size)(self._calc_robust_volatility)

        # Keep backward compatibility while using Numba internally
        self._jit_convexity_calculation = self._convexity_calculation
        self._jit_asymmetry_calculation = self._asymmetry_calculation

    def _convexity_calculation(self, perf_acceleration: np.ndarray,
                              vol_roc_smoothed: np.ndarray,
                              window: int) -> np.ndarray:
        """
        Calculate convexity correlation between performance acceleration and volatility change.

        Convexity is a key measure of antifragility - systems that show performance
        acceleration during volatility increases exhibit convexity benefits.

        Args:
            perf_acceleration: Performance acceleration array
            vol_roc_smoothed: Smoothed volatility rate of change array
            window: Rolling window size for correlation

        Returns:
            Convexity correlation array
        """
        n = len(perf_acceleration)
        if n != len(vol_roc_smoothed) or n < window:
            return np.full(n, 0.0)

        try:
            # Use Numba implementation
            return _convexity_calculation_impl(perf_acceleration, vol_roc_smoothed, window)
        except Exception as e:
            self.logger.warning(f"Numba convexity calculation failed: {e}, falling back to Python")

            # Fallback to vectorized numpy implementation
            convexity = np.full(n, 0.0)

            # Use vectorized rolling correlation
            for i in range(window, n):
                perf_window = perf_acceleration[i-window:i]
                vol_window = vol_roc_smoothed[i-window:i]

                # Calculate correlation if both arrays have variation
                perf_std = np.std(perf_window)
                vol_std = np.std(vol_window)

                if perf_std > 1e-9 and vol_std > 1e-9:
                    perf_mean = np.mean(perf_window)
                    vol_mean = np.mean(vol_window)

                    # Covariance
                    cov = np.mean((perf_window - perf_mean) * (vol_window - vol_mean))

                    # Correlation
                    corr = cov / (perf_std * vol_std)
                    convexity[i] = corr

            return convexity

    def _asymmetry_calculation(self, returns: np.ndarray, window: int, vol_regime: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate asymmetry metrics (skewness and kurtosis) with volatility weighting.

        Args:
            returns: Returns array
            window: Rolling window size
            vol_regime: Volatility regime array for weighting

        Returns:
            Tuple of (weighted skewness, kurtosis)
        """
        n = len(returns)
        if n < window or n != len(vol_regime):
            return np.full(n, 0.0), np.full(n, 3.0)

        try:
            # Use Numba implementation
            return _asymmetry_calculation_impl(returns, window, vol_regime)
        except Exception as e:
            self.logger.warning(f"Numba asymmetry calculation failed: {e}, falling back to Python")

            # Fallback to vectorized numpy implementation
            rolling_skew = np.full(n, 0.0)
            rolling_kurt = np.full(n, 3.0)

            # Calculate rolling skewness and kurtosis
            for i in range(window, n):
                window_returns = returns[i-window:i]

                # Calculate skewness
                mean_ret = np.mean(window_returns)
                std_ret = np.std(window_returns)

                if std_ret > 1e-9:
                    # Skewness
                    skew_sum = np.sum(((window_returns - mean_ret) / std_ret) ** 3)
                    skew = skew_sum / len(window_returns)
                    rolling_skew[i] = skew

                    # Kurtosis
                    kurt_sum = np.sum(((window_returns - mean_ret) / std_ret) ** 4)
                    kurt = kurt_sum / len(window_returns)
                    rolling_kurt[i] = kurt

            # Weight skewness by volatility regime
            weighted_skew = rolling_skew * vol_regime

            return weighted_skew, rolling_kurt


    # === NEW PUBLIC METHOD ===
    def calculate_convexity(self, dataframe: pd.DataFrame, period: int = 30) -> pd.Series:
        """
        Calculates the price convexity series (0-1), using Numba for optimization.
        Provides the public interface expected by external callers (like QStar).

        Args:
            dataframe (pd.DataFrame): Input DataFrame with 'close'.
            period (int): Rolling window period for Z-score normalization.

        Returns:
            pd.Series: Convexity score (0-1), aligned with input DataFrame index.
                       Defaults to 0.5 on error or insufficient data.
        """
        col_name = 'convexity' # The output column name expected by QStar
        default_series = pd.Series(0.5, index=dataframe.index)

        if 'close' not in dataframe.columns or dataframe['close'].isnull().all():
            self.logger.warning(f"Cannot calculate {col_name}: 'close' column missing or all NaN.")
            return default_series

        # Need period for rolling window + 1 for diff + 1 for pct_change
        if len(dataframe) < period + 2:
            self.logger.debug(f"Not enough data for {col_name} ({len(dataframe)} < {period+2}). Returning default.")
            return default_series

        try:
            # 1. Calculate Returns
            returns = dataframe['close'].pct_change(1)
            # 2. Calculate Raw Convexity (change in returns)
            # Fill initial NaN from diff with 0, handle potential NaNs in returns
            convexity_raw = returns.diff(1).fillna(0.0)

            # 3. Get NumPy array and handle potential non-finite values BEFORE numba
            convexity_values = convexity_raw.replace([np.inf, -np.inf], 0.0).fillna(0.0).values

            # 4. Call the Numba helper for rolling Z-score
            # (Assuming _calculate_rolling_zscore_numba helper exists as defined previously)
            # If the helper doesn't exist in this file, copy it here or use the Pandas apply method from previous answers.
            # For this example, assume _calculate_rolling_zscore_numba is available:
            if hasattr(self, '_calculate_rolling_zscore_numba') and callable(getattr(self, '_calculate_rolling_zscore_numba', None)):
                 # Use internal helper if it exists (hypothetical)
                 convexity_normalized_np = self._calculate_rolling_zscore_numba(convexity_values, period)
            elif 'numba' in sys.modules and hasattr(sys.modules[__name__], '_calculate_rolling_zscore_numba'):
                 # Use module-level helper if it exists
                 convexity_normalized_np = _calculate_rolling_zscore_numba(convexity_values, period)
            else:
                 # Fallback to Pandas apply if Numba helper isn't found (copy the safe_normalize lambda here)
                 self.logger.warning("Numba helper _calculate_rolling_zscore_numba not found, using Pandas apply for convexity.")
                 def safe_normalize(x_window: pd.Series) -> float:
                     std_val = x_window.std()
                     if pd.notna(std_val) and std_val > 1e-9:
                         mean_val = x_window.mean()
                         last_val = x_window.iloc[-1]
                         if pd.notna(last_val) and pd.notna(mean_val):
                              normalized = (last_val - mean_val) / std_val
                              return np.clip(normalized, -5.0, 5.0)
                         else: return 0.0
                     else: return 0.0
                 convexity_normalized = convexity_raw.rolling(window=period).apply(safe_normalize, raw=False)
                 convexity_normalized_np = convexity_normalized.ffill().bfill().fillna(0.0).values # Get numpy array

            # 5. Apply tanh transformation and scale to 0-1
            convexity_index_np = 0.5 * (np.tanh(convexity_normalized_np) + 1)

            # 6. Convert back to Pandas Series with original index
            convexity_index = pd.Series(convexity_index_np, index=dataframe.index)

            # 7. Final fillna for safety
            convexity_index = convexity_index.fillna(0.5)

            self.logger.debug(f"Calculated public convexity series. Last value: {convexity_index.iloc[-1]:.4f}")
            return convexity_index

        except Exception as e:
            self.logger.error(f"Error calculating public {col_name} series: {e}", exc_info=True)
            # Return the default neutral series on any error
            return default_series


    def calculate_fragility_score(self, dataframe: pd.DataFrame) -> Tuple[pd.Series, bool]:
        """
        Calculate fragility score with validation status.
        
        Args:
            dataframe: Input dataframe
            
        Returns:
            Tuple of (fragility Series, success flag)
        """
        # Create default result
        default_result = pd.Series(0.5, index=dataframe.index)
        
        try:
            # Validate required columns
            if 'convexity' not in dataframe.columns:
                self.logger.error("Cannot calculate fragility: 'convexity' column missing")
                return default_result, False
                
            # Validate that convexity has valid values
            if dataframe['convexity'].isna().all():
                self.logger.error("Cannot calculate fragility: 'convexity' column contains all NaN values")
                return default_result, False
                
            # Get volatility with detailed error handling
            try:
                volatility_result = self.calculate_robust_volatility(dataframe)
                vol_regime = volatility_result['vol_regime']
                
                # Validate volatility results
                if vol_regime.isna().all():
                    self.logger.error("Cannot calculate fragility: volatility calculation returned all NaN values")
                    return default_result, False
                    
            except Exception as e:
                self.logger.error(f"Cannot calculate fragility: volatility calculation failed - {e}")
                return default_result, False
                
            # Get antifragility with detailed error handling
            try:
                antifragility = self.calculate_antifragility_index(dataframe)
                
                # Validate antifragility results
                if antifragility.isna().all():
                    self.logger.error("Cannot calculate fragility: antifragility calculation returned all NaN values")
                    return default_result, False
                    
            except Exception as e:
                self.logger.error(f"Cannot calculate fragility: antifragility calculation failed - {e}")
                return default_result, False
                
            # At this point, all components are valid
            convexity = dataframe['convexity']
            
            # Calculate fragility with explicit NaN handling
            fragility = (1 - antifragility) * (0.7 + 0.3 * vol_regime) * (0.7 + 0.3 * (1 - convexity))
            
            # Handle any NaN values created during calculation
            fragility = fragility.fillna(0.5)
            
            # Clip to valid range
            fragility = np.clip(fragility, 0, 1)
            
            return fragility, True
            
        except Exception as e:
            self.logger.error(f"Fragility calculation error: {str(e)}", exc_info=True)
            return default_result, False

    def calculate_robust_volatility(self, dataframe: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        Calculate multiple volatility estimators and combine them robustly into a
        'vol_regime' score (0-1 scale) and a 'combined_vol' estimate.

        Args:
            dataframe: Input dataframe with OHLCV data.
            period: Lookback period for volatility calculation.

        Returns:
            pd.DataFrame: DataFrame indexed like the input, containing calculated
                          'combined_vol' and 'vol_regime' columns. Returns default
                          values on error or insufficient data.
        """
        self.logger.debug(f"Calculating robust volatility. Period={period}, Input len={len(dataframe)}")
        # Initialize output dataframe with defaults
        # Use float64 for consistency
        vol_metrics = pd.DataFrame(index=dataframe.index)
        vol_metrics['combined_vol'] = 0.0
        vol_metrics['vol_regime'] = 0.5 # Default to neutral

        component_status = {
            'yz_volatility': False,
            'atr_volatility': False,
            'garch_volatility': False,
            'parkinson_volatility': False
        }
        
        # Validate required columns
        required_cols = ['high', 'low', 'close', 'open']
        if not all(col in dataframe.columns for col in required_cols):
            missing = [col for col in required_cols if col not in dataframe.columns]
            self.logger.error(f"Missing required columns for volatility calculation: {missing}")
            return vol_metrics, {'success': False, **component_status}
        
        # Validate data length
        if len(dataframe) < period + 5:
            self.logger.warning(f"Insufficient data for volatility calculation: {len(dataframe)} rows < {period + 5} required")
            return vol_metrics, {'success': False, **component_status}
        
        try:
            # Check if we have enough data (need more than period for rolling calcs)
            if len(dataframe) < period + 5: # Added buffer
                self.logger.warning(f"Insufficient data ({len(dataframe)} < {period + 5}) for robust volatility calculation.")
                return vol_metrics # Return dataframe with defaults

            # --- Check Cache ---
            # Generate a reasonably stable cache key
            try:
                # Use last 2*period rows hash, length, and period for key
                hash_data = dataframe[['open', 'high', 'low', 'close']].iloc[-min(len(dataframe), 2*period):].to_numpy().tobytes()
                cache_key = f"vol_{period}_{len(dataframe)}_{hash(hash_data)}"
            except Exception as e_hash:
                 self.logger.warning(f"Could not generate cache key: {e_hash}")
                 cache_key = None # Proceed without cache if key fails

            if cache_key and cache_key in self._calculation_cache:
                self.logger.debug(f"Using cached volatility calculation for key: {cache_key[:10]}...")
                return self._calculation_cache[cache_key]
            # --- End Cache Check ---


            # --- Prepare Data ---
            required_cols = ['close', 'high', 'low', 'open']
            if not all(col in dataframe.columns for col in required_cols):
                self.logger.error(f"Missing required OHLC columns for volatility calculation.")
                return vol_metrics # Return dataframe with defaults

            # Use .values for Numba compatibility and potential speedup
            high = dataframe['high'].values.astype(np.float64)
            low = dataframe['low'].values.astype(np.float64)
            close = dataframe['close'].values.astype(np.float64)
            open_price = dataframe['open'].values.astype(np.float64)
            n = len(close)

            # --- Calculate Volatility Components ---

            # 1. Yang-Zhang (YZ) Volatility Calculation
            yz_variance = np.full(n, 0.0, dtype=np.float64) # Initialize
            try:
                overnight_returns = np.zeros(n, dtype=np.float64)
                if n > 1:
                    safe_open = np.maximum(open_price[1:], 1e-12) # Increased precision slightly
                    safe_prev_close = np.maximum(close[:-1], 1e-12)
                    overnight_returns[1:] = np.log(safe_open / safe_prev_close)
                    overnight_returns = np.nan_to_num(overnight_returns) # Ensure no NaNs

                yz_variance_calc, _ = _calc_robust_volatility_impl(high, low, close, open_price, overnight_returns, period)
                yz_variance = yz_variance_calc # Assign calculated value
                self.logger.debug(f"YZ variance calculated. Sample non-zero values: {yz_variance[yz_variance > 0][:5]}")
            except Exception as e_yz:
                self.logger.error(f"Error during Numba YZ volatility calculation: {e_yz}", exc_info=True)
                # Keep yz_variance as zeros if Numba failed

            yz_vol = np.sqrt(np.maximum(yz_variance, 0)) # Ensure non-negative before sqrt
            yz_vol_series = pd.Series(yz_vol, index=dataframe.index)

            # 2. ATR-based Volatility
            norm_atr = pd.Series(0.0, index=dataframe.index) # Initialize
            try:
                if TALIB_AVAILABLE and ta is not None:
                    atr = ta.ATR(dataframe, timeperiod=period)
                else: # Manual ATR if TA-Lib not available
                    tr1 = dataframe['high'] - dataframe['low']
                    tr2 = (dataframe['high'] - dataframe['close'].shift(1)).abs()
                    tr3 = (dataframe['low'] - dataframe['close'].shift(1)).abs()
                    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                    atr = tr.ewm(span=period, adjust=False).mean()

                norm_atr = (atr / dataframe['close'].replace(0, np.nan)).fillna(0) # Normalize by close price
                self.logger.debug(f"Normalized ATR calculated. Last value: {norm_atr.iloc[-1]:.6f}")
            except Exception as e_atr:
                self.logger.error(f"ATR calculation failed: {e_atr}", exc_info=True)


            # 3. GARCH-like Volatility
            garch_vol = pd.Series(0.0, index=dataframe.index) # Initialize
            try:
                returns = dataframe['close'].pct_change().fillna(0).values.astype(np.float64)
                # Winsorize extreme returns for stability
                returns_ql, returns_qh = np.percentile(returns, [1, 99]) if len(returns) > 2 else (-0.1, 0.1)
                returns_clipped = np.clip(returns, returns_ql, returns_qh)

                garch_vol_calc = _garch_volatility_impl(returns_clipped, self.params.garch_alpha_base)
                garch_vol = pd.Series(garch_vol_calc, index=dataframe.index)
                self.logger.debug(f"GARCH Volatility calculated. Last value: {garch_vol.iloc[-1]:.6f}")
            except Exception as e_garch:
                self.logger.error(f"GARCH volatility calculation failed: {e_garch}", exc_info=True)


            # 4. Parkinson Volatility (Optional, can be noisy)
            parkinson_vol = pd.Series(0.0, index=dataframe.index) # Initialize
            try:
                 high_low_ratio = np.log(np.maximum(1e-9, high) / np.maximum(1e-9, low))**2
                 park_vol_daily_sq = high_low_ratio / (4 * np.log(2)) # Parkinson number squared daily
                 park_variance = pd.Series(park_vol_daily_sq, index=dataframe.index).rolling(window=period).mean()
                 parkinson_vol = np.sqrt(park_variance.fillna(0)) * np.sqrt(252) # Annualize roughly
                 self.logger.debug(f"Parkinson Volatility calculated. Last value: {parkinson_vol.iloc[-1]:.6f}")
            except Exception as e_park:
                 self.logger.error(f"Parkinson volatility calculation failed: {e_park}", exc_info=True)


            # --- Combine Estimators into Volatility Regime Score (0-1) ---
            long_period = int(period * self.params.vol_lookback_factor)
            min_periods_long = max(10, long_period // 2) # Min periods for historical average

            if len(dataframe) < long_period:
                 self.logger.warning(f"Not enough data ({len(dataframe)} < {long_period}) for long-period volatility normalization. Regime might be less reliable.")
                 # Use shorter history if necessary, or skip normalization
                 hist_atr = norm_atr.rolling(window=period*2, min_periods=period).mean().replace(0, np.nan)
                 hist_garch = garch_vol.rolling(window=period*2, min_periods=period).mean().replace(0, np.nan)
                 hist_yz = yz_vol_series.rolling(window=period*2, min_periods=period).mean().replace(0, np.nan)
                 hist_park = parkinson_vol.rolling(window=period*2, min_periods=period).mean().replace(0, np.nan)
            else:
                hist_atr = norm_atr.rolling(window=long_period, min_periods=min_periods_long).mean().replace(0, np.nan)
                hist_garch = garch_vol.rolling(window=long_period, min_periods=min_periods_long).mean().replace(0, np.nan)
                hist_yz = yz_vol_series.rolling(window=long_period, min_periods=min_periods_long).mean().replace(0, np.nan)
                hist_park = parkinson_vol.rolling(window=long_period, min_periods=min_periods_long).mean().replace(0, np.nan)

            # Calculate normalized volatility regimes relative to history
            # Use .bfill() to propagate last known historical average forward for initial NaNs
            regime_atr = (norm_atr / hist_atr.bfill()).fillna(1.0) # Default to 1 if history is NaN
            regime_garch = (garch_vol / hist_garch.bfill()).fillna(1.0)
            regime_yz = (yz_vol_series / hist_yz.bfill()).fillna(1.0)
            regime_park = (parkinson_vol / hist_park.bfill()).fillna(1.0)

            # Create weights based on which estimators have valid (non-default) data
            # Give slightly higher weight to YZ and GARCH? (Example weights)
            weights = pd.DataFrame({
                'atr': (~pd.isna(norm_atr) & (norm_atr > 1e-9)).astype(float) * 0.20,
                'garch': (~pd.isna(garch_vol) & (garch_vol > 1e-9)).astype(float) * 0.30,
                'yz': (~pd.isna(yz_vol_series) & (yz_vol_series > 1e-9)).astype(float) * 0.35,
                'park': (~pd.isna(parkinson_vol) & (parkinson_vol > 1e-9)).astype(float) * 0.15
            }, index=dataframe.index)

            # Normalize weights so they sum to 1 (or 0 if all inputs are invalid)
            sum_weights = weights.sum(axis=1).replace(0, 1.0) # Avoid division by zero
            weights = weights.div(sum_weights, axis=0)

            # Weighted average of regime indicators
            combined_regime_raw = (
                regime_atr * weights['atr'] +
                regime_garch * weights['garch'] +
                regime_yz * weights['yz'] +
                regime_park * weights['park']
            )

            # Handle cases where all weights were zero
            combined_regime = combined_regime_raw.fillna(1.0) # Default to regime 1.0 if all failed

            # Transform to 0-1 scale using log and clipping
            # This maps: Ratio=1 -> 0.5, Ratio=e^1 -> 0.75, Ratio=e^2 -> 1.0 (clipped)
            #            Ratio=e^-1 -> 0.25, Ratio=e^-2 -> 0.0 (clipped)
            log_regime = np.log(combined_regime.replace(0, 1e-9)) # Avoid log(0)
            regime_score = (0.5 + (log_regime / 2.0)).clip(0, 1)

            # --- Assign final calculated values to output DataFrame ---
            # Use YZ as the primary quantitative volatility estimate
            vol_metrics['combined_vol'] = yz_vol_series.fillna(0.0)
            vol_metrics['vol_regime'] = regime_score.fillna(0.5) # Final score, default 0.5

            # --- Store in cache ---
            if cache_key:
                self._calculation_cache[cache_key] = vol_metrics.copy() # Cache a copy
            # --------------------

            self.logger.info(f"Robust volatility calculated. Last combined_vol={vol_metrics['combined_vol'].iloc[-1]:.6f}, vol_regime={vol_metrics['vol_regime'].iloc[-1]:.4f}")
            return vol_metrics

        # --- Outer Exception Handling ---
        except Exception as e:
            self.logger.error(f"Critical Error in calculate_robust_volatility: {e}", exc_info=True)
            # Return dataframe with defaults initialized at the start
            overall_success = any(component_status.values())
            
            if not overall_success:
                self.logger.warning("No volatility components succeeded")
                
            return vol_metrics, {'success': overall_success, **component_status}
        
    def calculate_antifragility_index(self,
                                     dataframe: pd.DataFrame,
                                     vol_period: int = 21,
                                     perf_period: int = 63,
                                     corr_window: int = 42,
                                     smoothing_span: int = 10) -> pd.Series:
        """
        Calculate Antifragility Index based on Taleb's concepts.

        Higher values (closer to 1) indicate more antifragility (benefit from volatility).
        Values around 0.5 indicate robustness (neutral to volatility).
        Values closer to 0 indicate fragility (harmed by volatility).

        Args:
            dataframe: Input dataframe with OHLCV data
            vol_period: Lookback period for volatility calculation
            perf_period: Lookback period for performance assessment
            corr_window: Rolling window for correlations
            smoothing_span: EWMA smoothing span for the final index

        Returns:
            Pandas Series with Antifragility Index values
        """
        try:
            # Generate cache key
            cache_key = f"antifragility_{vol_period}_{perf_period}_{corr_window}_{smoothing_span}_{len(dataframe)}_{hash(dataframe['close'].iloc[-perf_period:].to_numpy().tobytes())}"
            if cache_key in self._calculation_cache:
                self.logger.debug(f"Using cached antifragility calculation")
                return self._calculation_cache[cache_key]

            # Check if we have enough data
            if len(dataframe) < max(vol_period, perf_period, corr_window) + 5:
                self.logger.warning(f"Insufficient data for antifragility calculation ({len(dataframe)} candles)")
                return pd.Series(0.5, index=dataframe.index)

            # 1. Calculate robust volatility metrics
            vol_metrics = self.calculate_robust_volatility(dataframe, period=vol_period)
            volatility = vol_metrics['combined_vol']
            vol_regime = vol_metrics['vol_regime']

            # 2. Calculate volatility rate of change
            vol_roc = vol_regime.pct_change().fillna(0)
            vol_roc_smoothed = vol_roc.ewm(span=vol_period//3, min_periods=5).mean().fillna(0)

            # 3. Calculate performance and acceleration
            # Use log returns for better statistical properties
            log_perf_returns = np.log(dataframe['close'].div(dataframe['close'].shift(perf_period))).fillna(0)

            # Performance momentum and acceleration
            perf_momentum = log_perf_returns.diff().fillna(0)
            perf_acceleration = perf_momentum.diff().ewm(span=perf_period//3, min_periods=5).mean().fillna(0)

            # 4. Component A: Convexity - Performance Acceleration vs Volatility Change
            # Does performance accelerate when volatility increases?
            convexity_corr = self._jit_convexity_calculation(
                perf_acceleration.values,
                vol_roc_smoothed.values,
                corr_window
            )

            # Convert back to Series
            convexity_corr_series = pd.Series(convexity_corr, index=dataframe.index)

            # Scale correlation to [0, 1]
            comp_a_convexity = (convexity_corr_series + 1) / 2

            # 5. Component B: Asymmetry - Skewness & Kurtosis under Stress
            log_returns_daily = np.log(dataframe['close'].div(dataframe['close'].shift(1))).fillna(0)

            # Calculate weighted skewness and kurtosis
            weighted_skew, rolling_kurt = self._jit_asymmetry_calculation(
                log_returns_daily.values,
                perf_period,
                vol_regime.values
            )

            # Convert back to Series
            weighted_skew_series = pd.Series(weighted_skew, index=dataframe.index)
            rolling_kurt_series = pd.Series(rolling_kurt, index=dataframe.index)

            # Calculate asymmetry score
            favorable_tail_profile = (np.tanh(weighted_skew_series) + 1) / 2

            # Penalize excessive kurtosis during stress
            kurt_penalty_factor = np.maximum(
                0,
                1 - np.maximum(0, (rolling_kurt_series - 5)) * 0.2 *
                np.maximum(1, vol_regime) * (1 - favorable_tail_profile)
            )

            comp_b_asymmetry = (favorable_tail_profile * kurt_penalty_factor).clip(0, 1)

            # 6. Component C: Recovery Velocity - Performance after Volatility Spikes
            # How well does performance fare *after* a period of increased volatility?
            future_perf_horizon = int(vol_period * self.params.recovery_horizon_factor)
            future_log_perf = np.log(
                dataframe['close'].shift(-future_perf_horizon) / dataframe['close']
            ).fillna(0)

            # Correlation between current volatility *change* and *subsequent* performance
            recovery_corr = future_log_perf.rolling(window=corr_window, min_periods=corr_window//2) \
                                        .corr(vol_roc_smoothed) \
                                        .fillna(0)

            comp_c_recovery = (recovery_corr + 1) / 2

            # 7. Component D: Volatility Benefit Ratio
            # Simplified check: Did performance rate improve more than volatility rate increased?
            perf_roc_smoothed = log_perf_returns.pct_change().ewm(span=perf_period//3, min_periods=5).mean().fillna(0)

            # Ratio - handle potential zeros in denominator
            vol_roc_abs_min = 1e-6
            benefit_ratio_raw = perf_roc_smoothed / vol_roc_smoothed.replace(0, np.nan)
            benefit_ratio_raw = benefit_ratio_raw.fillna(0)

            # Transform to 0-1 range
            comp_d_benefit_ratio = (np.tanh(benefit_ratio_raw * 0.5) + 1) / 2

            # 8. Combine Components with Weights from parameters
            weights = {
                'convexity': self.params.convexity_weight,
                'asymmetry': self.params.asymmetry_weight,
                'recovery': self.params.recovery_weight,
                'benefit_ratio': self.params.benefit_ratio_weight
            }

            antifragility_raw = (
                comp_a_convexity * weights['convexity'] +
                comp_b_asymmetry * weights['asymmetry'] +
                comp_c_recovery * weights['recovery'] +
                comp_d_benefit_ratio * weights['benefit_ratio']
            )

            # 9. Final Smoothing
            antifragility_index = antifragility_raw.ewm(span=smoothing_span, min_periods=smoothing_span//2).mean()

            # Ensure output is in [0, 1] range
            result = antifragility_index.fillna(0.5).clip(0, 1)

            # Store in cache
            self._calculation_cache[cache_key] = result

            self.logger.info(f"Antifragility Index calculated. Last value: {result.iloc[-1]:.4f}")

            return result

        except Exception as e:
            self.logger.error(f"Error calculating Antifragility Index: {e}", exc_info=True)
            return pd.Series(0.5, index=dataframe.index)

    def analyze(self, prices: np.ndarray, volumes: np.ndarray) -> Dict[str, Any]:
        """
        Analyze price and volume data to generate antifragility signal.
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
            
            # Calculate antifragility index
            antifragility_series = self.calculate_antifragility_index(df)
            
            # Get latest value as signal
            latest_antifragility = antifragility_series.iloc[-1] if len(antifragility_series) > 0 else 0.5
            
            # Calculate confidence based on data stability
            if len(antifragility_series) >= 10:
                recent_values = antifragility_series.tail(10)
                confidence = 1.0 - np.std(recent_values)  # Lower std = higher confidence
                confidence = max(0.1, min(1.0, confidence))  # Clamp to reasonable range
            else:
                confidence = 0.5  # Default confidence for insufficient data
            
            return {
                "signal": float(latest_antifragility),
                "confidence": float(confidence),
                "antifragility_score": float(latest_antifragility),
                "analysis_type": "antifragility",
                "data_points": len(prices)
            }
            
        except Exception as e:
            self.logger.error(f"Error in analyze method: {e}")
            return {
                "signal": 0.5,
                "confidence": 0.0,
                "error": str(e),
                "analysis_type": "antifragility"
            }
