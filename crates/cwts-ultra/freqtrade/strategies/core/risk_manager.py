# Advanced Risk Management
import logging
import config
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import math
import threading
import scipy
from scipy import stats
import scaler
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from collections import deque
from dataclasses import dataclass, field
import hashlib

# --- Numba Import ---
try:
    import numba as nb
    from numba import njit, float64, int64, boolean
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):
        def decorator(func): return func
        return decorator if args and callable(args[0]) else decorator
    boolean = bool # Fallback type alias
    
from cache_manager import CircularBuffer
try:
    from whale_detector import WhaleDetector, WhaleParameters # Assuming these exist
except ImportError:
    WhaleDetector = None; WhaleParameters = None
    logging.warning("WhaleDetector class not found.")
try:
    from black_swan_detector import BlackSwanDetector, BlackSwanParameters # Assuming these exist
except ImportError:
    BlackSwanDetector = None; BlackSwanParameters = None
    logging.warning("BlackSwanDetector class not found.")
    
logger = logging.getLogger("risk_manager_unified")
logger_via_neg = logging.getLogger("risk_manager_unified.via_negativa")
logger_anomaly = logging.getLogger("risk_manager_unified.anomaly_detector")
if not logger.hasHandlers():
     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
     


@dataclass
class AnomalyDetectorParams:
    # ... (keep fields as defined before) ...
    volume_z_score_threshold: float = 2.5
    # ... etc ...
    bs_window_size: int = 100
    use_ml_anomaly: bool = False
    ml_contamination: float = 0.03
    ml_n_estimators: int = 100
    

logger = logging.getLogger("antifragile_risk_manager")


class RiskManager:
    """
    Advanced risk management with dynamic position sizing, volatility adaptive models,
    and comprehensive drawdown protection.

    Implements sophisticated risk metrics including conditional Value-at-Risk (CVaR),
    expected shortfall, and advanced position sizing strategies based on market conditions.
    """

    def __init__(
        self,
        confidence_level: float = 0.95,
        max_position_pct: float = 0.25,
        min_position_pct: float = 0.01,
        max_drawdown: float = 0.15,
        rebalancing_frequency: int = 10,
    ):
        self.logger = logging.getLogger(__name__)  # Can use module logger too
        self.confidence_level = confidence_level
        self.max_position_pct = max_position_pct
        self.min_position_pct = min_position_pct
        self.max_drawdown = max_drawdown
        self.rebalancing_frequency = rebalancing_frequency

        self._lock = threading.RLock()

        # super().__init__(config)
        # Add required components
        # self.risk_manager = RiskManager(position_size=0.1)
        # Input validation with informative error messages
        self.logger.info(
            f"RiskManager initialized with confidence={confidence_level}, max_pos={max_position_pct}"
        )
        if not 0 < confidence_level < 1:
            logger.warning(
                f"Confidence level {confidence_level} outside valid range (0,1). Using default 0.95."
            )
            confidence_level = 0.95

        if max_position_pct <= 0 or max_position_pct > 1:
            logger.warning(
                f"Invalid max_position_pct {max_position_pct}. Must be between 0-1. Using default 0.25."
            )
            max_position_pct = 0.25

        if min_position_pct <= 0 or min_position_pct >= max_position_pct:
            logger.warning(
                f"Invalid min_position_pct {min_position_pct}. Using {max_position_pct / 10}."
            )
            min_position_pct = max_position_pct / 10

        if max_drawdown <= 0 or max_drawdown > 0.5:
            logger.warning(
                f"Invalid max_drawdown {max_drawdown}. Must be between 0-0.5. Using default 0.15."
            )
            max_drawdown = 0.15

        # Core risk parameters
        self.confidence_level = confidence_level
        self.max_position_pct = max_position_pct
        self.min_position_pct = min_position_pct
        self.max_drawdown = max_drawdown
        self.rebalancing_frequency = rebalancing_frequency

        # Adaptive parameters
        self.volatility_scaling = True
        self.correlation_adjustment = True
        self.market_impact_model = (
            "square_root"  # "linear", "square_root", or "power_law"
        )
        self.decay_factor = (
            0.94  # Exponential decay for historical data (higher = slower decay)
        )

        # Risk state tracking
        self.var_cache = {}  # Cache VaR calculations
        self.cvar_cache = {}  # Cache CVaR calculations
        self.position_history = []  # Track position sizing history
        self.current_drawdown = 0.0
        self.peak_balance = 0.0
        self.last_rebalance = 0
        self.trade_count = 0

        # Market regime tracking
        self.volatility_regime = "normal"  # "low", "normal", "high", "extreme"
        self.correlation_regime = "normal"  # "low", "normal", "high", "breakdown"
        self.risk_multipliers = {
            "volatility": {
                "low": 1.2,  # Increase sizing in low vol
                "normal": 1.0,  # Normal sizing
                "high": 0.7,  # Reduce in high vol
                "extreme": 0.4,  # Significantly reduce in extreme vol
            },
            "correlation": {
                "low": 1.1,  # Slightly increase in low correlation
                "normal": 1.0,  # Normal sizing
                "high": 0.8,  # Reduce in high correlation
                "breakdown": 0.5,  # Significantly reduce during correlation breakdown
            },
        }

        # Performance metrics
        self.win_count = 0
        self.loss_count = 0
        self.profit_sum = 0
        self.loss_sum = 0

        # Initialize with reasonable Kelly criterion parameters
        self.kelly_fraction = 0.5  # Half-Kelly for conservatism

        logger.info(
            f"Enhanced Risk Manager initialized with confidence: {confidence_level}, "
            f"position range: {min_position_pct}-{max_position_pct}, max drawdown: {max_drawdown}"
        )

    def calculate_var(
        self, returns: pd.Series, cache_key: str = None, method: str = "historical"
    ) -> float:
        """
        Calculate Value at Risk with multiple methodologies and robust error handling

        Args:
            returns: Historical returns series
            cache_key: Optional cache key
            method: VaR calculation method ("historical", "parametric", or "cornish_fisher")

        Returns:
            Value at Risk estimate
        """
        # Try cache first if cache_key provided
        if cache_key and cache_key in self.var_cache:
            # Check if cache is recent (< 1 hour old)
            cache_time, cached_var = self.var_cache[cache_key]
            if (time.time() - cache_time) < 3600:  # 1 hour in seconds
                return cached_var

        # Input validation
        if not isinstance(returns, pd.Series):
            logger.warning(
                "Invalid returns type for VaR calculation. Expected pandas Series."
            )
            return 0.05  # Conservative default

        if returns.empty:
            logger.warning("Empty returns series for VaR calculation.")
            return 0.05  # Conservative default

        # Calculate VaR with comprehensive error handling
        try:
            # Apply exponential weighting to recent returns
            if len(returns) > 20:
                weights = np.array([self.decay_factor**i for i in range(len(returns))])
                weights = weights / weights.sum()  # Normalize
                weighted_returns = returns.values * weights
            else:
                weighted_returns = returns.values

            # Remove outliers for more robust calculation
            q1 = np.percentile(weighted_returns, 1)
            q3 = np.percentile(weighted_returns, 99)
            iqr = q3 - q1
            filtered_returns = weighted_returns[
                (weighted_returns >= q1 - 3 * iqr) & (weighted_returns <= q3 + 3 * iqr)
            ]

            if len(filtered_returns) < 10:  # Need sufficient data
                logger.warning(
                    f"Insufficient data for robust VaR: {len(filtered_returns)} samples after filtering"
                )
                return 0.05  # Conservative default

            # Calculate VaR based on specified method
            var = None

            if method == "historical":
                # Historical simulation method
                var = np.abs(
                    np.percentile(filtered_returns, (1 - self.confidence_level) * 100)
                )

            elif method == "parametric":
                # Parametric (normal distribution) method
                mean = np.mean(filtered_returns)
                std = np.std(filtered_returns)
                z_score = stats.norm.ppf(1 - self.confidence_level)
                var = np.abs(mean + z_score * std)

            elif method == "cornish_fisher":
                # Cornish-Fisher expansion for non-normal returns
                try:
                    from scipy import stats

                    mean = np.mean(filtered_returns)
                    std = np.std(filtered_returns)
                    skew = stats.skew(filtered_returns)
                    kurt = stats.kurtosis(filtered_returns)

                    # Normal z-score
                    z = stats.norm.ppf(1 - self.confidence_level)

                    # Cornish-Fisher adjustment
                    z_cf = (
                        z
                        + (z**2 - 1) * skew / 6
                        + (z**3 - 3 * z) * kurt / 24
                        - (2 * z**3 - 5 * z) * skew**2 / 36
                    )

                    var = np.abs(mean + z_cf * std)
                except ImportError:
                    logger.warning(
                        "SciPy not available for Cornish-Fisher VaR, using historical method"
                    )
                    var = np.abs(
                        np.percentile(
                            filtered_returns, (1 - self.confidence_level) * 100
                        )
                    )
                except Exception as e:
                    logger.warning(
                        f"Error in Cornish-Fisher calculation: {e}, using historical method"
                    )
                    var = np.abs(
                        np.percentile(
                            filtered_returns, (1 - self.confidence_level) * 100
                        )
                    )
            else:
                logger.warning(f"Unknown VaR method: {method}, using historical method")
                var = np.abs(
                    np.percentile(filtered_returns, (1 - self.confidence_level) * 100)
                )

            # Ensure VaR is valid
            if var is None:
                logger.warning("VaR calculation failed, using default")
                var = 0.05

            # Sanity check on VaR value
            if var > 0.5:  # Unreasonably high VaR
                logger.warning(
                    f"Unusually high VaR calculated: {var:.4f}. Capping at 0.5"
                )
                var = 0.5
            elif var < 0.001:  # Unreasonably low VaR
                logger.warning(
                    f"Unusually low VaR calculated: {var:.4f}. Setting minimum 0.001"
                )
                var = 0.001

            # Cache result if cache_key provided
            if cache_key:
                self.var_cache[cache_key] = (time.time(), var)

            logger.info(
                f"Calculated VaR at {self.confidence_level * 100}%: {var:.4f} (method: {method})"
            )
            return var

        except Exception as e:
            logger.error(f"Error calculating VaR: {str(e)}")
            return 0.05  # Conservative default on error

    def calculate_cvar(self, returns: pd.Series, cache_key: str = None) -> float:
        """
        Calculate Conditional Value at Risk (Expected Shortfall)

        Args:
            returns: Historical returns series
            cache_key: Optional cache key

        Returns:
            CVaR estimate
        """
        # Try cache first if cache_key provided
        if cache_key and cache_key in self.cvar_cache:
            # Check if cache is recent (< 1 hour old)
            cache_time, cached_cvar = self.cvar_cache[cache_key]
            if (time.time() - cache_time) < 3600:  # 1 hour in seconds
                return cached_cvar

        # Input validation
        if not isinstance(returns, pd.Series):
            logger.warning(
                "Invalid returns type for CVaR calculation. Expected pandas Series."
            )
            return 0.07  # Conservative default

        if returns.empty:
            logger.warning("Empty returns series for CVaR calculation.")
            return 0.07  # Conservative default

        try:
            # Sort returns
            sorted_returns = np.sort(returns.values)

            # Find cutoff index based on confidence level
            cutoff_index = int(len(sorted_returns) * (1 - self.confidence_level))

            # Ensure we have at least one value
            if cutoff_index < 1:
                cutoff_index = 1

            # Calculate CVaR as average of worst returns
            cvar = np.abs(np.mean(sorted_returns[:cutoff_index]))

            # Sanity check on CVaR value
            if cvar > 0.5:  # Unreasonably high CVaR
                logger.warning(
                    f"Unusually high CVaR calculated: {cvar:.4f}. Capping at 0.5"
                )
                cvar = 0.5
            elif cvar < 0.001:  # Unreasonably low CVaR
                logger.warning(
                    f"Unusually low CVaR calculated: {cvar:.4f}. Setting minimum 0.001"
                )
                cvar = 0.001

            # Cache result if cache_key provided
            if cache_key:
                self.cvar_cache[cache_key] = (time.time(), cvar)

            logger.info(
                f"Calculated CVaR at {self.confidence_level * 100}%: {cvar:.4f}"
            )
            return cvar

        except Exception as e:
            logger.error(f"Error calculating CVaR: {str(e)}")
            return 0.07  # Conservative default on error

    # Add to RiskManager class
    def calculate_optimal_stoploss(
        self, entry_price, volatility, trend_strength, antifragility
    ):
        """Optimize stoploss based on market conditions for higher Sharpe ratio"""
        # Base stoploss calculation
        base_sl_pct = -0.05

        # Adjust for volatility - wider in low vol, tighter in high vol
        if volatility < 0.01:  # Low volatility
            vol_factor = 1.2  # Wider stoploss
        elif volatility > 0.02:  # High volatility
            vol_factor = 0.8  # Tighter stoploss
        else:
            vol_factor = 1.0

        # Adjust for trend - tighter in strong trends
        if abs(trend_strength) > 0.05:  # Strong trend
            trend_factor = 0.85
        else:
            trend_factor = 1.0

        # Adjust for antifragility - more aggressive with higher antifragility
        antifragility_factor = 0.8 + (antifragility * 0.3)  # Range from 0.8 to 1.1

        # Combined adjustment
        adjusted_sl_pct = base_sl_pct * vol_factor * trend_factor * antifragility_factor

        # Convert to price
        stoploss_price = entry_price * (1 + adjusted_sl_pct)

        return adjusted_sl_pct, stoploss_price

    def update_volatility_regime(self, returns: pd.Series):
        """
        Update volatility regime classification based on recent returns

        Args:
            returns: Historical returns series
        """
        if not isinstance(returns, pd.Series) or len(returns) < 20:
            return

        try:
            # Calculate recent volatility (standard deviation of returns)
            recent_vol = returns.tail(20).std()

            # Calculate longer-term volatility for comparison
            if len(returns) >= 60:
                baseline_vol = returns.tail(60).std()
            else:
                baseline_vol = recent_vol

            # Classify regime based on volatility level
            if recent_vol < 0.5 * baseline_vol:
                new_regime = "low"
            elif recent_vol < 1.5 * baseline_vol:
                new_regime = "normal"
            elif recent_vol < 2.5 * baseline_vol:
                new_regime = "high"
            else:
                new_regime = "extreme"

            # Log regime change
            if new_regime != self.volatility_regime:
                logger.info(
                    f"Volatility regime change: {self.volatility_regime} → {new_regime} "
                    f"(vol: {recent_vol:.4f}, baseline: {baseline_vol:.4f})"
                )
                self.volatility_regime = new_regime

        except Exception as e:
            logger.error(f"Error updating volatility regime: {str(e)}")

    def update_correlation_regime(self, returns_dict: Dict[str, pd.Series]):
        """
        Update correlation regime classification based on asset correlations

        Args:
            returns_dict: Dictionary of asset returns series
        """
        if not returns_dict or len(returns_dict) < 2:
            return

        try:
            # Create DataFrame from returns dictionary
            returns_df = pd.DataFrame(returns_dict)

            # Calculate correlation matrix
            corr_matrix = returns_df.corr()

            # Extract upper triangle of correlation matrix (excluding diagonal)
            upper_triangle = corr_matrix.values[
                np.triu_indices_from(corr_matrix.values, k=1)
            ]

            # Calculate average correlation
            avg_correlation = np.mean(upper_triangle)

            # Calculate correlation dispersion (standard deviation)
            corr_dispersion = np.std(upper_triangle)

            # Classify regime based on correlation level and dispersion
            if avg_correlation < 0.3:
                new_regime = "low"
            elif avg_correlation < 0.7:
                new_regime = "normal"
            elif corr_dispersion < 0.2:
                new_regime = "high"  # High and uniform correlations
            else:
                new_regime = (
                    "breakdown"  # High average but non-uniform (some extreme values)
                )

            # Log regime change
            if new_regime != self.correlation_regime:
                logger.info(
                    f"Correlation regime change: {self.correlation_regime} → {new_regime} "
                    f"(avg: {avg_correlation:.4f}, dispersion: {corr_dispersion:.4f})"
                )
                self.correlation_regime = new_regime

        except Exception as e:
            logger.error(f"Error updating correlation regime: {str(e)}")

    def calculate_kelly_fraction(self):
        """
        Calculate optimal Kelly fraction based on historical performance

        Returns:
            Kelly fraction between 0-1
        """
        try:
            # Need sufficient data
            if self.win_count + self.loss_count < 10:
                return 0.5  # Default half-Kelly

            # Calculate win rate
            win_rate = self.win_count / (self.win_count + self.loss_count)

            # Calculate average win and loss
            avg_win = self.profit_sum / max(1, self.win_count)
            avg_loss = abs(self.loss_sum) / max(1, self.loss_count)

            # Calculate Kelly fraction: f* = (bp - q) / b
            # where b = win/loss ratio, p = win probability, q = 1-p
            b = avg_win / max(0.001, avg_loss)  # Avoid division by zero
            p = win_rate
            q = 1 - p

            kelly = (b * p - q) / b

            # Limit Kelly to reasonable range
            kelly = max(0.0, min(1.0, kelly))

            # Apply half-Kelly for conservatism
            half_kelly = kelly * 0.5

            logger.info(
                f"Kelly fraction: {kelly:.4f} (half-Kelly: {half_kelly:.4f}), "
                f"win rate: {win_rate:.4f}, avg win/loss ratio: {b:.4f}"
            )

            return half_kelly

        except Exception as e:
            logger.error(f"Error calculating Kelly fraction: {str(e)}")
            return 0.5  # Conservative default

    def dynamic_position_sizing(
        self,
        current_balance: float,
        returns: pd.Series,
        risk_factor: float = 0.01,
        pair: str = None,
        market_info: dict = None,
    ) -> float:
        """
        Calculate optimal position size with comprehensive risk management

        Args:
            current_balance: Available capital
            returns: Historical returns for the asset
            risk_factor: Base risk factor (0-0.05)
            pair: Trading pair name
            market_info: Additional market information

        Returns:
            Optimal position size
        """
        # Track trades for rebalancing
        self.trade_count += 1

        # Update peak balance for drawdown tracking
        if current_balance > self.peak_balance:
            self.peak_balance = current_balance

        # Calculate current drawdown
        if self.peak_balance > 0:
            self.current_drawdown = 1 - (current_balance / self.peak_balance)

        # Input validation
        if current_balance <= 0:
            logger.warning(
                f"Invalid balance: {current_balance}. Using minimum position."
            )
            return current_balance * self.min_position_pct

        if risk_factor <= 0 or risk_factor > 0.05:
            logger.warning(
                f"Risk factor {risk_factor} outside reasonable range. Capping at 0.05."
            )
            risk_factor = min(0.05, max(0.001, risk_factor))

        # Extract parameters from market_info
        conviction = market_info.get("conviction", 0.5) if market_info else 0.5
        volatility = market_info.get("volatility", None) if market_info else None
        antifragility = market_info.get("antifragility", 1.0) if market_info else 1.0

        try:
            # Update volatility regime
            if returns is not None and len(returns) > 20:
                self.update_volatility_regime(returns)

            # Recalculate Kelly fraction periodically
            if self.trade_count % 10 == 0:
                self.kelly_fraction = self.calculate_kelly_fraction()

            # Use CVaR instead of VaR for more conservative sizing in stressed markets
            # Calculate risk measure based on volatility regime
            cache_key = f"{pair}_risk" if pair else None

            if self.volatility_regime in ["high", "extreme"]:
                # Use CVaR in high volatility regimes
                risk_measure = self.calculate_cvar(returns, cache_key)
            else:
                # Use VaR normally, varying method by regime
                if self.volatility_regime == "low":
                    var_method = "parametric"  # More efficient in low vol
                else:
                    var_method = "cornish_fisher"  # Better for tail risk

                risk_measure = self.calculate_var(returns, cache_key, method=var_method)

            # Apply Kelly-inspired position sizing
            # Base calculation: balance * risk_factor * kelly_fraction / risk_measure
            raw_position = (
                current_balance
                * risk_factor
                * self.kelly_fraction
                / max(risk_measure, 0.001)
            )

            # Apply regime-based adjustments
            vol_multiplier = self.risk_multipliers["volatility"].get(
                self.volatility_regime, 1.0
            )
            corr_multiplier = self.risk_multipliers["correlation"].get(
                self.correlation_regime, 1.0
            )

            # Apply conviction scaling (higher conviction = larger position)
            conviction_multiplier = 0.7 + (conviction * 0.6)  # 0.7 to 1.3 range

            # Apply antifragility adjustment
            antifragility_multiplier = 0.8 + (antifragility * 0.4)  # 0.8 to 1.2 range

            # Combine multipliers
            combined_multiplier = (
                vol_multiplier
                * corr_multiplier
                * conviction_multiplier
                * antifragility_multiplier
            )

            # Apply drawdown protection
            if self.current_drawdown > 0:
                # Reduce position sizing proportionally to drawdown
                drawdown_factor = 1.0 - (self.current_drawdown / self.max_drawdown)
                drawdown_factor = max(0.2, drawdown_factor)  # Never go below 20%
                combined_multiplier *= drawdown_factor

                # Log significant drawdown adjustments
                if drawdown_factor < 0.8:
                    logger.warning(
                        f"Reducing position due to drawdown: {self.current_drawdown:.2%}, "
                        f"factor: {drawdown_factor:.2f}"
                    )

            # Apply combined multiplier
            adjusted_position = raw_position * combined_multiplier

            # Apply sensible limits
            max_position = current_balance * self.max_position_pct
            min_position = current_balance * self.min_position_pct

            # Clamp within position range
            final_position = max(min_position, min(adjusted_position, max_position))

            # Record for history
            self.position_history.append(
                {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "balance": current_balance,
                    "risk_measure": risk_measure,
                    "raw_position": raw_position,
                    "final_position": final_position,
                    "pair": pair,
                    "multiplier": combined_multiplier,
                    "current_drawdown": self.current_drawdown,
                    "volatility_regime": self.volatility_regime,
                    "correlation_regime": self.correlation_regime,
                }
            )

            # Keep history from growing too large
            if len(self.position_history) > 1000:
                self.position_history = self.position_history[-1000:]

            logger.info(
                f"Position sizing: {final_position:.2f} ({(final_position / current_balance) * 100:.1f}% of balance), "
                f"multiplier: {combined_multiplier:.2f}, VR: {self.volatility_regime}, CR: {self.correlation_regime}"
            )
            return final_position

        except Exception as e:
            logger.error(f"Error in position sizing: {str(e)}")
            return current_balance * self.min_position_pct  # Conservative fallback

    def update_trade_result(self, profit: float, trade_info: Dict = None):
        """
        Update performance metrics with trade result

        Args:
            profit: Trade profit (positive) or loss (negative)
            trade_info: Additional trade information
        """
        try:
            # Update win/loss counts and sums
            if profit > 0:
                self.win_count += 1
                self.profit_sum += profit
            else:
                self.loss_count += 1
                self.loss_sum += profit  # Loss is negative

            # Update peak balance if provided in trade_info
            if trade_info and "balance" in trade_info:
                if trade_info["balance"] > self.peak_balance:
                    self.peak_balance = trade_info["balance"]

            # Recalculate drawdown if balance provided
            if trade_info and "balance" in trade_info and self.peak_balance > 0:
                self.current_drawdown = 1 - (trade_info["balance"] / self.peak_balance)

            logger.info(
                f"Trade result: {'Win' if profit > 0 else 'Loss'} {profit:.2%}, "
                f"W/L: {self.win_count}/{self.loss_count}, "
                f"Drawdown: {self.current_drawdown:.2%}"
            )

        except Exception as e:
            logger.error(f"Error updating trade result: {str(e)}")

    def get_risk_metrics(self):
        """
        Get comprehensive risk metrics

        Returns:
            Dictionary of risk metrics
        """
        total_trades = self.win_count + self.loss_count
        win_rate = self.win_count / max(1, total_trades)
        profit_factor = abs(self.profit_sum / max(0.001, self.loss_sum))

        return {
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "total_trades": total_trades,
            "current_drawdown": self.current_drawdown,
            "volatility_regime": self.volatility_regime,
            "correlation_regime": self.correlation_regime,
            "kelly_fraction": self.kelly_fraction,
            "risk_allocation": self.max_position_pct,
        }

    def recover(self): # Make sure the recover method exists
        """Recovers the RiskManager component."""
        self.logger.warning("RiskManager recovery triggered!")
        try:
            with self._lock: # Now this lock exists
                 # Add specific recovery logic for RiskManager if any
                 # e.g., clear caches, reset state variables
                 self.var_cache = {}
                 self.cvar_cache = {}
                 self.current_drawdown = 0.0
                 self.peak_balance = 0.0 # Reset peak balance? Maybe not desirable.
                 self.trade_count = 0
                 self.win_count = 0
                 self.loss_count = 0
                 self.profit_sum = 0
                 self.loss_sum = 0
            self.logger.info("RiskManager recovery attempt finished successfully.")
        except Exception as e_rec:
            self.logger.error(f"Error during RiskManager recovery: {e_rec}", exc_info=True)

class AntifragileRiskManager(RiskManager):
    """
    Advanced risk management system that becomes stronger from volatility.

    This system implements Taleb's antifragility concepts to create dynamic
    risk parameters that adapt to market conditions and past performance.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs) # Call parent init (which should add _lock)
        self.logger = logging.getLogger(f"{__name__}.AntifragileRiskManager")
        self.logger.info("Antifragile Risk Manager initialized (Placeholder Logic).")

        self.position_sizing_factors = {
            "win_streak": 0.1,  # Increase sizing on win streaks
            "loss_streak": -0.15,  # Decrease sizing on loss streaks
            "volatility": -0.2,  # Reduce sizing in high volatility
            "conviction": 0.15,  # Increase sizing with high conviction
            "antifragility": 0.2,  # Increase sizing with high antifragility
        }

        # Dynamic stoploss and take profit parameters
        self.base_stoploss = -0.05
        self.base_roi = {"0": 0.05, "10": 0.04, "20": 0.03, "40": 0.02}

        # Order flow tracking
        self.trades_history = []
        self.win_streak = 0
        self.loss_streak = 0
        self.current_drawdown = 0
        self.max_drawdown = 0
        self.profit_factor = 1.0

        # Black swan protection
        self.black_swan_threshold = 3.0  # Z-score for black swan detection
        self.black_swan_protection = False
        self.reduced_exposure = 1.0  # Factor to reduce exposure (1.0 = full exposure)

        # Performance metrics
        self.total_profit = 0
        self.total_trades = 0
        self.winning_trades = 0

        logger.info("Antifragile Risk Manager initialized")

    def calculate_position_size(
        self, current_balance, antifragility_score, volatility, conviction_score, pair
    ):
        """
        Calculate optimal position size using antifragile principles.

        Args:
            current_balance: Current account balance
            antifragility_score: Market antifragility metric (0-2, higher is more antifragile)
            volatility: Current market volatility (standard deviation)
            conviction_score: Signal conviction (0-1)
            pair: Trading pair

        Returns:
            Recommended position size
        """
        # Base position size - Kelly-inspired
        win_rate = self.winning_trades / max(1, self.total_trades)
        profit_factor = self.profit_factor

        # Calculate half-Kelly position size (more conservative)
        if win_rate > 0 and profit_factor > 1:
            kelly_pct = (win_rate * profit_factor - (1 - win_rate)) / profit_factor
            base_pct = max(0.01, min(0.25, kelly_pct / 2))  # Half-Kelly, capped at 25%
        else:
            base_pct = 0.05  # Default 5% when no history

        # Apply antifragile adjustments
        adjustments = 0

        # Streaks adjustment - increase size on wins, decrease on losses
        if self.win_streak >= 3:
            streak_adj = min(
                self.win_streak * self.position_sizing_factors["win_streak"], 0.3
            )
            adjustments += streak_adj
            logger.info(
                f"Win streak ({self.win_streak}): +{streak_adj:.2f} size adjustment"
            )

        if self.loss_streak >= 2:
            streak_adj = max(
                -0.5, self.loss_streak * self.position_sizing_factors["loss_streak"]
            )
            adjustments += streak_adj
            logger.info(
                f"Loss streak ({self.loss_streak}): {streak_adj:.2f} size adjustment"
            )

        # Volatility adjustment - reduce size in high volatility
        volatility_factor = 0
        if volatility > 0.03:  # High volatility
            volatility_factor = (
                (volatility - 0.03) * 10 * self.position_sizing_factors["volatility"]
            )
            adjustments += volatility_factor
            logger.info(
                f"High volatility ({volatility:.4f}): {volatility_factor:.2f} size adjustment"
            )

        # Conviction adjustment - increase size with high conviction signals
        if conviction_score > 0.7:
            conviction_adj = (
                (conviction_score - 0.7)
                * self.position_sizing_factors["conviction"]
                * 2
            )
            adjustments += conviction_adj
            logger.info(
                f"High conviction ({conviction_score:.2f}): +{conviction_adj:.2f} size adjustment"
            )

        # Antifragility adjustment - increase size in antifragile markets
        antifragility_adj = (antifragility_score - 1.0) * self.position_sizing_factors[
            "antifragility"
        ]
        adjustments += antifragility_adj

        # Apply all adjustments to base percentage
        final_pct = base_pct * (1 + adjustments)

        # Safety caps
        final_pct = max(0.01, min(0.25, final_pct))

        # Black swan protection - reduce all position sizes
        if self.black_swan_protection:
            final_pct *= self.reduced_exposure
            logger.info(
                f"Black swan protection active: exposure reduced to {self.reduced_exposure:.2f}"
            )

        # Calculate actual position size
        position_size = current_balance * final_pct

        logger.info(
            f"Position size for {pair}: {position_size:.2f} ({final_pct:.2%} of balance)"
        )

        return position_size

    def calculate_dynamic_stoploss(
        self, antifragility_score, volatility, trend_strength, entry_price
    ):
        """
        Calculate dynamic stoploss based on market conditions
        """
        # Start with base stoploss
        base_sl = self.base_stoploss

        # Adjust for volatility - tighter in low vol, wider in high vol
        volatility_factor = volatility / 0.01  # Normalize to typical volatility
        volatility_adjustment = -0.01 * (
            volatility_factor - 1
        )  # Tighter when less volatile

        # Adjust for trend - tighter in strong trends
        trend_adjustment = -0.01 * abs(trend_strength) * 5  # Tighter in strong trends

        # Adjust for antifragility - tighter in antifragile markets
        antifragility_adjustment = -0.01 * (antifragility_score - 1)

        # Calculate final stoploss
        final_sl = (
            base_sl
            + volatility_adjustment
            + trend_adjustment
            + antifragility_adjustment
        )

        # Safety caps
        final_sl = max(-0.15, min(-0.02, final_sl))

        # Calculate actual stop price
        stop_price = entry_price * (1 + final_sl)

        logger.info(f"Dynamic stoploss: {final_sl:.2%} (price: {stop_price:.4f})")

        return final_sl, stop_price

    def calculate_dynamic_takeprofits(
        self,
        antifragility_score,
        volatility,
        trend_strength,
        conviction_score,
        entry_price,
    ):
        """
        Calculate multiple take profit levels based on market conditions
        """
        # Base take profit levels - percentage and required candles
        base_roi = self.base_roi.copy()

        # More aggressive take profits in volatile markets
        if volatility > 0.02:
            volatility_factor = 1 + (volatility - 0.02) * 5
            # Increase profit targets but reduce time
            base_roi = {
                "0": base_roi["0"] * volatility_factor,
                "5": base_roi["10"] * volatility_factor,  # Faster exit
                "15": base_roi["20"] * volatility_factor * 0.9,
                "30": base_roi["40"] * volatility_factor * 0.8,
            }

        # More patient take profits in trending markets
        elif abs(trend_strength) > 0.05:
            trend_factor = 1 + abs(trend_strength) * 10
            # Increase profit targets and extend time
            base_roi = {
                "0": base_roi["0"] * trend_factor,
                "15": base_roi["10"] * trend_factor,  # Slower exit
                "30": base_roi["20"] * trend_factor,
                "60": base_roi["40"] * trend_factor,
            }

        # Adjust based on conviction
        if conviction_score > 0.8:
            # High conviction - higher targets, more patience
            for k in base_roi:
                base_roi[k] *= 1.2

        # Adjust based on antifragility
        if antifragility_score > 1.2:
            # Higher antifragility - can be more patient
            for k in base_roi:
                base_roi[k] *= 1.1

        # Calculate take profit prices
        tp_prices = {int(k): entry_price * (1 + v) for k, v in base_roi.items()}

        logger.info(f"Dynamic take profits: {base_roi}")

        return base_roi, tp_prices

    def update_with_trade_result(self, win, profit_ratio, trade_duration):
        """
        Update risk parameters based on trade outcome
        """
        self.total_trades += 1

        if win:
            self.winning_trades += 1
            self.win_streak += 1
            self.loss_streak = 0
            self.total_profit += profit_ratio
        else:
            self.win_streak = 0
            self.loss_streak += 1
            self.total_profit += profit_ratio
            self.current_drawdown -= profit_ratio

        # Update profit factor
        if self.total_trades >= 5:
            winning_trades = (
                self.trades_history[-20:]
                if len(self.trades_history) >= 20
                else self.trades_history
            )
            profit_sum = sum(t["profit"] for t in winning_trades if t["profit"] > 0)
            loss_sum = abs(sum(t["profit"] for t in winning_trades if t["profit"] < 0))
            self.profit_factor = profit_sum / max(loss_sum, 1e-9)

        # Track maximum drawdown
        self.max_drawdown = min(self.max_drawdown, self.current_drawdown)

        # Reset drawdown if profitable
        if self.current_drawdown >= 0:
            self.current_drawdown = 0

        # Add to trade history
        self.trades_history.append(
            {"win": win, "profit": profit_ratio, "duration": trade_duration}
        )

        # Keep only last 100 trades
        if len(self.trades_history) > 100:
            self.trades_history = self.trades_history[-100:]

        logger.info(
            f"Trade result - Win: {win}, Profit: {profit_ratio:.2%}, Win rate: {self.winning_trades / self.total_trades:.2%}"
        )

    def recover(self):
          """Recovers the AntifragileRiskManager component."""
          self.logger.warning("AntifragileRiskManager recovery triggered!")
          try:
              # Call parent recovery first
              super().recover()
              # Add specific Antifragile recovery steps here
              # e.g., reset specific state related to antifragility metrics
              self.logger.info("AntifragileRiskManager recovery attempt finished successfully.")
          except Exception as e_rec:
               self.logger.error(f"Error during AntifragileRiskManager recovery: {e_rec}", exc_info=True)


@dataclass
class ViaNegativaParams:
    """Parameters specifically for Via Negativa checks."""
    # Thresholds for scores/flags coming FROM AnomalyDetector
    bs_prob_threshold: float = 0.7
    whale_score_threshold: float = 0.7
    fragility_threshold: float = 0.7
    correlation_breakdown_threshold: float = 0.6
    flash_crash_prob_threshold: float = 0.6 # If score passed
    liquidity_crisis_threshold: float = 0.7 # If score passed
    # Add threshold for SOC Critical if needed (e.g., check string directly or add bool threshold)
    # soc_critical_threshold: bool = True # Example if checking boolean

    # Thresholds for checks done INTERNALLY by ViaNegativa on dataframe
    vn_liquidity_threshold: float = 0.5 # Volume drop ratio
    vn_liquidity_lookback: int = 5
    vn_momentum_reversal_threshold: float = -0.8 # Correlation or sign change logic
    vn_momentum_lookback: int = 10
    vn_volatility_expansion_threshold: float = 2.0 # Vol ratio
    vn_volatility_lookback: int = 10
    vn_regime_break_corr_change: float = 0.5 # Diff in autocorrelation
    vn_regime_break_vol_change: float = 2.0 # Ratio of std dev
    vn_regime_break_lookback: int = 20
    vn_tail_event_threshold: float = 4.0 # Threshold for internal tail check (Z-score/Sigma)
    # Add vn_tail_event_lookback if different from regime_break lookback
    vn_tail_event_lookback: int = 20
    

class ViaNegativaFilter:
    """
    Implements Taleb's Via Negativa principle - improvement by removal rather than addition.

    This approach focuses on identifying and avoiding potentially harmful trades rather than
    trying to predict profitable ones. It's based on the idea that it's easier to identify
    what will fail than what will succeed.

    The filter maintains a growing list of risk patterns based on past failures and market
    conditions that have historically led to losses.
    """

    def __init__(self, params: Optional[ViaNegativaParams] = None, max_memory_entries: int = 1000):
        """
        Initialize the Via Negativa filter.

        Args:
            max_memory_entries: Maximum number of historical failure patterns to store
        """
        self.logger = logging.getLogger(__name__)
        self.params = params if params is not None else ViaNegativaParams() # Store passed params or create default
        self.max_entries = max_memory_entries
        self.risk_patterns = self._initialize_risk_patterns()
        self.historical_failures = CircularBuffer(max_memory_entries)
        self.active_flags = set()
        self.flag_history = {}
        self.last_update_time = time.time()

        # Thread safety
        self._lock = threading.RLock()

        logger.info("Via Negativa filter initialized with base risk patterns")

    def _initialize_risk_patterns(self) -> Dict[str, Dict[str, Any]]:
        """
        Initialize base risk patterns to avoid.

        Returns:
            Dict of risk patterns with detection parameters
        """
        return {
            "liquidity_risk": {
                "description": "Sudden drop in trading volume indicating potential liquidity issues",
                "threshold": 0.5,  # 50% drop in volume
                "lookback_period": 5,  # Compare to previous 5 periods
                "severity": "high",
            },
            "tail_event": {
                "description": "Extreme price movement beyond normal distribution expectations",
                "threshold": 4.0,  # 4 standard deviations
                "lookback_period": 20,
                "severity": "extreme",
            },
            "adverse_whale_activity": {
                "description": "Large holder activity potentially moving against position",
                "threshold": 0.7,  # Confidence threshold
                "severity": "high",
            },
            "regime_break": {
                "description": "Sudden change in market statistical properties",
                "threshold": 0.8,  # Correlation breakdown threshold
                "lookback_period": 20,
                "severity": "high",
            },
            "volatility_expansion": {
                "description": "Rapid increase in market volatility",
                "threshold": 2.0,  # Volatility doubling
                "lookback_period": 10,
                "severity": "medium",
            },
            "momentum_reversal": {
                "description": "Sudden reversal in price momentum",
                "threshold": -0.8,  # Correlation between recent and previous momentum
                "lookback_period": 10,
                "severity": "medium",
            },
            "correlation_breakdown": {
                "description": "Breakdown of normal correlations between assets",
                "threshold": 0.4,  # Change in correlation
                "severity": "high",
            },
        }

    # Inside ViaNegativaFilter class in your risk_manager.py

    def identify_risk_flags(
        self, dataframe: pd.DataFrame, anomalies: Dict[str, Any]
    ) -> List[str]:
        """
        Identify potential risk flags based on market data and anomalies.
        Now handles score inputs in 'anomalies' dict.

        Args:
            dataframe: Market data (ensure it has needed columns like 'close', 'volume')
            anomalies: Detected market anomalies (can contain scores or booleans)
                       Example: {'black_swan': 0.8, 'whale_activity': 0.75, 'soc_critical': True}

        Returns:
            List of active risk flag names
        """
        if dataframe.empty:
            return []

        logger = logging.getLogger(__name__) # Use module/class logger
        logger.debug("ViaNegativa: Identifying risk flags...")
        # Log the received anomalies dict for debugging
        logger.debug(f"  Received anomalies dict: {anomalies}")

        try:
            with self._lock:
                self.active_flags.clear()
                detected_flags = set() # Use a temporary set
                if anomalies.get("bs_prob", 0.0) > self.params.bs_prob_threshold: detected_flags.add("tail_event") # Check external detector output
                if anomalies.get("whale_score", 0.0) > self.params.whale_score_threshold: detected_flags.add("adverse_whale_activity")
                if anomalies.get("soc_regime", "normal") == 'critical': detected_flags.add("soc_critical")
                if anomalies.get("fragility", 0.0) > self.params.fragility_threshold: detected_flags.add("high_fragility")


                # --- Check each risk pattern ---

                # Liquidity Risk (based on dataframe volume)
                if self._detect_liquidity_risk(dataframe):
                    detected_flags.add("liquidity_risk")
                    logger.debug("  Flag ADDED: liquidity_risk (volume check)")

                # Tail Event (Check anomaly dict first, then dataframe)
                # Assuming anomaly dict contains probability/score > threshold means True
                bs_prob = anomalies.get("black_swan", 0.0) # Get score, default 0
                bs_threshold = 0.7 # Example threshold for black swan prob
                if isinstance(bs_prob, (float, int)) and bs_prob > bs_threshold:
                    detected_flags.add("tail_event")
                    logger.debug(f"  Flag ADDED: tail_event (from anomalies dict: bs_prob={bs_prob:.3f} > {bs_threshold})")
                elif self._detect_tail_event(dataframe): # Fallback dataframe check
                    detected_flags.add("tail_event")
                    logger.debug("  Flag ADDED: tail_event (from dataframe check)")

                # Adverse Whale Activity (Check anomaly dict)
                # Use threshold defined in risk_patterns
                whale_score = anomalies.get("whale_activity", 0.0)
                whale_threshold = self.risk_patterns.get("adverse_whale_activity", {}).get("threshold", 0.7)
                if isinstance(whale_score, (float, int)) and whale_score > whale_threshold:
                    detected_flags.add("adverse_whale_activity")
                    logger.debug(f"  Flag ADDED: adverse_whale_activity (from anomalies dict: whale_score={whale_score:.3f} > {whale_threshold})")

                # Regime Break (based on dataframe)
                if self._detect_regime_break(dataframe):
                    detected_flags.add("regime_break")
                    logger.debug("  Flag ADDED: regime_break (dataframe check)")

                # Volatility Expansion (based on dataframe)
                if self._detect_volatility_expansion(dataframe):
                    detected_flags.add("volatility_expansion")
                    logger.debug("  Flag ADDED: volatility_expansion (dataframe check)")

                # Momentum Reversal (based on dataframe)
                if self._detect_momentum_reversal(dataframe):
                    detected_flags.add("momentum_reversal")
                    logger.debug("  Flag ADDED: momentum_reversal (dataframe check)")

                # Correlation Breakdown (Check anomaly dict)
                # Assuming a boolean flag or a score > threshold
                corr_breakdown_input = anomalies.get("correlation_breakdown", False)
                corr_breakdown_threshold = self.risk_patterns.get("correlation_breakdown", {}).get("threshold", 0.4) # Example threshold if score provided
                if isinstance(corr_breakdown_input, bool) and corr_breakdown_input:
                    detected_flags.add("correlation_breakdown")
                    logger.debug("  Flag ADDED: correlation_breakdown (from anomalies dict: boolean True)")
                elif isinstance(corr_breakdown_input, (float, int)) and corr_breakdown_input > corr_breakdown_threshold:
                     detected_flags.add("correlation_breakdown")
                     logger.debug(f"  Flag ADDED: correlation_breakdown (from anomalies dict: score={corr_breakdown_input:.3f} > {corr_breakdown_threshold})")
               
                # SOC Critical (using SOC regime string)
                soc_regime = anomalies.get("soc_regime", "normal")
                if soc_regime == 'critical': # Direct check for 'critical' string
                    detected_flags.add("soc_critical")
                    logger.debug("  Flag ADDED: soc_critical (regime is 'critical')")
                    
                # SOC Critical (Check anomaly dict - assumes boolean)
                if anomalies.get("soc_critical", False):
                    detected_flags.add("soc_critical")
                    logger.debug("  Flag ADDED: soc_critical (from anomalies dict: boolean True)")

                # High Fragility (Check anomaly dict - assumes score)
                fragility_score = anomalies.get("high_fragility", 0.0) # Use a descriptive key
                fragility_threshold = 0.7 # Example threshold
                if isinstance(fragility_score, (float, int)) and fragility_score > fragility_threshold:
                    detected_flags.add("high_fragility")
                    logger.debug(f"  Flag ADDED: high_fragility (from anomalies dict: score={fragility_score:.3f} > {fragility_threshold})")

                # --- Update active flags and history ---
                self.active_flags = detected_flags
                if self.active_flags: # Log only if flags were found
                    logger.info(f"Via Negativa identified risk flags: {list(self.active_flags)}")
                else:
                     logger.debug("Via Negativa: No risk flags identified.")
                self._update_flag_history() # Update history regardless

                return list(self.active_flags)

        except Exception as e:
            logger.error(f"Error identifying risk flags: {e}", exc_info=True)
            return [] # Return empty list on error

    def _detect_liquidity_risk(self, dataframe: pd.DataFrame) -> bool:
        """
        Detect potential liquidity risk from volume data.

        Args:
            dataframe: Market data with volume information

        Returns:
            bool: True if liquidity risk detected
        """
        lookback = self.params.vn_liquidity_lookback
        threshold = self.params.vn_liquidity_threshold
        if "volume" not in dataframe.columns or len(dataframe) < 10:
            return False

        try:
            # Get pattern parameters
            pattern = self.risk_patterns["liquidity_risk"]
            threshold = pattern["threshold"]
            lookback = pattern["lookback_period"]

            # Calculate recent volume
            recent_vol = dataframe["volume"].tail(lookback).mean()

            # Calculate historical volume (avoiding overlap)
            if len(dataframe) > lookback * 2:
                historical_vol = (
                    dataframe["volume"].iloc[-(lookback * 2) : -lookback].mean()
                )
            else:
                # Not enough data for comparison
                return False

            # Check for significant volume drop
            if historical_vol > 0:  # Avoid division by zero
                volume_ratio = recent_vol / historical_vol
                is_risk = volume_ratio < threshold
                logger.debug(f"Liquidity Check: Ratio={volume_ratio:.2f}, Threshold={threshold}, Risk={is_risk}") # Optional detailed log
                return is_risk
            return False
        except Exception as e: logger.error(f"Error detecting liquidity risk: {e}"); return False


    def _detect_tail_event(self, dataframe: pd.DataFrame) -> bool:
        """
        Detect tail events (black swans) directly from price data.
        CORRECTED: Calculates max_deviation before using it.

        Args:
            dataframe: Market price data

        Returns:
            bool: True if tail event detected
        """
        # Get the logger instance (ensure logger is defined for the class or module)
        logger = logging.getLogger(__name__) # Or self.logger if defined in __init__

        if "close" not in dataframe.columns or dataframe['close'].isnull().all():
            # logger.debug("Tail Event Check: No 'close' data.")
            return False
        if len(dataframe) < 30: # Need enough data for lookback+recent and std calc
            # logger.debug(f"Tail Event Check: Insufficient data ({len(dataframe)} < 30).")
            return False

        try:
            # Get pattern parameters - Use self.params if defined, otherwise hardcode/defaults
            # Assuming self.params exists and has ViaNegativaParams structure
            #threshold = getattr(self.params, 'vn_tail_event_threshold', 4.0) # Example using 4.0 default if not in params
            #lookback = getattr(self.params, 'vn_tail_event_lookback', 20) # Example using 20 default
            
            # --- Access thresholds via self.params ---

            threshold = self.params.vn_tail_event_threshold # Default defined in ViaNegativaParams dataclass
            lookback = self.params.vn_regime_break_lookback # Assuming same lookback for consistency? Or add vn_tail_event_lookback to params
            
            # Calculate returns, drop initial NaN
            returns = dataframe["close"].pct_change().dropna()

            # Ensure enough returns calculated for comparison periods
            if len(returns) < lookback + 5:
                 # logger.debug(f"Tail Event Check: Insufficient returns data ({len(returns)} < {lookback + 5}).")
                 return False

            # Calculate historical standard deviation (excluding recent window)
            historical_returns = returns.iloc[:-(lookback // 4)] # Exclude recent portion used for max move
            historical_std = historical_returns.tail(lookback).std()

            # Check recent returns for extreme movements
            recent_returns = returns.tail(5) # Look at last 5 returns

            # --- >>> INSERT CALCULATION HERE <<< ---
            max_abs_recent_return = abs(recent_returns).max() if not recent_returns.empty else 0.0
            historical_std_safe = max(historical_std, 1e-9) if pd.notna(historical_std) else 1e-9 # Safe std dev

            max_deviation = max_abs_recent_return / historical_std_safe
            # --- >>> END INSERTED CALCULATION <<< ---

            # Ensure max_deviation is a finite number
            if not np.isfinite(max_deviation):
                 max_deviation = 0.0

            # Perform the check
            is_risk = max_deviation > threshold
            logger.debug(f"Tail Event Check: Max Deviation={max_deviation:.2f} sigma, Threshold={threshold:.1f} sigma, Risk={is_risk}") # Log calculation result
            return is_risk

        except Exception as e:
            logger.error(f"Error detecting tail event: {e}", exc_info=True) # Log full traceback on error
            return False # Return False on error


    def _detect_regime_break(self, dataframe: pd.DataFrame) -> bool:
        """
        Detect statistical breaks in market regime. Uses thresholds from self.params.

        Args:
            dataframe: Market data (needs 'close' column)

        Returns:
            bool: True if regime break detected
        """
        # Get parameters directly from the stored config object
        lookback = self.params.vn_regime_break_lookback
        corr_change_threshold = self.params.vn_regime_break_corr_change
        vol_change_threshold = self.params.vn_regime_break_vol_change

        # Validate inputs
        if "close" not in dataframe.columns or dataframe['close'].isnull().all():
            # self.logger.debug("Regime Break Check: Missing or all NaN 'close' data.")
            return False
        if len(dataframe) < (lookback * 2) + 1:
            # self.logger.debug(f"Regime Break Check: Insufficient data ({len(dataframe)} < {lookback * 2}).")
            self.logger.debug(f"Regime Break Check: Insufficient data ({len(dataframe)} < {lookback * 2 + 1}).")
            return False

        try:
            # Calculate returns (handle potential NaNs/Infs)
            returns = dataframe["close"].iloc[1:].pct_change().replace([np.inf, -np.inf], np.nan).dropna()
            # Check if we have enough valid returns data
            if len(returns) < lookback * 2:
                # self.logger.debug(f"Regime Break Check: Insufficient valid returns ({len(returns)} < {lookback * 2}).")
                return False

            # Divide into recent and historical periods
            recent_returns = returns.tail(lookback)
            historical_returns = returns.iloc[-(lookback * 2) : -lookback]

            # --- Calculate Volatility Change ---
            recent_vol = recent_returns.std()
            historical_vol = historical_returns.std()
            vol_change_ratio = 1.0 # Default if historical vol is zero/NaN
            if pd.notna(historical_vol) and historical_vol > 1e-9:
                 if pd.notna(recent_vol):
                      vol_change_ratio = recent_vol / historical_vol
                 # else: Keep ratio 1.0 if recent_vol is NaN

            # --- Calculate Autocorrelation Change ---
            recent_autocorr_raw = recent_returns.autocorr(lag=1)
            recent_autocorr = 0.0 if pd.isna(recent_autocorr_raw) else recent_autocorr_raw

            historical_autocorr_raw = historical_returns.autocorr(lag=1)
            historical_autocorr = 0.0 if pd.isna(historical_autocorr_raw) else historical_autocorr_raw

            autocorr_change = abs(recent_autocorr - historical_autocorr)

            # --- Determine Regime Break ---
            volatility_changed = (vol_change_ratio > vol_change_threshold) or \
                                 (vol_change_ratio < (1 / vol_change_threshold)) # Check both high increase and high decrease
            correlation_changed = autocorr_change > corr_change_threshold

            is_break = volatility_changed and correlation_changed

            # self.logger.debug(f"Regime Break Check: VolRatio={vol_change_ratio:.2f} (Thresh={vol_change_threshold}), "
            #                  f"CorrChange={autocorr_change:.2f} (Thresh={corr_change_threshold}), Break={is_break}")

            return is_break

        except Exception as e:
            self.logger.error(f"Error detecting regime break: {e}", exc_info=True)
            return False

    def _detect_volatility_expansion(self, dataframe: pd.DataFrame) -> bool:
        """
        Detect rapid expansion in market volatility.

        Args:
            dataframe: Market data

        Returns:
            bool: True if volatility expansion detected
        """
        lookback = self.params.vn_volatility_lookback
        threshold = self.params.vn_volatility_expansion_threshold
        if "close" not in dataframe.columns or len(dataframe) < 20:
            return False

        try:
            # Get pattern parameters
            pattern = self.risk_patterns["volatility_expansion"]
            threshold = pattern["threshold"]
            lookback = pattern["lookback_period"]

            # Calculate returns
            returns = dataframe["close"].pct_change().dropna()

            # Check if we have enough data
            if len(returns) < lookback * 2:
                return False

            # Calculate recent and previous volatility
            recent_vol = returns.tail(lookback).std()
            prev_vol = returns.iloc[-(lookback * 2) : -lookback].std()

            # Check for volatility expansion
            if prev_vol > 0:
                vol_ratio = recent_vol / prev_vol
                return vol_ratio > threshold

            return False

        except Exception as e:
            logger.error(f"Error detecting volatility expansion: {e}")
            return False

    def _detect_momentum_reversal(self, dataframe: pd.DataFrame) -> bool:
        """
        Detect sudden reversal in price momentum.

        Args:
            dataframe: Market data

        Returns:
            bool: True if momentum reversal detected
        """
        lookback = self.params.vn_momentum_lookback
        threshold = self.params.vn_momentum_reversal_threshold # Note: This is typically negative
        if "close" not in dataframe.columns or len(dataframe) < 20:
            return False

        try:
            # Get pattern parameters
            pattern = self.risk_patterns["momentum_reversal"]
            threshold = pattern["threshold"]
            lookback = pattern["lookback_period"]

            # Calculate price changes over different periods
            if len(dataframe) >= lookback * 2:
                # Recent momentum (last lookback periods)
                recent_change = (
                    dataframe["close"].tail(lookback).pct_change(lookback - 1).iloc[-1]
                )

                # Previous momentum
                prev_change = (
                    dataframe["close"]
                    .iloc[-(lookback * 2) : -lookback]
                    .pct_change(lookback - 1)
                    .iloc[-1]
                )

                # Check for momentum reversal
                # A momentum reversal occurs when:
                # - Both momentum values are significant (not close to zero)
                # - They have opposite signs
                # - Or their correlation is strongly negative

                if abs(recent_change) > 0.01 and abs(prev_change) > 0.01:
                    if np.sign(recent_change) != np.sign(prev_change):
                        return True

                    # Calculate correlation (simplified for small samples)
                    correlation = -1.0 if recent_change * prev_change < 0 else 1.0

                    return correlation < threshold

            return False

        except Exception as e:
            logger.error(f"Error detecting momentum reversal: {e}")
            return False

    def _detect_correlation_breakdown(self, dataframe: pd.DataFrame) -> bool:
        """
        Detect breakdown in normal asset correlations.

        Args:
            dataframe: Market data with correlation information

        Returns:
            bool: True if correlation breakdown detected
        """
        if "correlation_matrix" not in dataframe.columns:
            return False

        try:
            # Get pattern parameters
            pattern = self.risk_patterns["correlation_breakdown"]
            threshold = pattern["threshold"]

            # This is a placeholder implementation
            # In a complete implementation, we would analyze correlation matrices
            # comparing recent to historical correlations

            return False

        except Exception as e:
            logger.error(f"Error detecting correlation breakdown: {e}")
            return False

    def _update_flag_history(self) -> None:
        """
        Update history of active flags for tracking and analysis.
        """
        current_time = time.time()

        # Only update if it's been at least 5 minutes since last update
        if current_time - self.last_update_time < 300:
            return

        self.last_update_time = current_time
        timestamp = pd.Timestamp.now().isoformat()

        for flag in self.active_flags:
            if flag not in self.flag_history:
                self.flag_history[flag] = []

            self.flag_history[flag].append(timestamp)

            # Keep only recent history
            if len(self.flag_history[flag]) > 100:
                self.flag_history[flag] = self.flag_history[flag][-100:]

    def get_active_flags(self) -> List[str]:
        """
        Get current active risk flags.

        Returns:
            List of active risk flag names
        """
        with self._lock:
            return list(self.active_flags)

    def add_failure_pattern(self, pattern_data: Dict[str, Any]) -> None:
        """
        Add a new failure pattern from observed market conditions.

        Args:
            pattern_data: Information about the failure pattern
        """
        with self._lock:
            # Store in historical failures
            self.historical_failures.append(pattern_data)

            # Extract pattern attributes for future detection
            # This would be expanded in a complete implementation to
            # learn from failures and adjust detection algorithms

            logger.info(
                f"Added new failure pattern: {pattern_data.get('description', 'unknown')}"
            )

    def get_flag_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about historical risk flag occurrences.

        Returns:
            Dict with statistics for each flag type
        """
        with self._lock:
            stats = {}

            for flag, history in self.flag_history.items():
                stats[flag] = {
                    "count": len(history),
                    "first_seen": history[0] if history else None,
                    "last_seen": history[-1] if history else None,
                    "severity": self.risk_patterns.get(flag, {}).get(
                        "severity", "unknown"
                    ),
                }

            return stats

    def recover(self):
        """Recovers the ViaNegativaFilter component."""
        self.logger.warning("ViaNegativaFilter recovery triggered!")
        try:
            with self._lock:
                 self.active_flags.clear()
                 # Optionally clear historical_failures or flag_history if desired
                 # self.historical_failures.clear()
                 # self.flag_history.clear()
            self.logger.info("ViaNegativaFilter recovery attempt finished successfully (cleared active flags).")
        except Exception as e_rec:
             self.logger.error(f"Error during ViaNegativaFilter recovery: {e_rec}", exc_info=True)

class BarbellAllocator:
    """
    Implements Nassim Taleb's Barbell Strategy for position sizing and risk management.

    The Barbell Strategy combines extremely safe positions (80-90% of capital) with small
    allocations to highly asymmetric opportunities with significant upside potential.
    This approach minimizes downside while maintaining exposure to positive Black Swans.

    Key features:
    - Dynamic allocation between safe and speculative positions
    - Adjustments based on market conditions and strategy antifragility
    - Asymmetric risk-reward targeting for speculative portion
    - Hardware-aware optimization for allocation calculations
    """

    def __init__(
        self,
        safe_allocation: float = 0.85,
        speculative_allocation: float = 0.15,
        min_safe_allocation: float = 0.75,
        hardware_layer=None,
    ):
        """
        Initialize the Barbell allocator.

        Args:
            safe_allocation: Default allocation to safe positions (0-1)
            speculative_allocation: Default allocation to speculative positions (0-1)
            min_safe_allocation: Minimum allocation to safe positions (0-1)
            hardware_layer: Optional hardware abstraction layer
        """
        # Validate input parameters
        self._validate_allocations(
            safe_allocation, speculative_allocation, min_safe_allocation
        )

        self.default_safe_allocation = safe_allocation
        self.default_spec_allocation = speculative_allocation
        self.min_safe_allocation = min_safe_allocation
        self.hardware_layer = hardware_layer

        # Tracking allocations over time
        self.allocation_history = deque(maxlen=1000)

        # Factor weights for allocation adjustments
        self.adjustment_weights = {
            "antifragility": 0.3,
            "volatility": 0.2,
            "anomalies": 0.2,
            "drawdown": 0.3,
        }

        # State tracking
        self.current_safe_allocation = safe_allocation
        self.current_spec_allocation = speculative_allocation
        self.last_allocation_time = time.time()

        # Thread safety
        self._lock = threading.RLock()

        # Asymmetric payoff targets
        self.asymmetric_targets = {
            "bull": {"upside_target": 3.0, "downside_risk": 1.0},  # 3:1 reward-risk
            "bear": {"upside_target": 5.0, "downside_risk": 1.0},  # 5:1 reward-risk
            "volatile": {"upside_target": 4.0, "downside_risk": 1.0},  # 4:1 reward-risk
            "normal": {"upside_target": 2.5, "downside_risk": 1.0},  # 2.5:1 reward-risk
        }

        logger.info(
            f"Barbell Allocator initialized with {safe_allocation:.1%} safe, "
            f"{speculative_allocation:.1%} speculative"
        )

    def _validate_allocations(
        self,
        safe_allocation: float,
        speculative_allocation: float,
        min_safe_allocation: float,
    ) -> None:
        """
        Validate allocation parameters.

        Args:
            safe_allocation: Allocation to safe positions
            speculative_allocation: Allocation to speculative positions
            min_safe_allocation: Minimum allocation to safe positions

        Raises:
            ValueError: If allocations are invalid
        """
        if not 0 <= safe_allocation <= 1:
            raise ValueError(
                f"safe_allocation must be between 0 and 1, got {safe_allocation}"
            )

        if not 0 <= speculative_allocation <= 1:
            raise ValueError(
                f"speculative_allocation must be between 0 and 1, got {speculative_allocation}"
            )

        if abs(safe_allocation + speculative_allocation - 1.0) > 1e-6:
            raise ValueError(
                f"safe_allocation and speculative_allocation must sum to 1.0, "
                f"got {safe_allocation + speculative_allocation}"
            )

        if not 0 <= min_safe_allocation <= safe_allocation:
            raise ValueError(
                f"min_safe_allocation must be between 0 and safe_allocation, "
                f"got {min_safe_allocation} with safe_allocation={safe_allocation}"
            )

    def calculate_allocation(
        self,
        dataframe: pd.DataFrame,
        antifragility: float,
        anomalies: Dict[str, Any],
        drawdown: float = 0.0,
    ) -> Tuple[float, float]:
        """
        Calculate optimal barbell allocation based on market conditions.

        Args:
            dataframe: Market data
            antifragility: Strategy's antifragility score (0-1)
            anomalies: Detected market anomalies
            drawdown: Current drawdown level (0-1)

        Returns:
            Tuple of (safe_allocation, speculative_allocation)
        """
        try:
            with self._lock:
                # Starting with default allocations
                safe = self.default_safe_allocation
                spec = self.default_spec_allocation

                # Extract market regime
                regime = self._extract_regime(dataframe)

                # 1. Adjust based on antifragility
                antifragility_adjustment = self._calculate_antifragility_adjustment(
                    antifragility
                )

                # 2. Adjust based on volatility
                volatility = self._extract_volatility(dataframe)
                volatility_adjustment = self._calculate_volatility_adjustment(
                    volatility
                )

                # 3. Adjust based on market anomalies
                anomaly_adjustment = self._calculate_anomaly_adjustment(anomalies)

                # 4. Adjust based on drawdown
                drawdown_adjustment = self._calculate_drawdown_adjustment(drawdown)

                # Combine adjustments with weights
                weights = self.adjustment_weights
                safe_adjustment = (
                    weights["antifragility"] * antifragility_adjustment
                    + weights["volatility"] * volatility_adjustment
                    + weights["anomalies"] * anomaly_adjustment
                    + weights["drawdown"] * drawdown_adjustment
                )

                # Apply adjustment to safe allocation
                safe += safe_adjustment

                # Ensure we maintain minimum safe allocation
                safe = max(self.min_safe_allocation, min(0.95, safe))

                # Speculative allocation is the remainder
                spec = 1.0 - safe

                # Store current allocations
                self.current_safe_allocation = safe
                self.current_spec_allocation = spec

                # Add to history
                timestamp = pd.Timestamp.now().isoformat()
                self.allocation_history.append(
                    {
                        "timestamp": timestamp,
                        "safe_allocation": safe,
                        "spec_allocation": spec,
                        "regime": regime,
                        "antifragility": antifragility,
                        "volatility": volatility,
                        "anomalies": anomalies.get("anomalies_detected", False),
                        "drawdown": drawdown,
                    }
                )

                # Log significant changes
                if abs(safe - self.default_safe_allocation) > 0.05:
                    logger.info(
                        f"Barbell allocation adjusted: {safe:.1%} safe, {spec:.1%} speculative "
                        f"(regime: {regime}, antifragility: {antifragility:.2f})"
                    )

                return safe, spec

        except Exception as e:
            logger.error(f"Error calculating barbell allocation: {e}")
            return self.default_safe_allocation, self.default_spec_allocation

    def _extract_regime(self, dataframe: pd.DataFrame) -> str:
        """
        Extract market regime from dataframe.

        Args:
            dataframe: Market data

        Returns:
            str: Market regime
        """
        # Check if regime is directly available
        if "regime" in dataframe.columns:
            regime = dataframe["regime"].iloc[-1]
            if not pd.isna(regime):
                return regime

        # Simple regime detection if not provided
        try:
            if "close" in dataframe.columns and len(dataframe) > 20:
                # Calculate returns
                returns = dataframe["close"].pct_change().dropna()
                volatility = returns.std()
                recent_return = returns.tail(20).mean() * 20

                if volatility > 0.03:  # High volatility
                    return "volatile"
                elif recent_return > 0.05:  # Strong uptrend
                    return "bull"
                elif recent_return < -0.05:  # Strong downtrend
                    return "bear"
                else:
                    return "normal"
        except Exception:
            pass

        return "normal"  # Default to normal regime

    def _extract_volatility(self, dataframe: pd.DataFrame) -> float:
        """
        Extract volatility from dataframe.

        Args:
            dataframe: Market data

        Returns:
            float: Volatility measure
        """
        # Check if volatility is directly available
        if "volatility" in dataframe.columns:
            vol = dataframe["volatility"].iloc[-1]
            if not pd.isna(vol):
                return vol

        # Calculate from returns if not provided
        try:
            if "close" in dataframe.columns and len(dataframe) >= 20:
                returns = dataframe["close"].pct_change().dropna()
                return returns.tail(20).std()
        except Exception:
            pass

        return 0.02  # Default moderate volatility

    def _calculate_antifragility_adjustment(self, antifragility: float) -> float:
        """
        Calculate allocation adjustment based on antifragility.

        Higher antifragility allows for more speculative allocation
        (negative adjustment to safe allocation).

        Args:
            antifragility: Antifragility score (0-1)

        Returns:
            float: Adjustment to safe allocation
        """
        # Neutral point is 0.5
        # - Above 0.5: reduce safe allocation (more speculative)
        # - Below 0.5: increase safe allocation (more conservative)
        return (0.5 - antifragility) * 0.2  # Scale to a reasonable adjustment range

    def _calculate_volatility_adjustment(self, volatility: float) -> float:
        """
        Calculate allocation adjustment based on market volatility.

        Higher volatility increases safe allocation for fragile strategies,
        but may decrease it for antifragile ones.

        Args:
            volatility: Market volatility measure

        Returns:
            float: Adjustment to safe allocation
        """
        # Map volatility to adjustment
        # - Normal volatility (0.01-0.02): No adjustment
        # - Low volatility (<0.01): Slight decrease in safe allocation
        # - High volatility (>0.02): Increase in safe allocation

        if volatility < 0.01:  # Low volatility
            return -0.05  # Reduce safe allocation slightly
        elif volatility > 0.03:  # High volatility
            return 0.10  # Increase safe allocation
        elif volatility > 0.02:  # Moderate-high volatility
            return 0.05  # Small increase in safe allocation
        else:  # Normal volatility
            return 0.0  # No adjustment

    def _calculate_anomaly_adjustment(self, anomalies: Dict[str, Any]) -> float:
        """
        Calculate allocation adjustment based on market anomalies.

        Detected anomalies generally increase safe allocation
        unless the strategy is highly antifragile.

        Args:
            anomalies: Detected market anomalies

        Returns:
            float: Adjustment to safe allocation
        """
        # No anomalies: no adjustment
        if not anomalies.get("anomalies_detected", False):
            return 0.0

        # Get anomaly confidence and types
        confidence = anomalies.get("confidence", 0.5)
        anomaly_types = anomalies.get("anomaly_types", [])

        # Base adjustment on confidence
        adjustment = confidence * 0.2  # Scale to reasonable range

        # Specific anomaly types
        if "black_swan" in anomaly_types or "tail_event" in anomaly_types:
            # Black swan events warrant larger adjustment
            adjustment += 0.1

        elif "high_volatility" in anomaly_types:
            # Already handled by volatility adjustment, reduce to avoid double-counting
            adjustment *= 0.5

        return adjustment

    def _calculate_drawdown_adjustment(self, drawdown: float) -> float:
        """
        Calculate allocation adjustment based on current drawdown.

        Higher drawdown increases safe allocation to protect capital.

        Args:
            drawdown: Current drawdown level (0-1)

        Returns:
            float: Adjustment to safe allocation
        """
        # No adjustment for small drawdowns
        if drawdown < 0.05:
            return 0.0

        # Progressive adjustment as drawdown increases
        if drawdown < 0.10:
            return 0.05  # 5-10% drawdown
        elif drawdown < 0.15:
            return 0.10  # 10-15% drawdown
        elif drawdown < 0.20:
            return 0.15  # 15-20% drawdown
        else:
            return 0.20  # >20% drawdown

    def get_asymmetric_targets(self, regime: str) -> Dict[str, float]:
        """
        Get asymmetric risk-reward targets for the speculative portion.

        Args:
            regime: Current market regime

        Returns:
            Dict with upside target and downside risk
        """
        return self.asymmetric_targets.get(regime, self.asymmetric_targets["normal"])

    def get_allocation_history(self) -> List[Dict[str, Any]]:
        """
        Get history of allocation decisions.

        Returns:
            List of allocation records
        """
        with self._lock:
            return list(self.allocation_history)

    def set_asymmetric_target(
        self, regime: str, upside_target: float, downside_risk: float
    ) -> None:
        """
        Set asymmetric risk-reward target for a specific regime.

        Args:
            regime: Market regime
            upside_target: Target reward multiple
            downside_risk: Risk multiple (usually 1.0)
        """
        if upside_target <= downside_risk:
            raise ValueError(f"Upside target must be greater than downside risk")

        with self._lock:
            self.asymmetric_targets[regime] = {
                "upside_target": upside_target,
                "downside_risk": downside_risk,
            }

            logger.info(
                f"Set {regime} regime asymmetric target: {upside_target}:{downside_risk} "
                f"reward-risk ratio"
            )


class LuckVsSkillAnalyzer:
    """
    Implements Taleb's concept from "Fooled by Randomness" - distinguishing
    between luck and skill in trading performance.

    This analyzer uses Monte Carlo simulations and statistical methods to estimate
    what percentage of observed performance might be attributable to luck versus skill,
    helping to prevent overconfidence and strategy overfitting.

    Key features:
    - Monte Carlo simulation of random trading to establish baseline
    - Statistical significance testing of strategy performance
    - Confidence adjustment based on sample size
    - Randomized entry/exit timing tests
    """

    def __init__(
        self,
        monte_carlo_simulations: int = 1000,
        confidence_level: float = 0.95,
        min_trades_for_analysis: int = 20,
    ):
        """
        Initialize the luck vs. skill analyzer.

        Args:
            monte_carlo_simulations: Number of simulations to run
            confidence_level: Statistical confidence level
            min_trades_for_analysis: Minimum trades required for meaningful analysis
        """
        self.monte_carlo_simulations = monte_carlo_simulations
        self.confidence_level = confidence_level
        self.min_trades_for_analysis = min_trades_for_analysis

        # Performance tracking
        self.performance_history = []
        self.simulation_results = []
        self.last_analysis_time = time.time()
        self.last_luck_factor = 0.5  # Starting point

        # Random state for reproducibility
        self.random_state = np.random.RandomState(42)

        # Thread safety
        self._lock = threading.RLock()

        logger.info(
            f"Luck vs. Skill Analyzer initialized with {monte_carlo_simulations} simulations"
        )

    def estimate_luck_component(self, dataframe: pd.DataFrame) -> float:
        """
        Estimate what percentage of performance might be luck vs. skill.

        Args:
            dataframe: Market data with strategy performance if available

        Returns:
            float: Luck factor (0-1 scale where higher values indicate more luck)
        """
        with self._lock:
            # Check if we have enough data and should run the analysis
            current_time = time.time()

            # Performance optimization: only run full analysis every hour
            # or when explicitly requested (when dataframe contains performance data)
            if (
                current_time - self.last_analysis_time < 3600
                and "strategy_returns" not in dataframe.columns
                and "returns" not in dataframe.columns
            ):
                return self.last_luck_factor

            # Reset timer
            self.last_analysis_time = current_time

            try:
                # Extract or calculate strategy returns
                strategy_returns = self._extract_strategy_returns(dataframe)

                # If we don't have enough data, return last result
                if (
                    strategy_returns is None
                    or len(strategy_returns) < self.min_trades_for_analysis
                ):
                    return self.last_luck_factor

                # Check if we have market returns
                market_returns = self._extract_market_returns(dataframe)

                # Estimate luck component
                luck_factor = self._estimate_luck_factor(
                    strategy_returns, market_returns
                )

                # Update last result
                self.last_luck_factor = luck_factor

                return luck_factor

            except Exception as e:
                logger.error(f"Error estimating luck component: {e}")
                return self.last_luck_factor

    def _extract_strategy_returns(
        self, dataframe: pd.DataFrame
    ) -> Optional[np.ndarray]:
        """
        Extract strategy returns from dataframe.

        Args:
            dataframe: Market data with strategy returns

        Returns:
            numpy.ndarray: Array of strategy returns or None if not available
        """
        # Check direct columns
        for col in ["strategy_returns", "returns", "trade_returns"]:
            if col in dataframe.columns and not dataframe[col].isna().all():
                return dataframe[col].dropna().values

        # Calculate from entry/exit signals if available
        try:
            if (
                "entry" in dataframe.columns
                and "exit" in dataframe.columns
                and "close" in dataframe.columns
                and len(dataframe) > 20
            ):
                # Simplified calculation
                in_trade = False
                entry_price = 0
                returns = []

                for i in range(len(dataframe)):
                    if not in_trade and dataframe["entry"].iloc[i]:
                        in_trade = True
                        entry_price = dataframe["close"].iloc[i]
                    elif in_trade and dataframe["exit"].iloc[i]:
                        in_trade = False
                        exit_price = dataframe["close"].iloc[i]
                        returns.append(exit_price / entry_price - 1)

                if returns:
                    return np.array(returns)
        except Exception as e:
            logger.debug(f"Error calculating returns from signals: {e}")

        # No returns available
        return None

    def _extract_market_returns(self, dataframe: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Extract market returns from dataframe.

        Args:
            dataframe: Market data

        Returns:
            numpy.ndarray: Array of market returns or None if not available
        """
        # Check if market returns are directly available
        if (
            "market_returns" in dataframe.columns
            and not dataframe["market_returns"].isna().all()
        ):
            return dataframe["market_returns"].dropna().values

        # Calculate from close prices
        try:
            if "close" in dataframe.columns and len(dataframe) > 1:
                returns = dataframe["close"].pct_change().dropna().values
                return returns
        except Exception as e:
            logger.debug(f"Error calculating market returns: {e}")

        return None

    def _estimate_luck_factor(
        self, strategy_returns: np.ndarray, market_returns: Optional[np.ndarray] = None
    ) -> float:
        """
        Estimate luck factor using simulation and statistical tests.

        Args:
            strategy_returns: Array of strategy returns
            market_returns: Optional array of market returns

        Returns:
            float: Luck factor (0-1)
        """
        try:
            # Calculate key performance metrics
            total_return = np.prod(1 + strategy_returns) - 1
            win_rate = np.mean(strategy_returns > 0)
            sharpe = np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-10)

            # Run Monte Carlo simulations
            sim_total_returns = []
            sim_win_rates = []
            sim_sharpes = []

            sample_size = len(strategy_returns)

            for _ in range(self.monte_carlo_simulations):
                # Two approaches to simulation:

                if market_returns is not None and len(market_returns) >= sample_size:
                    # 1. Random entry timing (resample from market returns)
                    indices = self.random_state.choice(
                        len(market_returns) - sample_size, size=1
                    )[0]
                    sim_returns = market_returns[indices : indices + sample_size]
                else:
                    # 2. Bootstrap from strategy returns (preserves distribution)
                    sim_returns = self.random_state.choice(
                        strategy_returns, size=sample_size, replace=True
                    )

                # Calculate performance metrics
                sim_total_returns.append(np.prod(1 + sim_returns) - 1)
                sim_win_rates.append(np.mean(sim_returns > 0))
                sim_sharpes.append(np.mean(sim_returns) / (np.std(sim_returns) + 1e-10))

            # Convert to numpy arrays
            sim_total_returns = np.array(sim_total_returns)
            sim_win_rates = np.array(sim_win_rates)
            sim_sharpes = np.array(sim_sharpes)

            # Calculate percentile ranks (higher percentile = less likely due to luck)
            total_return_percentile = np.mean(sim_total_returns >= total_return)
            win_rate_percentile = np.mean(sim_win_rates >= win_rate)
            sharpe_percentile = np.mean(sim_sharpes >= sharpe)

            # Calculate luck factor (inverse of skill evidence)
            # Average the percentiles with weights
            luck_factor = (
                0.4 * total_return_percentile
                + 0.3 * win_rate_percentile
                + 0.3 * sharpe_percentile
            )

            # Adjust confidence based on sample size
            confidence_adjustment = self.adjust_confidence(luck_factor, sample_size)

            # Blend with prior for stability
            blended_luck_factor = (
                0.7 * confidence_adjustment + 0.3 * self.last_luck_factor
            )

            logger.info(
                f"Luck factor estimate: {blended_luck_factor:.2f} (sample size: {sample_size}, "
                f"total return: {total_return:.2%}, win rate: {win_rate:.2%})"
            )

            return blended_luck_factor

        except Exception as e:
            logger.error(f"Error in luck factor estimation: {e}")
            return self.last_luck_factor

    def adjust_confidence(self, base_confidence: float, sample_size: int) -> float:
        """
        Adjust confidence levels based on statistical significance.

        Args:
            base_confidence: Base confidence level
            sample_size: Number of samples

        Returns:
            float: Adjusted confidence level
        """
        # Smaller sample sizes should reduce confidence due to higher variance
        # The adjustment factor approaches 1.0 as sample size increases
        if sample_size < self.min_trades_for_analysis:
            # Very small sample - high luck component
            return 0.8

        # Square root relationship provides reasonable scaling
        confidence_adjustment = min(1.0, math.sqrt(sample_size) / math.sqrt(100))

        # Apply adjustment:
        # - For high base_confidence (indicating luck), reduce slightly
        # - For low base_confidence (indicating skill), increase more significantly
        if base_confidence > 0.5:
            # High luck estimate - discount less
            adjusted = base_confidence * (1.0 - (1.0 - confidence_adjustment) * 0.5)
        else:
            # Low luck estimate (high skill) - discount more
            adjusted = base_confidence * (1.0 - (1.0 - confidence_adjustment) * 0.8)

        return adjusted

    def add_performance_data(
        self, strategy_returns: np.ndarray, market_returns: np.ndarray
    ) -> None:
        """
        Add performance data for more accurate analysis.

        Args:
            strategy_returns: Array of strategy returns
            market_returns: Array of market returns
        """
        with self._lock:
            self.performance_history.append(
                {
                    "timestamp": pd.Timestamp.now().isoformat(),
                    "strategy_returns": strategy_returns.copy(),
                    "market_returns": market_returns.copy(),
                    "sample_size": len(strategy_returns),
                }
            )

            # Limit history size
            if len(self.performance_history) > 100:
                self.performance_history = self.performance_history[-100:]

    def get_luck_component_breakdown(self) -> Dict[str, float]:
        """
        Get detailed breakdown of the luck component.

        Returns:
            Dict with detailed breakdown of luck factors
        """
        # This would be expanded in a full implementation with
        # detailed attribution of luck vs. skill factors
        return {
            "overall_luck_factor": self.last_luck_factor,
            "confidence_level": self.confidence_level,
            "sample_size_adequacy": min(1.0, len(self.performance_history) / 100),
        }


class ReputationSystem:
    """
    Implements Taleb's "Skin in the Game" principle - system components
    gain or lose influence based on the consequences of their decisions.

    This system encourages honest signaling and meritocracy among components
    by rewarding successful predictions and penalizing failures, with stakes
    proportional to confidence.

    Key features:
    - Component reputation tracking over time
    - Stake-weighted reputation updates
    - Proportional loss penalties based on confidence
    - Diminishing returns for already-established components
    """

    def __init__(self, adaptation_rate: float = 0.05, max_history: int = 1000):
        """
        Initialize the reputation system.

        Args:
            adaptation_rate: Rate at which reputation changes (0-1)
            max_history: Maximum history entries to store
        """
        self.adaptation_rate = adaptation_rate
        self.max_history = max_history
        self.component_reputation = {}
        self.component_stakes = {}
        self.reputation_history = {}

        # Thread safety
        self._lock = threading.RLock()

        logger.info(
            f"Reputation System initialized with adaptation rate {adaptation_rate}"
        )

    def register_component(
        self, component_id: str, initial_reputation: float = 1.0
    ) -> None:
        """
        Register a new component with the reputation system.

        Args:
            component_id: Unique identifier for the component
            initial_reputation: Initial reputation score (usually 1.0)
        """
        if not 0.1 <= initial_reputation <= 2.0:
            raise ValueError(
                f"Initial reputation must be between 0.1 and 2.0, got {initial_reputation}"
            )

        with self._lock:
            self.component_reputation[component_id] = initial_reputation
            self.component_stakes[component_id] = 0.0
            self.reputation_history[component_id] = []

            logger.info(
                f"Registered component {component_id} with reputation {initial_reputation}"
            )

    def place_stake(
        self, component_id: str, stake_amount: float, prediction: Any
    ) -> float:
        """
        Component places a stake on its prediction.

        Args:
            component_id: Identifier of the component
            stake_amount: Amount of "reputation" being staked (0-1)
            prediction: The prediction being made

        Returns:
            float: Current influence level based on reputation
        """
        with self._lock:
            # Auto-register if necessary
            if component_id not in self.component_reputation:
                self.register_component(component_id)

            # Validate stake amount
            stake_amount = max(0.0, min(1.0, stake_amount))

            # Record the stake
            self.component_stakes[component_id] = stake_amount

            # Return current influence level
            return self.component_reputation[component_id]

    def update_reputation(
        self,
        component_id: str,
        outcome_value: float,
        confidence: Optional[float] = None,
    ) -> float:
        """
        Update reputation based on prediction outcome.

        Args:
            component_id: Identifier of the component
            outcome_value: Value representing success/failure (-1 to +1 range)
            confidence: Optional confidence level of the prediction (0-1)

        Returns:
            float: New reputation value
        """
        with self._lock:
            # Get current values
            current_rep = self.component_reputation.get(component_id, 1.0)
            stake = self.component_stakes.get(component_id, 0.0)

            # If confidence provided, use it to scale stake
            if confidence is not None:
                effective_stake = stake * confidence
            else:
                effective_stake = stake

            # Higher stakes mean higher reputation impact
            # Scale impact between 0.5x to 2x the adaptation rate based on stake
            impact_scale = 0.5 + 1.5 * effective_stake
            impact = self.adaptation_rate * impact_scale

            # Calculate reputation change based on outcome
            if outcome_value >= 0:
                # Positive outcome - gain reputation, with diminishing returns
                # The higher the current reputation, the smaller the gain
                diminishing_factor = 2.0 - current_rep / 2.0  # Range from 2.0 to 1.0
                gain = impact * outcome_value * diminishing_factor
                new_rep = current_rep + gain
            else:
                # Negative outcome - lose reputation proportional to stake and confidence
                # Higher stakes and confidence lead to bigger losses
                loss = abs(impact * outcome_value) * current_rep * effective_stake
                new_rep = current_rep + loss  # outcome_value is negative

            # Ensure reputation stays in valid range
            new_rep = max(0.1, min(2.0, new_rep))

            # Update reputation
            self.component_reputation[component_id] = new_rep

            # Track history
            self.reputation_history[component_id].append(
                {
                    "timestamp": pd.Timestamp.now().isoformat(),
                    "reputation": new_rep,
                    "outcome_value": outcome_value,
                    "stake": stake,
                }
            )

            # Limit history size
            if len(self.reputation_history[component_id]) > self.max_history:
                self.reputation_history[component_id] = self.reputation_history[
                    component_id
                ][-self.max_history :]

            # Reset stake
            self.component_stakes[component_id] = 0.0

            logger.debug(
                f"Updated {component_id} reputation: {current_rep:.2f} → {new_rep:.2f} "
                f"(outcome: {outcome_value:.2f}, stake: {stake:.2f})"
            )

            return new_rep

    def get_reputation(self, component_id: str) -> float:
        """
        Get current reputation of a component.

        Args:
            component_id: Identifier of the component

        Returns:
            float: Current reputation value
        """
        with self._lock:
            return self.component_reputation.get(component_id, 1.0)

    def get_all_reputations(self) -> Dict[str, float]:
        """
        Get reputation values for all components.

        Returns:
            Dict mapping component IDs to reputation values
        """
        with self._lock:
            return self.component_reputation.copy()

    def get_reputation_history(self, component_id: str) -> List[Dict[str, Any]]:
        """
        Get reputation history for a component.

        Args:
            component_id: Identifier of the component

        Returns:
            List of reputation history entries
        """
        with self._lock:
            return self.reputation_history.get(component_id, []).copy()

    def calculate_influence_distribution(self) -> Dict[str, float]:
        """
        Calculate normalized influence distribution across components.

        Returns:
            Dict mapping component IDs to influence values (sum to 1)
        """
        with self._lock:
            # Sum total reputation
            total_reputation = sum(self.component_reputation.values())

            if total_reputation <= 0:
                # Equal distribution if no reputation
                influence = {
                    comp_id: 1.0 / len(self.component_reputation)
                    for comp_id in self.component_reputation
                }
            else:
                # Proportional distribution
                influence = {
                    comp_id: rep / total_reputation
                    for comp_id, rep in self.component_reputation.items()
                }

            return influence



class EnhancedMarketAnomalyDetector:
    """
    Advanced market anomaly detection with whale activity tracking,
    black swan identification, and predictive flash crash detection.

    Uses multi-dimensional analysis to identify unusual market conditions
    and protect against sudden adverse movements.
    """

    def __init__(
        self,
        volume_threshold: float = 2.0,
        volatility_threshold: float = 2.5,
        correlation_threshold: float = 0.8,
        use_machine_learning: bool = True,
        whale_params: Optional[WhaleParameters] = None,
        bs_params: Optional[BlackSwanParameters] = None,
         # --- Pass other dependencies like IQAD ---
         iqad_instance=None,
         # --- Params for this class itself ---
         params: Optional[AnomalyDetectorParams] = None,
         cache_size: int = 100,
         log_level: str = "INFO"
    ):
        self.params = params if params is not None else AnomalyDetectorParams()
        self.iqad = iqad_instance # Pass IQAD if needed by detectors
        self.cache_size = cache_size
        self._calculation_cache = {}
        
        self.logger = logger_anomaly
        self.logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        if not self.logger.handlers: # Basic handler setup
            handler = logging.StreamHandler(); formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'); handler.setFormatter(formatter); self.logger.addHandler(handler)
        
        # --- Instantiate Delegated Detectors ---
        self.whale_detector_inst = None
        if WhaleDetector:
             try:
                 # Pass IQAD if WhaleDetector needs it
                 self.whale_detector_inst = WhaleDetector(iqad_instance=self.iqad, params=whale_params, cache_size=cache_size, log_level=log_level)
                 self.logger.info("Internal WhaleDetector instance created.")
             except Exception as e: self.logger.error(f"Failed to create WhaleDetector instance: {e}")
        
        self.black_swan_detector_inst = None
        if BlackSwanDetector:
             try:
                 # Pass IQAD if BlackSwanDetector needs it
                 self.black_swan_detector_inst = BlackSwanDetector(iqad_instance=self.iqad, params=bs_params, cache_size=cache_size, log_level=log_level)
                 self.logger.info("Internal BlackSwanDetector instance created.")
             except Exception as e: self.logger.error(f"Failed to create BlackSwanDetector instance: {e}")
        # --- End Instantiate ---
        
        self.ml_model = None; self.ml_scaler = None # Keep ML logic if desired
        if self.params.use_ml_anomaly: self._initialize_ml_model()
        
        self.logger.info(f"Enhanced Anomaly Detector Initialized (Delegation Mode). Whale: {self.whale_detector_inst is not None}, BS: {self.black_swan_detector_inst is not None}, ML: {self.ml_model is not None}")

        # Validate inputs
        if volume_threshold <= 1:
            logger.warning(f"Volume threshold must be > 1. Setting to default 2.0")
            volume_threshold = 2.0

        if volatility_threshold <= 1:
            logger.warning(f"Volatility threshold must be > 1. Setting to default 2.5")
            volatility_threshold = 2.5

        if correlation_threshold < 0 or correlation_threshold > 1:
            logger.warning(
                f"Correlation threshold must be between 0-1. Setting to default 0.8"
            )
            correlation_threshold = 0.8

        # Detection thresholds
        self.volume_threshold = volume_threshold
        self.volatility_threshold = volatility_threshold
        self.correlation_threshold = correlation_threshold

        # Additional detection parameters
        self.liquidity_threshold = 1.5  # Sudden liquidity drop threshold
        self.momentum_threshold = 3.0  # Extreme momentum threshold
        self.rsi_threshold = 15  # Oversold threshold for RSI
        self.surge_threshold = 0.05  # 5% price surge in single candle

        # Machine learning components (enabled by default)
        self.use_machine_learning = use_machine_learning
        self.ml_model = None
        self.ml_scaler = None

        if use_machine_learning:
            self._initialize_ml_model()

        # Anomaly tracking
        self.detected_anomalies = []
        self.anomaly_indicators = {}

        # Statistical models
        self.distribution_models = {}
        self.correlation_matrix = None

        logger.info(
            f"Enhanced Anomaly Detector initialized with vol={volume_threshold}, "
            f"vola={volatility_threshold}, corr={correlation_threshold}, "
            f"ML={use_machine_learning}"
        )

    def _initialize_ml_model(self):
        """Initialize machine learning model for anomaly detection"""
        try:
            # Try to import required libraries, fail gracefully if not available
            import sklearn
            from sklearn.ensemble import IsolationForest
            from sklearn.preprocessing import StandardScaler

            # Initialize isolation forest for unsupervised anomaly detection
            self.ml_model = IsolationForest(
                n_estimators=100,
                max_samples="auto",
                contamination=0.03,  # Expect 3% of data to be anomalous
                random_state=42,
            )

            # Initialize scaler for feature normalization
            self.ml_scaler = StandardScaler()

            logger.info("Machine learning anomaly detection model initialized")

        except ImportError as e:
            logger.warning(f"Machine learning libraries not available: {e}")
            logger.warning("Disabling machine learning anomaly detection")
            self.use_machine_learning = False

        except Exception as e:
            logger.error(f"Error initializing ML model: {e}")
            self.use_machine_learning = False

    def _clear_cache(self): # Clear internal cache and delegate cache clearing
        self._calculation_cache.clear()
        if hasattr(self.whale_detector_inst, '_clear_cache'): self.whale_detector_inst._clear_cache()
        if hasattr(self.black_swan_detector_inst, '_clear_cache'): self.black_swan_detector_inst._clear_cache()
        self.logger.debug("Anomaly detector caches cleared.")
        
    def detect_flash_crash_risk(
        self, price_series: pd.Series, volume_series: pd.Series, order_book: Dict = None
    ) -> Dict:
        """
        Detect conditions indicating increased risk of a flash crash

        Args:
            price_series: Price data
            volume_series: Volume data
            order_book: Optional order book data

        Returns:
            Dictionary with flash crash risk assessment
        """
        if not isinstance(price_series, pd.Series) or price_series.empty:
            logger.warning(
                "Invalid or empty price series for flash crash risk detection."
            )
            return {"risk_level": "unknown", "probability": 0, "metrics": {}}

        try:
            # Initialize result
            result = {"risk_level": "low", "probability": 0, "metrics": {}}

            # 1. Calculate price momentum and acceleration
            if len(price_series) >= 20:
                # Momentum (rate of change)
                price_roc = price_series.pct_change(5).iloc[-1]
                result["metrics"]["price_momentum"] = price_roc

                # Acceleration (change in momentum)
                roc_series = price_series.pct_change(5)
                acceleration = roc_series.diff(5).iloc[-1]
                result["metrics"]["price_acceleration"] = acceleration

            # 2. Check volume cliff (sudden drop in volume)
            if len(volume_series) >= 20:
                recent_vol_avg = volume_series.tail(5).mean()
                prior_vol_avg = volume_series.iloc[:-5].tail(15).mean()

                if prior_vol_avg > 0:
                    volume_cliff = recent_vol_avg / prior_vol_avg
                    result["metrics"]["volume_cliff"] = volume_cliff

            # 3. Analyze order book imbalance if available
            if order_book is not None:
                try:
                    # Calculate bid-ask imbalance
                    bids = sum(order_book.get("bids", []))
                    asks = sum(order_book.get("asks", []))

                    if asks > 0:
                        book_imbalance = bids / asks
                        result["metrics"]["book_imbalance"] = book_imbalance

                    # Calculate order book depth
                    book_depth = bids + asks
                    result["metrics"]["book_depth"] = book_depth
                except Exception as e:
                    logger.debug(f"Error analyzing order book: {e}")

            # 4. Check for bid-ask spread widening
            if "spread" in order_book:
                result["metrics"]["spread"] = order_book["spread"]

            # Determine flash crash risk
            risk_indicators = []

            # Check price momentum (strong negative momentum)
            if (
                "price_momentum" in result["metrics"]
                and result["metrics"]["price_momentum"] < -0.03
            ):
                risk_indicators.append(
                    f"price_momentum={result['metrics']['price_momentum']:.2f}"
                )

            # Check price acceleration (increasing downward momentum)
            if (
                "price_acceleration" in result["metrics"]
                and result["metrics"]["price_acceleration"] < -0.01
            ):
                risk_indicators.append(
                    f"price_acceleration={result['metrics']['price_acceleration']:.2f}"
                )

            # Check volume cliff (sudden drop in volume)
            if (
                "volume_cliff" in result["metrics"]
                and result["metrics"]["volume_cliff"] < 0.5
            ):
                risk_indicators.append(
                    f"volume_cliff={result['metrics']['volume_cliff']:.2f}"
                )

            # Check order book imbalance (more asks than bids)
            if (
                "book_imbalance" in result["metrics"]
                and result["metrics"]["book_imbalance"] < 0.7
            ):
                risk_indicators.append(
                    f"book_imbalance={result['metrics']['book_imbalance']:.2f}"
                )

            # Check spread widening
            if (
                "spread" in result["metrics"] and result["metrics"]["spread"] > 0.002
            ):  # 0.2% spread
                risk_indicators.append(f"spread={result['metrics']['spread']:.4f}")

            # Calculate risk level and probability
            if len(risk_indicators) >= 3:
                # High risk with multiple indicators
                risk_level = "high"
                probability = min(0.9, 0.5 + 0.1 * len(risk_indicators))
            elif len(risk_indicators) >= 2:
                # Moderate risk
                risk_level = "moderate"
                probability = 0.3 + 0.1 * len(risk_indicators)
            elif len(risk_indicators) >= 1:
                # Low risk with only one indicator
                risk_level = "elevated"
                probability = 0.2
            else:
                # No risk indicators
                risk_level = "low"
                probability = 0.05

            result["risk_level"] = risk_level
            result["probability"] = probability

            if risk_indicators:
                result["indicators"] = risk_indicators

                if risk_level in ["moderate", "high"]:
                    logger.warning(
                        f"Flash crash risk detected: {risk_level} (probability: {probability:.2f}). "
                        f"Indicators: {', '.join(risk_indicators)}"
                    )

            return result

        except Exception as e:
            logger.error(f"Error detecting flash crash risk: {e}")
            return {"risk_level": "unknown", "probability": 0, "metrics": {}}

    def detect_correlation_breakdown(self, returns_dict: Dict[str, pd.Series]) -> Dict:
        """
        Detect breakdowns in asset correlations that could indicate market stress

        Args:
            returns_dict: Dictionary of asset returns series

        Returns:
            Dictionary with correlation breakdown assessment
        """
        if not returns_dict or len(returns_dict) < 3:
            logger.warning("Insufficient assets for correlation breakdown detection.")
            return {"detected": False, "confidence": 0, "metrics": {}}

        try:
            # Initialize result
            result = {"detected": False, "confidence": 0, "metrics": {}}

            # Convert to DataFrame
            returns_df = pd.DataFrame(returns_dict)

            # 1. Calculate current correlation matrix
            current_corr = returns_df.tail(20).corr()

            # 2. Calculate historical correlation matrix
            if len(returns_df) >= 60:
                hist_corr = returns_df.iloc[:-20].tail(40).corr()

                # 3. Measure correlation stability
                # Calculate Frobenius norm of difference matrix
                if len(current_corr) > 0 and len(hist_corr) > 0:
                    diff_matrix = current_corr - hist_corr
                    frob_norm = np.sqrt((diff_matrix**2).sum().sum())
                    result["metrics"]["correlation_change"] = frob_norm

                    # Store correlation matrices
                    self.correlation_matrix = current_corr

            # 4. Calculate average correlation
            if len(current_corr) > 0:
                # Extract upper triangle
                upper_triangle = current_corr.values[
                    np.triu_indices_from(current_corr.values, k=1)
                ]
                avg_correlation = np.mean(upper_triangle)
                result["metrics"]["avg_correlation"] = avg_correlation

                # Calculate correlation dispersion
                corr_dispersion = np.std(upper_triangle)
                result["metrics"]["correlation_dispersion"] = corr_dispersion

            # Determine if correlation breakdown detected
            breakdown_indicators = []

            # Check correlation change
            if (
                "correlation_change" in result["metrics"]
                and result["metrics"]["correlation_change"] > 0.5
            ):
                breakdown_indicators.append(
                    f"correlation_change={result['metrics']['correlation_change']:.2f}"
                )

            # Check average correlation drop
            if (
                "avg_correlation" in result["metrics"]
                and result["metrics"]["avg_correlation"] < 0.3
            ):
                breakdown_indicators.append(
                    f"avg_correlation={result['metrics']['avg_correlation']:.2f}"
                )

            # Check correlation dispersion (high = breakdown)
            if (
                "correlation_dispersion" in result["metrics"]
                and result["metrics"]["correlation_dispersion"] > 0.4
            ):
                breakdown_indicators.append(
                    f"correlation_dispersion={result['metrics']['correlation_dispersion']:.2f}"
                )

            # Final determination
            if len(breakdown_indicators) >= 2:
                confidence = min(0.9, 0.5 + 0.15 * len(breakdown_indicators))
                result["detected"] = True
                result["confidence"] = confidence
                result["indicators"] = breakdown_indicators

                logger.warning(
                    f"Correlation breakdown detected with {confidence:.2f} confidence. "
                    f"Indicators: {', '.join(breakdown_indicators)}"
                )

            return result

        except Exception as e:
            logger.error(f"Error detecting correlation breakdown: {e}")
            return {"detected": False, "confidence": 0, "metrics": {}}

    def detect_liquidity_crisis(
        self, volume_series: pd.Series, spread_series: pd.Series = None
    ) -> Dict:
        """
        Detect liquidity crisis conditions in the market

        Args:
            volume_series: Trading volume data
            spread_series: Optional bid-ask spread data

        Returns:
            Dictionary with liquidity crisis assessment
        """
        if not isinstance(volume_series, pd.Series) or volume_series.empty:
            logger.warning(
                "Invalid or empty volume series for liquidity crisis detection."
            )
            return {"detected": False, "confidence": 0, "metrics": {}}

        try:
            # Initialize result
            result = {"detected": False, "confidence": 0, "metrics": {}}

            # 1. Calculate volume trend
            if len(volume_series) >= 50:
                # Moving average comparison
                recent_vol = volume_series.tail(10).mean()
                hist_vol = volume_series.iloc[:-10].tail(40).mean()

                if hist_vol > 0:
                    volume_trend = recent_vol / hist_vol
                    result["metrics"]["volume_trend"] = volume_trend

                    # Calculate volume volatility
                    vol_volatility = (
                        volume_series.tail(20).std() / volume_series.tail(20).mean()
                    )
                    result["metrics"]["volume_volatility"] = vol_volatility

            # 2. Analyze spread data if available
            if spread_series is not None and len(spread_series) > 0:
                # Calculate spread widening
                if len(spread_series) >= 20:
                    recent_spread = spread_series.tail(5).mean()
                    hist_spread = spread_series.iloc[:-5].tail(15).mean()

                    if hist_spread > 0:
                        spread_widening = recent_spread / hist_spread
                        result["metrics"]["spread_widening"] = spread_widening

            # Determine if liquidity crisis detected
            liquidity_indicators = []

            # Check volume trend (decreasing volume)
            if (
                "volume_trend" in result["metrics"]
                and result["metrics"]["volume_trend"] < 0.7
            ):
                liquidity_indicators.append(
                    f"volume_trend={result['metrics']['volume_trend']:.2f}"
                )

            # Check volume volatility (unstable volume)
            if (
                "volume_volatility" in result["metrics"]
                and result["metrics"]["volume_volatility"] > 0.5
            ):
                liquidity_indicators.append(
                    f"volume_volatility={result['metrics']['volume_volatility']:.2f}"
                )

            # Check spread widening
            if (
                "spread_widening" in result["metrics"]
                and result["metrics"]["spread_widening"] > 1.5
            ):
                liquidity_indicators.append(
                    f"spread_widening={result['metrics']['spread_widening']:.2f}"
                )

            # Final determination
            if len(liquidity_indicators) >= 2:
                confidence = min(0.9, 0.5 + 0.15 * len(liquidity_indicators))
                result["detected"] = True
                result["confidence"] = confidence
                result["indicators"] = liquidity_indicators

                logger.warning(
                    f"Liquidity crisis detected with {confidence:.2f} confidence. "
                    f"Indicators: {', '.join(liquidity_indicators)}"
                )

            return result

        except Exception as e:
            logger.error(f"Error detecting liquidity crisis: {e}")
            return {"detected": False, "confidence": 0, "metrics": {}}

    def detect_ml_anomalies(self, data: pd.DataFrame) -> Dict:
        """
        Detect anomalies using machine learning

        Args:
            data: DataFrame with market data

        Returns:
            Dictionary with ML anomaly detection results
        """
        if not self.use_machine_learning or self.ml_model is None:
            return {"detected": False, "confidence": 0, "scores": []}

        if not isinstance(data, pd.DataFrame) or data.empty or len(data) < 50:
            logger.warning("Invalid or insufficient data for ML anomaly detection.")
            return {"detected": False, "confidence": 0, "scores": []}

        try:
            # Extract features for anomaly detection
            features = self._extract_ml_features(data)

            if features is None or len(features) < 10:
                return {"detected": False, "confidence": 0, "scores": []}

            # Normalize features
            scaled_features = self.ml_scaler.fit_transform(features)

            # Run anomaly detection
            predictions = self.ml_model.predict(scaled_features)
            scores = self.ml_model.decision_function(scaled_features)

            # Convert predictions (-1 for anomalies, 1 for normal)
            anomalies = predictions == -1

            # Check recent data points
            recent_window = 5
            recent_anomalies = anomalies[-recent_window:]
            anomaly_count = sum(recent_anomalies)

            # Prepare result
            result = {
                "detected": anomaly_count > 0,
                "confidence": anomaly_count / recent_window,
                "anomaly_count": anomaly_count,
                "scores": scores[-recent_window:].tolist(),
            }

            if result["detected"]:
                logger.info(
                    f"ML anomaly detected with {result['confidence']:.2f} confidence. "
                    f"Anomaly count: {anomaly_count}/{recent_window}"
                )

            return result

        except Exception as e:
            logger.error(f"Error in ML anomaly detection: {e}")
            return {"detected": False, "confidence": 0, "scores": []}

    def _extract_ml_features(self, data: pd.DataFrame) -> np.ndarray:
        """Extract features for ML anomaly detection"""
        try:
            # Identify available columns
            available_columns = set(data.columns)
            features_list = []

            # Price features
            if "close" in available_columns:
                # Returns at different lookbacks
                for period in [1, 5, 10]:
                    if len(data) > period:
                        returns = data["close"].pct_change(period).fillna(0)
                        features_list.append(returns)

                # Volatility
                if len(data) >= 20:
                    volatility = (
                        data["close"].pct_change().rolling(window=20).std().fillna(0)
                    )
                    features_list.append(volatility)

            # Volume features
            if "volume" in available_columns:
                # Volume changes
                volume_change = data["volume"].pct_change().fillna(0)
                features_list.append(volume_change)

                # Volume relative to moving average
                if len(data) >= 20:
                    vol_ma_ratio = data["volume"] / data["volume"].rolling(
                        window=20
                    ).mean().fillna(1)
                    features_list.append(vol_ma_ratio)

            # Combine features
            if not features_list:
                return None

            features_df = pd.concat(features_list, axis=1)
            features_df.columns = [f"feature_{i}" for i in range(len(features_list))]

            # Handle NaN values
            features_df = features_df.fillna(0)

            return features_df.values

        except Exception as e:
            logger.error(f"Error extracting ML features: {e}")
            return None

    def _calculate_gini(self, array: np.ndarray) -> float:
        """
        Calculate Gini coefficient (measure of inequality/concentration)

        Args:
            array: Numpy array of values

        Returns:
            Gini coefficient (0 = perfect equality, 1 = perfect inequality)
        """
        try:
            # Sort array
            sorted_array = np.sort(array)
            n = len(sorted_array)

            if n <= 1 or np.sum(sorted_array) == 0:
                return 0

            # Calculate cumulative sum of sorted array
            cumsum = np.cumsum(sorted_array)

            # Calculate Gini coefficient using the formula
            # G = (2 * sum(i*x_i) / (n * sum(x_i))) - (n+1)/n
            indices = np.arange(1, n + 1)
            return (2 * np.sum(indices * sorted_array) / (n * np.sum(sorted_array))) - (
                n + 1
            ) / n

        except Exception as e:
            logger.error(f"Error calculating Gini coefficient: {e}")
            return 0

    def detect_anomalies(
        self,
        price_series: pd.Series,
        volume_series: pd.Series,
        additional_data: Dict = None,
    ) -> Dict:
        """
        Comprehensive anomaly detection combining multiple detection methods

        Args:
            price_series: Price data
            volume_series: Volume data
            additional_data: Optional additional market data

        Returns:
            Dictionary with anomaly detection results
        """
        if not isinstance(price_series, pd.Series) or price_series.empty:
            logger.warning("Invalid or empty price series for anomaly detection.")
            return {"anomalies_detected": False, "metrics": {}}

        if not isinstance(volume_series, pd.Series) or volume_series.empty:
            logger.warning("Invalid or empty volume series for anomaly detection.")
            return {"anomalies_detected": False, "metrics": {}}

        try:
            # Run individual anomaly detectors
            whale_result = self.detect_whale_activity(volume_series, price_series)
            black_swan_result = self.detect_black_swan(price_series, volume_series)

            # Optional detectors if additional data provided
            flash_crash_result = None
            correlation_result = None
            liquidity_result = None
            ml_result = None

            if additional_data:
                # Run flash crash detector if order book data available
                if "order_book" in additional_data:
                    flash_crash_result = self.detect_flash_crash_risk(
                        price_series, volume_series, additional_data["order_book"]
                    )

                # Run correlation breakdown detector if multiple assets available
                if "returns_dict" in additional_data:
                    correlation_result = self.detect_correlation_breakdown(
                        additional_data["returns_dict"]
                    )

                # Run liquidity crisis detector if spread data available
                if "spread_series" in additional_data:
                    liquidity_result = self.detect_liquidity_crisis(
                        volume_series, additional_data["spread_series"]
                    )

                # Run ML-based anomaly detection if market data available
                if "market_data" in additional_data:
                    ml_result = self.detect_ml_anomalies(additional_data["market_data"])

            # Combine results
            result = {
                "anomalies_detected": False,
                "anomaly_types": [],
                "confidence": 0,
                "metrics": {},
                "detailed_results": {},
            }

            # Store detailed results
            result["detailed_results"]["whale_activity"] = whale_result
            result["detailed_results"]["black_swan"] = black_swan_result

            if flash_crash_result:
                result["detailed_results"]["flash_crash"] = flash_crash_result

            if correlation_result:
                result["detailed_results"]["correlation_breakdown"] = correlation_result

            if liquidity_result:
                result["detailed_results"]["liquidity_crisis"] = liquidity_result

            if ml_result:
                result["detailed_results"]["ml_anomalies"] = ml_result

            # Determine if any anomalies detected
            anomaly_types = []
            confidence_scores = []

            # Check whale activity
            if whale_result.get("detected", False):
                anomaly_types.append("whale_activity")
                confidence_scores.append(whale_result.get("confidence", 0))

                # Add whale activity metrics to overall metrics
                for key, value in whale_result.get("metrics", {}).items():
                    result["metrics"][f"whale_{key}"] = value

            # Check black swan
            if black_swan_result.get("detected", False):
                anomaly_types.append(
                    f"black_swan_{black_swan_result.get('severity', 'moderate')}"
                )
                confidence_scores.append(black_swan_result.get("confidence", 0))

                # Add black swan metrics to overall metrics
                for key, value in black_swan_result.get("metrics", {}).items():
                    result["metrics"][f"black_swan_{key}"] = value

            # Check flash crash risk
            if flash_crash_result and flash_crash_result.get(
                "risk_level", "low"
            ) not in [
                "low",
                "unknown",
            ]:
                anomaly_types.append(
                    f"flash_crash_risk_{flash_crash_result.get('risk_level')}"
                )
                confidence_scores.append(flash_crash_result.get("probability", 0))

                # Add flash crash metrics to overall metrics
                for key, value in flash_crash_result.get("metrics", {}).items():
                    result["metrics"][f"flash_crash_{key}"] = value

            # Check correlation breakdown
            if correlation_result and correlation_result.get("detected", False):
                anomaly_types.append("correlation_breakdown")
                confidence_scores.append(correlation_result.get("confidence", 0))

                # Add correlation metrics to overall metrics
                for key, value in correlation_result.get("metrics", {}).items():
                    result["metrics"][f"correlation_{key}"] = value

            # Check liquidity crisis
            if liquidity_result and liquidity_result.get("detected", False):
                anomaly_types.append("liquidity_crisis")
                confidence_scores.append(liquidity_result.get("confidence", 0))

                # Add liquidity metrics to overall metrics
                for key, value in liquidity_result.get("metrics", {}).items():
                    result["metrics"][f"liquidity_{key}"] = value

            # Check ML anomalies
            if ml_result and ml_result.get("detected", False):
                anomaly_types.append("ml_anomaly")
                confidence_scores.append(ml_result.get("confidence", 0))

                # Add ML metrics to overall metrics
                if "scores" in ml_result:
                    result["metrics"]["ml_anomaly_scores"] = ml_result["scores"]

            # Finalize result
            if anomaly_types:
                result["anomalies_detected"] = True
                result["anomaly_types"] = anomaly_types

                # Calculate overall confidence as weighted average
                if confidence_scores:
                    result["confidence"] = sum(confidence_scores) / len(
                        confidence_scores
                    )

                # Track detected anomalies for historical reference
                timestamp = datetime.now().isoformat()
                anomaly_record = {
                    "timestamp": timestamp,
                    "anomaly_types": anomaly_types,
                    "confidence": result["confidence"],
                    "metrics": result["metrics"],
                }
                self.detected_anomalies.append(anomaly_record)

                # Log detected anomalies
                logger.warning(
                    f"Market anomalies detected: {', '.join(anomaly_types)}. "
                    f"Overall confidence: {result['confidence']:.2f}"
                )

            return result

        except Exception as e:
            logger.error(f"Error in comprehensive anomaly detection: {e}")
            return {"anomalies_detected": False, "metrics": {}}

    def get_historical_anomalies(self, days_back: int = 7) -> List[Dict]:
        """
        Retrieve historical anomaly detections

        Args:
            days_back: Number of days to look back

        Returns:
            List of historical anomaly records
        """
        try:
            # Calculate cutoff timestamp
            cutoff = datetime.now() - timedelta(days=days_back)
            cutoff_str = cutoff.isoformat()

            # Filter anomalies by timestamp
            recent_anomalies = [
                anomaly
                for anomaly in self.detected_anomalies
                if anomaly["timestamp"] >= cutoff_str
            ]

            return recent_anomalies

        except Exception as e:
            logger.error(f"Error retrieving historical anomalies: {e}")
            return []

    def get_anomaly_summary(self) -> Dict:
        """
        Get summary statistics of detected anomalies

        Returns:
            Dictionary with anomaly summary statistics
        """
        try:
            if not self.detected_anomalies:
                return {"total_anomalies": 0}

            # Count anomalies by type
            anomaly_counts = {}

            for record in self.detected_anomalies:
                for anomaly_type in record["anomaly_types"]:
                    if anomaly_type not in anomaly_counts:
                        anomaly_counts[anomaly_type] = 0
                    anomaly_counts[anomaly_type] += 1

            # Calculate average confidence
            avg_confidence = sum(
                record["confidence"] for record in self.detected_anomalies
            ) / len(self.detected_anomalies)

            # Get most recent anomaly
            most_recent = max(self.detected_anomalies, key=lambda x: x["timestamp"])

            # Prepare summary
            summary = {
                "total_anomalies": len(self.detected_anomalies),
                "anomaly_counts": anomaly_counts,
                "average_confidence": avg_confidence,
                "most_recent": {
                    "timestamp": most_recent["timestamp"],
                    "types": most_recent["anomaly_types"],
                    "confidence": most_recent["confidence"],
                },
            }

            return summary

        except Exception as e:
            logger.error(f"Error generating anomaly summary: {e}")
            return {"total_anomalies": 0, "error": str(e)}
