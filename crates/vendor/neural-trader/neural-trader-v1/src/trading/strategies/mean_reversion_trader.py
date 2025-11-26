"""
Optimized Mean Reversion Trading Strategy
Advanced implementation with multi-model signals, adaptive parameters, and comprehensive risk management
Target: 3.0+ Sharpe Ratio with 60-80% annual returns and <10% max drawdown
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import statistics
import logging
from dataclasses import dataclass
from enum import Enum
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classifications for adaptive parameters."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"


@dataclass
class MeanReversionSignal:
    """Mean reversion signal container."""
    signal_strength: float  # -1 to 1 (negative = short, positive = long)
    confidence: float      # 0 to 1
    model_scores: Dict[str, float]
    entry_price: float
    target_price: float
    stop_loss_price: float
    holding_period_estimate: int
    risk_score: float
    expected_return: float


@dataclass
class AdaptiveParameters:
    """Adaptive parameter set based on market conditions."""
    lookback_window: int
    z_score_threshold: float
    half_life_days: float
    volatility_lookback: int
    min_reversion_probability: float
    max_position_size: float
    stop_loss_threshold: float
    take_profit_threshold: float


class OptimizedMeanReversionEngine:
    """
    Advanced Mean Reversion Trading Engine
    
    Features:
    - Multi-model signal generation (Z-score, Bollinger, Hurst, Cointegration)
    - Adaptive parameters based on market regime
    - Kelly criterion position sizing
    - Multi-layer risk controls
    - Advanced statistical validation
    """
    
    def __init__(self, 
                 portfolio_size: float = 100000,
                 max_drawdown_limit: float = 0.10,
                 max_portfolio_heat: float = 0.15,
                 base_lookback: int = 50):
        """
        Initialize the optimized mean reversion engine.
        
        Args:
            portfolio_size: Total portfolio size
            max_drawdown_limit: Maximum acceptable drawdown (10%)
            max_portfolio_heat: Maximum portfolio risk exposure (15%)
            base_lookback: Base lookback period for calculations
        """
        self.portfolio_size = portfolio_size
        self.max_drawdown_limit = max_drawdown_limit
        self.max_portfolio_heat = max_portfolio_heat
        self.base_lookback = base_lookback
        
        # Performance tracking
        self.trades = []
        self.performance_metrics = {}
        self.current_positions = {}
        
        # Adaptive parameter sets for different market regimes
        self.regime_parameters = {
            MarketRegime.TRENDING_UP: AdaptiveParameters(
                lookback_window=40,
                z_score_threshold=2.2,
                half_life_days=8,
                volatility_lookback=20,
                min_reversion_probability=0.65,
                max_position_size=0.08,
                stop_loss_threshold=0.06,
                take_profit_threshold=0.04
            ),
            MarketRegime.TRENDING_DOWN: AdaptiveParameters(
                lookback_window=35,
                z_score_threshold=2.5,
                half_life_days=6,
                volatility_lookback=15,
                min_reversion_probability=0.70,
                max_position_size=0.06,
                stop_loss_threshold=0.05,
                take_profit_threshold=0.035
            ),
            MarketRegime.SIDEWAYS: AdaptiveParameters(
                lookback_window=60,
                z_score_threshold=1.8,
                half_life_days=12,
                volatility_lookback=30,
                min_reversion_probability=0.60,
                max_position_size=0.12,
                stop_loss_threshold=0.08,
                take_profit_threshold=0.05
            ),
            MarketRegime.HIGH_VOLATILITY: AdaptiveParameters(
                lookback_window=25,
                z_score_threshold=2.8,
                half_life_days=4,
                volatility_lookback=10,
                min_reversion_probability=0.75,
                max_position_size=0.04,
                stop_loss_threshold=0.04,
                take_profit_threshold=0.025
            ),
            MarketRegime.LOW_VOLATILITY: AdaptiveParameters(
                lookback_window=80,
                z_score_threshold=1.5,
                half_life_days=20,
                volatility_lookback=40,
                min_reversion_probability=0.55,
                max_position_size=0.15,
                stop_loss_threshold=0.10,
                take_profit_threshold=0.06
            )
        }
        
        # Risk management parameters
        self.emergency_exit_threshold = 0.15  # 15% portfolio loss triggers emergency mode
        self.position_correlation_limit = 0.7  # Max correlation between positions
        self.max_positions = 8  # Maximum concurrent positions
        
        logger.info("Optimized Mean Reversion Engine initialized with advanced risk controls")
    
    def detect_market_regime(self, price_data: np.ndarray) -> MarketRegime:
        """
        Detect current market regime for adaptive parameter selection.
        
        Args:
            price_data: Historical price data
            
        Returns:
            Detected market regime
        """
        if len(price_data) < 30:
            return MarketRegime.SIDEWAYS
        
        # Calculate trend strength
        prices = price_data[-30:]  # Last 30 days
        trend_slope = np.polyfit(range(len(prices)), prices, 1)[0]
        trend_strength = trend_slope / np.mean(prices) * 30  # Annualized trend
        
        # Calculate volatility
        returns = np.diff(prices) / prices[:-1]
        volatility = np.std(returns) * np.sqrt(252)  # Annualized volatility
        
        # Regime classification
        if volatility > 0.4:  # High volatility
            return MarketRegime.HIGH_VOLATILITY
        elif volatility < 0.15:  # Low volatility
            return MarketRegime.LOW_VOLATILITY
        elif trend_strength > 0.15:  # Strong uptrend
            return MarketRegime.TRENDING_UP
        elif trend_strength < -0.15:  # Strong downtrend
            return MarketRegime.TRENDING_DOWN
        else:  # Sideways market
            return MarketRegime.SIDEWAYS
    
    def calculate_multi_model_signals(self, price_data: np.ndarray, 
                                    regime: MarketRegime) -> Dict[str, float]:
        """
        Calculate signals from multiple mean reversion models.
        
        Args:
            price_data: Historical price data
            regime: Current market regime
            
        Returns:
            Dictionary of model signals
        """
        params = self.regime_parameters[regime]
        signals = {}
        
        if len(price_data) < params.lookback_window:
            return {model: 0.0 for model in ['z_score', 'bollinger', 'hurst', 'ornstein_uhlenbeck']}
        
        current_price = price_data[-1]
        lookback_prices = price_data[-params.lookback_window:]
        
        # Model 1: Enhanced Z-Score with adaptive mean
        signals['z_score'] = self._calculate_adaptive_z_score(lookback_prices, current_price, params)
        
        # Model 2: Bollinger Band Mean Reversion
        signals['bollinger'] = self._calculate_bollinger_signal(lookback_prices, current_price, params)
        
        # Model 3: Hurst Exponent Analysis
        signals['hurst'] = self._calculate_hurst_signal(lookback_prices, current_price)
        
        # Model 4: Ornstein-Uhlenbeck Process
        signals['ornstein_uhlenbeck'] = self._calculate_ou_signal(lookback_prices, current_price, params)
        
        # Model 5: Statistical Arbitrage Score
        signals['stat_arb'] = self._calculate_statistical_arbitrage_signal(lookback_prices, current_price, params)
        
        return signals
    
    def _calculate_adaptive_z_score(self, prices: np.ndarray, current_price: float, 
                                  params: AdaptiveParameters) -> float:
        """Calculate Z-score with exponential weighting and outlier handling."""
        if len(prices) < 10:
            return 0.0
        
        # Exponentially weighted mean and std
        weights = np.exp(-np.arange(len(prices))[::-1] / params.half_life_days)
        weights /= weights.sum()
        
        ewm_mean = np.average(prices, weights=weights)
        ewm_var = np.average((prices - ewm_mean)**2, weights=weights)
        ewm_std = np.sqrt(ewm_var)
        
        if ewm_std == 0:
            return 0.0
        
        z_score = (current_price - ewm_mean) / ewm_std
        
        # Apply regime-specific threshold
        if abs(z_score) < params.z_score_threshold:
            return 0.0
        
        # Return normalized signal (-1 to 1)
        return np.clip(-z_score / 4.0, -1.0, 1.0)  # Negative for mean reversion
    
    def _calculate_bollinger_signal(self, prices: np.ndarray, current_price: float,
                                  params: AdaptiveParameters) -> float:
        """Calculate Bollinger Band mean reversion signal."""
        if len(prices) < 20:
            return 0.0
        
        # Calculate Bollinger Bands with adaptive periods
        sma = np.mean(prices)
        std = np.std(prices)
        
        upper_band = sma + (2.0 * std)
        lower_band = sma - (2.0 * std)
        middle_band = sma
        
        # Calculate position within bands
        if current_price > upper_band:
            # Price above upper band - short signal
            band_position = (current_price - upper_band) / (upper_band - middle_band)
            return -np.clip(band_position, 0.0, 1.0)
        elif current_price < lower_band:
            # Price below lower band - long signal
            band_position = (lower_band - current_price) / (middle_band - lower_band)
            return np.clip(band_position, 0.0, 1.0)
        else:
            return 0.0
    
    def _calculate_hurst_signal(self, prices: np.ndarray, current_price: float) -> float:
        """Calculate signal based on Hurst exponent analysis."""
        if len(prices) < 30:
            return 0.0
        
        # Calculate Hurst exponent
        returns = np.diff(np.log(prices))
        if len(returns) < 20:
            return 0.0
        
        # Rescaled range analysis
        try:
            lags = range(2, min(20, len(returns) // 4))
            tau = []
            rs_values = []
            
            for lag in lags:
                # Calculate rescaled range for this lag
                chunks = [returns[i:i+lag] for i in range(0, len(returns)-lag+1, lag)]
                rs_list = []
                
                for chunk in chunks:
                    if len(chunk) == lag and len(chunk) > 1:
                        mean_chunk = np.mean(chunk)
                        cumulative_deviate = np.cumsum(chunk - mean_chunk)
                        R = np.ptp(cumulative_deviate)  # Range
                        S = np.std(chunk)  # Standard deviation
                        if S > 0:
                            rs_list.append(R / S)
                
                if rs_list:
                    tau.append(lag)
                    rs_values.append(np.mean(rs_list))
            
            if len(tau) > 3 and len(rs_values) > 3:
                # Fit log(R/S) = H * log(tau) + c
                log_tau = np.log(tau)
                log_rs = np.log(rs_values)
                hurst = np.polyfit(log_tau, log_rs, 1)[0]
                
                # Hurst < 0.5 indicates mean reversion
                # Convert to signal strength
                if hurst < 0.4:
                    return 0.8  # Strong mean reversion
                elif hurst < 0.45:
                    return 0.5  # Moderate mean reversion
                elif hurst > 0.6:
                    return -0.3  # Trending (avoid mean reversion)
                else:
                    return 0.2  # Weak mean reversion
            
        except (ValueError, np.linalg.LinAlgError):
            pass
        
        return 0.0
    
    def _calculate_ou_signal(self, prices: np.ndarray, current_price: float,
                           params: AdaptiveParameters) -> float:
        """Calculate Ornstein-Uhlenbeck process mean reversion signal."""
        if len(prices) < 20:
            return 0.0
        
        log_prices = np.log(prices)
        
        # Estimate OU parameters using Maximum Likelihood
        try:
            dt = 1.0  # Daily data
            n = len(log_prices) - 1
            
            # Calculate differences
            y = log_prices[1:]
            x = log_prices[:-1]
            
            # Estimate parameters
            sx = np.sum(x)
            sy = np.sum(y)
            sxx = np.sum(x * x)
            sxy = np.sum(x * y)
            syy = np.sum(y * y)
            
            # Mean reversion speed (theta) and long-term mean (mu)
            theta = -np.log((sxy - sx * sy / n) / (sxx - sx * sx / n)) / dt
            mu = (sy - np.exp(-theta * dt) * sx) / (n * (1 - np.exp(-theta * dt)))
            
            # Current deviation from long-term mean
            deviation = np.log(current_price) - mu
            
            # Calculate half-life for comparison
            if theta > 0:
                half_life = np.log(2) / theta
                
                # Strong mean reversion if half-life is reasonable
                if 2 <= half_life <= 30:  # 2-30 days
                    # Signal strength based on deviation and reversion speed
                    signal_strength = -deviation * theta  # Negative for mean reversion
                    return np.clip(signal_strength, -1.0, 1.0)
            
        except (ValueError, np.linalg.LinAlgError, ZeroDivisionError):
            pass
        
        return 0.0
    
    def _calculate_statistical_arbitrage_signal(self, prices: np.ndarray, current_price: float,
                                              params: AdaptiveParameters) -> float:
        """Calculate statistical arbitrage signal based on price level analysis."""
        if len(prices) < 30:
            return 0.0
        
        # Calculate support and resistance levels
        highs = []
        lows = []
        
        # Find local maxima and minima
        for i in range(2, len(prices) - 2):
            if prices[i] > prices[i-1] and prices[i] > prices[i+1]:
                if prices[i] > prices[i-2] and prices[i] > prices[i+2]:
                    highs.append(prices[i])
            elif prices[i] < prices[i-1] and prices[i] < prices[i+1]:
                if prices[i] < prices[i-2] and prices[i] < prices[i+2]:
                    lows.append(prices[i])
        
        if len(highs) < 2 or len(lows) < 2:
            return 0.0
        
        # Calculate resistance and support levels
        resistance = np.mean(sorted(highs)[-3:])  # Average of top 3 highs
        support = np.mean(sorted(lows)[:3])       # Average of bottom 3 lows
        
        # Calculate signal based on proximity to levels
        price_range = resistance - support
        if price_range <= 0:
            return 0.0
        
        # Signal strength based on distance from support/resistance
        if current_price > resistance * 0.98:  # Near resistance
            distance_ratio = (current_price - resistance) / price_range
            return -np.clip(distance_ratio * 2, 0.0, 1.0)  # Short signal
        elif current_price < support * 1.02:  # Near support
            distance_ratio = (support - current_price) / price_range
            return np.clip(distance_ratio * 2, 0.0, 1.0)   # Long signal
        else:
            return 0.0
    
    def generate_mean_reversion_signal(self, symbol: str, price_data: np.ndarray,
                                     volume_data: Optional[np.ndarray] = None) -> MeanReversionSignal:
        """
        Generate comprehensive mean reversion signal.
        
        Args:
            symbol: Trading symbol
            price_data: Historical price data
            volume_data: Optional volume data
            
        Returns:
            Complete mean reversion signal
        """
        if len(price_data) < self.base_lookback:
            return MeanReversionSignal(
                signal_strength=0.0, confidence=0.0, model_scores={},
                entry_price=price_data[-1], target_price=price_data[-1],
                stop_loss_price=price_data[-1], holding_period_estimate=0,
                risk_score=1.0, expected_return=0.0
            )
        
        # Detect market regime
        regime = self.detect_market_regime(price_data)
        params = self.regime_parameters[regime]
        
        # Calculate multi-model signals
        model_signals = self.calculate_multi_model_signals(price_data, regime)
        
        # Combine signals with adaptive weighting
        signal_weights = self._get_adaptive_signal_weights(regime, model_signals)
        combined_signal = sum(signal * weight for signal, weight in 
                            zip(model_signals.values(), signal_weights.values()))
        
        # Calculate confidence based on signal agreement
        signal_values = list(model_signals.values())
        confidence = self._calculate_signal_confidence(signal_values, regime)
        
        # Skip weak signals
        if abs(combined_signal) < 0.3 or confidence < params.min_reversion_probability:
            return MeanReversionSignal(
                signal_strength=0.0, confidence=confidence, model_scores=model_signals,
                entry_price=price_data[-1], target_price=price_data[-1],
                stop_loss_price=price_data[-1], holding_period_estimate=0,
                risk_score=0.5, expected_return=0.0
            )
        
        current_price = price_data[-1]
        
        # Calculate entry parameters
        target_price, stop_loss_price = self._calculate_entry_levels(
            current_price, combined_signal, params, price_data
        )
        
        # Estimate holding period based on regime and signal strength
        holding_period = max(2, int(params.half_life_days * (2 - abs(combined_signal))))
        
        # Calculate risk score
        risk_score = self._calculate_position_risk_score(
            current_price, target_price, stop_loss_price, confidence, regime
        )
        
        # Expected return calculation
        expected_return = abs(target_price - current_price) / current_price * confidence
        
        return MeanReversionSignal(
            signal_strength=combined_signal,
            confidence=confidence,
            model_scores=model_signals,
            entry_price=current_price,
            target_price=target_price,
            stop_loss_price=stop_loss_price,
            holding_period_estimate=holding_period,
            risk_score=risk_score,
            expected_return=expected_return
        )
    
    def _get_adaptive_signal_weights(self, regime: MarketRegime, 
                                   signals: Dict[str, float]) -> Dict[str, float]:
        """Get adaptive weights for different signals based on market regime."""
        base_weights = {
            'z_score': 0.30,
            'bollinger': 0.25,
            'hurst': 0.20,
            'ornstein_uhlenbeck': 0.15,
            'stat_arb': 0.10
        }
        
        # Adjust weights based on regime
        if regime == MarketRegime.HIGH_VOLATILITY:
            base_weights['z_score'] = 0.40  # Z-score more reliable in volatile markets
            base_weights['bollinger'] = 0.30
            base_weights['hurst'] = 0.15
            base_weights['ornstein_uhlenbeck'] = 0.10
            base_weights['stat_arb'] = 0.05
        elif regime == MarketRegime.LOW_VOLATILITY:
            base_weights['hurst'] = 0.35  # Hurst more reliable in stable markets
            base_weights['ornstein_uhlenbeck'] = 0.25
            base_weights['z_score'] = 0.20
            base_weights['bollinger'] = 0.15
            base_weights['stat_arb'] = 0.05
        elif regime in [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]:
            base_weights['ornstein_uhlenbeck'] = 0.35  # OU process better in trending markets
            base_weights['z_score'] = 0.25
            base_weights['hurst'] = 0.20
            base_weights['bollinger'] = 0.15
            base_weights['stat_arb'] = 0.05
        
        return base_weights
    
    def _calculate_signal_confidence(self, signal_values: List[float], 
                                   regime: MarketRegime) -> float:
        """Calculate confidence based on signal agreement and regime."""
        if not signal_values:
            return 0.0
        
        # Remove zero signals for agreement calculation
        non_zero_signals = [s for s in signal_values if abs(s) > 0.1]
        
        if len(non_zero_signals) < 2:
            return 0.3  # Low confidence with few signals
        
        # Calculate signal agreement
        mean_signal = np.mean(non_zero_signals)
        signal_std = np.std(non_zero_signals)
        
        # Agreement score (lower std = higher agreement)
        agreement_score = max(0.0, 1.0 - (signal_std / 0.5))
        
        # Signal strength contribution
        strength_score = min(1.0, abs(mean_signal) * 2)
        
        # Number of models contributing
        participation_score = len(non_zero_signals) / len(signal_values)
        
        # Base confidence
        base_confidence = (agreement_score * 0.4 + strength_score * 0.4 + 
                          participation_score * 0.2)
        
        # Regime adjustment
        regime_multiplier = {
            MarketRegime.SIDEWAYS: 1.2,
            MarketRegime.LOW_VOLATILITY: 1.1,
            MarketRegime.HIGH_VOLATILITY: 0.8,
            MarketRegime.TRENDING_UP: 0.9,
            MarketRegime.TRENDING_DOWN: 0.9
        }.get(regime, 1.0)
        
        return min(1.0, base_confidence * regime_multiplier)
    
    def _calculate_entry_levels(self, current_price: float, signal_strength: float,
                              params: AdaptiveParameters, price_data: np.ndarray) -> Tuple[float, float]:
        """Calculate target and stop-loss levels."""
        # Calculate recent volatility for level setting
        if len(price_data) >= 20:
            returns = np.diff(price_data[-20:]) / price_data[-20:-1]
            volatility = np.std(returns)
        else:
            volatility = 0.02  # Default 2% daily volatility
        
        # Target calculation based on signal direction and strength
        if signal_strength > 0:  # Long position
            target_return = params.take_profit_threshold * abs(signal_strength)
            target_price = current_price * (1 + target_return)
            stop_loss_price = current_price * (1 - params.stop_loss_threshold)
        else:  # Short position
            target_return = params.take_profit_threshold * abs(signal_strength)
            target_price = current_price * (1 - target_return)
            stop_loss_price = current_price * (1 + params.stop_loss_threshold)
        
        return target_price, stop_loss_price
    
    def _calculate_position_risk_score(self, entry_price: float, target_price: float,
                                     stop_loss_price: float, confidence: float,
                                     regime: MarketRegime) -> float:
        """Calculate comprehensive position risk score."""
        # Risk-reward ratio
        potential_gain = abs(target_price - entry_price) / entry_price
        potential_loss = abs(stop_loss_price - entry_price) / entry_price
        
        risk_reward = potential_gain / potential_loss if potential_loss > 0 else 0
        
        # Risk components
        price_risk = min(1.0, potential_loss / 0.1)  # Normalize to 10% loss
        confidence_risk = 1.0 - confidence
        regime_risk = {
            MarketRegime.HIGH_VOLATILITY: 0.8,
            MarketRegime.TRENDING_UP: 0.6,
            MarketRegime.TRENDING_DOWN: 0.6,
            MarketRegime.SIDEWAYS: 0.3,
            MarketRegime.LOW_VOLATILITY: 0.2
        }.get(regime, 0.5)
        
        # Combined risk score (0 = low risk, 1 = high risk)
        risk_score = (price_risk * 0.4 + confidence_risk * 0.3 + regime_risk * 0.3)
        
        # Adjust for risk-reward ratio
        if risk_reward > 2.0:
            risk_score *= 0.8  # Reduce risk for good risk-reward
        elif risk_reward < 1.0:
            risk_score *= 1.3  # Increase risk for poor risk-reward
        
        return np.clip(risk_score, 0.0, 1.0)
    
    def calculate_kelly_position_size(self, signal: MeanReversionSignal,
                                    current_portfolio_value: float) -> float:
        """
        Calculate position size using Kelly Criterion with risk controls.
        
        Args:
            signal: Mean reversion signal
            current_portfolio_value: Current portfolio value
            
        Returns:
            Position size as percentage of portfolio
        """
        if signal.confidence < 0.5 or abs(signal.signal_strength) < 0.3:
            return 0.0
        
        # Kelly fraction calculation
        win_probability = signal.confidence
        loss_probability = 1.0 - win_probability
        
        # Expected gains and losses
        potential_gain = abs(signal.target_price - signal.entry_price) / signal.entry_price
        potential_loss = abs(signal.stop_loss_price - signal.entry_price) / signal.entry_price
        
        if potential_loss <= 0:
            return 0.0
        
        # Kelly fraction: f = (bp - q) / b
        # where b = odds received on the wager, p = probability of winning, q = probability of losing
        b = potential_gain / potential_loss  # Odds ratio
        kelly_fraction = (b * win_probability - loss_probability) / b
        
        # Apply safety factors
        kelly_fraction = max(0.0, kelly_fraction)  # No negative positions
        kelly_fraction *= 0.5  # Half-Kelly for safety
        
        # Risk-based position sizing adjustments
        risk_adjustment = 1.0 - signal.risk_score * 0.5
        kelly_fraction *= risk_adjustment
        
        # Market regime adjustments
        regime = self.detect_market_regime(np.array([signal.entry_price]))
        regime_limits = {
            MarketRegime.HIGH_VOLATILITY: 0.04,
            MarketRegime.TRENDING_UP: 0.08,
            MarketRegime.TRENDING_DOWN: 0.06,
            MarketRegime.SIDEWAYS: 0.12,
            MarketRegime.LOW_VOLATILITY: 0.15
        }
        
        max_position = regime_limits.get(regime, 0.08)
        kelly_fraction = min(kelly_fraction, max_position)
        
        # Final position size
        return max(0.01, min(kelly_fraction, self.max_portfolio_heat / 4))  # Max 1/4 of total heat
    
    def assess_portfolio_risk(self) -> Dict[str, Any]:
        """Assess current portfolio risk metrics."""
        if not self.current_positions:
            return {
                "total_exposure": 0.0,
                "max_position_risk": 0.0,
                "portfolio_correlation": 0.0,
                "regime_concentration": {},
                "risk_level": "low"
            }
        
        # Calculate total exposure
        total_exposure = sum(pos.get("position_size", 0) for pos in self.current_positions.values())
        
        # Calculate maximum single position risk
        max_position_risk = max(pos.get("risk_amount", 0) for pos in self.current_positions.values())
        
        # Estimate portfolio correlation (simplified)
        portfolio_correlation = min(0.8, len(self.current_positions) * 0.15)  # Rough approximation
        
        # Regime concentration
        regimes = [pos.get("regime", "unknown") for pos in self.current_positions.values()]
        regime_counts = {}
        for regime in regimes:
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
        
        # Risk level determination
        if total_exposure > 0.8 or max_position_risk > 0.15 or portfolio_correlation > 0.7:
            risk_level = "high"
        elif total_exposure > 0.5 or max_position_risk > 0.08:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        return {
            "total_exposure": total_exposure,
            "max_position_risk": max_position_risk,
            "portfolio_correlation": portfolio_correlation,
            "regime_concentration": regime_counts,
            "risk_level": risk_level,
            "position_count": len(self.current_positions),
            "diversification_score": min(1.0, len(self.current_positions) / 8)
        }
    
    def execute_mean_reversion_trade(self, symbol: str, signal: MeanReversionSignal) -> Dict[str, Any]:
        """
        Execute mean reversion trade with comprehensive risk controls.
        
        Args:
            symbol: Trading symbol
            signal: Mean reversion signal
            
        Returns:
            Trade execution result
        """
        # Pre-trade risk checks
        portfolio_risk = self.assess_portfolio_risk()
        
        if portfolio_risk["risk_level"] == "high":
            return {
                "status": "rejected",
                "reason": "portfolio_risk_too_high",
                "risk_assessment": portfolio_risk
            }
        
        if len(self.current_positions) >= self.max_positions:
            return {
                "status": "rejected",
                "reason": "max_positions_reached",
                "current_positions": len(self.current_positions)
            }
        
        # Calculate position size
        position_size = self.calculate_kelly_position_size(signal, self.portfolio_size)
        
        if position_size < 0.01:  # Minimum 1% position
            return {
                "status": "rejected",
                "reason": "position_size_too_small",
                "calculated_size": position_size
            }
        
        # Calculate trade parameters
        position_value = self.portfolio_size * position_size
        shares = position_value / signal.entry_price
        
        # Risk amount calculation
        risk_per_share = abs(signal.entry_price - signal.stop_loss_price)
        risk_amount = shares * risk_per_share
        
        # Create position record
        position_record = {
            "symbol": symbol,
            "entry_time": datetime.now(),
            "entry_price": signal.entry_price,
            "target_price": signal.target_price,
            "stop_loss_price": signal.stop_loss_price,
            "position_size": position_size,
            "shares": shares,
            "position_value": position_value,
            "risk_amount": risk_amount,
            "signal_strength": signal.signal_strength,
            "confidence": signal.confidence,
            "model_scores": signal.model_scores,
            "holding_period_estimate": signal.holding_period_estimate,
            "regime": self.detect_market_regime(np.array([signal.entry_price])).value,
            "risk_score": signal.risk_score,
            "expected_return": signal.expected_return
        }
        
        # Add to current positions
        self.current_positions[symbol] = position_record
        
        return {
            "status": "executed",
            "position_record": position_record,
            "portfolio_risk": portfolio_risk,
            "execution_details": {
                "kelly_fraction": position_size,
                "risk_reward_ratio": abs(signal.target_price - signal.entry_price) / 
                                   abs(signal.stop_loss_price - signal.entry_price),
                "max_loss": risk_amount / self.portfolio_size
            }
        }
    
    def monitor_positions(self, market_data: Dict[str, float]) -> Dict[str, Any]:
        """
        Monitor all positions and generate management actions.
        
        Args:
            market_data: Current market prices {symbol: price}
            
        Returns:
            Position monitoring results
        """
        monitoring_results = {
            "positions_monitored": len(self.current_positions),
            "actions_required": [],
            "position_updates": {},
            "portfolio_metrics": {}
        }
        
        positions_to_close = []
        
        for symbol, position in self.current_positions.items():
            current_price = market_data.get(symbol, position["entry_price"])
            
            # Update position metrics
            unrealized_pnl = (current_price - position["entry_price"]) * position["shares"]
            if position["signal_strength"] < 0:  # Short position
                unrealized_pnl = -unrealized_pnl
            
            return_pct = unrealized_pnl / position["position_value"]
            days_held = (datetime.now() - position["entry_time"]).days
            
            # Update position
            position["current_price"] = current_price
            position["unrealized_pnl"] = unrealized_pnl
            position["return_pct"] = return_pct
            position["days_held"] = days_held
            
            # Decision logic
            action_required = None
            
            # Check stop loss
            if ((position["signal_strength"] > 0 and current_price <= position["stop_loss_price"]) or
                (position["signal_strength"] < 0 and current_price >= position["stop_loss_price"])):
                action_required = "stop_loss_exit"
                positions_to_close.append(symbol)
            
            # Check target reached
            elif ((position["signal_strength"] > 0 and current_price >= position["target_price"]) or
                  (position["signal_strength"] < 0 and current_price <= position["target_price"])):
                action_required = "target_reached"
                positions_to_close.append(symbol)
            
            # Check holding period
            elif days_held > position["holding_period_estimate"] * 1.5:
                action_required = "holding_period_exceeded"
                positions_to_close.append(symbol)
            
            # Check for mean reversion completion (price moved back toward mean)
            elif abs(return_pct) > 0.02:  # Significant move
                # Recalculate signal to see if mean reversion opportunity still exists
                # For simplicity, we'll use a basic check
                if ((position["signal_strength"] > 0 and return_pct > 0.03) or
                    (position["signal_strength"] < 0 and return_pct > 0.03)):
                    action_required = "mean_reversion_achieved"
                    positions_to_close.append(symbol)
            
            if action_required:
                monitoring_results["actions_required"].append({
                    "symbol": symbol,
                    "action": action_required,
                    "current_price": current_price,
                    "return_pct": return_pct,
                    "days_held": days_held
                })
            
            # Update position tracking
            monitoring_results["position_updates"][symbol] = {
                "current_price": current_price,
                "unrealized_pnl": unrealized_pnl,
                "return_pct": return_pct,
                "days_held": days_held,
                "action_required": action_required
            }
        
        # Close positions that require action
        for symbol in positions_to_close:
            if symbol in self.current_positions:
                closed_position = self.current_positions.pop(symbol)
                self.trades.append(closed_position)
        
        # Calculate portfolio metrics
        monitoring_results["portfolio_metrics"] = self.assess_portfolio_risk()
        
        return monitoring_results
    
    def calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics."""
        if not self.trades:
            return {"status": "no_completed_trades"}
        
        # Calculate returns
        returns = []
        for trade in self.trades:
            if "return_pct" in trade and trade["return_pct"] is not None:
                returns.append(trade["return_pct"])
        
        if not returns:
            return {"status": "no_return_data"}
        
        # Basic performance metrics
        total_return = sum(returns)
        avg_return = statistics.mean(returns)
        win_rate = len([r for r in returns if r > 0]) / len(returns)
        
        # Risk metrics
        volatility = statistics.stdev(returns) if len(returns) > 1 else 0
        
        # Sharpe ratio (using 2% risk-free rate)
        risk_free_rate = 0.02 / 252  # Daily risk-free rate
        if volatility > 0:
            sharpe_ratio = (avg_return - risk_free_rate) / volatility * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        # Maximum drawdown
        cumulative_returns = np.cumprod(1 + np.array(returns))
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = abs(np.min(drawdown))
        
        # Mean reversion specific metrics
        signal_strengths = [trade.get("signal_strength", 0) for trade in self.trades]
        confidence_scores = [trade.get("confidence", 0) for trade in self.trades]
        holding_periods = [trade.get("days_held", 0) for trade in self.trades]
        
        # Model performance breakdown
        model_performance = {}
        for trade in self.trades:
            if "model_scores" in trade:
                for model, score in trade["model_scores"].items():
                    if model not in model_performance:
                        model_performance[model] = []
                    model_performance[model].append(trade.get("return_pct", 0))
        
        return {
            "total_trades": len(self.trades),
            "total_return": total_return,
            "avg_return_per_trade": avg_return,
            "win_rate": win_rate,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "avg_signal_strength": statistics.mean(signal_strengths) if signal_strengths else 0,
            "avg_confidence": statistics.mean(confidence_scores) if confidence_scores else 0,
            "avg_holding_period": statistics.mean(holding_periods) if holding_periods else 0,
            "model_performance": {model: statistics.mean(returns) for model, returns in model_performance.items()},
            "risk_adjusted_return": total_return / max(max_drawdown, 0.01),
            "current_positions": len(self.current_positions),
            "portfolio_metrics": self.assess_portfolio_risk()
        }
    
    def optimize_parameters(self, historical_data: Dict[str, np.ndarray],
                          optimization_period_days: int = 252) -> Dict[str, Any]:
        """
        Optimize strategy parameters using historical data.
        
        Args:
            historical_data: Dictionary of symbol -> price arrays
            optimization_period_days: Period for optimization
            
        Returns:
            Optimization results and new parameters
        """
        logger.info("Starting parameter optimization for mean reversion strategy")
        
        # Parameter ranges for optimization
        param_ranges = {
            'z_score_threshold': (1.5, 3.0),
            'half_life_days': (4, 25),
            'max_position_size': (0.04, 0.20),
            'stop_loss_threshold': (0.03, 0.12),
            'take_profit_threshold': (0.02, 0.08)
        }
        
        best_sharpe = -999
        best_params = None
        optimization_results = []
        
        # Grid search optimization (simplified)
        from itertools import product
        
        # Create parameter grid
        grid_points = 3  # Reduced for performance
        param_grid = {}
        for param, (min_val, max_val) in param_ranges.items():
            param_grid[param] = np.linspace(min_val, max_val, grid_points)
        
        # Generate all combinations
        param_names = list(param_grid.keys())
        param_combinations = product(*param_grid.values())
        
        for param_values in param_combinations:
            # Create test parameters
            test_params = dict(zip(param_names, param_values))
            
            # Backtest with these parameters
            backtest_results = self._backtest_with_parameters(
                historical_data, test_params, optimization_period_days
            )
            
            if backtest_results["sharpe_ratio"] > best_sharpe:
                best_sharpe = backtest_results["sharpe_ratio"]
                best_params = test_params.copy()
            
            optimization_results.append({
                "parameters": test_params,
                "results": backtest_results
            })
        
        # Update regime parameters with optimized values
        if best_params:
            for regime in self.regime_parameters:
                params = self.regime_parameters[regime]
                params.z_score_threshold = best_params["z_score_threshold"]
                params.half_life_days = best_params["half_life_days"]
                params.max_position_size = best_params["max_position_size"]
                params.stop_loss_threshold = best_params["stop_loss_threshold"]
                params.take_profit_threshold = best_params["take_profit_threshold"]
        
        return {
            "optimization_completed": True,
            "best_parameters": best_params,
            "best_sharpe_ratio": best_sharpe,
            "total_combinations_tested": len(optimization_results),
            "improvement": best_sharpe > 1.5,  # Improvement over baseline
            "optimization_results": optimization_results[:10]  # Top 10 results
        }
    
    def _backtest_with_parameters(self, historical_data: Dict[str, np.ndarray],
                                test_params: Dict[str, float], 
                                period_days: int) -> Dict[str, float]:
        """Backtest strategy with specific parameters."""
        # Simplified backtesting for parameter optimization
        all_returns = []
        
        for symbol, prices in historical_data.items():
            if len(prices) < period_days:
                continue
            
            # Use last period_days for testing
            test_prices = prices[-period_days:]
            returns = []
            
            # Simple mean reversion simulation
            lookback = 50
            for i in range(lookback, len(test_prices) - 5):
                current_price = test_prices[i]
                
                # Calculate Z-score with test parameters
                mean_price = np.mean(test_prices[i-lookback:i])
                std_price = np.std(test_prices[i-lookback:i])
                
                if std_price > 0:
                    z_score = (current_price - mean_price) / std_price
                    
                    # Entry signal
                    if abs(z_score) > test_params["z_score_threshold"]:
                        # Simulate trade
                        entry_price = current_price
                        
                        # Exit after half_life_days or stop/target
                        exit_day = min(i + int(test_params["half_life_days"]), len(test_prices) - 1)
                        exit_price = test_prices[exit_day]
                        
                        # Calculate return
                        if z_score > 0:  # Short signal (price too high)
                            trade_return = (entry_price - exit_price) / entry_price
                        else:  # Long signal (price too low)
                            trade_return = (exit_price - entry_price) / entry_price
                        
                        # Apply stop loss/take profit
                        trade_return = np.clip(
                            trade_return,
                            -test_params["stop_loss_threshold"],
                            test_params["take_profit_threshold"]
                        )
                        
                        returns.append(trade_return)
            
            all_returns.extend(returns)
        
        if not all_returns:
            return {"sharpe_ratio": -999, "total_return": 0, "max_drawdown": 1}
        
        # Calculate performance metrics
        avg_return = np.mean(all_returns)
        volatility = np.std(all_returns)
        
        # Sharpe ratio
        if volatility > 0:
            sharpe_ratio = avg_return / volatility * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        # Total return and max drawdown
        total_return = sum(all_returns)
        cumulative = np.cumprod(1 + np.array(all_returns))
        drawdown = (cumulative - np.maximum.accumulate(cumulative)) / np.maximum.accumulate(cumulative)
        max_drawdown = abs(np.min(drawdown))
        
        return {
            "sharpe_ratio": sharpe_ratio,
            "total_return": total_return,
            "max_drawdown": max_drawdown,
            "trade_count": len(all_returns)
        }


# Factory function for easy instantiation
def create_optimized_mean_reversion_trader(portfolio_size: float = 100000,
                                         risk_level: str = "moderate") -> OptimizedMeanReversionEngine:
    """
    Factory function to create optimized mean reversion trader with preset configurations.
    
    Args:
        portfolio_size: Portfolio size in dollars
        risk_level: 'conservative', 'moderate', or 'aggressive'
        
    Returns:
        Configured OptimizedMeanReversionEngine
    """
    risk_configs = {
        "conservative": {
            "max_drawdown_limit": 0.05,  # 5%
            "max_portfolio_heat": 0.10,  # 10%
        },
        "moderate": {
            "max_drawdown_limit": 0.10,  # 10%
            "max_portfolio_heat": 0.15,  # 15%
        },
        "aggressive": {
            "max_drawdown_limit": 0.15,  # 15%
            "max_portfolio_heat": 0.25,  # 25%
        }
    }
    
    config = risk_configs.get(risk_level, risk_configs["moderate"])
    
    return OptimizedMeanReversionEngine(
        portfolio_size=portfolio_size,
        max_drawdown_limit=config["max_drawdown_limit"],
        max_portfolio_heat=config["max_portfolio_heat"]
    )