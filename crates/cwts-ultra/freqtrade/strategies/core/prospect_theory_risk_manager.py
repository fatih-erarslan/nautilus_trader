"""
Enhanced Prospect Theory Risk Management System.

This module implements an advanced risk management system based on Nobel Prize-winning
Prospect Theory from behavioral economics. It features adaptive risk parameters,
market phase awareness, and cognitive diversity integration.

Implements concepts from:
- Kahneman, D., & Tversky, A. (1979). Prospect Theory: An Analysis of Decision under Risk.
- Tversky, A., & Kahneman, D. (1992). Advances in prospect theory: Cumulative representation of uncertainty.
- Barberis, N. C. (2013). Thirty years of prospect theory in economics: A review and assessment.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, TypeVar, Protocol, Callable
from enum import Enum, auto
import pandas as pd
from dataclasses import dataclass, field
import warnings
import logging
from functools import lru_cache
import weakref
import time

# Setup logging
logger = logging.getLogger(__name__)

# Type variables and protocols for better type hinting
T = TypeVar('T')
Number = Union[int, float]

class SignalProvider(Protocol):
    """Protocol defining the interface for signal providers."""
    
    def get_signal(self, **kwargs) -> float:
        """Get normalized signal value."""
        ...

class MarketPhase(Enum):
    """Enum representing different market phases in the Panarchy cycle."""
    GROWTH = auto()          # Increasing trend, accumulation
    CONSERVATION = auto()    # Mature trend, distribution
    RELEASE = auto()         # Breakdown, high volatility
    REORGANIZATION = auto()  # Bottoming, early accumulation
    
    def __str__(self) -> str:
        return self.name.lower()
    
    @classmethod
    def from_string(cls, phase_str: str) -> 'MarketPhase':
        """Convert string to MarketPhase, with fuzzy matching."""
        phase_str = phase_str.upper()
        for phase in cls:
            if phase.name.upper() == phase_str:
                return phase
        
        # Fuzzy matching for partial strings
        for phase in cls:
            if phase.name.upper().startswith(phase_str):
                return phase
                
        raise ValueError(f"Unknown market phase: {phase_str}")

@dataclass
class RiskParameters:
    """Dataclass storing risk parameters for easier management and serialization."""
    
    base_stoploss: float = -0.05
    base_take_profit: float = 0.10
    loss_aversion_factor: float = 2.0
    risk_sensitivity: float = 1.0
    dynamic_reference: bool = True
    alpha: float = 0.88  # Diminishing sensitivity for gains
    beta: float = 0.88   # Diminishing sensitivity for losses
    lambda_: float = 2.25  # Loss aversion coefficient
    gamma: float = 0.61  # Probability weighting parameter
    utility_threshold: float = 0.01
    reference_point_mode: str = 'entry' # Default to 'entry'
    reference_point_ma_period: int = 20 # Default MA period
    # Add other thresholds if the PT manager uses them directly from params
    high_vol_threshold: float = 0.05
    high_disp_threshold: float = 0.8
    
    def __post_init__(self):
        """Validate parameters after initialization."""
        if self.base_stoploss > 0:
            warnings.warn("Base stoploss should be negative. Converting to negative.")
            self.base_stoploss = -abs(self.base_stoploss)
            
        if self.base_take_profit < 0:
            warnings.warn("Base take profit should be positive. Converting to positive.")
            self.base_take_profit = abs(self.base_take_profit)
            
        if self.lambda_ < 1:
            warnings.warn("Loss aversion factor should be > 1 according to prospect theory.")

class ProspectTheoryRiskManager:
    """
    Advanced implementation of a risk management system based on Prospect Theory
    and behavioral economics principles.
    
    This class implements:
    1. Value function with asymmetric response to gains and losses
    2. Non-linear probability weighting function
    3. Reference point dependence
    4. Adaptive risk parameters based on market phase
    5. Integration with cognitive diversity metrics
    """
    
    # Class constants
    DEFAULT_PARAMS = RiskParameters()
    MAX_STOPLOSS = -0.01
    MIN_STOPLOSS = -0.25
    MAX_POSITION = 2.0
    MIN_POSITION = 0.1
    
    # Market phase multiplier mappings
    PHASE_STOPLOSS_MULTIPLIERS = {
        MarketPhase.GROWTH: 1.2,         # Wider stops in growth phase
        MarketPhase.CONSERVATION: 0.8,    # Tighter stops in mature trends
        MarketPhase.RELEASE: 0.6,         # Much tighter stops in volatile breakdowns
        MarketPhase.REORGANIZATION: 1.0   # Normal stops in bottoming phase
    }
    
    PHASE_POSITION_MULTIPLIERS = {
        MarketPhase.GROWTH: 1.2,         # Larger positions in growth phase
        MarketPhase.CONSERVATION: 0.8,   # Smaller positions in mature trends
        MarketPhase.RELEASE: 0.5,        # Much smaller in volatility
        MarketPhase.REORGANIZATION: 0.7  # Cautious in bottoming phase
    }
    
    PHASE_TARGETS = {
        MarketPhase.GROWTH: [1.5, 2.0, 3.0],       # Higher targets in growth
        MarketPhase.CONSERVATION: [1.2, 1.6, 2.2], # Moderate in mature trends
        MarketPhase.RELEASE: [1.8, 2.5, 4.0],      # Higher but spread in volatility
        MarketPhase.REORGANIZATION: [1.3, 2.0, 2.8] # Mixed in bottoming
    }
    
    def __init__(self, 
                 params: Optional[RiskParameters] = None,
                 enable_caching: bool = True,
                 cache_size: int = 128):
        """
        Initialize the risk management system with optional configuration parameters.
        
        Args:
            params: Optional RiskParameters object for configuration
            enable_caching: Whether to enable function result caching
            cache_size: Size of LRU cache for function results
        """
        self.params = params if params is not None else self.DEFAULT_PARAMS
        self._initialize_logging()
        
        # Performance metrics
        self._execution_times = {}
        self._call_counts = {}
        
        # Set up caching
        if enable_caching:
            self._setup_caching(cache_size)
            
        # Historical data for self-adapting behavior
        self._trade_history = []
        self._last_update_time = time.time()
        
        # State
        self._current_drawdown = 0.0
        
        logger.info(f"ProspectTheoryRiskManager initialized with parameters: {self.params}")
    
    def _initialize_logging(self):
        """Set up logging configuration."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def _setup_caching(self, cache_size: int):
        """Apply LRU caching to expensive functions."""
        self.calculate_value_function = lru_cache(maxsize=cache_size)(self.calculate_value_function)
        self.calculate_decision_weights = lru_cache(maxsize=cache_size)(self.calculate_decision_weights)
    
    def _time_execution(func):
        """Decorator to track execution time of methods."""
        def wrapper(self, *args, **kwargs):
            start_time = time.time()
            result = func(self, *args, **kwargs)
            execution_time = time.time() - start_time
            
            # Update metrics
            func_name = func.__name__
            if func_name not in self._execution_times:
                self._execution_times[func_name] = []
            if func_name not in self._call_counts:
                self._call_counts[func_name] = 0
                
            self._execution_times[func_name].append(execution_time)
            self._call_counts[func_name] += 1
            
            # Limit stored times to prevent memory growth
            if len(self._execution_times[func_name]) > 100:
                self._execution_times[func_name].pop(0)
                
            return result
        return wrapper
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Return performance metrics for monitoring."""
        metrics = {}
        for func_name, times in self._execution_times.items():
            if times:
                metrics[func_name] = {
                    'avg_time': sum(times) / len(times),
                    'max_time': max(times),
                    'min_time': min(times),
                    'call_count': self._call_counts.get(func_name, 0)
                }
        return metrics
        
    @_time_execution
    def calculate_value_function(self, 
                               x: float, 
                               alpha: Optional[float] = None,
                               beta: Optional[float] = None,
                               lambda_: Optional[float] = None) -> float:
        """
        Calculate Prospect Theory value function.
        
        The value function captures:
        1. Reference dependence - outcomes evaluated as gains/losses from reference point
        2. Loss aversion - steeper slope for losses than gains
        3. Diminishing sensitivity - concave for gains, convex for losses
        
        v(x) = x^alpha for x >= 0
        v(x) = -lambda * (-x)^beta for x < 0
        
        Args:
            x: Input value (gain or loss)
            alpha: Diminishing sensitivity for gains (default: from params)
            beta: Diminishing sensitivity for losses (default: from params)
            lambda_: Loss aversion parameter (default: from params)
            
        Returns:
            Subjective value according to Prospect Theory
        """
        # Use instance parameters if not specified
        alpha = alpha if alpha is not None else self.params.alpha
        beta = beta if beta is not None else self.params.beta
        lambda_ = lambda_ if lambda_ is not None else self.params.lambda_
        
        # Value function from Prospect Theory
        if x >= 0:
            return x ** alpha
        else:
            return -lambda_ * ((-x) ** beta)
    
    @_time_execution
    def calculate_decision_weights(self, 
                                 probabilities: List[float], 
                                 gamma: Optional[float] = None) -> List[float]:
        """
        Calculate decision weights using probability weighting function.
        
        People overweight small probabilities and underweight large probabilities.
        This function implements Prelec's probability weighting function:
        
        w(p) = p^gamma / (p^gamma + (1-p)^gamma)^(1/gamma)
        
        Args:
            probabilities: List of objective probabilities
            gamma: Probability weighting parameter (default: from params)
            
        Returns:
            List of subjective decision weights
        """
        gamma = gamma if gamma is not None else self.params.gamma
        
        weights = []
        for p in probabilities:
            # Ensure probability is in [0,1]
            p = max(0, min(1, p))
            
            if p == 0:
                weights.append(0)
            elif p == 1:
                weights.append(1)
            else:
                # Apply probability weighting function
                try:
                    numerator = p ** gamma
                    denominator = (p ** gamma + (1 - p) ** gamma) ** (1 / gamma)
                    weights.append(numerator / denominator)
                except (ZeroDivisionError, OverflowError):
                    # Fallback for numerical issues
                    self.logger.warning(f"Numerical error in probability weighting for p={p}, gamma={gamma}")
                    # Use linear weighting as fallback
                    weights.append(p)
                
        return weights
    
    def update_drawdown(self, current_drawdown: float):
        """Update the current portfolio drawdown."""
        self._current_drawdown = current_drawdown
        
    def set_market_phase(self, phase: Union[MarketPhase, str]):
        """Set the current market phase."""
        if isinstance(phase, str):
            self._current_phase = MarketPhase.from_string(phase)
        else:
            self._current_phase = phase
    
    @_time_execution
    def phase_adjusted_stoploss(self, 
                             current_phase: MarketPhase,
                             antifragility_score: float,
                             time_in_trade: int,
                             market_volatility: float,
                             signal_dispersion: float) -> float:
        """
        Calculate phase-adjusted stoploss based on market conditions.
        
        Args:
            current_phase: Current market phase
            antifragility_score: System antifragility score (0-1)
            time_in_trade: Number of candles in trade
            market_volatility: Current market volatility metric
            signal_dispersion: Dispersion among signals (0-1)
            
        Returns:
            Adjusted stoploss value (negative percentage)
        """
        # Get base multiplier for current market phase
        phase_multiplier = self.PHASE_STOPLOSS_MULTIPLIERS.get(current_phase, 1.0)
        
        # Adjust stoploss based on antifragility
        # Higher antifragility means system can handle more volatility
        antifragility_multiplier = 0.8 + (antifragility_score * 0.4)
        
        # Adjust for time in trade (widen stops over time)
        time_multiplier = min(1.3, 1 + (time_in_trade / 100))
        
        # Adjust for market volatility (tighter stops in high volatility)
        volatility_multiplier = 1.0 / (1.0 + (market_volatility * self.params.risk_sensitivity))
        
        # Adjust for signal dispersion (tighter stops with high disagreement)
        dispersion_multiplier = 1.0 - (signal_dispersion * 0.3)
        
        # Calculate final stoploss
        adjusted_stoploss = self.params.base_stoploss * phase_multiplier * antifragility_multiplier * \
                          time_multiplier * volatility_multiplier * dispersion_multiplier
        
        # Ensure stoploss doesn't get too wide or too narrow
        return max(self.MIN_STOPLOSS, min(self.MAX_STOPLOSS, adjusted_stoploss))
    
    @_time_execution
    def calculate_dynamic_reference_point(self,
                                       entry_price: float,
                                       current_price: float,
                                       ema_price: float,
                                       time_in_trade: int) -> float:
        """
        Calculate a dynamic reference point for evaluating gains/losses.
        
        A key insight of Prospect Theory is that outcomes are evaluated relative
        to a reference point, which can shift over time. This implements adaptive
        reference points for more realistic risk modeling.
        
        Args:
            entry_price: Original entry price
            current_price: Current market price
            ema_price: Exponential moving average price
            time_in_trade: Number of candles in trade
            
        Returns:
            Dynamic reference price
        """
        if not self.params.dynamic_reference:
            return entry_price
            
        # As time passes, shift reference point from entry price toward EMA
        time_factor = min(1.0, time_in_trade / 50)
        reference_point = (1 - time_factor) * entry_price + time_factor * ema_price
        
        return reference_point
    
    @_time_execution
    def calculate_position_size(self,
                              base_position: float,
                              current_phase: MarketPhase,
                              cognitive_diversity: float,
                              confidence_score: float,
                              drawdown: Optional[float] = None) -> float:
        """
        Calculate position size based on market conditions and Prospect Theory.
        
        Args:
            base_position: Base position size (e.g., 0.1 = 10% of portfolio)
            current_phase: Current market phase
            cognitive_diversity: Diversity among signals (0-1)
            confidence_score: Confidence in current prediction (0-1)
            drawdown: Current drawdown as decimal (0.05 = 5% drawdown)
            
        Returns:
            Adjusted position size
        """
        # Use class attribute if not provided
        drawdown = self._current_drawdown if drawdown is None else drawdown
        
        # Get phase-based multiplier
        phase_multiplier = self.PHASE_POSITION_MULTIPLIERS.get(current_phase, 1.0)
        
        # Higher confidence = larger position
        confidence_multiplier = 0.5 + (confidence_score * 0.8)
        
        # Higher cognitive diversity = smaller position (uncertainty penalty)
        diversity_multiplier = 1.0 - (cognitive_diversity * 0.4)
        
        # Reduce position size during drawdowns (loss aversion)
        drawdown_factor = 1.0 - (drawdown * self.params.loss_aversion_factor)
        drawdown_multiplier = max(0.3, drawdown_factor)
        
        # Calculate final position size
        adjusted_position = base_position * phase_multiplier * confidence_multiplier * \
                           diversity_multiplier * drawdown_multiplier
        
        # Ensure position size doesn't get too large or too small
        return max(base_position * self.MIN_POSITION, 
                   min(base_position * self.MAX_POSITION, adjusted_position))
    
    @_time_execution
    def calculate_take_profit_targets(self,
                                    entry_price: float,
                                    current_phase: MarketPhase,
                                    volatility: float) -> List[Tuple[float, float]]:
        """
        Calculate multiple take profit targets based on market conditions.
        
        Args:
            entry_price: Trade entry price
            current_phase: Current market phase
            volatility: Market volatility metric
            
        Returns:
            List of (target_price, size_percentage) tuples
        """
        # Get phase-specific target multipliers
        targets = self.PHASE_TARGETS.get(current_phase, [1.5, 2.0, 2.5])
        
        # Scale targets by base take profit and adjust for volatility
        volatility_factor = max(0.8, min(1.5, 1.0 + (volatility - 0.1) * 2))
        scaled_targets = [self.params.base_take_profit * t * volatility_factor for t in targets]
        
        # Calculate actual target prices
        target_prices = [entry_price * (1 + target) for target in scaled_targets]
        
        # Position sizing for each target (partial take profits)
        # For 3 targets: 50%, 30%, 20% of position
        size_percentages = [0.5, 0.3, 0.2]
        
        return list(zip(target_prices, size_percentages))
    
    @_time_execution
    def custom_stoploss_function(self,
                              current_profit: float,
                              entry_price: float,
                              current_price: float,
                              trade_duration: int,
                              **context) -> float:
        """
        Calculate custom stoploss for trading system integration.
        
        Args:
            current_profit: Current profit/loss as ratio
            entry_price: Trade entry price
            current_price: Current price
            trade_duration: Trade duration in candle periods
            context: Additional context information
            
        Returns:
            Stoploss value as a negative ratio
        """
        # Extract context variables
        current_phase = context.get('market_phase', MarketPhase.CONSERVATION)
        antifragility = context.get('antifragility_score', 0.5)
        volatility = context.get('market_volatility', 0.1)
        dispersion = context.get('signal_dispersion', 0.2)
        
        # Get base stoploss from phase adjustment
        stoploss = self.phase_adjusted_stoploss(
            current_phase, 
            antifragility, 
            trade_duration, 
            volatility, 
            dispersion
        )
        
        # Trailing logic - gradually raise stoploss as profit increases
        if current_profit > 0:
            # Calculate profit factor based on Prospect Theory (diminishing sensitivity)
            profit_factor = self.calculate_value_function(current_profit) / self.calculate_value_function(1.0)
            
            # Higher profits lead to tighter trailing stops
            trailing_factor = min(0.7, profit_factor)
            
            # Calculate trailing stoploss (never lower than base)
            trailing_stoploss = current_profit * trailing_factor
            stoploss = max(stoploss, trailing_stoploss)
        
        return stoploss
    
    @_time_execution
    def calculate_expected_utility(self,
                                probabilities: List[float],
                                outcomes: List[float]) -> float:
        """
        Calculate expected utility based on Prospect Theory.
        
        Args:
            probabilities: List of outcome probabilities
            outcomes: List of potential outcomes (as percentages)
            
        Returns:
            Expected utility according to Prospect Theory
        """
        if len(probabilities) != len(outcomes):
            self.logger.warning("Probability and outcome lists must be the same length")
            return 0.0
            
        # Calculate decision weights
        decision_weights = self.calculate_decision_weights(probabilities)
        
        # Calculate value for each outcome
        values = [self.calculate_value_function(outcome) for outcome in outcomes]
        
        # Calculate expected utility
        utility = sum(w * v for w, v in zip(decision_weights, values))
        
        return utility
    
    @_time_execution
    def evaluate_trade_opportunity(self,
                                 entry_signals: Dict[str, float],
                                 predicted_profit: float,
                                 predicted_loss: float,
                                 win_probability: float,
                                 market_phase: MarketPhase,
                                 **kwargs) -> Tuple[bool, float, Dict]:
        """
        Evaluate a trading opportunity using Prospect Theory.
        
        Args:
            entry_signals: Dictionary of entry signals
            predicted_profit: Predicted profit percentage
            predicted_loss: Predicted loss percentage (negative)
            win_probability: Probability of winning trade
            market_phase: Current market phase
            kwargs: Additional parameters
            
        Returns:
            Tuple of (take_trade, position_size, metadata)
        """
        # Extract cognitive diversity from entry signals
        if len(entry_signals) >= 2:
            # Use max dispersion between any signals as diversity measure
            signal_values = list(entry_signals.values())
            cognitive_diversity = max(abs(a - b) for a in signal_values for b in signal_values) / 1.0
        else:
            cognitive_diversity = 0.0
        
        # Calculate confidence as average of entry signals
        confidence = sum(entry_signals.values()) / len(entry_signals) if entry_signals else 0.5
        
        # Get current drawdown from kwargs or instance variable
        drawdown = kwargs.get('drawdown', self._current_drawdown)
        
        # Calculate position size
        base_position = kwargs.get('base_position', 0.1)  # Default 10% of portfolio
        position_size = self.calculate_position_size(
            base_position,
            market_phase,
            cognitive_diversity,
            confidence,
            drawdown
        )
        
        # Calculate expected utility
        outcomes = [predicted_profit, predicted_loss]
        probabilities = [win_probability, 1 - win_probability]
        expected_utility = self.calculate_expected_utility(probabilities, outcomes)
        
        # Determine whether to take the trade
        utility_threshold = kwargs.get('utility_threshold', self.params.utility_threshold)
        take_trade = expected_utility > utility_threshold and confidence > 0.55
        
        # Prepare metadata
        metadata = {
            'expected_utility': expected_utility,
            'cognitive_diversity': cognitive_diversity,
            'confidence': confidence,
            'decision_weights': self.calculate_decision_weights(probabilities),
            'threshold': utility_threshold,
            'position_size': position_size,
            'drawdown_factor': 1.0 - (drawdown * self.params.loss_aversion_factor),
            'market_phase': str(market_phase)
        }
        
        return take_trade, position_size, metadata
    
    @_time_execution
    def normalize_multiple_reference_points(self, 
                                         ref_points: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize weights of multiple reference points.
        Implements the reference point dependence concept from Prospect Theory.
        
        Args:
            ref_points: Dictionary mapping reference point names to their raw weights
            
        Returns:
            Dictionary with normalized weights summing to 1.0
        """
        total_weight = sum(ref_points.values())
        
        if total_weight == 0:
            # Equal weights if all zeros
            n_points = len(ref_points)
            return {k: 1.0/n_points for k in ref_points}
        
        return {k: v/total_weight for k, v in ref_points.items()}
    
    @_time_execution
    def get_composite_reference_point(self, 
                                   points_dict: Dict[str, float],
                                   weights: Optional[Dict[str, float]] = None) -> float:
        """
        Calculate a composite reference point from multiple sources.
        
        Args:
            points_dict: Dictionary mapping reference point names to their values
            weights: Optional dictionary of weights (will be normalized)
            
        Returns:
            Weighted average reference point
        """
        if not weights:
            # Equal weights if not specified
            n_points = len(points_dict)
            weights = {k: 1.0/n_points for k in points_dict} 
        else:
            # Ensure weights only include keys that are in points_dict
            weights = {k: v for k, v in weights.items() if k in points_dict}
            weights = self.normalize_multiple_reference_points(weights)
        
        # Calculate weighted average
        weighted_sum = sum(points_dict[k] * weights[k] for k in weights)
        
        return weighted_sum

    def record_trade(self, trade_data: Dict[str, Any]):
        """
        Record trade data for adaptive behavior.
        
        Args:
            trade_data: Dictionary containing trade information
        """
        # Add timestamp
        trade_data['timestamp'] = time.time()
        self._trade_history.append(trade_data)
        
        # Limit history size to prevent memory issues
        if len(self._trade_history) > 1000:
            self._trade_history = self._trade_history[-1000:]
            
        # Check if parameters should adapt
        self._adapt_parameters()
        
    def _adapt_parameters(self):
        """
        Adapt risk parameters based on recent trading performance.
        Implements self-organizing behavior to adapt to changing market conditions.
        """
        # Only adapt periodically
        current_time = time.time()
        if current_time - self._last_update_time < 3600:  # 1 hour
            return
            
        self._last_update_time = current_time
        
        # Need sufficient history
        if len(self._trade_history) < 20:
            return
            
        # Calculate recent performance
        recent_trades = self._trade_history[-20:]
        wins = sum(1 for trade in recent_trades if trade.get('profit', 0) > 0)
        losses = len(recent_trades) - wins
        
        if losses == 0:
            win_ratio = 1.0
        else:
            win_ratio = wins / len(recent_trades)
            
        avg_profit = sum(trade.get('profit', 0) for trade in recent_trades) / len(recent_trades)
        
        # Adapt parameters based on performance
        if win_ratio < 0.4:
            # Poor performance - reduce risk
            self.params.risk_sensitivity *= 0.9
            self.params.base_stoploss *= 0.9  # Tighter stops
            self.logger.info("Adapting parameters to reduce risk due to poor performance")
        elif win_ratio > 0.6 and avg_profit > 0:
            # Good performance - can take more risk
            self.params.risk_sensitivity *= 1.1
            self.params.base_stoploss *= 1.1  # Wider stops
            self.logger.info("Adapting parameters to increase risk due to good performance")
            
        # Ensure parameters stay within reasonable bounds
        self.params.risk_sensitivity = max(0.5, min(2.0, self.params.risk_sensitivity))
        self.params.base_stoploss = max(-0.15, min(-0.02, self.params.base_stoploss))
