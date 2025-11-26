"""Slippage Controller for execution quality optimization.

Measures and controls slippage through adaptive pricing,
market impact estimation, and execution quality metrics.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple, Any
from dataclasses import dataclass, field
from collections import deque
import statistics
import logging
import math

from .order_manager import OrderType

logger = logging.getLogger(__name__)


@dataclass
class SlippageMetrics:
    """Slippage measurement for an execution."""
    symbol: str
    side: str
    expected_price: float
    executed_price: float
    slippage_bps: float  # Basis points
    slippage_dollars: float
    quantity: float
    market_impact_bps: float
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class MarketImpactModel:
    """Market impact estimation model."""
    symbol: str
    avg_spread_bps: float = 5.0  # Average spread in basis points
    liquidity_factor: float = 1.0  # Liquidity multiplier
    volatility: float = 0.01  # Daily volatility
    avg_volume: float = 1000000  # Average daily volume
    impact_coefficient: float = 0.1  # Market impact coefficient


class SlippageController:
    """Advanced slippage control and measurement.
    
    Features:
    - Real-time slippage tracking
    - Adaptive limit pricing
    - Market impact estimation
    - Execution quality scoring
    - Historical slippage analysis
    """
    
    def __init__(self,
                 max_slippage_bps: float = 10.0,
                 adaptive_window: int = 100,
                 impact_lookback_minutes: int = 30):
        """Initialize slippage controller.
        
        Args:
            max_slippage_bps: Maximum acceptable slippage in basis points
            adaptive_window: Window size for adaptive pricing
            impact_lookback_minutes: Lookback period for impact analysis
        """
        self.max_slippage_bps = max_slippage_bps
        self.adaptive_window = adaptive_window
        self.impact_lookback_minutes = impact_lookback_minutes
        
        # Slippage history by symbol
        self._slippage_history: Dict[str, deque] = {}
        
        # Market impact models by symbol
        self._impact_models: Dict[str, MarketImpactModel] = {}
        
        # Recent executions for impact analysis
        self._recent_executions: deque = deque(maxlen=1000)
        
        # Adaptive pricing parameters by symbol
        self._adaptive_params: Dict[str, Dict[str, float]] = {}
        
        # Performance metrics
        self._metrics = {
            'total_executions': 0,
            'positive_slippage': 0,
            'negative_slippage': 0,
            'avg_slippage_bps': 0.0,
            'worst_slippage_bps': 0.0,
            'execution_quality_score': 100.0
        }
    
    async def adjust_price(self, symbol: str, side: str, order_type: OrderType,
                          base_price: Optional[float], urgency: str,
                          market_data: Dict[str, Any]) -> Optional[float]:
        """Adjust limit price based on slippage control.
        
        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            order_type: Order type
            base_price: Base limit price (if any)
            urgency: Order urgency
            market_data: Current market data
            
        Returns:
            Adjusted price or None for market orders
        """
        # Market orders don't need price adjustment
        if order_type == OrderType.MARKET:
            return None
        
        # Get adaptive parameters
        params = self._get_adaptive_params(symbol)
        
        # Calculate base adjustment based on urgency
        urgency_multiplier = {
            'high': 1.5,
            'medium': 1.0,
            'low': 0.5
        }.get(urgency, 1.0)
        
        # Get recent slippage for this symbol
        avg_slippage = self._get_average_slippage(symbol, side)
        
        # Estimate market impact
        impact_bps = await self._estimate_market_impact(
            symbol, side, market_data.get('quantity', 100)
        )
        
        # Calculate adjusted price
        if base_price is None:
            # No base price provided, use market data
            if side == 'buy':
                base_price = market_data['ask']
            else:
                base_price = market_data['bid']
        
        # Apply adjustments
        spread = market_data['spread']
        adjustment_factor = params['adjustment_factor'] * urgency_multiplier
        
        if side == 'buy':
            # For buys, add buffer to ensure execution
            buffer = spread * adjustment_factor
            # Account for expected slippage and impact
            buffer += base_price * (avg_slippage + impact_bps) / 10000
            adjusted_price = base_price + buffer
        else:
            # For sells, subtract buffer
            buffer = spread * adjustment_factor
            buffer += base_price * (avg_slippage + impact_bps) / 10000
            adjusted_price = base_price - buffer
        
        # Apply max slippage constraint
        max_deviation = base_price * self.max_slippage_bps / 10000
        if side == 'buy':
            adjusted_price = min(adjusted_price, base_price + max_deviation)
        else:
            adjusted_price = max(adjusted_price, base_price - max_deviation)
        
        logger.debug(f"Price adjustment for {symbol}: base={base_price:.2f}, "
                    f"adjusted={adjusted_price:.2f}, slippage={avg_slippage:.1f}bps, "
                    f"impact={impact_bps:.1f}bps")
        
        return round(adjusted_price, 2)
    
    async def record_execution(self, symbol: str, side: str, 
                              expected_price: float, executed_price: float,
                              quantity: float) -> SlippageMetrics:
        """Record execution and calculate slippage.
        
        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            expected_price: Expected execution price
            executed_price: Actual execution price
            quantity: Executed quantity
            
        Returns:
            Slippage metrics for the execution
        """
        # Calculate slippage
        if side == 'buy':
            slippage_dollars = executed_price - expected_price
        else:
            slippage_dollars = expected_price - executed_price
        
        slippage_bps = (slippage_dollars / expected_price) * 10000
        
        # Estimate market impact
        market_impact_bps = await self._estimate_market_impact(symbol, side, quantity)
        
        # Create metrics object
        metrics = SlippageMetrics(
            symbol=symbol,
            side=side,
            expected_price=expected_price,
            executed_price=executed_price,
            slippage_bps=slippage_bps,
            slippage_dollars=slippage_dollars * quantity,
            quantity=quantity,
            market_impact_bps=market_impact_bps
        )
        
        # Update history
        if symbol not in self._slippage_history:
            self._slippage_history[symbol] = deque(maxlen=self.adaptive_window)
        self._slippage_history[symbol].append(metrics)
        
        # Update recent executions
        self._recent_executions.append(metrics)
        
        # Update metrics
        self._update_metrics(metrics)
        
        # Update adaptive parameters
        self._update_adaptive_params(symbol, metrics)
        
        logger.info(f"Execution recorded: {symbol} {side} slippage={slippage_bps:.1f}bps "
                   f"impact={market_impact_bps:.1f}bps")
        
        return metrics
    
    async def _estimate_market_impact(self, symbol: str, side: str, 
                                     quantity: float) -> float:
        """Estimate market impact in basis points.
        
        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            quantity: Order quantity
            
        Returns:
            Estimated market impact in basis points
        """
        # Get or create impact model
        if symbol not in self._impact_models:
            self._impact_models[symbol] = MarketImpactModel(symbol=symbol)
        
        model = self._impact_models[symbol]
        
        # Simple square-root market impact model
        # Impact = coefficient * sqrt(quantity / avg_volume) * volatility
        participation_rate = quantity / model.avg_volume
        impact = model.impact_coefficient * math.sqrt(participation_rate) * model.volatility
        
        # Convert to basis points
        impact_bps = impact * 10000
        
        # Adjust for liquidity
        impact_bps *= model.liquidity_factor
        
        # Adjust for spread
        impact_bps += model.avg_spread_bps * 0.5  # Half spread cost
        
        return impact_bps
    
    def _get_adaptive_params(self, symbol: str) -> Dict[str, float]:
        """Get adaptive pricing parameters for symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary of adaptive parameters
        """
        if symbol not in self._adaptive_params:
            # Default parameters
            self._adaptive_params[symbol] = {
                'adjustment_factor': 0.5,
                'confidence': 0.5,
                'learning_rate': 0.1
            }
        return self._adaptive_params[symbol]
    
    def _update_adaptive_params(self, symbol: str, metrics: SlippageMetrics):
        """Update adaptive parameters based on execution.
        
        Args:
            symbol: Trading symbol
            metrics: Execution metrics
        """
        params = self._get_adaptive_params(symbol)
        
        # Simple adaptive algorithm
        # If slippage is positive (good), reduce adjustment
        # If slippage is negative (bad), increase adjustment
        if metrics.slippage_bps > 0:
            # Good execution, reduce adjustment
            params['adjustment_factor'] *= (1 - params['learning_rate'])
            params['confidence'] = min(1.0, params['confidence'] + 0.05)
        else:
            # Poor execution, increase adjustment
            params['adjustment_factor'] *= (1 + params['learning_rate'])
            params['confidence'] = max(0.0, params['confidence'] - 0.05)
        
        # Bounds
        params['adjustment_factor'] = max(0.1, min(2.0, params['adjustment_factor']))
    
    def _get_average_slippage(self, symbol: str, side: str) -> float:
        """Get average recent slippage for symbol and side.
        
        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            
        Returns:
            Average slippage in basis points
        """
        if symbol not in self._slippage_history:
            return 0.0
        
        relevant_metrics = [
            m.slippage_bps for m in self._slippage_history[symbol]
            if m.side == side
        ]
        
        if not relevant_metrics:
            return 0.0
        
        return statistics.mean(relevant_metrics)
    
    def _update_metrics(self, metrics: SlippageMetrics):
        """Update performance metrics.
        
        Args:
            metrics: Execution metrics
        """
        self._metrics['total_executions'] += 1
        
        if metrics.slippage_bps > 0:
            self._metrics['positive_slippage'] += 1
        else:
            self._metrics['negative_slippage'] += 1
        
        # Update average slippage
        count = self._metrics['total_executions']
        current_avg = self._metrics['avg_slippage_bps']
        self._metrics['avg_slippage_bps'] = (
            (current_avg * (count - 1) + metrics.slippage_bps) / count
        )
        
        # Update worst slippage
        if metrics.slippage_bps < self._metrics['worst_slippage_bps']:
            self._metrics['worst_slippage_bps'] = metrics.slippage_bps
        
        # Update execution quality score (0-100)
        # Based on percentage of executions within acceptable slippage
        acceptable_executions = self._metrics['positive_slippage'] + sum(
            1 for m in self._recent_executions
            if abs(m.slippage_bps) <= self.max_slippage_bps
        )
        self._metrics['execution_quality_score'] = (
            (acceptable_executions / count) * 100 if count > 0 else 100
        )
    
    def get_symbol_analysis(self, symbol: str) -> Dict[str, Any]:
        """Get detailed slippage analysis for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Analysis dictionary
        """
        if symbol not in self._slippage_history:
            return {
                'symbol': symbol,
                'executions': 0,
                'avg_slippage_bps': 0,
                'slippage_std_dev': 0,
                'positive_rate': 0,
                'market_impact_estimate': 0
            }
        
        history = list(self._slippage_history[symbol])
        slippages = [m.slippage_bps for m in history]
        
        return {
            'symbol': symbol,
            'executions': len(history),
            'avg_slippage_bps': statistics.mean(slippages) if slippages else 0,
            'slippage_std_dev': statistics.stdev(slippages) if len(slippages) > 1 else 0,
            'positive_rate': sum(1 for s in slippages if s > 0) / len(slippages) if slippages else 0,
            'market_impact_estimate': statistics.mean([m.market_impact_bps for m in history]) if history else 0,
            'recent_trend': self._calculate_trend(slippages)
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction.
        
        Args:
            values: List of values (newest last)
            
        Returns:
            Trend description
        """
        if len(values) < 3:
            return 'insufficient_data'
        
        # Simple linear regression on recent values
        recent = values[-10:]  # Last 10 values
        if len(recent) < 3:
            return 'insufficient_data'
        
        # Calculate slope
        x_values = list(range(len(recent)))
        x_mean = sum(x_values) / len(x_values)
        y_mean = sum(recent) / len(recent)
        
        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, recent))
        denominator = sum((x - x_mean) ** 2 for x in x_values)
        
        if denominator == 0:
            return 'stable'
        
        slope = numerator / denominator
        
        if slope > 0.1:
            return 'improving'
        elif slope < -0.1:
            return 'deteriorating'
        else:
            return 'stable'
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics.
        
        Returns:
            Dictionary of metrics
        """
        metrics = self._metrics.copy()
        
        # Add symbol-specific summaries
        symbol_summaries = {}
        for symbol in self._slippage_history:
            symbol_summaries[symbol] = self.get_symbol_analysis(symbol)
        
        metrics['symbol_analysis'] = symbol_summaries
        metrics['active_symbols'] = len(self._slippage_history)
        
        return metrics
    
    def get_execution_quality_report(self, hours: int = 24) -> Dict[str, Any]:
        """Generate execution quality report.
        
        Args:
            hours: Lookback period in hours
            
        Returns:
            Execution quality report
        """
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        recent_executions = [
            m for m in self._recent_executions
            if m.timestamp >= cutoff_time
        ]
        
        if not recent_executions:
            return {
                'period_hours': hours,
                'total_executions': 0,
                'execution_quality_score': 100.0
            }
        
        return {
            'period_hours': hours,
            'total_executions': len(recent_executions),
            'avg_slippage_bps': statistics.mean([m.slippage_bps for m in recent_executions]),
            'total_slippage_cost': sum([m.slippage_dollars for m in recent_executions]),
            'positive_slippage_rate': sum(1 for m in recent_executions if m.slippage_bps > 0) / len(recent_executions),
            'avg_market_impact_bps': statistics.mean([m.market_impact_bps for m in recent_executions]),
            'execution_quality_score': self._metrics['execution_quality_score']
        }
