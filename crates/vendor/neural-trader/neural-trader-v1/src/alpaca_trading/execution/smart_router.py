"""Smart Order Router for optimal execution.

Determines order type, time-in-force, and routing parameters
based on market conditions and signal urgency.
"""

import asyncio
from datetime import datetime, time as datetime_time
from typing import Dict, Optional, List, Any
from dataclasses import dataclass
import logging
from enum import Enum

from .order_manager import OrderType, TimeInForce

logger = logging.getLogger(__name__)


@dataclass
class RouteDecision:
    """Smart routing decision."""
    order_type: OrderType
    time_in_force: TimeInForce
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    extended_hours: bool = False
    split_orders: List[float] = None  # For order splitting
    routing_reason: str = ""
    confidence: float = 1.0


class MarketSession(Enum):
    """Market session types."""
    PRE_MARKET = "pre_market"     # 4:00 AM - 9:30 AM ET
    REGULAR = "regular"           # 9:30 AM - 4:00 PM ET
    AFTER_HOURS = "after_hours"   # 4:00 PM - 8:00 PM ET
    CLOSED = "closed"


class SmartRouter:
    """Smart order router for optimal execution.
    
    Features:
    - Market session aware routing
    - Order type optimization
    - Time-in-force selection
    - Large order splitting
    - Extended hours routing
    """
    
    def __init__(self, 
                 large_order_threshold: float = 1000,
                 max_order_splits: int = 5,
                 aggressive_spread_multiplier: float = 2.0,
                 passive_spread_discount: float = 0.5):
        """Initialize smart router.
        
        Args:
            large_order_threshold: Threshold for order splitting
            max_order_splits: Maximum number of order splits
            aggressive_spread_multiplier: Multiplier for aggressive orders
            passive_spread_discount: Discount for passive orders
        """
        self.large_order_threshold = large_order_threshold
        self.max_order_splits = max_order_splits
        self.aggressive_spread_multiplier = aggressive_spread_multiplier
        self.passive_spread_discount = passive_spread_discount
        
        # Performance metrics
        self._metrics = {
            'orders_routed': 0,
            'market_orders': 0,
            'limit_orders': 0,
            'stop_orders': 0,
            'orders_split': 0,
            'extended_hours_orders': 0
        }
    
    async def route_order(self, signal: Any, market_data: Dict[str, Any], 
                         quantity: float) -> RouteDecision:
        """Determine optimal routing for order.
        
        Args:
            signal: Trading signal
            market_data: Current market data
            quantity: Order quantity
            
        Returns:
            Routing decision
        """
        self._metrics['orders_routed'] += 1
        
        # Determine market session
        session = self._get_market_session()
        
        # Base routing decision on urgency and market conditions
        if signal.urgency == 'high' or session == MarketSession.CLOSED:
            return await self._route_urgent_order(signal, market_data, quantity, session)
        elif signal.urgency == 'low':
            return await self._route_passive_order(signal, market_data, quantity, session)
        else:
            return await self._route_standard_order(signal, market_data, quantity, session)
    
    async def _route_urgent_order(self, signal: Any, market_data: Dict[str, Any],
                                 quantity: float, session: MarketSession) -> RouteDecision:
        """Route urgent order requiring immediate execution.
        
        Args:
            signal: Trading signal
            market_data: Market data
            quantity: Order quantity
            session: Current market session
            
        Returns:
            Routing decision for urgent execution
        """
        self._metrics['market_orders'] += 1
        
        # Check if we need extended hours
        extended_hours = session in [MarketSession.PRE_MARKET, MarketSession.AFTER_HOURS]
        if extended_hours:
            self._metrics['extended_hours_orders'] += 1
        
        # For urgent orders in regular hours, use market order
        if session == MarketSession.REGULAR:
            return RouteDecision(
                order_type=OrderType.MARKET,
                time_in_force=TimeInForce.IOC,  # Immediate or cancel
                extended_hours=False,
                routing_reason="Urgent order during regular hours - using market order",
                confidence=0.95
            )
        
        # For extended hours, must use limit order
        if extended_hours:
            # Use aggressive pricing
            if signal.action == 'buy':
                limit_price = market_data['ask'] * (1 + 0.001)  # 0.1% above ask
            else:
                limit_price = market_data['bid'] * (1 - 0.001)  # 0.1% below bid
            
            return RouteDecision(
                order_type=OrderType.LIMIT,
                time_in_force=TimeInForce.GTX,  # Good till extended
                limit_price=round(limit_price, 2),
                extended_hours=True,
                routing_reason="Urgent order in extended hours - using aggressive limit",
                confidence=0.85
            )
        
        # Market closed - queue for next open
        return RouteDecision(
            order_type=OrderType.MARKET,
            time_in_force=TimeInForce.OPG,  # At open
            extended_hours=False,
            routing_reason="Market closed - queuing market order for open",
            confidence=0.8
        )
    
    async def _route_passive_order(self, signal: Any, market_data: Dict[str, Any],
                                  quantity: float, session: MarketSession) -> RouteDecision:
        """Route passive order seeking price improvement.
        
        Args:
            signal: Trading signal
            market_data: Market data
            quantity: Order quantity
            session: Current market session
            
        Returns:
            Routing decision for passive execution
        """
        self._metrics['limit_orders'] += 1
        
        # Calculate passive limit price
        spread = market_data['spread']
        if signal.action == 'buy':
            # Place order below bid for price improvement
            limit_price = market_data['bid'] - (spread * self.passive_spread_discount)
        else:
            # Place order above ask for price improvement
            limit_price = market_data['ask'] + (spread * self.passive_spread_discount)
        
        # Check for large orders that need splitting
        split_orders = None
        if quantity > self.large_order_threshold:
            split_orders = self._calculate_order_splits(quantity)
            self._metrics['orders_split'] += 1
        
        return RouteDecision(
            order_type=OrderType.LIMIT,
            time_in_force=TimeInForce.GTC,  # Good till cancelled
            limit_price=round(limit_price, 2),
            extended_hours=False,
            split_orders=split_orders,
            routing_reason="Passive order - seeking price improvement",
            confidence=0.9
        )
    
    async def _route_standard_order(self, signal: Any, market_data: Dict[str, Any],
                                   quantity: float, session: MarketSession) -> RouteDecision:
        """Route standard order with balanced execution.
        
        Args:
            signal: Trading signal
            market_data: Market data
            quantity: Order quantity
            session: Current market session
            
        Returns:
            Routing decision for standard execution
        """
        # Determine order type based on market conditions
        spread_percentage = (market_data['spread'] / market_data['ask']) * 100
        
        # Wide spread or low liquidity - use limit order
        if spread_percentage > 0.1 or session != MarketSession.REGULAR:
            self._metrics['limit_orders'] += 1
            
            # Standard limit pricing at mid-point
            if signal.action == 'buy':
                limit_price = (market_data['bid'] + market_data['ask']) / 2
            else:
                limit_price = (market_data['bid'] + market_data['ask']) / 2
            
            return RouteDecision(
                order_type=OrderType.LIMIT,
                time_in_force=TimeInForce.DAY,
                limit_price=round(limit_price, 2),
                extended_hours=session != MarketSession.REGULAR,
                routing_reason=f"Standard order - spread {spread_percentage:.2f}% using limit",
                confidence=0.85
            )
        
        # Tight spread and good liquidity - use market order
        else:
            self._metrics['market_orders'] += 1
            
            # Add stop loss if specified
            stop_price = None
            if signal.stop_loss:
                self._metrics['stop_orders'] += 1
                stop_price = signal.stop_loss
            
            return RouteDecision(
                order_type=OrderType.MARKET,
                time_in_force=TimeInForce.DAY,
                stop_price=stop_price,
                extended_hours=False,
                routing_reason="Standard order - tight spread using market order",
                confidence=0.9
            )
    
    def _get_market_session(self) -> MarketSession:
        """Determine current market session.
        
        Returns:
            Current market session
        """
        now = datetime.now()
        current_time = now.time()
        weekday = now.weekday()
        
        # Weekend - market closed
        if weekday >= 5:  # Saturday = 5, Sunday = 6
            return MarketSession.CLOSED
        
        # Define session times (Eastern Time)
        # Note: In production, handle timezone conversion properly
        pre_market_start = datetime_time(4, 0)
        regular_start = datetime_time(9, 30)
        regular_end = datetime_time(16, 0)
        after_hours_end = datetime_time(20, 0)
        
        if pre_market_start <= current_time < regular_start:
            return MarketSession.PRE_MARKET
        elif regular_start <= current_time < regular_end:
            return MarketSession.REGULAR
        elif regular_end <= current_time < after_hours_end:
            return MarketSession.AFTER_HOURS
        else:
            return MarketSession.CLOSED
    
    def _calculate_order_splits(self, quantity: float) -> List[float]:
        """Calculate order splits for large orders.
        
        Args:
            quantity: Total order quantity
            
        Returns:
            List of split quantities
        """
        # Simple equal splitting for now
        # In production, use more sophisticated algorithms
        num_splits = min(
            self.max_order_splits,
            int(quantity / (self.large_order_threshold / 2)) + 1
        )
        
        base_qty = quantity // num_splits
        remainder = quantity % num_splits
        
        splits = [base_qty] * num_splits
        if remainder > 0:
            splits[-1] += remainder
        
        return splits
    
    def should_use_stop_order(self, signal: Any, market_volatility: float) -> bool:
        """Determine if stop order should be used.
        
        Args:
            signal: Trading signal
            market_volatility: Current market volatility
            
        Returns:
            True if stop order recommended
        """
        # Use stop orders for:
        # 1. Explicit stop loss in signal
        # 2. High volatility markets
        # 3. Low confidence signals
        return (
            signal.stop_loss is not None or
            market_volatility > 0.02 or  # 2% volatility threshold
            signal.confidence < 0.7
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get routing performance metrics.
        
        Returns:
            Dictionary of metrics
        """
        metrics = self._metrics.copy()
        
        # Calculate percentages
        total = metrics['orders_routed']
        if total > 0:
            metrics['market_order_pct'] = (metrics['market_orders'] / total) * 100
            metrics['limit_order_pct'] = (metrics['limit_orders'] / total) * 100
            metrics['split_order_pct'] = (metrics['orders_split'] / total) * 100
            metrics['extended_hours_pct'] = (metrics['extended_hours_orders'] / total) * 100
        
        return metrics
