"""
Market Maker Trading Strategy

This strategy provides liquidity to prediction markets by placing bid and ask orders
around the current market price, capturing the spread while managing inventory risk.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple

from ..models import Market, MarketStatus, Order, OrderBook, OrderSide, OrderStatus
from .base import (
    PolymarketStrategy, StrategyConfig, TradingSignal, SignalStrength,
    SignalDirection, StrategyError
)

logger = logging.getLogger(__name__)


@dataclass
class MarketMakingSignal(TradingSignal):
    """Extended signal for market making with bid/ask prices"""
    bid_price: Optional[Decimal] = None
    ask_price: Optional[Decimal] = None
    bid_size: Decimal = Decimal('0')
    ask_size: Decimal = Decimal('0')
    spread: Decimal = Decimal('0')
    inventory_risk: float = 0.0  # 0 to 1
    is_arbitrage: bool = False
    
    def __post_init__(self):
        """Validate market making signal"""
        super().__post_init__()
        
        if self.bid_price and self.ask_price:
            if self.bid_price >= self.ask_price:
                raise ValueError("Bid price must be less than ask price")
            self.spread = self.ask_price - self.bid_price
        
        # For market making, we don't use the inherited target_price directly
        if not self.target_price and self.bid_price and self.ask_price:
            self.target_price = (self.bid_price + self.ask_price) / Decimal('2')
    
    @property
    def expected_profit(self) -> Decimal:
        """Calculate expected profit from spread capture"""
        if not self.bid_price or not self.ask_price:
            return Decimal('0')
        
        min_size = min(self.bid_size, self.ask_size)
        return self.spread * min_size * Decimal(str(self.confidence))


class MarketMakerStrategy(PolymarketStrategy):
    """
    Market making strategy for prediction markets
    
    Provides liquidity by maintaining bid/ask quotes, capturing spreads while
    managing inventory risk and market exposure.
    """
    
    def __init__(
        self,
        client,
        config: Optional[StrategyConfig] = None,
        spread_target: Decimal = Decimal('0.02'),  # 2% target spread
        max_inventory: Decimal = Decimal('500.0'),  # Maximum inventory per market
        inventory_skew_factor: Decimal = Decimal('0.3'),  # How much to skew prices based on inventory
        min_spread: Decimal = Decimal('0.001'),  # 0.1% minimum spread
        max_spread: Decimal = Decimal('0.1'),  # 10% maximum spread
        quote_refresh_interval: int = 30,  # Seconds between quote updates
        min_liquidity: int = 10000,  # Minimum market liquidity to participate
    ):
        """
        Initialize market maker strategy
        
        Args:
            client: Polymarket API client
            config: Strategy configuration
            spread_target: Target bid-ask spread
            max_inventory: Maximum position size per market
            inventory_skew_factor: Price adjustment factor based on inventory
            min_spread: Minimum allowed spread
            max_spread: Maximum allowed spread
            quote_refresh_interval: How often to update quotes
            min_liquidity: Minimum market liquidity requirement
        """
        super().__init__(client, config, "MarketMakerStrategy")
        
        self.spread_target = spread_target
        self.max_inventory = max_inventory
        self.inventory_skew_factor = inventory_skew_factor
        self.min_spread = min_spread
        self.max_spread = max_spread
        self.quote_refresh_interval = quote_refresh_interval
        self.min_liquidity = min_liquidity
        
        # Inventory tracking
        self.inventory: Dict[str, Decimal] = {}  # market_id -> net position
        self.active_orders: Dict[str, List[Order]] = {}  # market_id -> orders
        
        # Performance tracking
        self.spread_captures: Dict[str, List[Decimal]] = {}
        self.inventory_history: Dict[str, List[Tuple[datetime, Decimal]]] = {}
        
        logger.info(
            f"Initialized {self.name} with target spread {self.spread_target}, "
            f"max inventory {self.max_inventory}"
        )
    
    async def should_trade_market(self, market: Market) -> bool:
        """
        Determine if market is suitable for market making
        
        Args:
            market: Market to evaluate
            
        Returns:
            True if market is suitable for market making
        """
        # Check if market is active
        if market.status != MarketStatus.ACTIVE:
            return False
        
        # Check liquidity
        liquidity = market.metadata.get('liquidity', 0)
        if liquidity < self.min_liquidity:
            logger.debug(f"Market {market.id} liquidity {liquidity} below minimum")
            return False
        
        # Check volume
        volume_24h = market.metadata.get('volume_24h', 0)
        if volume_24h < self.min_liquidity / 2:
            logger.debug(f"Market {market.id} volume too low")
            return False
        
        # Check time to expiry (need enough time to capture spreads)
        if market.end_date:
            time_to_expiry = market.end_date - datetime.now()
            if time_to_expiry < timedelta(hours=24):
                logger.debug(f"Market {market.id} expiring too soon")
                return False
        
        # Check spread isn't too wide (indicating low liquidity)
        current_spread = market.metadata.get('spread', 1.0)
        if current_spread > 0.2:  # 20% spread is too wide
            logger.debug(f"Market {market.id} spread too wide: {current_spread}")
            return False
        
        return True
    
    async def analyze_market(self, market: Market) -> Optional[TradingSignal]:
        """
        Analyze market and generate market making signal
        
        Args:
            market: Market to analyze
            
        Returns:
            Market making signal with bid/ask quotes
        """
        try:
            # Get order book
            order_book = await self._get_order_book(market.id)
            if not order_book:
                logger.debug(f"No order book available for {market.id}")
                return None
            
            # Check for arbitrage opportunities first
            arb_signal = self._detect_arbitrage(order_book)
            if arb_signal:
                return arb_signal
            
            # Get current inventory
            inventory = self.inventory.get(market.id, Decimal('0'))
            
            # Calculate optimal quotes
            bid_price, ask_price = self._calculate_quotes(order_book, inventory)
            
            if not bid_price or not ask_price:
                logger.debug(f"Could not calculate valid quotes for {market.id}")
                return None
            
            # Calculate order sizes
            bid_size = self._calculate_order_size(inventory, OrderSide.BUY)
            ask_size = self._calculate_order_size(inventory, OrderSide.SELL)
            
            # Check risk limits
            if not self._check_inventory_limits(market.id, OrderSide.BUY, bid_size):
                bid_size = Decimal('0')
            if not self._check_inventory_limits(market.id, OrderSide.SELL, ask_size):
                ask_size = Decimal('0')
            
            if bid_size == 0 and ask_size == 0:
                logger.debug(f"Risk limits prevent trading in {market.id}")
                return None
            
            # Calculate inventory risk
            inventory_risk = self._calculate_inventory_risk(inventory)
            
            # Create market making signal
            signal = MarketMakingSignal(
                market_id=market.id,
                outcome="Yes",  # Market makers typically quote the Yes outcome
                direction=SignalDirection.BUY if bid_size > ask_size else SignalDirection.SELL,
                strength=self._calculate_signal_strength(order_book, inventory_risk),
                bid_price=bid_price,
                ask_price=ask_price,
                bid_size=bid_size,
                ask_size=ask_size,
                size=max(bid_size, ask_size),  # For compatibility with base class
                confidence=self._calculate_confidence(order_book, inventory_risk),
                reasoning=self._generate_reasoning(bid_price, ask_price, inventory),
                inventory_risk=inventory_risk,
                metadata={
                    'spread': float(ask_price - bid_price),
                    'inventory': float(inventory),
                    'strategy': self.name
                }
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Error analyzing market {market.id}: {str(e)}")
            return None
    
    async def _get_order_book(self, market_id: str) -> Optional[OrderBook]:
        """Get current order book for market"""
        try:
            # In real implementation, this would call the CLOB API
            # For now, return a mock order book
            order_book = OrderBook(
                market_id=market_id,
                outcome_id="Yes",
                bids=[
                    {"price": 0.48, "size": 100},
                    {"price": 0.47, "size": 200},
                    {"price": 0.46, "size": 150}
                ],
                asks=[
                    {"price": 0.52, "size": 120},
                    {"price": 0.53, "size": 180},
                    {"price": 0.54, "size": 140}
                ],
                timestamp=datetime.now()
            )
            return order_book
            
        except Exception as e:
            logger.error(f"Failed to get order book for {market_id}: {str(e)}")
            return None
    
    def _calculate_quotes(
        self,
        order_book: OrderBook,
        inventory: Decimal
    ) -> Tuple[Optional[Decimal], Optional[Decimal]]:
        """
        Calculate optimal bid and ask prices
        
        Args:
            order_book: Current order book
            inventory: Current inventory position
            
        Returns:
            Tuple of (bid_price, ask_price)
        """
        best_bid = order_book.best_bid
        best_ask = order_book.best_ask
        
        if not best_bid or not best_ask:
            # One-sided market, use mid price estimate
            if best_bid:
                mid_price = best_bid + self.spread_target / Decimal('2')
            elif best_ask:
                mid_price = best_ask - self.spread_target / Decimal('2')
            else:
                return None, None
        else:
            mid_price = order_book.mid_price
        
        # Calculate base spread
        current_spread = order_book.spread or self.spread_target
        
        # Adjust spread based on market conditions
        adjusted_spread = self._calculate_dynamic_spread(market_id=order_book.market_id)
        
        # Calculate inventory skew
        inventory_ratio = inventory / self.max_inventory if self.max_inventory > 0 else Decimal('0')
        skew = inventory_ratio * self.inventory_skew_factor
        
        # Adjust prices based on inventory
        # If long inventory, lower prices to encourage selling
        # If short inventory, raise prices to encourage buying
        bid_adjustment = -skew * adjusted_spread
        ask_adjustment = -skew * adjusted_spread
        
        # Calculate final quotes
        half_spread = adjusted_spread / Decimal('2')
        bid_price = mid_price - half_spread + bid_adjustment
        ask_price = mid_price + half_spread + ask_adjustment
        
        # Ensure we're inside the best bid/ask but not crossing
        if best_bid and best_ask:
            bid_price = max(bid_price, best_bid + Decimal('0.001'))
            ask_price = min(ask_price, best_ask - Decimal('0.001'))
        
        # Ensure minimum spread
        if ask_price - bid_price < self.min_spread:
            center = (bid_price + ask_price) / Decimal('2')
            bid_price = center - self.min_spread / Decimal('2')
            ask_price = center + self.min_spread / Decimal('2')
        
        # Ensure prices are within bounds
        bid_price = max(Decimal('0.01'), min(Decimal('0.99'), bid_price))
        ask_price = max(Decimal('0.01'), min(Decimal('0.99'), ask_price))
        
        # Don't quote if inventory is at limit
        if inventory >= self.max_inventory:
            bid_price = None  # Don't buy more
        elif inventory <= -self.max_inventory:
            ask_price = None  # Don't sell more
        
        return bid_price, ask_price
    
    def _calculate_order_size(
        self,
        inventory: Decimal,
        side: OrderSide
    ) -> Decimal:
        """Calculate order size based on inventory and risk"""
        # Base size as percentage of max position
        base_size = self.config.max_position_size * Decimal('0.1')
        
        # Adjust based on inventory
        inventory_ratio = abs(inventory) / self.max_inventory if self.max_inventory > 0 else Decimal('0')
        
        # Reduce size as inventory increases
        size_multiplier = Decimal('1') - inventory_ratio * Decimal('0.8')
        
        # Further reduce size if adding to position
        if (side == OrderSide.BUY and inventory > 0) or (side == OrderSide.SELL and inventory < 0):
            size_multiplier *= Decimal('0.5')
        
        final_size = base_size * size_multiplier
        
        # Ensure minimum size
        min_size = Decimal('10')
        return max(min_size, final_size)
    
    def _check_inventory_limits(
        self,
        market_id: str,
        side: OrderSide,
        size: Decimal
    ) -> bool:
        """Check if order would exceed inventory limits"""
        current_inventory = self.inventory.get(market_id, Decimal('0'))
        
        if side == OrderSide.BUY:
            new_inventory = current_inventory + size
        else:
            new_inventory = current_inventory - size
        
        return abs(new_inventory) <= self.max_inventory
    
    def _calculate_inventory_risk(self, inventory: Decimal) -> float:
        """Calculate risk score based on inventory level (0-1)"""
        if self.max_inventory == 0:
            return 0.0
        
        inventory_ratio = abs(inventory) / self.max_inventory
        
        # Exponential risk increase as inventory approaches limit
        risk = float(inventory_ratio ** 2)
        
        return min(1.0, risk)
    
    def _calculate_dynamic_spread(
        self,
        market_id: str,
        volatility: Optional[Decimal] = None
    ) -> Decimal:
        """Calculate dynamic spread based on market conditions"""
        base_spread = self.spread_target
        
        # Adjust for volatility if provided
        if volatility:
            volatility_multiplier = Decimal('1') + volatility * Decimal('2')
            base_spread *= volatility_multiplier
        
        # Ensure within bounds
        return max(self.min_spread, min(self.max_spread, base_spread))
    
    def _calculate_signal_strength(
        self,
        order_book: OrderBook,
        inventory_risk: float
    ) -> SignalStrength:
        """Calculate signal strength for market making"""
        # Market making signals are typically moderate strength
        # Adjust based on spread opportunity and risk
        
        spread_ratio = float(order_book.spread / self.spread_target) if order_book.spread else 1.0
        
        if spread_ratio > 2.0 and inventory_risk < 0.3:
            return SignalStrength.STRONG
        elif spread_ratio > 1.5 and inventory_risk < 0.5:
            return SignalStrength.MODERATE
        elif inventory_risk > 0.7:
            return SignalStrength.WEAK
        else:
            return SignalStrength.MODERATE
    
    def _calculate_confidence(
        self,
        order_book: OrderBook,
        inventory_risk: float
    ) -> float:
        """Calculate confidence in market making opportunity"""
        # Base confidence on order book depth and spread
        depth_score = min(1.0, (order_book.total_bid_size + order_book.total_ask_size) / 1000)
        
        # Adjust for inventory risk
        risk_adjustment = 1.0 - inventory_risk * 0.5
        
        # Adjust for spread opportunity
        if order_book.spread:
            spread_score = float(min(1.0, order_book.spread / self.spread_target))
        else:
            spread_score = 0.5
        
        confidence = (depth_score * 0.4 + spread_score * 0.3 + risk_adjustment * 0.3)
        
        return max(0.1, min(0.95, confidence))
    
    def _generate_reasoning(
        self,
        bid_price: Optional[Decimal],
        ask_price: Optional[Decimal],
        inventory: Decimal
    ) -> str:
        """Generate reasoning for market making signal"""
        if bid_price and ask_price:
            spread_pct = float((ask_price - bid_price) / ask_price) * 100
            reasoning = f"Market making opportunity with {spread_pct:.1f}% spread"
        elif bid_price:
            reasoning = "One-sided market making (bid only)"
        elif ask_price:
            reasoning = "One-sided market making (ask only)"
        else:
            reasoning = "No market making opportunity"
        
        if inventory != 0:
            reasoning += f". Current inventory: {inventory:.0f} shares"
        
        return reasoning
    
    def _detect_arbitrage(self, order_book: OrderBook) -> Optional[MarketMakingSignal]:
        """Detect immediate arbitrage opportunities"""
        if not order_book.best_bid or not order_book.best_ask:
            return None
        
        # Check for crossed market
        if order_book.best_bid >= order_book.best_ask:
            # Immediate arbitrage opportunity
            signal = MarketMakingSignal(
                market_id=order_book.market_id,
                outcome="Yes",
                direction=SignalDirection.BUY,
                strength=SignalStrength.VERY_STRONG,
                bid_price=order_book.best_ask,  # Buy at ask
                ask_price=order_book.best_bid,  # Sell at bid
                bid_size=Decimal('100'),  # Placeholder
                ask_size=Decimal('100'),
                size=Decimal('100'),
                confidence=0.99,
                reasoning="Crossed market arbitrage opportunity",
                is_arbitrage=True,
                metadata={'strategy': self.name}
            )
            return signal
        
        return None
    
    def _update_inventory(
        self,
        market_id: str,
        side: OrderSide,
        size: Decimal
    ):
        """Update inventory after a fill"""
        if market_id not in self.inventory:
            self.inventory[market_id] = Decimal('0')
        
        if side == OrderSide.BUY:
            self.inventory[market_id] += size
        else:
            self.inventory[market_id] -= size
        
        # Track inventory history
        if market_id not in self.inventory_history:
            self.inventory_history[market_id] = []
        
        self.inventory_history[market_id].append(
            (datetime.now(), self.inventory[market_id])
        )
    
    def get_inventory_position(self, market_id: str) -> Decimal:
        """Get current inventory position for a market"""
        return self.inventory.get(market_id, Decimal('0'))
    
    def _cleanup_orders(self, market_id: str):
        """Remove filled or cancelled orders"""
        if market_id in self.active_orders:
            self.active_orders[market_id] = [
                order for order in self.active_orders[market_id]
                if order.status in [OrderStatus.OPEN, OrderStatus.PARTIAL]
            ]
    
    async def _cancel_stale_orders(
        self,
        market_id: str,
        max_age_seconds: int = 300
    ):
        """Cancel orders older than max age"""
        if market_id not in self.active_orders:
            return
        
        now = datetime.now()
        stale_orders = [
            order for order in self.active_orders[market_id]
            if order.status == OrderStatus.OPEN and
            (now - order.created_at).total_seconds() > max_age_seconds
        ]
        
        for order in stale_orders:
            try:
                await self.client.cancel_order(order.id)
                logger.info(f"Cancelled stale order {order.id}")
            except Exception as e:
                logger.error(f"Failed to cancel order {order.id}: {str(e)}")
    
    def _record_spread_capture(self, market_id: str, spread: Decimal):
        """Record a spread capture event"""
        if market_id not in self.spread_captures:
            self.spread_captures[market_id] = []
        
        self.spread_captures[market_id].append(spread)
    
    def _record_trade(
        self,
        market_id: str,
        side: OrderSide,
        size: Decimal,
        price: Decimal
    ):
        """Record a completed trade"""
        # Update inventory
        self._update_inventory(market_id, side, size)
        
        # Record in metrics
        self.metrics.total_trades += 1
    
    def get_market_metrics(self, market_id: str) -> Dict[str, Any]:
        """Get performance metrics for a specific market"""
        spread_list = self.spread_captures.get(market_id, [])
        
        if spread_list:
            average_spread = sum(spread_list) / len(spread_list)
        else:
            average_spread = Decimal('0')
        
        return {
            'spread_captures': len(spread_list),
            'average_spread': average_spread,
            'current_inventory': self.inventory.get(market_id, Decimal('0')),
            'inventory_risk': self._calculate_inventory_risk(
                self.inventory.get(market_id, Decimal('0'))
            )
        }
    
    def calculate_market_pnl(
        self,
        market_id: str,
        current_price: Decimal
    ) -> Decimal:
        """Calculate P&L for a market"""
        # Simplified P&L calculation
        # In practice would track all trades and calculate realized + unrealized
        inventory = self.inventory.get(market_id, Decimal('0'))
        
        # Assume average entry at mid price
        unrealized_pnl = inventory * (current_price - Decimal('0.5'))
        
        # Add spread captures
        spreads = self.spread_captures.get(market_id, [])
        realized_pnl = sum(spreads) if spreads else Decimal('0')
        
        return realized_pnl + unrealized_pnl
    
    def _calculate_adaptive_size(
        self,
        base_size: Decimal,
        liquidity: Decimal,
        volatility: Decimal
    ) -> Decimal:
        """Calculate adaptive order size based on market conditions"""
        # Scale with liquidity
        liquidity_factor = min(Decimal('2'), liquidity / Decimal('50000'))
        
        # Scale inversely with volatility
        volatility_factor = Decimal('1') / (Decimal('1') + volatility * Decimal('5'))
        
        return base_size * liquidity_factor * volatility_factor
    
    async def _place_order(self, signal: TradingSignal) -> Optional[Order]:
        """Place order based on market making signal"""
        from ..models import OrderType
        
        if not isinstance(signal, MarketMakingSignal):
            logger.error("Invalid signal type for market maker")
            return None
        
        try:
            orders_placed = []
            
            # Place bid order if we have a bid price and size
            if signal.bid_price and signal.bid_size > 0:
                bid_order = Order(
                    id=f"mm_bid_{signal.market_id}_{datetime.now().timestamp()}",
                    market_id=signal.market_id,
                    outcome_id=signal.outcome,
                    side=OrderSide.BUY,
                    type=OrderType.LIMIT,
                    size=float(signal.bid_size),
                    price=float(signal.bid_price),
                    status=OrderStatus.PENDING,
                    created_at=datetime.now()
                )
                
                # In production, submit to CLOB API
                logger.info(
                    f"Placing BID order: {signal.bid_size} shares at {signal.bid_price} "
                    f"for market {signal.market_id}"
                )
                
                # Track active order
                if signal.market_id not in self.active_orders:
                    self.active_orders[signal.market_id] = []
                self.active_orders[signal.market_id].append(bid_order)
                orders_placed.append(bid_order)
            
            # Place ask order if we have an ask price and size
            if signal.ask_price and signal.ask_size > 0:
                ask_order = Order(
                    id=f"mm_ask_{signal.market_id}_{datetime.now().timestamp()}",
                    market_id=signal.market_id,
                    outcome_id=signal.outcome,
                    side=OrderSide.SELL,
                    type=OrderType.LIMIT,
                    size=float(signal.ask_size),
                    price=float(signal.ask_price),
                    status=OrderStatus.PENDING,
                    created_at=datetime.now()
                )
                
                # In production, submit to CLOB API
                logger.info(
                    f"Placing ASK order: {signal.ask_size} shares at {signal.ask_price} "
                    f"for market {signal.market_id}"
                )
                
                # Track active order
                if signal.market_id not in self.active_orders:
                    self.active_orders[signal.market_id] = []
                self.active_orders[signal.market_id].append(ask_order)
                orders_placed.append(ask_order)
            
            # Return the first order (for compatibility with base class)
            return orders_placed[0] if orders_placed else None
            
        except Exception as e:
            logger.error(f"Error placing market making orders: {str(e)}")
            return None
    
    async def update_quotes(self, market: Market):
        """Update quotes for an active market"""
        try:
            # Cancel stale orders first
            await self._cancel_stale_orders(market.id, self.quote_refresh_interval)
            
            # Generate new signal
            signal = await self.analyze_market(market)
            
            if signal:
                # Place new orders
                await self.execute_signal(signal)
                
        except Exception as e:
            logger.error(f"Error updating quotes for {market.id}: {str(e)}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        base_summary = super().get_performance_summary()
        
        # Add market making specific metrics
        total_spread_captures = sum(len(captures) for captures in self.spread_captures.values())
        
        avg_spreads = []
        for market_id, spreads in self.spread_captures.items():
            if spreads:
                avg_spreads.append(sum(spreads) / len(spreads))
        
        overall_avg_spread = sum(avg_spreads) / len(avg_spreads) if avg_spreads else Decimal('0')
        
        base_summary['market_making_metrics'] = {
            'total_spread_captures': total_spread_captures,
            'average_spread_captured': float(overall_avg_spread),
            'active_markets': len(self.inventory),
            'total_inventory_value': sum(abs(inv) for inv in self.inventory.values()),
            'markets_at_risk_limit': sum(
                1 for inv in self.inventory.values() 
                if abs(inv) >= self.max_inventory * Decimal('0.8')
            )
        }
        
        return base_summary