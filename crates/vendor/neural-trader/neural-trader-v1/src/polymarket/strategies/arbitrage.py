"""
Cross-Market Arbitrage Detection Strategy

Identifies and exploits price discrepancies between correlated markets
or between YES/NO shares in the same market.
"""

import asyncio
import logging
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Tuple, Optional, Any

from ..models import Market, MarketStatus, Order, OrderSide
from .base import (
    PolymarketStrategy, StrategyConfig, TradingSignal, SignalStrength,
    SignalDirection, StrategyError
)

logger = logging.getLogger(__name__)


class ArbitrageType(Enum):
    """Types of arbitrage opportunities"""
    YES_NO = "yes_no"  # YES + NO != 1.0
    CROSS_MARKET = "cross_market"  # Price differences between correlated markets
    TEMPORAL = "temporal"  # Mean reversion opportunities


@dataclass
class ArbitrageSignal(TradingSignal):
    """Extended signal for arbitrage opportunities"""
    market_a_id: str = ""
    market_b_id: str = ""  # Same as market_a for YES/NO arbitrage
    arbitrage_type: ArbitrageType = ArbitrageType.YES_NO
    expected_profit: Decimal = Decimal("0")
    trade_direction: str = "buy"  # "buy" or "sell"
    price_a: Decimal = Decimal("0")
    price_b: Decimal = Decimal("0")
    size_a: Decimal = Decimal("0")
    size_b: Decimal = Decimal("0")
    correlation: Optional[float] = None
    
    def __post_init__(self):
        """Validate arbitrage signal"""
        # Set inherited fields for compatibility
        if not self.market_id:
            self.market_id = self.market_a_id
        if not self.outcome:
            self.outcome = "Yes"
        if not self.direction:
            self.direction = SignalDirection.BUY if self.trade_direction == "buy" else SignalDirection.SELL
        if not self.target_price:
            self.target_price = self.price_a
        if not self.size:
            self.size = min(self.size_a, self.size_b)
        
        super().__post_init__()
        
        if self.expected_profit <= 0:
            raise ValueError("Expected profit must be positive")
        
        if self.arbitrage_type == ArbitrageType.YES_NO and self.market_a_id != self.market_b_id:
            raise ValueError("YES/NO arbitrage must be within same market")
    
    @property
    def is_profitable(self) -> bool:
        """Check if arbitrage is profitable after costs"""
        return self.expected_profit > 0
    
    @property
    def risk_adjusted_profit(self) -> Decimal:
        """Calculate risk-adjusted profit"""
        return self.expected_profit * Decimal(str(self.confidence))
    
    @property
    def total_cost(self) -> Decimal:
        """Calculate total cost of arbitrage"""
        if self.trade_direction == "buy":
            return (self.price_a * self.size_a) + (self.price_b * self.size_b)
        else:
            return Decimal('0')  # Selling doesn't require upfront cost


class ArbitrageStrategy(PolymarketStrategy):
    """
    Detects and executes arbitrage opportunities in prediction markets.
    
    Types of arbitrage:
    1. YES/NO arbitrage: When YES + NO prices != 1.0
    2. Cross-market: Price differences between correlated markets
    3. Temporal: Price inefficiencies over time
    """
    
    def __init__(self,
                 client,
                 config: Optional[StrategyConfig] = None,
                 min_profit_threshold: Decimal = Decimal('0.01'),
                 transaction_cost: Decimal = Decimal('0.002'),
                 max_position_size: Decimal = Decimal('1000.0'),
                 correlation_threshold: float = 0.7,
                 max_price_deviation: Decimal = Decimal('0.1')):
        """
        Initialize arbitrage strategy.
        
        Args:
            client: Polymarket API client
            config: Strategy configuration
            min_profit_threshold: Minimum profit percentage to consider
            transaction_cost: Cost per transaction as percentage
            max_position_size: Maximum position size in USD
            correlation_threshold: Minimum correlation for cross-market arb
            max_price_deviation: Maximum price deviation for temporal arb
        """
        super().__init__(client, config, "ArbitrageStrategy")
        
        self.min_profit_threshold = min_profit_threshold
        self.transaction_cost = transaction_cost
        self.max_position_size = max_position_size
        self.correlation_threshold = correlation_threshold
        self.max_price_deviation = max_price_deviation
        
        # Performance tracking
        self.arbitrage_history: List[Dict[str, Any]] = []
        self.correlation_cache: Dict[str, np.ndarray] = {}
        self.price_history: Dict[str, List[Tuple[datetime, Decimal]]] = {}
        
        logger.info(f"Initialized {self.name} with min profit {self.min_profit_threshold}")
    
    async def should_trade_market(self, market: Market) -> bool:
        """
        Determine if market is suitable for arbitrage
        
        Args:
            market: Market to evaluate
            
        Returns:
            True if market is suitable for arbitrage trading
        """
        # Only trade active markets
        if market.status != MarketStatus.ACTIVE:
            return False
        
        # Need sufficient time for arbitrage
        if market.end_date:
            time_to_expiry = market.end_date - datetime.now()
            if time_to_expiry < timedelta(hours=1):
                return False
        
        return True
    
    async def analyze_market(self, market: Market) -> Optional[TradingSignal]:
        """
        Analyze market for arbitrage opportunities
        
        Args:
            market: Market to analyze
            
        Returns:
            Arbitrage signal if opportunity found
        """
        signals = []
        
        # Check YES/NO arbitrage
        yes_no_signal = await self._detect_yes_no_arbitrage(market)
        if yes_no_signal:
            signals.append(yes_no_signal)
        
        # For temporal arbitrage, check price history
        temporal_signals = await self._detect_temporal_arbitrage_single(market)
        signals.extend(temporal_signals)
        
        # Return best opportunity
        if signals:
            return self._rank_opportunities(signals)[0]
        
        return None
    
    async def analyze_markets(self, markets: List[Market]) -> List[TradingSignal]:
        """
        Analyze multiple markets for cross-market arbitrage
        
        Args:
            markets: List of markets to analyze
            
        Returns:
            List of arbitrage signals
        """
        signals = []
        
        # Individual market arbitrage
        for market in markets:
            signal = await self.analyze_market(market)
            if signal:
                signals.append(signal)
        
        # Cross-market arbitrage
        if len(markets) >= 2:
            cross_market_signals = await self._detect_cross_market_arbitrage(markets)
            signals.extend(cross_market_signals)
        
        # Rank and filter
        ranked_signals = self._rank_opportunities(signals)
        
        # Apply risk limits
        valid_signals = []
        for signal in ranked_signals:
            if self._validate_arbitrage_signal(signal):
                valid_signals.append(signal)
        
        return valid_signals[:5]  # Limit to top 5 opportunities
    
    async def _detect_yes_no_arbitrage(self, market: Market) -> Optional[ArbitrageSignal]:
        """
        Detect arbitrage in YES/NO shares of the same market.
        
        In efficient markets: YES + NO = 1.0
        Arbitrage exists when this doesn't hold (accounting for fees).
        """
        yes_price = market.current_prices.get("Yes", Decimal('0'))
        no_price = market.current_prices.get("No", Decimal('0'))
        
        if yes_price == 0 or no_price == 0:
            return None
        
        total_price = yes_price + no_price
        
        # Account for transaction costs
        effective_cost = 2 * self.transaction_cost
        
        if total_price < Decimal('1.0') - effective_cost:
            # Buy both YES and NO, guaranteed $1 payout
            profit = Decimal('1.0') - total_price - effective_cost
            if profit > self.min_profit_threshold:
                trade_size = self._calculate_optimal_size(
                    expected_profit=profit,
                    confidence=0.95
                )
                
                signal = ArbitrageSignal(
                    market_id=market.id,
                    market_a_id=market.id,
                    market_b_id=market.id,
                    outcome="Yes",
                    arbitrage_type=ArbitrageType.YES_NO,
                    expected_profit=profit,
                    confidence=0.99,  # Very high confidence
                    trade_direction="buy",
                    price_a=yes_price,
                    price_b=no_price,
                    size_a=trade_size,
                    size_b=trade_size,
                    strength=SignalStrength.VERY_STRONG,
                    reasoning=f"YES/NO arbitrage: sum={total_price:.3f} < 1.0"
                )
                return signal
        
        elif total_price > Decimal('1.0') + effective_cost:
            # Sell both YES and NO (if we have inventory)
            profit = total_price - Decimal('1.0') - effective_cost
            if profit > self.min_profit_threshold:
                trade_size = self._calculate_optimal_size(
                    expected_profit=profit,
                    confidence=0.95
                )
                
                signal = ArbitrageSignal(
                    market_id=market.id,
                    market_a_id=market.id,
                    market_b_id=market.id,
                    outcome="Yes",
                    arbitrage_type=ArbitrageType.YES_NO,
                    expected_profit=profit,
                    confidence=0.99,
                    trade_direction="sell",
                    price_a=yes_price,
                    price_b=no_price,
                    size_a=trade_size,
                    size_b=trade_size,
                    strength=SignalStrength.VERY_STRONG,
                    reasoning=f"YES/NO arbitrage: sum={total_price:.3f} > 1.0"
                )
                return signal
        
        return None
    
    async def _detect_cross_market_arbitrage(
        self,
        markets: List[Market]
    ) -> List[ArbitrageSignal]:
        """Detect arbitrage between correlated markets"""
        opportunities = []
        
        # Calculate or retrieve correlation matrix
        correlation_matrix = await self._calculate_correlation_matrix(markets)
        n_markets = len(markets)
        
        for i in range(n_markets):
            for j in range(i + 1, n_markets):
                correlation = correlation_matrix[i, j]
                
                if abs(correlation) >= self.correlation_threshold:
                    market_a = markets[i]
                    market_b = markets[j]
                    
                    price_a = market_a.current_prices.get("Yes", Decimal('0'))
                    price_b = market_b.current_prices.get("Yes", Decimal('0'))
                    
                    if price_a == 0 or price_b == 0:
                        continue
                    
                    # Expected price relationship based on correlation
                    if correlation > 0:
                        expected_b = price_a * Decimal(str(correlation))
                        price_diff = abs(price_b - expected_b)
                    else:
                        expected_b = (Decimal('1') - price_a) * Decimal(str(abs(correlation)))
                        price_diff = abs(price_b - expected_b)
                    
                    # Check if price difference is significant
                    if price_diff > self.min_profit_threshold + 2 * self.transaction_cost:
                        # Calculate expected profit
                        profit_per_share = price_diff - 2 * self.transaction_cost
                        confidence = min(abs(correlation), 0.95)
                        
                        # Determine trade direction
                        if price_b > expected_b:
                            trade_direction = "sell_b_buy_a" if correlation > 0 else "buy_b_sell_a"
                        else:
                            trade_direction = "buy_b_sell_a" if correlation > 0 else "sell_b_buy_a"
                        
                        # Calculate trade size
                        trade_size = self._calculate_optimal_size(profit_per_share, confidence)
                        
                        signal = ArbitrageSignal(
                            market_id=market_a.id,
                            market_a_id=market_a.id,
                            market_b_id=market_b.id,
                            outcome="Yes",
                            arbitrage_type=ArbitrageType.CROSS_MARKET,
                            expected_profit=profit_per_share,
                            confidence=confidence,
                            trade_direction="buy" if "buy" in trade_direction else "sell",
                            price_a=price_a,
                            price_b=price_b,
                            size_a=trade_size,
                            size_b=trade_size,
                            correlation=float(correlation),
                            strength=self._calculate_signal_strength(confidence, profit_per_share),
                            reasoning=f"Cross-market arbitrage: correlation={correlation:.2f}"
                        )
                        opportunities.append(signal)
        
        return opportunities
    
    async def _detect_temporal_arbitrage_single(
        self,
        market: Market
    ) -> List[ArbitrageSignal]:
        """Detect temporal arbitrage for a single market"""
        market_id = market.id
        
        # Update price history
        current_price = market.current_prices.get("Yes", Decimal('0'))
        if current_price > 0:
            if market_id not in self.price_history:
                self.price_history[market_id] = []
            
            self.price_history[market_id].append((datetime.now(), current_price))
            
            # Keep only recent history
            cutoff_time = datetime.now() - timedelta(hours=24)
            self.price_history[market_id] = [
                (ts, price) for ts, price in self.price_history[market_id]
                if ts > cutoff_time
            ]
        
        # Check for temporal arbitrage
        if market_id not in self.price_history or len(self.price_history[market_id]) < 20:
            return []
        
        prices = [float(p) for _, p in self.price_history[market_id]]
        timestamps = [ts for ts, _ in self.price_history[market_id]]
        
        # Calculate statistics
        prices_array = np.array(prices)
        mean_price = np.mean(prices_array)
        std_price = np.std(prices_array)
        current_price_float = float(current_price)
        
        if std_price > 0:
            z_score = (current_price_float - mean_price) / std_price
            
            # Look for extreme deviations
            if abs(z_score) > 2.0:
                # Calculate mean reversion probability
                reversion_prob = 1 - (1 / (1 + np.exp(-abs(z_score) + 2)))
                
                # Expected profit from mean reversion
                expected_move = Decimal(str(mean_price)) - current_price
                profit = abs(expected_move) - self.transaction_cost
                
                if profit > self.min_profit_threshold:
                    trade_direction = "buy" if current_price < Decimal(str(mean_price)) else "sell"
                    trade_size = self._calculate_optimal_size(profit, reversion_prob)
                    
                    signal = ArbitrageSignal(
                        market_id=market_id,
                        market_a_id=market_id,
                        market_b_id=market_id,
                        outcome="Yes",
                        arbitrage_type=ArbitrageType.TEMPORAL,
                        expected_profit=profit,
                        confidence=reversion_prob,
                        trade_direction=trade_direction,
                        price_a=current_price,
                        price_b=Decimal(str(mean_price)),  # Target price
                        size_a=trade_size,
                        size_b=trade_size,
                        strength=self._calculate_signal_strength(reversion_prob, profit),
                        reasoning=f"Temporal arbitrage: z-score={z_score:.2f}"
                    )
                    return [signal]
        
        return []
    
    def _calculate_optimal_size(self, 
                              expected_profit: Decimal,
                              confidence: float) -> Decimal:
        """
        Calculate optimal position size using Kelly Criterion.
        
        Args:
            expected_profit: Expected profit percentage
            confidence: Confidence in the opportunity (0-1)
            
        Returns:
            Optimal position size
        """
        # Kelly fraction: f = p - q/b
        # p = probability of win, q = probability of loss, b = odds
        p = confidence
        q = 1 - confidence
        b = float(expected_profit)  # Simplified: profit-to-loss ratio
        
        kelly_fraction = (p * b - q) / b if b > 0 else 0
        
        # Apply conservative factor (quarter Kelly)
        conservative_fraction = kelly_fraction * 0.25
        
        # Ensure fraction is between 0 and 1
        fraction = max(0, min(1, conservative_fraction))
        
        return self.max_position_size * Decimal(str(fraction))
    
    def _calculate_signal_strength(
        self,
        confidence: float,
        expected_profit: Decimal
    ) -> SignalStrength:
        """Calculate signal strength from confidence and profit"""
        score = float(confidence * expected_profit)
        
        if score >= 0.15:
            return SignalStrength.VERY_STRONG
        elif score >= 0.10:
            return SignalStrength.STRONG
        elif score >= 0.05:
            return SignalStrength.MODERATE
        elif score >= 0.02:
            return SignalStrength.WEAK
        else:
            return SignalStrength.VERY_WEAK
    
    async def _calculate_correlation_matrix(
        self,
        markets: List[Market]
    ) -> np.ndarray:
        """Calculate correlation matrix between markets"""
        # Check cache
        market_ids = "_".join(sorted([m.id for m in markets]))
        if market_ids in self.correlation_cache:
            return self.correlation_cache[market_ids]
        
        # Get price history for correlation calculation
        n_markets = len(markets)
        correlation_matrix = np.eye(n_markets)
        
        # In production, this would fetch historical prices
        # For now, use a simple heuristic based on market categories
        for i in range(n_markets):
            for j in range(i + 1, n_markets):
                if i != j:
                    # Check if markets are in same category
                    cat_i = markets[i].metadata.get('category', '')
                    cat_j = markets[j].metadata.get('category', '')
                    
                    if cat_i == cat_j and cat_i:
                        # Same category - assume some correlation
                        correlation = 0.7
                    else:
                        # Different categories - assume low correlation
                        correlation = 0.1
                    
                    # Check for inverse relationships in questions
                    if self._are_inverse_markets(markets[i], markets[j]):
                        correlation = -0.9
                    
                    correlation_matrix[i, j] = correlation
                    correlation_matrix[j, i] = correlation
        
        # Cache result
        self.correlation_cache[market_ids] = correlation_matrix
        
        return correlation_matrix
    
    def _are_inverse_markets(self, market_a: Market, market_b: Market) -> bool:
        """Check if two markets have inverse relationship"""
        # Simple heuristic - check for opposite keywords
        question_a = market_a.question.lower()
        question_b = market_b.question.lower()
        
        inverse_pairs = [
            ('win', 'lose'),
            ('increase', 'decrease'),
            ('rise', 'fall'),
            ('bull', 'bear'),
            ('above', 'below')
        ]
        
        for word1, word2 in inverse_pairs:
            if (word1 in question_a and word2 in question_b) or \
               (word2 in question_a and word1 in question_b):
                return True
        
        return False
    
    def _rank_opportunities(self, 
                          opportunities: List[ArbitrageSignal]) -> List[ArbitrageSignal]:
        """
        Rank arbitrage opportunities by expected risk-adjusted return.
        
        Args:
            opportunities: List of detected opportunities
            
        Returns:
            Sorted list of opportunities (best first)
        """
        def score_opportunity(opp: ArbitrageSignal) -> float:
            # Score based on profit, confidence, and type
            base_score = float(opp.expected_profit * Decimal(str(opp.confidence)))
            
            # Type multipliers
            type_multipliers = {
                ArbitrageType.YES_NO: 1.5,      # Highest priority - guaranteed profit
                ArbitrageType.CROSS_MARKET: 1.0, # Medium priority
                ArbitrageType.TEMPORAL: 0.8      # Lower priority - less certain
            }
            
            return base_score * type_multipliers.get(opp.arbitrage_type, 1.0)
        
        return sorted(opportunities, key=score_opportunity, reverse=True)
    
    def _validate_arbitrage_signal(self, signal: ArbitrageSignal) -> bool:
        """Validate arbitrage signal meets requirements"""
        # Check profit threshold
        if signal.expected_profit < self.min_profit_threshold:
            return False
        
        # Check confidence
        if signal.confidence < self.config.min_confidence:
            return False
        
        # Check signal strength
        if signal.strength.value < self.config.min_signal_strength.value:
            return False
        
        # Check risk limits
        if not self._check_risk_limits(signal):
            return False
        
        return True
    
    async def _place_arbitrage_orders(
        self,
        signal: ArbitrageSignal
    ) -> List[Order]:
        """Place orders for arbitrage execution"""
        orders = []
        
        # This would integrate with the CLOB client to place actual orders
        # For now, return empty list
        logger.info(f"Would place arbitrage orders for {signal.arbitrage_type}")
        
        return orders
    
    async def _revalidate_arbitrage(
        self,
        signal: ArbitrageSignal
    ) -> bool:
        """Revalidate arbitrage opportunity before execution"""
        # Get current prices
        current_prices = await self._get_current_prices([signal.market_a_id, signal.market_b_id])
        
        if not current_prices:
            return False
        
        # Check if arbitrage still exists
        if signal.arbitrage_type == ArbitrageType.YES_NO:
            yes_price = current_prices.get(f"{signal.market_a_id}_Yes", Decimal('0'))
            no_price = current_prices.get(f"{signal.market_a_id}_No", Decimal('0'))
            
            total = yes_price + no_price
            if signal.trade_direction == "buy":
                return total < Decimal('1.0') - 2 * self.transaction_cost
            else:
                return total > Decimal('1.0') + 2 * self.transaction_cost
        
        return True
    
    async def _get_current_prices(
        self,
        market_ids: List[str]
    ) -> Dict[str, Decimal]:
        """Get current prices for markets"""
        # In production, this would fetch from API
        # For now, return mock data
        prices = {}
        for market_id in market_ids:
            prices[f"{market_id}_Yes"] = Decimal('0.5')
            prices[f"{market_id}_No"] = Decimal('0.5')
        
        return prices
    
    def _record_arbitrage_result(
        self,
        arbitrage_type: ArbitrageType,
        expected_profit: Decimal,
        realized_profit: Decimal,
        slippage: Decimal
    ):
        """Record arbitrage execution result"""
        result = {
            'type': arbitrage_type.value,
            'expected_profit': float(expected_profit),
            'realized_profit': float(realized_profit),
            'slippage': float(slippage),
            'timestamp': datetime.now(),
            'success': realized_profit > 0
        }
        
        self.arbitrage_history.append(result)
        
        # Update metrics
        if realized_profit > 0:
            self.metrics.winning_trades += 1
        else:
            self.metrics.losing_trades += 1
        
        self.metrics.total_trades += 1
        self.metrics.total_pnl += realized_profit
    
    def get_arbitrage_metrics(self) -> Dict[str, Any]:
        """Get arbitrage-specific performance metrics"""
        if not self.arbitrage_history:
            return {
                'total_arbitrages': 0,
                'success_rate': 0.0,
                'average_slippage': Decimal('0')
            }
        
        total = len(self.arbitrage_history)
        successful = sum(1 for r in self.arbitrage_history if r['success'])
        
        total_slippage = sum(r['slippage'] for r in self.arbitrage_history)
        avg_slippage = Decimal(str(total_slippage / total)) if total > 0 else Decimal('0')
        
        # Group by type
        by_type = {}
        for arb_type in ArbitrageType:
            type_results = [r for r in self.arbitrage_history if r['type'] == arb_type.value]
            if type_results:
                by_type[arb_type.value] = {
                    'count': len(type_results),
                    'success_rate': sum(1 for r in type_results if r['success']) / len(type_results),
                    'avg_profit': sum(r['realized_profit'] for r in type_results) / len(type_results)
                }
        
        return {
            'total_arbitrages': total,
            'success_rate': successful / total if total > 0 else 0.0,
            'average_slippage': avg_slippage,
            'profit_by_type': by_type
        }
    
    async def _place_order(self, signal: TradingSignal) -> Optional[Order]:
        """Place order based on arbitrage signal"""
        from ..models import OrderType, OrderStatus
        
        if not isinstance(signal, ArbitrageSignal):
            logger.error("Invalid signal type for arbitrage")
            return None
        
        try:
            # For arbitrage, we typically need to place multiple orders
            # This method returns the primary order for base class compatibility
            
            # Revalidate arbitrage opportunity
            if not await self._revalidate_arbitrage(signal):
                logger.warning(f"Arbitrage opportunity no longer valid for {signal.market_a_id}")
                return None
            
            # For YES/NO arbitrage
            if signal.arbitrage_type == ArbitrageType.YES_NO:
                if signal.trade_direction == "buy":
                    # Buy both YES and NO
                    yes_order = Order(
                        id=f"arb_yes_{signal.market_id}_{datetime.now().timestamp()}",
                        market_id=signal.market_id,
                        outcome_id="Yes",
                        side=OrderSide.BUY,
                        type=OrderType.MARKET,  # Use market orders for arbitrage
                        size=float(signal.size_a),
                        price=float(signal.price_a),
                        status=OrderStatus.PENDING,
                        created_at=datetime.now()
                    )
                    
                    no_order = Order(
                        id=f"arb_no_{signal.market_id}_{datetime.now().timestamp()}",
                        market_id=signal.market_id,
                        outcome_id="No",
                        side=OrderSide.BUY,
                        type=OrderType.MARKET,
                        size=float(signal.size_b),
                        price=float(signal.price_b),
                        status=OrderStatus.PENDING,
                        created_at=datetime.now()
                    )
                    
                    logger.info(
                        f"Placing YES/NO arbitrage BUY orders for {signal.market_id} - "
                        f"YES: {signal.size_a}@{signal.price_a}, NO: {signal.size_b}@{signal.price_b}"
                    )
                    
                    # Track arbitrage execution
                    self._record_arbitrage_result(
                        arbitrage_type=signal.arbitrage_type,
                        expected_profit=signal.expected_profit,
                        realized_profit=signal.expected_profit,  # Will be updated after execution
                        slippage=Decimal('0')  # Will be updated after execution
                    )
                    
                    return yes_order  # Return primary order
                    
                else:  # sell
                    # Sell both YES and NO (if we have inventory)
                    yes_order = Order(
                        id=f"arb_yes_{signal.market_id}_{datetime.now().timestamp()}",
                        market_id=signal.market_id,
                        outcome_id="Yes",
                        side=OrderSide.SELL,
                        type=OrderType.MARKET,
                        size=float(signal.size_a),
                        price=float(signal.price_a),
                        status=OrderStatus.PENDING,
                        created_at=datetime.now()
                    )
                    
                    logger.info(
                        f"Placing YES/NO arbitrage SELL orders for {signal.market_id}"
                    )
                    
                    return yes_order
            
            # For cross-market arbitrage
            elif signal.arbitrage_type == ArbitrageType.CROSS_MARKET:
                # Place orders in both markets
                order_a = Order(
                    id=f"arb_cross_a_{signal.market_a_id}_{datetime.now().timestamp()}",
                    market_id=signal.market_a_id,
                    outcome_id="Yes",
                    side=OrderSide.BUY if "buy" in signal.trade_direction else OrderSide.SELL,
                    type=OrderType.MARKET,
                    size=float(signal.size_a),
                    price=float(signal.price_a),
                    status=OrderStatus.PENDING,
                    created_at=datetime.now()
                )
                
                logger.info(
                    f"Placing cross-market arbitrage orders - "
                    f"Market A: {signal.market_a_id}, Market B: {signal.market_b_id}"
                )
                
                return order_a
            
            # For temporal arbitrage
            elif signal.arbitrage_type == ArbitrageType.TEMPORAL:
                order = Order(
                    id=f"arb_temp_{signal.market_id}_{datetime.now().timestamp()}",
                    market_id=signal.market_id,
                    outcome_id="Yes",
                    side=OrderSide.BUY if signal.trade_direction == "buy" else OrderSide.SELL,
                    type=OrderType.LIMIT,  # Use limit orders for temporal
                    size=float(signal.size),
                    price=float(signal.target_price),
                    status=OrderStatus.PENDING,
                    created_at=datetime.now()
                )
                
                logger.info(
                    f"Placing temporal arbitrage {signal.trade_direction} order for {signal.market_id}"
                )
                
                return order
                
        except Exception as e:
            logger.error(f"Error placing arbitrage orders: {str(e)}")
            return None