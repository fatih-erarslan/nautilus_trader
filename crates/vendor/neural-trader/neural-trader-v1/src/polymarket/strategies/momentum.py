"""
Momentum Trading Strategy

Follows probability trends in prediction markets, betting on continued
movement in the same direction.
"""

import asyncio
import logging
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple
from collections import deque

from ..models import Market, MarketStatus, Order, OrderSide
from .base import (
    PolymarketStrategy, StrategyConfig, TradingSignal, SignalStrength,
    SignalDirection, StrategyError
)

logger = logging.getLogger(__name__)


@dataclass
class MomentumIndicators:
    """Container for momentum technical indicators"""
    momentum: float = 0.0  # Rate of change
    velocity: float = 0.0  # First derivative
    acceleration: float = 0.0  # Second derivative
    volume_momentum: float = 0.0
    trend_direction: int = 0  # 1 for up, -1 for down, 0 for neutral
    short_ma: float = 0.0
    long_ma: float = 0.0
    macd: float = 0.0
    rsi: float = 50.0
    volatility: float = 0.0
    
    @property
    def is_bullish(self) -> bool:
        """Check if indicators are bullish"""
        return self.momentum > 0 and self.trend_direction > 0
    
    @property
    def is_bearish(self) -> bool:
        """Check if indicators are bearish"""
        return self.momentum < 0 and self.trend_direction < 0
    
    def calculate_composite_score(self) -> float:
        """Calculate weighted composite momentum score"""
        weights = {
            'momentum': 0.3,
            'velocity': 0.2,
            'acceleration': 0.15,
            'volume_momentum': 0.15,
            'macd': 0.2
        }
        
        # Normalize indicators
        norm_momentum = np.tanh(self.momentum * 10)
        norm_velocity = np.tanh(self.velocity * 100)
        norm_acceleration = np.tanh(self.acceleration * 1000)
        norm_volume = np.tanh(self.volume_momentum)
        norm_macd = np.tanh(self.macd * 10)
        
        score = (weights['momentum'] * norm_momentum +
                weights['velocity'] * norm_velocity +
                weights['acceleration'] * norm_acceleration +
                weights['volume_momentum'] * norm_volume +
                weights['macd'] * norm_macd)
        
        return score


class MomentumStrategy(PolymarketStrategy):
    """
    Trading strategy that follows probability momentum.
    
    Key principles:
    1. Strong trends tend to continue
    2. Volume confirms momentum
    3. Acceleration provides early signals
    4. Risk management through position sizing
    """
    
    def __init__(
        self,
        client,
        config: Optional[StrategyConfig] = None,
        short_period: int = 5,
        long_period: int = 20,
        momentum_threshold: Decimal = Decimal('0.05'),
        volume_confirmation: bool = True,
        trend_strength_threshold: float = 0.3,
        max_volatility: Decimal = Decimal('0.5'),
        use_acceleration: bool = True,
        stop_loss_pct: Decimal = Decimal('0.1')
    ):
        """
        Initialize momentum strategy.
        
        Args:
            client: Polymarket API client
            config: Strategy configuration
            short_period: Short-term moving average period
            long_period: Long-term moving average period
            momentum_threshold: Minimum momentum to trigger signal
            volume_confirmation: Require volume confirmation
            trend_strength_threshold: Minimum trend strength
            max_volatility: Maximum volatility tolerance
            use_acceleration: Consider acceleration in signals
            stop_loss_pct: Stop loss percentage
        """
        super().__init__(client, config, "MomentumStrategy")
        
        self.short_period = short_period
        self.long_period = long_period
        self.momentum_threshold = momentum_threshold
        self.volume_confirmation = volume_confirmation
        self.trend_strength_threshold = trend_strength_threshold
        self.max_volatility = max_volatility
        self.use_acceleration = use_acceleration
        self.stop_loss_pct = stop_loss_pct
        
        # Data storage
        self.price_history: Dict[str, deque] = {}
        self.volume_history: Dict[str, deque] = {}
        self.momentum_cache: Dict[str, MomentumIndicators] = {}
        
        logger.info(f"Initialized {self.name} with momentum threshold {self.momentum_threshold}")
    
    async def should_trade_market(self, market: Market) -> bool:
        """
        Determine if market is suitable for momentum trading
        
        Args:
            market: Market to evaluate
            
        Returns:
            True if market is suitable for momentum trading
        """
        # Check if market is active
        if market.status != MarketStatus.ACTIVE:
            return False
        
        # Need some history for momentum
        volume = market.metadata.get('volume_24h', 0)
        if volume < 5000:
            logger.debug(f"Market {market.id} volume too low for momentum")
            return False
        
        # Check time to expiry
        if market.end_date:
            time_to_expiry = market.end_date - datetime.now()
            if time_to_expiry < timedelta(hours=12):
                logger.debug(f"Market {market.id} expiring too soon for momentum")
                return False
        
        # Check price is not at extremes (less room for momentum)
        yes_price = market.current_prices.get("Yes", Decimal('0.5'))
        if yes_price < Decimal('0.05') or yes_price > Decimal('0.95'):
            logger.debug(f"Market {market.id} price too extreme for momentum")
            return False
        
        return True
    
    async def analyze_market(self, market: Market) -> Optional[TradingSignal]:
        """
        Analyze market momentum and generate trading signal
        
        Args:
            market: Market to analyze
            
        Returns:
            Trading signal if opportunity found
        """
        try:
            # Update price history
            self._update_price_history(market)
            
            # Calculate momentum indicators
            indicators = self._calculate_momentum_indicators(market.id)
            if not indicators:
                logger.debug(f"Insufficient data for momentum analysis of {market.id}")
                return None
            
            # Check momentum threshold
            if abs(indicators.momentum) < float(self.momentum_threshold):
                logger.debug(f"Momentum below threshold for {market.id}")
                return None
            
            # Volume confirmation if required
            if self.volume_confirmation and indicators.volume_momentum < 0:
                logger.debug(f"No volume confirmation for {market.id}")
                return None
            
            # Calculate trend strength
            trend_strength = self._calculate_trend_strength(indicators)
            if trend_strength < self.trend_strength_threshold:
                logger.debug(f"Trend strength too weak for {market.id}")
                return None
            
            # Check volatility
            if indicators.volatility > float(self.max_volatility):
                logger.debug(f"Volatility too high for {market.id}")
                return None
            
            # Generate signal
            signal = self._generate_momentum_signal(market, indicators, trend_strength)
            
            # Update cache
            self.momentum_cache[market.id] = indicators
            
            return signal
            
        except Exception as e:
            logger.error(f"Error analyzing momentum for {market.id}: {str(e)}")
            return None
    
    def _update_price_history(self, market: Market):
        """Update price and volume history for market"""
        market_id = market.id
        
        if market_id not in self.price_history:
            self.price_history[market_id] = deque(maxlen=self.long_period * 2)
            self.volume_history[market_id] = deque(maxlen=self.long_period * 2)
        
        # Get current price
        yes_price = float(market.current_prices.get("Yes", Decimal('0.5')))
        volume = float(market.metadata.get('volume_24h', 0))
        
        self.price_history[market_id].append(yes_price)
        self.volume_history[market_id].append(volume)
    
    def _calculate_momentum_indicators(self, market_id: str) -> Optional[MomentumIndicators]:
        """Calculate technical momentum indicators"""
        if market_id not in self.price_history:
            return None
        
        prices = list(self.price_history[market_id])
        volumes = list(self.volume_history[market_id])
        
        if len(prices) < self.long_period:
            return None
        
        indicators = MomentumIndicators()
        
        # Moving averages
        indicators.short_ma = np.mean(prices[-self.short_period:])
        indicators.long_ma = np.mean(prices[-self.long_period:])
        
        # Price momentum (rate of change)
        current_price = prices[-1]
        prev_price = prices[-self.short_period] if len(prices) >= self.short_period else prices[0]
        indicators.momentum = (current_price - prev_price) / prev_price if prev_price > 0 else 0
        
        # Velocity (first derivative)
        if len(prices) >= 3:
            indicators.velocity = (prices[-1] - prices[-3]) / 2
        
        # Acceleration (second derivative)
        if len(prices) >= 4:
            indicators.acceleration = prices[-1] - 2 * prices[-2] + prices[-3]
        
        # Volume momentum
        if len(volumes) >= self.short_period and sum(volumes) > 0:
            recent_vol = np.mean(volumes[-self.short_period:])
            past_vol = np.mean(volumes[-self.long_period:-self.short_period]) if len(volumes) >= self.long_period else recent_vol
            indicators.volume_momentum = (recent_vol - past_vol) / past_vol if past_vol > 0 else 0
        
        # Trend direction
        indicators.trend_direction = 1 if indicators.short_ma > indicators.long_ma else -1
        
        # MACD
        indicators.macd = indicators.short_ma - indicators.long_ma
        
        # RSI
        indicators.rsi = self._calculate_rsi(prices)
        
        # Volatility
        indicators.volatility = np.std(prices[-self.short_period:]) if len(prices) >= self.short_period else 0
        
        return indicators
    
    def _calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Calculate Relative Strength Index"""
        if len(prices) < period + 1:
            return 50.0
        
        price_changes = np.diff(prices[-period-1:])
        gains = price_changes[price_changes > 0]
        losses = -price_changes[price_changes < 0]
        
        avg_gain = np.mean(gains) if len(gains) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0.001
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_trend_strength(self, indicators: MomentumIndicators) -> float:
        """Calculate overall trend strength score"""
        # Composite score from momentum indicators
        momentum_score = indicators.calculate_composite_score()
        
        # Adjust for RSI extremes
        rsi_factor = 1.0
        if indicators.rsi > 70 or indicators.rsi < 30:
            rsi_factor = 0.8  # Reduce strength at extremes
        
        # Acceleration confirmation
        accel_factor = 1.0
        if self.use_acceleration:
            if indicators.momentum * indicators.acceleration > 0:
                accel_factor = 1.2  # Accelerating trend
            else:
                accel_factor = 0.8  # Decelerating trend
        
        trend_strength = abs(momentum_score) * rsi_factor * accel_factor
        
        return trend_strength
    
    def _generate_momentum_signal(
        self,
        market: Market,
        indicators: MomentumIndicators,
        trend_strength: float
    ) -> Optional[TradingSignal]:
        """Generate trading signal from momentum indicators"""
        # Determine direction
        if indicators.is_bullish:
            direction = SignalDirection.BUY
            outcome = "Yes"
        elif indicators.is_bearish:
            direction = SignalDirection.SELL
            outcome = "Yes"
        else:
            return None
        
        # Calculate target price
        current_price = market.current_prices.get("Yes", Decimal('0.5'))
        price_target = self._calculate_price_target(current_price, indicators, trend_strength)
        
        # Map trend strength to signal strength
        signal_strength = self._map_trend_to_signal_strength(trend_strength)
        
        # Calculate confidence
        confidence = self._calculate_confidence(indicators, trend_strength)
        
        # Calculate position size
        size = self._calculate_position_size(indicators, trend_strength, confidence)
        
        # Create signal
        signal = TradingSignal(
            market_id=market.id,
            outcome=outcome,
            direction=direction,
            strength=signal_strength,
            target_price=price_target,
            size=size,
            confidence=confidence,
            reasoning=self._generate_reasoning(indicators, trend_strength),
            metadata={
                'momentum': indicators.momentum,
                'velocity': indicators.velocity,
                'acceleration': indicators.acceleration,
                'rsi': indicators.rsi,
                'trend_strength': trend_strength,
                'strategy': self.name
            }
        )
        
        return signal
    
    def _calculate_price_target(
        self,
        current_price: Decimal,
        indicators: MomentumIndicators,
        trend_strength: float
    ) -> Decimal:
        """Calculate price target based on momentum"""
        # Base expected move from momentum
        base_move = Decimal(str(abs(indicators.momentum) * 0.5))  # 50% of momentum rate
        
        # Adjust by trend strength
        adjusted_move = base_move * Decimal(str(trend_strength))
        
        # Apply direction
        if indicators.trend_direction > 0:
            target = current_price + adjusted_move
        else:
            target = current_price - adjusted_move
        
        # Bound between 0 and 1
        return max(Decimal('0.01'), min(Decimal('0.99'), target))
    
    def _map_trend_to_signal_strength(self, trend_strength: float) -> SignalStrength:
        """Map trend strength to signal strength"""
        if trend_strength >= 0.8:
            return SignalStrength.VERY_STRONG
        elif trend_strength >= 0.6:
            return SignalStrength.STRONG
        elif trend_strength >= 0.4:
            return SignalStrength.MODERATE
        elif trend_strength >= 0.2:
            return SignalStrength.WEAK
        else:
            return SignalStrength.VERY_WEAK
    
    def _calculate_confidence(
        self,
        indicators: MomentumIndicators,
        trend_strength: float
    ) -> float:
        """Calculate confidence in momentum signal"""
        # Base confidence from trend strength
        base_confidence = min(trend_strength, 0.85)
        
        # RSI confirmation
        rsi_factor = 1.0
        if 30 < indicators.rsi < 70:  # Not overbought/oversold
            rsi_factor = 1.1
        else:
            rsi_factor = 0.9
        
        # Volume confirmation
        vol_factor = 1.0 + min(indicators.volume_momentum, 0.1)
        
        # Acceleration confirmation
        accel_factor = 1.0
        if indicators.momentum * indicators.acceleration > 0:
            accel_factor = 1.05
        
        confidence = base_confidence * rsi_factor * vol_factor * accel_factor
        
        return max(0.1, min(0.95, confidence))
    
    def _calculate_position_size(
        self,
        indicators: MomentumIndicators,
        trend_strength: float,
        confidence: float
    ) -> Decimal:
        """Calculate position size for momentum trade"""
        # Base allocation by trend strength
        if trend_strength >= 0.8:
            base_allocation = Decimal('0.12')
        elif trend_strength >= 0.6:
            base_allocation = Decimal('0.08')
        elif trend_strength >= 0.4:
            base_allocation = Decimal('0.05')
        else:
            base_allocation = Decimal('0.03')
        
        # Adjust by confidence
        allocation = base_allocation * Decimal(str(confidence))
        
        # Adjust for volatility (reduce size in high volatility)
        vol_factor = Decimal('1') - Decimal(str(min(indicators.volatility, 0.5)))
        allocation = allocation * vol_factor
        
        # Apply to max position size
        size = self.config.max_position_size * allocation
        
        return max(Decimal('10'), size)  # Minimum size
    
    def _generate_reasoning(
        self,
        indicators: MomentumIndicators,
        trend_strength: float
    ) -> str:
        """Generate human-readable reasoning for signal"""
        direction = "bullish" if indicators.is_bullish else "bearish"
        strength = "strong" if trend_strength >= 0.6 else "moderate"
        
        reasoning = (
            f"{strength.capitalize()} {direction} momentum detected with "
            f"{abs(indicators.momentum)*100:.1f}% price movement. "
            f"RSI: {indicators.rsi:.0f}, "
            f"Volume momentum: {indicators.volume_momentum*100:+.1f}%"
        )
        
        if indicators.acceleration != 0:
            accel_type = "accelerating" if indicators.momentum * indicators.acceleration > 0 else "decelerating"
            reasoning += f". Trend is {accel_type}"
        
        return reasoning
    
    async def should_exit_position(
        self,
        market: Market,
        entry_price: Decimal,
        position_direction: SignalDirection
    ) -> Tuple[bool, str]:
        """
        Determine if momentum position should be exited
        
        Args:
            market: Current market
            entry_price: Entry price
            position_direction: Direction of position
            
        Returns:
            Tuple of (should_exit, reason)
        """
        current_price = market.current_prices.get("Yes", Decimal('0.5'))
        
        # Stop loss
        pnl_pct = (current_price - entry_price) / entry_price
        if position_direction == SignalDirection.SELL:
            pnl_pct = -pnl_pct
        
        if pnl_pct < -self.stop_loss_pct:
            return True, "stop_loss"
        
        # Check current momentum
        indicators = self._calculate_momentum_indicators(market.id)
        if not indicators:
            return True, "insufficient_data"
        
        # Momentum reversal
        if position_direction == SignalDirection.BUY and indicators.momentum < -float(self.momentum_threshold):
            return True, "momentum_reversal"
        elif position_direction == SignalDirection.SELL and indicators.momentum > float(self.momentum_threshold):
            return True, "momentum_reversal"
        
        # RSI extremes
        if position_direction == SignalDirection.BUY and indicators.rsi > 80:
            return True, "overbought"
        elif position_direction == SignalDirection.SELL and indicators.rsi < 20:
            return True, "oversold"
        
        # Velocity divergence
        if position_direction == SignalDirection.BUY and indicators.velocity < -0.01:
            return True, "velocity_divergence"
        elif position_direction == SignalDirection.SELL and indicators.velocity > 0.01:
            return True, "velocity_divergence"
        
        return False, "hold"
    
    async def _place_order(self, signal: TradingSignal) -> Optional[Order]:
        """Place order based on momentum signal"""
        from ..models import OrderType, OrderStatus
        
        try:
            # Map signal direction to order side
            if signal.direction == SignalDirection.BUY:
                side = OrderSide.BUY
            else:
                side = OrderSide.SELL
            
            # Create order
            order = Order(
                id=f"momentum_{signal.market_id}_{datetime.now().timestamp()}",
                market_id=signal.market_id,
                outcome_id=signal.outcome,
                side=side,
                type=OrderType.LIMIT,
                size=float(signal.size),
                price=float(signal.target_price),
                status=OrderStatus.PENDING,
                created_at=datetime.now()
            )
            
            # In production, submit to CLOB API
            logger.info(
                f"Placing momentum {side.value} order for {signal.size} shares at {signal.target_price}"
            )
            
            # Update position tracking
            self.update_position(
                market_id=signal.market_id,
                outcome=signal.outcome,
                size=signal.size,
                price=signal.target_price
            )
            
            return order
            
        except Exception as e:
            logger.error(f"Error placing momentum order: {str(e)}")
            return None
    
    def get_momentum_metrics(self) -> Dict[str, Any]:
        """Get momentum-specific performance metrics"""
        total_markets = len(self.price_history)
        
        bullish_markets = sum(
            1 for cache in self.momentum_cache.values()
            if cache.is_bullish
        )
        
        bearish_markets = sum(
            1 for cache in self.momentum_cache.values()
            if cache.is_bearish
        )
        
        avg_momentum = np.mean([
            abs(cache.momentum) for cache in self.momentum_cache.values()
        ]) if self.momentum_cache else 0
        
        return {
            'markets_tracked': total_markets,
            'bullish_markets': bullish_markets,
            'bearish_markets': bearish_markets,
            'average_momentum': avg_momentum,
            'cache_size': len(self.momentum_cache)
        }