"""
Ensemble Trading Strategy

Combines signals from multiple trading strategies to generate
more robust and reliable trading decisions.
"""

import asyncio
import logging
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any, Set, Tuple
from collections import defaultdict

from ..models import Market, MarketStatus, Order, OrderSide
from .base import (
    PolymarketStrategy, StrategyConfig, TradingSignal, SignalStrength,
    SignalDirection, StrategyError
)
from .news_sentiment import NewsSentimentStrategy
from .market_maker import MarketMakerStrategy
from .arbitrage import ArbitrageStrategy
from .momentum import MomentumStrategy

logger = logging.getLogger(__name__)


@dataclass
class EnsembleSignal(TradingSignal):
    """Extended signal combining multiple strategies"""
    component_signals: List[TradingSignal] = field(default_factory=list)
    strategy_votes: Dict[str, str] = field(default_factory=dict)  # strategy_name -> direction
    consensus_score: float = 0.0
    disagreement_level: float = 0.0
    dominant_strategy: str = ""
    
    def __post_init__(self):
        """Calculate ensemble metrics after initialization"""
        super().__post_init__()
        
        if self.component_signals:
            # Calculate consensus
            buy_votes = sum(1 for s in self.component_signals if s.direction == SignalDirection.BUY)
            sell_votes = sum(1 for s in self.component_signals if s.direction == SignalDirection.SELL)
            total_votes = len(self.component_signals)
            
            if total_votes > 0:
                self.consensus_score = max(buy_votes, sell_votes) / total_votes
                self.disagreement_level = 1 - self.consensus_score
            
            # Find dominant strategy (highest confidence)
            if self.component_signals:
                dominant = max(self.component_signals, key=lambda s: s.confidence)
                self.dominant_strategy = dominant.metadata.get('strategy', 'unknown')
    
    @property
    def is_unanimous(self) -> bool:
        """Check if all strategies agree on direction"""
        return self.consensus_score == 1.0
    
    @property
    def is_strong_consensus(self) -> bool:
        """Check if there's strong agreement (>75%)"""
        return self.consensus_score >= 0.75
    
    def get_strategy_weights(self) -> Dict[str, float]:
        """Get weight contribution of each strategy"""
        weights = {}
        total_confidence = sum(s.confidence for s in self.component_signals)
        
        if total_confidence > 0:
            for signal in self.component_signals:
                strategy = signal.metadata.get('strategy', 'unknown')
                weights[strategy] = signal.confidence / total_confidence
        
        return weights


class EnsembleStrategy(PolymarketStrategy):
    """
    Ensemble strategy that combines multiple trading strategies.
    
    Combines signals from:
    - News Sentiment
    - Market Making
    - Arbitrage
    - Momentum
    
    Uses voting and weighted averaging to generate robust signals.
    """
    
    def __init__(
        self,
        client,
        config: Optional[StrategyConfig] = None,
        min_strategies: int = 2,
        min_consensus: float = 0.6,
        use_confidence_weighting: bool = True,
        adaptive_weights: bool = True,
        include_market_maker: bool = False,  # MM signals are different
        weight_decay: float = 0.95,
        performance_window: int = 100
    ):
        """
        Initialize ensemble strategy.
        
        Args:
            client: Polymarket API client
            config: Strategy configuration
            min_strategies: Minimum strategies that must agree
            min_consensus: Minimum consensus score (0-1)
            use_confidence_weighting: Weight by confidence scores
            adaptive_weights: Adapt weights based on performance
            include_market_maker: Include market maker signals
            weight_decay: Decay factor for performance tracking
            performance_window: Window for performance calculation
        """
        super().__init__(client, config, "EnsembleStrategy")
        
        self.min_strategies = min_strategies
        self.min_consensus = min_consensus
        self.use_confidence_weighting = use_confidence_weighting
        self.adaptive_weights = adaptive_weights
        self.include_market_maker = include_market_maker
        self.weight_decay = weight_decay
        self.performance_window = performance_window
        
        # Initialize component strategies
        self.strategies = {
            'news_sentiment': NewsSentimentStrategy(client, config),
            'momentum': MomentumStrategy(client, config),
            'arbitrage': ArbitrageStrategy(client, config)
        }
        
        if include_market_maker:
            self.strategies['market_maker'] = MarketMakerStrategy(client, config)
        
        # Performance tracking for adaptive weights
        self.strategy_performance: Dict[str, List[float]] = defaultdict(list)
        self.strategy_weights: Dict[str, float] = {
            name: 1.0 / len(self.strategies) for name in self.strategies
        }
        
        # Signal history for analysis
        self.ensemble_history: List[EnsembleSignal] = []
        
        logger.info(
            f"Initialized {self.name} with {len(self.strategies)} strategies, "
            f"min consensus {self.min_consensus}"
        )
    
    async def should_trade_market(self, market: Market) -> bool:
        """
        Determine if market is suitable for ensemble trading.
        
        Market must be suitable for at least min_strategies.
        """
        suitable_count = 0
        
        for strategy in self.strategies.values():
            if await strategy.should_trade_market(market):
                suitable_count += 1
        
        return suitable_count >= self.min_strategies
    
    async def analyze_market(self, market: Market) -> Optional[TradingSignal]:
        """
        Analyze market using ensemble of strategies.
        
        Args:
            market: Market to analyze
            
        Returns:
            Ensemble signal if consensus reached
        """
        try:
            # Collect signals from all strategies
            component_signals = await self._collect_component_signals(market)
            
            if len(component_signals) < self.min_strategies:
                logger.debug(
                    f"Not enough strategies generated signals for {market.id} "
                    f"({len(component_signals)}/{self.min_strategies})"
                )
                return None
            
            # Calculate ensemble metrics
            ensemble_signal = self._create_ensemble_signal(market, component_signals)
            
            if not ensemble_signal:
                return None
            
            # Check consensus threshold
            if ensemble_signal.consensus_score < self.min_consensus:
                logger.debug(
                    f"Consensus too low for {market.id}: "
                    f"{ensemble_signal.consensus_score:.2f} < {self.min_consensus}"
                )
                return None
            
            # Update performance tracking
            self._update_signal_history(ensemble_signal)
            
            return ensemble_signal
            
        except Exception as e:
            logger.error(f"Error in ensemble analysis for {market.id}: {str(e)}")
            return None
    
    async def _collect_component_signals(
        self,
        market: Market
    ) -> List[TradingSignal]:
        """Collect signals from all component strategies"""
        signals = []
        
        # Run strategies in parallel
        tasks = []
        strategy_names = []
        
        for name, strategy in self.strategies.items():
            tasks.append(strategy.analyze_market(market))
            strategy_names.append(name)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for name, result in zip(strategy_names, results):
            if isinstance(result, Exception):
                logger.warning(f"Strategy {name} failed: {str(result)}")
                continue
            
            if result is not None:
                # Add strategy name to metadata
                if 'strategy' not in result.metadata:
                    result.metadata['strategy'] = name
                signals.append(result)
        
        return signals
    
    def _create_ensemble_signal(
        self,
        market: Market,
        component_signals: List[TradingSignal]
    ) -> Optional[EnsembleSignal]:
        """Create ensemble signal from component signals"""
        if not component_signals:
            return None
        
        # Count votes by direction
        votes = defaultdict(list)
        strategy_votes = {}
        
        for signal in component_signals:
            votes[signal.direction].append(signal)
            strategy_name = signal.metadata.get('strategy', 'unknown')
            strategy_votes[strategy_name] = signal.direction.value
        
        # Determine consensus direction
        buy_signals = votes.get(SignalDirection.BUY, [])
        sell_signals = votes.get(SignalDirection.SELL, [])
        
        if len(buy_signals) > len(sell_signals):
            consensus_direction = SignalDirection.BUY
            consensus_signals = buy_signals
            outcome = "Yes"
        elif len(sell_signals) > len(buy_signals):
            consensus_direction = SignalDirection.SELL
            consensus_signals = sell_signals
            outcome = "Yes"
        else:
            # Tie - use confidence weighting
            buy_confidence = sum(s.confidence for s in buy_signals)
            sell_confidence = sum(s.confidence for s in sell_signals)
            
            if buy_confidence > sell_confidence:
                consensus_direction = SignalDirection.BUY
                consensus_signals = buy_signals
                outcome = "Yes"
            else:
                consensus_direction = SignalDirection.SELL
                consensus_signals = sell_signals
                outcome = "Yes"
        
        # Calculate ensemble metrics
        if self.use_confidence_weighting:
            # Weighted averages
            weights = self._get_signal_weights(consensus_signals)
            target_price = self._weighted_average(
                consensus_signals,
                lambda s: float(s.target_price),
                weights
            )
            confidence = self._weighted_average(
                consensus_signals,
                lambda s: s.confidence,
                weights
            )
            size = self._weighted_average(
                consensus_signals,
                lambda s: float(s.size),
                weights
            )
        else:
            # Simple averages
            target_price = np.mean([float(s.target_price) for s in consensus_signals])
            confidence = np.mean([s.confidence for s in consensus_signals])
            size = np.mean([float(s.size) for s in consensus_signals])
        
        # Determine signal strength
        strength = self._calculate_ensemble_strength(consensus_signals)
        
        # Create ensemble signal
        signal = EnsembleSignal(
            market_id=market.id,
            outcome=outcome,
            direction=consensus_direction,
            strength=strength,
            target_price=Decimal(str(target_price)),
            size=Decimal(str(size)),
            confidence=confidence,
            reasoning=self._generate_ensemble_reasoning(
                consensus_signals,
                strategy_votes
            ),
            component_signals=component_signals,
            strategy_votes=strategy_votes,
            metadata={
                'strategy': self.name,
                'component_count': len(component_signals),
                'consensus_direction': consensus_direction.value
            }
        )
        
        return signal
    
    def _get_signal_weights(
        self,
        signals: List[TradingSignal]
    ) -> List[float]:
        """Calculate weights for signals based on confidence and performance"""
        weights = []
        
        for signal in signals:
            strategy_name = signal.metadata.get('strategy', 'unknown')
            
            # Base weight from confidence
            base_weight = signal.confidence
            
            # Apply adaptive strategy weight if enabled
            if self.adaptive_weights and strategy_name in self.strategy_weights:
                base_weight *= self.strategy_weights[strategy_name]
            
            weights.append(base_weight)
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(signals) for _ in signals]
        
        return weights
    
    def _weighted_average(
        self,
        signals: List[TradingSignal],
        value_func: callable,
        weights: List[float]
    ) -> float:
        """Calculate weighted average of signal values"""
        values = [value_func(s) for s in signals]
        return sum(v * w for v, w in zip(values, weights))
    
    def _calculate_ensemble_strength(
        self,
        consensus_signals: List[TradingSignal]
    ) -> SignalStrength:
        """Calculate overall signal strength for ensemble"""
        # Average the signal strengths
        avg_strength_value = np.mean([s.strength.value for s in consensus_signals])
        
        # Map to nearest strength level
        if avg_strength_value >= SignalStrength.VERY_STRONG.value - 0.5:
            return SignalStrength.VERY_STRONG
        elif avg_strength_value >= SignalStrength.STRONG.value - 0.5:
            return SignalStrength.STRONG
        elif avg_strength_value >= SignalStrength.MODERATE.value - 0.5:
            return SignalStrength.MODERATE
        elif avg_strength_value >= SignalStrength.WEAK.value - 0.5:
            return SignalStrength.WEAK
        else:
            return SignalStrength.VERY_WEAK
    
    def _generate_ensemble_reasoning(
        self,
        consensus_signals: List[TradingSignal],
        strategy_votes: Dict[str, str]
    ) -> str:
        """Generate reasoning for ensemble signal"""
        total_strategies = len(strategy_votes)
        consensus_count = len(consensus_signals)
        
        reasoning = (
            f"Ensemble consensus: {consensus_count}/{total_strategies} strategies agree. "
        )
        
        # Add individual strategy summaries
        strategy_summaries = []
        for signal in consensus_signals:
            strategy_name = signal.metadata.get('strategy', 'unknown')
            summary = f"{strategy_name} ({signal.confidence:.0%} conf)"
            strategy_summaries.append(summary)
        
        reasoning += f"Contributing: {', '.join(strategy_summaries)}"
        
        return reasoning
    
    def _update_signal_history(self, signal: EnsembleSignal):
        """Update signal history and performance tracking"""
        self.ensemble_history.append(signal)
        
        # Keep only recent history
        if len(self.ensemble_history) > self.performance_window:
            self.ensemble_history = self.ensemble_history[-self.performance_window:]
    
    def update_strategy_performance(
        self,
        strategy_name: str,
        profit_loss: float
    ):
        """
        Update performance tracking for a strategy.
        
        Args:
            strategy_name: Name of the strategy
            profit_loss: Realized P&L from the strategy's signal
        """
        if strategy_name in self.strategy_performance:
            self.strategy_performance[strategy_name].append(profit_loss)
            
            # Apply decay to old performance
            if len(self.strategy_performance[strategy_name]) > self.performance_window:
                self.strategy_performance[strategy_name] = \
                    self.strategy_performance[strategy_name][-self.performance_window:]
            
            # Update adaptive weights
            if self.adaptive_weights:
                self._update_adaptive_weights()
    
    def _update_adaptive_weights(self):
        """Update strategy weights based on recent performance"""
        # Calculate performance scores
        performance_scores = {}
        
        for strategy_name, pnl_history in self.strategy_performance.items():
            if pnl_history:
                # Calculate weighted average with decay
                weighted_pnl = 0
                weight_sum = 0
                
                for i, pnl in enumerate(reversed(pnl_history)):
                    weight = self.weight_decay ** i
                    weighted_pnl += pnl * weight
                    weight_sum += weight
                
                avg_pnl = weighted_pnl / weight_sum if weight_sum > 0 else 0
                
                # Convert to score (sigmoid to bound between 0 and 1)
                score = 1 / (1 + np.exp(-avg_pnl))
                performance_scores[strategy_name] = score
        
        # Update weights
        if performance_scores:
            # Normalize scores
            total_score = sum(performance_scores.values())
            if total_score > 0:
                for strategy_name in self.strategy_weights:
                    if strategy_name in performance_scores:
                        self.strategy_weights[strategy_name] = \
                            performance_scores[strategy_name] / total_score
                    else:
                        # Default weight for strategies without history
                        self.strategy_weights[strategy_name] = \
                            1.0 / len(self.strategy_weights)
    
    async def _place_order(self, signal: TradingSignal) -> Optional[Order]:
        """Place order based on ensemble signal"""
        from ..models import OrderType, OrderStatus
        
        if not isinstance(signal, EnsembleSignal):
            logger.error("Invalid signal type for ensemble")
            return None
        
        try:
            # Use limit orders for ensemble
            if signal.direction == SignalDirection.BUY:
                side = OrderSide.BUY
            else:
                side = OrderSide.SELL
            
            # Create order
            order = Order(
                id=f"ensemble_{signal.market_id}_{datetime.now().timestamp()}",
                market_id=signal.market_id,
                outcome_id=signal.outcome,
                side=side,
                type=OrderType.LIMIT,
                size=float(signal.size),
                price=float(signal.target_price),
                status=OrderStatus.PENDING,
                created_at=datetime.now()
            )
            
            logger.info(
                f"Placing ensemble {side.value} order for {signal.size} shares at {signal.target_price} "
                f"(consensus: {signal.consensus_score:.0%})"
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
            logger.error(f"Error placing ensemble order: {str(e)}")
            return None
    
    def get_ensemble_metrics(self) -> Dict[str, Any]:
        """Get ensemble-specific performance metrics"""
        if not self.ensemble_history:
            return {
                'total_signals': 0,
                'average_consensus': 0.0,
                'unanimous_signals': 0,
                'strategy_weights': self.strategy_weights
            }
        
        total_signals = len(self.ensemble_history)
        unanimous = sum(1 for s in self.ensemble_history if s.is_unanimous)
        avg_consensus = np.mean([s.consensus_score for s in self.ensemble_history])
        avg_disagreement = np.mean([s.disagreement_level for s in self.ensemble_history])
        
        # Strategy participation
        strategy_participation = defaultdict(int)
        for signal in self.ensemble_history:
            for strategy_name in signal.strategy_votes:
                strategy_participation[strategy_name] += 1
        
        return {
            'total_signals': total_signals,
            'average_consensus': avg_consensus,
            'average_disagreement': avg_disagreement,
            'unanimous_signals': unanimous,
            'unanimous_rate': unanimous / total_signals if total_signals > 0 else 0,
            'strategy_weights': self.strategy_weights,
            'strategy_participation': dict(strategy_participation),
            'dominant_strategies': self._get_dominant_strategies()
        }
    
    def _get_dominant_strategies(self) -> Dict[str, int]:
        """Get count of times each strategy was dominant"""
        dominant_count = defaultdict(int)
        
        for signal in self.ensemble_history:
            if signal.dominant_strategy:
                dominant_count[signal.dominant_strategy] += 1
        
        return dict(dominant_count)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        base_summary = super().get_performance_summary()
        
        # Add ensemble-specific metrics
        ensemble_metrics = self.get_ensemble_metrics()
        base_summary['ensemble_metrics'] = ensemble_metrics
        
        # Add component strategy performance
        component_performance = {}
        for name, strategy in self.strategies.items():
            component_performance[name] = strategy.get_performance_summary()
        
        base_summary['component_strategies'] = component_performance
        
        return base_summary