"""
Neural Momentum Trading Strategy
Adaptive momentum strategy with neural prediction and dynamic optimization
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import asyncio
import json
from abc import ABC, abstractmethod

from src.models.neural.momentum_predictor import MomentumPredictor
from src.risk_management.adaptive_risk_manager import AdaptiveRiskManager
from src.monitoring.performance_tracker import PerformanceTracker

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MarketRegime:
    """Market regime classification"""
    volatility: str  # 'low', 'medium', 'high'
    trend_strength: float  # 0.0 to 1.0
    liquidity: str  # 'low', 'medium', 'high'
    correlation_regime: str  # 'low', 'medium', 'high'
    sentiment_regime: str  # 'bearish', 'neutral', 'bullish'
    
@dataclass
class MomentumSignal:
    """Momentum trading signal"""
    symbol: str
    direction: str  # 'long', 'short', 'neutral'
    strength: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    neural_prediction: float
    technical_score: float
    sentiment_score: float
    entry_price: float
    stop_loss: float
    target_price: float
    position_size: float
    timestamp: datetime
    
@dataclass
class StrategyParameters:
    """Dynamic strategy parameters"""
    momentum_threshold: float = 0.6
    rsi_overbought: float = 70
    rsi_oversold: float = 30
    macd_threshold: float = 0.0
    volume_threshold: float = 1.5
    neural_confidence_min: float = 0.7
    sentiment_threshold: float = 0.5
    max_position_size: float = 0.05
    stop_loss_pct: float = 0.02
    target_multiplier: float = 3.0
    pyramid_levels: int = 3
    correlation_limit: float = 0.7

class NeuralMomentumStrategy:
    """
    Comprehensive momentum trading strategy with neural predictions and dynamic optimization
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.parameters = StrategyParameters(**config.get('parameters', {}))
        self.neural_predictor = MomentumPredictor(config.get('neural_config', {}))
        self.risk_manager = AdaptiveRiskManager(config.get('risk_config', {}))
        self.performance_tracker = PerformanceTracker(config.get('monitoring_config', {}))
        
        # Strategy state
        self.positions = {}
        self.active_signals = {}
        self.market_regime = None
        self.last_regime_update = None
        self.performance_metrics = {}
        
        # Market data caches
        self.price_cache = {}
        self.technical_cache = {}
        self.sentiment_cache = {}
        
        logger.info("Neural Momentum Strategy initialized")
    
    async def analyze_market_conditions(self, symbols: List[str]) -> MarketRegime:
        """Analyze current market conditions and determine regime"""
        try:
            # Get market data for analysis
            market_data = await self._get_market_data(symbols)
            volatility_data = await self._calculate_volatility_metrics(market_data)
            correlation_data = await self._calculate_correlation_metrics(market_data)
            sentiment_data = await self._get_sentiment_data(symbols)
            
            # Classify volatility regime
            avg_volatility = np.mean([v['volatility'] for v in volatility_data.values()])
            if avg_volatility < 0.15:
                volatility_regime = 'low'
            elif avg_volatility < 0.25:
                volatility_regime = 'medium'
            else:
                volatility_regime = 'high'
            
            # Calculate trend strength
            trend_scores = []
            for symbol in symbols:
                if symbol in market_data:
                    trend_score = self._calculate_trend_strength(market_data[symbol])
                    trend_scores.append(trend_score)
            trend_strength = np.mean(trend_scores) if trend_scores else 0.5
            
            # Classify liquidity regime
            avg_volume = np.mean([d.get('volume', 0) for d in market_data.values()])
            if avg_volume > 1000000:
                liquidity_regime = 'high'
            elif avg_volume > 500000:
                liquidity_regime = 'medium'
            else:
                liquidity_regime = 'low'
            
            # Classify correlation regime
            avg_correlation = np.mean(list(correlation_data.values())) if correlation_data else 0.5
            if avg_correlation > 0.7:
                correlation_regime = 'high'
            elif avg_correlation > 0.4:
                correlation_regime = 'medium'
            else:
                correlation_regime = 'low'
            
            # Classify sentiment regime
            avg_sentiment = np.mean([s.get('sentiment', 0) for s in sentiment_data.values()])
            if avg_sentiment > 0.3:
                sentiment_regime = 'bullish'
            elif avg_sentiment < -0.3:
                sentiment_regime = 'bearish'
            else:
                sentiment_regime = 'neutral'
            
            regime = MarketRegime(
                volatility=volatility_regime,
                trend_strength=trend_strength,
                liquidity=liquidity_regime,
                correlation_regime=correlation_regime,
                sentiment_regime=sentiment_regime
            )
            
            self.market_regime = regime
            self.last_regime_update = datetime.now()
            
            logger.info(f"Market regime updated: {regime}")
            return regime
            
        except Exception as e:
            logger.error(f"Error analyzing market conditions: {e}")
            return MarketRegime('medium', 0.5, 'medium', 'medium', 'neutral')
    
    async def optimize_parameters(self, regime: MarketRegime) -> StrategyParameters:
        """Dynamically optimize strategy parameters based on market regime"""
        try:
            params = StrategyParameters()
            
            # Volatility-based adjustments
            if regime.volatility == 'high':
                params.momentum_threshold = 0.7  # Higher threshold for high vol
                params.stop_loss_pct = 0.03  # Wider stops
                params.max_position_size = 0.03  # Smaller positions
                params.neural_confidence_min = 0.8  # Higher confidence required
            elif regime.volatility == 'low':
                params.momentum_threshold = 0.5  # Lower threshold for low vol
                params.stop_loss_pct = 0.015  # Tighter stops
                params.max_position_size = 0.07  # Larger positions
                params.neural_confidence_min = 0.6  # Lower confidence required
            
            # Trend strength adjustments
            if regime.trend_strength > 0.7:
                params.target_multiplier = 4.0  # Higher targets in strong trends
                params.pyramid_levels = 4  # More pyramiding
            elif regime.trend_strength < 0.3:
                params.target_multiplier = 2.0  # Lower targets in weak trends
                params.pyramid_levels = 2  # Less pyramiding
            
            # Correlation adjustments
            if regime.correlation_regime == 'high':
                params.correlation_limit = 0.5  # Stricter correlation limits
                params.max_position_size *= 0.8  # Reduce position sizes
            
            # Sentiment adjustments
            if regime.sentiment_regime == 'bullish':
                params.sentiment_threshold = 0.3  # Lower bar for long entries
            elif regime.sentiment_regime == 'bearish':
                params.sentiment_threshold = 0.7  # Higher bar for long entries
            
            self.parameters = params
            logger.info(f"Parameters optimized for regime: {regime}")
            return params
            
        except Exception as e:
            logger.error(f"Error optimizing parameters: {e}")
            return self.parameters
    
    async def generate_signals(self, symbols: List[str]) -> List[MomentumSignal]:
        """Generate momentum trading signals"""
        signals = []
        
        try:
            # Update market regime if needed
            if (not self.last_regime_update or 
                datetime.now() - self.last_regime_update > timedelta(hours=1)):
                await self.analyze_market_conditions(symbols)
                await self.optimize_parameters(self.market_regime)
            
            for symbol in symbols:
                signal = await self._generate_signal_for_symbol(symbol)
                if signal:
                    signals.append(signal)
            
            # Filter signals based on correlation constraints
            signals = self._filter_correlated_signals(signals)
            
            logger.info(f"Generated {len(signals)} momentum signals")
            return signals
            
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            return []
    
    async def _generate_signal_for_symbol(self, symbol: str) -> Optional[MomentumSignal]:
        """Generate signal for individual symbol"""
        try:
            # Get market data
            market_data = await self._get_symbol_data(symbol)
            if not market_data:
                return None
            
            # Calculate technical indicators
            technical_scores = await self._calculate_technical_scores(symbol, market_data)
            if technical_scores['momentum_score'] < self.parameters.momentum_threshold:
                return None
            
            # Get neural prediction
            neural_prediction = await self.neural_predictor.predict(symbol, market_data)
            if neural_prediction['confidence'] < self.parameters.neural_confidence_min:
                return None
            
            # Get sentiment score
            sentiment_data = await self._get_symbol_sentiment(symbol)
            sentiment_score = sentiment_data.get('sentiment', 0)
            
            # Determine signal direction and strength
            direction, strength, confidence = self._determine_signal_characteristics(
                technical_scores, neural_prediction, sentiment_score
            )
            
            if direction == 'neutral':
                return None
            
            # Calculate entry, stop, and target prices
            current_price = market_data['price']
            entry_price = current_price
            
            if direction == 'long':
                stop_loss = entry_price * (1 - self.parameters.stop_loss_pct)
                target_price = entry_price * (1 + self.parameters.stop_loss_pct * self.parameters.target_multiplier)
            else:
                stop_loss = entry_price * (1 + self.parameters.stop_loss_pct)
                target_price = entry_price * (1 - self.parameters.stop_loss_pct * self.parameters.target_multiplier)
            
            # Calculate position size
            position_size = await self.risk_manager.calculate_position_size(
                symbol, entry_price, stop_loss, confidence
            )
            
            signal = MomentumSignal(
                symbol=symbol,
                direction=direction,
                strength=strength,
                confidence=confidence,
                neural_prediction=neural_prediction['prediction'],
                technical_score=technical_scores['momentum_score'],
                sentiment_score=sentiment_score,
                entry_price=entry_price,
                stop_loss=stop_loss,
                target_price=target_price,
                position_size=min(position_size, self.parameters.max_position_size),
                timestamp=datetime.now()
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")
            return None
    
    def _determine_signal_characteristics(self, technical: Dict, neural: Dict, sentiment: float) -> Tuple[str, float, float]:
        """Determine signal direction, strength, and confidence"""
        try:
            # Combine scores
            combined_score = (
                technical['momentum_score'] * 0.4 +
                neural['prediction'] * 0.4 +
                (sentiment + 1) / 2 * 0.2  # Normalize sentiment to 0-1
            )
            
            # Determine direction
            if combined_score > 0.6:
                direction = 'long'
            elif combined_score < 0.4:
                direction = 'short'
            else:
                direction = 'neutral'
            
            # Calculate strength (0-1)
            strength = abs(combined_score - 0.5) * 2
            
            # Calculate confidence based on agreement between indicators
            scores = [technical['momentum_score'], neural['prediction'], (sentiment + 1) / 2]
            confidence = 1 - np.std(scores)  # Lower std = higher confidence
            
            return direction, strength, confidence
            
        except Exception as e:
            logger.error(f"Error determining signal characteristics: {e}")
            return 'neutral', 0.0, 0.0
    
    async def _calculate_technical_scores(self, symbol: str, data: Dict) -> Dict[str, float]:
        """Calculate technical momentum scores"""
        try:
            # Mock technical calculations - in practice, use real technical analysis
            price = data.get('price', 100)
            volume = data.get('volume', 1000000)
            
            # RSI momentum component
            rsi = data.get('rsi', 50)
            rsi_score = 0.0
            if rsi > self.parameters.rsi_overbought:
                rsi_score = -0.5
            elif rsi < self.parameters.rsi_oversold:
                rsi_score = 0.8
            elif 45 < rsi < 55:
                rsi_score = 0.6
            
            # MACD momentum component
            macd = data.get('macd', 0)
            macd_score = min(abs(macd) / 2.0, 1.0) if macd != 0 else 0.0
            
            # Volume momentum component
            avg_volume = data.get('avg_volume', volume)
            volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0
            volume_score = min(volume_ratio / self.parameters.volume_threshold, 1.0)
            
            # Price velocity component
            price_change = data.get('price_change_pct', 0) / 100
            velocity_score = min(abs(price_change) * 10, 1.0)
            
            # Combined momentum score
            momentum_score = (rsi_score * 0.25 + macd_score * 0.25 + 
                            volume_score * 0.25 + velocity_score * 0.25)
            
            return {
                'momentum_score': momentum_score,
                'rsi_score': rsi_score,
                'macd_score': macd_score,
                'volume_score': volume_score,
                'velocity_score': velocity_score
            }
            
        except Exception as e:
            logger.error(f"Error calculating technical scores for {symbol}: {e}")
            return {'momentum_score': 0.0}
    
    def _filter_correlated_signals(self, signals: List[MomentumSignal]) -> List[MomentumSignal]:
        """Filter signals to avoid highly correlated positions"""
        try:
            if len(signals) <= 1:
                return signals
            
            filtered_signals = []
            correlation_matrix = self._get_correlation_matrix([s.symbol for s in signals])
            
            # Sort signals by confidence (highest first)
            sorted_signals = sorted(signals, key=lambda x: x.confidence, reverse=True)
            
            for signal in sorted_signals:
                # Check correlation with already selected signals
                correlated = False
                for selected in filtered_signals:
                    correlation = correlation_matrix.get((signal.symbol, selected.symbol), 0)
                    if abs(correlation) > self.parameters.correlation_limit:
                        correlated = True
                        break
                
                if not correlated:
                    filtered_signals.append(signal)
            
            return filtered_signals
            
        except Exception as e:
            logger.error(f"Error filtering correlated signals: {e}")
            return signals
    
    async def execute_trades(self, signals: List[MomentumSignal]) -> List[Dict]:
        """Execute trades based on signals"""
        executed_trades = []
        
        try:
            for signal in signals:
                # Check if we already have a position
                if signal.symbol in self.positions:
                    continue
                
                # Execute the trade
                trade_result = await self._execute_trade(signal)
                if trade_result:
                    executed_trades.append(trade_result)
                    self.positions[signal.symbol] = signal
                    self.active_signals[signal.symbol] = signal
                    
                    # Track performance
                    await self.performance_tracker.record_trade(trade_result)
            
            logger.info(f"Executed {len(executed_trades)} momentum trades")
            return executed_trades
            
        except Exception as e:
            logger.error(f"Error executing trades: {e}")
            return []
    
    async def manage_positions(self) -> List[Dict]:
        """Manage existing positions with dynamic stops and pyramiding"""
        management_actions = []
        
        try:
            for symbol, position in list(self.positions.items()):
                # Get current market data
                current_data = await self._get_symbol_data(symbol)
                if not current_data:
                    continue
                
                current_price = current_data['price']
                
                # Check for exit conditions
                exit_action = await self._check_exit_conditions(symbol, position, current_price)
                if exit_action:
                    management_actions.append(exit_action)
                    del self.positions[symbol]
                    del self.active_signals[symbol]
                    continue
                
                # Check for pyramiding opportunities
                pyramid_action = await self._check_pyramid_opportunity(symbol, position, current_price)
                if pyramid_action:
                    management_actions.append(pyramid_action)
                
                # Update trailing stops
                stop_update = await self._update_trailing_stop(symbol, position, current_price)
                if stop_update:
                    management_actions.append(stop_update)
                    self.positions[symbol].stop_loss = stop_update['new_stop']
            
            logger.info(f"Performed {len(management_actions)} position management actions")
            return management_actions
            
        except Exception as e:
            logger.error(f"Error managing positions: {e}")
            return []
    
    async def _check_exit_conditions(self, symbol: str, position: MomentumSignal, current_price: float) -> Optional[Dict]:
        """Check if position should be exited"""
        try:
            # Stop loss hit
            if position.direction == 'long' and current_price <= position.stop_loss:
                return {
                    'action': 'exit',
                    'symbol': symbol,
                    'reason': 'stop_loss',
                    'price': current_price,
                    'quantity': position.position_size
                }
            elif position.direction == 'short' and current_price >= position.stop_loss:
                return {
                    'action': 'exit',
                    'symbol': symbol,
                    'reason': 'stop_loss',
                    'price': current_price,
                    'quantity': position.position_size
                }
            
            # Target hit
            if position.direction == 'long' and current_price >= position.target_price:
                return {
                    'action': 'exit',
                    'symbol': symbol,
                    'reason': 'target_reached',
                    'price': current_price,
                    'quantity': position.position_size * 0.5  # Take partial profits
                }
            elif position.direction == 'short' and current_price <= position.target_price:
                return {
                    'action': 'exit',
                    'symbol': symbol,
                    'reason': 'target_reached',
                    'price': current_price,
                    'quantity': position.position_size * 0.5
                }
            
            # Time-based exit (10 days max)
            if datetime.now() - position.timestamp > timedelta(days=10):
                return {
                    'action': 'exit',
                    'symbol': symbol,
                    'reason': 'time_limit',
                    'price': current_price,
                    'quantity': position.position_size
                }
            
            # Momentum decay check
            neural_prediction = await self.neural_predictor.predict(symbol, {'price': current_price})
            if neural_prediction['confidence'] < 0.5:
                return {
                    'action': 'exit',
                    'symbol': symbol,
                    'reason': 'momentum_decay',
                    'price': current_price,
                    'quantity': position.position_size
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking exit conditions for {symbol}: {e}")
            return None
    
    async def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        try:
            return await self.performance_tracker.generate_report()
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            return {}
    
    # Helper methods for data access (mock implementations)
    async def _get_market_data(self, symbols: List[str]) -> Dict:
        """Get market data for symbols"""
        # Mock implementation - replace with real market data API
        return {symbol: {
            'price': 100 + hash(symbol) % 50,
            'volume': 1000000,
            'volatility': 0.2,
            'rsi': 50,
            'macd': 0.5
        } for symbol in symbols}
    
    async def _get_symbol_data(self, symbol: str) -> Dict:
        """Get data for single symbol"""
        market_data = await self._get_market_data([symbol])
        return market_data.get(symbol, {})
    
    async def _get_symbol_sentiment(self, symbol: str) -> Dict:
        """Get sentiment data for symbol"""
        # Mock implementation
        return {'sentiment': 0.3, 'confidence': 0.8}
    
    async def _get_sentiment_data(self, symbols: List[str]) -> Dict:
        """Get sentiment data for symbols"""
        return {symbol: await self._get_symbol_sentiment(symbol) for symbol in symbols}
    
    async def _calculate_volatility_metrics(self, market_data: Dict) -> Dict:
        """Calculate volatility metrics"""
        return {symbol: {'volatility': data.get('volatility', 0.2)} 
                for symbol, data in market_data.items()}
    
    async def _calculate_correlation_metrics(self, market_data: Dict) -> Dict:
        """Calculate correlation metrics"""
        symbols = list(market_data.keys())
        correlations = {}
        for i, symbol1 in enumerate(symbols):
            for symbol2 in symbols[i+1:]:
                # Mock correlation calculation
                correlations[(symbol1, symbol2)] = 0.3
        return correlations
    
    def _calculate_trend_strength(self, data: Dict) -> float:
        """Calculate trend strength score"""
        # Mock implementation
        return 0.6
    
    def _get_correlation_matrix(self, symbols: List[str]) -> Dict:
        """Get correlation matrix for symbols"""
        correlations = {}
        for i, symbol1 in enumerate(symbols):
            for symbol2 in symbols:
                if symbol1 != symbol2:
                    # Mock correlation
                    correlations[(symbol1, symbol2)] = 0.3
        return correlations
    
    async def _execute_trade(self, signal: MomentumSignal) -> Optional[Dict]:
        """Execute individual trade"""
        # Mock trade execution
        return {
            'symbol': signal.symbol,
            'direction': signal.direction,
            'quantity': signal.position_size,
            'price': signal.entry_price,
            'timestamp': datetime.now(),
            'trade_id': f"TRADE_{hash(signal.symbol)}_{int(datetime.now().timestamp())}"
        }
    
    async def _check_pyramid_opportunity(self, symbol: str, position: MomentumSignal, current_price: float) -> Optional[Dict]:
        """Check for pyramiding opportunities"""
        # Mock implementation
        return None
    
    async def _update_trailing_stop(self, symbol: str, position: MomentumSignal, current_price: float) -> Optional[Dict]:
        """Update trailing stop loss"""
        # Mock implementation
        return None