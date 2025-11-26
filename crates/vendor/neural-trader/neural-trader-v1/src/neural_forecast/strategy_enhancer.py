"""
Strategy Enhancer for Neural-Enhanced Trading Strategies.

This module integrates NHITS forecasting with existing trading strategies to provide:
- Neural-enhanced signal generation
- Multi-horizon forecast integration
- Confidence-based position sizing
- Risk-adjusted decision making
- Strategy performance optimization
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import json

from .nhits_forecaster import NHITSForecaster
from .neural_model_manager import NeuralModelManager


@dataclass
class NeuralSignal:
    """Neural forecasting signal structure."""
    timestamp: str
    symbol: str
    forecast_horizon: int
    point_forecast: List[float]
    confidence_intervals: Dict[str, Dict[str, List[float]]]
    confidence_score: float
    trend_direction: str  # 'bullish', 'bearish', 'neutral'
    volatility_forecast: float
    signal_strength: float  # 0-1 scale


@dataclass
class EnhancedTradingSignal:
    """Enhanced trading signal with neural forecasting."""
    symbol: str
    action: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float
    position_size: float
    entry_price: float
    target_price: float
    stop_loss: float
    holding_period: int  # days
    neural_signal: NeuralSignal
    traditional_signal: Dict[str, Any]
    combined_score: float
    risk_metrics: Dict[str, float]
    reasoning: str


class StrategyEnhancer:
    """
    Neural strategy enhancer that combines traditional trading strategies
    with NHITS forecasting for improved decision making.
    
    Features:
    - Multi-strategy neural enhancement
    - Adaptive forecast integration
    - Confidence-based position sizing
    - Risk-adjusted signal generation
    - Performance tracking and optimization
    """
    
    def __init__(
        self,
        model_manager: Optional[NeuralModelManager] = None,
        forecast_horizons: List[int] = None,
        confidence_threshold: float = 0.6,
        neural_weight: float = 0.4,
        traditional_weight: float = 0.6,
        enable_adaptive_weighting: bool = True,
        risk_adjustment_factor: float = 0.8
    ):
        """
        Initialize Strategy Enhancer.
        
        Args:
            model_manager: Neural model manager instance
            forecast_horizons: List of forecast horizons to use
            confidence_threshold: Minimum confidence for neural signals
            neural_weight: Weight for neural signals (0-1)
            traditional_weight: Weight for traditional signals (0-1)
            enable_adaptive_weighting: Enable adaptive weight adjustment
            risk_adjustment_factor: Risk adjustment multiplier
        """
        self.model_manager = model_manager or NeuralModelManager()
        self.forecast_horizons = forecast_horizons or [1, 3, 5, 10]  # 1-10 day horizons
        self.confidence_threshold = confidence_threshold
        self.neural_weight = neural_weight
        self.traditional_weight = traditional_weight
        self.enable_adaptive_weighting = enable_adaptive_weighting
        self.risk_adjustment_factor = risk_adjustment_factor
        
        # Strategy performance tracking
        self.performance_history: Dict[str, List[Dict]] = {}
        self.adaptive_weights: Dict[str, Dict[str, float]] = {}
        
        # Signal cache
        self.signal_cache: Dict[str, NeuralSignal] = {}
        self.cache_ttl = timedelta(minutes=15)  # 15-minute cache TTL
        
        # Logging setup
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
        
        self.logger.info("Strategy Enhancer initialized")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    async def enhance_momentum_strategy(
        self,
        symbol: str,
        market_data: Dict[str, Any],
        traditional_signal: Dict[str, Any],
        strategy_name: str = "momentum"
    ) -> EnhancedTradingSignal:
        """
        Enhance momentum trading strategy with neural forecasting.
        
        Args:
            symbol: Trading symbol
            market_data: Historical market data
            traditional_signal: Traditional momentum signal
            strategy_name: Strategy identifier
            
        Returns:
            Enhanced trading signal
        """
        try:
            # Generate neural signal
            neural_signal = await self._generate_neural_signal(symbol, market_data)
            
            # Combine signals with strategy-specific logic
            combined_signal = await self._combine_momentum_signals(
                traditional_signal, neural_signal, strategy_name
            )
            
            # Apply risk adjustments
            enhanced_signal = await self._apply_risk_adjustments(
                combined_signal, neural_signal, market_data
            )
            
            # Track performance
            await self._track_signal_performance(strategy_name, enhanced_signal)
            
            return enhanced_signal
            
        except Exception as e:
            self.logger.error(f"Momentum strategy enhancement failed: {str(e)}")
            return await self._fallback_signal(symbol, traditional_signal, "momentum_fallback")
    
    async def enhance_mean_reversion_strategy(
        self,
        symbol: str,
        market_data: Dict[str, Any],
        traditional_signal: Dict[str, Any],
        strategy_name: str = "mean_reversion"
    ) -> EnhancedTradingSignal:
        """
        Enhance mean reversion strategy with neural forecasting.
        
        Args:
            symbol: Trading symbol
            market_data: Historical market data
            traditional_signal: Traditional mean reversion signal
            strategy_name: Strategy identifier
            
        Returns:
            Enhanced trading signal
        """
        try:
            # Generate neural signal
            neural_signal = await self._generate_neural_signal(symbol, market_data)
            
            # Combine signals with mean reversion logic
            combined_signal = await self._combine_mean_reversion_signals(
                traditional_signal, neural_signal, strategy_name
            )
            
            # Apply risk adjustments
            enhanced_signal = await self._apply_risk_adjustments(
                combined_signal, neural_signal, market_data
            )
            
            # Track performance
            await self._track_signal_performance(strategy_name, enhanced_signal)
            
            return enhanced_signal
            
        except Exception as e:
            self.logger.error(f"Mean reversion strategy enhancement failed: {str(e)}")
            return await self._fallback_signal(symbol, traditional_signal, "mean_reversion_fallback")
    
    async def enhance_swing_strategy(
        self,
        symbol: str,
        market_data: Dict[str, Any],
        traditional_signal: Dict[str, Any],
        strategy_name: str = "swing"
    ) -> EnhancedTradingSignal:
        """
        Enhance swing trading strategy with neural forecasting.
        
        Args:
            symbol: Trading symbol
            market_data: Historical market data
            traditional_signal: Traditional swing signal
            strategy_name: Strategy identifier
            
        Returns:
            Enhanced trading signal
        """
        try:
            # Generate neural signal with longer horizons for swing trading
            neural_signal = await self._generate_neural_signal(
                symbol, market_data, horizons=[3, 5, 10, 15]
            )
            
            # Combine signals with swing trading logic
            combined_signal = await self._combine_swing_signals(
                traditional_signal, neural_signal, strategy_name
            )
            
            # Apply risk adjustments
            enhanced_signal = await self._apply_risk_adjustments(
                combined_signal, neural_signal, market_data
            )
            
            # Track performance
            await self._track_signal_performance(strategy_name, enhanced_signal)
            
            return enhanced_signal
            
        except Exception as e:
            self.logger.error(f"Swing strategy enhancement failed: {str(e)}")
            return await self._fallback_signal(symbol, traditional_signal, "swing_fallback")
    
    async def _generate_neural_signal(
        self,
        symbol: str,
        market_data: Dict[str, Any],
        horizons: Optional[List[int]] = None
    ) -> NeuralSignal:
        """
        Generate neural forecasting signal for a symbol.
        
        Args:
            symbol: Trading symbol
            market_data: Historical market data
            horizons: Forecast horizons to use
            
        Returns:
            Neural forecasting signal
        """
        # Check cache first
        cache_key = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M')}"
        if cache_key in self.signal_cache:
            cached_signal = self.signal_cache[cache_key]
            # Check if cache is still valid
            signal_time = datetime.fromisoformat(cached_signal.timestamp)
            if datetime.now() - signal_time < self.cache_ttl:
                return cached_signal
        
        try:
            # Prepare data for forecasting
            forecast_data = self._prepare_forecast_data(market_data)
            
            # Generate forecasts for multiple horizons
            horizons = horizons or self.forecast_horizons
            forecast_results = {}
            
            for horizon in horizons:
                # Update model horizon if needed
                if self.model_manager.forecaster:
                    self.model_manager.forecaster.horizon = horizon
                
                # Generate forecast
                result = await self.model_manager.predict(
                    data=forecast_data,
                    return_intervals=True
                )
                
                if result['success']:
                    forecast_results[horizon] = result
            
            # Process multi-horizon results
            neural_signal = self._process_multi_horizon_forecasts(
                symbol, forecast_results, market_data
            )
            
            # Cache the signal
            self.signal_cache[cache_key] = neural_signal
            
            return neural_signal
            
        except Exception as e:
            self.logger.error(f"Neural signal generation failed: {str(e)}")
            return self._create_neutral_signal(symbol)
    
    def _prepare_forecast_data(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare market data for neural forecasting.
        
        Args:
            market_data: Raw market data
            
        Returns:
            Formatted data for forecasting
        """
        try:
            # Extract price data
            if 'prices' in market_data:
                prices = market_data['prices']
            elif 'close' in market_data:
                prices = market_data['close']
            else:
                raise ValueError("No price data found in market_data")
            
            # Extract timestamps
            if 'timestamps' in market_data:
                timestamps = market_data['timestamps']
            elif 'dates' in market_data:
                timestamps = market_data['dates']
            else:
                # Generate synthetic timestamps
                timestamps = pd.date_range(
                    end=datetime.now(),
                    periods=len(prices),
                    freq='H'
                ).tolist()
            
            # Create forecast-ready format
            return {
                'ds': timestamps,
                'y': prices,
                'unique_id': 'main_series'
            }
            
        except Exception as e:
            self.logger.error(f"Data preparation failed: {str(e)}")
            raise
    
    def _process_multi_horizon_forecasts(
        self,
        symbol: str,
        forecast_results: Dict[int, Dict],
        market_data: Dict[str, Any]
    ) -> NeuralSignal:
        """
        Process multi-horizon forecast results into a single signal.
        
        Args:
            symbol: Trading symbol
            forecast_results: Results from multiple forecast horizons
            market_data: Original market data
            
        Returns:
            Processed neural signal
        """
        try:
            # Aggregate forecasts across horizons
            all_forecasts = []
            all_intervals = {}
            confidence_scores = []
            
            for horizon, result in forecast_results.items():
                if not result.get('success', False):
                    continue
                
                forecast = result['point_forecast']
                all_forecasts.extend(forecast)
                
                # Process confidence intervals
                if 'prediction_intervals' in result:
                    for level, intervals in result['prediction_intervals'].items():
                        if level not in all_intervals:
                            all_intervals[level] = {'lower': [], 'upper': []}
                        all_intervals[level]['lower'].extend(intervals['lower'])
                        all_intervals[level]['upper'].extend(intervals['upper'])
                
                # Calculate confidence score for this horizon
                if 'prediction_intervals' in result and '80%' in result['prediction_intervals']:
                    lower = np.array(result['prediction_intervals']['80%']['lower'])
                    upper = np.array(result['prediction_intervals']['80%']['upper'])
                    forecast_arr = np.array(forecast)
                    
                    # Confidence based on interval width (narrower = more confident)
                    interval_width = np.mean((upper - lower) / np.maximum(np.abs(forecast_arr), 1e-8))
                    confidence = max(0.1, min(1.0, 1.0 - interval_width))
                    confidence_scores.append(confidence)
            
            # Calculate overall metrics
            if not all_forecasts:
                return self._create_neutral_signal(symbol)
            
            # Get current price for trend analysis
            current_price = market_data.get('current_price', market_data.get('close', [0])[-1])
            
            # Determine trend direction
            short_term_forecast = np.mean(all_forecasts[:min(3, len(all_forecasts))])
            medium_term_forecast = np.mean(all_forecasts[:min(10, len(all_forecasts))])
            
            if medium_term_forecast > current_price * 1.02:
                trend_direction = 'bullish'
            elif medium_term_forecast < current_price * 0.98:
                trend_direction = 'bearish'
            else:
                trend_direction = 'neutral'
            
            # Calculate signal strength
            price_change_pct = (medium_term_forecast - current_price) / current_price if current_price > 0 else 0
            signal_strength = min(1.0, abs(price_change_pct) * 10)  # Scale to 0-1
            
            # Calculate volatility forecast
            forecast_std = np.std(all_forecasts) if len(all_forecasts) > 1 else 0
            volatility_forecast = forecast_std / current_price if current_price > 0 else 0
            
            # Overall confidence
            overall_confidence = np.mean(confidence_scores) if confidence_scores else 0.5
            
            return NeuralSignal(
                timestamp=datetime.now().isoformat(),
                symbol=symbol,
                forecast_horizon=max(forecast_results.keys()) if forecast_results else 1,
                point_forecast=all_forecasts[:20],  # Limit to first 20 points
                confidence_intervals=all_intervals,
                confidence_score=overall_confidence,
                trend_direction=trend_direction,
                volatility_forecast=volatility_forecast,
                signal_strength=signal_strength
            )
            
        except Exception as e:
            self.logger.error(f"Multi-horizon processing failed: {str(e)}")
            return self._create_neutral_signal(symbol)
    
    def _create_neutral_signal(self, symbol: str) -> NeuralSignal:
        """Create a neutral neural signal as fallback."""
        return NeuralSignal(
            timestamp=datetime.now().isoformat(),
            symbol=symbol,
            forecast_horizon=1,
            point_forecast=[0.0],
            confidence_intervals={},
            confidence_score=0.0,
            trend_direction='neutral',
            volatility_forecast=0.2,
            signal_strength=0.0
        )
    
    async def _combine_momentum_signals(
        self,
        traditional_signal: Dict[str, Any],
        neural_signal: NeuralSignal,
        strategy_name: str
    ) -> Dict[str, Any]:
        """
        Combine traditional momentum signal with neural forecast.
        
        Args:
            traditional_signal: Traditional momentum signal
            neural_signal: Neural forecasting signal
            strategy_name: Strategy identifier
            
        Returns:
            Combined signal
        """
        try:
            # Get adaptive weights if available
            weights = self._get_adaptive_weights(strategy_name)
            neural_weight = weights.get('neural', self.neural_weight)
            traditional_weight = weights.get('traditional', self.traditional_weight)
            
            # Extract traditional signal components
            trad_action = traditional_signal.get('action', 'HOLD')
            trad_confidence = traditional_signal.get('confidence', 0.5)
            trad_momentum_score = traditional_signal.get('momentum_score', 0.5)
            
            # Convert neural signal to momentum-like score
            neural_momentum_score = self._convert_neural_to_momentum_score(neural_signal)
            
            # Combine scores
            combined_momentum = (
                traditional_weight * trad_momentum_score +
                neural_weight * neural_momentum_score
            )
            
            # Determine action based on combined score and neural trend
            if neural_signal.trend_direction == 'bullish' and combined_momentum > 0.6:
                action = 'BUY'
                confidence = min(1.0, combined_momentum * neural_signal.confidence_score)
            elif neural_signal.trend_direction == 'bearish' and combined_momentum < 0.4:
                action = 'SELL'
                confidence = min(1.0, (1 - combined_momentum) * neural_signal.confidence_score)
            elif trad_action != 'HOLD' and neural_signal.confidence_score > self.confidence_threshold:
                action = trad_action
                confidence = traditional_signal.get('confidence', 0.5) * neural_signal.confidence_score
            else:
                action = 'HOLD'
                confidence = 0.3
            
            # Calculate position size based on confidence and volatility
            base_position = traditional_signal.get('position_size_pct', 0.1)
            volatility_adjustment = max(0.5, 1.0 - neural_signal.volatility_forecast)
            position_size = base_position * confidence * volatility_adjustment
            
            return {
                'action': action,
                'confidence': confidence,
                'combined_score': combined_momentum,
                'position_size': position_size,
                'neural_contribution': neural_weight * neural_momentum_score,
                'traditional_contribution': traditional_weight * trad_momentum_score,
                'volatility_adjustment': volatility_adjustment
            }
            
        except Exception as e:
            self.logger.error(f"Momentum signal combination failed: {str(e)}")
            return traditional_signal
    
    async def _combine_mean_reversion_signals(
        self,
        traditional_signal: Dict[str, Any],
        neural_signal: NeuralSignal,
        strategy_name: str
    ) -> Dict[str, Any]:
        """
        Combine traditional mean reversion signal with neural forecast.
        
        Args:
            traditional_signal: Traditional mean reversion signal
            neural_signal: Neural forecasting signal
            strategy_name: Strategy identifier
            
        Returns:
            Combined signal
        """
        try:
            # Get adaptive weights
            weights = self._get_adaptive_weights(strategy_name)
            neural_weight = weights.get('neural', self.neural_weight)
            traditional_weight = weights.get('traditional', self.traditional_weight)
            
            # Extract traditional components
            trad_action = traditional_signal.get('action', 'HOLD')
            trad_confidence = traditional_signal.get('confidence', 0.5)
            trad_reversion_score = traditional_signal.get('reversion_score', 0.5)
            
            # Convert neural signal for mean reversion
            neural_reversion_score = self._convert_neural_to_reversion_score(neural_signal)
            
            # Combine scores
            combined_reversion = (
                traditional_weight * trad_reversion_score +
                neural_weight * neural_reversion_score
            )
            
            # Mean reversion logic: buy when oversold, sell when overbought
            if combined_reversion < 0.3 and neural_signal.trend_direction != 'bearish':
                action = 'BUY'  # Oversold condition
                confidence = min(1.0, (1 - combined_reversion) * neural_signal.confidence_score)
            elif combined_reversion > 0.7 and neural_signal.trend_direction != 'bullish':
                action = 'SELL'  # Overbought condition
                confidence = min(1.0, combined_reversion * neural_signal.confidence_score)
            else:
                action = 'HOLD'
                confidence = 0.3
            
            # Position sizing
            base_position = traditional_signal.get('position_size_pct', 0.08)
            volatility_adjustment = max(0.4, 1.0 - neural_signal.volatility_forecast)
            position_size = base_position * confidence * volatility_adjustment
            
            return {
                'action': action,
                'confidence': confidence,
                'combined_score': combined_reversion,
                'position_size': position_size,
                'neural_contribution': neural_weight * neural_reversion_score,
                'traditional_contribution': traditional_weight * trad_reversion_score,
                'volatility_adjustment': volatility_adjustment
            }
            
        except Exception as e:
            self.logger.error(f"Mean reversion signal combination failed: {str(e)}")
            return traditional_signal
    
    async def _combine_swing_signals(
        self,
        traditional_signal: Dict[str, Any],
        neural_signal: NeuralSignal,
        strategy_name: str
    ) -> Dict[str, Any]:
        """
        Combine traditional swing signal with neural forecast.
        
        Args:
            traditional_signal: Traditional swing signal
            neural_signal: Neural forecasting signal
            strategy_name: Strategy identifier
            
        Returns:
            Combined signal
        """
        try:
            # Get adaptive weights
            weights = self._get_adaptive_weights(strategy_name)
            neural_weight = weights.get('neural', self.neural_weight)
            traditional_weight = weights.get('traditional', self.traditional_weight)
            
            # Extract traditional components
            trad_action = traditional_signal.get('action', 'HOLD')
            trad_confidence = traditional_signal.get('confidence', 0.5)
            trad_swing_score = traditional_signal.get('swing_score', 0.5)
            
            # Convert neural signal for swing trading
            neural_swing_score = self._convert_neural_to_swing_score(neural_signal)
            
            # Combine scores with emphasis on trend consistency
            combined_swing = (
                traditional_weight * trad_swing_score +
                neural_weight * neural_swing_score
            )
            
            # Swing trading logic: align with trend direction
            if neural_signal.trend_direction == 'bullish' and combined_swing > 0.55:
                action = 'BUY'
                confidence = min(1.0, combined_swing * neural_signal.confidence_score * 1.1)
            elif neural_signal.trend_direction == 'bearish' and combined_swing < 0.45:
                action = 'SELL'
                confidence = min(1.0, (1 - combined_swing) * neural_signal.confidence_score * 1.1)
            elif neural_signal.confidence_score > 0.7 and neural_signal.signal_strength > 0.5:
                # Strong neural signal overrides
                action = 'BUY' if neural_signal.trend_direction == 'bullish' else 'SELL'
                confidence = neural_signal.confidence_score * neural_signal.signal_strength
            else:
                action = 'HOLD'
                confidence = 0.4
            
            # Position sizing for swing trading (typically larger positions)
            base_position = traditional_signal.get('position_size_pct', 0.15)
            volatility_adjustment = max(0.6, 1.0 - neural_signal.volatility_forecast * 0.8)
            position_size = base_position * confidence * volatility_adjustment
            
            return {
                'action': action,
                'confidence': confidence,
                'combined_score': combined_swing,
                'position_size': position_size,
                'neural_contribution': neural_weight * neural_swing_score,
                'traditional_contribution': traditional_weight * trad_swing_score,
                'volatility_adjustment': volatility_adjustment
            }
            
        except Exception as e:
            self.logger.error(f"Swing signal combination failed: {str(e)}")
            return traditional_signal
    
    def _convert_neural_to_momentum_score(self, neural_signal: NeuralSignal) -> float:
        """Convert neural signal to momentum-compatible score."""
        try:
            # Base score from trend direction
            if neural_signal.trend_direction == 'bullish':
                base_score = 0.7
            elif neural_signal.trend_direction == 'bearish':
                base_score = 0.3
            else:
                base_score = 0.5
            
            # Adjust based on signal strength
            strength_adjustment = (neural_signal.signal_strength - 0.5) * 0.3
            
            # Adjust based on confidence
            confidence_adjustment = (neural_signal.confidence_score - 0.5) * 0.2
            
            # Combine adjustments
            momentum_score = base_score + strength_adjustment + confidence_adjustment
            
            return max(0.0, min(1.0, momentum_score))
            
        except Exception:
            return 0.5
    
    def _convert_neural_to_reversion_score(self, neural_signal: NeuralSignal) -> float:
        """Convert neural signal to mean reversion-compatible score."""
        try:
            # For mean reversion, we want opposite of trend for entry
            if neural_signal.trend_direction == 'bullish':
                base_score = 0.3  # Lower score for bullish (less likely to revert)
            elif neural_signal.trend_direction == 'bearish':
                base_score = 0.7  # Higher score for bearish (more likely to revert)
            else:
                base_score = 0.5
            
            # Adjust based on volatility (higher volatility = more reversion potential)
            volatility_adjustment = neural_signal.volatility_forecast * 0.2
            
            # Adjust based on confidence (but inverted for reversion)
            confidence_adjustment = (0.5 - neural_signal.confidence_score) * 0.1
            
            reversion_score = base_score + volatility_adjustment + confidence_adjustment
            
            return max(0.0, min(1.0, reversion_score))
            
        except Exception:
            return 0.5
    
    def _convert_neural_to_swing_score(self, neural_signal: NeuralSignal) -> float:
        """Convert neural signal to swing trading-compatible score."""
        try:
            # Base score from trend direction (swing follows trends)
            if neural_signal.trend_direction == 'bullish':
                base_score = 0.65
            elif neural_signal.trend_direction == 'bearish':
                base_score = 0.35
            else:
                base_score = 0.5
            
            # Strong adjustment based on signal strength (swing trades need strong signals)
            strength_adjustment = (neural_signal.signal_strength - 0.5) * 0.4
            
            # Confidence boost
            confidence_adjustment = (neural_signal.confidence_score - 0.5) * 0.25
            
            # Volatility consideration (moderate volatility preferred for swing)
            optimal_volatility = 0.15  # 15% is good for swing trading
            volatility_penalty = abs(neural_signal.volatility_forecast - optimal_volatility) * 0.5
            
            swing_score = base_score + strength_adjustment + confidence_adjustment - volatility_penalty
            
            return max(0.0, min(1.0, swing_score))
            
        except Exception:
            return 0.5
    
    async def _apply_risk_adjustments(
        self,
        combined_signal: Dict[str, Any],
        neural_signal: NeuralSignal,
        market_data: Dict[str, Any]
    ) -> EnhancedTradingSignal:
        """
        Apply risk adjustments to create final enhanced signal.
        
        Args:
            combined_signal: Combined signal from strategy
            neural_signal: Neural forecasting signal
            market_data: Market data for risk calculations
            
        Returns:
            Final enhanced trading signal
        """
        try:
            # Extract base components
            action = combined_signal['action']
            confidence = combined_signal['confidence']
            position_size = combined_signal['position_size']
            
            # Get current price
            current_price = market_data.get('current_price', market_data.get('close', [100])[-1])
            
            # Calculate target price from neural forecast
            if neural_signal.point_forecast:
                target_price = np.mean(neural_signal.point_forecast[:3])  # Average of first 3 forecasts
            else:
                target_price = current_price * (1.05 if action == 'BUY' else 0.95)
            
            # Calculate stop loss based on volatility
            volatility_multiplier = max(1.5, neural_signal.volatility_forecast * 10)
            if action == 'BUY':
                stop_loss = current_price * (1 - 0.05 * volatility_multiplier)
            elif action == 'SELL':
                stop_loss = current_price * (1 + 0.05 * volatility_multiplier)
            else:
                stop_loss = current_price
            
            # Risk-adjusted position sizing
            risk_adjusted_position = position_size * self.risk_adjustment_factor
            
            # Limit position size based on confidence and volatility
            max_position = 0.25 * confidence * (1 - neural_signal.volatility_forecast)
            final_position_size = min(risk_adjusted_position, max_position)
            
            # Calculate holding period based on forecast horizon and volatility
            base_holding_period = neural_signal.forecast_horizon
            volatility_adjustment = 1 - neural_signal.volatility_forecast
            holding_period = max(1, int(base_holding_period * volatility_adjustment))
            
            # Risk metrics
            risk_metrics = {
                'volatility_forecast': neural_signal.volatility_forecast,
                'confidence_score': neural_signal.confidence_score,
                'signal_strength': neural_signal.signal_strength,
                'position_risk': final_position_size / confidence if confidence > 0 else 0,
                'stop_loss_distance': abs(current_price - stop_loss) / current_price
            }
            
            # Generate reasoning
            reasoning = self._generate_enhanced_reasoning(
                action, neural_signal, combined_signal, risk_metrics
            )
            
            return EnhancedTradingSignal(
                symbol=neural_signal.symbol,
                action=action,
                confidence=confidence,
                position_size=final_position_size,
                entry_price=current_price,
                target_price=target_price,
                stop_loss=stop_loss,
                holding_period=holding_period,
                neural_signal=neural_signal,
                traditional_signal=combined_signal,
                combined_score=combined_signal['combined_score'],
                risk_metrics=risk_metrics,
                reasoning=reasoning
            )
            
        except Exception as e:
            self.logger.error(f"Risk adjustment failed: {str(e)}")
            # Return basic signal as fallback
            return EnhancedTradingSignal(
                symbol=neural_signal.symbol,
                action='HOLD',
                confidence=0.3,
                position_size=0.01,
                entry_price=100.0,
                target_price=100.0,
                stop_loss=95.0,
                holding_period=1,
                neural_signal=neural_signal,
                traditional_signal=combined_signal,
                combined_score=0.5,
                risk_metrics={},
                reasoning="Risk adjustment failed - using conservative defaults"
            )
    
    def _generate_enhanced_reasoning(
        self,
        action: str,
        neural_signal: NeuralSignal,
        combined_signal: Dict[str, Any],
        risk_metrics: Dict[str, float]
    ) -> str:
        """Generate human-readable reasoning for the enhanced signal."""
        try:
            reasoning_parts = []
            
            # Action reasoning
            reasoning_parts.append(f"Action: {action}")
            
            # Neural forecast contribution
            if neural_signal.trend_direction != 'neutral':
                reasoning_parts.append(
                    f"Neural forecast shows {neural_signal.trend_direction} trend "
                    f"(confidence: {neural_signal.confidence_score:.2f})"
                )
            
            # Signal strength
            if neural_signal.signal_strength > 0.7:
                reasoning_parts.append("Strong neural signal detected")
            elif neural_signal.signal_strength < 0.3:
                reasoning_parts.append("Weak neural signal - proceeding cautiously")
            
            # Combined score interpretation
            combined_score = combined_signal.get('combined_score', 0.5)
            if combined_score > 0.7:
                reasoning_parts.append("Strong combined signal from traditional + neural analysis")
            elif combined_score < 0.3:
                reasoning_parts.append("Weak combined signal suggests caution")
            
            # Risk considerations
            volatility = neural_signal.volatility_forecast
            if volatility > 0.25:
                reasoning_parts.append(f"High volatility ({volatility:.1%}) - reduced position size")
            elif volatility < 0.1:
                reasoning_parts.append(f"Low volatility ({volatility:.1%}) - favorable conditions")
            
            # Confidence level
            confidence = neural_signal.confidence_score
            if confidence > 0.8:
                reasoning_parts.append("High confidence in forecast")
            elif confidence < 0.4:
                reasoning_parts.append("Low confidence - conservative approach")
            
            return ". ".join(reasoning_parts) + "."
            
        except Exception:
            return f"Enhanced {action} signal based on neural-traditional strategy combination"
    
    def _get_adaptive_weights(self, strategy_name: str) -> Dict[str, float]:
        """Get adaptive weights based on strategy performance."""
        if not self.enable_adaptive_weighting or strategy_name not in self.adaptive_weights:
            return {'neural': self.neural_weight, 'traditional': self.traditional_weight}
        
        return self.adaptive_weights[strategy_name]
    
    async def _track_signal_performance(self, strategy_name: str, signal: EnhancedTradingSignal):
        """Track signal performance for adaptive weight adjustment."""
        try:
            if strategy_name not in self.performance_history:
                self.performance_history[strategy_name] = []
            
            signal_record = {
                'timestamp': datetime.now().isoformat(),
                'symbol': signal.symbol,
                'action': signal.action,
                'confidence': signal.confidence,
                'position_size': signal.position_size,
                'neural_confidence': signal.neural_signal.confidence_score,
                'neural_strength': signal.neural_signal.signal_strength,
                'combined_score': signal.combined_score
            }
            
            self.performance_history[strategy_name].append(signal_record)
            
            # Keep only recent history (last 1000 signals)
            if len(self.performance_history[strategy_name]) > 1000:
                self.performance_history[strategy_name] = self.performance_history[strategy_name][-1000:]
            
            # Update adaptive weights if we have enough history
            if len(self.performance_history[strategy_name]) > 50:
                await self._update_adaptive_weights(strategy_name)
                
        except Exception as e:
            self.logger.warning(f"Failed to track signal performance: {str(e)}")
    
    async def _update_adaptive_weights(self, strategy_name: str):
        """Update adaptive weights based on performance history."""
        # This would implement performance-based weight adjustment
        # For now, we keep the default weights
        # Future implementation could analyze signal success rates and adjust weights accordingly
        pass
    
    async def _fallback_signal(
        self,
        symbol: str,
        traditional_signal: Dict[str, Any],
        fallback_reason: str
    ) -> EnhancedTradingSignal:
        """Create fallback signal when neural enhancement fails."""
        try:
            neutral_neural = self._create_neutral_signal(symbol)
            
            return EnhancedTradingSignal(
                symbol=symbol,
                action=traditional_signal.get('action', 'HOLD'),
                confidence=traditional_signal.get('confidence', 0.3),
                position_size=traditional_signal.get('position_size_pct', 0.05),
                entry_price=traditional_signal.get('current_price', 100.0),
                target_price=traditional_signal.get('target_price', 105.0),
                stop_loss=traditional_signal.get('stop_loss', 95.0),
                holding_period=traditional_signal.get('holding_period', 5),
                neural_signal=neutral_neural,
                traditional_signal=traditional_signal,
                combined_score=traditional_signal.get('momentum_score', 0.5),
                risk_metrics={'fallback': True},
                reasoning=f"Using traditional signal only due to: {fallback_reason}"
            )
            
        except Exception as e:
            self.logger.error(f"Fallback signal creation failed: {str(e)}")
            neutral_neural = self._create_neutral_signal(symbol)
            
            return EnhancedTradingSignal(
                symbol=symbol,
                action='HOLD',
                confidence=0.1,
                position_size=0.01,
                entry_price=100.0,
                target_price=100.0,
                stop_loss=95.0,
                holding_period=1,
                neural_signal=neutral_neural,
                traditional_signal={},
                combined_score=0.5,
                risk_metrics={'error': True},
                reasoning="System error - using minimal risk default signal"
            )
    
    def get_enhancement_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about strategy enhancement performance.
        
        Returns:
            Enhancement statistics
        """
        try:
            stats = {
                'total_strategies': len(self.performance_history),
                'cached_signals': len(self.signal_cache),
                'adaptive_weighting_enabled': self.enable_adaptive_weighting,
                'current_weights': {
                    'neural': self.neural_weight,
                    'traditional': self.traditional_weight
                },
                'strategy_stats': {}
            }
            
            for strategy_name, history in self.performance_history.items():
                if history:
                    stats['strategy_stats'][strategy_name] = {
                        'total_signals': len(history),
                        'avg_confidence': np.mean([h['confidence'] for h in history]),
                        'avg_neural_confidence': np.mean([h['neural_confidence'] for h in history]),
                        'avg_position_size': np.mean([h['position_size'] for h in history]),
                        'action_distribution': self._calculate_action_distribution(history)
                    }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to generate statistics: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_action_distribution(self, history: List[Dict]) -> Dict[str, float]:
        """Calculate distribution of actions in signal history."""
        actions = [h['action'] for h in history]
        total = len(actions)
        
        if total == 0:
            return {}
        
        distribution = {}
        for action in ['BUY', 'SELL', 'HOLD']:
            count = actions.count(action)
            distribution[action] = count / total
        
        return distribution
    
    def clear_cache(self):
        """Clear signal cache to free memory."""
        self.signal_cache.clear()
        self.logger.info("Signal cache cleared")