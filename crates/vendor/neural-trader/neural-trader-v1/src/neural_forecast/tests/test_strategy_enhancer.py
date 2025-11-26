"""
Unit Tests for Strategy Enhancer.

Tests for neural-enhanced trading strategy integration including:
- Strategy enhancement workflows
- Signal combination and weighting
- Risk adjustment mechanisms
- Performance tracking
"""

import pytest
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from dataclasses import asdict

# Import the module under test
import sys
sys.path.insert(0, '/workspaces/ai-news-trader/src')

from neural_forecast.strategy_enhancer import (
    StrategyEnhancer, 
    NeuralSignal, 
    EnhancedTradingSignal,
    FallbackStrategy
)
from neural_forecast.neural_model_manager import NeuralModelManager


class TestStrategyEnhancer:
    """Test suite for StrategyEnhancer class."""
    
    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data for testing."""
        return {
            'prices': [100, 101, 102, 103, 104, 105],
            'timestamps': pd.date_range(start='2023-01-01', periods=6, freq='H').tolist(),
            'current_price': 105,
            'volume': [1000, 1200, 1100, 1300, 1150, 1250],
            'close': [100, 101, 102, 103, 104, 105]
        }
    
    @pytest.fixture
    def sample_traditional_signal(self):
        """Create sample traditional trading signal."""
        return {
            'action': 'BUY',
            'confidence': 0.75,
            'momentum_score': 0.68,
            'position_size_pct': 0.1,
            'current_price': 105,
            'target_price': 110,
            'stop_loss': 100,
            'holding_period': 5
        }
    
    @pytest.fixture
    def mock_model_manager(self):
        """Create mock model manager."""
        manager = Mock(spec=NeuralModelManager)
        manager.predict = AsyncMock()
        return manager
    
    @pytest.fixture
    def strategy_enhancer(self, mock_model_manager):
        """Create StrategyEnhancer instance for testing."""
        return StrategyEnhancer(
            model_manager=mock_model_manager,
            neural_weight=0.4,
            traditional_weight=0.6,
            enable_adaptive_weighting=False  # Disable for testing
        )
    
    @pytest.fixture
    def sample_neural_signal(self):
        """Create sample neural signal."""
        return NeuralSignal(
            timestamp=datetime.now().isoformat(),
            symbol="AAPL",
            forecast_horizon=12,
            point_forecast=[106, 107, 108, 109, 110],
            confidence_intervals={
                '80%': {
                    'lower': [104, 105, 106, 107, 108],
                    'upper': [108, 109, 110, 111, 112]
                }
            },
            confidence_score=0.8,
            trend_direction='bullish',
            volatility_forecast=0.15,
            signal_strength=0.7
        )
    
    def test_strategy_enhancer_initialization(self):
        """Test strategy enhancer initialization."""
        enhancer = StrategyEnhancer()
        
        assert enhancer.forecast_horizons == [1, 3, 5, 10]
        assert enhancer.confidence_threshold == 0.6
        assert enhancer.neural_weight == 0.4
        assert enhancer.traditional_weight == 0.6
        assert enhancer.enable_adaptive_weighting == True
        assert isinstance(enhancer.performance_history, dict)
        assert isinstance(enhancer.signal_cache, dict)
    
    def test_data_preparation(self, strategy_enhancer, sample_market_data):
        """Test market data preparation for forecasting."""
        forecast_data = strategy_enhancer._prepare_forecast_data(sample_market_data)
        
        assert 'ds' in forecast_data
        assert 'y' in forecast_data
        assert 'unique_id' in forecast_data
        assert len(forecast_data['y']) == len(sample_market_data['prices'])
        assert forecast_data['unique_id'] == 'main_series'
    
    def test_data_preparation_missing_prices(self, strategy_enhancer):
        """Test data preparation with missing price data."""
        invalid_data = {'volume': [1000, 1100], 'invalid': 'data'}
        
        with pytest.raises(ValueError, match="No price data found"):
            strategy_enhancer._prepare_forecast_data(invalid_data)
    
    def test_neural_signal_creation(self, sample_neural_signal):
        """Test neural signal data structure."""
        assert sample_neural_signal.symbol == "AAPL"
        assert sample_neural_signal.trend_direction == 'bullish'
        assert sample_neural_signal.confidence_score == 0.8
        assert len(sample_neural_signal.point_forecast) == 5
        assert '80%' in sample_neural_signal.confidence_intervals
    
    def test_create_neutral_signal(self, strategy_enhancer):
        """Test creation of neutral neural signal."""
        signal = strategy_enhancer._create_neutral_signal("AAPL")
        
        assert signal.symbol == "AAPL"
        assert signal.trend_direction == 'neutral'
        assert signal.confidence_score == 0.0
        assert signal.signal_strength == 0.0
        assert len(signal.point_forecast) == 1
    
    @pytest.mark.asyncio
    async def test_neural_signal_generation_mock(self, strategy_enhancer, sample_market_data):
        """Test neural signal generation with mocked model manager."""
        # Mock the prediction result
        mock_result = {
            'success': True,
            'point_forecast': [106, 107, 108],
            'prediction_intervals': {
                '80%': {
                    'lower': [104, 105, 106],
                    'upper': [108, 109, 110]
                }
            }
        }
        
        strategy_enhancer.model_manager.predict.return_value = mock_result
        
        # Mock the forecaster horizon setting
        mock_forecaster = Mock()
        strategy_enhancer.model_manager.forecaster = mock_forecaster
        
        signal = await strategy_enhancer._generate_neural_signal("AAPL", sample_market_data)
        
        assert signal.symbol == "AAPL"
        assert signal.success == True
        assert len(signal.point_forecast) > 0
        assert signal.confidence_score > 0
    
    @pytest.mark.asyncio
    async def test_neural_signal_generation_failure(self, strategy_enhancer, sample_market_data):
        """Test neural signal generation with prediction failure."""
        # Mock failed prediction
        strategy_enhancer.model_manager.predict.return_value = {
            'success': False,
            'error': 'Prediction failed'
        }
        
        signal = await strategy_enhancer._generate_neural_signal("AAPL", sample_market_data)
        
        assert signal.symbol == "AAPL"
        assert signal.trend_direction == 'neutral'
        assert signal.confidence_score == 0.0
    
    def test_multi_horizon_forecast_processing(self, strategy_enhancer, sample_market_data):
        """Test processing of multi-horizon forecast results."""
        forecast_results = {
            1: {
                'success': True,
                'point_forecast': [106],
                'prediction_intervals': {
                    '80%': {'lower': [104], 'upper': [108]}
                }
            },
            3: {
                'success': True,
                'point_forecast': [107, 108, 109],
                'prediction_intervals': {
                    '80%': {'lower': [105, 106, 107], 'upper': [109, 110, 111]}
                }
            }
        }
        
        # Add current price to market data
        sample_market_data['current_price'] = 105
        
        signal = strategy_enhancer._process_multi_horizon_forecasts(
            "AAPL", forecast_results, sample_market_data
        )
        
        assert signal.symbol == "AAPL"
        assert len(signal.point_forecast) > 0
        assert signal.trend_direction in ['bullish', 'bearish', 'neutral']
        assert 0 <= signal.confidence_score <= 1
        assert 0 <= signal.signal_strength <= 1
    
    def test_convert_neural_to_momentum_score(self, strategy_enhancer, sample_neural_signal):
        """Test conversion of neural signal to momentum score."""
        score = strategy_enhancer._convert_neural_to_momentum_score(sample_neural_signal)
        
        assert 0 <= score <= 1
        assert isinstance(score, float)
        
        # Test with bearish signal
        bearish_signal = sample_neural_signal
        bearish_signal.trend_direction = 'bearish'
        bearish_score = strategy_enhancer._convert_neural_to_momentum_score(bearish_signal)
        
        assert bearish_score < score  # Bearish should have lower momentum score
    
    def test_convert_neural_to_reversion_score(self, strategy_enhancer, sample_neural_signal):
        """Test conversion of neural signal to mean reversion score."""
        score = strategy_enhancer._convert_neural_to_reversion_score(sample_neural_signal)
        
        assert 0 <= score <= 1
        assert isinstance(score, float)
        
        # For mean reversion, bullish signals should have lower scores (less likely to revert)
        assert score < 0.5  # Bullish trend less likely to revert
    
    def test_convert_neural_to_swing_score(self, strategy_enhancer, sample_neural_signal):
        """Test conversion of neural signal to swing trading score."""
        score = strategy_enhancer._convert_neural_to_swing_score(sample_neural_signal)
        
        assert 0 <= score <= 1
        assert isinstance(score, float)
        
        # Swing trading follows trends, so bullish should have higher score
        assert score > 0.5  # Bullish trend good for swing trading
    
    @pytest.mark.asyncio
    async def test_combine_momentum_signals(self, strategy_enhancer, sample_traditional_signal, sample_neural_signal):
        """Test combination of traditional and neural momentum signals."""
        combined = await strategy_enhancer._combine_momentum_signals(
            sample_traditional_signal, sample_neural_signal, "momentum_test"
        )
        
        assert 'action' in combined
        assert 'confidence' in combined
        assert 'combined_score' in combined
        assert 'position_size' in combined
        assert combined['action'] in ['BUY', 'SELL', 'HOLD']
        assert 0 <= combined['confidence'] <= 1
    
    @pytest.mark.asyncio
    async def test_combine_mean_reversion_signals(self, strategy_enhancer, sample_traditional_signal, sample_neural_signal):
        """Test combination of traditional and neural mean reversion signals."""
        # Modify traditional signal for mean reversion
        reversion_signal = sample_traditional_signal.copy()
        reversion_signal['reversion_score'] = 0.8  # Overbought condition
        
        combined = await strategy_enhancer._combine_mean_reversion_signals(
            reversion_signal, sample_neural_signal, "reversion_test"
        )
        
        assert 'action' in combined
        assert 'confidence' in combined
        assert 'combined_score' in combined
        # Mean reversion should suggest sell on overbought + bullish neural
        assert combined['action'] in ['BUY', 'SELL', 'HOLD']
    
    @pytest.mark.asyncio
    async def test_combine_swing_signals(self, strategy_enhancer, sample_traditional_signal, sample_neural_signal):
        """Test combination of traditional and neural swing signals."""
        # Modify traditional signal for swing trading
        swing_signal = sample_traditional_signal.copy()
        swing_signal['swing_score'] = 0.7
        
        combined = await strategy_enhancer._combine_swing_signals(
            swing_signal, sample_neural_signal, "swing_test"
        )
        
        assert 'action' in combined
        assert 'confidence' in combined
        assert 'combined_score' in combined
        # Swing trading should align with bullish trend
        assert combined['action'] == 'BUY'
    
    @pytest.mark.asyncio
    async def test_apply_risk_adjustments(self, strategy_enhancer, sample_neural_signal, sample_market_data):
        """Test risk adjustment application."""
        combined_signal = {
            'action': 'BUY',
            'confidence': 0.8,
            'position_size': 0.15,
            'combined_score': 0.75
        }
        
        enhanced_signal = await strategy_enhancer._apply_risk_adjustments(
            combined_signal, sample_neural_signal, sample_market_data
        )
        
        assert isinstance(enhanced_signal, EnhancedTradingSignal)
        assert enhanced_signal.symbol == sample_neural_signal.symbol
        assert enhanced_signal.action == 'BUY'
        assert 0 <= enhanced_signal.confidence <= 1
        assert enhanced_signal.position_size > 0
        assert enhanced_signal.entry_price > 0
        assert enhanced_signal.holding_period > 0
        assert isinstance(enhanced_signal.risk_metrics, dict)
    
    @pytest.mark.asyncio
    async def test_enhance_momentum_strategy_full_workflow(self, strategy_enhancer, sample_market_data, sample_traditional_signal):
        """Test complete momentum strategy enhancement workflow."""
        # Mock neural signal generation
        with patch.object(strategy_enhancer, '_generate_neural_signal') as mock_neural:
            mock_neural.return_value = NeuralSignal(
                timestamp=datetime.now().isoformat(),
                symbol="AAPL",
                forecast_horizon=12,
                point_forecast=[107, 108],
                confidence_intervals={},
                confidence_score=0.75,
                trend_direction='bullish',
                volatility_forecast=0.12,
                signal_strength=0.8
            )
            
            enhanced_signal = await strategy_enhancer.enhance_momentum_strategy(
                "AAPL", sample_market_data, sample_traditional_signal
            )
            
            assert isinstance(enhanced_signal, EnhancedTradingSignal)
            assert enhanced_signal.symbol == "AAPL"
            assert enhanced_signal.action in ['BUY', 'SELL', 'HOLD']
    
    @pytest.mark.asyncio
    async def test_enhance_mean_reversion_strategy_full_workflow(self, strategy_enhancer, sample_market_data, sample_traditional_signal):
        """Test complete mean reversion strategy enhancement workflow."""
        # Modify signal for mean reversion
        reversion_signal = sample_traditional_signal.copy()
        reversion_signal['action'] = 'SELL'  # Overbought
        reversion_signal['reversion_score'] = 0.85
        
        with patch.object(strategy_enhancer, '_generate_neural_signal') as mock_neural:
            mock_neural.return_value = NeuralSignal(
                timestamp=datetime.now().isoformat(),
                symbol="AAPL",
                forecast_horizon=12,
                point_forecast=[103, 102],  # Declining forecast
                confidence_intervals={},
                confidence_score=0.7,
                trend_direction='bearish',
                volatility_forecast=0.18,
                signal_strength=0.6
            )
            
            enhanced_signal = await strategy_enhancer.enhance_mean_reversion_strategy(
                "AAPL", sample_market_data, reversion_signal
            )
            
            assert isinstance(enhanced_signal, EnhancedTradingSignal)
            assert enhanced_signal.symbol == "AAPL"
    
    @pytest.mark.asyncio
    async def test_enhance_swing_strategy_full_workflow(self, strategy_enhancer, sample_market_data, sample_traditional_signal):
        """Test complete swing strategy enhancement workflow."""
        swing_signal = sample_traditional_signal.copy()
        swing_signal['swing_score'] = 0.75
        
        with patch.object(strategy_enhancer, '_generate_neural_signal') as mock_neural:
            mock_neural.return_value = NeuralSignal(
                timestamp=datetime.now().isoformat(),
                symbol="AAPL",
                forecast_horizon=15,  # Longer horizon for swing
                point_forecast=[108, 110, 112],
                confidence_intervals={},
                confidence_score=0.8,
                trend_direction='bullish',
                volatility_forecast=0.14,
                signal_strength=0.85
            )
            
            enhanced_signal = await strategy_enhancer.enhance_swing_strategy(
                "AAPL", sample_market_data, swing_signal
            )
            
            assert isinstance(enhanced_signal, EnhancedTradingSignal)
            assert enhanced_signal.symbol == "AAPL"
            # Should favor BUY for strong bullish swing signal
            assert enhanced_signal.action == 'BUY'
    
    def test_generate_enhanced_reasoning(self, strategy_enhancer, sample_neural_signal):
        """Test enhanced reasoning generation."""
        combined_signal = {'combined_score': 0.8}
        risk_metrics = {'volatility_forecast': 0.15}
        
        reasoning = strategy_enhancer._generate_enhanced_reasoning(
            'BUY', sample_neural_signal, combined_signal, risk_metrics
        )
        
        assert isinstance(reasoning, str)
        assert 'BUY' in reasoning
        assert 'bullish' in reasoning.lower()
        assert reasoning.endswith('.')
    
    @pytest.mark.asyncio
    async def test_strategy_enhancement_with_exception(self, strategy_enhancer, sample_market_data, sample_traditional_signal):
        """Test strategy enhancement with exception handling."""
        # Mock neural signal generation to raise exception
        with patch.object(strategy_enhancer, '_generate_neural_signal', side_effect=Exception("Test error")):
            enhanced_signal = await strategy_enhancer.enhance_momentum_strategy(
                "AAPL", sample_market_data, sample_traditional_signal
            )
            
            # Should return fallback signal
            assert isinstance(enhanced_signal, EnhancedTradingSignal)
            assert enhanced_signal.symbol == "AAPL"
            assert 'fallback' in enhanced_signal.reasoning.lower()
    
    def test_signal_caching(self, strategy_enhancer, sample_market_data):
        """Test neural signal caching mechanism."""
        # Test cache key generation
        cache_key = strategy_enhancer._generate_cache_key(
            "AAPL", FallbackStrategy.SIMPLE_AVERAGE, (sample_market_data,), {}
        )
        
        assert isinstance(cache_key, str)
        assert "AAPL" in cache_key
        
        # Test cache storage and retrieval
        mock_signal = strategy_enhancer._create_neutral_signal("AAPL")
        strategy_enhancer.signal_cache[cache_key] = mock_signal
        
        assert cache_key in strategy_enhancer.signal_cache
        assert strategy_enhancer.signal_cache[cache_key].symbol == "AAPL"
    
    def test_performance_tracking(self, strategy_enhancer):
        """Test performance tracking functionality."""
        enhanced_signal = EnhancedTradingSignal(
            symbol="AAPL",
            action="BUY",
            confidence=0.8,
            position_size=0.1,
            entry_price=105,
            target_price=110,
            stop_loss=100,
            holding_period=5,
            neural_signal=strategy_enhancer._create_neutral_signal("AAPL"),
            traditional_signal={},
            combined_score=0.75,
            risk_metrics={},
            reasoning="Test signal"
        )
        
        # Test tracking
        asyncio.run(strategy_enhancer._track_signal_performance("test_strategy", enhanced_signal))
        
        assert "test_strategy" in strategy_enhancer.performance_history
        assert len(strategy_enhancer.performance_history["test_strategy"]) == 1
        
        signal_record = strategy_enhancer.performance_history["test_strategy"][0]
        assert signal_record['symbol'] == "AAPL"
        assert signal_record['action'] == "BUY"
        assert signal_record['confidence'] == 0.8
    
    def test_enhancement_statistics(self, strategy_enhancer):
        """Test enhancement statistics generation."""
        # Add some mock performance history
        strategy_enhancer.performance_history["momentum"] = [
            {
                'timestamp': datetime.now().isoformat(),
                'symbol': 'AAPL',
                'action': 'BUY',
                'confidence': 0.8,
                'position_size': 0.1,
                'neural_confidence': 0.75,
                'neural_strength': 0.7,
                'combined_score': 0.78
            },
            {
                'timestamp': datetime.now().isoformat(),
                'symbol': 'GOOGL',
                'action': 'SELL',
                'confidence': 0.6,
                'position_size': 0.05,
                'neural_confidence': 0.65,
                'neural_strength': 0.5,
                'combined_score': 0.62
            }
        ]
        
        stats = strategy_enhancer.get_enhancement_statistics()
        
        assert 'total_strategies' in stats
        assert 'current_weights' in stats
        assert 'strategy_stats' in stats
        assert stats['total_strategies'] == 1
        assert 'momentum' in stats['strategy_stats']
        
        momentum_stats = stats['strategy_stats']['momentum']
        assert momentum_stats['total_signals'] == 2
        assert 0 <= momentum_stats['avg_confidence'] <= 1
        assert 'action_distribution' in momentum_stats
    
    def test_cache_clearing(self, strategy_enhancer):
        """Test signal cache clearing."""
        # Add some cache entries
        strategy_enhancer.signal_cache['test1'] = strategy_enhancer._create_neutral_signal("AAPL")
        strategy_enhancer.signal_cache['test2'] = strategy_enhancer._create_neutral_signal("GOOGL")
        
        assert len(strategy_enhancer.signal_cache) == 2
        
        strategy_enhancer.clear_cache()
        
        assert len(strategy_enhancer.signal_cache) == 0
    
    def test_adaptive_weights_disabled(self, strategy_enhancer):
        """Test adaptive weights when disabled."""
        weights = strategy_enhancer._get_adaptive_weights("test_strategy")
        
        # Should return default weights when adaptive weighting is disabled
        assert weights['neural'] == strategy_enhancer.neural_weight
        assert weights['traditional'] == strategy_enhancer.traditional_weight


# Test runner for development
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])