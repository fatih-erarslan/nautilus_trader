"""
Comprehensive test suite for Neural Momentum Strategy
"""

import pytest
import asyncio
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from strategies.momentum.neural_momentum_strategy import NeuralMomentumStrategy, MomentumSignal, StrategyParameters
from strategies.momentum.backtesting_engine import BacktestingEngine, BacktestConfig
from strategies.momentum.strategy_orchestrator import StrategyOrchestrator
from models.neural.momentum_predictor import MomentumPredictor
from risk_management.adaptive_risk_manager import AdaptiveRiskManager
from monitoring.performance_tracker import PerformanceTracker

class TestNeuralMomentumStrategy:
    """Test cases for the Neural Momentum Strategy"""
    
    @pytest.fixture
    def strategy_config(self):
        """Default strategy configuration for testing"""
        return {
            'parameters': {
                'momentum_threshold': 0.6,
                'neural_confidence_min': 0.7,
                'max_position_size': 0.05,
                'stop_loss_pct': 0.02
            },
            'neural_config': {
                'input_dim': 50,
                'hidden_dims': [64, 32],
                'learning_rate': 0.001
            },
            'risk_config': {
                'max_portfolio_risk': 0.02,
                'volatility_lookback': 20
            },
            'monitoring_config': {
                'risk_free_rate': 0.02
            }
        }
    
    @pytest.fixture
    def strategy(self, strategy_config):
        """Create strategy instance for testing"""
        return NeuralMomentumStrategy(strategy_config)
    
    @pytest.mark.asyncio
    async def test_strategy_initialization(self, strategy):
        """Test strategy initialization"""
        assert strategy is not None
        assert isinstance(strategy.parameters, StrategyParameters)
        assert strategy.parameters.momentum_threshold == 0.6
        assert strategy.parameters.neural_confidence_min == 0.7
        assert len(strategy.positions) == 0
        assert len(strategy.active_signals) == 0
    
    @pytest.mark.asyncio
    async def test_market_regime_analysis(self, strategy):
        """Test market regime detection"""
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        
        # Mock market data
        with patch.object(strategy, '_get_market_data') as mock_data:
            mock_data.return_value = {
                'AAPL': {'price': 150, 'volatility': 0.25, 'volume': 1000000},
                'MSFT': {'price': 300, 'volatility': 0.20, 'volume': 800000},
                'GOOGL': {'price': 2500, 'volatility': 0.30, 'volume': 500000}
            }
            
            regime = await strategy.analyze_market_conditions(symbols)
            
            assert regime is not None
            assert hasattr(regime, 'volatility')
            assert hasattr(regime, 'trend_strength')
            assert regime.volatility in ['low', 'medium', 'high']
            assert 0 <= regime.trend_strength <= 1
    
    @pytest.mark.asyncio
    async def test_parameter_optimization(self, strategy):
        """Test dynamic parameter optimization"""
        from strategies.momentum.neural_momentum_strategy import MarketRegime
        
        # Test high volatility regime
        high_vol_regime = MarketRegime('high', 0.8, 'high', 'medium', 'bullish')
        optimized_params = await strategy.optimize_parameters(high_vol_regime)
        
        assert optimized_params.momentum_threshold > strategy.parameters.momentum_threshold
        assert optimized_params.stop_loss_pct > strategy.parameters.stop_loss_pct
        assert optimized_params.max_position_size < strategy.parameters.max_position_size
    
    @pytest.mark.asyncio
    async def test_signal_generation(self, strategy):
        """Test momentum signal generation"""
        symbols = ['AAPL', 'TSLA']
        
        # Mock all required methods
        with patch.object(strategy, 'analyze_market_conditions') as mock_regime, \
             patch.object(strategy, 'optimize_parameters') as mock_optimize, \
             patch.object(strategy, '_generate_signal_for_symbol') as mock_signal:
            
            # Mock market regime
            from strategies.momentum.neural_momentum_strategy import MarketRegime
            mock_regime.return_value = MarketRegime('medium', 0.6, 'high', 'medium', 'bullish')
            mock_optimize.return_value = strategy.parameters
            
            # Mock signal generation
            mock_signal.side_effect = [
                MomentumSignal(
                    symbol='AAPL',
                    direction='long',
                    strength=0.8,
                    confidence=0.75,
                    neural_prediction=0.7,
                    technical_score=0.8,
                    sentiment_score=0.3,
                    entry_price=150.0,
                    stop_loss=147.0,
                    target_price=156.0,
                    position_size=0.04,
                    timestamp=datetime.now()
                ),
                None  # No signal for TSLA
            ]
            
            signals = await strategy.generate_signals(symbols)
            
            assert len(signals) == 1
            assert signals[0].symbol == 'AAPL'
            assert signals[0].direction == 'long'
            assert signals[0].confidence >= 0.7
    
    @pytest.mark.asyncio
    async def test_trade_execution(self, strategy):
        """Test trade execution logic"""
        signals = [
            MomentumSignal(
                symbol='AAPL',
                direction='long',
                strength=0.8,
                confidence=0.75,
                neural_prediction=0.7,
                technical_score=0.8,
                sentiment_score=0.3,
                entry_price=150.0,
                stop_loss=147.0,
                target_price=156.0,
                position_size=0.04,
                timestamp=datetime.now()
            )
        ]
        
        # Mock trade execution
        with patch.object(strategy, '_execute_trade') as mock_execute:
            mock_execute.return_value = {
                'symbol': 'AAPL',
                'direction': 'long',
                'quantity': 0.04,
                'price': 150.0,
                'timestamp': datetime.now(),
                'trade_id': 'TEST_123'
            }
            
            executed_trades = await strategy.execute_trades(signals)
            
            assert len(executed_trades) == 1
            assert executed_trades[0]['symbol'] == 'AAPL'
            assert 'AAPL' in strategy.positions

class TestNeuralPredictor:
    """Test cases for the Neural Momentum Predictor"""
    
    @pytest.fixture
    def predictor_config(self):
        return {
            'input_dim': 50,
            'hidden_dims': [64, 32],
            'learning_rate': 0.001,
            'batch_size': 32
        }
    
    @pytest.fixture
    def predictor(self, predictor_config):
        return MomentumPredictor(predictor_config)
    
    @pytest.mark.asyncio
    async def test_predictor_initialization(self, predictor):
        """Test predictor initialization"""
        assert predictor is not None
        assert predictor.input_dim == 50
        assert predictor.model is not None
    
    @pytest.mark.asyncio
    async def test_feature_extraction(self, predictor):
        """Test feature extraction"""
        market_data = {
            'price': 150.0,
            'volume': 1000000,
            'rsi': 55.0,
            'macd': 0.5,
            'volatility': 0.2
        }
        
        features = await predictor._extract_features('AAPL', market_data)
        
        assert features is not None
        assert len(features) == predictor.input_dim
        assert all(isinstance(f, (int, float, np.integer, np.floating)) for f in features)
    
    @pytest.mark.asyncio
    async def test_prediction(self, predictor):
        """Test momentum prediction"""
        market_data = {
            'price': 150.0,
            'volume': 1000000,
            'rsi': 55.0,
            'macd': 0.5
        }
        
        # Mock the pretrained model loading
        with patch.object(predictor, '_load_pretrained_model'):
            predictor.is_trained = True
            
            prediction = await predictor.predict('AAPL', market_data)
            
            assert 'prediction' in prediction
            assert 'confidence' in prediction
            assert -1 <= prediction['prediction'] <= 1
            assert 0 <= prediction['confidence'] <= 1

class TestRiskManager:
    """Test cases for the Adaptive Risk Manager"""
    
    @pytest.fixture
    def risk_config(self):
        return {
            'max_portfolio_risk': 0.02,
            'max_position_size': 0.05,
            'volatility_lookback': 20
        }
    
    @pytest.fixture
    def risk_manager(self, risk_config):
        return AdaptiveRiskManager(risk_config)
    
    @pytest.mark.asyncio
    async def test_position_size_calculation(self, risk_manager):
        """Test position size calculation"""
        with patch.object(risk_manager, '_get_volatility_adjustment') as mock_vol, \
             patch.object(risk_manager, '_get_correlation_adjustment') as mock_corr, \
             patch.object(risk_manager, '_get_portfolio_heat_adjustment') as mock_heat:
            
            mock_vol.return_value = 1.0
            mock_corr.return_value = 1.0
            mock_heat.return_value = 1.0
            
            position_size = await risk_manager.calculate_position_size(
                symbol='AAPL',
                entry_price=150.0,
                stop_loss=147.0,
                confidence=0.75
            )
            
            assert position_size > 0
            assert position_size <= risk_manager.max_position_size
    
    @pytest.mark.asyncio
    async def test_risk_limit_checks(self, risk_manager):
        """Test risk limit validation"""
        new_position = {
            'symbol': 'AAPL',
            'position_size': 0.03,
            'direction': 'long',
            'value': 4500
        }
        
        checks = await risk_manager.check_risk_limits(new_position)
        
        assert isinstance(checks, dict)
        assert 'position_size_ok' in checks
        assert 'sector_exposure_ok' in checks
        assert 'correlation_ok' in checks
        assert all(isinstance(v, bool) for v in checks.values())

class TestBacktestingEngine:
    """Test cases for the Backtesting Engine"""
    
    @pytest.fixture
    def backtest_config(self):
        return BacktestConfig(
            start_date=datetime.now() - timedelta(days=365),
            end_date=datetime.now() - timedelta(days=1),
            initial_capital=100000,
            symbols=['AAPL', 'MSFT'],
            benchmark_symbol='SPY',
            transaction_costs=0.001,
            slippage=0.0005,
            max_positions=10,
            rebalance_frequency='daily',
            walk_forward_periods=12,
            optimization_metric='sharpe_ratio',
            regime_analysis=True,
            monte_carlo_runs=1000
        )
    
    @pytest.fixture
    def backtest_engine(self, backtest_config):
        return BacktestingEngine(backtest_config)
    
    @pytest.mark.asyncio
    async def test_backtest_initialization(self, backtest_engine):
        """Test backtest engine initialization"""
        assert backtest_engine.config is not None
        assert backtest_engine.portfolio_value == backtest_engine.config.initial_capital
        assert backtest_engine.cash == backtest_engine.config.initial_capital
    
    @pytest.mark.asyncio
    async def test_price_series_generation(self, backtest_engine):
        """Test realistic price series generation"""
        prices = backtest_engine._generate_price_series(252, 'AAPL')  # 1 year
        
        assert len(prices) == 252
        assert all(p > 0 for p in prices)  # All prices positive
        assert prices[0] == 100  # Starting price
        
        # Check for reasonable price movements
        returns = np.diff(np.log(prices))
        assert -0.2 < np.mean(returns) < 0.2  # Reasonable average return
        assert 0.1 < np.std(returns) * np.sqrt(252) < 0.5  # Reasonable volatility
    
    @pytest.mark.asyncio
    async def test_technical_indicators(self, backtest_engine):
        """Test technical indicator calculations"""
        prices = np.array([100, 102, 101, 103, 105, 104, 106, 108])
        
        # Test RSI calculation
        rsi = backtest_engine._calculate_rsi(prices)
        assert len(rsi) == len(prices)
        assert all(0 <= r <= 100 for r in rsi if not np.isnan(r))
        
        # Test MACD calculation
        macd, macd_signal = backtest_engine._calculate_macd(prices)
        assert len(macd) == len(prices)
        assert len(macd_signal) == len(prices)

class TestStrategyOrchestrator:
    """Test cases for the Strategy Orchestrator"""
    
    @pytest.fixture
    def orchestrator_config(self):
        return {
            'symbols': ['AAPL', 'MSFT', 'GOOGL'],
            'strategy': {
                'parameters': {'momentum_threshold': 0.6}
            },
            'risk_management': {
                'max_portfolio_risk': 0.02
            },
            'monitoring': {
                'risk_free_rate': 0.02
            },
            'trading_schedule': {
                'signal_generation_frequency': 300,
                'position_management_frequency': 60
            }
        }
    
    @pytest.fixture
    def orchestrator(self, orchestrator_config):
        return StrategyOrchestrator(orchestrator_config)
    
    def test_orchestrator_initialization(self, orchestrator):
        """Test orchestrator initialization"""
        assert orchestrator is not None
        assert isinstance(orchestrator.strategy, NeuralMomentumStrategy)
        assert isinstance(orchestrator.risk_manager, AdaptiveRiskManager)
        assert isinstance(orchestrator.performance_tracker, PerformanceTracker)
        assert not orchestrator.is_trading
    
    @pytest.mark.asyncio
    async def test_backtest_execution(self, orchestrator):
        """Test backtest execution through orchestrator"""
        backtest_config = {
            'start_date': '2023-01-01',
            'end_date': '2023-12-31',
            'initial_capital': 100000,
            'symbols': ['AAPL', 'MSFT']
        }
        
        results = await orchestrator.run_backtest(backtest_config)
        
        assert results['status'] == 'completed'
        assert 'summary' in results
        assert 'results_file' in results
    
    def test_status_reporting(self, orchestrator):
        """Test system status reporting"""
        status = orchestrator.get_status()
        
        assert isinstance(status, dict)
        assert 'is_trading' in status
        assert 'active_positions' in status
        assert 'trading_symbols' in status
        assert 'last_update' in status

class TestIntegration:
    """Integration tests for the complete system"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_signal_generation(self):
        """Test complete signal generation pipeline"""
        config = {
            'strategy': {
                'parameters': {'momentum_threshold': 0.6, 'neural_confidence_min': 0.7}
            },
            'risk_management': {'max_portfolio_risk': 0.02},
            'monitoring': {'risk_free_rate': 0.02},
            'symbols': ['AAPL']
        }
        
        orchestrator = StrategyOrchestrator(config)
        
        # Mock all external dependencies
        with patch.object(orchestrator.strategy, 'analyze_market_conditions') as mock_regime, \
             patch.object(orchestrator.strategy, '_get_symbol_data') as mock_data, \
             patch.object(orchestrator.strategy.neural_predictor, 'predict') as mock_neural:
            
            from strategies.momentum.neural_momentum_strategy import MarketRegime
            mock_regime.return_value = MarketRegime('medium', 0.6, 'high', 'medium', 'bullish')
            mock_data.return_value = {'price': 150.0, 'volume': 1000000, 'rsi': 55}
            mock_neural.return_value = {'prediction': 0.8, 'confidence': 0.75}
            
            # Generate signals
            signals = await orchestrator.strategy.generate_signals(['AAPL'])
            
            # Verify signals were generated (may be empty due to mocking, but should not error)
            assert isinstance(signals, list)
    
    @pytest.mark.asyncio 
    async def test_risk_integration(self):
        """Test risk management integration"""
        config = {
            'strategy': {'parameters': {'max_position_size': 0.05}},
            'risk_management': {'max_portfolio_risk': 0.02, 'max_position_size': 0.05},
            'monitoring': {'risk_free_rate': 0.02}
        }
        
        orchestrator = StrategyOrchestrator(config)
        
        # Test risk limit checking
        test_position = {
            'symbol': 'AAPL',
            'position_size': 0.03,
            'direction': 'long',
            'value': 4500
        }
        
        risk_checks = await orchestrator.risk_manager.check_risk_limits(test_position)
        
        assert isinstance(risk_checks, dict)
        assert all(isinstance(v, bool) for v in risk_checks.values())

# Performance benchmarks
class TestPerformance:
    """Performance and benchmark tests"""
    
    @pytest.mark.asyncio
    async def test_signal_generation_performance(self):
        """Test signal generation speed"""
        import time
        
        config = {
            'strategy': {'parameters': {'momentum_threshold': 0.6}},
            'risk_management': {'max_portfolio_risk': 0.02},
            'monitoring': {'risk_free_rate': 0.02}
        }
        
        strategy = NeuralMomentumStrategy(config['strategy'])
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'] * 10  # 50 symbols
        
        start_time = time.time()
        
        # Mock the heavy operations
        with patch.object(strategy, 'analyze_market_conditions') as mock_regime, \
             patch.object(strategy, '_generate_signal_for_symbol') as mock_signal:
            
            from strategies.momentum.neural_momentum_strategy import MarketRegime
            mock_regime.return_value = MarketRegime('medium', 0.6, 'high', 'medium', 'bullish')
            mock_signal.return_value = None
            
            signals = await strategy.generate_signals(symbols)
            
        end_time = time.time()
        
        # Should complete within reasonable time
        assert end_time - start_time < 5.0  # 5 seconds max
        assert isinstance(signals, list)

if __name__ == '__main__':
    pytest.main(['-v', __file__])