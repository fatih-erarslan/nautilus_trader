"""
Test Script for Neural Trading Workflow
Demonstrates integration between Flow Nexus and Neural Trader
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import asyncio
import pytest
from unittest.mock import Mock, patch
from neural_trading_workflow import NeuralTradingWorkflow, TradingSignal

class TestNeuralTradingWorkflow:
    """Test cases for the neural trading workflow"""
    
    def setup_method(self):
        """Set up test configuration"""
        self.config = {
            'symbols': ['AAPL', 'GOOGL', 'MSFT'],
            'max_position_size': 1000,
            'risk_tolerance': 'moderate',
            'trading_mode': 'paper'
        }
        self.workflow = NeuralTradingWorkflow(self.config)
    
    @pytest.mark.asyncio
    async def test_workflow_initialization(self):
        """Test workflow initialization"""
        result = await self.workflow.initialize_workflow()
        assert result == True
        assert self.workflow.workflow_id is not None
        assert 'neural-trader-' in self.workflow.workflow_id
    
    @pytest.mark.asyncio
    async def test_market_data_collection(self):
        """Test market data collection"""
        symbols = ['AAPL', 'GOOGL']
        market_data = await self.workflow.collect_market_data(symbols)
        
        assert len(market_data) == len(symbols)
        for symbol in symbols:
            assert symbol in market_data
            assert 'price' in market_data[symbol]
            assert 'volume' in market_data[symbol]
            assert 'timestamp' in market_data[symbol]
    
    @pytest.mark.asyncio
    async def test_sentiment_analysis(self):
        """Test neural sentiment analysis"""
        symbols = ['AAPL', 'GOOGL']
        sentiment_scores = await self.workflow.analyze_sentiment(symbols)
        
        assert len(sentiment_scores) == len(symbols)
        for symbol in symbols:
            assert symbol in sentiment_scores
            assert -1.0 <= sentiment_scores[symbol] <= 1.0
    
    @pytest.mark.asyncio
    async def test_neural_predictions(self):
        """Test neural network price predictions"""
        symbols = ['AAPL', 'GOOGL']
        predictions = await self.workflow.neural_price_prediction(symbols)
        
        assert len(predictions) == len(symbols)
        for symbol in symbols:
            pred_data = predictions[symbol]
            assert 'lstm_prediction' in pred_data
            assert 'transformer_prediction' in pred_data
            assert 'gru_prediction' in pred_data
            assert 'ensemble_prediction' in pred_data
            assert 'confidence' in pred_data
            assert 0.0 <= pred_data['confidence'] <= 1.0
    
    @pytest.mark.asyncio
    async def test_risk_assessment(self):
        """Test portfolio risk assessment"""
        symbols = ['AAPL', 'GOOGL']
        
        # Mock predictions data
        predictions = {
            'AAPL': {'ensemble_prediction': 0.05, 'confidence': 0.8},
            'GOOGL': {'ensemble_prediction': -0.03, 'confidence': 0.6}
        }
        
        risk_assessments = await self.workflow.assess_risk(symbols, predictions)
        
        assert len(risk_assessments) == len(symbols)
        for symbol in symbols:
            risk_data = risk_assessments[symbol]
            assert 'overall_risk' in risk_data
            assert 'recommended_position_pct' in risk_data
            assert 'max_position_size' in risk_data
            assert 0.0 <= risk_data['overall_risk'] <= 1.0
    
    @pytest.mark.asyncio
    async def test_signal_generation(self):
        """Test trading signal generation"""
        # Mock input data
        market_data = {
            'AAPL': {'price': 150.0, 'volume': 1000000, 'change': 0.02}
        }
        sentiment_scores = {'AAPL': 0.3}
        technical_signals = {
            'AAPL': {'rsi': 25, 'macd': 0.5, 'bollinger': 1, 'trend_strength': 0.2}
        }
        predictions = {
            'AAPL': {'ensemble_prediction': 0.05, 'confidence': 0.8}
        }
        risk_assessments = {
            'AAPL': {'overall_risk': 0.3, 'max_position_size': 500}
        }
        
        signals = await self.workflow.generate_trading_signals(
            market_data, sentiment_scores, technical_signals, 
            predictions, risk_assessments
        )
        
        assert len(signals) > 0
        signal = signals[0]
        assert isinstance(signal, TradingSignal)
        assert signal.symbol == 'AAPL'
        assert signal.signal_type in ['BUY', 'SELL', 'HOLD']
        assert 0.0 <= signal.confidence <= 1.0
    
    @pytest.mark.asyncio
    async def test_portfolio_optimization(self):
        """Test portfolio optimization"""
        # Mock signals
        signals = [
            TradingSignal(
                symbol='AAPL',
                signal_type='BUY',
                confidence=0.8,
                price=150.0,
                timestamp=None,
                reasoning='Test signal',
                neural_score=0.05,
                sentiment_score=0.3,
                technical_score=0.2,
                risk_score=0.3
            ),
            TradingSignal(
                symbol='GOOGL',
                signal_type='BUY', 
                confidence=0.6,
                price=2500.0,
                timestamp=None,
                reasoning='Test signal',
                neural_score=0.03,
                sentiment_score=0.1,
                technical_score=0.4,
                risk_score=0.4
            )
        ]
        
        optimization_result = await self.workflow.optimize_portfolio(signals)
        
        assert optimization_result['action'] in ['rebalance', 'hold']
        if optimization_result['action'] == 'rebalance':
            assert 'allocations' in optimization_result
            assert 'total_value' in optimization_result
            assert 'cash_deployed' in optimization_result
    
    @pytest.mark.asyncio
    async def test_complete_workflow(self):
        """Test the complete workflow execution"""
        result = await self.workflow.execute_workflow()
        
        assert result.workflow_id is not None
        assert result.status in ['completed', 'failed']
        assert result.execution_time > 0
        assert isinstance(result.performance_metrics, dict)
        
        if result.status == 'completed':
            assert len(result.signals) >= 0
            assert 'total_signals' in result.performance_metrics
            assert 'execution_time_seconds' in result.performance_metrics

# Manual test function for live testing
async def manual_test_workflow():
    """Manual test function for interactive testing"""
    print("\n🧪 Manual Neural Trading Workflow Test")
    print("=" * 50)
    
    config = {
        'symbols': ['AAPL', 'MSFT', 'TSLA'],
        'max_position_size': 1500,
        'risk_tolerance': 'moderate',
        'trading_mode': 'paper'
    }
    
    workflow = NeuralTradingWorkflow(config)
    
    print("⏳ Executing workflow...")
    result = await workflow.execute_workflow()
    
    print(f"\n📊 Test Results:")
    print(f"Status: {result.status}")
    print(f"Execution Time: {result.execution_time:.2f}s")
    print(f"Signals Generated: {len(result.signals)}")
    
    if result.performance_metrics:
        print(f"\n📈 Performance Metrics:")
        for key, value in result.performance_metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")
    
    if result.signals:
        print(f"\n🎯 Sample Signals:")
        for signal in result.signals[:3]:  # Show first 3
            print(f"  {signal.symbol}: {signal.signal_type} @ ${signal.price:.2f}")
            print(f"    Confidence: {signal.confidence:.2f}")
            print(f"    Neural Score: {signal.neural_score:.3f}")
    
    print(f"\n✅ Test completed successfully!")
    return result

if __name__ == "__main__":
    # Run manual test
    asyncio.run(manual_test_workflow())