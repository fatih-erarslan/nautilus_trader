"""
Tests for Trading Decision Engine Base Interface
Following TDD - Red-Green-Refactor approach
"""
import pytest
from abc import ABC, abstractmethod
from typing import Dict, List
from datetime import datetime


def test_decision_engine_interface():
    """Test that TradingDecisionEngine abstract interface is properly defined"""
    from src.trading.decision_engine.base import TradingDecisionEngine
    
    class TestEngine(TradingDecisionEngine):
        pass
    
    # Should fail - abstract methods not implemented
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        engine = TestEngine()


def test_decision_engine_abstract_methods():
    """Test that all required abstract methods are defined"""
    from src.trading.decision_engine.base import TradingDecisionEngine
    
    # Check that abstract methods exist
    assert hasattr(TradingDecisionEngine, 'process_sentiment')
    assert hasattr(TradingDecisionEngine, 'evaluate_portfolio')
    assert hasattr(TradingDecisionEngine, 'process_market_data')
    assert hasattr(TradingDecisionEngine, 'set_risk_parameters')
    assert hasattr(TradingDecisionEngine, 'get_active_signals')


@pytest.mark.asyncio
async def test_concrete_engine_implementation():
    """Test concrete implementation of TradingDecisionEngine"""
    from src.trading.decision_engine.base import TradingDecisionEngine
    from src.trading.decision_engine.models import TradingSignal, SignalType, RiskLevel, TradingStrategy, AssetType
    
    class ConcreteEngine(TradingDecisionEngine):
        def __init__(self):
            self.active_signals = []
            self.risk_params = {
                "max_position_size": 0.1,
                "max_portfolio_risk": 0.2
            }
        
        async def process_sentiment(self, sentiment_data: Dict) -> TradingSignal:
            # Simple implementation for testing
            return TradingSignal(
                id="test-signal-1",
                timestamp=datetime.now(),
                asset=sentiment_data.get("asset", "BTC"),
                asset_type=AssetType.CRYPTO,
                signal_type=SignalType.BUY if sentiment_data.get("sentiment_score", 0) > 0 else SignalType.SELL,
                strategy=TradingStrategy.SWING,
                strength=abs(sentiment_data.get("sentiment_score", 0)),
                confidence=sentiment_data.get("confidence", 0.5),
                risk_level=RiskLevel.MEDIUM,
                position_size=0.05,
                entry_price=100.0,
                stop_loss=95.0,
                take_profit=110.0,
                holding_period="3-7 days",
                source_events=sentiment_data.get("source_events", []),
                reasoning="Test signal"
            )
        
        async def evaluate_portfolio(self, current_positions: Dict) -> List[TradingSignal]:
            # Simple implementation for testing
            signals = []
            for asset, position in current_positions.items():
                if position.get("unrealized_pnl", 0) > 0.1:  # 10% profit
                    signals.append(TradingSignal(
                        id=f"rebalance-{asset}",
                        timestamp=datetime.now(),
                        asset=asset,
                        asset_type=AssetType.CRYPTO,
                        signal_type=SignalType.SELL,
                        strategy=TradingStrategy.POSITION,
                        strength=0.7,
                        confidence=0.8,
                        risk_level=RiskLevel.LOW,
                        position_size=position.get("size", 0.1) * 0.5,  # Sell half
                        entry_price=position.get("current_price", 100),
                        stop_loss=0,
                        take_profit=0,
                        holding_period="immediate",
                        source_events=["portfolio_rebalance"],
                        reasoning="Taking profits on position"
                    ))
            return signals
        
        async def process_market_data(self, market_data: Dict) -> List[TradingSignal]:
            # Simple implementation for testing
            signals = []
            if market_data.get("rsi", 50) < 30:  # Oversold
                signals.append(TradingSignal(
                    id="technical-signal-1",
                    timestamp=datetime.now(),
                    asset=market_data.get("asset", "BTC"),
                    asset_type=AssetType.CRYPTO,
                    signal_type=SignalType.BUY,
                    strategy=TradingStrategy.SWING,
                    strength=0.8,
                    confidence=0.7,
                    risk_level=RiskLevel.MEDIUM,
                    position_size=0.05,
                    entry_price=market_data.get("price", 100),
                    stop_loss=market_data.get("price", 100) * 0.95,
                    take_profit=market_data.get("price", 100) * 1.1,
                    holding_period="3-7 days",
                    source_events=["technical_analysis"],
                    reasoning="RSI oversold condition"
                ))
            return signals
        
        def set_risk_parameters(self, params: Dict):
            self.risk_params.update(params)
        
        def get_active_signals(self) -> List[TradingSignal]:
            return self.active_signals.copy()
    
    # Test the concrete implementation
    engine = ConcreteEngine()
    
    # Test process_sentiment
    sentiment_data = {
        "asset": "BTC",
        "sentiment_score": 0.75,
        "confidence": 0.85,
        "source_events": ["news-001"]
    }
    signal = await engine.process_sentiment(sentiment_data)
    assert signal.asset == "BTC"
    assert signal.signal_type == SignalType.BUY
    assert signal.strength == 0.75
    
    # Test evaluate_portfolio
    positions = {
        "ETH": {"size": 0.1, "unrealized_pnl": 0.15, "current_price": 3000}
    }
    rebalance_signals = await engine.evaluate_portfolio(positions)
    assert len(rebalance_signals) == 1
    assert rebalance_signals[0].signal_type == SignalType.SELL
    
    # Test process_market_data
    market_data = {
        "asset": "SOL",
        "price": 50,
        "rsi": 25
    }
    technical_signals = await engine.process_market_data(market_data)
    assert len(technical_signals) == 1
    assert technical_signals[0].signal_type == SignalType.BUY
    
    # Test set_risk_parameters
    engine.set_risk_parameters({"max_position_size": 0.05})
    assert engine.risk_params["max_position_size"] == 0.05
    
    # Test get_active_signals
    active = engine.get_active_signals()
    assert isinstance(active, list)


def test_signal_filter_interface():
    """Test SignalFilter abstract interface"""
    from src.trading.decision_engine.base import SignalFilter
    
    class TestFilter(SignalFilter):
        pass
    
    # Should fail - abstract method not implemented
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        filter = TestFilter()


def test_concrete_signal_filter():
    """Test concrete implementation of SignalFilter"""
    from src.trading.decision_engine.base import SignalFilter
    from src.trading.decision_engine.models import TradingSignal, RiskLevel
    
    class RiskLevelFilter(SignalFilter):
        def __init__(self, max_risk: RiskLevel):
            self.max_risk = max_risk
        
        def filter(self, signals: List[TradingSignal]) -> List[TradingSignal]:
            risk_order = {
                RiskLevel.LOW: 1,
                RiskLevel.MEDIUM: 2,
                RiskLevel.HIGH: 3,
                RiskLevel.EXTREME: 4
            }
            max_risk_value = risk_order[self.max_risk]
            
            return [
                signal for signal in signals 
                if risk_order[signal.risk_level] <= max_risk_value
            ]
    
    # Create test signals
    from datetime import datetime
    from src.trading.decision_engine.models import SignalType, TradingStrategy, AssetType
    
    signals = [
        TradingSignal(
            id="sig-1",
            timestamp=datetime.now(),
            asset="BTC",
            asset_type=AssetType.CRYPTO,
            signal_type=SignalType.BUY,
            strategy=TradingStrategy.SWING,
            strength=0.8,
            confidence=0.7,
            risk_level=RiskLevel.LOW,
            position_size=0.05,
            entry_price=100,
            stop_loss=95,
            take_profit=110,
            holding_period="3 days",
            source_events=["test"],
            reasoning="Test"
        ),
        TradingSignal(
            id="sig-2",
            timestamp=datetime.now(),
            asset="ETH",
            asset_type=AssetType.CRYPTO,
            signal_type=SignalType.BUY,
            strategy=TradingStrategy.MOMENTUM,
            strength=0.9,
            confidence=0.8,
            risk_level=RiskLevel.HIGH,
            position_size=0.1,
            entry_price=100,
            stop_loss=90,
            take_profit=120,
            holding_period="1 day",
            source_events=["test"],
            reasoning="Test"
        )
    ]
    
    # Test filter
    filter = RiskLevelFilter(max_risk=RiskLevel.MEDIUM)
    filtered_signals = filter.filter(signals)
    
    assert len(filtered_signals) == 1
    assert filtered_signals[0].id == "sig-1"