"""
Tests for Trading Decision Engine Models
Following TDD - Red-Green-Refactor approach
"""
import pytest
from datetime import datetime
from dataclasses import fields


def test_signal_type_enum():
    """Test that SignalType enum is properly defined"""
    from src.trading.decision_engine.models import SignalType
    
    # Test all required signal types exist
    assert SignalType.BUY.value == "BUY"
    assert SignalType.SELL.value == "SELL"
    assert SignalType.HOLD.value == "HOLD"
    assert SignalType.CLOSE.value == "CLOSE"
    assert SignalType.SHORT.value == "SHORT"
    assert SignalType.COVER.value == "COVER"


def test_risk_level_enum():
    """Test that RiskLevel enum is properly defined"""
    from src.trading.decision_engine.models import RiskLevel
    
    assert RiskLevel.LOW.value == "LOW"
    assert RiskLevel.MEDIUM.value == "MEDIUM"
    assert RiskLevel.HIGH.value == "HIGH"
    assert RiskLevel.EXTREME.value == "EXTREME"


def test_trading_strategy_enum():
    """Test that TradingStrategy enum is properly defined"""
    from src.trading.decision_engine.models import TradingStrategy
    
    assert TradingStrategy.SWING.value == "SWING"
    assert TradingStrategy.MOMENTUM.value == "MOMENTUM"
    assert TradingStrategy.MIRROR.value == "MIRROR"
    assert TradingStrategy.DAY_TRADE.value == "DAY_TRADE"
    assert TradingStrategy.POSITION.value == "POSITION"


def test_asset_type_enum():
    """Test that AssetType enum is properly defined"""
    from src.trading.decision_engine.models import AssetType
    
    assert AssetType.EQUITY.value == "EQUITY"
    assert AssetType.BOND.value == "BOND"
    assert AssetType.CRYPTO.value == "CRYPTO"
    assert AssetType.COMMODITY.value == "COMMODITY"
    assert AssetType.FOREX.value == "FOREX"


def test_trading_signal_model():
    """Test TradingSignal data model"""
    from src.trading.decision_engine.models import (
        TradingSignal, SignalType, RiskLevel, 
        TradingStrategy, AssetType
    )
    
    # Create a complete trading signal
    signal = TradingSignal(
        id="signal-123",
        timestamp=datetime.now(),
        asset="AAPL",
        asset_type=AssetType.EQUITY,
        signal_type=SignalType.BUY,
        strategy=TradingStrategy.SWING,
        strength=0.85,  # 0 to 1
        confidence=0.75,
        risk_level=RiskLevel.MEDIUM,
        position_size=0.05,  # 5% of portfolio
        entry_price=175.50,
        stop_loss=171.00,  # 2.5% stop for swing trade
        take_profit=182.00,  # 3.7% target
        holding_period="3-7 days",
        source_events=["news-001", "news-002"],
        reasoning="Technical breakout above 200-day MA with strong volume"
    )
    
    # Test all required fields
    assert signal.id == "signal-123"
    assert isinstance(signal.timestamp, datetime)
    assert signal.asset == "AAPL"
    assert signal.asset_type == AssetType.EQUITY
    assert signal.signal_type == SignalType.BUY
    assert signal.strategy == TradingStrategy.SWING
    assert signal.strength == 0.85
    assert signal.confidence == 0.75
    assert signal.risk_level == RiskLevel.MEDIUM
    assert signal.position_size == 0.05
    assert signal.entry_price == 175.50
    assert signal.stop_loss == 171.00
    assert signal.take_profit == 182.00
    assert signal.holding_period == "3-7 days"
    assert len(signal.source_events) == 2
    assert signal.reasoning == "Technical breakout above 200-day MA with strong volume"


def test_trading_signal_optional_fields():
    """Test TradingSignal optional fields"""
    from src.trading.decision_engine.models import (
        TradingSignal, SignalType, RiskLevel, 
        TradingStrategy, AssetType
    )
    
    # Create signal with minimal required fields
    signal = TradingSignal(
        id="signal-456",
        timestamp=datetime.now(),
        asset="BTC",
        asset_type=AssetType.CRYPTO,
        signal_type=SignalType.BUY,
        strategy=TradingStrategy.MOMENTUM,
        strength=0.9,
        confidence=0.8,
        risk_level=RiskLevel.HIGH,
        position_size=0.1,
        entry_price=50000,
        stop_loss=48000,
        take_profit=55000,
        holding_period="1-3 days",
        source_events=["news-003"],
        reasoning="Strong momentum on positive regulatory news"
    )
    
    # Test optional fields have default values
    assert signal.technical_indicators is None
    assert signal.mirror_source is None
    assert signal.momentum_score is None
    assert signal.metadata is None


def test_trading_signal_with_optional_fields():
    """Test TradingSignal with all optional fields populated"""
    from src.trading.decision_engine.models import (
        TradingSignal, SignalType, RiskLevel, 
        TradingStrategy, AssetType
    )
    
    signal = TradingSignal(
        id="signal-789",
        timestamp=datetime.now(),
        asset="ETH",
        asset_type=AssetType.CRYPTO,
        signal_type=SignalType.SELL,
        strategy=TradingStrategy.MIRROR,
        strength=0.7,
        confidence=0.85,
        risk_level=RiskLevel.MEDIUM,
        position_size=0.03,
        entry_price=3500,
        stop_loss=3650,
        take_profit=3200,
        holding_period="5-10 days",
        source_events=["filing-001"],
        reasoning="Following institutional selling pattern",
        technical_indicators={
            "rsi": 75,
            "macd": -50,
            "volume_ratio": 0.8
        },
        mirror_source="Bridgewater Associates",
        momentum_score=0.65,
        metadata={
            "sector": "crypto",
            "market_cap": "large"
        }
    )
    
    # Test optional fields
    assert signal.technical_indicators["rsi"] == 75
    assert signal.mirror_source == "Bridgewater Associates"
    assert signal.momentum_score == 0.65
    assert signal.metadata["sector"] == "crypto"


def test_trading_signal_validation():
    """Test TradingSignal field validation"""
    from src.trading.decision_engine.models import (
        TradingSignal, SignalType, RiskLevel, 
        TradingStrategy, AssetType
    )
    
    # Test strength validation (should be 0-1)
    with pytest.raises(ValueError, match="strength must be between 0 and 1"):
        TradingSignal(
            id="invalid-1",
            timestamp=datetime.now(),
            asset="AAPL",
            asset_type=AssetType.EQUITY,
            signal_type=SignalType.BUY,
            strategy=TradingStrategy.SWING,
            strength=1.5,  # Invalid: > 1
            confidence=0.8,
            risk_level=RiskLevel.MEDIUM,
            position_size=0.05,
            entry_price=175.50,
            stop_loss=171.00,
            take_profit=182.00,
            holding_period="3-7 days",
            source_events=["news-001"],
            reasoning="Test"
        )
    
    # Test confidence validation (should be 0-1)
    with pytest.raises(ValueError, match="confidence must be between 0 and 1"):
        TradingSignal(
            id="invalid-2",
            timestamp=datetime.now(),
            asset="AAPL",
            asset_type=AssetType.EQUITY,
            signal_type=SignalType.BUY,
            strategy=TradingStrategy.SWING,
            strength=0.8,
            confidence=-0.1,  # Invalid: < 0
            risk_level=RiskLevel.MEDIUM,
            position_size=0.05,
            entry_price=175.50,
            stop_loss=171.00,
            take_profit=182.00,
            holding_period="3-7 days",
            source_events=["news-001"],
            reasoning="Test"
        )
    
    # Test position_size validation (should be 0-1)
    with pytest.raises(ValueError, match="position_size must be between 0 and 1"):
        TradingSignal(
            id="invalid-3",
            timestamp=datetime.now(),
            asset="AAPL",
            asset_type=AssetType.EQUITY,
            signal_type=SignalType.BUY,
            strategy=TradingStrategy.SWING,
            strength=0.8,
            confidence=0.8,
            risk_level=RiskLevel.MEDIUM,
            position_size=2.0,  # Invalid: > 1
            entry_price=175.50,
            stop_loss=171.00,
            take_profit=182.00,
            holding_period="3-7 days",
            source_events=["news-001"],
            reasoning="Test"
        )


def test_portfolio_position_model():
    """Test PortfolioPosition model for tracking current positions"""
    from src.trading.decision_engine.models import PortfolioPosition, AssetType, TradingStrategy
    
    position = PortfolioPosition(
        asset="BTC",
        asset_type=AssetType.CRYPTO,
        size=0.15,  # 15% of portfolio
        entry_price=45000,
        current_price=48000,
        unrealized_pnl=0.0667,  # 6.67% gain
        stop_loss=43000,
        take_profit=50000,
        entry_time=datetime.now(),
        strategy=TradingStrategy.SWING
    )
    
    assert position.asset == "BTC"
    assert position.size == 0.15
    assert position.unrealized_pnl == pytest.approx(0.0667, 0.0001)
    assert position.strategy == TradingStrategy.SWING