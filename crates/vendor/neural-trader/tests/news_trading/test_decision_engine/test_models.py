"""Tests for Trading Decision Engine models following TDD."""

import pytest
from datetime import datetime
from src.news_trading.decision_engine.models import (
    TradingSignal, SignalType, RiskLevel, TradingStrategy, AssetType
)


class TestTradingSignalModel:
    """Test TradingSignal data model."""
    
    def test_trading_signal_creation(self):
        """Test creating a TradingSignal with all required fields."""
        signal = TradingSignal(
            id="signal-123",
            timestamp=datetime.now(),
            asset="AAPL",
            asset_type=AssetType.EQUITY,
            signal_type=SignalType.BUY,
            strategy=TradingStrategy.SWING,
            strength=0.85,
            confidence=0.75,
            risk_level=RiskLevel.MEDIUM,
            position_size=0.05,
            entry_price=175.50,
            stop_loss=171.00,
            take_profit=182.00,
            holding_period="3-7 days",
            source_events=["news-001", "news-002"],
            reasoning="Technical breakout above 200-day MA with strong volume"
        )
        
        assert signal.id == "signal-123"
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
        assert "Technical breakout" in signal.reasoning
        
    def test_signal_type_enum(self):
        """Test SignalType enum values."""
        assert SignalType.BUY.value == "BUY"
        assert SignalType.SELL.value == "SELL"
        assert SignalType.HOLD.value == "HOLD"
        assert SignalType.CLOSE.value == "CLOSE"
        assert SignalType.SHORT.value == "SHORT"
        assert SignalType.COVER.value == "COVER"
        
    def test_risk_level_enum(self):
        """Test RiskLevel enum values."""
        assert RiskLevel.LOW.value == "LOW"
        assert RiskLevel.MEDIUM.value == "MEDIUM"
        assert RiskLevel.HIGH.value == "HIGH"
        assert RiskLevel.EXTREME.value == "EXTREME"
        
    def test_trading_strategy_enum(self):
        """Test TradingStrategy enum values."""
        assert TradingStrategy.SWING.value == "SWING"
        assert TradingStrategy.MOMENTUM.value == "MOMENTUM"
        assert TradingStrategy.MIRROR.value == "MIRROR"
        assert TradingStrategy.DAY_TRADE.value == "DAY_TRADE"
        assert TradingStrategy.POSITION.value == "POSITION"
        
    def test_asset_type_enum(self):
        """Test AssetType enum values."""
        assert AssetType.EQUITY.value == "EQUITY"
        assert AssetType.BOND.value == "BOND"
        assert AssetType.CRYPTO.value == "CRYPTO"
        assert AssetType.COMMODITY.value == "COMMODITY"
        assert AssetType.FOREX.value == "FOREX"
        
    def test_signal_with_optional_fields(self):
        """Test TradingSignal with optional fields."""
        signal = TradingSignal(
            id="signal-456",
            timestamp=datetime.now(),
            asset="BTC",
            asset_type=AssetType.CRYPTO,
            signal_type=SignalType.BUY,
            strategy=TradingStrategy.MOMENTUM,
            strength=0.90,
            confidence=0.85,
            risk_level=RiskLevel.HIGH,
            position_size=0.03,
            entry_price=45000,
            stop_loss=43000,
            take_profit=50000,
            holding_period="1-4 weeks",
            source_events=["news-003"],
            reasoning="Strong momentum breakout",
            technical_indicators={
                "rsi": 65,
                "macd": "bullish",
                "volume": "high"
            },
            momentum_score=0.88,
            metadata={"news_sentiment": 0.75}
        )
        
        assert signal.technical_indicators["rsi"] == 65
        assert signal.momentum_score == 0.88
        assert signal.metadata["news_sentiment"] == 0.75
        
    def test_bond_signal(self):
        """Test creating a bond trading signal."""
        signal = TradingSignal(
            id="bond-signal-001",
            timestamp=datetime.now(),
            asset="US10Y",
            asset_type=AssetType.BOND,
            signal_type=SignalType.SHORT,
            strategy=TradingStrategy.SWING,
            strength=0.70,
            confidence=0.80,
            risk_level=RiskLevel.MEDIUM,
            position_size=0.10,
            entry_price=4.25,  # Yield
            stop_loss=4.40,
            take_profit=4.00,
            holding_period="1-3 months",
            source_events=["fed-announcement"],
            reasoning="Rising yields expected on hawkish Fed"
        )
        
        assert signal.asset_type == AssetType.BOND
        assert signal.signal_type == SignalType.SHORT
        assert signal.holding_period == "1-3 months"
        
    def test_mirror_trading_signal(self):
        """Test creating a mirror trading signal."""
        signal = TradingSignal(
            id="mirror-001",
            timestamp=datetime.now(),
            asset="BAC",
            asset_type=AssetType.EQUITY,
            signal_type=SignalType.BUY,
            strategy=TradingStrategy.MIRROR,
            strength=0.85,
            confidence=0.90,
            risk_level=RiskLevel.LOW,
            position_size=0.02,
            entry_price=32.50,
            stop_loss=30.00,
            take_profit=40.00,
            holding_period="6-12 months",
            source_events=["13F-berkshire"],
            reasoning="Following Berkshire Hathaway position",
            mirror_source="Berkshire Hathaway"
        )
        
        assert signal.strategy == TradingStrategy.MIRROR
        assert signal.mirror_source == "Berkshire Hathaway"
        assert signal.confidence == 0.90  # High confidence for Buffett