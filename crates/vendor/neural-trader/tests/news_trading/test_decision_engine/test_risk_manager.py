"""Tests for Risk Manager following TDD."""

import pytest
from src.news_trading.decision_engine.risk_manager import RiskManager
from src.news_trading.decision_engine.models import (
    TradingSignal, SignalType, RiskLevel, TradingStrategy, AssetType
)
from datetime import datetime
import uuid


class TestRiskManager:
    """Test suite for Risk Manager."""
    
    def test_risk_manager_initialization(self):
        """Test RiskManager initialization with parameters."""
        risk_manager = RiskManager(
            max_position_size=0.1,
            max_portfolio_risk=0.2,
            max_correlation=0.7
        )
        
        assert risk_manager.max_position_size == 0.1
        assert risk_manager.max_portfolio_risk == 0.2
        assert risk_manager.max_correlation == 0.7
        
    @pytest.mark.asyncio
    async def test_position_size_validation(self):
        """Test position size validation against risk limits."""
        risk_manager = RiskManager(max_position_size=0.1)
        
        # Signal exceeding max position size
        signal = TradingSignal(
            id="test-001",
            timestamp=datetime.now(),
            asset="AAPL",
            asset_type=AssetType.EQUITY,
            signal_type=SignalType.BUY,
            strategy=TradingStrategy.SWING,
            strength=0.8,
            confidence=0.7,
            risk_level=RiskLevel.HIGH,
            position_size=0.15,  # Exceeds max
            entry_price=175.00,
            stop_loss=170.00,
            take_profit=180.00,
            holding_period="3-7 days",
            source_events=["test"],
            reasoning="Test signal"
        )
        
        validated_signal = await risk_manager.validate_signal(signal)
        assert validated_signal.position_size <= 0.1
        
    @pytest.mark.asyncio
    async def test_portfolio_risk_adjustment(self):
        """Test signal adjustment based on portfolio risk."""
        risk_manager = RiskManager(max_portfolio_risk=0.2)
        
        # Existing positions with high risk
        current_positions = {
            "BTC": {"size": 0.3, "risk": 0.30, "asset_type": "crypto"},  # 9% risk  
            "ETH": {"size": 0.2, "risk": 0.30, "asset_type": "crypto"},  # 6% risk
            "TSLA": {"size": 0.1, "risk": 0.30, "asset_type": "equity"}  # 3% risk
        }  # Total existing risk: 18%
        
        # New signal that would exceed portfolio risk
        signal = TradingSignal(
            id="test-002",
            timestamp=datetime.now(),
            asset="SOL",
            asset_type=AssetType.CRYPTO,
            signal_type=SignalType.BUY,
            strategy=TradingStrategy.MOMENTUM,
            strength=0.9,
            confidence=0.8,
            risk_level=RiskLevel.HIGH,
            position_size=0.15,  # 15% position
            entry_price=100.00,
            stop_loss=85.00,  # 15% stop loss
            take_profit=120.00,
            holding_period="1-4 weeks",
            source_events=["test"],
            reasoning="Test signal"
        )
        
        validated_signal = await risk_manager.validate_signal(signal, current_positions)
        
        # Should reduce position size - first to max position limit (10%)
        assert validated_signal.position_size <= 0.1  # Max position size
        # The signal requested 15% but gets capped at 10%
        # With stop loss of 15%, the risk would be 1.5% (10% * 15%)
        # Current portfolio risk is 18%, so total would be 19.5%, under 20% limit
        assert validated_signal.position_size == 0.1
        
    def test_portfolio_risk_calculation(self):
        """Test portfolio risk calculation."""
        risk_manager = RiskManager()
        
        positions = {
            "BTC": {"size": 0.3, "risk": 0.05},
            "ETH": {"size": 0.2, "risk": 0.04},
            "ADA": {"size": 0.1, "risk": 0.03}
        }
        
        total_risk = risk_manager.calculate_portfolio_risk(positions)
        
        # Total risk = 0.3*0.05 + 0.2*0.04 + 0.1*0.03 = 0.015 + 0.008 + 0.003 = 0.026
        assert total_risk == pytest.approx(0.026, 0.001)
        
    def test_correlation_impact(self):
        """Test correlation-based position adjustment."""
        risk_manager = RiskManager(max_correlation=0.7)
        
        # Existing correlated positions
        positions = {
            "BTC": {"size": 0.1, "correlation_matrix": {"ETH": 0.9}},
            "ETH": {"size": 0.1, "correlation_matrix": {"BTC": 0.9}}
        }
        
        # New highly correlated asset
        correlation_data = {"BTC": 0.85, "ETH": 0.88}
        
        adjusted_size = risk_manager.adjust_for_correlation(
            proposed_size=0.1,
            asset="SOL",
            correlation_data=correlation_data,
            current_positions=positions
        )
        
        # Should reduce size due to high correlation
        assert adjusted_size < 0.1
        
    def test_stop_loss_validation(self):
        """Test stop loss validation for different asset types."""
        risk_manager = RiskManager()
        
        # Equity with tight stop
        equity_signal = {
            "asset_type": AssetType.EQUITY,
            "entry_price": 100.00,
            "stop_loss": 99.50,  # 0.5% stop
            "atr": 2.00
        }
        
        validated_stop = risk_manager.validate_stop_loss(equity_signal)
        assert validated_stop <= 98.00  # At least 2% stop for safety
        
        # Crypto with appropriate stop
        crypto_signal = {
            "asset_type": AssetType.CRYPTO,
            "entry_price": 50000.00,
            "stop_loss": 45000.00,  # 10% stop
            "atr": 2000.00
        }
        
        validated_stop = risk_manager.validate_stop_loss(crypto_signal)
        assert validated_stop == 45000.00  # Appropriate for crypto volatility
        
    def test_risk_level_scoring(self):
        """Test risk level scoring for positions."""
        risk_manager = RiskManager()
        
        # High risk signal
        high_risk_score = risk_manager.score_signal_risk(
            risk_level=RiskLevel.HIGH,
            volatility=0.05,
            position_size=0.08
        )
        
        # Low risk signal
        low_risk_score = risk_manager.score_signal_risk(
            risk_level=RiskLevel.LOW,
            volatility=0.01,
            position_size=0.02
        )
        
        assert high_risk_score > low_risk_score
        assert 0 <= high_risk_score <= 1
        assert 0 <= low_risk_score <= 1